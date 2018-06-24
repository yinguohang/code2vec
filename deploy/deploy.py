import argparse
import os
import signal
import subprocess
import threading
import time

from config import deploy_config
from utils import *

try:
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urlparse import urlparse, parse_qs

parser = argparse.ArgumentParser(description="""
A script that helps you to automatically upload PAI tasks.

* Put the entire deploy folder WITHIN your project directory. The script can be used as
  > python deploy.py ABS_PATH_TO_ODPSCMD [--suppress_stderr]

* Your project should be managed under git, and only files tracked by git will be uploaded to PAI.
  To see which files will be uploaded, run the following command
  > git ls-tree --full-tree -r --name-only HEAD

* Currently, the entry file is configured to be <YOUR_PROJECT_ROOT>/main.py. If your entry file is
  not this file, please open config.py and change variable 'main_entry_file'.

* To add hyperparameter or any other pai command arguments, please open deploy.py and change variable
  'pai_command'.

* You may use optional flag --suppress_stderr to hide messages from stderr.

""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('odps_path', help='Path to odps console')
parser.add_argument('--suppress_stderr', action='store_true')
args = parser.parse_args()

# Path to odps console
odps_path = args.odps_path

# boolean: if show stderr message
suppress_stderr = args.suppress_stderr

###################################
#  Creating code.tar.gz
###################################
write_stdout("Preparing tar ball...\n")

# Path to store code.tar.gz
tar_ball_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code.tar.gz")
tar_ball_location_pai = tar_ball_location

if detect_platform() == "WINDOWS":
    tar_ball_location_pai = tar_ball_location.replace("\\", "\\\\")
    tar_ball_location = "/" + tar_ball_location.replace("\\", "/").replace(":", "")

# Start tar process
tar_process = subprocess.Popen(["bash", "-c",
                                "cd `git rev-parse --show-toplevel`;"
                                + "git ls-tree --full-tree -r --name-only HEAD | "
                                + "tar -czvf \"" + tar_ball_location + "\" -T -"],
                               stdout=subprocess.PIPE)
threading.Thread(target=forward_fd, args=(tar_process.stdout, sys.stdout, None,
                                          lambda: tar_process.poll() is None)).start()
tar_process.wait()

write_stdout("The tar ball is stored at " + tar_ball_location + "\n\n")


###################################
#  Launch ODPS + Create PAI Task
###################################
write_stdout("Now launching odps...\n")

# Job url returned from odps
instance_id, job_url = None, None

# Pai commands
pai_command = "pai -name {}".format(deploy_config['pai_algo']) \
              + " -Dscript='file://{}'".format(tar_ball_location_pai) \
              + " -DentryFile='{}'".format(deploy_config['pai_main_entry_file']) \
              + " -DcheckpointDir='{}'".format(deploy_config['oss_role_arn']) \
              + " -Dbuckets='{}';".format(deploy_config['oss_role_arn'])

# Start odps process
odps_process = subprocess.Popen([odps_path, '--project', deploy_config['odps_project'], '-e', pai_command],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)


def odps_output_handler(buf):
    global job_url, instance_id
    out = "".join(buf)
    if out.endswith('\n'):
        if out.startswith("ID = "):
            instance_id = out[5:].strip()
        if out.startswith("http://logview.odps.aliyun-inc.com:8080"):
            job_url = out
            odps_process.terminate()


threading.Thread(target=forward_fd, args=(odps_process.stderr, sys.stdout, odps_output_handler,
                                          lambda: odps_process.poll() is not None)).start()
threading.Thread(target=forward_fd, args=(odps_process.stdout, sys.stdout, odps_output_handler,
                                          lambda: odps_process.poll() is not None)).start()


###################################
#  Create Job Abort Handler
###################################
def abort_signal_handler(signal, handler):
    global instance_id
    write_stderr("Now killing PAI instance...\n")
    try:
        odps_process.terminate()
    except OSError:
        pass

    odps_terminate_process = subprocess.Popen([odps_path, '--project', deploy_config['odps_project'],
                                               '-e', 'kill ' + instance_id],
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)

    def odps_terminate_output_handler(buf):
        out = "".join(buf)
        if out.endswith('\n'):
            if out.startswith("OK"):
                odps_terminate_process.terminate()

    threading.Thread(target=forward_fd,
                     args=(odps_terminate_process.stdout, sys.stdout, odps_terminate_output_handler,
                           lambda: odps_terminate_process.poll() is not None)).start()
    threading.Thread(target=forward_fd,
                     args=(odps_terminate_process.stderr, sys.stderr, odps_terminate_output_handler,
                           lambda: odps_terminate_process.poll() is not None)).start()

    odps_terminate_process.wait()
    sys.exit()


signal.signal(signal.SIGINT, abort_signal_handler)

odps_process.wait()

os.remove(tar_ball_location_pai)

write_stdout("\nPai job launched. Retrieving job information...\n\n")


###################################
#  Retrieve Job ID and Status
###################################
query = parse_qs(urlparse(job_url).query)
job_id = query['i'][0].strip()
token = query['token'][0].strip()

write_stdout("Job ID: {}\nToken: {}\n\n".format(job_id, token))

write_stdout("Now waiting for the task to start...\n")

status = ""
wait_pos, task_name = None, None
while True:
    if wait_pos is not None and task_name is not None:
        detail = send_odps_request(job_id, token, "detail&taskname=" + task_name)
        try:
            log_id = detail['mapReduce']['jobs'][0]['tasks'][0]['instances'][0]['logId']
            break
        except (KeyError, IndexError) as e:
            pass

    cached = retrieve_odps_status(job_id, token)
    if 'subStatusHistory' in cached:
        status, new_status = combine_overlap_string(status, format_odps_status_history(cached))
        sys.stdout.write(new_status)
        sys.stdout.flush()
    if 'taskName' in cached and cached['taskName'] != task_name:
        task_name = cached['taskName']
        print("Task Name: {}".format(task_name))
    if 'waitPos' in cached and cached['waitPos'] != wait_pos:
        wait_pos = cached['waitPos']
        print("Current Waitlist Position: {}".format(wait_pos))

    time.sleep(1)
print("")

#######################################
#  Connect to Remote Stdout & Stderr
#######################################
print("Task is now running. Connecting to remote console...\n")

stdout = ""
stderr = ""
status = "Running"
while status == "Running":
    status = retrieve_odps_status(job_id, token)['status']
    stdout, new_stdout = combine_overlap_string(stdout, retrieve_odps_log(job_id, token, log_id, log_type="Stdout"))
    sys.stdout.write(new_stdout)
    sys.stdout.flush()
    if not suppress_stderr:
        stderr, new_stderr = combine_overlap_string(stderr, retrieve_odps_log(job_id, token, log_id, log_type="Stderr"))
        sys.stderr.write('\x1b[1;31m' + new_stderr + '\x1b[0m')
        sys.stderr.flush()
    time.sleep(1)

print("\nFinished")
