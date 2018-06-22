import argparse
import os
import signal
import subprocess
import sys
import threading
import time

from utils import *

try:
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urlparse import urlparse, parse_qs

# Path to store code.tar.gz
tar_ball_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code.tar.gz")
tar_ball_location_pai = tar_ball_location

if detect_platform() == "WINDOWS":
    tar_ball_location_pai = tar_ball_location.replace("\\", "\\\\")
    tar_ball_location = "/" + tar_ball_location.replace("\\", "/").replace(":", "")

# Main entry file of the code
main_entry_file = "main.py"

# Pai commands
pai_commands = [
    "use kelude_open_dw;",
    "pai -name tensorflow140"
    + " -Dscript='file://{}'".format(tar_ball_location_pai)
    + " -DentryFile='{}'".format(main_entry_file)
    + " -DcheckpointDir='oss://apsalgo-hz/?role_arn=acs:ram::1396633922344963:role/aps-odps-algo&host=cn-hangzhou.oss-internal.aliyun-inc.com'"
    + " -Dbuckets='oss://apsalgo-hz/?role_arn=acs:ram::1396633922344963:role/aps-odps-algo&host=cn-hangzhou.oss-internal.aliyun-inc.com';"
]

parser = argparse.ArgumentParser(description="""
A script that helps you to automatically upload PAI tasks.

* Put the entire deploy folder WITHIN your project directory. The script can be used as
  > python deploy.py ABS_PATH_TO_ODPSCMD [--suppress_stderr]

* Your project should be managed under git, and only files tracked by git will be uploaded to PAI.
  To see which files will be uploaded, run the following command
  > git ls-tree --full-tree -r --name-only HEAD

* Currently, the entry file is configured to be <YOUR_PROJECT_ROOT>/main.py. If your entry file is
  not this file, please open deploy.py and change variable 'main_entry_file'.

* To add hyperparameter or any other pai command arguments, please open deploy.py and change variable
  'pai_commands'.

* You may use optional flag --suppress_stderr to hide messages from stderr.

""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('odps_path', help='Path to odps console')
parser.add_argument('--suppress_stderr', action='store_true')
args = parser.parse_args()

# Path to odps console
odps_path = args.odps_path

# boolean: if show stderr message
suppress_stderr = args.suppress_stderr

# Job url returned from odps
instance_id = None
job_url, job_id, token = None, None, None

# Printer lock
printer_lock = threading.Lock()

###################################
#  Creating code.tar.gz
###################################

print("Preparing tar ball...")
tar_process = subprocess.Popen(["bash", "-c",
                                "cd `git rev-parse --show-toplevel`;"
                                + "git ls-tree --full-tree -r --name-only HEAD | "
                                + "tar -czvf \"" + tar_ball_location + "\" -T -"],
                               stdout=subprocess.PIPE)
stdout_data, stderr_data = tar_process.communicate()
tar_process.wait()
print(stdout_data)
print("The tar ball is stored at " + tar_ball_location + "\n")

###################################
#  Launch ODPS + Create PAI Task
###################################
print("Now launching odps...")

odps_process = subprocess.Popen([odps_path],
                                shell=False, bufsize=0,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)


def output_handler(buf):
    global job_url, instance_id
    out = "".join(buf)
    if out.endswith('\n'):
        del buf[:]
        if out.startswith("ID = "):
            instance_id = out[5:].strip()
        if out.startswith("http://logview.odps.aliyun-inc.com:8080"):
            job_url = out
            odps_process.send_signal(signal.SIGINT)
    if out.startswith('odps@') and out.endswith('>'):
        del buf[:]
        if len(pai_commands) > 0:
            odps_process.stdin.write((pai_commands[0] + "\n").encode())
            pai_commands.pop(0)
            odps_process.stdin.flush()


threading.Thread(target=forward_fd,
                 args=(odps_process.stderr, sys.stdout, output_handler, lambda: job_url is not None)).start()
threading.Thread(target=forward_fd,
                 args=(odps_process.stdout, sys.stdout, output_handler, lambda: job_url is not None)).start()


###################################
#  Create Job Abort Handler
###################################
def abort_signal_handler(_, __):
    killed = False
    sys.stderr.write('\x1b[1;31m' + "\nNow killing PAI instance...\n" + '\x1b[0m')
    sys.stderr.flush()
    odps_process.terminate()

    odps_terminate_process = subprocess.Popen([odps_path],
                                              shell=False, bufsize=0,
                                              stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    pai_terminate_commands = [
        "use kelude_open_dw;",
        "kill INSTANCE_ID;"
    ]

    def odps_terminate_output_handler(buf):
        global job_url, instance_id, killed
        out = "".join(buf)
        if out.endswith('\n'):
            del buf[:]
            if out.startswith("OK"):
                odps_terminate_process.terminate()
                killed = True
        if out.startswith('odps@') and out.endswith('>'):
            del buf[:]
            if len(pai_terminate_commands) > 0:
                odps_terminate_process.stdin.write(
                    (pai_terminate_commands[0].replace("INSTANCE_ID", instance_id.strip()) + "\n").encode())
                pai_terminate_commands.pop(0)
                odps_terminate_process.stdin.flush()

    threading.Thread(target=forward_fd,
                     args=(odps_terminate_process.stdout, sys.stdout,
                           odps_terminate_output_handler, lambda: killed)).start()
    threading.Thread(target=forward_fd,
                     args=(odps_terminate_process.stderr, sys.stderr,
                           odps_terminate_output_handler, lambda: killed)).start()

    odps_terminate_process.wait()
    sys.exit()


signal.signal(signal.SIGINT, abort_signal_handler)

while job_url is None:
    time.sleep(0.5)

odps_process.terminate()


print("\nPai job launched. Retrieving job information...\n")

###################################
#  Retrieve Job ID and Status
###################################
parsed = urlparse(job_url)
job_id = parse_qs(parsed.query)['i'][0].strip()
token = parse_qs(parsed.query)['token'][0].strip()

print("ID: {}\nJob ID: {}\nToken: {}\n".format(instance_id, job_id, token))

print("Now waiting for the task to start...\n")

status = ""
wait_pos = None
task_name = None
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
