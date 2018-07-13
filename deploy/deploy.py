import argparse
import os
import signal
import subprocess
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
  > python deploy.py [--suppress_stderr] [--job_url JOB_URL]

* Your project should be managed under git, and only files tracked by git will be uploaded to PAI.
  To see which files will be uploaded, run the following command
  > git ls-tree --full-tree -r --name-only HEAD

* You should specify the path of your odps command-line tool by changing odps_path in config.py

* You may ignore certain folders by changing tar_exclude option in config.py

* You can select the entry for of your pai job by changing pai_main_entry_file option in config.py

* To add hyperparameter or any other pai command arguments, please open deploy.py and change variable
  'pai_command'.

* You may use optional flag --suppress_stderr to hide messages from stderr.

* If you want to monitor a job that has already been deployed, use --job_url option and provide
  the job url of your job

""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--instance_id', default=None, help='An already existed instance id')
parser.add_argument('--job_url', default=None, help='An already existed job url')
parser.add_argument('--suppress_stderr', action='store_true')
args = parser.parse_args()

# Path to odps console
odps_path = deploy_config['odps_path']

# Instance ID and job url from odps
instance_id, job_url = args.instance_id, args.job_url

# boolean: whether to show stderr message
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


def task_launch_tar():
    # Start tar process
    cd_cmd = "cd `git rev-parse --show-toplevel`"
    list_cmd = "git ls-tree --full-tree -r --name-only HEAD"
    tar_cmd = "tar -czvf \"" + tar_ball_location + "\""
    for exclude in deploy_config['tar_exclude']:
        tar_cmd += " --exclude \"{}\"".format(exclude)
    tar_cmd += " -T -"

    tar_process = subprocess.Popen(["bash", "-c",
                                    cd_cmd + ";" + list_cmd + " | " + tar_cmd],
                                   stdout=subprocess.PIPE)
    tar_stdout_thread = threading.Thread(target=forward_fd, args=(tar_process.stdout, sys.stdout, None,
                                                                  lambda: tar_process.poll() is None))
    tar_stderr_thread = threading.Thread(target=forward_fd, args=(tar_process.stderr, sys.stdout, None,
                                                                  lambda: tar_process.poll() is None))

    tar_stdout_thread.start()
    tar_stderr_thread.start()

    tar_process.wait()
    tar_stdout_thread.join()
    tar_stderr_thread.join()
    time.sleep(0.5)

    write_stdout("The tar ball is stored at " + tar_ball_location + "\n")


if instance_id is None or job_url is None:
    task_launch_tar()
else:
    write_stdout("Skipped\n")

###################################
#  Launch ODPS + Create PAI Task
###################################
write_stdout("\nNow launching odps...\n")

odps_process = None


def task_launch_odps():
    global odps_process
    # Pai commands
    pai_command = "pai -name {}".format(deploy_config['pai_algo'])
    pai_command += " -Dscript='file://{}'".format(tar_ball_location_pai)
    pai_command += " -DentryFile='{}'".format(deploy_config['pai_main_entry_file'])
    pai_command += " -DcheckpointDir='{}'".format(deploy_config['oss_role_arn'])
    pai_command += " -Dbuckets='{}';".format(deploy_config['oss_role_arn'])

    # Start odps process
    odps_process = subprocess.Popen([odps_path,
                                     '--project', deploy_config['odps_project'],
                                     '-e', pai_command],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)

    def odps_output_handler(buf):
        global job_url, instance_id
        output = "".join(buf)
        if output.endswith('\n'):
            output = output.strip()
            if output.startswith("ID = "):
                instance_id = output[5:]
            if output.startswith("http://logview.odps.aliyun-inc.com:8080"):
                job_url = output
                odps_process.terminate()
            if output.startswith("FAILED"):
                die("Pai Job Deployment Failed ({})".format(output))

    odps_stderr_thread = threading.Thread(target=forward_fd,
                                          args=(odps_process.stderr, sys.stdout, odps_output_handler,
                                                lambda: odps_process.poll() is not None))
    odps_stdout_thread = threading.Thread(target=forward_fd,
                                          args=(odps_process.stdout, sys.stdout, odps_output_handler,
                                                lambda: odps_process.poll() is not None))

    odps_stderr_thread.start()
    odps_stdout_thread.start()

    odps_process.wait()

    if detect_platform() != "WINDOWS":
        odps_stderr_thread.join()
        odps_stdout_thread.join()

        os.remove(tar_ball_location_pai)


###################################
#  Create Job Abort Handler
###################################
def task_terminate_handler():
    def abort_signal_handler(_, __):
        global instance_id, odps_process

        write_stderr("\nNow killing PAI instance...\n")
        if odps_process is not None:
            try:
                odps_process.terminate()
            except OSError:
                pass

        if instance_id is None:
            die("PAI Instance has not been launched.\n")

        odps_terminate_process = subprocess.Popen([odps_path, '--project', deploy_config['odps_project'],
                                                   '-e', 'kill ' + instance_id],
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE)

        def odps_terminate_output_handler(buf):
            output = "".join(buf)
            if output.endswith('\n'):
                output = output.strip()
                if output.startswith("OK") or output.startswith("FAILED"):
                    odps_terminate_process.terminate()

        odps_terminate_stdout_thread = threading.Thread(target=forward_fd,
                                                        args=(odps_terminate_process.stdout, sys.stdout,
                                                              odps_terminate_output_handler,
                                                              lambda: odps_terminate_process.poll() is not None))
        odps_terminate_stderr_thread = threading.Thread(target=forward_fd,
                                                        args=(odps_terminate_process.stderr, sys.stderr,
                                                              odps_terminate_output_handler,
                                                              lambda: odps_terminate_process.poll() is not None))
        odps_terminate_stdout_thread.start()
        odps_terminate_stderr_thread.start()

        odps_terminate_process.wait()
        odps_terminate_stdout_thread.join()
        odps_terminate_stderr_thread.join()

        exit(0)

    signal.signal(signal.SIGINT, abort_signal_handler)


task_terminate_handler()

if instance_id is None or job_url is None:
    task_launch_odps()
else:
    write_stdout("ID = {}\n".format(instance_id))
    write_stdout("{}\n".format(job_url))

if instance_id is None:
    die("Cannot retrieve instance ID")

if job_url is None:
    die("Cannot retrieve job url")


###################################
#  Retrieve Job ID
###################################
write_stdout("\nPai job launched. Retrieving job information...\n")

query = parse_qs(urlparse(job_url).query)
job_id = query['i'][0].strip()
token = query['token'][0].strip()

write_stdout("Job ID = {}\n".format(job_id))
write_stdout("Token  = {}\n".format(token))


###################################
#  Retrieve Job Status
###################################
write_stdout("\nNow waiting for the task to start...\n")

wait_pos, queue_length, task_name, log_id = None, None, None, None

status_history = ""

while log_id is None or len(log_id) == 0:
    assert_odps_is_running(job_id, token, task_name)
    cached = retrieve_odps_cached(job_id, token)
    if 'taskName' in cached and cached['taskName'] != task_name:
        task_name = cached['taskName']
    if 'waitPos' in cached and cached['waitPos'] != wait_pos:
        wait_pos = cached['waitPos']
    if 'queueLength' in cached and cached['queueLength'] != queue_length:
        queue_length = cached['queueLength']
    if 'subStatusHistory' in cached:
        status_history, new_status_history = \
            combine_overlap_string(status_history, format_odps_status_history(cached))
        write_stdout('\r' + new_status_history)

    if wait_pos is not None:
        write_stdout('\r< Current Waitlist Position: {} / {} >'.format(wait_pos, queue_length))
    if wait_pos == 0:
        write_stdout('\r' + ' ' * 50)

    if wait_pos is not None and wait_pos == 0 and task_name is not None:
        try:
            detail = retrieve_odps_detail(job_id, token, task_name)
            log_id = detail['mapReduce']['jobs'][0]['tasks'][0]['instances'][0]['logId']
        except (KeyError, IndexError) as e:
            pass
    time.sleep(1)

write_stdout("\nTask Name = {}\nLog ID = {}\n".format(task_name, log_id))


#######################################
#  Connect to Remote Stdout & Stderr
#######################################
write_stdout("\nTask is now running. Connecting to remote console...\n")

stdout, stderr = "", ""

while True:
    assert_odps_is_running(job_id, token, task_name)
    stdout, new_stdout = \
        combine_overlap_string(stdout, retrieve_odps_log(job_id, token, log_id, log_type="Stdout"))
    write_stdout(new_stdout)
    if not suppress_stderr:
        stderr, new_stderr = \
            combine_overlap_string(stderr, retrieve_odps_log(job_id, token, log_id, log_type="Stderr"))
        write_stderr(new_stderr)
    time.sleep(1)
