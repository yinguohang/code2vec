import argparse
import os
import subprocess
import thread
import time
import urlparse

from utils import *

parser = argparse.ArgumentParser(description='PAI deployment')
parser.add_argument('odps_path', help='Path to odps console')
parser.add_argument('--suppress_stderr', action='store_true')
args = parser.parse_args()

# Path to odps console
odps_path = args.odps_path

# boolean: if show stderr message
suppress_stderr = args.suppress_stderr

# Path to store code.tar.gz
tar_ball_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code.tar.gz")

# Main entry file of the code
main_entry_file = "main.py"

# Job url returned from odps
job_url = None

###################################
#  Creating code.tar.gz
###################################

print("Preparing tar ball...")
tar_process = subprocess.Popen(["bash", "-c",
                                "cd `git rev-parse --show-toplevel`;"
                                + "git ls-tree --full-tree -r --name-only HEAD | "
                                + "tar -czvf " + tar_ball_location + " -T -"],
                               stdout=subprocess.PIPE)
stdout_data, stderr_data = tar_process.communicate()
tar_process.wait()
print(stdout_data)
print("The tar ball is stored at " + tar_ball_location + "\n")

###################################
#  Launch ODPS + Create PAI Task
###################################
print("Now launching odps...")

process = subprocess.Popen([odps_path],
                           shell=False, bufsize=0,
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)

commands = [
    "use kelude_open_dw;",
    "pai -name tensorflow140"
    + " -Dscript='file://{}'".format(tar_ball_location)
    + " -DentryFile='{}'".format(main_entry_file)
    + " -DcheckpointDir='oss://apsalgo-hz/?role_arn=acs:ram::1396633922344963:role/aps-odps-algo&host=cn-hangzhou.oss-internal.aliyun-inc.com'"
    + " -Dbuckets='oss://apsalgo-hz/?role_arn=acs:ram::1396633922344963:role/aps-odps-algo&host=cn-hangzhou.oss-internal.aliyun-inc.com';"
]


def output_handler(buf):
    global job_url
    out = "".join(buf)
    if out.endswith('\n'):
        del buf[:]
        if out.startswith("http://logview.odps.aliyun-inc.com:8080"):
            job_url = out
            process.terminate()
    if out.startswith('odps@') and out.endswith('>'):
        del buf[:]
        if len(commands) > 0:
            process.stdin.write(commands[0] + "\n")
            commands.pop(0)
            process.stdin.flush()
        else:
            process.terminate()


thread.start_new_thread(forward_fd, (process.stderr, sys.stdout, output_handler, lambda: job_url is not None))
thread.start_new_thread(forward_fd, (process.stdout, sys.stdout, output_handler, lambda: job_url is not None))

process.wait()

print("Pai job launched. Retrieving job information...\n")

###################################
#  Retrieve Job ID and Status
###################################
parsed = urlparse.urlparse(job_url)
job_id = urlparse.parse_qs(parsed.query)['i'][0].strip()
token = urlparse.parse_qs(parsed.query)['token'][0].strip()

print("Job ID: {}\nToken: {}\n".format(job_id, token))

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
while True:
    stdout, new_stdout = combine_overlap_string(stdout, retrieve_odps_log(job_id, token, log_id, log_type="Stdout"))
    sys.stdout.write(new_stdout)
    sys.stdout.flush()
    if not suppress_stderr:
        stderr, new_stderr = combine_overlap_string(stderr, retrieve_odps_log(job_id, token, log_id, log_type="Stderr"))
        sys.stderr.write('\x1b[1;31m' + new_stderr + '\x1b[0m')
        sys.stderr.flush()
    time.sleep(1)
