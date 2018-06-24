import json
import platform
import sys

import requests
import tensorflow as tf
from tensorflow.python.framework.errors_impl import UnimplementedError, NotFoundError


def detect_platform():
    is_pai = True
    try:
        tf.gfile.GFile("oss://file_not_existed", "r").read()
    except UnimplementedError:
        is_pai = False
    except NotFoundError:
        pass
    return 'PAI' if is_pai else platform.system().upper()


def print_header(header):
    write_stdout('\n'.join(['', '#' * 40, '# ' + header, '#' * 40, '']))


def write_stdout(content):
    sys.stdout.write(content)
    sys.stdout.flush()


def write_stderr(content):
    sys.stderr.write('\x1b[1;31m' + content + '\x1b[0m')
    sys.stderr.flush()


def forward_fd(process_fd, sys_fd, handler=None, stop=lambda: False):
    buf = []
    while not stop():
        buf.append(process_fd.read(1).decode())
        if handler is not None:
            handler(buf)
        if buf[-1] in ['\n', '>']:
            sys_fd.write("".join(buf))
            sys_fd.flush()
            del buf[:]


def get_odps_url(job_id, token, task):
    return "http://service-corp.odps.aliyun-inc.com/api/projects/kelude_open_dw/instances/{}?{}&authorization_token={}" \
        .format(job_id, task, token)


def send_odps_request(job_id, token, task, is_json=True):
    r = requests.get("http://logview.odps.aliyun-inc.com:8080/proxy", headers={
        'odps-proxy-url': get_odps_url(job_id, token, task)
    })
    if is_json:
        return json.loads(r.text)
    return r.text


def combine_overlap_string(old, new):
    new = new[max(i for i in range(len(new) + 1) if old.endswith(new[:i])):]
    return old + new, new


def retrieve_odps_status(job_id, token):
    return send_odps_request(job_id, token, "cached")


def format_odps_status_history(status):
    formatted_output = ""
    if 'subStatusHistory' not in status:
        return ""
    for sub_status in status['subStatusHistory']:
        formatted_output += "{} {} {}\n".format(sub_status['start_time'], sub_status['code'], sub_status['description'])
    return formatted_output


def retrieve_odps_log(job_id, token, log_id, log_type="Stdout"):
    log = send_odps_request(job_id, token, "log&logtype={}&size=10000&id={}".format(log_type, log_id), is_json=False)
    return log
