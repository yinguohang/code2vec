import json
import platform
import sys
import threading
from collections import OrderedDict

import requests
import tensorflow as tf
import xmltodict
from dateutil.parser import parse
from tensorflow.python.framework.errors_impl import UnimplementedError, NotFoundError

printer_lock = threading.Lock()


def detect_platform():
    is_pai = True
    try:
        tf.gfile.GFile("oss://file_not_existed", "r").read()
    except UnimplementedError:
        is_pai = False
    except NotFoundError:
        pass
    return 'PAI' if is_pai else platform.system().upper()


def write_fd(fd, content):
    printer_lock.acquire()
    fd.write(content)
    fd.flush()
    printer_lock.release()


def write_stdout(content):
    write_fd(sys.stdout, content)


def write_stderr(content):
    write_fd(sys.stderr, '\x1b[1;31m' + content + '\x1b[0m')


def die(message):
    write_stderr('\n' + message + '\n')
    exit(1)


def forward_fd(process_fd, sys_fd, handler=None, stop=lambda: False):
    buf = []
    while not stop():
        buf.append(process_fd.read(1).decode(encoding=sys.getdefaultencoding()))
        if handler is not None:
            handler(buf)
        if buf[-1] in ['\n', '>']:
            write_fd(sys_fd, "".join(buf))
            del buf[:]


def get_odps_url(job_id, token, task):
    return "http://service-corp.odps.aliyun-inc.com/api/projects/kelude_open_dw/instances/" \
           + "{}?{}&authorization_token={}".format(job_id, task, token)


def send_odps_request(job_id, token, task, content_type=None):
    r = requests.get("http://logview.odps.aliyun-inc.com:8080/proxy", headers={
        'odps-proxy-url': get_odps_url(job_id, token, task)
    })
    if content_type is None:
        content_type = r.headers['content-type']

    if content_type in ["text/plain", "plain", "text"]:
        return r.text
    if content_type in ["application/json", "json"]:
        try:
            return json.loads(r.text)
        except ValueError:
            write_stderr(r.text)
            write_stderr("Parse JSON failed\n")
            return {}
    if content_type in ["application/xml", "xml"]:
        try:
            return xmltodict.parse(r.text)
        except ValueError:
            write_stderr(r.text)
            write_stderr("Parse XML failed\n")
            return {}
    return r.text


def combine_overlap_string(old, new):
    if type(new) is OrderedDict:
        write_stderr("Error encountered during printing remote output:\n")
        for key in new:
            write_stderr("{}: {}\n".format(key, new[key]))
        return
    new = new[max(i for i in range(len(new) + 1) if old.endswith(new[:i])):]
    return old + new, new


def assert_odps_is_running(job_id, token, task_name, status="Running"):
    s = retrieve_odps_status(job_id, token)
    if s != status:
        if task_name is not None:
            write_stderr("\nTask result: {}\n".format(retrieve_odps_result(job_id, token, task_name)))
        die("Pai job has been ended [{}]".format(s))


def retrieve_odps_status(job_id, token):
    return send_odps_request(job_id, token, "")['Instance']['Status']


def retrieve_odps_cached(job_id, token):
    return send_odps_request(job_id, token, "cached")


def retrieve_odps_result(job_id, token, task_name):
    result = send_odps_request(job_id, token, "result&taskname=" + task_name)
    return result['Instance']['Tasks']['Task']['Result']['#text']


def retrieve_odps_detail(job_id, token, task_name):
    return send_odps_request(job_id, token, "detail&taskname=" + task_name)


def retrieve_odps_log(job_id, token, log_id, log_type="Stdout"):
    log = send_odps_request(job_id, token, "log&logtype={}&size=10000&id={}".format(log_type, log_id))
    return log


def format_odps_status_history(status):
    formatted_output = ""
    if 'subStatusHistory' not in status:
        return ""
    for sub_status in status['subStatusHistory']:
        formatted_output += "{:%Y-%m-%d %H:%M:%S} - {}\n" \
            .format(parse(sub_status['start_time']), sub_status['description'])
    return formatted_output
