'''
define messages
'''
import os

from .util import message_id

class Message:

    def __init__(self):
        ''' generate message id '''
        self.msg_id = message_id()

################################################################################
# messages to runners
################################################################################
class RunTaskMessage(Message):

    def __init__(self, venv, spath, args):
        super().__init__()
        self.type = 'run_task'
        self.venv = venv
        self.args = args
        # process script
        os.path.isfile(spath)
        with open(spath, 'r') as fp:
            script_dict = {
                    'name', script,
                    'blob', fp.read()
                    }
        self.script = script_dict

class CheckStatusMessage(Message):

    def __init__(self, task_id):
        super().__init__()
        self.type = 'check_status'
        self.task_id = task_id

class TerminateMessage(Message):

    def __init__(self, task_id):
        super().__init__()
        self.type = 'terminate'
        self.task_id = task_id

################################################################################
# messages from runners
################################################################################
class StatusMessage(Message):

    def __init__(self, src_msg, status):
        super().__init__()
        self.type = 'status'
        self.status = status
        self.src_msg = src_msg

class TaskStructMessage(Message):

    def __init__(self, src_msg, task_id):
        super().__init__()
        self.type = 'task_struct'
        self.task_id = task_id
        self.src_msg = src_msg

class TaskEndedMessage(Message):

    def __init__(self, src_msg,
                       task_id,
                       status,
                       stdout,
                       stderr):
        super().__init__()
        self.type = 'task_ended'
        self.task_id = task_id
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.src_msg = src_msg

