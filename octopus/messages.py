'''
define messages
'''
import os

from .util import message_id
from .util import hash_fn

################################################################################
# helper classes
################################################################################
class Script:
    '''
    NOTE: everything is stored as strings
    '''
    def __init__(self, path):
        assert os.path.isfile(path), f'path does not exist: {path}'
        name = os.path.basename(path)
        self.name = name
        with open(path, 'r') as fp:
            self.content = fp.read()
        self.hash = hash_fn(self.content.encode())

class Input:
    def __init__(self, input, name='options'):
        # add extension if not specified
        if not os.path.splitext(name)[1]:
            name = name + '.txt'
        self.name = name
        self.content = input
        self.hash = hash_fn(self.content.encode())

################################################################################
# messages to runners
################################################################################
class Message:

    def __init__(self):
        ''' generate message id '''
        self.msg_id = message_id()

class RunTaskMessage(Message):

    def __init__(self, venv, spath, args, input):
        '''
        task ID automatically associated to
        message ID
        '''
        super().__init__()
        self.type = 'run_task'
        self.venv = venv
        self.args = args
        self.input = Input(input)
        self.script = Script(spath)
        # set task ID to message ID
        self.task_id = self.msg_id

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

