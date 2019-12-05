'''
define messages
'''
import os
import json
import jsonpickle

from .util import message_id
from .util import hash_fn

################################################################################
# serializer setup
################################################################################
jsonpickle.set_encoder_options('json', indent=4)

################################################################################
# helper classes
################################################################################
class Script:
    '''
    NOTE: everything is stored as strings
    '''
    def __init__(self, path=None, content=None, name=None, hash=None):
        if path:
            assert os.path.isfile(path), f'path does not exist: {path}'
            name = os.path.basename(path)
            with open(path, 'r') as fp:
                self.content = fp.read()
                self.name = name
        else:
            self.name = name
            self.content = content
        # finally
        self.hash = hash_fn(self.content.encode())

    def __repr__(self):
        keys = self.__dict__.keys()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))


class Input:
    def __init__(self, content, name='options', hash=None):
        # add extension if not specified
        if not os.path.splitext(name)[1]:
            name = name + '.txt'
        self.name = name
        self.content = content
        self.hash = hash_fn(self.content.encode())

    def __repr__(self):
        keys = self.__dict__.keys()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

class Task:
    '''
    Task Info
    '''
    def __init__(self, venv, args, input, script, task_id, attempt=None):
        self.venv = venv
        self.args = args
        self.input = input
        self.script = script
        self.task_id = task_id
        self.attempt = attempt if attempt else 1
        self.output = None

    def __repr__(self):
        keys = self.__dict__.keys()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def serialize(self):
        #return json.dumps(self, default=convert_to_rdict, indent=4)
        return jsonpickle.encode(self)

    def deserialize(self, buf):
        #return json.loads(buf, object_hook=dict_to_obj)
        return jsonpickle.decode(buf)

################################################################################
# messages to runners
################################################################################
class Message:

    def __init__(self):
        ''' generate message id '''
        self.msg_id = message_id()

    def __repr__(self):
        keys = self.__dict__.keys()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

class RunTaskMessage(Message):
    '''
    create a run task message.
    Use class method to create message from task, i.e.

        msg = RunTaskMessage.from_task(task)
    '''

    def __init__(self, venv, spath, args, input,
                 task_id=None, attempt=None, task=None):
        '''
        NOTE: set task ID to msg ID if a new task is being created
        '''
        super().__init__()
        self.type = 'run_task'
        task_id = task_id if task_id else self.msg_id
        attempt = attempt if attempt else 1

        if task:
            task.attempt += 1
            task.output = None
            self.task = task
        else:
            # create task
            self.task = Task(venv,
                            args,
                            Input(input),
                            Script(spath),
                            task_id,
                            attempt
                            )

    @classmethod
    def from_task(cls, task: Task):
    	return cls('', '', '', '', task=task)

    def get_task(self):
        return self.task

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

