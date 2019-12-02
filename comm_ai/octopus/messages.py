'''
define messages
'''
import os
import json

from .util import message_id
from .util import hash_fn

################################################################################
# help functions for object serialization
################################################################################
#def convert_to_dict(obj):
#    """
#    A function takes in a custom object and returns a dictionary representation of the object.
#    This dict representation includes meta data such as the object's module and class names.
#    """
#
#    #  Populate the dictionary with object meta data
#    obj_dict = {
#        "__class__": obj.__class__.__name__,
#        "__module__": obj.__module__
#    }
#
#    #  Populate the dictionary with object properties
#    obj_dict.update(obj.__dict__)
#
#    return obj_dict

def convert_to_rdict(obj):
    """
    A function takes in a custom object and returns a dictionary representation of the object.
    This dict representation includes meta data such as the object's module and class names.
    """

    #  Populate the dictionary with object meta data
    obj_dict = {
        "__class__": obj.__class__.__name__,
        "__module__": obj.__module__
    }

    #  Populate the dictionary with object properties
    #  Construct nested dict recursively
    for key, val in obj.__dict__.items():
        if hasattr(val, "__class__") and hasattr(val, "__dict__"):
            d = convert_to_rdict(val)
            obj_dict.update({key: d})
        else:
            obj_dict.update({key: val})

    return obj_dict


def dict_to_obj(our_dict):
    """
    Function that takes in a dict and returns a custom object associated with the dict.
    This function makes use of the "__module__" and "__class__" metadata in the dictionary
    to know which object type to create.
    """
    if "__class__" in our_dict:
        # Pop ensures we remove metadata from the dict to leave only the instance arguments
        class_name = our_dict.pop("__class__")

        # Get the module name from the dict and import it
        module_name = our_dict.pop("__module__")

        # We use the built in __import__ function since the module name is not yet known at runtime
        module = __import__(module_name)

        # Get the class from the module
        class_ = getattr(module,class_name)

        # Use dictionary unpacking to initialize the object
        obj = class_(**our_dict)
    else:
        obj = our_dict

    return obj


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
    def __init__(self, venv, args, input, script, task_id):
        self.venv = venv
        self.args = args
        self.input = input
        self.script = script
        self.task_id = task_id

    def __repr__(self):
        keys = self.__dict__.keys()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def serialize(self):
        return json.dumps(self, default=convert_to_rdict, indent=4)

    def deserialize(self, buf):
        return json.loads(buf, object_hook=dict_to_obj)

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

    def get_task(self):
        return Task(self.venv,
                    self.args,
                    self.input,
                    self.script,
                    self.task_id)

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

