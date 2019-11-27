'''
Implement a simple message queue.  Using rclone as message
store.

This forms the basis for a task scheduler system, over
very heterogeneous environment.

Queues are implemented with file structure:

For messages:
    queues/from_runners/messages...
           to_runners/messages...

Message storage:
    storage/from_runners/blobs...
            to_runners/blobs...

For task management:
    tasks/task_1...n

Model output:
    models/blobs...

Mutexes queue (push) synchronization:

    mutexes/queue_id/task_id.lock

A message contains a manifest and a blob (both id tagged).
Manifest is a python dict encoded as json.  A blob is a
tar.gzipped file to give some level of integrity checking
(gzip contains a CRC checksum).  For message[id] we have:

    manifest_id.json
    blob_id.tar.gz

Nodes have a name and belongs to group(s) (e.g., initiator,
runner and master).  A node can belong to multiple groups.
For example, a simple configuration creates a runner/master
node instance to perform both roles.
nodes: mumble15, nico19

Master performs maintainance tasks: node management,
queue allocation, queue trimming, etc.
Runners run tasks.

Message types: I envision these messages for now
to mimic the subprocess module

    run_task - to runners
    task_struct - from runner
    task_ended - from runner
    check_status - to runner
    status - from runner
    terminate - to runner

Message format:

    { src: node_name,
      dst: node or group name,
      msg: specific instance }

Message instance format:

    { type: run_task,
      virtual_env: interpretor path,
      script: script name (in blob)
      args: arguments for script }

    { type: task_struct,
      id: task_id }

    { type: task_ended,
      status: ok|error,
      stdout: ditto
      stderr: ditto }

    { type: check_status,
      id: task_id }

    { type: status,
      status: running|complete|error|stopped_by_user }

    { type: terminate,
      id: task_id }

Starting subprocess with specific venv
subprocess.Popen(["virtualenv1/bin/python", "my_script.py"])
subprocess.Popen(["virtualenv2/bin/python", "my_other_script.py"])
'''

from .messages import RunTaskMessage

def submit(venv = 'venv-tf2',
           script = '',
           args = [],
           input = []):
    ''' high level function for task submission'''

    assert script, 'missing script argument'

    msg = RunTaskMessage(venv, script, args, input)

    # submit to queue
    # q.submit(msg)

    # construct task state
    state = { 'task_id': task_id }

    return state


################################################################################
# OLD STUFF
################################################################################

class TaskInitiator:
    '''
    The main user facing class
    '''
    def __init__(self):
        pass

    def submit(messages):
        ''' 
        returns an iterator of outstanding tasks
        '''
        return

class TaskRunner:
    '''
    Receives tasks over message queue
    '''
    def __init__(self):
        pass

################################################################################
# define sendable/storable objects
################################################################################
class File:
    pass

################################################################################
# define rclone store
################################################################################
class Rclone:
    pass

################################################################################
# main queue component
################################################################################
class Queue:
    '''
    '''
    pass
