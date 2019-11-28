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

import sys
import json
import subprocess
from hotqueue import HotQueue as RedisQueue
from comm_ai.drive import save_to_file
from .messages import RunTaskMessage

class State:
    def __init__(self, task_id):
        self.task_id = task_id

class TaskGroup:
    '''
    Base class for Task handlers
    Include common parameters and features
    '''
    pass

class TaskInitiator:
    '''
    The main user facing class
    '''
    def __init__(self):
        # setup job queue
        #queue = HotQueue("myqueue", host="localhost", port=6379, db=0)
        self.task_queue = RedisQueue("to_runners")

    def submit(self,
               venv = 'venv-tf2',
               script = '',
               args = [],
               input = []):
        ''' high level function for task submission'''

        assert script, 'missing script argument'

        msg = RunTaskMessage(venv, script, args, input)

        # submit to queue
        self.task_queue.put(msg)

        # save to remote
        task = msg.get_task()
        task.status = 'submitted'
        tname = '.'.join((task.task_id, task.status))
        fname = '/'.join(('tasks', tname))
        print('saving task to:', fname)
        buf = json.dumps(task.as_dict(), indent=4)
        save_to_file(fname, buf, text=True)

        # construct task state
        return State(msg.task_id)


class TaskRunner:
    '''
    Receives tasks over message queue
    '''
    def __init__(self):
        # setup job queue
        #queue = HotQueue("myqueue", host="localhost", port=6379, db=0)
        self.task_queue = RedisQueue("to_runners")

    def main(self):
        # main function
        pass

    def run_task(self, task):

        # create script on disk
        script = task.script
        with open(script.name, 'w') as fp:
            fp.write(script.content)
        # create input file
        options = task.input
        with open(options.name, 'w') as fp:
            fp.write(options.content)

        cmd = []
        cmd.append(sys.executable)
        cmd.append(script.name)
        cmd.extend(['--options_from', 'stdin'])
        #cmd.extend(['--options_from', './options.json'])

        # command string
        cmd_str = ' '.join(cmd)
        print(f'command string: {cmd_str}')

        # invoke subprocess
        state = subprocess.run(cmd,
                               input=options.content,
                               universal_newlines=True,
                               #stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)

        print('process completed')
        return state


    def start(self):
        # create thread and runner indefinitely...

        while True:
            # get messsage from task_queue
            msg = self.task_queue.get(block=True)
            print('got msg:')
            #print(msg)
            print(msg.input)
            print(msg.script.name)

            # save to remote
            task = msg.get_task()
            task.status = 'started'
            tname = '.'.join((task.task_id, task.status))
            fname = '/'.join(('tasks', tname))
            print('saving task to:', fname)
            buf = json.dumps(task.as_dict(), indent=4)
            save_to_file(fname, buf, text=True)

            # run task
            print('running task in subprocess...')
            state = self.run_task(task)

            print('task complete')
            # save to remote
            if state.returncode:
                task.status = 'error'
            else:
                task.status = 'done'
            # save stdout
            task.stdout = state.stdout
            tname = '.'.join((task.task_id, task.status))
            fname = '/'.join(('tasks', tname))
            print('saving task to:', fname)
            buf = json.dumps(task.as_dict(), indent=4)
            save_to_file(fname, buf, text=True)
