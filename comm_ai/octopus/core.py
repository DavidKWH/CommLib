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

import os
import sys
import json
import jsonpickle
import subprocess
from hotqueue import HotQueue as RedisQueue
from comm_ai.drive import save_to_file
from .messages import RunTaskMessage
from .util import conn_retry

# decorate function to catch exceptions
save_to_file = conn_retry(save_to_file)

################################################################################
# Main classes
################################################################################
class State:
    def __init__(self, task_id):
        self.task_id = task_id

class TaskGroup:
    '''
    Base class for Task handlers
    Include common parameters and features
    '''
    # look for user local data
    dirname = os.path.join(os.environ['HOME'], '.octopus')

    # connection file
    conn_name = 'redis_conn.json'
    conn_name = os.path.join(dirname, conn_name)

    def __init__(self, conn=None):

        conn = {}
        conn['host'] = 'localhost'
        conn['port'] = 16639
        conn['password'] = 'passwd'

        self.conn = conn

    def save_connection(self):
        with open( type(self).conn_name, 'w' ) as fp:
            json.dump(self.conn, fp)


class TaskSubmitter(TaskGroup):
    '''
    The main user facing class
    '''
    def __init__(self):
        # setup job queue
        #queue = HotQueue("myqueue", host="localhost", port=6379, db=0)
        with open( type(self).conn_name, 'r' ) as fp:
            conn = json.load(fp)
        self.task_queue = RedisQueue("to_runners", **conn)
        #self.task_queue = RedisQueue("to_runners")

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
        buf = task.serialize()
        save_to_file(fname, buf, text=True)

        # construct task state
        return State(task.task_id)

    def resubmit(self, task_file):
        ''' resubmit a task from json file '''

        with open(task_file, 'r') as fp:
            data = fp.read()
            task = jsonpickle.decode(data)

        msg = RunTaskMessage.from_task(task)

        # submit to queue
        self.task_queue.put(msg)

        # save to remote
        task = msg.get_task()
        task.status = 'resubmitted'
        tname = '.'.join((task.task_id, task.status))
        fname = '/'.join(('tasks', tname))
        print('saving task to:', fname)
        buf = task.serialize()
        save_to_file(fname, buf, text=True)

        # construct task state
        return State(task.task_id)


class TaskRunner(TaskGroup):
    '''
    Receives tasks over message queue
    '''
    def __init__(self):
        # setup job queue
        #queue = HotQueue("myqueue", host="localhost", port=6379, db=0)
        with open( type(self).conn_name, 'r' ) as fp:
            conn = json.load(fp)
        self.task_queue = RedisQueue("to_runners", **conn)
        #self.task_queue = RedisQueue("to_runners")

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
            print('waiting for msg...')
            msg = self.task_queue.get(block=True)
            print('got msg:')
            print('msg.type', msg.type)
            print('msg.script', msg.task.script)
            print('msg.input', msg.task.input)
            #print(msg)
            #print(msg.input)
            #print(msg.script.name)

            # save to remote
            task = msg.get_task()
            task.status = 'started'
            tname = '.'.join((task.task_id, task.status))
            fname = '/'.join(('tasks', tname))
            print('saving task to:', fname)
            buf = task.serialize()
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
            # add stdout to task
            task.output = state.stdout
            tname = '.'.join((task.task_id, task.status))
            fname = '/'.join(('tasks', tname))
            print('saving task to:', fname)
            buf = task.serialize()
            save_to_file(fname, buf, text=True)
            # save stdout to file
            tname = '.'.join((task.task_id, 'output'))
            fname = '/'.join(('tasks', tname))
            print('saving output to:', fname)
            save_to_file(fname, state.stdout, text=True)
