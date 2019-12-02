'''tools for hyperparameter optimization

The function run_hyperopt_task() schedules tasks to be run either
locally or submitted to a task runner.  Three groups of parameters
can be specified.

  * _overrides_ is a dictionary of param-value pairs that are
    fixed for each task
  * _sweeps_ is a dictionary containing param and value lists.
    all lists are combined into a cartesian product and permuted.
  * _pairs_ is a list of dictionaries, each dictionary consists
    of a tuple of params and the value lists are zipped and iterated
    together as tuples.  The lists within each dictionary must have
    the same length.

Example:
    SNAME = os.path.basename(__file__).replace('.','_')
    SCRIPT = 'test_set_rparams.py'
    OUTDIR = 'sweep_adam_beta_2'

    override_dict = {
        'p.outdir': to_str(OUTDIR),
        'p.lr_sched.decay_schedule': [0, 19],
        'p.adam': 'p.adam1',
    }

    sweep_dict = {
        'p.n_layers': [3, 5],
        'p.n_epochs': [20, 30],
    }

    zip_dict = [
        { 'p.adam.beta_1': [0.5, 0.9],
          'p.adam.beta_2': [0.99, 0.999],
        },
    ]

    run_hyperopt_task(SCRIPT,
                      OUTDIR,
                      name=NAME,
                      overrides=override_dict,
                      sweeps=sweep_dict,
                      pairs=zip_dict,
                      )
'''

import os
import sys
import subprocess
import json
import itertools as itools
from collections import Iterable
from sys import exit

from comm_ai import drive
from comm_ai.octopus import TaskSubmitter

################################################################################
# support functions
################################################################################
def to_str(arg):
    return f'r"{arg}"'

def flatten_r(items):
    'Yield items from any nested iterable'
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten_r(x):
                yield sub_x
        else:
            yield x

################################################################################
# main algorithm
################################################################################
def run_hyperopt_task(script,
                      outdir,
                      name='sweep_hyperparam',
                      dry_run=False,
                      overrides={},
                      sweeps={},
                      pairs=[],
                      ):

    if dry_run: print('NOTE: dry run, no tasks will be scheduled')

    # ensure all lists (in pairs) have the same length
    for d in pairs:
        lens = [len(l) for l in d.values()]
        assert len(set(lens)) == 1, 'lists must have the same length'

    cmd = []
    cmd.append(sys.executable)
    cmd.append(script)
    #cmd.append('-h')
    #cmd.append('--override_pdict')
    cmd.extend(['--options_from', 'stdin'])
    #cmd.extend(['--options_from', './options.json'])
    #cmd.extend(['--params_from', 'rparams_test.json'])

    # command string
    cmd_str = ' '.join(cmd)
    print(f'command string: {cmd_str}')

    # ensure folder exists
    os.makedirs(outdir, exist_ok=True)

    # construct nested keys and values
    nested_keys = []
    nested_vals = []
    nested_keys.extend(sweeps.keys())
    nested_vals.extend(sweeps.values())
    for d in pairs:
        nested_keys.append( d.keys() )
        tuples = list( zip(*d.values()) )
        nested_vals.append( tuples )
    #print('nested_keys =', nested_keys)
    #print('nested_vals =', nested_vals)
    keys = list(flatten_r(nested_keys))
    #print('flatten keys =', keys)

    # save permutation table to file
    pname = '/'.join((outdir, name))
    fname = pname + '.index'
    with open(fname, 'w') as fp:
        print('writing index table to file:', fname)

        #for perm in itools.product(*nested_vals):
        for i, perm in enumerate(itools.product(*nested_vals)):
            vals = list(flatten_r(perm))
            #print('perm =', perm)
            #print('vals =', vals)
            config = {}
            [config.update({key: val}) for key, val in zip(keys, vals)]
            config_str = f'{i:04d}: {config}'
            fp.write(config_str + '\n')
            print(config_str)

    if dry_run: return

    # iterate over all possible permutations
    # spawn subprocess to train each permutation
    #for perm in itools.product(*nested_vals):
    for i, perm in enumerate(itools.product(*nested_vals)):
        vals = flatten_r(perm)
        config = {}
        [config.update({key: val}) for key, val in zip(keys, vals)]
        print(config)

        # append parameters to pdict
        pdict = overrides.copy()
        [pdict.update({opt: val}) for opt, val in config.items()]

        # send json to stdin
        overrides_json = json.dumps(pdict)
        print(f'json input: {overrides_json}')

        # invoke subprocess
        state = subprocess.run(cmd,
                                input=overrides_json,
                                universal_newlines=True,
                                #stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)

        # save log to file
        pname = f'{name}_{i:04d}'
        pname = '/'.join((outdir, pname))
        fname = pname + '.log'
        print('writing log to file:', fname)

        with open(fname, 'w') as fp:
            fp.write(f'command: {cmd_str}\n\n')
            fp.write(f'overrides: {overrides_json}\n\n')
            fp.write(state.stdout)

################################################################################
# remote task submission
################################################################################
def run_hyperopt_task_remote(script,
                      outdir,
                      mode='local',
                      #mode='remote',
                      venv='venv-tf2',
                      args=[],
                      name='sweep_hyperparam',
                      dry_run=False,
                      overrides={},
                      sweeps={},
                      pairs=[],
                      ):
    ''' submit remote task '''

    if dry_run: print('NOTE: dry run, no tasks will be scheduled')

    # ensure all lists (in pairs) have the same length
    for d in pairs:
        lens = [len(l) for l in d.values()]
        assert len(set(lens)) == 1, 'lists must have the same length'

    # script arguments
    #args = []
    #args.extend(['--options_from', 'stdin'])

    # command string
    #cmd_str = ' '.join(cmd)
    #print(f'command string: {cmd_str}')

    # ensure folder exists
    os.makedirs(outdir, exist_ok=True)

    # construct nested keys and values
    nested_keys = []
    nested_vals = []
    nested_keys.extend(sweeps.keys())
    nested_vals.extend(sweeps.values())
    for d in pairs:
        nested_keys.append( d.keys() )
        tuples = list( zip(*d.values()) )
        nested_vals.append( tuples )
    #print('nested_keys =', nested_keys)
    #print('nested_vals =', nested_vals)
    keys = list(flatten_r(nested_keys))
    #print('flatten keys =', keys)

    # save permutation table to file
    pname = '/'.join((outdir, name))
    fname = pname + '.index'
    with open(fname, 'w') as fp:
        print('writing index table to file:', fname)

        #for perm in itools.product(*nested_vals):
        for i, perm in enumerate(itools.product(*nested_vals)):
            vals = list(flatten_r(perm))
            #print('perm =', perm)
            #print('vals =', vals)
            config = {}
            [config.update({key: val}) for key, val in zip(keys, vals)]
            config_str = f'{i:04d}: {config}'
            fp.write(config_str + '\n')
            print(config_str)

    # save index file
    drive.save_file(fname, mime_type='text/plain')

    if dry_run: return

    # get task initiator instance
    task_queue = TaskSubmitter()

    # iterate over all possible permutations
    # submit task to octopus
    #for perm in itools.product(*nested_vals):
    for i, perm in enumerate(itools.product(*nested_vals)):
        vals = flatten_r(perm)
        config = {}
        [config.update({key: val}) for key, val in zip(keys, vals)]
        print(config)

        # append parameters to pdict
        pdict = overrides.copy()
        [pdict.update({opt: val}) for opt, val in config.items()]

        # send json to stdin
        overrides_json = json.dumps(pdict)
        print(f'json input: {overrides_json}')

        # construct task dict
        task = {}
        task['venv'] = venv
        task['script'] = script
        task['args'] = args
        task['input'] = overrides_json

        # submit to octopus
        # TODO: implement function
        state = task_queue.submit(**task)

        # save log to file
        pname = f'{name}_{i:04d}'
        pname = '/'.join((outdir, pname))
        fname = pname + '.log'
        print('writing log to file:', fname)

        with open(fname, 'w') as fp:
            fp.write(f'task_id: {state.task_id}\n\n')
            fp.write(f'task: {task}\n\n')
            fp.write(f'overrides: {overrides_json}\n\n')

        # save to google drive
        drive.save_file(fname)

