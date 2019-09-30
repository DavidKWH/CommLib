'''tools for hyperparameter optimization

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
        'p.adam.beta_2': [0.99, 0.999],
    }

    run_hyperopt_task(SCRIPT,
                      OUTDIR,
                      name=NAME, 
                      overrides=override_dict,
                      sweeps=sweep_dict,
                      )
'''

import os
import sys
import subprocess
import json
import itertools as itools
#from sys import exit

################################################################################
# support functions
################################################################################
def to_str(arg):
    return f'r"{arg}"'

################################################################################
# main algorithm
################################################################################
def run_hyperopt_task(script,
                      outdir,
                      name='sweep_hyperparam',
                      overrides={},
                      sweeps={}
                      ):
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

    # save combination table to file
    pname = '/'.join((outdir, name))
    fname = pname + '.index'
    with open(fname, 'w') as fp:
        print('writing index table to file:', fname)
        #for comb in itools.product(*sweeps.values()):
        for i, comb in enumerate(itools.product(*sweeps.values())):
            config = {}
            [config.update({key: val}) for key, val in zip(sweeps.keys(), comb)]
            config_str = f'{i:04d}: {config}'
            fp.write(config_str + '\n')
            #print(config_str)


    # iterate over all possible combinations
    # spawn subprocess to train each combination
    #for comb in itools.product(*sweeps.values()):
    for i, comb in enumerate(itools.product(*sweeps.values())):
        config = {}
        [config.update({key: val}) for key, val in zip(sweeps.keys(), comb)]
        print(config)
        #print(comb)

        # append parameters to odict
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

