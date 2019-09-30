'''Parser to process command line arguments

There are three methods to update default parameters defined in
a training script.
'''
import re
import sys
from sys import exit
from .exceptions import ArgumentError

################################################################################
# regex patterns
################################################################################
# valid option strings
short_opt_group = '-([a-zA-Z])'
#long_opt_group = '--([a-zA-Z]+(?:-[a-zA-Z]+)*)'
long_opt_group = '--([a-zA-Z_]\w*)'
param_group = '--(p[.](?:\w+)(?:[.]\w+)*)'
valid_opt_regex = '(?:' + short_opt_group + '|' + long_opt_group + '|' + param_group + ')$'
valid_opt = re.compile(valid_opt_regex)
# python expressions
python_expr = re.compile('py[(](.+)[)]$')
# invalid arguments
#invalid_arg = re.compile('(-|--)\w*')
# string literals
str_literal = re.compile('[a-zA-Z_][\w./]*')
# scalar literals
# [-] ( integer | lfloat | rfloat ) [ exponent ]
# non-capturing version of
# num_regex = '-?(\d+([.]\d*)?|[.]\d+)(e-?\d+)?'
num_regex = '-?(?:\d+(?:[.]\d*)?|[.]\d+)(?:e-?\d+)?'
num_literal = re.compile(num_regex+'$')
# (list|tuples) of scalars
num_group = f"({num_regex})"
num_cont = f"(?:\s*,\s*{num_group})*"
num_list_regex = '\[\s*' + num_group + num_cont + '\s*\]$'
num_list = re.compile(num_list_regex)
num_tuple_regex = '\(\s*' + num_group + num_cont + '\s*\)$'
num_tuple = re.compile(num_tuple_regex)

################################################################################
# helper functions
################################################################################
def reduce_or(iterable):
    '''pythonic reduce or (does not behave like logical or)'''
    return next((item for item in iterable if item is not None), None)

################################################################################
# parameter input control
################################################################################
def parse_opts(argv=None):
    '''
    parse all options in the command line
    returns options dict
    '''
    if argv is None:
        argv = sys.argv

    read_pfile = False
    pos_args = []
    opt_args = {}
    p_args = {}

    #print(f'command line args: {argv}')

    # define iterable
    args = iter(argv[1:])

    for arg in args:
        match = valid_opt.match(arg)
        if match:
            opt = reduce_or(match.groups())

            ### options with no values
            if opt in ('h','help'):
                exe_name = argv[0].split('/')[-1]
                print(f'Usage: {exe_name} [ options | params_options ]\n' +
                        'see https://github.com/DavidKWH/CommLib/comm_ai/parser.py')
                exit()
            # control behavior of RP
            if opt in ('create_new','override_pdict'):
                opt_args[opt] = True
                continue

            try:
                val = next(args)

                # read from params file
                if opt == 'params_from':
                    opt_args['pfile'] = val
                    continue
                # read from options file
                if opt == 'options_from':
                    opt_args['ofile'] = val
                    continue
                # python expressions
                expr_match = python_expr.match(val)
                if expr_match:
                    expr = expr_match.group(1)
                    p_args[opt] = expr
                    continue
                # special handling for string values
                str_match = str_literal.match(val)
                if str_match:
                    str_val = str_match.group(0)
                    p_args[opt] = "r'{}'".format(str_val)
                    continue
                # allowable value types
                if (num_literal.match(val) or
                    num_list.match(val) or
                    num_tuple.match(val)
                    ):
                    p_args[opt] = val
                    continue

                # do not permit anything else...
                raise ValueError(f'unrecognized value type: {val}')

            except StopIteration:
                raise ArgumentError(f':missing option value: {arg}')
        # invalid arguments
        #elif invalid_arg.match(arg):
        #    raise ArgumentError(f'RP: Invalid argument: {arg}')
        # assume positional argument
        else:
            pos_args.append(arg)

    #print('positional args: {pos_args}')
    #print(f'optional args: {opt_args}')
    print(f'{len(p_args)} params received')
    print(f'{len(opt_args)} options received')

    return p_args, opt_args

