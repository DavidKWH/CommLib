import sys
import re
from .exceptions import ArgumentError
from .exceptions import WriteError

################################################################################
# utility functions
################################################################################

def get_key(string, delimiter=',', index=0):
    return string.split(delimiter)[index]

def has_key(string, key, delimiter=','):
    return key in string.split(delimiter)

# valid option strings
valid_opt = re.compile('--((?:\w+)(?:[.]\w+)*)$')
# string option values
str_value = re.compile('str[(]([\w./]+)[)]$')
'''
NOTE: special treatment of strings is required
      to support valid python expressions as option values.
      String option values needs to be passed
      as 'str(outdir/foo)'
'''
# invalid arguments
invalid_arg = re.compile('(-|--)\w*')
# string literals
#str_literal = re.compile('[a-zA-Z_][\w]*')
'''
OLD:  This severely restrict the type of expressions
      allowed on the RHS.  We can execute any valid
      python expression within the current scope.
'''

def parse_opts(argv=None):

    if argv is None:
        argv = sys.argv

    pos_args = []
    opt_args = {}

    print('command line args')
    print(argv)

    args = iter(argv[1:])

    for arg in args:
        match = valid_opt.match(arg)
        if match:
            try:
                opt = match.group(1)
                val = next(args)
                # special handling for string values
                str_match = str_value.match(val)
                if str_match:
                    str_val = str_match.group(1)
                    print(str_val)
                    opt_args[opt] = "r'{}'".format(str_val)
                else:
                    opt_args[opt] = val
            except StopIteration:
                raise ArgumentError(f'RP: missing option value: {arg}')
        elif invalid_arg.match(arg):
            raise ArgumentError(f'RP: Invalid argument: {arg}')
        else:
            pos_args.append(arg)

    #print('positional args')
    #print(pos_args)
    print('optional args')
    print(opt_args)

    return opt_args

################################################################################
# implements recursive parameter structure
################################################################################
import numpy as np

class RecursiveParams:
    '''
    Recursive parameter structure with attribute like access

    A way to make structure access easier (matlab like)
    e.g.        p.rx.demod.param1 = 10
    instead of  p['rx']['demod']['param1'] = 10

    Automatic conversion of lists to arrays

    Can be json serialized.  See as_serializable() below.
    '''
    def __init__(self, **kwargs):
        # perform recursive conversion to RP
        for key, val in kwargs.items():
            #print('item ({}: {})'.format(key,val))
            if type(val) is dict:
                # construct lower level RP
                self.__dict__.update({key: RecursiveParams(**val)})
            elif type(val) is list:
                # list objects
                self.__dict__.update({key: np.array(val)})
            else:
                # anything else
                self.__dict__.update({key: val})

    def __repr__(self):
        #keys = sorted(self.__dict__)
        keys = self.__dict__.keys()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        #return "{}({})".format(type(self).__name__, ", ".join(items))
        return "{}({})".format("RP", ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def as_dict(self, exclude=()):
        '''
        return dictionary, with exclude filter
        intended for passing into function as kwargs
        NOTE: does not recurse
        '''
        new_dict = {}
        for key, val in self.__dict__.items():
            if key not in exclude:
                new_dict.update({key: val})
        return new_dict

    def as_serializable(self):
        '''
        used for json serialization
        e.g. p = RecursiveParams()
             json.dumps(p.as_serializable())
        '''
        # recursively reconstruct dict from RP
        new_dict = {}
        for key, val in self.__dict__.items():
            # determine value type
            # NOTE: forms a complicated triage (yes/no questions) procedure
            #       order is important in this case
            if callable(val) and hasattr(val, '__name__'):
                # function or class object
                new_dict.update({key:val.__name__})
            elif type(val) is RecursiveParams:
                # another RP instance
                new_dict.update({key: val.as_serializable()})
            elif np.isscalar(val) and isinstance(val, (np.number,np.ndarray) ):
                # numpy scalar (any integer/float or zero-dimension array)
                new_dict.update({key: val.item()})
            elif isinstance(val, np.ndarray):
                # numpy array (automatic conversion to python types)
                new_dict.update({key: val.tolist()})
            elif hasattr(val,'__dict__') or hasattr(val,'__slot__'):
                # instance of non-builtin type
                new_dict.update({key: type(val).__name__ + '()'})
            else:
                # instance of builtin type (numeric or string)
                new_dict.update({key: val})
        return new_dict

    def set_rparam(self, relpath, val, create_new=False):
        '''
        set recursive param to value

        format:
            'param1.param2.param3': 'value'
        '''
        fullpath = '.'.join(('self',relpath))

        # perform a whole bunch of sanity checks...
        try:
            obj = eval(fullpath)
            # if reference exist...
            if type(obj) is RecursiveParams:
                raise WriteError("RP: Cannot override param structure")
        except AttributeError as err:
            # re-raise exception, if attribute creation not permitted
            if not create_new:
                raise WriteError("RP: Attribute creation not permitted. " +
                                 "If this is intended, set create_new=True")

        # set param to val
        stmt = f'{fullpath} = {val}'
        print('eval:', stmt)
        exec(stmt)

    def set_rparams(self, pdict, create_new=False):
        '''
        override params defined in pdict
        '''
        for key, val in pdict.items():
            self.set_rparam(key, val, create_new)

    def parse_opts(self, argv=None):
        '''
        parse all options in the command line
        '''
        return parse_opts(argv)

    def process_opts(self, argv=None):
        '''
        override params defined in command line
        '''
        pdict = self.parse_opts(argv)
        self.set_rparams(pdict)

