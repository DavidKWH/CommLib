import sys
import re
import json
import os
import numpy as np
from .parser import parse_opts
from .exceptions import WriteError

################################################################################
# utility functions
################################################################################

def get_key(string, delimiter=',', index=0):
    return string.split(delimiter)[index]

def has_key(string, key, delimiter=','):
    return key in string.split(delimiter)

################################################################################
# implements recursive parameter structure
################################################################################

class RecursiveParams:
    '''
    Recursive parameter structure with attribute like access

    The RP was originally conceived as a way to make structure
    access easier (matlab like).

    e.g.        p.rx.demod.param1 = 10
    instead of  p['rx']['demod']['param1'] = 10

    RPs can be json serialized.  The types supported are strings,
    scalars and lists.  Thus, for a training run, one can save
    all parameters into a file and replay the exact scenario again
    by loading it from file.  See as_serializable() below.

    An entry can also be overridden using an intermediate format
    (usually specified in a dictionary):

    { 'p.n_layers'   : 5,
      'p.adam.beta_1': 0.9,
      'p.adam.beta_2': 0.999, }

    This allows for quick experimentation and hyperparameter tuning.

    '''
    def convert_to_rp(self, d):
        # perform recursive conversion to RP
        for key, val in d.items():
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

    def __init__(self, **kwargs):
        self.convert_to_rp(kwargs)

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

    def as_serializable(self, permissive=False):
        '''
        used for json serialization
        e.g. p = RecursiveParams()
             p.param = val
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
                raise ValueError('Cannot serialize function or class object')
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
                raise ValueError('Cannot serialize instance of a non-builtin type')
            else:
                # instance of builtin type (numeric or string)
                new_dict.update({key: val})
        return new_dict

    def _serialize_permissive(self):
        '''
        used for json serialization (old, very permissive)
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

    def save_rparams(self, fullpath):
        ''' save RP to file in json format '''
        fname, fext = os.path.splitext(fullpath)
        assert fext in ('', '.json')
        if not fext: fullpath = fname + '.json'
        with open(fullpath, 'w') as fp:
            print('writing params to file:', fullpath)
            json.dump(self.as_serializable(), fp, indent=4)


    def set_rparam(self, fullpath, val, create_new=False, override_pdict=False):
        ''' set recursive param to value '''
        # make instance accessible from p
        p = self
        #fullpath = '.'.join(('self',relpath))

        # perform a whole bunch of sanity checks...
        try:
            obj = eval(fullpath)
            # writing to existing RP
            if type(obj) is RecursiveParams:
                if not override_pdict:
                    raise WriteError("Cannot override param structure.  " +
                                     "If this is intended, set override_pdict=True")
        except AttributeError as err:
            # re-raise exception, if attribute creation not permitted
            if not create_new:
                raise WriteError("Attribute creation not permitted.  " +
                                 "If this is intended, set create_new=True")

        # set param to val
        stmt = f'{fullpath} = {val}'
        #print('eval:', stmt)
        exec(stmt)

    def set_rparams(self, pdict, **kwargs):
        ''' override params defined in pdict '''
        for key, val in pdict.items():
            self.set_rparam(key, val, **kwargs)

    def process_opts(self):
        ''' options processing '''
        print('parsing options from command line')
        pdict, odict = parse_opts()

        # special handling for param clobbering
        if 'pfile' in odict:
            pfile = odict.pop('pfile')
            print('reading from params file:')
            with open(pfile, 'r') as f:
                d = json.load(f)
            #print(f'read in dict: {d}')
            # clobber existing contents
            #print('clobbering RP...')
            self.__dict__.clear()
            # use params from file (in d)
            self.convert_to_rp(d)
            return

        # process options file
        if 'ofile' in odict:
            ofile = odict.pop('ofile')
            if ofile == 'stdin':
                print('reading options from stdin')
                # clobber pdict
                pdict = json.load(sys.stdin)
                #print(f'stdin dict: {pdict}')
                print(f'{len(pdict)} params received')
            else:
                print('reading from options file')
                # clobber pdict
                with open(ofile, 'r') as f:
                    pdict = json.load(f)
                #print(f'options dict: {pdict}')
                print(f'{len(pdict)} params received')

        # pdict may survive if not (pfile or ofile)
        # in that case, dict may hold options
        # from the command line

        # update params from pdict
        # pass in keyword args in odict
        self.set_rparams(pdict, **odict)

