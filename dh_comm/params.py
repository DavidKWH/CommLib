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

