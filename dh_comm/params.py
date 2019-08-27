# implements recursive parameter structure

class RecursiveParams:
    '''
    Recursive parameter structure with attribute like access

    A way to make structure access easier (matlab like)
    e.g.        p.rx.demod.param1 = 10
    instead of  p['rx']['demod']['param1'] = 10

    Can be json serialized.  See asdict() below.
    '''
    def __init__(self, **kwargs):
        # perform recursive conversion to RP
        for key, val in kwargs.items():
            #print('item ({}: {})'.format(key,val))
            if type(val) is not dict:
                self.__dict__.update({key: val})
            else:
                # construct lower level RP
                self.__dict__.update({key: RecursiveParams(**val)})

    def __repr__(self):
        #keys = sorted(self.__dict__)
        keys = self.__dict__.keys()
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        #return "{}({})".format(type(self).__name__, ", ".join(items))
        return "{}({})".format("RP", ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def asdict(self):
        '''
        used for json serialization
        e.g. p = RecursiveParams()
             json.dumps(p.asdict())
        '''
        # recursively reconstruct dict from RP
        new_dict = {}
        for key, val in self.__dict__.items():
            # determine value type
            if callable(val):
                # function or class object
                new_dict.update({key:val.__name__})
            elif type(val) is type(self):
                # another RP instance
                new_dict.update({key: val.asdict()})
            elif hasattr(val,'__dict__') or hasattr(val,'__slot__'):
                # instance of non-builtin type
                new_dict.update({key: type(val).__name__ + '()'})
            else:
                # instance of builtin type (numeric or string)
                new_dict.update({key: val})
        return new_dict

