from time import time, perf_counter
from datetime import timedelta
from math import log10, ceil
import numpy as np

################################################################################
# utility functions
################################################################################

################################################################################
# uncategorized classes
################################################################################
class DumpFullTensor:
    '''
    context manager for printing full numpy arrays
    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('threshold', np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)

class Timer(object):
    ''' 
    Implements context manager protocol
    '''
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        elapsed = round(time() - self.tstart)
        timeobj = timedelta(seconds=elapsed)
        print('Elapsed: ' + str(timeobj))

class PrecisionTimer(object):
    ''' 
    Implements context manager protocol
    '''
    units = ['s','ms','us','ns']

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = perf_counter()

    def __exit__(self, type, value, traceback):
        elapsed = perf_counter() - self.tstart
        if self.name:
            print('[%s]' % self.name,)
        for unit in self.units:
            #print(elapsed)
            if elapsed > 1.0:
                print('Elapsed: {:6.2f} {}'.format(elapsed, unit))
                return
            else:
                elapsed *= 1000

################################################################################
# old stuff
################################################################################

#class BitErrorRate:
#    '''
#    Implement the BER metric
#    '''
#    def __init__(self,K):
#        self.errs = 0
#        self.cnt = 0
#        self.K = K
#
#    def __call__(self, *args, **kwargs):
#        self.update_state(*args, **kwargs)
#
#    def update_state(self, true, pred):
#        K = self.K
#        errs = sum(true[:K] != pred[:K])
#        self.cnt += 1
#        self.errs += errs
#
#    def reset_states(self):
#        self.errs = 0
#        self.cnt = 0
#
#    def result(self):
#        cnt = self.cnt
#        errs = self.errs
#        K = self.K
#
#        total = cnt * K
#        return errs / total
