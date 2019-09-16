from time import time, perf_counter
from datetime import timedelta
from math import log10, ceil

################################################################################
# utility functions
################################################################################

#def get_key(string, delimiter=',', index=0):
#    return string.split(delimiter)[index]

################################################################################
# uncategorized classes
################################################################################

class BitErrorRate:
    '''
    Implement the BER metric
    '''
    def __init__(self,K):
        self.errs = 0
        self.cnt = 0
        self.K = K

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)

    def update_state(self, true, pred):
        K = self.K
        errs = sum(true[:K] != pred[:K])
        self.cnt += 1
        self.errs += errs

    def reset_states(self):
        self.errs = 0
        self.cnt = 0

    def result(self):
        cnt = self.cnt
        errs = self.errs
        K = self.K

        total = cnt * K
        return errs / total

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


