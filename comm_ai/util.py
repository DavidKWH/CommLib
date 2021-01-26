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
# (Old) used in sweep_coded_ber_perf.py
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

class SymbolErrorRate:
    '''
    Implement the SER metric
    '''
    def __init__(self, shape):
        # shape = [N_syms, nbpsv]
        self.errs = 0
        self.cnt = 0
        self.shape = shape
        self.N = shape[0]

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)

    def update_state(self, true, pred):
        assert np.array_equal(pred.shape, self.shape)
        bit_errs = np.sum(true != pred, axis=1)
        sym_errs = sum(bit_errs > 0)
        self.cnt += 1
        self.errs += sym_errs

    def reset_states(self):
        self.errs = 0
        self.cnt = 0

    def result(self):
        cnt = self.cnt
        errs = self.errs
        N = self.N

        total = cnt * N
        return errs / total


class RawBitErrorRate:
    '''
    Implement the BER metric (Uncoded)
    '''
    def __init__(self, shape):
        # shape = [N_syms, nbpsv]
        self.errs = 0
        self.cnt = 0
        self.shape = shape
        self.N = shape[0] * shape[1]

    def __call__(self, *args, **kwargs):
        self.update_state(*args, **kwargs)

    def update_state(self, true, pred):
        assert np.array_equal(pred.shape, self.shape)
        bit_errs = np.sum(true != pred)
        self.cnt += 1
        self.errs += bit_errs

    def reset_states(self):
        self.errs = 0
        self.cnt = 0

    def result(self):
        cnt = self.cnt
        errs = self.errs
        N = self.N

        total = cnt * N
        return errs / total

