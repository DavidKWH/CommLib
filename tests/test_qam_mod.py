# test qam modulation

import numpy as np
import numpy.random as rnd
import dh_comm as dhc

import pytest

@pytest.fixture
def qam_modulator():
    M = 16
    mod = dhc.QAMModulator(M)
    return mod

def test_16qam_const(qam_modulator):
    mod = qam_modulator
    mod.plot_constellation()

def test_16qam_mod(qam_modulator):
    mod = qam_modulator
    nbps = mod.nbps
    N = 10
    bit_mat = rnd.randint(2, size=(N, nbps))
    bit_vec = bit_mat.reshape(-1)
    syms = mod.map(bit_vec)
    print(syms)

if __name__ == '__main__':
    test_16qam_const()
