# test conversion from binary array to decimal
# via pad->packbits

import numpy as np
import numpy.random as rnd
import dh_comm as dhc

def test_bin2dec():

    bit_mat = rnd.randint(2, size=(5,3))
    dec_vec = dhc.bv2dec(bit_mat)

    print()
    print(bit_mat)
    print(dec_vec)

def test_dec2bin():
    dec_vec = rnd.randint(32, size=(5,1))
    bit_mat = dhc.dec2bv(dec_vec)

    print()
    print(dec_vec)
    print(bit_mat)

