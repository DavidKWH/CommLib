# test conversion from binary array to decimal
# via pad->packbits

import numpy as np
import numpy.random as rnd
import dh_comm as dhc

def test_crandn():

    noise = dhc.crandn(2,4)
    print()
    print(noise)



