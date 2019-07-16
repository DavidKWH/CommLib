# test comm channel data generation

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import dh_comm as dhc
import dh_comm.plotting as clt

import pytest

# param struct
class def_params:
    # comm.
    N_tx = 2
    N_rx = 2
    N_sts = 2
    M = 16 # modulation order
    nbps = np.log2(M).astype(int)
    snr_db = 20
    n_var = 10**(-snr_db/10)
    n_std = np.sqrt(n_var)
    ch_type = 'identity'

@pytest.fixture
def comm_data_src():
    M = 16
    p = def_params
    comm = dhc.CommDataSource(p)
    return comm

def test_2x2_16qam_generation(comm_data_src):
    N = 1000
    comm = comm_data_src
    bit_tsr, h_tsr, y_tsr = comm.gen_channel_output(N)
    y = y_tsr.reshape(N, -1)

    plt.figure()
    plt.subplot(1,2,1)
    clt.scatterc(y, s=10)
    plt.title("stream 1")
    plt.axis('equal')
    plt.subplot(1,2,2)
    clt.scatterc(y, s=10)
    plt.title("stream 2")
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    test_16qam_const()
