# implement full chain channel/demod/decoder
# NOTE: plug in a DNN-based demod by implementing
#       the Demodulator interface, see dh_comm/core.py

import dh_comm as dhc
from dh_comm import QAMModulator
from dh_comm import Demodulator
from dh_comm import Transmitter
from dh_comm import Receiver
from dh_comm import Channel

from dh_comm.ldpc.wigig import LdpcEncoder
from dh_comm.ldpc.wigig import LdpcDecoder
from dh_comm.ldpc.wigig import load_parity_matrix

from dh_comm.util import BitErrorRate

import numpy as np
import numpy.random as rnd
from numpy.random import randn
from functools import partial
from sys import exit

from importlib import reload
reload(dhc.core)
reload(dhc)

# param struct
class params:
    eps = 0.01
    # comm.
    ########################
    N_tx = 1
    N_rx = 1
    N_sts = 1
    M = 16 # modulation order
    #nbps =
    snr_db = 20
    #nvar =
    #nstd =
    # decoder
    ########################
    fdir = "./11ad"
    fname = "802.11ad-2012-R1_2.txt"
    fpath = "/".join((fdir, fname))
    #mode = 'bp'
    mode = 'oms'
    PM, PM_spec = load_parity_matrix(fname)
    (P,N) = PM.shape
    K = N - P
    max_iter = 24
    early_term = True
    beta = .15 # min-sum specific
    # channel
    ########################
    ch_type = 'identity'
    noise_type = 'fixed_var'
    #noise_type = 'rand_var'
    noise_dist = 'log_uniform'
    # log-uniform random noise (dB)
    log_u_a = -5
    log_u_b = 25
    # log-normal random noise (dB)
    log_n_mean = 0
    log_n_std = 1
    # simulation
    ########################
    name = 'ldpc_oms'
    snrs_db = np.arange(0,5+eps,.25)
    ber_tgt = 1e-5
    N_bits = round(1/ber_tgt * 100)
    N_cw = N_bits//K + 1 # ceil()
    min_errs = 100
    min_cws = 1000
    # misc
    ########################
    debug_en = False
    ########################
    ## dependent params
    ########################
    # comm.
    ########################
    @property
    def nbps(self):
        return np.log2(self.M).astype(int)
    @property
    def N_syms(self):
        return (self.N / self.nbps).astype(int)
    @property
    def N_raw(self):
        return (self.K / self.nbps).astype(int)
    @property
    def n_var(self):
        return 10**(-self.snr_db/10)
    @property
    def n_std(self):
        return 10**(-self.snr_db/20)

# abbreviate
p = params()

#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=False)

enc = LdpcEncoder(p.PM)
dec = LdpcDecoder()
mod = QAMModulator(p.M)
demod = Demodulator(p)
channel = Channel(p)
ber = BitErrorRate(p.K)

transmit = Transmitter(p, encoder=enc, modulator=mod)
receive = Receiver(p, demodulator=demod, decoder=dec)

for _ in range(10):

    syms_tsr, bits_list = transmit()
    y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
    decoded_bits_list, iter_list = receive(y_tsr, h_tsr)

    # update BER stats
    [ ber(bits, decoded_bits) for bits, decoded_bits in zip(bits_list, decoded_bits_list) ]

    # FIXME: debug
    sts_idx = 0
    bits = bits_list[sts_idx]
    decoded_bits = decoded_bits_list[sts_idx]
    num_iter = iter_list[sts_idx]
    errors = sum(bits !=  decoded_bits)
    print("Num Iter: {:2d} Errors: {:3d}".format(num_iter, errors))


exit()

