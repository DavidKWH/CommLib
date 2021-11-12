# perform BER sweep on model
# NOTE: one-bit specific tests
# NOTE: plug in a DNN-based demod by implementing
#       the Demodulator interface, see comm_ai/core.py

import comm_ai as comm
from comm_ai import QAMModulator
#from comm_ai import Demodulator
from comm_ai import SymbolEstimator
from comm_ai import GaussianDemapper
#from comm_ai import Transmitter
from comm_ai import SymbolTransmitter
from comm_ai import SymbolDetector
from comm_ai import Receiver
from comm_ai import Channel

#from comm_ai.one_bit import BussgangEstimator
#from comm_ai.one_bit import OneBitMLDemod
#from comm_ai.one_bit import OneBitMLDetector
#from comm_ai.tensorflow import DnnDemod
#from comm_ai.tensorflow import DnnDetector
from comm_ai.tensorflow import DnnEstimator

from comm_ai.ldpc.wigig import LdpcEncoder
from comm_ai.ldpc.wigig import LdpcDecoder
from comm_ai.ldpc.wigig import load_parity_matrix

#from comm_ai.util import BitErrorRate
#from comm_ai.util import SymbolErrorRate
from comm_ai.util import RawBitErrorRate
from comm_ai.util import NMSE
from comm_ai.util import Timer

from comm_ai.params import RecursiveParams as Params
from comm_ai.params import get_key
from comm_ai.params import has_key

import tensorflow as tf

import numpy as np
#import numpy.random as rnd
#from numpy.random import randn
from functools import partial
from sys import exit
import os

from importlib import reload
reload(comm.core)
reload(comm)

################################################################################
# BER sim controller
################################################################################
#class BerSimControl:
#    '''
#    controls SNR points and termination criteria
#    NOTE: can adapt coarse/fine SNR points based on
#          current progress
#    '''
#    def __init__(self, p):
#        self.p = p
#        self.iter = iter(p.snrs_db)
#
#    def next_snr(self):
#        iter_obj = self.iter
#        return next(iter_obj)
#
#    def get_term_flag(self, cum_err_vec, cw_cnt):
#        p = self.p
#        return (cum_errs >= p.min_errs and cw_cnt >= p.min_cws);

################################################################################
# Define simulation specific features
################################################################################
class SNRSweepParams(Params):
    ''' specialize parameter structure '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    ### SNR dependent params
    ########################
    @property
    def nbps(self):
        return np.log2(self.M).astype(int)
    @nbps.setter
    def nbps(self, x):
        pass

    @property
    def n_var(self):
        return 10**(-self.snr_db/10)
    @n_var.setter
    def n_var(self, x):
        pass

    @property
    def n_std(self):
        return 10**(-self.snr_db/20)
    @n_std.setter
    def n_std(self, x):
        pass

    ### decoder dependent params
    ########################
#    @property
#    def N_syms(self):
#        return (self.dec.N / self.nbps).astype(int)
#    @property
#    def N_raw(self):
#        return (self.dec.K / self.nbps).astype(int)

#    @property
#    def snrs_db(self):
#        return self.snr_ranges[self.dec.code_rate]

################################################################################
# Parameters
################################################################################
p = SNRSweepParams()
'''
param structure
NOTE: fixed param specification!
NOTE: it is safer to specify the name
      of the initializer/activation_fn
      than the actual callable.
'''

### comm. related parameters
#########################################
# one-bit receiver
#p.one_bit = False
p.one_bit = True

# general
p.N_tx = 4
#p.N_rx = 6
p.N_rx = 32
p.N_sts = p.N_tx
p.M = 4 # modulation order
#p.M = 16 # modulation order
p.nbps = np.log2(p.M).astype(int)
p.nbpsv = p.N_sts * p.nbps
#p.snr_db = 20
p.snr_db = 12
p.n_var = 10**(-p.snr_db/10)
p.n_std = np.sqrt(p.n_var)
#p.ch_type = 'identity'
#p.ch_type = 'rayleigh'
#p.ch_type = 'sv_dist'
#p.noise_type = 'fixed_var'
#p.noise_type = 'rand_var'
p.noise_dist = 'log_uniform'
# log-uniform random noise (dB)
p.log_u_a = 0
p.log_u_b = 25
# log-normal random noise (dB)
p.log_n_mean = 0
p.log_n_std = 1
# uniform sv distribution
p.u_a = 0.5
p.u_b = 3.0
# training modes
p.train0 = Params()
p.train0.ch_type = 'rayleigh'
p.train0.noise_type = 'fixed_var'
p.train1 = Params()
p.train1.ch_type = 'sv_dist'
p.train1.noise_type = 'fixed_var'
p.train2 = Params()
p.train2.ch_type = 'rayleigh'
p.train2.noise_type = 'rand_var'
p.train3 = Params()
p.train3.ch_type = 'rayleigh,batch_fixed'
p.train3.noise_type = 'rand_var'
### select training mode
p.train = p.train0
# test modes
p.test0 = Params()
p.test0.ch_type = 'rayleigh'
p.test0.noise_type = 'fixed_var'
### select test mode
p.test = p.test0

# default modes
p.default0 = Params()
p.default0.ch_type = 'rayleigh'
#p.default0.ch_type = 'identity'
#p.default0.ch_type = 'diagonal'
p.default0.noise_type = 'fixed_var'
p.default0.normalize = False
p.default1 = Params()
p.default1.ch_type = 'fixed'
p.default1.ch_file = 'golden_channel.npy'
p.default1.noise_type = 'fixed_var'
p.default = p.default0

### decoder parameters
########################
p.dec = Params()
p.dec.fname = "802.11ad-2012-R1_2.txt"
p.dec.code_rate = os.path.splitext(p.dec.fname)[0].split('-')[-1]
PM, PM_spec = load_parity_matrix(p.dec.fname)
p.dec.PM = PM
(P,N) = PM.shape
p.dec.P = P
p.dec.N = N
p.dec.K = N - P
p.pmax_iter = 24
p.early_term = True
p.beta = .15 # min-sum specific

### simulation parameters
#########################################
p.bname = 'ff_classifier'
p.sname = 'gaussian_ch'
# used for saving data
#p.label = '1bit_densenet_16qam_rand_h_rand_snr_sym_est_augment'
p.label = '1bit_1block_densenet_norm_out_rand_h_rand_snr_sym_est_qpsk'
p.eps = 0.01
#snrs_db = np.arange(0,3+eps,.25)
#snrs_db = [0]
#p.ber_target = 1e-5
#p.N_bits = round(1/p.ber_target * 100)
#p.N_cw = p.N_bits//p.dec.K + 1 # ceil()
#N_cw = 100
#min_errs = 100
p.N = 1000 # batch size
p.N_batches = 5000
p.N_syms = p.N * p.N_batches
p.min_errs = 500
#p.min_batches = 5000
p.min_batches = 500
#p.min_cws = 1000
#p.min_cws = 5000
### sweep ranges (code rate dependent)
snr_ranges = {
    'R1_2': np.arange(8,15+p.eps,1),
    'R5_8': np.arange(8,14,1),
    'R3_4': np.arange(8,14,1),
    'R13_16': np.arange(8,14,1),
    }
p.snr_ranges = snr_ranges

### overrides for linear estimator verification
#p.snrs_db = [10]
#p.snrs_db = np.arange(6,9+p.eps,1)
#p.snrs_db = np.arange(-1,4+p.eps,1)
#p.snrs_db = np.arange(-1,10+p.eps,1)
#p.snrs_db = np.arange(0,25+p.eps,1)
#p.snrs_db = np.arange(-10,20+p.eps,5)
p.snrs_db = np.arange(-10,30+p.eps,5)
#p.snrs_db = [20]

### model parameters
#########################################
p.m_dir = 'models'
#p.m_file = "neural_rx_train_12db_snr_lr_schedule.model/"
#p.m_file = "resnet_v2_7_layers_rand_snr_8ea12488400e7e92.model"
#p.m_file = "resnet_v2_7_layers_rand_snr_rand_h_baseline.model"
#p.m_file = "resnet_v1_7_layers_rand_snr_rand_h_6e2b7fe4341ce5e0.model"
#p.m_file = "hypernet_3_layers_256_node_rand_snr_rand_h_62bdd5a614639f53.model"
#p.m_file = "densenet_3_layers_rand_snr_rand_h_8b8b9d160c18a8ef.model"
#p.m_file = "densenet_3_layers_rand_snr_5-20_rand_h_6907ff7eb2f4537b.model"
#p.m_file = "densenet_3_layers_rand_snr_rand_h_nsubl_24_c1dc5b19aa041cbf.model"
#p.m_file = "densenet_4_layers_rand_snr_rand_h_77dbf0239fb12926.model"
#p.m_file = "densenet_3_layers_rand_snr_rand_h_x0_1024_ed795159b361d269.model"
#p.m_file = "densenet_3_layers_rand_snr_rand_h_sgd_optimizer_1d739eabf5be7a93.model" # SGD
#p.m_file = "densenet_3_layers_rand_snr_rand_h_l2_reg_e4007d29357fd501.model" # SGD + L2
#p.m_file = "densenet_symbol_est_3_layers_rand_snr_rand_h_8d86fb4b9e6c6783.model"
#p.m_file = "densenet_symbol_est_3_layers_rand_snr_rand_h_adam_c9eb071a3f6978f0.model"
#p.m_file = "densenet_symbol_est_1_layer_rand_snr_rand_h_adam_641773a7eaeedb0f.model"
#p.m_file = "densenet_symbol_est_augment_3_layers_rand_snr_rand_h_adam_bfff19734145878c.model"
#p.m_file = "densenet_symbol_est_augment_32x4_qpsk_3_blocks_rand_snr_rand_h_adam_0650a5d415d3d416.model"
#p.m_file = "densenet_symbol_est_32x4_qpsk_1_layer_rand_snr_rand_h_adam_082fe68056868e86.model"
p.m_file = "densenet_symbol_est_norm_out_32x4_qpsk_1_layer_rand_snr_rand_h_adam_9ab1b30354a75af8.model"
#p.m_file = "N/A"
p.debug_en = False


print()
print("BER Sim configuration")
print("=============================")
print(" Communications ")
print("-----------------------------")
print("code rate       =", p.dec.code_rate)
print("channel type    =", p.default.ch_type)
print("DNN model       =", p.m_file)
print("bname           =", p.bname)
print("label           =", p.label)
print("-----------------------------")
print("1-bit           =", p.one_bit)
print("N_syms          =", p.N_syms)
print("N_tx            =", p.N_tx)
print("N_rx            =", p.N_rx)
print("-----------------------------")

#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=False)

#enc = LdpcEncoder(p.dec.PM)
mod = QAMModulator(p.M)
#dnn_demod = DnnDemod(p)
channel = Channel(p, mode='default')
#sim_ctrl = BerSimControl(p)

#ideal_ber = BitErrorRate(p.dec.K)
#model_ber = BitErrorRate(p.dec.K)
ideal_ber = RawBitErrorRate((p.N,p.nbpsv))
model_ber = RawBitErrorRate((p.N,p.nbpsv))
model_mse = NMSE((p.N,p.N_tx))

metrics = [ideal_ber, model_ber, model_mse]

#transmit = Transmitter(p, encoder=None, modulator=mod)
transmit = SymbolTransmitter(p, modulator=mod)

# Ideal demodulator
#demod = Demodulator(p)
#demod = OneBitMLDemod(p)
#dec = LdpcDecoder(p.dec.PM)
#model_receive = Receiver(p, demodulator=demod, decoder=None)
#model_detect = SymbolDetector(p, modulator=mod)
#ml_detect = OneBitMLDetector(p, modulator=mod)

# Linear estimator + demapper
#estimator = SymbolEstimator(p, mode='mmse')
#estimator = SymbolEstimator(p, mode='zf')
#estimator = BussgangEstimator(p, mode='bmmse')
#demapper = GaussianDemapper(p)
#model_dec = LdpcDecoder(p.dec.PM)
#model_receive = Receiver(p, 
#        estimator=estimator,
#        demapper=demapper,
#        decoder=model_dec)
# DNN receiver
#dnn_receive = Receiver(p, demodulator=dnn_demod, decoder=dec)
# DNN detector
dnn_detect = DnnEstimator(p)

# select which model to use
model_detect = dnn_detect

################################################################################
# One bit quantizer
################################################################################
def quantize_o(x_cpx):
    '''perform component wise quantization'''
    return np.sign(x_cpx.real) + 1j* np.sign(x_cpx.imag)

################################################################################
# Start simulation
################################################################################
bers_model = np.zeros(len(p.snrs_db))
mse_vec_model = np.zeros(len(p.snrs_db))
for si,snr_db in enumerate(p.snrs_db):
    # set parameters
    p.snr_db = snr_db
    # reset metrics
    for metric in metrics:
        metric.reset_states()

    with Timer():
        for f_num in range(p.N_batches):
            batch_cnt = f_num + 1
            # transmit should return complex symbols
            # as well as bit representation
            syms_tsr, bits_mat = transmit()
            y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)

            # add 1-bit quantization
            if p.one_bit:
                y_tsr = quantize_o(y_tsr)

            # receiver should return bit representation
            # of symbol and perhaps other information (e.g. xhat)
            model_bits_mat, model_syms_mat, model_syms_est = model_detect(y_tsr, h_tsr, n_var_tsr)

            # update BER stats (via bit representation)
            model_ber(bits_mat, model_bits_mat)
            syms_mat = np.squeeze(syms_tsr)
            model_mse(syms_mat, model_syms_est)

            # early termination
            if (model_ber.errs >= p.min_errs and
                batch_cnt >= p.min_batches):
                break


    # end of snr run
    bers_model[si] = model_ber.result()
    mse_vec_model[si] = model_mse.result()
    template = '''SNR {: .2f} dB, model BER {:.3e}, MSE {:.3e}, batches = {}'''
    print( template.format(snr_db,
#                           ideal_ber.result(),
                           model_ber.result(),
                           model_mse.result(),
                           batch_cnt)
          )

    # terminate sweep
    if model_ber.errs == 0:
        break

################################################################################
# Save output to matfile
################################################################################
import scipy.io as sio

data = {}
data['name'] = '_'.join((p.bname,p.sname))
data['label'] = p.label
data['code_rate'] = p.dec.code_rate
data['bers_model'] = bers_model
data['mse_vec_model'] = mse_vec_model
data['snrs_db'] = p.snrs_db

out_dir = 'data'
bname  = 'sweep_snrs_uncoded_ber_mse_est'
fname = '_'.join((bname,p.label))
fname = '/'.join((out_dir,fname))
npzname = fname + '.npz'
# npy format
np.savez(npzname, **data)
print("saving data to file :", npzname)
# matfile format
#matname = fname + '.mat'
#sio.savemat(matname, data)
#print("saving data to file :", matname)

