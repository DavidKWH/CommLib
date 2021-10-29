# Interface components to Tensorflow
import os
import json
import hashlib
import numpy as np
import tensorflow as tf

from .core import preproc_channel
from .core import cplx2reals
from .core import reals2cplx
from .core import QAMModulator
from .core import Demodulator
from .core import Channel
from .core import Transmitter
from .core import SymbolTransmitter
from .core import bv2dec
from .core import dec2bv
from .core import deintrv_bits
from .one_bit import quantize_o


class MFBasedDataSet:
    '''
    Iterable Wrapper for Tensorflow model training (V2)
    NOTE: Will break existing scripts using V1 dataset
    '''

    def __init__(self, p, mode='none',
                          test=False,
                          one_bit=False):
        assert( hasattr(p,mode) )
        self.p = p
        self.n_iter = p.n_test_steps if test else p.n_train_steps
        self.test = test
        self.mode = mode
        # comm. components
        self.mod = QAMModulator(p.M)
        self.channel = Channel(p, mode)
        self.transmit = SymbolTransmitter(p, modulator=self.mod, training=True)

        self.one_bit = one_bit

        print("MFBasedDataSet: one_bit mode =", self.one_bit)

    def __repr__(self):
        return "Communication dataset iterable"

    # returns an iterator (self)
    def __iter__(self):
        self.cnt = 0 # reset count
        return self

    # returns the next item in the set
    def __next__(self):
        if self.cnt == self.n_iter:
            raise StopIteration()
        self.cnt += 1
        return self.get_training_data()

    # returns a generator (for tf.data.Dataset.from_generator())
    def get_generator(self):
        def ds_gen():
            while True:
                yield self.get_training_data()
        return ds_gen

    def get_training_data(self):
        '''
        transform data for DNN training
        NOTE: compute ideal LLRs only in test mode
        NOTE: always compose input as a sequence of tensors
              let the model decide was how to structure the tensors
              e.g., concat or differing pipeline

        outputs:
            in_seq = [y_mat, h_mat, n_var_mat]
            out_seq = [bit_mat, lambda_mat, sym_vec]
        '''
        p = self.p
        test = self.test
        mod = self.mod
        transmit = self.transmit
        channel = self.channel

        sym_vec = None
        lambda_mat = None
        N = p.batch_size
        N_tx_re = p.N_tx * 2
        N_syms_re = p.sqrt_M
        nbps = p.nbps

        # generate payload bits and symbols
        syms_tsr, bit_tsr = transmit(N, bit_tsr=True)
        # apply channel and noise
        y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
        # analog signal
        r_tsr = y_tsr

        # one bit quantization
        if self.one_bit:
            y_tsr = quantize_o(y_tsr)

        # compute sqrt(2*rho)
        rho_tsr = 1./n_var_tsr
        r2rho_tsr = np.sqrt(2.*rho_tsr)

        # flatten dimensions i>1
        y_mat = y_tsr.reshape(N,-1)
        r_mat = r_tsr.reshape(N,-1)
        sym_mat = syms_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        r2rho_mat = r2rho_tsr.reshape(N,-1)

        # convert to reals
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        r_mat = cplx2reals(r_mat)

        # generate bits and symbol id
        bit_tsr = deintrv_bits(bit_tsr)
        sym_id_mat = bv2dec(bit_tsr)
        sym_mat = cplx2reals(sym_mat)

        # convert to tensors
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        r_mat = tf.convert_to_tensor(r_mat, dtype=tf.float32)
        sym_mat = tf.convert_to_tensor(sym_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        r2rho_mat = tf.convert_to_tensor(r2rho_mat, dtype=tf.float32)
        #bit_mat = tf.convert_to_tensor(bit_mat, dtype=tf.bool)
        sym_id_mat = tf.convert_to_tensor(sym_id_mat, dtype=tf.int32)
        # zero input to network
        shape = (N, N_tx_re, N_syms_re)
        log_qi_mat = tf.zeros(shape)

        # output processing
        in_seq = [y_mat, h_tsr, r2rho_mat, n_var_mat]
        out_seq = [None, sym_id_mat, sym_mat, r_mat]
        aux_out = [log_qi_mat]

        return in_seq, out_seq, aux_out

