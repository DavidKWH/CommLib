# Interface components to Tensorflow
from . import core as dhc
from .core import QAMModulator
from .core import Demodulator
from .core import Channel
from .core import Transmitter

import numpy as np
import tensorflow as tf

class CommDataSet:
    '''
    Iterable Wrapper for Tensorflow model training
    '''

    def __init__(self, p, test=False):
        self.p = p
        self.n_iter = p.n_test_steps if test else p.n_train_steps
        self.test = test
        # comm. components
        self.mod = QAMModulator(p.M)
        self.demod = Demodulator(p)
        self.channel = Channel(p)
        self.transmit = Transmitter(p, modulator=self.mod, training=True)

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

    def get_training_data(self):
        '''
        transform data for DNN training
        NOTE: compute ideal LLRs only in test mode
        NOTE: always compose input as a sequence of tensors
              let the model decide was how to structure the tensors
              e.g., concat or differing pipeline
        '''
        p = self.p
        test = self.test
        mod = self.mod
        transmit = self.transmit
        channel = self.channel
        demod = self.demod

        lambda_mat = None
        N = p.batch_size

        # generate payload bits and symbols
        syms_tsr, bits_tsr = transmit(N)
        # apply channel and noise
        y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
        # generate LLRs (test mode)
        if test: lambda_mat = demod(y_tsr, h_tsr, n_var_tsr)

        # flatten dimensions i>1
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        out_mat = bits_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        # convert to reals
        h_mat = dhc.cplx2reals(h_mat)
        y_mat = dhc.cplx2reals(y_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        out_mat = tf.convert_to_tensor(out_mat, dtype=tf.bool)
        if test: lambda_mat = tf.convert_to_tensor(
                               lambda_mat, dtype=tf.float32)
        # output processing
        in_seq = [y_mat, h_mat, n_var_mat]
        return in_seq, out_mat, lambda_mat

    def get_const_input(self):
        """ transform constellation for DNN training """
        mod = self.mod
        syms, _ = mod.get_const()
        return np.r_[syms.real, syms.imag]


class DnnDemod:
    '''
    Encapsulate trained Tensorflow model
    Implements Demodulator interface
    '''

    def __init__(self, p):
        self.p = p
        print('loading model from file:', p.m_file)
        self.model = tf.saved_model.load(p.m_file)

    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        model = self.model
        N = y_tsr.shape[0]

        # flatten dimensions i>1
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        # convert to reals
        h_mat = dhc.cplx2reals(h_mat)
        y_mat = dhc.cplx2reals(y_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        # input processing
        in_seq = [y_mat, h_mat, n_var_mat]
        logits = model(in_seq)
        llrs = - logits.numpy().astype(float)
        return llrs

