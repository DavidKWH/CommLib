# Interface components to Tensorflow
import os
import json
import hashlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .core import cplx2reals
from .core import QAMModulator
from .core import Demodulator
from .core import Channel
from .core import Transmitter
from .core import bv2dec

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from .keras import PeriodicLRDecaySchedule
from .keras import AperiodicLRDecaySchedule
from .params import get_key
from .params import has_key
from .drive import save_file
from .drive import save_folder

################################################################################
# Support functions
################################################################################
def get_sim_id(p, size=8):
    '''generate simulation id via param based digest'''
    pbytes = json.dumps(p.as_serializable()).encode()
    sim_id = hashlib.blake2b(pbytes, digest_size=size).hexdigest()
    return sim_id

def get_prefix(p):
    '''generate filepath (ensure parent folder exists)'''
    # ensure folder exists
    os.makedirs(p.outdir, exist_ok=True)

    pname = '_'.join((p.bname, p.sname, get_sim_id(p)))
    pname = '/'.join((p.outdir, pname))
    return pname

def get_model_prefix(p, model_name):
    '''generate filepath (ensure parent folder exists)'''
    # ensure folder exists
    os.makedirs(p.outdir, exist_ok=True)

    sname = '_'.join((p.sname, model_name)) if model_name else p.sname
    pname = '_'.join((p.bname, sname, get_sim_id(p)))
    pname = '/'.join((p.outdir, pname))
    return pname

def plot_model(p, model, name=None, show=False):
    ''' plot image file '''

    pname = get_model_prefix(p, name)
    fname = pname + '.png'

    print('saving model graph to file:', fname)
    tf.keras.utils.plot_model(model, fname, show_shapes=True)
    if show:
        print('close plot to continue...')
        img = mpimg.imread(fname)
        plt.figure(dpi=175)
        #plt.imshow(img, interpolation='nearest')
        plt.imshow(img, interpolation='bilinear')
        plt.axis("off")   # turns off axes
        plt.axis("tight")  # gets rid of white border
        plt.axis("image")
        plt.show()

def save_model(p, model, save_to_remote=False):
    ''' save model to file '''

    # get prefix
    pname = get_prefix(p)

    # save params to file
    fname = pname + '.json'
    with open(fname, 'w') as fp:
        print('writing params to file:', fname)
        json.dump(p.as_serializable(), fp, indent=4)

    if save_to_remote: save_file(fname)

    # save model to file
    fname = pname + '.model'
    print('saving model to folder: ', fname)
    tf.saved_model.save(model, fname)

    if save_to_remote: save_folder(fname)

    # save weights for retrieval
    #fname = pname + '.weights'
    #print('saving weights to file: ', fname)
    #model.save_weights(fname, save_format='tf')

    # Load the state of the old model
    # NOTE: remember to call the model with data to create the weights
    #new_model.load_weights(fname)

################################################################################
# optimizer selection
################################################################################
def get_optimizer(p):
    '''create the optimizer'''

    # schedule type lookup
    schedule_types = {
            'periodic' : PeriodicLRDecaySchedule,
            'aperiodic': AperiodicLRDecaySchedule,
    }

    # optimizer type lookup
    optimizer_types = {
            'adam'    : Adam,
            'momentum': SGD,
    }

    # sanity check
    assert has_key(p.lr_sched.type, get_key(p.optimizer.type)), \
        "schedule and optimizer do not match"

    # setup schedule(s)
    mt_schedule = None
    lr_schedule = None
    if p.lr_sched is not None:
        Schedule = schedule_types[get_key(p.lr_sched.type)]
        kwargs = p.lr_sched.as_dict(exclude=('type'))
        lr_schedule = Schedule(**kwargs)

    # setup optimizer
    Optimizer = optimizer_types[get_key(p.optimizer.type)]
    # FIXME: make serializer error more informative...
    #if lr_schedule: p.optimizer.learning_rate = lr_schedule
    kwargs = p.optimizer.as_dict(exclude=('type'))
    optimizer = Optimizer(**kwargs)

################################################################################
# Comm functions
################################################################################
def quantize_o(x_cpx):
    '''perform component wise quantization'''
    return np.sign(x_cpx.real) + 1j* np.sign(x_cpx.imag)

################################################################################
# Comm dataset V2
################################################################################
class CommDataSet:
    '''
    Iterable Wrapper for Tensorflow model training (V2)
    NOTE: Will break existing scripts using V1 dataset
    '''

    def __init__(self, p, llr_output=False,
                          symbol_output=False,
                          maxlog_approx=False,
                          mode='none',
                          test=False,
                          transform=None,
                          one_bit=False):
        assert( hasattr(p,mode) )
        self.p = p
        self.n_iter = p.n_test_steps if test else p.n_train_steps
        self.maxlog_approx = maxlog_approx
        self.llr_output = llr_output
        self.sym_output = symbol_output
        self.test = test
        self.mode = mode
        # comm. components
        self.mod = QAMModulator(p.M)
        self.demod = Demodulator(p, modulator=self.mod,
                                    maxlog_approx=maxlog_approx)
        self.channel = Channel(p, mode)
        self.transmit = Transmitter(p, modulator=self.mod, training=True)
        self.in_transform = transform
        self.one_bit = one_bit

        print("CommDataSet one_bit mode =", self.one_bit)

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

        outputs:
            in_seq = [y_mat, h_mat, n_var_mat]
            out_seq = [bit_mat, lambda_mat, sym_vec]
        '''
        p = self.p
        test = self.test
        llr_output = self.llr_output or self.test
        sym_output = self.sym_output
        mod = self.mod
        transmit = self.transmit
        channel = self.channel
        demod = self.demod
        in_transform = self.in_transform

        sym_vec = None
        lambda_mat = None
        N = p.batch_size

        # generate payload bits and symbols
        syms_tsr, bit_tsr = transmit(N)
        # apply channel and noise
        y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
        # SNR is multiplexed on the same channel output...
        snr_tsr = n_var_tsr
        # one bit quantization
        if self.one_bit:
            y_tsr = quantize_o(y_tsr)
        # generate LLRs (test mode)
        if llr_output: lambda_mat = demod(y_tsr, h_tsr, n_var_tsr)

        # flatten dimensions i>1
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        bit_mat = bit_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        snr_mat = snr_tsr.reshape(N,-1)

        # symbol output?
        sym_vec = bv2dec(bit_mat) if sym_output else None

        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        snr_mat = tf.convert_to_tensor(snr_mat, dtype=tf.int32)
        bit_mat = tf.convert_to_tensor(bit_mat, dtype=tf.bool)
        # conditional outputs
        lambda_mat = tf.convert_to_tensor(lambda_mat, dtype=tf.float32) if llr_output else None
        sym_vec = tf.convert_to_tensor(sym_vec, dtype=tf.int32) if sym_output else None

        # output processing
        in_seq = [y_mat, h_mat, n_var_mat]
        if p.train.noise_type in ('rand_snr', 'fixed_snr'):
            in_seq = [y_mat, h_mat, snr_mat]
        if in_transform: in_seq = in_transform(in_seq)
        out_seq = [bit_mat, sym_vec]
        aux_out = lambda_mat

        return in_seq, out_seq, aux_out

    def transform_input(self, transform_fn):
        self.in_transform = transform_fn
        return self

    def get_const_input(self):
        """ transform constellation for DNN training """
        mod = self.mod
        syms, _ = mod.get_const()
        return np.r_[syms.real, syms.imag]


class DnnDemod:
    '''
    Encapsulate trained Tensorflow model
    Implements Demodulator interface

    NOTE: Use default mode for evaluating DNN model
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('loading model from file:', fname)
        self.model = tf.saved_model.load(fname)

    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        p = self.p
        model = self.model
        N = y_tsr.shape[0]

        # FIXME:specify dtype
        #dtype = tf.float64
        dtype = tf.float32

        # flatten dimensions i>1
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        snr_mat = n_var_mat
        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)
        snr_mat = tf.convert_to_tensor(snr_mat, dtype=tf.int32)
        # input processing
        in_seq = [y_mat, h_mat, n_var_mat]
        if p.default.noise_type in ('rand_snr', 'fixed_snr'):
            in_seq = [y_mat, h_mat, snr_mat]
        logits = model(in_seq)
        llrs = - logits.numpy().astype(float)
        return llrs


