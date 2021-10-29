# Interface components to Tensorflow
import os
import json
import hashlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

from .one_bit import OneBitMLDemod
from .one_bit import BussgangEstimator

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

    pname = get_model_prefix(p, 'model_summary')
    fname = pname + '.txt'
    print('saving model summary to file:', fname)

    from contextlib import redirect_stdout
    with open(fname, 'w') as f:
        with redirect_stdout(f):
            model.summary()

    if show:
        model.summary()

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

class QuantizerWithDCBias:
    def __init__(self, p):
        '''generate random DC bias'''
        import numpy.random as rnd

        # cplx = 2 real dimensions
        signs = rnd.binomial(1,p.dc.prob,size=(p.N_rx,2))
        signs = 2.0*signs - 1.0
        # save vars
        self.offsets = p.dc.offset * signs

        if p.dc.load_file is not None:
            print('loading DC offsets from file: ' + p.dc.load_file)
            self.offsets = np.load(p.dc.load_file)
        else:
            # save to file
            fname = 'dc_offsets.npy'
            np.save(fname, self.offsets)
            print('saving DC offsets to file: ' + fname)

    def __call__(self, x_cpx):
        offset_real = self.offsets[:,0]
        offset_real = offset_real[None,:,None]
        offset_imag = self.offsets[:,1]
        offset_imag = offset_imag[None,:,None]

        return np.sign(x_cpx.real + offset_real) + 1j * np.sign(x_cpx.imag + offset_imag)


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
                          one_bit=False,
                          symbol_only=False,
                          meta_learn=False):
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

        if one_bit:
            self.demod = OneBitMLDemod(p, modulator=self.mod,
                                          maxlog_approx=maxlog_approx)
        else:
            self.demod = Demodulator(p, modulator=self.mod,
                                        maxlog_approx=maxlog_approx)

        self.channel = Channel(p, mode)

        if symbol_only:
            self.transmit = SymbolTransmitter(p, modulator=self.mod, training=True)
        else:
            self.transmit = Transmitter(p, modulator=self.mod, training=True)

        if p.dc_bias:
            self.quantize_with_dc_bias = QuantizerWithDCBias(p)

        self.in_transform = transform
        self.one_bit = one_bit
        self.symbol_only = symbol_only
        self.meta_learn = meta_learn
        self.dc_bias = p.dc_bias

        print("CommDataSet: one_bit mode =", self.one_bit)
        print("CommDataSet: symbol only =", self.symbol_only)
        print("CommDataSet: meta learn =", self.meta_learn)
        print("CommDataSet: DC bias =", self.dc_bias)

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
        llr_output = self.llr_output
        sym_output = self.sym_output
        mod = self.mod
        transmit = self.transmit
        channel = self.channel
        demod = self.demod
        in_transform = self.in_transform

        sym_vec = None
        lambda_mat = None
        N = p.batch_size

        if self.meta_learn:
            # assume train and test the same
            N = p.mlearn.batch_size * 2

        # generate payload bits and symbols
        syms_tsr, bit_tsr = transmit(N)
        # apply channel and noise
        y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
        # output differs depending on noise_type
        if p.train.noise_type == 'rand_snr':
            snr_tsr = n_var_tsr
        # one bit quantization
        if self.one_bit:
            if self.dc_bias:
                y_tsr = self.quantize_with_dc_bias(y_tsr)
            else:
                y_tsr = quantize_o(y_tsr)
        # generate LLRs (test mode)
        if llr_output: lambda_mat = demod(y_tsr, h_tsr, n_var_tsr)

        # flatten dimensions i>1
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        bit_mat = bit_tsr.reshape(N,-1)
        sym_mat = syms_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        if p.train.noise_type == 'rand_snr':
            snr_mat = snr_tsr.reshape(N,-1)

        # symbol output?
        sym_vec = bv2dec(bit_mat) if sym_output else None

        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        sym_mat = cplx2reals(sym_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        sym_mat = tf.convert_to_tensor(sym_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        if p.train.noise_type == 'rand_snr':
            snr_mat = tf.convert_to_tensor(snr_mat, dtype=tf.int32)
        bit_mat = tf.convert_to_tensor(bit_mat, dtype=tf.bool)
        # conditional outputs
        lambda_mat = tf.convert_to_tensor(lambda_mat, dtype=tf.float32) if llr_output else None
        sym_vec = tf.convert_to_tensor(sym_vec, dtype=tf.int32) if sym_output else None

        if self.meta_learn:
            return y_mat, h_mat, n_var_mat, bit_mat, sym_vec

        # output processing
        in_seq = [y_mat, h_mat, n_var_mat]
        if p.train.noise_type == 'rand_snr':
            in_seq = [y_mat, h_mat, snr_mat]
        if in_transform: in_seq = in_transform(in_seq)
        out_seq = [bit_mat, sym_vec, sym_mat]
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


class SymbolDetectDataSet:
    '''
    Iterable Wrapper for Tensorflow model training (V2)
    NOTE: Will break existing scripts using V1 dataset
    '''

    def __init__(self, p, mode='none',
                          test=False,
                          one_bit=False,
                          augment=False):
        assert( hasattr(p,mode) )
        self.p = p
        self.n_iter = p.n_test_steps if test else p.n_train_steps
        self.test = test
        self.mode = mode
        # comm. components
        self.mod = QAMModulator(p.M)
        self.channel = Channel(p, mode)
        self.transmit = SymbolTransmitter(p, modulator=self.mod, training=True)

        if p.dc_bias:
            self.quantize_with_dc_bias = QuantizerWithDCBias(p)

        if augment:
            self.est = BussgangEstimator(p)

        self.one_bit = one_bit
        self.dc_bias = p.dc_bias
        self.augment = augment

        print("SymbolDetectDataSet: one_bit mode =", self.one_bit)
        print("SymbolDetectDataSet: DC bias =", self.dc_bias)
        print("SymbolDetectDataSet: augment =", self.augment)

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

        # generate payload bits and symbols
        syms_tsr, bit_tsr = transmit(N)
        # apply channel and noise
        y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
        # analog signal
        r_tsr = y_tsr
        # one bit quantization
        if self.one_bit:
            if self.dc_bias:
                y_tsr = self.quantize_with_dc_bias(y_tsr)
            else:
                y_tsr = quantize_o(y_tsr)

        # augment mode
        if self.augment:
            syms_bmmse, covar = self.est.estimate(y_tsr, h_tsr, n_var_tsr)
            syms_err_tsr = syms_tsr - syms_bmmse
            syms_tsr = syms_err_tsr

        # flatten dimensions i>1
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        r_mat = r_tsr.reshape(N,-1)
        bit_mat = bit_tsr.reshape(N,-1)
        sym_mat = syms_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)

        # symbol output?
        sym_vec = bv2dec(bit_mat)

        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        r_mat = cplx2reals(r_mat)
        sym_mat = cplx2reals(sym_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        r_mat = tf.convert_to_tensor(r_mat, dtype=tf.float32)
        sym_mat = tf.convert_to_tensor(sym_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        bit_mat = tf.convert_to_tensor(bit_mat, dtype=tf.bool)
        # conditional outputs
        sym_vec = tf.convert_to_tensor(sym_vec, dtype=tf.int32)

        # output processing
        in_seq = [y_mat, h_mat, n_var_mat]
        out_seq = [bit_mat, sym_vec, sym_mat, r_mat]
        aux_out = []

        return in_seq, out_seq, aux_out

    def transform_input(self, transform_fn):
        self.in_transform = transform_fn
        return self

    def get_const_input(self):
        """ transform constellation for DNN training """
        mod = self.mod
        syms, _ = mod.get_const()
        return np.r_[syms.real, syms.imag]


class ModelBasedDataSet:
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

        print("SymbolDetectDataSet: one_bit mode =", self.one_bit)

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

        # generate payload bits and symbols
        syms_tsr, bit_tsr = transmit(N)
        # apply channel and noise
        y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
        # analog signal
        r_tsr = y_tsr
        # one bit quantization
        if self.one_bit:
            y_tsr = quantize_o(y_tsr)

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        r_mat = r_tsr.reshape(N,-1)
        bit_mat = bit_tsr.reshape(N,-1)
        sym_mat = syms_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)

        # symbol output?
        sym_vec = bv2dec(bit_mat)

        # convert to reals
        #h_mat = cplx2reals(h_mat)
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        r_mat = cplx2reals(r_mat)
        sym_mat = cplx2reals(sym_mat)
        # convert to tensors
        #h_mat = tf.convert_to_tensor(h_mat, dtype=tf.float32)
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        r_mat = tf.convert_to_tensor(r_mat, dtype=tf.float32)
        sym_mat = tf.convert_to_tensor(sym_mat, dtype=tf.float32)
        x0_mat = tf.zeros_like(sym_mat)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        bit_mat = tf.convert_to_tensor(bit_mat, dtype=tf.bool)
        # conditional outputs
        sym_vec = tf.convert_to_tensor(sym_vec, dtype=tf.int32)

        # output processing
        in_seq = [y_mat, h_tsr, n_var_mat]
        out_seq = [bit_mat, sym_vec, sym_mat, r_mat]
        aux_out = [x0_mat]

        return in_seq, out_seq, aux_out

    def transform_input(self, transform_fn):
        self.in_transform = transform_fn
        return self

    def get_const_input(self):
        """ transform constellation for DNN training """
        mod = self.mod
        syms, _ = mod.get_const()
        return np.r_[syms.real, syms.imag]


class PermInvariantDataSet:
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

        if p.dc_bias:
            self.quantize_with_dc_bias = QuantizerWithDCBias(p)

        self.one_bit = one_bit
        self.dc_bias = p.dc_bias

        print("SymbolDetectDataSet: one_bit mode =", self.one_bit)
        print("SymbolDetectDataSet: DC bias =", self.dc_bias)

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

        # generate payload bits and symbols
        syms_tsr, bit_tsr = transmit(N)
        # apply channel and noise
        y_tsr, h_tsr, n_var_tsr = channel(syms_tsr)
        # analog signal
        r_tsr = y_tsr
        # one bit quantization
        if self.one_bit:
            if self.dc_bias:
                y_tsr = self.quantize_with_dc_bias(y_tsr)
            else:
                y_tsr = quantize_o(y_tsr)

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        r_mat = r_tsr.reshape(N,-1)
        bit_mat = bit_tsr.reshape(N,-1)
        sym_mat = syms_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)

        # symbol output?
        sym_vec = bv2dec(bit_mat)

        # convert to reals
        #h_mat = cplx2reals(h_mat)
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        r_mat = cplx2reals(r_mat)
        sym_mat = cplx2reals(sym_mat)
        # convert to tensors
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        r_mat = tf.convert_to_tensor(r_mat, dtype=tf.float32)
        sym_mat = tf.convert_to_tensor(sym_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)
        bit_mat = tf.convert_to_tensor(bit_mat, dtype=tf.bool)
        # conditional outputs
        sym_vec = tf.convert_to_tensor(sym_vec, dtype=tf.int32)

        # output processing
        in_seq = [y_mat, h_tsr, n_var_mat]
        out_seq = [bit_mat, sym_vec, sym_mat, r_mat]
        aux_out = []

        return in_seq, out_seq, aux_out

    def transform_input(self, transform_fn):
        self.in_transform = transform_fn
        return self

    def get_const_input(self):
        """ transform constellation for DNN training """
        mod = self.mod
        syms, _ = mod.get_const()
        return np.r_[syms.real, syms.imag]

################################################################################
# Dnn wrappers for evaluation
################################################################################
class DnnDemod:
    '''
    Encapsulate trained Tensorflow model
    Implements Demodulator interface
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('loading model from file:', fname)
        self.model = tf.saved_model.load(fname)

    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        model = self.model
        N = y_tsr.shape[0]

        # FIXME:specify dtype
        #dtype = tf.float64
        dtype = tf.float32

        # flatten dimensions i>1
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        #n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)

        # convert n_var to integer
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.int32)

        # input processing
        in_seq = [y_mat, h_mat, n_var_mat]
        logits = model(in_seq)
        llrs = - logits.numpy().astype(float)
        return llrs


class DnnDetector:
    '''
    Encapsulate trained Tensorflow model
    Implements Detector interface
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('DnnDetector: loading model from file:', fname)
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
        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)

        # convert n_var to integer
        #n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.int32)

        # input processing
        in_seq = [y_mat, h_mat, n_var_mat]
        logits = model(in_seq)
        # output processing
        #one_hot_pred = (logits == logits.max(axis=1))
        pred_idx = tf.argmax(logits, axis=1).numpy()
        bits_mat = dec2bv(pred_idx, p.nbpsv)
        syms_mat = None

        return bits_mat, syms_mat


class DnnEstimator:
    '''
    Encapsulate trained Tensorflow model
    Implements Estimator-Detector interface
    '''

    def __init__(self, p, augment=False):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('DnnEstimator: loading model from file:', fname)
        self.model = tf.saved_model.load(fname)
        # instantiate per symbol detector
        from .core import PerSymbolDetectorV2
        self.det = PerSymbolDetectorV2(p)
        # save options
        if augment:
            self.est = BussgangEstimator(p)

        self.augment = augment
        print('DnnEstimator: augment =', augment)


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
        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)

        # convert n_var to integer
        #n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.int32)

        # input processing
        in_seq = [y_mat, h_mat, n_var_mat]
        sym_est = model(in_seq)
        # convert to numpy
        sym_est = sym_est.numpy()
        # convert to complex
        sym_est = reals2cplx(sym_est)
        # add dimension for array multiplication
        sym_est = sym_est[:,:,None]
        # augment step
        if self.augment:
            syms_bmmse, covar = self.est.estimate(y_tsr, h_tsr, n_var_tsr)
            sym_est = sym_est + syms_bmmse
        # output processing
        bits_msd, syms_msd = self.det.compute_msd(sym_est)
        # return to 2d
        sym_est = np.squeeze(sym_est)

        #if self.augment:
        #    return bits_msd, syms_msd, sym_est, syms_bmmse

        return bits_msd, syms_msd, sym_est


class ModelBasedDnnEstimator:
    '''
    Encapsulate trained Tensorflow model
    Implements Estimator-Detector interface
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('ModelBasedDnnEstimator: loading model from file:', fname)
        self.model = tf.saved_model.load(fname)
        # instantiate per symbol detector
        from .core import PerSymbolDetectorV2
        self.det = PerSymbolDetectorV2(p)


    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        p = self.p
        model = self.model
        N = y_tsr.shape[0]
        x_dim = p.N_tx * 2

        # FIXME:specify dtype
        #dtype = tf.float64
        dtype = tf.float32

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)
        # convert to reals
        #h_mat = cplx2reals(h_mat)
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        # convert to tensors
        #h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)
        x0_mat = tf.zeros( (N, x_dim) )

        # convert n_var to integer
        #n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.int32)

        # input processing
        in_seq = [x0_mat, y_mat, h_tsr]
        sym_est = model(in_seq)
        # convert to numpy
        sym_est = sym_est.numpy()
        # convert to complex
        sym_est = reals2cplx(sym_est)
        # add dimension for array multiplication
        sym_est = sym_est[:,:,None]
        # output processing
        bits_msd, syms_msd = self.det.compute_msd(sym_est)
        # return to 2d
        sym_est = np.squeeze(sym_est)

        return bits_msd, syms_msd, sym_est

