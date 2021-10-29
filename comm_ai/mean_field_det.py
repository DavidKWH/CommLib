# Interface components to Tensorflow
import numpy as np
import tensorflow as tf

from .core import QAMModulator
from .core import preproc_channel
from .core import cplx2reals
from .core import reals2cplx
from .core import bv2dec
from .core import dec2bv
from .core import deintrv_bits
from .core import intrv_bits

class MeanFieldDnnDetector:
    '''
    Encapsulate trained Tensorflow model
    Implements Mean field Detector interface
    Converts log_qi to bits
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('MeanFieldDnnDetector: loading model from file:', fname)
        self.model = tf.saved_model.load(fname)

        self.mod = QAMModulator(p.M)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        #sym_mat, bit_mat = self.build_source_tables()
        sym_mat, bit_mat = self.mod.get_const_real()
        # save sym and bit vector table
        self.sym_mat = sym_mat
        self.bit_mat = bit_mat


    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        p = self.p
        model = self.model
        syms = self.sym_mat
        syms = syms.reshape(-1)
        bits = self.bit_mat
        N = y_tsr.shape[0]
        N_tx_re = p.N_tx * 2
        N_syms_re = p.sqrt_M

        # FIXME:specify dtype
        #dtype = tf.float64
        dtype = tf.float32

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,)
        # convert to reals
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        # compute r2rho
        rho_tsr = 1./n_var_mat
        r2rho_tsr = np.sqrt(2.*rho_tsr)

        # convert to tensors
        #h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)
        r2rho_tsr = tf.convert_to_tensor(r2rho_tsr, dtype=dtype)
        log_qi_in = tf.zeros( (N, N_tx_re, N_syms_re) )

        # input processing
        in_seq = [log_qi_in, y_mat, h_tsr, r2rho_tsr]
        log_qi = model(in_seq)
        # convert to numpy
        log_qi = log_qi.numpy()
        # get candidate
        mf_idx = np.argmax(log_qi, axis=-1)

        syms_mf = syms[mf_idx]
        bits_mf = bits[mf_idx,:]

        # comvert to complex, bit interleaved model
        syms_mf = reals2cplx(syms_mf)
        bits_mf = intrv_bits(bits_mf)

        return bits_mf, syms_mf

class MeanFieldDnnDetectorV2:
    '''
    Encapsulate trained Tensorflow model
    Implements Mean field Detector interface
    Converts log_qi to bits
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('MeanFieldDnnDetector: loading model from file:', fname)
        self.model = tf.saved_model.load(fname)

        self.mod = QAMModulator(p.M)

        # hack
        #print('update alpha')
        #print('before: ', self.model.trainable_variables)
        #self.model.trainable_variables[0].assign(0.1)
        #print('after: ', self.model.trainable_variables)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        #sym_mat, bit_mat = self.build_source_tables()
        sym_mat, bit_mat = self.mod.get_const_real()
        # save sym and bit vector table
        self.sym_mat = sym_mat
        self.bit_mat = bit_mat


    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        p = self.p
        model = self.model
        syms = self.sym_mat
        syms = syms.reshape(-1)
        bits = self.bit_mat
        N = y_tsr.shape[0]
        N_tx_re = p.N_tx * 2
        N_syms_re = p.sqrt_M

        # FIXME:specify dtype
        #dtype = tf.float64
        dtype = tf.float32

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,)
        # convert to reals
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        # compute r2rho
        rho_tsr = 1./n_var_mat
        r2rho_tsr = np.sqrt(2.*rho_tsr)

        # convert to tensors
        #h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)
        r2rho_tsr = tf.convert_to_tensor(r2rho_tsr, dtype=dtype)
        log_qi_in = tf.zeros( (N, N_tx_re, N_syms_re) )

        # input processing
        in_seq = [log_qi_in, y_mat, h_tsr, r2rho_tsr]
        [p_log_qi, t_log_qi] = model(in_seq)
        # convert to numpy
        log_qi = p_log_qi.numpy()
        # get candidate
        mf_idx = np.argmax(log_qi, axis=-1)

        syms_mf = syms[mf_idx]
        bits_mf = bits[mf_idx,:]

        # comvert to complex, bit interleaved model
        syms_mf = reals2cplx(syms_mf)
        bits_mf = intrv_bits(bits_mf)

        return bits_mf, syms_mf



class MeanFieldDnnDetectorV3:
    '''
    Encapsulate trained Tensorflow model
    Implements Mean field Detector interface
    Converts log_qi to bits
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('MeanFieldDnnDetector: loading model from file:', fname)
        self.model = tf.saved_model.load(fname)

        self.mod = QAMModulator(p.M)

        # hack
        #print('update alpha')
        #print('before: ', self.model.trainable_variables)
        #self.model.trainable_variables[0].assign(0.1)
        #print('after: ', self.model.trainable_variables)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        #sym_mat, bit_mat = self.build_source_tables()
        sym_mat, bit_mat = self.mod.get_const_real()
        # save sym and bit vector table
        self.sym_mat = sym_mat
        self.bit_mat = bit_mat


    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        p = self.p
        model = self.model
        syms = self.sym_mat
        syms = syms.reshape(-1)
        bits = self.bit_mat
        N = y_tsr.shape[0]
        N_tx_re = p.N_tx * 2
        N_syms_re = p.sqrt_M

        # FIXME:specify dtype
        #dtype = tf.float64
        dtype = tf.float32

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,)
        # convert to reals
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        # compute r2rho
        rho_tsr = 1./n_var_mat
        r2rho_tsr = np.sqrt(2.*rho_tsr)

        # convert to tensors
        #h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)
        r2rho_tsr = tf.convert_to_tensor(r2rho_tsr, dtype=dtype)
        log_qi_in = tf.zeros( (N, N_tx_re, N_syms_re) )

        # input processing
        in_seq = [log_qi_in, y_mat, h_tsr, r2rho_tsr, n_var_mat]
        [p_log_qi, t_log_qi] = model(in_seq)
        # convert to numpy
        log_qi = p_log_qi.numpy()
        # get candidate
        mf_idx = np.argmax(log_qi, axis=-1)

        syms_mf = syms[mf_idx]
        bits_mf = bits[mf_idx,:]

        # comvert to complex, bit interleaved model
        syms_mf = reals2cplx(syms_mf)
        bits_mf = intrv_bits(bits_mf)

        return bits_mf, syms_mf


class MeanFieldDnnDetectorS1:
    '''
    Encapsulate trained Tensorflow model
    Implements Mean field Detector interface
    Converts log_qi to bits
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('MeanFieldDnnDetector: loading model from file:', fname)
        self.model = tf.saved_model.load(fname)

        self.mod = QAMModulator(p.M)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        #sym_mat, bit_mat = self.build_source_tables()
        sym_mat, bit_mat = self.mod.get_const_real()
        # save sym and bit vector table
        self.sym_mat = sym_mat
        self.bit_mat = bit_mat


    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        p = self.p
        model = self.model
        syms = self.sym_mat
        syms = syms.reshape(-1)
        bits = self.bit_mat
        N = y_tsr.shape[0]
        N_tx_re = p.N_tx * 2
        N_syms_re = p.sqrt_M

        # FIXME:specify dtype
        dtype = tf.float32

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,)
        # convert to reals
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        # compute r2rho
        rho_tsr = 1./n_var_mat
        r2rho_tsr = np.sqrt(2.*rho_tsr)

        # convert to tensors
        #h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)
        r2rho_tsr = tf.convert_to_tensor(r2rho_tsr, dtype=dtype)
        log_qi_in = tf.zeros( (N, N_tx_re, N_syms_re) )

        # input processing
        in_seq = [log_qi_in, y_mat, h_tsr, r2rho_tsr, n_var_mat]
        #[p_log_qi, t_log_qi] = model(in_seq)
        p_log_qi = model(in_seq)
        # convert to numpy
        log_qi = p_log_qi.numpy()
        # get candidate
        mf_idx = np.argmax(log_qi, axis=-1)

        syms_mf = syms[mf_idx]
        bits_mf = bits[mf_idx,:]

        # comvert to complex, bit interleaved model
        syms_mf = reals2cplx(syms_mf)
        bits_mf = intrv_bits(bits_mf)

        return bits_mf, syms_mf


class MeanFieldDnnDetectorS3:
    '''
    Encapsulate trained Tensorflow model
    Implements Mean field Detector interface
    Converts log_qi to bits
    '''

    def __init__(self, p):
        self.p = p
        fname = '/'.join((p.m_dir, p.m_file))
        print('MeanFieldDnnDetector: loading model from file:', fname)
        self.model = tf.saved_model.load(fname)

        self.mod = QAMModulator(p.M)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        #sym_mat, bit_mat = self.build_source_tables()
        sym_mat, bit_mat = self.mod.get_const_real()
        # save sym and bit vector table
        self.sym_mat = sym_mat
        self.bit_mat = bit_mat


    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        p = self.p
        model = self.model
        syms = self.sym_mat
        syms = syms.reshape(-1)
        bits = self.bit_mat
        N = y_tsr.shape[0]
        N_tx_re = p.N_tx * 2
        N_syms_re = p.sqrt_M

        # FIXME:specify dtype
        dtype = tf.float32

        # flatten dimensions i>1
        #h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,)
        # convert to reals
        h_tsr = preproc_channel(h_tsr)
        y_mat = cplx2reals(y_mat)
        # compute r2rho
        rho_tsr = 1./n_var_mat
        r2rho_tsr = np.sqrt(2.*rho_tsr)

        # convert to tensors
        #h_mat = tf.convert_to_tensor(h_mat, dtype=dtype)
        h_tsr = tf.convert_to_tensor(h_tsr, dtype=dtype)
        y_mat = tf.convert_to_tensor(y_mat, dtype=dtype)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=dtype)
        r2rho_tsr = tf.convert_to_tensor(r2rho_tsr, dtype=dtype)
        log_qi_in = tf.zeros( (N, N_tx_re, N_syms_re) )

        # input processing
        in_seq = [log_qi_in, y_mat, h_tsr, r2rho_tsr, n_var_mat]
        [p_log_qi, t_log_qi] = model(in_seq)
        #p_log_qi = model(in_seq)
        # convert to numpy
        log_qi = p_log_qi.numpy()
        # get candidate
        mf_idx = np.argmax(log_qi, axis=-1)

        syms_mf = syms[mf_idx]
        bits_mf = bits[mf_idx,:]

        # comvert to complex, bit interleaved model
        syms_mf = reals2cplx(syms_mf)
        bits_mf = intrv_bits(bits_mf)

        return bits_mf, syms_mf

