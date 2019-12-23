# Essential communications components
# [DONE] need to match matlab functions' behavior by default
#  NOTE: matlab does not implement standard defined Gray coding
#        commpy implements binary encoding

# general dependencies
import numpy as np
import numpy.linalg as la
import numpy.random as rnd
# for crandn
from numpy.random import randn
# helper functions
from functools import partial
# package local
from .params import get_key
from .params import has_key

################################################################################
# comm. utility functions
################################################################################

def crandn(*args):
    '''
    standard complex normal random samples
    arguments compatible with randn()
    '''
    samps = np.sqrt(0.5) * (randn(*args) + randn(*args) * 1j)
    return samps

def cplx2reals(a):
    '''
    convert complex tensor of shape (M,N,...,K) into
    real tensor of shape (M,N,...,2k) by concatenating
    the real and image matrix along the last dimension
    '''
    assert (a.dtype == np.complex)
    if a.ndim == 1: a = a[:,np.newaxis]
    ac_all = np.concatenate([a.real, a.imag], axis=-1)
    return ac_all

def cplx2reals_old(a):
    '''
    convert complex array of shape (N,k) into
    alternating real/imag arrays of shape (N,2k)
    e.g. a   = [a1, a2, ...]
         out = [a1.real, a1.imag, a2.real, a2.imag, ...]
    '''
    assert (a.dtype == np.complex)
    if a.ndim == 1: a = a[:,np.newaxis]
    N = a.shape[0]
    ac_all = np.r_[a.real, a.imag]
    ac_all = ac_all.reshape(N,-1, order='F')
    return ac_all

def dec2bv(dec_vec, n=None):
    '''
    convert unsigned decimals to MSB-first binary vectors
    NOTE: bit vectors are MSB-first (e.g. 6 => '110')
    '''
    max_val = np.max(dec_vec)
    nbits = np.ceil(np.log2(max_val)).astype(int)

    if n: nbits = n

    dec_vec = dec_vec.reshape(-1,1)
    bit_mask = np.flip(1 << np.arange(nbits))
    bit_mat = (dec_vec & bit_mask) > 0

    return bit_mat

def bv2dec(bit_mat):
    '''
    convert MSB-first binary (row) vectors to unsigned decimals
    NOTE: bit vectors are MSB-first (e.g. 6 <= '110')
    '''
    (rows, cols) = bit_mat.shape

    base_vec = np.flip(1 << np.arange(cols))
    dec_vec = np.sum(bit_mat * base_vec[None,:], axis=1)

    return dec_vec

################################################################################
# comm. classes
# FIXME: split into separate files
################################################################################

################################################################################
# high level receiver class
################################################################################
class Receiver:
    '''
    Encapsulate demod and decoding
    '''

    def __init__(self,p,
                 demodulator = None,
                 estimator = None,
                 demapper = None,
                 decoder = None,
                 ):
        '''
        Assumptions:
            Either accept a demodulator that maps (y,H) directly to LLRs or
            a (estimator, demapper) pair that perform a two stage process
        '''

        assert (demodulator or
                not demodulator and (estimator and demapper)
                )

        self.p = p
        self.demod = demodulator
        self.est = estimator
        self.demap = demapper
        self.dec = decoder

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)

    def detect(self, y_tsr, h_tsr, n_var_tsr):
        '''
        when decoder is None, returns hard decisions and llrs
        when decoder is present, returns decoded bits and iterations
        NOTE: return bit vectors as list easier to handle
        '''
        p = self.p
        demod = self.demod
        dec = self.dec
        est = self.est
        demap = self.demap

        if demod:
            llr_mat = demod.compute_llrs(y_tsr, h_tsr, n_var_tsr)
        else:
            x_hat_tsr = est.estimate(y_tsr, h_tsr, n_var_tsr)
            llr_mat = demap.compute_llrs(x_hat_tsr, h_tsr, n_var_tsr)

        # reshape
        llr_tsr = llr_mat.reshape(p.N_syms, p.N_sts, p.nbps)

        # reshuffle llrs matrix
        # llrs_list contains N_sts llr vectors
        llrs_list = np.split(llr_tsr, p.N_sts, axis=1)
        llrs_list = [ llrs.reshape(-1) for llrs in llrs_list ]

        if dec == None:
            # generate hard decisions
            dec_bits_list = [ llrs < 0  for llrs in llrs_list ]
            return dec_bits_list, llrs_list
        else:
            # decode llrs
            dec_out = [ dec.decode(llrs) for llrs in llrs_list ]
            # extract columns (decoded_bits, iterations)
            columns = list(zip(*dec_out))
            dec_bits_list = columns[0]
            iter_list = columns[1]
            # remove parity bits
            dec_bits_list = [ llrs[:p.dec.K] for llrs in dec_bits_list ]
            return dec_bits_list, iter_list


################################################################################
# high level transmitter class
################################################################################
class Transmitter:
    '''
    Encapsulate all TX functions
    Support multi-stream processing
    NOTE: check init-time pre-conditions below
    '''

    def __init__(self, p,
                 encoder = None,
                 modulator = None,
                 training = False,
                 ):

        #NOTE: does not allow training and encoding
        assert(modulator)
        assert( not(training == True and encoder) )

        self.p = p
        self.mod = modulator
        self.enc = encoder
        self.training = training

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, N=None):
        '''
        generate raw payload bits and symbols
        NOTE: check runtime pre-conditions below
        NOTE: return bit vectors as list easier to handle
        '''
        p = self.p
        mod = self.mod
        enc = self.enc
        training = self.training

        #NOTE: allow N only in training mode
        assert( not(training == False and N != None) )

        # per stream processing
        ####################################
        if enc == None:
            N_raw = p.N_syms if N == None else N
            #raw_bit_tsr = rnd.randint(2, size=(p.N_sts, p.N_syms, p.nbps))
            raw_bit_tsr = rnd.randint(2, size=(p.N_sts, N_raw, p.nbps))
            bit_mat = raw_bit_tsr.reshape(p.N_sts, -1)
            raw_bits_list = list(bit_mat)
            mapper_bits_list = raw_bits_list
        else:
            raw_bit_tsr = rnd.randint(2, size=(p.N_sts, p.N_raw, p.nbps))
            raw_bit_mat = raw_bit_tsr.reshape(p.N_sts, -1)
            raw_bits_list = list(raw_bit_mat)
            # encode raw bits, produces N_sts codewords
            encoded_bits_list = [ enc.encode(raw_bits) for raw_bits in raw_bits_list ]
            mapper_bits_list = encoded_bits_list

        # map to symbols
        syms_list = [ mod.map(bits) for bits in mapper_bits_list ]
        syms = np.array(syms_list)

        # convert to stacked matrix form
        # for element-wise matrix multiplication using @
        # i.e., the first dimension is turned into a list
        #       containing elements in the remaining dimensions
        # NOTE: a @ b <=> np.matmul(a,b)
        # syms.shape = (N_sts, N_syms)
        syms_tr = syms.transpose()
        # syms_tr.shape = (N_syms, N_sts)
        N_syms = N if training else p.N_syms
        #sym_tsr = syms_tr.reshape(p.N_syms, p.N_sts, 1)
        sym_tsr = syms_tr.reshape(N_syms, p.N_sts, 1)

        # output processing
        if training:
            # reformat bits for training
            # list(bit_str) = N_sts elements of shape (N, p.nbps)
            bit_mat_trn = np.concatenate(list(raw_bit_tsr), axis=1)
            # bit_mat_trn.shape = (N, N_sts * nbps)
            bit_tsr_trn = bit_mat_trn.reshape(N, p.N_sts, p.nbps)
            bits_output = bit_tsr_trn
        else:
            bits_output = raw_bits_list

        # return symbols, raw bits
        return sym_tsr, bits_output


################################################################################
# channel class
################################################################################
class Channel:
    '''
    Encapsulate channel and noise generation, application
    '''

    def __init__(self, p, mode=None):

        # log uniform for now
        assert(p.noise_dist == 'log_uniform')

        self.p = p
        self.pm = pm = getattr(p,mode)
        self.ch_gen = partial(self.channels[get_key(pm.ch_type)], self)
        self.n_gen = partial(self.noise_table[get_key(pm.noise_type)], self)
        self.batch_fixed = has_key(pm.ch_type, 'batch_fixed')

    def __call__(self, syms):
        return self.apply(syms)

    '''
    channel generation functions
    '''
    def gen_ch_from_sv_dist(self, N, N_tx, N_rx):
        # generate matrices with specific sv distribution for DNN training
        assert(N_tx == N_rx)
        from scipy.stats import unitary_group
        p = self.p

        # generate random sv's
        sv_mat = rnd.uniform(low=p.u_a, high=p.u_b, size=(N,N_tx))
        # generate random unitary matrices
        U_tsr = unitary_group.rvs(dim=N_tx, size=N)
        V_tsr = unitary_group.rvs(dim=N_tx, size=N)
        S_list = [np.diag(sv) for sv in list(sv_mat)]
        S_tsr = np.array(S_list)

        H = U_tsr @ S_tsr @ V_tsr

        return H

    def gen_rayleigh_ch(self, N, N_tx, N_rx):
        #print('gen_rayleigh_ch, batch_fixed=', self.batch_fixed)
        # assume square matrix
        assert(N_tx == N_rx)
        scale = 1 / np.sqrt(N_tx)

        if self.batch_fixed:
            # repeat H N-times
            H = scale * crandn(N_tx, N_rx)
            H_tsr = np.tile(H, (N,1,1))
        else:
            H_tsr = scale * crandn(N, N_tx, N_rx)

        return H_tsr

    def gen_identity_ch(self, N, N_tx, N_rx):
        H = np.identity(N_tx).astype(complex)
        return np.tile(H,(N,1)).reshape(N, N_tx, N_rx)

    # channel selection
    channels = {
        'sv_dist'  : gen_ch_from_sv_dist,
        'identity' : gen_identity_ch,
        'rayleigh' : gen_rayleigh_ch,
    }

    '''
    noise generation functions
    '''
    def gen_noiseless(self, N, N_rx):
        n_tsr = np.zeros((N, N_rx, 1))
        n_var_tsr = np.zeros((N,1,1))
        return n_tsr, n_var_tsr

    def gen_fixed_var(self, N, N_rx):
        p = self.p
        std_n_tsr = crandn(N, N_rx, 1)
        n_tsr = p.n_std * std_n_tsr
        n_var_tsr = p.n_var * np.ones((N,1,1))
        return n_tsr, n_var_tsr

    def gen_rand_var(self, N, N_rx):
        p = self.p
        snr_tsr_db = rnd.uniform(low=p.log_u_a, high=p.log_u_b, size=(N,1,1))
        n_std_tsr = 10**(-snr_tsr_db/20)
        n_var_tsr = n_std_tsr**2
        std_n_tsr = crandn(N, N_rx, 1)
        n_tsr = n_std_tsr * std_n_tsr
        return n_tsr, n_var_tsr

    # noise generation
    noise_table = {
        'fixed_var' : gen_fixed_var,
        'rand_var' : gen_rand_var,
        'noiseless' : gen_noiseless,
    }

    def apply(self, sym_tsr):
        p = self.p
        ch_gen = self.ch_gen
        n_gen = self.n_gen

        # sym_tsr.shape = (N, N_tx, 1)
        N = sym_tsr.shape[0]
        # generate channel realizations
        h_tsr = ch_gen(N, p.N_tx, p.N_rx)
        # genrate (scaled) noise vectors
        n_tsr, n_var_tsr = n_gen(N, p.N_rx)
        # element-wise matrix multiplication via @
        y_tsr = h_tsr @ sym_tsr + n_tsr

        return y_tsr, h_tsr, n_var_tsr

    '''
    Extension to Channel class
    '''
    def gen_channels(self, N):
        p = self.p
        return self.ch_gen(N, p.N_tx, p.N_rx)

    def apply_channel(self, sym_tsr, h_mat):
        p = self.p
        n_gen = self.n_gen

        # sym_tsr.shape = (N, N_tx, 1)
        N = sym_tsr.shape[0]
        # reshape h
        h_tsr = h_mat[None,...]
        # genrate (scaled) noise vectors
        n_tsr, n_var_tsr = n_gen(N, p.N_rx)
        # element-wise matrix multiplication via @
        y_tsr = h_tsr @ sym_tsr + n_tsr

        return y_tsr, n_var_tsr

################################################################################
# demodulator classes
################################################################################
class Demodulator:
    '''
    Computes exact LLRs given transmit alphabets,
    receive symbols and exact channel knowledge
    '''

    def __init__(self, p,
                 modulator = None,
                 maxlog_approx = False
                 ):
        self.p = p
        self.maxlog_approx = maxlog_approx
        if modulator:
            self.mod = modulator
        else:
            # assume QAM used
            self.mod = QAMModulator(p.M)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        sym_mat, bit_mat = self.build_source_tables()
        # construct symbol subsets
        n_bits = p.nbps * p.N_sts
        sym_sets_1 = []
        sym_sets_0 = []
        for i in range(n_bits):
            idx_1 = (bit_mat[:,i] == 1)
            idx_0 = (bit_mat[:,i] == 0)
            sym_sets_1.append( sym_mat[idx_1,:] )
            sym_sets_0.append( sym_mat[idx_0,:] )
        # save them up
        self.sym_sets_1 = sym_sets_1
        self.sym_sets_0 = sym_sets_0

    # multi-stream bv/symbol tables
    #################################
    def vpermute(self, a,b):
        '''
        permute matrices a,b
        assume a,b with same dtype
        '''
        assert(a.dtype == b.dtype)
        p = self.p
        Na = a.shape[0]
        Nb = b.shape[0]
        ones_a = np.ones((Na,1))
        ones_b = np.ones((Nb,1))
        mat_1 = np.kron(a, ones_b).astype(a.dtype)
        mat_2 = np.kron(ones_a, b).astype(a.dtype)
        # merge column wise (axis=1)
        mat_all = np.c_[mat_1, mat_2]
        return mat_all

    def build_source_tables(self):
        '''
        construct multi-stream tables recursively
        '''
        p = self.p
        mod = self.mod
        sym_vec, bit_mat = mod.get_const()
        if p.N_sts == 1:
            return sym_vec, bit_mat
        else: # N_sts > 1
            sym_a, sym_b = sym_vec, sym_vec
            bit_a, bit_b = bit_mat, bit_mat
            for i in range(p.N_sts - 1):
                sym_a = self.vpermute(sym_a, sym_b)
                bit_a = self.vpermute(bit_a, bit_b)
            return sym_a, bit_a

    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        compute exact LLRs from compatible Y,H
        assume noise variance given in params
        NOTE: noise variance is needed to compute exact LLRs.
              In the case of MI-estimation and min-sum decoding,
              the LLRs can be left unscaled without degrading
              performance.
        '''
        p = self.p
        exact_llr = not self.maxlog_approx
        sym_sets_1 = self.sym_sets_1
        sym_sets_0 = self.sym_sets_0
        N = y_tsr.shape[0]

        # exact llr computation
        ########################
        n_bits = p.nbps * p.N_sts
        lambda_mat = np.zeros((N, n_bits))
        for ni in range(n_bits):

            # define l2_norm()
            l2_norm = partial(np.linalg.norm, ord=2, axis=2)

            # implement quad_mat_x using tensor broadcasting
            h_tsr_ex  = h_tsr[:,np.newaxis,:,:]
            syms_1 = sym_sets_1[ni]
            syms_1_ex = syms_1[np.newaxis,:,:,np.newaxis]
            syms_0 = sym_sets_0[ni]
            syms_0_ex = syms_0[np.newaxis,:,:,np.newaxis]
            y  = y_tsr[:,np.newaxis,:,:]

            # limited precision formulation for log_sum_exp
            scale = 1/(2*p.n_var)
            hs_1 = h_tsr_ex @ syms_1_ex
            quad_mat_1 = - l2_norm(y - hs_1)**2
            quad_mat_1 = np.squeeze(quad_mat_1, axis=2)
            qt_max_1 = np.amax(quad_mat_1, axis=1)
            if exact_llr:
                quad_mat_adj_1 = quad_mat_1 - qt_max_1[:,np.newaxis]
                exp_mat_adj_1 = np.exp( scale * quad_mat_adj_1 )
                sum_exp_adj_1 = np.sum(exp_mat_adj_1, axis=1)
                log_sum_exp_1 = scale * qt_max_1 + np.log(sum_exp_adj_1)
            else:
                log_sum_exp_1 = scale * qt_max_1

            hs_0 = h_tsr_ex @ syms_0_ex
            quad_mat_0 = - l2_norm(y - hs_0)**2
            quad_mat_0 = np.squeeze(quad_mat_0, axis=2)
            qt_max_0 = np.amax(quad_mat_0, axis=1)
            if exact_llr:
                quad_mat_adj_0 = quad_mat_0 - qt_max_0[:,np.newaxis]
                exp_mat_adj_0 = np.exp( scale * quad_mat_adj_0 )
                sum_exp_adj_0 = np.sum(exp_mat_adj_0, axis=1)
                log_sum_exp_0 = scale * qt_max_0 + np.log(sum_exp_adj_0)
            else:
                log_sum_exp_0 = scale * qt_max_0

            lambda_vec = log_sum_exp_0 - log_sum_exp_1;

            lambda_mat[:,ni] = lambda_vec

        return lambda_mat

class OneBitMLDemod(Demodulator):
    '''
    Compute the LLRs for ML 1-bit receiver
    '''

    def __init__(self, p,
                 modulator = None,
                 maxlog_approx = False
                 ):
        super().__init__(p, modulator, maxlog_approx)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        compute exact LLR for 1-bit receiver
        '''
        pass

################################################################################
# symbol detector classes
################################################################################
class SymbolEstimator:
    '''
    Symbol estimation (perform recovery of x_hat)
    Assume Linear Gaussian Channel
    '''
    def __init__(self, p, mode='mmse'):
        self.p = p
        self.mode = mode
        self.est = partial(self.estimators[mode], self)
        print(f'Symbol Estimator mode = {mode}')

    def zf_est(self, y_tsr, h_tsr, n_var_tsr):
        h_inv_tsr = la.pinv(h_tsr)
        w = h_inv_tsr
        return w @ y_tsr

    def mmse_est(self, y_tsr, h_tsr, n_var_tsr):

        def mmse_weight(h_mat, n_var):
            ''' matrix.H is a property function '''
            A = np.matrix(h_mat)
            N_tx = h_mat.shape[1]
            I = n_var * np.identity(N_tx)
            return la.inv(A.H @ A + I) * A.H

        w = [ mmse_weight(h_mat, n_var) for (h_mat, n_var) in zip(h_tsr, n_var_tsr) ]
        return w @ y_tsr

    # estimators
    estimators = {
        'mmse' : mmse_est,
        'zf'   : zf_est,
    }

    def estimate(self, y_tsr, h_tsr, n_var_tsr):
        '''
        call estimator function
        '''
        return self.est(y_tsr, h_tsr, n_var_tsr)

################################################################################
# demapper classes
################################################################################
class GaussianDemapper:
    '''
    Implements demapping in the symbol space for
    symbol detector evalution.
    Based on the Demodulator class.

    NOTE: demaps one stream at a time
    '''
    def __init__(self, p,
                 modulator = None,
                 maxlog_approx = False
                 ):
        self.p = p
        self.maxlog_approx = maxlog_approx
        if modulator:
            self.mod = modulator
        else:
            # assume QAM used
            self.mod = QAMModulator(p.M)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        sym_mat, bit_mat = self.build_source_tables()
        # construct symbol subsets
        n_bits = p.nbps * p.N_sts
        sym_sets_1 = []
        sym_sets_0 = []
        for i in range(n_bits):
            idx_1 = (bit_mat[:,i] == 1)
            idx_0 = (bit_mat[:,i] == 0)
            sym_sets_1.append( sym_mat[idx_1,:] )
            sym_sets_0.append( sym_mat[idx_0,:] )
        # save them up
        self.sym_sets_1 = sym_sets_1
        self.sym_sets_0 = sym_sets_0

    # multi-stream bv/symbol tables
    #################################
    def vpermute(self, a,b):
        '''
        permute matrices a,b
        assume a,b with same dtype
        '''
        assert(a.dtype == b.dtype)
        p = self.p
        Na = a.shape[0]
        Nb = b.shape[0]
        ones_a = np.ones((Na,1))
        ones_b = np.ones((Nb,1))
        mat_1 = np.kron(a, ones_b).astype(a.dtype)
        mat_2 = np.kron(ones_a, b).astype(a.dtype)
        # merge column wise (axis=1)
        mat_all = np.c_[mat_1, mat_2]
        return mat_all

    def build_source_tables(self):
        '''
        construct multi-stream tables recursively
        '''
        p = self.p
        mod = self.mod
        sym_vec, bit_mat = mod.get_const()
        if p.N_sts == 1:
            return sym_vec, bit_mat
        else: # N_sts > 1
            sym_a, sym_b = sym_vec, sym_vec
            bit_a, bit_b = bit_mat, bit_mat
            for i in range(p.N_sts - 1):
                sym_a = self.vpermute(sym_a, sym_b)
                bit_a = self.vpermute(bit_a, bit_b)
            return sym_a, bit_a


    def l2_norm_llrs(self, x_hat_tsr, n_var_tsr):
        '''
        compute single stream LLRs based on l2-norm

        Gaussian noise assumption:
            Thus we compute the matrix |x_hat - x|^2 where
        '''
        p = self.p
        exact_llr = not self.maxlog_approx
        sym_sets_1 = self.sym_sets_1
        sym_sets_0 = self.sym_sets_0
        N = x_hat_tsr.shape[0]

        # TODO: assert y_tsr and syms have the same dimensions
        import pdb; pdb.set_trace()

        # exact llr computation
        ########################
        #n_bits = p.nbps * p.N_sts
        n_bits = p.nbps * p.N_sts
        lambda_mat = np.zeros((N, n_bits))
        for ni in range(n_bits):

            # define l2_norm()
            l2_norm = partial(np.linalg.norm, ord=2, axis=2)

            # implement quad_mat_x using tensor broadcasting
            #h_tsr_ex  = h_tsr[:,np.newaxis,:,:]
            syms_1 = sym_sets_1[ni]
            syms_1_ex = syms_1[np.newaxis,:,:,np.newaxis]
            syms_0 = sym_sets_0[ni]
            syms_0_ex = syms_0[np.newaxis,:,:,np.newaxis]
            #y  = y_tsr[:,np.newaxis,:,:]
            x_hat  = x_hat_tsr[:,np.newaxis,:,:]

            # limited precision formulation for log_sum_exp
            scale = 1/(2*p.n_var)
            #hs_1 = h_tsr_ex @ syms_1_ex
            x_1 = syms_1_ex
            quad_mat_1 = - l2_norm(x_hat - x_1)**2
            quad_mat_1 = np.squeeze(quad_mat_1, axis=2)
            qt_max_1 = np.amax(quad_mat_1, axis=1)
            if exact_llr:
                quad_mat_adj_1 = quad_mat_1 - qt_max_1[:,np.newaxis]
                exp_mat_adj_1 = np.exp( scale * quad_mat_adj_1 )
                sum_exp_adj_1 = np.sum(exp_mat_adj_1, axis=1)
                log_sum_exp_1 = scale * qt_max_1 + np.log(sum_exp_adj_1)
            else:
                log_sum_exp_1 = scale * qt_max_1

            #hs_0 = h_tsr_ex @ syms_0_ex
            x_0 = syms_0_ex
            quad_mat_0 = - l2_norm(x_hat - x_0)**2
            quad_mat_0 = np.squeeze(quad_mat_0, axis=2)
            qt_max_0 = np.amax(quad_mat_0, axis=1)
            if exact_llr:
                quad_mat_adj_0 = quad_mat_0 - qt_max_0[:,np.newaxis]
                exp_mat_adj_0 = np.exp( scale * quad_mat_adj_0 )
                sum_exp_adj_0 = np.sum(exp_mat_adj_0, axis=1)
                log_sum_exp_0 = scale * qt_max_0 + np.log(sum_exp_adj_0)
            else:
                log_sum_exp_0 = scale * qt_max_0

            lambda_vec = log_sum_exp_0 - log_sum_exp_1;

            lambda_mat[:,ni] = lambda_vec

        return lambda_mat

    def compute_llrs(self, x_hat_tsr, h_tsr, n_var_tsr):
        '''
        use l2-norm based method only
        '''
        return self.l2_norm_llrs(x_hat_tsr, n_var_tsr)


################################################################################
# modulator classes
################################################################################
class QAMModulator:
    '''
    Implements (square) QAM modulation with gray-coding
    This matches the WIFI/LTE standard
    '''

    # define mapper functions
    def map_bpsk(self, bit_vec):
        bit_vec = bit_vec.astype(float)
        syms = 2*bit_vec - 1;
        syms = syms.reshape(-1)
        return syms

    def map_qpsk(self, bit_vec):
        bit_vec = bit_vec.astype(float)
        bv = np.reshape(bit_vec, (-1,2))
        bv_i = bv[:,0]
        bv_q = bv[:,1]
        syms = (2*bv_i - 1) + 1j*(2*bv_q - 1)
        syms = syms.reshape(-1)
        return syms

    def map_16qam(self, bit_vec):
        M = self.M
        lut = type(self).sym_luts[M]

        bv = np.reshape(bit_vec, (-1,4))
        bv_i = bv[:,0:2]
        bv_q = bv[:,2:4]
        dv_i = bv2dec(bv_i)
        dv_q = bv2dec(bv_q)
        syms = lut[dv_i] + 1j*lut[dv_q]
        syms = syms.reshape(-1)

        return syms

    def map_64qam(self, bit_vec):
        M = self.M
        lut = type(self).sym_luts[M]

        bv = np.reshape(bit_vec, (-1,6))
        bv_i = bv[:,0:3]
        bv_q = bv[:,3:6]
        dv_i = bv2dec(bv_i)
        dv_q = bv2dec(bv_q)
        syms = lut[dv_i] + 1j*lut[dv_q]
        syms = syms.reshape(-1)

        return syms

    def map_256qam(self, bit_vec):
        M = self.M
        lut = type(self).sym_luts[M]

        bv = np.reshape(bit_vec, (-1,8))
        bv_i = bv[:,0:4]
        bv_q = bv[:,4:8]
        dv_i = bv2dec(bv_i)
        dv_q = bv2dec(bv_q)
        syms = lut[dv_i] + 1j*lut[dv_q]
        syms = syms.reshape(-1)

        return syms

    # define mapper table
    mappers = {
    #   M : mapper
        2 : map_bpsk,
        4 : map_qpsk,
       16 : map_16qam,
       64 : map_64qam,
      256 : map_256qam,
    }

    # define scale table
    kmods = {
    #   M : kmod
        2 : 1,
        4 : 1/np.sqrt(2),
       16 : 1/np.sqrt(10),
       64 : 1/np.sqrt(42),
      256 : 1/np.sqrt(170),
    }

    # symbol lookup tables
    sym_luts = {
    #   M : lut
       16 : np.array([-3,-1, 3, 1]),
       64 : np.array([-7,-5,-1,-3, 7, 5, 1, 3]),
      256 : np.array([-15,-13,- 9,-11,- 1,- 3,- 7,- 5, 15, 13,  9, 11,  1,  3,  7,  5]),
    }

    # mod strings
    mod_strs = {
    #   M :  str
        2 : 'BPSK',
        4 : 'QPSK',
       16 : '16QAM',
       64 : '64QAM',
      256 : '256QAM',
    }

    def __init__(self, M):
        self.M = M
        self.nbps = np.log2(M).astype(int)
        self.mapper = type(self).mappers[M]
        self.kmod = type(self).kmods[M]

    def map(self, bit_vec, normalize=True):
        '''
        map MSB-first bit vectors to complex symbols
        returns symbol vector as 1d-array
        '''
        mapper = self.mapper
        kmod = self.kmod
        nbps = self.nbps

        # divisible by nbps (remainder == 0)
        assert( bit_vec.size % nbps == 0 )

        syms = mapper(self,bit_vec)

        if normalize: syms = kmod * syms

        return syms

    def get_const(self, normalize=True):
        '''
        generate symbol constellation and corresponding bit vectors
        NOTE: returns 2d-arrays explicitly
        '''
        M = self.M
        kmod = self.kmod
        nbps = self.nbps
        mapper = self.mapper

        x = np.arange(M)
        xm = x.reshape(-1,1)
        bm = dec2bv(xm)
        bv = bm.reshape(-1,1)
        syms = mapper(self,bv)
        syms = syms.reshape(-1,1)

        if normalize: syms = kmod * syms

        return syms, bm

    def plot_constellation(self):
        M = self.M
        nbps = self.nbps
        mapper = self.mapper

        x = np.arange(M)
        xm = x.reshape(-1,1)
        bm = dec2bv(xm)
        bv = bm.reshape(-1,1)
        y = mapper(self,bv)

        import matplotlib.pyplot as plt
        print('plotting constellation')
        plt.scatter(y.real, y.imag, s=10)
        plt.title( type(self).mod_strs[M] + ' constellation' )
        bin_txt = [np.binary_repr(i, width=nbps) for i in list(x)]
        [ plt.text(i+0.1, q, txt) for (i, q, txt) in zip(y.real, y.imag, bin_txt) ]
        plt.show()

