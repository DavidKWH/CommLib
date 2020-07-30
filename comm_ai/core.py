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
        when decoder is present, returns decoded bits, llrs and num iterations
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
            x_hat_tsr, covar_tsr = est.estimate(y_tsr, h_tsr, n_var_tsr)
            llr_mat = demap.compute_llrs(x_hat_tsr, h_tsr, covar_tsr)

        # matrix dimensions
        # llr_mat.shape = [N_syms, N_sts * nbps]

        n_bits = p.N_sts * p.nbps

        llr_tsr = llr_mat.reshape(n_bits, p.dec.N)
        llrs_list = list(llr_tsr)

        if dec == None:
            # generate hard decisions
            dec_bits_list = [ llrs < 0  for llrs in llrs_list ]
            return dec_bits_list, llrs_list, None
        else:
            # decode llrs
            dec_out = [ dec.decode(llrs) for llrs in llrs_list ]
            # extract columns (decoded_bits, iterations)
            columns = list(zip(*dec_out))
            dec_bits_list = columns[0]
            iter_list = columns[1]
            # remove parity bits
            dec_bits_list = [ llrs[:p.dec.K] for llrs in dec_bits_list ]
            return dec_bits_list, llrs_list, iter_list


################################################################################
# high level transmitter class
################################################################################
class SymbolTransmitter:
    '''
    Symbol level transmitter (for SER)
    '''
    def __init__(self, p,
                 modulator = None,
                 training = False,
                 ):

        assert(modulator)

        self.p = p
        self.mod = modulator
        self.training = training

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, N=None, debug=False):
        '''
        generate raw payload bits and symbols
        '''
        p = self.p
        mod = self.mod
        training = self.training
        tensor_output = training or debug

        # generate output symbols
        N_raw = p.N if N == None else N
        raw_bit_tsr = rnd.randint(2, size=(N_raw, p.N_sts * p.nbps))
        raw_bit_mats = np.split(raw_bit_tsr, p.N_sts, axis=1)
        raw_bit_list = [ bit_mat.reshape(-1) for bit_mat in raw_bit_mats ]

        syms_list = [ mod.map(bits) for bits in raw_bit_list ]
        syms_mat = np.array(syms_list)
        # transpose array
        syms_mat_tr = np.transpose(syms_mat)
        syms_tsr = syms_mat_tr[:,:, None]

        # return symbols, raw bits
        return syms_tsr, raw_bit_tsr


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

    def generate(self, N=None, debug=False):
        '''
        generate raw payload bits and symbols

        Three mode exists:

            1. Random uncoded payload generated symbols
            2. Random encoded payload generated symbols
            3. Special mode that allows user to specify
               the number of symbols

        Recommend using mode 1 for uncoded BER sims, mode 2 for coded BER
        ims.  Use mode 3 for analysis related to the distribution such as
        MI and raw distribution of LLRs.

        In the simplest case, return bit vectors as list for BER sims.
        '''
        p = self.p
        mod = self.mod
        enc = self.enc
        training = self.training
        tensor_output = training or debug

        #NOTE: allow N only in training|debug modes
        assert( not(tensor_output == False and N != None) )
        # sanity check, should never pass in N if encoder used
        #assert( (enc and N == None) or (not enc and N) )

        # per stream processing
        ####################################
        if enc == None:
            # Use N from function argument if encoder not defined
            # otherwise, use the N_syms parameter
            N_raw = p.dec.N if N == None else N
            #raw_bit_tsr = rnd.randint(2, size=(p.N_sts, N_raw, p.nbps))
            #raw_bit_mat = raw_bit_tsr.reshape(p.N_sts, -1)
            raw_bit_tsr = rnd.randint(2, size=(p.N_sts * p.nbps, N_raw))
            raw_bit_mat = raw_bit_tsr.reshape(p.N_sts * p.nbps, -1)
            raw_bits_list = list(raw_bit_mat)
            mapper_bits_list = raw_bits_list
        else:
            # compute N_raw using encoder/decoder parameters
            N_raw = p.dec.K
            raw_bit_tsr = rnd.randint(2, size=(p.N_sts * p.nbps, N_raw))
            raw_bit_mat = raw_bit_tsr.reshape(p.N_sts * p.nbps, -1)
            raw_bits_list = list(raw_bit_mat)
            # encode raw bits, produces (N_sts * p.nbps) codewords
            encoded_bits_list = [ enc.encode(raw_bits) for raw_bits in raw_bits_list ]
            mapper_bits_list = encoded_bits_list

        # map to symbols
        N_syms = N if tensor_output else p.dec.N
        syms_list = [ mod.map(bits) for bits in mapper_bits_list ]
        syms = np.array(syms_list)
        # map CW across symbols
        syms = syms.reshape(-1, p.N_sts)

        # Dimensions of arrays:
        # mapper_bits.shape = ( N_sts, dec.N )
        # syms.shape = ( N_syms, N_sts )
        # syms.shape = ( N_sts x N_syms )

        sym_tsr = syms.reshape(N_syms, p.N_sts, 1)

        # output processing
        # NOTE: bits_output format depends on whether training and debug mode
        #       check then tensor_output code above
        if tensor_output:
            ### we are in training mode ###
            # note that bits are mapped naturally across symbols
            bit_mat_trn = raw_bit_mat.reshape(-1, p.N_sts * p.nbps)
            # bit_mat_trn.shape = (N, N_sts * nbps)
            bits_output = bit_mat_trn
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
        self.ch_file = pm.ch_file if pm.ch_type == 'fixed' else None

    def __call__(self, syms):
        return self.apply(syms)

    '''
    channel generation functions
    '''
    def gen_diagonal_ch(self, N, N_tx, N_rx):
        p = self.p
        N_min = min(N_rx, N_tx)
        # generate random sv's
        sv_mat = rnd.uniform(low=p.u_a, high=p.u_b, size=(N,N_min))
        # allocate memory for chnannels
        H_tsr = np.zeros((N, N_rx, N_tx))
        # in-place assignment
        for (H, sv) in zip(H_tsr, sv_mat):
            H[:N_min,:N_min] = np.diag(sv)

        return H_tsr

    def gen_ch_from_sv_dist(self, N, N_tx, N_rx):
        '''generate matrices with specific sv distribution for DNN training'''
        from scipy.stats import unitary_group
        p = self.p
        N_min = min(N_rx, N_tx)

        # generate random sv's
        sv_mat = rnd.uniform(low=p.u_a, high=p.u_b, size=(N,N_min))
        # generate random unitary matrices
        U_tsr = unitary_group.rvs(dim=N_rx, size=N)
        V_tsr = unitary_group.rvs(dim=N_tx, size=N)
        S_tsr = np.zeros((N, N_rx, N_tx))
        # in-place assignment
        for (S, sv) in zip(S_tsr, sv_mat):
            S[:N_min,:N_min] = np.diag(sv)

        H = U_tsr @ S_tsr @ V_tsr

        return H

    def gen_rayleigh_ch(self, N, N_tx, N_rx):
        '''
        scale channel such that the noiseless
        received signal at each antennae is 1

        Add batch_fixed mode for DNN training
        '''
        scale = 1 / np.sqrt(N_tx)

        if self.batch_fixed:
            # repeat H N-times
            H = scale * crandn(N_rx, N_tx)
            H_tsr = np.tile(H, (N,1,1))
            # save to file
            #print('saving channel instance to file')
            #with open('channel.npy', 'wb') as f:
            #    np.save(f, H);
        else:
            H_tsr = scale * crandn(N, N_rx, N_tx)

        return H_tsr

    def gen_fixed_ch(self, N, N_tx, N_rx):
        '''
        load fixed channel from file
        Channel is already scaled
        '''
        ch_file = self.ch_file
        with open(ch_file, 'rb') as f:
            H = np.load(f);
        H_tsr = np.tile(H, (N,1,1))
        return H_tsr

    def gen_identity_ch(self, N, N_tx, N_rx):
        N_min = min(N_rx, N_tx)
        H = np.zeros((N_rx, N_tx), dtype=complex)
        H[:N_min,:N_min] = np.identity(N_min)
        return np.tile(H,(N,1)).reshape(N, N_rx, N_tx)


    # channel selection
    channels = {
        'diagonal' : gen_diagonal_ch,
        'sv_dist'  : gen_ch_from_sv_dist,
        'identity' : gen_identity_ch,
        'rayleigh' : gen_rayleigh_ch,
        'fixed'    : gen_fixed_ch,
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

    def gen_fixed_snr(self, N, N_rx):
        p = self.p
        n_std = 10**(-p.snr_db/20)
        std_n_tsr = crandn(N, N_rx, 1)
        n_tsr = n_std * std_n_tsr
        snr_tsr_db = p.snr_db * np.ones((N,1,1))
        return n_tsr, snr_tsr_db

    def gen_rand_snr(self, N, N_rx):
        p = self.p
        snr_tsr_db = rnd.randint(p.snr_lo, p.snr_hi+1, size=(N,1,1))
        n_std_tsr = 10**(-snr_tsr_db/20)
        std_n_tsr = crandn(N, N_rx, 1)
        n_tsr = n_std_tsr * std_n_tsr
        return n_tsr, snr_tsr_db

    # noise generation
    noise_table = {
        'fixed_var' : gen_fixed_var,
        'rand_var' : gen_rand_var,
        'fixed_snr' : gen_fixed_snr,
        'rand_snr' : gen_rand_snr,
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
class SymbolDetector:
    '''
    Compute the bit pattern corresponding to the
    ML symbol.  (for hard detection)
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
        # save sym and bit vector table
        self.sym_mat = sym_mat
        self.bit_mat = bit_mat

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
        return self.compute_ml(*args, **kwargs)

    def compute_ml(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        compute ML solution from compatible Y,H
        NOTE: assume noise variance given in params
        NOTE: noise variance is needed to compute exact LLRs.
              In the case of MI-estimation and min-sum decoding,
              the LLRs can be left unscaled without degrading
              performance.
        '''
        p = self.p
        N = y_tsr.shape[0]
        syms = self.sym_mat
        bits = self.bit_mat

        ## compute squared distance
        #############################

        # define l2_norm()
        l2_norm = partial(np.linalg.norm, ord=2, axis=2)

        # implement quad_mat_x using tensor broadcasting
        h_tsr_ex  = h_tsr[:,np.newaxis,:,:]
        syms_ex = syms[np.newaxis,:,:,np.newaxis]
        y = y_tsr[:,np.newaxis,:,:]

        hs = h_tsr_ex @ syms_ex
        quad_mat = l2_norm(y - hs)**2
        quad_mat = np.squeeze(quad_mat, axis=2)
        min_idx = np.argmin(quad_mat, axis=1)

        # return the ML solutions
        syms_ml = syms[min_idx,:]
        bits_ml = bits[min_idx,:]

        # return ML solutions
        return bits_ml, syms_ml


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
        NOTE: assume noise variance given in params
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
            scale = 1/p.n_var
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

################################################################################
# symbol detector classes
################################################################################
class SymbolEstimator:
    '''
    Symbol estimation (perform recovery of x_hat)
    Assume Linear Gaussian Channel, i.e.
        y = Hx + n

    Inputs:
        h_tsr:     channel realizations
        n_var_tsr: noise variance of n
    '''
    def __init__(self, p, mode='mmse'):
        self.p = p
        self.mode = mode
        self.est = partial(self.estimators[mode], self)
        print(f'Symbol Estimator mode = {mode}')

    def covar(self, w_mat, n_var):
        ''' assume noise variance is n_var * I '''
        n_var = n_var.item() # one item in array
        A = np.matrix(w_mat)
        Sigma = n_var * A @ A.H
        return Sigma

    def zf_est(self, y_tsr, h_tsr, n_var_tsr):

        h_pinv_tsr = la.pinv(h_tsr)
        w_tsr = h_pinv_tsr
        x_hat_tsr = w_tsr @ y_tsr

        return x_hat_tsr, w_tsr

    def mmse_est(self, y_tsr, h_tsr, n_var_tsr):
        '''
        Dimensions:
            y_tsr.shape     = (N, N_rx, 1)
            h_tsr.shape     = (N, N_rx, N_tx)
            n_var_tsr.shape = (N, 1, 1)
        '''
#        def mmse_weight(h_mat, n_var):
#            ''' matrix.H is a property function '''
#            A = np.matrix(h_mat)
#            N_tx = h_mat.shape[1]
#            I = n_var * np.identity(N_tx)
#            return la.inv(A.H @ A + I) @ A.H
#
#        w_tsr = [ mmse_weight(h_mat, n_var) for (h_mat, n_var) in zip(h_tsr, n_var_tsr) ]

        N_tx = h_tsr.shape[2]
        I = np.identity(N_tx)
        I = I[None,...] # for broadcasting
        H = h_tsr
        n_var = n_var_tsr

        # implement vectorized version of (see mmse_weight() above)
        # la.inv(H.H @ H + n_var * I) @ H.H
        H_herm = np.conj(H).swapaxes(1,2)
        w_tsr = la.inv(H_herm @ H + n_var * I) @ H_herm

        x_hat_tsr = w_tsr @ y_tsr

        return x_hat_tsr, w_tsr

    # estimators
    estimators = {
        'mmse' : mmse_est,
        'zf'   : zf_est,
    }

    def estimate(self, y_tsr, h_tsr, n_var_tsr):
        ''' calls estimator function '''
        x_hat_tsr, w_tsr = self.est(y_tsr, h_tsr, n_var_tsr)
        covar_tsr = [ self.covar(w_mat, n_var) for (w_mat, n_var) in zip(w_tsr, n_var_tsr) ]
        return x_hat_tsr, covar_tsr

################################################################################
# demapper classes
################################################################################
class GaussianDemapper:
    '''
    Implements demapping in the symbol space for
    symbol detector evalution.
    NOTE: demaps one stream at a time

    Ignores cross correlation between symbol estimates.

    Based on Demodulator class.
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
        n_bits = p.nbps
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
#    def vpermute(self, a,b):
#        '''
#        permute matrices a,b
#        assume a,b with same dtype
#        '''
#        assert(a.dtype == b.dtype)
#        p = self.p
#        Na = a.shape[0]
#        Nb = b.shape[0]
#        ones_a = np.ones((Na,1))
#        ones_b = np.ones((Nb,1))
#        mat_1 = np.kron(a, ones_b).astype(a.dtype)
#        mat_2 = np.kron(ones_a, b).astype(a.dtype)
#        # merge column wise (axis=1)
#        mat_all = np.c_[mat_1, mat_2]
#        return mat_all

    def build_source_tables(self):
        '''
        construct multi-stream tables recursively
        '''
        p = self.p
        mod = self.mod
        sym_vec, bit_mat = mod.get_const()
        return sym_vec, bit_mat
        #if p.N_sts == 1:
        #    return sym_vec, bit_mat
        #else: # N_sts > 1
        #    sym_a, sym_b = sym_vec, sym_vec
        #    bit_a, bit_b = bit_mat, bit_mat
        #    for i in range(p.N_sts - 1):
        #        sym_a = self.vpermute(sym_a, sym_b)
        #        bit_a = self.vpermute(bit_a, bit_b)
        #    return sym_a, bit_a

    def l2_norm_llrs(self, x_hat_tsr, n_var_tsr):
        '''
        compute single stream LLRs for an array of observed with
        associated noise variances.

        NOTE: unlike the receive side demod, n_var may vary with x_hat

        Gaussian noise assumption:
            Thus we compute the distance 1/n_var * |x_hat - x|^2
            for each candidate x
        '''
        p = self.p
        exact_llr = not self.maxlog_approx
        sym_sets_1 = self.sym_sets_1
        sym_sets_0 = self.sym_sets_0
        N = x_hat_tsr.shape[0]

        # n_var_tsr is the diagonal elements of
        # covariance matrices, should not be complex
        n_var_tsr = n_var_tsr.real

        # exact llr computation
        ########################
        n_bits = p.nbps
        lambda_mat = np.zeros((N, n_bits))
        for ni in range(n_bits):

            # define l2_norm()
            l2_norm = partial(np.linalg.norm, ord=2, axis=2)

            # implement quad_mat_x using tensor broadcasting
            syms_1 = sym_sets_1[ni]
            syms_1_ex = syms_1[np.newaxis,:,:,np.newaxis]
            syms_0 = sym_sets_0[ni]
            syms_0_ex = syms_0[np.newaxis,:,:,np.newaxis]
            x_hat  = x_hat_tsr[:,np.newaxis,:,:]

            # limited precision formulation for log_sum_exp
            scales = 1/n_var_tsr
            x_1 = syms_1_ex
            quad_mat_1 = - l2_norm(x_hat - x_1)**2
            quad_mat_1 = np.squeeze(quad_mat_1, axis=2)
            qt_max_1 = np.amax(quad_mat_1, axis=1, keepdims=True)
            if exact_llr:
                quad_mat_adj_1 = quad_mat_1 - qt_max_1
                exp_mat_adj_1 = np.exp( scales * quad_mat_adj_1 )
                sum_exp_adj_1 = np.sum(exp_mat_adj_1, axis=1, keepdims=True)
                log_sum_exp_1 = scales * qt_max_1 + np.log(sum_exp_adj_1)
            else:
                log_sum_exp_1 = scales * qt_max_1

            x_0 = syms_0_ex
            quad_mat_0 = - l2_norm(x_hat - x_0)**2
            quad_mat_0 = np.squeeze(quad_mat_0, axis=2)
            qt_max_0 = np.amax(quad_mat_0, axis=1, keepdims=True)
            if exact_llr:
                quad_mat_adj_0 = quad_mat_0 - qt_max_0
                exp_mat_adj_0 = np.exp( scales * quad_mat_adj_0 )
                sum_exp_adj_0 = np.sum(exp_mat_adj_0, axis=1, keepdims=True)
                log_sum_exp_0 = scales * qt_max_0 + np.log(sum_exp_adj_0)
            else:
                log_sum_exp_0 = scales * qt_max_0

            lambda_vec = log_sum_exp_0 - log_sum_exp_1;

            lambda_mat[:,ni] = np.squeeze(lambda_vec)

        return lambda_mat

    def compute_llrs(self, x_hat_tsr, h_tsr, covar_tsr):
        '''
        compute LLRs per stream (ignore cross correlation)

        Dimensions:
            x_hat_tsr.shape = (N, N_tx, 1)
            covar_tsr.shape = (N, N_tx, N_tx)

        '''
        axis = 1
        N_sections = x_hat_tsr.shape[axis]
        x_hats_list = np.split(x_hat_tsr, N_sections, axis=axis)
        n_var_list = [ np.diag(covar_mat) for covar_mat in covar_tsr ]
        n_var_tsr = np.array(n_var_list)
        n_vars_list = np.split(n_var_tsr, N_sections, axis=axis)
        # list of arrays with shape (N, N_bits)
        llrs_list = [ self.l2_norm_llrs(x_hats, n_vars) for (x_hats, n_vars) in
                            zip(x_hats_list, n_vars_list) ]
        llrs_concat = np.concatenate(llrs_list, axis=1)
        return llrs_concat


################################################################################
# modulator classes
################################################################################
class QAMModulator:
    '''
    Implements (square) QAM modulation with gray-coding
    This matches the WIFI/LTE standard

    Added 8PSK from 3GPP TS 45.004 version 9.0.0 Release 9
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

    def map_8psk(self, bit_vec):
        M = self.M
        lut = type(self).sym_luts[M]
        bv = np.reshape(bit_vec, (-1,3))
        dv = bv2dec(bv)
        syms = np.exp(1j*2.*np.pi*lut[dv]/8)
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
        8 : map_8psk,
       16 : map_16qam,
       64 : map_64qam,
      256 : map_256qam,
    }

    # define scale table
    kmods = {
    #   M : kmod
        2 : 1,
        4 : 1/np.sqrt(2),
        8 : 1,
       16 : 1/np.sqrt(10),
       64 : 1/np.sqrt(42),
      256 : 1/np.sqrt(170),
    }

    # symbol lookup tables
    sym_luts = {
    #   M : lut
        8 : np.array([ 3, 4, 2, 1, 6, 5, 7, 0]),
       16 : np.array([-3,-1, 3, 1]),
       64 : np.array([-7,-5,-1,-3, 7, 5, 1, 3]),
      256 : np.array([-15,-13,- 9,-11,- 1,- 3,- 7,- 5, 15, 13,  9, 11,  1,  3,  7,  5]),
    }

    # mod strings
    mod_strs = {
    #   M :  str
        2 : 'BPSK',
        4 : 'QPSK',
        8 : '8QPSK',
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

