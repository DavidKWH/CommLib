# placeholder for all comm. related utilities for now
# [TODO] add channel class
# [DONE] move CommDataSource to CommLib
# [DONE] need to match matlab functions' behavior by default
#  NOTE: matlab does not implement standard defined Gray coding
#        commpy implements binary encoding

# general dependencies
import numpy as np
import numpy.random as rnd
# for crandn
from numpy.random import randn
# helper functions
from functools import partial

def crandn(*args):
    '''
    standard complex normal random samples
    arguments compatible with randn()
    '''
    samps = np.sqrt(0.5) * (randn(*args) + randn(*args) * 1j)
    return samps

def cplx2reals(a):
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
    via low level unpackbits(), 8-bits max
    NOTE: bit vectors are MSB-first (e.g. 6 => '110')
    '''
    dec_vec = dec_vec.astype(np.uint8)
    max_val = max(dec_vec).item(0) # get only element in array
    nbits = np.floor(np.log2(max_val) + 1).astype(int)

    if n: nbits = n
    assert( nbits <= 8 )

    bit_mat = np.unpackbits(dec_vec, axis=1)

    return bit_mat[:,-nbits:]

def bv2dec(bit_mat):
    '''
    convert MSB-first binary (row) vectors to unsigned decimals
    using low level packbits(), 8-bits max
    NOTE: bit vectors are MSB-first (e.g. 6 => '110')
    '''
    (rows, cols) = bit_mat.shape

    assert( cols <= 8 )

    n_pad = 8 - cols
    bit_mat_list = list(bit_mat)
    pad_mat_list = [np.pad(bit_vec, (n_pad, 0), 'constant')
                    for bit_vec in bit_mat_list]
    pad_mat = np.array(pad_mat_list)
    dec_vec = np.packbits(pad_mat, axis=1)

    return dec_vec

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

class Transmitter:
    '''
    Encapsulate all TX functions
    Support multi-stream processing
    '''

    def __init__(self, p,
                 encoder = None,
                 modulator = None
                 ):
        assert(modulator)
        self.p = p
        self.mod = modulator
        self.enc = encoder

    def __call__(self):
        return self.generate()

    def generate(self):
        '''
        generate raw payload bits and symbols
        NOTE: output bit vectors as list
              easier to handle
        '''
        p = self.p
        mod = self.mod
        enc = self.enc

        # per stream processing
        ####################################
        if enc == None:
            raw_bit_tsr = rnd.randint(2, size=(p.N_sts, p.N_syms, p.nbps))
            bit_mat = raw_bit_tsr.reshape(p.N_sts, -1)
            # return raw bits
        else:
            raw_bit_tsr = rnd.randint(2, size=(p.N_sts, p.N_raw, p.nbps))
            raw_bit_mat = raw_bit_tsr.reshape(p.N_sts, -1)
            # encode raw bits, produces N_sts codewords
            bits_list = [ enc.encode(raw_bits) for raw_bits in list(raw_bit_mat) ]
            bit_mat = np.array(bits_list)
            bits_tsr = bit_mat.reshape(p.N_sts, p.N_syms, p.nbps)

        # map to symbols
        syms_list = [ mod.map(bits) for bits in list(bit_mat) ]
        syms = np.array(syms_list)

        # convert to stacked matrix form
        # for element-wise matrix multiplication using @
        # i.e., the first dimension is turned into a list
        #       containing elements in the remaining dimensions
        # NOTE: a @ b <=> np.matmul(a,b)
        # syms.shape = (N_sts, N_syms)
        syms_tr = syms.transpose()
        # syms_tr.shape = (N_syms, N_sts)
        syms_tsr = syms_tr.reshape(p.N_syms, p.N_sts, 1)

        bits_mat = raw_bit_tsr.reshape(p.N_sts, -1)
        bits_list = list(bits_mat)

        return syms_tsr, bits_list


class Receiver:
    '''
    Encapsulate demod and decoding
    '''

    def __init__(self,p,
                 demodulator = None,
                 decoder = None,
                 ):
        assert(demodulator)
        self.p = p
        self.demod = demodulator
        self.dec = decoder

    def __call__(self, y_tsr, h_tsr):
        return self.detect(y_tsr, h_tsr)

    def detect(self, y_tsr, h_tsr):
        '''
        when decoder is provided, returns hard decisions and llrs
        when decoder is absent, returns decoded bits and iterations
        NOTE: return bit vectors as list easier to handle
        '''
        p = self.p
        demod = self.demod
        dec = self.dec

        llr_mat = demod.compute_llrs(y_tsr, h_tsr)
        llr_tsr = llr_mat.reshape(p.N_syms, p.N_sts, p.nbps)
        # reshuffle llrs matrix
        # llrs_list contains N_sts llr vectors
        llrs_list = np.split(llr_tsr, p.N_sts, axis=1)
        llrs_list = [ llrs.reshape(-1) for llrs in llrs_list ]

        if demod == None:
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
            dec_bits_list = [ llrs[:p.K] for llrs in dec_bits_list ]
            return dec_bits_list, iter_list


class Channel:
    '''
    Encapsulate channel and noise generation, application
    '''

    def __init__(self, p):
        self.p = p
        self.ch_gen = partial(self.channels[p.ch_type], self)

    def __call__(self, syms):
        return self.apply(syms)

    '''
    channel generation functions
    '''
    def gen_rayleigh_ch(self, N, N_tx, N_rx):
        return crandn(N, N_tx, N_rx)

    def gen_identity_ch(self, N, N_tx, N_rx):
        H = np.identity(N_tx).astype(complex)
        return np.tile(H,(N,1)).reshape(N, N_tx, N_rx)

    # channel selection
    channels = {
        'identity' : gen_identity_ch,
        'rayleigh' : gen_rayleigh_ch
    }

    def apply(self, syms_tsr):
        p = self.p
        ch_gen = self.ch_gen

        # sym_tsr.shape = (N, N_rx, 1)
        N = syms_tsr.shape[0]
        # generate channel realizations
        h_tsr = ch_gen(N, p.N_tx, p.N_rx)
        # genrate noise vectors
        n_tsr = crandn(N, p.N_rx, 1)
        # element-wise matrix multiplication via @
        y_tsr = h_tsr @ syms_tsr + p.n_std * n_tsr
        return y_tsr, h_tsr


class CommDataSource:
    '''
    Generate data over a complex Gaussian channel
    Support multiple streams, N_tx < N_rx
    [TODO] add other channel scenarios
    [TODO] reuse Channel, Transmitter classes

    p must contain all relevant parameters
    '''

    def __init__(self, p):
        # save parameters
        self.p = p
        self.mod = QAMModulator(p.M)
        self.recv = Demodulator(p)
        self.ch_gen = partial(self.channels[p.ch_type], self)

    def __repr__(self):
        return "Communication data source"

    '''
    channel generation functions
    '''
    def gen_rayleigh_ch(self, N, N_tx, N_rx):
        return crandn(N, N_tx, N_rx)

    def gen_identity_ch(self, N, N_tx, N_rx):
        H = np.identity(N_tx).astype(complex)
        return np.tile(H,(N,1)).reshape(N, N_tx, N_rx)

    # channel selection
    channels = {
        'identity' : gen_identity_ch,
        'rayleigh' : gen_rayleigh_ch
    }

    def gen_channel_output(self, N):
        """ generate data for training """
        p = self.p
        mod = self.mod
        ch_gen = self.ch_gen

        # per stream processing
        # NOTE: array.reshape(-1) flatten array into 1-dim
        bit_tsr = rnd.randint(2, size=(p.N_sts, N, p.nbps))
        # bit_mat.shape = (N_sts, N*nbps)
        bit_mat = bit_tsr.reshape(p.N_sts, -1)
        syms_list = [ mod.map(bits) for bits in list(bit_mat) ]
        syms = np.array(syms_list)

        # reformat bits for training
        # list(bit_str) = N_sts elements of shape (N, p.nbps)
        bit_mat_trn = np.concatenate(list(bit_tsr), axis=1)
        # bit_mat_trn.shape = (N, N_sts * nbps)
        bit_tsr_trn = bit_mat_trn.reshape(N, p.N_sts, p.nbps)

        # convert to stacked matrix form
        # for element-wise matrix multiplication using @
        # i.e., the first dimension is turned into a list
        #       containing elements in the remaining dimensions
        # NOTE: a @ b <=> np.matmul(a,b)
        # syms.shape = (N_sts, N)
        syms_tr = syms.transpose()
        # syms_tr.shape = (N, N_sts)
        syms_tsr = syms_tr.reshape((N, p.N_sts, 1))
        # generate channel realizations
        h_tsr = ch_gen(N, p.N_tx, p.N_rx)
        # genrate noise vectors
        n_tsr = crandn(N, p.N_rx, 1)
        # element-wise matrix multiplication via @
        y_tsr = h_tsr @ syms_tsr + p.n_std * n_tsr

        return bit_tsr_trn, h_tsr, y_tsr

    def gen_receive_llrs(self, N):
        """
        compute exact llrs
        see Demodulator:compute_llrs()
        """
        recv = self.recv

        bit_tsr, h_tsr, y_tsr = self.gen_channel_output(N)

        bit_mat = bit_tsr.reshape(N,-1) # flatten tensor
        lambda_mat = recv.compute_llrs(y_tsr, h_tsr, N)

        return bit_mat, lambda_mat

    def gen_all_signals(self, N):
        """
        return channel output and llrs
        """
        recv = self.recv

        bit_tsr, h_tsr, y_tsr = self.gen_channel_output(N)

        lambda_mat = recv.compute_llrs(y_tsr, h_tsr, N)

        return bit_tsr, h_tsr, y_tsr, lambda_mat

    def get_const(self):
        """
        returns constellation symbols and bit vectors
        see QAMModulator:get_const()
        """
        mod = self.mod
        return mod.get_const()


class Demodulator:
    '''
    Computes exact LLRs given transmit alphabets,
    receive symbols and exact channel knowledge
    '''

    def __init__(self, p,
                 modulator = None,
                 ):
        self.p = p
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

    def compute_llrs(self, y_tsr, h_tsr, N=None, scaling=True):
        '''
        compute exact LLRs from compatible Y,H
        assume noise variance given in params
        NOTE: noise variance is needed to compute exact LLRs.
              In the case of MI-estimation and min-sum decoding,
              the LLRs can be left unscaled without degrading
              performance.
        '''
        p = self.p
        sym_sets_1 = self.sym_sets_1
        sym_sets_0 = self.sym_sets_0

        if N == None: N = p.N_syms

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
            quad_mat_adj_1 = quad_mat_1 - qt_max_1[:,np.newaxis]
            exp_mat_adj_1 = np.exp( scale * quad_mat_adj_1 )
            sum_exp_adj_1 = np.sum(exp_mat_adj_1, axis=1);
            log_sum_exp_1 = scale * qt_max_1 + np.log(sum_exp_adj_1);

            hs_0 = h_tsr_ex @ syms_0_ex
            quad_mat_0 = - l2_norm(y - hs_0)**2
            quad_mat_0 = np.squeeze(quad_mat_0, axis=2)
            qt_max_0 = np.amax(quad_mat_0, axis=1)
            quad_mat_adj_0 = quad_mat_0 - qt_max_0[:,np.newaxis]
            exp_mat_adj_0 = np.exp( scale * quad_mat_adj_0 )
            sum_exp_adj_0 = np.sum(exp_mat_adj_0, axis=1);
            log_sum_exp_0 = scale * qt_max_0 + np.log(sum_exp_adj_0);

            lambda_vec = log_sum_exp_0 - log_sum_exp_1;

            lambda_mat[:,ni] = lambda_vec

        return lambda_mat

