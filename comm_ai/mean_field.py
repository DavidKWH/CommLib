# general dependencies
import numpy as np
import numpy.linalg as la
from numpy.random import default_rng
rng = default_rng()
# helper functions
from functools import partial

from .core import crandn
from .core import preproc_channel
from .core import cplx2reals
from .core import reals2cplx

from .core import QAMModulator
#from .core import PerSymbolDetector
#from .core import PerSymbolDetectorV2

from scipy.stats import norm
from scipy.special import logsumexp

# helper functions
def compute_qi(log_qi):

    qi_unnorm = np.exp(log_qi)
    qi_sum = np.sum(qi_unnorm, axis=1, keepdims=True)
    qi_norm = qi_unnorm / qi_sum
    return qi_norm

class OneBitSampMeanFieldDetector:
    '''
    Compute the q_i mean field update via the sampling method
    the 1-bit receiver
    '''
    def __init__(self, p,
                 modulator = None,
                 alpha = 1,
                 ):
        self.p = p
        self.alpha = alpha

        print(f'SMF: alpha = {alpha}')

        if modulator:
            self.mod = modulator
        else:
            # assume QAM used
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
        return self.compute_mf_updates(*args, **kwargs)

    def compute_mf_updates(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        Interface function to the internal compute_llrs_real().
        convert to real equivalent system

        y_tsr.shape = [ N x N_rx x 1 ]
        h_tsr.shape = [ N x N_rx x N_tx ]
        '''
        #def y_to_real(y_vec):
        #    return np.concatenate([y_vec.real, y_vec.imag], axis=0)

        #def h_to_real(h_mat):
        #    h_real_t = np.hstack([h_mat.real, -h_mat.imag])
        #    h_real_b = np.hstack([h_mat.imag,  h_mat.real])
        #    h_real = np.vstack([h_real_t, h_real_b])
        #    return h_real

        #n_var_real_tsr = n_var_tsr / 2.
        #n_var_real_tsr = None
        #y_real_list = [ y_to_real(y_vec) for y_vec in y_tsr ]
        #y_real_tsr = np.array(y_real_list)
        #h_real_list = [ h_to_real(h_mat) for h_mat in h_tsr ]
        #h_real_tsr = np.array(h_real_list)

        n_var_real_tsr = None
        y_tsr = np.squeeze(y_tsr, axis=2)
        y_real_tsr = cplx2reals(y_tsr)
        h_real_tsr = preproc_channel(h_tsr)

        # call internal mf function
        bits_mf, syms_mf_real = self.compute_mf_updates_real(y_real_tsr, h_real_tsr, n_var_real_tsr)
        syms_mf = reals2cplx(syms_mf_real)

        return bits_mf, syms_mf

    def compute_mf_updates_real(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        Inputs from real system at this point...

        run MF update on each instance of Y,H
        NOTE: assume noise variance given in params
        NOTE: noise variance is needed to compute exact LLRs.
              In the case of MI-estimation and min-sum decoding,
              the LLRs can be left unscaled without degrading
              performance.

        Dimensions:
            syms_x.shape = (N x M_tx)
            h_tsr.shape  = (N x M_rx x M_tx)

        '''
        mf_res_list = [ self.samp_mf_update(y_vec, h_mat) 
                        for y_vec, h_mat in zip(y_tsr, h_tsr) ]
        mf_res = np.array(mf_res_list)
        bits_mf, syms_mf = np.split(mf_res, 2, axis=1)
        bits_mf = bits_mf.squeeze()
        syms_mf = syms_mf.squeeze()

        return bits_mf, syms_mf

    def samp_mf_update(self, y, H):
        ''' compute sampling MF update for a single instance y,H '''
        p = self.p
        syms = self.sym_mat
        syms = syms.reshape(-1)
        bits = self.bit_mat
        L_max = 10
        N_samp = 10
        # damping factor
        alpha = self.alpha

        # compute scaling
        rho = 1/p.n_var
        sqrt_2rho = np.sqrt(2*rho)

        # other parameters
        N_tx_real = p.N_tx * 2
        N_syms_re = syms.size
        sqrt_M = np.sqrt(p.M).astype(int)

        def compute_log_p(y, H, s_tsr):
            shape = s_tsr.shape
            # s_tsr.shape = [2K , N_samp , N_syms_re]
            s_mat = s_tsr.reshape(shape[0], -1)
            # s_mat.shape = [2K , (N_samp , N_syms_re)]

            sG = sqrt_2rho * np.diag(y) @ H
            term = sG @ s_mat
            # numerically stable version
            log_p_mat = norm.logcdf(term)
            # log_p_mat.shape = [2N , (N_samp * N_syms_re)]
            log_p_vec = np.sum(log_p_mat, axis=0)
            # log_p_tsr.shape = [N_samp , N_syms_re]
            log_p_tsr = log_p_vec.reshape( shape[1:] )

            return log_p_tsr

        def compute_log_ps(y, H, s_tsr):
            ''' sigmoid approximation '''
            shape = s_tsr.shape
            # s_tsr.shape = [2K , N_samp , N_syms_re]
            s_mat = s_tsr.reshape(shape[0], -1)
            # s_mat.shape = [2K , (N_samp * N_syms_re)]

            c = 1.702
            sG = sqrt_2rho * np.diag(y) @ H
            term = sG @ s_mat
            # sigmoid approximation
            # compute log p = log sigmoid
            log_p_mat = - np.log( 1 + np.exp(- c * term ) )
            # FIXME: consider this version
            #log_p_mat = - logsumexp((0, -c * term))
            # log_p_mat.shape = [2N , (N_samp * N_syms_re)]
            log_p_vec = np.sum(log_p_mat, axis=0)
            # log_p_tsr.shape = [N_samp , N_syms_re]
            log_p_tsr = log_p_vec.reshape( shape[1:] )

            return log_p_tsr


        def construct_samples(qi, xi_idx):
            ''' must align with the dimension of qi '''
            i_samp = np.zeros((N_tx_real, N_samp), dtype=int)
            for ii in np.arange(N_tx_real):
                samp = rng.multinomial(1,qi[ii,:],N_samp)
                indices = np.argmax(samp, axis=1)
                i_samp[ii,:] = indices
            s_samp = syms[i_samp]

            #s_map.shape = (N_tx_real, N_samp, N_syms_re)
            s_map = np.repeat(s_samp[...,None], N_syms_re, axis=-1)

            # insert into s_map
            s_map[xi_idx,:,:] = syms

            return s_map

        ''' start of mean field algorithm '''

        # define log q_i(x_i)
        log_qi = np.zeros((N_tx_real, sqrt_M))

        for mi in np.arange(L_max):
            #print(f'iter:{mi}')
            #qi = compute_qi(log_qi)
            #print(qi)

            for xi in np.arange(N_tx_real):

                ''' compute mean field estimate of log q_i '''

                # compute qi
                qi = compute_qi(log_qi)
                # sample from qi
                s_map = construct_samples(qi,xi)
                # compute log p using samples
                #log_p = compute_log_p(y, H, s_map)
                log_p = compute_log_ps(y, H, s_map)

                # approx with average
                ex_log_qi = np.mean(log_p, axis=0)

                # add damping
                log_qi[xi] = (1-alpha) * log_qi[xi] + alpha * ex_log_qi

                # fix for high SNR
                # shift all log_qi such that the maximum is at zero.
                max_log_qi = np.amax(log_qi, axis=-1, keepdims=True)
                log_qi = log_qi - max_log_qi

        # get candidate
        mf_idx = np.argmax(log_qi, axis=1)

        syms_mf = syms[mf_idx]
        bits_mf = bits[mf_idx,:]

        # reorder bits
        real_dim = bits_mf.shape[0]
        real_idx = real_dim // 2
        real_bits = bits_mf[:real_idx,:]
        imag_bits = bits_mf[real_idx:,:]
        bits_mf = np.concatenate((real_bits, imag_bits), axis=1)
        # flatten
        bits_mf = bits_mf.reshape(-1)

        return bits_mf, syms_mf



