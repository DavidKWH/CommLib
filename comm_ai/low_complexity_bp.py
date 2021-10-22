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


# sigmoid
def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

class OneBitLowCompBeliefPropGAI:
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

        print(f'LBP_GAI: alpha = {alpha}')

        if modulator:
            self.mod = modulator
        else:
            # assume QAM used
            self.mod = QAMModulator(p.M)

        ################################
        # precompute symbol tables
        ################################
        # generate bv/symbol tables
        sym_mat, bit_mat = self.mod.get_const_real()
        # save sym and bit vector table
        self.sym_mat = sym_mat
        self.bit_mat = bit_mat

    def __call__(self, *args, **kwargs):
        return self.batch_detect(*args, **kwargs)

    def batch_detect(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        Interface function to the internal compute_llrs_real().
        convert to real equivalent system

        y_tsr.shape = [ N x N_rx x 1 ]
        h_tsr.shape = [ N x N_rx x N_tx ]
        '''
        n_var_real_tsr = None
        y_tsr = np.squeeze(y_tsr, axis=2)
        y_real_tsr = cplx2reals(y_tsr)
        h_real_tsr = preproc_channel(h_tsr)

        # call function
        bits_mf, syms_mf_real = self.batch_detect_real(y_real_tsr, h_real_tsr, n_var_real_tsr)
        syms_mf = reals2cplx(syms_mf_real)

        return bits_mf, syms_mf

    def batch_detect_real(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
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
        mf_res_list = [ self.lbp_gai(y_vec, h_mat) 
                        for y_vec, h_mat in zip(y_tsr, h_tsr) ]
        mf_res = np.array(mf_res_list)
        bits_mf, syms_mf = np.split(mf_res, 2, axis=1)
        bits_mf = bits_mf.squeeze()
        syms_mf = syms_mf.squeeze()

        return bits_mf, syms_mf

    def lbp_gai(self, y, H):
        ''' Low complexity BP vai GAI '''

        p = self.p
        syms = self.sym_mat
        syms = syms.reshape(-1)
        syms_re = syms
        bits = self.bit_mat

        # pre-processing
        H_sqr = H**2

        # compute scaling
        #rho = 1/p.n_var
        #sqrt_2rho = np.sqrt(2*rho)
        n_var = p.n_var

        # other parameters
        N_tx_real = p.N_tx * 2
        N_rx_real = p.N_rx * 2
        N_sym_real = syms.size
        sqrt_M = np.sqrt(p.M).astype(int)
        Nbps_re = np.log2(sqrt_M).astype(int)

        ''' support functions '''

        def compute_mean_var(q_xj):
            # compute mean and variance of q_xj
            syms_re_sqr = syms_re**2

            mean_vec = np.sum(q_xj * syms_re[None,:], axis=1)
            sec_mnt_vec = np.sum(q_xj * syms_re_sqr[None,:], axis=1)
            var_vec = sec_mnt_vec - mean_vec**2

            return mean_vec, var_vec


        def compute_lh_yi_xj(i, j, s_tsr, mean_intf, std_intf):
            # s_tsr.shape = [N, M]
            # lh_tsr.shape = [N, M]

            c = 1.702
            #shape = s_tsr.shape
            #s_mat = s_tsr.reshape(shape[0],-1)
            s_mat = s_tsr
            term = y[i] * ( H[i,j] * s_mat + mean_intf ) / std_intf

            lh_mat = sigmoid( c * term )
            # reshape
            #out_shape = np.concatenate(((-1,), shape[1:]), axis=0)
            #lh_tsr = np.reshape( lh_mat, out_shape )
            lh_tsr = lh_mat

            return lh_tsr


        def compute_a_ij(i,j, mean_vec, var_vec):

            #b_idx = np.arange(N_tx_real)
            #c_idx = b_idx[b_idx != j]

            mean_isubset = mean_vec
            mean_isubset[j] = 0
            var_isubset = var_vec
            var_isubset[j] = 0
            H_isubset = H[i,:]
            H_sqr_isubset = H_sqr[i,:]

            mean_intf = np.sum(H_isubset * mean_isubset, axis=0)
            var_intf = np.sum(H_sqr_isubset * var_isubset, axis=0)
            var_intf = var_intf + n_var / 2
            std_intf = np.sqrt(var_intf)

            #for bi = np.arange(Nbps_re):
            s_tsr = syms_re
            lh_y_xj = compute_lh_yi_xj(i, j, s_tsr, mean_intf, std_intf)
            numer = np.sum(lh_y_xj[1])
            denom = np.sum(lh_y_xj[0])

            a_ij_elem = np.log(numer) - np.log(denom)
            #print(f'i,j=({i},{j}), a_ij={a_ij_elem}')

            return a_ij_elem


        def lbp_update_a_ij(b_ij, a_ij):

            for i in np.arange(N_rx_real):
                # extact row b_i
                b_i = b_ij[i,:,:]
                # compute q_i
                q_i = np.zeros((N_tx_real,N_sym_real))
                q_i[:,0:1] = 1 / (1 + np.exp(b_i))
                q_i[:,1:2] = 1 - q_i[:,0:1]
                mean_vec, var_vec = compute_mean_var(q_i)

                for j in np.arange(N_tx_real):

                    a_ij[i,j] = compute_a_ij(i, j, mean_vec, var_vec)

            return a_ij


        ''' start of LBP_GAI '''

        # initialize variables
        a_ij = np.zeros((N_rx_real, N_tx_real, Nbps_re)) # llrs obs->sym
        b_ij = np.zeros((N_rx_real, N_tx_real, Nbps_re)) # llrs sym->obs
        gm_j = np.zeros((N_tx_real, Nbps_re)) # gamma
        q_xj = np.zeros((N_tx_real, N_sym_real)) # probabilities

        LBP_MAX_ITER = 3

        for l in np.arange(LBP_MAX_ITER):
            # update pi
            a_ij = lbp_update_a_ij(b_ij, a_ij)
            # update final LLRs
            gm_j = np.sum(a_ij, axis=0)
            # update b_ij
            b_ij = gm_j - a_ij
            # compute probabilities
            q_xj[:,0:1] = 1 / (1 + np.exp(gm_j))
            q_xj[:,1:2] = 1 - q_xj[:,0:1]

        ''' end of LBP_GAI '''

        # get candidate
        mf_idx = np.argmax(q_xj, axis=1)

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



