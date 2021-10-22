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
from .core import PerSymbolDetectorV2

from scipy.stats import norm
from scipy.special import logsumexp


class OneBitVBFactoredQrDetector:
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

        #print(f'SMF: alpha = {alpha}')

        self.det = PerSymbolDetectorV2(p, modulator)

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

        # call estimate function
        syms_est_real = self.batch_estimate_real(y_real_tsr, h_real_tsr, n_var_real_tsr)

        syms_est = reals2cplx(syms_est_real)
        syms_est = syms_est[...,None]

        # per symbol detection
        bits_det, syms_det = self.det.compute_msd(syms_est)

        return bits_det, syms_det, syms_est


    def batch_estimate_real(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
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
        N_tx = self.p.N_tx

        #syms_est_list = [ self.est(y_vec, h_mat) 
        syms_est_list = [ self.vb_factored_q_r(y_vec, h_mat) 
                        for y_vec, h_mat in zip(y_tsr, h_tsr) ]
        syms_est = np.array(syms_est_list)
        syms_est = self.batch_normalize(syms_est, N_tx)

        return syms_est

    def batch_normalize(self,x,N):
        # x.shape = [N, N_tx]
        norms = la.norm(x,axis=1,keepdims=True)
        norms_x = np.sqrt(N) / norms * x
        return norms_x

    def vb_factored_q_r(self, y, H):
        ''' VB factored q(r) '''
        p = self.p
        n_var = p.n_var
        N_tx_real = p.N_tx * 2
        N_rx_real = p.N_rx * 2
        L_max = 10

        def compute_q_r( mu_x, S_x, H, S_n ):
            # factored q_r: update each r_i

            mu_rg  = H @ mu_x
            var_rg = np.diag(S_n)

            return mu_rg, var_rg


        def compute_q_x( mu_rg, var_rg, y, H, P_n ):

            # compute mean of r
            sigma = np.sqrt(n_var/2)
            term = mu_rg / np.sqrt(var_rg)
            #mu_r = mu_rg + y * norm.pdf( term ) / norm.cdf( y * term )
            mu_r = mu_rg +  y * np.exp( norm.logpdf(term) - norm.logcdf( y * term) ) * sigma

            # compute mu_x and S_x
            S_xp = 2 * np.identity(N_tx_real) # prior x
            H_T = np.transpose(H)

            P_x = S_xp + H_T @ P_n @ H 
            S_x = la.inv(P_x)
            mu_x = S_x @ H_T @ P_n @ mu_r

            return mu_x, S_x

        ''' start of VB factored q(r) '''

        # initializer variables
        mu_x = np.zeros((N_tx_real,))
        S_x = np.identity(N_tx_real)
        mu_rg = np.zeros((N_rx_real,)) # mean of underlying gaussians
        var_rg = np.ones((N_rx_real,)) # var of underlying gaussians

        # compute S_n, P_n
        S_n = n_var/2 * np.identity(N_rx_real)
        P_n = 2/n_var * np.identity(N_rx_real)

        for i in np.arange(L_max):
            mu_rg, var_rg = compute_q_r( mu_x, S_x, H, S_n )
            mu_xn, S_xn = compute_q_x( mu_rg, var_rg, y, H, P_n )

            #dst = la.norm(mu_xn - mu_x)
            #if la.norm(mu_xn - mu_x) < s_tol:
            #    break
            mu_x = mu_xn
            S_x = S_xn


        return mu_x



class OneBitVBTnSourceDetector:
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

        #print(f'SMF: alpha = {alpha}')

        self.det = PerSymbolDetectorV2(p, modulator)

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

        # call estimate function
        syms_est_real = self.batch_estimate_real(y_real_tsr, h_real_tsr, n_var_real_tsr)

        syms_est = reals2cplx(syms_est_real)
        syms_est = syms_est[...,None]

        # per symbol detection
        bits_det, syms_det = self.det.compute_msd(syms_est)

        return bits_det, syms_det, syms_est


    def batch_estimate_real(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
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
        N_tx = self.p.N_tx

        #syms_est_list = [ self.est(y_vec, h_mat) 
        syms_est_list = [ self.vb_tn_source(y_vec, h_mat) 
                        for y_vec, h_mat in zip(y_tsr, h_tsr) ]
        syms_est = np.array(syms_est_list)
        syms_est = self.batch_normalize(syms_est, N_tx)

        return syms_est

    def batch_normalize(self,x,N):
        # x.shape = [N, N_tx]
        norms = la.norm(x,axis=1,keepdims=True)
        norms_x = np.sqrt(N) / norms * x
        return norms_x

    def vb_tn_source(self, y, H):
        ''' VB tn source '''
        p = self.p
        n_var = p.n_var
        N_tx_real = p.N_tx * 2
        N_rx_real = p.N_rx * 2
        L_max = 10


        def compute_q_r( mu_x, y, H, S_n ):
            # factored q_r: update each r_i

            mu_rg  = H @ mu_x
            var_rg = np.diag(S_n)

            # compute mean of r (truncated gaussian)
            sigma = np.sqrt(n_var/2)
            term = mu_rg / np.sqrt(var_rg)
            #mu_r = mu_rg + y * norm.pdf( term ) / norm.cdf( y * term )
            pdf_over_cdf = np.exp( norm.logpdf(term) - norm.logcdf( y * term) )
            #print(f'pdf_over_cdf = {pdf_over_cdf}')
            #print(f'sigma {sigma}')
            mu_r = mu_rg +  y * np.exp( norm.logpdf(term) - norm.logcdf( y * term) ) * sigma

            return mu_r, mu_rg, var_rg


        def compute_q_x( mu_r, mu_x, y, H, P_n ):

            # compute mu_x and S_x of underlying gaussian
            S_xp = 2 * np.identity(N_tx_real) # prior x
            H_T = np.transpose(H)

            P_xg = S_xp + H_T @ P_n @ H     # precision matrix
            S_xg = la.inv(P_xg)             # covariance matrix
            mu_xg = S_xg @ H_T @ P_n @ mu_r # mean vector

            m_vec = np.zeros((N_tx_real,))
            var_vec = 1./np.diag(P_xg)
            std_vec = np.sqrt(var_vec)
            mu_x = np.copy(mu_x) # make a copy
            for i in np.arange(N_tx_real):
                idx_vec = np.arange(N_tx_real)
                ni_idx = idx_vec != i
                ld_ii = P_xg[i,i]       # lambda_{i,i}
                ld_ii_inv = 1./ld_ii    # lambda_{i,i}^\inv
                ld_ini = P_xg[i,ni_idx] # lambda_{i,not i}
                mu_x_ni = mu_x[ni_idx]
                mu_xg_ni = mu_xg[ni_idx]

                # compute m (truncated gaussian)
                m_vec[i] = mu_xg[i] - ld_ii_inv * ld_ini @ ( mu_x_ni - mu_xg_ni )

                # compute mean of x (truncated gaussian)
                a = - 1/np.sqrt(2) # assume QPSK
                b =   1/np.sqrt(2)
                alpha = (a - m_vec[i])/std_vec[i]
                beta  = (b - m_vec[i])/std_vec[i]
                denum = norm.cdf(beta) - norm.cdf(alpha)
                numer = norm.pdf(alpha) - norm.pdf(beta)

                mu_x[i] = m_vec[i] + numer / denum * std_vec[i]

            return mu_x, mu_xg, S_xg


        ''' start of VB TN source '''

        # initialize varibles
        mu_x = np.zeros((N_tx_real,))
        #S_x = np.identity(N_tx_real)
        mu_rg = np.zeros((N_rx_real,)) # mean of underlying gaussians
        var_rg = np.ones((N_rx_real,)) # var of underlying gaussians

        # compute S_n, P_n
        S_n = n_var/2 * np.identity(N_rx_real)
        P_n = 2/n_var * np.identity(N_rx_real)

        for i in np.arange(L_max):

            mu_r, mu_rg, var_rg = compute_q_r( mu_x, y, H, S_n )
            mu_xn, mu_xg, S_xg = compute_q_x( mu_r, mu_x, y, H, P_n )

            #dst = la.norm(mu_xn - mu_x)
            #if la.norm(mu_xn - mu_x) < s_tol:
            #    break
            mu_x = mu_xn
            #S_x = S_xn

        return mu_x

