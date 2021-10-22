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


class OneBitVarLowerBoundDetector:
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
        syms_est_list = [ self.var_lower_bound(y_vec, h_mat) 
                        for y_vec, h_mat in zip(y_tsr, h_tsr) ]
        syms_est = np.array(syms_est_list)
        syms_est = self.batch_normalize(syms_est, N_tx)

        return syms_est

    def batch_normalize(self,x,N):
        # x.shape = [N, N_tx]
        norms = la.norm(x,axis=1,keepdims=True)
        norms_x = np.sqrt(N) / norms * x
        return norms_x

    def var_lower_bound(self, y, H):
        ''' compute var_lower_bound '''

        p = self.p
        N_tx_real = p.N_tx * 2
        L_max = 10

        rho = 1/p.n_var
        sqrt_2rho = np.sqrt(2*rho)
        G = np.diag(y) @ H
        c = 1.702
        s_tol = 1e-5
        sG = c * sqrt_2rho * G

        def sigmoid(x):
            return np.exp(-np.logaddexp(0, -x))

        def compute_lambda(xi):
            scale = 1/(2*xi)
            term = sigmoid(xi) - 0.5
            return scale * term

        def variational_e_step(sG, xi):
            # compute mean and variance
            I = np.identity(N_tx_real)
            sG_T = np.transpose(sG)
            ld = compute_lambda(xi)

            P = I + 2 * sG_T @ np.diag(ld) @ sG
            S = la.inv(P)

            g_sum = np.sum(sG_T, axis=1)
            mu = 0.5 * S @ g_sum

            return(mu, S)

        def variational_m_step(mu, S, sG):
            sG_T = np.transpose(sG)

            # returns lambda vector
            mu = np.squeeze(mu)
            M = mu[:,None] @ mu[None,:]
            ex_term = S + M

            mat_prod = sG @ ex_term @ sG_T

            xi_sqr = np.diag(mat_prod)
            xi = np.sqrt(xi_sqr)

            return xi


        ''' start of em variational lower bound '''

        mu_hat = np.zeros((N_tx_real,))
        beta = 0.01
        S_hat = beta * np.identity(N_tx_real)

        for i in np.arange(L_max):
            xi = variational_m_step(mu_hat, S_hat, sG)
            mu_new, S_new = variational_e_step(sG, xi)

            #if la.norm(mu_new - mu_hat) < s_tol:
            #    break
            mu_hat = mu_new
            S_hat = S_new

        return mu_hat

