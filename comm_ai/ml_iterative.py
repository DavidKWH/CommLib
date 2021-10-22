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

class IterativeMLDetector:
    '''
    Compute the q_i mean field update via the sampling method
    the 1-bit receiver
    '''
    def __init__(self, p,
                 modulator = None,
                 mode = 'nml',
                 ):
        self.p = p

        # save detector selection
        self.mode = mode
        self.est = partial(self.estimators[mode], self)
        self.det = PerSymbolDetectorV2(p, modulator)

        print(f'mode = {mode}')

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

        syms_est_list = [ self.est(y_vec, h_mat) 
                        for y_vec, h_mat in zip(y_tsr, h_tsr) ]
        syms_est = np.array(syms_est_list)
        syms_est = self.batch_normalize(syms_est, N_tx)

        return syms_est

    def batch_normalize(self,x,N):
        # x.shape = [N, N_tx]
        norms = la.norm(x,axis=1,keepdims=True)
        norms_x = np.sqrt(N) / norms * x
        return norms_x

    # normalize
    def normalize(self, x,N):
        norm = la.norm(x)
        normed_x = np.sqrt(N) / norm * x
        return normed_x

    def ml_gradient(self, y, H, x):
        '''gradient for ML'''
        p = self.p
        sqrt_2rho = p.sqrt_2rho

        sG = sqrt_2rho * np.diag(y) @ H
        sG_T = np.transpose(sG)
        # compute phi/Phi
        term = sG @ x
        #scales1 = norm.pdf(term) / norm.cdf(term)
        # numerically stable version
        scales2 = np.exp( norm.logpdf(term) - norm.logcdf(term) )

        scales = scales2
        grad = sG_T @ scales

        return grad

    def nml_gradient(self, y, H, x):
        '''gradient for ML'''
        p = self.p
        sqrt_2rho = p.sqrt_2rho

        G = np.diag(y) @ H
        G_T = np.transpose(G)
        # compute phi/Phi
        term = sqrt_2rho * G @ x
        #scales1 = norm.pdf(term) / norm.cdf(term)
        # numerically stable version
        scales2 = np.exp( norm.logpdf(term) - norm.logcdf(term) )

        scales = scales2
        grad = G_T @ scales

        return grad


    def nml_est(self, y, H):
        ''' perform nML gradient descent '''
        p = self.p
        N_tx = p.N_tx
        L_max = 100
        alpha_nml = 0.01
        e_tol = 1e-3

        # set initial X
        G = np.diag(y) @ H
        G_T = np.transpose(G)
        GT1 = np.sum(G_T, axis=1)
        s_hat = self.normalize(GT1, N_tx)

        for i in np.arange(L_max):
            s_hat_new = s_hat + alpha_nml * self.nml_gradient(y,H,s_hat)
            if la.norm(s_hat_new)**2 > N_tx:
                s_hat_new = self.normalize(s_hat_new, N_tx)
            dst = la.norm(s_hat_new - s_hat)
            #print(f'{i}: dst={dst}')
            if la.norm(s_hat_new - s_hat) < e_tol * la.norm(s_hat):
                break
            s_hat = s_hat_new
        return s_hat_new

    def ml_est(self, y, H):
        ''' perform ML gradient descent '''
        p = self.p
        N_tx = p.N_tx
        L_max = 100
        alpha_ml = 0.0001
        e_tol = 1e-3

        # set initial X
        G = np.diag(y) @ H
        G_T = np.transpose(G)
        GT1 = np.sum(G_T, axis=1)
        s_hat = self.normalize(GT1, N_tx)

        for i in np.arange(L_max):
            s_hat_new = s_hat + alpha_ml * self.ml_gradient(y,H,s_hat)
            dst = la.norm(s_hat_new - s_hat)
            #print(f'{i}: dst={dst}')
            if la.norm(s_hat_new - s_hat) < e_tol * la.norm(s_hat):
                break
            s_hat = s_hat_new
        return s_hat_new



    # sigmoid
    def sigmoid(self, x):
        return np.exp(-np.logaddexp(0, -x))

    def rml_gradient(self,y,H,x):
        'gradient for reformulated ML'
        p = self.p
        sqrt_2rho = p.sqrt_2rho
        c = 1.702

        # compute the gradient
        sG = - c * sqrt_2rho * np.diag(y) @ H
        sG_T = np.transpose(sG)
        term = sG @ x
        scales = self.sigmoid(term)
        #print(f'0: {scales}')
        grad = sG_T @ scales

        return grad

    def rml_gradient_alt(self,y,H,x):
        'gradient for reformulated ML'
        p = self.p
        sqrt_2rho = p.sqrt_2rho
        c = 1.702

        # compute the gradient
        G = np.diag(y) @ H
        G_T = np.transpose(G)
        term = - c * sqrt_2rho * G @ x
        scales = self.sigmoid(term)
        #print(f'0: {scales}')
        grad = - G_T @ scales

        return grad

    def rml_est(self, y, H):
        # compute gradient ascent for RML
        p = self.p
        N_tx_real = p.N_tx * 2
        L_max = 30
        #alpha_rml = 0.001
        alpha_rml = 0.01
        #s_tol = 1e-5
        s_tol = 1e-3

        s_hat = np.zeros((N_tx_real,))
        for i in np.arange(L_max):
            #s_hat_new = s_hat - alpha_rml * self.rml_gradient(y,H,s_hat)
            s_hat_new = s_hat - alpha_rml * self.rml_gradient_alt(y,H,s_hat)
            dst = la.norm(s_hat_new - s_hat)
            #print(f'{i}: dst={dst}')
            if la.norm(s_hat_new - s_hat) < s_tol * la.norm(s_hat):
                break
            s_hat = s_hat_new
        return s_hat_new

    # estimators
    estimators = {
        'ml'  : ml_est,
        'nml' : nml_est,
        'rml' : rml_est,
    }




