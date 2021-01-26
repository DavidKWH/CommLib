# general dependencies
import numpy as np
import numpy.linalg as la
# helper functions
from functools import partial

from .core import QAMModulator
#from .core import PerSymbolDetector
from .core import PerSymbolDetectorV2
from scipy.stats import norm

# define diag_matrix operation
#def diag_mat(A, exp=1.):
#    return np.diag( np.diag(A)**exp )

def diag_mat_v(A_tsr, exp=1.):
    ''' vectorized version of diag_mat() '''
    # use einstein's notation
    # extract diagonal of the last two axes
    diag_mat = np.einsum('...ii->...i', A_tsr)
    diag_mat = diag_mat**exp
    # the diagonal should be reals
    diag_mat = diag_mat.real
    diag_tsr = np.zeros(A_tsr.shape)
    # assign to the diagonal of submatrices a vector of values
    np.einsum('...ii->...i', diag_tsr)[:,:] = diag_mat
    return diag_tsr

class BussgangDetector:
    '''
    Bussgang symbol detector for 1-bit channels, with
    Q(r) a component-wise sign operator
        r = Hx + z, y = Q(r)

    Based on BussgangEstimator
    '''
    def __init__(self, p, modulator=None, mode='bmmse'):
        self.p = p
        self.mode = mode
        self.est = partial(self.estimators[mode], self)
        #self.det = PerSymbolDetector(p, modulator)
        self.det = PerSymbolDetectorV2(p, modulator)
        print(f'Bussgang detector mode = {mode}')

#    def covar(self, w_mat, S_r):
#        ''' noise variance S_r '''
#        A = np.matrix(w_mat)
#        Sigma = A @ S_r @ A.H
#        return Sigma

    def bzf_est(self, y_tsr, A_tsr, S_n_tsr):

        A_pinv_tsr = la.pinv(A_tsr)
        W_tsr = A_pinv_tsr
        x_hat_tsr = W_tsr @ y_tsr

        return x_hat_tsr, W_tsr

    def bmmse_est(self, y_tsr, A_tsr, S_n_tsr):

        #def bmmse_weight(A, S_n):
        #    ''' NOTE: A is a tall matrix '''
        #    A = np.matrix(A)
        #    return A.H @ la.inv(A @ A.H + S_n)

        #W_tsr = [ bmmse_weight(A, S_n) for (A, S_n) in zip(A_tsr, S_n_tsr) ]

        # vectorized version of
        # A.H @ la.inv(A @ A.H + S_n)
        A = A_tsr
        S_n = S_n_tsr
        A_herm = np.conj(A).swapaxes(1,2)
        W_tsr = A_herm @ la.inv(A @ A_herm + S_n)

        x_hat_tsr = W_tsr @ y_tsr

        return x_hat_tsr, W_tsr

    # estimators
    estimators = {
        'bmmse' : bmmse_est,
        'bzf'   : bzf_est,
    }

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)

    def detect(self, y_tsr, h_tsr, n_var_tsr):
        '''
        Dimensions:
            y_tsr.shape     = (N, N_rx, 1)
            h_tsr.shape     = (N, N_rx, N_tx)
            n_var_tsr.shape = (N, 1, 1)

        TODO: consider vectorizing computation for speed
        NOTE: np.vectorize() is syntaxtic sugar (essentially for loops) does not
              offer speed ups.
        '''

        # compute equivalent linear channel parameters (S_r, A, S_n)
#        def compute_S_r(H, n_var):
#            H = np.matrix(H)
#            N_rx = H.shape[0]
#            I = n_var * np.identity(N_rx)
#            S_r = H @ H.H + I
#            return S_r
#
#        def compute_A(S_r, H):
#            ''' A = FH '''
#            scale = np.sqrt(2./np.pi)
#            F = scale * diag_mat( S_r, exp=-0.5 )
#            A = F @ H
#            return A
#
#        def compute_S_n(S_r, n_var):
#            # compute S_n
#            diag_mat_S_h = diag_mat( S_r, exp=-0.5 )
#            diag_mat_S_w = diag_mat( S_r, exp=-1.0 )
#            T_2 = diag_mat_S_h @ S_r @ diag_mat_S_h
#            np.clip(T_2.real, -1., 1., out=T_2.real)
#            np.clip(T_2.imag, -1., 1., out=T_2.imag)
#            T_1 = np.arcsin(T_2.real) + 1j*np.arcsin(T_2.imag)
#            S_n = 2./np.pi * (T_1 - T_2 + n_var * diag_mat_S_w)
#            return S_n

        #S_r_tsr = [ compute_S_r(h_mat, n_var) for (h_mat, n_var) in zip(h_tsr, n_var_tsr) ]
        #A_tsr = [ compute_A(S_r, h_mat) for (S_r, h_mat) in zip(S_r_tsr, h_tsr) ]
        #S_n_tsr = [ compute_S_n(S_r, n_var) for (S_r, n_var) in zip(S_r_tsr, n_var_tsr) ]

        N_rx = h_tsr.shape[1]
        N_tx = h_tsr.shape[2]
        H = h_tsr
        I = np.identity(N_rx)
        I = I[None,...]
        n_var = n_var_tsr

        # vectorized version of
        # H @ H.H + n_var * I
        H_herm = np.conj(H).swapaxes(1,2)
        S_r_tsr = H @ H_herm + n_var * I

        # vectorized version of compute_A()
        scale = np.sqrt(2./np.pi)
        diag_tsr_S_h = diag_mat_v( S_r_tsr, exp=-0.5 )
        F = scale * diag_tsr_S_h
        A_tsr = F @ H

        # vectorized version of compute_S_n()
        diag_tsr_S_w = diag_mat_v( S_r_tsr, exp=-1.0 )
        T_2 = diag_tsr_S_h @ S_r_tsr @ diag_tsr_S_h
        np.clip(T_2.real, -1., 1., out=T_2.real)
        np.clip(T_2.imag, -1., 1., out=T_2.imag)
        T_1 = np.arcsin(T_2.real) + 1j*np.arcsin(T_2.imag)
        S_n_tsr = 2./np.pi * (T_1 - T_2 + n_var * diag_tsr_S_w)

        x_hat_tsr, w_tsr = self.est(y_tsr, A_tsr, S_n_tsr)

        # vectorized version of covar()
        W = w_tsr
        W_herm = np.conj(W).swapaxes(1,2)
        covar_tsr = W @ S_n_tsr @ W_herm

        # equalize per symbol x_hat
        #WH = w_tsr @ h_tsr
        WH = w_tsr @ A_tsr
        WH_diag = np.diagonal(WH, axis1=1, axis2=2)
        # TODO: WH_diag using explicit einsum (enable broadcasting)
        #WH_diag = np.einsum('...ij,...ji->...i', w_tsr, h_tsr)
        # implements element-wise x_hat_tsr / WH_diag
        x_hat_eq = np.reciprocal(WH_diag)[...,None] * x_hat_tsr
        # assign to x_hat_tsr
        x_hat_tsr = x_hat_eq

        # rescale
        K_sqrt = np.sqrt(N_tx)
        x_hat_norms = la.norm(x_hat_eq, axis=1) # l2-norm
        x_hat_rescaled = [ x_hat / norm for (x_hat, norm) in zip(x_hat_eq, x_hat_norms) ]
        x_hat_rescaled = K_sqrt * np.array(x_hat_rescaled)
        # assign to x_hat_tsr
        x_hat_tsr = x_hat_rescaled

        # per symbol detection
        bits_msd, syms_msd = self.det.compute_msd(x_hat_tsr)

        return bits_msd, syms_msd


class BussgangEstimator:
    '''
    Add Bussgang estimator for 1-bit channels, with
    Q(r) a component-wise sign operator
        r = Hx + z, y = Q(r)
    '''
    def __init__(self, p, mode='bmmse'):
        self.p = p
        self.mode = mode
        self.est = partial(self.estimators[mode], self)
        print(f'Bussgang Estimator mode = {mode}')

#    def covar(self, w_mat, S_r):
#        ''' noise variance S_r '''
#        A = np.matrix(w_mat)
#        Sigma = A @ S_r @ A.H
#        return Sigma

    def bzf_est(self, y_tsr, A_tsr, S_n_tsr):

        A_pinv_tsr = la.pinv(A_tsr)
        W_tsr = A_pinv_tsr
        x_hat_tsr = W_tsr @ y_tsr

        return x_hat_tsr, W_tsr

    def bmmse_est(self, y_tsr, A_tsr, S_n_tsr):

        #def bmmse_weight(A, S_n):
        #    ''' NOTE: A is a tall matrix '''
        #    A = np.matrix(A)
        #    return A.H @ la.inv(A @ A.H + S_n)

        #W_tsr = [ bmmse_weight(A, S_n) for (A, S_n) in zip(A_tsr, S_n_tsr) ]

        # vectorized version of
        # A.H @ la.inv(A @ A.H + S_n)
        A = A_tsr
        S_n = S_n_tsr
        A_herm = np.conj(A).swapaxes(1,2)
        W_tsr = A_herm @ la.inv(A @ A_herm + S_n)

        x_hat_tsr = W_tsr @ y_tsr

        return x_hat_tsr, W_tsr

    # estimators
    estimators = {
        'bmmse' : bmmse_est,
        'bzf'   : bzf_est,
    }

    def estimate(self, y_tsr, h_tsr, n_var_tsr):
        '''
        Dimensions:
            y_tsr.shape     = (N, N_rx, 1)
            h_tsr.shape     = (N, N_rx, N_tx)
            n_var_tsr.shape = (N, 1, 1)

        TODO: consider vectorizing computation for speed
        NOTE: np.vectorize() is syntaxtic sugar (essentially for loops) does not
              offer speed ups.
        '''

        # compute equivalent linear channel parameters (S_r, A, S_n)
#        def compute_S_r(H, n_var):
#            H = np.matrix(H)
#            N_rx = H.shape[0]
#            I = n_var * np.identity(N_rx)
#            S_r = H @ H.H + I
#            return S_r
#
#        def compute_A(S_r, H):
#            ''' A = FH '''
#            scale = np.sqrt(2./np.pi)
#            F = scale * diag_mat( S_r, exp=-0.5 )
#            A = F @ H
#            return A
#
#        def compute_S_n(S_r, n_var):
#            # compute S_n
#            diag_mat_S_h = diag_mat( S_r, exp=-0.5 )
#            diag_mat_S_w = diag_mat( S_r, exp=-1.0 )
#            T_2 = diag_mat_S_h @ S_r @ diag_mat_S_h
#            np.clip(T_2.real, -1., 1., out=T_2.real)
#            np.clip(T_2.imag, -1., 1., out=T_2.imag)
#            T_1 = np.arcsin(T_2.real) + 1j*np.arcsin(T_2.imag)
#            S_n = 2./np.pi * (T_1 - T_2 + n_var * diag_mat_S_w)
#            return S_n

        #S_r_tsr = [ compute_S_r(h_mat, n_var) for (h_mat, n_var) in zip(h_tsr, n_var_tsr) ]
        #A_tsr = [ compute_A(S_r, h_mat) for (S_r, h_mat) in zip(S_r_tsr, h_tsr) ]
        #S_n_tsr = [ compute_S_n(S_r, n_var) for (S_r, n_var) in zip(S_r_tsr, n_var_tsr) ]

        N_rx = h_tsr.shape[1]
        N_tx = h_tsr.shape[2]
        H = h_tsr
        I = np.identity(N_rx)
        I = I[None,...]
        n_var = n_var_tsr

        # vectorized version of
        # H @ H.H + n_var * I
        H_herm = np.conj(H).swapaxes(1,2)
        S_r_tsr = H @ H_herm + n_var * I

        # vectorized version of compute_A()
        scale = np.sqrt(2./np.pi)
        diag_tsr_S_h = diag_mat_v( S_r_tsr, exp=-0.5 )
        F = scale * diag_tsr_S_h
        A_tsr = F @ H

        # vectorized version of compute_S_n()
        diag_tsr_S_w = diag_mat_v( S_r_tsr, exp=-1.0 )
        T_2 = diag_tsr_S_h @ S_r_tsr @ diag_tsr_S_h
        np.clip(T_2.real, -1., 1., out=T_2.real)
        np.clip(T_2.imag, -1., 1., out=T_2.imag)
        T_1 = np.arcsin(T_2.real) + 1j*np.arcsin(T_2.imag)
        S_n_tsr = 2./np.pi * (T_1 - T_2 + n_var * diag_tsr_S_w)

        x_hat_tsr, w_tsr = self.est(y_tsr, A_tsr, S_n_tsr)
        #covar_tsr = [ self.covar(w_mat, S_n) for (w_mat, S_n) in zip(w_tsr, S_n_tsr) ]
        # vectorized version of covar()
        W = w_tsr
        W_herm = np.conj(W).swapaxes(1,2)
        covar_tsr = W @ S_n_tsr @ W_herm

        # equalize per symbol x_hat
        #WH = w_tsr @ h_tsr
        WH = w_tsr @ A_tsr
        WH_diag = np.diagonal(WH, axis1=1, axis2=2)
        # TODO: WH_diag using explicit einsum (enable broadcasting)
        #WH_diag = np.einsum('...ij,...ji->...i', w_tsr, h_tsr)
        # implements element-wise x_hat_tsr / WH_diag
        x_hat_eq = np.reciprocal(WH_diag)[...,None] * x_hat_tsr
        # assign to x_hat_tsr
        x_hat_tsr = x_hat_eq

        # rescale
        K_sqrt = np.sqrt(N_tx)
        x_hat_norms = la.norm(x_hat_eq, axis=1) # l2-norm
        x_hat_rescaled = [ x_hat / norm for (x_hat, norm) in zip(x_hat_eq, x_hat_norms) ]
        x_hat_rescaled = K_sqrt * np.array(x_hat_rescaled)
        # assign to x_hat_tsr
        x_hat_tsr = x_hat_rescaled

        return x_hat_tsr, covar_tsr

class OneBitMLDetector:
    '''
    Find ML solution for 1-bit receiver given
    trasmit alphabets, receive signal and channel
    knowledge

    Method:
        The complex model will be converted to the real
        equivalent channel model.  ML is performed over
        the real model.

    Based on Demodulator class
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

        NOTE: symbol must first be converted to real vectors
        '''
        p = self.p
        mod = self.mod
        sym_vec, bit_mat = mod.get_const()

        def to_real(sym_vec):
            ''' concatenate along the 2nd dimension '''
            return np.concatenate([sym_vec.real, sym_vec.imag], axis=1)

        if p.N_sts == 1:
            return to_real(sym_vec), bit_mat
        else: # N_sts > 1
            sym_a, sym_b = sym_vec, sym_vec
            bit_a, bit_b = bit_mat, bit_mat
            for i in range(p.N_sts - 1):
                sym_a = self.vpermute(sym_a, sym_b)
                bit_a = self.vpermute(bit_a, bit_b)
            return to_real(sym_a), bit_a

    def __call__(self, *args, **kwargs):
        return self.compute_ml(*args, **kwargs)

    def compute_ml(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        Interface function to the internal compute_llrs_real().
        convert to real equivalent system
        '''
        def y_to_real(y_vec):
            return np.concatenate([y_vec.real, y_vec.imag], axis=0)

        def h_to_real(h_mat):
            h_real_t = np.hstack([h_mat.real, -h_mat.imag])
            h_real_b = np.hstack([h_mat.imag,  h_mat.real])
            h_real = np.vstack([h_real_t, h_real_b])
            return h_real

        #n_var_real_tsr = n_var_tsr / 2.
        n_var_real_tsr = None
        y_real_list = [ y_to_real(y_vec) for y_vec in y_tsr ]
        y_real_tsr = np.array(y_real_list)
        h_real_list = [ h_to_real(h_mat) for h_mat in h_tsr ]
        h_real_tsr = np.array(h_real_list)

        # call internal ml function
        return self.compute_ml_real(y_real_tsr, h_real_tsr, n_var_real_tsr)

    def compute_ml_real(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        Inputs from real system at this point...

        compute ML solution from compatible Y,H
        NOTE: assume noise variance given in params
        NOTE: noise variance is needed to compute exact LLRs.
              In the case of MI-estimation and min-sum decoding,
              the LLRs can be left unscaled without degrading
              performance.

        Dimensions:
            syms_x.shape = (N x M_tx)
            h_tsr.shape  = (N x M_rx x M_tx)

        '''
        p = self.p
        N = y_tsr.shape[0]
        syms = self.sym_mat
        bits = self.bit_mat

        # pre-compute scaling
        scale = 1./np.sqrt(p.n_var/2.)

        # implement using tensor broadcasting
        h_tsr_ex  = h_tsr[:,np.newaxis,:,:]
        syms_ex = syms[np.newaxis,:,:,np.newaxis]
        y  = y_tsr[:,np.newaxis,:,:]

        hs = h_tsr_ex @ syms_ex
        yhs = scale * y * hs
        phi = norm.cdf(yhs)
        phi = np.squeeze(phi, axis=3)
        lhood = np.prod(phi, axis=2)
        ml_idx = np.argmax(lhood, axis=1)

        # extract ML solutions
        syms_ml = syms[ml_idx,:]
        bits_ml = bits[ml_idx,:]

        # return ML solutions
        return bits_ml, syms_ml


class OneBitMLDemod:
    '''
    Compute exact LLRs for ML 1-bit receiver given
    trasmit alphabets, receive signal and channel
    knowledge

    Method:
        The complex model will be converted to the real
        equivalent channel model.  ML is performed over
        the real model.

    Based on Demodulator class
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

        NOTE: symbol must first be converted to real vectors
        '''
        p = self.p
        mod = self.mod
        sym_vec, bit_mat = mod.get_const()

        def to_real(sym_vec):
            ''' concatenate along the 2nd dimension '''
            return np.concatenate([sym_vec.real, sym_vec.imag], axis=1)

        if p.N_sts == 1:
            return to_real(sym_vec), bit_mat
        else: # N_sts > 1
            sym_a, sym_b = sym_vec, sym_vec
            bit_a, bit_b = bit_mat, bit_mat
            for i in range(p.N_sts - 1):
                sym_a = self.vpermute(sym_a, sym_b)
                bit_a = self.vpermute(bit_a, bit_b)
            return to_real(sym_a), bit_a

    def __call__(self, *args, **kwargs):
        return self.compute_llrs(*args, **kwargs)

    def compute_llrs(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        Interface function to the internal compute_llrs_real().
        convert to real equivalent system
        '''
        def y_to_real(y_vec):
            return np.concatenate([y_vec.real, y_vec.imag], axis=0)

        def h_to_real(h_mat):
            h_real_t = np.hstack([h_mat.real, -h_mat.imag])
            h_real_b = np.hstack([h_mat.imag,  h_mat.real])
            h_real = np.vstack([h_real_t, h_real_b])
            return h_real

        #n_var_real_tsr = n_var_tsr / 2.
        n_var_real_tsr = None
        y_real_list = [ y_to_real(y_vec) for y_vec in y_tsr ]
        y_real_tsr = np.array(y_real_list)
        h_real_list = [ h_to_real(h_mat) for h_mat in h_tsr ]
        h_real_tsr = np.array(h_real_list)

        # call internal llr compute function
        return self.compute_llrs_real(y_real_tsr, h_real_tsr, n_var_real_tsr)

    def compute_llrs_real(self, y_tsr, h_tsr, n_var_tsr, scaling=True):
        '''
        Inputs from real system at this point...

        compute exact LLRs from compatible Y,H
        NOTE: assume noise variance given in params
        NOTE: noise variance is needed to compute exact LLRs.
              In the case of MI-estimation and min-sum decoding,
              the LLRs can be left unscaled without degrading
              performance.

        Dimensions:
            syms_x.shape = (N x M_tx)
            h_tsr.shape  = (N x M_rx x M_tx)

        '''
        p = self.p
        exact_llr = not self.maxlog_approx
        sym_sets_1 = self.sym_sets_1
        sym_sets_0 = self.sym_sets_0
        N = y_tsr.shape[0]

#        def likelihood(y, H, x, scale):
#            ''' compute 1-bit likelihood'''
#
#            # compute per observed quantity
#            y = np.squeeze(y)
#            hx_vec = [ h @ x for h in H ]
#            #hx_vec = np.array(hx_list)
#            yhx_vec = scale * y * hx_vec
#            phi_vec = norm.cdf(yhx_vec)
#            # product reduce
#            lhood = np.prod(phi_vec)
#
#            return lhood

        # exact llr computation
        ########################
        n_bits = p.nbps * p.N_sts
        lambda_mat = np.zeros((N, n_bits))
        for ni in range(n_bits):

            syms_1 = sym_sets_1[ni]
            syms_0 = sym_sets_0[ni]

            N_cand = syms_1.shape[0]
            # pre-compute scaling
            scale = 1./np.sqrt(p.n_var/2.)

            # implement using tensor broadcasting
            h_tsr_ex  = h_tsr[:,np.newaxis,:,:]
            syms_1 = sym_sets_1[ni]
            syms_1_ex = syms_1[np.newaxis,:,:,np.newaxis]
            syms_0 = sym_sets_0[ni]
            syms_0_ex = syms_0[np.newaxis,:,:,np.newaxis]
            y  = y_tsr[:,np.newaxis,:,:]

            hs_1 = h_tsr_ex @ syms_1_ex
            yhs_1 = scale * y * hs_1
            phi_1 = norm.cdf(yhs_1)
            phi_1 = np.squeeze(phi_1, axis=3)
            lhood_1 = np.prod(phi_1, axis=2)

            log_eps = 1e-320 # add eps to prevent log underflow
            log_sum_lhood_1 = np.log( np.sum(lhood_1, axis=1) + log_eps)

            hs_0 = h_tsr_ex @ syms_0_ex
            yhs_0 = scale * y * hs_0
            phi_0 = norm.cdf(yhs_0)
            phi_0 = np.squeeze(phi_0, axis=3)
            lhood_0 = np.prod(phi_0, axis=2)

            log_sum_lhood_0 = np.log( np.sum(lhood_0, axis=1) + log_eps)

            lambda_vec = log_sum_lhood_0 - log_sum_lhood_1

            lambda_mat[:,ni] = lambda_vec

        return lambda_mat

