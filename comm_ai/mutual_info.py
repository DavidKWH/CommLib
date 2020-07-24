# mutual information estimation algorithms
import numpy as np
from scipy.special import psi
from warnings import warn
from functools import partial

def knn_mi_estimator(c_vec, y_mat, **kwargs):
    if np.issubsctype(c_vec, np.integer):
        return d2c_knn_mi_estimator(c_vec, y_mat, **kwargs)
    else:
        # [TODO] add KSG estimator
        raise RuntimeError('continuous source variable not implemented')

def d2c_knn_mi_estimator(c_vec, y_mat, nc=None, k=3, units=None):
    '''
    implements KSG type nonparametric MI estimator
    for discrete-continous variable pairs
    see (Ross, 2014). "Mutual information between discrete and continuous data sets."
    NOTE: uses l2-norm for distance comparison
    NOTE: employs adaptive search radius to reduce search space per sample
    ASSUMPIONS:
        samples are stacked along 1st dimension
    '''
    # check if y_mat is 1d (convert to 2d)
    if y_mat.ndim == 1:
        y_mat = y_mat[:,None]

    duplicates_detected = False
    Nt = c_vec.shape[0]
    d = y_mat.shape[1]

    N_max = 50000 # max num samples for kNN evaluation
    N_iter = min([Nt, N_max])

    # helper function (vector l2-norm along columns)
    vecnorm = partial(np.linalg.norm, ord=None, axis=1)

    # limit search radium for neighbor
    # NOTE: the real bottleneck is sorting
    # NOTE: assume samples are distributed uniformly over a sphere
    #       for radius approximation - highly heruistic...
    y_norm_max = np.max(vecnorm(y_mat))
    ce = 4./3.*np.pi # scale factor for volume of sphere
    rho_0 = y_norm_max / (Nt/ce)**(1./d) * k
    #print('rho_0 =', rho_0)

    # compute total in each class
    assert( np.max(c_vec) < nc )
    idx_vec = np.arange(nc)
    ind_mat = c_vec[:,None] == idx_vec[None,:]
    nc_vec = np.sum(ind_mat, axis=0)

    # build class table
    yc_mat_seq = []
    yc_idx_seq = []
    for ci in np.arange(nc):
        idxs = np.where(ind_mat[:,ci])[0]
        yc_idx_seq.append(idxs)
        yc_mat_seq.append(y_mat[idxs,:])

    n_vec = np.zeros(N_iter)
    m_vec = np.zeros(N_iter)

    # initial guess
    rho = rho_0

    # loop over data points
    for ii in np.arange(N_iter):
        y = y_mat[ii,:]
        c = c_vec[ii]
        y_cand = y_mat
        # select samples from class
        yc_cand = yc_mat_seq[c]
        yc_dst = vecnorm( yc_cand - y )

        ''' START: adjust search space '''
        rd_ind = yc_dst < rho
        if np.sum(rd_ind) > 256:
            # shrink search space
            rho /= 2.
            rd_ind = yc_dst < rho
            #print('rho {:.4f} => {:.4f}'.format(rho*2, rho))
        while np.sum(rd_ind) <= k:
            # enlarge search space
            rho *= 2.
            rd_ind = yc_dst < rho
            #assert(rho < 1e10), "rho => inf"
            assert(not np.isinf(rho)), "rho => inf"
            #print('rho {:.4f} => {:.4f}'.format(rho/2, rho))
        ycr_dst = yc_dst[rd_ind]
        # NOTE: use argsort() to get sorted indices
        sdst = np.sort(ycr_dst)
        ''' END: adjust search space '''

        rad = sdst[k]
        y_dst = vecnorm( y_cand - y )
        m = np.sum( y_dst < rad )
        if m == 0:
            duplicates_detected = True
            '''
            in theory two samples from a continuous distribution
            has zero probability of being the same (zero distance for "strong" norm)
            Thus, if a neighbor is zero distance apart, just treat them as different
            At this point, we assume the difference is so small no samples from
            other classes appear in the eps-ball B(x_i,eps).
            NOTE: severe saturation of the samples will violate this assumption
            '''
            m = k
        '''
        add the neighbor on the edge and subtract current point(i)
        they cancel each other
        '''
        '''
        print('[{:3d}] k = {:d}, c = {:d}, nc = {:d}, m = {:d}'.format(
                       ii, k, c, nc_vec[c], m) )
        '''
        n_vec[ii] = nc_vec[c]
        m_vec[ii] = m

    if duplicates_detected:
        warn('Duplicates detected.  Possible over estimate', RuntimeWarning)

    # compute MI estimate
    psi_vec = psi(n_vec) + psi(m_vec)
    mean_psi = np.mean(psi_vec)
    mi_est_nats = psi(Nt) + psi(k) - mean_psi
    mi_est_bits = np.log2(np.exp(1)) * mi_est_nats

    return mi_est_bits

