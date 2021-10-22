import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
# creating model
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
# layer specific
from tensorflow.keras.layers import Layer
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras import activations
#from tensorflow.keras import initializers
# constraints, initializers, etc.
from tensorflow.keras.constraints import NonNeg
# misc.
from functools import partial
# debug
#from .util import DumpFullTensor



# normal distribution
#norm  = tfp.distributions.Normal(loc=0., scale=1.)
laplace = tfp.distributions.Laplace(loc=0., scale=1.)

################################################################################
# helper functions
################################################################################
def compute_qi(log_qi):
    qi_unnorm = tf.exp(log_qi)
    qi_sum = tf.reduce_sum(qi_unnorm, axis=-1, keepdims=True)
    qi_norm = qi_unnorm / qi_sum
    return qi_norm

def cartprod(*arrays, reshape=True):
    # NOTE: stack them such that the output shape is
    # out.shape = [N, M, M,...]
    cartprod = tf.stack(tf.meshgrid(*arrays, indexing='ij'))
    return cartprod

def construct_qi_tsr(log_qi):
    log_qi_list = tf.unstack(log_qi)
    log_qi_tsr = cartprod(*log_qi_list, reshape=False)
    return log_qi_tsr



class MFNetLayerL1(Layer):
    '''
    Implements MFNet layer using laplace (based on V3)
    '''
    # define shared weight as class attribute
    alpha = None

    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)
        self.syms_re = syms_re

        self.initializer = kernel_initializer

        # pre-compute syms cartesian map
        sym_list = [syms_re for _ in tf.range(n_tx_re)]
        self.s_map = cartprod(*sym_list)

    def build(self, input_shape):
        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(self.n_tx_re, self.n_sym_re),
        #                         initializer=self.initializer,
        #                         trainable=True)

        # damping factor
        #initializer = tf.initializers.Constant(0.01)
        initializer = tf.initializers.Constant(1.0)
        #initializer = None
        #regularizer = tf.keras.regularizers.L2(0.01)
        regularizer = None
        trainable = False

        # class attribute (shared)
        if MFNetLayerL1.alpha is None:
            MFNetLayerL1.alpha = self.add_weight(name='alpha', # needed for tf.saved_model.save()
                                 shape=(),
                                 initializer=initializer,
                                 constraint=NonNeg(),
                                 regularizer=regularizer,
                                 trainable=trainable)
                                 #initializer=self.initializer,
                                 #trainable=True)

        #self.alpha = self.add_weight(name='alpha', # needed for tf.saved_model.save()
        #                         shape=(),
        #                         initializer=initializer,
        #                         constraint=NonNeg(),
        #                         regularizer=regularizer,
        #                         trainable=trainable)
                                 #initializer=self.initializer,
                                 #trainable=True)


    @tf.function
    def call(self, inputs, training=False):
        # define layer computation
        # inputs = [log_qi, G, sqrt_2rho]
        # dimensions
        # log_qi.shape = [N, N_tx_re, N_sym_re]
        # G.shape = [N, N_rx_re, N_tx_re]

        N_sym_re = self.n_sym_re
        N_tx_re = self.n_tx_re
        N_samp = self.n_samp

        #log_qi_in, G, sqrt_2rho = inputs
        log_qi, G, sqrt_2rho, n_var = inputs

        #n_var = tf.ones(shape=tf.shape(n_var)) * 0.01
        #n_var = self.sigma_sqr
        #rho = 1./n_var
        #sqrt_2rho = tf.sqrt(2.*rho)

        def compute_log_pl_bruteforce(G, s_tsr, sqrt_2rho):
            # let M = N_sym_re, K = N_tx_re
            # s_tsr.shape = (N_tx_re, M, M^(K-1))
            shape = tf.shape(s_tsr)
            s_mat = tf.reshape(s_tsr, (shape[0],-1))
            # s_mat.shape = (N_tx_re, M^K)

            sqrt_2rho = sqrt_2rho[:, tf.newaxis, tf.newaxis]
            # sG.shape = (N, N_rx_re, N_tx_re)
            sG = sqrt_2rho * G
            term = sG @ s_mat
            # numerically stable version
            log_p_mat = laplace.log_cdf(term)
            # log_p_mat.shape = [N, N_rx_re, M^K]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, M, M^(K-1)]
            out_shape = tf.concat(((-1,), shape[1:]),0)
            log_p_tsr = tf.reshape(log_p_vec, out_shape)
            #log_p_tsr = tf.reshape(log_p_vec, (-1, shape[1], shape[2]))

            return log_p_tsr


        def compute_log_p_bruteforce(G, s_tsr, sqrt_2rho):
            # let M = N_sym_re, K = N_tx_re
            # s_tsr.shape = (N_tx_re, M, M^(K-1))
            shape = tf.shape(s_tsr)
            s_mat = tf.reshape(s_tsr, (shape[0],-1))
            # s_mat.shape = (N_tx_re, M^K)

            sqrt_2rho = sqrt_2rho[:, tf.newaxis, tf.newaxis]
            # sG.shape = (N, N_rx_re, N_tx_re)
            sG = sqrt_2rho * G
            term = sG @ s_mat
            # numerically stable version
            log_p_mat = norm.log_cdf(term)
            # log_p_mat.shape = [N, N_rx_re, M^K]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, M, M^(K-1)]
            out_shape = tf.concat(((-1,), shape[1:]),0)
            log_p_tsr = tf.reshape(log_p_vec, out_shape)
            #log_p_tsr = tf.reshape(log_p_vec, (-1, shape[1], shape[2]))

            return log_p_tsr

        def compute_log_ps_bruteforce(G, s_tsr, sqrt_2rho):
            ''' sigmoid approximation '''
            # let M = N_sym_re, K = N_tx_re
            # s_tsr.shape = (N_tx_re, M, M^(K-1))
            shape = tf.shape(s_tsr)
            s_mat = tf.reshape(s_tsr, (shape[0],-1))
            # s_mat.shape = (N_tx_re, M^K)

            sqrt_2rho = sqrt_2rho[:, tf.newaxis, tf.newaxis]
            # sG.shape = (N, N_rx_re, N_tx_re)
            sG = sqrt_2rho * G
            term = sG @ s_mat
            # sigmoid approximation
            c = 1.702
            # numerical stable version of log-sigmoid
            log_p_mat = tf.math.log_sigmoid(c * term)
            # log_p_mat.shape = [N, N_rx_re, M^K]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, M, M^(K-1)]
            out_shape = tf.concat(((-1,), shape[1:]),0)
            log_p_tsr = tf.reshape(log_p_vec, out_shape)
            #log_p_tsr = tf.reshape(log_p_vec, (-1, shape[1], shape[2]))

            return log_p_tsr

        def update_log_qi_bruteforce(xi, log_qi):
            #tf.print(f'update_log_qi: xi = ', xi)
            alpha = self.alpha
            #beta = self.beta

            #log_qi.shape = (N, N_tx_re, N_sym_re)
            qi = compute_qi(log_qi)
            #qi = log_qi

            #qi[xi,:] = 1
            x_range = tf.range(N_tx_re)
            x_range = x_range[:,tf.newaxis]
            cond_insert = (x_range == xi)
            qi_new = tf.where(cond_insert, 1.0, qi)
            #qi_new = qi
            # construct tensor
            #qi_tsr.shape = (N, N_tx_re, M, M, ...), M = N_sym_re
            #qi_tsr = tf.map_fn(construct_qi_tsr, qi_new)
            qi_tsr = tf.vectorized_map(construct_qi_tsr, qi_new)

            # define transpose permutation
            sub_perm = tf.roll(tf.range(N_tx_re), -xi, 0)
            perm = tf.concat(([0, 1], sub_perm + 2), 0)
            qi_tsr = tf.transpose(qi_tsr, perm=perm)
            N = tf.shape(log_qi)[0]
            qi_tsr = tf.reshape(qi_tsr, (N, N_tx_re, N_sym_re, -1))

            # compute log_p
            # s_map.shape = (N_tx_re, M, M, ...)
            s_map = self.s_map
            perm = tf.concat(([0], sub_perm + 1), 0)
            s_map = tf.transpose(s_map, perm=perm)
            s_map = tf.reshape(s_map, (N_tx_re, N_sym_re, -1))
            #log_p_tsr = compute_log_p_bruteforce(G, s_map, sqrt_2rho)
            #log_p_tsr = compute_log_ps_bruteforce(G, s_map, sqrt_2rho)
            log_p_tsr = compute_log_pl_bruteforce(G, s_map, sqrt_2rho)
            log_p_tsr = log_p_tsr[:, tf.newaxis, ...]
            #log_p_tsr.shape = (N, 1, M, M, ...), M = N_sym_re

            # collect all integrands
            log_qi_ints = tf.concat((qi_tsr, log_p_tsr), axis=1)
            log_qi_prod = tf.reduce_prod(log_qi_ints, axis=1)
            #log_qi_prod.shape = [N, M, M, ...], M = N_sym_re

            # sum along all except xi index
            # NOTE: the summands are in the last dimension
            ex_log_qi = tf.reduce_sum(log_qi_prod, axis=-1)
            #ex_log_qi.shape = [N, N_sym_re]

            # assign to log_qi[xi] in numpy
            # log_qi.shape = [N, N_tx_re, N_sym_re]
            #log_qi[xi] = ex_log_qi (numpy)
            # apply damping
            #log_qi_xi = (1.-alpha) * log_qi[:,xi,:] + alpha * ex_log_qi
            #log_qi_xi = (1.-alpha) * log_qi[:,xi,:] + alpha * tf.stop_gradient(ex_log_qi)
            log_qi_xi = (1.-alpha) * log_qi[:,xi,:] + alpha * tf.stop_gradient(ex_log_qi)

            # generate log_qi_new in tensorflow
            log_qi_xi = tf.expand_dims(log_qi_xi, axis=1)
            x_range = tf.range(N_tx_re)
            x_range = tf.expand_dims(x_range, axis=-1)
            cond_insert = (x_range == xi)
            log_qi_new = tf.where(cond_insert, log_qi_xi, log_qi)

            # fix for high SNR
            # shift all log_qi such that the maximum is at zero.
            max_log_qi = tf.reduce_max(log_qi_new, axis=-1, keepdims=True)
            #log_qi_shifted = log_qi_new - max_log_qi
            log_qi_shifted = log_qi_new - tf.stop_gradient(max_log_qi)

            return log_qi_shifted

        #log_qi = update_log_qi_bruteforce(0, log_qi)
        #for xi in tf.range(N_tx_re):
        for xi in range(N_tx_re): # unroll loop
            log_qi = update_log_qi_bruteforce(xi, log_qi)

        return log_qi



class MFNetL1():
    '''
    Implements mean field net
    '''
    def __init__(self, n_out, n_in, n_const, n_samp, syms_re,
                       n_layers,
                       kernel_initializer=None):
        '''
        NOTE: channel dimension is inferred
        '''
        super().__init__()
        self.n_const = n_const
        self.n_in = n_in
        self.n_out = n_out
        self.n_samp = n_samp

        mfnet_layer = partial(MFNetLayerL1, kernel_initializer=kernel_initializer)

        self.log_qi = Input(shape=(n_in, n_const), name='log_qi_input')
        self.y = Input(shape=(n_out,), name='y_input')
        self.H = Input(shape=(n_out, n_in), name='H_input')
        self.r2rho = Input(shape=(), name='r2rho_input')
        self.n_var = Input(shape=(), name='n_var_input')

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(mfnet_layer(n_in, n_const, n_samp, syms_re))

        self.layers = layers


    def build(self):
        '''
        build model using functional API
        '''
        # set initial x to zero
        #x_shape = (1, self.n_in)
        #x_in = tf.zeros(x_shape)
        log_qi_in = self.log_qi
        y = self.y
        H = self.H
        r2rho = self.r2rho
        n_var = self.n_var
        G = tf.linalg.diag(y) @ H

        l_in = [log_qi_in, G, r2rho, n_var]
        l_out = []
        for layer in self.layers:
            log_qi_out = layer(l_in)
            l_in = [log_qi_out, G, r2rho, n_var]
            l_out.append(log_qi_out)

        log_qi_all = tf.stack(l_out)

        return Model(inputs=[log_qi_in, y, H, r2rho, n_var],
                     outputs=[log_qi_out, log_qi_all] , name='MFNetL1')

