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
# misc.
from functools import partial
# debug
#from .util import DumpFullTensor

# helper functions
def compute_qi(log_qi):
    qi_unnorm = tf.exp(log_qi)
    qi_sum = tf.reduce_sum(qi_unnorm, axis=-1, keepdims=True)
    qi_norm = qi_unnorm / qi_sum
    return qi_norm

# normal distribution
norm  = tfp.distributions.Normal(loc=0., scale=1.)

class MFSNetLayer(Layer):
    '''
    Implements MFNet layer
    '''
    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re, sqrt_2rho,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        self.n_syms_re = syms_re.size
        self.syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)
        self.sqrt_2rho = sqrt_2rho

        self.initializer = kernel_initializer

    def build(self, input_shape):
        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(self.n_tx_re, self.n_sym_re),
        #                         initializer=self.initializer,
        #                         trainable=True)


    def call(self, inputs, training=False):
        # define layer computation
        # inputs = [log_qi, G]
        # dimensions
        # log_qi.shape = [N, N_tx_re, N_sym_re]
        # G.shape = [N, N_rx_re, N_tx_re]

        N_tx_re = self.n_tx_re
        sqrt_2rho = self.sqrt_2rho

        log_qi, G = inputs

        def compute_log_p(G, s_mat, shape):
            # we compute the log likelihood for all combinations
            # sG.shape = (N, N_rx_re, N_tx_re)
            # s_mat.shape = (N, N_tx_re, N_samp * N_sym_re)
            sG = sqrt_2rho * G
            term = sG @ s_mat
            #log_p_mat = np.log( norm.cdf(term) )
            # numerically stable version
            log_p_mat = norm.log_cdf(term)
            # log_p_mat.shape = [N * N_rx_re * N_samp * N_sym_re]
            log_p_vec = np.sum(log_p_mat, axis=-2)
            # reshape
            log_p_tsr = log_p_vec.reshape( shape )

            return log_p_tsr

        def compute_log_ps(y, H, s_mat, shape):
            ''' sigmoid approximation '''
            c = 1.702

            sG = sqrt_2rho * np.diag(y) @ H
            term = sG @ s_mat

            log_p_mat = - np.log( 1 + np.exp(- c * term ) )
            # log_p_mat.shape = [2N * (N_samp)]
            log_p_vec = np.sum(log_p_mat, axis=0)
            # reshape
            log_p_tsr = log_p_vec.reshape( shape )

            return log_p_tsr


        def construct_samples(log_qi, xi_idx):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_syms_re = self.n_syms_re
            syms_re = self.syms_re

            #multi = tfp.distributions.Multinomial(1, probs=p)
            #samp = rng.multinomial(1,qi[ii,:],N_samp)
            multi = tfp.distributions.Multinomial(1, logits=log_qi)
            #samp.shape = (N_samp, N, N_tx_re, N_syms_re)
            samp = multi.sample(N_samp)
            #indices.shape = (N_samp, N, N_tx_re)
            indices = tf.argmax(samp, axis=-1)
            i_samp = tf.transpose(indices, perm=[1,2,0])
            #i_samp.shape = (N, N_tx_re, N_samp)
            #s_samp = syms_re[i_samp] (numpy)
            i_samp = tf.expand_dims(i_samp, axis=-1)
            s_samp = tf.gather_nd(syms_re, i_samp)

            # s_map.shape = (N, N_tx_re, N_samp * N_sym_re)
            s_map = tf.tile(s_samp, (1,1,N_syms_re))

            # generate combination of constellation
            #syms_rep = np.kron( syms_re, np.ones((1,N_samp)) )
            syms_rep = tf.repeat( syms_re, N_samp )

            # insert into s_map
            #s_map[:,xi_idx,:] = syms_rep (numpy)
            x_range = tf.range(N_tx_re)
            x_range = tf.expand_dims(x_range, axis=-1)
            cond_insert = (x_range == xi_idx)
            s_map_new = tf.where(cond_insert, syms_rep, s_map)

            return s_map_new


        def compute_log_qi(xi):

            # compute qi
            # NOTE: pass log_qi directly to Multinormial
            #compute_qi(log_qi)
            # sample from qi
            s_map = construct_samples(log_qi,xi)
            # compute log p using samples
            shape = (-1, N_tx_re, N_samp)
            log_p = compute_log_p(G, s_map, shape)

            # approx with average
            ex_log_qi = np.mean(log_p, axis=-1)

            #log_qi[xi] = ex_log_qi

            # fix for high SNR
            # shift all log_qi such that the maximum is at zero.
            max_log_qi = np.amax(log_qi, axis=-1, keepdims=True)
            log_qi = log_qi - max_log_qi
            import pdb; pdb.set_trace()

        log_qi_list = [ compute_log_qi(xi) for xi in tf.range(N_tx_re) ]

        #log_qi_new = tf.concatenate()

        return log_qi_new

class MFSNet():
    '''
    Implements mean field sampling net
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

        mfs_net_layer = partial(MFSNetLayer, kernel_initializer=kernel_initializer)

        self.log_qi = Input(shape=(n_in, n_const), name='log_qi_input')
        self.y = Input(shape=(n_out,), name='y_input')
        self.H = Input(shape=(n_out, n_in), name='H_input')

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(mfs_net_layer(n_in, n_out, n_samp, syms_re))

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
        G = tf.linalg.diag(y) @ H

        l_in = [log_qi_in, G]
        for layer in self.layers:
            log_qi_out = layer(l_in)
            l_in = [log_qi_out, G]

        # return log_qi
        # loss function may require this instead of qi

        #return Model(inputs=[y, H],
        return Model(inputs=[log_qi_in, y, H],
                     outputs=log_qi_out , name='MFSNet')


