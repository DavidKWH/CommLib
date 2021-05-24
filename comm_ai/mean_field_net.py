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
    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        self.n_syms_re = syms_re.size
        self.syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)

        self.initializer = kernel_initializer

    def build(self, input_shape):
        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(self.n_tx_re, self.n_sym_re),
        #                         initializer=self.initializer,
        #                         trainable=True)
        pass


    @tf.function
    def call(self, inputs, training=False):
        # define layer computation
        # inputs = [log_qi, G, sqrt_2rho]
        # dimensions
        # log_qi.shape = [N, N_tx_re, N_sym_re]
        # G.shape = [N, N_rx_re, N_tx_re]

        N_tx_re = self.n_tx_re
        N_samp = self.n_samp

        #log_qi_in, G, sqrt_2rho = inputs
        log_qi, G, sqrt_2rho = inputs

        def compute_log_p(G, s_tsr, sqrt_2rho):
            shape = s_tsr.shape
            # s_tsr.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_mat = tf.reshape(s_tsr, (shape[0],shape[1],-1))
            # s_mat.shape = (N, N_tx_re, N_samp * N_sym_re)

            sqrt_2rho = sqrt_2rho[..., tf.newaxis, tf.newaxis]
            # sG.shape = (N, N_rx_re, N_tx_re)
            sG = sqrt_2rho * G
            term = sG @ s_mat
            # numerically stable version
            log_p_mat = norm.log_cdf(term)
            # log_p_mat.shape = [N, N_rx_re, (N_samp * N_sym_re)]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, N_samp , N_syms_re]
            log_p_tsr = tf.reshape(log_p_vec, (-1,shape[2],shape[3]))

            return log_p_tsr

        def compute_log_ps(G, s_tsr, sqrt_2rho):
            ''' sigmoid approximation '''
            shape = tf.shape(s_tsr)
            # s_tsr.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_mat = tf.reshape(s_tsr, (shape[0],shape[1],-1))
            # s_mat.shape = (N, N_tx_re, N_samp * N_sym_re)

            sqrt_2rho = sqrt_2rho[..., tf.newaxis, tf.newaxis]
            # sG.shape = (N, N_rx_re, N_tx_re)
            sG = sqrt_2rho * G
            term = sG @ s_mat
            # sigmoid approximation
            c = 1.702
            # numerical stable version of log-sigmoid
            #log_p_mat = - tf.math.log( 1 + tf.exp(- c * term ) )
            log_p_mat = tf.math.log_sigmoid(c * term)
            # log_p_mat.shape = [N, N_rx_re, (N_samp * N_sym_re)]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, N_samp , N_syms_re]
            log_p_tsr = tf.reshape(log_p_vec, (-1,shape[2],shape[3]))

            return log_p_tsr


        def construct_samples(log_qi, xi_idx):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_syms_re = self.n_syms_re
            syms_re = self.syms_re

            multi = tfp.distributions.Multinomial(1, logits=log_qi)
            #samp.shape = (N_samp, N, N_tx_re, N_syms_re)
            samp = multi.sample(N_samp)
            #indices.shape = (N_samp, N, N_tx_re)
            indices = tf.argmax(samp, axis=-1)
            i_samp = tf.transpose(indices, perm=[1,2,0])
            #i_samp.shape = (N, N_tx_re, N_samp)
            #s_samp = syms_re[i_samp] # in numpy
            i_samp = tf.expand_dims(i_samp, axis=-1)
            #s_samp.shape = (N, N_tx_re, N_samp)
            s_samp = tf.gather_nd(syms_re, i_samp)

            s_samp = tf.expand_dims(s_samp, axis=-1)
            # s_map.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_map = tf.repeat(s_samp, N_syms_re, axis=-1)

            # insert into s_map in numpy
            #s_map[xi_idx,:,:] = syms (numpy)
            # generate new map in tensorflow
            x_range = tf.range(N_tx_re)
            x_range = x_range[:, tf.newaxis, tf.newaxis]
            cond_insert = (x_range == xi_idx)
            s_map_new = tf.where(cond_insert, syms_re, s_map)

            return s_map_new


        def update_log_qi(xi, log_qi):
            #tf.print(f'update_log_qi: xi = ', xi)

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_map = construct_samples(log_qi,xi)
            # compute log p using samples
            #log_p = compute_log_p(G, s_map, sqrt_2rho)
            log_p = compute_log_ps(G, s_map, sqrt_2rho)
            #log_p.shape (N, N_samp, N_tx_re)

            # approx with average
            ex_log_qi = tf.reduce_mean(log_p, axis=-2)

            # assign to log_qi[xi] in numpy
            # log_qi.shape = [N, N_tx_re, N_sym_re]
            #log_qi[xi] = ex_log_qi (numpy)
            # generate log_qi_new in tensorflow
            ex_log_qi = tf.expand_dims(ex_log_qi, axis=1)
            x_range = tf.range(N_tx_re)
            x_range = tf.expand_dims(x_range, axis=-1)
            cond_insert = (x_range == xi)
            log_qi_new = tf.where(cond_insert, ex_log_qi, log_qi)

            # fix for high SNR
            # shift all log_qi such that the maximum is at zero.
            max_log_qi = tf.reduce_max(log_qi_new, axis=-1, keepdims=True)
            log_qi_shifted = log_qi_new - max_log_qi

            return log_qi_shifted

        for xi in tf.range(N_tx_re):
            log_qi = update_log_qi(xi, log_qi)

        return log_qi

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
        self.r2rho = Input(shape=(), name='r2rho_input')

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
        r2rho = self.r2rho
        G = tf.linalg.diag(y) @ H

        l_in = [log_qi_in, G, r2rho]
        for layer in self.layers:
            log_qi_out = layer(l_in)
            l_in = [log_qi_out, G, r2rho]

        # return log_qi
        # loss function may require this instead of qi

        #return Model(inputs=[y, H],
        return Model(inputs=[log_qi_in, y, H, r2rho],
                     outputs=log_qi_out , name='MFSNet')


