import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
# creating model
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
# layer specific
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras import activations
#from tensorflow.keras import initializers
# constraints, initializers, etc.
from tensorflow.keras.constraints import NonNeg
# custom layer
from .keras import ResidualV2
from .keras import HyperDenseV2
# misc.
from functools import partial
# debug
#from .util import DumpFullTensor



# normal distribution
norm  = tfp.distributions.Normal(loc=0., scale=1.)

class MFSNetLayer(Layer):
    '''
    Implements MFSNet layer
    '''
    # define shared weight as class attribute
    alpha = None

    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        self.syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)

        self.initializer = kernel_initializer

    def build(self, input_shape):
        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(self.n_tx_re, self.n_sym_re),
        #                         initializer=self.initializer,
        #                         trainable=True)

        # damping factor
        initializer = tf.ones_initializer()
        #initializer = None
        trainable = True

        # class attribute (shared)
        if MFSNetLayer.alpha is None:
            MFSNetLayer.alpha = self.add_weight(name='alpha', # needed for tf.saved_model.save()
                                 shape=(),
                                 initializer=initializer,
                                 constraint=NonNeg(),
                                 trainable=trainable)
                                 #initializer=self.initializer,
                                 #trainable=True)

        #self.alpha = self.add_weight(name='alpha', # needed for tf.saved_model.save()
        #                         shape=(),
        #                         initializer=initializer,
        #                         constraint=NonNeg(),
        #                         trainable=trainable)
        #                         #initializer=self.initializer,
        #                         #trainable=True)


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
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
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
            log_p_mat = tf.math.log_sigmoid(c * term)
            # log_p_mat.shape = [N, N_rx_re, (N_samp * N_sym_re)]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
            log_p_tsr = tf.reshape(log_p_vec, (-1,shape[2],shape[3]))

            return log_p_tsr

        def construct_samples_relaxed(log_qi, xi_idx):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            #multi = tfp.distributions.Multinomial(1, logits=log_qi)
            temp = 0.1
            relaxed = tfp.distributions.RelaxedOneHotCategorical(temp, logits=log_qi)

            #samp.shape = (N_samp, N, N_tx_re, N_sym_re)
            samp = relaxed.sample(N_samp)

            #if training == False:
            #    # discretize softmax during inference
            #    indices = tf.argmax(samp, axis=-1)
            #    samp = tf.one_hot(indices, N_sym_re, dtype=samp.dtype)

            # soft combine
            #c_samp.shape = (N_samp, N, N_tx_re)
            c_samp = samp @ syms_re[:,tf.newaxis]
            c_samp = tf.squeeze(c_samp, axis=-1)
            #s_samp.shape = (N, N_tx_re, N_samp)
            s_samp = tf.transpose(c_samp, perm=[1,2,0])

            s_samp = tf.expand_dims(s_samp, axis=-1)
            # s_map.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_map = tf.repeat(s_samp, N_sym_re, axis=-1)

            # insert into s_map in numpy
            #s_map[xi_idx,:,:] = syms (numpy)
            # generate new map in tensorflow
            x_range = tf.range(N_tx_re)
            x_range = x_range[:, tf.newaxis, tf.newaxis]
            cond_insert = (x_range == xi_idx)
            s_map_new = tf.where(cond_insert, syms_re, s_map)

            return s_map_new


        def construct_samples(log_qi, xi_idx):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            multi = tfp.distributions.Multinomial(1, logits=log_qi)
            #samp.shape = (N_samp, N, N_tx_re, N_sym_re)
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
            s_map = tf.repeat(s_samp, N_sym_re, axis=-1)

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
            alpha = self.alpha

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_map = construct_samples(log_qi,xi)
            #s_map = construct_samples_relaxed(log_qi,xi)
            # compute log p using samples
            #log_p = compute_log_p(G, s_map, sqrt_2rho)
            # use sigmoid approx
            log_p = compute_log_ps(G, s_map, sqrt_2rho)
            #log_p.shape (N, N_samp, N_tx_re)

            # approx with average
            ex_log_qi = tf.reduce_mean(log_p, axis=-2)
            # apply damping
            log_qi_xi = (1-alpha) * log_qi[:,xi,:] + alpha * tf.stop_gradient(ex_log_qi)

            # assign to log_qi[xi] in numpy
            # log_qi.shape = [N, N_tx_re, N_sym_re]
            #log_qi[xi] = ex_log_qi (numpy)
            # generate log_qi_new in tensorflow
            log_qi_xi = tf.expand_dims(log_qi_xi, axis=1)

            x_range = tf.range(N_tx_re)
            x_range = tf.expand_dims(x_range, axis=-1)
            cond_insert = (x_range == xi)
            log_qi_new = tf.where(cond_insert, log_qi_xi, log_qi)

            # fix for high SNR
            # shift all log_qi such that the maximum is at zero.
            max_log_qi = tf.reduce_max(log_qi_new, axis=-1, keepdims=True)
            log_qi_shifted = log_qi_new - tf.stop_gradient(max_log_qi)

            return log_qi_shifted


        #for xi in tf.range(N_tx_re):
        for xi in range(N_tx_re): # unroll loop
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
        self.n_layers = n_layers

        mfs_net_layer = partial(MFSNetLayer, kernel_initializer=kernel_initializer)

        self.log_qi = Input(shape=(n_in, n_const), name='log_qi_input')
        self.y = Input(shape=(n_out,), name='y_input')
        self.H = Input(shape=(n_out, n_in), name='H_input')
        self.r2rho = Input(shape=(), name='r2rho_input')

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(mfs_net_layer(n_in, n_const, n_samp, syms_re))

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



class MFSNetLayerS1(Layer):
    '''
    Implements MFSNet layer
    alpha(sigma) = w * sigma^2 (w is learned)
    '''
    # define shared weight as class attribute
    w = None

    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        self.syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)

        self.initializer = kernel_initializer

    def build(self, input_shape):
        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(self.n_tx_re, self.n_sym_re),
        #                         initializer=self.initializer,
        #                         trainable=True)

        # damping factor
        initializer = tf.initializers.Constant(1.0)
        #initializer = None
        regularizer = tf.keras.regularizers.L2(0.01)
        #regularizer = None
        trainable = False

        # class attribute (shared)
        if MFSNetLayerS1.w is None:
            MFSNetLayerS1.w = self.add_weight(name='w', # needed for tf.saved_model.save()
                                 shape=(),
                                 initializer=initializer,
                                 constraint=NonNeg(),
                                 trainable=trainable)
                                 #initializer=self.initializer,
                                 #trainable=True)

        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(),
        #                         initializer=initializer,
        #                         constraint=NonNeg(),
        #                         trainable=trainable)
        #                         #initializer=self.initializer,
        #                         #trainable=True)


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
        log_qi, G, sqrt_2rho, n_var = inputs

        # compute sqrt_2rho
        n_var_clipped = tf.maximum(0.01, n_var)
        rho = 1./n_var_clipped
        sqrt_2rho = tf.sqrt(2.*rho)

        # compute alpha
        w = self.w
        alpha = tf.minimum(1., w * n_var_clipped)
        alpha = alpha[:,tf.newaxis]

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
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
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
            log_p_mat = tf.math.log_sigmoid(c * term)
            # log_p_mat.shape = [N, N_rx_re, (N_samp * N_sym_re)]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
            log_p_tsr = tf.reshape(log_p_vec, (-1,shape[2],shape[3]))

            return log_p_tsr

        def construct_samples_relaxed(log_qi, xi_idx):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            #multi = tfp.distributions.Multinomial(1, logits=log_qi)
            temp = 0.1
            relaxed = tfp.distributions.RelaxedOneHotCategorical(temp, logits=log_qi)

            #samp.shape = (N_samp, N, N_tx_re, N_sym_re)
            samp = relaxed.sample(N_samp)

            #if training == False:
            #    # discretize softmax during inference
            #    indices = tf.argmax(samp, axis=-1)
            #    samp = tf.one_hot(indices, N_sym_re, dtype=samp.dtype)

            # soft combine
            #c_samp.shape = (N_samp, N, N_tx_re)
            c_samp = samp @ syms_re[:,tf.newaxis]
            c_samp = tf.squeeze(c_samp, axis=-1)
            #s_samp.shape = (N, N_tx_re, N_samp)
            s_samp = tf.transpose(c_samp, perm=[1,2,0])

            s_samp = tf.expand_dims(s_samp, axis=-1)
            # s_map.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_map = tf.repeat(s_samp, N_sym_re, axis=-1)

            # insert into s_map in numpy
            #s_map[xi_idx,:,:] = syms (numpy)
            # generate new map in tensorflow
            x_range = tf.range(N_tx_re)
            x_range = x_range[:, tf.newaxis, tf.newaxis]
            cond_insert = (x_range == xi_idx)
            s_map_new = tf.where(cond_insert, syms_re, s_map)

            return s_map_new


        def construct_samples(log_qi, xi_idx):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            multi = tfp.distributions.Multinomial(1, logits=log_qi)
            #samp.shape = (N_samp, N, N_tx_re, N_sym_re)
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
            s_map = tf.repeat(s_samp, N_sym_re, axis=-1)

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
            #alpha = self.alpha

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_map = construct_samples(log_qi,xi)
            #s_map = construct_samples_relaxed(log_qi,xi)
            # compute log p using samples
            #log_p = compute_log_p(G, s_map, sqrt_2rho)
            # use sigmoid approx
            log_p = compute_log_ps(G, s_map, sqrt_2rho)
            #log_p.shape (N, N_samp, N_tx_re)

            # approx with average
            ex_log_qi = tf.reduce_mean(log_p, axis=-2)
            # apply damping
            log_qi_xi = (1-alpha) * log_qi[:,xi,:] + alpha * tf.stop_gradient(ex_log_qi)

            # assign to log_qi[xi] in numpy
            # log_qi.shape = [N, N_tx_re, N_sym_re]
            #log_qi[xi] = ex_log_qi (numpy)
            # generate log_qi_new in tensorflow
            log_qi_xi = tf.expand_dims(log_qi_xi, axis=1)

            x_range = tf.range(N_tx_re)
            x_range = tf.expand_dims(x_range, axis=-1)
            cond_insert = (x_range == xi)
            log_qi_new = tf.where(cond_insert, log_qi_xi, log_qi)

            # fix for high SNR
            # shift all log_qi such that the maximum is at zero.
            max_log_qi = tf.reduce_max(log_qi_new, axis=-1, keepdims=True)
            log_qi_shifted = log_qi_new - tf.stop_gradient(max_log_qi)

            return log_qi_shifted


        #for xi in tf.range(N_tx_re):
        for xi in range(N_tx_re): # unroll loop
            log_qi = update_log_qi(xi, log_qi)

        return log_qi


class MFSNetS1():
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
        self.n_layers = n_layers

        mfs_net_layer = partial(MFSNetLayerS1, kernel_initializer=kernel_initializer)

        self.log_qi = Input(shape=(n_in, n_const), name='log_qi_input')
        self.y = Input(shape=(n_out,), name='y_input')
        self.H = Input(shape=(n_out, n_in), name='H_input')
        self.r2rho = Input(shape=(), name='r2rho_input')
        self.n_var = Input(shape=(), name='n_var_input')

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(mfs_net_layer(n_in, n_const, n_samp, syms_re))

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
        for layer in self.layers:
            log_qi_out = layer(l_in)
            l_in = [log_qi_out, G, r2rho, n_var]

        # return log_qi
        # loss function may require this instead of qi

        #return Model(inputs=[y, H],
        return Model(inputs=[log_qi_in, y, H, r2rho, n_var],
                     outputs=log_qi_out , name='MFSNetS1')


################################################################################
# Parallel updates
################################################################################

class ParallelMFSNetLayerS1(Layer):
    '''
    Implements MFSNet layer
    alpha(sigma) = w * sigma^2 (w is learned)
    '''
    # define shared weight as class attribute
    w = None

    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        self.syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)

        self.initializer = kernel_initializer

    def build(self, input_shape):
        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(self.n_tx_re, self.n_sym_re),
        #                         initializer=self.initializer,
        #                         trainable=True)

        # damping factor
        initializer = tf.initializers.Constant(1.0)
        #initializer = None
        regularizer = tf.keras.regularizers.L2(0.01)
        #regularizer = None
        trainable = False

        # class attribute (shared)
        if ParallelMFSNetLayerS1.w is None:
            ParallelMFSNetLayerS1.w = self.add_weight(name='w', # needed for tf.saved_model.save()
                                 shape=(),
                                 initializer=initializer,
                                 constraint=NonNeg(),
                                 trainable=trainable)
                                 #initializer=self.initializer,
                                 #trainable=True)

        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                         shape=(),
        #                         initializer=initializer,
        #                         constraint=NonNeg(),
        #                         trainable=trainable)
        #                         #initializer=self.initializer,
        #                         #trainable=True)


    #@tf.function
    def call(self, inputs, training=False):
        # define layer computation
        # inputs = [log_qi, G, sqrt_2rho]
        # dimensions
        # log_qi.shape = [N, N_tx_re, N_sym_re]
        # G.shape = [N, N_rx_re, N_tx_re]

        N_tx_re = self.n_tx_re
        N_samp = self.n_samp

        #log_qi_in, G, sqrt_2rho = inputs
        log_qi, G, sqrt_2rho, n_var = inputs

        # compute sqrt_2rho
        n_var_clipped = tf.maximum(0.01, n_var)
        #n_var_clipped = tf.maximum(0.03, n_var)
        rho = 1./n_var_clipped
        sqrt_2rho = tf.sqrt(2.*rho)

        # compute alpha
        w = self.w
        alpha = tf.minimum(1., w * n_var_clipped)
        alpha = alpha[:,tf.newaxis]

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
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
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
            log_p_mat = tf.math.log_sigmoid(c * term)
            # log_p_mat.shape = [N, N_rx_re, (N_samp * N_sym_re)]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
            log_p_tsr = tf.reshape(log_p_vec, (-1,shape[2],shape[3]))

            return log_p_tsr

        def construct_samples(log_qi):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            multi = tfp.distributions.Multinomial(1, logits=log_qi)
            #samp.shape = (N_samp, N, N_tx_re, N_sym_re)
            samp = multi.sample(N_samp)
            #indices.shape = (N_samp, N, N_tx_re)
            indices = tf.argmax(samp, axis=-1)
            #indices = tf.squeeze(indices, axis=3)
            i_samp = tf.transpose(indices, perm=[1,2,0])
            #i_samp.shape = (N, N_tx_re, N_samp)
            #s_samp = syms_re[i_samp] # in numpy
            i_samp = tf.expand_dims(i_samp, axis=-1)
            #s_samp.shape = (N, N_tx_re, N_samp)
            s_samp = tf.gather_nd(syms_re, i_samp)

            return s_samp

        def filter_samples(s_samp, xi_idx):
            '''
            s_samp.shape = (N, N_tx_re, N_samp)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            s_samp = tf.expand_dims(s_samp, axis=-1)
            # s_map.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_map = tf.repeat(s_samp, N_sym_re, axis=-1)

            # insert into s_map in numpy
            #s_map[xi_idx,:,:] = syms (numpy)
            # generate new map in tensorflow
            x_range = tf.range(N_tx_re)
            x_range = x_range[:, tf.newaxis, tf.newaxis]
            cond_insert = (x_range == xi_idx)
            s_map_new = tf.where(cond_insert, syms_re, s_map)

            return s_map_new


        def update_log_qi_xi(xi, log_qi, s_samp):
            #tf.print(f'update_log_qi: xi = ', xi)
            #alpha = self.alpha

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_map = filter_samples(s_samp, xi)
            # compute log p using samples
            #log_p = compute_log_p(G, s_map, sqrt_2rho)
            # use sigmoid approx
            log_p = compute_log_ps(G, s_map, sqrt_2rho)
            #log_p.shape (N, N_samp, N_tx_re)

            # approx with average
            ex_log_qi = tf.reduce_mean(log_p, axis=-2)
            # apply damping
            log_qi_xi = (1-alpha) * log_qi[:,xi,:] + alpha * tf.stop_gradient(ex_log_qi)

            return log_qi_xi

        def parallel_update_log_qi(log_qi):

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_samp = construct_samples(log_qi)

            log_qi_seq = []
            for xi in range(N_tx_re): # unroll loop
                log_qi_xi = update_log_qi_xi(xi, log_qi, s_samp)
                log_qi_seq.append(log_qi_xi)

            log_qi_new = tf.stack(log_qi_seq, axis=1)

            max_log_qi = tf.reduce_max(log_qi_new, axis=-1, keepdims=True)
            log_qi_shifted = log_qi_new - tf.stop_gradient(max_log_qi)

            return log_qi_shifted

        return parallel_update_log_qi(log_qi)


class ParallelMFSNetS1():
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
        self.n_layers = n_layers

        mfs_net_layer = partial(ParallelMFSNetLayerS1, kernel_initializer=kernel_initializer)

        self.log_qi = Input(shape=(n_in, n_const), name='log_qi_input')
        self.y = Input(shape=(n_out,), name='y_input')
        self.H = Input(shape=(n_out, n_in), name='H_input')
        self.r2rho = Input(shape=(), name='r2rho_input')
        self.n_var = Input(shape=(), name='n_var_input')

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(mfs_net_layer(n_in, n_const, n_samp, syms_re))

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
        for layer in self.layers:
            log_qi_out = layer(l_in)
            l_in = [log_qi_out, G, r2rho, n_var]

        # return log_qi
        # loss function may require this instead of qi

        #return Model(inputs=[y, H],
        return Model(inputs=[log_qi_in, y, H, r2rho, n_var],
                     outputs=log_qi_out , name='ParallelMFSNetS1')



################################################################################
# learn H function
################################################################################

class MLP(Layer):
    '''
    Implement MLP with N hidden layers"
    '''
    def __init__(self, units):
        super().__init__()

        kernel_init = 'VarianceScaling'
        activation = 'relu'
        dense_hidden_layer = partial(Dense, activation=activation,
                                     use_bias=True,
                                     kernel_initializer=kernel_init)
        dense_output_layer = partial(Dense, activation=None,
                                     use_bias=True,
                                     kernel_initializer=kernel_init)

        self.dense_h_layer = dense_hidden_layer(units)
        #self.dense_h_layer2 = dense_hidden_layer(units)
        self.dense_o_layer = dense_output_layer(units)

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        #tf.print('Alpha:training', training)
        x = inputs
        x = self.dense_h_layer(x)
        #x = self.dense_h_layer2(x)
        x = self.dense_o_layer(x)

        return x


class AlphaFunction(Layer):
    '''
    Implement alpha function
    '''
    def __init__(self, units):
        super().__init__()

        kernel_init = None
        activation = 'relu'
        dense_nonlinear_layer = partial(Dense, activation=activation,
                                        use_bias=True,
                                        kernel_initializer=kernel_init)
        dense_linear_layer = partial(Dense, activation=None,
                                     use_bias=True,
                                     kernel_initializer=kernel_init)

        self.dense_nl_layer = dense_nonlinear_layer(units)
        self.dense_l_layer = dense_nonlinear_layer(1)

    def build(self, input_shape):
        #initializer = tf.initializers.Constant(0.5)
        #trainable = True
        #self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
        #                        shape=(),
        #                        initializer=initializer,
        #                        constraint=NonNeg(),
        #                        trainable=trainable)
        pass

    def call(self, inputs, training=False):
        n_var = inputs
        n_var_clipped = tf.minimum(1.0, n_var)
        n_var_comp = 1.0 - n_var_clipped
        x = n_var_comp
        x = self.dense_nl_layer(x)
        x = self.dense_l_layer(x)
        alpha_out = n_var_comp + x
        #tf.print('n_var', n_var[0], 'in', n_var_comp[0], 'out', alpha_out[0])

        return alpha_out
        #return n_var_comp
        #return self.w


class HFunction(Layer):

    def __init__(self, units):
        super().__init__()

        alpha_units = 8
        #alpha_units = 16
        self.alpha_fcn = AlphaFunction(alpha_units)

        activation = 'relu'
        #activation = 'elu'
        #activation = 'swish'
        #activation = 'sigmoid'
        kernel_init = 'VarianceScaling'
        batch_norm = True
        #batch_norm = False
        #kernel_reg = tf.keras.regularizers.l2(1e-4)
        #kernel_reg = tf.keras.regularizers.l2(1e-3)
        kernel_reg = None
        residual_layer = partial(ResidualV2, activation=activation,
                                             kernel_initializer=kernel_init,
                                             kernel_regularizer=kernel_reg,
                                             batch_normalization=batch_norm)
        dense_linear_layer = partial(Dense, activation=None,
                                            kernel_initializer=kernel_init,
                                            kernel_regularizer=kernel_reg)

        #kernel_init = tf.initializers.Constant(0.0)
        dense_output_layer = partial(HyperDenseV2, activation=None,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)

        dense_in_layer = partial(HyperDenseV2, activation=None,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)
        dense_hidden_layer = partial(HyperDenseV2, activation=activation,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)

        init_zero = tf.initializers.Constant(0.0)
        dense_out_layer = partial(HyperDenseV2, activation=activation,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=init_zero,
                                     kernel_regularizer=kernel_reg)

        # expanded dimension
        #factor = 2
        #self.expand_layer = dense_linear_layer(units*factor)
        #self.contract_layer = dense_output_layer(units)
        #self.contract_layer.trainable=False
        #self.res_fcn = residual_layer(units*factor)

        # augmented MLP (for learning the residue)
        self.in_layer = dense_in_layer(units*3)
        self.hidden_layer = dense_hidden_layer(units*3)
        self.out_layer = dense_out_layer(units)
        #self.out_layer.trainable=False

        # single residual layer
        #self.res_fcn = residual_layer(units)
        self.units = units

    def build(self, input_shape):

        pass


    def call(self, inputs, training=False):
        #tf.print('Alpha:training', training)
        # log_qi_diff.shape = [N, N_tx_re, N_sym_re]
        n_var, log_qi_diff = inputs

        shape = tf.shape(log_qi_diff)
        log_qi_diff = tf.reshape(log_qi_diff, (shape[0], self.units))

        alpha = self.alpha_fcn(n_var)
        alpha = alpha[:,:,tf.newaxis]
        alpha_clipped = tf.minimum(0.99, alpha)

        # expanded dimension
        #x = self.expand_layer(log_qi_diff)
        #x = self.res_fcn(x)
        #x = self.contract_layer(x)
        #log_qi_out = x

        # augmented MLP (for learning the residue)
        #x = self.in_layer(log_qi_diff)
        #x = self.hidden_layer(x)
        #x = self.out_layer(x)
        #log_qi_out = log_qi_diff + x

        # single residual layer
        #log_qi_out = self.res_fcn(log_qi_diff)
        log_qi_out = log_qi_diff

        log_qi_out = tf.reshape(log_qi_out, shape)

        return alpha_clipped * log_qi_out
        #return log_qi_out


class ParallelMFSNetLayerS2(Layer):
    '''
    Implements MFSNet layer for
      log_qi = update + H ( log_qi - update )
    where H is learned
    '''
    # define shared weight as class attribute
    w = None

    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re, h_fcn,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        self.syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)

        self.initializer = kernel_initializer
        self.h_fcn = h_fcn

    def build(self, input_shape):
        pass


    #@tf.function
    def call(self, inputs, training=False):
        # define layer computation
        # inputs = [log_qi, G, sqrt_2rho]
        # dimensions
        # log_qi.shape = [N, N_tx_re, N_sym_re]
        # G.shape = [N, N_rx_re, N_tx_re]

        N_tx_re = self.n_tx_re
        N_samp = self.n_samp

        #log_qi_in, G, sqrt_2rho = inputs
        log_qi, G, sqrt_2rho, n_var = inputs

        # compute sqrt_2rho
        n_var_clipped = tf.maximum(0.01, n_var)
        #n_var_clipped = tf.maximum(0.03, n_var)
        rho = 1./n_var_clipped
        sqrt_2rho = tf.sqrt(2.*rho)

        # pre-proc n_var
        n_var_clipped = n_var_clipped[:,tf.newaxis]

        # compute alpha
        #w = self.w
        #alpha = tf.minimum(1., w * n_var_clipped)
        #alpha = alpha[:,tf.newaxis]

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
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
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
            log_p_mat = tf.math.log_sigmoid(c * term)
            # log_p_mat.shape = [N, N_rx_re, (N_samp * N_sym_re)]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
            log_p_tsr = tf.reshape(log_p_vec, (-1,shape[2],shape[3]))

            return log_p_tsr

        def construct_samples(log_qi):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            multi = tfp.distributions.Multinomial(1, logits=log_qi)
            #samp.shape = (N_samp, N, N_tx_re, N_sym_re)
            samp = multi.sample(N_samp)
            #indices.shape = (N_samp, N, N_tx_re)
            indices = tf.argmax(samp, axis=-1)
            #indices = tf.squeeze(indices, axis=3)
            i_samp = tf.transpose(indices, perm=[1,2,0])
            #i_samp.shape = (N, N_tx_re, N_samp)
            #s_samp = syms_re[i_samp] # in numpy
            i_samp = tf.expand_dims(i_samp, axis=-1)
            #s_samp.shape = (N, N_tx_re, N_samp)
            s_samp = tf.gather_nd(syms_re, i_samp)

            return s_samp

        def filter_samples(s_samp, xi_idx):
            '''
            s_samp.shape = (N, N_tx_re, N_samp)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            s_samp = tf.expand_dims(s_samp, axis=-1)
            # s_map.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_map = tf.repeat(s_samp, N_sym_re, axis=-1)

            # insert into s_map in numpy
            #s_map[xi_idx,:,:] = syms (numpy)
            # generate new map in tensorflow
            x_range = tf.range(N_tx_re)
            x_range = x_range[:, tf.newaxis, tf.newaxis]
            cond_insert = (x_range == xi_idx)
            s_map_new = tf.where(cond_insert, syms_re, s_map)

            return s_map_new


        def update_log_qi_xi(xi, log_qi, s_samp):
            #tf.print(f'update_log_qi: xi = ', xi)
            #alpha = self.alpha

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_map = filter_samples(s_samp, xi)
            # compute log p using samples
            #log_p = compute_log_p(G, s_map, sqrt_2rho)
            # use sigmoid approx
            log_p = compute_log_ps(G, s_map, sqrt_2rho)
            #log_p.shape (N, N_samp, N_tx_re)

            # approx with average
            ex_log_qi = tf.reduce_mean(log_p, axis=-2)
            # apply damping
            #log_qi_xi = (1-alpha) * log_qi[:,xi,:] + alpha * tf.stop_gradient(ex_log_qi)
            # pass mf update as is [S2]
            log_qi_xi = tf.stop_gradient(ex_log_qi)

            return log_qi_xi

        def parallel_update_log_qi(log_qi, n_var):

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_samp = construct_samples(log_qi)

            log_qi_seq = []
            for xi in range(N_tx_re): # unroll loop
                log_qi_xi = update_log_qi_xi(xi, log_qi, s_samp)
                log_qi_seq.append(log_qi_xi)

            #log_qi_new = tf.stack(log_qi_seq, axis=1)
            # compute learned function
            log_qi_update = tf.stack(log_qi_seq, axis=1)
            log_qi_diff = log_qi - log_qi_update
            #alpha = 0.99
            #log_qi_new = log_qi_update + alpha * log_qi_diff
            #log_qi_new = log_qi_update
            log_qi_new = log_qi_update + self.h_fcn((n_var, log_qi_diff))

            max_log_qi = tf.reduce_max(log_qi_new, axis=-1, keepdims=True)
            log_qi_shifted = log_qi_new - tf.stop_gradient(max_log_qi)

            return log_qi_shifted

        ''' main logic in parallel function '''
        return parallel_update_log_qi(log_qi, n_var_clipped)


class ParallelMFSNetS2():
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
        self.n_layers = n_layers

        mfs_net_layer = partial(ParallelMFSNetLayerS2, kernel_initializer=kernel_initializer)

        self.log_qi = Input(shape=(n_in, n_const), name='log_qi_input')
        self.y = Input(shape=(n_out,), name='y_input')
        self.H = Input(shape=(n_out, n_in), name='H_input')
        self.r2rho = Input(shape=(), name='r2rho_input')
        self.n_var = Input(shape=(), name='n_var_input')

        # common function
        self.h_fcn = HFunction(n_in * n_const)

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(mfs_net_layer(n_in, n_const, n_samp, syms_re, self.h_fcn))

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
        for layer in self.layers:
            log_qi_out = layer(l_in)
            l_in = [log_qi_out, G, r2rho, n_var]

        # return log_qi
        # loss function may require this instead of qi

        #return Model(inputs=[y, H],
        return Model(inputs=[log_qi_in, y, H, r2rho, n_var],
                     outputs=log_qi_out , name='ParallelMFSNetS2')


################################################################################
# RIM-RNN implementation
################################################################################

class InitFunction(Layer):

    def __init__(self, units, gru_units):
        super().__init__()

        activation = 'relu'
        #activation = 'elu'
        #activation = 'swish'
        #activation = 'sigmoid'
        #kernel_init = 'VarianceScaling'
        kernel_init = None
        batch_norm = True
        #batch_norm = False
        kernel_reg = None

        #kernel_init = tf.initializers.Constant(0.0)
        dense_in_layer = partial(HyperDenseV2, activation=None,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)
        dense_out_layer = partial(HyperDenseV2, activation=activation,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)


        # augmented MLP (for learning the residue)
        self.in_layer = dense_in_layer(units)
        self.out_layer = dense_out_layer(gru_units)
        #self.out_layer.trainable=False

        self.units = units
        self.gru_units = gru_units

    def build(self, input_shape):

        pass


    def call(self, inputs, training=False):
        # implements
        # state_new = self.h1([log_qi_diff, state])
        # log_qi_diff.shape = [N, N_tx_re, N_sym_re]

        n_var = inputs

        x = self.in_layer(n_var)
        state_init = self.out_layer(x)

        return state_init


class HsFunction(Layer):

    def __init__(self, units, gru_units):
        super().__init__()

        activation = 'relu'
        #activation = 'elu'
        #activation = 'swish'
        #activation = 'sigmoid'
        #kernel_init = 'VarianceScaling'
        kernel_init = None
        batch_norm = True
        #batch_norm = False
        kernel_reg = None

        #kernel_init = tf.initializers.Constant(0.0)
        dense_in_layer = partial(HyperDenseV2, activation=None,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)
        dense_hidden_layer = partial(HyperDenseV2, activation=activation,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)
        dense_out_layer = partial(HyperDenseV2, activation=activation,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)


        self.gru = tf.keras.layers.GRUCell(gru_units)

        self.units = units
        self.gru_units = gru_units

    def build(self, input_shape):

        pass


    def call(self, inputs, training=False):
        # implements
        # state_new = self.h1([log_qi_diff, state])
        # log_qi_diff.shape = [N, N_tx_re, N_sym_re]

        log_qi_diff, state = inputs
        shape = tf.shape(log_qi_diff)

        # flatten log_qi, log_qi_update
        log_qi_diff = tf.reshape(log_qi_diff, (shape[0], self.units))
        inputs_comb = tf.concat((log_qi_diff, state), axis=1)

        # input to GRU
        output, state_new = self.gru(inputs_comb, state)

        return state_new


class HmFunction(Layer):

    def __init__(self, units, hidden_units):
        super().__init__()

        activation = 'relu'
        #activation = 'elu'
        #activation = 'swish'
        #activation = 'sigmoid'
        #kernel_init = 'VarianceScaling'
        kernel_init = None
        batch_norm = True
        #batch_norm = False
        kernel_reg = None

        #kernel_init = tf.initializers.Constant(0.0)
        dense_in_layer = partial(HyperDenseV2, activation=None,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)
        dense_hidden_layer = partial(HyperDenseV2, activation=activation,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)
        dense_out_layer = partial(HyperDenseV2, activation=activation,
                                     batch_normalization=batch_norm,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg)


        # augmented MLP (for learning the residue)
        h1_units = hidden_units
        h2_units = hidden_units
        self.in_layer = dense_in_layer(h1_units)
        self.hidden_layer = dense_hidden_layer(h2_units)
        self.out_layer = dense_out_layer(units)
        #self.out_layer.trainable=False

        self.units = units
        self.hidden_units = hidden_units

    def build(self, input_shape):

        pass


    def call(self, inputs, training=False):
        # implements
        # corr_term = self.h2([log_qi_diff, state_new])
        # log_qi_diff.shape = [N, N_tx_re, N_sym_re]

        log_qi_diff, state_new = inputs
        shape = tf.shape(log_qi_diff)

        # flatten log_qi, log_qi_update
        log_qi_diff = tf.reshape(log_qi_diff, (shape[0], self.units))
        inputs_comb = tf.concat((log_qi_diff, state_new), axis=1)

        # augmented MLP (for learning the residue)
        x = self.in_layer(inputs_comb)
        x = self.hidden_layer(x)
        x = self.out_layer(x)

        log_qi_corr = tf.reshape(x, shape)

        return log_qi_corr


class ParallelMFSNetLayerS3(Layer):
    '''
    Implements MFSNet layer for
      log_qi = update + H ( log_qi - update )
      let m_bar = current message
      let m     = mf update
          n_var = (sigma^2)
      m_bar = m + hm ( m, m_bar, n_var, state )
      state = hs ( m, m_bar, n_var, state )
    '''
    # define shared weight as class attribute
    w = None

    def __init__(self, n_tx_re, n_sym_re, n_samp, syms_re, hs_fcn, hm_fcn,
                       kernel_initializer=None):
        super().__init__()
        self.n_tx_re = n_tx_re
        self.n_sym_re = n_sym_re
        self.n_samp = n_samp
        self.syms_re = tf.convert_to_tensor(syms_re, dtype=tf.float32)

        self.initializer = kernel_initializer
        self.hs_fcn = hs_fcn
        self.hm_fcn = hm_fcn

    def build(self, input_shape):
        pass


    #@tf.function
    def call(self, inputs, training=False):
        # define layer computation
        # inputs = [log_qi, G, sqrt_2rho]
        # dimensions
        # log_qi.shape = [N, N_tx_re, N_sym_re]
        # G.shape = [N, N_rx_re, N_tx_re]

        N_tx_re = self.n_tx_re
        N_samp = self.n_samp

        #log_qi_in, G, sqrt_2rho = inputs
        log_qi, G, sqrt_2rho, n_var, state = inputs

        # compute sqrt_2rho
        n_var_clipped = tf.maximum(0.01, n_var)
        rho = 1./n_var_clipped
        sqrt_2rho = tf.sqrt(2.*rho)

        # pre-proc n_var
        n_var_clipped = n_var_clipped[:,tf.newaxis]

        # compute alpha
        #w = self.w
        #alpha = tf.minimum(1., w * n_var_clipped)
        #alpha = alpha[:,tf.newaxis]

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
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
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
            log_p_mat = tf.math.log_sigmoid(c * term)
            # log_p_mat.shape = [N, N_rx_re, (N_samp * N_sym_re)]
            log_p_vec = tf.reduce_sum(log_p_mat, axis=1)
            # log_p_tsr.shape = [N, N_samp , N_sym_re]
            log_p_tsr = tf.reshape(log_p_vec, (-1,shape[2],shape[3]))

            return log_p_tsr

        def construct_samples(log_qi):
            '''
            log_qi.shape = (N, N_tx_re, N_sym_re)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            multi = tfp.distributions.Multinomial(1, logits=log_qi)
            #samp.shape = (N_samp, N, N_tx_re, N_sym_re)
            samp = multi.sample(N_samp)
            #indices.shape = (N_samp, N, N_tx_re)
            indices = tf.argmax(samp, axis=-1)
            #indices = tf.squeeze(indices, axis=3)
            i_samp = tf.transpose(indices, perm=[1,2,0])
            #i_samp.shape = (N, N_tx_re, N_samp)
            #s_samp = syms_re[i_samp] # in numpy
            i_samp = tf.expand_dims(i_samp, axis=-1)
            #s_samp.shape = (N, N_tx_re, N_samp)
            s_samp = tf.gather_nd(syms_re, i_samp)

            return s_samp

        def filter_samples(s_samp, xi_idx):
            '''
            s_samp.shape = (N, N_tx_re, N_samp)
            '''
            N_tx_re = self.n_tx_re
            N_samp = self.n_samp
            N_sym_re = self.n_sym_re
            syms_re = self.syms_re

            s_samp = tf.expand_dims(s_samp, axis=-1)
            # s_map.shape = (N, N_tx_re, N_samp, N_sym_re)
            s_map = tf.repeat(s_samp, N_sym_re, axis=-1)

            # insert into s_map in numpy
            #s_map[xi_idx,:,:] = syms (numpy)
            # generate new map in tensorflow
            x_range = tf.range(N_tx_re)
            x_range = x_range[:, tf.newaxis, tf.newaxis]
            cond_insert = (x_range == xi_idx)
            s_map_new = tf.where(cond_insert, syms_re, s_map)

            return s_map_new


        def update_log_qi_xi(xi, log_qi, s_samp):
            #tf.print(f'update_log_qi: xi = ', xi)
            #alpha = self.alpha

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_map = filter_samples(s_samp, xi)
            # compute log p using samples
            #log_p = compute_log_p(G, s_map, sqrt_2rho)
            # use sigmoid approx
            log_p = compute_log_ps(G, s_map, sqrt_2rho)
            #log_p.shape (N, N_samp, N_tx_re)

            # approx with average
            ex_log_qi = tf.reduce_mean(log_p, axis=-2)
            # apply damping
            #log_qi_xi = (1-alpha) * log_qi[:,xi,:] + alpha * tf.stop_gradient(ex_log_qi)
            # pass mf update as is [S2]
            log_qi_xi = tf.stop_gradient(ex_log_qi)

            return log_qi_xi

        def parallel_update_log_qi(log_qi, n_var, state):

            # NOTE: pass log_qi directly to tfp.Multinormial
            # sample from qi
            s_samp = construct_samples(log_qi)

            log_qi_seq = []
            for xi in range(N_tx_re): # unroll loop
                log_qi_xi = update_log_qi_xi(xi, log_qi, s_samp)
                log_qi_seq.append(log_qi_xi)

            #log_qi_new = tf.stack(log_qi_seq, axis=1)

            # compute learned function
            log_qi_update = tf.stack(log_qi_seq, axis=1)
            log_qi_diff = log_qi - log_qi_update
            #alpha = 0.99
            #log_qi_new = log_qi_update + alpha * log_qi_diff
            #log_qi_new = log_qi_update


            # reversed expression
            #log_qi_update = tf.stack(log_qi_seq, axis=1)
            #log_qi_diff = log_qi_update - log_qi
            #alpha = 0.01
            #log_qi_new = log_qi + alpha * log_qi_diff

            state_new = self.hs_fcn([log_qi_diff, state])
            corr_term = self.hm_fcn([log_qi_diff, state_new])

            #log_qi_new = log_qi + corr_term
            log_qi_new = log_qi_update + corr_term
            #log_qi_new = log_qi_update
            #state_new = []

            max_log_qi = tf.reduce_max(log_qi_new, axis=-1, keepdims=True)
            log_qi_shifted = log_qi_new - tf.stop_gradient(max_log_qi)

            return log_qi_shifted, state_new

        ''' main logic in parallel function '''
        return parallel_update_log_qi(log_qi, n_var_clipped, state)


class ParallelMFSNetS3():
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
        self.n_layers = n_layers

        mfs_net_layer = partial(ParallelMFSNetLayerS3, kernel_initializer=kernel_initializer)

        self.log_qi = Input(shape=(n_in, n_const), name='log_qi_input')
        self.y = Input(shape=(n_out,), name='y_input')
        self.H = Input(shape=(n_out, n_in), name='H_input')
        self.r2rho = Input(shape=(), name='r2rho_input')
        self.n_var = Input(shape=(), name='n_var_input')

        # common functions
        units = n_const * n_in
        gru_units = 64 # GRU state
        int_units = 8 # internal state
        self.init_fcn = InitFunction(int_units, gru_units)
        self.hs_fcn = HsFunction(units, gru_units)
        hidden_units = 32
        self.hm_fcn = HmFunction(units, hidden_units)

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(mfs_net_layer(n_in, n_const, n_samp, syms_re, 
                                        self.hs_fcn, self.hm_fcn))

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

        n_var_ex = n_var[:,tf.newaxis]
        n_var_clipped = tf.minimum(1.0, n_var_ex)
        state = self.init_fcn(n_var_ex)
        #state = []

        l_in = [log_qi_in, G, r2rho, n_var, state]
#        l_out = []
        for layer in self.layers:
            log_qi_out, state_new = layer(l_in)
            l_in = [log_qi_out, G, r2rho, n_var, state_new]
#            l_out.append(log_qi_out)

#        log_qi_all = tf.stack(l_out)

        # return log_qi
        # loss function may require this instead of qi

        #return Model(inputs=[y, H],
        return Model(inputs=[log_qi_in, y, H, r2rho, n_var],
                     #outputs=[log_qi_out, log_qi_all], name='ParallelMFSNetS3')
                     outputs=log_qi_out, name='ParallelMFSNetS3')


