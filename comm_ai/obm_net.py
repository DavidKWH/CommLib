import tensorflow as tf
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

import tensorflow_probability as tfp
norm = tfp.distributions.Normal(loc=0., scale=1.)
laplace = tfp.distributions.Laplace(loc=0., scale=1.)

def rhazard(x):
    out = tf.exp( norm.log_prob(x) - norm.log_cdf(x) )
    return out

def laplace_rhazard(x):
    out = tf.exp( laplace.log_prob(x) - laplace.log_cdf(x) )
    return out

class NormalizeLayer(Layer):
    '''
    Implements Normalize layer
    '''
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.sqrt_K = tf.sqrt(K)

    def build(self, input_shape):
        pass

    def call(self, inputs, training=False):
        # dimensions
        # x.shape = [Nb, Nx]
        x = inputs

        # normalize output
        x_mag = tf.norm(x, axis=1, keepdims=True)
        x = self.sqrt_K / x_mag * x

        return x

class OBMNetLayer(Layer):
    '''
    Implements OBMNet layer
    '''
    # define shared weight as class attribute
    beta = None

    def __init__(self, n_in, n_out,
                       kernel_initializer=None):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.initializer = kernel_initializer
        self.sigmoid = tf.keras.activations.sigmoid

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', # needed for tf.saved_model.save()
                                     shape=(),
                                     initializer=self.initializer,
                                     trainable=True)

        # class attribute (shared)
        if OBMNetLayer.beta is None:
            OBMNetLayer.beta = self.add_weight(name='beta', # needed for tf.saved_model.save()
                                     shape=(),
                                     initializer=self.initializer,
                                     #constraint=NonNeg(),
                                     trainable=True)

        # object level
        #self.beta = self.add_weight(name='beta', # needed for tf.saved_model.save()
        #                            shape=(),
        #                            initializer=self.initializer,
        #                            constraint=NonNeg(),
        #                            trainable=True)

    def call(self, inputs, training=False):
        # define layer computation
        # inputs = [x, G]
        # dimensions
        # x.shape = [Nb, Nx]
        # G.shape = [N_out, n_in]

        x_in, G = inputs

        # compute Gx
        x = tf.expand_dims(x_in, axis=1)
        #x = tf.matmul(x, -tf.transpose(G, perm=[0,2,1]) )
        x = tf.matmul(x, -G, transpose_b=True)
        s = self.sigmoid(self.beta * x)
        #s = rhazard(self.beta * x)
        #s = laplace_rhazard(self.beta * x)

        # compute G^T s
        x = tf.matmul(s, G)
        x = tf.squeeze(x, axis=1)
        x = self.alpha * x
        x = x + x_in

        return x

class OBMNet():
    '''
    Implements OBMNet
    '''
    def __init__(self, y_dim, x_dim,
                       n_layers,
                       kernel_initializer=None):
        '''
        NOTE: channel dimension is inferred
        '''
        super().__init__()
        self.n_in = n_in = x_dim
        self.n_out = n_out = y_dim
        K = tf.convert_to_tensor(x_dim // 2, dtype=tf.float32)

        obm_net_layer = partial(OBMNetLayer, kernel_initializer=kernel_initializer)

        self.x = Input(shape=(x_dim,), name='x_input')
        self.y = Input(shape=(y_dim,), name='y_input')
        self.H = Input(shape=(y_dim,x_dim), name='H_input')

        layers = []
        # build all layers
        for i in range(n_layers):
            layers.append(obm_net_layer(n_in, n_out))

        self.layers = layers
        self.norm_layer = NormalizeLayer(K)

    def build(self):
        '''
        build model using functional API
        '''
        # set initial x to zero
        #x_shape = (1, self.n_in)
        #x_in = tf.zeros(x_shape)
        x_in = self.x
        y = self.y
        H = self.H
        G = tf.linalg.diag(y) @ H

        l_in = [x_in, G]
        for layer in self.layers:
            x_out = layer(l_in)
            l_in = [x_out, G]

        # normalize output
        x_out = self.norm_layer(x_out)

        #return Model(inputs=[y, H],
        return Model(inputs=[x_in, y, H],
                     outputs=x_out , name='OBMNet')


