import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.metrics import MeanMetricWrapper
from functools import partial

################################################################################
# BER metric
################################################################################
def bit_error_rate(y_true, y_pred):
    # assumes bit input
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)
    errs = tf.cast(tf.not_equal(y_true, y_pred), tf.float32)
    return tf.reduce_mean(errs, axis=-1)

class BitErrorRate(MeanMetricWrapper):
    '''
    Based on Accuracy class in tf.keras.metric
    '''
    def __init__(self, name='BER', dtype=None):
        super(BitErrorRate, self).__init__(bit_error_rate, name, dtype=dtype)

################################################################################
# Custom Layers
################################################################################
class BatchNormedDense(Layer):
    '''
    Implements batch normalized dense layer (Ioffe, Szegedy, 2015)
    '''
    def __init__(self, units,
                       activation=None,
                       kernel_initializer=None):
        super(BatchNormedDense, self).__init__()

        dense_linear_layer = partial(Dense, activation=None,
                                            use_bias=False,
                                            kernel_initializer=kernel_initializer)
        batch_norm_layer = BatchNormalization

        self.d_layer = dense_linear_layer(units)
        self.bn_layer = batch_norm_layer()
        self.activation = activation

    def call(self, inputs, training=False):
        x = inputs
        x = self.d_layer(x)
        x = self.bn_layer(x, training=training)
        if self.activation is not None:
            return self.activation(x)
        return x

class Residual(Layer):
    '''
    Implements residual layer (Kaiming He et al., 2015)
    Added batch normalization option

    NOTE: if input and output dimensions are not equal
          specify unmatched_dimensions=True, the residual
          layer becomes: f(x) + A*x
          instead of:    f(x) + x
    '''
    def __init__(self, units,
                       activation=None,
                       kernel_initializer=None,
                       batch_normalization=False,
                       unmatched_dimensions=False):
        super(Residual, self).__init__()

        assert( activation is not None )
        assert( kernel_initializer is not None )

        batch_norm = batch_normalization
        use_bias = not batch_normalization
        dense_linear_layer = partial(Dense, activation=None,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer)
        batch_norm_layer = BatchNormalization

        self.res_layer_1 = dense_linear_layer(units)
        self.res_layer_2 = dense_linear_layer(units)
        self.bn_layer_1 = batch_norm_layer() if batch_norm else None
        self.bn_layer_2 = batch_norm_layer() if batch_norm else None

        self.activation = activation
        self.batch_norm = batch_norm
        self.unmatched_dims = unmatched_dimensions
        self.initializer = kernel_initializer
        self.units = units

    def build(self, input_shape):
        if self.unmatched_dims:
            assert( input_shape[-1] != self.units )
            self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
                                    shape=(input_shape[-1], self.units),
                                    initializer=self.initializer,
                                    trainable=True)
            self.transform = lambda x : tf.matmul( x, self.w )
        else:
            assert( input_shape[-1] == self.units )
            self.transform = lambda x : x

    def call(self, inputs, training=False):
        batch_norm = self.batch_norm
        x = inputs
        x = self.res_layer_1(x)
        if batch_norm: x = self.bn_layer_1(x, training=training)
        x = self.activation(x)
        x = self.res_layer_2(x)
        if batch_norm: x = self.bn_layer_2(x, training=training)
        x = x + self.transform(inputs)
        return self.activation(x)

class Square(Layer):
    '''
    Implements a dense layer with square activation
    '''
    def __init__(self, units,
                       kernel_initializer=None):
        super(Square, self).__init__()

        dense_output_layer = partial(Dense, activation=None,
                                            kernel_initializer=kernel_initializer)

        self.layer_1 = dense_output_layer(units)

    def call(self, inputs):
        x = inputs
        x = self.layer_1(x)
        return tf.math.square(x)

class MaxOut(Layer):
    '''
    Implements maxout layer (Goodfellow et al., 2013)

    DONE: add neural selection layer for gating candidates at the
          input to the max function
    '''
    def __init__(self, units,
                       kernel_initializer=None,
                       gate_function=tf.math.sigmoid):
        super(MaxOut, self).__init__()
        self.units = units
        self.initializer = kernel_initializer
        self.gate_fn = (gate_function if gate_function is not None
                                      else lambda x : x)

    def build(self, input_shape):
        self.w = self.add_weight(name='w', # needed for tf.saved_model.save()
                                 shape=(input_shape[-1], self.units),
                                 initializer=self.initializer,
                                 trainable=True)

    def call(self, inputs):
        w = self.w
        gate_fn = self.gate_fn
        gi = gate_fn(w)
        # element-wise multiply
        x = tf.math.multiply(inputs[:,:,None] , gi[None,:,:])
        out = tf.math.reduce_max(x, axis=2)
        return out

################################################################################
# Examples
################################################################################
'''
class BinaryTruePositives(Metric):
    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
            self.true_positives.assign_add(tf.reduce_sum(values))
    def result(self):
        return self.true_positives
'''

'''
class Linear(Layer):

  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b
'''

