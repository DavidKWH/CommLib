import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.metrics import MeanMetricWrapper

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

################################################################################
# define dense layer with constant input
################################################################################

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

'''
class CommSymDense(Layer):
    def __init__(self,
               units,
               syms,
               activation=None,
               use_bias=True,
               kernel_initializer=tf.keras.initializers.glorot_uniform,
               bias_initializer=tf.keras.initializers.zeros,
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.units = int(units)
        self.syms = syms
        self.syms_size = syms.size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.supports_masking = True
#        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `CSDense` '
                             'should be defined. Found `None`.')
#       self.input_spec = InputSpec(min_ndim=2,
#                                   axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        self.kernel_sym = self.add_variable('kernel_sym',
                                        shape=[self.syms_size, self.units],
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                            shape=[self.units,],
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            dtype=self.dtype,
                                            trainable=True)
        else:
            self.bias = None
        self.built = True


    def call(self, inputs):
        syms = tf.convert_to_tensor(self.syms, dtype=self.dtype)
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        outputs = tf.matmul(inputs, self.kernel) + tf.matmul(syms, self.kernel_sym)
        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs
'''
