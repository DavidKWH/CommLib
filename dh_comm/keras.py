import tensorflow as tf
from tensorflow.keras.layers import Layer

################################################################################
# define dense layer with constant input
################################################################################
class CommSymDense(tf.keras.layers.Layer):
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


