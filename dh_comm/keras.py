import tensorflow as tf
# metric specific
from tensorflow.python.keras.metrics import MeanMetricWrapper
# schedule specific
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
# layer specific
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import activations
#from tensorflow.keras import initializers
# misc.
from functools import partial

################################################################################
# Custom Metrics
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
# Custom Schedules
################################################################################
class MomentumSchedule:
    '''
    Implements momentum schedule in (Sutskever et al., 2013)

    NOTE: Always perform staircase updates

    NOTE: The step context is not passed to momentum callables.
          Have to manage step counter in the training loop manually
          by calling the update() method

    TODO: Will this 'synchronized' update cause a runtime penalty?
          This update requires a lock to the MT data structure.
    '''
    def __init__(self,
            initial_momentum=0.5,
            maximum_momentum=0.995,
            final_momentum=0.9,
            decay_steps=250,
            #staircase=True,
            name=None):
        super(MomentumSchedule, self).__init__()

        self.initial_momentum = initial_momentum
        self.maximum_momentum = maximum_momentum
        self.final_momentum = final_momentum
        self.decay_steps = decay_steps
        #self.staircase = staircase
        self.final_iteration = False;
        self.base = 1. - initial_momentum
        self.name = name
        self.step = 0

    def update(self):
        # must be called synchronized
        self.step += 1

    def __call__(self):
        step = self.step
        with tf.name_scope(self.name or 'MomentumSchedule') as name:
            initial_momentum = tf.convert_to_tensor( self.initial_momentum )
            dtype = initial_momentum.dtype
            maximum_momentum = tf.cast( self.maximum_momentum, dtype)
            final_momentum = tf.cast( self.final_momentum, dtype)
            decay_steps = tf.cast( self.decay_steps, dtype)
            base = tf.cast( self.base, dtype)
            global_step = tf.cast(step, dtype)

            if self.final_iteration:
                return final_momentum
            p = global_step / decay_steps
            #if self.staircase:
            p = tf.math.floor(p)
            # compute momentum term
            proposed = 1.0 - base / (p + 1)
            return tf.math.minimum( proposed , self.maximum_momentum )

    def final_iteration(self):
        self.final_iteration = True

    def get_config(self):
        # FIXME
        return {}

class LRDecaySchedule(LearningRateSchedule):
    '''
    Base class for periodic and aperiodic schedules
    '''
    def __init__(self):
        super(LRDecaySchedule, self).__init__()

    '''
    decay algorithms
    '''
    def power_based_decay(self, p):
        return tf.math.pow(self.decay_rate, p)

    def log_based_decay(self, p):
        return tf.math.pow(10., -p*self.decay_exp)

    # decay algo lookup
    decay_algos = {
        'power_based' : power_based_decay,
        'log_based'   : log_based_decay,
    }

class PeriodicLRDecaySchedule(LRDecaySchedule):
    '''
    Implements (periodic) learning rate schedule
    Based on exponential decay scheme used in (He et al., 2016)

    NOTE: Always perform staircase updates

    NOTE: The decay schedule exploits the homogeneous floating point
          pipeline available in GPU/TPU based hardware, e.g simple
          parallelizable unconditional scalar computations.  Thus,
          cannot assume integer/conditional logic is more efficient,
          as this is highly dependent on the available hardware.

    NOTE: Prefer parallelizable computation
    NOTE: No integer nodes available in graph computation

    decay_rate_dbp: decay in dB power
                    e.g. (10, 20, 30) dBP = (10x, 100x, 1000x)
                         (3, 6, 9) dBP = (2x, 4x, 8x)
    '''

    def __init__(self,
            initial_learning_rate = 0.1,
            minimum_learning_rate = 1e-4,
            decay_steps = 5000,
            decay_rate = None,
            decay_rate_dbp = 10,
            #staircase = True,
            name = None
            ):
        super(PeriodicLRDecaySchedule, self).__init__()
        to_tensor = partial( tf.convert_to_tensor, dtype=tf.float32 )
        self.initial_learning_rate = to_tensor(initial_learning_rate)
        self.minimum_learning_rate = to_tensor(minimum_learning_rate)
        self.decay_steps = to_tensor(decay_steps)
        self.decay_rate = to_tensor(decay_rate) if decay_rate is not None else None
        self.decay_exp = to_tensor(decay_rate_dbp / 10)
        #self.staircase = staircase
        self.name = name
        assert(decay_rate or decay_rate_dbp)
        decay_mode = 'power_based' if decay_rate is not None else 'log_based'
        self.decay_fn = partial(self.decay_algos[decay_mode], self)

    def __call__(self, step):
        '''
        NOTE: Only an instance of LearningRateSchedule is passed
              the current step during training
        '''
        with tf.name_scope(self.name or'LRDecaySchedule') as name:
            initial_lr = self.initial_learning_rate
            minimum_lr = self.minimum_learning_rate
            decay_steps = self.decay_steps
            global_step = tf.cast(step, decay_steps.dtype)

            p = global_step / decay_steps
            #if self.staircase:
            #    p = tf.math.floor(p)
            p = tf.math.floor(p)
            # compute exponential decay
            decay_factor = self.decay_fn(p)
            proposed = initial_lr * decay_factor
            #print('step {} proposed {}'.format(step, proposed))
            return tf.math.maximum( minimum_lr, proposed )

    def get_config(self):
        # FIXME
        return {}

class AperiodicLRDecaySchedule(LRDecaySchedule):
    '''
    Implements explicit absolute decay schedule in epochs

    decay_schedule: decay schedule specified via
                    zero-based indexing, e.g.
                    (0,1,2,...) = (1st, 2nd, 3rd) epoch
                    Thus, 0 is the starting epoch, should
                    not be in a typical decay_schedule.
    '''
    def __init__(self,
            initial_learning_rate = 0.1,
            minimum_learning_rate = 1e-4,
            #decay_steps = 5000,
            steps_per_epoch = 5000,
            decay_rate = None,
            decay_rate_dbp = 10,
            decay_schedule = [], # zero-based indexing
            #staircase = True,
            name = None
            ):
        super(AperiodicLRDecaySchedule, self).__init__()
        assert(decay_schedule)
        assert(decay_rate or decay_rate_dbp)
        to_tensor = partial( tf.convert_to_tensor, dtype=tf.float32 )

        self.initial_learning_rate = to_tensor(initial_learning_rate)
        self.minimum_learning_rate = to_tensor(minimum_learning_rate)
        self.steps_per_epoch = to_tensor(steps_per_epoch)
        self.decay_rate = to_tensor(decay_rate) if decay_rate is not None else None
        self.decay_exp = to_tensor(decay_rate_dbp / 10)
        self.decay_schedule = to_tensor(decay_schedule)
        #self.staircase = staircase
        self.name = name
        decay_mode = 'power_based' if decay_rate is not None else 'log_based'
        self.decay_fn = partial(self.decay_algos[decay_mode], self)

    def __call__(self, step):
        '''
        NOTE: Keep function stateless! Better parallelism

        NOTE: Only an instance of LearningRateSchedule is passed
              the current step during training
        '''
        with tf.name_scope(self.name or'AperiodicLRDecaySchedule') as name:
            initial_lr = self.initial_learning_rate
            minimum_lr = self.minimum_learning_rate
            steps_per_epoch = self.steps_per_epoch
            global_step = tf.cast(step, steps_per_epoch.dtype)
            decay_schedule = self.decay_schedule

            # compute exponent (use integer arithmetic)
            pr = global_step / steps_per_epoch
            pi  = tf.math.floor(pr)
            # tf.math.greater_equal returns bools
            ge_vec = (pi >= decay_schedule)
            ge_vec = tf.cast( ge_vec, tf.float32 )
            p = tf.reduce_sum(ge_vec)

            # compute exponential decay
            decay_factor = self.decay_fn(p)
            proposed = initial_lr * decay_factor
            #print('decay_sch {} ge_vec {}'.format(decay_schedule, ge_vec))
            #print('step {} p {} proposed {}'.format(step, p, proposed))
            return tf.math.maximum( minimum_lr, proposed )

    def get_config(self):
        # FIXME
        return {}
    pass

################################################################################
# Custom Layers
################################################################################
class HyperDense(Layer):
    '''
    Implements batch normalized dense layer (Ioffe and Szegedy, 2015)
    Enable via the batch_normalization option

    NOTE: The subclassed layer's name is inferred from the class name.
    NOTE: The Layer class produce the correct context (i.e. name scope)
          No special handling required
    '''
    def __init__(self, units,
                       activation=None,
                       kernel_initializer=None,
                       batch_normalization=False):
        super(HyperDense, self).__init__()

        batch_norm = batch_normalization
        use_bias = not batch_normalization
        dense_linear_layer = partial(Dense, activation=None,
                                            use_bias=use_bias,
                                            kernel_initializer=kernel_initializer)
        batch_norm_layer = BatchNormalization

        self.dl_layer = dense_linear_layer(units)
        self.bn_layer = batch_norm_layer() if batch_norm else None
        self.activation = activations.get(activation)
        self.batch_norm = batch_norm

    def call(self, inputs, training=False):
        x = inputs
        x = self.dl_layer(x)
        if self.batch_norm:
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

        self.dl_layer_1 = dense_linear_layer(units)
        self.dl_layer_2 = dense_linear_layer(units)
        self.bn_layer_1 = batch_norm_layer() if batch_norm else None
        self.bn_layer_2 = batch_norm_layer() if batch_norm else None

        self.activation = activations.get(activation)
        self.batch_norm = batch_norm
        self.unmatched_dims = unmatched_dimensions
        self.initializer = kernel_initializer
        self.units = units

    def build(self, input_shape):
        if self.unmatched_dims:
            assert( input_shape[-1] != self.units )
            self.w = self.add_weight(name='W', # needed for tf.saved_model.save()
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
        x = self.dl_layer_1(x)
        if batch_norm: x = self.bn_layer_1(x, training=training)
        x = self.activation(x)
        x = self.dl_layer_2(x)
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
# Old stuff
################################################################################
#class BatchNormedDense(Layer):
#    '''
#    Implements batch normalized dense layer (Ioffe and Szegedy, 2015)
#    '''
#    def __init__(self, units,
#                       activation=None,
#                       kernel_initializer=None):
#        super(BatchNormedDense, self).__init__()
#
#        dense_linear_layer = partial(Dense, activation=None,
#                                            use_bias=False,
#                                            kernel_initializer=kernel_initializer)
#        batch_norm_layer = BatchNormalization
#
#        self.d_layer = dense_linear_layer(units)
#        self.bn_layer = batch_norm_layer()
#        self.activation = activation
#
#    def call(self, inputs, training=False):
#        x = inputs
#        x = self.d_layer(x)
#        x = self.bn_layer(x, training=training)
#        if self.activation is not None:
#            return self.activation(x)
#        return x

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

