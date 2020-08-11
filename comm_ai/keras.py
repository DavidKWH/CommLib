import tensorflow as tf
import numpy as np
# loss specific
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.keras import backend as K
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
# debug
from .util import DumpFullTensor

################################################################################
# Custom Losses and Metrics
################################################################################
def complement_entropy(y_true, y_pred,
                       alpha=0.,
                       beta=1.,
                       guided_comp_entropy=False,
                       from_logits=True):
    '''
    Implement compliment entropy loss. See
    Complemetary Objective Training, Hao-Yun Chen et al., 2019
    Improving Adversarial Robustness via Guided Complement Entropy,
    Hao-Yun Chen et al., 2019
    NOTE: numerical conditioning in (1),(2)
    NOTE: any arguments can be passed into this function via
          the LossFunctionWrapper class

    Default COT computes sample complement entropy
        samp_ent = q * log_q * mask
    When GCE is selected
        samp_ent = Yg^alpha * q * log_q * mask

    Assumptions:
        y_true are tensor of integers of rank R
        y_pred are tensor of logits of rank R+1
        axis = batch_dims = R
    '''
    y_pred = tf.convert_to_tensor(y_pred)
    num_classes = y_pred.shape[-1]
    # NOTE: the two statements don't work in graph mode...
    #axis = batch_dims = y_true.ndim
    #axis = batch_dims = tf.rank(y_true)
    axis = batch_dims = K.ndim(y_true)
    zeros = tf.constant(0.)
    ones = tf.constant(1.)

    p_hat = tf.nn.softmax(y_pred)
    # get predicted prob of ground truth class
    Yg = tf.gather(p_hat, y_true[...,None],
                   axis=axis,
                   batch_dims=batch_dims)
#    print('sum_Yg',K.sum(Yg))
    # compute 1-Yg
    Yg_comp = (1.-Yg) + 1e-7 # (1)
    # compute normalized distribution q_hat
    q = p_hat / Yg_comp
    log_q = tf.math.log(q + 1e-10) # (2)
    # compute mask for complementary classes
    mask = tf.one_hot(y_true, num_classes,
                      on_value=zeros,
                      off_value=ones)
#    print(y_true)
#    print(tf.where(mask == 0))
#    with DumpFullTensor():
#        print('full mask')
#        print(mask.numpy())
    # sample complement entropy
    samp_ent = q * log_q * mask
    # GCE specific
    if guided_comp_entropy:
        alpha = tf.cast(alpha, y_pred.dtype)
        Yg_alpha = Yg**alpha
        samp_ent *= Yg_alpha
    # average over complementary set
    loss = tf.reduce_mean(samp_ent, axis=-1)
    # scale according to paper recommendation
    loss *= beta
    # NOTE: minimizing negative entropy
    return loss

class ComplementEntropy(LossFunctionWrapper):
    '''
    Wrapper class for complement_entropy()
    '''
    def __init__(self,
                 alpha=0.,
                 beta=1.,
                 guided_comp_entropy=False,
                 from_logits=True,
                 name='complement_entropy'):
        assert from_logits
        super().__init__(complement_entropy,
                         alpha=alpha,
                         beta=beta,
                         guided_comp_entropy=guided_comp_entropy,
                         from_logits=from_logits,
                         name=name)

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
        self.step = tf.Variable(0.)

    def update(self):
        # must be called synchronized
        self.step.assign_add(1.)

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

################################################################################
# Custom Learning Rate Schedules
################################################################################
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
class GraphConv(Layer):
    '''
    Implements a single GCN layer (Kipf and Welling, 2017)

    NOTE:
        X is a NxC matrix, each row x_i is a 1xC vector of features
        N is the number of nodes in the graph
    '''
    def __init(self, units=None,
                     activation=None,
                     kernel_initializer=None,
                     adjacency_matrix=None,
                     **kwargs):
        super().__init__(**kwargs)

        assert A, "must specify adjacency matrix"
        A = adjacency_matrix
        # square matrix
        assert A.ndim == 2 and len(set(A.shape)) == 1, \
                "adjacency matrix must be square"
        A_til = A + np.eye(A.shape[0])
        d_til = np.sum(A_til, axis=1)
        D_til = np.diag(d_til)
        d_til_nh = - np.sqrt(d_til)
        D_til_nh = np.diag(d_til_nh)
        # precompute (D^{-half} A D^{-half})
        A_hat = D_til_nh @ A @ D_til_nh
        # convert to sparse tensor
        # NOTE: have to use tf.sparse routines for computations
        #       see tf.sparse docs
        A_hat_sparse = tf.sparse.from_dense(A_hat)

        self.A_hat = A_hat
        # input must have the same dimension as anyone dimension of A
        self.units = unit
        self.activation = activation
        self.initializer = kernel_initializer

    def build(self, input_shape):
        '''
        input.shape = (B, N, C)
            B = batch size
            N = num nodes in graph
            C = num features per node

        W.shape = (C, H)
            C = input_shape(-1)
            H = units,
        '''
        self.W = self.add_weight(name='W', # needed for tf.saved_model.save()
                                shape=(input_shape[-1], self.units),
                                initializer=self.initializer,
                                trainable=True)

    def call(self, inputs, training=False):
        '''
        input.shape = (B, N, inputs.shape[-1])
        output.shape = (B, N, units)
        '''

        # function to be applied to each batch sample
        def graph_conv(X):
            W = self.W
            A_hat = self.A_hat
            return A_hat @ X @ W

        x = inputs
        x = tf.map_fn(graph_conv, x)
        if self.batch_norm:
            x = self.bn_layer(x, training=training)
        if self.activation is not None:
            return self.activation(x)
        return x

class OneHot(Layer):
    '''
    Implements a one-hot representation conversion layer
    '''
    def __init__(self, depth, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

    def call(self, input, training=False):
        s_o = tf.one_hot(input, self.depth)
        s_o = tf.squeeze(s_o, axis=1)
        return s_o

class EFLM(Layer):
    '''
    Implements Embedded feature-wise linear modulation layer
    '''
    def __init__(self, units, depth,
                 kernel_initializer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.depth = depth
        self.initializer = kernel_initializer

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha', # needed for tf.saved_model.save()
                                     shape=(self.depth, self.units),
                                     initializer=self.initializer,
                                     trainable=True)

        self.beta = self.add_weight(name='beta', # needed for tf.saved_model.save()
                                    shape=(self.depth, self.units),
                                    initializer=self.initializer,
                                    trainable=True)

    def call(self, inputs, training=False):
        '''
        dimensions:
            x.shape = (N, units)
            s.shape = (N,)
            s_o.shape = (N, depth)
            self.alpha = (depth, units)
            self.beta = (depth, units)
            alpha.shape = (N, units)
            beta.shape = (N, units)
        '''
        # select alpha and beta for FLM
        x = inputs[0] # features
        s = inputs[1] # SNR (integers)
        s_o = tf.one_hot(s, self.depth)
        s_o = tf.squeeze(s_o, axis=1)
        alpha = tf.matmul(s_o, self.alpha)
        beta = tf.matmul(s_o, self.beta)

        # compute affine transformation
        output = alpha * x + beta
        return output

class FLMResidual(Layer):
    '''
    Implements residual layer (Kaiming He et al., 2015)
    Added batch normalization option
    Add feature wise linear modulation

    NOTE: if input and output dimensions are not equal
          specify unmatched_dimensions=True, the residual
          layer becomes: f(x) + A*x
          instead of:    f(x) + x
    '''
    def __init__(self, units,
                       activation=None,
                       kernel_initializer=None,
                       batch_normalization=False,
                       depth=None,
                       feature_linear_modulation=False,
                       unmatched_dimensions=False):
        super().__init__()

        assert( activation is not None )
        assert( kernel_initializer is not None )

        flm = feature_linear_modulation
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

        self.flm_layer_1 = EFLM(units, depth) if flm else None
        self.flm_layer_2 = EFLM(units, depth) if flm else None
        self.flm = flm

        self.activation = activations.get(activation)
        self.batch_norm = batch_norm
        self.unmatched_dims = unmatched_dimensions
        self.initializer = kernel_initializer
        self.units = units

    def build(self, input_shape):
        assert isinstance( input_shape, (list, tuple) )
        main_shape = input_shape[0]
        if self.unmatched_dims:
            assert( main_shape[-1] != self.units )
            self.w = self.add_weight(name='W', # needed for tf.saved_model.save()
                                    shape=(input_shape[-1], self.units),
                                    initializer=self.initializer,
                                    trainable=True)
            self.transform = lambda x : tf.matmul( x, self.w )
        else:
            assert( main_shape[-1] == self.units )
            self.transform = lambda x : x

    def call(self, inputs, training=False):
        batch_norm = self.batch_norm
        flm = self.flm
        x = inputs[0]
        s = inputs[1]
        x = self.dl_layer_1(x)
        if batch_norm: x = self.bn_layer_1(x, training=training)
        if flm: x = self.flm_layer_1([x, s])
        x = self.activation(x)
        x = self.dl_layer_2(x)
        if batch_norm: x = self.bn_layer_2(x, training=training)
        if flm: x = self.flm_layer_2([x, s])
        x = x + self.transform(inputs[0])
        return self.activation(x)

    def num_layers(self):
        return 2

class FLMDense(Layer):
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
                       depth=None,
                       feature_linear_modulation=False,
                       batch_normalization=False,
                       **kwargs):
        super().__init__(**kwargs)

        flm = feature_linear_modulation
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

        self.flm_layer = EFLM(units, depth) if flm else None
        self.flm = flm

    def call(self, inputs, training=False):
        x = inputs[0]
        s = inputs[1]
        x = self.dl_layer(x)
        if self.batch_norm:
            x = self.bn_layer(x, training=training)
        if self.flm:
            x = self.flm_layer([x,s])
        if self.activation is not None:
            return self.activation(x)
        return x

    def num_layers(self):
        return 1


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
                       batch_normalization=False,
                       **kwargs):
        super(HyperDense, self).__init__(**kwargs)

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

    def num_layers(self):
        return 1


class HyperDenseV2(Layer):
    '''
    Implements preactivated dense layer (for Resnet V2)
    Implements batch normalized dense layer (Ioffe and Szegedy, 2015)
    Enable via the batch_normalization option

    NOTE: The subclassed layer's name is inferred from the class name.
    NOTE: The Layer class produce the correct context (i.e. name scope)
          No special handling required
    '''
    def __init__(self, units,
                       activation=None,
                       kernel_initializer=None,
                       batch_normalization=False,
                       **kwargs):
        super().__init__(**kwargs)

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
        if self.batch_norm:
            x = self.bn_layer(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        x = self.dl_layer(x)
        return x

    def num_layers(self):
        return 1


class HyperNetDenseV2(Layer):
    '''
    Implements dense layer with layer embedding input from Hyper network (Ha et al, 2016)
    Implements batch normalized dense layer (Ioffe and Szegedy, 2015)
    Enable via the batch_normalization option

    NOTE: The subclassed layer's name is inferred from the class name.
    NOTE: The Layer class produce the correct context (i.e. name scope)
          No special handling required
    '''
    def __init__(self, units,
                       activation=None,
                       kernel_initializer=None,
                       batch_normalization=False,
                       **kwargs):
        super().__init__(**kwargs)

        batch_norm = batch_normalization
        use_bias = not batch_normalization
        #dense_linear_layer = partial(Dense, activation=None,
        #                                    use_bias=use_bias,
        #                                    kernel_initializer=kernel_initializer)
        batch_norm_layer = BatchNormalization

        #self.dl_layer = dense_linear_layer(units)
        self.bn_layer = batch_norm_layer() if batch_norm else None
        self.activation = activations.get(activation)
        self.batch_norm = batch_norm
        self.use_bias = use_bias
        self.units = units

    def build(self, input_shape):
        # inputs = [x, [z_weight, z_bias] ]
        x_shape = input_shape[0]
        self.weight_shape = (-1, x_shape[-1], self.units)

    def call(self, inputs, training=False):
        # x : input
        # z : embedding
        # W, b : from hyper network
        x, z = inputs
        w, b = z
        x = tf.reshape(x, (-1,1,x.shape[-1]) )
        W = tf.reshape(w, self.weight_shape)
        if self.batch_norm:
            x = self.bn_layer(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        #x = self.dl_layer(x)
        # implement Wx + b
        x = tf.squeeze(tf.matmul( x, W ), 1) + b
        # implement Wx + b, dependent on use_bias flag
        #x = tf.squeeze(tf.matmul( x, W ), 1)
        #if use_bias:
        #    x = x + b
        return x

    def num_layers(self):
        return 1



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

    def num_layers(self):
        return 2


class ResidualV2(Layer):
    '''
    Implements revised residual layer (Kaiming He et al., 2016)

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
        super().__init__()

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
        if batch_norm: x = self.bn_layer_1(x, training=training)
        x = self.activation(x)
        x = self.dl_layer_1(x)
        if batch_norm: x = self.bn_layer_2(x, training=training)
        x = self.activation(x)
        x = self.dl_layer_2(x)

        return inputs + x

    def num_layers(self):
        return 2


class HyperNetResidualV2(Layer):
    '''
    Implements revised residual layer (Kaiming He et al., 2016)
    With parameters passed in from Hyper network

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
        super().__init__()

        assert( activation is not None )
        assert( kernel_initializer is not None )

        batch_norm = batch_normalization
        use_bias = not batch_normalization
        hypernet_dense_layer = partial(HyperNetDenseV2, activation=None,
                                            #use_bias=use_bias,
                                            kernel_initializer=kernel_initializer,
                                            batch_normalization=False)
        batch_norm_layer = BatchNormalization

        self.dl_layer_1 = hypernet_dense_layer(units)
        self.dl_layer_2 = hypernet_dense_layer(units)
        self.bn_layer_1 = batch_norm_layer() if batch_norm else None
        self.bn_layer_2 = batch_norm_layer() if batch_norm else None

        self.activation = activations.get(activation)
        self.batch_norm = batch_norm
        self.unmatched_dims = unmatched_dimensions
        self.initializer = kernel_initializer
        self.units = units

    def build(self, input_shape):
        # input = [ x, [ z1_weight, z1_bias ], [ z2_weight, z2_bias] ]
        if self.unmatched_dims:
            x_shape = input_shape[0]
            assert( x_shape[-1] != self.units )
            self.w = self.add_weight(name='W', # needed for tf.saved_model.save()
                                    shape=(input_shape[-1], self.units),
                                    initializer=self.initializer,
                                    trainable=True)

            self.transform = lambda x : tf.matmul( x, self.w )
        else:
            x_shape = input_shape[0]
            assert( x_shape[-1] == self.units )
            self.transform = lambda x : x


    def call(self, inputs, training=False):
        # x = input to main network
        # z1, z2 = embedding from hyper network
        batch_norm = self.batch_norm
        xin, z1, z2 = inputs
        x = xin
        if batch_norm: x = self.bn_layer_1(x, training=training)
        x = self.activation(x)
        x = self.dl_layer_1([x, z1])
        if batch_norm: x = self.bn_layer_2(x, training=training)
        x = self.activation(x)
        x = self.dl_layer_2([x, z2])

        return xin + x

    def num_layers(self):
        return 2


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
# Self conceived layers
################################################################################
class Zeros(Layer):
    '''
    Splits out zeros with capatible shape
    NOTE: pass in batch_size to avoid issues
    '''
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.z = tf.zeros((input_shape[-1], self.units))
        self.zeros = lambda x : tf.matmul( x, self.z )

    def call(self, inputs):
        x = inputs
        return self.zeros(x)


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


