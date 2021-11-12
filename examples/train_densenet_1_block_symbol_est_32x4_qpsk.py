'''
# Final neural RX implementation
# Tensorflow classifier model for LLR computation
# Experiment with residual network
# Added batch normalization
# Added learning rate schedule for Adam and SGD
# Use new parameter structure
'''
from importlib import reload
from functools import partial
from sys import exit

import numpy as np
#import numpy.matlib as npm # optional package (must import explicitly)
#import numpy.random as rnd
#import matplotlib.pyplot as plt

#from tensorflow_probability import distributions as tfd
#import tensorflow_probability as tfp
#import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import concatenate
#from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import comm_ai as comm
#from comm_ai import QAMModulator
#from comm_ai import Demodulator
#from comm_ai import Transmitter
#from comm_ai import Receiver
#from comm_ai import Channel
from comm_ai.tensorflow import CommDataSet
from comm_ai.tensorflow import plot_model
from comm_ai.tensorflow import save_model
from comm_ai.keras import ComplementEntropy
from comm_ai.keras import BitErrorRate
#from comm_ai.keras import Residual
from comm_ai.keras import ResidualV2
#from comm_ai.keras import FLMResidual
#from comm_ai.keras import HyperDense
from comm_ai.keras import HyperDenseV2
from comm_ai.keras import DenseBlock
#from comm_ai.keras import Zeros
from comm_ai.keras import PeriodicLRDecaySchedule
from comm_ai.keras import AperiodicLRDecaySchedule
#from comm_ai.keras import MomentumSchedule
from comm_ai.util import Timer
from comm_ai.params import RecursiveParams as Params
from comm_ai.params import get_key
from comm_ai.params import has_key

reload(comm)
reload(comm.core)

tf_version = tf.__version__  #pylint: disable=no-member
#assert tf_version.startswith('2.0'), 'requires tensorflow-2.0.x'
print('tensorflow             ', tf_version)
#print('tensorflow_probability ', tfp.__version__)

p = Params()
'''
param structure
NOTE: fixed param specification!
NOTE: it is safer to specify the name
      of the initializer/activation_fn
      than the actual callable.
'''

### comm. related parameters
#########################################
p.one_bit = True
#p.one_bit = False
p.dc_bias = False
p.N_tx = 4
p.N_rx = 32
p.N_sts = p.N_tx
#p.M = 16 # modulation order
p.M = 4 # modulation order
p.nbps = np.log2(p.M).astype(int)
#p.snr_db = 20
#p.snr_db = 2
p.snr_db = 15
p.n_var = 10**(-p.snr_db/10)
p.n_std = np.sqrt(p.n_var)
#p.ch_type = 'identity'
#p.ch_type = 'rayleigh'
#p.ch_type = 'sv_dist'
#p.noise_type = 'fixed_var'
#p.noise_type = 'rand_var'
p.noise_dist = 'log_uniform'
# log-uniform random noise (dB)
#p.log_u_a = 5
p.log_u_a = -10
p.log_u_b = 20
# log-normal random noise (dB)
p.log_n_mean = 0
p.log_n_std = 1
# uniform sv distribution
p.u_a = 0.0
p.u_b = 3.0
# snr_range (for training)
#p.snr_lo = 0
#p.snr_hi = 15
p.snr_lo = -10
p.snr_hi = 20
# channel covariance spec
ch = Params()
ch.d  = 0.5 # unit lambda
ch.phi_vec_deg = [-30, 30] # location of user [-60, 60] in degrees
#ch.phi_vec_deg = [0] # location of user [-60, 60] in degrees
ch.phi_vec_deg = np.array(ch.phi_vec_deg)
ch.phi_vec_rad = ch.phi_vec_deg * np.pi / 180 # in radians
#ch.sig_phi_deg = 10. # angular standard distribution (ASD) in degrees
ch.sig_phi_deg = 10. # angular standard distribution (ASD) in degrees
ch.sig_phi_rad = ch.sig_phi_deg * np.pi / 180 # in radians; for computation
p.chan_cov = ch

# training modes
p.train0 = Params()
p.train0.ch_type = 'rayleigh'
p.train0.noise_type = 'fixed_var'
p.train1 = Params()
p.train1.ch_type = 'sv_dist'
p.train1.noise_type = 'fixed_var'
p.train2 = Params()
p.train2.ch_type = 'rayleigh'
p.train2.noise_type = 'rand_var'
p.train3 = Params()
p.train3.ch_type = 'rayleigh,batch_fixed'
p.train3.noise_type = 'rand_var'
p.train4 = Params()
p.train4.ch_type = 'rayleigh'
p.train4.noise_type = 'rand_snr'
p.train5 = Params()
p.train5.ch_type = 'rayleigh'
p.train5.noise_type = 'fixed_snr'
p.train6 = Params()
p.train6.ch_type = 'fixed'
p.train6.ch_file = 'golden_channel.npy'
#p.train6.ch_file = 'alt_channel.npy'
p.train6.noise_type = 'rand_var'
p.train7 = Params()
p.train7.ch_type = 'covar'
p.train7.noise_type = 'rand_var'
### select training mode
p.train = p.train2
# test modes
p.test0 = Params()
p.test0.ch_type = 'rayleigh'
p.test0.noise_type = 'fixed_var'
p.test1 = Params()
p.test1.ch_type = 'fixed'
p.test1.ch_file = 'golden_channel.npy'
#p.test1.ch_file = 'alt_channel.npy'
p.test1.noise_type = 'fixed_var'
p.test2 = Params()
p.test2.ch_type = 'covar'
p.test2.noise_type = 'fixed_var'
### select test mode
p.test = p.test0

### deep learning parameters
#########################################
p.bname = 'densenet_symbol_est_32x4_qpsk'
#p.sname = '3_layers_rand_snr_rand_h_l2_reg'
p.sname = '1_layer_rand_snr_rand_h_adam'
p.outdir = 'models'
p.verbose = True
#p.verbose = False
p.graph_mode = True
#p.graph_mode = False
p.deterministic = False
p.seed = 959

### architecture specific
#########################################
# real dimensions
p.y_dim = p.N_rx * 2             # receive symbol
p.c_dim = (p.N_tx * p.N_rx) * 2  # channel matrix
p.x_dim = p.y_dim + p.c_dim      # deep net input
p.b_dim = p.N_sts * p.nbps       # deep net output
#p.s_dim = p.M ** p.N_sts         # alphabet size
p.s_dim = p.N_tx * 2             # num users in real dimensions
p.n_layers = 1                   # hidden layers
#p.n_layers = 4                   # hidden layers
#p.h_dim = 256
p.h_dim = 1024                   # nodes per hidden layer
#p.kernel_init = tf.keras.initializers.VarianceScaling()
p.kernel_init = 'VarianceScaling'
#p.activation_fn = tf.keras.activations.relu
p.activation_fn = 'relu'
#p.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
#p.kernel_regularizer = l2(1e-4)
p.kernel_regularizer = None
#p.activation_fn = 'leaky_relu'
p.batch_normalization = True
p.output_fn = None # return logits
# tf.nn.sigmoid_cross_entropy_with_logits

### densenet specific
#########################################
dn = Params()
#dn.x_0_dim = 256
dn.x_0_dim = 512
#dn.x_0_dim = 1024
#dn.x_0_dim = 2048
#dn.n_sublayers = 12
#dn.n_sublayers = [6, 12, 24]
#dn.n_sublayers = [6, 12, 32]
#dn.n_sublayers = [32, 12, 12]
dn.n_sublayers = 24
dn.growth_rate = 32
#dn.growth_rate = 48
dn.compression_ratio = 0.5
#dn.compression_ratio = 1
# post-processing
import collections
if not isinstance(dn.n_sublayers, collections.Sequence):
    dn.n_sublayers = [dn.n_sublayers] * p.n_layers
# sanity check
#assert dn.x_0_dim >= 2 * (p.y_dim + p.c_dim + 1)
assert len(dn.n_sublayers) == p.n_layers
# save to params
p.densenet = dn

### training specific
#########################################
p.n_train_steps = 5000
#p.n_train_steps = 1000
#p.n_train_steps = 100
#p.n_train_steps = 10
p.n_test_steps = 1000
#p.n_test_steps = 100
#p.n_test_steps = 10
p.n_epochs = 40
#p.n_epochs = 20
#p.n_epochs = 10
#p.n_epochs = 5
#p.n_epochs = 1
p.batch_size = 1024
#p.batch_size = 256
#p.batch_size = 10
#p.batch_dims = 1
# cross entropy
p.loss0 = Params()
p.loss0.type = 'CE'
p.loss1 = Params()
p.loss1.type = 'JCE'
p.loss1.lambda_ = 0.9
p.loss2 = Params()
p.loss2.type = 'ML'
# select overall loss
p.loss = p.loss0
# complement entropy
p.loss0_c = Params()
p.loss0_c.type = 'COT'
p.loss0_c.beta = 1/(p.s_dim -1)
p.loss0_c.alpha = 0
p.loss0_c.guided_comp_entropy = False
p.loss0_c.from_logits = True
# guided complement entropy
p.loss1_c = Params()
p.loss1_c.type = 'GCE'
p.loss1_c.beta = 1/np.log(p.s_dim-1)
p.loss1_c.alpha = 1/3
p.loss1_c.guided_comp_entropy = True
p.loss1_c.from_logits = True
# disable comp entropy
p.loss2_c = Params()
p.loss2_c.type = 'CE'
# select complementary loss
#p.loss_c = p.loss0_c
# Adam optimizer
# from original paper
p.adam0 = Params()
p.adam0.type = 'adam,aggressive'
p.adam0.learning_rate = 1e-3
p.adam0.beta_1 = 0.9
p.adam0.beta_2 = 0.999
# conservative setting
p.adam1 = Params()
p.adam1.type = 'adam,convervative'
p.adam1.learning_rate = 1e-4
p.adam1.beta_1 = 0.5
p.adam1.beta_2 = 0.9
# AMSGrad
p.adam2 = Params()
p.adam2.type = 'adam,amsgrad'
p.adam2.learning_rate = 1e-3
p.adam2.beta_1 = 0.9
p.adam2.beta_2 = 0.999
p.adam2.amsgrad = True
p.adam = p.adam2
# Momentum optimizer
p.mtm = Params()
p.mtm.type = 'sgd'
p.mtm.learning_rate = None
p.mtm.momentum = 0.9
p.mtm.nesterov = True
### select optimizer
p.optimizer = p.adam
#p.optimizer = p.mtm
# LR scehdules
p.lr_sched0 = Params()
p.lr_sched0.type = 'periodic,sgd'
p.lr_sched0.initial_learning_rate = 0.1
p.lr_sched0.minimum_learning_rate = 1e-4
p.lr_sched0.decay_rate_dbp = 10
p.lr_sched0.decay_steps = 3*p.n_train_steps
p.lr_sched1 = Params()
p.lr_sched1.type = 'aperiodic,adam'
p.lr_sched1.initial_learning_rate = 1e-3
p.lr_sched1.minimum_learning_rate = 1e-5
p.lr_sched1.steps_per_epoch = p.n_train_steps
p.lr_sched1.decay_rate_dbp = 10
p.lr_sched1.decay_schedule = [10, 15]
p.lr_sched2 = Params()
p.lr_sched2.type = 'aperiodic,sgd'
#p.lr_sched2.initial_learning_rate = 0.1
p.lr_sched2.initial_learning_rate = 0.01
p.lr_sched2.minimum_learning_rate = 1e-5
p.lr_sched2.steps_per_epoch = p.n_train_steps
p.lr_sched2.decay_rate_dbp = 10
p.lr_sched2.decay_schedule = [20, 30]
p.lr_sched3 = Params()
p.lr_sched3.type = 'aperiodic,adam'
p.lr_sched3.initial_learning_rate = 1e-3
p.lr_sched3.minimum_learning_rate = 1e-6
p.lr_sched3.steps_per_epoch = p.n_train_steps
p.lr_sched3.decay_rate_dbp = 10
p.lr_sched3.decay_schedule = [10, 25]
### select lr schedule
p.lr_sched = p.lr_sched3
#p.lr_sched = p.lr_sched2
# MT schedules
p.mt_sched0 = Params()
p.mt_sched0.type = 'original'
p.mt_sched0.initial_momentum = 0.5
p.mt_sched0.maximum_momentum = 0.9
p.mt_sched0.decay_steps = 250
### FLM specific

print()
print("Sim configuration")
print("=============================")
print(" Communications ")
print("-----------------------------")
print("one_bit         =", p.one_bit)
print("Training:")
print("channel type    =", p.train.ch_type)
print("noise type      =", p.train.noise_type)
print("Test:")
print("channel type    =", p.test.ch_type)
print("noise type      =", p.test.noise_type)
print("-----------------------------")
print(" Network ")
print("-----------------------------")
print("network type    =", p.bname)
print("num layers      =", p.n_layers)
print("num units       =", p.h_dim)
print("-----------------------------")
print(" Training ")
print("-----------------------------")
print("scheme          =", p.sname)
print("verbose         =", p.verbose)
print("deterministic   =", p.deterministic)
print("graph mode      =", p.graph_mode)
print("loss            =", p.loss.type)
#print("complement loss =", p.loss_c.type)
print("optimizer       =", p.optimizer.type)
print("lr schedule     =", p.lr_sched.type if p.lr_sched else None)
print("batch size      =", p.batch_size)
print("num epochs      =", p.n_epochs)
print("num train steps =", p.n_train_steps)
print("num test steps  =", p.n_test_steps)
print("-----------------------------")

################################################################################
# simulation setup
################################################################################
if p.deterministic:
    np.random.seed(p.seed)
    tf.set_random_seed(p.seed)

################################################################################
# define DNN model
################################################################################
class NeuralRx(Model):
    '''
    Densenet implementation
    '''
    def __init__(self, p,
                 n_layers, y_dim, c_dim, h_dim, s_dim,
                 activation_fn,
                 kernel_init,
                 batch_norm):
        super().__init__()

        self.n_layers = n_layers
        self.y_dim = y_dim
        self.c_dim = c_dim
        self.h_dim = h_dim
        self.s_dim = s_dim

        x_0_dim = p.densenet.x_0_dim
        n_sublayers = p.densenet.n_sublayers
        growth_rate = p.densenet.growth_rate
        compression_ratio = p.densenet.compression_ratio

        #activation_fn = p.activation_fn
        #kernel_init = p.kernel_init
        #batch_norm = p.batch_normalization
        kernel_reg = p.kernel_regularizer

        dense_hidden_layer = partial(HyperDenseV2, activation=None,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg,
                                     batch_normalization=False)
        dense_block_layer = partial(DenseBlock, activation=activation_fn,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg,
                                     batch_normalization=batch_norm)
        transition_layer = partial(HyperDenseV2, activation=activation_fn,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg,
                                     batch_normalization=batch_norm)
        dense_output_layer = partial(HyperDenseV2, activation=None,
                                     kernel_initializer=kernel_init,
                                     kernel_regularizer=kernel_reg,
                                     batch_normalization=False)

        self.rx_input = Input(shape=(y_dim,), name='rx_input')
        self.ch_input = Input(shape=(c_dim,), name='ch_input')
        self.nvar_input = Input(shape=(1,), name='nvar_input')
        #self.snr_input = Input(shape=(1,), name='snr_input', dtype=tf.int32)

        # construct hidden layers
        layers = []
        # one dense layer
        layers.append(dense_hidden_layer(x_0_dim))
        n_output = x_0_dim
        for l in range(n_layers):
            # add dense block
            layers.append(dense_block_layer(n_sublayers[l], growth_rate))
            n_output += n_sublayers[l] * growth_rate
            # add transition layer
            if l < n_layers - 1:
                # compute compressed n_output
                n_output = np.floor(compression_ratio * n_output)
                layers.append(transition_layer(n_output))
        #self.h_layers = Sequential(layers, name='hidden_layers')
        # add each layer explicitly (layers will show up in the diagram)
        self.h_layers = layers
        print('Hidden layers:')
        print(layers)

        self.s_out = dense_output_layer(s_dim, name='sym_pred') # sym predictor logits

    def build(self):
        '''
        build model using the functional API
        '''
        y = self.rx_input
        h = self.ch_input
        n_var = self.nvar_input
        #s = self.snr_input

        x = concatenate([y, h, n_var])
        for layer in self.h_layers:
            x = layer(x)
        o = self.s_out(x)

        return Model(inputs=[y, h, n_var],
                     outputs=o,
                     name='nrx_model')

################################################################################
# instantiate objects/callables
################################################################################
train_dataset = CommDataSet(p, symbol_output=True,
                               test=False,
                               mode='train',
                               one_bit=p.one_bit,
                               symbol_only=True)
test_dataset = CommDataSet(p, symbol_output=True,
                              test=True,
                              mode='test',
                              one_bit=p.one_bit,
                              symbol_only=True)
nrx = NeuralRx(p,
               p.n_layers, p.y_dim, p.c_dim, p.h_dim, p.s_dim,
               p.activation_fn,
               p.kernel_init,
               p.batch_normalization)

# create tf model instance
model = nrx.build()

plot_model(p, model, show=p.verbose)


# define loss functions
#loss_function = BinaryCrossentropy(from_logits=True)
#bin_cross_ent = BinaryCrossentropy(from_logits=True)
#loss_function = SparseCategoricalCrossentropy(from_logits=True)
loss_function = MeanSquaredError()


# schedule type lookup
schedule_types = {
        'periodic' : PeriodicLRDecaySchedule,
        'aperiodic': AperiodicLRDecaySchedule,
}

# optimizer type lookup
optimizer_types = {
        'adam' : Adam,
        'sgd'  : SGD,
}

# setup schedule(s)
mt_schedule = None
lr_schedule = None
if p.lr_sched is not None:
    # sanity check
    assert has_key(p.lr_sched.type, get_key(p.optimizer.type)), \
        "schedule and optimizer do not match"
    Schedule = schedule_types[get_key(p.lr_sched.type)]
    kwargs = p.lr_sched.as_dict(exclude=('type'))
    lr_schedule = Schedule(**kwargs)

# setup optimizer
Optimizer = optimizer_types[get_key(p.optimizer.type)]
kwargs = p.optimizer.as_dict(exclude=('type'))
if lr_schedule:
    # work around: make param serializable
    p.optimizer.learning_rate = 'lr_schedule'
    kwargs['learning_rate'] = lr_schedule
optimizer = Optimizer(**kwargs)

# add to model instance
# NOTE: the optimizer's state (slot variables, one per parameter)
#       are saved if both the optimizer and the parameter are saved.
#model.loss = loss_callable
#model.optimizer = optimizer

# define metric to evaluate loss and accuracy
train_loss = Mean(name='train_loss')
test_loss = Mean(name='test_loss')
train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')
true_accuracy = SparseCategoricalAccuracy(name='true_accuracy')
#train_ber = BitErrorRate(name='train_ber')
#test_ber = BitErrorRate(name='test_ber')
#true_ber = BitErrorRate(name='true_ber')

metrics = [train_loss, test_loss,
           train_accuracy, test_accuracy, true_accuracy,
#           train_ber, test_ber, true_ber,
           ]

################################################################################
# define training and test graphs, evaluate metrics at each step
################################################################################
#@tf.function
def train_step(train_input, train_output):
    with tf.GradientTape() as tape:
        sym_output = model(train_input, training=True)
        train_error = loss_function(train_output, sym_output)
    gradients = tape.gradient(train_error, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #posterior = tf.nn.softmax(logits)
    #hard_decisions = logits > 0

    train_loss(train_error)
    #train_accuracy(train_output, posterior)
    #train_ber(train_output, hard_decisions)

#@tf.function
#def test_step(test_input, test_output, lambda_mat):
def test_step(test_input, test_output):
    sym_output = model(test_input, training=False)
    test_error = loss_function(test_output, sym_output)
    #posterior = tf.nn.softmax(logits)
    #hard_decisions = logits > 0
    #ideal_decisions = lambda_mat < 0

    test_loss(test_error)
    #test_accuracy(test_output, posterior)
    #true_accuracy(test_output, ideal_decisions)
    #test_ber(test_output, hard_decisions)
    #true_ber(test_output, ideal_decisions)

################################################################################
# define computation graph
################################################################################
train_graph = tf.function(train_step)
test_graph = tf.function(test_step)

if p.graph_mode:
    train_step = train_graph
    test_step = test_graph

################################################################################
# begin training
################################################################################
def var_mi(cond_entropy_nats):
    '''computes variation mutual information (in bits)
    from conditional entropy (in nats)'''
    # NOTE: input is - E_{p(x,y)} [ q(x|y) ] per bit
    #       have to multiply by nbits
    nbits = p.nbps * p.N_sts
    src_entropy = nbits
    cond_entropy_bits = nbits * np.log2(np.exp(1)) * cond_entropy_nats
    return src_entropy - cond_entropy_bits

print("Start training...")

for epoch in range(p.n_epochs):

    for metric in metrics:
        metric.reset_states()

    with Timer():
        #for i, (inputs, outputs) in enumerate(train_dataset):
        for inputs, outputs, aux_out in train_dataset:
            syms_mat = outputs[2]
            train_step(inputs, syms_mat)
            # maintain step context ourselves
            if mt_schedule: mt_schedule.update()

        for inputs, outputs, aux_out in test_dataset:
            syms_mat, lambda_mat = outputs[2], aux_out
            #test_step(inputs, syms_vec, lambda_mat)
            test_step(inputs, syms_mat)

    template = '''
Epoch {:2d}, Loss:     Train {:.3f}, Test : {:.3f}
      {}  MI LB:    Train {:.3f}, Test : {:.3f}
      {}  Accuracy: Train {:.3f}, Test : {:.3f}, True : {:.3f}
      {}  SER:      Train {:.3f}, Test : {:.3f}, True : {:.3f}
'''


    print(template.format(epoch+1,
                          train_loss.result(),
                          test_loss.result(),
                          ' '*2,
                          var_mi(train_loss.result()),
                          var_mi(test_loss.result()),
                          ' '*2,
                          train_accuracy.result(),
                          test_accuracy.result(),
                          true_accuracy.result(),
                          ' '*2,
                          1 - train_accuracy.result(),
                          1 - test_accuracy.result(),
                          1 - true_accuracy.result(),
                          )
          )

################################################################################
# training end
################################################################################

################################################################################
# save to file
################################################################################
save_model(p, model)

