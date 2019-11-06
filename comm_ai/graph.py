'''
graph related tools
'''
import numpy as np
import tensorflow as tf
from functools import partial
import os

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from .core import QAMModulator
from .core import Demodulator
from .core import Transmitter
from .core import Receiver
from .core import Channel
from .core import cplx2reals
from .core import bv2dec

################################################################################
# module initialization
################################################################################
proc_local = True

if not proc_local:
    cores = os.sched_getaffinity(0)
    num_cores = len(cores)
    print(f'num cores available: {num_cores}')
    print(f'using subprocesses for neighbor search')
    #SelectedExecutor = ThreadPoolExecutor
    SelectedExecutor = ProcessPoolExecutor

################################################################################
# support functions
################################################################################
def get_slices(nlists, idxs, sel=None):
    ''' slice each list in nlists and return
        only the selected sublists if sel is
        specified
    '''
    if sel: # apply filter
        sliced_lists = [lst[idxs] for i, lst in enumerate(nlists) if i in sel]
    else: # process all elements
        sliced_lists = [lst[idxs] for lst in nlists]
    return sliced_lists

def concat_sublist_elems(nlists):
    ''' concatenate the elements of the sublists
        assume sublists have the same shape
    '''
    n_iter = zip(*nlists)
    concatenated_lists = [np.concatenate(lst) for lst in n_iter]
    return concatenated_lists

################################################################################
# define classes
################################################################################
class RandomVectorSampler:
    '''
    Generate random samples
    '''
    def __init__(self, N, K):
        self.N = N
        self.K = K
        self.N_div = N//K * K
        self.arr = np.arange(N)
        self.new_generator()

    def new_generator(self):
        arr = self.arr
        N_div = self.N_div
        K = self.K
        # in-place shuff of array
        np.random.shuffle(arr)
        rnd_idx = arr[: N_div]
        rnd_idx = rnd_idx.reshape(-1, K)
        # save iterator
        self.iter = iter(rnd_idx)

    def get(self):
        '''return indices of K samples'''
        try:
            samples = next(self.iter)
        except StopIteration:
            # reshuffle array
            self.new_generator()
            samples = next(self.iter)

        return samples

#class NegativeSampler:
#    '''
#    Generate negative samples for NCE
#    '''
#    def __init__(self, source, K):
#        self.source = source
#        self.N = N = source.shape[0]
#        self.K = K
#        self.N_div = N//K * K
#        self.arr = np.arange(N)
#        self.new_generator()
#
#    def new_generator(self):
#        arr = self.arr
#        N_div = self.N_div
#        K = self.K
#        # in-place shuff of array
#        np.random.shuffle(arr)
#        rnd_idx = arr[: N_div]
#        rnd_idx = rnd_idx.reshape(-1, K)
#        # save iterator
#        self.iter = iter(rnd_idx)
#
#    def get(self):
#        '''return indices of K samples'''
#        try:
#            samples = next(self.iter)
#        except StopIteration:
#            # reshuffle array
#            self.new_generator()
#            samples = next(self.iter)
#
#        return samples

class NeighborFinder:
    '''
    Return the neighbors of a given node
    '''
    def __init__(self, p, per_ch_table):
        assert(per_ch_table)
        self.p = p
        self.per_ch_table = per_ch_table
        self.radius = self.init_radius(per_ch_table[0])

    def init_radius(self, y_tsr):
        p = self.p
        K = p.graph_l.n_neighbors
        Nt = p.graph_l.n_out

        y_tsr = np.squeeze(y_tsr)
        d = y_tsr.shape[-1]

        # limit search radius for neighbors
        # NOTE: the real bottleneck is sorting
        # NOTE: assume samples are distributed uniformly over a sphere
        #       for radius approximation - highly heruistic...
        vecnorm = partial(np.linalg.norm, ord=None, axis=1)
        y_norm_max = np.max(vecnorm(y_tsr))
        ce = 4./3.*np.pi # scale factor for volume of sphere
        rho_0 = y_norm_max / (Nt/ce)**(1./d) * K
        #print('rho_0', rho_0)

        return rho_0

    def find(self, y_vec):
        '''returns the full indices of K nearest neighbors'''
        # FIXME: add radius adaptation
        p = self.p
        K = p.graph_l.n_neighbors
        N_out = p.N_out

        # get per channel data
        rx_data = self.per_ch_table
        # tuples (y_tsr, sym_tsr, sym_vec)
        y_tsr = rx_data[0]

        # compute distance in y-space
        y_vec = np.squeeze(y_vec)
        y_tsr = np.squeeze(y_tsr)
        y_diff = y_tsr - y_vec[None,:]
        yd_norm = np.linalg.norm(y_diff, axis=1)

        ''' START: adjust search space '''
        # load search radius
        rho = self.radius
        #print('rho', rho)
        yd_ind = yd_norm < rho
        if np.sum(yd_ind) > 256:
            # shrink search space
            rho /= 2.
            yd_ind = yd_norm < rho
            #print(f'rho {rho*2:.4f} => {rho:.4f}')
        while np.sum(yd_ind) <= K:
            # enlarge serach space
            rho *= 2.
            assert not np.isinf(rho), "rho -> inf"
            yd_ind = yd_norm < rho
            #print(f'rho {rho/2:.4f} => {rho:.4f}')
        yd_idx = np.where(yd_ind)[0]
        ycr_norm = yd_norm[yd_idx]
        # sort list (index)
        ycr_idx = np.argsort(ycr_norm)
        yds_idx = yd_idx[ycr_idx]
        # update radius
        self.radius = rho
        ''' END: adjust search space '''

        # return indices for K-NN
        idx = np.arange(K)+1
        nb_idx = yds_idx[idx]

        # return full indices
        return nb_idx


class FullNeighborFinder:
    '''
    Return the neighbors of a given node
    '''
    def __init__(self, p, per_ch_tables):
        assert(per_ch_tables)
        self.p = p
        self.per_ch_tables = per_ch_tables

        # initialize radii
        self.radii = [self.init_radius(rx_data[0]) for rx_data in per_ch_tables]

    def init_radius(self, y_tsr):
        p = self.p
        K = p.graph_l.n_neighbors
        Nt = p.graph_l.n_out

        y_tsr = np.squeeze(y_tsr)
        d = y_tsr.shape[-1]

        # limit search radius for neighbors
        # NOTE: the real bottleneck is sorting
        # NOTE: assume samples are distributed uniformly over a sphere
        #       for radius approximation - highly heruistic...
        vecnorm = partial(np.linalg.norm, ord=None, axis=1)
        y_norm_max = np.max(vecnorm(y_tsr))
        ce = 4./3.*np.pi # scale factor for volume of sphere
        rho_0 = y_norm_max / (Nt/ce)**(1./d) * K
        #print('rho_0', rho_0)

        return rho_0

    def find(self, h_idx, y_vec):
        '''returns the full indices of K nearest neighbors'''
        # FIXME: add radius adaptation
        p = self.p
        K = p.graph_l.n_neighbors
        N_out = p.N_out

        # get per channel data
        rx_data = self.per_ch_tables[h_idx]
        # tuples (y_tsr, sym_tsr, sym_vec)
        y_tsr = rx_data[0]

        # compute distance in y-space
        y_vec = np.squeeze(y_vec)
        y_tsr = np.squeeze(y_tsr)
        y_diff = y_tsr - y_vec[None,:]
        yd_norm = np.linalg.norm(y_diff, axis=1)

        ''' START: adjust search space '''
        # load search radius
        rho = self.radii[h_idx]
        #print('rho', rho)
        yd_ind = yd_norm < rho
        if np.sum(yd_ind) > 256:
            # shrink search space
            rho /= 2.
            yd_ind = yd_norm < rho
            #print(f'rho[{h_idx}] {rho*2:.4f} => {rho:.4f}')
        while np.sum(yd_ind) <= K:
            # enlarge serach space
            rho *= 2.
            assert not np.isinf(rho), "rho -> inf"
            yd_ind = yd_norm < rho
            #print(f'rho[{h_idx}] {rho/2:.4f} => {rho:.4f}')
        yd_idx = np.where(yd_ind)[0]
        ycr_norm = yd_norm[yd_idx]
        # sort list (index)
        ycr_idx = np.argsort(ycr_norm)
        yds_idx = yd_idx[ycr_idx]
        # update radius
        self.radii[h_idx] = rho
        ''' END: adjust search space '''

        # return indices for K-NN
        idx = np.arange(K)+1
        nb_idx = yds_idx[idx]

        # return full indices
        return nb_idx + h_idx * N_out

################################################################################
# define classes
################################################################################
def gen_neighbor_table(p, rx_out):
    '''
    one unit of work for thread/process scheduling
    NOTE: run as standalone function to avoid
          pickling error when subprocessing
    '''
    #p = self.p
    neighbor_finder = NeighborFinder(p, rx_out)
    # rx_out = (y_tsr, sym_tsr, sym_vec, n_var_tsr)
    y_tsr =  rx_out[0]

    # lookup K nearest-neighbors
    nb_idx_list = [ neighbor_finder.find(y_vec) for y_vec in y_tsr ]
    nb_idx_mat = np.array(list(nb_idx_list))

    return nb_idx_mat

class CommGraph:
    '''
    Generate channel data for training

      Dimension of the graph: first we have the two
    high-level parameters: N_ch and N_out.

        N_ch = num of channel realizations, H
        N_out = num of channel uses for each H

    Total number of nodes:

        N_total = N_ch * N_out

    For each node we generate a K-nearest neighbor
    table of shape (N_total, K).

    All data are store in the following variables:

        self.features
        self.graph
        self.per_ch_tables
    '''
    def __init__(self, p, llr_output=False,
                          symbol_output=False,
                          maxlog_approx=False,
                          mode='default',
                          ):
        assert( hasattr(p,mode) )
        self.p = p
        self.maxlog_approx = maxlog_approx
        self.llr_output = llr_output
        self.sym_output = symbol_output
        self.mode = mode
        # comm. components
        self.mod = QAMModulator(p.M)
        self.demod = Demodulator(p, modulator=self.mod,
                                    maxlog_approx=maxlog_approx)
        self.channel = Channel(p, mode)
        self.transmit = Transmitter(p, modulator=self.mod, training=True)
        # graph components (create lazily)
        self.neighbor_finder = None

        # internal store
        self.features = None
        self.graph = None
        self.per_ch_tables = None

    def __repr__(self):
        return "Communication Graph"

    def gen_channel_output(self, N, h_mat):
        '''generate N output from channel h'''
        transmit = self.transmit
        channel = self.channel

        # generate payload bits and symbols
        sym_tsr, bit_tsr = transmit(N)
        # convert bit tensor to sym index
        bit_mat = bit_tsr.reshape(N,-1)
        sym_vec = bv2dec(bit_mat)
        # apply channel and noise
        y_tsr, n_var_tsr = channel.apply_channel(sym_tsr, h_mat)

        return y_tsr, sym_tsr, sym_vec, n_var_tsr

#    def _gen_neighbor_table(self, rx_out):
#        '''
#        one unit of work for thread/process scheduling
#        '''
#        p = self.p
#        neighbor_finder = NeighborFinder(p, rx_out)
#        # rx_out = (y_tsr, sym_tsr, sym_vec, n_var_tsr)
#        y_tsr =  rx_out[0]
#
#        # lookup K nearest-neighbors
#        nb_idx_list = [ neighbor_finder.find(y_vec) for y_vec in y_tsr ]
#        nb_idx_mat = np.array(list(nb_idx_list))
#
#        return nb_idx_mat

    def is_null(self):
        return not (self.features and self.graph and self.per_ch_tables)

    def save_graph(self, name='graph'):
        p = self.p
        print(f'saving graph to file: {name}_xx.npz')
        ''' save graph to file '''
        fname = '_'.join((name,'features.npz'))
        np.savez(fname, *self.features)
        fname = '_'.join((name,'graph.npz'))
        np.savez(fname, self.graph)
        # save pertinent parameters
        kwargs = p.graph_l.as_dict()
        fname = '_'.join((name,'params.npz'))
        np.savez(fname, **kwargs)

    def load_graph(self, name='graph'):
        print(f'loading graph to file: {name}_xx.npz')
        ''' load graph from file '''
        fname = '_'.join((name,'features.npz'))
        self.features = np.load(fname)
        fname = '_'.join((name,'graph.npz'))
        self.graph = np.load(fname)
        fname = '_'.join((name,'params.npz'))
        params = np.load(fname)
        # sanity check
        p = self.p
        assert params == p.graph_l.as_dict(), 'dicts not the same'
        # [TODO] rebuild per_ch_tables from data


    def gen_graph(self):
        '''
        try place everything in memory (not scalable)
        '''
        p = self.p
        ch = self.channel
        rx_gen = self.gen_channel_output
        #nb_gen = self._gen_neighbor_table
        nb_gen = partial( gen_neighbor_table, p )
        N_ch = p.N_ch
        N_out = p.N_out
        N_tx = p.N_tx
        N_rx = p.N_rx

        # generate channel realizations
        h_tsr = ch.gen_channels(N_ch)
        h_mat_list = list(h_tsr)

        # generate per channel dataset
        # returns list of tables, each an output from rx_gen()
        # each table is itself a list of tensors
        # (y_tsr, sym_tsr, sym_vec, n_var_tsr)
        rx_out_tables = [rx_gen(N_out, h_mat) for h_mat in h_mat_list]
        # save per channel tables
        self.per_ch_tables = rx_out_tables

        # find neighbors per rx_out_table
        '''
         Performance summary:
          * approx. 2x speed up using ProcessPool over 8 cores
          * presumably due to subprocess overhead and memory contention
        '''
        #if proc_local:
            #neighbor_tables = [nb_gen(rx_out) for rx_out in rx_out_tables]
        if not proc_local:
            with SelectedExecutor(max_workers=num_cores) as executor:
                proc_iter  = executor.map( nb_gen, rx_out_tables )
            neighbor_tables = list(proc_iter)

        #for nb1, nb2 in zip(neighbor_tables, nb_idx_list):
        #    np.testing.assert_array_equal(nb1, nb2)

        ################################################################################
        # merging all tables into tensors
        ################################################################################
        # concatenate neighbor tables
        nb_idx_mat = concat_sublist_elems(neighbor_tables)

        # contencate per channel data
        rx_out_list = concat_sublist_elems(rx_out_tables)
        y_out_tsr, sym_out_tsr, sym_out_vec, n_var_tsr = rx_out_list

        # full channel tensor
        h_out_tsr = np.repeat(h_tsr, N_out, axis=0)

        # full channel index
        h_idx = np.arange(N_ch)
        h_out_idx = np.repeat(h_idx, N_out, axis=0)

        # collect edge features
        features = (h_out_idx, h_out_tsr, y_out_tsr, sym_out_tsr, sym_out_vec, n_var_tsr)

        # save graph and features
        self.features = features
        self.graph = nb_idx_mat

        print('graph generated')

        # return contcatenated features
        #return h_out_idx, h_out_tsr, y_out_tsr, sym_out_tsr, sym_out_vec, n_var_tsr

    def _gen_source_data(self):
        '''
        try place everything in memory (not scalable)
        '''
        p = self.p
        ch = self.channel
        rx_gen = self.gen_channel_output
        N_ch = p.N_ch
        N_out = p.N_out
        N_tx = p.N_tx
        N_rx = p.N_rx

        # generate channel realizations
        h_tsr = ch.gen_channels(N_ch)
        h_mat_list = list(h_tsr)

        # generate per channel dataset
        # returns list of tuples (y_tsr, sym_tsr, sym_vec)
        rx_out_list = [rx_gen(N_out, h_mat) for h_mat in h_mat_list]
        # save per channel table
        self.per_ch_tables = rx_out_list

        # contencate per channel data
        rx_out_list = concat_sublist_elems(rx_out_list)
        y_out_tsr, sym_out_tsr, sym_out_vec, n_var_tsr = rx_out_list

        # full channel tensor
        h_out_tsr = np.repeat(h_tsr, N_out, axis=0)

        # full channel index
        h_idx = np.arange(N_ch)
        h_out_idx = np.repeat(h_idx, N_out, axis=0)

        # return contcatenated features
        return h_out_idx, h_out_tsr, y_out_tsr, sym_out_tsr, sym_out_vec, n_var_tsr

    def _find_neighbors(self, h_idx, y_vec):
        '''returns the full indices of K nearest neighbors'''
        # FIXME: add radius adaptation
        p = self.p
        K = p.graph_l.n_neighbors
        N_out = p.N_out

        # get per channel data
        rx_data = self.per_ch_tables[h_idx]
        # tuples (y_tsr, sym_tsr, sym_vec)
        y_tsr = rx_data[0]

        # compute distance in y-space
        y_vec = np.squeeze(y_vec)
        y_tsr = np.squeeze(y_tsr)
        y_diff = y_tsr - y_vec[None,:]
        yd_norm = np.linalg.norm(y_diff, axis=1)

        # sort list (index)
        #yd_sort = np.sort(yd_norm)
        yds_idx = np.argsort(yd_norm)

        # return indices for K-NN
        idx = np.arange(K)+1
        nb_idx = yds_idx[idx]

        # return full indices
        return nb_idx + h_idx * N_out

    def _gen_graph(self):
        '''
        generate a comm graph

        NOTE: At this point, we have a minimal representation of the graph
              where connections are defined in terms of indice pairs (edges).
        NOTE: populating the neighbor features and generating negative
              samples happen at training.  See CommGraphDataSet class
        '''
        p = self.p
        K = p.graph_l.n_neighbors

        # generate source data (features)
        source = self._gen_source_data()

        # create neighbor finder
        if self.neighbor_finder is None:
            assert(self.per_ch_tables)
            print('creating neighbor finder instance')
            self.neighbor_finder = FullNeighborFinder(p, self.per_ch_tables)
            neighbor_finder = self.neighbor_finder

        # generate K-NN data (edges)
        h_idx_vec, h_tsr, y_tsr, sym_tsr, sym_idx_vec, n_var_tsr = source

        # lookup K nearest-neighbors
        '''
         Performance summary:
          * 2.5x speed up with radius adaptation (method 2)
          * the other major factor is that the tensor does not fit in cache
          * slight improvement, using all CPU cores with thread pool (method 3)
            memory contention seems to be the issue...
         Method 1: sort all points with no search radius (sort all points)
         Method 2: sort with adaptive sort radius
         Method 3: use ThreadPoolExecutor instead of list comprehension
        '''
        #nb_idx_list = [ self._find_neighbors(h_idx, y_vec)
        #                  for h_idx, y_vec in zip(h_idx_vec, y_tsr) ]
        nb_idx_list = [ neighbor_finder.find(h_idx, y_vec)
                          for h_idx, y_vec in zip(h_idx_vec, y_tsr) ]
        #with SelectedExecutor(max_workers=num_cores) as executor:
        #    nb_idx_list = executor.map( neighbor_finder.find, h_idx_vec, y_tsr)
        nb_idx_mat = np.array(list(nb_idx_list))
        #np.testing.assert_array_equal(nb_idx_mat_1, nb_idx_mat_2)

        # save graph and features
        self.features = source
        self.graph = nb_idx_mat

    # these functions are used during training
    def get_neighbor_features(self, node_idx):
        '''return tuple of neighbor features'''

        # get neighbor indices
        nb_idxs = self.graph[node_idx]
        # get neighbor features
        nb_features = get_slices(self.features, nb_idxs, [2,3])
        # return type = [y_out_tsr, sym_out_tsr]
        return nb_features

    def get_negative_features(self, h_idx):
        '''return tuple of negative sample features'''
        p = self.p
        neg_sampler = self.negative_sampler
        N_ch = p.N_ch
        #K = p.graph_l.n_neighbors
        #L = p.graph_l.n_neg_samp

        neg_idxs = neg_sampler.get()
        # convert to full indices
        neg_idxs = neg_idxs + h_idx * N_ch
        # get features
        neg_features = get_slices(self.features, neg_idxs, [2,3])
        # return type = [y_out_tsr, sym_out_tsr]
        return neg_features

class CommGraphDataSet(CommGraph):
    '''
    Iterable Wrapper for Tensorflow model training

    Parameters defined in p.graph_l:
        n_nodes_pm - number of nodes per minibatch
        n_neighbors - defines K of the K-NN
        n_nb_samp - number of samples used from each K-NN neighborhood
        n_neg_samp - number of negative samples for each context pair

    1) For each node, get all neighbor features.
    2) For each context pair, gen negative samples
    '''
    def __init__(self, p,
                       test=False,
                       transform=None,
                       **kwargs):
        super().__init__(p, **kwargs)
        self.p = p
        self.n_iter = p.n_test_steps if test else p.n_train_steps
        self.in_transform = transform

        # always generate graph on creation for now
        #self._gen_graph()
        #self.gen_graph()
        self.minibatch_sampler = RandomVectorSampler(p.graph_l.n_total,
                                                     p.graph_l.n_nodes_pm)
        self.negative_sampler = RandomVectorSampler(p.graph_l.n_out,
                                                    p.graph_l.n_neg_samp *
                                                    p.graph_l.n_neighbors )


    def __repr__(self):
        return "Communication Graph Iterable"

    def __iter__(self):
        self.cnt = 0 # reset count
        return self

    def __next__(self):
        '''returns the next item in the set'''
        if self.cnt == self.n_iter:
            raise StopIteration()
        self.cnt += 1
        return self.get_training_data()

    def gen_batch_data(self):
        '''generate batch data'''
        assert not self.is_null(), 'no graph present'
        p = self.p
        mbatch_sampler = self.minibatch_sampler
        #K = p.graph_l.n_neighbors
        L = p.graph_l.n_neg_samp
        #M = p.graph_l.n_nodes_pm
        N = p.graph_l.batch_size

        # sample nodes for batch
        node_idxs = mbatch_sampler.get()
        node_features = get_slices(self.features, node_idxs)

        # --- grab neighbor features ---
        nb_features = [self.get_neighbor_features(node_idx) for node_idx in node_idxs ]
        # NOTE: neighbor shares features with associated node
        #       num of features is fewer due to sharing
        # nb_features = [y_out_tsr, sym_out_tsr]
        nb_features = concat_sublist_elems(nb_features)

        # --- gen negative samples per context (node/neighbor) pair
        h_idx_vec = node_features[0]
        neg_features = [ self.get_negative_features(h_idx) for h_idx in h_idx_vec ]
        # NOTE: negative sample shares features with associated node
        #       num of features is fewer due to sharing
        # neg_features = [y_out_tsr, sym_out_tsr]
        neg_features = concat_sublist_elems(neg_features)
        # reshape negative samples
        out_shapes = [feature.shape[1:] for feature in neg_features]
        neg_features = [ feature.reshape(L, N, *shape)
                        for feature, shape in zip(neg_features, out_shapes) ]

        # broadcast node features to neighbors
        K = p.graph_l.n_neighbors
        node_features = [ np.repeat(feature, K, axis=0) for feature in node_features ]

        # output
        return node_features, nb_features, neg_features

    def get_training_data(self):
        '''pre-process data for tensorflow'''
        p = self.p
        N = p.graph_l.batch_size
        L = p.graph_l.n_neg_samp

        # get feature populated data
        data = self.gen_batch_data()
        node_features, nb_features, neg_features = data

        h_idx, h_tsr, y_tsr, sym_tsr, sym_vec, n_var_tsr = node_features
        nb_y_tsr, nb_sym_tsr = nb_features
        neg_y_tsr, neg_sym_tsr = neg_features

        # flatten dimensions
        h_mat = h_tsr.reshape(N,-1)
        y_mat = y_tsr.reshape(N,-1)
        sym_mat = sym_tsr.reshape(N,-1)
        n_var_mat = n_var_tsr.reshape(N,-1)

        nb_y_mat = nb_y_tsr.reshape(N,-1)
        nb_sym_mat = nb_sym_tsr.reshape(N,-1)

        neg_y_mat = neg_y_tsr.reshape(L,N,-1)
        neg_sym_mat = neg_sym_tsr.reshape(L,N,-1)

        # convert to reals
        h_mat = cplx2reals(h_mat)
        y_mat = cplx2reals(y_mat)
        sym_mat = cplx2reals(sym_mat)

        nb_y_mat = cplx2reals(nb_y_mat)
        nb_sym_mat = cplx2reals(nb_sym_mat)

        neg_y_mat = cplx2reals(neg_y_mat)
        neg_sym_mat = cplx2reals(neg_sym_mat)

        # convert to tensors
        h_mat = tf.convert_to_tensor(h_mat, dtype=tf.float32)
        y_mat = tf.convert_to_tensor(y_mat, dtype=tf.float32)
        sym_mat = tf.convert_to_tensor(sym_mat, dtype=tf.float32)
        n_var_mat = tf.convert_to_tensor(n_var_mat, dtype=tf.float32)

        nb_y_mat = tf.convert_to_tensor(nb_y_mat, dtype=tf.float32)
        nb_sym_mat = tf.convert_to_tensor(nb_sym_mat, dtype=tf.float32)

        neg_y_mat = tf.convert_to_tensor(neg_y_mat, dtype=tf.float32)
        neg_sym_mat = tf.convert_to_tensor(neg_sym_mat, dtype=tf.float32)

        # output processing
        node_ft = [sym_mat, y_mat, h_mat, n_var_mat]
        nb_ft = [nb_sym_mat, nb_y_mat]
        neg_ft = [neg_sym_mat, neg_y_mat]

        #if in_transform: in_seq = in_transform(in_seq)
        #out_seq = [bit_mat, sym_vec]
        #aux_out = lambda_mat

        return node_ft, nb_ft, neg_ft

