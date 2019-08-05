
import numpy as np
#np.set_printoptions(threshold=np.inf)
#np.set_printoptions(threshold=False)

class LdpcDecoder:
    '''
    Wrapper class for c-implementation
    '''
    def __init__(self, PM, pbeta=0.15, max_iter=24, early_term=1):
        Z = 42
        K = PM.shape[0]
        nrows_blk = K//Z
        self.impl = self.create_instance(nrows_blk, pbeta, max_iter, early_term)

    def create_instance(self, nrows, pbeta, max_iter, flag):
        if nrows == 8:
            print('r1/2 decoder')
            from .ldpc_decoder_r1_2 import LdpcDecoder as Impl
        elif nrows == 6:
            print('r5/8 decoder')
            from .ldpc_decoder_r5_8 import LdpcDecoder as Impl
        elif nrows == 4:
            print('r3/4 decoder')
            from .ldpc_decoder_r3_4 import LdpcDecoder as Impl
        elif nrows == 3:
            print('r13/16 decoder')
            from .ldpc_decoder_r13_16 import LdpcDecoder as Impl
        else:
            assert('unsupported code rate')
        return Impl(pbeta, max_iter, flag)

    def decode(self, llrs):
        return self.impl.decode(llrs)

class LdpcEncoder:
    ''' 
    encode payload bits given parity matrix PM, PM_spec
    exploit structure in the block cyclic shifted code
    '''

    def __init__(self, PM, PM_spec=None):
        '''
        NOTE: sparse matrix operations (dot) is preferred for 
              matrices with 1% sparsity or less
        '''
        from scipy.sparse import csr_matrix
        from numpy.linalg import inv
        (P, N) = PM.shape
        K = N - P

        self.P = P
        self.N = N
        self.K = K
        self.PM = csr_matrix(PM)
        self.Px = csr_matrix(PM[:,:K])
        Pc_inv = inv(PM[:,K:])
        Pc_inv = np.abs(Pc_inv)
        Pc_inv = Pc_inv.astype(int)
        self.Pc_inv = csr_matrix(Pc_inv)

        # Enables optimized encoder function opt_encode()
        if PM_spec:
            T = len(PM_spec)
            Z = P//T
            self.T = T
            self.Z = Z
            self.St = [spec_seq[-T:] for spec_seq in PM_spec]

    def syndrome(self, cw):
        PM = self.PM
        return np.mod(PM.dot(cw), 2)

    def encode(self, x):
        ''' 
        Generic encode function
        Given Px x + Pc c = 0
        Computes c = Pc^inv Px x
        '''
        Px = self.Px
        Pc_inv = self.Pc_inv

        t = np.mod(Px.dot(x), 2)
        c = np.mod(Pc_inv.dot(t), 2)
        cw = np.concatenate((x,c), axis=None)

        return cw

    def opt_encode(self, x):
        '''
        Use back substitution at block matrix level
         - the PxP parity submatrix is (block) lower triangular
        Perform cyclic rotation for forward and inverse operations
        Math:
        properties of modulo 2 operations
         * a + b = 0 implies a = b
         * a - b <=> a + b
        Algorithm:
        Let PM = [P_1 P_2] then
            P_1 x + P_2 c = 0
        Define 
            t = P_x x
            t = [ t_1 t_2 ... t_T ]
            c = [ c_1 c_2 ... c_T ]
            and for the i-th row Pi of 
            the block matrix P_2
            Pi = [ Pi_1 Pi_2 ... ]
            where T = P/Z, t_i, c_i are Zx1 vectors
            c_i = Pi^inv ( t_i + Pi_1 c_1 + ... + Pi_{i-1} c_{i-1} )
            for i = 1,2,...,T
        NOTE: Pi_j are ZxZ rotation matrices
        '''
        assert(self.St)

        Px = self.Px
        Z = self.Z
        T = self.T

        # compute t = P_x * x
        t = np.mod(Px.dot(x), 2)
        t_mat = t.reshape(-1,Z)
        t_vecs = list(t_mat)

        St = self.St
        c_vecs = []
        # for each row ti in spec matrix St
        for ti in range(T):
            rots = St[ti]
            #sum_part = np.zeros(Z)
            sum_part = t_vecs[ti]
            for si in range(ti):
                rot = rots[si]
                # update only when rots[si] is non-null
                if (rot != None): sum_part += np.roll(c_vecs[si], -rot)
            c_vec = np.roll(sum_part, rots[ti])
            c_vecs.append(c_vec)

        c = np.concatenate(c_vecs, axis=None)
        c = np.mod(c, 2)
        cw = np.concatenate((x,c), axis=None)

        return cw

'''
# debug version
        # for each row i in spec matrix St
        for ti in range(T):
            rots = St[ti]
            #sum_part = np.zeros(Z)
            sum_part = t_vecs[ti]
            for si in range(ti):
                rot = rots[si]
                # update only when rots[si] is non-null
                #if (rot != None):
                #    print('{},{} adding ({}) rotated c_vec'.format(ti,si,rot) )
                if (rot != None): sum_part += np.roll(c_vecs[si], -rot)
            #print('({}) derotate sum_part'.format(rots[ti]) )
            c_vec = np.roll(sum_part, rots[ti])
            c_vecs.append(c_vec)
'''

def load_parity_matrix(fname, delimiter='-'):
    '''
    returns full parity matrix from text file spec
    NOTE: designed for 802.11ad parity matrices
    '''
    class params:
        pass

    '''
    NOTE: importing package resources (text files)
          requires using the importlib.resources tools
    NOTE: use backport for pre-python3.7 releases
    '''
    import importlib_resources as pkg_res
    from . import code_spec  # resource folder

    import re
    p = params
    #with open(fname,'r') as f:
    with pkg_res.open_text(code_spec, fname) as f:
        # read constants
        for _ in range(3):
            line = f.readline()
            # match pattern and capture "groups"
            # group(0) matches the entire pattern (needlessly)
            # use tuple output from groups()
            pattern = re.compile("([A-Z]) = (\d+)") # e.g. N = 123
            m = pattern.match(line)
            name = m.groups()[0]
            value = int(m.groups()[1])
            setattr(params, name, value)
            pvalue = getattr(params, name)
            print("{} = {}".format(name, pvalue))
        f.readline() # skip empty line

        Z = p.Z
        P = p.P
        N = p.N

        eye_Z = np.eye(Z,dtype=int)
        null_Z = np.zeros((Z,Z),dtype=int)
        PM = np.array([],dtype=int).reshape(0,N)

        PM_spec = []

        for line in f:
            seq = line.split()
            z_sel = [val == delimiter for val in seq]
            # create circular shifted matrices
            mat_seq = [null_Z if zero else np.roll(eye_Z, int(val), axis=1) 
                                       for (zero,val) in zip(z_sel,seq)]
            spec_seq = [None if zero else int(val)
                                       for (zero,val) in zip(z_sel,seq)]
            PM_row = np.hstack(mat_seq)
            PM = np.vstack((PM, PM_row))
            PM_spec.append(spec_seq)

    return PM, PM_spec






