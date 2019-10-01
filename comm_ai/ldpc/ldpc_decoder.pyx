'''
c-wrapper module for layered_ldpc_dec.c
TODO: create wrapper class
NOTE: this c-code is hardcoded to implement a specific
      LDPC code.
'''

# c bool type
from libcpp cimport bool

################################################################################
# define signature to c-function
# NOTE: this can be defined in a separate
#       header file, typically decoder.pxd
#       which can be imported with
#         cimport decoder
# NOTE: specify #define'd constants with
#   enum: N
################################################################################
# see layered_ldpc_dec.c
#   flag : 1 = early termination
cdef extern from "layered_ldpc_dec.h":
    enum: N
    ctypedef struct decoder_t:
        pass
    decoder_t *new_decoder();
    void free_decoder(decoder_t *dec);

    bool layered_decoder_run(decoder_t *dec,
                             const double LLRs[N],
                             const double pbeta,
                             int max_iter,
                             int flag,
                             unsigned char decoded_bits[N],
                             unsigned char* check,
                             int* num_iter
                            )

import numpy as np
cimport numpy as np
cimport cython

################################################################################
# define class wrapper
################################################################################
cdef class LdpcDecoder:

    cdef decoder_t *dec
    # python accessible
    cdef public double pbeta
    cdef int max_iter
    cdef int flag

    # default values are pretty good
    def __cinit__(self, double pbeta=0.15,
                        int max_iter=24,
                        int flag=1):
        self.pbeta = pbeta
        self.max_iter = max_iter
        self.flag = flag

        self.dec = new_decoder()
        if self.dec is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.dec is not NULL:
            free_decoder(self.dec)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    # NOTE: requires llrs be c-contiguous and non-null
    def decode(self, const double[::1] llrs not None):

        assert llrs.shape[0] == N

        # ubyte (in numpy) == unsigned char (in C)
        cdef np.ndarray decoded_bits = np.zeros([N], dtype=np.ubyte)
        # pass a c-compatible view of the np array to c-function
        cdef unsigned char [:] dbits_view = decoded_bits
        cdef unsigned char check
        cdef int num_iter

        # extract c-pointer from memview
        # TODO: this portion can release the GIL,
        #       we are calling an external C-function; hence
        #       no interaction with the python interpreter
        #       e.g. surround the code with <with nogil>
        #with nogil:
        #    <call external c-function with GIL released>
        layered_decoder_run(self.dec, &llrs[0],
                            self.pbeta, self.max_iter, self.flag,
                            &dbits_view[0], &check, &num_iter)

        return decoded_bits, num_iter

################################################################################
# define wrapper function (test)
################################################################################
'''
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
# NOTE: requires llrs be c-contiguous and non-null
# TODO: when can I release the GIL?
def ldpc_decode(const double[::1] llrs not None,
                const double pbeta,
                int max_iter,
                int flag):

    assert llrs.shape[0] == N

    # ubyte (in numpy) == unsigned char (in C)
    cdef np.ndarray decoded_bits = np.zeros([N], dtype=np.ubyte)
    # pass a c-compatible view of the np array to c-function
    cdef unsigned char [:] dbits_view = decoded_bits
    cdef unsigned char check
    cdef int num_iter
    cdef decoder_t *dec = new_decoder()

    # extract c-pointer from memview
    layered_decoder_run(dec, &llrs[0], pbeta, max_iter, flag,
                        &dbits_view[0], &check, &num_iter)

    free_decoder(dec)

    return decoded_bits, check, num_iter
'''

