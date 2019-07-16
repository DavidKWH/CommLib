// implement layered LDPC decoder for block irregular LDPC
// Reference: A reduced complexity decoder architecture via
//            layered decoding of LDPC codes (Hocevar, 2004)
// Taken from: workspace/matlab/layered_ldpc
// TODO: define object like structure
//       previous implementation use static memory (not thread safe)
//
// define
//   m    parity check (row) index
//   j    bit (column) index
//   k    parity check group (block) index
//   N(m) set of all bit indices connected to check node m
//   M(j) set of all check node indices connected to bit j
//   Z    block size
//   K    data length
//   P    num parity check nodes
//   N    code length
//
#include "layered_ldpc_dec.h"

#include "stdlib.h"
#define MALLOC(n)  malloc(n)
#define FREE(p)    free(p)
#define PERSIST(p)
#define ATEXIT(x)  atexit(x)

//#define DBG_EN
#ifdef  DBG_EN
  #ifdef MATLAB_MEX_FILE
#include "mex.h"
#define DBG_PRINTF(fmt, ...) mexPrintf(fmt, ##__VA_ARGS__)
  #else
#include "stdio.h"
#define DBG_PRINTF(fmt, ...) fprintf(fmt, ##__VA_ARGS__)
  #endif
#else
#define DBG_PRINTF(fmt, ...)
#endif

//# define USE_CNODE_FCN 1               // specify check node algorithm

#if USE_CNODE_FCN == 1
  #define USE_OFFSET_MIN
  #define LLR_SUM llr_offset_min
#elif USE_CNODE_FCN == 2
  #define USE_LLR_SUM_FAST
  #define LLR_SUM llr_sum_fast
#elif uSe_CNODE_FCN == 3
  #define USE_LLR_SUM_RECURSE
  #error "FIXME"
#else
  #error "Unkown check node function"
#endif

#define USE_EARLY_TERMINATION          // early termination (requires computing syndrome for each iteration)
#define FCN_OPT_UNROLL_LOOPS
//#define MAX_ITER   30

#if 0 // place holder for header file
// code properties
#define NUM_LAYERS 1
#define Z          1
#define K          1
#define P          1
#define N          1
// internal structures
#define MAX_N        2
#define MAX_NM       2
uint16_t layer_seq[NUM_LAYERS];
uint32_t n_set_table[P][MAX_N];
uint32_t n_set_len[P];
uint32_t n_set_ex_table[P][N][MAX_NM];
uint32_t n_set_ex_len[P][N];
#endif

#define LAYER(k)          layer_seq[k]
#define N_SET(m)          n_set_table[m]
#define N_SET_LEN(m)      n_set_len[m]
#define N_SET_EX(m,j)     n_set_ex_table[m][j]
#define N_SET_EX_IDX(m,j) n_set_ex_index_table[m][j]
#define N_SET_EX_LEN(m)   (n_set_len[m]-1)

// define memory structures
//////////////////////////////
typedef struct decoder {
    // input/output bit LLRs
    double L_io[N];
    // hard bit decisions
    // user provided array
    unsigned char* L_hd;
    // check nodes
    double R[P][MAX_N];
    // syndrome
    bool S[P];
    // offset
    double beta;
} decoder_t;

decoder_t *new_decoder() {
    return (decoder_t*) MALLOC(sizeof(decoder_t));
}

void free_decoder(decoder_t *dec) {
    FREE(dec);
}

// simplify pointer access
#define L_IO  dec->L_io
#define L_HD  dec->L_hd
#define BETA  dec->beta
#define R_MSG dec->R
#define SYN   dec->S

//////////////////////////////
// helper functions
//////////////////////////////
#include "float.h"
#include "math.h"
int8_t sgn(double val) {
    return (0.0 < val) - (val < 0.0);
}

double psi_fn_gl(double x) {

    const double eps = 9.357622968840175e-14;
    double op = fabs( tanh( x/2.0 ) );
    bool sw = op < eps;
    return sw ? -30.0 : log( op );
}

double psi_fn(double x) {

    double op = fabs( tanh( x/2.0 ) );
    return log( op );
}

double amin_fn(double ax, double y) {

    double ay = fabs( y );
    return fmin( ax,ay );
}

double offmax_fn(decoder_t *dec, double x) {

    return fmax( x - BETA , 0.0 );
}

// offset-min function for the min-sum algorithm
double llr_offset_min(decoder_t *dec, uint32_t m, uint32_t j) {

    double R_mj;
    double A_mj = DBL_MAX;
    int8_t s_mj = 1;

    // iterate over the check node set excluding bit j
    for(uint32_t i = 0; i<N_SET_EX_LEN(m); i++) {
        //uint32_t n = N_SET_EX(m,j)[i];
        uint32_t k = N_SET_EX_IDX(m,j)[i];
        uint32_t n = N_SET(m)[k];
        double L_mn;
        L_mn = L_IO[n] - R_MSG[m][k];
        A_mj = amin_fn(A_mj, L_mn);
        s_mj *= sgn(L_mn);
    }
    R_mj = s_mj * offmax_fn(dec, A_mj);

    DBG_PRINTF("(m,j)=(%d,%d), A=%g s=%d R=%g\n",m,j,A_mj,s_mj,R_mj);

    return R_mj;
}

// NOTE: The psi() function is related to the
//       involution approach (see my tex doc).
//       It expresses the involution function as
//       in terms of the tanh function which has
//       known fast implementations.
//       This approach replaces the product of
//       tanh's in the original "tanh rule" such that
//       add-subtract can be use to come up with the
//       extrinsic llr's for each neighbor bit node.
// TODO: This is unstable when x is close to 0 in
//       log(x), which happens often when we are not
//       certain about a bit.  The tanh rule appears
//       to have an issue when u is close to 1 in
//       tanh^inv(u).  Need to verify.
double llr_sum_fast(decoder_t *dec, uint32_t m, uint32_t j) {

    double R_mj;
    double A_mj = 0.0;
    int8_t s_mj = 1;
    // iterate over the check node set excluding bit j
    for(uint32_t i = 0; i<N_SET_EX_LEN(m); i++) {
        //uint32_t n = N_SET_EX(m,j)[i];
        uint32_t k = N_SET_EX_IDX(m,j)[i];
        uint32_t n = N_SET(m)[k];
        double L_mn;
        L_mn = L_IO[n] - R_MSG[m][k];
        A_mj += psi_fn(L_mn);
        s_mj *= sgn(L_mn);
    }
    R_mj = - s_mj * psi_fn_gl(A_mj);

    DBG_PRINTF("(m,j)=(%d,%d), A=%g s=%d R=%g\n",m,j,A_mj,s_mj,R_mj);

    return R_mj;
}

#if 0
double llr_sum_recurse(uint32_t m, uint32_t j, uint32_t n) {

    if (n > 1) {
        double lhs = llr_sum_recurse(m,j,n-1);
        double rhs = L[m][N_SET_EX(m,j)[n-1]];
        return log( (1+exp(lhs + rhs)) / (exp(lhs) + exp(rhs)) );
    } else {
        return L[m][N_SET_EX(m,j)[0]];
    }
}

double llr_sum_stable(uint32_t m, uint32_t j) {

    return llr_sum_recurse(m,j,LEN_N_SET_EX(m,j));
}
#endif

#if 0
FCN_OPT_UNROLL_LOOPS
void hard_decision() {
    for (uint32_t j=0; j<N; j++) {
        L_hd[j] = L_io[j] < 0.0;
    }
}
#endif

FCN_OPT_UNROLL_LOOPS
bool parity_check(decoder_t *dec) {
    uint32_t syn = 0;
    for (uint32_t m=0; m<P; m++) {
        syn += SYN[m];
    }
    return syn == 0;
}

//////////////////////////////
// algorithm starts
//////////////////////////////
bool layered_decoder_run(decoder_t *dec,
                         const double LLRs[N],
                         const double pbeta,
                         int max_iter,
                         int flag,
                         unsigned char decoded_bits[N],
                         unsigned char* check,
                         int* num_iter
                        )
{
    bool pcheck = false;
    // assign pointers
    L_HD = decoded_bits;
    // offset parameter
    BETA = pbeta;

    // bit change detection
    bool L_bit_changed = false;

    // initialize LLRs
    for (uint32_t j=0; j<N; j++) {
        L_IO[j] = LLRs[j];
        L_HD[j] = (L_IO[j] < 0.0);
    }
    // reset R memory
    for (uint32_t m=0; m<P; m++) {
        for (uint32_t j=0; j<MAX_N; j++) {
            R_MSG[m][j] = 0.0;
        }
    }

    // full iteration entire code
    for (uint32_t i=0; i<max_iter; i++) {
        DBG_PRINTF("iter %d\n",i);
        L_bit_changed = false;
        // sub iteration through each layer
        for (uint32_t k=0; k<NUM_LAYERS; k++) {
            // compute block(k)
            for (uint32_t z=0; z<Z; z++) {
                uint32_t l = LAYER(k);
                uint32_t m = z + l*Z;
                bool S_m = 0;
                for(uint32_t i = 0; i<N_SET_LEN(m); i++) {
                    uint32_t j = N_SET(m)[i];
                    double L_mj;
                    bool   L_hd_j;
                    L_mj = L_IO[j] - R_MSG[m][i];
                    R_MSG[m][i] = LLR_SUM(dec, m,i);
                    L_IO[j] = L_mj + R_MSG[m][i];
                    L_hd_j = (L_IO[j] < 0.0);
                    L_bit_changed |= (L_hd_j != L_HD[j]);
                    L_HD[j] = L_hd_j;
                    S_m ^= L_HD[j];
                }
                SYN[m] = S_m;
                DBG_PRINTF("S[%d] = %d\n",m,S_m);
            }
        }

        for (uint32_t j=0; j<N; j++) {
            DBG_PRINTF("L_hd[%d] = %d\n",j,L_hd[j]);
        }

        // hard decision
        //hard_decision();

        // perform parity check
        pcheck = parity_check(dec) && (!L_bit_changed);

        // early termination
        if ((flag & 0x1) && pcheck) {
            *num_iter = i+1;
            *check = pcheck;
            return true;
        }

    }

    *num_iter = max_iter;
    *check = pcheck;
    return pcheck;
}

