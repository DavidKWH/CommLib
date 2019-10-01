#ifndef LAYERED_LDPC_DEC_H
#define LAYERED_LDPC_DEC_H

#include "stdbool.h"
#include "stdint.h"

#if CODE_RATE == 1
    #warning "R1/2 code selected for compilation"
    #include "PM_11ad_R1_2.h"
#elif CODE_RATE == 2
    #warning "R5/8 code selected for compilation"
    #include "PM_11ad_R5_8.h"
#elif CODE_RATE == 3
    #warning "R3/4 code selected for compilation"
    #include "PM_11ad_R3_4.h"
#elif CODE_RATE == 4
    #warning "R13/16 code selected for compilation"
    #include "PM_11ad_R13_16.h"
#else
    #error "Unsupported code rate"
#endif

typedef struct decoder decoder_t;
decoder_t *new_decoder();
void free_decoder(decoder_t *dec);

// bool_T = unsigned char in matlab
bool layered_decoder_run(decoder_t *dec,
                         const double LLRs[N],
                         const double pbeta,
                         int max_iter,
                         int flag,
                         unsigned char decoded_bits[N],
                         unsigned char* check,
                         int* num_iter
                        );

#endif /* LAYERED_LDPC_DEC_H */
