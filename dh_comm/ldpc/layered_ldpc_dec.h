#ifndef LAYERED_LDPC_DEC_H
#define LAYERED_LDPC_DEC_H

#include "stdbool.h"
#include "stdint.h"

#include "PM_11ad_R1_2.h"

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
