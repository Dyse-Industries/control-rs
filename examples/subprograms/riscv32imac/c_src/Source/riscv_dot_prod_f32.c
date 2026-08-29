#include "riscv_math.h"

void riscv_dot_prod_f32(const float *pSrcA, const float *pSrcB, uint32_t blockSize, float *result) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < blockSize; i++) {
        sum += pSrcA[i] * pSrcB[i];
    }
    *result = sum;
}
