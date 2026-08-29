#include "riscv_math.h"

void riscv_scale_f32(const float *pSrc, float scale, float *pDst, uint32_t blockSize) {
    for (uint32_t i = 0; i < blockSize; i++) {
        pDst[i] = pSrc[i] * scale;
    }
}
