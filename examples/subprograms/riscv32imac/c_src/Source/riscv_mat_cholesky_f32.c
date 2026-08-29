#include "riscv_math.h"

riscv_status riscv_mat_cholesky_f32(const riscv_matrix_instance_f32 *pSrc, riscv_matrix_instance_f32 *pDst) {
    if (pSrc->numRows != pSrc->numCols || pDst->numRows != pDst->numCols || pSrc->numRows != pDst->numRows) {
        return RISCV_MATH_SIZE_MISMATCH;
    }

    uint16_t n = pSrc->numRows;
    const float *pA = pSrc->pData;
    float *pL = pDst->pData;

    // If out-of-place, copy pA to pL first
    if (pA != pL) {
        for (uint32_t k = 0; k < (uint32_t)n * n; k++) {
            pL[k] = pA[k];
        }
    }

    // In-place lower triangular Cholesky over pL (modifying only lower triangle i >= j)
    for (uint16_t i = 0; i < n; i++) {
        for (uint16_t j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (uint16_t k = 0; k < j; k++) {
                sum += pL[i * n + k] * pL[j * n + k];
            }

            if (i == j) {
                float val = pL[i * n + i] - sum;
                if (val <= 0.0f) {
                    return RISCV_MATH_DECOMPOSITION_FAILURE;
                }
                pL[i * n + i] = sqrtf(val);
            } else {
                float diag = pL[j * n + j];
                if (diag == 0.0f) {
                    return RISCV_MATH_DECOMPOSITION_FAILURE;
                }
                pL[i * n + j] = (pL[i * n + j] - sum) / diag;
            }
        }
    }

    return RISCV_MATH_SUCCESS;
}
