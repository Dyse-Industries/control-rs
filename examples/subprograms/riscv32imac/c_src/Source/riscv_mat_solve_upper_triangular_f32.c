#include "riscv_math.h"

riscv_status riscv_mat_solve_upper_triangular_f32(const riscv_matrix_instance_f32 *pSrcA, const riscv_matrix_instance_f32 *pSrcB, riscv_matrix_instance_f32 *pDst) {
    if (pSrcA->numRows != pSrcA->numCols || pSrcA->numRows != pSrcB->numRows || pSrcB->numCols != pDst->numCols || pSrcA->numRows != pDst->numRows) {
        return RISCV_MATH_SIZE_MISMATCH;
    }

    uint16_t n = pSrcA->numRows;
    uint16_t m = pSrcB->numCols;

    const float *pU = pSrcA->pData;
    const float *pB = pSrcB->pData;
    float *pX = pDst->pData;

    for (uint16_t c = 0; c < m; c++) {
        for (int32_t i = (int32_t)n - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (uint16_t k = (uint16_t)(i + 1); k < n; k++) {
                sum += pU[i * n + k] * pX[k * m + c];
            }

            float diag = pU[i * n + i];
            if (diag == 0.0f) {
                return RISCV_MATH_SINGULAR;
            }

            pX[i * m + c] = (pB[i * m + c] - sum) / diag;
        }
    }

    return RISCV_MATH_SUCCESS;
}
