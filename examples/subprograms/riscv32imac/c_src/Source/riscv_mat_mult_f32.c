#include "riscv_math.h"

riscv_status riscv_mat_mult_f32(const riscv_matrix_instance_f32 *pSrcA, const riscv_matrix_instance_f32 *pSrcB, riscv_matrix_instance_f32 *pDst) {
    if (pSrcA->numCols != pSrcB->numRows || pSrcA->numRows != pDst->numRows || pSrcB->numCols != pDst->numCols) {
        return RISCV_MATH_SIZE_MISMATCH;
    }

    uint16_t numRowsA = pSrcA->numRows;
    uint16_t numColsB = pSrcB->numCols;
    uint16_t numColsA = pSrcA->numCols;

    const float *pInA = pSrcA->pData;
    const float *pInB = pSrcB->pData;
    float *pOut = pDst->pData;

    for (uint16_t r = 0; r < numRowsA; r++) {
        for (uint16_t c = 0; c < numColsB; c++) {
            float sum = 0.0f;
            for (uint16_t k = 0; k < numColsA; k++) {
                sum += pInA[r * numColsA + k] * pInB[k * numColsB + c];
            }
            pOut[r * numColsB + c] = sum;
        }
    }

    return RISCV_MATH_SUCCESS;
}
