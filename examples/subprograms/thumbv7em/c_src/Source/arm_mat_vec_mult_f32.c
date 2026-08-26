#include "arm_math.h"

void arm_mat_vec_mult_f32(const arm_matrix_instance_f32 *pSrcMat, const float *pVec, float *pDst) {
    uint16_t numRows = pSrcMat->numRows;
    uint16_t numCols = pSrcMat->numCols;
    const float *pMat = pSrcMat->pData;

    for (uint16_t r = 0; r < numRows; r++) {
        float sum = 0.0f;
        for (uint16_t c = 0; c < numCols; c++) {
            sum += pMat[r * numCols + c] * pVec[c];
        }
        pDst[r] = sum;
    }
}
