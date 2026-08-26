#include "arm_math.h"

void arm_mat_init_f32(arm_matrix_instance_f32 *S, uint16_t nRows, uint16_t nCols, float *pData) {
    S->numRows = nRows;
    S->numCols = nCols;
    S->pData = pData;
}
