#ifndef ARM_MATH_H
#define ARM_MATH_H

#include <stdint.h>

#define sqrtf __builtin_sqrtf

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ARM_MATH_SUCCESS = 0,
    ARM_MATH_ARGUMENT_ERROR = -1,
    ARM_MATH_LENGTH_ERROR = -2,
    ARM_MATH_SIZE_MISMATCH = -3,
    ARM_MATH_NANINF = -4,
    ARM_MATH_SINGULAR = -5,
    ARM_MATH_TEST_FAILURE = -6,
    ARM_MATH_DECOMPOSITION_FAILURE = -7
} arm_status;

typedef struct {
    uint16_t numRows;
    uint16_t numCols;
    float *pData;
} arm_matrix_instance_f32;

void arm_mat_init_f32(arm_matrix_instance_f32 *S, uint16_t nRows, uint16_t nCols, float *pData);
arm_status arm_mat_mult_f32(const arm_matrix_instance_f32 *pSrcA, const arm_matrix_instance_f32 *pSrcB, arm_matrix_instance_f32 *pDst);
void arm_mat_vec_mult_f32(const arm_matrix_instance_f32 *pSrcMat, const float *pVec, float *pDst);
void arm_dot_prod_f32(const float *pSrcA, const float *pSrcB, uint32_t blockSize, float *result);
void arm_cmplx_dot_prod_f32(const float *pSrcA, const float *pSrcB, uint32_t numSamples, float *realResult, float *imagResult);
void arm_scale_f32(const float *pSrc, float scale, float *pDst, uint32_t blockSize);
arm_status arm_mat_cholesky_f32(const arm_matrix_instance_f32 *pSrc, arm_matrix_instance_f32 *pDst);
arm_status arm_mat_solve_upper_triangular_f32(const arm_matrix_instance_f32 *pSrcA, const arm_matrix_instance_f32 *pSrcB, arm_matrix_instance_f32 *pDst);

#ifdef __cplusplus
}
#endif

#endif /* ARM_MATH_H */
