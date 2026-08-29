#ifndef RISCV_MATH_H
#define RISCV_MATH_H

#include <stdint.h>

#define sqrtf __builtin_sqrtf

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RISCV_MATH_SUCCESS = 0,
    RISCV_MATH_ARGUMENT_ERROR = -1,
    RISCV_MATH_LENGTH_ERROR = -2,
    RISCV_MATH_SIZE_MISMATCH = -3,
    RISCV_MATH_NANINF = -4,
    RISCV_MATH_SINGULAR = -5,
    RISCV_MATH_TEST_FAILURE = -6,
    RISCV_MATH_DECOMPOSITION_FAILURE = -7
} riscv_status;

typedef struct {
    uint16_t numRows;
    uint16_t numCols;
    float *pData;
} riscv_matrix_instance_f32;

void riscv_mat_init_f32(riscv_matrix_instance_f32 *S, uint16_t nRows, uint16_t nCols, float *pData);
riscv_status riscv_mat_mult_f32(const riscv_matrix_instance_f32 *pSrcA, const riscv_matrix_instance_f32 *pSrcB, riscv_matrix_instance_f32 *pDst);
void riscv_mat_vec_mult_f32(const riscv_matrix_instance_f32 *pSrcMat, const float *pVec, float *pDst);
void riscv_dot_prod_f32(const float *pSrcA, const float *pSrcB, uint32_t blockSize, float *result);
void riscv_cmplx_dot_prod_f32(const float *pSrcA, const float *pSrcB, uint32_t numSamples, float *realResult, float *imagResult);
void riscv_scale_f32(const float *pSrc, float scale, float *pDst, uint32_t blockSize);
riscv_status riscv_mat_cholesky_f32(const riscv_matrix_instance_f32 *pSrc, riscv_matrix_instance_f32 *pDst);
riscv_status riscv_mat_solve_upper_triangular_f32(const riscv_matrix_instance_f32 *pSrcA, const riscv_matrix_instance_f32 *pSrcB, riscv_matrix_instance_f32 *pDst);

#ifdef __cplusplus
}
#endif

#endif /* RISCV_MATH_H */
