#include "arm_math.h"

void arm_cmplx_dot_prod_f32(const float *pSrcA, const float *pSrcB, uint32_t numSamples, float *realResult, float *imagResult) {
    float sumReal = 0.0f;
    float sumImag = 0.0f;

    for (uint32_t i = 0; i < numSamples; i++) {
        float a_r = pSrcA[2 * i + 0];
        float a_i = pSrcA[2 * i + 1];
        float b_r = pSrcB[2 * i + 0];
        float b_i = pSrcB[2 * i + 1];

        // Conjugate dot product: conj(a) * b = (a_r - j*a_i) * (b_r + j*b_i)
        // = (a_r*b_r + a_i*b_i) + j*(a_r*b_i - a_i*b_r)
        sumReal += (a_r * b_r + a_i * b_i);
        sumImag += (a_r * b_i - a_i * b_r);
    }

    *realResult = sumReal;
    *imagResult = sumImag;
}
