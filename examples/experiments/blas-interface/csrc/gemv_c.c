// Reference C GEMV, same shape as variant A's DynDense::gemv_dyn (runtime
// rows/cols/lda/order, plain buffer indexing) -- the storage-side struct is
// unchanged, only the subprogram crosses into C. Compiled by build.rs into a
// staticlib and declared `extern "C"` in src/lib.rs (feature "cffi").
#include <stddef.h>

// order: 1 = row-major, 2 = col-major (matches MatrixLayout's repr in
// src/lib.rs).
void gemv_c_dyn(const float *buf, size_t rows, size_t cols, size_t lda,
                 int order, const float *x, float *y) {
    for (size_t i = 0; i < rows; i++) {
        float acc = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            size_t off = (order == 2) ? j * lda + i : i * lda + j;
            acc += buf[off] * x[j];
        }
        y[i] = acc;
    }
}
