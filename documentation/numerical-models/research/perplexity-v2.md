# Research Bibliography for Numerical Model Design

![Date Badge](https://img.shields.io/badge/Date-July_26,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Complete-brightgreen)
![Scope Badge](https://img.shields.io/badge/Scope-Control--rs%20Numerical%20Models-orange)

---

## Purpose

This bibliography supports the research framework established in the **Master List of Claims and Assumptions Across Numerical Model Design Docs**. Each entry is annotated to identify which specific claim, assumption, or verification benchmark it validates. References are organized by the five research tracks and the architectural domains they serve.

This document complements the prior *Verified References Bibliography for control-rs Numerical Models* (session 1e0c3661) by adding: (a) explicit BLAS Level 1/2/3 specification papers, (b) the Bunch-Kaufman $LDL^T$ primary source, (c) the Van Loan ZOH integral formula, (d) Wilkinson's two foundational texts, (e) additional control systems textbooks, (f) sparse matrix references, (g) safety standards, (h) property-based testing, and (i) a comprehensive claim-to-source coverage matrix mapping every design document claim to its supporting references.

---

## Table of Contents

1. [Foundational Numerical Linear Algebra](#1-foundational-numerical-linear-algebra)
2. [Polynomial Algorithms and Root Finding](#2-polynomial-algorithms-and-root-finding)
3. [Control Systems and Discretization](#3-control-systems-and-discretization)
4. [DSP, Fixed-Point, and Embedded Numerics](#4-dsp-fixed-point-and-embedded-numerics)
5. [Embedded Rust, Memory Layout, and FFI](#5-embedded-rust-memory-layout-and-ffi)
6. [Structured and Sparse Storage Backends](#6-structured-and-sparse-storage-backends)
7. [Tensor Operations and TinyML](#7-tensor-operations-and-tinyml)
8. [Benchmarking, HIL Validation, and Testing](#8-benchmarking-hil-validation-and-testing)
9. [Claim-to-Source Coverage Matrix](#9-claim-to-source-coverage-matrix)

---

## 1. Foundational Numerical Linear Algebra

**Supports:** Matrix model claims (column-major BLAS/LAPACK matching, $LDL^T$ superiority, Faddeev-LeVerrier characteristic polynomial recurrence, zero-copy views, FFI safety); State-Space BLAS-mapped state propagation; Research Track 1 (numerical stability and high-order solvers).

### Textbooks

**[1]** Golub, Gene H., and Charles F. Van Loan. *Matrix Computations*. 4th ed. Baltimore: Johns Hopkins University Press, 2013. ISBN 978-1-4214-0794-4.
- **DOI:** [10.1137/1.9781421407944](https://doi.org/10.1137/1.9781421407944)
- **Relevance:** The canonical reference for dense matrix factorizations (LU, QR, Cholesky, $LDL^T$), BLAS-level operations, and the companion matrix eigenvalue problem. Directly supports claims about column-major layout matching LAPACK conventions, $LDL^T$ decomposition properties, and FFI memory safety for C kernel delegation.

**[2]** Higham, Nicholas J. *Accuracy and Stability of Numerical Algorithms*. 2nd ed. Philadelphia: SIAM, 2002. ISBN 0-89871-521-0.
- **DOI:** [10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- **Relevance:** Definitive treatment of floating-point rounding error analysis, backward and forward stability, and condition number theory. Supports the IEEE 754 compliance assumption, Faddeev-LeVerrier characteristic polynomial recurrence analysis, and the matrix inversion ill-conditioning hazard for $N_x > 10$.

**[3]** Trefethen, Lloyd N., and David Bau III. *Numerical Linear Algebra*. Philadelphia: SIAM, 1997. ISBN 0-89871-361-7.
- **DOI:** [10.1137/1.9781611977165](https://doi.org/10.1137/1.9781611977165) (25th Anniversary Edition, 2022)
- **Relevance:** Foundational text for backward stability concepts, conditioning of matrix operations, and the QR algorithm. Validates the normwise backward stability claim for companion matrix root-finding and the LU partial pivoting default solver assumption.

**[4]** Demmel, James W. *Applied Numerical Linear Algebra*. Philadelphia: SIAM, 1997. ISBN 0-89871-389-7.
- **DOI:** [10.1137/1.9781611971446](https://doi.org/10.1137/1.9781611971446)
- **Relevance:** Covers LAPACK design philosophy, condition number estimation, and practical stability analysis. Supports the BLAS/LAPACK kernel delegation architecture and the TRSM substitution kernel approach for Tustin transforms.

**[5]** Horn, Roger A., and Charles R. Johnson. *Matrix Analysis*. 2nd ed. Cambridge: Cambridge University Press, 2013. ISBN 978-0-521-83940-2.
- **DOI:** [10.1017/CBO9781139020411](https://doi.org/10.1017/CBO9781139020411)
- **Relevance:** Comprehensive reference for matrix theory including similarity transformations, canonical forms, and eigenvalue theory. Supports the equivalence-under-canonical-transformations claim for state-space models and companion matrix direct conversion.

**[6]** Wilkinson, James H. *The Algebraic Eigenvalue Problem*. Oxford: Clarendon Press, 1965. ISBN 0-19-853403-5.
- **Relevance:** The foundational text on eigenvalue computation accuracy, backward error analysis, and the QR algorithm. Supports Faddeev-LeVerrier stability analysis, companion matrix eigenvalue computation, and the overall IEEE 754 compliance framework.

**[7]** Wilkinson, James H. *Rounding Errors in Algebraic Processes*. Englewood Cliffs, NJ: Prentice-Hall, 1963. Reprinted by Dover, 1994. ISBN 0-486-67999-3.
- **Relevance:** Origins of backward error analysis for polynomial evaluation and matrix operations. Supports the Horner's method optimal evaluation claim (rounding error minimization) and the fixed-point precision drift analysis framework.

### BLAS and LAPACK Specifications

**[8]** Lawson, C. L., R. J. Hanson, D. R. Kincaid, and F. T. Krogh. "Basic Linear Algebra Subprograms for Fortran Usage." *ACM Transactions on Mathematical Software* 5, no. 3 (September 1979): 308--323.
- **DOI:** [10.1145/355841.355847](https://doi.org/10.1145/355841.355847)
- **Relevance:** The original Level 1 BLAS specification (AXPY, dot products, vector norms). Directly supports the BLAS Level 1 kernel delegation claim for matrix and state-space operations.

**[9]** Dongarra, J. J., J. Du Croz, S. Hammarling, and R. J. Hanson. "An Extended Set of FORTRAN Basic Linear Algebra Subprograms." *ACM Transactions on Mathematical Software* 14, no. 1 (March 1988): 1--17.
- **DOI:** [10.1145/42288.42291](https://doi.org/10.1145/42288.42291)
- **Relevance:** Level 2 BLAS specification (GEMV, TRSV). Supports the BLAS Level 2 mapping claim for state-space propagation $x[k+1] = Ax[k] + Bu[k]$.

**[10]** Dongarra, J. J., J. Du Croz, I. S. Duff, and S. Hammarling. "A Set of Level 3 Basic Linear Algebra Subprograms." *ACM Transactions on Mathematical Software* 16, no. 1 (March 1990): 1--17.
- **DOI:** [10.1145/77626.79170](https://doi.org/10.1145/77626.79170)
- **Relevance:** Level 3 BLAS specification (GEMM, TRSM). Supports the BLAS Level 3 mapping claim for batch state propagation, Tustin transform triangular solvers, and Kalman filter covariance updates.

**[11]** Anderson, E., Z. Bai, C. Bischof, S. Blackford, J. Demmel, J. Dongarra, J. Du Croz, A. Greenbaum, S. Hammarling, A. McKenney, and D. Sorensen. *LAPACK Users' Guide*. 3rd ed. Philadelphia: SIAM, 1999. ISBN 0-89871-447-8.
- **DOI:** [10.1137/1.9780898719604](https://doi.org/10.1137/1.9780898719604)
- **URL:** [https://www.netlib.org/lapack/lug/](https://www.netlib.org/lapack/lug/)
- **Relevance:** Reference for LAPACK factorization routines (LU, QR, Cholesky, $LDL^T$) and their column-major storage conventions. Supports the column-major array layout claim for zero-copy FFI routing to CMSIS-DSP.

### Symmetric Indefinite and Matrix Exponential Methods

**[12]** Bunch, James R., and Linda Kaufman. "Some Stable Methods for Calculating Inertia and Solving Symmetric Linear Systems." *Mathematics of Computation* 31, no. 137 (January 1977): 163--179.
- **DOI:** [10.1090/S0025-5718-1977-0428694-0](https://doi.org/10.1090/S0025-5718-1977-0428694-0)
- **Relevance:** The Bunch-Kaufman pivoting strategy for $LDL^T$ factorization of symmetric indefinite matrices. Directly supports the $LDL^T$ superiority claim --- avoiding square roots while maintaining numerical stability for symmetric systems.

**[13]** Moler, Cleve B., and Charles F. Van Loan. "Nineteen Dubious Ways to Compute the Exponential of a Matrix." *SIAM Review* 20, no. 4 (October 1978): 801--836.
- **DOI:** [10.1137/1020098](https://doi.org/10.1137/1020098)
- **Relevance:** Seminal survey of matrix exponential computation methods. Supports the ZOH augmented matrix exponential discretization approach and the scaling-and-squaring method selection.

**[14]** Moler, Cleve B., and Charles F. Van Loan. "Nineteen Dubious Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later." *SIAM Review* 45, no. 1 (2003): 3--49.
- **DOI:** [10.1137/S00361445024180](https://doi.org/10.1137/S00361445024180)
- **Relevance:** Updated survey covering modern scaling-and-squaring with Pade approximation. Supports Research Track 1 on numerical stability of matrix exponentials in ZOH discretization for higher-order systems.

**[15]** Van Loan, Charles F. "Computing Integrals Involving the Matrix Exponential." *IEEE Transactions on Automatic Control* 23, no. 3 (June 1978): 395--404.
- **DOI:** [10.1109/TAC.1978.1101743](https://doi.org/10.1109/TAC.1978.1101743)
- **Relevance:** The Van Loan formula for computing matrix exponential integrals via augmented matrix exponentiation. Directly supports the ZOH discretization accuracy claim for the augmented matrix exponential $e^{MT_s}$ approach.

**[16]** Higham, Nicholas J. "The Scaling and Squaring Method for the Matrix Exponential Revisited." *SIAM Journal on Matrix Analysis and Applications* 26, no. 4 (2005): 1179--1193.
- **DOI:** [10.1137/04061101X](https://doi.org/10.1137/04061101X)
- **Relevance:** Theoretical foundation for the scaling-and-squaring method with Pade approximants underlying MATLAB's `expm` function. Should be cited alongside Moler & Van Loan (2003) for the ZOH matrix exponential computation.

**[17]** Al-Mohy, Awad H., and Nicholas J. Higham. "A New Scaling and Squaring Algorithm for the Matrix Exponential." *SIAM Journal on Matrix Analysis and Applications* 31, no. 3 (2009): 970--989.
- **DOI:** [10.1137/09074721X](https://doi.org/10.1137/09074721X)
- **Relevance:** Improved scaling-and-squaring algorithm alleviating the overscaling problem. Supports Research Track 1 on numerical stability of matrix exponentials for higher-order state-space ZOH discretization.

**[17a]** Higham, Nicholas J. "The Scaling and Squaring Method for the Matrix Exponential Revisited." *SIAM Review* 51, no. 4 (2009): 747--764.
- **DOI:** [10.1137/090768539](https://doi.org/10.1137/090768539)
- **Relevance:** SIAM Review/SIGEST survey version of the scaling-and-squaring method, providing an accessible overview of matrix exponential computation methods and their relative accuracy. Serves as a summary companion to the technical papers [16] and [17], supporting Research Track 1 on numerical stability of matrix exponentials for ZOH discretization.

**[18]** Frison, Gianluca, Dimitris Kouzoupis, Tommaso Sartor, Andrea Zanelli, and Moritz Diehl. "BLASFEO: Basic Linear Algebra Subroutines for Embedded Optimization." *ACM Transactions on Mathematical Software* 44, no. 4 (2018): 1--30.
- **DOI:** [10.1145/3210754](https://doi.org/10.1145/3210754)
- **Relevance:** Panel-major BLAS layout trade-offs for embedded systems optimized for small-to-medium matrices fitting in cache. Directly relevant to `control-rs` static-memory matrix design and the column-major vs. panel-major layout decision for embedded BLAS delegation.

---

## 2. Polynomial Algorithms and Root Finding

**Supports:** Polynomial model claims (Horner's method optimal evaluation, ascending power storage, compile-time capacity verification, companion matrix direct conversion); Research Track 2 (fixed-point precision drift in Horner evaluations).

### Textbooks

**[19]** Knuth, Donald E. *The Art of Computer Programming, Volume 2: Seminumerical Algorithms*. 3rd ed. Reading, MA: Addison-Wesley, 1997. ISBN 0-201-89684-2.
- **Relevance:** Authoritative treatment of polynomial arithmetic, Horner's method evaluation, and floating-point rounding analysis. Supports the Horner's method optimal evaluation claim ($N-1$ additions and multiplications) and the ascending power storage rationale for direct index-to-exponent mapping.

**[20]** Wilkinson, James H. *Rounding Errors in Algebraic Processes*. Englewood Cliffs, NJ: Prentice-Hall, 1963. (See entry [7] above.)
- **Relevance:** Also supports this section --- the definitive analysis of polynomial root-finding sensitivity and the conditioning of companion matrices. Validates that Horner evaluation minimizes rounding errors relative to naive power-sum evaluation.

**[21]** Faddeev, D. K., and V. N. Faddeeva. *Computational Methods of Linear Algebra*. San Francisco: W. H. Freeman, 1963.
- **Relevance:** The original source for the Faddeev-LeVerrier trace-based characteristic polynomial recurrence for matrix-to-polynomial conversion. Supports the Faddeev-LeVerrier algorithm. Caution: the standard Faddeev-LeVerrier recurrence involves division by $k$; the design doc's "division-free" characterization requires implementation-specific verification against the exact variant used. No verified primary source was found for a division-free Faddeev-LeVerrier variant.

**[22]** Henrici, Peter. *Applied and Computational Complex Analysis, Volume 1: Power Series --- Integration --- Conformal Mapping --- Location of Zeros*. New York: Wiley, 1974. ISBN 0-471-37244-6.
- **Relevance:** Comprehensive treatment of Horner's method error analysis, polynomial evaluation, and root-finding sensitivity. Supports the Horner's method optimal evaluation claim and the polynomial evaluation rounding error framework.

### Polynomial Root-Finding Algorithms

**[23]** Aurentz, Jared L., Thomas Mach, Raf Vandebril, and David S. Watkins. "Fast and Backward Stable Computation of Roots of Polynomials." *SIAM Journal on Matrix Analysis and Applications* 36, no. 3 (2015): 942--973.
- **DOI:** [10.1137/140983434](https://doi.org/10.1137/140983434)
- **Relevance:** The primary source for the companion matrix $O(N^2)$ root-finding claim. Proves normwise backward stability of the unitary-plus-rank-one rotator approach in upper Hessenberg form, directly validating the fast companion matrix root-finding claim in the Matrix model.

**[24]** Aurentz, Jared L., Thomas Mach, Leonardo Robol, Raf Vandebril, and David S. Watkins. "Fast and Backward Stable Computation of Roots of Polynomials, Part II: Backward Error Analysis; Companion Matrix and Companion Pencil." arXiv:1611.02435, 2016.
- **DOI:** [10.48550/arXiv.1611.02435](https://doi.org/10.48550/arXiv.1611.02435)
- **Relevance:** Extended backward error analysis for companion matrix and companion pencil formulations. Supports Research Track 1 on numerical stability of high-order polynomial root solvers and the companion matrix direct conversion claim.

**[25]** Aurentz, Jared L., Thomas Mach, Leonardo Robol, Raf Vandebril, and David S. Watkins. *Core-Chasing Algorithms for the Eigenvalue Problem*. Philadelphia: SIAM, 2018.
- **DOI:** [10.1137/1.9781611975345](https://doi.org/10.1137/1.9781611975345)
- **Relevance:** SIAM monograph consolidating the companion matrix QR algorithm theory. The definitive reference for the unitary-plus-rank-one exploitation technique. Cite alongside the 2015 journal paper for a complete treatment.

**[26]** Bini, Dario A., Paola Boito, Yuli Eidelman, Luca Gemignani, and Israel Gohberg. "A Fast Implicit QR Eigenvalue Algorithm for Companion Matrices." *Linear Algebra and its Applications* 432, no. 8 (2010): 2006--2031.
- **DOI:** [10.1016/j.laa.2009.08.003](https://doi.org/10.1016/j.laa.2009.08.003)
- **Relevance:** Alternative $O(N^2)$ time, $O(N)$ space implicit QR algorithm for companion matrices using generator-based representation. Provides an alternative algorithmic approach to the Aurentz et al. method.

**[27]** Van Barel, Marc, Raf Vandebril, Paul Van Dooren, and Katrijn Frederix. "Implicit Double Shift QR-Algorithm for Companion Matrices." *Numerische Mathematik* 116 (2010): 177--212.
- **DOI:** [10.1007/s00211-010-0302-y](https://doi.org/10.1007/s00211-010-0302-y)
- **Relevance:** Alternative implicit double-shift QR for companion matrices using Givens rotation representation. Supports the landscape of available companion matrix eigenvalue algorithms.

---

## 3. Control Systems and Discretization

**Supports:** State-Space model claims (BLAS-mapped propagation, ZOH/Tustin discretization accuracy, canonical form equivalence); Transfer Function model claims (direct frequency evaluation, canonical transformation); Research Track 1 (numerical stability of high-order system solvers).

### Core Textbooks

**[28]** Kailath, Thomas. *Linear Systems*. Englewood Cliffs, NJ: Prentice-Hall, 1980. ISBN 0-13-536961-4.
- **Relevance:** The definitive reference for linear system theory, including state-space descriptions, controllability/observability canonical forms (CCF/OCF), and polynomial matrix descriptions. Supports the equivalence-under-canonical-transformations claim and the companion matrix CCF conversion for polynomials and transfer functions.

**[29]** Chen, Chi-Tsong. *Linear System Theory and Design*. 4th ed. New York: Oxford University Press, 2013. ISBN 978-0-19-995957-0.
- **Relevance:** Standard textbook for state-space canonical forms, transfer function realization, and similarity transformations. Supports the CCF/OCF conversion claims and the strict properness assumption for state-space conversion ($N < D$, i.e., numerator degree strictly less than denominator degree).

**[30]** Franklin, Gene F., J. David Powell, and Michael L. Workman. *Digital Control of Dynamic Systems*. 3rd ed. Half Moon Bay, CA: Ellis-Kagle Press, 1998. ISBN 0-9791226-0-0.
- **Relevance:** Primary reference for ZOH discretization, bilinear (Tustin) transform, and digital controller design. Directly supports the ZOH exact step-invariant discretization claim and the Tustin frequency-prewarped discretization approach.

**[31]** Astrom, Karl J., and Bjorn Wittenmark. *Computer-Controlled Systems: Theory and Design*. 3rd ed. Mineola, NY: Dover Publications, 2011. ISBN 978-0-486-48613-0.
- **Relevance:** Comprehensive treatment of sampled-data systems, ZOH discretization, and digital controller implementation. Supports the discretization accuracy claims and the LTI system invariant assumption.

**[32]** Astrom, Karl J., and Richard M. Murray. *Feedback Systems: An Introduction for Scientists and Engineers*. 2nd ed. Princeton: Princeton University Press, 2021. ISBN 978-0-691-19398-4.
- **URL:** [https://press.princeton.edu/books/hardcover/9780691193984/feedback-systems](https://press.princeton.edu/books/hardcover/9780691193984/feedback-systems) (free online: [https://fbsbook.org](https://fbsbook.org))
- **Relevance:** Modern treatment of state-space tools, Lyapunov stability, reachability, observability, estimators, and frequency domain design. Supports the state-space model equivalence claims and the Kalman filter covariance update benchmark.

**[32a]** Ogata, Katsuhiko. *Modern Control Engineering*. 5th ed. Upper Saddle River, NJ: Pearson/Prentice Hall, 2010. ISBN 978-0-13-615673-4.
- **Relevance:** Standard undergraduate/graduate reference for state-space analysis, transfer functions, ZOH discretization, and control system design. Supports the canonical form conversion claims, ZOH discretization methodology, and the strict properness requirement for state-space conversion ($N < D$, i.e., numerator degree strictly less than denominator degree).

### Discrete-Time Signal Processing

**[33]** Oppenheim, Alan V., and Ronald W. Schafer. *Discrete-Time Signal Processing*. 3rd ed. Upper Saddle River, NJ: Prentice Hall, 2010. ISBN 0-13-198842-5.
- **Relevance:** The authoritative DSP reference for convolution, frequency response evaluation, and digital filter structures. Supports the direct DSP convolution claim for transfer function interconnections and the direct frequency evaluation $H(j\omega)$ claim.

### Primary Source: Kalman Filter

**[34]** Kalman, Rudolf E. "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering* 82, no. 1 (March 1960): 35--45.
- **DOI:** [10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
- **Relevance:** Original derivation of the discrete Kalman filter and covariance update equations. Essential primary source for the Kalman filtering verification benchmark ($P_{k|k} = (I-KH)P$) in the Matrix model.

---

## 4. DSP, Fixed-Point, and Embedded Numerics

**Supports:** Matrix model assumptions (embedded CPU cache locality, FPU registers); Polynomial/State-Space/Transfer Function fixed-point claims; Research Track 2 (fixed-point scaling and precision drift in Q31/Q15).

### Textbooks

**[35]** Parhi, Keshab K. *VLSI Digital Signal Processing Systems: Design and Implementation*. New York: Wiley-Interscience, 1999. ISBN 0-471-24186-5.
- **Relevance:** Comprehensive reference for DSP algorithm implementation, pipelining, and fixed-point arithmetic. Supports the Horner's method cycle count analysis, convolution kernel design, and fixed-point scaling boundary claims for Q31/Q15 arithmetic.

**[36]** Lyons, Richard G. *Understanding Digital Signal Processing*. 3rd ed. Upper Saddle River, NJ: Prentice Hall, 2011. ISBN 0-13-702741-9.
- **Relevance:** Practical DSP reference covering convolution, discrete Fourier transforms, and fixed-point implementation issues. Supports the DSP convolution subprogram (CONV) delegation claim for transfer function interconnections.

**[37]** Koren, Israel. *Computer Arithmetic Algorithms*. 2nd ed. Natick, MA: A K Peters, 2002. ISBN 1-56881-160-8.
- **Relevance:** Reference for fixed-point arithmetic (Q-format), guard bit allocation, and overflow/saturation handling. Supports Research Track 2 on automated guard-bit allocation and dynamic scaling for fixed-point MCU targets lacking FPUs.

**[38]** Hennessy, John L., and David A. Patterson. *Computer Architecture: A Quantitative Approach*. 6th ed. Oxford: Morgan Kaufmann/Elsevier, 2017. ISBN 978-0-12-811905-1.
- **Relevance:** Canonical reference for cache locality, memory hierarchy, SIMD instruction set architecture, and FPU design. Supports the column-major cache locality assumption and the memory hierarchy analysis for embedded BLAS operations. Authors are 2017 ACM A.M. Turing Award recipients.

### Standards

**[39]** IEEE. *IEEE Standard for Floating-Point Arithmetic*. IEEE Std 754-2019. New York: IEEE, July 22, 2019.
- **DOI:** [10.1109/IEEESTD.2019.8766229](https://doi.org/10.1109/IEEESTD.2019.8766229)
- **Relevance:** The governing standard for floating-point arithmetic compliance. Directly supports the IEEE 754 compliance assumption across all models and the prohibition of `-ffast-math` non-associative loop reordering unless numerical drift is explicitly accepted.

### ARM Embedded Documentation

**[40]** ARM. *CMSIS-DSP Software Library Documentation*.
- **URL:** [https://arm-software.github.io/CMSIS-DSP/latest/](https://arm-software.github.io/CMSIS-DSP/latest/)
- **Relevance:** Official documentation for the CMSIS-DSP library including matrix operations, FIR filter convolution, and BLAS-equivalent functions. Supports the BLAS/DSP kernel delegation claim. Caution: CMSIS-DSP matrix routines use row-major storage, while BLAS/LAPACK use column-major. An FFI layer routing column-major `control-rs` matrices to CMSIS-DSP requires a layout adapter or transposition step, not a zero-copy passthrough. Also provides the Q31/Q15 fixed-point arithmetic reference implementation.

**[41]** ARM. *Cortex-M4 Technical Reference Manual*. ARM DDI 0439.
- **URL:** [https://developer.arm.com/documentation/ddi0439/latest](https://developer.arm.com/documentation/ddi0439/latest)
- **Relevance:** Primary hardware reference for the Cortex-M4 FPU (32 single-precision registers) and DWT cycle counter. Note: the Cortex-M4 does not have a general-purpose L1 data cache; cache locality claims should target the Cortex-M7 (entry [42]) or be qualified as MPU/TCM-dependent. Supports the embedded hardware architecture assumption about FPU register count and the DWT cycle profiling methodology.

**[42]** ARM. *Cortex-M7 Technical Reference Manual*. ARM DDI 0480.
- **URL:** [https://developer.arm.com/documentation/ddi0480/latest](https://developer.arm.com/documentation/ddi0480/latest)
- **Relevance:** Primary hardware reference for the Cortex-M7 (double-precision FPU, L1 caches, tightly-coupled memory). Supports the cache locality assumption and the HIL cycle profiling framework.

**[43]** Yiu, Joseph. *The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors*. 3rd ed. Oxford: Newnes/Elsevier, 2013. ISBN 978-0-12-408082-9.
- **Relevance:** Comprehensive guide to Cortex-M architecture, exception handling, stack models, and FPU operation. Supports the 4KB stack safety bound assumption, the stack headroom assumption, and the DWT cycle counter methodology for HIL benchmarking.

---

## 5. Embedded Rust, Memory Layout, and FFI

**Supports:** Global architectural principles (peer storage container architecture, type-level dimension bounds, `#[repr(C)]` layout); FFI memory safety claims; Research Track 3 (structured storage backends).

### Rust Language and Ecosystem

**[44]** The Rust Reference. *The Rust Programming Language Reference*.
- **URL:** [https://doc.rust-lang.org/reference/](https://doc.rust-lang.org/reference/)
- **Relevance:** Official specification for `#[repr(C)]` layout guarantees, `const generics` limitations, and `unsafe` semantics. Supports the FFI memory safety claim (Peano dimension bounds coupled with `#[repr(C)]` prevent buffer overflow in C DSP/BLAS kernels) and the type-level dimension bounds assumption about stable Rust const generic arithmetic insufficiency.

**[45]** The Rustonomicon. *The Dark Arts of Unsafe Rust*.
- **URL:** [https://doc.rust-lang.org/nomicon/](https://doc.rust-lang.org/nomicon/)
- **Relevance:** Authoritative guide to `unsafe` Rust, FFI safety invariants, and memory layout guarantees. Supports the zero-copy views claim and the FFI memory safety guarantees for passing stack-allocated buffers to unchecked C kernels.

**[46]** The Embedded Rust Book.
- **URL:** [https://docs.rust-embedded.org/book/](https://docs.rust-embedded.org/book/)
- **Relevance:** Guide to `#![no_std]` development, cross-compilation for ARM Cortex-M targets, and memory layout for embedded Rust. Supports the `#![no_std]` runtime assumption, the peer storage container architecture claim about avoiding heap allocations, and the 4KB stack safety bound design rationale.

**[47]** Dimforge. *nalgebra: Linear Algebra Library for Rust*.
- **URL:** [https://github.com/dimforge/nalgebra](https://github.com/dimforge/nalgebra) (docs: [https://nalgebra.org](https://nalgebra.org/docs/user_guide/getting_started/))
- **Relevance:** Real-world Rust implementation of decoupled storage trait architecture (`ArrayStorage` for stack, `MatrixView`/`MatrixViewMut` for zero-copy views), type-level dimension tracking, and `#![no_std]` support. The `control-rs` design closely parallels nalgebra's architecture and serves as an implementation precedent.

### Safety Standards

**[48]** ISO. *ISO 26262-6:2018 Road Vehicles --- Functional Safety --- Part 6: Product Development at the Software Level*. Geneva: ISO, 2018.
- **URL:** [https://www.iso.org/standard/68388.html](https://www.iso.org/standard/68388.html)
- **Relevance:** Safety-critical embedded software development requirements, software unit design and implementation, software unit verification. Supports the safety and determinism design rationale for the `#![no_std]` stack-allocated architecture.

**[49]** RTCA/EUROCAE. *DO-178C: Software Considerations in Airborne Systems and Equipment Certification*. Washington, DC: RTCA, 2011.
- **URL:** [https://www.rtca.org/do-178/](https://www.rtca.org/do-178/)
- **Relevance:** Airborne software certification framework, software verification processes, configuration management for safety-critical systems. Supports the verification and validation framework for embedded numerical algorithms.

**[49a]** ISO/IEC/IEEE. *ISO/IEC/IEEE 29119-3:2021 Software and Systems Engineering --- Software Testing --- Part 3: Test Documentation*. Geneva: ISO, 2021.
- **URL:** [https://www.iso.org/standard/78785.html](https://www.iso.org/standard/78785.html)
- **Relevance:** International standard for test documentation, including test plans, test cases, and test reports. Supports the verification benchmark framework for documenting HIL test results, property-based testing outcomes, and numerical stability validation across all five models. Replaces the earlier IEEE 829-2008 standard.

---

## 6. Structured and Sparse Storage Backends

**Supports:** Research Track 3 (structured and sparse storage backends); State-Space structural sparsity potential assumption; Matrix structural specialization efficiency claim.

### Textbooks

**[50]** Davis, Timothy A. *Direct Methods for Sparse Linear Systems*. Philadelphia: SIAM, 2006. ISBN 0-89871-613-6.
- **DOI:** [10.1137/1.9780898718881](https://doi.org/10.1137/1.9780898718881)
- **Relevance:** Foundational reference for sparse matrix data structures (CSR, CSC, compressed formats), fill-reducing orderings, and sparse LU/Cholesky factorization. Directly supports Research Track 3 on implementing `SparseStorage` backends conforming to `Storage<T, R, C>`.

**[51]** Saad, Yousef. *Iterative Methods for Sparse Linear Systems*. 2nd ed. Philadelphia: SIAM, 2003. ISBN 0-89871-534-2.
- **DOI:** [10.1137/1.9780898718003](https://doi.org/10.1137/1.9780898718003)
- **Relevance:** Comprehensive treatment of Krylov subspace methods, preconditioning, and sparse matrix-vector products. Supports the `ZeroStorage` backend design for $D=0$ feedforward matrices and the structural sparsity exploitation in state-space models.

**[52]** Duff, Iain S., Albert M. Erisman, and John K. Reid. *Direct Methods for Sparse Matrices*. 2nd ed. Oxford: Oxford University Press, 2017. ISBN 978-0-19-850838-0.
- **DOI:** [10.1093/acprof:oso/9780198508380.001.0001](https://doi.org/10.1093/acprof:oso/9780198508380.001.0001)
- **Relevance:** Classic reference for sparse matrix theory, including graph-theoretic analysis of fill-in and data structure design. Supports the structural sparsity potential assumption for companion form, tridiagonal, and zero-feedforward system matrices.

---

## 7. Tensor Operations and TinyML

**Supports:** Tensor model claims (N-dimensional storage decoupling, column-major stride matching, compile-time coordinate safety, zero-copy sub-tensor views and in-place contractions); Research Track 5 (MIMO and Edge AI integration).

### Tensor Theory

**[53]** Kolda, Tamara G., and Brett W. Bader. "Tensor Decompositions and Applications." *SIAM Review* 51, no. 3 (September 2009): 455--500.
- **DOI:** [10.1137/07070111X](https://doi.org/10.1137/07070111X)
- **Relevance:** The definitive survey on tensor decompositions, multi-way array notation, and tensor contraction operations. Supports the Einstein summation (`contract_into`) claim, the column-major stride matching claim for multi-dimensional coordinate mapping, and the N-dimensional storage decoupling architecture.

### Edge AI and TinyML

**[54]** Lai, Liangzhen, Naveen Suda, and Vikas Chandra. "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs." arXiv:1801.06601, 2018.
- **DOI:** [10.48550/arXiv.1801.06601](https://doi.org/10.48550/arXiv.1801.06601)
- **Relevance:** Primary reference for neural network inference kernels optimized for Cortex-M processors. Supports the Edge AI and spatial grid capability claim for TinyML neural network inference using tensor weight/activation storage on microcontrollers.

**[55]** Warden, Pete, and Daniel Situnayake. *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. Sebastopol, CA: O'Reilly Media, 2019. ISBN 978-1-4920-5204-3.
- **Relevance:** Practical guide to deploying ML models on microcontrollers. Supports Research Track 5 on validating zero-copy interoperability between Tensor, Matrix, and TransferFunctionMatrix in edge control loops and the end-to-end benchmark pipeline design for neural network controller inference.

**[56]** David, Robert, Jared Duke, Adv Jain, Vijay Janapa Reddi, Natalie Jeffries, Jianming Li, Nick Kreeger, Ian Nappier, Meghna Natraj, Tina Wang, and Pete Warden. "TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems." *Proceedings of Machine Learning and Systems (MLSys)* 3 (2021): 800--811.
- **URL:** [https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf](https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf)
- **Relevance:** Peer-reviewed description of the TFLM inference framework for running deep-learning models on embedded systems with static memory constraints. The peer-reviewed companion to the Warden & Situnayake practitioner book, directly relevant to the `tensor-design.md` Edge AI / TinyML section.

---

## 8. Benchmarking, HIL Validation, and Testing

**Supports:** Research Track 4 (HIL cycle profiling); Matrix model verification benchmark (cycle counts vs CMSIS-DSP); State-Space HIL real-time cycle deadline checks; Tensor cache locality assumptions; Polynomial model property-based testing benchmark.

### Hardware Profiling

**[57]** ARM. *ARM Cortex-M Architecture Reference Manual (ARMv7-M)*. ARM DDI 0403.
- **URL:** [https://developer.arm.com/documentation/ddi0403/latest](https://developer.arm.com/documentation/ddi0403/latest)
- **Relevance:** Primary specification for the Data Watchpoint and Trace (DWT) unit, including the cycle counter (CYCCNT) register. Directly supports the HIL cycle profiling methodology claim about using hardware DWT cycle counters for cycle-accurate performance measurement. Note: DWT provides cycle counting and data watchpoints, but not cache-miss counters such as $I1mr$/$D1mr$ --- those are Cachegrind/Valgrind host-side simulation metrics, not hardware DWT outputs. The bibliography distinguishes DWT cycle counting (on-target HIL) from host-side cache simulation (Cachegrind/Valgrind).

**[58]** ARM. *CoreSight Architecture Specification*. ARM IHI 0029.
- **URL:** [https://developer.arm.com/documentation/ihi0029/latest](https://developer.arm.com/documentation/ihi0029/latest)
- **Relevance:** Specification for the debug and trace infrastructure used in HIL benchmarking setups. Supports the real-time cycle deadline verification methodology for state-space and matrix operations on embedded targets.

**[59]** Yiu, Joseph. *The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors*. 3rd ed. (See entry [43] above.)
- **Relevance:** Also supports this section --- detailed coverage of Cortex-M memory model, cache behavior, and DWT-based performance profiling. Validates the column-major cache locality assumption and provides the methodology for HIL cycle counting benchmarks.

### Property-Based Testing

**[60]** Claessen, Koen, and John Hughes. "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs." *ACM SIGPLAN Notices* 35, no. 9 (September 2000): 268--279.
- **DOI:** [10.1145/351240.351266](https://doi.org/10.1145/351240.351266)
- **Relevance:** The original property-based testing methodology paper. Directly informs the `proptest` crate usage in `control-rs` for verifying algebraic identities ($P+Q=Q+P$, $P(QR)=(PQ)R$) as specified in the Polynomial model verification benchmark.

**[60a]** The `proptest` Crate: Property-Based Testing for Rust.
- **URL:** [https://proptest-rs.github.io/proptest/](https://proptest-rs.github.io/proptest/) (source: [https://github.com/proptest-rs/proptest](https://github.com/proptest-rs/proptest))
- **Relevance:** The Rust implementation of property-based testing used in `control-rs`. Provides the actual API and strategy combinators for generating random matrix/polynomial test cases to verify algebraic identities. Cite alongside [60] for the methodology and this entry for the Rust-specific implementation.

---

## 9. Claim-to-Source Coverage Matrix

This matrix maps every major claim and assumption from the design documents to its primary supporting reference(s).

| Claim / Assumption | Primary Sources | Secondary Sources |
| :--- | :--- | :--- |
| **Column-major BLAS/LAPACK matching** | [1] Golub & Van Loan; [11] LAPACK Users' Guide; [8]--[10] BLAS specs | [18] BLASFEO; [40] CMSIS-DSP (row-major --- needs layout adapter) |
| **$LDL^T$ avoids square roots (symmetric systems)** | [12] Bunch & Kaufman; [1] Golub & Van Loan | [2] Higham |
| **$O(N^2)$ companion root-finding with backward stability** | [23] Aurentz et al. (2015); [24] Aurentz et al. (2016) | [25] Aurentz et al. (2018); [6] Wilkinson (AEP); [3] Trefethen & Bau; [26] Bini et al.; [27] Van Barel et al. |
| **Faddeev-LeVerrier characteristic polynomial recurrence** | [21] Faddeev & Faddeeva; [2] Higham; [6] Wilkinson (AEP) | [19] Knuth. Caution: "division-free" variant unverified |
| **Horner's method optimal evaluation ($N-1$ ops)** | [19] Knuth; [7]/[20] Wilkinson (REAP); [22] Henrici | [2] Higham |
| **Zero-copy views and FFI safety** | [44] Rust Reference; [45] Rustonomicon | [11] LAPACK; [40] CMSIS-DSP (requires layout adapter); [47] nalgebra |
| **Type-level dimension bounds (Peano types)** | [44] Rust Reference (const generics) | [46] Embedded Rust Book; [47] nalgebra |
| **4KB stack safety bound** | [46] Embedded Rust Book; [43] Yiu | [41] Cortex-M4 TRM; [42] Cortex-M7 TRM; [48] ISO 26262 |
| **BLAS Level 2/3 state propagation** | [9] Level 2 BLAS; [10] Level 3 BLAS | [28] Kailath; [30] Franklin et al. |
| **ZOH exact step-invariant discretization** | [15] Van Loan (1978); [13] Moler & Van Loan (1978) | [14] Moler & Van Loan (2003); [16] Higham (2005); [17] Al-Mohy & Higham (2009); [17a] Higham (2009 SIAM Rev.); [30] Franklin et al. |
| **Tustin/bilinear transform via TRSM** | [10] Level 3 BLAS (TRSM); [30] Franklin et al. | [31] Astrom & Wittenmark; [33] Oppenheim & Schafer |
| **Canonical form equivalence (CCF/OCF)** | [28] Kailath; [29] Chen | [5] Horn & Johnson; [32] Astrom & Murray |
| **Direct DSP convolution for interconnections** | [33] Oppenheim & Schafer; [36] Lyons | [35] Parhi |
| **Direct $H(j\omega)$ frequency evaluation** | [33] Oppenheim & Schafer | [19] Knuth (Horner); [22] Henrici |
| **IEEE 754 compliance / `-ffast-math` prohibition** | [39] IEEE 754-2019 | [2] Higham; [6] Wilkinson (AEP) |
| **Embedded CPU cache locality (Cortex-M4/M7)** | [42] Cortex-M7 TRM (L1 caches); [38] Hennessy & Patterson | [43] Yiu. Note: Cortex-M4 lacks general L1 cache |
| **Matrix inversion ill-conditioning ($N_x > 10$)** | [2] Higham; [4] Demmel | [1] Golub & Van Loan |
| **Tensor column-major stride matching** | [53] Kolda & Bader | [1] Golub & Van Loan |
| **Tensor Einstein summation / `contract_into`** | [53] Kolda & Bader | --- |
| **Edge AI / TinyML on Cortex-M** | [54] CMSIS-NN; [55] Warden & Situnayake; [56] David et al. (TFLM) | [40] CMSIS-DSP |
| **HIL DWT cycle profiling** | [57] ARMv7-M ARM (CYCCNT); [43] Yiu | [58] CoreSight Spec. Note: DWT counts cycles, not cache misses |
| **Sparse storage backend design** | [50] Davis; [51] Saad; [52] Duff et al. | --- |
| **Fixed-point Q31/Q15 scaling** | [37] Koren; [35] Parhi | [40] CMSIS-DSP |
| **Ascending power storage (index-to-exponent mapping)** | [19] Knuth; [33] Oppenheim & Schafer | [20] Wilkinson (REAP); [22] Henrici |
| **Companion matrix direct polynomial conversion** | [23] Aurentz et al.; [28] Kailath | [29] Chen |
| **Proper rational function / strict properness for CCF** | [28] Kailath; [29] Chen; [32a] Ogata | [30] Franklin et al. Strict: $N < D$ |
| **Structural sparsity potential (companion, tridiagonal)** | [50] Davis; [52] Duff et al. | [28] Kailath |
| **Property-based testing (algebraic identities)** | [60] Claessen & Hughes (QuickCheck); [60a] proptest crate | [49a] ISO/IEC/IEEE 29119-3 (test documentation) |
| **Kalman filter covariance update benchmark** | [34] Kalman (1960) | [32] Astrom & Murray |
| **Safety-critical software certification** | [48] ISO 26262-6; [49] DO-178C; [49a] ISO/IEC/IEEE 29119-3 | [46] Embedded Rust Book |

---

## Research Track Cross-Reference

| Research Track | Key References |
| :--- | :--- |
| **Track 1: Numerical Stability & High-Order Solvers** | [2], [3], [4], [6], [14], [15], [16], [17], [17a], [23], [24], [25] |
| **Track 2: Fixed-Point Scaling & Precision Drift** | [7]/[20], [21], [22], [35], [37], [39], [40] |
| **Track 3: Structured & Sparse Storage Backends** | [44], [45], [47], [50], [51], [52] |
| **Track 4: HIL Cycle Profiling** | [41], [42], [43], [57], [58] |
| **Track 5: MIMO & Edge AI Integration** | [53], [54], [55], [56] |
---

## DOI / Official Link Quick-Reference

| # | Reference | DOI / URL |
| :--- | :--- | :--- |
| 1 | Golub & Van Loan (2013) | [10.1137/1.9781421407944](https://doi.org/10.1137/1.9781421407944) |
| 2 | Higham, Accuracy & Stability (2002) | [10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027) |
| 3 | Trefethen & Bau (1997) | [10.1137/1.9781611977165](https://doi.org/10.1137/1.9781611977165) |
| 4 | Demmel (1997) | [10.1137/1.9781611971446](https://doi.org/10.1137/1.9781611971446) |
| 5 | Horn & Johnson (2013) | [10.1017/CBO9781139020411](https://doi.org/10.1017/CBO9781139020411) |
| 8 | Lawson et al. BLAS Level 1 (1979) | [10.1145/355841.355847](https://doi.org/10.1145/355841.355847) |
| 9 | Dongarra et al. BLAS Level 2 (1988) | [10.1145/42288.42291](https://doi.org/10.1145/42288.42291) |
| 10 | Dongarra et al. BLAS Level 3 (1990) | [10.1145/77626.79170](https://doi.org/10.1145/77626.79170) |
| 11 | LAPACK Users' Guide (1999) | [10.1137/1.9780898719604](https://doi.org/10.1137/1.9780898719604) |
| 12 | Bunch & Kaufman (1977) | [10.1090/S0025-5718-1977-0428694-0](https://doi.org/10.1090/S0025-5718-1977-0428694-0) |
| 13 | Moler & Van Loan (1978) | [10.1137/1020098](https://doi.org/10.1137/1020098) |
| 14 | Moler & Van Loan (2003) | [10.1137/S00361445024180](https://doi.org/10.1137/S00361445024180) |
| 15 | Van Loan, IEEE TAC (1978) | [10.1109/TAC.1978.1101743](https://doi.org/10.1109/TAC.1978.1101743) |
| 16 | Higham, Scaling & Squaring (2005) | [10.1137/04061101X](https://doi.org/10.1137/04061101X) |
| 17 | Al-Mohy & Higham (2009) | [10.1137/09074721X](https://doi.org/10.1137/09074721X) |
| 17a | Higham, SIAM Review (2009) | [10.1137/090768539](https://doi.org/10.1137/090768539) |
| 18 | Frison et al. BLASFEO (2018) | [10.1145/3210754](https://doi.org/10.1145/3210754) |
| 23 | Aurentz et al. (2015) | [10.1137/140983434](https://doi.org/10.1137/140983434) |
| 24 | Aurentz et al. Part II (2016) | [10.48550/arXiv.1611.02435](https://doi.org/10.48550/arXiv.1611.02435) |
| 25 | Aurentz et al. SIAM book (2018) | [10.1137/1.9781611975345](https://doi.org/10.1137/1.9781611975345) |
| 26 | Bini et al. (2010) | [10.1016/j.laa.2009.08.003](https://doi.org/10.1016/j.laa.2009.08.003) |
| 27 | Van Barel et al. (2010) | [10.1007/s00211-010-0302-y](https://doi.org/10.1007/s00211-010-0302-y) |
| 34 | Kalman (1960) | [10.1115/1.3662552](https://doi.org/10.1115/1.3662552) |
| 39 | IEEE 754-2019 | [10.1109/IEEESTD.2019.8766229](https://doi.org/10.1109/IEEESTD.2019.8766229) |
| 50 | Davis, Direct Methods (2006) | [10.1137/1.9780898718881](https://doi.org/10.1137/1.9780898718881) |
| 51 | Saad, Iterative Methods (2003) | [10.1137/1.9780898718003](https://doi.org/10.1137/1.9780898718003) |
| 52 | Duff et al. (2017) | [10.1093/acprof:oso/9780198508380.001.0001](https://doi.org/10.1093/acprof:oso/9780198508380.001.0001) |
| 53 | Kolda & Bader (2009) | [10.1137/07070111X](https://doi.org/10.1137/07070111X) |
| 54 | CMSIS-NN (2018) | [10.48550/arXiv.1801.06601](https://doi.org/10.48550/arXiv.1801.06601) |
| 56 | David et al. TFLM (2021) | [MLSys Proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf) |
| 48 | ISO 26262-6:2018 | [iso.org/standard/68388](https://www.iso.org/standard/68388.html) |
| 49 | DO-178C (2011) | [rtca.org/do-178](https://www.rtca.org/do-178/) |
| 49a | ISO/IEC/IEEE 29119-3:2021 | [iso.org/standard/78785](https://www.iso.org/standard/78785.html) |
| 60 | Claessen & Hughes (2000) | [10.1145/351240.351266](https://doi.org/10.1145/351240.351266) |
| 60a | proptest crate | [proptest-rs.github.io](https://proptest-rs.github.io/proptest/) |

---

*This bibliography is a living document. As implementation and benchmarking proceed, entries should be updated with empirical results citations and additional primary sources discovered during verification.*
