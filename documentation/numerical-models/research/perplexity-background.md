# Verified References Bibliography for control-rs Numerical Models

## Overview

This document provides a verified, source-linked bibliography for the nine technical domains identified in the background research evaluation. Each reference includes its DOI or official URL, a quality tier classification, and notes on replacements or supplements where the original report's citations were weak or outdated.

### Quality Tier Legend
- **[Canonical Textbook]** — Widely cited standard reference textbook (peer-reviewed academic publisher)
- **[Peer-Reviewed]** — Published in a peer-reviewed journal or conference proceedings
- **[SIAM Monograph]** — Published in the SIAM Fundamentals of Algorithms series
- **[Practitioner Reference]** — Authoritative industry/practitioner book (not peer-reviewed)
- **[Standard]** — Official technical standard from a recognized body (IEEE, ISO, RTCA)
- **[Official Docs]** — Official project/vendor documentation
- **[Supplement]** — Additional reference not in the original report, recommended for stronger grounding
- **[Replace]** — Replaces a weaker citation from the original report

---

## 1. Numerical Linear Algebra & Matrix Decompositions

### 1.1 Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- **DOI:** [10.1137/1.9781421407944](https://doi.org/10.1137/1.9781421407944)
- **Quality:** [Canonical Textbook]
- **Relevance:** The definitive reference for LU, Cholesky, QR, and all matrix factorizations. Covers pivoting strategies, triangular solvers, and BLAS-level operations.
- **Notes:** Published by JHU Press, distributed by SIAM. ISBN 978-1-4214-0794-4.

### 1.2 Anderson, E., et al. (1999). *LAPACK Users' Guide* (3rd ed.). SIAM.
- **DOI:** [10.1137/1.9780898719604](https://doi.org/10.1137/1.9780898719604)
- **Quality:** [Canonical Textbook]
- **Relevance:** Reference for BLAS Level 1/2/3 subprogram specifications, column-major storage conventions, and LAPACK algorithm design.
- **Notes:** SIAM Software, Environments, and Tools Series, Vol. 9. ISBN 978-0-89871-447-0.

### 1.3 Frison, G., Kouzoupis, D., Sartor, T., Zanelli, A., & Diehl, M. (2018). BLASFEO: Basic Linear Algebra Subroutines for Embedded Optimization. *ACM Transactions on Mathematical Software*, 44(4), 1–30.
- **DOI:** [10.1145/3210754](https://doi.org/10.1145/3210754)
- **Quality:** [Peer-Reviewed]
- **Relevance:** Panel-major (BLASFEO-style) layout trade-offs for embedded systems; performance optimized for small-to-medium matrices fitting in cache. Directly relevant to `control-rs` static-memory matrix design.
- **Notes:** Preprint available at [arXiv:1704.02457](https://arxiv.org/abs/1704.02457).

### 1.4 Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- **DOI:** [10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- **Quality:** [Canonical Textbook]
- **Relevance:** Condition numbers, IEEE 754 floating-point analysis, non-associativity under `-ffast-math`, normal equation squaring penalty, backward stability of LU/Cholesky/QR.
- **Notes:** ISBN 978-0-89871-521-7.

### 1.5 [Supplement] Higham, N. J. (2005). The Scaling and Squaring Method for the Matrix Exponential Revisited. *SIAM Journal on Matrix Analysis and Applications*, 26(4), 1179–1193.
- **DOI:** [10.1137/04061101X](https://doi.org/10.1137/04061101X)
- **Quality:** [Peer-Reviewed]
- **Relevance:** Provides the theoretical foundation for the scaling-and-squaring method with Pade approximants used in ZOH discretization. Should be cited alongside Moler & Van Loan (2003) for matrix exponential methods.

### 1.6 [Supplement] Higham, N. J. (2009). The Scaling and Squaring Method for the Matrix Exponential Revisited. *SIAM Review*, 51(4), 747–764.
- **DOI:** [10.1137/090768539](https://doi.org/10.1137/090768539)
- **Quality:** [Peer-Reviewed]
- **Relevance:** SIAM Review survey of the state-of-the-art matrix exponential computation. Complements Moler & Van Loan (2003).

---

## 2. Polynomial Algebra, Root-Finding & Fast Matrix Algorithms

### 2.1 Faddeev, D. K., & Faddeeva, V. N. (1963). *Computational Methods of Linear Algebra*. W. H. Freeman and Company.
- **Quality:** [Canonical Textbook]
- **Relevance:** Faddeev–LeVerrier trace-based characteristic polynomial recurrence for matrix-to-polynomial conversion.
- **Notes:** No DOI available (pre-DOI era). Available via library repositories. A modern treatment is also found in Golub & Van Loan (2013), Section 7.2. Note: the standard Faddeev–LeVerrier recurrence involves division by k; the original report's "division-free" characterization should be verified for the specific variant used.

### 2.2 Bini, D. A., Boito, P., Eidelman, Y., Gemignani, L., & Gohberg, I. (2010). A Fast Implicit QR Eigenvalue Algorithm for Companion Matrices. *Linear Algebra and its Applications*, 432(8), 2006–2031.
- **DOI:** [10.1016/j.laa.2009.08.003](https://doi.org/10.1016/j.laa.2009.08.003)
- **Quality:** [Peer-Reviewed]
- **Relevance:** O(N^2) time, O(N) space implicit QR algorithm for companion matrices using generator-based representation. Core algorithm for fast polynomial root-finding.

### 2.3 [Replace] Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2015). Fast and Backward Stable Computation of Roots of Polynomials. *SIAM Journal on Matrix Analysis and Applications*, 36(3), 942–973.
- **DOI:** [10.1137/140983434](https://doi.org/10.1137/140983434)
- **Quality:** [Peer-Reviewed] [Replace]
- **Relevance:** Companion QR algorithm with O(N^2) time and O(N) memory using planar rotators over unitary-plus-rank-one Hessenberg matrices. Proved backward stable.
- **Notes:** This is the **published journal version** of the "TW Reports" technical report cited in the original report. Cite this instead of the KU Leuven TW Report. A Part II follow-up appeared in *SIAM J. Matrix Anal. Appl.*, 39(3), 1245–1269 (2018) at [arXiv:1611.02435](https://arxiv.org/abs/1611.02435).

### 2.4 [Supplement] Aurentz, J. L., Mach, T., Robol, L., Vandebril, R., & Watkins, D. S. (2018). *Core-Chasing Algorithms for the Eigenvalue Problem*. SIAM.
- **Quality:** [SIAM Monograph] [Supplement]
- **Relevance:** SIAM monograph consolidating the companion matrix QR algorithm theory. The definitive reference for the unitary-plus-rank-one exploitation technique.
- **Notes:** Fundamentals of Algorithms series. Cite alongside the 2015 journal paper for a complete treatment.

### 2.5 Henrici, P. (1974). *Applied and Computational Complex Analysis, Volume 1: Power Series—Integration—Conformal Mapping—Location of Zeros*. Wiley.
- **Quality:** [Canonical Textbook]
- **Relevance:** Horner's method error analysis, polynomial evaluation and division fundamentals.
- **Notes:** ISBN 978-0-471-37244-6. No DOI available (pre-DOI era). Available via [Wiley](https://www.wiley.com/en-ae/Applied+and+Computational+Complex+Analysis,+3+Volume+Set-p-9780471598923).

### 2.6 [Supplement] Van Barel, M., Vandebril, R., Van Dooren, P., & Frederix, K. (2010). Implicit Double Shift QR-Algorithm for Companion Matrices. *Numerische Mathematik*, 116, 177–212.
- **DOI:** [10.1007/s00211-010-0302-y](https://doi.org/10.1007/s00211-010-0302-y)
- **Quality:** [Peer-Reviewed]
- **Relevance:** Alternative implicit double-shift QR for companion matrices using Givens rotation representation.

---

## 3. Control Systems Theory & State-Space Modeling

### 3.1 Ogata, K. (2010). *Modern Control Engineering* (5th ed.). Prentice Hall / Pearson.
- **Quality:** [Canonical Textbook]
- **Relevance:** LTI state-space representations, controllability/observability, canonical forms, similarity transformations.
- **Notes:** ISBN 978-0-13-615673-4. Available via [Pearson](https://www.pearson.com/en-us/subject-catalog/p/modern-control-engineering/P200000003521/9780136156734).

### 3.2 Åström, K. J., & Murray, R. M. (2021). *Feedback Systems: An Introduction for Scientists and Engineers* (2nd ed.). Princeton University Press.
- **Quality:** [Canonical Textbook]
- **Relevance:** State-space tools, Lyapunov stability, reachability, observability, estimators, matrix exponential in linear control, frequency domain design.
- **Notes:** ISBN 978-0-691-19398-4. Published February 2021. Available via [Princeton University Press](https://press.princeton.edu/books/hardcover/9780691193984/feedback-systems). Free online version available at [https://fbsbook.org](https://fbsbook.org).

### 3.3 Kailath, T. (1980). *Linear Systems*. Prentice-Hall.
- **Quality:** [Canonical Textbook]
- **Relevance:** System interconnections (series, parallel, feedback), controllability/observability theory, state-space to transfer function equivalence.
- **Notes:** ISBN 978-0-13-536961-6. No DOI available (pre-DOI era). Considered the definitive graduate-level reference for linear systems theory.

### 3.4 [Supplement] Kalman, R. E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.
- **DOI:** [10.1115/1.3662552](https://doi.org/10.1115/1.3662552)
- **Quality:** [Peer-Reviewed] [Supplement]
- **Relevance:** Original derivation of the discrete Kalman filter and covariance update equations. Essential primary source for the Kalman filtering section of `state-space-design.md`.
- **Notes:** Published in ASME Transactions, *J. Basic Eng.*, March 1960. Available via [ASME Digital Collection](https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction).

---

## 4. Digital Signal Processing & Model Discretization

### 4.1 Moler, C., & Van Loan, C. (2003). Nineteen Dubious Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1), 3–49.
- **DOI:** [10.1137/S00361445024180](https://doi.org/10.1137/S00361445024180)
- **Quality:** [Peer-Reviewed]
- **Relevance:** Survey of matrix exponential computation methods, including scaling-and-squaring with Pade approximants for ZOH discretization. Identifies which methods are numerically stable.
- **Notes:** Available via [SIAM Review](https://epubs.siam.org/doi/10.1137/S00361445024180).

### 4.2 [Supplement] Al-Mohy, A. H., & Higham, N. J. (2009). A New Scaling and Squaring Algorithm for the Matrix Exponential. *SIAM Journal on Matrix Analysis and Applications*, 31(3), 970–989.
- **DOI:** [10.1137/09074721X](https://doi.org/10.1137/09074721X)
- **Quality:** [Peer-Reviewed] [Supplement]
- **Relevance:** Improved scaling-and-squaring algorithm that alleviates the overscaling problem. Builds upon the Higham (2005) algorithm that underlies MATLAB's `expm` function. Should be cited alongside Moler & Van Loan (2003) as the modern state-of-the-art for matrix exponential computation in ZOH discretization.

### 4.3 Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Pearson.
- **Quality:** [Canonical Textbook]
- **Relevance:** Bilinear (Tustin) transform, frequency response, FIR/IIR filter representation, polynomial convolution for transfer function algebra.
- **Notes:** ISBN 978-0-13-198842-2. 3rd edition, 2009.

### 4.4 Franklin, G. F., Powell, J. D., & Workman, M. L. (1998). *Digital Control of Dynamic Systems* (3rd ed.). Addison-Wesley.
- **Quality:** [Canonical Textbook]
- **Relevance:** ZOH discretization, digital filter design, continuous-to-discrete conversion techniques, frequency pre-warping for bilinear transform.
- **Notes:** ISBN 978-0-201-82054-6. No DOI available.

---

## 5. Multidimensional Tensor Computations & Edge AI / TinyML

### 5.1 Kolda, T. G., & Bader, B. W. (2009). Tensor Decompositions and Applications. *SIAM Review*, 51(3), 455–500.
- **DOI:** [10.1137/07070111X](https://doi.org/10.1137/07070111X)
- **Quality:** [Peer-Reviewed]
- **Relevance:** Tensor contraction, Einstein summation, n-dimensional stride and index mapping conventions.
- **Notes:** Available via [SIAM Review](https://epubs.siam.org/doi/abs/10.1137/07070111X) and [ACM Digital Library](https://dl.acm.org/doi/10.1137/07070111X).

### 5.2 [Replace] Warden, P., & Situnayake, D. (2019). *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. O'Reilly Media.
- **Quality:** [Practitioner Reference] [Replace]
- **Relevance:** TinyML weight/bias/activation representation in static ROM/RAM, microcontroller neural network inference.
- **Notes:** This is the primary practitioner reference for TinyML. The original report also cited "Raychev et al. (2021), arXiv:2102.01255" — an arXiv-only survey. **Recommend citing the Warden & Situnayake book as the primary reference** and dropping the arXiv-only survey. A 2nd edition of the book exists (2022). ISBN 978-1-4920-8203-6.

### 5.3 [Supplement] David, R., Duke, J., Jain, A., Janapa Reddi, V., Jeffries, N., Li, J., Kreeger, N., Nappier, I., Natraj, M., Regev, S., Rhodes, R., Wang, T., & Warden, P. (2021). TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems. *Proceedings of Machine Learning and Systems (MLSys)*, 3, 800–811.
- **Quality:** [Peer-Reviewed] [Supplement]
- **Relevance:** Peer-reviewed description of the TFLM inference framework for running deep-learning models on embedded systems with static memory constraints. Directly relevant to the `tensor-design.md` Edge AI / TinyML section.
- **Notes:** Available via [MLSys 2021 Proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf). This is the peer-reviewed companion to the Warden & Situnayake practitioner book.

---

## 6. Fixed-Point Arithmetic & Numerical Stability Analysis

### 6.1 Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- **DOI:** [10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027)
- **Quality:** [Canonical Textbook]
- **Relevance:** Condition numbers, IEEE 754 non-associativity, normal equation squaring penalty, floating-point error analysis.
- **Notes:** See Section 1.4 above for full details.

### 6.2 IEEE Computer Society. (2019). *IEEE Standard for Floating-Point Arithmetic* (IEEE Std 754-2019).
- **DOI:** [10.1109/IEEESTD.2019.8766229](https://doi.org/10.1109/IEEESTD.2019.8766229)
- **Quality:** [Standard]
- **Relevance:** Formal specification of IEEE 754 floating-point formats, rounding modes, exception handling. Essential for understanding non-associativity under `-ffast-math`.
- **Notes:** Available via [IEEE Xplore](https://ieeexplore.ieee.org/document/8766229). Electronic ISBN 978-1-5044-5924-2.

### 6.3 [Supplement] ARM CMSIS-DSP Documentation.
- **Quality:** [Official Docs] [Supplement]
- **Relevance:** Q31/Q15 fixed-point arithmetic reference implementation. The CMSIS-DSP library provides optimized fixed-point kernels (q7, q15, q31, f32 datatypes) for Cortex-M4/M7 processors, directly relevant to the `control-rs` fixed-point arithmetic section.
- **Notes:**
  - Official ARM page: [developer.arm.com](https://developer.arm.com/Additional%20Resources/CMSIS%20DSP%20Software%20Library)
  - GitHub: [ARM-software/CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP)
  - ARM CMSIS technology page: [arm.com/technologies/cmsis](https://www.arm.com/technologies/cmsis)

### 6.4 [Replace] Yates, R. D. (2013). *Fixed-Point Arithmetic: An Introduction*. Digital Signal Processing Supplement.
- **Quality:** [Supplement]
- **Relevance:** Q-format fixed-point arithmetic fundamentals.
- **Notes:** This is an informal technical note, not a peer-reviewed publication. **Recommend supplementing or replacing** with the ARM CMSIS-DSP documentation (Section 6.3 above) for authoritative fixed-point arithmetic reference, and citing Higham (2002) for the numerical stability analysis.

---

## 7. Embedded Micro-Architecture, Real-Time Systems & Hardware Acceleration

### 7.1 Yiu, J. (2013). *The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors* (3rd ed.). Newnes / Elsevier.
- **Quality:** [Canonical Textbook]
- **Relevance:** FPU register pressure management (32 single-precision registers on Cortex-M4), NVIC, memory protection, DMA, cache behavior, CMSIS-DSP integration.
- **Notes:** ISBN 978-0-12-408082-9. Published October 2013 (3rd edition). Available via [Elsevier](https://shop.elsevier.com/books/the-definitive-guide-to-arm-cortex-m3-and-cortex-m4-processors/yiu/978-0-12-408082-9). Includes dedicated chapters on DSP features and CMSIS-DSP software libraries. Note: for Cortex-M7-specific cache behavior details, supplement with the ARM Cortex-M7 Architecture Reference Manual.

### 7.2 ARM Ltd. CMSIS-DSP Software Library Reference.
- **Quality:** [Official Docs]
- **Relevance:** ARM NEON SIMD intrinsics, CMSIS-DSP assembly kernels, zero-copy C-FFI pointer passing patterns.
- **Notes:** See Section 6.3 above for full URL references.

### 7.3 Hennessy, J. L., & Patterson, D. A. (2017). *Computer Architecture: A Quantitative Approach* (6th ed.). Morgan Kaufmann / Elsevier.
- **Quality:** [Canonical Textbook]
- **Relevance:** Cache locality, memory hierarchy, SIMD instruction set architecture, FPU design.
- **Notes:** ISBN 978-0-12-811905-1. Available via [Elsevier Educate](https://educate.elsevier.com/book/details/9780128119051). Authors are 2017 ACM A.M. Turing Award recipients. Note: a 7th edition (2025) now exists; the 6th edition (2017) matches the original report's citation.

---

## 8. Type-Level Metaprogramming & Memory Safety Architecture

### 8.1 [Replace] The Rust Project. *The Rustonomicon: The Dark Arts of Advanced and Unsafe Rust Programming*.
- **Quality:** [Official Docs] [Replace]
- **Relevance:** Unsafe Rust, memory safety architecture.
- **Notes:** Available at [doc.rust-lang.org/nomicon](https://doc.rust-lang.org/nomicon/). The original report cites "Rust Project Developers (2024)" — this is informal documentation, not a peer-reviewed publication. The Rustonomicon primarily covers unsafe Rust; for `#![no_std]` and embedded constraints, **supplement** with the following references:

### 8.2 [Supplement] The Rust Project. *The Rust Reference: Generic Parameters — Const Generics*.
- **Quality:** [Official Docs] [Supplement]
- **Relevance:** Compile-time const generic parameters for static shape and dimension validation (the Rust equivalent of Peano number type systems). Directly supports the `Dim`, `DimAdd`, `DimSub`, `DimMul` type-level arithmetic design.
- **Notes:** Available at [doc.rust-lang.org/reference/items/generics.html](https://doc.rust-lang.org/reference/items/generics.html). Also see the original RFC: [rust-lang/rfcs#2000](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md).

### 8.3 [Supplement] The Rust Project. *The Embedded Rust Book*.
- **Quality:** [Official Docs] [Supplement]
- **Relevance:** Official documentation for `#![no_std]` and `no_alloc` execution guarantees, stack memory budgeting, and embedded Rust development constraints.
- **Notes:** Available at [doc.rust-lang.org/embedded-book](https://doc.rust-lang.org/embedded-book/). More directly relevant to the `#![no_std]` / 4KB stack limit design than the Rustonomicon.

### 8.4 [Supplement] The nalgebra Project (Dimforge). *nalgebra: Linear Algebra Library for Rust*.
- **Quality:** [Official Docs] [Supplement]
- **Relevance:** Real-world Rust implementation of decoupled storage trait architecture (`ArrayStorage` for stack, `MatrixView`/`MatrixViewMut` for zero-copy views), type-level dimension tracking, and `#![no_std]` support. The `control-rs` design closely parallels nalgebra's architecture.
- **Notes:**
  - GitHub: [dimforge/nalgebra](https://github.com/dimforge/nalgebra)
  - Documentation: [nalgebra.org](https://nalgebra.org/docs/user_guide/getting_started/)

### 8.5 ISO. (2018). *ISO 26262-6:2018 Road vehicles — Functional safety — Part 6: Product development at the software level*.
- **Quality:** [Standard]
- **Relevance:** Safety-critical embedded software development requirements, software unit design and implementation, software unit verification.
- **Notes:** Available via [ISO](https://www.iso.org/standard/68388.html). Published 2018-12, 2nd edition, 57 pages. Technical Committee ISO/TC 22/SC 32.

### 8.6 RTCA / EUROCAE. (2011). *DO-178C: Software Considerations in Airborne Systems and Equipment Certification*.
- **Quality:** [Standard]
- **Relevance:** Airborne software certification framework, software verification processes, configuration management for safety-critical systems.
- **Notes:** Issued 2011-12-13 by RTCA Special Committee SC-205. Available via [RTCA Store](https://www.rtca.org/do-178/). European equivalent: ED-12C by EUROCAE. Referenced by FAA Advisory Circular AC 20-115D.

---

## 9. Verification, Validation & Property-Based Testing

### 9.1 Claessen, K., & Hughes, J. (2000). QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279.
- **DOI:** [10.1145/351240.351266](https://doi.org/10.1145/351240.351266)
- **Quality:** [Peer-Reviewed]
- **Relevance:** Property-based testing methodology for algebraic identity verification. Directly informs the `proptest` crate usage in `control-rs`.
- **Notes:** Published at ICFP '00. Available via [ACM Digital Library](https://dl.acm.org/doi/10.1145/351240.351266).

### 9.2 IEEE Computer Society. (2008). *IEEE Standard for Software and System Test Documentation* (IEEE Std 829-2008).
- **Quality:** [Standard]
- **Relevance:** Test documentation framework for verification and validation.
- **Notes:** Available via [IEEE Standards Association](https://standards.ieee.org/standard/829-2008.html). **Note:** IEEE 829-2008 has been **superseded** by the ISO/IEC/IEEE 29119 series (2013). Consider citing ISO/IEC/IEEE 29119-3:2021 ([ISO page](https://www.iso.org/standard/79429.html)) for the current standard on test documentation.

---

## Summary: Recommended Changes to Original Report

### References to Replace

| Original Citation | Replacement | Reason |
|:---|:---|:---|
| Aurentz et al. (2014), *TW Reports*, KU Leuven | Aurentz et al. (2015), *SIAM J. Matrix Anal. Appl.* 36(3):942–973, [DOI: 10.1137/140983434](https://doi.org/10.1137/140983434) | The TW Report was a preprint; the peer-reviewed journal version exists |
| Raychev et al. (2021), arXiv:2102.01255 | Warden & Situnayake (2019), O'Reilly (practitioner) + David et al. (2021), MLSys (peer-reviewed) | arXiv-only survey replaced with practitioner book + peer-reviewed TFLM paper |
| Yates (2013), informal note | ARM CMSIS-DSP documentation + Higham (2002) | Informal note replaced with official vendor docs and canonical textbook |
| Rustonomicon (2024) for `#![no_std]` | Embedded Rust Book + Rust Reference (Const Generics) + nalgebra docs | Rustonomicon covers unsafe Rust; Embedded Rust Book is the authoritative `#![no_std]` reference |

### References to Add (Supplements)

| Supplement | DOI/URL | Domain | Justification |
|:---|:---|:---|---|
| Al-Mohy & Higham (2009) | [10.1137/09074721X](https://doi.org/10.1137/09074721X) | Matrix Exponential | Modern scaling-and-squaring algorithm for `expm` |
| Higham (2005) | [10.1137/04061101X](https://doi.org/10.1137/04061101X) | Matrix Exponential | Theoretical foundation for scaling-and-squaring with Pade |
| Higham (2009) | [10.1137/090768539](https://doi.org/10.1137/090768539) | Matrix Exponential | SIAM Review survey, accessible overview |
| Aurentz et al. (2018) SIAM book | SIAM Fundamentals of Algorithms | Polynomial Roots | Definitive monograph on core-chasing algorithms |
| Kalman (1960) | [10.1115/1.3662552](https://doi.org/10.1115/1.3662552) | State-Space | Original primary source for Kalman filter |
| David et al. TFLM (2021) | [MLSys 2021 Proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf) | TinyML | Peer-reviewed TFLM framework paper |
| Rust const generics RFC #2000 | [RFC #2000](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md) | Type-Level Metaprogramming | Official language feature specification |
| Embedded Rust Book | [doc.rust-lang.org/embedded-book](https://doc.rust-lang.org/embedded-book/) | `#![no_std]` / Memory Safety | Official `#![no_std]` and embedded constraints documentation |
| nalgebra library | [GitHub](https://github.com/dimforge/nalgebra) | Storage Trait Architecture | Real-world precedent for `control-rs` design |
| ARM CMSIS-DSP | [GitHub](https://github.com/ARM-software/CMSIS-DSP) | Fixed-Point / Hardware | Authoritative fixed-point arithmetic reference |
| Van Barel et al. (2010) | [10.1007/s00211-010-0302-y](https://doi.org/10.1007/s00211-010-0302-y) | Polynomial Roots | Alternative implicit double-shift QR for companion matrices |

### Standards Updates

| Standard | Status | Recommendation |
|:---|:---|:---|
| IEEE 829-2008 | **Superseded** by ISO/IEC/IEEE 29119 series (2013) | Cite ISO/IEC/IEEE 29119-3:2021 for current test documentation standard |

---

## DOI / Official Link Quick-Reference Table

| # | Reference | DOI / URL |
|:---|:---|:---|
| 1 | Golub & Van Loan (2013) | [10.1137/1.9781421407944](https://doi.org/10.1137/1.9781421407944) |
| 2 | Anderson et al. LAPACK Guide (1999) | [10.1137/1.9780898719604](https://doi.org/10.1137/1.9780898719604) |
| 3 | Frison et al. BLASFEO (2018) | [10.1145/3210754](https://doi.org/10.1145/3210754) |
| 4 | Higham, Accuracy & Stability (2002) | [10.1137/1.9780898718027](https://doi.org/10.1137/1.9780898718027) |
| 5 | Higham, Scaling & Squaring (2005) | [10.1137/04061101X](https://doi.org/10.1137/04061101X) |
| 6 | Higham, SIAM Rev. (2009) | [10.1137/090768539](https://doi.org/10.1137/090768539) |
| 7 | Bini et al. (2010) | [10.1016/j.laa.2009.08.003](https://doi.org/10.1016/j.laa.2009.08.003) |
| 8 | Aurentz et al. (2015) | [10.1137/140983434](https://doi.org/10.1137/140983434) |
| 9 | Van Barel et al. (2010) | [10.1007/s00211-010-0302-y](https://doi.org/10.1007/s00211-010-0302-y) |
| 10 | Moler & Van Loan (2003) | [10.1137/S00361445024180](https://doi.org/10.1137/S00361445024180) |
| 11 | Al-Mohy & Higham (2009) | [10.1137/09074721X](https://doi.org/10.1137/09074721X) |
| 12 | Kolda & Bader (2009) | [10.1137/07070111X](https://doi.org/10.1137/07070111X) |
| 13 | Claessen & Hughes (2000) | [10.1145/351240.351266](https://doi.org/10.1145/351240.351266) |
| 14 | IEEE 754-2019 | [10.1109/IEEESTD.2019.8766229](https://doi.org/10.1109/IEEESTD.2019.8766229) |
| 15 | ISO 26262-6:2018 | [ISO standard page](https://www.iso.org/standard/68388.html) |
| 16 | DO-178C (2011) | [RTCA page](https://www.rtca.org/do-178/) |
| 17 | IEEE 829-2008 (superseded) | [IEEE SA page](https://standards.ieee.org/standard/829-2008.html) |
| 18 | ISO/IEC/IEEE 29119-3:2021 | [ISO standard page](https://www.iso.org/standard/79429.html) |
| 19 | Kalman (1960) | [10.1115/1.3662552](https://doi.org/10.1115/1.3662552) |
| 20 | David et al. TFLM (2021) | [MLSys Proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf) |
| 21 | ARM CMSIS-DSP | [GitHub](https://github.com/ARM-software/CMSIS-DSP) |
| 22 | Rust const generics RFC | [RFC #2000](https://github.com/rust-lang/rfcs/blob/master/text/2000-const-generics.md) |
| 23 | Rust Reference (generics) | [doc.rust-lang.org](https://doc.rust-lang.org/reference/items/generics.html) |
| 24 | Embedded Rust Book | [doc.rust-lang.org/embedded-book](https://doc.rust-lang.org/embedded-book/) |
| 25 | nalgebra | [GitHub](https://github.com/dimforge/nalgebra) |
| 26 | Åström & Murray (2021) | [Princeton UP](https://press.princeton.edu/books/hardcover/9780691193984/feedback-systems) |
| 27 | Ogata (2010) | [Pearson](https://www.pearson.com/en-us/subject-catalog/p/modern-control-engineering/P200000003521/9780136156734) |
| 28 | Yiu (2013) | [Elsevier](https://shop.elsevier.com/books/the-definitive-guide-to-arm-cortex-m3-and-cortex-m4-processors/yiu/978-0-12-408082-9) |
| 29 | Hennessy & Patterson (2017) | [Elsevier Educate](https://educate.elsevier.com/book/details/9780128119051) |
| 30 | Warden & Situnayake (2019) | O'Reilly Media (ISBN 978-1-4920-8203-6) |
| 31 | Faddeev & Faddeeva (1963) | W. H. Freeman (no DOI; pre-DOI era — verify via library catalog) |
| 32 | Henrici (1974) | [Wiley](https://www.wiley.com/en-ae/Applied+and+Computational+Complex+Analysis,+3+Volume+Set-p-9780471598923) |
| 33 | Kailath (1980) | Prentice-Hall (no DOI; pre-DOI era — verify via library catalog) |
| 34 | Oppenheim & Schafer (2009) | Pearson (ISBN 978-0-13-198842-2 — verify via publisher) |
| 35 | Franklin, Powell & Workman (1998) | Addison-Wesley (no DOI — verify via library catalog) |
| 36 | Aurentz et al. (2018) SIAM book | SIAM Fundamentals of Algorithms (verify via [SIAM](https://epubs.siam.org)) |
