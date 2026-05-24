//! # Testing
//!
//! The testing of these modules must ensure that the reference implementations act as perfect
//! oracles. Because the provided source code acts as a fallback for platforms lacking hardware-
//! acceleration; any deviation expected behavior will cascade into the rest of the crate. Any
//! future hardware-accelerated backbones, such as those utilizing Arm NEON intrinsics, must be
//! cross-verified against these exact software outputs to ensure absolute behavioral parity across
//! different deployment targets.
//!
//! ## Unit Verification
//!
//! **The oracle testing mechanism**: This involves verifying the exact deterministic outputs of
//! the fallback and reference implementations, specifically the unaccelerated basic subprogram
//! structures.
//!
//! ## Procedural Verification
//!
//! Rather than relying on hardcoded arrays, property testing continuously mutates vector and
//! matrix structures over millions of iterations to detect exceedingly rare data-dependent
//! anomalies that standard unit tests cannot uncover.
//!
//! ## Static Validation
//!
//! Because type-level mathematics executes entirely during compilation, the testing methodology
//! requires specialized frameworks capable of asserting that mathematically invalid code fails to
//! compile.
//!
//!
//! Here is the condensed, high-priority verification checklist extracted from the architectural report. It is organized by operational domain to ensure exhaustive coverage of compile-time constraints, numeric edge cases, and geometric boundaries.
//
// ---
//
// ## 1. Static Validation Checklist (Type-Level Mathematics)
//
// These tests validate that matrix and tensor geometry constraints are resolved correctly by the Rust compiler's trait solver at compile time.
//
// * [x] **Zero-Byte Memory Footprint:** Statically assert that Peano base types (`Zero`) and wrapper structures (`Successor<T>`) occupy exactly `0` bytes of memory at runtime.
// * [x] **Constant Generic Hoisting:** Verify that primitive numeric literals successfully hoist to their corresponding Peano representations (e.g., a dimension literal of `5` structurally matches `Successor` wrapped four times around a base `Successor`).
// * [x] **Addition Commutativity:** Assert that the trait solver evaluates `DimAdd<A, B>` as strictly equal to `DimAdd<B, A>` to guarantee layout independence during matrix concatenation.
// * [x] **Addition Base Cases:** Validate that adding any dimension to `Zero` yields the original dimension.
// * [x] **Subtraction Underflow Protection (Negative Compilation Test):** Ensure that attempting to compile a subtraction where the right-hand operand exceeds the left-hand operand (`DimSub<U2, U5>`) triggers a compile-time trait bound satisfaction error.
// * [x] **Multiplication Recursion Depth Limit:** Execute a type-level multiplication test up to the library's maximum macro-generated alias boundary (e.g., `U32`) to prove the compiler’s trait solver recursion limits are not breached.
// * [x] **Dynamic Min/Max Bounding:** Statically assert that type-level `Min` and `Max` traits correctly resolve bounding boxes for non-uniform tensor operations.
//
// ---
//
// ## 2. Level 1 Subprograms Checklist (Vector-to-Vector Operations)
//
// These runtime unit tests focus on precision degradation, iterator mechanics, and standard IEEE 754 floating-point pathologies within fallback vector loops.
//
// ### Scalar-Vector Operations (AXPY)
//
// * [x] **Mismatched Length Boundary Panic:** Assert that running the operation with vectors of unequal sizes triggers a debug panic under a debug profile.
// * [x] **Zero-Scale Identity Preservation:** Confirm that if the scaling scalar $\alpha = 0.0$, the destination vector remains completely unaltered, even if the input vector contains high-entropy data.
// * [x] **Zero-Vector Multiplicative Invariance:** Verify that if the input vector consists entirely of zeroes, the destination vector remains unchanged for any arbitrary value of $\alpha$.
// * [x] **NaN Poisoning Propagation:** Ensure that if $\alpha = \text{NaN}$, the entire output vector becomes poisoned with NaNs according to IEEE 754 rules.
// * [x] **Infinity Multiplicative Edge Cases:** Assert that if $\alpha = \infty$, multiplying by an input element of `0.0` yields `NaN`, while multiplying by non-zero inputs correctly yields $\pm\infty$.
//
// ### Dot Product & Norms (DOT / NRM2)
//
// * [x] **Geometric Orthogonality Verification:** Verify that the dot product of two mathematically perpendicular vectors returns exactly `0.0`.
// * [x] **Euclidean Norm Identity:** Confirm that the dot product of a vector with itself exactly matches its squared Euclidean norm.
// * [x] **Catastrophic Cancellation Tracking:** Construct a specialized vector pair interleaving massive and minuscule magnitudes to quantify and document precision loss derived from standard left-fold iterator accumulations.
// * [x] **Norm Premature Domain Overflow:** Verify that vectors composed of large, normal floats whose squares exceed representational limits collapse gracefully to `Infinity` during the summation step (pre-square root).
// * [x] **Norm Premature Domain Underflow:** Confirm that vectors composed of extremely small floats whose squares underflow collapse to `0.0` during the summation step.
//
// ### Index of Maximum Absolute Value (IAMAX)
//
// * [x] **Empty Slice Resolution Boundary:** Assert that passing an empty slice gracefully yields an index of `0` rather than causing an out-of-bounds runtime panic.
// * [x] **Iterator Stability (Duplicate Maximums):** Provide an array containing duplicate absolute maximum values at different indices; verify that the function consistently returns the *latest* index in the sequence.
// * [x] **Partial Ordering Corruptions (Embedded NaNs):** Document the position-dependent index distortion that occurs when a vector contains embedded NaNs, ensuring predictable degradation behaviors when signal integrity is compromised.
//
// ---
//
// ## 3. Level 2 & Level 3 Subprograms Checklist (Matrix Operations)
//
// These tests address row-major stride mathematics, dimensional flattening, pointer shifts, and buffer-clearing vulnerabilities.
//
// ### Matrix-Vector Operations (GEMV)
//
// * [x] **Asymmetric Rectangular Tall Processing:** Validate chunking iterators on tall geometries to ensure the outer loop correctly spans an elongated destination vector while the inner loops process shortened matrix rows.
// * [x] **Asymmetric Rectangular Wide Processing:** Validate chunking iterators on wide geometries to ensure the outer loop handles a shortened destination vector while computing expansive inner dot products.
// * [x] **Padded Memory Buffer Slices:** Provide an intentionally oversized matrix buffer while keeping the destination vector constrained; verify that the zipping iterators halt precisely at the destination boundary, safely ignoring trailing alignment padding.
// * [x] **Destination Overwrite Identity ($\alpha = 1.0, \beta = 0.0$):** Initialize the destination vector with high-entropy randomized data; assert that the vector is entirely eclipsed and overwritten by the calculation with zero residual data leakage.
// * [x] **Destination Suppression Identity ($\alpha = 0.0, \beta = 1.0$):** Confirm that the destination vector memory state is perfectly preserved and no matrix multiplication occurs.
//
// ### Matrix-Matrix Operations (GEMM)
//
// * [x] **Square Identity Stride Validation:** Multiply a sequentially populated square matrix against a perfect identity matrix; assert that the output layout exactly matches the original matrix.
// * [x] **Asymmetric Shared Axis Bounds:** Multiply matrices populated entirely with unit values; verify that every element in the destination matrix exactly equals the length of the internal shared dimension to prove row-major indexing does not drift out of phase.
// * [x] **Tainted State Nullification Failure ($\beta = 0.0$):** Initialize a destination matrix with NaNs and execute a multiplication with $\beta = 0.0$. Confirm that the matrix remains completely poisoned with NaNs (since $\text{NaN} \times 0.0 = \text{NaN}$ under IEEE 754 rules), proving that mathematical nullification does not safely clear memory buffers.
//
// ---
//
// ## 4. Property Fuzzing & Stability Checklist
//
// These tests utilize a fuzzer framework to execute millions of randomized iterations across the computational parameters space.
//
// * [x] **Symmetric Topology Preservation:** Continuously generate randomized symmetric matrices ($A = A^T$); verify that computing $A^2$ via GEMM always produces an output matrix that is identically symmetric.
// * [x] **Distributive Variance Bounds:** Generate a randomized matrix and two randomized vectors to compare $M(v_1 + v_2)$ against $Mv_1 + Mv_2$. Verify that the variance induced by floating-point associativity losses never exceeds a strictly defined, empirically calibrated epsilon threshold.
// * [x] **Denormalized/Subnormal Signal Decays:** Inject subnormal vector arrays into the loops; verify that decaying signals scale gradually down to absolute zero without thread lockups, crashes, or runtime pipeline panics.
// * [x] **Clone Call Performance Benchmarking:** Pass arrays of custom mock types containing internal atomic invocation counters into the subprograms; assert that the total number of structural clones remains strictly bounded within the algorithmic limits to prevent allocation overhead.
// * [x] **Mixed-Sign Zero Invariance:** Pass inputs featuring distinct positive (`+0.0`) and negative (`-0.0`) signed zeros; verify that bitwise sign parity is mathematically maintained through identity loops.

mod complex_num_tests;
mod convolution_tests;
mod fft_tests;
mod num_trait_tests;
mod num_type_tests;
mod op_tests;
mod storage_tests;
mod subprogram_tests;
