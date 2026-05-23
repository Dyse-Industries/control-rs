### Phase 1: The Concrete Foundation (Weeks 1-2)

*The goal here is to un-comment `polynomial.rs`, `state_space.rs`, and `transfer_function.rs` and make them compile
using the absolute simplest memory structures.*

* **Task 1: The Iterator Initializer:** Write that single
  `storage::from_iterator<T, const N: usize>(iter: impl Iterator<Item = T>) -> ArithmeticResult<[T; N]>` function using
  `MaybeUninit`. Keep it isolated.
* **Task 2: State-Space Structs:** Define your `StateSpace` model using pure `[T; ROWS * COLS]` (or 1D/2D arrays). Do
  not write generic matrix traits; just hardcode the dense arrays.
* **Task 3: Basic Math Wiring:** Ensure `subprograms::level1::AXPY` and your matrix multiplication functions work
  seamlessly with your new `StateSpace` structs.
* **Milestone:** You can define a state-space system `(A, B, C, D)` in code and multiply it by a state vector `x`
  without the compiler yelling at you.

---

### Phase 2: Dynamics & Simulation (Week 3)

*Before you can control a system, you have to be able to step it forward in time. Your `integrators` module needs life.*

* **Task 1: Forward/Backward Euler:** Implement the simplest possible numerical integrators.
* **Task 2: Runge-Kutta 4 (RK4):** Implement a fixed-step RK4 integrator. This will heavily stress-test your `math::ops`
  and array allocations, ensuring you aren't blowing up the stack.
* **Milestone:** You can pass a `StateSpace` model into an integrator and simulate its open-loop response over time.

---

### Phase 3: Classical Tools (Week 4)

*Get an easy win. Classical control is mathematically simple but structurally important.*

* **Task 1: The PID Controller:** Implement a standard, anti-windup PID controller in `classical_tools`. It only
  requires basic scalar or vector arithmetic, meaning you can focus purely on the API design.
* **Task 2: Hardware-in-the-loop Mock:** Write a test that loops the PID controller output into a simple simulated
  plant (using your RK4 integrator) and tracks a reference signal.
* **Milestone:** A fully functioning, testable PID loop using your custom math types.

---

### Phase 4: Modern Tools (Weeks 5-6)

*This is the core of `control-rs`. Now that you know your arrays and integrators work, build the matrix-heavy
algorithms.*

* **Task 1: LQR State Feedback:** Implement $u = -Kx$. Assume the $K$ matrix is pre-computed offline (e.g., via
  MATLAB/Python) and passed in as a const generic array.
* **Task 2: The Discrete Kalman Filter:** Implement the predict and update steps. This will require matrix transposition
  and inversion. *This* is where you will find out exactly what matrix math you actually need, without having guessed
  beforehand.
* **Milestone:** You can simulate a noisy state-space model, filter it with a Kalman Filter, and control it with LQR.

---

### Phase 5: The "Wait, do I need Storage.rs?" Review (Week 7)

*Only after you have a working Kalman Filter and LQR controller do you look back at your code.*

* **Task 1: Refactoring:** Look at where your code is ugly. Are you duplicating dense matrix multiplication everywhere?
  Is the lack of a sparse representation actually hurting your CPU cycles on the hardware?
* **Task 2:** If you feel the pain, abstract it. If the simple `[T; N]` arrays are fast enough and clean enough, you
  leave them alone and move on to Nonlinear/Robust tools.