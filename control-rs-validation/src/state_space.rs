//! State-space suite: stiffness, long-horizon accumulation, and rank loss.
//!
//! The kernels exercise the places where a discrete realization stops being
//! trustworthy: a $5 \times 10^{3}$ stiffness ratio pushed through ZOH, a 2000-step
//! trajectory that integrates per-step rounding, a similarity transform by an
//! ill-conditioned $T$, and controllability/observability matrices of a graded
//! system that are numerically rank-deficient.
//!
//! Timing lives in `benches/numerical_models.rs`, not here.

use serde_json::{Value, json};

use control_rs::matrix::{ColVector, Owned, RowVector};
use control_rs::state_space::ArrayStateSpace;

/// Datasets gated by the comparator; every other key is context only.
pub const GATED_PATHS: &[&str] = &[
    "phase_portrait/theta",
    "phase_portrait/theta_dot",
    "stiff_zoh/ad",
    "stiff_zoh/bd",
    "stiff_step/y",
    "similarity/a_tilde",
    "ctrb/matrix",
    "obsv/matrix",
];

/// Samples in the long-horizon trajectory.
const HORIZON: usize = 2000;

/// Sample period of the pendulum realization (s).
const PENDULUM_TS: f64 = 0.05;

/// Sample period of the stiff realization (s).
const STIFF_TS: f64 = 1e-3;

/// Sampled trajectory $(\theta_k, \dot\theta_k)$.
type Trajectory = (Vec<f64>, Vec<f64>);

/// Order of the graded system used for the rank-loss kernels.
const GRADED_N: usize = 6;

/// Discrete realization of a lightly damped pendulum linearization.
///
/// Damping is small on purpose: over `HORIZON` steps the trajectory keeps
/// whatever per-step discretization error the implementation makes instead of
/// decaying it away.
struct Pendulum {
    /// ZOH discretization at `PENDULUM_TS`.
    sys_d: ArrayStateSpace<f64, 2, 1, 1>,
}

impl Pendulum {
    /// Discretize $\ddot\theta + b\dot\theta + \omega_0^2 \theta = u$.
    fn new(omega0: f64, b: f64, dt: f64) -> Self {
        let a_c =
            Owned::<f64, 2, 2>::from_rows([[0.0, 1.0], [-omega0 * omega0, -b]]);
        let b_c = ColVector::<f64, 2>::from_column([0.0, 1.0]);
        let c_c = RowVector::<f64, 2>::from_row([1.0, 0.0]);
        let d_c = Owned::<f64, 1, 1>::scalar(0.0);

        Self {
            sys_d: ArrayStateSpace::continuous(a_c, b_c, c_c, d_c)
                .to_discrete_zoh(dt),
        }
    }

    /// Free response from `x0`, returning $(\theta_k, \dot\theta_k)$.
    fn simulate(&self, x0: [f64; 2], n_steps: usize) -> Trajectory {
        let mut x = ColVector::<f64, 2>::from_column(x0);
        let u = Owned::<f64, 1, 1>::scalar(0.0);
        let mut theta = Vec::with_capacity(n_steps);
        let mut theta_dot = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            theta.push(x.get(0, 0).copied().unwrap_or(0.0));
            theta_dot.push(x.get(1, 0).copied().unwrap_or(0.0));
            let (x_next, _) = self.sys_d.step(&x, &u);
            x = x_next;
        }

        (theta, theta_dot)
    }
}

/// 2000-step free response released near the inverted equilibrium.
fn phase_portrait() -> Value {
    let pendulum = Pendulum::new(2.0, 0.02, PENDULUM_TS);
    let (theta, theta_dot) =
        pendulum.simulate([std::f64::consts::PI - 0.15, 0.5], HORIZON);

    json!({
        "theta": theta,
        "theta_dot": theta_dot,
        "ts": PENDULUM_TS,
        "steps": HORIZON,
    })
}

/// Continuous stiff plant with eigenvalues $-1$ and $-5 \times 10^{3}$.
fn stiff_plant() -> ArrayStateSpace<f64, 2, 1, 1> {
    ArrayStateSpace::<f64, 2, 1, 1>::continuous(
        [[-1.0, 0.0], [0.0, -5.0e3]],
        [[1.0], [1.0]],
        [[1.0, 1.0]],
        [[0.0]],
    )
}

/// ZOH discretization of the stiff plant at `STIFF_TS`.
///
/// The stiffness ratio is $5 \times 10^{3}$ and $|\lambda_{\text{fast}} T_s| = 5$,
/// which is where a series-based $\Psi = \int_0^{T_s} e^{A\tau} d\tau$ starts
/// to lose digits: $B_d$ here is accurate to $\sim 10^{-5}$ relative, against
/// $10^{-10}$ at $|\lambda T_s| = 3$. Past $|\lambda T_s| \approx 8$ the
/// current implementation diverges outright, so the kernel sits at the edge of
/// the usable range on purpose.
fn stiff_zoh() -> Value {
    let sys_d = stiff_plant().to_discrete_zoh(STIFF_TS);

    json!({ "ad": sys_d.a().to_rows(), "bd": sys_d.b().to_rows(), "ts": STIFF_TS })
}

/// 500-sample unit-step response of the stiff discrete realization.
fn stiff_step() -> Value {
    let sys_d = stiff_plant().to_discrete_zoh(STIFF_TS);
    let u = Owned::<f64, 1, 1>::scalar(1.0);
    let mut x = Owned::<f64, 2, 1>::zero();
    let mut y = Vec::with_capacity(500);

    for _ in 0..500 {
        let (x_next, y_k) = sys_d.step(&x, &u);
        y.push(y_k.get(0, 0).copied().unwrap_or(0.0));
        x = x_next;
    }

    json!({ "y": y })
}

/// Similarity transform by an ill-conditioned $T$, $\kappa_2(T) \approx 10^{8}$.
fn similarity() -> Value {
    let sys = ArrayStateSpace::<f64, 2, 1, 1>::continuous(
        [[0.0, 1.0], [-4.0, -0.8]],
        [[0.0], [1.0]],
        [[1.0, 0.0]],
        [[0.0]],
    );
    let t = Owned::<f64, 2, 2>::from_rows([[1.0, 1.0e8], [1.0e-8, 2.0]]);
    let similar = sys
        .similarity_transform(&t)
        .expect("det(T) = 2 - 1 = 1, so T is invertible");

    json!({ "a_tilde": similar.a().to_rows(), "t": t.to_rows() })
}

/// Graded 6-state system whose modes span six decades.
fn graded_system() -> ArrayStateSpace<f64, GRADED_N, 1, 1> {
    let a = Owned::<f64, GRADED_N, GRADED_N>::from_fn(|i, j| {
        let decade = i32::try_from(i).expect("GRADED_N fits in i32") - 3;
        if i == j {
            -(10f64.powi(decade))
        } else {
            0.01 / ((i + j + 1) as f64)
        }
    });
    let b = Owned::<f64, GRADED_N, 1>::from_fn(|i, _| 1.0 / ((i + 1) as f64));
    let c = Owned::<f64, 1, GRADED_N>::from_fn(|_, j| 1.0 / ((j + 1) as f64));
    let d = Owned::<f64, 1, 1>::scalar(0.0);

    ArrayStateSpace::continuous(a, b, c, d)
}

/// Controllability matrix of the graded system.
///
/// Columns $A^k B$ collapse onto the dominant mode as $k$ grows, so the
/// matrix is numerically rank-deficient and its trailing columns are pure
/// rounding.
fn ctrb() -> Value {
    json!({ "matrix": graded_system().controllability_matrix::<GRADED_N>().to_rows() })
}

/// Observability matrix of the same graded system.
fn obsv() -> Value {
    json!({ "matrix": graded_system().observability_matrix::<GRADED_N>().to_rows() })
}

/// Assemble the full state-space-suite payload.
#[must_use]
pub fn payload() -> Value {
    json!({
        "phase_portrait": phase_portrait(),
        "stiff_zoh": stiff_zoh(),
        "stiff_step": stiff_step(),
        "similarity": similarity(),
        "ctrb": ctrb(),
        "obsv": obsv(),
    })
}

/// Emit `results/state_space.rust.h5`.
pub fn run() {
    println!("state_space: stiffness and long-horizon accumulation");
    crate::write_rust_container("state_space", &payload(), GATED_PATHS);
}
