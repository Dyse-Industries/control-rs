//! # Classical Control
//!
//! Single-input, single-output analysis, compensator construction and
//! firmware execution over [`Polynomial`](crate::polynomial::Polynomial) and
//! [`TransferFunction`](crate::transfer_function::TransferFunction).
//!
//! - **Analysis**: [`root_locus`], [`routh_count`], [`frequency_response`],
//!   [`nyquist_encirclements`], [`stability_margins`] and [`step_info`].
//! - **Compensators**: [`pid`](fn@pid), [`lead`] and [`lag`] build ordinary
//!   `TransferFunction` values.
//! - **Execution**: [`DirectForm2T`], [`Df1`] and [`Df2t`] sections in a
//!   [`BiquadCascade`], [`to_sections`], [`quantize`], [`is_stable`] and the
//!   discrete [`Pid`] controller.
//!
//! Every routine writes into caller-owned, statically sized buffers. No
//! routine allocates or plots. Errors arise at construction and analysis
//! only: per-sample `update`, `step`, `reset` and `set_coefficients` are
//! infallible.

pub use compensator::{PidForm, lag, lead, pid};
pub use locus::{LocusResult, root_locus};
pub use margins::{
    GainCrossing, GainCrossings, Margins, PhaseCrossing, PhaseCrossings,
    stability_margins,
};
pub use nyquist::{Contour, NyquistCount, nyquist_encirclements};
pub use pid::{AntiWindup, Pid, PidParams};
pub use realization::{
    BiquadCascade, Df1, Df2t, DirectForm2T, Section, SectionCoefficients,
    all_stable, is_stable, quantize, to_sections,
};
pub use response::{Response, ResponsePoint, frequency_response};
pub use routh::{RouthCount, routh_count};
pub use step::{StepInfo, StepOptions, StepResult, step_info};

use crate::math::num_types::{Const, Dim, DimAdd};
use crate::polynomial::RootError;
use crate::transfer_function::ArrayTransferFunction;
use core::fmt;

pub mod compensator;
pub mod locus;
pub mod margins;
pub mod nyquist;
pub mod pid;
pub mod realization;
pub mod response;
pub mod routh;
pub mod step;

/// Canonical type-level encoding of `Const<N>`.
pub(crate) type TypeNum<const N: usize> = <Const<N> as Dim>::TypeNum;

/// Type-level `N + 1`.
pub(crate) type Succ<const N: usize> = <TypeNum<N> as DimAdd<Const<1>>>::Output;

/// A constructed transfer function or the reason it cannot be built.
pub type TfResult<T, const N: usize, const D: usize> =
    Result<ArrayTransferFunction<T, N, D>, ClassicalError>;

/// Errors of the classical control toolbox.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassicalError {
    /// Root finding failed for a locus gain or a factorization (FR-1, FR-10).
    Root(RootError),
    /// A required leading coefficient is zero (FR-2, FR-8).
    ZeroLeadingCoefficient,
    /// The Nyquist contour passes through -1 (FR-4).
    ContourThroughCriticalPoint,
    /// A parameter is non-positive where positivity is required, or
    /// `u_min > u_max` (FR-6, FR-7, FR-11, FR-12).
    InvalidParameter,
    /// The requested PID form is improper and has no filter (FR-6).
    Improper,
    /// A realization was requested from a continuous system (FR-8, FR-10).
    NotDiscrete,
    /// A Nyquist count was requested for a discrete system (FR-4).
    NotContinuous,
    /// The section count does not match the factored order (FR-10).
    SectionCount,
    /// A coefficient of this section is outside the fixed-point range
    /// (FR-19).
    CoefficientRange {
        /// Index of the offending section.
        section: usize,
    },
}

impl fmt::Display for ClassicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Root(e) => write!(f, "root finding failed: {e}"),
            Self::ZeroLeadingCoefficient => {
                write!(f, "leading coefficient is zero")
            }
            Self::ContourThroughCriticalPoint => {
                write!(f, "Nyquist contour passes through -1")
            }
            Self::InvalidParameter => write!(f, "invalid parameter"),
            Self::Improper => {
                write!(f, "transfer function is improper or exceeds capacity")
            }
            Self::NotDiscrete => write!(f, "system is not discrete"),
            Self::NotContinuous => write!(f, "system is not continuous"),
            Self::SectionCount => {
                write!(f, "section count does not match the factored order")
            }
            Self::CoefficientRange { section } => write!(
                f,
                "coefficient of section {section} is outside the fixed-point range"
            ),
        }
    }
}

impl core::error::Error for ClassicalError {}

impl From<RootError> for ClassicalError {
    fn from(e: RootError) -> Self {
        Self::Root(e)
    }
}

/// Force-linked ETS suite descriptors of the realization and PID tests.
#[cfg(all(feature = "ets", not(test)))]
pub mod suites {
    pub use super::pid::tests::SUITE_DESCRIPTOR_PTR as pid;
    pub use super::realization::tests::SUITE_DESCRIPTOR_PTR as realization;
}
