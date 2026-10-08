//! Lockstep lifecycle suite for the QEMU targets.
//!
//! The step reads no peripheral. It returns `-0.5 * x` for the host's input
//! `x`, so a host simulation of the integrator `x + u` that starts at 1
//! halves its state every step, exactly in `f32`.

use control_rs_ets::{LoopContext, LoopStatus};
use control_rs_macros::ets_suite;

#[ets_suite]
/// Lockstep integrator over host input.
pub mod lockstep {
    use super::{LoopContext, LoopStatus};

    #[setup]
    fn setup() -> Result<(), &'static str> {
        Ok(())
    }

    /// Returns half the negated input.
    #[step]
    fn integrator(ctx: &LoopContext<'_, f32>) -> LoopStatus<f32> {
        let x = ctx.input().copied().unwrap_or(0.0);
        LoopStatus::Running(-0.5 * x)
    }

    #[reset]
    fn reset() -> Result<(), &'static str> {
        Ok(())
    }

    #[teardown]
    fn teardown() -> Result<(), &'static str> {
        Ok(())
    }
}
