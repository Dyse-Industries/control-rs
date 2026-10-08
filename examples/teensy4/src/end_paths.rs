//! Lifecycle suite that exercises every end path of a lifecycle run.
//!
//! The `mode` setting selects the behavior; the host sets it before each run.

use control_rs_ets::{LoopContext, LoopStatus};
use control_rs_macros::ets_suite;

#[ets_suite]
/// Exercises every end path of a lifecycle run.
pub mod lifecycle_end_paths {
    use super::{LoopContext, LoopStatus};
    use control_rs_ets::settings::{Setting, SettingValue};

    /// Selects the end path of the lifecycle case.
    pub static MODE: u8 = 0;

    fn _mode() -> u8 {
        match MODE.get() {
            SettingValue::U8(v) => v,
            _ => 0,
        }
    }

    /// The mode setting names a defined end path.
    fn mode_is_valid() {
        assert!(_mode() <= 11);
    }

    #[setup]
    fn setup() -> Result<(), &'static str> {
        match _mode() {
            5 => Err("setup failed in mode 5"),
            _ => Ok(()),
        }
    }

    /// Returns `(k, -0.5 * x)` for the newest input `x` and ends as `mode` says.
    #[step(link_timeout_ms = 1000)]
    fn end_paths(ctx: &LoopContext<'_, f32>) -> LoopStatus<(u64, f32)> {
        let k = ctx.step();
        let x = ctx.input().copied().unwrap_or(0.0);
        let out = (k, -0.5 * x);
        match _mode() {
            0 | 5 | 6 | 8 | 11 => LoopStatus::Running(out),
            1 if k >= 10 => LoopStatus::Pass(None),
            2 if k >= 5 => LoopStatus::Pass(None),
            2 => LoopStatus::Warn(out, Some("warning before step 5")),
            3 if k == 5 => LoopStatus::Fail(Some("failed at step 5")),
            4 if k == 5 => LoopStatus::Error(Some("error at step 5")),
            7 | 10 if k == 5 => LoopStatus::Pass(None),
            9 if k == 5 => panic!("mode 9 panics at step 5"),
            1 | 3 | 4 | 7 | 9 | 10 => LoopStatus::Running(out),
            _ => LoopStatus::Error(Some("unknown mode")),
        }
    }

    #[reset]
    fn reset() -> Result<(), &'static str> {
        match _mode() {
            6 => Err("reset failed in mode 6"),
            _ => Ok(()),
        }
    }

    #[teardown]
    fn teardown() -> Result<(), &'static str> {
        match _mode() {
            7 => Err("teardown failed in mode 7"),
            10 => panic!("mode 10 panics in teardown"),
            _ => Ok(()),
        }
    }
}
