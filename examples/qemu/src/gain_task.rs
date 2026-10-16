//! A lifecycle task suite: scales each input sample by a host-adjustable gain.
//!
//! The same file builds into every QEMU target and the Teensy 4.1 firmware.

use control_rs_macros::ets_suite;

#[ets_suite]
/// Gain stage task: `output = GAIN * input`.
pub mod gain_task_suite {
    use control_rs_ets::exec::{
        HookOf, InputOf, LifecycleTask, SetupOf, StepOf, TaskStep,
    };
    use control_rs_ets::wire::{ExecMode, Param};
    use control_rs_macros::Packet;
    use serde::{Deserialize, Serialize};

    /// Output scale.
    #[param(min = 0.0, max = 10.0)]
    pub static GAIN: f32 = 2.0;

    /// One scalar sample.
    #[derive(Serialize, Deserialize, Packet)]
    pub struct Sample {
        pub value: f32,
    }

    /// The task state.
    pub struct Gain {
        scale: f32,
    }

    fn scale() -> f32 {
        match GAIN.get() {
            Param::F32(v) => v,
            _ => 0.0,
        }
    }

    /// Scales each input sample by `GAIN`.
    impl LifecycleTask for Gain {
        const MODES: u8 =
            ExecMode::Lockstep.bit() | ExecMode::FreeRunning.bit();
        const LINK_TIMEOUT_MS: u32 = 1000;
        type Input = Sample;
        type Output = Sample;

        fn setup() -> SetupOf<Self> {
            Ok(Self { scale: scale() })
        }

        fn step(
            &mut self,
            _step: u64,
            input: InputOf<'_, Self>,
        ) -> StepOf<Self> {
            TaskStep::Continue(Sample {
                value: self.scale * input.map_or(0.0, |s| s.value),
            })
        }

        fn reset(&mut self) -> HookOf {
            self.scale = scale();
            Ok(())
        }

        fn teardown(self) -> HookOf {
            Ok(())
        }

        fn safe_state() {}
    }
}
