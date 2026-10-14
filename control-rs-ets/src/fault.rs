//! Fault path state.
//!
//! # Description
//!
//! The runner publishes the active executable and the run phase here. A panic or exception
//! handler reads them after it masks interrupts and calls [`settle`], which drives the active
//! executable to its safe state at most once. No destructor runs on a fault, so teardown never
//! depends on `Drop`.
//!
//! # Core Concepts
//!
//! - **Phase**: `Idle`, `Armed` (setup may have run, steps may run), `TearingDown` and `Done`.
//! - **Settle**: From `Armed` the safe state runs once; from `TearingDown` a failed teardown is
//!   recorded without re-entry; from `Idle` or `Done` nothing happens.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::fault::{self, Settled};
//!
//! assert_eq!(fault::settle(), Settled::Nothing);
//! ```
//!
//! # Limitations
//!
//! - **Board capture**: Retained RAM and board fault registers are left to the binary.

use crate::cell::StateCell;
use crate::exec::ExecDescriptor;

static ACTIVE: Active = StateCell::new(None);
static PHASE: StateCell<Phase> = StateCell::new(Phase::Idle);

type Active = StateCell<Option<&'static ExecDescriptor>>;

/// The phase of a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// The run is armed: setup may have run and steps may run.
    Armed,
    /// Teardown has completed.
    Done,
    /// No run is active.
    Idle,
    /// Teardown is running.
    TearingDown,
}

/// What [`settle`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Settled {
    /// The fault came during teardown; the teardown is recorded as failed.
    FailedTeardown,
    /// Nothing was active.
    Nothing,
    /// The safe state of the active executable ran.
    SafeState,
}

/// Returns the phase of the current run.
#[must_use]
pub fn phase() -> Phase {
    PHASE.get()
}

/// Publishes the active executable and arms the run.
pub fn arm(exec: &'static ExecDescriptor) {
    ACTIVE.set(Some(exec));
    PHASE.set(Phase::Armed);
}

/// Moves the run to `phase`.
pub fn set_phase(phase: Phase) {
    PHASE.set(phase);
}

/// Clears the active executable and returns to `Idle`.
pub fn clear() {
    PHASE.set(Phase::Idle);
    ACTIVE.set(None);
}

/// Settles an active run on a fault.
pub fn settle() -> Settled {
    match PHASE.get() {
        Phase::Armed => {
            PHASE.set(Phase::TearingDown);
            if let Some(exec) = ACTIVE.get() {
                (exec.safe_state)();
            }
            PHASE.set(Phase::Done);
            Settled::SafeState
        }
        Phase::TearingDown => {
            PHASE.set(Phase::Done);
            Settled::FailedTeardown
        }
        Phase::Idle | Phase::Done => Settled::Nothing,
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::exec::{TaskIo, TaskOutcome};
    use crate::wire::Verdict;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static SAFE_STATES: AtomicUsize = AtomicUsize::new(0);
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn step(_: &mut TaskIo<'_>) -> TaskOutcome {
        TaskOutcome::Finish(Verdict::Pass, None)
    }

    fn count_safe_state() {
        SAFE_STATES.fetch_add(1, Ordering::SeqCst);
    }

    #[test]
    fn phase_transition_calls_safe_state_once() {
        static EXEC: ExecDescriptor = {
            let mut d = ExecDescriptor::case("e", "", step);
            d.safe_state = count_safe_state;
            d
        };
        let _g = SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        SAFE_STATES.store(0, Ordering::SeqCst);
        clear();
        assert_eq!(settle(), Settled::Nothing);

        arm(&EXEC);
        assert_eq!(phase(), Phase::Armed);
        assert_eq!(settle(), Settled::SafeState);
        assert_eq!(SAFE_STATES.load(Ordering::SeqCst), 1);
        assert_eq!(phase(), Phase::Done);
        assert_eq!(settle(), Settled::Nothing, "never from Done");

        arm(&EXEC);
        set_phase(Phase::TearingDown);
        assert_eq!(settle(), Settled::FailedTeardown);
        assert_eq!(
            SAFE_STATES.load(Ordering::SeqCst),
            1,
            "never from TearingDown"
        );

        clear();
        assert_eq!(settle(), Settled::Nothing);
        assert_eq!(SAFE_STATES.load(Ordering::SeqCst), 1, "never from Idle");
    }
}
