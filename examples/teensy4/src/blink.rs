//! Lifecycle suite that toggles the built-in LED at a frequency given by a
//! setting.

use core::cell::RefCell;
use core::sync::atomic::{AtomicU32, Ordering};

use control_rs_ets::{LoopContext, LoopStatus};
use control_rs_macros::ets_suite;
use cortex_m::interrupt::Mutex;
use teensy4_bsp::board;

use crate::MILLISECONDS;

/// The built-in LED, owned by the suite once `setup` in `main` hands it over.
static LED: Mutex<RefCell<Option<board::Led>>> = Mutex::new(RefCell::new(None));

/// Millisecond count at which the LED toggles next.
static NEXT_TOGGLE_MS: AtomicU32 = AtomicU32::new(0);

/// Moves the LED pin into the suite.
pub fn hand_over(led: board::Led) {
    cortex_m::interrupt::free(|cs| {
        if let Ok(mut slot) = LED.borrow(cs).try_borrow_mut() {
            *slot = Some(led);
        }
    });
}

/// Runs `f` on the LED inside a short critical section.
fn with_led(f: impl FnOnce(&board::Led)) {
    cortex_m::interrupt::free(|cs| {
        if let Ok(slot) = LED.borrow(cs).try_borrow() {
            if let Some(led) = slot.as_ref() {
                f(led);
            }
        }
    });
}

/// Turns the LED off and schedules the first toggle for now.
fn restart() {
    with_led(|led| led.clear());
    NEXT_TOGGLE_MS
        .store(MILLISECONDS.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// Toggles the LED when `blink_hz` says it is due.
fn tick(blink_hz: u32) {
    if blink_hz == 0 {
        return;
    }
    let now = MILLISECONDS.load(Ordering::Relaxed);
    let next = NEXT_TOGGLE_MS.load(Ordering::Relaxed);
    if (now.wrapping_sub(next) as i32) >= 0 {
        with_led(|led| led.toggle());
        let half_period_ms = (500 / blink_hz).max(1);
        NEXT_TOGGLE_MS
            .store(now.wrapping_add(half_period_ms), Ordering::Relaxed);
    }
}

#[ets_suite]
/// Blinks the built-in LED at a frequency set from the host.
pub mod led_blink {
    use super::{LoopContext, LoopStatus};
    use control_rs_ets::settings::{Setting, SettingValue};

    /// Blink frequency in hertz. Zero holds the LED off.
    pub static BLINK_HZ: u32 = 1;

    #[setup]
    fn setup() -> Result<(), &'static str> {
        super::restart();
        Ok(())
    }

    /// Toggles the LED every half period of the blink frequency.
    #[step(link_timeout_ms = 1000)]
    fn blink(_: &LoopContext<'_, ()>) -> LoopStatus<()> {
        let hz = match BLINK_HZ.get() {
            SettingValue::U32(v) => v,
            _ => 0,
        };
        super::tick(hz);
        LoopStatus::Running(())
    }

    #[reset]
    fn reset() -> Result<(), &'static str> {
        super::restart();
        Ok(())
    }

    #[teardown]
    fn teardown() -> Result<(), &'static str> {
        // Leave the LED off when the run ends.
        super::with_led(|led| led.clear());
        Ok(())
    }
}
