# ClientClock Design Document

**Implementation Order:** 1
**Estimated Time:** 0.5 days

![Date Badge](https://img.shields.io/badge/Date-May_23,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Needs%20Review-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Context and Objective

The HIL harness will require a timekeeping interface. The type of
timer peripherals available on each mcu are different; The interface
should also not depend on a specific hardware bus/device.

To achieve this, the system relies on a common abstraction over the transport
layer. This allows the core test runner logic to be oblivious to whether it's
communicating over Segger RTT, a UART serial port, or Ethernet. In an embedded
context (`#![no_std]`), this interface must be deterministic, non-blocking, and
allocation-free.

## 2. Core Mechanics

### 2.1. Middleware Trait

The core of the abstraction is the `ClientClock` trait. Implementations of this
trait encapsulate the hardware-specific details of reading the current time.

```rust
pub trait ClientClock {
    /// Returns the current time in milliseconds.
    fn now_ms(&self) -> u32;

    /// Returns the current time in microseconds.
    fn now_us(&self) -> u64;
}
```

## 3. Usage

### 3.1. Client side (Target MCU)

This is the side the user must implement. For whatever project they are
using they will need to implement the trait for the hardware.

```rust
// Example skeleton for a target implementation
struct SysTickClock {
    systick: hardware::SysTick,
}

impl ClientClock for SysTickClock {
    fn now_ms(&self) -> u32 {
        // ... hardware specific implementation ...
        (self.now_us() / 1000) as u32
    }
}
```