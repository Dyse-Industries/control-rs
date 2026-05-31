//! Hardware-in-the-loop test suite definitions.
//!
//! This module provides the necessary structures and traits for creating
//! and running test suites on embedded hardware.

use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

/// Describes a single test executable.
#[derive(Debug, Clone, Copy)]
pub struct ExecDescriptor {
    /// The name of the test executable.
    pub name: &'static str,
    /// A function pointer to the test executable.
    pub test_fn: fn(),
}

/// Describes a test suite.
pub struct SuiteDescriptor {
    /// The name of the test suite.
    pub name: &'static str,
    /// A slice of test executables in this suite.
    pub executables: &'static [ExecDescriptor],
    /// A slice of configurable settings for this suite.
    pub settings: &'static [&'static dyn Setting],
}

/// A value that can be gotten or set by the test runner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SettingValue {
    /// An 8-bit unsigned integer value.
    U8(u8),
    /// A 32-bit unsigned integer value.
    U32(u32),
}

/// The type of a setting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SettingType {
    /// An 8-bit unsigned integer type.
    U8,
    /// A 32-bit unsigned integer type.
    U32,
}

/// A trait for a configurable setting.
///
/// This trait allows the test runner to interact with settings of different
/// types in a uniform way.
/// `Sync` is required so the implementors can be placed in `static` memory.
pub trait Setting: Sync {
    /// Returns the name of the setting.
    fn name(&self) -> &'static str;
    /// Returns the expected type of the setting.
    fn expected_type(&self) -> SettingType;
    /// Gets the current value of the setting.
    fn get(&self) -> SettingValue;
    /// Sets the value of the setting.
    fn set(&self, value: SettingValue) -> Result<(), &'static str>;
}

/// A wrapper for a `u32` setting that can be safely shared between threads.
pub struct AtomicU32Setting {
    name: &'static str,
    value: AtomicU32,
}

impl AtomicU32Setting {
    /// Creates a new `AtomicU32Setting`.
    pub const fn new(name: &'static str, initial_value: u32) -> Self {
        Self {
            name,
            value: AtomicU32::new(initial_value),
        }
    }
}

impl Setting for AtomicU32Setting {
    fn name(&self) -> &'static str {
        self.name
    }
    fn expected_type(&self) -> SettingType {
        SettingType::U32
    }

    fn get(&self) -> SettingValue {
        SettingValue::U32(self.value.load(Ordering::Relaxed))
    }

    fn set(&self, value: SettingValue) -> Result<(), &'static str> {
        if let SettingValue::U32(v) = value {
            self.value.store(v, Ordering::Relaxed);
            Ok(())
        } else {
            Err("Type mismatch: expected U32")
        }
    }
}

/// A wrapper for a `u8` setting that can be safely shared between threads.
pub struct AtomicU8Setting {
    name: &'static str,
    value: AtomicU8,
}

impl AtomicU8Setting {
    /// Creates a new `AtomicU8Setting`.
    pub const fn new(name: &'static str, initial_value: u8) -> Self {
        Self {
            name,
            value: AtomicU8::new(initial_value),
        }
    }
}

impl Setting for AtomicU8Setting {
    fn name(&self) -> &'static str {
        self.name
    }
    fn expected_type(&self) -> SettingType {
        SettingType::U8
    }

    fn get(&self) -> SettingValue {
        SettingValue::U8(self.value.load(Ordering::Relaxed))
    }

    fn set(&self, value: SettingValue) -> Result<(), &'static str> {
        if let SettingValue::U8(v) = value {
            self.value.store(v, Ordering::Relaxed);
            Ok(())
        } else {
            Err("Type mismatch: expected U8")
        }
    }
}