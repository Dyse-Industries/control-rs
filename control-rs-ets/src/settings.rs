//! Embedded Test Server (ETS) test suite configuration settings.
//!
//! # Description
//!
//! This module provides the `Setting` trait, value representations and thread-safe atomic wrappers
//! to manage configurable parameters within a test suite. These parameters can be queried, listed,
//! and updated dynamically by the host during ETS testing.
//!
//! # Core Concepts
//!
//! - **Setting Trait**: Provides a uniform interface to read and write typed values, retrieve names
//!   and descriptions and perform type checks.
//! - **Atomic Setting Wrappers**: Thread-safe implementations for primitive types (`bool`, `f32`,
//!   `i32`, `u8`, etc.) that allow tests to read configuration parameters while running, without
//!   requiring heavy lock primitives.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::settings::{Setting, SettingValue, AtomicU32Setting};
//!
//! // Define a setting with a default value of 100
//! static CYCLE_LIMIT: AtomicU32Setting = AtomicU32Setting::new(
//!     "cycle_limit",
//!     "Maximum cycles before test timeout",
//!     100
//! );
//!
//! assert_eq!(CYCLE_LIMIT.name(), "cycle_limit");
//! assert_eq!(CYCLE_LIMIT.get(), SettingValue::U32(100));
//!
//! // Update setting value safely
//! assert!(CYCLE_LIMIT.set(SettingValue::U32(200)).is_ok());
//! assert_eq!(CYCLE_LIMIT.get(), SettingValue::U32(200));
//! ```
//!
//! # Features
//!
//! - **Target Agnostic**: Standard Rust atomic primitives are used to offer thread-safe operation.
//!
//! # Limitations
//!
//! - **Type Coercion**: Values must match the expected type exactly; implicit conversions are not supported.

#![allow(clippy::arbitrary_source_item_ordering)]

use core::sync::atomic::{
    AtomicBool, AtomicI8, AtomicI32, AtomicU8, AtomicU16, AtomicU32, Ordering,
};

#[cfg(target_has_atomic = "64")]
use core::sync::atomic::AtomicU64;

/// A trait for a configurable setting.
///
/// This trait allows the Server to interact with settings of different
/// types in a uniform way.
/// `Sync` is required so the implementors can be placed in `static` memory.
///
/// # Safety
/// Implementors must ensure that get and set operations are thread-safe and maintain consistency.
/// This trait does not use `unsafe` code.
///
/// # Panics
/// Implementations of this trait do not panic under normal usage.
///
/// # Example
/// ```
/// use control_rs_ets::settings::{Setting, SettingValue, AtomicBoolSetting};
///
/// let s = AtomicBoolSetting::new("my_flag", "A flag", false);
/// assert_eq!(s.name(), "my_flag");
/// ```
pub trait Setting: Sync {
    /// Returns the description of the setting.
    ///
    /// # Returns
    /// * `&'static str` - The static description text.
    fn description(&self) -> &'static str;

    /// Returns the expected type of the setting.
    ///
    /// # Returns
    /// * `SettingType` - The enum representing the type of value this setting accepts.
    fn expected_type(&self) -> SettingType;

    /// Gets the current value of the setting.
    ///
    /// # Returns
    /// * `SettingValue` - The current wrapped value of the setting.
    fn get(&self) -> SettingValue;

    /// Returns the name of the setting.
    ///
    /// # Returns
    /// * `&'static str` - The static string identifier of the setting.
    fn name(&self) -> &'static str;

    /// Sets the value of the setting.
    ///
    /// # Arguments
    /// * `value` - The new value wrapper.
    ///
    /// # Returns
    /// * `SetResult`
    ///     * `Ok(())` - If the value was successfully updated.
    ///     * `Err(&'static str)` - If the value type does not match the expected type.
    ///
    /// # Errors
    /// Returns an error if the type of the provided value does not match the setting's expected type.
    fn set(&self, value: SettingValue) -> SetResult;
}

/// The type of a setting.
///
/// Defines the supported primitive data types for test suite settings.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize,
)]
pub enum SettingType {
    /// A boolean type.
    Bool,
    /// A 32-bit floating point type.
    F32,
    /// A 32-bit signed integer type.
    I32,
    /// An 8-bit signed integer type.
    I8,
    /// A 16-bit unsigned integer type.
    U16,
    /// A 32-bit unsigned integer type.
    U32,
    /// A 64-bit unsigned integer type.
    U64,
    /// An 8-bit unsigned integer type.
    U8,
}

/// A value that can be gotten or set by the Server.
///
/// Encapsulates the actual setting value payload corresponding to a `SettingType`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum SettingValue {
    /// A boolean value.
    Bool(bool),
    /// A 32-bit floating point value.
    F32(f32),
    /// A 32-bit signed integer value.
    I32(i32),
    /// An 8-bit signed integer value.
    I8(i8),
    /// A 16-bit unsigned integer value.
    U16(u16),
    /// A 32-bit unsigned integer value.
    U32(u32),
    /// A 64-bit unsigned integer value.
    U64(u64),
    /// An 8-bit unsigned integer value.
    U8(u8),
}

impl PartialEq for SettingValue {
    /// Compares two `SettingValue` instances for equality.
    ///
    /// # Returns
    /// * `bool` - `true` if types match and values are equal. Note that `F32` performs a bitwise comparison.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bool(l), Self::Bool(r)) => l == r,
            (Self::F32(l), Self::F32(r)) => l.to_bits() == r.to_bits(),
            (Self::I32(l), Self::I32(r)) => l == r,
            (Self::I8(l), Self::I8(r)) => l == r,
            (Self::U16(l), Self::U16(r)) => l == r,
            (Self::U32(l), Self::U32(r)) => l == r,
            (Self::U64(l), Self::U64(r)) => l == r,
            (Self::U8(l), Self::U8(r)) => l == r,
            _ => false,
        }
    }
}

impl Eq for SettingValue {}

/// The result of a set operation.
///
/// Indicates success or returns a static string description of the error.
pub type SetResult = Result<(), &'static str>;

macro_rules! impl_setting {
    ($struct_name:ident, $inner_type:ty, $atomic_type:ty, $type_variant:ident, $value_variant:ident, $desc:literal, $example_val:expr) => {
        #[doc = concat!(
            $desc,
            "\n\n",
            "This structure implements the [Setting] trait, wrapping an atomic variable ",
            "to allow safe multi-threaded access and modification of the configuration value.\n\n",
            "# Safety\n",
            "This struct does not use `unsafe` code.\n\n",
            "# Panics\n",
            "Instantiating or accessing this struct does not panic.\n\n",
            "# Example\n",
            "```\n",
            "use control_rs_ets::settings::{Setting, SettingValue};\n",
            "let setting = control_rs_ets::settings::", stringify!($struct_name), "::new(\"test_setting\", \"A test setting\", ", stringify!($example_val), ");\n",
            "assert_eq!(setting.name(), \"test_setting\");\n",
            "assert_eq!(setting.get(), SettingValue::", stringify!($value_variant), "(", stringify!($example_val), "));\n",
            "```"
        )]
        pub struct $struct_name {
            description: &'static str,
            name: &'static str,
            value: $atomic_type,
        }

        impl $struct_name {
            /// Creates a new setting wrapper.
            ///
            /// # Arguments
            /// * `name` - Static unique identifier.
            /// * `description` - Descriptive label for the setting.
            /// * `initial_value` - Starting value.
            ///
            /// # Returns
            /// * `Self` - The initialized setting instance.
            #[must_use]
            pub const fn new(
                name: &'static str,
                description: &'static str,
                initial_value: $inner_type,
            ) -> Self {
                Self {
                    description,
                    name,
                    value: <$atomic_type>::new(initial_value),
                }
            }
        }

        impl Setting for $struct_name {
            /// Returns the description of the setting.
            ///
            /// # Returns
            /// * `&'static str` - The static description.
            fn description(&self) -> &'static str {
                self.description
            }

            /// Returns the expected type of the setting.
            ///
            /// # Returns
            /// * `SettingType` - Expected type representation.
            fn expected_type(&self) -> SettingType {
                SettingType::$type_variant
            }

            /// Gets the current value.
            ///
            /// # Returns
            /// * `SettingValue` - Current wrapped value.
            fn get(&self) -> SettingValue {
                SettingValue::$value_variant(self.value.load(Ordering::Relaxed))
            }

            /// Returns the name of the setting.
            ///
            /// # Returns
            /// * `&'static str` - The setting's name.
            fn name(&self) -> &'static str {
                self.name
            }

            /// Sets the value.
            ///
            /// # Arguments
            /// * `value` - Setting value payload.
            ///
            /// # Returns
            /// * `SetResult` - Success or error description.
            ///
            /// # Errors
            /// Returns `Err` if the setting type mismatches.
            fn set(&self, value: SettingValue) -> SetResult {
                if let SettingValue::$value_variant(v) = value {
                    self.value.store(v, Ordering::Relaxed);
                    Ok(())
                } else {
                    Err(concat!(
                        "Type mismatch: expected ",
                        stringify!($type_variant)
                    ))
                }
            }
        }
    };
}

impl_setting!(
    AtomicBoolSetting,
    bool,
    AtomicBool,
    Bool,
    Bool,
    "A wrapper for a `bool` setting that can be safely shared between threads.",
    true
);

impl_setting!(
    AtomicI32Setting,
    i32,
    AtomicI32,
    I32,
    I32,
    "A wrapper for a `i32` setting that can be safely shared between threads.",
    10i32
);

impl_setting!(
    AtomicI8Setting,
    i8,
    AtomicI8,
    I8,
    I8,
    "A wrapper for a `i8` setting that can be safely shared between threads.",
    5i8
);

impl_setting!(
    AtomicU16Setting,
    u16,
    AtomicU16,
    U16,
    U16,
    "A wrapper for a `u16` setting that can be safely shared between threads.",
    20u16
);

impl_setting!(
    AtomicU32Setting,
    u32,
    AtomicU32,
    U32,
    U32,
    "A wrapper for a `u32` setting that can be safely shared between threads.",
    100u32
);

#[cfg(target_has_atomic = "64")]
impl_setting!(
    AtomicU64Setting,
    u64,
    AtomicU64,
    U64,
    U64,
    "A wrapper for a `u64` setting that can be safely shared between threads.",
    1000u64
);

impl_setting!(
    AtomicU8Setting,
    u8,
    AtomicU8,
    U8,
    U8,
    "A wrapper for a `u8` setting that can be safely shared between threads.",
    42u8
);

/// A wrapper for a `f32` setting that can be safely shared between threads.
///
/// This structure implements the [Setting] trait, wrapping an atomic representation of a float.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Instantiating or accessing this struct does not panic.
///
/// # Example
/// ```
/// use control_rs_ets::settings::{Setting, SettingValue, AtomicF32Setting};
///
/// let setting = AtomicF32Setting::new("cutoff_freq", "Lowpass cutoff frequency", 15.5);
/// assert_eq!(setting.name(), "cutoff_freq");
/// assert_eq!(setting.get(), SettingValue::F32(15.5));
/// ```
pub struct AtomicF32Setting {
    description: &'static str,
    name: &'static str,
    value: AtomicU32,
}

impl AtomicF32Setting {
    /// Creates a new `AtomicF32Setting`.
    ///
    /// # Arguments
    /// * `name` - Static unique identifier.
    /// * `description` - Descriptive label.
    /// * `initial_value` - Starting float value.
    ///
    /// # Returns
    /// * `Self` - The initialized setting instance.
    #[must_use]
    pub const fn new(
        name: &'static str,
        description: &'static str,
        initial_value: f32,
    ) -> Self {
        Self {
            description,
            name,
            value: AtomicU32::new(initial_value.to_bits()),
        }
    }
}

impl Setting for AtomicF32Setting {
    /// Returns the description of the setting.
    ///
    /// # Returns
    /// * `&'static str` - The static description.
    fn description(&self) -> &'static str {
        self.description
    }

    /// Returns the expected type of the setting.
    ///
    /// # Returns
    /// * `SettingType` - `SettingType::F32`
    fn expected_type(&self) -> SettingType {
        SettingType::F32
    }

    /// Gets the current value.
    ///
    /// # Returns
    /// * `SettingValue` - Current wrapped float value.
    fn get(&self) -> SettingValue {
        SettingValue::F32(f32::from_bits(self.value.load(Ordering::Relaxed)))
    }

    /// Returns the name of the setting.
    ///
    /// # Returns
    /// * `&'static str` - The setting's name.
    fn name(&self) -> &'static str {
        self.name
    }

    /// Sets the value.
    ///
    /// # Arguments
    /// * `value` - Setting value payload.
    ///
    /// # Returns
    /// * `SetResult` - Success or error description.
    ///
    /// # Errors
    /// Returns `Err` if the setting type is not `SettingValue::F32`.
    fn set(&self, value: SettingValue) -> SetResult {
        if let SettingValue::F32(v) = value {
            self.value.store(v.to_bits(), Ordering::Relaxed);
            Ok(())
        } else {
            Err("Type mismatch: expected F32")
        }
    }
}
