//! Hardware-in-the-loop test suite definitions.
//!
//! This module provides the necessary structures and traits for creating
//! and running test suites on embedded hardware.

#![allow(clippy::arbitrary_source_item_ordering)]

use core::sync::atomic::{
    AtomicBool, AtomicI8, AtomicI32, AtomicU8, AtomicU16, AtomicU32, Ordering,
};

#[cfg(target_has_atomic = "64")]
use core::sync::atomic::AtomicU64;

/// A trait for a configurable setting.
///
/// This trait allows the test runner to interact with settings of different
/// types in a uniform way.
/// `Sync` is required so the implementors can be placed in `static` memory.
pub trait Setting: Sync {
    /// Returns the description of the setting.
    fn description(&self) -> &'static str;
    /// Returns the expected type of the setting.
    fn expected_type(&self) -> SettingType;
    /// Gets the current value of the setting.
    fn get(&self) -> SettingValue;
    /// Returns the name of the setting.
    fn name(&self) -> &'static str;
    /// Sets the value of the setting.
    ///
    /// # Errors
    /// Returns an error if the type of the provided value does not match the setting's expected type.
    fn set(&self, value: SettingValue) -> SetResult;
}

/// The type of a setting.
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

/// A value that can be gotten or set by the test runner.
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
pub type SetResult = Result<(), &'static str>;

macro_rules! impl_setting {
    ($struct_name:ident, $inner_type:ty, $atomic_type:ty, $type_variant:ident, $value_variant:ident, $desc:literal) => {
        #[doc = $desc]
        pub struct $struct_name {
            description: &'static str,
            name: &'static str,
            value: $atomic_type,
        }

        impl $struct_name {
            /// Creates a new setting wrapper.
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
            fn description(&self) -> &'static str {
                self.description
            }

            fn expected_type(&self) -> SettingType {
                SettingType::$type_variant
            }

            fn get(&self) -> SettingValue {
                SettingValue::$value_variant(self.value.load(Ordering::Relaxed))
            }

            fn name(&self) -> &'static str {
                self.name
            }

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
    "A wrapper for a `bool` setting that can be safely shared between threads."
);

impl_setting!(
    AtomicI32Setting,
    i32,
    AtomicI32,
    I32,
    I32,
    "A wrapper for a `i32` setting that can be safely shared between threads."
);

impl_setting!(
    AtomicI8Setting,
    i8,
    AtomicI8,
    I8,
    I8,
    "A wrapper for a `i8` setting that can be safely shared between threads."
);

impl_setting!(
    AtomicU16Setting,
    u16,
    AtomicU16,
    U16,
    U16,
    "A wrapper for a `u16` setting that can be safely shared between threads."
);

impl_setting!(
    AtomicU32Setting,
    u32,
    AtomicU32,
    U32,
    U32,
    "A wrapper for a `u32` setting that can be safely shared between threads."
);

#[cfg(target_has_atomic = "64")]
impl_setting!(
    AtomicU64Setting,
    u64,
    AtomicU64,
    U64,
    U64,
    "A wrapper for a `u64` setting that can be safely shared between threads."
);

impl_setting!(
    AtomicU8Setting,
    u8,
    AtomicU8,
    U8,
    U8,
    "A wrapper for a `u8` setting that can be safely shared between threads."
);

/// A wrapper for a `f32` setting that can be safely shared between threads.
pub struct AtomicF32Setting {
    description: &'static str,
    name: &'static str,
    value: AtomicU32,
}

impl AtomicF32Setting {
    /// Creates a new `AtomicF32Setting`.
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
    fn description(&self) -> &'static str {
        self.description
    }

    fn expected_type(&self) -> SettingType {
        SettingType::F32
    }

    fn get(&self) -> SettingValue {
        SettingValue::F32(f32::from_bits(self.value.load(Ordering::Relaxed)))
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn set(&self, value: SettingValue) -> SetResult {
        if let SettingValue::F32(v) = value {
            self.value.store(v.to_bits(), Ordering::Relaxed);
            Ok(())
        } else {
            Err("Type mismatch: expected F32")
        }
    }
}
