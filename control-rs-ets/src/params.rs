//! Parameter table of a test suite.
//!
//! # Description
//!
//! A suite's parameters form a flat table of named primitives that the host reads and writes by
//! name or by index, in the style of the `MAVLink` parameter protocol. Each [`ParamEntry`] owns its
//! name, description, bounds and value. One entry type holds any [`Param`], so a table needs no
//! trait and no per-type wrapper.
//!
//! # Core Concepts
//!
//! - **Entry**: Name, description, bounds and an atomic value cell. The cell's type is the type of
//!   the bounds.
//! - **Table**: A `&'static [&'static ParamEntry]` in the suite descriptor. Index is the position.
//! - **Write rule**: A write whose type differs from the entry's, or whose value lies outside the
//!   bounds, leaves the value unchanged and reports an error.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::params::ParamEntry;
//! use control_rs_ets::wire::Param;
//!
//! static GAIN: ParamEntry =
//!     ParamEntry::u32("gain", "Loop gain", 100).bounded(Param::U32(10), Param::U32(500));
//!
//! assert_eq!(GAIN.get(), Param::U32(100));
//! assert!(GAIN.set(Param::U32(200)).is_ok());
//! assert!(GAIN.set(Param::U32(900)).is_err());
//! assert!(GAIN.set(Param::U8(1)).is_err());
//! assert_eq!(GAIN.get(), Param::U32(200));
//! ```
//!
//! # Limitations
//!
//! - **Types**: `u8`, `u16`, `u32`, `i32`, `f32` and `bool` only.

use core::sync::atomic::{AtomicU32, Ordering};

use crate::wire::Param;

/// A table of parameters in a suite descriptor.
pub type ParamTable = &'static [&'static ParamEntry];

/// The result of a write: success or a static description of the refusal.
pub type SetResult = Result<(), &'static str>;

/// A named, bounded, atomically stored primitive.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Accessing or writing an entry does not panic.
pub struct ParamEntry {
    description: &'static str,
    max: Param,
    min: Param,
    name: &'static str,
    bits: AtomicU32,
}

impl ParamEntry {
    /// Returns the largest accepted value.
    #[must_use]
    pub const fn max(&self) -> Param {
        self.max
    }

    /// Returns the smallest accepted value.
    #[must_use]
    pub const fn min(&self) -> Param {
        self.min
    }

    /// Returns the name the host addresses the entry by.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the description reported at discovery.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        self.description
    }

    /// Creates an entry from a value and bounds of one type.
    const fn new(
        name: &'static str,
        description: &'static str,
        init: u32,
        limits: (Param, Param),
    ) -> Self {
        Self {
            description,
            max: limits.1,
            min: limits.0,
            name,
            bits: AtomicU32::new(init),
        }
    }

    /// Creates a `bool` entry.
    #[must_use]
    pub const fn bool(
        name: &'static str,
        description: &'static str,
        init: bool,
    ) -> Self {
        Self::new(
            name,
            description,
            init as u32,
            (Param::Bool(false), Param::Bool(true)),
        )
    }

    /// Creates an `f32` entry bounded by the type limits.
    #[must_use]
    pub const fn f32(
        name: &'static str,
        description: &'static str,
        init: f32,
    ) -> Self {
        Self::new(
            name,
            description,
            init.to_bits(),
            (Param::F32(f32::MIN), Param::F32(f32::MAX)),
        )
    }

    /// Creates an `i32` entry bounded by the type limits.
    #[must_use]
    pub const fn i32(
        name: &'static str,
        description: &'static str,
        init: i32,
    ) -> Self {
        Self::new(
            name,
            description,
            init.cast_unsigned(),
            (Param::I32(i32::MIN), Param::I32(i32::MAX)),
        )
    }

    /// Creates a `u16` entry bounded by the type limits.
    #[must_use]
    pub const fn u16(
        name: &'static str,
        description: &'static str,
        init: u16,
    ) -> Self {
        Self::new(
            name,
            description,
            init as u32,
            (Param::U16(u16::MIN), Param::U16(u16::MAX)),
        )
    }

    /// Creates a `u32` entry bounded by the type limits.
    #[must_use]
    pub const fn u32(
        name: &'static str,
        description: &'static str,
        init: u32,
    ) -> Self {
        Self::new(
            name,
            description,
            init,
            (Param::U32(u32::MIN), Param::U32(u32::MAX)),
        )
    }

    /// Creates a `u8` entry bounded by the type limits.
    #[must_use]
    pub const fn u8(
        name: &'static str,
        description: &'static str,
        init: u8,
    ) -> Self {
        Self::new(
            name,
            description,
            init as u32,
            (Param::U8(u8::MIN), Param::U8(u8::MAX)),
        )
    }

    /// Narrows the bounds of an entry; `min` and `max` must have the entry's type, otherwise the
    /// bounds are left unchanged.
    #[must_use]
    pub const fn bounded(mut self, min: Param, max: Param) -> Self {
        if same_type(min, self.min) && same_type(max, self.min) {
            self.min = min;
            self.max = max;
        }
        self
    }

    /// Reads the stored value.
    #[must_use]
    pub fn get(&self) -> Param {
        let bits = self.bits.load(Ordering::Relaxed);
        match self.min {
            Param::U8(_) => Param::U8(u8::try_from(bits).unwrap_or(u8::MAX)),
            Param::U16(_) => {
                Param::U16(u16::try_from(bits).unwrap_or(u16::MAX))
            }
            Param::U32(_) => Param::U32(bits),
            Param::I32(_) => Param::I32(bits.cast_signed()),
            Param::F32(_) => Param::F32(f32::from_bits(bits)),
            Param::Bool(_) => Param::Bool(bits != 0),
        }
    }

    /// Writes a value of the entry's type inside its bounds.
    ///
    /// # Errors
    /// Returns a description if the type differs or the value is outside the bounds; the stored
    /// value is unchanged.
    pub fn set(&self, value: Param) -> SetResult {
        let bits = match value {
            Param::U8(v) => u32::from(v),
            Param::U16(v) => u32::from(v),
            Param::U32(v) => v,
            Param::I32(v) => v.cast_unsigned(),
            Param::F32(v) => v.to_bits(),
            Param::Bool(v) => u32::from(v),
        };
        if !same_type(value, self.min) {
            return Err("Type mismatch");
        }
        if !within(value, self.min, self.max) {
            return Err("Value out of bounds");
        }
        self.bits.store(bits, Ordering::Relaxed);
        Ok(())
    }
}

/// Returns whether two values have the same type.
const fn same_type(a: Param, b: Param) -> bool {
    matches!(
        (a, b),
        (Param::U8(_), Param::U8(_))
            | (Param::U16(_), Param::U16(_))
            | (Param::U32(_), Param::U32(_))
            | (Param::I32(_), Param::I32(_))
            | (Param::F32(_), Param::F32(_))
            | (Param::Bool(_), Param::Bool(_))
    )
}

/// Returns whether `value` lies in `[min, max]`; values of different types are never inside.
fn within(value: Param, min: Param, max: Param) -> bool {
    match (value, min, max) {
        (Param::U8(v), Param::U8(lo), Param::U8(hi)) => lo <= v && v <= hi,
        (Param::U16(v), Param::U16(lo), Param::U16(hi)) => lo <= v && v <= hi,
        (Param::U32(v), Param::U32(lo), Param::U32(hi)) => lo <= v && v <= hi,
        (Param::I32(v), Param::I32(lo), Param::I32(hi)) => lo <= v && v <= hi,
        (Param::F32(v), Param::F32(lo), Param::F32(hi)) => lo <= v && v <= hi,
        (Param::Bool(_), Param::Bool(_), Param::Bool(_)) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_type_round_trips_through_the_cell() {
        let entries = [
            (ParamEntry::u8("a", "", 7), Param::U8(200)),
            (ParamEntry::u16("b", "", 7), Param::U16(60_000)),
            (ParamEntry::u32("c", "", 7), Param::U32(4_000_000_000)),
            (ParamEntry::i32("d", "", -7), Param::I32(i32::MIN)),
            (ParamEntry::f32("e", "", 1.5), Param::F32(-2.25)),
            (ParamEntry::bool("f", "", false), Param::Bool(true)),
        ];
        for (entry, next) in entries {
            assert_ne!(entry.get(), next);
            assert_eq!(entry.set(next), Ok(()));
            assert_eq!(entry.get(), next);
        }
    }

    #[test]
    fn constructors_store_their_initial_value_and_type_limits() {
        let e = ParamEntry::i32("n", "d", -5);
        assert_eq!((e.name(), e.description()), ("n", "d"));
        assert_eq!(e.get(), Param::I32(-5));
        assert_eq!(
            (e.min(), e.max()),
            (Param::I32(i32::MIN), Param::I32(i32::MAX))
        );
        assert_eq!(ParamEntry::bool("b", "", true).get(), Param::Bool(true));
        assert_eq!(ParamEntry::f32("f", "", 0.5).get(), Param::F32(0.5));
    }

    #[test]
    fn a_wrong_type_or_out_of_bounds_write_changes_nothing() {
        let e = ParamEntry::u32("gain", "", 100)
            .bounded(Param::U32(10), Param::U32(500));
        assert_eq!((e.min(), e.max()), (Param::U32(10), Param::U32(500)));
        assert_eq!(e.set(Param::U8(1)), Err("Type mismatch"));
        assert_eq!(e.set(Param::U32(9)), Err("Value out of bounds"));
        assert_eq!(e.set(Param::U32(501)), Err("Value out of bounds"));
        assert_eq!(e.get(), Param::U32(100));
        assert_eq!(e.set(Param::U32(10)), Ok(()));
        assert_eq!(e.set(Param::U32(500)), Ok(()));
    }

    #[test]
    fn bounds_of_another_type_are_ignored() {
        let e =
            ParamEntry::u8("x", "", 3).bounded(Param::U32(1), Param::U32(2));
        assert_eq!((e.min(), e.max()), (Param::U8(0), Param::U8(255)));
        let f = ParamEntry::f32("y", "", 0.0)
            .bounded(Param::F32(-1.0), Param::F32(1.0));
        assert!(f.set(Param::F32(2.0)).is_err());
        assert!(f.set(Param::F32(0.5)).is_ok());
        let i =
            ParamEntry::i32("z", "", 0).bounded(Param::I32(-3), Param::I32(3));
        assert!(i.set(Param::I32(-4)).is_err());
        let s =
            ParamEntry::u16("w", "", 0).bounded(Param::U16(1), Param::U16(9));
        assert!(s.set(Param::U16(0)).is_err());
    }

    #[test]
    fn bounds_apply_only_when_both_match_the_entry_type() {
        let lo_wrong =
            ParamEntry::u8("a", "", 3).bounded(Param::U32(1), Param::U8(2));
        let hi_wrong =
            ParamEntry::u8("b", "", 3).bounded(Param::U8(1), Param::U32(2));
        for e in [lo_wrong, hi_wrong] {
            assert_eq!((e.min(), e.max()), (Param::U8(0), Param::U8(255)));
        }
    }

    #[test]
    fn each_side_of_a_u8_bound_rejects_values() {
        let e =
            ParamEntry::u8("g", "", 50).bounded(Param::U8(10), Param::U8(200));
        assert_eq!(e.set(Param::U8(5)), Err("Value out of bounds"));
        assert_eq!(e.set(Param::U8(250)), Err("Value out of bounds"));
        assert_eq!(e.set(Param::U8(10)), Ok(()));
        assert_eq!(e.set(Param::U8(200)), Ok(()));
    }

    #[test]
    fn a_table_is_addressed_by_position() {
        static A: ParamEntry = ParamEntry::u8("a", "", 1);
        static B: ParamEntry = ParamEntry::bool("b", "", false);
        static TABLE: ParamTable = &[&A, &B];
        let table = TABLE;
        assert_eq!(table.iter().position(|p| p.name() == "b"), Some(1));
        assert_eq!(table.first().map(|p| p.get()), Some(Param::U8(1)));
    }
}
