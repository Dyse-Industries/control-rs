//! Suites, executables and settings a target reports at discovery.
//!
//! # Description
//!
//! A [`Catalog`] is built from the discovery telemetry (`SuiteInfo`, `ExecutableInfo` and
//! `ParamValue`). It holds everything the host knows about a target, so discovery, setting
//! display and setting edits need no copy of the firmware image or its debug information.
//!
//! # Core Concepts
//!
//! - **Item id**: An executable is addressed by `(suite, item)`; cases and the task of a suite
//!   share one index space.
//! - **Task**: An executable that allows any mode besides `Atomic`.
//! - **Missing entries**: Each setting carries its index and the suite's count, so a dropped
//!   value message leaves a hole the session detects and refills by index.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets_host::OwnedTelemetry;
//! use control_rs_ets_host::catalog::Catalog;
//!
//! let mut catalog = Catalog::default();
//! catalog.apply(&OwnedTelemetry::SuiteInfo {
//!     description: String::new(),
//!     name: "pid".to_string(),
//!     setting_count: 0,
//!     suite_id: 0,
//!     test_count: 0,
//! });
//! assert_eq!(catalog.suites.len(), 1);
//! ```

use control_rs_ets::wire::{ExecMode, Param};

use crate::bridge::{OwnedTelemetry, PacketDecl};

/// Address of an executable: `(suite, item)`.
pub type ItemId = (u16, u16);

/// Entries by index; `None` marks an entry not yet received.
pub type Slots<T> = Vec<Option<T>>;

/// A mutable slot of [`Slots`].
type SlotMut<'a, T> = Option<&'a mut Option<T>>;

/// Target metadata from `TargetInfo`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TargetDetails {
    /// Board identifier (`0` when unknown).
    pub board_id: u16,
    /// Core clock in hertz (`0` when unknown).
    pub core_clock_hz: u32,
    /// FPU bits: 0 single, 1 double precision.
    pub fpu_flags: u8,
    /// Wire revision the target speaks.
    pub protocol_version: u8,
    /// Whether the board binding provides a release source.
    pub release_source: bool,
}

/// What the host knows about a target after discovery.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Catalog {
    /// Suites by identifier.
    pub suites: Vec<SuiteEntry>,
}

/// A suite and its members.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SuiteEntry {
    /// Description reported at discovery.
    pub description: String,
    /// Executables by item; `None` marks an entry not yet received.
    pub executables: Slots<ExecEntry>,
    /// Suite name.
    pub name: String,
    /// Settings by index; `None` marks an entry not yet received.
    pub settings: Slots<SettingEntry>,
}

/// A case or a task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecEntry {
    /// Description reported at discovery.
    pub description: String,
    /// Encoding of the input packet.
    pub input: PacketDecl,
    /// Host link timeout in milliseconds (`0` is none).
    pub link_timeout_ms: u32,
    /// Bit `n` is set when `ExecMode` variant `n` is allowed.
    pub modes: u8,
    /// Executable name.
    pub name: String,
    /// Encoding of the output packet.
    pub output: PacketDecl,
}

/// A named, typed setting with its last known value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SettingEntry {
    /// Description reported at discovery.
    pub description: String,
    /// Index within the suite.
    pub index: u16,
    /// Largest accepted value.
    pub max: Param,
    /// Smallest accepted value.
    pub min: Param,
    /// Setting name.
    pub name: String,
    /// Last value the target reported.
    pub value: Param,
}

impl ExecEntry {
    /// Whether `mode` is allowed.
    #[must_use]
    pub const fn allows(&self, mode: ExecMode) -> bool {
        self.modes & mode.bit() != 0
    }

    /// Whether this executable is a lifecycle task rather than a case.
    #[must_use]
    pub const fn is_task(&self) -> bool {
        self.modes & !ExecMode::Atomic.bit() != 0
    }
}

impl SuiteEntry {
    /// Records an `ExecutableInfo` message.
    fn apply_exec(&mut self, telemetry: &OwnedTelemetry) {
        let OwnedTelemetry::TestInfo {
            test_id,
            name,
            description,
            input,
            link_timeout_ms,
            modes,
            output,
            ..
        } = telemetry
        else {
            return;
        };
        if let Some(slot) = Catalog::slot(&mut self.executables, *test_id) {
            *slot = Some(ExecEntry {
                description: description.clone(),
                input: input.clone(),
                link_timeout_ms: *link_timeout_ms,
                modes: *modes,
                name: name.clone(),
                output: output.clone(),
            });
        }
    }

    /// Records a `ParamValue` message.
    fn apply_setting(&mut self, telemetry: &OwnedTelemetry) {
        let OwnedTelemetry::SettingInfo {
            setting_id,
            name,
            description,
            value,
            min,
            max,
            ..
        } = telemetry
        else {
            return;
        };
        if let Some(slot) = Catalog::slot(&mut self.settings, *setting_id) {
            *slot = Some(SettingEntry {
                description: description.clone(),
                index: *setting_id,
                max: *max,
                min: *min,
                name: name.clone(),
                value: *value,
            });
        }
    }
}

impl Catalog {
    /// Records one discovery message; other telemetry is ignored.
    pub fn apply(&mut self, telemetry: &OwnedTelemetry) {
        match telemetry {
            OwnedTelemetry::SuiteInfo {
                suite_id,
                name,
                description,
                setting_count,
                test_count,
            } => {
                let Some(suite) = self.suite_slot(*suite_id) else {
                    return;
                };
                suite.name.clone_from(name);
                suite.description.clone_from(description);
                suite.settings.resize(usize::from(*setting_count), None);
                suite.executables.resize(usize::from(*test_count), None);
            }
            OwnedTelemetry::TestInfo { suite_id, .. } => {
                let Some(suite) = self.suite_slot(*suite_id) else {
                    return;
                };
                suite.apply_exec(telemetry);
            }
            OwnedTelemetry::SettingInfo { suite_id, .. } => {
                let Some(suite) = self.suite_slot(*suite_id) else {
                    return;
                };
                suite.apply_setting(telemetry);
            }
            _ => {}
        }
    }

    /// The executable at `id`, if it was reported.
    #[must_use]
    pub fn exec(&self, id: ItemId) -> Option<&ExecEntry> {
        self.suites
            .get(usize::from(id.0))?
            .executables
            .get(usize::from(id.1))?
            .as_ref()
    }

    /// Ids of every case, in discovery order; tasks are not included.
    #[must_use]
    pub fn case_ids(&self) -> Vec<ItemId> {
        self.exec_ids().filter(|&id| self.is_case(id)).collect()
    }

    /// Ids of every executable, cases and tasks.
    pub fn exec_ids(&self) -> impl Iterator<Item = ItemId> + '_ {
        (0_u16..).zip(&self.suites).flat_map(|(suite, s)| {
            (0_u16..)
                .zip(&s.executables)
                .filter(|(_, e)| e.is_some())
                .map(move |(item, _)| (suite, item))
        })
    }

    /// Settings a dropped message left unreported, as `(suite, index)`.
    #[must_use]
    pub fn missing_settings(&self) -> Vec<ItemId> {
        (0_u16..)
            .zip(&self.suites)
            .flat_map(|(suite, s)| {
                (0_u16..)
                    .zip(&s.settings)
                    .filter(|(_, e)| e.is_none())
                    .map(move |(index, _)| (suite, index))
            })
            .collect()
    }

    /// Executables a dropped message left unreported, as `(suite, item)`.
    ///
    /// Unlike settings, the wire has no index refetch for `ExecutableInfo`; a
    /// non-empty result means discovery is incomplete and must fail closed.
    #[must_use]
    pub fn missing_executables(&self) -> Vec<ItemId> {
        (0_u16..)
            .zip(&self.suites)
            .flat_map(|(suite, s)| {
                (0_u16..)
                    .zip(&s.executables)
                    .filter(|(_, e)| e.is_none())
                    .map(move |(item, _)| (suite, item))
            })
            .collect()
    }

    /// The setting `name` of `suite`, if it was reported.
    #[must_use]
    pub fn setting(&self, suite: u16, name: &str) -> Option<&SettingEntry> {
        self.suites
            .get(usize::from(suite))?
            .settings
            .iter()
            .flatten()
            .find(|s| s.name == name)
    }

    fn is_case(&self, id: ItemId) -> bool {
        self.exec(id).is_some_and(|e| !e.is_task())
    }

    fn slot<T>(items: &mut Slots<T>, index: u16) -> SlotMut<'_, T> {
        let index = usize::from(index);
        if items.len() <= index {
            items.resize_with(index.saturating_add(1), || None);
        }
        items.get_mut(index)
    }

    fn suite_slot(&mut self, suite: u16) -> Option<&mut SuiteEntry> {
        let index = usize::from(suite);
        if self.suites.len() <= index {
            self.suites
                .resize_with(index.saturating_add(1), SuiteEntry::default);
        }
        self.suites.get_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn suite_info(settings: u16, tests: u16) -> OwnedTelemetry {
        OwnedTelemetry::SuiteInfo {
            description: "d".to_string(),
            name: "s".to_string(),
            setting_count: settings,
            suite_id: 0,
            test_count: tests,
        }
    }

    fn test_info(test_id: u16, modes: u8) -> OwnedTelemetry {
        OwnedTelemetry::TestInfo {
            description: String::new(),
            input: PacketDecl::default(),
            link_timeout_ms: 0,
            modes,
            name: format!("t{test_id}"),
            output: PacketDecl::default(),
            suite_id: 0,
            test_id,
        }
    }

    fn setting_info(index: u16) -> OwnedTelemetry {
        OwnedTelemetry::SettingInfo {
            count: 2,
            description: String::new(),
            max: Param::U8(9),
            min: Param::U8(0),
            name: format!("k{index}"),
            setting_id: index,
            suite_id: 0,
            value: Param::U8(1),
        }
    }

    #[test]
    fn cases_and_tasks_share_ids_and_tasks_are_excluded_from_cases() {
        let mut c = Catalog::default();
        c.apply(&suite_info(0, 2));
        c.apply(&test_info(0, ExecMode::Atomic.bit()));
        c.apply(&test_info(
            1,
            ExecMode::Lockstep.bit() | ExecMode::Atomic.bit(),
        ));
        assert_eq!(c.exec_ids().collect::<Vec<_>>(), [(0, 0), (0, 1)]);
        assert_eq!(c.case_ids(), [(0, 0)]);
        assert!(c.exec((0, 1)).is_some_and(ExecEntry::is_task));
        assert!(c.exec((0, 1)).is_some_and(|e| e.allows(ExecMode::Lockstep)));
        assert!(c.exec((3, 0)).is_none());
    }

    #[test]
    fn count_and_index_detect_missing_entry() {
        let mut c = Catalog::default();
        c.apply(&suite_info(2, 0));
        c.apply(&setting_info(1));
        assert_eq!(c.missing_settings(), [(0, 0)]);
        c.apply(&setting_info(0));
        assert_eq!(c.missing_settings(), Vec::<ItemId>::new());
        assert_eq!(c.setting(0, "k1").map(|s| s.index), Some(1));
        assert!(c.setting(0, "none").is_none());
    }

    #[test]
    fn count_and_index_detect_missing_executable() {
        let mut c = Catalog::default();
        c.apply(&suite_info(0, 2));
        c.apply(&test_info(1, ExecMode::Atomic.bit()));
        assert_eq!(c.missing_executables(), [(0, 0)]);
        assert_eq!(c.case_ids(), [(0, 1)]);
        c.apply(&test_info(0, ExecMode::Atomic.bit()));
        assert_eq!(c.missing_executables(), Vec::<ItemId>::new());
        assert_eq!(c.case_ids(), [(0, 0), (0, 1)]);
    }
}
