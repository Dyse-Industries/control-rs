//! Dashboard state: what the operator sees and how keys change it.
//!
//! The state is a pure function of the [`Update`]s it received and the keys it was given. It
//! owns no transport and no session: every request it wants is returned to the caller.

use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Receiver;

use control_rs_ets::wire::{ExecMode, ItemState, Param, Verdict};
use control_rs_ets_host::catalog::{
    Catalog, ItemId, SettingEntry, TargetDetails,
};
use control_rs_ets_host::session::{CaseOutcome, CaseStatus, Stats};
use control_rs_ets_host::task_run::{RunStatus, TaskRecord};
use crossterm::event::KeyEvent;
use ratatui::widgets::TableState;

use crate::keys::{Command, Mode, command};
use crate::worker::{Request, Update};

/// Default release period of a `Periodic` task: 10 ms.
const DEFAULT_PERIOD_NS: u32 = 10_000_000;

/// Log lines kept for the logs panel.
const LOG_LIMIT: usize = 1000;

/// Shortest and longest release periods the keys select: 100 µs and 1 s.
const PERIOD_RANGE_NS: (u32, u32) = (100_000, 1_000_000_000);

/// Modes the dashboard starts: those that need no plant on the host.
const TASK_MODES: [ExecMode; 3] = [
    ExecMode::FreeRunning,
    ExecMode::Periodic,
    ExecMode::Triggered,
];

/// The last decimated output sample of a task: step and bytes.
pub type Sample = Option<(u64, Vec<u8>)>;

/// Which table is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Suites with their settings.
    Settings,
    /// Suites with their cases and tasks.
    Tests,
}

/// What the dashboard believes about the target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Discovery has not finished.
    Discovering,
    /// Nothing is running.
    Idle,
    /// The link is gone.
    LinkLost(String),
    /// The target speaks another wire revision.
    Mismatch {
        /// Revision of this host.
        host: u32,
        /// Revision of the target.
        target: u32,
    },
    /// An item is running.
    Running(ItemId),
}

/// Result of a case, as the tree shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseState {
    /// The case failed, faulted or went silent.
    Failed,
    /// The case passed.
    Passed,
    /// The case is queued.
    Pending,
    /// The case is running.
    Running,
}

/// What the tree keeps per case; it survives a target reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaseRow {
    /// Statistics of the last run.
    pub stats: Option<Stats>,
    /// Result of the last run.
    pub state: CaseState,
}

/// What the tree keeps per task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskRow {
    /// Chosen mode; `None` until the first run or a mode key.
    pub mode: Option<ExecMode>,
    /// Release period for `Periodic`, in nanoseconds (`0` is the default).
    pub period_ns: u32,
    /// Record of the last run.
    pub record: Option<TaskRecord>,
    /// Last decimated output sample: step and bytes.
    pub sample: Sample,
    /// Last reported state.
    pub state: Option<ItemState>,
}

/// One visible row of the active table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Row {
    /// A case.
    Case {
        /// The case.
        id: ItemId,
        /// Whether it is the last row of its suite.
        last: bool,
        /// Case name.
        name: String,
    },
    /// A setting.
    Setting {
        /// Index within the suite.
        index: u16,
        /// Whether it is the last row of its suite.
        last: bool,
        /// Setting name.
        name: String,
        /// Suite that owns it.
        suite: u16,
    },
    /// A suite header.
    Suite {
        /// Whether the suite is folded.
        collapsed: bool,
        /// Suite name.
        name: String,
        /// Suite identifier.
        suite: u16,
    },
    /// A lifecycle task.
    Task {
        /// The task.
        id: ItemId,
        /// Whether it is the last row of its suite.
        last: bool,
        /// Task name.
        name: String,
    },
}

/// The dashboard.
pub struct App {
    /// Cached results of cases; kept across resets and rediscovery.
    pub cases: HashMap<ItemId, CaseRow>,
    /// The catalog of the last discovery.
    pub catalog: Catalog,
    /// Suites folded in the tree.
    pub collapsed: HashSet<u16>,
    /// Whether the screen must be redrawn.
    pub dirty: bool,
    /// Target metadata of the last discovery.
    pub details: Option<TargetDetails>,
    /// Text being typed in a setting prompt.
    pub edit: String,
    /// Recorded faults, oldest first.
    pub faults: Vec<String>,
    /// Active filter query.
    pub filter: String,
    /// Description of the link.
    pub link: String,
    /// Log lines in arrival order.
    pub logs: Vec<String>,
    /// What the keyboard is feeding.
    pub mode: Mode,
    /// One line of help for the selection.
    pub note: String,
    /// Whether the operator asked to leave.
    pub quit: bool,
    /// Selection of the active table.
    pub table: TableState,
    /// State of the target as far as the dashboard knows.
    pub status: Status,
    /// Description of the target.
    pub target: String,
    /// Per-task run state.
    pub tasks: HashMap<ItemId, TaskRow>,
    /// Active table.
    pub view: View,
}

impl Default for TaskRow {
    fn default() -> Self {
        Self {
            mode: None,
            period_ns: DEFAULT_PERIOD_NS,
            record: None,
            sample: None,
            state: None,
        }
    }
}

impl App {
    /// Creates an empty dashboard for `target` over `link`.
    #[must_use]
    pub fn new(target: String, link: String) -> Self {
        let mut table = TableState::default();
        table.select(Some(0));
        Self {
            cases: HashMap::new(),
            catalog: Catalog::default(),
            collapsed: HashSet::new(),
            dirty: true,
            details: None,
            edit: String::new(),
            faults: Vec::new(),
            filter: String::new(),
            link,
            logs: Vec::new(),
            mode: Mode::Normal,
            note: String::new(),
            quit: false,
            table,
            status: Status::Discovering,
            target,
            tasks: HashMap::new(),
            view: View::Tests,
        }
    }

    /// Applies every update that is ready and returns how many there were; never blocks.
    pub fn drain(&mut self, updates: &Receiver<Update>) -> usize {
        let mut count = 0_usize;
        while let Ok(update) = updates.try_recv() {
            self.on_update(update);
            count = count.saturating_add(1);
        }
        count
    }

    /// Applies one update from the worker.
    pub fn on_update(&mut self, update: Update) {
        self.dirty = true;
        match update {
            Update::Case(outcome) => self.on_case(&outcome),
            Update::CaseStarted(id) => {
                self.cases.insert(
                    id,
                    CaseRow {
                        stats: None,
                        state: CaseState::Running,
                    },
                );
                self.status = Status::Running(id);
            }
            Update::Catalog(catalog, details) => {
                self.catalog = *catalog;
                self.details = Some(details);
                self.status = Status::Idle;
                self.clamp();
            }
            Update::Fault(text) => {
                self.logs.push(format!("> [FAULT] {text}"));
                self.faults.push(text);
            }
            Update::LinkLost(text) => {
                self.logs.push(format!("> [LINK] {text}"));
                self.status = Status::LinkLost(text);
            }
            Update::Log(line) => self.logs.push(line),
            Update::Mismatch { host, target } => {
                self.status = Status::Mismatch { host, target };
            }
            Update::Idle => {
                if matches!(self.status, Status::Running(_)) {
                    self.status = Status::Idle;
                }
            }
            Update::Sample { id, k, bytes } => {
                self.tasks.entry(id).or_default().sample = Some((k, bytes));
            }
            Update::Setting(suite, entry) => self.on_setting(suite, &entry),
            Update::TaskState(id, state) => {
                self.tasks.entry(id).or_default().state = Some(state);
            }
            Update::TaskDone(id, record) => {
                self.tasks.entry(id).or_default().record = Some(*record);
            }
        }
        if self.logs.len() > LOG_LIMIT {
            let excess = self.logs.len().saturating_sub(LOG_LIMIT);
            self.logs.drain(..excess);
        }
    }

    /// Applies one key; returns the request it asks for, if any.
    pub fn on_key(&mut self, key: KeyEvent) -> Option<Request> {
        let cmd = command(key, self.mode)?;
        self.dirty = true;
        match self.mode {
            Mode::Normal => self.on_command(cmd),
            Mode::Filter => {
                self.on_filter(cmd);
                None
            }
            Mode::Edit => self.on_edit(cmd),
        }
    }

    /// The rows of the active table, after the filter and the folds.
    #[must_use]
    pub fn rows(&self) -> Vec<Row> {
        let query = self.filter.to_lowercase();
        let mut rows = Vec::new();
        for (suite, entry) in (0_u16..).zip(&self.catalog.suites) {
            let suite_hit = entry.name.to_lowercase().contains(&query);
            let members = self.members(suite, &query, suite_hit);
            if !query.is_empty() && !suite_hit && members.is_empty() {
                continue;
            }
            let collapsed = self.collapsed.contains(&suite);
            rows.push(Row::Suite {
                collapsed,
                name: entry.name.clone(),
                suite,
            });
            if !collapsed {
                rows.extend(members);
            }
        }
        rows
    }

    /// The selected row, if any.
    #[must_use]
    pub fn selected(&self) -> Option<Row> {
        self.rows().into_iter().nth(self.table.selected()?)
    }

    /// The setting the selected row shows.
    #[must_use]
    pub fn selected_setting(&self) -> Option<&SettingEntry> {
        let Some(Row::Setting { suite, index, .. }) = self.selected() else {
            return None;
        };
        self.catalog
            .suites
            .get(usize::from(suite))?
            .settings
            .get(usize::from(index))?
            .as_ref()
    }

    /// Target metadata for the header: protocol revision, board, clock and FPU class.
    #[must_use]
    pub fn target_text(&self) -> String {
        if let Status::Mismatch { host, target } = self.status {
            return format!(
                "PROTOCOL MISMATCH: host v{host}, target v{target}"
            );
        }
        self.details.map_or_else(String::new, |d| {
            let fpu = match d.fpu_flags & 0b11 {
                0 => "none",
                1 => "single",
                _ => "double",
            };
            format!(
                "Protocol v{} | Board {:#06x} | Clock {} MHz | FPU {fpu}",
                d.protocol_version,
                d.board_id,
                d.core_clock_hz / 1_000_000
            )
        })
    }

    /// Modes the dashboard offers for the task `id`.
    #[must_use]
    pub fn task_modes(&self, id: ItemId) -> Vec<ExecMode> {
        let release = self.details.is_some_and(|d| d.release_source);
        let Some(exec) = self.catalog.exec(id) else {
            return Vec::new();
        };
        TASK_MODES
            .into_iter()
            .filter(|&m| exec.allows(m))
            .filter(|&m| m != ExecMode::Triggered || release)
            .collect()
    }

    /// Mode the next run of the task `id` uses.
    #[must_use]
    pub fn task_mode(&self, id: ItemId) -> Option<ExecMode> {
        let modes = self.task_modes(id);
        self.tasks
            .get(&id)
            .and_then(|t| t.mode)
            .filter(|m| modes.contains(m))
            .or_else(|| modes.first().copied())
    }

    /// Counts of passed and failed cases.
    #[must_use]
    pub fn totals(&self) -> (usize, usize) {
        let count =
            |wanted| self.cases.values().filter(|c| c.state == wanted).count();
        (count(CaseState::Passed), count(CaseState::Failed))
    }

    fn members(&self, suite: u16, query: &str, suite_hit: bool) -> Vec<Row> {
        let Some(entry) = self.catalog.suites.get(usize::from(suite)) else {
            return Vec::new();
        };
        let hit = |name: &str| {
            query.is_empty() || suite_hit || name.to_lowercase().contains(query)
        };
        let mut rows = Vec::new();
        match self.view {
            View::Tests => {
                for (item, exec) in (0_u16..).zip(&entry.executables) {
                    let Some(exec) = exec.as_ref().filter(|e| hit(&e.name))
                    else {
                        continue;
                    };
                    let id = (suite, item);
                    let name = exec.name.clone();
                    rows.push(if exec.is_task() {
                        Row::Task {
                            id,
                            last: false,
                            name,
                        }
                    } else {
                        Row::Case {
                            id,
                            last: false,
                            name,
                        }
                    });
                }
            }
            View::Settings => {
                for (index, setting) in (0_u16..).zip(&entry.settings) {
                    let Some(setting) =
                        setting.as_ref().filter(|s| hit(&s.name))
                    else {
                        continue;
                    };
                    rows.push(Row::Setting {
                        index,
                        last: false,
                        name: setting.name.clone(),
                        suite,
                    });
                }
            }
        }
        if let Some(last) = rows.last_mut() {
            match last {
                Row::Case { last, .. }
                | Row::Setting { last, .. }
                | Row::Task { last, .. } => *last = true,
                Row::Suite { .. } => {}
            }
        }
        rows
    }

    fn on_case(&mut self, outcome: &CaseOutcome) {
        let passed = outcome.status == CaseStatus::Done(Verdict::Pass);
        self.cases.insert(
            outcome.id,
            CaseRow {
                stats: outcome.stats,
                state: if passed {
                    CaseState::Passed
                } else {
                    CaseState::Failed
                },
            },
        );
    }

    fn on_setting(&mut self, suite: u16, entry: &SettingEntry) {
        let Some(slot) = self
            .catalog
            .suites
            .get_mut(usize::from(suite))
            .and_then(|s| s.settings.get_mut(usize::from(entry.index)))
        else {
            return;
        };
        let old = slot.as_ref().map(|s| s.value);
        *slot = Some(entry.clone());
        let was = old
            .map_or_else(String::new, |v| format!("{} -> ", format_param(v)));
        self.logs.push(format!(
            "> [SETTING] {} = {was}{}",
            entry.name,
            format_param(entry.value)
        ));
    }

    fn on_command(&mut self, cmd: Command) -> Option<Request> {
        match cmd {
            Command::Quit => {
                self.quit = true;
                Some(Request::Quit)
            }
            Command::RunAll if self.idle() => {
                for id in self.catalog.case_ids() {
                    self.cases.insert(
                        id,
                        CaseRow {
                            stats: None,
                            state: CaseState::Pending,
                        },
                    );
                }
                self.faults.clear();
                Some(Request::RunAll)
            }
            Command::Stop => Some(Request::Stop),
            Command::Filter => {
                self.mode = Mode::Filter;
                None
            }
            Command::View => {
                self.view = match self.view {
                    View::Settings => View::Tests,
                    View::Tests => View::Settings,
                };
                self.table.select(Some(0));
                self.clamp();
                None
            }
            Command::Enter => self.enter(),
            other => {
                self.on_selection(other);
                None
            }
        }
    }

    /// Commands that act on the selected row.
    fn on_selection(&mut self, cmd: Command) {
        match cmd {
            Command::Up => self.step(false),
            Command::Down => self.step(true),
            Command::Toggle => self.toggle_selected_suite(),
            Command::TaskMode => self.cycle_mode(),
            Command::PeriodDown => self.scale_period(false),
            Command::PeriodUp => self.scale_period(true),
            Command::Describe => self.describe(),
            _ => {}
        }
    }

    fn on_filter(&mut self, cmd: Command) {
        match cmd {
            Command::Char(c) => self.filter.push(c),
            Command::Backspace => {
                self.filter.pop();
            }
            Command::Cancel => {
                self.filter.clear();
                self.mode = Mode::Normal;
            }
            Command::Enter => self.mode = Mode::Normal,
            _ => {}
        }
        self.table.select(Some(0));
        self.clamp();
    }

    fn on_edit(&mut self, cmd: Command) -> Option<Request> {
        match cmd {
            Command::Char(c) => self.edit.push(c),
            Command::Backspace => {
                self.edit.pop();
            }
            Command::Cancel => {
                self.edit.clear();
                self.mode = Mode::Normal;
            }
            Command::Enter => return self.commit_edit(),
            _ => {}
        }
        None
    }

    /// Parses the typed value as the setting's type and asks for one write by name.
    fn commit_edit(&mut self) -> Option<Request> {
        let (Some(Row::Setting { suite, .. }), Some(entry)) =
            (self.selected(), self.selected_setting().cloned())
        else {
            self.mode = Mode::Normal;
            return None;
        };
        match parse_param(&self.edit, entry.value) {
            Ok(value) => {
                self.edit.clear();
                self.mode = Mode::Normal;
                Some(Request::SetSetting {
                    suite,
                    name: entry.name,
                    value,
                })
            }
            Err(message) => {
                self.note = message;
                None
            }
        }
    }

    fn enter(&mut self) -> Option<Request> {
        match self.selected()? {
            Row::Suite { .. } => self.toggle_selected_suite(),
            Row::Case { id, .. } if self.idle() => {
                self.status = Status::Running(id);
                return Some(Request::RunCase(id));
            }
            Row::Task { id, .. } if self.idle() => {
                let mode = self.task_mode(id)?;
                let period_ns = self
                    .tasks
                    .get(&id)
                    .map_or(DEFAULT_PERIOD_NS, |t| t.period_ns);
                let row = self.tasks.entry(id).or_default();
                row.record = None;
                row.sample = None;
                self.status = Status::Running(id);
                return Some(Request::RunTask {
                    id,
                    mode,
                    period_ns,
                });
            }
            Row::Setting { .. } => {
                if let Some(entry) = self.selected_setting() {
                    self.edit = format_param(entry.value);
                    self.mode = Mode::Edit;
                }
            }
            Row::Case { .. } | Row::Task { .. } => {}
        }
        None
    }

    fn cycle_mode(&mut self) {
        let Some(Row::Task { id, .. }) = self.selected() else {
            return;
        };
        let modes = self.task_modes(id);
        let current = self.task_mode(id);
        let next = current
            .and_then(|c| modes.iter().position(|m| *m == c))
            .and_then(|i| {
                modes.get(i.saturating_add(1)).or_else(|| modes.first())
            })
            .copied();
        self.tasks.entry(id).or_default().mode = next;
    }

    fn scale_period(&mut self, up: bool) {
        let Some(Row::Task { id, .. }) = self.selected() else {
            return;
        };
        let row = self.tasks.entry(id).or_default();
        let (low, high) = PERIOD_RANGE_NS;
        row.period_ns = if up {
            row.period_ns.saturating_mul(2).min(high)
        } else {
            (row.period_ns / 2).max(low)
        };
    }

    fn describe(&mut self) {
        self.note = match self.selected() {
            Some(Row::Setting { .. }) => {
                self.selected_setting().map_or_else(String::new, |s| {
                    format!(
                        "{}: {} [{} .. {}]",
                        s.name,
                        s.description,
                        format_param(s.min),
                        format_param(s.max)
                    )
                })
            }
            Some(Row::Case { id, .. } | Row::Task { id, .. }) => self
                .catalog
                .exec(id)
                .map(|e| format!("{}: {}", e.name, e.description))
                .unwrap_or_default(),
            _ => String::new(),
        };
    }

    fn toggle_selected_suite(&mut self) {
        let Some(
            Row::Suite { suite, .. }
            | Row::Case { id: (suite, _), .. }
            | Row::Task { id: (suite, _), .. }
            | Row::Setting { suite, .. },
        ) = self.selected()
        else {
            return;
        };
        if !self.collapsed.remove(&suite) {
            self.collapsed.insert(suite);
        }
        self.clamp();
    }

    fn step(&mut self, down: bool) {
        let len = self.rows().len();
        if len == 0 {
            return;
        }
        let at = self.table.selected().unwrap_or(0);
        let to = if down {
            let next = at.saturating_add(1);
            if next >= len { 0 } else { next }
        } else {
            at.checked_sub(1).unwrap_or_else(|| len.saturating_sub(1))
        };
        self.table.select(Some(to));
    }

    fn clamp(&mut self) {
        let len = self.rows().len();
        if len == 0 {
            self.table.select(None);
        } else {
            let at = self.table.selected().unwrap_or(0);
            self.table.select(Some(at.min(len.saturating_sub(1))));
        }
    }

    fn idle(&self) -> bool {
        self.status == Status::Idle
    }
}

/// A value as the operator reads and types it.
#[must_use]
pub fn format_param(value: Param) -> String {
    match value {
        Param::Bool(v) => v.to_string(),
        Param::F32(v) => v.to_string(),
        Param::I32(v) => v.to_string(),
        Param::U16(v) => v.to_string(),
        Param::U32(v) => v.to_string(),
        Param::U8(v) => v.to_string(),
    }
}

/// The type of a value, for the setting console.
#[must_use]
pub const fn param_type(value: Param) -> &'static str {
    match value {
        Param::Bool(_) => "bool",
        Param::F32(_) => "f32",
        Param::I32(_) => "i32",
        Param::U16(_) => "u16",
        Param::U32(_) => "u32",
        Param::U8(_) => "u8",
    }
}

/// Parses `raw` as the type of `current`.
///
/// # Errors
/// Returns a message naming the type when `raw` does not parse.
pub fn parse_param(raw: &str, current: Param) -> Result<Param, String> {
    let text = raw.trim();
    let bad = || format!("invalid {} '{text}'", param_type(current));
    match current {
        Param::Bool(_) => match text {
            "true" | "1" => Ok(Param::Bool(true)),
            "false" | "0" => Ok(Param::Bool(false)),
            _ => Err(bad()),
        },
        Param::F32(_) => text.parse().map(Param::F32).map_err(|_| bad()),
        Param::I32(_) => text.parse().map(Param::I32).map_err(|_| bad()),
        Param::U16(_) => text.parse().map(Param::U16).map_err(|_| bad()),
        Param::U32(_) => text.parse().map(Param::U32).map_err(|_| bad()),
        Param::U8(_) => text.parse().map(Param::U8).map_err(|_| bad()),
    }
}

/// How a task run ended, for the task view.
#[must_use]
pub const fn run_status_text(status: RunStatus) -> &'static str {
    match status {
        RunStatus::Completed => "completed",
        RunStatus::Faulted => "faulted",
        RunStatus::Unresponsive => "unresponsive",
        RunStatus::UnreportedFault => "unreported fault",
    }
}

#[cfg(test)]
mod tests {
    use control_rs_ets::wire::PacketInfo;
    use control_rs_ets_host::bridge::PacketDecl;
    use control_rs_ets_host::catalog::{ExecEntry, SuiteEntry};
    use crossterm::event::{KeyCode, KeyModifiers};

    use super::*;

    fn exec(name: &str, modes: u8) -> ExecEntry {
        ExecEntry {
            description: format!("{name} description"),
            input: PacketDecl::from(PacketInfo {
                type_name: "",
                encoding: control_rs_ets::wire::PayloadEncoding::Postcard,
            }),
            link_timeout_ms: 0,
            modes,
            name: name.to_string(),
            output: PacketDecl::default(),
        }
    }

    fn setting(name: &str, value: Param) -> SettingEntry {
        SettingEntry {
            description: format!("{name} gain"),
            index: 0,
            max: Param::U8(9),
            min: Param::U8(0),
            name: name.to_string(),
            value,
        }
    }

    fn catalog() -> Catalog {
        Catalog {
            suites: vec![SuiteEntry {
                description: String::new(),
                executables: vec![
                    Some(exec("alpha", ExecMode::Atomic.bit())),
                    Some(exec("beta", ExecMode::Atomic.bit())),
                    Some(exec("loop", 0b1_1111)),
                ],
                name: "pid".to_string(),
                settings: vec![Some(setting("kp", Param::U8(3)))],
            }],
        }
    }

    fn details(release_source: bool) -> TargetDetails {
        TargetDetails {
            board_id: 0x42,
            core_clock_hz: 600_000_000,
            fpu_flags: 1,
            protocol_version: 2,
            release_source,
        }
    }

    fn ready(release_source: bool) -> App {
        let mut app = App::new("QEMU".to_string(), "pipe".to_string());
        app.on_update(Update::Catalog(
            Box::new(catalog()),
            details(release_source),
        ));
        app
    }

    fn key(app: &mut App, code: KeyCode) -> Option<Request> {
        app.on_key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn stats(cycles: u64) -> Stats {
        Stats {
            cycles,
            time_ns: 5_000,
            stack_peak: 64,
            ..Stats::default()
        }
    }

    fn passed(id: ItemId, cycles: u64) -> Update {
        Update::Case(CaseOutcome {
            id,
            message: None,
            stats: Some(stats(cycles)),
            status: CaseStatus::Done(Verdict::Pass),
        })
    }

    #[test]
    fn target_info_rendered() {
        let app = ready(false);
        assert_eq!(
            app.target_text(),
            "Protocol v2 | Board 0x0042 | Clock 600 MHz | FPU single"
        );
        let mut skewed = ready(false);
        skewed.on_update(Update::Mismatch { host: 2, target: 9 });
        assert_eq!(
            skewed.target_text(),
            "PROTOCOL MISMATCH: host v2, target v9"
        );
    }

    #[test]
    fn tree_from_discovery() {
        let app = ready(false);
        let names: Vec<_> = app
            .rows()
            .into_iter()
            .map(|r| match r {
                Row::Suite { name, .. }
                | Row::Case { name, .. }
                | Row::Task { name, .. }
                | Row::Setting { name, .. } => name,
            })
            .collect();
        assert_eq!(names, ["pid", "alpha", "beta", "loop"]);
        assert_eq!(app.status, Status::Idle);
    }

    #[test]
    fn tasks_beside_cases() {
        let mut app = ready(false);
        let rows = app.rows();
        assert!(matches!(rows.get(3), Some(Row::Task { last: true, .. })));
        // Enter on a case runs it; Run all never names the task.
        key(&mut app, KeyCode::Down);
        assert_eq!(
            key(&mut app, KeyCode::Enter),
            Some(Request::RunCase((0, 0)))
        );
        app.on_update(Update::Idle);
        assert_eq!(key(&mut app, KeyCode::Char('r')), Some(Request::RunAll));
        assert_eq!(app.cases.len(), 2, "the task has no case row");
        app.on_update(Update::Idle);
        // Enter on the task starts it in the first mode the host can drive.
        key(&mut app, KeyCode::Down);
        key(&mut app, KeyCode::Down);
        assert_eq!(
            key(&mut app, KeyCode::Enter),
            Some(Request::RunTask {
                id: (0, 2),
                mode: ExecMode::FreeRunning,
                period_ns: DEFAULT_PERIOD_NS,
            })
        );
    }

    #[test]
    fn telemetry_table_rows() {
        let mut app = ready(false);
        app.on_update(passed((0, 0), 1_000));
        let row = app.cases.get(&(0, 0)).unwrap();
        assert_eq!(row.state, CaseState::Passed);
        assert_eq!(
            row.stats.map(|s| (s.cycles, s.time_ns)),
            Some((1_000, 5_000))
        );
        assert_eq!(app.totals(), (1, 0));
    }

    #[test]
    fn log_buffer_order() {
        let mut app = ready(false);
        for line in ["one", "two", "three"] {
            app.on_update(Update::Log(line.to_string()));
        }
        assert_eq!(app.logs, ["one", "two", "three"]);
        for n in 0..LOG_LIMIT {
            app.on_update(Update::Log(n.to_string()));
        }
        assert_eq!(app.logs.len(), LOG_LIMIT);
        assert_eq!(app.logs.last().map(String::as_str), Some("999"));
    }

    #[test]
    fn setting_console_edit() {
        let mut app = ready(false);
        key(&mut app, KeyCode::Char('p'));
        assert_eq!(app.view, View::Settings);
        key(&mut app, KeyCode::Down);
        key(&mut app, KeyCode::Char('d'));
        assert_eq!(app.note, "kp: kp gain [0 .. 9]");
        assert_eq!(key(&mut app, KeyCode::Enter), None);
        assert_eq!(app.mode, Mode::Edit);
        assert_eq!(app.edit, "3");
        key(&mut app, KeyCode::Backspace);
        key(&mut app, KeyCode::Char('7'));
        assert_eq!(
            key(&mut app, KeyCode::Enter),
            Some(Request::SetSetting {
                suite: 0,
                name: "kp".to_string(),
                value: Param::U8(7),
            })
        );
        assert_eq!(app.mode, Mode::Normal);
        // The echo replaces the value and is logged.
        let entry = SettingEntry {
            value: Param::U8(7),
            ..app.selected_setting().unwrap().clone()
        };
        app.on_update(Update::Setting(0, Box::new(entry)));
        assert_eq!(
            app.logs.last().map(String::as_str),
            Some("> [SETTING] kp = 3 -> 7")
        );
    }

    #[test]
    fn a_value_of_the_wrong_type_is_refused_in_the_prompt() {
        let mut app = ready(false);
        key(&mut app, KeyCode::Char('p'));
        key(&mut app, KeyCode::Down);
        key(&mut app, KeyCode::Enter);
        key(&mut app, KeyCode::Char('x'));
        assert_eq!(key(&mut app, KeyCode::Enter), None);
        assert_eq!(app.mode, Mode::Edit);
        assert_eq!(app.note, "invalid u8 '3x'");
        key(&mut app, KeyCode::Esc);
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn param_parsing_follows_the_type() {
        assert_eq!(
            parse_param("yes", Param::Bool(false)),
            Err("invalid bool 'yes'".to_string())
        );
        assert_eq!(parse_param("1", Param::Bool(false)), Ok(Param::Bool(true)));
        assert_eq!(parse_param(" 2.5 ", Param::F32(0.0)), Ok(Param::F32(2.5)));
        assert_eq!(parse_param("-4", Param::I32(0)), Ok(Param::I32(-4)));
        assert!(parse_param("-4", Param::U32(0)).is_err());
        assert_eq!(format_param(Param::U16(9)), "9");
        assert_eq!(param_type(Param::F32(0.0)), "f32");
    }

    #[test]
    fn fault_and_link_status() {
        let mut app = ready(false);
        app.on_update(Update::Fault("panic at f.rs:3: boom".to_string()));
        assert_eq!(app.faults.len(), 1);
        app.on_update(Update::LinkLost("target exited".to_string()));
        assert_eq!(app.status, Status::LinkLost("target exited".to_string()));
        assert!(app.logs.iter().any(|l| l.contains("[FAULT]")));
        assert_eq!(
            key(&mut app, KeyCode::Char('r')),
            None,
            "no runs on a dead link"
        );
    }

    #[test]
    fn triggered_hidden_without_release_source() {
        let without = ready(false);
        assert_eq!(
            without.task_modes((0, 2)),
            [ExecMode::FreeRunning, ExecMode::Periodic]
        );
        let mut with = ready(true);
        assert_eq!(with.task_modes((0, 2)).last(), Some(&ExecMode::Triggered));
        // The mode key cycles through what is offered; the period keys scale it.
        for _ in 0..3 {
            key(&mut with, KeyCode::Down);
        }
        key(&mut with, KeyCode::Char('m'));
        assert_eq!(with.task_mode((0, 2)), Some(ExecMode::Periodic));
        key(&mut with, KeyCode::Char(']'));
        key(&mut with, KeyCode::Char(']'));
        key(&mut with, KeyCode::Char('['));
        assert_eq!(
            with.tasks.get(&(0, 2)).map(|t| t.period_ns),
            Some(DEFAULT_PERIOD_NS * 2)
        );
        let record = TaskRecord {
            missed_releases: 2,
            jitter_max_ns: 500,
            ..TaskRecord::default()
        };
        with.on_update(Update::TaskDone((0, 2), Box::new(record)));
        assert_eq!(
            with.tasks
                .get(&(0, 2))
                .and_then(|t| t.record.as_ref())
                .map(|r| r.missed_releases),
            Some(2)
        );
        assert_eq!(run_status_text(RunStatus::Unresponsive), "unresponsive");
    }

    #[test]
    fn reset_preserves_cached_metrics() {
        let mut app = ready(false);
        app.on_update(passed((0, 0), 1_000));
        let before = app.cases.clone();
        // A reset rediscovers the target and publishes the catalog again.
        app.on_update(Update::Catalog(Box::new(catalog()), details(false)));
        assert_eq!(app.cases, before);
    }

    #[test]
    fn poll_does_not_block() {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = ready(false);
        let started = std::time::Instant::now();
        assert_eq!(app.drain(&rx), 0);
        assert!(started.elapsed() < std::time::Duration::from_millis(20));
        tx.send(Update::Log("x".to_string())).unwrap();
        tx.send(Update::Idle).unwrap();
        assert_eq!(app.drain(&rx), 2);
    }

    #[test]
    fn the_filter_and_the_folds_shape_the_rows() {
        let mut app = ready(false);
        key(&mut app, KeyCode::Char('f'));
        for c in "bet".chars() {
            key(&mut app, KeyCode::Char(c));
        }
        key(&mut app, KeyCode::Enter);
        assert_eq!(app.rows().len(), 2, "the suite header and the match");
        key(&mut app, KeyCode::Char('f'));
        key(&mut app, KeyCode::Esc);
        assert_eq!(app.rows().len(), 4);
        key(&mut app, KeyCode::Char(' '));
        assert_eq!(app.rows().len(), 1, "a folded suite shows only its header");
        key(&mut app, KeyCode::Char(' '));
        assert_eq!(app.rows().len(), 4);
        assert_eq!(key(&mut app, KeyCode::Char('s')), Some(Request::Stop));
        assert_eq!(key(&mut app, KeyCode::Char('q')), Some(Request::Quit));
        assert!(app.quit);
    }

    #[test]
    fn a_running_item_blocks_new_runs_and_idle_releases_them() {
        let mut app = ready(false);
        app.on_update(Update::CaseStarted((0, 0)));
        assert_eq!(app.status, Status::Running((0, 0)));
        assert_eq!(key(&mut app, KeyCode::Char('r')), None);
        app.on_update(Update::Idle);
        assert_eq!(app.status, Status::Idle);
    }

    #[test]
    fn samples_and_states_reach_the_task_row() {
        let mut app = ready(false);
        app.on_update(Update::Sample {
            id: (0, 2),
            k: 9,
            bytes: vec![1, 2],
        });
        app.on_update(Update::TaskState((0, 2), ItemState::Running));
        let row = app.tasks.get(&(0, 2)).unwrap();
        assert_eq!(row.sample, Some((9, vec![1, 2])));
        assert_eq!(row.state, Some(ItemState::Running));
    }
}
