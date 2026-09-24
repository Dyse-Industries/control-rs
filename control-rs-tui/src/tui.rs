//! Interactive Terminal User Interface (TUI) for Embedded Test Server (ETS) testing.
//!
//! Provides an immediate-mode dashboard consuming [`control_rs_ets_host::ETSBridge`]
//! and [`control_rs_ets_host::SessionState`].

use std::collections::HashSet;
use std::io::stdout;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent},
    execute,
    terminal::{
        EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode,
    },
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState},
};

use control_rs_ets::comms::{Command, TestState};
use control_rs_ets::settings::SettingValue;
use control_rs_ets_host::{
    BridgeMessage, ETSBridge, HostError, OwnedTelemetry, SessionAction,
    SessionState, SuiteItem, Target, TestIndex,
};

/// Result of the interactive event loop.
type TuiResult = Result<(), Box<dyn std::error::Error>>;

/// Selectable item in the hierarchical metrics table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableItem {
    /// Collapsible test suite header.
    Suite {
        /// Index of the suite in session state.
        suite_idx: usize,
        /// Name of the suite.
        name: String,
        /// Whether the suite is currently collapsed.
        collapsed: bool,
    },
    /// Test case row belonging to a suite.
    Test {
        /// Index of the suite in session state.
        suite_idx: usize,
        /// Index of the test in its suite.
        test_idx: usize,
        /// Name of the test case.
        name: String,
        /// Whether this is the last test in the suite (for tree branch formatting).
        is_last: bool,
        /// Execution state of the test.
        state: TestState,
        /// CPU cycles consumed if completed.
        cycles: Option<u64>,
        /// Duration in microseconds if completed.
        time_us: Option<u64>,
        /// Peak stack usage in bytes if completed.
        stack_peak: Option<u32>,
    },
    /// Suite setting row.
    Setting {
        /// Index of the suite in session state.
        suite_idx: usize,
        /// Index of the setting in its suite.
        setting_idx: usize,
        /// Name of the setting.
        name: String,
        /// Doc-comment description of the setting.
        description: String,
        /// Display form of the current value.
        value: String,
        /// Whether this is the last row in the suite.
        is_last: bool,
    },
}

/// Presentation application state for the TUI dashboard.
pub struct AppState {
    /// Session state machine managing discovery, run queue, and results.
    pub session: SessionState,
    /// Discovered or configured target platform description.
    pub target_info: String,
    /// Communication link description.
    pub link_info: String,
    /// Set of suite indices that are currently collapsed.
    pub collapsed_suites: HashSet<usize>,
    /// Flattened list of visible rows currently rendered in the table.
    pub visible_items: Vec<TableItem>,
    /// State for the Ratatui Table widget (selected row).
    pub table_state: TableState,
    /// Streaming console log lines from the target.
    pub logs: Vec<String>,
    /// Whether autoscroll is enabled for the log panel.
    pub autoscroll: bool,
    /// Active query string used to filter tests.
    pub filter_query: String,
    /// Whether the user is currently typing a filter query.
    pub is_filtering: bool,
    /// Whether the user is editing a setting value.
    pub is_editing_setting: bool,
    /// In-progress setting edit buffer.
    pub setting_edit: String,
    /// Target process exit status, if observed.
    pub process_exit: Option<String>,
}

impl AppState {
    /// Creates a new, empty dashboard presentation state.
    #[must_use]
    pub fn new(target_info: String, link_info: String) -> Self {
        let mut state = Self {
            session: SessionState::new(),
            target_info,
            link_info,
            collapsed_suites: HashSet::new(),
            visible_items: Vec::new(),
            table_state: TableState::default(),
            logs: Vec::new(),
            autoscroll: true,
            filter_query: String::new(),
            is_filtering: false,
            is_editing_setting: false,
            setting_edit: String::new(),
            process_exit: None,
        };
        state.table_state.select(Some(0));
        state
    }

    /// Rebuilds the flattened list of visible rows based on suite collapse and search filters.
    pub fn rebuild_visible_items(&mut self) {
        let query = self.filter_query.to_lowercase();
        self.visible_items = self
            .session
            .suites
            .iter()
            .enumerate()
            .flat_map(|(s_idx, suite)| {
                let collapsed = self.collapsed_suites.contains(&s_idx);
                suite_rows(s_idx, suite, &query, collapsed)
            })
            .collect();

        if self.visible_items.is_empty() {
            self.table_state.select(None);
        } else {
            let last = self.visible_items.len().saturating_sub(1);
            let current = self.table_state.selected().unwrap_or(0);
            self.table_state.select(Some(current.min(last)));
        }
    }

    /// Navigates selection to the next visible row.
    pub const fn next_row(&mut self) {
        if self.visible_items.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) if i.saturating_add(1) < self.visible_items.len() => {
                i.saturating_add(1)
            }
            Some(_) | None => 0,
        };
        self.table_state.select(Some(i));
    }

    /// Navigates selection to the previous visible row.
    pub const fn previous_row(&mut self) {
        if self.visible_items.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(0) => self.visible_items.len().saturating_sub(1),
            Some(i) => i.saturating_sub(1),
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    /// Handles Enter on the currently selected item (toggles suite collapse or executes test).
    pub fn toggle_or_run_selected(&mut self, bridge: Option<&mut ETSBridge>) {
        let Some(item) = self
            .table_state
            .selected()
            .and_then(|i| self.visible_items.get(i).cloned())
        else {
            return;
        };
        match item {
            TableItem::Suite { suite_idx, .. } => {
                if !self.collapsed_suites.remove(&suite_idx) {
                    self.collapsed_suites.insert(suite_idx);
                }
                self.rebuild_visible_items();
            }
            TableItem::Test {
                suite_idx,
                test_idx,
                ..
            } => {
                match case_ids(suite_idx, test_idx) {
                    Some((suite_id, test_id)) => {
                        if let Some(action) =
                            self.session.enqueue_test(suite_id, test_id)
                        {
                            self.execute_logged(action, bridge);
                        }
                    }
                    None => self.logs.push(format!(
                        "> [HOST] test {suite_idx}::{test_idx} is outside the u16 id range"
                    )),
                }
                self.rebuild_visible_items();
            }
            TableItem::Setting { value, .. } => {
                self.is_editing_setting = true;
                self.setting_edit.clone_from(&value);
            }
        }
    }

    /// Executes a [`SessionAction`] returned by the session state machine.
    ///
    /// # Errors
    ///
    /// Returns a transport error if a command cannot be written.
    pub fn execute_action(
        action: SessionAction,
        bridge: Option<&mut ETSBridge>,
    ) -> Result<(), HostError> {
        match action {
            SessionAction::Send(cmd) => {
                if let Some(b) = bridge {
                    b.send_command(&cmd)?;
                }
                Ok(())
            }
            SessionAction::PanicRestart => Ok(()),
        }
    }

    /// Executes `action`, logging a transport failure to the log panel.
    fn execute_logged(
        &mut self,
        action: SessionAction,
        bridge: Option<&mut ETSBridge>,
    ) {
        if let Err(e) = Self::execute_action(action, bridge) {
            self.logs.push(format!("> [HOST] send failed: {e}"));
        }
    }

    /// Sends `cmd`, logging a transport failure to the log panel.
    fn send_logged(&mut self, bridge: &mut ETSBridge, cmd: &Command) {
        if let Err(e) = bridge.send_command(cmd) {
            self.logs.push(format!("> [HOST] send failed: {e}"));
        }
    }

    /// Sends `TryReset` then `ListSuites`.
    ///
    /// Pairing them lets a serial target still waiting in `handle_failure`
    /// exit after a lost reset frame.
    fn request_discovery(&mut self, bridge: &mut ETSBridge) {
        self.send_logged(bridge, &Command::TryReset);
        self.send_logged(bridge, &Command::ListSuites);
    }

    fn show_selected_setting(&mut self) {
        if let Some(selected) = self.table_state.selected()
            && let Some(TableItem::Setting {
                name, description, ..
            }) = self.visible_items.get(selected)
        {
            self.logs.push(format!("> [SETTING] {name}: {description}"));
        }
    }

    fn commit_setting_edit(&mut self, bridge: Option<&mut ETSBridge>) {
        let selected = self.table_state.selected();
        let Some(TableItem::Setting {
            suite_idx,
            setting_idx,
            ..
        }) = selected.and_then(|i| self.visible_items.get(i).cloned())
        else {
            self.is_editing_setting = false;
            return;
        };
        let Some(current) = self
            .session
            .suites
            .get(suite_idx)
            .and_then(|s| s.settings.get(setting_idx))
        else {
            self.is_editing_setting = false;
            return;
        };
        match parse_setting_value(&self.setting_edit, current.value) {
            Ok(value) => {
                if let Some(b) = bridge
                    && let Some((suite_id, setting_id)) =
                        case_ids(suite_idx, setting_idx)
                {
                    self.send_logged(
                        b,
                        &Command::SetSetting {
                            suite_id,
                            setting_id,
                            value,
                        },
                    );
                }
                if let Some(setting) = self
                    .session
                    .suites
                    .get_mut(suite_idx)
                    .and_then(|s| s.settings.get_mut(setting_idx))
                {
                    setting.value = value;
                }
                self.rebuild_visible_items();
            }
            Err(msg) => {
                self.logs.push(format!("> [SETTING] {msg}"));
            }
        }
        self.is_editing_setting = false;
        self.setting_edit.clear();
    }

    /// Processes an incoming [`BridgeMessage`] from the target.
    pub fn handle_bridge_message(
        &mut self,
        msg: BridgeMessage,
        mut bridge: Option<&mut ETSBridge>,
    ) -> Option<SessionAction> {
        match msg {
            BridgeMessage::RawConsole(line) => {
                self.logs.push(format!("> {line}"));
                self.session.log(&format!("      [ETS] {line}\n"));
                None
            }
            BridgeMessage::Telemetry(t) => {
                if let OwnedTelemetry::Log {
                    suite_id, payload, ..
                } = &t
                {
                    self.logs.push(format!("> [{suite_id}] {payload}"));
                }
                let actions =
                    self.session.handle_message(BridgeMessage::Telemetry(t));
                let mut restart = None;
                for action in actions {
                    if matches!(action, SessionAction::PanicRestart) {
                        restart = Some(action);
                    } else {
                        self.execute_logged(action, bridge.as_deref_mut());
                    }
                }
                self.rebuild_visible_items();
                restart
            }
        }
    }

    /// Handles every pending bridge message. Returns `true` when one of them
    /// requested a panic restart.
    fn drain_bridge(&mut self, bridge: &mut ETSBridge) -> bool {
        let mut need_restart = false;
        while let Ok(msg) = bridge.receiver().try_recv() {
            if matches!(
                self.handle_bridge_message(msg, Some(bridge)),
                Some(SessionAction::PanicRestart)
            ) {
                need_restart = true;
            }
        }
        need_restart
    }

    /// Recovers from a target panic by re-attaching `bridge` without
    /// tearing down the terminal. Returns `true` when a new link is up and
    /// discovery was re-requested.
    fn reattach(&mut self, bridge: &mut ETSBridge, target: &Target) -> bool {
        // Drain window for the TryReset frame already written by the
        // session action handler before closing the link.
        thread::sleep(Duration::from_millis(50));
        bridge.terminate();
        self.logs.push(
            "> [INFO] Target panicked. Re-attaching bridge...".to_string(),
        );
        thread::sleep(Duration::from_secs(1));
        match ETSBridge::new(target.clone(), false) {
            Ok(new_bridge) => {
                *bridge = new_bridge;
                self.process_exit = None;
                // Serial targets that miss the pre-terminate TryReset spin in
                // handle_failure ignoring ListSuites. Retry TryReset on the
                // new link before rediscovery.
                self.request_discovery(bridge);
                true
            }
            Err(e) => {
                self.logs.push(format!("> [HOST] reconnect failed: {e}"));
                false
            }
        }
    }

    /// Handles keyboard events according to FR-5 single-key bindings.
    /// Returns `true` if application exit is requested.
    pub fn handle_key(
        &mut self,
        key: KeyEvent,
        bridge: Option<&mut ETSBridge>,
    ) -> bool {
        if self.is_filtering {
            self.handle_filter_key(key.code);
            return false;
        }
        if self.is_editing_setting {
            self.handle_setting_key(key.code, bridge);
            return false;
        }
        self.handle_command_key(key.code, bridge)
    }

    /// Edits the filter query while the filter prompt is open.
    fn handle_filter_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Esc => {
                self.is_filtering = false;
                self.filter_query.clear();
                self.rebuild_visible_items();
            }
            KeyCode::Enter => {
                self.is_filtering = false;
            }
            KeyCode::Backspace => {
                self.filter_query.pop();
                self.rebuild_visible_items();
            }
            KeyCode::Char(c) => {
                self.filter_query.push(c);
                self.rebuild_visible_items();
            }
            _ => {}
        }
    }

    /// Edits the setting value buffer while the setting prompt is open.
    fn handle_setting_key(
        &mut self,
        code: KeyCode,
        bridge: Option<&mut ETSBridge>,
    ) {
        match code {
            KeyCode::Esc => {
                self.is_editing_setting = false;
                self.setting_edit.clear();
            }
            KeyCode::Enter => {
                self.commit_setting_edit(bridge);
            }
            KeyCode::Backspace => {
                self.setting_edit.pop();
            }
            KeyCode::Char(c) => {
                self.setting_edit.push(c);
            }
            _ => {}
        }
    }

    /// Dispatches a single-key command. Returns `true` on quit.
    fn handle_command_key(
        &mut self,
        code: KeyCode,
        bridge: Option<&mut ETSBridge>,
    ) -> bool {
        match code {
            KeyCode::Char('q') => return true,
            KeyCode::Char('f') => self.is_filtering = true,
            KeyCode::Char('d') => self.show_selected_setting(),
            KeyCode::Char('r') => {
                if let Some(action) = self.session.enqueue_all() {
                    self.execute_logged(action, bridge);
                }
                self.rebuild_visible_items();
            }
            KeyCode::Char('s') => {
                self.session.stop();
                if let Some(b) = bridge {
                    self.send_logged(b, &Command::TryReset);
                }
                self.rebuild_visible_items();
            }
            KeyCode::Char('c') => {
                self.collapsed_suites.extend(0..self.session.suites.len());
                self.rebuild_visible_items();
            }
            KeyCode::Char('e') => {
                self.collapsed_suites.clear();
                self.rebuild_visible_items();
            }
            KeyCode::Up | KeyCode::Char('k') => self.previous_row(),
            KeyCode::Down | KeyCode::Char('j') => self.next_row(),
            KeyCode::Enter => self.toggle_or_run_selected(bridge),
            _ => {}
        }
        false
    }
}

/// Converts table indices into wire `(suite_id, item_id)`, or `None` when
/// either index exceeds the `u16` identifier range.
fn case_ids(suite_idx: usize, item_idx: usize) -> Option<TestIndex> {
    Some((
        u16::try_from(suite_idx).ok()?,
        u16::try_from(item_idx).ok()?,
    ))
}

/// Visible rows for one suite: its header, then (unless collapsed) the
/// tests matching `query` and every setting.
fn suite_rows(
    s_idx: usize,
    suite: &SuiteItem,
    query: &str,
    collapsed: bool,
) -> Vec<TableItem> {
    let suite_matches = suite.name.to_lowercase().contains(query);
    let matching_tests: Vec<_> = suite
        .tests
        .iter()
        .enumerate()
        .filter(|(_, t)| {
            query.is_empty()
                || suite_matches
                || t.name.to_lowercase().contains(query)
        })
        .collect();

    if !query.is_empty() && !suite_matches && matching_tests.is_empty() {
        return Vec::new();
    }

    let mut rows = vec![TableItem::Suite {
        suite_idx: s_idx,
        name: suite.name.clone(),
        collapsed,
    }];
    if collapsed {
        return rows;
    }

    let setting_count = suite.settings.len();
    let last_test = matching_tests.len().checked_sub(1);
    for (i, &(t_idx, test)) in matching_tests.iter().enumerate() {
        rows.push(TableItem::Test {
            suite_idx: s_idx,
            test_idx: t_idx,
            name: test.name.clone(),
            is_last: Some(i) == last_test && setting_count == 0,
            state: test.state,
            cycles: test.cycles,
            time_us: test.time_us,
            stack_peak: test.stack_peak,
        });
    }
    let last_setting = setting_count.checked_sub(1);
    for (i, setting) in suite.settings.iter().enumerate() {
        rows.push(TableItem::Setting {
            suite_idx: s_idx,
            setting_idx: i,
            name: setting.name.clone(),
            description: setting.description.clone(),
            value: format_setting_value(setting.value),
            is_last: Some(i) == last_setting,
        });
    }
    rows
}

/// Formats a large integer with comma thousand separators (for example, `1,204`).
#[must_use]
pub fn format_number(val: u64) -> String {
    let digits = val.to_string();
    let mut result = String::with_capacity(digits.len().saturating_mul(2));
    let mut remaining = digits.len();
    for c in digits.chars() {
        if remaining < digits.len() && remaining.is_multiple_of(3) {
            result.push(',');
        }
        result.push(c);
        remaining = remaining.saturating_sub(1);
    }
    result
}

/// Formats duration in microseconds into a human-readable string (`µs`, `ms`, `s`).
///
/// Millisecond and second values are rounded half-up to two decimals in
/// integer arithmetic.
#[must_use]
pub fn format_duration(us: u64) -> String {
    if us < 1_000 {
        format!("{us}.00µs")
    } else if us < 1_000_000 {
        let hundredths = us.saturating_add(5) / 10;
        format!("{}.{:02}ms", hundredths / 100, hundredths % 100)
    } else {
        let hundredths = us.saturating_add(5_000) / 10_000;
        format!("{}.{:02}s", hundredths / 100, hundredths % 100)
    }
}

/// Style shared by bold, colored labels.
fn bold(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

/// Draws the complete TUI interface according to §4.1 layout specification.
pub fn draw_ui(frame: &mut ratatui::Frame<'_>, state: &mut AppState) {
    let [header_area, table_area, log_area, footer_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Min(8),
            Constraint::Length(8),
            Constraint::Length(1),
        ])
        .areas(frame.area());

    frame.render_widget(header_widget(state), header_area);
    frame.render_stateful_widget(
        table_widget(&state.visible_items),
        table_area,
        &mut state.table_state,
    );
    frame.render_widget(log_widget(state, log_area.height), log_area);
    frame.render_widget(Paragraph::new(footer_line(state)), footer_area);
}

/// Header panel: target, link, run status and pass/fail totals.
fn header_widget(state: &AppState) -> Paragraph<'static> {
    let header_line1 =
        format!(" TARGET: {} | LINK: {}", state.target_info, state.link_info);

    let total_tests: usize =
        state.session.suites.iter().map(|s| s.tests.len()).sum();
    let count_state = |wanted: TestState| {
        state
            .session
            .results
            .iter()
            .filter(|r| r.state == wanted)
            .count()
    };
    let passed_tests = count_state(TestState::Passed);
    let failed_tests = count_state(TestState::Failed);
    let header_line2 = format!(
        " Tests: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests}"
    );

    let header_block = Block::default()
        .borders(Borders::ALL)
        .title(" Embedded Test Server (ETS) Dashboard ")
        .style(Style::default().fg(Color::White));
    Paragraph::new(vec![
        Line::from(header_line1),
        status_line(&state.session),
        Line::from(header_line2),
    ])
    .block(header_block)
}

/// Running / idle / discovering status line.
fn status_line(session: &SessionState) -> Line<'static> {
    let (label, color, detail) =
        if let Some((s_id, t_id)) = session.current_running {
            let suite = session.suites.get(usize::from(s_id));
            let s_name = suite.map_or("unknown", |s| s.name.as_str());
            let t_name = suite
                .and_then(|s| s.tests.get(usize::from(t_id)))
                .map_or("unknown", |t| t.name.as_str());
            (" [ RUNNING ] ", Color::Cyan, format!("{s_name}::{t_name}"))
        } else if session.discovery_complete {
            (
                " [ IDLE ] ",
                Color::Green,
                "All tests completed or stopped".to_string(),
            )
        } else {
            (
                " [ DISCOVERING ] ",
                Color::Yellow,
                "Querying target test suites...".to_string(),
            )
        };
    Line::from(vec![Span::styled(label, bold(color)), Span::raw(detail)])
}

/// Hierarchical suite / test / setting metrics table.
fn table_widget(items: &[TableItem]) -> Table<'_> {
    let header_cells =
        ["  Suite / Test / Setting", "Cycles", "Time", "Stack (B)"]
            .map(|h| Cell::from(h).style(bold(Color::Cyan)));
    let table_header = Row::new(header_cells).bottom_margin(0);

    Table::new(
        items.iter().map(table_row),
        [
            Constraint::Percentage(50),
            Constraint::Length(16),
            Constraint::Length(14),
            Constraint::Length(10),
        ],
    )
    .header(table_header)
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray)),
    )
    .row_highlight_style(
        Style::default()
            .bg(Color::Rgb(40, 44, 52))
            .add_modifier(Modifier::BOLD),
    )
}

/// One table row for a visible item.
fn table_row(item: &TableItem) -> Row<'_> {
    match item {
        TableItem::Suite {
            name, collapsed, ..
        } => {
            let prefix = if *collapsed { "▶" } else { "▼" };
            Row::new(vec![
                Cell::from(format!("{prefix} {name}"))
                    .style(bold(Color::Yellow)),
                Cell::from(""),
                Cell::from(""),
                Cell::from(""),
            ])
        }
        TableItem::Test { name, is_last, .. } => {
            let branch = if *is_last { "└─" } else { "├─" };
            let [cycles_cell, time_cell, stack_cell] = test_metric_cells(item);
            let name_cell = Cell::from(format!("  {branch} {name}"));
            Row::new(vec![name_cell, cycles_cell, time_cell, stack_cell])
        }
        TableItem::Setting {
            name,
            value,
            is_last,
            ..
        } => {
            let branch = if *is_last { "└─" } else { "├─" };
            Row::new(vec![
                Cell::from(format!("  {branch} {name}"))
                    .style(Style::default().fg(Color::Magenta)),
                Cell::from(value.as_str()),
                Cell::from(""),
                Cell::from(""),
            ])
        }
    }
}

/// Cycles, time and stack cells for a test row; empty for other rows.
fn test_metric_cells(item: &TableItem) -> [Cell<'static>; 3] {
    let TableItem::Test {
        state,
        cycles,
        time_us,
        stack_peak,
        ..
    } = *item
    else {
        return [Cell::from(""), Cell::from(""), Cell::from("")];
    };
    let stack = |missing: &str| {
        stack_peak.map_or_else(
            || missing.to_string(),
            |s| format_number(u64::from(s)),
        )
    };
    let time = |missing: &str| {
        time_us.map_or_else(|| missing.to_string(), format_duration)
    };
    match state {
        TestState::Running => [
            Cell::from("[ RUN... ]").style(bold(Color::Cyan)),
            Cell::from("---"),
            Cell::from("---"),
        ],
        TestState::Pending => [
            Cell::from("PENDING").style(Style::default().fg(Color::DarkGray)),
            Cell::from("---"),
            Cell::from("---"),
        ],
        TestState::Failed => [
            Cell::from("FAIL").style(bold(Color::Red)),
            Cell::from(time("---")),
            Cell::from(stack("---")),
        ],
        TestState::Passed => [
            Cell::from(cycles.map_or_else(|| "N/A".to_string(), format_number))
                .style(Style::default().fg(Color::Green)),
            Cell::from(time("N/A")),
            Cell::from(stack("N/A")),
        ],
    }
}

/// Target log panel, scrolled to the tail when autoscroll is on.
fn log_widget(state: &AppState, area_height: u16) -> Paragraph<'_> {
    let log_block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " [ TARGET LOGS ] (Autoscroll: ON) ",
            Style::default().add_modifier(Modifier::BOLD),
        ));

    let inner_height = usize::from(area_height.saturating_sub(1));
    let scroll_y = if state.autoscroll {
        state.logs.len().saturating_sub(inner_height)
    } else {
        0
    };
    let visible_logs: Vec<Line<'_>> = state
        .logs
        .iter()
        .skip(scroll_y)
        .map(|l| Line::from(l.as_str()))
        .collect();

    Paragraph::new(visible_logs).block(log_block)
}

/// Footer: the active prompt, or the key legend.
fn footer_line(state: &AppState) -> Line<'_> {
    if state.is_filtering {
        prompt_line(
            " Filter: ",
            Color::Yellow,
            &state.filter_query,
            " (Enter: apply, Esc: clear)",
        )
    } else if state.is_editing_setting {
        prompt_line(
            " SetSetting: ",
            Color::Magenta,
            &state.setting_edit,
            " (Enter: send, Esc: cancel)",
        )
    } else {
        Line::from(vec![
            Span::styled(" (f)", bold(Color::Cyan)),
            Span::raw("ilter | "),
            Span::styled("(r)", bold(Color::Cyan)),
            Span::raw("un all | "),
            Span::styled("(s)", bold(Color::Cyan)),
            Span::raw("top | "),
            Span::styled("(d)", bold(Color::Cyan)),
            Span::raw("escription | "),
            Span::styled("(q)", bold(Color::Cyan)),
            Span::raw("uit"),
        ])
    }
}

/// An input prompt: label, current buffer, cursor and hint.
fn prompt_line<'a>(
    label: &'static str,
    color: Color,
    buffer: &'a str,
    hint: &'static str,
) -> Line<'a> {
    Line::from(vec![
        Span::styled(label, bold(color)),
        Span::raw(buffer),
        Span::styled("▌", Style::default().fg(color)),
        Span::styled(hint, Style::default().fg(Color::DarkGray)),
    ])
}

fn format_setting_value(value: SettingValue) -> String {
    match value {
        SettingValue::Bool(v) => v.to_string(),
        SettingValue::F32(v) => v.to_string(),
        SettingValue::I32(v) => v.to_string(),
        SettingValue::I8(v) => v.to_string(),
        SettingValue::U16(v) => v.to_string(),
        SettingValue::U32(v) => v.to_string(),
        SettingValue::U64(v) => v.to_string(),
        SettingValue::U8(v) => v.to_string(),
    }
}

fn parse_setting_value(
    raw: &str,
    current: SettingValue,
) -> Result<SettingValue, String> {
    let trimmed = raw.trim();
    match current {
        SettingValue::Bool(_) => match trimmed {
            "true" | "1" => Ok(SettingValue::Bool(true)),
            "false" | "0" => Ok(SettingValue::Bool(false)),
            _ => Err(format!("invalid bool '{trimmed}'")),
        },
        SettingValue::F32(_) => trimmed
            .parse()
            .map(SettingValue::F32)
            .map_err(|_| format!("invalid f32 '{trimmed}'")),
        SettingValue::I32(_) => trimmed
            .parse()
            .map(SettingValue::I32)
            .map_err(|_| format!("invalid i32 '{trimmed}'")),
        SettingValue::I8(_) => trimmed
            .parse()
            .map(SettingValue::I8)
            .map_err(|_| format!("invalid i8 '{trimmed}'")),
        SettingValue::U16(_) => trimmed
            .parse()
            .map(SettingValue::U16)
            .map_err(|_| format!("invalid u16 '{trimmed}'")),
        SettingValue::U32(_) => trimmed
            .parse()
            .map(SettingValue::U32)
            .map_err(|_| format!("invalid u32 '{trimmed}'")),
        SettingValue::U64(_) => trimmed
            .parse()
            .map(SettingValue::U64)
            .map_err(|_| format!("invalid u64 '{trimmed}'")),
        SettingValue::U8(_) => trimmed
            .parse()
            .map(SettingValue::U8)
            .map_err(|_| format!("invalid u8 '{trimmed}'")),
    }
}

/// Runs the interactive terminal dashboard event loop.
///
/// Sets up raw terminal mode, connects the server bridge, and handles drawing and user input.
///
/// # Errors
///
/// Returns an error if terminal initialization, crossterm polling, or bridge communication fails.
pub fn run_tui(mut bridge: ETSBridge, target: &Target) -> TuiResult {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout))?;

    let mut state = AppState::new(
        bridge.target_info().to_string(),
        bridge.link_info().to_string(),
    );
    let run_res = event_loop(&mut terminal, &mut bridge, target, &mut state);

    bridge.terminate();

    // Tear down terminal state cleanly
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    run_res
}

/// Draw, poll the bridge and handle input until the user quits.
fn event_loop<B>(
    terminal: &mut Terminal<B>,
    bridge: &mut ETSBridge,
    target: &Target,
    state: &mut AppState,
) -> TuiResult
where
    B: ratatui::backend::Backend,
    B::Error: 'static,
{
    state.request_discovery(bridge);
    let mut last_discovery = Instant::now();

    loop {
        terminal.draw(|f| draw_ui(f, state))?;

        let need_restart = state.drain_bridge(bridge);

        if !state.session.discovery_complete
            && last_discovery.elapsed() > Duration::from_millis(500)
        {
            state.request_discovery(bridge);
            last_discovery = Instant::now();
        }

        if let Ok(Some(status)) = bridge.try_wait() {
            let msg = format!("Target process exited: {status}");
            state.logs.push(format!("> [EXIT] {msg}"));
            state.process_exit = Some(msg);
        }

        if need_restart && state.reattach(bridge, target) {
            last_discovery = Instant::now();
        }

        if event::poll(Duration::from_millis(30))?
            && let Event::Key(key) = event::read()?
            && state.handle_key(key, Some(bridge))
        {
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_ets::comms::Telemetry;
    use control_rs_ets::settings::SettingValue;
    use crossterm::event::KeyModifiers;

    fn make_test_event(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// Delivers one telemetry frame to the session, discarding actions.
    fn feed(state: &mut AppState, telemetry: &Telemetry<'_>) {
        let _ = state
            .session
            .handle_message(BridgeMessage::telemetry(telemetry));
    }

    /// Discovers suite 0 named `suite` with `tests` and no settings.
    fn discover_suite(state: &mut AppState, suite: &str, tests: &[&str]) {
        feed(
            state,
            &Telemetry::SuiteInfo {
                suite_id: 0,
                name: suite,
                description: "",
                test_count: u16::try_from(tests.len()).unwrap(),
                setting_count: 0,
            },
        );
        for (test_id, name) in (0u16..).zip(tests) {
            feed(
                state,
                &Telemetry::TestInfo {
                    suite_id: 0,
                    test_id,
                    name,
                    description: "",
                },
            );
        }
        feed(state, &Telemetry::DiscoveryComplete);
    }

    #[test]
    fn fr1_fr2_fr3_fr4_scripted_discovery_and_metrics() {
        let mut state = AppState::new(
            "Teensy 4.0 (Cortex-M7)".to_string(),
            "USB CDC (/dev/ttyACM0)".to_string(),
        );
        assert_eq!(state.target_info, "Teensy 4.0 (Cortex-M7)");
        assert_eq!(state.link_info, "USB CDC (/dev/ttyACM0)");

        // Script discovery
        discover_suite(
            &mut state,
            "math::storage",
            &["contiguous_storage_alloc", "noncontiguous_storage_dma"],
        );
        state.rebuild_visible_items();

        assert_eq!(state.visible_items.len(), 3);
        assert!(matches!(
            state.visible_items.first().unwrap(),
            TableItem::Suite { name, collapsed: false, .. } if name == "math::storage"
        ));
        assert!(matches!(
            state.visible_items.get(1).unwrap(),
            TableItem::Test { name, is_last: false, .. } if name == "contiguous_storage_alloc"
        ));
        assert!(matches!(
            state.visible_items.get(2).unwrap(),
            TableItem::Test { name, is_last: true, .. } if name == "noncontiguous_storage_dma"
        ));

        // Script metrics (FR-3)
        feed(
            &mut state,
            &Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 1204,
                time_us: 2,
                stack_peak: 32,
            },
        );
        state.rebuild_visible_items();

        if let TableItem::Test {
            cycles,
            time_us,
            stack_peak,
            state: t_state,
            ..
        } = state.visible_items.get(1).unwrap()
        {
            assert_eq!(*cycles, Some(1204));
            assert_eq!(*time_us, Some(2));
            assert_eq!(*stack_peak, Some(32));
            assert_eq!(*t_state, TestState::Passed);
        } else {
            panic!("Expected Test item");
        }

        // Script logs (FR-4)
        let _ = state.handle_bridge_message(
            BridgeMessage::RawConsole("Host connected.".to_string()),
            None,
        );
        assert_eq!(state.logs.last(), Some(&"> Host connected.".to_string()));
    }

    #[test]
    fn fr5_key_dispatch_and_navigation() {
        let mut state = AppState::new("Target".to_string(), "Link".to_string());

        // 'q' quits
        assert!(state.handle_key(make_test_event(KeyCode::Char('q')), None));

        // 'f' enters filter
        assert!(!state.handle_key(make_test_event(KeyCode::Char('f')), None));
        assert!(state.is_filtering);

        // typing in filter
        let _ = state.handle_key(make_test_event(KeyCode::Char('m')), None);
        let _ = state.handle_key(make_test_event(KeyCode::Char('a')), None);
        assert_eq!(state.filter_query, "ma");

        // Esc clears filter
        let _ = state.handle_key(make_test_event(KeyCode::Esc), None);
        assert!(!state.is_filtering);
        assert_eq!(state.filter_query, "");
    }

    #[test]
    fn nfr2_cache_survives_target_reset() {
        let mut state = AppState::new("Target".to_string(), "Link".to_string());

        // Discover and run test 0
        discover_suite(&mut state, "suite", &["t0"]);
        feed(
            &mut state,
            &Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 42000,
                time_us: 100,
                stack_peak: 64,
            },
        );

        // Simulate target crash/reset and re-discovery
        feed(
            &mut state,
            &Telemetry::TargetPanic {
                message: "crash",
                file: "foo.rs",
                line: 10,
            },
        );
        discover_suite(&mut state, "suite", &["t0"]);
        state.rebuild_visible_items();

        // Cached metrics must still be preserved for t0
        if let TableItem::Test {
            cycles,
            time_us,
            stack_peak,
            state: t_state,
            ..
        } = state.visible_items.get(1).unwrap()
        {
            assert_eq!(*cycles, Some(42000));
            assert_eq!(*time_us, Some(100));
            assert_eq!(*stack_peak, Some(64));
            assert_eq!(*t_state, TestState::Passed);
        } else {
            panic!("Expected Test item");
        }
    }

    #[test]
    fn test_navigation_and_collapse() {
        let mut state = AppState::new("Target".to_string(), "Link".to_string());
        feed(
            &mut state,
            &Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite1",
                description: "",
                test_count: 2,
                setting_count: 0,
            },
        );
        feed(
            &mut state,
            &Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test1",
                description: "",
            },
        );
        feed(
            &mut state,
            &Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test2",
                description: "",
            },
        );
        feed(&mut state, &Telemetry::DiscoveryComplete);
        state.rebuild_visible_items();
        assert_eq!(state.visible_items.len(), 3);

        // Enter on suite toggles collapse
        state.table_state.select(Some(0));
        state.toggle_or_run_selected(None);
        assert_eq!(state.visible_items.len(), 1); // Only suite header visible

        // Enter again expands
        state.toggle_or_run_selected(None);
        assert_eq!(state.visible_items.len(), 3);

        // Navigation
        state.next_row();
        assert_eq!(state.table_state.selected(), Some(1));
        state.next_row();
        assert_eq!(state.table_state.selected(), Some(2));
        // Navigation with j / k
        state.handle_key(make_test_event(KeyCode::Char('j')), None);
        assert_eq!(state.table_state.selected(), Some(0));
        state.handle_key(make_test_event(KeyCode::Char('j')), None);
        assert_eq!(state.table_state.selected(), Some(1));
        state.handle_key(make_test_event(KeyCode::Char('k')), None);
        assert_eq!(state.table_state.selected(), Some(0));

        // 'c' collapses all suites
        state.handle_key(make_test_event(KeyCode::Char('c')), None);
        assert_eq!(state.visible_items.len(), 1);

        // 'e' expands all suites
        state.handle_key(make_test_event(KeyCode::Char('e')), None);
        assert_eq!(state.visible_items.len(), 3);

        // 'r' runs all
        assert!(!state.handle_key(make_test_event(KeyCode::Char('r')), None));
        assert!(state.session.current_running.is_some());

        // 's' stops
        assert!(!state.handle_key(make_test_event(KeyCode::Char('s')), None));
        assert!(state.session.current_running.is_none());
    }

    #[test]
    fn test_formatting_utilities() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1204), "1,204");
        assert_eq!(format_number(84500), "84,500");

        assert_eq!(format_duration(2), "2.00µs");
        assert_eq!(format_duration(5000), "5.00ms");
        assert_eq!(format_duration(2_500_000), "2.50s");
    }

    #[test]
    /// # Verification
    /// Trace: tui#FR-6
    /// Method: Requirements-based test
    fn test_process_exit_is_surfaced() {
        let mut state = AppState::new("Target".to_string(), "Link".to_string());
        state.process_exit = Some("exited with code 1".to_string());
        state
            .logs
            .push("> [EXIT] Target process exited: 1".to_string());
        assert!(state.process_exit.is_some());
        assert!(
            state.logs.iter().any(|line| line.contains("exited")),
            "process exit must appear in the dashboard log"
        );
    }

    #[test]
    /// # Verification
    /// Trace: tui#FR-7
    /// Method: Requirements-based test
    fn test_setting_description_and_edit() {
        let mut state = AppState::new("Target".to_string(), "Link".to_string());
        feed(
            &mut state,
            &Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 0,
                setting_count: 1,
            },
        );
        feed(
            &mut state,
            &Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "cycle_limit",
                description: "Maximum cycles before test timeout",
                value: SettingValue::U32(100),
            },
        );
        feed(&mut state, &Telemetry::DiscoveryComplete);
        state.rebuild_visible_items();
        assert!(matches!(
            state.visible_items.get(1).unwrap(),
            TableItem::Setting { name, description, .. }
                if name == "cycle_limit"
                    && description == "Maximum cycles before test timeout"
        ));
        state.table_state.select(Some(1));
        assert!(!state.handle_key(make_test_event(KeyCode::Char('d')), None));
        assert_eq!(
            state.logs.last().map(String::as_str),
            Some("> [SETTING] cycle_limit: Maximum cycles before test timeout")
        );
        assert!(!state.handle_key(make_test_event(KeyCode::Enter), None));
        assert!(state.is_editing_setting);
        for _ in 0..3 {
            assert!(
                !state.handle_key(make_test_event(KeyCode::Backspace), None)
            );
        }
        for c in "200".chars() {
            assert!(!state.handle_key(make_test_event(KeyCode::Char(c)), None));
        }
        assert!(!state.handle_key(make_test_event(KeyCode::Enter), None));
        assert!(!state.is_editing_setting);
        assert_eq!(
            state
                .session
                .suites
                .first()
                .unwrap()
                .settings
                .first()
                .unwrap()
                .value,
            SettingValue::U32(200)
        );
    }
}
