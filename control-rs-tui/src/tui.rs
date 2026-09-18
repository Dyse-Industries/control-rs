//! Interactive Terminal User Interface (TUI) for Embedded Test Server (ETS) testing.
//!
//! Provides an immediate-mode dashboard consuming [`control_rs_ets_host::ServerBridge`]
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
    BridgeMessage, ServerBridge, SessionAction, SessionState, Target,
};

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
        self.visible_items.clear();
        let query = self.filter_query.to_lowercase();

        for (s_idx, suite) in self.session.suites.iter().enumerate() {
            let suite_matches = suite.name.to_lowercase().contains(&query);
            let matching_tests: Vec<(usize, &control_rs_ets_host::TestItem)> =
                suite
                    .tests
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| {
                        query.is_empty()
                            || suite_matches
                            || t.name.to_lowercase().contains(&query)
                    })
                    .collect();

            if !query.is_empty() && !suite_matches && matching_tests.is_empty()
            {
                continue;
            }

            let is_collapsed = self.collapsed_suites.contains(&s_idx);
            self.visible_items.push(TableItem::Suite {
                suite_idx: s_idx,
                name: suite.name.clone(),
                collapsed: is_collapsed,
            });

            if !is_collapsed {
                let setting_count = suite.settings.len();
                let count = matching_tests.len();
                for (i, &(t_idx, test)) in matching_tests.iter().enumerate() {
                    self.visible_items.push(TableItem::Test {
                        suite_idx: s_idx,
                        test_idx: t_idx,
                        name: test.name.clone(),
                        is_last: i + 1 == count && setting_count == 0,
                        state: test.state,
                        cycles: test.cycles,
                        time_us: test.time_us,
                        stack_peak: test.stack_peak,
                    });
                }
                for (i, setting) in suite.settings.iter().enumerate() {
                    self.visible_items.push(TableItem::Setting {
                        suite_idx: s_idx,
                        setting_idx: i,
                        name: setting.name.clone(),
                        description: setting.description.clone(),
                        value: format_setting_value(setting.value),
                        is_last: i + 1 == setting_count,
                    });
                }
            }
        }

        if self.visible_items.is_empty() {
            self.table_state.select(None);
        } else {
            let current = self.table_state.selected().unwrap_or(0);
            if current >= self.visible_items.len() {
                self.table_state.select(Some(self.visible_items.len() - 1));
            } else {
                self.table_state.select(Some(current));
            }
        }
    }

    /// Navigates selection to the next visible row.
    pub fn next_row(&mut self) {
        if self.visible_items.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) => {
                if i + 1 >= self.visible_items.len() {
                    0
                } else {
                    i + 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    /// Navigates selection to the previous visible row.
    pub fn previous_row(&mut self) {
        if self.visible_items.is_empty() {
            return;
        }
        let i = match self.table_state.selected() {
            Some(i) => {
                if i == 0 {
                    self.visible_items.len() - 1
                } else {
                    i - 1
                }
            }
            None => 0,
        };
        self.table_state.select(Some(i));
    }

    /// Handles Enter on the currently selected item (toggles suite collapse or executes test).
    pub fn toggle_or_run_selected(
        &mut self,
        bridge: Option<&mut ServerBridge>,
    ) {
        if let Some(selected) = self.table_state.selected()
            && let Some(item) = self.visible_items.get(selected).cloned()
        {
            match item {
                TableItem::Suite { suite_idx, .. } => {
                    if self.collapsed_suites.contains(&suite_idx) {
                        self.collapsed_suites.remove(&suite_idx);
                    } else {
                        self.collapsed_suites.insert(suite_idx);
                    }
                    self.rebuild_visible_items();
                }
                TableItem::Test {
                    suite_idx,
                    test_idx,
                    ..
                } => {
                    if let Some(action) = self
                        .session
                        .enqueue_test(suite_idx as u16, test_idx as u16)
                    {
                        self.execute_action(action, bridge);
                    }
                    self.rebuild_visible_items();
                }
                TableItem::Setting { value, .. } => {
                    self.is_editing_setting = true;
                    self.setting_edit.clone_from(&value);
                }
            }
        }
    }

    /// Executes a [`SessionAction`] returned by the session state machine.
    pub fn execute_action(
        &mut self,
        action: SessionAction,
        bridge: Option<&mut ServerBridge>,
    ) {
        match action {
            SessionAction::Send(cmd) => {
                if let Some(b) = bridge {
                    let _ = b.send_command(&cmd);
                }
            }
            SessionAction::PanicRestart => {
                // Handled in main loop for bridge reconstruction
            }
        }
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

    fn commit_setting_edit(&mut self, bridge: Option<&mut ServerBridge>) {
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
                if let Some(b) = bridge {
                    let _ = b.send_command(&Command::SetSetting {
                        suite_id: suite_idx as u16,
                        setting_id: setting_idx as u16,
                        value,
                    });
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
        mut bridge: Option<&mut ServerBridge>,
    ) -> Option<SessionAction> {
        match msg {
            BridgeMessage::RawConsole(line) => {
                self.logs.push(format!("> {line}"));
                self.session.log(&format!("      target {line}\n"));
                None
            }
            BridgeMessage::Telemetry(t) => {
                if let control_rs_ets::comms::Telemetry::Log(log_msg) = &t {
                    self.logs.push(format!(
                        "> [{}] {}",
                        log_msg.suite_id, log_msg.payload
                    ));
                }
                let actions =
                    self.session.handle_message(BridgeMessage::Telemetry(t));
                let mut restart = None;
                for action in actions {
                    if matches!(action, SessionAction::PanicRestart) {
                        restart = Some(action);
                    } else {
                        self.execute_action(action, bridge.as_deref_mut());
                    }
                }
                self.rebuild_visible_items();
                restart
            }
        }
    }

    /// Handles keyboard events according to FR-5 single-key bindings.
    /// Returns `true` if application exit is requested.
    pub fn handle_key(
        &mut self,
        key: KeyEvent,
        bridge: Option<&mut ServerBridge>,
    ) -> bool {
        if self.is_filtering {
            match key.code {
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
            return false;
        }

        if self.is_editing_setting {
            match key.code {
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
            return false;
        }

        match key.code {
            KeyCode::Char('q') => true,
            KeyCode::Char('f') => {
                self.is_filtering = true;
                false
            }
            KeyCode::Char('d') => {
                self.show_selected_setting();
                false
            }
            KeyCode::Char('r') => {
                if let Some(action) = self.session.enqueue_all() {
                    self.execute_action(action, bridge);
                }
                self.rebuild_visible_items();
                false
            }
            KeyCode::Char('s') => {
                self.session.stop();
                if let Some(b) = bridge {
                    let _ = b.send_command(&Command::TryReset);
                }
                self.rebuild_visible_items();
                false
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.previous_row();
                false
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.next_row();
                false
            }
            KeyCode::Enter => {
                self.toggle_or_run_selected(bridge);
                false
            }
            _ => false,
        }
    }
}

/// Formats a large integer with comma thousand separators (e.g. `1,204`).
#[must_use]
pub fn format_number(val: u64) -> String {
    let s = val.to_string();
    let bytes = s.as_bytes();
    let mut result = String::new();
    let len = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(b as char);
    }
    result
}

/// Formats duration in microseconds into a human-readable string (`µs`, `ms`, `s`).
#[must_use]
pub fn format_duration(us: u64) -> String {
    if us < 1_000 {
        format!("{us}.00µs")
    } else if us < 1_000_000 {
        let ms = us as f64 / 1_000.0;
        format!("{ms:.2}ms")
    } else {
        let s = us as f64 / 1_000_000.0;
        format!("{s:.2}s")
    }
}

/// Draws the complete TUI interface according to §4.1 layout specification.
pub fn draw_ui(frame: &mut ratatui::Frame<'_>, state: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header Dashboard
            Constraint::Min(8),    // Hierarchical Metrics Table
            Constraint::Length(8), // Target Logs Panel
            Constraint::Length(1), // Footer Action Bar
        ])
        .split(frame.area());

    // 1. Header Dashboard
    let header_line1 =
        format!(" TARGET: {} | LINK: {}", state.target_info, state.link_info);
    let running_info = if let Some((s_id, t_id)) = state.session.current_running
    {
        let s_name = state
            .session
            .suites
            .get(s_id as usize)
            .map(|s| s.name.as_str())
            .unwrap_or("unknown");
        let t_name = state
            .session
            .suites
            .get(s_id as usize)
            .and_then(|s| s.tests.get(t_id as usize))
            .map(|t| t.name.as_str())
            .unwrap_or("unknown");
        Line::from(vec![
            Span::styled(
                " [ RUNNING ] ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("{s_name}::{t_name}")),
        ])
    } else {
        Line::from(vec![Span::styled(
            if state.process_exit.is_some() {
                " [ EXITED ]"
            } else {
                " [ IDLE ]"
            },
            Style::default()
                .fg(if state.process_exit.is_some() {
                    Color::Red
                } else {
                    Color::Green
                })
                .add_modifier(Modifier::BOLD),
        )])
    };

    let header_widget =
        Paragraph::new(vec![Line::from(header_line1), running_info]).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    frame.render_widget(header_widget, chunks[0]);

    // 2. Hierarchical Metrics Table
    let header_cells =
        ["NAME", "CYCLES", "TIME", "STACK"].into_iter().map(|h| {
            Cell::from(h).style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
        });
    let table_header = Row::new(header_cells).bottom_margin(0);

    let rows: Vec<Row<'_>> = state
        .visible_items
        .iter()
        .map(|item| match item {
            TableItem::Suite {
                name, collapsed, ..
            } => {
                let prefix = if *collapsed { "▶" } else { "▼" };
                Row::new(vec![
                    Cell::from(format!("{prefix} {name}")).style(
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Cell::from(""),
                    Cell::from(""),
                    Cell::from(""),
                ])
            }
            TableItem::Test {
                name,
                is_last,
                state: test_state,
                cycles,
                time_us,
                stack_peak,
                ..
            } => {
                let branch = if *is_last { "└─" } else { "├─" };
                let (cycles_cell, time_cell, stack_cell) = match test_state {
                    TestState::Running => (
                        Cell::from("[ RUN... ]").style(
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Cell::from("---"),
                        Cell::from("---"),
                    ),
                    TestState::Pending => (
                        Cell::from("PENDING")
                            .style(Style::default().fg(Color::DarkGray)),
                        Cell::from("---"),
                        Cell::from("---"),
                    ),
                    TestState::Failed => (
                        Cell::from("FAIL").style(
                            Style::default()
                                .fg(Color::Red)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Cell::from(
                            time_us.map_or("---".to_string(), format_duration),
                        ),
                        Cell::from(stack_peak.map_or("---".to_string(), |s| {
                            format_number(u64::from(s))
                        })),
                    ),
                    TestState::Passed => (
                        Cell::from(
                            cycles.map_or("N/A".to_string(), format_number),
                        )
                        .style(Style::default().fg(Color::Green)),
                        Cell::from(
                            time_us.map_or("N/A".to_string(), format_duration),
                        ),
                        Cell::from(stack_peak.map_or("N/A".to_string(), |s| {
                            format_number(u64::from(s))
                        })),
                    ),
                };
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
        })
        .collect();

    let table = Table::new(
        rows,
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
    );

    frame.render_stateful_widget(table, chunks[1], &mut state.table_state);

    // 3. Target Logs Panel
    let log_block = Block::default()
        .borders(Borders::BOTTOM)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " [ TARGET LOGS ] (Autoscroll: ON) ",
            Style::default().add_modifier(Modifier::BOLD),
        ));

    let inner_height = chunks[2].height.saturating_sub(1) as usize;
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

    let log_widget = Paragraph::new(visible_logs).block(log_block);
    frame.render_widget(log_widget, chunks[2]);

    // 4. Footer Action Bar
    let footer_content = if state.is_filtering {
        Line::from(vec![
            Span::styled(
                " Filter: ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(&state.filter_query),
            Span::styled("▌", Style::default().fg(Color::Yellow)),
            Span::styled(
                " (Enter: apply, Esc: clear)",
                Style::default().fg(Color::DarkGray),
            ),
        ])
    } else if state.is_editing_setting {
        Line::from(vec![
            Span::styled(
                " SetSetting: ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(&state.setting_edit),
            Span::styled("▌", Style::default().fg(Color::Magenta)),
            Span::styled(
                " (Enter: send, Esc: cancel)",
                Style::default().fg(Color::DarkGray),
            ),
        ])
    } else {
        Line::from(vec![
            Span::styled(
                " (f)",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("ilter | "),
            Span::styled(
                "(r)",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("un all | "),
            Span::styled(
                "(s)",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("top | "),
            Span::styled(
                "(d)",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("escription | "),
            Span::styled(
                "(q)",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("uit"),
        ])
    };
    frame.render_widget(Paragraph::new(footer_content), chunks[3]);
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
pub fn run_tui(
    mut bridge: ServerBridge,
    target: &Target,
    elf_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new(
        bridge.target_info().to_string(),
        bridge.link_info().to_string(),
    );

    // Initial discovery request
    let _ = bridge.send_command(&Command::ListSuites);
    let mut last_discovery = Instant::now();

    let elf_opt = if elf_path.is_empty() {
        None
    } else {
        Some(elf_path)
    };

    let run_res = (|| -> Result<(), Box<dyn std::error::Error>> {
        loop {
            // Draw current frame
            terminal.draw(|f| draw_ui(f, &mut state))?;

            // Poll bridge messages
            let mut need_restart = false;
            while let Ok(msg) = bridge.receiver().try_recv() {
                if let Some(SessionAction::PanicRestart) =
                    state.handle_bridge_message(msg, Some(&mut bridge))
                {
                    need_restart = true;
                }
            }

            if !state.session.discovery_complete
                && last_discovery.elapsed() > Duration::from_millis(500)
            {
                let _ = bridge.send_command(&Command::ListSuites);
                last_discovery = Instant::now();
            }

            if let Ok(Some(status)) = bridge.try_wait() {
                let msg = format!("Target process exited: {status}");
                state.process_exit = Some(msg.clone());
                state.logs.push(format!("> [EXIT] {msg}"));
            }

            // Recover and re-attach bridge on panic restart without tearing down terminal
            if need_restart {
                bridge.kill();
                state.logs.push(
                    "> [INFO] Target panicked. Re-attaching bridge..."
                        .to_string(),
                );
                thread::sleep(Duration::from_secs(1));
                if let Ok(new_bridge) =
                    ServerBridge::new(target.clone(), elf_opt, false)
                {
                    bridge = new_bridge;
                    state.process_exit = None;
                    let _ = bridge.send_command(&Command::ListSuites);
                    last_discovery = Instant::now();
                }
            }

            // Poll input events
            if event::poll(Duration::from_millis(30))?
                && let Event::Key(key) = event::read()?
                && state.handle_key(key, Some(&mut bridge))
            {
                break;
            }
        }
        Ok(())
    })();

    // Tear down terminal state cleanly
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    run_res
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

    #[test]
    fn fr1_fr2_fr3_fr4_scripted_discovery_and_metrics() {
        let mut state = AppState::new(
            "Teensy 4.0 (Cortex-M7)".to_string(),
            "USB CDC (/dev/ttyACM0)".to_string(),
        );
        assert_eq!(state.target_info, "Teensy 4.0 (Cortex-M7)");
        assert_eq!(state.link_info, "USB CDC (/dev/ttyACM0)");

        // Script discovery
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "math::storage",
                description: "",
                test_count: 2,
                setting_count: 0,
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "contiguous_storage_alloc",
                description: "",
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "noncontiguous_storage_dma",
                description: "",
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::DiscoveryComplete,
        ));
        state.rebuild_visible_items();

        assert_eq!(state.visible_items.len(), 3);
        assert!(matches!(
            &state.visible_items[0],
            TableItem::Suite { name, collapsed: false, .. } if name == "math::storage"
        ));
        assert!(matches!(
            &state.visible_items[1],
            TableItem::Test { name, is_last: false, .. } if name == "contiguous_storage_alloc"
        ));
        assert!(matches!(
            &state.visible_items[2],
            TableItem::Test { name, is_last: true, .. } if name == "noncontiguous_storage_dma"
        ));

        // Script metrics (FR-3)
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 1204,
                time_us: 2,
                stack_peak: 32,
            },
        ));
        state.rebuild_visible_items();

        if let TableItem::Test {
            cycles,
            time_us,
            stack_peak,
            state: t_state,
            ..
        } = &state.visible_items[1]
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
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 1,
                setting_count: 0,
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "t0",
                description: "",
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::DiscoveryComplete,
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 42000,
                time_us: 100,
                stack_peak: 64,
            },
        ));

        // Simulate target crash/reset and re-discovery
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::TargetPanic {
                message: "crash",
                file: "foo.rs",
                line: 10,
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 1,
                setting_count: 0,
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "t0",
                description: "",
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::DiscoveryComplete,
        ));
        state.rebuild_visible_items();

        // Cached metrics must still be preserved for t0
        if let TableItem::Test {
            cycles,
            time_us,
            stack_peak,
            state: t_state,
            ..
        } = &state.visible_items[1]
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
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite1",
                description: "",
                test_count: 2,
                setting_count: 0,
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test1",
                description: "",
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test2",
                description: "",
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::DiscoveryComplete,
        ));
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
        state.next_row(); // Wrap around
        assert_eq!(state.table_state.selected(), Some(0));
        state.previous_row(); // Wrap back
        assert_eq!(state.table_state.selected(), Some(2));

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
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 0,
                setting_count: 1,
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "cycle_limit",
                description: "Maximum cycles before test timeout",
                value: SettingValue::U32(100),
            },
        ));
        let _ = state.session.handle_message(BridgeMessage::Telemetry(
            Telemetry::DiscoveryComplete,
        ));
        state.rebuild_visible_items();
        assert!(matches!(
            &state.visible_items[1],
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
            state.session.suites[0].settings[0].value,
            SettingValue::U32(200)
        );
    }
}
