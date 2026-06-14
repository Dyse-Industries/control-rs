//! Interactive Terminal User Interface (TUI) for hardware-in-the-loop (HIL) testing.
//! Allows running, stopping, and filtering tests, as well as modifying settings dynamically.
//!
//!
//! TODO:
//!   - When the user enters (r)un, the runner should begin with the current
//!     selection. This will allow users to "skip" failing tests.

use std::io::stdout;
use std::time::Duration;

use crossterm::{
    event::{self, Event, KeyCode},
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
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use crate::bridge::{BridgeMessage, QemuBridge, Target};
use control_rs_hil::comms::{Command, Telemetry, TestState};
use control_rs_hil::settings::SettingValue;

/// Represents a test item in the TUI list.
struct TestItem {
    /// Name of the test case.
    name: String,
    /// Description of the test case.
    description: String,
    /// Current execution state of the test.
    state: TestState,
    /// Cycles consumed by the test if it completed.
    cycles: Option<u64>,
    /// Duration of the test in microseconds if it completed.
    time_us: Option<u64>,
    /// Stack high-water mark in bytes if it completed.
    stack_peak: Option<u32>,
}

/// Represents a configuration setting item in the TUI list.
struct SettingItem {
    /// Name of the setting.
    name: String,
    /// Description of the setting.
    description: String,
    /// Value of the setting.
    value: SettingValue,
}

/// Represents a test suite containing tests and settings.
struct SuiteItem {
    /// Name of the suite.
    name: String,
    /// Description of the suite.
    description: String,
    /// Tests inside this suite.
    tests: Vec<TestItem>,
    /// Config settings inside this suite.
    settings: Vec<SettingItem>,
    /// Whether the suite is collapsed in the TUI view.
    collapsed: bool,
}

/// State of the interactive TUI application.
struct AppState {
    /// List of test suites.
    suites: Vec<SuiteItem>,
    /// Buffer of console logs displayed in the log panel.
    console_logs: Vec<String>,
    /// Index in the flattened list of tests/settings for UI navigation.
    selected_item_idx: usize,
    /// Queue of test indices scheduled to run.
    run_queue: Vec<(u16, u16)>,
    /// Index of the test currently being executed.
    current_running: Option<(u16, u16)>,
    /// Active query string used to filter tests.
    filter_query: String,
    /// Whether the user is currently typing a filter.
    is_filtering: bool,
    /// Information string about the target platform.
    target_info: String,
    /// Information string about the communication link.
    link_info: String,
    /// Whether initial test discovery has finished.
    discovery_complete: bool,
    /// Whether the user is currently editing a setting value.
    editing_setting: Option<(u16, u16)>,
    /// Input buffer for the setting value.
    setting_input: String,
    /// State for the stateful test/settings list widget.
    list_state: ListState,
    /// Whether the selected item's description details modal is open.
    show_details_modal: bool,
}

impl AppState {
    /// Creates a new `AppState` with empty suites and default values.
    fn new() -> Self {
        Self {
            suites: Vec::new(),
            console_logs: Vec::new(),
            selected_item_idx: 0,
            run_queue: Vec::new(),
            current_running: None,
            filter_query: String::new(),
            is_filtering: false,
            target_info: "QEMU Cortex-M7 (MPS2-AN500)".to_string(),
            link_info: "Semihosting (Interactive)".to_string(),
            discovery_complete: false,
            editing_setting: None,
            setting_input: String::new(),
            list_state: ListState::default(),
            show_details_modal: false,
        }
    }

    /// Adds a new log entry to the console log buffer, enforcing a maximum limit.
    fn add_log(&mut self, log: String) {
        self.console_logs.push(log);
        if self.console_logs.len() > 100 {
            self.console_logs.remove(0);
        }
    }

    /// Returns a flattened list of items for UI rendering and navigation
    /// (Suite headers, Tests, and Settings).
    fn get_flat_items(&self) -> Vec<FlatItem<'_>> {
        let mut flat = Vec::new();
        for (s_idx, suite) in self.suites.iter().enumerate() {
            if self.is_filtering
                && !suite
                    .name
                    .to_lowercase()
                    .contains(&self.filter_query.to_lowercase())
            {
                // If suite name doesn't match and no tests match, skip.
                let any_test_matches = suite.tests.iter().any(|t| {
                    t.name
                        .to_lowercase()
                        .contains(&self.filter_query.to_lowercase())
                });
                if !any_test_matches {
                    continue;
                }
            }

            flat.push(FlatItem::SuiteHeader {
                suite_id: s_idx as u16,
                name: suite.name.clone(),
                collapsed: suite.collapsed,
            });

            if suite.collapsed && !self.is_filtering {
                continue;
            }

            for (t_idx, test) in suite.tests.iter().enumerate() {
                if self.is_filtering
                    && !test
                        .name
                        .to_lowercase()
                        .contains(&self.filter_query.to_lowercase())
                {
                    continue;
                }
                flat.push(FlatItem::Test {
                    suite_id: s_idx as u16,
                    test_id: t_idx as u16,
                    item: test,
                });
            }

            for (set_idx, setting) in suite.settings.iter().enumerate() {
                if self.is_filtering {
                    continue; // Skip settings when filtering tests
                }
                flat.push(FlatItem::Setting {
                    suite_id: s_idx as u16,
                    setting_id: set_idx as u16,
                    item: setting,
                });
            }
        }
        flat
    }

    /// Handles incoming telemetry messages from the target and updates the application state.
    fn handle_telemetry(
        &mut self,
        telemetry: Telemetry<'static>,
        cmd_tx: &mut Vec<Command>,
    ) {
        match telemetry {
            Telemetry::SuiteInfo {
                suite_id,
                name,
                description,
                ..
            } => {
                let id = suite_id as usize;
                while self.suites.len() <= id {
                    self.suites.push(SuiteItem {
                        name: String::new(),
                        description: String::new(),
                        tests: Vec::new(),
                        settings: Vec::new(),
                        collapsed: false,
                    });
                }
                self.suites[id].name = name.to_string();
                self.suites[id].description = description.to_string();
            }
            Telemetry::TestInfo {
                suite_id,
                test_id,
                name,
                description,
            } => {
                let s_id = suite_id as usize;
                let t_id = test_id as usize;
                while self.suites[s_id].tests.len() <= t_id {
                    self.suites[s_id].tests.push(TestItem {
                        name: String::new(),
                        description: String::new(),
                        state: TestState::Pending,
                        cycles: None,
                        time_us: None,
                        stack_peak: None,
                    });
                }
                self.suites[s_id].tests[t_id].name = name.to_string();
                self.suites[s_id].tests[t_id].description =
                    description.to_string();
            }
            Telemetry::SettingInfo {
                suite_id,
                setting_id,
                name,
                description,
                value,
            } => {
                let s_id = suite_id as usize;
                let set_id = setting_id as usize;
                while self.suites[s_id].settings.len() <= set_id {
                    self.suites[s_id].settings.push(SettingItem {
                        name: String::new(),
                        description: String::new(),
                        value: SettingValue::U8(0),
                    });
                }
                self.suites[s_id].settings[set_id].name = name.to_string();
                self.suites[s_id].settings[set_id].description =
                    description.to_string();
                self.suites[s_id].settings[set_id].value = value;
            }
            Telemetry::DiscoveryComplete => {
                self.add_log("[Host] Discovery complete.".to_string());
                self.discovery_complete = true;
            }
            Telemetry::TestStateChange {
                suite_id,
                test_id,
                state: new_state,
            } => {
                let s_id = suite_id as usize;
                let t_id = test_id as usize;
                self.suites[s_id].tests[t_id].state = new_state;

                if new_state == TestState::Running {
                    self.current_running = Some((suite_id, test_id));
                } else {
                    self.current_running = None;
                    // Trigger next test in queue if any
                    if !self.run_queue.is_empty() {
                        let (next_s, next_t) = self.run_queue.remove(0);
                        cmd_tx.push(Command::RunExecutable {
                            suite_id: next_s,
                            test_id: next_t,
                        });
                    }
                }
            }
            Telemetry::MetricReport {
                suite_id,
                test_id,
                cycles,
                time_us,
                stack_peak,
            } => {
                let s_id = suite_id as usize;
                let t_id = test_id as usize;
                self.suites[s_id].tests[t_id].cycles = Some(cycles);
                self.suites[s_id].tests[t_id].time_us = Some(time_us);
                self.suites[s_id].tests[t_id].stack_peak = Some(stack_peak);
                self.add_log(format!(
                    "[PASS] {}::{} ({} cycles, {}us, {}B stk)",
                    self.suites[s_id].name,
                    self.suites[s_id].tests[t_id].name,
                    cycles,
                    time_us,
                    stack_peak
                ));
            }
            Telemetry::Log(log) => {
                self.add_log(format!("[LOG] {}", log.payload));
            }
            Telemetry::TargetPanic {
                message,
                file,
                line,
            } => {
                self.add_log(format!(
                    "[PANIC] Target crashed: '{}' at {}:{}",
                    message, file, line
                ));
                if let Some((s_id, t_id)) = self.current_running {
                    let s_idx = s_id as usize;
                    let t_idx = t_id as usize;
                    if s_idx < self.suites.len()
                        && t_idx < self.suites[s_idx].tests.len()
                    {
                        self.suites[s_idx].tests[t_idx].state =
                            TestState::Failed;
                    }
                }
                self.current_running = None;
                self.run_queue.clear();
            }
        }
    }

    /// Handles the target process exit signal, logging the status and clearing the execution queues.
    fn handle_target_exit(&mut self, status: std::process::ExitStatus) {
        if self.current_running.is_some() || !self.run_queue.is_empty() {
            self.add_log(format!(
                "[Host] Target process exited unexpectedly: {}",
                status
            ));
        } else {
            self.add_log(format!("[Host] Target process exited: {}", status));
        }
        self.current_running = None;
        self.run_queue.clear();
    }

    /// Processes keyboard inputs and returns whether the application should exit.
    fn handle_key(
        &mut self,
        key_code: KeyCode,
        cmd_tx: &mut Vec<Command>,
    ) -> bool {
        if self.show_details_modal {
            self.show_details_modal = false;
            return false;
        }

        let old_selected_id = {
            let flat_items = self.get_flat_items();
            flat_items.get(self.selected_item_idx).map(|item| item.id())
        };

        let mut exit_tui = false;
        if self.is_filtering {
            match key_code {
                KeyCode::Char(c) => {
                    self.filter_query.push(c);
                    self.selected_item_idx = 0;
                }
                KeyCode::Backspace => {
                    self.filter_query.pop();
                    self.selected_item_idx = 0;
                }
                KeyCode::Esc | KeyCode::Enter => {
                    self.is_filtering = false;
                }
                _ => {}
            }
        } else if let Some((suite_id, setting_id)) = self.editing_setting {
            match key_code {
                KeyCode::Char(c) => {
                    if c.is_ascii_digit() {
                        self.setting_input.push(c);
                    }
                }
                KeyCode::Backspace => {
                    self.setting_input.pop();
                }
                KeyCode::Esc => {
                    self.editing_setting = None;
                    self.setting_input.clear();
                }
                KeyCode::Enter => {
                    let s_idx = suite_id as usize;
                    let set_idx = setting_id as usize;
                    if s_idx < self.suites.len()
                        && set_idx < self.suites[s_idx].settings.len()
                    {
                        let setting_item =
                            &self.suites[s_idx].settings[set_idx];
                        let parsed = match setting_item.value {
                            SettingValue::U32(_) => self
                                .setting_input
                                .parse::<u32>()
                                .map(SettingValue::U32),
                            SettingValue::U8(_) => self
                                .setting_input
                                .parse::<u8>()
                                .map(SettingValue::U8),
                        };
                        match parsed {
                            Ok(val) => {
                                cmd_tx.push(Command::SetSetting {
                                    suite_id,
                                    setting_id,
                                    value: val,
                                });
                            }
                            Err(_) => {
                                self.add_log(format!(
                                    "[Host] Failed to parse value '{}' for setting '{}'",
                                    self.setting_input, setting_item.name
                                ));
                            }
                        }
                    }
                    self.editing_setting = None;
                    self.setting_input.clear();
                }
                _ => {}
            }
        } else {
            match key_code {
                KeyCode::Char('q') => {
                    exit_tui = true;
                }
                KeyCode::Char('d') => {
                    self.show_details_modal = true;
                }
                KeyCode::Char('r') => {
                    // Run tests starting from current selection (or all if selection doesn't precede a test)
                    let flat_items = self.get_flat_items();
                    let start_from = flat_items
                        .iter()
                        .skip(self.selected_item_idx)
                        .find_map(|item| match item {
                            FlatItem::Test {
                                suite_id, test_id, ..
                            } => Some((*suite_id, *test_id)),
                            _ => None,
                        });

                    drop(flat_items);

                    self.run_queue.clear();

                    // Create the flat iterator and push ALL tests into the queue in normal order
                    let all_tests = self.suites.iter().enumerate().flat_map(
                        |(s_idx, suite)| {
                            (0..suite.tests.len())
                                .map(move |t_idx| (s_idx as u16, t_idx as u16))
                        },
                    );

                    self.run_queue.extend(all_tests);

                    // Find the starting test and rotate the queue
                    if let Some(start) = start_from {
                        if let Some(pos) = self
                            .run_queue
                            .iter()
                            .position(|&curr| curr == start)
                        {
                            // `rotate_left` shifts everything left by `pos` indices.
                            // The `pos` elements that fall off the front are moved to the back!
                            self.run_queue.rotate_left(pos);
                        }
                    }

                    // Kick off the next test if the runner is idle
                    if self.current_running.is_none()
                        && !self.run_queue.is_empty()
                    {
                        let (next_s, next_t) = self.run_queue.remove(0);

                        cmd_tx.push(Command::RunExecutable {
                            suite_id: next_s,
                            test_id: next_t,
                        });
                    }
                }
                KeyCode::Char('s') => {
                    self.run_queue.clear();
                    self.current_running = None;
                    self.add_log("[Host] Stopped execution queue.".to_string());
                }
                KeyCode::Char('f') => {
                    self.is_filtering = true;
                    self.filter_query.clear();
                }
                KeyCode::Up => {
                    let flat_len = self.get_flat_items().len();
                    if flat_len > 0 {
                        if self.selected_item_idx == 0 {
                            self.selected_item_idx = flat_len - 1;
                        } else {
                            self.selected_item_idx -= 1;
                        }
                    }
                }
                KeyCode::Down => {
                    let flat_len = self.get_flat_items().len();
                    if flat_len > 0 {
                        self.selected_item_idx =
                            (self.selected_item_idx + 1) % flat_len;
                    }
                }
                KeyCode::Enter => {
                    let flat_items = self.get_flat_items();
                    let matched_item = flat_items
                        .get(self.selected_item_idx)
                        .map(|item| match item {
                            FlatItem::Test {
                                suite_id, test_id, ..
                            } => (Some((*suite_id, *test_id)), None, None),
                            FlatItem::Setting {
                                suite_id,
                                setting_id,
                                item,
                            } => (
                                None,
                                Some((*suite_id, *setting_id, item.value)),
                                None,
                            ),
                            FlatItem::SuiteHeader { suite_id, .. } => {
                                (None, None, Some(*suite_id))
                            }
                        });

                    drop(flat_items);

                    if let Some((run_test, edit_setting, toggle_suite)) =
                        matched_item
                    {
                        if let Some((suite_id, test_id)) = run_test {
                            if self.current_running.is_none() {
                                cmd_tx.push(Command::RunExecutable {
                                    suite_id,
                                    test_id,
                                });
                            }
                        } else if let Some((suite_id, setting_id, value)) =
                            edit_setting
                        {
                            self.editing_setting = Some((suite_id, setting_id));
                            self.setting_input = match value {
                                SettingValue::U32(v) => v.to_string(),
                                SettingValue::U8(v) => v.to_string(),
                            };
                        } else if let Some(suite_id) = toggle_suite {
                            let s_idx = suite_id as usize;
                            if s_idx < self.suites.len() {
                                self.suites[s_idx].collapsed =
                                    !self.suites[s_idx].collapsed;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        let is_navigation = matches!(key_code, KeyCode::Up | KeyCode::Down);

        if !is_navigation {
            if let Some(id) = old_selected_id {
                let new_flat_items = self.get_flat_items();
                if let Some(pos) =
                    new_flat_items.iter().position(|item| item.id() == id)
                {
                    self.selected_item_idx = pos;
                } else {
                    // Fallback: if it's a test/setting inside a collapsed suite, select the suite header
                    let fallback_id = match id {
                        ItemId::Test { suite_id, .. }
                        | ItemId::Setting { suite_id, .. } => {
                            Some(ItemId::SuiteHeader { suite_id })
                        }
                        _ => None,
                    };
                    if let Some(f_id) = fallback_id {
                        if let Some(pos) = new_flat_items
                            .iter()
                            .position(|item| item.id() == f_id)
                        {
                            self.selected_item_idx = pos;
                        } else {
                            self.selected_item_idx = self
                                .selected_item_idx
                                .min(new_flat_items.len().saturating_sub(1));
                        }
                    } else {
                        self.selected_item_idx = self
                            .selected_item_idx
                            .min(new_flat_items.len().saturating_sub(1));
                    }
                }
            } else {
                let flat_len = self.get_flat_items().len();
                if flat_len > 0 && self.selected_item_idx >= flat_len {
                    self.selected_item_idx = flat_len - 1;
                }
            }
        } else {
            let flat_len = self.get_flat_items().len();
            if flat_len > 0 && self.selected_item_idx >= flat_len {
                self.selected_item_idx = flat_len - 1;
            }
        }

        exit_tui
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemId {
    SuiteHeader { suite_id: u16 },
    Test { suite_id: u16, test_id: u16 },
    Setting { suite_id: u16, setting_id: u16 },
}

impl FlatItem<'_> {
    fn id(&self) -> ItemId {
        match *self {
            FlatItem::SuiteHeader { suite_id, .. } => {
                ItemId::SuiteHeader { suite_id }
            }
            FlatItem::Test {
                suite_id, test_id, ..
            } => ItemId::Test { suite_id, test_id },
            FlatItem::Setting {
                suite_id,
                setting_id,
                ..
            } => ItemId::Setting {
                suite_id,
                setting_id,
            },
        }
    }
}

/// Flattened item used to represent a row in the test/setting list widget.
enum FlatItem<'a> {
    /// A header for a test suite.
    SuiteHeader {
        /// Suite ID.
        suite_id: u16,
        /// Suite name.
        name: String,
        /// Whether the suite is collapsed.
        collapsed: bool,
    },
    /// A configuration setting.
    Setting {
        /// Parent suite ID.
        suite_id: u16,
        /// Setting ID.
        setting_id: u16,
        /// Reference to the setting item.
        item: &'a SettingItem,
    },
    /// A test case.
    Test {
        /// Parent suite ID.
        suite_id: u16,
        /// Test ID.
        test_id: u16,
        /// Reference to the test item.
        item: &'a TestItem,
    },
}

/// Runs the interactive TUI.
pub fn run_tui(
    bridge: QemuBridge,
    target: &Target,
    elf_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout_handle = stdout();
    execute!(stdout_handle, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout_handle);
    let mut terminal = Terminal::new(backend)?;

    let run_result = run_tui_loop(&mut terminal, bridge, target, elf_path);

    let disable_raw_ok = disable_raw_mode();
    let leave_screen_ok =
        execute!(terminal.backend_mut(), LeaveAlternateScreen);

    run_result?;
    disable_raw_ok?;
    leave_screen_ok?;

    Ok(())
}

/// The main event loop for the TUI, handling drawing, polling messages, and input.
fn run_tui_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    mut bridge: QemuBridge,
    target: &Target,
    elf_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = AppState::new();

    state.target_info = bridge.target_info().to_string();
    state.link_info = bridge.link_info().to_string();

    // Trigger initial test discovery
    let mut last_send = std::time::Instant::now();
    let _ = bridge.send_command(&Command::ListSuites);
    state.add_log(
        "[Host] Connected to target. Triggering discovery...".to_string(),
    );

    let mut exit_tui = false;
    let mut cmd_tx = Vec::new();

    while !exit_tui {
        // Draw TUI
        terminal.draw(|f| draw_ui(f, &mut state))?;

        // Periodically retry ListSuites if discovery has not completed
        if !state.discovery_complete
            && last_send.elapsed() > Duration::from_millis(500)
        {
            let _ = bridge.send_command(&Command::ListSuites);
            last_send = std::time::Instant::now();
        }

        // Handle all process & telemetry messages from bridge
        let mut should_restart_bridge = false;
        while let Ok(msg) = bridge.receiver().try_recv() {
            match msg {
                BridgeMessage::Telemetry(telemetry) => {
                    if let Telemetry::TargetPanic { .. } = telemetry {
                        should_restart_bridge = true;
                    }
                    state.handle_telemetry(telemetry, &mut cmd_tx);
                }
                BridgeMessage::RawConsole(line) => {
                    state.add_log(format!("[CONSOLE] {}", line));
                }
            }
        }

        if should_restart_bridge {
            state.add_log(
                "[Host] Target panic detected. Re-opening comms...".to_string(),
            );
            cmd_tx.clear();

            let _ = bridge.send_command(&Command::OkToReset);
            bridge.kill();

            // Wait 2 seconds for the target to reset and re-establish connection
            std::thread::sleep(std::time::Duration::from_secs(2));

            bridge = match target {
                Target::Qemu => QemuBridge::new(elf_path, target.clone())?,
                Target::Serial { .. } => QemuBridge::new("", target.clone())?,
            };

            state.discovery_complete = false;
            let _ = bridge.send_command(&Command::ListSuites);
            last_send = std::time::Instant::now();
        }

        // Send any commands generated during telemetry handling
        for cmd in cmd_tx.drain(..) {
            let _ = bridge.send_command(&cmd);
        }

        // Check if QEMU process exited
        if let Ok(Some(status)) = bridge.try_wait() {
            state.handle_target_exit(status);
        }

        // Handle user keyboard inputs
        if event::poll(Duration::from_millis(20))? {
            if let Event::Key(key) = event::read()? {
                exit_tui = state.handle_key(key.code, &mut cmd_tx);
            }
        }

        // Send any commands generated during key handling
        for cmd in cmd_tx.drain(..) {
            let _ = bridge.send_command(&cmd);
        }
    }

    // Clean up bridge
    let _ = bridge.send_command(&Command::OkToReset);
    bridge.kill();
    std::thread::sleep(std::time::Duration::from_millis(100));
    Ok(())
}

/// Renders the entire terminal user interface layout.
fn draw_ui(f: &mut ratatui::Frame<'_>, state: &mut AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Content (Split test list & logs)
            Constraint::Length(3), // Footer
        ])
        .split(f.area());

    // 1. Render Header
    let target_span = Span::styled(
        &state.target_info,
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    );
    let link_span =
        Span::styled(&state.link_info, Style::default().fg(Color::Magenta));
    let header_text = vec![Line::from(vec![
        Span::raw(" TARGET: "),
        target_span,
        Span::raw(" | LINK: "),
        link_span,
    ])];
    let header = Paragraph::new(header_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" control-rs HIL Console "),
    );
    f.render_widget(header, chunks[0]);

    // Split mid panel
    let mid_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(55), // Suites & Tests
            Constraint::Percentage(45), // Logs / Console
        ])
        .split(chunks[1]);

    // 2. Render Suites & Tests List
    let flat_items = state.get_flat_items();
    let list_items: Vec<ListItem<'_>> = flat_items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let is_selected = idx == state.selected_item_idx;
            let style = if is_selected {
                Style::default().bg(Color::DarkGray).fg(Color::White)
            } else {
                Style::default()
            };

            match item {
                FlatItem::SuiteHeader {
                    name, collapsed, ..
                } => {
                    let icon = if *collapsed { "►" } else { "▼" };
                    let text = format!("{icon} {name}");
                    ListItem::new(Line::from(Span::styled(
                        text,
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )))
                    .style(style)
                }
                FlatItem::Test { item, .. } => {
                    let symbol = match item.state {
                        TestState::Pending => "[ ---- ]",
                        TestState::Running => "[ RUN  ]",
                        TestState::Passed => "[ PASS ]",
                        TestState::Failed => "[ FAIL ]",
                    };
                    let symbol_color = match item.state {
                        TestState::Pending => Color::Gray,
                        TestState::Running => Color::Yellow,
                        TestState::Passed => Color::Green,
                        TestState::Failed => Color::Red,
                    };
                    let mut line_spans = vec![
                        Span::raw("  ├─ "),
                        Span::styled(symbol, Style::default().fg(symbol_color)),
                        Span::raw(format!(" {} ", item.name)),
                    ];

                    if let (Some(c), Some(t)) = (item.cycles, item.time_us) {
                        let detail_str = if let Some(sp) = item.stack_peak {
                            format!("({c} cyc, {t}us, {sp}B stk)")
                        } else {
                            format!("({c} cyc, {t}us)")
                        };
                        line_spans.push(Span::styled(
                            detail_str,
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    ListItem::new(Line::from(line_spans)).style(style)
                }
                FlatItem::Setting {
                    suite_id,
                    setting_id,
                    item,
                } => {
                    let is_editing =
                        state.editing_setting == Some((*suite_id, *setting_id));
                    let val_str = if is_editing {
                        format!("{}█", state.setting_input)
                    } else {
                        match item.value {
                            SettingValue::U32(v) => format!("{v}"),
                            SettingValue::U8(v) => format!("{v}"),
                        }
                    };
                    ListItem::new(Line::from(vec![
                        Span::raw("  ⚙  "),
                        Span::styled(
                            format!("{}: ", item.name),
                            Style::default().fg(Color::Blue),
                        ),
                        Span::styled(val_str, Style::default().fg(Color::Cyan)),
                    ]))
                    .style(style)
                }
            }
        })
        .collect();

    let list = List::new(list_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Test Suites & Config Settings "),
    );
    state.list_state.select(Some(state.selected_item_idx));
    f.render_stateful_widget(list, mid_chunks[0], &mut state.list_state);

    // 3. Render Logs
    let log_items: Vec<ListItem<'_>> = state
        .console_logs
        .iter()
        .map(|line| {
            let style = if line.contains("[FAIL]") || line.contains("[PANIC]") {
                Style::default().fg(Color::Red)
            } else if line.contains("[PASS]") {
                Style::default().fg(Color::Green)
            } else if line.contains("[LOG]") {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(Span::styled(line, style)))
        })
        .collect();
    let log_list = List::new(log_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Target Console / RTT Logs "),
    );
    let mut log_state = ListState::default();
    if !state.console_logs.is_empty() {
        log_state.select(Some(state.console_logs.len().saturating_sub(1)));
    }
    f.render_stateful_widget(log_list, mid_chunks[1], &mut log_state);

    // 4. Render Footer
    let footer_text = if state.show_details_modal {
        " Press any key to close description details modal".to_string()
    } else if state.is_filtering {
        format!(
            " FILTER QUERY: {} | Press Enter/Esc to finish",
            state.filter_query
        )
    } else if let Some((suite_id, setting_id)) = state.editing_setting {
        let s_idx = suite_id as usize;
        let set_idx = setting_id as usize;
        let setting_name = if s_idx < state.suites.len()
            && set_idx < state.suites[s_idx].settings.len()
        {
            &state.suites[s_idx].settings[set_idx].name
        } else {
            "Setting"
        };
        format!(
            " EDIT SETTING '{}': {}█ | Press Enter to save, Esc to cancel",
            setting_name, state.setting_input
        )
    } else {
        "(r)un all | (s)top execution | (f)ilter tests | (Enter) edit/run/toggle | (d)escription | (q)uit".to_string()
    };
    let footer = Paragraph::new(footer_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Keyboard Commands "),
    );
    f.render_widget(footer, chunks[2]);

    // 5. Render details modal popup if open (anchored to bottom-left)
    if state.show_details_modal {
        let size = f.area();
        let popup_width = 60.min(size.width);
        let popup_height = 10.min(size.height);

        let x = 0;
        let y = chunks[2].y.saturating_sub(popup_height);
        let popup_rect = ratatui::layout::Rect {
            x,
            y,
            width: popup_width,
            height: popup_height,
        };

        let flat_items = state.get_flat_items();
        if let Some(selected_flat_item) =
            flat_items.get(state.selected_item_idx)
        {
            let description_owned: String;
            let (name, type_str) = match selected_flat_item {
                FlatItem::SuiteHeader { suite_id, name, .. } => {
                    let s_idx = *suite_id as usize;
                    description_owned = if s_idx < state.suites.len() {
                        state.suites[s_idx].description.clone()
                    } else {
                        String::new()
                    };
                    (name.as_str(), "Suite")
                }
                FlatItem::Test { item, .. } => {
                    let mut desc = item.description.clone();
                    if let (Some(c), Some(t)) = (item.cycles, item.time_us) {
                        use std::fmt::Write;
                        let _ = write!(
                            &mut desc,
                            "\n\nMetrics:\n- CPU Cycles: {}\n- Execution Time: {} us",
                            c, t
                        );
                        if let Some(sp) = item.stack_peak {
                            let _ = write!(
                                &mut desc,
                                "\n- Stack High-Water: {} bytes",
                                sp
                            );
                        }
                    }
                    description_owned = desc;
                    (item.name.as_str(), "Test")
                }
                FlatItem::Setting { item, .. } => {
                    description_owned = item.description.clone();
                    (item.name.as_str(), "Setting")
                }
            };

            let modal_block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Cyan))
                .title(format!(" Description: {} ({}) ", name, type_str));

            let display_desc = if description_owned.is_empty() {
                "No description provided."
            } else {
                description_owned.as_str()
            };

            let paragraph = Paragraph::new(display_desc)
                .block(modal_block)
                .wrap(ratatui::widgets::Wrap { trim: true });

            f.render_widget(ratatui::widgets::Clear, popup_rect);
            f.render_widget(paragraph, popup_rect);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use control_rs_hil::comms::LogMessage;

    fn make_exit_status() -> std::process::ExitStatus {
        std::process::Command::new("true")
            .status()
            .unwrap_or_else(|_| {
                #[cfg(unix)]
                {
                    use std::os::unix::process::ExitStatusExt;
                    std::process::ExitStatus::from_raw(0)
                }
                #[cfg(not(unix))]
                panic!("Cannot construct ExitStatus on this platform");
            })
    }

    #[test]
    fn test_initial_state() {
        let state = AppState::new();
        assert!(state.suites.is_empty());
        assert!(state.console_logs.is_empty());
        assert_eq!(state.selected_item_idx, 0);
        assert!(state.run_queue.is_empty());
        assert!(state.current_running.is_none());
        assert!(state.filter_query.is_empty());
        assert!(!state.is_filtering);
        assert!(!state.discovery_complete);
    }

    #[test]
    fn test_log_limit_enforced() {
        let mut state = AppState::new();
        for i in 0..120 {
            state.add_log(format!("log {i}"));
        }
        assert_eq!(state.console_logs.len(), 100);
        assert_eq!(state.console_logs[0], "log 20");
        assert_eq!(state.console_logs[99], "log 119");
    }

    #[test]
    fn test_telemetry_discovery() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        // Add suite 0
        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite_zero",
                description: "",
                test_count: 2,
                setting_count: 1,
            },
            &mut cmd_tx,
        );

        // Add test 0 in suite 0
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test_zero",
                description: "",
            },
            &mut cmd_tx,
        );

        // Add test 1 in suite 0
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test_one",
                description: "",
            },
            &mut cmd_tx,
        );

        // Add setting 0 in suite 0
        state.handle_telemetry(
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "setting_zero",
                description: "",
                value: SettingValue::U8(3),
            },
            &mut cmd_tx,
        );

        // Complete discovery
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        assert!(state.discovery_complete);
        assert_eq!(state.suites.len(), 1);
        assert_eq!(state.suites[0].name, "suite_zero");
        assert_eq!(state.suites[0].tests.len(), 2);
        assert_eq!(state.suites[0].tests[0].name, "test_zero");
        assert_eq!(state.suites[0].tests[1].name, "test_one");
        assert_eq!(state.suites[0].settings.len(), 1);
        assert_eq!(state.suites[0].settings[0].name, "setting_zero");

        // Verify flat items representation
        let flat_items = state.get_flat_items();
        assert_eq!(flat_items.len(), 4); // 1 header + 2 tests + 1 setting
    }

    #[test]
    fn test_telemetry_test_state_and_queue() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        // Set up a suite and two tests
        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite_zero",
                description: "",
                test_count: 2,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test_zero",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test_one",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        // Trigger run all via handle_key
        state.handle_key(KeyCode::Char('r'), &mut cmd_tx);

        // Check that test_zero is started and test_one is in run_queue
        assert_eq!(state.current_running, None); // handle_key starts run but doesn't transition state to Running until Telemetry
        assert_eq!(state.run_queue, vec![(0, 1)]);
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::RunExecutable {
                suite_id: 0,
                test_id: 0
            }
        ));
        cmd_tx.clear();

        // Simulate Telemetry state change to Running
        state.handle_telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Running,
            },
            &mut cmd_tx,
        );
        assert_eq!(state.current_running, Some((0, 0)));

        // Simulate Test passed
        state.handle_telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Passed,
            },
            &mut cmd_tx,
        );
        assert_eq!(state.current_running, None);
        // Next test in queue (0, 1) should be triggered
        assert_eq!(state.run_queue.len(), 0);
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::RunExecutable {
                suite_id: 0,
                test_id: 1
            }
        ));
    }

    #[test]
    fn test_telemetry_metrics_and_logs() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite_zero",
                description: "",
                test_count: 1,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test_zero",
                description: "",
            },
            &mut cmd_tx,
        );

        state.handle_telemetry(
            Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                cycles: 4200,
                time_us: 150,
                stack_peak: 256,
            },
            &mut cmd_tx,
        );

        assert_eq!(state.suites[0].tests[0].cycles, Some(4200));
        assert_eq!(state.suites[0].tests[0].time_us, Some(150));
        assert_eq!(state.suites[0].tests[0].stack_peak, Some(256));
        assert!(state.console_logs.iter().any(|l| l.contains("[PASS]")));

        state.handle_telemetry(
            Telemetry::Log(LogMessage {
                timestamp_us: 1000,
                suite_id: 0,
                test_id: 0,
                payload: "hello",
            }),
            &mut cmd_tx,
        );
        assert!(state.console_logs.iter().any(|l| l.contains("[LOG] hello")));
    }

    #[test]
    fn test_telemetry_target_panic() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        // Setup suite and test first
        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite_zero",
                description: "",
                test_count: 1,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test_zero",
                description: "",
            },
            &mut cmd_tx,
        );
        state.suites[0].tests[0].state = TestState::Running;

        state.run_queue = vec![(0, 1), (0, 2)];
        state.current_running = Some((0, 0));

        state.handle_telemetry(
            Telemetry::TargetPanic {
                message: "OOM",
                file: "main.rs",
                line: 10,
            },
            &mut cmd_tx,
        );

        assert!(state.run_queue.is_empty());
        assert!(state.current_running.is_none());
        assert_eq!(state.suites[0].tests[0].state, TestState::Failed);
        assert!(
            state
                .console_logs
                .iter()
                .any(|l| l.contains("[PANIC] Target crashed: 'OOM'"))
        );
    }

    #[test]
    fn test_filtering_logic() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        // Add dummy tests and setting
        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "SuiteAlpha",
                description: "",
                test_count: 2,
                setting_count: 1,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test_apple",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test_banana",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "config_val",
                description: "",
                value: SettingValue::U8(3),
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        // Transition to filtering mode
        state.handle_key(KeyCode::Char('f'), &mut cmd_tx);
        assert!(state.is_filtering);
        assert!(state.filter_query.is_empty());

        // Type 'ap'
        state.handle_key(KeyCode::Char('a'), &mut cmd_tx);
        state.handle_key(KeyCode::Char('p'), &mut cmd_tx);
        assert_eq!(state.filter_query, "ap");

        // Verify flat items only match test_apple (and SuiteAlpha header because a test inside matches)
        // Settings are skipped when filtering
        let flat_items = state.get_flat_items();
        assert_eq!(flat_items.len(), 2); // Header + test_apple
        match &flat_items[0] {
            FlatItem::SuiteHeader { name, .. } => {
                assert_eq!(name, "SuiteAlpha");
            }
            _ => panic!("Expected SuiteHeader"),
        }
        match &flat_items[1] {
            FlatItem::Test { item, .. } => assert_eq!(item.name, "test_apple"),
            _ => panic!("Expected Test"),
        }

        // Backspace
        state.handle_key(KeyCode::Backspace, &mut cmd_tx);
        assert_eq!(state.filter_query, "a");

        // Esc to exit filtering
        state.handle_key(KeyCode::Esc, &mut cmd_tx);
        assert!(!state.is_filtering);
    }

    #[test]
    fn test_navigation_and_selection() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 2,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test0",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test1",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        // 3 flat items: 1 header, 2 tests
        assert_eq!(state.selected_item_idx, 0);

        // Move Down
        state.handle_key(KeyCode::Down, &mut cmd_tx);
        assert_eq!(state.selected_item_idx, 1);

        // Move Down again
        state.handle_key(KeyCode::Down, &mut cmd_tx);
        assert_eq!(state.selected_item_idx, 2);

        // Wrap around Down
        state.handle_key(KeyCode::Down, &mut cmd_tx);
        assert_eq!(state.selected_item_idx, 0);

        // Wrap around Up
        state.handle_key(KeyCode::Up, &mut cmd_tx);
        assert_eq!(state.selected_item_idx, 2);
    }

    #[test]
    fn test_setting_value_editing() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 0,
                setting_count: 2,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "setting_u8",
                description: "",
                value: SettingValue::U8(3),
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 1,
                name: "setting_u32",
                description: "",
                value: SettingValue::U32(1000),
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        // selected_item_idx = 0 is SuiteHeader
        // selected_item_idx = 1 is setting_u8
        state.selected_item_idx = 1;

        // Enter edit mode
        state.handle_key(KeyCode::Enter, &mut cmd_tx);
        assert_eq!(state.editing_setting, Some((0, 0)));
        assert_eq!(state.setting_input, "3");

        // Backspace to clear it
        state.handle_key(KeyCode::Backspace, &mut cmd_tx);
        assert_eq!(state.setting_input, "");

        // Type '5'
        state.handle_key(KeyCode::Char('5'), &mut cmd_tx);
        assert_eq!(state.setting_input, "5");

        // Press Enter to submit
        state.handle_key(KeyCode::Enter, &mut cmd_tx);
        assert_eq!(state.editing_setting, None);
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::SetSetting {
                suite_id: 0,
                setting_id: 0,
                value: SettingValue::U8(5)
            }
        ));
        cmd_tx.clear();

        // selected_item_idx = 2 is setting_u32
        state.selected_item_idx = 2;
        // Enter edit mode
        state.handle_key(KeyCode::Enter, &mut cmd_tx);
        assert_eq!(state.editing_setting, Some((0, 1)));
        assert_eq!(state.setting_input, "1000");

        // Type '0' to make it '10000'
        state.handle_key(KeyCode::Char('0'), &mut cmd_tx);
        assert_eq!(state.setting_input, "10000");

        // Press Enter to submit
        state.handle_key(KeyCode::Enter, &mut cmd_tx);
        assert_eq!(state.editing_setting, None);
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::SetSetting {
                suite_id: 0,
                setting_id: 1,
                value: SettingValue::U32(10000)
            }
        ));
    }

    #[test]
    fn test_suite_collapsible() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 2,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test0",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test1",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        // Initially 3 items: 1 header, 2 tests
        assert_eq!(state.get_flat_items().len(), 3);

        // Select suite header (idx = 0) and press Enter to collapse
        state.selected_item_idx = 0;
        state.handle_key(KeyCode::Enter, &mut cmd_tx);

        assert!(state.suites[0].collapsed);
        // Only 1 item now: header
        assert_eq!(state.get_flat_items().len(), 1);

        // Clamping selected_item_idx when list is smaller
        state.selected_item_idx = 2; // out of bounds now
        state.handle_key(KeyCode::Char('f'), &mut cmd_tx); // enters filtering, should clamp
        state.handle_key(KeyCode::Esc, &mut cmd_tx); // exits filtering
        assert_eq!(state.selected_item_idx, 0);

        // Press Enter to expand again
        state.selected_item_idx = 0;
        state.handle_key(KeyCode::Enter, &mut cmd_tx);
        assert!(!state.suites[0].collapsed);
        assert_eq!(state.get_flat_items().len(), 3);
    }

    #[test]
    fn test_run_starts_at_selection() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 3,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test0",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test1",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 2,
                name: "test2",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);
        cmd_tx.clear();

        // 4 flat items: 1 header, 3 tests
        // Let's select test1 (idx = 2)
        state.selected_item_idx = 2;

        // Press 'r'
        state.handle_key(KeyCode::Char('r'), &mut cmd_tx);

        // Should start running test1 and have test2 and test0 (rotated to the back) in the run_queue
        assert_eq!(state.run_queue, vec![(0, 2), (0, 0)]);
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::RunExecutable {
                suite_id: 0,
                test_id: 1
            }
        ));
    }

    #[test]
    fn test_target_exits() {
        let mut state = AppState::new();
        let status = make_exit_status();

        state.current_running = Some((0, 0));
        state.run_queue = vec![(0, 1)];

        state.handle_target_exit(status);

        assert!(state.current_running.is_none());
        assert!(state.run_queue.is_empty());
        assert!(
            state
                .console_logs
                .iter()
                .any(|l| l.contains("exited unexpectedly"))
        );
    }

    #[test]
    fn test_requeue_already_run_tests() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        // 1. Discover 2 tests
        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 2,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test0",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 1,
                name: "test1",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        // 2. Mark test0 as Passed and test1 as Failed (simulating they have already run)
        state.suites[0].tests[0].state = TestState::Passed;
        state.suites[0].tests[1].state = TestState::Failed;

        // 3. Press 'r' to Run All again
        state.handle_key(KeyCode::Char('r'), &mut cmd_tx);

        // Confirm that the TUI enqueues both tests and sends a RunExecutable command for test0
        assert_eq!(state.run_queue, vec![(0, 1)]);
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::RunExecutable {
                suite_id: 0,
                test_id: 0
            }
        ));
    }

    #[test]
    fn test_send_proper_commands_after_reset() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        // 1. Discovery
        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 1,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test0",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);
        cmd_tx.clear();

        // 2. Run the test
        state.handle_key(KeyCode::Char('r'), &mut cmd_tx);
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::RunExecutable {
                suite_id: 0,
                test_id: 0
            }
        ));
        cmd_tx.clear();

        // Simulate state transition to Running
        state.handle_telemetry(
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Running,
            },
            &mut cmd_tx,
        );
        assert_eq!(state.current_running, Some((0, 0)));

        // 3. Target resets (simulate target exit)
        let status = make_exit_status();
        state.handle_target_exit(status);

        // Verify that current_running and queue are cleared
        assert!(state.current_running.is_none());
        assert!(state.run_queue.is_empty());

        // 4. Target boots back up (simulate re-discovery telemetry)
        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite",
                description: "",
                test_count: 1,
                setting_count: 0,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test0",
                description: "",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);
        cmd_tx.clear();

        // 5. Select test0 and run it again (Enter)
        state.selected_item_idx = 1; // Item 0 is SuiteHeader, Item 1 is test0
        state.handle_key(KeyCode::Enter, &mut cmd_tx);

        // Verify TUI sends the proper RunExecutable command
        assert_eq!(cmd_tx.len(), 1);
        assert!(matches!(
            cmd_tx[0],
            Command::RunExecutable {
                suite_id: 0,
                test_id: 0
            }
        ));
    }

    #[test]
    fn test_details_modal() {
        let mut state = AppState::new();
        let mut cmd_tx = Vec::new();

        state.handle_telemetry(
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "suite0",
                description: "suite0_description",
                test_count: 1,
                setting_count: 1,
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "test0",
                description: "test0_description",
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "setting0",
                description: "setting0_description",
                value: SettingValue::U8(10),
            },
            &mut cmd_tx,
        );
        state.handle_telemetry(Telemetry::DiscoveryComplete, &mut cmd_tx);

        assert!(!state.show_details_modal);

        // Press 'd' to open the details modal
        state.handle_key(KeyCode::Char('d'), &mut cmd_tx);
        assert!(state.show_details_modal);

        // Press any key (e.g., Up) to close it.
        // It should close, and NOT execute the normal action of that key (which is moving selection).
        state.selected_item_idx = 1;
        state.handle_key(KeyCode::Up, &mut cmd_tx);
        assert!(!state.show_details_modal);
        assert_eq!(state.selected_item_idx, 1);
    }
}
