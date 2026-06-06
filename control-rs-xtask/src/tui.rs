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
    widgets::{Block, Borders, List, ListItem, Paragraph},
};

use crate::bridge::{BridgeMessage, QemuBridge};
use control_rs_hil::comms::{Command, Telemetry, TestState};
use control_rs_hil::hil_test::SettingValue;

struct TestItem {
    name: String,
    state: TestState,
    cycles: Option<u64>,
    time_us: Option<u64>,
}

struct SettingItem {
    name: String,
    value: SettingValue,
}

struct SuiteItem {
    name: String,
    tests: Vec<TestItem>,
    settings: Vec<SettingItem>,
}

struct AppState {
    suites: Vec<SuiteItem>,
    console_logs: Vec<String>,
    selected_item_idx: usize, // Index in the flattened list of tests/settings
    run_queue: Vec<(u16, u16)>,
    current_running: Option<(u16, u16)>,
    filter_query: String,
    is_filtering: bool,
    target_info: String,
    link_info: String,
}

impl AppState {
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
        }
    }

    fn add_log(&mut self, log: String) {
        self.console_logs.push(log);
        if self.console_logs.len() > 100 {
            self.console_logs.remove(0);
        }
    }

    // Returns a flattened list of items for UI rendering and navigation
    // (Suite headers, Tests, and Settings)
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
                _suite_id: s_idx as u16,
                name: suite.name.clone(),
            });

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
}

enum FlatItem<'a> {
    SuiteHeader {
        _suite_id: u16,
        name: String,
    },
    Test {
        suite_id: u16,
        test_id: u16,
        item: &'a TestItem,
    },
    Setting {
        suite_id: u16,
        setting_id: u16,
        item: &'a SettingItem,
    },
}

/// Runs the interactive TUI.
pub fn run_tui(
    mut bridge: QemuBridge,
) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout_handle = stdout();
    execute!(stdout_handle, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout_handle);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new();

    // Dynamically update header based on hil.toml configuration
    let config = crate::bridge::HilConfig::load();
    if config.target == "serial" {
        if let Some(serial) = &config.serial {
            state.target_info = "Teensy 4.0 (Cortex-M7)".to_string();
            state.link_info = format!("USB CDC ({})", serial.port);
        }
    } else if let Some(qemu) = &config.qemu {
        state.target_info = format!("QEMU ({})", qemu.cpu);
        state.link_info = format!("Semihosting ({})", qemu.machine);
    }

    // Trigger initial test discovery
    let mut last_send = std::time::Instant::now();
    let _ = bridge.send_command(&Command::ListSuites);
    state.add_log(
        "[Host] Connected to target. Triggering discovery...".to_string(),
    );

    let mut discovery_complete = false;
    let mut exit_tui = false;

    while !exit_tui {
        // Draw TUI
        terminal.draw(|f| draw_ui(f, &state))?;

        // Periodically retry ListSuites if discovery has not completed
        if !discovery_complete
            && last_send.elapsed() > Duration::from_millis(500)
        {
            let _ = bridge.send_command(&Command::ListSuites);
            last_send = std::time::Instant::now();
        }

        // Handle process & telemetry messages from bridge
        while let Ok(msg) = bridge.receiver().try_recv() {
            match msg {
                BridgeMessage::Telemetry(telemetry) => match telemetry {
                    Telemetry::SuiteInfo { suite_id, name, .. } => {
                        let id = suite_id as usize;
                        while state.suites.len() <= id {
                            state.suites.push(SuiteItem {
                                name: String::new(),
                                tests: Vec::new(),
                                settings: Vec::new(),
                            });
                        }
                        state.suites[id].name = name.to_string();
                    }
                    Telemetry::TestInfo {
                        suite_id,
                        test_id,
                        name,
                    } => {
                        let s_id = suite_id as usize;
                        let t_id = test_id as usize;
                        while state.suites[s_id].tests.len() <= t_id {
                            state.suites[s_id].tests.push(TestItem {
                                name: String::new(),
                                state: TestState::Pending,
                                cycles: None,
                                time_us: None,
                            });
                        }
                        state.suites[s_id].tests[t_id].name = name.to_string();
                    }
                    Telemetry::SettingInfo {
                        suite_id,
                        setting_id,
                        name,
                        value,
                    } => {
                        let s_id = suite_id as usize;
                        let set_id = setting_id as usize;
                        while state.suites[s_id].settings.len() <= set_id {
                            state.suites[s_id].settings.push(SettingItem {
                                name: String::new(),
                                value: SettingValue::U8(0),
                            });
                        }
                        state.suites[s_id].settings[set_id].name =
                            name.to_string();
                        state.suites[s_id].settings[set_id].value = value;
                    }
                    Telemetry::DiscoveryComplete => {
                        state.add_log("[Host] Discovery complete.".to_string());
                        discovery_complete = true;
                    }
                    Telemetry::TestStateChange {
                        suite_id,
                        test_id,
                        state: new_state,
                    } => {
                        let s_id = suite_id as usize;
                        let t_id = test_id as usize;
                        state.suites[s_id].tests[t_id].state = new_state;

                        if new_state == TestState::Running {
                            state.current_running = Some((suite_id, test_id));
                        } else {
                            state.current_running = None;
                            // Trigger next test in queue if any
                            if !state.run_queue.is_empty() {
                                let (next_s, next_t) =
                                    state.run_queue.remove(0);
                                let _ = bridge.send_command(
                                    &Command::RunExecutable {
                                        suite_id: next_s,
                                        test_id: next_t,
                                    },
                                );
                            }
                        }
                    }
                    Telemetry::MetricReport {
                        suite_id,
                        test_id,
                        cycles,
                        time_us,
                    } => {
                        let s_id = suite_id as usize;
                        let t_id = test_id as usize;
                        state.suites[s_id].tests[t_id].cycles = Some(cycles);
                        state.suites[s_id].tests[t_id].time_us = Some(time_us);
                        state.add_log(format!(
                            "[PASS] {}::{} ({} cycles, {}us)",
                            state.suites[s_id].name,
                            state.suites[s_id].tests[t_id].name,
                            cycles,
                            time_us
                        ));
                    }
                    Telemetry::Log(log) => {
                        state.add_log(format!("[LOG] {}", log.payload));
                    }
                    Telemetry::TargetPanic {
                        message,
                        file,
                        line,
                    } => {
                        state.add_log(format!(
                            "[PANIC] Target crashed: '{}' at {}:{}",
                            message, file, line
                        ));
                        state.current_running = None;
                        state.run_queue.clear();
                    }
                },
                BridgeMessage::RawConsole(line) => {
                    state.add_log(format!("[CONSOLE] {}", line));
                }
            }
        }

        // Check if QEMU process exited
        match bridge.try_wait() {
            Ok(Some(status)) => {
                if state.current_running.is_some()
                    || !state.run_queue.is_empty()
                {
                    state.add_log(format!(
                        "[Host] Target process exited unexpectedly: {}",
                        status
                    ));
                } else {
                    state.add_log(format!(
                        "[Host] Target process exited: {}",
                        status
                    ));
                }
                state.current_running = None;
                state.run_queue.clear();
            }
            _ => {}
        }

        // Handle user keyboard inputs
        if event::poll(Duration::from_millis(20))? {
            if let Event::Key(key) = event::read()? {
                if state.is_filtering {
                    match key.code {
                        KeyCode::Char(c) => {
                            state.filter_query.push(c);
                            state.selected_item_idx = 0;
                        }
                        KeyCode::Backspace => {
                            state.filter_query.pop();
                            state.selected_item_idx = 0;
                        }
                        KeyCode::Esc | KeyCode::Enter => {
                            state.is_filtering = false;
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Char('q') => {
                            exit_tui = true;
                        }
                        KeyCode::Char('r') => {
                            // Run all tests
                            state.run_queue.clear();
                            for (s_idx, suite) in
                                state.suites.iter().enumerate()
                            {
                                for (t_idx, _) in suite.tests.iter().enumerate()
                                {
                                    state
                                        .run_queue
                                        .push((s_idx as u16, t_idx as u16));
                                }
                            }
                            if !state.run_queue.is_empty()
                                && state.current_running.is_none()
                            {
                                let (next_s, next_t) =
                                    state.run_queue.remove(0);
                                let _ = bridge.send_command(
                                    &Command::RunExecutable {
                                        suite_id: next_s,
                                        test_id: next_t,
                                    },
                                );
                            }
                        }
                        KeyCode::Char('s') => {
                            state.run_queue.clear();
                            state.current_running = None;
                            state.add_log(
                                "[Host] Stopped execution queue.".to_string(),
                            );
                        }
                        KeyCode::Char('f') => {
                            state.is_filtering = true;
                            state.filter_query.clear();
                        }
                        KeyCode::Up => {
                            let flat_len = state.get_flat_items().len();
                            if flat_len > 0 {
                                if state.selected_item_idx == 0 {
                                    state.selected_item_idx = flat_len - 1;
                                } else {
                                    state.selected_item_idx -= 1;
                                }
                            }
                        }
                        KeyCode::Down => {
                            let flat_len = state.get_flat_items().len();
                            if flat_len > 0 {
                                state.selected_item_idx =
                                    (state.selected_item_idx + 1) % flat_len;
                            }
                        }
                        KeyCode::Enter => {
                            let flat_items = state.get_flat_items();
                            if let Some(item) =
                                flat_items.get(state.selected_item_idx)
                            {
                                match item {
                                    FlatItem::Test {
                                        suite_id,
                                        test_id,
                                        ..
                                    } => {
                                        if state.current_running.is_none() {
                                            let _ = bridge.send_command(
                                                &Command::RunExecutable {
                                                    suite_id: *suite_id,
                                                    test_id: *test_id,
                                                },
                                            );
                                        }
                                    }
                                    FlatItem::Setting {
                                        suite_id,
                                        setting_id,
                                        item,
                                    } => {
                                        // Simple toggler for setting values
                                        let next_val = match item.value {
                                            SettingValue::U32(v) => {
                                                SettingValue::U32(
                                                    if v == 1000 {
                                                        5000
                                                    } else {
                                                        1000
                                                    },
                                                )
                                            }
                                            SettingValue::U8(v) => {
                                                SettingValue::U8(if v == 3 {
                                                    5
                                                } else {
                                                    3
                                                })
                                            }
                                        };
                                        let _ = bridge.send_command(
                                            &Command::SetSetting {
                                                suite_id: *suite_id,
                                                setting_id: *setting_id,
                                                value: next_val,
                                            },
                                        );
                                    }
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Clean up terminal
    bridge.kill();
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn draw_ui(f: &mut ratatui::Frame, state: &AppState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Content (Split test list & logs)
            Constraint::Length(3), // Footer
        ])
        .split(f.size());

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
    let list_items: Vec<ListItem> = flat_items
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
                FlatItem::SuiteHeader { name, .. } => {
                    let text = format!("▼ {}", name);
                    ListItem::new(Line::from(Span::styled(
                        text,
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )))
                }
                FlatItem::Test { item, .. } => {
                    let symbol = match item.state {
                        TestState::Pending => "[ PEND ]",
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
                        line_spans.push(Span::styled(
                            format!("({} cyc, {}us)", c, t),
                            Style::default().fg(Color::DarkGray),
                        ));
                    }
                    ListItem::new(Line::from(line_spans)).style(style)
                }
                FlatItem::Setting { item, .. } => {
                    let val_str = match item.value {
                        SettingValue::U32(v) => format!("{}", v),
                        SettingValue::U8(v) => format!("{}", v),
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
    f.render_widget(list, mid_chunks[0]);

    // 3. Render Logs
    let log_items: Vec<ListItem> = state
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
    f.render_widget(log_list, mid_chunks[1]);

    // 4. Render Footer
    let footer_text = if state.is_filtering {
        format!(
            " FILTER QUERY: {} | Press Enter/Esc to finish",
            state.filter_query
        )
    } else {
        "(r)un all | (s)top execution | (f)ilter tests | (Enter) toggle/run | (q)uit".to_string()
    };
    let footer = Paragraph::new(footer_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Keyboard Commands "),
    );
    f.render_widget(footer, chunks[2]);
}
