//! Drawing: the layout of the dashboard.

use std::fmt::Write as _;

use control_rs_ets::wire::{ExecMode, ItemState, Verdict};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Paragraph, Row as TableRow, Table,
};

use crate::app::{
    App, CaseState, Row, Status, TaskRow, View, format_param, param_type,
    run_status_text,
};
use crate::keys::Mode;

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

/// Formats a duration in microseconds as `µs`, `ms` or `s`, rounded half-up to two decimals.
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

/// Draws the dashboard.
pub fn draw(frame: &mut Frame<'_>, app: &mut App) {
    let [header, table, task, logs, footer] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Min(6),
            Constraint::Length(3),
            Constraint::Length(7),
            Constraint::Length(1),
        ])
        .areas(frame.area());
    frame.render_widget(header_widget(app), header);
    let widget = table_widget(app);
    frame.render_stateful_widget(widget, table, &mut app.table);
    frame.render_widget(task_widget(app), task);
    frame.render_widget(log_widget(app, logs.height), logs);
    frame.render_widget(Paragraph::new(footer_line(app)), footer);
}

fn bold(color: Color) -> Style {
    Style::default().fg(color).add_modifier(Modifier::BOLD)
}

fn header_widget(app: &App) -> Paragraph<'static> {
    let (passed, failed) = app.totals();
    let total: usize =
        app.catalog.suites.iter().map(|s| s.executables.len()).sum();
    let mut counts =
        format!(" Items: {total} | Passed: {passed} | Failed: {failed}");
    if !app.faults.is_empty() {
        let last = app.faults.last().map_or("", String::as_str);
        let _ =
            write!(counts, " | Faults: {} (last: {last})", app.faults.len());
    }
    let target = app.target_text();
    Paragraph::new(vec![
        Line::from(format!(" TARGET: {} | LINK: {}", app.target, app.link)),
        status_line(app),
        Line::from(counts),
        Line::from(format!(" {target}")),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Embedded Test Server (ETS) Dashboard "),
    )
}

fn status_line(app: &App) -> Line<'static> {
    let (label, color, detail) = match &app.status {
        Status::Discovering => (
            " [ DISCOVERING ] ",
            Color::Yellow,
            "Querying target suites...".to_string(),
        ),
        Status::Idle => (" [ IDLE ] ", Color::Green, "Ready".to_string()),
        Status::LinkLost(text) => (" [ LINK LOST ] ", Color::Red, text.clone()),
        Status::Mismatch { .. } => (
            " [ MISMATCH ] ",
            Color::Red,
            "The target speaks another wire revision".to_string(),
        ),
        Status::Running((suite, item)) => {
            let name = app
                .catalog
                .exec((*suite, *item))
                .map_or("unknown", |e| e.name.as_str());
            (
                " [ RUNNING ] ",
                Color::Cyan,
                format!("{suite}/{item} {name}"),
            )
        }
    };
    Line::from(vec![Span::styled(label, bold(color)), Span::raw(detail)])
}

fn table_widget(app: &App) -> Table<'static> {
    let heads: [&str; 4] = match app.view {
        View::Tests => ["  Suite / Case / Task", "Cycles", "Time", "Stack (B)"],
        View::Settings => ["  Suite / Setting", "Type", "Value", "Range"],
    };
    Table::new(
        app.rows().into_iter().map(|r| table_row(app, &r)),
        [
            Constraint::Percentage(45),
            Constraint::Length(16),
            Constraint::Length(14),
            Constraint::Min(10),
        ],
    )
    .header(TableRow::new(
        heads.map(|h| Cell::from(h).style(bold(Color::Cyan))),
    ))
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

const fn branch(last: bool) -> &'static str {
    if last { "└─" } else { "├─" }
}

fn table_row(app: &App, row: &Row) -> TableRow<'static> {
    match row {
        Row::Suite {
            collapsed, name, ..
        } => {
            let arrow = if *collapsed { "▶" } else { "▼" };
            TableRow::new(vec![
                Cell::from(format!("{arrow} {name}"))
                    .style(bold(Color::Yellow)),
                Cell::from(""),
                Cell::from(""),
                Cell::from(""),
            ])
        }
        Row::Case { id, last, name } => {
            let [a, b, c] = case_cells(app, *id);
            TableRow::new(vec![
                Cell::from(format!("  {} {name}", branch(*last))),
                a,
                b,
                c,
            ])
        }
        Row::Task { id, last, name } => {
            let [a, b, c] = task_cells(app, *id);
            TableRow::new(vec![
                Cell::from(format!("  {} ↻ {name}", branch(*last)))
                    .style(Style::default().fg(Color::Blue)),
                a,
                b,
                c,
            ])
        }
        Row::Setting { .. } => setting_row(app, row),
    }
}

fn setting_row(app: &App, row: &Row) -> TableRow<'static> {
    let Row::Setting {
        index,
        last,
        name,
        suite,
    } = row
    else {
        return TableRow::new(Vec::<Cell<'static>>::new());
    };
    let entry = app
        .catalog
        .suites
        .get(usize::from(*suite))
        .and_then(|s| s.settings.get(usize::from(*index)))
        .and_then(Option::as_ref);
    let [ty, value, range] = entry.map_or_else(
        || [String::new(), String::new(), String::new()],
        |s| {
            [
                param_type(s.value).to_string(),
                format_param(s.value),
                format!("{} .. {}", format_param(s.min), format_param(s.max)),
            ]
        },
    );
    TableRow::new(vec![
        Cell::from(format!("  {} {name}", branch(*last)))
            .style(Style::default().fg(Color::Magenta)),
        Cell::from(ty),
        Cell::from(value),
        Cell::from(range),
    ])
}

fn case_cells(app: &App, id: (u16, u16)) -> [Cell<'static>; 3] {
    let Some(row) = app.cases.get(&id) else {
        return [Cell::from(""), Cell::from(""), Cell::from("")];
    };
    let time = row.stats.map_or_else(
        || "---".to_string(),
        |s| format_duration(s.time_ns / 1000),
    );
    let stack = row.stats.map_or_else(
        || "---".to_string(),
        |s| format_number(u64::from(s.stack_peak)),
    );
    match row.state {
        CaseState::Running => [
            Cell::from("[ RUN... ]").style(bold(Color::Cyan)),
            Cell::from("---"),
            Cell::from("---"),
        ],
        CaseState::Pending => [
            Cell::from("PENDING").style(Style::default().fg(Color::DarkGray)),
            Cell::from("---"),
            Cell::from("---"),
        ],
        CaseState::Failed => [
            Cell::from("FAIL").style(bold(Color::Red)),
            Cell::from(time),
            Cell::from(stack),
        ],
        CaseState::Passed => [
            Cell::from(row.stats.map_or_else(
                || "N/A".to_string(),
                |s| format_number(s.cycles),
            ))
            .style(Style::default().fg(Color::Green)),
            Cell::from(time),
            Cell::from(stack),
        ],
    }
}

const fn mode_label(mode: Option<ExecMode>) -> &'static str {
    match mode {
        Some(ExecMode::Atomic) => "Atomic",
        Some(ExecMode::Lockstep) => "Lockstep",
        Some(ExecMode::FreeRunning) => "FreeRunning",
        Some(ExecMode::Periodic) => "Periodic",
        Some(ExecMode::Triggered) => "Triggered",
        None => "-",
    }
}

const fn verdict_label(verdict: Verdict) -> &'static str {
    match verdict {
        Verdict::Pass => "pass",
        Verdict::Fail => "fail",
        Verdict::Error => "error",
        Verdict::Aborted => "aborted",
        Verdict::TimedOut => "timed out",
        Verdict::Bounded => "bounded",
        Verdict::Overrun => "overrun",
    }
}

fn task_cells(app: &App, id: (u16, u16)) -> [Cell<'static>; 3] {
    let mode = mode_label(app.task_mode(id));
    let Some(row) = app.tasks.get(&id) else {
        return [Cell::from(mode), Cell::from(""), Cell::from("")];
    };
    let running = matches!(row.state, Some(ItemState::Running));
    let steps = row
        .record
        .as_ref()
        .map(|r| r.steps)
        .or_else(|| row.sample.as_ref().map(|(k, _)| k.saturating_add(1)));
    let verdict = row.record.as_ref().and_then(|r| r.verdict);
    [
        if running {
            Cell::from("[ RUN... ]").style(bold(Color::Cyan))
        } else {
            Cell::from(mode)
        },
        Cell::from(steps.map_or_else(String::new, |s| {
            format!("{} steps", format_number(s))
        })),
        Cell::from(verdict.map_or("", verdict_label)),
    ]
}

/// The readout of the selected task, or the help line of the selection.
fn task_widget(app: &App) -> Paragraph<'static> {
    let text = match app.selected() {
        Some(Row::Task { id, .. }) => task_readout(app, id, app.tasks.get(&id)),
        _ => app.note.clone(),
    };
    Paragraph::new(Line::from(text)).block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(" [ TASK ] "),
    )
}

fn task_readout(app: &App, id: (u16, u16), row: Option<&TaskRow>) -> String {
    let mode = app.task_mode(id);
    let period = row.map_or(0, |t| t.period_ns);
    let mut text = format!(" mode: {}", mode_label(mode));
    if mode == Some(ExecMode::Periodic) {
        let _ =
            write!(text, " @ {}", format_duration(u64::from(period) / 1000));
    }
    let Some(row) = row else {
        return text;
    };
    if let Some((k, bytes)) = &row.sample {
        let hex = bytes.iter().take(8).fold(String::new(), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        });
        let _ = write!(text, " | sample {k}: {hex}");
    }
    if let Some(record) = &row.record {
        let _ = write!(
            text,
            " | {} | missed releases: {} | jitter: {}",
            run_status_text(record.status),
            record.missed_releases,
            format_duration(u64::from(record.jitter_max_ns) / 1000)
        );
    }
    text
}

fn log_widget(app: &App, height: u16) -> Paragraph<'_> {
    let inner = usize::from(height.saturating_sub(1));
    let skip = app.logs.len().saturating_sub(inner);
    Paragraph::new(
        app.logs
            .iter()
            .skip(skip)
            .map(|l| Line::from(l.as_str()))
            .collect::<Vec<_>>(),
    )
    .block(
        Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " [ TARGET LOGS ] (Autoscroll: ON) ",
                Style::default().add_modifier(Modifier::BOLD),
            )),
    )
}

fn footer_line(app: &App) -> Line<'_> {
    match app.mode {
        Mode::Filter => prompt(
            " Filter: ",
            Color::Yellow,
            &app.filter,
            " (Enter: apply, Esc: clear)",
        ),
        Mode::Edit => prompt(
            " Write setting: ",
            Color::Magenta,
            &app.edit,
            " (Enter: send, Esc: cancel)",
        ),
        Mode::Normal => Line::from(vec![
            Span::styled(" (f)", bold(Color::Cyan)),
            Span::raw("ilter | "),
            Span::styled("(r)", bold(Color::Cyan)),
            Span::raw("un all | "),
            Span::styled("(p)", bold(Color::Cyan)),
            Span::raw("settings | "),
            Span::styled("(m)", bold(Color::Cyan)),
            Span::raw("ode | "),
            Span::styled("([ ])", bold(Color::Cyan)),
            Span::raw("period | "),
            Span::styled("(s)", bold(Color::Cyan)),
            Span::raw("top | "),
            Span::styled("(d)", bold(Color::Cyan)),
            Span::raw("escribe | "),
            Span::styled("(q)", bold(Color::Cyan)),
            Span::raw("uit"),
        ]),
    }
}

fn prompt<'a>(
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

#[cfg(test)]
mod tests {
    use control_rs_ets::wire::Param;
    use control_rs_ets_host::bridge::PacketDecl;
    use control_rs_ets_host::catalog::{
        Catalog, ExecEntry, SettingEntry, SuiteEntry, TargetDetails,
    };
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use super::*;
    use crate::worker::Update;

    fn app() -> App {
        let mut app = App::new("QEMU".to_string(), "pipe".to_string());
        let exec = |name: &str, modes| {
            Some(ExecEntry {
                description: String::new(),
                input: PacketDecl::default(),
                link_timeout_ms: 0,
                modes,
                name: name.to_string(),
                output: PacketDecl::default(),
            })
        };
        app.on_update(Update::Catalog(
            Box::new(Catalog {
                suites: vec![SuiteEntry {
                    description: String::new(),
                    executables: vec![
                        exec("alpha", ExecMode::Atomic.bit()),
                        exec("loop", 0b1_1111),
                    ],
                    name: "pid".to_string(),
                    settings: vec![Some(SettingEntry {
                        description: String::new(),
                        index: 0,
                        max: Param::U8(9),
                        min: Param::U8(0),
                        name: "kp".to_string(),
                        value: Param::U8(3),
                    })],
                }],
            }),
            TargetDetails {
                board_id: 1,
                core_clock_hz: 1_000_000,
                fpu_flags: 0,
                protocol_version: 2,
                release_source: false,
            },
        ));
        app
    }

    fn screen(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content()
            .chunks(usize::from(terminal.backend().buffer().area.width))
            .map(|row| {
                row.iter()
                    .map(ratatui::buffer::Cell::symbol)
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn unchanged_frame_writes_nothing() {
        let mut terminal = Terminal::new(TestBackend::new(110, 32)).unwrap();
        let mut app = app();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let first = terminal.backend().buffer().clone();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        assert_eq!(first.diff(terminal.backend().buffer()).len(), 0);
    }

    #[test]
    fn the_tree_the_console_and_the_status_render() {
        let mut terminal = Terminal::new(TestBackend::new(110, 32)).unwrap();
        let mut app = app();
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let tests = screen(&terminal);
        assert!(tests.contains("▼ pid"));
        assert!(tests.contains("↻ loop"));
        assert!(tests.contains("[ IDLE ]"));
        assert!(tests.contains("Board 0x0001"));
        app.on_update(Update::Fault("boom".to_string()));
        app.on_key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Char('p'),
            crossterm::event::KeyModifiers::NONE,
        ));
        terminal.draw(|f| draw(f, &mut app)).unwrap();
        let settings = screen(&terminal);
        assert!(settings.contains("kp"));
        assert!(settings.contains("0 .. 9"));
        assert!(settings.contains("Faults: 1 (last: boom)"));
    }

    #[test]
    fn numbers_and_durations_are_formatted() {
        assert_eq!(format_number(1_204), "1,204");
        assert_eq!(format_number(12), "12");
        assert_eq!(format_duration(999), "999.00µs");
        assert_eq!(format_duration(1_000), "1.00ms");
        assert_eq!(format_duration(1_500_000), "1.50s");
        assert_eq!(mode_label(None), "-");
        assert_eq!(verdict_label(Verdict::Overrun), "overrun");
    }
}
