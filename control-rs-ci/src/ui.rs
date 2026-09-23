//! Terminal and cargo-style status line formatting utilities.

use std::env;
use std::fmt;

use anstream::ColorChoice;
use anstyle::AnsiColor;

/// Cargo status-line width: right-aligned 12-char verb.
pub const CARGO_STATUS_WIDTH: usize = 12;

/// Cargo `ERROR`: bright red bold diagnostic prefix / status verb.
pub const ERROR: anstyle::Style = AnsiColor::BrightRed.on_default().bold();

/// Muted ANSI color palette for multi-group concurrent execution.
///
/// Uses standard (less bright) ANSI colors. Does not include `Magenta`, which is
/// reserved exclusively for the built-in `exclusive` group.
pub const GROUP_PALETTE: [anstyle::AnsiColor; 5] = [
    AnsiColor::Cyan,
    AnsiColor::BrightCyan,
    AnsiColor::Blue,
    AnsiColor::BrightBlue,
    AnsiColor::Green,
];

/// Cargo status `HEADER`: bright green bold verb (for example, `Running`, `Finished`, `Writing`).
pub const HEADER: anstyle::Style = AnsiColor::BrightGreen.on_default().bold();

/// Help arg / placeholder style (bright yellow).
pub const HELP_ARG: anstyle::Style = AnsiColor::BrightYellow.on_default();

/// Help flag / subcommand / literal style (cyan).
pub const HELP_FLAG: anstyle::Style = AnsiColor::Cyan.on_default();

/// Help header style (bright green bold).
pub const HELP_HEADER: anstyle::Style =
    AnsiColor::BrightGreen.on_default().bold();

/// Legacy alias for `GROUP_PALETTE`.
pub const LANE_PALETTE: [anstyle::AnsiColor; 5] = GROUP_PALETTE;

/// Cargo status `STATUS_INFO`: cyan bold verb (for example, `Skipping`, `Listing`).
pub const STATUS_INFO: anstyle::Style = AnsiColor::Cyan.on_default().bold();

/// Cargo `WARNING`: bright yellow bold diagnostic prefix / status verb.
pub const WARNING: anstyle::Style = AnsiColor::BrightYellow.on_default().bold();

/// Returns a deterministic ANSI color styling from the palette by active group index.
#[must_use]
pub fn group_style(index: usize) -> anstyle::Style {
    let slot = index.checked_rem(GROUP_PALETTE.len()).unwrap_or(0);
    GROUP_PALETTE
        .get(slot)
        .copied()
        .unwrap_or(AnsiColor::Cyan)
        .on_default()
}

/// Legacy alias for `group_style`.
#[must_use]
pub fn lane_style(index: usize) -> anstyle::Style {
    group_style(index)
}

/// Returns the distinct ANSI color styling for the built-in exclusive group.
///
/// Uses standard Magenta, giving exclusive execution a unique, non-colliding color.
#[must_use]
pub const fn exclusive_style() -> anstyle::Style {
    AnsiColor::Magenta.on_default()
}

/// Formats a group tag with brackets and color (for example, `[cargo] ` or `[exclusive] `).
///
/// Returns an empty string if `group` is `None` (a gate outside any group) or empty.
#[must_use]
pub fn format_group_tag(
    group: Option<&str>,
    style: Option<anstyle::Style>,
) -> String {
    match (group, style) {
        (Some(name), Some(st)) if !name.is_empty() => {
            format!("{st}[{name}]{st:#} ")
        }
        (Some(name), None) if !name.is_empty() => {
            format!("[{name}] ")
        }
        _ => String::new(),
    }
}

/// Legacy alias for `format_group_tag`.
#[must_use]
pub fn format_lane_tag(
    lane: Option<&str>,
    style: Option<anstyle::Style>,
) -> String {
    format_group_tag(lane, style)
}

/// Apply cargo-compatible color choice to anstream's global default.
///
/// `CARGO_TERM_COLOR` is cargo-specific and is not read by `anstream` itself.
/// `always` / `never` override; any other value (including unset) is `auto`,
/// which honors `NO_COLOR`, `CLICOLOR`, `TERM=dumb`, and TTY detection.
pub fn init_color() {
    cargo_color_choice().write_global();
}

fn parse_cargo_color_choice(val: Option<&str>) -> ColorChoice {
    match val {
        Some("always") => ColorChoice::Always,
        Some("never") => ColorChoice::Never,
        _ => ColorChoice::Auto,
    }
}

fn cargo_color_choice() -> ColorChoice {
    let var = env::var("CARGO_TERM_COLOR").ok();
    parse_cargo_color_choice(var.as_deref())
}

/// Formats a cargo-style status line without color.
#[must_use]
pub fn format_status(status: &str, msg: impl fmt::Display) -> String {
    format!("{status:>CARGO_STATUS_WIDTH$} {msg}")
}

/// Prints a cargo-style green status line to stderr (for example, `     Running gate`).
pub fn status(status_verb: &str, msg: impl fmt::Display) {
    init_color();
    anstream::eprintln!(
        "{HEADER}{status_verb:>CARGO_STATUS_WIDTH$}{HEADER:#} {msg}"
    );
}

/// Prints a cargo-style cyan info line to stderr.
pub fn status_info(status_verb: &str, msg: impl fmt::Display) {
    init_color();
    anstream::eprintln!(
        "{STATUS_INFO}{status_verb:>CARGO_STATUS_WIDTH$}{STATUS_INFO:#} {msg}"
    );
}

/// Prints a cargo-style yellow warning line to stderr.
pub fn warning(status_verb: &str, msg: impl fmt::Display) {
    init_color();
    anstream::eprintln!(
        "{WARNING}{status_verb:>CARGO_STATUS_WIDTH$}{WARNING:#} {msg}"
    );
}

/// Prints a cargo-style red failure line to stderr.
pub fn failure(status_verb: &str, msg: impl fmt::Display) {
    init_color();
    anstream::eprintln!(
        "{ERROR}{status_verb:>CARGO_STATUS_WIDTH$}{ERROR:#} {msg}"
    );
}

/// Formats the verbose echo prefix for a gate (for example, `[verify] cross-compare | `).
#[must_use]
pub fn format_echo_prefix(tag: &str, gate: &str) -> String {
    let dim = anstyle::Style::new().dimmed();
    format!("{tag}{dim}{gate} |{dim:#} ")
}

/// Prints one line of gate output to stderr behind its echo prefix.
pub fn gate_output(prefix: &str, line: &str) {
    init_color();
    anstream::eprintln!("{prefix}{line}");
}

/// Prints a standard `error: {msg}` diagnostic to stderr.
pub fn error(msg: impl fmt::Display) {
    init_color();
    anstream::eprintln!("{ERROR}error{ERROR:#}: {msg}");
}

/// Prints a standard `warning: {msg}` diagnostic to stderr.
pub fn warn_diag(msg: impl fmt::Display) {
    init_color();
    anstream::eprintln!("{WARNING}warning{WARNING:#}: {msg}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cargo_status_12_column_width() {
        assert_eq!(format_status("Running", "fmt"), "     Running fmt");
        assert_eq!(
            format_status("Finished", "ci-pipeline"),
            "    Finished ci-pipeline"
        );
        assert_eq!(
            format_status("Passed", "test in 0.5s"),
            "      Passed test in 0.5s"
        );
    }

    #[test]
    fn test_cargo_color_choice() {
        assert_eq!(
            parse_cargo_color_choice(Some("always")),
            ColorChoice::Always
        );
        assert_eq!(parse_cargo_color_choice(Some("never")), ColorChoice::Never);
        assert_eq!(parse_cargo_color_choice(Some("auto")), ColorChoice::Auto);
        assert_eq!(parse_cargo_color_choice(None), ColorChoice::Auto);
        assert_eq!(
            parse_cargo_color_choice(Some("unknown")),
            ColorChoice::Auto
        );
    }

    #[test]
    fn test_lane_tags_and_styles() {
        assert_eq!(format_lane_tag(None, None), "");
        assert_eq!(format_lane_tag(Some(""), None), "");
        assert_eq!(format_lane_tag(Some("exclusive"), None), "[exclusive] ");
        assert_eq!(format_lane_tag(Some("cargo"), None), "[cargo] ");

        let excl_style = exclusive_style();
        let excl_tag = format_group_tag(Some("exclusive"), Some(excl_style));
        assert!(excl_tag.contains("[exclusive]"));

        let style0 = lane_style(0);
        let tag0 = format_lane_tag(Some("cargo"), Some(style0));
        assert!(tag0.contains("[cargo]"));
        assert!(tag0.ends_with(' '));

        let style1 = lane_style(1);
        let tag1 = format_lane_tag(Some("audit"), Some(style1));
        assert!(tag1.contains("[audit]"));
        assert!(tag1.ends_with(' '));
        assert_ne!(tag0, tag1);

        let style2 = lane_style(2);
        let tag2 = format_lane_tag(Some("static"), Some(style2));
        assert!(tag2.contains("[static]"));
        assert!(tag2.ends_with(' '));
        assert_ne!(tag1, tag2);

        // Verify exclusive has its own distinct styling that does not collide with groups
        assert_ne!(excl_tag, tag0);
        assert_ne!(excl_tag, tag1);
        assert_ne!(excl_tag, tag2);
    }
}
