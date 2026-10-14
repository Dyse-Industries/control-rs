//! Key bindings: one key, one command.

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

/// What the keyboard is feeding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// A setting value is being typed.
    Edit,
    /// The filter query is being typed.
    Filter,
    /// Keys are commands.
    Normal,
}

/// A command bound to a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Delete the last typed character.
    Backspace,
    /// Leave a prompt without applying it.
    Cancel,
    /// A typed character.
    Char(char),
    /// Show the description of the selected setting.
    Describe,
    /// Move the selection down.
    Down,
    /// Start the selected case or task, edit the selected setting or fold the suite.
    Enter,
    /// Open the filter prompt.
    Filter,
    /// Halve the release period of the selected task.
    PeriodDown,
    /// Double the release period of the selected task.
    PeriodUp,
    /// Leave the dashboard.
    Quit,
    /// Run every case.
    RunAll,
    /// Stop the running task or sequence.
    Stop,
    /// Cycle the mode of the selected task.
    TaskMode,
    /// Fold or unfold the selected suite.
    Toggle,
    /// Move the selection up.
    Up,
    /// Switch between the test tree and the setting console.
    View,
}

/// The command `key` means in `mode`, if any.
#[must_use]
pub fn command(key: KeyEvent, mode: Mode) -> Option<Command> {
    if key.kind == KeyEventKind::Release {
        return None;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL)
        && key.code == KeyCode::Char('c')
    {
        return Some(Command::Quit);
    }
    match mode {
        Mode::Normal => normal(key.code),
        Mode::Filter | Mode::Edit => match key.code {
            KeyCode::Char(c) => Some(Command::Char(c)),
            KeyCode::Backspace => Some(Command::Backspace),
            KeyCode::Enter => Some(Command::Enter),
            KeyCode::Esc => Some(Command::Cancel),
            _ => None,
        },
    }
}

/// Commands in normal mode.
const fn normal(code: KeyCode) -> Option<Command> {
    match code {
        KeyCode::Char('q') => Some(Command::Quit),
        KeyCode::Char('r') => Some(Command::RunAll),
        KeyCode::Char('s') => Some(Command::Stop),
        KeyCode::Char('f') => Some(Command::Filter),
        KeyCode::Char('d') => Some(Command::Describe),
        KeyCode::Char('m') => Some(Command::TaskMode),
        KeyCode::Char('[') => Some(Command::PeriodDown),
        KeyCode::Char(']') => Some(Command::PeriodUp),
        KeyCode::Char('p') | KeyCode::Tab => Some(Command::View),
        KeyCode::Up | KeyCode::Char('k') => Some(Command::Up),
        KeyCode::Down | KeyCode::Char('j') => Some(Command::Down),
        KeyCode::Left | KeyCode::Right | KeyCode::Char(' ') => {
            Some(Command::Toggle)
        }
        KeyCode::Enter => Some(Command::Enter),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    #[test]
    fn key_to_command() {
        for (code, expected) in [
            (KeyCode::Char('q'), Command::Quit),
            (KeyCode::Char('r'), Command::RunAll),
            (KeyCode::Char('s'), Command::Stop),
            (KeyCode::Char('f'), Command::Filter),
            (KeyCode::Char('p'), Command::View),
            (KeyCode::Tab, Command::View),
            (KeyCode::Enter, Command::Enter),
            (KeyCode::Char('j'), Command::Down),
            (KeyCode::Up, Command::Up),
            (KeyCode::Char(' '), Command::Toggle),
            (KeyCode::Char('m'), Command::TaskMode),
            (KeyCode::Char('['), Command::PeriodDown),
            (KeyCode::Char(']'), Command::PeriodUp),
            (KeyCode::Char('d'), Command::Describe),
        ] {
            assert_eq!(command(press(code), Mode::Normal), Some(expected));
        }
        assert_eq!(command(press(KeyCode::Char('x')), Mode::Normal), None);
    }

    #[test]
    fn prompts_take_text_and_control_c_always_quits() {
        for mode in [Mode::Filter, Mode::Edit] {
            assert_eq!(
                command(press(KeyCode::Char('q')), mode),
                Some(Command::Char('q'))
            );
            assert_eq!(
                command(press(KeyCode::Esc), mode),
                Some(Command::Cancel)
            );
            assert_eq!(
                command(press(KeyCode::Backspace), mode),
                Some(Command::Backspace)
            );
            assert_eq!(command(press(KeyCode::Up), mode), None);
        }
        let ctrl_c = KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL);
        assert_eq!(command(ctrl_c, Mode::Edit), Some(Command::Quit));
        let mut release = press(KeyCode::Char('q'));
        release.kind = KeyEventKind::Release;
        assert_eq!(command(release, Mode::Normal), None);
    }
}
