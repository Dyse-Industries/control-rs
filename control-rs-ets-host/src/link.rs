//! Host-side link seam between a session and the target.
//!
//! [`HostLink`] is the host counterpart of the target `Link`. It moves whole
//! commands to the target and whole events back, and hides the transport
//! (QEMU subprocess, serial port, in-memory fake). [`ETSBridge`] implements
//! it, so code written against the trait runs on every existing transport.
//!
//! # Examples
//!
//! ```
//! use control_rs_ets::comms::Command;
//! use control_rs_ets_host::link::{HostLink, discover};
//!
//! fn start<L: HostLink>(link: &mut L) -> Result<(), L::Error> {
//!     discover(link)
//! }
//!
//! # fn check(link: &mut control_rs_ets_host::ETSBridge) {
//! let _ = start(link);
//! # }
//! # let _ = Command::ListSuites;
//! ```

use std::sync::mpsc::TryRecvError;

use control_rs_ets::comms::Command;

use crate::bridge::{BridgeMessage, ETSBridge, OwnedTelemetry};
use crate::error::HostError;

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn discover_sends_reset_then_list() {
        let mut fake = ETSBridge::fake(None);
        discover(&mut fake.bridge).unwrap();
        let written = fake.written.lock().unwrap().clone();
        let mut expected = Vec::new();
        for cmd in [Command::TryReset, Command::ListSuites] {
            let mut buf = [0u8; 64];
            let n = control_rs_ets::comms::FrameEncoder::frame_command(
                &cmd, &mut buf,
            )
            .unwrap();
            expected.extend_from_slice(buf.get(..n).unwrap());
        }
        assert_eq!(written, expected);
    }

    #[test]
    fn poll_reports_events_idle_and_closed() {
        let mut fake = ETSBridge::fake(None);
        assert_eq!(fake.bridge.poll(), Polled::Idle);
        fake.tx
            .send(BridgeMessage::RawConsole("hi".to_string()))
            .unwrap();
        assert_eq!(
            fake.bridge.poll(),
            Polled::Event(LinkEvent::Console("hi".to_string()))
        );
        fake.tx
            .send(BridgeMessage::Telemetry(OwnedTelemetry::DiscoveryComplete))
            .unwrap();
        assert_eq!(
            fake.bridge.poll(),
            Polled::Event(LinkEvent::Telemetry(
                OwnedTelemetry::DiscoveryComplete
            ))
        );
        drop(fake.tx);
        assert_eq!(fake.bridge.poll(), Polled::Closed);
    }

    #[test]
    fn exited_follows_the_fake_exit_budget() {
        let mut fake = ETSBridge::fake(Some(2));
        assert!(!fake.bridge.exited().unwrap());
        assert!(fake.bridge.exited().unwrap());
    }

    #[test]
    fn close_shuts_the_bridge_down() {
        let mut fake = ETSBridge::fake(None);
        fake.bridge.close();
        assert!(fake.bridge.is_shut_down());
        assert_eq!(fake.bridge.describe(), "fake");
    }
}

/// Outcome of a link operation that returns no value.
pub type UnitOf<L> = Result<(), <L as HostLink>::Error>;

/// Outcome of a link status query.
pub type FlagOf<L> = Result<bool, <L as HostLink>::Error>;

/// One event delivered by a [`HostLink`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkEvent {
    /// Console text that is not a protocol frame.
    Console(String),
    /// A decoded telemetry message.
    Telemetry(OwnedTelemetry),
}

/// Result of a non-blocking [`HostLink::poll`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Polled {
    /// An event is available.
    Event(LinkEvent),
    /// No event is pending.
    Idle,
    /// The target side of the link is gone and no events remain.
    Closed,
}

/// Host-side message link to a target.
pub trait HostLink {
    /// Transport error.
    type Error;

    /// Sends one command to the target.
    ///
    /// # Errors
    ///
    /// Returns the transport error when the command cannot be written.
    fn send(&mut self, cmd: &Command) -> UnitOf<Self>;

    /// Returns the next pending event without blocking.
    fn poll(&mut self) -> Polled;

    /// Whether the target process or device is known to have exited.
    ///
    /// # Errors
    ///
    /// Returns the transport error when the status cannot be queried.
    fn exited(&mut self) -> FlagOf<Self>;

    /// Releases the transport. Further sends fail or are dropped.
    fn close(&mut self);

    /// Human-readable description of the link.
    fn describe(&self) -> &str;
}

impl From<BridgeMessage> for LinkEvent {
    fn from(msg: BridgeMessage) -> Self {
        match msg {
            BridgeMessage::RawConsole(text) => Self::Console(text),
            BridgeMessage::Telemetry(tel) => Self::Telemetry(tel),
        }
    }
}

impl HostLink for ETSBridge {
    type Error = HostError;

    fn send(&mut self, cmd: &Command) -> Result<(), HostError> {
        self.send_command(cmd)
    }

    fn poll(&mut self) -> Polled {
        match self.receiver().try_recv() {
            Ok(msg) => Polled::Event(msg.into()),
            Err(TryRecvError::Empty) => Polled::Idle,
            Err(TryRecvError::Disconnected) => Polled::Closed,
        }
    }

    fn exited(&mut self) -> Result<bool, HostError> {
        self.try_wait().map(|status| status.is_some()).map_err(|e| {
            HostError::Transport {
                source: format!("failed to query target status: {e}").into(),
            }
        })
    }

    fn close(&mut self) {
        self.terminate();
    }

    fn describe(&self) -> &str {
        self.link_info()
    }
}

/// Sends cooperative reset then suite discovery.
///
/// `handle_failure` on the target waits only for `TryReset`. After a panic
/// reconnect, `ListSuites` alone leaves a serial target spinning forever if
/// the earlier `TryReset` was lost when the link closed. `TryReset` is a no-op
/// in the normal server command loop, so pairing it with every discovery
/// attempt is safe.
///
/// # Errors
///
/// Returns the link error when either command cannot be sent.
pub fn discover<L: HostLink>(link: &mut L) -> UnitOf<L> {
    link.send(&Command::TryReset)?;
    link.send(&Command::ListSuites)
}
