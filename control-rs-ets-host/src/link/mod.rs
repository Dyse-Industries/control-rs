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
//! use control_rs_ets::wire::Command;
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

use control_rs_ets::wire::Command;

use crate::bridge::{BridgeMessage, ETSBridge, OwnedTelemetry};
use crate::error::HostError;
pub use message::{MessageLink, MessageReader, MessageWriter};

#[cfg(target_os = "linux")]
pub mod can;
pub mod message;

#[cfg(all(test, unix))]
mod tests {
    use control_rs_ets::wire::Telemetry;

    use super::*;

    type Messages = std::sync::mpsc::Receiver<Vec<u8>>;
    type MessageSender = std::sync::mpsc::Sender<Vec<u8>>;
    type PairOf = (MessageLink<QueueWriter>, MessageSender, Messages);

    /// A target-side byte stream over in-memory buffers.
    struct Pipe {
        inbound: std::collections::VecDeque<u8>,
        outbound: Vec<u8>,
    }

    /// Delivers whole messages that a test "transport" assembled.
    struct QueueReader(Messages);

    /// Collects the messages the host link writes.
    struct QueueWriter(MessageSender);

    impl control_rs_ets::link::ByteStream for Pipe {
        type Error = ();

        fn read(
            &mut self,
            buf: &mut [u8],
        ) -> control_rs_ets::link::CountOf<Self> {
            let mut n = 0_usize;
            for slot in buf.iter_mut() {
                let Some(b) = self.inbound.pop_front() else {
                    break;
                };
                *slot = b;
                n = n.saturating_add(1);
            }
            Ok(n)
        }

        fn write(&mut self, buf: &[u8]) -> control_rs_ets::link::CountOf<Self> {
            self.outbound.extend_from_slice(buf);
            Ok(buf.len())
        }
    }

    impl MessageReader for QueueReader {
        fn next_message(&mut self) -> std::io::Result<Option<Vec<u8>>> {
            Ok(self
                .0
                .recv_timeout(std::time::Duration::from_millis(5))
                .ok())
        }
    }

    impl MessageWriter for QueueWriter {
        fn write_message(&mut self, msg: &[u8]) -> std::io::Result<()> {
            self.0.send(msg.to_vec()).map_err(|_| {
                std::io::Error::from(std::io::ErrorKind::BrokenPipe)
            })
        }
    }

    #[test]
    fn discover_sends_reset_then_list() {
        let mut fake = ETSBridge::fake(None);
        discover(&mut fake.bridge).unwrap();
        let written = fake.written.lock().unwrap().clone();
        let mut expected = Vec::new();
        for cmd in [Command::TryReset, Command::ListSuites] {
            let mut payload = [0u8; 64];
            let used = postcard::to_slice(&cmd, &mut payload).unwrap().len();
            let mut buf = [0u8; 80];
            let n = control_rs_ets::link::stream::encode_frame(
                payload.get(..used).unwrap(),
                &mut buf,
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

    fn encode_telemetry(tel: &Telemetry<'_>) -> Vec<u8> {
        let mut buf = [0u8; 512];
        postcard::to_slice(tel, &mut buf)
            .map(|b| b.to_vec())
            .unwrap_or_default()
    }

    fn command_bytes(cmd: &Command<'_>) -> Vec<u8> {
        let mut buf = [0u8; 512];
        postcard::to_slice(cmd, &mut buf)
            .map(|b| b.to_vec())
            .unwrap_or_default()
    }

    #[test]
    fn host_target_pairs_interoperate_stream() {
        use control_rs_ets::link::{Link, StreamLink};
        let mut fake = ETSBridge::fake(None);
        fake.bridge.send(&Command::ListSuites).unwrap();
        let wire = fake.written.lock().unwrap().clone();
        let mut target = StreamLink::new(Pipe {
            inbound: wire.into(),
            outbound: Vec::new(),
        });
        let mut buf = [0u8; 512];
        let mut got = None;
        for _ in 0..16 {
            if let Some(n) = target.recv(&mut buf).unwrap() {
                got = Some(buf.get(..n).unwrap().to_vec());
            }
        }
        assert_eq!(got, Some(command_bytes(&Command::ListSuites)));

        let tel = Telemetry::DiscoveryComplete;
        assert_eq!(target.send(&encode_telemetry(&tel)), Ok(true));
        let (tx, rx) = std::sync::mpsc::channel();
        let mut reader = control_rs_ets::link::stream::FrameReader::new();
        let mut raw = Vec::new();
        for &b in &target.stream().outbound {
            crate::bridge::process_incoming_byte(b, &mut reader, &mut raw, &tx);
        }
        assert!(matches!(
            rx.try_recv().unwrap(),
            BridgeMessage::Telemetry(OwnedTelemetry::DiscoveryComplete)
        ));
    }

    fn message_pair() -> PairOf {
        let (to_host, host_rx) = std::sync::mpsc::channel();
        let (host_tx, from_host) = std::sync::mpsc::channel();
        let link = MessageLink::new(
            QueueReader(host_rx),
            QueueWriter(host_tx),
            "pair".to_string(),
        );
        (link, to_host, from_host)
    }

    fn next_event(link: &mut MessageLink<QueueWriter>) -> Polled {
        for _ in 0..400 {
            match link.poll() {
                Polled::Idle => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                other => return other,
            }
        }
        Polled::Idle
    }

    #[test]
    fn host_target_pairs_interoperate_usb() {
        use control_rs_ets::link::usb::{Assembler, Packets};
        let (mut link, to_host, from_host) = message_pair();
        let cmd = Command::TaskInput {
            suite: 0,
            item: 1,
            seq: 3,
            payload: &[7; 100],
        };
        link.send(&cmd).unwrap();
        let sent = from_host.recv().unwrap();
        let mut target = Assembler::<512>::new();
        let mut got = None;
        for packet in Packets::new(&sent, 64) {
            got = target.push(packet, 64).unwrap().map(<[u8]>::to_vec);
        }
        assert_eq!(got, Some(command_bytes(&cmd)));

        let tel = encode_telemetry(&Telemetry::DiscoveryComplete);
        let mut host_side = Assembler::<512>::new();
        for packet in Packets::new(&tel, 64) {
            if let Some(m) = host_side.push(packet, 64).unwrap() {
                to_host.send(m.to_vec()).unwrap();
            }
        }
        assert_eq!(
            next_event(&mut link),
            Polled::Event(LinkEvent::Telemetry(
                OwnedTelemetry::DiscoveryComplete
            ))
        );
    }

    #[test]
    fn host_target_pairs_interoperate_can() {
        use control_rs_ets::link::can::{Reassembler, Rx, Segmenter};
        let (mut link, to_host, from_host) = message_pair();
        let cmd = Command::TaskInput {
            suite: 0,
            item: 1,
            seq: 3,
            payload: &[9; 100],
        };
        link.send(&cmd).unwrap();
        let sent = from_host.recv().unwrap();
        let mut tx = Segmenter::new(&sent).unwrap();
        let mut rx = Reassembler::<512>::new();
        let mut frame = [0u8; 8];
        let mut got = None;
        while !tx.is_done() {
            while let Some(n) = tx.next_frame(&mut frame) {
                match rx.push(frame.get(..n).unwrap()).unwrap() {
                    Rx::Message(m) => got = Some(m.to_vec()),
                    Rx::FlowControl(fc) => tx.flow_control(&fc).unwrap(),
                    Rx::None => {}
                }
            }
        }
        assert_eq!(got, Some(command_bytes(&cmd)));
        to_host
            .send(encode_telemetry(&Telemetry::DiscoveryComplete))
            .unwrap();
        assert_eq!(
            next_event(&mut link),
            Polled::Event(LinkEvent::Telemetry(
                OwnedTelemetry::DiscoveryComplete
            ))
        );
    }

    #[test]
    fn custom_link_drives_discovery() {
        /// A user-defined link that records what the tools send it.
        struct Custom(Vec<String>);

        impl HostLink for Custom {
            type Error = HostError;

            fn send(&mut self, cmd: &Command<'_>) -> UnitOf<Self> {
                self.0.push(format!("{cmd:?}"));
                Ok(())
            }

            fn poll(&mut self) -> Polled {
                Polled::Idle
            }

            fn exited(&mut self) -> FlagOf<Self> {
                Ok(false)
            }

            fn close(&mut self) {}

            fn describe(&self) -> &'static str {
                "custom"
            }
        }

        let mut link = Custom(Vec::new());
        discover(&mut link).unwrap();
        assert_eq!(link.0, vec!["TryReset", "ListSuites"]);
    }
}

pub mod usb;

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
    fn send(&mut self, cmd: &Command<'_>) -> UnitOf<Self>;

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

    fn send(&mut self, cmd: &Command<'_>) -> Result<(), HostError> {
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
