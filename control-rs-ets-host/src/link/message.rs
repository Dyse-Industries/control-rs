//! Whole-message host links.
//!
//! # Description
//!
//! A [`MessageLink`] is a [`HostLink`] over a transport that already delimits messages: USB bulk
//! transfers and ISO-TP sockets. A background thread reads whole messages, decodes them as
//! `Telemetry` and queues them for [`HostLink::poll`]. A user adds a host link for a new comms
//! device by implementing [`MessageReader`] and [`MessageWriter`] for it, or by implementing
//! [`HostLink`] directly, and rebuilding the host tools with it.
//!
//! # Core Concepts
//!
//! - **Reader**: [`MessageReader::next_message`] waits up to a short timeout and returns one
//!   whole message, `None` on a timeout or an error when the transport is gone.
//! - **Writer**: [`MessageWriter::write_message`] sends one whole message.
//! - **Shutdown**: Closing or dropping the link stops the reader thread.
//!
//! # Usage
//!
//! ```
//! use std::io;
//! use control_rs_ets::wire::Command;
//! use control_rs_ets_host::link::{HostLink, MessageLink, MessageReader, MessageWriter};
//!
//! struct Silent;
//! impl MessageReader for Silent {
//!     fn next_message(&mut self) -> io::Result<Option<Vec<u8>>> { Ok(None) }
//! }
//! impl MessageWriter for Silent {
//!     fn write_message(&mut self, _: &[u8]) -> io::Result<()> { Ok(()) }
//! }
//!
//! let mut link = MessageLink::new(Silent, Silent, "silent".to_string());
//! link.send(&Command::ListSuites)?;
//! link.close();
//! # Ok::<(), control_rs_ets_host::HostError>(())
//! ```

use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError, channel};
use std::thread::{self, JoinHandle};

use control_rs_ets::wire::{Command, MAX_MESSAGE_SIZE, Telemetry};

use super::{FlagOf, HostLink, LinkEvent, Polled, UnitOf};
use crate::bridge::BridgeMessage;
use crate::error::HostError;

/// Result of waiting for a message: a message, a timeout (`None`) or a transport error.
pub type MessageOf = io::Result<Option<Vec<u8>>>;

type Events = Receiver<BridgeMessage>;
type ReaderHandle = Option<JoinHandle<()>>;
type Shutdown = Arc<AtomicBool>;

/// Reads whole messages from a transport.
pub trait MessageReader: Send + 'static {
    /// Waits up to a short timeout for the next whole message.
    ///
    /// # Errors
    /// Returns an error when the transport is gone; the link then reports itself closed.
    fn next_message(&mut self) -> MessageOf;
}

/// Writes whole messages to a transport.
pub trait MessageWriter: Send + 'static {
    /// Sends one whole message.
    ///
    /// # Errors
    /// Returns an error when the message cannot be written.
    fn write_message(&mut self, msg: &[u8]) -> io::Result<()>;
}

/// A [`HostLink`] over a message-delimited transport.
pub struct MessageLink<W> {
    description: String,
    events: Events,
    reader: ReaderHandle,
    shutdown: Shutdown,
    writer: W,
}

impl<W: MessageWriter> MessageLink<W> {
    /// Starts the reader thread over `reader` and returns the link.
    #[must_use]
    pub fn new<R: MessageReader>(
        mut reader: R,
        writer: W,
        description: String,
    ) -> Self {
        let (tx, events) = channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let stop = Arc::clone(&shutdown);
        let handle = thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                let msg = match reader.next_message() {
                    Ok(Some(bytes)) => decode(&bytes),
                    Ok(None) => continue,
                    Err(e) => {
                        let _ = tx.send(BridgeMessage::RawConsole(format!(
                            "link closed: {e}"
                        )));
                        break;
                    }
                };
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });
        Self {
            description,
            events,
            reader: Some(handle),
            shutdown,
            writer,
        }
    }
}

impl<W: MessageWriter> HostLink for MessageLink<W> {
    type Error = HostError;

    fn send(&mut self, cmd: &Command<'_>) -> UnitOf<Self> {
        let mut buf = [0u8; MAX_MESSAGE_SIZE];
        let used = postcard::to_slice(cmd, &mut buf)
            .map_err(|e| HostError::Transport {
                source: format!("Failed to serialize command: {e}").into(),
            })?
            .len();
        let msg = buf.get(..used).unwrap_or(&[]);
        self.writer
            .write_message(msg)
            .map_err(|e| HostError::Transport {
                source: format!("I/O failure sending command: {e}").into(),
            })
    }

    fn poll(&mut self) -> Polled {
        match self.events.try_recv() {
            Ok(msg) => Polled::Event(LinkEvent::from(msg)),
            Err(TryRecvError::Empty) => Polled::Idle,
            Err(TryRecvError::Disconnected) => Polled::Closed,
        }
    }

    fn exited(&mut self) -> FlagOf<Self> {
        Ok(self.reader.as_ref().is_none_or(JoinHandle::is_finished))
    }

    fn close(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.reader.take() {
            let _ = handle.join();
        }
    }

    fn describe(&self) -> &str {
        &self.description
    }
}

impl<W> Drop for MessageLink<W> {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.reader.take() {
            let _ = handle.join();
        }
    }
}

/// Decodes one message as telemetry.
fn decode(bytes: &[u8]) -> BridgeMessage {
    match postcard::from_bytes::<Telemetry<'_>>(bytes) {
        Ok(tel) => BridgeMessage::telemetry(&tel),
        Err(e) => BridgeMessage::RawConsole(format!(
            "Postcard decode failed: {e} (payload: {bytes:02X?})"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::mpsc::Sender;
    use std::time::Duration;

    use crate::bridge::OwnedTelemetry;

    type Feed = Receiver<Vec<u8>>;
    type Recorded = Arc<Mutex<Vec<Vec<u8>>>>;
    type Fixture = (MessageLink<Recorder>, Sender<Vec<u8>>, Recorded);

    /// A reader fed by a channel; a hang-up ends the stream.
    struct ChannelReader(Feed);

    /// A writer that records messages.
    struct Recorder(Recorded);

    impl MessageReader for ChannelReader {
        fn next_message(&mut self) -> MessageOf {
            match self.0.recv_timeout(Duration::from_millis(5)) {
                Ok(m) => Ok(Some(m)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => Ok(None),
                Err(_) => Err(io::Error::from(io::ErrorKind::BrokenPipe)),
            }
        }
    }

    impl MessageWriter for Recorder {
        fn write_message(&mut self, msg: &[u8]) -> io::Result<()> {
            self.0.lock().unwrap().push(msg.to_vec());
            Ok(())
        }
    }

    fn link() -> Fixture {
        let (tx, rx) = channel();
        let sent = Arc::new(Mutex::new(Vec::new()));
        let link = MessageLink::new(
            ChannelReader(rx),
            Recorder(Arc::clone(&sent)),
            "memory".to_string(),
        );
        (link, tx, sent)
    }

    fn wait_event(link: &mut MessageLink<Recorder>) -> Polled {
        for _ in 0..400 {
            match link.poll() {
                Polled::Idle => thread::sleep(Duration::from_millis(5)),
                other => return other,
            }
        }
        Polled::Idle
    }

    fn encoded(tel: &Telemetry<'_>) -> Vec<u8> {
        let mut buf = [0u8; MAX_MESSAGE_SIZE];
        postcard::to_slice(tel, &mut buf).unwrap().to_vec()
    }

    #[test]
    fn commands_go_out_whole_and_telemetry_comes_in_decoded() {
        let (mut link, tx, sent) = link();
        link.send(&Command::ListSuites).unwrap();
        assert_eq!(sent.lock().unwrap().clone(), vec![vec![0u8]]);
        tx.send(encoded(&Telemetry::DiscoveryComplete)).unwrap();
        assert_eq!(
            wait_event(&mut link),
            Polled::Event(LinkEvent::Telemetry(
                OwnedTelemetry::DiscoveryComplete
            ))
        );
        assert_eq!(link.describe(), "memory");
        assert!(!link.exited().unwrap());
    }

    #[test]
    fn an_undecodable_message_is_reported_as_console_text() {
        let (mut link, tx, _) = link();
        tx.send(vec![0xFF, 0xFF, 0xFF]).unwrap();
        let Polled::Event(LinkEvent::Console(text)) = wait_event(&mut link)
        else {
            panic!("expected console text");
        };
        assert!(text.contains("Postcard decode failed"));
    }

    #[test]
    fn a_lost_transport_closes_the_link() {
        let (mut link, tx, _) = link();
        drop(tx);
        assert!(matches!(
            wait_event(&mut link),
            Polled::Event(LinkEvent::Console(_))
        ));
        assert_eq!(wait_event(&mut link), Polled::Closed);
        assert!(link.exited().unwrap());
    }

    #[test]
    fn dropping_the_link_stops_and_joins_the_reader() {
        let (link, tx, _) = link();
        drop(link);
        assert!(tx.send(Vec::new()).is_err(), "reader still holds the feed");
    }

    #[test]
    fn closing_stops_the_reader() {
        let (mut link, _tx, _) = link();
        link.close();
        assert!(link.exited().unwrap());
    }
}
