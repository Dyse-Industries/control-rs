//! CAN host link over Linux ISO-TP sockets.
//!
//! # Description
//!
//! [`open`] binds a `SocketCAN` `ISO-TP` socket (`AF_CAN`, `CAN_ISOTP`) and returns a [`MessageLink`].
//! The kernel segments messages and answers Flow Control, which is the protocol the target's
//! `can` utilities implement.
//!
//! # Limitations
//!
//! - **Platform**: Linux only.
//! - **Hardware**: The link is verified by compilation; a CAN interface and a target are required
//!   to exercise `open`.

use std::io;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Duration;

use socketcan_isotp::{Id, IsoTpSocket, StandardId};

use super::message::{MessageLink, MessageOf, MessageReader, MessageWriter};
use crate::error::HostError;

/// How long the reader sleeps when no message is ready.
const IDLE_WAIT: Duration = Duration::from_millis(1);

/// Result of opening the CAN link.
pub type OpenOf = Result<MessageLink<CanWriter>, HostError>;

/// Result of one socket read.
type ReadOf = io::Result<Vec<u8>>;

/// Which CAN interface and identifiers to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanConfig {
    /// `SocketCAN` interface name, for example `can0`.
    pub interface: String,
    /// Identifier the host receives on (the target's transmit identifier).
    pub rx_id: u16,
    /// Identifier the host transmits on (the target's receive identifier).
    pub tx_id: u16,
}

/// Socket shared by the reader thread and the writer.
type SharedSocket = Arc<Mutex<IsoTpSocket>>;

/// Reads messages from the ISO-TP socket.
pub struct CanReader {
    socket: SharedSocket,
}

/// Writes messages to the ISO-TP socket.
pub struct CanWriter {
    socket: SharedSocket,
}

impl MessageReader for CanReader {
    fn next_message(&mut self) -> MessageOf {
        let result = {
            let mut socket =
                self.socket.lock().unwrap_or_else(PoisonError::into_inner);
            socket.read().map(<[u8]>::to_vec)
        };
        settle(result)
    }
}

impl MessageWriter for CanWriter {
    fn write_message(&mut self, msg: &[u8]) -> io::Result<()> {
        self.socket
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .write(msg)
    }
}

/// Opens the ISO-TP socket described by `config`.
///
/// # Errors
/// Returns [`HostError::Transport`] if an identifier is out of range or the socket cannot be
/// opened.
pub fn open(config: &CanConfig) -> OpenOf {
    let fail = |what: &str, e: &dyn std::fmt::Debug| HostError::Transport {
        source: format!("CAN {what}: {e:?}").into(),
    };
    let id = |raw: u16| {
        StandardId::new(raw)
            .map(Id::from)
            .ok_or_else(|| fail("identifier out of range", &raw))
    };
    let socket = IsoTpSocket::open(
        &config.interface,
        id(config.rx_id)?,
        id(config.tx_id)?,
    )
    .map_err(|e| fail("open failed", &e))?;
    socket
        .set_nonblocking(true)
        .map_err(|e| fail("nonblocking mode failed", &e))?;
    let socket = Arc::new(Mutex::new(socket));
    let description = format!("CAN ISO-TP {}", config.interface);
    Ok(MessageLink::new(
        CanReader {
            socket: Arc::clone(&socket),
        },
        CanWriter { socket },
        description,
    ))
}

/// Maps a socket read to a reader step; a read with nothing ready idles briefly.
fn settle(result: ReadOf) -> MessageOf {
    match result {
        Ok(message) => Ok(Some(message)),
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
            thread::sleep(IDLE_WAIT);
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_with_nothing_ready_is_idle_and_other_failures_propagate() {
        assert!(matches!(settle(Ok(vec![1, 2])), Ok(Some(m)) if m == [1, 2]));
        let idle = settle(Err(io::Error::from(io::ErrorKind::WouldBlock)));
        assert!(matches!(idle, Ok(None)));
        let failed = settle(Err(io::Error::from(io::ErrorKind::BrokenPipe)));
        assert_eq!(failed.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }
}
