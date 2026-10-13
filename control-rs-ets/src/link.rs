//! Message link between the target server and the host.
//!
//! # Description
//!
//! A [`Link`] moves one complete, verified message per call. The server encodes and decodes
//! messages (Postcard) and holds no framing code. A link delimits and verifies messages with the
//! framing native to its transport. [`StreamLink`] provides the sync, length and CRC-16 frame for
//! byte streams such as UART, USB CDC-ACM and semihosting.
//!
//! # Core Concepts
//!
//! - **Whole messages**: `recv` returns one complete message or nothing; `send` accepts all of a
//!   message or none of it.
//! - **Non-blocking**: Neither call waits. A link that cannot accept a message reports busy.
//! - **Byte streams**: [`ByteStream`] is the non-blocking byte-level transport under
//!   [`StreamLink`]. Partial reads and writes are normal.

use crate::comms::{FrameEncoder, FrameReader, MAX_FRAME_SIZE};

/// Bytes read from the stream per `recv` call.
const READ_CHUNK: usize = 64;

/// Result of `Link::recv` for link `L`: a message length, nothing ready, or a transport error.
pub type RecvOf<L> = Result<Option<usize>, <L as Link>::Error>;

/// Result of `Link::send` for link `L`: accepted whole (`true`), busy (`false`) or a transport error.
pub type SendOf<L> = Result<bool, <L as Link>::Error>;

/// Result of an operation on link `L` that returns nothing.
pub type UnitOf<L> = Result<(), <L as Link>::Error>;

/// Result of a `ByteStream` read or write for stream `S`: a byte count or a transport error.
pub type CountOf<S> = Result<usize, <S as ByteStream>::Error>;

/// Result of a `ByteStream` operation that returns nothing.
pub type StreamUnitOf<S> = Result<(), <S as ByteStream>::Error>;

/// Whole-message transport between the target server and the host.
///
/// # Safety
/// Implementations must guarantee safe hardware access during transfer. This trait does not use
/// `unsafe` code.
///
/// # Panics
/// Trait operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::link::{Link, RecvOf, SendOf};
///
/// struct Echo { held: Option<[u8; 4]> }
/// impl Link for Echo {
///     type Error = ();
///     fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self> {
///         match self.held.take() {
///             Some(message) => {
///                 buf[..4].copy_from_slice(&message);
///                 Ok(Some(4))
///             }
///             None => Ok(None),
///         }
///     }
///     fn send(&mut self, message: &[u8]) -> SendOf<Self> {
///         if self.held.is_some() || message.len() != 4 {
///             return Ok(false);
///         }
///         let mut held = [0u8; 4];
///         held.copy_from_slice(message);
///         self.held = Some(held);
///         Ok(true)
///     }
/// }
///
/// let mut link = Echo { held: None };
/// assert_eq!(link.send(&[1, 2, 3, 4]), Ok(true));
/// let mut buf = [0u8; 8];
/// assert_eq!(link.recv(&mut buf), Ok(Some(4)));
/// ```
pub trait Link {
    /// The error type associated with transport failures.
    type Error;

    /// Copies one complete, verified message into `buf` and returns its length, or `None` when
    /// no message is ready. A message longer than `buf` is a transport error or is dropped by
    /// the link.
    ///
    /// # Errors
    /// Returns a transport error if reading fails.
    fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self>;

    /// Accepts all of `message` or nothing. `Ok(false)` reports back-pressure and transmits nothing.
    ///
    /// # Errors
    /// Returns a transport error if writing fails.
    fn send(&mut self, msg: &[u8]) -> SendOf<Self>;

    /// Closes the link after a clean run (for example, signaling semihosting exit).
    fn close(&mut self) {}

    /// Closes the link after a failure.
    fn close_on_failure(&mut self) {}
}

/// Non-blocking byte-level transport under [`StreamLink`].
///
/// # Safety
/// Implementations must guarantee safe hardware access during transfer. This trait does not use
/// `unsafe` code.
///
/// # Panics
/// Trait operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
///
/// struct Empty;
/// impl ByteStream for Empty {
///     type Error = ();
///     fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> { Ok(0) }
///     fn write(&mut self, buf: &[u8]) -> CountOf<Self> { Ok(buf.len()) }
/// }
///
/// let link = StreamLink::new(Empty);
/// ```
pub trait ByteStream {
    /// The error type associated with transport failures.
    type Error;

    /// Reads the bytes available without blocking; `0` means nothing is ready.
    ///
    /// # Errors
    /// Returns a transport error if reading fails.
    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self>;

    /// Writes a prefix of `buf` without blocking and returns its length; a short count is
    /// back-pressure.
    ///
    /// # Errors
    /// Returns a transport error if writing fails.
    fn write(&mut self, buf: &[u8]) -> CountOf<Self>;

    /// Closes the stream after a clean run.
    fn close(&mut self) {}

    /// Closes the stream after a failure.
    fn close_on_failure(&mut self) {}
}

/// Link over a [`ByteStream`] using the 6-byte sync, length and CRC-16 frame.
///
/// `recv` reassembles frames from however the stream chunks its bytes and keeps bytes read past
/// one frame for the next call. `send` frames the message and writes what the stream accepts; the
/// remainder is written by later calls, and a new message is refused as busy until it drains.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Link operations do not panic.
pub struct StreamLink<S> {
    reader: FrameReader,
    rx: [u8; READ_CHUNK],
    rx_len: usize,
    rx_pos: usize,
    stream: S,
    tx: [u8; MAX_FRAME_SIZE],
    tx_len: usize,
    tx_pos: usize,
}

impl<S> StreamLink<S> {
    /// Creates a link over `stream`.
    pub const fn new(stream: S) -> Self {
        Self {
            reader: FrameReader::new(),
            rx: [0; READ_CHUNK],
            rx_len: 0,
            rx_pos: 0,
            stream,
            tx: [0; MAX_FRAME_SIZE],
            tx_len: 0,
            tx_pos: 0,
        }
    }

    /// Borrows the underlying stream.
    pub const fn stream(&self) -> &S {
        &self.stream
    }

    /// Mutably borrows the underlying stream.
    pub const fn stream_mut(&mut self) -> &mut S {
        &mut self.stream
    }
}

impl<S: ByteStream> StreamLink<S> {
    /// Writes the pending frame bytes the stream accepts.
    fn drain_tx(&mut self) -> StreamUnitOf<S> {
        while self.tx_pos < self.tx_len {
            let Some(pending) = self.tx.get(self.tx_pos..self.tx_len) else {
                break;
            };
            let written = self.stream.write(pending)?;
            if written == 0 {
                break;
            }
            self.tx_pos = self.tx_pos.saturating_add(written);
        }
        if self.tx_pos >= self.tx_len {
            self.tx_pos = 0;
            self.tx_len = 0;
        }
        Ok(())
    }

    /// Feeds buffered bytes to the frame reader until one frame completes.
    fn next_frame(&mut self, buf: &mut [u8]) -> Option<usize> {
        while self.rx_pos < self.rx_len {
            let byte = self.rx.get(self.rx_pos).copied()?;
            self.rx_pos = self.rx_pos.saturating_add(1);
            if let Some(payload) = self.reader.handle_byte(byte) {
                let dest = buf.get_mut(..payload.len())?;
                dest.copy_from_slice(payload);
                return Some(payload.len());
            }
        }
        None
    }
}

impl<S: ByteStream> Link for StreamLink<S> {
    type Error = S::Error;

    fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self> {
        self.drain_tx()?;
        if let Some(len) = self.next_frame(buf) {
            return Ok(Some(len));
        }
        let count = self.stream.read(&mut self.rx)?;
        self.rx_len = count.min(READ_CHUNK);
        self.rx_pos = 0;
        Ok(self.next_frame(buf))
    }

    fn send(&mut self, msg: &[u8]) -> SendOf<Self> {
        self.drain_tx()?;
        if self.tx_len != 0 {
            return Ok(false);
        }
        let Ok(len) = FrameEncoder::frame_payload(msg, &mut self.tx) else {
            return Ok(false);
        };
        self.tx_len = len;
        self.tx_pos = 0;
        self.drain_tx()?;
        Ok(true)
    }

    fn close(&mut self) {
        self.stream.close();
    }

    fn close_on_failure(&mut self) {
        self.stream.close_on_failure();
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    /// Scripted read chunks.
    type Chunks = Vec<Vec<u8>>;

    /// Stream that serves scripted chunks and accepts writes until its `room` is used up.
    struct ScriptedStream {
        room: usize,
        chunks: Chunks,
        written: Vec<u8>,
    }

    impl ScriptedStream {
        fn new(chunks: Chunks, room: usize) -> Self {
            Self {
                room,
                chunks,
                written: Vec::new(),
            }
        }
    }

    impl ByteStream for ScriptedStream {
        type Error = ();

        fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
            if self.chunks.is_empty() {
                return Ok(0);
            }
            let chunk = self.chunks.remove(0);
            let n = chunk.len().min(buf.len());
            for (dst, src) in buf.iter_mut().zip(chunk.iter()) {
                *dst = *src;
            }
            Ok(n)
        }

        fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
            let n = buf.len().min(self.room);
            self.room = self.room.saturating_sub(n);
            self.written.extend_from_slice(buf.get(..n).unwrap());
            Ok(n)
        }
    }

    fn frame(payload: &[u8]) -> Vec<u8> {
        let mut dest = std::vec![0u8; MAX_FRAME_SIZE];
        let len = FrameEncoder::frame_payload(payload, &mut dest).unwrap();
        dest.truncate(len);
        dest
    }

    fn drain(link: &mut StreamLink<ScriptedStream>) -> Chunks {
        let mut out = Vec::new();
        let mut buf = [0u8; 512];
        for _ in 0..64 {
            if let Some(n) = link.recv(&mut buf).unwrap() {
                out.push(buf.get(..n).unwrap().to_vec());
            }
        }
        out
    }

    #[test]
    fn send_frames_a_message_with_sync_length_and_crc() {
        let mut link = StreamLink::new(ScriptedStream::new(Vec::new(), 512));
        assert_eq!(link.send(&[1, 2, 3]), Ok(true));
        assert_eq!(link.stream().written, frame(&[1, 2, 3]));
        assert_eq!(link.stream().written.len(), 3 + 6);
    }

    #[test]
    fn a_short_write_is_finished_by_later_calls_and_refuses_a_second_message() {
        let mut link = StreamLink::new(ScriptedStream::new(Vec::new(), 4));
        assert_eq!(link.send(&[9; 10]), Ok(true));
        assert_eq!(link.stream().written.len(), 4);
        assert_eq!(link.send(&[7]), Ok(false), "busy until the frame drains");
        link.stream_mut().room = 100;
        let mut buf = [0u8; 8];
        assert_eq!(link.recv(&mut buf), Ok(None));
        assert_eq!(link.stream().written, frame(&[9; 10]));
        assert_eq!(link.send(&[7]), Ok(true));
    }

    #[test]
    fn recv_reassembles_a_frame_split_across_reads() {
        let bytes = frame(&[5, 6, 7, 8]);
        let (a, b) = bytes.split_at(3);
        let mut link = StreamLink::new(ScriptedStream::new(
            std::vec![a.to_vec(), b.to_vec()],
            512,
        ));
        assert_eq!(drain(&mut link), std::vec![std::vec![5, 6, 7, 8]]);
    }

    #[test]
    fn recv_returns_every_frame_of_one_read_in_order() {
        let mut bytes = frame(&[1]);
        bytes.extend(frame(&[2, 2]));
        bytes.extend(frame(&[3, 3, 3]));
        let mut link =
            StreamLink::new(ScriptedStream::new(std::vec![bytes], 512));
        assert_eq!(
            drain(&mut link),
            std::vec![std::vec![1], std::vec![2, 2], std::vec![3, 3, 3]]
        );
    }

    #[test]
    fn recv_drops_a_corrupted_frame_and_keeps_the_next() {
        let mut bad = frame(&[4, 4, 4]);
        if let Some(byte) = bad.last_mut() {
            *byte ^= 0xFF;
        }
        bad.extend(frame(&[8]));
        let mut link =
            StreamLink::new(ScriptedStream::new(std::vec![bad], 512));
        assert_eq!(drain(&mut link), std::vec![std::vec![8]]);
    }

    #[test]
    fn an_oversized_message_is_refused() {
        let mut link = StreamLink::new(ScriptedStream::new(Vec::new(), 512));
        assert_eq!(link.send(&[0u8; 513]), Ok(false));
        assert_eq!(link.stream().written, Vec::<u8>::new());
    }

    #[test]
    fn a_maximum_size_message_round_trips() {
        let payload = std::vec![0xA5u8; 512];
        let mut tx = StreamLink::new(ScriptedStream::new(Vec::new(), 1024));
        assert_eq!(tx.send(&payload), Ok(true));
        let wire = tx.stream().written.clone();
        let chunks = wire.chunks(64).map(<[u8]>::to_vec).collect();
        let mut rx = StreamLink::new(ScriptedStream::new(chunks, 512));
        assert_eq!(drain(&mut rx), std::vec![payload]);
    }
}
