//! Stream link: whole messages over a non-blocking byte stream.
//!
//! # Description
//!
//! [`StreamLink`] wraps each message in a sync, length and CRC-16 frame and reassembles frames
//! from however the stream chunks its bytes. [`FrameReader`] and [`encode_frame`] are the frame
//! codec; the host uses the same codec for its serial and pipe links.
//!
//! # Core Concepts
//!
//! - **Frame**: `0xAA 0x55`, a big-endian `u16` payload length, the payload and a big-endian
//!   CRC-16/IBM-SDLC over the payload (6 bytes of overhead).
//! - **Resynchronization**: A bad length or CRC drops the frame and the reader hunts for the next
//!   sync header.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::link::stream::{FrameReader, encode_frame};
//!
//! let mut wire = [0u8; 32];
//! let len = encode_frame(&[1, 2, 3], &mut wire)?;
//! let mut reader = FrameReader::new();
//! let payload = wire[..len].iter().find_map(|&b| reader.handle_byte(b).map(<[u8]>::to_vec));
//! assert_eq!(payload, Some(vec![1, 2, 3]));
//! # Ok::<(), control_rs_ets::link::stream::FrameError>(())
//! ```

use super::{ByteStream, Link, RecvOf, SendOf};
use crate::wire::MAX_MESSAGE_SIZE;

/// Sync (2) + length (2) + CRC-16 (2).
pub const FRAME_OVERHEAD: usize = 6;
/// Maximum encoded frame size (`MAX_MESSAGE_SIZE` + `FRAME_OVERHEAD`).
pub const MAX_FRAME_SIZE: usize = MAX_MESSAGE_SIZE + FRAME_OVERHEAD;
/// Bytes read from the stream per `recv` call.
const READ_CHUNK: usize = 64;
const START_BYTE_1: u8 = 0xAA;
const START_BYTE_2: u8 = 0x55;

/// A verified payload, if a frame completed.
pub type FrameOf<'a> = Option<&'a [u8]>;

/// Result of a `ByteStream` operation that returns nothing.
type StreamUnitOf<S> = Result<(), <S as ByteStream>::Error>;

/// Failure to frame a payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameError {
    /// The destination cannot hold the frame or the payload exceeds [`MAX_MESSAGE_SIZE`].
    BufferFull,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReaderState {
    ReadingPayload { len: usize, read: usize },
    WaitChecksum1 { len: usize },
    WaitChecksum2 { len: usize, crc1: u8 },
    WaitLen1,
    WaitLen2,
    WaitStart1,
    WaitStart2,
}

/// State machine that de-frames a stream of bytes into verified payloads.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Byte handling does not panic.
pub struct FrameReader {
    payload_buffer: [u8; MAX_MESSAGE_SIZE],
    state: ReaderState,
    temp_len: u16,
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
        let Ok(len) = encode_frame(msg, &mut self.tx) else {
            return Ok(false);
        };
        self.tx_len = len;
        self.tx_pos = 0;
        self.drain_tx()?;
        Ok(true)
    }

    fn close_on_failure(&mut self) {
        self.stream.close_on_failure();
    }
}

impl Default for FrameReader {
    fn default() -> Self {
        Self::new()
    }
}

impl FrameReader {
    /// Process a single incoming byte. Returns `Some(&[u8])` when a complete,
    /// checksum-verified payload has been received.
    ///
    /// # Arguments
    /// * `byte` - Incoming serial stream byte.
    ///
    /// # Returns
    /// * `Option<&[u8]>`
    ///     * `Some(frame_payload)` - A byte slice reference to the verified frame payload buffer.
    ///     * `None` - If the frame is still incomplete or the CRC check failed.
    pub fn handle_byte(&mut self, byte: u8) -> FrameOf<'_> {
        match self.state {
            ReaderState::WaitStart1 => {
                if byte == START_BYTE_1 {
                    self.state = ReaderState::WaitStart2;
                }
            }
            ReaderState::WaitStart2 => {
                if byte == START_BYTE_2 {
                    self.state = ReaderState::WaitLen1;
                } else if byte == START_BYTE_1 {
                    // Orphan/noise 0xAA left us in WaitStart2; this 0xAA may
                    // start a real frame, so stay armed for 0x55.
                } else {
                    self.state = ReaderState::WaitStart1;
                }
            }
            ReaderState::WaitLen1 => {
                self.temp_len = u16::from(byte) << 8;
                self.state = ReaderState::WaitLen2;
            }
            ReaderState::WaitLen2 => {
                self.temp_len |= u16::from(byte);
                let len = self.temp_len as usize;
                if len == 0 || len > MAX_MESSAGE_SIZE {
                    // Invalid length, reset state machine
                    self.state = ReaderState::WaitStart1;
                } else {
                    self.state = ReaderState::ReadingPayload { len, read: 0 };
                }
            }
            ReaderState::ReadingPayload { len, ref mut read } => {
                if let Some(slot) = self.payload_buffer.get_mut(*read) {
                    *slot = byte;
                }
                *read = (*read).saturating_add(1);
                if *read == len {
                    self.state = ReaderState::WaitChecksum1 { len };
                }
            }
            ReaderState::WaitChecksum1 { len } => {
                self.state = ReaderState::WaitChecksum2 { len, crc1: byte };
            }
            ReaderState::WaitChecksum2 { len, crc1 } => {
                let crc_value = (u16::from(crc1) << 8) | u16::from(byte);
                let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);
                let calculated =
                    crc.checksum(self.payload_buffer.get(..len).unwrap_or(&[]));

                self.state = ReaderState::WaitStart1;
                if calculated == crc_value {
                    return self.payload_buffer.get(..len);
                }
            }
        }
        None
    }

    /// Returns true if the reader is currently idle (waiting for a frame start).
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        matches!(self.state, ReaderState::WaitStart1)
    }

    /// Creates a new `FrameReader`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            payload_buffer: [0u8; MAX_MESSAGE_SIZE],
            state: ReaderState::WaitStart1,
            temp_len: 0,
        }
    }
}

/// Frames `payload` into `dest` and returns the frame length.
///
/// # Errors
/// Returns [`FrameError::BufferFull`] if `payload` exceeds [`MAX_MESSAGE_SIZE`] or `dest` is
/// smaller than the frame.
pub fn encode_frame(
    payload: &[u8],
    dest: &mut [u8],
) -> Result<usize, FrameError> {
    let len = payload.len();
    if len > MAX_MESSAGE_SIZE {
        return Err(FrameError::BufferFull);
    }
    let total = len
        .checked_add(FRAME_OVERHEAD)
        .ok_or(FrameError::BufferFull)?;
    let frame = dest.get_mut(..total).ok_or(FrameError::BufferFull)?;
    let len_u16 = u16::try_from(len).map_err(|_| FrameError::BufferFull)?;
    let crc = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC).checksum(payload);
    let length_bytes = [(len_u16 >> 8) as u8, (len_u16 & 0xFF) as u8];
    let crc_bytes = [(crc >> 8) as u8, (crc & 0xFF) as u8];
    let (head, rest) = frame.split_at_mut(2);
    head.copy_from_slice(&[START_BYTE_1, START_BYTE_2]);
    let (length, rest) = rest.split_at_mut(2);
    length.copy_from_slice(&length_bytes);
    let (body, tail) = rest.split_at_mut(len);
    body.copy_from_slice(payload);
    tail.copy_from_slice(&crc_bytes);
    Ok(total)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::link::CountOf;
    use std::vec::Vec;

    /// Scripted read chunks.
    type Chunks = Vec<Vec<u8>>;

    /// Stream that serves scripted chunks and accepts writes until its `room` is used up.
    struct ScriptedStream {
        room: usize,
        chunks: Chunks,
        written: Vec<u8>,
        zero_writes: usize,
        closed_on_failure: bool,
    }

    impl ScriptedStream {
        fn new(chunks: Chunks, room: usize) -> Self {
            Self {
                room,
                chunks,
                written: Vec::new(),
                zero_writes: 0,
                closed_on_failure: false,
            }
        }
    }

    impl ByteStream for ScriptedStream {
        type Error = ();

        fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
            if self.chunks.is_empty() {
                return Ok(0);
            }
            let mut chunk = self.chunks.remove(0);
            let n = chunk.len().min(buf.len());
            for (dst, src) in buf.iter_mut().zip(chunk.iter()) {
                *dst = *src;
            }
            if n < chunk.len() {
                self.chunks.insert(0, chunk.split_off(n));
            }
            Ok(n)
        }

        fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
            assert!(!buf.is_empty(), "write called with empty buffer");
            let n = buf.len().min(self.room);
            if n == 0 {
                self.zero_writes = self.zero_writes.saturating_add(1);
                assert!(self.zero_writes <= 3, "drain_tx looped on zero write");
            } else {
                self.zero_writes = 0;
            }
            self.room = self.room.saturating_sub(n);
            self.written.extend_from_slice(buf.get(..n).unwrap());
            Ok(n)
        }

        fn close_on_failure(&mut self) {
            self.closed_on_failure = true;
        }
    }

    fn frame(payload: &[u8]) -> Vec<u8> {
        let mut dest = std::vec![0u8; MAX_FRAME_SIZE];
        let len = encode_frame(payload, &mut dest).unwrap();
        dest.truncate(len);
        dest
    }

    fn drain(link: &mut StreamLink<ScriptedStream>) -> Chunks {
        let mut out = Vec::new();
        let mut buf = [0u8; 512];
        for _ in 0..4096 {
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
    fn short_write_reports_busy_until_drained() {
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
    fn corrupt_frame_dropped() {
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

    #[test]
    fn any_chunking_delivers_all() {
        let mut wire = frame(&[1]);
        wire.extend(frame(&[2, 2]));
        wire.extend(frame(&[3; 200]));
        let expected =
            std::vec![std::vec![1], std::vec![2, 2], std::vec![3; 200]];
        for size in 1..=wire.len() {
            let chunks = wire.chunks(size).map(<[u8]>::to_vec).collect();
            let mut link = StreamLink::new(ScriptedStream::new(chunks, 512));
            assert_eq!(drain(&mut link), expected, "chunk size {size}");
        }
    }

    #[test]
    fn a_stream_that_accepts_nothing_never_blocks() {
        let mut link = StreamLink::new(ScriptedStream::new(Vec::new(), 0));
        assert_eq!(link.send(&[1, 2]), Ok(true), "queued whole");
        assert_eq!(link.send(&[3]), Ok(false), "busy until drained");
        let mut buf = [0u8; 8];
        assert_eq!(link.recv(&mut buf), Ok(None));
        assert_eq!(link.stream().written, Vec::<u8>::new());
    }

    #[test]
    fn the_reader_hunts_for_sync_and_rejects_bad_frames() {
        let mut reader = FrameReader::default();
        assert!(reader.is_idle());
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        assert!(!reader.is_idle());
        assert!(reader.handle_byte(0x00).is_none());
        assert!(reader.is_idle());

        // Zero and oversize lengths reset the reader.
        for len in [[0x00, 0x00], [0x02, 0x01]] {
            assert!(reader.handle_byte(START_BYTE_1).is_none());
            assert!(reader.handle_byte(START_BYTE_2).is_none());
            assert!(reader.handle_byte(len[0]).is_none());
            assert!(reader.handle_byte(len[1]).is_none());
            assert!(reader.is_idle());
        }

        // An orphan sync byte does not consume the real frame's first byte.
        let good = frame(&[9, 9]);
        assert!(reader.handle_byte(START_BYTE_1).is_none());
        let decoded = good
            .iter()
            .filter_map(|&b| reader.handle_byte(b).map(<[u8]>::to_vec))
            .last();
        assert_eq!(decoded, Some(std::vec![9, 9]));

        // A flipped checksum byte drops the frame and the reader goes idle.
        let mut bad = frame(&[9, 9]);
        let last = bad.len() - 2;
        if let Some(byte) = bad.get_mut(last) {
            *byte ^= 0xFF;
        }
        assert!(bad.iter().all(|&b| reader.handle_byte(b).is_none()));
        assert!(reader.is_idle());
    }

    #[test]
    fn framing_checks_room_and_writes_a_big_endian_length() {
        let payload = [1u8; 300];
        let mut dest = [0u8; MAX_FRAME_SIZE];
        let len = encode_frame(&payload, &mut dest).unwrap();
        assert_eq!(len, 306);
        assert_eq!(
            dest.get(..4),
            Some(&[START_BYTE_1, START_BYTE_2, 1, 44][..])
        );
        let mut exact = [0u8; 16];
        assert_eq!(encode_frame(&[7; 10], &mut exact), Ok(16));
        assert_eq!(
            encode_frame(&[7; 10], exact.get_mut(..15).unwrap()),
            Err(FrameError::BufferFull)
        );
        assert_eq!(
            encode_frame(&[0u8; MAX_MESSAGE_SIZE + 1], &mut dest),
            Err(FrameError::BufferFull)
        );
        let full = [0x5A_u8; MAX_MESSAGE_SIZE];
        assert_eq!(encode_frame(&full, &mut dest), Ok(MAX_FRAME_SIZE));
    }

    #[test]
    fn stream_link_close_on_failure_calls_stream() {
        let mut link = StreamLink::new(ScriptedStream::new(Vec::new(), 0));
        assert!(!link.stream().closed_on_failure);
        link.close_on_failure();
        assert!(link.stream().closed_on_failure);
    }
}
