//! ISO-TP segmentation for classical CAN, for a user's link.
//!
//! # Description
//!
//! A message longer than one CAN frame travels as an ISO-TP transfer (ISO 15765-2): a First Frame
//! with the total length, Consecutive Frames with a 4-bit sequence counter and a Flow Control
//! frame from the receiver. [`Segmenter`] turns a message into frames and [`Reassembler`] joins
//! received frames into a message and produces the Flow Control reply. Both work on classical
//! 8-byte frames and hold no driver, identifier or clock, so a user's [`Link`](super::Link)
//! composes them with its CAN controller calls and pacing.
//!
//! # Core Concepts
//!
//! - **Single Frame**: A message of up to 7 bytes in one frame with a 1-byte header.
//! - **First Frame and Consecutive Frames**: Longer messages. The First Frame declares the total
//!   length (up to 4095 bytes); each Consecutive Frame spends one byte on its header.
//! - **Flow Control**: The receiver answers the First Frame; the reassembler asks for all
//!   frames without a block limit and without separation time.
//! - **Timeouts**: The helpers have no clock. A link calls [`Reassembler::reset`] when a transfer
//!   stalls and drops the [`Segmenter`] when its peer stops answering.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::link::can::{Reassembler, Rx, Segmenter};
//!
//! let message: Vec<u8> = (0..20).collect();
//! let mut tx = Segmenter::new(&message)?;
//! let mut rx = Reassembler::<64>::new();
//! let mut frame = [0u8; 8];
//! let mut received = None;
//! loop {
//!     while let Some(n) = tx.next_frame(&mut frame) {
//!         match rx.push(&frame[..n])? {
//!             Rx::Message(m) => received = Some(m.to_vec()),
//!             Rx::FlowControl(fc) => tx.flow_control(&fc)?,
//!             Rx::None => {}
//!         }
//!     }
//!     if tx.is_done() { break; }
//! }
//! assert_eq!(received, Some(message));
//! # Ok::<(), control_rs_ets::link::can::IsoTpError>(())
//! ```

const CONSECUTIVE: u8 = 0x20;
/// Data bytes of a Consecutive Frame.
const CONSECUTIVE_DATA: usize = 7;
const FIRST: u8 = 0x10;
/// Data bytes of a First Frame.
const FIRST_DATA: usize = 6;
const FLOW: u8 = 0x30;
/// Maximum message length of an ISO-TP transfer with a 12-bit length.
pub const MAX_ISOTP_LEN: usize = 4095;
/// Data bytes of a Single Frame.
const SINGLE_DATA: usize = 7;

/// A failure of an ISO-TP transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsoTpError {
    /// A frame is shorter than its header declares or has an invalid header.
    Malformed,
    /// A Consecutive Frame arrived out of sequence; the partial message is discarded.
    Sequence,
    /// The message is longer than [`MAX_ISOTP_LEN`].
    TooLong,
    /// A frame arrived that the receiver does not expect in its state.
    Unexpected,
    /// The peer answered Flow Control with overflow.
    Overflow,
}

/// What a received frame produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rx<'a> {
    /// The frame completed a message.
    Message(&'a [u8]),
    /// The frame is a First Frame; send this Flow Control frame to the sender.
    FlowControl([u8; 3]),
    /// The frame was accepted and the message is not complete.
    None,
}

/// Splits one message into ISO-TP frames.
#[derive(Debug, Clone)]
pub struct Segmenter<'a> {
    awaiting_flow: bool,
    block_left: u8,
    block_size: u8,
    message: &'a [u8],
    pos: usize,
    separation: u8,
    sequence: u8,
    started: bool,
}

/// Joins ISO-TP frames into a message of at most `N` bytes.
#[derive(Debug, Clone)]
pub struct Reassembler<const N: usize> {
    buf: [u8; N],
    complete: bool,
    expected: usize,
    len: usize,
    next_sequence: u8,
    receiving: bool,
}

impl<'a> Segmenter<'a> {
    /// Starts a transfer of `message`.
    ///
    /// # Errors
    /// Returns [`IsoTpError::TooLong`] if the message is longer than [`MAX_ISOTP_LEN`] and
    /// [`IsoTpError::Malformed`] if it is empty.
    pub const fn new(message: &'a [u8]) -> Result<Self, IsoTpError> {
        if message.len() > MAX_ISOTP_LEN {
            return Err(IsoTpError::TooLong);
        }
        if message.is_empty() {
            return Err(IsoTpError::Malformed);
        }
        Ok(Self {
            awaiting_flow: false,
            block_left: 0,
            block_size: 0,
            message,
            pos: 0,
            separation: 0,
            sequence: 1,
            started: false,
        })
    }

    /// Writes the next frame into `out` and returns its length, or `None` when the transfer is
    /// finished or waits for Flow Control.
    pub fn next_frame(&mut self, out: &mut [u8; 8]) -> Option<usize> {
        if self.awaiting_flow || self.is_done() {
            return None;
        }
        let remaining = self.message.len().saturating_sub(self.pos);
        if !self.started {
            self.started = true;
            return Some(self.first_frame(remaining, out));
        }
        let take = remaining.min(CONSECUTIVE_DATA);
        let chunk =
            self.message.get(self.pos..self.pos.saturating_add(take))?;
        *out.first_mut()? = CONSECUTIVE | self.sequence;
        out.get_mut(1..=take)?.copy_from_slice(chunk);
        self.pos = self.pos.saturating_add(take);
        self.sequence = self.sequence.wrapping_add(1) & 0x0F;
        if self.block_size != 0 {
            self.block_left = self.block_left.saturating_sub(1);
            if self.block_left == 0 {
                self.awaiting_flow = self.pos < self.message.len();
            }
        }
        Some(take.saturating_add(1))
    }

    fn first_frame(&mut self, remaining: usize, out: &mut [u8; 8]) -> usize {
        if remaining <= SINGLE_DATA {
            let (head, data) = out.split_at_mut(1);
            if let (Some(h), Some(dst), Some(src)) = (
                head.first_mut(),
                data.get_mut(..remaining),
                self.message.get(..remaining),
            ) {
                *h = u8::try_from(remaining).unwrap_or(0);
                dst.copy_from_slice(src);
            }
            self.pos = self.message.len();
            return remaining.saturating_add(1);
        }
        let len = u16::try_from(self.message.len()).unwrap_or(0);
        let (hi, lo) = ((len >> 8) as u8, (len & 0xFF) as u8);
        let (head, data) = out.split_at_mut(2);
        if let Some(src) = self.message.get(..FIRST_DATA) {
            head.copy_from_slice(&[FIRST | (hi & 0x0F), lo]);
            data.copy_from_slice(src);
        }
        self.pos = FIRST_DATA;
        self.awaiting_flow = true;
        8
    }

    /// Applies a received Flow Control frame.
    ///
    /// # Errors
    /// Returns [`IsoTpError::Overflow`] if the receiver reports overflow and
    /// [`IsoTpError::Malformed`] if `frame` is not a Flow Control frame.
    pub const fn flow_control(
        &mut self,
        frame: &[u8],
    ) -> Result<(), IsoTpError> {
        let &[head, block, separation, ..] = frame else {
            return Err(IsoTpError::Malformed);
        };
        if head & 0xF0 != FLOW {
            return Err(IsoTpError::Malformed);
        }
        match head & 0x0F {
            0 => {
                self.block_size = block;
                self.block_left = block;
                self.separation = separation;
                self.awaiting_flow = false;
                Ok(())
            }
            1 => Ok(()),
            2 => Err(IsoTpError::Overflow),
            _ => Err(IsoTpError::Malformed),
        }
    }

    /// Returns `true` when every frame of the message has been produced.
    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.started && self.pos >= self.message.len()
    }

    /// Returns `true` while the transfer waits for a Flow Control frame.
    #[must_use]
    pub const fn is_waiting(&self) -> bool {
        self.awaiting_flow
    }

    /// Returns the raw separation time the receiver requested between Consecutive Frames.
    #[must_use]
    pub const fn separation_time(&self) -> u8 {
        self.separation
    }
}

impl<const N: usize> Default for Reassembler<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Reassembler<N> {
    /// Creates an empty reassembler.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buf: [0; N],
            complete: false,
            expected: 0,
            len: 0,
            next_sequence: 0,
            receiving: false,
        }
    }

    /// Discards any partial message, for example after a stalled transfer.
    pub const fn reset(&mut self) {
        self.complete = false;
        self.len = 0;
        self.receiving = false;
    }

    /// Adds one received frame.
    ///
    /// # Errors
    /// Returns [`IsoTpError::Sequence`] for a Consecutive Frame out of sequence,
    /// [`IsoTpError::Unexpected`] for a Consecutive Frame without a First Frame or a Flow Control
    /// frame, and [`IsoTpError::Malformed`] for a frame shorter than its header declares.
    pub fn push(&mut self, frame: &[u8]) -> Result<Rx<'_>, IsoTpError> {
        if self.complete {
            self.reset();
        }
        let head = *frame.first().ok_or(IsoTpError::Malformed)?;
        match head & 0xF0 {
            0x00 => self.single(head, frame),
            FIRST => self.first(head, frame),
            CONSECUTIVE => self.consecutive(head, frame),
            _ => {
                self.reset();
                Err(IsoTpError::Unexpected)
            }
        }
    }

    fn single(&mut self, head: u8, frame: &[u8]) -> Result<Rx<'_>, IsoTpError> {
        self.reset();
        let len = usize::from(head & 0x0F);
        if len == 0 || len > SINGLE_DATA {
            return Err(IsoTpError::Malformed);
        }
        let data = frame.get(1..=len).ok_or(IsoTpError::Malformed)?;
        let dest = self.buf.get_mut(..len).ok_or(IsoTpError::TooLong)?;
        dest.copy_from_slice(data);
        self.len = len;
        self.complete = true;
        Ok(Rx::Message(self.buf.get(..len).unwrap_or(&[])))
    }

    fn first(&mut self, head: u8, frame: &[u8]) -> Result<Rx<'_>, IsoTpError> {
        self.reset();
        let low = *frame.get(1).ok_or(IsoTpError::Malformed)?;
        let total = (usize::from(head & 0x0F) << 8) | usize::from(low);
        let data = frame.get(2..8).ok_or(IsoTpError::Malformed)?;
        if total <= SINGLE_DATA {
            return Err(IsoTpError::Malformed);
        }
        if total > N {
            return Ok(Rx::FlowControl([FLOW | 2, 0, 0]));
        }
        let dest = self.buf.get_mut(..FIRST_DATA).ok_or(IsoTpError::TooLong)?;
        dest.copy_from_slice(data);
        self.len = FIRST_DATA;
        self.expected = total;
        self.next_sequence = 1;
        self.receiving = true;
        Ok(Rx::FlowControl([FLOW, 0, 0]))
    }

    fn consecutive(
        &mut self,
        head: u8,
        frame: &[u8],
    ) -> Result<Rx<'_>, IsoTpError> {
        if !self.receiving {
            return Err(IsoTpError::Unexpected);
        }
        if head & 0x0F != self.next_sequence {
            self.reset();
            return Err(IsoTpError::Sequence);
        }
        let take = self.expected.saturating_sub(self.len).min(CONSECUTIVE_DATA);
        let Some(data) = frame.get(1..=take) else {
            self.reset();
            return Err(IsoTpError::Malformed);
        };
        let end = self.len.saturating_add(take);
        let Some(dest) = self.buf.get_mut(self.len..end) else {
            self.reset();
            return Err(IsoTpError::TooLong);
        };
        dest.copy_from_slice(data);
        self.len = end;
        self.next_sequence = self.next_sequence.wrapping_add(1) & 0x0F;
        if self.len < self.expected {
            return Ok(Rx::None);
        }
        self.receiving = false;
        self.complete = true;
        Ok(Rx::Message(self.buf.get(..self.len).unwrap_or(&[])))
    }
}

/// Returns `true` when `frame` is a Flow Control frame, which belongs to the [`Segmenter`].
#[must_use]
pub fn is_flow_control(frame: &[u8]) -> bool {
    frame.first().is_some_and(|b| b & 0xF0 == FLOW)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    type Message = Option<Vec<u8>>;

    fn transfer(message: &[u8], rx: &mut Reassembler<512>) -> Message {
        let mut tx = Segmenter::new(message).unwrap();
        let mut frame = [0u8; 8];
        let mut out = None;
        for _ in 0..1024 {
            while let Some(n) = tx.next_frame(&mut frame) {
                match rx.push(frame.get(..n).unwrap()).unwrap() {
                    Rx::Message(m) => out = Some(m.to_vec()),
                    Rx::FlowControl(fc) => tx.flow_control(&fc).unwrap(),
                    Rx::None => {}
                }
            }
            if tx.is_done() {
                break;
            }
        }
        out
    }

    #[test]
    fn a_short_message_is_one_single_frame() {
        let mut tx = Segmenter::new(&[9, 8, 7]).unwrap();
        let mut frame = [0u8; 8];
        assert_eq!(tx.next_frame(&mut frame), Some(4));
        assert_eq!(&frame[..4], &[0x03, 9, 8, 7]);
        assert_eq!(tx.next_frame(&mut frame), None);
        assert!(tx.is_done());
    }

    #[test]
    fn segments_carry_one_pci_byte() {
        let message: Vec<u8> = (0..20).collect();
        let mut tx = Segmenter::new(&message).unwrap();
        let mut frame = [0u8; 8];
        assert_eq!(tx.next_frame(&mut frame), Some(8));
        assert_eq!(&frame, &[0x10, 20, 0, 1, 2, 3, 4, 5]);
        assert!(tx.is_waiting());
        assert_eq!(tx.next_frame(&mut frame), None, "waits for flow control");
        tx.flow_control(&[0x30, 0, 0]).unwrap();
        assert_eq!(tx.next_frame(&mut frame), Some(8));
        assert_eq!(&frame, &[0x21, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(tx.next_frame(&mut frame), Some(8));
        assert_eq!(&frame, &[0x22, 13, 14, 15, 16, 17, 18, 19]);
        assert_eq!(tx.next_frame(&mut frame), None);
        assert!(tx.is_done());
    }

    #[test]
    fn first_frame_answered_with_flow_control() {
        let message: Vec<u8> = (0..20).collect();
        let mut tx = Segmenter::new(&message).unwrap();
        let mut rx = Reassembler::<64>::new();
        let mut frame = [0u8; 8];
        tx.next_frame(&mut frame);
        assert_eq!(rx.push(&frame), Ok(Rx::FlowControl([0x30, 0, 0])));
        assert_eq!(rx.push(&[0x21, 6, 7, 8, 9, 10, 11, 12]), Ok(Rx::None));
    }

    #[test]
    fn messages_round_trip_across_the_sequence_wrap() {
        let mut rx = Reassembler::<512>::new();
        for len in [1usize, 7, 8, 13, 14, 112, 113, 120, 300, 512] {
            let message: Vec<u8> =
                (0..len).map(|i| u8::try_from(i % 253).unwrap()).collect();
            assert_eq!(transfer(&message, &mut rx), Some(message), "len {len}");
        }
    }

    #[test]
    fn lost_segment_discards_message() {
        let message: Vec<u8> = (0..40).collect();
        let mut tx = Segmenter::new(&message).unwrap();
        let mut rx = Reassembler::<64>::new();
        let mut frame = [0u8; 8];
        tx.next_frame(&mut frame);
        rx.push(&frame).unwrap();
        tx.flow_control(&[0x30, 0, 0]).unwrap();
        tx.next_frame(&mut frame);
        rx.push(&frame).unwrap();
        tx.next_frame(&mut frame); // lost
        tx.next_frame(&mut frame);
        assert_eq!(rx.push(&frame), Err(IsoTpError::Sequence));
        assert_eq!(
            rx.push(&[0x24, 0, 0, 0, 0, 0, 0, 0]),
            Err(IsoTpError::Unexpected)
        );
    }

    #[test]
    fn a_message_longer_than_the_reassembler_is_refused_with_overflow() {
        let message: Vec<u8> = (0..100).collect();
        let mut tx = Segmenter::new(&message).unwrap();
        let mut rx = Reassembler::<32>::new();
        let mut frame = [0u8; 8];
        tx.next_frame(&mut frame);
        let Ok(Rx::FlowControl(fc)) = rx.push(&frame) else {
            panic!("expected flow control");
        };
        assert_eq!(fc, [0x32, 0, 0]);
        assert_eq!(tx.flow_control(&fc), Err(IsoTpError::Overflow));
    }

    #[test]
    fn a_block_size_pauses_the_sender_until_the_next_flow_control() {
        let message: Vec<u8> = (0..60).collect();
        let mut tx = Segmenter::new(&message).unwrap();
        let mut frame = [0u8; 8];
        tx.next_frame(&mut frame);
        tx.flow_control(&[0x30, 2, 5]).unwrap();
        assert_eq!(tx.separation_time(), 5);
        assert!(tx.next_frame(&mut frame).is_some());
        assert!(tx.next_frame(&mut frame).is_some());
        assert_eq!(tx.next_frame(&mut frame), None);
        assert!(tx.is_waiting());
        tx.flow_control(&[0x30, 0, 0]).unwrap();
        assert!(tx.next_frame(&mut frame).is_some());
    }

    #[test]
    fn a_block_ending_on_the_last_frame_does_not_wait() {
        let message: Vec<u8> = (0..20).collect();
        let mut tx = Segmenter::new(&message).unwrap();
        let mut frame = [0u8; 8];
        assert!(!tx.is_waiting());
        tx.next_frame(&mut frame);
        assert!(tx.is_waiting());
        tx.flow_control(&[0x30, 2, 0]).unwrap();
        assert!(!tx.is_waiting());
        assert!(tx.next_frame(&mut frame).is_some());
        assert!(tx.next_frame(&mut frame).is_some());
        assert!(tx.is_done());
        assert!(!tx.is_waiting());
    }

    #[test]
    fn malformed_frames_are_rejected() {
        let mut rx = Reassembler::<64>::new();
        assert_eq!(rx.push(&[]), Err(IsoTpError::Malformed));
        assert_eq!(rx.push(&[0x05, 1, 2]), Err(IsoTpError::Malformed));
        assert_eq!(rx.push(&[0x00, 1]), Err(IsoTpError::Malformed));
        assert_eq!(
            rx.push(&[0x10, 5, 0, 0, 0, 0, 0, 0]),
            Err(IsoTpError::Malformed)
        );
        assert_eq!(rx.push(&[0x40, 0]), Err(IsoTpError::Unexpected));
        assert!(is_flow_control(&[0x31]));
        assert!(!is_flow_control(&[0x21]));
        assert_eq!(Segmenter::new(&[]).unwrap_err(), IsoTpError::Malformed);
        assert_eq!(
            Segmenter::new(&[0u8; MAX_ISOTP_LEN + 1]).unwrap_err(),
            IsoTpError::TooLong
        );
        let mut tx = Segmenter::new(&[0u8; 20]).unwrap();
        assert_eq!(tx.flow_control(&[0x21]), Err(IsoTpError::Malformed));
        assert_eq!(tx.flow_control(&[0x33, 0, 0]), Err(IsoTpError::Malformed));
        assert_eq!(tx.flow_control(&[0x31, 0, 0]), Ok(()));
    }

    #[test]
    fn a_new_first_frame_restarts_a_partial_transfer() {
        let a: Vec<u8> = (0..30).collect();
        let b: Vec<u8> = (100..120).collect();
        let mut rx = Reassembler::<64>::new();
        let mut frame = [0u8; 8];
        let mut tx = Segmenter::new(&a).unwrap();
        tx.next_frame(&mut frame);
        rx.push(&frame).unwrap();
        let mut rx2 = Reassembler::<512>::new();
        assert_eq!(transfer(&b, &mut rx2), Some(b.clone()));
        let mut tx_b = Segmenter::new(&b).unwrap();
        tx_b.next_frame(&mut frame);
        assert_eq!(rx.push(&frame), Ok(Rx::FlowControl([0x30, 0, 0])));
    }
}
