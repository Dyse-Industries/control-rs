//! USB bulk transfer framing for a user's link.
//!
//! # Description
//!
//! A bulk message needs no ETS framing bytes: the transfer ends at a short packet. A message
//! whose length is an exact multiple of the endpoint packet size ends with a zero-length packet.
//! [`Packets`] splits a message into endpoint packets and [`Assembler`] joins received packets
//! back into the message. Both are driver-free, so a user's [`Link`](super::Link) composes them
//! with its endpoint calls.
//!
//! # Core Concepts
//!
//! - **Short packet**: A packet shorter than the endpoint maximum packet size ends a message.
//! - **Zero-length packet**: Follows a message whose length is a multiple of the packet size.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::link::usb::{Assembler, Packets};
//!
//! let message = [7u8; 128];
//! let mut rx = Assembler::<512>::new();
//! let mut out = None;
//! for packet in Packets::new(&message, 64) {
//!     if let Some(done) = rx.push(packet, 64)? {
//!         out = Some(done.len());
//!     }
//! }
//! assert_eq!(out, Some(128));
//! # Ok::<(), control_rs_ets::link::usb::Overflow>(())
//! ```

/// Splits a message into bulk packets of at most the endpoint packet size.
///
/// The last item is shorter than the packet size; when the message length is a multiple of the
/// packet size, the last item is the empty zero-length packet.
#[derive(Debug, Clone)]
pub struct Packets<'a> {
    done: bool,
    max_packet: usize,
    message: &'a [u8],
}

/// Joins received bulk packets into a message of at most `N` bytes.
#[derive(Debug, Clone)]
pub struct Assembler<const N: usize> {
    buf: [u8; N],
    complete: bool,
    /// After an overflow on a full packet, remaining packets of that transfer are ignored until a
    /// short packet ends it (FR-5).
    discarding: bool,
    len: usize,
}

/// Result of adding a packet: the finished message, if any, or an overflow.
pub type PushOf<'a> = Result<Option<&'a [u8]>, Overflow>;

/// The message does not fit the assembler; it is discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overflow;

impl<'a> Packets<'a> {
    /// Splits `message` for an endpoint with packet size `max_packet` (at least 1).
    #[must_use]
    pub const fn new(message: &'a [u8], max_packet: usize) -> Self {
        Self {
            done: max_packet == 0,
            max_packet,
            message,
        }
    }
}

impl<'a> Iterator for Packets<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        if self.message.len() < self.max_packet {
            self.done = true;
            return Some(self.message);
        }
        let (packet, rest) = self.message.split_at(self.max_packet);
        self.message = rest;
        Some(packet)
    }
}

impl<const N: usize> Default for Assembler<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Assembler<N> {
    /// Creates an empty assembler.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buf: [0; N],
            complete: false,
            discarding: false,
            len: 0,
        }
    }

    /// Discards any partial message.
    pub const fn reset(&mut self) {
        self.complete = false;
        self.discarding = false;
        self.len = 0;
    }

    /// Adds one received packet and returns the message when the packet ends it.
    ///
    /// A packet shorter than `max_packet` ends the message. An empty message is not returned.
    /// After an overflow on a full packet, later packets of the same transfer are discarded until
    /// a short packet arrives so a transfer tail cannot become a phantom message (FR-5).
    ///
    /// # Errors
    /// Returns [`Overflow`] and discards the message if it exceeds `N` bytes.
    pub fn push(&mut self, packet: &[u8], max_packet: usize) -> PushOf<'_> {
        if self.complete {
            self.reset();
        }
        if self.discarding {
            if packet.len() < max_packet {
                self.discarding = false;
            }
            return Ok(None);
        }
        let end = self.len.saturating_add(packet.len());
        let Some(dest) = self.buf.get_mut(self.len..end) else {
            // Overflow on a full packet leaves a short/ZLP terminator still on the wire.
            self.len = 0;
            self.complete = false;
            self.discarding = packet.len() >= max_packet;
            return Err(Overflow);
        };
        dest.copy_from_slice(packet);
        self.len = end;
        if packet.len() >= max_packet {
            return Ok(None);
        }
        if self.len == 0 {
            return Ok(None);
        }
        self.complete = true;
        Ok(self.buf.get(..self.len))
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec::Vec;

    type Message = Option<Vec<u8>>;

    fn round_trip(message: &[u8], max_packet: usize) -> Message {
        let mut rx = Assembler::<512>::new();
        let mut out = None;
        for packet in Packets::new(message, max_packet) {
            out = rx.push(packet, max_packet).unwrap().map(<[u8]>::to_vec);
        }
        out
    }

    #[test]
    fn message_bytes_equal_payload() {
        let message = [3u8; 150];
        let packets: Vec<usize> =
            Packets::new(&message, 64).map(<[u8]>::len).collect();
        assert_eq!(packets, std::vec![64, 64, 22]);
        let total: usize = packets.iter().sum();
        assert_eq!(total, message.len());
    }

    #[test]
    fn an_exact_multiple_ends_with_a_zero_length_packet() {
        let message = [3u8; 128];
        let packets: Vec<usize> =
            Packets::new(&message, 64).map(<[u8]>::len).collect();
        assert_eq!(packets, std::vec![64, 64, 0]);
        let one: Vec<usize> =
            Packets::new(&[1u8; 512], 512).map(<[u8]>::len).collect();
        assert_eq!(one, std::vec![512, 0]);
    }

    #[test]
    fn messages_round_trip_at_every_length_around_the_packet_size() {
        for len in [1usize, 63, 64, 65, 127, 128, 129, 511, 512] {
            let message: Vec<u8> =
                (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
            assert_eq!(round_trip(&message, 64), Some(message), "length {len}");
        }
    }

    #[test]
    fn a_message_larger_than_the_assembler_is_discarded() {
        let mut rx = Assembler::<8>::new();
        assert_eq!(rx.push(&[1; 8], 8), Ok(None));
        assert_eq!(rx.push(&[1; 1], 8), Err(Overflow));
        assert_eq!(rx.push(&[2; 3], 8), Ok(Some(&[2u8, 2, 2][..])));
    }

    #[test]
    fn overflow_on_a_full_packet_discards_the_rest_of_the_transfer() {
        let mut rx = Assembler::<512>::new();
        for _ in 0..8 {
            assert_eq!(rx.push(&[1; 64], 64), Ok(None));
        }
        assert_eq!(rx.push(&[1; 64], 64), Err(Overflow));
        // Later full packets in the oversized transfer must remain discarded and not clear the discarding state.
        assert_eq!(rx.push(&[9; 64], 64), Ok(None));
        // Short terminator of the oversized transfer must not assemble as a message.
        assert_eq!(rx.push(&[9; 24], 64), Ok(None));
        assert_eq!(rx.push(&[2; 3], 64), Ok(Some(&[2u8, 2, 2][..])));
    }

    #[test]
    fn the_assembler_starts_a_new_message_after_a_complete_one() {
        let mut rx = Assembler::<16>::new();
        assert_eq!(rx.push(&[1, 2], 4), Ok(Some(&[1u8, 2][..])));
        assert_eq!(rx.push(&[3], 4), Ok(Some(&[3u8][..])));
    }

    #[test]
    fn a_lone_zero_length_packet_is_not_a_message() {
        let mut rx = Assembler::<16>::new();
        assert_eq!(rx.push(&[], 4), Ok(None));
    }

    #[test]
    fn a_zero_packet_size_yields_no_packets() {
        assert_eq!(Packets::new(&[1, 2], 0).count(), 0);
    }
}
