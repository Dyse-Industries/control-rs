//! Message link between the target server and the host.
//!
//! # Description
//!
//! A [`Link`] moves one complete, verified message per call. The server encodes and decodes
//! messages (Postcard) and holds no framing code. A link delimits and verifies messages with the
//! framing native to its transport. Users implement [`Link`] for their device. The crate provides
//! [`StreamLink`] for byte streams (UART, USB CDC-ACM, semihosting) and the [`usb`] and [`can`]
//! utilities that a user's link composes with its driver.
//!
//! # Core Concepts
//!
//! - **Whole messages**: `recv` returns one complete message or nothing; `send` accepts all of a
//!   message or none of it.
//! - **Non-blocking**: Neither call waits. A link that cannot accept a message reports busy.
//! - **Close**: Dropping a link closes it. [`Link::close_on_failure`] is a provided method with an
//!   empty body that a link overrides to signal a failed run.
//! - **Utilities**: [`usb::Packets`], [`usb::Assembler`], [`can::Segmenter`] and
//!   [`can::Reassembler`] implement the protocol logic of a transport once, without a driver.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::link::{Link, RecvOf, SendOf};
//!
//! struct Loopback { held: Option<([u8; 8], usize)> }
//! impl Link for Loopback {
//!     type Error = ();
//!     fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self> {
//!         let Some((bytes, len)) = self.held.take() else { return Ok(None) };
//!         buf[..len].copy_from_slice(&bytes[..len]);
//!         Ok(Some(len))
//!     }
//!     fn send(&mut self, msg: &[u8]) -> SendOf<Self> {
//!         if self.held.is_some() || msg.len() > 8 { return Ok(false); }
//!         let mut bytes = [0u8; 8];
//!         bytes[..msg.len()].copy_from_slice(msg);
//!         self.held = Some((bytes, msg.len()));
//!         Ok(true)
//!     }
//! }
//!
//! let mut link = Loopback { held: None };
//! assert_eq!(link.send(&[1, 2, 3]), Ok(true));
//! let mut buf = [0u8; 8];
//! assert_eq!(link.recv(&mut buf), Ok(Some(3)));
//! ```

pub use stream::StreamLink;

pub mod can;
pub mod stream;
pub mod usb;

/// Result of a `ByteStream` read or write for stream `S`: a byte count or a transport error.
pub type CountOf<S> = Result<usize, <S as ByteStream>::Error>;

/// Result of `Link::recv` for link `L`: a message length, nothing ready, or a transport error.
pub type RecvOf<L> = Result<Option<usize>, <L as Link>::Error>;

/// Result of `Link::send` for link `L`: accepted whole (`true`), busy (`false`) or a transport error.
pub type SendOf<L> = Result<bool, <L as Link>::Error>;

/// Result of an operation on link `L` that returns nothing.
pub type UnitOf<L> = Result<(), <L as Link>::Error>;

/// Whole-message transport between the target server and the host.
///
/// # Safety
/// Implementations must guarantee safe hardware access during transfer. This trait does not use
/// `unsafe` code.
///
/// # Panics
/// Trait operations do not panic.
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

    /// Accepts all of `msg` or nothing. `Ok(false)` reports back-pressure and transmits nothing.
    ///
    /// # Errors
    /// Returns a transport error if writing fails.
    fn send(&mut self, msg: &[u8]) -> SendOf<Self>;

    /// Signals a failed run before the target resets. The fault path cannot drop the link, so a
    /// link that distinguishes failure from a clean close overrides this method.
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

    /// Signals a failed run before the target resets.
    fn close_on_failure(&mut self) {}
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    /// A stream that accepts at most `room` bytes in total.
    struct Capped {
        calls: usize,
        room: usize,
        written: usize,
    }

    impl ByteStream for Capped {
        type Error = ();

        fn read(&mut self, _buf: &mut [u8]) -> CountOf<Self> {
            Ok(0)
        }

        fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
            self.calls = self.calls.saturating_add(1);
            assert!(self.calls < 1000, "write loop does not terminate");
            let n = buf.len().min(self.room);
            self.room = self.room.saturating_sub(n);
            self.written = self.written.saturating_add(n);
            Ok(n)
        }
    }

    #[test]
    fn busy_send_transmits_nothing() {
        let mut link = StreamLink::new(Capped {
            calls: 0,
            room: 4,
            written: 0,
        });
        assert_eq!(link.send(&[1; 10]), Ok(true));
        let after_first = link.stream().written;
        assert_eq!(link.send(&[2; 3]), Ok(false));
        assert_eq!(
            link.stream().written,
            after_first,
            "a refused message adds no bytes"
        );
        link.stream_mut().room = 100;
        let mut buf = [0u8; 4];
        assert_eq!(link.recv(&mut buf), Ok(None));
        assert_eq!(link.send(&[2; 3]), Ok(true));
        assert_eq!(link.stream().written, 16 + 9);
    }

    #[test]
    fn oversize_send_is_error() {
        let mut link = StreamLink::new(Capped {
            calls: 0,
            room: 4096,
            written: 0,
        });
        assert_eq!(link.send(&[0; 513]), Ok(false));
        assert_eq!(link.stream().written, 0);
        let mut packets = usb::Packets::new(&[0; 513], 64);
        assert_eq!(packets.by_ref().count(), 9);
        assert_eq!(
            can::Segmenter::new(&[0; 4096]).unwrap_err(),
            can::IsoTpError::TooLong
        );
    }
}
