//! USB bulk host link.
//!
//! # Description
//!
//! [`open`] claims a vendor bulk interface with `nusb` and returns a [`MessageLink`]. A message is
//! written as a bulk transfer that ends at a short packet or a zero-length packet, and a received
//! message is read until its short packet, matching the target's `usb` utilities.
//!
//! # Limitations
//!
//! - **Hardware**: The link is verified by compilation and against in-memory transports; a
//!   physical bulk target is required to exercise `open`.

use std::io::{self, Read, Write};
use std::time::Duration;

use nusb::MaybeFuture;
use nusb::io::{EndpointRead, EndpointWrite};
use nusb::transfer::{Bulk, In, Out};

use super::message::{MessageLink, MessageOf, MessageReader, MessageWriter};
use crate::error::HostError;

/// How long a read waits before it reports a timeout.
const READ_TIMEOUT: Duration = Duration::from_millis(50);
/// Transfer buffer size: a multiple of the full-speed and high-speed packet sizes.
const TRANSFER_SIZE: usize = 512;

/// Result of opening the bulk link.
pub type OpenOf = Result<MessageLink<UsbWriter>, HostError>;

/// Where to find the target's bulk interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsbBulkConfig {
    /// Address of the bulk IN endpoint.
    pub in_endpoint: u8,
    /// Interface number to claim.
    pub interface: u8,
    /// Address of the bulk OUT endpoint.
    pub out_endpoint: u8,
    /// USB product identifier.
    pub product_id: u16,
    /// USB vendor identifier.
    pub vendor_id: u16,
}

/// Reads messages from the bulk IN endpoint.
pub struct UsbReader {
    partial: Vec<u8>,
    reader: EndpointRead<Bulk>,
}

/// Writes messages to the bulk OUT endpoint.
pub struct UsbWriter {
    writer: EndpointWrite<Bulk>,
}

impl MessageReader for UsbReader {
    fn next_message(&mut self) -> MessageOf {
        let mut message = self.reader.until_short_packet();
        match message.read_to_end(&mut self.partial) {
            Ok(_) => {
                message.consume_end().map_err(|e| {
                    io::Error::new(io::ErrorKind::InvalidData, e)
                })?;
                Ok(Some(std::mem::take(&mut self.partial)))
            }
            Err(e) => on_error(e),
        }
    }
}

impl MessageWriter for UsbWriter {
    fn write_message(&mut self, msg: &[u8]) -> io::Result<()> {
        self.writer.write_all(msg)?;
        self.writer.flush_end()
    }
}

/// Opens the bulk interface described by `config`.
///
/// # Errors
/// Returns [`HostError::Transport`] if the device is absent or the interface cannot be claimed.
pub fn open(config: &UsbBulkConfig) -> OpenOf {
    let fail = |what: &str, e: &dyn std::fmt::Display| HostError::Transport {
        source: format!("USB bulk {what}: {e}").into(),
    };
    let info = nusb::list_devices()
        .wait()
        .map_err(|e| fail("enumeration failed", &e))?
        .find(|d| is_target(d.vendor_id(), d.product_id(), *config))
        .ok_or_else(|| fail("device not found", &"no matching device"))?;
    let device = info.open().wait().map_err(|e| fail("open failed", &e))?;
    let interface = device
        .claim_interface(config.interface)
        .wait()
        .map_err(|e| fail("claim failed", &e))?;
    let reader = interface
        .endpoint::<Bulk, In>(config.in_endpoint)
        .map_err(|e| fail("IN endpoint unavailable", &e))?
        .reader(TRANSFER_SIZE)
        .with_read_timeout(READ_TIMEOUT);
    let writer = interface
        .endpoint::<Bulk, Out>(config.out_endpoint)
        .map_err(|e| fail("OUT endpoint unavailable", &e))?
        .writer(TRANSFER_SIZE);
    let description = format!(
        "USB bulk {:04x}:{:04x}",
        config.vendor_id, config.product_id
    );
    Ok(MessageLink::new(
        UsbReader {
            partial: Vec::new(),
            reader,
        },
        UsbWriter { writer },
        description,
    ))
}

/// Returns whether a device with these identifiers is the one `config` describes.
const fn is_target(
    vendor_id: u16,
    product_id: u16,
    config: UsbBulkConfig,
) -> bool {
    vendor_id == config.vendor_id && product_id == config.product_id
}

/// Maps a failed read to a reader step; a timeout means no message is ready.
fn on_error(e: io::Error) -> MessageOf {
    if e.kind() == io::ErrorKind::TimedOut {
        Ok(None)
    } else {
        Err(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: UsbBulkConfig = UsbBulkConfig {
        in_endpoint: 0x81,
        interface: 0,
        out_endpoint: 0x01,
        product_id: 0x0002,
        vendor_id: 0x0001,
    };

    #[test]
    fn a_device_matches_only_on_both_identifiers() {
        assert!(is_target(0x0001, 0x0002, CONFIG));
        assert!(!is_target(0x0001, 0x0003, CONFIG));
        assert!(!is_target(0x0003, 0x0002, CONFIG));
        assert!(!is_target(0x0003, 0x0004, CONFIG));
    }

    #[test]
    fn a_read_timeout_is_idle_and_other_failures_propagate() {
        let idle = on_error(io::Error::from(io::ErrorKind::TimedOut));
        assert!(matches!(idle, Ok(None)));
        let failed = on_error(io::Error::from(io::ErrorKind::BrokenPipe));
        assert_eq!(failed.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }
}
