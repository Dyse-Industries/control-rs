//! Terminal User Interface (TUI) components.
//!
//! Provides the host-side interface for communicating with the target or background processes,
//! and the client-side server for managing the TUI state and event loop.

use std::io;

/// The host-side interface used by the TUI to interact with the system or target.
pub trait HostInterface {
    /// Writes data to the host interface.
    fn write_data(&mut self, data: &[u8]) -> io::Result<usize>;

    /// Reads data from the host interface.
    fn read_data(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

/// A generic implementation of `HostInterface` for standard streams.
pub struct StreamHostInterface<R, W> {
    reader: R,
    writer: W,
}

impl<R: io::Read, W: io::Write> StreamHostInterface<R, W> {
    /// Creates a new `StreamHostInterface` from a reader and writer.
    pub fn new(reader: R, writer: W) -> Self {
        Self { reader, writer }
    }
}

impl<R: io::Read, W: io::Write> HostInterface for StreamHostInterface<R, W> {
    fn write_data(&mut self, data: &[u8]) -> io::Result<usize> {
        self.writer.write(data)
    }

    fn read_data(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.reader.read(buf)
    }
}

/// The client-side server managing the TUI state and rendering loop.
pub struct ClientServer<H: HostInterface> {
    host: H,
    is_running: bool,
}

impl<H: HostInterface> ClientServer<H> {
    /// Initializes a new client-side server with the given host interface.
    pub fn new(host: H) -> Self {
        Self {
            host,
            is_running: false,
        }
    }

    /// Starts the server's main event loop.
    pub fn run(&mut self) -> io::Result<()> {
        self.is_running = true;

        let mut buffer = [0u8; 1024];

        while self.is_running {
            match self.host.read_data(&mut buffer) {
                Ok(0) => {
                    // EOF reached
                    break;
                }
                Ok(n) => {
                    self.process_payload(&buffer[..n])?;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                    continue;
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }

        Ok(())
    }

    /// Stops the server loop.
    pub fn stop(&mut self) {
        self.is_running = false;
    }

    /// Processes an incoming payload from the host interface.
    fn process_payload(&mut self, _payload: &[u8]) -> io::Result<()> {
        // TODO: decode payload and update TUI state
        Ok(())
    }
}