//! Runs suite discovery over any [`HostLink`] and reports what the link saw.
//!
//! The same generic code drives QEMU, serial and in-memory links. This
//! example uses the in-memory link, so it needs no target. Run it with
//! `cargo run -p control-rs-ets-host --features fake-link --example link_discovery`.

use control_rs_ets_host::link::discover;
use control_rs_ets_host::{
    BridgeMessage, ETSBridge, HostError, HostLink, LinkEvent, OwnedTelemetry,
    Polled,
};

/// Events gathered from a link, or its transport error.
type Collected<L> = Result<Vec<LinkEvent>, <L as HostLink>::Error>;

/// Sends discovery and collects every pending event.
fn collect<L: HostLink>(link: &mut L) -> Collected<L> {
    discover(link)?;
    let mut events = Vec::new();
    while let Polled::Event(event) = link.poll() {
        events.push(event);
    }
    Ok(events)
}

fn main() -> Result<(), HostError> {
    let mut fake = ETSBridge::fake(None);
    let sent = fake
        .tx
        .send(BridgeMessage::Telemetry(OwnedTelemetry::DiscoveryComplete));
    assert!(sent.is_ok(), "the fake target channel is open");
    let events = collect(&mut fake.bridge)?;
    println!("link: {}", fake.bridge.describe());
    for event in &events {
        println!("event: {event:?}");
    }
    Ok(())
}
