//! Runs the lifecycle task of the QEMU example firmware against a host-side plant.
//!
//! The firmware task (`examples/qemu/src/gain_task.rs`) multiplies each sample by its `GAIN`
//! setting. The host plant feeds the output back as the next input, so the sequence grows by the
//! gain each step. Build the firmware first (`cargo build --release` in `examples/qemu`), then run
//! `cargo run -p control-rs-ets-host --example qemu_task`.

use control_rs_ets::exec::{Packet, postcard_decode, postcard_encode};
use control_rs_ets::wire::{PacketInfo, PayloadEncoding, Verdict};
use control_rs_ets_host::catalog::ExecEntry;
use control_rs_ets_host::task_run::{MemoryObserver, TaskRun};
use control_rs_ets_host::{
    ETSBridge, HostError, RunControl, Session, Target, TaskOptions, TaskSim,
};

/// One scalar sample, as the firmware task declares it.
#[derive(serde::Serialize, serde::Deserialize)]
struct Sample {
    value: f32,
}

/// The plant: the output of step `k` is the input of step `k + 1`.
struct Loopback;

impl Packet for Sample {
    const INFO: PacketInfo<'static> = PacketInfo {
        type_name: "Sample",
        encoding: PayloadEncoding::Postcard,
    };

    fn decode(bytes: &[u8]) -> Option<Self> {
        postcard_decode(bytes)
    }

    fn encode(&self, out: &mut [u8]) -> Option<usize> {
        postcard_encode(self, out)
    }
}

impl TaskSim for Loopback {
    type Input = Sample;
    type Output = Sample;

    fn initial(&mut self) -> Sample {
        Sample { value: 1.0 }
    }

    fn advance(&mut self, _k: u64, y: &Sample) -> Sample {
        Sample { value: y.value }
    }
}

fn main() -> Result<(), HostError> {
    let mut session =
        Session::open(ETSBridge::new(Target::qemu_arm(), false)?)?;
    let task = session
        .catalog()
        .exec_ids()
        .find(|&id| session.catalog().exec(id).is_some_and(ExecEntry::is_task))
        .ok_or(HostError::Discovery)?;
    let mut observer = MemoryObserver::default();
    let control = RunControl::default();
    let record = session.run_task(TaskRun {
        control: &control,
        id: task,
        observer: &mut observer,
        opts: TaskOptions {
            max_steps: 5,
            ..TaskOptions::default()
        },
        sim: Loopback,
    })?;
    for (k, bytes) in &observer.steps {
        let value = Sample::decode(bytes).map(|s| s.value);
        println!("step {k}: {value:?}");
    }
    assert_eq!(record.verdict, Some(Verdict::Bounded));
    println!("{record:?}");
    session.close();
    Ok(())
}
