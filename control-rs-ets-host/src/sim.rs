//! Host simulation hook for lifecycle tasks.
//!
//! A [`TaskSim`] produces the input packet of step `0` and maps the output
//! packet of step `k` to the input packet of step `k + 1`. The session stores
//! the byte-level [`ErasedSim`] form so one session type serves every packet
//! type.
//!
//! # Example
//!
//! ```
//! use control_rs_ets_host::sim::{TaskSim, erase};
//!
//! /// Integrator over `f32` packets.
//! struct Integrator(f32);
//! impl TaskSim for Integrator {
//!     type Input = f32;
//!     type Output = f32;
//!     fn initial(&mut self) -> f32 { self.0 }
//!     fn advance(&mut self, _: u64, u: f32) -> f32 { self.0 += u; self.0 }
//! }
//!
//! let mut sim = erase(Integrator(1.0));
//! let first = sim.initial_bytes().unwrap();
//! assert_eq!(postcard::from_bytes::<f32>(&first), Ok(1.0));
//! let mut u = [0u8; 8];
//! let u = postcard::to_slice(&-0.5_f32, &mut u).unwrap();
//! let next = sim.advance_bytes(0, u).unwrap();
//! assert_eq!(postcard::from_bytes::<f32>(&next), Ok(0.5));
//! ```

use std::vec::Vec;

use control_rs_ets::MAX_PACKET_SIZE;

/// A simulation packet's encoded bytes, or the failure to produce them.
pub type SimResult = Result<Vec<u8>, SimError>;

/// A boxed [`ErasedSim`].
pub type BoxedSim = Box<dyn ErasedSim>;

/// Failure to encode or decode a simulation packet.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SimError {
    /// The target's output packet does not decode as the simulation's output.
    #[error("output packet does not decode: {0}")]
    Decode(String),
    /// The simulation's input does not encode within `MAX_PACKET_SIZE`.
    #[error("input packet does not encode: {0}")]
    Encode(String),
}

/// A host-side plant simulation driven by a task's output packets.
pub trait TaskSim {
    /// The packet the simulation sends to the target.
    type Input: serde::Serialize;
    /// The packet the target sends to the simulation.
    type Output: serde::de::DeserializeOwned;

    /// The input of step `0`.
    fn initial(&mut self) -> Self::Input;

    /// The input of step `k + 1`, given the output of step `k`.
    fn advance(&mut self, k: u64, output: Self::Output) -> Self::Input;
}

/// Byte-level form of a [`TaskSim`].
pub trait ErasedSim: Send {
    /// The input of step `k + 1` as encoded bytes, given the encoded output
    /// of step `k`.
    ///
    /// # Errors
    ///
    /// Returns [`SimError`] when `output` does not decode or the next input
    /// does not encode.
    fn advance_bytes(&mut self, k: u64, output: &[u8]) -> SimResult;

    /// The input of step `0` as encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns [`SimError`] when the input does not encode.
    fn initial_bytes(&mut self) -> SimResult;

    /// The simulation's input type name, as `core::any::type_name` reports it.
    fn input_type(&self) -> &'static str;

    /// The simulation's output type name, as `core::any::type_name` reports it.
    fn output_type(&self) -> &'static str;
}

/// Adapter from a typed [`TaskSim`] to [`ErasedSim`].
struct Erased<S>(S);

impl<S> ErasedSim for Erased<S>
where
    S: TaskSim + Send,
    S::Input: 'static,
    S::Output: 'static,
{
    fn advance_bytes(&mut self, k: u64, output: &[u8]) -> SimResult {
        let output = postcard::from_bytes::<S::Output>(output)
            .map_err(|e| SimError::Decode(e.to_string()))?;
        encode(&self.0.advance(k, output))
    }

    fn initial_bytes(&mut self) -> SimResult {
        encode(&self.0.initial())
    }

    fn input_type(&self) -> &'static str {
        core::any::type_name::<S::Input>()
    }

    fn output_type(&self) -> &'static str {
        core::any::type_name::<S::Output>()
    }
}

/// Wraps `sim` in its byte-level form.
#[must_use]
pub fn erase<S>(sim: S) -> BoxedSim
where
    S: TaskSim + Send + 'static,
    S::Input: 'static,
    S::Output: 'static,
{
    Box::new(Erased(sim))
}

/// Encodes `value` into at most [`MAX_PACKET_SIZE`] bytes.
fn encode<T: serde::Serialize>(value: &T) -> SimResult {
    let mut buf = [0u8; MAX_PACKET_SIZE];
    postcard::to_slice(value, &mut buf)
        .map(|written| written.to_vec())
        .map_err(|e| SimError::Encode(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{BridgeMessage, OwnedTelemetry};
    use crate::session::{
        SessionAction, SessionState, TaskStart, lifecycle_session,
    };
    use control_rs_ets::comms::TaskRunState;
    use std::time::Instant;

    /// An input packet and the step it is for.
    type SentInput = (u64, Vec<u8>);

    /// Integrator over `f32` packets.
    struct Integrator {
        x: f32,
        advanced: Vec<u64>,
    }

    /// A simulation whose packet types are `u8`.
    struct Bytes;

    impl TaskSim for Integrator {
        type Input = f32;
        type Output = f32;

        fn initial(&mut self) -> f32 {
            self.x
        }

        fn advance(&mut self, k: u64, u: f32) -> f32 {
            self.advanced.push(k);
            self.x += u;
            self.x
        }
    }

    impl TaskSim for Bytes {
        type Input = u8;
        type Output = u8;

        fn initial(&mut self) -> u8 {
            0
        }

        fn advance(&mut self, _: u64, o: u8) -> u8 {
            o
        }
    }

    fn integrator(x: f32) -> BoxedSim {
        erase(Integrator {
            x,
            advanced: Vec::new(),
        })
    }

    fn pkt(v: f32) -> Vec<u8> {
        encode(&v).unwrap()
    }

    fn lockstep() -> TaskStart {
        TaskStart {
            suite_id: 0,
            max_steps: 100,
            lockstep: true,
            duration: None,
        }
    }

    fn running() -> BridgeMessage {
        BridgeMessage::Telemetry(OwnedTelemetry::TaskState {
            suite_id: 0,
            test_id: 0,
            state: TaskRunState::Running,
            message: None,
        })
    }

    fn sample(seq: u64, v: f32) -> BridgeMessage {
        BridgeMessage::Telemetry(OwnedTelemetry::TaskSample {
            suite_id: 0,
            test_id: 0,
            seq,
            payload: pkt(v),
        })
    }

    /// The `SendInput` among `actions`: `(seq, payload)`.
    fn sent_input(actions: &[SessionAction]) -> Option<SentInput> {
        actions.iter().find_map(|a| match a {
            SessionAction::SendInput { seq, payload, .. } => {
                Some((*seq, payload.clone()))
            }
            _ => None,
        })
    }

    fn session_with(sim: BoxedSim) -> SessionState {
        let mut s = lifecycle_session();
        s.set_sim(Some(sim));
        s
    }

    #[test]
    fn erased_sim_maps_output_k_to_input_k_plus_1() {
        let mut sim = integrator(1.0);
        assert_eq!(sim.initial_bytes().unwrap(), pkt(1.0));
        assert_eq!(sim.advance_bytes(0, &pkt(-0.5)).unwrap(), pkt(0.5));
        assert_eq!(sim.advance_bytes(1, &pkt(-0.25)).unwrap(), pkt(0.25));
        assert_eq!(sim.input_type(), "f32");
        assert_eq!(sim.output_type(), "f32");
    }

    #[test]
    fn erased_sim_reports_undecodable_output() {
        let mut sim = integrator(0.0);
        assert!(matches!(
            sim.advance_bytes(0, &[]),
            Err(SimError::Decode(_))
        ));
    }

    #[test]
    fn task_sim_drives_lockstep() {
        let mut s = session_with(integrator(1.0));
        let start = s.start_task(lockstep(), Instant::now()).unwrap();
        assert!(matches!(start, SessionAction::Send(_)));

        // Input 0 follows the run's first state.
        let first = s.handle_message(running());
        assert_eq!(sent_input(&first), Some((0, pkt(1.0))));

        // Each output k answers with input k + 1.
        let a = s.handle_message(sample(0, -0.5));
        assert_eq!(sent_input(&a), Some((1, pkt(0.5))));
        let b = s.handle_message(sample(1, -0.25));
        assert_eq!(sent_input(&b), Some((2, pkt(0.25))));

        let run = s.task_run.as_ref().unwrap();
        assert_eq!(
            run.inputs,
            [(0, pkt(1.0)), (1, pkt(0.5)), (2, pkt(0.25))],
            "inputs are recorded in send order"
        );
    }

    #[test]
    fn task_sim_type_mismatch_warns() {
        let mut s = session_with(erase(Bytes));
        assert!(s.start_task(lockstep(), Instant::now()).is_ok());
        assert_eq!(
            s.logs.matches("Warning: simulation packet types").count(),
            1,
            "one warning, and the run proceeds"
        );

        let mut ok = session_with(integrator(0.0));
        assert!(ok.start_task(lockstep(), Instant::now()).is_ok());
        assert!(!ok.logs.contains("Warning"), "matching types are silent");
    }
}
