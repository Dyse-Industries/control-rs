//! Quality gate catalog and runner registry.

pub mod cargo;
pub mod coverage;
pub mod cross_compare;
pub mod deny;
pub mod geiger;
pub mod git;
pub mod metrics;
pub mod mutants;
pub mod semver;
pub mod vale;
pub mod valgrind;

pub use self::cargo::CargoArgvGate;
pub use self::coverage::CoverageGate;
pub use self::cross_compare::CrossCompareGate;
pub use self::deny::DenyGate;
pub use self::geiger::GeigerGate;
pub use self::git::GitHygieneGate;
pub use self::metrics::MetricsGate;
pub use self::mutants::MutantsGate;
pub use self::semver::SemverGate;
pub use self::vale::ValeGate;
pub use self::valgrind::ValgrindGate;

use std::sync::Arc;

use crate::config::{GateConfig, GatePolicy};
use crate::quality_gate::QualityGate;

/// Instantiates all built-in quality gates according to the workspace configuration.
#[must_use]
pub fn build_all_gates(config: &GateConfig) -> Vec<Arc<dyn QualityGate>> {
    let mut gates: Vec<Arc<dyn QualityGate>> = Vec::new();

    // Canonical execution sequence:
    // 1. Clean (if configured)
    if config.policy_for("clean") != GatePolicy::Skip {
        gates.push(Arc::new(CargoArgvGate::clean()));
    }
    // 2. Format
    if config.policy_for("fmt") != GatePolicy::Skip {
        gates.push(Arc::new(CargoArgvGate::fmt()));
    }
    // 3. Clippy
    if config.policy_for("clippy") != GatePolicy::Skip {
        gates.push(Arc::new(CargoArgvGate::clippy()));
    }
    // 4. Type Check
    if config.policy_for("check") != GatePolicy::Skip {
        gates.push(Arc::new(CargoArgvGate::check()));
    }
    // 5. Build
    if config.policy_for("build") != GatePolicy::Skip {
        gates.push(Arc::new(CargoArgvGate::build()));
    }
    // 6. Test
    if config.policy_for("test") != GatePolicy::Skip {
        gates.push(Arc::new(CargoArgvGate::test()));
    }
    // 7. Coverage
    if config.policy_for("coverage") != GatePolicy::Skip {
        gates.push(Arc::new(CoverageGate));
    }
    // 8. Metrics
    if config.policy_for("metrics") != GatePolicy::Skip {
        gates.push(Arc::new(MetricsGate::new(config.metrics.clone())));
    }
    // 9. Git Hygiene
    if config.policy_for("git") != GatePolicy::Skip {
        gates.push(Arc::new(GitHygieneGate::new(config.git.clone())));
    }
    // 10. Vale Prose Linter
    if config.policy_for("vale") != GatePolicy::Skip {
        gates.push(Arc::new(ValeGate::new(config.vale.clone())));
    }
    // 11. Cargo Deny
    if config.policy_for("deny") != GatePolicy::Skip {
        gates.push(Arc::new(DenyGate));
    }
    // 12. Cargo Geiger
    if config.policy_for("geiger") != GatePolicy::Skip {
        gates.push(Arc::new(GeigerGate::new(config.geiger.clone())));
    }
    // 13. Cargo SemVer
    if config.policy_for("semver") != GatePolicy::Skip {
        gates.push(Arc::new(SemverGate::new(config.semver.clone())));
    }
    // 14. Cargo Mutants
    if config.policy_for("mutants") != GatePolicy::Skip {
        gates.push(Arc::new(MutantsGate::new(config.mutants.clone())));
    }
    // 15. Valgrind Memcheck
    if config.policy_for("valgrind") != GatePolicy::Skip {
        gates.push(Arc::new(ValgrindGate::new(config.valgrind.clone())));
    }
    // 16. Cross-Comparison Gate
    if config.policy_for("cross-compare") != GatePolicy::Skip {
        gates.push(Arc::new(CrossCompareGate));
    }

    gates
}
