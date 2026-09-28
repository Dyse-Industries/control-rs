//! Process-level tests of the `allow-audit` binary.

#[cfg(test)]
mod allow_audit_process {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    const BIN: &str = env!("CARGO_BIN_EXE_allow-audit");

    /// One suppression site, as it appears in `a.rs`.
    const SOURCE: &str = "#[allow(clippy::panic)]\nfn f() {}\n";

    fn tree(name: &str, baseline: &str) -> PathBuf {
        let root = std::env::temp_dir()
            .join(format!("control_rs_ci_audit_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("a.rs"), SOURCE).unwrap();
        fs::write(root.join("baseline.txt"), baseline).unwrap();
        root
    }

    fn audit(root: &Path, args: &[&str]) -> Output {
        Command::new(BIN)
            .current_dir(root)
            .args(["--root", ".", "--baseline", "baseline.txt"])
            .args(args)
            .env("NO_COLOR", "1")
            .output()
            .unwrap()
    }

    fn text(out: &Output) -> String {
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    }

    fn git(root: &Path, args: &[&str]) {
        let status = Command::new("git")
            .current_dir(root)
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
            .args(args)
            .output()
            .unwrap()
            .status;
        assert!(status.success(), "git {args:?} failed");
    }

    #[test]
    fn help_prints_usage_and_succeeds() {
        let out = Command::new(BIN).arg("--help").output().unwrap();
        assert_eq!(out.status.code(), Some(0));
        assert!(text(&out).contains("Usage:"));
    }

    #[test]
    fn a_matching_baseline_passes() {
        let root = tree("clean", "a.rs clippy::panic 1\n");
        let out = audit(&root, &[]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        assert!(text(&out).contains("none new, none stale"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_new_suppression_fails_without_a_stale_report() {
        let root = tree("added", "");
        let out = audit(&root, &[]);
        assert_eq!(out.status.code(), Some(1));
        let text = text(&out);
        assert!(text.contains("1 new clippy suppressions"), "{text}");
        assert!(!text.contains("list more sites than remain"), "{text}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_stale_baseline_entry_fails_without_a_new_report() {
        let root =
            tree("stale", "a.rs clippy::panic 1\nb.rs clippy::panic 1\n");
        let out = audit(&root, &[]);
        assert_eq!(out.status.code(), Some(1));
        let text = text(&out);
        assert!(text.contains("list more sites than remain"), "{text}");
        assert!(!text.contains("new clippy suppressions"), "{text}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_baseline_that_grew_since_the_base_ref_fails() {
        let root = tree("growth", "");
        git(&root, &["init", "-q"]);
        git(&root, &["add", "-A"]);
        git(&root, &["commit", "-q", "-m", "base"]);
        // The baseline now lists the site the tree already has: no new or stale
        // entry, but the baseline is larger than at HEAD.
        fs::write(root.join("baseline.txt"), "a.rs clippy::panic 1\n").unwrap();
        let out = audit(&root, &["--base-ref", "HEAD"]);
        assert_eq!(out.status.code(), Some(1), "{}", text(&out));
        assert!(text(&out).contains("exceed HEAD"));

        // The same baseline committed at the ref is not growth.
        git(&root, &["add", "-A"]);
        git(&root, &["commit", "-q", "-m", "grow"]);
        let out = audit(&root, &["--base-ref", "HEAD"]);
        assert_eq!(out.status.code(), Some(0), "{}", text(&out));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unreadable_base_ref_is_a_usage_error() {
        let root = tree("bad_ref", "a.rs clippy::panic 1\n");
        git(&root, &["init", "-q"]);
        let out = audit(&root, &["--base-ref", "no-such-ref"]);
        assert_eq!(out.status.code(), Some(2));
        assert!(text(&out).contains("Cannot read"));
        let _ = fs::remove_dir_all(&root);
    }
}
