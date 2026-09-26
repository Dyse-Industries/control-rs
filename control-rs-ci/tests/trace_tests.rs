//! Command-line tests of `trace-reqs`, `trace-marks` and `trace-check`.

#[cfg(test)]
mod cli {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use regex::Regex;
    use serde_json::Value;

    /// A document without defects whose requirement names gate `test`.
    const CLEAN: &str = "\
- **FR-1 — Size**: The widget shall report its size.

| Requirements | Kind   | Gate   | Step |
|:-------------|:-------|:-------|:-----|
| FR-1         | `test` | `test` | Size |

| Requirements | Claim | Oracle | Measure | Bound |
|:-------------|:------|:-------|:--------|:------|
| FR-1         | Size  | Test   | Bytes   | Exact |
";

    /// The template's patterns, reading the `*-design.md` files under `docs`
    /// and markers in the `.rs` files under `src`.
    const CONFIG: &str = r##"id = '(?:FR|NFR|C)-[0-9]+[a-z]?'
doc = '[a-z0-9-]+'
files = ["docs"]
doc_suffix = "-design"
definition = '^- \*\*(?:FR|NFR|C)-'
retired = []

[references]
plan = '^\|[^|]+\| `[a-z-]+` +\|'
acceptance = '^\|(?:[^|]*\|){5}$'

[markers]
files = ["src"]
suffixes = [".rs"]
marker = "#[req("
"##;

    /// A duplicate definition, a missing acceptance reference and an
    /// unresolved reference.
    const DEFECTIVE: &str = "\
- **FR-1 — Size**: The widget shall report its size.
- **FR-1 — Again**: The widget shall repeat.

| FR-1, FR-7 | `test` | `test` | Size |
";

    /// A `gate.toml` that defines gate `test`.
    const GATES: &str = "[test]\ncommand = \"cargo test\"\n";

    /// A requirement whose plan names no gate.
    const UNCHECKED: &str = "\
- **FR-1 — Size**: The widget shall report its size.

| FR-1 | `inspection` |  | Review |

| FR-1 | Size | Review | Violations | 0 |
";

    /// A file to create: its path and contents.
    type Fixture<'a> = (&'a str, &'a str);

    /// A fresh working directory holding the configuration and `files`.
    fn workdir(name: &str, files: &[Fixture<'_>]) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("control_rs_ci_trace_{name}"));
        let _ = fs::remove_dir_all(&dir);
        for (path, text) in [("trace.toml", CONFIG), ("gate.toml", GATES)]
            .iter()
            .chain(files)
        {
            let path = dir.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        dir
    }

    fn run(bin: &str, dir: &Path, args: &[&str]) -> Output {
        Command::new(bin)
            .current_dir(dir)
            .args(args)
            .output()
            .unwrap()
    }

    fn trace_reqs(dir: &Path) -> Output {
        run(
            env!("CARGO_BIN_EXE_trace-reqs"),
            dir,
            &["--config", "trace.toml", "--out", "out/reqs.jsonl"],
        )
    }

    fn trace_marks(dir: &Path) -> Output {
        run(
            env!("CARGO_BIN_EXE_trace-marks"),
            dir,
            &["--config", "trace.toml", "--out", "out/marks.jsonl"],
        )
    }

    fn trace_check(dir: &Path) -> Output {
        run(
            env!("CARGO_BIN_EXE_trace-check"),
            dir,
            &[
                "--reqs",
                "out/reqs.jsonl",
                "--marks",
                "out/marks.jsonl",
                "--gates",
                "gate.toml",
                "--results",
                "results",
                "--out",
                "out/trace-report.json",
            ],
        )
    }

    fn stdout_lines(output: &Output) -> Vec<String> {
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn rows(path: &Path) -> Vec<Value> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    /// The string at a JSON pointer.
    fn text<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
        value.pointer(pointer).and_then(Value::as_str)
    }

    /// A `test.result.json` with `verdict`.
    fn result(verdict: &str) -> String {
        format!(
            "{{\"gate\":\"test\",\"verdict\":\"{verdict}\",\"exit_code\":0,\
             \"duration_secs\":0.1,\"summary\":null,\"log_file\":\"test.log\"}}"
        )
    }

    /// Runs `trace-reqs` on `doc`, then `trace-check` with the given results
    /// and markers; returns the check's exit code and report.
    fn check(name: &str, doc: &str, extra: &[Fixture<'_>]) -> (i32, Value) {
        let mut files =
            vec![("docs/widget-design.md", doc), ("out/marks.jsonl", "")];
        files.extend_from_slice(extra);
        let dir = workdir(name, &files);
        assert!(trace_reqs(&dir).status.success());
        let output = trace_check(&dir);
        let report =
            fs::read_to_string(dir.join("out/trace-report.json")).unwrap();
        (
            output.status.code().unwrap(),
            serde_json::from_str(&report).unwrap(),
        )
    }

    #[test]
    fn a_clean_document_passes_and_writes_its_rows() {
        let dir = workdir("clean", &[("docs/widget-design.md", CLEAN)]);
        let output = trace_reqs(&dir);
        assert_eq!(output.status.code(), Some(0));
        assert!(stdout_lines(&output).is_empty());
        let kinds: Vec<_> = rows(&dir.join("out/reqs.jsonl"))
            .iter()
            .map(|row| text(row, "/kind").unwrap().to_string())
            .collect();
        assert_eq!(kinds, ["definition", "plan", "acceptance"]);
    }

    #[test]
    fn defects_print_as_path_line_message_and_fail_the_run() {
        let dir = workdir("defects", &[("docs/widget-design.md", DEFECTIVE)]);
        let output = trace_reqs(&dir);
        assert_eq!(output.status.code(), Some(1));
        let lines = stdout_lines(&output);
        let format = Regex::new(r"^[^:]+:[0-9]+: .+$").unwrap();
        assert!(lines.iter().all(|line| format.is_match(line)), "{lines:?}");
        assert_eq!(
            lines,
            [
                "docs/widget-design.md:1: widget#FR-1 has no acceptance \
                 reference",
                "docs/widget-design.md:2: widget#FR-1 is defined more than \
                 once; first definition at docs/widget-design.md:1",
                "docs/widget-design.md:4: widget#FR-7 is not defined",
            ]
        );
    }

    #[test]
    fn rows_have_the_six_fields_and_reruns_are_identical() {
        let dir = workdir("rows", &[("docs/widget-design.md", DEFECTIVE)]);
        trace_reqs(&dir);
        let first = fs::read(dir.join("out/reqs.jsonl")).unwrap();
        trace_reqs(&dir);
        assert_eq!(fs::read(dir.join("out/reqs.jsonl")).unwrap(), first);
        let fields: BTreeSet<&str> =
            ["schema", "id", "kind", "file", "line", "text"].into();
        let rows = rows(&dir.join("out/reqs.jsonl"));
        for row in &rows {
            let keys: BTreeSet<&str> = row
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(keys, fields);
            assert_eq!(row.pointer("/schema").and_then(Value::as_u64), Some(1));
        }
        let lines: Vec<_> = rows
            .iter()
            .map(|r| r.pointer("/line").and_then(Value::as_u64))
            .collect();
        assert!(lines.is_sorted());
    }

    #[test]
    fn usage_and_configuration_errors_exit_with_code_two() {
        let dir = workdir("usage", &[]);
        let bin = env!("CARGO_BIN_EXE_trace-reqs");
        assert_eq!(run(bin, &dir, &["--help"]).status.code(), Some(0));
        assert_eq!(run(bin, &dir, &["--out", "o"]).status.code(), Some(2));
        assert_eq!(trace_reqs(&dir).status.code(), Some(2));
    }

    #[test]
    fn roots_select_suffixed_files_below_directories_and_named_files() {
        let dir = workdir(
            "roots",
            &[
                ("docs/widget-design.md", CLEAN),
                ("docs/deep/gadget-design.md", CLEAN),
                ("docs/notes.md", "- **FR-1 — Stray**: not read."),
                ("extra/solo.md", CLEAN),
            ],
        );
        let config = CONFIG.replace(
            "files = [\"docs\"]",
            "files = [\"docs\", \"extra/solo.md\"]",
        );
        fs::write(dir.join("trace.toml"), config).unwrap();
        assert_eq!(trace_reqs(&dir).status.code(), Some(0));
        let files: BTreeSet<String> = rows(&dir.join("out/reqs.jsonl"))
            .iter()
            .map(|row| text(row, "/file").unwrap().to_string())
            .collect();
        assert_eq!(
            files,
            [
                "docs/deep/gadget-design.md",
                "docs/widget-design.md",
                "extra/solo.md"
            ]
            .map(String::from)
            .into()
        );
    }

    #[test]
    fn a_missing_root_is_a_configuration_error() {
        let dir = workdir("missing-root", &[]);
        assert_eq!(trace_reqs(&dir).status.code(), Some(2));
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_links_are_never_followed() {
        let dir = workdir(
            "link",
            &[
                ("docs/widget-design.md", CLEAN),
                ("other/x-design.md", CLEAN),
            ],
        );
        std::os::unix::fs::symlink(
            "widget-design.md",
            dir.join("docs/alias-design.md"),
        )
        .unwrap();
        std::os::unix::fs::symlink("../other", dir.join("docs/other")).unwrap();
        assert_eq!(trace_reqs(&dir).status.code(), Some(0));
        let files: BTreeSet<String> = rows(&dir.join("out/reqs.jsonl"))
            .iter()
            .map(|row| text(row, "/file").unwrap().to_string())
            .collect();
        assert_eq!(files, ["docs/widget-design.md".to_string()].into());
    }

    #[test]
    fn trace_marks_scans_source_in_order_and_reruns_identically() {
        let dir = workdir(
            "marks",
            &[
                ("src/b.rs", "#[req(\"widget#FR-1\")]\n#[test]\nfn t() {}\n"),
                ("src/a/c.rs", "\n\n#[req(\"widget#FR-2\")]\nfn u() {}\n"),
                ("src/notes.txt", "#[req(\"widget#FR-3\")]\n"),
            ],
        );
        assert!(trace_marks(&dir).status.success());
        let first = fs::read(dir.join("out/marks.jsonl")).unwrap();
        let found: Vec<String> = rows(&dir.join("out/marks.jsonl"))
            .iter()
            .map(|row| {
                format!(
                    "{}:{}",
                    text(row, "/file").unwrap(),
                    text(row, "/id").unwrap()
                )
            })
            .collect();
        assert_eq!(found, ["src/a/c.rs:widget#FR-2", "src/b.rs:widget#FR-1"]);
        trace_marks(&dir);
        assert_eq!(fs::read(dir.join("out/marks.jsonl")).unwrap(), first);
    }

    #[test]
    fn an_unqualified_marker_fails_trace_marks() {
        let dir = workdir("bad-marker", &[("src/a.rs", "#[req(\"FR-1\")]\n")]);
        let output = trace_marks(&dir);
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(
            stdout_lines(&output),
            ["src/a.rs:1: marker ID FR-1 is not qualified as <doc>#<id>"]
        );
    }

    #[test]
    fn a_passing_gate_verifies_and_the_trace_passes() {
        let pass = result("pass");
        let (code, report) =
            check("verified", CLEAN, &[("results/test.result.json", &pass)]);
        assert_eq!(code, 0);
        assert_eq!(text(&report, "/requirements/0/status"), Some("Verified"));
        assert_eq!(text(&report, "/requirements/0/gates/test"), Some("pass"));
        assert_eq!(
            report.pointer("/counts/Verified").and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    fn a_failing_gate_fails_the_trace() {
        let fail = result("fail");
        let (code, report) =
            check("failed", CLEAN, &[("results/test.result.json", &fail)]);
        assert_eq!(code, 1);
        assert_eq!(text(&report, "/requirements/0/status"), Some("Failed"));
    }

    #[test]
    fn a_missing_or_skipped_result_leaves_it_unverified() {
        let (code, report) = check("unverified", CLEAN, &[]);
        assert_eq!(code, 1);
        assert_eq!(text(&report, "/requirements/0/status"), Some("Unverified"));
        assert_eq!(
            report.pointer("/requirements/0/gates/test"),
            Some(&Value::Null)
        );
        let skipped = result("skipped");
        let (code, _) =
            check("skipped", CLEAN, &[("results/test.result.json", &skipped)]);
        assert_eq!(code, 1);
    }

    #[test]
    fn a_requirement_without_gate_is_unchecked_and_passes() {
        let (code, report) = check("unchecked", UNCHECKED, &[]);
        assert_eq!(code, 0);
        assert_eq!(text(&report, "/requirements/0/status"), Some("Unchecked"));
    }

    #[test]
    fn an_unresolved_marker_fails_the_trace() {
        let pass = result("pass");
        let marker = "{\"schema\":1,\"id\":\"widget#FR-9\",\"kind\":\"marker\",\
                      \"file\":\"src/a.rs\",\"line\":4,\"text\":\"fn t\"}\n";
        let (code, report) = check(
            "marker",
            CLEAN,
            &[
                ("results/test.result.json", &pass),
                ("out/marks.jsonl", marker),
            ],
        );
        assert_eq!(code, 1);
        assert_eq!(
            text(&report, "/unresolved_markers/0/id"),
            Some("widget#FR-9")
        );
    }
}
