//! Command-line tests of `trace-reqs` and `trace-check`.

#[cfg(test)]
mod cli {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use control_rs_ci::trace::SCHEMA;
    use regex::Regex;
    use serde_json::Value;

    /// A document without defects: one requirement and one `libtest` condition.
    const CLEAN: &str = "\
# Widget (widget)

- **FR-1 — Size**: The widget shall report its size.

| Condition | Requirement | Method    | Target                | Criterion        |
|:----------|:------------|:----------|:----------------------|:-----------------|
| VC-1.1    | FR-1        | `libtest` | `widget::tests::size` | Exact size match |
";

    /// A duplicate definition, a requirement without condition and a
    /// condition with an undefined parent.
    const DEFECTIVE: &str = "\
# Widget (widget)

- **FR-1 — Size**: The widget shall report its size.
- **FR-1 — Again**: The widget shall repeat.

| Condition | Requirement | Method    | Target | Criterion |
|:----------|:------------|:----------|:-------|:----------|
| VC-7.1    | FR-7        | `libtest` | `a::t` | Other     |
";

    /// Three requirements: one `libtest` condition each for FR-1 and FR-2 and
    /// a `review` condition for FR-3.
    const MATRIX: &str = "\
# Widget (widget)

- **FR-1 — Size**: The widget shall report its size.
- **FR-2 — Mass**: The widget shall report its mass.
- **FR-3 — Style**: The widget shall follow the style guide.

| Condition | Requirement | Method    | Target                | Criterion        |
|:----------|:------------|:----------|:----------------------|:-----------------|
| VC-1.1    | FR-1        | `libtest` | `widget::tests::size` | Exact size match |
| VC-2.1    | FR-2        | `libtest` | `widget::tests::mass` | Exact mass match |
| VC-3.1    | FR-3        | `review`  | —                     | Style review     |
";

    /// A file to create: its path and contents.
    type Fixture<'a> = (&'a str, &'a str);

    fn config() -> String {
        "id = '(?:FR|NFR|C)-[0-9]+[a-z]?'\n\
         condition = 'VC-[0-9]+(?:\\.[0-9]+[a-z]?)?'\n\
         doc = '[a-z0-9-]+'\n\
         files = [\"docs\"]\n\
         doc_id = '^#\\s+.*\\((?P<doc>[a-z0-9-]+)\\)'\n\
         definition = '^- \\*\\*(?:FR|NFR|C)-'\n\
         verification = '^\\| *(?:[a-z0-9-]+#)?VC-'\n\
         methods = [\"libtest\", \"analysis\", \"inspection\", \"review\"]\n\
         automated_methods = [\"libtest\"]\n\
         retired = []\n\n\
         [method.libtest]\n\
         result_artifact = \"out/test.log\"\n"
            .to_string()
    }

    /// A fresh working directory holding the configuration and `files`.
    fn workdir(name: &str, files: &[Fixture<'_>]) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("control_rs_ci_trace_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cfg_str = config();
        for (path, text) in std::iter::once(("trace.toml", cfg_str.as_str()))
            .chain(files.iter().copied())
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

    fn trace_check(dir: &Path) -> Output {
        run(
            env!("CARGO_BIN_EXE_trace-check"),
            dir,
            &[
                "--config",
                "trace.toml",
                "--reqs",
                "out/reqs.jsonl",
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

    /// Runs `trace-reqs` on `doc`, then `trace-check` with the given
    /// fixtures; returns the check's exit code and report.
    fn check(name: &str, doc: &str, extra: &[Fixture<'_>]) -> (i32, Value) {
        let mut files = vec![("docs/widget-design.md", doc)];
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
        assert_eq!(
            output.status.code(),
            Some(0),
            "{:?}",
            stdout_lines(&output)
        );
        assert!(stdout_lines(&output).is_empty());
        let rows = rows(&dir.join("out/reqs.jsonl"));
        let kinds: Vec<_> = rows
            .iter()
            .map(|row| text(row, "/kind").unwrap().to_string())
            .collect();
        assert_eq!(kinds, ["definition", "condition"]);
        assert_eq!(
            rows.get(1).and_then(|r| text(r, "/targets/0")),
            Some("widget::tests::size")
        );
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
                "docs/widget-design.md:3: widget#FR-1 has no verification \
                 condition",
                "docs/widget-design.md:4: widget#FR-1 is defined more than \
                 once; first definition at docs/widget-design.md:3",
                "docs/widget-design.md:8: condition widget#VC-7.1 references \
                 undefined requirement widget#FR-7",
            ]
        );
    }

    #[test]
    fn rows_have_the_schema_fields_and_reruns_are_identical() {
        let dir = workdir("rows", &[("docs/widget-design.md", DEFECTIVE)]);
        trace_reqs(&dir);
        let first = fs::read(dir.join("out/reqs.jsonl")).unwrap();
        trace_reqs(&dir);
        assert_eq!(fs::read(dir.join("out/reqs.jsonl")).unwrap(), first);
        let common = ["schema", "id", "kind", "file", "line", "text"];
        let rows = rows(&dir.join("out/reqs.jsonl"));
        for row in &rows {
            let keys: BTreeSet<&str> = row
                .as_object()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            let mut fields: BTreeSet<&str> = common.into();
            if text(row, "/kind") == Some("condition") {
                fields.extend(["parents", "method", "targets"]);
            }
            assert_eq!(keys, fields);
            assert_eq!(
                row.pointer("/schema").and_then(Value::as_u64),
                Some(u64::from(SCHEMA))
            );
        }
        let lines: Vec<_> = rows
            .iter()
            .map(|r| r.pointer("/line").and_then(Value::as_u64))
            .collect();
        assert!(lines.is_sorted());
    }

    #[test]
    fn every_row_and_the_report_carry_the_schema_version() {
        let dir = workdir(
            "schema",
            &[
                ("docs/widget-design.md", CLEAN),
                ("out/test.log", "test widget::tests::size ... ok\n"),
            ],
        );
        assert!(trace_reqs(&dir).status.success());
        assert!(trace_check(&dir).status.success());
        let schema = Some(u64::from(SCHEMA));
        let reqs = rows(&dir.join("out/reqs.jsonl"));
        assert!(!reqs.is_empty());
        for row in &reqs {
            assert_eq!(row.pointer("/schema").and_then(Value::as_u64), schema);
        }
        let report: Value = serde_json::from_str(
            &fs::read_to_string(dir.join("out/trace-report.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(report.pointer("/schema").and_then(Value::as_u64), schema);
    }

    #[test]
    fn trace_check_rejects_rows_of_another_schema() {
        let other = format!(
            "{{\"schema\":{},\"id\":\"widget#FR-1\",\"kind\":\"definition\",\
             \"file\":\"docs/widget-design.md\",\"line\":1,\"text\":\"x\"}}\n",
            SCHEMA.saturating_add(1)
        );
        let dir = workdir("schema-other", &[("out/reqs.jsonl", &other)]);
        assert_eq!(trace_check(&dir).status.code(), Some(2));
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
        let gadget = CLEAN.replace("Widget (widget)", "Gadget (gadget)");
        let solo = CLEAN.replace("Widget (widget)", "Solo (solo)");
        let dir = workdir(
            "roots",
            &[
                ("docs/widget-design.md", CLEAN),
                ("docs/deep/gadget-design.md", &gadget),
                ("docs/notes.txt", "- **FR-1 — Stray**: not read."),
                ("extra/solo.md", &solo),
            ],
        );
        let config_str = config().replace(
            "files = [\"docs\"]",
            "files = [\"docs\", \"extra/solo.md\"]",
        );
        fs::write(dir.join("trace.toml"), config_str).unwrap();
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
    fn missing_doc_id_declaration_fails_trace_reqs() {
        let missing_id = "\
- **FR-1 — Size**: The widget shall report its size.

| Condition | Requirement | Method    | Target | Criterion |
|:----------|:------------|:----------|:-------|:----------|
| VC-1.1    | FR-1        | `libtest` | `a::t` | Exact size match |
";
        let dir =
            workdir("missing-id", &[("docs/widget-design.md", missing_id)]);
        let output = trace_reqs(&dir);
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(
            stdout_lines(&output),
            [
                "docs/widget-design.md:1: missing document ID declaration matching doc_id"
            ]
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
    fn passed_unrun_and_review_conditions_take_their_status() {
        let log = "test widget::tests::size ... ok\n";
        let (code, report) = check("matrix", MATRIX, &[("out/test.log", log)]);
        assert_eq!(code, 1);
        let status = |i: usize| {
            text(&report, &format!("/requirements/{i}/status"))
                .map(str::to_string)
        };
        assert_eq!(status(0).as_deref(), Some("Pass"));
        assert_eq!(status(1).as_deref(), Some("Unrun"));
        assert_eq!(status(2).as_deref(), Some("Review"));
        for (key, count) in [("Pass", 1), ("Unrun", 1), ("Review", 1)] {
            assert_eq!(
                report
                    .pointer(&format!("/counts/{key}"))
                    .and_then(Value::as_u64),
                Some(count),
                "{key}"
            );
        }
        assert_eq!(
            text(&report, "/requirements/0/conditions/0/items/0/target"),
            Some("widget::tests::size")
        );
        assert_eq!(text(&report, "/review/0/id"), Some("widget#VC-3.1"));
        assert_eq!(text(&report, "/review/0/method"), Some("review"));
    }

    #[test]
    fn every_automated_target_passed_passes_and_review_does_not_fail() {
        let log = "test widget::tests::size ... ok\n\
                   test widget::tests::mass ... ok\n";
        let (code, report) = check("covered", MATRIX, &[("out/test.log", log)]);
        assert_eq!(code, 0);
        assert_eq!(text(&report, "/requirements/2/status"), Some("Review"));
    }

    #[test]
    fn a_failed_target_fails_the_trace_with_a_located_defect() {
        let log = "test widget::tests::size ... FAILED\n";
        let dir = workdir(
            "failed",
            &[("docs/widget-design.md", CLEAN), ("out/test.log", log)],
        );
        assert!(trace_reqs(&dir).status.success());
        let output = trace_check(&dir);
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(
            stdout_lines(&output),
            ["docs/widget-design.md:7: condition widget#VC-1.1 target \
                 'widget::tests::size' failed in test.log"]
        );
    }
}
