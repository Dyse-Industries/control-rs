//! Trace scanner edge cases: custom marker spans and fallback method cells.

#[cfg(test)]
mod trace_gap {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use serde_json::Value;

    /// A file to create: its path and contents.
    type Fixture<'a> = (&'a str, &'a str);

    fn config(marker: &str) -> String {
        format!(
            "id = '(?:FR|NFR|C)-[0-9]+[a-z]?'\n\
         condition = 'VC-[0-9]+(?:\\.[0-9]+[a-z]?)?'\n\
         doc = '[a-z0-9-]+'\n\
         files = [\"docs\"]\n\
         doc_id = '^#\\s+.*\\((?P<doc>[a-z0-9-]+)\\)'\n\
         definition = '^- \\*\\*(?:FR|NFR|C)-'\n\
         verification = '^\\| *(?:[a-z0-9-]+#)?VC-'\n\
         methods = [\"test\", \"analysis\", \"inspection\", \"review\"]\n\
         marked_methods = [\"test\"]\n\
         retired = []\n\n\
         [markers]\n\
         files = [\"src\"]\n\
         suffixes = [\".rs\"]\n\
         marker = \"{marker}\"\n"
        )
    }

    fn workdir(name: &str, marker: &str, files: &[Fixture<'_>]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_ci_trace_gap_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let cfg = config(marker);
        for (path, text) in std::iter::once(("trace.toml", cfg.as_str()))
            .chain(files.iter().copied())
        {
            let path = dir.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, text).unwrap();
        }
        dir
    }

    fn run(bin: &str, dir: &Path, out: &str) -> Output {
        Command::new(bin)
            .current_dir(dir)
            .args(["--config", "trace.toml", "--out", out])
            .output()
            .unwrap()
    }

    fn rows(path: &Path) -> Vec<Value> {
        fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn a_parenthesised_custom_marker_spans_lines_until_it_closes() {
        let source = "// req(\n    \"widget#VC-1.1\",\n    \"widget#VC-2.1\"\n)\nfn t() {}\n";
        let dir = workdir("span", "// req(", &[("src/a.rs", source)]);
        let output =
            run(env!("CARGO_BIN_EXE_trace-marks"), &dir, "out/marks.jsonl");
        assert_eq!(output.status.code(), Some(0), "{output:?}");
        let ids: Vec<_> = rows(&dir.join("out/marks.jsonl"))
            .iter()
            .filter_map(|row| {
                row.pointer("/id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(ids, ["widget#VC-1.1", "widget#VC-2.1"]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_fallback_method_is_the_first_code_span_without_any_id() {
        let doc = "\
# Widget (widget)

- **FR-1 — Size**: The widget shall report its size.

| Condition | Requirement | Method | Criterion |
|:----------|:------------|:-------|:----------|
| VC-1.1    | FR-1        | `FR-1` `bogus` | Exact |
";
        let dir =
            workdir("method", "unused", &[("docs/widget-design.md", doc)]);
        // The unknown method makes the run fail, but the rows are still written.
        let _ = run(env!("CARGO_BIN_EXE_trace-reqs"), &dir, "out/reqs.jsonl");
        let methods: Vec<_> = rows(&dir.join("out/reqs.jsonl"))
            .iter()
            .filter_map(|row| {
                row.pointer("/method")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        assert_eq!(methods, ["bogus"]);
        let _ = fs::remove_dir_all(&dir);
    }
}
