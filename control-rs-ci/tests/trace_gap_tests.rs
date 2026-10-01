//! Trace scanner edge case: the fallback method cell.

#[cfg(test)]
mod trace_gap {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output};

    use serde_json::Value;

    /// A file to create: its path and contents.
    type Fixture<'a> = (&'a str, &'a str);

    fn config() -> String {
        String::from(
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
         result_artifact = \"test.log\"\n",
        )
    }

    fn workdir(name: &str, files: &[Fixture<'_>]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_ci_trace_gap_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let cfg = config();
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
    fn the_fallback_method_is_the_first_code_span_without_any_id() {
        let doc = "\
# Widget (widget)

- **FR-1 — Size**: The widget shall report its size.

| Condition | Requirement | Method | Criterion |
|:----------|:------------|:-------|:----------|
| VC-1.1    | FR-1        | `FR-1` `bogus` | Exact |
";
        let dir = workdir("method", &[("docs/widget-design.md", doc)]);
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
