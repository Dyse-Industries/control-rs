//! Project size metrics for the report.
//!
//! Counts files and lines per top-level directory and language. Rust and
//! Python lines are code, comment, documentation or blank; Markdown lines are
//! documentation or blank. A line inside a `/* */` block comment counts as
//! code. Directories named `target` or starting with `.` are skipped, and
//! symbolic links are not followed.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

/// Line counts of one language.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Number of files.
    pub files: usize,
    /// Lines of code.
    pub code: usize,
    /// Plain comment lines.
    pub comment: usize,
    /// Documentation lines: Rust `///` and `//!`, and every non-blank
    /// Markdown line.
    pub doc: usize,
    /// Blank lines.
    pub blank: usize,
}

/// Counts per language name.
pub type Languages = BTreeMap<String, Counts>;

/// Counts of a tree.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Metrics {
    /// Counts per top-level directory; files at the root are under `.`.
    pub areas: BTreeMap<String, Languages>,
    /// Counts per language over the whole tree.
    pub total: Languages,
}

impl Metrics {
    /// A Markdown table of counts per language over the whole tree.
    #[must_use]
    pub fn render(&self) -> String {
        let mut out = String::from(
            "| Language | Files | Code | Comment | Doc | Blank | Lines |\n|:---|---:|---:|---:|---:|---:|---:|\n",
        );
        for (lang, c) in &self.total {
            let _ = writeln!(
                out,
                "| {lang} | {} | {} | {} | {} | {} | {} |",
                c.files,
                c.code,
                c.comment,
                c.doc,
                c.blank,
                c.lines()
            );
        }
        out
    }
}

impl Counts {
    /// Sum of every line.
    #[must_use]
    pub const fn lines(&self) -> usize {
        self.code
            .saturating_add(self.comment)
            .saturating_add(self.doc)
            .saturating_add(self.blank)
    }

    const fn add(&mut self, other: &Self) {
        self.files = self.files.saturating_add(other.files);
        self.code = self.code.saturating_add(other.code);
        self.comment = self.comment.saturating_add(other.comment);
        self.doc = self.doc.saturating_add(other.doc);
        self.blank = self.blank.saturating_add(other.blank);
    }
}

/// The language of a file extension, if counted.
fn language(ext: &str) -> Option<&'static str> {
    match ext {
        "rs" => Some("Rust"),
        "md" => Some("Markdown"),
        "py" => Some("Python"),
        "toml" => Some("TOML"),
        _ => None,
    }
}

/// Counts the lines of `text` written in `lang`.
#[must_use]
pub fn count(lang: &str, text: &str) -> Counts {
    let mut counts = Counts {
        files: 1,
        ..Counts::default()
    };
    for line in text.lines().map(str::trim) {
        let slot = if line.is_empty() {
            &mut counts.blank
        } else {
            match lang {
                "Markdown" => &mut counts.doc,
                "Rust"
                    if line.starts_with("///") || line.starts_with("//!") =>
                {
                    &mut counts.doc
                }
                "Rust" if line.starts_with("//") => &mut counts.comment,
                "Python" | "TOML" if line.starts_with('#') => {
                    &mut counts.comment
                }
                _ => &mut counts.code,
            }
        };
        *slot = slot.saturating_add(1);
    }
    counts
}

/// Measures every counted file under `root`.
///
/// # Errors
/// Returns an I/O error if a directory or file cannot be read.
pub fn measure(root: &Path) -> io::Result<Metrics> {
    let mut metrics = Metrics::default();
    visit(root, root, &mut metrics)?;
    Ok(metrics)
}

fn visit(root: &Path, dir: &Path, metrics: &mut Metrics) -> io::Result<()> {
    let mut entries: Vec<_> =
        fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            if name != "target" && !name.starts_with('.') {
                visit(root, &path, metrics)?;
            }
            continue;
        }
        let lang = path.extension().and_then(|e| e.to_str()).and_then(language);
        let Some(lang) = lang.filter(|_| kind.is_file()) else {
            continue;
        };
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let counts = count(lang, &text);
        let area = path
            .strip_prefix(root)
            .ok()
            .filter(|rel| rel.components().count() > 1)
            .and_then(|rel| rel.components().next())
            .map_or_else(
                || ".".to_string(),
                |c| c.as_os_str().to_string_lossy().into_owned(),
            );
        metrics
            .areas
            .entry(area)
            .or_default()
            .entry(lang.to_string())
            .or_default()
            .add(&counts);
        metrics
            .total
            .entry(lang.to_string())
            .or_default()
            .add(&counts);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_lines_split_into_code_comment_doc_and_blank() {
        let c = count(
            "Rust",
            "//! crate\n\n/// item\nfn f() {}\n  // note\nlet x = 1; // tail\n",
        );
        assert_eq!(
            (c.files, c.code, c.comment, c.doc, c.blank, c.lines()),
            (1, 2, 1, 2, 1, 6)
        );
    }

    #[test]
    fn markdown_and_python_lines() {
        let md = count("Markdown", "# T\n\ntext\n");
        assert_eq!((md.doc, md.blank, md.code), (2, 1, 0));
        let py = count("Python", "# c\nx = 1\n\n");
        assert_eq!((py.comment, py.code, py.blank), (1, 1, 1));
    }

    #[test]
    fn measure_groups_by_area_and_skips_target_and_hidden() {
        let dir = std::env::temp_dir()
            .join(format!("control_rs_ci_metrics_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        for sub in ["a/src", "target/x", ".hid"] {
            fs::create_dir_all(dir.join(sub)).unwrap();
        }
        fs::write(dir.join("a/src/l.rs"), "fn f() {}\n").unwrap();
        fs::write(dir.join("a/n.txt"), "ignored\n").unwrap();
        fs::write(dir.join("README.md"), "# R\n").unwrap();
        fs::write(dir.join("target/x/t.rs"), "fn t() {}\n").unwrap();
        fs::write(dir.join(".hid/h.rs"), "fn h() {}\n").unwrap();
        let m = measure(&dir).unwrap();
        let _ = fs::remove_dir_all(&dir);
        assert_eq!(m.areas.keys().collect::<Vec<_>>(), [".", "a"]);
        assert_eq!(m.total.get("Rust").map(|c| c.files), Some(1));
        assert_eq!(m.total.get("Markdown").map(|c| c.doc), Some(1));
        let table = m.render();
        assert!(
            table.contains("| Rust | 1 | 1 | 0 | 0 | 0 | 1 |"),
            "{table}"
        );
        assert!(
            table.contains("| Markdown | 1 | 0 | 0 | 1 | 0 | 1 |"),
            "{table}"
        );
    }
}
