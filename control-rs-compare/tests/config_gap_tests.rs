//! Default values, tolerance lookup and suite resolution edge cases.

#[cfg(test)]
mod config_gaps {
    use std::fs;
    use std::path::{Path, PathBuf};

    use control_rs_compare::config::{CompareConfigFile, ToleranceTable};

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "control_rs_compare_cfg_{name}_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn parse(text: &str) -> CompareConfigFile {
        toml::from_str(text).unwrap()
    }

    #[test]
    fn omitted_settings_take_their_documented_defaults() {
        for general in [
            parse("").compare,
            parse("[compare]\n").compare,
            CompareConfigFile::default().compare,
        ] {
            assert_eq!(general.title, "control-rs Cross-Comparison Suite");
            assert_eq!(general.out_dir, "results");
            assert_eq!(general.timeout_secs, 120);
            assert!(general.strict);
        }

        let config = parse("[[suite]]\nname = \"s\"\n");
        let suite = config.inlined_suites.first().unwrap();
        assert_eq!(suite.true_oracle, "scipy");

        let table: ToleranceTable =
            toml::from_str("[tolerances.\"a/b\"]\nbound = 1.0\n").unwrap();
        assert_eq!(table.find_signal("a/b").unwrap().policy, "all_of");
    }

    #[test]
    fn signals_are_found_by_key_table_and_declared_name() {
        let table: ToleranceTable = toml::from_str(
            "[tolerances.first]\nsignal = \"matrix/a\"\nbound = 1.0\n\
             [tolerances.second]\nsignal = \"matrix/b\"\nbound = 2.0\n\
             [signals.\"direct/key\"]\nbound = 3.0\n",
        )
        .unwrap();
        assert_eq!(table.find_signal("matrix/a").unwrap().bound, Some(1.0));
        assert_eq!(table.find_signal("matrix/b").unwrap().bound, Some(2.0));
        assert_eq!(table.find_signal("direct/key").unwrap().bound, Some(3.0));
        assert!(table.find_signal("matrix/c").is_none());
    }

    fn write_suite(dir: &Path, rel: &str, body: &str) {
        let path = dir.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }

    #[test]
    fn child_suites_are_loaded_once_and_overridden_by_inline_suites() {
        let dir = scratch("resolve");
        write_suite(
            &dir,
            "child/compare.toml",
            "[[suite]]\nname = \"shared\"\ntrue_oracle = \"child\"\n\
             [[suite]]\nname = \"only_child\"\n",
        );
        write_suite(
            &dir,
            "root/compare.toml",
            "suites = [\"../child\"]\n\
             [compare]\nsuites = [\"../child\"]\n\
             [[suite]]\nname = \"shared\"\ntrue_oracle = \"inline\"\n\
             [[suite]]\nname = \"only_inline\"\n",
        );
        let path = dir.join("root/compare.toml");
        let plan = CompareConfigFile::load_from_file(&path)
            .unwrap()
            .resolve_master_plan(&path)
            .unwrap();
        let names: Vec<_> =
            plan.suites.iter().map(|s| s.name.as_str()).collect();
        // The child listed twice loads once; an inline suite replaces the
        // child suite of the same name in place and others are appended.
        assert_eq!(names, ["shared", "only_child", "only_inline"]);
        assert_eq!(plan.suites.first().unwrap().true_oracle, "inline");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_child_suite_is_an_error_naming_it() {
        let dir = scratch("missing");
        write_suite(&dir, "root/compare.toml", "suites = [\"../absent\"]\n");
        let path = dir.join("root/compare.toml");
        let err = CompareConfigFile::load_from_file(&path)
            .unwrap()
            .resolve_master_plan(&path)
            .unwrap_err()
            .to_string();
        assert!(err.contains("'../absent' not found"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn suite_paths_are_resolved_against_the_config_directory() {
        let config = parse(
            "[[suite]]\nname = \"s\"\ntolerance_table = \"./tol/../tolerances.toml\"\n\
             [[suite.variants]]\nname = \"v\"\ntype = \"command\"\n\
             output_file = \"../../out/./v.h5\"\n\
             manifest_path = \"./Cargo.toml\"\nscript = \"./run.py\"\n",
        );
        let plan = config
            .resolve_master_plan(Path::new("base/sub/compare.toml"))
            .unwrap();
        let suite = plan.suites.first().unwrap();
        assert_eq!(
            suite.tolerance_table.as_deref(),
            Some("base/sub/tolerances.toml")
        );
        let variant = suite.variants.first().unwrap();
        assert_eq!(variant.output_file, "out/v.h5");
        assert_eq!(
            variant.manifest_path.as_deref(),
            Some("base/sub/Cargo.toml")
        );
        assert_eq!(variant.script.as_deref(), Some("base/sub/run.py"));
    }

    #[test]
    fn parent_components_beyond_the_base_are_kept() {
        let config = parse(
            "[[suite]]\nname = \"s\"\n\
             [[suite.variants]]\nname = \"v\"\ntype = \"command\"\n\
             output_file = \"../../../out.h5\"\n",
        );
        let plan = config
            .resolve_master_plan(Path::new("sub/compare.toml"))
            .unwrap();
        let variant = plan.suites.first().unwrap().variants.first().unwrap();
        assert_eq!(variant.output_file, "../../out.h5");
    }

    #[test]
    fn absolute_paths_are_left_untouched() {
        let config = parse(
            "[[suite]]\nname = \"s\"\ntolerance_table = \"/abs/../tol.toml\"\n\
             [[suite.variants]]\nname = \"v\"\ntype = \"command\"\n\
             output_file = \"/abs/../out.h5\"\n",
        );
        let plan = config
            .resolve_master_plan(Path::new("sub/compare.toml"))
            .unwrap();
        let suite = plan.suites.first().unwrap();
        assert_eq!(suite.tolerance_table.as_deref(), Some("/abs/../tol.toml"));
        assert_eq!(
            suite.variants.first().unwrap().output_file,
            "/abs/../out.h5"
        );
    }
}
