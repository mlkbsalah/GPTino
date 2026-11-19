//! Discovery lists only immediate entries and never follows symlink entries.

use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

pub fn runs(path: &Path) -> io::Result<Vec<PathBuf>> {
    let mut runs = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_name().to_string_lossy().starts_with('.') && entry.file_type()?.is_dir() {
            runs.push(entry.path());
        }
    }
    runs.sort_unstable_by(|a, b| b.cmp(a));
    Ok(runs)
}

pub fn files(path: &Path) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() == Some("jsonl".as_ref()) && entry.file_type()?.is_file() {
            files.push(path);
        }
    }
    files.sort_unstable();
    Ok(files)
}

#[derive(Default, Debug)]
pub struct Preview {
    pub series: Vec<(String, Vec<(f64, f64)>)>,
    pub x_label: &'static str,
}

pub fn preview(path: &Path) -> io::Result<Preview> {
    let reader = BufReader::new(fs::File::open(path)?);
    let mut series = BTreeMap::new();
    let mut saw_object = false;
    let mut all_have_step = true;
    for (i, line) in reader.split(b'\n').enumerate() {
        let Ok(Value::Object(fields)) = serde_json::from_slice(&line?) else {
            continue;
        };
        // The trainer flushes an empty object after its last step.
        if fields.is_empty() {
            continue;
        }
        let step = fields.iter().find_map(|(key, value)| {
            key.eq_ignore_ascii_case("step")
                .then(|| value.as_f64())
                .flatten()
        });
        saw_object = true;
        all_have_step &= step.is_some();
        let line = (i + 1) as f64;
        for (key, value) in fields {
            if !key.eq_ignore_ascii_case("step") {
                if let Some(y) = value.as_f64() {
                    series
                        .entry(key)
                        .or_insert_with(Vec::new)
                        .push((line, step, y));
                }
            }
        }
    }
    let use_step = saw_object && all_have_step;
    Ok(Preview {
        series: series
            .into_iter()
            .map(|(key, points)| {
                let points = points
                    .into_iter()
                    .map(|(line, step, y)| (if use_step { step.unwrap() } else { line }, y))
                    .collect();
                (key, points)
            })
            .collect(),
        x_label: if use_step { "Step" } else { "Line" },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn discovers_only_immediate_matching_entries_in_order() -> io::Result<()> {
        let dir = tempdir()?;
        let root = dir.path();
        let run = root.join("b");
        for name in ["a", ".slurm", ".other", "b", "b/nested.jsonl"] {
            fs::create_dir(root.join(name))?;
        }
        fs::write(root.join("ordinary"), b"")?;
        for name in [
            "task_eval.jsonl",
            "config.yaml",
            "nested.jsonl/x.jsonl",
            "metrics.jsonl",
        ] {
            fs::write(run.join(name), b"")?;
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&run, root.join("linked"))?;
            std::os::unix::fs::symlink(run.join("metrics.jsonl"), run.join("link.jsonl"))?;
        }
        assert_eq!(runs(root)?, [run.clone(), root.join("a")]);
        assert_eq!(
            files(&run)?,
            [run.join("metrics.jsonl"), run.join("task_eval.jsonl")]
        );
        Ok(())
    }

    #[test]
    fn distinguishes_empty_from_missing() -> io::Result<()> {
        let dir = tempdir()?;
        assert!(runs(dir.path())?.is_empty());
        assert!(files(dir.path())?.is_empty());
        let missing = dir.path().join("missing");
        assert!(runs(&missing).is_err());
        assert!(files(&missing).is_err());
        assert!(preview(&missing).is_err());
        Ok(())
    }

    #[test]
    fn groups_numeric_fields_by_key_using_step() -> io::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("metrics.jsonl");
        fs::write(
            &path,
            b"{\"sTEp\":8,\"z\":2,\"a\":1}\n42\nbad\n{\"Step\":3,\"a\":2.5,\"nested\":{\"x\":1}}\n{}",
        )?;
        let data = preview(&path)?;
        assert_eq!(data.x_label, "Step");
        assert_eq!(
            data.series,
            [
                ("a".into(), vec![(8., 1.), (3., 2.5)]),
                ("z".into(), vec![(8., 2.)]),
            ]
        );
        Ok(())
    }

    #[test]
    fn falls_back_to_source_lines_for_every_point() -> io::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("metrics.jsonl");
        for last in [r#"{"note":"no step"}"#, r#"{"Step":"9"}"#] {
            fs::write(
                &path,
                format!("\n{{\"Step\":8,\"x\":2}}\nbad\n{{\"step\":7,\"x\":3}}\n{last}"),
            )?;
            let data = preview(&path)?;
            assert_eq!(data.x_label, "Line");
            assert_eq!(data.series, [("x".into(), vec![(2., 2.), (4., 3.)])]);
        }
        Ok(())
    }

    #[test]
    fn reads_beyond_former_cap_and_picks_up_append() -> io::Result<()> {
        use std::io::Write;
        let dir = tempdir()?;
        let path = dir.path().join("metrics.jsonl");
        let padding = " ".repeat(256 * 1024);
        fs::write(
            &path,
            format!("{padding}\n{{\"step\":9,\"x\":1}}\n{{\"x\":"),
        )?;
        let data = preview(&path)?;
        assert_eq!(data.x_label, "Step");
        assert_eq!(data.series, [("x".into(), vec![(9., 1.)])]);

        fs::OpenOptions::new()
            .append(true)
            .open(&path)?
            .write_all(b"2}")?;
        let data = preview(&path)?;
        assert_eq!(data.x_label, "Line");
        assert_eq!(data.series, [("x".into(), vec![(2., 1.), (3., 2.)])]);
        Ok(())
    }
}
