use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::KeyCode;
use ratatui::widgets::ListState;

use crate::files;

pub struct Comparison {
    pub path: PathBuf,
    pub preview: files::Preview,
}

#[derive(Default)]
pub struct App {
    pub root: PathBuf,
    pub runs: Vec<PathBuf>,
    pub run_list: ListState,
    pub run: Option<PathBuf>,
    pub files: Vec<PathBuf>,
    pub file_list: ListState,
    pub preview: files::Preview,
    pub comparisons: Vec<Comparison>,
    pub trend: bool,
    pub status: &'static str,
    pub auto_refresh: bool,

    pub error: Option<String>,
}

fn run_status(run: &Path) -> &'static str {
    let meta = fs::read(run.join("metadata/meta.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok());
    match meta.as_ref().and_then(|m| m["status"].as_str()) {
        Some("running") => return "Reported running",
        Some("completed") => return "Completed",
        Some("failed") => return "Failed",
        _ => (),
    }
    let age = fs::metadata(run.join("metrics.jsonl"))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok());
    if age.is_some_and(|age| age <= Duration::from_secs(300)) {
        "Recent activity"
    } else {
        "Unknown"
    }
}

impl App {
    pub fn new(root: PathBuf, runs: Vec<PathBuf>) -> Self {
        let run_list = ListState::default().with_selected((!runs.is_empty()).then_some(0));
        Self {
            root,
            runs,
            run_list,
            status: "Unknown",
            auto_refresh: true,
            ..Self::default()
        }
    }

    fn clear_previews(&mut self) {
        self.preview = files::Preview::default();
        for comparison in &mut self.comparisons {
            comparison.preview = files::Preview::default();
        }
    }

    fn load_preview(&mut self) {
        self.clear_previews();
        self.error = None;
        if let Some(i) = self.file_list.selected() {
            let path = &self.files[i];
            match files::preview(path) {
                Ok(preview) => self.preview = preview,
                Err(e) => self.error = Some(format!("{}: {e}", path.display())),
            }
            if self.preview.x_label != "Step" || self.preview.series.is_empty() {
                return;
            }
            for comparison in &mut self.comparisons {
                let base = &comparison.path;
                if Some(base) == self.run.as_ref() {
                    continue;
                }
                let result = files::files(base).and_then(|paths| {
                    paths
                        .iter()
                        .find(|p| p.file_name() == path.file_name())
                        .map(|p| files::preview(p))
                        .transpose()
                });
                match result {
                    Ok(Some(mut preview)) if preview.x_label == "Step" => {
                        preview.series.retain(|(key, _)| {
                            self.preview
                                .series
                                .binary_search_by(|(k, _)| k.cmp(key))
                                .is_ok()
                        });
                        if !preview.series.is_empty() {
                            comparison.preview = preview;
                        }
                    }
                    Err(e) => self.error = Some(format!("Comparison {}: {e}", base.display())),
                    _ => (),
                }
            }
        }
    }

    pub fn refresh(&mut self) {
        let (dir, listing, paths, list) = match &self.run {
            Some(run) => {
                self.status = run_status(run);
                (run, files::files(run), &mut self.files, &mut self.file_list)
            }
            None => (
                &self.root,
                files::runs(&self.root),
                &mut self.runs,
                &mut self.run_list,
            ),
        };
        let next = match listing {
            Ok(next) => next,
            Err(e) => {
                self.error = Some(format!("{}: {e}", dir.display()));
                if self.run.is_some() {
                    self.clear_previews();
                }
                return;
            }
        };
        let old_index = list.selected().unwrap_or(0);
        let old_path = paths.get(old_index);
        let same = old_path.and_then(|p| next.iter().position(|entry| entry == p));

        list.select(same.or_else(|| next.len().checked_sub(1).map(|last| old_index.min(last))));
        *paths = next;
        self.error = None;
        if self.run.is_some() {
            self.load_preview();
        } else {
            self.comparisons.retain(|c| self.runs.contains(&c.path));
        }
    }

    pub fn key(&mut self, key: KeyCode) {
        use KeyCode::*;
        match key {
            Up | Char('k') | Down | Char('j') => {
                let detail = self.run.is_some();
                let (list, len) = if detail {
                    (&mut self.file_list, self.files.len())
                } else {
                    (&mut self.run_list, self.runs.len())
                };
                let Some(i) = list.selected() else {
                    return;
                };
                let next = if matches!(key, Up | Char('k')) {
                    i.saturating_sub(1)
                } else {
                    (i + 1).min(len - 1)
                };
                if next != i {
                    list.select(Some(next));
                    if detail {
                        self.load_preview();
                    }
                }
            }
            Enter if self.run.is_none() => {
                if let Some(i) = self.run_list.selected() {
                    self.run = Some(self.runs[i].clone());
                    self.files.clear();
                    self.file_list = ListState::default();

                    self.refresh();
                }
            }
            Esc | Backspace if self.run.is_some() => {
                self.run = None;
                self.error = None;
            }
            Char('b') if self.run.is_none() => {
                if let Some(i) = self.run_list.selected() {
                    let path = &self.runs[i];
                    self.error = None;
                    if let Some(index) = self.comparisons.iter().position(|c| &c.path == path) {
                        self.comparisons.remove(index);
                    } else if self.comparisons.len() < 4 {
                        self.comparisons.push(Comparison {
                            path: path.clone(),
                            preview: files::Preview::default(),
                        });
                    } else {
                        self.error = Some("Unmark a run before comparing more than four.".into());
                    }
                }
            }
            Char('s') if self.run.is_some() => self.trend = !self.trend,
            Char('a') => self.auto_refresh = !self.auto_refresh,
            Char('r') => self.refresh(),

            _ => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use KeyCode::*;

    #[test]
    fn detail_status_refreshes_even_if_listing_fails() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let run = dir.path();
        fs::write(run.join("metrics.jsonl"), "")?;
        let mut app = App::new(run.into(), vec![run.into()]);
        app.key(Enter);
        assert_eq!(app.status, "Recent activity");
        fs::create_dir(run.join("metadata"))?;
        fs::write(run.join("metadata/meta.json"), r#"{"status":"completed"}"#)?;
        app.refresh();
        assert_eq!(app.status, "Completed");
        dir.close()?;
        app.refresh();
        assert_eq!(app.status, "Unknown");
        assert!(app.error.is_some());
        Ok(())
    }

    #[test]
    fn compares_matching_step_metrics() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let base = dir.path().join("base");
        let run = dir.path().join("run");
        fs::create_dir(&base)?;
        fs::create_dir(&run)?;
        let base_file = base.join("metrics.jsonl");
        fs::write(&base_file, r#"{"Step":1,"loss":4,"old":2}"#)?;
        fs::write(run.join("metrics.jsonl"), r#"{"step":3,"loss":2,"new":1}"#)?;
        let other = dir.path().join("other");
        fs::create_dir(&other)?;
        fs::write(other.join("metrics.jsonl"), r#"{"step":2,"new":5}"#)?;
        let mut app = App::new(dir.path().into(), vec![base.clone(), other.clone(), run]);
        app.key(Char('b'));
        app.key(Char('b'));
        assert!(app.comparisons.is_empty());
        app.key(Char('b'));
        app.key(Down);
        app.key(Char('b'));
        app.key(Down);
        app.key(Char('b'));
        app.key(Enter);
        assert_eq!(
            app.comparisons[0].preview.series,
            [("loss".into(), vec![(1., 4.)])]
        );
        assert_eq!(
            app.comparisons[1].preview.series,
            [("new".into(), vec![(2., 5.)])]
        );
        assert!(app.comparisons[2].preview.series.is_empty());
        fs::write(base_file, r#"{"loss":3}"#)?;
        app.key(Char('r'));
        assert!(app.comparisons[0].preview.series.is_empty());
        assert_eq!(
            app.comparisons[1].preview.series,
            [("new".into(), vec![(2., 5.)])]
        );
        assert_eq!(app.preview.series.len(), 2);
        app.key(Esc);
        fs::remove_dir_all(base)?;
        app.refresh();
        assert_eq!(app.comparisons.len(), 2);
        assert_eq!(app.comparisons[0].path, other);
        Ok(())
    }

    #[test]
    fn browse_and_refresh_a_changing_run() -> std::io::Result<()> {
        let dir = tempfile::tempdir()?;
        let mut app = App::new(dir.path().into(), vec![]);
        app.key(Down);
        app.key(Enter);
        assert!(app.run.is_none());
        let run = dir.path().join("run");
        fs::create_dir(&run)?;
        app.key(Char('r'));
        app.key(Enter);
        app.key(Down);
        assert_eq!(app.run.as_ref(), Some(&run));
        let metrics = run.join("metrics.jsonl");
        fs::write(&metrics, r#"{"a":1,"b":2}"#)?;
        app.key(Char('r'));

        fs::write(run.join("a.jsonl"), r#"{"loss":4}"#)?;
        fs::write(&metrics, r#"{"a":2,"b":3}"#)?;
        app.key(Char('r'));
        assert_eq!(app.file_list.selected(), Some(1));
        assert_eq!(app.preview.series.len(), 2);
        assert_eq!(app.preview.series[1].1[0], (1.0, 3.0));
        app.key(Up);
        assert_eq!(app.preview.series[0].0, "loss");
        app.key(Down);
        fs::remove_file(metrics)?;
        app.key(Char('r'));
        assert_eq!(app.preview.series[0].0, "loss");
        app.key(Esc);
        assert!(app.run.is_none());
        Ok(())
    }
}
