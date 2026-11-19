mod app;
mod files;
mod ui;

use std::{env, io, path::PathBuf};

use app::App;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};

const USAGE: &str =
    "Usage: run-viewer [RUNS_DIR]\nPlot numeric metrics from run files. RUNS_DIR defaults to runs/.";

fn main() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let arg = args.next();
    if arg.as_deref().is_some_and(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    if args.next().is_some() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, USAGE));
    }
    let root = PathBuf::from(arg.unwrap_or_else(|| "runs".into()));
    let runs = files::runs(&root)
        .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", root.display())))?;
    let mut app = App::new(root, runs);
    let mut terminal = ratatui::init();
    let result = browse(&mut terminal, &mut app);
    ratatui::restore();
    result
}

fn browse(terminal: &mut ratatui::DefaultTerminal, app: &mut App) -> io::Result<()> {
    use std::time::{Duration, Instant};
    let interval = Duration::from_secs(5);
    let mut next = Instant::now() + interval;
    loop {
        if app.auto_refresh && Instant::now() >= next {
            app.refresh();
            next = Instant::now() + interval;
        }
        terminal.draw(|frame| ui::draw(frame, app))?;
        if app.auto_refresh && !event::poll(next.saturating_duration_since(Instant::now()))? {
            continue;
        }
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if key.code == KeyCode::Char('q')
                || (key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL))
            {
                return Ok(());
            }
            app.key(key.code);
        }
    }
}
