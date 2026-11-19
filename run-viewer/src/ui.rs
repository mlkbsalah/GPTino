use std::{borrow::Cow, path::Path};

use ratatui::{
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, List, ListState, Paragraph, Wrap},
    Frame,
};

use crate::app::App;

fn single_line(value: &str) -> String {
    value.replace(char::is_control, "�")
}

fn name(path: &Path) -> String {
    single_line(
        &path
            .file_name()
            .unwrap_or(path.as_os_str())
            .to_string_lossy(),
    )
}

const MARK_COLORS: [Color; 4] = [Color::Yellow, Color::Magenta, Color::Green, Color::LightRed];

fn path_list(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    rows: impl Iterator<Item = Line<'static>>,
    state: &mut ListState,
) {
    let list = List::new(rows)
        .block(Block::bordered().title(title))
        .highlight_symbol("> ")
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    frame.render_stateful_widget(list, area, state);
}

fn tick(value: f64) -> String {
    if value != 0.0 && !(0.001..1e6).contains(&value.abs()) {
        format!("{value:.2e}")
    } else {
        format!("{value:.3}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}

fn bounds(values: impl Iterator<Item = f64>) -> [f64; 2] {
    let (mut low, mut high) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    if low == high {
        let pad = (low.abs() * 0.05).max(0.5);
        low = (low - pad).max(f64::MIN);
        high = (high + pad).min(f64::MAX);
    }
    [low, high]
}

fn axis(title: &str, [low, high]: [f64; 2]) -> Axis<'_> {
    Axis::default()
        .title(title)
        .bounds([low, high])
        .labels([tick(low), tick(high)])
}

// Shared x intervals keep differently sampled runs aligned in comparison mode.
fn trend(points: &[(f64, f64)], bounds: [f64; 2], columns: usize) -> Vec<(f64, f64)> {
    let [left, right] = bounds;
    let mut bins = vec![(0.0, 0.0, 0usize); columns];
    for &(x, y) in points {
        let column = ((x - left) / (right - left) * columns as f64) as usize;
        let (mean_x, mean_y, count) = &mut bins[column.min(columns - 1)];
        *count += 1;
        *mean_x += (x - *mean_x) / *count as f64;
        *mean_y += (y - *mean_y) / *count as f64;
    }
    bins.into_iter()
        .filter(|b| b.2 != 0)
        .map(|(x, y, _)| (x, y))
        .collect()
}

#[test]
fn trend_collapses_crowded_steps_but_keeps_isolated_points() {
    let points = [(0., 2.), (1., 8.), (9., 7.)];
    assert_eq!(trend(&points, [0., 9.], 3), [(0.5, 5.), (9., 7.)]);
}

fn dataset(points: &[(f64, f64)], color: Color) -> Dataset<'_> {
    Dataset::default()
        .marker(Marker::Braille)
        .graph_type(if points.len() == 1 {
            GraphType::Scatter
        } else {
            GraphType::Line
        })
        .style(Style::default().fg(color))
        .data(points)
}

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < 40 || area.height < 8 {
        frame.render_widget(Paragraph::new("Terminal too small"), area);
        return;
    }

    let hints = if app.run.is_some() {
        if app.trend {
            "Trend [s: raw]  ↑↓/jk select  Esc back  r refresh  q quit"
        } else {
            "Raw [s: trend]  ↑↓/jk select  Esc back  r refresh  q quit"
        }
    } else {
        "↑↓/jk select  Enter open  b mark/unmark  r refresh  q/Ctrl+C quit"
    };
    let mode = if app.auto_refresh {
        "Live 5s"
    } else {
        "Paused"
    };
    let hints = format!("{mode} [a]  {hints}");
    let footer = Paragraph::new(hints.as_str()).wrap(Wrap { trim: true });
    let mut legend = Vec::new();
    if let Some(run) = &app.run {
        legend.push(Line::styled(
            format!("[{}] Current: {}", app.status, name(run)),
            Color::Cyan,
        ));
        for (i, comparison) in app
            .comparisons
            .iter()
            .enumerate()
            .filter(|(_, c)| &c.path != run)
        {
            let matched = comparison.preview.series.len();
            let note = if matched == 0 {
                " (no matching metrics)".into()
            } else if matched < app.preview.series.len() {
                format!(" ({matched}/{} matched)", app.preview.series.len())
            } else {
                String::new()
            };
            legend.push(Line::styled(
                format!("[{}]{note} {}", i + 1, name(&comparison.path)),
                MARK_COLORS[i],
            ));
        }
    }
    let [header, body, error, keys] = Layout::vertical([
        Constraint::Length(legend.len().max(1) as u16),
        Constraint::Min(3),
        Constraint::Length(u16::from(app.error.is_some())),
        Constraint::Length(if area.width < 55 && app.run.is_some() {
            3
        } else if usize::from(area.width) < hints.chars().count() {
            2
        } else {
            1
        }),
    ])
    .areas(area);
    frame.render_widget(footer, keys);
    if let Some(message) = &app.error {
        frame.render_widget(
            Paragraph::new(format!("Error: {}", single_line(message)))
                .style(Style::default().fg(Color::Red)),
            error,
        );
    }

    let Some(run) = &app.run else {
        frame.render_widget(
            Paragraph::new(format!(
                "Marked: {}/4  Runs: {}",
                app.comparisons.len(),
                single_line(&app.root.to_string_lossy())
            )),
            header,
        );
        if app.runs.is_empty() {
            frame.render_widget(
                Paragraph::new("No runs found").block(Block::bordered()),
                body,
            );
        } else {
            let rows = app.runs.iter().map(|path| {
                let mark = app.comparisons.iter().position(|c| &c.path == path);
                let indicator = mark.map_or_else(
                    || Span::raw("    "),
                    |i| Span::styled(format!("[{}] ", i + 1), MARK_COLORS[i]),
                );
                Line::from(vec![indicator, Span::raw(name(path))])
            });
            path_list(frame, body, "Runs", rows, &mut app.run_list);
        }
        return;
    };

    frame.render_widget(Paragraph::new(legend), header);
    let [files, preview] =
        Layout::horizontal([Constraint::Percentage(30), Constraint::Fill(1)]).areas(body);
    path_list(
        frame,
        files,
        "Files",
        app.files.iter().map(|p| Line::raw(name(p))),
        &mut app.file_list,
    );

    let series = &app.preview.series;
    if series.is_empty() {
        let message = if app.files.is_empty() {
            "No metric files found"
        } else {
            "No numeric data"
        };
        frame.render_widget(
            Paragraph::new(message).block(Block::bordered().title("Preview")),
            preview,
        );
        return;
    }

    let max_cols = usize::from(preview.width / 24).max(1);
    let cols = series
        .len()
        .div_ceil(usize::from(preview.height / 12).max(1))
        .min(max_cols);
    let rows = series.len().div_ceil(cols);
    if rows * 6 > usize::from(preview.height) {
        frame.render_widget(
            Paragraph::new("Enlarge terminal to show all plots").wrap(Wrap { trim: true }),
            preview,
        );
        return;
    }
    let bands = Layout::vertical(vec![Constraint::Fill(1); rows]).split(preview);
    for (metrics, &band) in series.chunks(cols).zip(bands.iter()) {
        let cells = Layout::horizontal(vec![Constraint::Fill(1); metrics.len()]).split(band);
        for ((metric, points), &cell) in metrics.iter().zip(cells.iter()) {
            let mut lines = vec![(Color::Cyan, Cow::Borrowed(points.as_slice()))];
            for (i, comparison) in app
                .comparisons
                .iter()
                .enumerate()
                .filter(|(_, c)| &c.path != run)
            {
                if let Some((_, points)) = comparison
                    .preview
                    .series
                    .iter()
                    .find(|(key, _)| key == metric)
                {
                    lines.push((MARK_COLORS[i], Cow::Borrowed(points.as_slice())));
                }
            }
            let x_bounds = bounds(lines.iter().flat_map(|(_, ps)| ps.iter().map(|p| p.0)));
            if app.trend {
                let columns = usize::from(cell.width.saturating_sub(10).max(1));
                for (_, points) in &mut lines {
                    *points = Cow::Owned(trend(points, x_bounds, columns));
                }
            }
            let y_bounds = bounds(lines.iter().flat_map(|(_, ps)| ps.iter().map(|p| p.1)));
            let datasets = lines
                .iter()
                .map(|(color, ps)| dataset(ps, *color))
                .collect();
            let title = single_line(metric);
            let chart = Chart::new(datasets)
                .block(Block::bordered().title(title))
                .x_axis(axis(app.preview.x_label, x_bounds))
                .y_axis(axis("Value", y_bounds));
            frame.render_widget(chart, cell);
        }
    }
}
