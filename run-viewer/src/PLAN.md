# Run Viewer — Minimal Implementation Plan

## Goal

Build a standalone Rust crate using Ratatui to browse experiment runs in a terminal, including over SSH on the cluster.

The first version is a read-only file browser: list runs, open a run, select a metric file, and inspect its raw contents. No curves, metric parsing, or changes to the Python training framework.

## Scope

### Included

- A single **Runs** view listing immediate, non-hidden subdirectories of a supplied runs directory.
- Open a run to list its immediate `.jsonl` files.
- Preview the selected file as scrollable plain text.
- Manual refresh with `r`.
- Helpful empty states and visible file-access errors.
- Keyboard navigation and reliable terminal cleanup.

### Explicitly excluded

- Charts, metric summaries, and run comparisons.
- JSON parsing or validation: partial JSON records remain visible as text.
- Automatic refresh, file watching, and incremental tailing.
- Slurm integration or inferred job status.
- Editing, deleting, uploading, or otherwise modifying run files.
- Databases, tracking APIs, async runtimes, or Python integration.

## Existing Data Contract

The current trainer writes these files directly inside a run directory:

```text
runs/
  .slurm/                  # Ignore hidden directories
  <run-name>/
    metrics.jsonl          # Training metrics
    task_eval.jsonl        # Optional downstream evaluation metrics
    config.yaml
    sampler.txt
    metadata/
    checkpoints/
```

Discover `.jsonl` files rather than hard-coding the two metric filenames. Do not recurse into subdirectories or open checkpoint files. Treat every non-hidden immediate directory as a potential run, even when it has no metric files yet.

## Proposed Interface

From the repository root:

```sh
cargo run --manifest-path run-viewer/Cargo.toml -- runs/
```

After building:

```sh
run-viewer/target/release/run-viewer runs/
```

Accept one optional positional directory, defaulting to `runs` relative to the current working directory. Support `--help`. A missing or unreadable root directory should produce a clear error before entering terminal mode.

### Runs view

- Header: `Runs` and the root directory.
- Scrollable list of run names, sorted reverse lexicographically so the existing timestamp-prefixed names appear newest first.
- Footer with key bindings.
- Enter opens the selected run.

### Run detail view

- Header: selected run name.
- Left pane: `.jsonl` filenames, sorted alphabetically.
- Right pane: raw contents of the selected file.
- Footer: navigation, scrolling, refresh, and back controls.
- Show `No metric files found` when the run has no JSONL files.

No general tab system is needed for one view. A Runs header and a run-detail screen are enough for this first version.

### Key Bindings

| Key | Action |
| --- | --- |
| Up / Down or k / j | Select a run or metric file |
| Enter | Open selected run |
| Esc / Backspace | Return from run details to the run list |
| PageUp / PageDown | Scroll the file preview vertically |
| Left / Right | Scroll the file preview horizontally |
| Home | Reset preview scroll position |
| r | Refresh the current directory listing and preview |
| q / Ctrl+C | Quit from either view |

Keep selection stable by path on refresh where possible. If an entry disappears, clamp the selection to a remaining entry or show an empty state. Changing the selected file resets preview scrolling.

## Crate Structure

Keep the crate independent of Python and of any repository-root Cargo workspace so it can be extracted later.

```text
run-viewer/
  Cargo.toml
  Cargo.lock
  .gitignore
  README.md
  src/
    PLAN.md
    main.rs       # Arguments, terminal lifecycle, event loop
    app.rs        # Screen state and navigation actions
    files.rs      # Run/file discovery and bounded text loading
    ui.rs         # Ratatui rendering
```

Commit `Cargo.lock` because this is an application. Add `target/` to the crate-local `.gitignore`.

### Dependencies

- `ratatui`: layout, lists, text preview, styling.
- `crossterm`: terminal events and input.
- Optionally `tempfile` as a dev dependency for filesystem tests.

Use `std` for filesystem access, argument handling, and errors. No need for Clap, Serde, Tokio, or a logging framework at this scope. Select mutually compatible stable Ratatui and Crossterm versions when implementing, and lock them.

## Implementation Steps

### 1. Scaffold the independent crate

- Add the manifest, local ignore file, and executable entry point.
- Implement directory argument handling and help output.
- Validate the root directory before starting the TUI.

### 2. Implement filesystem reading

Expose small, testable functions to:

- List non-hidden immediate run directories.
- List immediate regular `.jsonl` files inside a run.
- Load a bounded preview of a selected file.

For simplicity, do not follow symlink entries in discovery. Document this behavior.

Read at most the first **256 KiB** of a file, plus one byte to detect truncation. Show an explicit notice when only a prefix is displayed. This avoids loading huge training histories while keeping the implementation simple. Convert invalid UTF-8 lossily so one bad byte cannot crash the viewer.

Read only when selecting a file or refreshing, never during rendering. Do not attempt to parse JSON: an incomplete last line is expected while training writes.

### 3. Implement application state

Keep only the state needed for:

- Root directory and run list.
- Current screen: runs or run details.
- Selected run and file.
- Cached preview text, truncation flag, and scroll offsets.
- An optional user-facing error message.

Use Ratatui list state for selection and list scrolling. Bound preview scrolling to the available content and the renderer's supported range. Handle empty lists without indexing them.

### 4. Render the two screens

- Render the run list with a clear selected-row highlight.
- Render run details with a file-list pane and a plain-text preview pane.
- Include key hints and visible errors.
- Show a small-terminal message when the layout cannot fit usefully.
- Do not wrap long JSONL lines; allow horizontal scrolling instead.
- Display control characters safely rather than letting file content affect terminal behavior.

### 5. Add the event loop and cleanup

Use a synchronous event loop: draw, wait for an event, update state, repeat.

- Handle keyboard press events and resize events.
- Keep all directory/file reads outside rendering.
- Treat missing or unreadable run files as recoverable UI errors.
- Restore raw mode and the alternate screen on normal exit and propagated errors.
- Use the selected Ratatui version's supported panic-cleanup mechanism, or install an explicit cleanup hook if needed.

### 6. Document and validate

Add a README with:

- Build and launch commands.
- Directory and file-discovery rules.
- Key bindings.
- Preview size limit and manual-refresh behavior.
- Read-only guarantees.
- Cluster deployment notes: compile for the cluster's Linux architecture; a macOS binary will not run there. The viewer can run outside Apptainer if it can read the run directory.

## Tests

### Filesystem tests

- Ignore `.slurm` and other hidden directories.
- Exclude ordinary files from the run list.
- Verify deterministic run and file sorting.
- Discover `metrics.jsonl` and optional `task_eval.jsonl`.
- Ignore other extensions and nested directories.
- Handle empty directories and missing paths.
- Confirm symlink discovery behavior on supported platforms.
- Handle invalid UTF-8 and incomplete JSON text.
- Enforce the preview limit and correctly report truncation.

### State tests

- Navigation is safe with no runs or files.
- Selection stays within bounds.
- Enter and back switch screens correctly.
- Refresh preserves an existing selection by path.
- Removing the selected entry does not panic.
- Changing the selected file resets scrolling.

### Manual acceptance checks

- Open the existing repository's `runs/` directory.
- Select a run and inspect both metric files when present.
- Scroll long lines and a multi-page preview.
- Append a line externally and confirm `r` reloads the preview.
- Remove a selected file and confirm a recoverable state.
- Resize the terminal, then quit and confirm normal terminal behavior.
- Confirm that browsing does not modify any experiment files.

### Commands

Run from the repository root:

```sh
cargo fmt --manifest-path run-viewer/Cargo.toml --check
cargo test --manifest-path run-viewer/Cargo.toml
cargo clippy --manifest-path run-viewer/Cargo.toml --all-targets -- -D warnings
cargo build --manifest-path run-viewer/Cargo.toml --release
```

Rust and Cargo were not available in the inspected environment when this plan was written. Implementation validation requires an environment with the Rust toolchain installed.

## Definition of Done

A user can launch the standalone executable against a runs directory, select a run, inspect its metric files as text, refresh manually, and exit cleanly. Nothing in the Python training framework or experiment outputs is changed.
