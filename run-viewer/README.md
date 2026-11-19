# Run viewer

From the repository root:

```sh
cargo run --manifest-path run-viewer/Cargo.toml -- runs/
```

The directory defaults to `runs/` in the current working directory. Runs are immediate, non-hidden directories, listed in reverse filename order. Enter opens a run and lists its regular `.jsonl` files alphabetically. Neither listing follows symlink entries or descends into subdirectories.

- **↑/↓ or k/j** select a run or file; **Enter** opens a run.
- **b** on the runs list toggles the highlighted run for comparison. Mark up to four runs, then press **Enter** on the run whose metrics you want to inspect. Its plots are cyan; the legend identifies the other runs by color. Opening a marked run never plots it twice.
- **s** in a run toggles **Raw** (default) and **Trend**. Trend averages samples in horizontal intervals sized to the plot width. All curves on a plot use the same intervals. Widen the terminal for finer detail, or switch back to Raw to inspect individual spikes.
- **a** pauses or resumes automatic refresh (on by default, every 5 seconds).
- **r** immediately rereads the current listing and all comparison previews. **Esc/Backspace** goes back; **q/Ctrl+C** quits.

The viewer never writes to run files. It shows all top-level numeric fields from the selected file together in a grid of line charts. Resize the terminal to give the plots more space. The x-axis uses `Step` (case-insensitive) when every nonempty parsed object has a numeric step; otherwise it uses the source line number. Missing metric values, non-object records, and incomplete JSON lines are skipped. The viewer reads the entire selected log and each matching comparison log line by line on refresh. Very large logs may take longer to reload; pause live refresh for manual inspection. Overlays require the same filename, the same metric name, and a numeric step axis in the current file and the comparison file. A missing match skips only that curve; other matching runs remain visible. 

The run header shows the Slurm launcher's status from `metadata/meta.json`: **Reported running**, **Completed**, or **Failed**. This is not a scheduler query: an interrupted launcher can leave a stale running status. Without a recognized status, **Recent activity** means `metrics.jsonl` was modified within five minutes; otherwise the status is **Unknown**, not assumed finished.

For SSH use, build on the cluster (or for its Linux architecture): `cargo build --release --manifest-path run-viewer/Cargo.toml`. Run `run-viewer/target/release/run-viewer runs/` there. A macOS binary won't work on Linux. No Python or Apptainer environment is needed if the directory is readable.
