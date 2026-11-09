# Evidence for `docs/reviews/20261109-cell-timeout.md`

Every file here was produced by a command in that note, on the host it names (Apple M5 Max,
macOS 26.7.2, rustc 1.96.0, the registry arm). Nothing here is a campaign measurement.

| file | command |
|---|---|
| `cargo-test.log` | `cargo test --locked -- --test-threads=1` |
| `reference-math-probe.txt` | `cargo test --release --locked reference_math_cost -- --ignored --nocapture`, three rounds, then the medians |
| `cut-run.log`, `cut-run.csv` | `fbench run --threads 1 --family svd_thin --dtype f64 --rows faer-fork --n 512,1024 --reps 1 --prime-ms 0 --cell-timeout 1 --csv cut-run.csv` |
| `kill-before.log` | the same run, `--n 512,4096 --cell-timeout 0`, built at `ef4bf4e`, `SIGKILL` after 15 s: no CSV was written at all, so there is none here |
| `kill-after.log`, `kill-after.csv` | the same run and kill on the current head: four finished rows on disk, 12 heartbeat lines (2 builds, 5 starts, 4 dones); the last start is the `4096` cell, whose parent was `SIGKILL`ed, so it has no done by construction. The child exited via the parent watchdog; `ps` after 2 s showed it gone |
| `long-cell.log`, `long-cell.csv` | `svd_thin f64 --n 4096 --batch 2 --cell-timeout 3600`: four cells, `timed_ms` 33.8–34.2 s and `check_ms` 153–155 s each, all `ok`; total 12m31s; both child pids gone from `ps` afterwards |

| `docs/reviews/20261110-cell-timeout-invariant.md` | the terminal-row table; `fbench/tests/child_cells.rs` runs it with `fbench/tests/fake_child.sh` (8 rows) |
| `red-*.log` | one test (or the named pair) with the fix removed; the fix is back in the tree. `red-3b` is the verification-side `prepare` poll, `red-4`/`red-4b` the two error-classification polls, `red-1b` the deadline recomputed at run start, `red-2b` the high-level `solve` mid-polls, `red-3c` a real CSV write failure, `red-1c` the per-variant construction charge, `red-2c` the `qr`/`householder` compound polls, `red-3d` the overflow backstop, `red-2d` both the child and parent deadlines removed (the kill is the guarantee), `red-1d` the old "cut every remaining variant" behavior, `red-2e` the variant count taken after construction (1), `red-3e` a second start on the kill path, `red-4e` the old `getppid()`-after-start watchdog, `red-1e` a cooperative timeout result not respawned (the next variant marked failed), `red-2f` the final child wait without a kill (hang after the last result), `red-3f` done status always `ok`, `red-3g` a missing done, `red-W` an extra poll site the waiting-point table does not list, `red-W2` an extra row in the transition table (the row-count assert fails) |

`kill-before.csv` is absent on purpose: the point of the comparison is that the old binary left no
file behind.
