# Evidence for `docs/reviews/20261109-cell-timeout.md`

Every file here was produced by a command in that note, on the host it names (Apple M5 Max,
macOS 26.7.2, rustc 1.96.0, the registry arm). Nothing here is a campaign measurement.

| file | command |
|---|---|
| `cargo-test.log` | `cargo test --locked -- --test-threads=1` |
| `reference-math-probe.txt` | `cargo test --release --locked reference_math_cost -- --ignored --nocapture`, three rounds, then the medians |
| `cut-run.log`, `cut-run.csv` | `fbench run --threads 1 --family svd_thin --dtype f64 --rows faer-fork --n 512,1024 --reps 1 --prime-ms 0 --cell-timeout 1 --csv cut-run.csv` |
| `kill-before.log` | the same run, `--n 512,4096 --cell-timeout 0`, built at `ef4bf4e`, `SIGKILL` after 15 s: no CSV was written at all, so there is none here |
| `kill-after.log`, `kill-after.csv` | the same run and kill on the current head: four finished rows on disk, seven heartbeat lines, the last `CELL start … 4096` |
| `red-*.log` | one test (or the named pair) with the fix removed; the fix is back in the tree. `red-3b` is the verification-side `prepare` poll, `red-4`/`red-4b` the two error-classification polls, `red-1b` the deadline recomputed at run start, `red-2b` the high-level `solve` mid-polls, `red-3c` a real CSV write failure |

`kill-before.csv` is absent on purpose: the point of the comparison is that the old binary left no
file behind.
