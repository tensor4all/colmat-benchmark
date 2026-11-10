# Evidence for `docs/reviews/20261109-cell-timeout.md`

Every file here was produced by a command in that note, on the host it names (Apple M5 Max,
macOS 26.7.2, rustc 1.96.0, the registry arm). Nothing here is a campaign measurement.

**Which head each file belongs to.** The acceptance and profile runs below were recorded at
**`42a90be`**, the code head at the time; the commits that added them (`e4ef8ef`, `c9b5a60`) are
documentation only, and the later review fixes (parent protocol handling and the child's stdout
reader) do not touch the timed path, the reference math or the CSV columns. Those files do not name
a head themselves, so it is said per file below: they are **not** measurements of the review-fix
head. `cargo-test.log` is the exception -- it was re-run at the head in its own first line.

| file | command |
|---|---|
| `cargo-test.log` | (head `8755546`, recorded in the file's first line) `cargo test --all --locked`: 39 unit passed (1 ignored) + 32 child-process tests. Replaces the `42a90be` version of this file, which had 15 child-process tests |
| `reference-math-probe.txt` | (head `42a90be`) `cargo test --release --locked reference_math_cost -- --ignored --nocapture`, three rounds, then the medians |
| `cut-run.log`, `cut-run.csv` | (head `42a90be`) `fbench run --threads 1 --family svd_thin --dtype f64 --rows faer-fork --n 512,1024 --reps 1 --prime-ms 0 --cell-timeout 1 --csv cut-run.csv` |
| `kill-before.log` | the same run, `--n 512,4096 --cell-timeout 0`, built at `ef4bf4e`, `SIGKILL` after 15 s: no CSV was written at all, so there is none here |
| `kill-after.log`, `kill-after.csv` | (head `42a90be`) the same run and kill: four finished rows on disk, 5 starts / 4 dones (3 builds); the last start is the `4096` cell, whose parent was `SIGKILL`ed, so it has no done by construction. The log ends with the actual check: parent pid 41149 SIGKILLed at unix 1794252745; `ps -p 41153 -o pid=,ppid=,stat=` returned rc=1 (empty) 2 s later, via the parent-pid watchdog |
| `long-cell.log`, `long-cell.csv` | (head `42a90be`) `svd_thin f64 --n 4096 --batch 2 --cell-timeout 3600`: four cells, `timed_ms` 33.7–34.0 s and `check_ms` 151.5–157.7 s each, all `ok`; total 12m33.0s. The log ends with the check itself: for pids 41329 and 44547, `ps -p <pid> -o pid=,ppid=,stat=` returned rc=1 (empty) at unix 1794253506 |

| `acceptance-5min.log/csv` | (head `42a90be`) owner-ruling ①: `svd_thin f64 --n 4096 --batch 3 --reps 2 --cell-timeout 3600`; 22m45.3s, four ~5.7-min cells, child pids 13664/2223 gone (`ps` rc=1) |
| `acceptance-garbage.log/csv` | (head `42a90be`) owner-ruling ②: a fake child returns `garbage`/`NaN` for the C order only; run exit 1 and exactly one failed row (C), F `ok` |
| `acceptance-final-crash.log/csv` | (head `42a90be`) owner-ruling ③: fake child reports its last result then exits 3; run exit 1, both rows failed |
| `docs/reviews/20261110-cell-timeout-invariant.md` | the terminal-row table; `fbench/tests/child_cells.rs` runs it with `fbench/tests/fake_child.sh` (9 rows) |
| `red-*.log` | one test (or the named pair) with the fix removed; the fix is back in the tree. `red-3b` is the verification-side `prepare` poll, `red-4`/`red-4b` the two error-classification polls, `red-1b` the deadline recomputed at run start, `red-2b` the high-level `solve` mid-polls, `red-3c` a real CSV write failure, `red-1c` the per-variant construction charge, `red-2c` the `qr`/`householder` compound polls, `red-3d` the overflow backstop, `red-2d` both the child and parent deadlines removed (the kill is the guarantee), `red-1d` the old "cut every remaining variant" behavior, `red-2e` the variant count taken after construction (1), `red-3e` a second start on the kill path, `red-4e` the old `getppid()`-after-start watchdog, `red-1e` a cooperative timeout result not respawned (the next variant marked failed), `red-2f` the final child wait without a kill (hang after the last result), `red-3f` done status always `ok`, `red-3g` a missing done, `red-1f` done lines for the first build id removed (the key-set assert), `red-1g` the same mutation with the old test (it passed: the gap), `red-2h` the old kill test on a fast cell (it failed: the host dependence), `red-3h` the fake `hang` made cooperative (no hard kill), `red-1i` a skipped family printed as MISMATCH, `red-2i` no parent deadline (the test's outer bound fires), `red-1j` a failed spawn propagated with no row/done, `red-2k` a clock that never expires (the unit test's 5 s bound fails), `red-2l` the same with the bound removed (the test hangs), `red-1k` a crash after a result treated as the cooperative cut, `red-2m` a malformed child result returned early with no row, `red-1l` a crash after the final result settled as ok, `red-2n` garbage/NaN numbers accepted, `red-2o` a short sample set accepted, `red-1m` a SKIP crash settled as skipped, `red-2p` header-only CSV with the new row assert, `red-2q` the same with the old test (it passed), `red-1n` a COUNT flood starving the deadline (the test's 10 s bound fires), `red-1o` a late SKIP finishing variant 1 twice, `red-W` an extra poll site the waiting-point table does not list, `red-W2` an extra row in the transition table (the row-count assert fails) |

`kill-before.csv` is absent on purpose: the point of the comparison is that the old binary left no
file behind.
