# A per-cell time limit, its reach, and the reference math's constant factor

Issue #57. One weekend campaign ran 27.6 h on a single `n = 4096` cell — the serial
`shared::gram_residual` inside `svd_thin`'s check — and finished with nothing to show for it:
nothing cut the cell off, and the run's CSV was written once at the end, so the cells that *had*
finished left no row either. "Still measuring" and "stuck" looked the same from outside.

This note covers the PR that follows #60 (`3b58258`). Evidence logs are in
`docs/reviews/20261109-cell-timeout-logs/` and every number below names the file it comes from.

## 1. The limit, and what it covers

* `--cell-timeout SECS` on `run` and `verify`, `FBENCH_CELL_TIMEOUT` when the flag is absent,
  `DEFAULT_CELL_TIMEOUT_S = 3600`, `0` disables. A cell is one `Op`, one CSV row.
* The clock is thread-local (`timing::DEADLINE`, `timing::CUT`) and is armed on the thread that
  runs the cell: `run_cell`/`measure_armed` arm it, so a row that runs on a team member or a worker
  is cut there too.
* A cut cell is `status = timeout`, note `timeout <limit>`, with no partial time; the loop moves on
  to the next cell. In `verify` a cut prints `TIMEOUT <cell>` rather than `MISMATCH <cell>`.

Review item 1 of #58: the expensive reference calculations that build an `Op`
(`fbench/src/main.rs:440` `build_ops`, called at `:698`) — `cholesky`/`solve` build their input
with the cubic `shared::mul_adj_b`, and the LAPACK `gemm` builds its whole reference product — used
to run *before* the clock was armed, so no limit applied to them. The cell's **absolute
deadline is fixed before construction**, construction runs under it, and each op `build_ops`
produced gets its own deadline from that same stamp, less the construction time
(`variant_deadline`, `main.rs:1014`; `Budget`, `main.rs:1006`): every variant pays the construction,
and no variant spends the budget of the ones after it. If construction crossed
the limit, **every** op it produced is cut without running, including the second variant of `lu`,
`cholesky`, `solve`, `svd_*` and `eigh`: those ops stand on reference math that gave up part-way
(a Cholesky input of `(n+1)I` instead of `M M^H + (n+1)I` factorizes and checks fine, so nothing
about the row would say so).

The high-level solve routes are two library calls in one member too: `a.partial_piv_lu()` then
`solve_in_place`, and `partial_lu_of` then `plu_solve` (`backend.rs:1621`, `:2403`).

Review item 2: a batch `Op` is many library calls. `timing::measure` polled once per repetition, so
after the limit passed every remaining batch member still ran. Every harness-owned batch loop now
polls before each call and returns `timing::cut_error()` (34 poll sites in `fbench/src/backend.rs`,
7 in `fbench/src/lapack.rs`), which the executor records as `timeout`, not as a library failure.
That covers the loops that are *not* timed calls as well: `householder_apply`'s `prepare`
factorizes every member, `qr_factor`'s check forms `Q` with the library, and `solve`/`qr` are one
member with two library calls (factor, then solve / form `Q`), polled between the two.

Review item 3: `prepare` is outside the *timed* region but inside the cell, and it is not free
(`householder_apply` factorizes there). `timing::measure` now polls immediately after every
`prepare` — priming, each timed repetition, and the restore after the check — and the verification
path polls after its `prepare` too. A call that would start after the limit is not started.

One more hole, from the same review: a stage that returned `Err` *after* the limit had passed was
classified as `failed`. `timing::stage` and the verification path now check the clock before
classifying a returned error, so it is a `timeout` row (`timing.rs:78`).

Red, per item, with the fix removed (`docs/reviews/20261109-cell-timeout-logs/`):

| log | fix removed | result |
|---|---|---|
| `red-1-construction.log` | clock not armed before `build_ops` | `the_cell_clock_covers_the_reference_construction` FAILED |
| `red-2-batch.log` | the 34 batch polls deleted | `a_batch_op_does_not_start_a_call_after_the_limit` FAILED (`Ok(())`) |
| `red-3-prepare.log` | the post-`prepare` polls deleted | two `measure` prepare tests FAILED |
| `red-3b-verification-prepare.log` | the verification `prepare` poll deleted | `verification_does_not_run_after_a_prepare_past_the_limit` FAILED |
| `red-4-stage-error.log` | `stage`'s poll deleted | `a_stage_error_after_the_limit_is_a_cut` FAILED |
| `red-4b-stage-error-main.log` | the verification error-first poll deleted | `a_stage_that_fails_after_the_limit_is_a_cut_not_a_failure` FAILED |
| `red-1b-deadline-construction.log` | the deadline recomputed at the start of the run | `the_cell_deadline_includes_the_reference_construction` FAILED (`ok`, construction not charged) |
| `red-2b-highlevel-solve.log` | the two high-level `solve` mid-polls deleted | both solve tests FAILED (`Ok(())`) |
| `red-3c-csv-write.log` | `Csv::push` no longer records the write error | `a_csv_that_lost_a_row_is_a_failed_run` FAILED (real `Broken pipe (os error 32)`) |

Green: `cargo test --locked` → `test result: ok. 30 passed; 0 failed; 1 ignored`
(`cargo-test.log`). The ignored one is the timing probe below.

The clock is cooperative: a library call already in progress cannot be interrupted. What the
change does guarantee is that **no new call starts** after the limit, which is what the three items
were about.

## 2. The reference math's constant factor

`gram_residual` is `n^2 m` inner products; at `n = 4096` that is 6.9e10 of them, and it ran with
`shared::idx`'s layout branch inside the innermost loop, walking a strided column for one of the two
layouts. Three things changed, no mathematics:

* the layout decision moved outside every loop (a columns kernel and a rows kernel);
* `idx` is gone from the inner loops: the `p` walk is an addition, or a slice;
* the dot product accumulates `f64` real and imaginary parts, so a contiguous walk vectorizes.

Whichever layout makes the walk strided is transposed once into a scratch buffer
(`shared::transpose`), because a strided column costs more than the copy. `mul` and `mul_adj_b` got
the mirror of their existing column-major fast path: a streamed row update (the `idx`-hoisted
dot-product form measured no better than the old code, and at `n = 2048` it was 9% slower).

`residual_max` is untouched: a `NaN` or an `inf` in the input is still an infinite residual, never a
zero.

### Measured

`cargo test --release --locked reference_math_cost -- --ignored --nocapture`, three rounds, medians;
the old and the new implementation are timed **in the same process**, so both see the same host and
the same load. Raw rounds: `reference-math-probe.txt`.

| n | call | layout | old (ms) | new (ms) | old/new |
|---|---|---|---|---|---|
| 512 | `gram_residual` | C | 204.7 | 55.1 | 3.72 |
| 512 | `gram_adj_residual` | F | 200.7 | 55.9 | 3.59 |
| 512 | `mul` | C | 140.3 | 15.2 | 9.23 |
| 512 | `mul_adj_b` | C | 50.5 | 15.7 | 3.22 |
| 1024 | `gram_residual` | C | 1840.0 | 506.3 | 3.63 |
| 1024 | `gram_adj_residual` | F | 1861.9 | 513.6 | 3.63 |
| 1024 | `mul` | C | 1346.3 | 121.2 | 11.11 |
| 1024 | `mul_adj_b` | C | 483.9 | 129.2 | 3.75 |
| 2048 | `gram_residual` | C | 32732.0 | 4415.8 | 7.41 |
| 2048 | `gram_adj_residual` | F | 36105.8 | 4525.4 | 7.98 |
| 2048 | `mul` | C | 22453.0 | 1018.3 | 22.05 |
| 2048 | `mul_adj_b` | C | 4388.5 | 1102.2 | 3.98 |

The layout that was already contiguous is a small **regression**, not free: at `n = 512` the
contiguous gram paths are 5–8% slower in every one of the three rounds (`c gram_adj` 50.5 → 54.6 ms,
`f gram_residual` 50.9 → 55.1 ms), at `n = 1024` 1–6% slower, and at `n = 2048` the real-type
`gram_residual` medians are 4542.7 → 5301.2 ms (+17%), though that cell's three old-side rounds were
4330, 4543 and 5776 ms on a host under load, so the 2048 figure is the least certain one here. The
contiguous `mul`/`mul_adj_b` paths are unchanged within 1–2%. The trade is deliberate: the strided
path is 3.6–8x faster, the contiguous path loses 1–8% (except the noisy 2048 case), and a check that
calls both is still far ahead. A cell on the contiguous side only is therefore up to ~8% slower at
the sizes measured here; that is a real cost, stated rather than hidden.

`n = 4096` was **not** measured; nothing above is extrapolated to it.

Host and arm for every number in this section: Apple M5 Max (arm64, macOS 26.7.2), rustc 1.96.0,
single-threaded, the **registry arm** (`cargo build --locked --release`, `t4a-faer` from the
committed lock), not the path-linked measured checkout. No core pinning is available on darwin, so
these are harness-development timings, not campaign numbers.

### The equality test

`shared::tests::the_fast_reference_math_agrees_with_the_old_one` keeps the pre-change loops as the
executable specification (`old_mul`, `old_mul_adj_b`, `old_gram_residual`, `old_gram_adj_residual`,
verbatim) and compares: `f64` and `c64` within `1e-12 * max(1, |old|)` for several shapes and both
layouts, an orthonormal `q` at exactly 0, and — separately —
`the_fast_residuals_still_reject_nonfinite_input` for `NaN`/`inf`.

This is **not** a bitwise-identity claim. The gram loops now accumulate an `f64` pair, so an `f32`
residual differs from the old value by `f32`'s own eps; `the_fast_f32_residual_agrees_to_f32_precision`
pins that at `1e-5`. The real and complex `f64` paths happen to round identically, which the test
would not notice either way.

Red for this test (`red-A-arithmetic.log`, `red-A-transpose-shape.log`): flipping one sign in the
columns kernel, and passing the wrong shape to `transpose` in `mul_adj_b`, each fail it. The second
was a real bug this test caught while the change was being written.

## 3. The record is written as the run goes

The rows used to be buffered and written once at the end. After the 27.6 h stall there was no CSV
at all, so the run directory could not distinguish "not measured" from "hung".

* `Csv` (`main.rs:832`) writes the header when the file is opened and one row per cell as the cell
  finishes, flushed. A killed run leaves every row it earned. A write that fails sets `failed`, and
  the process exits 2 (`main.rs:802`): a report missing a cell must not look like success. The test
  that pins this writes to a socket whose peer is already closed, so the write really fails; it
  asserts the recorded failure, not just the exit-code arithmetic.
* `heartbeat` (`main.rs:876`) writes one line per cell to **stderr** (`CELL start …` /
  `CELL done … <status> total_ms=…`), never to stdout, which the recorder parses for `MISMATCH` and
  `all comparisons within tolerance`. `CELL start` is printed **before** the reference construction,
  so a construction stall is a cell that started and has not finished, not a silence.
* A cut cell is a row with the existing `status = timeout` value; no column was added for the cut,
  so a reader that predates it keeps working. The separate timing change in this PR does append
  `timed_ms` and `check_ms` to the CSV, so a reader that requires exactly 18 fields rejects the
  20-field rows; none of the repository readings does.

Evidence (`docs/reviews/20261109-cell-timeout-logs/`):

* `cut-run.log`, `cut-run.csv`: `run --n 512,1024 --cell-timeout 1` — the four `n = 512` cells are
  `ok` rows with both times, the four `n = 1024` cells are `timeout 1` rows with empty times, exit
  code 1, 8 rows in the file.
* `kill-before.log` (empty, no CSV at all) against `kill-after.log`: the same invocation
  (`--n 512,4096`, no limit) killed with `SIGKILL` 15 s in. Before this change the killed process
  left **no CSV and no log line**; now it leaves **4 finished rows on disk** (one per `n = 512`
  variant) and 7 heartbeat lines, the last of which is `CELL start … 4096` — the cell that was still
  running when the process died.

## What this does not change

* The measurement: the timed region, priming, repetition count, statistic, thread budget, and the
  columns the campaign declares in `benchmarks/campaign.yaml` are untouched. `timed_ms` and
  `check_ms` stay appended diagnostics, and a cut cell has neither.
* The verification *method*: whether `gram_residual` should sample columns rather than run the full
  product is a separate decision (a random-projection mode is a separate issue, not this PR).
* A library call in progress is not interruptible; the limit is enforced at poll points.

## Checks

```
cargo build --locked          # Finished, 0 errors
cargo test --locked           # test result: ok. 30 passed; 0 failed; 1 ignored
```

`cargo-test.log` holds the second. CI's `harness` job (#60) runs `cargo check -p fbench --locked`
and `cargo test -p fbench --bins --locked` against the committed lock, so the test result is a
local run; the merge gate is the `harness` job being green on this PR head.

The timing probe is `#[ignore]`d so it never slows the suite:
`cargo test --release --locked reference_math_cost -- --ignored --nocapture`.

Not checked here: the path-linked measured checkout (`scripts/build_for_faer_rev.sh`) and the vendor
arms; the campaign's own gates own that, and nothing in this change touches the library calls.
