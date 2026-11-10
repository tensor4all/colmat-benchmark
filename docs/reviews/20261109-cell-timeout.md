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
* Each build runs in a **child process** (`--cell-spec`, hidden): the parent spawns the child,
  reads one `RESULT` per variant, and **kills the child (`SIGKILL`) when that variant's deadline
  passes**. The killed variant is the **only** one cut: the parent respawns a child with `from=`
  for the remaining variants, which then run normally. This is the guarantee: it does not depend on
  the harness finding every library call.
* The variant count is fixed by the parent **before construction** (`variant_count`), so a build
  killed while it is still building leaves a terminal row for every variant: the building one is
  cut, the rest are respawned and run.
* The child is given the parent's pid (`parent_pid=`), checks `getppid()` against it **at startup**
  and on a 500 ms watchdog, and exits immediately if it is not that parent. An external `SIGKILL`
  of the parent therefore leaves no orphan; a `getppid()` read after the parent already died (which
  returns 1) is never mistaken for a live parent.
* One `CELL start` and one terminal row per variant; the kill path never re-emits a start. The only
  exception to "a child exit is a failure" is the **cooperative construction cut**: a `timeout`
  result followed by a **successful** exit. A crash after an `ok` result, or any exit with no
  result, closes the pending variant as `failed`. The **last** variant's row and a **SKIP** row are
  held until the child's exit is known, so a crash after the final result, or a SKIP followed by a
  non-zero exit, is failed too. A SKIP is only a build-level skip **before any result**; a late SKIP
  is a protocol error for the pending variant and the remaining variants are respawned. A malformed `RESULT`
  (unknown status, unparseable/NaN/infinite/negative number, or an `ok` without finite
  timed/check times and exactly `--reps` samples) or an unexpected variant index takes the same
  failed-cell path and also reaps the child.
* The deadline is checked **before** `recv_timeout`: a queued `COUNT` message would otherwise be
  returned ahead of the timeout forever, so a child that floods `COUNT` would never be killed.
* Every wait on a child is bounded: one `try_wait`, then `SIGKILL`, then a 5 s `try_wait` loop;
  there is no blocking `wait()`. A child that cannot be killed inside the bound is **abandoned and
  logged** (`reap timed out`), and its cell is recorded as failed/cut -- so the run does not stall,
  but such a child can outlive the run when SIGKILL itself cannot stop it (uninterruptible I/O).
  The invariant and its transition table are in
  `docs/reviews/20261110-cell-timeout-invariant.md`.
* The recorder's interface does not change: `--locked`/`--no-verify`/`--publish` and the `--csv`
  path stay where they were. The CSV is still written by the parent, at the same path, and only
  the hidden `--cell-spec` argument exists between parent and child.
* The cooperative clock (`timing::DEADLINE`, thread-local, armed on the executing thread) stays as
  an **early-exit optimisation** inside the child. It can stop a reference loop sooner, but nothing
  is promised to depend on it.
* A cut cell is `status = timeout`, note `timeout <limit>`, with no partial time; the loop moves on
  to the next cell. In `verify` a cut prints `TIMEOUT <cell>` rather than `MISMATCH <cell>`.

Review item 1 of #58: the expensive reference calculations that build an `Op` — `cholesky`/`solve`
build their input with the cubic `shared::mul_adj_b`, the LAPACK `gemm` builds its whole reference
product — used to run *before* the clock was armed, so no limit applied to them. Both the cooperative
clock and the parent deadline now start **before construction** (the child arms before `ops_for`,
and the parent starts variant 0's deadline before the spawn). Each variant `build_ops` produced
gets its own deadline from its own start, less the construction time (`variant_deadline`,
`main.rs`): every variant pays the construction, and no variant spends the budget of the ones after
it. If construction crossed the limit, every op it produced is cut without running: those ops stand
on reference math that gave up part-way (a Cholesky input of `(n+1)I` instead of
`M M^H + (n+1)I` factorizes and checks fine, so nothing about the row would say so).

The high-level solve routes are two library calls in one member too: `a.partial_piv_lu()` then
`solve_in_place`, and `partial_lu_of` then `plu_solve` (`backend.rs:1621`, `:2403`); so are the
high-level `qr` (`a.qr()` then `compute_thin_Q`, `backend.rs:1382`) and `householder_apply`'s check
(`qr_in_place` then the block apply, `backend.rs:665`). Each now polls at its entry.

Review item 2: a batch `Op` is many library calls. `timing::measure` polled once per repetition, so
after the limit passed every remaining batch member still ran. The cooperative polls are still at
every harness-owned batch loop (36 sites in `fbench/src/backend.rs`, 7 in `fbench/src/lapack.rs`)
and at every compound boundary (`solve`, `qr`, `householder_apply`'s prepare and check), which lets
the child exit early. The authoritative stop is now the parent's kill: even a library call or a
loop that no poll covers cannot outlive the variant's deadline. Every wait is listed in
`docs/reviews/20261110-cell-timeout-waiting-points.md`.

Review item 3: `prepare` is outside the *timed* region but inside the cell, and it is not free
(`householder_apply` factorizes there). `timing::measure` now polls immediately after every
`prepare` — priming, each timed repetition, and the restore after the check — and the verification
path polls after its `prepare` too. A call that would start after the limit is not started.

One more hole, from the same review: a stage that returned `Err` *after* the limit had passed was
classified as `failed`. `timing::stage` and the verification path now check the clock before
classifying a returned error, so it is a `timeout` row (`timing.rs:78`).

A limit the platform clock cannot represent no longer disarms the cell: `deadline_at` returns an
already-expired deadline (an explicit cut), and `parse` refuses such a `--cell-timeout` before the
run starts. Returning `None` here used to mean "no limit" to every caller.

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
| `red-1c-per-cell-deadline.log` | the construction time is no longer charged per variant | `two_variants_do_not_share_one_deadline` FAILED |
| `red-2c-compound-entry.log` | the high-level `qr` / `householder_apply` check polls deleted | both tests FAILED |
| `red-3d-overflow.log` | an unrepresentable limit returns `None` again | `an_unrepresentable_limit_cuts_the_cell` FAILED |
| `red-2d-child-kill.log` | the parent waits for the child without a deadline (no kill) | `a_killed_cell_leaves_a_row_and_the_next_cell_runs` FAILED |
| `red-1d-kill-remaining.log` | a kill cuts every remaining variant (old) | the transition table FAILED (a cut variant is followed by `ok`, not by cut rows) |
| `red-2e-construction-count.log` | the variant count wrongly fixed at 1 (compiling mutation) | the transition table FAILED: one start per variant is `1/1`, so variants 2/3 have no rows |
| `red-3e-double-start.log` | the spawn path emits a second start | the transition table FAILED (one start per variant) |
| `red-4e-parent-pid.log` | the child records `getppid()` after it starts instead of the passed pid | `the_child_exits_when_the_given_parent_is_not_its_parent` FAILED |
| `red-1e-coop-cut-failed.log` | a cooperative timeout result is not respawned; the next variant is marked failed | the transition table FAILED (`cut first` expects `timeout,ok,ok`) |
| `red-2f-final-wait.log` | the last child is waited for without a kill | the hang-after-last-result test FAILED (`None` after its 10 s deadline) |
| `red-3f-done-status.log` | every done says `status=ok` | the transition table FAILED on the done status |
| `red-3g-missing-done.log` | the done line for the second variant is omitted | the transition table FAILED on the done count |
| `red-1f-done-keys.log` | all done lines for build id 1 are dropped | the transition table FAILED on the start/done key sets |
| `red-1g-done-keys-old-test.log` | the same mutation with the key-set assert removed (the old test) | it **passed**, which is the gap the assert closes |
| `red-2h-fast-cell.log` | the old first-variant kill test with a cell that finishes under 1 s | it FAILED: the old test depended on the host being slow |
| `red-3h-no-hard-kill.log` | the fake `hang` made to report a cooperative timeout instead of sleeping | the hard-kill test FAILED: no `CELL kill`, no pid |
| `red-1i-verify-skip.log` | a skipped family is printed as `MISMATCH` | `verify_reports_an_unsupported_family_as_skip_not_mismatch` FAILED |
| `red-2i-no-parent-deadline.log` | the parent does not arm a deadline (the test's 10 s bound fires) | the hard-kill test FAILED instead of hanging |
| `red-1j-spawn-failure.log` | a spawn failure propagates with `?` | the spawn-failure test FAILED: exit 2, no row/done |
| `red-2k-clock-bound.log` | `expired()` always false, with the test's 5 s bound | the clock test FAILED in 5.01 s (`cut` false) |
| `red-2l-clock-unbounded.log` | the same with the bound removed | `timeout 12` exits 124: the test hangs |
| `red-1k-crash-after-result.log` | the cooperative exception accepts any exit after a result | the crash-after-result test FAILED (variant 2 became `ok`) |
| `red-2m-protocol-error.log` | a malformed result returns early instead of closing the cell | the malformed-result test FAILED (exit 2, no row) |
| `red-1l-final-crash.log` | the last row is settled without looking at the child's exit | the crash-after-last-result test FAILED (ok rows, exit 0) |
| `red-2n-bad-numbers.log` | permissive numeric parsing and no result validation | the garbage/NaN test FAILED (ok row) |
| `red-2o-short-samples.log` | no `--reps` sample-count check | the short-sample test FAILED (ok row) |
| `red-1m-skip-crash.log` | the SKIP row is written before the reap and its status ignored | the skip-then-crash test FAILED (skipped rows, exit 0) |
| `red-2p-header-only.log` | protocol errors write no row, with the new row-count assert | the garbage/NaN and short-sample tests FAILED (header-only CSV) |
| `red-2q-header-only-old-test.log` | the same header-only CSV with the old tests (no row-count assert) | they **passed**, which is the gap the assert closes |
| `red-1n-count-flood.log` | no clock check before `recv_timeout`; a `COUNT` flood keeps the queue full | the count-flood test FAILED (the test's 10 s bound fires) |
| `red-1o-late-skip.log` | a SKIP after a result finishes variant 1 again and returns | the late-SKIP test FAILED (no variant 3) |

Green: `cargo test --locked` → `test result: ok. 37 passed; 0 failed; 1 ignored` (unit) and
`test result: ok. 15 passed` (`fbench/tests/child_cells.rs`: first-variant hard kill with the fake
child, finished run, the 9-row transition table, the wrong-parent pid check, the
hang-after-last-result reap, the spawn-failure row, a crash after a result, a malformed child
result, a crash after the last result, garbage/NaN numbers, a short sample set, and a SKIP followed
by a crash, a COUNT flood, and a late SKIP).

End to end, with the parent enforcing the limit:
* `long-cell` (`--n 4096 --batch 2`, limit 3600 s), regenerated at head `42a90be`: four cells,
  each 33.7-34.0 s timed plus 151.5-157.7 s check, all `ok`; total 12m33.0s. The log itself records
  the check (`ps -p 41329`, `ps -p 44547`, both rc=1, empty, at unix 1794253506).
* `cut-run` (`--n 512,1024 --cell-timeout 1`): 512 runs `ok`, 1024 is cut; 8 starts / 8 dones.
* The first-variant hard kill is deterministic: the fake child sleeps before reporting anything, so
  the parent's `SIGKILL` (not a slow real cell) is what the test requires. `hang first` is a table
  row (the table is now 9 rows) and the test asserts the kill line's pid is gone from `ps`.
  The kill line records `pid`, `variant`, `at_ms` and `reaped_at`, so the `ps` check has its time
  and pid in the log (the integration test also runs `ps -p` and asserts it is gone).
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
* `heartbeat` (`main.rs`) writes to **stderr**, never to stdout, which the recorder parses for
  `MISMATCH` and `all comparisons within tolerance`. The parent prints one `CELL build` (with the
  child pid), one `CELL start` per variant (with `id=` and `variant=<k>/<n>`), and exactly one
  `CELL done` per variant with the same id and variant. A deadline-killed variant is closed by the
  parent with `CELL done ... status=timeout cut=timeout`, so every start has a done; `cut-run.log`
  has 8 starts and 8 dones. If the *parent* is killed externally there is no done for the variant
  that was running -- the parent is the writer -- and `kill-after.log` shows that shape (5 starts,
  4 dones); its child exits via the parent-pid watchdog.
* A cut cell is a row with the existing `status = timeout` value; no column was added for the cut.
  This PR *did* append `timed_ms` and `check_ms`, so rows are 20 fields instead of 18: readers that
  require exactly 18 fields reject them. I did not run the recorder or any CSV consumer against
  18-field and 20-field fixtures, so **no compatibility with those readers is claimed here**; the
  recorder's own check reads the process exit and the `MISMATCH` word, not the CSV width.

Evidence (`docs/reviews/20261109-cell-timeout-logs/`):

* `cut-run.log`, `cut-run.csv`: `run --n 512,1024 --cell-timeout 1` — the four `n = 512` cells are
  `ok` rows with both times, the four `n = 1024` cells are `timeout 1` rows with empty times, exit
  code 1, 8 rows in the file.
* `kill-before.log` (empty, no CSV at all) against `kill-after.log`: the same invocation
  (`--n 512,4096`, no limit) killed with `SIGKILL` 15 s in. Before this change the killed process
  left **no CSV and no log line**; now it leaves **4 finished rows on disk** (one per `n = 512`
  variant) and 5 starts / 4 dones (3 builds); the last line is `CELL start … 4096`, the cell that
  was still running when the parent died. The log records the check itself: parent pid 41149
  SIGKILLed at unix 1794252745, then `ps -p 41153 -o pid=,ppid=,stat=` returned rc=1 (empty output)
  2 s later.

## 5巡目裁定 (owner, 2026-11-09): scope of this PR

This PR closes: an externally enforced per-cell deadline (child + parent kill); exactly one terminal
row per variant (`ok` / `cut=timeout` / `error`); broken output is never recorded as success
(numeric parse/NaN/protocol violations become `error` rows and fail the run); and the final
variant's `reap_child` exit status is not discarded.

**Not closed here (follow-up issue; see the PR body for the explicit list):** the late-`SKIP`
double-finalisation, the deadline-vs-queued-message priority, and heartbeat/CSV column
completeness. The first two are already implemented and tested as defensive guards of the
one-terminal-row invariant, but they are not acceptance gates for this PR; the third is unchanged.

Acceptance measured for the ruling (`docs/reviews/20261109-cell-timeout-logs/`):

1. `acceptance-5min`: limit 3600 s, `svd_thin --n 4096 --batch 3 --reps 2`: four cells of
   ~5.7 min each completed (`ok`), total 22m45.3s; child pids 13664/2223 gone from `ps` (rc=1).
2. `acceptance-garbage`: a fake child returns `garbage`/`NaN` for one order: run exit 1,
   **exactly one failed row** (the other order is `ok`).
3. `acceptance-final-crash`: a fake child reports its last result and exits 3: run exit 1, rows
   failed (not exit 0).
4. `cargo test --all --locked`: 39 unit passed (1 ignored) + 15 child-process tests passed (29 now;
   the review fixes added 14).

### 6巡目: the review findings, fixed inside that scope

1. **A broken `RESULT` index must not kill the parent.** `18446744073709551615` parses as a
   `usize`, so the "wrong variant" error message computed `index + 1` and aborted the parent in
   debug builds: exit 101, no CSV rows, no reason. The index is now range-checked where it is
   parsed, once, and an out-of-range index is a protocol error like any other.
2. **Output after the last result was read and discarded.** The receive loop ended as soon as
   `next == n`, so a child that reported its last result and then printed anything else was
   recorded `ok`. The parent now reads the child out to the end (bounded by `REAP_GRACE`), reaps
   it, and reads again what the child said while it was exiting; any extra line fails the held
   last row -- the same root as acceptance item 3. Two gaps this opened in the first attempt were
   found by the independent review and closed with their own red runs: a line that lands during
   the reap, and an undecodable byte, which `BufReader::lines()` had turned into a clean end of
   output. Decoding that byte leniently then opened two more, also fixed and red first: a broken
   line *before* a good result was skipped as an unknown line, and the reader's own line-ending
   trim repaired `1.0\r\r` into a valid sample. The reader now asks `wait_line` for whole lines,
   reports a line that is not text as broken output (a `failed` row for the running variant, an
   extra line after the last one), and strips exactly what `lines()` strips -- one `\n` and then
   one `\r`, so a final line with no newline keeps its carriage return.
3. **A line the protocol does not define was dropped, so acceptance depended on its position.** The
   receive loop's fall-through arm discarded anything that was not `COUNT`/`RESULT`/`SKIP`, while
   the same line after the last result failed the row: a child printing `garbage` and then a
   correct `RESULT` was recorded `ok`. An unknown line now goes down the same failed-cell path as
   every other protocol violation (`protocol_error`, one helper the three violation sites share),
   so a line is judged by what it is, not by when it arrived.
4. **A `SKIP` was a licence to say anything else.** The skip branch settled its row as soon as it
   saw the `SKIP`, so `SKIP` followed by `garbage` and a clean exit was recorded `skipped` -- the
   same gap as item 2, on the other settlement path. The branch now drains the child around its
   reap too and writes a `failed` row when anything followed the skip.
5. **A `RESULT` line with the wrong number of fields was accepted.** `parse_result` read the fields
   it knew (`get(1)`..`get(6)`) and ignored whatever else was on the line, so
   `RESULT 0 ok "" 1 2 1 garbage` -- one field too many -- was an `ok` measurement: the stray text
   travelled on the same line, past both the number checks and the extra-output check. The line's
   shape is now checked where it is parsed (seven fields, the shape the child writes in one
   `println!`), and a field more or fewer is broken output, hence a `failed` row.
6. **A finite sample whose derived value overflowed was written as `inf` on an `ok` row.** Each
   field was checked where it is parsed, but the row also carries derived numbers: `per_item_us`
   scales the summary by 1000 (so `1e308` ms became `inf`), and the median averages two samples (so
   two `1e308` samples did too). The CSV then held `inf` under `status = ok`. `finish` now runs one
   check on every row it is about to write -- `total_ms`, `per_item_us`, `timed_ms`, `check_ms` must
   be finite -- and an `ok` row that fails it becomes a `failed` row, which fails the run.
7. **A `SKIP` line had the same hole the `RESULT` line had.** Only its second field was read and the
   rest was dropped, so `SKIP fake skip garbage` (three fields, exit 0) became a `skipped` row: the
   field-count rule was enforced on one line kind and not the other. `SKIP` is now checked for its
   two fields before it is settled, in both directions, and a malformed one closes the running
   variant as `failed` like any other broken output.
8. **A final child the parent had to kill was still a successful measurement.** The held last row was
   settled `failed` only for a non-zero exit the parent did *not* cause; `Reaped::Killed` (the parent
   SIGKILLed a child that reported its last result and then hung) and `Reaped::TimedOut` (the child
   could not be reaped at all) both kept the row as reported, so such a run exited 0 with
   `status=ok`. Acceptance item 3 is "the final cell died, so the run fails", and a cell the harness
   had to cut is no more a completed measurement than one that crashed: both now write a `failed`
   row with the reason (`the final cell did not complete: the child did not exit after reporting its
   last result (signal: 9 (SIGKILL))`), log `CELL final-reap`, and fail the run. The note does not
   claim the child died -- a `TimedOut` reap means the parent stopped waiting, not that the process
   is gone. The existing test that asserted exit 0 for this
   (`a_child_that_hangs_after_its_last_result_does_not_stall_the_run`) now asserts exit 1 and the
   `ok, ok, failed` rows, keeping its point that the parent itself never stalls.
9. **`COUNT` was the last unchecked line.** Its arm read nothing at all: the field count and the
   payload were never looked at, so `COUNT garbage extra` followed by a good `RESULT` was a
   successful run -- the same fail-open the `RESULT`, `SKIP` and unknown-line paths had. `COUNT` is
   now two fields with an integer payload, checked before it is used, and a line that says anything
   else closes the running variant as `failed`. A repeated *well-formed* `COUNT` stays accepted: a
   flooding child must still be caught by the deadline (`count_flood`), not by the protocol, and that
   queued-vs-deadline question stays where the ruling put it.

Red → green, both against a fake child: `huge_index` (old: `panicked at main.rs:1352: attempt to
add with overflow`, exit 101; new: exit 1, rows `failed,ok`) and `garbage_after_last` (old: exit 0
with three `ok` rows; new: exit 1, rows `ok,ok,failed`), plus `late_garbage` and
`bad_utf8_after_last` for the two gaps above (both were exit 0 with all-`ok` rows; now exit 1 with
the last row `failed`), and `bad_utf8_before_result` / `extra_cr_sample` / `bare_cr_at_eof` for the
reader regressions (all three were exit 0; now exit 1 with a `failed` row), and
`garbage_before_result` for item 3 (old: `garbage` then a valid `RESULT` -> exit 0 and two `ok`
rows; new: exit 1 with `failed,ok,ok`), and `garbage_after_skip` for item 4 (old: exit 0 with a
`skipped` row; new: exit 1 with a `failed` row), `extra_field` / `short_result` for item 5 (old:
exit 0 with `ok` rows; new: exit 1 with `failed` rows), and `huge_sample` / `huge_sample_pair` for
item 6 (old: exit 0 with `per_item_us=inf` on an `ok` row; new: exit 1, `failed` rows whose note names
the number it refused), `skip_extra_field` / `skip_short` for item 7 (old: exit 0 with `skipped`
rows; new: exit 1 with `failed` rows), and `final-hang` for item 8 (old: exit 0 with `ok` rows; new:
exit 1 with `failed` rows), and `bad_count` / `bad_count_payload` for item 9 (old: exit 0 with
`ok` rows; new: exit 1 with `failed` rows). `cargo test --locked` is now 39 unit passed (1 ignored)
+ 29 child-process tests. None of these extends what the PR closes: the three follow-up
items above are unchanged.

Two further review notes were checked and left alone, with the reason recorded here:

* **A `COUNT` line is still position-dependent** (accepted any number of times before a result,
  rejected after the last one). Its shape and payload are checked now, but its value only records how
  long construction took and goes no further; rejecting a repeated `COUNT` would change
  `count_flood`'s expected `timeout`-by-kill rows, which is the queued-message-vs-deadline behaviour
  this PR keeps out (it is in the "not closed" list above).
* **A build-level `SKIP` from a replacement child still re-finishes variant 1 and drops the rest.**
  That is **#62**, already listed above as not closed; the new arms respawn exactly the way the
  pre-existing protocol-error and exit paths do, so they inherit it rather than create it.

`review: gpt-6.1-sol PASS` (independent review of the fix for the two findings, four rounds; rounds
1-3 found two gaps in the trailing-output check and two regressions the first attempt introduced --
all fixed, each with the red run recorded above). The audit's round-2 review of that head returned
one finding (item 3, the position-dependent acceptance of an unknown line), fixed above; the audit
also asked that the long acceptance runs be tied to a head, which the evidence index now does.

**Which head the long runs belong to.** The acceptance and profile logs in
`docs/reviews/20261109-cell-timeout-logs/` were recorded at `42a90be`, the code head at the time;
the commits that added them (`e4ef8ef`, `c9b5a60`) are docs-only, and the later review fixes touch
the parent's protocol handling and the child's stdout reader, not the timed path, the reference math
or the CSV columns. Those runs have not been repeated at the review-fix head and nothing here claims
they were; the index says which head each file belongs to. `cargo-test.log` is the exception: it is
this head's run, and its first line says so.

## What this does not change

* The measurement: the timed *call*, priming, repetition count, statistic, thread budget, and the
  columns the campaign declares in `benchmarks/campaign.yaml` are unchanged. `timed_ms` and
  `check_ms` stay appended diagnostics, and a cut cell has neither. The timed loop now polls the
  clock once before each batch member: base (`3b58258`) vs head (`34ff512`) `gemm` at n=512, 1
  thread, 7 reps, median of three alternating rounds: C 4.608 vs 4.615 ms (+0.14%), F 4.654 vs
  4.594 ms (-1.28%) -- below noise at this size, not a general overhead bound.
* The verification *method*: whether `gram_residual` should sample columns rather than run the full
  product is a separate decision (a random-projection mode is a separate issue, not this PR).
* A library call in progress is not interruptible; the limit is enforced at poll points. Every
  polled wait is listed in `docs/reviews/20261110-cell-timeout-waiting-points.md`, and a test keeps
  that table equal to the source's poll sites.
* Nothing here measures the path-linked campaign build (`scripts/build_for_faer_rev.sh`) or either
  vendor arm; CI's `harness` job builds the registry arm only. No timeout or number is claimed for
  those.

## Checks

```
cargo build --locked          # Finished, 0 errors
cargo test --all --locked     # 39 unit passed (1 ignored) + 29 child-process tests passed
```

`cargo-test.log` holds the test run at `e1aa47d` (the head is in the file's first line, and its
command is `cargo test --all --locked`) -- 39 unit passed, 1 ignored, 29 child-process tests. CI's `harness` job (#60) runs `cargo check -p fbench --locked`
and `cargo test -p fbench --locked` against the committed lock (no `--bins`, so
`fbench/tests/child_cells.rs` runs too), so the test result is a local run; the merge gate is the
`harness` job being green on this PR head.

The timing probe is `#[ignore]`d so it never slows the suite:
`cargo test --release --locked reference_math_cost -- --ignored --nocapture`.

Not checked here: the path-linked measured checkout (`scripts/build_for_faer_rev.sh`) and the vendor
arms; the campaign's own gates own that, and nothing in this change touches the library calls.
