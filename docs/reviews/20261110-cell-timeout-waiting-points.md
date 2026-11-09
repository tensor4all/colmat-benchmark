# Waiting points under a cell's clock

Every place in the harness that can wait — a polled loop, a library call that may take minutes, a
prepare/check stage, I/O, or the parent's wait on a cell child — and how the cell's limit is seen
there. The list is checked mechanically by `waiting_points_are_enumerated_and_polled` in
`fbench/src/main.rs`: the test reads this table and each source file with `include_str!`, and asserts
that the set of poll sites in the sources is exactly the set of `if expired()` /
`if timing::expired()` rows here. Adding, moving or removing a poll without updating this table fails
the test, and so does adding a compound library call without a poll before it
(`compound_library_calls_are_guarded`).

"Polled" below means the cooperative clock is read at that line. `expired()` is cheap (one
thread-local `Cell<Option<Instant>>` read); a library call already running cannot be interrupted by
it, which is why the poll is placed *before* each call. **The guarantee is not the poll**: each build
runs in a child process (`--cell-spec`) and the parent kills it at the cell's deadline, so a wait
this list has not discovered still stops. The cooperative polls are an early-exit optimisation.

## Polled waits (64 sites)

| site | function | what it waits for / guards |
|---|---|---|
| `fbench/src/main.rs:1874` | `run_cell` | verification stage boundary or returned error |
| `fbench/src/main.rs:1876` | `run_cell` | verification stage boundary or returned error |
| `fbench/src/main.rs:1882` | `run_cell` | verification stage boundary or returned error |
| `fbench/src/main.rs:1884` | `run_cell` | verification stage boundary or returned error |
| `fbench/src/main.rs:1890` | `run_cell` | verification stage boundary or returned error |
| `fbench/src/main.rs:1892` | `run_cell` | verification stage boundary or returned error |
| `fbench/src/backend.rs:366` | `gemm_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:457` | `qr_factor_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:479` | `qr_factor_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:630` | `householder_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:641` | `householder_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:660` | `householder_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:674` | `householder_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:842` | `lu_in_place_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:900` | `cholesky_in_place_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:984` | `solve_in_place_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1000` | `solve_in_place_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1107` | `full_piv_lu_in_place_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1195` | `svd_in_place_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1292` | `eigh_in_place_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1378` | `qr_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1388` | `qr_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1457` | `lu_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1514` | `full_piv_lu_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1570` | `cholesky_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1624` | `solve_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1630` | `solve_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1689` | `triangular_solve_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1750` | `svd_values_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1834` | `svd_thin_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:1894` | `eigh_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2053` | `qr_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2077` | `qr_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2152` | `lu_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2223` | `full_piv_lu_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2315` | `cholesky_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2406` | `solve_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2412` | `solve_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2483` | `triangular_solve_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2549` | `svd_values_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2628` | `svd_thin_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/backend.rs:2714` | `eigh_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/lapack.rs:941` | `gemm_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/lapack.rs:1009` | `qr_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/lapack.rs:1111` | `lu_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/lapack.rs:1204` | `cholesky_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/lapack.rs:1271` | `solve_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/lapack.rs:1362` | `svd_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/lapack.rs:1491` | `eigh_op` | library call before a batch member, or between a member's two calls / in its prepare or check |
| `fbench/src/shared.rs:332` | `mul` | reference-math loop (`mul`, `mul_adj_b`, `transpose`, gram inner products) |
| `fbench/src/shared.rs:353` | `mul` | reference-math loop (`mul`, `mul_adj_b`, `transpose`, gram inner products) |
| `fbench/src/shared.rs:382` | `mul_adj_b` | reference-math loop (`mul`, `mul_adj_b`, `transpose`, gram inner products) |
| `fbench/src/shared.rs:403` | `mul_adj_b` | reference-math loop (`mul`, `mul_adj_b`, `transpose`, gram inner products) |
| `fbench/src/shared.rs:447` | `transpose` | reference-math loop (`mul`, `mul_adj_b`, `transpose`, gram inner products) |
| `fbench/src/shared.rs:463` | `gram_residual_columns` | reference-math loop (`mul`, `mul_adj_b`, `transpose`, gram inner products) |
| `fbench/src/shared.rs:508` | `gram_adj_residual_rows` | reference-math loop (`mul`, `mul_adj_b`, `transpose`, gram inner products) |
| `fbench/src/timing.rs:96` | `stage` | priming, prepared stage, timed repetition, check, or a stage error |
| `fbench/src/timing.rs:134` | `measure` | priming, prepared stage, timed repetition, check, or a stage error |
| `fbench/src/timing.rs:138` | `measure` | priming, prepared stage, timed repetition, check, or a stage error |
| `fbench/src/timing.rs:144` | `measure` | priming, prepared stage, timed repetition, check, or a stage error |
| `fbench/src/timing.rs:151` | `measure` | priming, prepared stage, timed repetition, check, or a stage error |
| `fbench/src/timing.rs:162` | `measure` | priming, prepared stage, timed repetition, check, or a stage error |
| `fbench/src/timing.rs:169` | `measure` | priming, prepared stage, timed repetition, check, or a stage error |
| `fbench/src/timing.rs:175` | `measure` | priming, prepared stage, timed repetition, check, or a stage error |

## Waits outside the cooperative clock

| site | what it waits for | how the limit is enforced |
|---|---|---|
| `fbench/src/main.rs` `run_build_in_child` (`Command::spawn`) | starting the cell child | the variant's deadline starts before the spawn |
| `fbench/src/main.rs` `run_build_in_child` (`recv_timeout`) | the child's `COUNT`/`RESULT` lines | on timeout the parent sends `SIGKILL` |
| `fbench/src/main.rs` `run_build_in_child` (`kill`, `wait`) | the child's exit / reap | this is the hard stop; `wait` reaps, so `ps` shows no child |
| `fbench/src/main.rs` child reader thread (`BufReader::lines`) | the child's stdout | no deadline of its own: the pipe closes when the child is killed |
| `fbench/src/main.rs` `Csv::push` | a row write + flush to the CSV | between cells; a failed write sets `failed` and the run exits 2 |
| `fbench/src/main.rs` `heartbeat` | one `eprintln!` to stderr | between/around cells; the parent owns the start/done pair |
| `fbench/src/vendor.rs` `identity` | reading the linked vendor's identity at startup | before any cell |
| `fbench/src/main.rs` `configure_threads` | building the process's Rayon pool (parent and child) | before any cell |

## Not measured here

The path-linked campaign build (`scripts/build_for_faer_rev.sh`) and the vendor arms
(`lapack-openblas`, `lapack-mkl`) are not exercised by this table or by the registry-arm tests: they
need the pinned checkouts and a vendor library, and CI's `harness` job builds the registry arm only.
Nothing here claims a timeout, a cell cost or a number for those arms.

Poll sites per file (also asserted by the test, from this table):
`main.rs=6`, `backend.rs=36`, `lapack.rs=7`, `shared.rs=7`, `timing.rs=8` (total 64).
