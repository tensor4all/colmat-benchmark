# `faer-kernels-ohtaka-small` on `ohtaka-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `92977df64b206b0f3a7120db69247ddd76bf80e3`
- toolkit commit: `34cf62b2504fa68857c5c1a33c125d22c5620452`
- measured binary SHA-256: `4561380d851943a642ed7b26c4b48c54303e6199c6de32f2dc9da5f4f9c8348d`
- hardware profile: `ohtaka-cpu`
- host role: `measurement` — published dataset host
- observer: **unavailable** (observer unavailable: no CPU lies outside the measured L3 domains): the recorder and the harness driver ran on the measured cores, recorder affinity `64-127`; the observer audit was not run, so compare these cells only with runs recorded the same way
- timestamp: `2026-11-20T08:03:43.096166Z`
- timing policy: v2, median of 3 reps, priming 100 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts; sampled on the measured cores the recorder shares, so the gate cannot tell the recorder's own load from foreign load
- command: `scripts/record_run.py ohtaka-cpu faer-kernels-ohtaka-small --vendor openblas --checkout extern/t4a-faer --upstream extern/upstream-faer --label ohtaka-97 --reps 3 --aa 2 --prime-ms 100`
- raw data: `data/results/ohtaka-cpu/faer-kernels-ohtaka-small/20261120T074420Z-ohtaka-97/`

## What was measured

- **Workload** — Direct calls into one faer revision: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: Vendor C-order cells use CBLAS/LAPACKE row-major entry points. Workspace allocation or layout conversion internal to those APIs remains timed; these are not prepared-column-major or transpose measurements.
  Known blind spot: `gemm` covers the tall class's two Householder legs (a 16-wide basis and a 240-column target over 4096 rows: the reduction and the write-back) beside a vendor `?gemm` on the same leg. A single leg beside a vendor gemm is a diagnostic reference, not the equivalent implementation of the whole update, and the row is labelled accordingly.
  Known blind spot: `householder_apply` measures block apply and the one-block sequence form as two distinct rows, labelled with basis width min(m,n) and target width n. QR fixture preparation is outside timing. Independently varied basis/target widths and multi-panel sequences are not covered.
  Known blind spot: Allocation performed inside the measured call stays inside the clock. The fork's two row splits allocate their partial buffers inside the public call, and the public scratch function does not expose them; the campaign reports that as it is instead of changing faer's API to make the boundary look cleaner.
  Known blind spot: `cholesky`, `eigh`, `solve` and `triangular_solve` are measured on square shapes only: the matrix they factor or solve with is square by definition (self-adjoint for the first two, a triangular factor for the last). `lu` and `full_piv_lu` are not gated that way -- they take the declared rectangular shapes too, with `min(m, n)` pivots and a trapezoidal `L` or `U`, which is what `dgetrf` does as well -- so a rectangular LU cell is measured in both orders beside its square one.
  Known blind spot: BLIS, sparse, iterative and GPU arms are later declarations, not machinery built now.
  Known blind spot: A vendor arm records the library it linked. Only OpenBLAS reports the kernel its runtime dispatched to; MKL names no kernel and no processor count, so its manifest leaves both empty rather than inferring them.
  Known blind spot: The upstream faer row is not measured on this profile: the 2-process-set budget (`--aa 2`) needs a 2-row arm, so only the colmat fork and OpenBLAS are measured. The upstream faer revision printed in the page header is the pinned baseline of the campaign, not a measured column.
  Known blind spot: Machine state not shown elsewhere: the ISSP ohtaka compute node exposes no cpufreq interface, so boost and governor are unrecorded; the layout is 2 sockets, 8 NUMA nodes of 16 cores (node k = CPUs 16k..16k+15), SMT off, 16 MiB L3 per 4-core CCX. Timing settings are reps 3, 2 process sets, priming 100 ms, below the 250 ms of the frozen v2 baseline (150 ms was rejected there because one cell read about 3% fast), so single cells may read a few percent fast.
  Known blind spot: No MKL arm and no SMT: this suite is measured on one ISSP ohtaka node with SMT off; at 128T every CPU is measured and no observer CPU exists (the run is recorded with an observer-unavailable caveat that covers the whole run, although the recorder was pinned to CPUs 64-127 for the rows up to 64T).
  Known blind spot: Known limitation: Coverage is the declared grid only (see `runs`): a subset of families (gemm, LU, Cholesky, QR factor) at the sizes and thread budgets of the ohtaka-cpu profile. SVD, eigen and the other decompositions are not measured on this profile.
- **Rows** — one row per engine in every table below:
  - `faer-fork` — The measured library: the tensor4all fork (`colmat`, `[lib] name = "colmat"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/t4a-faer.rev`
    Caveat: The fork crate is `colmat` (`[lib] name = "colmat"`); the upstream baseline is `faer` (`faer_upstream = { package = "faer" }`). They are distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. measured at 93da6217, colmat-rs main at the 2026-11-17 head: no tag names that revision, and the published crates.io colmat 0.34.0 was cut from e1c83482. The 2026-11-17 zen5-cpu recordings were taken at this revision.
    As measured: `faer-fork` (the provider in the measured revision; see library and upstream above)
  - `lapack-openblas` — CBLAS `?gemm` plus LAPACK drivers linked against OpenBLAS: vendor calls and their marshalling, vendor-owned threading. The vendor's own thread budget is set by the harness before the row and read back, and the vendor identity and the dispatched kernel are recorded in the manifest.
    Identity: the OpenBLAS build recorded in this run's manifest (`providers[openblas]`)
    Caveat: LAPACK and BLAS own their threading, so this row's thread budget is the vendor's, not the harness pool's. Where a family has no LAPACK route the row is absent and the CSV says the case was skipped for it.
    As measured: linked static; openblas 0.3.32; kernel dispatched to `Zen`; pthread threading; config `OpenBLAS 0.3.32 DYNAMIC_ARCH NO_AFFINITY Zen MAX_THREADS=128`

## Hardware

- CPU: `AMD EPYC 7702`
- logical CPUs: `128`
- L3: `16384K`
- L3 domains: `16 MiB shared per 4-core CCX, 32 CCXs (CPUs 4k-4k+3)`
- OS / arch: `Linux 4.18.0-372.9.1.el8.x86_64` / `x86_64`
- hostname: `c18u01n3`
- CPU sets: 1T -> `0`, 16T -> `0-15`, 32T -> `0-31`, 64T -> `0-63`, 128T -> `0-127`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 3 chronological calls; 2 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.65%, worst 36.09% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 15 of 50 engine cells exceed 5% and carry `!`; median within-process relative SD 1.33%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
The `vs faer-fork` column is the rotation-averaged ratio
exp(mean over starts of log(engine median / faer-fork median)),
with the per-start ratios beside it: `R > 1`
means the engine was slower than that first-declared row in that start. It removes a
row-position
effect only under one position effect shared by the engines and stable across starts,
and only where every declared engine has a measured row (`*` marks a cell where one has
no row, so no ratio there is position-balanced). `!` marks a cell where some engine's own
per-start median range exceeds 5% of its median, and `p` a balanced cell whose estimated
position-effect ratio exceeds 1.05. Flagged cells are retained, not discarded; the
between-start variation can exceed the difference being compared.
The `vs` columns mark only differences larger than the spread this run recorded for those
two rows (combined between-start SD, floored at 5%), larger than the sum of the two rows' `±`
and larger than each row's `W`, always with the measured library as the subject:
`🔵 <row> X faster than <row>` / `🔴 <row> X slower than <row>` (the subject is the first column of the table, the compared row is named in
the cell, not only in the column header), `·` inside the band or within the rows' `±`, `?` a row's
`W` is as large as the difference (not evidence either way; rerun the cell), `(marginal: …)` a
verdict whose largest spread is at least 60% of its difference, `-` that row is not
measured in this arm. `notes` carries the cell warnings (`*` a declared row has no measurement,
`!` between-start range over 5%, `p` position effect over 5%, dispatch-dominated means the
row is under 20 us so 4T/8T pay thread-region start-up). Per-start ratios and every
chronological sample are in the run directory named at the top of this page (`run*.csv`).
These describe the recorded revisions; they are not a regression claim.

## gemm-ohtaka, f64, f order, 1T (CPU 0)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 503.437 ± 1.0473 (W 1.34971) | 408.061 ± 2.33416 (W 0.187948) | 🔴 faer-fork 1.23× slower than `lapack-openblas` |  |

## gemm-ohtaka, f64, f order, 16T (CPU 0-15)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 45.677 ± 0.472733 (W 0.731703) | 47.111 ± 2.26824 (W 0.207457) | · | `!` between-start range over 5% |

## gemm-ohtaka, f64, f order, 32T (CPU 0-31)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 36.7546 ± 0.451258 (W 0.233059) | 35.3397 ± 4.23796 (W 0.604077) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-ohtaka, f64, f order, 64T (CPU 0-63)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 29.6241 ± 0.309093 (W 0.105518) | 36.5851 ± 2.74465 (W 0.620517) | 🔵 faer-fork 1.23× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-ohtaka, f64, f order, 128T (CPU 0-127)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 41.8219 ± 10.674 (W 0.095643) | 85.1513 ± 6.50018 (W 3.3554) | 🔵 faer-fork 2.04× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |

## linalg-ohtaka, f64, f order, 1T (CPU 0)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n2048/b1` | 169.249 ± 0.397429 (W 0.619643) | 78.4094 ± 0.545102 (W 0.539179) | 🔴 faer-fork 2.16× slower than `lapack-openblas` |  |
| `cholesky/n2048/b1 [api-inclusive]` | 138.857 ± 0.581513 (W 0.393902) | - | - | `*` one row has no measurement here |
| `lu/n2048/b1` | 175.844 ± 0.242629 (W 0.51372) | 182.93 ± 0.351858 (W 0.757192) | · |  |
| `lu/n2048/b1 [api-inclusive]` | 197.015 ± 1.32959 (W 1.00246) | - | - | `*` one row has no measurement here |
| `qr_factor/n2048/b1` | 398.429 ± 3.96259 (W 1.59341) | 807.542 ± 23.879 (W 1.88396) | 🔵 faer-fork 2.03× faster than `lapack-openblas` |  |

## linalg-ohtaka, f64, f order, 16T (CPU 0-15)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n2048/b1` | 16.3191 ± 0.117474 (W 1.23384) | 20.2765 ± 0.15621 (W 2.00713) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `cholesky/n2048/b1 [api-inclusive]` | 27.95 ± 0.495139 (W 0.350275) | - | - | `*` one row has no measurement here |
| `lu/n2048/b1` | 43.7359 ± 0.470201 (W 0.83735) | 57.4188 ± 2.17571 (W 0.818424) | 🔵 faer-fork 1.31× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/n2048/b1 [api-inclusive]` | 69.4497 ± 0.826862 (W 1.98783) | - | - | `*` one row has no measurement here |
| `qr_factor/n2048/b1` | 39.0216 ± 0.461818 (W 1.04074) | 442.917 ± 4.1759 (W 2.90573) | 🔵 faer-fork 11.35× faster than `lapack-openblas` |  |

## linalg-ohtaka, f64, f order, 32T (CPU 0-31)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n2048/b1` | 19.0237 ± 0.0854086 (W 1.43338) | 20.8068 ± 0.0951808 (W 0.283342) | 🔵 faer-fork 1.09× faster than `lapack-openblas` (marginal: spread 0.80 of the difference) |  |
| `cholesky/n2048/b1 [api-inclusive]` | 28.648 ± 2.925 (W 0.659271) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n2048/b1` | 42.185 ± 0.491436 (W 1.14266) | 94.4364 ± 1.09712 (W 0.931988) | 🔵 faer-fork 2.24× faster than `lapack-openblas` |  |
| `lu/n2048/b1 [api-inclusive]` | 66.0346 ± 0.336229 (W 1.4134) | - | - | `*` one row has no measurement here |
| `qr_factor/n2048/b1` | 42.3064 ± 0.324541 (W 0.579899) | 452.979 ± 4.65823 (W 4.05121) | 🔵 faer-fork 10.71× faster than `lapack-openblas` |  |

## linalg-ohtaka, f64, f order, 64T (CPU 0-63)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n2048/b1` | 21.1907 ± 2.5658 (W 0.688128) | 24.2012 ± 0.450257 (W 0.845828) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n2048/b1 [api-inclusive]` | 33.7714 ± 0.487423 (W 0.771328) | - | - | `*` one row has no measurement here |
| `lu/n2048/b1` | 48.2655 ± 1.73433 (W 2.53504) | 174.309 ± 18.0309 (W 33.1322) | 🔵 faer-fork 3.61× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/n2048/b1 [api-inclusive]` | 74.1596 ± 0.207856 (W 1.90402) | - | - | `*` one row has no measurement here |
| `qr_factor/n2048/b1` | 67.668 ± 6.02228 (W 2.88154) | 441.018 ± 60.2303 (W 25.6255) | 🔵 faer-fork 6.52× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |

## linalg-ohtaka, f64, f order, 128T (CPU 0-127)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n2048/b1` | 32.875 ± 0.0171756 (W 0.102312) | 24.8253 ± 0.815842 (W 1.13776) | 🔴 faer-fork 1.32× slower than `lapack-openblas` |  |
| `cholesky/n2048/b1 [api-inclusive]` | 46.4998 ± 0.458547 (W 0.261296) | - | - | `*` one row has no measurement here |
| `lu/n2048/b1` | 62.8526 ± 0.505171 (W 0.905395) | 783.744 ± 72.4423 (W 1071.67) | `?` not evidence vs `lapack-openblas`: a row's `W` is as large as the difference (rerun) | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n2048/b1 [api-inclusive]` | 84.8587 ± 2.3666 (W 0.589644) | - | - | `*` one row has no measurement here |
| `qr_factor/n2048/b1` | 92.166 ± 3.63525 (W 2.95233) | 547.389 ± 134.977 (W 728.471) | `?` not evidence vs `lapack-openblas`: a row's `W` is as large as the difference (rerun) | `!` between-start range over 5%; `p` position effect over 5% |

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
