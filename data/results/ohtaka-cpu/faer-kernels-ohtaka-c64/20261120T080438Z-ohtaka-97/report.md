# `faer-kernels-ohtaka-c64` on `ohtaka-cpu`

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
- timestamp: `2026-11-20T19:02:13.034517Z`
- timing policy: v2, median of 3 reps, priming 100 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts; sampled on the measured cores the recorder shares, so the gate cannot tell the recorder's own load from foreign load
- command: `scripts/record_run.py ohtaka-cpu faer-kernels-ohtaka-c64 --vendor openblas --checkout extern/t4a-faer --upstream extern/upstream-faer --label ohtaka-97 --reps 3 --aa 2 --prime-ms 100`
- raw data: `data/results/ohtaka-cpu/faer-kernels-ohtaka-c64/20261120T080438Z-ohtaka-97/`

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
  Known blind spot: Residual-check coverage narrowed: the residual check runs at 2048 (every thread budget) only; the 8192 c64 spot cells are timed on the same code paths without a residual check at their own size, because the harness check product is a single-threaded blocked triple loop. Where this page says every timed cell matches a verification census, read it with this limit.
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
- CPU sets: 64T -> `0-63`, 128T -> `0-127`

Every verified cell matches a successful verification census, including operation variants.
Verification was narrowed (`verification.grid`): only the shapes n = 2048, 2048x2048x2048 were verified. Cells at the other declared shapes of those populations were timed and had to succeed, but no verification checked them; a population the grid does not name was verified at every shape.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 3 chronological calls; 2 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 8.69%, worst 111.28% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 26 of 40 engine cells exceed 5% and carry `!`; median within-process relative SD 2.10%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

## gemm-ohtaka, c64, f order, 64T (CPU 0-63)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 95.6812 ± 28.7104 (W 1.05613) | 71.541 ± 4.18888 (W 0.987909) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/8192x8192x8192/b1` | 4812.56 ± 815.7 (W 258.118) | 3423.77 ± 196.143 (W 196.608) | 🔴 faer-fork 1.41× slower than `lapack-openblas` (marginal: spread 0.73 of the difference) | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-ohtaka, c64, f order, 128T (CPU 0-127)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 93.8904 ± 3.14048 (W 0.407758) | 165.21 ± 7.06617 (W 2.21407) | 🔵 faer-fork 1.76× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/8192x8192x8192/b1` | 5171.03 ± 394.173 (W 983.243) | 3915.59 ± 254.542 (W 191.589) | 🔴 faer-fork 1.32× slower than `lapack-openblas` (marginal: spread 0.78 of the difference) | `!` between-start range over 5%; `p` position effect over 5% |

## linalg-ohtaka, c64, f order, 64T (CPU 0-63)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n2048/b1` | 38.4877 ± 0.20689 (W 1.9894) | 64.2069 ± 7.64403 (W 3.72403) | 🔵 faer-fork 1.67× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n2048/b1 [api-inclusive]` | 58.5656 ± 3.38664 (W 0.944355) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n8192/b1` | 2334.55 ± 11.009 (W 4.79634) | 1374.87 ± 92.3381 (W 30.1694) | 🔴 faer-fork 1.70× slower than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n8192/b1 [api-inclusive]` | 2562.99 ± 4.79015 (W 12.0122) | - | - | `*` one row has no measurement here |
| `lu/n2048/b1` | 104.699 ± 11.5029 (W 1.94454) | 312.218 ± 70.079 (W 34.847) | 🔵 faer-fork 2.98× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n2048/b1 [api-inclusive]` | 143.553 ± 11.0392 (W 1.71738) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n8192/b1` | 6023.49 ± 2243.38 (W 892.07) | 9906.48 ± 7795.2 (W 4756.2) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n8192/b1 [api-inclusive]` | 6482.51 ± 41.7922 (W 108.724) | - | - | `*` one row has no measurement here |
| `qr_factor/n2048/b1` | 127.53 ± 1.68336 (W 5.58116) | 555.199 ± 71.8102 (W 19.9205) | 🔵 faer-fork 4.35× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n8192/b1` | 7766.64 ± 108.117 (W 42.0831) | 31607.6 ± 3814.79 (W 12332.6) | 🔵 faer-fork 4.07× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |

## linalg-ohtaka, c64, f order, 128T (CPU 0-127)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n2048/b1` | 71.1856 ± 0.081441 (W 0.316939) | 144.852 ± 8.09134 (W 1.40817) | 🔵 faer-fork 2.03× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n2048/b1 [api-inclusive]` | 94.8766 ± 0.693991 (W 1.31609) | - | - | `*` one row has no measurement here |
| `cholesky/n8192/b1` | 3024.82 ± 33.1569 (W 3.62618) | 2034.89 ± 131.04 (W 259.114) | 🔴 faer-fork 1.49× slower than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n8192/b1 [api-inclusive]` | 3112.87 ± 33.6346 (W 85.3289) | - | - | `*` one row has no measurement here |
| `lu/n2048/b1` | 154.974 ± 0.562034 (W 0.382624) | 1302.28 ± 510.218 (W 4275.44) | `?` not evidence vs `lapack-openblas`: a row's `W` is as large as the difference (rerun) | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n2048/b1 [api-inclusive]` | 190.114 ± 2.12544 (W 1.09783) | - | - | `*` one row has no measurement here |
| `lu/n8192/b1` | 12382.8 ± 896.598 (W 698.291) | 30819.3 ± 12159.9 (W 7445.42) | 🔵 faer-fork 2.49× faster than `lapack-openblas` (marginal: spread 0.71 of the difference) | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n8192/b1 [api-inclusive]` | 17713 ± 28.5476 (W 348.83) | - | - | `*` one row has no measurement here |
| `qr_factor/n2048/b1` | 182.033 ± 7.79277 (W 4.64595) | 726.076 ± 119.051 (W 5.35254) | 🔵 faer-fork 3.99× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n8192/b1` | 8749.64 ± 1811.2 (W 158.348) | 89301.3 ± 32685.7 (W 15274.4) | 🔵 faer-fork 10.21× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
