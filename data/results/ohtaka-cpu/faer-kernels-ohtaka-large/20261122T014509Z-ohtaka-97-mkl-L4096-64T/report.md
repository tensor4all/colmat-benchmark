# `faer-kernels-ohtaka-large` on `ohtaka-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `2f3694e6fff8705cb8bbb83891fcaf4c5c3ec878`
- toolkit commit: `34cf62b2504fa68857c5c1a33c125d22c5620452`
- measured binary SHA-256: `28f65c9377b26accb0bb2db3767a576cff7269aef894be3e4e835ae53b3af97f`
- hardware profile: `ohtaka-cpu`
- host role: `measurement` — published dataset host
- timestamp: `2026-11-22T01:49:17.391470Z`
- timing policy: v2, median of 3 reps, priming 100 ms (time-based, identical on every row)
- timing baseline: **not comparable with this campaign's v2 pages** (priming 100 ms < 250 ms; 3 reps < 5; the frozen v2 settings are priming 250 ms and 5 reps, docs/protocol-v2-baseline.md). Too little priming reads a few percent fast, so compare these cells only with runs recorded the same way; a verdict here also needs a ratio beyond 1.10×
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `scripts/record_run.py ohtaka-cpu faer-kernels-ohtaka-large --vendor mkl --checkout /home/k0156/k015600/cb97c/extern/t4a-faer --upstream /home/k0156/k015600/cb97c/extern/upstream-faer --label ohtaka-97-mkl-L4096-64T --reps 3 --aa 2 --prime-ms 100 --threads 64 --shapes 4096,4096x4096x4096 --families gemm,lu,cholesky,qr_factor --regimes gemm-ohtaka,linalg-ohtaka`
- raw data: `data/results/ohtaka-cpu/faer-kernels-ohtaka-large/20261122T014509Z-ohtaka-97-mkl-L4096-64T/`

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
  Known blind spot: The upstream faer row is not measured on this profile: the 2-process-set budget (`--aa 2`) needs a 2-row arm, so each run measures only the colmat fork and its one vendor arm (MKL for this suite on ohtaka; the small and c64 suites carry the OpenBLAS arm). The upstream faer revision printed in the page header is the pinned baseline of the campaign, not a measured column.
  Known blind spot: Machine state not shown elsewhere: the ISSP ohtaka compute node exposes no cpufreq interface, so boost and governor are unrecorded; the layout is 2 sockets, 8 NUMA nodes of 16 cores (node k = CPUs 16k..16k+15), SMT off, 16 MiB L3 per 4-core CCX. Timing settings are reps 3, 2 process sets, priming 100 ms, below the 250 ms of the frozen v2 baseline (150 ms was rejected there because one cell read about 3% fast), so single cells may read a few percent fast.
  Known blind spot: No OpenBLAS arm here and no SMT: this suite is recorded with the MKL arm (oneapi_mkl/2025.2.0 module, MKL 2025.2 with its Intel OpenMP runtime) on one ISSP ohtaka node with SMT off, so its vendor row is not directly comparable with the OpenBLAS rows of the small and c64 suites; at 128T every CPU is measured and no observer CPU exists (the run is recorded with an observer-unavailable caveat that covers the whole run, although the recorder was pinned to CPUs 64-127 for the rows up to 64T).
  Known blind spot: Recorded in slices: the grid is covered by 18 partial runs under `data/results/ohtaka-cpu/faer-kernels-ohtaka-large/` (4096 at each thread budget; 8192 per thread budget and per family), each a 30-minute `i8cpu` allocation on its own node of the same type. The published page is one of those runs (the publisher keeps one run per page and marks it partial); the other slices are in the raw data, not on the page. Cells from different slices were timed on different nodes and in different sessions.
  Known blind spot: Known limitation: Coverage is the declared grid only (see `runs`): a subset of families (gemm, LU, Cholesky, QR factor) at the sizes and thread budgets of the ohtaka-cpu profile. SVD, eigen and the other decompositions are not measured on this profile.
  Known blind spot: Residual-check coverage narrowed: the verification census (`verify`) covers the 4096 cells only, at every thread budget. A timed process also checks the result of its last timed call at the cell's own size (the harness's `timing::measure` ends in the cell's check), so a wrong 8192 result fails its row; that check is not part of the census and records no residual. Where this page says every timed cell matches a verification census, read it with this limit.
- **Rows** — one row per engine in every table below:
  - `faer-fork` — The measured library: the tensor4all fork (`colmat`, `[lib] name = "colmat"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/t4a-faer.rev`
    Caveat: The fork crate is `colmat` (`[lib] name = "colmat"`); the upstream baseline is `faer` (`faer_upstream = { package = "faer" }`). They are distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. measured at 93da6217, colmat-rs main at the 2026-11-17 head: no tag names that revision, and the published crates.io colmat 0.34.0 was cut from e1c83482. The 2026-11-17 zen5-cpu recordings were taken at this revision.
    As measured: `faer-fork` (the provider in the measured revision; see library and upstream above)
  - `lapack-mkl` — The same calls against Intel oneAPI MKL, linked from the installation `MKLROOT` names. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run; no page carries two vendor arms. The LP64 interface layer and the threading layer are pinned by the harness before the first MKL call, because MKL otherwise lets the environment choose them and the interface layer is an ABI.
    Identity: the MKL build recorded in this run's manifest (`providers[mkl]`)
    Caveat: MKL reports no dispatched-kernel name and no processor count, so its manifest records neither rather than inferring them. Its threading is vendor-owned, as for OpenBLAS.
    As measured: linked shared; mkl 2025.2; kernel not reported by the vendor

## Hardware

- CPU: `AMD EPYC 7702`
- logical CPUs: `128`
- L3: `16384K`
- L3 domains: `16 MiB shared per 4-core CCX, 32 CCXs (CPUs 4k-4k+3)`
- OS / arch: `Linux 4.18.0-372.9.1.el8.x86_64` / `x86_64`
- hostname: `c15u01n3`
- CPU sets: 64T -> `0-63`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 3 chronological calls; 2 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 2.31%, worst 3.81% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 0 of 10 engine cells exceed 5% and carry `!`; median within-process relative SD 0.48%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
Below the timing baseline (see the header) `·` also means a ratio within 1.10×.

## gemm-ohtaka, f64, f order, 64T (CPU 0-63)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `gemm/4096x4096x4096/b1` | 187.016 ± 0.129748 (W 0.418752) | 231.571 ± 0.349497 (W 0.819424) | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |

## linalg-ohtaka, f64, f order, 64T (CPU 0-63)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `cholesky/n4096/b1` | 108.592 ± 2.27231 (W 0.573467) | 111.946 ± 3.01977 (W 4.0318) | · |  |
| `cholesky/n4096/b1 [api-inclusive]` | 149.33 ± 2.56428 (W 0.489033) | - | - | `*` one row has no measurement here |
| `lu/n4096/b1` | 220.475 ± 2.69316 (W 3.18157) | 162.263 ± 2.96478 (W 1.55708) | 🔴 faer-fork 1.36× slower than `lapack-mkl` |  |
| `lu/n4096/b1 [api-inclusive]` | 278.218 ± 6.1381 (W 1.68063) | - | - | `*` one row has no measurement here |
| `qr_factor/n4096/b1` | 306.06 ± 4.72654 (W 1.59924) | 281.877 ± 0.584077 (W 1.2817) | · |  |

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
