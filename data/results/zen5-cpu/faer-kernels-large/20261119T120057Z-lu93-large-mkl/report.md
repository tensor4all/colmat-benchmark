# `faer-kernels-large` on `zen5-cpu`

- colmat (fork, measured) commit: `a5536ad3f4b4a70344ae73500b6edf4782522fac`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `9988f5a5efaca26af1a6e80dfd5736564263dc66`
- toolkit commit: `ec1366c391cbd5023116ddcb6fdadc9c5b76f582`
- measured binary SHA-256: `0081dbf1739db6d31ec996b1ff138e70b8c19e3755c87e8909a1c94ce28fa2a5`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-19T12:07:52.621948Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `scripts/record_run.py zen5-cpu faer-kernels-large --checkout extern/t4a-faer --upstream extern/upstream-faer --vendor mkl --engines faer-fork,lapack-mkl --families lu --dtypes f64 --threads 4,8 --regimes linalg-large,linalg-boundary --shapes 1024,1536,255,257,511,513 --reps 5 --aa 6 --prime-ms 250 --label lu93-large-mkl --no-publish`
- raw data: `data/results/zen5-cpu/faer-kernels-large/20261119T120057Z-lu93-large-mkl/`

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
  Known blind spot: Known limitation: Coverage is the declared grid: f64/c64/f32/c32 at 1/4/8T for the small suite, f64/c64 at 4/8T for the large suite's 1024/1536 + boundary tier; the adoption-only 2048/3072/4096 tiers are not measured.
- **Rows** — one row per engine in every table below:
  - `faer-fork` — The measured library: the tensor4all fork (`colmat`, `[lib] name = "colmat"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/t4a-faer.rev`
    Caveat: The fork crate is `colmat` (`[lib] name = "colmat"`); the upstream baseline is `faer` (`faer_upstream = { package = "faer" }`). They are distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. measured at 93da6217, colmat-rs main at the 2026-11-17 head: no tag names that revision, and the published crates.io colmat 0.34.0 was cut from e1c83482. The 2026-11-17 zen5-cpu recordings were taken at this revision.
    As measured: `faer-fork` (the provider in the measured revision; see library and upstream above)
  - `lapack-mkl` — The same calls against Intel oneAPI MKL, linked from the installation `MKLROOT` names. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run; no page carries two vendor arms. The LP64 interface layer and the threading layer are pinned by the harness before the first MKL call, because MKL otherwise lets the environment choose them and the interface layer is an ABI.
    Identity: the MKL build recorded in this run's manifest (`providers[mkl]`)
    Caveat: MKL reports no dispatched-kernel name and no processor count, so its manifest records neither rather than inferring them. Its threading is vendor-owned, as for OpenBLAS.
    As measured: linked shared; mkl 2026.2; kernel not reported by the vendor

## Hardware

- CPU: `AMD Ryzen AI 9 HX 470`
- logical CPUs: `24`
- L3: `24 MiB (2 instances)`
- L3 domains: `0-3: 16 MiB shared; 4-11: 8 MiB shared`
- OS / arch: `Linux 7.0.0-43-generic` / `x86_64`
- hostname: `shinaoka-EVO-X1Pro`
- CPU sets: 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.23%, worst 5.71% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 2 of 60 engine cells exceed 5% and carry `!`; median within-process relative SD 0.41%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
two rows (combined between-start SD, floored at 5%), always with the measured library as the subject:
`🔵 <row> X faster than <row>` / `🔴 <row> X slower than <row>` (the subject is the first column of the table, the compared row is named in
the cell, not only in the column header), `·` inside the band, `-` that row is not
measured in this arm. `notes` carries the cell warnings (`*` a declared row has no measurement,
`!` between-start range over 5%, `p` position effect over 5%, dispatch-dominated means the
row is under 20 us so 4T/8T pay thread-region start-up). Per-start ratios and every
chronological sample are in the run directory named at the top of this page (`run*.csv`).
These describe the recorded revisions; they are not a regression claim.

## linalg-large, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n1024/b1` | 10.8391 ± 0.0304577 (W 0.0290482) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 19.0363 ± 0.119669 (W 0.0497183) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 31.4389 ± 0.313361 (W 0.198959) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 50.0547 ± 0.231442 (W 0.107017) | - | - | `*` one row has no measurement here |

## linalg-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n1024/b1` | 7.02266 ± 0.156435 (W 0.0329548) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1024/b1 [api-inclusive]` | 15.8066 ± 0.0860784 (W 0.0602124) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 19.599 ± 0.0471954 (W 0.120066) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 39.6631 ± 0.433302 (W 0.196188) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n1024/b1` | 8.80239 ± 0.0259067 (W 0.0159206) | 8.83647 ± 0.0696366 (W 0.0141958) | · |  |
| `lu/n1024/b1 [api-inclusive]` | 16.4041 ± 0.0594267 (W 0.0334777) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 26.39 ± 0.0665676 (W 0.0429294) | 27.8504 ± 0.117463 (W 0.0550935) | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `lu/n1536/b1 [api-inclusive]` | 43.3481 ± 0.201099 (W 0.188655) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n1024/b1` | 5.76069 ± 0.0162441 (W 0.013963) | 5.25821 ± 0.0259412 (W 0.0209767) | 🔴 faer-fork 1.10× slower than `lapack-mkl` |  |
| `lu/n1024/b1 [api-inclusive]` | 13.2063 ± 0.0606338 (W 0.0424865) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 16.2644 ± 0.0557536 (W 0.0695547) | 16.5247 ± 0.0738332 (W 0.0828589) | · |  |
| `lu/n1536/b1 [api-inclusive]` | 33.3674 ± 0.142438 (W 0.103051) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `lu/1024`, `lu/1536`.

## linalg-boundary, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n255/b1` | 0.429203 ± 0.00307298 (W 0.00340121) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.719074 ± 0.00258285 (W 0.00576791) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.42469 ± 0.00232197 (W 0.00303584) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.785587 ± 0.0051889 (W 0.0036421) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.75559 ± 0.00795648 (W 0.00431905) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.57467 ± 0.0464778 (W 0.0102116) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.85095 ± 0.00753245 (W 0.00670949) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.5399 ± 0.03026 (W 0.0106246) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n255/b1` | 0.420628 ± 0.00144686 (W 0.00271727) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.688092 ± 0.0044689 (W 0.00214935) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.353564 ± 0.00177146 (W 0.00195534) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.747086 ± 0.0014474 (W 0.00395257) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.28525 ± 0.00268164 (W 0.00324788) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.20966 ± 0.0128522 (W 0.00958661) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.35315 ± 0.00329981 (W 0.00459195) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.17868 ± 0.0330968 (W 0.00841081) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n255/b1` | 0.30593 ± 0.00273711 (W 0.00322652) | 0.287586 ± 0.00251102 (W 0.00247244) | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `lu/n255/b1 [api-inclusive]` | 0.611606 ± 0.00247471 (W 0.00289185) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.326443 ± 0.0031774 (W 0.00279016) | 0.297134 ± 0.00186836 (W 0.00267251) | 🔴 faer-fork 1.10× slower than `lapack-mkl` |  |
| `lu/n257/b1 [api-inclusive]` | 0.656134 ± 0.0047626 (W 0.00439382) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.48216 ± 0.00682193 (W 0.00487049) | 1.40331 ± 0.00753323 (W 0.00545346) | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 2.96035 ± 0.0112609 (W 0.00707332) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.54926 ± 0.00693948 (W 0.00472072) | 1.42948 ± 0.00664382 (W 0.00466336) | 🔴 faer-fork 1.08× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.98415 ± 0.0632464 (W 0.00760333) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-boundary, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n255/b1` | 0.278695 ± 0.00171196 (W 0.00190052) | 0.272834 ± 0.00231288 (W 0.00278972) | · |  |
| `lu/n255/b1 [api-inclusive]` | 0.578696 ± 0.00432454 (W 0.00331741) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.290296 ± 0.00145061 (W 0.00210646) | 0.282583 ± 0.00133105 (W 0.00230031) | · |  |
| `lu/n257/b1 [api-inclusive]` | 0.625101 ± 0.00513479 (W 0.00278072) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.12369 ± 0.00573555 (W 0.0026035) | 1.00478 ± 0.00558676 (W 0.00395721) | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 2.61244 ± 0.00876973 (W 0.00761464) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.16488 ± 0.00428059 (W 0.0035116) | 1.02785 ± 0.00345934 (W 0.00442496) | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.59837 ± 0.00740153 (W 0.00816727) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `lu/255`, `lu/257`, `lu/511`, `lu/513`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
