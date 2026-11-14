# `faer-kernels-large` on `zen5-cpu`

- t4a-faer (fork, measured) commit: `325e482b6532624d06236ad6991a41c57b0c23d1`
- t4a-faer version: `0.28.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `b6a4cda3c9c8041de3110494bff9fe09eaf60aac`
- toolkit commit: `d487991f2fa2e1a151f78389c5ef449a614e750c`
- measured binary SHA-256: `3a4a0f9413b5180191d62b274e8507303a63c0369bc8fa5749368025696a9d52`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-14T05:20:50.748901Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `/home/shinaoka/projects/tensor4all/colmat-benchmark/scripts/record_run.py zen5-cpu faer-kernels-large --checkout /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/t4a-faer --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor openblas --engines faer-fork,lapack-openblas --dtypes f64,c64 --reps 5 --aa 6 --prime-ms 250 --label pub-325e482-20261113-large1024-openblas --no-publish --threads 4,8 --regimes linalg-large,gemm-large,linalg-boundary --shapes 1024,1024x1024x1024,255,257,511,513`
- raw data: `data/results/zen5-cpu/faer-kernels-large/20261114T014635Z-pub-325e482-20261113-large1024-openblas/`

## What was measured

- **Workload** — Direct calls into one faer revision: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: Vendor C-order cells use CBLAS/LAPACKE row-major entry points. Workspace allocation or layout conversion internal to those APIs remains timed; these are not prepared-column-major or transpose measurements.
  Known blind spot: `gemm` covers the tall class's two Householder legs (a 16-wide basis and a 240-column target over 4096 rows: the reduction and the write-back) beside a vendor `?gemm` on the same leg. A single leg beside a vendor gemm is a diagnostic reference, not the equivalent implementation of the whole update, and the row is labelled accordingly.
  Known blind spot: `householder_apply` measures block apply and the one-block sequence form as two distinct rows, labelled with basis width min(m,n) and target width n. QR fixture preparation is outside timing. Independently varied basis/target widths and multi-panel sequences are not covered.
  Known blind spot: Allocation performed inside the measured call stays inside the clock. The fork's two row splits allocate their partial buffers inside the public call, and the public scratch function does not expose them; the campaign reports that as it is instead of changing faer's API to make the boundary look cleaner.
  Known blind spot: `cholesky`, `eigh`, `solve` and `triangular_solve` are measured on square shapes only: the matrix they factor or solve with is square by definition (self-adjoint for the first two, a triangular factor for the last). `lu` and `full_piv_lu` are not gated that way -- they take the declared rectangular shapes too, with `min(m, n)` pivots and a trapezoidal `L` or `U`, which is what `dgetrf` does as well -- so a rectangular LU cell is measured in both orders beside its square one.
  Known blind spot: `f32`/`c32`, the large regime's 1024 shapes, BLIS, sparse, iterative and GPU arms are later declarations, not machinery built now.
  Known blind spot: A vendor arm records the library it linked. Only OpenBLAS reports the kernel its runtime dispatched to; MKL names no kernel and no processor count, so its manifest leaves both empty rather than inferring them.
  Known blind spot: Known limitation (71b25ffb recording): multi-thread figures include threadteam 0.2.0 and are not final for the 0.3.0 runtime.
- **Rows** — one row per engine in every table below:
  - `faer-fork` — The measured library: the tensor4all fork (`t4a-faer`, `[lib] name = "faer"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/t4a-faer.rev`
    Caveat: Both faer crates declare `[lib] name = "faer"`. They are distinct dependency keys (`faer_fork = { package = "t4a-faer" }`, `faer_upstream = { package = "faer" }`) and therefore distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. For the 71b25ffb recording: measured at 71b25ffb, the commit immediately before the 0.28.0 release commit 325e482; the only difference is Cargo.lock and faer/Cargo.toml (threadteam 0.2.0 in the measured build, 0.3.0 as shipped, plus the version string). Single-thread numbers apply to 0.28.0 as released; multi-thread numbers carry that threadteam difference and will be re-measured in the next campaign.
    As measured: `faer-fork` (the provider in the measured revision; see library and upstream above)
  - `lapack-openblas` — CBLAS `?gemm` plus LAPACK drivers linked against OpenBLAS: vendor calls and their marshalling, vendor-owned threading. The vendor's own thread budget is set by the harness before the row and read back, and the vendor identity and the dispatched kernel are recorded in the manifest.
    Identity: the OpenBLAS build recorded in this run's manifest (`providers[openblas]`)
    Caveat: LAPACK and BLAS own their threading, so this row's thread budget is the vendor's, not the harness pool's. Where a family has no LAPACK route the row is absent and the CSV says the case was skipped for it.
    As measured: linked static; openblas 0.3.32; kernel dispatched to `Cooperlake`; pthread threading; config `OpenBLAS 0.3.32 DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=24`

## Hardware

- CPU: `AMD Ryzen AI 9 HX 470`
- logical CPUs: `24`
- L3: `24 MiB (2 instances)`
- L3 domains: `0-3: 16 MiB shared; 4-11: 8 MiB shared`
- OS / arch: `Linux 7.0.0-38-generic` / `x86_64`
- hostname: `shinaoka-EVO-X1Pro`
- CPU sets: 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.02%, worst 6378.28% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 35 of 976 engine cells exceed 5% and carry `!`; median within-process relative SD 0.25%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 7.32446 ± 0.163504 (W 0.0178147) | 19.9296 ± 0.120343 (W 0.0312159) | 🔵 faer-fork 2.72× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n1024/b1 [in_place]` | 7.42487 ± 0.0395932 (W 0.0116876) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 116.934 ± 0.293352 (W 0.900053) | 144.054 ± 0.575239 (W 0.258453) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 114.764 ± 0.253815 (W 0.115828) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 66.6509 ± 0.283377 (W 0.137535) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 58.3824 ± 0.160029 (W 0.0894435) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 21.707 ± 0.11342 (W 0.0540595) | 26.403 ± 0.321539 (W 0.248229) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `lu/n1024/b1 [in_place]` | 13.4449 ± 0.0397287 (W 0.0368772) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 51.2731 ± 0.229032 (W 0.164388) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 19.5847 ± 0.139339 (W 0.0445998) | 53.8415 ± 1.38556 (W 0.110966) | 🔵 faer-fork 2.75× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n1024/b1` | 14.3545 ± 0.0326017 (W 0.0245603) | 25.7532 ± 0.562138 (W 0.0421335) | 🔵 faer-fork 1.79× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n1024/b1 [in_place]` | 12.8489 ± 0.0761835 (W 0.0675493) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 243.186 ± 0.861044 (W 0.792719) | 460.063 ± 3.16254 (W 1.92747) | 🔵 faer-fork 1.89× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 221.627 ± 0.580588 (W 0.357507) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 123.807 ± 0.382677 (W 0.178387) | 228.248 ± 3.25136 (W 0.668458) | 🔵 faer-fork 1.84× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 123.906 ± 0.394606 (W 0.168193) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.086893 ± 0.000750843 (W 0.000535887) | - | - | `*` one row has no measurement here |

## linalg-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 6.15124 ± 0.0623806 (W 0.0258342) | 18.2494 ± 0.0757819 (W 0.030157) | 🔵 faer-fork 2.97× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [in_place]` | 5.40277 ± 0.015028 (W 0.0124509) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 91.2855 ± 0.52551 (W 0.596882) | 115.095 ± 0.489601 (W 0.19951) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 90.5318 ± 0.166536 (W 0.147051) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 54.6366 ± 0.190945 (W 0.090418) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 45.6396 ± 0.246328 (W 0.185588) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 18.5882 ± 0.0590153 (W 0.0445247) | 20.6298 ± 1.08377 (W 0.703453) | 🔵 faer-fork 1.11× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/n1024/b1 [in_place]` | 9.65057 ± 0.0505397 (W 0.0267573) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 33.5938 ± 0.209688 (W 0.0412907) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 10.5487 ± 0.0509731 (W 0.0193363) | 45.7096 ± 2.00077 (W 0.0675103) | 🔵 faer-fork 4.33× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n1024/b1` | 11.8388 ± 0.319993 (W 0.154319) | 20.4833 ± 1.93022 (W 0.056636) | 🔵 faer-fork 1.73× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n1024/b1 [in_place]` | 9.00144 ± 0.0430685 (W 0.0267787) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 181.319 ± 1.01049 (W 0.847516) | 417.453 ± 4.27182 (W 2.59926) | 🔵 faer-fork 2.30× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 160.309 ± 0.270309 (W 0.253537) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 97 ± 0.680679 (W 0.0957846) | 216.245 ± 4.58984 (W 0.2943) | 🔵 faer-fork 2.23× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 96.9582 ± 0.406443 (W 0.262405) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.0744845 ± 0.000622054 (W 0.0004608) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 5.20737 ± 0.022597 (W 0.00892796) | 6.48699 ± 0.0148709 (W 0.0180738) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [in_place]` | 4.32681 ± 0.0181327 (W 0.00706575) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 114.784 ± 0.201526 (W 0.92764) | 128.065 ± 0.549289 (W 0.114604) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 112.567 ± 0.159673 (W 0.100525) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 64.3648 ± 0.160276 (W 0.0979249) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 61.8997 ± 0.325012 (W 0.0927036) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 19.1893 ± 0.0647119 (W 0.0353117) | 12.7723 ± 0.0434509 (W 0.0241924) | 🔴 faer-fork 1.50× slower than `lapack-openblas` |  |
| `lu/n1024/b1 [in_place]` | 11.8107 ± 0.0423177 (W 0.0235027) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 48.6461 ± 0.0622647 (W 0.132785) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 17.4345 ± 0.0338633 (W 0.0382289) | 37.3845 ± 0.227239 (W 0.0473164) | 🔵 faer-fork 2.14× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 12.2064 ± 0.045163 (W 0.0187182) | 12.3239 ± 0.054173 (W 0.0147489) | · |  |
| `solve/n1024/b1 [in_place]` | 10.8597 ± 0.0308745 (W 0.0174908) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 236.586 ± 0.348401 (W 0.568644) | 412.668 ± 5.30659 (W 3.61862) | 🔵 faer-fork 1.74× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 214.957 ± 0.493332 (W 0.236845) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 115.379 ± 0.431342 (W 0.167942) | 212.826 ± 2.91233 (W 0.588032) | 🔵 faer-fork 1.84× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 115.605 ± 0.515306 (W 0.100802) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.110017 ± 0.00109741 (W 0.000923038) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 3.73951 ± 0.0689598 (W 0.0125568) | 4.64167 ± 0.0214947 (W 0.0123504) | 🔵 faer-fork 1.24× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n1024/b1 [in_place]` | 2.76546 ± 0.0269099 (W 0.00590633) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 89.4315 ± 0.150105 (W 0.622457) | 100.719 ± 0.332595 (W 0.218878) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 88.2038 ± 0.198281 (W 0.0955305) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 52.2391 ± 0.271916 (W 0.0762877) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 46.9043 ± 0.202181 (W 0.394727) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 16.1437 ± 0.139656 (W 0.0423704) | 6.71785 ± 0.0317593 (W 0.0132457) | 🔴 faer-fork 2.40× slower than `lapack-openblas` |  |
| `lu/n1024/b1 [in_place]` | 8.74648 ± 0.0201774 (W 0.0229927) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 31.1548 ± 0.132057 (W 0.0660254) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 9.11352 ± 0.0139873 (W 0.0226009) | 30.2824 ± 0.25879 (W 0.0312703) | 🔵 faer-fork 3.32× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 9.17143 ± 0.0149228 (W 0.0197448) | 6.57264 ± 0.00731694 (W 0.0140669) | 🔴 faer-fork 1.40× slower than `lapack-openblas` |  |
| `solve/n1024/b1 [in_place]` | 7.64127 ± 0.0119473 (W 0.0232974) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 174.348 ± 0.286838 (W 0.492099) | 373.401 ± 4.05354 (W 3.89227) | 🔵 faer-fork 2.14× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 153.607 ± 0.540738 (W 0.398752) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 88.5782 ± 0.184625 (W 0.228294) | 200.061 ± 2.31267 (W 0.694805) | 🔵 faer-fork 2.26× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 88.6329 ± 0.171811 (W 0.0628544) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.094892 ± 0.000839048 (W 0.000452541) | - | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 19.5735 ± 0.120525 (W 0.228598) | 34.0984 ± 0.100311 (W 0.035772) | 🔵 faer-fork 1.74× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [in_place]` | 20.1215 ± 0.0442439 (W 0.0181068) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 244.904 ± 0.36699 (W 0.698625) | 436.976 ± 1.68782 (W 1.01977) | 🔵 faer-fork 1.78× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 233.696 ± 0.548103 (W 0.531715) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 153.659 ± 0.515155 (W 0.957351) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 136.707 ± 0.397818 (W 0.66711) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 52.8717 ± 0.133382 (W 0.166818) | 49.5445 ± 1.72737 (W 0.605113) | 🔴 faer-fork 1.07× slower than `lapack-openblas` | `!` between-start range over 5% |
| `lu/n1024/b1 [in_place]` | 35.8167 ± 0.134144 (W 0.160793) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 167.875 ± 0.704935 (W 0.798133) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 64.1523 ± 0.371179 (W 0.138598) | 108.297 ± 0.28482 (W 0.244882) | 🔵 faer-fork 1.69× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 39.5015 ± 0.0761241 (W 0.153742) | 49.3043 ± 1.50725 (W 1.41999) | 🔵 faer-fork 1.25× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n1024/b1 [in_place]` | 35.2555 ± 0.171924 (W 0.179502) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 577.813 ± 1.65746 (W 1.43381) | 752.194 ± 4.12324 (W 3.01724) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 534.916 ± 1.01483 (W 0.885004) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 317.399 ± 1.02927 (W 3.17187) | 347.581 ± 1.12474 (W 2.29533) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 316.94 ± 1.18306 (W 0.773287) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.192649 ± 0.0024264 (W 0.0033615) | - | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 13.7526 ± 0.0930868 (W 0.106413) | 25.3403 ± 0.143592 (W 0.0223873) | 🔵 faer-fork 1.84× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [in_place]` | 13.6898 ± 0.0211784 (W 0.0178859) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 165.567 ± 0.436409 (W 0.658159) | 240.167 ± 0.548241 (W 0.780996) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 155.342 ± 0.206628 (W 0.361403) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 95.5785 ± 0.553283 (W 0.493811) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 76.8466 ± 0.416184 (W 0.260103) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 41.4705 ± 0.275058 (W 0.329293) | 34.6629 ± 1.10283 (W 1.03871) | 🔴 faer-fork 1.20× slower than `lapack-openblas` | `!` between-start range over 5% |
| `lu/n1024/b1 [in_place]` | 22.9829 ± 0.030547 (W 0.033882) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 102.497 ± 0.340005 (W 0.577629) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 35.1044 ± 0.0565491 (W 0.0347208) | 77.6226 ± 0.961009 (W 0.998597) | 🔵 faer-fork 2.21× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 27.273 ± 0.0589135 (W 0.0626933) | 30.7858 ± 0.57522 (W 0.213204) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `solve/n1024/b1 [in_place]` | 22.9179 ± 0.0693858 (W 0.0609398) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 364.998 ± 0.978613 (W 1.31441) | 567.869 ± 2.87794 (W 4.14634) | 🔵 faer-fork 1.56× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 321.525 ± 1.24592 (W 1.38575) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 195.273 ± 0.753056 (W 2.89273) | 276.959 ± 0.697059 (W 2.53214) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 193.698 ± 0.679703 (W 0.946856) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.134401 ± 0.0019616 (W 0.000682757) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 16.8456 ± 0.0186813 (W 0.0204529) | 19.6132 ± 0.0196342 (W 0.0187642) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [in_place]` | 15.108 ± 0.0276489 (W 0.0136621) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 241.468 ± 0.345014 (W 0.798416) | 422.219 ± 0.344576 (W 1.73327) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 230.552 ± 0.345258 (W 0.379085) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 149.191 ± 0.533676 (W 0.525672) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 140.828 ± 0.22332 (W 0.337197) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 47.4462 ± 0.313889 (W 0.0997215) | 32.8577 ± 0.0709633 (W 0.0354385) | 🔴 faer-fork 1.44× slower than `lapack-openblas` |  |
| `lu/n1024/b1 [in_place]` | 32.2378 ± 0.0401716 (W 0.0394783) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 162.208 ± 0.183671 (W 0.701568) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 60.4017 ± 0.0892191 (W 0.062249) | 88.8107 ± 0.44791 (W 0.24402) | 🔵 faer-fork 1.47× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 34.9268 ± 0.0384917 (W 0.0355145) | 32.7707 ± 0.0432202 (W 0.0512524) | 🔴 faer-fork 1.07× slower than `lapack-openblas` |  |
| `solve/n1024/b1 [in_place]` | 32.2455 ± 0.0431149 (W 0.0267706) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 570.933 ± 0.729266 (W 1.72037) | 703.378 ± 1.78579 (W 1.63408) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 528.386 ± 1.01764 (W 0.889163) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 310.475 ± 0.933583 (W 3.18625) | 328.069 ± 0.87711 (W 0.809816) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 309.039 ± 1.2948 (W 0.631071) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.213223 ± 0.00128337 (W 0.00382433) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 10.9813 ± 0.0368887 (W 0.0227382) | 10.993 ± 0.0263643 (W 0.0231726) | · |  |
| `cholesky/n1024/b1 [in_place]` | 9.07091 ± 0.00845907 (W 0.0189998) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 163.027 ± 0.506853 (W 0.620715) | 224.301 ± 0.602237 (W 1.09744) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [in_place]` | 152.874 ± 0.192301 (W 0.137206) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 89.8605 ± 0.910964 (W 0.748921) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [in_place]` | 77.0574 ± 0.222527 (W 0.46327) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 35.7871 ± 0.326178 (W 0.114456) | 16.907 ± 0.0492405 (W 0.0336958) | 🔴 faer-fork 2.12× slower than `lapack-openblas` |  |
| `lu/n1024/b1 [in_place]` | 20.3748 ± 0.55159 (W 0.0246253) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1024/b1` | 96.5482 ± 0.147873 (W 0.213523) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 32.1028 ± 0.0620252 (W 0.0349393) | 59.5191 ± 0.263401 (W 0.0884321) | 🔵 faer-fork 1.85× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 22.849 ± 0.0780416 (W 0.044022) | 17.1533 ± 0.0501511 (W 0.0413197) | 🔴 faer-fork 1.33× slower than `lapack-openblas` |  |
| `solve/n1024/b1 [in_place]` | 19.5165 ± 0.0375494 (W 0.0188426) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 357.006 ± 0.580639 (W 0.717664) | 529.439 ± 1.70264 (W 1.93393) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [in_place]` | 313.594 ± 1.20899 (W 0.608755) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 188.129 ± 1.33337 (W 2.96414) | 260.061 ± 1.97393 (W 0.831567) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [in_place]` | 186.701 ± 0.339698 (W 0.77009) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.15316 ± 0.00186759 (W 0.000750893) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `full_piv_lu/1024`, `qr/1024`, `triangular_solve/1024`.

## gemm-large, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.5879 ± 0.0232524 (W 0.0130337) | 20.5713 ± 0.0194639 (W 0.0168385) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |

## gemm-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.99016 ± 5.94633 (W 0.0131582) | 10.6498 ± 0.138635 (W 0.0274826) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.5795 ± 0.0204808 (W 0.0155148) | 20.5823 ± 0.0360087 (W 0.0251624) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |

## gemm-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.96814 ± 1.17645 (W 0.0142933) | 10.6296 ± 0.0150429 (W 0.0163721) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.9609 ± 0.0740577 (W 0.0432347) | 82.5331 ± 0.103203 (W 0.0416047) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.194 ± 2.61901 (W 0.0642951) | 41.8483 ± 0.0778829 (W 0.218375) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 70.007 ± 0.108013 (W 0.0603746) | 82.5643 ± 0.0720573 (W 0.0477319) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.1367 ± 2.67966 (W 0.0879927) | 41.7884 ± 0.0310087 (W 0.0417203) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |

## linalg-boundary, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.235729 ± 0.00215299 (W 0.00177628) | 0.363492 ± 0.00349947 (W 0.00316641) | 🔵 faer-fork 1.54× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 0.54731 ± 0.00166239 (W 0.00228404) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.223381 ± 0.00190368 (W 0.00198732) | 0.307223 ± 0.00181228 (W 0.00273763) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [in_place]` | 0.553041 ± 0.0019665 (W 0.00215672) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.956877 ± 0.00557658 (W 0.00344746) | 1.3966 ± 0.00527648 (W 0.0049745) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 1.57775 ± 0.00471539 (W 0.00369746) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.90066 ± 0.00469828 (W 0.00311961) | 1.39358 ± 0.0128976 (W 0.00311348) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [in_place]` | 1.58287 ± 0.0051203 (W 0.0029037) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 6.39749 ± 0.023695 (W 0.0126603) | 4.81555 ± 0.020812 (W 0.00838303) | 🔴 faer-fork 1.33× slower than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 6.3863 ± 0.0255192 (W 0.0117318) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 6.40188 ± 0.0128591 (W 0.00945107) | 4.83047 ± 0.0146523 (W 0.00624426) | 🔴 faer-fork 1.33× slower than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 6.38286 ± 0.0211889 (W 0.00909722) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 23.9425 ± 0.0466068 (W 0.204491) | 19.5916 ± 0.0946346 (W 0.0305806) | 🔴 faer-fork 1.22× slower than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 23.6644 ± 0.0514691 (W 0.0250315) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 24.1104 ± 0.0409923 (W 0.123986) | 19.994 ± 0.0499297 (W 0.0211177) | 🔴 faer-fork 1.21× slower than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 24.2285 ± 0.0428797 (W 0.0233372) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 2.24315 ± 0.00836926 (W 0.0105829) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [in_place]` | 1.24354 ± 0.00624342 (W 0.00512347) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.55254 ± 0.0119176 (W 0.0075067) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [in_place]` | 1.26879 ± 0.00999584 (W 0.00430608) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 13.0107 ± 0.0507505 (W 0.0315643) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 6.17895 ± 0.025033 (W 0.011129) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.31101 ± 0.0195816 (W 0.0088141) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 6.3017 ± 0.0208521 (W 0.0101271) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.839389 ± 0.00387546 (W 0.00428383) | 0.523867 ± 0.0165513 (W 0.00341026) | 🔴 faer-fork 1.60× slower than `lapack-openblas` | `!` between-start range over 5% |
| `lu/n255/b1 [in_place]` | 0.478917 ± 0.00280463 (W 0.0028163) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.835778 ± 0.00316926 (W 0.00543803) | 0.529313 ± 0.00239339 (W 0.00332881) | 🔴 faer-fork 1.58× slower than `lapack-openblas` |  |
| `lu/n257/b1 [in_place]` | 0.513543 ± 0.00392194 (W 0.00462417) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 4.40452 ± 0.0213414 (W 0.00812244) | 2.86275 ± 0.036699 (W 0.00834297) | 🔴 faer-fork 1.54× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 2.13628 ± 0.0102264 (W 0.00975853) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.73521 ± 0.070684 (W 0.0138358) | 2.80949 ± 0.0109732 (W 0.00584364) | 🔴 faer-fork 1.33× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 2.25101 ± 0.0116534 (W 0.00996348) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.56668 ± 0.00963206 (W 0.00779054) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.67438 ± 0.0127578 (W 0.00664257) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 8.39617 ± 0.0458455 (W 0.0210099) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 8.69028 ± 0.0638638 (W 0.0194432) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.659606 ± 0.00272387 (W 0.00287749) | 1.18128 ± 0.0115966 (W 0.00348507) | 🔵 faer-fork 1.79× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 0.695013 ± 0.00152391 (W 0.00216741) | 1.12268 ± 0.00620165 (W 0.00336595) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 3.20979 ± 0.0115671 (W 0.0127826) | 4.71794 ± 0.0295935 (W 0.00977704) | 🔵 faer-fork 1.47× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 3.22858 ± 0.0120234 (W 0.00921449) | 4.89511 ± 0.0179497 (W 0.00890755) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.561051 ± 0.00233003 (W 0.00274804) | 0.521376 ± 0.0146791 (W 0.00225351) | 🔴 faer-fork 1.08× slower than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n255/b1 [in_place]` | 0.470713 ± 0.00227169 (W 0.00279976) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.839193 ± 0.00495753 (W 0.00496204) | 0.505681 ± 0.00186488 (W 0.00262492) | 🔴 faer-fork 1.66× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 0.498148 ± 0.00275039 (W 0.00299844) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 4.11392 ± 0.0667558 (W 0.0205142) | 2.70622 ± 0.0206801 (W 0.0103486) | 🔴 faer-fork 1.52× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 2.00253 ± 0.00621006 (W 0.0105626) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.70218 ± 0.0173066 (W 0.0129467) | 2.71519 ± 0.0168322 (W 0.0085408) | 🔴 faer-fork 1.36× slower than `lapack-openblas` |  |
| `solve/n513/b1 [in_place]` | 2.11897 ± 0.00792132 (W 0.00997438) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 11.2939 ± 0.0610443 (W 0.0170616) | 11.8159 ± 0.0309034 (W 0.026446) | · |  |
| `svd_thin/n255/b1 [in_place]` | 11.2792 ± 0.0382189 (W 0.0204288) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 9.03079 ± 0.0139341 (W 0.0260639) | 11.9688 ± 0.0251911 (W 0.0317469) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 9.01177 ± 0.0215286 (W 0.0170124) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 44.9294 ± 0.0661722 (W 0.0616132) | 62.2828 ± 0.526011 (W 0.187811) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [in_place]` | 44.8457 ± 0.13395 (W 0.0521561) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 41.2017 ± 0.109601 (W 0.0633311) | 63.3654 ± 0.40377 (W 0.0974519) | 🔵 faer-fork 1.54× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 41.1852 ± 0.120186 (W 0.0335924) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.33209 ± 0.0157576 (W 0.00850238) | 6.13247 ± 0.0189344 (W 0.0114345) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [in_place]` | 5.34432 ± 0.0280202 (W 0.0131836) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.11079 ± 0.0208832 (W 0.012232) | 6.17921 ± 0.0207177 (W 0.00679749) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [in_place]` | 5.13197 ± 0.0134256 (W 0.00951934) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 25.0112 ± 0.123878 (W 0.037589) | 35.4715 ± 0.357494 (W 0.059299) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 25.0021 ± 0.136278 (W 0.0477289) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 22.2668 ± 0.0601014 (W 0.0240981) | 35.9855 ± 0.366297 (W 0.0387095) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 22.298 ± 0.102389 (W 0.0352752) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.013064 ± 0.000151097 (W 0.000105042) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.01309 ± 0.000114869 (W 0.00013753) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.03208 ± 0.000308306 (W 0.000181758) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0315785 ± 0.000422144 (W 0.000168109) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.230522 ± 0.00214056 (W 0.00120352) | 0.361845 ± 0.0028265 (W 0.00267487) | 🔵 faer-fork 1.57× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 0.536555 ± 0.00352514 (W 0.00231207) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.206838 ± 0.0042077 (W 0.00154417) | 0.306701 ± 0.00221211 (W 0.00287896) | 🔵 faer-fork 1.48× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n257/b1 [in_place]` | 0.542455 ± 0.00123509 (W 0.0023988) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.816544 ± 0.0026387 (W 0.0030605) | 1.23129 ± 0.0105904 (W 0.00506452) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 1.41024 ± 0.00533952 (W 0.00347695) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.753247 ± 0.01654 (W 0.00157881) | 1.26703 ± 0.0114988 (W 0.00514201) | 🔵 faer-fork 1.68× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n513/b1 [in_place]` | 1.41753 ± 0.00668872 (W 0.0042698) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 6.15025 ± 0.00626501 (W 0.0114527) | 4.4486 ± 0.0118206 (W 0.00605377) | 🔴 faer-fork 1.38× slower than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 6.11562 ± 0.0158838 (W 0.0100951) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 6.08755 ± 0.0216574 (W 0.0105765) | 4.43393 ± 0.0176028 (W 0.00673327) | 🔴 faer-fork 1.37× slower than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 6.07961 ± 0.0289817 (W 0.0119753) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 21.4546 ± 0.209063 (W 0.222481) | 15.9989 ± 0.0349021 (W 0.0182811) | 🔴 faer-fork 1.34× slower than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 21.2145 ± 0.0194518 (W 0.0431197) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 21.9808 ± 0.163382 (W 0.190191) | 16.3851 ± 0.0557153 (W 0.0240346) | 🔴 faer-fork 1.34× slower than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 21.6669 ± 0.0586628 (W 0.0480288) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 2.1931 ± 0.0105687 (W 0.00699651) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [in_place]` | 1.14283 ± 0.00511906 (W 0.00443742) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.52488 ± 0.00759628 (W 0.00470488) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [in_place]` | 1.16563 ± 0.00686557 (W 0.00422375) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 11.8281 ± 0.0643271 (W 0.0223379) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 4.67084 ± 0.0306264 (W 0.0109954) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.11193 ± 0.0353667 (W 0.0156032) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 4.8091 ± 0.209486 (W 0.00846651) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n255/b1` | 0.833781 ± 24.6522 (W 0.00613249) | 0.488429 ± 0.0123529 (W 0.00465227) | · | `!` between-start range over 5% |
| `lu/n255/b1 [in_place]` | 0.433378 ± 0.00136202 (W 0.00222945) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.815465 ± 10.3769 (W 0.0030411) | 0.463283 ± 0.00173015 (W 0.00239028) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n257/b1 [in_place]` | 0.458966 ± 0.00223282 (W 0.0026374) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 4.121 ± 0.019094 (W 0.0106653) | 1.8684 ± 0.0224976 (W 0.00423851) | 🔴 faer-fork 2.21× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 1.73486 ± 0.00780503 (W 0.00638335) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.4804 ± 0.036501 (W 0.00785744) | 1.89651 ± 0.00966846 (W 0.00560008) | 🔴 faer-fork 1.84× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 1.82775 ± 0.00724982 (W 0.00592933) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.40212 ± 0.00772238 (W 0.00593272) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.48419 ± 0.00746377 (W 0.00448406) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 6.16608 ± 0.0262641 (W 0.0137007) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 6.64136 ± 0.101024 (W 0.0237755) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.612749 ± 0.00185052 (W 0.00344862) | 1.17113 ± 0.0143046 (W 0.00294614) | 🔵 faer-fork 1.91× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 0.645003 ± 0.00318787 (W 0.00232878) | 1.10381 ± 0.0748315 (W 0.00324074) | 🔵 faer-fork 1.71× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr_factor/n511/b1` | 1.88144 ± 0.0097292 (W 0.00628423) | 3.95445 ± 0.0514485 (W 0.0057611) | 🔵 faer-fork 2.10× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 2.25046 ± 0.00457551 (W 0.00868336) | 4.04052 ± 0.0199239 (W 0.00774526) | 🔵 faer-fork 1.80× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.559203 ± 0.00177983 (W 0.00277414) | 0.454292 ± 0.0146512 (W 0.00294457) | 🔴 faer-fork 1.23× slower than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n255/b1 [in_place]` | 0.424845 ± 0.00196801 (W 0.00262161) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.826302 ± 0.00824842 (W 0.00432055) | 0.460348 ± 0.00396152 (W 0.00245488) | 🔴 faer-fork 1.79× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 0.450771 ± 0.00225621 (W 0.00323459) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.84708 ± 0.0699842 (W 0.0170246) | 1.82242 ± 0.0123092 (W 0.00404387) | 🔴 faer-fork 2.11× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 1.61497 ± 0.00545329 (W 0.00710143) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.42181 ± 0.0249848 (W 0.00925813) | 1.85732 ± 0.0363253 (W 0.00564195) | 🔴 faer-fork 1.84× slower than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n513/b1 [in_place]` | 1.70847 ± 0.00401341 (W 0.00458641) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.8453 ± 0.0181094 (W 0.015301) | 11.6595 ± 0.053282 (W 0.0101266) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `svd_thin/n255/b1 [in_place]` | 10.8228 ± 0.0158483 (W 0.0139553) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.46588 ± 0.0203561 (W 0.0115152) | 11.6988 ± 0.0244558 (W 0.016797) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 8.44098 ± 0.0134098 (W 0.014536) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 39.2495 ± 0.0824325 (W 0.0699344) | 58.5855 ± 0.41533 (W 0.129939) | 🔵 faer-fork 1.49× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [in_place]` | 39.2027 ± 0.0791886 (W 0.0784765) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 35.4332 ± 0.0376936 (W 0.0593723) | 58.9644 ± 0.432836 (W 0.188391) | 🔵 faer-fork 1.66× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 35.2132 ± 0.0810774 (W 0.0397471) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.27042 ± 0.0134666 (W 0.0110001) | 6.12774 ± 0.0235462 (W 0.0108803) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [in_place]` | 5.26324 ± 0.0222871 (W 0.0149577) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.02525 ± 0.0118356 (W 0.0101811) | 6.15356 ± 0.0227444 (W 0.00832838) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [in_place]` | 5.01722 ± 0.0415843 (W 0.00954624) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 22.9635 ± 0.0609226 (W 0.0355783) | 34.7325 ± 0.309733 (W 0.0642342) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 22.9938 ± 0.136262 (W 0.0521522) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 20.1525 ± 0.0583708 (W 0.0282496) | 35.1777 ± 0.187435 (W 0.0378893) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 20.1321 ± 0.0625814 (W 0.0261508) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.013164 ± 0.00016196 (W 0.000117004) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.013114 ± 0.000181459 (W 0.000109812) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.030262 ± 0.000245619 (W 0.000222333) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0301325 ± 0.000381551 (W 0.00020422) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.236012 ± 0.00423916 (W 0.00228563) | 0.316962 ± 0.00220245 (W 0.00264334) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 0.199221 ± 0.00506404 (W 0.00206128) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1` | 0.222817 ± 0.00113522 (W 0.00167944) | 0.257676 ± 0.00185433 (W 0.00333742) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [in_place]` | 0.189838 ± 0.00123488 (W 0.00202173) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.952743 ± 0.00441569 (W 0.00347237) | 1.1937 ± 0.00297153 (W 0.00478925) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 0.769432 ± 0.00249465 (W 0.00336773) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.893721 ± 0.00590233 (W 0.00298939) | 1.18633 ± 0.00198119 (W 0.00391968) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [in_place]` | 0.767775 ± 0.00478153 (W 0.00361888) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 6.39658 ± 0.018503 (W 0.0112821) | 4.74972 ± 0.0166683 (W 0.00832494) | 🔴 faer-fork 1.35× slower than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 6.38152 ± 0.0293647 (W 0.0073797) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 6.38349 ± 0.0226444 (W 0.00784062) | 4.7765 ± 0.0148919 (W 0.0066847) | 🔴 faer-fork 1.34× slower than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 6.35817 ± 0.0100053 (W 0.00664558) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 23.8346 ± 0.0596531 (W 0.0339351) | 19.104 ± 0.0780681 (W 0.025742) | 🔴 faer-fork 1.25× slower than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 23.5782 ± 0.0482154 (W 0.0190727) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 24.0911 ± 0.0471685 (W 0.0263944) | 19.5246 ± 0.041191 (W 0.0208096) | 🔴 faer-fork 1.23× slower than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 24.1901 ± 0.0707349 (W 0.0171991) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 2.16323 ± 0.00928522 (W 0.00884105) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [in_place]` | 1.15758 ± 0.00465941 (W 0.00340575) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.43469 ± 0.00619879 (W 0.0061837) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [in_place]` | 1.16154 ± 0.00618008 (W 0.00403234) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 12.5341 ± 0.0836642 (W 0.0196625) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 5.94293 ± 0.0245867 (W 0.0103584) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.79335 ± 0.0217337 (W 0.00997824) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 6.00022 ± 0.0388971 (W 0.0110494) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.74834 ± 0.00261737 (W 0.00505567) | 0.435891 ± 0.00201086 (W 0.00259919) | 🔴 faer-fork 1.72× slower than `lapack-openblas` |  |
| `lu/n255/b1 [in_place]` | 0.396135 ± 0.00538132 (W 0.00384957) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.731416 ± 0.00459373 (W 0.00346753) | 0.445364 ± 0.00130556 (W 0.00273472) | 🔴 faer-fork 1.64× slower than `lapack-openblas` |  |
| `lu/n257/b1 [in_place]` | 0.409593 ± 0.00304473 (W 0.00394676) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.8838 ± 0.0115491 (W 0.00907716) | 2.38688 ± 0.0104757 (W 0.0066849) | 🔴 faer-fork 1.63× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 1.83995 ± 0.00633201 (W 0.00594517) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.26649 ± 0.0194155 (W 0.00943653) | 2.42294 ± 0.0132264 (W 0.00487744) | 🔴 faer-fork 1.35× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 1.92011 ± 0.0065974 (W 0.0061414) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.46303 ± 0.00645237 (W 0.00871249) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.54113 ± 0.0091998 (W 0.00795395) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 7.79056 ± 0.045362 (W 0.0162709) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 8.02134 ± 0.020675 (W 0.0150115) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.578759 ± 0.00268916 (W 0.00338226) | 1.11208 ± 0.012526 (W 0.00294802) | 🔵 faer-fork 1.92× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 0.596971 ± 0.00155281 (W 0.00264977) | 1.0323 ± 0.004631 (W 0.0031121) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 2.91678 ± 0.01014 (W 0.00424753) | 4.43528 ± 0.028434 (W 0.00737269) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 2.88296 ± 0.00699039 (W 0.00443509) | 4.51348 ± 0.0113113 (W 0.0081494) | 🔵 faer-fork 1.57× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.477087 ± 0.00204913 (W 0.00322209) | 0.420573 ± 0.00134188 (W 0.00262848) | 🔴 faer-fork 1.13× slower than `lapack-openblas` |  |
| `solve/n255/b1 [in_place]` | 0.387177 ± 0.00245107 (W 0.00203356) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.729731 ± 0.00435608 (W 0.0032101) | 0.427277 ± 0.0032299 (W 0.00258586) | 🔴 faer-fork 1.71× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 0.399355 ± 0.00237625 (W 0.00323641) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.5865 ± 0.0162185 (W 0.00944558) | 2.32839 ± 0.00606656 (W 0.00622739) | 🔴 faer-fork 1.54× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 1.73867 ± 0.00511479 (W 0.0048632) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.18352 ± 0.0173041 (W 0.00966418) | 2.35625 ± 0.00655349 (W 0.00523242) | 🔴 faer-fork 1.35× slower than `lapack-openblas` |  |
| `solve/n513/b1 [in_place]` | 1.8211 ± 0.00256812 (W 0.00412977) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 11.3963 ± 0.0320282 (W 0.0152605) | 10.7791 ± 0.0316345 (W 0.0129069) | 🔴 faer-fork 1.06× slower than `lapack-openblas` |  |
| `svd_thin/n255/b1 [in_place]` | 11.3597 ± 0.0236372 (W 0.0148679) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 9.02186 ± 0.0298522 (W 0.0143063) | 10.9401 ± 0.0280526 (W 0.022897) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 9.02565 ± 0.016221 (W 0.0188756) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 44.8053 ± 0.0482099 (W 0.0269386) | 57.099 ± 0.364233 (W 0.071143) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [in_place]` | 44.6351 ± 0.0618438 (W 0.0424736) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 41.0878 ± 0.0998737 (W 0.0319412) | 57.4163 ± 0.406658 (W 0.0574478) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 41.0707 ± 0.0630359 (W 0.0336124) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.31739 ± 0.0252416 (W 0.0121333) | 6.03949 ± 0.0178121 (W 0.00735045) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [in_place]` | 5.3062 ± 0.0275472 (W 0.0081816) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.17575 ± 0.0257975 (W 0.0113881) | 6.10395 ± 0.00611085 (W 0.011499) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [in_place]` | 5.17008 ± 0.0172497 (W 0.0111912) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 24.9575 ± 0.0509333 (W 0.0376744) | 35.4256 ± 0.352596 (W 0.0427447) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 24.7499 ± 0.0912548 (W 0.0432599) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 22.1022 ± 0.0655085 (W 0.0221342) | 35.5008 ± 0.357448 (W 0.0343626) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 22.1321 ± 0.0687073 (W 0.0206521) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.015329 ± 0.000138425 (W 0.00031084) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.015815 ± 0.000231677 (W 0.0003567) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.0401605 ± 0.000397622 (W 0.00051689) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0395035 ± 0.000287881 (W 0.00056294) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.229593 ± 0.00150168 (W 0.00169013) | 0.31995 ± 0.00208459 (W 0.00289948) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 0.193732 ± 0.00455954 (W 0.00152409) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1` | 0.206132 ± 0.00461065 (W 0.00158407) | 0.259016 ± 0.00170983 (W 0.0019484) | 🔵 faer-fork 1.26× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n257/b1 [in_place]` | 0.172127 ± 0.00394574 (W 0.00164693) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.818747 ± 0.00435881 (W 0.00287593) | 1.04706 ± 0.00318043 (W 0.00283977) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 0.620176 ± 0.003657 (W 0.00396887) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.750877 ± 0.0162788 (W 0.00333305) | 1.03725 ± 0.00170538 (W 0.00300697) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [in_place]` | 0.620167 ± 0.00788668 (W 0.00405851) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 6.15554 ± 0.0162997 (W 0.0127552) | 4.38068 ± 0.0108619 (W 0.00747956) | 🔴 faer-fork 1.41× slower than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 6.12277 ± 0.00916177 (W 0.0133291) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 6.08426 ± 0.0201216 (W 0.0142948) | 4.36438 ± 0.012688 (W 0.00875462) | 🔴 faer-fork 1.39× slower than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 6.05956 ± 0.0140809 (W 0.0111991) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 21.2766 ± 0.181128 (W 0.22989) | 15.5206 ± 0.0415971 (W 0.0474747) | 🔴 faer-fork 1.37× slower than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 21.1177 ± 0.0485426 (W 0.0183607) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 21.6626 ± 0.144531 (W 0.161905) | 15.9194 ± 0.0369615 (W 0.022866) | 🔴 faer-fork 1.36× slower than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 21.6155 ± 0.0354545 (W 0.0156297) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 2.12266 ± 0.0106713 (W 0.0108468) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [in_place]` | 1.10279 ± 0.00603709 (W 0.0047971) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.40798 ± 0.0187761 (W 0.00629053) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [in_place]` | 1.10936 ± 0.00909157 (W 0.00429493) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 11.3119 ± 0.0809689 (W 0.0172974) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 4.57682 ± 0.0256698 (W 0.0107126) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 5.59851 ± 0.045222 (W 0.0135163) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 4.62333 ± 0.0191256 (W 0.00892461) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.73595 ± 0.00380072 (W 0.00342438) | 0.374038 ± 0.00215458 (W 0.00240288) | 🔴 faer-fork 1.97× slower than `lapack-openblas` |  |
| `lu/n255/b1 [in_place]` | 0.379381 ± 0.00275438 (W 0.00220267) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.713394 ± 0.00487494 (W 0.00289604) | 0.379636 ± 0.00271297 (W 0.00265119) | 🔴 faer-fork 1.88× slower than `lapack-openblas` |  |
| `lu/n257/b1 [in_place]` | 0.389183 ± 0.00366658 (W 0.00394077) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.61092 ± 0.0140059 (W 0.00944511) | 1.49967 ± 0.006091 (W 0.00397164) | 🔴 faer-fork 2.41× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 1.5636 ± 0.0179506 (W 0.00515533) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 2.9839 ± 0.0119512 (W 0.00983087) | 1.51147 ± 0.00607426 (W 0.00538398) | 🔴 faer-fork 1.97× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 1.63001 ± 0.00476439 (W 0.00589935) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.29017 ± 0.00587854 (W 0.0063251) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.35437 ± 0.00763547 (W 0.00582209) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.5506 ± 0.032706 (W 0.0195771) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 5.99403 ± 0.041035 (W 0.0185292) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.565605 ± 0.00276261 (W 0.00213619) | 1.09254 ± 0.013737 (W 0.00309541) | 🔵 faer-fork 1.93× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 0.580869 ± 0.00150562 (W 0.00221617) | 1.01853 ± 0.00338509 (W 0.00231761) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 1.74962 ± 0.0373055 (W 0.00642555) | 3.65567 ± 0.00424396 (W 0.00356495) | 🔵 faer-fork 2.09× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr_factor/n513/b1` | 2.04367 ± 0.00705943 (W 0.00565462) | 3.66964 ± 0.0104553 (W 0.00358146) | 🔵 faer-fork 1.80× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.474535 ± 0.00281127 (W 0.00219313) | 0.376648 ± 0.001259 (W 0.0033207) | 🔴 faer-fork 1.26× slower than `lapack-openblas` |  |
| `solve/n255/b1 [in_place]` | 0.375457 ± 0.00155228 (W 0.00245911) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.716153 ± 0.00549396 (W 0.00509555) | 0.379702 ± 0.00288913 (W 0.00267955) | 🔴 faer-fork 1.89× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 0.381348 ± 0.00478668 (W 0.00267307) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.35219 ± 0.019301 (W 0.0105482) | 1.46349 ± 0.00714064 (W 0.0051429) | 🔴 faer-fork 2.29× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 1.457 ± 0.0161806 (W 0.00374899) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 2.92619 ± 0.0294373 (W 0.00844059) | 1.47819 ± 0.0142982 (W 0.0066939) | 🔴 faer-fork 1.98× slower than `lapack-openblas` |  |
| `solve/n513/b1 [in_place]` | 1.52796 ± 0.0044146 (W 0.00331217) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.8884 ± 0.0188486 (W 0.0153849) | 10.6754 ± 0.0292799 (W 0.0149228) | · |  |
| `svd_thin/n255/b1 [in_place]` | 10.8641 ± 0.026254 (W 0.0171375) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.4644 ± 0.140393 (W 0.0129328) | 10.6531 ± 0.0151862 (W 0.0124452) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 8.44316 ± 0.0194226 (W 0.0112996) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 39.2036 ± 0.068162 (W 0.145316) | 52.6672 ± 0.35181 (W 0.0532308) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [in_place]` | 39.0553 ± 0.0265801 (W 0.0336287) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 35.1086 ± 0.0377554 (W 0.0264758) | 52.9155 ± 0.282768 (W 0.0648715) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 34.9794 ± 0.0338305 (W 0.0816428) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.25755 ± 0.0136185 (W 0.0104571) | 6.04854 ± 0.0218523 (W 0.00721326) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [in_place]` | 5.24737 ± 0.0130237 (W 0.0102753) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.04821 ± 0.00885369 (W 0.00981876) | 6.07321 ± 0.0130656 (W 0.0048734) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [in_place]` | 5.03653 ± 0.012148 (W 0.00721781) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 22.8692 ± 0.0877536 (W 0.0350427) | 34.9746 ± 0.239553 (W 0.0298427) | 🔵 faer-fork 1.53× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 22.8248 ± 0.0753505 (W 0.0148727) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 20.0778 ± 0.0479941 (W 0.0775014) | 34.9762 ± 0.289065 (W 0.0216003) | 🔵 faer-fork 1.74× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 20.0077 ± 0.02512 (W 0.0334665) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.015509 ± 0.000137465 (W 0.000236808) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.015574 ± 8.14872e-05 (W 0.000183724) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.038407 ± 0.000381192 (W 0.000335826) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.037981 ± 0.000452004 (W 0.000328555) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.672799 ± 0.00649887 (W 0.00284) | 0.608335 ± 0.000766546 (W 0.00220159) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 1.55651 ± 0.00591143 (W 0.00381619) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.617333 ± 0.00467123 (W 0.0039025) | 0.596502 ± 0.00280999 (W 0.00277304) | · |  |
| `cholesky/n257/b1 [in_place]` | 1.56649 ± 0.00303218 (W 0.00556615) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.88502 ± 0.013683 (W 0.00995706) | 3.5689 ± 0.0285443 (W 0.012524) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 4.69495 ± 0.011973 (W 0.00695509) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.8053 ± 0.0097627 (W 0.00541299) | 3.57361 ± 0.0267459 (W 0.00600576) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [in_place]` | 4.71285 ± 0.0148091 (W 0.00820941) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.72882 ± 0.022185 (W 0.0117693) | 11.9481 ± 0.0349085 (W 0.0273114) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 8.69402 ± 0.0180245 (W 0.0127615) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.66725 ± 0.0106003 (W 0.0140745) | 12.2187 ± 0.0532863 (W 0.0231638) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 8.68117 ± 0.0253755 (W 0.0121659) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 39.4558 ± 0.354895 (W 0.476364) | 56.679 ± 0.0852307 (W 0.0639837) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 39.2174 ± 0.0498055 (W 0.0676235) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 39.9261 ± 0.423468 (W 0.443154) | 57.2177 ± 0.128278 (W 0.0670077) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 39.8643 ± 0.0438275 (W 0.0340924) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 4.05497 ± 0.0275848 (W 0.01239) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [in_place]` | 1.9667 ± 0.0128565 (W 0.00468005) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.6946 ± 0.00690356 (W 0.00539158) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [in_place]` | 1.99896 ± 0.0113101 (W 0.00610033) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 20.5882 ± 0.0784803 (W 0.0390426) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 11.1054 ± 0.0808045 (W 0.029513) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 15.1168 ± 0.0716911 (W 0.0452474) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 11.2426 ± 0.0574874 (W 0.0290586) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 1.9433 ± 0.00653723 (W 0.00356165) | 1.27723 ± 0.00232735 (W 0.00288913) | 🔴 faer-fork 1.52× slower than `lapack-openblas` |  |
| `lu/n255/b1 [in_place]` | 1.11474 ± 0.00474897 (W 0.00371369) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.84992 ± 0.00752923 (W 0.00640161) | 1.29159 ± 0.00313521 (W 0.00451844) | 🔴 faer-fork 1.43× slower than `lapack-openblas` |  |
| `lu/n257/b1 [in_place]` | 1.13875 ± 0.00636059 (W 0.00306925) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 9.78469 ± 0.0214663 (W 0.0314599) | 5.99162 ± 0.0164655 (W 0.0126148) | 🔴 faer-fork 1.63× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 5.55356 ± 0.0114222 (W 0.00891264) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 9.70231 ± 0.0591859 (W 0.017946) | 6.06721 ± 0.0136907 (W 0.0122602) | 🔴 faer-fork 1.60× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 5.74009 ± 0.0149175 (W 0.0115363) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 4.21741 ± 0.0172636 (W 0.00714058) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 4.31912 ± 0.00599761 (W 0.00924805) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 25.5643 ± 0.111943 (W 0.0451977) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 26.1529 ± 0.0856668 (W 0.127993) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.62185 ± 0.00220917 (W 0.00399841) | 2.57919 ± 0.00672551 (W 0.00354549) | 🔵 faer-fork 1.59× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 1.69123 ± 0.00463602 (W 0.00408378) | 2.80792 ± 0.00937676 (W 0.00459166) | 🔵 faer-fork 1.66× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 9.43803 ± 0.0425566 (W 0.0141741) | 13.1878 ± 0.0269042 (W 0.0180842) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 9.65741 ± 0.0190791 (W 0.0145206) | 13.9646 ± 0.0417172 (W 0.0165553) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 1.87707 ± 0.00891818 (W 0.00701452) | 1.24774 ± 0.00924962 (W 0.00222189) | 🔴 faer-fork 1.50× slower than `lapack-openblas` |  |
| `solve/n255/b1 [in_place]` | 1.11706 ± 0.00504513 (W 0.00431081) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.86824 ± 0.0109268 (W 0.00567894) | 1.26199 ± 0.00590211 (W 0.00264339) | 🔴 faer-fork 1.48× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 1.14207 ± 0.00391173 (W 0.00338294) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 9.72012 ± 0.0255765 (W 0.0305186) | 5.92804 ± 0.0328765 (W 0.0207616) | 🔴 faer-fork 1.64× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 5.55125 ± 0.0116937 (W 0.0177358) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 9.86737 ± 0.112883 (W 0.0311549) | 6.01638 ± 0.0142101 (W 0.0192911) | 🔴 faer-fork 1.64× slower than `lapack-openblas` |  |
| `solve/n513/b1 [in_place]` | 5.72154 ± 0.0153513 (W 0.00894351) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 16.4113 ± 0.0475051 (W 0.0324279) | 17.8038 ± 0.0744088 (W 0.0279116) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `svd_thin/n255/b1 [in_place]` | 16.3494 ± 0.0469006 (W 0.018614) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 13.9493 ± 0.0409578 (W 0.0214634) | 18.7478 ± 0.135341 (W 0.042696) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 13.8814 ± 0.0346714 (W 0.0244794) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 77.1985 ± 0.121454 (W 0.131696) | 89.193 ± 0.277477 (W 0.101891) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [in_place]` | 76.8856 ± 0.109432 (W 0.219405) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 74.3991 ± 0.127902 (W 0.0925427) | 92.2664 ± 0.223745 (W 0.104898) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 74.0992 ± 0.162364 (W 0.0935793) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 7.98995 ± 0.0864578 (W 0.0100449) | 7.86168 ± 0.0156453 (W 0.00999813) | · |  |
| `svd_values/n255/b1 [in_place]` | 7.83132 ± 0.0151446 (W 0.0183597) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 7.57372 ± 0.0899825 (W 0.0134102) | 8.18834 ± 0.0203053 (W 0.0117427) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [in_place]` | 7.52956 ± 0.0102563 (W 0.0183953) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 41.2473 ± 0.226568 (W 0.060197) | 36.1647 ± 0.0963882 (W 0.0378405) | 🔴 faer-fork 1.14× slower than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 41.1845 ± 0.160253 (W 0.0574444) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 39.3683 ± 0.282128 (W 0.0732478) | 37.1982 ± 0.0438649 (W 0.0805255) | 🔴 faer-fork 1.06× slower than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 39.3232 ± 0.0998289 (W 0.0904311) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.021972 ± 0.000240165 (W 0.000146919) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.0223665 ± 0.000107303 (W 0.00014078) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.056295 ± 0.000618445 (W 0.000320461) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0559445 ± 0.000521654 (W 0.000246906) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.606081 ± 0.00283213 (W 0.00286431) | 0.868453 ± 0.00328012 (W 0.00178597) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 1.49818 ± 0.00230647 (W 0.00389111) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.546478 ± 0.00200795 (W 0.00252114) | 0.538339 ± 0.0018241 (W 0.00314507) | · |  |
| `cholesky/n257/b1 [in_place]` | 1.50025 ± 0.00285451 (W 0.00277289) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.309 ± 0.0176346 (W 0.00810643) | 2.73548 ± 0.00922504 (W 0.0100756) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 4.05826 ± 0.00792359 (W 0.00514051) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.1737 ± 0.0151675 (W 0.0103273) | 2.41291 ± 0.0158113 (W 0.0145388) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [in_place]` | 4.05602 ± 0.00650902 (W 0.00484146) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 7.68789 ± 0.138901 (W 0.00773564) | 10.7078 ± 0.0179996 (W 0.0142529) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 7.65875 ± 0.0102687 (W 0.00518355) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 7.68496 ± 0.0162611 (W 0.0102746) | 10.9758 ± 0.0664137 (W 0.0206467) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 7.6779 ± 0.0192415 (W 0.00696564) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 30.2923 ± 0.0625913 (W 0.0475347) | 43.9268 ± 0.110423 (W 0.115189) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 30.1609 ± 0.0399901 (W 0.0298789) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 30.6893 ± 0.0428581 (W 0.360743) | 44.1525 ± 0.0562397 (W 0.0617838) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 30.6039 ± 0.0667631 (W 0.0237478) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 3.72563 ± 0.0255052 (W 0.0112297) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [in_place]` | 1.59343 ± 0.00924297 (W 0.00521773) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.36858 ± 0.0183681 (W 0.00767815) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [in_place]` | 1.61333 ± 0.0116512 (W 0.00477194) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 17.0078 ± 0.0687218 (W 0.0376467) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 7.25651 ± 0.0408125 (W 0.0251897) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 11.4122 ± 0.0559142 (W 0.0277732) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 7.37281 ± 0.0335481 (W 0.0232) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 1.83106 ± 0.0122923 (W 0.00741804) | 0.904756 ± 0.0048219 (W 0.00197257) | 🔴 faer-fork 2.02× slower than `lapack-openblas` |  |
| `lu/n255/b1 [in_place]` | 0.973631 ± 0.00295581 (W 0.00283998) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.73237 ± 0.00871442 (W 0.00535744) | 0.913604 ± 0.00498755 (W 0.00270966) | 🔴 faer-fork 1.90× slower than `lapack-openblas` |  |
| `lu/n257/b1 [in_place]` | 0.974527 ± 0.00186948 (W 0.00151944) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 8.39756 ± 0.0362417 (W 0.0144444) | 3.67841 ± 0.00593261 (W 0.00993122) | 🔴 faer-fork 2.28× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 3.96989 ± 0.007245 (W 0.00517739) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 8.30335 ± 0.036812 (W 0.0156961) | 3.74254 ± 0.0393918 (W 0.0184926) | 🔴 faer-fork 2.22× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 4.08622 ± 0.00737211 (W 0.00758917) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 3.25792 ± 0.0163518 (W 0.0100627) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.34687 ± 0.0114425 (W 0.00940907) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 16.608 ± 0.0963853 (W 0.107751) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 17.4888 ± 0.0747904 (W 0.0381395) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.30344 ± 0.00802839 (W 0.00740336) | 2.30523 ± 0.0134211 (W 0.00435915) | 🔵 faer-fork 1.77× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 1.3564 ± 0.00420405 (W 0.00416777) | 2.45553 ± 0.0143837 (W 0.00703642) | 🔵 faer-fork 1.81× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 5.05385 ± 0.01288 (W 0.0105825) | 9.89634 ± 0.185055 (W 0.0261527) | 🔵 faer-fork 1.96× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr_factor/n513/b1` | 5.71891 ± 0.0318688 (W 0.00986136) | 10.386 ± 0.0424417 (W 0.016283) | 🔵 faer-fork 1.82× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 1.7696 ± 0.00956862 (W 0.0048415) | 0.889774 ± 0.0134941 (W 0.00216772) | 🔴 faer-fork 1.99× slower than `lapack-openblas` |  |
| `solve/n255/b1 [in_place]` | 0.970272 ± 0.00165059 (W 0.00209785) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.73618 ± 0.00726725 (W 0.0043798) | 0.900172 ± 0.00651392 (W 0.00478113) | 🔴 faer-fork 1.93× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 0.970449 ± 0.0030178 (W 0.00153885) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 8.30859 ± 0.0729986 (W 0.0500168) | 3.66872 ± 0.0526458 (W 0.0165849) | 🔴 faer-fork 2.26× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 3.93476 ± 0.0105333 (W 0.00855993) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 8.34341 ± 0.0440015 (W 0.0170299) | 3.7422 ± 0.0161312 (W 0.0191283) | 🔴 faer-fork 2.23× slower than `lapack-openblas` |  |
| `solve/n513/b1 [in_place]` | 4.03947 ± 0.0116087 (W 0.00823261) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 14.5767 ± 0.0879917 (W 0.026167) | 15.6233 ± 0.0826624 (W 0.235953) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `svd_thin/n255/b1 [in_place]` | 14.4397 ± 0.0528811 (W 0.0169732) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 11.898 ± 0.0835043 (W 0.0173678) | 16.3845 ± 0.0358589 (W 0.0273621) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 11.7641 ± 0.0202528 (W 0.0195739) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 58.2228 ± 0.268019 (W 0.221704) | 68.8053 ± 0.20683 (W 0.134749) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [in_place]` | 58.0072 ± 0.180624 (W 0.0560466) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 54.9897 ± 0.181519 (W 0.0600438) | 71.2188 ± 0.207141 (W 0.124803) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 54.6067 ± 0.371937 (W 0.147713) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 7.37945 ± 0.0207603 (W 0.0102056) | 7.2457 ± 0.0144792 (W 0.0080123) | · |  |
| `svd_values/n255/b1 [in_place]` | 7.3591 ± 0.0281914 (W 0.01308) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 7.04863 ± 0.0999483 (W 0.0155366) | 7.5027 ± 0.0471519 (W 0.00924039) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [in_place]` | 6.98053 ± 0.0097382 (W 0.00711791) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 33.4275 ± 0.205484 (W 0.0750468) | 28.567 ± 0.106907 (W 0.0415184) | 🔴 faer-fork 1.17× slower than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 33.3535 ± 0.0646664 (W 0.0687868) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 31.287 ± 0.0636645 (W 0.0491973) | 29.2398 ± 0.0754007 (W 0.0453294) | 🔴 faer-fork 1.07× slower than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 31.0521 ± 0.0603955 (W 0.0455966) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0216305 ± 4.2192e-05 (W 0.000121924) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.021886 ± 3.91714e-05 (W 0.00012933) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.0505995 ± 0.000755188 (W 0.000324513) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.050645 ± 0.000936323 (W 0.000215808) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.663919 ± 0.00358193 (W 0.00172913) | 0.551412 ± 0.00161384 (W 0.00251909) | 🔴 faer-fork 1.20× slower than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 0.572666 ± 0.00289963 (W 0.00360791) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.619799 ± 0.00225593 (W 0.00330988) | 0.538018 ± 0.00173662 (W 0.00158461) | 🔴 faer-fork 1.15× slower than `lapack-openblas` |  |
| `cholesky/n257/b1 [in_place]` | 0.543467 ± 0.00297257 (W 0.00342518) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.82651 ± 0.0157712 (W 0.00375413) | 3.19064 ± 0.00366968 (W 0.00384542) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 2.43566 ± 0.00528149 (W 0.00380304) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.74407 ± 0.00459996 (W 0.00597255) | 3.19641 ± 0.00327431 (W 0.0055349) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [in_place]` | 2.40381 ± 0.00298225 (W 0.00474858) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.72964 ± 0.0146782 (W 0.0158345) | 11.6981 ± 0.032394 (W 0.0172945) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 8.68184 ± 0.0266569 (W 0.0147784) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.66628 ± 0.013926 (W 0.01543) | 12.0172 ± 0.0451382 (W 0.0166482) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 8.64706 ± 0.00909245 (W 0.00867881) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 39.427 ± 0.445177 (W 0.485658) | 54.454 ± 0.137776 (W 0.0916553) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 39.1384 ± 0.0305547 (W 0.0374606) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.6126 ± 0.403319 (W 0.423894) | 55.1151 ± 0.104844 (W 0.0838144) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 39.8236 ± 0.0389007 (W 0.0344645) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 3.88493 ± 0.0202619 (W 0.00937541) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [in_place]` | 1.85221 ± 0.011673 (W 0.00682148) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.53154 ± 0.0145446 (W 0.00801804) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [in_place]` | 1.88869 ± 0.0095744 (W 0.00431084) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 19.6093 ± 0.0823598 (W 0.0247508) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 10.6652 ± 0.0303275 (W 0.0285995) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 13.9183 ± 0.0495343 (W 0.0402206) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 10.7444 ± 0.0435686 (W 0.0175277) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 1.78789 ± 0.00719189 (W 0.0052969) | 1.14275 ± 0.00241872 (W 0.00314891) | 🔴 faer-fork 1.56× slower than `lapack-openblas` |  |
| `lu/n255/b1 [in_place]` | 1.01714 ± 0.00505272 (W 0.00289778) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.69201 ± 0.00854806 (W 0.00524347) | 1.15851 ± 0.0030406 (W 0.00296942) | 🔴 faer-fork 1.46× slower than `lapack-openblas` |  |
| `lu/n257/b1 [in_place]` | 1.0192 ± 0.0038175 (W 0.00427392) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 8.79183 ± 0.0386178 (W 0.0153011) | 5.03593 ± 0.00704142 (W 0.00442591) | 🔴 faer-fork 1.75× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 5.08337 ± 0.0117462 (W 0.00599243) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 8.7046 ± 0.0523623 (W 0.0261717) | 5.0854 ± 0.00479093 (W 0.00807342) | 🔴 faer-fork 1.71× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 5.17611 ± 0.0176888 (W 0.00786449) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 4.04099 ± 0.0162549 (W 0.00885598) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 4.15657 ± 0.0111512 (W 0.00639428) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 24.5645 ± 0.0554422 (W 0.149523) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 25.099 ± 0.0416046 (W 0.0252305) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.52439 ± 0.0018265 (W 0.0028498) | 2.46703 ± 0.00783893 (W 0.00445266) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 1.56871 ± 0.00288389 (W 0.00375059) | 2.68941 ± 0.00985343 (W 0.00644472) | 🔵 faer-fork 1.71× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 8.94201 ± 0.00793757 (W 0.00781828) | 12.2079 ± 0.0268949 (W 0.0131928) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 9.03587 ± 0.0127795 (W 0.0139747) | 12.9526 ± 0.0178415 (W 0.00919548) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 1.71993 ± 0.0116829 (W 0.00524371) | 1.1197 ± 0.0028292 (W 0.00270985) | 🔴 faer-fork 1.54× slower than `lapack-openblas` |  |
| `solve/n255/b1 [in_place]` | 1.01822 ± 0.00289698 (W 0.00347124) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.70684 ± 0.0147279 (W 0.00783419) | 1.14273 ± 0.00503149 (W 0.0032492) | 🔴 faer-fork 1.49× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 1.02472 ± 0.00232606 (W 0.00375624) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 8.73583 ± 0.0686691 (W 0.0162407) | 5.00197 ± 0.015957 (W 0.00753142) | 🔴 faer-fork 1.75× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 5.08333 ± 0.014846 (W 0.00624205) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 8.79179 ± 0.0238377 (W 0.0225805) | 5.04875 ± 0.00700192 (W 0.00713099) | 🔴 faer-fork 1.74× slower than `lapack-openblas` |  |
| `solve/n513/b1 [in_place]` | 5.17147 ± 0.00643623 (W 0.00736739) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 16.416 ± 0.11919 (W 0.0216346) | 15.5854 ± 0.0373008 (W 0.0136716) | 🔴 faer-fork 1.05× slower than `lapack-openblas` |  |
| `svd_thin/n255/b1 [in_place]` | 16.3558 ± 0.0244886 (W 0.0167224) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 13.8568 ± 0.0618338 (W 0.0226763) | 16.3746 ± 0.0333047 (W 0.0152999) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 13.8052 ± 0.0101702 (W 0.0168375) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 76.9182 ± 0.188497 (W 0.0686173) | 78.0004 ± 0.178986 (W 0.0867269) | · |  |
| `svd_thin/n511/b1 [in_place]` | 76.6917 ± 0.0820843 (W 0.361237) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 74.1044 ± 0.258874 (W 0.1032) | 81.0831 ± 0.176587 (W 0.0725152) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 73.9045 ± 0.16292 (W 0.0586525) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 7.91012 ± 0.0236381 (W 0.0193099) | 7.75418 ± 0.0195957 (W 0.00716809) | · |  |
| `svd_values/n255/b1 [in_place]` | 7.88351 ± 0.0178307 (W 0.0150741) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 7.5962 ± 0.0243273 (W 0.0172363) | 8.07911 ± 0.0243718 (W 0.0124947) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [in_place]` | 7.57231 ± 0.0304098 (W 0.0123498) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 41.0725 ± 0.360863 (W 0.0550102) | 35.1019 ± 0.115713 (W 0.0457027) | 🔴 faer-fork 1.17× slower than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 40.8661 ± 0.144219 (W 0.0668123) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 39.1351 ± 0.225085 (W 0.0648918) | 36.1096 ± 0.107944 (W 0.0288401) | 🔴 faer-fork 1.08× slower than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 39.0463 ± 0.0791245 (W 0.0561667) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0226175 ± 0.000239524 (W 0.000253915) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.0224075 ± 0.000157633 (W 0.000233939) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.058021 ± 0.00109188 (W 0.00030809) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.057998 ± 0.000693493 (W 0.000511813) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.602082 ± 0.0053981 (W 0.00353754) | 0.814296 ± 0.00203824 (W 0.00244816) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [in_place]` | 0.508916 ± 0.00197709 (W 0.0024974) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.546578 ± 0.00132951 (W 0.00288876) | 0.478332 ± 0.00299635 (W 0.00248216) | 🔴 faer-fork 1.14× slower than `lapack-openblas` |  |
| `cholesky/n257/b1 [in_place]` | 0.474 ± 0.00220981 (W 0.00243066) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.25005 ± 0.015166 (W 0.00749587) | 2.36913 ± 0.0074492 (W 0.00945283) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [in_place]` | 1.82234 ± 0.00625736 (W 0.0038812) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.13541 ± 0.0119913 (W 0.00613355) | 2.03168 ± 0.0111732 (W 0.00695196) | · |  |
| `cholesky/n513/b1 [in_place]` | 1.7539 ± 0.00815981 (W 0.00817493) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 7.67265 ± 0.111268 (W 0.00958106) | 10.5616 ± 0.0402033 (W 0.0196616) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `eigh/n255/b1 [in_place]` | 7.66023 ± 0.0109528 (W 0.00989212) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 7.68716 ± 0.0123745 (W 0.0086529) | 10.8015 ± 0.0170048 (W 0.0149228) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `eigh/n257/b1 [in_place]` | 7.66943 ± 0.0148025 (W 0.00543506) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 30.2794 ± 0.354742 (W 0.420534) | 42.4974 ± 0.19566 (W 0.0885152) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `eigh/n511/b1 [in_place]` | 30.1487 ± 0.0568362 (W 0.0267327) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 30.6553 ± 0.412829 (W 0.410918) | 42.6873 ± 0.0935327 (W 0.108492) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `eigh/n513/b1 [in_place]` | 30.5821 ± 0.0369218 (W 0.131731) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 3.5348 ± 0.0772965 (W 0.0118343) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n255/b1 [in_place]` | 1.50018 ± 0.0121014 (W 0.0056) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.20208 ± 0.111956 (W 0.0102608) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n257/b1 [in_place]` | 1.5349 ± 0.00966332 (W 0.00458915) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 16.041 ± 0.0769343 (W 0.0326986) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [in_place]` | 7.01384 ± 0.0193615 (W 0.0277804) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 10.3228 ± 0.0696816 (W 0.0231682) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [in_place]` | 7.08813 ± 0.0474477 (W 0.0162485) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 1.67321 ± 0.00727715 (W 0.00566734) | 0.766956 ± 0.00386984 (W 0.00262463) | 🔴 faer-fork 2.18× slower than `lapack-openblas` |  |
| `lu/n255/b1 [in_place]` | 0.902603 ± 0.00242118 (W 0.00237307) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.56099 ± 0.00587382 (W 0.00470802) | 0.774063 ± 0.00260886 (W 0.00317532) | 🔴 faer-fork 2.02× slower than `lapack-openblas` |  |
| `lu/n257/b1 [in_place]` | 0.885471 ± 0.00248019 (W 0.00209105) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 7.45158 ± 0.0837889 (W 0.0162734) | 2.76462 ± 0.00660338 (W 0.00651516) | 🔴 faer-fork 2.70× slower than `lapack-openblas` |  |
| `lu/n511/b1 [in_place]` | 3.71131 ± 0.00533462 (W 0.00419681) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 7.29396 ± 0.0842377 (W 0.0217393) | 2.78935 ± 0.00624662 (W 0.0037088) | 🔴 faer-fork 2.61× slower than `lapack-openblas` |  |
| `lu/n513/b1 [in_place]` | 3.78566 ± 0.0049549 (W 0.0056295) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 3.10181 ± 0.0452215 (W 0.00732765) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.1944 ± 0.0101407 (W 0.0105676) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 15.5544 ± 0.0818495 (W 0.0282832) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 16.4692 ± 0.0671475 (W 0.204747) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.24203 ± 0.00493811 (W 0.00656966) | 2.18667 ± 0.00383408 (W 0.00337624) | 🔵 faer-fork 1.76× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 1.27076 ± 0.00516057 (W 0.00387873) | 2.32967 ± 0.0114147 (W 0.00385353) | 🔵 faer-fork 1.83× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 4.73585 ± 0.0043689 (W 0.0093982) | 8.93278 ± 0.0275479 (W 0.017302) | 🔵 faer-fork 1.89× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 5.29923 ± 0.0102941 (W 0.00866905) | 9.36127 ± 0.0351831 (W 0.0164694) | 🔵 faer-fork 1.77× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 1.60807 ± 0.00919741 (W 0.00540917) | 0.759331 ± 0.00378711 (W 0.00425915) | 🔴 faer-fork 2.12× slower than `lapack-openblas` |  |
| `solve/n255/b1 [in_place]` | 0.897897 ± 0.00215695 (W 0.00323211) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.58609 ± 0.0109606 (W 0.00364323) | 0.77167 ± 0.00597798 (W 0.00332231) | 🔴 faer-fork 2.06× slower than `lapack-openblas` |  |
| `solve/n257/b1 [in_place]` | 0.881812 ± 0.00268549 (W 0.00231737) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 7.26061 ± 0.0239598 (W 0.0179966) | 2.76978 ± 0.0165456 (W 0.0093169) | 🔴 faer-fork 2.62× slower than `lapack-openblas` |  |
| `solve/n511/b1 [in_place]` | 3.68593 ± 0.0107722 (W 0.00476151) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 7.32817 ± 0.0328957 (W 0.0214936) | 2.80614 ± 0.0156543 (W 0.0062911) | 🔴 faer-fork 2.61× slower than `lapack-openblas` |  |
| `solve/n513/b1 [in_place]` | 3.73522 ± 0.0115653 (W 0.00962505) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 14.4523 ± 0.176102 (W 0.021706) | 13.4063 ± 0.0228804 (W 0.0269901) | 🔴 faer-fork 1.08× slower than `lapack-openblas` |  |
| `svd_thin/n255/b1 [in_place]` | 14.3881 ± 0.0246706 (W 0.0199009) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 11.7966 ± 0.132315 (W 0.0218318) | 14.1026 ± 0.0583838 (W 0.0271914) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [in_place]` | 11.7283 ± 0.0366156 (W 0.047743) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 57.9034 ± 0.293465 (W 0.239934) | 57.7445 ± 0.163123 (W 0.0884892) | · |  |
| `svd_thin/n511/b1 [in_place]` | 57.6708 ± 0.113766 (W 0.0583063) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 54.7782 ± 0.300745 (W 0.0595755) | 60.0759 ± 0.221976 (W 0.0661093) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [in_place]` | 54.2184 ± 0.0767162 (W 0.0594461) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 7.3946 ± 0.0171096 (W 0.00975497) | 7.12151 ± 0.00595322 (W 0.00837082) | · |  |
| `svd_values/n255/b1 [in_place]` | 7.36319 ± 0.02885 (W 0.0055939) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 7.06913 ± 0.0140598 (W 0.0106529) | 7.39152 ± 0.0126393 (W 0.0060549) | · |  |
| `svd_values/n257/b1 [in_place]` | 7.03449 ± 0.0148065 (W 0.0124633) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 33.1816 ± 0.397568 (W 0.102606) | 27.4889 ± 0.0689179 (W 0.0394955) | 🔴 faer-fork 1.21× slower than `lapack-openblas` |  |
| `svd_values/n511/b1 [in_place]` | 33.1907 ± 0.124904 (W 0.0430937) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 31.0098 ± 0.114453 (W 0.0315377) | 28.1959 ± 0.0874725 (W 0.0318422) | 🔴 faer-fork 1.10× slower than `lapack-openblas` |  |
| `svd_values/n513/b1 [in_place]` | 30.8085 ± 0.122916 (W 0.0373078) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.022868 ± 0.000312864 (W 0.00020159) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.023353 ± 0.000151981 (W 0.000192594) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.057873 ± 0.00161309 (W 0.000353017) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n513/b1` | 0.0582685 ± 0.000848504 (W 0.000494869) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `full_piv_lu/255`, `full_piv_lu/257`, `full_piv_lu/511`, `full_piv_lu/513`, `qr/255`, `qr/257`, `qr/511`, `qr/513`, `triangular_solve/255`, `triangular_solve/257`, `triangular_solve/511`, `triangular_solve/513`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
