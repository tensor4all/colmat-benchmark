# `faer-kernels-large` on `zen5-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `e7a1baa48af15fb7767986acb7501818796ec8ec`
- toolkit commit: `ec1366c391cbd5023116ddcb6fdadc9c5b76f582`
- measured binary SHA-256: `74bf9a34402917de064ab6a5ff43c0547437eb7a8e909cd937371a0668f3bbaa`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-21T19:11:24.792199Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu faer-kernels-large --vendor openblas --engines faer-fork,lapack-openblas --threads 4,8 --regimes linalg-large,gemm-large,linalg-boundary --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 --reps 5 --aa 6 --prime-ms 250 --label pub-93da6217-20261120-large-openblas`
- raw data: `data/results/zen5-cpu/faer-kernels-large/20261121T112606Z-pub-93da6217-20261120-large-openblas/`

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
  Known blind spot: A vendor arm records the library it linked. OpenBLAS and AOCL report the kernel their runtime dispatched to (AOCL for BLIS and libflame separately); MKL names no kernel and no processor count, so its manifest leaves both empty rather than inferring them.
  Known blind spot: Known limitation: Coverage is the declared grid: f64/c64/f32/c32 at 1/4/8T for the small suite, f64/c64 at 4/8T for the large suite's 1024/1536 + boundary tier; the adoption-only 2048/3072/4096 tiers are not measured.
- **Rows** — one row per engine in every table below:
  - `faer-fork` — The measured library: the tensor4all fork (`colmat`, `[lib] name = "colmat"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/t4a-faer.rev`
    Caveat: The fork crate is `colmat` (`[lib] name = "colmat"`); the upstream baseline is `faer` (`faer_upstream = { package = "faer" }`). They are distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. measured at 93da6217, colmat-rs main at the 2026-11-17 head: no tag names that revision, and the published crates.io colmat 0.34.0 was cut from e1c83482. The 2026-11-17 zen5-cpu recordings were taken at this revision.
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
- OS / arch: `Linux 7.0.0-43-generic` / `x86_64`
- hostname: `f990be0ea5d6`
- CPU sets: 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 0.91%, worst 14.54% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 9 of 1088 engine cells exceed 5% and carry `!`; median within-process relative SD 0.23%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
| `cholesky/n1024/b1` | 5.26941 ± 0.0225747 (W 0.00928748) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 5.46371 ± 0.028307 (W 0.0139314) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 14.9089 ± 0.0420236 (W 0.0129416) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 16.6695 ± 0.203289 (W 0.150426) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 119.358 ± 0.251496 (W 0.127771) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 126.363 ± 0.0631124 (W 0.402565) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 325.697 ± 0.647231 (W 0.426165) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 341.352 ± 0.814427 (W 0.904934) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 58.9843 ± 0.182508 (W 0.130825) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 66.6809 ± 0.098973 (W 0.186843) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 248.535 ± 0.471043 (W 1.21808) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 268.862 ± 0.753396 (W 1.55754) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 10.9226 ± 0.636503 (W 0.0175265) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1024/b1 [api-inclusive]` | 19.0689 ± 0.935232 (W 0.0684493) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1536/b1` | 32.2846 ± 0.0488297 (W 0.118859) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 49.8248 ± 0.201832 (W 0.128276) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 39.8133 ± 0.118445 (W 0.0974038) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 50.7114 ± 0.239533 (W 0.0672537) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 131.841 ± 0.368367 (W 0.533725) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 154.857 ± 0.205791 (W 0.295094) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 19.4012 ± 0.0683701 (W 0.0341169) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 58.9551 ± 0.125029 (W 0.143054) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 10.9413 ± 0.034517 (W 0.0271467) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 12.1886 ± 0.0499241 (W 0.100237) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 30.759 ± 0.11427 (W 0.414139) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 34.8444 ± 0.205786 (W 0.229202) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 205.663 ± 0.304955 (W 0.462028) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 227.39 ± 0.292231 (W 0.393249) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 702.123 ± 3.12893 (W 0.633037) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 752.185 ± 1.71035 (W 1.49143) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 105.468 ± 0.591382 (W 0.355616) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 105.826 ± 0.445689 (W 0.255182) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 410.108 ± 1.57648 (W 1.03049) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 417.476 ± 1.60782 (W 1.35845) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.0839785 ± 0.00160692 (W 0.000641914) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.230105 ± 0.00266765 (W 0.00427379) | - | - | `*` one row has no measurement here |

## linalg-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 3.03849 ± 0.0157066 (W 0.0089142) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 3.9407 ± 0.0440948 (W 0.00585448) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 8.02478 ± 0.0255145 (W 0.0102268) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 10.785 ± 0.087209 (W 0.0458638) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 73.8706 ± 0.116156 (W 0.0562161) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 80.3813 ± 0.0974186 (W 0.217233) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 190.801 ± 0.348644 (W 0.27592) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 205.288 ± 0.288779 (W 0.429749) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 45.5336 ± 0.0855035 (W 0.0912528) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 54.4975 ± 0.173662 (W 0.135653) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 147.381 ± 0.802484 (W 0.845692) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 170.016 ± 0.489935 (W 1.01994) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 7.10343 ± 0.0430341 (W 0.0345818) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 16.1019 ± 0.159024 (W 0.0481163) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 20.1085 ± 0.0758258 (W 0.0544369) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 39.8992 ± 0.138695 (W 0.147889) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 21.8735 ± 0.134292 (W 0.0339156) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 33.2019 ± 0.145109 (W 0.256208) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 71.0801 ± 0.109648 (W 0.282241) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 99.042 ± 0.347049 (W 0.272319) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 10.6676 ± 0.0143289 (W 0.0200661) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 32.0894 ± 0.0716381 (W 0.0539039) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 6.74436 ± 0.0150841 (W 0.027114) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 9.09694 ± 0.122086 (W 0.178031) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 19.4105 ± 0.0715611 (W 0.0509392) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 24.2387 ± 0.132179 (W 0.266569) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 146.26 ± 0.432201 (W 0.295559) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 166.887 ± 0.456364 (W 0.56334) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 446.748 ± 1.03561 (W 1.27457) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 493.58 ± 2.03595 (W 1.26309) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 78.5159 ± 0.337506 (W 0.126916) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 78.548 ± 0.274419 (W 0.212774) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 260.543 ± 1.16859 (W 0.898295) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 267.718 ± 1.43979 (W 1.49366) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.0744695 ± 0.00101819 (W 0.000422) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.154536 ± 0.00233121 (W 0.00124619) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 4.06386 ± 0.0144071 (W 0.00693645) | 6.4747 ± 0.0320366 (W 0.0165317) | 🔵 faer-fork 1.59× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 5.02665 ± 0.0207502 (W 0.00770464) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 12.8229 ± 0.0551077 (W 0.0175144) | 19.3517 ± 0.112848 (W 0.0201916) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 14.9997 ± 0.0846641 (W 0.0240201) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 118.751 ± 0.265089 (W 0.0903179) | 128.308 ± 0.318846 (W 0.194223) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [api-inclusive]` | 125.867 ± 0.160205 (W 0.311781) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 323.806 ± 1.04638 (W 0.597779) | 362.002 ± 0.737463 (W 0.493839) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `eigh/n1536/b1 [api-inclusive]` | 339.96 ± 0.918941 (W 0.927129) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 61.7314 ± 0.0840292 (W 0.189269) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 64.2707 ± 0.136469 (W 0.0811337) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 262.207 ± 0.642026 (W 1.01822) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 261.476 ± 0.648908 (W 1.01082) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 8.89354 ± 0.0314542 (W 0.0169689) | 12.7672 ± 0.0596467 (W 0.0280655) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `lu/n1024/b1 [api-inclusive]` | 16.4981 ± 0.0701639 (W 0.0345862) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 26.632 ± 0.116633 (W 0.0681984) | 32.695 ± 0.0491864 (W 0.0283752) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `lu/n1536/b1 [api-inclusive]` | 44.0292 ± 0.147522 (W 0.0583461) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 37.1338 ± 0.159745 (W 0.0296896) | 96.0921 ± 0.319041 (W 0.196586) | 🔵 faer-fork 2.59× faster than `lapack-openblas` |  |
| `qr/n1024/b1 [api-inclusive]` | 48.1586 ± 0.217668 (W 0.12309) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 119.928 ± 0.285281 (W 0.218927) | 277.802 ± 0.323943 (W 0.260168) | 🔵 faer-fork 2.32× faster than `lapack-openblas` |  |
| `qr/n1536/b1 [api-inclusive]` | 148.791 ± 0.606453 (W 0.314649) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 17.2449 ± 0.0577419 (W 0.0425022) | 36.6931 ± 0.140785 (W 0.0246061) | 🔵 faer-fork 2.13× faster than `lapack-openblas` |  |
| `qr_factor/n1536/b1` | 52.9896 ± 0.209447 (W 0.0543096) | 104.377 ± 0.230281 (W 0.0930289) | 🔵 faer-fork 1.97× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 8.58701 ± 0.0231123 (W 0.0124383) | 12.3105 ± 0.0385562 (W 0.0160725) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `solve/n1024/b1 [api-inclusive]` | 10.0335 ± 0.0450608 (W 0.0148627) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 25.9511 ± 0.0303955 (W 0.0299033) | 31.5828 ± 0.0707795 (W 0.0284644) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `solve/n1536/b1 [api-inclusive]` | 29.5008 ± 0.0531852 (W 0.0344164) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 198.933 ± 0.638223 (W 0.291606) | 408.88 ± 5.01311 (W 0.721372) | 🔵 faer-fork 2.06× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 220.914 ± 0.258527 (W 0.582168) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 675.414 ± 3.50898 (W 0.966231) | 1169.16 ± 4.56223 (W 1.17987) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 727.881 ± 2.30281 (W 1.38208) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 97.2345 ± 0.522565 (W 0.18674) | 213.622 ± 3.25444 (W 0.430981) | 🔵 faer-fork 2.20× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 97.2909 ± 0.234258 (W 0.134052) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 384.232 ± 0.908843 (W 0.855934) | 611.035 ± 6.09149 (W 1.26383) | 🔵 faer-fork 1.59× faster than `lapack-openblas` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 393.505 ± 0.979608 (W 1.73715) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.104386 ± 0.0011352 (W 0.000809668) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.253695 ± 0.00168389 (W 0.00420723) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 2.39414 ± 0.00678368 (W 0.00677999) | 4.64267 ± 0.0402893 (W 0.0158418) | 🔵 faer-fork 1.94× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 3.45686 ± 0.0113663 (W 0.00356646) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 6.89749 ± 0.0198616 (W 0.0153739) | 11.4442 ± 0.0282318 (W 0.0155614) | 🔵 faer-fork 1.66× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 9.31113 ± 0.0517502 (W 0.022249) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 73.2605 ± 0.0733933 (W 0.0559063) | 100.957 ± 0.327372 (W 0.111064) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [api-inclusive]` | 79.8431 ± 0.0746306 (W 0.176806) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 189.187 ± 0.312126 (W 0.810893) | 270.86 ± 0.450076 (W 0.402908) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `eigh/n1536/b1 [api-inclusive]` | 203.849 ± 0.322798 (W 0.515381) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 46.6375 ± 0.0848166 (W 0.0562397) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 51.8176 ± 0.0867813 (W 0.247971) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 149.927 ± 0.240935 (W 1.27337) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 162.836 ± 0.886273 (W 1.0296) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 5.80145 ± 0.0133038 (W 0.0123417) | 6.69837 ± 0.0335042 (W 0.0161992) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `lu/n1024/b1 [api-inclusive]` | 13.3983 ± 0.0808993 (W 0.0310105) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 16.4094 ± 0.0629605 (W 0.0724427) | 16.6387 ± 0.0815689 (W 0.0856263) | · |  |
| `lu/n1536/b1 [api-inclusive]` | 34.0048 ± 0.0873293 (W 0.152373) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 19.6226 ± 0.266382 (W 0.0333429) | 77.8088 ± 0.295349 (W 0.0714023) | 🔵 faer-fork 3.97× faster than `lapack-openblas` |  |
| `qr/n1024/b1 [api-inclusive]` | 30.7228 ± 0.158601 (W 0.117449) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 63.7197 ± 0.256082 (W 0.248417) | 220.809 ± 0.622646 (W 0.537413) | 🔵 faer-fork 3.47× faster than `lapack-openblas` |  |
| `qr/n1536/b1 [api-inclusive]` | 92.5345 ± 0.371663 (W 0.388163) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 9.05767 ± 0.024747 (W 0.0269616) | 29.8687 ± 0.194456 (W 0.0282946) | 🔵 faer-fork 3.30× faster than `lapack-openblas` |  |
| `qr_factor/n1536/b1` | 28.1811 ± 0.109358 (W 0.0413717) | 82.1391 ± 0.302228 (W 0.116288) | 🔵 faer-fork 2.91× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 5.40521 ± 0.0127028 (W 0.00705985) | 6.55312 ± 0.0186449 (W 0.0128214) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `solve/n1024/b1 [api-inclusive]` | 6.96912 ± 0.028652 (W 0.012616) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 15.6716 ± 0.0362304 (W 0.0413238) | 16.2219 ± 0.0566113 (W 0.0253086) | · |  |
| `solve/n1536/b1 [api-inclusive]` | 19.009 ± 0.0456494 (W 0.030026) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 139.345 ± 0.151196 (W 0.20706) | 375.548 ± 3.95312 (W 1.20592) | 🔵 faer-fork 2.70× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 160.012 ± 0.21341 (W 0.36135) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 421.047 ± 0.632639 (W 0.712689) | 1021.89 ± 6.04948 (W 1.22904) | 🔵 faer-fork 2.43× faster than `lapack-openblas` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 470.249 ± 0.978064 (W 1.16191) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 70.0595 ± 0.252329 (W 0.19944) | 199.307 ± 3.16051 (W 0.393288) | 🔵 faer-fork 2.84× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 70.2416 ± 0.201493 (W 0.05177) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 235.108 ± 0.585542 (W 0.721174) | 542.525 ± 2.68212 (W 0.712649) | 🔵 faer-fork 2.31× faster than `lapack-openblas` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 244.104 ± 0.661875 (W 1.80601) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.092904 ± 0.00124097 (W 0.000501461) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.176814 ± 0.00390935 (W 0.00254462) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 14.8535 ± 0.0314574 (W 0.0145724) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 18.0989 ± 0.085757 (W 0.0527137) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 47.1381 ± 0.0751321 (W 0.0981557) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 70.0729 ± 0.310574 (W 0.30158) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 235.711 ± 0.205855 (W 0.2683) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 246.832 ± 0.213896 (W 0.501135) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 783.236 ± 2.58713 (W 1.41371) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 823.247 ± 1.26336 (W 1.42642) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 136.602 ± 0.318344 (W 0.409277) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 154.67 ± 1.06 (W 0.997781) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 689.638 ± 0.456271 (W 1.99595) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 693.161 ± 1.08876 (W 2.47091) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 33.6424 ± 0.196116 (W 0.189325) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 50.7985 ± 0.293324 (W 0.236013) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 107.874 ± 0.268756 (W 0.444856) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 138.644 ± 0.376718 (W 0.263628) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 143.351 ± 0.173806 (W 0.407163) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 166.827 ± 0.451041 (W 0.332649) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 466.77 ± 0.838222 (W 0.546901) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 514.686 ± 1.25335 (W 0.695265) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 63.5421 ± 0.198472 (W 0.120262) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 206.297 ± 0.592732 (W 0.470231) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 33.6298 ± 0.23604 (W 0.143328) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 37.5959 ± 0.160637 (W 0.143182) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 107.889 ± 0.33191 (W 0.287492) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 138.435 ± 0.671456 (W 0.274081) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 509.564 ± 1.60399 (W 0.991669) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 554.987 ± 1.19614 (W 1.62538) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1826.5 ± 4.05197 (W 2.44246) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1923.51 ± 3.92328 (W 3.39312) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 285.703 ± 1.49604 (W 0.658222) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 286.832 ± 1.36985 (W 3.32108) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1129.93 ± 3.0292 (W 1.90773) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 1150.63 ± 4.19824 (W 2.38886) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.189267 ± 0.00164474 (W 0.00344566) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.542345 ± 0.00456818 (W 0.00753484) | - | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.33727 ± 0.00985971 (W 0.0540791) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 11.6215 ± 0.18052 (W 0.10185) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 25.0236 ± 0.0480552 (W 0.0339858) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 48.7862 ± 0.24365 (W 0.224187) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 137.785 ± 0.18737 (W 0.213686) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 148.061 ± 0.169724 (W 0.617255) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 425.927 ± 1.26151 (W 1.51941) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 465.932 ± 0.504316 (W 1.48223) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 77.1804 ± 0.489761 (W 0.387486) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 95.4679 ± 0.391622 (W 0.687674) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 484.974 ± 1.51781 (W 2.70919) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 515.675 ± 2.70791 (W 3.75602) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 20.9597 ± 0.0547987 (W 0.124313) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 38.9961 ± 0.234928 (W 0.140266) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 64.9734 ± 0.166809 (W 0.204737) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 98.5454 ± 0.377587 (W 0.329279) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 78.7868 ± 0.180226 (W 0.188227) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 101.418 ± 0.241998 (W 0.36292) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 256.016 ± 0.276944 (W 0.594798) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 303.373 ± 0.578144 (W 0.360845) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 35.0236 ± 0.0907596 (W 0.0495852) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 114.486 ± 0.396566 (W 0.304754) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 21.3311 ± 0.0587862 (W 0.0272253) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 25.4747 ± 0.0890188 (W 0.0922469) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 65.3653 ± 0.241381 (W 0.289115) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 98.3593 ± 0.323398 (W 0.292365) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 303.707 ± 0.311346 (W 0.998906) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 349.277 ± 0.982281 (W 1.53412) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1208.33 ± 1.95542 (W 2.61452) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1302.82 ± 2.10304 (W 3.26027) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 171.208 ± 1.01522 (W 0.753636) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 171.689 ± 0.669119 (W 2.92808) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 809.685 ± 2.8299 (W 2.17014) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 827.864 ± 2.73056 (W 2.22842) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.133489 ± 0.00220809 (W 0.00104334) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.393549 ± 0.00353343 (W 0.00649792) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 14.7157 ± 0.0195261 (W 0.017925) | 19.4224 ± 0.0433479 (W 0.0192009) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 16.5268 ± 0.0536436 (W 0.0170031) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 47.0326 ± 0.0622588 (W 0.0944525) | 63.6039 ± 0.203508 (W 0.065185) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 66.4565 ± 0.191477 (W 0.260027) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 234.233 ± 0.360543 (W 0.379175) | 419.57 ± 1.96757 (W 1.21055) | 🔵 faer-fork 1.79× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [api-inclusive]` | 245.266 ± 0.327868 (W 0.390271) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 781.034 ± 1.76651 (W 1.23026) | 1849.44 ± 4.06196 (W 2.80883) | 🔵 faer-fork 2.37× faster than `lapack-openblas` |  |
| `eigh/n1536/b1 [api-inclusive]` | 820.195 ± 0.572218 (W 1.37622) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 140.785 ± 0.763101 (W 0.733828) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 149.299 ± 1.485 (W 1.6322) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 676.016 ± 2.31949 (W 2.24242) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 681.357 ± 2.98707 (W 2.53155) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 29.9215 ± 0.135874 (W 0.0896014) | 32.6657 ± 0.114571 (W 0.0330628) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `lu/n1024/b1 [api-inclusive]` | 45.5629 ± 0.116117 (W 0.194586) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 92.9092 ± 0.26487 (W 0.115247) | 104.02 ± 0.297005 (W 0.132967) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `lu/n1536/b1 [api-inclusive]` | 127.755 ± 0.772013 (W 0.297005) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 138.163 ± 0.318617 (W 0.10651) | 223.04 ± 0.643429 (W 0.499928) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `qr/n1024/b1 [api-inclusive]` | 161.626 ± 0.35251 (W 0.356708) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 448.628 ± 1.04528 (W 0.560714) | 682.161 ± 0.918313 (W 0.66739) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `qr/n1536/b1 [api-inclusive]` | 503.412 ± 0.83297 (W 0.588216) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 59.9337 ± 0.213231 (W 0.146715) | 87.4918 ± 0.157884 (W 0.090608) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `qr_factor/n1536/b1` | 191.551 ± 0.397251 (W 0.263804) | 268.173 ± 0.999599 (W 0.335167) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 29.9979 ± 0.149432 (W 0.137283) | 32.6291 ± 0.038301 (W 0.0267099) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `solve/n1024/b1 [api-inclusive]` | 33.3479 ± 0.106919 (W 0.186115) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 93.0997 ± 0.179573 (W 0.0839854) | 103.723 ± 0.136133 (W 0.0484847) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `solve/n1536/b1 [api-inclusive]` | 127.944 ± 0.370962 (W 0.152883) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 502.844 ± 1.25131 (W 0.62485) | 718.183 ± 1.92578 (W 2.00063) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 547.36 ± 1.15171 (W 1.04592) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1802.47 ± 2.01337 (W 1.75004) | 2269.5 ± 6.75387 (W 3.88034) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1903.08 ± 7.53099 (W 2.8583) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 278.507 ± 0.966543 (W 0.734669) | 353.185 ± 0.696406 (W 0.915482) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 279.84 ± 1.48331 (W 3.1737) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1106.86 ± 2.56037 (W 1.56964) | 1190.69 ± 3.22582 (W 1.82992) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 1124.52 ± 1.98576 (W 2.52022) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.211867 ± 0.00316705 (W 0.00388691) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.65718 ± 0.00757201 (W 0.00543904) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.30678 ± 0.0366025 (W 0.0128978) | 10.864 ± 0.0448227 (W 0.0233526) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 10.2509 ± 0.105403 (W 0.064534) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 25.3946 ± 0.0603118 (W 0.0334078) | 33.1983 ± 0.0656932 (W 0.0394751) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 44.9927 ± 0.204424 (W 0.145218) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 136.449 ± 0.291228 (W 0.510968) | 229.098 ± 0.599611 (W 0.538073) | 🔵 faer-fork 1.68× faster than `lapack-openblas` |  |
| `eigh/n1024/b1 [api-inclusive]` | 146.471 ± 0.130077 (W 0.436311) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 423.49 ± 0.703815 (W 1.09884) | 1433.98 ± 3.4638 (W 3.23293) | 🔵 faer-fork 3.39× faster than `lapack-openblas` |  |
| `eigh/n1536/b1 [api-inclusive]` | 462.926 ± 1.07815 (W 0.991626) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 76.9905 ± 0.484934 (W 0.323154) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 90.4176 ± 0.255334 (W 0.392424) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 474.007 ± 4.12625 (W 2.0209) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 505.303 ± 1.6572 (W 3.09057) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 18.0144 ± 0.0594508 (W 0.0331989) | 16.8251 ± 0.0225036 (W 0.021652) | 🔴 faer-fork 1.07× slower than `lapack-openblas` |  |
| `lu/n1024/b1 [api-inclusive]` | 33.7658 ± 0.0588692 (W 0.0952479) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 52.9122 ± 0.107604 (W 0.143676) | 53.0018 ± 0.120473 (W 0.129319) | · |  |
| `lu/n1536/b1 [api-inclusive]` | 87.5744 ± 0.619668 (W 0.242735) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 72.5598 ± 0.124899 (W 0.242044) | 147.949 ± 0.542386 (W 0.240516) | 🔵 faer-fork 2.04× faster than `lapack-openblas` |  |
| `qr/n1024/b1 [api-inclusive]` | 95.9579 ± 0.148378 (W 0.233087) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 237.811 ± 0.33021 (W 0.314185) | 440.215 ± 0.819656 (W 0.574279) | 🔵 faer-fork 1.85× faster than `lapack-openblas` |  |
| `qr/n1536/b1 [api-inclusive]` | 292.271 ± 0.605111 (W 0.589527) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 32.0219 ± 0.058472 (W 0.0445782) | 59.0673 ± 0.189571 (W 0.0660887) | 🔵 faer-fork 1.84× faster than `lapack-openblas` |  |
| `qr_factor/n1536/b1` | 102.724 ± 0.167826 (W 0.158621) | 171.094 ± 0.102909 (W 0.374088) | 🔵 faer-fork 1.67× faster than `lapack-openblas` |  |
| `solve/n1024/b1` | 17.9018 ± 0.049985 (W 0.0515877) | 17.2102 ± 0.0449331 (W 0.0347936) | · |  |
| `solve/n1024/b1 [api-inclusive]` | 21.0324 ± 0.0490546 (W 0.0380101) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 53.3806 ± 0.0311785 (W 0.129068) | 53.5467 ± 0.111877 (W 0.0582014) | · |  |
| `solve/n1536/b1 [api-inclusive]` | 88.09 ± 0.706293 (W 0.319585) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 297.481 ± 1.05478 (W 0.842566) | 551.272 ± 2.38329 (W 1.45452) | 🔵 faer-fork 1.85× faster than `lapack-openblas` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 341.418 ± 0.562955 (W 0.911767) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1185.24 ± 2.73409 (W 2.07927) | 1813.74 ± 4.108 (W 2.36085) | 🔵 faer-fork 1.53× faster than `lapack-openblas` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1282.51 ± 2.74361 (W 2.50593) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 164.703 ± 1.10212 (W 0.87341) | 285.244 ± 0.612208 (W 0.719884) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 164.507 ± 0.755562 (W 2.07923) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 785.848 ± 2.79064 (W 2.40502) | 1062.47 ± 1.76451 (W 1.22371) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 804.714 ± 2.79378 (W 2.62496) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.152791 ± 0.00272293 (W 0.000953511) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.502018 ± 0.00328559 (W 0.0061362) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024`, `cholesky/1536`, `eigh/1024`, `eigh/1536`, `full_piv_lu/1024`, `full_piv_lu/1536`, `lu/1024`, `lu/1536`, `qr/1024`, `qr/1536`, `qr_factor/1024`, `qr_factor/1536`, `solve/1024`, `solve/1536`, `svd_thin/1024`, `svd_thin/1536`, `svd_values/1024`, `svd_values/1536`, `triangular_solve/1024`, `triangular_solve/1536`.

## gemm-large, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.5264 ± 0.0226355 (W 0.0143026) | 20.4838 ± 0.0241629 (W 0.0277616) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 59.2833 ± 0.133711 (W 0.027519) | 67.7872 ± 0.130893 (W 0.0706095) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |

## gemm-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.97404 ± 0.00275016 (W 0.00678884) | 10.6137 ± 0.0112417 (W 0.0141756) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 30.1006 ± 0.0529947 (W 0.0336974) | 34.8213 ± 0.0806652 (W 0.0583955) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.5352 ± 0.0248092 (W 0.0134888) | 20.4791 ± 0.0201586 (W 0.0290339) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 59.3996 ± 0.128106 (W 0.0184145) | 67.9189 ± 0.209257 (W 0.0592118) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |

## gemm-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.96933 ± 0.0110851 (W 0.00841616) | 10.6306 ± 0.0493134 (W 0.0296971) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 30.1186 ± 0.0676738 (W 0.0472673) | 34.8365 ± 0.0667571 (W 0.0650545) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.7167 ± 0.0862735 (W 0.0732022) | 82.3515 ± 0.106486 (W 0.0457753) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 238.332 ± 0.469002 (W 0.297556) | 275.203 ± 0.913336 (W 0.237565) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |

## gemm-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.1383 ± 0.0471682 (W 0.0211235) | 41.9213 ± 0.118674 (W 0.0433364) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 119.455 ± 0.357314 (W 0.178079) | 138.747 ± 0.570522 (W 0.192418) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.7549 ± 0.0738308 (W 0.0303791) | 82.3481 ± 0.220208 (W 0.0692171) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 238.434 ± 0.518289 (W 0.256549) | 275.712 ± 0.876898 (W 0.341179) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.1099 ± 0.0544641 (W 0.0407248) | 41.9252 ± 0.109085 (W 0.0609506) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 119.724 ± 0.494453 (W 0.157515) | 138.63 ± 0.408278 (W 0.146027) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## linalg-boundary, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.168758 ± 0.00074395 (W 0.00152076) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.196344 ± 0.00153035 (W 0.00160879) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.154984 ± 0.00109195 (W 0.00143513) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.182928 ± 0.00107401 (W 0.00160757) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.706056 ± 0.00105579 (W 0.00216575) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.860742 ± 0.00307891 (W 0.00279203) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.683098 ± 0.00244209 (W 0.00196287) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.936869 ± 0.00392815 (W 0.00324264) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.8424 ± 0.00989192 (W 0.00878477) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.84401 ± 0.018084 (W 0.0101376) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.76432 ± 0.00880878 (W 0.00520062) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.77529 ± 0.0202745 (W 0.00964367) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.7116 ± 0.0362625 (W 0.0257696) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 24.6152 ± 0.0280735 (W 0.0317195) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.2237 ± 0.0252155 (W 0.0262282) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 25.2285 ± 0.0361786 (W 0.030667) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.25844 ± 0.00961993 (W 0.00370147) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.24578 ± 0.0195421 (W 0.0085286) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.29767 ± 0.00531101 (W 0.00480954) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.54268 ± 0.0137692 (W 0.00667309) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 6.28237 ± 0.0143851 (W 0.0150241) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 13.0408 ± 0.0352782 (W 0.0274352) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.42361 ± 0.0161639 (W 0.0150553) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 7.35235 ± 0.0289218 (W 0.0130088) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.434541 ± 0.00230339 (W 0.00300831) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.70488 ± 0.00799151 (W 0.00568239) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.416036 ± 0.00162159 (W 0.00386668) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.768335 ± 0.0093009 (W 0.00354964) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.75939 ± 0.0117763 (W 0.013234) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.51122 ± 0.0195868 (W 0.00888448) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.85546 ± 0.00979743 (W 0.00720771) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.49866 ± 0.0314154 (W 0.0109434) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.04884 ± 0.00135249 (W 0.00177107) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.56319 ± 0.00523364 (W 0.00862704) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.11518 ± 0.00219175 (W 0.00392795) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.65712 ± 0.00464062 (W 0.00566303) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.83463 ± 0.0120055 (W 0.0111491) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 8.35319 ± 0.0471831 (W 0.0224979) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 6.00275 ± 0.0132707 (W 0.0100763) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 8.52655 ± 0.0291635 (W 0.0174926) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.660318 ± 0.00101883 (W 0.00328808) | - | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 0.696414 ± 0.00286502 (W 0.00245928) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 3.20855 ± 0.0104081 (W 0.00916546) | - | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 3.21323 ± 0.01205 (W 0.0136226) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.353233 ± 0.00208269 (W 0.00284984) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.708448 ± 0.00391942 (W 0.00385191) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.421355 ± 0.00163841 (W 0.00241197) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.771597 ± 0.00626956 (W 0.00643708) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.72155 ± 0.00873977 (W 0.0095505) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.47226 ± 0.0281694 (W 0.0168446) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.81402 ± 0.0095632 (W 0.00469335) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.44363 ± 0.128159 (W 0.0241355) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n255/b1` | 7.95933 ± 0.0126715 (W 0.0102319) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 7.98411 ± 0.010758 (W 0.00960529) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.16169 ± 0.0250321 (W 0.0135086) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 8.18307 ± 0.0128803 (W 0.018745) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.753 ± 0.0557779 (W 0.0342617) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 35.8245 ± 0.0562839 (W 0.0465842) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.5523 ± 0.0118824 (W 0.0474121) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 36.6601 ± 0.0717024 (W 0.0509003) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 4.01315 ± 0.0165204 (W 0.00669329) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 4.00906 ± 0.0172601 (W 0.0124661) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.10432 ± 0.00996018 (W 0.00867765) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 4.11306 ± 0.00610206 (W 0.00609157) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.4983 ± 0.0739577 (W 0.0245704) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 17.493 ± 0.100961 (W 0.0222666) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.6667 ± 0.11168 (W 0.039614) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 17.7313 ± 0.0708571 (W 0.0262669) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.012429 ± 0.000133515 (W 0.000111918) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.0128995 ± 0.000205101 (W 0.0001423) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.02946 ± 0.000281248 (W 0.000184723) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.029886 ± 0.000288881 (W 0.000279257) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.152618 ± 0.00125986 (W 0.000851131) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.181497 ± 0.00137835 (W 0.00128275) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.143541 ± 0.00091078 (W 0.00116436) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.169079 ± 0.000924725 (W 0.00117936) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.514939 ± 0.00375259 (W 0.00251781) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.69019 ± 0.00217506 (W 0.00279891) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.507038 ± 0.00208377 (W 0.00221192) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.801757 ± 0.0127447 (W 0.00200439) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 3.99077 ± 0.0122494 (W 0.00674639) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 3.99745 ± 0.0110687 (W 0.00592104) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.95888 ± 0.0151858 (W 0.00535019) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 3.9674 ± 0.0086511 (W 0.00811612) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.1846 ± 0.0468304 (W 0.0462178) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 16.1844 ± 0.0334687 (W 0.0327621) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.7283 ± 0.0559401 (W 0.0287528) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 16.76 ± 0.0463747 (W 0.0195017) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.15557 ± 0.00645583 (W 0.00476507) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.18318 ± 0.00894138 (W 0.0120763) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.18494 ± 0.00618518 (W 0.00386727) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.51073 ± 0.00867017 (W 0.00511668) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 4.78124 ± 0.0434701 (W 0.00901234) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.8172 ± 0.0435932 (W 0.0250418) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.84186 ± 0.0332819 (W 0.0125523) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.1255 ± 0.0159025 (W 0.0180024) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.422998 ± 0.00163237 (W 0.00201426) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.676689 ± 0.00669894 (W 0.00306248) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.354577 ± 0.00159346 (W 0.00233028) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.733401 ± 0.00976472 (W 0.00484425) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.28597 ± 0.00266309 (W 0.00708213) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.14658 ± 0.0222887 (W 0.0137509) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.35817 ± 0.00532449 (W 0.00522597) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.14766 ± 0.0396927 (W 0.0106013) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.841927 ± 0.00209586 (W 0.00162186) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.39791 ± 0.00650445 (W 0.00656602) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 0.902378 ± 0.00498727 (W 0.00279482) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.47516 ± 0.0078087 (W 0.00517604) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 3.30253 ± 0.0073288 (W 0.00887361) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 6.10623 ± 0.0115364 (W 0.0343795) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 3.75244 ± 0.00731922 (W 0.00975055) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 6.53093 ± 0.0128563 (W 0.0226474) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.613733 ± 0.00294653 (W 0.00234147) | - | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 0.647609 ± 0.00251981 (W 0.00214283) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 1.87551 ± 0.0611607 (W 0.00296351) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n513/b1` | 2.25164 ± 0.0102628 (W 0.0129411) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.34177 ± 0.00212082 (W 0.00300195) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.677712 ± 0.00575615 (W 0.00323444) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.358494 ± 0.000774786 (W 0.00214537) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.745951 ± 0.00385046 (W 0.00391851) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.25475 ± 0.00841321 (W 0.005115) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.16312 ± 0.0272942 (W 0.00988036) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.32312 ± 0.00838292 (W 0.0045198) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.11844 ± 0.0241431 (W 0.00866255) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.29121 ± 0.0473982 (W 0.0102495) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 7.3143 ± 0.0152243 (W 0.011576) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.40543 ± 0.025376 (W 0.00678799) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 7.45022 ± 0.0212984 (W 0.0132716) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.5154 ± 0.066183 (W 0.318031) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 29.6175 ± 0.0612829 (W 0.0588773) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.9678 ± 0.108144 (W 0.0487224) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 30.0946 ± 0.0813078 (W 0.0447783) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.91373 ± 0.0111485 (W 0.00713022) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 3.91934 ± 0.00722769 (W 0.00721703) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.96904 ± 0.0203849 (W 0.00577542) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 3.97395 ± 0.0122527 (W 0.00869316) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.4587 ± 0.0368893 (W 0.0133819) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 15.477 ± 0.0497273 (W 0.0160213) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.5241 ± 0.0651594 (W 0.0143248) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 15.5895 ± 0.0503804 (W 0.0324342) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0129835 ± 5.37649e-05 (W 6.8331e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.0131235 ± 5.45772e-05 (W 8.21007e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.030046 ± 0.000275981 (W 0.000168297) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0304405 ± 0.000217278 (W 0.000185982) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.159291 ± 0.000651761 (W 0.00126295) | 0.312939 ± 0.00192619 (W 0.00251728) | 🔵 faer-fork 1.96× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.200982 ± 0.00129449 (W 0.00145053) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.149349 ± 0.000670642 (W 0.000973171) | 0.257081 ± 0.00122081 (W 0.0019917) | 🔵 faer-fork 1.72× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.187594 ± 0.000972762 (W 0.00154684) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.672614 ± 0.00284673 (W 0.00194941) | 1.1812 ± 0.00530662 (W 0.00405508) | 🔵 faer-fork 1.76× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.85768 ± 0.00302237 (W 0.00276836) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.673951 ± 0.00252305 (W 0.00267414) | 1.18455 ± 0.00404697 (W 0.00267003) | 🔵 faer-fork 1.76× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.821503 ± 0.00375927 (W 0.00281808) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.84499 ± 0.0152434 (W 0.00619632) | 4.78622 ± 0.0128988 (W 0.00747066) | 🔴 faer-fork 1.22× slower than `lapack-openblas` |  |
| `eigh/n255/b1 [api-inclusive]` | 5.85731 ± 0.0166686 (W 0.0124228) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.76597 ± 0.0125678 (W 0.00633112) | 4.83203 ± 0.0090677 (W 0.00784932) | 🔴 faer-fork 1.19× slower than `lapack-openblas` |  |
| `eigh/n257/b1 [api-inclusive]` | 5.76813 ± 0.00549969 (W 0.00813796) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.6504 ± 0.0259538 (W 0.0268554) | 19.2982 ± 0.0865722 (W 0.0416172) | 🔴 faer-fork 1.28× slower than `lapack-openblas` |  |
| `eigh/n511/b1 [api-inclusive]` | 24.5134 ± 0.0334663 (W 0.0209571) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.0837 ± 0.0436498 (W 0.0335804) | 19.5938 ± 0.0728291 (W 0.0208648) | 🔴 faer-fork 1.28× slower than `lapack-openblas` |  |
| `eigh/n513/b1 [api-inclusive]` | 25.0795 ± 0.0271047 (W 0.0228865) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.16647 ± 0.00624409 (W 0.00257343) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.15988 ± 0.00421341 (W 0.0101105) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.17672 ± 0.00512675 (W 0.0053332) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.43724 ± 0.0154377 (W 0.00604516) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 5.99827 ± 0.0123812 (W 0.00931456) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 12.524 ± 0.0351558 (W 0.0305668) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.06498 ± 0.0266307 (W 0.00937398) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.83731 ± 0.0246579 (W 0.0172744) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.302965 ± 0.00101896 (W 0.00310179) | 0.432819 ± 0.000628552 (W 0.00199852) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 0.612247 ± 0.003039 (W 0.00337573) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.324271 ± 0.0025221 (W 0.00398505) | 0.44379 ± 0.0012532 (W 0.0019758) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 0.661103 ± 0.0044446 (W 0.00399362) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.48816 ± 0.00422682 (W 0.00395686) | 2.38108 ± 0.00736177 (W 0.00308996) | 🔵 faer-fork 1.60× faster than `lapack-openblas` |  |
| `lu/n511/b1 [api-inclusive]` | 3.00588 ± 0.00729247 (W 0.0107062) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.549 ± 0.0047532 (W 0.00360882) | 2.40663 ± 0.00327953 (W 0.0051161) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 2.99167 ± 0.0154961 (W 0.00661771) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.968729 ± 0.00119817 (W 0.00184905) | 2.03921 ± 0.00750773 (W 0.00475354) | 🔵 faer-fork 2.11× faster than `lapack-openblas` |  |
| `qr/n255/b1 [api-inclusive]` | 1.45952 ± 0.00690559 (W 0.00397674) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.00477 ± 0.00255567 (W 0.00146591) | 2.01645 ± 0.00678688 (W 0.00408632) | 🔵 faer-fork 2.01× faster than `lapack-openblas` |  |
| `qr/n257/b1 [api-inclusive]` | 1.52857 ± 0.00544509 (W 0.00658707) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.47998 ± 0.00970719 (W 0.00418113) | 8.8794 ± 0.0677092 (W 0.0122841) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `qr/n511/b1 [api-inclusive]` | 7.74868 ± 0.0223615 (W 0.0165525) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 5.5888 ± 0.0109515 (W 0.00898981) | 9.24688 ± 0.0304794 (W 0.0122803) | 🔵 faer-fork 1.65× faster than `lapack-openblas` |  |
| `qr/n513/b1 [api-inclusive]` | 7.93674 ± 0.0379552 (W 0.0105971) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.579958 ± 0.00151366 (W 0.0024161) | 1.07887 ± 0.00445023 (W 0.00200961) | 🔵 faer-fork 1.86× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 0.601853 ± 0.00230992 (W 0.00297986) | 1.04315 ± 0.00326931 (W 0.00270256) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 2.90684 ± 0.00926621 (W 0.00428546) | 4.35928 ± 0.0218058 (W 0.00887057) | 🔵 faer-fork 1.50× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 2.8708 ± 0.00596494 (W 0.00261313) | 4.54469 ± 0.0173854 (W 0.00926769) | 🔵 faer-fork 1.58× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.309975 ± 0.00183093 (W 0.00197922) | 0.417034 ± 0.00301277 (W 0.002218) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 0.623157 ± 0.00332895 (W 0.00353345) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.327085 ± 0.00217391 (W 0.00273324) | 0.42504 ± 0.00154928 (W 0.00230252) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 0.657636 ± 0.00534997 (W 0.00461327) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.45271 ± 0.00555867 (W 0.0062197) | 2.31757 ± 0.00431626 (W 0.00459738) | 🔵 faer-fork 1.60× faster than `lapack-openblas` |  |
| `solve/n511/b1 [api-inclusive]` | 2.9354 ± 0.0208544 (W 0.00970482) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.51055 ± 0.0057104 (W 0.0042497) | 2.34612 ± 0.00831062 (W 0.00514024) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `solve/n513/b1 [api-inclusive]` | 2.93478 ± 0.0152843 (W 0.00954135) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.93047 ± 0.0129139 (W 0.0119871) | 11.1931 ± 0.0395529 (W 0.0117664) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.95414 ± 0.0238368 (W 0.00991435) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.15419 ± 0.0534282 (W 0.00989228) | 11.4679 ± 0.0395029 (W 0.0192946) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 8.14856 ± 0.0114691 (W 0.0108396) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.611 ± 0.0586627 (W 0.0397301) | 57.6638 ± 0.263788 (W 0.0984126) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 35.6815 ± 0.0540565 (W 0.0501483) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.3762 ± 0.0505426 (W 0.0241582) | 59.226 ± 0.179517 (W 0.0697621) | 🔵 faer-fork 1.63× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 36.5289 ± 0.0631071 (W 0.0459621) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.9829 ± 0.0136625 (W 0.0101653) | 6.01197 ± 0.0177555 (W 0.00699335) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [api-inclusive]` | 3.98064 ± 0.00884162 (W 0.0093275) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.11232 ± 0.0110229 (W 0.00710316) | 6.13027 ± 0.0161088 (W 0.0106371) | 🔵 faer-fork 1.49× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [api-inclusive]` | 4.12515 ± 0.00755812 (W 0.00989058) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.4062 ± 0.0699238 (W 0.0197743) | 35.0557 ± 0.179678 (W 0.0370128) | 🔵 faer-fork 2.01× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [api-inclusive]` | 17.4411 ± 0.0528199 (W 0.0253173) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.5913 ± 0.0428395 (W 0.0159377) | 35.7281 ± 0.227591 (W 0.0324319) | 🔵 faer-fork 2.03× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [api-inclusive]` | 17.6392 ± 0.0549134 (W 0.0237307) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.014572 ± 0.000190814 (W 0.000282734) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.0151625 ± 0.000189343 (W 0.000294221) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.0377255 ± 0.000319127 (W 0.00052069) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.037821 ± 0.000578814 (W 0.000391428) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.14544 ± 0.00152525 (W 0.00115708) | 0.315258 ± 0.00122243 (W 0.00208574) | 🔵 faer-fork 2.17× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.183698 ± 0.00118099 (W 0.0014328) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.136569 ± 0.000635989 (W 0.00180563) | 0.25978 ± 0.00173538 (W 0.0017039) | 🔵 faer-fork 1.90× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.173218 ± 0.000775696 (W 0.000823265) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.507796 ± 0.00161029 (W 0.00241066) | 1.03583 ± 0.00550046 (W 0.00320674) | 🔵 faer-fork 2.04× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.69372 ± 0.00146778 (W 0.00250752) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.509566 ± 0.00226779 (W 0.00229701) | 1.03999 ± 0.00253083 (W 0.00299589) | 🔵 faer-fork 2.04× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.670703 ± 0.0010333 (W 0.00356112) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 3.99407 ± 0.00618329 (W 0.00824018) | 4.37813 ± 0.0104589 (W 0.00613026) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `eigh/n255/b1 [api-inclusive]` | 4.00105 ± 0.0101527 (W 0.00765385) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.96052 ± 0.00702551 (W 0.0053887) | 4.39947 ± 0.0112793 (W 0.00962675) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `eigh/n257/b1 [api-inclusive]` | 3.96811 ± 0.0288607 (W 0.0106116) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.1196 ± 0.0402366 (W 0.0218471) | 15.5639 ± 0.0605873 (W 0.0233124) | · |  |
| `eigh/n511/b1 [api-inclusive]` | 16.1419 ± 0.0315102 (W 0.0230174) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.5614 ± 0.051669 (W 0.0180913) | 15.9317 ± 0.0450638 (W 0.0287026) | · |  |
| `eigh/n513/b1 [api-inclusive]` | 16.5909 ± 0.0460338 (W 0.019141) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.10029 ± 0.00723952 (W 0.0029404) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.09771 ± 0.00863345 (W 0.00808402) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.09943 ± 0.00464831 (W 0.00276457) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.40095 ± 0.00539016 (W 0.00412813) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 4.5428 ± 0.0441403 (W 0.00673152) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.3457 ± 0.0424924 (W 0.015192) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.6634 ± 0.0367263 (W 0.0113918) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 5.62663 ± 0.0254001 (W 0.0187762) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.27994 ± 0.00219244 (W 0.00154663) | 0.373134 ± 0.00203629 (W 0.00190765) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 0.584712 ± 0.00795149 (W 0.00264988) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.289982 ± 0.00201723 (W 0.00167677) | 0.37905 ± 0.00190725 (W 0.00244877) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 0.627608 ± 0.00719072 (W 0.00299964) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.13194 ± 0.00346673 (W 0.00380082) | 1.49276 ± 0.0254892 (W 0.00679005) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `lu/n511/b1 [api-inclusive]` | 2.63594 ± 0.0166099 (W 0.00704244) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.17274 ± 0.0135767 (W 0.002831) | 1.50223 ± 0.00823559 (W 0.00660395) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 2.62737 ± 0.0125518 (W 0.00893581) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.797231 ± 0.00358022 (W 0.00252411) | 1.9909 ± 0.00717009 (W 0.00465104) | 🔵 faer-fork 2.50× faster than `lapack-openblas` |  |
| `qr/n255/b1 [api-inclusive]` | 1.28896 ± 0.00694749 (W 0.00659034) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 0.823077 ± 0.025358 (W 0.00274071) | 1.89393 ± 0.00661807 (W 0.00350637) | 🔵 faer-fork 2.30× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/n257/b1 [api-inclusive]` | 1.35847 ± 0.0149039 (W 0.00733362) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 3.1274 ± 0.015965 (W 0.00706703) | 6.7884 ± 0.0116963 (W 0.0106145) | 🔵 faer-fork 2.17× faster than `lapack-openblas` |  |
| `qr/n511/b1 [api-inclusive]` | 5.49186 ± 0.0190488 (W 0.0208448) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 3.49369 ± 0.0100449 (W 0.00887529) | 6.97946 ± 0.0127143 (W 0.0100547) | 🔵 faer-fork 2.00× faster than `lapack-openblas` |  |
| `qr/n513/b1 [api-inclusive]` | 5.93179 ± 0.0223063 (W 0.0166049) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.565843 ± 0.00243754 (W 0.00250858) | 1.09195 ± 0.00658733 (W 0.00175113) | 🔵 faer-fork 1.93× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 0.582968 ± 0.00276642 (W 0.00239157) | 1.0214 ± 0.00381317 (W 0.00115467) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 1.73663 ± 0.00879868 (W 0.00690984) | 3.68481 ± 0.016465 (W 0.00988791) | 🔵 faer-fork 2.12× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 2.03415 ± 0.00751539 (W 0.00558686) | 3.7189 ± 0.0145334 (W 0.00473441) | 🔵 faer-fork 1.83× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.284639 ± 0.00144224 (W 0.00208056) | 0.37516 ± 0.0023734 (W 0.00258179) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 0.593013 ± 0.00503418 (W 0.00354669) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.2935 ± 0.00101476 (W 0.00213921) | 0.379513 ± 0.00158132 (W 0.00201635) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 0.633047 ± 0.00564763 (W 0.00444845) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.09938 ± 0.00320774 (W 0.00363187) | 1.45149 ± 0.0117097 (W 0.00504927) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `solve/n511/b1 [api-inclusive]` | 2.61302 ± 0.0140671 (W 0.00842868) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.14172 ± 0.00415577 (W 0.00321187) | 1.48095 ± 0.0115266 (W 0.0046732) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `solve/n513/b1 [api-inclusive]` | 2.60472 ± 0.0159271 (W 0.00890045) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.23699 ± 0.0224788 (W 0.0217843) | 11.0757 ± 0.0674207 (W 0.0155724) | 🔵 faer-fork 1.53× faster than `lapack-openblas` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.28392 ± 0.0261266 (W 0.0165938) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.37303 ± 0.0221038 (W 0.0124829) | 11.1563 ± 0.0298563 (W 0.025494) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 7.43819 ± 0.0151346 (W 0.00943536) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.3905 ± 0.069465 (W 0.0363732) | 53.3077 ± 0.30248 (W 0.0575961) | 🔵 faer-fork 1.81× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 29.5035 ± 0.0676158 (W 0.0278157) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.8768 ± 0.0813216 (W 0.041847) | 54.8603 ± 0.494327 (W 0.150928) | 🔵 faer-fork 1.84× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 29.9632 ± 0.0553676 (W 0.0214612) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.89279 ± 0.01156 (W 0.00777865) | 6.03981 ± 0.0329152 (W 0.00557978) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [api-inclusive]` | 3.88342 ± 0.01862 (W 0.00476024) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.99201 ± 0.0124925 (W 0.00719993) | 6.10403 ± 0.0152873 (W 0.00684434) | 🔵 faer-fork 1.53× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [api-inclusive]` | 3.97724 ± 0.0139162 (W 0.00435942) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.4236 ± 0.0414139 (W 0.0280689) | 34.7949 ± 0.277601 (W 0.0259687) | 🔵 faer-fork 2.26× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [api-inclusive]` | 15.4311 ± 0.0550602 (W 0.022762) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.4552 ± 0.0419258 (W 0.0244306) | 34.9638 ± 0.288743 (W 0.0206183) | 🔵 faer-fork 2.26× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [api-inclusive]` | 15.515 ± 0.0242207 (W 0.0153211) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.015659 ± 0.000179646 (W 0.00020208) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.0157735 ± 0.000107508 (W 0.000131787) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.0375185 ± 0.000158524 (W 0.000355049) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0378685 ± 0.000445841 (W 0.000321585) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.510149 ± 0.00326072 (W 0.00204643) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.575144 ± 0.00216039 (W 0.00291134) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.464823 ± 0.00164756 (W 0.00183356) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.524712 ± 0.0030463 (W 0.002436) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.29947 ± 0.00246961 (W 0.00254395) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 2.77526 ± 0.0417935 (W 0.0150907) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.24338 ± 0.003604 (W 0.00219179) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 2.69885 ± 0.0100433 (W 0.00367896) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.30791 ± 0.0163725 (W 0.0114453) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 8.3339 ± 0.0300594 (W 0.0192026) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.32801 ± 0.022378 (W 0.0104282) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 8.34203 ± 0.0166999 (W 0.00886565) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.4973 ± 0.0527356 (W 0.0297979) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 40.641 ± 0.110925 (W 0.034601) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.3919 ± 0.0673886 (W 0.0516601) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 40.4812 ± 0.0627493 (W 0.0462931) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.96699 ± 0.0185134 (W 0.00464172) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 4.054 ± 0.0122563 (W 0.00757879) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.01976 ± 0.0111611 (W 0.00775301) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.69812 ± 0.0317861 (W 0.0044779) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 11.085 ± 0.0506976 (W 0.0203938) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 20.5529 ± 0.100674 (W 0.050325) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 11.2642 ± 0.0422086 (W 0.0121994) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 15.0178 ± 0.122005 (W 0.0488611) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.939061 ± 0.00463548 (W 0.00278192) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.71682 ± 0.00865018 (W 0.0103267) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.02303 ± 0.00164275 (W 0.00379891) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.71552 ± 0.00228142 (W 0.006479) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 5.05305 ± 0.0217087 (W 0.0369249) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 9.23145 ± 0.0533568 (W 0.0478544) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 5.30546 ± 0.0280287 (W 0.00767363) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 9.31124 ± 0.0385041 (W 0.0348019) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 3.06145 ± 0.00791046 (W 0.00353958) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 4.20052 ± 0.013248 (W 0.0104474) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.19111 ± 0.0111458 (W 0.00428594) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 4.29699 ± 0.0190507 (W 0.00740315) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 19.7149 ± 0.032033 (W 0.0189071) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 25.2876 ± 0.0820973 (W 0.0772768) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 20.1905 ± 0.0612514 (W 0.0324103) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 25.989 ± 0.11936 (W 0.0294896) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.61431 ± 0.0019026 (W 0.00215427) | - | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.68209 ± 0.0028509 (W 0.00261786) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 9.33927 ± 0.0296355 (W 0.0173384) | - | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 9.58154 ± 0.0106565 (W 0.0175089) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.860374 ± 0.00230331 (W 0.00345381) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.70615 ± 0.015359 (W 0.0116037) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.02519 ± 0.00270327 (W 0.00363723) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.73043 ± 0.00688518 (W 0.00595284) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 5.0874 ± 0.0532694 (W 0.0402261) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 9.32741 ± 0.0356602 (W 0.0570047) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 5.30844 ± 0.00968281 (W 0.00653814) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 9.40755 ± 0.0727392 (W 0.0371774) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.5873 ± 0.0321664 (W 0.0190987) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 12.6651 ± 0.0288558 (W 0.0219832) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 12.8929 ± 0.0567373 (W 0.0198032) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 12.9555 ± 0.0668891 (W 0.0258001) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 67.57 ± 0.279141 (W 0.0901249) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 67.8337 ± 0.190532 (W 0.0996993) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 69.1227 ± 0.209876 (W 0.091102) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 69.4053 ± 0.153178 (W 0.337362) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.0198 ± 0.0334941 (W 0.01705) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 6.00336 ± 0.0333682 (W 0.0184843) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.17279 ± 0.0337343 (W 0.0247692) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 6.17533 ± 0.0166573 (W 0.0113974) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 32.6974 ± 0.164193 (W 0.0666894) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 32.9098 ± 0.166819 (W 0.058935) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 33.1171 ± 0.152195 (W 0.059028) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 33.4014 ± 0.163723 (W 0.081361) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0217295 ± 0.000120192 (W 0.000108619) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.0218645 ± 9.77875e-05 (W 0.00014763) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.055195 ± 0.00165077 (W 0.000302805) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n513/b1` | 0.0549945 ± 0.000635134 (W 0.000294914) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.437128 ± 0.00168738 (W 0.00201888) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.51417 ± 0.00307445 (W 0.00190677) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.39335 ± 0.00226022 (W 0.0016096) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.461307 ± 0.00280813 (W 0.00213887) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.56903 ± 0.00504865 (W 0.00435073) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 2.08462 ± 0.00995857 (W 0.00631659) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.49116 ± 0.00864432 (W 0.00447736) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 1.99894 ± 0.0058989 (W 0.00479378) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.54366 ± 0.01878 (W 0.00636083) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.55571 ± 0.0124941 (W 0.00557391) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.51054 ± 0.0309258 (W 0.00576522) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.5437 ± 0.0239111 (W 0.00656271) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.4441 ± 0.0403056 (W 0.0367916) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 25.4417 ± 0.0533272 (W 0.0189009) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.9152 ± 0.0238248 (W 0.0248407) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 26.0129 ± 0.0738033 (W 0.0269875) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.58957 ± 0.0117651 (W 0.00537346) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.73968 ± 0.0273665 (W 0.013208) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.62069 ± 0.0193268 (W 0.00638815) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.34041 ± 0.010992 (W 0.00386426) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.22954 ± 0.0242375 (W 0.0156838) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 16.9665 ± 0.0625101 (W 0.0322304) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.35801 ± 0.0168122 (W 0.0102492) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 11.347 ± 0.0280337 (W 0.0287545) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.805048 ± 0.00195022 (W 0.00282333) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.53925 ± 0.0129504 (W 0.00796209) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.803127 ± 0.00249906 (W 0.00181928) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.54749 ± 0.00565126 (W 0.00482316) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.34526 ± 0.00983162 (W 0.0071212) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 7.67565 ± 0.0271897 (W 0.0204102) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.47265 ± 0.0210796 (W 0.00941143) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 7.73212 ± 0.0707469 (W 0.0187424) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.05968 ± 0.0073677 (W 0.00420941) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 3.24809 ± 0.00875186 (W 0.00816631) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 2.16048 ± 0.00868516 (W 0.00664488) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 3.34585 ± 0.0206972 (W 0.0101243) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 10.436 ± 0.0280297 (W 0.0205824) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 16.4527 ± 0.0930034 (W 0.0364687) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 11.3043 ± 0.039527 (W 0.0267827) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 17.332 ± 0.0770882 (W 0.0560177) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.29493 ± 0.00713213 (W 0.00336078) | - | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.3534 ± 0.0162766 (W 0.00462945) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 5.02815 ± 0.0168481 (W 0.00818557) | - | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 5.69264 ± 0.0305496 (W 0.0118892) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.718746 ± 0.00312277 (W 0.00228471) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.5544 ± 0.0103566 (W 0.00547477) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.805027 ± 0.00428784 (W 0.00177071) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.56386 ± 0.00609861 (W 0.00373062) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.32706 ± 0.00984249 (W 0.00693404) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 7.717 ± 0.0266804 (W 0.0361643) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.45675 ± 0.0561957 (W 0.00873314) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 7.74701 ± 0.0269734 (W 0.0325276) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.4507 ± 0.0340184 (W 0.0155825) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 10.4983 ± 0.0423187 (W 0.0222812) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.6891 ± 0.0551772 (W 0.0191486) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 10.7119 ± 0.0528554 (W 0.0186878) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 48.1198 ± 0.120839 (W 0.0970904) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 48.4005 ± 0.117767 (W 0.0598188) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 49.2375 ± 0.13553 (W 0.0831207) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 49.407 ± 0.0802926 (W 0.0804641) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.48574 ± 0.0484311 (W 0.0173988) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 5.48334 ± 0.0154222 (W 0.00625802) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.62247 ± 0.0312887 (W 0.0119045) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 5.62942 ± 0.0217334 (W 0.00995158) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 25.2256 ± 0.0695457 (W 0.0532685) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 25.2149 ± 0.143721 (W 0.0575042) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 25.614 ± 0.127325 (W 0.061288) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 25.6743 ± 0.0903424 (W 0.0439515) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0214785 ± 8.60808e-05 (W 9.20306e-05) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.021779 ± 0.000118474 (W 0.000156457) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.0507375 ± 0.000569676 (W 0.000256589) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.051172 ± 0.000733673 (W 0.000277917) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.499613 ± 0.00159739 (W 0.00200767) | 0.553029 ± 0.00103816 (W 0.00176277) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.575385 ± 0.00154312 (W 0.00220722) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.449785 ± 0.0025277 (W 0.00180864) | 0.538359 ± 0.0016461 (W 0.00172936) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.526537 ± 0.00288595 (W 0.00273784) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.26156 ± 0.00377686 (W 0.00378366) | 3.18182 ± 0.00427712 (W 0.00319061) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 2.60868 ± 0.008069 (W 0.00611202) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.21465 ± 0.00364596 (W 0.00470324) | 3.18612 ± 0.00644198 (W 0.00379111) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 2.53933 ± 0.00957989 (W 0.00584318) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.29788 ± 0.0182839 (W 0.00849073) | 12.0803 ± 0.0253147 (W 0.0108175) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `eigh/n255/b1 [api-inclusive]` | 8.32191 ± 0.017971 (W 0.0115203) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.32516 ± 0.0175294 (W 0.00781083) | 12.3506 ± 0.033856 (W 0.0273452) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `eigh/n257/b1 [api-inclusive]` | 8.34436 ± 0.0101722 (W 0.0119592) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.2637 ± 0.0177415 (W 0.0333695) | 56.3055 ± 0.112854 (W 0.0666522) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `eigh/n511/b1 [api-inclusive]` | 40.378 ± 0.0525915 (W 0.046693) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.1511 ± 0.06806 (W 0.0239566) | 56.5872 ± 0.107899 (W 0.0420447) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `eigh/n513/b1 [api-inclusive]` | 40.2575 ± 0.0488687 (W 0.0281381) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.85485 ± 0.0089022 (W 0.0048259) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.89235 ± 0.0126933 (W 0.0114876) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.90211 ± 0.00810929 (W 0.00544769) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.53174 ± 0.016613 (W 0.00678964) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 10.6855 ± 0.0262592 (W 0.0143648) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 19.6389 ± 0.0373569 (W 0.0292727) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 10.8185 ± 0.033659 (W 0.0208618) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 13.8756 ± 0.0698307 (W 0.0304036) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.863665 ± 0.00862461 (W 0.0101162) | 1.13614 ± 0.00349747 (W 0.00265665) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 1.56282 ± 0.0162039 (W 0.0093275) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.906646 ± 0.00237572 (W 0.00344614) | 1.15139 ± 0.00334136 (W 0.00293537) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 1.57045 ± 0.00552136 (W 0.00403168) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 4.58204 ± 0.0314495 (W 0.00947014) | 5.01095 ± 0.00732729 (W 0.00458878) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `lu/n511/b1 [api-inclusive]` | 8.20771 ± 0.115813 (W 0.0518246) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 4.77781 ± 0.00668351 (W 0.0103244) | 5.05257 ± 0.0119619 (W 0.00540962) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 8.21931 ± 0.0433461 (W 0.012619) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.96726 ± 0.00534081 (W 0.00440091) | 4.8079 ± 0.00697798 (W 0.00281048) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `qr/n255/b1 [api-inclusive]` | 4.04066 ± 0.0115327 (W 0.0116411) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.05097 ± 0.00148087 (W 0.00406686) | 5.27244 ± 0.0112704 (W 0.00884984) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `qr/n257/b1 [api-inclusive]` | 4.16405 ± 0.0318131 (W 0.0122929) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 19.0174 ± 0.0420935 (W 0.0247289) | 26.9462 ± 0.0348465 (W 0.0446631) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `qr/n511/b1 [api-inclusive]` | 24.2739 ± 0.0379705 (W 0.0662215) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 19.3955 ± 0.0486976 (W 0.0222782) | 28.6253 ± 0.037794 (W 0.0164537) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `qr/n513/b1 [api-inclusive]` | 24.9171 ± 0.0974599 (W 0.0508395) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.52069 ± 0.00168825 (W 0.00247708) | 2.44256 ± 0.00424157 (W 0.00396836) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 1.56692 ± 0.00142059 (W 0.00247377) | 2.66154 ± 0.00547938 (W 0.00373497) | 🔵 faer-fork 1.70× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 8.83773 ± 0.00978716 (W 0.0127632) | 12.1139 ± 0.0180867 (W 0.0274172) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 8.97789 ± 0.0123473 (W 0.00988721) | 12.8687 ± 0.023589 (W 0.00952692) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.86941 ± 0.013089 (W 0.00397915) | 1.11718 ± 0.00218129 (W 0.00233811) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 1.56954 ± 0.00894488 (W 0.0104965) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.913449 ± 0.00421343 (W 0.00270534) | 1.13628 ± 0.00359876 (W 0.00239729) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 1.59416 ± 0.0211364 (W 0.00579319) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 4.57261 ± 0.0170019 (W 0.0241855) | 4.97729 ± 0.0150893 (W 0.0118333) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `solve/n511/b1 [api-inclusive]` | 8.27441 ± 0.0551423 (W 0.0512623) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 4.79566 ± 0.00643759 (W 0.00557841) | 5.02395 ± 0.0204358 (W 0.0100675) | · |  |
| `solve/n513/b1 [api-inclusive]` | 8.31557 ± 0.0774364 (W 0.0316574) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.4936 ± 0.0232502 (W 0.0186997) | 17.9169 ± 0.0449073 (W 0.0267219) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 12.563 ± 0.0265222 (W 0.0220512) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 12.8349 ± 0.0430635 (W 0.018917) | 18.7506 ± 0.0532439 (W 0.0244477) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 12.8768 ± 0.0370389 (W 0.0189937) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 67.1945 ± 0.120298 (W 0.0802527) | 87.2188 ± 0.159937 (W 0.129325) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 67.5619 ± 0.186262 (W 0.0983525) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 68.797 ± 0.214699 (W 0.0805385) | 90.4245 ± 0.24644 (W 0.0565329) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 68.9794 ± 0.161451 (W 0.380453) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.0103 ± 0.0114817 (W 0.0172104) | 9.51212 ± 0.034297 (W 0.0135663) | 🔵 faer-fork 1.58× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [api-inclusive]` | 6.03725 ± 0.0123381 (W 0.0160823) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.18779 ± 0.0401614 (W 0.014976) | 9.94571 ± 0.0386195 (W 0.0120097) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [api-inclusive]` | 6.20673 ± 0.0274007 (W 0.0118978) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 32.7182 ± 0.0943925 (W 0.072274) | 42.4855 ± 0.136113 (W 0.0272253) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [api-inclusive]` | 32.6067 ± 0.0765192 (W 0.0548774) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 32.9876 ± 0.120757 (W 0.0946434) | 43.5978 ± 0.154608 (W 0.0493298) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [api-inclusive]` | 33.1305 ± 0.113877 (W 0.0710759) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.020537 ± 0.000204799 (W 0.000126781) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.0207925 ± 0.000175195 (W 0.00018454) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.055706 ± 0.000843116 (W 0.000307691) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.056043 ± 0.000531786 (W 0.000347744) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.43429 ± 0.00153499 (W 0.0021358) | 0.813331 ± 0.00286765 (W 0.00171267) | 🔵 faer-fork 1.87× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.509322 ± 0.00134388 (W 0.001795) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.384586 ± 0.00142826 (W 0.0018337) | 0.476172 ± 0.00228783 (W 0.00267067) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.462203 ± 0.00245683 (W 0.00214563) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.56457 ± 0.00587728 (W 0.00304785) | 2.36348 ± 0.0086075 (W 0.00719178) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 1.92913 ± 0.00887574 (W 0.0046221) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.48745 ± 0.0058896 (W 0.00305915) | 2.03554 ± 0.00555001 (W 0.00481861) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 1.83227 ± 0.0124625 (W 0.00581465) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.53397 ± 0.025119 (W 0.00752065) | 10.7464 ± 0.0323471 (W 0.0169326) | 🔵 faer-fork 1.94× faster than `lapack-openblas` |  |
| `eigh/n255/b1 [api-inclusive]` | 5.56806 ± 0.0211048 (W 0.00597313) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.51523 ± 0.0233256 (W 0.00515258) | 11.0183 ± 0.0527687 (W 0.0156941) | 🔵 faer-fork 2.00× faster than `lapack-openblas` |  |
| `eigh/n257/b1 [api-inclusive]` | 5.53314 ± 0.0260246 (W 0.00716538) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.1659 ± 0.0515985 (W 0.0449765) | 42.8038 ± 0.103855 (W 0.0485419) | 🔵 faer-fork 1.70× faster than `lapack-openblas` |  |
| `eigh/n511/b1 [api-inclusive]` | 25.1764 ± 0.0522444 (W 0.0285241) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.6808 ± 0.0517427 (W 0.0428091) | 43.0899 ± 0.149198 (W 0.210968) | 🔵 faer-fork 1.68× faster than `lapack-openblas` |  |
| `eigh/n513/b1 [api-inclusive]` | 25.7664 ± 0.0374331 (W 0.0365501) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.50564 ± 0.00698706 (W 0.00341261) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.55283 ± 0.0102419 (W 0.00923151) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.53879 ± 0.0389121 (W 0.00426253) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.17861 ± 0.0197295 (W 0.00530989) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.03974 ± 0.0253345 (W 0.0132386) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 16.0106 ± 0.0371939 (W 0.0296012) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.11724 ± 0.0234097 (W 0.0124015) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 10.279 ± 0.0456189 (W 0.029675) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.699705 ± 0.00248395 (W 0.00177021) | 0.762162 ± 0.00403654 (W 0.00360068) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 1.39126 ± 0.00445439 (W 0.00535744) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.72085 ± 0.00311653 (W 0.00206493) | 0.769612 ± 0.00444828 (W 0.00305718) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 1.4061 ± 0.0124171 (W 0.00612899) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.07878 ± 0.0138497 (W 0.00509073) | 2.75002 ± 0.00642352 (W 0.00732994) | 🔴 faer-fork 1.12× slower than `lapack-openblas` |  |
| `lu/n511/b1 [api-inclusive]` | 6.6324 ± 0.0614376 (W 0.0232091) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.14478 ± 0.0330395 (W 0.0056361) | 2.78115 ± 0.00961745 (W 0.00435698) | 🔴 faer-fork 1.13× slower than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 6.61716 ± 0.0568442 (W 0.0164322) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.01521 ± 0.00406921 (W 0.00536387) | 3.83169 ± 0.0148026 (W 0.00484656) | 🔵 faer-fork 1.90× faster than `lapack-openblas` |  |
| `qr/n255/b1 [api-inclusive]` | 3.103 ± 0.0367179 (W 0.0122772) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 2.07026 ± 0.0079097 (W 0.00477768) | 4.18885 ± 0.0239051 (W 0.00520597) | 🔵 faer-fork 2.02× faster than `lapack-openblas` |  |
| `qr/n257/b1 [api-inclusive]` | 3.18297 ± 0.0125549 (W 0.0101122) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 9.9659 ± 0.0242697 (W 0.0255837) | 17.8613 ± 0.19156 (W 0.0147104) | 🔵 faer-fork 1.79× faster than `lapack-openblas` |  |
| `qr/n511/b1 [api-inclusive]` | 15.3592 ± 0.0545953 (W 0.0297614) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 10.6851 ± 0.0322238 (W 0.013064) | 19.214 ± 0.0711615 (W 0.0206638) | 🔵 faer-fork 1.80× faster than `lapack-openblas` |  |
| `qr/n513/b1 [api-inclusive]` | 16.3307 ± 0.0547355 (W 0.0237047) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.24176 ± 0.00340546 (W 0.00323951) | 2.17596 ± 0.00833815 (W 0.00471818) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `qr_factor/n257/b1` | 1.27288 ± 0.00475763 (W 0.00306914) | 2.32695 ± 0.00821713 (W 0.00274285) | 🔵 faer-fork 1.83× faster than `lapack-openblas` |  |
| `qr_factor/n511/b1` | 4.70332 ± 0.0157843 (W 0.0104715) | 8.90011 ± 0.0340032 (W 0.0137285) | 🔵 faer-fork 1.89× faster than `lapack-openblas` |  |
| `qr_factor/n513/b1` | 5.28814 ± 0.0278658 (W 0.00903613) | 9.27777 ± 0.0532867 (W 0.0173982) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `solve/n255/b1` | 0.700705 ± 0.00139302 (W 0.00181775) | 0.75741 ± 0.00308492 (W 0.00281553) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 1.4107 ± 0.00350745 (W 0.00515023) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.725619 ± 0.00352818 (W 0.00240744) | 0.766819 ± 0.00550246 (W 0.0026471) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 1.42069 ± 0.0100762 (W 0.0046036) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.0685 ± 0.0128976 (W 0.00676414) | 2.76515 ± 0.0123786 (W 0.00445102) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `solve/n511/b1 [api-inclusive]` | 6.64974 ± 0.0390244 (W 0.0206595) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.1499 ± 0.00763734 (W 0.00365596) | 2.78392 ± 0.0144381 (W 0.00639959) | 🔴 faer-fork 1.13× slower than `lapack-openblas` |  |
| `solve/n513/b1 [api-inclusive]` | 6.70223 ± 0.0567656 (W 0.0184212) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.3891 ± 0.0266526 (W 0.0140888) | 15.8102 ± 0.0563317 (W 0.02213) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 10.4407 ± 0.0376848 (W 0.0167951) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.6046 ± 0.0399326 (W 0.0248436) | 16.5368 ± 0.0690986 (W 0.0223571) | 🔵 faer-fork 1.56× faster than `lapack-openblas` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 10.6377 ± 0.0339371 (W 0.0129948) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 47.8509 ± 0.151109 (W 0.0500578) | 67.0363 ± 0.167582 (W 0.0522209) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 48.227 ± 0.0924217 (W 0.0924074) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 48.9726 ± 0.122904 (W 0.0954168) | 69.7664 ± 0.26097 (W 0.0703928) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 49.2831 ± 0.191874 (W 0.0480503) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.50232 ± 0.0152884 (W 0.0082898) | 8.90049 ± 0.0405273 (W 0.0105327) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `svd_values/n255/b1 [api-inclusive]` | 5.50639 ± 0.0209018 (W 0.0150471) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.65984 ± 0.0298625 (W 0.011083) | 9.26522 ± 0.0410389 (W 0.0101778) | 🔵 faer-fork 1.64× faster than `lapack-openblas` |  |
| `svd_values/n257/b1 [api-inclusive]` | 5.67732 ± 0.0288698 (W 0.0153724) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 25.0707 ± 0.122045 (W 0.0573433) | 34.962 ± 0.0725557 (W 0.0195154) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `svd_values/n511/b1 [api-inclusive]` | 25.1023 ± 0.078773 (W 0.0606898) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 25.3211 ± 0.158281 (W 0.292875) | 35.7318 ± 0.166069 (W 0.0292661) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `svd_values/n513/b1 [api-inclusive]` | 25.4477 ± 0.106103 (W 0.0643754) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.022736 ± 0.000148293 (W 0.000164938) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.0229415 ± 0.000255281 (W 0.000146514) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.0563325 ± 0.00161672 (W 0.000316293) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n513/b1` | 0.0569275 ± 0.000568619 (W 0.000309823) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/255`, `cholesky/257`, `cholesky/511`, `cholesky/513`, `eigh/255`, `eigh/257`, `eigh/511`, `eigh/513`, `full_piv_lu/255`, `full_piv_lu/257`, `full_piv_lu/511`, `full_piv_lu/513`, `lu/255`, `lu/257`, `lu/511`, `lu/513`, `qr/255`, `qr/257`, `qr/511`, `qr/513`, `qr_factor/255`, `qr_factor/257`, `qr_factor/511`, `qr_factor/513`, `solve/255`, `solve/257`, `solve/511`, `solve/513`, `svd_thin/255`, `svd_thin/257`, `svd_thin/511`, `svd_thin/513`, `svd_values/255`, `svd_values/257`, `svd_values/511`, `svd_values/513`, `triangular_solve/255`, `triangular_solve/257`, `triangular_solve/511`, `triangular_solve/513`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
