# `faer-kernels-large` on `zen5-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `e7a1baa48af15fb7767986acb7501818796ec8ec`
- toolkit commit: `ec1366c391cbd5023116ddcb6fdadc9c5b76f582`
- measured binary SHA-256: `6ba8fea0520b48e8b9c3541aef0d4d9f4dda98e9b9d102866e3f862e9a1ad118`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-21T11:25:37.042522Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu faer-kernels-large --vendor aocl --engines faer-fork,lapack-aocl --threads 4,8 --regimes linalg-large,gemm-large,linalg-boundary --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 --reps 5 --aa 6 --prime-ms 250 --label pub-93da6217-20261120-large-aocl`
- raw data: `data/results/zen5-cpu/faer-kernels-large/20261121T034114Z-pub-93da6217-20261120-large-aocl/`

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
  - `lapack-aocl` — The same calls against AMD AOCL, the vendor-native LAPACK stack on AMD hosts: LAPACKE from AOCL-LAPACK (`libflame`), CBLAS from the threaded AOCL-BLAS (`libblis-mt`), linked from the versioned tree `AOCL_ROOT` names and loaded through the rpath the build records. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run. Measured on AMD profiles only.
    Identity: the AOCL build recorded in this run's manifest (`providers[aocl]`)
    Caveat: One AOCL call is governed by three per-thread counts (OpenMP, BLIS, libflame); the harness sets the budget through OpenMP and refuses the cell unless all three read back equal to it, before each row and after its timed loop. The manifest records both libraries' versions, the files the loader mapped, and the dispatched kernels (`blis=<config>, flame=<isa>`). Threading is vendor-owned, as for OpenBLAS and MKL.
    As measured: linked shared; aocl 6.0.0; kernel dispatched to `blis=zen5, flame=avx512dq`; openmp threading; config `AOCL-BLAS 6.0.0 Build 20261026; AOCL-LAPACK 6.0.0 Build 20261026, supports LAPACK 3.12.1 w/o Index-64 interfaces; mapped=/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libaoclutils.so:/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libblis-mt.so.6.0.0:/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libflame.so:/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`

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
Spread guide (computed from this run's own rows): per-engine between-start range median 0.97%, worst 74.53% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 13 of 1088 engine cells exceed 5% and carry `!`; median within-process relative SD 0.25%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 5.02499 ± 0.0217497 (W 0.0103922) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 5.42886 ± 0.0403729 (W 0.0158964) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 14.383 ± 0.03367 (W 0.0190103) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 16.7364 ± 0.168907 (W 0.110764) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 120.118 ± 0.452627 (W 0.0856343) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 127.032 ± 0.245983 (W 0.328686) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 327.089 ± 1.20637 (W 0.485995) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 342.154 ± 1.37999 (W 1.04448) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 59.0908 ± 0.229206 (W 0.208186) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 66.6676 ± 0.153327 (W 0.292885) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 250.426 ± 1.03783 (W 1.45522) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 270.378 ± 1.08784 (W 1.3805) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 10.9011 ± 0.0334228 (W 0.0372164) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 19.0006 ± 0.110994 (W 0.0561039) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 31.9108 ± 0.0970845 (W 0.147825) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 49.4869 ± 0.390461 (W 0.230523) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 39.6928 ± 0.347327 (W 0.0754676) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 50.7557 ± 0.227149 (W 0.0963757) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 131.383 ± 0.546042 (W 0.428614) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 154.228 ± 0.937729 (W 0.498065) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 19.3465 ± 0.0716276 (W 0.0349695) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 58.8274 ± 0.188138 (W 0.148135) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 10.7438 ± 0.106049 (W 0.103308) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 12.1069 ± 0.0490195 (W 0.0598796) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 30.9127 ± 0.0414571 (W 0.0525713) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 34.4764 ± 0.226038 (W 0.243673) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 204.991 ± 0.624672 (W 0.276721) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 226.327 ± 0.685802 (W 0.476779) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 701.91 ± 2.42054 (W 1.14476) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 748.269 ± 2.22892 (W 1.85007) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 105.891 ± 0.761252 (W 0.271473) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 105.467 ± 0.538071 (W 0.213424) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 408.117 ± 1.02664 (W 1.11771) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 417.182 ± 1.40215 (W 1.41227) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.09085 ± 0.00159206 (W 0.000863976) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.236051 ± 0.00120533 (W 0.00517004) | - | - | `*` one row has no measurement here |

## linalg-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 3.16102 ± 0.00754087 (W 0.00692996) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 3.90774 ± 0.0193828 (W 0.0135357) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 8.17913 ± 0.0284265 (W 0.0124197) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 10.8349 ± 0.0547971 (W 0.0507257) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 74.4553 ± 0.045291 (W 0.0704067) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 81.0124 ± 0.368313 (W 0.202469) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 192.003 ± 0.545525 (W 0.215007) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 205.674 ± 0.582902 (W 0.405689) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 45.4299 ± 0.133545 (W 0.0623442) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 54.4063 ± 0.450382 (W 0.694771) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 146.747 ± 0.28603 (W 0.475402) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 169.007 ± 0.78643 (W 0.971283) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 6.85554 ± 0.0402341 (W 0.0317578) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 16.0637 ± 0.136 (W 0.103246) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 19.7917 ± 0.112805 (W 0.169764) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 39.7538 ± 0.112143 (W 0.181899) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 22.0561 ± 0.306235 (W 0.0899641) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 33.2393 ± 0.176177 (W 0.121777) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 71.0713 ± 0.130177 (W 0.221449) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 98.6958 ± 0.142831 (W 0.14802) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 10.5108 ± 0.0457807 (W 0.0129857) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 31.9805 ± 0.0641934 (W 0.0896578) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 6.82795 ± 0.0474926 (W 0.0299266) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 8.94137 ± 0.0273965 (W 0.0719268) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 19.4651 ± 0.0437049 (W 0.0688795) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 24.2186 ± 0.17428 (W 0.185552) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 145.618 ± 0.14505 (W 0.31845) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 166.187 ± 0.535884 (W 0.446096) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 445.197 ± 0.895421 (W 0.553563) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 489.812 ± 1.49643 (W 1.29442) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 78.458 ± 0.830171 (W 0.202591) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 78.588 ± 0.281011 (W 0.0669762) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 259.726 ± 0.801805 (W 0.986327) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 266.452 ± 1.34468 (W 1.00735) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.074223 ± 0.00247446 (W 0.000527512) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n1536/b1` | 0.159612 ± 0.000903204 (W 0.00299146) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 4.03718 ± 0.0087341 (W 0.00889273) | 3.57912 ± 0.00725365 (W 0.0277888) | 🔴 faer-fork 1.13× slower than `lapack-aocl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 5.03556 ± 0.0150583 (W 0.017092) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 12.607 ± 0.0713727 (W 0.0288225) | 11.3379 ± 0.0128861 (W 0.106591) | 🔴 faer-fork 1.11× slower than `lapack-aocl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 15.0487 ± 0.0555474 (W 0.0323769) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 119.615 ± 0.194363 (W 0.0975913) | 150.204 ± 0.327914 (W 0.498159) | 🔵 faer-fork 1.26× faster than `lapack-aocl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 126.278 ± 0.291518 (W 0.441748) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 325.559 ± 1.08532 (W 0.409823) | 453.938 ± 1.95246 (W 1.04189) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 340.329 ± 0.883494 (W 0.830114) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 61.6172 ± 0.163674 (W 0.0546123) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 64.2986 ± 0.318342 (W 0.144036) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 261.895 ± 1.62395 (W 0.955249) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 263.127 ± 0.918727 (W 1.07206) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 8.84824 ± 0.0254224 (W 0.0175467) | 16.4568 ± 0.0390742 (W 0.025292) | 🔵 faer-fork 1.86× faster than `lapack-aocl` |  |
| `lu/n1024/b1 [api-inclusive]` | 16.4809 ± 0.0748169 (W 0.0357961) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 26.5062 ± 0.0796276 (W 0.0276363) | 41.3507 ± 0.106228 (W 0.0549662) | 🔵 faer-fork 1.56× faster than `lapack-aocl` |  |
| `lu/n1536/b1 [api-inclusive]` | 43.8471 ± 0.197749 (W 0.159359) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 37.1737 ± 0.129591 (W 0.119147) | 60.9391 ± 0.140494 (W 0.146468) | 🔵 faer-fork 1.64× faster than `lapack-aocl` |  |
| `qr/n1024/b1 [api-inclusive]` | 48.1486 ± 0.165845 (W 0.09356) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 120.044 ± 0.383142 (W 0.281096) | 187.757 ± 0.387078 (W 0.341608) | 🔵 faer-fork 1.56× faster than `lapack-aocl` |  |
| `qr/n1536/b1 [api-inclusive]` | 148.714 ± 1.15153 (W 0.372612) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 17.1797 ± 0.0585878 (W 0.0233722) | 25.756 ± 0.0656157 (W 0.0246859) | 🔵 faer-fork 1.50× faster than `lapack-aocl` |  |
| `qr_factor/n1536/b1` | 52.8594 ± 0.221453 (W 0.0592503) | 75.2567 ± 0.219281 (W 0.0841944) | 🔵 faer-fork 1.42× faster than `lapack-aocl` |  |
| `solve/n1024/b1` | 8.39403 ± 0.0207953 (W 0.0181868) | 8.60062 ± 0.0536266 (W 0.0134562) | · |  |
| `solve/n1024/b1 [api-inclusive]` | 9.92077 ± 0.0219318 (W 0.0144983) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 25.5915 ± 0.045832 (W 0.0297488) | 25.8968 ± 0.0742185 (W 0.0313263) | · |  |
| `solve/n1536/b1 [api-inclusive]` | 29.3179 ± 0.0269591 (W 0.0235759) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 198.192 ± 0.25733 (W 0.280823) | 321.007 ± 1.38474 (W 0.973479) | 🔵 faer-fork 1.62× faster than `lapack-aocl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 219.68 ± 0.761265 (W 0.518606) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 677.449 ± 1.68744 (W 0.871807) | 981.246 ± 3.10158 (W 2.6726) | 🔵 faer-fork 1.45× faster than `lapack-aocl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 725.01 ± 2.41639 (W 1.71328) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 97.1049 ± 0.494243 (W 0.131229) | 163.683 ± 0.761091 (W 0.66105) | 🔵 faer-fork 1.69× faster than `lapack-aocl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 97.3811 ± 0.451733 (W 0.153338) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 382.674 ± 0.679454 (W 0.926112) | 528.364 ± 3.1981 (W 1.37434) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 391.087 ± 1.11363 (W 1.01727) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.109175 ± 0.00102326 (W 0.00144241) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.258422 ± 0.0029737 (W 0.00526752) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 2.44367 ± 0.00993904 (W 0.00563729) | 1.94354 ± 0.00787498 (W 0.0189238) | 🔴 faer-fork 1.26× slower than `lapack-aocl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 3.46396 ± 0.0370152 (W 0.0090657) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 7.00806 ± 0.0418056 (W 0.0207389) | 6.04537 ± 0.0219985 (W 0.0436534) | 🔴 faer-fork 1.16× slower than `lapack-aocl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 9.25355 ± 0.0225768 (W 0.0188705) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 73.8324 ± 0.127983 (W 0.0774773) | 126.943 ± 0.643648 (W 0.668114) | 🔵 faer-fork 1.72× faster than `lapack-aocl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 80.4579 ± 0.145934 (W 0.269692) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 190.278 ± 0.214401 (W 0.535852) | 389.229 ± 2.13947 (W 1.40067) | 🔵 faer-fork 2.05× faster than `lapack-aocl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 204.291 ± 0.369672 (W 0.722665) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 46.6482 ± 0.165731 (W 0.0838656) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 51.7887 ± 0.121676 (W 0.0614579) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 149.094 ± 0.456452 (W 0.364492) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 162.16 ± 1.06978 (W 0.962229) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 5.77094 ± 0.0124036 (W 0.0144704) | 8.88234 ± 0.0417328 (W 0.0158077) | 🔵 faer-fork 1.54× faster than `lapack-aocl` |  |
| `lu/n1024/b1 [api-inclusive]` | 13.3471 ± 0.111688 (W 0.0303151) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 16.3214 ± 0.041474 (W 0.0449547) | 23.3889 ± 0.0402095 (W 0.0475879) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `lu/n1536/b1 [api-inclusive]` | 33.6847 ± 0.142102 (W 0.137293) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 19.7107 ± 0.0609278 (W 0.0215158) | 34.4576 ± 0.120414 (W 0.0843746) | 🔵 faer-fork 1.75× faster than `lapack-aocl` |  |
| `qr/n1024/b1 [api-inclusive]` | 30.6599 ± 0.0558692 (W 0.0388303) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 63.8509 ± 0.137634 (W 0.059084) | 114.644 ± 0.657843 (W 0.551406) | 🔵 faer-fork 1.80× faster than `lapack-aocl` |  |
| `qr/n1536/b1 [api-inclusive]` | 91.6456 ± 0.897295 (W 0.331351) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 9.04984 ± 0.0330251 (W 0.0191098) | 14.6474 ± 0.0877343 (W 0.0293311) | 🔵 faer-fork 1.62× faster than `lapack-aocl` |  |
| `qr_factor/n1536/b1` | 28.2602 ± 0.1145 (W 0.0499952) | 40.6856 ± 0.10648 (W 0.108193) | 🔵 faer-fork 1.44× faster than `lapack-aocl` |  |
| `solve/n1024/b1` | 5.58267 ± 0.0333644 (W 0.0116087) | 5.10609 ± 0.0256688 (W 0.014594) | 🔴 faer-fork 1.09× slower than `lapack-aocl` |  |
| `solve/n1024/b1 [api-inclusive]` | 6.81335 ± 0.0136247 (W 0.0169145) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 15.8767 ± 0.052841 (W 0.0291046) | 14.9748 ± 0.0509362 (W 0.0363017) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `solve/n1536/b1 [api-inclusive]` | 18.9536 ± 0.0372396 (W 0.0374446) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 138.964 ± 0.195293 (W 0.464314) | 281.871 ± 3.0884 (W 1.48503) | 🔵 faer-fork 2.03× faster than `lapack-aocl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 159.294 ± 0.311425 (W 0.554564) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 420.085 ± 0.597604 (W 0.893162) | 844.992 ± 1.25067 (W 1.82074) | 🔵 faer-fork 2.01× faster than `lapack-aocl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 465.313 ± 1.56871 (W 1.18717) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 69.8778 ± 0.191334 (W 0.0450187) | 153.299 ± 0.944687 (W 0.645335) | 🔵 faer-fork 2.19× faster than `lapack-aocl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 69.9413 ± 0.16423 (W 0.0729561) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 234.223 ± 0.547632 (W 0.707661) | 487.365 ± 1.81226 (W 0.560747) | 🔵 faer-fork 2.08× faster than `lapack-aocl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 243.032 ± 0.566571 (W 0.864232) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.0907115 ± 0.000853679 (W 0.000859716) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.173476 ± 0.00119666 (W 0.00222833) | - | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 14.834 ± 0.0478333 (W 0.0211022) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 18.1986 ± 0.114884 (W 0.0552831) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 47.1582 ± 0.16441 (W 0.117302) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 70.3132 ± 0.229617 (W 0.276418) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 236.397 ± 0.327996 (W 0.620696) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 246.387 ± 0.236781 (W 0.706011) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 783.34 ± 1.64054 (W 1.61293) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 820.962 ± 0.798879 (W 1.7247) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 136.622 ± 0.435123 (W 1.07626) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 154.194 ± 1.51417 (W 1.02202) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 684.757 ± 3.52161 (W 4.28583) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 683.397 ± 7.37266 (W 1.16593) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 33.7487 ± 0.264677 (W 0.197047) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 50.7495 ± 0.190447 (W 0.140508) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 107.24 ± 0.440086 (W 0.535589) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 138.086 ± 0.692907 (W 0.356095) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 142.669 ± 0.279701 (W 0.224184) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 166.466 ± 0.355938 (W 0.306134) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 465.538 ± 0.658156 (W 0.620372) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 510.914 ± 0.746373 (W 0.84524) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 63.6169 ± 0.156931 (W 0.122741) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 206.667 ± 0.955569 (W 0.527438) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 33.5887 ± 0.247299 (W 0.211799) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 51.0346 ± 0.486504 (W 0.205723) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 106.763 ± 0.743472 (W 0.805801) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 137.745 ± 0.719982 (W 0.525368) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 512.6 ± 1.37691 (W 1.08545) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 555.989 ± 2.06949 (W 1.42856) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1836.81 ± 2.9651 (W 2.76644) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1905.18 ± 10.6353 (W 3.86842) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 290.281 ± 0.93995 (W 0.827537) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 291.766 ± 1.27889 (W 3.00418) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1144.72 ± 2.83621 (W 1.60205) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 1161.66 ± 3.33876 (W 2.53376) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.193366 ± 0.00229014 (W 0.00322424) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.554875 ± 0.0042022 (W 0.00459209) | - | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.2812 ± 0.0214271 (W 0.0111733) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 11.7421 ± 0.209457 (W 0.051534) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 24.9311 ± 0.0787033 (W 0.0242555) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 48.6577 ± 0.393052 (W 0.136867) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 138.18 ± 0.234801 (W 0.279483) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 147.903 ± 0.783919 (W 0.596037) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 426.74 ± 0.946488 (W 1.45816) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 464.51 ± 0.747573 (W 2.10196) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 77.3436 ± 0.548397 (W 0.168099) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 95.8238 ± 0.348396 (W 0.364695) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 483.275 ± 2.86493 (W 3.82923) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 510.611 ± 3.60974 (W 2.70255) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 21.002 ± 0.0583266 (W 0.0374945) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 39.07 ± 0.165773 (W 0.126274) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 64.3417 ± 0.289366 (W 0.185495) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 98.2884 ± 0.528629 (W 0.45869) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 78.3972 ± 0.123572 (W 0.290157) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 101.449 ± 0.316714 (W 0.258226) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 255.162 ± 0.358153 (W 0.406562) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 300.147 ± 0.954415 (W 0.514042) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 34.9622 ± 0.108476 (W 0.0332508) | - | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 114.708 ± 0.417586 (W 0.317353) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 21.5652 ± 0.149654 (W 0.0740612) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 25.5849 ± 0.107393 (W 0.0655937) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 64.9148 ± 0.445858 (W 0.40894) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 98.3528 ± 0.571735 (W 0.38916) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 307.047 ± 1.07185 (W 0.944089) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 350.892 ± 1.34416 (W 1.05653) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1213.48 ± 2.06718 (W 2.20976) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1290.45 ± 4.81748 (W 4.19803) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 175.4 ± 0.94912 (W 1.28162) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 174.813 ± 0.377065 (W 0.733333) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 811.824 ± 7.09581 (W 2.92042) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 830.563 ± 2.4256 (W 3.49781) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.132088 ± 0.00152387 (W 0.000603608) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.386321 ± 0.00652336 (W 0.00622322) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 14.6619 ± 0.0300615 (W 0.0252638) | 15.4316 ± 0.0555501 (W 0.0290192) | 🔵 faer-fork 1.05× faster than `lapack-aocl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 16.558 ± 0.0708505 (W 0.025314) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 47.0449 ± 0.0948246 (W 0.0554744) | 46.7297 ± 0.0869563 (W 0.084846) | · |  |
| `cholesky/n1536/b1 [api-inclusive]` | 66.3732 ± 0.11842 (W 0.0816212) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 234.549 ± 0.165346 (W 0.379868) | 613.836 ± 2.27689 (W 1.40711) | 🔵 faer-fork 2.62× faster than `lapack-aocl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 244.751 ± 0.404639 (W 0.652243) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 780.758 ± 1.50928 (W 1.12434) | 2464.66 ± 6.85863 (W 3.14531) | 🔵 faer-fork 3.16× faster than `lapack-aocl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 819.027 ± 0.575478 (W 1.09669) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 141.568 ± 0.738466 (W 0.554401) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 148.855 ± 0.907062 (W 0.516639) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 673.078 ± 4.72684 (W 2.15479) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 677.429 ± 4.74426 (W 1.93605) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 30.0806 ± 0.103468 (W 0.11096) | 41.2038 ± 0.112538 (W 0.0540899) | 🔵 faer-fork 1.37× faster than `lapack-aocl` |  |
| `lu/n1024/b1 [api-inclusive]` | 45.4811 ± 0.13812 (W 0.143647) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 93.1596 ± 0.185524 (W 0.0538329) | 135.378 ± 0.705926 (W 0.239019) | 🔵 faer-fork 1.45× faster than `lapack-aocl` |  |
| `lu/n1536/b1 [api-inclusive]` | 127.119 ± 1.00328 (W 0.296967) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 137.614 ± 0.25891 (W 0.245588) | 206.619 ± 0.455293 (W 0.412613) | 🔵 faer-fork 1.50× faster than `lapack-aocl` |  |
| `qr/n1024/b1 [api-inclusive]` | 160.805 ± 0.409343 (W 0.232179) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 448.318 ± 0.867903 (W 0.502957) | 656.802 ± 2.62867 (W 0.57782) | 🔵 faer-fork 1.47× faster than `lapack-aocl` |  |
| `qr/n1536/b1 [api-inclusive]` | 499.218 ± 0.806868 (W 0.560462) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 59.7785 ± 0.190215 (W 0.230714) | 104.161 ± 0.448763 (W 0.211169) | 🔵 faer-fork 1.74× faster than `lapack-aocl` |  |
| `qr_factor/n1536/b1` | 191.482 ± 0.269228 (W 0.360301) | 340.541 ± 2.75928 (W 0.198059) | 🔵 faer-fork 1.78× faster than `lapack-aocl` |  |
| `solve/n1024/b1` | 30.0562 ± 0.162184 (W 0.177552) | 35.4127 ± 0.156478 (W 0.0359695) | 🔵 faer-fork 1.18× faster than `lapack-aocl` |  |
| `solve/n1024/b1 [api-inclusive]` | 45.4998 ± 0.26934 (W 0.205499) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 93.3619 ± 0.186705 (W 0.15513) | 121.715 ± 0.357257 (W 0.150884) | 🔵 faer-fork 1.30× faster than `lapack-aocl` |  |
| `solve/n1536/b1 [api-inclusive]` | 126.136 ± 0.921907 (W 0.147215) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 506.694 ± 1.58389 (W 0.830571) | 672.619 ± 1.89741 (W 1.95527) | 🔵 faer-fork 1.33× faster than `lapack-aocl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 548.763 ± 1.69913 (W 1.89261) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1812.16 ± 8.23731 (W 1.70943) | 2104.66 ± 2.35413 (W 2.36218) | 🔵 faer-fork 1.16× faster than `lapack-aocl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1891.22 ± 7.15805 (W 3.07274) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 283.066 ± 1.00772 (W 0.411207) | 359.347 ± 4.75461 (W 1.93011) | 🔵 faer-fork 1.27× faster than `lapack-aocl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 284.347 ± 0.544212 (W 3.16228) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1118.44 ± 1.62706 (W 0.9722) | 1211.56 ± 2.23288 (W 1.60058) | 🔵 faer-fork 1.08× faster than `lapack-aocl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 1134.91 ± 4.94015 (W 2.13318) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.212809 ± 0.00256297 (W 0.00388787) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.651839 ± 0.00669779 (W 0.00730746) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.37898 ± 0.0271044 (W 0.0213269) | 9.57381 ± 0.0898559 (W 0.0186749) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 10.1839 ± 0.0591653 (W 0.0473348) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 25.3574 ± 0.0641534 (W 0.0325683) | 26.7648 ± 0.0628191 (W 0.0563343) | 🔵 faer-fork 1.06× faster than `lapack-aocl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 44.9016 ± 0.293237 (W 0.121081) | - | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 136.657 ± 0.209571 (W 0.137612) | 467.517 ± 1.87208 (W 2.0863) | 🔵 faer-fork 3.42× faster than `lapack-aocl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 146.128 ± 0.46411 (W 0.624712) | - | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 423.854 ± 0.911528 (W 1.10641) | 2291.39 ± 9.06172 (W 20.1202) | 🔵 faer-fork 5.41× faster than `lapack-aocl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 462.405 ± 1.61655 (W 2.80989) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 77.6684 ± 0.45402 (W 0.168611) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 90.5737 ± 0.611799 (W 0.421689) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 471.122 ± 3.71125 (W 3.30102) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 503.865 ± 2.5934 (W 2.35617) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 18.1929 ± 0.0464271 (W 0.0316016) | 18.7309 ± 0.053545 (W 0.133527) | · |  |
| `lu/n1024/b1 [api-inclusive]` | 33.6569 ± 0.127899 (W 0.0646783) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 53.2132 ± 0.0943095 (W 0.163007) | 63.5584 ± 0.136151 (W 0.16905) | 🔵 faer-fork 1.19× faster than `lapack-aocl` |  |
| `lu/n1536/b1 [api-inclusive]` | 87.7916 ± 0.254889 (W 0.452069) | - | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 72.3676 ± 0.119912 (W 0.187229) | 115.356 ± 0.362354 (W 0.590721) | 🔵 faer-fork 1.59× faster than `lapack-aocl` |  |
| `qr/n1024/b1 [api-inclusive]` | 95.6129 ± 0.294412 (W 0.125822) | - | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 236.939 ± 0.427635 (W 0.346228) | 363.033 ± 1.15638 (W 0.416448) | 🔵 faer-fork 1.53× faster than `lapack-aocl` |  |
| `qr/n1536/b1 [api-inclusive]` | 289.418 ± 1.11906 (W 0.716738) | - | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 31.8601 ± 0.0763154 (W 0.029492) | 55.9447 ± 0.216735 (W 0.0864787) | 🔵 faer-fork 1.76× faster than `lapack-aocl` |  |
| `qr_factor/n1536/b1` | 102.686 ± 0.0980994 (W 0.115136) | 181.139 ± 0.209378 (W 0.354429) | 🔵 faer-fork 1.76× faster than `lapack-aocl` |  |
| `solve/n1024/b1` | 18.2104 ± 0.0283125 (W 0.0327439) | 16.273 ± 0.0256882 (W 0.0307594) | 🔴 faer-fork 1.12× slower than `lapack-aocl` |  |
| `solve/n1024/b1 [api-inclusive]` | 21.1555 ± 0.049048 (W 0.0439269) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 53.5913 ± 0.120722 (W 0.110547) | 57.7607 ± 0.732257 (W 0.0747344) | 🔵 faer-fork 1.08× faster than `lapack-aocl` |  |
| `solve/n1536/b1 [api-inclusive]` | 87.2663 ± 0.583234 (W 0.15833) | - | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 299.634 ± 0.822962 (W 0.711591) | 521.499 ± 0.764434 (W 1.42972) | 🔵 faer-fork 1.74× faster than `lapack-aocl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 343.134 ± 0.696698 (W 1.12197) | - | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1184.61 ± 4.73203 (W 1.42814) | 1734.59 ± 5.24635 (W 3.59197) | 🔵 faer-fork 1.46× faster than `lapack-aocl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1266.76 ± 6.69813 (W 3.87974) | - | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 167.36 ± 1.22104 (W 1.06451) | 309.199 ± 0.8143 (W 1.00009) | 🔵 faer-fork 1.85× faster than `lapack-aocl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 167.839 ± 0.873115 (W 0.396118) | - | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 786.627 ± 2.71606 (W 2.87406) | 1126.26 ± 5.72837 (W 1.60238) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 805.054 ± 4.96827 (W 3.36025) | - | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.153543 ± 0.00733303 (W 0.0013536) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n1536/b1` | 0.497969 ± 0.0239215 (W 0.00600752) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024`, `cholesky/1536`, `eigh/1024`, `eigh/1536`, `full_piv_lu/1024`, `full_piv_lu/1536`, `lu/1024`, `lu/1536`, `qr/1024`, `qr/1536`, `qr_factor/1024`, `qr_factor/1536`, `solve/1024`, `solve/1536`, `svd_thin/1024`, `svd_thin/1536`, `svd_values/1024`, `svd_values/1536`, `triangular_solve/1024`, `triangular_solve/1536`.

## gemm-large, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.5183 ± 0.0412339 (W 0.0116011) | 17.9136 ± 0.10934 (W 0.0152031) | · |  |
| `gemm/1536x1536x1536/b1` | 59.1688 ± 0.240033 (W 0.102876) | 60.2219 ± 0.398396 (W 0.0726962) | · |  |

## gemm-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.95357 ± 0.0151723 (W 0.00788701) | 9.16842 ± 0.0373533 (W 0.0137624) | · |  |
| `gemm/1536x1536x1536/b1` | 30.0075 ± 0.0738857 (W 0.0313629) | 30.363 ± 0.286968 (W 0.0379563) | · |  |

## gemm-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.4921 ± 0.0249428 (W 0.0205207) | 18.0468 ± 0.157159 (W 0.0291457) | · |  |
| `gemm/1536x1536x1536/b1` | 59.2426 ± 0.220643 (W 0.115723) | 60.6946 ± 0.426157 (W 0.0654776) | · |  |

## gemm-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.95223 ± 0.0209357 (W 0.0107415) | 9.15749 ± 0.0206384 (W 0.0130988) | · |  |
| `gemm/1536x1536x1536/b1` | 30.0158 ± 0.0866143 (W 0.0261031) | 30.8514 ± 0.280898 (W 0.059778) | · |  |

## gemm-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.6635 ± 0.0822176 (W 0.0774679) | 75.3183 ± 0.264262 (W 0.0846611) | 🔵 faer-fork 1.08× faster than `lapack-aocl` |  |
| `gemm/1536x1536x1536/b1` | 238.225 ± 0.671147 (W 0.312679) | 236.997 ± 0.223681 (W 0.112463) | · |  |

## gemm-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.0922 ± 0.0928072 (W 0.0292412) | 38.1233 ± 0.165852 (W 0.0341489) | 🔵 faer-fork 1.09× faster than `lapack-aocl` |  |
| `gemm/1536x1536x1536/b1` | 118.986 ± 0.417658 (W 0.097161) | 121.889 ± 0.136083 (W 0.10016) | · |  |

## gemm-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.5938 ± 0.0713337 (W 0.0345952) | 75.4066 ± 0.116969 (W 0.101613) | 🔵 faer-fork 1.08× faster than `lapack-aocl` |  |
| `gemm/1536x1536x1536/b1` | 238.097 ± 0.620436 (W 0.385996) | 237.077 ± 0.104599 (W 0.11758) | · |  |

## gemm-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.0371 ± 0.101421 (W 0.0188286) | 38.033 ± 0.0983622 (W 0.170239) | 🔵 faer-fork 1.09× faster than `lapack-aocl` |  |
| `gemm/1536x1536x1536/b1` | 119.222 ± 0.391713 (W 0.155257) | 121.937 ± 0.170638 (W 0.149921) | · |  |

## linalg-boundary, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.169429 ± 0.00140419 (W 0.0015525) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.195428 ± 0.00229785 (W 0.00153266) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.154317 ± 0.00139344 (W 0.000664218) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.183127 ± 0.00153302 (W 0.000865394) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.710632 ± 0.00127008 (W 0.00149158) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.861348 ± 0.0038163 (W 0.0043654) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.685276 ± 0.00590763 (W 0.00177714) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.942054 ± 0.0118801 (W 0.00310822) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.87505 ± 0.016466 (W 0.00898002) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.8661 ± 0.0183631 (W 0.00859277) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.7936 ± 0.0169247 (W 0.00836649) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.80747 ± 0.0157401 (W 0.00816477) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.8678 ± 0.0249643 (W 0.0283106) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 24.9021 ± 0.0370931 (W 0.0223968) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.3186 ± 0.0402964 (W 0.0265587) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 25.22 ± 0.0378006 (W 0.0363835) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.25398 ± 0.00635947 (W 0.0041343) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.26098 ± 0.010313 (W 0.00927663) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.28843 ± 0.0050153 (W 0.00404645) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.5664 ± 0.00823262 (W 0.00728378) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 6.26923 ± 0.021821 (W 0.0133649) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 13.0447 ± 0.104956 (W 0.0305318) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.36884 ± 0.0180222 (W 0.00720236) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 7.35828 ± 0.0262848 (W 0.0111526) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.431546 ± 0.00142891 (W 0.00233741) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.700713 ± 0.00473115 (W 0.00302882) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.420743 ± 0.00385795 (W 0.00457425) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.770425 ± 0.00280434 (W 0.00319841) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.75166 ± 0.00627055 (W 0.00463301) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.49159 ± 0.0197937 (W 0.0170969) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.85322 ± 0.00830853 (W 0.00666433) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.48515 ± 0.0528959 (W 0.0104013) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.04802 ± 0.00413316 (W 0.00239492) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.57868 ± 0.00452159 (W 0.00632505) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.11669 ± 0.00247552 (W 0.00181618) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.6657 ± 0.00761064 (W 0.00746541) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.80525 ± 0.0108117 (W 0.0104636) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 8.34648 ± 0.0251772 (W 0.0206361) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 5.97671 ± 0.0171939 (W 0.00903227) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 8.59439 ± 0.0487567 (W 0.0219824) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.662501 ± 0.00299317 (W 0.00236724) | - | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 0.694376 ± 0.00252934 (W 0.00210857) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 3.18894 ± 0.00861203 (W 0.00650743) | - | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 3.19816 ± 0.00408927 (W 0.00408785) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.355579 ± 0.00259824 (W 0.00251821) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.710843 ± 0.00800808 (W 0.00433282) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.420516 ± 0.000457188 (W 0.00233472) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.772839 ± 0.00351695 (W 0.00458124) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.71916 ± 0.0109175 (W 0.00917919) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.45591 ± 0.0190548 (W 0.0063893) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.80554 ± 0.00824019 (W 0.00407299) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.46207 ± 0.0343007 (W 0.00860706) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.9448 ± 0.0250402 (W 0.0158437) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 7.95905 ± 0.0346138 (W 0.0155616) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.11342 ± 0.0375973 (W 0.0123667) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 8.13535 ± 0.038193 (W 0.0162584) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.7799 ± 0.078585 (W 0.0430074) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 35.9298 ± 0.0920775 (W 0.0765383) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.5902 ± 0.0568952 (W 0.0408169) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 36.6783 ± 0.119122 (W 0.0544546) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 4.01746 ± 0.0145598 (W 0.00970806) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 4.02203 ± 0.011713 (W 0.010198) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.10517 ± 0.0167499 (W 0.00888031) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 4.10436 ± 0.013461 (W 0.00925007) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.5209 ± 0.103826 (W 0.0156709) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 17.6128 ± 0.0574152 (W 0.0181006) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.7701 ± 0.0549779 (W 0.0304397) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 17.7772 ± 0.0927787 (W 0.0201061) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0132955 ± 0.000111202 (W 0.00013151) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.0135705 ± 9.66038e-05 (W 9.19964e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.0318545 ± 0.000246212 (W 0.000202206) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0325705 ± 0.000228921 (W 0.000246632) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.151589 ± 0.0005112 (W 0.00142668) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.180958 ± 0.00204619 (W 0.00119626) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.142886 ± 0.00201577 (W 0.00111568) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.168321 ± 0.00223689 (W 0.00103628) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.523052 ± 0.00326141 (W 0.00251981) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.701641 ± 0.00457538 (W 0.00285474) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.503711 ± 0.00425207 (W 0.00228463) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.79104 ± 0.00723711 (W 0.00239763) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 4.01972 ± 0.0127506 (W 0.00887486) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 4.01797 ± 0.0111604 (W 0.00560053) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.97094 ± 0.0318346 (W 0.00684247) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 3.97991 ± 0.0094157 (W 0.0036932) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.2977 ± 0.033841 (W 0.0320226) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 16.3287 ± 0.046748 (W 0.0325895) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.8172 ± 0.124724 (W 0.0167048) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 16.8016 ± 0.123644 (W 0.0202674) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.15455 ± 0.00477767 (W 0.00526085) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.19126 ± 0.0131182 (W 0.00881064) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.19791 ± 0.0110314 (W 0.0049466) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.51591 ± 0.0103466 (W 0.00438365) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 4.71117 ± 0.019032 (W 0.0111425) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.8786 ± 0.0246239 (W 0.0286426) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.7971 ± 0.0262347 (W 0.00926092) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.09289 ± 0.0299365 (W 0.0206603) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.419683 ± 0.0021229 (W 0.00224742) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.675333 ± 0.00860766 (W 0.00342533) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.353471 ± 0.00287958 (W 0.00268752) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.737102 ± 0.007441 (W 0.00338355) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.29705 ± 0.00337489 (W 0.00624598) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.14977 ± 0.00601418 (W 0.00994935) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.35352 ± 0.00424018 (W 0.00374749) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.10216 ± 0.00860587 (W 0.00867385) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.84378 ± 0.00449247 (W 0.00228833) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.39902 ± 0.00409351 (W 0.00480133) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 0.899243 ± 0.00359544 (W 0.0022935) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.49165 ± 0.0042966 (W 0.00581222) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 3.30059 ± 0.0125245 (W 0.00926062) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 6.10908 ± 0.0434867 (W 0.0200419) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 3.73868 ± 0.0171052 (W 0.00768946) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 6.55548 ± 0.0256919 (W 0.0174061) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.613411 ± 0.0200995 (W 0.00256535) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n257/b1` | 0.646996 ± 0.0017041 (W 0.0016125) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 1.87586 ± 0.00305274 (W 0.0043813) | - | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 2.23706 ± 0.0088509 (W 0.00480082) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.342828 ± 0.00144134 (W 0.00282205) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.6812 ± 0.00332978 (W 0.00374772) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.35631 ± 0.00200092 (W 0.00308135) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.745316 ± 0.00653336 (W 0.0038763) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.24703 ± 0.00455662 (W 0.00477728) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.1557 ± 0.0187599 (W 0.00984851) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.31589 ± 0.00383992 (W 0.00481767) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.11449 ± 0.0352997 (W 0.00889429) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.25297 ± 0.0087301 (W 0.0138077) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 7.25944 ± 0.0194702 (W 0.00950092) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.36088 ± 0.00849125 (W 0.0124323) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 7.37891 ± 0.0117655 (W 0.00829833) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.4442 ± 0.0952919 (W 0.0618192) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 29.5345 ± 0.113976 (W 0.0737948) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.9393 ± 0.0662507 (W 0.107523) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 29.9374 ± 0.095408 (W 0.0809274) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.89984 ± 0.00746065 (W 0.00610904) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 3.89525 ± 0.0112235 (W 0.00642365) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.97625 ± 0.0136395 (W 0.00571515) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 3.94875 ± 0.0062349 (W 0.00527713) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.4249 ± 0.0175578 (W 0.0213582) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 15.3962 ± 0.0612084 (W 0.0156863) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.5562 ± 0.0576801 (W 0.0265415) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 15.5507 ± 0.0592582 (W 0.0233017) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.012744 ± 0.000109405 (W 0.000126011) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.012949 ± 0.000166422 (W 8.98497e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.029425 ± 0.000249742 (W 0.000215391) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0290145 ± 0.000347019 (W 0.000196497) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.159042 ± 0.000860004 (W 0.0010099) | 0.140539 ± 0.00108372 (W 0.00112053) | 🔴 faer-fork 1.13× slower than `lapack-aocl` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.194275 ± 0.00138548 (W 0.00121534) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.147944 ± 0.00136893 (W 0.00125845) | 0.14657 ± 0.0008749 (W 0.00224131) | · |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.181858 ± 0.00187268 (W 0.00208596) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.675326 ± 0.00142898 (W 0.00220331) | 0.531421 ± 0.00580431 (W 0.00965587) | 🔴 faer-fork 1.27× slower than `lapack-aocl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.85195 ± 0.00400844 (W 0.00342919) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.671899 ± 0.00287779 (W 0.00297367) | 0.555221 ± 0.00310238 (W 0.00554304) | 🔴 faer-fork 1.21× slower than `lapack-aocl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.809352 ± 0.00363802 (W 0.00282099) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.87197 ± 0.0263799 (W 0.00674569) | 5.75169 ± 0.00905866 (W 0.0116807) | · |  |
| `eigh/n255/b1 [api-inclusive]` | 5.86883 ± 0.0208982 (W 0.00712877) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.78565 ± 0.019805 (W 0.00805011) | 5.85682 ± 0.0176494 (W 0.0150791) | · |  |
| `eigh/n257/b1 [api-inclusive]` | 5.79635 ± 0.0150259 (W 0.00519737) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.8067 ± 0.0572706 (W 0.0289134) | 26.0339 ± 0.07019 (W 0.0436854) | · |  |
| `eigh/n511/b1 [api-inclusive]` | 24.8448 ± 0.0354548 (W 0.0267373) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.2291 ± 0.0421914 (W 0.0238628) | 26.1 ± 0.0756009 (W 0.0664453) | · |  |
| `eigh/n513/b1 [api-inclusive]` | 25.0924 ± 0.0722817 (W 0.0222993) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.18269 ± 0.0117084 (W 0.00438752) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.17085 ± 0.00821802 (W 0.0092509) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.2005 ± 0.00730964 (W 0.00436304) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.45357 ± 0.0120512 (W 0.00757818) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 6.0417 ± 0.025695 (W 0.00989144) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 12.5261 ± 0.0463341 (W 0.0279189) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.09425 ± 0.00926386 (W 0.0139374) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.83254 ± 0.0203347 (W 0.0142206) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.303652 ± 0.00276086 (W 0.00359245) | 0.360656 ± 0.00406987 (W 0.00320658) | 🔵 faer-fork 1.19× faster than `lapack-aocl` |  |
| `lu/n255/b1 [api-inclusive]` | 0.609581 ± 0.00330815 (W 0.00431495) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.327867 ± 0.00259667 (W 0.00314435) | 0.374515 ± 0.00280507 (W 0.00204916) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `lu/n257/b1 [api-inclusive]` | 0.654051 ± 0.00214107 (W 0.00295809) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.49119 ± 0.00699549 (W 0.00567924) | 1.59427 ± 0.0063846 (W 0.00536719) | 🔵 faer-fork 1.07× faster than `lapack-aocl` |  |
| `lu/n511/b1 [api-inclusive]` | 2.96678 ± 0.0228408 (W 0.00521723) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.55826 ± 0.00791243 (W 0.00335388) | 1.74526 ± 0.00903099 (W 0.0055506) | 🔵 faer-fork 1.12× faster than `lapack-aocl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.96066 ± 0.0204463 (W 0.00672862) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.970627 ± 0.00162963 (W 0.0011782) | 1.76846 ± 0.00449665 (W 0.00710515) | 🔵 faer-fork 1.82× faster than `lapack-aocl` |  |
| `qr/n255/b1 [api-inclusive]` | 1.46922 ± 0.00313529 (W 0.00899556) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.00602 ± 0.00536242 (W 0.00135599) | 1.84952 ± 0.00407363 (W 0.00765752) | 🔵 faer-fork 1.84× faster than `lapack-aocl` |  |
| `qr/n257/b1 [api-inclusive]` | 1.54541 ± 0.00434442 (W 0.00726983) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.48361 ± 0.0081002 (W 0.00505099) | 8.37931 ± 0.0102498 (W 0.0161616) | 🔵 faer-fork 1.53× faster than `lapack-aocl` |  |
| `qr/n511/b1 [api-inclusive]` | 7.74131 ± 0.0188368 (W 0.0130411) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 5.58714 ± 0.0111213 (W 0.00784) | 8.61791 ± 0.0116787 (W 0.0183894) | 🔵 faer-fork 1.54× faster than `lapack-aocl` |  |
| `qr/n513/b1 [api-inclusive]` | 7.94979 ± 0.0706656 (W 0.0246301) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.581571 ± 0.00329411 (W 0.00174378) | 1.02451 ± 0.00626573 (W 0.00453308) | 🔵 faer-fork 1.76× faster than `lapack-aocl` |  |
| `qr_factor/n257/b1` | 0.599057 ± 0.00265263 (W 0.00259693) | 1.06687 ± 0.00621163 (W 0.00427311) | 🔵 faer-fork 1.78× faster than `lapack-aocl` |  |
| `qr_factor/n511/b1` | 2.90933 ± 0.00852137 (W 0.00368135) | 4.67095 ± 0.0202298 (W 0.0114621) | 🔵 faer-fork 1.61× faster than `lapack-aocl` |  |
| `qr_factor/n513/b1` | 2.86872 ± 0.00355779 (W 0.00280237) | 4.798 ± 0.0179513 (W 0.0104166) | 🔵 faer-fork 1.67× faster than `lapack-aocl` |  |
| `solve/n255/b1` | 0.312809 ± 0.00182349 (W 0.00187772) | 0.314196 ± 0.00266371 (W 0.00347849) | · |  |
| `solve/n255/b1 [api-inclusive]` | 0.618398 ± 0.00311207 (W 0.00344162) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.328223 ± 0.00200723 (W 0.00184314) | 0.329596 ± 0.000797491 (W 0.00322153) | · |  |
| `solve/n257/b1 [api-inclusive]` | 0.660338 ± 0.00235959 (W 0.00464787) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.4525 ± 0.00695515 (W 0.0037303) | 1.30973 ± 0.00939685 (W 0.0045978) | 🔴 faer-fork 1.11× slower than `lapack-aocl` |  |
| `solve/n511/b1 [api-inclusive]` | 2.91796 ± 0.0095177 (W 0.00964578) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.51859 ± 0.00694572 (W 0.00530787) | 1.44597 ± 0.0069521 (W 0.00422001) | · |  |
| `solve/n513/b1 [api-inclusive]` | 2.92924 ± 0.0327392 (W 0.00870619) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.92276 ± 0.0240059 (W 0.0151687) | 11.1282 ± 0.074958 (W 0.0295851) | 🔵 faer-fork 1.40× faster than `lapack-aocl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.9508 ± 0.0223966 (W 0.0126818) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.11736 ± 0.0366971 (W 0.0121085) | 11.2636 ± 0.0434392 (W 0.0385507) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 8.13265 ± 0.0252546 (W 0.016481) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.6818 ± 0.0694962 (W 0.032961) | 44.9583 ± 0.49679 (W 0.114432) | 🔵 faer-fork 1.26× faster than `lapack-aocl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 35.7619 ± 0.104337 (W 0.032987) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.3819 ± 0.0899735 (W 0.0605573) | 45.6787 ± 0.183874 (W 0.0849328) | 🔵 faer-fork 1.26× faster than `lapack-aocl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 36.4367 ± 0.0560611 (W 0.0380397) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 4.00531 ± 0.0121116 (W 0.00932437) | 6.99715 ± 0.0397797 (W 0.0288797) | 🔵 faer-fork 1.75× faster than `lapack-aocl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 3.99355 ± 0.00951715 (W 0.00858816) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.12793 ± 0.01299 (W 0.00987195) | 7.05912 ± 0.0243898 (W 0.0377316) | 🔵 faer-fork 1.71× faster than `lapack-aocl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 4.11157 ± 0.0124718 (W 0.00787366) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.4805 ± 0.0741351 (W 0.0206307) | 26.898 ± 0.0929946 (W 0.0482883) | 🔵 faer-fork 1.54× faster than `lapack-aocl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 17.4691 ± 0.0679686 (W 0.0175024) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.6848 ± 0.0657061 (W 0.014829) | 27.0917 ± 0.0821601 (W 0.0662568) | 🔵 faer-fork 1.53× faster than `lapack-aocl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 17.6542 ± 0.0581174 (W 0.02422) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0157545 ± 0.000329058 (W 0.000257517) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.0158645 ± 0.000161574 (W 0.000209642) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.03978 ± 0.000346447 (W 0.00111108) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.039584 ± 0.000349638 (W 0.000541751) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.143169 ± 0.00100118 (W 0.000914053) | 0.133774 ± 0.0130402 (W 0.00192826) | · | `!` between-start range over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.180119 ± 0.00109516 (W 0.00135231) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.134606 ± 0.00172725 (W 0.000901125) | 0.150177 ± 0.000474597 (W 0.00153433) | 🔵 faer-fork 1.12× faster than `lapack-aocl` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.168765 ± 0.00171622 (W 0.00157296) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.502754 ± 0.00287863 (W 0.00293959) | 0.348927 ± 0.00498108 (W 0.00406068) | 🔴 faer-fork 1.44× slower than `lapack-aocl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.681668 ± 0.0014929 (W 0.00229469) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.505483 ± 0.00260012 (W 0.00292667) | 0.36208 ± 0.00540943 (W 0.00480053) | 🔴 faer-fork 1.40× slower than `lapack-aocl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.650421 ± 0.0043976 (W 0.00328603) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 4.00807 ± 0.00955321 (W 0.00589221) | 5.59416 ± 0.0236081 (W 0.0246512) | 🔵 faer-fork 1.40× faster than `lapack-aocl` |  |
| `eigh/n255/b1 [api-inclusive]` | 4.01406 ± 0.00949243 (W 0.00510912) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.96803 ± 0.0308997 (W 0.0133269) | 5.68026 ± 0.0249409 (W 0.0233592) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `eigh/n257/b1 [api-inclusive]` | 3.97241 ± 0.00444777 (W 0.00739648) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.2141 ± 0.0153495 (W 0.0152059) | 23.9602 ± 0.0812313 (W 0.0859973) | 🔵 faer-fork 1.48× faster than `lapack-aocl` |  |
| `eigh/n511/b1 [api-inclusive]` | 16.2732 ± 0.0466382 (W 0.0171867) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.6431 ± 0.0310913 (W 0.0204407) | 24.0326 ± 0.0841157 (W 0.0659448) | 🔵 faer-fork 1.44× faster than `lapack-aocl` |  |
| `eigh/n513/b1 [api-inclusive]` | 16.6401 ± 0.0554873 (W 0.0167713) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.10326 ± 0.00593016 (W 0.00401768) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.11201 ± 0.00856694 (W 0.00762231) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.11419 ± 0.00799412 (W 0.00450341) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.40769 ± 0.0117951 (W 0.00462364) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 4.58252 ± 0.0153383 (W 0.010231) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.3709 ± 0.0527692 (W 0.0147693) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.62627 ± 0.02764 (W 0.00799855) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 5.59387 ± 0.020482 (W 0.0097962) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.279131 ± 0.00148613 (W 0.00221254) | 0.309232 ± 0.00485952 (W 0.00318511) | 🔵 faer-fork 1.11× faster than `lapack-aocl` |  |
| `lu/n255/b1 [api-inclusive]` | 0.580481 ± 0.00231075 (W 0.00275186) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.289834 ± 0.00139295 (W 0.00174649) | 0.338853 ± 0.00230341 (W 0.00176291) | 🔵 faer-fork 1.17× faster than `lapack-aocl` |  |
| `lu/n257/b1 [api-inclusive]` | 0.620791 ± 0.00502521 (W 0.00368569) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.13122 ± 0.0032458 (W 0.00270488) | 1.45423 ± 0.011147 (W 0.00412864) | 🔵 faer-fork 1.29× faster than `lapack-aocl` |  |
| `lu/n511/b1 [api-inclusive]` | 2.61712 ± 0.0129683 (W 0.00898662) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.17452 ± 0.00172026 (W 0.00301302) | 1.2436 ± 0.0102184 (W 0.00561932) | 🔵 faer-fork 1.06× faster than `lapack-aocl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.61877 ± 0.0239953 (W 0.00843366) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.796751 ± 0.00348302 (W 0.00268604) | 1.40007 ± 0.0133957 (W 0.0139552) | 🔵 faer-fork 1.76× faster than `lapack-aocl` |  |
| `qr/n255/b1 [api-inclusive]` | 1.28671 ± 0.00520787 (W 0.00302914) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 0.820667 ± 0.00309242 (W 0.00322922) | 1.48317 ± 0.00942012 (W 0.0147801) | 🔵 faer-fork 1.81× faster than `lapack-aocl` |  |
| `qr/n257/b1 [api-inclusive]` | 1.35822 ± 0.00604775 (W 0.00524322) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 3.13071 ± 0.00884625 (W 0.00755538) | 5.39174 ± 0.0272336 (W 0.0224069) | 🔵 faer-fork 1.72× faster than `lapack-aocl` |  |
| `qr/n511/b1 [api-inclusive]` | 5.47937 ± 0.0173377 (W 0.0183851) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 3.491 ± 0.0074191 (W 0.0077601) | 5.62553 ± 0.01587 (W 0.0289284) | 🔵 faer-fork 1.61× faster than `lapack-aocl` |  |
| `qr/n513/b1 [api-inclusive]` | 5.93268 ± 0.0326494 (W 0.0221984) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.566258 ± 0.00132768 (W 0.00164005) | 0.786928 ± 0.00495649 (W 0.00822032) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `qr_factor/n257/b1` | 0.582484 ± 0.00167438 (W 0.00213149) | 0.823273 ± 0.00681075 (W 0.00846558) | 🔵 faer-fork 1.41× faster than `lapack-aocl` |  |
| `qr_factor/n511/b1` | 1.73324 ± 0.00858671 (W 0.00337407) | 3.00005 ± 0.0168039 (W 0.0140281) | 🔵 faer-fork 1.73× faster than `lapack-aocl` |  |
| `qr_factor/n513/b1` | 2.03924 ± 0.00530529 (W 0.00495551) | 3.10753 ± 0.01578 (W 0.0252789) | 🔵 faer-fork 1.52× faster than `lapack-aocl` |  |
| `solve/n255/b1` | 0.283187 ± 0.00137446 (W 0.00151693) | 0.289215 ± 0.00335441 (W 0.00325883) | · |  |
| `solve/n255/b1 [api-inclusive]` | 0.59359 ± 0.00489419 (W 0.00344322) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.292329 ± 0.00102919 (W 0.00179172) | 0.297476 ± 0.00355879 (W 0.00334229) | · |  |
| `solve/n257/b1 [api-inclusive]` | 0.631009 ± 0.00595716 (W 0.00277514) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.09193 ± 0.0050849 (W 0.00264324) | 1.22267 ± 0.00801618 (W 0.00426117) | 🔵 faer-fork 1.12× faster than `lapack-aocl` |  |
| `solve/n511/b1 [api-inclusive]` | 2.60348 ± 0.0169618 (W 0.00883631) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.13919 ± 0.00391873 (W 0.00326452) | 1.10529 ± 0.00502004 (W 0.0069144) | · |  |
| `solve/n513/b1 [api-inclusive]` | 2.58773 ± 0.0117064 (W 0.00694103) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.18897 ± 0.0103659 (W 0.0134514) | 13.6112 ± 0.0636499 (W 0.0660578) | 🔵 faer-fork 1.89× faster than `lapack-aocl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.1968 ± 0.00999594 (W 0.00887147) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.35985 ± 0.021887 (W 0.0107512) | 13.7206 ± 0.0839256 (W 0.0816124) | 🔵 faer-fork 1.86× faster than `lapack-aocl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 7.37098 ± 0.0189848 (W 0.0100443) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.3519 ± 0.0545292 (W 0.058283) | 44.3832 ± 0.256135 (W 0.110901) | 🔵 faer-fork 1.51× faster than `lapack-aocl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 29.3826 ± 0.0440132 (W 0.0351833) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.7871 ± 0.0590142 (W 0.0345529) | 44.7117 ± 0.0147013 (W 0.119512) | 🔵 faer-fork 1.50× faster than `lapack-aocl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 29.8208 ± 0.0527808 (W 0.0556185) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.88066 ± 0.0164589 (W 0.0058387) | 9.72252 ± 0.111594 (W 0.0458083) | 🔵 faer-fork 2.51× faster than `lapack-aocl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 3.86472 ± 0.00582686 (W 0.00626691) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.971 ± 0.00856057 (W 0.00981987) | 9.78172 ± 0.0599184 (W 0.0619144) | 🔵 faer-fork 2.46× faster than `lapack-aocl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 3.96326 ± 0.0156444 (W 0.00716736) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.3718 ± 0.0272361 (W 0.0163848) | 29.4321 ± 0.124511 (W 0.197272) | 🔵 faer-fork 1.91× faster than `lapack-aocl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 15.4043 ± 0.0384516 (W 0.017269) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.4453 ± 0.044985 (W 0.0225273) | 29.7973 ± 0.249815 (W 0.0653855) | 🔵 faer-fork 1.93× faster than `lapack-aocl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 15.4314 ± 0.034696 (W 0.0189894) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.014958 ± 0.000838739 (W 0.000225931) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.0147225 ± 9.24568e-05 (W 0.000157035) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.036223 ± 0.000303404 (W 0.000296779) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0356315 ± 0.000359097 (W 0.000491083) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.510533 ± 0.00144178 (W 0.00256568) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.576887 ± 0.00297548 (W 0.00219533) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.464514 ± 0.00177208 (W 0.00174581) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.529718 ± 0.00249254 (W 0.00209993) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.30486 ± 0.00611436 (W 0.00489867) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 2.80073 ± 0.00857047 (W 0.00428036) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.24264 ± 0.00590547 (W 0.00635198) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 2.72933 ± 0.0129936 (W 0.00507066) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.34391 ± 0.020347 (W 0.0116681) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 8.35536 ± 0.0150851 (W 0.00601406) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.3593 ± 0.0196379 (W 0.0129165) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 8.40155 ± 0.0275563 (W 0.0197073) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.5994 ± 0.039525 (W 0.0300204) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 40.7679 ± 0.0327568 (W 0.0365982) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.5661 ± 0.060265 (W 0.0342018) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 40.6401 ± 0.0638328 (W 0.0439046) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.98965 ± 0.013763 (W 0.00542402) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 4.09095 ± 0.0121974 (W 0.0107669) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.03741 ± 0.00969673 (W 0.00578215) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.719 ± 0.00482804 (W 0.00484626) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 11.1994 ± 0.0299272 (W 0.0154877) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 20.6179 ± 0.0274696 (W 0.0315523) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 11.324 ± 0.0344199 (W 0.0226249) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 15.0854 ± 0.0232097 (W 0.031765) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.961779 ± 0.00373516 (W 0.00352779) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.72171 ± 0.00523606 (W 0.00701345) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.03707 ± 0.00591808 (W 0.00272487) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.75719 ± 0.00808384 (W 0.00463986) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 5.12121 ± 0.0374408 (W 0.0206086) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 9.34829 ± 0.0507469 (W 0.0472996) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 5.36214 ± 0.0164252 (W 0.0122687) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 9.49892 ± 0.0975895 (W 0.0222615) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 3.05267 ± 0.0124235 (W 0.0039864) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 4.19005 ± 0.0263472 (W 0.00952235) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.17498 ± 0.00574388 (W 0.00378029) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 4.31945 ± 0.0137327 (W 0.0122317) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 19.6564 ± 0.0458045 (W 0.0162025) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 25.2566 ± 0.0687466 (W 0.0420209) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 20.1254 ± 0.0635051 (W 0.0195954) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 26.0029 ± 0.082491 (W 0.0443892) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.60719 ± 0.0067214 (W 0.00283024) | - | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.67462 ± 0.00584569 (W 0.00245507) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 9.32643 ± 0.03386 (W 0.0180819) | - | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 9.56812 ± 0.0447737 (W 0.0157131) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.878167 ± 0.00330097 (W 0.00166727) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.73616 ± 0.00794847 (W 0.00478069) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.04522 ± 0.00279118 (W 0.00343598) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.77115 ± 0.00466257 (W 0.0052688) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 5.10802 ± 0.0123106 (W 0.00950163) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 9.37879 ± 0.0307767 (W 0.0225993) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 5.36899 ± 0.0130555 (W 0.012673) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 9.55778 ± 0.0555138 (W 0.0163281) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.9022 ± 0.0139104 (W 0.0161815) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 12.9627 ± 0.019537 (W 0.023382) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 13.188 ± 0.0162605 (W 0.0132375) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 13.2536 ± 0.0235891 (W 0.021192) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 68.986 ± 0.174271 (W 0.0736119) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 69.1954 ± 0.0941918 (W 0.0730582) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 70.546 ± 0.105415 (W 0.167414) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 70.7287 ± 0.187641 (W 0.0663707) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.3644 ± 0.022366 (W 0.008822) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 6.36765 ± 0.0167744 (W 0.00933763) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.5607 ± 0.0264084 (W 0.0129431) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 6.5394 ± 0.0245873 (W 0.00774058) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 34.383 ± 0.11796 (W 0.0433072) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 34.5029 ± 0.0878292 (W 0.0222014) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 34.8872 ± 0.0706221 (W 0.0417758) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 34.8932 ± 0.0760107 (W 0.0416717) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.021881 ± 0.000104762 (W 0.000109329) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.0220415 ± 0.000124566 (W 0.000102174) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.0557045 ± 0.00104388 (W 0.000799825) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.0563665 ± 0.00100834 (W 0.000531558) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.438522 ± 0.00154829 (W 0.0023817) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.512035 ± 0.00371448 (W 0.00336379) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.394405 ± 0.00634462 (W 0.00158885) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.462333 ± 0.00281187 (W 0.00229863) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.57517 ± 0.00552952 (W 0.00385577) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 2.11482 ± 0.0138025 (W 0.00748594) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.48881 ± 0.0184891 (W 0.00350593) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 2.01898 ± 0.0124469 (W 0.00516478) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.56374 ± 0.0176142 (W 0.00957918) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.5788 ± 0.0129761 (W 0.00846961) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.55312 ± 0.029627 (W 0.010256) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.55983 ± 0.0168797 (W 0.00833636) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.4696 ± 0.0536941 (W 0.0774753) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 25.5339 ± 0.0319397 (W 0.0124294) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 26.0355 ± 0.0478958 (W 0.0303022) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 26.1147 ± 0.0746839 (W 0.0289536) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.58546 ± 0.00347097 (W 0.00596772) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.7545 ± 0.00799779 (W 0.00810113) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.6161 ± 0.0145635 (W 0.00459404) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.37881 ± 0.0223951 (W 0.00644859) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.33077 ± 0.0204051 (W 0.025451) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 17.0538 ± 0.0617255 (W 0.0359623) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.43073 ± 0.0144927 (W 0.0180918) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 11.4372 ± 0.107725 (W 0.0182659) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.827968 ± 0.00181548 (W 0.00185493) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.5729 ± 0.00890369 (W 0.0057273) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.816218 ± 0.00266762 (W 0.00227589) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.58225 ± 0.00634548 (W 0.0052893) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.39662 ± 0.00718525 (W 0.00658286) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 7.79472 ± 0.0550059 (W 0.0277609) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.50287 ± 0.0691005 (W 0.00752689) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n513/b1 [api-inclusive]` | 7.84646 ± 0.0323271 (W 0.0149449) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.05317 ± 0.0109821 (W 0.00334901) | - | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 3.25189 ± 0.0114297 (W 0.00930605) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 2.15518 ± 0.00577273 (W 0.00532745) | - | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 3.34773 ± 0.011864 (W 0.0103503) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 10.4593 ± 0.0424655 (W 0.0163087) | - | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 16.4687 ± 0.0601823 (W 0.0246824) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 11.2959 ± 0.0390061 (W 0.0259599) | - | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 17.3406 ± 0.0445856 (W 0.0269321) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.29084 ± 0.00571738 (W 0.00405711) | - | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.3437 ± 0.00314421 (W 0.00389865) | - | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 5.03283 ± 0.00992008 (W 0.00574955) | - | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 5.67681 ± 0.0245936 (W 0.0154406) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.740851 ± 0.00173866 (W 0.00175083) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.58115 ± 0.00641875 (W 0.00447815) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.822515 ± 0.00325195 (W 0.00238225) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.6017 ± 0.00705898 (W 0.00508847) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.38769 ± 0.0074106 (W 0.0102149) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 7.80301 ± 0.0269625 (W 0.023006) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.4899 ± 0.00734479 (W 0.00980317) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 7.90744 ± 0.0363827 (W 0.0329888) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.6471 ± 0.0421362 (W 0.0181607) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 10.7063 ± 0.029715 (W 0.018013) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.9431 ± 0.0912775 (W 0.0249746) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 10.9883 ± 0.0433113 (W 0.0211703) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 49.1433 ± 0.0660853 (W 0.0420398) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 49.3898 ± 0.0785233 (W 0.400002) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 50.3524 ± 0.209747 (W 0.220316) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 50.4055 ± 0.0756441 (W 0.0513374) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.74291 ± 0.0286588 (W 0.0154412) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 5.74379 ± 0.0218961 (W 0.00963617) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.89055 ± 0.0289991 (W 0.0100855) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 5.88822 ± 0.0699075 (W 0.0079584) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 26.4269 ± 0.0937345 (W 0.0432562) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 26.4205 ± 0.0654763 (W 0.0561737) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 26.7438 ± 0.0453176 (W 0.0552369) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 26.8197 ± 0.0720778 (W 0.0352659) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.021425 ± 0.000136739 (W 7.76525e-05) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.02138 ± 8.86125e-05 (W 8.94409e-05) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.05065 ± 0.00101761 (W 0.000357726) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n513/b1` | 0.0499585 ± 0.000715435 (W 0.000280664) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.497793 ± 0.00153707 (W 0.00259144) | 0.71195 ± 0.00398508 (W 0.00485067) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.575399 ± 0.00125933 (W 0.00275033) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.451479 ± 0.00127287 (W 0.00184511) | 0.797342 ± 0.00595475 (W 0.00372096) | 🔵 faer-fork 1.77× faster than `lapack-aocl` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.527694 ± 0.00198712 (W 0.00241635) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.25136 ± 0.00872657 (W 0.00377523) | 2.73178 ± 0.0145519 (W 0.0117154) | 🔵 faer-fork 1.21× faster than `lapack-aocl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 2.59566 ± 0.00872416 (W 0.00479286) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.21197 ± 0.00842815 (W 0.00431198) | 2.85881 ± 0.0114067 (W 0.0138459) | 🔵 faer-fork 1.29× faster than `lapack-aocl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 2.53432 ± 0.00538753 (W 0.0042269) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.32744 ± 0.0233385 (W 0.00892055) | 20.3551 ± 0.361759 (W 0.0259211) | 🔵 faer-fork 2.44× faster than `lapack-aocl` |  |
| `eigh/n255/b1 [api-inclusive]` | 8.35734 ± 0.0217786 (W 0.01213) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.36763 ± 0.0131771 (W 0.00607822) | 20.8505 ± 0.0510446 (W 0.0180799) | 🔵 faer-fork 2.49× faster than `lapack-aocl` |  |
| `eigh/n257/b1 [api-inclusive]` | 8.39416 ± 0.0121682 (W 0.0077805) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.3252 ± 0.0704158 (W 0.0279277) | 79.4462 ± 0.372014 (W 0.0553317) | 🔵 faer-fork 1.97× faster than `lapack-aocl` |  |
| `eigh/n511/b1 [api-inclusive]` | 40.4498 ± 0.0827193 (W 0.0461702) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.2729 ± 0.0405439 (W 0.0406647) | 80.9781 ± 0.539754 (W 0.135524) | 🔵 faer-fork 2.01× faster than `lapack-aocl` |  |
| `eigh/n513/b1 [api-inclusive]` | 40.3433 ± 0.066004 (W 0.0406772) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.87371 ± 0.0154556 (W 0.00361139) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.93198 ± 0.0129771 (W 0.0081513) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.91458 ± 0.0136878 (W 0.00342) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.54692 ± 0.00928235 (W 0.00593442) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 10.758 ± 0.0352659 (W 0.0194114) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 19.7058 ± 0.0858352 (W 0.0369519) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 10.8242 ± 0.0358409 (W 0.0147982) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 13.9193 ± 0.0835779 (W 0.0895411) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.881439 ± 0.00836259 (W 0.0100124) | 0.610382 ± 0.00548858 (W 0.00297753) | 🔴 faer-fork 1.44× slower than `lapack-aocl` |  |
| `lu/n255/b1 [api-inclusive]` | 1.5695 ± 0.0106501 (W 0.00714058) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.923994 ± 0.00242022 (W 0.00319862) | 0.627139 ± 0.00299094 (W 0.00269896) | 🔴 faer-fork 1.47× slower than `lapack-aocl` |  |
| `lu/n257/b1 [api-inclusive]` | 1.59743 ± 0.00378409 (W 0.0059971) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 4.64495 ± 0.0398073 (W 0.0423857) | 4.20639 ± 0.0119486 (W 0.00392957) | 🔴 faer-fork 1.10× slower than `lapack-aocl` |  |
| `lu/n511/b1 [api-inclusive]` | 8.21952 ± 0.0734543 (W 0.033439) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 4.84202 ± 0.0127692 (W 0.0107879) | 4.24832 ± 0.00825298 (W 0.00615953) | 🔴 faer-fork 1.14× slower than `lapack-aocl` |  |
| `lu/n513/b1 [api-inclusive]` | 8.32017 ± 0.0379509 (W 0.0140775) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.96665 ± 0.0105644 (W 0.00239872) | 4.05013 ± 0.0114511 (W 0.00892535) | 🔵 faer-fork 1.37× faster than `lapack-aocl` |  |
| `qr/n255/b1 [api-inclusive]` | 4.0345 ± 0.0070697 (W 0.00536488) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.04902 ± 0.0117385 (W 0.00393729) | 4.20184 ± 0.0157489 (W 0.00918383) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `qr/n257/b1 [api-inclusive]` | 4.13439 ± 0.0322896 (W 0.00626773) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 18.9849 ± 0.0325025 (W 0.0175053) | 26.7166 ± 0.08022 (W 0.0272553) | 🔵 faer-fork 1.41× faster than `lapack-aocl` |  |
| `qr/n511/b1 [api-inclusive]` | 24.3043 ± 0.0422324 (W 0.023535) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 19.3674 ± 0.0529805 (W 0.0179619) | 27.0775 ± 0.0574576 (W 0.0294784) | 🔵 faer-fork 1.40× faster than `lapack-aocl` |  |
| `qr/n513/b1 [api-inclusive]` | 24.8801 ± 0.0187884 (W 0.0414435) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.51715 ± 0.00547748 (W 0.00472211) | 1.95353 ± 0.00795509 (W 0.00403391) | 🔵 faer-fork 1.29× faster than `lapack-aocl` |  |
| `qr_factor/n257/b1` | 1.56319 ± 0.00415324 (W 0.00599148) | 2.01063 ± 0.00953922 (W 0.00445306) | 🔵 faer-fork 1.29× faster than `lapack-aocl` |  |
| `qr_factor/n511/b1` | 8.81956 ± 0.0158257 (W 0.0187713) | 13.8672 ± 0.060537 (W 0.0166043) | 🔵 faer-fork 1.57× faster than `lapack-aocl` |  |
| `qr_factor/n513/b1` | 8.94294 ± 0.0268956 (W 0.0128836) | 13.9572 ± 0.0466703 (W 0.0164559) | 🔵 faer-fork 1.56× faster than `lapack-aocl` |  |
| `solve/n255/b1` | 0.883178 ± 0.00393168 (W 0.00603649) | 0.562893 ± 0.00290767 (W 0.00226275) | 🔴 faer-fork 1.57× slower than `lapack-aocl` |  |
| `solve/n255/b1 [api-inclusive]` | 1.58127 ± 0.0098859 (W 0.00654853) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.926875 ± 0.00310301 (W 0.00392842) | 0.588218 ± 0.00243084 (W 0.00201769) | 🔴 faer-fork 1.58× slower than `lapack-aocl` |  |
| `solve/n257/b1 [api-inclusive]` | 1.61456 ± 0.0136824 (W 0.00562921) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 4.63227 ± 0.0393021 (W 0.0299823) | 4.08441 ± 0.00926914 (W 0.00788025) | 🔴 faer-fork 1.13× slower than `lapack-aocl` |  |
| `solve/n511/b1 [api-inclusive]` | 8.27904 ± 0.0759835 (W 0.0428586) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 4.84211 ± 0.011518 (W 0.0135992) | 4.14457 ± 0.020743 (W 0.00617556) | 🔴 faer-fork 1.17× slower than `lapack-aocl` |  |
| `solve/n513/b1 [api-inclusive]` | 8.34111 ± 0.0397842 (W 0.0150803) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.8077 ± 0.0188478 (W 0.016452) | 16.7403 ± 0.0597897 (W 0.173525) | 🔵 faer-fork 1.31× faster than `lapack-aocl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 12.8843 ± 0.0269142 (W 0.0175613) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 13.1454 ± 0.0141481 (W 0.0229283) | 16.0867 ± 0.0491512 (W 0.0203384) | 🔵 faer-fork 1.22× faster than `lapack-aocl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 13.202 ± 0.0357242 (W 0.0237392) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 68.7019 ± 0.187892 (W 0.0849343) | 81.0887 ± 0.403158 (W 0.147528) | 🔵 faer-fork 1.18× faster than `lapack-aocl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 68.8853 ± 0.186512 (W 0.24888) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 70.2016 ± 0.0926719 (W 0.0639703) | 81.1506 ± 0.218968 (W 0.132412) | 🔵 faer-fork 1.16× faster than `lapack-aocl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 70.4939 ± 0.150001 (W 0.0589173) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.3883 ± 0.0129615 (W 0.0118442) | 9.47467 ± 0.0645542 (W 1.36695) | 🔵 faer-fork 1.48× faster than `lapack-aocl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 6.39902 ± 0.0178662 (W 0.0123515) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.57284 ± 0.0148362 (W 0.0142483) | 8.73285 ± 0.0248196 (W 0.0142789) | 🔵 faer-fork 1.33× faster than `lapack-aocl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 6.56553 ± 0.0111658 (W 0.00766954) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 34.1839 ± 0.129154 (W 0.0375117) | 43.2115 ± 0.233096 (W 0.0798206) | 🔵 faer-fork 1.26× faster than `lapack-aocl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 34.3502 ± 0.0833075 (W 0.0296481) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 34.7086 ± 0.10736 (W 0.0304245) | 43.1831 ± 0.215817 (W 0.0464884) | 🔵 faer-fork 1.24× faster than `lapack-aocl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 34.7877 ± 0.130455 (W 0.0321874) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.0231785 ± 0.000246783 (W 0.000177446) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.023098 ± 8.64423e-05 (W 0.000129774) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.060018 ± 0.00129617 (W 0.000375511) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n513/b1` | 0.0591865 ± 0.000613119 (W 0.000477017) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.427858 ± 0.00263962 (W 0.0023866) | 0.738475 ± 0.0108946 (W 0.0068827) | 🔵 faer-fork 1.73× faster than `lapack-aocl` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.509472 ± 0.00138328 (W 0.00263691) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.382742 ± 0.00182203 (W 0.0016416) | 0.802097 ± 0.0104962 (W 0.00749513) | 🔵 faer-fork 2.10× faster than `lapack-aocl` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.459321 ± 0.00165282 (W 0.00201316) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.55036 ± 0.00416414 (W 0.00365515) | 2.23476 ± 0.0128547 (W 0.0239964) | 🔵 faer-fork 1.44× faster than `lapack-aocl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 1.92307 ± 0.0121918 (W 0.00659232) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.47981 ± 0.00525714 (W 0.00451564) | 2.27112 ± 0.0161842 (W 0.0151873) | 🔵 faer-fork 1.53× faster than `lapack-aocl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 1.82655 ± 0.00372587 (W 0.0039573) | - | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.5523 ± 0.0105967 (W 0.00975953) | 21.1682 ± 0.167626 (W 0.0316946) | 🔵 faer-fork 3.81× faster than `lapack-aocl` |  |
| `eigh/n255/b1 [api-inclusive]` | 5.58504 ± 0.0158437 (W 0.011627) | - | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.53786 ± 0.0237586 (W 0.0105649) | 21.4835 ± 0.144176 (W 0.0470784) | 🔵 faer-fork 3.88× faster than `lapack-aocl` |  |
| `eigh/n257/b1 [api-inclusive]` | 5.55782 ± 0.0209304 (W 0.00852314) | - | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.1925 ± 0.0391744 (W 0.0206036) | 73.3227 ± 0.373727 (W 0.0735991) | 🔵 faer-fork 2.91× faster than `lapack-aocl` |  |
| `eigh/n511/b1 [api-inclusive]` | 25.3298 ± 0.052522 (W 0.0169275) | - | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.7213 ± 0.0830134 (W 0.0494163) | 73.8452 ± 0.223216 (W 0.373456) | 🔵 faer-fork 2.87× faster than `lapack-aocl` |  |
| `eigh/n513/b1 [api-inclusive]` | 25.8462 ± 0.0848839 (W 0.0480606) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.50788 ± 0.0742697 (W 0.00246127) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.55862 ± 0.00949422 (W 0.00854195) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.53756 ± 0.0157775 (W 0.00294654) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.19338 ± 0.0151099 (W 0.00658448) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.0704 ± 0.0195354 (W 0.0117857) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 16.0806 ± 0.0344877 (W 0.0280168) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.11855 ± 0.0343896 (W 0.0146999) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 10.3505 ± 0.0331213 (W 0.0403288) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.717445 ± 0.00435848 (W 0.00199841) | 0.473579 ± 0.00379142 (W 0.00225428) | 🔴 faer-fork 1.51× slower than `lapack-aocl` |  |
| `lu/n255/b1 [api-inclusive]` | 1.40823 ± 0.00514453 (W 0.00462667) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.734088 ± 0.000403137 (W 0.00190425) | 0.497619 ± 0.00738817 (W 0.00336406) | 🔴 faer-fork 1.48× slower than `lapack-aocl` |  |
| `lu/n257/b1 [api-inclusive]` | 1.42167 ± 0.00336339 (W 0.00323282) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.13518 ± 0.0242285 (W 0.00623101) | 2.16057 ± 0.00551588 (W 0.00554267) | 🔴 faer-fork 1.45× slower than `lapack-aocl` |  |
| `lu/n511/b1 [api-inclusive]` | 6.73915 ± 0.0553084 (W 0.0264422) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.18971 ± 0.00898042 (W 0.00668504) | 2.19718 ± 0.0189402 (W 0.012877) | 🔴 faer-fork 1.45× slower than `lapack-aocl` |  |
| `lu/n513/b1 [api-inclusive]` | 6.75482 ± 0.056453 (W 0.0163286) | - | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.01564 ± 0.0361245 (W 0.00565498) | 2.77344 ± 0.00814789 (W 0.0127432) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `qr/n255/b1 [api-inclusive]` | 3.08489 ± 0.00879354 (W 0.0115655) | - | - | `*` one row has no measurement here |
| `qr/n257/b1` | 2.0651 ± 0.00652004 (W 0.00212105) | 2.96197 ± 0.00961465 (W 0.019661) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `qr/n257/b1 [api-inclusive]` | 3.16601 ± 0.0149546 (W 0.00777975) | - | - | `*` one row has no measurement here |
| `qr/n511/b1` | 9.9666 ± 0.0402214 (W 0.0144726) | 15.7193 ± 0.0427945 (W 0.0157359) | 🔵 faer-fork 1.58× faster than `lapack-aocl` |  |
| `qr/n511/b1 [api-inclusive]` | 15.3105 ± 0.0620088 (W 0.0229918) | - | - | `*` one row has no measurement here |
| `qr/n513/b1` | 10.6431 ± 0.0269596 (W 0.0180179) | 16.0642 ± 0.0523739 (W 0.0232426) | 🔵 faer-fork 1.51× faster than `lapack-aocl` |  |
| `qr/n513/b1 [api-inclusive]` | 16.3022 ± 0.0528652 (W 0.0464884) | - | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.24484 ± 0.00849374 (W 0.00405609) | 1.43677 ± 0.0072228 (W 0.00725971) | 🔵 faer-fork 1.15× faster than `lapack-aocl` |  |
| `qr_factor/n257/b1` | 1.27045 ± 0.00448109 (W 0.00430596) | 1.51045 ± 0.0104841 (W 0.00716605) | 🔵 faer-fork 1.19× faster than `lapack-aocl` |  |
| `qr_factor/n511/b1` | 4.72398 ± 0.0162249 (W 0.00769804) | 8.1525 ± 0.0458445 (W 0.0192448) | 🔵 faer-fork 1.73× faster than `lapack-aocl` |  |
| `qr_factor/n513/b1` | 5.26563 ± 0.0171026 (W 0.00954125) | 8.2835 ± 0.0158727 (W 0.0136369) | 🔵 faer-fork 1.57× faster than `lapack-aocl` |  |
| `solve/n255/b1` | 0.718308 ± 0.00336652 (W 0.00263804) | 0.439897 ± 0.00374126 (W 0.00177629) | 🔴 faer-fork 1.63× slower than `lapack-aocl` |  |
| `solve/n255/b1 [api-inclusive]` | 1.41947 ± 0.00893203 (W 0.00708144) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.734687 ± 0.00331099 (W 0.00259393) | 0.474125 ± 0.00480291 (W 0.00207848) | 🔴 faer-fork 1.55× slower than `lapack-aocl` |  |
| `solve/n257/b1 [api-inclusive]` | 1.43365 ± 0.00469318 (W 0.0071189) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.1195 ± 0.0106072 (W 0.00547647) | 2.12121 ± 0.00156256 (W 0.0113958) | 🔴 faer-fork 1.47× slower than `lapack-aocl` |  |
| `solve/n511/b1 [api-inclusive]` | 6.70234 ± 0.0296665 (W 0.019504) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.1684 ± 0.0112466 (W 0.00585508) | 2.17403 ± 0.00936792 (W 0.00709418) | 🔴 faer-fork 1.46× slower than `lapack-aocl` |  |
| `solve/n513/b1 [api-inclusive]` | 6.76734 ± 0.0588869 (W 0.0157272) | - | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.6063 ± 0.0286981 (W 0.0256481) | 16.7259 ± 3.14808 (W 3.97057) | 🔵 faer-fork 1.58× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/n255/b1 [api-inclusive]` | 10.622 ± 0.0480194 (W 0.0137381) | - | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.8431 ± 0.0439835 (W 0.0170461) | 14.0742 ± 0.0352985 (W 0.0211447) | 🔵 faer-fork 1.30× faster than `lapack-aocl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 10.8734 ± 0.0346378 (W 0.0312055) | - | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 48.9704 ± 0.0640279 (W 0.186834) | 63.5477 ± 0.249368 (W 0.14612) | 🔵 faer-fork 1.30× faster than `lapack-aocl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 49.042 ± 0.0934695 (W 0.0504132) | - | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 50.0924 ± 0.121087 (W 0.0648353) | 61.5694 ± 0.284961 (W 0.162691) | 🔵 faer-fork 1.23× faster than `lapack-aocl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 50.3363 ± 0.206336 (W 0.0537917) | - | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.74061 ± 0.0207082 (W 0.00698266) | 11.9411 ± 4.35453 (W 6.98067) | 🔵 faer-fork 2.08× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_values/n255/b1 [api-inclusive]` | 5.7473 ± 0.0173706 (W 0.00836107) | - | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.90991 ± 0.0213023 (W 0.0148309) | 8.19219 ± 0.0818744 (W 0.0210241) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 5.91429 ± 0.0169834 (W 0.0112504) | - | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 26.1269 ± 0.0794833 (W 0.0234159) | 37.2467 ± 0.245147 (W 4.94596) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 26.1833 ± 0.379606 (W 0.0559777) | - | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 26.5034 ± 0.0526048 (W 0.0379181) | 35.1437 ± 0.214812 (W 0.167749) | 🔵 faer-fork 1.33× faster than `lapack-aocl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 26.5307 ± 0.0912794 (W 0.0528001) | - | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.022848 ± 0.000136826 (W 0.000156649) | - | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.0231835 ± 0.000134001 (W 0.000116659) | - | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.0567855 ± 0.000362891 (W 0.00036291) | - | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.056381 ± 0.000293284 (W 0.000780439) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/255`, `cholesky/257`, `cholesky/511`, `cholesky/513`, `eigh/255`, `eigh/257`, `eigh/511`, `eigh/513`, `full_piv_lu/255`, `full_piv_lu/257`, `full_piv_lu/511`, `full_piv_lu/513`, `lu/255`, `lu/257`, `lu/511`, `lu/513`, `qr/255`, `qr/257`, `qr/511`, `qr/513`, `qr_factor/255`, `qr_factor/257`, `qr_factor/511`, `qr_factor/513`, `solve/255`, `solve/257`, `solve/511`, `solve/513`, `svd_thin/255`, `svd_thin/257`, `svd_thin/511`, `svd_thin/513`, `svd_values/255`, `svd_values/257`, `svd_values/511`, `svd_values/513`, `triangular_solve/255`, `triangular_solve/257`, `triangular_solve/511`, `triangular_solve/513`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
