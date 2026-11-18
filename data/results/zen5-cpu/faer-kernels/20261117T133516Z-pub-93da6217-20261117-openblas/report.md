# `faer-kernels` on `zen5-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `e3248beacb919a55d9ffbc7346844431108e6eae`
- toolkit commit: `b7299cf625c537a37eef2782e8759445f1d8aef8`
- measured binary SHA-256: `45d173c5ac2a0059cc16293632f6bf72b2dbc9177b105e2743f79ec8e7975717`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-17T18:03:02.066173Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `/home/shinaoka/projects/tensor4all/colmat-benchmark/scripts/record_run.py zen5-cpu faer-kernels --checkout /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/t4a-faer --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor openblas --engines faer-fork,lapack-openblas --reps 5 --aa 6 --prime-ms 250 --label pub-93da6217-20261117-openblas --no-publish --threads 1,4,8`
- raw data: `data/results/zen5-cpu/faer-kernels/20261117T133516Z-pub-93da6217-20261117-openblas/`

## What was measured

- **Workload** — Direct calls into one faer revision: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: No row allocates inside its timed `run`: the vendor row calls LAPACKE's `_work` entry points with a workspace the op allocated in `prepare`, and the faer rows call their in-place/`*_into` routes on a prepared matrix, prepared rights and a `MemStack` sized by the library's own `*_scratch`. A vendored driver family is measured column-major only, because LAPACKE's row-major path allocates a transposed copy and transposes in and out of it inside the call in every `_work` variant; a C-order cell of such a family is recorded `skipped` with that reason rather than charged a conversion the faer row does not pay. `gemm` keeps both orders (CBLAS's row-major entry point moves no data).
  Known blind spot: The owning faer routes (`PartialPivLu::new`, `Llt::new`, `qr`, `gesdd`/`syevd` wrappers, `partial_piv_lu().solve()`) allocate their outputs and their scratch inside the timed call. They are kept, as the call a user makes, but labelled `api-inclusive` and published as a separate secondary table: an `api-inclusive` row is never divided against a vendor row, whose route is the prepared one.
  Known blind spot: A library's own allocation inside a timed call is not visible to the harness's allocator probe when it happens in C: LAPACKE's non-`_work` wrappers reach libc `malloc` through `LAPACKE_malloc`. That route is not used by this harness (the probe would not see it), so the `_work` entry points are enforced by construction and by the column-major rule, not by the probe.
  Known blind spot: `gemm` covers the tall class's two Householder legs (a 16-wide basis and a 240-column target over 4096 rows: the reduction and the write-back) beside a vendor `?gemm` on the same leg. A single leg beside a vendor gemm is a diagnostic reference, not the equivalent implementation of the whole update, and the row is labelled accordingly.
  Known blind spot: `householder_apply` measures block apply and the one-block sequence form as two distinct rows, labelled with basis width min(m,n) and target width n. QR fixture preparation is outside timing. Independently varied basis/target widths and multi-panel sequences are not covered.
  Known blind spot: A copy a library makes inside its own call is that library's cost, and it is never divided against a vendor row: on a row-major matrix the fork's LU factors into a column-major copy it allocates (`lu/partial_pivoting/factor.rs:396-418`), and a row-major cell of a vendored driver family is a `skipped` cell, so no vendor ratio is formed there. Allocation performed inside the measured call stays inside the clock, and no _primary_ row does it. The fork's two row splits allocate their partial buffers inside the public call and the public scratch function does not expose them; that is the fork's cost, charged to the fork, and the owning routes that pay it are published as `api-inclusive` rows rather than divided against a vendor row.
  Known blind spot: `cholesky`, `eigh`, `solve` and `triangular_solve` are measured on square shapes only: the matrix they factor or solve with is square by definition (self-adjoint for the first two, a triangular factor for the last). `lu` and `full_piv_lu` are not gated that way -- they take the declared rectangular shapes too, with `min(m, n)` pivots and a trapezoidal `L` or `U`, which is what `dgetrf` does as well -- so a rectangular LU cell is measured in both orders beside its square one.
  Known blind spot: `f32`/`c32` are declared and measured as of `docs/harness-dtype-axis.md`; the vendor arm has a route for `gemm` in those two dtypes and reports the LAPACK families as skipped for them. The large regime's 1024 shapes, BLIS, sparse, iterative and GPU arms are later declarations, not machinery built now.
  Known blind spot: A vendor arm records the library it linked. Only OpenBLAS reports the kernel its runtime dispatched to; MKL names no kernel and no processor count, so its manifest leaves both empty rather than inferring them.
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
- OS / arch: `Linux 7.0.0-38-generic` / `x86_64`
- hostname: `shinaoka-EVO-X1Pro`
- CPU sets: 1T -> `4`, 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.32%, worst 23647.29% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 373 of 5064 engine cells exceed 5% and carry `!`; median within-process relative SD 0.43%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

## gemm-small, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002364 ± 8.25631e-06 (W 2.72534e-05) | 0.002234 ± 8.94427e-06 (W 2.24685e-05) | 🔴 faer-fork 1.06× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.017117 ± 4.66065e-05 (W 4.13874e-05) | 0.0168515 ± 5.83747e-05 (W 4.09072e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.056816 ± 6.78233e-05 (W 9.08319e-05) | 0.056395 ± 0.000281079 (W 6.30421e-05) | · |  |

## gemm-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002414 ± 9.1815e-06 (W 2.81904e-05) | 0.002214 ± 1.24967e-05 (W 2.21636e-05) | 🔴 faer-fork 1.09× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.011176 ± 0.000180012 (W 0.000182998) | 0.0168615 ± 5.5955e-05 (W 0.000636582) | 🔵 faer-fork 1.51× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.022883 ± 9.62713e-05 (W 0.00046105) | 0.0563105 ± 5.96121e-05 (W 7.40688e-05) | 🔵 faer-fork 2.46× faster than `lapack-openblas` |  |

## gemm-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002425 ± 1.76144e-05 (W 2.93723e-05) | 0.002209 ± 1.97019e-05 (W 2.7143e-05) | 🔴 faer-fork 1.10× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0096325 ± 6.1418e-05 (W 6.53208e-05) | 0.016752 ± 8.44214e-05 (W 3.58078e-05) | 🔵 faer-fork 1.74× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0200075 ± 7.34411e-05 (W 7.03508e-05) | 0.05621 ± 0.000225657 (W 0.000128535) | 🔵 faer-fork 2.81× faster than `lapack-openblas` |  |

## gemm-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0023495 ± 1.95857e-05 (W 2.79483e-05) | 0.002234 ± 0 (W 2.44872e-05) | 🔴 faer-fork 1.05× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.017147 ± 0.000110858 (W 5.15609e-05) | 0.016862 ± 6.22115e-05 (W 4.86373e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.056801 ± 0.000250608 (W 8.96955e-05) | 0.056405 ± 0.000202768 (W 7.03963e-05) | · |  |

## gemm-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002424 ± 1.51526e-05 (W 2.55291e-05) | 0.002204 ± 8.94427e-06 (W 2.44736e-05) | 🔴 faer-fork 1.10× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0111505 ± 0.000111571 (W 0.000174213) | 0.0168665 ± 9.95704e-05 (W 2.59662e-05) | 🔵 faer-fork 1.51× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.023223 ± 0.000145438 (W 0.000390115) | 0.0561945 ± 0.000380548 (W 4.28332e-05) | 🔵 faer-fork 2.42× faster than `lapack-openblas` |  |

## gemm-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002419 ± 1.68612e-05 (W 2.25878e-05) | 0.002219 ± 2.09762e-05 (W 2.34215e-05) | 🔴 faer-fork 1.09× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.009643 ± 5.4191e-05 (W 7.07405e-05) | 0.016881 ± 9.39503e-05 (W 2.2331e-05) | 🔵 faer-fork 1.75× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.019982 ± 6.71565e-05 (W 5.66537e-05) | 0.056445 ± 0.000184441 (W 9.39541e-05) | 🔵 faer-fork 2.82× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0088215 ± 8.42417e-06 (W 2.53307e-05) | 0.0115915 ± 6.32163e-05 (W 3.47516e-05) | 🔵 faer-fork 1.31× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.067766 ± 0.000191459 (W 0.000609796) | 0.086156 ± 0.000209801 (W 0.000162214) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.227192 ± 0.000420805 (W 0.00181975) | 0.276751 ± 0.000975152 (W 0.00193486) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009573 ± 0.000152583 (W 0.000257176) | 0.0115665 ± 4.06321e-05 (W 4.47896e-05) | 🔵 faer-fork 1.21× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.028679 ± 0.000171337 (W 0.000198766) | 0.0260535 ± 0.000110958 (W 0.000215392) | 🔴 faer-fork 1.10× slower than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.0782465 ± 0.000391998 (W 0.00043458) | 0.0793175 ± 0.000273876 (W 0.00101281) | · |  |

## gemm-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009698 ± 5.36656e-05 (W 9.40665e-05) | 0.0115515 ± 5.30563e-05 (W 3.93237e-05) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.026545 ± 0.000151683 (W 7.53184e-05) | 0.026184 ± 0.000100133 (W 0.000238481) | · |  |
| `gemm/96x96x96/b1` | 0.039689 ± 0.000243171 (W 0.000138443) | 0.055468 ± 0.000170488 (W 0.00086074) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |

## gemm-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0088215 ± 1.42466e-05 (W 2.45085e-05) | 0.0115915 ± 1.6561e-05 (W 6.6854e-05) | 🔵 faer-fork 1.31× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0676715 ± 9.98412e-05 (W 6.87306e-05) | 0.086075 ± 0.000247019 (W 0.000122901) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.226847 ± 0.000692075 (W 0.0019644) | 0.276632 ± 0.000969615 (W 0.00178874) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009538 ± 0.000145934 (W 9.87567e-05) | 0.0115965 ± 5.88911e-05 (W 5.86274e-05) | 🔵 faer-fork 1.22× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.028834 ± 0.000183485 (W 0.000186077) | 0.026224 ± 0.00015288 (W 0.000291749) | 🔴 faer-fork 1.10× slower than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.078266 ± 0.000283119 (W 0.00038901) | 0.0793935 ± 0.000489237 (W 0.000971721) | · |  |

## gemm-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009648 ± 6.5549e-05 (W 0.000101653) | 0.011606 ± 4.32235e-05 (W 4.47201e-05) | 🔵 faer-fork 1.20× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.02662 ± 0.000144486 (W 7.98377e-05) | 0.0261085 ± 0.000210105 (W 0.000241259) | · |  |
| `gemm/96x96x96/b1` | 0.0395435 ± 0.000277263 (W 9.21409e-05) | 0.0553235 ± 0.000204031 (W 0.000303898) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |

## gemm-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0012675 ± 7.65001e-05 (W 1.667e-05) | 0.001162 ± 1.16905e-05 (W 2.72836e-05) | 🔴 faer-fork 1.09× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0086365 ± 4.63195e-05 (W 2.26599e-05) | 0.008471 ± 4.57839e-05 (W 2.55691e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.028493 ± 3.43536e-05 (W 3.46217e-05) | 0.0282275 ± 0.000228381 (W 4.01119e-05) | · |  |

## gemm-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001323 ± 1.90333e-05 (W 2.87025e-05) | 0.001157 ± 8.16497e-06 (W 2.54751e-05) | 🔴 faer-fork 1.14× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.006221 ± 0.000171146 (W 0.00018351) | 0.008461 ± 3.07679e-05 (W 2.92531e-05) | 🔵 faer-fork 1.36× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.012859 ± 0.000143046 (W 0.00018068) | 0.0282425 ± 0.000104863 (W 0.000506084) | 🔵 faer-fork 2.20× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001327 ± 3.45871e-05 (W 2.41152e-05) | 0.001157 ± 1.7224e-05 (W 2.47731e-05) | 🔴 faer-fork 1.15× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.005345 ± 8.71956e-05 (W 7.54359e-05) | 0.0084605 ± 2.70851e-05 (W 2.80166e-05) | 🔵 faer-fork 1.58× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.010725 ± 9.91295e-05 (W 7.64228e-05) | 0.0282075 ± 0.000110286 (W 4.04876e-05) | 🔵 faer-fork 2.63× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0012575 ± 8.18332e-06 (W 1.51576e-05) | 0.001152 ± 1.09545e-05 (W 2.62918e-05) | 🔴 faer-fork 1.09× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.008631 ± 2.46069e-05 (W 3.15199e-05) | 0.008461 ± 4.05976e-05 (W 2.75585e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0284985 ± 7.22523e-05 (W 6.39322e-05) | 0.0282175 ± 2.79815e-05 (W 2.7141e-05) | · |  |

## gemm-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001333 ± 7.63326e-06 (W 2.09645e-05) | 0.001157 ± 1.6171e-05 (W 2.20397e-05) | 🔴 faer-fork 1.15× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0062065 ± 0.000101171 (W 0.000165111) | 0.008446 ± 4.28972e-05 (W 3.19277e-05) | 🔵 faer-fork 1.36× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.013024 ± 0.000112114 (W 0.000261216) | 0.028212 ± 0.000149961 (W 3.31205e-05) | 🔵 faer-fork 2.17× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001333 ± 1.22583e-05 (W 2.60818e-05) | 0.001152 ± 1.36626e-05 (W 2.85867e-05) | 🔴 faer-fork 1.16× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0054255 ± 4.69628e-05 (W 9.93544e-05) | 0.0084305 ± 3.30192e-05 (W 2.05759e-05) | 🔵 faer-fork 1.55× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.010735 ± 8.21381e-05 (W 7.3698e-05) | 0.0282025 ± 0.000163212 (W 9.57682e-05) | 🔵 faer-fork 2.63× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0045185 ± 2.35004e-05 (W 2.51351e-05) | 0.00551 ± 1.99039e-05 (W 2.56404e-05) | 🔵 faer-fork 1.22× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.033983 ± 6.32469e-05 (W 0.000765329) | 0.038632 ± 0.000125656 (W 0.000231574) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.113336 ± 0.000370061 (W 0.00023514) | 0.123335 ± 0.000252063 (W 0.000824222) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005275 ± 0.000123062 (W 0.000137792) | 0.0055055 ± 2.52223e-05 (W 2.68668e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.019902 ± 0.0001394 (W 8.07418e-05) | 0.012148 ± 0.000162069 (W 0.000244811) | 🔴 faer-fork 1.64× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.041903 ± 0.000299238 (W 0.000176398) | 0.035937 ± 0.000154386 (W 0.000301253) | 🔴 faer-fork 1.17× slower than `lapack-openblas` |  |

## gemm-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005495 ± 0.000224922 (W 0.000160958) | 0.005501 ± 2.62583e-05 (W 2.69161e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.017993 ± 0.000131546 (W 6.31843e-05) | 0.012363 ± 0.000272393 (W 0.000307645) | 🔴 faer-fork 1.46× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0388425 ± 0.000128977 (W 6.92916e-05) | 0.025773 ± 0.000112313 (W 0.000186745) | 🔴 faer-fork 1.51× slower than `lapack-openblas` |  |

## gemm-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.004499 ± 2.12195e-05 (W 3.20789e-05) | 0.005485 ± 1.87083e-05 (W 2.15057e-05) | 🔵 faer-fork 1.22× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0339985 ± 0.000108663 (W 3.36357e-05) | 0.0386065 ± 0.000136666 (W 0.000126537) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.113551 ± 0.000596982 (W 0.000594722) | 0.123446 ± 0.000398373 (W 0.00174085) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005345 ± 0.000169974 (W 0.000126651) | 0.00552 ± 2.04124e-05 (W 1.81478e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0199375 ± 0.000178427 (W 0.000106054) | 0.012343 ± 0.000250703 (W 0.000290364) | 🔴 faer-fork 1.62× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.041953 ± 0.000216149 (W 0.000183979) | 0.035922 ± 0.000153888 (W 0.000377037) | 🔴 faer-fork 1.17× slower than `lapack-openblas` |  |

## gemm-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00541 ± 0.000228123 (W 0.000196587) | 0.00553 ± 2.60768e-05 (W 2.25975e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.018064 ± 0.000107595 (W 8.4733e-05) | 0.0122075 ± 5.04529e-05 (W 0.000179818) | 🔴 faer-fork 1.48× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.038943 ± 0.000160587 (W 8.89499e-05) | 0.0259085 ± 0.000150466 (W 0.000178644) | 🔴 faer-fork 1.50× slower than `lapack-openblas` |  |

## gemm-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.133244 ± 0.000892939 (W 0.00230737) | 0.162193 ± 0.000632276 (W 0.00174078) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.453047 ± 0.00836633 (W 0.00347358) | 0.538536 ± 0.0127883 (W 0.00371963) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 1.07401 ± 0.0470077 (W 0.00249726) | 1.26279 ± 0.0416184 (W 0.00349338) | 🔵 faer-fork 1.18× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 3.61731 ± 0.10439 (W 0.0090714) | 4.21407 ± 0.141249 (W 0.00857802) | 🔵 faer-fork 1.16× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 8.57802 ± 0.258216 (W 0.00777218) | 10.0072 ± 0.215283 (W 0.0144804) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 29.3747 ± 0.767837 (W 0.0658162) | 33.7098 ± 0.59369 (W 0.0304995) | 🔵 faer-fork 1.15× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0451745 ± 0.0003957 (W 0.000308433) | 0.050524 ± 4.87742 (W 0.000710399) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.119553 ± 0.00101249 (W 0.00121466) | 0.148832 ± 0.00106153 (W 0.0019182) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.290291 ± 0.0011892 (W 0.00186333) | 0.342925 ± 0.00222358 (W 0.00238485) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.916192 ± 0.00235044 (W 0.00247703) | 1.10126 ± 0.0058346 (W 0.00495716) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.20864 ± 0.010001 (W 0.00686849) | 2.57416 ± 0.0152447 (W 0.00755084) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 7.48876 ± 0.0272235 (W 0.0105983) | 8.71476 ± 0.0271057 (W 0.0147009) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.032039 ± 0.000340071 (W 0.000640718) | 4.51205 ± 4.91101 (W 0.00186376) | 🔵 faer-fork 140.83× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.0862205 ± 0.000278963 (W 0.000611988) | 0.0842725 ± 0.000681855 (W 0.000642508) | · |  |
| `gemm/256x256x256/b1` | 0.165479 ± 0.00170893 (W 0.00124406) | 0.182926 ± 0.00114417 (W 0.0014993) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.480557 ± 0.0011572 (W 0.0031878) | 0.557771 ± 0.00184699 (W 0.00223654) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.1386 ± 0.00279531 (W 0.00421539) | 1.32277 ± 0.00354216 (W 0.00339039) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 3.90472 ± 0.0308587 (W 0.0110938) | 4.55841 ± 0.0391577 (W 0.0087976) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.133759 ± 0.000357291 (W 0.00151594) | 0.162617 ± 0.000510279 (W 0.00200508) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.451315 ± 0.00684474 (W 0.00261031) | 0.538175 ± 0.013786 (W 0.00309603) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 1.06806 ± 0.0294964 (W 0.00142165) | 1.26505 ± 0.0352873 (W 0.00278341) | 🔵 faer-fork 1.18× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 3.5993 ± 0.155996 (W 0.00600365) | 4.22634 ± 0.132335 (W 0.00576779) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 8.57855 ± 0.307482 (W 0.0155255) | 10.0409 ± 0.207095 (W 0.0149514) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 29.2901 ± 0.940095 (W 0.0393437) | 33.79 ± 0.430846 (W 0.0294874) | 🔵 faer-fork 1.15× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0454195 ± 0.000157015 (W 0.000343754) | 0.0501835 ± 0.000369205 (W 0.0013845) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.119968 ± 0.000925911 (W 0.00138055) | 0.149348 ± 0.000246898 (W 0.00145721) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.29007 ± 0.00131611 (W 0.00267246) | 0.34077 ± 0.00116587 (W 0.0024099) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.916757 ± 0.00444143 (W 0.00311149) | 1.1016 ± 0.00781467 (W 0.00259385) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.20793 ± 0.00558192 (W 0.00636713) | 2.57412 ± 0.00304408 (W 0.00684914) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 7.50741 ± 0.0235235 (W 0.00929929) | 8.7042 ± 0.0544356 (W 0.0356988) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0318845 ± 0.00032479 (W 0.0010148) | 0.029741 ± 0.000993145 (W 0.000546646) | 🔴 faer-fork 1.07× slower than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.085695 ± 0.000474538 (W 0.000876843) | 0.084152 ± 0.000571271 (W 0.000586457) | · |  |
| `gemm/256x256x256/b1` | 0.164718 ± 0.000924057 (W 0.00172997) | 0.182645 ± 0.000806341 (W 0.00194321) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.478707 ± 0.00165642 (W 0.00328022) | 0.556533 ± 0.0032747 (W 0.00211193) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.13909 ± 0.00210218 (W 0.00337711) | 1.32148 ± 0.00265013 (W 0.00444418) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 3.90405 ± 0.0165695 (W 0.0114067) | 4.55427 ± 0.00928253 (W 0.00844033) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.536913 ± 0.00292606 (W 0.00242975) | 0.656675 ± 0.00733508 (W 0.00217143) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 1.80557 ± 0.00598935 (W 0.0023852) | 2.17123 ± 0.00477636 (W 0.00381333) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 4.26809 ± 0.0120974 (W 0.00614202) | 5.15639 ± 0.0120772 (W 0.00653947) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 14.402 ± 0.0265855 (W 0.0190181) | 17.2522 ± 0.051727 (W 0.0279312) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 34.2205 ± 0.0754573 (W 0.0421518) | 40.8157 ± 0.100531 (W 0.0439566) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 116.221 ± 0.259317 (W 0.132076) | 137.044 ± 0.335193 (W 0.364919) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.15342 ± 0.000741612 (W 0.000541517) | 0.17734 ± 0.00130673 (W 0.00165061) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.458252 ± 0.00281914 (W 0.00245522) | 0.559461 ± 0.00263656 (W 0.00258054) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.10992 ± 0.00413198 (W 0.00198337) | 1.33296 ± 0.00495454 (W 0.00329294) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 3.62523 ± 0.0113621 (W 0.00899885) | 4.39436 ± 0.0124293 (W 0.00779334) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 8.64093 ± 0.0134451 (W 0.0122765) | 10.433 ± 0.0369201 (W 0.0197338) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 29.4357 ± 0.0719471 (W 0.0344415) | 34.7917 ± 0.0839311 (W 0.0543562) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.102411 ± 0.000509993 (W 0.000352672) | 0.094351 ± 0.000491338 (W 0.000319253) | 🔴 faer-fork 1.09× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.23557 ± 0.000796752 (W 0.00165908) | 0.297084 ± 0.00112658 (W 0.00256522) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.583941 ± 0.00100581 (W 0.00221774) | 0.685987 ± 0.034106 (W 0.00198221) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 1.83516 ± 0.00423469 (W 0.0053815) | 2.2073 ± 0.00465854 (W 0.00345928) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 4.46425 ± 0.0145078 (W 0.0087395) | 5.25977 ± 0.00940456 (W 0.00664632) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 15.0086 ± 0.0333576 (W 0.0198727) | 17.7301 ± 0.0539333 (W 0.0168832) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.536606 ± 0.00107869 (W 0.00243618) | 0.656054 ± 0.00220628 (W 0.00237807) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 1.80588 ± 0.00563605 (W 0.00457594) | 2.17303 ± 0.00596298 (W 0.0057536) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 4.26271 ± 0.0117255 (W 0.00463645) | 5.15471 ± 0.0142176 (W 0.00893914) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 14.3956 ± 0.0383962 (W 0.0230035) | 17.2681 ± 0.0508733 (W 0.0162215) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 34.2077 ± 0.0644376 (W 0.0372762) | 40.8236 ± 0.116755 (W 0.034823) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 116.16 ± 0.302244 (W 0.0799654) | 137.087 ± 0.306147 (W 0.0977853) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.153481 ± 0.000507944 (W 0.00064961) | 0.177475 ± 0.00102465 (W 0.00185934) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.457047 ± 0.00679742 (W 0.00211512) | 0.560348 ± 0.00275247 (W 0.00174178) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.11025 ± 0.00456691 (W 0.00277505) | 1.33171 ± 0.00450041 (W 0.00462519) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 3.63003 ± 0.0130992 (W 0.00821279) | 4.38912 ± 0.00409781 (W 0.00715602) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 8.66153 ± 0.020993 (W 0.0229108) | 10.4237 ± 0.0326385 (W 0.0163109) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 29.365 ± 0.0864329 (W 0.04597) | 34.8488 ± 0.0612977 (W 0.0355767) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.102431 ± 0.00056565 (W 0.00025253) | 0.0943405 ± 0.00102999 (W 0.000422453) | 🔴 faer-fork 1.09× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.235474 ± 0.000767835 (W 0.00198962) | 0.296669 ± 0.000545912 (W 0.00200557) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.583372 ± 0.000718466 (W 0.00235314) | 0.68513 ± 0.00118902 (W 0.00230107) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.8363 ± 0.00829936 (W 0.00422101) | 2.20849 ± 0.115462 (W 0.00295885) | 🔵 faer-fork 1.20× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 4.45603 ± 0.00906658 (W 0.00586412) | 5.26253 ± 0.0102946 (W 0.00598112) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 15.0333 ± 0.0156056 (W 0.0511967) | 17.7382 ± 0.0439706 (W 0.0151039) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0668795 ± 0.000251774 (W 0.00108072) | 0.07263 ± 0.000547039 (W 7.42607e-05) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.223405 ± 0.000678837 (W 0.00189285) | 0.234706 ± 0.00191052 (W 0.00202131) | · |  |
| `gemm/256x256x256/b1` | 0.531593 ± 0.00201129 (W 0.00236491) | 0.559591 ± 0.00421996 (W 0.00233914) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.79514 ± 0.0051 (W 0.00265609) | 1.86434 ± 0.00900573 (W 0.00272817) | · |  |
| `gemm/512x512x512/b1` | 4.25368 ± 0.0121667 (W 0.00445313) | 4.41325 ± 0.0283455 (W 0.0069161) | · |  |
| `gemm/768x768x768/b1` | 14.4745 ± 0.0405967 (W 0.0191668) | 14.7782 ± 0.0568597 (W 0.0186302) | · |  |

## gemm-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.02146 ± 0.000307346 (W 0.000420682) | 0.023814 ± 0.000180251 (W 0.000599129) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.083731 ± 0.000256399 (W 0.00055362) | 0.0702605 ± 0.000497437 (W 0.00133696) | 🔴 faer-fork 1.19× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.157212 ± 0.000675863 (W 0.000350249) | 0.155505 ± 0.000996293 (W 0.00193966) | · |  |
| `gemm/384x384x384/b1` | 0.465794 ± 0.00295059 (W 0.00393844) | 0.497004 ± 0.00148373 (W 0.00229257) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.11472 ± 0.003003 (W 0.00165408) | 1.15219 ± 0.00455517 (W 0.00462215) | · |  |
| `gemm/768x768x768/b1` | 3.67116 ± 0.0107469 (W 0.00577525) | 3.79025 ± 0.0141824 (W 0.00785121) | · |  |

## gemm-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.021595 ± 0.000213389 (W 0.000464865) | 0.015078 ± 0.000226989 (W 0.00040302) | 🔴 faer-fork 1.43× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.0486505 ± 0.000448474 (W 0.000824758) | 0.039944 ± 0.000312105 (W 0.000487835) | 🔴 faer-fork 1.22× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.10757 ± 0.000705895 (W 0.000711769) | 0.083876 ± 0.000467038 (W 0.000917308) | 🔴 faer-fork 1.28× slower than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.249484 ± 0.000501794 (W 0.0020777) | 0.256417 ± 0.00157794 (W 0.00164249) | · |  |
| `gemm/512x512x512/b1` | 0.598084 ± 0.00119389 (W 0.00258963) | 0.59193 ± 0.00224414 (W 0.00243245) | · |  |
| `gemm/768x768x768/b1` | 1.88746 ± 0.00592968 (W 0.00456859) | 1.90848 ± 0.0107861 (W 0.00618231) | · |  |

## gemm-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0666845 ± 0.000207234 (W 0.000656615) | 0.0722895 ± 0.000276544 (W 0.000740003) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.223942 ± 0.00123344 (W 0.00186967) | 0.23605 ± 0.00180387 (W 0.00160182) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.533277 ± 0.00274856 (W 0.00299002) | 0.560331 ± 0.00226027 (W 0.00213333) | · |  |
| `gemm/384x384x384/b1` | 1.79594 ± 0.00545271 (W 0.00294634) | 1.86523 ± 0.0104205 (W 0.0020212) | · |  |
| `gemm/512x512x512/b1` | 4.25023 ± 0.0207177 (W 0.00346816) | 4.39175 ± 0.0287076 (W 0.00482292) | · |  |
| `gemm/768x768x768/b1` | 14.5164 ± 0.057109 (W 0.0309823) | 14.7931 ± 0.0462363 (W 0.0237156) | · |  |

## gemm-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0218305 ± 0.000143755 (W 0.000378306) | 0.0240995 ± 0.000217655 (W 0.000441279) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.0834805 ± 0.000410951 (W 0.000805299) | 0.070607 ± 0.00024249 (W 0.00108366) | 🔴 faer-fork 1.18× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.15784 ± 0.000894255 (W 0.000452869) | 0.15557 ± 0.00148105 (W 0.00184202) | · |  |
| `gemm/384x384x384/b1` | 0.464382 ± 0.00236281 (W 0.00421099) | 0.496933 ± 0.00267067 (W 0.00334999) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.11697 ± 0.00353361 (W 0.00258305) | 1.15022 ± 0.00497395 (W 0.00497458) | · |  |
| `gemm/768x768x768/b1` | 3.66968 ± 0.0116957 (W 0.00523597) | 3.78603 ± 0.013821 (W 0.0146541) | · |  |

## gemm-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.02155 ± 0.000164163 (W 0.000386159) | 0.0154185 ± 0.00059308 (W 0.000415072) | 🔴 faer-fork 1.40× slower than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.048626 ± 0.000662328 (W 0.0008737) | 0.0401595 ± 0.000298636 (W 0.000651199) | 🔴 faer-fork 1.21× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.10761 ± 0.000656307 (W 0.000611535) | 0.0842025 ± 0.000529881 (W 0.000735738) | 🔴 faer-fork 1.28× slower than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.248362 ± 0.000951115 (W 0.00263383) | 0.255927 ± 0.00143322 (W 0.00188593) | · |  |
| `gemm/512x512x512/b1` | 0.598594 ± 0.00263189 (W 0.00272877) | 0.592706 ± 0.0161246 (W 0.00228332) | · | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 1.88894 ± 0.00504832 (W 0.00608645) | 1.91618 ± 0.0112552 (W 0.00410615) | · |  |

## gemm-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.266115 ± 0.00101274 (W 0.001731) | 0.286143 ± 0.00100414 (W 0.00222384) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.89982 ± 0.00261722 (W 0.00202492) | 0.946572 ± 0.00272774 (W 0.00120806) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 2.1304 ± 0.00731818 (W 0.0028552) | 2.24888 ± 0.0064884 (W 0.00311982) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 7.18179 ± 0.0161726 (W 0.00902272) | 7.42478 ± 0.0214574 (W 0.0109702) | · |  |
| `gemm/512x512x512/b1` | 17.0142 ± 0.069811 (W 0.0175328) | 17.6502 ± 0.0530118 (W 0.0280542) | · |  |
| `gemm/768x768x768/b1` | 57.8265 ± 0.207108 (W 0.0819144) | 59.2327 ± 0.178401 (W 0.0784058) | · |  |

## gemm-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.087217 ± 0.00033856 (W 0.000215559) | 0.0786715 ± 0.000721403 (W 0.000812435) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.233034 ± 0.000859688 (W 0.00158719) | 0.246293 ± 0.00124149 (W 0.00201035) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.572569 ± 0.00251018 (W 0.00231831) | 0.57736 ± 0.00315511 (W 0.00330403) | · |  |
| `gemm/384x384x384/b1` | 1.81144 ± 0.0063593 (W 0.00272578) | 1.88728 ± 0.00574718 (W 0.00177327) | · |  |
| `gemm/512x512x512/b1` | 4.35889 ± 0.0128167 (W 0.00523126) | 4.52853 ± 0.0280854 (W 0.0120712) | · |  |
| `gemm/768x768x768/b1` | 14.6901 ± 0.0341463 (W 0.0218547) | 15.1973 ± 0.0636021 (W 0.0231087) | · |  |

## gemm-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.055844 ± 0.000264258 (W 0.000422053) | 0.043446 ± 0.000293169 (W 0.000488839) | 🔴 faer-fork 1.29× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.158221 ± 0.00130749 (W 0.000609352) | 0.130413 ± 0.00119914 (W 0.00111125) | 🔴 faer-fork 1.21× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.310774 ± 0.000348128 (W 0.00184876) | 0.296909 ± 0.00208171 (W 0.00200906) | · |  |
| `gemm/384x384x384/b1` | 0.921313 ± 0.00114017 (W 0.00101943) | 0.951718 ± 0.00309 (W 0.0023896) | · |  |
| `gemm/512x512x512/b1` | 2.22325 ± 0.0125667 (W 0.00269599) | 2.28224 ± 0.0113506 (W 0.0061632) | · |  |
| `gemm/768x768x768/b1` | 7.51012 ± 0.0245727 (W 0.00855293) | 7.65348 ± 0.0290376 (W 0.0152675) | · |  |

## gemm-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.266329 ± 0.000820547 (W 0.00208649) | 0.286164 ± 0.00111017 (W 0.00231939) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.898807 ± 0.00296666 (W 0.00213201) | 0.947261 ± 0.00326378 (W 0.00154831) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 2.13612 ± 0.00668765 (W 0.00622277) | 2.24666 ± 0.00890213 (W 0.00511219) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 7.177 ± 0.0233336 (W 0.00605384) | 7.41365 ± 0.022149 (W 0.0101866) | · |  |
| `gemm/512x512x512/b1` | 17.0409 ± 0.0668638 (W 0.0160711) | 17.6657 ± 0.0761085 (W 0.0158528) | · |  |
| `gemm/768x768x768/b1` | 57.8012 ± 0.1693 (W 0.101286) | 59.2341 ± 0.202428 (W 0.0584643) | · |  |

## gemm-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0873175 ± 0.000289823 (W 0.000163445) | 0.078426 ± 0.000644074 (W 0.000718286) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.233651 ± 0.00100771 (W 0.00172948) | 0.245301 ± 0.00118942 (W 0.00186012) | · |  |
| `gemm/256x256x256/b1` | 0.574038 ± 0.00198384 (W 0.00219608) | 0.576591 ± 0.00179963 (W 0.00299036) | · |  |
| `gemm/384x384x384/b1` | 1.81205 ± 0.00553321 (W 0.00262759) | 1.89055 ± 0.00722948 (W 0.00380847) | · |  |
| `gemm/512x512x512/b1` | 4.3529 ± 0.0150077 (W 0.00455879) | 4.53793 ± 0.021762 (W 0.00888966) | · |  |
| `gemm/768x768x768/b1` | 14.6916 ± 0.0486604 (W 0.00854071) | 15.1643 ± 0.0296873 (W 0.0510343) | · |  |

## gemm-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0562345 ± 0.000491798 (W 0.000588598) | 0.0435165 ± 0.000274017 (W 0.000753732) | 🔴 faer-fork 1.29× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.158797 ± 0.000797292 (W 0.000442624) | 0.131585 ± 0.000858958 (W 0.00144687) | 🔴 faer-fork 1.21× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.310731 ± 0.000534022 (W 0.00170576) | 0.29832 ± 0.00239879 (W 0.00179835) | · |  |
| `gemm/384x384x384/b1` | 0.922289 ± 0.00222113 (W 0.00181333) | 0.955176 ± 0.00310022 (W 0.0031333) | · |  |
| `gemm/512x512x512/b1` | 2.22236 ± 0.00787218 (W 0.00400386) | 2.28149 ± 0.011743 (W 0.0032294) | · |  |
| `gemm/768x768x768/b1` | 7.51037 ± 0.0318766 (W 0.0117281) | 7.66688 ± 0.0286492 (W 0.0156012) | · |  |

## gemm-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.27729 ± 0.00631639 (W 0.006846) | 1.65546 ± 0.0118447 (W 0.00579554) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.07584 ± 0.0249994 (W 0.0052536) | 1.46889 ± 0.00959983 (W 0.0118687) | 🔵 faer-fork 1.37× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 1.08915 ± 0.00336156 (W 0.00315922) | 1.37134 ± 0.0163909 (W 0.0035685) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.422882 ± 0.000699645 (W 0.00406925) | 0.506323 ± 0.00374367 (W 0.00563376) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.30789 ± 0.00173807 (W 0.00236009) | 0.475907 ± 0.0017504 (W 0.00976121) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.276746 ± 0.00203791 (W 0.00160046) | 0.347771 ± 0.0011434 (W 0.00207515) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.419561 ± 0.00180118 (W 0.00295382) | 0.283033 ± 0.00259799 (W 0.00296103) | 🔴 faer-fork 1.48× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.186246 ± 0.00123702 (W 0.00171618) | 0.312356 ± 2.23724 (W 0.00193148) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.141724 ± 0.00250235 (W 0.00102839) | 0.18003 ± 1.43601 (W 0.000647398) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.03994 ± 0.00631469 (W 0.00252875) | 1.70349 ± 0.00973103 (W 0.00414992) | 🔵 faer-fork 1.64× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.09593 ± 0.00352627 (W 0.00277141) | 1.43653 ± 0.0056066 (W 0.0046724) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.12829 ± 0.00462423 (W 0.00413734) | 1.37506 ± 0.00758627 (W 0.00349419) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.323398 ± 0.00276624 (W 0.00351935) | 0.426214 ± 0.00412657 (W 0.00369237) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.298586 ± 0.00188255 (W 0.00229932) | 0.484358 ± 0.00338561 (W 0.00262666) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.296883 ± 0.00203974 (W 0.00265256) | 0.366188 ± 0.00144345 (W 0.00255033) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |

## gemm-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.194467 ± 0.00161594 (W 0.00387358) | 0.225214 ± 0.00172588 (W 0.00284603) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.180891 ± 0.00238071 (W 0.002798) | 0.32525 ± 0.00300915 (W 0.00215425) | 🔵 faer-fork 1.80× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.156115 ± 0.00032953 (W 0.00142652) | 0.197348 ± 0.00140698 (W 0.00183155) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.14349 ± 0.0115707 (W 0.00785395) | 5.98262 ± 0.0126686 (W 0.0155532) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 4.35896 ± 0.0496919 (W 0.0149669) | 5.78617 ± 0.0203352 (W 0.0206152) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 4.37601 ± 0.020558 (W 0.00722231) | 5.41685 ± 0.0252369 (W 0.0111667) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.65381 ± 0.0287415 (W 0.00625389) | 1.73508 ± 0.00588229 (W 0.00521444) | · |  |
| `gemm/4096x16x240/b1` | 1.15481 ± 0.00466036 (W 0.00510758) | 1.67678 ± 0.00958403 (W 0.010871) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.12717 ± 0.00383652 (W 0.00248796) | 1.38787 ± 0.00352932 (W 0.0036662) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.899938 ± 0.00762874 (W 0.0108189) | 0.934976 ± 0.00859258 (W 0.0081182) | · |  |
| `gemm/4096x16x240/b1` | 0.598105 ± 0.00299472 (W 0.00418808) | 1.00295 ± 0.675602 (W 0.0090966) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.574626 ± 0.00301113 (W 0.00274892) | 0.699564 ± 0.00187441 (W 0.00266597) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.06032 ± 0.0137325 (W 0.00907393) | 5.80239 ± 0.0387444 (W 0.0126716) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 4.36583 ± 0.00937357 (W 0.00847203) | 5.79563 ± 0.0125523 (W 0.0220169) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 4.43594 ± 0.013886 (W 0.0141775) | 5.54981 ± 0.0108584 (W 0.0107445) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.18727 ± 0.0042423 (W 0.00273606) | 1.50846 ± 0.00696711 (W 0.00531522) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.16031 ± 0.00458587 (W 0.00401972) | 1.73908 ± 0.00311009 (W 0.00370745) | 🔵 faer-fork 1.50× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.16594 ± 0.00452745 (W 0.00468677) | 1.44894 ± 0.00372696 (W 0.00319278) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.660165 ± 0.00304019 (W 0.00488795) | 0.824917 ± 0.00390913 (W 0.004413) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.590373 ± 0.00344367 (W 0.00435799) | 1.01732 ± 0.0523729 (W 0.0155909) | 🔵 faer-fork 1.72× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.579967 ± 0.00371223 (W 0.0040934) | 0.771853 ± 0.00160503 (W 0.00282461) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.545452 ± 0.0106725 (W 0.00313255) | 0.695352 ± 0.00190221 (W 0.00300481) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.540673 ± 0.00210639 (W 0.00239275) | 0.642392 ± 0.0119708 (W 0.0029346) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.541154 ± 0.00275278 (W 0.00291805) | 0.595029 ± 0.00653069 (W 0.00299708) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.409164 ± 0.00172125 (W 0.0023967) | 0.215782 ± 0.00161772 (W 0.00176655) | 🔴 faer-fork 1.90× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.143758 ± 0.000538722 (W 0.00138007) | 0.165248 ± 0.00113038 (W 0.00130917) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.13974 ± 0.000935234 (W 0.000436062) | 0.152274 ± 0.000985472 (W 0.00163353) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.407024 ± 0.00184445 (W 0.0021675) | 0.119618 ± 0.00119521 (W 0.00114807) | 🔴 faer-fork 3.40× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.077189 ± 0.000365656 (W 0.000721897) | 0.090178 ± 0.000831521 (W 0.00162582) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.0720395 ± 0.00042423 (W 0.000277703) | 0.086351 ± 0.00526259 (W 0.000269379) | 🔵 faer-fork 1.20× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.537797 ± 0.0033424 (W 0.00312623) | 0.641149 ± 0.00695622 (W 0.00302667) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.545068 ± 0.00306919 (W 0.00226412) | 0.679291 ± 0.0417901 (W 0.00266134) | 🔵 faer-fork 1.25× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.55571 ± 0.00239176 (W 0.00278409) | 0.694156 ± 0.0123076 (W 0.00202197) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |

## gemm-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.177697 ± 0.001322 (W 0.0014514) | 0.183511 ± 0.000523619 (W 0.0019936) | · |  |
| `gemm/4096x16x240/b1` | 0.145671 ± 0.00135833 (W 0.000624606) | 0.185967 ± 0.00677968 (W 0.0020646) | 🔵 faer-fork 1.28× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.156356 ± 0.000667116 (W 0.0026542) | 0.177922 ± 0.00214876 (W 0.00245986) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |

## gemm-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.111704 ± 0.000509724 (W 0.00120753) | 0.108528 ± 0.00040876 (W 0.000406765) | · |  |
| `gemm/4096x16x240/b1` | 0.081612 ± 0.000298677 (W 0.000353429) | 0.122899 ± 0.000543723 (W 0.00319329) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.0792075 ± 0.00021554 (W 0.000350929) | 0.103698 ± 0.0014996 (W 0.00247504) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.12343 ± 0.0281015 (W 0.00656323) | 2.46482 ± 0.0288461 (W 0.00832659) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 2.15912 ± 0.023402 (W 0.00589499) | 2.75049 ± 0.00991363 (W 0.00509935) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 2.16632 ± 0.00528173 (W 0.00412386) | 2.39933 ± 0.0321042 (W 0.00383022) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.81865 ± 0.00354552 (W 0.00328352) | 0.712255 ± 0.00320178 (W 0.00251236) | 🔴 faer-fork 1.15× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.582159 ± 0.00270634 (W 0.00355241) | 0.71922 ± 0.00176139 (W 0.00186551) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.553152 ± 0.00209989 (W 0.00236311) | 0.606092 ± 0.00288217 (W 0.00262726) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.813307 ± 0.00306859 (W 0.00256461) | 0.385659 ± 0.00416522 (W 0.00365139) | 🔴 faer-fork 2.11× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.300986 ± 0.000879527 (W 0.00201427) | 0.436063 ± 0.000580265 (W 0.00302046) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.278925 ± 0.00134727 (W 0.00198856) | 0.305138 ± 0.000534678 (W 0.00173097) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.03663 ± 0.0061479 (W 0.00509376) | 2.58967 ± 0.0152468 (W 0.00799298) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 2.17801 ± 0.00670348 (W 0.0058074) | 2.76175 ± 0.0127747 (W 0.0055103) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 2.19856 ± 0.00561729 (W 0.00708349) | 2.58024 ± 0.0212547 (W 0.00650915) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |

## gemm-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.594565 ± 0.0029832 (W 0.00287141) | 0.644375 ± 0.00675187 (W 0.00322969) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.571154 ± 0.00259783 (W 0.00311823) | 0.77499 ± 0.003158 (W 0.00315622) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.574342 ± 0.00106732 (W 0.00236547) | 0.647706 ± 0.00422161 (W 0.00244721) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |

## gemm-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.324634 ± 0.00240016 (W 0.00394173) | 0.34051 ± 0.00266714 (W 0.00265643) | · |  |
| `gemm/4096x16x240/b1` | 0.291022 ± 0.00118257 (W 0.00186908) | 0.441321 ± 0.0195161 (W 0.00563665) | 🔵 faer-fork 1.52× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.310964 ± 0.000695426 (W 0.00191708) | 0.338303 ± 0.0021068 (W 0.00229372) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.12844 ± 0.00646009 (W 0.00408811) | 1.37624 ± 0.00568494 (W 0.00376978) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.296238 ± 0.000515714 (W 0.0022737) | 0.366712 ± 5.98363 (W 0.0027231) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.155454 ± 0.000681219 (W 0.0018701) | 3.07229 ± 3.85937 (W 0.00174629) | 🔵 faer-fork 19.76× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.08446 ± 0.0092003 (W 0.00261992) | 1.37012 ± 0.0043358 (W 0.00426906) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.277227 ± 0.00187773 (W 0.00219885) | 0.349011 ± 0.00160999 (W 0.00211398) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.141188 ± 0.000536086 (W 0.00166648) | 0.17975 ± 1.13543 (W 0.00168301) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.44184 ± 0.0112979 (W 0.0150651) | 5.54023 ± 0.00940715 (W 0.0112907) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |

## gemm-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.16107 ± 0.0126243 (W 0.00325369) | 1.44518 ± 0.00218497 (W 0.004404) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.57673 ± 0.00288117 (W 0.00435948) | 0.772275 ± 0.00307324 (W 0.00256632) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.37617 ± 0.0196482 (W 0.0115729) | 5.41203 ± 0.0216813 (W 0.0115717) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.12944 ± 0.00155179 (W 0.0027577) | 1.38865 ± 0.00419523 (W 0.0030815) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.572495 ± 0.00212157 (W 0.00274287) | 0.741213 ± 2.1292 (W 0.00463355) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.554126 ± 0.00271591 (W 0.00247732) | 0.689482 ± 0.0142281 (W 0.00272242) | 🔵 faer-fork 1.24× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.154699 ± 0.00278356 (W 0.00215906) | 0.176459 ± 0.00143889 (W 0.00225835) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |

## gemm-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.0791525 ± 0.00060389 (W 0.000358004) | 0.103738 ± 0.00248014 (W 0.00270601) | 🔵 faer-fork 1.31× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.54162 ± 0.00266002 (W 0.0030327) | 0.594306 ± 0.00140945 (W 0.00324097) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.139495 ± 0.000619902 (W 0.00037497) | 0.15239 ± 0.000574001 (W 0.00158787) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.072134 ± 0.000578253 (W 0.000311001) | 0.0817725 ± 0.00105254 (W 0.00145889) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.19045 ± 0.00776543 (W 0.00559802) | 2.56466 ± 0.0264384 (W 0.00535102) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.573331 ± 0.00227132 (W 0.00265241) | 0.6466 ± 0.00387441 (W 0.00272008) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.310388 ± 0.00114345 (W 0.0019826) | 0.338187 ± 2.29468 (W 0.0018638) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.16342 ± 0.00741535 (W 0.00461162) | 2.40058 ± 0.00744847 (W 0.0058478) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.552815 ± 0.00260543 (W 0.00261187) | 0.609199 ± 0.00126169 (W 0.00201252) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.278675 ± 0.000741734 (W 0.00180751) | 0.305509 ± 0.00420236 (W 0.00193684) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## linalg-small, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003797 ± 0.000172564 (W 4.51062e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003171 ± 2.25093e-05 (W 4.4597e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0133695 ± 0.000114252 (W 0.000130278) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.011903 ± 4.99026e-05 (W 9.17895e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0302315 ± 0.000121769 (W 0.000135892) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.027857 ± 0.000177089 (W 9.32941e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.111553 ± 0.000573602 (W 0.000903721) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.111803 ± 0.000306556 (W 0.0018829) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.572481 ± 0.00224896 (W 0.00279589) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.575888 ± 0.00154625 (W 0.00248498) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.33834 ± 0.00474716 (W 0.00336383) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.36485 ± 0.00588578 (W 0.00442269) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01068 ± 0.000116612 (W 6.31413e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013866 ± 0.000265336 (W 0.000102028) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.049307 ± 0.000215296 (W 0.000169503) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0590595 ± 0.000178407 (W 0.000207453) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.137431 ± 0.00105052 (W 0.00177487) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.164292 ± 0.000992922 (W 0.00269076) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0085255 ± 7.96084e-05 (W 5.01695e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008451 ± 8.2995e-05 (W 4.40151e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.041302 ± 0.000234167 (W 0.000195677) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0409165 ± 0.000205359 (W 0.000106243) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.122724 ± 0.000534415 (W 0.000344088) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.121857 ± 0.00035838 (W 0.000895426) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006337 ± 8.72095e-05 (W 5.06869e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008466 ± 0.000146617 (W 0.000127661) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.023168 ± 0.000180315 (W 8.47743e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.02968 ± 0.000113896 (W 0.000232722) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0550175 ± 0.000357601 (W 0.000977364) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0762675 ± 0.000573831 (W 0.00255824) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0362825 ± 0.000149841 (W 0.000183349) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0325805 ± 0.000125746 (W 0.000234793) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.107555 ± 0.000716813 (W 0.0010838) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.117208 ± 0.000375565 (W 0.00099494) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.235208 ± 0.000951904 (W 0.00228545) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.297268 ± 0.00161541 (W 0.00352144) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.024521 ± 0.000115715 (W 0.000479146) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.069424 ± 0.000350603 (W 0.000332172) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.141178 ± 0.000767638 (W 0.00126384) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006697 ± 3.75002e-05 (W 2.83849e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009553 ± 3.01662e-05 (W 6.46826e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0217405 ± 8.4089e-05 (W 8.92747e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0307675 ± 0.00019395 (W 0.000815572) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0494525 ± 0.000296606 (W 0.000118786) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0773745 ± 0.00141728 (W 0.00262474) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.167117 ± 0.00225658 (W 0.00161738) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.168193 ± 0.00116762 (W 0.00155693) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.682767 ± 0.00491179 (W 0.0025908) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.680122 ± 0.0124327 (W 0.00211884) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.56253 ± 0.00392629 (W 0.00425358) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.57108 ± 0.00661419 (W 0.00436875) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058539 ± 0.000311256 (W 0.000134508) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0589245 ± 0.000366899 (W 0.000901092) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.208523 ± 0.00140405 (W 0.0019194) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.20926 ± 0.000547229 (W 0.00176078) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.479083 ± 0.00162804 (W 0.00273066) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.481698 ± 0.00286762 (W 0.00304544) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000741 ± 4.27395e-06 (W 2.74355e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001613 ± 1.83485e-05 (W 3.98913e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00278 ± 3.88008e-05 (W 3.32385e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003787 ± 0.000213579 (W 3.60182e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003161 ± 3.86868e-05 (W 4.24202e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013615 ± 0.000137361 (W 0.000111592) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0121775 ± 6.74609e-05 (W 8.9299e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0308625 ± 8.87709e-05 (W 0.000148936) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.028513 ± 0.00017625 (W 0.000121046) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0763775 ± 0.000420096 (W 0.00155682) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.076628 ± 0.00011234 (W 0.000221213) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.314165 ± 0.000717069 (W 0.00215789) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.327109 ± 0.00290564 (W 0.00209266) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.773818 ± 0.0031131 (W 0.00195044) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.77575 ± 0.00259243 (W 0.00147062) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011882 ± 0.000189273 (W 5.65143e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015284 ± 0.000123985 (W 9.79777e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0531095 ± 0.000372202 (W 0.000179486) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0615795 ± 0.000324142 (W 0.000174691) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.151062 ± 0.00151749 (W 0.00109735) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.176725 ± 0.00132057 (W 0.00217765) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006212 ± 9.27737e-05 (W 0.0001518) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005956 ± 0.000100231 (W 8.74158e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.019411 ± 0.000135855 (W 0.000230192) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.020934 ± 0.000361464 (W 0.000459979) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.04551 ± 0.000161137 (W 0.00029428) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0470475 ± 0.00038605 (W 0.000647058) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0065425 ± 6.88198e-05 (W 7.24185e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008906 ± 9.99913e-05 (W 0.000106605) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.028804 ± 0.000331892 (W 0.000453436) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0302515 ± 0.000366556 (W 0.000195088) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0633985 ± 0.000631507 (W 0.00104321) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.07266 ± 0.00196113 (W 0.00141132) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.0378105 ± 8.80583e-05 (W 0.000134239) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.034204 ± 0.000264948 (W 0.000222114) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.114924 ± 0.00121566 (W 0.00129526) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.111353 ± 0.000440334 (W 0.000356197) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.221457 ± 0.00186159 (W 0.00197545) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.253382 ± 0.00342972 (W 0.00283832) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0255075 ± 0.00011808 (W 0.000119118) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0761475 ± 0.00109319 (W 0.000239325) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.149975 ± 0.00283533 (W 0.000343296) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007419 ± 7.11102e-05 (W 5.5708e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010269 ± 8.56543e-05 (W 7.15336e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.025768 ± 0.000171867 (W 0.000194029) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0328365 ± 0.000154727 (W 0.000159598) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0548975 ± 0.000516601 (W 0.000638963) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0781655 ± 0.000496449 (W 0.00260731) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.170543 ± 0.0010562 (W 0.00161975) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.175758 ± 0.00354589 (W 0.00123766) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.561032 ± 0.0114551 (W 0.00221556) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.563114 ± 0.00516908 (W 0.00223387) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.13574 ± 0.00209848 (W 0.00335528) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.14398 ± 0.003416 (W 0.00266298) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059792 ± 0.000310938 (W 8.91573e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0604275 ± 0.000333564 (W 0.000148167) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.215983 ± 0.00122544 (W 0.00166331) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.215778 ± 0.000765844 (W 0.00166927) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.517825 ± 0.00327963 (W 0.0025077) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.519056 ± 0.0035621 (W 0.00255798) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000972 ± 1.49755e-05 (W 2.9265e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002174 ± 2.27193e-05 (W 4.06194e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003727 ± 4.81819e-05 (W 4.22299e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0038025 ± 3.36675e-05 (W 3.10182e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003246 ± 4.133e-05 (W 6.88452e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013515 ± 0.000323946 (W 0.000114312) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012042 ± 0.000251933 (W 7.91469e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.030572 ± 0.000525866 (W 0.000150235) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0285535 ± 0.00015889 (W 0.000123654) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0719695 ± 0.00104649 (W 0.000237459) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0726805 ± 0.000533995 (W 0.000211081) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.234312 ± 0.00141203 (W 0.00227346) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.246533 ± 0.00139696 (W 0.00255019) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.502322 ± 0.00146079 (W 0.00279133) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.504425 ± 0.00289793 (W 0.00302518) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011917 ± 9.14799e-05 (W 7.00555e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015153 ± 0.00021233 (W 6.39437e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.053294 ± 0.000264916 (W 0.000161893) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0615145 ± 0.000221626 (W 0.000182001) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.157007 ± 0.000982042 (W 0.00110587) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.183927 ± 0.00197562 (W 0.00179899) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0061665 ± 6.14676e-05 (W 0.000294977) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006201 ± 0.00011682 (W 0.000136019) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.016842 ± 0.000200077 (W 0.000528657) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.016696 ± 0.000251275 (W 0.000457529) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.032366 ± 0.000425283 (W 0.000535142) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.034349 ± 0.000469946 (W 0.000571479) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0065725 ± 5.7772e-05 (W 5.72214e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008782 ± 0.000141493 (W 9.70991e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.029545 ± 0.000428461 (W 0.000506416) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.030487 ± 0.000180012 (W 0.000275982) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.066754 ± 0.000582342 (W 0.00110777) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.077144 ± 0.00153917 (W 0.00219044) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.0376805 ± 0.000149543 (W 0.000129373) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0340385 ± 0.000257111 (W 0.00019154) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.109058 ± 0.000604148 (W 0.000583722) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.110837 ± 0.000743501 (W 0.000361811) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.209746 ± 0.00619023 (W 0.00210034) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.248739 ± 0.00170446 (W 0.00296249) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025308 ± 9.7729e-05 (W 0.000117109) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.077405 ± 0.000457848 (W 0.000198106) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.145451 ± 0.00219585 (W 0.000602021) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007439 ± 0.000101765 (W 7.03287e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010224 ± 5.09928e-05 (W 5.19224e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.026419 ± 0.000351609 (W 0.0005458) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0325955 ± 0.000170154 (W 0.000157627) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.057697 ± 0.000345122 (W 0.000515175) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.081492 ± 0.000395915 (W 0.00246554) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.169991 ± 0.00130164 (W 0.000201617) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.172161 ± 0.00131902 (W 0.000698628) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.556145 ± 0.00395009 (W 0.00258149) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.557213 ± 0.00177497 (W 0.00210633) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.14511 ± 0.00689258 (W 0.00335708) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.14852 ± 0.0226623 (W 0.00258188) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n32/b1` | 0.059436 ± 0.000371454 (W 0.000151966) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059962 ± 0.000304404 (W 0.000141818) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.216168 ± 0.000897259 (W 0.00176312) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.216799 ± 0.00065658 (W 0.00188883) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.534727 ± 0.00189886 (W 0.0021224) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.534867 ± 0.00190709 (W 0.00202449) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0009615 ± 1.49321e-05 (W 2.83705e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002159 ± 5.52871e-05 (W 3.35912e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003747 ± 5.11534e-05 (W 4.94334e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0025045 ± 9.19074e-05 (W 2.07657e-05) | 0.0029405 ± 3.84656e-05 (W 3.2779e-05) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0031155 ± 2.23644e-05 (W 6.24871e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0099535 ± 0.000218326 (W 5.77526e-05) | 0.0135055 ± 0.000149787 (W 0.00075922) | 🔵 faer-fork 1.36× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.011707 ± 8.80312e-05 (W 6.15553e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.023844 ± 0.000340988 (W 6.81513e-05) | 0.0324755 ± 0.000133799 (W 0.000118228) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.027201 ± 0.000192046 (W 8.62914e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.111332 ± 0.000519841 (W 0.00166971) | 0.117719 ± 0.000844613 (W 0.0016159) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.111578 ± 0.000603479 (W 0.00191866) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.572041 ± 0.00185366 (W 0.00235404) | 0.422948 ± 0.00262726 (W 0.002137) | 🔴 faer-fork 1.35× slower than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.575348 ± 0.000308305 (W 0.00264569) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.33939 ± 0.00486039 (W 0.00339575) | 0.92865 ± 0.00545758 (W 0.00207532) | 🔴 faer-fork 1.44× slower than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.36351 ± 0.0027016 (W 0.00472037) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01062 ± 8.64099e-05 (W 6.61173e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0120925 ± 0.00025722 (W 9.60317e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0495475 ± 0.000373803 (W 0.00106787) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.052563 ± 0.000443799 (W 0.000104477) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.136635 ± 0.000421199 (W 0.00182694) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.149283 ± 0.00147753 (W 0.00228366) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008546 ± 7.40822e-05 (W 7.77472e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0084505 ± 4.45305e-05 (W 5.34904e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.041207 ± 0.000225647 (W 0.000121116) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.040761 ± 0.000146652 (W 0.00131067) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.125008 ± 0.00217682 (W 0.000646621) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.123996 ± 0.000684123 (W 0.000224683) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0051145 ± 3.86816e-05 (W 8.90114e-05) | 0.007404 ± 7.41458e-05 (W 0.000105408) | 🔵 faer-fork 1.45× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006542 ± 9.60555e-05 (W 0.000105338) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020493 ± 0.000319169 (W 0.000199759) | 0.030111 ± 0.000148009 (W 0.000168407) | 🔵 faer-fork 1.47× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.023659 ± 8.37655e-05 (W 0.000245081) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.048956 ± 0.000209157 (W 0.000438358) | 0.06993 ± 0.000211138 (W 0.000237773) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0633285 ± 0.0014799 (W 0.00235855) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.028062 ± 0.000281267 (W 0.000119012) | 0.032595 ± 0.000419873 (W 0.000114603) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.030457 ± 0.000203396 (W 0.000126785) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.105893 ± 0.000673479 (W 0.00097632) | 0.128279 ± 0.00113745 (W 0.00166886) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.110896 ± 0.000898654 (W 0.00186636) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.242757 ± 0.00210165 (W 0.00231786) | 0.318138 ± 0.00184597 (W 0.00256829) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.281254 ± 0.00104949 (W 0.0030484) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0169165 ± 0.000224906 (W 0.000123756) | 0.0151935 ± 0.000312494 (W 9.19869e-05) | 🔴 faer-fork 1.11× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.069349 ± 0.000648732 (W 0.000258097) | 0.0587145 ± 0.00048431 (W 0.000188379) | 🔴 faer-fork 1.18× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.14804 ± 0.00153196 (W 0.00187025) | 0.166209 ± 0.00179546 (W 0.00187698) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.005921 ± 4.53505e-05 (W 4.73885e-05) | 0.0076545 ± 5.46897e-05 (W 7.01016e-05) | 🔵 faer-fork 1.29× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0075235 ± 4.28505e-05 (W 6.83521e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020659 ± 9.32532e-05 (W 6.52319e-05) | 0.02947 ± 0.000284151 (W 0.00011543) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.0240345 ± 0.000219707 (W 0.000684325) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.049407 ± 0.000279528 (W 0.000253308) | 0.069565 ± 0.000365416 (W 0.000119097) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.064004 ± 0.00182902 (W 0.00271737) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.170652 ± 0.00100996 (W 0.00190325) | 0.171905 ± 0.000781334 (W 0.00179554) | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.170653 ± 0.00193182 (W 0.00157423) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.693217 ± 0.00701193 (W 0.00207238) | 0.626557 ± 0.00267747 (W 0.00251823) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.696557 ± 0.00357228 (W 0.00280199) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.58215 ± 0.00321462 (W 0.00398805) | 1.47356 ± 0.00979779 (W 0.00368125) | 🔴 faer-fork 1.07× slower than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.58746 ± 0.00405679 (W 0.00492751) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058564 ± 0.000306178 (W 0.000938457) | 0.0722 ± 0.000329215 (W 0.000136905) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.058824 ± 0.000220552 (W 0.000119539) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.210963 ± 0.000545661 (W 0.0019686) | 0.265395 ± 0.00123 (W 0.00235326) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.211615 ± 0.000682752 (W 0.00204118) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.490739 ± 0.0010699 (W 0.00316097) | 0.653979 ± 0.00275278 (W 0.00270644) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.491162 ± 0.00356218 (W 0.00240411) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000801 ± 5.30723e-06 (W 2.87962e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001808 ± 3.45876e-05 (W 7.41938e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0029855 ± 2.36136e-05 (W 5.24247e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0024795 ± 4.26181e-05 (W 2.05564e-05) | 0.002906 ± 2.66133e-05 (W 2.69059e-05) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003116 ± 1.7224e-05 (W 6.06094e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010079 ± 0.000255784 (W 5.35795e-05) | 0.013505 ± 0.00011871 (W 8.58064e-05) | 🔵 faer-fork 1.34× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0119675 ± 0.000216402 (W 9.15189e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.024245 ± 0.000398623 (W 7.99303e-05) | 0.032546 ± 0.000300861 (W 0.000181172) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.027902 ± 0.000414075 (W 9.67491e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0762015 ± 0.00131781 (W 0.000285577) | 0.101825 ± 0.00045243 (W 0.00080348) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.0771485 ± 0.00200339 (W 0.000186714) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.314681 ± 0.000490165 (W 0.00218941) | 0.348223 ± 0.00113933 (W 0.00221038) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.325747 ± 0.00139755 (W 0.00266964) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.772955 ± 0.00125951 (W 0.00167844) | 0.759094 ± 0.00134307 (W 0.00165063) | · |  |
| `eigh/n96/b1 [api-inclusive]` | 0.774879 ± 0.00334172 (W 0.00234475) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0118925 ± 0.000117475 (W 5.54226e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0132745 ± 8.5277e-05 (W 7.82926e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.052853 ± 0.000937937 (W 0.000152028) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0547925 ± 0.000573348 (W 0.000232415) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.150501 ± 0.00143046 (W 0.000874771) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.16029 ± 0.000851486 (W 0.00233463) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006537 ± 8.64006e-05 (W 0.000227303) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006422 ± 3.18852e-05 (W 0.000162958) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.019787 ± 0.000126017 (W 0.000195487) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.019466 ± 9.12886e-05 (W 0.000195505) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.044538 ± 0.000222748 (W 0.000208932) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.044518 ± 0.000248879 (W 0.000286241) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00529 ± 9.20145e-05 (W 7.97288e-05) | 0.007424 ± 6.31401e-05 (W 9.9237e-05) | 🔵 faer-fork 1.40× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0068725 ± 8.92803e-05 (W 8.76482e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020699 ± 0.000242589 (W 0.000289916) | 0.030051 ± 0.000270944 (W 0.000103421) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0239245 ± 0.000351018 (W 0.000274834) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.042534 ± 0.00030722 (W 0.00103572) | 0.06982 ± 0.000228097 (W 0.000191022) | 🔵 faer-fork 1.64× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0587095 ± 0.001274 (W 0.00215747) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.029675 ± 0.000247007 (W 0.000139439) | 0.032375 ± 0.000615969 (W 0.000160505) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.0323855 ± 0.000321277 (W 0.00017828) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.097687 ± 0.000801256 (W 0.00129422) | 0.124187 ± 0.000504768 (W 0.000982065) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.103659 ± 0.000799727 (W 0.0003287) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.188345 ± 0.000245651 (W 0.00184688) | 0.30294 ± 0.00199958 (W 0.00204201) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.233971 ± 0.00104298 (W 0.00249142) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.017588 ± 0.000160095 (W 8.47337e-05) | 0.015103 ± 0.000185531 (W 0.000109337) | 🔴 faer-fork 1.16× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.068803 ± 0.000159567 (W 0.000357329) | 0.058194 ± 0.000662655 (W 0.00176418) | 🔴 faer-fork 1.18× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.135191 ± 0.00108776 (W 0.000347431) | 0.166781 ± 0.00227479 (W 0.00178252) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.006507 ± 5.4191e-05 (W 4.14553e-05) | 0.0076845 ± 9.44472e-05 (W 4.7921e-05) | 🔵 faer-fork 1.18× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00818 ± 3.65919e-05 (W 5.93147e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0232635 ± 8.04672e-05 (W 0.000157905) | 0.0296505 ± 0.000373015 (W 9.02562e-05) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.025813 ± 0.000214284 (W 0.000211182) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.046672 ± 9.85224e-05 (W 0.00028764) | 0.070336 ± 0.000472146 (W 0.000256396) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.063238 ± 0.0013956 (W 0.00177972) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.173793 ± 0.00210141 (W 0.00187331) | 0.173678 ± 0.0012156 (W 0.00177457) | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.174526 ± 0.00114932 (W 0.00137668) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.562162 ± 0.00164406 (W 0.00242275) | 0.629337 ± 0.00219405 (W 0.00257106) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.564612 ± 0.0020275 (W 0.00225822) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.1561 ± 0.00706124 (W 0.00334759) | 1.45087 ± 0.00219732 (W 0.00478739) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15551 ± 0.00630469 (W 0.00260458) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059601 ± 0.0020542 (W 0.000111768) | 0.07221 ± 0.000422918 (W 0.000136975) | 🔵 faer-fork 1.21× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_values/n32/b1 [api-inclusive]` | 0.0598165 ± 0.000250041 (W 0.000198001) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.219203 ± 0.00089541 (W 0.00173207) | 0.26588 ± 0.000747121 (W 0.00211057) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.218738 ± 0.000777543 (W 0.00163943) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.527518 ± 0.00381184 (W 0.00283396) | 0.656794 ± 0.00116416 (W 0.002075) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.527898 ± 0.00174561 (W 0.0022687) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0010165 ± 2.13581e-05 (W 3.22056e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002344 ± 2.46069e-05 (W 3.44968e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0039575 ± 2.8757e-05 (W 4.38543e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00248 ± 0.000117984 (W 2.71952e-05) | 0.002906 ± 3.06964e-05 (W 6.94808e-05) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003201 ± 5.04645e-05 (W 4.68357e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010023 ± 5.74421e-05 (W 4.26475e-05) | 0.013495 ± 0.000120459 (W 0.000104289) | 🔵 faer-fork 1.35× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.011982 ± 5.0816e-05 (W 8.0728e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0239295 ± 0.000277933 (W 7.3702e-05) | 0.03241 ± 0.000142381 (W 0.00012476) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.0278115 ± 0.000446179 (W 0.000147425) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.071859 ± 0.000152796 (W 0.00070407) | 0.100527 ± 0.000279911 (W 0.00148017) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.0724095 ± 0.000198328 (W 0.000255917) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.235403 ± 0.00381911 (W 0.00186049) | 0.341105 ± 0.00153809 (W 0.00216596) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.245784 ± 0.00277686 (W 0.00271256) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.502676 ± 0.00254187 (W 0.00295954) | 0.730918 ± 0.00244473 (W 0.00293024) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.503734 ± 0.00175801 (W 0.002727) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011897 ± 0.000124671 (W 5.39684e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013235 ± 0.000231823 (W 0.000107056) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.053049 ± 0.000404258 (W 0.000100496) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0548775 ± 0.000308854 (W 0.000137984) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.156286 ± 0.00210445 (W 0.000516673) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.169611 ± 0.00163957 (W 0.00244206) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005951 ± 7.55425e-05 (W 0.000128713) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005936 ± 0.000110309 (W 0.000104876) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0145875 ± 0.000149163 (W 0.000221203) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.015053 ± 8.62964e-05 (W 0.000245167) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0304715 ± 0.000265971 (W 0.000371284) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0308825 ± 0.000297394 (W 0.000154862) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005259 ± 7.27892e-05 (W 5.59017e-05) | 0.007429 ± 0.000160475 (W 0.000117899) | 🔵 faer-fork 1.41× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006743 ± 4.36654e-05 (W 0.000100711) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02142 ± 0.000277605 (W 0.000212086) | 0.0300155 ± 0.000110343 (W 0.000138452) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024245 ± 0.000185387 (W 0.000232656) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.043306 ± 0.00021611 (W 0.0012387) | 0.0698005 ± 0.000355793 (W 0.0018309) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0607635 ± 0.000544176 (W 0.00231338) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.029515 ± 0.000445812 (W 0.000127377) | 0.03248 ± 0.000349321 (W 0.000100035) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.03205 ± 0.000325726 (W 0.000173029) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.096179 ± 0.00046282 (W 0.000430711) | 0.12609 ± 0.000548947 (W 0.000708051) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.104901 ± 0.00168585 (W 0.000484469) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.181734 ± 0.00155267 (W 0.0018285) | 0.306526 ± 0.001826 (W 0.00213695) | 🔵 faer-fork 1.69× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.230465 ± 0.00132087 (W 0.00286127) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.017638 ± 0.000166354 (W 0.000100764) | 0.015124 ± 0.000250259 (W 9.01301e-05) | 🔴 faer-fork 1.17× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0691135 ± 0.000682969 (W 0.000402747) | 0.0582335 ± 0.000482606 (W 0.000229893) | 🔴 faer-fork 1.19× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.133393 ± 0.00226864 (W 0.000433659) | 0.166566 ± 0.00178871 (W 0.00154295) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.0065225 ± 2.57345e-05 (W 3.44215e-05) | 0.007524 ± 3.50238e-05 (W 5.69885e-05) | 🔵 faer-fork 1.15× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0081255 ± 7.47808e-05 (W 4.92834e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.022702 ± 0.000199241 (W 0.000110909) | 0.0302665 ± 0.000174075 (W 0.000129563) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.025633 ± 0.000170577 (W 0.000178887) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.046732 ± 0.000185193 (W 0.000272053) | 0.070126 ± 0.000394038 (W 0.00160984) | 🔵 faer-fork 1.50× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.0666895 ± 0.00142926 (W 0.00235055) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.172776 ± 0.00880248 (W 0.000878666) | 0.174034 ± 0.000643337 (W 0.00169393) | · | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.174526 ± 0.00109291 (W 0.000852468) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.556453 ± 0.00136873 (W 0.00224071) | 0.631382 ± 0.00241795 (W 0.0021407) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.561251 ± 0.00298362 (W 0.00326593) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.15823 ± 0.0062888 (W 0.00228023) | 1.4544 ± 0.00575426 (W 0.00421416) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.16045 ± 0.00342148 (W 0.00273809) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059281 ± 0.000261835 (W 0.000113691) | 0.0718235 ± 0.000572513 (W 0.000950957) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.059902 ± 0.000340505 (W 0.000131876) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.218858 ± 0.000924547 (W 0.00170395) | 0.266181 ± 0.00178415 (W 0.00241128) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.219054 ± 0.00119447 (W 0.0018086) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.544574 ± 0.00235787 (W 0.00203918) | 0.65516 ± 0.00232689 (W 0.00208353) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.546968 ± 0.00245135 (W 0.00188755) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000987 ± 2.36643e-05 (W 3.67401e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002339 ± 2.81241e-05 (W 3.37382e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003958 ± 4.80236e-05 (W 5.00899e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.006096 ± 8.35001e-05 (W 4.36298e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005535 ± 4.74089e-05 (W 6.10307e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.032946 ± 0.000223291 (W 0.00011921) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.031845 ± 0.00016525 (W 0.000126925) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.089247 ± 0.000198488 (W 0.000343871) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0873075 ± 0.000460832 (W 0.000288899) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.141493 ± 0.000669231 (W 0.00172895) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.142741 ± 0.00228669 (W 0.00150829) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.74096 ± 0.00374173 (W 0.00331903) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.746026 ± 0.00227768 (W 0.00232146) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.80517 ± 0.00709343 (W 0.00484749) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.81054 ± 0.00447569 (W 0.00359395) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.016836 ± 0.000119338 (W 9.53769e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.020468 ± 0.000142339 (W 0.00011235) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.095939 ± 0.000424166 (W 0.00036142) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.108372 ± 0.000479391 (W 0.00222912) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.286718 ± 0.00182829 (W 0.00232279) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.350053 ± 0.00136549 (W 0.00267272) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028478 ± 0.000170294 (W 0.000875842) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028483 ± 7.60701e-05 (W 0.000128826) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.161681 ± 0.000883156 (W 0.00141811) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.159918 ± 0.000583472 (W 0.000283992) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.473587 ± 0.00327948 (W 0.00170226) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.47171 ± 0.00359104 (W 0.000921424) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0125235 ± 8.2026e-05 (W 5.54543e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0147975 ± 0.000235913 (W 5.00274e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.054206 ± 0.000114915 (W 0.000850257) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.062446 ± 0.000385747 (W 0.000397951) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.140542 ± 0.00029887 (W 0.00169643) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.206164 ± 0.00172051 (W 0.00254587) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0731665 ± 0.000580518 (W 0.000449822) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.068272 ± 0.00060757 (W 0.000316765) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.293452 ± 0.00116975 (W 0.00253409) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.333473 ± 0.00288519 (W 0.00305046) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.712033 ± 0.00385352 (W 0.00332484) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.851487 ± 0.0036054 (W 0.00319479) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.041117 ± 0.000288729 (W 0.000202114) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.146348 ± 0.00060674 (W 0.00170146) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.34611 ± 0.0024125 (W 0.00204889) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012744 ± 4.44091e-05 (W 4.20853e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01622 ± 7.68906e-05 (W 0.000128011) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.051927 ± 0.000291278 (W 0.000887491) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.064971 ± 0.000236144 (W 0.00041863) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.136799 ± 0.000700096 (W 0.00185312) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.213913 ± 0.00248581 (W 0.00286043) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.219505 ± 0.000825213 (W 0.00235916) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.220236 ± 0.00171022 (W 0.00139794) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.01965 ± 0.0109946 (W 0.00213115) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.02197 ± 0.00579965 (W 0.00221822) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.47613 ± 0.00920126 (W 0.00582983) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.48595 ± 0.0114629 (W 0.00683665) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0733365 ± 0.000229216 (W 0.000177326) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0737025 ± 0.000440436 (W 9.37441e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.31142 ± 0.000864792 (W 0.00234278) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.312989 ± 0.000847877 (W 0.00206128) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.812998 ± 0.00297855 (W 0.0029249) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.809642 ± 0.00123724 (W 0.00296683) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001167 ± 1.36626e-05 (W 3.49999e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00289 ± 1.20111e-05 (W 4.34948e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0053 ± 2.00067e-05 (W 4.54819e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0060765 ± 7.70688e-05 (W 3.21627e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0055855 ± 5.52718e-05 (W 5.16791e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0307575 ± 0.000120816 (W 0.000160668) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.030722 ± 8.13644e-05 (W 0.000101923) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0825035 ± 0.000464357 (W 0.000281979) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.083215 ± 0.000586479 (W 0.000272597) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.102431 ± 0.000265119 (W 0.000355185) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.103693 ± 0.000570109 (W 0.000380042) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.432871 ± 0.00153253 (W 0.00210283) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.434762 ± 0.00132546 (W 0.00244297) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.92586 ± 0.00309325 (W 0.00169476) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.932168 ± 0.00237292 (W 0.00265088) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0182335 ± 0.000157916 (W 0.000110937) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0216205 ± 0.000173287 (W 7.68534e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.0987645 ± 0.000271926 (W 0.000661997) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.108247 ± 0.000726995 (W 0.000575109) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.23081 ± 0.00139449 (W 0.00197168) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.304297 ± 0.00296962 (W 0.00262616) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0143265 ± 9.0897e-05 (W 0.000209859) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0142815 ± 5.29931e-05 (W 0.000148215) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0619105 ± 0.000442774 (W 0.000449668) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.062271 ± 0.000220881 (W 0.000332369) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.154494 ± 0.000299754 (W 0.00101483) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.154529 ± 0.000184696 (W 0.00184161) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012648 ± 1.50422e-05 (W 0.000105494) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015163 ± 0.000119843 (W 0.000100333) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.056696 ± 0.000712793 (W 0.00110981) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.061575 ± 0.00128425 (W 0.00116877) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.138418 ± 0.00168697 (W 0.0014088) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.191191 ± 0.00163215 (W 0.00220903) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0714185 ± 0.000337258 (W 0.000682135) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.063568 ± 0.00025024 (W 0.000313301) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.192303 ± 0.000566234 (W 0.00180524) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.221778 ± 0.000956777 (W 0.00249213) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.417041 ± 0.00246846 (W 0.00249308) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.515917 ± 0.00402953 (W 0.00436842) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.040866 ± 0.000327582 (W 0.000380738) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.12592 ± 0.000454159 (W 0.000386427) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.267744 ± 0.000839598 (W 0.00250948) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0134755 ± 0.000151266 (W 7.84971e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.017006 ± 0.000111068 (W 0.0001294) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.055228 ± 0.000672606 (W 0.000415482) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0646155 ± 0.000951378 (W 0.00122685) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.12085 ± 0.000793689 (W 0.00132953) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.191055 ± 0.00235911 (W 0.00315345) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.213713 ± 0.000532155 (W 0.00166786) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.216564 ± 0.00795352 (W 0.00187768) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.742398 ± 0.00238596 (W 0.00241565) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.747131 ± 0.00331166 (W 0.00216604) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.55495 ± 0.00354215 (W 0.00338163) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.56475 ± 0.00183681 (W 0.00334919) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0748045 ± 0.000523018 (W 0.000190839) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0749145 ± 0.000424991 (W 0.00010311) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.318267 ± 0.00193514 (W 0.00141491) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.318268 ± 0.00112576 (W 0.00178043) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.725297 ± 0.00196856 (W 0.0025495) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.726153 ± 0.00486582 (W 0.00267294) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001432 ± 2.13448e-05 (W 3.3543e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0034715 ± 3.7276e-05 (W 3.73867e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0061715 ± 4.04496e-05 (W 4.04583e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0061915 ± 5.03614e-05 (W 4.16779e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005631 ± 5.60916e-05 (W 5.43636e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.030893 ± 0.00017994 (W 0.000133175) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.030607 ± 0.000249916 (W 0.00015247) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0835455 ± 0.000343483 (W 0.000343722) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.083651 ± 0.000378636 (W 0.000281123) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.097948 ± 0.000290375 (W 0.000246685) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0994405 ± 0.000924117 (W 0.000404314) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.328803 ± 0.00541081 (W 0.00650381) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.331228 ± 0.00179547 (W 0.00651039) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.684562 ± 0.0046325 (W 0.00657675) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.688382 ± 0.00539411 (W 0.0125791) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018129 ± 0.000129446 (W 8.8652e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.021545 ± 0.000105972 (W 0.000163654) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.102376 ± 0.00101305 (W 0.00106623) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.113086 ± 0.000567622 (W 0.00048105) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.234627 ± 0.0005907 (W 0.00162044) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.308755 ± 0.00183788 (W 0.00274557) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0110105 ± 5.8739e-05 (W 0.000257522) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.010495 ± 7.57777e-05 (W 0.000104613) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.03777 ± 0.000223357 (W 0.000480555) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.038101 ± 0.000159797 (W 0.000367871) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.087678 ± 0.000556716 (W 0.000946262) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0883745 ± 0.000235871 (W 0.000659739) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012754 ± 7.08284e-05 (W 8.53323e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015093 ± 0.000105262 (W 0.000106542) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0626315 ± 0.000345063 (W 0.00052866) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.062376 ± 0.00016577 (W 0.000292144) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.145466 ± 0.0013998 (W 0.00109503) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.190651 ± 0.00282456 (W 0.00171986) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0723695 ± 0.000397497 (W 0.0013727) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.062797 ± 0.000496649 (W 0.000437147) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.175222 ± 0.000638747 (W 0.00178848) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.211154 ± 0.00154986 (W 0.0022559) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.349406 ± 0.000846039 (W 0.00324681) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.472075 ± 0.00386055 (W 0.00425423) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.039945 ± 0.000171021 (W 0.000176884) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.122223 ± 0.00108659 (W 0.000629682) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.243409 ± 0.00111361 (W 0.00208642) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013515 ± 7.16705e-05 (W 7.04063e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016937 ± 0.000198155 (W 9.98262e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.057347 ± 0.000251812 (W 0.000381277) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.066945 ± 0.000194014 (W 0.000344512) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.119303 ± 0.000676754 (W 0.000834368) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.19348 ± 0.00321684 (W 0.00388195) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.214474 ± 0.000477077 (W 0.00172315) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.217541 ± 0.00759908 (W 0.00238649) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.732125 ± 0.00341999 (W 0.00198345) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.736339 ± 0.00348499 (W 0.00194015) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.49875 ± 0.00961842 (W 0.00461013) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.50324 ± 0.00673278 (W 0.00337088) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0742635 ± 0.000365771 (W 0.000148812) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0747645 ± 0.000458059 (W 0.0001412) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.327941 ± 0.000727558 (W 0.00183345) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.32782 ± 0.000627207 (W 0.00155295) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.745304 ± 0.00298192 (W 0.00304639) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.752467 ± 0.00375006 (W 0.00288102) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001433 ± 1.77811e-05 (W 4.68733e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0034615 ± 1.92916e-05 (W 4.05628e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0061715 ± 6.4526e-05 (W 4.26195e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0045135 ± 8.84475e-05 (W 3.24256e-05) | 0.004879 ± 2.85774e-05 (W 4.73482e-05) | 🔵 faer-fork 1.08× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005465 ± 7.41395e-05 (W 6.55067e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.028127 ± 0.000199059 (W 4.26785e-05) | 0.0279925 ± 0.000196717 (W 7.10734e-05) | · |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.0314285 ± 0.000198958 (W 0.000110438) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.079353 ± 0.000433781 (W 0.00100167) | 0.0786015 ± 0.000460484 (W 0.000302901) | · |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.086371 ± 0.000783654 (W 0.00104782) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.141578 ± 0.00159265 (W 0.00181078) | 0.149344 ± 0.00077139 (W 0.00177961) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.143216 ± 0.00256683 (W 0.00154104) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.739812 ± 0.00226026 (W 0.00226022) | 0.645187 ± 0.00399807 (W 0.00306126) | 🔴 faer-fork 1.15× slower than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.74415 ± 0.00363281 (W 0.00258264) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.80238 ± 0.00340359 (W 0.00440806) | 1.61001 ± 0.00539577 (W 0.00270219) | 🔴 faer-fork 1.12× slower than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.81485 ± 0.00636566 (W 0.00442156) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0169715 ± 0.000157324 (W 7.41163e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01879 ± 0.00115384 (W 0.000125184) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0957075 ± 0.000660571 (W 0.000415951) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.10162 ± 0.00481617 (W 0.00126249) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.286875 ± 0.00106908 (W 0.0023006) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.337595 ± 0.00148523 (W 0.00300099) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028533 ± 0.00012801 (W 0.000115834) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028418 ± 0.000115484 (W 9.55887e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.159849 ± 0.000662708 (W 0.00124506) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.158811 ± 0.0005232 (W 0.00192886) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.466433 ± 0.0033573 (W 0.00176551) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.467911 ± 0.00315161 (W 0.00137324) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011352 ± 0.000118657 (W 9.25208e-05) | 0.0113965 ± 5.90785e-05 (W 9.67911e-05) | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013235 ± 4.23352e-05 (W 9.51165e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0501035 ± 0.00040918 (W 0.000290748) | 0.0551475 ± 0.000332229 (W 0.000147514) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.055513 ± 0.000368777 (W 0.000183726) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.134271 ± 0.000470329 (W 0.00168268) | 0.153391 ± 0.000393848 (W 0.00155227) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.192715 ± 0.0012493 (W 0.00227605) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.064009 ± 0.000366208 (W 0.000284618) | 0.0601875 ± 0.000324229 (W 0.000130047) | 🔴 faer-fork 1.06× slower than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.066995 ± 0.00026277 (W 0.000424264) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.297987 ± 0.00201006 (W 0.00180552) | 0.309812 ± 0.00236467 (W 0.00211472) | · |  |
| `qr/n64/b1 [api-inclusive]` | 0.328647 ± 0.00272528 (W 0.00333374) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.73306 ± 0.002343 (W 0.00228729) | 0.823306 ± 0.00316717 (W 0.0034098) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.837786 ± 0.00407365 (W 0.00454961) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.032045 ± 0.000227078 (W 0.000252917) | 0.028117 ± 0.000392293 (W 6.34914e-05) | 🔴 faer-fork 1.14× slower than `lapack-openblas` |  |
| `qr_factor/n64/b1` | 0.154138 ± 0.0007844 (W 0.0016805) | 0.126675 ± 0.000845953 (W 0.00168987) | 🔴 faer-fork 1.22× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.372123 ± 0.0013992 (W 0.0025726) | 0.347393 ± 0.00221248 (W 0.00230222) | 🔴 faer-fork 1.07× slower than `lapack-openblas` |  |
| `solve/n32/b1` | 0.012343 ± 0.000203943 (W 0.00011184) | 0.012659 ± 5.87551e-05 (W 6.26192e-05) | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014397 ± 0.000169734 (W 0.000108993) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0532395 ± 0.000520916 (W 0.000190334) | 0.0589145 ± 0.0002962 (W 0.00074352) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.057627 ± 0.00055662 (W 0.000347126) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.139209 ± 0.00127022 (W 0.0018084) | 0.16093 ± 0.000905476 (W 0.00186226) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.19885 ± 0.00230416 (W 0.003318) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.217516 ± 0.000887368 (W 0.00209075) | 0.222535 ± 0.0013246 (W 0.00176333) | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.218387 ± 0.0012675 (W 0.00151848) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.00604 ± 0.00461722 (W 0.000858075) | 0.934652 ± 0.005466 (W 0.0020949) | 🔴 faer-fork 1.08× slower than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 1.0082 ± 0.0031747 (W 0.00243628) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.46204 ± 0.0103589 (W 0.0043568) | 2.49847 ± 0.0070747 (W 0.00453839) | · |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.47457 ± 0.00873341 (W 0.00707948) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.074018 ± 0.000394888 (W 0.00010666) | 0.101489 ± 0.000863862 (W 0.000827741) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.074559 ± 0.000391467 (W 0.00147464) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.309395 ± 0.00108465 (W 0.00242934) | 0.398994 ± 0.00168971 (W 0.00207295) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.310042 ± 0.00206963 (W 0.00190299) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.806366 ± 0.00307548 (W 0.0018059) | 1.0085 ± 0.00910551 (W 0.00261941) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.811573 ± 0.00493808 (W 0.0025987) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001167 ± 1.18954e-05 (W 3.36263e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002896 ± 3.94331e-05 (W 3.85335e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0051945 ± 5.35192e-05 (W 5.46677e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004403 ± 9.64788e-05 (W 3.4479e-05) | 0.004854 ± 2.87083e-05 (W 5.32338e-05) | 🔵 faer-fork 1.10× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0055205 ± 9.59493e-05 (W 6.33695e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.026214 ± 0.000123993 (W 0.000143322) | 0.0320995 ± 0.00038535 (W 0.000852504) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.0304465 ± 7.11871e-05 (W 0.000172395) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0730705 ± 0.000565876 (W 0.000264403) | 0.073567 ± 0.00100699 (W 0.00108912) | · |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.0826235 ± 0.000344975 (W 0.000254264) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.102686 ± 0.000876785 (W 0.000315046) | 0.136253 ± 0.000413875 (W 0.00140005) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.103122 ± 0.00043434 (W 0.000320892) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.433543 ± 0.0022204 (W 0.00193679) | 0.56859 ± 0.00183863 (W 0.00239574) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.435723 ± 0.00253218 (W 0.00188843) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.926938 ± 0.00238244 (W 0.00316237) | 1.59723 ± 0.00548462 (W 0.00237929) | 🔵 faer-fork 1.72× faster than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.930773 ± 0.00378818 (W 0.00239847) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018068 ± 0.00018946 (W 9.60488e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019747 ± 0.000264502 (W 0.000100579) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.09668 ± 0.000520192 (W 0.000354638) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.101473 ± 0.000835931 (W 0.000602612) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.229768 ± 0.00167651 (W 0.00254122) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.289809 ± 0.00265636 (W 0.00193673) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0148575 ± 0.000136639 (W 0.000164989) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.014933 ± 0.000106143 (W 0.000184572) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0614545 ± 0.000138464 (W 0.000284848) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.061454 ± 0.000337302 (W 0.000364345) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.151297 ± 0.000505923 (W 0.00136648) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.151732 ± 0.000593595 (W 0.00176156) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011476 ± 0.00012706 (W 9.26745e-05) | 0.0114315 ± 6.72032e-05 (W 0.000136721) | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013606 ± 0.000212368 (W 0.00011181) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0480545 ± 0.000886658 (W 0.00134841) | 0.054997 ± 0.000197708 (W 0.000138635) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0564845 ± 0.00127597 (W 0.00115746) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.111678 ± 0.00107012 (W 0.00158379) | 0.153861 ± 0.000652551 (W 0.00156842) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.175356 ± 0.0020648 (W 0.0018958) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.057928 ± 0.000540103 (W 0.000385412) | 0.0599015 ± 0.000687887 (W 0.000316265) | · |  |
| `qr/n32/b1 [api-inclusive]` | 0.061349 ± 0.000264455 (W 0.000298161) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.177986 ± 0.000280315 (W 0.00158866) | 0.261297 ± 0.00144527 (W 0.00238446) | 🔵 faer-fork 1.47× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.216423 ± 0.000938496 (W 0.0027848) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.386159 ± 0.00187374 (W 0.0025907) | 0.708016 ± 0.00181998 (W 0.00258601) | 🔵 faer-fork 1.83× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.503908 ± 0.00239771 (W 0.00340375) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0305565 ± 0.000237881 (W 0.000195618) | 0.027637 ± 0.000332199 (W 8.03574e-05) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `qr_factor/n64/b1` | 0.116622 ± 0.000611218 (W 0.000301816) | 0.12634 ± 0.000533503 (W 0.00212872) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.247567 ± 0.000717463 (W 0.00181258) | 0.448466 ± 0.00253708 (W 0.0019444) | 🔵 faer-fork 1.81× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.012829 ± 0.000115181 (W 8.94013e-05) | 0.0126185 ± 6.87238e-05 (W 8.72007e-05) | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0150535 ± 0.000169066 (W 0.000159884) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.054071 ± 0.00094466 (W 0.00118975) | 0.0594755 ± 0.000341944 (W 0.000233609) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.058369 ± 0.00107752 (W 0.00105674) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.115015 ± 0.000795202 (W 0.000730935) | 0.159612 ± 0.000867201 (W 0.0018776) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.176989 ± 0.00186491 (W 0.00262436) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.211093 ± 0.000858655 (W 0.00162884) | 0.223451 ± 0.00117706 (W 0.00240239) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.21196 ± 0.0045672 (W 0.00185612) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.729928 ± 0.00186363 (W 0.00208202) | 0.934064 ± 0.00179761 (W 0.00140047) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.734049 ± 0.00166875 (W 0.00199294) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.5596 ± 0.00353743 (W 0.00404781) | 2.43541 ± 0.00478506 (W 0.00339281) | 🔵 faer-fork 1.56× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.57 ± 0.00732427 (W 0.00331727) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0752505 ± 0.000404046 (W 9.69957e-05) | 0.100803 ± 0.000453484 (W 0.000224448) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.0758415 ± 0.000366868 (W 7.05377e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.315061 ± 0.00154917 (W 0.0019473) | 0.401367 ± 0.00413745 (W 0.00284964) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.31687 ± 0.00147976 (W 0.00138142) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.721185 ± 0.00142249 (W 0.00253694) | 1.26674 ± 0.00282274 (W 0.00364212) | 🔵 faer-fork 1.76× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.725437 ± 0.00367628 (W 0.0031254) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0014525 ± 1.85005e-05 (W 3.72251e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0035265 ± 3.45683e-05 (W 4.15167e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0060865 ± 3.72044e-05 (W 3.71994e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004348 ± 7.7133e-05 (W 3.50274e-05) | 0.004899 ± 2.33809e-05 (W 5.72406e-05) | 🔵 faer-fork 1.13× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00557 ± 9.85097e-05 (W 6.26983e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.025878 ± 9.13756e-05 (W 0.000140563) | 0.033808 ± 0.000563274 (W 0.000792499) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.0303365 ± 0.000132575 (W 0.000129104) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.074002 ± 0.000249566 (W 0.000213551) | 0.079563 ± 0.000350372 (W 0.00104405) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.082769 ± 0.000439343 (W 0.000176701) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.098113 ± 0.000838632 (W 0.00027161) | 0.135292 ± 0.000501729 (W 0.000425326) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.0985535 ± 0.000343611 (W 0.000310679) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.327536 ± 0.00576507 (W 0.00588527) | 0.556853 ± 0.00257897 (W 0.00212745) | 🔵 faer-fork 1.70× faster than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.327104 ± 0.00731608 (W 0.00823763) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 0.678832 ± 0.00962916 (W 0.0150486) | 1.62085 ± 0.00571192 (W 0.00524782) | 🔵 faer-fork 2.39× faster than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.687391 ± 0.00297853 (W 0.00876532) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0179785 ± 0.000309714 (W 9.11692e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019692 ± 0.000281257 (W 9.21819e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.101695 ± 0.00104697 (W 0.000648575) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.105612 ± 0.000768759 (W 0.000509754) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.232538 ± 0.00200331 (W 0.00168771) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.29532 ± 0.00408996 (W 0.00422194) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.010991 ± 8.29307e-05 (W 0.000132748) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.011035 ± 0.000132389 (W 0.00012527) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.038161 ± 0.000186666 (W 0.000362383) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.038617 ± 0.000204613 (W 0.000604569) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.085975 ± 0.000452897 (W 0.00138409) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.087013 ± 0.000565705 (W 0.0010907) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011552 ± 0.000159374 (W 0.000100683) | 0.011306 ± 8.11583e-05 (W 9.76256e-05) | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013525 ± 5.42181e-05 (W 0.000131954) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0491815 ± 0.000342439 (W 0.00041929) | 0.054837 ± 0.000307279 (W 0.000107814) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.056325 ± 0.000423563 (W 0.000394616) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.108747 ± 0.000371339 (W 0.000733689) | 0.153296 ± 0.000579952 (W 0.00213729) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.173994 ± 0.00156478 (W 0.00167812) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.057527 ± 0.000290605 (W 0.000484553) | 0.060392 ± 0.000493337 (W 0.000951655) | · |  |
| `qr/n32/b1 [api-inclusive]` | 0.0607735 ± 0.00032693 (W 0.000415827) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.164361 ± 0.000644684 (W 0.00231175) | 0.259048 ± 0.00252555 (W 0.00248174) | 🔵 faer-fork 1.58× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.206124 ± 0.00858678 (W 0.00327427) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.325743 ± 0.00224254 (W 0.00238785) | 0.724099 ± 0.00353316 (W 0.00278446) | 🔵 faer-fork 2.22× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.456587 ± 0.00466771 (W 0.00448717) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.030632 ± 0.000191664 (W 0.00014356) | 0.028162 ± 0.000249473 (W 7.9559e-05) | 🔴 faer-fork 1.09× slower than `lapack-openblas` |  |
| `qr_factor/n64/b1` | 0.113527 ± 0.000842352 (W 0.000448949) | 0.125639 ± 0.000843926 (W 0.00175675) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.230079 ± 0.00121053 (W 0.00174015) | 0.48025 ± 0.00324168 (W 0.00159656) | 🔵 faer-fork 2.09× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.012679 ± 8.95299e-05 (W 8.43782e-05) | 0.012674 ± 6.66653e-05 (W 5.00086e-05) | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014958 ± 5.30299e-05 (W 8.90861e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.053104 ± 0.000161208 (W 0.000326543) | 0.0592955 ± 0.000193968 (W 0.000119686) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.060628 ± 0.000105604 (W 0.000267906) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.110275 ± 0.000299824 (W 0.000514207) | 0.160274 ± 0.000569646 (W 0.00201493) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.175427 ± 0.00177377 (W 0.0027617) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.213298 ± 0.00723284 (W 0.0017396) | 0.224042 ± 0.000625127 (W 0.00237728) | · | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.212751 ± 0.000474528 (W 0.00169906) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.718809 ± 0.00185312 (W 0.0023255) | 0.932226 ± 0.00359688 (W 0.00134764) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.7272 ± 0.00654484 (W 0.00251808) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.49482 ± 0.00332476 (W 0.00489744) | 2.50326 ± 0.0115382 (W 0.00364545) | 🔵 faer-fork 1.67× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.50416 ± 0.0043991 (W 0.00429323) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0753655 ± 0.000234698 (W 9.44075e-05) | 0.100352 ± 0.000843585 (W 0.000972966) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.075626 ± 0.000282462 (W 0.000124388) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.325396 ± 0.00138145 (W 0.00188216) | 0.39935 ± 0.00192856 (W 0.00246058) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.325551 ± 0.00106878 (W 0.00117492) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.74261 ± 0.00379649 (W 0.00301536) | 1.34801 ± 0.00572097 (W 0.00313271) | 🔵 faer-fork 1.82× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.751422 ± 0.00448593 (W 0.0029593) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001458 ± 1.28128e-05 (W 5.19503e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003702 ± 2.65832e-05 (W 3.54231e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006082 ± 6.25484e-05 (W 6.46415e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003226 ± 5.00017e-05 (W 3.36912e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00254 ± 3.26599e-05 (W 5.19645e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0135205 ± 0.00078257 (W 2.29736e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.009964 ± 4.44552e-05 (W 6.2469e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0260735 ± 0.000118449 (W 0.000146715) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0211345 ± 5.55722e-05 (W 8.04535e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.086235 ± 0.000364851 (W 0.000933141) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.086907 ± 0.000452609 (W 0.000243343) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.322116 ± 0.00215267 (W 0.00195309) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.324926 ± 0.00366156 (W 0.0024568) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.948798 ± 0.00501295 (W 0.00313331) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.955972 ± 0.00308442 (W 0.0021606) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.010054 ± 0.000103035 (W 3.95024e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0127635 ± 0.000134767 (W 6.48905e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0403255 ± 0.000407985 (W 8.64788e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.047574 ± 0.000204197 (W 0.000209984) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.09938 ± 0.000968773 (W 0.000656955) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.114468 ± 0.000720585 (W 0.0010675) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006437 ± 5.16398e-05 (W 7.64446e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006452 ± 4.49073e-05 (W 4.00289e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.026194 ± 0.000120899 (W 0.000136126) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.026058 ± 0.000153623 (W 7.02148e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.077048 ± 0.000408399 (W 0.000218357) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.077114 ± 0.000227828 (W 0.000236917) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0063115 ± 0.000123083 (W 5.34667e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00794 ± 6.67553e-05 (W 0.000105037) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0201575 ± 0.000167094 (W 7.13061e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0253925 ± 0.000436949 (W 0.000281616) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.044002 ± 0.000234222 (W 0.000224242) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.054923 ± 0.000287145 (W 0.000353685) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.030622 ± 0.000110219 (W 0.000112977) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.027717 ± 7.86649e-05 (W 0.000117671) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.090464 ± 0.000540027 (W 0.00108953) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.102511 ± 0.000481841 (W 0.000708938) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.192584 ± 0.000575533 (W 0.00179337) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.239537 ± 0.00160622 (W 0.00216718) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.021074 ± 0.000182651 (W 9.7481e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0591805 ± 0.000405819 (W 0.000227222) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.113041 ± 0.000809073 (W 0.00242149) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007088 ± 7.11805e-05 (W 4.85447e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0093175 ± 8.49251e-05 (W 4.31611e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019577 ± 0.000132099 (W 9.84076e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.026495 ± 0.000182571 (W 0.000146572) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.04035 ± 0.000158283 (W 0.000207535) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0538455 ± 0.000289624 (W 0.000192557) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.12584 ± 0.00125148 (W 0.00123904) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.127127 ± 0.00160055 (W 0.00157745) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.530332 ± 0.011792 (W 0.0026046) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.529928 ± 0.00444439 (W 0.00274561) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.20707 ± 0.00550098 (W 0.00351232) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.20906 ± 0.00479442 (W 0.00289036) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0474885 ± 0.00025034 (W 0.000102912) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.047779 ± 0.000329485 (W 0.000922006) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.168284 ± 0.00078651 (W 0.00195667) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.167893 ± 0.000672533 (W 0.00170808) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.366679 ± 0.00283271 (W 0.00252865) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.380319 ± 0.00140426 (W 0.00270006) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000731 ± 1.00083e-05 (W 3.1733e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001543 ± 2.2286e-05 (W 3.03e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0026095 ± 3.86975e-05 (W 3.17515e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003231 ± 0.00012695 (W 2.7767e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002635 ± 5.39444e-05 (W 3.84538e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013535 ± 0.000713144 (W 5.15487e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0100585 ± 7.49551e-05 (W 6.94958e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.026224 ± 0.00108934 (W 9.72202e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.0213045 ± 0.00113176 (W 0.000142967) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.0647055 ± 0.000165687 (W 0.000220823) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.064886 ± 0.000609632 (W 0.000265221) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.252401 ± 0.000665079 (W 0.00191452) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.252345 ± 0.000958675 (W 0.00202371) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.546076 ± 0.00207037 (W 0.00192101) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.559273 ± 0.00252626 (W 0.00421814) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011116 ± 0.000298617 (W 4.8439e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013986 ± 0.000418758 (W 0.000161309) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.043441 ± 0.000443258 (W 0.000174692) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.049968 ± 0.000697955 (W 0.000134893) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.106273 ± 0.000981496 (W 0.000433327) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.119242 ± 0.0013405 (W 0.000598097) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005395 ± 0.000133816 (W 0.000159256) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005425 ± 0.000205368 (W 0.000133325) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.013891 ± 0.000228424 (W 0.000427321) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0141115 ± 0.000174029 (W 0.00033352) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0333525 ± 0.000187358 (W 0.000561428) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0345295 ± 0.000469149 (W 0.000685026) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0063315 ± 5.11485e-05 (W 4.91927e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0081955 ± 9.64794e-05 (W 8.58115e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020258 ± 0.000120936 (W 8.19905e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0253675 ± 0.000460774 (W 0.000190929) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.053189 ± 0.00125715 (W 0.00170899) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.0527835 ± 0.000590134 (W 0.000430817) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.03228 ± 0.000352798 (W 0.000105516) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0295005 ± 0.000282001 (W 0.000152798) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.104425 ± 0.000360946 (W 0.000762121) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.101184 ± 0.000724594 (W 0.000195885) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.203359 ± 0.00177777 (W 0.00233321) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.196642 ± 0.000851169 (W 0.00169671) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0221965 ± 0.000322892 (W 9.2391e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.069845 ± 0.000562423 (W 0.000230239) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.1341 ± 0.000752165 (W 0.000326181) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007634 ± 8.29227e-05 (W 7.92367e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009833 ± 4.56965e-05 (W 0.000101389) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020924 ± 0.000147662 (W 0.000118276) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0276565 ± 0.000156649 (W 0.000231921) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.046096 ± 0.000394832 (W 0.000839424) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0552675 ± 0.000303898 (W 0.00045291) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.128545 ± 0.00201652 (W 0.000212519) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.129466 ± 0.00346463 (W 0.000253984) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.440391 ± 0.00302428 (W 0.00225171) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.454221 ± 0.00170209 (W 0.00384936) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.883595 ± 0.00505319 (W 0.00310036) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.883922 ± 0.00287968 (W 0.00278893) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0485005 ± 0.000273847 (W 0.000124243) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.048696 ± 0.000281795 (W 0.000140163) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.170407 ± 0.000815083 (W 0.00179887) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.169586 ± 0.00103873 (W 0.000770192) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.379257 ± 0.00191335 (W 0.00204117) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.376482 ± 0.00224559 (W 0.00210387) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000966 ± 1.47975e-05 (W 3.87703e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002049 ± 1.97019e-05 (W 3.27871e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0033715 ± 5.94945e-05 (W 3.94358e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003231 ± 0.000111848 (W 2.51896e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002629 ± 3.73667e-05 (W 4.99669e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0135905 ± 7.6548e-05 (W 5.08686e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0099735 ± 5.41864e-05 (W 8.59516e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.026279 ± 7.69946e-05 (W 0.000136889) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0214145 ± 0.000100048 (W 9.76738e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0613145 ± 0.000236145 (W 0.000219439) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.061795 ± 0.000406129 (W 0.000195116) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.199072 ± 0.000579086 (W 0.00198023) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.199236 ± 0.00118384 (W 0.00170492) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.412027 ± 0.0024188 (W 0.00317665) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.427438 ± 0.00337498 (W 0.00398452) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011126 ± 0.000145797 (W 5.67331e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013996 ± 9.07607e-05 (W 0.000103276) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0435215 ± 0.000574289 (W 0.000100763) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0499685 ± 0.000541434 (W 0.000230404) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.106498 ± 0.000966359 (W 0.000118908) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.119048 ± 0.00110257 (W 0.000581015) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006281 ± 0.000222684 (W 0.000470137) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006262 ± 0.00026328 (W 0.000452408) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.012418 ± 0.000127338 (W 0.000387102) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0126085 ± 0.000292502 (W 0.0002715) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0251465 ± 0.000286117 (W 0.00052765) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0254625 ± 0.000371868 (W 0.00067359) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006312 ± 4.70815e-05 (W 7.23277e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00815 ± 0.000160582 (W 9.15494e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020353 ± 9.0037e-05 (W 0.000107746) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0253225 ± 0.000196562 (W 0.000190189) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0609585 ± 0.000791819 (W 0.00219241) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.053214 ± 0.000326524 (W 0.000647609) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.032065 ± 0.000259311 (W 0.000120431) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0294295 ± 0.000190556 (W 0.000173382) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.1022 ± 0.000443912 (W 0.000594358) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.09911 ± 0.000933209 (W 0.000250384) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.194703 ± 0.00189715 (W 0.00314635) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.193606 ± 0.00156993 (W 0.00208799) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0220615 ± 0.000109239 (W 0.000106358) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.070431 ± 0.000340738 (W 0.000244848) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.127849 ± 0.00109657 (W 0.000326253) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0075645 ± 6.48866e-05 (W 3.98671e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0097635 ± 5.58951e-05 (W 4.33022e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020774 ± 0.000156466 (W 7.85361e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.027617 ± 0.000357833 (W 0.000156157) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.05368 ± 0.000973787 (W 0.00126076) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.055764 ± 0.000226753 (W 0.000750006) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.128169 ± 0.00100634 (W 0.00021219) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.128514 ± 0.00478808 (W 0.000111653) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.439464 ± 0.0102022 (W 0.00224869) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.456015 ± 0.00409883 (W 0.00405141) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.879199 ± 0.00503585 (W 0.00416843) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.879192 ± 0.0155166 (W 0.00393318) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0484155 ± 0.000181105 (W 0.000123865) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0487255 ± 0.000313944 (W 0.000197014) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.170017 ± 0.00122847 (W 0.00116842) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.170453 ± 0.000730988 (W 0.000418183) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.381536 ± 0.00100818 (W 0.0025646) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.380329 ± 0.00181425 (W 0.00276969) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000982 ± 1.36626e-05 (W 2.93938e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0020385 ± 2.92147e-05 (W 4.12551e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0033765 ± 1.70763e-05 (W 4.14931e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001904 ± 4.3545e-05 (W 1.40876e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00255 ± 1.50167e-05 (W 5.61817e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008366 ± 6.81635e-05 (W 2.14294e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.009954 ± 3.45683e-05 (W 5.02353e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.017693 ± 5.01996e-05 (W 7.05971e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.020834 ± 0.000152487 (W 6.963e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0862805 ± 0.000917655 (W 0.00110967) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.086581 ± 0.00353791 (W 0.000342291) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.322507 ± 0.0137371 (W 0.00222229) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.323944 ± 0.000878462 (W 0.00229759) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.946257 ± 0.00469728 (W 0.00287712) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.952723 ± 0.0040551 (W 0.00296909) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0100035 ± 9.89897e-05 (W 4.60507e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.011301 ± 7.7133e-05 (W 9.81497e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0400445 ± 0.00029363 (W 7.50134e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0423435 ± 0.00032635 (W 0.000755161) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.099556 ± 0.00113772 (W 0.000300769) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.102857 ± 0.0168653 (W 0.00157858) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0068125 ± 4.70489e-05 (W 9.61766e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006813 ± 3.88158e-05 (W 4.23652e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.027642 ± 8.2843e-05 (W 8.49927e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.027221 ± 0.000169197 (W 9.89568e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0829645 ± 0.00046384 (W 0.00106453) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.082919 ± 0.000385829 (W 0.000979928) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005024 ± 4.93575e-05 (W 5.45684e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006317 ± 9.03327e-05 (W 6.14704e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017528 ± 0.000127787 (W 0.000235999) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.020263 ± 0.000256664 (W 0.000284839) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.038552 ± 0.000399016 (W 0.000167) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.042278 ± 0.000181315 (W 0.000911026) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.024035 ± 0.000127333 (W 8.90873e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.026319 ± 0.000173348 (W 8.21567e-05) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0904135 ± 0.000318671 (W 0.000223673) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.0950275 ± 0.000476502 (W 0.000255853) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.216855 ± 0.0010222 (W 0.00207235) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.223056 ± 0.00155851 (W 0.0020082) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0147475 ± 8.21235e-05 (W 0.000102536) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.060122 ± 0.000385493 (W 0.000156058) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.127804 ± 0.000755582 (W 0.00081753) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.005886 ± 0.000100085 (W 7.30174e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007599 ± 0.000118322 (W 9.26652e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.018234 ± 0.000145419 (W 6.01221e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.020719 ± 0.000173706 (W 6.29633e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.03773 ± 0.000164476 (W 0.000129344) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0413775 ± 0.000328315 (W 0.00019324) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.135262 ± 0.000831294 (W 0.00167916) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.136224 ± 0.00127185 (W 0.000852984) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.532606 ± 0.00274254 (W 0.00235133) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.530945 ± 0.00268222 (W 0.00261384) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.2072 ± 0.00494589 (W 0.00272721) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.20916 ± 0.00471501 (W 0.00303184) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0485255 ± 0.000209562 (W 0.00077846) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0486855 ± 0.000164372 (W 0.000108093) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.172491 ± 0.000847263 (W 0.0020909) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.173408 ± 0.000853151 (W 0.00251623) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.371401 ± 0.00192185 (W 0.00292923) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.384892 ± 0.00186235 (W 0.0025797) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0007915 ± 1.14134e-05 (W 2.56016e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001748 ± 1.00747e-05 (W 3.78976e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00272 ± 2.33809e-05 (W 3.64319e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0019235 ± 9.98779e-05 (W 1.61993e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002645 ± 3.32666e-05 (W 4.99812e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0083805 ± 0.000941987 (W 3.01276e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.009993 ± 5.42122e-05 (W 6.69063e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.017843 ± 0.00113105 (W 6.64268e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.020959 ± 0.000909228 (W 8.683e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.064134 ± 0.000173903 (W 0.000188505) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.064816 ± 0.000276271 (W 0.000255904) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.253197 ± 0.00136182 (W 0.00163005) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.253222 ± 0.0009164 (W 0.00158289) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.543948 ± 0.00264591 (W 0.0026042) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.557308 ± 0.00257376 (W 0.00503234) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011001 ± 0.000110431 (W 3.74016e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0124685 ± 9.64552e-05 (W 8.7142e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.043476 ± 0.000648644 (W 0.000119017) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.045054 ± 0.000259279 (W 0.000189433) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.105737 ± 0.000366312 (W 0.000447885) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.1075 ± 0.000627819 (W 0.000699651) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005285 ± 0.000188457 (W 0.000225405) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00535 ± 0.000159863 (W 0.000164899) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014306 ± 0.000107961 (W 0.000305448) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0141615 ± 0.000143987 (W 0.00022783) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.030287 ± 0.000172222 (W 0.000415187) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.030507 ± 0.000351927 (W 0.000303767) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0049895 ± 3.60592e-05 (W 4.48611e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006447 ± 3.31889e-05 (W 6.88152e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017718 ± 0.000187195 (W 0.000180296) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0202275 ± 0.000244639 (W 0.000218028) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.036643 ± 0.000493736 (W 0.000371962) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0404955 ± 0.00021551 (W 0.000562587) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.025518 ± 0.000217878 (W 8.13033e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.027957 ± 0.000188163 (W 0.000144354) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.089221 ± 0.000617953 (W 0.000333641) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.09364 ± 0.000663064 (W 0.000309072) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.17165 ± 0.000810838 (W 0.00185463) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.179515 ± 0.00118314 (W 0.0017569) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.015729 ± 0.000115627 (W 0.000143108) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.062712 ± 0.000459287 (W 0.000217992) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.118692 ± 0.000587386 (W 0.000438004) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006432 ± 1.87083e-05 (W 5.78776e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008 ± 8.59651e-05 (W 6.77145e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019171 ± 0.000160437 (W 0.000107118) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0219705 ± 0.00017989 (W 0.000130592) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.038888 ± 0.000131978 (W 0.000339404) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0428595 ± 0.000276492 (W 0.000400602) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.138153 ± 0.0100856 (W 0.000976642) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.137746 ± 0.000577466 (W 0.000181539) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.437665 ± 0.00195215 (W 0.00276324) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.454295 ± 0.00719554 (W 0.0049847) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.889396 ± 0.00192087 (W 0.00192799) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.887221 ± 0.0022671 (W 0.00202751) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0495175 ± 0.000283119 (W 9.54437e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0495125 ± 0.000394984 (W 0.00011968) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.174465 ± 0.00101595 (W 0.000939895) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.174465 ± 0.000774356 (W 0.00090759) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.385242 ± 0.00140977 (W 0.00181377) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.381189 ± 0.00252128 (W 0.00196755) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001002 ± 1.64165e-05 (W 3.33313e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0022345 ± 2.88837e-05 (W 4.75546e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003546 ± 4.47545e-05 (W 4.01994e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001894 ± 0.000129732 (W 2.58275e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002655 ± 1.38624e-05 (W 3.69474e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008356 ± 0.00072582 (W 3.08963e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.009969 ± 7.32132e-05 (W 7.24719e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.016731 ± 0.00113505 (W 5.45015e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.021064 ± 0.000822922 (W 0.000105797) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.06132 ± 0.000346928 (W 0.000202952) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0617505 ± 0.000428515 (W 0.000235804) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.199191 ± 0.00137793 (W 0.00191465) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.199667 ± 0.00224505 (W 0.00166991) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.412424 ± 0.00206219 (W 0.00303525) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.428634 ± 0.00300798 (W 0.00431726) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0111555 ± 0.000107537 (W 4.48864e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012483 ± 4.93964e-05 (W 0.000110335) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.043822 ± 0.000619074 (W 0.00019635) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0447835 ± 0.000280668 (W 0.000240178) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.106293 ± 0.000368742 (W 0.000478837) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.10739 ± 0.000632321 (W 0.000492368) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005295 ± 0.000144181 (W 0.000135232) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0051745 ± 3.70805e-05 (W 0.000122523) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.011522 ± 0.000375197 (W 0.000359678) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0121775 ± 0.000220125 (W 0.000354326) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.021996 ± 0.000402463 (W 0.000407166) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.022462 ± 0.000375708 (W 0.000431422) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005009 ± 9.18704e-05 (W 7.00404e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006427 ± 9.64711e-05 (W 6.75974e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017688 ± 0.000175529 (W 0.000163674) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.020248 ± 0.00018557 (W 0.000224568) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.03736 ± 0.000197858 (W 0.000304468) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0413575 ± 0.000305635 (W 0.000395395) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.025432 ± 0.000250451 (W 0.000105152) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.027907 ± 0.000318504 (W 0.000146738) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.088505 ± 0.000715372 (W 0.000248171) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.093479 ± 0.000489318 (W 0.00026609) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.167573 ± 0.00148354 (W 0.00177059) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.175822 ± 0.0017573 (W 0.00132233) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.015764 ± 0.000183838 (W 0.000129269) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0628975 ± 0.000726595 (W 0.000222377) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.11815 ± 0.0011099 (W 0.000301585) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006377 ± 4.09878e-05 (W 8.94871e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007985 ± 5.89251e-05 (W 5.24683e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019211 ± 0.000238228 (W 9.73716e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0218155 ± 0.000204397 (W 0.000175295) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0386325 ± 0.00033491 (W 0.000370513) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.04273 ± 0.000393626 (W 0.000788385) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.137116 ± 0.00663669 (W 0.00017893) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.137496 ± 0.00718323 (W 0.000192012) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.433999 ± 0.000931079 (W 0.00223389) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.451357 ± 0.00381859 (W 0.0038474) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.886711 ± 0.00919573 (W 0.00312642) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.88527 ± 0.00205982 (W 0.00314037) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.049542 ± 0.000226301 (W 0.000141395) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.050008 ± 0.000379455 (W 0.000166622) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.173384 ± 0.0011074 (W 0.00178744) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.174545 ± 0.000472544 (W 0.00034047) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.387321 ± 0.000983238 (W 0.00223993) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.387136 ± 0.000906072 (W 0.00217734) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001007 ± 1.8619e-05 (W 3.15804e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002169 ± 3.0605e-05 (W 4.16608e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0035015 ± 3.43162e-05 (W 6.0632e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012724 ± 6.24233e-05 (W 6.10737e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0120775 ± 6.81784e-05 (W 7.79507e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.070672 ± 0.000240777 (W 0.000284754) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0669495 ± 0.000146603 (W 0.000222284) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.115416 ± 0.000217474 (W 0.00170628) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.110045 ± 0.000341899 (W 0.00167499) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.104529 ± 0.0014621 (W 0.000252607) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.105266 ± 0.000734217 (W 0.000181624) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.40779 ± 0.01703 (W 0.00251429) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.414477 ± 0.0030115 (W 0.00301899) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.1706 ± 0.000970912 (W 0.00302194) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.17396 ± 0.00182632 (W 0.00260592) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013761 ± 8.58846e-05 (W 5.84412e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.017202 ± 0.000100399 (W 0.000117335) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0638635 ± 0.000325463 (W 0.000171779) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0744535 ± 0.000586353 (W 0.000229371) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.177837 ± 0.00142899 (W 0.00219096) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.207141 ± 0.0011612 (W 0.00223452) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.018024 ± 0.000114869 (W 0.000104867) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0179985 ± 7.8326e-05 (W 7.40294e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0922015 ± 0.000410439 (W 0.00161142) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0914905 ± 0.000145786 (W 0.000165272) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.268666 ± 0.00057132 (W 0.00223671) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.267328 ± 0.000757391 (W 0.00219203) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.010179 ± 0.000268202 (W 8.26912e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012188 ± 0.000137114 (W 9.19371e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0388325 ± 0.000120815 (W 0.000174989) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0456 ± 0.000243196 (W 0.000234983) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0939695 ± 0.000488796 (W 0.00161844) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.117259 ± 0.00161098 (W 0.00334157) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0538555 ± 0.000257869 (W 0.000185509) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.050434 ± 0.000432888 (W 0.000269814) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.187765 ± 0.000988354 (W 0.0018619) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.195289 ± 0.000279908 (W 0.00182862) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.440329 ± 0.0026152 (W 0.00203953) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.498554 ± 0.00147797 (W 0.00300536) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0323855 ± 0.000170561 (W 0.000186663) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.1045 ± 0.000524012 (W 0.000264953) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.230355 ± 0.000770244 (W 0.00201597) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0109255 ± 6.96288e-05 (W 5.59182e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01333 ± 5.81745e-05 (W 4.74964e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0378455 ± 0.000108162 (W 0.000124988) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0464865 ± 0.000120796 (W 0.000125678) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0914055 ± 0.000504043 (W 0.000109023) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.117373 ± 0.00179363 (W 0.00242257) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.16123 ± 0.00111526 (W 0.00156142) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.162502 ± 0.00226199 (W 0.00228031) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.685377 ± 0.00818595 (W 0.00244171) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.688206 ± 0.00325823 (W 0.00272701) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.6438 ± 0.00532393 (W 0.00431493) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.65019 ± 0.0143965 (W 0.00412839) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.057402 ± 0.000268853 (W 0.000148034) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.057953 ± 0.000493301 (W 0.00159271) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.218848 ± 0.00141592 (W 0.00216861) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.219443 ± 0.000736239 (W 0.00203473) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.53886 ± 0.00223907 (W 0.0025036) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.539701 ± 0.00352375 (W 0.00294142) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001052 ± 4.08248e-06 (W 3.83021e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002455 ± 1.49755e-05 (W 6.49809e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004313 ± 3.4448e-05 (W 5.0016e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012679 ± 4.4364e-05 (W 5.11211e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012037 ± 3.2776e-05 (W 7.78562e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0701955 ± 0.000533826 (W 0.000229355) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066534 ± 0.000273174 (W 0.000166206) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.113045 ± 0.000276435 (W 0.000279048) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108597 ± 0.000545559 (W 0.000342598) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.085289 ± 0.000480175 (W 0.000260787) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.085509 ± 0.000155281 (W 0.000213909) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.323387 ± 0.000738993 (W 0.00189345) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.324364 ± 0.00107566 (W 0.00195231) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.69324 ± 0.00224747 (W 0.00237632) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.693486 ± 0.00131707 (W 0.00232383) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0149175 ± 0.000232824 (W 6.96287e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018289 ± 0.000164949 (W 8.60918e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.06707 ± 0.000520681 (W 0.000108499) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.076894 ± 0.000479614 (W 0.000239049) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.170523 ± 0.000938069 (W 0.00190455) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.200895 ± 0.00587584 (W 0.0025057) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.009588 ± 0.000130958 (W 0.000146911) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009503 ± 0.000165082 (W 0.00012936) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.036859 ± 9.29955e-05 (W 0.00037553) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.037144 ± 0.000106145 (W 0.000559676) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0880895 ± 9.82861e-05 (W 0.000606792) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.089056 ± 0.000365624 (W 0.000630683) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0101185 ± 0.000124071 (W 9.93003e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012238 ± 0.000221838 (W 0.000106751) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.039278 ± 0.000599689 (W 0.000591745) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.045354 ± 0.000358506 (W 0.000611036) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.097256 ± 0.000869733 (W 0.00116687) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.106318 ± 0.000217978 (W 0.00226878) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0545265 ± 0.000182162 (W 0.000180368) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.051656 ± 0.000368604 (W 0.00020784) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.144419 ± 0.000482578 (W 0.00151235) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.145556 ± 0.000559947 (W 0.000443125) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.288142 ± 0.0015661 (W 0.0023971) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.323247 ± 0.0020413 (W 0.00235787) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.032886 ± 0.000118136 (W 0.000118091) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0960995 ± 0.00017734 (W 0.000229431) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.193655 ± 0.000546809 (W 0.0016765) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011461 ± 4.32049e-05 (W 4.66195e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0139105 ± 7.51603e-05 (W 5.70059e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.041918 ± 0.000412525 (W 0.00033069) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0482545 ± 0.000180895 (W 0.000323278) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.090724 ± 0.000747802 (W 0.00114292) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.111633 ± 0.000633433 (W 0.00146073) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.164877 ± 0.000917927 (W 0.000574027) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.164096 ± 0.0017513 (W 0.000398289) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.535658 ± 0.00273643 (W 0.00236024) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.535734 ± 0.00171618 (W 0.00223109) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.11525 ± 0.00118518 (W 0.00231873) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.12101 ± 0.00157465 (W 0.00310992) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0590405 ± 0.00034731 (W 0.000132087) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059396 ± 0.000304701 (W 9.92488e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.224784 ± 0.00123037 (W 0.00146585) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.224303 ± 0.000991519 (W 0.00175096) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.526385 ± 0.00292211 (W 0.00240232) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.526801 ± 0.00303539 (W 0.00230027) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001313 ± 1.34722e-05 (W 3.82144e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003031 ± 0.000370853 (W 2.60655e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005125 ± 2.31574e-05 (W 3.66108e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012794 ± 5.14344e-05 (W 8.18893e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012132 ± 0.000106143 (W 6.73586e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.070086 ± 0.000178627 (W 0.000169572) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066499 ± 0.000180184 (W 0.000232325) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.112509 ± 0.000208713 (W 0.000278186) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.109234 ± 0.000383274 (W 0.000421864) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.082133 ± 0.000268143 (W 0.00019815) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0826395 ± 0.0003685 (W 0.000273485) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.260231 ± 0.00106187 (W 0.00160181) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.262115 ± 0.00101056 (W 0.00158542) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.546122 ± 0.00226548 (W 0.00347359) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.549604 ± 0.000970001 (W 0.00290092) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014893 ± 0.000146481 (W 6.47421e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018349 ± 8.68648e-05 (W 0.000146747) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0672305 ± 0.000610732 (W 0.000123181) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.076863 ± 0.000199427 (W 0.000200029) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.176945 ± 0.00156631 (W 0.00190923) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.208524 ± 0.00224392 (W 0.00375022) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008005 ± 0.000114134 (W 0.000236249) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00798 ± 0.000142974 (W 0.000177288) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0244105 ± 0.000270873 (W 0.000425318) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0247815 ± 0.000311917 (W 0.000333794) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0561095 ± 0.000417469 (W 0.000852123) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.056946 ± 0.000529434 (W 0.000630601) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0100035 ± 0.000105238 (W 8.44272e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0121575 ± 8.87784e-05 (W 0.000104658) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.045204 ± 0.000696138 (W 0.000836524) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.045575 ± 0.000501483 (W 0.000644735) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.101455 ± 0.00162612 (W 0.0019795) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.111633 ± 0.00127537 (W 0.00212398) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.054462 ± 0.000199119 (W 0.000151368) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.051626 ± 0.000163169 (W 0.000203945) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.134831 ± 0.000821611 (W 0.00159656) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.138708 ± 0.000855482 (W 0.000466443) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.257224 ± 0.000569557 (W 0.00202327) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.30048 ± 0.00230686 (W 0.00310404) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0328565 ± 0.000198039 (W 0.000146228) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.094476 ± 0.000575676 (W 0.00027269) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.178733 ± 0.000672882 (W 0.00146901) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0113915 ± 5.35304e-05 (W 6.76394e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013821 ± 7.91598e-05 (W 7.93399e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.044613 ± 0.000407991 (W 0.00055898) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0482695 ± 0.00010918 (W 0.000528837) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0938845 ± 0.000721554 (W 0.000874644) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.114659 ± 0.000613706 (W 0.00203295) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.164347 ± 0.000657473 (W 0.000249177) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.164897 ± 0.000617489 (W 0.00019065) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.522067 ± 0.00981625 (W 0.00228174) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.523015 ± 0.0024503 (W 0.00234215) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.08919 ± 0.00448195 (W 0.00366877) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.09299 ± 0.00238703 (W 0.00338049) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0593905 ± 0.00026668 (W 9.26804e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060768 ± 0.00031747 (W 0.00013098) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.225575 ± 0.00119603 (W 0.00275663) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.225911 ± 0.00109814 (W 0.00175489) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.538058 ± 0.00144356 (W 0.00228365) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.534687 ± 0.00179905 (W 0.00216609) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0013075 ± 2.55786e-05 (W 2.70974e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003011 ± 2.67638e-05 (W 3.31777e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0051395 ± 4.98558e-05 (W 3.42679e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011516 ± 4.9371e-05 (W 4.85774e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012057 ± 8.81151e-05 (W 7.5084e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.065477 ± 0.00031992 (W 0.000948575) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.06687 ± 0.000203336 (W 0.000246754) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.106483 ± 0.000198424 (W 0.000907254) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.109764 ± 0.00031061 (W 0.000976295) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.104691 ± 0.00020736 (W 0.00166497) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.105 ± 0.00092972 (W 0.000916439) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.407855 ± 0.00289018 (W 0.00246009) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.415154 ± 0.0108513 (W 0.00259973) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 1.16867 ± 0.00493083 (W 0.00373109) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.17459 ± 0.00485098 (W 0.00304938) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0137105 ± 7.83932e-05 (W 7.33147e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015379 ± 8.54985e-05 (W 0.000116404) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0639545 ± 0.00043071 (W 0.000816236) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0676465 ± 0.00427465 (W 0.000833766) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.177671 ± 0.000995128 (W 0.00209978) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.192088 ± 0.00211854 (W 0.00227336) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.018138 ± 0.0001167 (W 8.53864e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.018058 ± 0.000177528 (W 0.000100663) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.091165 ± 0.000231681 (W 0.000225669) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0907785 ± 0.000187726 (W 0.000260429) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.273324 ± 0.0005785 (W 0.00221226) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.272633 ± 0.000545577 (W 0.00233285) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008636 ± 0.000101931 (W 8.08459e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010314 ± 7.24911e-05 (W 7.77832e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0356265 ± 0.000308333 (W 0.000312902) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0387725 ± 0.000133497 (W 0.000132134) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0873885 ± 0.000573046 (W 0.000608261) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.101579 ± 0.0010358 (W 0.00236826) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.046201 ± 0.000274023 (W 0.000232947) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0487705 ± 0.000123974 (W 0.000107725) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.183202 ± 0.000737348 (W 0.00185094) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.188526 ± 0.000947168 (W 0.00168281) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.449296 ± 0.00323371 (W 0.00194596) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.483546 ± 0.000766009 (W 0.00449291) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0247415 ± 8.4579e-05 (W 0.000152398) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.103112 ± 0.000348341 (W 0.00180445) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.237022 ± 0.000757701 (W 0.00246387) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.009763 ± 3.14643e-05 (W 5.68273e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.011281 ± 6.74537e-05 (W 5.71489e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.036057 ± 0.000150433 (W 0.000109475) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0394485 ± 6.46704e-05 (W 8.76086e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.088981 ± 0.00117144 (W 0.000205686) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.102195 ± 0.00246984 (W 0.0026525) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.161901 ± 0.000956841 (W 0.00168115) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.163664 ± 0.0019514 (W 0.00163887) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.686425 ± 0.00429055 (W 0.00248433) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.686468 ± 0.0063458 (W 0.00327293) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.63847 ± 0.00198553 (W 0.00429051) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.64076 ± 0.00295062 (W 0.00331484) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058509 ± 8.49009e-05 (W 7.92636e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0588445 ± 0.00030229 (W 7.75297e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.218668 ± 0.00158982 (W 0.00202787) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.219419 ± 0.00118905 (W 0.00191457) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.545804 ± 0.00275104 (W 0.00303619) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.54498 ± 0.00235819 (W 0.00272576) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001117 ± 1.26491e-05 (W 3.30746e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00248 ± 1.75547e-05 (W 3.49874e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004013 ± 3.35798e-05 (W 5.00097e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011411 ± 4.92612e-05 (W 4.32648e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012022 ± 2.01072e-05 (W 7.97326e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0647705 ± 0.000291418 (W 0.00014023) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066549 ± 0.000298635 (W 0.000172525) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.103412 ± 0.000304228 (W 0.00027341) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.107821 ± 0.000562842 (W 0.000431424) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.085159 ± 0.000479818 (W 0.000279228) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.085705 ± 0.000462406 (W 0.000240399) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.323407 ± 0.000497841 (W 0.00162584) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.324504 ± 0.000685349 (W 0.00192354) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.692759 ± 0.00193501 (W 0.00240594) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.695696 ± 0.0119912 (W 0.00260617) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0149175 ± 0.0001395 (W 5.86791e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0165155 ± 0.000114858 (W 0.000155937) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.066865 ± 0.000485561 (W 0.000146926) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.069775 ± 0.000229461 (W 0.000230249) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.170202 ± 0.000842514 (W 0.000664999) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.186032 ± 0.00148376 (W 0.00290906) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.009838 ± 0.000122611 (W 0.000195359) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0098935 ± 0.000133408 (W 0.000118099) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.036418 ± 0.000177911 (W 0.000247341) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.036012 ± 0.000234139 (W 0.000300086) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0877185 ± 0.000319563 (W 0.000858645) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0877385 ± 0.000172832 (W 0.000866269) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008581 ± 5.06386e-05 (W 7.85588e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010364 ± 7.35772e-05 (W 8.9746e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0341585 ± 0.00060104 (W 0.000343909) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0388025 ± 0.000364698 (W 0.000461326) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0756865 ± 0.00155991 (W 0.00104021) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0908645 ± 0.000601187 (W 0.00125714) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0473185 ± 0.000276143 (W 0.00114303) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0496475 ± 0.000250097 (W 0.000179309) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.131474 ± 0.000471076 (W 0.000382463) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.138508 ± 0.000595541 (W 0.00044247) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.262468 ± 0.00146466 (W 0.00180992) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.307644 ± 0.00136868 (W 0.00316924) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025342 ± 0.000158231 (W 0.000139952) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0882045 ± 0.000289608 (W 0.0003531) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.175412 ± 0.000456503 (W 0.000410304) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.010134 ± 7.63948e-05 (W 5.58876e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.011837 ± 8.19034e-05 (W 6.6306e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0373595 ± 0.000445978 (W 0.000459289) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0411265 ± 0.00019196 (W 0.00029677) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.079348 ± 0.00116815 (W 0.000440564) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0971155 ± 0.000924346 (W 0.0024269) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.166074 ± 0.00594474 (W 0.00191433) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.164803 ± 0.000842501 (W 0.000203542) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.534867 ± 0.00138043 (W 0.00208018) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.53483 ± 0.00516629 (W 0.00262003) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.11665 ± 0.00236109 (W 0.00327987) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.12469 ± 0.00417037 (W 0.00350485) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.060042 ± 0.000407112 (W 0.000110503) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0600815 ± 0.000317602 (W 0.000105049) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.224348 ± 0.000786278 (W 0.00162355) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.223943 ± 0.00117966 (W 0.00178803) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.530738 ± 0.00181538 (W 0.0021447) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.528274 ± 0.00198464 (W 0.00268213) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0013525 ± 2.37522e-05 (W 2.76458e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0030855 ± 2.9568e-05 (W 3.59076e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0050445 ± 4.08689e-05 (W 4.3869e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011366 ± 3.91088e-05 (W 5.11828e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0120625 ± 3.39848e-05 (W 5.23622e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0646205 ± 0.000203834 (W 0.000150945) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0663685 ± 0.000299412 (W 0.000218131) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.103212 ± 0.000221843 (W 0.000245493) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108653 ± 0.000399281 (W 0.000374198) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0816075 ± 0.000519688 (W 0.000245446) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0829545 ± 0.000999448 (W 0.000193652) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.260375 ± 0.00124291 (W 0.0018328) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.261161 ± 0.00143494 (W 0.00183008) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.544845 ± 0.0024425 (W 0.00270329) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.548361 ± 0.00353222 (W 0.00371385) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.015043 ± 0.0002449 (W 7.1912e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0164255 ± 0.000119834 (W 0.000170496) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0673905 ± 0.00170775 (W 0.000123624) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0702155 ± 0.000706252 (W 0.00017376) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.17687 ± 0.00146298 (W 0.00199943) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.194913 ± 0.00201003 (W 0.00252118) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0077045 ± 0.000195597 (W 0.000170895) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.007995 ± 0.00013194 (W 0.000208176) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.024075 ± 0.000336614 (W 0.000486462) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.024711 ± 0.000130374 (W 0.000364317) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0535295 ± 0.000637437 (W 0.000816928) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.053219 ± 0.000244729 (W 0.000983281) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0087565 ± 0.000291058 (W 8.16599e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010489 ± 0.000352713 (W 0.000116714) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034585 ± 0.000602618 (W 0.000387914) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.0385615 ± 0.000529373 (W 0.000503637) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0780505 ± 0.00116382 (W 0.0013607) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.095663 ± 0.001185 (W 0.00213716) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.046807 ± 0.000246722 (W 0.000158475) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0496575 ± 0.000316172 (W 0.000229331) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.124101 ± 0.000641007 (W 0.00114376) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.132146 ± 0.000607162 (W 0.000527044) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.234287 ± 0.000942427 (W 0.00197473) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.285017 ± 0.0110752 (W 0.00251394) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.025327 ± 0.000171009 (W 0.000151702) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.085865 ± 0.00043305 (W 0.000391432) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.16594 ± 0.000571576 (W 0.000496854) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.010309 ± 7.53658e-05 (W 7.61914e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.011817 ± 8.84119e-05 (W 9.01826e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0374495 ± 0.000327873 (W 0.000458983) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.041046 ± 0.000218105 (W 0.000333971) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.079533 ± 0.000417006 (W 0.000646907) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0997305 ± 0.00187884 (W 0.00256397) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.164467 ± 0.00156047 (W 0.00100311) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.166264 ± 0.00117394 (W 0.000287078) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.519337 ± 0.00994396 (W 0.00262584) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.521878 ± 0.00374563 (W 0.00280754) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.09524 ± 0.010768 (W 0.00391381) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.09479 ± 0.00419261 (W 0.00305355) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0602675 ± 0.000278928 (W 0.000119916) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.062091 ± 0.000215361 (W 0.000145139) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.224608 ± 0.000959725 (W 0.00198941) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.225194 ± 0.00102527 (W 0.00164404) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.541875 ± 0.00195446 (W 0.00303798) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.541384 ± 0.00392402 (W 0.00246299) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0013475 ± 5.31977e-06 (W 3.04245e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003056 ± 1.76068e-05 (W 3.27441e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0050145 ± 3.13076e-05 (W 4.66796e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `lu/32`, `lu/64`, `lu/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `solve/32`, `solve/64`, `solve/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `triangular_solve/32`, `triangular_solve/64`, `triangular_solve/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0575725 ± 0.00016685 (W 0.00029746) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.0601975 ± 0.00023723 (W 0.000693106) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.36515 ± 0.00135778 (W 0.00276274) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.346936 ± 0.00125801 (W 0.00208601) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.91731 ± 0.0122213 (W 0.00897878) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.18616 ± 0.0149646 (W 0.00365128) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.57696 ± 0.010985 (W 0.00486786) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.56535 ± 0.00654121 (W 0.00647942) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.6306 ± 0.0324036 (W 0.0102019) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.6539 ± 0.0403861 (W 0.0101759) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 51.0633 ± 0.23705 (W 0.0962503) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 50.8511 ± 0.242453 (W 0.0840993) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.366053 ± 0.00195677 (W 0.00239994) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.339723 ± 0.00121662 (W 0.00288704) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.9091 ± 0.0152628 (W 0.00778452) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.0538 ± 0.00857419 (W 0.00831924) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 21.4811 ± 0.0442447 (W 0.146865) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 22.3482 ± 0.0822711 (W 0.0786894) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.260396 ± 0.00152552 (W 0.0022112) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.259514 ± 0.00122027 (W 0.00250073) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.90312 ± 0.0115461 (W 0.00377504) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.89829 ± 0.0118216 (W 0.00525674) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9517 ± 0.0366709 (W 0.0330473) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.9819 ± 0.0122197 (W 0.0387455) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.103864 ± 0.000606525 (W 0.00188149) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.165744 ± 0.00143828 (W 0.0019304) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.569767 ± 0.00069873 (W 0.00282787) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.955225 ± 0.00879141 (W 0.00305744) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.86824 ± 0.011931 (W 0.00605343) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.73631 ± 0.083011 (W 0.0228734) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.450825 ± 0.00488967 (W 0.00286347) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.569352 ± 0.00181155 (W 0.00295948) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.90617 ± 0.00759219 (W 0.0047456) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.225 ± 0.00829471 (W 0.00782697) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 18.677 ± 0.0562424 (W 0.0419805) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 20.3231 ± 0.151379 (W 0.0247454) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.2521 ± 0.0018653 (W 0.00174865) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.52935 ± 0.00563582 (W 0.00444551) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 8.62079 ± 0.0152377 (W 0.0157657) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0951675 ± 0.000263318 (W 0.000185352) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.128059 ± 0.000294761 (W 0.00152879) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.535924 ± 0.00173638 (W 0.00346076) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.708584 ± 0.00231035 (W 0.00361465) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.74551 ± 0.0115774 (W 0.00642252) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.32648 ± 0.0181762 (W 0.0189332) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.87076 ± 0.00315001 (W 0.00584479) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.89272 ± 0.004641 (W 0.00428829) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.5325 ± 0.0274966 (W 0.0229805) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 14.5867 ± 0.044768 (W 0.0277156) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 86.5934 ± 0.163644 (W 0.103642) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 86.9407 ± 0.246618 (W 0.417905) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.927629 ± 0.00296593 (W 0.0031622) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.927123 ± 0.00365129 (W 0.00209861) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.48386 ± 0.00550847 (W 0.00924095) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.49408 ± 0.0124521 (W 0.00985691) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 37.2039 ± 0.08081 (W 0.0654509) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 37.2676 ± 0.287612 (W 0.0472145) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004283 ± 1.94079e-05 (W 6.10482e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.012048 ± 0.000140282 (W 0.000101539) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.042008 ± 0.00100609 (W 0.000457478) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.048911 ± 0.000738878 (W 0.000206731) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.056295 ± 0.000856165 (W 0.000956109) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.181798 ± 0.00127465 (W 0.00219045) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.229021 ± 0.00107057 (W 0.00164041) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.970625 ± 0.00453244 (W 0.00237282) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.00141 ± 0.00817274 (W 0.00868491) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.45425 ± 0.00269462 (W 0.00273444) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.45716 ± 0.00582931 (W 0.00385036) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.71794 ± 0.0150737 (W 0.0090886) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.71445 ± 0.0122119 (W 0.00917883) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.968 ± 0.0340194 (W 0.0266264) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.7443 ± 0.0348866 (W 0.0254947) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.333502 ± 0.00326849 (W 0.00384641) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.350473 ± 0.00244884 (W 0.00398126) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.99222 ± 0.0101234 (W 0.00449063) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.3019 ± 0.0108432 (W 0.0125321) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.4995 ± 0.020893 (W 0.0229847) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.2545 ± 0.0376182 (W 0.0398117) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.084813 ± 0.000528198 (W 0.000738543) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.084613 ± 0.000376856 (W 0.000664057) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.517334 ± 0.00414585 (W 0.00290228) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.520345 ± 0.00160306 (W 0.00276272) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.16758 ± 0.0186585 (W 0.0119873) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.17667 ± 0.0118101 (W 0.0100603) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.116977 ± 0.000800728 (W 0.00108494) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.157663 ± 0.00118563 (W 0.00184064) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.387792 ± 0.0037834 (W 0.0051559) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.724951 ± 0.00278547 (W 0.00728393) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.8449 ± 0.0089439 (W 0.00524104) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.57968 ± 0.037291 (W 0.0106496) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.350012 ± 0.00157469 (W 0.00256923) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.43291 ± 0.00119744 (W 0.00338617) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.067 ± 0.00329959 (W 0.0020341) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.59943 ± 0.00716902 (W 0.00781334) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.92575 ± 0.0397592 (W 0.00909002) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.50883 ± 0.032019 (W 0.0193228) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.228541 ± 0.0033565 (W 0.00233433) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.665107 ± 0.00306978 (W 0.0024245) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 3.25189 ± 0.00765764 (W 0.010341) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.09393 ± 0.000648424 (W 0.00125723) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.121916 ± 0.000657134 (W 0.000996786) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.392888 ± 0.00552269 (W 0.0030592) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.737444 ± 0.00370798 (W 0.00383239) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.72884 ± 0.00512063 (W 0.00711844) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.55368 ± 0.0370106 (W 0.0147676) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.01309 ± 0.00813157 (W 0.00376042) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.0223 ± 0.00757655 (W 0.00202803) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.12999 ± 0.0148645 (W 0.0130123) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.15493 ± 0.0232139 (W 0.0110247) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 36.6592 ± 0.0652353 (W 0.052618) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 36.8011 ± 0.156551 (W 0.0505958) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.95876 ± 0.0026382 (W 0.00225012) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.962255 ± 0.00416147 (W 0.00217708) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.15448 ± 0.0170737 (W 0.0104476) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.14763 ± 0.0155788 (W 0.009857) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.4922 ± 0.0735951 (W 0.026057) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.5339 ± 0.0514505 (W 0.0225092) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005445 ± 6.08693e-05 (W 5.30873e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01308 ± 0.00010238 (W 0.000129692) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.03203 ± 0.000122219 (W 0.000152213) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0494175 ± 0.000460267 (W 0.000207011) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.057302 ± 0.000338687 (W 0.000802732) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.160474 ± 0.00123123 (W 0.00283805) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.21495 ± 0.00159675 (W 0.00250277) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.794585 ± 0.00491894 (W 0.00316757) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.832267 ± 0.00272719 (W 0.00447372) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.94643 ± 0.00235301 (W 0.00201432) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.947289 ± 0.0020435 (W 0.00292137) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.99215 ± 0.00777854 (W 0.00559812) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 4.02169 ± 0.00849767 (W 0.00923545) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.5858 ± 0.037944 (W 0.02551) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.5019 ± 0.0411754 (W 0.0279948) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.335545 ± 0.00389239 (W 0.00426749) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.359083 ± 0.00361094 (W 0.0035779) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.88423 ± 0.0089221 (W 0.0110668) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.22683 ± 0.0116053 (W 0.0114723) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.1284 ± 0.0467065 (W 0.0170531) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.9503 ± 0.0753557 (W 0.0208794) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0596865 ± 0.000635489 (W 0.001725) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0594205 ± 0.00103137 (W 0.00135232) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.285376 ± 0.00235201 (W 0.00263565) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.285647 ± 0.0014625 (W 0.00315299) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.28104 ± 0.017614 (W 0.00779841) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.27172 ± 0.00886236 (W 0.0091943) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.124527 ± 0.00110718 (W 0.00123075) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.158204 ± 0.00164602 (W 0.000935595) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.326533 ± 0.00127108 (W 0.00308492) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.690356 ± 0.00431763 (W 0.00429093) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.3632 ± 0.00750634 (W 0.00527416) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.2245 ± 0.0428937 (W 0.0104972) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.327775 ± 0.00179193 (W 0.00320882) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.415663 ± 0.00299601 (W 0.00208808) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.880399 ± 0.00455479 (W 0.00339471) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.41773 ± 0.0077156 (W 0.0050474) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.48481 ± 0.0828106 (W 0.00672616) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 6.1379 ± 0.0527919 (W 0.0167922) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.209195 ± 0.00101565 (W 0.00213065) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.614655 ± 0.00364904 (W 0.00223454) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.94332 ± 0.00824843 (W 0.0044103) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.087798 ± 0.000576682 (W 0.000750538) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.120064 ± 0.000576724 (W 0.000453226) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.322261 ± 0.00233729 (W 0.0023501) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.703353 ± 0.00454587 (W 0.00373681) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.25434 ± 0.00602271 (W 0.00484826) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.17309 ± 0.0324609 (W 0.00980412) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.9951 ± 0.0207729 (W 0.00483202) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99223 ± 0.0101092 (W 0.00530017) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.40222 ± 0.0221348 (W 0.0182071) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.44303 ± 0.0847689 (W 0.021735) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.3033 ± 0.0529128 (W 0.0397389) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.5294 ± 0.110953 (W 0.0408659) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.00489 ± 0.00465461 (W 0.00358711) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.01168 ± 0.00734573 (W 0.00303286) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.0142 ± 0.0159243 (W 0.00816616) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.02548 ± 0.0131047 (W 0.013209) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.359 ± 0.0444792 (W 0.0213189) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.3752 ± 0.101205 (W 0.0271502) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005445 ± 6.66646e-05 (W 5.48515e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013014 ± 0.00011401 (W 0.000104226) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030151 ± 0.000267821 (W 0.000143452) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.048691 ± 0.000228215 (W 0.000108003) | 0.063168 ± 0.000229106 (W 0.000258178) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.053936 ± 0.00019612 (W 0.000163206) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.282 ± 0.000482409 (W 0.00263916) | 0.366096 ± 0.00338178 (W 0.00249956) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.303987 ± 0.000673247 (W 0.00250376) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.96128 ± 0.00774624 (W 0.00466793) | 2.54508 ± 0.00812444 (W 0.00704248) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.0165 ± 0.00890444 (W 0.00583421) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.5789 ± 0.00930727 (W 0.00761468) | 1.78731 ± 0.00332571 (W 0.00694777) | 🔴 faer-fork 1.44× slower than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 2.56131 ± 0.00748456 (W 0.00526758) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.6288 ± 0.0367291 (W 0.0134029) | 8.63171 ± 0.02659 (W 0.022116) | 🔴 faer-fork 1.23× slower than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 10.6348 ± 0.0445414 (W 0.0122905) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 50.8598 ± 0.272298 (W 0.0426065) | 49.0583 ± 0.26643 (W 0.0561434) | · |  |
| `eigh/n512/b1 [api-inclusive]` | 50.8696 ± 0.155892 (W 0.0492142) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.326773 ± 0.00150522 (W 0.00276627) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.315352 ± 0.000932291 (W 0.00203891) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.87808 ± 0.00664019 (W 0.0103762) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.94627 ± 0.00778599 (W 0.0129472) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 22.3756 ± 1.111 (W 0.0312425) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 21.7839 ± 0.111236 (W 0.0363525) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.25752 ± 0.00153279 (W 0.00258574) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.256367 ± 0.00101504 (W 0.00235508) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.8761 ± 0.00657017 (W 0.00501629) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.87377 ± 0.00685793 (W 0.00496444) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.914 ± 0.0433601 (W 0.0372172) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.8206 ± 0.0404735 (W 0.0398477) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.09664 ± 0.000470257 (W 0.000994959) | 0.140733 ± 0.000778614 (W 0.00125455) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.143467 ± 0.000921357 (W 0.00186965) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.571236 ± 0.00119078 (W 0.00289563) | 0.775845 ± 0.00146516 (W 0.00244062) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.854322 ± 0.00564758 (W 0.00466553) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.75061 ± 0.00503709 (W 0.00525347) | 4.93879 ± 0.0141208 (W 0.00732522) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 5.15814 ± 0.0344565 (W 0.0102473) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.456506 ± 0.00554321 (W 0.00236905) | 0.810304 ± 0.00136206 (W 0.00561493) | 🔵 faer-fork 1.78× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.537461 ± 0.00203929 (W 0.00416126) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.63571 ± 0.00524537 (W 0.00463528) | 4.5466 ± 0.0137361 (W 0.0141175) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 3.09332 ± 0.00848102 (W 0.00768973) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 17.5742 ± 0.123347 (W 0.026559) | 31.6884 ± 0.0723768 (W 0.0444014) | 🔵 faer-fork 1.80× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 19.6906 ± 0.131465 (W 0.0442203) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.256613 ± 0.0020606 (W 0.00189929) | 0.365976 ± 0.00108875 (W 0.00260188) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 1.28985 ± 0.00492115 (W 0.00299586) | 1.80943 ± 0.00509952 (W 0.00535041) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 7.6741 ± 0.0366618 (W 0.0110818) | 12.2262 ± 0.0187874 (W 0.0163741) | 🔵 faer-fork 1.59× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.095208 ± 0.000624967 (W 0.000801301) | 0.136675 ± 0.00144708 (W 0.00164678) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.101849 ± 0.000326116 (W 0.00160762) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.561651 ± 0.00177404 (W 0.0029576) | 0.763597 ± 0.00327594 (W 0.00401844) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.599376 ± 0.00204301 (W 0.00335393) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.70576 ± 0.00941226 (W 0.00567458) | 4.88173 ± 0.014882 (W 0.015969) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 3.84754 ± 0.0120349 (W 0.0119954) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.86029 ± 0.0118076 (W 0.00692672) | 2.9841 ± 0.00912154 (W 0.0116342) | · |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.87413 ± 0.00569713 (W 0.0067403) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.4917 ± 0.0248688 (W 0.0175354) | 16.7272 ± 0.0316052 (W 0.019506) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 14.5076 ± 0.0207669 (W 0.0176854) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 85.8452 ± 0.145311 (W 0.355747) | 114.805 ± 0.802727 (W 0.332889) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 86.0734 ± 0.152811 (W 0.0823818) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.905773 ± 0.0020522 (W 0.00228805) | 1.30857 ± 0.00685959 (W 0.00411842) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.907641 ± 0.00490484 (W 0.0022727) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.38212 ± 0.02092 (W 0.0110801) | 7.50203 ± 0.0486539 (W 0.0135614) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.36044 ± 0.0174472 (W 0.0107908) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 36.1467 ± 0.0638309 (W 0.0387664) | 51.5307 ± 0.1728 (W 0.240134) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 36.0991 ± 0.174335 (W 0.0280625) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0046785 ± 2.77399e-05 (W 6.70834e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.012724 ± 5.21821e-05 (W 7.61111e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0435265 ± 0.000816669 (W 0.0002263) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.041973 ± 0.000450991 (W 0.000611462) | 0.0656575 ± 0.00039959 (W 0.00161146) | 🔵 faer-fork 1.56× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.0502235 ± 0.000816925 (W 0.000375504) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.147004 ± 0.00118381 (W 0.00189863) | 0.232298 ± 0.00192514 (W 0.00179581) | 🔵 faer-fork 1.58× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.188231 ± 0.00110732 (W 0.00194772) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.651213 ± 0.00235136 (W 0.00220483) | 1.13353 ± 0.00665245 (W 0.00476361) | 🔵 faer-fork 1.74× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.833467 ± 0.00616499 (W 0.00234321) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.45094 ± 0.00428977 (W 0.00333056) | 1.44158 ± 0.0042981 (W 0.00460369) | · |  |
| `eigh/n128/b1 [api-inclusive]` | 1.45544 ± 0.00519106 (W 0.00400264) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.67499 ± 0.0140186 (W 0.0112382) | 5.69697 ± 0.019653 (W 0.0109515) | · |  |
| `eigh/n256/b1 [api-inclusive]` | 5.68084 ± 0.0207507 (W 0.00895291) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.7957 ± 0.0259301 (W 0.0230361) | 25.9776 ± 0.216871 (W 0.0421059) | · |  |
| `eigh/n512/b1 [api-inclusive]` | 24.6148 ± 0.0331173 (W 0.0240895) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.32253 ± 0.00308409 (W 0.00508566) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.325847 ± 0.00274458 (W 0.0045656) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.98414 ± 0.0116121 (W 0.0059737) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.19787 ± 0.00855289 (W 0.00766659) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.7274 ± 0.0225136 (W 0.0189372) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.6986 ± 0.0300969 (W 0.029537) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.083876 ± 0.000367821 (W 0.000927599) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0841965 ± 0.000650318 (W 0.000412941) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.525174 ± 0.00335058 (W 0.00379631) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.527152 ± 0.00162915 (W 0.0032146) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.18938 ± 0.00962996 (W 0.00965784) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.18277 ± 0.012992 (W 0.0113728) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0786915 ± 0.0010861 (W 0.000695514) | 0.141183 ± 0.000767424 (W 0.00171565) | 🔵 faer-fork 1.79× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.133143 ± 0.0017806 (W 0.00150018) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.331778 ± 0.00354568 (W 0.00260651) | 0.440341 ± 0.00127007 (W 0.00293879) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.612541 ± 0.00838434 (W 0.00474425) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.57882 ± 0.00573098 (W 0.00413524) | 2.39753 ± 0.0065867 (W 0.00519665) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.99296 ± 0.0170117 (W 0.0102645) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.305153 ± 0.0027149 (W 0.00178879) | 0.795952 ± 0.00294328 (W 0.00376123) | 🔵 faer-fork 2.61× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.402228 ± 0.00289396 (W 0.00286461) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.984415 ± 0.00190363 (W 0.00266046) | 3.26465 ± 0.00499166 (W 0.00749115) | 🔵 faer-fork 3.32× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 1.4763 ± 0.00581047 (W 0.00645057) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.57451 ± 0.0144123 (W 0.0101269) | 18.8391 ± 0.0818158 (W 0.0332144) | 🔵 faer-fork 3.38× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 7.81753 ± 0.0171733 (W 0.021118) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.209195 ± 0.00133418 (W 0.00176134) | 0.381665 ± 0.00254434 (W 0.00231696) | 🔵 faer-fork 1.82× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 0.589152 ± 0.00311716 (W 0.00335536) | 1.40784 ± 0.00723185 (W 0.00364837) | 🔵 faer-fork 2.39× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 2.95607 ± 0.00829193 (W 0.00631668) | 7.44942 ± 0.0327568 (W 0.0148441) | 🔵 faer-fork 2.52× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.082669 ± 0.00011588 (W 0.00164512) | 0.102681 ± 0.000501811 (W 0.00116451) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.09645 ± 0.000752531 (W 0.000711021) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.320957 ± 0.00281562 (W 0.00226383) | 0.416911 ± 0.00172196 (W 0.00286097) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.624012 ± 0.00316123 (W 0.00476956) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.49732 ± 0.00449157 (W 0.00419962) | 2.31819 ± 0.00579918 (W 0.00599648) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 2.95593 ± 0.00930381 (W 0.00985968) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.99242 ± 0.00586659 (W 0.0040518) | 2.94225 ± 0.00581114 (W 0.00452903) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99931 ± 0.00723679 (W 0.00275474) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.04919 ± 0.0216534 (W 0.0150555) | 14.2218 ± 0.0566433 (W 0.0263142) | 🔵 faer-fork 1.77× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 8.06257 ± 0.0189521 (W 0.0132717) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 35.7277 ± 0.0723346 (W 0.0489366) | 87.3693 ± 1.39546 (W 0.104196) | 🔵 faer-fork 2.45× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 35.9074 ± 0.0662219 (W 0.0462038) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.942975 ± 0.00355714 (W 0.00170772) | 1.32705 ± 0.00216992 (W 0.00280426) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.945285 ± 0.00270515 (W 0.00293281) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.01544 ± 0.0117977 (W 0.00909151) | 7.32609 ± 0.00859678 (W 0.0084088) | 🔵 faer-fork 1.82× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 4.0251 ± 0.0158815 (W 0.00967713) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.4718 ± 0.0240535 (W 0.0180229) | 48.4361 ± 0.445729 (W 0.064568) | 🔵 faer-fork 2.77× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 17.5199 ± 0.0419872 (W 0.0193608) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005856 ± 2.37522e-05 (W 6.05566e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0154185 ± 0.000209559 (W 0.000274773) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.039258 ± 0.000470168 (W 0.000388335) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.043391 ± 0.000376026 (W 0.00063776) | 0.0657675 ± 0.000714335 (W 0.000823765) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.050975 ± 0.000608993 (W 0.000636196) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.137161 ± 0.00157978 (W 0.00115055) | 0.235198 ± 0.00187186 (W 0.00318493) | 🔵 faer-fork 1.71× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.17193 ± 0.00102101 (W 0.00131189) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.518619 ± 0.0024249 (W 0.00275017) | 1.00883 ± 0.00327976 (W 0.00330982) | 🔵 faer-fork 1.95× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.673088 ± 0.00257531 (W 0.00435451) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.942205 ± 0.00529001 (W 0.00271064) | 1.37551 ± 0.00432486 (W 0.0033258) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.943819 ± 0.00715694 (W 0.00363872) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.9399 ± 0.00672618 (W 0.00624757) | 5.27941 ± 0.0167326 (W 0.00991916) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.96821 ± 0.0161948 (W 0.00536893) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.3992 ± 0.182741 (W 0.021819) | 22.3176 ± 0.0930765 (W 0.032576) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 16.3451 ± 0.0551345 (W 0.0162476) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.327856 ± 0.00268976 (W 0.00274812) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.337723 ± 0.00261817 (W 0.00272276) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.87146 ± 0.00787671 (W 0.00807768) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.11441 ± 0.00712307 (W 0.00872634) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.1541 ± 0.0368289 (W 0.032705) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.3788 ± 0.0314577 (W 0.0225506) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0561195 ± 0.00082695 (W 0.000484906) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0566705 ± 0.000485343 (W 0.00152139) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.288852 ± 0.00158809 (W 0.00160728) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.293678 ± 0.00239771 (W 0.00272817) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.28512 ± 0.00585042 (W 0.00989453) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.2819 ± 0.0107389 (W 0.00771119) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0791325 ± 0.000489065 (W 0.000524894) | 0.141003 ± 0.00111565 (W 0.00206277) | 🔵 faer-fork 1.78× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.134526 ± 0.00152198 (W 0.000901133) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.292314 ± 0.0014854 (W 0.00208841) | 0.372783 ± 0.00195896 (W 0.00269606) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.583201 ± 0.00341981 (W 0.00333989) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.20645 ± 0.00618926 (W 0.00491828) | 1.51338 ± 0.00573454 (W 0.00697002) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.65072 ± 0.0175129 (W 0.0131509) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.275423 ± 0.00197596 (W 0.00206414) | 0.784888 ± 0.00381925 (W 0.00450691) | 🔵 faer-fork 2.85× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.382854 ± 0.00216407 (W 0.00266025) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.813084 ± 0.00372273 (W 0.00332025) | 3.15212 ± 0.0128894 (W 0.00862681) | 🔵 faer-fork 3.88× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 1.29307 ± 0.0192531 (W 0.00743204) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.24991 ± 0.00921369 (W 0.00730698) | 16.519 ± 0.143955 (W 0.0293618) | 🔵 faer-fork 5.08× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 5.53251 ± 0.146107 (W 0.0228546) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.199837 ± 0.00317072 (W 0.00203205) | 0.381571 ± 0.00156832 (W 0.00233307) | 🔵 faer-fork 1.91× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 0.573904 ± 0.00172422 (W 0.00247281) | 1.37877 ± 0.00732001 (W 0.00540695) | 🔵 faer-fork 2.40× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 1.79279 ± 0.00480131 (W 0.00415701) | 6.61813 ± 0.0178715 (W 0.0135394) | 🔵 faer-fork 3.69× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.0785415 ± 0.000812343 (W 0.00170087) | 0.137261 ± 0.000700963 (W 0.00225413) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.094962 ± 0.000640838 (W 0.000657087) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.277277 ± 0.00110042 (W 0.00228764) | 0.376566 ± 0.00231495 (W 0.0032994) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.592507 ± 0.00302835 (W 0.00516196) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.08961 ± 0.00382494 (W 0.0034355) | 1.46036 ± 0.00849495 (W 0.00698945) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 2.59588 ± 0.0149603 (W 0.0068281) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.97154 ± 0.00736798 (W 0.00616787) | 2.92365 ± 0.00706284 (W 0.00754129) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.97211 ± 0.00674242 (W 0.00362233) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.26963 ± 0.0119294 (W 0.0149929) | 14.0054 ± 0.0623738 (W 0.0237566) | 🔵 faer-fork 1.93× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.31933 ± 0.0295536 (W 0.0166941) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 29.4317 ± 0.490831 (W 0.0468699) | 83.1648 ± 0.397966 (W 0.0969837) | 🔵 faer-fork 2.83× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 29.5456 ± 0.518453 (W 0.073567) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.980148 ± 0.00337413 (W 0.00195618) | 1.33397 ± 0.00673732 (W 0.00360698) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.982937 ± 0.00537428 (W 0.00367472) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.88702 ± 0.0135737 (W 0.00543249) | 7.32204 ± 0.0167673 (W 0.00810785) | 🔵 faer-fork 1.88× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.90039 ± 0.0129859 (W 0.010265) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.38 ± 0.013067 (W 0.0242783) | 47.9834 ± 0.311003 (W 0.0243004) | 🔵 faer-fork 3.12× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 15.422 ± 0.0284356 (W 0.023026) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.006066 ± 2.40866e-05 (W 5.93346e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015564 ± 0.000223319 (W 0.000177109) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0378105 ± 0.000553955 (W 0.000292371) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.181719 ± 0.00061581 (W 0.00193129) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.189779 ± 0.0007064 (W 0.00170013) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.14738 ± 0.0029768 (W 0.0035015) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.16411 ± 0.00422532 (W 0.00287353) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.54949 ± 0.0170637 (W 0.0110675) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 7.71219 ± 0.0306299 (W 0.0157691) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.59067 ± 0.013778 (W 0.00441069) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 3.57862 ± 0.0140717 (W 0.0053196) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 18.2265 ± 0.0491725 (W 0.0240999) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 18.2745 ± 0.0495908 (W 0.0229909) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 107.261 ± 0.410521 (W 0.0826644) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 107.637 ± 0.227332 (W 0.245937) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.833373 ± 0.0249142 (W 0.00683686) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.926478 ± 0.0310606 (W 0.00483912) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 6.06254 ± 0.0474261 (W 0.0112999) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.64345 ± 0.0550858 (W 0.0151372) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 41.4953 ± 0.201634 (W 0.13829) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 44.2298 ± 0.131457 (W 0.073477) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.05437 ± 0.00497313 (W 0.00317424) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.05202 ± 0.00557683 (W 0.00248978) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.56539 ± 0.0175013 (W 0.0131592) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.57396 ± 0.019908 (W 0.00970606) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.8753 ± 0.118448 (W 0.106578) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.8935 ± 0.15825 (W 0.105182) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.288959 ± 0.00194199 (W 0.00294116) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.43787 ± 0.0025846 (W 0.00361324) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.90385 ± 0.00401393 (W 0.00544853) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.66563 ± 0.0133842 (W 0.00500043) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.4171 ± 0.0254482 (W 0.0253014) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 17.8864 ± 0.0654513 (W 0.043602) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.46698 ± 0.0023221 (W 0.00379776) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 1.7132 ± 0.00628011 (W 0.00399269) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 10.1872 ± 0.0147966 (W 0.0158417) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 11.0352 ± 0.0887337 (W 0.0212732) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 70.208 ± 0.0864751 (W 0.0839551) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 74.2392 ± 0.0586877 (W 0.173927) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.678272 ± 0.00176972 (W 0.00287735) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 4.64867 ± 0.0122066 (W 0.0124051) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 30.0969 ± 0.0864692 (W 0.0680353) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.279747 ± 0.0011076 (W 0.00224464) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.447413 ± 0.00159678 (W 0.00336369) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.88406 ± 0.00806206 (W 0.00354256) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 2.07886 ± 0.00774586 (W 0.00378576) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.2972 ± 0.0124717 (W 0.0234387) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 14.786 ± 0.0710723 (W 0.0251714) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.86947 ± 0.0169688 (W 0.00659379) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 4.88829 ± 0.0151242 (W 0.0060458) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 30.1958 ± 0.10128 (W 0.0460569) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 30.3463 ± 0.063711 (W 0.0409885) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 198.753 ± 0.481004 (W 0.309214) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 199.415 ± 0.280656 (W 0.38157) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.69384 ± 0.00647534 (W 0.00369208) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.69799 ± 0.0062115 (W 0.00483247) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 12.0924 ± 0.0372054 (W 0.0280284) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 12.1082 ± 0.0660018 (W 0.0243227) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 87.1225 ± 0.311646 (W 0.136868) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 87.2331 ± 0.29974 (W 0.336754) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007975 ± 2.08207e-05 (W 4.58246e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024541 ± 0.00014745 (W 0.000121856) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.089627 ± 0.000687639 (W 0.00238598) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.138307 ± 0.00197683 (W 0.00162056) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.158214 ± 0.00107099 (W 0.00239808) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.521362 ± 0.00201095 (W 0.00343391) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.614038 ± 0.00214102 (W 0.00355501) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.31211 ± 0.00680982 (W 0.00499243) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.7819 ± 0.0159655 (W 0.00799707) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.95234 ± 0.00692914 (W 0.00669473) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.95956 ± 0.00874434 (W 0.00578172) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.91935 ± 0.0113148 (W 0.0118212) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 7.95623 ± 0.0287011 (W 0.0161823) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.9958 ± 0.0639606 (W 0.051074) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 40.0438 ± 0.0433377 (W 0.0500971) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.608019 ± 0.00566105 (W 0.00423664) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.725141 ± 0.0057388 (W 0.00408129) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.33935 ± 0.0174125 (W 0.0101134) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 4.07501 ± 0.010585 (W 0.0137811) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 16.6816 ± 0.0473895 (W 0.0265188) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 20.7054 ± 0.0963021 (W 0.0413939) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.286878 ± 0.00071032 (W 0.00216936) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.287441 ± 0.000471886 (W 0.00237075) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.95117 ± 0.00779776 (W 0.00494057) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.95368 ± 0.00448092 (W 0.00243347) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6236 ± 0.0299203 (W 0.0224203) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6034 ± 0.0189 (W 0.0187414) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.231576 ± 0.00132084 (W 0.00225711) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.364299 ± 0.00369924 (W 0.00555698) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.00166 ± 0.0134846 (W 0.00797144) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.7291 ± 0.00838445 (W 0.0134369) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.06534 ± 0.0291006 (W 0.0357705) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 9.3914 ± 0.0327651 (W 0.0718317) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.734503 ± 0.0026891 (W 0.00282759) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.937571 ± 0.00397747 (W 0.00360252) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.11063 ± 0.0107424 (W 0.00871528) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 4.21465 ± 0.0171741 (W 0.011451) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 19.8614 ± 0.061728 (W 0.0223643) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 25.509 ± 0.063412 (W 0.0420659) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.443696 ± 0.00177209 (W 0.00299997) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.6444 ± 0.00795619 (W 0.00330984) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 9.42504 ± 0.041163 (W 0.0225438) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.202352 ± 0.000925958 (W 0.00178477) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.369193 ± 0.0030403 (W 0.00436668) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.965622 ± 0.00414157 (W 0.00878629) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.73002 ± 0.015849 (W 0.0142261) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.1486 ± 0.0392856 (W 0.0274695) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 9.43417 ± 0.107423 (W 0.0604019) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.7406 ± 0.00987713 (W 0.00967954) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.75743 ± 0.0106259 (W 0.00593905) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 13.125 ± 0.0218958 (W 0.0220572) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 13.2086 ± 0.0274013 (W 0.0258835) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 69.9801 ± 0.209374 (W 0.105843) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 70.4336 ± 0.104634 (W 0.113674) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.29196 ± 0.00633448 (W 0.00640046) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.28602 ± 0.00189002 (W 0.00291498) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.5628 ± 0.0173067 (W 0.0101644) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.5835 ± 0.00939664 (W 0.0125964) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 35.22 ± 0.0869703 (W 0.0459419) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 35.3014 ± 0.10357 (W 0.0405058) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009107 ± 2.42212e-05 (W 5.94844e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.021901 ± 8.50882e-05 (W 0.000139043) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.056801 ± 0.000664378 (W 0.000252396) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.137556 ± 0.000976713 (W 0.00122645) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.155199 ± 0.00167206 (W 0.00142329) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.460106 ± 0.00424332 (W 0.00399748) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.544739 ± 0.00283006 (W 0.00269117) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.65064 ± 0.00640152 (W 0.00405202) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.10519 ± 0.0178883 (W 0.0086719) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32001 ± 0.00959796 (W 0.00439121) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.32519 ± 0.00792506 (W 0.00418383) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.42845 ± 0.0149815 (W 0.0109738) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.44897 ± 0.0139129 (W 0.0089333) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.3776 ± 0.0355814 (W 0.0225575) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 25.5013 ± 0.0643693 (W 0.0280573) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.587413 ± 0.00403066 (W 0.00376884) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.711199 ± 0.00460804 (W 0.00597794) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.9922 ± 0.0177862 (W 0.00762042) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.73298 ± 0.0143069 (W 0.0117963) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.9036 ± 0.0128045 (W 0.0251137) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 17.2983 ± 0.134112 (W 0.0603512) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.18003 ± 0.000734884 (W 0.00120248) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.180981 ± 0.000571413 (W 0.00168204) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00723 ± 0.0053829 (W 0.0028851) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00801 ± 0.0048334 (W 0.00379206) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.90905 ± 0.0352427 (W 0.040191) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.88959 ± 0.0177786 (W 0.0364865) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.231236 ± 0.0015337 (W 0.00183378) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.349109 ± 0.0032599 (W 0.00281451) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.78126 ± 0.00388158 (W 0.00382588) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.5434 ± 0.010356 (W 0.00515484) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.45592 ± 0.0114617 (W 0.011936) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.79266 ± 0.0526394 (W 0.0700043) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.580166 ± 0.00310112 (W 0.00254489) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.809448 ± 0.0202213 (W 0.0065344) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 2.12832 ± 0.00669865 (W 0.00323705) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.28829 ± 0.0132905 (W 0.0131759) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.8095 ± 0.019554 (W 0.0301508) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 16.6279 ± 0.340387 (W 0.0687048) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.387176 ± 0.00111146 (W 0.00272874) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.33521 ± 0.00272811 (W 0.00326283) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.15024 ± 0.186308 (W 0.0239478) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.184834 ± 0.00117674 (W 0.0018639) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.355422 ± 0.00260787 (W 0.00259538) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.798771 ± 0.00297229 (W 0.003061) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.54889 ± 0.0133062 (W 0.00578273) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.4265 ± 0.00828248 (W 0.0115946) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.83029 ± 0.0551681 (W 0.0198471) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.5379 ± 0.0112355 (W 0.0110756) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.54678 ± 0.014696 (W 0.00645671) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.9621 ± 0.0244451 (W 0.0244621) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 11.0368 ± 0.0625627 (W 0.0216863) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 50.4721 ± 0.0977813 (W 0.081424) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 50.7286 ± 0.18394 (W 0.0920338) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.28237 ± 0.00671849 (W 0.00335052) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.28427 ± 0.00805831 (W 0.00433276) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.94168 ± 0.0286238 (W 0.00906083) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.94669 ± 0.0390606 (W 0.0128243) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 27.2844 ± 0.0772086 (W 0.0257981) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 27.3691 ± 0.0810062 (W 0.0310134) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009177 ± 2.42212e-05 (W 7.2034e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.021155 ± 0.000202627 (W 0.000134403) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.050469 ± 0.000596222 (W 0.000300569) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.166906 ± 0.000465937 (W 0.00210589) | 0.169966 ± 0.00065548 (W 0.00195737) | · |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.17934 ± 0.00155541 (W 0.00190214) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.08515 ± 0.00149484 (W 0.00202299) | 1.22861 ± 0.00715455 (W 0.00488098) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 1.12379 ± 0.00424889 (W 0.00319138) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.36104 ± 0.0188971 (W 0.0120286) | 8.74307 ± 0.0178943 (W 0.022088) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 7.50504 ± 0.0124408 (W 0.00928447) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.57381 ± 0.00593017 (W 0.0100873) | 3.21801 ± 0.00857445 (W 0.00795239) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 3.57423 ± 0.00934624 (W 0.00862093) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 18.1741 ± 0.0566614 (W 0.0291973) | 19.4104 ± 0.0712089 (W 0.0238825) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 18.1894 ± 0.0533457 (W 0.0314932) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 107.235 ± 0.217288 (W 0.127839) | 139.592 ± 0.730864 (W 0.138904) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 107.476 ± 0.375251 (W 0.17128) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.801614 ± 0.011814 (W 0.00501445) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.905452 ± 0.00556536 (W 0.00580781) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 5.95976 ± 0.0474405 (W 0.0139679) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.47166 ± 0.0938752 (W 0.0315279) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 43.0153 ± 0.113301 (W 0.0464906) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 43.0158 ± 0.100617 (W 0.0646757) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.04253 ± 0.00360021 (W 0.00182483) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.04495 ± 0.00398955 (W 0.00367589) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.48814 ± 0.0151838 (W 0.0210678) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.49384 ± 0.020674 (W 0.0101893) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.6773 ± 0.20309 (W 0.121883) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.6698 ± 0.0954089 (W 0.0894229) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.283013 ± 0.00166166 (W 0.00237848) | 0.328712 ± 0.00122414 (W 0.00199607) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.411888 ± 0.00182021 (W 0.00283543) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.87066 ± 0.00476991 (W 0.00410113) | 2.24246 ± 0.00175738 (W 0.00300162) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 2.5219 ± 0.0278617 (W 0.00633783) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.2339 ± 0.0552383 (W 0.0200762) | 15.8819 ± 0.0225288 (W 0.0351576) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 16.8423 ± 0.0426425 (W 0.0404527) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.49908 ± 0.00431923 (W 0.00310938) | 1.95955 ± 0.00529814 (W 0.00482652) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 1.68398 ± 0.0077627 (W 0.00478107) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 9.92277 ± 0.014142 (W 0.0145009) | 13.2586 ± 0.0222754 (W 0.0198951) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 10.8536 ± 0.0277868 (W 0.0141602) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 68.2498 ± 0.17349 (W 0.0690535) | 89.0695 ± 0.291878 (W 0.218037) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 73.222 ± 0.151745 (W 0.163438) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.714801 ± 0.0039881 (W 0.00364749) | 0.733648 ± 0.0021241 (W 0.00212289) | · |  |
| `qr_factor/n256/b1` | 4.38264 ± 0.0116333 (W 0.00929627) | 5.15218 ± 0.0255509 (W 0.00810847) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 28.314 ± 0.0645917 (W 0.0246394) | 35.076 ± 0.0727538 (W 0.0366721) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.295641 ± 0.00101155 (W 0.00231795) | 0.346866 ± 0.00166277 (W 0.00255672) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.419286 ± 0.00175596 (W 0.00266192) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.90623 ± 0.0079449 (W 0.00723807) | 2.35396 ± 0.0161848 (W 0.00577448) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.96735 ± 0.00453498 (W 0.00306879) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.3186 ± 0.0164237 (W 0.0157797) | 16.2164 ± 0.0454888 (W 0.0221763) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 13.9994 ± 0.0299106 (W 0.0349701) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.84428 ± 0.0175079 (W 0.00754353) | 5.48414 ± 0.0175889 (W 0.00959978) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.8698 ± 0.0121586 (W 0.00481143) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 30.0146 ± 0.11811 (W 0.0386567) | 35.6455 ± 0.0881954 (W 0.0427855) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 30.0267 ± 0.105925 (W 0.0386727) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 197.295 ± 0.722387 (W 0.508896) | 230.487 ± 1.57857 (W 0.470751) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 198.372 ± 0.517535 (W 0.425629) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.66344 ± 0.00466289 (W 0.00626979) | 2.23091 ± 0.0070429 (W 0.00800741) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.66736 ± 0.00450474 (W 0.00383357) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.8791 ± 0.0304894 (W 0.0273255) | 15.5795 ± 0.0424387 (W 0.0311968) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 11.8741 ± 0.0479806 (W 0.0205493) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 86.1799 ± 0.273703 (W 0.111921) | 96.9606 ± 0.88318 (W 0.326365) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 86.2338 ± 0.466233 (W 0.222358) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0079895 ± 2.90218e-05 (W 5.14006e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0247765 ± 0.00021934 (W 0.000135266) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0899525 ± 0.000459784 (W 0.00225191) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.130368 ± 0.00196263 (W 0.00127233) | 0.100417 ± 0.000530816 (W 0.000908977) | 🔴 faer-fork 1.30× slower than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.146097 ± 0.00174113 (W 0.0020554) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.484482 ± 0.00372484 (W 0.00235048) | 0.497572 ± 0.00158652 (W 0.00183066) | · |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.570587 ± 0.00436193 (W 0.00370628) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.22438 ± 0.00544288 (W 0.00369115) | 3.12773 ± 0.00860275 (W 0.00460603) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.5781 ± 0.0109056 (W 0.00837404) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.93892 ± 0.00431373 (W 0.00343934) | 2.55923 ± 0.00667837 (W 0.00475253) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.9437 ± 0.00639127 (W 0.00517048) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.85959 ± 0.00729476 (W 0.0140736) | 11.3492 ± 0.0316092 (W 0.0102126) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 7.87956 ± 0.012361 (W 0.0103375) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.7436 ± 0.0902364 (W 0.025424) | 52.3497 ± 0.0635489 (W 0.0599743) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 39.8498 ± 0.0522555 (W 0.036141) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.589673 ± 0.00840129 (W 0.00448779) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.699233 ± 0.00504469 (W 0.00471998) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.3456 ± 0.141845 (W 0.0180619) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.88818 ± 0.0123895 (W 0.0122523) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 17.2237 ± 0.0614697 (W 0.0304582) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 19.7006 ± 0.0899332 (W 0.0518833) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.285511 ± 0.00111305 (W 0.00169951) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.286418 ± 0.000446499 (W 0.00183516) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.95097 ± 0.00332987 (W 0.00285889) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.95613 ± 0.00252552 (W 0.00266988) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6759 ± 0.0292012 (W 0.0325347) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6614 ± 0.0425933 (W 0.0269128) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.20082 ± 0.00250392 (W 0.00266413) | 0.21453 ± 0.00125997 (W 0.00235945) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.339322 ± 0.00326651 (W 0.00498485) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.902388 ± 0.0139722 (W 0.00582326) | 1.16436 ± 0.0040912 (W 0.00398858) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.57569 ± 0.00648064 (W 0.0111649) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.6615 ± 0.0108964 (W 0.0429161) | 5.08097 ± 0.00776624 (W 0.00449672) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 8.31227 ± 0.0589555 (W 0.0482104) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.688058 ± 0.00284955 (W 0.00317157) | 1.34757 ± 0.00557638 (W 0.0063382) | 🔵 faer-fork 1.96× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.912881 ± 0.0042794 (W 0.00367397) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.02464 ± 0.0120658 (W 0.00534528) | 7.00174 ± 0.0361352 (W 0.0139226) | 🔵 faer-fork 2.31× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 4.04949 ± 0.0269551 (W 0.00671573) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 19.1882 ± 0.0443648 (W 0.0268395) | 36.1042 ± 0.0770896 (W 0.0363939) | 🔵 faer-fork 1.88× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 24.4175 ± 0.0716477 (W 0.0280812) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.415569 ± 0.00226291 (W 0.00241726) | 0.652527 ± 0.00201386 (W 0.00353264) | 🔵 faer-fork 1.57× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 1.55127 ± 0.00307304 (W 0.00619506) | 3.02703 ± 0.00888725 (W 0.00496798) | 🔵 faer-fork 1.95× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 8.92657 ± 0.0252241 (W 0.0122556) | 15.016 ± 0.0252316 (W 0.024653) | 🔵 faer-fork 1.68× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.203348 ± 0.0038022 (W 0.0038062) | 0.207532 ± 0.000640914 (W 0.00163515) | · |  |
| `solve/n128/b1 [api-inclusive]` | 0.339577 ± 0.00265523 (W 0.00290317) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.894626 ± 0.00887081 (W 0.00967364) | 1.12534 ± 0.0053796 (W 0.00408217) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.57844 ± 0.0135666 (W 0.0136848) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.64663 ± 0.00974225 (W 0.0493104) | 4.99987 ± 0.0061629 (W 0.00874413) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 8.29913 ± 0.0707681 (W 0.0622827) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.70701 ± 0.0108504 (W 0.00772496) | 4.61792 ± 0.0144622 (W 0.0070201) | 🔵 faer-fork 1.71× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.71856 ± 0.00483629 (W 0.00695427) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.9224 ± 0.0233373 (W 0.0218827) | 22.4136 ± 0.152427 (W 0.0337695) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 12.9757 ± 0.0329479 (W 0.022713) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 69.1871 ± 0.175901 (W 0.0795431) | 113.796 ± 0.463822 (W 0.237901) | 🔵 faer-fork 1.64× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 69.2674 ± 0.278891 (W 0.0699352) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.2698 ± 0.00310173 (W 0.0049867) | 2.391 ± 0.0108804 (W 0.0051765) | 🔵 faer-fork 1.88× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.27291 ± 0.00723175 (W 0.00264754) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.39671 ± 0.0210701 (W 0.0144234) | 11.5841 ± 0.0430898 (W 0.0183431) | 🔵 faer-fork 1.81× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.42554 ± 0.0161104 (W 0.0154007) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 34.3335 ± 0.177129 (W 0.0468897) | 51.5094 ± 0.210495 (W 0.0782287) | 🔵 faer-fork 1.50× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 34.4243 ± 0.141537 (W 0.0445003) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009237 ± 7.65001e-05 (W 8.81732e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.022988 ± 8.08497e-05 (W 0.000167279) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0613995 ± 0.00053926 (W 0.000304179) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.129541 ± 0.00135933 (W 0.00157718) | 0.132892 ± 0.00116292 (W 0.0017874) | · |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.145141 ± 0.000942542 (W 0.00133321) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.425337 ± 0.00328656 (W 0.00312571) | 0.453484 ± 0.00131055 (W 0.00263256) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.500722 ± 0.00316794 (W 0.00268326) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.57938 ± 0.00668071 (W 0.00572003) | 2.0211 ± 0.00374804 (W 0.00668579) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.91526 ± 0.00948841 (W 0.00512364) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.3103 ± 0.00595599 (W 0.00470344) | 2.45361 ± 0.0107196 (W 0.00481877) | 🔵 faer-fork 1.87× faster than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.31488 ± 0.00637305 (W 0.00514002) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.37102 ± 0.0131694 (W 0.00918796) | 10.2178 ± 0.140097 (W 0.012754) | 🔵 faer-fork 1.90× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.39559 ± 0.0236703 (W 0.00841469) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.1506 ± 0.0401314 (W 0.0289205) | 40.2015 ± 0.140414 (W 0.0599065) | 🔵 faer-fork 1.60× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 25.2517 ± 0.0550348 (W 0.0222571) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.564511 ± 0.00372099 (W 0.00432372) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.684069 ± 0.00520793 (W 0.00576687) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.95838 ± 0.00743308 (W 0.012778) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.5336 ± 0.0138778 (W 0.011937) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 13.2687 ± 0.0742293 (W 0.0292838) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 16.1703 ± 0.051147 (W 0.029216) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.179029 ± 0.000918601 (W 0.0013647) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.180761 ± 0.0010653 (W 0.00137658) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.0185 ± 0.00265461 (W 0.00183059) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.0257 ± 0.00346737 (W 0.00211016) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 8.05111 ± 0.0809085 (W 0.0953197) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 8.0846 ± 0.0834723 (W 0.0614139) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.187063 ± 0.000557396 (W 0.0015134) | 0.201615 ± 0.000936756 (W 0.00298901) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.321698 ± 0.00313457 (W 0.0023647) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.740405 ± 0.00438616 (W 0.00421157) | 0.796909 ± 0.00280083 (W 0.00262015) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.39965 ± 0.00613371 (W 0.00471523) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.17957 ± 0.00446693 (W 0.00641127) | 2.85633 ± 0.00960343 (W 0.00783619) | 🔴 faer-fork 1.11× slower than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 6.74278 ± 0.0488652 (W 0.0152898) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.546994 ± 0.00168506 (W 0.00250342) | 1.27081 ± 0.00512523 (W 0.00522386) | 🔵 faer-fork 2.32× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.788919 ± 0.00778325 (W 0.00571203) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.0703 ± 0.00719761 (W 0.00491724) | 6.07505 ± 0.0447811 (W 0.0121319) | 🔵 faer-fork 2.93× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 3.13153 ± 0.0099742 (W 0.0111037) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.2043 ± 0.0075013 (W 0.0244234) | 27.6059 ± 0.315877 (W 0.0583685) | 🔵 faer-fork 2.71× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 15.6019 ± 0.0890232 (W 0.0296084) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.368798 ± 0.0018678 (W 0.00227963) | 0.643053 ± 0.00411519 (W 0.00292309) | 🔵 faer-fork 1.74× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 1.27318 ± 0.00303726 (W 0.0026551) | 2.7477 ± 0.0156481 (W 0.00726259) | 🔵 faer-fork 2.16× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 4.85025 ± 0.00698326 (W 0.00658546) | 11.8554 ± 0.0366597 (W 0.0122187) | 🔵 faer-fork 2.44× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.188251 ± 0.000719252 (W 0.00202572) | 0.199551 ± 0.00268883 (W 0.00211036) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.325341 ± 0.00178169 (W 0.00259116) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.704798 ± 0.00160524 (W 0.00266173) | 0.770208 ± 0.00484695 (W 0.00425638) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.40467 ± 0.0111509 (W 0.00633305) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.081 ± 0.00826232 (W 0.00494683) | 2.80615 ± 0.0128412 (W 0.00595029) | 🔴 faer-fork 1.10× slower than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 6.7361 ± 0.0588096 (W 0.0178876) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.50211 ± 0.00918625 (W 0.00665347) | 4.59239 ± 0.0287248 (W 0.0149969) | 🔵 faer-fork 1.84× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.51857 ± 0.0234097 (W 0.00721954) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.7545 ± 0.0378543 (W 0.0184771) | 20.8499 ± 0.104618 (W 0.0351406) | 🔵 faer-fork 1.94× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 10.8097 ± 0.0461006 (W 0.0222351) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.521 ± 0.0953666 (W 0.0744402) | 94.747 ± 0.95507 (W 0.206794) | 🔵 faer-fork 1.91× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 49.7801 ± 0.128793 (W 0.0525295) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.26405 ± 0.0135447 (W 0.0038893) | 2.47138 ± 0.0158163 (W 0.0114705) | 🔵 faer-fork 1.96× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.26788 ± 0.00475003 (W 0.00699593) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.76751 ± 0.0239595 (W 0.00830082) | 11.4874 ± 0.0662272 (W 0.0196603) | 🔵 faer-fork 1.99× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.79002 ± 0.023256 (W 0.00913755) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 26.4435 ± 0.0668452 (W 0.0677632) | 46.1672 ± 0.333148 (W 0.0448434) | 🔵 faer-fork 1.75× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 26.4363 ± 0.0373251 (W 0.0336413) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009317 ± 7.66649e-05 (W 6.58491e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.023493 ± 6.97636e-05 (W 9.60195e-05) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.059045 ± 0.000326463 (W 0.000455316) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0476435 ± 0.00134412 (W 0.000153298) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.04043 ± 0.000220621 (W 0.000135855) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.218648 ± 0.000968061 (W 0.00216331) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.229337 ± 0.00207835 (W 0.00314599) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.38655 ± 0.00583407 (W 0.00517041) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.28355 ± 0.0043382 (W 0.00317561) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.7556 ± 0.00692324 (W 0.00503508) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.74305 ± 0.0043092 (W 0.00432961) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.58417 ± 0.0184638 (W 0.00997757) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.59725 ± 0.0266018 (W 0.0141256) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.5855 ± 0.174458 (W 0.0355047) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.5613 ± 0.17444 (W 0.0373193) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.267614 ± 0.0021543 (W 0.00254939) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.238189 ± 0.00137101 (W 0.0020482) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.4681 ± 0.0074863 (W 0.00983843) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.46392 ± 0.00834115 (W 0.00719826) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.2982 ± 0.0913147 (W 0.0216718) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.2622 ± 0.431358 (W 0.0371174) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.148201 ± 0.000864784 (W 0.000191811) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.147831 ± 0.000465601 (W 0.000662751) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00289 ± 0.00729813 (W 0.00431008) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00308 ± 0.00665537 (W 0.00323147) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.45956 ± 0.0288371 (W 0.0119242) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.46799 ± 0.0295719 (W 0.0184368) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0764675 ± 0.000236931 (W 0.000988915) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.094711 ± 0.000440871 (W 0.000375803) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.367175 ± 0.00173318 (W 0.00326592) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.567157 ± 0.00311157 (W 0.00366396) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.12389 ± 0.00585527 (W 0.0051938) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.19037 ± 0.0323228 (W 0.0107162) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.354966 ± 0.00215001 (W 0.0025818) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.456512 ± 0.0034548 (W 0.00304906) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.82603 ± 0.00767393 (W 0.00332659) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.93094 ± 0.0109686 (W 0.00751845) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.5935 ± 0.0284936 (W 0.0270206) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 11.0397 ± 0.0413924 (W 0.0189844) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.188301 ± 0.00158687 (W 0.00171772) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.04581 ± 0.00503218 (W 0.00265409) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.2564 ± 0.0435478 (W 0.0105795) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.068768 ± 0.000839784 (W 0.000339116) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.095162 ± 0.0008422 (W 0.000246638) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.334734 ± 0.00190777 (W 0.00244254) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.444214 ± 0.00287497 (W 0.00252267) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.98641 ± 0.00742546 (W 0.00446513) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.60831 ± 0.0427368 (W 0.0166716) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.15787 ± 0.0114017 (W 0.00543196) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.16453 ± 0.0207797 (W 0.00553958) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.25846 ± 0.0270043 (W 0.0174284) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.31325 ± 0.0286588 (W 0.0193388) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 50.204 ± 0.304681 (W 0.0826246) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 50.3214 ± 0.186755 (W 0.0632946) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.660822 ± 0.00287741 (W 0.00278545) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.659619 ± 0.00357191 (W 0.00317137) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.35864 ± 0.0149246 (W 0.0078385) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.3692 ± 0.0185088 (W 0.00879126) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 20.2422 ± 0.123401 (W 0.0197903) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 20.2231 ± 0.0983013 (W 0.0298807) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.003476 ± 1.51493e-05 (W 4.3528e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0088865 ± 6.3626e-05 (W 6.30264e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.025142 ± 5.89915e-05 (W 0.000138383) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0439165 ± 0.000134464 (W 0.000176933) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.0377 ± 8.77876e-05 (W 0.000135528) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.14247 ± 0.000794107 (W 0.00216941) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.170778 ± 0.000968518 (W 0.00258607) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.648057 ± 0.00569157 (W 0.0060519) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.663424 ± 0.0114873 (W 0.00364434) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.04579 ± 0.00304455 (W 0.00304555) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.04574 ± 0.00229046 (W 0.00217801) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.72596 ± 0.0109155 (W 0.00889618) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.7286 ± 0.00844004 (W 0.00789438) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.1988 ± 0.0548128 (W 0.0313788) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.124 ± 0.0302408 (W 0.016817) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.23383 ± 0.000735849 (W 0.00161828) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.24211 ± 0.000944103 (W 0.00242623) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.13583 ± 0.00880651 (W 0.00528577) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.25955 ± 0.00846212 (W 0.00508037) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.37116 ± 0.0168329 (W 0.0174802) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 8.06816 ± 0.0259029 (W 0.0174196) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.051521 ± 0.000218626 (W 0.000284001) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.053044 ± 0.000385178 (W 0.000703461) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.277757 ± 0.00219421 (W 0.00232285) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.280292 ± 0.00141191 (W 0.00170338) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.94786 ± 0.0045498 (W 0.00469727) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.93981 ± 0.00787252 (W 0.00380154) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0872725 ± 0.00101841 (W 0.00112074) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.086692 ± 0.000704527 (W 0.000948209) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.296185 ± 0.0022174 (W 0.00334516) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.445238 ± 0.00291881 (W 0.00385431) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.2617 ± 0.0116748 (W 0.00920811) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.97735 ± 0.0229149 (W 0.00828704) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.318012 ± 0.00223482 (W 0.00307521) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.33051 ± 0.00162399 (W 0.00259769) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.772957 ± 0.00332021 (W 0.00237848) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.05835 ± 0.0058007 (W 0.00414002) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.67587 ± 0.0147421 (W 0.00710658) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.84197 ± 0.0137624 (W 0.0171583) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.198084 ± 0.00170123 (W 0.00178777) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.544744 ± 0.00479289 (W 0.00271123) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.24162 ± 0.00861359 (W 0.00639413) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0780955 ± 0.00118829 (W 0.0013423) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0925325 ± 0.000568844 (W 0.000586469) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.298671 ± 0.000903977 (W 0.00267711) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.457381 ± 0.00119439 (W 0.00284469) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.193 ± 0.00483703 (W 0.0034509) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.95479 ± 0.00581346 (W 0.00528642) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.51525 ± 0.0112125 (W 0.00423782) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.51814 ± 0.00198642 (W 0.00334477) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.83638 ± 0.0128865 (W 0.00963973) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.85232 ± 0.026484 (W 0.012528) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.7905 ± 0.0507813 (W 0.0354418) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 24.8314 ± 0.0475005 (W 0.0485427) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.693785 ± 0.00262754 (W 0.00244503) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.694233 ± 0.00316552 (W 0.00225183) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.01173 ± 0.00587137 (W 0.0110981) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.02686 ± 0.00861062 (W 0.00827011) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 12.3691 ± 0.0541117 (W 0.0210383) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 12.4312 ± 0.0408204 (W 0.0313093) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004634 ± 4.36654e-05 (W 5.37494e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.011256 ± 7.70552e-05 (W 7.24037e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.026529 ± 0.000212315 (W 0.000183133) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.042985 ± 0.00145356 (W 0.0001231) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.037475 ± 0.00129735 (W 0.00037572) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.139565 ± 0.000846425 (W 0.00120438) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.165724 ± 0.00160187 (W 0.002427) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.568569 ± 0.00515761 (W 0.00506797) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.571749 ± 0.00257891 (W 0.00269798) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.743453 ± 0.00496724 (W 0.00278078) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.744753 ± 0.00251613 (W 0.0018205) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.76833 ± 0.0135042 (W 0.00517933) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.77337 ± 0.0100189 (W 0.00641846) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.7503 ± 0.0515789 (W 0.015402) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.7102 ± 0.0760814 (W 0.014882) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.237682 ± 0.000712767 (W 0.00192862) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.252406 ± 0.00137879 (W 0.00175811) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.08807 ± 0.0130375 (W 0.00768076) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.26626 ± 0.0109717 (W 0.00755917) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.76163 ± 0.0408389 (W 0.0170921) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.55973 ± 0.0442769 (W 0.0137582) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.040371 ± 0.00061351 (W 0.00134443) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0411415 ± 0.00097209 (W 0.0011511) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.162008 ± 0.000990983 (W 0.00210092) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.163374 ± 0.000532571 (W 0.00180354) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.04363 ± 0.00615878 (W 0.00602699) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.04301 ± 0.00200262 (W 0.00565233) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0970455 ± 0.000874451 (W 0.0017096) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.086907 ± 0.000531879 (W 0.000583743) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.251513 ± 0.006333 (W 0.00203695) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.434018 ± 0.00274346 (W 0.0040065) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.965547 ± 0.00388808 (W 0.00420771) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.78265 ± 0.0105977 (W 0.00866324) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.273003 ± 0.00149416 (W 0.00188878) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.31717 ± 0.0023365 (W 0.0032427) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.667459 ± 0.00328362 (W 0.00268214) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.980696 ± 0.0129014 (W 0.00643441) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.27552 ± 0.00905522 (W 0.00651612) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.63079 ± 0.0137855 (W 0.013497) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.186502 ± 0.00248156 (W 0.00179987) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.501193 ± 0.00464928 (W 0.0031738) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.46235 ± 0.0135021 (W 0.00509647) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0796435 ± 0.000925461 (W 0.00104553) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.092477 ± 0.000500899 (W 0.000659388) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.254309 ± 0.00112199 (W 0.0022828) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.442113 ± 0.00940637 (W 0.00284913) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 0.915866 ± 0.00520368 (W 0.00291889) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.77286 ± 0.0109363 (W 0.00568626) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.49727 ± 0.0037879 (W 0.00430173) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.49987 ± 0.01011 (W 0.00452628) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.43884 ± 0.0112413 (W 0.0137016) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.46882 ± 0.0301023 (W 0.0173997) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 21.4785 ± 0.0765537 (W 0.032298) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 21.6045 ± 0.0737583 (W 0.0355481) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.710103 ± 0.00177471 (W 0.00256223) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.706269 ± 0.00277396 (W 0.00234066) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.95981 ± 0.0150989 (W 0.00604566) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.9733 ± 0.00969289 (W 0.00809816) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.3631 ± 0.0478706 (W 0.017323) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.3685 ± 0.0405461 (W 0.0245297) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004609 ± 4.77232e-05 (W 4.54879e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.011261 ± 9.24395e-05 (W 6.57791e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.025813 ± 0.000293104 (W 0.00010785) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.034404 ± 0.000152263 (W 0.000846487) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.039904 ± 0.000183585 (W 6.51276e-05) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.176854 ± 0.000758273 (W 0.00197952) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.195023 ± 0.000691227 (W 0.00171101) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.08695 ± 0.00517404 (W 0.00453025) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.15693 ± 0.00383555 (W 0.00323485) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.74999 ± 0.0087244 (W 0.00479617) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.7435 ± 0.0078842 (W 0.00548264) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.52658 ± 0.0149658 (W 0.0123715) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.55969 ± 0.0246845 (W 0.0115202) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.3494 ± 0.0657149 (W 0.0336125) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.4697 ± 0.118756 (W 0.0554435) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.211339 ± 0.00124 (W 0.00208649) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.216724 ± 0.00158767 (W 0.0017669) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.33095 ± 0.0633243 (W 0.00808895) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.38377 ± 0.0111371 (W 0.00826261) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.255 ± 0.357948 (W 0.0167583) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.9655 ± 0.0996907 (W 0.0417725) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.152229 ± 0.000883323 (W 0.00217438) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.150009 ± 0.000670734 (W 0.00165252) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.01098 ± 0.00568269 (W 0.00235808) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01364 ± 0.00340593 (W 0.00357434) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.53591 ± 0.0147065 (W 0.0174589) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.56921 ± 0.0290649 (W 0.0222758) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.071889 ± 0.000432445 (W 0.00156599) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.077104 ± 0.000670925 (W 0.000581181) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.358417 ± 0.00205571 (W 0.0026314) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.485426 ± 0.0040945 (W 0.00447447) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.22978 ± 0.00833684 (W 0.00705374) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.83685 ± 0.0156312 (W 0.00714023) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.39702 ± 0.0025071 (W 0.00229161) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.428975 ± 0.00421581 (W 0.00376825) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.60526 ± 0.0164621 (W 0.00419498) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.81469 ± 0.0089774 (W 0.00660021) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 9.67596 ± 0.023806 (W 0.0125508) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 10.5294 ± 0.0317577 (W 0.0185187) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.218107 ± 0.0011863 (W 0.00193247) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.842541 ± 0.0059464 (W 0.00208799) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 4.37235 ± 0.0284015 (W 0.0141071) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.068227 ± 0.000513187 (W 0.00179757) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0738125 ± 0.000676783 (W 0.000208535) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.345453 ± 0.00165087 (W 0.00304105) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.35966 ± 0.00154437 (W 0.00318008) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.1698 ± 0.00906711 (W 0.0048723) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.25236 ± 0.00779278 (W 0.00723647) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.16459 ± 0.011789 (W 0.00465037) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.15723 ± 0.00908475 (W 0.00831798) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.24371 ± 0.0338733 (W 0.00987116) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.30549 ± 0.0237232 (W 0.0161876) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.743 ± 0.136314 (W 0.0644681) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.8577 ± 0.207601 (W 0.0551583) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.656133 ± 0.00461382 (W 0.00277622) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.654199 ± 0.00297167 (W 0.00365696) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.26473 ± 0.0107701 (W 0.0121471) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.26954 ± 0.0106494 (W 0.00802292) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.7535 ± 0.0287943 (W 0.036838) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.7585 ± 0.111259 (W 0.0181185) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.003957 ± 2.54853e-05 (W 4.91022e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.010164 ± 6.121e-05 (W 7.38749e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0274315 ± 0.000159404 (W 0.000130432) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0304765 ± 0.00161274 (W 0.000193814) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.0369335 ± 0.00138205 (W 0.000248499) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.106398 ± 0.000550949 (W 0.000418618) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.135347 ± 0.000871395 (W 0.000660789) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.415567 ± 0.00157594 (W 0.00241321) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.533959 ± 0.0035847 (W 0.00248373) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.04527 ± 0.00441148 (W 0.00280131) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.04677 ± 0.00134096 (W 0.00355043) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.70213 ± 0.0106389 (W 0.0109126) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.69756 ± 0.0119244 (W 0.00708957) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.0474 ± 0.0590404 (W 0.0158369) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 14.9939 ± 0.0290937 (W 0.0157642) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.223064 ± 0.00160121 (W 0.00292876) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.222519 ± 0.000927355 (W 0.00161776) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.07733 ± 0.0034095 (W 0.00508452) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.18189 ± 0.00906919 (W 0.00543023) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.41885 ± 0.021845 (W 0.0181759) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.75311 ± 0.0231109 (W 0.0185911) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0539255 ± 0.000339567 (W 0.000367987) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.05363 ± 0.000832257 (W 0.000639152) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.288 ± 0.00127581 (W 0.0025038) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.290104 ± 0.00239146 (W 0.00244413) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.96553 ± 0.00346845 (W 0.00366844) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.96594 ± 0.00580799 (W 0.00336535) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0598115 ± 0.000760256 (W 0.00172277) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0670345 ± 0.00102823 (W 0.000761317) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.220156 ± 0.00146676 (W 0.0032591) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.3626 ± 0.00249154 (W 0.00326677) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.986006 ± 0.00501424 (W 0.00530098) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.6136 ± 0.0255367 (W 0.00687036) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.262042 ± 0.00196512 (W 0.0020645) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.301095 ± 0.00272012 (W 0.00315192) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.698959 ± 0.0036333 (W 0.0023956) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.937029 ± 0.00384389 (W 0.00373236) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.37616 ± 0.0132504 (W 0.0104296) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.35607 ± 0.0124578 (W 0.00918951) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.178401 ± 0.000845608 (W 0.00190571) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.462512 ± 0.00385392 (W 0.00249747) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.97633 ± 0.00923366 (W 0.00294181) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0638195 ± 0.000613499 (W 0.000438226) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.070356 ± 0.000632313 (W 0.000923831) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.228214 ± 0.00183691 (W 0.00220988) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.366092 ± 0.00300381 (W 0.00341798) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.969465 ± 0.00555309 (W 0.00519982) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.60469 ± 0.00554769 (W 0.00759805) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.51312 ± 0.0038253 (W 0.00324316) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.51817 ± 0.00898633 (W 0.00470295) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.77498 ± 0.0157123 (W 0.0128288) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.80591 ± 0.0266445 (W 0.0104364) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.2498 ± 0.0363306 (W 0.0243746) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 24.3462 ± 0.0305991 (W 0.0348222) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.69053 ± 0.00329202 (W 0.00219279) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.688348 ± 0.00224314 (W 0.00232125) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.9191 ± 0.0103174 (W 0.00695443) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.9256 ± 0.00724808 (W 0.00596621) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.8697 ± 0.0381658 (W 0.014651) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.8913 ± 0.0321269 (W 0.0233696) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00505 ± 6.65703e-05 (W 5.83038e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0124885 ± 4.40019e-05 (W 6.34299e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0317645 ± 0.000482368 (W 0.000605303) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.030056 ± 0.00114699 (W 0.000159434) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.0368535 ± 0.00106938 (W 0.000230219) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.108302 ± 0.00233739 (W 0.00174416) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.130123 ± 0.0023515 (W 0.000516711) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.346535 ± 0.00118488 (W 0.00310663) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.446937 ± 0.0018487 (W 0.00246203) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.744065 ± 0.00356403 (W 0.00237769) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.745092 ± 0.00348804 (W 0.00325874) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.75243 ± 0.028832 (W 0.00951986) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.74997 ± 0.00795758 (W 0.00603691) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.5333 ± 0.0448423 (W 0.0136535) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.5569 ± 0.0223355 (W 0.00897933) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.232077 ± 0.00461124 (W 0.00218232) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.231326 ± 0.000960067 (W 0.00178252) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.07153 ± 0.0109117 (W 0.00541979) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.18467 ± 0.00692824 (W 0.00535086) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.77321 ± 0.107147 (W 0.0135656) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.25311 ± 0.0262373 (W 0.0251594) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0362325 ± 0.000444082 (W 0.000653363) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0383165 ± 0.000714526 (W 0.000911271) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.170777 ± 0.00206005 (W 0.00229182) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.172208 ± 0.00156728 (W 0.00163308) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.06596 ± 0.00563116 (W 0.0058289) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.07221 ± 0.00444707 (W 0.00508315) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.059746 ± 0.000758114 (W 0.000645786) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.066985 ± 0.000724748 (W 0.000712034) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.20939 ± 0.00250247 (W 0.00293792) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.348745 ± 0.00303306 (W 0.00314909) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.835852 ± 0.00590155 (W 0.00326625) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.43959 ± 0.00826614 (W 0.00715255) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.247751 ± 0.0018483 (W 0.00176022) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.286764 ± 0.0013382 (W 0.00316221) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.616489 ± 0.00288647 (W 0.0037085) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.86399 ± 0.00543952 (W 0.00402428) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.13207 ± 0.0120049 (W 0.00479241) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.14141 ± 0.00881874 (W 0.0107919) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.173503 ± 0.00149196 (W 0.00122873) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.462837 ± 0.00427581 (W 0.00322759) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.33207 ± 0.00870624 (W 0.00417931) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0635585 ± 0.000837975 (W 0.000466723) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.070627 ± 0.000625822 (W 0.000376475) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.212361 ± 0.00142718 (W 0.0019721) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.356047 ± 0.00300759 (W 0.00250058) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.774814 ± 0.00526227 (W 0.00307673) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.43172 ± 0.00946279 (W 0.00636039) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.49767 ± 0.00635584 (W 0.00258981) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.50299 ± 0.0038838 (W 0.0040471) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.39024 ± 0.0221498 (W 0.011191) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.40888 ± 0.0103718 (W 0.0069078) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 20.9819 ± 0.0717623 (W 0.0223231) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 21.0699 ± 0.0245776 (W 0.0313037) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.706947 ± 0.00216712 (W 0.00227468) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.705693 ± 0.00329332 (W 0.00192403) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.86573 ± 0.0123255 (W 0.00553354) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.87514 ± 0.0115051 (W 0.00694083) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 10.9054 ± 0.0546085 (W 0.0138457) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 10.9243 ± 0.045646 (W 0.0196046) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0050645 ± 4.88085e-05 (W 6.22193e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013029 ± 0.000486753 (W 9.40553e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0310275 ± 0.000327722 (W 0.00024894) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.213788 ± 0.000718739 (W 0.00216174) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.210376 ± 0.00103954 (W 0.00165594) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.914092 ± 0.00371102 (W 0.00381318) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.881394 ± 0.00241913 (W 0.0032781) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.99616 ± 0.0134943 (W 0.00806466) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.71988 ± 0.0255428 (W 0.0103517) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.30565 ± 0.00984152 (W 0.00395951) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.30242 ± 0.00459364 (W 0.00510472) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.9039 ± 0.0357644 (W 0.018723) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.9062 ± 0.0195185 (W 0.0182389) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.8111 ± 0.127426 (W 0.0779265) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.9068 ± 0.168299 (W 0.0819499) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.457838 ± 0.0190086 (W 0.00383781) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.43188 ± 0.00369514 (W 0.00316663) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.58828 ± 0.0150334 (W 0.00748704) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.73458 ± 0.00916723 (W 0.0115762) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.4173 ± 0.175166 (W 0.0401532) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 26.2791 ± 0.177337 (W 0.0977356) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.564686 ± 0.00156366 (W 0.00260944) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.566448 ± 0.00202221 (W 0.00340878) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.89198 ± 0.00725536 (W 0.01229) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.89508 ± 0.00925532 (W 0.00810854) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 29.0385 ± 0.0713457 (W 0.0573566) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 29.0166 ± 0.0605482 (W 0.0454029) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.182585 ± 0.0010458 (W 0.0018572) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.251053 ± 0.00308511 (W 0.00241464) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.06282 ± 0.00516886 (W 0.00217927) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.47604 ± 0.00615912 (W 0.00990167) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.16185 ± 0.0189693 (W 0.00774829) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 9.25057 ± 0.0502174 (W 0.0208369) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.875425 ± 0.00328265 (W 0.0020959) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.992414 ± 0.00396323 (W 0.00334265) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.54846 ± 0.0130004 (W 0.0137597) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.85031 ± 0.0104534 (W 0.00934058) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 36.554 ± 0.0865901 (W 0.259764) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 38.1345 ± 0.0950724 (W 0.0535115) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.428187 ± 0.00436894 (W 0.00271718) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.6522 ± 0.00562508 (W 0.0071189) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 15.967 ± 0.04852 (W 0.0419963) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.176915 ± 0.00118732 (W 0.0018689) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.210792 ± 0.001633 (W 0.0015681) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.03348 ± 0.00248293 (W 0.00217225) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.21002 ± 0.00348861 (W 0.00645179) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 7.04615 ± 0.0166389 (W 0.00985924) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.77615 ± 0.0215979 (W 0.0292501) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.13335 ± 0.0115447 (W 0.00745006) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.15874 ± 0.019275 (W 0.0048927) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 16.8516 ± 0.0708355 (W 0.0321279) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 16.8603 ± 0.0437571 (W 0.0203636) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 105.805 ± 0.234599 (W 0.0916776) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 106.077 ± 0.176154 (W 0.133648) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.0645 ± 0.00366129 (W 0.00243895) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.06329 ± 0.004931 (W 0.0030163) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.32337 ± 0.0428696 (W 0.0189198) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.29681 ± 0.0186669 (W 0.0185151) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 44.2526 ± 0.306297 (W 0.0773712) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 44.1974 ± 0.379328 (W 0.11724) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.006211 ± 2.88958e-05 (W 7.57236e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017022 ± 0.000105198 (W 6.49675e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.054606 ± 0.000706593 (W 0.000953543) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.192559 ± 0.00022765 (W 0.0019311) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.195279 ± 0.000455962 (W 0.00191969) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.604546 ± 0.00156368 (W 0.00335201) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.616097 ± 0.0014736 (W 0.00424205) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.0877 ± 0.0064278 (W 0.00349344) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.14441 ± 0.00642359 (W 0.00541473) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.33141 ± 0.00529585 (W 0.00563611) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.32982 ± 0.00351158 (W 0.00424718) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.28681 ± 0.0116282 (W 0.00583673) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.29468 ± 0.012815 (W 0.00809626) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.5651 ± 0.0372748 (W 0.0332386) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.5373 ± 0.0528552 (W 0.0197942) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.369351 ± 0.00227491 (W 0.00406531) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.381811 ± 0.00156351 (W 0.00276191) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.17926 ± 0.00862465 (W 0.00701377) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.48371 ± 0.0081784 (W 0.00673447) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.5891 ± 0.0555252 (W 0.0233237) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 14.2273 ± 0.0299312 (W 0.022851) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.159893 ± 0.000383958 (W 0.00164764) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.160429 ± 0.000359157 (W 0.000809325) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00828 ± 0.00193443 (W 0.00176651) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01137 ± 0.00250394 (W 0.00302032) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.37184 ± 0.0119686 (W 0.0145218) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.36795 ± 0.0127833 (W 0.0183177) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.174129 ± 0.00166289 (W 0.00204799) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.218145 ± 0.00150654 (W 0.00201258) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.651396 ± 0.00351277 (W 0.004576) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.97742 ± 0.00525389 (W 0.00525739) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.12164 ± 0.00500422 (W 0.00886424) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.87242 ± 0.0634317 (W 0.018184) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.483434 ± 0.00264706 (W 0.00300061) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.568893 ± 0.00166883 (W 0.00371995) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.80043 ± 0.00594113 (W 0.00459993) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.29651 ± 0.00619274 (W 0.00739854) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.7046 ± 0.0450856 (W 0.0152655) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 13.1618 ± 0.0651051 (W 0.0381734) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.306997 ± 0.00147074 (W 0.00186451) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.01 ± 0.00385238 (W 0.00207549) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.326 ± 0.0160634 (W 0.00948967) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.150515 ± 0.00108297 (W 0.00116821) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.182575 ± 0.000928277 (W 0.00142968) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.641379 ± 0.00446309 (W 0.00215359) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.974606 ± 0.00286974 (W 0.00406923) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.9509 ± 0.0119515 (W 0.00999711) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.77452 ± 0.0218062 (W 0.020132) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.95618 ± 0.0102299 (W 0.00793407) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.97224 ± 0.00612771 (W 0.00497773) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.26684 ± 0.0217131 (W 0.0133132) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.32031 ± 0.0256007 (W 0.00921467) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 40.3774 ± 0.103741 (W 0.0361243) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 40.5728 ± 0.0948258 (W 0.0482529) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.950614 ± 0.00505763 (W 0.00383913) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.947607 ± 0.00490575 (W 0.00284612) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.09062 ± 0.0178131 (W 0.00824271) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.12003 ± 0.0103861 (W 0.00635925) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.4246 ± 0.0462116 (W 0.0306655) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.4791 ± 0.0325944 (W 0.0405492) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007319 ± 5.32932e-05 (W 3.25616e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017238 ± 9.48629e-05 (W 6.29319e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0406955 ± 0.000377811 (W 0.000181745) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.192168 ± 0.000616444 (W 0.00187908) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.193452 ± 0.000417655 (W 0.00162594) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.578879 ± 0.0028483 (W 0.00466912) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.585727 ± 0.00170917 (W 0.00291727) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.72337 ± 0.00491691 (W 0.00632432) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.80947 ± 0.0080383 (W 0.00801638) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.983058 ± 0.0022064 (W 0.00209327) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.987926 ± 0.00272824 (W 0.00209198) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.95955 ± 0.00909624 (W 0.00601353) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.97562 ± 0.0132203 (W 0.00471525) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.6664 ± 0.028581 (W 0.0210013) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.6832 ± 0.025001 (W 0.025135) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.372288 ± 0.00197437 (W 0.00338066) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.392172 ± 0.00189701 (W 0.00279174) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.03275 ± 0.00798827 (W 0.00765063) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.36614 ± 0.0102185 (W 0.0103206) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.8144 ± 0.0294025 (W 0.0244124) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.6123 ± 0.0416454 (W 0.0173247) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.102466 ± 0.000762801 (W 0.000985718) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.102561 ± 0.000704979 (W 0.00126461) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.526221 ± 0.00363622 (W 0.00250085) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.531327 ± 0.00106537 (W 0.00201198) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.86237 ± 0.00427139 (W 0.0085052) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.85912 ± 0.00545407 (W 0.00937292) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.180768 ± 0.00125779 (W 0.00163826) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.213763 ± 0.000871509 (W 0.00196306) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.551337 ± 0.00196497 (W 0.00384129) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.905277 ± 0.00393498 (W 0.00422071) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.31406 ± 0.00561865 (W 0.00497949) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.16158 ± 0.030054 (W 0.0195647) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.41294 ± 0.00216734 (W 0.00275996) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.514583 ± 0.00485618 (W 0.00315972) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.32276 ± 0.00212437 (W 0.00274335) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.87758 ± 0.0514582 (W 0.00782083) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 5.81047 ± 0.0159242 (W 0.0163053) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.56998 ± 0.0510944 (W 0.023348) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.274862 ± 0.000696591 (W 0.00207406) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.887336 ± 0.000926133 (W 0.00199188) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.99155 ± 0.00387181 (W 0.00567712) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.140519 ± 0.000923754 (W 0.00155052) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.176764 ± 0.00120215 (W 0.00106451) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.537636 ± 0.00276309 (W 0.00218597) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.9181 ± 0.00844683 (W 0.00312616) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.12396 ± 0.00619293 (W 0.00443097) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.07813 ± 0.0228148 (W 0.015057) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.89191 ± 0.00977112 (W 0.00554689) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.89972 ± 0.00557231 (W 0.00332555) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.42323 ± 0.107103 (W 0.0167897) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.47844 ± 0.0241136 (W 0.0138522) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 31.65 ± 0.0675579 (W 0.0394075) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 31.8742 ± 0.0796758 (W 0.0334645) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.977515 ± 0.00441076 (W 0.00306593) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.974963 ± 0.0041561 (W 0.00309213) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.00774 ± 0.0170433 (W 0.0103343) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.01623 ± 0.0224505 (W 0.0089691) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.8058 ± 0.0519421 (W 0.0224684) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.8276 ± 0.0585015 (W 0.0173676) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007409 ± 4.36654e-05 (W 3.5162e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.016912 ± 0.0001185 (W 8.8623e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0384775 ± 0.000222235 (W 0.0001886) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.199551 ± 0.000860823 (W 0.00167071) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.204241 ± 0.000440156 (W 0.00145614) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.815685 ± 0.00261191 (W 0.00258359) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.838723 ± 0.00257504 (W 0.00312048) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.45458 ± 0.0104752 (W 0.0096602) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.53191 ± 0.0220587 (W 0.0067731) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.30204 ± 0.00676526 (W 0.00495396) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.30445 ± 0.0041907 (W 0.00515064) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.8709 ± 0.0334604 (W 0.0148035) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.8792 ± 0.0182595 (W 0.013684) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.7072 ± 0.0953921 (W 0.100021) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.8417 ± 0.185489 (W 0.0786159) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.404239 ± 0.00737848 (W 0.00256076) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.403862 ± 0.00221239 (W 0.00287201) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.39698 ± 0.0169497 (W 0.0110279) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.60096 ± 0.0114153 (W 0.00884656) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.2245 ± 0.730697 (W 0.0371712) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 25.5255 ± 0.122912 (W 0.0370243) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.562685 ± 0.00270875 (W 0.00245403) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.559202 ± 0.00216223 (W 0.00300822) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.86021 ± 0.0135355 (W 0.0100107) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.86571 ± 0.00445821 (W 0.0134125) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 28.9169 ± 0.0704969 (W 0.044778) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 28.8803 ± 0.0764549 (W 0.0617102) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.176623 ± 0.000339319 (W 0.00189349) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.227083 ± 0.00215607 (W 0.002415) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.07215 ± 0.00177046 (W 0.00382233) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.3662 ± 0.00682771 (W 0.00508559) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.23108 ± 0.0135238 (W 0.0103361) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 8.66225 ± 0.0289905 (W 0.0184412) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.887364 ± 0.00416789 (W 0.0027735) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.966643 ± 0.00441866 (W 0.00353132) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.27924 ± 0.0106575 (W 0.00779551) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.73854 ± 0.0107075 (W 0.00847247) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 35.5752 ± 0.0848252 (W 0.0463193) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 37.5213 ± 0.217506 (W 0.0966105) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.437656 ± 0.0010889 (W 0.00234698) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.38507 ± 0.0117977 (W 0.00290795) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 14.9671 ± 0.034399 (W 0.0195804) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.175216 ± 0.000607425 (W 0.00188393) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.182946 ± 0.000606044 (W 0.00184672) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.05865 ± 0.00247688 (W 0.00252202) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.09891 ± 0.00167193 (W 0.00342571) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 7.1283 ± 0.0118859 (W 0.0131353) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.28247 ± 0.0145089 (W 0.0116407) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.1297 ± 0.0403029 (W 0.00672424) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.14798 ± 0.0245671 (W 0.00645472) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 16.7651 ± 0.0677211 (W 0.0352108) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 16.7719 ± 0.0518681 (W 0.0258216) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 104.825 ± 0.252604 (W 0.0879608) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 105.02 ± 0.58425 (W 0.128797) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.04099 ± 0.00470526 (W 0.0028392) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.0454 ± 0.00434474 (W 0.00149072) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.1559 ± 0.034529 (W 0.0317947) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.18362 ± 0.033048 (W 0.0182479) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 43.0708 ± 0.180484 (W 0.22674) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 43.2012 ± 0.26448 (W 0.116302) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0060765 ± 4.38516e-05 (W 5.56208e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015524 ± 0.000178744 (W 5.96822e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.051431 ± 0.000504481 (W 0.000291261) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.180456 ± 0.000469999 (W 0.0018746) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188686 ± 0.000638112 (W 0.00165858) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.530112 ± 0.00191195 (W 0.00252478) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.571439 ± 0.00196113 (W 0.0025553) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.77727 ± 0.00514801 (W 0.00322143) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.96554 ± 0.00611898 (W 0.00425842) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32714 ± 0.006383 (W 0.00313536) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.32943 ± 0.00620995 (W 0.00270172) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.24292 ± 0.0115298 (W 0.00881121) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.24816 ± 0.00815787 (W 0.00903115) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.3752 ± 0.0396401 (W 0.0221413) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.4176 ± 0.0440221 (W 0.0383195) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.353598 ± 0.00645053 (W 0.00325411) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.356312 ± 0.00242365 (W 0.00358794) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.11455 ± 0.0111713 (W 0.00814342) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.34971 ± 0.00973263 (W 0.0105759) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.5707 ± 0.0392689 (W 0.0218185) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.6677 ± 0.0294573 (W 0.0293118) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.159041 ± 0.00076778 (W 0.00197974) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.159662 ± 0.000486432 (W 0.00181154) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.01074 ± 0.00491684 (W 0.00272081) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.0108 ± 0.00289579 (W 0.00147732) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.37354 ± 0.0137626 (W 0.0138573) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.38107 ± 0.0169447 (W 0.0107249) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.135873 ± 0.00236962 (W 0.001092) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.193024 ± 0.00073598 (W 0.00227769) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.5696 ± 0.00260052 (W 0.00313394) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.864345 ± 0.00477813 (W 0.00414016) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.81668 ± 0.00642975 (W 0.00673939) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.27353 ± 0.0141234 (W 0.0171668) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.444162 ± 0.000905983 (W 0.00280585) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.541609 ± 0.00120288 (W 0.00288787) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.69876 ± 0.00632433 (W 0.00459413) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.18849 ± 0.00642588 (W 0.00711568) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.2865 ± 0.0323883 (W 0.0149622) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 12.4815 ± 0.0445477 (W 0.0218412) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.283547 ± 0.00189323 (W 0.00192323) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.915444 ± 0.00467697 (W 0.00264584) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.01567 ± 0.0136168 (W 0.0111835) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.140717 ± 0.00245988 (W 0.000853141) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.15566 ± 0.000640451 (W 0.000906341) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.556917 ± 0.00192428 (W 0.00234757) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.864436 ± 0.00359524 (W 0.00546118) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.68203 ± 0.00494239 (W 0.00476811) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.16797 ± 0.0205589 (W 0.00853152) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.95427 ± 0.00713195 (W 0.00544805) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.96285 ± 0.0077383 (W 0.00535827) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.20267 ± 0.0215324 (W 0.0148786) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.23859 ± 0.015496 (W 0.0190101) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 39.5182 ± 0.142524 (W 0.0413412) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 39.6465 ± 0.0929485 (W 0.0299367) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.925604 ± 0.00163104 (W 0.00359969) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.929526 ± 0.00669963 (W 0.00288735) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.9559 ± 0.021332 (W 0.00637817) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.98283 ± 0.0123925 (W 0.0104948) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.5127 ± 0.0288421 (W 0.0276935) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.6098 ± 0.0783656 (W 0.0471067) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007323 ± 2.86176e-05 (W 4.29336e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017838 ± 0.000131636 (W 0.000213221) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.042549 ± 0.000112399 (W 0.000424443) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.179486 ± 0.000505526 (W 0.00144025) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.187389 ± 0.000481435 (W 0.00171285) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.507891 ± 0.00302987 (W 0.00321764) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.541319 ± 0.00325466 (W 0.00333336) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.46694 ± 0.00427131 (W 0.00398574) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.6292 ± 0.00636019 (W 0.00340224) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.977726 ± 0.00511794 (W 0.00262577) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.988063 ± 0.00796854 (W 0.00214721) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.92491 ± 0.0101353 (W 0.00736518) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.9359 ± 0.00726182 (W 0.00636592) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.4634 ± 0.0198285 (W 0.0211195) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.5215 ± 0.0386346 (W 0.0434965) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.359335 ± 0.00495442 (W 0.00349212) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.364609 ± 0.00175489 (W 0.00302933) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.98565 ± 0.0118119 (W 0.00716638) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.22206 ± 0.00627923 (W 0.0101263) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.7906 ± 0.0278026 (W 0.0225394) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.0665 ± 0.0201684 (W 0.0210037) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.100689 ± 0.000653747 (W 0.000814532) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.101584 ± 0.000779285 (W 0.000649263) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.536225 ± 0.00381104 (W 0.00227958) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.539324 ± 0.00217917 (W 0.00218654) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.85742 ± 0.00543207 (W 0.0168957) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.85755 ± 0.00538275 (W 0.0126859) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.132142 ± 0.00072761 (W 0.000692731) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.188716 ± 0.00156212 (W 0.0021114) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.500386 ± 0.00268508 (W 0.00242294) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.792912 ± 0.00345677 (W 0.00415765) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.11854 ± 0.00507939 (W 0.00480784) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.56859 ± 0.0207973 (W 0.00863143) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.374302 ± 0.00202204 (W 0.00262936) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.484196 ± 0.00255116 (W 0.00513974) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.26211 ± 0.00151445 (W 0.00247211) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.76655 ± 0.00799406 (W 0.00721233) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.59208 ± 0.0139669 (W 0.0152378) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 7.90874 ± 0.00909867 (W 0.0260303) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.258016 ± 0.000724897 (W 0.00189523) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.83024 ± 0.00111359 (W 0.00222063) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.81464 ± 0.00304805 (W 0.00547462) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.133744 ± 0.00140691 (W 0.00118449) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.150471 ± 0.00185879 (W 0.000741137) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.483049 ± 0.00078128 (W 0.00373798) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.796449 ± 0.0080842 (W 0.00311533) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.94303 ± 0.00216967 (W 0.00231036) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.46674 ± 0.0189909 (W 0.0125557) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.88281 ± 0.00443988 (W 0.00469204) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.89051 ± 0.0117534 (W 0.00325054) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.33463 ± 0.0170784 (W 0.0138487) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.37298 ± 0.0197114 (W 0.0145499) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.8136 ± 0.0765164 (W 0.0278387) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.9655 ± 0.0570285 (W 0.0322493) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.958255 ± 0.00240616 (W 0.00179938) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.957332 ± 0.00224913 (W 0.00239322) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.87455 ± 0.00928012 (W 0.00864686) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.8925 ± 0.0123443 (W 0.00575589) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.8916 ± 0.0493476 (W 0.0192226) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 15.9651 ± 0.0630267 (W 0.0243024) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007679 ± 0.000103923 (W 8.17026e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.018244 ± 8.38183e-05 (W 0.000147234) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.044217 ± 0.000310147 (W 0.000299617) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `lu/128`, `lu/256`, `lu/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `solve/128`, `solve/256`, `solve/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `triangular_solve/128`, `triangular_solve/256`, `triangular_solve/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.968405 ± 0.00776299 (W 0.00264014) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.844839 ± 0.0104385 (W 0.00299458) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 53.7065 ± 0.523121 (W 0.31892) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 49.8705 ± 0.475869 (W 0.25656) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.559187 ± 0.00263375 (W 0.0034617) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.555746 ± 0.00308711 (W 0.002919) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.1088 ± 0.264367 (W 0.0552759) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.0586 ± 0.225826 (W 0.0859827) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.326253 ± 0.00305704 (W 0.00281208) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.349064 ± 0.00302566 (W 0.00305695) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 12.8858 ± 0.0408216 (W 0.0408632) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 13.8341 ± 0.0457654 (W 0.0436478) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.45567 ± 0.0034903 (W 0.00395139) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.71601 ± 0.0106923 (W 0.00517452) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 68.2775 ± 0.176326 (W 0.295809) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 72.2549 ± 0.237173 (W 0.269245) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.918892 ± 0.00427413 (W 0.00410065) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 31.9955 ± 0.377166 (W 0.17009) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.02586 ± 0.00878028 (W 0.004908) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.04099 ± 0.01399 (W 0.00396989) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 83.4164 ± 0.21956 (W 0.323138) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 83.6715 ± 0.172956 (W 0.175619) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.01616 ± 0.00590513 (W 0.00384259) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01637 ± 0.00214847 (W 0.00368813) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 37.4712 ± 0.109784 (W 0.0931451) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 37.5118 ± 0.108398 (W 0.0486283) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.570998 ± 0.00228613 (W 0.00294759) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.605914 ± 0.0040086 (W 0.00297776) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 16.1026 ± 0.112118 (W 0.0696013) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 17.0956 ± 0.0609708 (W 0.0443707) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.170664 ± 0.00117975 (W 0.00276614) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.167301 ± 0.00228059 (W 0.00286085) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.54019 ± 0.0221383 (W 0.021573) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.52467 ± 0.0272422 (W 0.0255845) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.281199 ± 0.00188533 (W 0.00221878) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.314398 ± 0.00130004 (W 0.00193957) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.18939 ± 0.031001 (W 0.0414077) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.38369 ± 0.0623214 (W 0.033902) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.783672 ± 0.00298779 (W 0.0039751) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.12206 ± 0.00795397 (W 0.00668443) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 23.7192 ± 0.145291 (W 0.178188) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 31.4484 ± 0.140063 (W 0.0762007) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.595865 ± 0.00178519 (W 0.00358268) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.1281 ± 0.0806187 (W 0.0384017) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.34363 ± 0.00957532 (W 0.00423744) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.35749 ± 0.0138403 (W 0.00415391) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 36.89 ± 0.0642924 (W 0.0889094) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 37.117 ± 0.0459665 (W 0.0852263) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.869972 ± 0.00502063 (W 0.00493996) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.873092 ± 0.00811732 (W 0.00452258) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 22.0602 ± 0.145543 (W 0.068635) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 22.071 ± 0.0808299 (W 0.0336837) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.532706 ± 0.00554868 (W 0.00395945) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.598538 ± 0.00441449 (W 0.00337202) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 10.1677 ± 0.0460889 (W 0.027098) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 12.2821 ± 0.0419457 (W 0.0401711) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.10472 ± 0.00144521 (W 0.00294485) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.106299 ± 0.00181584 (W 0.00367042) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.75769 ± 0.0469353 (W 0.042662) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.73462 ± 0.050822 (W 0.0544615) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.251451 ± 0.00158644 (W 0.00266138) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.310756 ± 0.00117063 (W 0.00184487) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.28621 ± 0.0186719 (W 0.0258172) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 6.36277 ± 0.0414835 (W 0.0386501) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.701712 ± 0.00240711 (W 0.00395323) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.04348 ± 0.00282204 (W 0.0064617) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 16.2987 ± 0.0169493 (W 0.0610556) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 25.4193 ± 0.0965078 (W 0.109841) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.54926 ± 0.0029031 (W 0.00459327) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 9.79043 ± 0.0158705 (W 0.0326889) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.26812 ± 0.0567329 (W 0.00395489) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.27619 ± 0.00762579 (W 0.00403911) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 30.5743 ± 0.107119 (W 0.112162) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 30.8179 ± 0.042495 (W 0.0826763) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.849768 ± 0.0110843 (W 0.00499198) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.851907 ± 0.00689107 (W 0.00485417) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.5111 ± 0.0476054 (W 0.0397176) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.5377 ± 0.0245278 (W 0.0276706) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.809678 ± 0.00873973 (W 0.00355387) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.737424 ± 0.00134406 (W 0.00286873) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 52.1423 ± 0.540262 (W 0.23192) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 47.157 ± 0.360108 (W 0.139792) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.603103 ± 0.00499349 (W 0.00391734) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.601718 ± 0.00570797 (W 0.00331084) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.827 ± 0.13153 (W 0.0612297) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.7412 ± 0.126543 (W 0.0473286) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.22539 ± 0.00113941 (W 0.00289092) | 0.254644 ± 0.00084714 (W 0.00239577) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.24171 ± 0.000608373 (W 0.00191421) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.8356 ± 0.0390331 (W 0.0209652) | 12.0237 ± 0.0533397 (W 0.0173582) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.3535 ± 0.0598637 (W 0.0218621) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.25288 ± 0.00522308 (W 0.00399154) | 1.87058 ± 0.00991741 (W 0.00638132) | 🔵 faer-fork 1.49× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.56807 ± 0.00445435 (W 0.00427243) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 61.182 ± 0.104612 (W 0.123768) | 86.5755 ± 0.335884 (W 0.159147) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 69.0126 ± 0.154742 (W 0.233195) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.667778 ± 0.00237071 (W 0.00270656) | 0.924268 ± 0.00264501 (W 0.002582) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 24.8383 ± 0.0801192 (W 0.0616711) | 38.4406 ± 0.202495 (W 0.149436) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 1.98885 ± 0.00259727 (W 0.0030868) | 3.19318 ± 0.0266246 (W 0.00709918) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.99664 ± 0.0178857 (W 0.0037324) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 77.8839 ± 0.23961 (W 0.219269) | 120.013 ± 0.593517 (W 0.225668) | 🔵 faer-fork 1.54× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 78.0127 ± 0.203512 (W 0.104392) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.936329 ± 0.00397742 (W 0.00306964) | 1.55399 ± 0.00394514 (W 0.00443749) | 🔵 faer-fork 1.66× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.939241 ± 0.00331083 (W 0.00329316) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.5172 ± 0.0213873 (W 0.0411115) | 53.9025 ± 0.51338 (W 0.101047) | 🔵 faer-fork 1.71× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.4949 ± 0.0853602 (W 0.0400331) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.490232 ± 0.00627843 (W 0.0068918) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.509154 ± 0.0110642 (W 0.00433643) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 15.322 ± 0.0431819 (W 0.0339102) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 14.5261 ± 0.0905329 (W 0.0569364) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.206392 ± 0.00266291 (W 0.000891281) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.203861 ± 0.0031491 (W 0.00100189) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 11.1368 ± 0.00982607 (W 0.0205587) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 11.1357 ± 0.00797184 (W 0.0180171) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.184706 ± 0.000729711 (W 0.00163393) | 0.195654 ± 0.0012596 (W 0.00198465) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.213087 ± 0.000983149 (W 0.00245001) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.26932 ± 0.0163117 (W 0.00949462) | 6.22036 ± 0.0329825 (W 0.0107383) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.75903 ± 0.021434 (W 0.0235806) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.700127 ± 0.00221853 (W 0.00365333) | 1.42798 ± 0.00895685 (W 0.0047253) | 🔵 faer-fork 2.04× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.981071 ± 0.00361801 (W 0.00282785) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 21.6721 ± 0.0352885 (W 0.103225) | 36.6757 ± 0.237135 (W 0.0619115) | 🔵 faer-fork 1.69× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 28.588 ± 0.057863 (W 0.069717) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.515039 ± 0.00195432 (W 0.00245479) | 0.990481 ± 0.00612529 (W 0.00194808) | 🔵 faer-fork 1.92× faster than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 10.818 ± 0.0237982 (W 0.0210276) | 19.9669 ± 0.0992912 (W 0.0268118) | 🔵 faer-fork 1.85× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 1.30888 ± 0.00240472 (W 0.0046624) | 3.01481 ± 0.0127089 (W 0.00502176) | 🔵 faer-fork 2.30× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.32059 ± 0.00438269 (W 0.00512649) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 31.4919 ± 0.104297 (W 0.0625051) | 65.9396 ± 0.346264 (W 0.0592711) | 🔵 faer-fork 2.09× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 31.7227 ± 0.0749749 (W 0.0736142) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.794339 ± 0.00138414 (W 0.00239957) | 1.61843 ± 0.0057876 (W 0.00382708) | 🔵 faer-fork 2.04× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.795616 ± 0.00205291 (W 0.0022177) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 16.1426 ± 0.116205 (W 0.0420322) | 34.7506 ± 0.428625 (W 0.0560586) | 🔵 faer-fork 2.15× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 16.1946 ± 0.0236129 (W 0.0322171) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.461518 ± 0.0071571 (W 0.0047891) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.4922 ± 0.00300303 (W 0.00322584) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.94809 ± 0.0290462 (W 0.0284675) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 9.71258 ± 0.0526404 (W 0.0441008) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.129352 ± 0.00111234 (W 0.00204272) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.128876 ± 0.00277202 (W 0.00185297) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 8.82669 ± 0.0201989 (W 0.0295558) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 8.83062 ± 0.0278303 (W 0.0298867) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.177532 ± 0.000754208 (W 0.00173042) | 0.178048 ± 0.00133429 (W 0.00194596) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.208244 ± 0.00205173 (W 0.00238937) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.2283 ± 0.00910443 (W 0.00661188) | 3.73853 ± 0.0112574 (W 0.00755386) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 3.76974 ± 0.00371271 (W 0.0132141) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.600308 ± 0.00198178 (W 0.00268289) | 1.47482 ± 0.00444125 (W 0.00397499) | 🔵 faer-fork 2.46× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.897267 ± 0.00120461 (W 0.00430266) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 15.8583 ± 0.510859 (W 0.0465252) | 28.6351 ± 0.343306 (W 0.108552) | 🔵 faer-fork 1.81× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 22.5259 ± 0.0770061 (W 0.0638909) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.496574 ± 0.00287736 (W 0.0037954) | 0.99324 ± 0.0071869 (W 0.00342858) | 🔵 faer-fork 2.00× faster than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 8.42772 ± 0.0689912 (W 0.032166) | 17.4704 ± 0.0966671 (W 0.0280328) | 🔵 faer-fork 2.07× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 1.20828 ± 0.00402042 (W 0.00357307) | 2.98736 ± 0.0152961 (W 0.00847886) | 🔵 faer-fork 2.47× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.22618 ± 0.00348547 (W 0.00316294) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 25.0864 ± 0.0671795 (W 0.0568918) | 58.1627 ± 0.410108 (W 0.0647123) | 🔵 faer-fork 2.32× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 25.2603 ± 0.116015 (W 0.0860735) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.771453 ± 0.00258591 (W 0.00333972) | 1.62809 ± 0.00435051 (W 0.00438291) | 🔵 faer-fork 2.11× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.776832 ± 0.00267491 (W 0.00199685) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 13.6607 ± 0.242441 (W 0.0245327) | 32.5371 ± 0.343912 (W 0.0423211) | 🔵 faer-fork 2.38× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.7221 ± 0.0676917 (W 0.028848) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.90175 ± 0.00781184 (W 0.00496929) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.77128 ± 0.0090983 (W 0.00709084) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 153.499 ± 0.563812 (W 0.812097) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 149.178 ± 0.952544 (W 0.853863) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.1911 ± 0.00964758 (W 0.00339725) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.19326 ± 0.00358641 (W 0.00288982) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 137.23 ± 0.258168 (W 0.231886) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 137.227 ± 0.181512 (W 0.202121) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.919504 ± 0.00404177 (W 0.00315541) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.961286 ± 0.00631084 (W 0.00465504) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 44.4991 ± 0.178052 (W 0.0431802) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 46.3201 ± 0.483443 (W 0.114528) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.48332 ± 0.0188397 (W 0.0115646) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 5.08041 ± 0.0107308 (W 0.0152095) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 239.706 ± 0.698837 (W 0.551195) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 249.877 ± 0.635638 (W 1.01353) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.34276 ± 0.0126971 (W 0.00586472) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 98.306 ± 0.326703 (W 0.280864) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 5.44268 ± 0.0201034 (W 0.00510695) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.48552 ± 0.0178548 (W 0.00621694) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 272.823 ± 0.740232 (W 0.512961) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 273.528 ± 1.10057 (W 6.34069) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.51292 ± 0.00677658 (W 0.00457119) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.53755 ± 0.00458178 (W 0.00507932) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 113.857 ± 1.55954 (W 0.586629) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 115.14 ± 1.48306 (W 0.33366) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.852468 ± 0.00856041 (W 0.00398675) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.926417 ± 0.00895843 (W 0.00298834) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 49.8541 ± 0.171329 (W 0.458777) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 51.9336 ± 0.19509 (W 0.380403) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.599757 ± 0.00267538 (W 0.00300161) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.599206 ± 0.0015009 (W 0.00212899) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 37.191 ± 0.0712273 (W 0.047677) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 37.1438 ± 0.060544 (W 0.0486047) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.610613 ± 0.00346994 (W 0.00434571) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.683415 ± 0.0025229 (W 0.00337837) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 18.2364 ± 0.0355589 (W 0.0409455) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 21.1169 ± 0.154917 (W 0.115055) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.83729 ± 0.0110366 (W 0.00541478) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.60152 ± 0.05445 (W 0.00782444) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 76.9617 ± 0.190905 (W 0.317009) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 91.4818 ± 0.209439 (W 0.199774) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.24468 ± 0.00253825 (W 0.00379245) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 36.411 ± 0.0812641 (W 0.0858586) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.66921 ± 0.0273591 (W 0.0063381) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.69609 ± 0.0274272 (W 0.00893266) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 98.6339 ± 0.114255 (W 0.153633) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 99.6073 ± 0.0381902 (W 0.118001) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.65535 ± 0.00559809 (W 0.00530699) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.65785 ± 0.0157994 (W 0.00458212) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 52.2495 ± 0.115342 (W 0.0975429) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 52.3974 ± 0.124954 (W 0.07007) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.717542 ± 0.00804515 (W 0.00394823) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.80958 ± 0.00718132 (W 0.00371429) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 25.8648 ± 0.110075 (W 0.129928) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 28.9027 ± 0.143773 (W 0.150016) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.317498 ± 0.00370054 (W 0.00615449) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.315057 ± 0.00683892 (W 0.00420497) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 24.206 ± 0.071208 (W 0.0836006) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 24.1527 ± 0.0771229 (W 0.377492) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.529516 ± 0.00177486 (W 0.00408894) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.619322 ± 0.00232086 (W 0.00375152) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.2071 ± 0.034788 (W 0.0319954) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 16.8809 ± 0.0374269 (W 0.0540875) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.46259 ± 0.00887895 (W 0.00709057) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.23521 ± 0.0687662 (W 0.0132528) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 48.3275 ± 0.145699 (W 0.116083) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 68.8811 ± 0.341101 (W 0.253306) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.11252 ± 0.00737443 (W 0.0057719) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 26.7325 ± 0.0896977 (W 0.0390475) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.29234 ± 0.0276473 (W 0.00476175) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.33126 ± 0.018542 (W 0.00675082) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 73.8485 ± 0.0511893 (W 0.143656) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 74.6785 ± 0.138078 (W 0.121372) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.55271 ± 0.00472967 (W 0.0042713) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.56755 ± 0.0160663 (W 0.00498518) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 43.272 ± 0.0699927 (W 0.056865) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 43.369 ± 1.49843 (W 0.410936) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.70793 ± 0.0105476 (W 0.00316714) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.6694 ± 0.00754175 (W 0.00487866) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 150.42 ± 0.575503 (W 0.858758) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 142.911 ± 0.625716 (W 0.838011) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.26671 ± 0.0116706 (W 0.00765147) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.25658 ± 0.0122198 (W 0.00439417) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 138.101 ± 0.16474 (W 0.166326) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 138.11 ± 0.146578 (W 0.144277) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.817754 ± 0.00616862 (W 0.00632402) | 0.84832 ± 0.00402125 (W 0.00251807) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.850758 ± 0.0065402 (W 0.00445097) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 40.0876 ± 0.0730081 (W 0.0717442) | 44.9909 ± 0.0964932 (W 0.0418575) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 41.1571 ± 0.160752 (W 0.0445081) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.30191 ± 0.0164942 (W 0.00779761) | 5.19154 ± 0.0242073 (W 0.00935334) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 4.91302 ± 0.0214116 (W 0.0109035) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 228.971 ± 1.56267 (W 0.335729) | 296.63 ± 0.656391 (W 0.495217) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 243.407 ± 0.698441 (W 0.827133) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.02217 ± 0.00911953 (W 0.00203874) | 1.76878 ± 0.0107772 (W 0.00242198) | 🔴 faer-fork 1.14× slower than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 87.5954 ± 0.315886 (W 0.0897293) | 111.938 ± 0.430919 (W 0.265221) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 5.32724 ± 0.0246161 (W 0.010076) | 6.15032 ± 0.0261394 (W 0.0106136) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.36512 ± 0.0276577 (W 0.00971359) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 260.744 ± 0.536229 (W 0.397985) | 346.595 ± 0.489361 (W 0.747163) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 261.502 ± 0.225407 (W 6.54733) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.39654 ± 0.0144442 (W 0.00741579) | 2.67769 ± 0.00765798 (W 0.00522854) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.42707 ± 0.0080306 (W 0.00492809) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 100.785 ± 0.331051 (W 0.127841) | 136.074 ± 0.782812 (W 0.36996) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 100.935 ± 0.230959 (W 0.276797) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.78103 ± 0.0071719 (W 0.00288433) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.82069 ± 0.0112241 (W 0.00465555) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 47.6533 ± 0.153892 (W 0.145828) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 46.7683 ± 0.140275 (W 0.112182) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.659984 ± 0.00489692 (W 0.00277955) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.657956 ± 0.00667823 (W 0.00257943) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 39.2875 ± 0.0766614 (W 0.173483) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 39.284 ± 0.034311 (W 0.106444) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.517559 ± 0.0011661 (W 0.00334479) | 0.563704 ± 0.00134379 (W 0.00205944) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.572702 ± 0.001914 (W 0.00299897) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 14.5534 ± 0.0315489 (W 0.041106) | 22.1054 ± 0.0391633 (W 0.0190045) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 15.6259 ± 0.0557127 (W 0.0589385) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.78534 ± 0.00510089 (W 0.00282599) | 2.19861 ± 0.00481807 (W 0.00542451) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.41207 ± 0.0142108 (W 0.00849622) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 71.695 ± 0.128173 (W 0.0511727) | 93.1372 ± 0.441055 (W 0.138335) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 85.6386 ± 0.293632 (W 0.199849) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.152 ± 0.00360157 (W 0.00357648) | 0.728102 ± 0.00695773 (W 0.00193997) | 🔴 faer-fork 1.58× slower than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 32.2047 ± 0.112439 (W 0.0646468) | 35.8047 ± 0.0991392 (W 0.0546555) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 2.57423 ± 0.0218243 (W 0.00416821) | 3.04244 ± 0.0197438 (W 0.00785296) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.59241 ± 0.00749454 (W 0.00369449) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 86.6314 ± 0.0940629 (W 0.17748) | 122.234 ± 0.365359 (W 0.177211) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 87.2858 ± 0.114557 (W 0.177382) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.56831 ± 0.00628207 (W 0.00449507) | 1.63918 ± 0.00460666 (W 0.00615837) | · |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.57993 ± 0.00473411 (W 0.00650122) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 40.1857 ± 0.0694435 (W 0.0859429) | 55.8771 ± 0.371803 (W 0.0694663) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 40.257 ± 0.084783 (W 0.0702296) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.6502 ± 0.00213574 (W 0.00366834) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.71143 ± 0.00548474 (W 0.0026957) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 23.4554 ± 0.0808111 (W 0.127343) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 23.602 ± 0.122956 (W 0.0982488) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.380597 ± 0.00128834 (W 0.00402613) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.383179 ± 0.00297403 (W 0.00261148) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 25.8072 ± 0.901421 (W 1.02301) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 25.3812 ± 0.852549 (W 1.07465) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.453277 ± 0.00210664 (W 0.00183556) | 0.445545 ± 0.00227473 (W 0.00225324) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.511996 ± 0.00103179 (W 0.00297964) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.428 ± 0.0282139 (W 0.0314857) | 12.5827 ± 0.0454071 (W 0.0158087) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.3715 ± 0.0288678 (W 0.0348408) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.39594 ± 0.00653339 (W 0.00359315) | 1.90945 ± 0.00811341 (W 0.0035471) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.04851 ± 0.0419754 (W 0.00953741) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 48.6555 ± 0.109524 (W 0.118761) | 60.8089 ± 0.236204 (W 0.0686524) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 62.4301 ± 0.385552 (W 0.18739) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.03097 ± 0.0035179 (W 0.00392013) | 0.585365 ± 0.00293878 (W 0.00286563) | 🔴 faer-fork 1.76× slower than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 23.7212 ± 0.042069 (W 0.0403305) | 24.3522 ± 0.0689905 (W 0.0327432) | · |  |
| `svd_thin/1024x64/b1` | 2.18511 ± 0.00798695 (W 0.00789361) | 2.59801 ± 0.0195897 (W 0.00688599) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.21309 ± 0.00983192 (W 0.010199) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 61.3198 ± 0.168173 (W 0.0778306) | 87.6448 ± 0.314117 (W 0.391833) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 62.1436 ± 0.14076 (W 0.0929202) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.47214 ± 0.00458518 (W 0.00647977) | 1.50013 ± 0.0126609 (W 0.00491145) | · |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.48216 ± 0.00521387 (W 0.00398055) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.3777 ± 0.110742 (W 0.053305) | 44.3125 ± 0.394091 (W 0.369673) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.3821 ± 0.0828695 (W 0.0772582) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.674121 ± 0.0130878 (W 0.0103288) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.547034 ± 0.00806769 (W 0.00309573) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 26.1395 ± 0.138649 (W 0.14078) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 23.9465 ± 0.206077 (W 0.12758) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.283012 ± 0.00300969 (W 0.00134902) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.280613 ± 0.00185084 (W 0.00217906) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.4547 ± 0.0361808 (W 0.034433) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.4585 ± 0.0194043 (W 0.0361265) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.31667 ± 0.00133739 (W 0.00204219) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.239191 ± 0.000834133 (W 0.00258358) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 7.63825 ± 0.0234609 (W 0.0196957) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 8.35259 ± 0.163775 (W 0.123028) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.965769 ± 0.00504052 (W 0.00286135) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.03652 ± 0.00493999 (W 0.00384747) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 36.7711 ± 0.822819 (W 0.197672) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 38.5792 ± 0.288327 (W 0.0959533) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.688087 ± 0.00211648 (W 0.00418842) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 18.5823 ± 0.0911641 (W 0.0805068) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.36429 ± 0.00534441 (W 0.00304715) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.36969 ± 0.00391418 (W 0.00516806) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 44.9567 ± 0.656037 (W 0.11884) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 45.1664 ± 0.705087 (W 0.106082) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.733586 ± 0.00430653 (W 0.00511808) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.732372 ± 0.00331812 (W 0.00424874) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 21.8828 ± 0.0694106 (W 0.0632878) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 21.8126 ± 0.0919646 (W 0.0418756) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.464594 ± 0.00306344 (W 0.00358987) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.479273 ± 0.00654447 (W 0.00469966) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.68481 ± 0.0554215 (W 0.0336909) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 10.9676 ± 0.0601588 (W 0.0536579) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.089252 ± 0.00153335 (W 0.00152179) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.087723 ± 0.00104905 (W 0.00132961) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.64076 ± 0.0144078 (W 0.0160377) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.64713 ± 0.0105987 (W 0.0189531) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.219765 ± 0.00143943 (W 0.00201561) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.234519 ± 0.00127482 (W 0.00168308) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.94884 ± 0.0390781 (W 0.0143633) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.20598 ± 0.0601751 (W 0.10365) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.568729 ± 0.00217333 (W 0.00368003) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.749006 ± 0.00645438 (W 0.00321247) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 13.213 ± 0.0799719 (W 0.0363838) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 17.921 ± 0.0906288 (W 0.145448) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.459785 ± 0.00195144 (W 0.00260146) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 7.9582 ± 0.0454307 (W 0.0289674) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.01685 ± 0.002413 (W 0.00250748) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.02729 ± 0.00650709 (W 0.00188195) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 21.3621 ± 0.0688413 (W 0.024023) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 21.4378 ± 0.530789 (W 0.0487488) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.685376 ± 0.00461313 (W 0.00348045) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.686063 ± 0.00144695 (W 0.0023203) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 14.3993 ± 0.0701282 (W 0.0267541) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 14.4178 ± 0.0668402 (W 0.0205687) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.449438 ± 0.00477995 (W 0.00362806) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.484357 ± 0.00125393 (W 0.00355694) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 7.08614 ± 0.0207085 (W 0.0287735) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.91942 ± 0.0473111 (W 0.0476638) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.0758365 ± 0.00171479 (W 0.00271756) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.0768385 ± 0.00107188 (W 0.00167903) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 2.59368 ± 0.0352071 (W 0.0224907) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 2.59043 ± 0.00988826 (W 0.0169315) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.194943 ± 0.0012995 (W 0.00179459) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.228446 ± 0.00154425 (W 0.00190765) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.85432 ± 0.00932062 (W 0.0149438) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 4.77862 ± 0.0521779 (W 0.0862716) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.534225 ± 0.005491 (W 0.00461606) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.706541 ± 0.00319766 (W 0.00384581) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 9.22043 ± 0.0181209 (W 0.0392674) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.3589 ± 0.0752284 (W 0.12846) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.425373 ± 0.00191558 (W 0.00293503) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.89706 ± 0.0206913 (W 0.0217388) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.962624 ± 0.0070064 (W 0.00196188) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.969972 ± 0.00152305 (W 0.00134045) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 17.6759 ± 0.0536625 (W 0.093482) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 17.7693 ± 0.0305456 (W 0.0397473) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.667507 ± 0.00282598 (W 0.0027284) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.672608 ± 0.00599618 (W 0.00389186) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 12.8692 ± 0.0379453 (W 0.0192463) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 12.8659 ± 0.0384133 (W 0.0196031) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.494615 ± 0.00241217 (W 0.00368957) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.462255 ± 0.00190659 (W 0.00383208) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 24.3332 ± 0.0632804 (W 0.0518043) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 21.5727 ± 0.0409536 (W 0.0320345) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.301941 ± 0.0014784 (W 0.00238492) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.301857 ± 0.00140234 (W 0.00281986) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.8198 ± 0.0413628 (W 0.0466134) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.7952 ± 0.0440862 (W 0.0421987) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.149074 ± 0.00598242 (W 0.00156269) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.155722 ± 0.000729534 (W 0.00163503) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.64273 ± 0.0657659 (W 0.00851452) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.81088 ± 0.0842685 (W 0.0175693) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.746358 ± 0.00249073 (W 0.00279614) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.907223 ± 0.00863497 (W 0.00391847) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 31.9642 ± 0.253776 (W 0.0638985) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 35.5828 ± 0.195776 (W 0.0661926) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.441678 ± 0.00164504 (W 0.00338753) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.924 ± 0.0888393 (W 0.0280148) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.36187 ± 0.0205346 (W 0.00391896) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.36489 ± 0.00366153 (W 0.00404081) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 42.181 ± 0.135072 (W 0.0489923) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 42.4006 ± 0.115464 (W 0.0960329) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.695697 ± 0.00229029 (W 0.00193232) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.697461 ± 0.00254139 (W 0.00268341) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 18.7005 ± 0.0997347 (W 0.0278937) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 18.6849 ± 0.0751633 (W 0.0316517) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.376209 ± 0.00199499 (W 0.00412173) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.396903 ± 0.00310762 (W 0.0031523) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.2615 ± 0.031643 (W 0.0172901) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.7579 ± 0.0388951 (W 0.0243952) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.105227 ± 0.000444033 (W 0.000801058) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.104911 ± 0.000353493 (W 0.000962685) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.49686 ± 0.0114675 (W 0.0123031) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.49187 ± 0.0120987 (W 0.0105228) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.141173 ± 0.000621889 (W 0.00109661) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.152995 ± 0.00115891 (W 0.00182447) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.70515 ± 0.00875797 (W 0.00567288) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.9119 ± 0.00746611 (W 0.00895312) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.487252 ± 0.00346635 (W 0.00242895) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.618922 ± 0.00211795 (W 0.0031327) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 11.7426 ± 0.0776021 (W 0.0186433) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.8875 ± 0.077785 (W 0.0418163) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.387923 ± 0.00144511 (W 0.00264109) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 6.71049 ± 0.0317739 (W 0.011766) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.00146 ± 0.00373869 (W 0.00256743) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.00903 ± 0.005974 (W 0.00334678) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 18.669 ± 0.0458416 (W 0.0240916) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 18.7555 ± 0.061709 (W 0.0151387) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.642247 ± 0.00172291 (W 0.00267172) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.642453 ± 0.00144974 (W 0.00273115) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 11.1126 ± 0.0466459 (W 0.0195339) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 11.092 ± 0.0346899 (W 0.0202067) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.38353 ± 0.00374115 (W 0.00371292) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.405067 ± 0.00372648 (W 0.0044911) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.84423 ± 0.0278061 (W 0.0219756) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 6.75706 ± 0.0228598 (W 0.0199135) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.0693395 ± 0.000331902 (W 0.000948138) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.068964 ± 0.000870555 (W 0.000563812) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 3.65123 ± 0.258145 (W 0.089351) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 3.71034 ± 0.256035 (W 0.11297) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.135267 ± 0.00191553 (W 0.0013306) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.14764 ± 0.00112383 (W 0.00145666) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.23803 ± 0.00887746 (W 0.00731801) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.46652 ± 0.119621 (W 0.00912349) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.440335 ± 0.00188214 (W 0.00217736) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.578677 ± 0.00282551 (W 0.00429349) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 8.21468 ± 0.0354713 (W 0.0105088) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 11.4272 ± 0.0402617 (W 0.0304255) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.373107 ± 0.00646792 (W 0.00278387) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.16224 ± 0.0441895 (W 0.0182365) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.947154 ± 0.00634717 (W 0.00244753) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.955681 ± 0.00416927 (W 0.00267909) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 14.9388 ± 0.0484839 (W 0.0272848) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 15.0427 ± 0.0496496 (W 0.0391886) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.625155 ± 0.00862143 (W 0.00226084) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.62797 ± 0.00310898 (W 0.00235189) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 9.50987 ± 0.0458585 (W 0.0116903) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 9.51603 ± 0.0192071 (W 0.0172516) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.22055 ± 0.00262272 (W 0.00505723) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.09259 ± 0.0018761 (W 0.00255996) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 63.7306 ± 0.43027 (W 0.304838) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 60.8716 ± 0.620894 (W 0.473726) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.10746 ± 0.00337404 (W 0.00380575) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.1085 ± 0.00371389 (W 0.00226596) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 68.6745 ± 0.0637688 (W 0.225088) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 68.5902 ± 0.113954 (W 0.0892851) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.677618 ± 0.00416366 (W 0.00364186) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.64022 ± 0.00382277 (W 0.00243059) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 23.2145 ± 0.0723366 (W 0.028) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 25.0223 ± 0.102755 (W 0.087409) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.49547 ± 0.0169235 (W 0.00427904) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.77846 ± 0.00382478 (W 0.00807825) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 125.321 ± 0.458071 (W 0.660523) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 129.602 ± 0.347943 (W 0.298595) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.40908 ± 0.00371372 (W 0.00475058) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 53.469 ± 0.168617 (W 0.0967774) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.07968 ± 0.0250909 (W 0.00830378) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.08613 ± 0.00612102 (W 0.00507255) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 140.494 ± 0.488012 (W 0.270977) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 140.814 ± 0.214353 (W 0.316253) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.50681 ± 0.00498254 (W 0.00688539) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.50908 ± 0.00287036 (W 0.00432121) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 59.6335 ± 0.0988458 (W 0.0742199) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 59.727 ± 0.0524309 (W 0.0916391) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.70048 ± 0.00430028 (W 0.00360997) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.734925 ± 0.00584929 (W 0.00337062) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 20.3513 ± 0.0599092 (W 0.0305872) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 21.647 ± 0.053858 (W 0.0593542) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.299246 ± 0.00190152 (W 0.00181448) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.299905 ± 0.00142472 (W 0.00136196) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.4833 ± 0.0349976 (W 0.0339425) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.4868 ± 0.0322992 (W 0.0368884) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.484351 ± 0.00236814 (W 0.00277953) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.522187 ± 0.00137936 (W 0.00221558) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 11.6884 ± 0.0673996 (W 0.0499916) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 13.0293 ± 0.0453623 (W 0.0404845) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.11315 ± 0.00206173 (W 0.00373248) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.46722 ± 0.00441284 (W 0.00535682) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 40.5028 ± 0.217932 (W 0.0820079) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 49.3437 ± 0.121952 (W 0.213513) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.808346 ± 0.00294929 (W 0.00299043) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 21.6247 ± 0.0486865 (W 0.0252149) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.67746 ± 0.0071268 (W 0.00561858) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.69488 ± 0.010104 (W 0.00508462) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 53.1592 ± 0.135535 (W 0.315355) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 53.402 ± 0.185712 (W 0.0977931) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.07459 ± 0.00330451 (W 0.00552279) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.07402 ± 0.0104646 (W 0.00349233) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 29.6058 ± 0.0678658 (W 0.0440008) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 29.6489 ± 0.0573929 (W 0.039706) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.640003 ± 0.00596393 (W 0.00420139) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.694449 ± 0.00400526 (W 0.00313569) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 13.1087 ± 0.0598455 (W 0.0435901) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 15.7804 ± 0.047737 (W 0.0381337) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.175132 ± 0.0075964 (W 0.00435933) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.175308 ± 0.00179023 (W 0.00200671) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.57513 ± 0.0366966 (W 0.0507589) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.55959 ± 0.0451252 (W 0.0565626) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.440437 ± 0.00254891 (W 0.00342141) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.502645 ± 0.00297452 (W 0.00233026) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.53512 ± 0.0211942 (W 0.0208643) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 11.0761 ± 0.0500273 (W 0.0412225) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.91742 ± 0.00346509 (W 0.00509811) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.28063 ± 0.00602096 (W 0.00652288) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 25.8486 ± 0.0672302 (W 0.0829496) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 37.0087 ± 0.13729 (W 0.147095) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.729364 ± 0.00309765 (W 0.00278172) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 15.603 ± 0.0494847 (W 0.0324982) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.48826 ± 0.0111279 (W 0.00276367) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.48641 ± 0.00970577 (W 0.00442626) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 40.0922 ± 0.0281809 (W 0.0698586) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 40.4008 ± 0.094403 (W 0.0938946) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.01743 ± 0.00554629 (W 0.00350193) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01521 ± 0.00853761 (W 0.00437204) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 24.5268 ± 0.139433 (W 0.0457451) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.566 ± 0.437266 (W 0.173615) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.997834 ± 0.0027482 (W 0.00217054) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.986443 ± 0.00215944 (W 0.00283108) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 60.1913 ± 0.163999 (W 0.587235) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 56.5657 ± 0.484363 (W 0.202678) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.14955 ± 0.00348265 (W 0.00209271) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.14805 ± 0.00437001 (W 0.00251721) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 69.2748 ± 0.0968313 (W 0.0654723) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 69.3446 ± 0.0800264 (W 0.0735454) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.513421 ± 0.00267111 (W 0.00356125) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.531495 ± 0.00226621 (W 0.00305883) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 21.2689 ± 0.0649072 (W 0.0397276) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 21.7371 ± 0.115136 (W 0.118039) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.3067 ± 0.00716137 (W 0.00655219) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.65504 ± 0.0103153 (W 0.00650322) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 117.635 ± 0.263239 (W 0.168312) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 125.634 ± 0.125494 (W 0.185421) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.14899 ± 0.00324174 (W 0.00337459) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 46.9226 ± 0.0629475 (W 0.0556559) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.05204 ± 0.00698798 (W 0.00966912) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.05459 ± 0.0108763 (W 0.00592291) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 135.101 ± 0.277152 (W 0.170819) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 135.421 ± 0.206297 (W 0.192364) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.43516 ± 0.00581904 (W 0.00476418) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.43471 ± 0.00364581 (W 0.00418277) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 54.1102 ± 0.10518 (W 0.0655484) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 54.2744 ± 0.0920378 (W 0.121512) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.600189 ± 0.00312113 (W 0.00276935) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.632103 ± 0.00235721 (W 0.00350006) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 18.4358 ± 0.0539574 (W 0.0263908) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 18.1022 ± 0.0543831 (W 0.0332322) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.344714 ± 0.00120993 (W 0.00208961) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.344739 ± 0.00234229 (W 0.00240986) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 19.294 ± 0.0166263 (W 0.0179472) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 19.2926 ± 0.0135714 (W 0.0212779) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.389022 ± 0.00204798 (W 0.00271258) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.419567 ± 0.00182682 (W 0.00280111) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.95561 ± 0.0271877 (W 0.0152446) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 9.45037 ± 0.0194308 (W 0.0179525) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.0425 ± 0.00501289 (W 0.00174199) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.33793 ± 0.00796935 (W 0.00474917) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 38.3442 ± 0.0956211 (W 0.0849994) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 45.0972 ± 0.134237 (W 0.132115) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.71085 ± 0.00153222 (W 0.00351031) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 18.8862 ± 0.0328526 (W 0.020497) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.63849 ± 0.0058843 (W 0.00402896) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.65276 ± 0.00446937 (W 0.00532471) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 47.8875 ± 0.0847518 (W 0.077151) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 48.0491 ± 0.107089 (W 0.110672) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.01293 ± 0.00348882 (W 0.00332402) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01174 ± 0.00319807 (W 0.00170771) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 23.9743 ± 0.059621 (W 0.0373225) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.0713 ± 0.0262141 (W 0.0357334) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.561055 ± 0.00244846 (W 0.00326433) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.596131 ± 0.00355352 (W 0.0041211) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 12.1648 ± 0.0439128 (W 0.0300357) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 12.2116 ± 0.0622117 (W 0.0145559) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.199107 ± 0.0010095 (W 0.00133937) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.199708 ± 0.00130545 (W 0.00180314) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.5903 ± 0.0396212 (W 0.0543393) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.5835 ± 0.0263332 (W 0.0437926) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.367655 ± 0.00128822 (W 0.00253623) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.397217 ± 0.00137005 (W 0.00242833) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 7.04653 ± 0.0149983 (W 0.0191388) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.54544 ± 0.0437096 (W 0.0165831) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.850218 ± 0.00350985 (W 0.00268282) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.1496 ± 0.00402654 (W 0.00657531) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 26.0455 ± 0.0430903 (W 0.0901061) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 32.9074 ± 0.079923 (W 0.0886101) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.651404 ± 0.00212916 (W 0.00315719) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.9345 ± 0.0495515 (W 0.0230283) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.44395 ± 0.0125926 (W 0.00376695) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.45339 ± 0.00632586 (W 0.0058721) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 34.7408 ± 0.0937045 (W 0.0571059) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 35.0024 ± 0.172845 (W 0.0872384) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.957363 ± 0.00484063 (W 0.00242275) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.954353 ± 0.00408381 (W 0.00203706) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 18.9521 ± 0.0827001 (W 0.0335165) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.0044 ± 0.0609005 (W 0.0254747) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `lu/1024x64`, `lu/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `solve/1024x64`, `solve/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 52.1175 ± 0.351001 (W 0.0993801) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 79.2128 ± 0.365901 (W 0.642461) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.791001 ± 0.00236894 (W 0.00358997) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.26238 ± 0.00500369 (W 0.0042685) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.622 ± 0.0589508 (W 0.0315835) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 16.1301 ± 0.167056 (W 0.12412) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.279002 ± 0.000844378 (W 0.00262446) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.432493 ± 0.002144 (W 0.00291458) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 29.0514 ± 0.175991 (W 0.0751821) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 30.8862 ± 0.415287 (W 0.156021) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.02858 ± 0.00540735 (W 0.00455985) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.753473 ± 0.00260551 (W 0.00264351) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.4655 ± 0.259887 (W 0.0560341) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.994379 ± 0.00437576 (W 0.00558611) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 77.7145 ± 0.26554 (W 0.118102) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 78.0668 ± 0.196958 (W 0.418339) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.9805 ± 0.00780772 (W 0.00394308) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.9876 ± 0.00816398 (W 0.00332422) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 31.663 ± 0.0983119 (W 0.0792434) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.6291 ± 0.093667 (W 0.0494192) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.940306 ± 0.00268566 (W 0.00415655) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.940513 ± 0.00400316 (W 0.00298337) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 15.5702 ± 0.0780294 (W 0.030176) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 33.7553 ± 0.214186 (W 0.077122) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.463147 ± 0.00222966 (W 0.00438024) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.769356 ± 0.00506239 (W 0.00445352) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.51366 ± 0.0120336 (W 0.0229508) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.62479 ± 0.0197284 (W 0.0501656) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.117855 ± 0.00182995 (W 0.00266007) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.227559 ± 0.00154279 (W 0.00202321) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.5289 ± 0.0343877 (W 0.0236082) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 10.4238 ± 0.0447998 (W 0.0453961) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.636442 ± 0.00291165 (W 0.00501054) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.378286 ± 0.00227161 (W 0.00196215) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.1125 ± 0.196547 (W 0.0292822) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.616098 ± 0.0025436 (W 0.00518098) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 31.4378 ± 0.103735 (W 0.0661145) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 31.6411 ± 0.0561775 (W 0.0779464) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.30686 ± 0.0403284 (W 0.00561732) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.3166 ± 0.00369461 (W 0.00440478) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 16.2376 ± 0.0327446 (W 0.0296835) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 16.2787 ± 0.0438321 (W 0.0254236) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.792593 ± 0.00341119 (W 0.00148716) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.796938 ± 0.00322015 (W 0.00323588) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.2033 ± 2.05737 (W 0.0389815) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 27.8214 ± 4.02002 (W 0.112187) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.436716 ± 0.00643343 (W 0.00236678) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.716102 ± 0.0578907 (W 0.00352892) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 2.12343 ± 0.127802 (W 0.0108792) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.9543 ± 0.341463 (W 0.0183059) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.094782 ± 0.00146781 (W 0.00222002) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.197653 ± 0.0215074 (W 0.00221283) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 7.57127 ± 0.596261 (W 0.0456847) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 7.11745 ± 0.964887 (W 0.0218064) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.571611 ± 0.00487622 (W 0.00532375) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.327642 ± 0.025461 (W 0.00184669) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 7.33323 ± 0.267049 (W 0.0262028) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.551899 ± 0.0103212 (W 0.00475847) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 25.1241 ± 1.48361 (W 0.115493) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 25.2457 ± 1.44469 (W 0.066336) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.2069 ± 0.0343512 (W 0.00432439) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.22505 ± 0.0347021 (W 0.00412929) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 13.7477 ± 0.968063 (W 0.0336012) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 13.8015 ± 0.905321 (W 0.0264816) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.777448 ± 0.0186676 (W 0.00389498) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.777288 ± 0.0216599 (W 0.00396014) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 88.5361 ± 0.599712 (W 0.536847) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 76.2607 ± 0.938077 (W 0.339323) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.21683 ± 0.00628506 (W 0.006468) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.15829 ± 0.0045206 (W 0.00409928) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 12.7017 ± 0.0564305 (W 0.0395966) | 19.1768 ± 0.0501877 (W 0.0331617) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 13.2379 ± 0.111241 (W 0.0333364) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.306466 ± 0.000781172 (W 0.00250247) | 0.658878 ± 0.00243157 (W 0.00275413) | 🔵 faer-fork 2.15× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.320915 ± 0.00128567 (W 0.00300392) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.6094 ± 0.18697 (W 0.141238) | 57.0619 ± 0.354712 (W 0.273527) | 🔵 faer-fork 2.14× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.5343 ± 0.243018 (W 0.135607) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.604799 ± 0.00235192 (W 0.00279169) | 1.45336 ± 0.0117606 (W 0.00506812) | 🔵 faer-fork 2.40× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.627014 ± 0.00135106 (W 0.00270809) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 25.0362 ± 0.299284 (W 0.320838) | 53.8752 ± 0.624702 (W 0.230052) | 🔵 faer-fork 2.15× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 0.565852 ± 0.00313296 (W 0.00364318) | 1.37577 ± 0.00640644 (W 0.00450622) | 🔵 faer-fork 2.43× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 83.3832 ± 0.626386 (W 0.323745) | 200.806 ± 0.900989 (W 0.76662) | 🔵 faer-fork 2.41× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 83.4418 ± 0.606804 (W 0.231214) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.02818 ± 0.00943884 (W 0.00459993) | 4.47038 ± 0.0215282 (W 0.0110793) | 🔵 faer-fork 2.20× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.0403 ± 0.00748121 (W 0.00559574) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 37.3165 ± 0.0709913 (W 0.049905) | 91.7774 ± 0.374958 (W 0.254822) | 🔵 faer-fork 2.46× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 37.4036 ± 0.0687532 (W 0.09594) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.00826 ± 0.00343787 (W 0.00469046) | 2.24011 ± 0.0140278 (W 0.0109302) | 🔵 faer-fork 2.22× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.00779 ± 0.00529464 (W 0.00577167) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 32.9363 ± 0.135651 (W 0.0719144) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 31.0369 ± 0.159138 (W 0.116678) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.663349 ± 0.00222914 (W 0.00361245) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.675372 ± 0.00445403 (W 0.00425505) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.39644 ± 0.00949043 (W 0.0101634) | 10.9421 ± 0.0470868 (W 0.020675) | 🔵 faer-fork 3.22× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 3.85177 ± 0.0155266 (W 0.0185975) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.096716 ± 0.000413635 (W 0.000312793) | 0.43758 ± 0.00156066 (W 0.00231469) | 🔵 faer-fork 4.52× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.116688 ± 0.0081533 (W 0.000330006) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 7.01768 ± 0.0348754 (W 0.0292016) | 30.164 ± 0.661513 (W 0.180442) | 🔵 faer-fork 4.30× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 7.61983 ± 0.033198 (W 0.0257511) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.220126 ± 0.000572831 (W 0.00182087) | 1.32616 ± 0.00465826 (W 0.00443803) | 🔵 faer-fork 6.02× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.249421 ± 0.000276061 (W 0.00189151) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.53416 ± 0.0192568 (W 0.0122506) | 28.1812 ± 0.321251 (W 0.108951) | 🔵 faer-fork 4.31× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 0.200243 ± 0.000800768 (W 0.00159666) | 1.27713 ± 0.00547628 (W 0.00509112) | 🔵 faer-fork 6.38× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 36.8758 ± 0.0659503 (W 0.136949) | 122.6 ± 1.3371 (W 0.711908) | 🔵 faer-fork 3.32× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.9586 ± 0.0934506 (W 0.0951186) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.34207 ± 0.00575494 (W 0.0036064) | 3.96363 ± 0.0127587 (W 0.0123921) | 🔵 faer-fork 2.95× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.35519 ± 0.00348024 (W 0.00496099) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.943 ± 0.0511691 (W 0.0428287) | 63.4488 ± 0.604195 (W 0.407297) | 🔵 faer-fork 2.89× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.97 ± 0.076885 (W 0.0343272) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.861168 ± 0.00441977 (W 0.00560462) | 2.14441 ± 0.0127609 (W 0.00782313) | 🔵 faer-fork 2.49× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.865787 ± 0.00458833 (W 0.00469597) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 25.9498 ± 4.59161 (W 0.0683579) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 25.3006 ± 3.57041 (W 0.0617226) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.607617 ± 0.00806857 (W 0.00456972) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.623816 ± 0.0175434 (W 0.00377354) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 1.87759 ± 0.101608 (W 0.0123698) | 8.57654 ± 0.108834 (W 0.0271497) | 🔵 faer-fork 4.57× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 2.32934 ± 0.131041 (W 0.00730234) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.0628525 ± 0.00247295 (W 0.000397444) | 0.395659 ± 0.00510987 (W 0.00232534) | 🔵 faer-fork 6.30× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.0882045 ± 0.00399167 (W 0.000442607) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 3.87257 ± 0.564039 (W 0.028691) | 27.8465 ± 4.82064 (W 0.14897) | 🔵 faer-fork 7.19× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 4.33846 ± 0.650484 (W 0.0118853) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.165624 ± 0.00292406 (W 0.00197885) | 1.32487 ± 0.0162943 (W 0.0069595) | 🔵 faer-fork 8.00× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.199282 ± 0.00737905 (W 0.00172685) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 3.58209 ± 0.553576 (W 0.00743223) | 27.0047 ± 4.96011 (W 0.169482) | 🔵 faer-fork 7.54× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.147576 ± 0.00226933 (W 0.000601454) | 1.27701 ± 0.00896862 (W 0.00451306) | 🔵 faer-fork 8.65× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 30.5438 ± 1.7131 (W 0.0982045) | 118.797 ± 21.835 (W 0.233075) | 🔵 faer-fork 3.89× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 30.7542 ± 1.72553 (W 0.0899114) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.26383 ± 0.0373125 (W 0.00379156) | 3.95132 ± 0.0412522 (W 0.010844) | 🔵 faer-fork 3.13× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.2799 ± 0.036716 (W 0.00359976) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 19.4403 ± 1.20045 (W 0.0501639) | 61.3679 ± 10.1559 (W 0.343897) | 🔵 faer-fork 3.16× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.5023 ± 1.16877 (W 0.0687667) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.8432 ± 0.0193579 (W 0.00578398) | 2.17461 ± 0.0386531 (W 0.00910162) | 🔵 faer-fork 2.58× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.847113 ± 0.0240116 (W 0.00537706) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 150.393 ± 0.950544 (W 1.0545) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 224.694 ± 1.52619 (W 0.892876) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.66105 ± 0.00686208 (W 0.00539007) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.60273 ± 0.0435508 (W 0.014725) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 40.7284 ± 0.100651 (W 0.0563879) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 47.8612 ± 0.183081 (W 0.0738849) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.869887 ± 0.00269663 (W 0.0023526) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.989958 ± 0.0110041 (W 0.00405126) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 102.631 ± 0.322169 (W 0.175593) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 106.058 ± 0.473546 (W 0.388573) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.75801 ± 0.0100044 (W 0.00734122) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 2.52223 ± 0.0109841 (W 0.00597175) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 96.9504 ± 0.168365 (W 0.159238) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 2.59724 ± 0.00976019 (W 0.0096656) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 261.754 ± 0.682929 (W 0.465208) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 262.323 ± 0.679765 (W 6.5904) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.34936 ± 0.0136745 (W 0.012326) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.39759 ± 0.0245378 (W 0.0120272) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 100.843 ± 0.34949 (W 0.214587) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 100.814 ± 0.341653 (W 0.216242) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.37953 ± 0.00739885 (W 0.00439088) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.40909 ± 0.00761774 (W 0.00546867) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 47.0384 ± 0.262678 (W 0.135168) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 83.8598 ± 0.492093 (W 0.328457) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.730471 ± 0.00197531 (W 0.00269867) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.37744 ± 0.00217836 (W 0.00607915) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.4067 ± 0.0458585 (W 0.0375441) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 17.2271 ± 0.61155 (W 0.285355) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.293577 ± 0.00170235 (W 0.00210972) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.414392 ± 0.040932 (W 0.0431735) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 29.6613 ± 0.20863 (W 0.175939) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 31.828 ± 0.0660357 (W 0.068831) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.04255 ± 0.0015345 (W 0.00476951) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.867832 ± 0.00342958 (W 0.00232341) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.9258 ± 0.0805503 (W 0.0315295) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.978727 ± 0.00496529 (W 0.00377731) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 87.2492 ± 0.159851 (W 0.154596) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 87.8284 ± 0.210127 (W 0.11324) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.61172 ± 0.0149434 (W 0.00798588) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.62565 ± 0.00857884 (W 0.00896078) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 40.2263 ± 0.0774244 (W 0.0619316) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 40.2641 ± 0.114423 (W 0.071232) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.56093 ± 0.00504688 (W 0.00558396) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.56778 ± 0.00618366 (W 0.00413712) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 23.052 ± 10.2917 (W 0.0607073) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 56.2092 ± 16.6356 (W 0.37654) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.600907 ± 0.0492253 (W 0.00370084) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.21477 ± 0.175984 (W 0.00466791) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 6.42208 ± 0.182008 (W 0.0194258) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 11.8229 ± 0.252291 (W 0.0640373) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.200634 ± 0.00549646 (W 0.00134124) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.358332 ± 0.0938247 (W 0.00240768) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 17.4002 ± 0.589578 (W 0.0479527) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 19.4478 ± 0.976947 (W 0.0852435) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.766335 ± 0.0072291 (W 0.00464148) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.610453 ± 0.106258 (W 0.00262686) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 16.3993 ± 0.698183 (W 0.0593282) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.723768 ± 0.00731768 (W 0.0039633) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 62.2882 ± 2.35003 (W 0.161336) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 62.8648 ± 2.42288 (W 0.195026) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 2.22635 ± 0.143977 (W 0.00564103) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.25182 ± 0.144836 (W 0.0063344) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 31.4618 ± 1.61671 (W 0.0551306) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.5288 ± 1.56351 (W 0.0823174) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.45941 ± 0.0492188 (W 0.00557893) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.46826 ± 0.0522296 (W 0.00426815) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 254.601 ± 1.00596 (W 1.22618) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 221.09 ± 1.63456 (W 1.10114) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 2.52644 ± 0.0234028 (W 0.00972567) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.48098 ± 0.0212701 (W 0.010112) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 42.2717 ± 0.163254 (W 0.0711193) | 53.679 ± 0.091701 (W 0.0887192) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 43.2406 ± 0.242357 (W 0.0619128) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.870816 ± 0.00573823 (W 0.00591955) | 1.18141 ± 0.00527182 (W 0.00258325) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.895672 ± 0.00280701 (W 0.00343446) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 101.692 ± 0.421839 (W 0.530348) | 150.652 ± 0.847882 (W 0.419668) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 101.896 ± 0.239266 (W 0.399052) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.36857 ± 0.0107796 (W 0.00484651) | 3.08247 ± 0.0459531 (W 0.00428788) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 2.40787 ± 0.00884721 (W 0.00510663) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 95.7559 ± 0.347415 (W 0.312043) | 142.6 ± 0.762558 (W 1.17146) | 🔵 faer-fork 1.49× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 2.23142 ± 0.0089835 (W 0.00425955) | 2.91061 ± 0.0221076 (W 0.00577027) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 273.258 ± 0.352779 (W 0.460529) | 633.935 ± 1.67064 (W 1.55009) | 🔵 faer-fork 2.32× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 274.332 ± 0.763321 (W 5.92964) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.46147 ± 0.0337332 (W 0.00942912) | 11.261 ± 0.102659 (W 0.0218064) | 🔵 faer-fork 2.06× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.49942 ± 0.0162801 (W 0.00865858) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 113.244 ± 1.06932 (W 0.280301) | 280.966 ± 0.62093 (W 0.826337) | 🔵 faer-fork 2.48× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 113.625 ± 1.13732 (W 0.512879) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.51118 ± 0.00913204 (W 0.00672015) | 5.2131 ± 0.0686583 (W 0.0128152) | 🔵 faer-fork 2.08× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.53274 ± 0.0100789 (W 0.00466445) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 90.4215 ± 0.522807 (W 0.316867) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 79.211 ± 0.5839 (W 0.319724) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.23043 ± 0.00331379 (W 0.00379778) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.24952 ± 0.0047878 (W 0.00739054) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.437 ± 0.558218 (W 0.458804) | 28.5294 ± 0.0403998 (W 0.0267974) | 🔵 faer-fork 2.49× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 13.6228 ± 0.76211 (W 0.514433) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.258868 ± 0.0407504 (W 0.0194835) | 0.69329 ± 0.00555342 (W 0.00249446) | 🔵 faer-fork 2.68× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.379506 ± 0.0340292 (W 0.0415243) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 26.3291 ± 0.0451909 (W 0.0446104) | 57.8488 ± 0.845959 (W 0.245603) | 🔵 faer-fork 2.20× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.7738 ± 0.0869741 (W 0.0674111) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.708731 ± 0.00336523 (W 0.00208309) | 1.0913 ± 0.00245568 (W 0.00229874) | 🔵 faer-fork 1.54× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.767421 ± 0.00209933 (W 0.00327552) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 24.6906 ± 0.0993882 (W 0.0301293) | 53.2014 ± 0.891305 (W 0.418914) | 🔵 faer-fork 2.15× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 0.651648 ± 0.00330066 (W 0.00332489) | 0.964404 ± 0.00305654 (W 0.00286439) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 99.531 ± 0.130037 (W 0.125546) | 398.799 ± 1.46648 (W 0.453949) | 🔵 faer-fork 4.01× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 100.044 ± 0.119389 (W 0.210754) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.71936 ± 0.0272966 (W 0.00784572) | 9.08333 ± 0.0523676 (W 0.0225908) | 🔵 faer-fork 3.34× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.73384 ± 0.00751534 (W 0.00739685) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 52.2647 ± 0.104957 (W 0.0983436) | 197.486 ± 0.582641 (W 0.335954) | 🔵 faer-fork 3.78× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 52.449 ± 0.0880077 (W 0.0960784) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.65389 ± 0.00764737 (W 0.00499929) | 4.56895 ± 0.341085 (W 0.0189616) | 🔵 faer-fork 2.76× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.66115 ± 0.0032572 (W 0.00442559) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 53.5568 ± 16.0888 (W 0.149087) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 52.605 ± 16.4482 (W 0.290332) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 1.03322 ± 0.0591502 (W 0.00353704) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.08503 ± 0.0721659 (W 0.00345664) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 6.64962 ± 0.155062 (W 0.0144667) | 21.7487 ± 0.20467 (W 0.0278599) | 🔵 faer-fork 3.27× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 7.63114 ± 0.200199 (W 0.0149491) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.194514 ± 0.00303069 (W 0.00128893) | 0.570283 ± 0.0167986 (W 0.0033915) | 🔵 faer-fork 2.93× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.246465 ± 0.0110356 (W 0.00236925) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 14.0188 ± 1.03854 (W 0.191952) | 40.5866 ± 6.27145 (W 0.062828) | 🔵 faer-fork 2.90× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 15.3271 ± 0.952776 (W 0.0373578) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.434347 ± 0.0110379 (W 0.00256187) | 0.803507 ± 0.0175791 (W 0.00282925) | 🔵 faer-fork 1.85× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.502067 ± 0.0205404 (W 0.00278164) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 13.0824 ± 0.984015 (W 0.0267102) | 37.4946 ± 6.329 (W 0.167743) | 🔵 faer-fork 2.87× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.39591 ± 0.00893897 (W 0.00183936) | 0.678588 ± 0.0111281 (W 0.00323967) | 🔵 faer-fork 1.71× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 74.6842 ± 2.71672 (W 0.099524) | 359.648 ± 22.4218 (W 0.865832) | 🔵 faer-fork 4.82× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 75.3322 ± 2.69291 (W 0.0827407) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 2.34247 ± 0.201298 (W 0.00927599) | 9.00588 ± 0.450744 (W 0.0501095) | 🔵 faer-fork 3.84× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.36928 ± 0.210215 (W 0.00900061) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 43.4504 ± 2.04983 (W 0.111668) | 184.984 ± 10.6139 (W 0.484842) | 🔵 faer-fork 4.26× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 43.4274 ± 2.11671 (W 0.0610425) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.5693 ± 0.096482 (W 0.00600418) | 4.61013 ± 0.238016 (W 0.0174673) | 🔵 faer-fork 2.94× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.56234 ± 0.139377 (W 0.00431183) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 24.4166 ± 0.0323257 (W 0.0324981) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 38.5682 ± 0.269752 (W 0.181305) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.476479 ± 0.00217919 (W 0.00411078) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.968503 ± 0.00211375 (W 0.00370153) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.19491 ± 0.0112437 (W 0.0121129) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.29857 ± 0.0907555 (W 0.0895962) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.181482 ± 0.000912213 (W 0.00189412) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.34245 ± 0.00341902 (W 0.00309687) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 17.1556 ± 0.133905 (W 0.0338339) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 17.4178 ± 0.169502 (W 0.0739154) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.840595 ± 0.00435294 (W 0.00413856) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.607888 ± 0.00490755 (W 0.00362119) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.311 ± 0.137358 (W 0.0375321) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.815991 ± 0.00512497 (W 0.00588127) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 42.3285 ± 0.201965 (W 0.0425847) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 42.4126 ± 0.191991 (W 0.0803253) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.36019 ± 0.0207287 (W 0.0030269) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.36336 ± 0.0163157 (W 0.00323583) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 18.6474 ± 0.0921986 (W 0.0244906) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 18.7386 ± 0.0609728 (W 0.0298187) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.690302 ± 0.00293172 (W 0.00326569) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.691973 ± 0.00103071 (W 0.00267527) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 9.14682 ± 0.0321853 (W 0.0167247) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 19.3075 ± 0.160911 (W 0.144067) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.348877 ± 0.00279149 (W 0.00450151) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.621578 ± 0.00219673 (W 0.00479497) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.00087 ± 0.00697231 (W 0.00596429) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.90436 ± 0.0413659 (W 0.0868048) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.081282 ± 0.00071132 (W 0.000532511) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.175598 ± 0.00100101 (W 0.000583234) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 6.79465 ± 0.453224 (W 0.0239499) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 6.62315 ± 0.444376 (W 0.0844524) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.59445 ± 0.0035856 (W 0.00626435) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.365471 ± 0.0523586 (W 0.00636785) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 6.55623 ± 0.413573 (W 0.0217367) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.576013 ± 0.00380328 (W 0.0054259) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 18.52 ± 0.0476388 (W 0.0354137) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 18.6703 ± 0.0356435 (W 0.0260715) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.01798 ± 0.00519544 (W 0.00246729) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.02282 ± 0.00391174 (W 0.00309708) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 11.1559 ± 0.0619603 (W 0.0250753) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 11.187 ± 0.0402103 (W 0.0150203) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.638784 ± 0.000689551 (W 0.0024973) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.638204 ± 0.00235741 (W 0.00223037) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 6.75396 ± 0.240964 (W 0.0174697) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 16.1792 ± 2.47737 (W 0.0945369) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.355938 ± 0.00176508 (W 0.00322047) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.646128 ± 0.01175 (W 0.00973422) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 1.30671 ± 0.0370648 (W 0.00598674) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 3.15161 ± 0.752924 (W 0.07013) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.0665995 ± 0.000944312 (W 0.000819789) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.149869 ± 0.000758676 (W 0.000646263) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 5.22442 ± 0.137076 (W 0.022474) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 4.8423 ± 0.683803 (W 0.0640257) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.553778 ± 0.00532817 (W 0.00504405) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.322017 ± 0.0225739 (W 0.00587658) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 5.06919 ± 0.10053 (W 0.024996) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.534698 ± 0.00535194 (W 0.00509949) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 15.0297 ± 1.03566 (W 0.0577968) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 14.9944 ± 1.04526 (W 0.0431052) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 0.959296 ± 0.0114411 (W 0.00241303) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.966385 ± 0.0116781 (W 0.00299581) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 9.58237 ± 0.340724 (W 0.0167022) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 9.58382 ± 0.331566 (W 0.0352319) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.620741 ± 0.0038676 (W 0.00253793) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.629194 ± 0.00637882 (W 0.00282708) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 39.115 ± 1.9355 (W 0.0652303) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 36.316 ± 0.184423 (W 0.0619859) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.912597 ± 0.00488309 (W 0.00328704) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.885344 ± 0.239752 (W 0.002052) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 7.21874 ± 0.0316433 (W 0.0232155) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 7.30181 ± 0.0266215 (W 0.0162265) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.247266 ± 0.0018959 (W 0.00171855) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.253262 ± 0.00168406 (W 0.00222047) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 15.0401 ± 0.0651894 (W 0.0325915) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 15.1343 ± 0.0722952 (W 0.0365755) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.400277 ± 0.00174303 (W 0.00281354) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.488142 ± 0.00424208 (W 0.00356811) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 14.142 ± 0.109701 (W 0.0400313) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.372985 ± 0.00415672 (W 0.00291673) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 44.8695 ± 0.603039 (W 0.0597061) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 45.0231 ± 0.57965 (W 0.0339978) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.35856 ± 0.00374405 (W 0.00386085) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.36566 ± 0.004461 (W 0.00335007) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.672 ± 0.800862 (W 0.0507711) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.7051 ± 0.751487 (W 0.0300003) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.724684 ± 0.0063797 (W 0.00440189) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.7328 ± 0.00394875 (W 0.00345032) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.729 ± 0.050952 (W 0.0288746) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 17.4292 ± 0.0643166 (W 0.028368) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.551821 ± 0.00562363 (W 0.00799872) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.539486 ± 0.00247748 (W 0.00453841) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.96772 ± 0.0118265 (W 0.00432057) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.19752 ± 0.00788958 (W 0.00813835) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0784365 ± 0.000152927 (W 0.000285309) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.0900225 ± 0.000428656 (W 0.000426676) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 4.07968 ± 0.0167351 (W 0.00822903) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.41139 ± 0.00908259 (W 0.00694167) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.160234 ± 0.00064945 (W 0.00176002) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.237868 ± 0.00263356 (W 0.00693575) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.82913 ± 0.0164077 (W 0.00527829) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.145937 ± 0.000446211 (W 0.000234054) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 21.3606 ± 0.0740422 (W 0.0396575) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 21.3708 ± 0.0822202 (W 0.0333514) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.01471 ± 0.00267445 (W 0.00199955) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.02403 ± 0.00191664 (W 0.00262578) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 14.3319 ± 0.0418304 (W 0.0249747) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 14.3443 ± 0.0726358 (W 0.0247798) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.686444 ± 0.00367278 (W 0.00765291) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.687999 ± 0.00278783 (W 0.00242254) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 14.3848 ± 1.87197 (W 0.0372351) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 14.2931 ± 1.95815 (W 0.0246732) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.554239 ± 0.00509093 (W 0.00764576) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.554828 ± 0.00410191 (W 0.00818197) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.10381 ± 0.0185821 (W 0.00519736) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.39225 ± 0.0223528 (W 0.00367651) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.051 ± 0.000304605 (W 0.000546097) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.0653065 ± 0.00341852 (W 0.00112277) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 2.28525 ± 0.174953 (W 0.0077144) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 2.60218 ± 0.129824 (W 0.00685194) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.123385 ± 0.00100856 (W 0.00243458) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.203786 ± 0.00720785 (W 0.0054746) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 2.12415 ± 0.131774 (W 0.00562185) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.10967 ± 0.00116232 (W 0.000170233) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 17.5935 ± 2.3655 (W 0.047766) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 17.725 ± 2.35519 (W 0.048889) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 0.960578 ± 0.0194933 (W 0.00440605) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.970077 ± 0.0231499 (W 0.00212728) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 12.7642 ± 0.737376 (W 0.0321742) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 12.7957 ± 0.766065 (W 0.0189698) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.665107 ± 0.00624851 (W 0.00336495) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.670683 ± 0.0121678 (W 0.00398121) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 58.3329 ± 0.569442 (W 0.181007) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 91.2362 ± 0.86662 (W 0.520387) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.891935 ± 0.00259459 (W 0.00239284) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.50431 ± 0.00487723 (W 0.00506083) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 22.0836 ± 0.0202316 (W 0.0268572) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 26.7328 ± 0.0688449 (W 0.0699118) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.539705 ± 0.0033482 (W 0.0032828) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.676435 ± 0.00324335 (W 0.00313842) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 54.9737 ± 0.131986 (W 0.189763) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 58.3896 ± 0.127584 (W 0.352626) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.64356 ± 0.156411 (W 0.00310337) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 1.47165 ± 0.00591476 (W 0.00511118) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 52.171 ± 0.285524 (W 0.122059) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.5616 ± 0.00624531 (W 0.00454707) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 135.576 ± 0.152262 (W 0.359262) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 135.916 ± 0.2942 (W 0.187565) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.04941 ± 0.00544564 (W 0.00592523) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.0547 ± 0.00396723 (W 0.00539614) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 54.2945 ± 0.102043 (W 0.107795) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 54.2391 ± 0.090786 (W 0.0528334) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.42974 ± 0.00280431 (W 0.00480087) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.42852 ± 0.00439235 (W 0.00431034) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.1019 ± 0.0350515 (W 0.027323) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 37.212 ± 0.135863 (W 0.177354) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.489878 ± 0.00312769 (W 0.00376691) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.838703 ± 0.00276461 (W 0.00449914) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.27789 ± 0.026762 (W 0.0124282) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.91448 ± 0.0589828 (W 0.0942018) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.199303 ± 0.00217006 (W 0.00172012) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.299044 ± 0.000652797 (W 0.00197824) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 16.9061 ± 0.567583 (W 0.0276606) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 18.0205 ± 0.0353597 (W 0.057416) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.745637 ± 0.00208477 (W 0.00427546) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.57046 ± 0.00242453 (W 0.00437639) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.0513 ± 0.0354042 (W 0.0530463) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.71086 ± 0.00377608 (W 0.00357355) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 48.2723 ± 0.088953 (W 0.105518) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 48.4771 ± 0.111084 (W 0.0918705) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.66371 ± 0.00352302 (W 0.00482851) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.67094 ± 0.00458772 (W 0.00577715) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 24.048 ± 0.0513916 (W 0.0348433) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.0861 ± 0.0302492 (W 0.0314895) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.00995 ± 0.00412702 (W 0.00253251) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.00916 ± 0.00768306 (W 0.00436963) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.7979 ± 2.30988 (W 0.0706612) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 30.1101 ± 4.12839 (W 0.0675551) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.450902 ± 0.00657163 (W 0.0019623) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.757438 ± 0.0747865 (W 0.0026323) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 3.60399 ± 0.326715 (W 0.0103916) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 7.00364 ± 0.519277 (W 0.045823) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.152329 ± 0.00159457 (W 0.00146563) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.242217 ± 0.024029 (W 0.00239345) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 10.6665 ± 0.335592 (W 0.0285472) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 11.2716 ± 0.897733 (W 0.0603011) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.593723 ± 0.00494639 (W 0.00387659) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.42621 ± 0.0419225 (W 0.00293936) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 10.2252 ± 0.282287 (W 0.0256412) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.569222 ± 0.00315882 (W 0.00376281) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 35.3109 ± 1.32163 (W 0.134439) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 35.4536 ± 1.4166 (W 0.0955798) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.46689 ± 0.0410893 (W 0.00613232) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.47265 ± 0.037418 (W 0.00461047) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 18.9888 ± 0.949189 (W 0.0589168) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.0688 ± 0.928142 (W 0.0348286) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.953438 ± 0.0183116 (W 0.00346953) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.953858 ± 0.0182074 (W 0.00243662) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 96.9329 ± 0.835496 (W 0.525407) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 88.199 ± 0.90284 (W 0.357093) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.41278 ± 0.00239863 (W 0.00344344) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.39234 ± 0.0234712 (W 0.00421612) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 22.8408 ± 0.0233336 (W 0.0421082) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 23.324 ± 0.0538928 (W 0.0638643) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.554027 ± 0.00354696 (W 0.00301307) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.569052 ± 0.00228483 (W 0.00301477) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 54.0044 ± 0.0694962 (W 0.231285) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 54.9093 ± 0.129564 (W 0.152617) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.33523 ± 0.00297808 (W 0.00368018) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 1.35852 ± 0.00233578 (W 0.00405233) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 50.9984 ± 0.134892 (W 0.0808304) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.25423 ± 0.00471473 (W 0.00382666) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 140.969 ± 0.966528 (W 0.298818) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 141.123 ± 0.0953611 (W 0.241799) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.10825 ± 0.00523264 (W 0.00725871) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.11962 ± 0.00519654 (W 0.00789894) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 59.7052 ± 0.114619 (W 0.0918568) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 59.7684 ± 0.0796296 (W 0.106018) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.51611 ± 0.0103219 (W 0.00865099) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.5123 ± 0.00537196 (W 0.00589724) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 35.354 ± 0.127016 (W 0.0761749) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 34.0115 ± 0.0588048 (W 0.0791547) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.702079 ± 0.00336489 (W 0.00431148) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.724122 ± 0.0025302 (W 0.00488655) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.10248 ± 0.0161627 (W 0.0109047) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.55343 ± 0.00971961 (W 0.0148338) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.168214 ± 0.000946346 (W 0.00162671) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.189798 ± 0.00133654 (W 0.0018188) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 13.9935 ± 0.0515348 (W 0.0388119) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 14.6426 ± 0.122191 (W 0.0399176) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.426825 ± 0.00189738 (W 0.00218076) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.460049 ± 0.00237256 (W 0.00231999) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.1781 ± 0.0287274 (W 0.0299) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.391485 ± 0.00117197 (W 0.00238857) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 53.5529 ± 0.0952901 (W 0.0877149) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 53.7658 ± 0.085492 (W 0.0958215) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.70494 ± 0.00370328 (W 0.00605087) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.71436 ± 0.00509487 (W 0.0048696) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 29.6618 ± 0.0751446 (W 0.0467666) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 29.6509 ± 0.023706 (W 0.0393793) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.07596 ± 0.00422195 (W 0.00517457) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.07726 ± 0.00851798 (W 0.00392456) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 27.2924 ± 3.92893 (W 0.0785784) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 27.1379 ± 4.22758 (W 0.0559811) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.616166 ± 0.0185967 (W 0.00398164) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.641838 ± 0.0149958 (W 0.00308636) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 3.37273 ± 0.103779 (W 0.0094813) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 3.74468 ± 0.126834 (W 0.00824116) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.104816 ± 0.000891367 (W 0.00163628) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.131866 ± 0.00499035 (W 0.000798666) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 7.37352 ± 0.551491 (W 0.019106) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 7.9094 ± 0.623356 (W 0.0325094) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.27588 ± 0.00312648 (W 0.00262264) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.31333 ± 0.00874105 (W 0.00211643) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 6.90201 ± 0.520538 (W 0.0142593) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.251384 ± 0.00322904 (W 0.00167982) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 40.6103 ± 1.73657 (W 0.0584751) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 40.7487 ± 1.67972 (W 0.12093) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.51022 ± 0.0474279 (W 0.00646045) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.524 ± 0.121295 (W 0.00753226) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 24.6287 ± 1.20443 (W 0.121813) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.6285 ± 1.60687 (W 0.0909965) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.02389 ± 0.0241256 (W 0.00510828) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.03506 ± 0.0293895 (W 0.00414185) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `lu/256x4096`, `lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `solve/256x4096`, `solve/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
