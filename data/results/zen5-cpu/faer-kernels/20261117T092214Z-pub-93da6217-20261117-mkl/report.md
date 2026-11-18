# `faer-kernels` on `zen5-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above); `faer-upstream` (the provider in the measured revision; see library and upstream above)
- harness commit: `cf51a3ea12ba4d187cf8188d7619578e4cd3dbef`
- toolkit commit: `b7299cf625c537a37eef2782e8759445f1d8aef8`
- measured binary SHA-256: `f4f9a0ba5346b33e547051da3309d6bfc806a9f1f6e6bff219f2170492031268`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-17T13:34:55.817006Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `/home/shinaoka/projects/tensor4all/colmat-benchmark/scripts/record_run.py zen5-cpu faer-kernels --checkout /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/t4a-faer --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor mkl --engines faer-fork,faer-upstream,lapack-mkl --reps 5 --aa 3 --prime-ms 250 --label pub-93da6217-20261117-mkl --no-publish --threads 1,4,8`
- raw data: `data/results/zen5-cpu/faer-kernels/20261117T092214Z-pub-93da6217-20261117-mkl/`

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
  - `faer-upstream` — Upstream `faer` at the tag the fork is based on, in the same binary as the fork row. The baseline that says whether a fork change moved the number.
    Identity: the upstream revision in this run's manifest (`upstream.commit`), pinned by `pins/upstream-faer.rev`
    Caveat: Same crate-name and feature-unification caveat as `faer-fork`. A change to upstream at a fixed fork revision produces a different page rather than overwriting one.
    As measured: `faer-upstream` (the provider in the measured revision; see library and upstream above)
  - `lapack-mkl` — The same calls against Intel oneAPI MKL, linked from the installation `MKLROOT` names. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run; no page carries two vendor arms. The LP64 interface layer and the threading layer are pinned by the harness before the first MKL call, because MKL otherwise lets the environment choose them and the interface layer is an ABI.
    Identity: the MKL build recorded in this run's manifest (`providers[mkl]`)
    Caveat: MKL reports no dispatched-kernel name and no processor count, so its manifest records neither rather than inferring them. Its threading is vendor-owned, as for OpenBLAS.
    As measured: linked shared; mkl 2026.2; kernel not reported by the vendor

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
Each process retains 5 chronological calls; 3 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 0.80%, worst 185.08% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 401 of 9408 engine cells exceed 5% and carry `!`; median within-process relative SD 0.48%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002365 ± 1.09697e-05 (W 7.07616e-05) | 0.002354 ± 2.94449e-05 (W 7.84047e-05) | 0.004478 ± 1.52753e-05 (W 6.61151e-05) | · | 🔵 faer-fork 1.89× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.017092 ± 0.000116363 (W 4.67675e-05) | 0.017223 ± 3.5171e-05 (W 4.54973e-05) | 0.024335 ± 1.73205e-05 (W 0.000166452) | · | 🔵 faer-fork 1.42× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.056875 ± 0.000156273 (W 0.000103419) | 0.056865 ± 0.000321312 (W 0.000103171) | 0.063098 ± 0.000215144 (W 0.00192315) | · | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |

## gemm-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002435 ± 2.47049e-05 (W 2.58109e-05) | 0.003997 ± 9.11062e-05 (W 0.000284037) | 0.005139 ± 3.15753e-05 (W 0.000177371) | 🔵 faer-fork 1.64× faster than `faer-upstream` | 🔵 faer-fork 2.11× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01116 ± 0.000155393 (W 0.000329766) | 0.019266 ± 0.000991037 (W 0.00133607) | 0.013746 ± 4.63717e-05 (W 0.000133874) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.022952 ± 0.000151163 (W 0.000286725) | 0.062997 ± 0.000626104 (W 0.0130862) | 0.022291 ± 8.05005e-05 (W 0.000122038) | 🔵 faer-fork 2.74× faster than `faer-upstream` | · |  |

## gemm-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002435 ± 2.60832e-05 (W 2.46232e-05) | 0.004067 ± 0.000148987 (W 0.000234771) | 0.00494 ± 8.49137e-05 (W 0.000192597) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 2.03× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.009528 ± 5.54106e-05 (W 6.47248e-05) | 0.010579 ± 5.87963e-05 (W 0.000149108) | 0.009949 ± 0.000115052 (W 0.000174342) | 🔵 faer-fork 1.11× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020118 ± 0.000176116 (W 0.000107098) | 0.021239 ± 0.000226959 (W 0.000274171) | 0.014867 ± 6.22923e-05 (W 0.000133219) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔴 faer-fork 1.35× slower than `lapack-mkl` |  |

## gemm-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002375 ± 5.03322e-05 (W 1.82291e-05) | 0.002324 ± 1.58219e-05 (W 3.33062e-05) | 0.004459 ± 2.09841e-05 (W 0.000117453) | · | 🔵 faer-fork 1.88× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.017081 ± 2.09841e-05 (W 8.10327e-05) | 0.017282 ± 4.04145e-05 (W 4.43869e-05) | 0.024236 ± 3.20052e-05 (W 0.000376458) | · | 🔵 faer-fork 1.42× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.056837 ± 8.24763e-05 (W 5.18488e-05) | 0.057105 ± 0.000342931 (W 0.00149331) | 0.062826 ± 5.98693e-05 (W 0.000134276) | · | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |

## gemm-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002425 ± 5.7735e-06 (W 1.82016e-05) | 0.003848 ± 9.46942e-05 (W 0.000227314) | 0.00508 ± 0.000170354 (W 0.000150775) | 🔵 faer-fork 1.59× faster than `faer-upstream` | 🔵 faer-fork 2.09× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.011241 ± 0.000176693 (W 9.46478e-05) | 0.019256 ± 0.000329819 (W 0.000883455) | 0.013826 ± 0.000115803 (W 0.000176401) | 🔵 faer-fork 1.71× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.022902 ± 0.000122094 (W 0.00039518) | 0.064218 ± 0.000537751 (W 0.00165191) | 0.022352 ± 0.000151461 (W 0.000182416) | 🔵 faer-fork 2.80× faster than `faer-upstream` | · |  |

## gemm-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002415 ± 1.12694e-05 (W 3.24854e-05) | 0.004308 ± 0.000148987 (W 0.000261873) | 0.005079 ± 0.000142011 (W 0.000169467) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 2.10× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.009548 ± 3.27159e-05 (W 8.42449e-05) | 0.010791 ± 0.000153353 (W 0.000261695) | 0.009999 ± 0.00012454 (W 0.000228371) | 🔵 faer-fork 1.13× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.019947 ± 5.50757e-05 (W 8.86471e-05) | 0.02108 ± 0.000216444 (W 0.000222359) | 0.014928 ± 0.000101933 (W 0.000193375) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔴 faer-fork 1.34× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.008837 ± 2.02567e-05 (W 2.39228e-05) | 0.008877 ± 3.84318e-05 (W 9.87639e-05) | 0.011191 ± 6.55744e-05 (W 0.000145842) | · | 🔵 faer-fork 1.27× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.068277 ± 0.000336705 (W 0.000133492) | 0.067845 ± 2.10792e-05 (W 7.6467e-05) | 0.082794 ± 0.00102733 (W 0.00130352) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.227285 ± 0.00056943 (W 0.00226528) | 0.227212 ± 0.000487784 (W 0.00154591) | 0.275111 ± 0.000640115 (W 0.00427766) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |

## gemm-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009278 ± 7.91854e-05 (W 5.80026e-05) | 0.008856 ± 2.00083e-05 (W 2.73551e-05) | 0.014096 ± 0.00012454 (W 0.000151156) | · | 🔵 faer-fork 1.52× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.029094 ± 0.000186815 (W 0.001256) | 0.040836 ± 0.000711586 (W 0.00125957) | 0.035606 ± 4.99032e-05 (W 0.000386327) | 🔵 faer-fork 1.40× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.078466 ± 0.000731059 (W 0.000335385) | 0.097971 ± 0.000810462 (W 0.0018968) | 0.098593 ± 0.000376816 (W 0.00109976) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |

## gemm-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009719 ± 0.000106068 (W 0.000126768) | 0.008857 ± 4.74377e-05 (W 3.25914e-05) | 0.015008 ± 0.000646608 (W 0.000807) | 🔴 faer-fork 1.10× slower than `faer-upstream` | 🔵 faer-fork 1.54× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.026559 ± 7.63763e-05 (W 5.249e-05) | 0.027882 ± 0.000215484 (W 0.00017648) | 0.016821 ± 4.39356e-05 (W 0.000409261) | · | 🔴 faer-fork 1.58× slower than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.039603 ± 0.00022316 (W 0.000129547) | 0.041076 ± 0.00057438 (W 0.000371711) | 0.042839 ± 0.000102504 (W 0.000591449) | · | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |

## gemm-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.008816 ± 2.08167e-05 (W 2.28736e-05) | 0.008906 ± 4.57202e-05 (W 9.85759e-05) | 0.011111 ± 0.000141774 (W 0.000254932) | · | 🔵 faer-fork 1.26× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.067927 ± 0.000220061 (W 0.000117388) | 0.067925 ± 0.000179984 (W 0.000135869) | 0.081932 ± 0.000413349 (W 0.00224424) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.227266 ± 0.00019466 (W 0.00221608) | 0.227212 ± 0.00143731 (W 0.00244037) | 0.272456 ± 0.00274155 (W 0.0033961) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009338 ± 0.00030867 (W 7.47549e-05) | 0.008847 ± 1.70392e-05 (W 4.55555e-05) | 0.013986 ± 4.3589e-05 (W 0.000140961) | 🔴 faer-fork 1.06× slower than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.029034 ± 4.16333e-05 (W 0.00017254) | 0.039844 ± 0.000882439 (W 0.000999284) | 0.035697 ± 0.00017436 (W 0.000230129) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.078445 ± 9.05373e-05 (W 0.000294365) | 0.096571 ± 0.00219154 (W 0.00102169) | 0.098855 ± 0.000710417 (W 0.000886112) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |

## gemm-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009678 ± 8.14453e-05 (W 6.90014e-05) | 0.008846 ± 3.52184e-05 (W 8.10555e-05) | 0.015529 ± 0.000578414 (W 0.000817271) | 🔴 faer-fork 1.09× slower than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.02662 ± 0.000184906 (W 8.23408e-05) | 0.027772 ± 0.000167482 (W 0.000218831) | 0.016491 ± 6.1101e-05 (W 0.000175406) | · | 🔴 faer-fork 1.61× slower than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.039743 ± 0.000422143 (W 5.55275e-05) | 0.040716 ± 0.000188813 (W 0.000125874) | 0.042971 ± 0.000228204 (W 0.000266063) | · | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |

## gemm-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001272 ± 6.08276e-06 (W 1.50433e-05) | 0.001252 ± 1.48436e-05 (W 8.437e-05) | 0.001733 ± 1.09697e-05 (W 0.00010902) | · | 🔵 faer-fork 1.36× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.008646 ± 6.35085e-05 (W 4.76046e-05) | 0.008626 ± 1.1547e-05 (W 8.1561e-05) | 0.011151 ± 0.000168625 (W 0.000217953) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.028514 ± 0.000104165 (W 0.000123034) | 0.028483 ± 3.21455e-05 (W 3.32611e-05) | 0.036087 ± 0.000637778 (W 0.000175269) | · | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |

## gemm-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001322 ± 1.53948e-05 (W 2.21427e-05) | 0.002815 ± 0.000126623 (W 0.00012813) | 0.003477 ± 0.000117758 (W 0.00015057) | 🔵 faer-fork 2.13× faster than `faer-upstream` | 🔵 faer-fork 2.63× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.006311 ± 0.000231389 (W 0.000333204) | 0.009698 ± 0.000873249 (W 0.0013234) | 0.008606 ± 0.00012897 (W 0.000164676) | 🔵 faer-fork 1.54× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.012924 ± 0.000148561 (W 0.000238507) | 0.03195 ± 0.000427606 (W 0.00085858) | 0.015789 ± 9.53939e-05 (W 0.000182601) | 🔵 faer-fork 2.47× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001342 ± 6.08276e-06 (W 3.25008e-05) | 0.003557 ± 0.000160053 (W 0.00040058) | 0.003928 ± 0.000111447 (W 0.000200721) | 🔵 faer-fork 2.65× faster than `faer-upstream` | 🔵 faer-fork 2.93× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.005309 ± 1.53948e-05 (W 4.65328e-05) | 0.006422 ± 8.7178e-05 (W 0.000233785) | 0.00547 ± 8.03181e-05 (W 0.000174377) | 🔵 faer-fork 1.21× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.01067 ± 7e-05 (W 8.96103e-05) | 0.011973 ± 0.00012454 (W 0.000212204) | 0.01081 ± 6.80686e-05 (W 0.000218118) | 🔵 faer-fork 1.12× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001262 ± 1e-05 (W 2.41309e-05) | 0.001242 ± 5.7735e-06 (W 2.94958e-05) | 0.001713 ± 2.08167e-05 (W 9.37459e-05) | · | 🔵 faer-fork 1.36× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.008646 ± 1.18462e-05 (W 2.40832e-05) | 0.008626 ± 1.52753e-05 (W 9.1214e-05) | 0.011281 ± 8.73689e-05 (W 0.000108074) | · | 🔵 faer-fork 1.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.028503 ± 2.04206e-05 (W 8.34865e-05) | 0.028573 ± 5.56776e-05 (W 9.67016e-05) | 0.036257 ± 0.000591498 (W 0.000719409) | · | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |

## gemm-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001333 ± 2.51064e-05 (W 2.16402e-05) | 0.002855 ± 0.000162543 (W 0.000212415) | 0.003427 ± 0.000185338 (W 0.000220894) | 🔵 faer-fork 2.14× faster than `faer-upstream` | 🔵 faer-fork 2.57× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.006191 ± 6.56531e-05 (W 0.000233258) | 0.009768 ± 0.000332591 (W 0.00109625) | 0.008576 ± 0.000202567 (W 0.000186495) | 🔵 faer-fork 1.58× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.013094 ± 0.000143643 (W 0.000213612) | 0.03217 ± 0.000239166 (W 0.001369) | 0.015769 ± 8.52467e-05 (W 0.000149977) | 🔵 faer-fork 2.46× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001313 ± 1.70392e-05 (W 2.41309e-05) | 0.003347 ± 0.000227423 (W 0.000308376) | 0.003948 ± 7.22519e-05 (W 0.000227379) | 🔵 faer-fork 2.55× faster than `faer-upstream` | 🔵 faer-fork 3.01× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0054 ± 4.72582e-05 (W 7.47543e-05) | 0.006402 ± 0.000107858 (W 0.00025408) | 0.005591 ± 4.04145e-05 (W 0.000216241) | 🔵 faer-fork 1.19× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.01071 ± 5.50757e-05 (W 7.33846e-05) | 0.012013 ± 0.000198016 (W 0.000163793) | 0.01084 ± 6.76782e-05 (W 0.000175813) | 🔵 faer-fork 1.12× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.004529 ± 1.48436e-05 (W 8.00831e-05) | 0.004509 ± 3.25115e-05 (W 3.54866e-05) | 0.00559 ± 7.55447e-05 (W 9.92184e-05) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.034033 ± 0.000170978 (W 8.60233e-05) | 0.034233 ± 0.000113359 (W 4.86826e-05) | 0.041317 ± 0.0003505 (W 0.000528201) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.113531 ± 0.000447828 (W 0.000154765) | 0.113543 ± 7.06918e-05 (W 0.000327884) | 0.134418 ± 0.00237621 (W 0.00253505) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |

## gemm-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.004949 ± 0.000135769 (W 0.000204769) | 0.006142 ± 3.78594e-05 (W 0.000184701) | 0.0053 ± 7.13933e-05 (W 0.000138728) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.019806 ± 0.000160032 (W 7.99719e-05) | 0.027041 ± 0.000532962 (W 0.00213822) | 0.019045 ± 0.000132855 (W 0.00023194) | 🔵 faer-fork 1.37× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.041608 ± 0.000145661 (W 0.000195499) | 0.079117 ± 0.000751583 (W 0.00131332) | 0.050925 ± 0.000127017 (W 0.000661084) | 🔵 faer-fork 1.90× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |

## gemm-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00508 ± 0.000272116 (W 4.99229e-05) | 0.006342 ± 0.000148436 (W 0.0003095) | 0.004428 ± 0.000214189 (W 0.000154855) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔴 faer-fork 1.15× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.018024 ± 0.000155017 (W 6.37519e-05) | 0.018895 ± 0.000191364 (W 0.000345366) | 0.009678 ± 0.000104403 (W 0.000141468) | · | 🔴 faer-fork 1.86× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.039103 ± 0.000104403 (W 9.27534e-05) | 0.039995 ± 0.000336537 (W 0.000384552) | 0.022932 ± 6.09945e-05 (W 0.000248892) | · | 🔴 faer-fork 1.71× slower than `lapack-mkl` |  |

## gemm-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.004518 ± 4.56107e-05 (W 8.26638e-05) | 0.004508 ± 5.7735e-07 (W 7.80276e-05) | 0.00554 ± 5.18684e-05 (W 0.000169184) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.034013 ± 0.000155907 (W 0.000118828) | 0.034103 ± 1.42945e-05 (W 5.61249e-05) | 0.041468 ± 7.35323e-05 (W 0.000661749) | · | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.113571 ± 0.000365752 (W 0.00163092) | 0.113569 ± 0.000148624 (W 0.00165883) | 0.135451 ± 0.00162697 (W 0.00198489) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |

## gemm-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00505 ± 6.40859e-05 (W 0.000168016) | 0.006011 ± 4.16333e-05 (W 8.73001e-05) | 0.00536 ± 8.7178e-05 (W 0.00016947) | 🔵 faer-fork 1.19× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.019947 ± 0.000235018 (W 0.00016) | 0.026068 ± 0.000460937 (W 0.00223105) | 0.019046 ± 7.84241e-05 (W 0.000259067) | 🔵 faer-fork 1.31× faster than `faer-upstream` | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.041737 ± 0.000129114 (W 9.54919e-05) | 0.079168 ± 0.000626254 (W 0.00148974) | 0.050353 ± 0.000509403 (W 0.000594511) | 🔵 faer-fork 1.90× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |

## gemm-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005129 ± 0.000194243 (W 0.000170822) | 0.006442 ± 0.000156462 (W 0.000172047) | 0.004358 ± 0.000191364 (W 0.000133627) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔴 faer-fork 1.18× slower than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.018044 ± 0.000120968 (W 0.000101982) | 0.019155 ± 0.000191251 (W 0.000402086) | 0.009708 ± 0.000120652 (W 0.000197435) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔴 faer-fork 1.86× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.039003 ± 9.0185e-05 (W 6.2915e-05) | 0.040244 ± 0.000221811 (W 0.000285328) | 0.023072 ± 0.000307747 (W 0.000171027) | · | 🔴 faer-fork 1.69× slower than `lapack-mkl` |  |

## gemm-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.133809 ± 0.000344109 (W 0.00169031) | 0.133119 ± 0.000435179 (W 0.00176007) | 0.165706 ± 0.00129773 (W 0.00267208) | · | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.454905 ± 0.00153307 (W 0.00261) | 0.456124 ± 0.00304458 (W 0.00324441) | 0.485689 ± 0.00230248 (W 0.00324906) | · | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 1.0778 ± 0.00407643 (W 0.00319292) | 1.07792 ± 0.00383494 (W 0.00158391) | 1.20297 ± 0.0101563 (W 0.00284969) | · | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 3.62444 ± 0.00753372 (W 0.00962856) | 3.61823 ± 0.0139864 (W 0.0081759) | 3.73948 ± 0.00414422 (W 0.00619926) | · | · |  |
| `gemm/512x512x512/b1` | 8.62357 ± 0.0135818 (W 0.0111643) | 8.62675 ± 0.028243 (W 0.0198136) | 9.83828 ± 0.0800372 (W 0.0308351) | · | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 29.4686 ± 0.0628629 (W 0.0376748) | 29.4461 ± 0.083001 (W 0.387676) | 30.3893 ± 0.252746 (W 0.0510038) | · | · |  |

## gemm-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.045415 ± 0.000294738 (W 0.000173569) | 0.075079 ± 0.000894133 (W 0.0117556) | 0.054761 ± 0.000252866 (W 0.0002297) | 🔵 faer-fork 1.65× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.120625 ± 0.000763366 (W 0.00152467) | 0.159987 ± 0.00381403 (W 0.00481728) | 0.137185 ± 0.00197668 (W 0.000408578) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.29051 ± 0.000228771 (W 0.00181341) | 0.305285 ± 0.00827977 (W 0.00710387) | 0.343759 ± 0.000737101 (W 0.00187432) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.917966 ± 0.00460285 (W 0.00216572) | 0.915272 ± 0.00376159 (W 0.0150968) | 1.00515 ± 0.00208377 (W 0.00223105) | · | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.21497 ± 0.00235655 (W 0.0042075) | 2.21404 ± 0.00479054 (W 0.00735983) | 3.03361 ± 0.0231749 (W 0.00561973) | · | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 7.51837 ± 0.0122165 (W 0.0125465) | 7.53823 ± 0.015524 (W 0.0130635) | 8.49043 ± 0.00667437 (W 0.0150022) | · | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |

## gemm-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.03223 ± 0.000965869 (W 0.0012445) | 0.110225 ± 0.0248133 (W 0.0186381) | 0.03218 ± 0.000207445 (W 0.000142778) | 🔵 faer-fork 3.42× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.086301 ± 0.000232509 (W 0.000495771) | 0.205722 ± 0.00704969 (W 0.0158419) | 0.074629 ± 0.000398954 (W 0.00170805) | 🔵 faer-fork 2.38× faster than `faer-upstream` | 🔴 faer-fork 1.16× slower than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.165076 ± 0.000493119 (W 0.00155339) | 0.250925 ± 0.0270763 (W 0.0273488) | 0.178534 ± 0.000447595 (W 0.000853637) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.481031 ± 0.00156809 (W 0.00262868) | 0.559816 ± 0.00350909 (W 0.00630852) | 0.513872 ± 0.00259746 (W 0.00272322) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.14193 ± 0.00628064 (W 0.00206022) | 1.18865 ± 0.00427212 (W 0.0090792) | 1.53186 ± 0.00460582 (W 0.0108623) | · | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 3.88294 ± 0.0149959 (W 0.00957705) | 3.89385 ± 0.0193426 (W 0.0109873) | 4.65392 ± 0.0120924 (W 0.023028) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.133911 ± 0.000381371 (W 0.00248638) | 0.133809 ± 0.000545078 (W 0.00181509) | 0.16669 ± 0.000839797 (W 0.00205421) | · | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.453939 ± 0.000217314 (W 0.00303842) | 0.453348 ± 0.000934367 (W 0.00229667) | 0.484005 ± 0.000387962 (W 0.00216146) | · | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 1.07115 ± 0.00376476 (W 0.00345769) | 1.07277 ± 0.00414386 (W 0.00158913) | 1.20197 ± 0.00126712 (W 0.00317131) | · | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 3.62512 ± 0.0136628 (W 0.00848767) | 3.61704 ± 0.00596464 (W 0.00684618) | 3.7377 ± 0.027245 (W 0.00650528) | · | · |  |
| `gemm/512x512x512/b1` | 8.62644 ± 0.0385119 (W 0.0171063) | 8.61919 ± 0.00830785 (W 0.0254011) | 9.91348 ± 0.0317862 (W 0.0126821) | · | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 29.4792 ± 0.108336 (W 0.0392151) | 29.4286 ± 0.207537 (W 0.0643147) | 30.6073 ± 0.107915 (W 0.0351526) | · | · |  |

## gemm-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.045314 ± 0.000228685 (W 0.000219174) | 0.080489 ± 0.00698184 (W 0.0125649) | 0.055142 ± 0.000452335 (W 0.000188874) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.119854 ± 0.000832514 (W 0.00174951) | 0.160127 ± 0.00144475 (W 0.00332857) | 0.137555 ± 0.000603242 (W 0.00171994) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.290744 ± 0.000672584 (W 0.00143848) | 0.305004 ± 0.00850956 (W 0.00688363) | 0.344685 ± 0.000638788 (W 0.00291964) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.917997 ± 0.00218647 (W 0.00158979) | 0.916484 ± 0.00265616 (W 0.00357968) | 1.00581 ± 0.000804003 (W 0.00205121) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.21284 ± 0.0024896 (W 0.00487783) | 2.21438 ± 0.00715742 (W 0.00485994) | 3.02288 ± 0.033471 (W 0.00862344) | · | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 7.51805 ± 0.00633424 (W 0.0207952) | 7.53188 ± 0.00972498 (W 0.0100094) | 8.47992 ± 0.012374 (W 0.0251394) | · | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |

## gemm-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.03231 ± 0.00114846 (W 0.000909163) | 0.112048 ± 0.0222343 (W 0.0225947) | 0.03213 ± 8.5967e-05 (W 0.000286598) | 🔵 faer-fork 3.47× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.086821 ± 0.000394812 (W 0.00100177) | 0.192981 ± 0.00559201 (W 0.0125458) | 0.07494 ± 0.00048814 (W 0.000180056) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔴 faer-fork 1.16× slower than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.167471 ± 0.00158281 (W 0.00212486) | 0.251507 ± 0.0307406 (W 0.0376683) | 0.178762 ± 0.00200692 (W 0.00187011) | 🔵 faer-fork 1.50× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.479306 ± 0.00102019 (W 0.00298599) | 0.561174 ± 0.00283687 (W 0.00465926) | 0.514724 ± 0.00244948 (W 0.00246505) | 🔵 faer-fork 1.17× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.13803 ± 0.00487604 (W 0.0031354) | 1.18425 ± 0.00792522 (W 0.00895289) | 1.52844 ± 0.0199081 (W 0.00553931) | · | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 3.87835 ± 0.0114686 (W 0.0101606) | 3.8902 ± 0.0211864 (W 0.0105655) | 4.66319 ± 0.00772374 (W 0.0217311) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.537894 ± 0.00178804 (W 0.00230631) | 0.540681 ± 0.00206398 (W 0.00204995) | 0.647331 ± 0.00590839 (W 0.00516252) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 1.81484 ± 0.00440202 (W 0.00464147) | 1.80585 ± 0.00573698 (W 0.00397991) | 2.15665 ± 0.00706794 (W 0.0145926) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 4.26618 ± 0.0187941 (W 0.00964624) | 4.28278 ± 0.00782889 (W 0.00851957) | 5.1545 ± 0.0195511 (W 0.0160805) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 14.4146 ± 0.0584948 (W 0.0282118) | 14.4462 ± 0.0308693 (W 0.0207764) | 17.199 ± 0.0708742 (W 0.028137) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 34.2308 ± 0.130516 (W 0.0489989) | 34.3312 ± 0.135653 (W 0.0406513) | 41.0099 ± 0.112982 (W 0.0822724) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 116.544 ± 0.239275 (W 0.107559) | 116.847 ± 0.371746 (W 0.0652465) | 138.377 ± 0.428601 (W 0.183916) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |

## gemm-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.153785 ± 0.000473675 (W 0.000216542) | 0.169387 ± 0.000897678 (W 0.00391516) | 0.233973 ± 0.0029557 (W 0.00290771) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.458237 ± 0.000283475 (W 0.00227039) | 0.46009 ± 0.00339474 (W 0.00239977) | 0.746075 ± 0.00098611 (W 0.00205353) | · | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 1.1141 ± 0.00295164 (W 0.00158269) | 1.11397 ± 0.0049093 (W 0.00198164) | 1.79525 ± 0.00767419 (W 0.00910662) | · | 🔵 faer-fork 1.61× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 3.64073 ± 0.0059566 (W 0.00450794) | 3.63714 ± 0.0102519 (W 0.00429488) | 5.80491 ± 0.0232549 (W 0.0153206) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 8.6625 ± 0.0195792 (W 0.00991175) | 8.68263 ± 0.0324982 (W 0.0140252) | 13.8736 ± 0.0739335 (W 0.0344632) | · | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 29.429 ± 0.0598016 (W 0.0299108) | 29.4366 ± 0.111407 (W 0.0458117) | 46.5831 ± 0.143671 (W 0.0531193) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |

## gemm-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.103092 ± 0.000446132 (W 0.00164651) | 0.141963 ± 0.0113948 (W 0.0105761) | 0.094908 ± 0.000252631 (W 0.000920632) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.234755 ± 0.000965044 (W 0.00213796) | 0.278055 ± 0.00148124 (W 0.00435948) | 0.295507 ± 0.00153281 (W 0.00196107) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.579768 ± 0.00375196 (W 0.0023646) | 0.60771 ± 0.000632097 (W 0.00314928) | 0.702442 ± 0.00392853 (W 0.00375696) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.82813 ± 0.00329949 (W 0.00404464) | 1.83357 ± 0.00587376 (W 0.011101) | 2.27475 ± 0.0421556 (W 0.00826047) | · | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 4.44938 ± 0.0166312 (W 0.0149448) | 4.43757 ± 0.0223924 (W 0.00822671) | 5.5589 ± 0.0176158 (W 0.0185387) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 14.9101 ± 0.0188023 (W 0.0255965) | 14.9379 ± 0.0439218 (W 0.0151196) | 19.0215 ± 0.10316 (W 0.152689) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |

## gemm-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.539718 ± 0.00256396 (W 0.00244037) | 0.537634 ± 0.00184556 (W 0.00243711) | 0.649761 ± 0.00306012 (W 0.00867968) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 1.81594 ± 0.00440858 (W 0.00268873) | 1.81864 ± 0.0020752 (W 0.0024947) | 2.15299 ± 0.00748195 (W 0.0174496) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 4.29019 ± 0.0155409 (W 0.00572236) | 4.27899 ± 0.0119594 (W 0.00644433) | 5.14023 ± 0.0122288 (W 0.0176443) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 14.3993 ± 0.0485097 (W 0.0147289) | 14.4278 ± 0.0322299 (W 0.0264623) | 17.2127 ± 0.0616905 (W 0.0224247) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 34.3776 ± 0.13617 (W 0.02434) | 34.3762 ± 0.0636784 (W 0.0454669) | 40.9767 ± 0.0801217 (W 0.0639707) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 116.633 ± 0.249316 (W 0.387474) | 116.874 ± 0.294535 (W 0.149601) | 138.1 ± 0.303958 (W 0.15482) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |

## gemm-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.154006 ± 0.00022248 (W 0.000722954) | 0.170696 ± 0.000573034 (W 0.000841733) | 0.234055 ± 0.00162397 (W 0.0025463) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.457902 ± 0.00125015 (W 0.00272324) | 0.458708 ± 0.00387948 (W 0.00290303) | 0.746097 ± 0.00339318 (W 0.00603666) | · | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 1.11432 ± 0.00533821 (W 0.00162733) | 1.1145 ± 0.00183229 (W 0.00139456) | 1.79986 ± 0.0052442 (W 0.00386234) | · | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 3.63706 ± 0.000934273 (W 0.00445417) | 3.6384 ± 0.00935777 (W 0.00662037) | 5.77735 ± 0.0422114 (W 0.0129712) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 8.68246 ± 0.0142085 (W 0.0268229) | 8.66939 ± 0.0217738 (W 0.0176942) | 13.8903 ± 0.0678078 (W 0.0126428) | · | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 29.4746 ± 0.075059 (W 0.0178194) | 29.475 ± 0.0478869 (W 0.0242889) | 46.6115 ± 0.0814547 (W 0.0245528) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |

## gemm-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.102981 ± 0.000433429 (W 0.000671261) | 0.122027 ± 0.0103267 (W 0.0125648) | 0.094707 ± 0.000142388 (W 0.00157949) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.235487 ± 0.000774638 (W 0.00128964) | 0.279758 ± 0.00131534 (W 0.00618703) | 0.295823 ± 0.00105039 (W 0.00222105) | 🔵 faer-fork 1.19× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.580215 ± 0.000986182 (W 0.00223397) | 0.610822 ± 0.0015635 (W 0.00323876) | 0.703754 ± 0.00283112 (W 0.00475635) | 🔵 faer-fork 1.05× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.82364 ± 0.00564776 (W 0.0073293) | 1.82444 ± 0.00214547 (W 0.00985068) | 2.28062 ± 0.00506716 (W 0.00988059) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 4.43806 ± 0.011577 (W 0.00641049) | 4.43176 ± 0.0159109 (W 0.00529531) | 5.54064 ± 0.0090967 (W 0.0182589) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 15.0013 ± 0.135347 (W 0.0330197) | 14.9334 ± 0.0472498 (W 0.0181539) | 19.0189 ± 0.0577436 (W 0.171922) | · | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |

## gemm-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.067206 ± 0.000200577 (W 8.47349e-05) | 0.067015 ± 0.000255226 (W 0.00137575) | 0.081961 ± 0.000338274 (W 0.000609059) | · | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.223596 ± 0.00222859 (W 0.00214101) | 0.223434 ± 0.00171362 (W 0.00195525) | 0.277426 ± 0.00136747 (W 0.00306248) | · | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.538366 ± 0.00263162 (W 0.00243398) | 0.535602 ± 0.000832086 (W 0.00276395) | 0.654334 ± 0.00327135 (W 0.00504571) | · | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.81238 ± 0.00256895 (W 0.00512016) | 1.80261 ± 0.00239299 (W 0.00384994) | 2.15801 ± 0.00223143 (W 0.0128654) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 4.27763 ± 0.0189801 (W 0.0109132) | 4.2755 ± 0.0158544 (W 0.00408399) | 5.11379 ± 0.0154479 (W 0.0212426) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 14.6393 ± 0.0643325 (W 0.0277609) | 14.5477 ± 0.0216505 (W 0.0187777) | 17.156 ± 0.0536366 (W 0.0464529) | · | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |

## gemm-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.021661 ± 0.000211472 (W 0.00041831) | 0.072786 ± 0.000976411 (W 0.00574635) | 0.036018 ± 9.01499e-05 (W 0.000524258) | 🔵 faer-fork 3.36× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.084168 ± 0.000801857 (W 0.00118138) | 0.205051 ± 0.00247373 (W 0.0297699) | 0.101419 ± 0.00106862 (W 0.00124375) | 🔵 faer-fork 2.44× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.158355 ± 0.000724812 (W 0.00112851) | 0.226493 ± 0.0166015 (W 0.0161407) | 0.257429 ± 0.00125086 (W 0.00328886) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.468488 ± 0.00234862 (W 0.00269434) | 0.544819 ± 0.00279299 (W 0.00573785) | 0.746372 ± 0.00227925 (W 0.00335866) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.11981 ± 0.00162902 (W 0.00265906) | 1.18132 ± 0.00374393 (W 0.00360625) | 1.79219 ± 0.00756571 (W 0.00184298) | 🔵 faer-fork 1.05× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 3.67632 ± 0.00705746 (W 0.00711959) | 3.67836 ± 0.00427694 (W 0.00557233) | 5.79778 ± 0.00890081 (W 0.0218967) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |

## gemm-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.021841 ± 0.00035975 (W 0.00059012) | 0.074338 ± 0.000696026 (W 0.000835094) | 0.016531 ± 8.90524e-05 (W 0.000292101) | 🔵 faer-fork 3.40× faster than `faer-upstream` | 🔴 faer-fork 1.32× slower than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.049321 ± 0.000460622 (W 0.00113411) | 0.248432 ± 0.00435202 (W 0.0457197) | 0.043581 ± 0.000210072 (W 0.000513749) | 🔵 faer-fork 5.04× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.108191 ± 0.00062753 (W 0.00126802) | 0.31008 ± 0.0840536 (W 0.090955) | 0.095037 ± 0.000560715 (W 0.000762991) | 🔵 faer-fork 2.87× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.248972 ± 0.00271907 (W 0.00218359) | 0.504889 ± 0.0141578 (W 0.0708488) | 0.29415 ± 0.00182108 (W 0.0017346) | 🔵 faer-fork 2.03× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 0.593484 ± 0.000571772 (W 0.00260083) | 0.757104 ± 0.00937206 (W 0.0550257) | 0.698499 ± 0.00114615 (W 0.00413121) | 🔵 faer-fork 1.28× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 1.87554 ± 0.00189184 (W 0.00339558) | 2.05184 ± 0.00813196 (W 0.00462942) | 2.25678 ± 0.0136506 (W 0.00473802) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.066994 ± 0.000187948 (W 0.000972782) | 0.067365 ± 0.00017003 (W 0.000306768) | 0.082824 ± 0.000254638 (W 0.00145768) | · | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.223794 ± 0.000329915 (W 0.00200603) | 0.223468 ± 0.000726126 (W 0.00229594) | 0.275241 ± 0.000727463 (W 0.00554201) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.534834 ± 0.00165957 (W 0.00262604) | 0.535121 ± 0.00135898 (W 0.00332578) | 0.649054 ± 0.00426152 (W 0.00386221) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.80263 ± 0.00502803 (W 0.00515495) | 1.80729 ± 0.00319099 (W 0.003894) | 2.15768 ± 0.00334615 (W 0.00723011) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 4.2678 ± 0.0259306 (W 0.0117026) | 4.27065 ± 0.0189748 (W 0.00899179) | 5.10191 ± 0.0176411 (W 0.0173826) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 14.6057 ± 0.0936192 (W 0.0160298) | 14.5295 ± 0.0363935 (W 0.0144663) | 17.1333 ± 0.0199777 (W 0.0414621) | · | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |

## gemm-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.021921 ± 0.000366042 (W 0.000436343) | 0.072445 ± 0.00155381 (W 0.00137449) | 0.035636 ± 0.000377414 (W 0.00038548) | 🔵 faer-fork 3.30× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.084558 ± 0.000874001 (W 0.00149066) | 0.202625 ± 0.00142836 (W 0.0298142) | 0.099916 ± 0.00158617 (W 0.00100798) | 🔵 faer-fork 2.40× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.158376 ± 0.000912648 (W 0.000698742) | 0.235767 ± 0.0115669 (W 0.0233587) | 0.253309 ± 0.00195758 (W 0.00269841) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.466728 ± 0.00150599 (W 0.00340789) | 0.543968 ± 0.00224007 (W 0.00566115) | 0.748682 ± 0.00215361 (W 0.00310717) | 🔵 faer-fork 1.17× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.11765 ± 0.00338788 (W 0.00206237) | 1.17765 ± 0.00222337 (W 0.00541015) | 1.7988 ± 0.00622487 (W 0.00183059) | 🔵 faer-fork 1.05× faster than `faer-upstream` | 🔵 faer-fork 1.61× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 3.67469 ± 0.00965513 (W 0.00703812) | 3.68128 ± 0.00333918 (W 0.00445278) | 5.83424 ± 0.0183079 (W 0.0113946) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |

## gemm-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.02167 ± 3.56417e-05 (W 0.000299006) | 0.073796 ± 0.000945348 (W 0.00129409) | 0.016942 ± 0.000309001 (W 0.000301273) | 🔵 faer-fork 3.41× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.04829 ± 0.000428867 (W 0.000944445) | 0.250277 ± 0.0010144 (W 0.00353274) | 0.043471 ± 8.2282e-05 (W 0.00185149) | 🔵 faer-fork 5.18× faster than `faer-upstream` | 🔴 faer-fork 1.11× slower than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.108352 ± 0.000487034 (W 0.00103121) | 0.162652 ± 0.0723985 (W 0.0916958) | 0.095326 ± 0.000283904 (W 0.00115446) | · | 🔴 faer-fork 1.14× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.248965 ± 0.00309534 (W 0.00202074) | 0.498698 ± 0.0002255 (W 0.0621116) | 0.292447 ± 0.00358352 (W 0.00180312) | 🔵 faer-fork 2.00× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.594506 ± 0.00198724 (W 0.00350607) | 0.813343 ± 0.0215985 (W 0.0568511) | 0.696255 ± 0.00191686 (W 0.00529683) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 1.87855 ± 0.0087938 (W 0.00381774) | 2.05218 ± 0.00811448 (W 0.0112301) | 2.26902 ± 0.00969106 (W 0.0057516) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |

## gemm-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.267337 ± 0.00165848 (W 0.0020544) | 0.268459 ± 0.00053278 (W 0.00180441) | 0.320289 ± 0.00239597 (W 0.00458084) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.905463 ± 0.00442317 (W 0.00203996) | 0.900494 ± 0.00354437 (W 0.00193538) | 1.08705 ± 0.00777643 (W 0.00712894) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 2.13262 ± 0.00315726 (W 0.00621626) | 2.14031 ± 0.00507418 (W 0.00337364) | 2.53228 ± 0.015742 (W 0.0179164) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 7.21158 ± 0.0337325 (W 0.00725286) | 7.19916 ± 0.0479414 (W 0.00949526) | 8.48764 ± 0.0158675 (W 0.0270154) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 17.0767 ± 0.0513296 (W 0.0182292) | 17.0877 ± 0.0530625 (W 0.0161009) | 20.3216 ± 0.15206 (W 0.0347822) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 57.9958 ± 0.124666 (W 0.0749944) | 58.0385 ± 0.176482 (W 0.0935955) | 69.141 ± 0.0688931 (W 0.124227) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |

## gemm-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.087362 ± 0.000260001 (W 0.000188476) | 0.095769 ± 0.000528689 (W 0.00205462) | 0.127778 ± 0.00128405 (W 0.00187126) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.233497 ± 0.000617503 (W 0.00173036) | 0.26773 ± 0.00400999 (W 0.00465597) | 0.371879 ± 0.00293064 (W 0.00646656) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.574518 ± 0.000431143 (W 0.00239635) | 0.584146 ± 0.00569209 (W 0.00738259) | 0.893595 ± 0.0061734 (W 0.00612496) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.81302 ± 0.00225848 (W 0.00343255) | 1.81596 ± 0.00137025 (W 0.0017008) | 2.88955 ± 0.0133858 (W 0.00965199) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 4.35874 ± 0.00974385 (W 0.00326593) | 4.37095 ± 0.00362348 (W 0.0074259) | 7.01479 ± 0.0462324 (W 0.0137046) | · | 🔵 faer-fork 1.61× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 14.7393 ± 0.0102838 (W 0.0131996) | 14.7186 ± 0.0319708 (W 0.0189926) | 23.3359 ± 0.0499657 (W 0.0338471) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |

## gemm-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.056675 ± 0.000165742 (W 0.000719661) | 0.133119 ± 0.00172712 (W 0.00337615) | 0.0483 ± 0.00143797 (W 0.00027062) | 🔵 faer-fork 2.35× faster than `faer-upstream` | 🔴 faer-fork 1.17× slower than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.158586 ± 0.000411438 (W 0.00118375) | 0.27529 ± 0.0352479 (W 0.0344398) | 0.149838 ± 0.000723225 (W 0.00163285) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.310611 ± 0.00136566 (W 0.00241379) | 0.411155 ± 0.021776 (W 0.02262) | 0.341325 ± 0.00140912 (W 0.00260551) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.91457 ± 0.00101513 (W 0.00145218) | 0.994625 ± 0.00345084 (W 0.00565364) | 1.12076 ± 0.0316406 (W 0.00584773) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.20815 ± 0.000822274 (W 0.00374641) | 2.27918 ± 0.00586702 (W 0.00650438) | 2.66343 ± 0.00791155 (W 0.0118332) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 7.46291 ± 0.0265883 (W 0.0138462) | 7.46552 ± 0.0084635 (W 0.00671259) | 9.13202 ± 0.0353359 (W 0.0262028) | · | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |

## gemm-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.267024 ± 0.000551181 (W 0.00246194) | 0.266944 ± 0.00157412 (W 0.00354648) | 0.321688 ± 0.000887852 (W 0.00435788) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.903148 ± 0.00244382 (W 0.000909987) | 0.901938 ± 0.00227542 (W 0.0024889) | 1.07609 ± 0.0108177 (W 0.0180324) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 2.13676 ± 0.00305662 (W 0.00327) | 2.13625 ± 0.0053465 (W 0.00222298) | 2.53524 ± 0.0127824 (W 0.0122392) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 7.19517 ± 0.0236514 (W 0.012813) | 7.20166 ± 0.0105834 (W 0.0103755) | 8.52798 ± 0.0429579 (W 0.0137859) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 17.0738 ± 0.0343605 (W 0.0141573) | 17.0406 ± 0.115847 (W 0.0264893) | 20.2976 ± 0.0823723 (W 0.0456514) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 58.0088 ± 0.159908 (W 0.0775456) | 58.1936 ± 0.288709 (W 0.0245051) | 69.2907 ± 0.346168 (W 0.123493) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |

## gemm-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.087382 ± 0.000331663 (W 0.000137019) | 0.097101 ± 0.00137094 (W 0.00274075) | 0.125442 ± 0.00221571 (W 0.00272683) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.234446 ± 0.000632853 (W 0.00117923) | 0.270116 ± 0.00247041 (W 0.00326921) | 0.375656 ± 0.00412713 (W 0.00356691) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.575149 ± 0.000766237 (W 0.00196303) | 0.57756 ± 0.00430221 (W 0.00649642) | 0.896611 ± 0.000657965 (W 0.00398645) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.81439 ± 0.00143626 (W 0.00373218) | 1.81672 ± 0.00355602 (W 0.00215795) | 2.87145 ± 0.00696001 (W 0.0121454) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 4.36031 ± 0.00993302 (W 0.00852306) | 4.37426 ± 0.00839283 (W 0.00512576) | 7.07165 ± 0.00890886 (W 0.0160894) | · | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 14.7374 ± 0.0221816 (W 0.0168488) | 14.7361 ± 0.0335913 (W 0.0240594) | 23.2226 ± 0.155454 (W 0.0504493) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |

## gemm-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.056826 ± 0.000861878 (W 0.000492536) | 0.132657 ± 0.00105192 (W 0.00254688) | 0.048901 ± 0.000672191 (W 0.000650707) | 🔵 faer-fork 2.33× faster than `faer-upstream` | 🔴 faer-fork 1.16× slower than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.158997 ± 0.00038392 (W 0.00137306) | 0.277495 ± 0.000187014 (W 0.00735068) | 0.149959 ± 0.000491345 (W 0.00236381) | 🔵 faer-fork 1.75× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.309226 ± 0.00242954 (W 0.000971187) | 0.380451 ± 0.0229546 (W 0.0254261) | 0.341208 ± 0.00270065 (W 0.0031312) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.91431 ± 0.00155334 (W 0.00299389) | 0.996243 ± 0.00477055 (W 0.00530653) | 1.12795 ± 0.00400507 (W 0.00883748) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.20622 ± 0.00153073 (W 0.00200082) | 2.27567 ± 0.00804836 (W 0.00554894) | 2.67368 ± 0.0100577 (W 0.00881963) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 7.44943 ± 0.00692119 (W 0.00772815) | 7.45416 ± 0.00348372 (W 0.0118285) | 9.10783 ± 0.0247316 (W 0.0276244) | · | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |

## gemm-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.2836 ± 0.00390986 (W 0.00527179) | 1.27429 ± 0.00700212 (W 0.00503099) | 1.64985 ± 0.121766 (W 0.00463471) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 1.08117 ± 0.00489393 (W 0.00160135) | 1.08052 ± 0.0360137 (W 0.00488731) | 1.5144 ± 0.0319223 (W 0.0285796) | · | 🔵 faer-fork 1.40× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 1.08657 ± 0.00416263 (W 0.00297712) | 1.08603 ± 0.00258169 (W 0.00058474) | 1.55842 ± 0.0106817 (W 0.0132964) | · | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |

## gemm-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.422235 ± 0.00185399 (W 0.00449554) | 0.43001 ± 0.00365459 (W 0.0043901) | 0.413793 ± 0.0167736 (W 0.00534188) | · | · | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.30765 ± 0.000626738 (W 0.00254791) | 0.31022 ± 0.00117275 (W 0.00256922) | 0.428945 ± 0.00155867 (W 0.0021028) | · | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.277439 ± 0.000645948 (W 0.00220244) | 0.30494 ± 0.00202038 (W 0.00234698) | 0.388266 ± 0.000539474 (W 0.00243358) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |

## gemm-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.422079 ± 0.0695466 (W 0.00308576) | 0.425951 ± 0.00206395 (W 0.00412041) | 0.268348 ± 0.00245948 (W 0.00396545) | · | 🔴 faer-fork 1.57× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.186706 ± 0.000339462 (W 0.00515685) | 0.191017 ± 0.000533233 (W 0.00371216) | 0.289839 ± 0.000662781 (W 0.00329198) | · | 🔵 faer-fork 1.55× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.141864 ± 0.000447838 (W 0.00211537) | 0.20439 ± 0.0082402 (W 0.00293618) | 0.197126 ± 0.00185573 (W 0.00225007) | 🔵 faer-fork 1.44× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.04206 ± 0.0126642 (W 0.00123284) | 1.04241 ± 0.0163538 (W 0.0022532) | 2.57377 ± 0.0259829 (W 0.00852107) | · | 🔵 faer-fork 2.47× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 1.09723 ± 0.000746308 (W 0.00144027) | 1.10059 ± 0.00163306 (W 0.00411783) | 1.53704 ± 0.010039 (W 0.0137619) | · | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 1.13851 ± 0.0164157 (W 0.00222633) | 1.14164 ± 0.00963851 (W 0.00230891) | 1.23987 ± 0.0055216 (W 0.00236515) | · | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |

## gemm-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.326667 ± 0.00320985 (W 0.00371852) | 0.707643 ± 0.00450911 (W 0.00551591) | 0.63683 ± 0.000537292 (W 0.0043661) | 🔵 faer-fork 2.17× faster than `faer-upstream` | 🔵 faer-fork 1.95× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.29952 ± 0.00312584 (W 0.00314509) | 0.300018 ± 0.00124734 (W 0.00344763) | 0.435751 ± 0.00451636 (W 0.00528881) | · | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.297033 ± 0.0015049 (W 0.00156001) | 0.297937 ± 0.00303549 (W 0.00246666) | 0.329361 ± 0.00157702 (W 0.00165259) | · | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |

## gemm-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.198197 ± 0.00232276 (W 0.0039825) | 1.14066 ± 0.00284132 (W 0.011636) | 0.340316 ± 0.000695317 (W 0.00383278) | 🔵 faer-fork 5.76× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.182128 ± 0.000366624 (W 0.00304724) | 0.181908 ± 0.00176508 (W 0.00251828) | 0.296101 ± 0.00046363 (W 0.00240003) | · | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.15601 ± 0.000705596 (W 0.00168103) | 0.156732 ± 0.0003073 (W 0.000584508) | 0.169787 ± 0.000715536 (W 0.00164751) | · | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |

## gemm-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.15545 ± 0.00896755 (W 0.00690522) | 4.16654 ± 0.0121437 (W 0.0148466) | 5.76383 ± 0.0182099 (W 0.0157673) | · | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 4.38705 ± 0.0137068 (W 0.0166622) | 4.408 ± 0.0387387 (W 0.0199977) | 6.3018 ± 0.0624843 (W 0.032863) | · | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 4.39395 ± 0.00905044 (W 0.00637069) | 4.39711 ± 0.00720666 (W 0.00422153) | 5.26983 ± 0.00938398 (W 0.0140849) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.65558 ± 0.00184157 (W 0.00389537) | 1.66231 ± 0.00944815 (W 0.00527216) | 1.958 ± 0.00599491 (W 0.00659183) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 1.15668 ± 0.00146533 (W 0.00409566) | 1.16012 ± 0.0197823 (W 0.00466164) | 2.18194 ± 0.0249078 (W 0.026068) | · | 🔵 faer-fork 1.89× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 1.13731 ± 0.0037577 (W 0.0022177) | 1.13876 ± 0.00211054 (W 0.00242773) | 2.06364 ± 0.00852194 (W 0.0105517) | · | 🔵 faer-fork 1.81× faster than `lapack-mkl` |  |

## gemm-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.903381 ± 0.02821 (W 0.0103029) | 0.900165 ± 0.0062134 (W 0.00425042) | 0.953049 ± 0.00335753 (W 0.00543689) | · | 🔵 faer-fork 1.05× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.600938 ± 0.00670453 (W 0.00183393) | 0.601304 ± 0.00357271 (W 0.00225189) | 1.25959 ± 0.184066 (W 0.0235409) | · | 🔵 faer-fork 2.10× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.57261 ± 0.00621535 (W 0.00271386) | 0.624718 ± 0.0196787 (W 0.0159767) | 0.758037 ± 0.00351826 (W 0.00247754) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.32× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.08288 ± 0.00721004 (W 0.00206525) | 4.07942 ± 0.0132407 (W 0.00477524) | 5.75147 ± 0.0207163 (W 0.0482858) | · | 🔵 faer-fork 1.41× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 4.36569 ± 0.00647863 (W 0.00751598) | 4.38788 ± 0.0213534 (W 0.0101929) | 6.30512 ± 0.0374081 (W 0.0601458) | · | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 4.43508 ± 0.00388608 (W 0.00375983) | 4.44051 ± 0.0150721 (W 0.0138832) | 5.7 ± 0.0245797 (W 0.0136057) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |

## gemm-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.18621 ± 0.00369408 (W 0.00199744) | 1.95702 ± 0.00499128 (W 0.0108623) | 20.617 ± 0.23661 (W 0.0287161) | 🔵 faer-fork 1.65× faster than `faer-upstream` | 🔵 faer-fork 17.38× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 1.16729 ± 0.00338731 (W 0.00330716) | 1.15115 ± 0.00211983 (W 0.00463029) | 2.38839 ± 0.0175176 (W 0.0695611) | · | 🔵 faer-fork 2.05× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 1.16931 ± 0.000860069 (W 0.00474631) | 1.16436 ± 0.000599543 (W 0.00319597) | 2.04183 ± 0.0142204 (W 0.0320305) | · | 🔵 faer-fork 1.75× faster than `lapack-mkl` |  |

## gemm-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.659132 ± 0.00182857 (W 0.00600582) | 2.58572 ± 0.004794 (W 0.0277001) | 3.76144 ± 0.00939825 (W 0.00774888) | 🔵 faer-fork 3.92× faster than `faer-upstream` | 🔵 faer-fork 5.71× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.593243 ± 0.000948462 (W 0.00471598) | 0.589514 ± 0.00406517 (W 0.0040531) | 1.81253 ± 0.0287544 (W 0.0248507) | · | 🔵 faer-fork 3.06× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.576141 ± 0.00115013 (W 0.00375885) | 0.575536 ± 0.00278449 (W 0.00333549) | 1.09959 ± 0.0296464 (W 0.058061) | · | 🔵 faer-fork 1.91× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.549998 ± 0.016488 (W 0.0034734) | 0.547619 ± 0.00106145 (W 0.00386727) | 0.730986 ± 0.0069607 (W 0.0039647) | · | 🔵 faer-fork 1.33× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.543191 ± 0.018044 (W 0.0023159) | 0.546584 ± 0.017267 (W 0.00209393) | 0.670624 ± 0.00750618 (W 0.00306203) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.549472 ± 0.00281685 (W 0.00216371) | 0.540994 ± 0.00245434 (W 0.00255427) | 0.685294 ± 0.000637825 (W 0.00223944) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |

## gemm-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.40847 ± 0.00070364 (W 0.00287411) | 0.411484 ± 0.00204902 (W 0.0018157) | 0.265933 ± 0.00414903 (W 0.00183846) | · | 🔴 faer-fork 1.54× slower than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.144198 ± 0.000240965 (W 0.00225442) | 0.14963 ± 0.00112376 (W 0.00313422) | 0.238146 ± 0.00141048 (W 0.00157734) | · | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.140081 ± 0.000416775 (W 0.00195073) | 0.153146 ± 0.000615613 (W 0.000902618) | 0.370919 ± 0.00107705 (W 0.00298225) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 2.65× faster than `lapack-mkl` |  |

## gemm-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.40899 ± 0.00148064 (W 0.0019933) | 0.416084 ± 0.00208373 (W 0.00189058) | 0.133037 ± 0.000755525 (W 0.00187666) | · | 🔴 faer-fork 3.07× slower than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.077404 ± 0.000694522 (W 0.000448177) | 0.088955 ± 0.00205686 (W 0.00499015) | 0.114012 ± 0.000334515 (W 0.00165818) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.072174 ± 0.000428113 (W 0.000261631) | 0.104103 ± 0.000364217 (W 0.00216947) | 0.116255 ± 0.000772698 (W 0.00140399) | 🔵 faer-fork 1.44× faster than `faer-upstream` | 🔵 faer-fork 1.61× faster than `lapack-mkl` |  |

## gemm-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.538058 ± 0.00535115 (W 0.00318578) | 0.546777 ± 0.00138297 (W 0.00292617) | 0.736279 ± 0.00805168 (W 0.00356584) | · | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.549267 ± 0.000989192 (W 0.00202535) | 0.54954 ± 0.00393821 (W 0.00280063) | 1.49189 ± 0.00269219 (W 0.0175451) | · | 🔵 faer-fork 2.72× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.557552 ± 0.00274069 (W 0.00259629) | 0.556075 ± 0.00279139 (W 0.00359827) | 0.755806 ± 0.00316718 (W 0.00351312) | · | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |

## gemm-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.177249 ± 0.00170208 (W 0.00103034) | 0.367352 ± 0.000756889 (W 0.00538229) | 0.738611 ± 0.00149102 (W 0.00606969) | 🔵 faer-fork 2.07× faster than `faer-upstream` | 🔵 faer-fork 4.17× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.146374 ± 0.000322802 (W 0.00136971) | 0.146424 ± 0.000928962 (W 0.00229954) | 0.523802 ± 0.00348518 (W 0.00743536) | · | 🔵 faer-fork 3.58× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.156783 ± 0.000564742 (W 0.00120588) | 0.158076 ± 0.00208776 (W 0.0032758) | 0.270446 ± 0.0021657 (W 0.00151404) | · | 🔵 faer-fork 1.72× faster than `lapack-mkl` |  |

## gemm-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.112859 ± 0.00181527 (W 0.00132481) | 0.579013 ± 0.00248219 (W 0.00893915) | 0.375847 ± 0.000663788 (W 0.00271646) | 🔵 faer-fork 5.13× faster than `faer-upstream` | 🔵 faer-fork 3.33× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.081832 ± 0.000199141 (W 0.0007246) | 0.082032 ± 0.000137987 (W 0.00115295) | 0.232254 ± 0.00271413 (W 0.00541576) | · | 🔵 faer-fork 2.84× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.078867 ± 0.000208135 (W 0.000296449) | 0.080299 ± 0.000288722 (W 0.000402845) | 0.125725 ± 0.000462973 (W 0.00150959) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |

## gemm-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.12835 ± 0.00170483 (W 0.00459922) | 2.1338 ± 0.00268806 (W 0.00507951) | 2.79349 ± 0.0780137 (W 0.00336111) | · | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 2.16708 ± 0.00315162 (W 0.00258409) | 2.16938 ± 0.025109 (W 0.00480122) | 2.84242 ± 0.0150615 (W 0.0503953) | · | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 2.17107 ± 0.00308073 (W 0.00345691) | 2.17543 ± 0.0012142 (W 0.00285038) | 2.59705 ± 0.00568412 (W 0.00635295) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.815956 ± 0.00395687 (W 0.00392497) | 0.825041 ± 0.00128969 (W 0.00118519) | 0.956284 ± 0.00493872 (W 0.00162844) | · | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.583623 ± 0.00237122 (W 0.00213036) | 0.582143 ± 0.0026912 (W 0.00283101) | 0.943694 ± 0.0902238 (W 0.0233813) | · | 🔵 faer-fork 1.62× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.556465 ± 0.0020396 (W 0.00295919) | 0.574514 ± 0.00277946 (W 0.00456406) | 1.0223 ± 0.00422371 (W 0.0032259) | · | 🔵 faer-fork 1.84× faster than `lapack-mkl` |  |

## gemm-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.819845 ± 0.0010212 (W 0.00329023) | 0.827208 ± 0.0029333 (W 0.00342099) | 0.459012 ± 0.00203892 (W 0.00555249) | · | 🔴 faer-fork 1.79× slower than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.300648 ± 0.00083894 (W 0.00223931) | 0.302954 ± 0.000539427 (W 0.00280738) | 0.464545 ± 0.00296313 (W 0.00239724) | · | 🔵 faer-fork 1.55× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.276724 ± 0.000556504 (W 0.00227246) | 0.335104 ± 0.00203294 (W 0.00846285) | 0.378097 ± 0.00156253 (W 0.00201825) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |

## gemm-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.04214 ± 0.000668799 (W 0.0035334) | 2.03922 ± 0.00786747 (W 0.00555748) | 2.82596 ± 0.00792551 (W 0.00867066) | · | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 2.17765 ± 0.00790162 (W 0.0043526) | 2.18145 ± 0.0103351 (W 0.00662791) | 2.91106 ± 0.0311701 (W 0.0474679) | · | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 2.20166 ± 0.00462752 (W 0.00565625) | 2.2017 ± 0.00340847 (W 0.00464823) | 2.75807 ± 0.00613405 (W 0.0118931) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |

## gemm-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.594766 ± 0.00198279 (W 0.00213664) | 0.974173 ± 0.00401016 (W 0.00825174) | 1.70412 ± 0.00474614 (W 0.00513908) | 🔵 faer-fork 1.64× faster than `faer-upstream` | 🔵 faer-fork 2.87× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.575668 ± 0.00290766 (W 0.00185834) | 0.570326 ± 0.00153251 (W 0.00177451) | 0.955596 ± 0.0050147 (W 0.0227339) | · | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.575679 ± 0.00409237 (W 0.00205025) | 0.573565 ± 0.00158517 (W 0.00228702) | 0.931038 ± 0.0110476 (W 0.00484799) | · | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |

## gemm-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.324262 ± 0.00269252 (W 0.0026211) | 1.27624 ± 0.00970661 (W 0.0174815) | 0.852684 ± 0.00609025 (W 0.00488643) | 🔵 faer-fork 3.94× faster than `faer-upstream` | 🔵 faer-fork 2.63× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.290019 ± 0.0015473 (W 0.00153247) | 0.289498 ± 0.000427325 (W 0.00179026) | 0.460926 ± 0.00280438 (W 0.00314705) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.309775 ± 0.00112404 (W 0.0018362) | 0.309115 ± 0.000356045 (W 0.00190888) | 0.402091 ± 0.00308641 (W 0.00316711) | · | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |

## gemm-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.13118 ± 0.00657184 (W 0.00244562) | 1.12529 ± 0.0065553 (W 0.00195917) | 1.24089 ± 0.00192676 (W 0.00219851) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |

## gemm-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.297632 ± 0.00062307 (W 0.00183631) | 0.297576 ± 0.000551126 (W 0.00243238) | 0.327101 ± 0.00139784 (W 0.00336414) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |

## gemm-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.155531 ± 0.000767645 (W 0.00160422) | 0.157102 ± 0.00114311 (W 0.0013055) | 0.168242 ± 0.00102169 (W 0.00195522) | · | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |

## gemm-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.08989 ± 0.00524149 (W 0.00110676) | 1.08862 ± 0.010004 (W 0.00162071) | 1.55279 ± 0.00767695 (W 0.00345464) | · | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |

## gemm-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.278387 ± 0.0018419 (W 0.00226366) | 0.302682 ± 0.000855773 (W 0.00272301) | 0.388613 ± 0.00170463 (W 0.00152166) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |

## gemm-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.141745 ± 0.00039718 (W 0.00248111) | 0.202296 ± 0.000567896 (W 0.00449301) | 0.193882 ± 0.000338486 (W 0.0010236) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |

## gemm-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.44848 ± 0.00141919 (W 0.00638732) | 4.44746 ± 0.00315736 (W 0.00378696) | 5.70316 ± 0.00350793 (W 0.0144586) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |

## gemm-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.1725 ± 0.00353565 (W 0.00315626) | 1.16767 ± 0.00129367 (W 0.00667044) | 2.04555 ± 0.0143518 (W 0.0186691) | · | 🔵 faer-fork 1.74× faster than `lapack-mkl` |  |

## gemm-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.573898 ± 0.00563118 (W 0.00385124) | 0.576097 ± 0.00134275 (W 0.00418733) | 1.0579 ± 0.011894 (W 0.0316987) | · | 🔵 faer-fork 1.84× faster than `lapack-mkl` |  |

## gemm-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.39476 ± 0.0146838 (W 0.00679192) | 4.41209 ± 0.0186206 (W 0.0142038) | 5.27029 ± 0.0205206 (W 0.0135913) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.13334 ± 0.00190022 (W 0.00104332) | 1.14577 ± 0.00130126 (W 0.00197828) | 2.06191 ± 0.00334781 (W 0.00491762) | · | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |

## gemm-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.573873 ± 0.00147918 (W 0.00262308) | 0.621622 ± 0.00349707 (W 0.0159441) | 0.754364 ± 0.00353584 (W 0.00348404) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |

## gemm-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.557756 ± 0.00316259 (W 0.00329752) | 0.556594 ± 0.00234795 (W 0.00246018) | 0.756515 ± 0.00371353 (W 0.00419799) | · | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |

## gemm-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.156452 ± 0.000945337 (W 0.00256993) | 0.157384 ± 0.00148013 (W 0.0022006) | 0.266508 ± 0.000565695 (W 0.0020519) | · | 🔵 faer-fork 1.70× faster than `lapack-mkl` |  |

## gemm-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.078517 ± 0.000693989 (W 0.000339598) | 0.080481 ± 0.000722684 (W 0.00160424) | 0.127008 ± 0.00128069 (W 0.00113193) | · | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |

## gemm-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.545866 ± 0.00407831 (W 0.00268269) | 0.54437 ± 0.00369664 (W 0.00252497) | 0.684022 ± 0.00263429 (W 0.00259703) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |

## gemm-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.140482 ± 0.000136793 (W 0.000398962) | 0.152575 ± 0.000348124 (W 0.00286998) | 0.372572 ± 0.00111707 (W 0.00184291) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 2.65× faster than `lapack-mkl` |  |

## gemm-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.071854 ± 0.000233967 (W 0.000417763) | 0.104424 ± 0.00114231 (W 0.00247123) | 0.117078 ± 0.000772854 (W 0.00143647) | 🔵 faer-fork 1.45× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |

## gemm-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.2081 ± 0.0144554 (W 0.0062716) | 2.1992 ± 0.00188468 (W 0.00450102) | 2.75487 ± 0.0180444 (W 0.00964294) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |

## gemm-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.575641 ± 0.00115116 (W 0.00247623) | 0.575591 ± 0.00157785 (W 0.00194895) | 0.938797 ± 0.00667218 (W 0.0038361) | · | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |

## gemm-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.310531 ± 0.00134184 (W 0.00226805) | 0.311061 ± 0.000305962 (W 0.00227689) | 0.404883 ± 0.00232467 (W 0.0032529) | · | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |

## gemm-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.17252 ± 0.00389772 (W 0.00313381) | 2.1774 ± 0.0159139 (W 0.00179192) | 2.59781 ± 0.00612964 (W 0.00693651) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.554146 ± 0.00174567 (W 0.00270855) | 0.575649 ± 0.00282693 (W 0.00337731) | 1.02312 ± 0.00311859 (W 0.00337806) | · | 🔵 faer-fork 1.85× faster than `lapack-mkl` |  |

## gemm-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.277937 ± 0.00183814 (W 0.00537002) | 0.338353 ± 0.000868242 (W 0.00916821) | 0.378643 ± 0.0187839 (W 0.00205834) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` | `!` between-start range over 5% |

## linalg-small, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004067 ± 0.0001245 (W 4.26052e-05) | 0.004198 ± 0.000122882 (W 6.31205e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003447 ± 2.91605e-05 (W 0.000114374) | 0.003466 ± 2.6839e-05 (W 9.68365e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013986 ± 0.000384648 (W 0.000168671) | 0.03206 ± 0.0012922 (W 4.13316e-05) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012343 ± 0.000238132 (W 7.79744e-05) | 0.016782 ± 0.00114541 (W 7.90645e-05) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031579 ± 8.18535e-05 (W 0.00172338) | 0.164857 ± 0.000723372 (W 0.00169411) | - | 🔵 faer-fork 5.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.028704 ± 0.000155725 (W 0.000105049) | 0.033081 ± 0.000155725 (W 0.00010015) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.111959 ± 0.00576562 (W 0.00190838) | 0.108462 ± 0.00024964 (W 0.000181426) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.112079 ± 0.000456545 (W 0.0011317) | 0.108792 ± 0.00582189 (W 0.000392782) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.573382 ± 0.0029887 (W 0.00249615) | 0.531699 ± 0.000877714 (W 0.00261792) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.575506 ± 0.00195912 (W 0.00281795) | 0.536101 ± 0.00454858 (W 0.00321728) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.3395 ± 0.0046317 (W 0.00249697) | 1.25232 ± 0.00124637 (W 0.00282778) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.35875 ± 0.000563549 (W 0.00446564) | 1.27524 ± 0.00527962 (W 0.00542946) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01086 ± 0.0001253 (W 0.000104006) | 0.009077 ± 7.55447e-05 (W 2.90568e-05) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013765 ± 6.08276e-06 (W 0.000127174) | 0.010811 ± 5.00033e-05 (W 4.90133e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.050735 ± 0.00145461 (W 0.000224134) | 0.042529 ± 0.000418534 (W 0.000210387) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.058078 ± 0.000164621 (W 0.000172509) | 0.046757 ± 0.000190867 (W 0.000162475) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.14488 ± 0.00126905 (W 0.00205891) | 0.121106 ± 0.000496323 (W 0.000309838) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.163564 ± 0.000508167 (W 0.00261978) | 0.136134 ± 0.00018273 (W 0.00244507) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008466 ± 6.35085e-05 (W 5.50454e-05) | 0.008125 ± 4.56107e-05 (W 8.1867e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008456 ± 4.3589e-05 (W 7.07192e-05) | 0.008166 ± 3.23574e-05 (W 6.89202e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.040836 ± 0.000181475 (W 7.55195e-05) | 0.040395 ± 0.000136346 (W 8.58487e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.041166 ± 0.000140144 (W 7.31662e-05) | 0.040265 ± 0.0001305 (W 8.84833e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.121867 ± 0.000598957 (W 0.00164268) | 0.119953 ± 0.000313082 (W 0.00216023) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.121616 ± 0.000480463 (W 0.0017181) | 0.120315 ± 0.00050112 (W 0.0016735) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006452 ± 2.08167e-05 (W 7.83722e-05) | 0.007704 ± 9.66454e-05 (W 4.77305e-05) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008266 ± 4.0501e-05 (W 7.95192e-05) | 0.007394 ± 5.9808e-05 (W 0.000118799) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.023183 ± 0.000201329 (W 9.53483e-05) | 0.034303 ± 0.000393947 (W 9.72111e-05) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.029194 ± 0.000350206 (W 0.000182503) | 0.0266 ± 0.00030531 (W 0.000132252) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.055052 ± 0.000209277 (W 0.000193797) | 0.085078 ± 0.00102415 (W 0.000249951) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.07493 ± 0.000329596 (W 0.00271979) | 0.068819 ± 0.000462008 (W 0.00209413) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.036478 ± 0.000231215 (W 0.000166386) | 0.040736 ± 0.000225965 (W 0.000152231) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.03186 ± 0.000244602 (W 0.00020406) | 0.033232 ± 0.000198109 (W 0.000171959) | - | · | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.106799 ± 0.00012546 (W 0.00203346) | 0.174255 ± 0.000549602 (W 0.00168828) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.116396 ± 0.00089582 (W 0.000532582) | 0.121977 ± 0.000648546 (W 0.00202392) | - | · | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.234106 ± 0.000229 (W 0.00205773) | 0.387371 ± 0.000638658 (W 0.00270137) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.292864 ± 0.000797489 (W 0.00322818) | 0.307752 ± 0.00152472 (W 0.00329428) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.024475 ± 0.000145661 (W 0.000149169) | 0.028864 ± 5.85947e-05 (W 0.000106833) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.067596 ± 0.000389744 (W 0.000174309) | 0.132837 ± 0.000724921 (W 0.00227973) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.137637 ± 0.000705728 (W 0.000302067) | 0.290019 ± 0.00123116 (W 0.00219051) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006812 ± 5.2915e-05 (W 6.10508e-05) | 0.008606 ± 0.000195021 (W 6.07141e-05) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009478 ± 1e-05 (W 9.49279e-05) | 0.008366 ± 0.000191068 (W 0.000141726) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021951 ± 0.00016045 (W 0.0001005) | 0.036418 ± 0.000703024 (W 4.50555e-05) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.030697 ± 0.000115717 (W 8.93308e-05) | 0.027933 ± 0.000156526 (W 8.39661e-05) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.049963 ± 0.000277921 (W 0.000121984) | 0.084247 ± 0.000726885 (W 0.00174985) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.076193 ± 6.45316e-05 (W 0.00275328) | 0.067766 ± 6.7159e-05 (W 0.0025922) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.168183 ± 0.00137698 (W 0.00232371) | 0.184493 ± 0.00203126 (W 0.00243555) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.169115 ± 0.00324107 (W 0.00211669) | 0.184392 ± 7.60197e-05 (W 0.00184729) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.679349 ± 0.00262322 (W 0.00343169) | 0.91371 ± 0.00208869 (W 0.00275018) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.680311 ± 0.00231742 (W 0.00261379) | 0.916475 ± 0.0022952 (W 0.00140259) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.56638 ± 0.00636379 (W 0.00330801) | 2.1635 ± 0.00660714 (W 0.0040704) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.56168 ± 0.00788246 (W 0.00388379) | 2.24921 ± 0.00512395 (W 0.00698243) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058639 ± 0.000380816 (W 0.000154533) | 0.141463 ± 0.000538331 (W 0.000393056) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.058849 ± 0.000322306 (W 0.00162205) | 0.141404 ± 0.00105786 (W 0.00131535) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.212305 ± 0.00067842 (W 0.00209717) | 0.520617 ± 0.000903106 (W 0.00272091) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.20888 ± 0.00115126 (W 0.00180698) | 0.51722 ± 0.000733507 (W 0.00267946) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.478264 ± 0.000433094 (W 0.0034804) | 1.17781 ± 0.00334748 (W 0.00221718) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.47652 ± 0.00222051 (W 0.00200485) | 1.18376 ± 0.00421497 (W 0.00298144) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000751 ± 5.7735e-06 (W 2.87802e-05) | 0.000712 ± 5.50757e-06 (W 3.06464e-05) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001713 ± 9.50438e-06 (W 5.14228e-05) | 0.001613 ± 1.52753e-05 (W 7.42967e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003036 ± 2.5697e-05 (W 7.43989e-05) | 0.002846 ± 2.3094e-05 (W 3.91063e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003837 ± 0.00012741 (W 4.27808e-05) | 0.004218 ± 0.000107503 (W 8.089e-05) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003486 ± 0.00015885 (W 5.88328e-05) | 0.003457 ± 0.000235139 (W 0.000111433) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014377 ± 0.000446696 (W 0.00016016) | 0.034313 ± 0.000155522 (W 4.84768e-05) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012513 ± 0.000326151 (W 6.31364e-05) | 0.014928 ± 0.00112219 (W 5.79336e-05) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.03199 ± 5.74195e-05 (W 7.30151e-05) | 0.056055 ± 0.0591781 (W 0.00600677) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.029305 ± 7.50555e-05 (W 0.000108297) | 0.036588 ± 0.0022726 (W 0.000317227) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.076803 ± 0.000619331 (W 0.000396744) | 0.108643 ± 0.000471134 (W 0.00135522) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.077164 ± 0.000621821 (W 0.00021563) | 0.109524 ± 5.25389e-05 (W 0.000608244) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.316459 ± 0.00202951 (W 0.00202384) | 0.534415 ± 0.00310079 (W 0.00182737) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.328454 ± 0.000518348 (W 0.00312182) | 0.533694 ± 0.00113135 (W 0.00291486) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.774878 ± 0.00301975 (W 0.00259142) | 1.23597 ± 0.0039985 (W 0.00485031) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.780148 ± 0.00390537 (W 0.00318001) | 1.23174 ± 0.00732524 (W 0.00234567) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011792 ± 5.82266e-05 (W 5.06873e-05) | 0.008897 ± 6.69104e-05 (W 6.34129e-05) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015038 ± 8.14064e-05 (W 8.79204e-05) | 0.0107 ± 3.05505e-05 (W 8.79204e-05) | - | 🔴 faer-fork 1.41× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.052658 ± 0.000747675 (W 0.000152379) | 0.042339 ± 7.37111e-05 (W 0.000233967) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.060373 ± 0.000320952 (W 0.000252825) | 0.046897 ± 0.00056689 (W 0.000155523) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.148546 ± 0.000822125 (W 0.00094117) | 0.120264 ± 0.000523941 (W 0.000273789) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.172703 ± 0.00122601 (W 0.00302636) | 0.135433 ± 0.00153889 (W 0.000925746) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.007053 ± 0.0001105 (W 0.000486838) | 0.008375 ± 2.5632e-05 (W 7.10866e-05) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006212 ± 0.000130512 (W 0.000108532) | 0.008365 ± 3.5171e-05 (W 5.11009e-05) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.020448 ± 0.000196011 (W 0.00042965) | 0.019937 ± 0.00026963 (W 0.000285705) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.020739 ± 0.000320921 (W 0.000216526) | 0.018876 ± 0.000191469 (W 0.000269635) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.046537 ± 3.09892e-05 (W 0.000400639) | 0.044733 ± 0.000350929 (W 0.000334307) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.047368 ± 0.000126623 (W 0.000391651) | 0.043722 ± 0.000100683 (W 0.000957011) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006542 ± 7.23418e-05 (W 5.98857e-05) | 0.007594 ± 9.07377e-05 (W 7.4364e-05) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008506 ± 3.751e-05 (W 9.09357e-05) | 0.007423 ± 0.000127017 (W 0.000230955) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.029395 ± 0.000676055 (W 0.000640542) | 0.036478 ± 0.000991784 (W 0.000492338) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.029555 ± 0.000276653 (W 0.000222947) | 0.030146 ± 0.000313784 (W 0.000868511) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.064109 ± 0.000900858 (W 0.000986116) | 0.088135 ± 0.00145041 (W 0.00101554) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.070482 ± 0.000385759 (W 0.000574928) | 0.075761 ± 0.00109194 (W 0.00133563) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.03796 ± 0.000280643 (W 0.000106547) | 0.054632 ± 0.000465844 (W 0.000802173) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.033302 ± 5.56776e-05 (W 0.000136665) | 0.046617 ± 0.000162225 (W 0.000248197) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.10696 ± 0.00114343 (W 0.000488986) | 0.245176 ± 0.000788102 (W 0.00240796) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.109073 ± 0.000607281 (W 0.00054463) | 0.187461 ± 0.00170503 (W 0.00214101) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.206025 ± 0.000934748 (W 0.00215053) | 0.500091 ± 0.00341136 (W 0.00372718) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.247252 ± 0.000639857 (W 0.0034632) | 0.399063 ± 0.00106702 (W 0.00316172) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025567 ± 9.06661e-05 (W 0.000153721) | 0.042248 ± 0.000280215 (W 0.000565658) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.074399 ± 0.000316598 (W 0.000260759) | 0.207486 ± 0.00132614 (W 0.0020345) | - | 🔵 faer-fork 2.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.145992 ± 0.000673494 (W 0.00152637) | 0.429362 ± 0.00213164 (W 0.00263804) | - | 🔵 faer-fork 2.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007544 ± 0.000350571 (W 9.1815e-05) | 0.008336 ± 0.000150965 (W 5.26308e-05) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010078 ± 7.59408e-05 (W 8.17313e-05) | 0.008246 ± 0.000104083 (W 0.0002004) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.02653 ± 0.000562719 (W 0.000479394) | 0.03744 ± 0.000685573 (W 0.000613092) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.03179 ± 0.000240438 (W 6.88208e-05) | 0.030396 ± 0.000165168 (W 0.000678679) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.055443 ± 0.000470164 (W 0.000599331) | 0.085429 ± 0.0030694 (W 0.000428644) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.075882 ± 0.000736741 (W 0.00230277) | 0.072164 ± 0.000509315 (W 0.000800869) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.17192 ± 0.00286154 (W 0.000461288) | 0.183203 ± 0.000810704 (W 0.00228805) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.170588 ± 0.0148724 (W 0.00204975) | 0.184244 ± 0.00103927 (W 0.0017806) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.557237 ± 0.0020425 (W 0.00320244) | 0.9079 ± 0.00414721 (W 0.00165492) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.560784 ± 0.00103659 (W 0.00240509) | 0.911757 ± 0.00195199 (W 0.00185782) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.13535 ± 0.00648664 (W 0.00408792) | 2.0937 ± 0.0100941 (W 0.00220177) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.14013 ± 0.00454355 (W 0.00319375) | 2.09545 ± 0.00632843 (W 0.00399768) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059441 ± 0.000543864 (W 0.000194204) | 0.141223 ± 0.000251596 (W 0.000309615) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059671 ± 0.000490816 (W 0.000247924) | 0.141274 ± 0.000575445 (W 0.00144339) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.215371 ± 0.00137489 (W 0.00128847) | 0.517096 ± 0.00221791 (W 0.0022925) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.218857 ± 0.000986431 (W 0.0019289) | 0.517056 ± 0.00311398 (W 0.00222518) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.516685 ± 0.00178957 (W 0.00186567) | 1.18972 ± 0.00438518 (W 0.00338918) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.516922 ± 0.00178739 (W 0.00245766) | 1.19022 ± 0.00272574 (W 0.00388652) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000952 ± 3.21455e-05 (W 2.92626e-05) | 0.000711 ± 1.76163e-05 (W 6.80603e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002144 ± 5.7735e-06 (W 3.90538e-05) | 0.001603 ± 1.52753e-05 (W 0.000108643) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003737 ± 0.0001 (W 4.87852e-05) | 0.002825 ± 2.12211e-05 (W 4.97996e-05) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003707 ± 0.000146186 (W 4.44972e-05) | 0.004018 ± 1.95021e-05 (W 7.62443e-05) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003407 ± 0.000123452 (W 4.73951e-05) | 0.003546 ± 0.000180436 (W 8.61934e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014227 ± 0.000100135 (W 0.0001155) | 0.034113 ± 0.00118511 (W 3.80697e-05) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012694 ± 7.26659e-05 (W 6.12234e-05) | 0.016981 ± 7.10422e-05 (W 7.76157e-05) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031248 ± 0.000491032 (W 8.22265e-05) | 0.054781 ± 0.000309662 (W 0.0442033) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.029155 ± 0.000419469 (W 8.36433e-05) | 0.038231 ± 0.00120239 (W 0.000423984) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.072183 ± 0.000174818 (W 0.000235648) | 0.108773 ± 0.00272244 (W 0.000193738) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.072895 ± 0.000396649 (W 0.00157012) | 0.111746 ± 0.0129312 (W 0.000398217) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.234336 ± 0.00177682 (W 0.00201412) | 0.535757 ± 0.00221294 (W 0.00212158) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.248222 ± 0.000923003 (W 0.00163251) | 0.533634 ± 0.00262784 (W 0.00193109) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.500292 ± 8.16721e-05 (W 0.00328686) | 1.24697 ± 0.00360802 (W 0.00434806) | - | 🔵 faer-fork 2.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.501304 ± 0.00297485 (W 0.00254629) | 1.24564 ± 0.00577943 (W 0.00218457) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011772 ± 0.000110274 (W 4.63681e-05) | 0.008917 ± 6.65833e-05 (W 6.32456e-05) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014707 ± 0.000170969 (W 4.96991e-05) | 0.01074 ± 0.000100732 (W 5.10715e-05) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.051987 ± 0.000759809 (W 7.31437e-05) | 0.041898 ± 0.000179556 (W 0.000112873) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.060052 ± 6.19785e-05 (W 0.000166223) | 0.046597 ± 0.000477953 (W 0.000232005) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.15551 ± 0.000435606 (W 0.00122842) | 0.119853 ± 0.000417689 (W 0.00185118) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.181117 ± 0.000223842 (W 0.00146597) | 0.135559 ± 0.000873151 (W 0.00203215) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006152 ± 6.00028e-05 (W 0.000105453) | 0.008375 ± 2.04206e-05 (W 7.15136e-05) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006352 ± 0.000153948 (W 0.000203532) | 0.008466 ± 9.59184e-05 (W 6.38608e-05) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014457 ± 0.000196927 (W 0.000110315) | 0.019677 ± 0.000561772 (W 0.000413504) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01662 ± 0.000195986 (W 0.000239481) | 0.019637 ± 2.6839e-05 (W 0.000365819) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.032611 ± 0.000466208 (W 0.000718992) | 0.056435 ± 0.000790772 (W 0.00135505) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.033983 ± 0.000379194 (W 0.000753102) | 0.057445 ± 0.00091671 (W 0.00448248) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006602 ± 4.41626e-05 (W 7.68193e-05) | 0.007624 ± 9.13072e-05 (W 6.866e-05) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008516 ± 4.04145e-05 (W 7.8867e-05) | 0.007344 ± 0.000115717 (W 0.000138594) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.028333 ± 0.000234609 (W 0.000365906) | 0.036568 ± 0.000542553 (W 0.00028864) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.029375 ± 3.55949e-05 (W 0.00019771) | 0.030887 ± 0.000106353 (W 0.000376797) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.067955 ± 0.000581662 (W 0.00101323) | 0.090398 ± 0.000839164 (W 0.000555225) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.073046 ± 0.000621652 (W 0.0025103) | 0.078005 ± 1e-06 (W 0.0026438) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.03757 ± 0.000301055 (W 0.000174669) | 0.063296 ± 0.000695646 (W 0.000546451) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.033201 ± 0.000329597 (W 0.000133054) | 0.053901 ± 0.000531846 (W 0.00131178) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.113983 ± 0.00192484 (W 0.000929466) | 0.290932 ± 0.00440026 (W 0.00349242) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.10731 ± 0.000348976 (W 0.000463196) | 0.23345 ± 0.00248512 (W 0.00239514) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.208248 ± 0.00197212 (W 0.0023169) | 0.591719 ± 0.0052143 (W 0.00563718) | - | 🔵 faer-fork 2.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.240668 ± 0.000275273 (W 0.00281136) | 0.493509 ± 0.00543299 (W 0.00584693) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025397 ± 0.000158767 (W 0.000117983) | 0.051866 ± 0.000742808 (W 0.00123073) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.074098 ± 0.000346458 (W 0.000372053) | 0.251648 ± 0.00156831 (W 0.00251786) | - | 🔵 faer-fork 3.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.138696 ± 0.000924106 (W 0.000863065) | 0.506854 ± 0.00123322 (W 0.00647975) | - | 🔵 faer-fork 3.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007454 ± 4.93288e-05 (W 2.90207e-05) | 0.008455 ± 0.000181843 (W 6.50385e-05) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009928 ± 8.17945e-05 (W 5.59464e-05) | 0.008396 ± 5.80259e-05 (W 0.000182431) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.026159 ± 0.000207625 (W 0.000278116) | 0.03788 ± 3.4641e-05 (W 0.000295186) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.031588 ± 0.000166154 (W 0.000110083) | 0.031008 ± 0.00015064 (W 0.000293381) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.057265 ± 0.000314415 (W 0.000463798) | 0.0869 ± 0.000222361 (W 0.000378134) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.078465 ± 0.00021185 (W 0.00267722) | 0.07513 ± 0.000366132 (W 0.00233663) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.169906 ± 0.00229792 (W 0.000140133) | 0.182296 ± 0.0036371 (W 0.00147554) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.170647 ± 0.00112854 (W 0.00162225) | 0.18457 ± 0.000466601 (W 0.00218354) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.555013 ± 0.00274992 (W 0.00239147) | 0.91218 ± 0.00121329 (W 0.00185633) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.55941 ± 0.000980948 (W 0.00227215) | 0.912119 ± 0.00143884 (W 0.00236359) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.14583 ± 0.00274833 (W 0.00680109) | 2.11452 ± 0.00213245 (W 0.00244527) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.14715 ± 0.00199358 (W 0.00384021) | 2.12232 ± 0.00707936 (W 0.00710005) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.05918 ± 0.000284799 (W 0.000148055) | 0.141203 ± 0.000424294 (W 0.00162261) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.06012 ± 0.000521834 (W 0.000197619) | 0.141694 ± 0.00117067 (W 0.00202958) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.215917 ± 0.00093764 (W 0.00154085) | 0.515269 ± 0.00221032 (W 0.00249828) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.214105 ± 0.000617195 (W 0.00184402) | 0.516691 ± 0.00174457 (W 0.00262856) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.533893 ± 0.00165143 (W 0.00202401) | 1.18198 ± 0.00426065 (W 0.00291899) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.539525 ± 0.00106235 (W 0.00213949) | 1.19458 ± 0.0036912 (W 0.00242864) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000961 ± 1.12694e-05 (W 2.96648e-05) | 0.000682 ± 2.0664e-05 (W 2.50998e-05) | - | 🔴 faer-fork 1.41× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002154 ± 5.7735e-06 (W 4.65102e-05) | 0.001613 ± 1.1547e-05 (W 6.63792e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003737 ± 9.07377e-05 (W 6.18061e-05) | 0.002875 ± 2.02567e-05 (W 4.05993e-05) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002795 ± 0.00020321 (W 9.62808e-05) | 0.002655 ± 0.000115036 (W 2.86356e-05) | 0.002995 ± 2.04206e-05 (W 0.000120724) | · | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003386 ± 1.0504e-05 (W 7.18631e-05) | 0.003337 ± 0.000149534 (W 6.44298e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01057 ± 0.000313084 (W 5.40759e-05) | 0.014868 ± 0.00104674 (W 5.09637e-05) | 0.012774 ± 0.000252171 (W 6.64831e-05) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012213 ± 0.000279321 (W 0.000124841) | 0.016541 ± 8.3865e-05 (W 8.4476e-05) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.024866 ± 5.03322e-05 (W 7.47141e-05) | 0.030587 ± 0.00108134 (W 4.32435e-05) | 0.035256 ± 0.000177305 (W 0.000164488) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.028222 ± 7.508e-05 (W 7.45265e-05) | 0.03224 ± 0.00112854 (W 0.000136475) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.111537 ± 0.000418092 (W 0.0020519) | 0.108943 ± 0.000640709 (W 0.00125739) | 0.094846 ± 0.000471829 (W 0.000172323) | · | 🔴 faer-fork 1.18× slower than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.11229 ± 0.000346931 (W 0.00160694) | 0.108903 ± 0.000266016 (W 0.000376099) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.572914 ± 0.00485183 (W 0.0030523) | 0.531884 ± 0.00110526 (W 0.00218942) | 0.379429 ± 0.00273746 (W 0.00230744) | 🔴 faer-fork 1.08× slower than `faer-upstream` | 🔴 faer-fork 1.51× slower than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.575731 ± 0.00134338 (W 0.00236895) | 0.534148 ± 0.00262396 (W 0.0020792) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.33849 ± 0.00395915 (W 0.00310701) | 1.24968 ± 0.0017687 (W 0.0031292) | 0.871993 ± 0.0045725 (W 0.00231296) | 🔴 faer-fork 1.07× slower than `faer-upstream` | 🔴 faer-fork 1.53× slower than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.35801 ± 0.00362023 (W 0.00645364) | 1.2757 ± 0.00144669 (W 0.00339819) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01082 ± 8.62168e-05 (W 2.12132e-05) | 0.008977 ± 0.000141846 (W 3.28634e-05) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012162 ± 0.000156771 (W 9.34735e-05) | 0.01062 ± 7.77839e-05 (W 0.000100679) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.050795 ± 0.000345853 (W 0.00164903) | 0.042509 ± 4.16333e-05 (W 0.000164888) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.053239 ± 0.000103523 (W 0.000165149) | 0.045676 ± 0.000635491 (W 0.000204583) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.14501 ± 0.00553054 (W 0.00171496) | 0.121185 ± 0.000300661 (W 0.00198818) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.15032 ± 0.000217233 (W 0.00283876) | 0.134612 ± 0.000601749 (W 0.00243558) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008486 ± 9.29157e-05 (W 0.000106669) | 0.008165 ± 3.21455e-05 (W 0.00013572) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008506 ± 3.78594e-05 (W 4.32435e-05) | 0.008215 ± 3.27159e-05 (W 4.02032e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.040335 ± 0.000162112 (W 0.000115456) | 0.040194 ± 0.000216749 (W 0.00212084) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.040376 ± 8.19776e-05 (W 0.00145334) | 0.040105 ± 9.07377e-05 (W 4.76445e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.123009 ± 0.000328938 (W 0.00044822) | 0.121928 ± 0.000850044 (W 0.000275361) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.124382 ± 0.000197569 (W 0.000154302) | 0.122088 ± 0.000766677 (W 9.21954e-05) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005139 ± 3.55012e-05 (W 4.30732e-05) | 0.005921 ± 0.000117189 (W 3.91152e-05) | 0.010299 ± 7.2111e-05 (W 0.000113134) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 2.00× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006732 ± 8.2282e-05 (W 0.000160729) | 0.007283 ± 1.47309e-05 (W 0.00014489) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020589 ± 0.000105532 (W 0.000193211) | 0.022522 ± 0.000119304 (W 6.10737e-05) | 0.034154 ± 1.95021e-05 (W 0.000179065) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.023824 ± 0.000170587 (W 0.000278079) | 0.025728 ± 0.000180666 (W 0.000212086) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.049182 ± 6.80686e-05 (W 0.0017399) | 0.053289 ± 0.000185831 (W 0.000210901) | 0.076483 ± 0.00032377 (W 0.000329356) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.061465 ± 0.000472745 (W 0.00244523) | 0.066083 ± 0.000365285 (W 0.0020081) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.028112 ± 5.53203e-05 (W 0.000140408) | 0.030136 ± 0.000168009 (W 0.000166659) | 0.02167 ± 0.000268099 (W 0.000180251) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔴 faer-fork 1.30× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.030436 ± 0.000138684 (W 0.000115699) | 0.032731 ± 0.000108721 (W 0.00128416) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.103804 ± 0.000730954 (W 0.00202355) | 0.113161 ± 0.00047944 (W 0.00181255) | 0.136424 ± 0.000967503 (W 0.0018901) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.109224 ± 0.000639575 (W 0.0018871) | 0.11883 ± 0.000576077 (W 0.000167828) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.242271 ± 0.00151708 (W 0.00184473) | 0.262307 ± 0.00130333 (W 0.00288783) | 0.369767 ± 0.00205938 (W 0.00407148) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.276775 ± 0.000509319 (W 0.0014489) | 0.29961 ± 0.000637522 (W 0.00112021) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.016511 ± 3.78594e-05 (W 0.000143315) | 0.018825 ± 5.7735e-06 (W 0.000236041) | 0.01057 ± 0.000151327 (W 0.000125817) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔴 faer-fork 1.56× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067726 ± 0.000336601 (W 0.000119299) | 0.077544 ± 0.00025204 (W 0.000331457) | 0.0539 ± 0.000125819 (W 0.000131675) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔴 faer-fork 1.26× slower than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.144198 ± 0.00152585 (W 0.00191899) | 0.168834 ± 0.0016022 (W 0.0019134) | 0.167692 ± 0.00091586 (W 0.00237536) | 🔵 faer-fork 1.17× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.005931 ± 5.03322e-05 (W 4.19202e-05) | 0.006342 ± 1e-05 (W 5.26308e-05) | 0.010339 ± 0.000125033 (W 0.000161933) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.74× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007544 ± 4.67654e-05 (W 5.8414e-05) | 0.007805 ± 2.3388e-05 (W 6.17357e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021069 ± 0.000210571 (W 9.23591e-05) | 0.023193 ± 0.000309171 (W 9.0212e-05) | 0.033522 ± 0.000186065 (W 0.000160746) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.025017 ± 4.39356e-05 (W 0.000142655) | 0.0266 ± 5.68624e-05 (W 6.83023e-05) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.049452 ± 0.00026119 (W 0.000178091) | 0.05378 ± 0.000240327 (W 0.00203101) | 0.075079 ± 0.00054129 (W 0.000367923) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.063598 ± 0.000333467 (W 0.00253829) | 0.065502 ± 0.000449926 (W 0.00270832) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.168705 ± 0.00114769 (W 0.000543747) | 0.183811 ± 0.000539344 (W 0.00226612) | 0.163885 ± 0.000183118 (W 0.00192718) | 🔵 faer-fork 1.09× faster than `faer-upstream` | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.169575 ± 0.000813745 (W 0.000397824) | 0.185114 ± 0.00112686 (W 0.00174432) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.692534 ± 0.0101106 (W 0.00363688) | 0.960935 ± 0.00410765 (W 0.00331731) | 0.68383 ± 0.00067908 (W 0.00232847) | 🔵 faer-fork 1.39× faster than `faer-upstream` | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.693909 ± 0.00732998 (W 0.00193702) | 0.963881 ± 0.00411621 (W 0.0033471) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.58052 ± 0.00344073 (W 0.00548644) | 2.1548 ± 0.00678145 (W 0.00663126) | 1.5639 ± 0.00119882 (W 0.00441772) | 🔵 faer-fork 1.36× faster than `faer-upstream` | · |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.5852 ± 0.00319562 (W 0.00425211) | 2.2421 ± 0.00331089 (W 0.00505434) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058038 ± 0.000307198 (W 0.000184071) | 0.142094 ± 0.000149503 (W 0.000798202) | 0.056956 ± 0.000245495 (W 0.000128569) | 🔵 faer-fork 2.45× faster than `faer-upstream` | · |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.058519 ± 0.000406503 (W 0.000151447) | 0.142796 ± 0.000490796 (W 0.00183007) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.21556 ± 0.00211508 (W 0.00215015) | 0.540845 ± 0.00102392 (W 0.00286593) | 0.246668 ± 0.00103605 (W 0.00192112) | 🔵 faer-fork 2.51× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.211115 ± 0.00181519 (W 0.00148326) | 0.536293 ± 0.00185188 (W 0.00324797) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.490342 ± 0.000974854 (W 0.00303586) | 1.19092 ± 0.00632485 (W 0.00371006) | 0.586296 ± 0.00119196 (W 0.00249007) | 🔵 faer-fork 2.43× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.489531 ± 0.00316014 (W 0.00286267) | 1.19027 ± 0.00485279 (W 0.00423281) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000832 ± 6.08276e-06 (W 6.67031e-05) | 0.000791 ± 1.70392e-05 (W 3.51454e-05) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002114 ± 3.66106e-05 (W 5.93296e-05) | 0.001973 ± 1.73205e-05 (W 8.36558e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003366 ± 4.96689e-05 (W 4.93275e-05) | 0.003166 ± 2.51064e-05 (W 3.12442e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002785 ± 4.39356e-05 (W 3.07441e-05) | 0.002515 ± 0.000133493 (W 5.33854e-05) | 0.002986 ± 1.1547e-05 (W 0.000112472) | 🔴 faer-fork 1.11× slower than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003426 ± 1.0504e-05 (W 3.4641e-05) | 0.003317 ± 0.000155802 (W 4.886e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01078 ± 8.7178e-05 (W 4.03733e-05) | 0.014848 ± 0.00104197 (W 8.04133e-05) | 0.012704 ± 6.11583e-05 (W 0.000175727) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012524 ± 7.95634e-05 (W 9.40596e-05) | 0.016591 ± 0.000182511 (W 5.46461e-05) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.025458 ± 0.000131341 (W 6.41872e-05) | 0.032801 ± 0.000170587 (W 0.000247205) | 0.035436 ± 9.13072e-05 (W 0.000134682) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.028654 ± 8.0829e-05 (W 0.00010523) | 0.038101 ± 0.00138856 (W 0.000432163) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.076553 ± 0.00039054 (W 0.000202806) | 0.108612 ± 0.0017851 (W 0.000331927) | 0.100077 ± 0.000634193 (W 0.000454152) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.077615 ± 0.000586576 (W 0.000267111) | 0.108762 ± 0.00324013 (W 0.000353507) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.317003 ± 0.00133738 (W 0.00175268) | 0.53535 ± 0.00253978 (W 0.00237501) | 0.26316 ± 0.000828169 (W 0.00229796) | 🔵 faer-fork 1.69× faster than `faer-upstream` | 🔴 faer-fork 1.20× slower than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.327021 ± 0.00107317 (W 0.0023309) | 0.532435 ± 0.000265227 (W 0.0019962) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.771151 ± 0.0047319 (W 0.00472937) | 1.23397 ± 0.00522258 (W 0.00318407) | 0.572659 ± 0.0035995 (W 0.0027789) | 🔵 faer-fork 1.60× faster than `faer-upstream` | 🔴 faer-fork 1.35× slower than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.778003 ± 0.00266646 (W 0.00158037) | 1.23325 ± 0.00216678 (W 0.0029792) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011972 ± 9.81495e-05 (W 3.999e-05) | 0.008977 ± 4.98029e-05 (W 6.95428e-05) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013425 ± 0.000124231 (W 0.000122453) | 0.010499 ± 0.000187981 (W 5.62868e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.053019 ± 0.000595073 (W 0.000142408) | 0.042369 ± 0.000243023 (W 0.000109316) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.055654 ± 0.00955198 (W 0.00022697) | 0.046026 ± 0.00104942 (W 0.000114539) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.146414 ± 0.00196514 (W 0.00137759) | 0.119863 ± 0.000215948 (W 0.00180954) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.158636 ± 0.000490215 (W 0.00268042) | 0.132437 ± 0.00105391 (W 0.00223699) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006271 ± 7.14866e-05 (W 0.00016436) | 0.008516 ± 6.05007e-05 (W 4.08693e-05) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006493 ± 8.31084e-05 (W 0.000204669) | 0.008466 ± 8.24035e-05 (W 0.000111937) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.019296 ± 7.23418e-05 (W 0.000174946) | 0.019196 ± 0.000297271 (W 0.000256346) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.019637 ± 7.07602e-05 (W 0.000173804) | 0.019236 ± 0.000354444 (W 0.000249691) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.044563 ± 0.000208727 (W 0.000446761) | 0.04342 ± 0.000311407 (W 0.000175164) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.045064 ± 0.000469287 (W 0.000193136) | 0.043391 ± 0.000320725 (W 0.000254638) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00533 ± 7.99396e-05 (W 6.30079e-05) | 0.005701 ± 9.29157e-05 (W 5.74648e-05) | 0.01033 ± 0.000121355 (W 0.000154332) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.94× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006933 ± 4.72582e-05 (W 7.13127e-05) | 0.007154 ± 8.68927e-05 (W 7.59493e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02127 ± 7.59408e-05 (W 0.000144088) | 0.025748 ± 0.0012049 (W 0.00114289) | 0.034163 ± 0.000341215 (W 0.00022411) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.61× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.024346 ± 0.000360205 (W 0.000305441) | 0.029464 ± 0.000441423 (W 0.000287076) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.04286 ± 0.000152306 (W 0.000333761) | 0.05859 ± 0.000287918 (W 0.00165597) | 0.076473 ± 9.89966e-05 (W 0.00167259) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.78× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.058699 ± 0.000736769 (W 0.00218692) | 0.073196 ± 0.000579878 (W 0.00371042) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.029175 ± 0.0001005 (W 9.67171e-05) | 0.043051 ± 0.000244917 (W 0.000326233) | 0.021119 ± 0.000211369 (W 0.00012) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.031699 ± 0.000146401 (W 0.000186079) | 0.046257 ± 0.000296665 (W 0.00033765) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.095698 ± 0.000447439 (W 0.000472647) | 0.176028 ± 0.00135333 (W 0.00191681) | 0.09165 ± 0.000976081 (W 0.00182537) | 🔵 faer-fork 1.84× faster than `faer-upstream` | · |  |
| `qr/n64/b1 [api-inclusive]` | 0.102231 ± 0.000301458 (W 0.000288271) | 0.185205 ± 0.00116382 (W 0.00229979) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.185366 ± 0.00133415 (W 0.00194965) | 0.351023 ± 0.00276456 (W 0.00349228) | 0.241399 ± 0.00135557 (W 0.00180579) | 🔵 faer-fork 1.89× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.231541 ± 0.000334714 (W 0.00286559) | 0.394705 ± 0.000609549 (W 0.00350157) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.017472 ± 8.41843e-05 (W 0.000115794) | 0.03195 ± 0.000297232 (W 0.000582451) | 0.01075 ± 0.000133167 (W 0.000103199) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔴 faer-fork 1.63× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067446 ± 0.000427638 (W 0.000344862) | 0.147605 ± 0.00096695 (W 0.00158937) | 0.05371 ± 0.000385385 (W 0.000252341) | 🔵 faer-fork 2.19× faster than `faer-upstream` | 🔴 faer-fork 1.26× slower than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.131626 ± 0.00108906 (W 0.000401786) | 0.292466 ± 0.0020155 (W 0.00244834) | 0.167413 ± 0.000917841 (W 0.00221328) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.006512 ± 6.55744e-05 (W 3.67042e-05) | 0.006392 ± 6.71342e-05 (W 7.75713e-05) | 0.010289 ± 3.78594e-05 (W 0.000108256) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008165 ± 7.62911e-05 (W 3.72317e-05) | 0.007915 ± 7.13933e-05 (W 7.50333e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.022883 ± 0.000263881 (W 0.000170972) | 0.025657 ± 0.0001929 (W 0.000627294) | 0.033512 ± 0.000275228 (W 0.000187257) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.026058 ± 0.000263913 (W 0.000239419) | 0.029365 ± 0.000197536 (W 0.000283849) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.046487 ± 6.70099e-05 (W 0.000317649) | 0.056816 ± 0.000193357 (W 0.000396208) | 0.07521 ± 0.000197315 (W 0.000199643) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.06416 ± 0.000376387 (W 0.00246471) | 0.070241 ± 0.000551136 (W 0.00291132) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.173413 ± 0.00155882 (W 0.000469141) | 0.186809 ± 0.0054411 (W 0.00194757) | 0.173152 ± 0.00117657 (W 0.0017462) | 🔵 faer-fork 1.08× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.173893 ± 0.000782476 (W 0.00034731) | 0.186238 ± 0.00156405 (W 0.00229996) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.562176 ± 0.000766982 (W 0.00296595) | 0.950586 ± 0.0050035 (W 0.000945345) | 0.626657 ± 0.00231336 (W 0.00336208) | 🔵 faer-fork 1.69× faster than `faer-upstream` | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.561365 ± 0.000826926 (W 0.00181236) | 0.952262 ± 0.00290754 (W 0.00204766) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.15035 ± 0.00358654 (W 0.00417069) | 2.09342 ± 0.00827172 (W 0.00876279) | 1.32417 ± 0.00365531 (W 0.00453579) | 🔵 faer-fork 1.82× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15578 ± 0.00564836 (W 0.00299246) | 2.09232 ± 0.00400749 (W 0.00503376) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.05884 ± 0.000347501 (W 0.000185141) | 0.142797 ± 0.000380934 (W 0.000293895) | 0.058519 ± 0.000808389 (W 0.000198189) | 🔵 faer-fork 2.43× faster than `faer-upstream` | · |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.05953 ± 0.000186897 (W 0.000236028) | 0.143067 ± 0.000426509 (W 0.00140053) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.218537 ± 0.00186187 (W 0.00178023) | 0.538913 ± 0.00410229 (W 0.00306297) | 0.249927 ± 0.00125167 (W 0.00219024) | 🔵 faer-fork 2.47× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.221311 ± 0.000707223 (W 0.00149025) | 0.540069 ± 0.00247613 (W 0.00270295) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.524145 ± 0.00190834 (W 0.00246379) | 1.20007 ± 0.00552773 (W 0.00351355) | 0.584477 ± 0.000536025 (W 0.00249501) | 🔵 faer-fork 2.29× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.523995 ± 0.00226459 (W 0.00237095) | 1.20071 ± 0.00820842 (W 0.00267873) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001022 ± 0 (W 3.78153e-05) | 0.000792 ± 5.7735e-07 (W 6.695e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002354 ± 2.62742e-05 (W 4.13437e-05) | 0.001854 ± 3.4356e-05 (W 8.66095e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004038 ± 1.73205e-05 (W 3.11978e-05) | 0.003246 ± 2.08167e-05 (W 3.28816e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002505 ± 2.64575e-05 (W 2.09093e-05) | 0.002465 ± 0.000176466 (W 4.71402e-05) | 0.002996 ± 1.73205e-05 (W 0.000135167) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003086 ± 4.6188e-05 (W 4.69042e-05) | 0.003476 ± 0.000185041 (W 9.4246e-05) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01069 ± 0.000590498 (W 3.89127e-05) | 0.012974 ± 0.00104731 (W 4.14729e-05) | 0.012734 ± 0.000104978 (W 0.000126389) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012414 ± 0.000375581 (W 0.000152523) | 0.014677 ± 5.85947e-05 (W 4.0635e-05) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.025197 ± 0.000454258 (W 8.34865e-05) | 0.031357 ± 0.0017156 (W 0.000643193) | 0.034975 ± 0.000313307 (W 0.000135563) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.028674 ± 0.000611522 (W 0.000159844) | 0.03727 ± 0.00106422 (W 0.000356714) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.072113 ± 0.000315628 (W 0.000314263) | 0.108722 ± 0.00582603 (W 0.00204928) | 0.101297 ± 0.000457015 (W 0.000310406) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` | `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.072565 ± 0.000545008 (W 0.000303944) | 0.109214 ± 0.0018035 (W 0.000539103) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.234717 ± 0.0016037 (W 0.00243099) | 0.534625 ± 0.00166846 (W 0.00300975) | 0.250071 ± 0.00079695 (W 0.00163819) | 🔵 faer-fork 2.28× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.253837 ± 0.00400836 (W 0.00290886) | 0.532541 ± 0.00207726 (W 0.00227273) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.500913 ± 0.00159474 (W 0.00190619) | 1.24463 ± 0.00312848 (W 0.00502841) | 0.563369 ± 0.00278485 (W 0.00328169) | 🔵 faer-fork 2.48× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.505761 ± 0.0015145 (W 0.0037061) | 1.24388 ± 0.00120559 (W 0.00514828) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011921 ± 0.00017164 (W 5.94138e-05) | 0.008926 ± 3.04467e-05 (W 6.66956e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013184 ± 0.000304808 (W 7.14143e-05) | 0.010549 ± 6.37207e-05 (W 4.21462e-05) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.052397 ± 0.000246423 (W 0.000173405) | 0.042058 ± 0.000186022 (W 0.000183306) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.054672 ± 9.3602e-05 (W 0.000194711) | 0.046066 ± 0.000390489 (W 0.000160144) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.154935 ± 0.000487911 (W 0.000780439) | 0.120742 ± 0.000714665 (W 0.00177717) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.167321 ± 0.0010369 (W 0.00282914) | 0.133548 ± 0.00136651 (W 0.00248498) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005972 ± 3.5171e-05 (W 6.75218e-05) | 0.008466 ± 4.41626e-05 (W 0.000111491) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005941 ± 5.19615e-05 (W 0.000126174) | 0.008536 ± 6.02771e-05 (W 0.00012419) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014968 ± 0.000265651 (W 0.000139776) | 0.020057 ± 0.000271932 (W 0.000211046) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014467 ± 0.000110259 (W 0.000185042) | 0.020138 ± 0.000261094 (W 0.00031259) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.030517 ± 0.00022519 (W 0.000196286) | 0.055042 ± 6.65658e-05 (W 0.000893164) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.030175 ± 9.00019e-05 (W 0.000243307) | 0.054863 ± 0.001069 (W 0.00102136) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00529 ± 2.88675e-05 (W 8.03816e-05) | 0.00576 ± 0.000205297 (W 0.000189972) | 0.01039 ± 3.56417e-05 (W 0.000166383) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.96× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006813 ± 3.23574e-05 (W 9.077e-05) | 0.007153 ± 8.96289e-05 (W 0.000146926) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020818 ± 0.00020656 (W 0.000188693) | 0.026108 ± 0.000148561 (W 0.000400879) | 0.034153 ± 0.000418427 (W 0.00204426) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.64× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024456 ± 5.0797e-05 (W 0.000323729) | 0.029805 ± 0.000193081 (W 0.00038262) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.042379 ± 0.000306862 (W 0.000509005) | 0.060631 ± 0.000523672 (W 0.000345667) | 0.075118 ± 0.000766849 (W 0.000312865) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔵 faer-fork 1.77× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.060933 ± 0.000379015 (W 0.00248546) | 0.07618 ± 0.000429665 (W 0.00223754) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.029144 ± 0.000138529 (W 8.93432e-05) | 0.052157 ± 0.000656768 (W 0.00110459) | 0.02119 ± 0.00020576 (W 0.000117593) | 🔵 faer-fork 1.79× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.031559 ± 0.000103055 (W 0.000203389) | 0.054942 ± 0.000300523 (W 0.000454756) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.095388 ± 0.000200177 (W 0.00029145) | 0.228495 ± 0.0029167 (W 0.00226638) | 0.084968 ± 0.000614606 (W 0.00154444) | 🔵 faer-fork 2.40× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.101319 ± 0.000575425 (W 0.000378019) | 0.231241 ± 0.00341417 (W 0.00282822) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.180315 ± 0.00121027 (W 0.00299053) | 0.457312 ± 0.00597696 (W 0.00315203) | 0.21736 ± 0.000766087 (W 0.00220245) | 🔵 faer-fork 2.54× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.225315 ± 0.00011563 (W 0.00276415) | 0.489692 ± 0.00308177 (W 0.00264103) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.017242 ± 4.93288e-05 (W 9.27971e-05) | 0.040585 ± 0.000904514 (W 0.001509) | 0.01061 ± 0.000216049 (W 0.000140909) | 🔵 faer-fork 2.35× faster than `faer-upstream` | 🔴 faer-fork 1.63× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067074 ± 0.000342228 (W 0.000219969) | 0.193951 ± 0.00131159 (W 0.00269989) | 0.053831 ± 0.000877752 (W 0.00013102) | 🔵 faer-fork 2.89× faster than `faer-upstream` | 🔴 faer-fork 1.25× slower than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.129471 ± 0.00112787 (W 0.000285917) | 0.377984 ± 0.00550469 (W 0.00419363) | 0.165895 ± 0.00114616 (W 0.00223376) | 🔵 faer-fork 2.92× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.006592 ± 4.04145e-05 (W 5.26308e-05) | 0.006271 ± 2.6839e-05 (W 7.18345e-05) | 0.010379 ± 5.08626e-05 (W 0.000150364) | · | 🔵 faer-fork 1.57× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008055 ± 3.21455e-05 (W 5.69667e-05) | 0.007775 ± 0.00010504 (W 8.30139e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.023463 ± 0.000250779 (W 0.000319875) | 0.025928 ± 0.000227156 (W 0.000300581) | 0.033613 ± 8.27063e-05 (W 0.0001977) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.025818 ± 7e-05 (W 0.000139235) | 0.030006 ± 0.000177156 (W 0.000333242) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.047068 ± 0.000470946 (W 0.000275874) | 0.059361 ± 0.000166304 (W 0.000486644) | 0.07496 ± 0.000348199 (W 0.00189896) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.066324 ± 0.000275466 (W 0.00262036) | 0.073647 ± 0.000301583 (W 0.00148412) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.171496 ± 0.000252951 (W 0.00136347) | 0.186087 ± 0.0101462 (W 0.00303745) | 0.173623 ± 0.000829608 (W 0.00173469) | 🔵 faer-fork 1.09× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.173072 ± 0.000579836 (W 0.000257776) | 0.185956 ± 0.00116319 (W 0.00381839) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.560223 ± 0.00138914 (W 0.0029751) | 0.955419 ± 0.00138547 (W 0.00106879) | 0.61238 ± 0.00063859 (W 0.00320036) | 🔵 faer-fork 1.71× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.567647 ± 0.00223783 (W 0.00195323) | 0.957923 ± 0.00128083 (W 0.00157094) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.15476 ± 0.00477718 (W 0.00489172) | 2.11374 ± 0.00123462 (W 0.0031316) | 1.265 ± 0.00194233 (W 0.0034411) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15967 ± 0.00643209 (W 0.00394202) | 2.11086 ± 0.00437283 (W 0.00525532) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.05914 ± 0.000319739 (W 8.15475e-05) | 0.142532 ± 0.000451013 (W 0.000246893) | 0.057247 ± 0.000357562 (W 0.00144624) | 🔵 faer-fork 2.41× faster than `faer-upstream` | · |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.059489 ± 0.000281518 (W 0.000146626) | 0.143186 ± 0.000825918 (W 0.000294384) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.217074 ± 0.00226622 (W 0.0019618) | 0.537451 ± 0.00217262 (W 0.00226505) | 0.249525 ± 0.000246033 (W 0.0020982) | 🔵 faer-fork 2.48× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.218377 ± 0.00133982 (W 0.00152257) | 0.540276 ± 0.00340086 (W 0.00232582) | - | 🔵 faer-fork 2.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.541257 ± 0.000608091 (W 0.00236367) | 1.18694 ± 0.00592239 (W 0.00359384) | 0.577025 ± 0.000675636 (W 0.00228875) | 🔵 faer-fork 2.19× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.550984 ± 0.00070313 (W 0.00204043) | 1.20192 ± 0.00591284 (W 0.00240937) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000982 ± 1.52753e-05 (W 3.34664e-05) | 0.000771 ± 5.7735e-07 (W 8.37514e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002315 ± 6.07481e-05 (W 4.1944e-05) | 0.001843 ± 2.09841e-05 (W 0.000102006) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003987 ± 2.08167e-05 (W 7.16812e-05) | 0.003206 ± 4.2194e-05 (W 5.31977e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005791 ± 0.000118462 (W 8.79204e-05) | 0.010259 ± 7.86787e-05 (W 6.57366e-05) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005259 ± 4.50592e-05 (W 0.000111862) | 0.004889 ± 7.76745e-05 (W 6.68072e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.033011 ± 0.000176163 (W 0.00013639) | 0.07539 ± 0.000179101 (W 0.000251526) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.031368 ± 0.000175187 (W 0.000155959) | 0.029184 ± 0.0002405 (W 7.89303e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.087583 ± 0.000231805 (W 0.00036898) | 0.439737 ± 0.00420778 (W 0.00227321) | - | 🔵 faer-fork 5.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.086982 ± 0.000794008 (W 0.000323646) | 0.082263 ± 0.000409337 (W 0.000209322) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.141433 ± 0.000320877 (W 0.00220909) | 0.137766 ± 0.0141531 (W 0.00162469) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.142775 ± 0.000481785 (W 0.00204301) | 0.137406 ± 0.000165581 (W 0.00195698) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.737671 ± 0.00411362 (W 0.00249888) | 0.699621 ± 0.00347392 (W 0.00220229) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.741556 ± 0.00339016 (W 0.00163248) | 0.722643 ± 0.00330554 (W 0.00396653) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.797 ± 0.00801247 (W 0.00423618) | 1.71291 ± 0.00494449 (W 0.00443363) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.80878 ± 0.00543965 (W 0.00344649) | 1.71183 ± 0.00636422 (W 0.00242289) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017062 ± 0.000115326 (W 7.30068e-05) | 0.013966 ± 0.000138601 (W 5.71078e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.020939 ± 4.04145e-05 (W 0.00141041) | 0.01595 ± 0.000131149 (W 7.6702e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.099135 ± 0.000760287 (W 0.00152319) | 0.084707 ± 0.000205508 (W 0.00155291) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.108292 ± 0.000538463 (W 0.00063178) | 0.091782 ± 0.00100262 (W 0.00189032) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.286175 ± 0.00155878 (W 0.00174854) | 0.266345 ± 0.000991064 (W 0.00191859) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.351277 ± 0.00153317 (W 0.00206657) | 0.314007 ± 0.00261003 (W 0.00297277) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028503 ± 0.000179528 (W 0.000102778) | 0.027882 ± 0.000142244 (W 0.000118597) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028363 ± 0.000150444 (W 7.82828e-05) | 0.028072 ± 0.00020664 (W 8.65985e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.159668 ± 0.000434363 (W 0.0022074) | 0.158004 ± 0.000693065 (W 0.00176666) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.16056 ± 0.000589102 (W 0.00219752) | 0.158054 ± 0.00104342 (W 0.00213348) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.471678 ± 0.00127608 (W 0.00142538) | 0.466688 ± 0.00398073 (W 0.00209773) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.470018 ± 0.000370872 (W 0.000987246) | 0.467384 ± 0.00119352 (W 0.00251494) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012313 ± 9.07377e-05 (W 6.14003e-05) | 0.01072 ± 8.7178e-05 (W 6.98011e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.014868 ± 3.84318e-05 (W 5.72713e-05) | 0.011652 ± 2.36714e-05 (W 9.27362e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.053199 ± 0.000218276 (W 0.000126251) | 0.055123 ± 0.000327017 (W 0.000234732) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.062406 ± 6.65833e-05 (W 0.00152482) | 0.054301 ± 0.000176353 (W 0.000126965) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.140091 ± 0.000816918 (W 0.00187834) | 0.145783 ± 0.00124213 (W 0.00176584) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.207736 ± 0.00239438 (W 0.00317295) | 0.190584 ± 0.00225633 (W 0.00210194) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.073687 ± 0.000387918 (W 0.000447776) | 0.078306 ± 0.000509514 (W 0.00163533) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.068697 ± 0.000433283 (W 0.000340692) | 0.068798 ± 0.000468577 (W 0.000298199) | - | · | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.294417 ± 0.000794054 (W 0.0018443) | 0.391451 ± 0.00181422 (W 0.00317448) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.334242 ± 0.000712553 (W 0.00379947) | 0.337066 ± 0.000465526 (W 0.00317511) | - | · | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.714839 ± 0.00346312 (W 0.00182242) | 0.912837 ± 0.00552981 (W 0.00119439) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.852224 ± 0.00300161 (W 0.00390515) | 0.859257 ± 0.00190973 (W 0.00185441) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.040866 ± 0.000156558 (W 0.000181175) | 0.046326 ± 0.000417797 (W 0.000254275) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.145681 ± 0.000584421 (W 0.00179055) | 0.237251 ± 0.00215733 (W 0.00203378) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.34923 ± 0.00151214 (W 0.00242917) | 0.538246 ± 0.00352551 (W 0.00242216) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012603 ± 0.000131149 (W 7.58235e-05) | 0.011992 ± 0.000110151 (W 6.32234e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01658 ± 0.000168049 (W 0.000157798) | 0.013395 ± 6.02771e-05 (W 4.64683e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.051466 ± 0.000324146 (W 8.78635e-05) | 0.056936 ± 0.000378083 (W 0.000235309) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.06407 ± 0.000242776 (W 0.00197789) | 0.056505 ± 0.000142945 (W 0.00185507) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.135722 ± 0.000221378 (W 0.00165342) | 0.149128 ± 0.000462343 (W 0.00141892) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.211485 ± 0.00406516 (W 0.00265899) | 0.192318 ± 8.60484e-05 (W 0.00294547) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.22049 ± 0.00307355 (W 0.00188275) | 0.232432 ± 0.00114761 (W 0.00191329) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.224419 ± 0.00173846 (W 0.0020437) | 0.234967 ± 0.00154429 (W 0.00154411) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.0187 ± 0.00575835 (W 0.00162444) | 1.21538 ± 0.0020884 (W 0.00239002) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.01831 ± 0.003402 (W 0.00215947) | 1.2163 ± 0.00625387 (W 0.00213905) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.46871 ± 0.0129089 (W 0.00422873) | 3.13122 ± 0.0202858 (W 0.00531208) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.47913 ± 0.00903376 (W 0.00412502) | 3.11576 ± 0.010725 (W 0.00364815) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.074048 ± 0.000147309 (W 0.000108126) | 0.16126 ± 0.00109156 (W 0.0015732) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.073727 ± 0.000242838 (W 0.00188675) | 0.159858 ± 0.000726909 (W 0.00161664) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.311449 ± 0.00106053 (W 0.00237549) | 0.646268 ± 0.00380169 (W 0.00232365) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.31217 ± 0.000598434 (W 0.00244669) | 0.645005 ± 0.00285433 (W 0.00178571) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.807038 ± 0.00351496 (W 0.0020368) | 1.4989 ± 0.0108837 (W 0.00468478) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.811058 ± 0.00341986 (W 0.00282127) | 1.50344 ± 0.00404196 (W 0.00441875) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001182 ± 2.08167e-05 (W 8.5512e-05) | 0.001072 ± 2.3094e-05 (W 7.52994e-05) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002916 ± 3.40637e-05 (W 5.13352e-05) | 0.002755 ± 3.78594e-05 (W 5.93296e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00533 ± 4.93288e-05 (W 6.83462e-05) | 0.00496 ± 5.11892e-05 (W 3.92594e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005982 ± 4.51331e-05 (W 6.96369e-05) | 0.010329 ± 5.7735e-06 (W 3.83406e-05) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0053 ± 3.64737e-05 (W 8.51311e-05) | 0.004949 ± 2.08167e-05 (W 3.56413e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.030707 ± 0.000275741 (W 0.000268924) | 0.075901 ± 6.50641e-05 (W 0.000238316) | - | 🔵 faer-fork 2.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.030026 ± 0.000210571 (W 0.000101637) | 0.029105 ± 9.79302e-05 (W 0.000138374) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.082644 ± 0.000337817 (W 0.000267687) | 0.138578 ± 0.00367363 (W 0.120131) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.082754 ± 7.3328e-05 (W 0.000466825) | 0.086712 ± 0.000581149 (W 0.0011024) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.102752 ± 0.00161305 (W 0.000295599) | 0.137827 ± 0.000133508 (W 0.00203263) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.10339 ± 0.000319519 (W 0.000274061) | 0.137676 ± 0.000488911 (W 0.00247661) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.432886 ± 0.000458594 (W 0.00155804) | 0.666972 ± 0.00282125 (W 0.00234446) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.434079 ± 0.00211121 (W 0.00252652) | 0.669897 ± 0.00278004 (W 0.00203987) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.920664 ± 0.00665557 (W 0.00214713) | 1.5386 ± 0.00752971 (W 0.00356336) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.927167 ± 0.00270709 (W 0.00389907) | 1.54494 ± 0.00360888 (W 0.0035007) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018905 ± 0.00147162 (W 0.000104976) | 0.014066 ± 0.00015019 (W 0.000138554) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.02179 ± 0.00016885 (W 9.28073e-05) | 0.01584 ± 6.245e-05 (W 9.68468e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.098123 ± 0.00117762 (W 0.000773186) | 0.085099 ± 0.000407836 (W 0.000253821) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.108502 ± 0.000335555 (W 0.000580592) | 0.091722 ± 0.000451048 (W 0.000272875) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.233174 ± 0.00306944 (W 0.00247281) | 0.267067 ± 0.000780699 (W 0.00227252) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.304537 ± 0.00303711 (W 0.0020372) | 0.267498 ± 0.0011703 (W 0.00265278) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.014848 ± 0.000165742 (W 0.000239845) | 0.028342 ± 0.000106927 (W 0.000145366) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.014406 ± 0.000145225 (W 0.000166238) | 0.028282 ± 6.84617e-05 (W 0.000145929) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.061635 ± 0.000247388 (W 0.000345101) | 0.060583 ± 3.40783e-05 (W 0.000463641) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.061933 ± 0.000196031 (W 0.001318) | 0.060012 ± 0.000720542 (W 0.000396911) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.153967 ± 0.000163127 (W 0.00209689) | 0.150781 ± 0.00204809 (W 0.00199111) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.153902 ± 0.000960189 (W 0.000521233) | 0.149839 ± 0.000435613 (W 0.000772617) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012624 ± 3.25115e-05 (W 5.32381e-05) | 0.01077 ± 3e-05 (W 4.82701e-05) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015248 ± 8.5967e-05 (W 6.61982e-05) | 0.011802 ± 0.000121244 (W 6.59545e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.059892 ± 0.000234578 (W 0.000734538) | 0.054902 ± 0.000233432 (W 0.000345284) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.061554 ± 0.000894256 (W 0.00154793) | 0.054762 ± 0.00032528 (W 0.000373092) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.136825 ± 0.0018827 (W 0.00115922) | 0.143206 ± 0.000372872 (W 0.00093837) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.188412 ± 0.00164139 (W 0.00382021) | 0.140271 ± 0.00220426 (W 0.0023822) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.070472 ± 0.000501033 (W 0.000781471) | 0.092122 ± 0.000181302 (W 0.000537057) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0644 ± 0.000307225 (W 0.000344639) | 0.082183 ± 0.000350571 (W 0.000514262) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.189794 ± 0.00100948 (W 0.00241656) | 0.402802 ± 0.00156257 (W 0.00292716) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.221482 ± 0.00184578 (W 0.00258931) | 0.34397 ± 0.0020595 (W 0.00149465) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.416067 ± 0.00275984 (W 0.00214749) | 0.76537 ± 0.00302687 (W 0.00497086) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.519006 ± 0.00046288 (W 0.00358281) | 0.703555 ± 0.00170318 (W 0.00222685) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.041998 ± 0.000264456 (W 0.000652347) | 0.060042 ± 0.000240604 (W 0.0017546) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.126535 ± 0.00052433 (W 0.000594745) | 0.291723 ± 0.00079462 (W 0.00208889) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.268209 ± 0.00119506 (W 0.00225395) | 0.577575 ± 0.00181118 (W 0.00212982) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013395 ± 2.08167e-05 (W 5.85662e-05) | 0.011922 ± 8.66025e-05 (W 4.55335e-05) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016871 ± 9.02053e-05 (W 7.51419e-05) | 0.013185 ± 7.42765e-05 (W 0.00143089) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.055153 ± 0.000746929 (W 0.000567825) | 0.056756 ± 0.000473318 (W 0.000140905) | - | · | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.066373 ± 0.00116767 (W 0.000979627) | 0.056626 ± 0.000413812 (W 0.00190192) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.113171 ± 0.00072055 (W 0.000459708) | 0.14501 ± 0.000709733 (W 0.00150877) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.191927 ± 0.00107184 (W 0.00480176) | 0.147085 ± 0.00251311 (W 0.00211716) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.213922 ± 0.0011172 (W 0.00206102) | 0.234076 ± 0.000148956 (W 0.00158645) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.215554 ± 0.000650771 (W 0.00124548) | 0.233876 ± 0.0131834 (W 0.00266278) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.741641 ± 0.00150432 (W 0.00239385) | 1.14383 ± 0.00413696 (W 0.00180693) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.748871 ± 0.00418186 (W 0.00357956) | 1.14896 ± 0.00145816 (W 0.00274319) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54812 ± 0.00972998 (W 0.00407835) | 2.77258 ± 0.0127355 (W 0.00416123) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.56037 ± 0.00520971 (W 0.00336587) | 2.77726 ± 0.010476 (W 0.00267645) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075119 ± 0.000398711 (W 0.000132281) | 0.160459 ± 0.000223943 (W 0.00191616) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.075431 ± 0.000636505 (W 0.000122091) | 0.16091 ± 0.00060785 (W 0.00198398) | - | 🔵 faer-fork 2.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.318684 ± 0.00253008 (W 0.00203783) | 0.650592 ± 0.0008265 (W 0.00193346) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.319185 ± 0.0017064 (W 0.00183553) | 0.65002 ± 0.00121513 (W 0.00271307) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.723575 ± 0.00300146 (W 0.00254937) | 1.50528 ± 0.00234875 (W 0.00248798) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.723965 ± 0.00513795 (W 0.00250752) | 1.50484 ± 0.00631467 (W 0.00266847) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001453 ± 1e-05 (W 3.1372e-05) | 0.001072 ± 2e-05 (W 3.42053e-05) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003477 ± 3.10966e-05 (W 2.30065e-05) | 0.002695 ± 3.84318e-05 (W 2.91925e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006151 ± 3.751e-05 (W 6.34374e-05) | 0.00502 ± 3.99625e-05 (W 4.73212e-05) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005761 ± 3.05014e-05 (W 5.96674e-05) | 0.010319 ± 6.70099e-05 (W 2.96513e-05) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005209 ± 7.25971e-05 (W 8.8172e-05) | 0.004909 ± 5.56776e-05 (W 4.43847e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.030657 ± 0.000196044 (W 0.000172092) | 0.075609 ± 0.000212331 (W 0.000246143) | - | 🔵 faer-fork 2.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.030377 ± 0.000102715 (W 0.000170424) | 0.029244 ± 0.000101737 (W 7.41654e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.082384 ± 0.000312241 (W 0.000370513) | 0.138786 ± 0.000507521 (W 0.00157764) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.082341 ± 0.000263627 (W 0.00152823) | 0.087291 ± 0.00022754 (W 0.00164205) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.097532 ± 0.000270962 (W 0.000392432) | 0.137303 ± 0.000175762 (W 0.00197553) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.098762 ± 0.000556598 (W 0.0012182) | 0.138088 ± 0.000313711 (W 0.00177105) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.328596 ± 0.00657004 (W 0.00649387) | 0.669687 ± 0.00263393 (W 0.00339864) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.322931 ± 0.00577398 (W 0.00812051) | 0.668465 ± 0.00246115 (W 0.00211443) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.681748 ± 0.00565669 (W 0.00551849) | 1.53982 ± 0.00365745 (W 0.00626537) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.689996 ± 0.00501594 (W 0.00285878) | 1.54472 ± 0.00851032 (W 0.00258748) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018363 ± 6.22923e-05 (W 7.96191e-05) | 0.013926 ± 6.35085e-05 (W 9.85657e-05) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0218 ± 0.000104213 (W 8.38523e-05) | 0.015859 ± 4.03361e-05 (W 9.93141e-05) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.102459 ± 0.000921978 (W 0.00103281) | 0.084528 ± 0.000813067 (W 0.000255775) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.111869 ± 0.0006147 (W 0.000813785) | 0.091949 ± 0.000471993 (W 0.00209847) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.232643 ± 0.00150049 (W 0.0021974) | 0.266717 ± 0.000218383 (W 0.00202648) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.307773 ± 0.00121056 (W 0.00265765) | 0.265856 ± 0.000200909 (W 0.00170802) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.01063 ± 2.64575e-05 (W 0.000143977) | 0.028083 ± 0.000119654 (W 6.66723e-05) | - | 🔵 faer-fork 2.64× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.01056 ± 0.000136077 (W 0.000199746) | 0.028292 ± 0.00012108 (W 0.000101288) | - | 🔵 faer-fork 2.68× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.03767 ± 0.00072781 (W 0.000348646) | 0.060653 ± 0.000117304 (W 0.000302848) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.03807 ± 0.000236913 (W 0.000300675) | 0.060843 ± 0.000204769 (W 0.000364047) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.087533 ± 0.000638846 (W 0.000358289) | 0.15042 ± 0.00126031 (W 0.0013234) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.088105 ± 0.000319929 (W 0.00251647) | 0.150921 ± 0.00065626 (W 0.00176806) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012713 ± 9.53432e-05 (W 9.38845e-05) | 0.0109 ± 1.95021e-05 (W 0.000115336) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015159 ± 6.11583e-05 (W 0.000152428) | 0.011862 ± 1.73205e-05 (W 0.000123147) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.056334 ± 0.000716412 (W 0.000522807) | 0.054983 ± 0.000360871 (W 0.00022737) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.062486 ± 0.000105532 (W 0.000435421) | 0.054751 ± 9.17624e-05 (W 0.000237245) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.14462 ± 0.00069933 (W 0.00188326) | 0.14496 ± 0.000940519 (W 0.00118157) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.188311 ± 0.000337503 (W 0.00222545) | 0.144136 ± 0.00084875 (W 0.0018156) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.073026 ± 0.000341705 (W 0.000753232) | 0.100525 ± 0.000602141 (W 0.00130948) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.063286 ± 0.000367334 (W 0.000526485) | 0.089717 ± 0.000515434 (W 0.00257784) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.18245 ± 0.00117009 (W 0.00184499) | 0.440249 ± 0.000754607 (W 0.004113) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.212141 ± 0.000719284 (W 0.00193152) | 0.385241 ± 0.00122251 (W 0.00392973) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.351408 ± 0.00151804 (W 0.00221823) | 0.82438 ± 0.0046318 (W 0.00664584) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.474143 ± 0.000820957 (W 0.00507806) | 0.77849 ± 0.00922522 (W 0.00305363) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.039945 ± 3.04467e-05 (W 0.000201851) | 0.068887 ± 0.00056717 (W 0.001467) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.121777 ± 0.000641421 (W 0.00191275) | 0.319325 ± 0.0022622 (W 0.00539427) | - | 🔵 faer-fork 2.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.242727 ± 0.000713374 (W 0.00194727) | 0.625745 ± 0.00942067 (W 0.00775288) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013325 ± 0.000136504 (W 5.86967e-05) | 0.011882 ± 0.000146118 (W 9.60588e-05) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016961 ± 5.54106e-05 (W 0.000105024) | 0.013195 ± 9.50631e-05 (W 0.00160245) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.056724 ± 6.13297e-05 (W 0.00031259) | 0.056646 ± 0.000276762 (W 0.000554196) | - | · | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.066033 ± 0.000168049 (W 8.27539e-05) | 0.056645 ± 3.17857e-05 (W 0.000110545) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.11961 ± 0.00100369 (W 0.000536014) | 0.146393 ± 0.00113556 (W 0.00109567) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.192064 ± 0.0011023 (W 0.0046511) | 0.147372 ± 0.000789877 (W 0.00202993) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.217405 ± 0.00161199 (W 0.0022388) | 0.233772 ± 2.55147e-05 (W 0.0018704) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.214429 ± 0.000523384 (W 0.00180279) | 0.235118 ± 0.00144464 (W 0.00452561) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.729936 ± 0.00148723 (W 0.00275999) | 1.149 ± 0.00228712 (W 0.00254848) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.732725 ± 0.00272238 (W 0.00191566) | 1.15253 ± 0.000395401 (W 0.00283117) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.48782 ± 0.000686628 (W 0.00596642) | 2.79213 ± 0.00845431 (W 0.00400564) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.49819 ± 0.0060911 (W 0.00288135) | 2.80242 ± 0.0133046 (W 0.00512455) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.07524 ± 0.000463177 (W 0.00012327) | 0.160596 ± 0.000461885 (W 0.00215802) | - | 🔵 faer-fork 2.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.075149 ± 6.67108e-05 (W 0.00024798) | 0.16081 ± 0.000549718 (W 0.00170025) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.32546 ± 0.000661327 (W 0.00216946) | 0.64996 ± 0.00103892 (W 0.00346841) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.328031 ± 0.00205166 (W 0.00150068) | 0.651963 ± 0.000947861 (W 0.00310525) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.742623 ± 0.0017807 (W 0.00240551) | 1.50674 ± 0.00234091 (W 0.00587924) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.748313 ± 0.00207545 (W 0.00212606) | 1.51115 ± 0.00277032 (W 0.00451434) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001423 ± 2.28108e-05 (W 2.56574e-05) | 0.001092 ± 1.52753e-05 (W 7.70065e-05) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003447 ± 2.5632e-05 (W 3.74059e-05) | 0.002715 ± 1.52753e-05 (W 3.47879e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006141 ± 2.3094e-05 (W 4.95106e-05) | 0.004979 ± 5.7735e-06 (W 3.67423e-05) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004319 ± 2.66521e-05 (W 2.34521e-05) | 0.004158 ± 4.20357e-05 (W 7.96869e-05) | 0.007053 ± 2.64575e-05 (W 7.86715e-05) | · | 🔵 faer-fork 1.63× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00523 ± 4.3589e-05 (W 5.6745e-05) | 0.004889 ± 2.64575e-05 (W 6.86964e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.027491 ± 9.66282e-05 (W 6.66273e-05) | 0.02631 ± 0.000212417 (W 9.01732e-05) | 0.041307 ± 0.000176353 (W 0.000171855) | · | 🔵 faer-fork 1.50× faster than `lapack-mkl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.031138 ± 0.000170098 (W 0.000102567) | 0.028864 ± 0.000251536 (W 0.00195095) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.077805 ± 0.000171561 (W 9.78734e-05) | 0.076683 ± 0.000190351 (W 0.00184726) | 0.119363 ± 0.000953998 (W 0.00227955) | · | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.08612 ± 0.000231085 (W 0.000157814) | 0.081722 ± 0.000110636 (W 0.000209285) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.141094 ± 0.000447424 (W 0.000404118) | 0.137476 ± 0.00176918 (W 0.00184674) | 0.141914 ± 0.000332802 (W 0.00202127) | · | · |  |
| `eigh/n32/b1 [api-inclusive]` | 0.141605 ± 0.000958351 (W 0.00168629) | 0.137285 ± 0.000833018 (W 0.00177475) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.737999 ± 0.00284305 (W 0.00248977) | 0.698044 ± 0.00347582 (W 0.00245203) | 0.775301 ± 0.00357378 (W 0.00223345) | 🔴 faer-fork 1.06× slower than `faer-upstream` | · |  |
| `eigh/n64/b1 [api-inclusive]` | 0.738723 ± 0.003276 (W 0.00247189) | 0.717985 ± 0.00327382 (W 0.00273407) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.79367 ± 0.0081125 (W 0.00336768) | 1.71303 ± 0.00364784 (W 0.00344921) | 1.97017 ± 0.00707604 (W 0.00278369) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.80247 ± 0.0078171 (W 0.00210734) | 1.71112 ± 0.00720187 (W 0.00248577) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.016952 ± 7.03586e-05 (W 9.12595e-05) | 0.013986 ± 8.71799e-05 (W 5.21536e-05) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018885 ± 0.000212768 (W 0.000161716) | 0.01595 ± 0.000189985 (W 7.86778e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.098343 ± 0.000406399 (W 0.000562774) | 0.084286 ± 0.000363723 (W 0.00156856) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.101489 ± 0.000100732 (W 0.000230977) | 0.090288 ± 0.000215019 (W 0.00168235) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.288218 ± 0.00550365 (W 0.00327868) | 0.266847 ± 0.00606789 (W 0.00179765) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.334922 ± 0.00234498 (W 0.00329978) | 0.312391 ± 0.000530983 (W 0.00232179) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028383 ± 1.52753e-05 (W 0.000171395) | 0.027942 ± 0.000180831 (W 9.71082e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028433 ± 0.000120968 (W 9.44749e-05) | 0.027972 ± 0.000106927 (W 0.00016585) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.159278 ± 0.000570017 (W 0.00245672) | 0.156621 ± 0.000591578 (W 0.00020828) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.158707 ± 0.0012629 (W 0.00250085) | 0.156752 ± 0.000475508 (W 0.000226573) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.464885 ± 0.00187395 (W 0.00180106) | 0.46203 ± 0.000579307 (W 0.00239938) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.467369 ± 0.00212467 (W 0.00101338) | 0.466027 ± 0.00118849 (W 0.00290391) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011081 ± 0.000159478 (W 7.97697e-05) | 0.010019 ± 5.56776e-05 (W 8.61974e-05) | 0.01601 ± 4.50925e-05 (W 0.000206556) | 🔴 faer-fork 1.11× slower than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013185 ± 5.0797e-05 (W 7.93108e-05) | 0.011782 ± 0.000145029 (W 8.10555e-05) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.049743 ± 0.000341361 (W 0.00020058) | 0.047759 ± 0.000332892 (W 0.000128504) | 0.079197 ± 0.000325833 (W 0.000423267) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.055303 ± 0.000237691 (W 0.00183358) | 0.053098 ± 0.000356156 (W 6.51022e-05) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.133067 ± 0.000930069 (W 0.00183162) | 0.131595 ± 0.000829044 (W 0.00186997) | 0.200703 ± 0.00110451 (W 0.00199087) | · | 🔵 faer-fork 1.51× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.19237 ± 0.000984564 (W 0.00197336) | 0.186968 ± 0.00269393 (W 0.00268907) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.063989 ± 0.000473044 (W 0.000311918) | 0.065282 ± 0.000505175 (W 0.000410692) | 0.052148 ± 0.000592449 (W 0.000122251) | · | 🔴 faer-fork 1.23× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.066875 ± 0.000271048 (W 0.000255871) | 0.068758 ± 0.000295691 (W 0.000313961) | - | · | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.298435 ± 0.00183829 (W 0.00243692) | 0.305168 ± 0.00243482 (W 0.00280296) | 0.422526 ± 0.00195978 (W 0.00311576) | · | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.326798 ± 0.000465543 (W 0.00338681) | 0.336536 ± 0.00233729 (W 0.00351217) | - | · | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.732379 ± 0.00242096 (W 0.0036341) | 0.757098 ± 0.00387301 (W 0.00268181) | 1.14777 ± 0.00467684 (W 0.00316764) | · | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.832217 ± 0.00344786 (W 0.00349472) | 0.851683 ± 0.00477381 (W 0.0031867) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.031849 ± 0.000178298 (W 0.000289702) | 0.034063 ± 0.000265688 (W 0.000296427) | 0.025367 ± 0.00032773 (W 0.000181954) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔴 faer-fork 1.26× slower than `lapack-mkl` |  |
| `qr_factor/n64/b1` | 0.151872 ± 0.000509 (W 0.0015882) | 0.163424 ± 0.000736261 (W 0.00233153) | 0.15026 ± 0.000366606 (W 0.00134495) | 🔵 faer-fork 1.08× faster than `faer-upstream` | · |  |
| `qr_factor/n96/b1` | 0.370622 ± 0.00205367 (W 0.00276897) | 0.399313 ± 0.00212023 (W 0.00353948) | 0.465876 ± 0.00256189 (W 0.00220458) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.012012 ± 0.000147988 (W 0.00011453) | 0.011011 ± 0.000142818 (W 4.41588e-05) | 0.01602 ± 0.000166233 (W 0.000120562) | 🔴 faer-fork 1.09× slower than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014517 ± 0.000211266 (W 0.00018097) | 0.013354 ± 5.00033e-05 (W 0.000113935) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.052648 ± 0.00036529 (W 0.000344938) | 0.050895 ± 0.000533153 (W 0.000126193) | 0.077224 ± 0.00030491 (W 0.000212679) | · | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.056825 ± 0.000376644 (W 0.00163174) | 0.054762 ± 0.000403598 (W 6.17592e-05) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.13958 ± 0.000929284 (W 0.00184299) | 0.133718 ± 0.000782804 (W 0.00199315) | 0.194673 ± 0.000756504 (W 0.00188687) | · | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.195924 ± 0.000790853 (W 0.0046382) | 0.190334 ± 0.00261425 (W 0.00281823) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.221472 ± 0.0029551 (W 0.00230864) | 0.232594 ± 0.00103833 (W 0.00179719) | 0.244556 ± 0.000948125 (W 0.00233549) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.218517 ± 0.00166724 (W 0.00238626) | 0.231793 ± 0.00100205 (W 0.00244217) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.00588 ± 0.0108699 (W 0.000890436) | 1.24421 ± 0.00546501 (W 0.00285496) | 1.28409 ± 0.00416842 (W 0.00307912) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 1.00369 ± 0.00492034 (W 0.00155536) | 1.24798 ± 0.00677788 (W 0.00171532) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.46935 ± 0.00738911 (W 0.00600412) | 3.06818 ± 0.0114428 (W 0.00240204) | 3.3223 ± 0.0152459 (W 0.00310948) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.46615 ± 0.00984504 (W 0.00494526) | 3.07998 ± 0.0126999 (W 0.00447102) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075501 ± 0.000248355 (W 0.000119316) | 0.158585 ± 0.000596516 (W 0.00174432) | 0.096831 ± 0.00141823 (W 0.00182455) | 🔵 faer-fork 2.10× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.074769 ± 0.000410853 (W 0.000151701) | 0.158296 ± 0.000188152 (W 0.00181205) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.309296 ± 0.00193728 (W 0.00151543) | 0.623649 ± 0.00273414 (W 0.00223287) | 0.464758 ± 0.0021812 (W 0.00202718) | 🔵 faer-fork 2.02× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.309555 ± 0.00109859 (W 0.00290684) | 0.622828 ± 0.00288564 (W 0.00238586) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.809324 ± 0.00320853 (W 0.00239005) | 1.47385 ± 0.00955814 (W 0.00314139) | 1.29984 ± 0.00462119 (W 0.00196115) | 🔵 faer-fork 1.82× faster than `faer-upstream` | 🔵 faer-fork 1.61× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.810787 ± 0.000739258 (W 0.00227875) | 1.47263 ± 0.00781724 (W 0.00183428) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001313 ± 2.3388e-05 (W 4.35236e-05) | 0.001362 ± 6.08276e-06 (W 6.34445e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002946 ± 3.5171e-05 (W 7.38397e-05) | 0.002915 ± 0 (W 2.80927e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005369 ± 2.09841e-05 (W 5.77339e-05) | 0.00536 ± 1.73205e-05 (W 3.11448e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004208 ± 0.000147886 (W 5.91608e-05) | 0.004108 ± 4.87545e-05 (W 3.10693e-05) | 0.007204 ± 7.59408e-05 (W 9.07204e-05) | · | 🔵 faer-fork 1.71× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00528 ± 4.16333e-05 (W 5.81833e-05) | 0.004969 ± 6.08276e-05 (W 4.65833e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.025949 ± 8.36082e-05 (W 0.000161568) | 0.026509 ± 0.000162759 (W 0.000135876) | 0.041378 ± 0.000269891 (W 0.00149148) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.029906 ± 0.000173494 (W 0.000101957) | 0.028954 ± 0.000189597 (W 7.86778e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.073297 ± 0.00012995 (W 0.000249137) | 0.079478 ± 0.000382572 (W 0.0015393) | 0.120745 ± 0.000328836 (W 0.00160955) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.081832 ± 0.000312501 (W 0.00061411) | 0.085599 ± 0.000518731 (W 0.000735504) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.101916 ± 0.000273964 (W 0.000289338) | 0.137973 ± 0.000514032 (W 0.00205925) | 0.146993 ± 0.000864336 (W 0.00171241) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.102962 ± 0.000453001 (W 0.000162713) | 0.138077 ± 0.00086571 (W 0.00211684) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.432171 ± 0.00238252 (W 0.00243351) | 0.667172 ± 0.00291664 (W 0.002115) | 0.526444 ± 0.00190741 (W 0.0023974) | 🔵 faer-fork 1.54× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.434275 ± 0.00212066 (W 0.00202166) | 0.670418 ± 0.00142592 (W 0.00235628) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.920194 ± 0.00220301 (W 0.00116033) | 1.53894 ± 0.00631806 (W 0.00389513) | 1.09648 ± 0.00694352 (W 0.00239882) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.923148 ± 0.00518958 (W 0.00274217) | 1.54454 ± 0.00228764 (W 0.00424727) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018334 ± 6.35085e-05 (W 6.65823e-05) | 0.013966 ± 3.51188e-05 (W 5.05885e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019767 ± 9.0185e-05 (W 0.000156921) | 0.01574 ± 8.60252e-05 (W 6.07314e-05) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.09699 ± 0.00254148 (W 0.000445212) | 0.085129 ± 0.00017407 (W 0.000424339) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.100929 ± 0.00142162 (W 0.000769017) | 0.089937 ± 0.000273606 (W 0.000189723) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.230539 ± 0.00135133 (W 0.00147013) | 0.266448 ± 0.000616264 (W 0.00204235) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.285571 ± 0.000315305 (W 0.00317112) | 0.263543 ± 0.000919041 (W 0.00210587) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.015018 ± 0.000115803 (W 0.000213509) | 0.028193 ± 8.96735e-05 (W 0.000108946) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.014847 ± 0.000208663 (W 0.000228082) | 0.028262 ± 0.000144535 (W 0.000133713) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.061182 ± 0.00023553 (W 0.00035967) | 0.060313 ± 0.0003021 (W 0.000367227) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.061495 ± 0.000176001 (W 0.000282766) | 0.059989 ± 0.000130012 (W 0.00187711) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.151302 ± 0.000771728 (W 0.00176065) | 0.147941 ± 0.000152355 (W 0.000788589) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.151187 ± 0.00108775 (W 0.00161856) | 0.149057 ± 0.00141104 (W 0.00137173) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011421 ± 3.45977e-05 (W 0.000116285) | 0.010079 ± 0.000100683 (W 0.000122077) | 0.01597 ± 0.000188237 (W 0.000120125) | 🔴 faer-fork 1.13× slower than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013415 ± 4e-05 (W 0.000134251) | 0.011922 ± 0.000157513 (W 7.10296e-05) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.048481 ± 0.000730633 (W 0.000406145) | 0.047809 ± 0.000316128 (W 9.60432e-05) | 0.079448 ± 0.000643632 (W 0.000275392) | · | 🔵 faer-fork 1.64× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.055614 ± 0.000441604 (W 0.00108195) | 0.05329 ± 0.000535824 (W 0.000233864) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.11276 ± 0.00974026 (W 0.00182884) | 0.128529 ± 0.00136286 (W 0.0015175) | 0.202437 ± 0.000145317 (W 0.00235773) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.80× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.173293 ± 0.000690628 (W 0.00186276) | 0.141042 ± 0.000113006 (W 0.00276762) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.058168 ± 0.000380881 (W 0.000485349) | 0.079137 ± 0.000559581 (W 0.0015524) | 0.051847 ± 0.000226628 (W 0.000158803) | 🔵 faer-fork 1.36× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.062066 ± 0.000265575 (W 0.000465834) | 0.082985 ± 0.000770845 (W 0.000521102) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.179804 ± 0.00160907 (W 0.00201409) | 0.309907 ± 0.00211854 (W 0.00243461) | 0.260086 ± 0.00118609 (W 0.00249048) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.217865 ± 0.00301162 (W 0.00268782) | 0.34279 ± 0.00104705 (W 0.00291074) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.382372 ± 0.00134785 (W 0.00190176) | 0.590393 ± 0.00480388 (W 0.00299228) | 0.706826 ± 0.00481314 (W 0.00277428) | 🔵 faer-fork 1.54× faster than `faer-upstream` | 🔵 faer-fork 1.85× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.498588 ± 0.00129811 (W 0.00494275) | 0.699066 ± 0.00414412 (W 0.00292442) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.030847 ± 0.000110666 (W 0.000264654) | 0.04798 ± 0.00779149 (W 0.000596814) | 0.025347 ± 0.000215948 (W 0.000139922) | 🔵 faer-fork 1.56× faster than `faer-upstream` | 🔴 faer-fork 1.22× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n64/b1` | 0.116718 ± 0.00112266 (W 0.000347053) | 0.211245 ± 0.00181139 (W 0.00245064) | 0.152195 ± 0.00185804 (W 0.00188161) | 🔵 faer-fork 1.81× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.246891 ± 0.00215843 (W 0.00148466) | 0.427058 ± 0.00305229 (W 0.0021421) | 0.466609 ± 0.00391817 (W 0.00289536) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.89× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.012714 ± 0.000213854 (W 0.000129635) | 0.01091 ± 6.98594e-05 (W 4.86826e-05) | 0.01611 ± 5.68624e-05 (W 9.0111e-05) | 🔴 faer-fork 1.17× slower than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014878 ± 8.73689e-05 (W 6.68072e-05) | 0.013085 ± 7.6922e-05 (W 9.9986e-05) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.051897 ± 0.00124361 (W 0.00012227) | 0.050554 ± 0.000179556 (W 5.86984e-05) | 0.077484 ± 0.000486995 (W 0.000209693) | · | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.060041 ± 0.00106611 (W 0.00147341) | 0.054622 ± 0.000480182 (W 0.00018743) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.114153 ± 0.00134249 (W 0.00041936) | 0.131215 ± 0.00196258 (W 0.00171327) | 0.195915 ± 0.000307147 (W 0.00207334) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.178051 ± 0.0015033 (W 0.00250011) | 0.143468 ± 0.00135089 (W 0.00271301) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.215511 ± 0.00209492 (W 0.00171315) | 0.231271 ± 0.00243499 (W 0.00168518) | 0.25269 ± 0.00115895 (W 0.00175578) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.213267 ± 0.0020026 (W 0.00197329) | 0.232362 ± 0.00132553 (W 0.00143356) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.73171 ± 0.00479487 (W 0.00238782) | 1.17486 ± 0.00462981 (W 0.00192178) | 0.975957 ± 0.00334501 (W 0.00157254) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.737553 ± 0.00342967 (W 0.00274086) | 1.17815 ± 0.00197525 (W 0.00224613) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.55385 ± 0.0122904 (W 0.00534664) | 2.73653 ± 0.00910693 (W 0.00630256) | 2.41649 ± 0.00454123 (W 0.00412228) | 🔵 faer-fork 1.76× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.56166 ± 0.00654045 (W 0.00278581) | 2.74158 ± 0.00636784 (W 0.0050234) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075631 ± 0.00041353 (W 7.34541e-05) | 0.158585 ± 0.000389123 (W 0.00233406) | 0.097632 ± 0.000955925 (W 0.000236534) | 🔵 faer-fork 2.10× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.076261 ± 0.00064125 (W 0.000114701) | 0.158205 ± 0.000366497 (W 0.00186562) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.317451 ± 0.00262869 (W 0.00215818) | 0.627879 ± 0.00225881 (W 0.0027503) | 0.465828 ± 0.00226807 (W 0.0028712) | 🔵 faer-fork 1.98× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.317591 ± 0.0019385 (W 0.0015128) | 0.626467 ± 0.00198999 (W 0.00268) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.72107 ± 0.00501432 (W 0.00421136) | 1.47696 ± 0.00364282 (W 0.00528071) | 1.31119 ± 0.00500413 (W 0.00271198) | 🔵 faer-fork 2.05× faster than `faer-upstream` | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.725991 ± 0.00523336 (W 0.00227371) | 1.47921 ± 0.00554012 (W 0.00390955) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001452 ± 1.12694e-05 (W 2.29848e-05) | 0.001263 ± 2.04206e-05 (W 7.99706e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003437 ± 2.66521e-05 (W 4.56651e-05) | 0.002905 ± 2.46374e-05 (W 3.7659e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006102 ± 6.35085e-06 (W 3.62933e-05) | 0.00541 ± 4.93288e-05 (W 4.51796e-05) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004298 ± 1.1547e-05 (W 4.65833e-05) | 0.004188 ± 3.27159e-05 (W 9.26051e-05) | 0.007123 ± 5.1316e-05 (W 0.000137186) | · | 🔵 faer-fork 1.66× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005199 ± 7.02163e-05 (W 4.97916e-05) | 0.005 ± 2.09841e-05 (W 5.50926e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.026169 ± 0.000103055 (W 8.24209e-05) | 0.026379 ± 9.17624e-05 (W 0.000103123) | 0.041007 ± 0.000856735 (W 0.000223063) | · | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.029675 ± 4.50925e-05 (W 0.000163616) | 0.029034 ± 8.3865e-05 (W 0.000100445) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.072075 ± 0.00045117 (W 0.000139245) | 0.079368 ± 0.000210231 (W 0.000275899) | 0.119242 ± 0.000896748 (W 0.000297397) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.081983 ± 0.000414272 (W 0.000188359) | 0.0863 ± 0.000196723 (W 0.0004883) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.097531 ± 0.000209831 (W 0.000374266) | 0.138115 ± 0.000952343 (W 0.00201661) | 0.148407 ± 0.0363791 (W 0.00198827) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.098724 ± 0.000630622 (W 0.000376882) | 0.138558 ± 0.00181129 (W 0.00194451) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.319856 ± 0.00654504 (W 0.00669774) | 0.669498 ± 0.00191032 (W 0.00338528) | 0.50515 ± 0.00248638 (W 0.00229369) | 🔵 faer-fork 2.09× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.319179 ± 0.000769842 (W 0.00304011) | 0.668104 ± 0.00201024 (W 0.00204094) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.684666 ± 0.00441798 (W 0.0150497) | 1.54221 ± 0.00326062 (W 0.004624) | 0.989425 ± 0.00322842 (W 0.00568244) | 🔵 faer-fork 2.25× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.686439 ± 0.00364253 (W 0.00329613) | 1.54828 ± 0.00298601 (W 0.0049136) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018314 ± 0.000164418 (W 9.25376e-05) | 0.014096 ± 2e-05 (W 9.05991e-05) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019686 ± 7.25971e-05 (W 7.1121e-05) | 0.015889 ± 0.000185716 (W 9.75203e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.101349 ± 0.000169142 (W 0.000717233) | 0.084908 ± 0.000672417 (W 0.00172826) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.104804 ± 0.000560521 (W 0.00148642) | 0.090156 ± 0.000466886 (W 0.0022081) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.232549 ± 0.00222482 (W 0.0016309) | 0.266882 ± 0.000444451 (W 0.0016144) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.292329 ± 0.000872767 (W 0.00324215) | 0.265475 ± 0.00190445 (W 0.00306496) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.01096 ± 0.000125033 (W 0.000174084) | 0.028022 ± 0.000130512 (W 8.33559e-05) | - | 🔵 faer-fork 2.56× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.010961 ± 4.59384e-05 (W 0.000121882) | 0.028002 ± 9.86577e-05 (W 0.000119822) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.038081 ± 0.00589987 (W 0.000323539) | 0.060782 ± 0.000328883 (W 0.000504526) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.037679 ± 7.59408e-05 (W 0.000783411) | 0.060763 ± 0.00012256 (W 0.000515003) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.08676 ± 0.000341087 (W 0.000587333) | 0.148707 ± 0.00013402 (W 0.00140582) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.086621 ± 0.000172917 (W 0.000379327) | 0.148625 ± 0.000422827 (W 0.000626469) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011301 ± 2.64575e-05 (W 7.82943e-05) | 0.010079 ± 4.3589e-05 (W 9.64946e-05) | 0.01619 ± 0.000105357 (W 0.000173019) | 🔴 faer-fork 1.12× slower than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013465 ± 6.1101e-05 (W 0.000122882) | 0.011982 ± 0.000110151 (W 0.000130576) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.04846 ± 0.000308254 (W 0.000239293) | 0.047868 ± 0.000300711 (W 0.000120188) | 0.079847 ± 4.05093e-05 (W 0.000163003) | · | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.055502 ± 0.000254127 (W 0.000317734) | 0.053049 ± 6.37992e-05 (W 0.000136088) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.10756 ± 0.000276374 (W 0.000622204) | 0.13048 ± 0.000174855 (W 0.00162715) | 0.202402 ± 0.000156643 (W 0.0020782) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.88× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.173624 ± 0.001578 (W 0.00215998) | 0.141664 ± 0.000449269 (W 0.00210252) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.057917 ± 0.000350914 (W 0.00140656) | 0.088343 ± 0.00179366 (W 0.00106948) | 0.051837 ± 0.000320401 (W 0.00157831) | 🔵 faer-fork 1.53× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.060882 ± 0.000129114 (W 0.000362675) | 0.090438 ± 0.000808814 (W 0.00124256) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.1662 ± 0.0005392 (W 0.00168013) | 0.352787 ± 0.00349045 (W 0.00215367) | 0.232944 ± 0.000703773 (W 0.00197377) | 🔵 faer-fork 2.12× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.206715 ± 0.000498556 (W 0.00224419) | 0.385451 ± 0.00137514 (W 0.00377847) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.328912 ± 0.00233671 (W 0.0030258) | 0.673953 ± 0.00547953 (W 0.00293621) | 0.617388 ± 0.0021988 (W 0.00232432) | 🔵 faer-fork 2.05× faster than `faer-upstream` | 🔵 faer-fork 1.88× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.459045 ± 0.00293089 (W 0.00283373) | 0.785212 ± 0.0128985 (W 0.0027349) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.030716 ± 0.000253628 (W 0.000255221) | 0.056173 ± 0.000494951 (W 0.00110241) | 0.025527 ± 0.000266135 (W 0.000155352) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔴 faer-fork 1.20× slower than `lapack-mkl` |  |
| `qr_factor/n64/b1` | 0.112479 ± 0.000196526 (W 0.00199819) | 0.25215 ± 0.0038135 (W 0.0054362) | 0.150661 ± 0.00103231 (W 0.00201001) | 🔵 faer-fork 2.24× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.229534 ± 0.00127 (W 0.00187702) | 0.492327 ± 0.00283705 (W 0.00270977) | 0.469114 ± 0.00251273 (W 0.00295578) | 🔵 faer-fork 2.14× faster than `faer-upstream` | 🔵 faer-fork 2.04× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.012573 ± 7.5368e-05 (W 7.59684e-05) | 0.01098 ± 8e-05 (W 5.87222e-05) | 0.01602 ± 9.89966e-05 (W 0.000127657) | 🔴 faer-fork 1.15× slower than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014858 ± 0.000112398 (W 8.73167e-05) | 0.013045 ± 0.000146857 (W 5.503e-05) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.052558 ± 0.000151318 (W 0.000281191) | 0.050745 ± 0.000405449 (W 0.00186409) | 0.077624 ± 0.00037055 (W 0.000254488) | · | 🔵 faer-fork 1.48× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.05915 ± 0.00074152 (W 0.000306391) | 0.054973 ± 0.000236646 (W 0.000219322) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.110787 ± 0.000399069 (W 0.000624209) | 0.133148 ± 0.000469501 (W 0.000411696) | 0.194348 ± 0.000485897 (W 0.00259586) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.75× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.175326 ± 0.00231526 (W 0.00269879) | 0.144719 ± 0.000674115 (W 0.00265207) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.212402 ± 0.000340286 (W 0.00306179) | 0.231301 ± 0.000402609 (W 0.00204432) | 0.254675 ± 0.0010991 (W 0.00351449) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.211183 ± 0.00155961 (W 0.00154124) | 0.231466 ± 0.00117691 (W 0.00181596) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.717537 ± 0.000496409 (W 0.00237275) | 1.17856 ± 0.00462754 (W 0.00233462) | 0.941824 ± 0.00503925 (W 0.00240462) | 🔵 faer-fork 1.64× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.722486 ± 0.00302156 (W 0.00257133) | 1.17548 ± 0.00225193 (W 0.00457474) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.48799 ± 0.00368485 (W 0.00230773) | 2.75383 ± 0.00814671 (W 0.00646079) | 2.24105 ± 0.00694266 (W 0.00484787) | 🔵 faer-fork 1.85× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.50031 ± 0.0033086 (W 0.00545585) | 2.74473 ± 0.0138613 (W 0.00554598) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.076342 ± 0.000109555 (W 0.000114098) | 0.158495 ± 0.000618428 (W 0.00230613) | 0.097351 ± 0.000939928 (W 0.000193184) | 🔵 faer-fork 2.08× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.076512 ± 6.39296e-05 (W 0.000306454) | 0.158856 ± 0.000484933 (W 0.00182703) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.323822 ± 0.00112071 (W 0.00185109) | 0.625045 ± 0.000836986 (W 0.00311506) | 0.464134 ± 0.00228912 (W 0.00307333) | 🔵 faer-fork 1.93× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.325511 ± 0.001213 (W 0.00183178) | 0.627209 ± 0.000432792 (W 0.00251413) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.742412 ± 0.00355376 (W 0.00140745) | 1.47575 ± 0.00524554 (W 0.00456049) | 1.30361 ± 0.00199852 (W 0.00231879) | 🔵 faer-fork 1.99× faster than `faer-upstream` | 🔵 faer-fork 1.76× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.742062 ± 0.00134435 (W 0.00321151) | 1.48048 ± 0.00260701 (W 0.00440887) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001463 ± 1.58219e-05 (W 3.98158e-05) | 0.001262 ± 9.50438e-06 (W 2.59673e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003416 ± 1.58219e-05 (W 6.70395e-05) | 0.002925 ± 5.7735e-07 (W 2.2832e-05) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006061 ± 3.17857e-05 (W 5.64296e-05) | 0.00535 ± 1.73205e-05 (W 5.26308e-05) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003246 ± 5.03322e-05 (W 3.56399e-05) | 0.003988 ± 0.000109697 (W 3.52463e-05) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002565 ± 2.02567e-05 (W 0.000107765) | 0.002605 ± 3.17857e-05 (W 7.79102e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013696 ± 0.00107041 (W 4.14873e-05) | 0.034114 ± 0.00133481 (W 0.000121257) | - | 🔵 faer-fork 2.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010048 ± 6.55515e-05 (W 0.000101743) | 0.010148 ± 7.19745e-05 (W 5.40019e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.026229 ± 1.1547e-05 (W 9.63701e-05) | 0.159208 ± 0.000268727 (W 0.00208188) | - | 🔵 faer-fork 6.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.021249 ± 6.4291e-05 (W 8.6454e-05) | 0.020939 ± 0.000141234 (W 0.000129268) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.08637 ± 0.000611434 (W 0.00168748) | 0.087413 ± 0.00158071 (W 0.000245724) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.087443 ± 0.000260577 (W 0.000311766) | 0.087042 ± 0.000383072 (W 0.00192062) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.32269 ± 0.00138641 (W 0.00224808) | 0.319865 ± 0.00115951 (W 0.00216389) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.324554 ± 0.000834897 (W 0.00246858) | 0.319597 ± 0.00178197 (W 0.00260937) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.950489 ± 0.0035191 (W 0.00263152) | 0.904634 ± 0.00239968 (W 0.00253848) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.958371 ± 0.00794953 (W 0.00397353) | 0.907918 ± 0.00200196 (W 0.00331928) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.010099 ± 5.03322e-05 (W 2.70185e-05) | 0.008386 ± 0.000219933 (W 3.28222e-05) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013184 ± 5.9808e-05 (W 0.000140412) | 0.010379 ± 5.50757e-05 (W 0.000113886) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.041337 ± 0.000539556 (W 8.27979e-05) | 0.034815 ± 0.00050687 (W 0.000142117) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.049662 ± 0.000424539 (W 0.000125137) | 0.039413 ± 0.000365048 (W 0.000206944) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.10178 ± 0.00173088 (W 0.0010357) | 0.084348 ± 2.51064e-05 (W 0.000504158) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.118101 ± 0.000689104 (W 0.000638439) | 0.093143 ± 0.000276046 (W 0.00128163) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006351 ± 3.05014e-05 (W 9.46161e-05) | 0.006251 ± 3.4356e-05 (W 8.19097e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006332 ± 3.78594e-05 (W 3.94614e-05) | 0.006162 ± 4.04145e-05 (W 5.9011e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.026159 ± 0.000212039 (W 9.06934e-05) | 0.025648 ± 0.000121244 (W 0.000173003) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.026198 ± 0.000260137 (W 0.000114978) | 0.025557 ± 0.00012779 (W 7.79949e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.077434 ± 0.000237426 (W 0.00212816) | 0.07526 ± 0.000463455 (W 0.000254806) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.076943 ± 0.00043131 (W 0.000200101) | 0.07528 ± 0.000135769 (W 0.00189225) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006171 ± 0.00011415 (W 4.04512e-05) | 0.007884 ± 5.54106e-05 (W 4.0978e-05) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008105 ± 0.000135769 (W 0.000159559) | 0.007064 ± 5.05008e-05 (W 7.24521e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.019647 ± 0.000217945 (W 5.79077e-05) | 0.033622 ± 0.000782044 (W 0.00014498) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.02662 ± 0.0002474 (W 0.000337324) | 0.022933 ± 0.000225411 (W 0.000168102) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.043471 ± 0.000241144 (W 0.000323875) | 0.079348 ± 0.00113804 (W 0.000551767) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.058348 ± 0.000422004 (W 0.000364891) | 0.050163 ± 0.000320776 (W 0.000250202) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.030146 ± 0.000271577 (W 4.94975e-05) | 0.035025 ± 0.000315875 (W 9.08047e-05) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.028222 ± 0.000105311 (W 0.00013846) | 0.029675 ± 0.000202897 (W 9.31681e-05) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.092702 ± 0.000261504 (W 0.00137791) | 0.158295 ± 0.000491795 (W 0.0016037) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.10204 ± 5.13258e-05 (W 0.0017508) | 0.110004 ± 0.000476992 (W 0.000189605) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.195754 ± 0.00133472 (W 0.00179296) | 0.344381 ± 0.00112474 (W 0.00213667) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.238984 ± 0.000551817 (W 0.0019281) | 0.257589 ± 0.00198972 (W 0.00233793) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.020739 ± 3e-05 (W 0.000107504) | 0.025537 ± 0.00016552 (W 0.000124619) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.061244 ± 0.000402522 (W 0.00200317) | 0.123139 ± 0.000256438 (W 0.00146568) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.116055 ± 0.000497957 (W 0.000387144) | 0.259714 ± 0.000667868 (W 0.00225224) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007053 ± 8.62168e-05 (W 5.93296e-05) | 0.008526 ± 0.000125033 (W 4.1833e-05) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009678 ± 6.65833e-05 (W 7.48345e-05) | 0.007815 ± 6.245e-05 (W 8.94444e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019586 ± 6.15386e-05 (W 6.78181e-05) | 0.034264 ± 0.000412934 (W 0.000144395) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.028363 ± 4.93288e-05 (W 0.000141419) | 0.023644 ± 0.000186039 (W 6.41872e-05) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.039884 ± 0.000602584 (W 0.00211944) | 0.079078 ± 0.000677463 (W 0.000142931) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.057307 ± 0.000607943 (W 0.000238024) | 0.049653 ± 0.000135277 (W 0.002019) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.125854 ± 0.000910759 (W 0.001747) | 0.136544 ± 0.000321934 (W 0.000249991) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.128199 ± 0.000812056 (W 0.00154443) | 0.137697 ± 0.000419941 (W 0.000174506) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.531343 ± 0.000570955 (W 0.0025404) | 0.53535 ± 0.00465467 (W 0.00213248) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.535546 ± 0.00343486 (W 0.00348028) | 0.533152 ± 0.00207619 (W 0.00257153) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.21564 ± 0.00630164 (W 0.00374087) | 1.56163 ± 0.00415485 (W 0.0045359) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.21607 ± 0.0016855 (W 0.00364361) | 1.59072 ± 0.00495428 (W 0.00624713) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.04794 ± 0.000229144 (W 0.00012129) | 0.110786 ± 0.000497605 (W 0.000149289) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.04808 ± 0.000157513 (W 0.000188988) | 0.110946 ± 0.00115627 (W 0.00153382) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.168485 ± 0.000755805 (W 0.00199979) | 0.398141 ± 0.00135372 (W 0.00236904) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.169385 ± 0.00115405 (W 0.0016318) | 0.397399 ± 0.00122724 (W 0.00221053) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.368368 ± 0.00204423 (W 0.00219802) | 0.881487 ± 0.00301452 (W 0.000877526) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.369127 ± 0.00215211 (W 0.00272075) | 0.88168 ± 0.00370508 (W 0.00116167) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000731 ± 1e-05 (W 3.6169e-05) | 0.000711 ± 1.67432e-05 (W 8.437e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001573 ± 3.05505e-05 (W 8.20366e-05) | 0.001473 ± 0 (W 7.19722e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.002705 ± 4.6188e-05 (W 7.10634e-05) | 0.002605 ± 4.63717e-05 (W 2.97876e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003005 ± 0.000214795 (W 3.9627e-05) | 0.004018 ± 0.000135769 (W 2.19089e-05) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002675 ± 4.3589e-05 (W 3.28634e-05) | 0.002614 ± 2.0664e-05 (W 9.64997e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013735 ± 0.000945755 (W 3.68687e-05) | 0.03212 ± 0.00125458 (W 8.96103e-05) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010139 ± 4.04145e-05 (W 9.004e-05) | 0.010189 ± 6.96443e-05 (W 8.61394e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.026619 ± 0.000240188 (W 7.25066e-05) | 0.051717 ± 0.00131828 (W 0.000285139) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.02151 ± 4.50925e-05 (W 6.7507e-05) | 0.026389 ± 0.00149947 (W 0.000426492) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.06405 ± 0.000200602 (W 0.00022838) | 0.086722 ± 0.000157172 (W 0.000115101) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.06509 ± 0.000674071 (W 0.000139654) | 0.086692 ± 0.00100538 (W 0.00212514) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.25267 ± 0.000865218 (W 0.00200157) | 0.326488 ± 0.00494744 (W 0.00226746) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.254544 ± 0.00157995 (W 0.00160589) | 0.326509 ± 0.000890996 (W 0.00268511) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.545737 ± 0.00139386 (W 0.00256115) | 0.901699 ± 0.00174125 (W 0.00277222) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.557588 ± 0.000467116 (W 0.00448989) | 0.898954 ± 0.000700388 (W 0.00277877) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011511 ± 0.000184906 (W 6.54385e-05) | 0.008336 ± 4.7085e-05 (W 5.18382e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014397 ± 7.16961e-05 (W 7.91151e-05) | 0.010299 ± 8.05005e-05 (W 6.76018e-05) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.044062 ± 0.000623107 (W 8.74826e-05) | 0.034514 ± 0.000196741 (W 7.14143e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.052057 ± 0.000551615 (W 0.000292773) | 0.038802 ± 0.000372668 (W 0.000230789) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.10724 ± 0.000929977 (W 0.000803838) | 0.084367 ± 0.000778584 (W 0.00111833) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.123095 ± 0.000600635 (W 0.000318758) | 0.09298 ± 0.000536045 (W 0.000364915) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00556 ± 9.31504e-05 (W 0.000228007) | 0.006422 ± 5.7735e-06 (W 9.41568e-05) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005681 ± 3.61985e-05 (W 8.5675e-05) | 0.006401 ± 2.62742e-05 (W 8.45707e-05) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.015719 ± 0.0001757 (W 0.000411459) | 0.01612 ± 0.000541617 (W 0.0008779) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014096 ± 0.00036722 (W 0.000268183) | 0.013716 ± 5.27289e-05 (W 0.000391209) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.033472 ± 0.000411656 (W 0.00193886) | 0.031419 ± 0.000450023 (W 0.000762678) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.034655 ± 0.000512432 (W 0.000839522) | 0.031988 ± 0.000201356 (W 0.00042792) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006232 ± 3.21455e-05 (W 4.8737e-05) | 0.007564 ± 0.000152753 (W 7.63819e-05) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008315 ± 7e-05 (W 6.07067e-05) | 0.007083 ± 0.000137961 (W 8.11172e-05) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020058 ± 0.000106397 (W 9.58817e-05) | 0.034625 ± 0.000181387 (W 0.000490103) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.026319 ± 4.63717e-05 (W 0.000225637) | 0.025988 ± 1.47309e-05 (W 0.000957064) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.052598 ± 0.000669572 (W 0.00127034) | 0.087673 ± 0.000597169 (W 0.00138995) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.055151 ± 0.000272658 (W 0.000784456) | 0.060623 ± 0.000284166 (W 0.0015442) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.031419 ± 4.96689e-05 (W 9.1106e-05) | 0.04831 ± 0.000208522 (W 0.000636364) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.029996 ± 0.000376494 (W 0.000143956) | 0.043181 ± 1.05357e-05 (W 0.00041693) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.100073 ± 0.00308167 (W 0.00143798) | 0.238487 ± 0.00157717 (W 0.00297267) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.100988 ± 0.000128974 (W 0.000531355) | 0.18263 ± 0.000639841 (W 0.00249603) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.201595 ± 0.00142957 (W 0.00231455) | 0.487084 ± 0.00210645 (W 0.00217084) | - | 🔵 faer-fork 2.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.196306 ± 0.000668837 (W 0.00200443) | 0.362545 ± 0.000396477 (W 0.00239851) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.02169 ± 0.000161658 (W 7.78929e-05) | 0.039172 ± 0.000320501 (W 0.000468144) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.069297 ± 0.000525856 (W 0.000173015) | 0.201399 ± 0.000436993 (W 0.00225183) | - | 🔵 faer-fork 2.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.132773 ± 0.0010948 (W 0.000219828) | 0.409392 ± 0.00152506 (W 0.00362594) | - | 🔵 faer-fork 3.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007594 ± 0.000137961 (W 4.43531e-05) | 0.008145 ± 1.73205e-05 (W 6.42199e-05) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010259 ± 3.51188e-05 (W 6.70835e-05) | 0.007825 ± 2.25167e-05 (W 7.81236e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020397 ± 0.000176438 (W 7.91833e-05) | 0.035505 ± 0.000196454 (W 0.000392557) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.029135 ± 0.000321063 (W 0.000154723) | 0.02669 ± 0.000157259 (W 0.000630557) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.045173 ± 0.000715855 (W 0.0006154) | 0.086701 ± 0.00151919 (W 0.00163232) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.058519 ± 0.000591799 (W 0.000669413) | 0.059158 ± 0.000427755 (W 0.000394844) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.12863 ± 0.00182327 (W 0.000134169) | 0.137727 ± 0.000636073 (W 0.00239959) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.131054 ± 0.0112196 (W 0.00113019) | 0.137837 ± 0.00122885 (W 0.00191568) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.443255 ± 0.00146728 (W 0.00192167) | 0.539735 ± 0.00253149 (W 0.00249426) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.458654 ± 0.00249466 (W 0.00435093) | 0.544413 ± 0.0080759 (W 0.00286546) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.891821 ± 0.00503945 (W 0.00317486) | 1.5321 ± 0.00962474 (W 0.00267017) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.90443 ± 0.00373671 (W 0.00372821) | 1.53116 ± 0.00244569 (W 0.00464076) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.048891 ± 0.000283385 (W 0.000145449) | 0.110816 ± 0.000225214 (W 0.0020826) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.049702 ± 0.000319174 (W 0.000128589) | 0.110897 ± 0.000423118 (W 0.000334986) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.170703 ± 0.00128885 (W 0.00222913) | 0.400075 ± 0.000347451 (W 0.00288877) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.171565 ± 0.00112775 (W 0.000837461) | 0.399633 ± 0.00159958 (W 0.00271228) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.380749 ± 0.000837112 (W 0.00268911) | 0.884727 ± 0.000355068 (W 0.00244528) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.381721 ± 0.000538016 (W 0.00230405) | 0.883535 ± 0.00453036 (W 0.00210004) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000962 ± 1.47309e-05 (W 4.52692e-05) | 0.000701 ± 1.1547e-05 (W 2.47225e-05) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002023 ± 2.09841e-05 (W 3.98898e-05) | 0.001503 ± 2e-05 (W 6.16441e-05) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003366 ± 1e-05 (W 3.93853e-05) | 0.002575 ± 4.3589e-05 (W 5.98331e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003266 ± 0.000159478 (W 2.82931e-05) | 0.004027 ± 0.000122231 (W 2.2676e-05) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002675 ± 1.0504e-05 (W 3.64692e-05) | 0.002705 ± 2.64575e-05 (W 9.17126e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013685 ± 0.000307215 (W 2.7408e-05) | 0.03218 ± 0.00125083 (W 2.73861e-05) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010079 ± 0.000107858 (W 5.89067e-05) | 0.010249 ± 5.19615e-06 (W 6.76018e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.02698 ± 0.000353319 (W 9.2543e-05) | 0.050644 ± 0.00128576 (W 0.00126831) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.021691 ± 5.22526e-05 (W 0.000111221) | 0.027431 ± 0.00125884 (W 0.000293804) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.060883 ± 0.00018124 (W 0.000109413) | 0.086491 ± 0.000329292 (W 0.00260213) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.061585 ± 0.000330125 (W 0.000194171) | 0.088665 ± 0.00100892 (W 0.000249049) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.19813 ± 0.000792682 (W 0.00177359) | 0.324705 ± 0.00133224 (W 0.00227122) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.200093 ± 0.00151986 (W 0.00190604) | 0.325581 ± 0.00187808 (W 0.00141514) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.410153 ± 0.00432828 (W 0.00258232) | 0.915391 ± 0.00427122 (W 0.00355708) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.412903 ± 0.00207993 (W 0.00272958) | 0.917254 ± 0.00192595 (W 0.0038169) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011392 ± 4.74377e-05 (W 3.91152e-05) | 0.008415 ± 7.19745e-05 (W 9.31085e-05) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014397 ± 0.0001505 (W 0.000106395) | 0.010309 ± 4.16333e-05 (W 0.000128325) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.044172 ± 0.000229129 (W 8.11985e-05) | 0.034524 ± 0.000357413 (W 6.54843e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.052377 ± 0.000478213 (W 9.17077e-05) | 0.038972 ± 0.000134946 (W 8.64292e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.10714 ± 0.00117958 (W 0.000795131) | 0.085139 ± 0.00115968 (W 0.00171429) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.123508 ± 0.000277421 (W 0.00044834) | 0.092803 ± 0.000131154 (W 0.00124434) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006653 ± 0.000106397 (W 0.000584213) | 0.006402 ± 1.73205e-05 (W 6.92618e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006512 ± 0.000100664 (W 0.000463231) | 0.006392 ± 2.51661e-05 (W 5.5857e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.013144 ± 0.00020656 (W 0.000372151) | 0.024906 ± 0.000693912 (W 0.00123619) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.013044 ± 0.000167482 (W 0.0001971) | 0.025097 ± 0.000218198 (W 0.00143809) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.025036 ± 0.000264059 (W 0.000512406) | 0.043972 ± 0.000540768 (W 0.000905653) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.025267 ± 0.000241109 (W 0.000361161) | 0.043912 ± 0.00103196 (W 0.000819441) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006302 ± 2.94449e-05 (W 4.47471e-05) | 0.007725 ± 0.000175009 (W 0.000131302) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008406 ± 1.0504e-05 (W 4.94389e-05) | 0.007083 ± 7.63763e-05 (W 9.79556e-05) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020208 ± 3.51188e-05 (W 0.000101026) | 0.035486 ± 0.000403651 (W 0.000238167) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.027011 ± 0.000215663 (W 0.000174554) | 0.026449 ± 0.000593669 (W 0.000829941) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.064088 ± 0.00128375 (W 0.00146617) | 0.089257 ± 0.000571194 (W 0.000873584) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.055402 ± 0.000174626 (W 0.000613731) | 0.063078 ± 0.000529907 (W 0.00181536) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.031739 ± 7.12344e-05 (W 3.43511e-05) | 0.057057 ± 0.000627748 (W 0.00119023) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.029845 ± 0.00015308 (W 9.95706e-05) | 0.051496 ± 0.000886525 (W 0.0014902) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.102882 ± 0.00175448 (W 0.0013028) | 0.280693 ± 0.00297426 (W 0.00401811) | - | 🔵 faer-fork 2.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.099896 ± 1.21244e-05 (W 0.000424178) | 0.227269 ± 0.00404039 (W 0.00125061) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.194889 ± 0.000215651 (W 0.00250429) | 0.583517 ± 0.00499869 (W 0.00624703) | - | 🔵 faer-fork 2.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.192836 ± 0.000216003 (W 0.00210373) | 0.457612 ± 0.00264622 (W 0.00332075) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.02159 ± 9.53939e-05 (W 0.000111769) | 0.04873 ± 0.00111284 (W 0.000973657) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.069079 ± 0.000266354 (W 0.000170092) | 0.243043 ± 0.00443129 (W 0.00334208) | - | 🔵 faer-fork 3.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.127588 ± 0.00127949 (W 0.00163024) | 0.483713 ± 0.00461164 (W 0.00727926) | - | 🔵 faer-fork 3.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007584 ± 7.0946e-05 (W 4.65833e-05) | 0.008356 ± 0.000114282 (W 6.78233e-05) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010199 ± 9.53939e-05 (W 7.46324e-05) | 0.007954 ± 4.78226e-05 (W 8.62392e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020789 ± 0.000195517 (W 9.98499e-05) | 0.036218 ± 0.000685406 (W 0.000232349) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.029064 ± 0.000166433 (W 0.000100634) | 0.02695 ± 0.00011682 (W 0.000204125) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.052136 ± 0.000818541 (W 0.0020713) | 0.086071 ± 0.000651714 (W 0.000609099) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.058258 ± 0.000307106 (W 0.000682529) | 0.061283 ± 0.000488906 (W 0.000861409) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.129381 ± 0.00251464 (W 0.000192576) | 0.139209 ± 0.000596957 (W 0.0021243) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.131675 ± 0.00115568 (W 0.00024004) | 0.139309 ± 0.00871068 (W 0.00231965) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.439038 ± 0.00116364 (W 0.00200585) | 0.546459 ± 0.00717358 (W 0.0042848) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.454907 ± 0.00248011 (W 0.00395905) | 0.542862 ± 0.00770078 (W 0.0038812) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.88046 ± 0.003181 (W 0.00274498) | 1.56354 ± 0.00160422 (W 0.00469628) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.883682 ± 0.00416036 (W 0.00327268) | 1.55996 ± 0.00152128 (W 0.00710138) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.048821 ± 0.000301785 (W 9.27362e-05) | 0.111137 ± 0.000423316 (W 0.000161802) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.049512 ± 0.000159478 (W 0.000220627) | 0.110945 ± 0.000353684 (W 0.00153268) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.171576 ± 0.00123411 (W 0.00160575) | 0.400109 ± 0.00100148 (W 0.00302826) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.173133 ± 0.00111222 (W 0.0019318) | 0.400676 ± 0.00146247 (W 0.00210618) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.384509 ± 0.00105978 (W 0.00183678) | 0.885635 ± 0.00269543 (W 0.00138663) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.383935 ± 0.00190243 (W 0.00248267) | 0.88559 ± 0.00064964 (W 0.00223493) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000962 ± 1.1547e-05 (W 4.50555e-05) | 0.000721 ± 5.7735e-07 (W 7.25003e-05) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002054 ± 2.13776e-05 (W 2.22553e-05) | 0.001473 ± 2.88675e-05 (W 7.8613e-05) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003407 ± 3.80832e-05 (W 4.06227e-05) | 0.002584 ± 5.50757e-06 (W 7.33485e-05) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001954 ± 2.88675e-05 (W 2.52626e-05) | 0.001864 ± 4.59021e-05 (W 7.18909e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002595 ± 2.64575e-05 (W 4.52769e-05) | 0.002575 ± 3.21455e-05 (W 9.42338e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008476 ± 3.15753e-05 (W 7.54533e-05) | 0.008446 ± 6.08276e-06 (W 7.205e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010009 ± 0.000101474 (W 0.000126888) | 0.010259 ± 9.45163e-05 (W 0.000175289) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.017733 ± 4.3589e-05 (W 7.96869e-05) | 0.017423 ± 9.30609e-05 (W 3.83184e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.020788 ± 0.000146277 (W 4.96991e-05) | 0.021039 ± 3.27159e-05 (W 9.13783e-05) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.087574 ± 0.00455973 (W 0.000298885) | 0.087182 ± 0.00050327 (W 0.000162127) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.087042 ± 6.4291e-05 (W 0.00166726) | 0.087122 ± 0.000242143 (W 0.00215818) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.323903 ± 0.00075423 (W 0.00262757) | 0.318655 ± 0.00179534 (W 0.00260581) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.326347 ± 0.000594093 (W 0.00218252) | 0.318896 ± 0.00251728 (W 0.00191452) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.948706 ± 0.00354577 (W 0.00138197) | 0.907788 ± 0.00346857 (W 0.00379966) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.95843 ± 0.0017512 (W 0.00194858) | 0.911874 ± 0.0177975 (W 0.00184326) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01023 ± 9.23093e-05 (W 4.26286e-05) | 0.008375 ± 6.0803e-05 (W 4.91355e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.011571 ± 1.76163e-05 (W 0.000155069) | 0.009848 ± 0.000121655 (W 0.000171628) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.041287 ± 0.000499984 (W 0.00010139) | 0.034504 ± 0.000345789 (W 4.3359e-05) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.04303 ± 0.000277128 (W 0.000109088) | 0.037741 ± 0.000121903 (W 0.00025926) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.101769 ± 0.000687016 (W 0.00259904) | 0.084427 ± 0.000675907 (W 0.00138566) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.104073 ± 0.00088963 (W 0.00173881) | 0.09069 ± 0.00072282 (W 0.00197627) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006763 ± 7.03586e-05 (W 7.10162e-05) | 0.006642 ± 5.89661e-05 (W 9.08636e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006843 ± 2.51661e-05 (W 2.88097e-05) | 0.006663 ± 0.0001005 (W 4.37299e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.027111 ± 7.99396e-05 (W 7.37584e-05) | 0.026619 ± 0.000109531 (W 0.000115062) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.027241 ± 0.000123087 (W 5.26308e-05) | 0.026569 ± 0.000312714 (W 6.63046e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.082513 ± 0.000222419 (W 0.0017642) | 0.0808 ± 0.000514265 (W 0.00200419) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.082464 ± 0.000498075 (W 0.000196643) | 0.0808 ± 0.000347565 (W 0.00146996) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.004909 ± 9.29157e-05 (W 5.87214e-05) | 0.0054 ± 7.55667e-05 (W 2.39625e-05) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006291 ± 8.0829e-05 (W 0.000114334) | 0.006733 ± 6.245e-05 (W 0.00010789) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017292 ± 7.84241e-05 (W 0.000171406) | 0.018845 ± 0.00012741 (W 0.00018727) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.020007 ± 9.53939e-05 (W 0.00192545) | 0.021971 ± 0.000117205 (W 0.000122066) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.037731 ± 0.000329485 (W 0.000241213) | 0.041598 ± 0.000340926 (W 0.000196281) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.041928 ± 0.000395664 (W 0.000266067) | 0.047118 ± 0.000341541 (W 0.000222034) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.023914 ± 0.000167989 (W 0.00125037) | 0.027191 ± 0.000166934 (W 8.78362e-05) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.025948 ± 0.000192718 (W 7.98204e-05) | 0.029355 ± 0.000237502 (W 8.69155e-05) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.089968 ± 0.000124527 (W 0.00175392) | 0.101659 ± 0.000340906 (W 0.00189802) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.094636 ± 0.000335188 (W 0.00190382) | 0.107009 ± 5.18684e-05 (W 0.00246876) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.215601 ± 0.000547206 (W 0.00211661) | 0.240818 ± 0.000790768 (W 0.00226974) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.222574 ± 0.00193369 (W 0.00218321) | 0.249434 ± 0.00129183 (W 0.00207951) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.014467 ± 8.3865e-05 (W 0.000145327) | 0.017703 ± 0.000190867 (W 0.000149547) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.059952 ± 0.000263801 (W 0.00159282) | 0.072024 ± 0.000150334 (W 0.000184747) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.127978 ± 0.000119654 (W 0.00227491) | 0.154529 ± 0.000452417 (W 0.00151776) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.005981 ± 7.2111e-05 (W 6.30872e-05) | 0.006031 ± 0.000138564 (W 4.63271e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007484 ± 5.2915e-05 (W 9.82054e-05) | 0.007283 ± 3.10966e-05 (W 8.6205e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.018134 ± 0.000215871 (W 3.61843e-05) | 0.019236 ± 0.000130615 (W 6.95291e-05) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.020748 ± 0.0001138 (W 0.000196973) | 0.022261 ± 0.000160614 (W 8.84873e-05) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.038231 ± 0.000248261 (W 0.00021327) | 0.042119 ± 0.000238957 (W 0.00183448) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.041898 ± 0.000246395 (W 0.00030612) | 0.046657 ± 0.000117304 (W 8.89129e-05) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.134991 ± 0.000425466 (W 0.00164678) | 0.145931 ± 0.000184858 (W 0.00195603) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.136454 ± 0.000771689 (W 0.00247452) | 0.146163 ± 0.000658048 (W 0.0025736) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.534749 ± 0.00482113 (W 0.00311778) | 0.535126 ± 0.00502536 (W 0.0018947) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.534735 ± 0.00683916 (W 0.00205885) | 0.535746 ± 0.00298996 (W 0.00205442) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.21924 ± 0.0027936 (W 0.00459243) | 1.56249 ± 0.00218203 (W 0.00345176) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.21704 ± 0.00431739 (W 0.00370396) | 1.58098 ± 0.00295395 (W 0.0039242) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.048951 ± 0.000462548 (W 0.00124869) | 0.108351 ± 0.00114892 (W 0.000150297) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.049011 ± 0.000277357 (W 0.000106039) | 0.109584 ± 0.000460892 (W 0.00128668) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.172201 ± 0.000861361 (W 0.00182667) | 0.398181 ± 0.00199858 (W 0.00242694) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.172782 ± 0.00202671 (W 0.00128619) | 0.397833 ± 0.000308 (W 0.00289797) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.374357 ± 0.00306182 (W 0.00291968) | 0.863975 ± 0.000652607 (W 0.00237249) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.37488 ± 0.00056349 (W 0.00241803) | 0.864155 ± 0.00231089 (W 0.002574) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000792 ± 5.19615e-06 (W 3.10838e-05) | 0.000761 ± 2.0664e-05 (W 2.72452e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001804 ± 3.76431e-05 (W 4.01659e-05) | 0.001703 ± 5.7735e-06 (W 4.08656e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.002895 ± 4.74377e-05 (W 9.24457e-05) | 0.002745 ± 3.4641e-05 (W 8.19408e-05) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001934 ± 2e-05 (W 2.16864e-05) | 0.001924 ± 9.29157e-05 (W 8.41742e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002684 ± 1.51767e-05 (W 6.09032e-05) | 0.002585 ± 1.78979e-05 (W 0.000110947) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008536 ± 1.52753e-05 (W 3.43511e-05) | 0.008485 ± 2.85832e-05 (W 4.219e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010099 ± 3.49333e-05 (W 3.5344e-05) | 0.010318 ± 7.35323e-05 (W 6.41872e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.018013 ± 7.83603e-05 (W 4.82701e-05) | 0.02171 ± 0.000204042 (W 0.000443769) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.021228 ± 2.00083e-05 (W 8.08591e-05) | 0.02682 ± 0.000503211 (W 0.000592684) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.064388 ± 2.31157e-05 (W 0.000292088) | 0.086702 ± 0.000590878 (W 0.00018519) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.064609 ± 0.000220659 (W 0.000336189) | 0.087403 ± 0.00313786 (W 0.000193213) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.251388 ± 0.000477097 (W 0.00148013) | 0.321739 ± 0.00290913 (W 0.00208533) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.252601 ± 0.00122767 (W 0.00174798) | 0.326658 ± 0.0024868 (W 0.0020295) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.549799 ± 0.00581737 (W 0.00277744) | 0.899705 ± 0.00425502 (W 0.00338769) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.560218 ± 0.00332807 (W 0.00369399) | 0.899761 ± 0.00109 (W 0.00180943) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011361 ± 9.31504e-05 (W 6.46475e-05) | 0.008365 ± 3.46458e-05 (W 4.71307e-05) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012473 ± 4.3589e-05 (W 0.000113798) | 0.009888 ± 0.00012822 (W 0.000133795) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.043581 ± 0.00048894 (W 0.000111068) | 0.034524 ± 0.00015624 (W 0.000147822) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.044963 ± 0.000509023 (W 0.000165831) | 0.038462 ± 0.000716224 (W 0.000127178) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.10723 ± 0.00143739 (W 0.000639145) | 0.084848 ± 0.000214929 (W 0.00178961) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.107791 ± 0.000702516 (W 0.000232055) | 0.092091 ± 0.000241581 (W 0.00154669) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005129 ± 7.76745e-05 (W 0.00018158) | 0.006783 ± 6.94286e-05 (W 6.17357e-05) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0052 ± 2.52389e-05 (W 0.000192349) | 0.006752 ± 0.000116863 (W 5.38266e-05) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014217 ± 0.000138023 (W 0.000390077) | 0.013615 ± 7.63763e-05 (W 0.00062541) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014386 ± 0.00031925 (W 0.00044338) | 0.013846 ± 0.000340554 (W 0.000356668) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.030567 ± 0.000171485 (W 0.00121545) | 0.029014 ± 0.000459805 (W 0.000335329) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.030517 ± 0.000218694 (W 0.00171247) | 0.029364 ± 0.000109865 (W 0.000294968) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.004949 ± 2.94449e-05 (W 3.20936e-05) | 0.00541 ± 5.7735e-06 (W 4.46789e-05) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006332 ± 4.96689e-05 (W 9.02774e-05) | 0.006873 ± 0.000104403 (W 0.000185688) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017312 ± 0.000177128 (W 0.000181108) | 0.0214 ± 0.00020459 (W 0.000516043) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.019836 ± 0.000117959 (W 0.000201939) | 0.024955 ± 0.000103636 (W 0.000201792) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.036478 ± 0.000201894 (W 0.000334495) | 0.051644 ± 0.000388796 (W 0.000705608) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.040274 ± 0.000831441 (W 0.000423105) | 0.058068 ± 0.000402534 (W 0.000899065) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.025428 ± 0.000114896 (W 7.15423e-05) | 0.039574 ± 0.000285272 (W 0.000402483) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.027732 ± 0.00013543 (W 0.000106894) | 0.04274 ± 0.000483308 (W 0.000313598) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.088985 ± 0.000101681 (W 0.000242143) | 0.172125 ± 0.000500929 (W 0.00255379) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.093382 ± 0.000522784 (W 0.000159669) | 0.180316 ± 0.00137006 (W 0.00223982) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.170547 ± 0.000677812 (W 0.00178255) | 0.341947 ± 0.00140081 (W 0.00385356) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.178628 ± 0.00100187 (W 0.001705) | 0.356684 ± 0.00224461 (W 0.00313863) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.015488 ± 3.49333e-05 (W 8.6487e-05) | 0.031118 ± 0.000226206 (W 0.000354367) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.062635 ± 0.000236514 (W 0.000172949) | 0.144218 ± 0.000311254 (W 0.00185498) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.11809 ± 0.000419199 (W 0.000376874) | 0.282225 ± 0.00365501 (W 0.00293224) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006472 ± 0.000105357 (W 3.27918e-05) | 0.006002 ± 6.05558e-05 (W 5.22494e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007955 ± 9.76746e-05 (W 7.62247e-05) | 0.007394 ± 3.4356e-05 (W 5.81403e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019065 ± 0.000272838 (W 6.14003e-05) | 0.02177 ± 4.78226e-05 (W 0.000288763) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.02177 ± 0.000292005 (W 0.000118107) | 0.025367 ± 0.000769801 (W 0.00161248) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.03828 ± 0.000396217 (W 0.000480251) | 0.049843 ± 0.000427506 (W 0.0014587) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.04261 ± 0.000220959 (W 0.000522245) | 0.056434 ± 0.000382558 (W 0.00130835) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.138548 ± 0.000725066 (W 0.000442013) | 0.14691 ± 0.00103015 (W 0.00178727) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.138718 ± 0.000703455 (W 0.000237225) | 0.146058 ± 0.00215873 (W 0.0019229) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.442864 ± 0.00180422 (W 0.00219799) | 0.542479 ± 0.00283321 (W 0.00238413) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.453662 ± 0.0047195 (W 0.00271582) | 0.545365 ± 0.00325973 (W 0.00301034) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.897377 ± 0.00355797 (W 0.00257792) | 1.52193 ± 0.01001 (W 0.00588944) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.907856 ± 0.00391998 (W 0.00125579) | 1.53053 ± 0.00495389 (W 0.00323553) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.050253 ± 0.000408512 (W 0.000234149) | 0.109611 ± 0.000196364 (W 0.00202348) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.050282 ± 0.000496329 (W 0.000145111) | 0.109123 ± 0.000330531 (W 0.000164971) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.175001 ± 0.000726631 (W 0.00197096) | 0.400236 ± 0.00186503 (W 0.00325259) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.174846 ± 0.000626952 (W 0.00132637) | 0.397921 ± 0.00244062 (W 0.00242644) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.386379 ± 0.000978232 (W 0.00235522) | 0.868547 ± 0.00184438 (W 0.00353554) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.388042 ± 0.00107067 (W 0.00237586) | 0.869639 ± 0.00254437 (W 0.00423566) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001002 ± 2.0664e-05 (W 2.32873e-05) | 0.000761 ± 5.7735e-07 (W 7.12482e-05) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002214 ± 3.05505e-05 (W 4.00287e-05) | 0.001744 ± 1.18462e-05 (W 7.00307e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003587 ± 4.16333e-05 (W 4.05487e-05) | 0.002795 ± 5.7735e-06 (W 7.99137e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001964 ± 0.000160053 (W 3.86303e-05) | 0.001924 ± 5.7735e-06 (W 0.000105044) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002695 ± 4.04145e-05 (W 4.72229e-05) | 0.002655 ± 2.08167e-05 (W 7.82943e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008506 ± 7.26659e-05 (W 3.95752e-05) | 0.008486 ± 6.65833e-05 (W 8.86408e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010079 ± 8.14453e-05 (W 4.96991e-05) | 0.01038 ± 2.62742e-05 (W 8.78704e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.018255 ± 8.25288e-05 (W 5.54076e-05) | 0.022782 ± 0.000266896 (W 0.000358758) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.02139 ± 0.00023767 (W 4.74342e-05) | 0.027471 ± 0.000180888 (W 0.000337213) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.060803 ± 0.000466241 (W 0.000206309) | 0.086772 ± 0.000189863 (W 0.000261802) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.061034 ± 0.000211866 (W 0.000239936) | 0.087674 ± 0.00091059 (W 0.000199537) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.197057 ± 0.00573659 (W 0.00178063) | 0.322731 ± 0.0238706 (W 0.00227409) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.198366 ± 0.000181533 (W 0.00174167) | 0.326663 ± 0.00111554 (W 0.00230372) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.412752 ± 0.0030522 (W 0.00288693) | 0.915611 ± 0.00593093 (W 0.00308299) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.411256 ± 0.0121251 (W 0.00233412) | 0.918642 ± 0.00835079 (W 0.00307012) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n32/b1` | 0.011401 ± 0.000192959 (W 5.70018e-05) | 0.008456 ± 0.000107364 (W 0.000389068) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012724 ± 5.91636e-05 (W 7.13877e-05) | 0.010019 ± 0.000131341 (W 0.000128974) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.044082 ± 0.000202436 (W 9.75925e-05) | 0.034424 ± 5.1316e-05 (W 3.64692e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.045435 ± 0.000284384 (W 0.000115195) | 0.03754 ± 0.000138023 (W 7.34316e-05) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.106748 ± 0.00118041 (W 0.000642504) | 0.084898 ± 0.000309187 (W 0.00222459) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.108012 ± 0.000638819 (W 0.000688413) | 0.092172 ± 0.000911551 (W 0.00166303) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00523 ± 0.000103923 (W 0.000108024) | 0.006813 ± 6.68356e-05 (W 8.04506e-05) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00528 ± 7.13933e-05 (W 0.000138746) | 0.006703 ± 1.76163e-05 (W 8.03119e-05) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.012433 ± 0.000414862 (W 0.000471026) | 0.025076 ± 0.000282739 (W 0.00110394) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.011982 ± 0.00012454 (W 0.000456778) | 0.025006 ± 0.000228007 (W 0.00118942) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.022402 ± 0.000149901 (W 0.000364553) | 0.040666 ± 0.000769797 (W 0.00060758) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.022171 ± 0.000236348 (W 0.000242776) | 0.041146 ± 0.00059195 (W 0.00175165) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.004979 ± 7.60285e-05 (W 6.64921e-05) | 0.00536 ± 2.08167e-05 (W 7.95613e-05) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006462 ± 3.27159e-05 (W 7.93429e-05) | 0.006733 ± 4.93288e-05 (W 9.8027e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017493 ± 5.17397e-05 (W 0.000215778) | 0.02174 ± 0.000153189 (W 0.000290767) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.020078 ± 0.000172705 (W 0.00017708) | 0.025197 ± 0.000335277 (W 0.00126758) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.036168 ± 0.000336051 (W 0.000346458) | 0.055343 ± 0.000405132 (W 0.00065727) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.040785 ± 0.000245024 (W 0.000445765) | 0.05969 ± 0.000393579 (W 0.000783872) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.025507 ± 3.56417e-05 (W 7.52017e-05) | 0.047307 ± 0.00231477 (W 0.00106936) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.027711 ± 0.000126808 (W 8.31865e-05) | 0.051225 ± 0.000846263 (W 0.000898954) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.088906 ± 0.000370307 (W 0.000165772) | 0.220918 ± 0.00175799 (W 0.00365316) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.093422 ± 0.00058243 (W 0.00033866) | 0.221062 ± 0.00191343 (W 0.00206288) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.16664 ± 0.000184831 (W 0.000450291) | 0.440894 ± 0.00236338 (W 0.00474849) | - | 🔵 faer-fork 2.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.176215 ± 0.000622524 (W 0.000670349) | 0.452666 ± 0.00119968 (W 0.00406134) | - | 🔵 faer-fork 2.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.015549 ± 2.3388e-05 (W 9.24662e-05) | 0.039052 ± 0.000516396 (W 0.00132558) | - | 🔵 faer-fork 2.51× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.062336 ± 3.49333e-05 (W 7.76737e-05) | 0.190612 ± 0.00299994 (W 0.00458358) | - | 🔵 faer-fork 3.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.117509 ± 0.000754778 (W 0.000766181) | 0.367178 ± 0.00714053 (W 0.00430237) | - | 🔵 faer-fork 3.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006362 ± 7.76745e-05 (W 7.58288e-05) | 0.006052 ± 4.33628e-05 (W 6.55385e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007984 ± 4.59384e-05 (W 7.02652e-05) | 0.007474 ± 6e-05 (W 2.50998e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019476 ± 4.04145e-05 (W 6.08786e-05) | 0.02194 ± 0.00020297 (W 0.000200924) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.02185 ± 0.000113873 (W 6.88716e-05) | 0.025588 ± 0.000243824 (W 0.00136791) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.038962 ± 0.000457371 (W 0.000399565) | 0.052698 ± 6.69104e-05 (W 0.00093294) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.042769 ± 0.000611 (W 0.00125593) | 0.058598 ± 0.000591677 (W 0.000359111) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.138728 ± 0.000236011 (W 0.000139504) | 0.146243 ± 0.00203779 (W 0.0023041) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.13929 ± 0.0037445 (W 0.000250811) | 0.147274 ± 0.000446098 (W 0.00166668) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.435371 ± 0.00112704 (W 0.00202786) | 0.543784 ± 0.00343863 (W 0.00411606) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.453535 ± 0.000772026 (W 0.00338091) | 0.548221 ± 0.00316357 (W 0.0039913) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.887029 ± 0.0116869 (W 0.00257978) | 1.55527 ± 0.00343053 (W 0.00702905) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.887734 ± 0.00292647 (W 0.00354531) | 1.55976 ± 0.00525016 (W 0.00890844) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.049863 ± 0.00012822 (W 0.000141022) | 0.109224 ± 0.000108242 (W 0.00238462) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.050364 ± 0.000297761 (W 9.20342e-05) | 0.109995 ± 0.000597315 (W 0.000216003) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.176258 ± 0.00117445 (W 0.000620431) | 0.399744 ± 0.000745217 (W 0.00247842) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.176729 ± 0.00132277 (W 0.00183768) | 0.401708 ± 0.00134981 (W 0.00237982) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.387291 ± 0.00193701 (W 0.00280186) | 0.866194 ± 0.0028076 (W 0.00118587) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.388634 ± 0.00342391 (W 0.00237202) | 0.867557 ± 0.00168275 (W 0.00219259) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001022 ± 1e-05 (W 3.34664e-05) | 0.000782 ± 1.12694e-05 (W 7.74293e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002254 ± 4.6188e-05 (W 4.1833e-05) | 0.001673 ± 1e-05 (W 6.9601e-05) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003476 ± 2.60832e-05 (W 4.75111e-05) | 0.002795 ± 3.51188e-05 (W 9.8185e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012703 ± 8.49137e-05 (W 7.13674e-05) | 0.011181 ± 5.34821e-05 (W 1.09545e-05) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012073 ± 3.40637e-05 (W 7.93996e-05) | 0.012053 ± 0.000144627 (W 0.000100256) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.070442 ± 0.000199234 (W 0.000214321) | 0.088004 ± 0.000304921 (W 0.000285754) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066614 ± 0.00020664 (W 0.00163489) | 0.066724 ± 0.000308513 (W 0.00128822) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.115736 ± 0.000647168 (W 0.00052133) | 0.472279 ± 0.000415722 (W 0.00260598) | - | 🔵 faer-fork 4.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.110085 ± 0.000340783 (W 0.00023467) | 0.110025 ± 0.000399267 (W 0.00145164) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.104294 ± 0.000860508 (W 0.00207685) | 0.104565 ± 0.000746507 (W 0.00017201) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.104996 ± 0.000488414 (W 0.000376375) | 0.104535 ± 0.000351019 (W 0.00198584) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.406256 ± 0.000634032 (W 0.00267177) | 0.410615 ± 0.00438614 (W 0.0024713) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.410708 ± 0.0037509 (W 0.00272775) | 0.406129 ± 0.00075508 (W 0.00287318) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.17732 ± 0.00279112 (W 0.0039133) | 1.12429 ± 0.0012837 (W 0.00177143) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.17641 ± 0.000519841 (W 0.0027317) | 1.15852 ± 0.000834578 (W 0.00340353) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013846 ± 4.67654e-05 (W 7.21124e-05) | 0.01072 ± 8.79223e-05 (W 7.60736e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.017292 ± 0.000193124 (W 0.000103079) | 0.012282 ± 3.23574e-05 (W 0.000122373) | - | 🔴 faer-fork 1.41× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.064861 ± 0.000333182 (W 0.0014161) | 0.052768 ± 5.1316e-05 (W 0.000105499) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.07517 ± 0.000635183 (W 0.000189774) | 0.057217 ± 0.000643547 (W 0.00178) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.180586 ± 0.000151596 (W 0.00214072) | 0.151573 ± 0.00102901 (W 0.00125786) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.206234 ± 0.000935091 (W 0.002569) | 0.16692 ± 0.000376816 (W 0.00290876) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.017864 ± 6.5041e-05 (W 8.77286e-05) | 0.017733 ± 0.000119304 (W 0.000105927) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.017753 ± 9.0185e-05 (W 8.95545e-05) | 0.017583 ± 0.000105879 (W 0.000130082) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.091801 ± 0.000500561 (W 0.000177995) | 0.090999 ± 0.000243662 (W 0.000644051) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.091891 ± 0.000159478 (W 0.00149955) | 0.091209 ± 6.62898e-05 (W 0.000141634) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.267057 ± 0.00151477 (W 0.00281926) | 0.266889 ± 0.00196966 (W 0.00283454) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.26837 ± 0.000449501 (W 0.00231408) | 0.265494 ± 0.00049538 (W 0.0019261) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.009788 ± 5.50757e-05 (W 6.86236e-05) | 0.009738 ± 0.000115036 (W 7.22233e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012282 ± 0.000246313 (W 0.000117504) | 0.009617 ± 0.000318577 (W 0.000120562) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.03763 ± 0.00010583 (W 9.91464e-05) | 0.044653 ± 0.000346541 (W 0.000182796) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.045274 ± 0.000337817 (W 0.00208488) | 0.036899 ± 0.000244357 (W 0.000136384) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.09129 ± 0.000411303 (W 0.00179523) | 0.1127 ± 0.000462018 (W 0.00163917) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.116197 ± 3.5171e-05 (W 0.0023294) | 0.096971 ± 0.000475955 (W 0.00241787) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.05367 ± 0.000124527 (W 0.000223829) | 0.057067 ± 0.000613776 (W 0.00204976) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.050654 ± 0.000194054 (W 0.000206914) | 0.050454 ± 0.000743201 (W 0.000150663) | - | · | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.187991 ± 0.000397519 (W 0.00248526) | 0.276946 ± 0.000191147 (W 0.0020906) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.196487 ± 0.000582305 (W 0.00201077) | 0.201396 ± 0.000320347 (W 0.00153764) | - | · | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.442146 ± 0.00148641 (W 0.00329436) | 0.640857 ± 0.00435367 (W 0.0028138) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.499974 ± 0.0011257 (W 0.00255663) | 0.51582 ± 0.00194855 (W 0.00343627) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.03232 ± 4.50925e-05 (W 0.00167584) | 0.036347 ± 0.000224121 (W 0.000118597) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.104345 ± 0.000300982 (W 0.00183799) | 0.185866 ± 0.000951203 (W 0.00179998) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.230149 ± 0.000525071 (W 0.00179207) | 0.417179 ± 0.00311824 (W 0.00242682) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0106 ± 7.23418e-05 (W 4.28404e-05) | 0.01083 ± 0.000100167 (W 3.83706e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013546 ± 6.39296e-05 (W 0.000130929) | 0.01066 ± 0.000272116 (W 0.000181191) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.03715 ± 0.000151397 (W 0.00013588) | 0.047519 ± 0.000816546 (W 0.000132869) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.047639 ± 7.05715e-05 (W 8.39125e-05) | 0.039343 ± 0.000206027 (W 0.000134165) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.088946 ± 0.000415216 (W 0.00027682) | 0.116527 ± 0.000371539 (W 0.00163834) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.11793 ± 0.00072903 (W 0.00251533) | 0.099204 ± 0.000341968 (W 0.0028274) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.160078 ± 0.000823602 (W 0.0016142) | 0.171298 ± 0.00045741 (W 0.00205629) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.160839 ± 0.000812571 (W 0.00177096) | 0.17211 ± 0.000797469 (W 0.00162997) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.68731 ± 0.0047367 (W 0.00270136) | 0.686478 ± 0.00405461 (W 0.00258877) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.689333 ± 0.0039107 (W 0.00248916) | 0.702759 ± 0.00404026 (W 0.00358296) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.65109 ± 0.0161152 (W 0.00369139) | 2.03965 ± 0.00676571 (W 0.00482101) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.66202 ± 0.00596995 (W 0.0062776) | 2.04868 ± 0.00714328 (W 0.00519181) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.056496 ± 0.00057149 (W 0.000111783) | 0.118501 ± 0.000315753 (W 0.00173187) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.058589 ± 0.000592807 (W 0.000146483) | 0.120815 ± 0.000155885 (W 0.000246219) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.22504 ± 0.000823347 (W 0.00232691) | 0.46235 ± 0.00134133 (W 0.00242909) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.220281 ± 0.000572717 (W 0.00170239) | 0.464955 ± 0.00140455 (W 0.00280764) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.541618 ± 0.000993496 (W 0.00254119) | 1.04573 ± 0.0046957 (W 0.00347612) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.543461 ± 0.00311941 (W 0.00240599) | 1.05215 ± 0.00388571 (W 0.00261471) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001062 ± 1e-05 (W 2.94958e-05) | 0.000962 ± 2.64575e-05 (W 7.48679e-05) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002524 ± 2.28108e-05 (W 8.52074e-05) | 0.002255 ± 2.00083e-05 (W 2.89707e-05) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004378 ± 1.12694e-05 (W 3.88484e-05) | 0.003997 ± 2.6839e-05 (W 3.52321e-05) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012814 ± 7.6922e-05 (W 4.31312e-05) | 0.01117 ± 1.48436e-05 (W 8.62804e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012073 ± 8.99463e-05 (W 6.9809e-05) | 0.012092 ± 0.000166764 (W 0.000102958) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.07017 ± 0.000379944 (W 0.000194166) | 0.088265 ± 0.000117779 (W 9.85409e-05) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066594 ± 0.000872105 (W 0.00025896) | 0.066574 ± 0.000127005 (W 0.000208567) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.113421 ± 0.000410152 (W 0.00026963) | 0.135118 ± 0.000611863 (W 0.133036) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108452 ± 0.000367131 (W 0.000296034) | 0.115094 ± 0.000309049 (W 0.000625193) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.084598 ± 0.000247322 (W 0.00180425) | 0.104184 ± 0.000211637 (W 0.00174908) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.085257 ± 0.000266456 (W 0.000302024) | 0.104633 ± 0.00266212 (W 0.00238027) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.321538 ± 0.00129244 (W 0.00225421) | 0.389245 ± 0.00178717 (W 0.00201953) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.32229 ± 0.00143602 (W 0.00214758) | 0.390909 ± 0.00243531 (W 0.00276259) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.693024 ± 0.00251768 (W 0.00306601) | 1.02822 ± 0.00188794 (W 0.00276203) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.692854 ± 0.000943967 (W 0.00279865) | 1.03256 ± 0.000858098 (W 0.00112697) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014727 ± 0.000404348 (W 3.24037e-05) | 0.010619 ± 0.00017091 (W 5.36656e-05) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018514 ± 0.000285015 (W 7.62889e-05) | 0.012443 ± 0.000113271 (W 0.000153164) | - | 🔴 faer-fork 1.49× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.067133 ± 0.000723467 (W 0.000169174) | 0.052779 ± 0.000117951 (W 7.62712e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.077364 ± 0.00743033 (W 0.00019956) | 0.057548 ± 0.000297106 (W 0.000194991) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.169922 ± 0.00240771 (W 0.0019129) | 0.151813 ± 0.000518582 (W 0.00210905) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.199797 ± 0.00737287 (W 0.00234636) | 0.165147 ± 0.000477587 (W 0.00224574) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.010098 ± 0.000290569 (W 0.000345402) | 0.017893 ± 3.27159e-05 (W 8.28855e-05) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009507 ± 8.32666e-05 (W 0.000157297) | 0.017753 ± 5.08626e-05 (W 0.000133515) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.036738 ± 0.000297602 (W 0.000703048) | 0.036267 ± 0.000145041 (W 0.00039657) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.036908 ± 0.000150932 (W 0.000409891) | 0.035455 ± 0.000126429 (W 0.000158988) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.088324 ± 0.000419488 (W 0.00170902) | 0.086241 ± 0.000256782 (W 0.000653492) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.088796 ± 0.000697883 (W 0.000855572) | 0.085838 ± 0.000140146 (W 0.000425729) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.009768 ± 0.000122231 (W 8.93308e-05) | 0.009778 ± 2.51661e-05 (W 4.36726e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012233 ± 0.000192873 (W 5.77062e-05) | 0.009438 ± 4.0501e-05 (W 0.000112726) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.044392 ± 0.000515385 (W 0.000830869) | 0.047178 ± 0.000434583 (W 0.00116169) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.044843 ± 0.000584894 (W 0.000438169) | 0.040676 ± 0.000163025 (W 0.00100215) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.094944 ± 0.000228728 (W 0.00115852) | 0.115555 ± 0.00108872 (W 0.00112211) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.106158 ± 0.00104738 (W 0.00235851) | 0.101296 ± 0.000337963 (W 0.00205496) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.054812 ± 0.00018155 (W 0.000246997) | 0.070461 ± 0.000773693 (W 0.00138564) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.052188 ± 0.000125654 (W 0.000208723) | 0.064289 ± 0.00055303 (W 0.000573708) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.140718 ± 0.000166304 (W 0.00166734) | 0.32242 ± 0.00410682 (W 0.00263849) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.14676 ± 0.00114864 (W 0.000593288) | 0.2417 ± 0.00217144 (W 0.00238639) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.284119 ± 0.00225571 (W 0.00188703) | 0.622629 ± 0.000909053 (W 0.00360962) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.324655 ± 0.00208079 (W 0.00237545) | 0.495603 ± 0.00257397 (W 0.0025763) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.033191 ± 8.72372e-05 (W 0.00012145) | 0.049222 ± 0.00055797 (W 0.00191649) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.097129 ± 0.000278073 (W 0.000289977) | 0.250216 ± 0.000460059 (W 0.00261225) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.195504 ± 0.000123005 (W 0.00200711) | 0.504136 ± 0.000540733 (W 0.00352244) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011251 ± 0.000111505 (W 5.5857e-05) | 0.01076 ± 8.73689e-05 (W 8.23663e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013915 ± 0.000163147 (W 0.000110924) | 0.010659 ± 9.15442e-05 (W 0.000134106) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.041046 ± 0.000231157 (W 0.000460995) | 0.048871 ± 0.000220953 (W 0.000419706) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.047298 ± 0.000796574 (W 0.000368442) | 0.041858 ± 0.000956108 (W 0.000700374) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.089544 ± 0.000594961 (W 0.00190597) | 0.115385 ± 0.00102076 (W 0.000658971) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.111137 ± 0.000429221 (W 0.00263382) | 0.100044 ± 0.000969176 (W 0.00212491) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.163554 ± 0.00116916 (W 0.000283575) | 0.170317 ± 0.000989412 (W 0.00213649) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.16645 ± 0.00312767 (W 0.000259933) | 0.171359 ± 0.00280692 (W 0.00222874) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.535378 ± 0.000961033 (W 0.00191617) | 0.648297 ± 0.00622494 (W 0.00253812) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.538931 ± 0.00193713 (W 0.00207199) | 0.651878 ± 0.000501135 (W 0.00176042) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.12739 ± 0.00722391 (W 0.00396375) | 1.8374 ± 0.00353686 (W 0.00497042) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.12922 ± 0.00382439 (W 0.00346776) | 1.85215 ± 0.00416734 (W 0.00183689) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058007 ± 0.000429932 (W 0.000193076) | 0.118748 ± 0.000106613 (W 0.00169643) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.058728 ± 0.000380773 (W 0.000222768) | 0.119192 ± 0.00026569 (W 0.00026494) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.223401 ± 0.00179493 (W 0.00153344) | 0.456088 ± 0.00112117 (W 0.00280284) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.226622 ± 0.00517691 (W 0.00175203) | 0.455097 ± 0.00122002 (W 0.00224862) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.524026 ± 0.00183648 (W 0.00249749) | 1.05816 ± 0.00138676 (W 0.00174971) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.524667 ± 0.000526767 (W 0.00249956) | 1.06062 ± 0.00481417 (W 0.000862292) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001323 ± 2.51064e-05 (W 2.33088e-05) | 0.000942 ± 1.57162e-05 (W 8.34596e-05) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003005 ± 2.58908e-05 (W 2.52052e-05) | 0.002234 ± 0.000280678 (W 0.000123609) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00514 ± 2.5632e-05 (W 3.79763e-05) | 0.003938 ± 2.09841e-05 (W 4.49366e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012853 ± 0.000125021 (W 3.50714e-05) | 0.011161 ± 6.50641e-05 (W 4.02492e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012102 ± 4.72582e-05 (W 7.25899e-05) | 0.012103 ± 3.55012e-05 (W 7.10859e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.070812 ± 0.000823419 (W 0.000206459) | 0.088114 ± 0.000155532 (W 0.00191456) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.067034 ± 0.000195526 (W 0.000208459) | 0.066884 ± 0.000146118 (W 0.00111899) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.11356 ± 0.000146507 (W 0.000376637) | 0.135971 ± 0.0220289 (W 0.131017) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.108632 ± 0.000352467 (W 0.000328766) | 0.115684 ± 0.000192096 (W 0.00180516) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.081242 ± 0.000311043 (W 0.000291283) | 0.104303 ± 0.000236658 (W 0.00134957) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.082041 ± 0.000260229 (W 0.000262649) | 0.104735 ± 0.000263618 (W 0.00194484) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.259123 ± 0.000668972 (W 0.00170377) | 0.391639 ± 0.00596797 (W 0.00354014) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.261016 ± 0.000463947 (W 0.00180041) | 0.392145 ± 0.000767185 (W 0.00308574) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.547981 ± 0.0738824 (W 0.00273479) | 1.03083 ± 0.000818471 (W 0.00375147) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1 [api-inclusive]` | 0.545228 ± 0.0105529 (W 0.00238876) | 1.03407 ± 0.00187848 (W 0.00480888) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014997 ± 0.0002255 (W 6.69425e-05) | 0.01055 ± 7.97308e-05 (W 7.96869e-05) | - | 🔴 faer-fork 1.42× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018474 ± 0.000157608 (W 9.84749e-05) | 0.012313 ± 0.00018791 (W 5.63028e-05) | - | 🔴 faer-fork 1.50× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.067346 ± 0.00072888 (W 8.51311e-05) | 0.052697 ± 0.000664969 (W 0.000101559) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.077475 ± 0.000375377 (W 0.000461561) | 0.057376 ± 6.56531e-05 (W 0.000153141) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.176368 ± 0.00178307 (W 0.00146231) | 0.151933 ± 8.51352e-05 (W 0.00224773) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.205793 ± 0.0115584 (W 0.00262671) | 0.16599 ± 6.03517e-05 (W 0.00233721) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.007935 ± 8.18535e-05 (W 0.000171188) | 0.017693 ± 8.88819e-05 (W 4.82701e-05) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008115 ± 5.2915e-05 (W 0.000206689) | 0.017873 ± 9.71253e-05 (W 0.000155805) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.024245 ± 0.000140678 (W 0.000436714) | 0.035967 ± 0.000216888 (W 0.000287733) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.024245 ± 0.000347132 (W 0.000462795) | 0.035917 ± 0.00021394 (W 0.000598032) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.055242 ± 0.000335107 (W 0.000621429) | 0.095909 ± 0.000256206 (W 0.00125333) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.056225 ± 0.000579169 (W 0.000782228) | 0.095638 ± 0.000563948 (W 0.00194395) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.009868 ± 3e-05 (W 9.47101e-05) | 0.009758 ± 2e-05 (W 4.72229e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012424 ± 0.000114839 (W 8.1708e-05) | 0.009467 ± 2.3094e-05 (W 6.1606e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.044061 ± 0.000385803 (W 0.000677216) | 0.047268 ± 0.000430535 (W 0.00020809) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.045174 ± 0.000595973 (W 0.000741526) | 0.041257 ± 0.000383719 (W 0.00152195) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.101699 ± 0.000156205 (W 0.00129514) | 0.116296 ± 0.000395524 (W 0.00167277) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.110125 ± 0.000211285 (W 0.00224735) | 0.10314 ± 0.000966409 (W 0.00261275) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.055032 ± 0.000228417 (W 0.000237979) | 0.08052 ± 0.013019 (W 0.00213697) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.052097 ± 0.000173494 (W 0.000163908) | 0.072454 ± 0.00065661 (W 0.000807082) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.14065 ± 0.000317093 (W 0.000845874) | 0.355572 ± 0.00072351 (W 0.00752974) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.140311 ± 0.000877677 (W 0.000506349) | 0.284435 ± 0.00221945 (W 0.00279465) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.263721 ± 0.00161603 (W 0.00179126) | 0.68788 ± 0.00704718 (W 0.00900507) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.30278 ± 0.000675336 (W 0.0027489) | 0.576655 ± 0.00604825 (W 0.00598766) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.033242 ± 0.00011415 (W 0.000130652) | 0.057988 ± 0.00138799 (W 0.00183928) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.094654 ± 0.00024278 (W 0.000344165) | 0.284109 ± 0.0032481 (W 0.00623287) | - | 🔵 faer-fork 3.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.180356 ± 0.000753258 (W 0.00162859) | 0.567317 ± 0.00520159 (W 0.00877237) | - | 🔵 faer-fork 3.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011261 ± 8.88819e-05 (W 4.4025e-05) | 0.01076 ± 7.02377e-05 (W 7.4162e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014056 ± 4.58258e-05 (W 9.31467e-05) | 0.010971 ± 0.000364402 (W 0.000178278) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.043741 ± 0.000359212 (W 0.000613769) | 0.049662 ± 0.000225019 (W 0.000643142) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.047669 ± 6.09945e-05 (W 0.000351512) | 0.042479 ± 4.0501e-05 (W 0.00147824) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.09169 ± 7.60548e-05 (W 0.00120503) | 0.114832 ± 0.00056732 (W 0.00119732) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.113041 ± 0.000318858 (W 0.00200777) | 0.102691 ± 0.00028746 (W 0.00228803) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.164657 ± 0.00160455 (W 0.000500708) | 0.170307 ± 0.000490664 (W 0.00204886) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.164596 ± 0.000514689 (W 0.000165932) | 0.172399 ± 0.000617919 (W 0.00207078) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.519701 ± 0.00258502 (W 0.00200683) | 0.650663 ± 0.000648436 (W 0.00306167) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.533857 ± 0.00235119 (W 0.00324695) | 0.653969 ± 0.00318731 (W 0.00305407) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.09742 ± 0.00211848 (W 0.00292963) | 1.85718 ± 0.00548864 (W 0.00626438) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.09794 ± 0.00141593 (W 0.00261777) | 1.85634 ± 0.0020538 (W 0.00557559) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058138 ± 0.000356541 (W 0.000188065) | 0.119032 ± 0.000189071 (W 0.000193169) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.05903 ± 0.000632425 (W 0.000239343) | 0.119463 ± 0.000243032 (W 0.000193399) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.224689 ± 0.000690096 (W 0.00198142) | 0.458373 ± 0.00181736 (W 0.00244429) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.224435 ± 0.000631215 (W 0.00177655) | 0.456641 ± 0.000531638 (W 0.00216445) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.53175 ± 0.0021009 (W 0.00263375) | 1.04875 ± 0.00631912 (W 0.00117683) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.545248 ± 0.000842601 (W 0.00214817) | 1.05878 ± 0.00528247 (W 0.00168484) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001302 ± 1.53948e-05 (W 2.86042e-05) | 0.000971 ± 2.00083e-05 (W 8.73499e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003015 ± 2.04206e-05 (W 3.02374e-05) | 0.002264 ± 5.7735e-07 (W 7.48218e-05) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00515 ± 3.60555e-05 (W 3.02043e-05) | 0.004007 ± 5.0797e-05 (W 8.44885e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011511 ± 4.16333e-05 (W 4.07836e-05) | 0.011421 ± 7.60723e-05 (W 4.15969e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012032 ± 6.16468e-05 (W 0.000113882) | 0.011982 ± 2.88675e-05 (W 0.000133249) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.065051 ± 0.00032894 (W 0.00224843) | 0.064811 ± 0.000255408 (W 0.000186939) | - | · | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066524 ± 0.000388421 (W 0.000267797) | 0.066855 ± 0.00013027 (W 6.24195e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.106438 ± 0.000269482 (W 0.00169612) | 0.105247 ± 0.000425019 (W 0.000301413) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.109285 ± 0.000325593 (W 0.00039383) | 0.109294 ± 0.000337471 (W 0.000184806) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.104445 ± 0.000385919 (W 0.000162728) | 0.103313 ± 0.000467789 (W 0.00013787) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.105117 ± 0.000577721 (W 0.0016978) | 0.105797 ± 0.00102472 (W 0.00182828) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.406446 ± 0.000880725 (W 0.00224596) | 0.406189 ± 0.00282066 (W 0.00214616) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.413891 ± 0.00230787 (W 0.00252469) | 0.406206 ± 0.00142724 (W 0.00228028) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.17558 ± 0.00527888 (W 0.00662933) | 1.13035 ± 0.00270085 (W 0.00686412) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.17703 ± 0.0056674 (W 0.00305977) | 1.15604 ± 0.00320734 (W 0.00547501) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013726 ± 8.12835e-05 (W 3.82518e-05) | 0.01063 ± 0.000141067 (W 6.04152e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015188 ± 0.000122748 (W 9.37726e-05) | 0.012142 ± 6.5041e-05 (W 0.00012614) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.065052 ± 0.000822308 (W 9.59125e-05) | 0.054752 ± 0.0067041 (W 0.0001906) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.067506 ± 0.000433859 (W 0.000176862) | 0.056265 ± 0.000231215 (W 0.00127073) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.181289 ± 0.000892404 (W 0.00221993) | 0.151563 ± 0.00100217 (W 0.00129927) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.190334 ± 0.000240042 (W 0.00212412) | 0.164998 ± 0.00050128 (W 0.00290968) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.017823 ± 2.51661e-05 (W 0.000115629) | 0.017643 ± 0.000120554 (W 0.000156748) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.017974 ± 9.59705e-05 (W 6.1149e-05) | 0.017673 ± 9.60902e-05 (W 0.000104785) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.090209 ± 0.000299001 (W 0.000269361) | 0.090008 ± 0.000313784 (W 0.00200541) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.090739 ± 0.000426768 (W 0.000190461) | 0.090168 ± 0.000157906 (W 0.00181092) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.272647 ± 0.00150904 (W 0.00184469) | 0.271144 ± 0.00124779 (W 0.00253746) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.273108 ± 0.00151911 (W 0.00162285) | 0.269692 ± 0.00100088 (W 0.0022052) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008416 ± 6.98594e-05 (W 7.82125e-05) | 0.007805 ± 7.97517e-05 (W 4.48029e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010089 ± 0.000120625 (W 0.000144393) | 0.009187 ± 7.81025e-05 (W 0.000112718) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034254 ± 0.000173494 (W 0.000269413) | 0.03237 ± 0.000221886 (W 7.67802e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.037881 ± 0.000344399 (W 0.000206325) | 0.035627 ± 0.000361594 (W 0.000179737) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.084267 ± 0.000408694 (W 0.000276136) | 0.081032 ± 0.00032827 (W 0.000349651) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.09684 ± 0.000267266 (W 0.0027337) | 0.094456 ± 0.000211975 (W 0.000397415) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.045766 ± 8.16721e-05 (W 0.000153891) | 0.047528 ± 0.0007064 (W 0.000229675) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.04865 ± 0.000307106 (W 0.000279157) | 0.050363 ± 0.000482288 (W 0.0001011) | - | · | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.18264 ± 0.00183092 (W 0.00280479) | 0.191788 ± 0.000514187 (W 0.00196037) | - | · | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.187759 ± 0.00102138 (W 0.00213961) | 0.198069 ± 0.000741043 (W 0.00179717) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.449326 ± 0.000946849 (W 0.00265121) | 0.470439 ± 0.00301353 (W 0.00232926) | - | · | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.485273 ± 0.00164925 (W 0.00449126) | 0.507277 ± 0.00336015 (W 0.00373534) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.024545 ± 0.000157608 (W 0.000122556) | 0.02695 ± 0.000369514 (W 0.000220635) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.103123 ± 0.000115154 (W 0.000189497) | 0.113661 ± 0.000503827 (W 0.000187927) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.236701 ± 0.000563715 (W 0.00182017) | 0.263603 ± 0.00189719 (W 0.0024283) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.009337 ± 2.51661e-05 (W 3.98033e-05) | 0.008777 ± 5.72742e-05 (W 5.52377e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.011151 ± 6.37992e-05 (W 0.000102899) | 0.01026 ± 0.000128376 (W 6.88934e-05) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.035557 ± 3.60555e-05 (W 0.000113535) | 0.034083 ± 0.000275863 (W 8.61974e-05) | - | · | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.039554 ± 0.000362952 (W 0.000156335) | 0.038361 ± 0.00733801 (W 0.000538494) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.0861 ± 0.000547899 (W 0.00264767) | 0.084247 ± 0.000335054 (W 0.00161348) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.100647 ± 0.0003215 (W 0.00252989) | 0.09681 ± 0.000248818 (W 0.00254612) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.164145 ± 0.00163109 (W 0.0013624) | 0.172763 ± 0.000189432 (W 0.00158146) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.161532 ± 0.00256689 (W 0.00193258) | 0.174134 ± 0.000208167 (W 0.00126617) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.687986 ± 0.00445906 (W 0.00242647) | 0.693671 ± 0.00276132 (W 0.00308575) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.689788 ± 0.00147977 (W 0.00232344) | 0.713388 ± 0.00163271 (W 0.00387603) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.65089 ± 0.00387111 (W 0.00557033) | 2.04432 ± 0.00770868 (W 0.00849675) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.65488 ± 0.00275081 (W 0.00573296) | 2.07178 ± 0.00238649 (W 0.00388589) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058259 ± 0.000327522 (W 0.00162015) | 0.118822 ± 0.00045235 (W 0.0022336) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060072 ± 0.000556367 (W 0.0002775) | 0.121206 ± 0.00156294 (W 0.000204457) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.225351 ± 0.00227968 (W 0.00204549) | 0.455177 ± 0.00166949 (W 0.00278979) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.21929 ± 0.000691536 (W 0.00206406) | 0.458804 ± 0.00270394 (W 0.00201449) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.545258 ± 0.000846662 (W 0.00269075) | 1.04652 ± 0.00171438 (W 0.00234787) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.546741 ± 0.00191397 (W 0.00355563) | 1.05044 ± 0.00358891 (W 0.00156665) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001143 ± 2.52389e-05 (W 6.72993e-05) | 0.001152 ± 1.58219e-05 (W 6.54217e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002785 ± 2e-05 (W 8.55354e-05) | 0.002805 ± 2e-05 (W 3.52463e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004318 ± 2.70247e-05 (W 8.02515e-05) | 0.004359 ± 2.3388e-05 (W 4.92159e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011451 ± 2.36714e-05 (W 4.98327e-05) | 0.011342 ± 4.03361e-05 (W 7.80916e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012183 ± 5.25389e-05 (W 5.60732e-05) | 0.012022 ± 9.66454e-05 (W 5.92891e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.065342 ± 0.000101953 (W 0.000178802) | 0.065009 ± 9.8348e-05 (W 0.00188422) | - | · | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066223 ± 8.77515e-05 (W 0.000223036) | 0.066332 ± 0.000185584 (W 0.000127653) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.104094 ± 5.40648e-05 (W 0.000112144) | 0.108809 ± 0.000235848 (W 0.000323806) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.107901 ± 0.000351696 (W 0.000318178) | 0.114463 ± 0.000428222 (W 0.000481868) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.084849 ± 0.00201384 (W 0.000168688) | 0.104345 ± 0.00029905 (W 0.0026019) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.085279 ± 8.45833e-05 (W 0.000308818) | 0.104585 ± 0.000527394 (W 0.000244232) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.322223 ± 0.000523673 (W 0.00167755) | 0.388754 ± 0.00177911 (W 0.00213631) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.321999 ± 0.000314793 (W 0.00278387) | 0.389806 ± 0.000547164 (W 0.00207682) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.69233 ± 0.00272746 (W 0.00327808) | 1.0285 ± 0.00169635 (W 0.00345306) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.690549 ± 0.00272936 (W 0.00316754) | 1.03071 ± 0.000883518 (W 0.00138829) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014848 ± 0.000176976 (W 3.82649e-05) | 0.010569 ± 8.41249e-05 (W 9.14724e-05) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.016521 ± 0.00205169 (W 7.30068e-05) | 0.012063 ± 0.000242502 (W 0.000104405) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.066674 ± 0.000344192 (W 0.000103842) | 0.052568 ± 0.000130193 (W 9.02286e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.069239 ± 0.000523989 (W 0.000133668) | 0.056084 ± 0.00016856 (W 0.000113032) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.169666 ± 0.000397262 (W 0.00140853) | 0.151598 ± 0.000651485 (W 0.000237119) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.181869 ± 0.00161977 (W 0.00241958) | 0.164506 ± 0.000957505 (W 0.003) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.009678 ± 0.000144212 (W 9.15423e-05) | 0.017754 ± 9.15442e-05 (W 0.000155368) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009949 ± 0.000126808 (W 0.000158263) | 0.017942 ± 0.000103636 (W 8.20079e-05) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.035957 ± 4.20357e-05 (W 0.000242322) | 0.035406 ± 0.000421118 (W 0.000170822) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.036288 ± 0.000200067 (W 0.000613571) | 0.035276 ± 0.000121657 (W 0.000199069) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.087301 ± 0.000370192 (W 0.00153955) | 0.085369 ± 0.000149091 (W 0.00142601) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.087423 ± 0.00024597 (W 0.000609506) | 0.085589 ± 4.87545e-05 (W 0.00152321) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008466 ± 4.63717e-05 (W 5.49545e-05) | 0.007814 ± 4.07063e-05 (W 4.78142e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010269 ± 0.000112398 (W 0.000127806) | 0.009227 ± 9.86577e-05 (W 0.000112783) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034052 ± 0.000330547 (W 0.000316607) | 0.035326 ± 9.29157e-05 (W 0.00108615) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.036338 ± 0.000350839 (W 0.000295964) | 0.039313 ± 0.00022745 (W 0.000981452) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.073086 ± 0.000286414 (W 0.000262877) | 0.084265 ± 0.00105348 (W 0.00172027) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.091119 ± 0.00210939 (W 0.00203634) | 0.09802 ± 0.0011197 (W 0.00301666) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.047357 ± 0.000339898 (W 0.000104572) | 0.061074 ± 0.000204525 (W 0.000453391) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.049551 ± 0.000113539 (W 0.000201872) | 0.063807 ± 0.000338617 (W 0.00054076) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.130052 ± 0.000546875 (W 0.00049801) | 0.229938 ± 0.00097544 (W 0.00256302) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.137512 ± 0.000831548 (W 0.00036077) | 0.23908 ± 0.000522043 (W 0.00231707) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.262038 ± 0.00126471 (W 0.00247791) | 0.446131 ± 0.00454374 (W 0.00317139) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.308444 ± 0.00117004 (W 0.00129196) | 0.492034 ± 0.00321238 (W 0.00257323) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025317 ± 0.000164257 (W 0.000127414) | 0.039763 ± 0.000167933 (W 0.000717488) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.089256 ± 2.36925e-05 (W 0.000286593) | 0.173804 ± 0.000371947 (W 0.00257826) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.178308 ± 0.00139808 (W 0.00168751) | 0.34845 ± 0.00308948 (W 0.00291978) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.009829 ± 4.07063e-05 (W 5.54283e-05) | 0.009077 ± 4.50925e-05 (W 0.000403845) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.011611 ± 5.0797e-05 (W 0.000193275) | 0.010259 ± 6.16468e-05 (W 3.969e-05) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.036448 ± 0.000415437 (W 0.000181647) | 0.036808 ± 0.000184708 (W 0.00059755) | - | · | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.040365 ± 6.30106e-05 (W 0.000410563) | 0.040635 ± 0.000168226 (W 0.000652166) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.076823 ± 0.000375237 (W 0.000421428) | 0.085057 ± 0.000183358 (W 0.00169286) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.094106 ± 0.000178715 (W 0.0022559) | 0.098243 ± 0.000291023 (W 0.000492758) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.164162 ± 0.00168552 (W 0.000662933) | 0.173063 ± 0.000146584 (W 0.00184653) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.164192 ± 0.00146751 (W 0.000270687) | 0.174245 ± 0.00151224 (W 0.000594007) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.534646 ± 0.0148704 (W 0.00262382) | 0.658486 ± 0.00191011 (W 0.00232246) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.546538 ± 0.00702484 (W 0.00221898) | 0.663726 ± 0.00126887 (W 0.00222726) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.12935 ± 0.00517136 (W 0.0031118) | 1.8581 ± 0.00660074 (W 0.00499238) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.13386 ± 0.00855603 (W 0.00346477) | 1.86377 ± 0.00401603 (W 0.00264648) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.05895 ± 0.000557579 (W 0.000128569) | 0.118762 ± 0.000147215 (W 0.000514082) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059211 ± 0.000481501 (W 0.00016346) | 0.118731 ± 0.00049593 (W 0.00163553) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.222248 ± 0.000345515 (W 0.00210755) | 0.452101 ± 0.00291975 (W 0.00240033) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.22318 ± 0.00191844 (W 0.00209684) | 0.449607 ± 0.00132953 (W 0.00253933) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.532609 ± 0.00445773 (W 0.00226709) | 1.05526 ± 0.00238064 (W 0.00111793) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.533481 ± 0.00347927 (W 0.00166857) | 1.05374 ± 0.0083975 (W 0.00182235) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001353 ± 2.00083e-05 (W 2.24989e-05) | 0.001162 ± 2.08167e-05 (W 7.13043e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003076 ± 2.58908e-05 (W 1.54984e-05) | 0.002645 ± 1.73205e-05 (W 4.15933e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00509 ± 4.55741e-05 (W 4.10159e-05) | 0.004358 ± 1.52753e-05 (W 4.0289e-05) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011512 ± 3.4356e-05 (W 5.82855e-05) | 0.011381 ± 1.52753e-05 (W 5.03041e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012092 ± 4.07063e-05 (W 0.000113152) | 0.012142 ± 3.09892e-05 (W 8.97235e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.065321 ± 0.00025482 (W 0.000155978) | 0.065302 ± 0.000121577 (W 0.00021721) | - | · | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066995 ± 7.05715e-05 (W 0.000279096) | 0.066614 ± 0.000165502 (W 0.000170822) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.103801 ± 0.000372857 (W 0.000287194) | 0.109284 ± 0.00039137 (W 0.00113567) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108622 ± 0.000355012 (W 0.000338687) | 0.114333 ± 0.00058954 (W 0.00108708) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.08053 ± 0.000537029 (W 0.000286023) | 0.104214 ± 0.000664291 (W 0.00186527) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.081542 ± 0.000686975 (W 0.00175491) | 0.104935 ± 0.000153962 (W 0.000131428) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.258659 ± 0.00087347 (W 0.00208744) | 0.388103 ± 0.00195877 (W 0.00233267) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.259804 ± 0.000438726 (W 0.00160831) | 0.39205 ± 0.00147686 (W 0.00305364) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.548892 ± 0.00114292 (W 0.00355422) | 1.02838 ± 0.00395801 (W 0.00243041) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.548341 ± 0.00447374 (W 0.015239) | 1.03063 ± 0.00148205 (W 0.00284413) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014847 ± 0.000155027 (W 1.58177e-05) | 0.0106 ± 0.000116234 (W 7.17795e-05) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01632 ± 0.000119876 (W 9.95992e-05) | 0.012082 ± 8.1304e-05 (W 6.46088e-05) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.066935 ± 0.000240188 (W 9.56201e-05) | 0.052507 ± 0.000221315 (W 8.26601e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.069669 ± 0.000226974 (W 0.000213774) | 0.056294 ± 8.34286e-05 (W 0.00242895) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.175306 ± 0.000378537 (W 0.00137621) | 0.151061 ± 0.00018078 (W 0.000211616) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.190174 ± 0.00816114 (W 0.00326783) | 0.164196 ± 0.000358986 (W 0.00270375) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.007955 ± 8.77895e-05 (W 0.000137113) | 0.017763 ± 0.000121025 (W 0.000304684) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008024 ± 0.00131529 (W 0.000151337) | 0.017793 ± 8.32666e-05 (W 8.11172e-05) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.024605 ± 4.20357e-05 (W 0.000919093) | 0.036057 ± 0.000237363 (W 0.000629835) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.023735 ± 0.000100201 (W 0.00045153) | 0.035636 ± 0.000138023 (W 0.00186599) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.052849 ± 0.000541006 (W 0.00132146) | 0.095947 ± 0.00105671 (W 0.00121247) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.052628 ± 0.000474953 (W 0.000509997) | 0.096288 ± 7.02875e-05 (W 0.00166008) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008415 ± 6.16468e-05 (W 5.22427e-05) | 0.007825 ± 5.03322e-05 (W 6.80904e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010209 ± 8.96289e-05 (W 8.3666e-05) | 0.009357 ± 0.000116663 (W 3.93611e-05) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.033142 ± 9.05005e-05 (W 0.000500237) | 0.035937 ± 0.00017088 (W 0.00209462) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.037569 ± 0.00024125 (W 0.00064905) | 0.039884 ± 0.000332281 (W 0.00024404) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.074318 ± 0.00136523 (W 0.0013882) | 0.08572 ± 0.000430326 (W 0.00151603) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.093292 ± 0.000455223 (W 0.000949234) | 0.101029 ± 0.000163368 (W 0.00308522) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.047218 ± 0.000127163 (W 0.000176491) | 0.06968 ± 0.000735725 (W 0.00103274) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.049882 ± 0.000330531 (W 0.000152709) | 0.071964 ± 0.000486932 (W 0.000676022) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.124572 ± 0.000845973 (W 0.000323572) | 0.27364 ± 0.00254943 (W 0.00269417) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.132024 ± 0.000846511 (W 0.00154979) | 0.276882 ± 0.000988668 (W 0.0025167) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.237619 ± 0.00382456 (W 0.00492057) | 0.533855 ± 0.00430662 (W 0.002137) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.285792 ± 0.000678111 (W 0.00274942) | 0.579221 ± 0.00750597 (W 0.00440925) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025427 ± 0.000112694 (W 0.000118195) | 0.04831 ± 0.00113178 (W 0.00193254) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.086241 ± 0.000143841 (W 0.000337084) | 0.217836 ± 0.00309735 (W 0.00297325) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.168134 ± 0.00016405 (W 0.000551921) | 0.423759 ± 0.00284798 (W 0.00386873) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.009849 ± 8.08723e-05 (W 5.39648e-05) | 0.008817 ± 8.87149e-05 (W 4.89724e-05) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.011712 ± 3.05505e-05 (W 6.91375e-05) | 0.010279 ± 5.55908e-05 (W 3.06627e-05) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.036398 ± 0.000136504 (W 0.000246815) | 0.036467 ± 0.000471889 (W 0.000386651) | - | · | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.039463 ± 0.000448042 (W 0.000708296) | 0.040846 ± 0.000382694 (W 0.000260903) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.076833 ± 0.000321623 (W 0.000526337) | 0.086129 ± 0.0010485 (W 0.00129239) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.096289 ± 0.000212838 (W 0.00255021) | 0.101199 ± 0.00039054 (W 0.00236011) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.167022 ± 0.00187261 (W 0.000173402) | 0.172532 ± 0.000234783 (W 0.00205966) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.165138 ± 0.00034652 (W 0.0018582) | 0.175727 ± 0.000765586 (W 0.00202006) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.523745 ± 0.0200643 (W 0.00311552) | 0.658317 ± 0.00166203 (W 0.00289058) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.530919 ± 0.00153047 (W 0.0025607) | 0.658878 ± 0.00648194 (W 0.00249797) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.09392 ± 0.00203791 (W 0.00235346) | 1.87368 ± 0.00422857 (W 0.00397245) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.09903 ± 0.00323008 (W 0.00135972) | 1.86498 ± 0.00180092 (W 0.00356219) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059291 ± 0.00031286 (W 9.1627e-05) | 0.118942 ± 0.000301755 (W 0.00279591) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060002 ± 0.000653541 (W 0.000147761) | 0.119451 ± 0.000482481 (W 0.000307818) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.222404 ± 0.00169663 (W 0.00164878) | 0.450008 ± 0.00122254 (W 0.0023801) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.222651 ± 0.00134237 (W 0.00204471) | 0.450383 ± 0.0022947 (W 0.00284869) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.537711 ± 0.00277933 (W 0.00158307) | 1.04441 ± 0.000766835 (W 0.00209733) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.555597 ± 0.0179795 (W 0.0016857) | 1.05777 ± 0.00468424 (W 0.00219035) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n32/b1` | 0.001333 ± 2.0502e-05 (W 2.5499e-05) | 0.001152 ± 5.7735e-06 (W 7.51412e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003096 ± 1.0504e-05 (W 2.08638e-05) | 0.002605 ± 5.7735e-06 (W 9.39149e-05) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00501 ± 1.09697e-05 (W 5.34621e-05) | 0.004328 ± 0 (W 3.49285e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `lu/32`, `lu/64`, `lu/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `solve/32`, `solve/64`, `solve/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `triangular_solve/32`, `triangular_solve/64`, `triangular_solve/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.058027 ± 0.000260335 (W 0.000156679) | 0.297634 ± 0.0010809 (W 0.00239368) | - | 🔵 faer-fork 5.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.061064 ± 0.000370371 (W 0.000451889) | 0.070511 ± 0.00253224 (W 0.0002527) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.33506 ± 0.00137337 (W 0.00267463) | 0.829485 ± 0.00340154 (W 0.00307207) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.339663 ± 0.000669688 (W 0.00271765) | 0.389165 ± 0.002552 (W 0.0026235) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.6279 ± 0.0171688 (W 0.00734691) | 3.95781 ± 0.0158745 (W 0.00507115) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.14882 ± 0.0115859 (W 0.00628781) | 2.52996 ± 0.00904773 (W 0.0079689) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.57544 ± 0.00578771 (W 0.00894843) | 2.41138 ± 0.00564178 (W 0.00679681) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.56455 ± 0.0037283 (W 0.00608051) | 2.4084 ± 0.0206601 (W 0.00328253) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.6076 ± 0.0392546 (W 0.0107771) | 10.2001 ± 0.0465942 (W 0.0238956) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.6624 ± 0.00359676 (W 0.0153583) | 10.1907 ± 0.00394275 (W 0.0180535) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 50.7591 ± 0.158892 (W 0.104247) | 49.4993 ± 0.320856 (W 0.124138) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 50.9818 ± 0.348276 (W 0.0458201) | 49.5539 ± 0.370381 (W 0.082256) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.36496 ± 0.00185977 (W 0.00242097) | 0.317291 ± 0.00103627 (W 0.00252305) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.338145 ± 0.00100553 (W 0.00255633) | 0.31126 ± 0.00159991 (W 0.00368906) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.90904 ± 0.017098 (W 0.0107542) | 3.11117 ± 0.00955262 (W 0.00638164) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.05243 ± 0.0164129 (W 0.00797422) | 3.00435 ± 0.0135087 (W 0.00609752) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 21.6317 ± 0.0118726 (W 0.0517364) | 24.7519 ± 0.0370784 (W 0.043053) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 22.5519 ± 0.0590314 (W 0.0355974) | 22.7314 ± 0.139641 (W 0.0449206) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.259333 ± 0.0012272 (W 0.00236649) | 0.25792 ± 0.000105083 (W 0.00156246) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.258011 ± 0.00125992 (W 0.00287063) | 0.258522 ± 0.000176103 (W 0.00115002) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.90127 ± 0.00346303 (W 0.00447692) | 1.90671 ± 0.00357533 (W 0.00387823) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.90247 ± 0.00272403 (W 0.00623892) | 1.90628 ± 0.00398564 (W 0.00515973) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 16.0198 ± 0.0378852 (W 0.0450556) | 15.9501 ± 0.0435816 (W 0.0392547) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 16.0034 ± 0.0726555 (W 0.0417948) | 15.9457 ± 0.0327693 (W 0.0215802) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.104553 ± 0.000496965 (W 0.0001789) | 0.260455 ± 0.00312507 (W 0.00294126) | - | 🔵 faer-fork 2.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.123139 ± 0.000679218 (W 0.000390945) | 0.136535 ± 0.000534379 (W 0.00180608) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.569611 ± 0.00245705 (W 0.00332674) | 2.56756 ± 0.01055 (W 0.0138426) | - | 🔵 faer-fork 4.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.936795 ± 0.00300623 (W 0.00370153) | 1.07659 ± 0.00881312 (W 0.00405928) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.88944 ± 0.00832763 (W 0.0190669) | 20.378 ± 0.0190689 (W 0.0337301) | - | 🔵 faer-fork 5.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.69061 ± 0.0150421 (W 0.0124771) | 7.34597 ± 0.0343137 (W 0.0159657) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.447634 ± 0.00339293 (W 0.00233373) | 0.733477 ± 0.00432402 (W 0.00185127) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.561687 ± 0.00507447 (W 0.00420551) | 0.595341 ± 0.00295638 (W 0.00327798) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.93377 ± 0.00872076 (W 0.00881341) | 4.2523 ± 0.0124176 (W 0.00448223) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.19283 ± 0.0156458 (W 0.00309318) | 3.36442 ± 0.00741215 (W 0.0119164) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 18.6346 ± 0.0298172 (W 0.0381606) | 27.6355 ± 0.0487134 (W 0.0440727) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 20.2684 ± 0.174929 (W 0.0421612) | 21.41 ± 0.0716847 (W 0.4079) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.24666 ± 0.00104652 (W 0.0019871) | 0.533824 ± 0.00212519 (W 0.00323978) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.52304 ± 0.00632259 (W 0.00487263) | 2.88143 ± 0.0139338 (W 0.00942212) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 8.5951 ± 0.00991423 (W 0.0273908) | 17.5565 ± 0.0641864 (W 0.0341723) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.095298 ± 0.000170119 (W 0.00243809) | 0.188972 ± 0.00518441 (W 0.00495901) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.124671 ± 0.000323843 (W 0.00186347) | 0.135983 ± 0.00159306 (W 0.00228838) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.529556 ± 0.00273678 (W 0.00267967) | 1.89182 ± 0.0190338 (W 0.0101511) | - | 🔵 faer-fork 3.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.689525 ± 0.00103759 (W 0.00321997) | 0.806052 ± 0.00877541 (W 0.00282929) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.70648 ± 0.0189555 (W 0.00888477) | 13.8712 ± 0.0946358 (W 0.0273444) | - | 🔵 faer-fork 3.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.275 ± 0.00672515 (W 0.0111044) | 5.80447 ± 0.0149881 (W 0.0102148) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.87716 ± 0.000480816 (W 0.00403207) | 3.68623 ± 0.0134314 (W 0.00903142) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.89031 ± 0.00433602 (W 0.00553764) | 3.69236 ± 0.0032422 (W 0.00698256) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.5829 ± 0.0376901 (W 0.0171102) | 18.3851 ± 0.0200964 (W 0.0236794) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 14.5727 ± 0.0207435 (W 0.0212658) | 18.4462 ± 0.0148574 (W 0.0314549) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 86.6892 ± 0.0559429 (W 0.0588193) | 105.151 ± 0.107837 (W 0.150937) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 87.0869 ± 0.204089 (W 0.158914) | 105.436 ± 0.140779 (W 0.147884) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.922086 ± 0.00531239 (W 0.00228397) | 2.69683 ± 0.00759144 (W 0.00580919) | - | 🔵 faer-fork 2.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.92469 ± 0.00225928 (W 0.00258076) | 2.68911 ± 0.00923799 (W 0.00521099) | - | 🔵 faer-fork 2.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.52085 ± 0.0503619 (W 0.0115157) | 12.425 ± 0.0198014 (W 0.0131165) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.53567 ± 0.0160709 (W 0.0109697) | 12.437 ± 0.00821689 (W 0.0163168) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 37.2799 ± 0.0933736 (W 0.0636813) | 66.0679 ± 0.289068 (W 0.0587316) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 37.1679 ± 0.028886 (W 0.0387509) | 66.3468 ± 0.155434 (W 0.0740146) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004298 ± 1.52753e-05 (W 4.13799e-05) | 0.004088 ± 0.00023094 (W 7.43727e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.012864 ± 0.000180103 (W 0.000189394) | 0.012063 ± 0.000193236 (W 0.000153184) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.044773 ± 0.00117599 (W 0.000445717) | 0.042619 ± 0.000356804 (W 0.000373745) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.049742 ± 0.00104366 (W 0.000237803) | 0.206485 ± 0.000577701 (W 0.0698738) | - | 🔵 faer-fork 4.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.057156 ± 0.00115433 (W 0.000857864) | 0.081011 ± 0.00164641 (W 0.00175288) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.180165 ± 0.0017953 (W 0.00256534) | 0.722277 ± 0.0526773 (W 0.0611577) | - | 🔵 faer-fork 4.01× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.221262 ± 0.000424844 (W 0.00266575) | 0.379234 ± 0.00901837 (W 0.00886916) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.01542 ± 0.00702909 (W 0.00208135) | 2.26876 ± 0.012815 (W 0.102874) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.967748 ± 0.00487272 (W 0.00219822) | 1.61118 ± 0.0340646 (W 0.0382138) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.4528 ± 0.00365096 (W 0.00535768) | 2.33319 ± 0.000843561 (W 0.0040523) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.45981 ± 0.00484941 (W 0.00280892) | 2.34371 ± 0.00670484 (W 0.00251233) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.68938 ± 0.00746772 (W 0.0108878) | 6.95066 ± 0.0104117 (W 0.00912711) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.69339 ± 0.0171313 (W 0.00687006) | 7.01546 ± 0.0296474 (W 0.0157666) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.8865 ± 0.0122869 (W 0.0198654) | 25.1005 ± 0.0369338 (W 0.0225282) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.604 ± 0.0547767 (W 0.283301) | 25.1172 ± 0.0452077 (W 0.0248189) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.33286 ± 0.00186548 (W 0.00415377) | 0.314445 ± 6.38148e-05 (W 0.00232321) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.379797 ± 0.00326742 (W 0.00517718) | 0.311248 ± 0.0015742 (W 0.00373893) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.9994 ± 0.00564469 (W 0.00611974) | 3.11446 ± 0.00679174 (W 0.0101561) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.26457 ± 0.00201424 (W 0.00808187) | 2.77935 ± 0.0154216 (W 0.00940237) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.4762 ± 0.0196169 (W 0.0221093) | 17.6687 ± 0.00456822 (W 0.0238989) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.0766 ± 0.0328862 (W 0.025405) | 15.3518 ± 0.0108506 (W 0.010496) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.084217 ± 0.000381817 (W 0.00071801) | 0.081412 ± 0.000430892 (W 0.00129752) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.084277 ± 0.000437951 (W 0.000549114) | 0.080911 ± 0.00040202 (W 0.000550775) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.517122 ± 0.00213325 (W 0.00201003) | 0.517192 ± 0.00312839 (W 0.00365059) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.518055 ± 0.00112333 (W 0.0041744) | 0.516119 ± 0.00120471 (W 0.00315576) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.16465 ± 0.00781333 (W 0.00697483) | 4.09801 ± 0.0228058 (W 0.0102546) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.15819 ± 0.000299176 (W 0.0112359) | 4.10683 ± 0.0147589 (W 0.00889107) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.119373 ± 0.00219301 (W 0.00214247) | 0.276325 ± 0.00266257 (W 0.004491) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.153145 ± 0.00180074 (W 0.00216689) | 0.151872 ± 0.00105376 (W 0.00376319) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.406077 ± 0.0020453 (W 0.00367622) | 2.31695 ± 0.0113793 (W 0.0555166) | - | 🔵 faer-fork 5.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.702555 ± 0.00377772 (W 0.00503018) | 0.777219 ± 0.0133224 (W 0.0181044) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.78791 ± 0.00587002 (W 0.0226992) | 13.2132 ± 0.456309 (W 0.608356) | - | 🔵 faer-fork 7.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 3.52475 ± 0.0138138 (W 0.0134818) | 4.08513 ± 0.0336117 (W 0.0860247) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.343299 ± 0.000811273 (W 0.00241882) | 0.844472 ± 0.00430274 (W 0.00565022) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.428146 ± 0.00391908 (W 0.00331373) | 0.648926 ± 0.00422448 (W 0.00375838) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.05741 ± 0.0100972 (W 0.00378517) | 3.48795 ± 0.00843003 (W 0.0106555) | - | 🔵 faer-fork 3.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.5723 ± 0.0105787 (W 0.00578523) | 2.31918 ± 0.0273612 (W 0.00966105) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.88479 ± 0.00984965 (W 0.0124797) | 19.2549 ± 0.103716 (W 0.0596889) | - | 🔵 faer-fork 3.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.50609 ± 0.111051 (W 0.0842041) | 9.77938 ± 0.029478 (W 0.0204786) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.222905 ± 0.000448144 (W 0.00188995) | 0.727191 ± 0.000570113 (W 0.00318205) | - | 🔵 faer-fork 3.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.666679 ± 0.00200261 (W 0.00277907) | 3.09885 ± 0.00159436 (W 0.0125374) | - | 🔵 faer-fork 4.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 3.21484 ± 0.00220756 (W 0.00858789) | 16.6646 ± 0.117089 (W 0.0625086) | - | 🔵 faer-fork 5.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.094487 ± 0.000543435 (W 0.00132829) | 0.219958 ± 0.00314199 (W 0.00176537) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.116206 ± 0.000458186 (W 0.00190703) | 0.151031 ± 0.00287762 (W 0.00406841) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.38606 ± 0.00300211 (W 0.00394916) | 1.68556 ± 0.0219256 (W 0.0439718) | - | 🔵 faer-fork 4.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.715883 ± 0.00410604 (W 0.00383799) | 0.716474 ± 0.0024888 (W 0.00679525) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.69754 ± 0.0112163 (W 0.0059319) | 9.643 ± 0.0444264 (W 0.283717) | - | 🔵 faer-fork 5.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.43844 ± 0.012537 (W 0.00852128) | 3.71994 ± 0.0219124 (W 0.0739771) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.00441 ± 0.00477751 (W 0.00398824) | 2.77361 ± 0.00354695 (W 0.0115339) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.00503 ± 0.00548025 (W 0.0047465) | 2.82723 ± 0.0118909 (W 0.0142185) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.14713 ± 0.0173217 (W 0.0144891) | 13.1584 ± 0.0876419 (W 0.0281909) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.16323 ± 0.0171945 (W 0.0109067) | 13.1545 ± 0.0756439 (W 0.0461765) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 36.6567 ± 0.138502 (W 0.0331057) | 55.4114 ± 0.0538588 (W 0.0480407) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 36.8385 ± 0.114442 (W 0.0345214) | 55.7752 ± 0.173493 (W 0.549444) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.962902 ± 0.00390494 (W 0.00287384) | 2.17815 ± 0.00720018 (W 0.00713709) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.960145 ± 0.000512682 (W 0.00206722) | 2.19168 ± 0.00597473 (W 0.00290826) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.15055 ± 0.0206033 (W 0.00452539) | 10.4532 ± 0.021229 (W 0.0111992) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.15373 ± 0.0228848 (W 0.00865462) | 10.481 ± 0.0123796 (W 0.0150032) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.4721 ± 0.0580862 (W 0.0262264) | 41.9082 ± 0.0744622 (W 0.0405273) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.491 ± 0.097598 (W 0.0108668) | 41.9896 ± 0.0943713 (W 0.0458784) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005139 ± 9.50438e-06 (W 4.46352e-05) | 0.004097 ± 5.03322e-05 (W 6.50331e-05) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013194 ± 0.000133742 (W 0.000108394) | 0.012593 ± 0.000121533 (W 0.000109271) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.03226 ± 0.000107472 (W 0.000228631) | 0.03184 ± 0.000296717 (W 0.000316426) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.048199 ± 0.00111805 (W 0.000293744) | 0.108392 ± 0.00209424 (W 0.00194034) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.057828 ± 0.000687101 (W 0.000818098) | 0.070541 ± 0.00196737 (W 0.000526425) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.166089 ± 0.00219796 (W 0.00262155) | 0.57494 ± 0.00630807 (W 0.0462199) | - | 🔵 faer-fork 3.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.207386 ± 0.00126776 (W 0.00192802) | 0.37641 ± 0.0858792 (W 0.0212103) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.786415 ± 0.00628793 (W 0.0027368) | 1.97723 ± 0.0665755 (W 0.0453813) | - | 🔵 faer-fork 2.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.796473 ± 0.0060332 (W 0.00281389) | 1.46942 ± 0.019693 (W 0.0211193) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.953066 ± 0.00238179 (W 0.00198081) | 2.34013 ± 0.00338149 (W 0.00396722) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.959916 ± 0.00211009 (W 0.00291763) | 2.34391 ± 0.00189693 (W 0.00510158) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 4.0038 ± 0.0145013 (W 0.00672954) | 6.97354 ± 0.0172032 (W 0.0328052) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.99797 ± 0.00689692 (W 0.0100613) | 6.94285 ± 0.0370681 (W 0.0137575) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.5881 ± 0.0318064 (W 0.0176259) | 23.6149 ± 0.0148121 (W 0.0613693) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.5282 ± 0.0160541 (W 0.01755) | 23.9465 ± 0.0410812 (W 0.215279) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.332049 ± 0.00180162 (W 0.00400631) | 0.315177 ± 0.000654279 (W 0.00321322) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.345282 ± 0.00439022 (W 0.00269097) | 0.315097 ± 0.00106768 (W 0.00335435) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.88883 ± 0.00634858 (W 0.00537485) | 3.13043 ± 0.0136811 (W 0.00589857) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.1846 ± 0.0170642 (W 0.0114469) | 2.7815 ± 0.00630038 (W 0.00512091) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.1503 ± 0.0107155 (W 0.0157066) | 16.6551 ± 0.0405944 (W 0.0213307) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.7773 ± 0.0320684 (W 0.0232974) | 14.5978 ± 0.13308 (W 0.0166726) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.058789 ± 0.000867258 (W 0.000726204) | 0.099746 ± 0.000425508 (W 0.001257) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.05911 ± 0.000168622 (W 0.000785429) | 0.099425 ± 0.000605498 (W 0.00157627) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.283818 ± 0.00169175 (W 0.00316459) | 0.40271 ± 0.00569805 (W 0.014738) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.285502 ± 0.00131284 (W 0.00382081) | 0.393973 ± 0.0120278 (W 0.0116874) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.24639 ± 0.0134585 (W 0.016829) | 2.80123 ± 0.289039 (W 0.341464) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.26254 ± 0.011725 (W 0.0111803) | 2.81693 ± 0.356151 (W 0.341602) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.123089 ± 0.000266896 (W 0.00132012) | 0.272608 ± 0.00267416 (W 0.00253872) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.111748 ± 0.000948202 (W 0.00194481) | 0.149659 ± 0.000397895 (W 0.00270275) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.341757 ± 0.0027054 (W 0.00324845) | 2.2565 ± 0.0128079 (W 0.039035) | - | 🔵 faer-fork 6.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.672662 ± 0.00517835 (W 0.00473952) | 0.802745 ± 0.00871858 (W 0.0238581) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.31228 ± 0.0121626 (W 0.014058) | 11.849 ± 0.0365417 (W 0.185352) | - | 🔵 faer-fork 9.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.13198 ± 0.0225364 (W 0.014793) | 3.81666 ± 0.0341558 (W 0.0706279) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.326077 ± 0.000694908 (W 0.00245737) | 0.970778 ± 0.00900438 (W 0.00322279) | - | 🔵 faer-fork 2.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.409873 ± 0.0164998 (W 0.00338881) | 0.774462 ± 0.00448342 (W 0.0039653) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 0.864468 ± 0.00171506 (W 0.00221348) | 3.76191 ± 0.00984544 (W 0.0144498) | - | 🔵 faer-fork 4.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.40013 ± 0.00075003 (W 0.00664813) | 2.26911 ± 0.015667 (W 0.0121424) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.46733 ± 0.00992416 (W 0.0177712) | 19.2727 ± 0.0402102 (W 0.11149) | - | 🔵 faer-fork 5.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 6.16759 ± 0.0316131 (W 0.0204432) | 9.18844 ± 0.998738 (W 0.637366) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.207547 ± 0.00105865 (W 0.0019352) | 0.823643 ± 0.00484535 (W 0.00544759) | - | 🔵 faer-fork 3.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.62318 ± 0.000537456 (W 0.00284429) | 3.32557 ± 0.0131984 (W 0.0177083) | - | 🔵 faer-fork 5.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.91857 ± 0.00737788 (W 0.00328037) | 16.9223 ± 0.0288536 (W 0.0619418) | - | 🔵 faer-fork 8.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.098323 ± 0.000603856 (W 0.000659717) | 0.208028 ± 0.000904595 (W 0.0015837) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.113682 ± 0.00049221 (W 0.000689734) | 0.142716 ± 0.000726046 (W 0.00239807) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.328812 ± 0.00284941 (W 0.00211789) | 1.7087 ± 0.00907415 (W 0.0368313) | - | 🔵 faer-fork 5.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.675127 ± 0.0035276 (W 0.00402802) | 0.694363 ± 0.00853755 (W 0.0202883) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.26053 ± 0.00707092 (W 0.00244604) | 9.0445 ± 0.0406898 (W 0.0951874) | - | 🔵 faer-fork 7.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.11317 ± 0.046518 (W 0.00799079) | 3.56212 ± 0.0121557 (W 0.0386419) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.98688 ± 0.00587748 (W 0.00471831) | 2.79602 ± 0.00155172 (W 0.00480363) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99516 ± 0.00592081 (W 0.0101585) | 2.8093 ± 0.00519939 (W 0.00508937) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.43656 ± 0.0258499 (W 0.0154251) | 13.1517 ± 0.0724294 (W 0.0158737) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.47404 ± 0.010987 (W 0.0180041) | 13.1542 ± 0.0655579 (W 0.0381118) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.7013 ± 0.0897316 (W 0.0379539) | 52.5382 ± 0.149193 (W 0.096798) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.6408 ± 0.0229195 (W 0.0478932) | 52.4688 ± 0.0580976 (W 0.243211) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.00872 ± 0.00722297 (W 0.00247825) | 2.16491 ± 0.00290796 (W 0.00458636) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.00344 ± 0.010244 (W 0.00216565) | 2.173 ± 0.00272588 (W 0.00310579) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.03912 ± 0.0091759 (W 0.00720827) | 10.5043 ± 0.00218341 (W 0.0224236) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.03818 ± 0.0139668 (W 0.0074613) | 10.4646 ± 0.0703841 (W 0.0176039) | - | 🔵 faer-fork 2.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.5393 ± 0.234945 (W 0.0277543) | 40.1571 ± 0.11052 (W 0.0685051) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.5329 ± 0.0514084 (W 0.0326598) | 40.1224 ± 0.0782805 (W 0.0433432) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00523 ± 1.47309e-05 (W 2.79285e-05) | 0.004087 ± 3.66106e-05 (W 4.9295e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.012814 ± 0.000160032 (W 0.000135975) | 0.012603 ± 9.96009e-05 (W 0.00192866) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.029956 ± 0.000234592 (W 0.000160095) | 0.030066 ± 0.000266246 (W 0.000341717) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.047889 ± 0.00062806 (W 0.000178847) | 0.058278 ± 0.00337001 (W 0.000110513) | 0.085879 ± 0.000788414 (W 0.00018365) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.79× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.055022 ± 0.000488669 (W 0.000153455) | 0.063458 ± 0.00335175 (W 0.00019397) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.278816 ± 0.00249155 (W 0.00259082) | 0.303535 ± 0.00357961 (W 0.00383323) | 0.417082 ± 0.00197634 (W 0.00412854) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.309046 ± 0.00162249 (W 0.00225077) | 0.323082 ± 0.00569629 (W 0.00273845) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.89764 ± 0.0113342 (W 0.0122172) | 2.01889 ± 0.0051598 (W 0.00743024) | 3.16588 ± 0.0131518 (W 0.0200559) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.03714 ± 0.0191964 (W 0.00637284) | 2.06475 ± 0.00904786 (W 0.00617914) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.57217 ± 0.00653107 (W 0.00617736) | 2.41209 ± 0.00567422 (W 0.00306864) | 1.80964 ± 0.00107704 (W 0.00670024) | 🔴 faer-fork 1.07× slower than `faer-upstream` | 🔴 faer-fork 1.42× slower than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 2.56015 ± 0.00525913 (W 0.00462553) | 2.40488 ± 0.00461169 (W 0.00457983) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.5933 ± 0.0399408 (W 0.0110915) | 10.1364 ± 0.0304889 (W 0.012047) | 9.8377 ± 0.0134975 (W 0.0132387) | · | 🔴 faer-fork 1.08× slower than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 10.6097 ± 0.0198181 (W 0.00865338) | 10.1088 ± 0.0260569 (W 0.0167609) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 50.4299 ± 0.412685 (W 0.0449848) | 49.286 ± 0.242276 (W 0.355356) | 60.3115 ± 0.109192 (W 0.0915199) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 50.6047 ± 0.0943786 (W 0.0350081) | 48.9813 ± 0.371271 (W 0.0343054) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.329243 ± 0.0193419 (W 0.0027366) | 0.309066 ± 0.00256087 (W 0.00243725) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.317482 ± 0.00164769 (W 0.00212725) | 0.287796 ± 0.00132113 (W 0.0030984) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.88868 ± 0.0133586 (W 0.00694726) | 2.83685 ± 0.00568064 (W 0.00824374) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.97729 ± 0.156295 (W 0.0113881) | 2.84888 ± 0.0150426 (W 0.00656557) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1` | 22.5462 ± 0.0374353 (W 0.0405678) | 22.1299 ± 0.0823313 (W 0.030846) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 22.1929 ± 0.107075 (W 0.0704165) | 21.6103 ± 0.0279579 (W 0.0395601) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.256107 ± 0.000597631 (W 0.00240023) | 0.254664 ± 0.0009735 (W 0.0018228) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.255837 ± 0.0014765 (W 0.00244192) | 0.255784 ± 0.000493076 (W 0.0030154) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.87842 ± 0.00166074 (W 0.00598843) | 1.88024 ± 0.00235899 (W 0.00682886) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.87805 ± 0.00104843 (W 0.00401823) | 1.88017 ± 0.00432513 (W 0.00645601) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9597 ± 0.0653016 (W 0.216962) | 15.9597 ± 0.065215 (W 0.0266519) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.8832 ± 0.0197268 (W 0.0387847) | 15.886 ± 0.0902051 (W 0.0332506) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.097241 ± 0.000507439 (W 0.001879) | 0.104815 ± 0.00126384 (W 0.000330159) | 0.141584 ± 0.000200734 (W 0.00200124) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.104684 ± 0.000141799 (W 0.00181744) | 0.114111 ± 0.0012625 (W 0.000477563) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.572978 ± 0.00121656 (W 0.00363696) | 0.687662 ± 0.00436153 (W 0.00419608) | 0.79411 ± 0.00191783 (W 0.00162834) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.859058 ± 0.0033157 (W 0.00404976) | 0.94084 ± 0.00366272 (W 0.00368136) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.74673 ± 0.0106611 (W 0.00598357) | 5.04172 ± 0.0101574 (W 0.00950108) | 5.06506 ± 0.0131208 (W 0.0233596) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 5.15913 ± 0.0242499 (W 0.0350346) | 6.28339 ± 0.112618 (W 0.0294546) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.453065 ± 0.003091 (W 0.0023589) | 0.487999 ± 0.000896246 (W 0.00333548) | 0.872554 ± 0.00282625 (W 0.00381174) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.93× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.535216 ± 0.00036464 (W 0.00249487) | 0.572858 ± 0.00337605 (W 0.00284664) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.64848 ± 0.0145264 (W 0.0103465) | 2.74785 ± 0.0139935 (W 0.0064465) | 5.21479 ± 0.0172119 (W 0.00913802) | · | 🔵 faer-fork 1.97× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.09227 ± 0.0129005 (W 0.00489639) | 3.20673 ± 0.0130394 (W 0.00413514) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 17.75 ± 0.160796 (W 0.380333) | 18.1169 ± 0.157873 (W 0.0189183) | 35.414 ± 0.382855 (W 0.0500967) | · | 🔵 faer-fork 2.00× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 19.9436 ± 0.210807 (W 0.197521) | 20.1629 ± 0.0965553 (W 0.0243033) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.256695 ± 0.00282484 (W 0.00243104) | 0.285492 ± 0.00227939 (W 0.00224355) | 0.326749 ± 0.00248249 (W 0.00238905) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 1.28261 ± 0.00438701 (W 0.00276812) | 1.39134 ± 0.00646916 (W 0.010595) | 1.91196 ± 0.00453871 (W 0.00487157) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 7.75763 ± 0.0267714 (W 0.0137192) | 8.06467 ± 0.0349997 (W 0.0106379) | 13.4381 ± 0.048026 (W 0.017612) | · | 🔵 faer-fork 1.73× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.095557 ± 0.000626 (W 0.000192352) | 0.100848 ± 0.000485365 (W 0.00217103) | 0.13991 ± 0.000532501 (W 0.00166492) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.102602 ± 0.000329075 (W 0.00191651) | 0.111718 ± 0.000523377 (W 0.00216566) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.560955 ± 0.00173755 (W 0.00276664) | 0.662996 ± 0.0038604 (W 0.00217058) | 0.790903 ± 0.00269336 (W 0.00174274) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 1.41× faster than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 0.605328 ± 0.00263851 (W 0.00337128) | 0.663849 ± 0.000834811 (W 0.00432713) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.69849 ± 0.0146436 (W 0.0112348) | 4.85079 ± 0.0237884 (W 0.0134522) | 5.1197 ± 0.0227135 (W 0.0119068) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 3.87209 ± 0.0118018 (W 0.0100511) | 4.79046 ± 0.0198862 (W 0.0125489) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.8631 ± 0.00344631 (W 0.00601056) | 3.6752 ± 0.00592073 (W 0.00497475) | 3.21321 ± 0.00269601 (W 0.00908406) | 🔵 faer-fork 1.28× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.87278 ± 0.00557114 (W 0.00340203) | 3.66852 ± 0.00517593 (W 0.00463082) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.5062 ± 0.0169092 (W 0.0288028) | 18.3069 ± 0.0714026 (W 0.0165232) | 17.418 ± 0.072628 (W 0.020533) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 14.5173 ± 0.00982123 (W 0.0403125) | 18.3805 ± 0.0823332 (W 0.0374888) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 86.0524 ± 0.115876 (W 0.117881) | 104.514 ± 0.0732588 (W 0.18837) | 115.079 ± 0.416356 (W 0.3517) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 86.1417 ± 0.111749 (W 0.0627476) | 104.677 ± 0.0678931 (W 0.193645) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.905919 ± 0.00269459 (W 0.00213994) | 2.61481 ± 0.00983449 (W 0.00654028) | 1.11857 ± 0.00219372 (W 0.00312125) | 🔵 faer-fork 2.89× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.904804 ± 0.00134539 (W 0.00167998) | 2.61376 ± 0.00562165 (W 0.00333264) | - | 🔵 faer-fork 2.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.35779 ± 0.0171607 (W 0.00637211) | 12.2573 ± 0.0450515 (W 0.00604575) | 6.72194 ± 0.00831594 (W 0.00865119) | 🔵 faer-fork 2.29× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.3652 ± 0.013088 (W 0.0107894) | 12.2671 ± 0.0281801 (W 0.0333612) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 36.2984 ± 0.0251407 (W 0.0367208) | 65.0789 ± 0.221401 (W 0.106352) | 48.1937 ± 0.110099 (W 0.0383235) | 🔵 faer-fork 1.79× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 36.3913 ± 0.0659089 (W 0.0988662) | 65.5303 ± 0.0375276 (W 0.0888349) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004609 ± 7.40968e-05 (W 7.09458e-05) | 0.004439 ± 2.85832e-05 (W 2.56379e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.014577 ± 7e-05 (W 9.84845e-05) | 0.014036 ± 4e-05 (W 5.77062e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.047148 ± 0.000345876 (W 0.000275927) | 0.047387 ± 0.000819172 (W 0.000307203) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.04306 ± 0.000477001 (W 0.000747074) | 0.065322 ± 0.00177108 (W 0.00302585) | 0.088695 ± 0.000661145 (W 0.000494097) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 2.06× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.051045 ± 0.000482951 (W 0.000632295) | 0.06968 ± 0.00477946 (W 0.00183763) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.153887 ± 0.00277895 (W 0.0016327) | 0.281964 ± 0.00843966 (W 0.0115601) | 0.25796 ± 0.00630842 (W 0.0129794) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.68× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.189203 ± 0.00132102 (W 0.00123076) | 0.306572 ± 0.00944647 (W 0.00687764) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.68645 ± 0.00144949 (W 0.00205315) | 0.985532 ± 0.0194018 (W 0.0546938) | 1.42184 ± 0.00737922 (W 0.0366476) | 🔵 faer-fork 1.44× faster than `faer-upstream` | 🔵 faer-fork 2.07× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.837351 ± 0.00722784 (W 0.00348048) | 1.12623 ± 0.00431884 (W 0.0335689) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.44245 ± 0.0058485 (W 0.00297459) | 2.32347 ± 0.00376175 (W 0.00717014) | 1.04394 ± 0.0024627 (W 0.00321313) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.44881 ± 0.00368372 (W 0.0058934) | 2.33761 ± 0.00254488 (W 0.00379188) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.64925 ± 0.00647571 (W 0.00537456) | 6.91582 ± 0.0161539 (W 0.0106486) | 4.67359 ± 0.0116893 (W 0.0200454) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔴 faer-fork 1.21× slower than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.6431 ± 0.0118809 (W 0.0100439) | 6.94377 ± 0.0405319 (W 0.0103034) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.7684 ± 0.0694653 (W 0.0213253) | 24.6692 ± 0.0337214 (W 0.0252903) | 21.2376 ± 0.0416022 (W 0.0347067) | · | 🔴 faer-fork 1.17× slower than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 24.4758 ± 0.033132 (W 0.0212876) | 24.7613 ± 0.0358462 (W 0.0360316) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.323171 ± 0.00256975 (W 0.00373184) | 0.307112 ± 0.000929759 (W 0.00259029) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.361962 ± 0.00327042 (W 0.00459776) | 0.288837 ± 0.00186674 (W 0.00262695) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.96395 ± 0.0260565 (W 0.0106972) | 2.81352 ± 0.00777745 (W 0.00860342) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.17933 ± 0.00956612 (W 0.00727018) | 2.6277 ± 0.0246221 (W 0.00937688) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.6783 ± 0.0102927 (W 0.0175949) | 15.1214 ± 0.0198413 (W 0.0266545) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.6279 ± 0.013805 (W 0.0186597) | 14.2541 ± 0.023371 (W 0.0240876) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.084517 ± 0.000377361 (W 0.00131816) | 0.082073 ± 0.000356573 (W 0.000520752) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.084327 ± 0.000722144 (W 0.00219708) | 0.081793 ± 0.000673605 (W 0.00132639) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.524737 ± 0.00157383 (W 0.00443246) | 0.5216 ± 0.00133599 (W 0.00386635) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.526069 ± 0.00132403 (W 0.00329922) | 0.524485 ± 0.00346866 (W 0.00282407) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.17831 ± 0.0154897 (W 0.00653157) | 4.18704 ± 0.00404196 (W 0.0114662) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.16752 ± 0.0294192 (W 0.00901335) | 4.18498 ± 0.0200324 (W 0.010509) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.079768 ± 0.000136737 (W 0.00112721) | 0.118319 ± 0.00267405 (W 0.00437502) | 0.141513 ± 0.000423935 (W 0.000447355) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.77× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.133579 ± 0.000930399 (W 0.00146688) | 0.130353 ± 0.00182941 (W 0.00320267) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.33301 ± 0.00408662 (W 0.00244739) | 0.612648 ± 0.00326132 (W 0.0144943) | 0.297815 ± 0.00160265 (W 0.00224181) | 🔵 faer-fork 1.84× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.616089 ± 0.00507496 (W 0.00543251) | 0.64488 ± 0.00317297 (W 0.00851297) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.56739 ± 0.00521933 (W 0.00541394) | 2.97909 ± 0.0363004 (W 0.0755163) | 1.50359 ± 0.00157185 (W 0.00620092) | 🔵 faer-fork 1.90× faster than `faer-upstream` | · |  |
| `lu/n512/b1 [api-inclusive]` | 3.01931 ± 0.00283905 (W 0.0050821) | 3.09535 ± 0.0709297 (W 0.083495) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.304676 ± 0.00155975 (W 0.00227897) | 0.532349 ± 0.00150297 (W 0.00260273) | 0.32847 ± 0.00240879 (W 0.00297531) | 🔵 faer-fork 1.75× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.398441 ± 0.00121602 (W 0.00275366) | 0.628198 ± 0.00110107 (W 0.00351187) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.98312 ± 0.001653 (W 0.00267416) | 1.70894 ± 0.00530833 (W 0.00364434) | 1.55977 ± 0.00273876 (W 0.00322733) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.46322 ± 0.00099527 (W 0.00734731) | 2.16844 ± 0.00769017 (W 0.00824915) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.58119 ± 0.0164782 (W 0.0125108) | 7.08226 ± 0.0200058 (W 0.0147299) | 9.44818 ± 0.121047 (W 0.0254287) | 🔵 faer-fork 1.27× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 7.7911 ± 0.0248671 (W 0.0110713) | 8.68974 ± 0.0371944 (W 0.0568487) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.203668 ± 0.0010291 (W 0.00212886) | 0.424771 ± 0.0024689 (W 0.00332678) | 0.186528 ± 0.00193141 (W 0.00180971) | 🔵 faer-fork 2.09× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 0.585728 ± 0.00344351 (W 0.00288204) | 1.32926 ± 0.0047041 (W 0.00580408) | 0.604526 ± 0.00326736 (W 0.00403366) | 🔵 faer-fork 2.27× faster than `faer-upstream` | · |  |
| `qr_factor/n512/b1` | 2.94681 ± 0.00241772 (W 0.003262) | 4.45087 ± 0.0120635 (W 0.0177577) | 3.56733 ± 0.0102501 (W 0.00623263) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.084427 ± 0.000260064 (W 0.000942187) | 0.11272 ± 0.00113176 (W 0.0048812) | 0.140291 ± 0.000161948 (W 0.00208115) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.094636 ± 0.0010812 (W 0.000988356) | 0.127517 ± 0.00150278 (W 0.00245015) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.316138 ± 0.00304061 (W 0.0039113) | 0.53756 ± 0.0119006 (W 0.013178) | 0.278627 ± 0.00231455 (W 0.00270132) | 🔵 faer-fork 1.70× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 0.629401 ± 0.00299098 (W 0.0023735) | 0.582664 ± 0.00269317 (W 0.00795599) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.43219 ± 0.00550437 (W 0.00393989) | 2.69945 ± 0.0447351 (W 0.0822439) | 1.46996 ± 0.00277666 (W 0.0058292) | 🔵 faer-fork 1.88× faster than `faer-upstream` | · |  |
| `solve/n512/b1 [api-inclusive]` | 2.96504 ± 0.0234639 (W 0.0121976) | 2.76643 ± 0.0355787 (W 0.0496565) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.99226 ± 0.00444012 (W 0.00406222) | 2.76427 ± 0.00704777 (W 0.0169292) | 2.49022 ± 0.00320165 (W 0.0101812) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99963 ± 0.0139926 (W 0.00376494) | 2.80025 ± 0.00567274 (W 0.0157815) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.02709 ± 0.0121318 (W 0.00907344) | 13.0389 ± 0.0479357 (W 0.0155945) | 12.0799 ± 0.0290035 (W 0.0185656) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 8.03976 ± 0.0218358 (W 0.0137916) | 13.0736 ± 0.0653376 (W 0.0282872) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 35.8144 ± 0.0556376 (W 0.0439782) | 54.6091 ± 0.068522 (W 0.0648954) | 57.8326 ± 0.822352 (W 0.41973) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 1.61× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 35.8885 ± 0.0098317 (W 0.327663) | 54.7833 ± 0.460885 (W 0.571288) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.944989 ± 0.00192241 (W 0.0015362) | 2.11401 ± 0.00368756 (W 0.00392688) | 1.09416 ± 0.00156179 (W 0.00201623) | 🔵 faer-fork 2.24× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.94527 ± 0.0021145 (W 0.00113307) | 2.11933 ± 0.00253815 (W 0.00555942) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.05253 ± 0.0205036 (W 0.0159872) | 10.3738 ± 0.068991 (W 0.019731) | 6.03992 ± 0.00808242 (W 0.00846167) | 🔵 faer-fork 2.56× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 4.0357 ± 0.0179895 (W 0.00626252) | 10.292 ± 0.0263492 (W 0.0129678) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.4735 ± 0.0275188 (W 0.0201587) | 40.9511 ± 0.145606 (W 0.032147) | 25.7351 ± 0.233154 (W 0.0381082) | 🔵 faer-fork 2.34× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 17.562 ± 0.0782577 (W 0.0178934) | 41.0911 ± 0.0556803 (W 0.0492062) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00551 ± 8.88819e-05 (W 7.64349e-05) | 0.004468 ± 1.53948e-05 (W 4.0141e-05) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015629 ± 0.000157703 (W 0.000260312) | 0.013876 ± 5.19615e-05 (W 6.04086e-05) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.039533 ± 2.00083e-05 (W 0.000249492) | 0.037941 ± 0.000376816 (W 0.000521705) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.04272 ± 8.3668e-05 (W 0.000741874) | 0.060222 ± 0.00219448 (W 0.00105539) | 0.086932 ± 0.000409065 (W 0.000229626) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 2.03× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.050975 ± 0.00150522 (W 0.000794894) | 0.063599 ± 0.00337267 (W 0.000543707) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.132467 ± 0.00113576 (W 0.000583583) | 0.260384 ± 0.00426715 (W 0.0188445) | 0.170758 ± 0.011111 (W 0.0060143) | 🔵 faer-fork 1.97× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.173824 ± 0.000756255 (W 0.00289493) | 0.28443 ± 0.00717792 (W 0.0256155) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.486185 ± 0.0021552 (W 0.00376444) | 0.931564 ± 0.0151278 (W 0.0233573) | 0.794169 ± 0.0251811 (W 0.0160511) | 🔵 faer-fork 1.92× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.67651 ± 0.00160347 (W 0.00284737) | 1.02928 ± 0.0199022 (W 0.0224212) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.948046 ± 0.00175661 (W 0.00291287) | 2.33484 ± 0.0103188 (W 0.00446477) | 0.991978 ± 0.00182183 (W 0.00274354) | 🔵 faer-fork 2.46× faster than `faer-upstream` | · |  |
| `eigh/n128/b1 [api-inclusive]` | 0.95557 ± 0.00283055 (W 0.00279261) | 2.33754 ± 0.00445579 (W 0.00360434) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.9583 ± 0.009018 (W 0.00689429) | 6.90421 ± 0.0145158 (W 0.0162611) | 3.84438 ± 0.00658758 (W 0.0133685) | 🔵 faer-fork 1.74× faster than `faer-upstream` | · |  |
| `eigh/n256/b1 [api-inclusive]` | 3.94936 ± 0.00847627 (W 0.00670799) | 6.90434 ± 0.0371495 (W 0.0277281) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.4377 ± 0.0397829 (W 0.0249837) | 23.3115 ± 0.0563939 (W 0.0676911) | 14.4541 ± 0.033047 (W 0.0386803) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 16.3786 ± 0.070562 (W 0.0351324) | 23.3713 ± 0.0917498 (W 0.129959) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.325766 ± 0.00472003 (W 0.00370047) | 0.307703 ± 0.00119617 (W 0.00298634) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.331177 ± 0.00254931 (W 0.00331074) | 0.292565 ± 0.000966076 (W 0.00226695) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.86274 ± 0.0132012 (W 0.00620938) | 2.82557 ± 0.00739138 (W 0.00773471) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.0943 ± 0.0147276 (W 0.00688982) | 2.64156 ± 0.00738795 (W 0.00631636) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.1312 ± 0.0709098 (W 0.0247363) | 14.0623 ± 0.0627946 (W 0.0249083) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.3753 ± 0.0455997 (W 0.0277295) | 13.4237 ± 0.0906091 (W 0.0230344) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.056044 ± 0.000465889 (W 0.000668473) | 0.099235 ± 0.001044 (W 0.00145961) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.056094 ± 0.000647488 (W 0.000302984) | 0.099996 ± 0.000698689 (W 0.00114336) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.291884 ± 0.00246297 (W 0.00144192) | 0.405676 ± 0.0105086 (W 0.0133027) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.292084 ± 0.00284267 (W 0.00287126) | 0.408351 ± 0.00103932 (W 0.0117742) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.2738 ± 0.0041558 (W 0.00582001) | 2.97692 ± 0.013312 (W 0.375537) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.27678 ± 0.00737215 (W 0.00969687) | 2.95993 ± 0.385895 (W 0.308573) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.077745 ± 0.000560696 (W 0.00036643) | 0.114974 ± 0.000652704 (W 0.00147635) | 0.141714 ± 0.000387713 (W 0.0020538) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.091791 ± 0.000285932 (W 0.000511804) | 0.128099 ± 0.00019713 (W 0.00276471) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.293086 ± 0.00249406 (W 0.00292168) | 0.615446 ± 0.0232267 (W 0.0237016) | 0.287665 ± 0.00189092 (W 0.00286825) | 🔵 faer-fork 2.10× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.579789 ± 0.00212497 (W 0.00343023) | 0.666331 ± 0.0216946 (W 0.024959) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 1.21115 ± 0.00313459 (W 0.00516763) | 2.75088 ± 0.0494603 (W 0.045029) | 1.12307 ± 0.00239751 (W 0.00532828) | 🔵 faer-fork 2.27× faster than `faer-upstream` | 🔴 faer-fork 1.08× slower than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 2.62835 ± 0.0130586 (W 0.0102094) | 2.9226 ± 0.029252 (W 0.0342202) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.274421 ± 0.00167616 (W 0.00204559) | 0.671771 ± 0.0057513 (W 0.00291827) | 0.23656 ± 0.00105651 (W 0.00307067) | 🔵 faer-fork 2.45× faster than `faer-upstream` | 🔴 faer-fork 1.16× slower than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.378064 ± 0.00202156 (W 0.00434892) | 0.761237 ± 0.00597786 (W 0.00316368) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.811841 ± 0.00277769 (W 0.0026365) | 2.02323 ± 0.00311788 (W 0.0114885) | 1.03949 ± 0.00455575 (W 0.00423486) | 🔵 faer-fork 2.49× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.28943 ± 0.00596344 (W 0.0042675) | 2.15658 ± 0.0679194 (W 0.0146536) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 3.2441 ± 0.00213981 (W 0.00399936) | 7.14417 ± 0.0293881 (W 0.0412734) | 5.27841 ± 0.00647561 (W 0.0152902) | 🔵 faer-fork 2.20× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 5.54568 ± 0.0384619 (W 0.0171315) | 7.58442 ± 0.15019 (W 0.0465982) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.196245 ± 0.00179737 (W 0.00224789) | 0.529616 ± 0.00554895 (W 0.00353815) | 0.146543 ± 0.000860061 (W 0.00185836) | 🔵 faer-fork 2.70× faster than `faer-upstream` | 🔴 faer-fork 1.34× slower than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 0.5746 ± 0.0031155 (W 0.00369575) | 1.59902 ± 0.00468327 (W 0.00749173) | 0.540537 ± 0.000563981 (W 0.00315262) | 🔵 faer-fork 2.78× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 1.78041 ± 0.0111606 (W 0.00295609) | 4.7721 ± 0.0253377 (W 0.0410858) | 2.12526 ± 0.00986133 (W 0.0055909) | 🔵 faer-fork 2.68× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.083666 ± 0.00024173 (W 0.000341585) | 0.103794 ± 0.000436625 (W 0.000752159) | 0.140361 ± 0.000271177 (W 0.000248422) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.68× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.092893 ± 0.000362111 (W 0.000640099) | 0.120434 ± 0.000140678 (W 0.00101048) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.287686 ± 0.0027983 (W 0.00213118) | 0.579358 ± 0.02529 (W 0.0258957) | 0.265885 ± 0.00211134 (W 0.0048842) | 🔵 faer-fork 2.01× faster than `faer-upstream` | 🔴 faer-fork 1.08× slower than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.585782 ± 0.0121356 (W 0.00506916) | 0.596642 ± 0.0251129 (W 0.0197307) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 1.13875 ± 0.00464919 (W 0.006275) | 2.43216 ± 0.0167807 (W 0.0359219) | 1.01926 ± 0.00792 (W 0.0112718) | 🔵 faer-fork 2.14× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 2.59152 ± 0.0203374 (W 0.00709141) | 2.56235 ± 0.0346026 (W 0.0394367) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.97885 ± 0.0048741 (W 0.00608912) | 2.76939 ± 0.00451274 (W 0.00135371) | 2.41068 ± 0.0016087 (W 0.00561522) | 🔵 faer-fork 1.40× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.98458 ± 0.00577517 (W 0.00559113) | 2.78549 ± 0.00229056 (W 0.00559476) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.31713 ± 0.00646619 (W 0.0151271) | 13.0582 ± 0.00996221 (W 0.0306578) | 11.1962 ± 0.0197457 (W 0.0158892) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.35771 ± 0.0204042 (W 0.0101158) | 13.0653 ± 0.0346405 (W 0.0494613) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 29.6475 ± 0.0848519 (W 0.0329645) | 51.3761 ± 0.0767886 (W 0.070399) | 48.3783 ± 0.0847126 (W 0.0340238) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 29.7869 ± 0.0954686 (W 0.0337919) | 51.5054 ± 0.0548255 (W 0.129411) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.979142 ± 0.00236992 (W 0.00347537) | 2.09951 ± 0.0020275 (W 0.00276801) | 1.09748 ± 0.00277398 (W 0.0047079) | 🔵 faer-fork 2.14× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.980245 ± 0.00394151 (W 0.0032525) | 2.11261 ± 0.00767205 (W 0.00231446) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.92742 ± 0.00875446 (W 0.0118418) | 10.2794 ± 0.04079 (W 0.0295755) | 5.96005 ± 0.015349 (W 0.0062093) | 🔵 faer-fork 2.62× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.93577 ± 0.0241083 (W 0.0123542) | 10.308 ± 0.0132695 (W 0.0113234) | - | 🔵 faer-fork 2.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.5153 ± 0.0194541 (W 0.0171115) | 39.1091 ± 0.0368667 (W 0.0162133) | 22.2851 ± 0.0586676 (W 0.260719) | 🔵 faer-fork 2.52× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 15.5129 ± 0.00913819 (W 0.0434414) | 39.2006 ± 0.0498373 (W 0.0579307) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00552 ± 1.1547e-05 (W 4.80448e-05) | 0.004538 ± 2.3094e-05 (W 2.25455e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015479 ± 0.000214189 (W 0.000233014) | 0.013906 ± 2.51661e-05 (W 5.54076e-05) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.037259 ± 0.000673317 (W 0.000366953) | 0.035345 ± 0.000239013 (W 0.000353398) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.178733 ± 8.30321e-05 (W 0.0020289) | 0.813647 ± 0.00196561 (W 0.00235472) | - | 🔵 faer-fork 4.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.190896 ± 0.000799478 (W 0.00159504) | 0.183792 ± 0.000246879 (W 0.00238087) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.121 ± 0.00119596 (W 0.00206022) | 2.42418 ± 0.00430478 (W 0.00836507) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.16894 ± 0.000933859 (W 0.00303582) | 1.21195 ± 0.00148958 (W 0.00316972) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.39792 ± 0.00394785 (W 0.00879035) | 10.1004 ± 0.0148437 (W 0.0177023) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 7.69431 ± 0.0261622 (W 0.00965922) | 7.88334 ± 0.0059665 (W 0.0132294) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.57502 ± 0.00130097 (W 0.00497257) | 3.4099 ± 0.00691722 (W 0.0069779) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 3.57305 ± 0.00574804 (W 0.00763047) | 3.4139 ± 0.00231046 (W 0.00733986) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 18.1195 ± 0.0325706 (W 0.0156775) | 17.7252 ± 0.0286633 (W 0.0223266) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 18.2108 ± 0.0457768 (W 0.0386377) | 17.7887 ± 0.035909 (W 0.0266877) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 107.087 ± 0.156103 (W 0.35568) | 105.676 ± 0.271775 (W 0.0996515) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 107.18 ± 0.233443 (W 0.20084) | 105.696 ± 0.188619 (W 0.364994) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.850075 ± 0.00822047 (W 0.00847002) | 0.800619 ± 0.00197273 (W 0.00432455) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.919832 ± 0.00658528 (W 0.00516948) | 0.88505 ± 0.00436544 (W 0.00595362) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 6.09268 ± 0.00921866 (W 0.0124679) | 6.38528 ± 0.018424 (W 0.00813487) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.62993 ± 0.0180161 (W 0.0239627) | 6.58822 ± 0.118472 (W 0.0602778) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 41.2311 ± 0.0411111 (W 0.0697073) | 45.0989 ± 1.11648 (W 0.0429625) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 44.3377 ± 0.0316128 (W 0.19263) | 43.9575 ± 0.20008 (W 0.12902) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.05228 ± 0.00270721 (W 0.00351593) | 1.04824 ± 0.00207561 (W 0.00341778) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.05637 ± 0.00457784 (W 0.00358056) | 1.04962 ± 0.00244122 (W 0.00267914) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.55009 ± 0.00986161 (W 0.0296333) | 7.54287 ± 0.0133628 (W 0.0204179) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.55301 ± 0.0105196 (W 0.0151509) | 7.54517 ± 0.0162105 (W 0.0264226) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.7355 ± 0.106532 (W 0.101721) | 56.8642 ± 0.0150084 (W 0.231024) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.7988 ± 0.0580124 (W 0.101063) | 56.9276 ± 0.103201 (W 0.131694) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.287264 ± 0.00154978 (W 0.00231138) | 0.61179 ± 0.00242956 (W 0.00362064) | - | 🔵 faer-fork 2.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.438458 ± 0.00434671 (W 0.00332376) | 0.442956 ± 0.000830257 (W 0.00420265) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.89563 ± 0.00498718 (W 0.00366143) | 5.51331 ± 0.0272261 (W 0.0237891) | - | 🔵 faer-fork 2.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.66904 ± 0.00647522 (W 0.011313) | 2.81486 ± 0.00182888 (W 0.0100793) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.3937 ± 0.0249103 (W 0.0182504) | 29.9718 ± 0.0156595 (W 0.0495195) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 17.8708 ± 0.0743446 (W 0.0269399) | 18.1504 ± 0.0669767 (W 0.0486487) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.47089 ± 0.00102773 (W 0.0036525) | 1.86124 ± 0.00254237 (W 0.0041954) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 1.71511 ± 0.00610012 (W 0.00382062) | 1.75786 ± 0.00267273 (W 0.00463647) | - | · | - | `*` one row has no measurement here |
| `qr/n256/b1` | 10.1539 ± 0.0147755 (W 0.0117721) | 12.0287 ± 0.0127567 (W 0.026928) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 11.0006 ± 0.0300394 (W 0.0266802) | 11.2574 ± 0.00878899 (W 0.0137648) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 69.8366 ± 0.134197 (W 0.221903) | 77.6638 ± 0.269158 (W 0.162946) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 73.8728 ± 0.13473 (W 0.0846117) | 74.6067 ± 0.084362 (W 0.266335) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.681683 ± 0.000952627 (W 0.00210436) | 1.06875 ± 0.00247414 (W 0.00100563) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 4.67134 ± 0.0134583 (W 0.00810033) | 6.47433 ± 0.0472724 (W 0.0101026) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 30.0899 ± 0.0199002 (W 0.0447182) | 37.8439 ± 0.0761282 (W 0.0238875) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.276494 ± 0.000421691 (W 0.00211873) | 0.444794 ± 0.00157632 (W 0.00302233) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.44521 ± 0.00318406 (W 0.00252689) | 0.44796 ± 0.00417669 (W 0.00427282) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.86203 ± 0.00166939 (W 0.00173324) | 4.26885 ± 0.0113618 (W 0.0134433) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 2.08166 ± 0.00305278 (W 0.00336244) | 2.16503 ± 0.0039513 (W 0.00286252) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.2478 ± 0.0203291 (W 0.0343691) | 23.8512 ± 0.0517134 (W 0.0331059) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 14.741 ± 0.0414429 (W 0.0243956) | 14.9291 ± 0.0253261 (W 0.0359448) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.87858 ± 0.0180138 (W 0.00754904) | 5.69813 ± 0.0147824 (W 0.00870444) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 4.88117 ± 0.0230335 (W 0.00968882) | 5.70828 ± 0.0155095 (W 0.0100055) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 30.2237 ± 0.0110281 (W 0.0365306) | 33.9434 ± 0.0690595 (W 0.051997) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 30.2377 ± 0.0501759 (W 0.0549862) | 33.9132 ± 0.0267374 (W 0.0437667) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 198.77 ± 0.0680504 (W 0.40081) | 208.468 ± 1.13696 (W 0.403898) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 198.744 ± 0.299463 (W 0.25723) | 208.961 ± 0.557297 (W 0.515173) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.69811 ± 0.00531042 (W 0.00703104) | 3.45129 ± 0.00165127 (W 0.00630867) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.69364 ± 0.00265794 (W 0.00386423) | 3.45369 ± 0.00602457 (W 0.00685749) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 12.1757 ± 0.0303432 (W 0.0167915) | 19.2013 ± 0.0123105 (W 0.038581) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 12.1962 ± 0.0247578 (W 0.0200567) | 19.2211 ± 0.0533146 (W 0.0363027) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 87.1225 ± 0.291802 (W 0.280266) | 110.817 ± 1.06225 (W 0.415028) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 87.1686 ± 0.272869 (W 0.0775085) | 110.799 ± 1.00503 (W 0.102054) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.008035 ± 3.05505e-05 (W 0.0001001) | 0.007545 ± 5.55908e-05 (W 5.41295e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024927 ± 0.000250779 (W 0.000182297) | 0.023524 ± 7.60285e-05 (W 0.000110318) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.089867 ± 0.000241664 (W 0.00255511) | 0.086771 ± 0.00016544 (W 0.00173645) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.13995 ± 0.00328131 (W 0.00338147) | 0.801672 ± 0.144141 (W 0.114397) | - | 🔵 faer-fork 5.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.15578 ± 0.00133417 (W 0.0021421) | 0.177531 ± 0.00114627 (W 0.00338678) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.53161 ± 0.00101642 (W 0.00321489) | 1.97576 ± 0.12931 (W 0.129177) | - | 🔵 faer-fork 3.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.608902 ± 0.00296226 (W 0.00369528) | 0.80596 ± 0.0155992 (W 0.00566625) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.38579 ± 0.00902629 (W 0.00401752) | 5.32202 ± 0.071079 (W 0.197048) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.76683 ± 0.0132496 (W 0.00751375) | 3.38855 ± 0.0173486 (W 0.113759) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.94098 ± 0.00298342 (W 0.00339123) | 2.96834 ± 0.00553834 (W 0.00536917) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.94902 ± 0.0068652 (W 0.0070949) | 2.9882 ± 0.00316 (W 0.00285399) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.9078 ± 0.00509158 (W 0.0157413) | 11.5674 ± 0.0295664 (W 0.0160774) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 7.925 ± 0.00439164 (W 0.00979999) | 11.6544 ± 0.0455197 (W 0.0234697) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.9373 ± 0.0321245 (W 0.0285767) | 45.2476 ± 0.0599711 (W 0.0237915) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 39.9537 ± 0.0408469 (W 0.0396368) | 45.4314 ± 0.0913687 (W 0.055228) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.598092 ± 0.00181555 (W 0.00565772) | 0.799839 ± 0.00347652 (W 0.00460134) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.734385 ± 0.00459537 (W 0.00453928) | 0.769583 ± 0.00638785 (W 0.00695219) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.35839 ± 0.00933531 (W 0.0151144) | 6.44562 ± 0.0543693 (W 0.0119733) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 4.08376 ± 0.0101829 (W 0.00533738) | 6.01667 ± 0.0343013 (W 0.0138879) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 16.7319 ± 0.0151111 (W 0.029565) | 28.2621 ± 0.0847256 (W 0.0319579) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 20.7223 ± 0.12758 (W 0.0333058) | 25.6091 ± 0.0782713 (W 0.0479548) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.286923 ± 0.000577219 (W 0.0020291) | 0.282355 ± 0.00134461 (W 0.00182298) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.286523 ± 0.00187941 (W 0.00169235) | 0.282125 ± 0.000421544 (W 0.00204759) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.9581 ± 0.000764125 (W 0.00707618) | 1.94407 ± 0.00572452 (W 0.00329115) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.9518 ± 0.00278642 (W 0.00324738) | 1.9459 ± 0.0105422 (W 0.00530503) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6529 ± 0.0309363 (W 0.0213747) | 14.7001 ± 0.0279157 (W 0.017019) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6436 ± 0.0232864 (W 0.0240203) | 14.6793 ± 0.0266048 (W 0.0158588) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.228225 ± 0.00338318 (W 0.00206461) | 0.592652 ± 0.000847629 (W 0.00640003) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.362404 ± 0.00447982 (W 0.00323015) | 0.314394 ± 0.00119342 (W 0.0041184) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.00142 ± 0.00990472 (W 0.00603801) | 4.26965 ± 0.106429 (W 0.117641) | - | 🔵 faer-fork 4.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.71879 ± 0.00502555 (W 0.0120196) | 1.58259 ± 0.00898036 (W 0.01589) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.16961 ± 0.0645783 (W 0.00837791) | 17.0791 ± 0.503723 (W 0.622709) | - | 🔵 faer-fork 3.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 9.29866 ± 0.0332394 (W 0.0229141) | 7.45411 ± 0.0599478 (W 0.0494965) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.733676 ± 0.00164701 (W 0.00245845) | 1.34171 ± 0.00710868 (W 0.00398432) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.933398 ± 0.00339089 (W 0.00203465) | 1.21005 ± 0.00420692 (W 0.00413183) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.11647 ± 0.00752175 (W 0.00982393) | 6.34772 ± 0.0229857 (W 0.0206593) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 4.23503 ± 0.0048792 (W 0.0058945) | 5.2901 ± 0.0144794 (W 0.0115302) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 20.0045 ± 0.0616369 (W 0.0323812) | 31.766 ± 0.0643312 (W 0.0434773) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 25.4909 ± 0.0626329 (W 0.0272978) | 27.4452 ± 0.100628 (W 0.142011) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.445699 ± 0.000312459 (W 0.00270758) | 1.01878 ± 0.000234514 (W 0.00400671) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.65836 ± 0.00433593 (W 0.00298157) | 4.84327 ± 0.0040412 (W 0.00988064) | - | 🔵 faer-fork 2.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 9.49059 ± 0.0292596 (W 0.0153488) | 21.4307 ± 0.0680611 (W 0.0587339) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.191556 ± 0.000547888 (W 0.00174692) | 0.417967 ± 0.00203972 (W 0.00690074) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.3655 ± 0.00339035 (W 0.00286168) | 0.305157 ± 0.0009918 (W 0.00310588) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.979542 ± 0.00655092 (W 0.0140954) | 3.23478 ± 0.065558 (W 0.0589305) | - | 🔵 faer-fork 3.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.72548 ± 0.00664425 (W 0.0211052) | 1.45842 ± 0.0119318 (W 0.0204345) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.15745 ± 0.0299232 (W 0.0206273) | 12.6095 ± 0.291206 (W 0.425886) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 9.35739 ± 0.108401 (W 0.0532039) | 7.2648 ± 0.111801 (W 0.135816) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.73235 ± 0.0114666 (W 0.00493977) | 4.0407 ± 0.0098112 (W 0.0162877) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.74645 ± 0.00733309 (W 0.00592465) | 4.06472 ± 0.0269848 (W 0.0143087) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 13.1483 ± 0.0115165 (W 0.0226219) | 21.5407 ± 0.0930544 (W 0.0239111) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 13.1843 ± 0.0302978 (W 0.0410127) | 21.5874 ± 0.0250742 (W 0.0737482) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 70.3441 ± 0.106401 (W 0.0935112) | 87.0796 ± 0.0493195 (W 0.0704324) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 70.4366 ± 0.0378928 (W 0.0349466) | 89.5489 ± 0.0970562 (W 1.20422) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.2797 ± 0.00313433 (W 0.00346802) | 2.96816 ± 0.00265542 (W 0.0089326) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.28942 ± 0.00988384 (W 0.0034059) | 3.00483 ± 0.00361297 (W 0.00370756) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.57813 ± 0.0027724 (W 0.00878266) | 16.172 ± 0.0249406 (W 0.0324672) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.60716 ± 0.0103766 (W 0.011335) | 16.1812 ± 0.120757 (W 0.0274061) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 35.4302 ± 0.132142 (W 0.0284347) | 57.6592 ± 0.239005 (W 0.0425488) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 35.491 ± 0.0869401 (W 0.0561565) | 57.8806 ± 0.110336 (W 0.0717696) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009157 ± 3.60555e-05 (W 3.57771e-05) | 0.007564 ± 2e-05 (W 3.50742e-05) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.021761 ± 8.3668e-05 (W 7.59605e-05) | 0.023584 ± 0.000260385 (W 0.000432313) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.057417 ± 0.000696529 (W 0.000364668) | 0.062606 ± 0.000321352 (W 0.000329909) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.136545 ± 0.00142867 (W 0.00197199) | 0.532642 ± 0.146866 (W 0.140964) | - | 🔵 faer-fork 3.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.158545 ± 0.00127322 (W 0.000346288) | 0.165809 ± 0.00107566 (W 0.00126924) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.442864 ± 0.00218 (W 0.00263171) | 1.46148 ± 0.00725367 (W 0.202673) | - | 🔵 faer-fork 3.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.54249 ± 0.0014125 (W 0.00317891) | 0.790231 ± 0.0142248 (W 0.0141106) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.60192 ± 0.00420766 (W 0.00468072) | 3.6551 ± 0.225165 (W 0.276618) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.08981 ± 0.00237545 (W 0.00645517) | 2.74172 ± 0.00329758 (W 0.0458524) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32438 ± 0.0058925 (W 0.00224867) | 2.9673 ± 0.010764 (W 0.0018559) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.33213 ± 0.00257587 (W 0.00785753) | 2.97854 ± 0.00929808 (W 0.00487137) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.41894 ± 0.00465776 (W 0.0115872) | 11.33 ± 0.0188281 (W 0.0590141) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.44586 ± 0.0107037 (W 0.0123213) | 11.3342 ± 0.014782 (W 0.0279974) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.3098 ± 0.0404748 (W 0.0238) | 39.3154 ± 0.0579718 (W 0.252648) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 25.3637 ± 0.0214752 (W 0.0291384) | 39.4776 ± 0.266941 (W 0.220197) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.584138 ± 0.00451953 (W 0.00553345) | 0.802866 ± 0.0037647 (W 0.00351014) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.714762 ± 0.00488452 (W 0.00460839) | 0.767468 ± 0.000663838 (W 0.00381207) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.01224 ± 0.011691 (W 0.0114359) | 6.44439 ± 0.0156619 (W 0.0114402) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.7879 ± 0.013619 (W 0.0109065) | 5.98041 ± 0.0388116 (W 0.0173475) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.9555 ± 0.0409725 (W 0.0193558) | 25.639 ± 0.0448206 (W 0.0385082) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 17.1426 ± 0.0738473 (W 0.0600251) | 23.2396 ± 0.092583 (W 0.0721703) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.182379 ± 0.000520254 (W 0.00156666) | 0.278067 ± 0.00179995 (W 0.00391181) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.182129 ± 0.000967809 (W 0.0028688) | 0.274982 ± 0.00303424 (W 0.00444448) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.02016 ± 0.0502077 (W 0.00388557) | 1.30216 ± 0.083494 (W 0.0768522) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01656 ± 0.00203958 (W 0.00449007) | 1.29743 ± 0.0250063 (W 0.0835949) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.93802 ± 0.0338014 (W 0.0925167) | 7.78937 ± 1.22146 (W 1.51351) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.90548 ± 0.0109825 (W 0.0287085) | 10.1564 ± 1.41265 (W 1.25211) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.23091 ± 0.0010778 (W 0.00134614) | 0.579219 ± 0.00309487 (W 0.00290116) | - | 🔵 faer-fork 2.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.348218 ± 0.00288126 (W 0.00266425) | 0.303355 ± 0.000753146 (W 0.00239504) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.816862 ± 0.00106186 (W 0.00245574) | 4.16258 ± 0.0148953 (W 0.0458424) | - | 🔵 faer-fork 5.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.55932 ± 0.00649384 (W 0.0058856) | 1.50255 ± 0.0184528 (W 0.0101844) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.3841 ± 0.00902348 (W 0.00930803) | 13.7181 ± 0.136303 (W 0.0779894) | - | 🔵 faer-fork 4.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.72056 ± 0.0473522 (W 0.0193562) | 6.00187 ± 0.0214037 (W 0.0379572) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.584759 ± 0.00194638 (W 0.00258726) | 1.45266 ± 0.00560233 (W 0.00734825) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.814027 ± 0.00413009 (W 0.0073083) | 1.32926 ± 0.0114733 (W 0.00764996) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.13552 ± 0.0052455 (W 0.00717727) | 6.25018 ± 0.0461532 (W 0.043666) | - | 🔵 faer-fork 2.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.29882 ± 0.0124558 (W 0.00724874) | 4.35397 ± 0.381966 (W 0.0346773) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 10.7042 ± 0.0452132 (W 0.0225235) | 27.8675 ± 0.0532496 (W 0.183212) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 16.6038 ± 0.0912495 (W 0.0308884) | 23.5301 ± 2.13949 (W 1.95213) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.387291 ± 0.00132764 (W 0.00233345) | 1.10025 ± 0.00504569 (W 0.00921665) | - | 🔵 faer-fork 2.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.33514 ± 0.00172942 (W 0.00349056) | 4.96435 ± 0.0224126 (W 0.0249103) | - | 🔵 faer-fork 3.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.13548 ± 0.00899017 (W 0.0143562) | 20.2847 ± 0.100183 (W 0.165112) | - | 🔵 faer-fork 3.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.200252 ± 0.00538761 (W 0.00182684) | 0.409052 ± 0.00239103 (W 0.0045471) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.355903 ± 0.00345206 (W 0.00159393) | 0.29579 ± 0.000695267 (W 0.00224624) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.781074 ± 0.00168852 (W 0.00320869) | 3.17161 ± 0.0246171 (W 0.0201158) | - | 🔵 faer-fork 4.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.57039 ± 0.0026302 (W 0.00634313) | 1.36216 ± 0.00759615 (W 0.00808131) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.36882 ± 0.00586749 (W 0.00816402) | 11.0943 ± 0.185168 (W 0.110497) | - | 🔵 faer-fork 3.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.79805 ± 0.0662566 (W 0.0229314) | 5.80075 ± 0.0250822 (W 0.029258) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.54595 ± 0.00796292 (W 0.0056374) | 4.06426 ± 0.00974335 (W 0.00549691) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.55594 ± 0.0110553 (W 0.00421261) | 4.08602 ± 0.00982964 (W 0.0102612) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 11.0162 ± 0.0288283 (W 0.0194882) | 20.987 ± 0.0569865 (W 0.102507) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 11.0194 ± 0.0199347 (W 0.014762) | 21.0846 ± 0.0906138 (W 0.0724091) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 50.703 ± 0.125964 (W 0.073739) | 74.5491 ± 0.116436 (W 0.226724) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 50.7313 ± 0.16166 (W 0.103541) | 74.9171 ± 0.0863811 (W 0.81721) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.28187 ± 0.00270454 (W 0.00345249) | 2.95473 ± 0.000621126 (W 0.0052198) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.29053 ± 0.00403564 (W 0.00344598) | 2.9827 ± 0.00450307 (W 0.00546576) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.96591 ± 0.011213 (W 0.00705044) | 16.1632 ± 0.220744 (W 0.0480942) | - | 🔵 faer-fork 2.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.9705 ± 0.0222738 (W 0.0102643) | 16.0969 ± 0.218422 (W 0.0532444) | - | 🔵 faer-fork 2.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 27.4082 ± 0.0321261 (W 0.0277197) | 51.3288 ± 0.146068 (W 0.0602977) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 27.3898 ± 0.0226476 (W 0.0491574) | 51.4427 ± 0.167887 (W 0.0558052) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009177 ± 6e-05 (W 4.06977e-05) | 0.007514 ± 2.08167e-05 (W 6.49615e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.021149 ± 0.000127986 (W 0.000147388) | 0.023664 ± 0.0002105 (W 8.49147e-05) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.050604 ± 0.000676154 (W 0.000108675) | 0.059641 ± 0.000861883 (W 0.000528632) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.164417 ± 0.000870977 (W 0.00157938) | 0.1664 ± 0.000280935 (W 0.00196884) | 0.395327 ± 0.000976199 (W 0.00123361) | · | 🔵 faer-fork 2.40× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.178691 ± 0.000405332 (W 0.00257762) | 0.171449 ± 0.000311372 (W 0.000216396) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.0576 ± 0.00229333 (W 0.00322806) | 1.1057 ± 0.003023 (W 0.00295746) | 1.52885 ± 0.00874031 (W 0.00592503) | · | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 1.12133 ± 0.00160642 (W 0.00221632) | 1.12212 ± 0.00375361 (W 0.00455826) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.20594 ± 0.00523056 (W 0.00795568) | 7.34314 ± 0.00376271 (W 0.0150407) | 10.2652 ± 0.0119893 (W 0.0382429) | · | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 7.50904 ± 0.00855567 (W 0.0139438) | 7.49927 ± 0.0443834 (W 0.0116833) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.5692 ± 0.011483 (W 0.00824446) | 3.40477 ± 0.0113795 (W 0.012602) | 4.30287 ± 0.00742706 (W 0.00650634) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 3.5577 ± 0.0145793 (W 0.00477746) | 3.40873 ± 0.0108145 (W 0.0103365) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 18.0805 ± 0.0567253 (W 0.0200384) | 17.6672 ± 0.0268412 (W 0.0237906) | 26.0264 ± 0.027155 (W 0.0230827) | · | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 18.1194 ± 0.0146406 (W 0.0318557) | 17.6616 ± 0.11165 (W 0.0200327) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 106.743 ± 0.246872 (W 0.302838) | 105.389 ± 0.149927 (W 0.222442) | 169.083 ± 0.276319 (W 0.382188) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 107.194 ± 0.319585 (W 0.13566) | 105.251 ± 0.214923 (W 0.149602) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.79546 ± 0.0552445 (W 0.0055615) | 0.747571 ± 0.00581843 (W 0.00715275) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.900246 ± 0.00348645 (W 0.00481247) | 0.848992 ± 0.00396022 (W 0.00536307) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 6.02723 ± 0.353838 (W 0.0125185) | 5.86389 ± 0.0394273 (W 0.0126219) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.43585 ± 0.0174076 (W 0.0185371) | 6.29287 ± 0.0151544 (W 0.0133011) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 43.0037 ± 0.0503253 (W 0.0657877) | 42.68 ± 0.0854659 (W 0.0286808) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 43.1599 ± 0.305545 (W 0.216913) | 42.6463 ± 0.0899671 (W 0.0744148) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.03821 ± 0.00112755 (W 0.00353225) | 1.03825 ± 0.00392967 (W 0.00390942) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.04093 ± 0.000286194 (W 0.0017128) | 1.03514 ± 0.00618245 (W 0.000855243) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.4672 ± 0.0149133 (W 0.0165172) | 7.46608 ± 0.0141667 (W 0.013937) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.467 ± 0.0100836 (W 0.0196202) | 7.467 ± 0.00566143 (W 0.0274419) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.5592 ± 0.149841 (W 0.115179) | 56.4845 ± 0.138375 (W 0.134034) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.5354 ± 0.143916 (W 0.242611) | 56.4631 ± 0.0297314 (W 0.117541) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.282363 ± 0.00162329 (W 0.0019577) | 0.286944 ± 0.00124626 (W 0.00253425) | 0.431724 ± 0.000664894 (W 0.00216633) | · | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.409132 ± 0.000753735 (W 0.0035426) | 0.410735 ± 0.00155184 (W 0.00382405) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.86678 ± 0.00252799 (W 0.00239684) | 1.99831 ± 0.00394271 (W 0.00566956) | 3.06445 ± 0.00601227 (W 0.0107198) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.64× faster than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 2.51485 ± 0.00628447 (W 0.00498774) | 2.59868 ± 0.0177633 (W 0.00497609) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.2062 ± 0.00846145 (W 0.0213458) | 13.6633 ± 0.0215517 (W 0.0404092) | 19.3909 ± 0.0191483 (W 0.0341802) | · | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 16.844 ± 0.0367697 (W 0.0323614) | 17.0465 ± 0.0191892 (W 0.0416302) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.49192 ± 0.00159422 (W 0.00374025) | 1.53521 ± 0.00155141 (W 0.00309341) | 2.52825 ± 0.00460582 (W 0.00710323) | · | 🔵 faer-fork 1.69× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 1.68279 ± 0.0017727 (W 0.00386799) | 1.72504 ± 0.002033 (W 0.00475594) | - | · | - | `*` one row has no measurement here |
| `qr/n256/b1` | 9.90608 ± 0.00576532 (W 0.011199) | 10.0552 ± 0.0355531 (W 0.0139084) | 15.9458 ± 0.0416994 (W 0.0187391) | · | 🔵 faer-fork 1.61× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 10.8711 ± 0.0465897 (W 0.0219886) | 10.9795 ± 0.0327193 (W 0.0127439) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 68.1691 ± 0.293411 (W 0.0527268) | 68.8058 ± 0.0668167 (W 0.396093) | 100.359 ± 0.111427 (W 0.0531287) | · | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 72.9767 ± 0.0605656 (W 0.0768159) | 73.5234 ± 0.1301 (W 0.0944223) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.713332 ± 0.000752911 (W 0.0017948) | 0.754829 ± 0.000300562 (W 0.00183076) | 0.897533 ± 0.00403303 (W 0.0033568) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 4.37643 ± 0.00483267 (W 0.00511263) | 4.55429 ± 0.00911001 (W 0.00712256) | 5.54164 ± 0.000884197 (W 0.00932212) | · | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 28.2863 ± 0.0474797 (W 0.0282079) | 28.9263 ± 0.0288726 (W 0.039914) | 38.1281 ± 0.0204125 (W 0.0476881) | · | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.29134 ± 0.00148218 (W 0.00263983) | 0.281665 ± 0.0021773 (W 0.00174282) | 0.41346 ± 0.00116922 (W 0.00337491) | · | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.414612 ± 0.00320839 (W 0.00379935) | 0.405952 ± 0.00196886 (W 0.00643519) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.89544 ± 0.00244874 (W 0.00657238) | 1.93947 ± 0.0086306 (W 0.00487431) | 2.50582 ± 0.00502043 (W 0.00811048) | · | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 1.96436 ± 0.00513361 (W 0.00506958) | 1.96103 ± 0.00229071 (W 0.00225114) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.2823 ± 0.0196262 (W 0.0131533) | 13.5861 ± 0.0217924 (W 0.0173376) | 16.7071 ± 0.0233313 (W 0.0214493) | · | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 13.9625 ± 0.013414 (W 0.0237523) | 13.9462 ± 0.0215507 (W 0.0333968) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.8501 ± 0.0093005 (W 0.0183816) | 5.69211 ± 0.0192686 (W 0.0119733) | 6.85149 ± 0.0134451 (W 0.0218492) | 🔵 faer-fork 1.17× faster than `faer-upstream` | 🔵 faer-fork 1.41× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.85317 ± 0.0125435 (W 0.00896674) | 5.69487 ± 0.00993108 (W 0.00457341) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 30.0191 ± 0.0499347 (W 0.0434985) | 33.6431 ± 0.0869839 (W 0.0617134) | 44.6674 ± 0.16892 (W 0.279766) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 30.1355 ± 0.0825394 (W 0.0244375) | 33.705 ± 0.139539 (W 0.0405287) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 197.606 ± 0.270778 (W 0.277918) | 207.781 ± 0.900934 (W 0.425154) | 300.062 ± 0.621254 (W 0.609218) | 🔵 faer-fork 1.05× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 197.307 ± 0.473921 (W 0.302957) | 208.717 ± 0.74254 (W 0.273564) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.66826 ± 0.00386567 (W 0.005543) | 3.43631 ± 0.0108621 (W 0.0065906) | 2.63494 ± 0.00568615 (W 0.00644904) | 🔵 faer-fork 2.06× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.66889 ± 0.00318703 (W 0.00458191) | 3.43794 ± 0.00789821 (W 0.00622235) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.9832 ± 0.0427768 (W 0.0239779) | 18.9238 ± 0.158307 (W 0.0421119) | 20.5602 ± 0.0927606 (W 0.105572) | 🔵 faer-fork 1.58× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 11.9214 ± 0.0329902 (W 0.0151706) | 18.9494 ± 0.0997582 (W 0.0238846) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 86.0984 ± 0.198781 (W 0.0979121) | 108.739 ± 0.749408 (W 0.191312) | 154.825 ± 0.394891 (W 0.363571) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.80× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 86.3689 ± 0.354003 (W 0.123028) | 109.598 ± 0.897359 (W 0.440325) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.008285 ± 2.5697e-05 (W 5.93153e-05) | 0.008305 ± 8.94446e-05 (W 0.000101977) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.025537 ± 0.000457301 (W 0.000253676) | 0.024395 ± 7.14866e-05 (W 0.000128508) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.093795 ± 0.000800276 (W 0.000734363) | 0.090398 ± 0.000731164 (W 0.000363401) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.130753 ± 0.000600168 (W 0.00242516) | 0.154417 ± 0.00132202 (W 0.00457992) | 0.397529 ± 0.00233301 (W 0.00142119) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 3.04× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.145881 ± 0.00227475 (W 0.00152547) | 0.16651 ± 0.00132693 (W 0.00191274) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.493328 ± 0.00298712 (W 0.00212444) | 0.659187 ± 0.00181522 (W 0.00621518) | 0.523685 ± 0.00863446 (W 0.0115655) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.567054 ± 0.00416701 (W 0.00239745) | 0.715081 ± 0.0017013 (W 0.00478876) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.29812 ± 0.00861626 (W 0.00683768) | 2.74733 ± 0.0863036 (W 0.0884175) | 2.96431 ± 0.00693083 (W 0.01569) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.61465 ± 0.0156988 (W 0.00627364) | 3.03201 ± 0.153486 (W 0.141329) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.92746 ± 0.00469276 (W 0.00355133) | 2.94582 ± 0.00943268 (W 0.00954691) | 1.93351 ± 0.00550031 (W 0.00451946) | 🔵 faer-fork 1.53× faster than `faer-upstream` | · |  |
| `eigh/n128/b1 [api-inclusive]` | 1.92827 ± 0.000726331 (W 0.00130959) | 2.97699 ± 0.00853446 (W 0.00382727) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.84348 ± 0.00914942 (W 0.00941904) | 11.4409 ± 0.021895 (W 0.0200822) | 9.8225 ± 0.0204588 (W 0.0170352) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 7.8717 ± 0.00834588 (W 0.0104995) | 11.5581 ± 0.0239042 (W 0.0235355) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.7173 ± 0.0483097 (W 0.0293214) | 44.8475 ± 0.0404106 (W 0.0321495) | 51.8801 ± 0.164624 (W 0.0423807) | 🔵 faer-fork 1.13× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 39.7542 ± 0.0145127 (W 0.0231478) | 45.0021 ± 0.0450159 (W 0.0725465) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.587582 ± 0.011354 (W 0.00206263) | 0.754362 ± 0.00439438 (W 0.00824341) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.69937 ± 0.00291691 (W 0.00463998) | 0.741108 ± 0.00105736 (W 0.00505301) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.33155 ± 0.0169734 (W 0.00948736) | 5.90723 ± 0.0130532 (W 0.0104595) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.90546 ± 0.0181798 (W 0.0119555) | 5.73627 ± 0.0227258 (W 0.00759668) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 17.2187 ± 0.032199 (W 0.023165) | 25.8838 ± 0.123556 (W 0.0414755) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 19.7779 ± 0.0371749 (W 0.0435165) | 24.5623 ± 0.0647464 (W 0.0417743) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.28497 ± 0.00158593 (W 0.00225246) | 0.281514 ± 0.000466034 (W 0.00193172) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.285742 ± 0.00160032 (W 0.00153586) | 0.281955 ± 0.000774072 (W 0.00242766) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.95715 ± 0.00567167 (W 0.00466112) | 1.93082 ± 0.00385027 (W 0.00369009) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.95179 ± 0.00443263 (W 0.00387699) | 1.94269 ± 0.00687169 (W 0.00597658) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.7297 ± 0.0208117 (W 0.0211899) | 14.7633 ± 0.0533538 (W 0.029039) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.7072 ± 0.0139953 (W 0.0101542) | 14.7413 ± 0.0467872 (W 0.0220988) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.203187 ± 0.00246872 (W 0.0014099) | 0.261536 ± 0.00310758 (W 0.0046522) | 0.434508 ± 0.00105192 (W 0.00293308) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 2.14× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.330724 ± 0.00314223 (W 0.00384033) | 0.284229 ± 0.00127897 (W 0.00361294) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.898382 ± 0.0113151 (W 0.0078765) | 1.32232 ± 0.00417816 (W 0.0123077) | 0.902849 ± 0.00192479 (W 0.00489153) | 🔵 faer-fork 1.47× faster than `faer-upstream` | · |  |
| `lu/n256/b1 [api-inclusive]` | 1.56422 ± 0.0172053 (W 0.00870105) | 1.38282 ± 0.0101773 (W 0.0191315) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.68136 ± 0.0523148 (W 0.0557151) | 6.07919 ± 0.133421 (W 0.0993638) | 5.78621 ± 0.0062153 (W 0.0190857) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 8.31156 ± 0.0945462 (W 0.0691859) | 6.52357 ± 0.051159 (W 0.0989136) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.690094 ± 0.00181987 (W 0.00288913) | 0.970876 ± 0.00189273 (W 0.00311202) | 0.918459 ± 0.000596574 (W 0.00472458) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.911886 ± 0.00510117 (W 0.00293669) | 1.17122 ± 0.00688043 (W 0.00231894) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.04364 ± 0.00860465 (W 0.00691552) | 4.05093 ± 0.00714488 (W 0.00806049) | 4.31235 ± 0.0117507 (W 0.0108637) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 4.06459 ± 0.0114723 (W 0.00899226) | 5.07529 ± 0.00869185 (W 0.00819562) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 19.201 ± 0.0490785 (W 0.0284217) | 21.1677 ± 0.0271442 (W 0.0332321) | 26.8285 ± 0.0307414 (W 0.0306429) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 24.5476 ± 0.0753948 (W 0.112824) | 26.3746 ± 0.0480312 (W 0.0263576) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.412306 ± 0.00145203 (W 0.00209092) | 0.661391 ± 0.0030504 (W 0.00231284) | 0.447702 ± 0.0014341 (W 0.00218284) | 🔵 faer-fork 1.60× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 1.56381 ± 0.000729451 (W 0.00469718) | 2.59265 ± 0.0104493 (W 0.00488745) | 1.54053 ± 0.0026841 (W 0.00539936) | 🔵 faer-fork 1.66× faster than `faer-upstream` | · |  |
| `qr_factor/n512/b1` | 8.95501 ± 0.0160323 (W 0.0215687) | 10.9228 ± 0.0256278 (W 0.0189459) | 10.7205 ± 0.00569606 (W 0.0163428) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.204179 ± 0.00506688 (W 0.00150306) | 0.250306 ± 0.00189298 (W 0.00228298) | 0.416555 ± 0.000614717 (W 0.00342504) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 2.04× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.33818 ± 0.00103242 (W 0.00498227) | 0.275523 ± 0.00201778 (W 0.00304671) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.869106 ± 0.00765799 (W 0.0108939) | 1.15793 ± 0.0039076 (W 0.0165492) | 0.76312 ± 0.00144176 (W 0.00468092) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 1.56214 ± 0.00400545 (W 0.0124801) | 1.27802 ± 0.0120933 (W 0.0123147) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.67874 ± 0.0557299 (W 0.0558493) | 5.6817 ± 0.0314157 (W 0.0675277) | 5.12398 ± 0.00558926 (W 0.00684117) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 8.35397 ± 0.0658908 (W 0.0361548) | 6.36993 ± 0.0507897 (W 0.142469) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.70446 ± 0.00624998 (W 0.00559611) | 4.02269 ± 0.0117807 (W 0.0153353) | 3.96881 ± 0.0108336 (W 0.00861739) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.71575 ± 0.0108061 (W 0.00534833) | 4.04495 ± 0.0115836 (W 0.0138915) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.9427 ± 0.031266 (W 0.0333471) | 21.2213 ± 0.098764 (W 0.0260201) | 21.6745 ± 0.037181 (W 0.0291156) | 🔵 faer-fork 1.64× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 12.9596 ± 0.0596424 (W 0.0336341) | 21.3483 ± 0.0343862 (W 0.070069) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 69.2902 ± 0.0161982 (W 0.391448) | 86.1508 ± 0.209936 (W 0.0673633) | 119.158 ± 0.219498 (W 0.427606) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 69.5138 ± 0.14187 (W 0.0570201) | 88.7384 ± 0.371641 (W 1.01303) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.26167 ± 0.00229557 (W 0.00549529) | 2.92818 ± 0.00513039 (W 0.00836011) | 1.93649 ± 0.00478534 (W 0.00353826) | 🔵 faer-fork 2.32× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.2753 ± 0.00280862 (W 0.00488465) | 2.96831 ± 0.00347403 (W 0.00265809) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.41492 ± 0.0200476 (W 0.0107063) | 15.9646 ± 0.0419352 (W 0.0223993) | 10.8393 ± 0.0807427 (W 0.028884) | 🔵 faer-fork 2.49× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.41689 ± 0.0196163 (W 0.0130009) | 16.1142 ± 0.0718801 (W 0.0233206) | - | 🔵 faer-fork 2.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 34.6022 ± 0.0233958 (W 0.0345514) | 56.7572 ± 0.0858081 (W 0.0745755) | 54.9859 ± 0.146219 (W 0.08861) | 🔵 faer-fork 1.64× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 34.5899 ± 0.0549415 (W 0.0575501) | 57.1471 ± 0.140257 (W 0.123682) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009207 ± 6.05007e-05 (W 3.4641e-05) | 0.008205 ± 7.38941e-05 (W 5.6738e-05) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.022963 ± 6.84617e-05 (W 0.000298949) | 0.024756 ± 0.000353016 (W 6.57267e-05) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.061936 ± 0.000269359 (W 0.000429803) | 0.066674 ± 0.000772063 (W 0.00153929) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.12898 ± 0.00198551 (W 0.00176562) | 0.142485 ± 0.000480938 (W 0.00102992) | 0.395707 ± 0.00223376 (W 0.00398207) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 3.07× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.145201 ± 0.00142321 (W 0.00160068) | 0.153236 ± 0.0012466 (W 0.00142992) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.416155 ± 0.000390309 (W 0.00265177) | 0.645973 ± 0.00669432 (W 0.0187218) | 0.371612 ± 0.00385295 (W 0.00322625) | 🔵 faer-fork 1.55× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.503088 ± 0.0015499 (W 0.0031609) | 0.697338 ± 0.00304061 (W 0.0141238) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.54133 ± 0.104735 (W 0.00436404) | 2.10115 ± 0.0184915 (W 0.0252286) | 1.50337 ± 0.00755153 (W 0.0182837) | 🔵 faer-fork 1.36× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.91912 ± 0.0154798 (W 0.0071218) | 2.37099 ± 0.0756076 (W 0.0746602) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.31225 ± 0.00700534 (W 0.00532368) | 2.95759 ± 0.0034624 (W 0.00420859) | 1.59712 ± 0.00856146 (W 0.00416092) | 🔵 faer-fork 2.25× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.31737 ± 0.00156637 (W 0.00287618) | 2.97315 ± 0.00956898 (W 0.00671539) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.3699 ± 0.0116872 (W 0.00666961) | 11.2353 ± 0.0310007 (W 0.0637101) | 7.67687 ± 0.0308532 (W 0.0257657) | 🔵 faer-fork 2.09× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.39205 ± 0.0103385 (W 0.00738985) | 11.2394 ± 0.0896707 (W 0.0445514) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.1146 ± 0.0430721 (W 0.0349307) | 38.8101 ± 0.0782653 (W 0.188096) | 35.6092 ± 0.0266686 (W 0.040315) | 🔵 faer-fork 1.55× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 25.1987 ± 0.0700869 (W 0.0572654) | 39.0091 ± 0.0875111 (W 0.257502) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.573156 ± 0.0122881 (W 0.0032142) | 0.750408 ± 0.00524168 (W 0.0138952) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.696586 ± 0.00218629 (W 0.00396562) | 0.740028 ± 0.00451235 (W 0.00650723) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.94903 ± 0.0166193 (W 0.00608848) | 5.88912 ± 0.0237637 (W 0.0188596) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.5977 ± 0.0175662 (W 0.0189029) | 5.78608 ± 0.0569797 (W 0.0125221) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 13.2054 ± 0.0413408 (W 0.0415211) | 23.4442 ± 0.0161582 (W 0.0492956) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 16.1596 ± 0.0897465 (W 0.0367039) | 22.2368 ± 0.0224053 (W 0.0924058) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.180546 ± 0.000877239 (W 0.000990287) | 0.272958 ± 0.00171774 (W 0.00173409) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.180566 ± 0.00110826 (W 0.000839506) | 0.272398 ± 0.001413 (W 0.00222798) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.02683 ± 0.00305824 (W 0.00275175) | 1.33378 ± 0.10851 (W 0.0829374) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.02216 ± 0.00613556 (W 0.00391553) | 1.17789 ± 0.0911852 (W 0.100724) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 8.06349 ± 0.0394427 (W 0.123839) | 7.87381 ± 1.30445 (W 1.24587) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 8.07962 ± 0.0496742 (W 0.0874711) | 10.1247 ± 1.3947 (W 1.44837) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.18814 ± 0.00169914 (W 0.00120771) | 0.250948 ± 0.000736852 (W 0.00374533) | 0.432334 ± 0.00267764 (W 0.00430223) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔵 faer-fork 2.30× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.322009 ± 0.000707117 (W 0.00246852) | 0.271436 ± 0.001237 (W 0.00166319) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.739687 ± 0.00115811 (W 0.00309) | 1.22539 ± 0.0115572 (W 0.0148791) | 0.566174 ± 0.00552229 (W 0.00615122) | 🔵 faer-fork 1.66× faster than `faer-upstream` | 🔴 faer-fork 1.31× slower than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 1.40542 ± 0.00404333 (W 0.0040654) | 1.30142 ± 0.00680632 (W 0.013828) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.17508 ± 0.00605701 (W 0.00648334) | 4.65532 ± 0.0193482 (W 0.0253158) | 3.24591 ± 0.0111997 (W 0.0057944) | 🔵 faer-fork 1.47× faster than `faer-upstream` | · |  |
| `lu/n512/b1 [api-inclusive]` | 6.76364 ± 0.0764533 (W 0.036156) | 5.09956 ± 0.038804 (W 0.0324456) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.549404 ± 0.00244633 (W 0.00306977) | 1.09284 ± 0.00529341 (W 0.00564993) | 0.569702 ± 0.00150397 (W 0.00403546) | 🔵 faer-fork 1.99× faster than `faer-upstream` | · |  |
| `qr/n128/b1 [api-inclusive]` | 0.780033 ± 0.0030791 (W 0.00376155) | 1.29168 ± 0.00375848 (W 0.00537743) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.07525 ± 0.00691013 (W 0.00428837) | 4.04905 ± 0.0378671 (W 0.0358406) | 3.04888 ± 0.00499259 (W 0.0100729) | 🔵 faer-fork 1.95× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.12846 ± 0.0158693 (W 0.0077263) | 4.18053 ± 0.0248119 (W 0.0334438) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.1472 ± 0.0157958 (W 0.0206742) | 17.7516 ± 0.133176 (W 0.265663) | 13.6998 ± 0.0226065 (W 0.0195844) | 🔵 faer-fork 1.75× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 15.5327 ± 0.110528 (W 0.0128476) | 18.4265 ± 2.53617 (W 0.700638) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.37059 ± 0.00117658 (W 0.00194711) | 0.757161 ± 0.00246106 (W 0.00449983) | 0.30574 ± 0.00327393 (W 0.00260711) | 🔵 faer-fork 2.04× faster than `faer-upstream` | 🔴 faer-fork 1.21× slower than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 1.27112 ± 8.58545e-05 (W 0.00286385) | 2.76066 ± 0.0115388 (W 0.0234188) | 1.5873 ± 0.00809998 (W 0.00329986) | 🔵 faer-fork 2.17× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 4.82267 ± 0.0109222 (W 0.0233534) | 9.98119 ± 0.0531145 (W 0.116524) | 5.53646 ± 0.016049 (W 0.00887595) | 🔵 faer-fork 2.07× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.194141 ± 0.000487963 (W 0.00185636) | 0.24115 ± 0.000477861 (W 0.00193999) | 0.415774 ± 0.00124286 (W 0.00382569) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 2.14× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.324204 ± 0.000411162 (W 0.00314804) | 0.262098 ± 0.000490514 (W 0.00183209) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.73587 ± 0.0020929 (W 0.00172062) | 1.1066 ± 0.00592244 (W 0.0111226) | 0.535107 ± 0.00193698 (W 0.00351619) | 🔵 faer-fork 1.50× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 1.41225 ± 0.0105673 (W 0.00474409) | 1.17501 ± 0.00710724 (W 0.0102044) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.15652 ± 0.00371054 (W 0.0072877) | 4.45076 ± 0.0124639 (W 0.0537295) | 2.84426 ± 0.00564476 (W 0.00756094) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔴 faer-fork 1.11× slower than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 6.75202 ± 0.0444551 (W 0.0588815) | 4.90757 ± 0.0299052 (W 0.0354237) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.50578 ± 0.00665435 (W 0.00582106) | 4.04966 ± 0.00946105 (W 0.0129103) | 3.46775 ± 0.00741113 (W 0.00700837) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.52999 ± 0.00339944 (W 0.0058898) | 4.07715 ± 0.00280457 (W 0.0141732) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.7583 ± 0.00705622 (W 0.0113483) | 20.8625 ± 0.0454022 (W 0.0516756) | 17.0825 ± 0.0318027 (W 0.016717) | 🔵 faer-fork 1.94× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 10.8367 ± 0.0150641 (W 0.0511175) | 21.0437 ± 0.122773 (W 0.154802) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.6138 ± 0.00715306 (W 0.228241) | 73.8554 ± 0.171951 (W 0.486901) | 84.4815 ± 0.714766 (W 0.0411199) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 1.70× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 49.8276 ± 0.0819226 (W 0.0470934) | 73.9981 ± 0.357821 (W 0.608793) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.26716 ± 0.004626 (W 0.00319597) | 2.92187 ± 0.00544239 (W 0.00739109) | 1.84292 ± 0.00495532 (W 0.00391311) | 🔵 faer-fork 2.31× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.26847 ± 0.00380779 (W 0.00272679) | 2.96054 ± 0.0062623 (W 0.00545401) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.81774 ± 0.0166272 (W 0.00861078) | 15.8826 ± 0.00404735 (W 0.0286551) | 9.82193 ± 0.132327 (W 0.013499) | 🔵 faer-fork 2.73× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.81827 ± 0.0225571 (W 0.00400059) | 15.997 ± 0.154742 (W 0.0291825) | - | 🔵 faer-fork 2.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 26.5457 ± 0.0274091 (W 0.400381) | 50.3641 ± 0.02894 (W 0.413813) | 45.0781 ± 0.26153 (W 0.0568258) | 🔵 faer-fork 1.90× faster than `faer-upstream` | 🔵 faer-fork 1.70× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 26.5214 ± 0.0393263 (W 0.0591239) | 50.6682 ± 0.156428 (W 0.0559615) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009307 ± 6.84617e-05 (W 3.86549e-05) | 0.008086 ± 9.02238e-05 (W 0.000138619) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.023674 ± 3.60555e-05 (W 0.000175615) | 0.024556 ± 0.000297725 (W 0.000139224) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.05903 ± 0.000290057 (W 0.000322829) | 0.06445 ± 0.000407296 (W 0.00084962) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.047759 ± 0.000338851 (W 0.000145703) | 0.284811 ± 0.000545003 (W 0.00165555) | - | 🔵 faer-fork 5.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.040566 ± 0.000215007 (W 0.000117704) | 0.040154 ± 0.00211695 (W 0.000122246) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.216393 ± 0.00128539 (W 0.00203688) | 0.694789 ± 0.00194255 (W 0.00250192) | - | 🔵 faer-fork 3.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.230959 ± 0.00198146 (W 0.00306434) | 0.229427 ± 0.00120652 (W 0.00370338) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.38868 ± 0.0127134 (W 0.00590694) | 2.27582 ± 0.0169618 (W 0.00581833) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.32187 ± 0.0116539 (W 0.00251577) | 1.42111 ± 0.00272391 (W 0.00439745) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.76475 ± 0.000943953 (W 0.00306355) | 1.66604 ± 0.00408815 (W 0.00657572) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.74873 ± 0.00134939 (W 0.00512944) | 1.65909 ± 0.0022506 (W 0.00486427) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.61044 ± 0.0273944 (W 0.0125246) | 6.26809 ± 0.0116287 (W 0.00893173) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.60878 ± 0.0191687 (W 0.0152162) | 6.29632 ± 0.0235552 (W 0.00893958) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.5503 ± 0.133181 (W 0.113919) | 28.9358 ± 0.174937 (W 0.0450875) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.7252 ± 0.217073 (W 0.0707283) | 28.8479 ± 0.0868085 (W 0.0239721) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.264662 ± 0.00276823 (W 0.00619323) | 0.185536 ± 0.0014782 (W 0.00170625) | - | 🔴 faer-fork 1.43× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.24235 ± 0.00199914 (W 0.00217303) | 0.202378 ± 0.000640532 (W 0.00190008) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.45424 ± 0.0159034 (W 0.00768018) | 1.33912 ± 0.00786294 (W 0.00859256) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.48065 ± 0.00248597 (W 0.00791712) | 1.40375 ± 0.0104211 (W 0.00491595) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.1547 ± 0.0594779 (W 0.056146) | 13.3614 ± 0.116364 (W 0.0228814) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.2892 ± 0.0208694 (W 0.0364955) | 12.2215 ± 0.100956 (W 0.046708) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.146813 ± 0.00031201 (W 0.00190612) | 0.146222 ± 0.000635547 (W 0.00033335) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.147275 ± 0.000312058 (W 0.00232455) | 0.145942 ± 0.000419608 (W 0.000239437) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00244 ± 0.0060276 (W 0.00408195) | 0.993054 ± 0.00260371 (W 0.00242404) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.990988 ± 0.00887319 (W 0.0038837) | 0.994912 ± 0.00186845 (W 0.00378175) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.42912 ± 0.0297194 (W 0.0196041) | 7.42879 ± 0.0171455 (W 0.0217605) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.4093 ± 0.0406308 (W 0.0184364) | 7.4049 ± 0.0132878 (W 0.0136213) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.075932 ± 0.000771752 (W 0.000353893) | 0.150921 ± 0.000433167 (W 0.0014769) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.099695 ± 0.000956051 (W 0.000634936) | 0.092873 ± 0.000538463 (W 0.00180084) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.363697 ± 0.00165125 (W 0.00222878) | 1.49666 ± 0.00286733 (W 0.0117945) | - | 🔵 faer-fork 4.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.585186 ± 0.00111712 (W 0.00395039) | 0.624675 ± 0.00112803 (W 0.00335467) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.11299 ± 0.00112788 (W 0.00441509) | 10.9299 ± 0.0234875 (W 0.0176948) | - | 🔵 faer-fork 5.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.25236 ± 0.0120247 (W 0.00978235) | 3.7479 ± 0.053534 (W 0.00932827) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.359709 ± 0.000989521 (W 0.0026233) | 0.625895 ± 0.00180162 (W 0.00262251) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.458391 ± 0.00209373 (W 0.00238012) | 0.483588 ± 0.00333139 (W 0.00483199) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.79959 ± 0.0142215 (W 0.0104557) | 2.52387 ± 0.00618261 (W 0.0048692) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.91715 ± 0.00561215 (W 0.00517421) | 2.06472 ± 0.00248676 (W 0.00496371) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.5272 ± 0.0508659 (W 0.0137678) | 15.5445 ± 0.0315346 (W 0.0510606) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 10.9589 ± 0.0360975 (W 0.0152119) | 11.7038 ± 0.0253221 (W 0.0350145) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.1939 ± 0.000871664 (W 0.00208335) | 0.453241 ± 0.00215928 (W 0.00227858) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.04308 ± 0.00138951 (W 0.00207535) | 1.78954 ± 0.00284421 (W 0.00336919) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.2318 ± 0.00841164 (W 0.00746239) | 10.43 ± 0.146472 (W 0.0293025) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.068537 ± 0.000715306 (W 0.000163647) | 0.148767 ± 0.00131463 (W 0.0019147) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.104213 ± 0.00144279 (W 0.00193668) | 0.094976 ± 0.000977045 (W 0.000300305) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.328451 ± 0.000307533 (W 0.00240896) | 0.845256 ± 0.002587 (W 0.0115178) | - | 🔵 faer-fork 2.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.470343 ± 0.000567004 (W 0.00330917) | 0.508073 ± 0.001039 (W 0.0099913) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.96023 ± 0.00461242 (W 0.00236127) | 8.20607 ± 0.0238498 (W 0.019061) | - | 🔵 faer-fork 4.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.69549 ± 0.00699281 (W 0.00647444) | 3.16069 ± 0.044014 (W 0.00836185) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.16473 ± 0.00533423 (W 0.00438245) | 2.33067 ± 0.0139878 (W 0.00407009) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.16531 ± 0.00748165 (W 0.00560862) | 2.31548 ± 0.00680339 (W 0.00720633) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.33412 ± 0.0277418 (W 0.0170005) | 11.0854 ± 0.0344005 (W 0.0209549) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.36156 ± 0.00889974 (W 0.0163387) | 11.1703 ± 0.0279233 (W 0.0108951) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 50.4431 ± 0.102123 (W 0.0540462) | 61.195 ± 0.159627 (W 0.0689868) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 50.6615 ± 0.195978 (W 0.0756056) | 61.3783 ± 0.0314545 (W 0.0680532) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.663708 ± 0.00292509 (W 0.00330594) | 1.70522 ± 0.00284417 (W 0.00394502) | - | 🔵 faer-fork 2.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.659798 ± 0.00537906 (W 0.00407968) | 1.70528 ± 0.004822 (W 0.00195807) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.35959 ± 0.0125069 (W 0.00887084) | 7.85564 ± 0.0134018 (W 0.0146765) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.37182 ± 0.0134393 (W 0.0066461) | 7.89748 ± 0.0306609 (W 0.0168002) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 20.2139 ± 0.10548 (W 0.0155748) | 40.1054 ± 0.0363997 (W 0.0662614) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 20.2508 ± 0.039795 (W 0.0238792) | 40.2258 ± 0.125335 (W 0.0554406) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.003637 ± 6.1101e-05 (W 9.06642e-05) | 0.003466 ± 2.0502e-05 (W 2.96193e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.008946 ± 3.09892e-05 (W 0.000130305) | 0.008596 ± 1.57162e-05 (W 5.15878e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.027141 ± 0.000431015 (W 0.000112518) | 0.026719 ± 0.000240389 (W 0.000232579) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.044272 ± 0.000108167 (W 0.000192599) | 0.196786 ± 0.00205286 (W 0.0524271) | - | 🔵 faer-fork 4.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.03761 ± 0.000224255 (W 0.000225233) | 0.051085 ± 0.000337244 (W 0.00122879) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.149919 ± 0.00359533 (W 0.00163524) | 0.604506 ± 0.0516149 (W 0.0954971) | - | 🔵 faer-fork 4.03× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.168263 ± 0.00466643 (W 0.00291572) | 0.262628 ± 0.00519185 (W 0.0112825) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.665508 ± 0.007737 (W 0.00522334) | 1.61243 ± 0.065011 (W 0.0495228) | - | 🔵 faer-fork 2.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.696165 ± 0.00531609 (W 0.00253605) | 1.07346 ± 0.0137001 (W 0.0422483) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.04507 ± 0.00790657 (W 0.00363101) | 1.58052 ± 0.00468886 (W 0.00552299) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.04683 ± 0.00595571 (W 0.00158613) | 1.60501 ± 0.00884816 (W 0.0051651) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.73694 ± 0.00966225 (W 0.0051143) | 4.35198 ± 0.00640306 (W 0.00933432) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.72919 ± 0.00488187 (W 0.00463624) | 4.36787 ± 0.0413477 (W 0.0220044) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.2684 ± 0.017811 (W 0.0225628) | 15.3282 ± 0.0142638 (W 0.0268261) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.2489 ± 0.0439223 (W 0.0156848) | 15.2807 ± 0.0435155 (W 0.0229161) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.235087 ± 0.000972617 (W 0.00212707) | 0.189393 ± 0.000448346 (W 0.00213134) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.24728 ± 0.00152268 (W 0.0022304) | 0.20382 ± 0.00365847 (W 0.00179525) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.12096 ± 0.00868389 (W 0.00620859) | 1.33895 ± 0.00685117 (W 0.011416) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.27437 ± 0.00746636 (W 0.00379879) | 1.29747 ± 0.00800461 (W 0.00523534) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.30309 ± 0.0441272 (W 0.0124991) | 10.2025 ± 0.0594523 (W 0.0198922) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 8.13275 ± 0.0236854 (W 0.013501) | 9.06417 ± 0.061044 (W 0.0165038) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.053349 ± 0.000135769 (W 0.000496602) | 0.051296 ± 0.000812941 (W 0.000727291) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.052889 ± 0.00118851 (W 0.000499303) | 0.049211 ± 0.00035109 (W 0.000296054) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.27953 ± 0.00183379 (W 0.00247326) | 0.280221 ± 0.00138544 (W 0.00144416) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.280322 ± 0.00210973 (W 0.0014406) | 0.280102 ± 0.000932553 (W 0.00170336) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.94438 ± 0.00211772 (W 0.00572237) | 1.93305 ± 0.00557554 (W 0.00594308) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.94251 ± 0.00468884 (W 0.0056294) | 1.93641 ± 0.00663332 (W 0.00737707) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.099024 ± 0.00180289 (W 0.00144258) | 0.174124 ± 0.00252652 (W 0.00146628) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.09167 ± 0.000870238 (W 0.00085743) | 0.117328 ± 0.000921133 (W 0.0018104) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.296883 ± 0.00413277 (W 0.00349905) | 1.43533 ± 0.0248459 (W 0.0403957) | - | 🔵 faer-fork 4.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.462721 ± 0.00185435 (W 0.00295383) | 0.59159 ± 0.00554133 (W 0.00630905) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.26882 ± 0.00749362 (W 0.00849394) | 7.52222 ± 0.072023 (W 0.188154) | - | 🔵 faer-fork 5.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.04012 ± 0.00698053 (W 0.00800665) | 2.67532 ± 0.0302387 (W 0.038735) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.288477 ± 0.00128414 (W 0.00254523) | 0.782987 ± 0.00496984 (W 0.00266934) | - | 🔵 faer-fork 2.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.330035 ± 0.00206685 (W 0.00273197) | 0.570211 ± 0.00379083 (W 0.00406738) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.775223 ± 0.000856554 (W 0.00317496) | 2.56881 ± 0.0150797 (W 0.0192601) | - | 🔵 faer-fork 3.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.05448 ± 0.00310753 (W 0.0024538) | 1.70224 ± 0.0325961 (W 0.0059229) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.664 ± 0.00514974 (W 0.00307341) | 12.2292 ± 0.162027 (W 0.0318978) | - | 🔵 faer-fork 3.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.8548 ± 0.0113599 (W 0.0138164) | 6.55843 ± 0.0314635 (W 0.0120427) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.197828 ± 0.00110064 (W 0.00150358) | 0.645401 ± 0.00542318 (W 0.0046871) | - | 🔵 faer-fork 3.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.536308 ± 0.00216173 (W 0.00290719) | 2.36095 ± 0.0125348 (W 0.0135874) | - | 🔵 faer-fork 4.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.25261 ± 0.0149291 (W 0.00691681) | 10.9194 ± 0.158481 (W 0.0421395) | - | 🔵 faer-fork 4.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.076412 ± 0.00101522 (W 0.00137206) | 0.164106 ± 0.000943464 (W 0.00310453) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.097702 ± 0.000335112 (W 0.000632923) | 0.116336 ± 0.000563011 (W 0.00134426) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.296772 ± 0.00455585 (W 0.00222649) | 1.03704 ± 0.00925048 (W 0.0292322) | - | 🔵 faer-fork 3.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.471257 ± 0.00047944 (W 0.00351935) | 0.577064 ± 0.00469482 (W 0.00803636) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.20282 ± 0.0054368 (W 0.00983457) | 5.87821 ± 0.228789 (W 0.11754) | - | 🔵 faer-fork 4.89× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1 [api-inclusive]` | 2.02335 ± 0.00675376 (W 0.00615542) | 2.55334 ± 0.0305127 (W 0.0442594) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.52462 ± 0.00600827 (W 0.00541761) | 1.81249 ± 0.00576936 (W 0.00730698) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.52045 ± 0.00750812 (W 0.00575738) | 1.81374 ± 0.00816862 (W 0.00924065) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.85773 ± 0.0135778 (W 0.00907302) | 8.2807 ± 0.0557883 (W 0.0180497) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.87303 ± 0.0156929 (W 0.0103244) | 8.40826 ± 0.058931 (W 0.0666963) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.9664 ± 0.0598803 (W 0.0355305) | 35.3816 ± 0.142963 (W 0.0244564) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 25.0534 ± 0.0204832 (W 0.0316315) | 35.6158 ± 0.0485685 (W 0.0422565) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.69358 ± 0.00233997 (W 0.00262128) | 1.44356 ± 0.0221884 (W 0.0067714) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.71973 ± 0.0027405 (W 0.0030604) | 1.44452 ± 0.00180686 (W 0.00405706) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.98184 ± 0.00483774 (W 0.00831567) | 6.61632 ± 0.0124611 (W 0.0103165) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.0013 ± 0.0092025 (W 0.00969264) | 6.65304 ± 0.0171772 (W 0.0162393) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 12.3261 ± 0.0237921 (W 0.0242966) | 27.7416 ± 0.041771 (W 0.0364103) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 12.3278 ± 0.0607552 (W 0.0139265) | 27.8288 ± 0.0261498 (W 0.0287487) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004558 ± 5e-05 (W 3.59625e-05) | 0.003466 ± 3.19427e-05 (W 8.1867e-05) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.011161 ± 7.63763e-05 (W 4.47214e-05) | 0.008596 ± 1.52753e-05 (W 5.94138e-05) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.02648 ± 4.50925e-05 (W 0.000149229) | 0.022261 ± 0.000224025 (W 0.000945911) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.044283 ± 0.000180334 (W 0.000363723) | 0.095468 ± 0.00176475 (W 0.054749) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.03758 ± 0.000120652 (W 0.00023487) | 0.04863 ± 0.000710541 (W 0.000504683) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.135944 ± 0.00350267 (W 0.00175611) | 0.425512 ± 0.044954 (W 0.0521783) | - | 🔵 faer-fork 3.13× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.165107 ± 0.00380219 (W 0.00213603) | 0.280813 ± 0.00230976 (W 0.00862984) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.565894 ± 0.0107843 (W 0.00834995) | 1.42572 ± 0.036463 (W 0.0683653) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.616788 ± 0.0109329 (W 0.00198399) | 1.07136 ± 0.0134144 (W 0.0452864) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.749505 ± 0.00208774 (W 0.00252813) | 1.65697 ± 0.00219505 (W 0.00355663) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.754687 ± 0.00261726 (W 0.0027253) | 1.67327 ± 0.00228291 (W 0.00560087) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.77121 ± 0.00562642 (W 0.00399519) | 4.50979 ± 0.00790528 (W 0.0094752) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.79537 ± 0.0254825 (W 0.00667829) | 4.5189 ± 0.0206035 (W 0.0198029) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.7142 ± 0.0130636 (W 0.0129898) | 15.1275 ± 0.0286084 (W 0.0620016) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.6573 ± 0.0297202 (W 0.0102585) | 15.0946 ± 0.104136 (W 0.0455177) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.239255 ± 0.00217113 (W 0.00184895) | 0.188121 ± 0.00187915 (W 0.00247825) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.256628 ± 0.00129834 (W 0.00225156) | 0.20372 ± 0.0021047 (W 0.00277112) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.08052 ± 0.00832236 (W 0.00691118) | 1.33438 ± 0.00590106 (W 0.0105952) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.28658 ± 0.0122394 (W 0.00564627) | 1.29317 ± 0.012485 (W 0.00795348) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.74937 ± 0.0176756 (W 0.018567) | 9.71831 ± 0.0188798 (W 0.0229598) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.65703 ± 0.023828 (W 0.0142391) | 8.64781 ± 0.0254213 (W 0.0259174) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.038251 ± 0.000957609 (W 0.000816625) | 0.06954 ± 0.00046608 (W 0.00118374) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.039984 ± 0.0001805 (W 0.00134982) | 0.069159 ± 0.000358376 (W 0.00210869) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.163475 ± 0.00150571 (W 0.00122381) | 0.293135 ± 0.00593052 (W 0.00848562) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.165719 ± 0.00138377 (W 0.0013037) | 0.299588 ± 0.00034222 (W 0.00591784) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.05016 ± 0.0033238 (W 0.00780615) | 1.42942 ± 0.0222056 (W 0.150267) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.04692 ± 0.00236708 (W 0.00699318) | 1.38524 ± 0.137843 (W 0.135596) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.098232 ± 0.00188005 (W 0.00378809) | 0.173153 ± 0.00196492 (W 0.00197044) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.09107 ± 0.00107012 (W 0.000432104) | 0.118049 ± 0.0034862 (W 0.00196332) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.253242 ± 0.00109244 (W 0.00211319) | 1.4654 ± 0.00869327 (W 0.0232206) | - | 🔵 faer-fork 5.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.452683 ± 0.0010771 (W 0.00255332) | 0.594657 ± 0.00804251 (W 0.0128443) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.976058 ± 0.00260628 (W 0.00331693) | 7.22292 ± 0.0291677 (W 0.192182) | - | 🔵 faer-fork 7.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.86501 ± 0.0094003 (W 0.0035411) | 2.81399 ± 0.0785133 (W 0.103035) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.303877 ± 0.00196144 (W 0.00176149) | 0.960569 ± 0.0583035 (W 0.00695398) | - | 🔵 faer-fork 3.16× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.323814 ± 0.0168135 (W 0.00294628) | 0.767238 ± 0.00485812 (W 0.00759687) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 0.669708 ± 0.00157627 (W 0.00300247) | 3.02032 ± 0.0144863 (W 0.0218955) | - | 🔵 faer-fork 4.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.989323 ± 0.00147136 (W 0.00642623) | 2.0242 ± 0.00505474 (W 0.0137324) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.25626 ± 0.0180964 (W 0.00685033) | 12.7984 ± 0.122743 (W 0.0559633) | - | 🔵 faer-fork 5.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.62623 ± 0.0116263 (W 0.00898129) | 6.5566 ± 0.12836 (W 0.186616) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.185916 ± 0.00134945 (W 0.00169848) | 0.77303 ± 0.00322227 (W 0.00762038) | - | 🔵 faer-fork 4.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.507635 ± 0.00305669 (W 0.00253367) | 2.64857 ± 0.0285287 (W 0.0279838) | - | 🔵 faer-fork 5.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.47604 ± 0.0122752 (W 0.00485726) | 11.455 ± 0.103824 (W 0.0403883) | - | 🔵 faer-fork 7.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.082594 ± 0.000596566 (W 0.0016358) | 0.162503 ± 0.000188283 (W 0.00102857) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.099235 ± 0.00136574 (W 0.000642396) | 0.115054 ± 0.000810547 (W 0.00173) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.258682 ± 0.00200784 (W 0.00233256) | 1.03823 ± 0.0090046 (W 0.00916748) | - | 🔵 faer-fork 4.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.458725 ± 0.00109364 (W 0.00281646) | 0.579479 ± 0.0026992 (W 0.005392) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.921366 ± 0.00307356 (W 0.00327666) | 5.87056 ± 0.0292216 (W 0.100753) | - | 🔵 faer-fork 6.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.86014 ± 0.00458888 (W 0.00693394) | 2.59555 ± 0.0228796 (W 0.0818485) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.52048 ± 0.00271304 (W 0.00354409) | 1.96033 ± 0.00225867 (W 0.0110168) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.55353 ± 0.00223087 (W 0.00471812) | 1.97031 ± 0.0069496 (W 0.0100521) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.52952 ± 0.0128951 (W 0.00766945) | 8.63991 ± 0.029711 (W 0.0199651) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.58277 ± 0.0121021 (W 0.0113045) | 8.65971 ± 0.0110775 (W 0.0253354) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 21.7182 ± 0.0965227 (W 0.0533251) | 34.4966 ± 0.0792768 (W 0.0819877) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 21.7491 ± 0.0656754 (W 0.0197926) | 34.6964 ± 0.0340978 (W 0.0967416) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.714992 ± 0.00169325 (W 0.00198573) | 1.44496 ± 0.00356424 (W 0.0038772) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.718708 ± 0.000968133 (W 0.00378893) | 1.44012 ± 0.00209633 (W 0.00533881) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.98504 ± 0.0119012 (W 0.00585551) | 6.69844 ± 0.00230303 (W 0.00916796) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.9968 ± 0.00535849 (W 0.00677381) | 6.72002 ± 0.0139581 (W 0.0228564) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.4218 ± 0.0395599 (W 0.0105524) | 27.1846 ± 0.0297035 (W 0.0478955) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.3876 ± 0.0610904 (W 0.019176) | 27.2346 ± 0.0178017 (W 0.0390123) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004669 ± 5.08626e-05 (W 4.30116e-05) | 0.003517 ± 2.51661e-05 (W 0.000100446) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01108 ± 0.000104245 (W 6.08293e-05) | 0.008575 ± 3.00056e-05 (W 4.28626e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.025928 ± 0.000259422 (W 0.000127237) | 0.02126 ± 0.00047411 (W 0.00041872) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.034504 ± 7e-05 (W 8.61005e-05) | 0.033983 ± 0.000168028 (W 8.27665e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.040004 ± 0.000172975 (W 0.000109015) | 0.040425 ± 0.000285518 (W 0.000164668) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.174264 ± 0.00011682 (W 0.00207914) | 0.174575 ± 0.000341955 (W 0.00189381) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.194752 ± 0.00092016 (W 0.00191404) | 0.199863 ± 0.000544693 (W 0.00212547) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.06948 ± 0.00803859 (W 0.00572524) | 1.0819 ± 0.00135972 (W 0.00260651) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.1583 ± 0.00250984 (W 0.00273711) | 1.17603 ± 0.00187055 (W 0.00398255) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.76024 ± 0.00939412 (W 0.00631841) | 1.66819 ± 0.0107366 (W 0.00316361) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.7459 ± 0.0109547 (W 0.00325811) | 1.66426 ± 0.00923638 (W 0.00609655) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.57938 ± 0.0159356 (W 0.00923034) | 6.26338 ± 0.0249715 (W 0.0146212) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.59737 ± 0.0105712 (W 0.0166382) | 6.2526 ± 0.0177024 (W 0.00986929) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.4007 ± 0.253431 (W 0.0346408) | 28.654 ± 0.130115 (W 0.0269882) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.4657 ± 0.136152 (W 0.0349222) | 28.714 ± 0.230391 (W 0.0424098) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.217395 ± 0.0012315 (W 0.00181895) | 0.183471 ± 0.000812453 (W 0.00202555) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.218368 ± 0.00101737 (W 0.00163717) | 0.195162 ± 0.00189817 (W 0.00202371) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.32275 ± 0.0176451 (W 0.00423377) | 1.26511 ± 0.0148548 (W 0.00822516) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.36894 ± 0.00793838 (W 0.00766128) | 1.28587 ± 0.00136175 (W 0.00632534) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.2275 ± 0.372223 (W 0.0255032) | 11.9042 ± 0.0651386 (W 0.0121616) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.9222 ± 0.0681309 (W 0.0397336) | 11.4195 ± 0.0362841 (W 0.0444344) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.149469 ± 0.000705514 (W 0.000581516) | 0.148216 ± 0.00058769 (W 0.00174695) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.15087 ± 0.000978775 (W 0.0020535) | 0.147996 ± 0.000301244 (W 0.00258585) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.01027 ± 0.000750071 (W 0.00165426) | 1.00518 ± 0.00423213 (W 0.0064601) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.009 ± 0.0020227 (W 0.00324104) | 1.00456 ± 0.00444136 (W 0.000592392) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.51856 ± 0.00731294 (W 0.01958) | 7.5052 ± 0.0238783 (W 0.00943573) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.51884 ± 0.00580022 (W 0.0155017) | 7.50376 ± 0.0092933 (W 0.0371123) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.070672 ± 0.000707838 (W 0.000351539) | 0.076923 ± 0.000545333 (W 0.00038941) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.077234 ± 0.000478456 (W 0.00188047) | 0.086411 ± 0.000534853 (W 0.000334668) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.356605 ± 0.00173564 (W 0.0025222) | 0.375611 ± 0.00193688 (W 0.00446377) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.484263 ± 0.00454398 (W 0.00328427) | 0.519293 ± 0.00262026 (W 0.00271558) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.20417 ± 0.00289028 (W 0.00397546) | 2.50406 ± 0.00892863 (W 0.00638924) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.81299 ± 0.00881763 (W 0.00562265) | 3.13878 ± 0.00603992 (W 0.00997536) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.394115 ± 0.00335875 (W 0.00230033) | 0.433998 ± 0.00148432 (W 0.00215429) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.428048 ± 0.00218008 (W 0.00311293) | 0.468814 ± 0.00211453 (W 0.00164925) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.59415 ± 0.00310053 (W 0.00219541) | 1.72166 ± 0.00503572 (W 0.0022953) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.79936 ± 0.00326972 (W 0.00566884) | 1.95246 ± 0.00613105 (W 0.0084506) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 9.62501 ± 0.0319319 (W 0.0184281) | 10.0574 ± 0.0282275 (W 0.00826664) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 10.5002 ± 0.023488 (W 0.0277389) | 11.0316 ± 0.0079869 (W 0.01981) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.217273 ± 0.00101449 (W 0.00202273) | 0.258029 ± 0.000216228 (W 0.00182556) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.84082 ± 0.00357087 (W 0.00313964) | 0.97523 ± 0.00427536 (W 0.000540242) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 4.38044 ± 0.013252 (W 0.00996172) | 4.81683 ± 0.00418172 (W 0.00562772) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.069379 ± 0.000559185 (W 0.000256876) | 0.075981 ± 0.000482453 (W 0.000203888) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.077003 ± 0.00148199 (W 0.00213797) | 0.084988 ± 0.000476634 (W 0.00192437) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.352354 ± 0.00160791 (W 0.00317254) | 0.361624 ± 0.00249811 (W 0.00267461) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.364045 ± 0.00180985 (W 0.00382264) | 0.401087 ± 0.00181209 (W 0.00309767) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.17276 ± 0.00367765 (W 0.00726684) | 2.43813 ± 0.00143096 (W 0.00606573) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.27756 ± 0.0128037 (W 0.00630313) | 2.57102 ± 0.0107682 (W 0.00842026) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.16284 ± 0.00385187 (W 0.00680293) | 2.31685 ± 0.0288059 (W 0.00396956) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.16947 ± 0.00900878 (W 0.00654944) | 2.32878 ± 0.00930021 (W 0.00752006) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.3208 ± 0.0214702 (W 0.0257393) | 11.1456 ± 0.0247731 (W 0.0157525) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.35552 ± 0.0282648 (W 0.0124702) | 11.1792 ± 0.00804199 (W 0.00773687) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 50.0233 ± 0.178938 (W 0.121763) | 60.5339 ± 0.0230816 (W 0.137629) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 50.03 ± 0.149953 (W 0.0717689) | 60.7397 ± 0.100307 (W 0.103666) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.659057 ± 0.00156516 (W 0.00244752) | 1.66646 ± 0.00350155 (W 0.00201005) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.658197 ± 0.00482956 (W 0.00287933) | 1.66935 ± 0.00838931 (W 0.00917312) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.25631 ± 0.00659831 (W 0.0059939) | 7.68018 ± 0.00779667 (W 0.0119501) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.28518 ± 0.00441469 (W 0.00451932) | 7.74748 ± 0.0237411 (W 0.00918697) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.7246 ± 0.0199517 (W 0.0415509) | 39.6642 ± 0.142234 (W 0.0715126) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.7987 ± 0.0660218 (W 0.020248) | 39.7197 ± 0.13579 (W 0.0345231) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004578 ± 4.59384e-05 (W 3.30651e-05) | 0.004408 ± 3.51188e-05 (W 4.54973e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.010199 ± 0.000122231 (W 0.00010895) | 0.009768 ± 4.16333e-05 (W 4.342e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030748 ± 0.000301532 (W 0.000130753) | 0.029886 ± 9.21973e-05 (W 7.99074e-05) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.030787 ± 0.000137078 (W 0.000110459) | 0.042589 ± 0.00112861 (W 0.00185587) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.036999 ± 0.00022646 (W 0.000245826) | 0.050213 ± 0.00149074 (W 0.00116109) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.113532 ± 0.0013092 (W 0.00237267) | 0.210732 ± 0.00500141 (W 0.00681253) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.134912 ± 0.000741081 (W 0.000372336) | 0.239125 ± 0.00765282 (W 0.00779413) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.436242 ± 0.00301026 (W 0.00408735) | 0.668806 ± 0.0195952 (W 0.0359929) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.53743 ± 0.00111428 (W 0.00272959) | 0.788917 ± 0.0285171 (W 0.0281361) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.03759 ± 0.00598659 (W 0.00202673) | 1.58162 ± 0.001043 (W 0.00399257) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.043 ± 0.00526258 (W 0.00329235) | 1.60394 ± 0.00809519 (W 0.00742519) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.70148 ± 0.00795555 (W 0.00913336) | 4.3227 ± 0.00721179 (W 0.0104844) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.69965 ± 0.00197542 (W 0.00336698) | 4.33669 ± 0.00977795 (W 0.0152739) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.1477 ± 0.0107704 (W 0.0164839) | 15.1433 ± 0.079044 (W 0.0155798) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.1407 ± 0.00921483 (W 0.0155987) | 15.0384 ± 0.0278922 (W 0.0163919) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.222584 ± 0.000811772 (W 0.00157372) | 0.187149 ± 0.000823947 (W 0.00224182) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.221772 ± 0.00316014 (W 0.00157621) | 0.195384 ± 0.000205816 (W 0.00179875) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.07524 ± 0.00977432 (W 0.00671657) | 1.27073 ± 0.0073717 (W 0.00622622) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.1764 ± 0.00763928 (W 0.00628008) | 1.19177 ± 0.00724888 (W 0.00769085) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.38574 ± 0.0313198 (W 0.0219295) | 8.78795 ± 0.035067 (W 0.0300488) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.7711 ± 0.0107934 (W 0.0225048) | 8.4943 ± 0.024269 (W 0.0194707) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.054572 ± 0.000268075 (W 0.00116922) | 0.052979 ± 0.000298719 (W 0.000357151) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.054371 ± 0.000524215 (W 0.000878359) | 0.051917 ± 0.000335604 (W 0.000176361) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.288076 ± 0.000827698 (W 0.00211931) | 0.28475 ± 0.00199887 (W 0.00180383) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.28968 ± 0.000912785 (W 0.00178387) | 0.284028 ± 0.00162872 (W 0.00154697) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.95456 ± 0.00468958 (W 0.00453978) | 1.94926 ± 0.00778736 (W 0.00479793) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.95425 ± 0.00380032 (W 0.00796111) | 1.95545 ± 0.00755365 (W 0.00579044) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.060192 ± 0.000431073 (W 0.00167552) | 0.099085 ± 0.00224019 (W 0.00157162) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.065802 ± 0.00138963 (W 0.000685916) | 0.112559 ± 0.00156536 (W 0.00093201) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.219559 ± 0.0023676 (W 0.00326828) | 0.445149 ± 0.00472998 (W 0.00811769) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.360482 ± 0.00117479 (W 0.00265538) | 0.492887 ± 0.00579867 (W 0.00790939) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.991726 ± 0.00368852 (W 0.00452134) | 1.90362 ± 0.0448823 (W 0.055634) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.61265 ± 0.00631099 (W 0.00764689) | 2.01962 ± 0.0455328 (W 0.0482868) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.264252 ± 0.00196613 (W 0.00214518) | 0.512013 ± 0.00187178 (W 0.00426786) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.30091 ± 0.00148497 (W 0.00285271) | 0.554533 ± 0.00268278 (W 0.00301236) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.702777 ± 0.00345937 (W 0.00222389) | 1.41783 ± 0.00474173 (W 0.00474674) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.937936 ± 0.00314956 (W 0.00264238) | 1.59781 ± 0.0391056 (W 0.00539364) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.39732 ± 0.00436239 (W 0.00703515) | 5.00118 ± 0.0131028 (W 0.00817894) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.37797 ± 0.021508 (W 0.0105389) | 5.96751 ± 0.0118475 (W 0.0132968) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.178072 ± 0.000471233 (W 0.00220339) | 0.410174 ± 0.000534716 (W 0.00369757) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.463032 ± 0.000366544 (W 0.00273419) | 1.18448 ± 0.0100291 (W 0.00389237) | - | 🔵 faer-fork 2.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.98883 ± 0.0109752 (W 0.00483809) | 3.53358 ± 0.0106177 (W 0.00614566) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.063959 ± 0.000538022 (W 0.000443822) | 0.096008 ± 0.000483208 (W 0.00130716) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.071173 ± 0.0003105 (W 0.000640818) | 0.10765 ± 0.000455395 (W 0.00228429) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.23113 ± 0.00190823 (W 0.00169514) | 0.420623 ± 0.00531927 (W 0.00557702) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.35994 ± 0.00287992 (W 0.00291307) | 0.467289 ± 0.00520885 (W 0.00710131) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.945139 ± 0.00429259 (W 0.0059292) | 1.73785 ± 0.0439481 (W 0.044635) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.59956 ± 0.00497248 (W 0.00322101) | 1.97764 ± 0.0253013 (W 0.0484) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.5225 ± 0.0079455 (W 0.00387072) | 1.82654 ± 0.00484636 (W 0.00542744) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.52135 ± 0.0043512 (W 0.00305642) | 1.82837 ± 0.00792865 (W 0.0090037) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.81229 ± 0.0100382 (W 0.0098247) | 8.33955 ± 0.0308937 (W 0.0614969) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.82175 ± 0.0344331 (W 0.0129426) | 8.42729 ± 0.0929932 (W 0.0669801) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.4969 ± 0.0236957 (W 0.0512808) | 34.8804 ± 0.0806523 (W 0.0336018) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 24.5186 ± 0.0549701 (W 0.0405002) | 35.0135 ± 0.00964675 (W 0.0439023) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.689353 ± 0.00227056 (W 0.00255374) | 1.40453 ± 0.00296095 (W 0.00381349) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.712146 ± 0.0032146 (W 0.00265544) | 1.4088 ± 0.0009925 (W 0.00581927) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.89203 ± 0.00418083 (W 0.0048971) | 6.43451 ± 0.0142374 (W 0.00995611) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.91153 ± 0.00732952 (W 0.00730215) | 6.49386 ± 0.0217284 (W 0.0209022) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.8596 ± 0.0342459 (W 0.0263021) | 27.2441 ± 0.00587816 (W 0.0251402) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.8618 ± 0.0327097 (W 0.0257928) | 27.3103 ± 0.0314471 (W 0.0541782) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00519 ± 8.59089e-05 (W 6.57822e-05) | 0.004148 ± 2.12211e-05 (W 2.26561e-05) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.012022 ± 0.000164794 (W 6.23161e-05) | 0.009868 ± 7.80833e-05 (W 4.78456e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.03221 ± 0.000528291 (W 0.000354874) | 0.028583 ± 0.000356699 (W 0.000404347) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.030767 ± 0.00171461 (W 0.000121816) | 0.041798 ± 0.000761701 (W 0.00044711) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.0371 ± 0.00010602 (W 0.000586432) | 0.048581 ± 0.000449778 (W 0.000842023) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.10244 ± 0.00350265 (W 0.00120857) | 0.226082 ± 0.000834662 (W 0.0038283) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.130884 ± 0.00052823 (W 0.000299946) | 0.25264 ± 0.00144159 (W 0.00371216) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.32741 ± 0.00326141 (W 0.0017568) | 0.75138 ± 0.0244383 (W 0.0681303) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.45105 ± 0.00467859 (W 0.00315775) | 0.854411 ± 0.0157125 (W 0.0466122) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.750899 ± 0.00148386 (W 0.00292158) | 1.65581 ± 0.00198022 (W 0.00457529) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.748885 ± 0.000248818 (W 0.00193386) | 1.6721 ± 0.00162959 (W 0.00727488) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.75889 ± 0.00709538 (W 0.00567515) | 4.46653 ± 0.149711 (W 0.011734) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1 [api-inclusive]` | 2.798 ± 0.03249 (W 0.0120468) | 4.48047 ± 0.0228649 (W 0.0302172) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.5843 ± 0.0167046 (W 0.0182085) | 14.9182 ± 0.0849598 (W 0.0348576) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.5526 ± 0.0143585 (W 0.016427) | 14.9115 ± 0.083436 (W 0.0417926) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.231732 ± 0.0012544 (W 0.00135013) | 0.187228 ± 0.00167622 (W 0.00240867) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.230279 ± 0.00203297 (W 0.00158447) | 0.194572 ± 0.000511861 (W 0.00209188) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.06193 ± 0.00524536 (W 0.00740121) | 1.27201 ± 0.010071 (W 0.0095204) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.18432 ± 0.0103851 (W 0.00389777) | 1.18395 ± 0.0056177 (W 0.00742828) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.74784 ± 0.021586 (W 0.0202005) | 8.36644 ± 0.0106613 (W 0.0157541) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.28526 ± 0.0131238 (W 0.0280348) | 8.06954 ± 0.0540769 (W 0.0297704) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.037951 ± 8.14453e-05 (W 0.00115833) | 0.072255 ± 0.000244759 (W 0.00199821) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.038011 ± 0.000383789 (W 0.000489408) | 0.072675 ± 0.000601954 (W 0.00162498) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.173783 ± 0.00117422 (W 0.00232724) | 0.312031 ± 0.00277802 (W 0.00520174) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.172662 ± 0.000772543 (W 0.00138225) | 0.307283 ± 0.000669961 (W 0.00714563) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.08473 ± 0.00480925 (W 0.00386577) | 1.44754 ± 0.00821313 (W 0.109047) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.08658 ± 0.00156073 (W 0.00486191) | 1.44087 ± 0.115461 (W 0.0878155) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.059681 ± 0.000510216 (W 0.00191509) | 0.100427 ± 0.000930738 (W 0.000356266) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.067476 ± 0.000388562 (W 0.000783884) | 0.111147 ± 0.000673381 (W 0.000882627) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.209571 ± 0.00101474 (W 0.00235152) | 0.452623 ± 0.00403096 (W 0.00961223) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.35436 ± 0.00350123 (W 0.00626911) | 0.498449 ± 0.00357833 (W 0.00812452) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.83262 ± 0.00430472 (W 0.00352049) | 1.90942 ± 0.0707202 (W 0.0939156) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 1.44549 ± 0.0132752 (W 0.00564879) | 2.03116 ± 0.100975 (W 0.0981185) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.247241 ± 0.000579633 (W 0.00220059) | 0.748844 ± 0.00663279 (W 0.00694206) | - | 🔵 faer-fork 3.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.291643 ± 0.00205456 (W 0.00366699) | 0.764645 ± 0.00302757 (W 0.0121997) | - | 🔵 faer-fork 2.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.618252 ± 0.00250808 (W 0.00315193) | 1.85843 ± 0.011781 (W 0.00427847) | - | 🔵 faer-fork 3.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.865814 ± 0.00276647 (W 0.00372992) | 1.93831 ± 0.0176625 (W 0.0125042) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.12711 ± 0.00437475 (W 0.00678904) | 5.67484 ± 0.0059137 (W 0.0247) | - | 🔵 faer-fork 2.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.13669 ± 0.0166786 (W 0.00798477) | 5.93308 ± 0.0130952 (W 0.0238127) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.174765 ± 0.00017657 (W 0.00200881) | 0.559182 ± 0.00752879 (W 0.00407749) | - | 🔵 faer-fork 3.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.464154 ± 0.000678081 (W 0.00268371) | 1.46495 ± 0.0263342 (W 0.00615026) | - | 🔵 faer-fork 3.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.33325 ± 0.00133353 (W 0.00322417) | 4.1877 ± 0.0243474 (W 0.0118878) | - | 🔵 faer-fork 3.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.064751 ± 7.83603e-05 (W 0.000908331) | 0.095508 ± 0.00020694 (W 0.000475512) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.072435 ± 0.00131116 (W 0.000574527) | 0.106688 ± 0.000809741 (W 0.00126497) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.220671 ± 0.00134838 (W 0.00287215) | 0.426996 ± 0.00368383 (W 0.00543857) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.355512 ± 0.0018749 (W 0.00201701) | 0.469424 ± 0.00119394 (W 0.0109939) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.801854 ± 0.00528254 (W 0.00509764) | 1.85779 ± 0.0810055 (W 0.0926289) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1 [api-inclusive]` | 1.43102 ± 0.00883177 (W 0.008349) | 2.07813 ± 0.0713326 (W 0.096493) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 1.52296 ± 0.00240828 (W 0.00261122) | 1.97253 ± 0.00425409 (W 0.00959642) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.55976 ± 0.00347824 (W 0.0036379) | 1.97255 ± 0.00904559 (W 0.0135219) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.46643 ± 0.0197535 (W 0.00684024) | 8.62063 ± 0.0203908 (W 0.0369686) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.49884 ± 0.00996469 (W 0.00894161) | 8.63479 ± 0.0329747 (W 0.0177427) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 21.289 ± 0.0321804 (W 0.0137781) | 34.1724 ± 0.0489387 (W 0.0296752) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 21.3767 ± 0.0149296 (W 0.0269228) | 34.1714 ± 0.0417628 (W 0.0797893) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.709922 ± 0.000439969 (W 0.00204998) | 1.40478 ± 0.00564353 (W 0.00160185) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.713158 ± 0.00106861 (W 0.00184722) | 1.41188 ± 0.00283208 (W 0.00611306) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.89899 ± 0.0143215 (W 0.0115321) | 6.51778 ± 0.0108525 (W 0.0082151) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.88603 ± 0.00467325 (W 0.00532678) | 6.51741 ± 0.0199032 (W 0.0154705) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 10.9215 ± 0.0215232 (W 0.00912436) | 26.6768 ± 0.107242 (W 0.0351287) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 10.895 ± 0.0396539 (W 0.0186752) | 26.6601 ± 0.0640582 (W 0.028799) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005089 ± 3.49333e-05 (W 3.15801e-05) | 0.004098 ± 2.51661e-05 (W 3.28527e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.011962 ± 0.000123423 (W 5.9576e-05) | 0.009748 ± 7.23418e-05 (W 3.96232e-05) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030908 ± 5.52298e-05 (W 0.000252062) | 0.027531 ± 0.000610433 (W 0.000403446) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.213486 ± 0.00043724 (W 0.00147326) | 0.850799 ± 0.000698974 (W 0.00225188) | - | 🔵 faer-fork 3.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.209219 ± 1.63707e-05 (W 0.00191911) | 0.209559 ± 0.000257891 (W 0.00103755) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.902744 ± 0.00181678 (W 0.00300973) | 2.14543 ± 0.00167401 (W 0.00574946) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.884082 ± 0.00104141 (W 0.00675828) | 0.903465 ± 0.00400723 (W 0.00302079) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.82661 ± 0.0220772 (W 0.0101829) | 7.32723 ± 0.0214391 (W 0.0096576) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.71335 ± 0.0140525 (W 0.0162965) | 4.98068 ± 0.0160169 (W 0.00995623) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.31148 ± 0.00398809 (W 0.00472492) | 2.19339 ± 0.00577933 (W 0.00479283) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.30714 ± 0.00777956 (W 0.00520743) | 2.19543 ± 0.00862088 (W 0.00370712) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.8943 ± 0.0301803 (W 0.0127801) | 10.605 ± 0.0146301 (W 0.0149732) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.9071 ± 0.0142459 (W 0.0131768) | 10.6095 ± 0.028416 (W 0.0296288) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.7761 ± 0.107777 (W 0.07867) | 58.9924 ± 0.0114938 (W 0.0872341) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.9593 ± 0.132438 (W 0.129178) | 59.1526 ± 0.044738 (W 0.0687119) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.457469 ± 0.00145894 (W 0.00291612) | 0.364617 ± 0.00172467 (W 0.00269049) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.436323 ± 0.00355521 (W 0.00423732) | 0.384948 ± 0.00208008 (W 0.00189255) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.57019 ± 0.00703026 (W 0.0103204) | 3.48952 ± 0.223559 (W 0.00669494) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.7248 ± 0.00836724 (W 0.00918467) | 3.51529 ± 0.00571694 (W 0.0155689) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.2661 ± 0.0182111 (W 0.0309512) | 27.0223 ± 1.55048 (W 0.03409) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 26.2344 ± 0.0306546 (W 0.0324588) | 26.1629 ± 0.137557 (W 0.0241034) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.56419 ± 0.00287281 (W 0.00348994) | 0.562415 ± 0.000456623 (W 0.00503988) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.567384 ± 0.00160741 (W 0.00399043) | 0.561635 ± 0.000391858 (W 0.0030471) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.8955 ± 0.0130158 (W 0.00828273) | 3.90945 ± 0.0156312 (W 0.00508743) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.90094 ± 0.00913298 (W 0.00704837) | 3.89795 ± 0.00221268 (W 0.00777015) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 29.053 ± 0.134642 (W 0.0315057) | 28.984 ± 0.0294839 (W 0.0456429) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 28.9863 ± 0.0733082 (W 0.0399908) | 28.9325 ± 0.0133254 (W 0.0682565) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.17724 ± 0.000181853 (W 0.00170022) | 0.315816 ± 0.00190477 (W 0.00259888) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.208269 ± 0.000623628 (W 0.00212066) | 0.19862 ± 0.00137295 (W 0.00322072) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.03805 ± 0.00307762 (W 0.00247614) | 2.91784 ± 0.00369724 (W 0.0061274) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.47031 ± 0.00433871 (W 0.00543303) | 1.46682 ± 0.00586972 (W 0.00370668) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.06946 ± 0.0226544 (W 0.0117106) | 22.9722 ± 0.0501944 (W 0.0382815) | - | 🔵 faer-fork 3.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 9.19828 ± 0.00912108 (W 0.0254832) | 9.68164 ± 0.0974294 (W 0.0174512) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.875299 ± 0.00379179 (W 0.000847839) | 1.23317 ± 0.0127164 (W 0.0038099) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.99712 ± 0.00234779 (W 0.00282731) | 1.03444 ± 0.00431134 (W 0.00283553) | - | · | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.60747 ± 0.0116366 (W 0.0137779) | 6.74766 ± 0.00554592 (W 0.0263846) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.86759 ± 0.0234588 (W 0.0146097) | 6.04992 ± 0.0200589 (W 0.0154337) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 36.436 ± 0.0630893 (W 0.0582006) | 44.9988 ± 0.0779105 (W 0.0538338) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 38.1907 ± 0.192487 (W 0.122964) | 39.2441 ± 0.13533 (W 0.0970698) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.432834 ± 0.00122756 (W 0.0023316) | 0.767438 ± 0.00251871 (W 0.00343697) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.66541 ± 0.0208104 (W 0.00561629) | 3.82898 ± 0.00653871 (W 0.00493218) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 16.0762 ± 0.0436019 (W 0.0275062) | 24.4232 ± 0.0577415 (W 0.0244595) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.171069 ± 0.000212933 (W 0.00143496) | 0.243793 ± 0.00311106 (W 0.00396193) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.208287 ± 0.000713111 (W 0.00168875) | 0.19875 ± 0.0017004 (W 0.0043127) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.00877 ± 0.00241916 (W 0.0019349) | 2.26667 ± 0.00502373 (W 0.00994983) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.20161 ± 0.00142714 (W 0.00426092) | 1.20415 ± 0.00797865 (W 0.0054599) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 6.92319 ± 0.0162017 (W 0.00914256) | 16.5175 ± 0.107214 (W 0.0295095) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.72629 ± 0.0216577 (W 0.00994169) | 8.23733 ± 0.0447191 (W 0.0153056) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.15381 ± 0.0116775 (W 0.00823481) | 3.36437 ± 0.0161636 (W 0.00649832) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.1711 ± 0.00723185 (W 0.00853294) | 3.35484 ± 0.00249781 (W 0.00669431) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 16.8703 ± 0.0349367 (W 0.0175917) | 19.3764 ± 0.0614112 (W 0.0251494) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 16.8915 ± 0.135905 (W 0.0373756) | 19.3816 ± 0.0322082 (W 0.033326) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 106.028 ± 0.110423 (W 0.159519) | 120.601 ± 0.104717 (W 0.122775) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 106.187 ± 0.0991226 (W 0.0662209) | 121.256 ± 0.29796 (W 0.208607) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.06556 ± 0.00388473 (W 0.00392065) | 2.12942 ± 0.00343709 (W 0.00779051) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.06573 ± 0.00215504 (W 0.00542564) | 2.14421 ± 0.00844748 (W 0.00819062) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.3638 ± 0.0480998 (W 0.0199047) | 11.4475 ± 0.0230926 (W 0.029537) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.29949 ± 0.0208479 (W 0.0239535) | 11.5057 ± 0.0133078 (W 0.0144032) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 44.1127 ± 0.141877 (W 0.0569102) | 68.1609 ± 0.129223 (W 0.12039) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 44.3894 ± 0.0260501 (W 0.119266) | 68.6017 ± 0.112112 (W 0.140893) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.006211 ± 1.00167e-05 (W 9.50489e-05) | 0.005741 ± 2.64575e-05 (W 5.08547e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017031 ± 0.000108029 (W 0.000102567) | 0.01593 ± 0.000141846 (W 6.46003e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.05394 ± 0.000225227 (W 0.000444828) | 0.052106 ± 0.000800652 (W 0.00033981) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.19365 ± 0.000729287 (W 0.00177733) | 0.547208 ± 0.299654 (W 0.163074) | - | 🔵 faer-fork 2.83× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.195304 ± 0.000349515 (W 0.000863297) | 0.211785 ± 0.000794718 (W 0.00156134) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.608101 ± 0.0340833 (W 0.00389807) | 1.62827 ± 0.170053 (W 0.156265) | - | 🔵 faer-fork 2.68× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.618952 ± 0.00209808 (W 0.00349705) | 0.718647 ± 0.00864185 (W 0.0146999) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.11251 ± 0.00530698 (W 0.00837064) | 4.08208 ± 0.202151 (W 0.271754) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.15242 ± 0.0153823 (W 0.00299319) | 2.63154 ± 0.050203 (W 0.065284) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32631 ± 0.00209699 (W 0.00301988) | 1.92824 ± 0.00745166 (W 0.00373642) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.33266 ± 0.00460581 (W 0.00207624) | 1.94538 ± 0.00414717 (W 0.00439468) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.31632 ± 0.00592294 (W 0.0113801) | 7.12056 ± 0.105801 (W 0.0962321) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.34044 ± 0.0200413 (W 0.0191734) | 7.16364 ± 0.111767 (W 0.0993383) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.7096 ± 0.0551551 (W 0.0167206) | 26.9157 ± 0.025333 (W 0.0223825) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.7763 ± 0.0326856 (W 0.0236361) | 27.0319 ± 0.00685139 (W 0.0509534) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.371161 ± 0.00211041 (W 0.0038094) | 0.367143 ± 0.000323074 (W 0.00234468) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.423258 ± 0.00160652 (W 0.00348269) | 0.386318 ± 0.00229285 (W 0.00324877) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.19188 ± 0.00872077 (W 0.00590214) | 3.4833 ± 0.00539963 (W 0.00462132) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.48558 ± 0.00535601 (W 0.00916067) | 3.30544 ± 0.0152766 (W 0.00563334) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.591 ± 0.0651426 (W 0.0240635) | 18.7809 ± 0.0222651 (W 0.0192275) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 14.2356 ± 0.0208001 (W 0.0197219) | 17.2557 ± 0.23102 (W 0.0248726) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.160469 ± 0.000929702 (W 0.00145795) | 0.157774 ± 0.000661429 (W 0.00188623) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.159367 ± 0.00175549 (W 0.00161696) | 0.157684 ± 0.000488713 (W 0.000718023) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00389 ± 0.00149119 (W 0.00136378) | 1.00382 ± 0.00119581 (W 0.00166869) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00339 ± 0.0015601 (W 0.00317704) | 1.00522 ± 0.00220355 (W 0.00278871) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.37981 ± 0.0152881 (W 0.00930325) | 7.32869 ± 0.00452855 (W 0.0123576) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.37603 ± 0.0107665 (W 0.00721931) | 7.34694 ± 0.0135169 (W 0.0107898) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.168294 ± 0.00112858 (W 0.00236066) | 0.31701 ± 0.00174637 (W 0.00164211) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.215701 ± 0.000299595 (W 0.00223796) | 0.204079 ± 0.00147599 (W 0.00227834) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.642626 ± 0.0060949 (W 0.00526895) | 2.44482 ± 0.0526192 (W 0.0510362) | - | 🔵 faer-fork 3.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.961901 ± 0.00194036 (W 0.00439415) | 0.919242 ± 0.00904875 (W 0.00266093) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.0625 ± 0.00166762 (W 0.0119269) | 14.213 ± 0.44042 (W 0.334144) | - | 🔵 faer-fork 4.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 4.80927 ± 0.0232348 (W 0.0140315) | 4.62241 ± 0.0539583 (W 0.0439318) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.489091 ± 0.0031359 (W 0.00292364) | 1.02933 ± 0.00519945 (W 0.00196425) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.579448 ± 0.00262329 (W 0.00235639) | 0.818304 ± 0.00690262 (W 0.0032656) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.82144 ± 0.0033187 (W 0.0020426) | 4.08933 ± 0.0031711 (W 0.0146644) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.33109 ± 0.0145816 (W 0.00765336) | 3.19204 ± 0.00695281 (W 0.0106039) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.7004 ± 0.0401153 (W 0.00950313) | 22.4456 ± 0.0329713 (W 0.0569402) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 13.3521 ± 0.0438169 (W 0.0464269) | 15.493 ± 0.09744 (W 0.020488) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.313403 ± 0.00144663 (W 0.00144157) | 0.826268 ± 0.00468607 (W 0.00351696) | - | 🔵 faer-fork 2.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.03047 ± 0.00159825 (W 0.00245923) | 3.28831 ± 0.00955478 (W 0.0119959) | - | 🔵 faer-fork 3.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.45119 ± 0.00687215 (W 0.0157131) | 17.13 ± 0.059135 (W 0.0523555) | - | 🔵 faer-fork 3.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.145821 ± 0.000395509 (W 0.00192035) | 0.261948 ± 0.000773202 (W 0.00270376) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.178582 ± 0.00124131 (W 0.00239898) | 0.201926 ± 0.00256107 (W 0.0018955) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.618161 ± 0.00358309 (W 0.00189384) | 1.82945 ± 0.00853825 (W 0.0161501) | - | 🔵 faer-fork 2.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.963032 ± 0.00140086 (W 0.00342099) | 0.870211 ± 0.0029229 (W 0.00366509) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.90435 ± 0.00427478 (W 0.00726486) | 10.0992 ± 0.136108 (W 0.173462) | - | 🔵 faer-fork 3.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.70196 ± 0.018214 (W 0.0102434) | 4.44018 ± 0.0936902 (W 0.0545194) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.98026 ± 0.0018398 (W 0.00347898) | 2.48355 ± 0.00617197 (W 0.00744252) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.98692 ± 0.00706273 (W 0.00354177) | 2.48578 ± 0.00297147 (W 0.0062689) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.38552 ± 0.0298721 (W 0.0195068) | 13.0389 ± 0.0428513 (W 0.0633698) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.42624 ± 0.0221168 (W 0.0149709) | 13.1196 ± 0.0442926 (W 0.0705081) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 40.9863 ± 0.0113947 (W 0.049403) | 57.5297 ± 0.187174 (W 0.0749658) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 41.0969 ± 0.0834406 (W 0.0405724) | 57.8191 ± 0.386804 (W 0.481839) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.956131 ± 0.00363573 (W 0.00262497) | 1.84537 ± 0.00725059 (W 0.00127905) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.958294 ± 0.00586002 (W 0.00270827) | 1.85407 ± 0.00533864 (W 0.00323092) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.13605 ± 0.0142792 (W 0.0109181) | 9.84244 ± 0.0336984 (W 0.0162614) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.15452 ± 0.014716 (W 0.0113627) | 9.89811 ± 0.0103868 (W 0.0133342) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.7658 ± 0.0780138 (W 0.0492725) | 41.2677 ± 0.0625219 (W 0.0504072) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.811 ± 0.062903 (W 0.0469752) | 41.2368 ± 0.0972714 (W 0.0695821) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007324 ± 2.51661e-05 (W 4.92768e-05) | 0.005751 ± 6.08276e-06 (W 4.1919e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017092 ± 9.60902e-05 (W 8.13652e-05) | 0.01607 ± 2.51661e-05 (W 0.000419008) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.040916 ± 0.000170098 (W 0.00019216) | 0.039624 ± 0.000526048 (W 0.00109389) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.191948 ± 0.000394666 (W 0.00160618) | 0.244445 ± 0.000230717 (W 0.00152664) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.194953 ± 0.000575578 (W 0.00157859) | 0.205453 ± 9.07028e-05 (W 0.0014739) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.581403 ± 0.00402117 (W 0.00318082) | 1.09537 ± 0.00371736 (W 0.105402) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.596801 ± 4.30155e-05 (W 0.00246534) | 0.741923 ± 0.0215517 (W 0.0256617) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.7364 ± 0.00364166 (W 0.00394984) | 3.15236 ± 0.147317 (W 0.108977) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.83691 ± 0.00422977 (W 0.00303517) | 2.36443 ± 0.0269632 (W 0.0484251) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.991065 ± 0.00700634 (W 0.0029286) | 1.94741 ± 0.00177705 (W 0.0037473) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.999942 ± 0.00396786 (W 0.00259681) | 1.959 ± 0.0103698 (W 0.00552271) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 4.00431 ± 0.00621989 (W 0.00394413) | 6.89624 ± 0.0234712 (W 0.104492) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 4.02562 ± 0.00743821 (W 0.0061193) | 6.90921 ± 0.0181637 (W 0.110943) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.8472 ± 0.0273192 (W 0.0144317) | 24.189 ± 0.0144437 (W 0.113213) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.8961 ± 0.0732224 (W 0.0208445) | 24.3581 ± 0.0300581 (W 0.192929) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.369969 ± 0.000932031 (W 0.00247224) | 0.366913 ± 0.00273991 (W 0.00248841) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.393362 ± 0.00138831 (W 0.0029066) | 0.383946 ± 0.00197795 (W 0.00297768) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.02207 ± 0.00643932 (W 0.00908769) | 3.46105 ± 0.00858501 (W 0.00617519) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.38259 ± 0.00825979 (W 0.00751505) | 3.29164 ± 0.0182623 (W 0.0101537) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.8258 ± 0.0493087 (W 0.0626127) | 17.5968 ± 0.0758492 (W 0.0182699) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.7358 ± 0.104865 (W 0.0813123) | 16.0446 ± 0.0280191 (W 0.0302591) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.10228 ± 0.00122797 (W 0.00100965) | 0.171229 ± 0.00169701 (W 0.00169296) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.103684 ± 0.000275621 (W 0.000974197) | 0.17198 ± 0.00112002 (W 0.00243697) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.531731 ± 0.00384316 (W 0.00372389) | 0.722766 ± 0.0397249 (W 0.0392702) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.532491 ± 0.00151681 (W 0.00252034) | 0.726914 ± 0.0395216 (W 0.0255803) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.84778 ± 0.00591312 (W 0.0098732) | 3.95547 ± 0.653039 (W 0.730556) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.86489 ± 0.00720002 (W 0.00698692) | 3.97634 ± 0.604577 (W 0.533631) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.178994 ± 0.000721287 (W 0.000860583) | 0.314866 ± 0.0028469 (W 0.00243573) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.172862 ± 0.00115117 (W 0.00252835) | 0.19838 ± 0.000441604 (W 0.0030521) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.544755 ± 0.00167619 (W 0.00343917) | 2.38897 ± 0.0164806 (W 0.0262715) | - | 🔵 faer-fork 4.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.905105 ± 0.00340364 (W 0.00374159) | 0.948608 ± 0.0217311 (W 0.0224802) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.19151 ± 0.0066413 (W 0.00498975) | 12.1089 ± 0.214319 (W 0.259373) | - | 🔵 faer-fork 5.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.14481 ± 0.0122664 (W 0.0136965) | 4.26516 ± 0.0595567 (W 0.049119) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.415554 ± 0.00127479 (W 0.00319703) | 1.11319 ± 0.00977643 (W 0.0154876) | - | 🔵 faer-fork 2.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.520008 ± 0.00299246 (W 0.00346469) | 0.946203 ± 0.00383862 (W 0.00437523) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.34421 ± 0.000897952 (W 0.00423837) | 4.24339 ± 0.014256 (W 0.01786) | - | 🔵 faer-fork 3.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.91914 ± 0.0850923 (W 0.00593781) | 3.08089 ± 0.209693 (W 0.0722744) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 5.86064 ± 0.0115364 (W 0.00770785) | 20.8923 ± 0.106251 (W 0.130825) | - | 🔵 faer-fork 3.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.68055 ± 0.1428 (W 0.0276436) | 12.022 ± 0.112438 (W 0.0934163) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.282486 ± 0.000394188 (W 0.00212632) | 0.878696 ± 0.00611353 (W 0.0189097) | - | 🔵 faer-fork 3.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.915765 ± 0.00517094 (W 0.00570185) | 3.49182 ± 0.0137011 (W 0.0231125) | - | 🔵 faer-fork 3.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 3.04501 ± 0.0793212 (W 0.00848673) | 16.6881 ± 0.093221 (W 0.0748963) | - | 🔵 faer-fork 5.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.151823 ± 0.000989527 (W 0.00166435) | 0.253081 ± 0.00105204 (W 0.00198768) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.175968 ± 0.000557415 (W 0.0018988) | 0.196686 ± 0.00121386 (W 0.00275698) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.524337 ± 0.00090006 (W 0.00263103) | 1.82203 ± 0.0273325 (W 0.0265072) | - | 🔵 faer-fork 3.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.908111 ± 0.00363767 (W 0.00441462) | 0.886992 ± 0.00961463 (W 0.0218682) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.07092 ± 0.00750691 (W 0.0104814) | 9.47311 ± 0.0478197 (W 0.0818894) | - | 🔵 faer-fork 4.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.07615 ± 0.00374552 (W 0.0138131) | 3.94908 ± 0.0487514 (W 0.0349029) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.92372 ± 0.00456602 (W 0.00306374) | 2.52117 ± 0.00464838 (W 0.00747901) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.9239 ± 0.00477928 (W 0.00282732) | 2.54283 ± 0.0149945 (W 0.0101281) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.57311 ± 0.00129859 (W 0.0504071) | 12.9639 ± 0.0173634 (W 0.0254273) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.57505 ± 0.0199759 (W 0.00742154) | 13.0191 ± 0.0226158 (W 0.0516097) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 32.3283 ± 0.480872 (W 0.0336308) | 52.5508 ± 0.154907 (W 0.212014) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 32.3639 ± 0.0186549 (W 0.0448587) | 52.5203 ± 0.0685583 (W 0.13031) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.990967 ± 0.0016405 (W 0.00244859) | 1.85767 ± 0.0056341 (W 0.0286358) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.990947 ± 0.00442811 (W 0.00354773) | 1.86322 ± 0.002148 (W 0.00678251) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.08663 ± 0.00505533 (W 0.00927761) | 9.9644 ± 0.0242314 (W 0.0215347) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.10638 ± 0.00826956 (W 0.0072634) | 9.9665 ± 0.0415751 (W 0.0188312) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.1323 ± 0.0158684 (W 0.0302482) | 39.2474 ± 0.036916 (W 0.0623469) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 17.1618 ± 0.0098329 (W 0.0239611) | 39.2869 ± 0.0644863 (W 0.0468172) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007314 ± 4.00042e-05 (W 4.77525e-05) | 0.005741 ± 3.05505e-05 (W 4.08803e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.016992 ± 2.13776e-05 (W 0.000100421) | 0.01601 ± 0.000105987 (W 6.6748e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.038802 ± 0.000222419 (W 0.000108601) | 0.038131 ± 0.000802464 (W 0.000208196) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.196705 ± 0.000929002 (W 0.00195365) | 0.197937 ± 0.000145194 (W 0.00162801) | - | · | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.203108 ± 0.000587001 (W 0.00195828) | 0.203049 ± 0.000900002 (W 0.00215373) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.806602 ± 0.00300277 (W 0.00220128) | 0.815855 ± 0.00404025 (W 0.00267203) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.834243 ± 0.000801003 (W 0.00281504) | 0.836457 ± 0.00258239 (W 0.00302723) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.38286 ± 0.0126259 (W 0.0123298) | 4.43575 ± 0.00586157 (W 0.00974973) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.51939 ± 0.0078707 (W 0.00896222) | 4.51593 ± 0.0154207 (W 0.0121075) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.29768 ± 0.00159092 (W 0.0077978) | 2.18999 ± 0.00695081 (W 0.00350525) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.30261 ± 0.00183744 (W 0.00499396) | 2.18737 ± 0.00288563 (W 0.00645478) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.8507 ± 0.0207708 (W 0.0247701) | 10.5281 ± 0.0271259 (W 0.0225241) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.8514 ± 0.0353781 (W 0.00912322) | 10.5749 ± 0.0138535 (W 0.0255674) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.4926 ± 0.14152 (W 0.0469849) | 58.662 ± 0.166392 (W 0.10241) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.4883 ± 0.155866 (W 0.0638434) | 58.8221 ± 0.0494058 (W 0.178558) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.404735 ± 0.00113634 (W 0.00274658) | 0.367494 ± 0.0169482 (W 0.00214626) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.401098 ± 0.0381929 (W 0.00329458) | 0.360261 ± 0.00193697 (W 0.00252666) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 3.39075 ± 0.00595449 (W 0.0130392) | 3.20318 ± 0.0102324 (W 0.00927657) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.60105 ± 0.00611909 (W 0.00998027) | 3.34621 ± 0.0111364 (W 0.00979473) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.1498 ± 0.0273592 (W 0.0303396) | 24.8242 ± 1.48787 (W 0.0267232) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 25.6322 ± 0.0620925 (W 0.0802806) | 25.1186 ± 0.103248 (W 0.109998) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.560632 ± 0.00176442 (W 0.00274425) | 0.556997 ± 0.00167616 (W 0.00269841) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.558712 ± 0.00180966 (W 0.00217772) | 0.555462 ± 0.00357335 (W 0.00182856) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.85652 ± 0.00625078 (W 0.00578995) | 3.83839 ± 0.0156774 (W 0.00801149) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.85584 ± 0.00748941 (W 0.007152) | 3.84691 ± 0.00170818 (W 0.00913165) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 28.8375 ± 0.0392684 (W 0.0461888) | 28.9392 ± 0.080888 (W 0.068307) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 28.8252 ± 0.102152 (W 0.0367479) | 28.8485 ± 0.13024 (W 0.0335593) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.17198 ± 0.000370065 (W 0.000580975) | 0.165648 ± 0.000532666 (W 0.00185914) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.181348 ± 0.00128727 (W 0.00188098) | 0.175346 ± 0.0017035 (W 0.00203963) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.05269 ± 0.00418297 (W 0.00500853) | 1.0216 ± 0.00262489 (W 0.00226783) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.33523 ± 0.00263389 (W 0.00334964) | 1.31032 ± 0.00492709 (W 0.00388012) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.11586 ± 0.0160156 (W 0.00805844) | 7.48666 ± 0.0103081 (W 0.0107153) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 8.56308 ± 0.0327039 (W 0.0134864) | 8.67177 ± 0.00282272 (W 0.0129015) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.886143 ± 0.00513658 (W 0.000842896) | 0.933628 ± 0.00541915 (W 0.0021547) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.966202 ± 0.00711445 (W 0.00319623) | 1.00444 ± 0.000748057 (W 0.00165695) | - | · | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.27348 ± 0.00811542 (W 0.00796692) | 5.41556 ± 0.000364396 (W 0.00863261) | - | · | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.7402 ± 0.017528 (W 0.0180083) | 5.88153 ± 0.0173522 (W 0.0113421) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 35.7098 ± 0.0522014 (W 0.0281591) | 36.1401 ± 0.135841 (W 0.0586151) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 37.568 ± 0.0424648 (W 0.0497728) | 38.0526 ± 0.124263 (W 0.0807513) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.441022 ± 0.00232885 (W 0.00205787) | 0.485 ± 0.00383152 (W 0.00278552) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.40981 ± 0.0070286 (W 0.00656111) | 2.55519 ± 0.0114386 (W 0.00616598) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 15.054 ± 0.0258404 (W 0.0211034) | 15.6126 ± 0.0230274 (W 0.0201244) | - | · | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.170967 ± 0.000402419 (W 0.0015547) | 0.163183 ± 0.000271809 (W 0.00170656) | - | · | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.178533 ± 0.000494383 (W 0.00191214) | 0.174846 ± 0.000757809 (W 0.00219847) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.03136 ± 0.00345065 (W 0.00340009) | 1.00685 ± 0.000636744 (W 0.00129521) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.07722 ± 0.00202157 (W 0.00182171) | 1.04904 ± 0.00372875 (W 0.001844) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 7.01344 ± 0.00458453 (W 0.00782283) | 7.3516 ± 0.0147531 (W 0.0115677) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.20125 ± 0.0152654 (W 0.00818036) | 7.22533 ± 0.0234269 (W 0.00826134) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.1454 ± 0.0134435 (W 0.00502795) | 3.32848 ± 0.0150321 (W 0.00958239) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.16415 ± 0.00777841 (W 0.00633706) | 3.34671 ± 0.0243924 (W 0.0056489) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 16.7795 ± 0.0151917 (W 0.0414395) | 19.1922 ± 0.0594362 (W 0.0335319) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 16.7916 ± 0.0123337 (W 0.0216505) | 19.2422 ± 0.035356 (W 0.0736748) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 104.709 ± 0.22674 (W 0.0957072) | 119.854 ± 0.0278192 (W 0.420337) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 105.039 ± 0.130529 (W 0.143788) | 120.292 ± 0.288501 (W 0.413673) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.04683 ± 0.0024784 (W 0.00377836) | 2.10134 ± 0.00177657 (W 0.00644117) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.04932 ± 0.00648275 (W 0.0023144) | 2.11184 ± 0.00429876 (W 0.00695844) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.2033 ± 0.0159695 (W 0.0205711) | 11.2843 ± 0.0174335 (W 0.036219) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.21559 ± 0.0432299 (W 0.0145757) | 11.3462 ± 0.031031 (W 0.0241999) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 43.2476 ± 0.0857625 (W 0.115381) | 67.2559 ± 0.0608203 (W 0.0837012) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 43.4577 ± 0.330817 (W 0.0776844) | 67.5225 ± 0.0885576 (W 0.0448886) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005982 ± 2.00083e-05 (W 8.90915e-05) | 0.006151 ± 3.82143e-05 (W 4.42979e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017042 ± 0.000305511 (W 0.000153633) | 0.017192 ± 6.80686e-05 (W 4.15933e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.054751 ± 0.000291834 (W 0.000901058) | 0.054321 ± 0.000107503 (W 0.000480557) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.181207 ± 0.000271 (W 0.00146415) | 0.198359 ± 0.00250939 (W 0.00246553) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188851 ± 0.00039127 (W 0.00154946) | 0.206795 ± 0.00153112 (W 0.00162709) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.53161 ± 0.000680618 (W 0.00222207) | 0.610296 ± 0.00534112 (W 0.00507707) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.567727 ± 0.000689026 (W 0.00277715) | 0.648407 ± 0.00273543 (W 0.00786734) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.81285 ± 0.00742524 (W 0.00478808) | 2.01804 ± 0.107431 (W 0.0773631) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.97996 ± 0.0270022 (W 0.00863857) | 2.22842 ± 0.0408785 (W 0.0595065) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32301 ± 0.00224686 (W 0.00338868) | 1.92535 ± 0.00333106 (W 0.00290094) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.33221 ± 0.00196056 (W 0.00419826) | 1.95011 ± 0.0122256 (W 0.00604899) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.26491 ± 0.0127021 (W 0.00595682) | 6.90919 ± 0.0174792 (W 0.104972) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.30445 ± 0.0144909 (W 0.0133556) | 6.94578 ± 0.109071 (W 0.119777) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.5207 ± 0.0187666 (W 0.0145211) | 26.541 ± 0.0372892 (W 0.0308095) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.6373 ± 0.0168164 (W 0.0213724) | 26.6479 ± 0.017871 (W 0.0302195) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.351454 ± 0.00142948 (W 0.00289123) | 0.363156 ± 0.000136764 (W 0.00277184) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.393933 ± 0.00126183 (W 0.00276523) | 0.361614 ± 0.00116359 (W 0.00324433) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.13008 ± 0.00963953 (W 0.0136369) | 3.21177 ± 0.00354558 (W 0.0137713) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.35456 ± 0.0124718 (W 0.00749673) | 3.12881 ± 0.00587428 (W 0.00851146) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.6412 ± 0.0303687 (W 0.0277792) | 16.5423 ± 0.00568512 (W 0.0270955) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.7203 ± 0.0389546 (W 0.0233033) | 16.2341 ± 0.0597743 (W 0.0200523) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.158375 ± 0.000552905 (W 0.0012215) | 0.15589 ± 0.000397702 (W 0.00132996) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.158616 ± 0.000519369 (W 0.00146525) | 0.15577 ± 0.00065738 (W 0.00180362) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00584 ± 0.00119256 (W 0.00138762) | 1.00399 ± 0.00250001 (W 0.0025605) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00449 ± 0.000247561 (W 0.00198042) | 1.00589 ± 0.00187424 (W 0.00428343) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.37439 ± 0.0112699 (W 0.00877655) | 7.3768 ± 0.00389804 (W 0.0109282) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.38794 ± 0.00638475 (W 0.0113525) | 7.37946 ± 0.0109745 (W 0.0153041) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.131806 ± 0.000800863 (W 0.0010647) | 0.168664 ± 0.00184181 (W 0.00187553) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.186056 ± 0.00258501 (W 0.00184299) | 0.183522 ± 0.00190086 (W 0.00220526) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.551647 ± 0.00478066 (W 0.00237155) | 0.729518 ± 0.00312693 (W 0.00523831) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.836698 ± 0.00267361 (W 0.00445188) | 0.786534 ± 0.00204558 (W 0.00449079) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.76712 ± 0.00841213 (W 0.014828) | 3.5745 ± 0.0102253 (W 0.0904395) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.21615 ± 0.0214665 (W 0.0110523) | 3.66951 ± 0.0428646 (W 0.0726103) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.454126 ± 0.00274594 (W 0.00323144) | 0.710242 ± 0.0049467 (W 0.00338127) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.548691 ± 0.00111681 (W 0.00351954) | 0.796402 ± 0.00406247 (W 0.00196487) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.71984 ± 0.00130333 (W 0.00517182) | 2.57952 ± 0.00375259 (W 0.0109159) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.21249 ± 0.014006 (W 0.00684613) | 3.03741 ± 0.00287328 (W 0.00930507) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.4417 ± 0.0248095 (W 0.0175332) | 12.1791 ± 0.0250536 (W 0.0207969) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 12.6345 ± 0.00452839 (W 0.0236365) | 14.4058 ± 0.0363844 (W 0.0423227) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.287385 ± 0.00118075 (W 0.00171819) | 0.524035 ± 0.00297151 (W 0.00260818) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.938457 ± 0.00445941 (W 0.0019017) | 1.78584 ± 0.00479432 (W 0.00265023) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.14774 ± 0.00207944 (W 0.00539339) | 6.89487 ± 0.0048956 (W 0.0153145) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.136905 ± 0.000706827 (W 0.000833669) | 0.165999 ± 0.00063331 (W 0.00103381) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.148266 ± 0.00105486 (W 0.00101969) | 0.180606 ± 0.00305609 (W 0.00250209) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.528222 ± 0.00157191 (W 0.00321811) | 0.680678 ± 0.000803207 (W 0.00342493) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.845113 ± 0.00427765 (W 0.00341812) | 0.728817 ± 0.00519539 (W 0.00387805) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.58744 ± 0.0055898 (W 0.00781091) | 3.20685 ± 0.0196736 (W 0.0688268) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.08588 ± 0.0177271 (W 0.00970314) | 3.40082 ± 0.0529859 (W 0.0508292) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.97591 ± 0.00660951 (W 0.00351735) | 2.465 ± 0.0159176 (W 0.00678526) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.9811 ± 0.00126022 (W 0.0064092) | 2.47568 ± 0.00973717 (W 0.0063595) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.27027 ± 0.0444792 (W 0.0235976) | 12.8895 ± 0.0189962 (W 0.0727185) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.31612 ± 0.0311182 (W 0.0174969) | 12.9308 ± 0.0683236 (W 0.0672087) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 40.0892 ± 0.104026 (W 0.0357246) | 56.6632 ± 0.150538 (W 0.0642884) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 40.1974 ± 0.0386989 (W 0.0400043) | 56.9746 ± 0.0494612 (W 0.46478) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.932967 ± 0.00546548 (W 0.00361955) | 1.83451 ± 0.00633333 (W 0.00250645) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.929822 ± 0.00473355 (W 0.00351758) | 1.84593 ± 0.00518893 (W 0.00447867) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.9955 ± 0.0151196 (W 0.00882713) | 9.7298 ± 0.0198229 (W 0.0231776) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.01305 ± 0.00781226 (W 0.00770181) | 9.75443 ± 0.0153539 (W 0.0241725) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.9505 ± 0.106221 (W 0.0346053) | 40.2652 ± 0.02638 (W 0.0567048) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.006 ± 0.0252501 (W 0.0473667) | 40.2869 ± 0.0523191 (W 0.0410869) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007053 ± 4.93288e-05 (W 4.81373e-05) | 0.006071 ± 2.51661e-05 (W 4.31068e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017743 ± 7.02377e-05 (W 0.000136525) | 0.017092 ± 5.03322e-05 (W 0.000165714) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.04276 ± 0.000567785 (W 0.000416495) | 0.043812 ± 0.00019186 (W 0.000310494) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.180306 ± 0.000170978 (W 0.00123975) | 0.192378 ± 0.000916969 (W 0.00206381) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188851 ± 0.00036781 (W 0.001411) | 0.199882 ± 0.000847052 (W 0.00193433) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.505391 ± 0.00112743 (W 0.00192973) | 0.613863 ± 0.0239653 (W 0.0212661) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.547179 ± 0.001353 (W 0.00268117) | 0.653779 ± 0.0196701 (W 0.0284358) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.4494 ± 0.00389261 (W 0.00297581) | 1.75645 ± 0.0275147 (W 0.0405787) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.63653 ± 0.00218008 (W 0.00375105) | 1.90832 ± 0.0179491 (W 0.023398) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.986567 ± 0.00448761 (W 0.00248788) | 1.94347 ± 0.00785988 (W 0.00320945) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.99328 ± 0.00343366 (W 0.00130741) | 1.95885 ± 0.00710044 (W 0.00283741) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.95574 ± 0.0113339 (W 0.00445184) | 6.85385 ± 0.00834419 (W 0.0329156) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.97844 ± 0.00447085 (W 0.00375606) | 6.87288 ± 0.0288728 (W 0.114155) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.6145 ± 0.00829148 (W 0.0211397) | 23.8472 ± 0.0329617 (W 0.0891201) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.6914 ± 0.125836 (W 0.0254685) | 24.0629 ± 0.0569537 (W 0.0902324) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.352988 ± 0.0044957 (W 0.00290473) | 0.361153 ± 0.00111901 (W 0.00270646) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.361884 ± 0.00474045 (W 0.00441956) | 0.360312 ± 0.00214604 (W 0.00351833) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.98985 ± 0.00624573 (W 0.00556355) | 3.1947 ± 0.0174447 (W 0.00874723) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.23923 ± 0.00440736 (W 0.0109736) | 3.11727 ± 0.0179213 (W 0.00837673) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.7895 ± 0.00366612 (W 0.0174301) | 15.5479 ± 0.0744378 (W 0.0256121) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.0541 ± 0.00283594 (W 0.018683) | 15.0994 ± 0.032053 (W 0.0197101) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.10244 ± 0.00132052 (W 0.00246925) | 0.175046 ± 0.00156767 (W 0.00178162) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.1022 ± 0.000797672 (W 0.000731241) | 0.174805 ± 0.000482249 (W 0.00106239) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.540337 ± 0.00173755 (W 0.00276671) | 0.727094 ± 0.0391583 (W 0.0388506) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.541649 ± 0.00172669 (W 0.00196495) | 0.73007 ± 0.0046345 (W 0.0462176) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.85418 ± 0.0067545 (W 0.00650567) | 3.88315 ± 0.793739 (W 0.643472) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.85651 ± 0.015325 (W 0.00561807) | 3.8901 ± 0.814661 (W 0.783386) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.1292 ± 0.000688037 (W 0.000577336) | 0.162843 ± 0.000551437 (W 0.00118769) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.142686 ± 0.00106301 (W 0.000602095) | 0.178442 ± 0.000926531 (W 0.00086713) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.483971 ± 0.00344683 (W 0.00257644) | 0.722155 ± 0.0219406 (W 0.0195963) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.782007 ± 0.0049298 (W 0.00508139) | 0.780143 ± 0.00717013 (W 0.0210841) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.05925 ± 0.00315178 (W 0.00427808) | 3.09615 ± 0.0128537 (W 0.043029) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.52201 ± 0.0269552 (W 0.0097757) | 3.29817 ± 0.0457445 (W 0.0447841) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.378144 ± 0.00224946 (W 0.00286571) | 0.833001 ± 0.00343771 (W 0.0244046) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.48858 ± 0.00415782 (W 0.00213617) | 0.924261 ± 0.00615716 (W 0.00251629) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.29096 ± 0.00362619 (W 0.00483762) | 2.7538 ± 0.0253868 (W 0.0240009) | - | 🔵 faer-fork 2.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.78761 ± 0.0057812 (W 0.0039027) | 2.84693 ± 0.0118487 (W 0.0374497) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.62875 ± 0.0255856 (W 0.00825862) | 10.6728 ± 0.0479823 (W 0.0650111) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 7.89276 ± 0.00722839 (W 0.0142566) | 11.2479 ± 1.0119 (W 0.625076) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.265605 ± 0.000696292 (W 0.00164014) | 0.629081 ± 0.00183613 (W 0.00318663) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.857317 ± 0.00201534 (W 0.00282235) | 1.99834 ± 0.0118816 (W 0.011749) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.85807 ± 0.142275 (W 0.0095321) | 6.68825 ± 0.0217939 (W 0.0475408) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.135673 ± 0.000571624 (W 0.00062234) | 0.157042 ± 0.000553706 (W 0.00220374) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.146132 ± 0.00033207 (W 0.00119888) | 0.174185 ± 0.00139336 (W 0.00105044) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.473221 ± 0.00450432 (W 0.00201504) | 0.674046 ± 0.00523005 (W 0.0134706) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.785342 ± 0.00695413 (W 0.00303814) | 0.753283 ± 0.022731 (W 0.0215004) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 1.92144 ± 0.00304443 (W 0.00310896) | 2.84349 ± 0.0218263 (W 0.0445894) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.40831 ± 0.00906275 (W 0.00493875) | 3.01047 ± 0.0158423 (W 0.0259724) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.91383 ± 0.00421558 (W 0.00213188) | 2.50025 ± 0.0117305 (W 0.0139286) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.91975 ± 0.00244546 (W 0.00284597) | 2.51339 ± 0.00785529 (W 0.00118487) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.46628 ± 0.00434584 (W 0.008888) | 12.8162 ± 0.0245925 (W 0.0463239) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.4749 ± 0.00751686 (W 0.00915456) | 12.8811 ± 0.0247712 (W 0.0755588) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 31.4008 ± 0.0748295 (W 0.0331624) | 51.6372 ± 0.121665 (W 0.0661004) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 31.5504 ± 0.0470793 (W 0.0354795) | 51.8058 ± 0.0280863 (W 0.299411) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.972652 ± 0.00362335 (W 0.00190897) | 1.84482 ± 0.0335157 (W 0.0031628) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.973413 ± 0.0038749 (W 0.00227128) | 1.84308 ± 0.0082361 (W 0.00588351) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.96417 ± 0.0178805 (W 0.0163861) | 9.81115 ± 0.0178087 (W 0.0116844) | - | 🔵 faer-fork 2.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.96849 ± 0.0183392 (W 0.0127162) | 9.86732 ± 0.0258855 (W 0.0186199) | - | 🔵 faer-fork 2.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.2082 ± 0.0179177 (W 0.0127222) | 38.3474 ± 0.0525221 (W 0.0446681) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.2946 ± 0.414849 (W 0.0218261) | 38.2548 ± 0.0684515 (W 0.0533983) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007053 ± 4.58258e-05 (W 4.43531e-05) | 0.006091 ± 0.000148987 (W 4.63681e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.018365 ± 0.000190686 (W 7.58301e-05) | 0.016941 ± 1.52753e-05 (W 0.000198939) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.044673 ± 8.28674e-05 (W 0.000464827) | 0.043701 ± 0.000610411 (W 0.000376151) | - | · | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `lu/128`, `lu/256`, `lu/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `solve/128`, `solve/256`, `solve/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `triangular_solve/128`, `triangular_solve/256`, `triangular_solve/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.954668 ± 0.0022687 (W 0.00347198) | 1.21674 ± 0.00265439 (W 0.00439547) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.821771 ± 0.00347417 (W 0.00328913) | 0.816681 ± 0.00280019 (W 0.00380973) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 53.1372 ± 0.525057 (W 0.978594) | 100.32 ± 1.15073 (W 0.698265) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 49.2215 ± 0.248865 (W 0.0600411) | 53.2147 ± 0.356879 (W 0.689121) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.555335 ± 0.000628816 (W 0.00201589) | 0.557448 ± 0.00342968 (W 0.00421694) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.558139 ± 0.00104381 (W 0.00283344) | 0.567297 ± 0.00684401 (W 0.00360608) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.1256 ± 0.0844568 (W 0.12694) | 35.1942 ± 0.0327486 (W 0.391901) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.1518 ± 0.0453389 (W 0.0761573) | 35.162 ± 0.0309116 (W 0.0469141) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.326368 ± 0.015674 (W 0.00484098) | 0.850174 ± 0.0169136 (W 0.00500331) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.329804 ± 0.06582 (W 0.00251273) | 0.476448 ± 0.00784034 (W 0.00479262) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 12.542 ± 0.0377501 (W 0.0250697) | 39.6212 ± 0.322798 (W 0.0486412) | - | 🔵 faer-fork 3.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 13.4995 ± 0.064431 (W 0.0679364) | 18.9706 ± 0.011216 (W 0.0467541) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.48334 ± 0.0100281 (W 0.00583892) | 2.43437 ± 0.00151301 (W 0.00751749) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.71246 ± 0.00130166 (W 0.00552713) | 1.81987 ± 0.00978025 (W 0.00397204) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 67.8893 ± 0.34465 (W 0.2018) | 114.786 ± 0.0868921 (W 0.186989) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 72.4471 ± 0.456817 (W 0.440457) | 76.1893 ± 0.124549 (W 0.0904286) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.933969 ± 0.00158397 (W 0.00289935) | 1.87688 ± 0.0149211 (W 0.0047584) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 30.9482 ± 0.1009 (W 0.220535) | 78.2456 ± 0.100474 (W 0.182217) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.03682 ± 0.00388754 (W 0.00415948) | 2.50871 ± 0.00353129 (W 0.00388279) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.04328 ± 0.00612447 (W 0.00178481) | 2.51725 ± 0.0031756 (W 0.00636029) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 83.4322 ± 0.060323 (W 0.254679) | 88.5749 ± 0.00979827 (W 0.0901955) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 83.7303 ± 0.246832 (W 0.436676) | 88.8545 ± 0.233022 (W 0.15654) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.02071 ± 0.00567036 (W 0.00413957) | 1.49379 ± 0.00089436 (W 0.00612608) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.02259 ± 0.00408165 (W 0.00350929) | 1.4991 ± 0.00443861 (W 0.00583021) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 37.648 ± 0.0828975 (W 0.114099) | 44.9541 ± 0.0993655 (W 0.0360187) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 37.6638 ± 0.0873026 (W 0.0527902) | 45.0049 ± 0.0418787 (W 0.0697323) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.563304 ± 0.00579603 (W 0.00318443) | 1.20306 ± 0.00825883 (W 0.00420426) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.588497 ± 0.000978092 (W 0.007793) | 0.820696 ± 0.00968146 (W 0.00389912) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 16.0957 ± 0.049963 (W 0.0585854) | 48.2025 ± 0.166576 (W 0.199209) | - | 🔵 faer-fork 2.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 16.7721 ± 0.0928847 (W 0.0868379) | 21.5045 ± 1.08804 (W 0.0397528) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.167373 ± 0.00169585 (W 0.00274336) | 0.159358 ± 0.00170515 (W 0.00212462) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.170519 ± 0.000977698 (W 0.00195861) | 0.160088 ± 0.00468269 (W 0.00237644) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.45956 ± 0.0266462 (W 0.0308333) | 9.35978 ± 0.000932922 (W 0.0164601) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.46508 ± 0.0326032 (W 0.0279195) | 9.33147 ± 0.0295812 (W 0.0331333) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.281365 ± 0.00082779 (W 0.0034006) | 0.89137 ± 0.0105504 (W 0.00758781) | - | 🔵 faer-fork 3.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.297315 ± 0.000844575 (W 0.00165744) | 0.430813 ± 0.00442569 (W 0.00459139) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.21779 ± 0.0463678 (W 0.0462955) | 33.6915 ± 0.567985 (W 0.0477515) | - | 🔵 faer-fork 5.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.0474 ± 0.0999307 (W 0.0736213) | 12.737 ± 0.0418683 (W 0.0331236) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.755008 ± 0.0013331 (W 0.00305644) | 3.11059 ± 0.0338677 (W 0.0927457) | - | 🔵 faer-fork 4.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.11056 ± 0.000456235 (W 0.00502022) | 1.44711 ± 0.140454 (W 0.0071869) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 23.6578 ± 0.073199 (W 0.143239) | 84.7157 ± 0.653688 (W 0.465586) | - | 🔵 faer-fork 3.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 31.1559 ± 0.118032 (W 0.103647) | 39.0808 ± 3.16627 (W 0.461062) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.607472 ± 0.00491351 (W 0.0030798) | 2.95308 ± 0.132497 (W 0.144302) | - | 🔵 faer-fork 4.86× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 13.1383 ± 0.033022 (W 0.0917834) | 74.684 ± 0.493755 (W 0.216106) | - | 🔵 faer-fork 5.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.34001 ± 0.00351453 (W 0.00297924) | 2.13505 ± 0.00476898 (W 0.00970968) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.34882 ± 0.00204018 (W 0.00253024) | 2.15302 ± 0.0123509 (W 0.0103792) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 36.906 ± 0.108891 (W 0.118091) | 47.1918 ± 0.16764 (W 0.0876469) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 37.2395 ± 0.0534931 (W 0.0848069) | 47.5685 ± 0.0189466 (W 0.151012) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.878929 ± 0.004342 (W 0.00273509) | 1.53548 ± 0.00241118 (W 0.00385227) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.885108 ± 0.00816581 (W 0.00234573) | 1.56425 ± 0.0229716 (W 0.00491839) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 22.0661 ± 0.033653 (W 0.0322668) | 32.6952 ± 0.0846198 (W 0.0561329) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 22.1156 ± 0.0762066 (W 0.0321928) | 32.8329 ± 0.0583612 (W 0.0363275) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.53053 ± 0.00217011 (W 0.00504674) | 1.21339 ± 0.00687255 (W 0.00280157) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.570546 ± 0.00333289 (W 0.0032601) | 0.818158 ± 0.00767251 (W 0.0024894) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 10.1678 ± 0.0981792 (W 0.0565287) | 41.3228 ± 0.0614366 (W 0.0833364) | - | 🔵 faer-fork 4.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.9901 ± 0.0487452 (W 0.0361514) | 16.0078 ± 1.28357 (W 0.0482576) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.107203 ± 0.0018709 (W 0.0033832) | 0.148587 ± 0.00641208 (W 0.00506765) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.106179 ± 0.00117904 (W 0.00143479) | 0.141198 ± 0.00403097 (W 0.00685914) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.68362 ± 0.127283 (W 0.14197) | 6.64846 ± 0.0315281 (W 0.916926) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.69109 ± 0.011699 (W 0.0475165) | 6.69212 ± 0.111159 (W 0.80283) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.249897 ± 0.00181855 (W 0.0024244) | 0.953368 ± 0.0225035 (W 0.0114993) | - | 🔵 faer-fork 3.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.289912 ± 0.00463489 (W 0.00157783) | 0.425543 ± 0.00761893 (W 0.00479535) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.18486 ± 0.0144609 (W 0.0188435) | 32.3707 ± 0.430471 (W 0.0629928) | - | 🔵 faer-fork 7.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.9463 ± 0.0175567 (W 0.0547601) | 11.8008 ± 0.0313359 (W 0.0355361) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.675064 ± 0.00331842 (W 0.0032318) | 3.10443 ± 0.115122 (W 0.0890551) | - | 🔵 faer-fork 4.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.02932 ± 0.00425773 (W 0.00113924) | 1.25372 ± 0.0265984 (W 0.0289391) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 16.4205 ± 0.0549326 (W 0.0653363) | 79.8521 ± 0.0712489 (W 0.250474) | - | 🔵 faer-fork 4.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 25.2996 ± 0.124151 (W 0.0832707) | 34.6032 ± 0.615406 (W 3.14029) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.565347 ± 0.00401794 (W 0.00451746) | 3.08165 ± 0.052963 (W 0.111025) | - | 🔵 faer-fork 5.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 9.71047 ± 0.061049 (W 0.0213184) | 71.4823 ± 0.23197 (W 0.246557) | - | 🔵 faer-fork 7.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.25749 ± 0.0287909 (W 0.00715488) | 2.23612 ± 0.00564485 (W 0.00619691) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.27907 ± 0.00913598 (W 0.00488187) | 2.26079 ± 0.00574173 (W 0.059528) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 30.7147 ± 0.0144742 (W 0.109067) | 46.486 ± 0.162064 (W 0.275351) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 30.9575 ± 0.264907 (W 0.0844191) | 46.5215 ± 0.215107 (W 1.44249) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.845819 ± 0.00938067 (W 0.00446956) | 1.6166 ± 0.00783361 (W 0.0105884) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.844997 ± 0.00737147 (W 0.00544431) | 1.62841 ± 0.00338903 (W 0.00493321) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.5575 ± 0.117122 (W 0.0489294) | 33.5228 ± 0.0223831 (W 0.0984318) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.5545 ± 0.0637188 (W 0.0372031) | 33.5925 ± 0.0968315 (W 0.431828) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.809177 ± 0.000360236 (W 0.00601993) | 0.806261 ± 0.00165279 (W 0.00699928) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.742402 ± 0.00376552 (W 0.00282055) | 0.733446 ± 0.00222489 (W 0.0053699) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 51.8867 ± 0.457345 (W 0.18512) | 52.1482 ± 0.613455 (W 0.428491) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 46.8563 ± 0.201542 (W 0.46982) | 46.7986 ± 0.732598 (W 0.405343) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.604185 ± 0.00201433 (W 0.00351679) | 0.599618 ± 0.00337102 (W 0.00180931) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.598626 ± 0.0056245 (W 0.002941) | 0.596311 ± 0.000944538 (W 0.00355764) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.8505 ± 0.125993 (W 0.0885101) | 35.8425 ± 0.0764662 (W 0.0906513) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.7972 ± 0.0438271 (W 0.0287214) | 35.8298 ± 0.0498297 (W 0.0527058) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.226752 ± 0.00111898 (W 0.00279004) | 0.36492 ± 0.00150845 (W 0.00237704) | 0.246569 ± 0.00149342 (W 0.00239838) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.240999 ± 0.00137951 (W 0.00316698) | 0.385378 ± 0.000696175 (W 0.00314641) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.8103 ± 0.0375118 (W 0.0223781) | 12.7348 ± 0.0265686 (W 0.0341156) | 11.023 ± 0.0754824 (W 0.0182102) | 🔵 faer-fork 1.18× faster than `faer-upstream` | · |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.2946 ± 0.0194509 (W 0.0349106) | 13.3719 ± 0.0198948 (W 0.0451864) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.2636 ± 0.00255841 (W 0.00387711) | 1.38946 ± 0.00248885 (W 0.00457318) | 1.84681 ± 0.00399579 (W 0.00831086) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.57811 ± 0.00546054 (W 0.00440742) | 1.72682 ± 0.00460895 (W 0.00685379) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 61.4086 ± 0.0495848 (W 0.147809) | 62.4689 ± 0.105252 (W 0.257228) | 76.6016 ± 1.02635 (W 0.115167) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 69.6667 ± 0.083318 (W 0.095455) | 70.197 ± 0.132817 (W 0.302144) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.67098 ± 0.00129575 (W 0.00454825) | 0.797086 ± 0.00142988 (W 0.00244374) | 0.756319 ± 0.000370824 (W 0.00300337) | 🔵 faer-fork 1.19× faster than `faer-upstream` | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 24.96 ± 0.0941963 (W 0.0268134) | 26.0316 ± 0.0750711 (W 0.0316174) | 28.8659 ± 0.0618678 (W 0.0582491) | · | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 1.98968 ± 0.00342125 (W 0.00288227) | 2.45856 ± 0.00437773 (W 0.00352178) | 2.78075 ± 0.00394748 (W 0.0054993) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.99863 ± 0.00324523 (W 0.00287816) | 2.46723 ± 0.00965053 (W 0.00343881) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 77.9164 ± 0.0359624 (W 0.0901859) | 83.4045 ± 0.119856 (W 0.327465) | 110.975 ± 0.658256 (W 0.13214) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 78.3737 ± 0.163636 (W 0.160691) | 83.6591 ± 0.0298117 (W 0.383842) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.941443 ± 0.00177151 (W 0.00195567) | 1.40357 ± 0.00281979 (W 0.00327412) | 1.20472 ± 0.00135442 (W 0.00420046) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.941023 ± 0.00445236 (W 0.00340562) | 1.40702 ± 0.00294774 (W 0.00236066) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.684 ± 0.0187775 (W 0.170147) | 39.2252 ± 0.044648 (W 0.0466242) | 39.752 ± 0.296912 (W 0.0210645) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.7233 ± 0.040299 (W 0.0546341) | 39.2543 ± 0.0622554 (W 0.0705305) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.490187 ± 0.00211053 (W 0.00356595) | 0.802734 ± 0.000861646 (W 0.00244938) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.505372 ± 0.00282172 (W 0.00352544) | 0.726655 ± 0.0027991 (W 0.00280451) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 15.3293 ± 0.0662349 (W 0.0246784) | 15.9322 ± 0.0570267 (W 0.0357321) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 14.494 ± 0.0429614 (W 0.0393249) | 14.9049 ± 0.044117 (W 0.0542157) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.200705 ± 0.00142205 (W 0.00104178) | 0.197939 ± 0.000671143 (W 0.00197073) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.20396 ± 0.00230824 (W 0.000608942) | 0.199392 ± 0.00354353 (W 0.00195067) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 11.1401 ± 0.00672309 (W 0.0167946) | 11.1016 ± 0.0228027 (W 0.0143779) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 11.1607 ± 0.0188317 (W 0.0160886) | 11.0805 ± 0.0212616 (W 0.0171495) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.184966 ± 0.000608976 (W 0.00178875) | 0.328132 ± 0.00192843 (W 0.00298028) | 0.247672 ± 0.000858448 (W 0.00316705) | 🔵 faer-fork 1.77× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.212176 ± 0.00116196 (W 0.00186576) | 0.357306 ± 0.00259712 (W 0.00351098) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.31527 ± 0.00509449 (W 0.022965) | 6.48156 ± 0.0127887 (W 0.0191208) | 10.9915 ± 0.0260817 (W 0.013248) | 🔵 faer-fork 1.50× faster than `faer-upstream` | 🔵 faer-fork 2.55× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.77246 ± 0.00585135 (W 0.0151695) | 7.00289 ± 0.0102761 (W 0.0131647) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.69713 ± 0.00178282 (W 0.00205587) | 1.01725 ± 0.00317943 (W 0.00253449) | 0.740215 ± 0.00147219 (W 0.00905023) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.980428 ± 0.00779035 (W 0.00340975) | 1.13059 ± 0.126034 (W 0.0329662) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 21.7006 ± 0.0328255 (W 0.05691) | 26.4963 ± 0.0768324 (W 0.0687992) | 25.5515 ± 0.115789 (W 0.0763455) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 28.4846 ± 0.112242 (W 0.041451) | 28.2497 ± 1.32436 (W 2.92092) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.520755 ± 0.00487208 (W 0.00399167) | 0.840236 ± 0.00417234 (W 0.00381842) | 0.396322 ± 0.00185645 (W 0.00486839) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔴 faer-fork 1.31× slower than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 10.8674 ± 0.0200914 (W 0.0148753) | 15.5738 ± 0.0195966 (W 0.0368112) | 10.9731 ± 0.0319035 (W 0.0648333) | 🔵 faer-fork 1.43× faster than `faer-upstream` | · |  |
| `svd_thin/1024x64/b1` | 1.30565 ± 0.00535735 (W 0.00320283) | 2.08195 ± 0.00604617 (W 0.00544269) | 1.77006 ± 0.00399482 (W 0.00344269) | 🔵 faer-fork 1.59× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.31092 ± 0.00809244 (W 0.00306105) | 2.11298 ± 0.0042223 (W 0.0063507) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 31.626 ± 0.0628057 (W 0.0342263) | 41.9706 ± 0.0491986 (W 0.0803843) | 44.9082 ± 0.159834 (W 0.0882184) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 31.824 ± 0.0465533 (W 0.142924) | 42.1425 ± 0.0741412 (W 0.0585692) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.792776 ± 0.00300055 (W 0.00234624) | 1.44937 ± 0.00381015 (W 0.00121154) | 0.856011 ± 0.00264778 (W 0.00329887) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.800804 ± 0.000460192 (W 0.00239486) | 1.45486 ± 0.00587097 (W 0.00871125) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 16.227 ± 0.0239971 (W 0.0258686) | 26.8522 ± 0.0553245 (W 0.0774741) | 20.2204 ± 0.0423048 (W 0.0557491) | 🔵 faer-fork 1.65× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 16.1703 ± 0.0123554 (W 0.0210832) | 26.9936 ± 0.0615611 (W 0.108323) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.459585 ± 0.00477489 (W 0.00490499) | 0.802067 ± 0.0022242 (W 0.00261773) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.486649 ± 0.00465354 (W 0.00299108) | 0.730131 ± 0.00247559 (W 0.00419736) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.93229 ± 0.0054045 (W 0.0512778) | 10.6991 ± 0.0243532 (W 0.0257959) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 9.63441 ± 0.0970239 (W 0.0335918) | 10.2916 ± 0.0222185 (W 0.0140273) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.124907 ± 0.000548987 (W 0.0010827) | 0.19281 ± 0.00094983 (W 0.00684652) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.128369 ± 0.000911377 (W 0.000507147) | 0.193912 ± 0.00248071 (W 0.00756754) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 8.83641 ± 0.0200256 (W 0.0244628) | 7.99351 ± 0.649011 (W 0.693908) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 8.84122 ± 0.0160199 (W 0.0210122) | 8.06369 ± 0.72785 (W 0.513865) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.175517 ± 0.00241176 (W 0.00225535) | 0.325188 ± 0.00125821 (W 0.0027578) | 0.246751 ± 0.00101672 (W 0.00249148) | 🔵 faer-fork 1.85× faster than `faer-upstream` | 🔵 faer-fork 1.41× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.206856 ± 0.000452289 (W 0.00223966) | 0.351897 ± 0.00188991 (W 0.00367445) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.24631 ± 0.0117331 (W 0.0111329) | 5.54907 ± 0.0413276 (W 0.0237255) | 2.35754 ± 0.00968652 (W 0.00748658) | 🔵 faer-fork 1.71× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 3.76177 ± 0.0030865 (W 0.0103728) | 6.12969 ± 0.0228102 (W 0.0203539) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.60476 ± 0.00203746 (W 0.00268176) | 1.11568 ± 0.00346485 (W 0.00781888) | 0.554936 ± 0.00158139 (W 0.00253375) | 🔵 faer-fork 1.84× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.89555 ± 0.00227214 (W 0.00381085) | 1.17159 ± 0.0404575 (W 0.0312237) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 15.8928 ± 0.0365475 (W 0.042576) | 25.4956 ± 0.0610968 (W 0.191042) | 16.7374 ± 0.0343829 (W 0.14213) | 🔵 faer-fork 1.60× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 22.5217 ± 0.0432162 (W 0.0608044) | 27.3195 ± 3.19361 (W 2.80093) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.49245 ± 0.00170916 (W 0.00215134) | 0.924607 ± 0.00272822 (W 0.00286844) | 0.280714 ± 0.00106137 (W 0.00205382) | 🔵 faer-fork 1.88× faster than `faer-upstream` | 🔴 faer-fork 1.75× slower than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 8.52667 ± 0.0256229 (W 0.0382767) | 16.3587 ± 0.00205767 (W 0.103612) | 6.19735 ± 0.0336494 (W 0.0421752) | 🔵 faer-fork 1.92× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 1.21702 ± 0.00783911 (W 0.00276936) | 2.1847 ± 0.00434378 (W 0.00672638) | 1.61582 ± 0.00259493 (W 0.00717572) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.22194 ± 0.00410677 (W 0.00576439) | 2.2041 ± 0.00536796 (W 0.00990019) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 25.2286 ± 0.0587371 (W 0.0477342) | 40.8941 ± 0.106735 (W 0.144366) | 34.1701 ± 0.334843 (W 0.175523) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 25.4233 ± 0.0761208 (W 0.0730082) | 41.4056 ± 1.20562 (W 3.67144) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.76768 ± 0.00353428 (W 0.00195389) | 1.53303 ± 0.0063374 (W 0.00433452) | 0.740241 ± 0.00342679 (W 0.00458844) | 🔵 faer-fork 2.00× faster than `faer-upstream` | · |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.773154 ± 0.00303189 (W 0.0042838) | 1.54386 ± 0.0142176 (W 0.00831538) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 13.7729 ± 0.02092 (W 0.0416172) | 27.814 ± 0.0239285 (W 0.10633) | 15.1338 ± 0.0359712 (W 0.04468) | 🔵 faer-fork 2.02× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.7686 ± 0.0205697 (W 0.0252277) | 27.9183 ± 0.0382314 (W 0.113041) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.92297 ± 0.00759453 (W 0.00530178) | 2.83389 ± 0.0610425 (W 0.0261717) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.78868 ± 0.0103924 (W 0.00288667) | 1.80567 ± 0.0230881 (W 0.00499845) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 155.003 ± 0.716767 (W 0.913918) | 272.635 ± 1.86532 (W 0.88222) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 149.881 ± 0.0343513 (W 0.960967) | 156.025 ± 0.160713 (W 1.23473) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.19195 ± 0.000857155 (W 0.00411053) | 2.22296 ± 0.019558 (W 0.00496139) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.19413 ± 0.0018044 (W 0.00332131) | 2.22685 ± 0.0251281 (W 0.00258784) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 137.086 ± 0.144416 (W 0.0707326) | 137.128 ± 0.101975 (W 0.111255) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 137.204 ± 0.154396 (W 0.068583) | 137.185 ± 0.179714 (W 0.163965) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.922809 ± 0.0014724 (W 0.00234871) | 1.55033 ± 0.052416 (W 0.00739545) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.966801 ± 0.00410372 (W 0.00151138) | 1.03974 ± 0.00653608 (W 0.00197607) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 44.4788 ± 0.197952 (W 0.0915465) | 91.0991 ± 0.572708 (W 0.115628) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 46.3555 ± 0.289217 (W 0.198906) | 54.2453 ± 0.0760667 (W 0.0875552) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.54444 ± 0.00435712 (W 0.0100329) | 6.0096 ± 0.164798 (W 0.0132385) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 5.08505 ± 0.00678335 (W 0.00588411) | 5.3 ± 0.00424947 (W 0.00746146) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 238.697 ± 0.393592 (W 0.655261) | 317.303 ± 0.836186 (W 0.883742) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 250.189 ± 0.690662 (W 0.752739) | 259.635 ± 1.00218 (W 0.460342) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.35917 ± 0.0160065 (W 0.00277634) | 4.00282 ± 0.0393152 (W 0.0170636) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 98.1797 ± 0.104535 (W 0.179265) | 176.484 ± 0.286011 (W 0.287334) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 5.43538 ± 0.00521083 (W 0.00594817) | 5.95751 ± 0.00640292 (W 0.0102075) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.49141 ± 0.0205746 (W 0.0158298) | 5.99162 ± 0.0333892 (W 0.00734086) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 272.712 ± 0.624562 (W 0.456676) | 279.359 ± 0.122698 (W 0.463191) | - | · | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 273.171 ± 0.302638 (W 6.1329) | 280.953 ± 0.0520298 (W 6.01141) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.49649 ± 0.013838 (W 0.00675262) | 3.07293 ± 0.0231211 (W 0.0115393) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.53383 ± 0.0149001 (W 0.00822955) | 3.08012 ± 0.0198441 (W 0.00599243) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 113.596 ± 2.12088 (W 0.517092) | 121.996 ± 0.284596 (W 0.219725) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 113.21 ± 0.315274 (W 0.239295) | 123.217 ± 1.40965 (W 1.12522) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.862484 ± 0.0117041 (W 0.00738527) | 2.83312 ± 0.0737684 (W 0.0255268) | - | 🔵 faer-fork 3.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.928807 ± 0.00763353 (W 0.00524996) | 1.78738 ± 0.0243244 (W 0.00382082) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 49.9576 ± 0.363824 (W 0.258495) | 113.41 ± 0.508606 (W 0.416108) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 51.7869 ± 0.186849 (W 0.182749) | 62.7342 ± 0.325934 (W 0.625497) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.59631 ± 0.001659 (W 0.00245985) | 0.587624 ± 0.00379396 (W 0.00237445) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.596391 ± 0.00190753 (W 0.00270133) | 0.58878 ± 0.00330616 (W 0.00244925) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 37.1605 ± 0.0443153 (W 0.0554068) | 36.7536 ± 0.216081 (W 0.254039) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 37.1705 ± 0.041446 (W 0.0930304) | 36.7309 ± 0.216445 (W 0.284291) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.633161 ± 0.00081138 (W 0.00312508) | 1.34489 ± 0.0161947 (W 0.0131952) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.686048 ± 0.003618 (W 0.00255026) | 0.784341 ± 0.00646605 (W 0.00378406) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 18.4816 ± 0.0384588 (W 0.0751251) | 66.4426 ± 0.444184 (W 0.261902) | - | 🔵 faer-fork 3.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 21.0921 ± 0.18921 (W 0.220824) | 29.8696 ± 0.0307578 (W 0.080365) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.83924 ± 0.012135 (W 0.00382115) | 4.5096 ± 0.0254074 (W 0.0806119) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.58338 ± 0.00876159 (W 0.0085262) | 2.96072 ± 0.00560182 (W 0.0323636) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 77.2414 ± 0.26725 (W 0.740968) | 167.42 ± 0.503965 (W 0.686728) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 92.4142 ± 0.429221 (W 0.128924) | 108.359 ± 0.364665 (W 0.109235) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.2609 ± 0.00293174 (W 0.0061086) | 4.09214 ± 0.110264 (W 0.0848713) | - | 🔵 faer-fork 3.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 36.4197 ± 0.0798418 (W 0.128255) | 128.82 ± 0.528318 (W 0.498566) | - | 🔵 faer-fork 3.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.67372 ± 0.02616 (W 0.00476976) | 3.64102 ± 0.0114645 (W 0.00381899) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.68541 ± 0.02607 (W 0.00670225) | 3.67881 ± 0.00495594 (W 0.0073201) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 98.3058 ± 0.176259 (W 0.196769) | 117.526 ± 0.101803 (W 0.237197) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 99.2618 ± 0.171956 (W 0.295733) | 118.365 ± 0.0527773 (W 0.0947844) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.64829 ± 0.00465286 (W 0.00624217) | 2.46444 ± 0.0320135 (W 0.00366186) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.65742 ± 0.0222217 (W 0.00535547) | 2.43447 ± 0.02238 (W 0.00594716) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 52.46 ± 0.0109311 (W 0.387406) | 71.6629 ± 0.0560818 (W 0.0687236) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 52.4213 ± 0.0470219 (W 0.0786759) | 71.8874 ± 0.900646 (W 0.314344) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.720909 ± 0.0089554 (W 0.00449334) | 2.85223 ± 0.072867 (W 0.017738) | - | 🔵 faer-fork 3.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.811345 ± 0.0187181 (W 0.00596852) | 1.81798 ± 0.00634604 (W 0.00401038) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 25.6905 ± 0.0748389 (W 0.0856751) | 79.2103 ± 0.208331 (W 0.255289) | - | 🔵 faer-fork 3.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 28.6819 ± 0.190335 (W 0.122334) | 41.1267 ± 0.728166 (W 1.36946) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.329975 ± 0.00604395 (W 0.00611783) | 0.404494 ± 0.0402231 (W 0.0397794) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.314307 ± 0.000555666 (W 0.00364454) | 0.474797 ± 0.0456078 (W 0.0413841) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 24.1801 ± 0.06191 (W 0.181338) | 25.1596 ± 3.65005 (W 2.91558) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 24.1438 ± 0.140008 (W 0.121132) | 25.4151 ± 0.261502 (W 2.90512) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.554985 ± 0.00280688 (W 0.00258223) | 1.34964 ± 0.0300603 (W 0.0111011) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.625308 ± 0.00273796 (W 0.00266798) | 0.737716 ± 0.0186646 (W 0.00449457) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.2635 ± 0.055683 (W 0.0389814) | 61.9293 ± 0.612665 (W 0.0749721) | - | 🔵 faer-fork 4.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 16.5683 ± 0.245562 (W 0.053227) | 25.6424 ± 0.0225617 (W 0.0301845) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.4576 ± 0.00600158 (W 0.00439151) | 4.38146 ± 0.0878461 (W 0.0499267) | - | 🔵 faer-fork 3.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.22457 ± 0.0139542 (W 0.0148143) | 2.43648 ± 0.0281176 (W 0.1582) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 48.6715 ± 0.256835 (W 0.272807) | 153.115 ± 2.39691 (W 1.29974) | - | 🔵 faer-fork 3.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 68.8875 ± 0.351299 (W 0.15322) | 95.0425 ± 6.70316 (W 4.84353) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.1339 ± 0.00519237 (W 0.00419057) | 3.73722 ± 0.161565 (W 0.0514484) | - | 🔵 faer-fork 3.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 26.6656 ± 0.0965004 (W 0.0442225) | 122.469 ± 0.241158 (W 1.20747) | - | 🔵 faer-fork 4.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.33599 ± 0.0166395 (W 0.00590033) | 3.69684 ± 0.0247164 (W 0.0312187) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.32409 ± 0.0126435 (W 0.00456701) | 3.76409 ± 0.034506 (W 0.380334) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 73.7486 ± 0.0787417 (W 0.0821946) | 105.076 ± 0.335548 (W 1.01846) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 74.5596 ± 0.143997 (W 0.127933) | 106.845 ± 0.342586 (W 1.86651) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.56154 ± 0.020309 (W 0.00369615) | 2.48272 ± 0.0328941 (W 0.006788) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.56883 ± 0.00644981 (W 0.00445212) | 2.48448 ± 0.0222281 (W 0.00827034) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 43.3837 ± 0.0176051 (W 0.0631653) | 69.4967 ± 0.139495 (W 0.561852) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 43.4095 ± 0.0562317 (W 0.052128) | 69.4726 ± 1.05692 (W 0.67089) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.70425 ± 0.00647921 (W 0.003348) | 1.70293 ± 0.0119409 (W 0.00324232) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.67226 ± 0.0016536 (W 0.00374262) | 1.66879 ± 0.0114076 (W 0.00292521) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 152.286 ± 1.05467 (W 2.40147) | 151.822 ± 0.564197 (W 1.35893) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 144.907 ± 0.373493 (W 0.664065) | 143.205 ± 0.28857 (W 0.839558) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.2718 ± 0.0110551 (W 0.013588) | 2.25517 ± 0.0113207 (W 0.00490379) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.27001 ± 0.0111005 (W 0.00357087) | 2.25209 ± 0.012411 (W 0.00575067) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 138.11 ± 0.171898 (W 0.152299) | 138.191 ± 0.162399 (W 0.268645) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 137.971 ± 0.18721 (W 0.120287) | 138.159 ± 0.0317558 (W 0.115666) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.823094 ± 0.00125737 (W 0.00226658) | 0.886202 ± 0.0034718 (W 0.00659606) | 0.944629 ± 0.00297108 (W 0.00232307) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.859151 ± 0.00707453 (W 0.00397068) | 0.91741 ± 0.00180107 (W 0.00182276) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 40.2799 ± 0.0658057 (W 0.0575891) | 41.0216 ± 0.0444303 (W 0.033899) | 48.0126 ± 0.199696 (W 0.0890433) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 41.1757 ± 0.0619234 (W 0.0645294) | 42.1349 ± 0.0195007 (W 0.0256858) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.3001 ± 0.00960959 (W 0.0113123) | 4.53974 ± 0.0125 (W 0.00667172) | 5.63159 ± 0.0113633 (W 0.00842534) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 4.91941 ± 0.0137153 (W 0.011215) | 5.16056 ± 0.0309756 (W 0.0180052) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 229.216 ± 0.333137 (W 0.544786) | 231.987 ± 0.396672 (W 0.593923) | 287.097 ± 2.4558 (W 0.374989) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 243.624 ± 0.541086 (W 0.938394) | 248.112 ± 0.655934 (W 0.199973) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.02528 ± 0.00168765 (W 0.00437413) | 2.26885 ± 0.00750661 (W 0.00343431) | 2.22219 ± 0.0046448 (W 0.00431212) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 87.6884 ± 0.0854594 (W 0.212143) | 90.5404 ± 0.155637 (W 0.101378) | 100.111 ± 2.22612 (W 0.274644) | · | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 5.29767 ± 0.0385379 (W 0.00727635) | 5.85105 ± 0.0100598 (W 0.00667472) | 8.36044 ± 0.026024 (W 0.0157989) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.36646 ± 0.00623106 (W 0.0175215) | 5.87581 ± 0.0362643 (W 0.0118036) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 260.436 ± 0.29639 (W 0.457183) | 267.372 ± 0.228648 (W 0.540655) | 359.547 ± 0.670559 (W 0.851911) | · | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 261.444 ± 0.647601 (W 6.84913) | 268.156 ± 0.376144 (W 6.51417) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.39507 ± 0.0025268 (W 0.00474436) | 2.94416 ± 0.0145077 (W 0.00605836) | 3.61178 ± 0.0134949 (W 0.00862759) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.42617 ± 0.00633723 (W 0.0081862) | 2.95602 ± 0.00142628 (W 0.00517572) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 100.77 ± 0.390992 (W 0.276311) | 110.18 ± 0.125313 (W 0.176622) | 140.89 ± 1.13905 (W 0.361306) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 100.512 ± 0.501737 (W 0.0802885) | 110.399 ± 0.0523399 (W 0.425056) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.781988 ± 0.00397337 (W 0.00329741) | 1.70021 ± 0.00228807 (W 0.00440186) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.820027 ± 0.00326828 (W 0.00499528) | 1.66241 ± 0.00938006 (W 0.00506298) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 47.4942 ± 0.141693 (W 0.161855) | 49.5375 ± 0.252435 (W 0.332074) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 46.7869 ± 0.14708 (W 0.174224) | 49.2438 ± 0.272703 (W 0.290452) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.656422 ± 0.00436967 (W 0.00198195) | 0.652395 ± 0.00413303 (W 0.00252226) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.655526 ± 0.00130329 (W 0.00458731) | 0.652336 ± 0.000579242 (W 0.00296321) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 39.191 ± 0.121164 (W 0.21382) | 39.4509 ± 0.0463641 (W 0.0654314) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 39.1241 ± 0.114298 (W 0.17762) | 39.4851 ± 0.0545581 (W 0.100182) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.520148 ± 0.00360245 (W 0.00189769) | 0.619776 ± 0.00260082 (W 0.00189327) | 0.943385 ± 0.00257287 (W 0.00337759) | 🔵 faer-fork 1.19× faster than `faer-upstream` | 🔵 faer-fork 1.81× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.573693 ± 0.00040029 (W 0.00271612) | 0.680563 ± 0.00203912 (W 0.00185925) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 14.7304 ± 0.0233387 (W 0.0379393) | 16.5212 ± 0.0259393 (W 0.0336546) | 47.831 ± 0.102593 (W 0.0736184) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 3.25× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 15.6986 ± 0.0134598 (W 0.0700166) | 17.6838 ± 0.0454935 (W 0.0367673) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.7956 ± 0.00774956 (W 0.00730565) | 2.24728 ± 0.00808254 (W 0.00606806) | 2.48726 ± 0.0017922 (W 0.00305451) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.41463 ± 0.00710294 (W 0.0109289) | 2.84639 ± 0.00769372 (W 0.0320734) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 71.8282 ± 0.0586314 (W 0.102774) | 81.199 ± 0.0934932 (W 0.0779198) | 86.3892 ± 1.21669 (W 0.266591) | 🔵 faer-fork 1.13× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 85.5464 ± 0.0282511 (W 0.112306) | 96.0432 ± 6.96423 (W 0.168063) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.14476 ± 0.00447011 (W 0.00346432) | 1.6095 ± 0.0114054 (W 0.00487411) | 1.33976 ± 0.000905771 (W 0.00343957) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 32.1826 ± 0.0811846 (W 0.0358599) | 42.0452 ± 0.0445629 (W 0.0534764) | 31.0264 ± 0.263477 (W 0.253062) | 🔵 faer-fork 1.31× faster than `faer-upstream` | · |  |
| `svd_thin/1024x64/b1` | 2.57499 ± 0.0020584 (W 0.00290566) | 3.53556 ± 0.00648561 (W 0.00410893) | 5.31323 ± 0.0281704 (W 0.00858853) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 2.06× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.59201 ± 0.00216994 (W 0.00418319) | 3.57189 ± 0.00405246 (W 0.00355295) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 86.2587 ± 0.16075 (W 0.103936) | 105.312 ± 0.0223934 (W 0.229173) | 127.706 ± 0.601024 (W 0.6516) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 87.1286 ± 0.153215 (W 0.0952055) | 106.162 ± 0.179046 (W 0.0779245) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.5671 ± 0.00176989 (W 0.00470181) | 2.32834 ± 0.00758184 (W 0.0034049) | 2.72915 ± 0.00567519 (W 0.0142542) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 1.74× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.57171 ± 0.00645604 (W 0.00612922) | 2.35139 ± 0.012708 (W 0.00663315) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 40.3012 ± 0.0393984 (W 0.0306417) | 59.6339 ± 0.0878489 (W 0.0962639) | 47.0856 ± 0.0546436 (W 0.296353) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 40.2908 ± 0.103282 (W 0.087987) | 59.5428 ± 0.0895795 (W 0.166839) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.65334 ± 0.00777876 (W 0.00456843) | 1.70187 ± 0.0097838 (W 0.00757649) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.710627 ± 0.00248231 (W 0.00492149) | 1.65886 ± 0.00880192 (W 0.00420958) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 23.416 ± 0.073844 (W 0.119315) | 26.3058 ± 0.179934 (W 0.106864) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 23.9324 ± 0.147707 (W 0.0936485) | 26.5109 ± 0.0985778 (W 0.20366) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.37501 ± 0.01341 (W 0.00392926) | 0.511164 ± 0.0394057 (W 0.0427499) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.381251 ± 0.00308818 (W 0.00389062) | 0.54114 ± 0.046172 (W 0.0488134) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 25.9875 ± 1.10059 (W 1.34287) | 26.4998 ± 0.0443676 (W 0.102745) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 25.199 ± 0.618602 (W 0.676956) | 26.3271 ± 2.89144 (W 2.38567) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.462233 ± 0.00116959 (W 0.00161408) | 0.57333 ± 0.00198142 (W 0.00248161) | 0.943822 ± 0.00534676 (W 0.00281811) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 2.04× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.518577 ± 0.00444788 (W 0.00288696) | 0.635438 ± 0.00172549 (W 0.00385664) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.5874 ± 0.0147818 (W 0.0245937) | 12.5707 ± 0.0141589 (W 0.0234911) | 47.8525 ± 0.0936813 (W 0.0691735) | 🔵 faer-fork 1.19× faster than `faer-upstream` | 🔵 faer-fork 4.52× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.5453 ± 0.0214315 (W 0.0263258) | 13.6424 ± 0.0058902 (W 0.028688) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.40297 ± 0.000337396 (W 0.00332812) | 2.24071 ± 0.0114961 (W 0.0291664) | 2.17604 ± 0.00668608 (W 0.00357881) | 🔵 faer-fork 1.60× faster than `faer-upstream` | 🔵 faer-fork 1.55× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.0434 ± 0.0142619 (W 0.00525594) | 2.33853 ± 0.00729569 (W 0.0529408) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 48.7184 ± 0.0473727 (W 0.0889765) | 69.914 ± 0.420666 (W 1.09566) | 48.3248 ± 0.417508 (W 0.300327) | 🔵 faer-fork 1.44× faster than `faer-upstream` | · |  |
| `qr/4096x256/b1 [api-inclusive]` | 62.3848 ± 0.283534 (W 0.284652) | 75.4246 ± 5.62226 (W 1.36488) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.03499 ± 0.00452225 (W 0.00241112) | 1.6716 ± 0.00730932 (W 0.0132212) | 0.815659 ± 0.00154364 (W 0.0023508) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔴 faer-fork 1.27× slower than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 23.7634 ± 0.0336846 (W 0.107237) | 39.6727 ± 0.0590256 (W 0.181124) | 17.0191 ± 0.240999 (W 0.108487) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔴 faer-fork 1.40× slower than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 2.20408 ± 0.0037773 (W 0.00963704) | 3.55214 ± 0.0235764 (W 0.0278977) | 4.21522 ± 0.0131202 (W 0.00720187) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔵 faer-fork 1.91× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.22465 ± 0.0151265 (W 0.00508517) | 3.60639 ± 0.192334 (W 0.0198594) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 61.1203 ± 0.0700146 (W 0.371594) | 92.5221 ± 1.28281 (W 2.06461) | 71.8974 ± 0.74648 (W 0.436993) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 62.2229 ± 0.137598 (W 0.129724) | 94.0228 ± 0.273769 (W 1.81151) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.47241 ± 0.00201054 (W 0.00615997) | 2.39715 ± 0.00793312 (W 0.0119686) | 2.21734 ± 0.00563601 (W 0.00297803) | 🔵 faer-fork 1.63× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.4887 ± 0.000746538 (W 0.00201387) | 2.4018 ± 0.00907619 (W 0.0123683) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.4288 ± 0.0258373 (W 0.387472) | 57.0826 ± 0.215613 (W 0.345587) | 30.8424 ± 0.160975 (W 0.239018) | 🔵 faer-fork 1.82× faster than `faer-upstream` | · |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.3444 ± 0.0503156 (W 0.0458255) | 57.5029 ± 0.157656 (W 0.480681) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.66176 ± 0.0154056 (W 0.00296231) | 0.922981 ± 0.000555879 (W 0.00608138) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.567617 ± 0.012789 (W 0.00268932) | 0.513768 ± 0.0159634 (W 0.00700165) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 25.8315 ± 0.0966824 (W 0.116596) | 47.2661 ± 0.338884 (W 0.092054) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 24.0314 ± 0.145587 (W 0.217728) | 26.2979 ± 0.0658008 (W 0.0744571) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.279524 ± 0.00386288 (W 0.000742497) | 0.2778 ± 0.000837 (W 0.00215249) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.279711 ± 0.0023429 (W 0.00107414) | 0.276395 ± 0.000803132 (W 0.00171597) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.4221 ± 0.066334 (W 0.0343622) | 17.4072 ± 0.0272903 (W 0.11139) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.4662 ± 0.0655052 (W 0.0323017) | 17.3778 ± 0.0177935 (W 0.0542528) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.318043 ± 0.00322212 (W 0.00242029) | 0.762009 ± 0.00663878 (W 0.00222043) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.263023 ± 0.00186403 (W 0.00200509) | 0.307647 ± 0.00446206 (W 0.00640605) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 7.60754 ± 0.0114398 (W 0.0267547) | 27.3343 ± 0.0300351 (W 0.0449099) | - | 🔵 faer-fork 3.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 8.43425 ± 0.214144 (W 0.130051) | 10.7368 ± 0.0326682 (W 0.0222393) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.95484 ± 0.00561512 (W 0.00146851) | 1.8089 ± 0.00315466 (W 0.00271438) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.03312 ± 0.00123282 (W 0.00307473) | 1.1653 ± 0.00157334 (W 0.00293707) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 36.3724 ± 0.14111 (W 0.209099) | 69.121 ± 0.108466 (W 0.356714) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 38.4709 ± 0.213327 (W 0.116337) | 41.0144 ± 0.0928027 (W 0.0409111) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.68703 ± 0.000501501 (W 0.00709197) | 1.52194 ± 0.000763767 (W 0.00413054) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 18.5115 ± 0.25395 (W 0.115942) | 51.1872 ± 0.257117 (W 0.0838022) | - | 🔵 faer-fork 2.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.36695 ± 0.00773223 (W 0.00194212) | 1.58398 ± 0.00345813 (W 0.00422303) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.37406 ± 0.00410051 (W 0.00552633) | 1.73426 ± 0.00266959 (W 0.00722966) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 46.3834 ± 0.833943 (W 0.0567873) | 49.8897 ± 0.776933 (W 0.0429911) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 45.2421 ± 0.886141 (W 0.375174) | 50.2119 ± 0.772014 (W 0.103366) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.739499 ± 0.00583235 (W 0.00283615) | 1.14233 ± 0.00162556 (W 0.00396132) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.742853 ± 0.00175071 (W 0.00409356) | 1.13936 ± 0.0025841 (W 0.00439099) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 21.9098 ± 0.0522559 (W 0.0262589) | 27.3128 ± 0.717958 (W 0.0493663) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 21.8861 ± 0.0328444 (W 0.122918) | 27.4199 ± 0.708679 (W 0.0832985) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.454267 ± 0.00544359 (W 0.00547743) | 0.93408 ± 0.0064666 (W 0.00946335) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.48817 ± 0.00396364 (W 0.00937782) | 0.510422 ± 0.00286237 (W 0.0110546) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.53975 ± 0.0822967 (W 0.0343039) | 28.4482 ± 0.0677123 (W 0.0525339) | - | 🔵 faer-fork 2.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.2339 ± 0.128615 (W 0.0707827) | 12.2566 ± 0.0247904 (W 0.0909979) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.087453 ± 0.00202317 (W 0.00187132) | 0.082735 ± 0.000231561 (W 0.00187833) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.088234 ± 0.000360185 (W 0.00132483) | 0.082444 ± 0.000278837 (W 0.000732942) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.64086 ± 0.0191391 (W 0.0146409) | 4.55941 ± 0.00283509 (W 0.0129649) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.62647 ± 0.0224201 (W 0.0151623) | 4.56477 ± 0.0115611 (W 0.0620318) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.220953 ± 0.00125999 (W 0.00151693) | 0.772168 ± 0.0184085 (W 0.0114888) | - | 🔵 faer-fork 3.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.253733 ± 0.00263162 (W 0.00558038) | 0.308134 ± 0.00258175 (W 0.00519779) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.8789 ± 0.0130906 (W 0.0161281) | 24.0188 ± 0.151751 (W 0.0630249) | - | 🔵 faer-fork 6.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.43978 ± 0.0718556 (W 0.0755765) | 7.81295 ± 0.00267795 (W 0.015313) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.55334 ± 0.0037434 (W 0.0034711) | 2.24118 ± 0.00783855 (W 0.0241476) | - | 🔵 faer-fork 4.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.744055 ± 0.000285311 (W 0.00295273) | 0.945744 ± 0.0184099 (W 0.0226569) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 12.7352 ± 0.178543 (W 0.0164589) | 56.5814 ± 0.0768434 (W 0.133151) | - | 🔵 faer-fork 4.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 17.7504 ± 0.135071 (W 0.112452) | 20.376 ± 0.0575161 (W 0.019795) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.467702 ± 0.00195551 (W 0.00267178) | 2.1438 ± 0.0296904 (W 0.0246129) | - | 🔵 faer-fork 4.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 8.01042 ± 0.033175 (W 0.0281058) | 51.5465 ± 0.153737 (W 0.188558) | - | 🔵 faer-fork 6.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.01683 ± 0.00164619 (W 0.00384982) | 1.49975 ± 0.00155546 (W 0.00512118) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.02394 ± 0.00179354 (W 0.00284817) | 1.51313 ± 0.00586783 (W 0.00398578) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 21.4746 ± 0.0294746 (W 0.028632) | 28.4358 ± 0.54654 (W 0.0814898) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 21.6967 ± 0.0520874 (W 0.0263047) | 28.4819 ± 0.0710578 (W 0.0577532) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.678423 ± 0.00930549 (W 0.00529726) | 1.24623 ± 0.00606127 (W 0.0040479) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.690751 ± 0.00568908 (W 0.002284) | 1.25878 ± 0.00650839 (W 0.00845251) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 14.4544 ± 0.00633412 (W 0.0187299) | 21.5169 ± 0.0826571 (W 0.0404675) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 14.4182 ± 0.050775 (W 0.0167041) | 21.6416 ± 0.0383204 (W 0.0505168) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.446082 ± 0.00419763 (W 0.00606675) | 0.931703 ± 0.00874521 (W 0.00303742) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.502819 ± 0.00204171 (W 0.00932805) | 0.507811 ± 0.00762645 (W 0.00644638) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 7.03365 ± 0.0284677 (W 0.0293042) | 25.2216 ± 0.0519702 (W 0.0499646) | - | 🔵 faer-fork 3.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.96181 ± 0.132347 (W 0.0577204) | 10.4158 ± 0.0644024 (W 0.0400409) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.077866 ± 0.00091573 (W 0.0016679) | 0.093324 ± 0.00144981 (W 0.00291483) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.077424 ± 0.00128995 (W 0.00203197) | 0.092202 ± 0.00328403 (W 0.00360072) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 2.58332 ± 0.0623173 (W 0.015575) | 2.65358 ± 0.362409 (W 0.370052) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 2.59356 ± 0.0111752 (W 0.0197155) | 3.25295 ± 0.372178 (W 0.402779) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.198446 ± 0.000512041 (W 0.00295491) | 0.816423 ± 0.00334525 (W 0.00765403) | - | 🔵 faer-fork 4.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.246881 ± 0.00128524 (W 0.00217142) | 0.303847 ± 0.00217951 (W 0.00671302) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.8291 ± 0.00802423 (W 0.0106134) | 23.8772 ± 0.0674091 (W 0.0258785) | - | 🔵 faer-fork 8.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.01619 ± 0.0909774 (W 0.0886054) | 7.35081 ± 0.0164407 (W 0.0160563) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.530464 ± 0.0014463 (W 0.0055355) | 2.31954 ± 0.0133455 (W 0.046867) | - | 🔵 faer-fork 4.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.708643 ± 0.000907789 (W 0.00362502) | 1.04661 ± 0.00335493 (W 0.0180157) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 9.24501 ± 0.0809239 (W 0.0229407) | 55.8474 ± 0.342093 (W 0.272491) | - | 🔵 faer-fork 6.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.5935 ± 0.234645 (W 0.153173) | 20.6427 ± 0.369798 (W 0.528272) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.441726 ± 0.00479792 (W 0.00280743) | 2.18329 ± 0.0331324 (W 0.0330375) | - | 🔵 faer-fork 4.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.89513 ± 0.187054 (W 0.014739) | 51.4084 ± 0.284767 (W 0.199895) | - | 🔵 faer-fork 8.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1` | 0.973576 ± 0.00118856 (W 0.00197638) | 1.6202 ± 0.0139458 (W 0.00932247) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.981426 ± 0.00193479 (W 0.00292961) | 1.62244 ± 0.00924307 (W 0.0107543) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 17.8055 ± 0.0238435 (W 0.0235022) | 29.2824 ± 0.732197 (W 0.0932859) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 17.9005 ± 0.0837641 (W 0.0476598) | 29.3909 ± 0.0736244 (W 0.901958) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.675269 ± 0.00255453 (W 0.00194649) | 1.35479 ± 0.00629894 (W 0.00508549) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.673688 ± 0.000920609 (W 0.0033505) | 1.36617 ± 0.010151 (W 0.004789) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 12.9557 ± 0.803772 (W 0.0180448) | 22.8332 ± 0.109844 (W 0.065887) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.0113 ± 0.0466499 (W 0.0175026) | 22.9018 ± 0.0930411 (W 0.0295077) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.494962 ± 0.00133542 (W 0.0026523) | 0.487989 ± 0.015708 (W 0.00346434) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.463228 ± 0.00231038 (W 0.00875404) | 0.472912 ± 0.0131436 (W 0.00372788) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 24.5302 ± 0.0788282 (W 0.0369674) | 24.5177 ± 0.121315 (W 0.0762927) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 21.5548 ± 0.0685003 (W 0.0403104) | 21.6075 ± 0.223909 (W 0.0575732) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.301622 ± 0.00200067 (W 0.00293317) | 0.301192 ± 0.00224288 (W 0.000717514) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.303065 ± 0.00149733 (W 0.00239178) | 0.301612 ± 0.00101217 (W 0.000407182) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.8108 ± 0.00331033 (W 0.0358596) | 17.8054 ± 0.0414312 (W 0.0181515) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.8163 ± 0.030439 (W 0.0654651) | 17.8897 ± 0.0451424 (W 0.0491319) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.147565 ± 0.000454295 (W 0.00192654) | 0.240769 ± 0.000911862 (W 0.00246061) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.157013 ± 0.00111437 (W 0.00209818) | 0.250537 ± 0.00121112 (W 0.0026841) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.66409 ± 0.11179 (W 0.0094011) | 7.06566 ± 0.0304001 (W 0.0153661) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.77827 ± 0.122581 (W 0.00461679) | 7.17341 ± 0.0411793 (W 0.0249851) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.745239 ± 0.00096179 (W 0.00257478) | 0.931424 ± 0.00286444 (W 0.00431642) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.904835 ± 0.00203809 (W 0.00223536) | 1.1007 ± 0.00399761 (W 0.00261846) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 32.0756 ± 0.103027 (W 0.0538697) | 33.9221 ± 0.245361 (W 0.033218) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 35.661 ± 0.159994 (W 0.0550231) | 37.5884 ± 0.211262 (W 0.0430572) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.447585 ± 0.00367377 (W 0.00305838) | 0.629058 ± 0.00209338 (W 0.00303458) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.9565 ± 0.0579242 (W 0.0144655) | 15.6914 ± 0.0299147 (W 0.0231382) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.36692 ± 0.00387601 (W 0.00387977) | 1.58446 ± 0.00518637 (W 0.00667178) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.36795 ± 0.00913814 (W 0.00809322) | 1.73086 ± 0.00693357 (W 0.0100651) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 42.2571 ± 0.223536 (W 0.0395099) | 46.2499 ± 0.1056 (W 0.114422) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 42.405 ± 0.172839 (W 0.0977006) | 46.2303 ± 0.240453 (W 0.0677288) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.699053 ± 0.00210084 (W 0.00265476) | 1.09797 ± 0.0034314 (W 0.00175187) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.700756 ± 0.00157982 (W 0.00297153) | 1.09821 ± 0.00868345 (W 0.00529494) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 18.6971 ± 0.0758585 (W 0.0243111) | 24.0434 ± 0.00766411 (W 0.0364964) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 18.738 ± 0.0821407 (W 0.045596) | 24.1112 ± 0.0357524 (W 0.077202) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.371702 ± 0.00168797 (W 0.00360674) | 0.486379 ± 0.00238219 (W 0.00369031) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.389999 ± 0.00315569 (W 0.00304185) | 0.453225 ± 0.000923595 (W 0.00264281) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.18176 ± 0.00785367 (W 0.0192473) | 9.52642 ± 0.048463 (W 0.0679284) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.68168 ± 0.0225625 (W 0.00929047) | 8.89411 ± 0.00867866 (W 0.0176542) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.103223 ± 0.000751165 (W 0.000545892) | 0.100367 ± 0.000673796 (W 0.000468124) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.105116 ± 0.000793802 (W 0.00137566) | 0.100889 ± 0.000363148 (W 0.000859273) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.49607 ± 0.00747735 (W 0.0229596) | 5.42844 ± 0.00923705 (W 0.0188104) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.48545 ± 0.0209613 (W 0.011198) | 5.43234 ± 0.0138964 (W 0.0164832) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.140572 ± 0.00125973 (W 0.000967561) | 0.237303 ± 0.00280694 (W 0.00225894) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.151974 ± 0.000805126 (W 0.00120068) | 0.250027 ± 0.00445885 (W 0.00191758) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.6995 ± 0.00757083 (W 0.00414414) | 4.17128 ± 0.00901975 (W 0.0174396) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.90581 ± 0.00943921 (W 0.00621966) | 4.40006 ± 0.00562634 (W 0.0185747) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.488493 ± 0.00179734 (W 0.0024122) | 0.849734 ± 0.00351464 (W 0.0020666) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.617079 ± 0.000809459 (W 0.00368375) | 0.916607 ± 0.0286852 (W 0.0211324) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 11.8351 ± 0.0120913 (W 0.0181817) | 15.9856 ± 0.103081 (W 0.0255892) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.9355 ± 0.0801017 (W 0.0410739) | 17.0177 ± 1.66287 (W 0.0334961) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.388897 ± 0.00157924 (W 0.00233385) | 0.750578 ± 0.00320936 (W 0.00683841) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 6.70159 ± 0.0283064 (W 0.0100999) | 11.0657 ± 0.0193668 (W 0.035169) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.0049 ± 0.00292216 (W 0.00260084) | 1.51479 ± 0.00601163 (W 0.00280876) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.00584 ± 0.00147458 (W 0.00122235) | 1.52225 ± 0.00228892 (W 0.0059084) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 18.7884 ± 0.0324294 (W 0.0266996) | 25.6397 ± 0.0391123 (W 0.0825668) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 18.8692 ± 0.08191 (W 0.0294543) | 25.7388 ± 0.0784529 (W 0.0623788) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.646796 ± 0.00414757 (W 0.00321711) | 1.20503 ± 0.00205319 (W 0.00479289) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.646223 ± 0.00347746 (W 0.0033126) | 1.21561 ± 0.00947489 (W 0.00582504) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 11.092 ± 0.0705336 (W 0.0164146) | 18.1622 ± 0.0535264 (W 0.0235918) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 11.0919 ± 0.0531219 (W 0.0205745) | 18.1721 ± 0.076285 (W 0.0431407) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.384318 ± 0.00168822 (W 0.0064059) | 0.486487 ± 0.00175087 (W 0.003077) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.395728 ± 0.00261458 (W 0.00648765) | 0.454488 ± 0.0104826 (W 0.0104725) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.74596 ± 0.0749151 (W 0.0405027) | 7.11815 ± 0.0305503 (W 0.167333) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 6.67188 ± 0.029625 (W 0.0219796) | 7.17204 ± 0.658213 (W 0.150981) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.068968 ± 0.000631404 (W 0.000330803) | 0.114503 ± 0.0016827 (W 0.0022547) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.069331 ± 0.00115062 (W 0.000930079) | 0.115887 ± 0.00152215 (W 0.00361642) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.08978 ± 0.144015 (W 0.0919075) | 3.56217 ± 0.0451483 (W 0.321262) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.10795 ± 0.192668 (W 0.160657) | 3.57529 ± 0.0388935 (W 0.278964) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.134421 ± 0.00144646 (W 0.00125937) | 0.236471 ± 0.00148995 (W 0.0024525) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.147976 ± 0.000929823 (W 0.00168068) | 0.252862 ± 0.000667929 (W 0.00257079) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.23523 ± 0.0130136 (W 0.0050593) | 3.75451 ± 0.0120247 (W 0.00923719) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.46144 ± 0.00534098 (W 0.00595336) | 4.00146 ± 0.0180597 (W 0.0206654) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.443177 ± 0.00323727 (W 0.00247057) | 0.961324 ± 0.00191158 (W 0.00329555) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.582923 ± 0.000925322 (W 0.00385914) | 0.988586 ± 0.00261389 (W 0.0256345) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 8.27459 ± 0.0299506 (W 0.0103977) | 16.622 ± 0.184999 (W 0.117209) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 11.5294 ± 0.0425325 (W 0.0542269) | 16.8255 ± 0.423666 (W 0.959397) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.377754 ± 0.00201281 (W 0.00249577) | 0.842262 ± 0.00601855 (W 0.00606369) | - | 🔵 faer-fork 2.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.22532 ± 0.025478 (W 0.0109851) | 12.2427 ± 0.0547046 (W 0.0502706) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.956403 ± 0.0045888 (W 0.00326338) | 1.62257 ± 0.00506497 (W 0.0210449) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.965773 ± 0.00533746 (W 0.00293832) | 1.64756 ± 0.00965666 (W 0.01918) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 15.1169 ± 0.049663 (W 0.0277146) | 26.5019 ± 0.0335028 (W 0.138201) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 15.1913 ± 0.0143186 (W 0.0447717) | 26.6557 ± 0.664598 (W 0.537591) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.631705 ± 0.00131804 (W 0.00235603) | 1.3036 ± 0.00461616 (W 0.00432002) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.631487 ± 0.000756599 (W 0.00215497) | 1.30861 ± 0.00317204 (W 0.00641918) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 9.58061 ± 0.0296405 (W 0.013673) | 19.404 ± 0.0865254 (W 0.0430387) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 9.58746 ± 0.0112654 (W 0.0129706) | 19.4718 ± 0.0293947 (W 0.0674402) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.23428 ± 0.00283217 (W 0.00336566) | 1.38809 ± 0.354099 (W 0.00893438) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.1072 ± 0.00299514 (W 0.00246017) | 1.00016 ± 0.00393556 (W 0.00476238) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 63.6354 ± 1.01981 (W 0.207479) | 107.607 ± 0.553486 (W 0.572643) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 60.6232 ± 0.878265 (W 0.410024) | 62.2031 ± 0.776189 (W 0.740122) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.10606 ± 0.00554233 (W 0.00563669) | 1.1069 ± 0.00287003 (W 0.00177046) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.10675 ± 0.000274716 (W 0.00335259) | 1.10641 ± 0.000651077 (W 0.00245301) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 68.4867 ± 0.0852377 (W 0.0447487) | 68.5248 ± 0.135517 (W 0.0859915) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 68.543 ± 0.125369 (W 0.128957) | 68.5136 ± 0.0631953 (W 0.16557) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.648217 ± 0.00137463 (W 0.0023606) | 0.987821 ± 0.00138788 (W 0.00563332) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.627559 ± 0.0028218 (W 0.00256303) | 0.604075 ± 0.0105677 (W 0.00282769) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 23.1484 ± 0.076261 (W 0.0347376) | 46.9073 ± 0.280161 (W 0.121139) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 24.9486 ± 0.134644 (W 0.0769254) | 27.2784 ± 0.00564875 (W 0.0353787) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.58642 ± 0.00670486 (W 0.005569) | 4.38587 ± 0.0129303 (W 0.00579247) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.84754 ± 0.00537582 (W 0.00451517) | 3.03768 ± 0.0144881 (W 0.00567176) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 126.327 ± 0.367952 (W 0.470901) | 168.166 ± 0.144697 (W 0.25182) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 129.957 ± 0.0411078 (W 0.454312) | 134.712 ± 0.311927 (W 0.327303) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.4765 ± 0.00833845 (W 0.00392765) | 3.22895 ± 0.00821812 (W 0.00355622) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 54.7605 ± 0.380158 (W 0.192431) | 96.8913 ± 0.0784242 (W 0.207074) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.14503 ± 0.00983351 (W 0.00474258) | 3.39955 ± 0.00256228 (W 0.00439959) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.15755 ± 0.00678729 (W 0.00549467) | 3.40648 ± 0.00352379 (W 0.0052197) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 141.643 ± 0.360434 (W 0.347481) | 146.244 ± 0.149879 (W 0.150252) | - | · | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 141.874 ± 0.167933 (W 0.141901) | 146.495 ± 0.0519276 (W 0.216001) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.56417 ± 0.00421469 (W 0.0050168) | 2.03451 ± 0.00369952 (W 0.00643424) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.56911 ± 0.0146469 (W 0.0050831) | 2.03086 ± 0.00413144 (W 0.00583887) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 60.7528 ± 0.0753447 (W 0.0764257) | 67.4082 ± 0.0669429 (W 0.0675914) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 60.8112 ± 0.0727153 (W 0.0750767) | 67.4838 ± 0.00696122 (W 0.0800696) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.699885 ± 0.0118783 (W 0.00401457) | 1.35582 ± 0.0123402 (W 0.00664695) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.742667 ± 0.00338162 (W 0.00763725) | 1.00286 ± 0.0130286 (W 0.0074853) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 20.4603 ± 0.0980239 (W 0.0344873) | 50.1125 ± 0.152028 (W 0.145878) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 21.5322 ± 0.0692553 (W 0.0377652) | 25.9405 ± 1.0523 (W 0.826993) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.303375 ± 0.002675 (W 0.0016347) | 0.294429 ± 0.00211927 (W 0.00190829) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.298037 ± 0.000522032 (W 0.0014281) | 0.295361 ± 0.00132681 (W 0.00192207) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.5159 ± 0.0688631 (W 0.0362934) | 17.4982 ± 0.0129762 (W 0.0228824) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.5436 ± 0.0489399 (W 0.0238446) | 17.4793 ± 0.0223334 (W 0.0259615) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.470519 ± 0.00183939 (W 0.00273989) | 0.915621 ± 0.00621809 (W 0.00415622) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.513837 ± 0.00382422 (W 0.00322868) | 0.495936 ± 0.00535608 (W 0.00299638) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 11.524 ± 0.106998 (W 0.057218) | 33.9179 ± 0.178964 (W 0.111529) | - | 🔵 faer-fork 2.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 12.7911 ± 0.121024 (W 0.0767932) | 15.1506 ± 0.0331681 (W 0.0441154) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.17562 ± 0.00671495 (W 0.00779705) | 3.55994 ± 0.101707 (W 0.109048) | - | 🔵 faer-fork 3.03× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.53639 ± 0.00230732 (W 0.00313426) | 2.00931 ± 0.153708 (W 0.0450471) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 41.5626 ± 0.213804 (W 0.0885153) | 95.821 ± 0.265456 (W 0.528752) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 50.2069 ± 0.0788344 (W 0.115941) | 59.8186 ± 0.0420807 (W 0.0997385) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.871155 ± 0.00191992 (W 0.00225158) | 3.22407 ± 0.046459 (W 0.0909513) | - | 🔵 faer-fork 3.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 22.7773 ± 0.0340535 (W 0.0190572) | 76.6447 ± 0.164374 (W 0.42891) | - | 🔵 faer-fork 3.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.73672 ± 0.00524172 (W 0.00706616) | 2.3395 ± 0.00547774 (W 0.00486813) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.7454 ± 0.00367455 (W 0.00712879) | 2.35315 ± 0.00353755 (W 0.00819665) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 54.2967 ± 0.101837 (W 0.0961595) | 65.7651 ± 0.0918651 (W 0.161423) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 54.6902 ± 0.159758 (W 0.264642) | 65.9958 ± 0.065435 (W 0.124651) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.12835 ± 0.00485586 (W 0.00404987) | 1.82419 ± 0.0137693 (W 0.00445511) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.12949 ± 0.0060461 (W 0.00354423) | 1.82085 ± 0.00432385 (W 0.004411) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 30.5566 ± 0.0334635 (W 0.0400525) | 42.2329 ± 0.044467 (W 0.0448742) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 30.573 ± 0.0596421 (W 0.0439364) | 42.3668 ± 0.0497265 (W 0.0363994) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.634353 ± 0.0081235 (W 0.00489967) | 1.35567 ± 0.00403571 (W 0.00759081) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.706886 ± 0.000817959 (W 0.00449234) | 1.00498 ± 0.00646122 (W 0.00549877) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 13.1201 ± 0.0417483 (W 0.0536178) | 42.1361 ± 0.0496174 (W 0.0231368) | - | 🔵 faer-fork 3.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 15.6475 ± 0.0802759 (W 0.035774) | 20.6245 ± 0.52597 (W 0.364433) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.178763 ± 0.00287051 (W 0.00428915) | 0.269864 ± 0.0166845 (W 0.0134147) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.175141 ± 0.00169897 (W 0.003049) | 0.245698 ± 0.0162904 (W 0.0167011) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.4919 ± 0.0173806 (W 0.0404904) | 12.3516 ± 1.79898 (W 1.71662) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.48908 ± 0.0145705 (W 0.0813068) | 12.3885 ± 1.86603 (W 1.4495) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.432977 ± 0.000843419 (W 0.00258649) | 0.941606 ± 0.00907446 (W 0.00894143) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.496679 ± 0.0016458 (W 0.00244629) | 0.477423 ± 0.00221333 (W 0.00304096) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.46818 ± 0.0184179 (W 0.0156398) | 32.2905 ± 0.137546 (W 0.110859) | - | 🔵 faer-fork 3.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 11.0282 ± 0.031813 (W 0.0406846) | 13.2587 ± 0.0443647 (W 0.0374129) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.980851 ± 0.0041555 (W 0.00242123) | 3.27035 ± 0.0545148 (W 0.102892) | - | 🔵 faer-fork 3.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.3569 ± 0.00747053 (W 0.00473443) | 1.74611 ± 0.0138948 (W 0.0466351) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 26.8649 ± 0.11035 (W 0.122653) | 87.8056 ± 0.456786 (W 0.701508) | - | 🔵 faer-fork 3.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 37.8945 ± 0.0922545 (W 0.118506) | 49.9296 ± 1.63702 (W 2.87933) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.794282 ± 0.00263729 (W 0.00450691) | 3.03286 ± 0.0507629 (W 0.126478) | - | 🔵 faer-fork 3.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 16.5386 ± 0.0514214 (W 0.0267307) | 73.9606 ± 0.719932 (W 0.624132) | - | 🔵 faer-fork 4.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.54805 ± 0.00326818 (W 0.00723285) | 2.43685 ± 0.0108967 (W 0.0158111) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.56829 ± 0.00517062 (W 0.00398218) | 2.45116 ± 0.01299 (W 0.0140268) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 41.1168 ± 0.0548546 (W 0.0888211) | 60.4016 ± 0.332956 (W 0.498599) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 41.3976 ± 0.0459943 (W 0.072682) | 61.086 ± 0.297567 (W 1.19792) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.08486 ± 0.00260065 (W 0.00457248) | 1.91078 ± 0.00355456 (W 0.00481502) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.08712 ± 0.00579349 (W 0.0037066) | 1.91284 ± 0.00623411 (W 0.00945592) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 25.6248 ± 0.0751549 (W 0.0349696) | 42.3602 ± 0.152899 (W 0.246581) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 25.6934 ± 0.578796 (W 0.0416808) | 42.4075 ± 0.0420131 (W 0.194136) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.00015 ± 0.000249574 (W 0.00213438) | 0.935244 ± 0.00248247 (W 0.00182091) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.985836 ± 0.00031687 (W 0.0020722) | 0.918901 ± 0.00566413 (W 0.00271744) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 59.8413 ± 0.471473 (W 0.121469) | 59.1128 ± 0.466082 (W 0.185492) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 56.8978 ± 0.646715 (W 0.0928031) | 55.7516 ± 0.701197 (W 0.0727049) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.15038 ± 0.00204371 (W 0.00623127) | 1.15127 ± 0.0048549 (W 0.00413534) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.14911 ± 0.00405776 (W 0.00358305) | 1.15103 ± 0.0018244 (W 0.00178773) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 69.2723 ± 0.0118226 (W 0.0472819) | 69.2415 ± 0.0714479 (W 0.10176) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 69.2591 ± 0.0196126 (W 0.122125) | 69.3093 ± 0.116962 (W 0.0992961) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.488922 ± 0.00141164 (W 0.00240327) | 0.500312 ± 0.00429456 (W 0.00330604) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.506609 ± 0.00686631 (W 0.00349754) | 0.517449 ± 0.00262396 (W 0.00343396) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 21.0166 ± 0.0341557 (W 0.0234641) | 20.9698 ± 0.025811 (W 0.400197) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 21.6419 ± 0.0463503 (W 0.0641947) | 21.4363 ± 0.0541595 (W 0.0278276) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.36754 ± 0.00390341 (W 0.00620588) | 2.60474 ± 0.000921587 (W 0.00664804) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.72064 ± 0.00681821 (W 0.0095836) | 2.94606 ± 0.00899436 (W 0.00545292) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 118.502 ± 0.0823276 (W 0.145867) | 120.579 ± 0.197502 (W 0.171647) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 126.653 ± 0.0592962 (W 0.640091) | 129.061 ± 0.495586 (W 0.272281) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.21129 ± 0.00282071 (W 0.00246752) | 1.44503 ± 0.0014715 (W 0.00266578) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 47.9838 ± 0.0387615 (W 0.0322166) | 50.0815 ± 0.0472558 (W 0.051533) | - | · | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.10428 ± 0.00528269 (W 0.00783777) | 3.37235 ± 0.0114193 (W 0.0138939) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.11825 ± 0.0102506 (W 0.00983343) | 3.37227 ± 0.00483194 (W 0.0131334) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 136.033 ± 0.167227 (W 0.256687) | 140.753 ± 0.123099 (W 0.196561) | - | · | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 136.54 ± 0.135407 (W 0.365532) | 141.132 ± 0.189285 (W 0.306139) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.49555 ± 0.00236362 (W 0.00311705) | 1.9373 ± 0.0014025 (W 0.00400268) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.4938 ± 0.00131095 (W 0.00285671) | 1.93772 ± 0.00147133 (W 0.00312722) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 55.1784 ± 0.135503 (W 0.0859693) | 61.8622 ± 0.119708 (W 0.131721) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 55.224 ± 0.0570613 (W 0.0657056) | 61.7817 ± 0.0354508 (W 0.0773964) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.601831 ± 0.00222395 (W 0.00399513) | 0.936115 ± 0.00072868 (W 0.00221181) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.627102 ± 0.00293703 (W 0.00329771) | 0.922174 ± 0.00159023 (W 0.00395504) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 18.4409 ± 0.105723 (W 0.0190562) | 18.2654 ± 0.0899523 (W 0.0242932) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 18.1931 ± 0.072265 (W 0.0673015) | 17.6614 ± 0.0304307 (W 0.0584058) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.344522 ± 0.00156481 (W 0.00155453) | 0.33772 ± 0.000867068 (W 0.00252246) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.345525 ± 0.000640391 (W 0.00178493) | 0.338766 ± 0.000760276 (W 0.00214547) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 19.321 ± 0.024569 (W 0.0148458) | 19.3302 ± 0.0181379 (W 0.0182793) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 19.3413 ± 0.0275753 (W 0.018538) | 19.3319 ± 0.0293158 (W 0.0217683) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.366182 ± 0.00151424 (W 0.00217765) | 0.382703 ± 0.00206511 (W 0.00377012) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.394327 ± 0.00113684 (W 0.00271975) | 0.408001 ± 0.00457473 (W 0.0025255) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.76113 ± 0.010839 (W 0.0186019) | 8.94764 ± 0.0134821 (W 0.0151244) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 9.28877 ± 0.0166192 (W 0.0178686) | 9.4227 ± 0.00846324 (W 0.0125438) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.09877 ± 0.0057628 (W 0.003317) | 1.56607 ± 0.00168575 (W 0.00629731) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.39148 ± 0.00820122 (W 0.00408657) | 1.92874 ± 0.1398 (W 0.0432996) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 39.436 ± 0.0133636 (W 0.0599841) | 45.8584 ± 0.0352778 (W 0.0602174) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 46.1213 ± 0.10123 (W 0.064337) | 50.0478 ± 3.03715 (W 1.7587) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.765066 ± 0.000964896 (W 0.0020843) | 1.22829 ± 0.0060269 (W 0.0043692) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 19.9409 ± 0.0170263 (W 0.0151147) | 26.4049 ± 0.0457926 (W 0.0286077) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.69916 ± 0.0075462 (W 0.00338916) | 2.30904 ± 0.00490602 (W 0.00471831) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.713 ± 0.00760356 (W 0.00640944) | 2.32449 ± 0.00272953 (W 0.00381586) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 48.9254 ± 0.068223 (W 0.0820983) | 60.1683 ± 0.0531203 (W 0.0655616) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 49.2465 ± 0.0847747 (W 0.0877792) | 60.4057 ± 0.045667 (W 0.0956291) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.06546 ± 0.00564692 (W 0.00363491) | 1.73156 ± 0.00495135 (W 0.00296551) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.07177 ± 0.000862217 (W 0.00337069) | 1.74458 ± 0.0116942 (W 0.00663254) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 25.0269 ± 0.0944313 (W 0.0340047) | 36.5794 ± 0.0791599 (W 0.0551469) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 25.0392 ± 0.060015 (W 0.0315702) | 36.6281 ± 0.028893 (W 0.0908116) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.564443 ± 0.000600568 (W 0.00303255) | 0.934113 ± 0.00681192 (W 0.00394225) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.593548 ± 0.00103729 (W 0.00280219) | 0.914305 ± 0.00258363 (W 0.0019391) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 12.1227 ± 0.0270194 (W 0.0180276) | 12.111 ± 0.01854 (W 0.0262776) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 12.1793 ± 0.0573423 (W 0.0144255) | 12.0943 ± 0.0473347 (W 0.0478178) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.200609 ± 0.00148844 (W 0.00199683) | 0.298087 ± 0.0153909 (W 0.0167982) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.200364 ± 0.00211477 (W 0.00196292) | 0.322933 ± 0.0166376 (W 0.0146863) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.6248 ± 0.00824212 (W 0.0452469) | 13.4493 ± 1.54634 (W 1.55694) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.6346 ± 0.0546721 (W 0.0782619) | 13.5672 ± 0.0654847 (W 1.20385) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.350915 ± 0.00308892 (W 0.00196061) | 0.371483 ± 0.00217221 (W 0.00232869) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.379809 ± 0.00147869 (W 0.00111847) | 0.398834 ± 0.00258662 (W 0.00265423) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.86193 ± 0.00727168 (W 0.0105672) | 7.10785 ± 0.0324979 (W 0.029998) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.42757 ± 0.00814128 (W 0.0207408) | 7.64118 ± 0.0400633 (W 0.0161188) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.907755 ± 0.00216682 (W 0.00283328) | 1.64394 ± 0.0385552 (W 0.0139147) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.21957 ± 0.0053827 (W 0.00532596) | 1.68688 ± 0.0779085 (W 0.0248759) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 26.9019 ± 0.0842401 (W 0.0552097) | 41.1916 ± 0.518933 (W 0.674749) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 33.9072 ± 0.0879697 (W 0.0434688) | 43.0448 ± 1.11153 (W 2.2874) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.716958 ± 0.00242752 (W 0.00233059) | 1.32339 ± 0.00205966 (W 0.00565624) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 14.7713 ± 0.00317583 (W 0.0159064) | 26.2118 ± 0.245836 (W 0.294113) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.51269 ± 0.00276481 (W 0.00447234) | 2.40743 ± 0.00312706 (W 0.0130174) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.51699 ± 0.00289461 (W 0.00355979) | 2.42258 ± 0.00742896 (W 0.00954863) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 35.7335 ± 0.0703529 (W 0.0474067) | 54.7335 ± 0.378558 (W 0.656112) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 35.9704 ± 0.0662347 (W 0.281306) | 57.9599 ± 1.7915 (W 3.27193) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 1.02006 ± 0.00124746 (W 0.00186054) | 1.82614 ± 0.0033334 (W 0.00777292) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01975 ± 0.00127763 (W 0.00305138) | 1.83216 ± 0.00518193 (W 0.00413678) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.9188 ± 0.0102657 (W 0.0169238) | 36.7399 ± 0.190491 (W 0.22332) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.9521 ± 0.0814295 (W 0.0398098) | 36.5558 ± 0.256422 (W 0.224498) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `lu/1024x64`, `lu/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `solve/1024x64`, `solve/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 52.0815 ± 0.363093 (W 0.210858) | 52.0728 ± 0.826409 (W 0.231664) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 77.7243 ± 0.964898 (W 0.7913) | 77.1075 ± 0.273925 (W 0.249277) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.793822 ± 0.00525253 (W 0.00484876) | 0.781229 ± 0.00476949 (W 0.00214208) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.26077 ± 0.00740146 (W 0.00468353) | 1.25115 ± 0.00497999 (W 0.0069243) | - | · | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.6579 ± 0.0118406 (W 0.0169685) | 109.146 ± 1.12746 (W 0.197599) | - | 🔵 faer-fork 9.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 15.8233 ± 0.0202928 (W 0.0503456) | 18.0117 ± 0.0172692 (W 0.0289167) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.278295 ± 0.00150593 (W 0.00240855) | 3.19214 ± 0.0133194 (W 0.00815504) | - | 🔵 faer-fork 11.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.408773 ± 0.00789175 (W 0.00266661) | 0.515833 ± 0.00332723 (W 0.00356893) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 28.9859 ± 0.0682541 (W 0.0404645) | 28.8498 ± 0.0802836 (W 0.0471995) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 30.3754 ± 0.18184 (W 0.0427471) | 32.6028 ± 0.279269 (W 0.07368) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.02819 ± 0.00651463 (W 0.00636332) | 1.04227 ± 0.00391351 (W 0.00457949) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.743162 ± 0.00213242 (W 0.0027715) | 0.837753 ± 0.00326533 (W 0.00352589) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.7682 ± 0.376588 (W 0.0620186) | 27.3724 ± 0.40503 (W 0.296587) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.989074 ± 0.00795209 (W 0.00519379) | 1.00414 ± 0.00199132 (W 0.0035263) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 77.8167 ± 0.105753 (W 0.092464) | 83.2936 ± 0.192745 (W 0.0678977) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 78.2858 ± 0.093433 (W 0.399142) | 83.5392 ± 0.192653 (W 0.176297) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.99057 ± 0.00434303 (W 0.00255746) | 2.45455 ± 0.00752636 (W 0.00424232) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.99671 ± 0.00676094 (W 0.00353655) | 2.46425 ± 0.00885461 (W 0.00303984) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 31.7525 ± 0.202218 (W 0.0440185) | 39.2173 ± 0.082927 (W 0.0577079) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.783 ± 0.0444742 (W 0.113151) | 39.2583 ± 0.0870854 (W 0.0443585) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.942799 ± 0.00227473 (W 0.00228615) | 1.41674 ± 0.00207818 (W 0.00395717) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.942498 ± 0.00227831 (W 0.00361798) | 1.42265 ± 0.00412405 (W 0.0032044) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 15.558 ± 0.0182514 (W 0.0324288) | 16.2231 ± 0.069529 (W 0.0378635) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 33.7678 ± 0.224281 (W 0.198829) | 36.4296 ± 0.245054 (W 0.100697) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.461677 ± 0.00103513 (W 0.00315249) | 0.775217 ± 0.00264355 (W 0.00196096) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.755267 ± 0.00161055 (W 0.00417131) | 1.25198 ± 0.00759868 (W 0.00470632) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.52809 ± 0.0151343 (W 0.0217917) | 33.4802 ± 12.4419 (W 10.1682) | - | 🔵 faer-fork 9.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 6.40614 ± 0.0238704 (W 0.0769075) | 10.074 ± 0.520238 (W 0.309388) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.117421 ± 0.00059184 (W 0.00232182) | 1.65446 ± 0.0259062 (W 0.0381451) | - | 🔵 faer-fork 14.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.207283 ± 0.0002147 (W 0.00245444) | 0.401079 ± 0.0117079 (W 0.0244577) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 10.4857 ± 0.124673 (W 0.0377336) | 11.0121 ± 0.0574063 (W 0.0549763) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 10.1807 ± 0.0761086 (W 0.0513047) | 12.702 ± 0.118754 (W 0.037355) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.630324 ± 0.00362814 (W 0.00437924) | 0.709442 ± 0.00127993 (W 0.00438699) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.363067 ± 0.00176609 (W 0.00185827) | 0.525139 ± 0.00160803 (W 0.002916) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.1341 ± 0.0624621 (W 0.0305156) | 10.545 ± 0.0463927 (W 0.0260028) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.614253 ± 0.00259643 (W 0.00505223) | 0.687692 ± 0.000450915 (W 0.00367093) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 31.5586 ± 0.0488196 (W 0.0689257) | 42.0365 ± 0.133704 (W 0.0890346) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 31.7091 ± 0.0553959 (W 0.0651366) | 42.0125 ± 0.127572 (W 0.110118) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.29837 ± 0.00318203 (W 0.00253476) | 2.0841 ± 0.0118616 (W 0.00353297) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.3086 ± 0.00372648 (W 0.00295364) | 2.10352 ± 0.0162375 (W 0.00780475) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 16.2811 ± 0.0179954 (W 0.0253822) | 26.9393 ± 0.0983127 (W 0.0422395) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 16.2929 ± 0.11774 (W 0.0293948) | 26.8597 ± 0.0559511 (W 0.0373836) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.789591 ± 0.00201411 (W 0.0026717) | 1.46491 ± 0.00224599 (W 0.00709973) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.794841 ± 0.00326383 (W 0.00203898) | 1.48191 ± 0.0108701 (W 0.00397457) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.1291 ± 0.0509304 (W 0.0265443) | 10.8969 ± 0.0513667 (W 0.0244381) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 27.6756 ± 0.115036 (W 0.137055) | 30.7169 ± 0.112066 (W 0.10792) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.438538 ± 0.00454069 (W 0.00365074) | 0.780743 ± 0.00091859 (W 0.00271257) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.703508 ± 0.00322136 (W 0.00219068) | 1.26263 ± 0.00903905 (W 0.00670648) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.14566 ± 0.0104758 (W 0.0048953) | 30.6547 ± 4.38209 (W 0.13803) | - | 🔵 faer-fork 14.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.80852 ± 0.00903706 (W 0.0315527) | 7.89254 ± 0.0732328 (W 0.15014) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.094307 ± 0.000573997 (W 0.0014143) | 0.94887 ± 0.0110562 (W 0.150802) | - | 🔵 faer-fork 10.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.176498 ± 0.000669783 (W 0.00148313) | 0.353085 ± 0.0134412 (W 0.0113963) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 7.64442 ± 0.0561509 (W 0.0250783) | 9.54413 ± 0.150147 (W 0.268601) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 7.26919 ± 0.137623 (W 0.0511682) | 10.8429 ± 1.32231 (W 1.67521) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.566872 ± 0.000790728 (W 0.0049932) | 0.763641 ± 0.00534829 (W 0.00755577) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.316908 ± 0.0128451 (W 0.00177654) | 0.57923 ± 0.00307777 (W 0.00601765) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 7.31574 ± 0.0424443 (W 0.00928025) | 8.84115 ± 0.170964 (W 0.205628) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.549009 ± 0.00305234 (W 0.00539979) | 0.731314 ± 0.00520179 (W 0.00503649) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 25.132 ± 0.0547625 (W 0.0361734) | 41.1121 ± 0.148396 (W 0.226691) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 25.3637 ± 0.0293568 (W 0.0734643) | 41.5757 ± 1.16074 (W 3.25183) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.21295 ± 0.0243298 (W 0.00368746) | 2.18517 ± 0.00380349 (W 0.00628603) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.22723 ± 0.0197949 (W 0.00570581) | 2.19674 ± 0.00725549 (W 0.00671295) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 13.8567 ± 0.0372395 (W 0.0276152) | 27.8195 ± 0.096931 (W 0.070481) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 13.8721 ± 0.0282894 (W 0.0409074) | 27.9079 ± 0.0105438 (W 0.12165) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.770634 ± 0.00584537 (W 0.00254266) | 1.54526 ± 0.00677088 (W 0.00549517) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.774537 ± 0.00232004 (W 0.0017694) | 1.55192 ± 0.000413653 (W 0.00662645) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 88.6802 ± 0.304196 (W 0.307368) | 87.1431 ± 1.50711 (W 0.529224) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 76.9831 ± 0.995609 (W 0.480139) | 72.1618 ± 0.438884 (W 0.131349) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.21233 ± 0.00477337 (W 0.00519838) | 1.11555 ± 0.00431263 (W 0.00465984) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.17013 ± 0.00825835 (W 0.00335333) | 1.05107 ± 0.00316203 (W 0.0032455) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 12.6801 ± 0.0339448 (W 0.0774221) | 14.6341 ± 0.0220264 (W 0.016637) | 20.6948 ± 0.0778394 (W 0.0319113) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 13.1586 ± 0.0076154 (W 0.0277114) | 13.3545 ± 0.032128 (W 0.019522) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.308847 ± 0.00101553 (W 0.00292438) | 0.301112 ± 0.00242819 (W 0.00201923) | 0.431755 ± 0.00108289 (W 0.00264967) | · | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.320839 ± 0.00143182 (W 0.00243772) | 0.312343 ± 0.00572673 (W 0.00271208) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.6645 ± 0.127034 (W 0.0468795) | 26.624 ± 0.62444 (W 0.0731096) | 57.69 ± 0.134745 (W 0.513696) | · | 🔵 faer-fork 2.16× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.6261 ± 0.115646 (W 0.0789545) | 28.0719 ± 0.166765 (W 0.117373) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.601505 ± 0.00224367 (W 0.00285258) | 0.612002 ± 0.00723811 (W 0.00305943) | 1.25964 ± 0.00521003 (W 0.00252307) | · | 🔵 faer-fork 2.09× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.626552 ± 0.000511285 (W 0.00309489) | 0.632269 ± 0.00187971 (W 0.00294902) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 25.2215 ± 0.155028 (W 0.196253) | 25.1682 ± 0.443483 (W 0.476121) | 54.8007 ± 0.73867 (W 1.01966) | · | 🔵 faer-fork 2.17× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.565647 ± 0.00309918 (W 0.00345179) | 0.570014 ± 0.0020185 (W 0.00228859) | 1.18983 ± 0.0142154 (W 0.00277071) | · | 🔵 faer-fork 2.10× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 83.1712 ± 0.0624936 (W 0.137178) | 88.7001 ± 0.320191 (W 0.129696) | 245.64 ± 1.08224 (W 3.00961) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 2.95× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 83.5476 ± 0.0912946 (W 0.189132) | 88.9839 ± 0.165785 (W 0.283873) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.03008 ± 0.00112143 (W 0.00399391) | 2.52444 ± 0.00191757 (W 0.00806981) | 3.88111 ± 0.0684556 (W 0.0108307) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.91× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.04284 ± 0.00824617 (W 0.00232425) | 2.5344 ± 0.00234626 (W 0.00347722) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 37.6221 ± 0.0697974 (W 0.0871169) | 45.1362 ± 0.0511448 (W 0.0688314) | 191.582 ± 2.68806 (W 5.85478) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 5.09× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 37.5546 ± 0.0593491 (W 0.0416479) | 45.0222 ± 0.0736698 (W 0.0786396) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.01604 ± 0.00443296 (W 0.00272392) | 1.4794 ± 0.00147359 (W 0.00544918) | 2.4791 ± 0.00543486 (W 0.00603221) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔵 faer-fork 2.44× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.02225 ± 0.00355533 (W 0.00340807) | 1.48979 ± 0.00672195 (W 0.00526661) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 33.1488 ± 0.138459 (W 0.0755119) | 33.9678 ± 0.0545065 (W 0.0951542) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 31.1387 ± 0.238973 (W 0.237573) | 31.5139 ± 0.151121 (W 0.0643731) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.653719 ± 0.00237181 (W 0.00443187) | 1.109 ± 0.00559398 (W 0.00490933) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.678765 ± 0.00336254 (W 0.00266817) | 1.05288 ± 0.00484335 (W 0.00496002) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.41821 ± 0.00716228 (W 0.010697) | 5.58913 ± 0.354179 (W 0.545784) | 4.65536 ± 0.00830249 (W 0.012616) | 🔵 faer-fork 1.64× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 3.84725 ± 0.00960675 (W 0.0117448) | 5.44551 ± 0.238947 (W 0.282308) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.096699 ± 0.000173208 (W 0.000408531) | 0.136966 ± 0.00137399 (W 0.0225782) | 0.199462 ± 0.00480666 (W 0.00526566) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔵 faer-fork 2.06× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.116207 ± 0.00088193 (W 0.000384536) | 0.158317 ± 0.0133282 (W 0.0266611) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 7.03359 ± 0.0313074 (W 0.0118846) | 7.47424 ± 0.0513747 (W 0.0278206) | 12.1665 ± 0.0544783 (W 0.026345) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.73× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 7.57407 ± 0.0109633 (W 0.0224312) | 8.11503 ± 0.0666781 (W 0.0353229) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.22149 ± 0.00288118 (W 0.00170353) | 0.288642 ± 0.0016916 (W 0.00301503) | 0.387967 ± 0.00258263 (W 0.00307907) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔵 faer-fork 1.75× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.250748 ± 0.00109892 (W 0.00197429) | 0.32199 ± 0.0019101 (W 0.00285314) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.58659 ± 0.0370174 (W 0.0120228) | 7.07229 ± 0.0341667 (W 0.0333881) | 11.141 ± 0.0260548 (W 0.0289969) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.199262 ± 0.00077644 (W 0.00170658) | 0.264839 ± 0.000355781 (W 0.00247636) | 0.352832 ± 0.00338448 (W 0.00252797) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔵 faer-fork 1.77× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 36.819 ± 0.0804092 (W 0.0836838) | 47.3413 ± 0.0862305 (W 0.0897842) | 123.317 ± 0.232938 (W 0.37241) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 3.35× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 37.0711 ± 0.0363103 (W 0.0630942) | 47.5841 ± 0.0402366 (W 0.117802) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.34 ± 0.0101936 (W 0.00748353) | 2.14733 ± 0.00479716 (W 0.00529043) | 3.0675 ± 0.0280905 (W 0.0164043) | 🔵 faer-fork 1.60× faster than `faer-upstream` | 🔵 faer-fork 2.29× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.35053 ± 0.00843425 (W 0.00756399) | 2.17652 ± 0.00235628 (W 0.00677722) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.8871 ± 0.0336811 (W 0.0245947) | 32.7579 ± 0.0797033 (W 0.0908421) | 93.7616 ± 0.308726 (W 0.442463) | 🔵 faer-fork 1.50× faster than `faer-upstream` | 🔵 faer-fork 4.28× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.9355 ± 0.194419 (W 0.0539229) | 32.8478 ± 0.0318321 (W 0.0724462) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.855803 ± 0.00496572 (W 0.00335527) | 1.52203 ± 0.00914046 (W 0.00757631) | 2.09106 ± 0.0167985 (W 0.00580148) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 2.44× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.873891 ± 0.00394622 (W 0.0042516) | 1.53363 ± 0.0154379 (W 0.00753053) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 26.0446 ± 0.0993936 (W 0.0837385) | 27.2285 ± 0.215675 (W 0.0797534) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 25.364 ± 0.0828955 (W 0.0478452) | 26.2442 ± 0.977947 (W 0.394845) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.59746 ± 0.00160419 (W 0.00348927) | 1.10899 ± 0.00245057 (W 0.0053563) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.629444 ± 0.00400997 (W 0.00333666) | 1.05767 ± 0.0107052 (W 0.00805852) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.88228 ± 0.00974628 (W 0.0128018) | 3.26155 ± 0.0600672 (W 0.139834) | 2.44975 ± 0.0176184 (W 0.00773031) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 2.3416 ± 0.00063635 (W 0.00673443) | 3.3571 ± 0.0554015 (W 0.113786) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.062758 ± 0.000721647 (W 0.00030216) | 0.111178 ± 0.000733284 (W 0.00824547) | 0.100608 ± 0.000417854 (W 0.0017376) | 🔵 faer-fork 1.77× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.088635 ± 0.0174038 (W 0.000375414) | 0.137457 ± 0.001169 (W 0.00660356) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 3.87067 ± 0.0273817 (W 0.0122786) | 6.09485 ± 0.0808182 (W 0.257148) | 6.08865 ± 0.0136939 (W 0.0135679) | 🔵 faer-fork 1.57× faster than `faer-upstream` | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 4.3364 ± 0.0617967 (W 0.00982293) | 6.54463 ± 0.141747 (W 0.232008) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.166573 ± 0.000577064 (W 0.00174761) | 0.350256 ± 0.00155238 (W 0.00392825) | 0.231299 ± 0.000374588 (W 0.0032902) | 🔵 faer-fork 2.10× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.200296 ± 0.000972178 (W 0.00161049) | 0.386174 ± 0.00403331 (W 0.00275533) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.58916 ± 0.0157665 (W 0.0112111) | 5.31051 ± 0.128946 (W 0.0929825) | 5.63439 ± 0.0182103 (W 0.0127277) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.149457 ± 0.000847531 (W 0.000822488) | 0.322969 ± 0.00408283 (W 0.00424394) | 0.199953 ± 0.00137609 (W 0.00168391) | 🔵 faer-fork 2.16× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 30.6862 ± 0.0825742 (W 0.116018) | 46.3919 ± 0.146875 (W 0.194971) | 105.544 ± 0.438446 (W 0.301541) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔵 faer-fork 3.44× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 30.8686 ± 0.0420802 (W 0.209877) | 46.4681 ± 0.168676 (W 0.222677) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.25622 ± 0.000554328 (W 0.0068952) | 2.2461 ± 0.00391908 (W 0.00930387) | 2.97255 ± 0.0322194 (W 0.0170683) | 🔵 faer-fork 1.79× faster than `faer-upstream` | 🔵 faer-fork 2.37× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.27299 ± 0.00638048 (W 0.00433611) | 2.26102 ± 0.00440086 (W 0.00720776) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 19.5172 ± 0.0273306 (W 0.0389459) | 33.5218 ± 0.0752712 (W 0.356713) | 84.4876 ± 1.20146 (W 0.521481) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔵 faer-fork 4.33× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.5639 ± 1.48151 (W 0.0411842) | 33.6684 ± 0.0103413 (W 0.0686569) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.839774 ± 0.0081888 (W 0.00692103) | 1.60605 ± 0.00583665 (W 0.00727476) | 2.07638 ± 0.00202303 (W 0.00440261) | 🔵 faer-fork 1.91× faster than `faer-upstream` | 🔵 faer-fork 2.47× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.855646 ± 0.0115254 (W 0.00574014) | 1.6148 ± 0.0147855 (W 0.00405941) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 150.363 ± 0.505797 (W 0.665951) | 150.751 ± 0.605927 (W 1.31282) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 236.135 ± 0.971198 (W 0.492221) | 228.895 ± 1.08952 (W 0.674833) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.67097 ± 0.00106428 (W 0.00386529) | 1.66078 ± 0.00964938 (W 0.00377734) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.62841 ± 0.0133259 (W 0.0212339) | 2.50513 ± 0.0591134 (W 0.00450668) | - | · | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 40.7845 ± 0.0754813 (W 0.0440691) | 138.983 ± 1.58972 (W 0.193463) | - | 🔵 faer-fork 3.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 47.4746 ± 0.312428 (W 0.101648) | 46.8498 ± 0.117059 (W 0.101941) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.872625 ± 0.00143973 (W 0.00211055) | 3.80938 ± 0.0186611 (W 0.00685858) | - | 🔵 faer-fork 4.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.999754 ± 0.00107649 (W 0.00671632) | 1.04559 ± 0.00384894 (W 0.00228022) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 102.845 ± 0.153156 (W 0.0551012) | 102.998 ± 0.150992 (W 0.154772) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 107.799 ± 0.190182 (W 0.441495) | 108.312 ± 0.256763 (W 0.311038) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.73882 ± 0.0185805 (W 0.00781572) | 2.75951 ± 0.00345605 (W 0.0105983) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 2.51503 ± 0.00519257 (W 0.00556763) | 2.59858 ± 0.00626599 (W 0.00592473) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 96.8865 ± 0.167609 (W 0.0971697) | 97.4958 ± 0.311622 (W 0.329737) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 2.59708 ± 0.0040779 (W 0.00942748) | 2.60866 ± 0.00993912 (W 0.00682258) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 261.95 ± 0.281863 (W 0.630511) | 269.104 ± 0.897448 (W 0.355954) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 262.294 ± 0.211676 (W 6.71785) | 269.657 ± 0.949709 (W 6.38098) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.37202 ± 0.0363585 (W 0.0101991) | 5.9013 ± 0.0091506 (W 0.0101143) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.41774 ± 0.00426103 (W 0.00515125) | 5.9257 ± 0.00750692 (W 0.00982442) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 101.201 ± 0.152671 (W 0.0725303) | 110.212 ± 0.211803 (W 0.124499) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 101.098 ± 0.0179284 (W 0.221495) | 110.531 ± 0.19776 (W 0.248842) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.38386 ± 0.00211406 (W 0.006424) | 2.94155 ± 0.00346339 (W 0.00613706) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.41822 ± 0.00474902 (W 0.0134279) | 2.94447 ± 0.0134827 (W 0.00367057) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 47.0837 ± 0.151512 (W 0.0921429) | 49.4187 ± 0.278385 (W 0.247764) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 87.3007 ± 0.63172 (W 0.618627) | 86.615 ± 0.409608 (W 0.38536) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.731665 ± 0.00261491 (W 0.00228378) | 1.65565 ± 0.00202016 (W 0.00361307) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.3845 ± 0.00712872 (W 0.0064359) | 2.52297 ± 0.0080908 (W 0.00808631) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.4398 ± 0.0346128 (W 0.0345598) | 65.9494 ± 13.1455 (W 13.3851) | - | 🔵 faer-fork 5.76× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 16.3922 ± 0.589447 (W 0.479312) | 20.7675 ± 0.413578 (W 1.01038) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.294104 ± 0.000448269 (W 0.00240089) | 1.91275 ± 0.0698591 (W 0.34593) | - | 🔵 faer-fork 6.50× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.409538 ± 0.0447851 (W 0.0371535) | 0.587687 ± 0.0458453 (W 0.0539106) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 29.6087 ± 0.0454151 (W 0.209297) | 30.2663 ± 1.12645 (W 0.250596) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 31.636 ± 0.12349 (W 0.0679676) | 33.1715 ± 0.22428 (W 0.114826) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.0465 ± 0.0016624 (W 0.0048829) | 1.15539 ± 0.00293325 (W 0.00551218) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.88043 ± 0.00293652 (W 0.000837904) | 1.0363 ± 0.0072586 (W 0.00407648) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 28.0231 ± 0.0570281 (W 0.104195) | 28.7371 ± 0.0657749 (W 0.0833866) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.981772 ± 0.00592631 (W 0.00705455) | 1.06667 ± 0.00254475 (W 0.00400387) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 86.979 ± 0.187298 (W 0.0636455) | 106.154 ± 0.134538 (W 0.354512) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 87.8369 ± 0.159495 (W 0.147393) | 107.024 ± 0.0919616 (W 2.1326) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.61006 ± 0.00609589 (W 0.0074073) | 3.57153 ± 0.00781913 (W 0.0102531) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.63075 ± 0.0159973 (W 0.00423897) | 3.58454 ± 0.00552632 (W 0.00586277) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 40.3149 ± 0.0818563 (W 0.042722) | 59.6653 ± 0.0518594 (W 0.0439323) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 40.2934 ± 0.126347 (W 0.0660005) | 59.6393 ± 0.0819203 (W 0.0677936) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.55132 ± 0.00488241 (W 0.00466611) | 2.32175 ± 0.00451588 (W 0.00489507) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.56529 ± 0.00493603 (W 0.00497044) | 2.33887 ± 0.00278401 (W 0.00701584) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 23.1389 ± 0.27352 (W 0.15944) | 26.0623 ± 0.178409 (W 0.0573331) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 55.5919 ± 0.072995 (W 0.222385) | 59.9735 ± 0.568051 (W 0.223814) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.60358 ± 0.0051069 (W 0.00632806) | 1.6638 ± 0.00513278 (W 0.00506726) | - | 🔵 faer-fork 2.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.24528 ± 0.00522167 (W 0.00323571) | 2.52632 ± 0.00828111 (W 0.0100897) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.37475 ± 0.0238121 (W 0.0176507) | 36.1148 ± 0.617466 (W 5.50644) | - | 🔵 faer-fork 5.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 11.2592 ± 0.0238248 (W 0.0829862) | 13.912 ± 0.243048 (W 0.240496) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.19947 ± 0.00121593 (W 0.00201975) | 1.06851 ± 0.00548604 (W 0.146922) | - | 🔵 faer-fork 5.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.360252 ± 0.000322308 (W 0.00226681) | 0.445326 ± 0.00529716 (W 0.00901966) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 17.4208 ± 0.126012 (W 0.0902138) | 22.4282 ± 0.0744073 (W 0.899303) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 19.1096 ± 0.121054 (W 0.120138) | 25.286 ± 2.55364 (W 4.19269) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.76658 ± 0.00401582 (W 0.00718515) | 1.13767 ± 0.0133284 (W 0.0147161) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.616936 ± 0.000900584 (W 0.00309577) | 1.02299 ± 0.0128259 (W 0.0163611) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.2991 ± 0.0340219 (W 0.0393107) | 20.7785 ± 0.752509 (W 0.956107) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.726876 ± 0.0029979 (W 0.00345699) | 1.03268 ± 0.00578208 (W 0.0246086) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 61.8521 ± 0.0624317 (W 0.166259) | 94.0381 ± 0.457661 (W 1.77858) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 62.5318 ± 0.137334 (W 0.391332) | 95.6712 ± 0.721083 (W 2.63633) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.22147 ± 0.0106327 (W 0.00401487) | 3.59637 ± 0.0147724 (W 0.013695) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.26144 ± 0.00961713 (W 0.00480443) | 3.653 ± 0.135361 (W 0.301989) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 31.3872 ± 0.077785 (W 0.0472318) | 57.5339 ± 0.396156 (W 3.82645) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.4958 ± 0.183895 (W 0.0437426) | 57.6437 ± 0.169027 (W 0.512458) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.46979 ± 0.00568847 (W 0.0045388) | 2.3899 ± 0.00222463 (W 0.00889673) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.47891 ± 0.00331206 (W 0.00597129) | 2.38943 ± 0.0159815 (W 0.0117828) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 255.562 ± 3.71215 (W 0.897784) | 253.625 ± 8.70887 (W 1.24069) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 232.378 ± 1.25009 (W 0.55731) | 224.436 ± 1.04053 (W 0.660977) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 2.62242 ± 0.028648 (W 0.00941632) | 2.36916 ± 0.0411941 (W 0.0111968) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.51671 ± 0.0290628 (W 0.0105319) | 2.31919 ± 0.00791012 (W 0.00623216) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 42.2372 ± 0.065849 (W 0.059492) | 43.1056 ± 0.0236379 (W 0.0492956) | 71.7214 ± 0.0302151 (W 0.0670904) | · | 🔵 faer-fork 1.70× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 43.4406 ± 0.0674745 (W 0.386485) | 42.2885 ± 0.0829 (W 0.0472099) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.871195 ± 0.00797908 (W 0.00499066) | 0.861585 ± 0.00282991 (W 0.00421676) | 2.04936 ± 0.00778345 (W 0.00334434) | · | 🔵 faer-fork 2.35× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.896119 ± 0.00268424 (W 0.00329172) | 0.88992 ± 0.00193615 (W 0.00482712) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 101.95 ± 0.34028 (W 0.38589) | 101.879 ± 0.382129 (W 0.558102) | 168.347 ± 2.01317 (W 1.68297) | · | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 103.705 ± 0.322717 (W 0.419056) | 103.509 ± 0.0737515 (W 0.295446) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.37324 ± 0.0073763 (W 0.00406315) | 2.37623 ± 0.00459841 (W 0.00303684) | 3.59737 ± 0.00679762 (W 0.00634723) | · | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 2.4015 ± 0.030266 (W 0.00559803) | 2.41281 ± 0.00760114 (W 0.00638355) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 95.9901 ± 0.258069 (W 0.495876) | 95.9246 ± 0.894794 (W 0.268635) | 153.669 ± 0.690432 (W 0.447827) | · | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 2.24516 ± 0.00936657 (W 0.00529808) | 2.22218 ± 0.00164024 (W 0.00620467) | 3.32926 ± 0.0136755 (W 0.00669823) | · | 🔵 faer-fork 1.48× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 273.056 ± 0.574728 (W 0.389727) | 280.54 ± 0.30312 (W 0.35737) | 541.741 ± 0.815864 (W 1.89596) | · | 🔵 faer-fork 1.98× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 273.866 ± 0.391577 (W 6.1348) | 281.687 ± 0.68956 (W 5.98898) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.46547 ± 0.0299881 (W 0.0120516) | 6.02105 ± 0.0132274 (W 0.00648405) | 11.9658 ± 0.0353064 (W 0.0353815) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 2.19× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.53015 ± 0.0128367 (W 0.0163067) | 6.03463 ± 0.0446762 (W 0.0132982) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 114.96 ± 1.063 (W 0.402373) | 122.399 ± 0.482994 (W 0.4943) | 198.264 ± 1.85702 (W 0.887212) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 114.155 ± 0.708319 (W 0.547811) | 123.916 ± 0.903752 (W 0.429438) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.5166 ± 0.008024 (W 0.00535536) | 3.0782 ± 0.000252651 (W 0.00477382) | 5.34677 ± 0.0775229 (W 0.0129372) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 2.12× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.54425 ± 0.0042571 (W 0.00817345) | 3.06812 ± 0.0022304 (W 0.00620188) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 91.1134 ± 0.471497 (W 0.418038) | 92.52 ± 0.485451 (W 0.234476) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 83.2843 ± 0.282022 (W 0.699556) | 82.4903 ± 0.0814445 (W 0.526292) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.24118 ± 0.00662457 (W 0.0052908) | 2.40209 ± 0.0237488 (W 0.0139089) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.2679 ± 0.00141904 (W 0.00458205) | 2.29935 ± 0.0103227 (W 0.0116642) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.3695 ± 0.0272004 (W 0.0167112) | 16.1259 ± 0.81209 (W 1.23709) | 19.5111 ± 0.0606884 (W 0.0302027) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 12.4894 ± 0.0922767 (W 0.484803) | 18.0294 ± 0.0993272 (W 1.10074) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.337605 ± 0.0458683 (W 0.0434254) | 0.287871 ± 0.00340691 (W 0.0435475) | 0.494679 ± 0.000709821 (W 0.00320843) | 🔴 faer-fork 1.17× slower than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.29841 ± 0.00182141 (W 0.0398796) | 0.40521 ± 0.0470165 (W 0.0391924) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 26.309 ± 0.101277 (W 0.035086) | 26.9329 ± 0.0628839 (W 0.0368412) | 37.6108 ± 0.0140937 (W 0.086889) | · | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.7985 ± 0.116587 (W 0.120032) | 28.2794 ± 0.0534198 (W 0.0802203) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.711457 ± 0.00109426 (W 0.0024943) | 0.808688 ± 0.00310682 (W 0.00267663) | 1.3063 ± 0.00305115 (W 0.00481593) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.84× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.772981 ± 0.0011078 (W 0.00298676) | 0.869289 ± 0.00102337 (W 0.00325282) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 24.7316 ± 0.0302184 (W 0.165824) | 25.1651 ± 0.087853 (W 0.0213044) | 34.7435 ± 0.0545006 (W 0.0365053) | · | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.655784 ± 0.000821293 (W 0.00282723) | 0.72416 ± 0.000602513 (W 0.00313671) | 1.19492 ± 0.00318885 (W 0.00543871) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 99.2772 ± 0.134737 (W 0.114584) | 118.147 ± 0.0880217 (W 0.146096) | 185.753 ± 1.64011 (W 0.433143) | 🔵 faer-fork 1.19× faster than `faer-upstream` | 🔵 faer-fork 1.87× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 100.072 ± 0.139472 (W 0.13196) | 119.191 ± 0.045771 (W 0.358302) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.70907 ± 0.018547 (W 0.00834059) | 3.68839 ± 0.00719636 (W 0.0107619) | 7.23458 ± 0.0174001 (W 0.0132552) | 🔵 faer-fork 1.36× faster than `faer-upstream` | 🔵 faer-fork 2.67× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.73639 ± 0.0401805 (W 0.0109559) | 3.71737 ± 0.00867953 (W 0.0110538) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 52.3498 ± 0.0924006 (W 0.0961498) | 71.6424 ± 0.111766 (W 0.0923038) | 80.7998 ± 0.260219 (W 0.247431) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.54× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 52.343 ± 0.130702 (W 0.105985) | 71.6667 ± 0.280195 (W 0.0694666) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.64531 ± 0.00314375 (W 0.00470055) | 2.45837 ± 0.0215937 (W 0.00547254) | 3.53723 ± 0.0108637 (W 0.0143235) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 2.15× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.67433 ± 0.0186744 (W 0.00606624) | 2.42897 ± 0.00339699 (W 0.00562956) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 53.7573 ± 0.437731 (W 0.191394) | 58.0387 ± 0.221161 (W 0.193801) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 51.2986 ± 0.190712 (W 0.179479) | 56.1799 ± 0.83909 (W 0.526827) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.03827 ± 0.00376516 (W 0.00452678) | 2.39008 ± 0.0109809 (W 0.0118347) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.09106 ± 0.00422841 (W 0.00330102) | 2.26562 ± 0.0590745 (W 0.0126646) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 6.6405 ± 0.0210565 (W 0.0203511) | 9.05972 ± 0.0603749 (W 0.288408) | 9.96803 ± 0.0162733 (W 0.0242026) | 🔵 faer-fork 1.36× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 7.55256 ± 0.00306966 (W 0.0176229) | 9.63521 ± 0.41121 (W 0.416633) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.194912 ± 0.000222663 (W 0.00165336) | 0.227956 ± 0.00076452 (W 0.020768) | 0.321072 ± 0.00169371 (W 0.00280255) | 🔵 faer-fork 1.17× faster than `faer-upstream` | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.246167 ± 0.000853837 (W 0.00166705) | 0.282293 ± 0.00239796 (W 0.0205939) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 13.9712 ± 0.0508408 (W 0.0350149) | 19.7191 ± 0.953749 (W 1.2468) | 19.3326 ± 0.0190094 (W 0.0289807) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 15.2874 ± 0.873261 (W 0.38776) | 20.9666 ± 2.70482 (W 3.39159) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.433241 ± 0.0040118 (W 0.00256331) | 0.786159 ± 0.0102009 (W 0.0170049) | 0.722578 ± 0.0013709 (W 0.00302268) | 🔵 faer-fork 1.81× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.504655 ± 0.00209709 (W 0.00197324) | 0.853013 ± 0.00367006 (W 0.0093109) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 12.9678 ± 0.0443781 (W 0.0248961) | 18.1195 ± 0.806508 (W 0.839493) | 17.9807 ± 0.219585 (W 0.0249329) | 🔵 faer-fork 1.40× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.396015 ± 0.000539811 (W 0.00310483) | 0.695167 ± 0.0101131 (W 0.0170659) | 0.640629 ± 0.00114549 (W 0.00242193) | 🔵 faer-fork 1.76× faster than `faer-upstream` | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 74.3741 ± 0.0851532 (W 0.148771) | 106.02 ± 0.354079 (W 1.45151) | 113.087 ± 1.65129 (W 0.354664) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 75.1482 ± 0.0960295 (W 0.433296) | 106.89 ± 0.3396 (W 1.62792) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.34171 ± 0.0188838 (W 0.00570089) | 3.7362 ± 0.0430174 (W 0.0319357) | 6.13908 ± 0.0530265 (W 0.137588) | 🔵 faer-fork 1.60× faster than `faer-upstream` | 🔵 faer-fork 2.62× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.36673 ± 0.0191489 (W 0.00521296) | 3.79139 ± 0.0100791 (W 0.379328) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 43.565 ± 0.186106 (W 0.200709) | 69.5992 ± 0.300797 (W 0.631837) | 54.5926 ± 0.151624 (W 0.160574) | 🔵 faer-fork 1.60× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 43.3878 ± 0.065934 (W 0.0287624) | 69.3578 ± 0.171479 (W 0.268651) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.56654 ± 0.0275645 (W 0.00298095) | 2.5183 ± 0.00152964 (W 0.00754144) | 3.22567 ± 0.0404794 (W 0.0131063) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔵 faer-fork 2.06× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.55822 ± 0.00781453 (W 0.00567236) | 2.48273 ± 0.014553 (W 0.00929661) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 24.6926 ± 0.0716005 (W 0.0227788) | 24.5278 ± 0.0802006 (W 0.0420013) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 38.7246 ± 0.0965304 (W 0.0693253) | 38.4481 ± 0.068812 (W 0.140276) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.477778 ± 0.0017442 (W 0.00287094) | 0.465776 ± 0.00244969 (W 0.00224449) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.00609 ± 0.0144897 (W 0.00855222) | 1.05031 ± 0.00566721 (W 0.00498192) | - | · | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.20871 ± 0.0194168 (W 0.0123788) | 65.0532 ± 0.687716 (W 0.12546) | - | 🔵 faer-fork 10.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.39205 ± 0.0198861 (W 0.0636875) | 11.1818 ± 0.0590231 (W 0.0277479) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.181465 ± 0.000265293 (W 0.00176613) | 3.08548 ± 0.00545098 (W 0.0148148) | - | 🔵 faer-fork 17.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.360108 ± 0.00130566 (W 0.00234637) | 0.445662 ± 0.00308023 (W 0.0025047) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 16.9881 ± 0.252286 (W 0.022258) | 17.1173 ± 0.238836 (W 0.0248509) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 17.5877 ± 0.175726 (W 0.0945139) | 19.8057 ± 0.0694065 (W 0.135143) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.840512 ± 0.00245603 (W 0.00517158) | 0.851348 ± 0.0032692 (W 0.00361943) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.605498 ± 0.000883966 (W 0.00459554) | 0.634822 ± 0.00729764 (W 0.00295267) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.405 ± 0.263437 (W 0.0308516) | 16.2672 ± 0.0777533 (W 0.0234522) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.811679 ± 0.00439829 (W 0.00415079) | 0.829191 ± 0.0017613 (W 0.0041039) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 42.1824 ± 0.33427 (W 0.0688729) | 46.0272 ± 0.1512 (W 0.0418489) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 42.3999 ± 0.297695 (W 0.0567814) | 46.2383 ± 0.0957461 (W 0.0783566) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.3643 ± 0.00230064 (W 0.00475187) | 1.56745 ± 0.02239 (W 0.00362228) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.36381 ± 0.00365674 (W 0.00346713) | 1.71915 ± 0.00666788 (W 0.00544928) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 18.7156 ± 0.0506193 (W 0.0256296) | 24.0792 ± 0.0606846 (W 0.0383611) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 18.6905 ± 0.0223301 (W 0.0308549) | 24.2634 ± 0.140916 (W 0.0399583) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.698774 ± 0.00385973 (W 0.00200021) | 1.09107 ± 0.0029976 (W 0.00319583) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.697432 ± 0.000762707 (W 0.00342302) | 1.09365 ± 0.00262051 (W 0.00356835) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 9.16375 ± 0.015897 (W 0.020664) | 9.38837 ± 0.00699051 (W 0.019781) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 19.3683 ± 0.0860231 (W 0.0613014) | 21.8979 ± 0.175472 (W 0.0336332) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.352232 ± 0.00268228 (W 0.00289777) | 0.463539 ± 0.0014952 (W 0.00242869) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.649341 ± 0.000915217 (W 0.00271029) | 1.05944 ± 0.0157459 (W 0.0059018) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.00188 ± 0.00192081 (W 0.00550332) | 20.5758 ± 7.75707 (W 7.60846) | - | 🔵 faer-fork 10.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.23427 ± 0.319023 (W 0.0596731) | 7.42842 ± 0.182191 (W 0.114024) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.08097 ± 0.000127163 (W 0.000372003) | 0.96578 ± 0.391264 (W 0.373722) | - | 🔵 faer-fork 11.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.194641 ± 0.000892689 (W 0.00148908) | 0.332776 ± 0.0231978 (W 0.0219773) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 6.8389 ± 0.0124189 (W 0.00529344) | 7.34719 ± 0.0146236 (W 0.0102186) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 6.57224 ± 0.0684804 (W 0.0438341) | 9.25289 ± 0.0181419 (W 0.0151797) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.591148 ± 0.00174621 (W 0.00443358) | 0.674829 ± 0.00286807 (W 0.00796274) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.358564 ± 0.00391629 (W 0.0046069) | 0.465853 ± 0.000783591 (W 0.00333558) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.64733 ± 0.0622318 (W 0.0157523) | 7.10809 ± 0.0690575 (W 0.0133461) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.572889 ± 0.000514441 (W 0.00696984) | 0.647547 ± 0.00221515 (W 0.00634788) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 18.7969 ± 0.016908 (W 0.0266566) | 25.6455 ± 0.0915023 (W 0.0757669) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 18.7896 ± 0.0296754 (W 0.0335799) | 25.7882 ± 0.0519868 (W 0.523672) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.01172 ± 0.00182994 (W 0.00127104) | 1.49038 ± 0.000782885 (W 0.00507191) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.01678 ± 0.00198478 (W 0.00120171) | 1.5044 ± 0.00268354 (W 0.0054726) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 11.153 ± 0.0312168 (W 0.0143946) | 18.0786 ± 0.0575757 (W 0.0291055) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 11.1403 ± 0.0515823 (W 0.0148595) | 18.0894 ± 0.0229282 (W 0.0638315) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.640752 ± 0.00171194 (W 0.00351281) | 1.20449 ± 0.000659371 (W 0.00448352) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.639836 ± 0.00129856 (W 0.00240697) | 1.21484 ± 0.0028501 (W 0.00530604) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 6.7454 ± 0.00469156 (W 0.0229688) | 7.10261 ± 0.00820374 (W 0.0152513) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 16.3049 ± 0.0897249 (W 0.110574) | 19.0809 ± 0.0470891 (W 0.0712474) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.357128 ± 0.00242773 (W 0.00335034) | 0.46447 ± 0.00318007 (W 0.0036833) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.663612 ± 0.00109845 (W 0.00768457) | 1.04758 ± 0.0275441 (W 0.00270436) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 1.306 ± 0.00911134 (W 0.00862679) | 19.2046 ± 0.792557 (W 0.123176) | - | 🔵 faer-fork 14.70× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 3.4646 ± 0.00733609 (W 0.0613149) | 6.34653 ± 0.0593891 (W 0.0615295) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.066554 ± 0.000523041 (W 0.000781283) | 0.937491 ± 0.00944711 (W 0.0106252) | - | 🔵 faer-fork 14.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.171088 ± 0.000671918 (W 0.00189423) | 0.315989 ± 0.0037715 (W 0.00411015) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 5.22875 ± 0.0508494 (W 0.0375807) | 6.97838 ± 0.0586326 (W 0.0756822) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.74711 ± 0.0597785 (W 0.0522823) | 8.91283 ± 0.0870901 (W 0.0740094) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.554424 ± 0.00471565 (W 0.00517305) | 0.749944 ± 0.0020335 (W 0.0102532) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.327839 ± 0.00328828 (W 0.00744894) | 0.548278 ± 0.00365505 (W 0.00437797) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 5.0581 ± 0.0272506 (W 0.0183144) | 6.51918 ± 0.041019 (W 0.109755) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.537988 ± 0.00137368 (W 0.0052005) | 0.706142 ± 0.00182247 (W 0.0060594) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 15.037 ± 0.0413089 (W 0.0507025) | 26.5439 ± 0.00529793 (W 0.12372) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 15.1209 ± 0.0368921 (W 0.0291246) | 26.7188 ± 0.13397 (W 0.119408) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.97029 ± 0.00132741 (W 0.00163437) | 1.61405 ± 0.00712361 (W 0.00750667) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.979073 ± 0.00265962 (W 0.00197082) | 1.64008 ± 0.0304286 (W 0.0344356) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 9.63632 ± 0.0484117 (W 0.0116085) | 19.3771 ± 0.066646 (W 0.067001) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 9.65979 ± 0.00987857 (W 0.0145316) | 19.3645 ± 0.0603048 (W 0.0369312) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.627645 ± 5.83809e-05 (W 0.00202972) | 1.30662 ± 0.0116736 (W 0.00525769) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.628797 ± 0.00413447 (W 0.00135622) | 1.31614 ± 0.00864217 (W 0.00536518) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 39.1789 ± 0.351619 (W 0.0759926) | 37.2912 ± 2.61648 (W 0.057622) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 36.5647 ± 0.0793838 (W 0.054813) | 34.1998 ± 0.174006 (W 0.04519) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.921473 ± 0.00330428 (W 0.0068656) | 0.842495 ± 0.00586271 (W 0.00551921) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.903666 ± 0.0092037 (W 0.00490263) | 0.833319 ± 0.00650798 (W 0.00864868) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 7.22427 ± 0.0307997 (W 0.0187961) | 6.91209 ± 0.010588 (W 0.0160021) | - | · | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 7.24294 ± 0.0184136 (W 0.0127306) | 6.92809 ± 0.00995537 (W 0.0234606) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.24684 ± 0.00163233 (W 0.00233374) | 0.233215 ± 0.000407564 (W 0.00175322) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.253778 ± 0.000489627 (W 0.00226798) | 0.240102 ± 0.0011155 (W 0.00204808) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 14.9958 ± 0.0789309 (W 0.0206082) | 14.9098 ± 0.106046 (W 0.0255032) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 15.177 ± 0.149064 (W 0.0219727) | 15.4113 ± 0.0903807 (W 0.0524121) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.39888 ± 0.00171386 (W 0.00294365) | 0.402421 ± 0.00258724 (W 0.00335027) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.484945 ± 0.00479655 (W 0.0048852) | 0.414395 ± 0.00302299 (W 0.00286344) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 14.0419 ± 0.236764 (W 0.0529417) | 14.1607 ± 0.195344 (W 0.0222067) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.372247 ± 0.000108855 (W 0.00193847) | 0.375931 ± 0.000604484 (W 0.0019443) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 46.6328 ± 0.918386 (W 0.110141) | 48.7565 ± 0.164584 (W 0.0921974) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 46.6161 ± 0.888961 (W 0.0494363) | 48.7675 ± 0.180761 (W 0.10303) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.36858 ± 0.00391113 (W 0.00406663) | 1.57889 ± 0.00589346 (W 0.00399314) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.37181 ± 0.00329684 (W 0.00426299) | 1.72118 ± 0.00825059 (W 0.00682646) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.8347 ± 0.797248 (W 0.0250657) | 27.4459 ± 0.801437 (W 0.0525842) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.8145 ± 0.838819 (W 0.0355166) | 27.6034 ± 0.75906 (W 0.0410866) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.735628 ± 0.00156397 (W 0.00408673) | 1.14299 ± 0.00425336 (W 0.00306168) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.73993 ± 0.00258576 (W 0.00286357) | 1.14393 ± 0.00349126 (W 0.00328959) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.6668 ± 0.0725031 (W 0.0326215) | 18.3778 ± 0.0369798 (W 0.0734935) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 17.2797 ± 0.0931131 (W 0.0723308) | 17.8634 ± 0.108412 (W 0.116818) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.550868 ± 0.00736653 (W 0.00522194) | 0.851829 ± 0.00454611 (W 0.00306071) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.551041 ± 0.00392898 (W 0.00237877) | 0.833313 ± 0.00563848 (W 0.00238272) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.97035 ± 0.0036427 (W 0.00251006) | 2.89282 ± 0.00792929 (W 0.270173) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.20189 ± 0.00566165 (W 0.005462) | 3.0403 ± 0.270273 (W 0.221288) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.078355 ± 0.000253358 (W 0.000415402) | 0.119672 ± 0.00285144 (W 0.0105183) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.089566 ± 0.000192219 (W 0.000441405) | 0.131306 ± 0.0122499 (W 0.0173209) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 4.09348 ± 0.0150873 (W 0.0136632) | 4.51879 ± 0.00910178 (W 0.0135854) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.38448 ± 0.0285821 (W 0.00678339) | 4.86381 ± 0.014849 (W 0.0092503) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.160609 ± 0.000961 (W 0.00203926) | 0.239464 ± 0.00121913 (W 0.00241199) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.243234 ± 0.000537751 (W 0.00684486) | 0.253185 ± 0.00178488 (W 0.00184459) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.83387 ± 0.0117614 (W 0.00766872) | 4.26956 ± 0.569915 (W 0.0246058) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.146171 ± 0.000880682 (W 0.000194636) | 0.213666 ± 0.000469293 (W 0.00217014) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 21.5213 ± 0.0373169 (W 0.0373045) | 28.4434 ± 0.0549132 (W 0.0666449) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 21.5456 ± 0.0298705 (W 0.0305835) | 28.4268 ± 0.873733 (W 0.45728) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.01439 ± 0.00187951 (W 0.00113739) | 1.49172 ± 0.0287127 (W 0.00329206) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.01615 ± 0.00204116 (W 0.0010852) | 1.50814 ± 0.0117778 (W 0.0124504) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 14.3435 ± 0.0237034 (W 0.0292948) | 21.5932 ± 0.0421135 (W 0.0427219) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 14.3461 ± 0.0130601 (W 0.0324172) | 21.6919 ± 0.00201138 (W 0.0548023) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.682345 ± 0.00133283 (W 0.00432769) | 1.25435 ± 0.010493 (W 0.00393713) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.684918 ± 0.00349031 (W 0.00307313) | 1.26615 ± 0.00286503 (W 0.00441837) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 14.2095 ± 0.0345486 (W 0.0155119) | 15.2559 ± 0.018087 (W 0.0287219) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 14.2297 ± 0.0226373 (W 0.0253884) | 15.1671 ± 0.145558 (W 0.106913) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.562274 ± 0.00216945 (W 0.00701196) | 0.851287 ± 0.00723316 (W 0.0108641) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.557064 ± 0.0039226 (W 0.00965943) | 0.821165 ± 0.0167565 (W 0.00598209) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.10113 ± 0.00493963 (W 0.00419386) | 1.82399 ± 0.0371469 (W 0.0500671) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.3794 ± 0.0056872 (W 0.00567823) | 2.10103 ± 0.0331093 (W 0.0343495) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.050924 ± 0.000695002 (W 0.000493076) | 0.095949 ± 0.00123732 (W 0.00342871) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.06431 ± 0.000102144 (W 0.000529731) | 0.10756 ± 0.0032474 (W 0.00301258) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 2.26163 ± 0.0063419 (W 0.00547782) | 4.19315 ± 0.0426285 (W 0.101129) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 2.57118 ± 0.0270657 (W 0.0083055) | 4.45353 ± 0.0600319 (W 0.0597985) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.123962 ± 0.00131744 (W 0.000355134) | 0.315709 ± 0.00209499 (W 0.00498844) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.202059 ± 0.00581539 (W 0.00790028) | 0.339305 ± 0.00239487 (W 0.00417264) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 2.11096 ± 0.0073761 (W 0.00568048) | 3.76893 ± 0.0291758 (W 0.0704223) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.109755 ± 8.75233e-05 (W 0.000247874) | 0.276606 ± 0.00241839 (W 0.00417118) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 17.7143 ± 0.0817209 (W 0.0263022) | 29.4424 ± 0.326348 (W 0.178675) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 17.821 ± 0.0647814 (W 0.0251774) | 29.4815 ± 0.659372 (W 0.522229) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.969852 ± 0.00321883 (W 0.00300584) | 1.61717 ± 0.00921414 (W 0.00737668) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.978635 ± 0.00180132 (W 0.00185721) | 1.64612 ± 0.029129 (W 0.0107445) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 12.8597 ± 0.0539819 (W 0.0180542) | 22.8319 ± 0.832272 (W 0.0575148) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 12.8154 ± 0.0713735 (W 0.0283824) | 22.9243 ± 0.0701411 (W 0.0673398) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.675029 ± 0.00198435 (W 0.00321046) | 1.36284 ± 0.010108 (W 0.0114341) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.677147 ± 0.0011827 (W 0.00525683) | 1.35677 ± 0.0157289 (W 0.00678934) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 58.7328 ± 0.458389 (W 0.173225) | 58.9 ± 0.542421 (W 0.162311) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 90.9705 ± 0.549915 (W 0.426243) | 87.6738 ± 0.470776 (W 0.280703) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.897558 ± 0.00207388 (W 0.00435745) | 0.891202 ± 0.00582189 (W 0.00261663) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.51443 ± 0.00246127 (W 0.0122437) | 1.43186 ± 0.00750662 (W 0.00513541) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 22.0749 ± 0.0392335 (W 0.0255749) | 126.548 ± 2.43318 (W 0.217234) | - | 🔵 faer-fork 5.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 26.7557 ± 0.105349 (W 0.0834468) | 26.4657 ± 0.212554 (W 0.0620648) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.536511 ± 0.00144323 (W 0.00343652) | 3.61817 ± 0.00966434 (W 0.0206975) | - | 🔵 faer-fork 6.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.685697 ± 0.00388051 (W 0.00289607) | 0.747631 ± 0.00164737 (W 0.00195758) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 55.3749 ± 0.379704 (W 0.0471303) | 54.9618 ± 0.131753 (W 0.125496) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 57.8596 ± 0.233319 (W 0.173921) | 59.5377 ± 0.260236 (W 0.176043) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.64364 ± 0.00416372 (W 0.00631488) | 1.65892 ± 0.00322097 (W 0.00530303) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 1.47939 ± 0.00472633 (W 0.00341) | 1.56271 ± 0.0119501 (W 0.00568923) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 51.8514 ± 0.262702 (W 0.0630023) | 52.2518 ± 0.331086 (W 0.0823051) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.56344 ± 0.00470377 (W 0.0032701) | 1.57699 ± 0.00364376 (W 0.00583395) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 136.565 ± 0.424811 (W 0.341937) | 141.281 ± 0.378191 (W 0.171692) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 136.952 ± 0.15353 (W 0.406451) | 141.496 ± 0.235508 (W 0.359249) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.11447 ± 0.00204059 (W 0.00531123) | 3.39036 ± 0.00580823 (W 0.00373301) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.12837 ± 0.0127712 (W 0.00611327) | 3.39044 ± 0.00709343 (W 0.00422563) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 55.2166 ± 0.078453 (W 0.0645433) | 61.7334 ± 0.0898367 (W 0.107983) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 55.3003 ± 0.14465 (W 0.0752028) | 61.8233 ± 0.166057 (W 0.0796734) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.49161 ± 0.000637717 (W 0.00448393) | 1.93866 ± 0.00689269 (W 0.0057932) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.48926 ± 0.00611361 (W 0.00398722) | 1.93958 ± 0.00440276 (W 0.00367654) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.1457 ± 0.0392536 (W 0.0572773) | 17.9567 ± 0.0980549 (W 0.0433507) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 37.38 ± 0.0652791 (W 0.165296) | 39.3537 ± 0.304527 (W 0.120468) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.496201 ± 0.00207658 (W 0.00381633) | 0.882696 ± 0.00170385 (W 0.00246303) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.846277 ± 0.00232912 (W 0.00734113) | 1.44209 ± 0.0129792 (W 0.00438326) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.26363 ± 0.0131777 (W 0.0138897) | 60.869 ± 14.4343 (W 13.197) | - | 🔵 faer-fork 9.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 9.85489 ± 0.00521483 (W 0.040968) | 12.2497 ± 0.08026 (W 0.437026) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.197869 ± 0.00854658 (W 0.00411097) | 1.80698 ± 0.0155268 (W 0.396614) | - | 🔵 faer-fork 9.13× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.310185 ± 0.00103404 (W 0.00204002) | 0.482294 ± 0.0261307 (W 0.0425772) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 16.9519 ± 0.0438213 (W 0.0249167) | 17.3938 ± 0.0139016 (W 0.031853) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 17.8654 ± 0.0274021 (W 0.0478056) | 19.9164 ± 0.0373865 (W 0.0298716) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.750256 ± 0.00194819 (W 0.00461487) | 0.816142 ± 0.0022043 (W 0.00312443) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.584122 ± 0.00168315 (W 0.00266963) | 0.778018 ± 0.0144691 (W 0.00192327) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.067 ± 0.05988 (W 0.027195) | 16.5212 ± 0.0930465 (W 0.031125) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.713127 ± 0.00288272 (W 0.00432627) | 0.760929 ± 0.00303388 (W 0.00545371) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 49.4491 ± 0.23235 (W 0.150333) | 60.6863 ± 0.0624282 (W 0.147134) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 49.7384 ± 0.139342 (W 0.0776392) | 60.8717 ± 0.0431372 (W 0.156013) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.72495 ± 0.00667785 (W 0.0033686) | 2.3324 ± 0.00398065 (W 0.00675154) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.73143 ± 0.0085899 (W 0.00320274) | 2.34924 ± 0.00741906 (W 0.0091553) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 25.0979 ± 0.195729 (W 0.0267548) | 36.551 ± 0.0142835 (W 0.0940389) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 25.1968 ± 0.056641 (W 0.0466405) | 36.6058 ± 0.12208 (W 0.0877682) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.06487 ± 0.00164967 (W 0.00213031) | 1.73372 ± 0.00572696 (W 0.00418199) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.06818 ± 0.00482676 (W 0.00220609) | 1.73325 ± 0.00461193 (W 0.00702905) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.8362 ± 0.00531889 (W 0.0301108) | 11.7595 ± 0.056204 (W 0.0431808) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 30.1248 ± 0.0362381 (W 0.0754242) | 32.372 ± 0.200641 (W 0.126055) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.452624 ± 0.00100453 (W 0.00141852) | 0.882429 ± 0.000930004 (W 0.00426457) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.7752 ± 0.00113501 (W 0.00393826) | 1.42506 ± 0.00229178 (W 0.00379358) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.57287 ± 0.0136871 (W 0.0102309) | 33.7262 ± 0.444829 (W 0.323534) | - | 🔵 faer-fork 9.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 7.04746 ± 0.0810176 (W 0.0388954) | 9.1834 ± 0.0755306 (W 0.18102) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.15026 ± 0.000218321 (W 0.00232278) | 1.05328 ± 0.0117667 (W 0.163294) | - | 🔵 faer-fork 7.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.254224 ± 0.000356365 (W 0.00168036) | 0.390245 ± 0.00857815 (W 0.00226542) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.6536 ± 0.050916 (W 0.0208368) | 13.753 ± 0.324033 (W 0.332866) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 11.2465 ± 0.132616 (W 0.0347523) | 15.7039 ± 0.244991 (W 0.479422) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.594549 ± 0.001089 (W 0.00482331) | 0.835894 ± 0.00738453 (W 0.013336) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.437953 ± 0.000608207 (W 0.0021675) | 0.777696 ± 0.00780714 (W 0.00745762) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.102 ± 0.0351788 (W 0.0403091) | 12.9055 ± 0.205536 (W 0.424668) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.566727 ± 0.00184062 (W 0.00469829) | 0.779409 ± 0.00587611 (W 0.0129573) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 36.2177 ± 0.0304489 (W 0.066128) | 55.4702 ± 0.710326 (W 0.933658) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.3088 ± 0.0996911 (W 0.091695) | 55.9642 ± 0.735657 (W 3.4828) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.53813 ± 0.00799368 (W 0.00366276) | 2.42706 ± 0.0184286 (W 0.0112818) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.54096 ± 0.0107167 (W 0.00538405) | 2.43854 ± 0.00106515 (W 0.0114825) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 20.0085 ± 0.0355118 (W 0.0243734) | 36.752 ± 0.107164 (W 0.216648) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.9717 ± 0.0545201 (W 0.022141) | 36.7429 ± 0.0306988 (W 0.180242) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.01727 ± 0.00457689 (W 0.00360118) | 1.8233 ± 0.00786871 (W 0.00395375) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.01954 ± 0.00213715 (W 0.00294002) | 1.82223 ± 0.00595871 (W 0.00704757) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 97.0136 ± 1.32778 (W 0.59557) | 93.9962 ± 0.555714 (W 0.336843) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 86.9546 ± 0.477378 (W 0.226564) | 82.9232 ± 0.360441 (W 0.423051) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.41127 ± 0.00574208 (W 0.00268734) | 1.26133 ± 0.00789939 (W 0.00420884) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.38906 ± 0.00413789 (W 0.00752742) | 1.21551 ± 0.00652592 (W 0.0068992) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 22.8244 ± 0.0848922 (W 0.0388513) | 21.6074 ± 0.0204188 (W 0.0365647) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 23.3704 ± 0.0266361 (W 0.0780843) | 21.8013 ± 0.0369926 (W 0.0446254) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.554835 ± 0.00229348 (W 0.00442994) | 0.540328 ± 0.00110553 (W 0.00403145) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.570974 ± 0.00168575 (W 0.00252175) | 0.554212 ± 0.00165828 (W 0.00351081) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 53.7292 ± 0.143672 (W 0.24776) | 53.7837 ± 0.287624 (W 0.516946) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 54.5154 ± 0.14516 (W 0.102233) | 54.8387 ± 0.312299 (W 0.444736) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.33653 ± 0.00587044 (W 0.00441014) | 1.34105 ± 0.00164095 (W 0.00355945) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 1.3635 ± 0.00396306 (W 0.00473504) | 1.36579 ± 0.00321104 (W 0.00366515) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 50.6859 ± 0.209962 (W 0.183792) | 50.7254 ± 0.132177 (W 0.272818) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.25927 ± 0.00562422 (W 0.00464107) | 1.25735 ± 0.00521483 (W 0.00372114) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 141.821 ± 0.20144 (W 0.10683) | 146.123 ± 0.230265 (W 0.25476) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 142.349 ± 0.129041 (W 0.356524) | 146.771 ± 0.186208 (W 0.111073) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.17857 ± 0.00622639 (W 0.00573072) | 3.45557 ± 0.0110468 (W 0.00765002) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.17868 ± 0.00330177 (W 0.0113293) | 3.43591 ± 0.00778042 (W 0.00723006) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 60.9624 ± 0.0848689 (W 0.0782247) | 67.4738 ± 0.198519 (W 0.130268) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 60.903 ± 0.0887272 (W 0.0759849) | 67.5096 ± 0.153158 (W 0.0814789) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.5771 ± 0.00281453 (W 0.00444602) | 2.02838 ± 0.00652695 (W 0.00662784) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.57663 ± 0.0030356 (W 0.00601994) | 2.03419 ± 0.00844247 (W 0.00566066) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 35.3742 ± 0.0420943 (W 0.0517335) | 36.5524 ± 0.306879 (W 0.0858442) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 34.0312 ± 0.14677 (W 0.0505601) | 34.3207 ± 0.0655487 (W 0.0505224) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.699341 ± 0.00531279 (W 0.00338942) | 1.25104 ± 0.00420997 (W 0.00433824) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.724613 ± 0.00593138 (W 0.00622835) | 1.22775 ± 0.00379556 (W 0.00508063) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.08352 ± 0.0150264 (W 0.00906257) | 7.34083 ± 0.273617 (W 0.827158) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 6.53803 ± 0.016229 (W 0.00994141) | 7.72545 ± 0.383675 (W 0.572623) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.167412 ± 0.000651514 (W 0.00163217) | 0.259364 ± 0.0121082 (W 0.0349481) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.188901 ± 0.000404564 (W 0.00316631) | 0.265542 ± 0.00992509 (W 0.0362138) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 13.9648 ± 0.0540963 (W 0.0372963) | 14.4412 ± 0.0142247 (W 0.0528637) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 14.5326 ± 0.0399859 (W 0.0194847) | 15.0475 ± 0.101625 (W 0.097252) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.42905 ± 0.00161968 (W 0.00172197) | 0.516848 ± 0.00182631 (W 0.00205132) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.461901 ± 0.00125188 (W 0.00219506) | 0.55003 ± 0.00177478 (W 0.00271092) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.1496 ± 0.0182526 (W 0.0173245) | 13.5932 ± 0.0217235 (W 0.0233776) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.393989 ± 0.000720508 (W 0.00185445) | 0.466074 ± 0.00359783 (W 0.00406982) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 54.7777 ± 0.0773581 (W 0.343876) | 65.9981 ± 0.0800383 (W 0.0711094) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 54.9328 ± 0.084919 (W 0.139268) | 66.3266 ± 0.11168 (W 1.2397) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.76313 ± 0.00550981 (W 0.00353808) | 2.36375 ± 0.0119467 (W 0.0034033) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.77702 ± 0.00597458 (W 0.0124283) | 2.37986 ± 0.008953 (W 0.00425932) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 30.6896 ± 1.54108 (W 0.041014) | 42.2194 ± 0.0757547 (W 0.0446528) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 30.6601 ± 0.0504494 (W 0.0387896) | 42.3515 ± 0.0736863 (W 0.05355) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.13888 ± 0.00793298 (W 0.00792463) | 1.79645 ± 0.00918295 (W 0.00556347) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.13568 ± 0.00266414 (W 0.00525747) | 1.80124 ± 0.00349833 (W 0.00890294) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 27.2603 ± 0.096613 (W 0.178222) | 28.8545 ± 0.0424392 (W 0.0913488) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 27.2119 ± 0.197667 (W 0.102147) | 27.9222 ± 0.152574 (W 0.0623003) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.617887 ± 0.00226686 (W 0.00236016) | 1.25492 ± 0.00374442 (W 0.00268713) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.642779 ± 0.00378668 (W 0.00310835) | 1.22974 ± 0.0123919 (W 0.00345584) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.33151 ± 0.0157618 (W 0.00711136) | 4.18155 ± 0.039016 (W 0.0872151) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.72031 ± 0.00723027 (W 0.00636898) | 4.59428 ± 0.0658738 (W 0.103791) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.102841 ± 0.000536283 (W 0.00141383) | 0.163233 ± 0.000740089 (W 0.0123413) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.129864 ± 0.000365448 (W 0.000390844) | 0.190444 ± 0.00188969 (W 0.00289735) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.378 ± 0.0313916 (W 0.0130225) | 10.3897 ± 0.110143 (W 0.322148) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 7.87891 ± 0.0323784 (W 0.0162499) | 11.3338 ± 0.253986 (W 0.358702) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.275505 ± 0.000456213 (W 0.00202289) | 0.540188 ± 0.00488486 (W 0.0105735) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.312921 ± 0.00056997 (W 0.00149611) | 0.586775 ± 0.00622251 (W 0.0132457) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.9011 ± 0.0333182 (W 0.0113619) | 9.77747 ± 0.085483 (W 0.432629) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.251771 ± 0.00119852 (W 0.00247692) | 0.488704 ± 0.00813339 (W 0.00651749) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 41.5896 ± 0.112074 (W 0.118092) | 60.6838 ± 0.484744 (W 0.794615) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 41.682 ± 0.0546741 (W 0.0568027) | 61.5072 ± 0.760199 (W 4.15194) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.57979 ± 0.00406646 (W 0.00467281) | 2.4573 ± 0.0220158 (W 0.00469483) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.58558 ± 0.00918345 (W 0.00286741) | 2.48111 ± 0.00687741 (W 0.0140146) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 25.6218 ± 0.0532366 (W 0.0350882) | 42.4315 ± 0.0358611 (W 0.218399) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 25.6443 ± 0.0633358 (W 0.0866211) | 42.6279 ± 0.546777 (W 0.385418) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.08526 ± 0.00511272 (W 0.00455156) | 1.90291 ± 0.0145408 (W 0.00585636) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.0872 ± 0.00772971 (W 0.0040113) | 1.89759 ± 0.00782663 (W 0.0121418) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `lu/256x4096`, `lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `solve/256x4096`, `solve/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
