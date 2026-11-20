# `faer-kernels` on `zen5-cpu`

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
- timestamp: `2026-11-19T11:28:12.048534Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `scripts/record_run.py zen5-cpu faer-kernels --checkout extern/t4a-faer --upstream extern/upstream-faer --vendor mkl --engines faer-fork,lapack-mkl --families lu --dtypes f64 --threads 1,4,8 --regimes linalg-small,linalg-mid --shapes 32,64,96,128,256,512 --reps 5 --aa 6 --prime-ms 250 --label lu93-daily-mkl --no-publish`
- raw data: `data/results/zen5-cpu/faer-kernels/20261119T112125Z-lu93-daily-mkl/`

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
- CPU sets: 1T -> `4`, 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.43%, worst 3.98% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 0 of 90 engine cells exceed 5% and carry `!`; median within-process relative SD 0.69%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

## linalg-small, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n32/b1` | 0.006447 ± 3.27109e-05 (W 6.31486e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008566 ± 5.81091e-05 (W 7.67968e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.023404 ± 0.000112428 (W 0.000119144) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.029811 ± 0.00015523 (W 0.000207647) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.055439 ± 0.000402346 (W 0.000111639) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.076884 ± 0.000383533 (W 0.00210071) | - | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n32/b1` | 0.0065875 ± 7.89574e-05 (W 6.84107e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008791 ± 4.97621e-05 (W 8.94413e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.028709 ± 0.000214533 (W 0.000445181) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.030402 ± 0.000157549 (W 0.000150731) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.064852 ± 0.000619817 (W 0.00139408) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.073167 ± 0.000387122 (W 0.0021542) | - | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n32/b1` | 0.006612 ± 6.36577e-05 (W 6.00384e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0086965 ± 5.5913e-05 (W 9.19732e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.029771 ± 0.000280405 (W 0.000546446) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0305525 ± 0.000153224 (W 0.000195739) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.068213 ± 0.00068964 (W 0.00152834) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.075467 ± 0.000699153 (W 0.00212972) | - | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n32/b1` | 0.005105 ± 6.71089e-05 (W 7.79172e-05) | 0.01069 ± 5.18218e-05 (W 7.72022e-05) | 🔵 faer-fork 2.09× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0066275 ± 5.4286e-05 (W 5.58826e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020638 ± 0.000198334 (W 0.000277599) | 0.034585 ± 0.000125923 (W 0.000103363) | 🔵 faer-fork 1.68× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0235545 ± 0.000139266 (W 0.000144047) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.049107 ± 0.000263556 (W 0.000399628) | 0.076634 ± 0.000463148 (W 0.000205166) | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.061034 ± 0.000318433 (W 0.00212141) | - | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n32/b1` | 0.005305 ± 4.27103e-05 (W 5.65705e-05) | 0.0106145 ± 6.65445e-05 (W 0.000113999) | 🔵 faer-fork 2.00× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006903 ± 4.27395e-05 (W 8.46932e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0205635 ± 0.000159596 (W 0.000186275) | 0.034515 ± 0.000279161 (W 0.000193573) | 🔵 faer-fork 1.68× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.02398 ± 0.000234572 (W 0.000233577) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.04271 ± 0.000209765 (W 0.000250884) | 0.076293 ± 0.000276572 (W 0.000718069) | 🔵 faer-fork 1.79× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.058605 ± 0.000291452 (W 0.00122457) | - | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n32/b1` | 0.005305 ± 5.17687e-05 (W 6.35927e-05) | 0.0106145 ± 5.36672e-05 (W 0.000126224) | 🔵 faer-fork 2.00× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0068625 ± 9.14121e-05 (W 8.77277e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02128 ± 0.000139198 (W 0.000178266) | 0.034555 ± 0.000103795 (W 0.000109966) | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024105 ± 0.000182812 (W 0.000208958) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0429055 ± 0.000332572 (W 0.000240729) | 0.076829 ± 0.0002532 (W 0.000182889) | 🔵 faer-fork 1.79× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.060829 ± 0.000241613 (W 0.00197829) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `lu/32`, `lu/64`, `lu/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n128/b1` | 0.104797 ± 0.000151099 (W 0.000186598) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.127434 ± 0.000357233 (W 0.000988312) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.570796 ± 0.00112978 (W 0.00273587) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.9513 ± 0.00219164 (W 0.00288951) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.89209 ± 0.00976499 (W 0.00547317) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.7029 ± 0.0576455 (W 0.00823211) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n128/b1` | 0.116894 ± 0.00095095 (W 0.00177969) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.156058 ± 0.00181233 (W 0.00124566) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.409022 ± 0.00354112 (W 0.00212024) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.72002 ± 0.00298202 (W 0.00464404) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.77403 ± 0.00751776 (W 0.00638788) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.54974 ± 0.0262751 (W 0.0107334) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n128/b1` | 0.123982 ± 0.00193212 (W 0.00174433) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.115031 ± 0.000230721 (W 0.00108588) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.338145 ± 0.000995793 (W 0.00214327) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.688742 ± 0.00449015 (W 0.00438467) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.30201 ± 0.00364744 (W 0.00283516) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.19838 ± 0.0293598 (W 0.00875521) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n128/b1` | 0.0972175 ± 0.000274293 (W 0.00047814) | 0.143133 ± 0.00030779 (W 0.00132155) | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.104261 ± 0.000652286 (W 0.000736181) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.564739 ± 0.00269957 (W 0.00306623) | 0.795989 ± 0.00128198 (W 0.00270964) | 🔵 faer-fork 1.41× faster than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.846407 ± 0.00571754 (W 0.00271467) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.7225 ± 0.0100399 (W 0.00749691) | 5.06313 ± 0.00900518 (W 0.0066835) | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 5.1385 ± 0.0117524 (W 0.00737416) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n128/b1` | 0.078302 ± 0.00112011 (W 0.000650487) | 0.142232 ± 0.000473069 (W 0.00151153) | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.131256 ± 0.00147006 (W 0.000968541) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.332264 ± 0.00262087 (W 0.0028335) | 0.295604 ± 0.00145156 (W 0.00325) | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.618154 ± 0.00285699 (W 0.00394996) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.5674 ± 0.00446971 (W 0.00553132) | 1.49066 ± 0.00586142 (W 0.0057342) | 🔴 faer-fork 1.05× slower than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 2.98834 ± 0.0107751 (W 0.0079439) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `lu/n128/b1` | 0.0784875 ± 0.000688024 (W 0.000393607) | 0.142948 ± 0.000246991 (W 0.00151716) | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.090299 ± 0.000318782 (W 0.000468244) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.291502 ± 0.00134364 (W 0.00192865) | 0.285019 ± 0.000797224 (W 0.00305471) | · |  |
| `lu/n256/b1 [api-inclusive]` | 0.582178 ± 0.00279871 (W 0.00256089) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.20287 ± 0.00599368 (W 0.00522015) | 1.10995 ± 0.00517889 (W 0.00993026) | 🔴 faer-fork 1.08× slower than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 2.65925 ± 0.018238 (W 0.00867392) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `lu/128`, `lu/256`, `lu/512`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
