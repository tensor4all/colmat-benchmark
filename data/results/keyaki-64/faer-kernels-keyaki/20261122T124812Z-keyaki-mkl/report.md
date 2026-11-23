# `faer-kernels-keyaki` on `keyaki-64`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above); `faer-upstream` (the provider in the measured revision; see library and upstream above)
- harness commit: `654078f104657a84bd7d179799c57177411b2987`
- toolkit commit: `ec1366c391cbd5023116ddcb6fdadc9c5b76f582`
- measured binary SHA-256: `4cee43d7741bfd9ee3cf9257e89f43f4ac0924e5d57a99d2078a7befcb09ec33`
- hardware profile: `keyaki-64`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-22T14:56:14.365519Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `scripts/record_run.py keyaki-64 faer-kernels-keyaki --checkout extern/t4a-faer --upstream extern/upstream-faer --vendor mkl --engines faer-fork,faer-upstream,lapack-mkl --reps 5 --aa 3 --prime-ms 250 --label keyaki-mkl`
- raw data: `data/results/keyaki-64/faer-kernels-keyaki/20261122T124812Z-keyaki-mkl/`

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
  Known blind spot: A vendor arm records the library it linked. OpenBLAS and AOCL report the kernel their runtime dispatched to (AOCL for BLIS and libflame separately); MKL names no kernel and no processor count, so its manifest leaves both empty rather than inferring them.
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

- CPU: `AMD EPYC 7702P`
- logical CPUs: `64`
- L3: `256 MiB (16 instances)`
- L3 domains: `16 MiB shared per 4-core CCX, 16 CCXs (CPUs 4k-4k+3)`
- OS / arch: `Linux 6.8.0-107-generic` / `x86_64`
- hostname: `9d90f4a9adb0`
- CPU sets: 16T -> `4-19`, 32T -> `4-35`, 60T -> `4-63`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 3 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.91%, worst 32895.34% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 1540 of 4908 engine cells exceed 5% and carry `!`; median within-process relative SD 0.73%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
two rows (combined between-start SD, floored at 5%) and larger than each of the two rows' `±`,
always with the measured library as the subject:
`🔵 <row> X faster than <row>` / `🔴 <row> X slower than <row>` (the subject is the first column of the table, the compared row is named in
the cell, not only in the column header), `·` inside the band or within a row's `±`, `-` that row is not
measured in this arm. `notes` carries the cell warnings (`*` a declared row has no measurement,
`!` between-start range over 5%, `p` position effect over 5%, dispatch-dominated means the
row is under 20 us so 4T/8T pay thread-region start-up). Per-start ratios and every
chronological sample are in the run directory named at the top of this page (`run*.csv`).
These describe the recorded revisions; they are not a regression claim.

## gemm-small, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00316 ± 3.15753e-05 (W 2.58882e-05) | 0.007471 ± 0.000635164 (W 0.00128765) | 0.0111 ± 4.0501e-05 (W 0.000132862) | 🔵 faer-fork 2.36× faster than `faer-upstream` | 🔵 faer-fork 3.51× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01504 ± 2.13776e-05 (W 6.72473e-05) | 0.011041 ± 0.000779083 (W 0.000427333) | 0.020751 ± 0.00234926 (W 0.000444893) | 🔴 faer-fork 1.36× slower than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.013401 ± 0.000266896 (W 0.000284201) | 0.0145 ± 3.80832e-05 (W 0.000733681) | 0.022401 ± 0.00123718 (W 0.00122615) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.003121 ± 1.57162e-05 (W 2.11707e-05) | 0.01168 ± 1.26182 (W 0.0010515) | 0.01414 ± 0.000262831 (W 0.000472465) | · | 🔵 faer-fork 4.53× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.015601 ± 3e-05 (W 0.000144859) | 2.17625 ± 2.9058 (W 0.0215392) | 0.018161 ± 0.000586202 (W 0.000584001) | · | 🔵 faer-fork 1.16× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.01389 ± 0.000246313 (W 0.000391386) | 0.022551 ± 0.00383302 (W 0.00171599) | 0.022291 ± 0.000601692 (W 0.000593776) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.003161 ± 3.50761e-05 (W 1.14978e-05) | 0.015091 ± 0.0015513 (W 0.00152815) | 0.023432 ± 0.00652055 (W 0.000380471) | 🔵 faer-fork 4.77× faster than `faer-upstream` | 🔵 faer-fork 7.41× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.016091 ± 0.000132666 (W 0.000129693) | 0.022341 ± 0.00248476 (W 0.00162064) | 0.026061 ± 0.00181248 (W 0.000624058) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔵 faer-fork 1.62× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.013711 ± 0.000156845 (W 0.000257527) | 0.035722 ± 1.24769 (W 0.00234078) | 0.049282 ± 0.00951954 (W 0.00207699) | · | 🔵 faer-fork 3.59× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00315 ± 3.78594e-05 (W 3.83406e-05) | 0.00706 ± 1.25643 (W 0.0014084) | 0.01103 ± 0.00174484 (W 0.000236166) | · | 🔵 faer-fork 3.50× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.015121 ± 0.00131941 (W 0.000210084) | 0.01222 ± 0.000471233 (W 0.000889502) | 0.017831 ± 0.000618789 (W 0.000435034) | 🔴 faer-fork 1.24× slower than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.012731 ± 0.000629913 (W 0.000262069) | 0.01512 ± 2.87124 (W 0.000763511) | 0.021921 ± 0.000517977 (W 0.00216587) | · | 🔵 faer-fork 1.72× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00314 ± 1.73205e-05 (W 2.43146e-05) | 0.01 ± 0.000907341 (W 0.000586073) | 0.01562 ± 0.000162143 (W 0.000424347) | 🔵 faer-fork 3.18× faster than `faer-upstream` | 🔵 faer-fork 4.97× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.015361 ± 4.6188e-05 (W 8.66326e-05) | 2.19499 ± 1.26319 (W 0.0170608) | 0.018361 ± 0.000707272 (W 0.000492131) | 🔵 faer-fork 142.89× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.013151 ± 0.000260462 (W 0.000267862) | 0.016571 ± 0.00264161 (W 0.000748953) | 0.021801 ± 0.00222144 (W 0.000921836) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00317 ± 2.3094e-05 (W 2.34563e-05) | 0.01643 ± 0.00205144 (W 0.0021161) | 0.026271 ± 0.00366669 (W 0.0011214) | 🔵 faer-fork 5.18× faster than `faer-upstream` | 🔵 faer-fork 8.29× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0164 ± 5.70117e-05 (W 6.41872e-05) | 0.01463 ± 1.28706 (W 0.00268473) | 0.030331 ± 0.00238967 (W 0.00109296) | · | 🔵 faer-fork 1.85× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.01562 ± 0.00235051 (W 0.000398529) | 0.018291 ± 0.0039926 (W 0.00191566) | 0.029642 ± 0.00798565 (W 0.00148121) | · | 🔵 faer-fork 1.90× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.011081 ± 5.7735e-07 (W 6.13946e-05) | 0.01093 ± 2.09841e-05 (W 2.6857e-05) | 0.017681 ± 0.00141182 (W 0.000529292) | · | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01456 ± 0.000429872 (W 0.000230153) | 2.18044 ± 1.25093 (W 0.00796394) | 0.024321 ± 0.000325013 (W 0.000506162) | 🔵 faer-fork 149.76× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.031232 ± 0.000127071 (W 0.00021239) | 0.032972 ± 0.00033126 (W 0.00103122) | 0.042972 ± 0.000472687 (W 0.000404747) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |

## gemm-small, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.01109 ± 1.73205e-05 (W 1.90866e-05) | 0.010921 ± 1.51767e-05 (W 3.6487e-05) | 0.017891 ± 0.000184518 (W 0.000786893) | · | 🔵 faer-fork 1.61× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.014711 ± 0.000516776 (W 0.000336571) | 2.17148 ± 2.90096 (W 0.0154911) | 0.038882 ± 0.00126429 (W 0.00175145) | · | 🔵 faer-fork 2.64× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020321 ± 0.000389401 (W 0.000413497) | 0.022471 ± 0.000608422 (W 0.000593902) | 0.034082 ± 0.00175321 (W 0.000416954) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.68× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-small, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.01106 ± 6.95006e-05 (W 3.11657e-05) | 0.01093 ± 4.50592e-05 (W 2.8571e-05) | 0.025302 ± 0.00160207 (W 0.000696788) | · | 🔵 faer-fork 2.29× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0151 ± 0.000595279 (W 0.000268011) | 0.033042 ± 1.2532 (W 0.00179446) | 0.042562 ± 0.00274591 (W 0.00298974) | · | 🔵 faer-fork 2.82× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020691 ± 0.000549758 (W 0.000353624) | 0.030181 ± 1.28145 (W 0.00153462) | 0.056863 ± 0.0028603 (W 0.00185803) | · | 🔵 faer-fork 2.75× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-small, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.01101 ± 2.47049e-05 (W 2.46435e-05) | 0.01088 ± 1.18462e-05 (W 2.82542e-05) | 0.01851 ± 0.00119694 (W 0.000472366) | · | 🔵 faer-fork 1.68× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01443 ± 0.000766094 (W 0.000356964) | 0.015471 ± 2.87056 (W 0.000534856) | 0.023991 ± 0.000354034 (W 0.000285465) | · | 🔵 faer-fork 1.66× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.031102 ± 0.000339544 (W 0.000344693) | 0.032592 ± 0.00037072 (W 0.000588201) | 0.042742 ± 0.00138601 (W 0.00044852) | · | 🔵 faer-fork 1.37× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-small, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.01099 ± 1.78979e-05 (W 2.05985e-05) | 0.01085 ± 2.60832e-05 (W 2.61591e-05) | 0.017261 ± 0.000517333 (W 0.000541959) | · | 🔵 faer-fork 1.57× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01494 ± 0.000231042 (W 0.000438169) | 2.16631 ± 1.23981 (W 0.00282748) | 0.039352 ± 0.0016917 (W 0.00245464) | 🔵 faer-fork 145.00× faster than `faer-upstream` | 🔵 faer-fork 2.63× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020451 ± 0.000200749 (W 0.000444348) | 0.023681 ± 0.000705006 (W 0.000845534) | 0.036322 ± 0.00181019 (W 0.000412053) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 1.78× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-small, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.011101 ± 3.78594e-05 (W 4.16689e-05) | 0.01094 ± 1e-05 (W 3.51326e-05) | 0.042832 ± 0.00911702 (W 0.00133044) | · | 🔵 faer-fork 3.86× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.015191 ± 0.00105683 (W 0.000146046) | 0.025772 ± 1.24929 (W 0.00123462) | 0.042972 ± 0.00139102 (W 0.00167864) | · | 🔵 faer-fork 2.83× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020771 ± 0.000260064 (W 0.000350528) | 0.029961 ± 0.00322314 (W 0.00110457) | 0.062753 ± 0.00490369 (W 0.00108606) | 🔵 faer-fork 1.44× faster than `faer-upstream` | 🔵 faer-fork 3.02× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.023901 ± 0.00115001 (W 0.000583541) | 0.024151 ± 0.00122818 (W 0.000876676) | 0.039702 ± 0.000417892 (W 0.000285535) | · | 🔵 faer-fork 1.66× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.048503 ± 0.00012779 (W 0.000405302) | 0.050883 ± 3.24552 (W 0.00188207) | 0.070304 ± 0.000490541 (W 0.000370337) | · | 🔵 faer-fork 1.45× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.146547 ± 0.00542253 (W 0.00249543) | 0.150107 ± 0.00487521 (W 0.00286502) | 0.158258 ± 0.00508456 (W 0.00174424) | · | 🔵 faer-fork 1.08× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.355457 ± 0.00320063 (W 0.00922384) | 2.5618 ± 1.28081 (W 0.0142529) | 0.383958 ± 0.00832005 (W 0.00242269) | 🔵 faer-fork 7.21× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 0.794327 ± 0.00418383 (W 0.0154487) | 0.788617 ± 0.0100003 (W 0.0110125) | 1.06817 ± 0.0188388 (W 0.00279841) | · | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 2.69942 ± 0.031754 (W 0.0861604) | 2.61526 ± 0.00887807 (W 0.0232226) | 2.61533 ± 0.00582091 (W 0.0216526) | · | · |  |

## gemm-mid, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.017171 ± 0.000274955 (W 0.000338334) | 0.023201 ± 1.24549 (W 0.000647248) | 0.033112 ± 0.00091514 (W 0.000546152) | · | 🔵 faer-fork 1.93× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.034682 ± 0.000474447 (W 0.000290438) | 0.037732 ± 0.000827702 (W 0.000436887) | 0.053423 ± 0.00261696 (W 0.000439468) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.54× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.073953 ± 0.00132976 (W 0.000403108) | 0.077113 ± 0.00114008 (W 0.000672653) | 0.110175 ± 0.000331437 (W 0.00122436) | · | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.189819 ± 0.00381506 (W 0.0027875) | 0.191369 ± 1.25544 (W 0.0030816) | 0.250962 ± 0.00217468 (W 0.00260767) | · | 🔵 faer-fork 1.32× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 0.532705 ± 0.00571162 (W 0.00751391) | 2.70767 ± 1.25054 (W 0.00397304) | 0.63385 ± 0.00191113 (W 0.00277678) | 🔵 faer-fork 5.08× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/768x768x768/b1` | 1.63239 ± 0.010826 (W 0.0144356) | 5.33722 ± 0.654978 (W 0.386066) | 1.64347 ± 0.0298074 (W 0.0662543) | 🔵 faer-fork 3.27× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.016951 ± 0.000288675 (W 0.000215128) | 0.024731 ± 1.27482 (W 0.00282806) | 0.042472 ± 0.00548217 (W 0.00110391) | · | 🔵 faer-fork 2.51× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.028811 ± 0.0017052 (W 0.000215476) | 0.037181 ± 0.00208762 (W 0.00111197) | 0.078544 ± 0.00782716 (W 0.00206467) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 2.73× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.053253 ± 0.00150642 (W 0.000345872) | 0.059793 ± 0.00896067 (W 0.00170575) | 0.140257 ± 0.0371713 (W 0.00106178) | · | 🔵 faer-fork 2.63× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.122176 ± 0.00952789 (W 0.00196296) | 0.131116 ± 1.25011 (W 0.00238344) | 0.190819 ± 0.061846 (W 0.00241048) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 0.339816 ± 0.00205855 (W 0.00394057) | 0.341226 ± 0.000827103 (W 0.00240763) | 0.43508 ± 0.0200387 (W 0.00764167) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 1.32467 ± 0.0150743 (W 0.0111447) | 1.38874 ± 0.00794697 (W 0.0157527) | 1.15433 ± 0.0710423 (W 0.0293397) | · | 🔴 faer-fork 1.15× slower than `lapack-mkl` | `!` between-start range over 5% |

## gemm-mid, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.023521 ± 0.00025036 (W 0.00030286) | 0.024422 ± 0.00017406 (W 0.000351722) | 0.038972 ± 0.000626605 (W 0.000556372) | · | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.048912 ± 0.000441928 (W 0.000521296) | 0.050812 ± 0.000515186 (W 0.000802194) | 0.070223 ± 0.000499176 (W 0.000371718) | · | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.153367 ± 0.00649764 (W 0.00554248) | 0.146747 ± 0.003567 (W 0.00324131) | 0.151678 ± 0.00386234 (W 0.000924722) | · | · | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.360487 ± 0.00342728 (W 0.00476999) | 0.359257 ± 0.00135308 (W 0.00534153) | 0.385758 ± 0.00241808 (W 0.00195506) | · | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.781987 ± 0.00240009 (W 0.0573478) | 0.796398 ± 0.00546284 (W 0.00936613) | 1.05887 ± 0.00736897 (W 0.00909762) | · | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 2.68244 ± 0.0383172 (W 0.102651) | 2.62787 ± 1.61279 (W 0.0178683) | 2.63153 ± 0.0262397 (W 0.0274691) | · | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.016991 ± 0.000545252 (W 0.000240286) | 0.023911 ± 3.30045 (W 0.00102565) | 0.033202 ± 0.00177091 (W 0.000608498) | · | 🔵 faer-fork 1.95× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.034072 ± 0.000561656 (W 0.000228705) | 0.037442 ± 0.00119804 (W 0.000771095) | 0.053723 ± 0.00147533 (W 0.000342817) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.075064 ± 0.000854231 (W 0.000521658) | 0.076893 ± 1.25454 (W 0.00224736) | 0.109336 ± 0.00212904 (W 0.00153109) | · | 🔵 faer-fork 1.46× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.184479 ± 0.0042815 (W 0.00157543) | 0.19559 ± 0.0058948 (W 0.0030475) | 0.251082 ± 0.00166286 (W 0.00215253) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 0.531085 ± 0.00117377 (W 0.00725505) | 0.595888 ± 0.0388574 (W 0.0145533) | 0.63783 ± 0.00722103 (W 0.00233214) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 1.61412 ± 0.00330871 (W 0.0109651) | 1.65172 ± 2.13411 (W 0.0281451) | 1.66726 ± 0.0177612 (W 0.0535188) | · | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.017001 ± 0.000164621 (W 0.000182154) | 0.021981 ± 0.00247128 (W 0.00127815) | 0.047862 ± 0.00187913 (W 0.00124187) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 2.82× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.030422 ± 0.00234827 (W 0.000543967) | 0.030951 ± 0.00293173 (W 0.000988216) | 0.060533 ± 0.0218965 (W 0.00174672) | · | 🔵 faer-fork 1.99× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.053993 ± 0.00111157 (W 0.000291965) | 0.069084 ± 0.00542455 (W 0.00145343) | 0.170178 ± 0.0405971 (W 0.00264128) | 🔵 faer-fork 1.28× faster than `faer-upstream` | 🔵 faer-fork 3.15× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.126036 ± 0.00230413 (W 0.000784653) | 0.127406 ± 1.25771 (W 0.0019294) | 0.191909 ± 0.0177933 (W 0.00256825) | · | 🔵 faer-fork 1.52× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 0.337976 ± 0.00151856 (W 0.00379595) | 0.342066 ± 0.0027975 (W 0.00332158) | 0.449051 ± 0.0198587 (W 0.00862874) | · | 🔵 faer-fork 1.33× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 1.32759 ± 0.0218898 (W 0.00689848) | 3.48332 ± 1.82793 (W 0.121829) | 1.24549 ± 0.0557889 (W 0.0665613) | 🔵 faer-fork 2.62× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.054672 ± 0.00285071 (W 0.00140701) | 0.055263 ± 0.000879347 (W 0.00109329) | 0.085304 ± 0.00158 (W 0.00191599) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.149147 ± 0.000566951 (W 0.00155545) | 0.149317 ± 0.000796806 (W 0.00173382) | 0.228761 ± 0.00563402 (W 0.00234669) | · | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.384108 ± 0.00284723 (W 0.00379765) | 0.398029 ± 1.26587 (W 0.0149018) | 0.534055 ± 0.0015847 (W 0.00187983) | · | 🔵 faer-fork 1.39× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 1.1727 ± 0.0096215 (W 0.00928648) | 1.1629 ± 0.0254867 (W 0.0115528) | 1.54632 ± 0.00403403 (W 0.00760786) | · | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.61584 ± 0.0055208 (W 0.0100482) | 2.59869 ± 0.0298651 (W 0.0143053) | 3.57726 ± 0.0274408 (W 0.0199596) | · | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 8.98417 ± 0.0371589 (W 0.0530164) | 8.99016 ± 0.133476 (W 0.0996168) | 11.4672 ± 0.00411633 (W 0.0226801) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |

## gemm-mid, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.036701 ± 0.00131364 (W 0.000278667) | 0.039642 ± 0.000172143 (W 0.0004236) | 0.056783 ± 0.00108076 (W 0.000691533) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.55× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.080314 ± 0.000131721 (W 0.000400624) | 2.25912 ± 0.981718 (W 0.0186932) | 0.139266 ± 0.00101616 (W 0.0040486) | 🔵 faer-fork 28.13× faster than `faer-upstream` | 🔵 faer-fork 1.73× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.232441 ± 0.00260718 (W 0.00294446) | 0.233521 ± 0.00522957 (W 0.00283277) | 0.324485 ± 0.0051274 (W 0.0029533) | · | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.698433 ± 0.00174558 (W 0.00113793) | 0.697703 ± 1.28011 (W 0.00585162) | 0.851811 ± 0.00850784 (W 0.0206153) | · | 🔵 faer-fork 1.22× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 1.43811 ± 0.0298154 (W 0.0121442) | 1.45972 ± 2.30163 (W 0.0102492) | 1.92429 ± 0.0206904 (W 0.0294995) | · | 🔵 faer-fork 1.34× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/768x768x768/b1` | 4.93135 ± 0.0186604 (W 0.0152694) | 4.95362 ± 1.70769 (W 0.0472227) | 5.82139 ± 0.0188191 (W 0.0119437) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.028001 ± 0.000729505 (W 0.000457108) | 0.034431 ± 0.00414056 (W 0.00121191) | 0.061502 ± 0.0193951 (W 0.0014345) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 2.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.059473 ± 0.00169813 (W 0.000574124) | 0.062973 ± 0.000563057 (W 0.00151642) | 0.146267 ± 0.0224112 (W 0.00171119) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 2.46× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.131136 ± 0.00146043 (W 0.00204016) | 0.131286 ± 0.00354494 (W 0.00120372) | 0.246372 ± 0.0201682 (W 0.00329392) | · | 🔵 faer-fork 1.88× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.385768 ± 0.00371562 (W 0.00252438) | 0.390959 ± 0.00122336 (W 0.00282586) | 0.618179 ± 0.00725678 (W 0.00678865) | · | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.03906 ± 0.0888236 (W 0.00408403) | 2.44691 ± 0.903494 (W 0.00212451) | 1.33903 ± 0.0417868 (W 0.0231468) | 🔵 faer-fork 2.35× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/768x768x768/b1` | 3.30754 ± 0.0103501 (W 0.0149345) | 7.71376 ± 2.52811 (W 0.0776552) | 3.98267 ± 0.371427 (W 0.0720156) | 🔵 faer-fork 2.33× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.053973 ± 0.0011629 (W 0.000289039) | 0.057413 ± 0.00204221 (W 0.000482264) | 0.084524 ± 0.000912652 (W 0.00184541) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.57× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.148507 ± 0.00012741 (W 0.000825875) | 0.150317 ± 0.000295229 (W 0.00154115) | 0.231771 ± 0.00174345 (W 0.00183719) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.386698 ± 0.00196031 (W 0.00437579) | 0.389769 ± 3.16617 (W 0.0039873) | 0.533835 ± 0.00164865 (W 0.00328787) | · | 🔵 faer-fork 1.38× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 1.16917 ± 0.00381979 (W 0.00964969) | 1.13515 ± 2.66055 (W 0.0145528) | 1.54837 ± 0.004046 (W 0.0109878) | · | 🔵 faer-fork 1.32× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 2.61763 ± 0.00810321 (W 0.011651) | 2.58232 ± 0.0295348 (W 0.00850923) | 3.5908 ± 0.0183361 (W 0.0138185) | · | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 9.02741 ± 0.146229 (W 0.0249924) | 8.94097 ± 2.79448 (W 0.100705) | 11.4501 ± 0.0310307 (W 0.010253) | · | 🔵 faer-fork 1.27× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.037642 ± 0.00129595 (W 0.000324554) | 0.041481 ± 0.000541073 (W 0.000451926) | 0.056943 ± 0.000179602 (W 0.000981174) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.081294 ± 0.00112895 (W 0.00178658) | 0.082734 ± 1.25182 (W 0.00154312) | 0.137846 ± 0.00312663 (W 0.00254171) | · | 🔵 faer-fork 1.70× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.230341 ± 0.000525738 (W 0.00276934) | 0.234321 ± 0.00102344 (W 0.00209582) | 0.319365 ± 0.000866737 (W 0.00276733) | · | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.689743 ± 0.00207752 (W 0.00275136) | 0.728085 ± 1.15195 (W 0.00507327) | 0.859811 ± 0.00909977 (W 0.0194322) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 1.4389 ± 0.0114073 (W 0.0132208) | 1.44707 ± 0.00967519 (W 0.00650686) | 1.92542 ± 0.0173815 (W 0.0147244) | · | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 4.9185 ± 0.00730294 (W 0.0195568) | 4.95971 ± 1.70069 (W 0.0315103) | 5.84148 ± 0.0278151 (W 0.0292925) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-mid, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.027932 ± 0.00218118 (W 0.000291841) | 0.037502 ± 1.24979 (W 0.00210324) | 0.071254 ± 0.00202187 (W 0.00307224) | · | 🔵 faer-fork 2.55× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.060233 ± 0.00201222 (W 0.000620024) | 0.062623 ± 0.000419325 (W 0.00236115) | 0.136747 ± 0.0140785 (W 0.0014178) | · | 🔵 faer-fork 2.27× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.127116 ± 0.00121762 (W 0.00212635) | 0.131767 ± 0.000801218 (W 0.00103381) | 0.234651 ± 0.00052915 (W 0.00269205) | · | 🔵 faer-fork 1.85× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.385348 ± 0.000922442 (W 0.00163462) | 0.393269 ± 1.24487 (W 0.00313572) | 0.619569 ± 0.00219573 (W 0.0103941) | · | 🔵 faer-fork 1.61× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 0.983407 ± 0.0698716 (W 0.0083535) | 2.49687 ± 0.00717693 (W 0.00426482) | 1.33837 ± 0.0378529 (W 0.0206727) | 🔵 faer-fork 2.54× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/768x768x768/b1` | 3.28489 ± 0.0104149 (W 0.0182269) | 7.59712 ± 2.50824 (W 0.0199243) | 3.93088 ± 0.0209354 (W 0.0306683) | 🔵 faer-fork 2.31× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.175498 ± 0.00168448 (W 0.00223486) | 0.19251 ± 0.00638175 (W 0.00337632) | 0.249241 ± 0.00242002 (W 0.00446547) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.330545 ± 0.00635732 (W 0.0047054) | 0.350016 ± 0.0362904 (W 0.00647485) | 0.395559 ± 0.00206549 (W 0.000366335) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.155118 ± 0.00166346 (W 0.00203261) | 0.194469 ± 1.27191 (W 0.00612355) | 0.2059 ± 0.00032131 (W 0.00134295) | · | 🔵 faer-fork 1.33× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.175708 ± 0.00382214 (W 0.00339202) | 0.312075 ± 0.0787652 (W 0.0243328) | 0.205129 ± 0.00114694 (W 0.00296072) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.307295 ± 0.00431111 (W 0.00182664) | 0.358547 ± 1.01425 (W 0.00941147) | 0.40941 ± 0.00100305 (W 0.00251617) | · | 🔵 faer-fork 1.33× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.115085 ± 0.00595882 (W 0.00245604) | 0.196479 ± 1.25263 (W 0.0179742) | 0.166738 ± 0.00013 (W 0.00175346) | · | 🔵 faer-fork 1.45× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.173328 ± 0.00141736 (W 0.00453768) | 0.41958 ± 0.0701396 (W 0.00823709) | 0.175228 ± 0.00375895 (W 0.00229647) | 🔵 faer-fork 2.42× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.312595 ± 0.0123751 (W 0.00488025) | 0.380558 ± 0.935147 (W 0.010555) | 0.436671 ± 0.0374771 (W 0.0300549) | · | 🔵 faer-fork 1.40× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.118435 ± 0.00299406 (W 0.0023885) | 0.176729 ± 0.0164793 (W 0.00845303) | 0.167418 ± 0.00543042 (W 0.00121615) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 1.41× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.368107 ± 0.000598693 (W 0.0054474) | 1.40908 ± 0.0476912 (W 0.064211) | 0.304015 ± 0.00450093 (W 0.00237442) | 🔵 faer-fork 3.83× faster than `faer-upstream` | 🔴 faer-fork 1.21× slower than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.378248 ± 0.00545226 (W 0.00665032) | 0.472003 ± 0.899079 (W 0.00966444) | 0.726135 ± 0.118816 (W 0.0353867) | · | 🔵 faer-fork 1.92× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.150767 ± 0.00373097 (W 0.0018165) | 0.156087 ± 1.2553 (W 0.00245814) | 0.250371 ± 0.0119686 (W 0.00353985) | · | 🔵 faer-fork 1.66× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.238201 ± 0.00364308 (W 0.00232844) | 0.282214 ± 0.0719588 (W 0.00445046) | 0.186739 ± 0.000228408 (W 0.00194539) | · | 🔴 faer-fork 1.28× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.297024 ± 0.00401623 (W 0.00318942) | 0.314055 ± 1.02327 (W 0.0037042) | 0.750865 ± 0.00544352 (W 0.0113696) | · | 🔵 faer-fork 2.53× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.110036 ± 0.00225368 (W 0.00133803) | 2.29698 ± 1.27228 (W 0.0119727) | 0.231301 ± 0.00370984 (W 0.00350989) | 🔵 faer-fork 20.87× faster than `faer-upstream` | 🔵 faer-fork 2.10× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.239011 ± 0.00141797 (W 0.00546499) | 0.474642 ± 7.31203 (W 0.0297104) | 0.186549 ± 0.0416324 (W 0.00332647) | · | 🔴 faer-fork 1.28× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.299924 ± 0.000897937 (W 0.00254034) | 2.15854 ± 1.07725 (W 0.00328726) | 0.702384 ± 0.00877728 (W 0.00823671) | 🔵 faer-fork 7.20× faster than `faer-upstream` | 🔵 faer-fork 2.34× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.106775 ± 0.000585349 (W 0.00176022) | 0.101275 ± 3.25777 (W 0.0022368) | 0.247242 ± 0.104014 (W 0.00470659) | · | 🔵 faer-fork 2.32× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.537186 ± 0.0411065 (W 0.0109867) | 0.660322 ± 0.0910558 (W 0.0226115) | 0.859821 ± 0.0188478 (W 0.00172844) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.827419 ± 0.00882263 (W 0.0664183) | 0.748785 ± 0.00952085 (W 0.00887626) | 0.991577 ± 0.0171764 (W 0.0332755) | 🔴 faer-fork 1.11× slower than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.466312 ± 0.00461251 (W 0.00406757) | 0.459532 ± 1.16624 (W 0.00337829) | 0.738755 ± 0.00303497 (W 0.00227353) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.308185 ± 0.00759011 (W 0.00689218) | 0.4219 ± 0.0114999 (W 0.0161481) | 0.383758 ± 0.0039406 (W 0.00397649) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.697103 ± 0.0108421 (W 0.0407253) | 0.84444 ± 0.298452 (W 0.00653706) | 1.31175 ± 0.0536718 (W 0.0835912) | · | 🔵 faer-fork 1.88× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.265452 ± 0.00187915 (W 0.00328523) | 0.296675 ± 0.0168394 (W 0.00489779) | 1.31764 ± 0.00421463 (W 0.00618066) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 4.96× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.305454 ± 0.00278896 (W 0.00694881) | 0.438551 ± 0.0718438 (W 0.00903305) | 0.296884 ± 0.00190282 (W 0.00239769) | 🔵 faer-fork 1.44× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.713114 ± 0.018394 (W 0.0062682) | 1.87125 ± 1.13102 (W 0.0196655) | 1.35422 ± 0.03581 (W 0.0214275) | 🔵 faer-fork 2.62× faster than `faer-upstream` | 🔵 faer-fork 1.90× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.21491 ± 0.00140475 (W 0.00156787) | 2.38777 ± 1.21988 (W 0.0039429) | 0.975846 ± 0.00188216 (W 0.00390613) | 🔵 faer-fork 11.11× faster than `faer-upstream` | 🔵 faer-fork 4.54× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.691743 ± 0.00397009 (W 0.00371912) | 1.8931 ± 0.046095 (W 0.127927) | 11.3265 ± 3.16157 (W 0.0730988) | 🔵 faer-fork 2.74× faster than `faer-upstream` | 🔵 faer-fork 16.37× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.84245 ± 0.0169498 (W 0.00979228) | 0.815219 ± 0.0713365 (W 0.0111933) | 1.85763 ± 0.0670629 (W 0.0831089) | · | 🔵 faer-fork 2.21× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.431821 ± 0.0109994 (W 0.0028076) | 0.439201 ± 0.00952426 (W 0.00267614) | 0.792957 ± 0.0100424 (W 0.0026122) | · | 🔵 faer-fork 1.84× faster than `lapack-mkl` |  |

## gemm-tall, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.512144 ± 0.00701439 (W 0.00872294) | 0.545336 ± 0.00174803 (W 0.00721916) | 6.13364 ± 2.07248 (W 2.06474) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 11.98× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.551026 ± 0.0408847 (W 0.00301169) | 0.578998 ± 0.378429 (W 0.00262955) | 1.50558 ± 0.0381156 (W 0.115855) | · | 🔵 faer-fork 2.73× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.236962 ± 0.00122084 (W 0.00248671) | 0.246432 ± 0.0117681 (W 0.00269199) | 0.529475 ± 0.00716741 (W 0.00279891) | · | 🔵 faer-fork 2.23× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.502664 ± 0.00387743 (W 0.00521213) | 17.6348 ± 9.86176 (W 0.0956218) | 1.96 ± 0.00358266 (W 0.00221522) | 🔵 faer-fork 35.08× faster than `faer-upstream` | 🔵 faer-fork 3.90× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.62881 ± 0.00332656 (W 0.00391899) | 0.64623 ± 0.0393284 (W 0.00308993) | 1.20729 ± 0.0408833 (W 0.024227) | · | 🔵 faer-fork 1.92× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.213731 ± 0.00269441 (W 0.00189243) | 2.41272 ± 1.27004 (W 0.0257739) | 0.462832 ± 0.00570332 (W 0.00433241) | 🔵 faer-fork 11.29× faster than `faer-upstream` | 🔵 faer-fork 2.17× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.156178 ± 0.00323735 (W 0.0019594) | 0.176188 ± 1.26085 (W 0.00217203) | 0.250282 ± 0.0266929 (W 0.00338244) | · | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.106885 ± 0.00147218 (W 0.00260762) | 2.29064 ± 0.0155632 (W 0.0109896) | 0.229001 ± 0.00343238 (W 0.00395484) | 🔵 faer-fork 21.43× faster than `faer-upstream` | 🔵 faer-fork 2.14× faster than `lapack-mkl` |  |

## gemm-wide, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.107475 ± 0.00297118 (W 0.00188128) | 0.108655 ± 1.28831 (W 0.00268277) | 0.242542 ± 0.0181232 (W 0.00377502) | · | 🔵 faer-fork 2.26× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.158268 ± 0.00272389 (W 0.00154198) | 0.194189 ± 1.26192 (W 0.00988349) | 0.20735 ± 0.00170518 (W 0.00258895) | · | 🔵 faer-fork 1.31× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.119475 ± 0.00489545 (W 0.00208895) | 0.177639 ± 0.00703916 (W 0.0143005) | 0.165258 ± 0.00065395 (W 0.00167807) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-wide, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.120275 ± 0.00195841 (W 0.00187593) | 0.1958 ± 0.012775 (W 0.0159265) | 0.164198 ± 0.0126593 (W 0.00132366) | 🔵 faer-fork 1.63× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.42348 ± 0.00452395 (W 0.00300468) | 0.42257 ± 1.25133 (W 0.00278205) | 0.791048 ± 0.000992992 (W 0.00512096) | · | 🔵 faer-fork 1.87× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.236331 ± 0.0050829 (W 0.00287088) | 2.41044 ± 1.25549 (W 0.016484) | 0.514384 ± 0.00993207 (W 0.00200056) | 🔵 faer-fork 10.20× faster than `faer-upstream` | 🔵 faer-fork 2.18× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.21379 ± 0.00115781 (W 0.00200728) | 0.197369 ± 0.0109532 (W 0.00138108) | 0.456642 ± 0.00249631 (W 0.00327294) | 🔴 faer-fork 1.08× slower than `faer-upstream` | 🔵 faer-fork 2.14× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-wide, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.466222 ± 0.0126506 (W 0.00263258) | 0.459142 ± 0.00845399 (W 0.00110156) | 0.739755 ± 0.00156675 (W 0.00170929) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |

## gemm-wide, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.265322 ± 0.000834847 (W 0.00134216) | 0.297524 ± 0.0283354 (W 0.00570994) | 1.32115 ± 0.0036359 (W 0.00443692) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 4.98× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.21656 ± 0.00169554 (W 0.00181671) | 0.251192 ± 0.0129204 (W 0.00713113) | 0.979587 ± 0.00233972 (W 0.00278819) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 4.52× faster than `lapack-mkl` | `!` between-start range over 5% |

## linalg-small, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00651 ± 3.84318e-05 (W 3.00383e-05) | 0.00607 ± 2.12211e-05 (W 1.50732e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00537 ± 0.000405257 (W 2.49058e-05) | 0.00563 ± 0.000475763 (W 1.92666e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.021141 ± 0.000892711 (W 4.49444e-05) | 0.046042 ± 3.56417e-05 (W 2.81638e-05) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n64/b1 [api-inclusive]` | 0.019121 ± 0.000728583 (W 6.30079e-05) | 0.026851 ± 0.00295315 (W 3.57239e-05) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.049773 ± 5.50485e-05 (W 2.54028e-05) | 0.085934 ± 0.00525485 (W 0.00622799) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.043742 ± 6.245e-05 (W 6.1589e-05) | 0.064423 ± 2.50941 (W 0.00576675) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.118756 ± 0.00203551 (W 0.000354914) | 0.177218 ± 0.000120625 (W 0.00139641) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.122286 ± 0.00130426 (W 0.000420929) | 0.177729 ± 0.000302903 (W 0.00172556) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.354337 ± 0.00252305 (W 0.00179913) | 0.694233 ± 0.00499497 (W 0.00352546) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.377798 ± 0.0141333 (W 0.00565784) | 0.701293 ± 0.0012695 (W 0.00467818) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 0.759146 ± 0.000395769 (W 0.0030676) | 1.67127 ± 0.00132694 (W 0.00547525) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.763956 ± 0.0031339 (W 0.00349328) | 1.68463 ± 0.00852697 (W 0.00785002) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017311 ± 8.55395e-05 (W 5.13147e-05) | 0.01428 ± 0.000126532 (W 3.994e-05) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.022511 ± 3.05505e-05 (W 4.469e-05) | 0.016771 ± 0.000321455 (W 3.96232e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.077193 ± 8.87149e-05 (W 0.00012732) | 0.067673 ± 9.53939e-05 (W 0.000111992) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.091714 ± 0.000806083 (W 0.000116444) | 0.074873 ± 0.00017164 (W 7.55467e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.21012 ± 0.000322335 (W 0.00133357) | 0.183669 ± 0.00032187 (W 0.00138436) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.246122 ± 0.000269055 (W 0.0016475) | 0.20689 ± 0.000266896 (W 0.00144777) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.015361 ± 8.88819e-05 (W 4.46789e-05) | 0.016871 ± 0.000778158 (W 0.000403504) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.015471 ± 4.93288e-05 (W 6.20484e-05) | 0.0157 ± 0.000681733 (W 0.000743752) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.034852 ± 0.00296668 (W 0.000302283) | 0.051332 ± 0.0199601 (W 0.0149258) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.032942 ± 0.00254135 (W 0.000445232) | 0.070843 ± 0.0152399 (W 0.022834) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.067353 ± 0.00119684 (W 0.00119439) | 0.189059 ± 0.0373662 (W 0.023058) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.068473 ± 0.000870764 (W 0.00101303) | 0.201759 ± 0.0204623 (W 0.0133668) | - | 🔵 faer-fork 2.95× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.01103 ± 7.6922e-05 (W 4.0513e-05) | 0.011961 ± 0.000181907 (W 3.64733e-05) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013351 ± 0.000130615 (W 8.09401e-05) | 0.011481 ± 2.88675e-05 (W 4.53685e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.041092 ± 0.000125033 (W 0.000171814) | 0.055542 ± 0.0055518 (W 0.00454849) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.048933 ± 0.000372975 (W 0.00019962) | 2.2087 ± 2.87766 (W 0.00859228) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.093014 ± 0.000286235 (W 0.00122306) | 0.143657 ± 0.00983342 (W 0.00827209) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.125056 ± 0.000721942 (W 0.00036762) | 0.150917 ± 0.0142125 (W 0.0171793) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.055503 ± 4.68722e-05 (W 0.000120251) | 0.115455 ± 0.00317835 (W 0.00433133) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.053152 ± 0.00739163 (W 7.87928e-05) | 0.106495 ± 0.00296129 (W 0.00282569) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.22303 ± 0.00280141 (W 0.00195069) | 0.579867 ± 0.0197072 (W 0.0227921) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.190279 ± 0.000330454 (W 0.00150169) | 0.496284 ± 0.030935 (W 0.0101151) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.443971 ± 0.00493355 (W 0.00372102) | 1.34056 ± 0.0563108 (W 0.0758885) | - | 🔵 faer-fork 3.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.477723 ± 0.0202315 (W 0.00552963) | 1.18677 ± 0.120283 (W 0.0203697) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.034332 ± 7.23418e-05 (W 6.09204e-05) | 0.090374 ± 0.0029216 (W 0.00261619) | - | 🔵 faer-fork 2.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n64/b1` | 0.151167 ± 0.00244033 (W 0.000374751) | 0.503304 ± 0.0536828 (W 0.0120153) | - | 🔵 faer-fork 3.33× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.283164 ± 0.00854609 (W 0.00279031) | 1.12289 ± 0.127937 (W 0.0609104) | - | 🔵 faer-fork 3.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.012551 ± 1.47309e-05 (W 4.98628e-05) | 0.01301 ± 0.000156813 (W 4.22291e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016171 ± 0.000201743 (W 4.53784e-05) | 0.01258 ± 5.9808e-05 (W 5.71953e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.041422 ± 0.00019218 (W 0.000101242) | 0.065533 ± 0.00245213 (W 0.00512915) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.054083 ± 0.000275741 (W 7.41303e-05) | 0.050732 ± 0.00821761 (W 0.00765277) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.089764 ± 0.000197536 (W 0.00121219) | 0.145687 ± 0.00821925 (W 0.00679709) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.134676 ± 0.000682959 (W 0.000361161) | 0.160668 ± 2.49689 (W 0.00761597) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.268613 ± 0.000480729 (W 0.00166947) | 0.280413 ± 0.00120237 (W 0.00130185) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.278774 ± 0.00040651 (W 0.00152629) | 0.282363 ± 0.0003905 (W 0.00118993) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.867251 ± 0.00273526 (W 0.00346176) | 1.09972 ± 0.00252437 (W 0.0057771) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.872661 ± 0.002985 (W 0.00528664) | 1.11455 ± 6.23301 (W 0.00799159) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.72588 ± 0.0089829 (W 0.00532951) | 2.64762 ± 0.00650161 (W 0.0094409) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.73509 ± 0.00718479 (W 0.00540993) | 2.68264 ± 0.0130156 (W 0.0306389) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075884 ± 0.00016447 (W 6.86608e-05) | 0.159978 ± 0.000429578 (W 0.000101682) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.076443 ± 0.000140631 (W 8.07849e-05) | 0.160397 ± 0.000409598 (W 0.00221725) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.264233 ± 0.00107237 (W 0.00192957) | 0.571987 ± 0.000481352 (W 0.00194598) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.264402 ± 0.00122914 (W 0.00163183) | 0.571797 ± 0.00244156 (W 0.00238979) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.598998 ± 0.000870758 (W 0.00194251) | 1.29975 ± 0.00298901 (W 0.00229843) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.599339 ± 0.000738742 (W 0.00142663) | 1.29022 ± 0.00657347 (W 0.00221565) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00173 ± 5.7735e-06 (W 1.30843e-05) | 0.00105 ± 0 (W 8.28855e-05) | - | 🔴 faer-fork 1.65× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00384 ± 1.1547e-05 (W 1.68582e-05) | 0.002391 ± 1.57162e-05 (W 2.29826e-05) | - | 🔴 faer-fork 1.61× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006481 ± 2.0664e-05 (W 4.56103e-05) | 0.0042 ± 2.3388e-05 (W 7.12194e-05) | - | 🔴 faer-fork 1.54× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00648 ± 0.00047591 (W 3.34544e-05) | 0.006071 ± 4.14769e-05 (W 1.1371e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00542 ± 1.52753e-05 (W 2.52032e-05) | 0.004871 ± 0.000441656 (W 2.60768e-05) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.022521 ± 5.50757e-05 (W 6.06317e-05) | 0.046102 ± 2.64575e-05 (W 2.99366e-05) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.020391 ± 1e-05 (W 4.00275e-05) | 0.026851 ± 2.88675e-05 (W 3.61137e-05) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.049512 ± 0.00153872 (W 8.05866e-05) | 0.107085 ± 0.00167583 (W 0.0128494) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.044022 ± 0.00118374 (W 8.74769e-05) | 0.083914 ± 2.50571 (W 0.00915272) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.121976 ± 0.000291605 (W 0.00102036) | 0.176549 ± 5.11892e-05 (W 0.00123074) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.123116 ± 0.00046929 (W 0.000408222) | 0.177759 ± 7.38941e-05 (W 0.00117228) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.349267 ± 0.00349085 (W 0.00237268) | 0.698583 ± 0.00276334 (W 0.00191486) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.352537 ± 0.00283334 (W 0.00255769) | 5.04861 ± 0.00496363 (W 0.0171148) | - | 🔵 faer-fork 14.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.753916 ± 0.0010705 (W 0.00345888) | 1.68857 ± 0.0121346 (W 0.00621288) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.759186 ± 0.00675262 (W 0.00475) | 26.9233 ± 14.5806 (W 10.9521) | - | 🔵 faer-fork 35.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n32/b1` | 0.017261 ± 0.000135769 (W 3.13241e-05) | 0.01429 ± 6.78994e-05 (W 4.85211e-05) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.022531 ± 6.65833e-05 (W 4.52769e-05) | 0.016661 ± 7.53459e-05 (W 4.63702e-05) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.076414 ± 0.000312389 (W 9.81137e-05) | 0.068233 ± 0.000255408 (W 0.000185958) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.091544 ± 0.000210317 (W 0.000136698) | 0.075064 ± 0.000237136 (W 0.000211103) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.19979 ± 0.000220953 (W 0.00105522) | 0.19296 ± 0.000698088 (W 0.00142396) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.247802 ± 0.000229129 (W 0.00208716) | 0.2073 ± 0.00128234 (W 0.0013999) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.01552 ± 7.60723e-05 (W 4.5687e-05) | 0.015681 ± 0.00075368 (W 0.000547247) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.015411 ± 1.78979e-05 (W 2.18906e-05) | 0.01679 ± 0.000100732 (W 0.000147591) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.043062 ± 0.00540909 (W 0.000560741) | 0.060853 ± 0.0274111 (W 0.0311944) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.038682 ± 0.00198608 (W 0.000959508) | 0.105415 ± 0.0302308 (W 0.0329605) | - | 🔵 faer-fork 2.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.077463 ± 0.00396838 (W 0.00148995) | 0.162978 ± 0.0772786 (W 0.0228359) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.068284 ± 0.00110408 (W 0.000587422) | 0.311505 ± 0.0282559 (W 0.0439101) | - | 🔵 faer-fork 4.56× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.010991 ± 5.85947e-05 (W 5.32945e-05) | 0.01209 ± 7.26659e-05 (W 5.16169e-05) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013351 ± 2.08167e-05 (W 0.000104558) | 0.011461 ± 0.000100683 (W 5.75786e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.041342 ± 0.000262869 (W 6.40312e-05) | 0.060813 ± 0.00201996 (W 0.000857135) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.048862 ± 0.000160416 (W 0.00023714) | 0.049432 ± 0.00786031 (W 0.00746656) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.092615 ± 0.00075399 (W 0.000115639) | 0.136547 ± 0.0218824 (W 0.0362601) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.125986 ± 0.00071143 (W 0.000472419) | 4.48034 ± 2.5189 (W 0.0242111) | - | 🔵 faer-fork 35.56× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.055553 ± 0.000115326 (W 0.000119261) | 0.143006 ± 0.00210107 (W 0.00196639) | - | 🔵 faer-fork 2.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.053143 ± 0.000218251 (W 8.84155e-05) | 0.132166 ± 0.00758355 (W 0.00339554) | - | 🔵 faer-fork 2.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.227251 ± 0.00276381 (W 0.00220664) | 0.757386 ± 0.0322721 (W 0.0291511) | - | 🔵 faer-fork 3.33× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.189769 ± 0.000635531 (W 0.00182369) | 0.679872 ± 25.6442 (W 0.0208113) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.452371 ± 0.0150802 (W 0.00195351) | 1.78411 ± 0.103706 (W 0.0959523) | - | 🔵 faer-fork 3.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.502374 ± 0.0117067 (W 0.00586731) | 1.31491 ± 0.27411 (W 0.0875216) | - | 🔵 faer-fork 2.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.034332 ± 0.000175024 (W 6.54011e-05) | 0.120526 ± 0.00502162 (W 0.00241129) | - | 🔵 faer-fork 3.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n64/b1` | 0.153747 ± 0.00711915 (W 0.00134224) | 0.605388 ± 48.9523 (W 0.0210432) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.286733 ± 0.015329 (W 0.00288777) | 1.15518 ± 0.0263434 (W 0.110969) | - | 🔵 faer-fork 4.03× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.012461 ± 5.83981e-05 (W 4.63821e-05) | 0.01298 ± 6.05007e-05 (W 3.71255e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016191 ± 5.0797e-05 (W 4.87258e-05) | 0.012451 ± 8.18535e-05 (W 4.87678e-05) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.041682 ± 0.000185023 (W 0.000124378) | 0.063833 ± 0.00468143 (W 0.000822863) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.053983 ± 0.000279554 (W 9.71303e-05) | 0.053952 ± 0.00189948 (W 0.00848445) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.089314 ± 0.000136236 (W 0.000149504) | 0.175959 ± 0.010959 (W 0.0184609) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.135936 ± 0.000304746 (W 0.000485541) | 0.172198 ± 0.00942858 (W 0.0163738) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.268073 ± 0.00208483 (W 0.00165938) | 0.280983 ± 0.000363639 (W 0.00130065) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.278933 ± 0.00114338 (W 0.0012651) | 0.282514 ± 0.000296896 (W 0.00168457) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.863561 ± 0.0041503 (W 0.00349154) | 1.10387 ± 0.00198348 (W 0.00509898) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.870212 ± 0.00493867 (W 0.00314662) | 1.11497 ± 6.28214 (W 0.00419304) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.73626 ± 0.0163776 (W 0.00775656) | 2.7161 ± 0.0303507 (W 0.0100562) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.75594 ± 0.0567348 (W 0.0063562) | 2.69809 ± 0.00818065 (W 0.0152802) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n32/b1` | 0.075694 ± 0.000310859 (W 9.78994e-05) | 0.159908 ± 0.000505206 (W 0.00122216) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.076393 ± 0.000118196 (W 5.47476e-05) | 0.160538 ± 0.000355106 (W 0.00187788) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.262932 ± 0.00109437 (W 0.00121396) | 0.570787 ± 0.000546291 (W 0.00155962) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.263612 ± 0.000438216 (W 0.00115277) | 0.576848 ± 0.00375262 (W 0.00317804) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.599308 ± 0.00151254 (W 0.00187793) | 1.29258 ± 0.000375016 (W 0.00460357) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.601118 ± 0.00150347 (W 0.00183653) | 1.28825 ± 0.000924668 (W 0.00217174) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00172 ± 5.7735e-06 (W 1.73263e-05) | 0.00104 ± 5.7735e-06 (W 1.51658e-05) | - | 🔴 faer-fork 1.65× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00379 ± 5.7735e-06 (W 2.87263e-05) | 0.00235 ± 0 (W 1.49399e-05) | - | 🔴 faer-fork 1.61× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00626 ± 2.60832e-05 (W 1.89288e-05) | 0.003951 ± 1.12694e-05 (W 2.64991e-05) | - | 🔴 faer-fork 1.58× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.006511 ± 0.000445309 (W 2.34521e-05) | 0.00605 ± 1.52753e-05 (W 5.84748e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00538 ± 3.05505e-05 (W 2.3795e-05) | 0.00564 ± 6.08276e-06 (W 2.22329e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.022631 ± 0.000911939 (W 5.04103e-05) | 0.046113 ± 2.5697e-05 (W 5.51471e-05) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n64/b1 [api-inclusive]` | 0.018861 ± 0.000906495 (W 5.14782e-05) | 0.026831 ± 1.12694e-05 (W 1.53395e-05) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.049863 ± 0.000149901 (W 8.56522e-05) | 0.109495 ± 0.00421659 (W 0.0215089) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.043602 ± 0.00151376 (W 9.84947e-05) | 0.079433 ± 0.0164285 (W 0.00560939) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.123816 ± 0.0019678 (W 0.000965852) | 0.176478 ± 0.000104403 (W 0.00127936) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.123476 ± 0.00322568 (W 0.000469926) | 0.177618 ± 0.000349905 (W 0.00123692) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.364787 ± 0.00714639 (W 0.0026249) | 0.699223 ± 0.000964464 (W 0.00433512) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.371107 ± 0.00727829 (W 0.00362479) | 0.703103 ± 0.00579527 (W 0.0051621) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.769987 ± 0.00827567 (W 0.00356954) | 1.72237 ± 0.0273525 (W 0.0185505) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.776517 ± 0.00500046 (W 0.00490794) | 1.70815 ± 0.00441851 (W 0.0137019) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017321 ± 4.41626e-05 (W 2.86356e-05) | 0.014321 ± 2.64575e-05 (W 7.39818e-05) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.022571 ± 5.95007e-05 (W 5.72119e-05) | 0.01681 ± 6.02771e-05 (W 4.97715e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.077273 ± 0.000277471 (W 0.000124267) | 0.067824 ± 8.92767e-05 (W 0.000151123) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.091204 ± 0.000270389 (W 0.000120433) | 0.074833 ± 1.95021e-05 (W 0.00103375) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.21007 ± 0.000528488 (W 0.00118631) | 0.183279 ± 0.000217945 (W 0.00112311) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.253242 ± 0.000375011 (W 0.00159386) | 0.21375 ± 0.00221549 (W 0.00290428) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.015351 ± 5.67186e-05 (W 6.42441e-05) | 0.015491 ± 0.000176163 (W 0.000846647) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.015441 ± 4.50925e-05 (W 3.44413e-05) | 0.022341 ± 0.00305332 (W 0.0033913) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.059313 ± 0.00260742 (W 0.00187389) | 0.053623 ± 0.0362283 (W 0.0364361) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.057083 ± 0.00893822 (W 0.00162519) | 0.056692 ± 0.0210151 (W 0.0372716) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.101605 ± 0.00637147 (W 0.00186512) | 0.192229 ± 0.108114 (W 0.0682471) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.078514 ± 0.00548452 (W 0.00123104) | 0.329576 ± 0.0362509 (W 0.0483003) | - | 🔵 faer-fork 4.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.011021 ± 4.14769e-05 (W 4.80448e-05) | 0.01204 ± 0.000106068 (W 3.15167e-05) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013331 ± 6.11583e-05 (W 0.000140724) | 0.011511 ± 1.18462e-05 (W 4.90337e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.041582 ± 0.000147309 (W 0.000131833) | 0.067673 ± 0.0039382 (W 0.00983819) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.048862 ± 0.000287112 (W 0.000143806) | 0.052753 ± 1.27541 (W 0.000831122) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.093284 ± 0.000138553 (W 6.68079e-05) | 0.22982 ± 0.0548231 (W 0.0424648) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.130227 ± 0.000890741 (W 0.000851927) | 0.171578 ± 2.51073 (W 0.0222576) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.055563 ± 4.3589e-05 (W 0.000134552) | 0.176348 ± 0.00329521 (W 0.004964) | - | 🔵 faer-fork 3.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.052913 ± 0.000140475 (W 0.000128036) | 0.164808 ± 7.07959 (W 0.00340136) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.229391 ± 0.00504739 (W 0.00189701) | 0.939754 ± 0.0398639 (W 0.0400535) | - | 🔵 faer-fork 4.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.189329 ± 0.000324191 (W 0.00140833) | 0.840339 ± 0.0434373 (W 0.0348434) | - | 🔵 faer-fork 4.44× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.474713 ± 0.0164754 (W 0.00341916) | 2.24335 ± 0.0970394 (W 0.0646195) | - | 🔵 faer-fork 4.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.507864 ± 0.0162607 (W 0.0049537) | 1.92902 ± 0.330884 (W 0.109345) | - | 🔵 faer-fork 3.80× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.034251 ± 0.000127986 (W 0.000102447) | 0.144137 ± 0.00769347 (W 0.0059397) | - | 🔵 faer-fork 4.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n64/b1` | 0.175458 ± 0.00702827 (W 0.00259291) | 0.695813 ± 0.0548715 (W 0.0304342) | - | 🔵 faer-fork 3.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.327305 ± 0.023448 (W 0.0027029) | 1.67101 ± 0.0277512 (W 0.0383084) | - | 🔵 faer-fork 5.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.01254 ± 7.26659e-05 (W 3.18009e-05) | 0.013011 ± 5.80259e-05 (W 3.67872e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016301 ± 5.50757e-05 (W 4.51464e-05) | 0.012471 ± 3.10966e-05 (W 4.55005e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.041842 ± 5.7735e-05 (W 0.000155467) | 0.074683 ± 0.00528025 (W 0.0266249) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.054042 ± 9.02238e-05 (W 9.80372e-05) | 0.057882 ± 0.000743183 (W 0.00652179) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.090284 ± 0.00017088 (W 9.45844e-05) | 0.216991 ± 0.0506807 (W 0.0273101) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.141397 ± 0.000592714 (W 0.000504225) | 0.156448 ± 0.00188759 (W 0.00635869) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.268342 ± 0.000303447 (W 0.00133744) | 0.280363 ± 0.000335112 (W 0.00145404) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.278443 ± 0.000326548 (W 0.00128532) | 0.281553 ± 0.000211266 (W 0.00115198) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.872551 ± 0.00326267 (W 0.00548799) | 1.10603 ± 0.00554306 (W 0.00460672) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.877561 ± 0.00252993 (W 0.00341415) | 1.1149 ± 0.00693564 (W 0.00586657) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.7489 ± 0.0637091 (W 0.00498965) | 2.77294 ± 0.0318107 (W 0.041225) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1 [api-inclusive]` | 1.76931 ± 0.0530364 (W 0.00747113) | 2.78596 ± 0.00497968 (W 0.0199228) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n32/b1` | 0.075704 ± 0.000250267 (W 0.000118492) | 0.160058 ± 5.85947e-05 (W 0.000921686) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.075984 ± 0.000223383 (W 7.53943e-05) | 0.160597 ± 0.000255226 (W 0.0010813) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.263553 ± 0.000541554 (W 0.00117331) | 0.571327 ± 0.000627721 (W 0.00193115) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.264323 ± 0.00507515 (W 0.00126807) | 0.572337 ± 0.000642599 (W 0.00164485) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.600908 ± 0.00196537 (W 0.0028469) | 1.28658 ± 0.00104919 (W 0.00220668) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.600468 ± 0.000880171 (W 0.00191564) | 1.28792 ± 0.000839067 (W 0.00177825) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00174 ± 1e-05 (W 1.38636e-05) | 0.00105 ± 5.7735e-06 (W 2.34521e-05) | - | 🔴 faer-fork 1.66× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00385 ± 9.50438e-06 (W 2.59268e-05) | 0.00242 ± 1.12694e-05 (W 2.54951e-05) | - | 🔴 faer-fork 1.59× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00656 ± 4.50925e-05 (W 2.4682e-05) | 0.00426 ± 1.1547e-05 (W 8.56808e-05) | - | 🔴 faer-fork 1.54× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004191 ± 9.50438e-06 (W 4.32227e-05) | 0.004411 ± 2.52389e-05 (W 5.59929e-05) | 0.00604 ± 5.7735e-07 (W 2.72066e-05) | 🔵 faer-fork 1.05× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0052 ± 0.00045267 (W 1.7782e-05) | 0.005541 ± 2.95014e-05 (W 3.25761e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.015461 ± 0.000137961 (W 2.44377e-05) | 0.023311 ± 0.000195021 (W 0.000202302) | 0.022711 ± 1.1547e-05 (W 7.19389e-05) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.019941 ± 0.000771168 (W 4.97996e-05) | 0.026421 ± 0.000196554 (W 3.91178e-05) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.038271 ± 0.00133997 (W 5.90025e-05) | 0.060882 ± 0.00446697 (W 0.00760522) | 0.057773 ± 9.07377e-05 (W 0.00015441) | 🔵 faer-fork 1.59× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.045872 ± 0.00107519 (W 7.2926e-05) | 0.064583 ± 2.50242 (W 0.00423415) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.119716 ± 0.00253168 (W 0.000855467) | 0.176658 ± 0.000901943 (W 0.001329) | 0.159247 ± 0.000928648 (W 0.000506315) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.120586 ± 0.001626 (W 0.000507707) | 0.176738 ± 0.000130128 (W 0.00285322) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.350476 ± 0.000616434 (W 0.00245461) | 0.696243 ± 0.00136873 (W 0.0035069) | 0.399239 ± 0.00448656 (W 0.00355217) | 🔵 faer-fork 1.99× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.354637 ± 0.000269463 (W 0.00338574) | 0.698903 ± 0.00332666 (W 0.00298568) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.757836 ± 0.00218858 (W 0.00278326) | 1.67078 ± 0.0018803 (W 0.00517022) | 0.935705 ± 0.0193172 (W 0.00552448) | 🔵 faer-fork 2.20× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.767426 ± 0.00440659 (W 0.00301785) | 1.68503 ± 0.00159453 (W 0.0081409) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017241 ± 1e-05 (W 3.3003e-05) | 0.014121 ± 4.72582e-05 (W 2.61381e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019671 ± 0.000716682 (W 3.68008e-05) | 0.016501 ± 0.000113168 (W 6.14996e-05) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.077123 ± 0.000110183 (W 0.000106833) | 0.067434 ± 0.000242735 (W 0.00015782) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.081104 ± 0.00012741 (W 0.000177539) | 0.074493 ± 0.000578003 (W 0.00023412) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.21097 ± 0.00109153 (W 0.00200504) | 0.183659 ± 0.000432366 (W 0.00126493) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.222471 ± 7.60723e-05 (W 0.00144964) | 0.205589 ± 0.00129705 (W 0.00151866) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.01618 ± 4.68722e-05 (W 6.39234e-05) | 0.017401 ± 0.000170392 (W 0.000644228) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.016191 ± 5.55608e-05 (W 4.73519e-05) | 0.017211 ± 0.000148436 (W 0.000274281) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.030542 ± 0.00117986 (W 0.000505746) | 0.070194 ± 0.0152196 (W 0.00796355) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.031812 ± 0.000521664 (W 0.000490037) | 0.035702 ± 0.0129863 (W 0.0164846) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.058223 ± 0.000854276 (W 0.000950933) | 0.196879 ± 0.0319047 (W 0.0253477) | - | 🔵 faer-fork 3.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.058213 ± 0.00102059 (W 0.000487524) | 0.177539 ± 0.0144953 (W 0.0404201) | - | 🔵 faer-fork 3.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.00849 ± 7.64875e-05 (W 3.75406e-05) | 0.0092 ± 7.59408e-05 (W 5.2899e-05) | 0.016161 ± 9.84886e-05 (W 7.81614e-05) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.90× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010511 ± 4.93288e-05 (W 0.00016557) | 0.011371 ± 4.10731e-05 (W 5.80629e-05) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.033852 ± 0.000280416 (W 0.000234361) | 0.040082 ± 0.00061825 (W 0.00646025) | 0.050473 ± 0.000160001 (W 0.00136968) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.038812 ± 6.50641e-05 (W 0.000209476) | 0.044352 ± 0.00861177 (W 0.000252708) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.077814 ± 0.000226053 (W 0.00112134) | 0.105185 ± 0.0110168 (W 0.00851218) | 0.106485 ± 0.000582866 (W 0.00150099) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.100125 ± 0.000183576 (W 0.000201015) | 0.130336 ± 0.0115253 (W 0.00641579) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.046872 ± 0.00013 (W 0.000159732) | 0.099535 ± 0.00526552 (W 0.00420083) | 0.035572 ± 0.000563693 (W 0.000163961) | 🔵 faer-fork 2.12× faster than `faer-upstream` | 🔴 faer-fork 1.32× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.049802 ± 0.000137041 (W 9.89707e-05) | 0.100405 ± 0.00403066 (W 0.00220461) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.172118 ± 0.000247219 (W 0.00180796) | 0.529175 ± 0.016839 (W 0.0244682) | 0.138717 ± 0.00078886 (W 0.00206811) | 🔵 faer-fork 3.07× faster than `faer-upstream` | 🔴 faer-fork 1.24× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.179898 ± 0.000256145 (W 0.00137793) | 0.498874 ± 0.0185656 (W 0.00820366) | - | 🔵 faer-fork 2.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.375938 ± 0.000657368 (W 0.00356078) | 1.06106 ± 0.0784513 (W 0.043998) | 0.331955 ± 0.00180271 (W 0.00228142) | 🔵 faer-fork 2.82× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.454342 ± 0.00335106 (W 0.00420995) | 1.34639 ± 0.109232 (W 0.0616553) | - | 🔵 faer-fork 2.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.026201 ± 3.60555e-05 (W 8.07546e-05) | 0.077644 ± 0.00161461 (W 0.00166317) | 0.017781 ± 5.45619e-05 (W 7.91833e-05) | 🔵 faer-fork 2.96× faster than `faer-upstream` | 🔴 faer-fork 1.47× slower than `lapack-mkl` |  |
| `qr_factor/n64/b1` | 0.109615 ± 3.05014e-05 (W 0.000116641) | 0.371468 ± 0.0193217 (W 0.00822413) | 0.079364 ± 0.000171931 (W 7.97697e-05) | 🔵 faer-fork 3.39× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.236771 ± 0.000633588 (W 0.00259487) | 0.904473 ± 0.0733059 (W 0.0170549) | 0.235361 ± 0.000667108 (W 0.00225999) | 🔵 faer-fork 3.82× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n32/b1` | 0.010871 ± 3.5171e-05 (W 4.92879e-05) | 0.010081 ± 4.6188e-05 (W 3.41365e-05) | 0.016391 ± 0.000171553 (W 6.14101e-05) | 🔴 faer-fork 1.08× slower than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01325 ± 9.65419e-05 (W 7.83601e-05) | 0.01215 ± 6.60328e-05 (W 4.62418e-05) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.038192 ± 0.000110077 (W 0.000100625) | 0.052253 ± 0.00692107 (W 0.00417956) | 0.049702 ± 0.000270389 (W 7.17168e-05) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.043372 ± 0.000130767 (W 0.000107098) | 0.079304 ± 1.24639 (W 0.00929738) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.084524 ± 0.000386264 (W 9.98499e-05) | 0.109925 ± 0.0137687 (W 0.0156754) | 0.106635 ± 0.000235867 (W 0.000172858) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.111035 ± 0.000630026 (W 0.000238202) | 0.140126 ± 0.0130166 (W 0.00992069) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.271993 ± 0.00029023 (W 0.00156019) | 0.284524 ± 0.000526505 (W 0.00104379) | 0.274473 ± 0.00185831 (W 0.00236204) | · | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.282753 ± 0.000295173 (W 0.00168094) | 0.286513 ± 0.0010922 (W 0.00285359) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.873192 ± 0.00141173 (W 0.00359052) | 1.1527 ± 0.00264953 (W 0.00255998) | 1.1182 ± 0.0062711 (W 0.00402824) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.873631 ± 0.00467432 (W 0.00431091) | 1.1584 ± 0.000557199 (W 0.00370593) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.74432 ± 0.00211436 (W 0.00978913) | 2.65189 ± 0.00635985 (W 0.0131201) | 2.2211 ± 0.0054026 (W 0.00841271) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.76818 ± 0.0113867 (W 0.00393934) | 2.6747 ± 0.00915373 (W 4.12726) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075293 ± 0.00012897 (W 8.51369e-05) | 0.162108 ± 0.000193236 (W 0.00109569) | 0.073674 ± 0.000485627 (W 0.000125484) | 🔵 faer-fork 2.15× faster than `faer-upstream` | · |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.075874 ± 0.000335311 (W 7.80083e-05) | 0.162448 ± 0.000184481 (W 0.00129903) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.266003 ± 0.000202007 (W 0.00133117) | 0.592448 ± 0.00183328 (W 0.00150051) | 0.354187 ± 0.00123753 (W 0.00279811) | 🔵 faer-fork 2.23× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.266063 ± 0.000195021 (W 0.00113529) | 0.591788 ± 0.00270165 (W 0.00232219) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.607859 ± 0.00152133 (W 0.001839) | 1.30351 ± 0.000311716 (W 0.00175902) | 0.84034 ± 0.00283897 (W 0.00321465) | 🔵 faer-fork 2.14× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.609309 ± 0.00161376 (W 0.0025999) | 1.30765 ± 0.00596058 (W 0.00149925) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00161 ± 1.1547e-05 (W 2.86356e-05) | 0.00102 ± 1e-05 (W 1.73205e-05) | - | 🔴 faer-fork 1.58× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003521 ± 1.18462e-05 (W 3.48023e-05) | 0.00231 ± 1.52753e-05 (W 1.87136e-05) | - | 🔴 faer-fork 1.52× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0061 ± 5.50757e-06 (W 2.93462e-05) | 0.0042 ± 5.7735e-07 (W 2.18906e-05) | - | 🔴 faer-fork 1.45× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00422 ± 3.51188e-05 (W 4.70872e-05) | 0.004421 ± 1.51767e-05 (W 0.000110029) | 0.006011 ± 2.28108e-05 (W 1.52381e-05) | · | 🔵 faer-fork 1.42× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00455 ± 0.000384231 (W 2.39625e-05) | 0.005541 ± 1.09697e-05 (W 3.08107e-05) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.016981 ± 6.08276e-05 (W 4.02517e-05) | 0.023311 ± 0.000121655 (W 0.000191367) | 0.022581 ± 2.3094e-05 (W 7.98749e-05) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.018601 ± 0.000961457 (W 3.11448e-05) | 0.026411 ± 0.00293026 (W 5.54995e-05) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.039492 ± 0.00138622 (W 5.90449e-05) | 0.064383 ± 0.0192653 (W 0.0139429) | 0.057803 ± 0.000192873 (W 0.000239097) | 🔵 faer-fork 1.63× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.045712 ± 0.00138074 (W 7.34847e-05) | 0.077783 ± 0.00828718 (W 0.00784871) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.122686 ± 0.00185246 (W 0.000427346) | 0.176578 ± 0.000220303 (W 0.00128187) | 0.169278 ± 0.000426654 (W 0.00158001) | 🔵 faer-fork 1.44× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.125466 ± 0.00132085 (W 0.000631324) | 0.177838 ± 8.75233e-05 (W 0.00196513) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.349957 ± 0.0029065 (W 0.00263721) | 0.697653 ± 0.00395755 (W 0.00371335) | 0.40762 ± 0.00415724 (W 0.00564287) | 🔵 faer-fork 1.99× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.384448 ± 0.0207443 (W 0.00351919) | 0.697583 ± 0.00248373 (W 0.00405029) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 0.757336 ± 0.0064993 (W 0.00526116) | 1.68626 ± 0.0126958 (W 0.00857647) | 1.10259 ± 0.0393119 (W 0.00766493) | 🔵 faer-fork 2.23× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` | `!` between-start range over 5% |
| `eigh/n96/b1 [api-inclusive]` | 0.764026 ± 0.00239972 (W 0.00317673) | 1.69141 ± 0.00936654 (W 0.00703097) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017221 ± 0.000100135 (W 3.40323e-05) | 0.0141 ± 1.0504e-05 (W 2.95347e-05) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019531 ± 8.96289e-05 (W 4.91935e-05) | 0.016591 ± 0.000111805 (W 4.73202e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.076024 ± 4.63717e-05 (W 0.00105834) | 0.067863 ± 0.000135031 (W 0.000139104) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.081394 ± 0.000280951 (W 0.000279101) | 0.073563 ± 0.00014405 (W 0.000216463) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.199919 ± 0.000141539 (W 0.00121235) | 0.192739 ± 0.000248462 (W 0.00222213) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.223291 ± 0.000839663 (W 0.00219858) | 0.20519 ± 0.00100248 (W 0.00142473) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.016341 ± 6.55769e-05 (W 5.56884e-05) | 0.017081 ± 0.000507327 (W 0.000531876) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.016311 ± 4.99032e-05 (W 5.61712e-05) | 0.018031 ± 0.000150921 (W 0.000180198) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.038742 ± 0.00323596 (W 0.000321979) | 0.050773 ± 0.0286619 (W 0.0313519) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.034282 ± 0.00214994 (W 0.000827492) | 0.108195 ± 0.00948937 (W 0.0313135) | - | 🔵 faer-fork 3.16× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.064753 ± 0.00851284 (W 0.00135399) | 0.148007 ± 0.0932734 (W 0.0113383) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.060453 ± 0.00571777 (W 0.00150064) | 0.352967 ± 0.0303184 (W 0.0454598) | - | 🔵 faer-fork 5.84× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.00863 ± 0.000189597 (W 9.08036e-05) | 0.009221 ± 3.10966e-05 (W 6.35161e-05) | 0.01606 ± 0.000234052 (W 0.000125093) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.86× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.01066 ± 0.000160416 (W 9.51804e-05) | 0.01134 ± 5.45619e-05 (W 4.52471e-05) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034071 ± 0.0003095 (W 0.000293214) | 0.055962 ± 0.035744 (W 0.0111964) | 0.050442 ± 0.000320516 (W 0.000156248) | · | 🔵 faer-fork 1.48× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.038721 ± 0.000281727 (W 0.000228539) | 2.23168 ± 1.26674 (W 0.0106895) | - | 🔵 faer-fork 57.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.077743 ± 0.000331376 (W 0.000192767) | 0.108645 ± 0.0357311 (W 0.0369178) | 0.106725 ± 0.000588246 (W 0.00179582) | · | 🔵 faer-fork 1.37× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.101695 ± 0.00022053 (W 0.000472326) | 0.152427 ± 4.54153 (W 0.0175425) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.046912 ± 0.000218953 (W 0.000114439) | 0.125616 ± 0.00517198 (W 0.00423319) | 0.035542 ± 0.000655912 (W 0.000270415) | 🔵 faer-fork 2.68× faster than `faer-upstream` | 🔴 faer-fork 1.32× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.049912 ± 0.000142339 (W 0.000136562) | 0.121446 ± 0.00749833 (W 0.00346153) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.172208 ± 0.00025482 (W 0.00193012) | 0.64871 ± 0.022849 (W 0.0328013) | 0.134487 ± 0.000990399 (W 0.000852106) | 🔵 faer-fork 3.77× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.179139 ± 0.00026704 (W 0.00150796) | 0.586898 ± 0.0585603 (W 0.0164055) | - | 🔵 faer-fork 3.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.377218 ± 0.00873339 (W 0.00416786) | 1.56069 ± 0.0349321 (W 0.0414963) | 0.326875 ± 0.00136608 (W 0.0029458) | 🔵 faer-fork 4.14× faster than `faer-upstream` | 🔴 faer-fork 1.15× slower than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.461351 ± 0.0174811 (W 0.00518981) | 1.40384 ± 0.250807 (W 0.05649) | - | 🔵 faer-fork 3.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.026052 ± 0.000248194 (W 7.52609e-05) | 0.105405 ± 0.00593785 (W 0.00248465) | 0.0177 ± 9.51332e-05 (W 5.24423e-05) | 🔵 faer-fork 4.05× faster than `faer-upstream` | 🔴 faer-fork 1.47× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/n64/b1` | 0.110125 ± 0.000248355 (W 0.000184098) | 0.508634 ± 0.048497 (W 0.0111205) | 0.079544 ± 8e-05 (W 7.58696e-05) | 🔵 faer-fork 4.62× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n96/b1` | 0.237892 ± 0.00115702 (W 0.00279426) | 1.12356 ± 0.0909667 (W 0.0745782) | 0.233881 ± 0.000626844 (W 0.00184206) | 🔵 faer-fork 4.72× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n32/b1` | 0.010951 ± 0.000170177 (W 4.36612e-05) | 0.01012 ± 0.000114893 (W 4.27586e-05) | 0.016261 ± 5.50757e-05 (W 7.26168e-05) | 🔴 faer-fork 1.08× slower than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013 ± 4.56107e-05 (W 3.50457e-05) | 0.012081 ± 8.06866e-05 (W 3.37535e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.038251 ± 0.0001945 (W 6.53452e-05) | 0.052772 ± 0.0034221 (W 0.00107249) | 0.050272 ± 9.58662e-05 (W 8.84828e-05) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.043282 ± 0.000263122 (W 0.000158808) | 0.063543 ± 0.00794827 (W 0.00914027) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.084354 ± 0.000283608 (W 6.76757e-05) | 0.103525 ± 0.000946351 (W 0.0260287) | 0.106625 ± 0.000399291 (W 0.00133865) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.111785 ± 0.000357631 (W 0.000408809) | 0.165528 ± 0.0146436 (W 0.025957) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.272353 ± 0.000534509 (W 0.00131167) | 0.284704 ± 0.000348281 (W 0.00144082) | 0.293954 ± 0.00319427 (W 0.00287242) | · | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.282774 ± 0.00501402 (W 0.00137896) | 0.285224 ± 0.00084664 (W 0.00114068) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.871201 ± 0.00501964 (W 0.00614611) | 1.15198 ± 0.00333736 (W 0.0041398) | 1.18879 ± 0.0029201 (W 0.00447246) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.873651 ± 0.00693263 (W 0.00387199) | 1.15959 ± 6.25523 (W 0.00414294) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.75104 ± 0.0143453 (W 0.00806205) | 2.76068 ± 0.0177001 (W 0.0530894) | 2.29472 ± 0.00329547 (W 0.00610589) | 🔵 faer-fork 1.58× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.77066 ± 0.00956331 (W 0.00668658) | 2.71143 ± 0.00323637 (W 0.0109658) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075493 ± 0.000141539 (W 6.29389e-05) | 0.161908 ± 0.000207052 (W 0.000950082) | 0.073624 ± 0.000386954 (W 6.16628e-05) | 🔵 faer-fork 2.14× faster than `faer-upstream` | · |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.075914 ± 0.000152753 (W 7.59296e-05) | 0.162137 ± 0.000509372 (W 0.00110295) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.266682 ± 0.000675419 (W 0.00152216) | 0.591368 ± 0.000513939 (W 0.0016324) | 0.361248 ± 0.00131061 (W 0.00246716) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.266773 ± 0.000529066 (W 0.00113344) | 0.597478 ± 0.000531451 (W 0.00133977) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.610509 ± 0.000858604 (W 0.00259916) | 1.29493 ± 0.000937922 (W 0.00209216) | 0.8499 ± 0.00555989 (W 0.0026622) | 🔵 faer-fork 2.12× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.611059 ± 0.00126645 (W 0.00152324) | 1.29572 ± 0.00447429 (W 0.00310773) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00161 ± 5.7735e-06 (W 1.94936e-05) | 0.001 ± 5.7735e-06 (W 1.51063e-05) | - | 🔴 faer-fork 1.61× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00348 ± 5.7735e-06 (W 2.0693e-05) | 0.00222 ± 0 (W 2.49439e-05) | - | 🔴 faer-fork 1.57× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00554 ± 1.76163e-05 (W 3.38408e-05) | 0.00362 ± 5.7735e-06 (W 1.89789e-05) | - | 🔴 faer-fork 1.53× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00419 ± 0.000103923 (W 4.90836e-05) | 0.00443 ± 1.09697e-05 (W 6.05805e-05) | 0.00606 ± 1.51767e-05 (W 1.33116e-05) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00456 ± 0.000404846 (W 3.19562e-05) | 0.00555 ± 5.7735e-07 (W 2.2764e-05) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01696 ± 0.000952994 (W 3.47606e-05) | 0.023321 ± 1.52753e-05 (W 0.000193598) | 0.022701 ± 5e-05 (W 6.24756e-05) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.020541 ± 7.0946e-05 (W 6.37966e-05) | 0.026401 ± 4.3589e-05 (W 3.08756e-05) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.040032 ± 8.73689e-05 (W 8.35464e-05) | 0.073574 ± 0.0114574 (W 0.0130755) | 0.057543 ± 6.97161e-05 (W 0.00117894) | 🔵 faer-fork 1.84× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.044042 ± 0.00122964 (W 5.27371e-05) | 0.072963 ± 0.00371872 (W 0.00158368) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.121376 ± 0.000875723 (W 0.000622972) | 0.177029 ± 0.000361138 (W 0.00126761) | 0.189449 ± 0.00143503 (W 0.00285265) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.124586 ± 0.00236238 (W 0.00086662) | 0.177828 ± 0.00032023 (W 0.00185652) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.371247 ± 0.00448913 (W 0.00325738) | 0.697434 ± 0.00346968 (W 0.00427962) | 0.587978 ± 0.0217776 (W 0.00891503) | 🔵 faer-fork 1.88× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` | `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.369517 ± 0.011068 (W 0.00377386) | 0.704293 ± 2.52808 (W 0.00552276) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 0.763307 ± 0.00469595 (W 0.00423407) | 1.74557 ± 0.0260352 (W 0.018259) | 1.2802 ± 0.0144819 (W 0.0220819) | 🔵 faer-fork 2.29× faster than `faer-upstream` | 🔵 faer-fork 1.68× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.777557 ± 0.0041926 (W 0.0057693) | 1.7142 ± 0.00211331 (W 0.00768191) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017421 ± 6.61589e-05 (W 2.05475e-05) | 0.014121 ± 2.08167e-05 (W 4.26533e-05) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019651 ± 4.72582e-05 (W 5.12835e-05) | 0.016491 ± 4.16333e-05 (W 5.61445e-05) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.077393 ± 0.000661007 (W 7.83664e-05) | 0.067733 ± 0.00015308 (W 8.7116e-05) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.081264 ± 0.0026727 (W 0.000171316) | 0.073634 ± 0.000429221 (W 0.000157773) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.21018 ± 0.000645161 (W 0.0011806) | 0.183189 ± 0.000448144 (W 0.00106376) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.227801 ± 0.00145021 (W 0.00213322) | 0.20924 ± 0.000913583 (W 0.00156886) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.016221 ± 7.02875e-05 (W 3.61289e-05) | 0.017431 ± 0.000468081 (W 0.00117188) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.016201 ± 2.64575e-05 (W 4.16437e-05) | 0.024581 ± 0.00379812 (W 0.00363481) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.052703 ± 0.00763566 (W 0.0018986) | 0.064623 ± 0.0313656 (W 0.0224061) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.040232 ± 0.00493651 (W 0.00107292) | 0.094545 ± 0.025931 (W 0.0301718) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.071374 ± 0.0162891 (W 0.00168044) | 0.180428 ± 0.125377 (W 0.0244067) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.057473 ± 0.00732835 (W 0.00160675) | 0.311585 ± 0.0401691 (W 0.0508933) | - | 🔵 faer-fork 5.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.008561 ± 4.00042e-05 (W 7.48218e-05) | 0.009311 ± 0.001317 (W 5.49482e-05) | 0.01608 ± 0.000276695 (W 0.000131386) | · | 🔵 faer-fork 1.88× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.01055 ± 4.14769e-05 (W 0.00014951) | 0.01132 ± 5.54106e-05 (W 4.14644e-05) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.033592 ± 0.000156387 (W 0.000273705) | 0.055643 ± 0.000643413 (W 0.0206822) | 0.049912 ± 0.000765583 (W 0.000174889) | 🔵 faer-fork 1.66× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.038762 ± 0.000258908 (W 0.000149907) | 0.051243 ± 1.29995 (W 0.0126475) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.077704 ± 0.000105642 (W 0.000100932) | 0.178138 ± 0.0584185 (W 0.021844) | 0.106005 ± 0.000499032 (W 0.00136559) | 🔵 faer-fork 2.29× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.105895 ± 0.00248752 (W 0.000362166) | 0.168688 ± 2.56385 (W 0.008768) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.046872 ± 0.000328075 (W 0.000245092) | 0.160407 ± 0.00292747 (W 0.0053324) | 0.035392 ± 0.000528804 (W 0.000154344) | 🔵 faer-fork 3.42× faster than `faer-upstream` | 🔴 faer-fork 1.32× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.049973 ± 0.000484985 (W 0.000126211) | 0.151527 ± 0.00933215 (W 0.00492927) | - | 🔵 faer-fork 3.03× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.171958 ± 0.000285832 (W 0.00138887) | 0.877552 ± 0.0464991 (W 0.030491) | 0.137646 ± 0.000355655 (W 0.00141627) | 🔵 faer-fork 5.10× faster than `faer-upstream` | 🔴 faer-fork 1.25× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.179368 ± 0.000909863 (W 0.00162803) | 0.770306 ± 0.125316 (W 0.0220972) | - | 🔵 faer-fork 4.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.386878 ± 0.00666887 (W 0.00227946) | 2.05815 ± 0.0553847 (W 0.0859859) | 0.320406 ± 0.00386759 (W 0.00197749) | 🔵 faer-fork 5.32× faster than `faer-upstream` | 🔴 faer-fork 1.21× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.470592 ± 0.017731 (W 0.00417737) | 2.10907 ± 0.231851 (W 0.109271) | - | 🔵 faer-fork 4.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.026221 ± 0.000185203 (W 0.000104097) | 0.128446 ± 0.00372828 (W 0.00281666) | 0.01778 ± 0.000160376 (W 0.000109308) | 🔵 faer-fork 4.90× faster than `faer-upstream` | 🔴 faer-fork 1.47× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/n64/b1` | 0.109476 ± 0.000104165 (W 0.000201795) | 0.727814 ± 0.0811893 (W 0.0321029) | 0.079514 ± 0.000247972 (W 5.95819e-05) | 🔵 faer-fork 6.65× faster than `faer-upstream` | 🔴 faer-fork 1.38× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n96/b1` | 0.239901 ± 0.0089354 (W 0.00164132) | 1.23499 ± 0.0231718 (W 0.0310364) | 0.233901 ± 0.00023567 (W 0.00165777) | 🔵 faer-fork 5.15× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `solve/n32/b1` | 0.010841 ± 6.86319e-05 (W 6.43374e-05) | 0.009951 ± 2.12211e-05 (W 4.14765e-05) | 0.016201 ± 0.000172975 (W 8.161e-05) | 🔴 faer-fork 1.09× slower than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01328 ± 0.000144627 (W 5.72564e-05) | 0.012091 ± 5.50757e-05 (W 3.42389e-05) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.038721 ± 0.000216496 (W 0.000125817) | 0.058413 ± 0.0315193 (W 0.0178519) | 0.049633 ± 0.000340701 (W 6.23402e-05) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.043562 ± 0.0001861 (W 0.000170382) | 0.058393 ± 1.25176 (W 0.00535941) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.084694 ± 0.000219317 (W 8.23408e-05) | 0.184158 ± 0.0321201 (W 0.0439153) | 0.106425 ± 0.0003 (W 0.00152289) | 🔵 faer-fork 2.17× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.116786 ± 0.000981443 (W 0.000538207) | 0.170918 ± 0.0129959 (W 0.00488698) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.271923 ± 0.000750489 (W 0.00138819) | 0.283733 ± 0.000727235 (W 0.00116342) | 0.311964 ± 0.00177636 (W 0.00373344) | · | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.282243 ± 0.000330807 (W 0.00192507) | 0.285894 ± 0.00197481 (W 0.00172943) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.879962 ± 0.00259662 (W 0.00516878) | 1.14837 ± 0.00490981 (W 0.00532669) | 1.33507 ± 0.0629265 (W 0.00663121) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.884962 ± 0.00514182 (W 0.00430831) | 1.16728 ± 0.00772378 (W 0.0116929) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.79428 ± 0.105901 (W 0.00722927) | 2.78546 ± 0.01623 (W 0.0314232) | 2.6248 ± 0.144093 (W 0.0128003) | 🔵 faer-fork 1.55× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n96/b1 [api-inclusive]` | 1.85784 ± 0.157148 (W 0.0127742) | 2.80078 ± 22.7672 (W 0.0402323) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n32/b1` | 0.075553 ± 1.18462e-05 (W 7.76099e-05) | 0.161828 ± 0.000133717 (W 0.00114914) | 0.073824 ± 0.000193236 (W 0.000188104) | 🔵 faer-fork 2.14× faster than `faer-upstream` | · |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.076094 ± 0.000557704 (W 0.00011683) | 0.162257 ± 2.82902e-05 (W 0.00116564) | - | 🔵 faer-fork 2.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.265132 ± 0.00109722 (W 0.00121958) | 0.591128 ± 0.000947752 (W 0.00157609) | 0.362887 ± 0.00100992 (W 0.00199653) | 🔵 faer-fork 2.23× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.266242 ± 0.001198 (W 0.00107311) | 0.591978 ± 0.000985613 (W 0.00173799) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.609469 ± 0.00101619 (W 0.00166886) | 1.29542 ± 0.00428323 (W 0.00225325) | 0.85565 ± 0.00178752 (W 0.00184515) | 🔵 faer-fork 2.13× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.608958 ± 0.00174887 (W 0.00509782) | 1.29251 ± 0.00368831 (W 0.00115411) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00162 ± 1e-05 (W 1.7782e-05) | 0.00102 ± 5.7735e-06 (W 7.39594e-05) | - | 🔴 faer-fork 1.59× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00361 ± 1.1547e-05 (W 2.94958e-05) | 0.00234 ± 5.7735e-06 (W 7.75887e-05) | - | 🔴 faer-fork 1.54× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005831 ± 5.50757e-06 (W 3.35738e-05) | 0.00391 ± 3e-05 (W 9.13849e-05) | - | 🔴 faer-fork 1.49× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00966 ± 0.00073971 (W 4.94894e-05) | 0.012461 ± 5.7735e-07 (W 1.9321e-05) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.008961 ± 0.000661085 (W 3.87595e-05) | 0.007751 ± 0.000549688 (W 3.32611e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.045782 ± 0.000122882 (W 0.000108735) | 0.096374 ± 0.00256026 (W 0.000371996) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.046532 ± 0.00127079 (W 7.16819e-05) | 0.043962 ± 0.00315583 (W 0.000104528) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1` | 0.123116 ± 0.00144057 (W 0.00137555) | 0.167908 ± 0.0064554 (W 0.00416825) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.122715 ± 0.000240327 (W 0.000148065) | 0.134036 ± 1.24536 (W 0.00458355) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.160047 ± 0.00207416 (W 0.000998948) | 0.20944 ± 0.000327872 (W 0.00136014) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.163078 ± 0.00173702 (W 0.00110774) | 0.219191 ± 0.000479531 (W 0.0012365) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.521574 ± 0.00261736 (W 0.00248591) | 0.869522 ± 0.0014795 (W 0.00429834) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.532785 ± 0.00297457 (W 0.00287282) | 0.900402 ± 0.00175284 (W 0.00280759) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.1015 ± 0.00355717 (W 0.00222231) | 2.10908 ± 0.00837609 (W 0.0116322) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.12607 ± 0.00499778 (W 0.00510176) | 2.18804 ± 0.00738531 (W 0.00730868) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.025501 ± 3.60555e-05 (W 5.49927e-05) | 0.020321 ± 3.05505e-05 (W 3.96232e-05) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.032032 ± 7.83603e-05 (W 9.64899e-05) | 0.023441 ± 0.000200333 (W 0.000102372) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.143977 ± 5.32948e-05 (W 0.00163622) | 0.128136 ± 8.02517e-05 (W 0.00099081) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.162988 ± 0.000141539 (W 0.00132424) | 0.140067 ± 0.000155027 (W 0.0010457) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.383928 ± 0.00056003 (W 0.00212282) | 0.390909 ± 0.0001552 (W 0.00157965) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.500404 ± 0.00176551 (W 0.00272589) | 0.404889 ± 0.000162318 (W 0.00250237) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.022922 ± 0.000331598 (W 0.000314457) | 0.042712 ± 0.000330465 (W 0.00143237) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.020191 ± 0.000178979 (W 0.000425065) | 0.041642 ± 0.00125001 (W 0.00114426) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.063363 ± 0.00145193 (W 0.000859692) | 0.126256 ± 0.0360001 (W 0.0271628) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.063833 ± 0.00234282 (W 0.00116159) | 0.154137 ± 0.0223294 (W 0.0114325) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.143967 ± 0.00559558 (W 0.00136194) | 0.264222 ± 0.0151345 (W 0.0199307) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.145297 ± 0.00448119 (W 0.00270601) | 0.281163 ± 0.0226718 (W 0.0398042) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.019261 ± 3.4641e-05 (W 4.32435e-05) | 0.016551 ± 5.50757e-05 (W 3.73256e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.023502 ± 5.87963e-05 (W 7.93108e-05) | 0.017101 ± 1e-05 (W 3.17364e-05) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1` | 0.078644 ± 0.000132035 (W 0.000248166) | 0.083184 ± 0.00021362 (W 0.000163172) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.093794 ± 0.000816844 (W 0.000175161) | 0.078813 ± 0.000944283 (W 0.00119) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.19853 ± 0.00487892 (W 0.00682677) | 0.218761 ± 0.00720623 (W 0.00322345) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.340286 ± 0.00813262 (W 0.00760487) | 0.245092 ± 0.00108282 (W 0.0114411) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.112126 ± 0.00186251 (W 0.00131551) | 0.169628 ± 0.00407287 (W 0.00586262) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.108955 ± 0.00153939 (W 0.000277376) | 0.154907 ± 0.00608087 (W 0.00280178) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.368637 ± 0.00212821 (W 0.003496) | 0.855911 ± 0.0552823 (W 0.0267344) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.41639 ± 0.00535414 (W 0.002622) | 0.781857 ± 0.0310025 (W 0.00801572) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.663601 ± 0.00445299 (W 0.00421806) | 1.86449 ± 0.102473 (W 0.0350854) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.85059 ± 0.0110118 (W 0.00447948) | 1.6178 ± 0.11875 (W 0.067138) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.059273 ± 0.000102632 (W 0.000176395) | 0.114115 ± 7.28479 (W 0.00238978) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n64/b1` | 0.230401 ± 0.0019236 (W 0.0019625) | 0.604578 ± 0.0488342 (W 0.0326694) | - | 🔵 faer-fork 2.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.433671 ± 0.00165954 (W 0.00398411) | 1.20923 ± 0.0976389 (W 0.043357) | - | 🔵 faer-fork 2.79× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.021811 ± 2.08167e-05 (W 4.06202e-05) | 0.01891 ± 0.000205257 (W 3.17836e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1 [api-inclusive]` | 0.027571 ± 9.53939e-05 (W 4.27586e-05) | 0.019231 ± 8.5049e-05 (W 4.14729e-05) | - | 🔴 faer-fork 1.43× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1` | 0.080464 ± 0.000342394 (W 0.000190604) | 0.087164 ± 0.000873549 (W 0.00116986) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.103135 ± 0.000697424 (W 0.000181594) | 0.082584 ± 0.00027 (W 0.0024926) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.182709 ± 0.00849786 (W 0.00597569) | 0.232362 ± 0.00392671 (W 0.0137731) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.363057 ± 0.00600633 (W 0.00711467) | 0.249742 ± 1.25453 (W 0.00981878) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.355817 ± 0.00231759 (W 0.00168452) | 0.367218 ± 0.000128829 (W 0.00215201) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.348957 ± 0.00115071 (W 0.00122351) | 0.369678 ± 0.000124231 (W 0.00136761) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.2222 ± 0.00748888 (W 0.00386264) | 1.53182 ± 0.00762369 (W 0.0121794) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.23275 ± 0.00464296 (W 0.00380443) | 1.55365 ± 0.00591286 (W 0.00841656) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.43183 ± 0.000646452 (W 0.0072231) | 3.81753 ± 0.0203985 (W 0.0187814) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.4697 ± 0.00637378 (W 0.00422575) | 3.87765 ± 0.0107005 (W 0.0172608) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.098274 ± 0.00016547 (W 0.000123702) | 0.188339 ± 9.86577e-05 (W 0.00148842) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.099205 ± 0.000238132 (W 0.000106823) | 0.189599 ± 0.000378197 (W 0.00315109) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.436401 ± 0.000410837 (W 0.00150557) | 0.781917 ± 0.00476663 (W 0.00174603) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.437771 ± 0.00053 (W 0.00169644) | 0.784547 ± 0.00168482 (W 0.00140263) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 1.06624 ± 0.00327711 (W 0.00204826) | 1.87595 ± 0.00226024 (W 0.00153824) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 1.06612 ± 0.00323775 (W 0.00588505) | 1.87747 ± 0.00094504 (W 0.00161472) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00273 ± 0 (W 1.55949e-05) | 0.00188 ± 0 (W 7.71635e-05) | - | 🔴 faer-fork 1.45× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00649 ± 1e-05 (W 2.54814e-05) | 0.00468 ± 5.7735e-07 (W 3.37668e-05) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.01132 ± 2.09841e-05 (W 3.15167e-05) | 0.00845 ± 1.18462e-05 (W 5.9492e-05) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00856 ± 1.52753e-05 (W 4.37756e-05) | 0.012491 ± 2.3094e-05 (W 9.35804e-05) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00782 ± 0.000643765 (W 3.68958e-05) | 0.007711 ± 0.000591735 (W 3.68416e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.047753 ± 0.00130849 (W 0.000121993) | 0.096954 ± 0.000142163 (W 0.00224247) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.046782 ± 0.0011767 (W 0.000110164) | 0.044012 ± 0.000222336 (W 0.00010036) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.120625 ± 0.00257719 (W 0.000303364) | 0.178999 ± 1.25193 (W 0.00603749) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.123006 ± 0.00215808 (W 0.000101947) | 2.31404 ± 2.77044 (W 0.00775783) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.161808 ± 0.00164476 (W 0.001942) | 0.20938 ± 0.000342394 (W 0.00150964) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.164908 ± 0.00316071 (W 0.00243237) | 0.21932 ± 0.000289885 (W 0.00223189) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.526325 ± 0.00282502 (W 0.00300086) | 0.871141 ± 0.00305053 (W 0.00409611) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.538956 ± 0.00197289 (W 0.00293487) | 0.900153 ± 0.00195469 (W 0.00616844) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.11979 ± 0.00813541 (W 0.00338564) | 2.12022 ± 0.00365857 (W 0.00731213) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.13807 ± 0.00807668 (W 0.00895148) | 2.20485 ± 0.00895545 (W 8.88761) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.025391 ± 4e-05 (W 3.05647e-05) | 0.020461 ± 2e-05 (W 4.15933e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.032071 ± 3.50761e-05 (W 6.61763e-05) | 0.023321 ± 6.65833e-05 (W 8.16701e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.141256 ± 0.00856496 (W 0.000152342) | 0.132937 ± 0.000347635 (W 0.00185597) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.162888 ± 0.000637992 (W 0.00118919) | 0.144637 ± 0.00282684 (W 0.00170314) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.386848 ± 0.00151147 (W 0.00234488) | 0.391159 ± 0.0013775 (W 0.00156482) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.503434 ± 0.00107126 (W 0.00513629) | 0.40529 ± 0.00113618 (W 0.00148059) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.024021 ± 0.000932094 (W 0.000306056) | 0.041332 ± 0.000517333 (W 0.00125534) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.020411 ± 0.000419682 (W 0.000374139) | 0.042982 ± 0.00136258 (W 0.00278643) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.070303 ± 0.00209572 (W 0.000888324) | 0.114186 ± 0.0647176 (W 0.0803604) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.062373 ± 0.00100835 (W 0.00186631) | 0.239352 ± 0.0675038 (W 0.0637921) | - | 🔵 faer-fork 3.84× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.149587 ± 0.00231779 (W 0.00192373) | 0.242131 ± 0.0662669 (W 0.0890397) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.144577 ± 0.00306337 (W 0.00182061) | 0.517065 ± 0.17246 (W 0.161042) | - | 🔵 faer-fork 3.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.019261 ± 6.05833e-05 (W 9.19239e-05) | 0.016431 ± 3.55949e-05 (W 3.13241e-05) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.023621 ± 8.41843e-05 (W 0.000105139) | 0.017081 ± 3.45977e-05 (W 4.23934e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1` | 0.078883 ± 0.000613474 (W 0.000238369) | 0.083324 ± 0.00316443 (W 0.00120832) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.093705 ± 0.000144777 (W 0.000158093) | 0.078684 ± 0.000368556 (W 0.00132726) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.2012 ± 0.00713168 (W 0.00279264) | 0.228061 ± 0.00068352 (W 0.00299122) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.343717 ± 0.0093329 (W 0.00611398) | 0.252632 ± 1.24301 (W 0.0102766) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.111756 ± 0.000212745 (W 0.00113555) | 0.197079 ± 0.00477603 (W 0.00804669) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.108165 ± 5.68624e-05 (W 0.000389525) | 0.177119 ± 8.8309 (W 0.00380881) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.391158 ± 0.00523855 (W 0.00443578) | 1.02112 ± 0.0356747 (W 0.0559533) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.432061 ± 0.00135138 (W 0.00300784) | 0.862691 ± 0.0612178 (W 0.0160375) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.667511 ± 0.0186339 (W 0.00485575) | 2.59325 ± 0.134682 (W 0.127408) | - | 🔵 faer-fork 3.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.892593 ± 0.0137183 (W 0.00166119) | 2.69682 ± 0.057864 (W 0.137413) | - | 🔵 faer-fork 3.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.058953 ± 0.000100167 (W 0.000212986) | 0.146367 ± 8.76482 (W 0.00653046) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n64/b1` | 0.228111 ± 0.00617247 (W 0.00233308) | 0.738535 ± 0.0574015 (W 0.0350865) | - | 🔵 faer-fork 3.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.43407 ± 0.00489395 (W 0.00422166) | 1.56533 ± 0.121836 (W 0.058589) | - | 🔵 faer-fork 3.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.021811 ± 9.16515e-05 (W 3.11448e-05) | 0.018951 ± 0.000150533 (W 3.97492e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1 [api-inclusive]` | 0.027491 ± 4.6188e-05 (W 4.26404e-05) | 0.019121 ± 5.50757e-05 (W 3.02853e-05) | - | 🔴 faer-fork 1.44× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1` | 0.080634 ± 0.000199767 (W 0.00121204) | 0.086294 ± 0.0004769 (W 0.0013042) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.116586 ± 0.000334051 (W 0.000302796) | 0.082783 ± 0.00103379 (W 0.001069) | - | 🔴 faer-fork 1.41× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.191029 ± 0.00576663 (W 0.00425813) | 0.2221 ± 0.0128541 (W 0.0101521) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.362497 ± 0.00469692 (W 0.00380666) | 0.252152 ± 0.00381602 (W 0.00810635) | - | 🔴 faer-fork 1.44× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.356827 ± 0.00132005 (W 0.00120896) | 0.366567 ± 0.000906262 (W 0.00157634) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.348806 ± 0.000469646 (W 0.00203538) | 0.368638 ± 0.000188468 (W 0.00239141) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.23263 ± 0.00666395 (W 0.00701096) | 1.5329 ± 0.00319414 (W 0.00776756) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.25254 ± 0.00803988 (W 0.00436058) | 1.54877 ± 0.00236176 (W 0.0120379) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.45592 ± 0.0081948 (W 0.00648177) | 3.85412 ± 0.0295407 (W 0.0243178) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.50393 ± 0.00182325 (W 0.00941562) | 3.90859 ± 0.0139669 (W 0.0195308) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.098584 ± 0.000321492 (W 0.000249789) | 0.188539 ± 0.00023388 (W 0.00171994) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.099235 ± 0.00032131 (W 0.000163258) | 0.189679 ± 0.000314325 (W 0.00167829) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.435731 ± 0.000244199 (W 0.00141569) | 0.785357 ± 0.00123164 (W 0.00192739) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.437061 ± 0.00123414 (W 0.00182268) | 0.785048 ± 0.0013801 (W 0.00327655) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 1.0732 ± 0.000630601 (W 0.00153786) | 1.8722 ± 0.00241162 (W 0.00168747) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 1.07323 ± 0.00159438 (W 0.00148244) | 1.87573 ± 0.00451744 (W 0.00238392) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00273 ± 0 (W 1.78662e-05) | 0.00187 ± 5.7735e-06 (W 2.28035e-05) | - | 🔴 faer-fork 1.46× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.006461 ± 6.08276e-06 (W 1.91651e-05) | 0.00466 ± 1.0504e-05 (W 8.47656e-05) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.011391 ± 0.000132855 (W 5.83292e-05) | 0.00857 ± 2.51661e-05 (W 0.000106204) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00971 ± 0.000629688 (W 4.68754e-05) | 0.012461 ± 1e-05 (W 1.7672e-05) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.008961 ± 0.000617765 (W 4.8076e-05) | 0.00776 ± 4.07063e-05 (W 3.80697e-05) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.046232 ± 0.00122659 (W 0.000189421) | 0.097034 ± 0.000109865 (W 0.00109409) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.046432 ± 0.00116423 (W 7.15346e-05) | 0.044132 ± 0.000184752 (W 8.44393e-05) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.120126 ± 0.000185203 (W 0.00184589) | 0.171748 ± 0.00239045 (W 0.00216481) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.122616 ± 0.000135769 (W 0.00013007) | 0.119225 ± 1.2859 (W 0.00994397) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.165108 ± 0.00144956 (W 0.00134755) | 0.20927 ± 0.000274287 (W 0.00156905) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.167218 ± 0.00299072 (W 0.00285985) | 0.218641 ± 0.000626905 (W 0.00140233) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.536316 ± 0.00179202 (W 0.00343884) | 0.867781 ± 0.00190647 (W 0.00495224) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.544286 ± 0.00249174 (W 0.00318862) | 0.901402 ± 0.00682242 (W 0.00482006) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.15569 ± 0.00332311 (W 0.00470618) | 2.14904 ± 0.0156893 (W 0.022452) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.16947 ± 0.00796682 (W 0.00568368) | 2.1984 ± 0.0257105 (W 0.0197432) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.025621 ± 4.3589e-05 (W 4.20619e-05) | 0.02036 ± 5.74485e-05 (W 5.31977e-05) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.032031 ± 3.84318e-05 (W 6.76705e-05) | 0.023271 ± 1.1547e-05 (W 7.82943e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.144247 ± 0.000141892 (W 0.000992638) | 0.127746 ± 8.66025e-05 (W 0.00100068) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.177368 ± 0.000410163 (W 0.00206216) | 0.140357 ± 0.000411449 (W 0.000151322) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.393009 ± 0.000833185 (W 0.00185088) | 0.391368 ± 0.000316686 (W 0.00170639) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.516194 ± 0.00133897 (W 0.00391093) | 0.404419 ± 0.00183314 (W 0.00242418) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.021101 ± 0.00563958 (W 0.000193054) | 0.040892 ± 0.000678847 (W 0.00185083) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.021321 ± 0.0016212 (W 0.000284552) | 0.041042 ± 0.00548222 (W 0.000130882) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.072113 ± 0.0127381 (W 0.0018858) | 0.093484 ± 0.00403795 (W 0.0896241) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.059023 ± 0.0057429 (W 0.00107521) | 0.204289 ± 0.0437645 (W 0.056532) | - | 🔵 faer-fork 3.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.172578 ± 0.0147199 (W 0.00508191) | 0.235001 ± 0.153721 (W 0.186971) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.127326 ± 0.000352467 (W 0.00180873) | 0.499784 ± 0.125792 (W 0.15878) | - | 🔵 faer-fork 3.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.019261 ± 8.0829e-05 (W 9.41764e-05) | 0.016451 ± 1.76163e-05 (W 6.39234e-05) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.023552 ± 2.6839e-05 (W 0.000105201) | 0.017071 ± 3.51188e-05 (W 6.3073e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1` | 0.080064 ± 0.000545427 (W 0.000241616) | 0.082724 ± 0.000321921 (W 0.00112587) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.094095 ± 0.000104165 (W 0.000129917) | 0.078914 ± 0.0035307 (W 0.000239729) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.204049 ± 0.00918576 (W 0.00620708) | 0.22887 ± 0.00679197 (W 0.00825917) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.354546 ± 0.00667008 (W 0.00831452) | 0.247382 ± 0.0012079 (W 0.00353145) | - | 🔴 faer-fork 1.43× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.114276 ± 0.00166613 (W 0.000471533) | 0.233391 ± 0.00352118 (W 0.0113319) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.109995 ± 0.00113053 (W 0.000284895) | 0.21658 ± 0.0110747 (W 0.00517808) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.387139 ± 0.0104665 (W 0.00336859) | 1.20532 ± 0.0299412 (W 0.086865) | - | 🔵 faer-fork 3.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.450331 ± 0.00512977 (W 0.00325993) | 1.35905 ± 0.184401 (W 0.0610663) | - | 🔵 faer-fork 3.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.734005 ± 0.0178815 (W 0.00480977) | 3.07218 ± 0.266864 (W 0.111447) | - | 🔵 faer-fork 4.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.972266 ± 0.0221261 (W 0.00320693) | 2.88738 ± 0.520792 (W 0.0737591) | - | 🔵 faer-fork 2.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.058533 ± 0.000265016 (W 0.000237525) | 0.155708 ± 0.00222878 (W 0.0043476) | - | 🔵 faer-fork 2.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.279603 ± 0.00430006 (W 0.00388829) | 0.890353 ± 0.0849674 (W 0.0530021) | - | 🔵 faer-fork 3.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.460942 ± 0.00309536 (W 0.00395761) | 1.7342 ± 0.178913 (W 0.155332) | - | 🔵 faer-fork 3.76× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.021831 ± 3.51188e-05 (W 5.85662e-05) | 0.01883 ± 6.12127e-05 (W 5.10715e-05) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n32/b1 [api-inclusive]` | 0.027581 ± 2.3094e-05 (W 5.10216e-05) | 0.019131 ± 2.08167e-05 (W 2.04939e-05) | - | 🔴 faer-fork 1.44× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1` | 0.082194 ± 0.000138337 (W 0.000164408) | 0.086674 ± 0.000453578 (W 7.4493e-05) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.103745 ± 0.000185203 (W 0.000166686) | 0.082184 ± 2.51661e-05 (W 0.000237634) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.180199 ± 0.00143855 (W 0.00579709) | 0.232801 ± 0.0131231 (W 0.0146866) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.367787 ± 0.00632676 (W 0.00679234) | 0.259322 ± 0.0041995 (W 0.0072721) | - | 🔴 faer-fork 1.42× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.355777 ± 0.000468081 (W 0.00203446) | 0.367447 ± 0.000483529 (W 0.00220249) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.349037 ± 0.000269147 (W 0.00194843) | 0.369797 ± 0.00335263 (W 0.00301805) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.24593 ± 0.00455899 (W 0.00584886) | 1.51529 ± 0.00344571 (W 0.0138813) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.26827 ± 0.00904667 (W 0.00477369) | 1.5319 ± 0.0064248 (W 0.0132013) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.53847 ± 0.00684156 (W 0.0100613) | 3.92965 ± 0.023111 (W 0.0599621) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.58965 ± 0.0445153 (W 0.00887424) | 3.95619 ± 0.161874 (W 0.0513525) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n32/b1` | 0.098405 ± 9.11976e-05 (W 8.57514e-05) | 0.188579 ± 0.00164515 (W 0.00142474) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.098594 ± 3e-05 (W 0.000203655) | 0.189809 ± 0.000266521 (W 0.00123664) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.43633 ± 0.000215894 (W 0.00184098) | 0.780917 ± 0.00151595 (W 0.00139441) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.43717 ± 0.00185663 (W 0.00196889) | 0.785767 ± 0.00178397 (W 0.00149667) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 1.07959 ± 0.00255325 (W 0.00119171) | 1.87244 ± 0.000597051 (W 0.00123203) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 1.08193 ± 0.00127769 (W 0.00160151) | 1.8773 ± 0.00244169 (W 0.002655) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00273 ± 5.7735e-06 (W 1.41421e-05) | 0.001871 ± 1.67432e-05 (W 8.26438e-05) | - | 🔴 faer-fork 1.46× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.006501 ± 6.35085e-06 (W 1.91651e-05) | 0.00467 ± 1e-05 (W 2.26318e-05) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.011391 ± 0.000129693 (W 0.000115518) | 0.008551 ± 3.64737e-05 (W 8.8223e-05) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.007391 ± 2.85832e-05 (W 2.76442e-05) | 0.00742 ± 1.52753e-05 (W 3.42389e-05) | 0.011501 ± 3.60555e-05 (W 2.40062e-05) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.007841 ± 1.00167e-05 (W 3.46165e-05) | 0.007691 ± 0.000540139 (W 3.63222e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.042002 ± 5.68624e-05 (W 7.19722e-05) | 0.044252 ± 6.4291e-05 (W 3.27109e-05) | 0.063563 ± 0.000181475 (W 0.000193391) | 🔵 faer-fork 1.05× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.045543 ± 0.000105025 (W 9.33606e-05) | 0.043382 ± 0.00293296 (W 0.000140107) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1` | 0.108055 ± 0.000185716 (W 7.4027e-05) | 0.112145 ± 0.00717184 (W 0.00776685) | 0.174999 ± 0.000159897 (W 0.00135042) | · | 🔵 faer-fork 1.62× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.122256 ± 0.000254494 (W 5.75691e-05) | 0.115635 ± 0.0117373 (W 0.00676796) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.159507 ± 0.000890393 (W 0.00162071) | 0.20933 ± 0.000118731 (W 0.00237305) | 0.22791 ± 0.00137795 (W 0.00176309) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.162468 ± 0.00247034 (W 0.000860299) | 0.21864 ± 0.00012141 (W 0.0015218) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.519164 ± 0.00012822 (W 0.00246915) | 0.867961 ± 0.00423722 (W 0.00400996) | 0.764777 ± 0.00333063 (W 0.00398603) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.532015 ± 0.00143151 (W 0.0039495) | 0.904133 ± 0.00590862 (W 0.00653973) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.10489 ± 0.00556528 (W 0.00422904) | 2.10001 ± 0.0177485 (W 0.0147614) | 1.67397 ± 0.0365805 (W 0.0046404) | 🔵 faer-fork 1.90× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.12405 ± 0.012961 (W 0.00242418) | 2.17947 ± 0.00552471 (W 0.0083363) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.025471 ± 5.1316e-05 (W 4.44432e-05) | 0.020311 ± 3.05505e-05 (W 7.61577e-05) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.028641 ± 3.05505e-05 (W 6.71133e-05) | 0.023281 ± 1.52753e-05 (W 5.87367e-05) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.143487 ± 0.00606615 (W 0.000314754) | 0.127106 ± 0.000112694 (W 0.000249459) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.150388 ± 0.000529944 (W 0.000267593) | 0.136356 ± 9.49052e-05 (W 0.00136028) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.387898 ± 0.00093763 (W 0.00180393) | 0.389748 ± 0.00113069 (W 0.00152319) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.472132 ± 0.00119478 (W 0.00327592) | 0.397038 ± 0.000993886 (W 0.00183907) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.018741 ± 5.03322e-05 (W 0.000296951) | 0.040642 ± 0.00101614 (W 0.00128543) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.018661 ± 0.000427817 (W 0.000183326) | 0.042242 ± 0.000570993 (W 0.000870557) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.058223 ± 0.00687064 (W 0.000767368) | 0.113376 ± 0.0151714 (W 0.0201049) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.058023 ± 0.00774779 (W 0.001558) | 0.139607 ± 0.00259929 (W 0.0179276) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.132176 ± 0.00995071 (W 0.00340802) | 0.258752 ± 0.019391 (W 0.0250415) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.131326 ± 0.0116285 (W 0.0021299) | 0.273783 ± 0.0753927 (W 0.032587) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.017131 ± 0.00012434 (W 0.000108758) | 0.014531 ± 1e-05 (W 5.2757e-05) | 0.023152 ± 0.00012268 (W 9.74679e-05) | 🔴 faer-fork 1.18× slower than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.02026 ± 4.14769e-05 (W 9.05997e-05) | 0.017131 ± 1e-05 (W 7.84678e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1` | 0.071583 ± 0.000404145 (W 0.000186735) | 0.066103 ± 0.000111312 (W 0.00013055) | 0.111845 ± 0.000155349 (W 0.0024186) | 🔴 faer-fork 1.08× slower than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.081694 ± 0.000304138 (W 0.000154435) | 0.074274 ± 0.000319229 (W 0.000254744) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.175769 ± 0.00424726 (W 0.00377374) | 0.20062 ± 0.00160954 (W 0.00802116) | 0.274923 ± 0.00048 (W 0.00161359) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.300405 ± 0.00521412 (W 0.00523598) | 0.236411 ± 1.25064 (W 0.00536412) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.100815 ± 0.00128421 (W 0.00132411) | 0.148987 ± 0.00775218 (W 0.00443124) | 0.075944 ± 0.000132742 (W 8.70931e-05) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔴 faer-fork 1.33× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.104995 ± 0.00205173 (W 0.000660439) | 0.154768 ± 8.74904 (W 0.00402311) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.311655 ± 0.000591946 (W 0.00232118) | 0.706393 ± 0.00804549 (W 0.0437115) | 0.324495 ± 0.00818099 (W 0.00251098) | 🔵 faer-fork 2.27× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.403039 ± 0.00211126 (W 0.00477454) | 0.759656 ± 0.0271096 (W 0.0176692) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.574028 ± 0.00172778 (W 0.0042599) | 1.67162 ± 0.188058 (W 0.05334) | 0.919544 ± 0.0234404 (W 0.00156143) | 🔵 faer-fork 2.91× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.836749 ± 0.00623127 (W 0.00351804) | 1.6544 ± 0.12323 (W 0.043385) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.048742 ± 0.0001823 (W 0.000108519) | 4.24187 ± 3.71054 (W 7.58961) | 0.036241 ± 0.000129732 (W 0.00012931) | 🔵 faer-fork 87.03× faster than `faer-upstream` | 🔴 faer-fork 1.34× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n64/b1` | 0.195489 ± 0.00111577 (W 0.0023088) | 0.535816 ± 0.0369936 (W 0.0216767) | 0.20948 ± 9.56399e-05 (W 0.00163044) | 🔵 faer-fork 2.74× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.387378 ± 0.00389096 (W 0.00224524) | 1.05834 ± 0.0728456 (W 0.0253683) | 0.64736 ± 0.000990877 (W 0.00341624) | 🔵 faer-fork 2.73× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n32/b1` | 0.020781 ± 5.50757e-05 (W 3.11448e-05) | 0.016421 ± 0.000100664 (W 4.63066e-05) | 0.022691 ± 0.000108167 (W 7.49813e-05) | 🔴 faer-fork 1.27× slower than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `solve/n32/b1 [api-inclusive]` | 0.023971 ± 2.08167e-05 (W 4.88999e-05) | 0.019121 ± 2.58908e-05 (W 5.87367e-05) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1` | 0.079304 ± 0.000251913 (W 0.000130412) | 0.069264 ± 0.000324639 (W 0.000143963) | 0.107675 ± 0.000392598 (W 0.00195903) | 🔴 faer-fork 1.14× slower than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.090884 ± 8.0829e-05 (W 0.000146531) | 0.077594 ± 0.000195533 (W 0.000145479) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.21031 ± 0.00383286 (W 0.00420974) | 0.21266 ± 0.00314429 (W 0.00758073) | 0.264952 ± 0.000315595 (W 0.00176069) | · | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.332695 ± 0.00360242 (W 0.00882103) | 0.241692 ± 1.25224 (W 0.00412774) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.351457 ± 0.000647935 (W 0.00212326) | 0.362107 ± 0.00209538 (W 0.00245647) | 0.391399 ± 0.00217934 (W 0.0023816) | · | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.345396 ± 0.000372412 (W 0.00151001) | 0.365148 ± 0.00174093 (W 0.00198033) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.20161 ± 0.00729804 (W 0.00511832) | 1.56838 ± 0.00309939 (W 0.00914509) | 1.48914 ± 0.00497398 (W 0.00290021) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 1.21732 ± 0.00248054 (W 0.00533191) | 1.58401 ± 0.00342769 (W 0.00555389) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.41947 ± 0.00548633 (W 0.00485846) | 3.77183 ± 0.00453388 (W 0.0166638) | 3.48091 ± 0.0191808 (W 0.00688436) | 🔵 faer-fork 1.56× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.46492 ± 0.00485246 (W 0.00489328) | 3.78415 ± 0.0195804 (W 0.0155593) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.099105 ± 0.000320468 (W 5.4765e-05) | 0.186209 ± 7.37111e-05 (W 0.00120502) | 0.128657 ± 4.50592e-05 (W 0.000117717) | 🔵 faer-fork 1.88× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.099554 ± 0.000139787 (W 5.28507e-05) | 0.187419 ± 0.00011682 (W 0.00123176) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.43058 ± 0.000485582 (W 0.00131688) | 0.754556 ± 0.00103929 (W 0.0014538) | 0.64252 ± 0.00198306 (W 0.00270497) | 🔵 faer-fork 1.75× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.430611 ± 0.000613357 (W 0.00166429) | 0.756626 ± 0.000204287 (W 0.000909742) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 1.06132 ± 0.00212392 (W 0.00211889) | 1.84268 ± 0.00335112 (W 0.00194211) | 1.76869 ± 0.00376549 (W 0.00633408) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 1.06152 ± 0.0015562 (W 0.00100038) | 1.8443 ± 0.00229473 (W 0.00158309) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00266 ± 5.7735e-06 (W 1.52709e-05) | 0.00173 ± 5.7735e-06 (W 3.64692e-05) | - | 🔴 faer-fork 1.54× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00619 ± 1e-05 (W 2.72452e-05) | 0.004241 ± 1.53948e-05 (W 4.70021e-05) | - | 🔴 faer-fork 1.46× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.01158 ± 9.06661e-05 (W 0.000118124) | 0.00853 ± 4.6188e-05 (W 0.000115366) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00741 ± 5.7735e-06 (W 3.11962e-05) | 0.00739 ± 2.36714e-05 (W 9.72738e-05) | 0.011511 ± 1e-05 (W 0.00223421) | · | 🔵 faer-fork 1.55× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00895 ± 2.52389e-05 (W 3.36942e-05) | 0.00867 ± 0.000907432 (W 7.09246e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.041572 ± 0.00105945 (W 7.15486e-05) | 0.044352 ± 7.2111e-05 (W 0.00102825) | 0.063623 ± 7.0946e-05 (W 0.0002749) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.045942 ± 0.000201619 (W 5.63028e-05) | 0.042962 ± 8.2282e-05 (W 8.94941e-05) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.109365 ± 0.000241572 (W 0.00127278) | 0.101794 ± 0.00364394 (W 0.00411399) | 0.174718 ± 0.000178979 (W 0.00133602) | 🔴 faer-fork 1.07× slower than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.122296 ± 0.000454569 (W 0.000234295) | 0.126916 ± 1.25223 (W 0.00884769) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.162848 ± 0.000360759 (W 0.00217986) | 0.21202 ± 0.00153024 (W 0.00180583) | 0.238351 ± 0.00155696 (W 0.00240437) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.166888 ± 0.0021307 (W 0.0011286) | 0.21922 ± 0.000959391 (W 0.00184349) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.518485 ± 0.00466513 (W 0.00550421) | 0.868121 ± 0.00170034 (W 0.00730075) | 0.877182 ± 0.00615232 (W 0.00531129) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.533185 ± 0.00558217 (W 0.00464709) | 0.901232 ± 0.00337728 (W 0.00614125) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.10419 ± 0.0039222 (W 0.00257285) | 2.12033 ± 0.00978441 (W 0.00828672) | 1.86758 ± 0.0710285 (W 0.0225763) | 🔵 faer-fork 1.92× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` | `!` between-start range over 5% |
| `eigh/n96/b1 [api-inclusive]` | 1.13052 ± 0.0103445 (W 0.00441468) | 2.18019 ± 0.00826482 (W 0.0151463) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.025372 ± 2.08167e-05 (W 4.54665e-05) | 0.020451 ± 3.78594e-05 (W 7.4364e-05) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.028631 ± 3.78594e-05 (W 6.64244e-05) | 0.023361 ± 4.10731e-05 (W 0.0001105) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.140237 ± 0.000682959 (W 0.000267365) | 0.131797 ± 7.508e-05 (W 0.00146856) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.150727 ± 0.00203913 (W 0.00122985) | 0.135576 ± 0.00245846 (W 0.00109173) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.388268 ± 0.0030739 (W 0.00217568) | 0.389499 ± 0.000153657 (W 0.00155674) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.475552 ± 0.000923418 (W 0.00233506) | 0.395649 ± 0.00362329 (W 0.0021957) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.023021 ± 0.00470055 (W 0.000349328) | 0.042442 ± 0.000968143 (W 0.0014724) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.024401 ± 0.00296502 (W 0.000384003) | 0.043402 ± 0.000125033 (W 0.000278514) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.063343 ± 0.00550229 (W 0.00115984) | 0.110025 ± 0.0481495 (W 0.0456926) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.051233 ± 0.00172348 (W 0.00126691) | 0.20866 ± 0.0548542 (W 0.0633692) | - | 🔵 faer-fork 4.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.128136 ± 0.0140528 (W 0.00303873) | 0.237611 ± 0.0203009 (W 0.0468736) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.110135 ± 0.0126628 (W 0.00216037) | 0.268113 ± 0.0130342 (W 0.130834) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.01712 ± 5.50757e-06 (W 7.62771e-05) | 0.014351 ± 0.000100583 (W 4.8024e-05) | 0.023171 ± 0.000297377 (W 0.000124619) | 🔴 faer-fork 1.19× slower than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.020251 ± 5.50757e-05 (W 0.000107145) | 0.017101 ± 1.52753e-05 (W 7.29123e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1` | 0.071854 ± 0.000130997 (W 0.000123167) | 0.065893 ± 0.000285015 (W 0.00109554) | 0.112106 ± 0.000256262 (W 0.00128884) | 🔴 faer-fork 1.09× slower than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.082524 ± 0.000393954 (W 0.000119708) | 0.074453 ± 0.000192303 (W 0.00127845) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.177999 ± 0.00482983 (W 0.0043027) | 0.20069 ± 0.0029702 (W 0.0077591) | 0.274983 ± 0.000172143 (W 0.00217356) | 🔵 faer-fork 1.13× faster than `faer-upstream` | 🔵 faer-fork 1.54× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.313835 ± 0.00442015 (W 0.00634807) | 2.3788 ± 0.22602 (W 0.0102238) | - | 🔵 faer-fork 7.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1` | 0.102774 ± 0.00135929 (W 0.000448402) | 0.182749 ± 0.000740203 (W 0.00657231) | 0.075993 ± 0.000204035 (W 6.88135e-05) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔴 faer-fork 1.35× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.107675 ± 0.00210758 (W 0.000316038) | 0.174809 ± 0.0109565 (W 0.00252869) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.320105 ± 0.0116672 (W 0.00289953) | 0.895452 ± 0.0177465 (W 0.0457392) | 0.345467 ± 0.00768161 (W 0.0021963) | 🔵 faer-fork 2.80× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.42126 ± 0.0161966 (W 0.00369071) | 1.0647 ± 0.15103 (W 0.0360954) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.583148 ± 0.00567613 (W 0.00314205) | 2.18026 ± 0.0190094 (W 0.0900502) | 0.909463 ± 0.00217187 (W 0.0191958) | 🔵 faer-fork 3.74× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.85787 ± 0.00906594 (W 0.00303019) | 2.00893 ± 0.21824 (W 0.181437) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.048762 ± 0.00109786 (W 0.000203775) | 9.59405 ± 15.816 (W 6.74118) | 0.036081 ± 5.63057e-05 (W 7.4029e-05) | · | 🔴 faer-fork 1.35× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n64/b1` | 0.192959 ± 0.00143406 (W 0.00168934) | 0.588958 ± 0.0233531 (W 0.0425033) | 0.20904 ± 0.0001005 (W 0.00118673) | 🔵 faer-fork 3.05× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.383868 ± 0.00455003 (W 0.00283314) | 1.46936 ± 0.14423 (W 0.0761886) | 0.646431 ± 0.000228496 (W 0.00179257) | 🔵 faer-fork 3.83× faster than `faer-upstream` | 🔵 faer-fork 1.68× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n32/b1` | 0.020811 ± 3.05505e-05 (W 2.16795e-05) | 0.016241 ± 2.08167e-05 (W 5.56444e-05) | 0.022731 ± 0.000104403 (W 6.71133e-05) | 🔴 faer-fork 1.28× slower than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `solve/n32/b1 [api-inclusive]` | 0.023971 ± 2e-05 (W 6.14508e-05) | 0.01903 ± 7.55667e-05 (W 7.39676e-05) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1` | 0.079664 ± 2.88675e-05 (W 0.000123774) | 0.068823 ± 0.000155027 (W 0.000106429) | 0.107725 ± 0.000242487 (W 0.000107876) | 🔴 faer-fork 1.16× slower than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.104415 ± 0.000658172 (W 0.00022836) | 0.077714 ± 1.1547e-05 (W 0.000182171) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.212421 ± 0.00122932 (W 0.00482345) | 0.215541 ± 0.00667145 (W 0.00534258) | 0.264952 ± 0.000448438 (W 0.0016129) | · | 🔵 faer-fork 1.25× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.343526 ± 0.00352324 (W 0.00684993) | 0.249272 ± 1.25339 (W 0.0092698) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.351747 ± 0.000656392 (W 0.00170548) | 0.363347 ± 0.00152444 (W 0.0022356) | 0.413309 ± 0.0026397 (W 0.00327978) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.344657 ± 0.000334605 (W 0.00155629) | 0.364997 ± 0.00123541 (W 0.00240325) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.2083 ± 0.00535335 (W 0.00518385) | 1.57139 ± 0.0108443 (W 0.00632553) | 1.57179 ± 0.00585956 (W 0.00552031) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 1.22686 ± 0.00183364 (W 0.00426254) | 1.578 ± 0.00629001 (W 0.0139937) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.45851 ± 0.00548127 (W 0.00588319) | 3.80847 ± 0.0318731 (W 0.020363) | 3.52875 ± 0.0149058 (W 0.0153463) | 🔵 faer-fork 1.55× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.50602 ± 0.00749629 (W 0.00345418) | 3.83973 ± 0.0139813 (W 0.0215147) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.098774 ± 0.0001395 (W 0.000100907) | 0.186619 ± 0.000172143 (W 0.00115555) | 0.128676 ± 0.000617198 (W 0.00147027) | 🔵 faer-fork 1.89× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.099724 ± 0.000122094 (W 7.8066e-05) | 0.187009 ± 0.000114849 (W 0.00153763) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.430771 ± 0.000494247 (W 0.00196966) | 0.759046 ± 0.000984344 (W 0.00234894) | 0.642201 ± 0.0016532 (W 0.00202521) | 🔵 faer-fork 1.76× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.432041 ± 0.0009506 (W 0.00175006) | 0.756696 ± 0.000612454 (W 0.00160678) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 1.06584 ± 0.00511499 (W 0.00293455) | 1.83767 ± 0.00447942 (W 0.00232182) | 1.77013 ± 0.00195055 (W 0.00510703) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 1.06759 ± 0.000976222 (W 0.00322966) | 1.8449 ± 0.00371267 (W 0.00221773) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00266 ± 0 (W 2.19591e-05) | 0.00175 ± 0 (W 8.16284e-05) | - | 🔴 faer-fork 1.52× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00622 ± 1.53948e-05 (W 2.49058e-05) | 0.00427 ± 5.7735e-06 (W 2.68924e-05) | - | 🔴 faer-fork 1.46× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0103 ± 3.60555e-05 (W 3.00383e-05) | 0.007341 ± 5.63057e-05 (W 3.19875e-05) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00742 ± 1.52753e-05 (W 2.98714e-05) | 0.00742 ± 2.62742e-05 (W 3.9119e-05) | 0.0115 ± 2.6839e-05 (W 6.90312e-05) | · | 🔵 faer-fork 1.55× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00897 ± 1.52753e-05 (W 3.53171e-05) | 0.00864 ± 0.000583318 (W 2.58322e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.039952 ± 0.00106532 (W 0.00012502) | 0.044353 ± 5.48361e-05 (W 7.4364e-05) | 0.063563 ± 0.000162583 (W 0.000371847) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.045392 ± 0.00122521 (W 9.81896e-05) | 0.043362 ± 0.00315413 (W 5.31977e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1` | 0.107895 ± 0.00189797 (W 6.91108e-05) | 0.118356 ± 0.00690519 (W 0.00247017) | 0.175019 ± 1.00167e-05 (W 0.00137363) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.62× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.122236 ± 0.00244336 (W 0.000125444) | 0.121066 ± 1.26215 (W 0.00789686) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.164298 ± 0.00205213 (W 0.0011961) | 0.20937 ± 0.000215484 (W 0.00193122) | 0.250732 ± 0.0018469 (W 0.00181625) | 🔵 faer-fork 1.27× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.165578 ± 0.00303969 (W 0.00284896) | 0.219021 ± 0.000349476 (W 0.00175708) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.534876 ± 0.00406255 (W 0.00290497) | 0.863891 ± 0.0032363 (W 0.0055058) | 1.09826 ± 0.0427894 (W 0.0141118) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 2.05× faster than `lapack-mkl` | `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.551616 ± 0.00447684 (W 0.0028891) | 3.13569 ± 2.22033 (W 0.0203167) | - | 🔵 faer-fork 5.68× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 1.15487 ± 0.00776117 (W 0.00635296) | 2.11668 ± 0.00717449 (W 0.0191864) | 2.16255 ± 0.00326049 (W 0.0385699) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.87× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.16145 ± 0.0129278 (W 0.00528189) | 2.19077 ± 0.0806276 (W 0.0156252) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n32/b1` | 0.025531 ± 9.50438e-06 (W 5.47467e-05) | 0.020331 ± 0.000161658 (W 7.10634e-05) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.028751 ± 5.19615e-05 (W 5.52567e-05) | 0.023341 ± 4.6188e-05 (W 9.69392e-05) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.143076 ± 0.000466334 (W 0.00116833) | 0.126776 ± 0.000120554 (W 0.00116905) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.164318 ± 0.00120981 (W 0.00163919) | 0.135166 ± 0.000115717 (W 0.00112869) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.393469 ± 0.00177913 (W 0.00191884) | 0.389708 ± 0.00293535 (W 0.00177009) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.491733 ± 0.00261856 (W 0.0038632) | 0.395389 ± 0.000222785 (W 0.00185483) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.030182 ± 0.00379058 (W 0.00037637) | 0.040422 ± 0.000742316 (W 0.000740169) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.024201 ± 0.00229326 (W 0.000320702) | 0.043843 ± 0.000623905 (W 0.00537692) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.067383 ± 0.0101888 (W 0.00217516) | 0.138566 ± 0.0650204 (W 0.0741423) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.061583 ± 0.00606619 (W 0.00162943) | 0.128136 ± 0.0574474 (W 0.0507149) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.120786 ± 0.0034117 (W 0.00175563) | 0.255682 ± 0.110142 (W 0.130802) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.096605 ± 0.00189616 (W 0.00156702) | 0.389229 ± 0.113827 (W 0.146604) | - | 🔵 faer-fork 4.03× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.017141 ± 0.000156205 (W 0.000110432) | 0.01452 ± 2.0664e-05 (W 4.27586e-05) | 0.023272 ± 0.000101425 (W 0.000104939) | 🔴 faer-fork 1.18× slower than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.020281 ± 3.78594e-05 (W 6.42651e-05) | 0.017161 ± 8.18535e-05 (W 7.14717e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n64/b1` | 0.072114 ± 0.00018374 (W 0.000195498) | 0.065973 ± 0.0003177 (W 0.000206228) | 0.111575 ± 9.23418e-05 (W 0.00179342) | 🔴 faer-fork 1.09× slower than `faer-upstream` | 🔵 faer-fork 1.55× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.082374 ± 0.000180093 (W 0.000186869) | 0.074853 ± 0.00053392 (W 0.00127464) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.187319 ± 0.00339881 (W 0.00650531) | 0.21671 ± 0.00983999 (W 0.0115047) | 0.274203 ± 0.000295127 (W 0.00140955) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.324145 ± 0.00180506 (W 0.00874024) | 0.237141 ± 0.00559684 (W 0.00266017) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.100795 ± 0.00186944 (W 0.000430767) | 0.215961 ± 0.00260789 (W 0.00645922) | 0.075844 ± 0.00019 (W 0.000171596) | 🔵 faer-fork 2.14× faster than `faer-upstream` | 🔴 faer-fork 1.33× slower than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.105495 ± 0.00130908 (W 0.000263401) | 0.20387 ± 0.0183045 (W 0.00480792) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.328855 ± 0.00535234 (W 0.00244848) | 1.19375 ± 0.0807207 (W 0.0331475) | 0.353547 ± 0.0107054 (W 0.00291931) | 🔵 faer-fork 3.63× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.432611 ± 0.0117162 (W 0.00383119) | 1.05238 ± 0.169144 (W 0.0644912) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.646831 ± 0.00292695 (W 0.00483073) | 2.74992 ± 0.0467176 (W 0.12808) | 0.899673 ± 0.026543 (W 0.00562808) | 🔵 faer-fork 4.25× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.915063 ± 0.00584959 (W 0.00475798) | 2.3036 ± 0.628981 (W 0.119239) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n32/b1` | 0.048332 ± 0.000488774 (W 0.000129473) | 0.158668 ± 0.00345838 (W 0.00282903) | 0.036092 ± 3.05014e-05 (W 7.56194e-05) | 🔵 faer-fork 3.28× faster than `faer-upstream` | 🔴 faer-fork 1.34× slower than `lapack-mkl` |  |
| `qr_factor/n64/b1` | 0.19846 ± 0.000485799 (W 0.00208653) | 0.797028 ± 0.0398032 (W 0.042863) | 0.20945 ± 0.000381576 (W 0.00168798) | 🔵 faer-fork 4.02× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.409999 ± 0.00823917 (W 0.00307574) | 1.48127 ± 0.20879 (W 0.122059) | 0.646151 ± 0.0010655 (W 0.00205436) | 🔵 faer-fork 3.61× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n32/b1` | 0.020841 ± 5.7735e-06 (W 3.70135e-05) | 0.01626 ± 7.77839e-05 (W 3.34096e-05) | 0.022671 ± 0.000106927 (W 0.000116453) | 🔴 faer-fork 1.28× slower than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `solve/n32/b1 [api-inclusive]` | 0.024041 ± 5.02693e-05 (W 2.5499e-05) | 0.019281 ± 0.000190088 (W 7.77817e-05) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n64/b1` | 0.081044 ± 8.14453e-05 (W 0.000131339) | 0.069124 ± 9.02053e-05 (W 5.68173e-05) | 0.107625 ± 9.0185e-05 (W 0.00127899) | 🔴 faer-fork 1.17× slower than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.091374 ± 0.00025865 (W 0.000303979) | 0.077604 ± 0.0053484 (W 0.000231046) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1` | 0.20584 ± 0.00457855 (W 0.00470728) | 0.22571 ± 1.24203 (W 0.00898052) | 0.264262 ± 0.000365158 (W 0.0015138) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.348326 ± 0.00647884 (W 0.00549104) | 0.251592 ± 1.28631 (W 0.0116546) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.352216 ± 0.00165114 (W 0.00170173) | 0.362527 ± 0.00112545 (W 0.00189555) | 0.432 ± 0.000454459 (W 0.00270181) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.345586 ± 0.000735119 (W 0.00162627) | 0.365188 ± 0.000477368 (W 0.00206142) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.22422 ± 0.00775387 (W 0.00460449) | 1.56248 ± 0.0135022 (W 0.0122507) | 1.84495 ± 0.0760713 (W 0.00557413) | 🔵 faer-fork 1.28× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 1.2445 ± 0.0108531 (W 0.0038599) | 1.56016 ± 0.0119923 (W 0.0105168) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.5218 ± 0.00424623 (W 0.00805283) | 3.87867 ± 0.030287 (W 0.0528605) | 3.59383 ± 0.0134044 (W 0.0115038) | 🔵 faer-fork 1.54× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.56978 ± 0.0145002 (W 0.00847413) | 3.87693 ± 0.151643 (W 0.0614116) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n32/b1` | 0.099415 ± 9.34897e-05 (W 7.51146e-05) | 0.186089 ± 4.30155e-05 (W 0.00197195) | 0.128516 ± 7.57188e-05 (W 0.00113255) | 🔵 faer-fork 1.87× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.099655 ± 0.000396779 (W 8.12607e-05) | 0.187339 ± 0.000374789 (W 0.0114919) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.431261 ± 0.000758837 (W 0.00160221) | 0.756676 ± 0.0023876 (W 0.00223483) | 0.641871 ± 0.00257622 (W 0.00181389) | 🔵 faer-fork 1.75× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.431191 ± 0.000705505 (W 0.0019579) | 0.759956 ± 0.000649692 (W 0.0018883) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 1.07506 ± 0.00174459 (W 0.00126325) | 1.84155 ± 0.000995847 (W 0.00206881) | 1.76484 ± 0.00286362 (W 0.00496448) | 🔵 faer-fork 1.71× faster than `faer-upstream` | 🔵 faer-fork 1.64× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 1.07449 ± 0.00194549 (W 0.00534995) | 1.84136 ± 0.00133058 (W 0.00198146) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00266 ± 1.1547e-05 (W 1.48324e-05) | 0.001731 ± 2.25167e-05 (W 9.72481e-05) | - | 🔴 faer-fork 1.54× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00618 ± 2.0664e-05 (W 2.49239e-05) | 0.00422 ± 1.09697e-05 (W 7.17579e-05) | - | 🔴 faer-fork 1.46× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.01088 ± 3.45977e-05 (W 3.40925e-05) | 0.0079 ± 1.73205e-05 (W 2.86059e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `lu/32`, `lu/64`, `lu/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `solve/32`, `solve/64`, `solve/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `triangular_solve/32`, `triangular_solve/64`, `triangular_solve/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.087065 ± 0.00162359 (W 0.00138669) | 0.163008 ± 0.00291638 (W 0.00427063) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.094474 ± 0.001852 (W 9.73103e-05) | 0.113466 ± 2.49771 (W 0.0010318) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.330156 ± 0.00405898 (W 0.00139338) | 0.746565 ± 0.023599 (W 0.0134274) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.495084 ± 0.000426353 (W 0.00262413) | 0.588008 ± 0.0172795 (W 0.0144433) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.05872 ± 0.00409981 (W 0.00586352) | 2.11584 ± 0.0220113 (W 0.0270718) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.84267 ± 0.0101875 (W 0.00622919) | 3.15858 ± 0.0267246 (W 0.0140387) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.27907 ± 0.00750284 (W 0.00484027) | 3.15226 ± 0.00529378 (W 0.00817606) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.28117 ± 0.0110626 (W 0.00292278) | 3.16202 ± 0.0065269 (W 0.0121115) | - | 🔵 faer-fork 2.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 4.24757 ± 0.0163316 (W 0.00961791) | 10.1658 ± 0.167836 (W 0.0282719) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 4.33702 ± 0.0179279 (W 0.014282) | 10.3685 ± 0.0901583 (W 0.0493236) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.2187 ± 0.0525262 (W 0.02106) | 37.5048 ± 0.119814 (W 0.400933) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.3975 ± 0.0519079 (W 0.0293154) | 38.0947 ± 0.177784 (W 0.196732) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.489453 ± 0.000593745 (W 0.00893262) | 0.446091 ± 0.000346458 (W 0.00136104) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.488933 ± 0.0193396 (W 0.00210503) | 0.456242 ± 0.00335076 (W 0.00194667) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 3.02476 ± 0.0223306 (W 0.0107302) | 4.28862 ± 0.0192746 (W 0.00872268) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.53861 ± 0.0302985 (W 0.00604386) | 3.82342 ± 0.0284042 (W 0.00358621) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 24.0706 ± 0.199331 (W 0.0751788) | 32.9589 ± 0.16861 (W 0.120957) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 25.9325 ± 0.117825 (W 0.0280583) | 31.4587 ± 0.105288 (W 0.888712) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.099355 ± 0.00707544 (W 0.0015711) | 0.240411 ± 0.0233852 (W 0.0130519) | - | 🔵 faer-fork 2.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.100655 ± 0.0066399 (W 0.00186066) | 0.366757 ± 0.0283086 (W 0.0405767) | - | 🔵 faer-fork 3.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.406489 ± 0.00710914 (W 0.00456353) | 0.83063 ± 0.0662447 (W 0.03391) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.40913 ± 0.00818391 (W 0.00331434) | 0.850621 ± 0.207284 (W 0.0590551) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.15833 ± 0.000729109 (W 0.0363481) | 3.03912 ± 0.0377874 (W 0.0568684) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.14947 ± 0.00557016 (W 0.0209567) | 3.04465 ± 0.0365227 (W 0.037391) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.183969 ± 0.00507081 (W 0.00899851) | 0.384968 ± 0.00911622 (W 0.00310977) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.230111 ± 0.00256652 (W 0.00469257) | 0.372618 ± 4.9603 (W 1.9507) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.656781 ± 0.00402092 (W 0.00309294) | 3.98389 ± 0.0103535 (W 0.024821) | - | 🔵 faer-fork 6.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.27472 ± 0.00704615 (W 0.00400444) | 1.47295 ± 0.0302396 (W 0.0243938) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.64762 ± 0.0067277 (W 0.0173025) | 17.5614 ± 0.109292 (W 0.17694) | - | 🔵 faer-fork 6.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.23233 ± 0.0164745 (W 0.0120451) | 7.63475 ± 0.0231063 (W 0.0225419) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.709353 ± 0.00625202 (W 0.00368143) | 2.28963 ± 0.249858 (W 0.0451126) | - | 🔵 faer-fork 3.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.847691 ± 0.00633564 (W 0.00266185) | 1.69052 ± 0.176598 (W 0.057709) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.58623 ± 0.00530483 (W 0.00465169) | 7.23641 ± 0.455122 (W 0.0397503) | - | 🔵 faer-fork 4.56× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 2.4466 ± 0.00420189 (W 0.00468963) | 5.74723 ± 0.624291 (W 0.0509065) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 5.80605 ± 0.00266092 (W 0.0277511) | 35.7356 ± 0.284763 (W 0.245935) | - | 🔵 faer-fork 6.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.59156 ± 0.0306565 (W 0.0120704) | 22.3887 ± 0.537456 (W 0.699512) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.424801 ± 0.00141986 (W 0.00330203) | 1.66757 ± 0.145226 (W 0.0460312) | - | 🔵 faer-fork 3.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n256/b1` | 1.14481 ± 0.00209573 (W 0.0047981) | 6.54604 ± 0.586775 (W 0.115595) | - | 🔵 faer-fork 5.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 3.27348 ± 0.00409958 (W 0.00793059) | 30.6638 ± 0.175219 (W 0.157727) | - | 🔵 faer-fork 9.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.158647 ± 0.00131602 (W 0.0076863) | 0.308885 ± 0.006453 (W 0.00981857) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.297154 ± 0.00431516 (W 0.0138718) | 0.289744 ± 0.0148712 (W 0.00902892) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.766866 ± 0.00242894 (W 0.00498971) | 2.2955 ± 0.0120053 (W 0.01921) | - | 🔵 faer-fork 2.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.929394 ± 0.00974077 (W 0.00401823) | 1.41373 ± 0.0145655 (W 0.0175888) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.70502 ± 0.0112282 (W 0.0162595) | 13.0031 ± 0.035964 (W 0.0554666) | - | 🔵 faer-fork 4.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.53223 ± 0.0138808 (W 0.0175633) | 7.20063 ± 0.0367112 (W 0.0300469) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.994 ± 0.00994802 (W 0.00970795) | 3.93165 ± 0.00523145 (W 0.0262704) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.04336 ± 0.0133744 (W 0.00473477) | 3.95858 ± 0.0542808 (W 25.5597) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 11.3091 ± 0.0139738 (W 0.0101139) | 19.4655 ± 0.0388354 (W 0.0936292) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 11.5129 ± 0.0105954 (W 0.0110322) | 19.7902 ± 0.0657961 (W 0.0772007) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 47.0273 ± 0.0609681 (W 0.051357) | 92.4883 ± 1.30927 (W 2.20016) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 47.4874 ± 0.198769 (W 0.0652729) | 93.4432 ± 0.564563 (W 1.63027) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.15741 ± 0.00149923 (W 0.00216125) | 2.65204 ± 0.00188943 (W 0.0120615) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.18425 ± 0.00095495 (W 0.00252841) | 2.66087 ± 0.199396 (W 0.00489322) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 6.26713 ± 0.013207 (W 0.00961555) | 14.5772 ± 0.051688 (W 0.0360157) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.38618 ± 0.0258506 (W 0.00720523) | 14.5997 ± 0.112037 (W 0.0426821) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 27.2164 ± 0.0747439 (W 0.0453435) | 70.9349 ± 0.324344 (W 0.369433) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 27.2798 ± 0.131131 (W 0.0474718) | 71.2711 ± 0.125632 (W 0.906957) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00889 ± 9.50438e-06 (W 2.95855e-05) | 0.00602 ± 4.89932e-05 (W 5.22322e-05) | - | 🔴 faer-fork 1.48× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.023991 ± 7.82326e-05 (W 6.95859e-05) | 0.01831 ± 0.000200176 (W 7.49667e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.059613 ± 0.000440606 (W 0.000303898) | 0.048152 ± 0.000464901 (W 0.00170224) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.087804 ± 0.00203827 (W 0.000119666) | 0.174698 ± 0.0253304 (W 0.00798948) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.092565 ± 0.00192984 (W 0.000158105) | 4.45924 ± 2.50772 (W 0.0117855) | - | 🔵 faer-fork 48.17× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.335306 ± 0.0248045 (W 0.00319886) | 1.32711 ± 0.143481 (W 0.0655807) | - | 🔵 faer-fork 3.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.538325 ± 0.0339366 (W 0.00518835) | 0.647591 ± 0.0491539 (W 0.0401907) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.08599 ± 0.0421819 (W 0.00634051) | 4.79125 ± 0.139297 (W 0.0766434) | - | 🔵 faer-fork 4.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.97301 ± 0.110974 (W 0.00714646) | 3.2419 ± 0.0537302 (W 6.41124) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.24164 ± 0.00670828 (W 0.00658222) | 3.16256 ± 0.00669438 (W 0.016859) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.26654 ± 0.00568095 (W 0.00568978) | 3.21789 ± 0.0126617 (W 0.0112871) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 4.05218 ± 0.0105603 (W 0.00948266) | 10.5629 ± 0.151393 (W 0.0854125) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 4.13247 ± 0.0156303 (W 0.0106751) | 10.4677 ± 0.00123135 (W 0.100994) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 17.5969 ± 0.137033 (W 0.0339504) | 41.9807 ± 0.379971 (W 1.42503) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 17.7529 ± 0.0805529 (W 0.0451841) | 43.9979 ± 1.52203 (W 16.3546) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1` | 0.515575 ± 0.0118288 (W 0.00244842) | 0.446951 ± 0.000966483 (W 0.00136948) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.487513 ± 0.00155576 (W 0.00213604) | 0.451982 ± 0.00310023 (W 0.00233834) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.11819 ± 0.0256227 (W 0.0120508) | 4.27643 ± 0.000978928 (W 0.00984351) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.60043 ± 0.020299 (W 0.00558029) | 3.88952 ± 0.0394264 (W 0.00380858) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 24.0404 ± 0.187719 (W 0.0360235) | 33.068 ± 0.0127125 (W 0.0309758) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 26.3305 ± 0.143725 (W 0.0484979) | 31.3476 ± 0.111764 (W 0.0676951) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.098875 ± 0.0109273 (W 0.00216848) | 0.518145 ± 0.177097 (W 0.0669977) | - | 🔵 faer-fork 5.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.099845 ± 0.00349039 (W 0.00110906) | 0.577277 ± 0.0197224 (W 0.0135129) | - | 🔵 faer-fork 5.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.451011 ± 0.060564 (W 0.0111192) | 1.14792 ± 0.44086 (W 0.139512) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.436971 ± 0.00252001 (W 0.00588953) | 1.81631 ± 0.0328029 (W 0.0298819) | - | 🔵 faer-fork 4.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.53106 ± 0.00409801 (W 0.0112418) | 4.03294 ± 0.328511 (W 0.741621) | - | 🔵 faer-fork 2.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.5358 ± 0.00575845 (W 0.0140466) | 4.62662 ± 0.215756 (W 0.28584) | - | 🔵 faer-fork 3.01× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.189459 ± 0.00506895 (W 0.00706338) | 0.386268 ± 0.0319582 (W 0.0115666) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.222411 ± 0.0017411 (W 0.0113849) | 0.293244 ± 0.00331368 (W 0.0222351) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.670172 ± 0.037585 (W 0.00299257) | 4.24625 ± 0.0435333 (W 0.0372931) | - | 🔵 faer-fork 6.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 1.31495 ± 0.0371376 (W 0.00604898) | 34.1093 ± 18.814 (W 0.070738) | - | 🔵 faer-fork 25.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 2.43865 ± 0.105171 (W 0.021366) | 18.621 ± 1.08146 (W 0.592804) | - | 🔵 faer-fork 7.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 5.32105 ± 0.217961 (W 0.0100621) | 7.95254 ± 0.0260974 (W 26.1325) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.740155 ± 0.00554044 (W 0.00391435) | 3.13627 ± 0.0826124 (W 0.134776) | - | 🔵 faer-fork 4.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.892112 ± 0.020326 (W 0.00785578) | 2.91743 ± 0.573675 (W 0.118528) | - | 🔵 faer-fork 3.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.54758 ± 0.00316103 (W 0.00474204) | 11.8068 ± 0.136676 (W 0.269466) | - | 🔵 faer-fork 7.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.48078 ± 0.0192977 (W 0.00817501) | 6.57998 ± 3.08186 (W 0.142712) | - | 🔵 faer-fork 2.65× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 5.07982 ± 0.0119888 (W 0.0223614) | 51.4897 ± 2.66271 (W 3.06725) | - | 🔵 faer-fork 10.14× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 8.38319 ± 0.0200168 (W 0.0154697) | 35.571 ± 2.56632 (W 4.44062) | - | 🔵 faer-fork 4.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.468842 ± 0.0066943 (W 0.00192235) | 1.91833 ± 0.28226 (W 31.925) | - | 🔵 faer-fork 4.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n256/b1` | 1.27121 ± 0.0475495 (W 0.0066779) | 9.48283 ± 1.30602 (W 0.247411) | - | 🔵 faer-fork 7.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 3.32247 ± 0.0113177 (W 0.0146308) | 40.0454 ± 1.83038 (W 1.76267) | - | 🔵 faer-fork 12.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.150557 ± 0.00323315 (W 0.00726772) | 0.348557 ± 0.0350354 (W 0.0237376) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.229521 ± 0.00514709 (W 0.0118855) | 0.297514 ± 4.99765 (W 0.00529973) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.791237 ± 0.039428 (W 0.00401402) | 2.80305 ± 0.368882 (W 0.0673969) | - | 🔵 faer-fork 3.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.976557 ± 0.0387716 (W 0.00235404) | 1.70042 ± 36.6077 (W 7.21926) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 2.62206 ± 0.0598813 (W 0.0153204) | 14.9807 ± 0.401589 (W 0.307252) | - | 🔵 faer-fork 5.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.62354 ± 0.158319 (W 0.0203216) | 7.55741 ± 0.0430967 (W 44.2771) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 3.02996 ± 0.00937332 (W 0.0120289) | 4.08427 ± 0.0899814 (W 0.0754679) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.0729 ± 0.00881803 (W 0.00998972) | 4.0112 ± 0.013036 (W 0.0341454) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 11.3376 ± 0.0289092 (W 0.00845047) | 26.0202 ± 1.7643 (W 12.9275) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n256/b1 [api-inclusive]` | 11.5428 ± 0.0291136 (W 0.00975134) | 20.691 ± 0.0940837 (W 0.427527) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.1443 ± 0.161756 (W 0.108578) | 98.8383 ± 1.79423 (W 53.3815) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.674 ± 0.14105 (W 0.0778599) | 100.064 ± 0.367317 (W 3.4376) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.15792 ± 0.0024405 (W 0.00208321) | 2.65894 ± 0.0128946 (W 0.00609804) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.16095 ± 0.00164482 (W 0.00177746) | 2.66126 ± 0.201849 (W 0.0341089) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 6.28364 ± 0.00123348 (W 0.00896581) | 14.931 ± 0.0201813 (W 0.0583863) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.39554 ± 0.0232811 (W 0.0101134) | 14.9778 ± 0.0539247 (W 0.052139) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 29.5073 ± 0.124339 (W 0.0930371) | 82.3187 ± 19.179 (W 35.6454) | - | 🔵 faer-fork 2.79× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 29.2563 ± 0.196133 (W 0.0335628) | 76.3199 ± 1.55865 (W 48.9213) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00914 ± 2.95014e-05 (W 2.10071e-05) | 0.00629 ± 5.13842e-05 (W 9.2332e-05) | - | 🔴 faer-fork 1.45× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.023231 ± 9.84886e-05 (W 4.84768e-05) | 0.017491 ± 0.000540278 (W 0.000182779) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n512/b1` | 0.057103 ± 0.000825918 (W 0.000269834) | 0.047752 ± 1.27401 (W 0.00057817) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.086004 ± 0.00177588 (W 0.000154228) | 0.174178 ± 0.064936 (W 0.01526) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.094255 ± 0.00243719 (W 5.3975e-05) | 4.58436 ± 2.57672 (W 0.0263026) | - | 🔵 faer-fork 48.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.321216 ± 0.0263838 (W 0.00430624) | 1.68419 ± 0.0447787 (W 0.0904187) | - | 🔵 faer-fork 5.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.548126 ± 0.0318761 (W 0.00388195) | 0.753266 ± 0.202438 (W 0.0222889) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.13635 ± 0.0798967 (W 0.007261) | 5.29916 ± 0.126065 (W 0.146217) | - | 🔵 faer-fork 4.66× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.05288 ± 0.146504 (W 0.0255701) | 3.97324 ± 0.528337 (W 0.110943) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.24127 ± 0.0072898 (W 0.00473156) | 3.24412 ± 0.0259474 (W 0.0143635) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.25606 ± 0.0099804 (W 0.00617117) | 3.31308 ± 20.2489 (W 1.66974) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1` | 4.1627 ± 0.0243087 (W 0.011474) | 15.6688 ± 0.943738 (W 0.677744) | - | 🔵 faer-fork 3.76× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1 [api-inclusive]` | 4.22573 ± 0.0311771 (W 0.0090403) | 10.7895 ± 0.386473 (W 0.104506) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n512/b1` | 21.1731 ± 0.0780412 (W 0.0740744) | 73.1353 ± 26.4424 (W 28.3621) | - | 🔵 faer-fork 3.45× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n512/b1 [api-inclusive]` | 22.5244 ± 0.054537 (W 0.0556356) | 95.1661 ± 32.4352 (W 33.4541) | - | 🔵 faer-fork 4.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1` | 0.518044 ± 0.0140129 (W 0.00217933) | 0.447531 ± 0.00188598 (W 0.00213459) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.487863 ± 0.000650146 (W 0.00262293) | 0.454642 ± 0.00125525 (W 0.00175314) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.14416 ± 0.015749 (W 0.00688964) | 4.31092 ± 0.0487775 (W 0.00637458) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.64171 ± 0.0057762 (W 0.00535986) | 3.86763 ± 0.0165892 (W 0.00414277) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.351 ± 0.0855385 (W 0.0417231) | 35.6617 ± 1.72747 (W 0.185101) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 27.5701 ± 0.299133 (W 0.0414384) | 32.4984 ± 0.172436 (W 0.0927844) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.104005 ± 0.0312752 (W 0.00162546) | 0.481822 ± 0.173307 (W 0.0461953) | - | 🔵 faer-fork 4.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.105915 ± 0.0154587 (W 0.00111558) | 0.585098 ± 0.00722178 (W 0.0734536) | - | 🔵 faer-fork 5.52× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.545526 ± 0.0637909 (W 0.00786339) | 2.027 ± 0.364394 (W 0.0498355) | - | 🔵 faer-fork 3.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.41887 ± 0.0119335 (W 0.00869275) | 2.08749 ± 0.0189405 (W 0.0542034) | - | 🔵 faer-fork 4.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.18953 ± 0.204325 (W 0.0567458) | 6.00452 ± 0.0643411 (W 0.148505) | - | 🔵 faer-fork 2.74× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.72211 ± 0.0229726 (W 0.018538) | 5.90456 ± 0.0542605 (W 0.101278) | - | 🔵 faer-fork 3.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.174519 ± 0.00139548 (W 0.00175615) | 0.6161 ± 0.132595 (W 0.0214151) | - | 🔵 faer-fork 3.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.287713 ± 0.00254962 (W 0.00248173) | 0.280423 ± 0.00159188 (W 0.00734481) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.718854 ± 0.0270508 (W 0.00279435) | 5.67974 ± 0.0694009 (W 0.0819605) | - | 🔵 faer-fork 7.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 1.36513 ± 0.019391 (W 0.00564452) | 1.8599 ± 0.182623 (W 0.031114) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 2.40646 ± 0.0505339 (W 0.0151968) | 27.3225 ± 1.27978 (W 0.594707) | - | 🔵 faer-fork 11.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 5.40073 ± 0.0722275 (W 0.0252109) | 10.7815 ± 1.32194 (W 1.33454) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.824829 ± 0.0347298 (W 0.00397451) | 3.86875 ± 0.190201 (W 0.11532) | - | 🔵 faer-fork 4.69× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.961995 ± 0.018127 (W 0.0039745) | 3.95992 ± 0.514999 (W 0.0642186) | - | 🔵 faer-fork 4.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.63929 ± 0.00620342 (W 0.00883154) | 14.1406 ± 0.747961 (W 0.199268) | - | 🔵 faer-fork 8.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 2.59502 ± 0.0111252 (W 0.0152242) | 14.6329 ± 1.2657 (W 0.371281) | - | 🔵 faer-fork 5.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 5.02146 ± 0.00933056 (W 0.0175807) | 60.4856 ± 1.68919 (W 0.633868) | - | 🔵 faer-fork 12.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 8.62665 ± 0.0337644 (W 0.0225239) | 37.1546 ± 1.22314 (W 1.21547) | - | 🔵 faer-fork 4.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.515765 ± 0.0285931 (W 0.00255792) | 3.08847 ± 0.374382 (W 0.122924) | - | 🔵 faer-fork 5.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n256/b1` | 1.27914 ± 0.0857163 (W 0.00502524) | 7.86892 ± 0.538594 (W 0.176632) | - | 🔵 faer-fork 6.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 3.49854 ± 0.157518 (W 0.00929849) | 42.1248 ± 1.85945 (W 2.65163) | - | 🔵 faer-fork 12.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.161698 ± 0.00119433 (W 0.00238536) | 0.354676 ± 0.0751752 (W 0.00796902) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.301364 ± 0.00675164 (W 0.0111369) | 0.293024 ± 0.0204216 (W 0.0136028) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.803008 ± 0.0142704 (W 0.00434228) | 3.99781 ± 0.0720128 (W 0.144066) | - | 🔵 faer-fork 4.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.977096 ± 0.021252 (W 0.00428958) | 1.87984 ± 0.201221 (W 0.0937534) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 2.48924 ± 0.0587818 (W 0.0229375) | 19.3669 ± 0.115069 (W 0.075631) | - | 🔵 faer-fork 7.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.71561 ± 0.104428 (W 0.0247095) | 8.09274 ± 0.242928 (W 0.0864158) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 3.11054 ± 0.0785442 (W 0.00869783) | 4.76019 ± 0.0491249 (W 0.282121) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.19049 ± 0.160662 (W 0.00571227) | 4.28119 ± 0.100948 (W 0.0597446) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n256/b1` | 11.5774 ± 0.0188342 (W 0.00984315) | 48.9047 ± 3.844 (W 4.75432) | - | 🔵 faer-fork 4.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n256/b1 [api-inclusive]` | 11.9147 ± 0.190481 (W 0.0159935) | 21.2553 ± 3.19513 (W 23.2215) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1` | 52.0051 ± 0.0416437 (W 0.141189) | 129.93 ± 2.17829 (W 3.2586) | - | 🔵 faer-fork 2.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 52.0333 ± 0.0687634 (W 0.0778184) | 129.081 ± 1.53562 (W 38.7563) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.15719 ± 0.00141312 (W 0.00323631) | 2.64458 ± 0.00128302 (W 0.0396317) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.16207 ± 0.00136617 (W 0.00212788) | 2.65029 ± 0.203351 (W 0.0101244) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 6.36391 ± 0.00958823 (W 0.00659776) | 15.6552 ± 0.0480419 (W 0.321969) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.38768 ± 0.0119423 (W 0.00545381) | 15.3119 ± 0.0757685 (W 0.0756348) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 32.6327 ± 0.296583 (W 0.0955488) | 106.573 ± 50.3599 (W 17.634) | - | 🔵 faer-fork 3.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 32.3297 ± 0.744118 (W 0.162435) | 95.2254 ± 6.43801 (W 74.9115) | - | 🔵 faer-fork 2.95× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n128/b1` | 0.008921 ± 3.21455e-05 (W 3.24546e-05) | 0.00594 ± 4.6188e-05 (W 7.2016e-05) | - | 🔴 faer-fork 1.50× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.023741 ± 0.000221518 (W 7.544e-05) | 0.018111 ± 0.000117475 (W 6.16133e-05) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.062103 ± 0.000152753 (W 0.000294449) | 2.23306 ± 1.26042 (W 0.0165124) | - | 🔵 faer-fork 35.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.075643 ± 0.00154235 (W 0.00126192) | 0.077594 ± 0.00140045 (W 0.00341936) | 0.134996 ± 0.000106068 (W 0.00126577) | · | 🔵 faer-fork 1.78× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.085074 ± 0.00146729 (W 7.56307e-05) | 4.4391 ± 2.5116 (W 0.00947136) | - | 🔵 faer-fork 52.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.305274 ± 0.00366598 (W 0.00339015) | 0.397359 ± 0.0198513 (W 0.0084314) | 0.518474 ± 0.0115534 (W 0.050788) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔵 faer-fork 1.70× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.43133 ± 0.00403066 (W 0.00338199) | 0.474743 ± 4.44135 (W 0.0199122) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.07388 ± 0.00890698 (W 0.00883461) | 1.41201 ± 0.0305771 (W 0.0142582) | 1.15174 ± 0.0163379 (W 0.0425724) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.57614 ± 0.00512449 (W 0.00577429) | 1.75454 ± 0.0118821 (W 0.0295512) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.26672 ± 0.00781072 (W 0.00574814) | 3.14173 ± 0.00668686 (W 0.0136029) | 1.72822 ± 0.0197055 (W 0.00665348) | 🔵 faer-fork 2.48× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.28975 ± 0.0150471 (W 0.00492596) | 3.16586 ± 0.00955126 (W 0.0064629) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 4.04551 ± 0.0560373 (W 0.0119527) | 10.2981 ± 0.0593526 (W 0.0626451) | 6.20062 ± 0.00682766 (W 0.0128759) | 🔵 faer-fork 2.55× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 4.12794 ± 0.0320015 (W 0.00941236) | 10.3693 ± 0.1052 (W 0.0542955) | - | 🔵 faer-fork 2.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.2829 ± 0.0521284 (W 0.0244922) | 36.2019 ± 0.367841 (W 0.277274) | 20.3055 ± 0.0759973 (W 0.0572884) | 🔵 faer-fork 2.37× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 15.441 ± 0.0200188 (W 0.0201156) | 36.7785 ± 0.261497 (W 0.995948) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.469632 ± 0.00153837 (W 0.00210052) | 0.442041 ± 0.0011003 (W 0.00152411) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.449071 ± 0.00103968 (W 0.00279726) | 0.42712 ± 0.00123074 (W 0.00212028) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.97151 ± 0.00812586 (W 0.00766057) | 3.85435 ± 0.0404789 (W 0.00746641) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.36042 ± 0.0059547 (W 0.00450395) | 3.6331 ± 0.0370553 (W 0.0101765) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 23.4207 ± 0.118261 (W 0.0335151) | 29.409 ± 0.0539709 (W 0.0268111) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 25.3395 ± 0.119099 (W 0.0561681) | 28.2287 ± 0.0490241 (W 0.0290308) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.094185 ± 0.00324969 (W 0.00116782) | 0.245201 ± 0.0379583 (W 0.00873057) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.094574 ± 0.00213603 (W 0.00202421) | 0.366967 ± 0.056351 (W 0.0374454) | - | 🔵 faer-fork 3.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.331755 ± 0.00830971 (W 0.00265724) | 0.84196 ± 0.10486 (W 0.0443856) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.332046 ± 0.00949752 (W 0.00315434) | 1.06318 ± 0.0482043 (W 0.0713834) | - | 🔵 faer-fork 3.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.74462 ± 0.0117974 (W 0.0423762) | 2.85198 ± 0.0747538 (W 0.120806) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.7772 ± 0.0475461 (W 0.0339462) | 2.99996 ± 0.0469959 (W 0.0871393) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.149757 ± 0.00174233 (W 0.000957667) | 0.20544 ± 0.0033687 (W 0.0162063) | 0.196279 ± 0.000655159 (W 0.00140322) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.182129 ± 0.000497427 (W 0.0040713) | 0.242912 ± 5.00971 (W 0.0138735) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.564927 ± 0.02451 (W 0.00373844) | 1.10128 ± 0.0162861 (W 0.0284695) | 0.945655 ± 0.0689664 (W 0.033816) | 🔵 faer-fork 1.95× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 1.12287 ± 0.0135487 (W 0.00814317) | 1.20897 ± 0.0178745 (W 0.0172688) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.28455 ± 0.0108506 (W 0.0116246) | 4.29452 ± 0.0294853 (W 0.0301586) | 1.7639 ± 0.698011 (W 0.118129) | 🔵 faer-fork 1.88× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n512/b1 [api-inclusive]` | 4.54378 ± 0.0161448 (W 0.013006) | 4.75967 ± 0.0331634 (W 0.0163739) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.62224 ± 0.006505 (W 0.00555849) | 1.67671 ± 0.0231988 (W 0.044745) | 0.569357 ± 0.0194345 (W 0.0110809) | 🔵 faer-fork 2.69× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.84457 ± 0.0224355 (W 0.00487623) | 1.69788 ± 0.274062 (W 0.0713139) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.35581 ± 0.0132976 (W 0.0032688) | 5.80602 ± 0.362857 (W 0.146012) | 1.3234 ± 0.286098 (W 0.00956387) | 🔵 faer-fork 4.28× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n256/b1 [api-inclusive]` | 2.30574 ± 0.0190326 (W 0.00720276) | 5.48123 ± 0.10979 (W 0.138239) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 4.2063 ± 0.00149068 (W 0.01874) | 14.6392 ± 0.0227717 (W 0.157211) | 5.97283 ± 0.878245 (W 0.171824) | 🔵 faer-fork 3.48× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n512/b1 [api-inclusive]` | 7.88628 ± 0.00132214 (W 0.0143161) | 17.3699 ± 1.16893 (W 1.23496) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.379578 ± 0.00715597 (W 0.00284521) | 1.35842 ± 0.0432316 (W 0.0502338) | 0.322036 ± 0.0555177 (W 0.00606624) | 🔵 faer-fork 3.58× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n256/b1` | 1.04164 ± 0.0139813 (W 0.00314819) | 4.22071 ± 0.234769 (W 0.13113) | 0.825439 ± 0.208813 (W 0.00693222) | 🔵 faer-fork 4.05× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n512/b1` | 2.94889 ± 0.00345779 (W 0.00544534) | 10.1765 ± 1.49419 (W 0.328417) | 3.41785 ± 0.574765 (W 0.0722727) | 🔵 faer-fork 3.45× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n128/b1` | 0.171198 ± 0.0062997 (W 0.00924995) | 0.221161 ± 0.024167 (W 0.00755249) | 0.195049 ± 0.000260615 (W 0.00161668) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.264053 ± 0.00321991 (W 0.0107588) | 0.262223 ± 4.98616 (W 0.0148963) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.62959 ± 0.0111805 (W 0.00432025) | 1.04869 ± 0.0288295 (W 0.0140528) | 1.27993 ± 0.228657 (W 0.0156204) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 2.03× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.790648 ± 0.0147978 (W 0.00436977) | 1.1815 ± 0.0186057 (W 0.0166774) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.27979 ± 0.0101945 (W 0.0104474) | 3.65085 ± 0.0277532 (W 0.0173634) | 1.61725 ± 1.62895 (W 0.0341814) | 🔵 faer-fork 1.60× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n512/b1 [api-inclusive]` | 2.91768 ± 0.0155634 (W 0.0124976) | 4.38092 ± 0.020683 (W 0.0192579) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.97866 ± 0.0105501 (W 0.0115391) | 3.8922 ± 0.0220741 (W 0.0430033) | 3.80915 ± 0.0273304 (W 0.0126297) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 3.02006 ± 0.00704246 (W 0.00833627) | 3.93466 ± 0.0197187 (W 0.0310502) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 11.2721 ± 0.0213154 (W 0.00710912) | 19.4461 ± 0.0771301 (W 0.11644) | 16.4553 ± 0.0406707 (W 0.0137619) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 11.4703 ± 0.0320106 (W 0.00975222) | 19.8312 ± 0.0891721 (W 0.117606) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 44.3595 ± 0.1322 (W 0.0233395) | 89.0691 ± 0.892078 (W 1.22392) | 81.3326 ± 0.485602 (W 0.0698377) | 🔵 faer-fork 2.01× faster than `faer-upstream` | 🔵 faer-fork 1.83× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 44.9479 ± 0.0802859 (W 0.0356062) | 91.843 ± 0.763133 (W 1.87583) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.1326 ± 0.00313598 (W 0.00119206) | 2.58759 ± 0.00133254 (W 0.0285755) | 1.57902 ± 0.00850028 (W 0.00361643) | 🔵 faer-fork 2.28× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.16056 ± 0.000279344 (W 0.00118848) | 2.58096 ± 0.00698113 (W 0.0303796) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.06124 ± 0.0305637 (W 0.00602127) | 14.2921 ± 0.066714 (W 0.0613179) | 8.10085 ± 0.0219108 (W 0.00839092) | 🔵 faer-fork 2.36× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.13783 ± 0.0108129 (W 0.00621922) | 14.3483 ± 0.0193672 (W 0.0501276) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 24.8641 ± 0.0867975 (W 0.0286583) | 68.7355 ± 0.0790526 (W 0.132683) | 35.8593 ± 0.260304 (W 0.101197) | 🔵 faer-fork 2.76× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 24.9283 ± 0.0317371 (W 0.0360116) | 69.154 ± 0.183842 (W 0.424796) | - | 🔵 faer-fork 2.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00801 ± 5.50757e-05 (W 6.05582e-05) | 0.00553 ± 1.52753e-05 (W 6.48938e-05) | - | 🔴 faer-fork 1.45× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.027282 ± 0.000463972 (W 0.000186291) | 0.022321 ± 0.00242471 (W 0.000214537) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n512/b1` | 0.057433 ± 0.00122312 (W 0.000290378) | 0.053883 ± 0.00100466 (W 0.00171005) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.075693 ± 0.00162905 (W 0.000166356) | 0.101295 ± 0.0157603 (W 0.0148549) | 0.133066 ± 0.00025865 (W 0.00127128) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.76× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.082053 ± 0.00171219 (W 6.83169e-05) | 4.47914 ± 3.95383 (W 0.0196503) | - | 🔵 faer-fork 54.59× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.315345 ± 0.0171337 (W 0.00708153) | 0.682512 ± 0.018659 (W 0.0343947) | 0.596858 ± 0.0218372 (W 0.0356654) | 🔵 faer-fork 2.16× faster than `faer-upstream` | 🔵 faer-fork 1.89× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.43152 ± 0.0590345 (W 0.00421979) | 0.561507 ± 0.354116 (W 0.00997512) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.10414 ± 0.00744359 (W 0.0073231) | 2.53031 ± 0.0221674 (W 0.0338964) | 1.35565 ± 0.057989 (W 0.0480182) | 🔵 faer-fork 2.29× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.62856 ± 0.021639 (W 0.00462602) | 4.91399 ± 0.554977 (W 0.0826642) | - | 🔵 faer-fork 3.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.23405 ± 0.00750337 (W 0.00401256) | 3.16583 ± 0.0163917 (W 0.0185522) | 2.03834 ± 0.0200183 (W 0.0348192) | 🔵 faer-fork 2.57× faster than `faer-upstream` | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.24315 ± 0.00932146 (W 0.00495816) | 3.19495 ± 0.00813355 (W 0.00839711) | - | 🔵 faer-fork 2.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.84581 ± 0.0263826 (W 0.00495083) | 10.8125 ± 0.216399 (W 0.13689) | 7.62033 ± 0.0456351 (W 0.0591855) | 🔵 faer-fork 2.81× faster than `faer-upstream` | 🔵 faer-fork 1.98× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.90393 ± 0.0238639 (W 0.0167829) | 10.4226 ± 0.0856567 (W 0.0269346) | - | 🔵 faer-fork 2.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.9334 ± 0.0464931 (W 0.0399411) | 41.4337 ± 0.415962 (W 1.18402) | 22.3424 ± 0.331804 (W 0.103177) | 🔵 faer-fork 2.45× faster than `faer-upstream` | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 16.8556 ± 0.0333533 (W 0.01926) | 40.9627 ± 0.952076 (W 0.857246) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.466942 ± 0.000738801 (W 0.0016953) | 0.440031 ± 0.000863655 (W 0.00150093) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.448342 ± 0.000656727 (W 0.00236355) | 0.42411 ± 0.00200343 (W 0.00173869) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.02904 ± 0.0255006 (W 0.00796146) | 3.84257 ± 0.0098477 (W 0.00319616) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.44678 ± 0.00666721 (W 0.00549126) | 3.61954 ± 0.0146155 (W 0.00650241) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 23.7341 ± 0.0854532 (W 0.0253672) | 29.5178 ± 0.0842306 (W 0.0377816) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 25.716 ± 0.166735 (W 0.0328815) | 28.4831 ± 0.0362018 (W 0.0349621) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.103535 ± 0.00108099 (W 0.000740723) | 0.310025 ± 0.155154 (W 0.0310043) | - | 🔵 faer-fork 2.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.084494 ± 0.00316167 (W 0.000816658) | 0.61967 ± 0.00747698 (W 0.0382268) | - | 🔵 faer-fork 7.33× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.343016 ± 0.00234506 (W 0.00301226) | 0.84778 ± 0.0194186 (W 0.0331057) | - | 🔵 faer-fork 2.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.290184 ± 0.00856841 (W 0.00359866) | 1.88505 ± 0.0098597 (W 0.0353629) | - | 🔵 faer-fork 6.50× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.29189 ± 0.01337 (W 0.0421482) | 3.62466 ± 0.716088 (W 0.589693) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.29199 ± 0.0184074 (W 0.034698) | 4.86596 ± 0.378001 (W 0.553934) | - | 🔵 faer-fork 3.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.153077 ± 0.000457111 (W 0.00175012) | 0.252832 ± 0.0113298 (W 0.0230314) | 0.195079 ± 0.00036756 (W 0.00156444) | 🔵 faer-fork 1.65× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.182809 ± 0.00178244 (W 0.00201231) | 0.264593 ± 0.00555343 (W 0.00980719) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.603129 ± 0.0163769 (W 0.00448466) | 2.27558 ± 0.359336 (W 0.0869113) | 1.08413 ± 0.273458 (W 0.0140964) | 🔵 faer-fork 3.77× faster than `faer-upstream` | 🔵 faer-fork 1.80× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n256/b1 [api-inclusive]` | 1.17402 ± 0.026716 (W 0.00607223) | 1.59423 ± 18.8821 (W 0.0519881) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 2.27736 ± 0.106378 (W 0.0107563) | 9.31399 ± 0.81185 (W 0.741239) | 3.37096 ± 0.0396873 (W 0.0417316) | 🔵 faer-fork 4.09× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n512/b1 [api-inclusive]` | 4.79343 ± 0.155914 (W 0.00909964) | 5.65644 ± 2.1079 (W 1.29549) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.6424 ± 0.0163423 (W 0.00420417) | 2.64802 ± 0.0113972 (W 0.0919983) | 0.547296 ± 0.0682381 (W 0.010267) | 🔵 faer-fork 4.12× faster than `faer-upstream` | 🔴 faer-fork 1.17× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.872601 ± 0.0347733 (W 0.00269287) | 2.02079 ± 0.639697 (W 4.02798) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.33675 ± 0.00639351 (W 0.00359201) | 9.35723 ± 0.0223494 (W 0.133278) | 2.11096 ± 0.168772 (W 0.0328543) | 🔵 faer-fork 7.00× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n256/b1 [api-inclusive]` | 2.33316 ± 0.00929663 (W 0.0106757) | 6.74262 ± 3.05292 (W 0.259351) | - | 🔵 faer-fork 2.89× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 3.94944 ± 0.0131207 (W 0.012383) | 36.6419 ± 1.63511 (W 2.53875) | 6.42323 ± 0.158446 (W 0.11445) | 🔵 faer-fork 9.28× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 7.71617 ± 0.0225289 (W 0.0262056) | 33.2676 ± 1.35557 (W 6.67176) | - | 🔵 faer-fork 4.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.382838 ± 0.00652638 (W 0.00160402) | 1.54833 ± 0.177796 (W 0.0708231) | 0.445611 ± 0.0491702 (W 0.00994296) | 🔵 faer-fork 4.04× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n256/b1` | 1.12525 ± 0.0210019 (W 0.00362077) | 6.58228 ± 1.35951 (W 0.12131) | 1.3932 ± 0.128665 (W 0.011831) | 🔵 faer-fork 5.85× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n512/b1` | 3.04399 ± 0.152979 (W 0.00704969) | 13.7945 ± 6.20342 (W 2.53349) | 4.32144 ± 0.131152 (W 0.0380799) | 🔵 faer-fork 4.53× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n128/b1` | 0.155128 ± 0.00247038 (W 0.00200844) | 0.266442 ± 0.0465591 (W 0.0256337) | 0.19397 ± 8.96679e-05 (W 0.00156226) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.2037 ± 0.00661049 (W 0.00205753) | 8.95819 ± 5.02202 (W 0.0299721) | - | 🔵 faer-fork 43.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.662432 ± 0.0255125 (W 0.00401979) | 2.05846 ± 0.21638 (W 0.04525) | 1.12182 ± 0.129702 (W 0.0216995) | 🔵 faer-fork 3.11× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.838359 ± 0.0137107 (W 0.00318788) | 1.4254 ± 0.149998 (W 0.0366694) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 2.56506 ± 0.197169 (W 0.016276) | 7.98918 ± 0.00526197 (W 0.11472) | 3.59514 ± 1.01801 (W 0.0374595) | 🔵 faer-fork 3.11× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n512/b1 [api-inclusive]` | 3.18521 ± 0.0900428 (W 0.0197608) | 6.25133 ± 3.00618 (W 0.163294) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 3.01996 ± 0.0146887 (W 0.00810874) | 4.48387 ± 0.212383 (W 0.137128) | 4.08355 ± 0.00634105 (W 0.00719566) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n128/b1 [api-inclusive]` | 3.05907 ± 0.00351136 (W 0.0102331) | 3.98192 ± 0.013208 (W 0.0308833) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 11.2517 ± 0.0604171 (W 0.0171414) | 24.2236 ± 0.939954 (W 1.08326) | 16.9648 ± 0.107729 (W 0.0215337) | 🔵 faer-fork 2.15× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n256/b1 [api-inclusive]` | 11.4555 ± 0.0197486 (W 0.0138768) | 20.3879 ± 0.604673 (W 0.114011) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1` | 45.8505 ± 0.0622208 (W 0.0582388) | 99.396 ± 1.89853 (W 13.9719) | 82.1162 ± 0.496346 (W 0.181167) | 🔵 faer-fork 2.17× faster than `faer-upstream` | 🔵 faer-fork 1.79× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 46.5225 ± 0.108101 (W 0.0275513) | 101.234 ± 3.18945 (W 35.528) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 1.13347 ± 0.00134616 (W 0.00123088) | 2.57692 ± 0.00628346 (W 0.0095169) | 1.60801 ± 0.0147458 (W 0.00441472) | 🔵 faer-fork 2.27× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.13672 ± 0.00123573 (W 0.0011541) | 2.99693 ± 0.245202 (W 0.00670201) | - | 🔵 faer-fork 2.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 6.05586 ± 0.0286804 (W 0.0111287) | 14.7474 ± 0.109221 (W 0.0605864) | 8.41104 ± 0.04898 (W 0.0211991) | 🔵 faer-fork 2.44× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.17054 ± 0.0365669 (W 0.0100502) | 14.7333 ± 0.0191196 (W 0.087872) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 27.3841 ± 0.0794727 (W 0.0623327) | 75.8378 ± 1.76262 (W 48.686) | 36.7599 ± 0.398157 (W 0.0898632) | 🔵 faer-fork 2.77× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 26.9027 ± 0.137639 (W 0.0901557) | 75.4411 ± 1.58424 (W 6.93894) | - | 🔵 faer-fork 2.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00858 ± 1.95021e-05 (W 3.93865e-05) | 0.006101 ± 1.73205e-05 (W 2.33495e-05) | - | 🔴 faer-fork 1.41× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.020571 ± 3.51188e-05 (W 5.84808e-05) | 0.01578 ± 0.000339368 (W 0.000156193) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.053522 ± 0.000792244 (W 0.000442523) | 0.065884 ± 1.2467 (W 0.00495318) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.073924 ± 0.0018337 (W 9.48119e-05) | 0.141386 ± 0.0118766 (W 0.00581318) | 0.133297 ± 6.71342e-05 (W 0.00121206) | 🔵 faer-fork 1.91× faster than `faer-upstream` | 🔵 faer-fork 1.80× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.082644 ± 0.00187777 (W 0.000100718) | 0.117316 ± 0.000917137 (W 0.0081477) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.309985 ± 0.0210447 (W 0.00259591) | 0.764316 ± 0.0326541 (W 0.0549738) | 0.565507 ± 0.0224312 (W 0.0367623) | 🔵 faer-fork 2.47× faster than `faer-upstream` | 🔵 faer-fork 1.82× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.435661 ± 0.0667237 (W 0.00342376) | 0.627499 ± 0.253743 (W 0.124867) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.22865 ± 0.0287447 (W 0.0116924) | 2.96431 ± 0.0509566 (W 0.0738685) | 1.50032 ± 0.0392602 (W 0.125202) | 🔵 faer-fork 2.41× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.81278 ± 0.0403997 (W 0.00660469) | 3.18138 ± 1.61742 (W 0.124799) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.24877 ± 0.0122031 (W 0.00467005) | 3.26345 ± 0.0184355 (W 0.0322021) | 2.36301 ± 0.0148135 (W 0.0334978) | 🔵 faer-fork 2.61× faster than `faer-upstream` | 🔵 faer-fork 1.89× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.27199 ± 0.0163507 (W 0.00702011) | 3.23381 ± 0.0870242 (W 0.949786) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.95014 ± 0.0217916 (W 0.00974045) | 14.1903 ± 1.52483 (W 0.408363) | 8.08561 ± 0.0346976 (W 0.0674851) | 🔵 faer-fork 3.59× faster than `faer-upstream` | 🔵 faer-fork 2.05× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `eigh/n256/b1 [api-inclusive]` | 4.05113 ± 0.0398529 (W 0.0144492) | 10.9328 ± 0.0929419 (W 0.93799) | - | 🔵 faer-fork 2.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 20.4793 ± 0.173273 (W 0.0650138) | 60.9063 ± 1.06957 (W 6.71038) | 23.3635 ± 0.115076 (W 0.144691) | 🔵 faer-fork 2.97× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 21.6972 ± 0.16387 (W 0.052152) | 56.0164 ± 5.13467 (W 4.92206) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1` | 0.469052 ± 0.000738196 (W 0.00167964) | 0.440211 ± 0.000719884 (W 0.00179316) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.449201 ± 0.00277979 (W 0.00223289) | 0.42578 ± 0.00133365 (W 0.00199958) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.11579 ± 0.0075274 (W 0.00802565) | 3.82097 ± 0.0379321 (W 0.00404673) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.49752 ± 0.0146338 (W 0.00995601) | 3.59424 ± 0.0111358 (W 0.00507689) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.106 ± 0.043948 (W 0.0498664) | 32.4415 ± 3.62186 (W 0.296744) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 27.5687 ± 0.247552 (W 0.0499886) | 29.571 ± 0.0437514 (W 0.0968056) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.109216 ± 0.0152979 (W 0.000560704) | 0.611639 ± 0.144497 (W 0.0780173) | - | 🔵 faer-fork 5.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.105405 ± 0.00904908 (W 0.00193598) | 0.6427 ± 0.0571774 (W 0.072514) | - | 🔵 faer-fork 6.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.326375 ± 0.0279904 (W 0.00457886) | 1.96096 ± 0.440982 (W 0.173672) | - | 🔵 faer-fork 6.01× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.279003 ± 0.00371041 (W 0.00579784) | 2.22415 ± 0.0302141 (W 0.0287873) | - | 🔵 faer-fork 7.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.48665 ± 0.0133779 (W 0.0264008) | 6.06054 ± 0.0566195 (W 0.121681) | - | 🔵 faer-fork 4.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.46619 ± 0.0194542 (W 0.0274545) | 6.11518 ± 0.0298967 (W 0.0833357) | - | 🔵 faer-fork 4.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.158668 ± 0.00453072 (W 0.00453144) | 0.243272 ± 0.0928666 (W 0.0151424) | 0.195619 ± 0.000610246 (W 0.00180163) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.258443 ± 0.00171424 (W 0.00900999) | 0.257142 ± 0.0078955 (W 0.0190854) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.563167 ± 0.053389 (W 0.00463227) | 2.99735 ± 0.147925 (W 0.137447) | 1.06389 ± 0.056749 (W 0.0136168) | 🔵 faer-fork 5.32× faster than `faer-upstream` | 🔵 faer-fork 1.89× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n256/b1 [api-inclusive]` | 1.18039 ± 0.0333882 (W 0.00457214) | 1.42956 ± 0.230244 (W 0.0361092) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 2.3741 ± 0.163733 (W 0.0133525) | 10.8271 ± 0.320907 (W 0.089656) | 4.19129 ± 0.15614 (W 0.0944232) | 🔵 faer-fork 4.56× faster than `faer-upstream` | 🔵 faer-fork 1.77× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 4.7469 ± 0.0520626 (W 0.0199415) | 5.93103 ± 0.768397 (W 0.521913) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.641691 ± 0.0190992 (W 0.00452016) | 2.99712 ± 0.176923 (W 0.103478) | 0.461372 ± 0.0120008 (W 0.00765691) | 🔵 faer-fork 4.67× faster than `faer-upstream` | 🔴 faer-fork 1.39× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.858391 ± 0.0231577 (W 0.00254865) | 3.76139 ± 0.647363 (W 0.115842) | - | 🔵 faer-fork 4.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.39456 ± 0.00180913 (W 0.00530522) | 11.5676 ± 0.23978 (W 0.265871) | 1.97698 ± 0.0541427 (W 0.0216451) | 🔵 faer-fork 8.29× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 2.46398 ± 0.0115742 (W 0.0114614) | 14.7601 ± 3.77522 (W 0.320691) | - | 🔵 faer-fork 5.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 4.04785 ± 0.0365181 (W 0.0118044) | 42.1777 ± 1.71559 (W 0.491056) | 7.28051 ± 0.0537256 (W 0.144897) | 🔵 faer-fork 10.42× faster than `faer-upstream` | 🔵 faer-fork 1.80× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 7.90169 ± 0.0107317 (W 0.021492) | 33.8915 ± 1.76754 (W 1.56097) | - | 🔵 faer-fork 4.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.396238 ± 0.0266218 (W 0.00652746) | 2.31199 ± 0.246144 (W 0.0714724) | 0.317975 ± 0.00686906 (W 0.0049701) | 🔵 faer-fork 5.83× faster than `faer-upstream` | 🔴 faer-fork 1.25× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n256/b1` | 1.14841 ± 0.0120939 (W 0.00305409) | 8.23962 ± 1.98622 (W 0.184452) | 1.5059 ± 0.0704276 (W 0.0310774) | 🔵 faer-fork 7.17× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n512/b1` | 3.16715 ± 0.165938 (W 0.00591085) | 26.0018 ± 0.110744 (W 0.735378) | 5.64986 ± 0.206848 (W 0.083555) | 🔵 faer-fork 8.21× faster than `faer-upstream` | 🔵 faer-fork 1.78× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n128/b1` | 0.167348 ± 0.00474736 (W 0.00200437) | 0.406029 ± 0.0388243 (W 0.0469518) | 0.19448 ± 0.000115832 (W 0.00167315) | 🔵 faer-fork 2.43× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.265262 ± 0.00492619 (W 0.00816575) | 0.312845 ± 5.02696 (W 0.026218) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.654211 ± 0.00157645 (W 0.00357973) | 2.60372 ± 0.0809169 (W 0.0869977) | 1.19792 ± 0.0576226 (W 0.02337) | 🔵 faer-fork 3.98× faster than `faer-upstream` | 🔵 faer-fork 1.83× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.84029 ± 0.0130913 (W 0.00271561) | 2.06738 ± 0.436489 (W 0.259621) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 2.12415 ± 0.0168141 (W 0.00831954) | 9.5399 ± 0.0702544 (W 0.112574) | 4.48588 ± 1.09468 (W 0.0482728) | 🔵 faer-fork 4.49× faster than `faer-upstream` | 🔵 faer-fork 2.11× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n512/b1 [api-inclusive]` | 3.09442 ± 0.0146705 (W 0.0164142) | 6.84082 ± 1.07346 (W 0.294836) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 3.06251 ± 0.16435 (W 0.0121677) | 4.83542 ± 0.542956 (W 0.135519) | 4.23549 ± 0.336838 (W 0.0320843) | 🔵 faer-fork 1.58× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/n128/b1 [api-inclusive]` | 3.08935 ± 0.298328 (W 0.0104104) | 4.26171 ± 0.131856 (W 0.0535539) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n256/b1` | 11.4834 ± 0.0228905 (W 0.00874254) | 48.9836 ± 4.65187 (W 26.4376) | 18.4212 ± 1.26007 (W 0.19827) | 🔵 faer-fork 4.27× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/n256/b1 [api-inclusive]` | 11.7919 ± 0.267945 (W 0.0301654) | 23.2934 ± 1.55103 (W 1.93715) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1` | 49.8677 ± 0.102567 (W 0.0756591) | 126.416 ± 1.17006 (W 25.4719) | 87.1147 ± 0.978647 (W 0.44373) | 🔵 faer-fork 2.54× faster than `faer-upstream` | 🔵 faer-fork 1.75× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 50.094 ± 0.524108 (W 0.0986515) | 126.796 ± 4.04641 (W 45.0491) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 1.13044 ± 0.000688012 (W 0.000799816) | 2.57257 ± 0.00830456 (W 0.0289684) | 1.63838 ± 0.00187623 (W 0.00510171) | 🔵 faer-fork 2.28× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.1357 ± 0.00126028 (W 0.00145227) | 2.57234 ± 0.247716 (W 0.00127116) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 6.11245 ± 0.0100103 (W 0.00759377) | 19.0719 ± 1.65224 (W 0.25307) | 8.77197 ± 0.125043 (W 0.0333366) | 🔵 faer-fork 3.12× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/n256/b1 [api-inclusive]` | 6.16595 ± 0.0204382 (W 0.0109115) | 15.1857 ± 0.237017 (W 0.0496444) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 30.5633 ± 0.265873 (W 0.121607) | 123.171 ± 48.6384 (W 102.018) | 43.2226 ± 11.9913 (W 0.828971) | 🔵 faer-fork 4.03× faster than `faer-upstream` | 🔵 faer-fork 1.41× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 30.107 ± 0.528549 (W 0.114009) | 90.9494 ± 5.24847 (W 49.1907) | - | 🔵 faer-fork 3.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n128/b1` | 0.0079 ± 1.95021e-05 (W 2.54028e-05) | 0.0055 ± 2.08167e-05 (W 2.51635e-05) | - | 🔴 faer-fork 1.44× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.021451 ± 8.62168e-05 (W 4.61519e-05) | 0.016611 ± 0.00015201 (W 8.38403e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.056283 ± 8.32666e-05 (W 0.000299268) | 0.096444 ± 1.26759 (W 0.00990216) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.20742 ± 0.00332025 (W 0.00269861) | 0.383018 ± 0.003007 (W 0.010857) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.232731 ± 0.000281484 (W 0.00205802) | 4.58712 ± 2.50618 (W 0.00816144) | - | 🔵 faer-fork 19.71× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.729524 ± 0.00357263 (W 0.0045121) | 1.31894 ± 0.0245194 (W 0.0126038) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.07288 ± 0.000755996 (W 0.00466899) | 24.2317 ± 14.4705 (W 0.028481) | - | 🔵 faer-fork 22.59× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 2.12826 ± 0.0095893 (W 0.0145301) | 3.49184 ± 0.033806 (W 0.025839) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 3.59146 ± 0.0106768 (W 0.018276) | 4.87911 ± 0.116108 (W 0.0459949) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.84121 ± 0.0092875 (W 0.00375265) | 4.18058 ± 0.00123016 (W 0.0162493) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.8822 ± 0.00725988 (W 0.00552409) | 4.13883 ± 0.00723988 (W 0.0239479) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.69785 ± 0.0298481 (W 0.0134775) | 16.9769 ± 0.0402601 (W 0.133459) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.71535 ± 0.0353369 (W 0.0115831) | 17.1376 ± 0.104951 (W 0.114257) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.8975 ± 0.126117 (W 0.0521442) | 56.0097 ± 0.753888 (W 0.714838) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 26.3973 ± 0.0302188 (W 0.0427113) | 55.7717 ± 0.278269 (W 0.594226) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 1.02348 ± 0.0654785 (W 0.00352072) | 1.09771 ± 0.0062171 (W 0.00254355) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 1.20284 ± 0.0362036 (W 0.00626875) | 1.12997 ± 0.0162091 (W 0.00283446) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 6.88863 ± 0.0315169 (W 0.0197456) | 11.7022 ± 0.042939 (W 0.00970921) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 7.88246 ± 0.0168433 (W 0.011816) | 11.3902 ± 0.0663505 (W 0.0203157) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 27.3764 ± 0.0474124 (W 0.0280236) | 44.9937 ± 0.111331 (W 0.0224025) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 32.1558 ± 0.0305936 (W 0.0392803) | 44.2857 ± 0.22802 (W 0.071027) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.257562 ± 0.00322062 (W 0.00513909) | 0.519195 ± 0.0355875 (W 0.0429903) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.257252 ± 0.0037896 (W 0.0033267) | 0.646511 ± 0.171849 (W 0.0939703) | - | 🔵 faer-fork 2.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.04775 ± 0.0125296 (W 0.00816513) | 1.67068 ± 0.0547683 (W 0.0604225) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.06518 ± 0.00923656 (W 0.00988133) | 2.04599 ± 0.354314 (W 0.0649045) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 5.5077 ± 0.0145848 (W 0.0329449) | 7.18512 ± 0.0709704 (W 0.335277) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 5.50443 ± 0.0619416 (W 0.0437072) | 7.03575 ± 0.156925 (W 0.318908) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.376117 ± 0.00542825 (W 0.00515646) | 0.881942 ± 0.00850185 (W 0.00786842) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.640621 ± 0.0103814 (W 0.00465037) | 0.515064 ± 7.11512 (W 0.00622442) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 1.38261 ± 0.00218443 (W 0.00754508) | 6.66546 ± 0.00778996 (W 0.0162961) | - | 🔵 faer-fork 4.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.55358 ± 0.00407299 (W 0.00906996) | 2.91724 ± 0.014566 (W 0.0200575) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.05042 ± 0.0154177 (W 0.0178444) | 18.5621 ± 0.267782 (W 0.0822809) | - | 🔵 faer-fork 3.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 10.1709 ± 0.0338194 (W 0.0209039) | 10.7317 ± 0.0383913 (W 0.0425539) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.08342 ± 0.00874052 (W 0.00445102) | 3.32111 ± 0.0287816 (W 0.0922719) | - | 🔵 faer-fork 3.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 1.43406 ± 0.00398121 (W 0.00536791) | 2.87055 ± 0.239663 (W 0.0988945) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 3.08404 ± 0.0044112 (W 0.00771528) | 11.71 ± 0.396635 (W 0.273017) | - | 🔵 faer-fork 3.80× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 4.70225 ± 0.0145739 (W 0.0176921) | 9.3674 ± 0.480046 (W 0.055249) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 11.6986 ± 0.016291 (W 0.0369261) | 38.8621 ± 0.315443 (W 0.576303) | - | 🔵 faer-fork 3.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 17.9182 ± 0.0310474 (W 0.0361074) | 36.1067 ± 1.70062 (W 0.569151) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.665951 ± 0.00243265 (W 0.00457902) | 1.97277 ± 0.132656 (W 0.0238898) | - | 🔵 faer-fork 2.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n256/b1` | 2.0198 ± 0.00138405 (W 0.00557211) | 8.1178 ± 0.577963 (W 0.148831) | - | 🔵 faer-fork 4.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 6.32003 ± 0.0479999 (W 0.00555365) | 30.4494 ± 0.663553 (W 0.614178) | - | 🔵 faer-fork 4.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.330255 ± 0.00245146 (W 0.0033292) | 0.529095 ± 0.0108904 (W 0.0108927) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.672702 ± 0.00176162 (W 0.00457813) | 0.509514 ± 0.0106728 (W 0.00389193) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.45968 ± 0.00298863 (W 0.0135484) | 4.22778 ± 0.0289946 (W 0.0113541) | - | 🔵 faer-fork 2.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.74128 ± 0.0200259 (W 0.00881775) | 2.78549 ± 11.395 (W 2.54413) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 5.07962 ± 0.0368685 (W 0.0202594) | 13.4393 ± 0.1101 (W 0.0198566) | - | 🔵 faer-fork 2.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 6.74245 ± 0.0124384 (W 0.0217329) | 10.2456 ± 0.46298 (W 0.0645281) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 4.59977 ± 0.0214915 (W 0.0237921) | 6.0583 ± 0.0116081 (W 0.0326454) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 4.68801 ± 0.016172 (W 0.00824064) | 6.12322 ± 0.0299984 (W 0.0569208) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 18.2153 ± 0.0921033 (W 0.0246439) | 32.8575 ± 0.0451818 (W 0.126598) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 18.5208 ± 0.0833677 (W 0.037108) | 33.2755 ± 0.201722 (W 0.100539) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 71.4503 ± 0.169888 (W 0.078804) | 104.047 ± 0.463789 (W 0.179861) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 72.4653 ± 0.180685 (W 0.0634482) | 104.829 ± 0.199719 (W 1.92673) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 2.50742 ± 0.00654996 (W 0.0106166) | 4.08715 ± 0.00563568 (W 0.0397076) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 2.52025 ± 0.0187199 (W 0.00202655) | 4.12483 ± 0.858602 (W 0.0248844) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 10.7891 ± 0.0526517 (W 0.0250498) | 25.8353 ± 0.0995579 (W 0.0464337) | - | 🔵 faer-fork 2.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 10.9381 ± 0.029378 (W 0.0138199) | 25.7575 ± 0.0504991 (W 0.0521618) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 44.399 ± 0.469598 (W 0.0224019) | 74.8246 ± 0.575705 (W 0.0994657) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 44.0316 ± 1.44837 (W 0.0624911) | 76.0405 ± 0.772848 (W 0.156287) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n128/b1` | 0.01642 ± 0.000102357 (W 6.78985e-05) | 0.012781 ± 0.000301143 (W 7.54407e-05) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.040752 ± 0.00024173 (W 0.000204377) | 0.042272 ± 0.000346554 (W 0.000141421) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.096164 ± 0.000561282 (W 0.000656994) | 0.099565 ± 1.24934 (W 0.00211263) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.20463 ± 0.00269909 (W 0.00233966) | 0.396209 ± 0.00633423 (W 0.00873006) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.233841 ± 0.000230723 (W 0.00176807) | 0.255462 ± 2.51543 (W 0.00638733) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.790407 ± 0.045034 (W 0.00423756) | 1.90929 ± 0.15044 (W 0.0616927) | - | 🔵 faer-fork 2.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 1.12281 ± 0.0189436 (W 0.00686317) | 1.40013 ± 0.16694 (W 0.0826496) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 2.10536 ± 0.00725942 (W 0.0124678) | 5.86068 ± 0.158807 (W 0.20051) | - | 🔵 faer-fork 2.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 3.70562 ± 0.0113139 (W 0.0157019) | 5.00203 ± 0.190903 (W 0.0395209) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.78384 ± 0.00818556 (W 0.00790918) | 4.20772 ± 0.011288 (W 0.0152066) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.81886 ± 0.0232336 (W 0.00577849) | 4.1597 ± 8.03764 (W 0.022738) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1` | 7.75012 ± 0.0238193 (W 0.0212641) | 17.0536 ± 0.0253652 (W 0.065239) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 7.78651 ± 0.020713 (W 0.0250404) | 17.2411 ± 0.046121 (W 10.2178) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.4309 ± 0.120106 (W 0.0709384) | 59.6327 ± 7.58856 (W 19.2381) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n512/b1 [api-inclusive]` | 29.6334 ± 0.187341 (W 0.0353766) | 59.5813 ± 0.45631 (W 1.91549) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 1.06196 ± 0.0371367 (W 0.00377211) | 1.09902 ± 0.00337536 (W 0.00286425) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 1.20359 ± 0.0175516 (W 0.00705745) | 1.13037 ± 0.00999646 (W 0.00465831) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 7.16117 ± 0.0202807 (W 0.0119754) | 11.6792 ± 0.0194623 (W 0.0104913) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 8.08599 ± 0.0485768 (W 0.0103063) | 11.4155 ± 0.135983 (W 0.0157068) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 27.2613 ± 0.134884 (W 0.036476) | 44.3494 ± 0.0868028 (W 0.0309238) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 31.6351 ± 0.123846 (W 0.0370638) | 44.1959 ± 0.402267 (W 0.0242763) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.214051 ± 0.0209649 (W 0.00631388) | 0.473913 ± 0.165251 (W 0.0349001) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.21328 ± 0.0173749 (W 0.00435396) | 0.879962 ± 0.130083 (W 0.137191) | - | 🔵 faer-fork 4.13× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.84727 ± 0.0649465 (W 0.0120607) | 3.17865 ± 0.0568436 (W 0.140415) | - | 🔵 faer-fork 3.75× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.817638 ± 0.010062 (W 0.0103688) | 3.31947 ± 0.131539 (W 0.107675) | - | 🔵 faer-fork 4.06× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.58653 ± 0.00497717 (W 0.0292035) | 5.54526 ± 0.0252617 (W 0.0366583) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.56494 ± 0.0164103 (W 0.0371506) | 5.51262 ± 0.00572178 (W 0.0480189) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.378588 ± 0.00238886 (W 0.00203598) | 0.886192 ± 0.00736723 (W 0.00601647) | - | 🔵 faer-fork 2.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.652001 ± 0.00398924 (W 0.00455434) | 0.546316 ± 3.72874 (W 0.0103183) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 1.34363 ± 0.00580779 (W 0.0059954) | 6.67361 ± 0.0480597 (W 0.0407412) | - | 🔵 faer-fork 4.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.5315 ± 0.00972448 (W 0.00657467) | 3.05298 ± 0.0141441 (W 0.0155731) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.68996 ± 0.0206642 (W 0.0138092) | 19.6389 ± 0.961232 (W 0.420514) | - | 🔵 faer-fork 4.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 10.0884 ± 0.255686 (W 0.0299931) | 10.9192 ± 0.49266 (W 0.357193) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 1.05486 ± 0.0151919 (W 0.00357385) | 4.80126 ± 0.156078 (W 0.163559) | - | 🔵 faer-fork 4.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 1.41043 ± 0.0150581 (W 0.00538139) | 2.89043 ± 0.641483 (W 0.143991) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 2.85037 ± 0.0151782 (W 0.00536849) | 17.3296 ± 1.52143 (W 0.354288) | - | 🔵 faer-fork 6.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 4.60209 ± 0.0072876 (W 0.0117336) | 18.1832 ± 5.0992 (W 2.04066) | - | 🔵 faer-fork 3.95× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 9.67862 ± 0.0718254 (W 0.0249583) | 53.3552 ± 1.64285 (W 1.33257) | - | 🔵 faer-fork 5.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 16.8859 ± 0.0285546 (W 0.055598) | 45.6154 ± 4.54652 (W 5.43385) | - | 🔵 faer-fork 2.70× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.675352 ± 0.00942298 (W 0.00435288) | 2.43686 ± 0.158924 (W 0.20907) | - | 🔵 faer-fork 3.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n256/b1` | 2.06017 ± 0.00642661 (W 0.00658603) | 8.63466 ± 2.31653 (W 0.114324) | - | 🔵 faer-fork 4.19× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 6.11705 ± 0.0596186 (W 0.00506131) | 42.0697 ± 6.29543 (W 4.81231) | - | 🔵 faer-fork 6.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.355677 ± 0.00243185 (W 0.00368765) | 0.559206 ± 0.00304516 (W 0.0117187) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.680392 ± 0.00696054 (W 0.00384332) | 0.522825 ± 7.78864 (W 0.00512941) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 1.45461 ± 0.0983557 (W 0.0103935) | 4.35442 ± 0.260774 (W 0.0194039) | - | 🔵 faer-fork 2.99× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 1.73219 ± 0.0929064 (W 0.00437295) | 2.85334 ± 0.0603054 (W 6.14955) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 4.77704 ± 0.00948108 (W 0.0240785) | 17.0212 ± 0.594917 (W 0.524472) | - | 🔵 faer-fork 3.56× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1 [api-inclusive]` | 6.81612 ± 0.330444 (W 0.0611775) | 10.4919 ± 0.886139 (W 0.146352) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 4.63525 ± 0.0433696 (W 0.0185795) | 6.34749 ± 0.198318 (W 0.0895718) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1 [api-inclusive]` | 4.70239 ± 0.0297188 (W 0.0336541) | 6.22593 ± 0.0329065 (W 0.0499435) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 19.3069 ± 0.0525874 (W 0.0286262) | 34.948 ± 1.04497 (W 0.489523) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n256/b1 [api-inclusive]` | 19.5341 ± 0.000365965 (W 0.0225584) | 33.9824 ± 0.182272 (W 0.0763971) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 75.2143 ± 1.14427 (W 0.0817552) | 105.208 ± 0.77441 (W 3.807) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 76.0608 ± 0.283452 (W 0.0594035) | 107.599 ± 5.93615 (W 7.04927) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 2.54209 ± 0.0949638 (W 0.013574) | 4.10442 ± 0.0102075 (W 0.0273515) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1 [api-inclusive]` | 2.54682 ± 0.0904734 (W 0.0131321) | 4.12411 ± 0.860381 (W 0.0171743) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 11.4901 ± 0.0123325 (W 0.0202413) | 26.4673 ± 0.687041 (W 0.0997122) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 11.5061 ± 0.0508816 (W 0.0215669) | 26.5046 ± 0.0569605 (W 16.6807) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 47.8311 ± 1.11576 (W 0.0686122) | 82.7052 ± 1.35028 (W 20.2077) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 47.8435 ± 0.289661 (W 0.0555584) | 81.8272 ± 0.904468 (W 48.8728) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.016621 ± 0.000188486 (W 5.1674e-05) | 0.013111 ± 0.000162891 (W 4.84386e-05) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.041122 ± 0.000390171 (W 0.000195269) | 0.042342 ± 3.4641e-05 (W 0.000168819) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.096645 ± 0.000168226 (W 0.00092701) | 2.30326 ± 2.52287 (W 0.00519181) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.203979 ± 0.000945653 (W 0.00173412) | 0.401539 ± 0.0205219 (W 0.0156771) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.236911 ± 0.00200291 (W 0.00181733) | 0.267903 ± 2.51611 (W 0.0157932) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.774056 ± 0.0163959 (W 0.0110576) | 2.68694 ± 0.0153692 (W 0.132182) | - | 🔵 faer-fork 3.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.11459 ± 0.0355898 (W 0.0133019) | 1.53785 ± 0.240957 (W 0.0419617) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 2.14744 ± 0.104536 (W 0.0152162) | 7.4878 ± 0.175068 (W 0.107379) | - | 🔵 faer-fork 3.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 3.72792 ± 0.0558921 (W 0.0328119) | 5.00845 ± 0.143718 (W 0.174657) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.87633 ± 0.014463 (W 0.0107966) | 4.24634 ± 0.0283908 (W 0.0315734) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.90449 ± 0.012176 (W 0.00945105) | 4.21013 ± 0.0161302 (W 0.0337836) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.46125 ± 0.0673755 (W 0.0276416) | 18.7738 ± 0.570212 (W 0.251741) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1 [api-inclusive]` | 9.85849 ± 0.104894 (W 0.0344103) | 17.5483 ± 0.0214586 (W 0.0774761) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 40.0151 ± 0.167578 (W 0.173377) | 76.7351 ± 3.22905 (W 28.8878) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n512/b1 [api-inclusive]` | 40.9943 ± 0.0822202 (W 0.0692157) | 74.906 ± 2.50549 (W 28.7553) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1` | 1.12253 ± 0.0981575 (W 0.0131114) | 1.10121 ± 0.0034126 (W 0.00257136) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 1.23291 ± 0.0731622 (W 0.00884087) | 1.13289 ± 0.021598 (W 0.00339078) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 7.53243 ± 0.0289776 (W 0.0132951) | 11.69 ± 0.0277139 (W 0.0172855) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 8.53629 ± 0.00856761 (W 0.019212) | 11.4266 ± 0.017753 (W 0.0131137) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 28.3132 ± 0.114665 (W 0.0416314) | 47.1883 ± 1.31233 (W 0.646686) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 32.9168 ± 0.202707 (W 0.0452189) | 44.3361 ± 0.168809 (W 1.84262) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.199769 ± 0.0208459 (W 0.00475123) | 0.953285 ± 0.276127 (W 0.100047) | - | 🔵 faer-fork 4.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.19736 ± 0.00420883 (W 0.0048073) | 0.966746 ± 0.0251242 (W 0.13814) | - | 🔵 faer-fork 4.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.975566 ± 0.0443376 (W 0.0333785) | 2.99353 ± 0.471373 (W 0.296765) | - | 🔵 faer-fork 3.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.764506 ± 0.0108882 (W 0.0155076) | 3.41066 ± 0.0204891 (W 0.0394632) | - | 🔵 faer-fork 4.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.62241 ± 0.0905689 (W 0.117715) | 5.10157 ± 0.0235918 (W 0.0526831) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.68151 ± 0.0278465 (W 0.103409) | 5.14467 ± 0.00979722 (W 0.0627925) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.387648 ± 0.004285 (W 0.0175593) | 0.909453 ± 0.005738 (W 0.00989507) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.681642 ± 0.00803323 (W 0.00892824) | 0.529405 ± 0.00159352 (W 0.00512173) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.32555 ± 0.030546 (W 0.0030693) | 7.77373 ± 0.692221 (W 0.191914) | - | 🔵 faer-fork 5.86× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 2.58626 ± 0.067512 (W 0.00911817) | 3.16808 ± 0.0348738 (W 0.0193468) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.53247 ± 0.0259525 (W 0.0130484) | 27.2828 ± 4.86841 (W 1.03208) | - | 🔵 faer-fork 6.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 10.0583 ± 0.312532 (W 0.0240384) | 11.2554 ± 0.771919 (W 0.13852) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 1.17494 ± 0.0264113 (W 0.00315102) | 5.07178 ± 0.250644 (W 0.150112) | - | 🔵 faer-fork 4.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 1.55337 ± 0.0233707 (W 0.0114452) | 5.98609 ± 1.58589 (W 0.192416) | - | 🔵 faer-fork 3.85× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 2.80179 ± 0.00792821 (W 0.00945448) | 20.6725 ± 0.313767 (W 0.366078) | - | 🔵 faer-fork 7.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 4.56181 ± 0.0203943 (W 0.0155899) | 23.1332 ± 0.160939 (W 1.19487) | - | 🔵 faer-fork 5.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 9.37981 ± 0.0753419 (W 0.023695) | 55.0403 ± 1.1734 (W 1.00713) | - | 🔵 faer-fork 5.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 16.7945 ± 0.0526123 (W 0.0337774) | 53.296 ± 4.09832 (W 1.36741) | - | 🔵 faer-fork 3.17× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.712424 ± 0.0160757 (W 0.00664081) | 3.72301 ± 0.740842 (W 0.203185) | - | 🔵 faer-fork 5.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n256/b1` | 2.092 ± 0.0519638 (W 0.00580928) | 11.129 ± 2.28227 (W 0.591953) | - | 🔵 faer-fork 5.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 6.37721 ± 0.080971 (W 0.0183888) | 41.1756 ± 8.99581 (W 7.0999) | - | 🔵 faer-fork 6.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.360137 ± 0.00604201 (W 0.00233993) | 0.593158 ± 0.0227111 (W 0.0150127) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.704033 ± 0.0177475 (W 0.00855407) | 0.526695 ± 3.77128 (W 0.0101708) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 1.43501 ± 0.0103136 (W 0.0142829) | 6.39172 ± 1.40262 (W 0.0686509) | - | 🔵 faer-fork 4.45× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 1.79679 ± 0.0548956 (W 0.0103173) | 2.9131 ± 0.0531762 (W 0.0466318) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 4.57208 ± 0.00752673 (W 0.0168943) | 25.3668 ± 0.171838 (W 0.344105) | - | 🔵 faer-fork 5.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.03887 ± 0.137212 (W 0.0335839) | 11.1248 ± 1.0258 (W 0.778648) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 4.80825 ± 0.115034 (W 0.0136304) | 6.6821 ± 0.254063 (W 0.138864) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1 [api-inclusive]` | 5.07743 ± 0.448078 (W 0.0194721) | 6.32962 ± 0.160449 (W 0.0780758) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n256/b1` | 20.6949 ± 0.10401 (W 0.055804) | 40.0956 ± 0.932265 (W 1.2215) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 20.7939 ± 0.0969214 (W 0.0388284) | 35.5096 ± 0.138621 (W 0.896514) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 87.3495 ± 0.169687 (W 0.100212) | 137.739 ± 1.63643 (W 34.2588) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 87.6128 ± 0.543374 (W 0.172205) | 142.161 ± 2.43747 (W 40.235) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 2.57907 ± 0.389212 (W 0.00942252) | 4.21166 ± 0.807166 (W 0.0366739) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1 [api-inclusive]` | 2.58948 ± 0.582992 (W 0.00619443) | 4.16044 ± 0.84795 (W 0.0410616) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 11.994 ± 0.0767222 (W 0.0170596) | 27.0392 ± 0.150135 (W 0.156792) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 12.1542 ± 0.0361751 (W 0.0188531) | 26.9634 ± 0.0489555 (W 0.0380488) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 56.8293 ± 0.261343 (W 0.0748392) | 188.401 ± 43.3009 (W 109.299) | - | 🔵 faer-fork 3.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 56.9254 ± 0.752882 (W 0.145509) | 102.776 ± 0.935561 (W 73.4748) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.016601 ± 4.16333e-05 (W 6.28267e-05) | 0.012821 ± 0.000105879 (W 7.11709e-05) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.041892 ± 0.00031607 (W 0.000260346) | 0.041991 ± 7.78267e-05 (W 0.000257644) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.104325 ± 0.000431084 (W 0.000660893) | 0.108145 ± 1.27527 (W 0.00194806) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.187078 ± 0.000186736 (W 0.00205549) | 0.197279 ± 0.00713353 (W 0.00628312) | 0.607199 ± 0.00249205 (W 0.00234424) | 🔵 faer-fork 1.05× faster than `faer-upstream` | 🔵 faer-fork 3.25× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.21502 ± 6.05833e-05 (W 0.00187652) | 0.242632 ± 0.00188107 (W 0.00414876) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.784517 ± 0.00321468 (W 0.0044688) | 0.849191 ± 0.0065645 (W 0.00976232) | 0.652191 ± 0.00854687 (W 0.0205308) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔴 faer-fork 1.20× slower than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 1.00989 ± 0.00584864 (W 0.00355974) | 0.981286 ± 0.0208396 (W 0.0203636) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.50548 ± 0.0117999 (W 0.0144936) | 2.85563 ± 0.0273119 (W 0.0294072) | 1.7354 ± 0.00857881 (W 0.0433794) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔴 faer-fork 1.44× slower than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 3.3523 ± 0.0594453 (W 0.00722507) | 3.56455 ± 0.0453191 (W 0.0306058) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.78545 ± 0.0197174 (W 0.00306523) | 4.17786 ± 0.0191711 (W 0.0179305) | 2.66986 ± 0.0462694 (W 0.00532308) | 🔵 faer-fork 2.34× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.83965 ± 0.00799943 (W 0.00645669) | 4.15624 ± 0.0378402 (W 0.0286198) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.46454 ± 0.0371612 (W 0.0208034) | 16.838 ± 0.0562392 (W 0.0549647) | 11.4932 ± 0.0219525 (W 0.0201045) | 🔵 faer-fork 2.60× faster than `faer-upstream` | 🔵 faer-fork 1.78× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 6.57798 ± 0.0109949 (W 0.0160355) | 16.9577 ± 0.134821 (W 0.143168) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.8709 ± 0.0376303 (W 0.0650133) | 54.5021 ± 0.451661 (W 0.894872) | 44.9204 ± 0.362266 (W 0.0600789) | 🔵 faer-fork 2.19× faster than `faer-upstream` | 🔵 faer-fork 1.81× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 25.4651 ± 0.158646 (W 0.024022) | 54.5464 ± 0.71082 (W 0.82432) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.971076 ± 0.0102217 (W 0.00372625) | 1.0645 ± 0.000841309 (W 0.00221126) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 1.15124 ± 0.00525092 (W 0.00728927) | 1.07583 ± 0.00241024 (W 0.00349275) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 6.67385 ± 0.0186966 (W 0.0150078) | 10.8614 ± 0.0189643 (W 0.00534699) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 7.62631 ± 0.0359232 (W 0.0196548) | 10.6561 ± 0.0380294 (W 0.0134231) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 26.8418 ± 0.0687278 (W 0.0281691) | 41.5152 ± 0.322894 (W 0.0240693) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 31.3809 ± 0.0901773 (W 0.0370355) | 41.2606 ± 0.108522 (W 1.83952) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.21871 ± 0.0102127 (W 0.00416727) | 0.466962 ± 0.125768 (W 0.024013) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.20568 ± 0.00674448 (W 0.00282131) | 0.675603 ± 0.117391 (W 0.0505363) | - | 🔵 faer-fork 3.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.936124 ± 0.0072312 (W 0.00613551) | 1.81777 ± 0.156658 (W 0.0998919) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.942725 ± 0.00184565 (W 0.00668112) | 2.14139 ± 0.113463 (W 0.1479) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.93131 ± 0.0705682 (W 0.0212441) | 6.81377 ± 0.148372 (W 0.27025) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.99084 ± 0.0501912 (W 0.0627572) | 6.70423 ± 0.217863 (W 0.216135) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.352817 ± 0.00678163 (W 0.00415007) | 0.42799 ± 0.00233467 (W 0.00635192) | 0.603819 ± 0.000616271 (W 0.00321107) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.71× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.602749 ± 0.00629874 (W 0.00388203) | 0.478983 ± 0.00368666 (W 0.00736466) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.25858 ± 0.00736142 (W 0.00839031) | 2.11138 ± 0.0153746 (W 0.00931924) | 1.81001 ± 0.337625 (W 0.0347751) | 🔵 faer-fork 1.68× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n256/b1 [api-inclusive]` | 2.36437 ± 0.0106897 (W 0.00851707) | 2.28716 ± 0.0267878 (W 0.0130846) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.45214 ± 0.0423105 (W 0.014197) | 6.80245 ± 0.0602484 (W 0.0235677) | 4.06822 ± 1.04732 (W 0.133779) | 🔵 faer-fork 1.53× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n512/b1 [api-inclusive]` | 9.28268 ± 0.0574235 (W 0.0298725) | 7.97499 ± 0.0267613 (W 8.4987) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.933744 ± 0.00580718 (W 0.00467683) | 2.86773 ± 0.219045 (W 0.154866) | 1.03928 ± 0.173795 (W 0.0100605) | 🔵 faer-fork 3.07× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1 [api-inclusive]` | 1.38962 ± 0.00116214 (W 0.00260467) | 2.91977 ± 0.528883 (W 0.0814998) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 2.61749 ± 0.01343 (W 0.00470838) | 6.82968 ± 0.128914 (W 0.0926687) | 3.00683 ± 0.0830152 (W 0.06306) | 🔵 faer-fork 2.61× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 4.50651 ± 0.00830344 (W 0.00662571) | 8.65798 ± 0.0394407 (W 0.0601484) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 9.29516 ± 0.0178069 (W 0.023026) | 23.2123 ± 0.150446 (W 0.735164) | 11.7804 ± 0.0469079 (W 0.0409891) | 🔵 faer-fork 2.50× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 17.148 ± 0.0996166 (W 0.0290313) | 30.9391 ± 0.347747 (W 0.777882) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.612889 ± 0.00113767 (W 0.00242109) | 1.45799 ± 0.251018 (W 0.0719832) | 0.735324 ± 0.152561 (W 0.00559662) | 🔵 faer-fork 2.38× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n256/b1` | 1.86117 ± 0.015529 (W 0.00356236) | 5.60295 ± 0.457696 (W 0.175777) | 2.1576 ± 0.521135 (W 0.0415613) | 🔵 faer-fork 3.01× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n512/b1` | 5.73366 ± 0.0352402 (W 0.00891578) | 16.094 ± 2.56927 (W 0.724723) | 7.52144 ± 0.578604 (W 0.139159) | 🔵 faer-fork 2.81× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n128/b1` | 0.385958 ± 0.00160785 (W 0.00376315) | 0.40996 ± 0.0102052 (W 0.00742793) | 0.562087 ± 0.000960833 (W 0.00191567) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.46× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.623569 ± 0.00315418 (W 0.00678311) | 0.457902 ± 0.0107749 (W 0.00418617) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.2902 ± 0.0111368 (W 0.00510521) | 1.91094 ± 0.0151347 (W 0.0113424) | 1.68627 ± 0.468826 (W 0.0204308) | 🔵 faer-fork 1.48× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n256/b1 [api-inclusive]` | 1.56144 ± 0.0443772 (W 0.00451652) | 2.08759 ± 0.0194812 (W 0.0169624) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 4.49943 ± 0.0238568 (W 0.0200752) | 6.34027 ± 0.0224483 (W 0.0303815) | 3.00435 ± 0.0316557 (W 0.0507635) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔴 faer-fork 1.50× slower than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 5.94233 ± 0.0149536 (W 0.0227427) | 7.712 ± 0.0343278 (W 0.0288731) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.61859 ± 0.0312309 (W 0.0222916) | 6.04607 ± 0.0215819 (W 0.0237491) | 5.99982 ± 0.110117 (W 0.0185305) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.69376 ± 0.0194356 (W 0.0100219) | 6.14437 ± 0.0442606 (W 0.0526831) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 17.6955 ± 0.124209 (W 0.0336863) | 32.3004 ± 0.163218 (W 0.119774) | 25.6388 ± 0.209056 (W 0.0452007) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 18.0294 ± 0.107436 (W 0.0165876) | 32.4893 ± 0.101797 (W 0.111033) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 69.3506 ± 0.127316 (W 0.056689) | 102.607 ± 0.116196 (W 0.317422) | 120.262 ± 1.15905 (W 0.100664) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.73× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 70.3868 ± 0.166141 (W 0.0768971) | 103.591 ± 0.997584 (W 6.09219) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 2.49837 ± 0.0179786 (W 0.0125106) | 4.03654 ± 0.00403747 (W 0.00547443) | 3.31522 ± 0.185997 (W 0.0111832) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/n128/b1 [api-inclusive]` | 2.49829 ± 0.0215156 (W 0.0104796) | 4.06861 ± 0.044909 (W 0.0441121) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 10.3953 ± 0.0351092 (W 0.0169707) | 25.0455 ± 0.0385362 (W 0.0317296) | 14.9292 ± 0.486549 (W 0.020193) | 🔵 faer-fork 2.41× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/n256/b1 [api-inclusive]` | 10.4765 ± 0.025223 (W 0.0233557) | 25.1377 ± 0.0571718 (W 0.0406544) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 41.9969 ± 0.119137 (W 0.0488423) | 72.3361 ± 0.225911 (W 0.0686559) | 62.3749 ± 0.554334 (W 0.119424) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 42.1884 ± 0.254146 (W 0.0278035) | 73.411 ± 0.400149 (W 0.172309) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.023981 ± 0.000270247 (W 0.000340397) | 0.020071 ± 6.50641e-05 (W 0.000288149) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n256/b1` | 0.038302 ± 0.000450037 (W 0.000251726) | 0.037452 ± 9.07377e-05 (W 0.000159465) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.103635 ± 0.0045583 (W 0.000946673) | 0.111836 ± 0.00630396 (W 0.00253081) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.192739 ± 0.00160255 (W 0.00153104) | 0.235361 ± 0.00677578 (W 0.0146075) | 0.608649 ± 0.00377312 (W 0.00353415) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 3.16× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.216381 ± 0.0014146 (W 0.00145674) | 0.230421 ± 0.000506392 (W 0.00769297) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.764936 ± 0.0418704 (W 0.00807942) | 1.31603 ± 0.140299 (W 0.0421537) | 0.719714 ± 0.00782235 (W 0.0189884) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 1.00729 ± 0.0599807 (W 0.00853678) | 4.40621 ± 12.2483 (W 5.44856) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 2.34425 ± 0.00933761 (W 0.0186763) | 4.37047 ± 0.0189905 (W 0.167528) | 1.76842 ± 0.0324627 (W 0.0374704) | 🔵 faer-fork 1.86× faster than `faer-upstream` | 🔴 faer-fork 1.33× slower than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 3.4081 ± 0.0466496 (W 0.0136654) | 3.72764 ± 0.469158 (W 0.0912923) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.73062 ± 0.00794235 (W 0.00717759) | 4.19338 ± 0.00397578 (W 0.0243943) | 3.14635 ± 0.0662681 (W 0.0389798) | 🔵 faer-fork 2.42× faster than `faer-upstream` | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.76517 ± 0.0124969 (W 0.0051357) | 4.15006 ± 19.5087 (W 0.0267372) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1` | 7.4595 ± 0.0333182 (W 0.0392016) | 16.8716 ± 0.104828 (W 0.0891276) | 14.9341 ± 0.0807814 (W 0.0586825) | 🔵 faer-fork 2.26× faster than `faer-upstream` | 🔵 faer-fork 2.00× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 7.53471 ± 0.03214 (W 0.0152688) | 16.9549 ± 0.0337122 (W 0.0573325) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 28.5728 ± 0.0277038 (W 0.0913292) | 58.5442 ± 0.159812 (W 0.392936) | 55.7864 ± 0.233109 (W 0.0446662) | 🔵 faer-fork 2.05× faster than `faer-upstream` | 🔵 faer-fork 1.95× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 28.5554 ± 0.078625 (W 0.0699264) | 58.6439 ± 0.286904 (W 1.13773) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.959035 ± 0.0073516 (W 0.00345956) | 1.06604 ± 0.00281938 (W 0.00246683) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 1.16661 ± 0.00460201 (W 0.00905111) | 1.07786 ± 0.000793872 (W 0.00198826) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 6.86048 ± 0.0425338 (W 0.00647489) | 10.84 ± 0.0220307 (W 0.0102634) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 7.87718 ± 0.0451178 (W 0.0153053) | 10.6656 ± 0.111265 (W 0.0130458) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 26.656 ± 0.0883702 (W 0.034708) | 41.0315 ± 0.0878431 (W 0.0780019) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 30.9106 ± 0.0519169 (W 0.0435577) | 40.832 ± 0.153347 (W 0.0557434) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.185318 ± 0.019253 (W 0.00287402) | 0.785367 ± 0.189245 (W 0.058085) | - | 🔵 faer-fork 4.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.175678 ± 0.0062042 (W 0.00238655) | 0.942754 ± 0.0730719 (W 0.0744896) | - | 🔵 faer-fork 5.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.671492 ± 0.00686873 (W 0.00725865) | 1.45357 ± 0.948341 (W 0.0219708) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.663722 ± 0.00656802 (W 0.00730973) | 3.31439 ± 0.0594046 (W 0.112404) | - | 🔵 faer-fork 4.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.25133 ± 0.00756363 (W 0.0225214) | 4.95639 ± 0.0206084 (W 0.245455) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.27096 ± 0.0457188 (W 0.033766) | 4.98694 ± 0.0493942 (W 0.215495) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.355067 ± 0.00197854 (W 0.00317997) | 0.440931 ± 0.0117551 (W 0.00498078) | 0.604689 ± 0.00201299 (W 0.00288353) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.70× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.610209 ± 0.000947488 (W 0.00672305) | 0.506504 ± 3.73666 (W 0.0212412) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 1.26858 ± 0.0756117 (W 0.0068106) | 2.64788 ± 0.147533 (W 0.0460953) | 2.1216 ± 0.130532 (W 0.0572995) | 🔵 faer-fork 2.09× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n256/b1 [api-inclusive]` | 2.48598 ± 0.0781641 (W 0.00857071) | 2.36356 ± 0.0442407 (W 0.031033) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 4.12749 ± 0.0302947 (W 0.015663) | 10.2155 ± 0.99798 (W 0.387191) | 6.58047 ± 0.394586 (W 0.112697) | 🔵 faer-fork 2.47× faster than `faer-upstream` | 🔵 faer-fork 1.59× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n512/b1 [api-inclusive]` | 9.35337 ± 0.134493 (W 0.0412851) | 8.05411 ± 0.150117 (W 0.161623) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.903003 ± 0.0067808 (W 0.00372047) | 3.81689 ± 0.0433238 (W 0.109108) | 0.813849 ± 0.138451 (W 0.00803118) | 🔵 faer-fork 4.23× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1 [api-inclusive]` | 1.38222 ± 0.00705816 (W 0.0039914) | 3.28602 ± 1.05392 (W 0.166) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 2.44531 ± 0.0099864 (W 0.00572224) | 14.3825 ± 1.35828 (W 0.344349) | 3.72054 ± 0.0424309 (W 0.0470762) | 🔵 faer-fork 5.88× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n256/b1 [api-inclusive]` | 4.39412 ± 0.0465853 (W 0.0114161) | 18.0447 ± 5.05844 (W 0.6934) | - | 🔵 faer-fork 4.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 8.29037 ± 0.0955514 (W 0.0589996) | 42.6462 ± 1.11112 (W 1.25904) | 8.85665 ± 1.45724 (W 0.116967) | 🔵 faer-fork 5.14× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n512/b1 [api-inclusive]` | 15.8396 ± 0.0902852 (W 0.0337507) | 46.3119 ± 6.90349 (W 2.29179) | - | 🔵 faer-fork 2.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.597808 ± 0.0180645 (W 0.00275871) | 1.78924 ± 0.350677 (W 0.0685135) | 0.571817 ± 0.0904042 (W 0.00891607) | 🔵 faer-fork 2.99× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n256/b1` | 1.93198 ± 0.0220716 (W 0.00437609) | 7.29064 ± 1.4714 (W 0.391873) | 2.4877 ± 0.0240755 (W 0.0174317) | 🔵 faer-fork 3.77× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n512/b1` | 5.58985 ± 0.00911501 (W 0.00818842) | 20.4984 ± 9.51162 (W 6.92495) | 8.22693 ± 0.325123 (W 0.146156) | 🔵 faer-fork 3.67× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n128/b1` | 0.385708 ± 0.00160107 (W 0.0056668) | 0.445521 ± 0.0118468 (W 0.00757084) | 0.565637 ± 0.00117013 (W 0.00235233) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.63663 ± 0.00982042 (W 0.00443933) | 0.475813 ± 3.74309 (W 0.00859689) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 1.29816 ± 0.0324519 (W 0.00787823) | 3.49947 ± 0.508187 (W 0.0958313) | 1.92829 ± 0.461551 (W 0.0217409) | 🔵 faer-fork 2.70× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n256/b1 [api-inclusive]` | 1.66796 ± 0.0822185 (W 0.010207) | 2.19829 ± 0.109992 (W 0.0307057) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 4.18256 ± 0.0652369 (W 0.0193256) | 10.6449 ± 2.87703 (W 0.439437) | 6.20688 ± 0.361328 (W 0.118073) | 🔵 faer-fork 2.55× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n512/b1 [api-inclusive]` | 6.08249 ± 0.106864 (W 0.0187353) | 10.3046 ± 1.49656 (W 0.485581) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 4.63946 ± 0.0127628 (W 0.0137197) | 6.5669 ± 0.129413 (W 0.145778) | 6.88869 ± 0.0532163 (W 0.0634225) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.72698 ± 0.0294183 (W 0.0226056) | 6.24843 ± 0.0483478 (W 0.0663354) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 18.7857 ± 0.0039856 (W 0.0220777) | 34.2463 ± 0.697167 (W 0.872777) | 28.8667 ± 0.141486 (W 0.0926404) | 🔵 faer-fork 1.82× faster than `faer-upstream` | 🔵 faer-fork 1.54× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 19.049 ± 0.0121801 (W 0.0389823) | 33.4556 ± 0.0398486 (W 0.267015) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 72.8932 ± 0.21258 (W 0.198835) | 102.571 ± 0.847727 (W 41.6619) | 120.385 ± 0.654042 (W 0.237408) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 1.65× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 74.252 ± 0.202192 (W 0.169552) | 109.049 ± 8.0245 (W 6.21417) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 2.52275 ± 0.0152518 (W 0.0185967) | 4.05961 ± 0.0122139 (W 0.0301022) | 4.18822 ± 0.033648 (W 0.0300988) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 2.52797 ± 0.0175913 (W 0.0154648) | 4.05676 ± 0.00868754 (W 0.0387005) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.061 ± 0.0523933 (W 0.0157579) | 25.8356 ± 0.0649973 (W 0.0589836) | 18.0479 ± 0.0359854 (W 0.151329) | 🔵 faer-fork 2.34× faster than `faer-upstream` | 🔵 faer-fork 1.63× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 11.1328 ± 0.0325652 (W 0.0189109) | 25.8537 ± 0.086259 (W 0.0506558) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 45.7798 ± 0.0524159 (W 0.0558172) | 81.1861 ± 1.58364 (W 20.908) | 69.7132 ± 1.08637 (W 0.461935) | 🔵 faer-fork 1.77× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 45.8858 ± 0.0749268 (W 0.0574565) | 78.7733 ± 0.689722 (W 3.12066) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.015681 ± 9.70069e-05 (W 3.91306e-05) | 0.01151 ± 8.59089e-05 (W 4.93184e-05) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.038722 ± 0.000160935 (W 0.000143074) | 0.036872 ± 0.000153731 (W 0.000145142) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.133476 ± 0.0130229 (W 0.000425143) | 0.134387 ± 1.24223 (W 0.0116504) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.188369 ± 0.000571664 (W 0.00162113) | 0.253902 ± 0.0359458 (W 0.0165125) | 0.609499 ± 0.00275781 (W 0.00385543) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔵 faer-fork 3.24× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.21524 ± 0.0015103 (W 0.00223535) | 0.246142 ± 0.00332064 (W 0.00733016) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.763706 ± 0.0334373 (W 0.00407685) | 1.51242 ± 0.0400231 (W 0.0442108) | 0.757756 ± 0.0100266 (W 0.016519) | 🔵 faer-fork 1.98× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.985517 ± 0.0313063 (W 0.00739344) | 1.09855 ± 0.199871 (W 0.0199106) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 2.59366 ± 0.0440327 (W 0.0209168) | 4.82143 ± 0.236809 (W 0.0967715) | 2.55366 ± 0.0669024 (W 0.0633343) | 🔵 faer-fork 1.86× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 3.43242 ± 0.071391 (W 0.0132636) | 5.04821 ± 2.26848 (W 0.456376) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.80134 ± 0.00362564 (W 0.00998302) | 4.23419 ± 0.0211691 (W 0.0546778) | 3.70513 ± 0.0178191 (W 0.0317652) | 🔵 faer-fork 2.35× faster than `faer-upstream` | 🔵 faer-fork 2.06× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.83136 ± 0.0122373 (W 0.0109127) | 4.18925 ± 0.0200064 (W 0.0263105) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.18655 ± 0.0155464 (W 0.0309959) | 18.7656 ± 0.428812 (W 0.258751) | 18.7717 ± 0.398823 (W 0.062903) | 🔵 faer-fork 2.04× faster than `faer-upstream` | 🔵 faer-fork 2.04× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 9.64633 ± 0.0201792 (W 0.0152188) | 17.7526 ± 10.9066 (W 0.103337) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n512/b1` | 38.8446 ± 0.137545 (W 0.0767332) | 75.2397 ± 0.192729 (W 14.105) | 80.0843 ± 2.16668 (W 1.28478) | 🔵 faer-fork 1.94× faster than `faer-upstream` | 🔵 faer-fork 2.06× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 40.287 ± 0.349888 (W 0.175628) | 75.8115 ± 1.57614 (W 11.396) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.995378 ± 0.0148997 (W 0.00427761) | 1.0641 ± 0.00344289 (W 0.00196771) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 1.18588 ± 0.00292684 (W 0.00299632) | 1.07654 ± 0.00284379 (W 0.00174234) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 7.30531 ± 0.0249164 (W 0.0220596) | 10.8975 ± 0.019636 (W 0.013874) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 8.28321 ± 0.024846 (W 0.017646) | 10.7308 ± 0.0130245 (W 0.0235766) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 28.1987 ± 0.192398 (W 0.0634118) | 44.1926 ± 0.318145 (W 0.237058) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 32.6511 ± 0.401378 (W 0.064601) | 41.425 ± 0.193753 (W 0.434774) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.199399 ± 0.0417015 (W 0.00462632) | 0.980446 ± 0.32759 (W 0.0800245) | - | 🔵 faer-fork 4.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.171518 ± 0.0193056 (W 0.00323844) | 1.11367 ± 0.0682947 (W 0.0878375) | - | 🔵 faer-fork 6.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.63733 ± 0.0193154 (W 0.0067899) | 3.24342 ± 0.159028 (W 0.147857) | - | 🔵 faer-fork 5.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.61406 ± 0.00793418 (W 0.00453155) | 3.54512 ± 0.0256701 (W 0.0595989) | - | 🔵 faer-fork 5.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.24857 ± 0.0278546 (W 0.0557741) | 5.08939 ± 0.0524367 (W 0.0815156) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.24127 ± 0.0291297 (W 0.0469716) | 5.16262 ± 0.0393623 (W 0.0936116) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.355217 ± 0.00201789 (W 0.00433682) | 0.490303 ± 0.0321241 (W 0.0104768) | 0.607139 ± 0.00158406 (W 0.00325993) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔵 faer-fork 1.71× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.62532 ± 0.00582082 (W 0.004939) | 0.499243 ± 0.00742255 (W 0.00605028) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.18744 ± 0.0131046 (W 0.00636746) | 3.49445 ± 0.553725 (W 0.0802544) | 1.78531 ± 0.0919344 (W 0.040547) | 🔵 faer-fork 2.94× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n256/b1 [api-inclusive]` | 2.44168 ± 0.0510856 (W 0.00659069) | 2.42968 ± 0.0207553 (W 0.0412301) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.95731 ± 0.0505148 (W 0.0141946) | 14.3963 ± 3.41843 (W 0.34887) | 6.79717 ± 0.105906 (W 0.0729638) | 🔵 faer-fork 3.64× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n512/b1 [api-inclusive]` | 9.38709 ± 0.170227 (W 0.0508793) | 9.59624 ± 0.305714 (W 0.457536) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 1.0016 ± 0.00843894 (W 0.00967194) | 4.21598 ± 0.397107 (W 0.13924) | 0.864331 ± 0.0145414 (W 0.00593244) | 🔵 faer-fork 4.21× faster than `faer-upstream` | 🔴 faer-fork 1.16× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1 [api-inclusive]` | 1.47291 ± 0.0204541 (W 0.00830303) | 5.82937 ± 0.101268 (W 0.174283) | - | 🔵 faer-fork 3.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.39591 ± 0.0020304 (W 0.00901073) | 17.5567 ± 0.0339271 (W 0.28387) | 3.79042 ± 0.130813 (W 0.0595343) | 🔵 faer-fork 7.33× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 4.41789 ± 0.0231332 (W 0.00867375) | 14.901 ± 3.39282 (W 2.35006) | - | 🔵 faer-fork 3.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 8.94532 ± 0.401838 (W 0.0375474) | 43.7199 ± 3.67606 (W 0.565129) | 13.2224 ± 0.0689698 (W 0.121303) | 🔵 faer-fork 4.89× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n512/b1 [api-inclusive]` | 15.8068 ± 0.0851358 (W 0.0325018) | 52.0036 ± 2.58056 (W 1.22024) | - | 🔵 faer-fork 3.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.64411 ± 0.00779713 (W 0.00389077) | 2.06489 ± 0.560015 (W 0.103519) | 0.519434 ± 0.0112615 (W 0.00734561) | 🔵 faer-fork 3.21× faster than `faer-upstream` | 🔴 faer-fork 1.24× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n256/b1` | 1.96688 ± 0.0472388 (W 0.00771716) | 10.9977 ± 2.53552 (W 0.266564) | 3.18873 ± 0.0343254 (W 0.0301777) | 🔵 faer-fork 5.59× faster than `faer-upstream` | 🔵 faer-fork 1.62× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/n512/b1` | 5.95932 ± 0.0349053 (W 0.0152868) | 36.7321 ± 3.4154 (W 7.01284) | 9.27721 ± 0.086199 (W 0.125988) | 🔵 faer-fork 6.16× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n128/b1` | 0.393929 ± 0.000586544 (W 0.00315033) | 0.462182 ± 0.0126417 (W 0.0138633) | 0.561377 ± 2.64575e-05 (W 0.00218965) | 🔵 faer-fork 1.17× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.654561 ± 0.00589389 (W 0.00573693) | 0.481783 ± 0.00458119 (W 0.00628217) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.24715 ± 0.00841152 (W 0.00545366) | 5.13032 ± 0.0344635 (W 0.14146) | 1.65908 ± 0.215359 (W 0.0415754) | 🔵 faer-fork 4.11× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n256/b1 [api-inclusive]` | 1.64195 ± 0.0529611 (W 0.0106638) | 2.28924 ± 0.0130325 (W 0.0370061) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 4.13603 ± 0.61834 (W 0.0271876) | 17.2519 ± 0.0811698 (W 0.228382) | 7.36476 ± 1.23502 (W 0.388241) | 🔵 faer-fork 4.17× faster than `faer-upstream` | 🔵 faer-fork 1.78× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `solve/n512/b1 [api-inclusive]` | 6.17767 ± 0.212011 (W 0.0262879) | 8.88075 ± 1.42495 (W 0.51024) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 4.83209 ± 0.0975414 (W 0.0183609) | 6.37764 ± 0.307077 (W 0.176542) | 9.69957 ± 1.10984 (W 0.467279) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔵 faer-fork 2.01× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/n128/b1 [api-inclusive]` | 4.99791 ± 0.238321 (W 0.016874) | 6.3245 ± 0.0319739 (W 0.0696934) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n256/b1` | 20.3285 ± 0.0617123 (W 0.0242821) | 40.8413 ± 7.95755 (W 21.4865) | 34.1971 ± 2.5853 (W 1.27158) | 🔵 faer-fork 2.01× faster than `faer-upstream` | 🔵 faer-fork 1.68× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/n256/b1 [api-inclusive]` | 20.3985 ± 0.112257 (W 0.0378301) | 35.0545 ± 0.814138 (W 18.5765) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 85.4857 ± 0.355304 (W 0.198003) | 133.807 ± 3.71203 (W 10.1617) | 145.097 ± 2.85538 (W 2.00009) | 🔵 faer-fork 1.57× faster than `faer-upstream` | 🔵 faer-fork 1.70× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 85.6628 ± 0.0596582 (W 0.129794) | 135.502 ± 1.29199 (W 5.44461) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 2.52917 ± 0.0282412 (W 0.0146476) | 4.15717 ± 0.0409662 (W 0.0389677) | 4.97463 ± 1.6701 (W 0.118219) | 🔵 faer-fork 1.64× faster than `faer-upstream` | 🔵 faer-fork 1.97× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/n128/b1 [api-inclusive]` | 2.57157 ± 0.0181776 (W 0.00855271) | 4.08947 ± 0.0462765 (W 0.0520017) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.6689 ± 0.437079 (W 0.0155416) | 26.5098 ± 0.196427 (W 0.143499) | 23.5133 ± 4.41832 (W 1.4258) | 🔵 faer-fork 2.27× faster than `faer-upstream` | 🔵 faer-fork 2.02× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/n256/b1 [api-inclusive]` | 11.7737 ± 0.142316 (W 0.0194379) | 26.3557 ± 0.0314655 (W 0.0407946) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 55.2666 ± 0.649878 (W 0.0749839) | 101.396 ± 3.50974 (W 6.91305) | 83.4136 ± 8.66071 (W 0.922089) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.51× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 55.2806 ± 0.546058 (W 0.203929) | 100.147 ± 2.15435 (W 85.1843) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.015371 ± 0.000107503 (W 2.8413e-05) | 0.011131 ± 0.000149634 (W 8.89061e-05) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.044502 ± 0.000425715 (W 0.000292901) | 0.037241 ± 0.000205297 (W 0.000121059) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.184629 ± 0.00207403 (W 0.00832414) | 0.167698 ± 1.22651 (W 0.0163353) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `lu/128`, `lu/256`, `lu/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `solve/128`, `solve/256`, `solve/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `triangular_solve/128`, `triangular_solve/256`, `triangular_solve/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.0885 ± 0.00915516 (W 0.00450292) | 1.79834 ± 0.0194325 (W 0.00272324) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.13939 ± 0.00238936 (W 0.00294772) | 1.28921 ± 0.00624273 (W 0.00215586) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 15.9354 ± 0.0495169 (W 0.0295428) | 47.4722 ± 0.0629398 (W 0.0696331) | - | 🔵 faer-fork 2.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 20.0041 ± 0.0919333 (W 0.0212782) | 21.941 ± 0.172679 (W 0.0293302) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.251222 ± 0.00941502 (W 0.0044544) | 0.252522 ± 0.0407636 (W 0.0158852) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.239451 ± 0.00147672 (W 0.0047189) | 0.290774 ± 0.0173241 (W 0.021779) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.64421 ± 0.0265547 (W 0.0780924) | 6.22541 ± 0.0700139 (W 0.29822) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.71472 ± 0.0660954 (W 0.113539) | 6.21819 ± 0.1084 (W 0.114258) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.520835 ± 0.00297979 (W 0.00308474) | 1.34353 ± 0.014675 (W 0.00760249) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.583538 ± 0.00572491 (W 0.00381689) | 0.724954 ± 0.00612138 (W 0.00509923) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 7.51769 ± 0.0325467 (W 0.0162428) | 50.9486 ± 0.422285 (W 0.0749908) | - | 🔵 faer-fork 6.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 11.7723 ± 0.100392 (W 0.0255867) | 14.0172 ± 0.0693299 (W 0.0172993) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.2382 ± 0.00865759 (W 0.00608636) | 5.55333 ± 0.478428 (W 0.214398) | - | 🔵 faer-fork 4.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.71826 ± 0.00424464 (W 0.00743781) | 3.79298 ± 0.30513 (W 0.0895452) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 20.4872 ± 0.167354 (W 0.156585) | 141.34 ± 1.08097 (W 0.555976) | - | 🔵 faer-fork 6.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 30.5469 ± 0.122269 (W 0.0469283) | 58.5302 ± 0.814535 (W 0.833486) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.968806 ± 0.00395429 (W 0.00570207) | 4.72924 ± 0.442458 (W 0.278013) | - | 🔵 faer-fork 4.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 13.0008 ± 0.222508 (W 0.0256045) | 131.784 ± 0.897929 (W 0.325022) | - | 🔵 faer-fork 10.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.17413 ± 0.0084406 (W 0.00517446) | 4.97325 ± 0.574184 (W 0.115641) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.24875 ± 0.00845543 (W 0.00536869) | 4.69144 ± 0.470454 (W 0.0507461) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 33.8919 ± 0.0516414 (W 0.0950374) | 70.2019 ± 0.234858 (W 1.28766) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 34.6942 ± 0.0090613 (W 0.09067) | 70.406 ± 0.28336 (W 1.31902) | - | 🔵 faer-fork 2.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.45539 ± 0.0103889 (W 0.0073369) | 3.52756 ± 0.386519 (W 0.092549) | - | 🔵 faer-fork 2.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.46889 ± 0.0149885 (W 0.00943385) | 3.59534 ± 0.655322 (W 0.112838) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 24.2739 ± 0.0743293 (W 0.0605951) | 51.8669 ± 0.250727 (W 0.1955) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.6443 ± 0.0460938 (W 0.0654212) | 51.3436 ± 0.416396 (W 2.42162) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.11259 ± 0.00512793 (W 0.00506336) | 1.81191 ± 0.0195386 (W 0.00292567) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.17315 ± 0.00780317 (W 0.00377352) | 1.27646 ± 0.00154402 (W 0.00286144) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 15.4517 ± 0.0384372 (W 0.033927) | 45.6047 ± 0.653296 (W 0.115733) | - | 🔵 faer-fork 2.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 19.6797 ± 0.0701559 (W 0.0380597) | 23.9437 ± 1.53589 (W 1.0734) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.314565 ± 0.0666639 (W 0.00475967) | 0.63484 ± 0.232985 (W 0.0264144) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.312505 ± 0.0343964 (W 0.0056834) | 0.715824 ± 0.0270478 (W 0.0391823) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.47629 ± 0.0168331 (W 0.0466943) | 4.08135 ± 0.0538922 (W 0.0668444) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.52665 ± 0.0193511 (W 0.0827364) | 4.15859 ± 0.0513863 (W 0.0593015) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.515004 ± 0.0269408 (W 0.00461283) | 2.11559 ± 0.460355 (W 0.0618603) | - | 🔵 faer-fork 4.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.559756 ± 0.00847848 (W 0.0023835) | 0.765006 ± 0.0181881 (W 0.0135041) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.90588 ± 0.0629411 (W 0.0125398) | 52.1621 ± 0.714358 (W 0.106346) | - | 🔵 faer-fork 7.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 11.2723 ± 0.0657273 (W 0.0188716) | 13.8568 ± 0.249553 (W 0.0437886) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.37344 ± 0.0748397 (W 0.00810986) | 10.8153 ± 0.498226 (W 0.475466) | - | 🔵 faer-fork 7.87× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.73752 ± 0.183558 (W 0.00526131) | 9.08682 ± 0.182944 (W 0.290567) | - | 🔵 faer-fork 5.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 18.704 ± 0.0525435 (W 0.0421352) | 143.516 ± 0.0798237 (W 0.166669) | - | 🔵 faer-fork 7.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 29.9485 ± 0.124361 (W 0.0399431) | 58.795 ± 3.71169 (W 0.897584) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.985077 ± 0.188692 (W 0.00395384) | 6.63044 ± 2.69248 (W 0.63592) | - | 🔵 faer-fork 6.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 12.9446 ± 0.254725 (W 0.0363431) | 169.726 ± 0.74123 (W 25.9362) | - | 🔵 faer-fork 13.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.18636 ± 0.0687637 (W 0.00648628) | 9.82939 ± 0.024903 (W 0.29233) | - | 🔵 faer-fork 4.50× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.28735 ± 0.156486 (W 0.00597) | 5.94369 ± 1.55694 (W 0.427383) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 33.072 ± 0.0827566 (W 0.0813416) | 71.4349 ± 0.220815 (W 1.36149) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 34.1815 ± 0.26404 (W 0.065037) | 72.4974 ± 0.956869 (W 3.93521) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.44095 ± 0.104411 (W 0.00683837) | 7.78333 ± 0.178662 (W 0.312315) | - | 🔵 faer-fork 5.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.45585 ± 0.213644 (W 0.00695708) | 5.2374 ± 1.92839 (W 0.452755) | - | 🔵 faer-fork 3.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 25.3114 ± 0.231741 (W 0.234999) | 82.8575 ± 3.61793 (W 2.92093) | - | 🔵 faer-fork 3.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 25.9912 ± 0.649448 (W 0.166353) | 73.2285 ± 5.97098 (W 5.20475) | - | 🔵 faer-fork 2.82× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.16576 ± 0.140767 (W 0.00510097) | 1.82429 ± 0.051946 (W 0.00498037) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.20733 ± 0.163675 (W 0.00567485) | 1.3014 ± 0.0123444 (W 0.00492528) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 16.3309 ± 0.0683798 (W 0.0483521) | 47.4741 ± 0.875143 (W 0.0612565) | - | 🔵 faer-fork 2.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 20.8544 ± 1.63462 (W 0.0271686) | 35.4836 ± 2.75266 (W 2.45719) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.228121 ± 0.218366 (W 0.00919309) | 0.730664 ± 0.238648 (W 0.0651916) | - | 🔵 faer-fork 3.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.22251 ± 0.111588 (W 0.00363287) | 0.797978 ± 0.0522673 (W 0.0628347) | - | 🔵 faer-fork 3.59× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.32495 ± 0.0839741 (W 0.0390512) | 4.01632 ± 0.146276 (W 0.204053) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.27602 ± 0.0441314 (W 0.0518887) | 3.86068 ± 0.0556477 (W 0.280951) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.481103 ± 0.0682256 (W 0.00332261) | 2.8324 ± 0.625881 (W 0.0427679) | - | 🔵 faer-fork 5.89× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.555256 ± 0.00479537 (W 0.00294506) | 0.816549 ± 0.0134362 (W 0.0140613) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.57814 ± 0.0180572 (W 0.0199003) | 53.3626 ± 1.42234 (W 0.373553) | - | 🔵 faer-fork 8.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 11.5207 ± 0.1231 (W 0.0361975) | 13.8755 ± 0.197282 (W 0.0813244) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.42503 ± 0.179594 (W 0.00863178) | 12.434 ± 2.5187 (W 0.941526) | - | 🔵 faer-fork 8.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.90145 ± 0.305653 (W 0.00783563) | 10.1286 ± 2.73521 (W 0.264662) | - | 🔵 faer-fork 5.33× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 19.2714 ± 0.0306883 (W 0.0474729) | 190.589 ± 1.83532 (W 1.94193) | - | 🔵 faer-fork 9.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 31.6463 ± 0.0671422 (W 0.242303) | 129.633 ± 3.04137 (W 6.3819) | - | 🔵 faer-fork 4.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.07277 ± 0.191452 (W 0.00663094) | 6.27105 ± 4.24434 (W 0.273788) | - | 🔵 faer-fork 5.85× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 13.3979 ± 0.0114373 (W 0.0458977) | 185.134 ± 3.53829 (W 5.67334) | - | 🔵 faer-fork 13.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.40759 ± 0.0459001 (W 0.00779619) | 10.923 ± 1.01094 (W 0.293006) | - | 🔵 faer-fork 4.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.63496 ± 0.141478 (W 0.00506785) | 11.2266 ± 0.674948 (W 0.569355) | - | 🔵 faer-fork 4.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 35.6183 ± 0.32378 (W 0.216845) | 152.858 ± 2.38845 (W 6.9666) | - | 🔵 faer-fork 4.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 36.2682 ± 0.543213 (W 0.26303) | 146.922 ± 0.518558 (W 4.65535) | - | 🔵 faer-fork 4.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.53653 ± 0.0551755 (W 0.00521528) | 8.22758 ± 1.18063 (W 0.2267) | - | 🔵 faer-fork 5.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.65385 ± 0.137614 (W 0.00488628) | 4.91883 ± 2.22259 (W 0.196303) | - | 🔵 faer-fork 2.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 27.7103 ± 1.40389 (W 0.363058) | 109.086 ± 1.88796 (W 3.12283) | - | 🔵 faer-fork 3.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 27.7517 ± 0.506213 (W 0.9363) | 108.677 ± 1.19281 (W 3.19064) | - | 🔵 faer-fork 3.92× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.948785 ± 0.00539806 (W 0.00390893) | 1.24627 ± 0.00285565 (W 0.00258133) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.00532 ± 0.00535846 (W 0.00339062) | 1.15782 ± 0.00875978 (W 0.00357683) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 14.3689 ± 0.0795417 (W 0.0183358) | 15.588 ± 0.00919665 (W 0.0186121) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 16.6573 ± 0.0160117 (W 0.0239204) | 17.5816 ± 0.0766564 (W 0.0790317) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.165778 ± 0.00418275 (W 0.00322371) | 0.265382 ± 0.0640473 (W 0.0220412) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.165378 ± 0.00120964 (W 0.00511202) | 0.343706 ± 0.0947425 (W 0.0165621) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 6.20213 ± 0.0984262 (W 0.216892) | 5.93851 ± 0.281801 (W 0.648059) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 6.25062 ± 0.00787738 (W 0.193719) | 5.45822 ± 0.359942 (W 0.29792) | - | 🔴 faer-fork 1.15× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.337326 ± 0.00143147 (W 0.00280988) | 0.503924 ± 0.00646108 (W 0.00770522) | 0.315365 ± 0.00106444 (W 0.00170033) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.436001 ± 0.00957414 (W 0.00300116) | 0.604489 ± 0.00779354 (W 0.00451352) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.66566 ± 0.0290306 (W 0.0142995) | 7.51901 ± 0.178808 (W 0.0127703) | 3.15753 ± 0.0180606 (W 0.0155164) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔴 faer-fork 1.79× slower than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 8.09099 ± 0.039811 (W 0.0291574) | 10.0384 ± 0.048572 (W 0.0207796) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.993097 ± 0.080675 (W 0.00432911) | 2.72054 ± 1.13421 (W 0.0475863) | 1.09977 ± 0.746824 (W 0.0075381) | 🔵 faer-fork 2.74× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.58996 ± 0.088918 (W 0.00736314) | 3.05616 ± 0.434752 (W 0.156601) | - | 🔵 faer-fork 1.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 15.5021 ± 2.48882 (W 0.0278234) | 42.1331 ± 0.260411 (W 0.710112) | 13.9044 ± 0.183723 (W 0.24336) | 🔵 faer-fork 2.72× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 25.7538 ± 0.0383948 (W 0.0282139) | 52.9132 ± 0.426008 (W 0.530832) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.894492 ± 0.0299556 (W 0.000810635) | 2.00503 ± 0.477677 (W 0.095916) | 1.9261 ± 0.646278 (W 0.0175643) | 🔵 faer-fork 2.24× faster than `faer-upstream` | 🔵 faer-fork 2.15× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/4096x256/b1` | 11.4963 ± 0.118669 (W 0.0305497) | 30.9424 ± 0.72144 (W 0.233519) | 6.64946 ± 0.218558 (W 0.0764552) | 🔵 faer-fork 2.69× faster than `faer-upstream` | 🔴 faer-fork 1.73× slower than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/1024x64/b1` | 2.17842 ± 0.00402449 (W 0.00565939) | 4.99893 ± 0.732524 (W 0.127647) | 3.11946 ± 0.131934 (W 0.00563669) | 🔵 faer-fork 2.29× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.24944 ± 0.00536709 (W 0.0058084) | 3.95346 ± 0.98611 (W 0.120312) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 31.4128 ± 0.0775502 (W 0.0598482) | 67.4005 ± 1.38823 (W 0.952302) | 42.0139 ± 0.965101 (W 0.364589) | 🔵 faer-fork 2.15× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 32.2192 ± 0.0609593 (W 0.0574384) | 67.897 ± 0.735531 (W 1.13269) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.44583 ± 0.00302827 (W 0.00442573) | 3.85492 ± 0.479831 (W 0.0756176) | 1.42994 ± 0.117629 (W 0.00812661) | 🔵 faer-fork 2.67× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.46093 ± 0.0120345 (W 0.00445005) | 4.16898 ± 0.831406 (W 0.105544) | - | 🔵 faer-fork 2.85× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 22.5514 ± 0.0608949 (W 0.0491489) | 51.3521 ± 0.270192 (W 0.160197) | 20.0519 ± 0.196049 (W 0.300311) | 🔵 faer-fork 2.28× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 22.7099 ± 0.0275422 (W 0.0361778) | 50.2366 ± 0.872046 (W 0.761321) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.975716 ± 0.00798699 (W 0.00357281) | 1.24296 ± 0.00516915 (W 0.00390688) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.04264 ± 0.00568889 (W 0.00359884) | 1.16475 ± 0.0109384 (W 0.00436644) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 13.8246 ± 0.0915045 (W 0.0418382) | 17.3807 ± 1.6319 (W 0.899285) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 16.3198 ± 0.0267005 (W 0.0207923) | 19.235 ± 0.647491 (W 1.18604) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.20364 ± 0.0129238 (W 0.00232733) | 0.82461 ± 0.00864378 (W 0.0326209) | - | 🔵 faer-fork 4.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.181588 ± 0.00540698 (W 0.00164908) | 0.818639 ± 0.0259574 (W 0.0376359) | - | 🔵 faer-fork 4.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.60129 ± 0.0232522 (W 0.0517135) | 4.21667 ± 0.0618724 (W 0.155465) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.61726 ± 0.085986 (W 0.0348672) | 4.15361 ± 0.0368931 (W 0.23324) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.329146 ± 0.0040625 (W 0.00330807) | 0.869791 ± 0.271081 (W 0.0142397) | 0.316135 ± 0.00303317 (W 0.00664691) | 🔵 faer-fork 2.64× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.41443 ± 0.0109601 (W 0.00310908) | 0.647621 ± 12.3295 (W 0.0135557) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 5.22248 ± 0.000974115 (W 0.0116425) | 7.48556 ± 0.183914 (W 0.0410225) | 3.40985 ± 0.071956 (W 0.0163294) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔴 faer-fork 1.53× slower than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 7.80347 ± 0.00381389 (W 0.0108327) | 9.90981 ± 0.141913 (W 0.042336) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.01071 ± 0.0640953 (W 0.00253575) | 7.23502 ± 0.196297 (W 0.367005) | 1.69289 ± 0.0279757 (W 0.0434765) | 🔵 faer-fork 7.16× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.70855 ± 0.113454 (W 0.00487756) | 9.01814 ± 2.83915 (W 0.277872) | - | 🔵 faer-fork 5.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 15.059 ± 1.11593 (W 0.0854404) | 42.6592 ± 0.120218 (W 1.6468) | 14.8365 ± 0.292305 (W 0.496153) | 🔵 faer-fork 2.83× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 25.1892 ± 0.0581937 (W 0.140935) | 53.0015 ± 1.58144 (W 0.782576) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.829549 ± 0.0190149 (W 0.00237302) | 5.76235 ± 1.6622 (W 0.219816) | 0.765916 ± 0.37474 (W 0.0237261) | 🔵 faer-fork 6.95× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/4096x256/b1` | 12.1487 ± 0.0444702 (W 0.0830945) | 74.9443 ± 8.67918 (W 18.1322) | 11.1576 ± 0.819395 (W 0.851464) | 🔵 faer-fork 6.17× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/1024x64/b1` | 2.28371 ± 0.0239447 (W 0.00724339) | 10.0174 ± 0.405562 (W 0.305544) | 4.53716 ± 0.537644 (W 0.12984) | 🔵 faer-fork 4.39× faster than `faer-upstream` | 🔵 faer-fork 1.99× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.44933 ± 0.100074 (W 0.00557662) | 7.63877 ± 2.06218 (W 0.417686) | - | 🔵 faer-fork 3.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 31.5938 ± 0.0695836 (W 0.11567) | 69.3826 ± 1.56719 (W 1.15194) | 47.2691 ± 3.31095 (W 2.60254) | 🔵 faer-fork 2.20× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 32.4878 ± 0.0944545 (W 0.150038) | 70.4644 ± 0.34973 (W 3.21167) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.55039 ± 0.0915301 (W 0.00841971) | 7.53492 ± 0.154415 (W 0.136642) | 1.19509 ± 0.384462 (W 0.014796) | 🔵 faer-fork 4.86× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.69393 ± 0.158423 (W 0.00365851) | 3.74124 ± 1.6784 (W 0.179205) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 24.3227 ± 0.950143 (W 0.332861) | 92.1465 ± 2.70727 (W 4.85788) | 28.6954 ± 3.81447 (W 1.2135) | 🔵 faer-fork 3.79× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.1613 ± 0.27602 (W 0.110266) | 88.639 ± 7.7611 (W 13.5197) | - | 🔵 faer-fork 3.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.999738 ± 0.00453662 (W 0.00346639) | 1.24161 ± 0.00535155 (W 0.00230357) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.08769 ± 0.00833676 (W 0.00368108) | 1.16225 ± 0.0112951 (W 0.00376994) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 14.9521 ± 0.11476 (W 0.0332204) | 30.4581 ± 1.68846 (W 2.04385) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 18.71 ± 1.60177 (W 0.0323905) | 31.6837 ± 2.98835 (W 2.80484) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.508154 ± 0.00968146 (W 0.00546602) | 0.890442 ± 0.0197382 (W 0.051933) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.331296 ± 0.00570986 (W 0.00296555) | 0.911703 ± 0.00997744 (W 0.039435) | - | 🔵 faer-fork 2.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.55923 ± 0.0627096 (W 0.0817411) | 4.57578 ± 0.0936669 (W 0.189234) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.65414 ± 0.178048 (W 0.11717) | 4.70236 ± 0.0756994 (W 0.214863) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.331086 ± 0.00283161 (W 0.00382456) | 1.18771 ± 0.125109 (W 0.0160251) | 0.315525 ± 0.000134288 (W 0.00210776) | 🔵 faer-fork 3.59× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.412369 ± 0.00473127 (W 0.00334577) | 0.666452 ± 0.0281211 (W 0.00462066) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 5.16166 ± 0.0301235 (W 0.0228401) | 7.99612 ± 0.136404 (W 0.0733384) | 3.81959 ± 0.124134 (W 0.104459) | 🔵 faer-fork 1.55× faster than `faer-upstream` | 🔴 faer-fork 1.35× slower than `lapack-mkl` | `!` between-start range over 5% |
| `lu/4096x256/b1 [api-inclusive]` | 7.76002 ± 0.0121088 (W 0.0290194) | 9.9506 ± 0.187772 (W 0.0997958) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.13476 ± 0.155613 (W 0.00511278) | 8.31844 ± 0.91952 (W 0.448375) | 0.886032 ± 0.1116 (W 0.0158197) | 🔵 faer-fork 7.33× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.85361 ± 0.276078 (W 0.00738337) | 7.31419 ± 2.86914 (W 0.633438) | - | 🔵 faer-fork 3.95× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 17.6281 ± 0.582396 (W 0.542174) | 110.819 ± 1.32113 (W 4.80751) | 24.0828 ± 2.34174 (W 1.3769) | 🔵 faer-fork 6.29× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 26.368 ± 0.112498 (W 0.179813) | 128.983 ± 1.87399 (W 4.16817) | - | 🔵 faer-fork 4.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.03911 ± 0.156013 (W 0.00435738) | 4.38352 ± 2.22238 (W 0.297049) | 0.601628 ± 0.0876717 (W 0.0150211) | 🔵 faer-fork 4.22× faster than `faer-upstream` | 🔴 faer-fork 1.73× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/4096x256/b1` | 14.2553 ± 1.11233 (W 0.434225) | 87.8018 ± 1.4865 (W 6.12226) | 16.6416 ± 0.57534 (W 0.469799) | 🔵 faer-fork 6.16× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/1024x64/b1` | 2.37689 ± 0.225151 (W 0.00552301) | 11.0106 ± 0.0956866 (W 0.260568) | 4.98923 ± 0.579378 (W 0.106482) | 🔵 faer-fork 4.63× faster than `faer-upstream` | 🔵 faer-fork 2.10× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.56469 ± 0.245289 (W 0.00830593) | 6.70307 ± 1.18709 (W 0.486171) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 33.4642 ± 0.15767 (W 0.238481) | 156.702 ± 2.6803 (W 2.57376) | 60.5123 ± 1.44993 (W 1.49461) | 🔵 faer-fork 4.68× faster than `faer-upstream` | 🔵 faer-fork 1.81× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 34.2068 ± 0.456158 (W 0.114937) | 149.828 ± 3.79249 (W 5.30222) | - | 🔵 faer-fork 4.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.54487 ± 0.139012 (W 0.00432362) | 8.4121 ± 0.850776 (W 0.339136) | 1.21017 ± 0.0144441 (W 0.0117039) | 🔵 faer-fork 5.45× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.60618 ± 0.31926 (W 0.00434431) | 4.56916 ± 0.437883 (W 0.398689) | - | 🔵 faer-fork 2.84× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 27.2525 ± 0.956317 (W 1.21719) | 111.128 ± 2.16944 (W 1.6745) | 29.2194 ± 0.8046 (W 0.307698) | 🔵 faer-fork 4.08× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 26.8844 ± 0.688431 (W 1.23307) | 110.675 ± 0.574039 (W 2.88549) | - | 🔵 faer-fork 4.12× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.52151 ± 0.0104159 (W 0.00497384) | 3.34985 ± 0.000834081 (W 0.00522653) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.58274 ± 0.011131 (W 0.00332243) | 2.70875 ± 0.00199805 (W 0.00520205) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 23.7456 ± 0.639709 (W 0.599111) | 122.791 ± 0.488781 (W 0.115309) | - | 🔵 faer-fork 5.17× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 33.0811 ± 0.163221 (W 0.0565554) | 38.9877 ± 1.55138 (W 0.10314) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.506534 ± 0.00782788 (W 0.00738488) | 0.487703 ± 0.0287115 (W 0.0277732) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.516445 ± 0.00125681 (W 0.00769181) | 0.520915 ± 0.0959221 (W 0.0324186) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 15.6027 ± 0.104576 (W 0.130756) | 17.8765 ± 0.238061 (W 1.19677) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 15.5494 ± 0.146239 (W 0.0618041) | 17.797 ± 1.13524 (W 0.786865) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 1.00459 ± 0.00255375 (W 0.0045399) | 1.75458 ± 0.0139607 (W 0.0106372) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 1.10017 ± 0.00318004 (W 0.0041686) | 1.1662 ± 6.04934 (W 0.0103023) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 14.74 ± 0.0667209 (W 0.152643) | 84.5068 ± 1.52722 (W 0.0751667) | - | 🔵 faer-fork 5.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 24.447 ± 0.463142 (W 0.0229532) | 29.3012 ± 0.0429161 (W 0.0380531) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.39791 ± 0.0424819 (W 0.0107599) | 7.51779 ± 0.840394 (W 0.286096) | - | 🔵 faer-fork 3.14× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 3.43251 ± 0.013529 (W 0.0111243) | 4.92841 ± 0.720718 (W 0.211387) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 46.429 ± 0.212926 (W 0.422851) | 207.717 ± 4.94434 (W 0.818655) | - | 🔵 faer-fork 4.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 65.778 ± 0.271878 (W 0.497752) | 103.522 ± 0.326462 (W 0.455134) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.98038 ± 0.0168344 (W 0.00953669) | 5.21778 ± 0.218267 (W 0.136301) | - | 🔵 faer-fork 2.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 27.7341 ± 0.0962654 (W 0.0588256) | 183.507 ± 1.77722 (W 0.401489) | - | 🔵 faer-fork 6.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.60851 ± 0.00897789 (W 0.0111835) | 6.96778 ± 0.56271 (W 0.100875) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.70677 ± 0.0102311 (W 0.0117154) | 6.37891 ± 0.178376 (W 0.234246) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 71.1844 ± 0.08507 (W 0.835657) | 117.844 ± 1.09437 (W 0.770975) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 71.3351 ± 0.239203 (W 0.816973) | 119.69 ± 0.326758 (W 8.39882) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.50865 ± 0.0112678 (W 0.00635435) | 6.49401 ± 0.779778 (W 0.239683) | - | 🔵 faer-fork 2.59× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.52489 ± 0.0277495 (W 0.00674457) | 3.96732 ± 0.64218 (W 0.0578324) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 50.0028 ± 0.318436 (W 0.0661476) | 88.0841 ± 0.620041 (W 0.324554) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 49.6919 ± 0.210916 (W 0.0893319) | 88.9558 ± 1.08349 (W 4.84034) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.60365 ± 0.0126476 (W 0.00796818) | 3.35639 ± 0.0112727 (W 0.00327508) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.69749 ± 0.00682015 (W 0.00466485) | 2.7031 ± 0.00966566 (W 0.00772525) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 20.606 ± 1.72691 (W 0.649796) | 112.992 ± 0.19969 (W 0.13794) | - | 🔵 faer-fork 5.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 36.9154 ± 1.33707 (W 2.27136) | 40.6875 ± 3.80741 (W 2.94578) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.474553 ± 0.120743 (W 0.00506569) | 1.00782 ± 0.32628 (W 0.0791174) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.489773 ± 0.0243433 (W 0.00486458) | 1.02998 ± 0.0208533 (W 0.0451757) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.1234 ± 0.0703728 (W 0.138297) | 10.7889 ± 0.119129 (W 0.263279) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.1228 ± 0.0948094 (W 0.0889349) | 10.6262 ± 0.130567 (W 0.14049) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.979466 ± 0.00191532 (W 0.00467998) | 2.53031 ± 0.672419 (W 0.0143142) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 1.08603 ± 0.00963897 (W 0.00364528) | 1.17811 ± 0.0192706 (W 0.0102768) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 12.547 ± 0.0465704 (W 0.276925) | 89.9088 ± 0.570283 (W 0.0846727) | - | 🔵 faer-fork 7.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 23.5441 ± 0.308394 (W 0.0364411) | 27.9114 ± 0.0782373 (W 0.0274544) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.36839 ± 0.0466377 (W 0.0123096) | 11.925 ± 2.95985 (W 0.611642) | - | 🔵 faer-fork 5.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 3.40892 ± 0.0338791 (W 0.0142009) | 4.57291 ± 0.113717 (W 19.3796) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 43.1759 ± 0.18944 (W 0.700699) | 200.299 ± 6.69568 (W 0.876964) | - | 🔵 faer-fork 4.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 62.1012 ± 0.167785 (W 0.185557) | 87.5152 ± 7.1475 (W 7.11768) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.91419 ± 0.024259 (W 0.00314469) | 5.83744 ± 0.489392 (W 0.13354) | - | 🔵 faer-fork 3.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 26.629 ± 0.105749 (W 0.0431607) | 183.169 ± 1.84854 (W 0.322517) | - | 🔵 faer-fork 6.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.61199 ± 0.00423904 (W 0.0120178) | 10.9113 ± 0.209312 (W 1.30995) | - | 🔵 faer-fork 3.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.76332 ± 0.00176868 (W 0.00523936) | 6.52721 ± 0.0203811 (W 0.553708) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 66.712 ± 0.199198 (W 0.163991) | 114.085 ± 0.806956 (W 0.407376) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 69.0207 ± 0.0861817 (W 0.0692067) | 116.159 ± 0.187637 (W 9.34147) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.53266 ± 0.0391418 (W 0.0038581) | 10.7565 ± 0.153214 (W 0.393582) | - | 🔵 faer-fork 4.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.53228 ± 0.111935 (W 0.0140265) | 7.40093 ± 0.669894 (W 0.705007) | - | 🔵 faer-fork 2.92× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 50.1723 ± 0.263797 (W 0.119696) | 91.5084 ± 0.729783 (W 0.785565) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 50.3593 ± 0.525019 (W 0.0714833) | 91.7552 ± 0.302799 (W 4.65348) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.89036 ± 0.0983578 (W 0.00934144) | 3.41992 ± 0.079358 (W 0.00814937) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.94159 ± 0.0151328 (W 0.00730166) | 2.70059 ± 0.0134126 (W 0.0035474) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 22.3856 ± 0.454374 (W 0.0885952) | 111.947 ± 0.134729 (W 0.140179) | - | 🔵 faer-fork 5.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 48.685 ± 1.45713 (W 4.28756) | 53.2585 ± 1.01275 (W 4.42247) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.393939 ± 0.213043 (W 0.0149906) | 1.18878 ± 0.413638 (W 0.0562132) | - | 🔵 faer-fork 3.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.390109 ± 0.0732253 (W 0.00675176) | 1.25657 ± 0.0669153 (W 0.0633065) | - | 🔵 faer-fork 3.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.9321 ± 0.102735 (W 0.0810321) | 8.02061 ± 0.0448637 (W 0.0622241) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 10.0117 ± 0.131166 (W 0.108801) | 7.95721 ± 0.0233032 (W 0.0679228) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.940514 ± 0.0116491 (W 0.00247316) | 3.51519 ± 0.582822 (W 0.0166201) | - | 🔵 faer-fork 3.74× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 1.04701 ± 0.00155234 (W 0.00397251) | 1.19543 ± 6.36536 (W 0.0227447) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 12.4593 ± 0.0218621 (W 0.0317255) | 90.4319 ± 0.289529 (W 0.0873028) | - | 🔵 faer-fork 7.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 23.5118 ± 0.418334 (W 0.0296889) | 27.3136 ± 0.140592 (W 0.0664063) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.65327 ± 0.047216 (W 0.0111532) | 17.7621 ± 1.79763 (W 1.01116) | - | 🔵 faer-fork 6.69× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 3.57841 ± 0.327841 (W 0.0101586) | 6.36523 ± 1.77082 (W 0.532667) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 41.6405 ± 0.195933 (W 0.742954) | 200.188 ± 4.70307 (W 0.681958) | - | 🔵 faer-fork 4.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 64.0848 ± 0.59065 (W 0.108244) | 94.6918 ± 3.3369 (W 3.8175) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 2.07261 ± 0.301296 (W 0.00935907) | 7.74745 ± 3.25684 (W 1.18727) | - | 🔵 faer-fork 3.74× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 28.9924 ± 0.483398 (W 0.176815) | 186.832 ± 0.622296 (W 0.57942) | - | 🔵 faer-fork 6.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.88803 ± 0.0752239 (W 0.00534028) | 18.6247 ± 2.20541 (W 0.652861) | - | 🔵 faer-fork 4.79× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.90104 ± 0.0999983 (W 0.010326) | 10.3588 ± 0.621055 (W 0.489816) | - | 🔵 faer-fork 2.66× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 69.2335 ± 0.130671 (W 0.226894) | 114.38 ± 0.691703 (W 0.621838) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 71.5761 ± 0.337777 (W 0.186035) | 117.288 ± 0.406443 (W 9.7979) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.70776 ± 0.0692066 (W 0.00876591) | 12.224 ± 0.914962 (W 0.669901) | - | 🔵 faer-fork 4.51× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.6767 ± 0.253815 (W 0.0099725) | 8.48053 ± 1.76199 (W 0.381859) | - | 🔵 faer-fork 3.17× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 53.7364 ± 0.482145 (W 0.244192) | 119.531 ± 1.15158 (W 4.20741) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 54.1614 ± 0.378804 (W 0.275996) | 118.297 ± 5.89323 (W 5.77296) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.23021 ± 0.01431 (W 0.0045764) | 2.59663 ± 0.00234303 (W 0.00345609) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.4743 ± 0.00784515 (W 0.00658147) | 2.57921 ± 0.0108122 (W 0.00353845) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 23.297 ± 0.491333 (W 0.043134) | 25.4512 ± 0.0815741 (W 0.055716) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 27.1641 ± 0.357333 (W 0.0534986) | 30.0334 ± 1.13526 (W 0.0597151) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.395029 ± 0.00391107 (W 0.00683469) | 0.545076 ± 0.0636415 (W 0.0177913) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.386428 ± 0.00390008 (W 0.00858592) | 0.897703 ± 0.18862 (W 0.101837) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.075 ± 0.159942 (W 0.13668) | 18.5012 ± 1.11476 (W 1.68251) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 16.8777 ± 0.195512 (W 0.24115) | 16.5237 ± 0.131665 (W 1.08286) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.700213 ± 0.0133546 (W 0.00474864) | 0.852491 ± 0.00426244 (W 0.011433) | 1.40995 ± 0.0924703 (W 0.00412554) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 2.01× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.966256 ± 0.00619484 (W 0.0049992) | 1.09815 ± 0.025412 (W 0.00824777) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.4138 ± 0.217042 (W 0.158176) | 15.605 ± 0.175377 (W 0.0667737) | 57.2107 ± 6.70278 (W 0.0450685) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 4.27× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/4096x256/b1 [api-inclusive]` | 17.9456 ± 0.163837 (W 0.0334356) | 20.5512 ± 0.0524846 (W 0.0405092) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.00588 ± 0.00748338 (W 0.0161552) | 5.0619 ± 0.524532 (W 0.132856) | 2.40041 ± 0.120605 (W 0.00570276) | 🔵 faer-fork 2.52× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 3.2404 ± 0.0478532 (W 0.0129041) | 4.0255 ± 0.049197 (W 0.0725429) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 37.8234 ± 0.0657295 (W 0.287199) | 69.7849 ± 1.48402 (W 0.60025) | 30.6546 ± 0.385056 (W 0.315629) | 🔵 faer-fork 1.85× faster than `faer-upstream` | 🔴 faer-fork 1.23× slower than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 58.1167 ± 0.243972 (W 0.468941) | 89.9139 ± 9.21807 (W 1.78503) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.72122 ± 0.0175503 (W 0.00737306) | 3.65958 ± 0.567935 (W 0.134876) | 2.90146 ± 0.174054 (W 0.0561489) | 🔵 faer-fork 2.13× faster than `faer-upstream` | 🔵 faer-fork 1.69× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/4096x256/b1` | 25.6767 ± 0.271121 (W 0.0710097) | 48.9963 ± 0.86651 (W 0.148532) | 14.1534 ± 0.553767 (W 0.227668) | 🔵 faer-fork 1.91× faster than `faer-upstream` | 🔴 faer-fork 1.81× slower than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/1024x64/b1` | 3.59103 ± 0.00483704 (W 0.0080294) | 7.07292 ± 0.361834 (W 0.243227) | 6.09748 ± 0.0793532 (W 0.0552285) | 🔵 faer-fork 1.97× faster than `faer-upstream` | 🔵 faer-fork 1.70× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.69165 ± 0.00474917 (W 0.015257) | 6.31309 ± 1.16812 (W 0.201988) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 59.853 ± 0.131142 (W 0.534525) | 107.082 ± 0.421465 (W 0.38778) | 75.6002 ± 1.64692 (W 0.266016) | 🔵 faer-fork 1.79× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 60.705 ± 0.117556 (W 0.539255) | 107.887 ± 0.193906 (W 0.921564) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.47012 ± 0.0128889 (W 0.00772273) | 6.87427 ± 1.17915 (W 0.182776) | 3.1388 ± 0.0283395 (W 0.00906724) | 🔵 faer-fork 2.78× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.49399 ± 0.00270935 (W 0.00575033) | 3.9988 ± 0.810047 (W 0.228529) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 40.9854 ± 0.408968 (W 0.0640697) | 79.0854 ± 0.310376 (W 0.435697) | 32.0133 ± 1.43198 (W 0.127506) | 🔵 faer-fork 1.93× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 40.8778 ± 0.324854 (W 0.0872157) | 79.3438 ± 1.11612 (W 3.19573) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.34617 ± 0.0373902 (W 0.00805747) | 2.59858 ± 0.00922864 (W 0.00372964) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.59562 ± 0.0192313 (W 0.0122706) | 2.57307 ± 0.0214368 (W 0.00706496) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 29.8327 ± 4.79147 (W 4.34253) | 27.7546 ± 1.03038 (W 2.33513) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 33.508 ± 0.525267 (W 2.75339) | 37.6511 ± 6.26157 (W 2.57071) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.386828 ± 0.00313095 (W 0.00433833) | 0.480633 ± 0.375225 (W 0.0302245) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.377898 ± 0.0111137 (W 0.00284264) | 1.24634 ± 0.0851933 (W 0.11143) | - | 🔵 faer-fork 3.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.4166 ± 0.220888 (W 0.073929) | 12.0497 ± 0.0626485 (W 0.124007) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.1504 ± 0.0632942 (W 0.113154) | 12.2305 ± 0.119206 (W 0.326253) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.695293 ± 0.0139109 (W 0.0064526) | 0.933385 ± 0.00860693 (W 0.00903717) | 1.29636 ± 0.0368926 (W 0.00283094) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.86× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.948785 ± 0.0204143 (W 0.00388393) | 1.13559 ± 10.0936 (W 5.81482) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 10.7631 ± 0.46339 (W 0.0160449) | 14.157 ± 0.0364824 (W 0.0259146) | 57.7749 ± 10.092 (W 0.0733805) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔵 faer-fork 5.37× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/4096x256/b1 [api-inclusive]` | 15.9679 ± 0.068159 (W 0.0248185) | 19.4563 ± 0.192616 (W 0.0618012) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.00623 ± 0.021198 (W 0.00836386) | 10.7573 ± 1.99778 (W 0.71496) | 2.34015 ± 0.131784 (W 0.0223734) | 🔵 faer-fork 5.36× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 3.23789 ± 0.022408 (W 0.00772715) | 4.87535 ± 2.8028 (W 0.388661) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 32.1547 ± 1.0768 (W 0.167637) | 67.1951 ± 0.412246 (W 1.70891) | 32.209 ± 4.26302 (W 0.0589049) | 🔵 faer-fork 2.09× faster than `faer-upstream` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 54.054 ± 0.319207 (W 0.210379) | 75.806 ± 0.933574 (W 5.38201) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.97503 ± 0.219828 (W 0.00932043) | 8.02661 ± 2.77385 (W 0.610577) | 2.41434 ± 0.166555 (W 0.0904067) | 🔵 faer-fork 4.06× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/4096x256/b1` | 26.4888 ± 2.69527 (W 0.133335) | 50.9103 ± 0.259992 (W 0.332975) | 11.1462 ± 0.0605858 (W 0.0556256) | 🔵 faer-fork 1.92× faster than `faer-upstream` | 🔴 faer-fork 2.38× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/1024x64/b1` | 3.63367 ± 0.0490802 (W 0.0127742) | 11.3669 ± 1.3007 (W 1.68154) | 8.16433 ± 1.09593 (W 0.113893) | 🔵 faer-fork 3.13× faster than `faer-upstream` | 🔵 faer-fork 2.25× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.78876 ± 0.0911821 (W 0.00778822) | 6.31627 ± 2.0418 (W 0.504505) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 58.6048 ± 1.24613 (W 0.109435) | 105.27 ± 0.109465 (W 0.310388) | 63.7035 ± 1.14837 (W 0.174436) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 59.9766 ± 0.491153 (W 0.117045) | 107.352 ± 1.47861 (W 7.43867) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.51457 ± 0.0153234 (W 0.00550424) | 10.788 ± 0.227933 (W 0.457678) | 4.3231 ± 0.639965 (W 0.143239) | 🔵 faer-fork 4.29× faster than `faer-upstream` | 🔵 faer-fork 1.72× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.72841 ± 0.121795 (W 0.00722147) | 4.49056 ± 1.42159 (W 0.43098) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 41.8981 ± 0.023296 (W 0.162166) | 82.3557 ± 0.302313 (W 0.367543) | 33.0744 ± 0.311715 (W 0.132125) | 🔵 faer-fork 1.97× faster than `faer-upstream` | 🔴 faer-fork 1.27× slower than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 41.9573 ± 0.0180568 (W 0.183177) | 83.1756 ± 0.776127 (W 2.55016) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.5602 ± 0.0138091 (W 0.0077937) | 2.59495 ± 0.0168761 (W 0.00413501) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.84588 ± 0.00515322 (W 0.0064416) | 2.57248 ± 0.0148558 (W 0.00314061) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 37.9772 ± 1.35321 (W 1.96283) | 42.3985 ± 1.14867 (W 3.46974) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 42.7052 ± 0.661357 (W 3.83686) | 47.3445 ± 6.7614 (W 4.17035) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.726214 ± 0.214341 (W 0.00297225) | 1.28289 ± 0.0756863 (W 0.112699) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.719634 ± 0.130046 (W 0.00453926) | 1.36943 ± 0.0659737 (W 0.0881475) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.9411 ± 0.115752 (W 0.125156) | 8.69972 ± 0.190746 (W 0.334921) | - | 🔴 faer-fork 1.49× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.9075 ± 13.5383 (W 0.197354) | 8.90477 ± 0.300312 (W 0.36539) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.699763 ± 0.0039016 (W 0.00609035) | 1.50585 ± 0.00646537 (W 0.0284257) | 1.29766 ± 0.0250963 (W 0.0029229) | 🔵 faer-fork 2.15× faster than `faer-upstream` | 🔵 faer-fork 1.85× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.919174 ± 0.00768957 (W 0.0069977) | 1.13396 ± 0.0245692 (W 0.0148721) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 9.85822 ± 0.0760959 (W 0.0222838) | 13.5608 ± 0.0667395 (W 0.0619317) | 73.6385 ± 11.3744 (W 0.0700474) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔵 faer-fork 7.47× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/4096x256/b1 [api-inclusive]` | 15.1328 ± 0.014401 (W 0.0164076) | 19.0017 ± 0.101902 (W 0.0326932) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.1949 ± 0.0808398 (W 0.0105294) | 7.95101 ± 0.197889 (W 0.698168) | 2.27111 ± 0.0950865 (W 0.0520263) | 🔵 faer-fork 3.62× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 3.44243 ± 0.301614 (W 0.0110022) | 9.37082 ± 1.82235 (W 1.43453) | - | 🔵 faer-fork 2.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 34.5255 ± 1.13181 (W 0.0930996) | 64.1854 ± 0.454908 (W 1.10798) | 29.3236 ± 0.0591779 (W 0.11823) | 🔵 faer-fork 1.86× faster than `faer-upstream` | 🔴 faer-fork 1.18× slower than `lapack-mkl` | `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 56.0047 ± 0.377109 (W 0.515622) | 85.7965 ± 3.08672 (W 3.56284) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.88605 ± 0.0914564 (W 0.00653167) | 4.78963 ± 0.853128 (W 0.368644) | 1.24294 ± 0.0492525 (W 0.0116375) | 🔵 faer-fork 2.54× faster than `faer-upstream` | 🔴 faer-fork 1.52× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/4096x256/b1` | 28.5117 ± 0.539535 (W 0.421143) | 59.8238 ± 4.71528 (W 2.98695) | 15.4596 ± 0.963586 (W 1.75086) | 🔵 faer-fork 2.10× faster than `faer-upstream` | 🔴 faer-fork 1.84× slower than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/1024x64/b1` | 3.85759 ± 0.0712432 (W 0.011534) | 18.1123 ± 0.512049 (W 0.575678) | 6.85587 ± 0.451836 (W 0.0774089) | 🔵 faer-fork 4.70× faster than `faer-upstream` | 🔵 faer-fork 1.78× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.91155 ± 0.257436 (W 0.0104614) | 10.4993 ± 1.64529 (W 0.581947) | - | 🔵 faer-fork 2.68× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 61.1424 ± 2.071 (W 0.152805) | 105.925 ± 0.614051 (W 1.41512) | 78.5049 ± 9.40915 (W 4.60937) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 62.9833 ± 0.213793 (W 0.129031) | 107.894 ± 1.31358 (W 8.30163) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.69012 ± 0.010899 (W 0.00408596) | 13.2566 ± 1.3154 (W 0.87037) | 3.23714 ± 0.0957765 (W 0.0133532) | 🔵 faer-fork 4.93× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.77554 ± 0.085881 (W 0.00369646) | 7.55198 ± 2.06685 (W 0.610463) | - | 🔵 faer-fork 2.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 47.0693 ± 0.874051 (W 0.834232) | 110.338 ± 4.57943 (W 4.04671) | 41.3361 ± 2.04608 (W 2.52477) | 🔵 faer-fork 2.34× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 47.473 ± 0.627436 (W 1.66253) | 112.277 ± 4.46847 (W 3.50719) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `lu/1024x64`, `lu/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `solve/1024x64`, `solve/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 14.2939 ± 0.673983 (W 0.0118193) | 15.8906 ± 0.255277 (W 0.022835) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 34.6542 ± 0.353106 (W 0.117678) | 43.3274 ± 3.01852 (W 0.0761498) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.910913 ± 0.0094509 (W 0.00561734) | 1.22099 ± 0.000604759 (W 0.00396028) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.62821 ± 0.0239015 (W 0.00642919) | 2.05171 ± 0.2335 (W 0.00926414) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 2.7782 ± 0.242384 (W 0.016924) | 23.8416 ± 21.8253 (W 0.145103) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 11.6216 ± 1.42548 (W 0.148908) | 18.9567 ± 82.1668 (W 0.370609) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.150317 ± 0.00151302 (W 0.00307716) | 1.13805 ± 0.0068015 (W 0.0176725) | - | 🔵 faer-fork 7.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.337826 ± 0.0020044 (W 0.00279412) | 0.652631 ± 0.0192182 (W 0.0059342) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 10.495 ± 6.97096 (W 0.0424547) | 16.9941 ± 0.161398 (W 0.362216) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 9.99379 ± 0.146956 (W 0.137449) | 24.2779 ± 2.72301 (W 0.186405) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 1.06724 ± 0.0334157 (W 0.01232) | 1.64498 ± 0.0300188 (W 0.0149171) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.510534 ± 0.00125291 (W 0.0040057) | 1.31396 ± 0.0714478 (W 0.00829915) | - | 🔵 faer-fork 2.57× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 10.0972 ± 2.09132 (W 0.0580377) | 16.0955 ± 2.48083 (W 0.155832) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 1.05413 ± 0.0315824 (W 0.0116661) | 1.56175 ± 0.0046205 (W 0.0078026) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 31.6588 ± 0.097354 (W 0.0471977) | 67.3519 ± 1.55772 (W 0.98955) | - | 🔵 faer-fork 2.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 32.3331 ± 0.114507 (W 0.0734664) | 68.0782 ± 0.551527 (W 1.38183) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.17149 ± 0.0203974 (W 0.00519405) | 5.13555 ± 1.2723 (W 0.162982) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.25421 ± 0.0320033 (W 0.00829119) | 4.00368 ± 0.0453298 (W 0.0320242) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 22.6712 ± 0.0163408 (W 0.0295606) | 50.6985 ± 0.340643 (W 0.269221) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 22.8649 ± 0.0220088 (W 0.0470062) | 49.6841 ± 0.755502 (W 0.880475) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.44656 ± 0.0219555 (W 0.0048337) | 4.31048 ± 0.600008 (W 0.128284) | - | 🔵 faer-fork 2.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.46406 ± 0.0175888 (W 0.00305153) | 3.87039 ± 0.287755 (W 0.102012) | - | 🔵 faer-fork 2.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 13.7106 ± 0.084259 (W 0.0337427) | 15.6192 ± 5.18282 (W 0.269381) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 32.3629 ± 0.0828607 (W 0.156758) | 40.6153 ± 3.93115 (W 0.102406) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.944525 ± 0.0127604 (W 0.00887745) | 1.23478 ± 0.0156776 (W 0.00237665) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.70359 ± 0.0423508 (W 0.0191345) | 2.07366 ± 0.0101222 (W 0.00480376) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.304 ± 0.0114746 (W 0.0190398) | 15.7092 ± 0.733455 (W 0.12281) | - | 🔵 faer-fork 6.82× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 8.87302 ± 0.168017 (W 0.175051) | 16.6791 ± 0.922976 (W 0.177627) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.156167 ± 0.00516693 (W 0.00473225) | 0.921684 ± 0.0141552 (W 0.031689) | - | 🔵 faer-fork 5.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.347856 ± 0.00639688 (W 0.00283524) | 0.716264 ± 0.0550869 (W 0.0158599) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 10.1889 ± 0.414171 (W 0.0292339) | 17.4955 ± 0.0359323 (W 0.112406) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 8.99905 ± 0.153441 (W 0.24717) | 24.8976 ± 0.813637 (W 2.4858) | - | 🔵 faer-fork 2.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 1.0607 ± 0.0045012 (W 0.0132132) | 2.42705 ± 0.18518 (W 0.0549317) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.513214 ± 0.0408238 (W 0.00654894) | 1.75049 ± 0.518552 (W 0.143877) | - | 🔵 faer-fork 3.41× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 9.78965 ± 0.0488338 (W 0.0431904) | 16.3422 ± 0.0787945 (W 0.0673893) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.00696 ± 0.0409382 (W 0.0104225) | 1.76662 ± 0.0286098 (W 0.0180218) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 31.4774 ± 0.0266149 (W 0.132899) | 68.885 ± 0.561454 (W 1.38554) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 32.4655 ± 0.148288 (W 0.185716) | 70.7328 ± 0.153535 (W 3.18792) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.17116 ± 0.00822017 (W 0.00462903) | 10.0775 ± 0.243894 (W 0.32452) | - | 🔵 faer-fork 4.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.28552 ± 0.0242478 (W 0.0049309) | 7.74777 ± 1.85602 (W 0.557537) | - | 🔵 faer-fork 3.39× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 24.2939 ± 0.265468 (W 0.0887813) | 88.2211 ± 2.34427 (W 4.65176) | - | 🔵 faer-fork 3.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.2418 ± 0.205609 (W 0.211452) | 88.2613 ± 5.60541 (W 9.86479) | - | 🔵 faer-fork 3.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.44443 ± 0.106307 (W 0.00374389) | 7.52413 ± 0.0793627 (W 0.209709) | - | 🔵 faer-fork 5.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.44769 ± 0.21111 (W 0.00433989) | 3.48795 ± 2.33941 (W 16.5467) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 15.0027 ± 0.0862148 (W 0.11566) | 29.7412 ± 0.50053 (W 0.874087) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 34.5463 ± 0.293943 (W 0.215555) | 43.4363 ± 4.8075 (W 0.289482) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.985516 ± 0.090473 (W 0.00606891) | 1.21666 ± 0.0080012 (W 0.00320113) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.89686 ± 0.236495 (W 0.0159064) | 2.10354 ± 0.0768796 (W 0.00429762) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 2.42397 ± 0.070875 (W 0.0438536) | 15.1577 ± 0.394282 (W 0.161613) | - | 🔵 faer-fork 6.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 10.2367 ± 0.952777 (W 0.0713576) | 17.8944 ± 0.120394 (W 0.240817) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.166347 ± 0.00531401 (W 0.00591741) | 0.827429 ± 0.0262549 (W 0.0207419) | - | 🔵 faer-fork 4.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.354496 ± 0.00416212 (W 0.00317701) | 1.63231 ± 3.5369 (W 0.0376672) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 10.2117 ± 0.342987 (W 0.0503143) | 26.9087 ± 0.478878 (W 0.53318) | - | 🔵 faer-fork 2.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 9.96797 ± 0.430207 (W 0.232629) | 36.0722 ± 2.73569 (W 1.6804) | - | 🔵 faer-fork 3.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 1.07632 ± 0.0135486 (W 0.0111343) | 2.95008 ± 0.103887 (W 0.107394) | - | 🔵 faer-fork 2.74× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.532675 ± 0.00994469 (W 0.00241871) | 2.15424 ± 0.671327 (W 0.198152) | - | 🔵 faer-fork 4.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 9.90807 ± 0.0476939 (W 0.0243291) | 24.6343 ± 2.17676 (W 0.727554) | - | 🔵 faer-fork 2.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 1.01423 ± 0.0866686 (W 0.00681454) | 1.96157 ± 0.418516 (W 0.0333835) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 33.1998 ± 0.45956 (W 0.318684) | 154.087 ± 2.39033 (W 5.05592) | - | 🔵 faer-fork 4.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 34.4798 ± 0.096337 (W 0.238973) | 149.341 ± 1.97002 (W 9.15761) | - | 🔵 faer-fork 4.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.31283 ± 0.0521633 (W 0.0135363) | 10.9845 ± 0.424202 (W 0.227681) | - | 🔵 faer-fork 4.75× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.46003 ± 0.0833587 (W 0.00634724) | 9.2887 ± 3.05407 (W 0.248457) | - | 🔵 faer-fork 3.78× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 26.8543 ± 1.12816 (W 0.237093) | 112.812 ± 2.21682 (W 1.55745) | - | 🔵 faer-fork 4.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 26.9906 ± 1.52217 (W 0.945483) | 107.411 ± 1.85585 (W 5.08359) | - | 🔵 faer-fork 3.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.60833 ± 0.0381345 (W 0.00538577) | 9.32735 ± 0.208373 (W 0.359443) | - | 🔵 faer-fork 5.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.87216 ± 0.186032 (W 0.00544689) | 8.17305 ± 1.66457 (W 0.292668) | - | 🔵 faer-fork 4.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 28.7849 ± 0.0925704 (W 0.0447698) | 30.7017 ± 0.333117 (W 0.0368752) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 30.0964 ± 0.114327 (W 0.0426271) | 31.7057 ± 0.0978939 (W 0.0443335) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.43579 ± 0.0299215 (W 0.00833331) | 1.69019 ± 0.023666 (W 0.00696399) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.54551 ± 0.0274611 (W 0.0304068) | 1.63092 ± 0.00245681 (W 0.00685375) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.40366 ± 0.0570549 (W 0.0600856) | 3.42794 ± 0.169992 (W 0.133397) | 2.80089 ± 0.0152047 (W 0.0148614) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.61424 ± 0.0042587 (W 0.0162213) | 5.20103 ± 0.326854 (W 0.324597) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.101055 ± 0.00477227 (W 0.00137177) | 0.181569 ± 0.0116633 (W 0.0158922) | 0.220001 ± 0.0257519 (W 0.00393993) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔵 faer-fork 2.18× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.172268 ± 0.0125423 (W 0.00197686) | 0.264042 ± 7.35419 (W 0.013393) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 3.13817 ± 0.0139566 (W 0.00703506) | 9.99747 ± 0.0389616 (W 0.111073) | 4.7545 ± 0.0270486 (W 0.0716426) | 🔵 faer-fork 3.19× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 4.97291 ± 0.222088 (W 0.0283508) | 11.5358 ± 0.061118 (W 0.269531) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.251832 ± 0.000339559 (W 0.00182155) | 0.865191 ± 0.0269246 (W 0.0267649) | 0.285074 ± 0.000815596 (W 0.00402471) | 🔵 faer-fork 3.44× faster than `faer-upstream` | 🔵 faer-fork 1.13× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.344696 ± 0.00284583 (W 0.00293409) | 0.969756 ± 0.0250021 (W 0.019413) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 2.88185 ± 0.0121207 (W 0.0122161) | 8.90454 ± 0.0873905 (W 0.194286) | 4.27033 ± 0.0110369 (W 0.0860804) | 🔵 faer-fork 3.09× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.21683 ± 0.00213672 (W 0.00245551) | 0.806069 ± 0.034636 (W 0.0160241) | 0.232541 ± 0.00228164 (W 0.00377732) | 🔵 faer-fork 3.72× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 33.6248 ± 0.187595 (W 0.0651303) | 70.8749 ± 0.210975 (W 1.40335) | 123.865 ± 0.353739 (W 0.234533) | 🔵 faer-fork 2.11× faster than `faer-upstream` | 🔵 faer-fork 3.68× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 34.4142 ± 0.104911 (W 0.0561088) | 70.8344 ± 1.21499 (W 4.17197) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.18354 ± 0.000705762 (W 0.00431816) | 4.44182 ± 0.90356 (W 0.0965108) | 5.44473 ± 0.0625671 (W 0.0332687) | 🔵 faer-fork 2.03× faster than `faer-upstream` | 🔵 faer-fork 2.49× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.25463 ± 0.00396052 (W 0.00474336) | 4.60013 ± 0.397501 (W 0.0934672) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 24.2649 ± 0.0448555 (W 0.0417014) | 52.4852 ± 0.436953 (W 0.178636) | 99.4596 ± 0.455285 (W 0.0568816) | 🔵 faer-fork 2.16× faster than `faer-upstream` | 🔵 faer-fork 4.10× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.4276 ± 0.0390015 (W 0.0533466) | 51.7299 ± 0.444526 (W 2.5124) | - | 🔵 faer-fork 2.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.4436 ± 0.00395731 (W 0.00497587) | 3.67478 ± 0.25303 (W 0.105573) | 3.551 ± 0.0505168 (W 0.00511754) | 🔵 faer-fork 2.55× faster than `faer-upstream` | 🔵 faer-fork 2.46× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.4656 ± 0.00233304 (W 0.00531665) | 3.51763 ± 0.448054 (W 0.0553026) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 26.2356 ± 0.053963 (W 0.0264474) | 29.3925 ± 0.601275 (W 0.206011) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 28.1448 ± 0.0527726 (W 0.0267313) | 29.8486 ± 0.0642885 (W 0.0571169) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.53875 ± 0.0305941 (W 0.00870694) | 1.67224 ± 0.01862 (W 0.00430347) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.55187 ± 0.0374013 (W 0.00701458) | 1.60574 ± 0.0173849 (W 0.00575633) | - | · | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.12413 ± 0.112892 (W 0.0219986) | 3.54561 ± 0.333589 (W 0.117371) | 2.78108 ± 0.00954809 (W 0.0172259) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.4659 ± 0.0338136 (W 0.065059) | 5.78855 ± 0.134599 (W 0.130443) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.106685 ± 0.00203385 (W 0.00204947) | 0.346606 ± 0.0171312 (W 0.016423) | 0.263782 ± 0.0244153 (W 0.0121828) | 🔵 faer-fork 3.25× faster than `faer-upstream` | 🔵 faer-fork 2.47× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.185619 ± 0.00368399 (W 0.00275857) | 0.433061 ± 0.285778 (W 0.0254936) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 2.09264 ± 0.0483305 (W 0.0463511) | 9.84592 ± 0.254575 (W 0.139568) | 2.58718 ± 0.0614144 (W 0.014116) | 🔵 faer-fork 4.71× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 4.2521 ± 0.0505542 (W 0.074346) | 11.7071 ± 0.265081 (W 0.216736) | - | 🔵 faer-fork 2.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.235931 ± 0.00222448 (W 0.00218552) | 1.8486 ± 0.0602338 (W 0.124014) | 0.306404 ± 0.0350684 (W 0.00851441) | 🔵 faer-fork 7.84× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.373868 ± 0.00999301 (W 0.00287326) | 1.16256 ± 0.0175258 (W 0.0305158) | - | 🔵 faer-fork 3.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 1.87032 ± 0.0755631 (W 0.0306433) | 8.75762 ± 0.0756492 (W 0.153) | 2.31236 ± 0.0118746 (W 0.0154167) | 🔵 faer-fork 4.68× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.199989 ± 0.00154031 (W 0.00147739) | 1.71774 ± 0.108667 (W 0.144318) | 0.252382 ± 0.0246767 (W 0.0096133) | 🔵 faer-fork 8.59× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/256x4096/b1` | 33.1547 ± 1.85698 (W 0.0750061) | 70.8937 ± 0.989134 (W 1.39556) | 126.758 ± 0.980811 (W 0.612957) | 🔵 faer-fork 2.14× faster than `faer-upstream` | 🔵 faer-fork 3.82× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 33.851 ± 0.173141 (W 0.06991) | 72.6911 ± 0.249867 (W 4.48688) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.1972 ± 0.0412766 (W 0.00347919) | 10.0662 ± 0.171111 (W 0.358661) | 5.86147 ± 0.0161535 (W 0.0449249) | 🔵 faer-fork 4.58× faster than `faer-upstream` | 🔵 faer-fork 2.67× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.28729 ± 0.0488876 (W 0.00581222) | 7.31624 ± 2.41695 (W 0.319854) | - | 🔵 faer-fork 3.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 25.3824 ± 0.371933 (W 0.282601) | 87.1488 ± 5.50915 (W 5.08926) | 103.919 ± 1.29624 (W 2.48115) | 🔵 faer-fork 3.43× faster than `faer-upstream` | 🔵 faer-fork 4.09× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 25.3208 ± 0.311055 (W 0.0876023) | 73.545 ± 5.54353 (W 9.68701) | - | 🔵 faer-fork 2.90× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.4386 ± 0.109965 (W 0.00594295) | 7.29371 ± 0.240768 (W 0.358934) | 3.90113 ± 0.0419205 (W 0.00550278) | 🔵 faer-fork 5.07× faster than `faer-upstream` | 🔵 faer-fork 2.71× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.4819 ± 0.128794 (W 0.00491795) | 3.55611 ± 0.124086 (W 0.156785) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 27.6213 ± 0.0231588 (W 0.0382088) | 31.2904 ± 0.526938 (W 0.120622) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 29.672 ± 0.0650205 (W 0.0476975) | 32.1129 ± 1.0088 (W 0.285965) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 1.53621 ± 0.0348355 (W 0.0102312) | 1.6857 ± 0.0151174 (W 0.00262731) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.57698 ± 0.0527559 (W 0.035361) | 1.62983 ± 0.0150298 (W 0.00412607) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 2.1871 ± 0.0639524 (W 0.0178442) | 5.13319 ± 0.0646339 (W 0.153787) | 2.90763 ± 0.102409 (W 0.210196) | 🔵 faer-fork 2.35× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.28476 ± 0.051155 (W 0.0277672) | 7.41504 ± 0.116385 (W 0.192282) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.138506 ± 0.028689 (W 0.00225657) | 0.440291 ± 0.0186629 (W 0.0319859) | 0.315805 ± 0.0206705 (W 0.0208533) | 🔵 faer-fork 3.18× faster than `faer-upstream` | 🔵 faer-fork 2.28× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.233181 ± 0.0138615 (W 0.003724) | 0.713604 ± 0.361772 (W 0.0295604) | - | 🔵 faer-fork 3.06× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 2.03761 ± 0.069531 (W 0.0295932) | 22.4637 ± 0.367295 (W 0.471961) | 2.43174 ± 0.0456627 (W 0.0563046) | 🔵 faer-fork 11.02× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 4.11361 ± 0.0498533 (W 0.112986) | 23.191 ± 0.811497 (W 1.36436) | - | 🔵 faer-fork 5.64× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.241231 ± 0.0011694 (W 0.00390155) | 2.0444 ± 0.0308191 (W 0.068638) | 0.490233 ± 0.0627297 (W 0.00668024) | 🔵 faer-fork 8.47× faster than `faer-upstream` | 🔵 faer-fork 2.03× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.381258 ± 0.00324094 (W 0.00741541) | 1.52575 ± 0.666245 (W 0.407253) | - | 🔵 faer-fork 4.00× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 1.77348 ± 0.0920285 (W 0.0297174) | 18.5932 ± 2.54337 (W 1.79579) | 2.17742 ± 0.0756233 (W 0.0595736) | 🔵 faer-fork 10.48× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr_factor/64x1024/b1` | 0.20288 ± 0.00441079 (W 0.00323462) | 1.26026 ± 0.447742 (W 2.93852) | 0.317445 ± 0.00317391 (W 0.00613087) | 🔵 faer-fork 6.21× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/256x4096/b1` | 35.3523 ± 0.257434 (W 0.493484) | 151.24 ± 5.54802 (W 4.45976) | 130.002 ± 1.21909 (W 0.309996) | 🔵 faer-fork 4.28× faster than `faer-upstream` | 🔵 faer-fork 3.68× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.1453 ± 0.107609 (W 0.312335) | 148.366 ± 8.99314 (W 18.7759) | - | 🔵 faer-fork 4.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 2.33192 ± 0.203481 (W 0.00856415) | 11.7797 ± 0.485519 (W 0.385393) | 6.86682 ± 0.580549 (W 0.0957503) | 🔵 faer-fork 5.05× faster than `faer-upstream` | 🔵 faer-fork 2.94× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.54794 ± 0.386607 (W 0.0103061) | 10.5242 ± 2.98439 (W 1.14511) | - | 🔵 faer-fork 4.13× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 27.8422 ± 1.36231 (W 0.283955) | 110.781 ± 1.4294 (W 1.49482) | 108.346 ± 1.89115 (W 5.09417) | 🔵 faer-fork 3.98× faster than `faer-upstream` | 🔵 faer-fork 3.89× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 27.5364 ± 0.90534 (W 0.851632) | 108.334 ± 3.45238 (W 4.28431) | - | 🔵 faer-fork 3.93× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.5236 ± 0.167489 (W 0.00520005) | 9.45435 ± 0.162728 (W 0.218696) | 4.36941 ± 0.203933 (W 0.011335) | 🔵 faer-fork 6.21× faster than `faer-upstream` | 🔵 faer-fork 2.87× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.54432 ± 0.370556 (W 0.00470226) | 7.66828 ± 1.62769 (W 0.491024) | - | 🔵 faer-fork 4.97× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, c order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 21.9963 ± 0.137041 (W 0.0840967) | 24.5988 ± 0.156744 (W 0.0512658) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 97.9514 ± 0.221734 (W 0.0801556) | 106.266 ± 0.26037 (W 0.135106) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.17233 ± 0.0153519 (W 0.00404022) | 2.54723 ± 0.00655313 (W 0.00470672) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.05879 ± 0.0138447 (W 0.00605731) | 3.43156 ± 0.0735904 (W 0.019811) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.18682 ± 0.0480931 (W 0.143366) | 24.863 ± 2.82804 (W 0.193643) | - | 🔵 faer-fork 4.02× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 13.344 ± 0.0105874 (W 0.240386) | 18.2911 ± 2.18783 (W 0.223998) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.283243 ± 0.00545213 (W 0.00270822) | 1.15532 ± 0.010299 (W 0.0205079) | - | 🔵 faer-fork 4.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.533176 ± 0.00039887 (W 0.00243333) | 5.17693 ± 4.36498 (W 1.38387) | - | 🔵 faer-fork 9.71× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 12.7622 ± 0.559493 (W 0.045155) | 26.6747 ± 0.134567 (W 0.363121) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 17.0455 ± 0.0186609 (W 0.0142778) | 31.9559 ± 13.4949 (W 0.515924) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.943015 ± 0.0761588 (W 0.00437477) | 1.77045 ± 0.13193 (W 0.0317785) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.84158 ± 0.00210989 (W 0.00377226) | 6.09436 ± 39.0383 (W 0.125923) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 11.6292 ± 0.112146 (W 0.0779962) | 24.8813 ± 0.199558 (W 0.277925) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.897263 ± 0.0058273 (W 0.00490269) | 1.62448 ± 0.0390639 (W 0.0257142) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 61.5086 ± 0.199662 (W 0.592523) | 110.01 ± 0.542994 (W 0.758571) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 62.2234 ± 0.213013 (W 0.61804) | 110.135 ± 0.0469744 (W 7.76112) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.7105 ± 0.00854201 (W 0.0117492) | 7.67269 ± 1.20088 (W 0.182526) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.74352 ± 0.00628404 (W 0.00867298) | 5.71964 ± 0.584969 (W 0.0416159) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 40.8641 ± 0.119472 (W 0.0777143) | 79.4229 ± 1.03853 (W 0.327718) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 41.0224 ± 0.0204502 (W 0.0743809) | 79.1419 ± 0.470764 (W 0.346037) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.46813 ± 0.00842323 (W 0.00560563) | 5.61831 ± 1.68249 (W 0.227885) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.4884 ± 0.0157706 (W 0.00817677) | 4.56668 ± 0.759747 (W 0.0894785) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, c order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 29.1015 ± 3.22722 (W 0.896918) | 31.1709 ± 4.81224 (W 1.00173) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 90.2321 ± 0.20877 (W 0.0896693) | 99.119 ± 1.73621 (W 0.13515) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.29036 ± 0.00806316 (W 0.00623325) | 2.54437 ± 0.00390969 (W 0.00624728) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.23578 ± 0.0192708 (W 0.00634161) | 3.41457 ± 1.57177 (W 0.0141681) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 4.20314 ± 0.0539756 (W 0.0165311) | 16.7443 ± 1.03644 (W 0.157104) | - | 🔵 faer-fork 3.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 14.0461 ± 0.57585 (W 0.0645545) | 48.0086 ± 22.3802 (W 0.24523) | - | 🔵 faer-fork 3.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.283114 ± 0.00210073 (W 0.00308045) | 0.938315 ± 0.077336 (W 0.020261) | - | 🔵 faer-fork 3.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.595378 ± 0.00185996 (W 0.00285978) | 0.809888 ± 5.70597 (W 0.0129672) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 8.15314 ± 0.543708 (W 0.0403532) | 22.0616 ± 0.0576228 (W 0.244937) | - | 🔵 faer-fork 2.71× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 13.4618 ± 0.0720436 (W 0.0330869) | 36.4272 ± 14.7606 (W 1.61369) | - | 🔵 faer-fork 2.71× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.865071 ± 0.000951963 (W 0.0043728) | 2.95426 ± 0.418492 (W 0.178455) | - | 🔵 faer-fork 3.42× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.812799 ± 0.00575376 (W 0.00504173) | 2.48831 ± 2.27479 (W 0.174968) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 7.28374 ± 0.082112 (W 0.0231779) | 20.264 ± 0.151685 (W 0.253544) | - | 🔵 faer-fork 2.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.831819 ± 0.0165362 (W 0.00729444) | 1.79197 ± 0.290663 (W 0.119358) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 59.6162 ± 0.570391 (W 1.0474) | 108.89 ± 0.283062 (W 0.266335) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 59.3249 ± 0.340339 (W 0.0798052) | 109.163 ± 0.49228 (W 7.77673) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.72821 ± 0.0088729 (W 0.0128446) | 11.6413 ± 2.71223 (W 0.829405) | - | 🔵 faer-fork 3.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.78873 ± 0.0102267 (W 0.00820609) | 6.56525 ± 2.34665 (W 0.50154) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 41.9168 ± 0.104413 (W 0.17881) | 83.5622 ± 0.768136 (W 0.397298) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 42.0306 ± 0.68561 (W 0.171694) | 82.9493 ± 0.0685087 (W 0.605092) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.48657 ± 0.0889572 (W 0.00358079) | 10.7442 ± 0.295355 (W 0.556405) | - | 🔵 faer-fork 4.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.54798 ± 0.194891 (W 0.00971807) | 8.20934 ± 1.22565 (W 1.02033) | - | 🔵 faer-fork 3.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, c order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 36.9612 ± 0.153275 (W 3.60962) | 39.2196 ± 4.3421 (W 3.61132) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 88.4053 ± 0.600646 (W 0.113962) | 109.891 ± 6.90993 (W 0.399364) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 1.4879 ± 0.00572499 (W 0.00690008) | 2.55301 ± 0.0133102 (W 0.00558625) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.44587 ± 0.0185693 (W 0.00650925) | 3.59795 ± 1.01795 (W 0.01098) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 3.22881 ± 0.0262663 (W 0.0137376) | 13.5646 ± 0.541184 (W 0.133091) | - | 🔵 faer-fork 4.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 14.2663 ± 0.234791 (W 0.0879713) | 17.2961 ± 0.47315 (W 8.92241) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.321645 ± 0.00450533 (W 0.00657392) | 0.965776 ± 0.0635406 (W 0.0697773) | - | 🔵 faer-fork 3.00× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.609739 ± 0.0138374 (W 0.00361605) | 0.780778 ± 0.0613212 (W 0.0184418) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 8.18862 ± 0.57752 (W 0.0203301) | 21.104 ± 0.196488 (W 0.470981) | - | 🔵 faer-fork 2.58× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 13.1781 ± 0.0895397 (W 0.11893) | 41.058 ± 35.8825 (W 5.38585) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.857471 ± 0.0226944 (W 0.00355222) | 3.56694 ± 0.111495 (W 0.116661) | - | 🔵 faer-fork 4.16× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.828849 ± 0.0168966 (W 0.00347555) | 6.52397 ± 2.6715 (W 0.11632) | - | 🔵 faer-fork 7.87× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 6.77319 ± 0.126365 (W 0.0261226) | 19.1136 ± 0.0732565 (W 0.240283) | - | 🔵 faer-fork 2.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.797888 ± 0.0020534 (W 0.00246102) | 2.24146 ± 0.361292 (W 0.184223) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 62.5915 ± 0.119455 (W 0.145353) | 108.64 ± 1.08139 (W 0.911565) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 61.9602 ± 0.125203 (W 0.204416) | 110.656 ± 4.45431 (W 7.60335) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 3.88938 ± 0.0379954 (W 0.0097945) | 20.8939 ± 2.8324 (W 0.58541) | - | 🔵 faer-fork 5.37× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.97693 ± 0.120458 (W 0.00968282) | 8.01105 ± 2.17506 (W 0.587406) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 45.4098 ± 0.991676 (W 0.899622) | 111.893 ± 6.55746 (W 3.36067) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 45.6083 ± 0.277471 (W 0.594405) | 116.266 ± 1.20244 (W 3.86922) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.66962 ± 0.146331 (W 0.00835456) | 13.5714 ± 1.85843 (W 0.523678) | - | 🔵 faer-fork 5.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.64322 ± 0.572008 (W 0.00719432) | 5.4895 ± 0.906557 (W 0.451332) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, f order, 16T (CPU 4-19)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 91.2742 ± 0.190185 (W 0.211245) | 99.2708 ± 0.479975 (W 0.145065) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 94.1569 ± 0.0675954 (W 0.128351) | 100.773 ± 0.378 (W 0.0556335) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.77865 ± 0.0413265 (W 0.00614442) | 3.15234 ± 0.0941883 (W 0.00305223) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.92251 ± 0.0317674 (W 0.0210167) | 3.01361 ± 0.00751696 (W 0.00502714) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 5.48987 ± 0.0366622 (W 0.0434273) | 7.62585 ± 0.0847815 (W 0.230151) | 11.1047 ± 0.0144536 (W 0.0213528) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔵 faer-fork 2.02× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 9.4487 ± 0.0542184 (W 0.201023) | 10.7252 ± 0.237653 (W 0.325871) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.20833 ± 0.00310986 (W 0.00301311) | 0.306014 ± 0.000563685 (W 0.00448882) | 0.291824 ± 0.0017543 (W 0.00331312) | 🔵 faer-fork 1.47× faster than `faer-upstream` | 🔵 faer-fork 1.40× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.334136 ± 0.0046943 (W 0.00239174) | 0.449522 ± 2.45787 (W 0.00633134) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 8.86349 ± 0.0267469 (W 0.0248381) | 20.688 ± 0.331447 (W 0.249501) | 13.1753 ± 0.026686 (W 0.0494428) | 🔵 faer-fork 2.33× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 12.6197 ± 0.0106845 (W 0.018938) | 24.959 ± 2.66515 (W 3.6224) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.476323 ± 0.00291718 (W 0.00305974) | 1.60761 ± 0.107687 (W 0.0446688) | 0.662342 ± 0.00268413 (W 0.00441434) | 🔵 faer-fork 3.38× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.650991 ± 0.00220475 (W 0.0017797) | 1.63976 ± 0.0474308 (W 0.0585336) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 8.11816 ± 0.00957254 (W 0.0517726) | 18.5456 ± 0.242946 (W 0.288372) | 12.0903 ± 0.0110003 (W 0.0369533) | 🔵 faer-fork 2.28× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.42837 ± 0.00126651 (W 0.00320239) | 1.18374 ± 0.0370225 (W 0.0462537) | 0.578897 ± 0.00119845 (W 0.00313286) | 🔵 faer-fork 2.76× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 73.1602 ± 0.595062 (W 0.92302) | 120.763 ± 0.536029 (W 0.51052) | 138.856 ± 1.57567 (W 0.105565) | 🔵 faer-fork 1.65× faster than `faer-upstream` | 🔵 faer-fork 1.90× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 72.6839 ± 0.375256 (W 1.02329) | 120.867 ± 0.641478 (W 0.413326) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.72139 ± 0.00106501 (W 0.00923289) | 7.18539 ± 1.30047 (W 0.338977) | 8.71035 ± 0.041812 (W 0.0153011) | 🔵 faer-fork 1.93× faster than `faer-upstream` | 🔵 faer-fork 2.34× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.76029 ± 0.00624348 (W 0.0127073) | 6.19328 ± 0.321221 (W 0.170926) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 50.0128 ± 0.113215 (W 0.0658431) | 88.3392 ± 0.177927 (W 0.290428) | 72.7623 ± 0.270584 (W 0.700124) | 🔵 faer-fork 1.77× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 49.7438 ± 0.0339322 (W 0.0454354) | 88.5821 ± 0.10831 (W 4.75371) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.52177 ± 0.0254588 (W 0.00602924) | 6.39554 ± 1.1956 (W 0.232951) | 4.54249 ± 0.0216805 (W 0.0430077) | 🔵 faer-fork 2.54× faster than `faer-upstream` | 🔵 faer-fork 1.80× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.53555 ± 0.0150125 (W 0.00751343) | 3.96691 ± 0.426432 (W 0.0337625) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, f order, 32T (CPU 4-35)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 82.5209 ± 0.0887384 (W 0.220406) | 91.1491 ± 0.718038 (W 0.123017) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 86.4276 ± 0.0271229 (W 0.0904768) | 93.203 ± 0.449059 (W 0.0925445) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.86719 ± 0.0139454 (W 0.00740595) | 3.15 ± 0.0998337 (W 0.00343454) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.05711 ± 0.0272548 (W 0.00590748) | 3.12693 ± 0.0247259 (W 0.0114028) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.94172 ± 0.0209677 (W 0.0370105) | 5.95142 ± 0.0422048 (W 0.193844) | 5.21633 ± 0.0156952 (W 0.0102762) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 8.21765 ± 0.0945088 (W 0.0752355) | 10.5204 ± 16.1776 (W 12.2224) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.22146 ± 0.00422836 (W 0.00212857) | 0.410359 ± 0.0134808 (W 0.0156166) | 0.366067 ± 0.0464914 (W 0.00598017) | 🔵 faer-fork 1.85× faster than `faer-upstream` | 🔵 faer-fork 1.65× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.464782 ± 0.01396 (W 0.00291904) | 0.459102 ± 2.47761 (W 0.0144055) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 4.97285 ± 0.00251425 (W 0.00882254) | 17.4104 ± 0.0825142 (W 0.280023) | 7.37839 ± 0.0123591 (W 0.0136471) | 🔵 faer-fork 3.50× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 8.93516 ± 0.0299642 (W 0.0144325) | 21.0698 ± 0.162017 (W 2.10627) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.403959 ± 0.00188612 (W 0.00302245) | 2.51408 ± 0.454876 (W 0.090688) | 0.664122 ± 0.010084 (W 0.0071224) | 🔵 faer-fork 6.22× faster than `faer-upstream` | 🔵 faer-fork 1.64× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.649131 ± 0.0124288 (W 0.00546197) | 1.77376 ± 0.515222 (W 0.0700998) | - | 🔵 faer-fork 2.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 4.47937 ± 0.00524968 (W 0.00955196) | 15.7269 ± 0.116558 (W 0.168312) | 6.41314 ± 0.0121359 (W 0.0182004) | 🔵 faer-fork 3.51× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.362387 ± 0.00414812 (W 0.00376257) | 1.67037 ± 36.8502 (W 13.5549) | 0.524545 ± 0.00533491 (W 0.00560927) | · | 🔵 faer-fork 1.45× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/256x4096/b1` | 68.559 ± 0.935348 (W 0.159592) | 116.285 ± 1.04012 (W 0.330298) | 131.272 ± 2.17018 (W 0.0914235) | 🔵 faer-fork 1.70× faster than `faer-upstream` | 🔵 faer-fork 1.91× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 69.5536 ± 0.391703 (W 0.0892298) | 117.926 ± 0.392776 (W 9.37787) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.74461 ± 0.011922 (W 0.00984531) | 11.8069 ± 2.01022 (W 0.62446) | 11.2445 ± 0.0251934 (W 0.0322512) | 🔵 faer-fork 3.15× faster than `faer-upstream` | 🔵 faer-fork 3.00× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.80596 ± 0.018744 (W 0.00849175) | 6.66328 ± 1.93943 (W 0.588102) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 50.1645 ± 0.260121 (W 0.127294) | 90.3554 ± 0.115389 (W 0.273456) | 72.1151 ± 0.29659 (W 1.02647) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 50.4539 ± 0.160456 (W 0.134865) | 90.6266 ± 0.470272 (W 4.62626) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.60694 ± 0.0474709 (W 0.0100652) | 11.0064 ± 0.160666 (W 0.439399) | 6.33169 ± 0.0059692 (W 0.0238406) | 🔵 faer-fork 4.22× faster than `faer-upstream` | 🔵 faer-fork 2.43× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.61577 ± 0.0729528 (W 0.0068403) | 9.75664 ± 3.46557 (W 0.591209) | - | 🔵 faer-fork 3.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, f order, 60T (CPU 4-63)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 80.6053 ± 0.248271 (W 0.256583) | 90.6093 ± 0.30165 (W 0.203223) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 84.8093 ± 0.0206532 (W 0.0814356) | 93.6191 ± 0.293645 (W 0.255636) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 2.08934 ± 0.0272539 (W 0.00368679) | 3.14811 ± 0.0726207 (W 0.00772443) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.29458 ± 0.0051607 (W 0.010555) | 3.11834 ± 0.0898307 (W 0.00826337) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 3.17916 ± 0.0213304 (W 0.0425474) | 5.65219 ± 0.179451 (W 0.118363) | 3.67949 ± 0.0507682 (W 0.0127308) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 7.35518 ± 0.0276953 (W 0.376016) | 11.1345 ± 16.8876 (W 0.50461) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.2208 ± 0.0195556 (W 0.00253508) | 0.454701 ± 0.0104523 (W 0.0188775) | 0.363818 ± 0.0212841 (W 0.0102869) | 🔵 faer-fork 2.06× faster than `faer-upstream` | 🔵 faer-fork 1.65× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.521065 ± 0.063177 (W 0.004148) | 0.473783 ± 0.0556751 (W 0.0225092) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 3.38497 ± 0.0175829 (W 0.0129592) | 17.4507 ± 0.138883 (W 0.186316) | 5.95228 ± 0.0525159 (W 0.0410798) | 🔵 faer-fork 5.16× faster than `faer-upstream` | 🔵 faer-fork 1.76× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 8.02612 ± 0.0599506 (W 0.189739) | 21.8664 ± 0.129439 (W 3.91854) | - | 🔵 faer-fork 2.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.403759 ± 0.00808987 (W 0.00324824) | 2.00278 ± 0.732971 (W 0.165023) | 0.829239 ± 0.0495722 (W 0.00997333) | 🔵 faer-fork 4.96× faster than `faer-upstream` | 🔵 faer-fork 2.05× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.661961 ± 0.00834654 (W 0.00534179) | 2.21732 ± 0.41134 (W 0.29515) | - | 🔵 faer-fork 3.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 2.93756 ± 0.0306352 (W 0.00850383) | 15.6205 ± 0.0689443 (W 0.0840272) | 5.15683 ± 0.0306598 (W 0.0490079) | 🔵 faer-fork 5.32× faster than `faer-upstream` | 🔵 faer-fork 1.76× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.364827 ± 0.00419085 (W 0.00277679) | 2.24088 ± 0.250345 (W 0.306708) | 0.652141 ± 0.0308866 (W 0.0192377) | 🔵 faer-fork 6.14× faster than `faer-upstream` | 🔵 faer-fork 1.79× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/256x4096/b1` | 70.7286 ± 2.57354 (W 0.206053) | 117.566 ± 1.10522 (W 0.760446) | 134.84 ± 2.32924 (W 0.191132) | 🔵 faer-fork 1.66× faster than `faer-upstream` | 🔵 faer-fork 1.91× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 72.51 ± 0.274052 (W 0.266495) | 121.317 ± 4.32254 (W 10.9461) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 3.94374 ± 0.114672 (W 0.00961018) | 17.467 ± 0.232545 (W 0.597536) | 9.20242 ± 0.0441249 (W 0.0217279) | 🔵 faer-fork 4.43× faster than `faer-upstream` | 🔵 faer-fork 2.33× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.92116 ± 0.291243 (W 0.00860501) | 7.70098 ± 1.52094 (W 0.54919) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 53.4593 ± 0.306001 (W 0.306442) | 117.947 ± 0.760249 (W 3.39263) | 76.0328 ± 2.42731 (W 0.5337) | 🔵 faer-fork 2.21× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 54.0548 ± 0.204925 (W 0.85334) | 117.575 ± 1.04694 (W 5.058) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.82816 ± 0.147365 (W 0.012231) | 11.2336 ± 0.676058 (W 0.534928) | 4.53227 ± 0.0641716 (W 0.0104958) | 🔵 faer-fork 3.97× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.71448 ± 0.49392 (W 0.0101478) | 7.43694 ± 0.394333 (W 0.852525) | - | 🔵 faer-fork 2.74× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `lu/256x4096`, `lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `solve/256x4096`, `solve/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
