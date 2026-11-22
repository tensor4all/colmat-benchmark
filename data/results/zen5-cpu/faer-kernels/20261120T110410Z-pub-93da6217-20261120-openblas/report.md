# `faer-kernels` on `zen5-cpu`

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
- timestamp: `2026-11-20T15:30:56.664050Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu faer-kernels --vendor openblas --engines faer-fork,lapack-openblas --threads 1,4,8 --reps 5 --aa 6 --prime-ms 250 --label pub-93da6217-20261120-openblas`
- raw data: `data/results/zen5-cpu/faer-kernels/20261120T110410Z-pub-93da6217-20261120-openblas/`

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
- CPU sets: 1T -> `4`, 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.26%, worst 30131.90% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 264 of 5064 engine cells exceed 5% and carry `!`; median within-process relative SD 0.39%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
| `gemm/32x32x32/b1` | 0.0023645 ± 1.74059e-05 (W 2.05309e-05) | 0.002224 ± 3.39116e-05 (W 2.97946e-05) | 🔴 faer-fork 1.06× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0171215 ± 7.14714e-05 (W 6.10476e-05) | 0.016836 ± 8.97344e-05 (W 3.8673e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0567355 ± 0.000223031 (W 5.97567e-05) | 0.056295 ± 0.000266953 (W 0.000550292) | · |  |

## gemm-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002429 ± 4.46348e-05 (W 2.91191e-05) | 0.002204 ± 1.50555e-05 (W 3.09928e-05) | 🔴 faer-fork 1.10× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.011116 ± 0.000108832 (W 0.000182759) | 0.0168015 ± 6.18245e-05 (W 4.507e-05) | 🔵 faer-fork 1.51× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0226875 ± 0.000336119 (W 0.000363555) | 0.05626 ± 0.00015452 (W 4.26169e-05) | 🔵 faer-fork 2.48× faster than `lapack-openblas` |  |

## gemm-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002404 ± 8.61781e-06 (W 2.49347e-05) | 0.002219 ± 1.94079e-05 (W 3.06324e-05) | 🔴 faer-fork 1.08× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.009548 ± 4.13461e-05 (W 0.000101787) | 0.0168365 ± 0.000107553 (W 4.28635e-05) | 🔵 faer-fork 1.76× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020047 ± 9.34247e-05 (W 9.51795e-05) | 0.0562995 ± 0.000251259 (W 0.00153256) | 🔵 faer-fork 2.81× faster than `lapack-openblas` |  |

## gemm-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0023545 ± 2.44459e-05 (W 1.99878e-05) | 0.002224 ± 1.60052e-05 (W 2.91718e-05) | 🔴 faer-fork 1.06× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.017097 ± 6.36566e-05 (W 4.85358e-05) | 0.016797 ± 6.50661e-05 (W 3.59598e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.05674 ± 0.000237136 (W 9.72977e-05) | 0.0562095 ± 0.000250081 (W 0.000115486) | · |  |

## gemm-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0023995 ± 3.79772e-05 (W 2.24618e-05) | 0.002224 ± 7.52773e-06 (W 2.69146e-05) | 🔴 faer-fork 1.08× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0112255 ± 0.00016582 (W 0.00018444) | 0.016761 ± 9.22036e-05 (W 0.000707023) | 🔵 faer-fork 1.49× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0230825 ± 0.000291137 (W 0.000362868) | 0.0563145 ± 0.00017299 (W 4.26113e-05) | 🔵 faer-fork 2.44× faster than `lapack-openblas` |  |

## gemm-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002389 ± 1.63758e-05 (W 3.22904e-05) | 0.0022045 ± 1.02713e-05 (W 2.44872e-05) | 🔴 faer-fork 1.08× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.009638 ± 8.15363e-05 (W 9.09623e-05) | 0.0168365 ± 6.12876e-05 (W 3.51958e-05) | 🔵 faer-fork 1.75× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.019952 ± 8.25197e-05 (W 9.24766e-05) | 0.0562795 ± 0.000239137 (W 0.00118734) | 🔵 faer-fork 2.82× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.008817 ± 2.52639e-05 (W 3.92021e-05) | 0.011571 ± 3.82657e-05 (W 3.20256e-05) | 🔵 faer-fork 1.31× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0677165 ± 0.000102381 (W 6.98134e-05) | 0.0861345 ± 0.000329428 (W 0.000912606) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.226656 ± 0.00088157 (W 0.00146007) | 0.275937 ± 0.00112326 (W 0.00117828) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009513 ± 0.000174738 (W 0.000152607) | 0.011501 ± 4.79892e-05 (W 3.30712e-05) | 🔵 faer-fork 1.21× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.028578 ± 0.000220878 (W 0.000260479) | 0.0261535 ± 0.000121772 (W 0.000141494) | 🔴 faer-fork 1.09× slower than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.0779 ± 0.000293007 (W 0.000214811) | 0.0790465 ± 0.000205298 (W 0.000523311) | · |  |

## gemm-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009663 ± 7.17403e-05 (W 0.000105446) | 0.011567 ± 5.0875e-05 (W 3.65913e-05) | 🔵 faer-fork 1.20× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.026624 ± 0.000129528 (W 8.84591e-05) | 0.026153 ± 0.000145011 (W 0.000310206) | · |  |
| `gemm/96x96x96/b1` | 0.0396435 ± 0.000131586 (W 0.000105002) | 0.055373 ± 0.000339835 (W 0.000375668) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |

## gemm-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0088265 ± 2.39082e-05 (W 3.70112e-05) | 0.0115715 ± 6.84563e-05 (W 4.25498e-05) | 🔵 faer-fork 1.31× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0677865 ± 0.000216094 (W 0.00119148) | 0.0859445 ± 0.000310335 (W 0.000143371) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.22645 ± 0.00103934 (W 0.00128616) | 0.276423 ± 0.00081862 (W 0.00123646) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0094575 ± 0.000172012 (W 9.77068e-05) | 0.0115215 ± 5.2627e-05 (W 2.97206e-05) | 🔵 faer-fork 1.22× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0286935 ± 0.000140709 (W 0.000238972) | 0.026019 ± 0.00013087 (W 0.00015152) | 🔴 faer-fork 1.10× slower than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.0779045 ± 0.000415691 (W 0.000265279) | 0.0790565 ± 0.000510972 (W 0.000575817) | · |  |

## gemm-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009693 ± 0.000121943 (W 8.94918e-05) | 0.011607 ± 3.94495e-05 (W 3.36717e-05) | 🔵 faer-fork 1.20× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.026644 ± 0.000126792 (W 8.19548e-05) | 0.026073 ± 0.000110822 (W 0.000283632) | · |  |
| `gemm/96x96x96/b1` | 0.039679 ± 0.000111022 (W 0.00012955) | 0.055403 ± 0.000334615 (W 0.000355005) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |

## gemm-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001282 ± 1.34226e-05 (W 2.95472e-05) | 0.0011625 ± 1.37174e-05 (W 2.78352e-05) | 🔴 faer-fork 1.10× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.008606 ± 2.89856e-05 (W 3.20384e-05) | 0.0084455 ± 2.791e-05 (W 3.34518e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.028428 ± 0.00012466 (W 6.35573e-05) | 0.0281475 ± 0.000143895 (W 5.25857e-05) | · |  |

## gemm-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0013475 ± 2.63869e-05 (W 2.95446e-05) | 0.001147 ± 1.47196e-05 (W 3.21686e-05) | 🔴 faer-fork 1.17× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.006056 ± 0.00021786 (W 0.000329043) | 0.008451 ± 2.34414e-05 (W 3.59066e-05) | 🔵 faer-fork 1.40× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.012984 ± 0.000295609 (W 0.000298796) | 0.0281875 ± 8.81901e-05 (W 3.78845e-05) | 🔵 faer-fork 2.17× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0013325 ± 1.87341e-05 (W 2.58753e-05) | 0.001162 ± 1.50555e-05 (W 3.01241e-05) | 🔴 faer-fork 1.15× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00535 ± 7.23706e-05 (W 7.24254e-05) | 0.0084405 ± 3.72957e-05 (W 3.41114e-05) | 🔵 faer-fork 1.58× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0106945 ± 8.16915e-05 (W 7.4692e-05) | 0.0281725 ± 9.62116e-05 (W 4.36858e-05) | 🔵 faer-fork 2.63× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0012575 ± 1.42466e-05 (W 2.92043e-05) | 0.001167 ± 1.97484e-05 (W 2.92394e-05) | 🔴 faer-fork 1.08× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.008606 ± 3.85768e-05 (W 3.2834e-05) | 0.0084455 ± 5.74247e-05 (W 2.75444e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.028443 ± 0.000111679 (W 4.48715e-05) | 0.0281725 ± 0.000103287 (W 4.87466e-05) | · |  |

## gemm-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0013425 ± 2.51714e-05 (W 2.84225e-05) | 0.001152 ± 1.32916e-05 (W 2.60086e-05) | 🔴 faer-fork 1.17× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0061965 ± 6.68952e-05 (W 0.00014495) | 0.0084555 ± 3.34091e-05 (W 3.49047e-05) | 🔵 faer-fork 1.36× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.013089 ± 0.000174723 (W 0.000161157) | 0.0281975 ± 8.98458e-05 (W 3.68232e-05) | 🔵 faer-fork 2.15× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001327 ± 1.25007e-05 (W 3.2282e-05) | 0.001167 ± 1.54919e-05 (W 2.90689e-05) | 🔴 faer-fork 1.14× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00532 ± 6.27429e-05 (W 7.66717e-05) | 0.008441 ± 4.28194e-05 (W 3.43055e-05) | 🔵 faer-fork 1.59× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0107045 ± 2.65255e-05 (W 0.000122488) | 0.028097 ± 0.000140345 (W 0.0014568) | 🔵 faer-fork 2.62× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0044935 ± 5.50151e-06 (W 3.30026e-05) | 0.005505 ± 2.28947e-05 (W 2.33424e-05) | 🔵 faer-fork 1.23× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.033943 ± 4.09276e-05 (W 3.82037e-05) | 0.0385815 ± 0.00019258 (W 0.000640485) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.113024 ± 0.00042348 (W 0.000218702) | 0.123139 ± 0.000399418 (W 0.00121484) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005114 ± 4.51398e-05 (W 0.000232712) | 0.005475 ± 3.50238e-05 (W 2.20365e-05) | 🔵 faer-fork 1.07× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0199515 ± 0.000128924 (W 0.000162633) | 0.012343 ± 5.03653e-05 (W 0.000159394) | 🔴 faer-fork 1.62× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.041587 ± 0.000335452 (W 0.000118105) | 0.035666 ± 0.000142232 (W 0.000317718) | 🔴 faer-fork 1.17× slower than `lapack-openblas` |  |

## gemm-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005164 ± 3.74028e-05 (W 6.65049e-05) | 0.00552 ± 2.51051e-05 (W 2.02638e-05) | 🔵 faer-fork 1.07× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.018003 ± 8.88154e-05 (W 9.37426e-05) | 0.012348 ± 0.000234918 (W 0.00022865) | 🔴 faer-fork 1.46× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0390125 ± 0.00024964 (W 0.000104037) | 0.0258385 ± 0.000174569 (W 0.000216043) | 🔴 faer-fork 1.51× slower than `lapack-openblas` |  |

## gemm-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.004513 ± 2.38642e-05 (W 3.53432e-05) | 0.0055 ± 1.42887e-05 (W 2.40923e-05) | 🔵 faer-fork 1.22× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0339335 ± 0.000133043 (W 3.54885e-05) | 0.038391 ± 0.000244609 (W 7.33025e-05) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.113 ± 0.000459802 (W 0.00122206) | 0.122873 ± 0.000370451 (W 0.000207136) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005325 ± 0.000177077 (W 0.00018547) | 0.00548 ± 1.73253e-05 (W 1.70318e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.019927 ± 0.000114578 (W 8.39588e-05) | 0.012378 ± 0.00018936 (W 0.000269795) | 🔴 faer-fork 1.61× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0416475 ± 0.000379581 (W 0.000228543) | 0.0357365 ± 0.000156883 (W 0.000338325) | 🔴 faer-fork 1.17× slower than `lapack-openblas` |  |

## gemm-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005179 ± 9.46481e-05 (W 7.81449e-05) | 0.00551 ± 2.44186e-05 (W 2.97473e-05) | 🔵 faer-fork 1.06× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0179835 ± 3.92972e-05 (W 0.000136565) | 0.0122225 ± 0.000128516 (W 0.000211427) | 🔴 faer-fork 1.47× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0388875 ± 0.00012735 (W 0.00014213) | 0.025803 ± 0.000208608 (W 0.000229865) | 🔴 faer-fork 1.51× slower than `lapack-openblas` |  |

## gemm-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.132837 ± 0.000489363 (W 0.000496474) | 0.162747 ± 0.000621778 (W 0.00148807) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.451683 ± 0.000536189 (W 0.002179) | 0.535621 ± 0.00298024 (W 0.00235323) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.06683 ± 0.00387099 (W 0.00216488) | 1.26265 ± 0.00271488 (W 0.00555998) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 3.5986 ± 0.0102277 (W 0.00725014) | 4.20502 ± 0.0118723 (W 0.00786688) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 8.55888 ± 0.0290462 (W 0.0148004) | 10.0102 ± 0.0250382 (W 0.0206897) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 29.2502 ± 0.150901 (W 0.0373511) | 33.5477 ± 0.101519 (W 0.0444051) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |

## gemm-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0454445 ± 0.000296629 (W 0.000241925) | 0.0506745 ± 2.42791 (W 0.00104127) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.120008 ± 0.000594643 (W 0.00149097) | 0.148991 ± 0.00110514 (W 0.00140417) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.289461 ± 0.00134588 (W 0.00133343) | 0.342539 ± 0.00133203 (W 0.00155125) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.913431 ± 0.00284162 (W 0.00184093) | 1.09731 ± 0.00568534 (W 0.00529517) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.2039 ± 0.00739412 (W 0.00364504) | 2.57153 ± 0.0091237 (W 0.00392271) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 7.48092 ± 0.0138366 (W 0.010342) | 8.69359 ± 0.0205847 (W 0.0169652) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0321565 ± 0.000609906 (W 0.00101929) | 0.0297575 ± 3.66044 (W 0.00049158) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.086186 ± 0.000286372 (W 0.000935874) | 0.084403 ± 0.000809053 (W 0.000955393) | · |  |
| `gemm/256x256x256/b1` | 0.163725 ± 0.000954318 (W 0.000952131) | 0.182886 ± 0.00362678 (W 0.00150116) | 🔵 faer-fork 1.12× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.475931 ± 0.00219709 (W 0.00220349) | 0.55529 ± 0.00558718 (W 0.00298252) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.13333 ± 0.00295778 (W 0.00323336) | 1.31212 ± 0.00529042 (W 0.00459178) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 3.89253 ± 0.0128088 (W 0.0152672) | 4.52267 ± 0.00885334 (W 0.0115789) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.133624 ± 0.000462687 (W 0.000660659) | 0.162392 ± 0.000578287 (W 0.00102475) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.449175 ± 0.00131998 (W 0.00151321) | 0.535739 ± 0.00206617 (W 0.00211542) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.0659 ± 0.00474494 (W 0.00181914) | 1.26109 ± 0.00418352 (W 0.00422745) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 3.58773 ± 0.0116692 (W 0.00529807) | 4.19432 ± 0.0184129 (W 0.00965407) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 8.54474 ± 0.032252 (W 0.0112145) | 10.005 ± 0.0354765 (W 0.0207993) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 29.2726 ± 0.114369 (W 0.0502185) | 33.597 ± 0.141336 (W 0.0333255) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0452885 ± 8.29859e-05 (W 0.000385641) | 0.050449 ± 0.000398516 (W 0.00119973) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.120458 ± 0.00107254 (W 0.00151306) | 0.148772 ± 0.00105288 (W 0.00172596) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.290073 ± 0.00179441 (W 0.00122851) | 0.342573 ± 0.000549724 (W 0.00213573) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.912849 ± 0.00394479 (W 0.00379938) | 1.10065 ± 0.00721501 (W 0.00490682) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.20336 ± 0.00667814 (W 0.00460757) | 2.57476 ± 0.00749123 (W 0.00459489) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 7.50072 ± 0.0182984 (W 0.0239629) | 8.70918 ± 0.0236169 (W 0.0126281) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.032182 ± 0.000534359 (W 0.00119969) | 0.030033 ± 2.43563 (W 0.000571222) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.0857645 ± 0.000265796 (W 0.000818681) | 0.0840965 ± 0.000607791 (W 0.000559755) | · |  |
| `gemm/256x256x256/b1` | 0.164452 ± 0.00116791 (W 0.00144973) | 0.182189 ± 4.81929 (W 0.00160855) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.476872 ± 0.0021639 (W 0.0025198) | 0.554172 ± 0.00216434 (W 0.00296528) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.13039 ± 0.00359338 (W 0.0024582) | 1.31235 ± 0.0240958 (W 0.00531846) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 3.87741 ± 0.0183867 (W 0.0105733) | 4.53451 ± 0.0180763 (W 0.0252469) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.536032 ± 0.00214277 (W 0.00265064) | 0.653714 ± 0.00305437 (W 0.00204128) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 1.80311 ± 0.00550741 (W 0.00381994) | 2.16972 ± 0.00768436 (W 0.00464074) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 4.26349 ± 0.0127659 (W 0.00729407) | 5.14562 ± 0.0150761 (W 0.00830843) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 14.3504 ± 0.0642387 (W 0.0142879) | 17.229 ± 0.0617751 (W 0.0165017) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 34.1797 ± 0.0771379 (W 0.049365) | 40.7299 ± 0.0874697 (W 0.0388118) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 115.849 ± 0.334209 (W 0.071985) | 136.884 ± 0.325422 (W 0.409563) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.15333 ± 0.000706628 (W 0.000506293) | 0.177295 ± 0.00082256 (W 0.00166754) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.456693 ± 0.000870016 (W 0.00245812) | 0.558974 ± 0.000909474 (W 0.00164248) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.10895 ± 0.000753261 (W 0.00148278) | 1.32778 ± 0.0058032 (W 0.00385208) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 3.61499 ± 0.00328408 (W 0.00386193) | 4.38195 ± 0.0104094 (W 0.00715415) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 8.63028 ± 0.00725055 (W 0.0143879) | 10.4019 ± 0.0133837 (W 0.0183567) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 29.3092 ± 0.0421944 (W 0.0230473) | 34.7657 ± 0.0475096 (W 0.0371699) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.102505 ± 9.7342e-05 (W 0.000509226) | 0.0942755 ± 0.000386992 (W 0.000694519) | 🔴 faer-fork 1.09× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.23409 ± 0.000946283 (W 0.00139811) | 0.294304 ± 0.000811281 (W 0.00168954) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.578358 ± 0.00184597 (W 0.00174876) | 0.681298 ± 0.00225835 (W 0.00176487) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.82217 ± 0.00568525 (W 0.00419917) | 2.19104 ± 0.00705811 (W 0.00425466) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 4.42373 ± 0.021197 (W 0.00909931) | 5.21708 ± 0.0155952 (W 0.00692115) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 14.9011 ± 0.0529891 (W 0.020564) | 17.6187 ± 0.0429 (W 0.0733593) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.536846 ± 0.00138076 (W 0.00165596) | 0.653791 ± 0.00320447 (W 0.00211625) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 1.79985 ± 0.00595754 (W 0.00438102) | 2.15781 ± 0.0117616 (W 0.00255907) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 4.25533 ± 0.0197235 (W 0.0084398) | 5.13802 ± 0.0192943 (W 0.00575939) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 14.348 ± 0.0499099 (W 0.0179099) | 17.2355 ± 0.045178 (W 0.0252733) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 34.1438 ± 0.0872343 (W 0.038704) | 40.7142 ± 0.0988029 (W 0.0329366) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 115.86 ± 0.338631 (W 0.0777563) | 136.762 ± 0.633654 (W 0.174652) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.153179 ± 0.000634399 (W 0.000629676) | 0.178111 ± 0.000482451 (W 0.00127674) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.457604 ± 0.00119374 (W 0.00208634) | 0.559194 ± 0.00210229 (W 0.00202372) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.10871 ± 0.00316562 (W 0.0010376) | 1.32851 ± 0.00391483 (W 0.00291525) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 3.61975 ± 0.00899188 (W 0.00642554) | 4.37912 ± 0.014899 (W 0.0105908) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 8.62967 ± 0.00242609 (W 0.0111574) | 10.4116 ± 0.00997912 (W 0.0148941) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 29.298 ± 0.0557905 (W 0.0212562) | 34.7461 ± 0.0616036 (W 0.032627) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.102656 ± 0.000400184 (W 0.000534065) | 0.094206 ± 0.000422991 (W 0.00108656) | 🔴 faer-fork 1.09× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.233521 ± 0.00106224 (W 0.00147751) | 0.294607 ± 0.00127773 (W 0.00161936) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.578755 ± 0.00171781 (W 0.00182493) | 0.679777 ± 0.00310884 (W 0.00216935) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.81879 ± 0.00711842 (W 0.00553739) | 2.18712 ± 0.00704674 (W 0.00430511) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 4.42685 ± 0.014909 (W 0.00588599) | 5.22395 ± 0.0149003 (W 0.0109316) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 14.9028 ± 0.0583118 (W 0.012065) | 17.6075 ± 0.0397707 (W 0.0484734) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |

## gemm-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.066819 ± 0.000262763 (W 0.000146137) | 0.072405 ± 0.000439463 (W 0.000331312) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.223659 ± 0.000790669 (W 0.00117393) | 0.235237 ± 0.001211 (W 0.0011912) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.532967 ± 0.00159205 (W 0.0023693) | 0.558421 ± 0.0020737 (W 0.00224328) | · |  |
| `gemm/384x384x384/b1` | 1.79536 ± 0.00399847 (W 0.00444429) | 1.85899 ± 0.00869571 (W 0.0063427) | · |  |
| `gemm/512x512x512/b1` | 4.24826 ± 0.0179979 (W 0.00530535) | 4.38103 ± 0.0276072 (W 0.00698899) | · |  |
| `gemm/768x768x768/b1` | 14.5048 ± 0.0254377 (W 0.0268966) | 14.7973 ± 0.0387649 (W 0.0209522) | · |  |

## gemm-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0219105 ± 0.000379807 (W 0.000384104) | 0.024235 ± 0.000618826 (W 0.000780187) | 🔵 faer-fork 1.11× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.0841515 ± 0.000478494 (W 0.000616384) | 0.0707215 ± 0.000686757 (W 0.00103349) | 🔴 faer-fork 1.19× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.157974 ± 0.000689924 (W 0.000452682) | 0.155975 ± 0.000915704 (W 0.00120545) | · |  |
| `gemm/384x384x384/b1` | 0.464413 ± 0.00229523 (W 0.00271302) | 0.495181 ± 0.00116843 (W 0.00310362) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.11195 ± 0.00337253 (W 0.0029205) | 1.14962 ± 0.00272341 (W 0.00530864) | · |  |
| `gemm/768x768x768/b1` | 3.65255 ± 0.00769913 (W 0.00864629) | 3.76972 ± 0.00642953 (W 0.0137884) | · |  |

## gemm-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.022096 ± 0.000309568 (W 0.000371188) | 0.0157595 ± 0.000455755 (W 0.000387034) | 🔴 faer-fork 1.40× slower than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.048736 ± 0.000467925 (W 0.00107555) | 0.039774 ± 0.000505162 (W 0.000489537) | 🔴 faer-fork 1.23× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.107039 ± 0.00036522 (W 0.000772706) | 0.0831945 ± 0.000490809 (W 0.000533536) | 🔴 faer-fork 1.29× slower than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.248339 ± 0.00158502 (W 0.002038) | 0.255088 ± 0.000730187 (W 0.00154171) | · |  |
| `gemm/512x512x512/b1` | 0.594604 ± 0.00249702 (W 0.00314132) | 0.588162 ± 0.0016166 (W 0.00290951) | · |  |
| `gemm/768x768x768/b1` | 1.88063 ± 0.00449724 (W 0.00520107) | 1.90536 ± 0.00781026 (W 0.00668641) | · |  |

## gemm-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0669635 ± 0.000277798 (W 0.000105449) | 0.072149 ± 0.000752594 (W 0.000319745) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.223815 ± 0.000867659 (W 0.00123014) | 0.235468 ± 0.00163906 (W 0.00140325) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.531443 ± 0.00261984 (W 0.00243829) | 0.557602 ± 0.00386482 (W 0.0024005) | · |  |
| `gemm/384x384x384/b1` | 1.7975 ± 0.00545317 (W 0.00210268) | 1.86871 ± 0.00694997 (W 0.00515498) | · |  |
| `gemm/512x512x512/b1` | 4.24434 ± 0.011866 (W 0.00845045) | 4.37656 ± 0.0102691 (W 0.0100885) | · |  |
| `gemm/768x768x768/b1` | 14.512 ± 0.0585613 (W 0.0204721) | 14.7722 ± 0.0407831 (W 0.0170242) | · |  |

## gemm-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.021826 ± 0.000291983 (W 0.000455991) | 0.0239795 ± 0.000179386 (W 0.000409813) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.083806 ± 0.00043738 (W 0.000928968) | 0.070667 ± 0.000557965 (W 0.000974187) | 🔴 faer-fork 1.19× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.157823 ± 0.000823213 (W 0.000541633) | 0.15517 ± 0.000992793 (W 0.00134627) | · |  |
| `gemm/384x384x384/b1` | 0.464358 ± 0.00210676 (W 0.00317463) | 0.49469 ± 0.000949245 (W 0.00224145) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.11204 ± 0.00240189 (W 0.00210986) | 1.14643 ± 0.00300609 (W 0.00443938) | · |  |
| `gemm/768x768x768/b1` | 3.65509 ± 0.0152231 (W 0.00580309) | 3.77241 ± 0.0139058 (W 0.0182643) | · |  |

## gemm-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.02175 ± 0.000231695 (W 0.000433585) | 0.0154185 ± 0.00031086 (W 0.000392232) | 🔴 faer-fork 1.41× slower than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.048673 ± 0.000782921 (W 0.000995401) | 0.0395785 ± 0.00049403 (W 0.000497162) | 🔴 faer-fork 1.23× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.107214 ± 0.000413216 (W 0.000545329) | 0.083345 ± 0.000735971 (W 0.000845557) | 🔴 faer-fork 1.29× slower than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.24695 ± 0.00203229 (W 0.00301764) | 0.253933 ± 0.0014079 (W 0.00146423) | · |  |
| `gemm/512x512x512/b1` | 0.593831 ± 0.00236565 (W 0.00290747) | 0.590141 ± 0.00286465 (W 0.00294453) | · |  |
| `gemm/768x768x768/b1` | 1.87833 ± 0.00433616 (W 0.00404814) | 1.90563 ± 0.00811235 (W 0.00681667) | · |  |

## gemm-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.266494 ± 0.000984387 (W 0.00135856) | 0.286682 ± 0.000692651 (W 0.00205403) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.897559 ± 0.00306357 (W 0.00155188) | 0.944235 ± 0.00337731 (W 0.000998732) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 2.12641 ± 0.00526508 (W 0.00489601) | 2.23692 ± 0.00870037 (W 0.00317627) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 7.17966 ± 0.0196941 (W 0.00677805) | 7.4069 ± 0.0309158 (W 0.011273) | · |  |
| `gemm/512x512x512/b1` | 17.0025 ± 0.0459446 (W 0.0203429) | 17.6085 ± 0.0295318 (W 0.0231011) | · |  |
| `gemm/768x768x768/b1` | 57.8464 ± 0.256499 (W 0.0518802) | 59.1168 ± 0.202782 (W 0.0572129) | · |  |

## gemm-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0876225 ± 0.000312892 (W 0.000252081) | 0.078179 ± 0.000609057 (W 0.000858716) | 🔴 faer-fork 1.12× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.232895 ± 0.000876808 (W 0.00137915) | 0.244785 ± 0.000982422 (W 0.00185561) | · |  |
| `gemm/256x256x256/b1` | 0.5712 ± 0.00103994 (W 0.00237359) | 0.574749 ± 0.000644516 (W 0.00284716) | · |  |
| `gemm/384x384x384/b1` | 1.80552 ± 0.0175424 (W 0.00315565) | 1.88532 ± 0.0048821 (W 0.00175208) | · |  |
| `gemm/512x512x512/b1` | 4.33671 ± 0.00886805 (W 0.00755088) | 4.48755 ± 0.0260159 (W 0.00575311) | · |  |
| `gemm/768x768x768/b1` | 14.6376 ± 0.0273106 (W 0.0164709) | 15.139 ± 0.0252237 (W 0.0475558) | · |  |

## gemm-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.055698 ± 0.000495615 (W 0.000647258) | 0.0431605 ± 0.000275908 (W 0.000500483) | 🔴 faer-fork 1.29× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.15826 ± 0.000989339 (W 0.000923803) | 0.129928 ± 0.000752493 (W 0.00111518) | 🔴 faer-fork 1.22× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.308906 ± 0.00112309 (W 0.0013611) | 0.295071 ± 0.0017139 (W 0.0012649) | · |  |
| `gemm/384x384x384/b1` | 0.915748 ± 0.0031483 (W 0.00172146) | 0.947993 ± 0.00298945 (W 0.00152642) | · |  |
| `gemm/512x512x512/b1` | 2.21291 ± 0.0122467 (W 0.00435569) | 2.26772 ± 0.00619209 (W 0.00798366) | · |  |
| `gemm/768x768x768/b1` | 7.46929 ± 0.0222705 (W 0.0148657) | 7.63086 ± 0.0239554 (W 0.0504753) | · |  |

## gemm-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.2664 ± 0.00152694 (W 0.00127643) | 0.285595 ± 0.00121104 (W 0.00160619) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.899287 ± 0.00356965 (W 0.00145109) | 0.942424 ± 0.00271161 (W 0.00314803) | · |  |
| `gemm/256x256x256/b1` | 2.12605 ± 0.00705694 (W 0.00367494) | 2.22919 ± 0.00670099 (W 0.0036648) | · |  |
| `gemm/384x384x384/b1` | 7.17489 ± 0.0212459 (W 0.012125) | 7.39933 ± 0.035269 (W 0.00701831) | · |  |
| `gemm/512x512x512/b1` | 16.9986 ± 0.0319403 (W 0.0282068) | 17.6141 ± 0.0614501 (W 0.0269967) | · |  |
| `gemm/768x768x768/b1` | 57.8987 ± 0.132558 (W 0.0610487) | 59.211 ± 0.15904 (W 0.0719271) | · |  |

## gemm-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.087713 ± 0.000310357 (W 0.000198822) | 0.0779545 ± 0.000720276 (W 0.000539692) | 🔴 faer-fork 1.13× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.233104 ± 0.000616961 (W 0.00107205) | 0.244537 ± 0.000881682 (W 0.00144617) | · |  |
| `gemm/256x256x256/b1` | 0.572266 ± 0.00131084 (W 0.00174538) | 0.574124 ± 0.00160208 (W 0.00211258) | · |  |
| `gemm/384x384x384/b1` | 1.8053 ± 0.0033913 (W 0.00295527) | 1.88307 ± 0.00301375 (W 0.00239358) | · |  |
| `gemm/512x512x512/b1` | 4.33441 ± 0.00595045 (W 0.00952955) | 4.49088 ± 0.0154666 (W 0.00761382) | · |  |
| `gemm/768x768x768/b1` | 14.6465 ± 0.0248955 (W 0.0192534) | 15.1436 ± 0.0372349 (W 0.0344607) | · |  |

## gemm-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0558735 ± 0.000587152 (W 0.000522937) | 0.0432875 ± 0.000275625 (W 0.00043108) | 🔴 faer-fork 1.29× slower than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.158273 ± 0.000868329 (W 0.00097546) | 0.129765 ± 0.000653272 (W 0.00157709) | 🔴 faer-fork 1.22× slower than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.309447 ± 0.00115921 (W 0.00176925) | 0.296412 ± 0.00174242 (W 0.00180597) | · |  |
| `gemm/384x384x384/b1` | 0.916244 ± 0.00276032 (W 0.00131671) | 0.947262 ± 0.00381254 (W 0.00193545) | · |  |
| `gemm/512x512x512/b1` | 2.20917 ± 0.00941428 (W 0.00367923) | 2.26667 ± 0.00821241 (W 0.00505199) | · |  |
| `gemm/768x768x768/b1` | 7.47559 ± 0.027115 (W 0.0144265) | 7.61768 ± 0.0253713 (W 0.0311277) | · |  |

## gemm-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.27236 ± 0.00599212 (W 0.00506435) | 1.63673 ± 0.0084468 (W 0.00413822) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.07972 ± 0.0271058 (W 0.00660988) | 1.46083 ± 0.00259108 (W 0.00736758) | 🔵 faer-fork 1.35× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 1.08616 ± 0.00647352 (W 0.00148456) | 1.36516 ± 0.0064975 (W 0.00169585) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.421187 ± 0.00240217 (W 0.00418539) | 0.502914 ± 0.00979981 (W 0.00490241) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.308179 ± 0.00222026 (W 0.00235121) | 0.476652 ± 0.00113256 (W 0.00196255) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.275514 ± 0.00135024 (W 0.00113777) | 0.347572 ± 0.00165634 (W 0.00154543) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.419487 ± 0.00182412 (W 0.00295824) | 0.282294 ± 0.00165655 (W 0.0030498) | 🔴 faer-fork 1.49× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.187105 ± 0.000683176 (W 0.0020954) | 0.312053 ± 0.00108067 (W 0.00136945) | 🔵 faer-fork 1.67× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.141373 ± 0.000808153 (W 0.00119771) | 0.178747 ± 0.0070672 (W 0.00116109) | 🔵 faer-fork 1.26× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.03063 ± 0.00444138 (W 0.00195374) | 1.68727 ± 0.00752417 (W 0.00439989) | 🔵 faer-fork 1.64× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.09198 ± 0.0128935 (W 0.0054898) | 1.42775 ± 0.00753943 (W 0.0121352) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.12323 ± 0.00373368 (W 0.00271143) | 1.36812 ± 0.00449085 (W 0.00313279) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.32028 ± 0.000962867 (W 0.00313074) | 0.423501 ± 0.00233042 (W 0.0026283) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.297604 ± 0.00142631 (W 0.00232133) | 0.481583 ± 0.00163852 (W 0.00198157) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.294052 ± 0.00129647 (W 0.00192466) | 0.364359 ± 0.000934665 (W 0.0017394) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.195239 ± 0.001705 (W 0.00265269) | 0.222643 ± 0.000902332 (W 0.00278172) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.181401 ± 0.00154636 (W 0.00218839) | 0.324458 ± 0.000988207 (W 0.00123418) | 🔵 faer-fork 1.79× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.154591 ± 0.000867997 (W 0.00141888) | 0.198008 ± 1.12626 (W 0.00231438) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.14141 ± 0.0141404 (W 0.00774289) | 5.96123 ± 0.022181 (W 0.0143895) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 4.36595 ± 0.0221198 (W 0.00971728) | 5.78217 ± 0.0211696 (W 0.0174682) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 4.3666 ± 0.0104719 (W 0.00785095) | 5.39977 ± 0.0115528 (W 0.0115356) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.65253 ± 0.00756894 (W 0.00644196) | 1.72828 ± 0.0189461 (W 0.00561876) | · |  |
| `gemm/4096x16x240/b1` | 1.1511 ± 0.00343477 (W 0.00407592) | 1.67536 ± 0.0087679 (W 0.004377) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.12645 ± 0.0040439 (W 0.00188766) | 1.38386 ± 0.00126975 (W 0.00238447) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.88823 ± 0.0114541 (W 0.0163555) | 0.933354 ± 0.00529543 (W 0.0118658) | · |  |
| `gemm/4096x16x240/b1` | 0.590352 ± 0.00373927 (W 0.00374323) | 1.00022 ± 0.00437183 (W 0.00416418) | 🔵 faer-fork 1.69× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.569861 ± 0.00306128 (W 0.00259269) | 0.694611 ± 0.00315643 (W 0.00262923) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.0577 ± 0.0101101 (W 0.0148893) | 5.72309 ± 0.0257196 (W 0.00747292) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 4.33773 ± 0.0179915 (W 0.00619847) | 5.77635 ± 0.0288481 (W 0.0167356) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 4.40802 ± 0.0211498 (W 0.00834694) | 5.53096 ± 0.0208618 (W 0.00557777) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.17501 ± 0.00410655 (W 0.00268843) | 1.49794 ± 0.00625544 (W 0.00450675) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.15991 ± 0.0057723 (W 0.0030474) | 1.7342 ± 0.00316842 (W 0.0051384) | 🔵 faer-fork 1.50× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.15962 ± 0.00492078 (W 0.00508222) | 1.44034 ± 0.00513849 (W 0.00430587) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.654965 ± 0.00287307 (W 0.00619105) | 0.824134 ± 0.00393858 (W 0.00400239) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.589109 ± 0.00270832 (W 0.00375222) | 1.01386 ± 0.00709634 (W 0.00370113) | 🔵 faer-fork 1.72× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.571326 ± 0.00352039 (W 0.00375758) | 0.766458 ± 0.00184348 (W 0.00260224) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.542717 ± 0.00244525 (W 0.00220289) | 0.691046 ± 0.00146539 (W 0.00227283) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.542202 ± 0.0142677 (W 0.00169713) | 0.643721 ± 0.00986342 (W 0.00228127) | 🔵 faer-fork 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.539391 ± 0.00265739 (W 0.00192846) | 0.594524 ± 0.00191819 (W 0.00195464) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.40799 ± 0.00388313 (W 0.00209032) | 0.213022 ± 0.00136543 (W 0.00133056) | 🔴 faer-fork 1.92× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.143964 ± 0.000831963 (W 0.00104385) | 0.16439 ± 0.00203705 (W 0.000871041) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.139764 ± 0.00023933 (W 0.000436987) | 0.152224 ± 0.00116002 (W 0.000710572) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.407796 ± 0.00151103 (W 0.00200304) | 0.122662 ± 0.00121486 (W 0.00154985) | 🔴 faer-fork 3.32× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.0775125 ± 0.000790271 (W 0.000694913) | 0.089311 ± 0.000700903 (W 0.000292488) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.071839 ± 0.000417323 (W 0.000347435) | 0.081771 ± 0.000568569 (W 0.000625949) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |

## gemm-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.537832 ± 0.00472152 (W 0.00278125) | 0.638601 ± 0.00575532 (W 0.00191934) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.544787 ± 0.00257966 (W 0.00183833) | 0.676759 ± 0.00534832 (W 0.00141983) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.551403 ± 0.00355334 (W 0.0022847) | 0.681613 ± 0.00730479 (W 0.00272993) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.176016 ± 0.00181965 (W 0.00206061) | 0.182869 ± 0.000770416 (W 0.0016531) | · |  |
| `gemm/4096x16x240/b1` | 0.146039 ± 0.000513452 (W 0.00165962) | 0.184495 ± 0.00895488 (W 0.001684) | 🔵 faer-fork 1.26× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.156126 ± 0.00159311 (W 0.0020376) | 0.177111 ± 0.000770574 (W 0.00121807) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |

## gemm-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.110892 ± 0.00114045 (W 0.000873778) | 0.109318 ± 0.000473072 (W 0.00151263) | · |  |
| `gemm/4096x16x240/b1` | 0.0811365 ± 0.000438049 (W 0.00048193) | 0.121847 ± 0.00194476 (W 0.0025516) | 🔵 faer-fork 1.50× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.078546 ± 0.000582963 (W 0.000319955) | 0.102631 ± 0.00147023 (W 0.00159188) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.11547 ± 0.0109653 (W 0.00413025) | 2.46605 ± 0.00907441 (W 0.00776656) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 2.15048 ± 0.013703 (W 0.0028696) | 2.72952 ± 0.0136527 (W 0.0031834) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 2.15714 ± 0.00792283 (W 0.00355584) | 2.40261 ± 0.00959678 (W 0.00297044) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.814326 ± 0.0028817 (W 0.00295092) | 0.707536 ± 0.00216043 (W 0.00290556) | 🔴 faer-fork 1.15× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.581419 ± 0.00157014 (W 0.00252653) | 0.715585 ± 0.00173255 (W 0.00202776) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.549014 ± 0.00159131 (W 0.00192637) | 0.604019 ± 0.00273779 (W 0.00172365) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.810153 ± 0.00265542 (W 0.00278717) | 0.380951 ± 0.0020996 (W 0.00246762) | 🔴 faer-fork 2.13× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.299499 ± 0.000429813 (W 0.00168905) | 0.435612 ± 0.96103 (W 0.00193728) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.277389 ± 0.00184545 (W 0.00228066) | 0.30293 ± 1.08455 (W 0.00177672) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.02457 ± 0.0103058 (W 0.00323851) | 2.57141 ± 0.0167893 (W 0.00711768) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 2.17007 ± 0.00957435 (W 0.00655813) | 2.75601 ± 0.0135277 (W 0.00399288) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 2.18854 ± 0.00945453 (W 0.00251945) | 2.54335 ± 0.0270544 (W 0.00471074) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.590413 ± 0.00243443 (W 0.0034684) | 0.645945 ± 0.00340314 (W 0.00259809) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.570442 ± 0.00335906 (W 0.00303336) | 0.771559 ± 0.00323085 (W 0.00607044) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.571697 ± 0.00178658 (W 0.0022597) | 0.644143 ± 0.00210138 (W 0.00299877) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |

## gemm-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.323692 ± 0.00188791 (W 0.00210956) | 0.340119 ± 0.00129052 (W 0.00173029) | · |  |
| `gemm/4096x16x240/b1` | 0.289725 ± 0.00165491 (W 0.00125005) | 0.439764 ± 3.4167 (W 0.00517982) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.307957 ± 0.00150953 (W 0.00147855) | 0.334917 ± 0.00225689 (W 0.00192405) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.13008 ± 0.00586134 (W 0.00252139) | 1.36607 ± 0.0153722 (W 0.00456048) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |

## gemm-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.295198 ± 0.00138947 (W 0.00145099) | 0.364057 ± 0.00123769 (W 0.00201816) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |

## gemm-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.154955 ± 0.000511692 (W 0.00110173) | 0.195881 ± 2.97515 (W 0.00193962) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.08347 ± 0.00355006 (W 0.00105144) | 1.36617 ± 0.00745537 (W 0.00344929) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |

## gemm-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.276778 ± 0.00193656 (W 0.0016881) | 0.346801 ± 0.00106162 (W 0.00162928) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |

## gemm-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.140844 ± 0.000951046 (W 0.00113208) | 0.178581 ± 1.13579 (W 0.00103669) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.43183 ± 0.0100697 (W 0.0130029) | 5.53153 ± 0.0307986 (W 0.00892786) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |

## gemm-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.15708 ± 0.00298175 (W 0.00411169) | 1.43713 ± 0.00340333 (W 0.00347378) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |

## gemm-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.57089 ± 0.00421552 (W 0.0037532) | 0.765089 ± 0.00326897 (W 0.00248016) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.37467 ± 0.0179454 (W 0.0112461) | 5.39924 ± 0.0216191 (W 0.0148363) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.12429 ± 0.00392728 (W 0.00256401) | 1.38172 ± 0.00992394 (W 0.0055361) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.569126 ± 0.00217638 (W 0.00221618) | 0.693931 ± 0.0023117 (W 0.00252793) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |

## gemm-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.551463 ± 0.00319339 (W 0.0021163) | 0.687696 ± 0.0050628 (W 0.00269674) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |

## gemm-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.155757 ± 0.00169797 (W 0.00259456) | 0.176444 ± 0.00209011 (W 0.00120612) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |

## gemm-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.0788315 ± 0.000318917 (W 0.000415696) | 0.103583 ± 0.00327713 (W 0.00219275) | 🔵 faer-fork 1.31× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.542918 ± 0.00450797 (W 0.00174402) | 0.594973 ± 0.00647025 (W 0.00222411) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.139318 ± 0.000586479 (W 0.00039523) | 0.151938 ± 0.000891718 (W 0.000763889) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.071611 ± 0.000190933 (W 0.000452688) | 0.081432 ± 0.000484863 (W 0.000142132) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.1904 ± 0.00769321 (W 0.00399323) | 2.53965 ± 0.0247694 (W 0.00436087) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.57053 ± 0.00184106 (W 0.00241182) | 0.641755 ± 0.00373906 (W 0.00263704) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.308669 ± 0.000957129 (W 0.00151969) | 0.335033 ± 0.00191958 (W 0.00155096) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.16249 ± 0.0126242 (W 0.0023808) | 2.40414 ± 0.0116657 (W 0.00364882) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.551584 ± 0.0023194 (W 0.00212112) | 0.60453 ± 0.00258755 (W 0.00166808) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.276797 ± 0.00169176 (W 0.00129903) | 0.302146 ± 1.08484 (W 0.00161623) | · | `!` between-start range over 5%; `p` position effect over 5% |

## linalg-small, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004208 ± 7.60044e-05 (W 2.83691e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003507 ± 1.5371e-05 (W 5.85045e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014086 ± 0.000406853 (W 0.000690194) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012538 ± 0.000319595 (W 9.34909e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031349 ± 0.000446379 (W 0.000111574) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.028824 ± 0.000589881 (W 8.79e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.110855 ± 0.00105176 (W 0.000720272) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.111852 ± 0.000617052 (W 0.000259216) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.570531 ± 0.00108501 (W 0.00211715) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.575044 ± 0.00220775 (W 0.00215113) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.33488 ± 0.00427966 (W 0.00339653) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.35784 ± 0.0040767 (W 0.00356121) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0105705 ± 0.000112038 (W 6.49976e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013631 ± 0.000108709 (W 0.000114303) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0494495 ± 0.000137856 (W 0.000234598) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.057753 ± 0.000276048 (W 0.000204755) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.136167 ± 0.000546691 (W 0.00127995) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.161822 ± 0.00082025 (W 0.00209563) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0083915 ± 7.01249e-05 (W 8.0785e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0083805 ± 6.21442e-05 (W 4.64393e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0410615 ± 0.000198768 (W 0.000734079) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0410815 ± 0.000242526 (W 0.000122472) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.122051 ± 0.000505514 (W 0.00111506) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.121957 ± 0.000792337 (W 0.00161455) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006372 ± 2.58199e-05 (W 6.4073e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00815 ± 8.4579e-05 (W 6.75508e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.023489 ± 0.00017356 (W 9.33255e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.028629 ± 8.82534e-05 (W 0.000161427) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.055614 ± 0.000321076 (W 0.000159101) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.074639 ± 0.000361862 (W 0.000957918) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.036017 ± 0.000276587 (W 0.000122026) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.031599 ± 0.000417284 (W 0.000196186) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.106128 ± 0.000773885 (W 0.000588329) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.115324 ± 0.00234661 (W 0.000352993) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.231659 ± 0.000731558 (W 0.00139785) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.292893 ± 0.000901421 (W 0.00240908) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.024345 ± 0.000128688 (W 0.000113326) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0681315 ± 0.000345555 (W 0.000655457) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.13958 ± 0.000448841 (W 0.00119646) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0066525 ± 2.6664e-05 (W 4.62393e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009277 ± 3.20354e-05 (W 6.61272e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0218815 ± 0.00012302 (W 8.3114e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.029966 ± 0.000122466 (W 0.000108467) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.049643 ± 0.000252832 (W 8.61893e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0756805 ± 0.00183722 (W 0.00249108) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.167768 ± 0.00784952 (W 0.00115118) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.167463 ± 0.00215974 (W 0.00161394) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.678616 ± 0.00878408 (W 0.00152424) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.681091 ± 0.00623496 (W 0.0018933) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.56736 ± 0.0067091 (W 0.00286692) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.56866 ± 0.00851356 (W 0.00287242) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0586095 ± 0.000106826 (W 0.000116935) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0589845 ± 0.000244237 (W 0.000175253) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.213474 ± 0.00118877 (W 0.00133892) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.208844 ± 0.00095838 (W 0.0014603) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.478808 ± 0.00300962 (W 0.00176406) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.479344 ± 0.00205406 (W 0.00213067) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0007165 ± 2.00566e-05 (W 2.90078e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001603 ± 1.71318e-05 (W 4.70697e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.002735 ± 3.78153e-05 (W 5.51556e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004143 ± 0.000263106 (W 3.85742e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003556 ± 3.27455e-05 (W 5.60779e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0142115 ± 0.000429953 (W 7.95856e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012804 ± 5.25268e-05 (W 8.70638e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.03192 ± 0.000122993 (W 0.000175355) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.029441 ± 0.00015908 (W 0.000129074) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.076407 ± 0.00173196 (W 0.000247784) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.076846 ± 0.000347446 (W 0.000283043) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.315229 ± 0.00107846 (W 0.00145663) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.327282 ± 0.00262587 (W 0.00273054) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.770902 ± 0.00218542 (W 0.00252606) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.771652 ± 0.00400227 (W 0.00218512) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011546 ± 0.000125954 (W 5.00674e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0147675 ± 0.000153998 (W 0.000118446) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0508245 ± 0.000471579 (W 7.9089e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0598985 ± 0.000335873 (W 0.000136393) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.144845 ± 0.00100166 (W 0.00084834) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.171009 ± 0.000704933 (W 0.00253365) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006897 ± 0.00013579 (W 0.000308965) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006061 ± 0.000119428 (W 0.000109334) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0207385 ± 0.000278536 (W 0.000350821) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0189765 ± 0.000193066 (W 0.000259549) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0466085 ± 0.000507069 (W 0.000637395) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.045109 ± 0.00018181 (W 0.000440767) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006632 ± 3.45683e-05 (W 6.49775e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008451 ± 6.15481e-05 (W 0.000114261) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0271555 ± 0.000229194 (W 0.000236905) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0292845 ± 0.000208956 (W 0.00020433) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0666485 ± 0.00112791 (W 0.00147486) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0715905 ± 0.000451165 (W 0.00183677) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0373795 ± 0.000262427 (W 0.000135249) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0329915 ± 0.000124617 (W 0.000188058) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.110606 ± 0.000997893 (W 0.00100232) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.109114 ± 0.00082486 (W 0.000373907) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.216572 ± 0.00153948 (W 0.00175169) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.246881 ± 0.00144901 (W 0.00238058) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0252275 ± 8.85974e-05 (W 9.72692e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.075626 ± 0.000301672 (W 0.000256831) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.148807 ± 0.00151807 (W 0.000501646) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007394 ± 6.43752e-05 (W 5.13394e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0099245 ± 5.20679e-05 (W 7.85197e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.026544 ± 0.000380657 (W 0.000422026) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.03144 ± 0.000233388 (W 0.000143747) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.053349 ± 0.00039811 (W 0.00066552) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.075284 ± 0.000269279 (W 0.00251336) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.169971 ± 0.00154557 (W 0.00106963) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.170692 ± 0.000890978 (W 0.000383361) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.554849 ± 0.00216047 (W 0.00169166) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.557561 ± 0.00501059 (W 0.00254387) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.13681 ± 0.0045311 (W 0.00220046) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.13978 ± 0.00399943 (W 0.00229564) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0595575 ± 0.000281987 (W 0.000131226) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0602725 ± 0.000254541 (W 0.000118372) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.217682 ± 0.00104964 (W 0.00123385) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.215935 ± 0.00142574 (W 0.00121254) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.51381 ± 0.00247031 (W 0.00156701) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.515377 ± 0.00140394 (W 0.00226025) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000922 ± 1.68572e-05 (W 3.20671e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0020535 ± 1.19276e-05 (W 3.21033e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003406 ± 4.30999e-05 (W 4.67748e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004198 ± 0.000165771 (W 3.41354e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0034565 ± 2.6438e-05 (W 4.64326e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014251 ± 0.000312888 (W 7.92662e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0127685 ± 0.000272315 (W 6.44469e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031739 ± 0.000614852 (W 0.000197568) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.02935 ± 0.000477072 (W 0.000153845) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.071585 ± 0.000977083 (W 0.00020383) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0723915 ± 0.000194456 (W 0.000274117) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.236321 ± 0.00284896 (W 0.0016611) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.240398 ± 0.00632547 (W 0.00227644) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 0.499324 ± 0.00340918 (W 0.00254936) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.500544 ± 0.00362549 (W 0.00260085) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011807 ± 0.00010508 (W 6.70828e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0147915 ± 0.000289931 (W 0.000136832) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0524475 ± 0.000282322 (W 9.67048e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0599815 ± 0.000199973 (W 0.000230782) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.155923 ± 0.00115252 (W 0.000737828) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.181208 ± 0.00765732 (W 0.00248062) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006132 ± 0.000209017 (W 0.000170828) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0061065 ± 7.73968e-05 (W 0.000167625) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014747 ± 0.000190318 (W 0.00017608) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0166455 ± 0.000254381 (W 0.000438742) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.032394 ± 0.000239511 (W 0.000596876) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0338985 ± 0.00032592 (W 0.000817795) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0065875 ± 4.63465e-05 (W 3.51946e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0084005 ± 8.33763e-05 (W 5.56075e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.030222 ± 0.00058684 (W 0.000627711) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.0292935 ± 0.000151659 (W 0.000203279) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.068494 ± 0.000807273 (W 0.00100633) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.073464 ± 0.000638634 (W 0.00130499) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0374695 ± 0.000243803 (W 0.000181058) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.032945 ± 0.000149435 (W 0.000153343) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.114006 ± 0.000301477 (W 0.000862961) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.106967 ± 0.000679189 (W 0.000403027) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.211679 ± 0.00166603 (W 0.00158133) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.242857 ± 0.00158084 (W 0.00261251) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025067 ± 0.000143711 (W 0.000115752) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0758415 ± 0.0133979 (W 0.000303379) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.141502 ± 0.000651901 (W 0.000395684) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007414 ± 0.000112452 (W 6.0111e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010059 ± 0.000212067 (W 6.73358e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0265455 ± 0.00019271 (W 0.000306578) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.031664 ± 0.000243904 (W 0.000149355) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0576825 ± 0.000765084 (W 0.00125376) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.078498 ± 0.000462418 (W 0.00267618) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.169913 ± 0.00128671 (W 0.000306054) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.170366 ± 0.00105078 (W 0.000383724) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.553531 ± 0.00253125 (W 0.00188988) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.554706 ± 0.0014819 (W 0.00198435) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.14256 ± 0.00890595 (W 0.00169835) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15303 ± 0.00679839 (W 0.00304411) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0597515 ± 0.000442934 (W 0.000141325) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0599085 ± 0.000313622 (W 0.000151491) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.214555 ± 0.000889098 (W 0.00100593) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.215583 ± 0.00100992 (W 0.001247) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.534461 ± 0.00379182 (W 0.0014769) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.535816 ± 0.00320669 (W 0.00183662) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0009465 ± 2.05953e-05 (W 2.85446e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002154 ± 2.89828e-05 (W 4.81298e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0036865 ± 3.4616e-05 (W 4.79909e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0028155 ± 0.000207076 (W 3.12597e-05) | 0.002926 ± 6.60316e-05 (W 4.22004e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0035065 ± 2.96018e-05 (W 7.47718e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010655 ± 0.0003865 (W 5.52315e-05) | 0.012969 ± 0.000149867 (W 8.81388e-05) | 🔵 faer-fork 1.22× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012578 ± 0.000461056 (W 9.91841e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0249215 ± 3.72688e-05 (W 7.73873e-05) | 0.0315285 ± 0.00017602 (W 0.000119947) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.0288145 ± 9.91432e-05 (W 0.000104154) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.111121 ± 0.000573214 (W 0.00161345) | 0.121651 ± 0.00120459 (W 0.000550915) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.112068 ± 0.000517902 (W 0.000805287) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.571238 ± 0.00198035 (W 0.00207492) | 0.441121 ± 0.0036432 (W 0.00238272) | 🔴 faer-fork 1.29× slower than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.575261 ± 0.00181956 (W 0.00238462) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.32919 ± 0.00417163 (W 0.00291602) | 0.992759 ± 0.00640101 (W 0.00251399) | 🔴 faer-fork 1.34× slower than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.36006 ± 0.00352693 (W 0.00415378) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01067 ± 6.30608e-05 (W 6.74455e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0121325 ± 0.00231058 (W 8.28705e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0494615 ± 0.000267478 (W 0.000144348) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0523175 ± 0.000519311 (W 0.0002136) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.136569 ± 0.000363458 (W 0.000294619) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.148977 ± 0.000418055 (W 0.0015356) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0083905 ± 3.51511e-05 (W 0.000104816) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0084155 ± 8.89504e-05 (W 7.87621e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0405155 ± 0.000220916 (W 9.84936e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0405905 ± 0.000112665 (W 0.000151545) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.1241 ± 0.000631318 (W 0.00129892) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.123655 ± 0.000610304 (W 0.00102448) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005055 ± 0.00013727 (W 6.80826e-05) | 0.007444 ± 0.000106661 (W 7.9154e-05) | 🔵 faer-fork 1.47× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0065025 ± 0.000153975 (W 0.000102447) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020453 ± 0.00024085 (W 0.000191739) | 0.030041 ± 0.000231795 (W 0.000157097) | 🔵 faer-fork 1.47× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.023375 ± 0.000150722 (W 0.000215112) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.049111 ± 0.000345574 (W 0.000281212) | 0.0697175 ± 0.000346667 (W 0.000173281) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0613445 ± 0.000190134 (W 0.00216857) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0278885 ± 0.000157131 (W 0.000155746) | 0.031594 ± 0.000206642 (W 0.000161578) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.030126 ± 0.000147094 (W 0.000131885) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.102719 ± 0.000465381 (W 0.0003325) | 0.125663 ± 0.00102502 (W 0.00142995) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.108251 ± 0.000363425 (W 0.00115601) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.239686 ± 0.00173084 (W 0.00125063) | 0.315421 ± 0.00126866 (W 0.00161321) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.277971 ± 0.00121381 (W 0.00290909) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.016556 ± 0.000102738 (W 8.95044e-05) | 0.014246 ± 0.00015076 (W 9.30836e-05) | 🔴 faer-fork 1.16× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0674805 ± 0.000303252 (W 0.000196121) | 0.056801 ± 0.000497415 (W 0.00018068) | 🔴 faer-fork 1.19× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.145932 ± 0.000463462 (W 0.0015501) | 0.161931 ± 0.00100286 (W 0.00115671) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.005936 ± 2.16025e-05 (W 4.66064e-05) | 0.007655 ± 8.80772e-05 (W 9.12588e-05) | 🔵 faer-fork 1.29× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0074335 ± 4.03175e-05 (W 3.45982e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021055 ± 0.000140424 (W 7.87564e-05) | 0.029555 ± 0.000208171 (W 0.000115049) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.024186 ± 0.000152359 (W 0.000140308) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0494435 ± 0.000728297 (W 0.000949735) | 0.069554 ± 0.000556611 (W 0.00019663) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.0631625 ± 0.000914831 (W 0.00242869) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.170318 ± 0.00132889 (W 0.00125762) | 0.179372 ± 0.000976862 (W 0.00150767) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.171419 ± 0.0111722 (W 0.00125631) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.689592 ± 0.00365878 (W 0.00233395) | 0.655963 ± 0.00250452 (W 0.0016327) | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.689687 ± 0.00280316 (W 0.00206915) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.58375 ± 0.00899807 (W 0.00475336) | 1.5972 ± 0.00565043 (W 0.00441772) | · |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.58957 ± 0.00646536 (W 0.00422998) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058386 ± 0.000551152 (W 0.000113065) | 0.069664 ± 0.000625536 (W 0.000173604) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.0587515 ± 0.000335266 (W 0.00117708) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.215541 ± 0.000995236 (W 0.00101331) | 0.263416 ± 0.000632283 (W 0.00180886) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.21163 ± 0.00155246 (W 0.00106626) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.489949 ± 0.0025725 (W 0.00163614) | 0.651795 ± 0.00331098 (W 0.00171666) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.490932 ± 0.00232854 (W 0.00158658) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000802 ± 1.85194e-05 (W 3.03655e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001794 ± 1.74059e-05 (W 4.81925e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0029955 ± 6.05827e-05 (W 6.34707e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0028105 ± 0.000236708 (W 2.2043e-05) | 0.002925 ± 2.73526e-05 (W 4.7982e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003537 ± 2.64247e-05 (W 4.65372e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010895 ± 0.000487291 (W 5.83086e-05) | 0.012824 ± 8.78658e-05 (W 0.000114129) | 🔵 faer-fork 1.18× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012864 ± 8.85673e-05 (W 9.37342e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.025593 ± 0.000129385 (W 5.97739e-05) | 0.0312735 ± 0.000210904 (W 0.000104081) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.0294105 ± 0.000163316 (W 0.000118152) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.075934 ± 0.000421429 (W 0.00028283) | 0.10865 ± 0.000818212 (W 0.00102157) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.076737 ± 0.000558708 (W 0.00027206) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.314065 ± 0.00142043 (W 0.00159907) | 0.369011 ± 0.00144386 (W 0.00166519) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.325628 ± 0.00101951 (W 0.00241384) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.768736 ± 0.00280735 (W 0.00209574) | 0.803557 ± 0.00417016 (W 0.0015497) | · |  |
| `eigh/n96/b1 [api-inclusive]` | 0.772039 ± 0.00328698 (W 0.00210004) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0114565 ± 0.000169447 (W 4.44362e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01335 ± 0.000176137 (W 0.000107257) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0510565 ± 0.000423276 (W 0.000167479) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.054366 ± 0.000674105 (W 0.000233525) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.14258 ± 0.00140875 (W 0.000957291) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.15899 ± 0.00150616 (W 0.00173144) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006502 ± 6.9857e-05 (W 0.000158271) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006492 ± 7.94072e-05 (W 0.000122019) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.019913 ± 0.000165905 (W 0.000365713) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.019481 ± 0.000131662 (W 0.000147359) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.044708 ± 0.000187007 (W 0.00028654) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0443625 ± 0.000205402 (W 0.000314085) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005325 ± 5.43093e-05 (W 0.000102582) | 0.0074845 ± 0.000146934 (W 8.361e-05) | 🔵 faer-fork 1.41× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0067375 ± 4.34542e-05 (W 8.45923e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.021084 ± 0.000233974 (W 0.000225473) | 0.030157 ± 0.000364478 (W 0.000154227) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0239995 ± 0.000230569 (W 0.00023662) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0434575 ± 0.000327539 (W 0.000292599) | 0.070101 ± 0.000446509 (W 0.000167922) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.058529 ± 0.000180341 (W 0.00204201) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.028999 ± 0.000267874 (W 0.000124235) | 0.0316085 ± 0.000313563 (W 0.000144878) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.031239 ± 0.000206691 (W 0.000100724) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0954055 ± 0.000290385 (W 0.00031456) | 0.123608 ± 0.000896474 (W 0.00121853) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.101644 ± 0.00073244 (W 0.00032443) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.186213 ± 0.000885096 (W 0.00111496) | 0.302751 ± 0.00133679 (W 0.00195154) | 🔵 faer-fork 1.63× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.231833 ± 0.0015903 (W 0.00236526) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.017257 ± 8.21764e-05 (W 0.000133118) | 0.014252 ± 0.000170668 (W 0.000106028) | 🔴 faer-fork 1.21× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067626 ± 0.000311757 (W 0.000247701) | 0.0562015 ± 0.000869411 (W 0.00163063) | 🔴 faer-fork 1.20× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.132114 ± 0.000651337 (W 0.000439105) | 0.164766 ± 0.000671438 (W 0.00155498) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.006527 ± 0.000136622 (W 5.84272e-05) | 0.0076345 ± 8.04628e-05 (W 8.89346e-05) | 🔵 faer-fork 1.17× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0081005 ± 0.000148917 (W 4.87255e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0233185 ± 0.000207447 (W 0.000165254) | 0.0297955 ± 0.000217459 (W 9.7821e-05) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.025543 ± 7.47389e-05 (W 0.000154721) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0464115 ± 0.000827225 (W 0.000219624) | 0.0700305 ± 0.000454848 (W 0.000112022) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.0633925 ± 0.00103615 (W 0.00252929) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.172586 ± 0.00932031 (W 0.00142742) | 0.180777 ± 0.000421983 (W 0.00138227) | · | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.172873 ± 0.000936835 (W 0.000250057) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.556696 ± 0.00251126 (W 0.00175032) | 0.656985 ± 0.00282862 (W 0.00176973) | 🔵 faer-fork 1.18× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.558407 ± 0.00371576 (W 0.00222655) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.15325 ± 0.00596573 (W 0.00319383) | 1.56931 ± 0.00688049 (W 0.00302259) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15206 ± 0.00518312 (W 0.00231083) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0597875 ± 0.000431911 (W 0.00013865) | 0.06969 ± 0.000365134 (W 0.00202614) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.0602065 ± 0.000374989 (W 0.000127868) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.221187 ± 0.0022828 (W 0.00119381) | 0.263565 ± 0.000828494 (W 0.00178696) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.217818 ± 0.000874424 (W 0.00118828) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.52335 ± 0.000847804 (W 0.00193404) | 0.658675 ± 0.00209733 (W 0.00155678) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.525003 ± 0.00169415 (W 0.00175053) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000977 ± 8.16497e-06 (W 2.49827e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002209 ± 4.08248e-05 (W 3.68348e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003632 ± 3.18852e-05 (W 4.29276e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002815 ± 0.000165144 (W 1.11907e-05) | 0.0028705 ± 2.91256e-05 (W 4.81536e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0034565 ± 2.53824e-05 (W 5.05692e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0106945 ± 7.9089e-05 (W 4.37763e-05) | 0.0129445 ± 0.000114643 (W 6.47139e-05) | 🔵 faer-fork 1.21× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0127735 ± 4.93113e-05 (W 8.31072e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.025288 ± 0.000238713 (W 7.17525e-05) | 0.0316045 ± 0.000195419 (W 0.000600576) | 🔵 faer-fork 1.25× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.029425 ± 0.000127944 (W 9.43394e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0716705 ± 0.00263559 (W 0.00026778) | 0.105719 ± 0.000408028 (W 0.00114865) | 🔵 faer-fork 1.48× faster than `lapack-openblas` | `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.0721965 ± 0.000482287 (W 0.000279743) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.233192 ± 0.0018082 (W 0.00146516) | 0.355286 ± 0.000828261 (W 0.0016478) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.241585 ± 0.00772582 (W 0.00230922) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 0.499742 ± 0.00306745 (W 0.00300824) | 0.777498 ± 0.00366381 (W 0.002529) | 🔵 faer-fork 1.56× faster than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.501367 ± 0.00145711 (W 0.00164561) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0118125 ± 0.00041189 (W 7.4159e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01318 ± 0.00033339 (W 9.97822e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0528465 ± 0.000601844 (W 0.000741399) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.054386 ± 0.000261711 (W 0.000187344) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.155293 ± 0.00107088 (W 0.000566843) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.168347 ± 0.00168932 (W 0.00140358) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005886 ± 4.73022e-05 (W 0.000142515) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0058765 ± 6.61896e-05 (W 0.000103602) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0145365 ± 8.63381e-05 (W 0.000205517) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014572 ± 6.61322e-05 (W 0.000212654) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.030196 ± 0.000255431 (W 0.000166798) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0305565 ± 0.000245374 (W 0.000283357) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005315 ± 7.96869e-05 (W 7.11044e-05) | 0.007429 ± 8.95917e-05 (W 9.30622e-05) | 🔵 faer-fork 1.40× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0067625 ± 5.67007e-05 (W 0.000115616) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020954 ± 0.000180744 (W 0.000206463) | 0.030091 ± 0.000168855 (W 0.000116109) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0240645 ± 0.00021197 (W 0.000282565) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.043371 ± 0.000121974 (W 0.00039581) | 0.0702225 ± 0.000371287 (W 0.00127113) | 🔵 faer-fork 1.62× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0618425 ± 0.0010635 (W 0.00138362) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.029075 ± 0.000136849 (W 0.000150001) | 0.031614 ± 0.000366599 (W 0.000121364) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.031314 ± 0.000113685 (W 0.000172206) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.094469 ± 0.000268615 (W 0.000371148) | 0.125382 ± 0.000798844 (W 0.00106725) | 🔵 faer-fork 1.33× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.101073 ± 0.000447606 (W 0.000519313) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.179698 ± 0.00117619 (W 0.00144845) | 0.30487 ± 0.00130238 (W 0.00171986) | 🔵 faer-fork 1.70× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.227314 ± 0.00117873 (W 0.00258146) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0172415 ± 0.000139978 (W 9.43611e-05) | 0.0141665 ± 0.00045039 (W 8.42725e-05) | 🔴 faer-fork 1.22× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067601 ± 0.000410479 (W 0.000235172) | 0.056753 ± 0.000547907 (W 0.000184115) | 🔴 faer-fork 1.19× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.130904 ± 0.000824726 (W 0.000336161) | 0.16483 ± 0.00059734 (W 0.000909448) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.006422 ± 5.80129e-05 (W 8.91039e-05) | 0.0076785 ± 0.000125562 (W 0.000101729) | 🔵 faer-fork 1.20× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00808 ± 9.53764e-05 (W 6.75845e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.022992 ± 8.88446e-05 (W 0.000151949) | 0.029455 ± 0.000314267 (W 8.44281e-05) | 🔵 faer-fork 1.28× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.025803 ± 0.000236457 (W 9.49233e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.047279 ± 0.000380339 (W 0.000215119) | 0.0698705 ± 0.000429981 (W 0.000105267) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.0658145 ± 0.000948932 (W 0.00214062) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.171559 ± 0.00114821 (W 0.000659838) | 0.178779 ± 0.00110098 (W 0.00126869) | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.172534 ± 0.00139875 (W 0.000244386) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.554997 ± 0.00265393 (W 0.00159615) | 0.660976 ± 0.00270086 (W 0.00183178) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.556512 ± 0.0161615 (W 0.00180882) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.156 ± 0.00620675 (W 0.00246999) | 1.57636 ± 0.00357182 (W 0.00330657) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.16979 ± 0.00384714 (W 0.00214319) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0594005 ± 0.000445173 (W 0.000124215) | 0.0697945 ± 0.000629765 (W 0.000154536) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.0598765 ± 0.000373373 (W 0.000147434) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.218918 ± 0.000824369 (W 0.00145759) | 0.263118 ± 0.00164754 (W 0.00168362) | 🔵 faer-fork 1.20× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.218563 ± 0.000852546 (W 0.00114467) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.544675 ± 0.00164878 (W 0.00156103) | 0.65943 ± 0.00384642 (W 0.00257246) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.542384 ± 0.00260927 (W 0.00157553) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000967 ± 1.94079e-05 (W 4.20546e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002349 ± 3.15621e-05 (W 4.16952e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0040075 ± 2.75729e-05 (W 3.96623e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0062415 ± 0.000141169 (W 3.19561e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0052755 ± 3.30681e-05 (W 6.1045e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.033127 ± 0.000167554 (W 0.00142916) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0314985 ± 0.000199532 (W 0.000132726) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.089495 ± 0.000586016 (W 0.000845422) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.086795 ± 0.000506507 (W 0.000196902) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.141439 ± 0.000591576 (W 0.00112489) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.142165 ± 0.00099926 (W 0.00148366) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.732951 ± 0.00348925 (W 0.00278549) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.737972 ± 0.00267811 (W 0.00219261) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.79284 ± 0.00510458 (W 0.0038519) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.80529 ± 0.00592286 (W 0.00204954) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0169065 ± 6.71347e-05 (W 9.98811e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.020518 ± 0.000138624 (W 0.000113193) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.0960285 ± 0.0059923 (W 0.000826317) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.107601 ± 0.000676595 (W 0.00143574) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.286146 ± 0.0177949 (W 0.00132773) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.348256 ± 0.00118548 (W 0.0022257) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028484 ± 0.000155482 (W 0.000137087) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028363 ± 0.000127236 (W 6.05591e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.160617 ± 0.000799795 (W 0.000839987) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.159747 ± 0.000619151 (W 0.00098642) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.47099 ± 0.00291673 (W 0.000779097) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.470594 ± 0.00225135 (W 0.00105261) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0125035 ± 0.000114718 (W 7.55157e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0147175 ± 0.000104739 (W 0.000151372) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0534645 ± 0.000221485 (W 0.00022596) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0618955 ± 0.000595747 (W 0.0003713) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.13991 ± 0.00078091 (W 0.0011186) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.207244 ± 0.00112511 (W 0.00176799) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0737275 ± 0.000605072 (W 0.00188669) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0687275 ± 0.000465899 (W 0.000337402) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.294111 ± 0.000918746 (W 0.001625) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.333007 ± 0.00204173 (W 0.00255381) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.711666 ± 0.00411719 (W 0.0018054) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.847517 ± 0.00312204 (W 0.00315657) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.040851 ± 0.000140682 (W 0.000252343) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.147479 ± 0.000501503 (W 0.00120821) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.349778 ± 0.00157548 (W 0.00174551) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0123835 ± 7.08447e-05 (W 5.90446e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01585 ± 0.00011098 (W 6.2945e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.051131 ± 0.000213815 (W 0.000741415) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0637735 ± 0.000288612 (W 9.54546e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.135672 ± 0.000846644 (W 0.00138087) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.208894 ± 0.000675544 (W 0.00264104) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.220087 ± 0.00197545 (W 0.00162217) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.221522 ± 0.00485358 (W 0.0017252) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 1.01058 ± 0.0042602 (W 0.00263055) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.01142 ± 0.00376493 (W 0.00259827) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.43889 ± 0.0159994 (W 0.00626455) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.44231 ± 0.00994358 (W 0.00651907) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.07516 ± 0.000246536 (W 0.00144689) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.074279 ± 0.000409757 (W 0.00025505) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.307885 ± 0.00112426 (W 0.00169126) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.309588 ± 0.00150551 (W 0.00126218) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.776412 ± 0.00558727 (W 0.00259939) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.779136 ± 0.00396952 (W 0.00162295) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0011725 ± 1.52403e-05 (W 3.32223e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002895 ± 2.15932e-05 (W 3.55183e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00529 ± 3.50238e-05 (W 5.60605e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.006101 ± 7.40281e-05 (W 3.70308e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005325 ± 2.56418e-05 (W 6.48755e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.030977 ± 0.000167754 (W 0.000175832) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.030111 ± 0.00013425 (W 0.000125663) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.083944 ± 0.00051173 (W 0.000215227) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.082773 ± 0.000407941 (W 0.000290012) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.10225 ± 0.000350944 (W 0.000320746) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.103416 ± 0.000461756 (W 0.000372637) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.434576 ± 0.00296599 (W 0.00169745) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.434085 ± 0.00104498 (W 0.00171523) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.918241 ± 0.00389346 (W 0.00174813) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.927203 ± 0.00292544 (W 0.00121642) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018204 ± 0.00113639 (W 0.000105442) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.02156 ± 0.000124724 (W 0.000119644) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.097051 ± 0.000854773 (W 0.000649972) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.108105 ± 0.00112231 (W 0.000464128) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.23318 ± 0.00296304 (W 0.00215403) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.304136 ± 0.00298432 (W 0.00283256) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.014707 ± 0.000175704 (W 0.000213255) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.014257 ± 0.00010034 (W 0.000162341) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0615645 ± 0.000573107 (W 0.00103182) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.061495 ± 0.000345242 (W 0.000437218) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.153356 ± 0.000484146 (W 0.00116717) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.15323 ± 0.000468228 (W 0.0015966) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0126035 ± 0.000191452 (W 8.80669e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0149275 ± 0.000170378 (W 9.26918e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0584455 ± 0.000606512 (W 0.000418299) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0621955 ± 0.000335255 (W 0.000574808) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.136713 ± 0.000708216 (W 0.00128958) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.188473 ± 0.00125896 (W 0.0017591) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0709165 ± 0.000308219 (W 0.00045935) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0635255 ± 0.00033012 (W 0.000385595) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.190796 ± 0.00110438 (W 0.0013209) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.219344 ± 0.00124099 (W 0.00241181) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.414282 ± 0.00203419 (W 0.00183511) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.518467 ± 0.00553539 (W 0.0042914) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0425395 ± 0.000250156 (W 0.000596734) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.126561 ± 0.000401787 (W 0.000426285) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.269577 ± 0.00157082 (W 0.00164089) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013149 ± 0.000124484 (W 7.53169e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016661 ± 9.35102e-05 (W 6.26251e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0542255 ± 0.00063687 (W 0.000476346) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0654865 ± 0.000372736 (W 0.000415254) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.12058 ± 0.000511413 (W 0.00102169) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.189773 ± 0.00111688 (W 0.00343521) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.21394 ± 0.00154595 (W 0.00126999) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.215979 ± 0.000936875 (W 0.00125233) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.733004 ± 0.00272829 (W 0.00217948) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.741102 ± 0.00195218 (W 0.00300459) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54043 ± 0.00521828 (W 0.00288853) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55055 ± 0.00525644 (W 0.00343805) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0766045 ± 0.000513448 (W 0.000153542) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.075423 ± 0.000460586 (W 0.000203399) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.311742 ± 0.00206742 (W 0.00138786) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.312613 ± 0.00179702 (W 0.000928567) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.710221 ± 0.00240029 (W 0.00334106) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.711588 ± 0.00369853 (W 0.00262891) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0014375 ± 2.35853e-05 (W 3.49235e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003446 ± 1.29216e-05 (W 3.80565e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0061415 ± 6.1027e-05 (W 4.17461e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.006011 ± 9.84488e-05 (W 5.48996e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005305 ± 5.87211e-05 (W 7.67955e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0309685 ± 0.000133121 (W 0.000656807) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0300405 ± 0.000187188 (W 0.000132307) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.083711 ± 0.000418222 (W 0.000294426) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0830735 ± 0.000355824 (W 0.000302823) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0974365 ± 0.000526654 (W 0.000547752) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.09873 ± 0.00336354 (W 0.000395117) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.320626 ± 0.00650011 (W 0.00676933) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.327458 ± 0.00617659 (W 0.00711995) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.680454 ± 0.00262389 (W 0.00380759) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.685716 ± 0.00410052 (W 0.00286209) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018299 ± 0.000219064 (W 8.9156e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.02139 ± 0.000169672 (W 0.000132884) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.10316 ± 0.000455695 (W 0.000667157) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.111582 ± 0.000817747 (W 0.00049608) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.232905 ± 0.00316108 (W 0.00192354) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.307696 ± 0.00284365 (W 0.00260266) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0106545 ± 0.000122693 (W 0.000149165) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0106645 ± 0.000111652 (W 0.000131294) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0372555 ± 0.000149863 (W 0.000350364) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0374655 ± 0.000309256 (W 0.000432003) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0871985 ± 0.000477896 (W 0.000831652) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0874695 ± 0.000374446 (W 0.000684415) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012739 ± 4.98197e-05 (W 0.000104059) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.014923 ± 8.20522e-05 (W 0.00018204) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0634495 ± 0.000317828 (W 0.00064564) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.061559 ± 0.000211298 (W 0.000285594) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.14525 ± 0.000951306 (W 0.00149386) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.186414 ± 0.000755885 (W 0.00136355) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.070213 ± 0.000796695 (W 0.000542972) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0632545 ± 0.000823924 (W 0.000473856) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.181105 ± 0.000711369 (W 0.00140574) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.20987 ± 0.00171265 (W 0.00302985) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.34849 ± 0.000762447 (W 0.00211829) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.472141 ± 0.00393026 (W 0.00369409) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.043217 ± 0.000445384 (W 0.000793647) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.122172 ± 0.000488195 (W 0.000720951) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.243204 ± 0.0379011 (W 0.00191135) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.0131545 ± 5.17365e-05 (W 6.0681e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0166115 ± 5.62056e-05 (W 8.97193e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0558005 ± 0.000224871 (W 0.00033872) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0652435 ± 0.000499928 (W 0.000177182) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.12613 ± 0.000681158 (W 0.000963916) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.187899 ± 0.00269079 (W 0.00336398) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.214857 ± 0.00908474 (W 0.00123516) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.216995 ± 0.0390703 (W 0.00150392) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.719544 ± 0.00322853 (W 0.00246643) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.725413 ± 0.010492 (W 0.00172459) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.47847 ± 0.00579549 (W 0.0045424) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.49323 ± 0.00706152 (W 0.00393624) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075556 ± 0.000307988 (W 0.000663453) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0755925 ± 0.000497802 (W 0.00018402) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.319628 ± 0.00213406 (W 0.00157379) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.322452 ± 0.000592471 (W 0.0012607) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.735322 ± 0.00520939 (W 0.0022584) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.737338 ± 0.00226705 (W 0.00198858) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0014475 ± 0.000230379 (W 2.84344e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003456 ± 1.59677e-05 (W 3.8989e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0061165 ± 4.26493e-05 (W 4.20364e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004604 ± 2.12955e-05 (W 2.71645e-05) | 0.004909 ± 0.000100659 (W 6.50553e-05) | 🔵 faer-fork 1.07× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0053145 ± 4.49463e-05 (W 6.00277e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0281375 ± 0.000135228 (W 8.97131e-05) | 0.027858 ± 0.000107129 (W 9.94698e-05) | · |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.0311725 ± 0.000278606 (W 0.000106319) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0798335 ± 0.000501559 (W 0.000122075) | 0.0783905 ± 0.000416911 (W 0.000196411) | · |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.0855995 ± 0.000423735 (W 0.000739194) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.141353 ± 0.000852799 (W 0.00134665) | 0.15401 ± 0.000738515 (W 0.000594804) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.142248 ± 0.000817196 (W 0.00135301) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.73509 ± 0.00317832 (W 0.0025757) | 0.667103 ± 0.00340106 (W 0.00205329) | 🔴 faer-fork 1.10× slower than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.741237 ± 0.00562127 (W 0.00196177) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.7953 ± 0.00688374 (W 0.00374366) | 1.72521 ± 0.00763848 (W 0.00299539) | · |  |
| `eigh/n96/b1 [api-inclusive]` | 1.79632 ± 0.00888547 (W 0.00286994) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0168565 ± 6.07972e-05 (W 0.000102447) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018755 ± 4.58712e-05 (W 8.74405e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0958435 ± 0.000496239 (W 0.000398514) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.100812 ± 0.000511541 (W 0.000475138) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.286839 ± 0.000900716 (W 0.00201602) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.335019 ± 0.00223934 (W 0.00258381) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028378 ± 8.94392e-05 (W 9.58858e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028459 ± 0.000144848 (W 8.96583e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.158515 ± 0.000536331 (W 0.00170043) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.158627 ± 0.000530068 (W 0.00124684) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.465004 ± 0.00283661 (W 0.00190598) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.463686 ± 0.0027062 (W 0.00153295) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011527 ± 0.000134002 (W 7.94941e-05) | 0.0112715 ± 6.80118e-05 (W 0.000106098) | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013315 ± 0.000119401 (W 0.000129051) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0504155 ± 0.000312324 (W 0.000267088) | 0.0551125 ± 0.000187178 (W 0.000185392) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0549335 ± 0.000358128 (W 0.000220053) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.135827 ± 0.000539665 (W 0.00106907) | 0.153386 ± 0.000740122 (W 0.00105503) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.194236 ± 0.00155289 (W 0.00229971) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.064192 ± 0.000189368 (W 0.000323951) | 0.0569305 ± 0.000418029 (W 0.000172849) | 🔴 faer-fork 1.13× slower than `lapack-openblas` |  |
| `qr/n32/b1 [api-inclusive]` | 0.0670075 ± 0.000273476 (W 0.00113142) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.297244 ± 0.000920526 (W 0.00197003) | 0.304907 ± 0.00191812 (W 0.00181241) | · |  |
| `qr/n64/b1 [api-inclusive]` | 0.326445 ± 0.00181623 (W 0.00333045) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.733268 ± 0.00211255 (W 0.00325561) | 0.812741 ± 0.00311045 (W 0.00188048) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.834362 ± 0.00241767 (W 0.00371068) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.032356 ± 0.000244154 (W 0.000229674) | 0.0253375 ± 0.000218498 (W 0.000585747) | 🔴 faer-fork 1.28× slower than `lapack-openblas` |  |
| `qr_factor/n64/b1` | 0.152985 ± 0.000926465 (W 0.00173501) | 0.121316 ± 0.000795968 (W 0.00119631) | 🔴 faer-fork 1.26× slower than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.372124 ± 0.00169999 (W 0.00158368) | 0.337983 ± 0.00165157 (W 0.00148731) | 🔴 faer-fork 1.10× slower than `lapack-openblas` |  |
| `solve/n32/b1` | 0.012012 ± 5.16204e-05 (W 3.93408e-05) | 0.012448 ± 5.02464e-05 (W 7.98419e-05) | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014051 ± 0.000108488 (W 0.000141027) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.052227 ± 0.000408566 (W 0.000375905) | 0.0585365 ± 0.000262227 (W 0.000192543) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.057407 ± 0.000271694 (W 0.000841535) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.140295 ± 0.000839352 (W 0.00174989) | 0.160955 ± 0.000580526 (W 0.00166543) | 🔵 faer-fork 1.15× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.197886 ± 0.00229641 (W 0.00267039) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.217631 ± 0.00139017 (W 0.00171202) | 0.22987 ± 0.00137306 (W 0.0014528) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.218873 ± 0.0010851 (W 0.00145222) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.997753 ± 0.00637525 (W 0.00102286) | 0.999912 ± 0.00492654 (W 0.0019771) | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 1.00214 ± 0.00455331 (W 0.00176692) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.42979 ± 0.00872954 (W 0.00523491) | 2.83185 ± 0.00825821 (W 0.00433174) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.43341 ± 0.00961632 (W 0.00356656) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075932 ± 0.000315705 (W 0.000258933) | 0.0995245 ± 0.000556559 (W 0.000835858) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.07538 ± 0.000433361 (W 0.000585002) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.304411 ± 0.00160379 (W 0.00180021) | 0.413057 ± 0.00356972 (W 0.00182558) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.306081 ± 0.00131269 (W 0.00160777) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.771929 ± 0.00366174 (W 0.00238333) | 1.12991 ± 0.00574929 (W 0.00268965) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.775467 ± 0.00511602 (W 0.00188949) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001172 ± 5.16398e-06 (W 2.96917e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00274 ± 9.83192e-06 (W 3.76864e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0048995 ± 3.56969e-05 (W 5.28368e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0045585 ± 0.000124756 (W 2.57006e-05) | 0.0049145 ± 8.30036e-05 (W 4.98321e-05) | 🔵 faer-fork 1.08× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005345 ± 2.86054e-05 (W 6.67748e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.026499 ± 0.000114003 (W 0.000104268) | 0.026781 ± 0.000223878 (W 0.000510365) | · |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.029816 ± 0.000194364 (W 0.000107768) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0749265 ± 0.000463134 (W 0.000350135) | 0.0774595 ± 0.000590463 (W 0.000882532) | · |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.081727 ± 0.000386025 (W 0.000301828) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.102248 ± 0.000401034 (W 0.000458425) | 0.141589 ± 0.000312109 (W 0.00115979) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.103693 ± 0.000970983 (W 0.000452717) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.433667 ± 0.00230706 (W 0.00225254) | 0.588148 ± 0.00228934 (W 0.00162095) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.434998 ± 0.00226489 (W 0.00183418) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.920886 ± 0.00194715 (W 0.00189714) | 1.6824 ± 0.00448347 (W 0.00335587) | 🔵 faer-fork 1.83× faster than `lapack-openblas` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.927273 ± 0.00259208 (W 0.00190076) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0180245 ± 0.00011177 (W 0.000106708) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0198015 ± 5.36196e-05 (W 0.000119297) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.096585 ± 0.00184238 (W 0.000397676) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.100937 ± 0.000763633 (W 0.0004987) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.229919 ± 0.00573572 (W 0.00197075) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.288216 ± 0.00300622 (W 0.00159069) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.014898 ± 0.000160713 (W 0.00014777) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.014763 ± 0.000116019 (W 0.000181201) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.061847 ± 0.000307886 (W 0.000286206) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.061344 ± 0.000295805 (W 0.000232217) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.150105 ± 0.000828445 (W 0.0011301) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.1507 ± 0.00045935 (W 0.00061211) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0115665 ± 8.8495e-05 (W 0.000102682) | 0.011256 ± 0.000125817 (W 0.000134318) | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.01343 ± 0.000127186 (W 0.000118768) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0496195 ± 0.000236306 (W 0.000569831) | 0.0552425 ± 0.000339201 (W 0.000846808) | 🔵 faer-fork 1.11× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0553635 ± 0.000637507 (W 0.000507037) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.110704 ± 0.00122887 (W 0.000545602) | 0.153605 ± 0.00060975 (W 0.00122526) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.173923 ± 0.00113446 (W 0.00172197) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0579595 ± 0.000282561 (W 0.000399183) | 0.056946 ± 0.000235468 (W 0.000105879) | · |  |
| `qr/n32/b1 [api-inclusive]` | 0.061465 ± 0.000408425 (W 0.000344953) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.176078 ± 0.00099647 (W 0.00160776) | 0.255846 ± 0.0010923 (W 0.00207738) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.214214 ± 0.00287306 (W 0.00295286) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.382726 ± 0.00207642 (W 0.00176182) | 0.693838 ± 0.00299282 (W 0.00177921) | 🔵 faer-fork 1.81× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.501642 ± 0.00359589 (W 0.00306244) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.031014 ± 0.00019123 (W 0.000242552) | 0.025183 ± 0.000290889 (W 0.00076496) | 🔴 faer-fork 1.23× slower than `lapack-openblas` |  |
| `qr_factor/n64/b1` | 0.11626 ± 0.000457372 (W 0.000468325) | 0.121671 ± 0.00105875 (W 0.00130632) | · |  |
| `qr_factor/n96/b1` | 0.246532 ± 0.0011959 (W 0.00159896) | 0.436842 ± 0.00180008 (W 0.00189069) | 🔵 faer-fork 1.77× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.0127895 ± 0.000189446 (W 0.000111868) | 0.0126285 ± 9.09043e-05 (W 0.000101539) | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014733 ± 7.33251e-05 (W 8.66534e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0530405 ± 0.00124429 (W 0.000772928) | 0.0588015 ± 0.000389045 (W 0.00114195) | 🔵 faer-fork 1.11× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.059345 ± 0.000446339 (W 0.000286733) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.116021 ± 0.000345145 (W 0.000635295) | 0.160558 ± 0.000891086 (W 0.00162928) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.176254 ± 0.000432404 (W 0.00216822) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.213599 ± 0.00125271 (W 0.00131993) | 0.231241 ± 0.000849584 (W 0.00147622) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.215121 ± 0.00847006 (W 0.00106423) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.720141 ± 0.0035429 (W 0.00300984) | 0.999494 ± 0.00504154 (W 0.000982676) | 🔵 faer-fork 1.39× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.727719 ± 0.005649 (W 0.00207368) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54079 ± 0.00913 (W 0.00488699) | 2.76179 ± 0.00892774 (W 0.00592418) | 🔵 faer-fork 1.79× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.54854 ± 0.00564289 (W 0.00296092) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0772085 ± 0.000703448 (W 0.000115395) | 0.099322 ± 0.000522736 (W 0.000869093) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.076858 ± 0.000661581 (W 0.000152736) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.308025 ± 0.00118964 (W 0.00147938) | 0.415123 ± 0.00364972 (W 0.00205221) | 🔵 faer-fork 1.35× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.310759 ± 0.000972768 (W 0.00116126) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.706187 ± 0.00259538 (W 0.00202298) | 1.3741 ± 0.00335223 (W 0.00347066) | 🔵 faer-fork 1.95× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.712424 ± 0.00333633 (W 0.00202729) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001387 ± 1.18954e-05 (W 3.45465e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003261 ± 2.93513e-05 (W 4.13626e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0056905 ± 3.62808e-05 (W 5.6489e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0044585 ± 6.04662e-05 (W 1.81808e-05) | 0.004904 ± 4.84424e-05 (W 5.74551e-05) | 🔵 faer-fork 1.10× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00529 ± 5.48671e-05 (W 5.82348e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.026259 ± 0.000173561 (W 0.000156564) | 0.0358015 ± 0.000573783 (W 0.000603596) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.029785 ± 0.000176768 (W 0.000157652) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.074292 ± 0.000265631 (W 0.000261973) | 0.078422 ± 0.000769949 (W 0.00159734) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.082224 ± 0.000169255 (W 0.000319027) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.097387 ± 0.000437035 (W 0.000420611) | 0.140275 ± 0.000788749 (W 0.00109468) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.0983365 ± 0.000575782 (W 0.000364604) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.324145 ± 0.00621405 (W 0.00701687) | 0.572723 ± 0.00121056 (W 0.00204333) | 🔵 faer-fork 1.77× faster than `lapack-openblas` | `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.315514 ± 0.00694845 (W 0.0069613) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.685549 ± 0.015923 (W 0.0135082) | 1.69912 ± 0.00474444 (W 0.00260006) | 🔵 faer-fork 2.48× faster than `lapack-openblas` | `!` between-start range over 5% |
| `eigh/n96/b1 [api-inclusive]` | 0.683671 ± 0.00265422 (W 0.00806376) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0182635 ± 0.000123353 (W 8.18133e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0196925 ± 8.74018e-05 (W 0.000102417) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.101726 ± 0.00256725 (W 0.000602562) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.105188 ± 0.00124577 (W 0.000721523) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.235519 ± 0.000969317 (W 0.00155569) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.293395 ± 0.00335824 (W 0.00333406) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.011105 ± 0.00161637 (W 0.000254771) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.011116 ± 9.72331e-05 (W 0.000152098) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0381005 ± 0.000318545 (W 0.000389396) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0378605 ± 0.000257023 (W 0.000216977) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.086233 ± 0.00035206 (W 0.00108272) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0867165 ± 0.000347706 (W 0.00104696) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0116475 ± 0.000117595 (W 7.28207e-05) | 0.011236 ± 9.5229e-05 (W 0.000110724) | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.01345 ± 0.000120009 (W 0.000137357) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0490745 ± 0.000164273 (W 0.000267871) | 0.0548665 ± 0.00025539 (W 0.000139208) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.0551855 ± 0.000300469 (W 0.000305763) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.10875 ± 0.000618435 (W 0.000575186) | 0.153511 ± 0.000690271 (W 0.000333052) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.174028 ± 0.00176591 (W 0.00185701) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.057392 ± 0.000385956 (W 0.000415717) | 0.0573085 ± 0.000537574 (W 0.000903483) | · |  |
| `qr/n32/b1 [api-inclusive]` | 0.061254 ± 0.00389035 (W 0.000401607) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.162502 ± 0.000787236 (W 0.00187452) | 0.252285 ± 0.00175019 (W 0.0016099) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `qr/n64/b1 [api-inclusive]` | 0.203219 ± 0.00137504 (W 0.0023267) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.325472 ± 0.0010355 (W 0.00232122) | 0.709754 ± 0.00528436 (W 0.00222783) | 🔵 faer-fork 2.18× faster than `lapack-openblas` |  |
| `qr/n96/b1 [api-inclusive]` | 0.45512 ± 0.00367499 (W 0.00430923) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0307385 ± 0.000323178 (W 0.000359988) | 0.025032 ± 0.000356951 (W 6.687e-05) | 🔴 faer-fork 1.23× slower than `lapack-openblas` |  |
| `qr_factor/n64/b1` | 0.111841 ± 0.000459431 (W 0.000415356) | 0.120735 ± 0.000895924 (W 0.000162119) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `qr_factor/n96/b1` | 0.228096 ± 0.00129167 (W 0.00155539) | 0.470757 ± 0.00210392 (W 0.00191203) | 🔵 faer-fork 2.06× faster than `lapack-openblas` |  |
| `solve/n32/b1` | 0.0127035 ± 0.000100593 (W 7.05297e-05) | 0.0124825 ± 0.000116352 (W 7.8977e-05) | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014793 ± 9.08928e-05 (W 7.5798e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0524305 ± 0.000251423 (W 0.000318993) | 0.0587795 ± 0.00010441 (W 9.13829e-05) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.058946 ± 0.00042442 (W 0.000328551) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.111814 ± 0.000388013 (W 0.000388567) | 0.160332 ± 0.000464501 (W 0.0014019) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.174167 ± 0.000702344 (W 0.0034007) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.212523 ± 0.00134226 (W 0.00130523) | 0.232262 ± 0.00181006 (W 0.00187077) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.214785 ± 0.00257988 (W 0.00127015) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.710134 ± 0.00667152 (W 0.00335265) | 1.00838 ± 0.00540473 (W 0.00190139) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.712533 ± 0.00377859 (W 0.00241917) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.47855 ± 0.00712117 (W 0.00444991) | 2.83455 ± 0.010649 (W 0.00523223) | 🔵 faer-fork 1.92× faster than `lapack-openblas` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.49045 ± 0.00303523 (W 0.00512345) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.076662 ± 0.000343496 (W 0.000211034) | 0.0995555 ± 0.000590493 (W 0.00191611) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.076983 ± 0.000379258 (W 0.00012525) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.318115 ± 0.00104266 (W 0.00101667) | 0.412117 ± 0.00313856 (W 0.00200026) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.320512 ± 0.00110309 (W 0.0013767) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.730644 ± 0.0049898 (W 0.00194345) | 1.46645 ± 0.00818353 (W 0.00492953) | 🔵 faer-fork 2.01× faster than `lapack-openblas` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.733709 ± 0.00319954 (W 0.00178619) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001458 ± 3.0605e-05 (W 3.05213e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0034915 ± 2.17937e-05 (W 3.45568e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006021 ± 7.28579e-05 (W 6.31944e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003337 ± 8.70802e-05 (W 3.32836e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0026 ± 2.92301e-05 (W 5.32903e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0139205 ± 0.000429024 (W 4.11399e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010134 ± 3.98882e-05 (W 8.62485e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.026249 ± 0.0010669 (W 0.00015299) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.02113 ± 0.000867824 (W 8.95886e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.0866465 ± 0.00301929 (W 0.000839795) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.0871175 ± 0.00025364 (W 0.00121213) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.323377 ± 0.00148239 (W 0.00149337) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.324095 ± 0.00175453 (W 0.00164308) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.949329 ± 0.004467 (W 0.00354405) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.949029 ± 0.00419674 (W 0.0024918) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0100435 ± 0.000109119 (W 3.58436e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0130245 ± 0.000106133 (W 6.71861e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.040325 ± 0.000178992 (W 8.50673e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0486155 ± 0.000249712 (W 8.09375e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.098253 ± 0.000471966 (W 0.000264897) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.116556 ± 0.000406379 (W 0.00128357) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006472 ± 3.50714e-05 (W 0.000110286) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0064575 ± 3.66265e-05 (W 6.33193e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.026048 ± 0.000144508 (W 0.000111256) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0261535 ± 9.94257e-05 (W 0.000106911) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0770585 ± 0.000261974 (W 0.000177716) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0769885 ± 0.00055295 (W 0.000123454) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0063125 ± 5.04288e-05 (W 5.88803e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00811 ± 0.000106708 (W 9.06891e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020278 ± 0.000191211 (W 8.7965e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0260585 ± 0.000152964 (W 0.000189893) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.044184 ± 0.00015696 (W 0.000173234) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.057753 ± 0.00031085 (W 0.000298545) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.031103 ± 0.000146797 (W 6.8092e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.028303 ± 9.38822e-05 (W 0.000222864) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0902175 ± 0.000560742 (W 0.000591859) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.103432 ± 0.000732031 (W 0.000938302) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.192879 ± 0.00138912 (W 0.00122042) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.24206 ± 0.00113249 (W 0.0016069) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.021269 ± 0.000110661 (W 9.28706e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.05915 ± 0.00051514 (W 0.00013217) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.11268 ± 0.000813329 (W 0.000218419) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0073635 ± 6.28885e-05 (W 5.17373e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0098235 ± 0.000101982 (W 4.60426e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020233 ± 0.000228034 (W 6.89457e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.028173 ± 0.000234104 (W 8.2165e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0413535 ± 0.000241676 (W 0.000152921) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.058085 ± 0.000352226 (W 0.000736535) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.125674 ± 0.00113205 (W 0.00119747) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.126565 ± 0.00859271 (W 0.00138844) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.529181 ± 0.00231609 (W 0.00200835) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.529323 ± 0.0025937 (W 0.00170342) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.20435 ± 0.00379977 (W 0.00378245) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.20568 ± 0.00568263 (W 0.00243091) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0474005 ± 0.000406838 (W 0.000142287) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.047941 ± 0.00035686 (W 0.000140031) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.169625 ± 0.000939219 (W 0.00134) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.169902 ± 0.000941331 (W 0.00142287) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.369966 ± 0.000452203 (W 0.00182619) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.37004 ± 0.00185511 (W 0.00192714) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001017 ± 1.47196e-05 (W 3.98375e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002189 ± 3.65605e-05 (W 3.10046e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0035765 ± 3.35002e-05 (W 5.66719e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003156 ± 0.000232147 (W 2.84081e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002655 ± 2.31661e-05 (W 3.38678e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0134355 ± 0.001173 (W 5.27499e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010079 ± 6.25033e-05 (W 8.04327e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0265455 ± 0.00116559 (W 0.000272905) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.0217155 ± 0.00102074 (W 9.60278e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.0641015 ± 0.000213952 (W 0.000176313) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.064676 ± 0.000173258 (W 0.000171262) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.253763 ± 0.00059683 (W 0.00164706) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.253372 ± 0.000630364 (W 0.00120607) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.546185 ± 0.00171248 (W 0.00295294) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.56046 ± 0.00358484 (W 0.0037573) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011266 ± 9.37737e-05 (W 4.06909e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014127 ± 8.45421e-05 (W 8.42205e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0426255 ± 0.000520747 (W 6.11467e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.051155 ± 0.000131851 (W 0.000162915) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.104023 ± 0.00054458 (W 0.000382171) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.120878 ± 0.000476722 (W 0.000155743) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006222 ± 0.000258595 (W 0.000455307) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0053345 ± 0.00015125 (W 0.000155264) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.015724 ± 0.0002573 (W 0.000528564) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014131 ± 0.000273551 (W 0.000334976) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.034174 ± 0.00075157 (W 0.000592919) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0330565 ± 0.000561129 (W 0.000639093) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006482 ± 9.85901e-05 (W 6.75526e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008215 ± 7.50744e-05 (W 9.98405e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020483 ± 0.000146754 (W 0.000118147) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.026424 ± 0.000198071 (W 0.000243337) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0548925 ± 0.000397633 (W 0.00145752) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.055884 ± 0.000261043 (W 0.000378813) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.032191 ± 0.000150262 (W 7.67168e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.029701 ± 0.000155948 (W 0.000112415) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.104383 ± 0.00119182 (W 0.0010557) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.102471 ± 0.000542599 (W 0.00033331) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.206771 ± 0.00258045 (W 0.00266612) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.198209 ± 0.000524416 (W 0.00145438) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.022082 ± 0.000298659 (W 8.22313e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.070782 ± 0.00044606 (W 0.000177247) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.135563 ± 0.00157571 (W 0.000542894) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007789 ± 4.10727e-05 (W 4.12303e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010364 ± 0.000125604 (W 5.66721e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0213145 ± 0.000150392 (W 0.000110554) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.029405 ± 0.000131015 (W 8.55115e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0478155 ± 0.00043503 (W 0.000661624) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0591855 ± 0.00058762 (W 0.000534341) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.127822 ± 0.00197563 (W 0.000236268) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.129144 ± 0.00136393 (W 0.000337495) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.442226 ± 0.00706848 (W 0.00169659) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.45579 ± 0.00095438 (W 0.00323292) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.880833 ± 0.00302161 (W 0.00323069) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.882578 ± 0.00389111 (W 0.0036978) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.04833 ± 0.000358595 (W 0.000130994) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0491465 ± 0.000211992 (W 0.00019148) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.171247 ± 0.000661909 (W 0.00141184) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.172761 ± 0.000899553 (W 0.00120895) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.381137 ± 0.00193561 (W 0.00154448) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.381467 ± 0.00195929 (W 0.00180451) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001247 ± 1.60924e-05 (W 4.24678e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00265 ± 2.04124e-05 (W 3.53372e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004268 ± 3.38625e-05 (W 4.50962e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003266 ± 7.70498e-05 (W 3.40375e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002585 ± 1.62532e-05 (W 4.219e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0138005 ± 0.000538337 (W 5.58696e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010089 ± 5.60372e-05 (W 9.48359e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0258375 ± 0.00104988 (W 0.000101548) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.0218455 ± 0.00103394 (W 7.35121e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.061211 ± 0.000321539 (W 0.00016833) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0620905 ± 0.00274585 (W 0.000221785) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.199408 ± 0.00120299 (W 0.00166092) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.200744 ± 0.00190795 (W 0.00153795) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.417146 ± 0.0059733 (W 0.00340611) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.417396 ± 0.00599605 (W 0.00207448) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0113055 ± 0.00014938 (W 4.04383e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0143115 ± 0.00025358 (W 8.29362e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0432355 ± 0.000515049 (W 9.05343e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0509495 ± 0.0002352 (W 0.00017799) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.106595 ± 0.000732286 (W 0.000233929) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.121281 ± 0.000831999 (W 0.000244262) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006272 ± 0.000359544 (W 0.000424597) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0062665 ± 0.00021458 (W 0.000384269) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.012673 ± 0.000359062 (W 0.000289328) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0127735 ± 0.000485526 (W 0.000275711) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.025327 ± 0.000635203 (W 0.000432303) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.024877 ± 0.000426518 (W 0.000668524) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0064215 ± 7.11911e-05 (W 4.43235e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0083055 ± 5.52051e-05 (W 8.68516e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0205175 ± 0.000158305 (W 0.000106559) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0263695 ± 0.000189094 (W 0.000168534) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.062649 ± 0.00101404 (W 0.00169443) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.055899 ± 0.000161563 (W 0.000354966) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0324205 ± 0.000143534 (W 7.53987e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.029876 ± 0.000560901 (W 0.000151875) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.105373 ± 0.0011578 (W 0.00118267) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.101408 ± 0.000497993 (W 0.000268443) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.195677 ± 0.00110873 (W 0.00171818) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.195274 ± 0.000879019 (W 0.00160558) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0221455 ± 0.000157431 (W 7.08325e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.070848 ± 0.000337504 (W 0.000234054) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.12983 ± 0.0011028 (W 0.000369911) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00786 ± 6.12658e-05 (W 6.22083e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010449 ± 5.85001e-05 (W 6.36169e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0213645 ± 0.00014789 (W 6.26986e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.02953 ± 0.000102596 (W 8.36349e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0528695 ± 0.000930115 (W 0.00158086) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0584535 ± 0.000359757 (W 0.000741816) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.127941 ± 0.000813465 (W 0.000356807) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.12895 ± 0.000225389 (W 0.000185763) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.440366 ± 0.00977264 (W 0.00254365) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.458537 ± 0.00991503 (W 0.00415079) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 0.874675 ± 0.00310452 (W 0.00403586) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.878061 ± 0.00396802 (W 0.00454855) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0484825 ± 0.00027193 (W 0.000149181) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0489355 ± 0.000369851 (W 0.000166921) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.171609 ± 0.00075607 (W 0.0015513) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.172976 ± 0.000774007 (W 0.000681442) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.383429 ± 0.00118735 (W 0.00177202) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.384024 ± 0.00216873 (W 0.0019028) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001262 ± 1.32275e-05 (W 3.61253e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0026845 ± 1.9395e-05 (W 5.1663e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0043725 ± 3.93467e-05 (W 5.28961e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0019335 ± 0.000122606 (W 3.23913e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0026095 ± 2.9936e-05 (W 4.91217e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008491 ± 3.80022e-05 (W 2.31019e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010279 ± 4.89643e-05 (W 5.75246e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0179485 ± 0.000720408 (W 6.07061e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0214855 ± 0.000869717 (W 8.49852e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.086857 ± 0.000684935 (W 0.00022025) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0872635 ± 0.000697891 (W 0.00119474) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.324055 ± 0.0013059 (W 0.00194312) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.325184 ± 0.0010993 (W 0.00177815) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.946683 ± 0.00569253 (W 0.00224294) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.955269 ± 0.00437376 (W 0.00181183) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.010144 ± 0.000119503 (W 4.70096e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.011311 ± 5.10833e-05 (W 4.63782e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.039951 ± 0.000157664 (W 7.64405e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0421755 ± 0.000397308 (W 0.000129592) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.0978915 ± 0.000329805 (W 0.000701611) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.101539 ± 0.000435724 (W 0.00139537) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.007018 ± 4.03733e-05 (W 7.19108e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.007038 ± 5.89251e-05 (W 3.51699e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0275555 ± 0.000136077 (W 0.000107486) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0273565 ± 0.000186581 (W 0.000126751) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.082201 ± 0.000446594 (W 0.000257224) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0820405 ± 0.000536054 (W 0.000762062) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.004944 ± 4.90051e-05 (W 5.17183e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0062365 ± 5.50542e-05 (W 5.79408e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017548 ± 0.000247217 (W 0.000166537) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0197615 ± 0.000254685 (W 0.000186764) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.038511 ± 0.000222776 (W 0.000195234) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.042223 ± 0.000399213 (W 0.000363689) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.024305 ± 0.00016744 (W 0.000105496) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.026599 ± 0.000217575 (W 8.29897e-05) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0923505 ± 0.000550599 (W 0.00141292) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.0966505 ± 0.000496926 (W 0.00120995) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.220068 ± 0.001652 (W 0.0013946) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.22608 ± 0.00194257 (W 0.00160593) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0149435 ± 0.00014193 (W 0.000102205) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0618655 ± 0.000364189 (W 0.000235835) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.130648 ± 0.00119682 (W 0.00106703) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0059815 ± 4.12977e-05 (W 5.18671e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0074735 ± 3.44117e-05 (W 4.56168e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.018304 ± 0.000175097 (W 6.25191e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0208185 ± 0.000111501 (W 7.53676e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.039189 ± 0.000452682 (W 0.000232438) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.042205 ± 0.000179421 (W 0.000145181) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.134651 ± 0.00155936 (W 0.00149734) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.136258 ± 0.0011926 (W 0.000791867) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.530968 ± 0.00436604 (W 0.0018034) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.533384 ± 0.00223742 (W 0.00227231) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.20799 ± 0.00493358 (W 0.00206743) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.2089 ± 0.0191085 (W 0.00327619) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.04841 ± 0.00011553 (W 0.000699298) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.04847 ± 0.000222354 (W 0.000159682) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.174896 ± 0.00124849 (W 0.0016509) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.174992 ± 0.00116309 (W 0.00124802) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.373697 ± 0.00141361 (W 0.00174602) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.373762 ± 0.00275157 (W 0.00172252) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000806 ± 1.16175e-05 (W 3.59718e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001729 ± 1.70958e-05 (W 3.06036e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00272 ± 1.21106e-05 (W 4.51802e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001969 ± 3.40612e-05 (W 2.65796e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002675 ± 0.000246531 (W 5.61689e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008481 ± 3.0605e-05 (W 1.72085e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010229 ± 5.18089e-05 (W 5.87188e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.018339 ± 0.00101933 (W 5.00617e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.022046 ± 0.00026598 (W 8.20044e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0642145 ± 0.000265599 (W 0.000190401) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.064836 ± 0.000133842 (W 0.000225498) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.252293 ± 0.000831419 (W 0.0016217) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.253329 ± 0.00078284 (W 0.000929566) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.547451 ± 0.00217375 (W 0.00266855) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.560836 ± 0.00201884 (W 0.00407441) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0112255 ± 4.23824e-05 (W 4.14853e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012353 ± 0.000123457 (W 6.24502e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.042294 ± 0.000215428 (W 7.98182e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.044529 ± 0.000401673 (W 0.000143165) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.103371 ± 0.00064331 (W 0.000212589) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.106732 ± 0.000752616 (W 0.00023255) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005275 ± 0.000150081 (W 0.000197399) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.004954 ± 7.05334e-05 (W 9.42244e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0138915 ± 0.000259462 (W 0.000293687) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.013781 ± 9.44678e-05 (W 0.000304886) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.030588 ± 0.00029832 (W 0.000290651) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.030363 ± 0.000290518 (W 0.000229594) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.004969 ± 5.30723e-05 (W 5.20635e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0062315 ± 7.71684e-05 (W 7.67567e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017688 ± 0.000170365 (W 0.000159859) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.019967 ± 0.000196761 (W 0.000197783) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.03749 ± 0.000440148 (W 0.000394902) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.040871 ± 0.00025099 (W 0.000491675) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0256035 ± 0.000103121 (W 9.27977e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.027892 ± 0.000189013 (W 0.000123978) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0903175 ± 0.000392902 (W 0.00031461) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.0948245 ± 0.000518008 (W 0.000265257) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.172528 ± 0.00114899 (W 0.00114852) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.180271 ± 0.00158929 (W 0.00135399) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0158645 ± 0.000152183 (W 0.000120933) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0632645 ± 0.000354628 (W 0.000230718) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.119986 ± 0.00116188 (W 0.000323785) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006392 ± 5.18292e-05 (W 7.20461e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008015 ± 0.000199219 (W 0.000102717) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0192665 ± 7.6662e-05 (W 9.86807e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0218665 ± 0.000204462 (W 0.000120286) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.040236 ± 0.000224519 (W 0.000330489) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0429365 ± 0.000257044 (W 0.000490724) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.137937 ± 0.00151724 (W 0.000264425) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.137706 ± 0.000379949 (W 0.000278673) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.437327 ± 0.00139585 (W 0.00195467) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.451964 ± 0.00284008 (W 0.00267653) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.884912 ± 0.00218467 (W 0.00130736) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.889482 ± 0.00296868 (W 0.00254592) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0494035 ± 0.000405958 (W 0.000110022) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.050113 ± 0.000270551 (W 0.000140099) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.175336 ± 0.000785166 (W 0.00141469) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.177509 ± 0.000595523 (W 0.00127601) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.385211 ± 0.00149898 (W 0.00194467) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.386505 ± 0.00108515 (W 0.00113504) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000997 ± 2.58199e-05 (W 4.12371e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002144 ± 2.85003e-05 (W 3.56094e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0033265 ± 4.75132e-05 (W 5.10499e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0019585 ± 1.75955e-05 (W 1.93292e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0025945 ± 1.76711e-05 (W 6.10251e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00834 ± 0.000917536 (W 3.2452e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0102385 ± 1.45968e-05 (W 0.000106868) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.017182 ± 0.00127139 (W 4.8297e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.021965 ± 0.000229255 (W 0.000132546) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.060848 ± 0.00036791 (W 0.000186419) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0618505 ± 0.000457652 (W 0.000219296) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.199162 ± 0.000790721 (W 0.00113088) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.200555 ± 0.000993102 (W 0.0014593) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.416625 ± 0.00278071 (W 0.00226857) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.421446 ± 0.00934925 (W 0.00258949) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0112705 ± 0.000176619 (W 3.82248e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0123725 ± 7.66316e-05 (W 0.000102074) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0432255 ± 0.000196663 (W 7.5942e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0444885 ± 0.000454533 (W 0.000124468) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.106012 ± 0.000573972 (W 0.000200129) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.106654 ± 0.000561412 (W 0.000213459) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00528 ± 7.52879e-05 (W 0.00018915) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0052195 ± 0.000107152 (W 0.000151099) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.011632 ± 0.0002604 (W 0.000382142) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.012098 ± 0.000296412 (W 0.000583404) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0219455 ± 0.000174383 (W 0.000314784) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0219615 ± 0.000191951 (W 0.000225088) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.004989 ± 8.34684e-05 (W 6.9356e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0063415 ± 8.55547e-05 (W 0.000116097) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017728 ± 0.000205148 (W 0.000183421) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.020021 ± 0.00014883 (W 0.000248186) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0373735 ± 0.000262048 (W 0.000359094) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.041102 ± 0.000351938 (W 0.000523597) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.025623 ± 0.000250856 (W 0.000115895) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0280575 ± 0.000188685 (W 0.000125447) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0905595 ± 0.000715276 (W 0.000428276) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.094817 ± 0.0168331 (W 0.000423888) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.170039 ± 0.00158259 (W 0.00116923) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.177607 ± 0.00135646 (W 0.000652978) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0159395 ± 0.000311604 (W 0.000139829) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0632365 ± 0.000470376 (W 0.000292065) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.117945 ± 0.00109858 (W 0.000547567) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0064165 ± 8.36915e-05 (W 5.26374e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008035 ± 7.60465e-05 (W 4.78498e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0193145 ± 0.000154205 (W 0.000104009) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.022036 ± 7.51691e-05 (W 0.000125947) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.039418 ± 0.000547359 (W 0.000373084) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0428195 ± 0.000605475 (W 0.000701484) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.136783 ± 0.0012484 (W 0.000223567) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.138149 ± 0.00177285 (W 0.000197192) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.432644 ± 0.00309964 (W 0.00211882) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.451428 ± 0.00344877 (W 0.00263002) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.882734 ± 0.00402721 (W 0.0027662) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.885385 ± 0.00493396 (W 0.00326308) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.049352 ± 0.00031639 (W 0.000117503) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.049963 ± 0.000382731 (W 0.000140162) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.175359 ± 0.00124837 (W 0.00162208) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.176639 ± 0.000807308 (W 0.0011889) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.388904 ± 0.00181393 (W 0.00197115) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.389011 ± 0.00161017 (W 0.0013665) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001012 ± 2.94958e-05 (W 3.27122e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002144 ± 3.02903e-05 (W 2.82979e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004143 ± 3.09219e-05 (W 4.00786e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012799 ± 4.38072e-05 (W 6.65315e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0120875 ± 7.2963e-05 (W 5.35897e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0705165 ± 0.000384762 (W 0.000306041) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066364 ± 0.000342302 (W 0.00127819) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.115029 ± 0.000653418 (W 0.00162838) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.109505 ± 0.000589832 (W 0.000587497) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.103914 ± 0.000606554 (W 0.00134109) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.104653 ± 0.00148029 (W 0.000871364) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.407724 ± 0.0238602 (W 0.00186836) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.41171 ± 0.00191485 (W 0.00205768) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.17689 ± 0.00470101 (W 0.00167785) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.1708 ± 0.0050605 (W 0.00104266) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0135805 ± 6.85879e-05 (W 4.55166e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.017352 ± 0.000135333 (W 6.9258e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0637635 ± 0.000285927 (W 0.000115964) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0752505 ± 0.00507201 (W 0.00113344) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.17657 ± 0.000842117 (W 0.000938005) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.205961 ± 0.00114549 (W 0.00192879) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.017898 ± 7.63424e-05 (W 0.000101445) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0178935 ± 9.62928e-05 (W 0.000101623) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0920505 ± 0.000415054 (W 0.00114969) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.091992 ± 0.000469432 (W 0.000811732) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.26899 ± 0.000944373 (W 0.00192609) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.269031 ± 0.00137208 (W 0.00175477) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.010114 ± 9.48994e-05 (W 8.47888e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012323 ± 0.000144833 (W 0.000155616) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0388785 ± 0.000255865 (W 0.000197455) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.046206 ± 9.55149e-05 (W 0.000146737) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.093578 ± 0.000549195 (W 0.000211404) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.117668 ± 0.000833919 (W 0.00221522) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.053697 ± 0.000258548 (W 0.000856434) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.050095 ± 0.000166414 (W 0.000165856) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.185717 ± 0.00107038 (W 0.00122937) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.195175 ± 0.00089465 (W 0.00152907) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.436879 ± 0.00132449 (W 0.0016156) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.496627 ± 0.00325773 (W 0.00224993) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.032526 ± 0.000119606 (W 0.00012075) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.104211 ± 0.00043501 (W 0.000818253) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.229874 ± 0.00116691 (W 0.00136998) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011026 ± 0.000130811 (W 8.49503e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013691 ± 0.000128899 (W 5.65403e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0379055 ± 0.000178594 (W 0.000130868) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0473095 ± 0.000167954 (W 0.000123155) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.090522 ± 0.000383073 (W 0.0012559) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.118805 ± 0.000522897 (W 0.0024938) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.16068 ± 0.00135239 (W 0.00094281) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.161121 ± 0.00249097 (W 0.00108668) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.683629 ± 0.0200049 (W 0.00170829) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.683472 ± 0.00732637 (W 0.00199951) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.63691 ± 0.00758235 (W 0.00259357) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.63712 ± 0.00865292 (W 0.00292079) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.057324 ± 0.000274464 (W 9.54531e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0576975 ± 0.000324002 (W 0.000756338) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.224626 ± 0.000922618 (W 0.00129895) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.221104 ± 0.000815104 (W 0.00143084) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.537144 ± 0.00262775 (W 0.00192926) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.53843 ± 0.0034236 (W 0.00296483) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001057 ± 1.47196e-05 (W 3.57071e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002465 ± 2.78616e-05 (W 4.25786e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004308 ± 3.29267e-05 (W 5.08203e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012679 ± 4.25629e-05 (W 4.0858e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0120625 ± 5.66592e-05 (W 7.14164e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.070057 ± 0.000278198 (W 0.000187699) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066328 ± 0.000293282 (W 0.000200839) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.112868 ± 0.000543816 (W 0.000309461) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.1089 ± 0.000668345 (W 0.000373819) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.084644 ± 0.000148981 (W 0.000198483) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.085009 ± 0.000393059 (W 0.000198191) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.32214 ± 0.00111563 (W 0.00131514) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.322183 ± 0.00122879 (W 0.00135286) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.689169 ± 0.00235542 (W 0.00223092) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.691974 ± 0.0022771 (W 0.00229307) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014477 ± 7.0919e-05 (W 4.90641e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0182745 ± 0.000181368 (W 0.00011424) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0654525 ± 0.000350452 (W 0.000146163) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0768485 ± 0.000548696 (W 0.000127506) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.16683 ± 0.0014742 (W 0.000598508) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.198985 ± 0.00175283 (W 0.00246888) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.010159 ± 0.000198718 (W 0.000194738) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009723 ± 0.000146493 (W 0.000195153) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.037376 ± 0.000204885 (W 0.000591346) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.036396 ± 0.000195485 (W 0.000479231) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.088823 ± 0.000308265 (W 0.00065467) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0876355 ± 0.000480537 (W 0.000679644) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.010214 ± 0.000136675 (W 7.66932e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0124035 ± 5.19769e-05 (W 0.000114448) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.044638 ± 0.00030433 (W 0.000588961) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0464575 ± 0.000221133 (W 0.00048628) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.10218 ± 0.00141859 (W 0.00159348) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.110124 ± 0.00190502 (W 0.00299795) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.054839 ± 0.000187166 (W 0.000192515) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0515725 ± 0.000140693 (W 0.000217456) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.143371 ± 0.000715975 (W 0.000731963) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.146242 ± 0.000557135 (W 0.000357145) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.286635 ± 0.00158893 (W 0.00159902) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.325253 ± 0.00180879 (W 0.00244334) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0332175 ± 0.000118431 (W 0.000152589) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.096174 ± 0.000251556 (W 0.000392424) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.193811 ± 0.000646046 (W 0.00131696) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011582 ± 8.67064e-05 (W 9.03073e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014222 ± 0.000156666 (W 0.000102177) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0417625 ± 0.000325456 (W 0.00057483) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0486755 ± 0.000568088 (W 0.000733214) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.086261 ± 0.000120568 (W 0.00104206) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.112259 ± 0.000889813 (W 0.00202009) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.163134 ± 0.00164502 (W 0.000738652) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.164759 ± 0.00132569 (W 0.000348934) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.530198 ± 0.00214518 (W 0.00174712) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.533268 ± 0.00202007 (W 0.00166277) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.11171 ± 0.00379542 (W 0.00324173) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.11387 ± 0.00262938 (W 0.00337857) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0586765 ± 0.00042526 (W 0.000158779) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0592915 ± 0.00042287 (W 0.000136606) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.228141 ± 0.00125138 (W 0.00137739) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.227309 ± 0.000730601 (W 0.00110191) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.522805 ± 0.00251478 (W 0.00161103) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.52347 ± 0.00119192 (W 0.00256623) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001307 ± 1.54973e-05 (W 2.9419e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0029905 ± 1.5884e-05 (W 4.34879e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005099 ± 1.76144e-05 (W 6.23054e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0126485 ± 5.91292e-05 (W 4.51522e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012058 ± 5.9127e-05 (W 9.6315e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0704585 ± 0.000273574 (W 0.000760378) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0667695 ± 0.000321126 (W 0.000210967) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.112939 ± 0.000432451 (W 0.000193613) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.109151 ± 0.000223751 (W 0.000373322) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0817695 ± 0.000652615 (W 0.000193002) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0819125 ± 0.000727322 (W 0.000210507) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.258671 ± 0.00279123 (W 0.00138112) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.260174 ± 0.00134711 (W 0.00122384) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.545381 ± 0.00212577 (W 0.00188409) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.546639 ± 0.0032888 (W 0.00793635) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0147225 ± 0.000149073 (W 5.45634e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018313 ± 0.000293804 (W 0.000110178) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.066711 ± 0.000339586 (W 9.07635e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0771455 ± 0.000566578 (W 0.000202125) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.17735 ± 0.00148547 (W 0.00118907) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.207957 ± 0.00166018 (W 0.00246905) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0081005 ± 6.74418e-05 (W 0.000148284) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0080195 ± 0.000110182 (W 0.000194835) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.024005 ± 0.000257938 (W 0.000527194) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.024596 ± 0.000158927 (W 0.000562102) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.055343 ± 0.000489971 (W 0.000446201) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0563905 ± 0.000560514 (W 0.000647784) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.010149 ± 8.15908e-05 (W 7.01659e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012418 ± 8.20796e-05 (W 7.90774e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0496925 ± 0.000453125 (W 0.00111359) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.046169 ± 0.000367597 (W 0.000713767) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.101745 ± 0.00117447 (W 0.000886818) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.111663 ± 0.00097017 (W 0.00136613) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0550285 ± 0.000263634 (W 0.000216284) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.051449 ± 0.00036362 (W 0.00016217) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.139404 ± 0.00100524 (W 0.00100464) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.137871 ± 0.00509767 (W 0.000525392) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.262768 ± 0.000633924 (W 0.00159928) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.299317 ± 0.00117199 (W 0.00287505) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.033165 ± 0.000223148 (W 0.000217409) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0943625 ± 0.000679579 (W 0.000238149) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.178607 ± 0.000957247 (W 0.00111424) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0115415 ± 0.000115425 (W 0.00010719) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0144115 ± 0.000137711 (W 8.78653e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.041526 ± 0.000298532 (W 0.000547015) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0486735 ± 0.000259767 (W 0.000440461) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0923065 ± 0.00084673 (W 0.000798873) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.115847 ± 0.00144218 (W 0.00256424) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.162877 ± 0.000946789 (W 0.00130371) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.16508 ± 0.00122072 (W 0.000382231) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.520064 ± 0.0113713 (W 0.0015906) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.521432 ± 0.00708543 (W 0.00161132) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.08909 ± 0.00414742 (W 0.00304511) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.09808 ± 0.00454148 (W 0.00299995) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058862 ± 0.000331059 (W 0.000903065) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0593005 ± 0.000258124 (W 0.00016224) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.226391 ± 0.000997209 (W 0.00145196) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.226612 ± 0.000656633 (W 0.00146591) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.534141 ± 0.00157863 (W 0.00114731) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.53659 ± 0.00462191 (W 0.00235686) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001317 ± 1.56258e-05 (W 3.47831e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003016 ± 1.45694e-05 (W 4.70514e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00512 ± 2.54611e-05 (W 4.52884e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011466 ± 7.90959e-05 (W 4.38219e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012033 ± 7.62076e-05 (W 6.30443e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0649205 ± 0.000349653 (W 0.000149177) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066519 ± 0.00021041 (W 0.00127541) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.106008 ± 0.000481854 (W 0.000696251) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.109272 ± 0.000484572 (W 0.000586393) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.104528 ± 0.000548092 (W 0.00110365) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.104683 ± 0.000270634 (W 0.00125949) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.407203 ± 0.00665471 (W 0.00155656) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.411776 ± 0.0011874 (W 0.0021799) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.17516 ± 0.00465965 (W 0.0014927) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.16925 ± 0.00342967 (W 0.00313946) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013675 ± 0.000136039 (W 3.65782e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0153385 ± 8.23869e-05 (W 8.51437e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.063768 ± 0.000702868 (W 0.000696463) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0672705 ± 0.000417492 (W 9.61096e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.176163 ± 0.000842735 (W 0.00117002) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.189732 ± 0.0116674 (W 0.00268235) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.017939 ± 0.000104693 (W 8.55228e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0179745 ± 6.63375e-05 (W 0.000103916) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0906345 ± 0.000323745 (W 0.00131418) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0908875 ± 0.000378443 (W 0.00073894) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.272903 ± 0.00166145 (W 0.00172603) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.272701 ± 0.00118916 (W 0.00185174) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008606 ± 8.19766e-05 (W 5.87278e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010249 ± 9.796e-05 (W 0.000136958) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0355865 ± 0.000308499 (W 0.000144055) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0387235 ± 0.000379985 (W 0.000164259) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.08592 ± 0.000362428 (W 0.000248474) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.098603 ± 0.000458949 (W 0.00218907) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.04553 ± 0.000225541 (W 0.000232707) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.047879 ± 0.000299237 (W 0.00017781) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.182511 ± 0.000758587 (W 0.00145185) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.188147 ± 0.00102451 (W 0.00150205) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.443173 ± 0.00175039 (W 0.0017759) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.481527 ± 0.00212112 (W 0.00275767) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0242305 ± 0.000128058 (W 0.000156359) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.103051 ± 0.000564988 (W 0.00105839) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.238175 ± 0.00156026 (W 0.00162446) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.009854 ± 6.68443e-05 (W 0.000100137) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0117465 ± 6.6395e-05 (W 0.000118973) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.036484 ± 6.53781e-05 (W 0.000866059) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.039954 ± 0.000226938 (W 8.60297e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.088032 ± 0.00014468 (W 0.00122694) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.102184 ± 0.000401821 (W 0.00184944) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.161757 ± 0.00195292 (W 0.00128507) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.161802 ± 0.000775587 (W 0.000686754) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.691733 ± 0.0251581 (W 0.00145611) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.684592 ± 0.0228257 (W 0.00252694) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.63252 ± 0.00513897 (W 0.00277999) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.63574 ± 0.00976438 (W 0.00294232) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0583305 ± 0.00037311 (W 8.74382e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0587115 ± 0.000524659 (W 0.000110509) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.223482 ± 0.000625597 (W 0.00176548) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.220734 ± 0.00102378 (W 0.00115503) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.540083 ± 0.00228075 (W 0.00276111) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.542263 ± 0.0018424 (W 0.00207728) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001082 ± 1.3784e-05 (W 3.2892e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0024595 ± 2.12642e-05 (W 5.82961e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0039775 ± 2.28969e-05 (W 4.3609e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0114615 ± 5.30811e-05 (W 4.56043e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012043 ± 3.84053e-05 (W 9.07735e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0648605 ± 0.000271912 (W 0.000236417) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0661075 ± 0.000381972 (W 0.000163572) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.103763 ± 0.000418152 (W 0.000269804) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.107606 ± 0.000570721 (W 0.00026979) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.08485 ± 0.000460139 (W 0.000263014) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.08495 ± 0.000478093 (W 0.00021178) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.321503 ± 0.00115205 (W 0.00127289) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.322082 ± 0.0007708 (W 0.00159453) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.692383 ± 0.00171117 (W 0.00191978) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.69257 ± 0.00250202 (W 0.00164922) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0148825 ± 0.000331494 (W 5.33604e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0164105 ± 0.00170986 (W 8.18397e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.065764 ± 0.000337747 (W 0.000104) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0693455 ± 0.000543398 (W 0.000129971) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.166245 ± 0.00140997 (W 0.00134384) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.181648 ± 0.0055549 (W 0.00185285) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0099285 ± 9.12157e-05 (W 0.000256349) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0099935 ± 0.000177916 (W 0.000136895) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.036359 ± 0.000353426 (W 0.000302947) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0363405 ± 0.000134198 (W 0.000217114) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0872025 ± 0.00030865 (W 0.00144707) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0866315 ± 0.000287341 (W 0.000283075) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008641 ± 8.59622e-05 (W 8.91276e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.01035 ± 0.000120129 (W 0.000145587) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034931 ± 0.000180424 (W 0.000356723) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.038276 ± 0.00058503 (W 0.000561271) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0770235 ± 0.00140975 (W 0.00148142) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.091357 ± 0.000713044 (W 0.00268826) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0465175 ± 0.000273181 (W 0.000186938) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0493425 ± 0.000273013 (W 0.000221704) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.129559 ± 0.000605777 (W 0.000377065) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.137914 ± 0.000552689 (W 0.000499284) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.261329 ± 0.00167975 (W 0.00142797) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.307002 ± 0.00159462 (W 0.00266784) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0249955 ± 8.81998e-05 (W 0.000132379) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0885125 ± 0.000938247 (W 0.000346526) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.176966 ± 0.000801993 (W 0.0011086) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.010329 ± 0.000118058 (W 0.000115358) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.012228 ± 9.04516e-05 (W 7.74333e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0374545 ± 0.000301943 (W 0.000244266) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0410265 ± 0.000409039 (W 0.000692061) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0790805 ± 0.000198636 (W 0.000429346) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.095721 ± 0.000363483 (W 0.000721829) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.165252 ± 0.00316456 (W 0.000290206) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.165787 ± 0.0107716 (W 0.00044244) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.529946 ± 0.00233476 (W 0.00174658) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.531629 ± 0.00247328 (W 0.00183359) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.11945 ± 0.00176329 (W 0.00204788) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.1208 ± 0.00410542 (W 0.00236623) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0595925 ± 0.000338552 (W 0.000113114) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059878 ± 0.000357998 (W 0.000101083) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.226827 ± 0.000799484 (W 0.00120605) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.224314 ± 0.00101895 (W 0.00110925) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.525671 ± 0.00212804 (W 0.00194306) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.528749 ± 0.00328331 (W 0.0015762) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001317 ± 2.06954e-05 (W 2.79085e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0029205 ± 1.85005e-05 (W 3.93559e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004723 ± 3.65919e-05 (W 4.05853e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011391 ± 4.25566e-05 (W 4.0744e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.011972 ± 6.2341e-05 (W 0.000111043) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.064613 ± 0.000223731 (W 0.000696317) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0666645 ± 0.000326127 (W 0.000108918) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.103727 ± 0.000378827 (W 0.000237449) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108672 ± 0.000448579 (W 0.000338785) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0817915 ± 0.000414478 (W 0.000308072) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.082189 ± 0.000490198 (W 0.00017041) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.259474 ± 0.000937548 (W 0.00166863) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.26012 ± 0.000969831 (W 0.00131077) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.5462 ± 0.00325115 (W 0.00215978) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.544811 ± 0.00934935 (W 0.0112888) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0148025 ± 0.000283366 (W 5.4949e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0163705 ± 0.00150277 (W 0.000109937) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.06696 ± 0.00080716 (W 0.000148571) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0696195 ± 0.000453663 (W 0.000141165) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.176745 ± 0.00130121 (W 0.00114688) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.19328 ± 0.00302635 (W 0.00183689) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0079945 ± 9.54933e-05 (W 0.000209621) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.007895 ± 0.000139858 (W 0.000195868) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0241655 ± 0.000285791 (W 0.000431192) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.024651 ± 0.00219247 (W 0.00032012) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.052601 ± 0.000484657 (W 0.000364721) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.052578 ± 0.000494816 (W 0.000330359) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0087115 ± 6.6955e-05 (W 6.83176e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0103395 ± 8.2131e-05 (W 0.000165231) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034544 ± 0.000308463 (W 0.000371371) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.038351 ± 0.000439242 (W 0.000277971) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0750815 ± 0.00131406 (W 0.00113888) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.094136 ± 0.000738653 (W 0.00235872) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.04666 ± 0.000119806 (W 0.000162044) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.049406 ± 0.000320732 (W 0.000143793) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.123906 ± 0.000788325 (W 0.000718614) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.130559 ± 0.000863816 (W 0.000499743) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.234416 ± 0.00155084 (W 0.00128675) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.284668 ± 0.00267793 (W 0.00270144) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.024986 ± 0.000206425 (W 0.000138301) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0859915 ± 0.000568634 (W 0.000399142) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.166218 ± 0.0058804 (W 0.000651076) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.0103295 ± 0.00010016 (W 0.000131028) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.012373 ± 5.26789e-05 (W 0.000130881) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.037408 ± 0.000160903 (W 0.000279216) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0410325 ± 0.000256795 (W 0.000552877) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0793495 ± 0.00045839 (W 0.000364351) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.098215 ± 0.000421436 (W 0.00132217) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.16422 ± 0.00372227 (W 0.00120126) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.165583 ± 0.00237545 (W 0.000286073) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.51674 ± 0.0021773 (W 0.00186815) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.519706 ± 0.00368152 (W 0.00198808) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.09145 ± 0.00431053 (W 0.00214959) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.10021 ± 0.00936568 (W 0.00165158) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059711 ± 0.000375723 (W 9.81339e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060042 ± 0.000328749 (W 0.000119) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.225188 ± 0.00130741 (W 0.00142315) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.225536 ± 0.000941609 (W 0.00115562) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.539322 ± 0.00398326 (W 0.00184442) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.539817 ± 0.00302358 (W 0.00149853) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0013225 ± 1.8411e-05 (W 4.01381e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003081 ± 1.25539e-05 (W 6.28857e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0050695 ± 4.97755e-05 (W 5.75213e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `lu/32`, `lu/64`, `lu/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `solve/32`, `solve/64`, `solve/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `triangular_solve/32`, `triangular_solve/64`, `triangular_solve/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0591635 ± 0.000595343 (W 0.000189767) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.061528 ± 0.000711134 (W 0.000229577) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.369201 ± 0.00125271 (W 0.0033491) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.34228 ± 0.00127805 (W 0.00190886) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.91631 ± 0.0120396 (W 0.00675554) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.13827 ± 0.00809048 (W 0.00686194) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.56304 ± 0.0112685 (W 0.00341297) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.55717 ± 0.0129117 (W 0.00561118) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.6061 ± 0.0458347 (W 0.00987105) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.6284 ± 0.0598024 (W 0.015333) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 50.892 ± 0.355846 (W 0.124469) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 50.7457 ± 0.240901 (W 0.0479039) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.36357 ± 0.00221388 (W 0.00205872) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.335283 ± 0.00196685 (W 0.00208213) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.90362 ± 0.00983342 (W 0.00938263) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.04903 ± 0.0165032 (W 0.0113081) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 21.6443 ± 8.43023 (W 0.0681505) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 22.3663 ± 8.34582 (W 0.0382234) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.258126 ± 0.0010306 (W 0.00162877) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.259104 ± 0.000804108 (W 0.00161416) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.89828 ± 0.00795064 (W 0.003101) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.89129 ± 0.00542394 (W 0.00466114) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9072 ± 0.0422167 (W 0.0267242) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.9351 ± 0.0315089 (W 0.0290748) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.104 ± 0.000509851 (W 0.000235478) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.161906 ± 0.00100045 (W 0.00136001) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.570242 ± 0.00322799 (W 0.00323646) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.937737 ± 0.00611408 (W 0.00294306) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.8761 ± 0.0137687 (W 0.00603971) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.67108 ± 0.0370155 (W 0.0200533) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.447196 ± 0.00391258 (W 0.00249944) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.558697 ± 0.001313 (W 0.00222632) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.87725 ± 0.00906724 (W 0.00822213) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.19394 ± 0.0129642 (W 0.0057537) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 18.7087 ± 0.0728318 (W 0.0658213) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 20.0733 ± 0.103117 (W 0.0247594) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.247166 ± 0.00140233 (W 0.00170922) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.51429 ± 0.00532769 (W 0.00462868) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 8.46685 ± 0.083667 (W 0.0329882) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0951145 ± 0.000513398 (W 0.000804369) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.163361 ± 0.00136183 (W 0.00185244) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.535708 ± 0.00299093 (W 0.00219051) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.690863 ± 0.00763693 (W 0.00293667) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.7238 ± 0.0114639 (W 0.010765) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.25876 ± 0.0453165 (W 0.00938867) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.87024 ± 0.00792461 (W 0.00269004) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.86648 ± 0.00811099 (W 0.00637934) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.5444 ± 0.0407305 (W 0.021508) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 14.5578 ± 0.0735033 (W 0.0315659) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 86.8758 ± 0.201561 (W 0.133433) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 86.7639 ± 0.329653 (W 0.125124) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.92453 ± 0.00320401 (W 0.00360826) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.923262 ± 0.00290031 (W 0.00299036) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.47544 ± 0.0307507 (W 0.00958988) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.47403 ± 0.0165044 (W 0.0134486) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 37.3687 ± 0.0550946 (W 0.0544088) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 37.5193 ± 0.19689 (W 0.0426889) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0042935 ± 3.56875e-05 (W 6.12e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0120175 ± 0.000107982 (W 0.000126409) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0403655 ± 0.000343705 (W 0.000338502) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0510085 ± 0.00062527 (W 0.000332077) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.0587305 ± 0.00075061 (W 0.000669022) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.20801 ± 0.00194266 (W 0.0014445) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.22426 ± 0.00146789 (W 0.00181175) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.09116 ± 0.00384288 (W 0.00316896) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.968271 ± 0.0114526 (W 0.00353523) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.44719 ± 0.00604772 (W 0.00523445) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.44993 ± 0.00413371 (W 0.00354595) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.68369 ± 0.0212082 (W 0.00856016) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.66492 ± 0.0212133 (W 0.0112419) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.7996 ± 0.0847619 (W 0.0186923) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.5548 ± 0.0238947 (W 0.0332173) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.333911 ± 0.00134783 (W 0.00268613) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.340207 ± 0.00393366 (W 0.00469581) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.98514 ± 0.00853217 (W 0.0076622) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.24882 ± 0.00894157 (W 0.00796517) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.5032 ± 0.0368001 (W 0.0392215) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.075 ± 0.0739774 (W 0.0265452) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0845635 ± 0.000150876 (W 0.000677755) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0842805 ± 0.000752599 (W 0.000928644) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.514209 ± 0.00335949 (W 0.00345126) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.517443 ± 0.000878857 (W 0.00224325) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.17326 ± 0.0156802 (W 0.0118786) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.16014 ± 0.0117715 (W 0.00991075) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.118677 ± 0.00150454 (W 0.0022828) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.153064 ± 0.00107345 (W 0.00126795) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.401091 ± 0.00314373 (W 0.00518101) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.706379 ± 0.00481315 (W 0.00275364) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.78528 ± 0.00573953 (W 0.0137225) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.50278 ± 0.0335136 (W 0.0112648) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.341053 ± 0.00267865 (W 0.0022717) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.426091 ± 0.00304386 (W 0.00250973) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.04477 ± 0.00385999 (W 0.0023285) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.57095 ± 0.00464623 (W 0.00902709) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.98478 ± 0.023681 (W 0.0217939) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.28032 ± 0.0395058 (W 0.0373387) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.225577 ± 0.00100823 (W 0.00107501) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.664042 ± 0.00309511 (W 0.00243634) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 3.21193 ± 0.0126005 (W 0.0132482) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0940155 ± 0.000814418 (W 0.00097272) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.116108 ± 0.000677591 (W 0.000446623) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.386961 ± 0.00260755 (W 0.00292945) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.712931 ± 0.00372105 (W 0.00381038) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.72063 ± 0.0100887 (W 0.00705716) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.44784 ± 0.0374332 (W 0.00985001) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.00056 ± 0.00926981 (W 0.00354451) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.0179 ± 0.00674901 (W 0.0052517) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.12124 ± 0.0273117 (W 0.011204) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.16509 ± 0.0191689 (W 0.0138977) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 36.6573 ± 0.0768185 (W 0.113197) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 36.7949 ± 0.092879 (W 0.0308051) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.958157 ± 0.0042217 (W 0.00294085) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.957074 ± 0.00318157 (W 0.00360509) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.12428 ± 0.0186524 (W 0.012537) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.13205 ± 0.0112854 (W 0.00763212) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.4286 ± 0.0613398 (W 0.0211537) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.4937 ± 0.0459079 (W 0.0296146) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0051095 ± 4.25864e-05 (W 8.39653e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0125635 ± 0.000199804 (W 0.000152281) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0302265 ± 0.000238727 (W 0.000227386) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0506845 ± 0.00049923 (W 0.000416085) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.0585595 ± 0.000572203 (W 0.000802292) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.163198 ± 0.0018022 (W 0.00157689) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.207469 ± 0.00185324 (W 0.00208529) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.797217 ± 0.0018679 (W 0.00328897) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.80085 ± 0.00843934 (W 0.00227879) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.938897 ± 0.00525019 (W 0.00235758) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.944944 ± 0.00195629 (W 0.00266728) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.97161 ± 0.00856388 (W 0.00870759) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.96618 ± 0.0141886 (W 0.00880608) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.4633 ± 0.0510799 (W 0.01924) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.4682 ± 0.0338469 (W 0.022026) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.334898 ± 0.00280095 (W 0.00247361) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.391947 ± 0.00684448 (W 0.00363914) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.88418 ± 0.0101328 (W 0.00723814) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.19138 ± 0.00936777 (W 0.00748951) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.1295 ± 0.0233801 (W 0.0242668) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.8436 ± 0.0349455 (W 0.0222984) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.058289 ± 0.000445764 (W 0.000850478) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.058694 ± 0.000678233 (W 0.00110462) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.284988 ± 0.00198216 (W 0.00205107) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.285066 ± 0.00148365 (W 0.00229959) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.25526 ± 0.0133179 (W 0.0106496) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.24856 ± 0.0260478 (W 0.0168798) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.122418 ± 0.00160963 (W 0.00129538) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.154036 ± 0.000986425 (W 0.000526046) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.340043 ± 0.00290964 (W 0.00198411) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.680916 ± 0.00448798 (W 0.00356993) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.29456 ± 0.00917567 (W 0.00575377) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.15125 ± 0.0125431 (W 0.0101194) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.322357 ± 0.00242818 (W 0.00176204) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.404562 ± 0.00282244 (W 0.00358426) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.851234 ± 0.00290462 (W 0.00251938) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.39534 ± 0.00449775 (W 0.00648222) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.43332 ± 0.0129818 (W 0.00849272) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 6.05559 ± 0.0235077 (W 0.014268) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.207157 ± 0.00104952 (W 0.00111561) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.610847 ± 0.00358748 (W 0.00199223) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.92469 ± 0.00689884 (W 0.00354396) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0985885 ± 0.000325641 (W 0.000916646) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.114561 ± 0.000301604 (W 0.000482929) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.330936 ± 0.00274316 (W 0.00200975) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.68859 ± 0.00611961 (W 0.00417661) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.25922 ± 0.00549901 (W 0.00719149) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.11162 ± 0.019578 (W 0.0115262) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.98241 ± 0.00857577 (W 0.00605919) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99165 ± 0.00771903 (W 0.00444118) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.38077 ± 0.0282807 (W 0.013245) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.45546 ± 0.0230558 (W 0.0159204) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.3631 ± 0.0929315 (W 0.0359928) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.4924 ± 0.0570327 (W 0.0342177) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.0075 ± 0.0063972 (W 0.0025157) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.00987 ± 0.00374876 (W 0.00298571) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.00439 ± 0.0148245 (W 0.00975854) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.00058 ± 0.00830171 (W 0.00872623) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.3355 ± 0.0348155 (W 0.0201174) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.3752 ± 0.0572413 (W 0.0330558) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005435 ± 6.21624e-05 (W 7.21873e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013059 ± 0.000153402 (W 0.000148847) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030291 ± 0.000168517 (W 0.000227126) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0495705 ± 0.000334456 (W 9.96645e-05) | 0.0622675 ± 0.000584052 (W 0.000122637) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.0565425 ± 0.000315872 (W 9.12393e-05) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.285979 ± 0.00102298 (W 0.00191717) | 0.364476 ± 0.00131294 (W 0.00198259) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.312206 ± 0.0011339 (W 0.00227853) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.94757 ± 0.00806778 (W 0.00442113) | 2.52036 ± 0.00840623 (W 0.00414897) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.01752 ± 0.00989736 (W 0.00685412) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.55555 ± 0.0144557 (W 0.00382155) | 1.83508 ± 0.00672097 (W 0.00556395) | 🔴 faer-fork 1.39× slower than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 2.5482 ± 0.00960888 (W 0.00631343) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.5709 ± 0.0421069 (W 0.0229127) | 8.72983 ± 0.0298412 (W 0.0145154) | 🔴 faer-fork 1.21× slower than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 10.619 ± 0.0580227 (W 0.0171529) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 50.513 ± 0.312785 (W 0.0464727) | 48.9521 ± 0.248903 (W 0.0547913) | · |  |
| `eigh/n512/b1 [api-inclusive]` | 50.5063 ± 0.178656 (W 0.0538148) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.32526 ± 0.00210665 (W 0.00202525) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.315326 ± 0.00140855 (W 0.00183482) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.8738 ± 0.0130481 (W 0.00980694) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.95333 ± 0.0205581 (W 0.0133955) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 22.4319 ± 8.07248 (W 0.0277212) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 21.849 ± 8.18325 (W 0.0451974) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.25477 ± 0.000402031 (W 0.00150669) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.254551 ± 0.00193804 (W 0.00211618) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.87048 ± 0.00472843 (W 0.00539692) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.87398 ± 0.0071926 (W 0.00557813) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.7962 ± 0.0359701 (W 0.0435568) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.7932 ± 0.0449558 (W 0.0215273) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.097093 ± 0.000713409 (W 0.000867902) | 0.139785 ± 0.000545079 (W 0.00127523) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.144211 ± 0.00121949 (W 0.00157721) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.571211 ± 0.00203699 (W 0.00251055) | 0.774255 ± 0.00352036 (W 0.00229443) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.855533 ± 0.00194595 (W 0.00261721) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.74894 ± 0.0147383 (W 0.00809429) | 4.92747 ± 0.0166549 (W 0.00914116) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 5.16081 ± 0.10993 (W 0.0132166) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.453201 ± 0.0010785 (W 0.00154345) | 0.799038 ± 0.00846746 (W 0.00816112) | 🔵 faer-fork 1.76× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.532515 ± 0.00188098 (W 0.00263298) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.62743 ± 0.0142847 (W 0.00328193) | 4.49921 ± 0.0134433 (W 0.0148437) | 🔵 faer-fork 1.71× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 3.08117 ± 0.0144668 (W 0.00621742) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 17.4616 ± 0.0629269 (W 0.0218656) | 31.3544 ± 0.0721719 (W 0.0407553) | 🔵 faer-fork 1.80× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 19.5791 ± 0.0874191 (W 0.0700405) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.250908 ± 0.00200554 (W 0.00178156) | 0.361785 ± 0.00156811 (W 0.00207805) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 1.28335 ± 0.00549052 (W 0.00304016) | 1.79911 ± 0.00894526 (W 0.00487923) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 7.62131 ± 0.0202899 (W 0.0148305) | 12.1224 ± 0.0377444 (W 0.0190279) | 🔵 faer-fork 1.59× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.0960255 ± 0.000374786 (W 0.000246994) | 0.138001 ± 0.00081234 (W 0.00150238) | 🔵 faer-fork 1.44× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.141454 ± 0.000694524 (W 0.00215132) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.565937 ± 0.00172198 (W 0.00259261) | 0.760006 ± 0.00275622 (W 0.00236907) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.600344 ± 0.00131261 (W 0.00193465) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.7152 ± 0.0138433 (W 0.00854231) | 4.84467 ± 0.0228791 (W 0.00941994) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 3.85982 ± 0.0105655 (W 0.00977498) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.86115 ± 0.0140934 (W 0.00276781) | 3.08593 ± 0.0127202 (W 0.00927523) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.85184 ± 0.0181985 (W 0.0041928) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.4947 ± 0.0290722 (W 0.0212673) | 17.1782 ± 0.0556383 (W 0.0238652) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 14.5048 ± 0.0816405 (W 0.0496068) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 86.2566 ± 0.297352 (W 0.22546) | 115.325 ± 1.02085 (W 0.295612) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 86.3412 ± 0.156513 (W 0.226754) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.902881 ± 0.0044075 (W 0.00113561) | 1.31453 ± 0.00372558 (W 0.00290794) | 🔵 faer-fork 1.46× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.90256 ± 0.00544751 (W 0.00108242) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.34083 ± 0.0281442 (W 0.0108749) | 7.50099 ± 0.0189105 (W 0.00991035) | 🔵 faer-fork 1.40× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.3509 ± 0.0158303 (W 0.0121787) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 36.4477 ± 0.14584 (W 0.0373951) | 51.6757 ± 0.443149 (W 0.161804) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 36.3903 ± 0.161771 (W 0.0698843) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004669 ± 4.59148e-05 (W 7.25519e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.012789 ± 9.52969e-05 (W 7.92933e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0429145 ± 0.000261479 (W 0.000289142) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.043961 ± 0.000736692 (W 0.000629221) | 0.070923 ± 0.000542278 (W 0.000850129) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.0536145 ± 0.000354065 (W 0.000272108) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.158334 ± 0.00130938 (W 0.001377) | 0.24641 ± 0.0011107 (W 0.00287899) | 🔵 faer-fork 1.56× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.195982 ± 0.00128003 (W 0.00153581) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.701502 ± 0.00472501 (W 0.0026224) | 1.16626 ± 0.00552868 (W 0.00369401) | 🔵 faer-fork 1.66× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.850167 ± 0.00380755 (W 0.00210525) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.44461 ± 0.0042707 (W 0.00372464) | 1.46 ± 0.00623873 (W 0.00260956) | · |  |
| `eigh/n128/b1 [api-inclusive]` | 1.45203 ± 0.00942407 (W 0.00362775) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.63709 ± 0.0246081 (W 0.00648058) | 5.73437 ± 0.0269346 (W 0.009261) | · |  |
| `eigh/n256/b1 [api-inclusive]` | 5.62759 ± 0.0112862 (W 0.0120911) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.6578 ± 0.0532338 (W 0.0213439) | 26.0918 ± 0.172663 (W 0.0362687) | 🔵 faer-fork 1.06× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 24.4124 ± 0.0552352 (W 0.0238034) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.320269 ± 0.00237489 (W 0.00364029) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.322358 ± 0.003262 (W 0.00298646) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.96001 ± 0.0100026 (W 0.00856718) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.1617 ± 0.00891116 (W 0.0102934) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.6656 ± 0.0448872 (W 0.0156502) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.6327 ± 0.0453143 (W 0.0278155) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.083246 ± 0.00037523 (W 0.000380271) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.083856 ± 0.000554408 (W 0.000882945) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.520196 ± 0.0036024 (W 0.0031431) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.523402 ± 0.00184438 (W 0.00221987) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.18749 ± 0.00837294 (W 0.0106514) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.18431 ± 0.0135716 (W 0.00879186) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0787735 ± 0.00102411 (W 0.000547225) | 0.140631 ± 0.000874406 (W 0.00113257) | 🔵 faer-fork 1.79× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.133951 ± 0.00112682 (W 0.00121457) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.330732 ± 0.00248699 (W 0.00295666) | 0.438053 ± 0.00212228 (W 0.00233027) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.617333 ± 0.00272683 (W 0.00412024) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.57083 ± 0.00746865 (W 0.00514504) | 2.39628 ± 0.0136976 (W 0.00464452) | 🔵 faer-fork 1.53× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 3.01751 ± 0.0198022 (W 0.00732299) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.300966 ± 0.00195625 (W 0.00132443) | 0.795751 ± 0.00208268 (W 0.00242648) | 🔵 faer-fork 2.64× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.397179 ± 0.00340627 (W 0.00247943) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.981559 ± 0.001945 (W 0.00183145) | 3.21376 ± 0.012164 (W 0.00987056) | 🔵 faer-fork 3.27× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 1.4639 ± 0.00858524 (W 0.0066948) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.54918 ± 0.0194026 (W 0.00704019) | 18.3964 ± 0.0597068 (W 0.0218292) | 🔵 faer-fork 3.32× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 7.73537 ± 0.0200507 (W 0.0167842) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.203699 ± 0.000432741 (W 0.00151211) | 0.385794 ± 0.00163831 (W 0.001684) | 🔵 faer-fork 1.89× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 0.58492 ± 0.00269826 (W 0.00262455) | 1.3854 ± 0.00507692 (W 0.00503487) | 🔵 faer-fork 2.37× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 2.92645 ± 0.00764412 (W 0.00580101) | 7.27809 ± 0.0800439 (W 0.016644) | 🔵 faer-fork 2.49× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.0837295 ± 0.000566181 (W 0.000670755) | 0.102498 ± 0.00137246 (W 0.00182076) | 🔵 faer-fork 1.22× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.095129 ± 0.000403902 (W 0.000562525) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.329781 ± 0.0032074 (W 0.00206597) | 0.419212 ± 0.00231208 (W 0.00171716) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.624223 ± 0.0040321 (W 0.00264884) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.50477 ± 0.00266158 (W 0.00672492) | 2.32014 ± 0.0145074 (W 0.00450995) | 🔵 faer-fork 1.54× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 2.95701 ± 0.0119799 (W 0.011397) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.98325 ± 0.00842981 (W 0.00404302) | 3.07461 ± 0.00819711 (W 0.00418088) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99606 ± 0.00544671 (W 0.00549331) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.01621 ± 0.0229023 (W 0.0128229) | 14.5556 ± 0.0104225 (W 0.017251) | 🔵 faer-fork 1.82× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 8.05694 ± 0.0218459 (W 0.0108079) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 35.7711 ± 0.0616485 (W 0.0521333) | 87.6776 ± 0.370626 (W 0.125129) | 🔵 faer-fork 2.45× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 35.9273 ± 0.0479645 (W 0.039035) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.937747 ± 0.00534518 (W 0.00278581) | 1.33953 ± 0.0139023 (W 0.00250837) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.937191 ± 0.00423485 (W 0.00274791) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.99661 ± 0.0195718 (W 0.00950899) | 7.32579 ± 0.0134045 (W 0.0147782) | 🔵 faer-fork 1.83× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 4.01157 ± 0.0197811 (W 0.00796463) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.4294 ± 0.073933 (W 0.0185983) | 48.1771 ± 0.317179 (W 0.0484819) | 🔵 faer-fork 2.76× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 17.4646 ± 0.0610188 (W 0.0194442) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005475 ± 3.40588e-05 (W 6.49229e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.014953 ± 0.000242703 (W 0.000129831) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.038214 ± 0.000447273 (W 0.000456884) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0442975 ± 0.000888779 (W 0.000411992) | 0.0697615 ± 0.00084975 (W 0.000994876) | 🔵 faer-fork 1.57× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.054 ± 0.000325611 (W 0.000698862) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.141164 ± 0.00154701 (W 0.00128338) | 0.23792 ± 0.00321968 (W 0.00217292) | 🔵 faer-fork 1.69× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.180606 ± 0.00161067 (W 0.00183004) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.519751 ± 0.00140883 (W 0.00294405) | 1.02651 ± 0.00569035 (W 0.00465487) | 🔵 faer-fork 1.97× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.685029 ± 0.00270678 (W 0.0023636) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.934497 ± 0.00274686 (W 0.00204723) | 1.38695 ± 0.00591285 (W 0.00328338) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.938411 ± 0.0016439 (W 0.00202671) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.93257 ± 0.0560265 (W 0.00630095) | 5.34604 ± 0.0152454 (W 0.010465) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.9309 ± 0.0128913 (W 0.00684965) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.2969 ± 0.0584981 (W 0.0225558) | 22.3661 ± 0.115847 (W 0.0263963) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 16.2978 ± 0.0608414 (W 0.0191052) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.326771 ± 0.0046554 (W 0.00312049) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.371355 ± 0.00296357 (W 0.00303822) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.85697 ± 0.00549192 (W 0.00877154) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.10167 ± 0.0111916 (W 0.0109616) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.1246 ± 0.0228008 (W 0.0196948) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.3689 ± 0.0271392 (W 0.0252502) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0558895 ± 0.000394916 (W 0.000740696) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0563695 ± 0.00039467 (W 0.000566477) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.288551 ± 0.00191196 (W 0.00232615) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.291554 ± 0.00220933 (W 0.00194897) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.30688 ± 0.00724041 (W 0.0115809) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.31243 ± 0.00968498 (W 0.0133797) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0788535 ± 0.000501547 (W 0.000392245) | 0.140162 ± 0.000738063 (W 0.00130663) | 🔵 faer-fork 1.78× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.133768 ± 0.000306439 (W 0.000633348) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.29149 ± 0.00161439 (W 0.0024941) | 0.371222 ± 0.00174722 (W 0.00160002) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.587824 ± 0.00356547 (W 0.00329823) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.20719 ± 0.00764106 (W 0.00702843) | 1.51573 ± 0.00818581 (W 0.00459298) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.65057 ± 0.0182438 (W 0.00808473) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.274624 ± 0.00105866 (W 0.00148767) | 0.783844 ± 0.00499309 (W 0.00309385) | 🔵 faer-fork 2.85× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.376706 ± 0.0019851 (W 0.00292707) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.806239 ± 0.0171818 (W 0.00163132) | 3.12343 ± 0.018258 (W 0.0155911) | 🔵 faer-fork 3.87× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 1.28923 ± 0.0068005 (W 0.00534039) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.21531 ± 0.00519603 (W 0.00580193) | 16.305 ± 0.171765 (W 0.040941) | 🔵 faer-fork 5.07× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 5.48949 ± 0.0116604 (W 0.0192938) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.1947 ± 0.000736072 (W 0.00157548) | 0.378296 ± 0.0018823 (W 0.00165331) | 🔵 faer-fork 1.94× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 0.571243 ± 0.00960104 (W 0.00302279) | 1.37308 ± 0.00492536 (W 0.00404871) | 🔵 faer-fork 2.40× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 1.77212 ± 0.00417727 (W 0.00263301) | 6.55154 ± 0.0336625 (W 0.0249977) | 🔵 faer-fork 3.70× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.082162 ± 0.00049561 (W 0.000449445) | 0.136326 ± 0.0014191 (W 0.00197012) | 🔵 faer-fork 1.66× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.093734 ± 0.000592691 (W 0.000430079) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.290338 ± 0.00157592 (W 0.00190365) | 0.379372 ± 0.00357857 (W 0.00390594) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.592718 ± 0.00502919 (W 0.00470606) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.15859 ± 0.00419447 (W 0.00382383) | 1.4755 ± 0.00213865 (W 0.00680499) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 2.60922 ± 0.0171459 (W 0.00987274) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.96997 ± 0.00719021 (W 0.00650564) | 3.05436 ± 0.00762052 (W 0.00511807) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.97098 ± 0.00705193 (W 0.00411328) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.2762 ± 0.013088 (W 0.0113366) | 14.3423 ± 0.0429949 (W 0.0223616) | 🔵 faer-fork 1.97× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.31837 ± 0.00892076 (W 0.016543) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 29.4858 ± 0.0699309 (W 0.0379696) | 84.0531 ± 0.561458 (W 0.0926427) | 🔵 faer-fork 2.85× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 29.5879 ± 0.0847885 (W 0.0290211) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.980746 ± 0.00664507 (W 0.00303288) | 1.34246 ± 0.00483114 (W 0.00257189) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.987181 ± 0.0116942 (W 0.00316527) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.88862 ± 0.0140461 (W 0.010066) | 7.26331 ± 0.0277561 (W 0.012112) | 🔵 faer-fork 1.87× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.89275 ± 0.0135546 (W 0.00655276) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.3982 ± 0.0431804 (W 0.0205051) | 46.7423 ± 0.224456 (W 0.054357) | 🔵 faer-fork 3.04× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 15.4059 ± 0.041583 (W 0.0149221) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0058455 ± 2.72305e-05 (W 8.24216e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0156985 ± 0.00023668 (W 0.000209337) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0376505 ± 0.000377244 (W 0.000285193) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.181372 ± 0.000804483 (W 0.00128703) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188382 ± 0.000804851 (W 0.00135172) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.1465 ± 0.00343631 (W 0.00239407) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.15934 ± 0.00418998 (W 0.00265678) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.53877 ± 0.022907 (W 0.018267) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 7.67342 ± 0.0360669 (W 0.0161384) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.56711 ± 0.0109179 (W 0.00311736) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 3.55237 ± 0.0138919 (W 0.00480743) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 18.0451 ± 0.0776024 (W 0.0118826) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 18.0853 ± 0.0616328 (W 0.0160203) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 106.749 ± 0.301332 (W 0.153365) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 106.777 ± 0.364256 (W 0.146136) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.845748 ± 0.0122146 (W 0.0047925) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.921412 ± 0.00697232 (W 0.00496422) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 6.1114 ± 0.0473198 (W 0.0143429) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.58928 ± 0.0301421 (W 0.0193405) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 40.9552 ± 0.074275 (W 0.0532024) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 43.9424 ± 0.162052 (W 0.0786067) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.04938 ± 0.00525159 (W 0.00162131) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.04757 ± 0.00712128 (W 0.00207779) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.50876 ± 0.02446 (W 0.0104264) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.51994 ± 0.0258718 (W 0.00614098) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.543 ± 0.17362 (W 0.175897) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.625 ± 0.160859 (W 0.128059) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.286139 ± 0.00075176 (W 0.00198786) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.438304 ± 0.0016234 (W 0.00263349) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.89978 ± 0.00774475 (W 0.0067959) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.66836 ± 0.00243619 (W 0.00676941) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.3539 ± 0.0421738 (W 0.0184932) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 17.8797 ± 0.0660924 (W 0.0377866) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.4742 ± 0.00601266 (W 0.00497328) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 1.70673 ± 0.00841586 (W 0.00328006) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 10.1715 ± 0.0245105 (W 0.00889974) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 10.9522 ± 0.0286589 (W 0.0166918) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 70.0641 ± 0.32564 (W 0.069244) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 73.6812 ± 0.176652 (W 0.0539911) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.68185 ± 0.00416245 (W 0.00199327) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 4.5714 ± 0.016951 (W 0.00699849) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 29.7051 ± 0.256823 (W 0.0800007) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.278442 ± 0.00194577 (W 0.00185822) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.444414 ± 0.00270362 (W 0.00330584) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.87094 ± 0.00791229 (W 0.00628435) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 2.07617 ± 0.0118888 (W 0.00406636) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.2182 ± 0.0394819 (W 0.0184141) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 14.7443 ± 0.0522297 (W 0.033597) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.77252 ± 0.0180463 (W 0.00632893) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 4.77473 ± 0.0109894 (W 0.00583722) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.4194 ± 0.0725478 (W 0.0640891) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 29.5401 ± 0.137242 (W 0.0774761) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 194.148 ± 0.883007 (W 0.548847) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 194.743 ± 0.747629 (W 0.596909) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.59769 ± 0.00622991 (W 0.00507643) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.60053 ± 0.00460666 (W 0.00302246) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.4283 ± 0.0855618 (W 0.083669) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 11.4064 ± 0.0879528 (W 0.064541) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 82.9981 ± 0.344914 (W 0.197937) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 83.0342 ± 0.111727 (W 0.248128) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0079795 ± 3.4355e-05 (W 4.04974e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0243605 ± 0.000211777 (W 0.000113136) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0881455 ± 0.000642541 (W 0.000979567) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.139844 ± 0.000782422 (W 0.00213725) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.155911 ± 0.00175399 (W 0.00143859) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.537841 ± 0.00196926 (W 0.00247309) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.609288 ± 0.0036694 (W 0.0034765) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.38488 ± 0.00643399 (W 0.00680227) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.75477 ± 0.0810925 (W 0.00429957) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 1.94291 ± 0.00672735 (W 0.00286762) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.94899 ± 0.00752208 (W 0.00281375) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.88286 ± 0.022642 (W 0.0118837) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 7.89784 ± 0.0184073 (W 0.0137946) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.8298 ± 0.0403666 (W 0.0346072) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 39.9823 ± 0.0482854 (W 0.0253037) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.599775 ± 0.00598852 (W 0.00380961) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.724809 ± 0.00777192 (W 0.00426998) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.35212 ± 0.0090853 (W 0.0104481) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 4.08219 ± 0.0119338 (W 0.0144239) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 16.6779 ± 0.107259 (W 0.0218955) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 20.6518 ± 0.0717426 (W 0.0695841) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.286479 ± 0.00053411 (W 0.00222194) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.286422 ± 0.00122079 (W 0.00142406) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.94267 ± 0.00486448 (W 0.00421897) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.94449 ± 0.00553044 (W 0.00327631) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6269 ± 0.0257982 (W 0.0156458) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6074 ± 0.0330504 (W 0.0193064) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.230009 ± 0.00185069 (W 0.00186648) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.361067 ± 0.00147292 (W 0.00426802) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.965806 ± 0.00388706 (W 0.00723205) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.71624 ± 0.0196876 (W 0.00782428) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.16073 ± 0.0350015 (W 0.0382114) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 9.38352 ± 0.0589563 (W 0.0518506) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.733234 ± 0.00314637 (W 0.00333407) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.936381 ± 0.00443836 (W 0.00528516) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.1327 ± 0.0155155 (W 0.00662483) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 4.21268 ± 0.0360602 (W 0.0179318) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 19.9505 ± 0.0529975 (W 0.0165583) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 25.3621 ± 0.0611539 (W 0.0268893) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.446074 ± 0.00199418 (W 0.00173348) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.6446 ± 0.00643186 (W 0.00217654) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 9.41542 ± 0.0255306 (W 0.0353291) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.201887 ± 0.000736532 (W 0.00182771) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.36277 ± 0.00330499 (W 0.00552168) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.974179 ± 0.0124571 (W 0.0107893) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.71179 ± 0.0113909 (W 0.0145157) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.06487 ± 0.0113728 (W 0.0292229) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 9.39335 ± 0.0990027 (W 0.0588037) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.7088 ± 0.0111351 (W 0.00819696) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.71305 ± 0.00816897 (W 0.00633771) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.7768 ± 0.041957 (W 0.0189779) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 12.8259 ± 0.0310471 (W 0.0239589) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 68.6592 ± 0.103481 (W 0.290098) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 68.8288 ± 0.264693 (W 0.114776) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.25939 ± 0.00602081 (W 0.00429417) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.25527 ± 0.00310569 (W 0.00323685) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.19576 ± 0.0284396 (W 0.0163133) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.22196 ± 0.0283536 (W 0.0148415) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 33.3827 ± 0.125789 (W 0.0660085) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 33.6387 ± 0.0969281 (W 0.061187) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009102 ± 3.44248e-05 (W 5.19034e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0217105 ± 0.000155831 (W 0.00016186) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0560545 ± 0.000923883 (W 0.00027052) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.139452 ± 0.000930081 (W 0.000986922) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.1566 ± 0.00143515 (W 0.00166828) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.459017 ± 0.00247078 (W 0.00318847) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.54183 ± 0.00325515 (W 0.00249081) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.64739 ± 0.00145641 (W 0.00365301) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.08882 ± 0.0125019 (W 0.00494782) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.31283 ± 0.00664429 (W 0.00292838) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.32164 ± 0.00625433 (W 0.00405373) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.38799 ± 0.00984958 (W 0.0100491) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.40992 ± 0.0317312 (W 0.0144558) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.145 ± 0.0571412 (W 0.0242659) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 25.312 ± 0.0621087 (W 0.0363541) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.578828 ± 0.0049921 (W 0.00292386) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.716773 ± 0.00360121 (W 0.00340969) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.98797 ± 0.0140914 (W 0.00802105) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.73789 ± 0.0158407 (W 0.00926109) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.8965 ± 0.101553 (W 0.0227068) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 17.0571 ± 0.055591 (W 0.0375483) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.180545 ± 0.000879742 (W 0.00108007) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.18116 ± 0.0010678 (W 0.00168583) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00894 ± 0.00583177 (W 0.00336713) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01146 ± 0.00584297 (W 0.00399472) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.90045 ± 0.0171936 (W 0.0271191) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.90568 ± 0.0223223 (W 0.0278817) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.23091 ± 0.00117105 (W 0.00157231) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.347861 ± 0.00282144 (W 0.0027533) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.807363 ± 0.00343355 (W 0.00277635) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.54274 ± 0.0127029 (W 0.00443525) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.36554 ± 0.0485611 (W 0.0133262) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.73618 ± 0.0579272 (W 0.0236941) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.592655 ± 0.0276335 (W 0.00292889) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.809495 ± 0.00469997 (W 0.0054071) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.12299 ± 0.00866993 (W 0.00448517) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.28299 ± 0.0166192 (W 0.0125454) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.6771 ± 0.0389512 (W 0.0184103) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 16.58 ± 0.0649043 (W 0.0223631) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.387384 ± 0.000537563 (W 0.00255908) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.33365 ± 0.00322431 (W 0.0027345) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.13198 ± 0.107739 (W 0.00777172) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.200186 ± 0.000887354 (W 0.00112788) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.352525 ± 0.00205432 (W 0.0029129) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.776559 ± 0.00197793 (W 0.00323476) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.55819 ± 0.00688928 (W 0.00503981) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.34409 ± 0.00972383 (W 0.00953135) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.75004 ± 0.0824002 (W 0.0266744) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.50584 ± 0.00965769 (W 0.00548379) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.52251 ± 0.00869693 (W 0.0113896) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.6709 ± 0.0298711 (W 0.0155594) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 10.7552 ± 0.0182022 (W 0.0168148) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.3501 ± 0.174003 (W 0.116759) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.4907 ± 0.163269 (W 0.0661749) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.26091 ± 0.00694455 (W 0.00509084) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.25755 ± 0.00627206 (W 0.00348346) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.68766 ± 0.013431 (W 0.00903979) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.69305 ± 0.0112956 (W 0.0129955) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 26.0553 ± 0.113842 (W 0.0429167) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 26.2057 ± 0.108069 (W 0.0454131) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009192 ± 6.60316e-05 (W 6.5136e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0212255 ± 0.000103454 (W 9.5382e-05) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.050894 ± 0.000296466 (W 0.000277858) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.167367 ± 0.000526137 (W 0.00147886) | 0.169986 ± 0.000729547 (W 0.00132572) | · |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.178136 ± 0.000695837 (W 0.00136251) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.0785 ± 0.00310111 (W 0.00170882) | 1.21788 ± 0.00432422 (W 0.00364742) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 1.11792 ± 0.00468948 (W 0.00452766) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.33018 ± 0.00764834 (W 0.0150386) | 8.69207 ± 0.030108 (W 0.0135179) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 7.48659 ± 0.0150844 (W 0.00888952) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.5628 ± 0.0138892 (W 0.00341171) | 3.43734 ± 0.0123976 (W 0.00744877) | · |  |
| `eigh/n128/b1 [api-inclusive]` | 3.54941 ± 0.00917956 (W 0.00654987) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 18.0142 ± 0.0552458 (W 0.037059) | 20.2823 ± 0.0854811 (W 0.0181116) | 🔵 faer-fork 1.13× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 18.0301 ± 0.058964 (W 0.0126284) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 106.633 ± 0.398094 (W 0.113722) | 142.701 ± 0.809146 (W 0.379613) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 106.564 ± 0.455748 (W 0.121934) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.79242 ± 0.0090979 (W 0.0052397) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.897999 ± 0.0114605 (W 0.00742015) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 5.96394 ± 0.0120158 (W 0.0160279) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.41272 ± 0.0221867 (W 0.0141717) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 42.7501 ± 0.192281 (W 0.0330352) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 42.6678 ± 0.102738 (W 0.0970218) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.03662 ± 0.00549552 (W 0.00126051) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.03477 ± 0.00469323 (W 0.00148674) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.42927 ± 0.0274134 (W 0.00892636) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.43371 ± 0.0205773 (W 0.00548151) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.4595 ± 0.154686 (W 0.242836) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.2496 ± 0.283672 (W 0.162858) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.283598 ± 0.00171364 (W 0.00176974) | 0.327845 ± 0.00138607 (W 0.00169783) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.40984 ± 0.00248385 (W 0.00228109) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.86854 ± 0.0105958 (W 0.00270137) | 2.23246 ± 0.00671761 (W 0.00509597) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 2.51266 ± 0.00844635 (W 0.00668865) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.2193 ± 0.0457996 (W 0.0217076) | 15.77 ± 0.0336653 (W 0.0227663) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 16.7495 ± 0.0489976 (W 0.0446947) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.49065 ± 0.00354964 (W 0.00237408) | 1.94788 ± 0.00897764 (W 0.00630685) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 1.68569 ± 0.0053678 (W 0.00307517) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 9.8759 ± 0.0399806 (W 0.0146853) | 13.2196 ± 0.0555493 (W 0.0146031) | 🔵 faer-fork 1.34× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 10.8155 ± 0.0344959 (W 0.0132484) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 67.8672 ± 0.256296 (W 0.0960366) | 88.6827 ± 0.314102 (W 0.123416) | 🔵 faer-fork 1.31× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 72.7187 ± 0.241799 (W 0.0858395) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.710912 ± 0.00297914 (W 0.00217748) | 0.717183 ± 0.00426548 (W 0.00192929) | · |  |
| `qr_factor/n256/b1` | 4.37264 ± 0.01235 (W 0.00677406) | 5.11138 ± 0.0116792 (W 0.00585606) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 28.2287 ± 0.140395 (W 0.0153304) | 34.9085 ± 0.0843595 (W 0.029197) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.294452 ± 0.00244046 (W 0.00231377) | 0.345234 ± 0.00127241 (W 0.00227545) | 🔵 faer-fork 1.17× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.417146 ± 0.00149107 (W 0.00333326) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.90058 ± 0.00813876 (W 0.00461368) | 2.34644 ± 0.0135987 (W 0.00485652) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.96374 ± 0.00415654 (W 0.00462632) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.3121 ± 0.0819282 (W 0.0170427) | 16.0713 ± 0.0469799 (W 0.0163513) | 🔵 faer-fork 1.21× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 13.952 ± 0.0393654 (W 0.0427591) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.75092 ± 0.0237769 (W 0.00755007) | 5.98361 ± 0.010835 (W 0.0102632) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.74485 ± 0.0120946 (W 0.00705769) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.1161 ± 0.0916704 (W 0.0771171) | 37.5727 ± 0.197945 (W 0.0513637) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 29.232 ± 0.0829071 (W 0.054651) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 192.786 ± 0.844936 (W 0.609137) | 237.502 ± 0.630483 (W 0.392973) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 193.197 ± 0.581101 (W 0.413041) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.57158 ± 0.00538402 (W 0.0040577) | 2.51693 ± 0.00810698 (W 0.00572364) | 🔵 faer-fork 1.60× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.57132 ± 0.00767399 (W 0.00314918) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.1774 ± 0.0462764 (W 0.0540741) | 17.1448 ± 0.0780241 (W 0.024317) | 🔵 faer-fork 1.53× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 11.1561 ± 0.04422 (W 0.059464) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 82.1563 ± 0.507352 (W 0.28161) | 103.191 ± 0.827777 (W 0.325754) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 82.018 ± 0.377969 (W 0.36053) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007339 ± 8.53089e-05 (W 6.43698e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0217795 ± 0.000253648 (W 0.000148793) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.085429 ± 0.00108336 (W 0.00137578) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.133609 ± 0.00150275 (W 0.00214353) | 0.116121 ± 0.000855888 (W 0.000962598) | 🔴 faer-fork 1.15× slower than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.143575 ± 0.00227953 (W 0.00160328) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.502378 ± 0.00211671 (W 0.00186299) | 0.545238 ± 0.00313712 (W 0.00279769) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.568828 ± 0.00251752 (W 0.00272159) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.29178 ± 0.0111622 (W 0.00484625) | 3.25546 ± 0.00641795 (W 0.0107142) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.5881 ± 0.0114045 (W 0.00684681) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.93039 ± 0.0078618 (W 0.00489387) | 2.73699 ± 0.0131398 (W 0.0048484) | 🔵 faer-fork 1.42× faster than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.93497 ± 0.00780272 (W 0.00378583) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.84 ± 0.0154777 (W 0.0112795) | 12.1745 ± 0.042015 (W 0.0137306) | 🔵 faer-fork 1.55× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 7.86384 ± 0.017871 (W 0.0100374) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.6 ± 0.0666072 (W 0.0343804) | 55.7171 ± 0.119676 (W 0.0414481) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 39.719 ± 0.0793461 (W 0.0415517) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.582919 ± 0.00339253 (W 0.00375463) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.702198 ± 0.00577177 (W 0.0042015) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.31145 ± 0.0148803 (W 0.012246) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.88508 ± 0.0277184 (W 0.00882134) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 16.954 ± 0.0594297 (W 0.0221319) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 19.5851 ± 0.114729 (W 0.0407837) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.284672 ± 0.001065 (W 0.0015698) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.285753 ± 0.000540121 (W 0.00130112) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.94211 ± 0.00396259 (W 0.00434265) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.9441 ± 0.00635105 (W 0.00410949) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6412 ± 0.0316671 (W 0.0259206) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6266 ± 0.0702248 (W 0.0235558) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.198774 ± 0.00312553 (W 0.00293368) | 0.214395 ± 0.000885237 (W 0.00121769) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.333391 ± 0.000794749 (W 0.00431116) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.896977 ± 0.0100834 (W 0.0111504) | 1.16184 ± 0.00547089 (W 0.00268296) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.58012 ± 0.00560805 (W 0.00643217) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.67061 ± 0.0394945 (W 0.0436142) | 5.08388 ± 0.0185474 (W 0.00473022) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 8.20115 ± 0.0646746 (W 0.0448822) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.686099 ± 0.00216731 (W 0.00288706) | 1.32846 ± 0.00651843 (W 0.00452143) | 🔵 faer-fork 1.94× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.90816 ± 0.00510717 (W 0.00303897) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.0237 ± 0.0054477 (W 0.00413603) | 7.05691 ± 0.0217825 (W 0.0117376) | 🔵 faer-fork 2.33× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 4.04622 ± 0.0459819 (W 0.011439) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 19.1458 ± 0.0879016 (W 0.0155229) | 36.3099 ± 0.12933 (W 0.0421459) | 🔵 faer-fork 1.90× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 24.2471 ± 0.0510464 (W 0.252544) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.413685 ± 0.00198197 (W 0.0018004) | 0.630128 ± 0.00508714 (W 0.00222059) | 🔵 faer-fork 1.52× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 1.55395 ± 0.00429096 (W 0.00326956) | 3.0091 ± 0.0154647 (W 0.00686691) | 🔵 faer-fork 1.94× faster than `lapack-openblas` |  |
| `qr_factor/n512/b1` | 8.90986 ± 0.0260629 (W 0.0106966) | 14.973 ± 0.0566295 (W 0.0202225) | 🔵 faer-fork 1.68× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.206315 ± 0.00250645 (W 0.00265445) | 0.209625 ± 0.00122828 (W 0.00190648) | · |  |
| `solve/n128/b1 [api-inclusive]` | 0.339033 ± 0.00378687 (W 0.00371903) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.900805 ± 0.0112453 (W 0.00912236) | 1.13624 ± 0.00534712 (W 0.00298609) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.56432 ± 0.0224359 (W 0.00716866) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.65636 ± 0.041061 (W 0.0470653) | 5.05767 ± 0.0171281 (W 0.00937923) | 🔵 faer-fork 1.09× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 8.23723 ± 0.0317917 (W 0.0381579) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.67118 ± 0.0115878 (W 0.005151) | 5.15769 ± 0.0115832 (W 0.0087988) | 🔵 faer-fork 1.93× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.69269 ± 0.0103963 (W 0.00602365) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.584 ± 0.0237255 (W 0.0194623) | 24.3671 ± 0.0969359 (W 0.0320528) | 🔵 faer-fork 1.94× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 12.6485 ± 0.0318882 (W 0.0307418) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 67.6101 ± 0.193699 (W 0.219619) | 119.599 ± 0.549334 (W 0.568915) | 🔵 faer-fork 1.77× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 67.7742 ± 0.128081 (W 0.0754064) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.23752 ± 0.00528683 (W 0.00250016) | 2.68557 ± 0.009181 (W 0.00641443) | 🔵 faer-fork 2.17× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.2347 ± 0.00358891 (W 0.00338114) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.03762 ± 0.0316164 (W 0.0114235) | 13.1292 ± 0.0468502 (W 0.0267578) | 🔵 faer-fork 2.17× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.06212 ± 0.0235189 (W 0.0109705) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 32.626 ± 0.274389 (W 0.0551128) | 58.0679 ± 0.159807 (W 0.0674188) | 🔵 faer-fork 1.78× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 32.7983 ± 0.221924 (W 0.0450605) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.008405 ± 4.73737e-05 (W 5.06183e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0212795 ± 0.000474699 (W 0.000194556) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.056385 ± 0.000469119 (W 0.000288019) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.130741 ± 0.00138915 (W 0.00137426) | 0.143287 ± 0.00129799 (W 0.00162893) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.144945 ± 0.00115371 (W 0.00141983) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.428503 ± 0.00341184 (W 0.00327394) | 0.478431 ± 0.00179821 (W 0.00269199) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.502282 ± 0.00197127 (W 0.00274114) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.57269 ± 0.00339686 (W 0.0045855) | 2.07942 ± 0.00746503 (W 0.00799762) | 🔵 faer-fork 1.32× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.91274 ± 0.00313873 (W 0.00446959) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.30321 ± 0.00203197 (W 0.00393264) | 2.61972 ± 0.0114782 (W 0.00625403) | 🔵 faer-fork 2.01× faster than `lapack-openblas` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.30842 ± 0.00417225 (W 0.00351896) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.32995 ± 0.00700149 (W 0.00522467) | 11.0405 ± 0.0164787 (W 0.0117601) | 🔵 faer-fork 2.07× faster than `lapack-openblas` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.35509 ± 0.011126 (W 0.00740216) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.9043 ± 0.0456722 (W 0.0257391) | 43.8139 ± 0.0641456 (W 0.0243439) | 🔵 faer-fork 1.76× faster than `lapack-openblas` |  |
| `eigh/n512/b1 [api-inclusive]` | 24.9772 ± 0.0506399 (W 0.0163602) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.571915 ± 0.00581212 (W 0.00386148) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.687871 ± 0.00332909 (W 0.00474113) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.92331 ± 0.0194047 (W 0.00831585) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.57465 ± 0.060246 (W 0.0128108) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 13.0762 ± 0.0961656 (W 0.0298914) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 16.0047 ± 0.0613903 (W 0.0347238) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.178992 ± 0.00111397 (W 0.000802818) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.180532 ± 0.000870592 (W 0.00133473) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.01778 ± 0.0032831 (W 0.00334314) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01911 ± 0.00308366 (W 0.00255652) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 8.03654 ± 0.103514 (W 0.0926023) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 8.04089 ± 0.0976977 (W 0.10652) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.187059 ± 0.000968361 (W 0.00111745) | 0.201728 ± 0.00096061 (W 0.00212806) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.321616 ± 0.00204674 (W 0.00191873) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.738044 ± 0.00276732 (W 0.00260603) | 0.794914 ± 0.00141346 (W 0.00244737) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.40483 ± 0.00740296 (W 0.00658779) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.16174 ± 0.00852837 (W 0.00565501) | 2.82973 ± 0.00838707 (W 0.00435206) | 🔴 faer-fork 1.12× slower than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 6.68662 ± 0.0380854 (W 0.0217326) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.545017 ± 0.00247846 (W 0.00221842) | 1.2544 ± 0.00456244 (W 0.0044252) | 🔵 faer-fork 2.30× faster than `lapack-openblas` |  |
| `qr/n128/b1 [api-inclusive]` | 0.780887 ± 0.00235294 (W 0.00383264) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.06232 ± 0.00524031 (W 0.00380639) | 5.98295 ± 0.0136552 (W 0.0197442) | 🔵 faer-fork 2.90× faster than `lapack-openblas` |  |
| `qr/n256/b1 [api-inclusive]` | 3.12908 ± 0.0160125 (W 0.010643) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.1158 ± 0.0369352 (W 0.0115775) | 27.4745 ± 0.136472 (W 0.0576571) | 🔵 faer-fork 2.72× faster than `lapack-openblas` |  |
| `qr/n512/b1 [api-inclusive]` | 15.4707 ± 0.0818029 (W 0.038347) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.364716 ± 0.00145165 (W 0.00208545) | 0.631192 ± 0.00415299 (W 0.00290278) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `qr_factor/n256/b1` | 1.27563 ± 0.177051 (W 0.00629197) | 2.71774 ± 0.0118184 (W 0.00751046) | 🔵 faer-fork 2.13× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr_factor/n512/b1` | 4.80627 ± 0.0141285 (W 0.00397376) | 11.8601 ± 0.0278599 (W 0.0266377) | 🔵 faer-fork 2.47× faster than `lapack-openblas` |  |
| `solve/n128/b1` | 0.192598 ± 0.000887091 (W 0.00158857) | 0.200599 ± 0.0010548 (W 0.00222244) | · |  |
| `solve/n128/b1 [api-inclusive]` | 0.324918 ± 0.0015082 (W 0.00284243) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.736973 ± 0.00320982 (W 0.00253079) | 0.775643 ± 0.00410701 (W 0.00331224) | 🔵 faer-fork 1.05× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.40998 ± 0.00809625 (W 0.00457469) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.15824 ± 0.0105229 (W 0.00778063) | 2.79099 ± 0.01142 (W 0.00465279) | 🔴 faer-fork 1.13× slower than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 6.65166 ± 0.050963 (W 0.0156491) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.47518 ± 0.0141493 (W 0.00745641) | 5.12453 ± 0.026307 (W 0.00972496) | 🔵 faer-fork 2.07× faster than `lapack-openblas` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.49397 ± 0.00802566 (W 0.00437066) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.462 ± 0.0287687 (W 0.0157923) | 22.8417 ± 0.0385636 (W 0.0279492) | 🔵 faer-fork 2.18× faster than `lapack-openblas` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 10.5059 ± 0.0338666 (W 0.0210552) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 48.2859 ± 0.0952703 (W 0.255719) | 103.277 ± 0.389441 (W 0.0922961) | 🔵 faer-fork 2.14× faster than `lapack-openblas` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 48.6246 ± 0.180402 (W 0.142205) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.23823 ± 0.0053644 (W 0.00334756) | 2.78435 ± 0.00747731 (W 0.00909592) | 🔵 faer-fork 2.25× faster than `lapack-openblas` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.24063 ± 0.00340295 (W 0.00305718) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.54134 ± 0.018017 (W 0.00928356) | 13.0173 ± 0.058274 (W 0.0223163) | 🔵 faer-fork 2.35× faster than `lapack-openblas` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.54509 ± 0.0212726 (W 0.0121664) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 25.1474 ± 0.101315 (W 0.0532762) | 52.3378 ± 0.131743 (W 0.102889) | 🔵 faer-fork 2.08× faster than `lapack-openblas` |  |
| `svd_values/n512/b1 [api-inclusive]` | 25.2555 ± 0.147861 (W 0.0264983) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0093575 ± 4.09178e-05 (W 5.81111e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0229575 ± 0.000189753 (W 0.000199145) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.058358 ± 0.000265521 (W 0.000448925) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.048005 ± 0.00148058 (W 0.000149613) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.040474 ± 0.00167551 (W 0.000704071) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.219279 ± 0.00383753 (W 0.00182795) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.23165 ± 0.00426883 (W 0.00295196) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.3915 ± 0.00853137 (W 0.00698054) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.31827 ± 0.0074712 (W 0.0025227) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.74173 ± 0.00841662 (W 0.00279797) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.74051 ± 0.00895306 (W 0.0036778) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.54856 ± 0.0231696 (W 0.0091132) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.5593 ± 0.0244935 (W 0.00592983) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.3389 ± 0.148426 (W 0.0294061) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.4182 ± 0.142449 (W 0.0290931) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.262614 ± 0.00192644 (W 0.0018467) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.239772 ± 0.00122562 (W 0.00154744) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.44528 ± 0.00668835 (W 0.00618375) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.46288 ± 0.00623891 (W 0.00442076) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.0697 ± 0.0803692 (W 0.0210418) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.2331 ± 0.0511961 (W 0.0209748) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.148328 ± 0.000779057 (W 0.00129509) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.147696 ± 0.000939082 (W 0.00118191) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.989403 ± 0.00447858 (W 0.00109187) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.989444 ± 0.00725036 (W 0.00120236) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.37948 ± 0.0401883 (W 0.010719) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.40218 ± 0.0392404 (W 0.0197764) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0776465 ± 0.00080562 (W 0.000226499) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.099422 ± 0.00100765 (W 0.000517482) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.365459 ± 0.00130343 (W 0.00209818) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.585452 ± 0.00245712 (W 0.00283485) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.10988 ± 0.00754952 (W 0.00209294) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.26066 ± 0.0187813 (W 0.00570848) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.354202 ± 0.00174721 (W 0.00189872) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.458929 ± 0.00280379 (W 0.00293581) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.80361 ± 0.00701091 (W 0.00269212) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.92054 ± 0.00738466 (W 0.00360765) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.4852 ± 0.0515713 (W 0.0168312) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 10.9875 ± 0.04696 (W 0.086877) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.18838 ± 0.000772108 (W 0.00182576) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.05511 ± 0.00279699 (W 0.00203888) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.22506 ± 0.019521 (W 0.0088711) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.070745 ± 0.000742349 (W 0.000208871) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.10285 ± 0.00100461 (W 0.00125562) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.334096 ± 0.00166534 (W 0.00161965) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.470711 ± 0.00221134 (W 0.00272572) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.97162 ± 0.0104451 (W 0.0022949) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.69069 ± 0.0220793 (W 0.00612864) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.14792 ± 0.00887601 (W 0.0034316) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.17017 ± 0.00697825 (W 0.00458374) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.20844 ± 0.0390565 (W 0.00820476) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.26633 ± 0.0390595 (W 0.00972148) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.8194 ± 0.203855 (W 0.213452) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.8675 ± 0.139403 (W 0.0436317) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.660733 ± 0.0035881 (W 0.00234546) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.661233 ± 0.0040753 (W 0.00210005) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.34497 ± 0.0122289 (W 0.00981562) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.37264 ± 0.0139205 (W 0.00711838) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 20.0512 ± 0.106733 (W 0.0242701) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 20.0889 ± 0.132402 (W 0.0196882) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004769 ± 3.15621e-05 (W 5.6242e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.011582 ± 6.46759e-05 (W 6.32419e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030289 ± 0.000217469 (W 0.000176982) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0445175 ± 0.00201862 (W 0.000372828) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.037941 ± 0.000206664 (W 0.000199237) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.149003 ± 0.00379397 (W 0.00100701) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.171619 ± 0.0012219 (W 0.00216695) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.668834 ± 0.00901195 (W 0.00464953) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.697046 ± 0.00440908 (W 0.00231008) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.04633 ± 0.00364409 (W 0.0015772) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.04897 ± 0.00320665 (W 0.00256864) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.72565 ± 0.00871999 (W 0.00513129) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.74183 ± 0.0120654 (W 0.00629296) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.2668 ± 0.0314485 (W 0.013129) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.1327 ± 0.0255741 (W 0.0232725) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.234635 ± 0.00223743 (W 0.00183253) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.246806 ± 0.00232555 (W 0.00189759) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.11639 ± 0.00792841 (W 0.00473589) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.27047 ± 0.00702081 (W 0.00451464) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.31922 ± 0.0247642 (W 0.0235475) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 8.12123 ± 0.0341175 (W 0.0179352) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.054551 ± 0.000396413 (W 0.000715574) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0510235 ± 0.000331332 (W 0.000530092) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.277679 ± 0.00165291 (W 0.00198538) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.278705 ± 0.00206373 (W 0.00179093) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.93106 ± 0.0029372 (W 0.00684042) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.93253 ± 0.00894311 (W 0.00390886) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0964585 ± 0.00100428 (W 0.00154149) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.098691 ± 0.000776823 (W 0.000885187) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.295742 ± 0.00471059 (W 0.00308255) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.466146 ± 0.00300097 (W 0.00355002) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.23981 ± 0.00563108 (W 0.0101339) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.03962 ± 0.0113221 (W 0.0116673) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.315222 ± 0.00163649 (W 0.00348914) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.333098 ± 0.000960587 (W 0.00276483) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.767264 ± 0.00279192 (W 0.00342101) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.05883 ± 0.00473025 (W 0.00385165) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.6484 ± 0.0196032 (W 0.00691188) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.86389 ± 0.0339997 (W 0.0117839) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.200692 ± 0.00154737 (W 0.00129891) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.540666 ± 0.00220266 (W 0.00139817) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.2531 ± 0.00870038 (W 0.0060334) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0779505 ± 0.000970751 (W 0.00149669) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0995645 ± 0.000765592 (W 0.000694632) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.299189 ± 0.00258963 (W 0.00369341) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.47595 ± 0.00144372 (W 0.00339782) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.21527 ± 0.00425315 (W 0.0084904) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.03136 ± 0.00717531 (W 0.0182825) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.50674 ± 0.00616936 (W 0.00327091) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.50952 ± 0.00721422 (W 0.00389479) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.81392 ± 0.0175045 (W 0.0104442) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.81684 ± 0.0196067 (W 0.00963385) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.7144 ± 0.068488 (W 0.0328173) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 24.7997 ± 0.046741 (W 0.0293217) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.693333 ± 0.00189937 (W 0.00196532) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.693438 ± 0.00542912 (W 0.00219896) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.98914 ± 0.00843207 (W 0.00738112) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.99951 ± 0.011886 (W 0.00724668) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 12.3475 ± 0.0232103 (W 0.0144016) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 12.3717 ± 0.0376155 (W 0.0216851) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0057505 ± 3.91787e-05 (W 4.26897e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0134795 ± 0.000127908 (W 5.62966e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030742 ± 0.000265474 (W 0.000131042) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0438995 ± 0.00122827 (W 0.000343381) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.0381345 ± 0.00122539 (W 0.000297594) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.140284 ± 0.0034233 (W 0.000708041) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.16499 ± 0.00302898 (W 0.00209064) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.571495 ± 0.00773747 (W 0.00762165) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.603084 ± 0.00641006 (W 0.00310757) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.743287 ± 0.00284522 (W 0.00349511) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.746061 ± 0.00276145 (W 0.00226961) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.77043 ± 0.00866365 (W 0.00628672) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.78342 ± 0.00511293 (W 0.0059262) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.6945 ± 0.00714141 (W 0.00626541) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.6692 ± 0.0349823 (W 0.014766) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.237945 ± 0.00165197 (W 0.0014844) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.261305 ± 0.00181394 (W 0.00360268) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.07907 ± 0.00313778 (W 0.00670637) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.25888 ± 0.00570127 (W 0.00549608) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.69315 ± 0.0115953 (W 0.0120208) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.64024 ± 0.0242182 (W 0.0195344) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0367825 ± 0.000506007 (W 0.000740535) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0409725 ± 0.000568683 (W 0.00153333) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.161637 ± 0.000575042 (W 0.00200233) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.164696 ± 0.0139925 (W 0.00227016) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.03103 ± 0.0055698 (W 0.00601041) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.04206 ± 0.00852135 (W 0.00736738) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.108082 ± 0.00131046 (W 0.00343609) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.09194 ± 0.000304808 (W 0.000932442) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.251879 ± 0.00081161 (W 0.00203799) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.454 ± 0.00369418 (W 0.00282744) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.979621 ± 0.00184281 (W 0.00322355) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.86563 ± 0.0194955 (W 0.00480403) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.29928 ± 0.0033538 (W 0.00233709) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.318671 ± 0.00222953 (W 0.00302313) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.671221 ± 0.00212017 (W 0.00315349) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.985523 ± 0.00868844 (W 0.00582577) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.24622 ± 0.010178 (W 0.00440046) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.64604 ± 0.0187391 (W 0.0219369) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.188279 ± 0.00103424 (W 0.00131704) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.509738 ± 0.00316441 (W 0.00244435) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.46796 ± 0.00395124 (W 0.00517526) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.079021 ± 0.000614807 (W 0.00132756) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0995045 ± 0.000346555 (W 0.000548191) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.260497 ± 0.00105335 (W 0.00181212) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.462486 ± 0.00449817 (W 0.00248041) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.937499 ± 0.0035895 (W 0.00539212) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.84989 ± 0.0103144 (W 0.00862559) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.50036 ± 0.00470134 (W 0.0037811) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.50167 ± 0.00680684 (W 0.00406407) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.45139 ± 0.0100565 (W 0.00628862) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.45703 ± 0.0182285 (W 0.00792687) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 21.3158 ± 0.0416607 (W 0.0265448) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 21.4185 ± 0.0579624 (W 0.0282866) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.711831 ± 0.00435526 (W 0.00266546) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.713399 ± 0.00567398 (W 0.00198834) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.98092 ± 0.0117731 (W 0.00871117) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.97969 ± 0.0151418 (W 0.0105296) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.3176 ± 0.0482445 (W 0.0178949) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.3311 ± 0.0432698 (W 0.017756) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005961 ± 5.60952e-05 (W 5.81571e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.014076 ± 0.000106298 (W 7.9572e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0311725 ± 0.000114006 (W 0.000161693) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.03444 ± 0.000200369 (W 5.94464e-05) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.040886 ± 0.000359098 (W 0.000724207) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.177128 ± 0.00116021 (W 0.00126681) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.200006 ± 0.00137041 (W 0.00147728) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.08467 ± 0.00504413 (W 0.00305684) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.17272 ± 0.00293504 (W 0.00242598) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.73945 ± 0.00635714 (W 0.00268242) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.74312 ± 0.00803457 (W 0.00492083) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.51807 ± 0.0256718 (W 0.00893234) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.53566 ± 0.0459508 (W 0.00739186) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.3625 ± 0.18834 (W 0.0164643) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.2119 ± 0.130709 (W 0.0447895) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.209414 ± 0.00172163 (W 0.00151753) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.214475 ± 0.00134686 (W 0.00159878) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.31006 ± 0.0069762 (W 0.00730419) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.36258 ± 0.0146713 (W 0.00681739) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.0285 ± 0.0735899 (W 0.0182368) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.862 ± 0.0372224 (W 0.0317739) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.150848 ± 0.000729983 (W 0.00121059) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.150955 ± 0.000706006 (W 0.000643654) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00466 ± 0.00329879 (W 0.00226697) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00607 ± 0.00357355 (W 0.00172122) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.4853 ± 0.0354646 (W 0.0119963) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.49501 ± 0.0363715 (W 0.0171282) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0720995 ± 0.000758974 (W 0.000802146) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.077688 ± 0.000431284 (W 0.0011084) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.358968 ± 0.0011777 (W 0.00205365) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.483275 ± 0.00337867 (W 0.0028031) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.21617 ± 0.00810294 (W 0.00253291) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.8093 ± 0.00914886 (W 0.00660236) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.39911 ± 0.00189441 (W 0.0020123) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.430333 ± 0.00386434 (W 0.00290275) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.60995 ± 0.0108472 (W 0.00226779) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.80143 ± 0.00994146 (W 0.00250207) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 9.57754 ± 0.0363344 (W 0.0093893) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 10.4607 ± 0.0415526 (W 0.00955919) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.220302 ± 0.00131916 (W 0.00136922) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.840689 ± 0.0057462 (W 0.00240535) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 4.35127 ± 0.0170303 (W 0.00707209) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0703615 ± 0.000600484 (W 0.000181818) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.075822 ± 0.00047247 (W 0.000406301) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.354145 ± 0.00190462 (W 0.00173997) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.368718 ± 0.00247451 (W 0.00182806) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.19036 ± 0.00946095 (W 0.00409585) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.25735 ± 0.00864004 (W 0.00646758) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.14216 ± 0.0085979 (W 0.00163238) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.17719 ± 0.0076426 (W 0.00267673) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.22292 ± 0.0356789 (W 0.00694275) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.24847 ± 0.0354689 (W 0.0095552) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.4285 ± 0.248401 (W 0.0500641) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.4307 ± 0.21902 (W 0.0730003) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.657147 ± 0.00146279 (W 0.00251416) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.655831 ± 0.00262546 (W 0.00161866) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.24193 ± 0.0189805 (W 0.00370744) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.27367 ± 0.0152372 (W 0.00558221) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.5832 ± 0.107877 (W 0.0250912) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.5167 ± 0.157843 (W 0.0141519) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0039475 ± 2.46225e-05 (W 5.20302e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.009993 ± 6.14025e-05 (W 7.57956e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0273105 ± 0.000248607 (W 0.00013047) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.031118 ± 0.00121674 (W 0.000160128) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.037809 ± 0.00104864 (W 0.00021577) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.115289 ± 0.001194 (W 0.00073579) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.141435 ± 0.000668902 (W 0.000527202) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.443885 ± 0.00157038 (W 0.00224844) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.553566 ± 0.00507082 (W 0.00298492) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.04212 ± 0.00399184 (W 0.00259069) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.04766 ± 0.00366898 (W 0.00182198) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.70263 ± 0.0107428 (W 0.00664624) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.71532 ± 0.0124469 (W 0.00754261) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.127 ± 0.101715 (W 0.0139832) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.004 ± 0.0118213 (W 0.0156324) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.215766 ± 0.00092406 (W 0.00187635) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.218943 ± 0.00158033 (W 0.00143069) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.073 ± 0.00715979 (W 0.00551758) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.16178 ± 0.00652034 (W 0.00612368) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.34542 ± 0.0151893 (W 0.014984) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.70829 ± 0.0280154 (W 0.0139411) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.055288 ± 0.00044379 (W 0.000446967) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0532535 ± 0.000352744 (W 0.000278237) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.286873 ± 0.00102397 (W 0.00223541) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.288555 ± 0.00103838 (W 0.00165859) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.9549 ± 0.00569734 (W 0.00477732) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.95499 ± 0.00425678 (W 0.00420872) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.061665 ± 0.000560524 (W 0.000810832) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0745845 ± 0.00162702 (W 0.000824729) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.221123 ± 0.00208592 (W 0.00224422) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.36024 ± 0.00205265 (W 0.0031976) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.995172 ± 0.00505316 (W 0.00429869) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.61474 ± 0.00571666 (W 0.00612336) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.262192 ± 0.00131854 (W 0.00137069) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.302071 ± 0.00246263 (W 0.00267722) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.703364 ± 0.00296851 (W 0.00168071) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.943905 ± 0.00371065 (W 0.00472492) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.3877 ± 0.0146284 (W 0.00412074) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.39007 ± 0.0392314 (W 0.0114676) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.180864 ± 0.00131058 (W 0.00144574) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.469209 ± 0.0034005 (W 0.00182789) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.98468 ± 0.00987044 (W 0.00400437) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.06498 ± 0.000642115 (W 0.000854564) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0716685 ± 0.000678518 (W 0.000816438) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.23024 ± 0.00199227 (W 0.00149128) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.367954 ± 0.0032726 (W 0.00336899) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.982065 ± 0.00430031 (W 0.00293825) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.60188 ± 0.00698654 (W 0.00392158) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.50956 ± 0.00418235 (W 0.00470317) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.51231 ± 0.00519065 (W 0.00243419) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.76048 ± 0.0199712 (W 0.0110693) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.7755 ± 0.0190537 (W 0.0110319) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.2255 ± 0.0277404 (W 0.0308323) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 24.252 ± 0.0560428 (W 0.0371781) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.689358 ± 0.00289939 (W 0.00258608) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.689453 ± 0.00325918 (W 0.00180816) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.90771 ± 0.0115334 (W 0.00942175) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.90865 ± 0.0131934 (W 0.00735556) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.8581 ± 0.0210473 (W 0.0167178) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.8599 ± 0.0297466 (W 0.0176771) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004799 ± 8.21381e-05 (W 6.66224e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0119375 ± 4.81913e-05 (W 5.70657e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030421 ± 0.000258728 (W 0.000427212) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0304665 ± 0.00182871 (W 0.000191366) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.0385155 ± 0.00128895 (W 0.000409111) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.108704 ± 0.000543133 (W 0.00151196) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.136641 ± 0.000454394 (W 0.000698359) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.34916 ± 0.0016975 (W 0.00230236) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.463705 ± 0.00148652 (W 0.00217688) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.742728 ± 0.00170929 (W 0.00208381) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.744115 ± 0.0028575 (W 0.00389522) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.74793 ± 0.00337726 (W 0.00348482) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.75994 ± 0.00704663 (W 0.00525216) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.5632 ± 0.0477785 (W 0.0105725) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.5383 ± 0.0169024 (W 0.0140932) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.231675 ± 0.00232186 (W 0.00165029) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.237528 ± 0.00309463 (W 0.00179175) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.0642 ± 0.010361 (W 0.00613279) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.16098 ± 0.00466649 (W 0.00735138) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.69377 ± 0.0158156 (W 0.015912) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.22282 ± 0.02604 (W 0.0208911) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0363295 ± 0.000298976 (W 0.000418128) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0389195 ± 0.000548258 (W 0.00075015) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.172799 ± 0.00103329 (W 0.00188627) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.171732 ± 0.00130137 (W 0.00171757) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.07005 ± 0.00688983 (W 0.00637434) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.07049 ± 0.00185785 (W 0.0053014) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.061058 ± 0.000688866 (W 0.000620102) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0679365 ± 0.00100985 (W 0.000613372) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.210583 ± 0.000887938 (W 0.00138767) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.34918 ± 0.0038483 (W 0.00276902) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.832717 ± 0.0018786 (W 0.00309041) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.44442 ± 0.00449576 (W 0.00552823) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.251346 ± 0.0030102 (W 0.0013693) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.294631 ± 0.00627623 (W 0.00395419) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 0.618783 ± 0.00119627 (W 0.00210647) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.865516 ± 0.00748037 (W 0.00470558) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.12781 ± 0.0105265 (W 0.00409961) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.14255 ± 0.0113294 (W 0.00885577) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.17484 ± 0.000502167 (W 0.00105215) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.467765 ± 0.00374624 (W 0.00161514) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.34222 ± 0.0173928 (W 0.00261667) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.065603 ± 0.000691015 (W 0.000648604) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.071456 ± 0.00106854 (W 0.000583379) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.221762 ± 0.000473474 (W 0.00144153) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.357289 ± 0.00438361 (W 0.0025919) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.809214 ± 0.00201655 (W 0.00308957) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.43597 ± 0.00786388 (W 0.00646856) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.50337 ± 0.00713894 (W 0.00330539) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.50841 ± 0.00540536 (W 0.00306628) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.38348 ± 0.0111815 (W 0.00643565) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.40961 ± 0.0251021 (W 0.0119931) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 20.9078 ± 0.0733296 (W 0.0385182) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 20.9882 ± 0.0492929 (W 0.0300396) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.707473 ± 0.00333464 (W 0.00269965) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.70772 ± 0.00318104 (W 0.00258017) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.87327 ± 0.00653653 (W 0.00613175) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.87591 ± 0.0167875 (W 0.0060421) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 10.8747 ± 0.0371167 (W 0.014071) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 10.8694 ± 0.0246877 (W 0.0146799) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0051545 ± 5.1976e-05 (W 5.58351e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.012844 ± 5.0044e-05 (W 8.03583e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.031414 ± 0.00034236 (W 0.000229605) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.212352 ± 0.000872193 (W 0.00145756) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.209021 ± 0.000870823 (W 0.00123515) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.906222 ± 0.00500225 (W 0.00247666) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.879408 ± 0.0034019 (W 0.00236845) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.95734 ± 0.0194388 (W 0.0070623) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.68397 ± 0.016772 (W 0.00590658) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.29199 ± 0.0113854 (W 0.00353877) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.29467 ± 0.00862176 (W 0.00621706) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.8449 ± 0.0384276 (W 0.0122448) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.8416 ± 0.0381029 (W 0.00971806) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.271 ± 0.247088 (W 0.0691674) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.4968 ± 0.154578 (W 0.0646868) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.455209 ± 0.00275513 (W 0.00244472) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.429879 ± 0.00159818 (W 0.00176739) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.54772 ± 0.017876 (W 0.00966035) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.69664 ± 0.0148441 (W 0.00754229) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.1751 ± 0.161451 (W 0.0453026) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 26.0337 ± 0.109371 (W 0.0515019) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.562191 ± 0.00189987 (W 0.00228508) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.561552 ± 0.00319702 (W 0.00182471) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.88128 ± 0.0129604 (W 0.0088122) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.87403 ± 0.0140177 (W 0.00540776) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 28.822 ± 0.091197 (W 0.027297) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 28.875 ± 0.115036 (W 0.0564838) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.179869 ± 0.000841948 (W 0.00172444) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.249609 ± 0.00107127 (W 0.00196444) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.04028 ± 0.00429741 (W 0.00234492) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.45805 ± 0.00436079 (W 0.00337318) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.04111 ± 0.0237358 (W 0.00920382) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 9.18293 ± 0.13744 (W 0.0153028) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.869956 ± 0.00605638 (W 0.00114874) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.986931 ± 0.0048676 (W 0.00214835) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.51057 ± 0.0203428 (W 0.00741967) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.80061 ± 0.0270909 (W 0.00443267) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 36.2785 ± 0.123556 (W 0.287777) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 37.8622 ± 0.134587 (W 0.0404268) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.427111 ± 0.00184395 (W 0.00179869) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.64445 ± 0.00896706 (W 0.00296283) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 15.7244 ± 0.0501178 (W 0.0187131) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.173944 ± 0.00159129 (W 0.001399) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.249754 ± 0.000872645 (W 0.00155104) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.01038 ± 0.0041193 (W 0.00522517) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.19606 ± 0.00445717 (W 0.00292341) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 6.9119 ± 0.0251295 (W 0.00931091) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.69646 ± 0.0320318 (W 0.0104547) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.10433 ± 0.00929499 (W 0.00246355) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.11819 ± 0.012054 (W 0.0048138) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 16.772 ± 0.0566807 (W 0.0481768) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 16.7738 ± 0.0655521 (W 0.0380157) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 105.696 ± 0.406503 (W 0.173462) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 105.73 ± 0.525502 (W 0.357404) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.05634 ± 0.00781783 (W 0.00218449) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.06285 ± 0.00495137 (W 0.0021803) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.30357 ± 0.0314969 (W 0.0524274) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.31177 ± 0.0250085 (W 0.0432431) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 44.3086 ± 0.180755 (W 0.127693) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 44.4305 ± 0.253617 (W 0.112152) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0062265 ± 3.45876e-05 (W 5.01537e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0169965 ± 6.85877e-05 (W 5.82227e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0539035 ± 0.000405738 (W 0.000264302) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.194679 ± 0.000847113 (W 0.00118321) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.194912 ± 0.000870225 (W 0.00124014) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.608066 ± 0.00134568 (W 0.00322062) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.61629 ± 0.00147741 (W 0.0017622) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.15959 ± 0.00775111 (W 0.00442867) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.1559 ± 0.0161332 (W 0.00770069) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.33007 ± 0.00381728 (W 0.00266465) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.33536 ± 0.00636601 (W 0.00277867) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.29982 ± 0.00808889 (W 0.00887743) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.29467 ± 0.0159099 (W 0.0107735) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.5333 ± 0.0531217 (W 0.0331291) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.5491 ± 0.0448477 (W 0.0191836) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.368723 ± 0.00226034 (W 0.00276739) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.382653 ± 0.0106333 (W 0.00333716) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 2.18031 ± 0.0151908 (W 0.00887518) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.47589 ± 0.0114752 (W 0.00508032) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.6162 ± 0.043465 (W 0.0208091) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 14.2326 ± 0.0753968 (W 0.0291122) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.160166 ± 0.000347318 (W 0.000696211) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.160234 ± 0.000161292 (W 0.000917119) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00818 ± 0.00321721 (W 0.00199112) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00864 ± 0.00341071 (W 0.00406487) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.35546 ± 0.00986883 (W 0.00746107) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.37075 ± 0.0124747 (W 0.0105978) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.172811 ± 0.00113562 (W 0.0018265) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.219127 ± 0.00113352 (W 0.00247239) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.644486 ± 0.00232966 (W 0.00385058) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.965577 ± 0.00387857 (W 0.00544402) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.01282 ± 0.0100873 (W 0.00822073) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.8192 ± 0.0301049 (W 0.010067) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.479164 ± 0.00196491 (W 0.00202897) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.572378 ± 0.00523702 (W 0.00354005) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.79097 ± 0.00212875 (W 0.00435703) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.31184 ± 0.00314782 (W 0.00902068) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.6502 ± 0.0449971 (W 0.00733292) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 13.0988 ± 0.115004 (W 0.0220051) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.308462 ± 0.00128538 (W 0.00177342) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.01455 ± 0.00265343 (W 0.00147002) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.34527 ± 0.0214752 (W 0.00915391) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.149031 ± 0.000926058 (W 0.00109987) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.181517 ± 0.00112346 (W 0.0016043) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.620684 ± 0.00335738 (W 0.00304898) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.964549 ± 0.00287803 (W 0.00431071) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.88786 ± 0.00892784 (W 0.0110659) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.67471 ± 0.0310065 (W 0.0192336) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.95788 ± 0.0087493 (W 0.00371214) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.97312 ± 0.0190424 (W 0.00595145) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.30623 ± 0.0244984 (W 0.0219824) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.3367 ± 0.0181789 (W 0.0119597) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 40.7678 ± 0.124672 (W 0.0689027) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 40.8384 ± 0.0904496 (W 0.0800179) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.951707 ± 0.00497497 (W 0.00340867) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.952087 ± 0.00718097 (W 0.00408852) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.14925 ± 0.00936911 (W 0.015819) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.13477 ± 0.0145127 (W 0.0153794) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.7498 ± 0.124268 (W 0.0602139) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.7077 ± 0.123514 (W 0.245555) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007329 ± 4.70815e-05 (W 4.4396e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0171965 ± 6.82954e-05 (W 9.76414e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0409955 ± 0.000409251 (W 0.000169522) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.191811 ± 0.000832804 (W 0.00117557) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.194561 ± 0.000523236 (W 0.00149211) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.581732 ± 0.00363894 (W 0.00264251) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.592075 ± 0.00251307 (W 0.00206865) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.72415 ± 0.00667859 (W 0.0033955) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.8196 ± 0.0017999 (W 0.00415918) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.982514 ± 0.00334462 (W 0.00145715) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.988554 ± 0.00135607 (W 0.00202836) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.9636 ± 0.00277991 (W 0.00663926) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.97305 ± 0.00423316 (W 0.00504559) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.6495 ± 0.0834366 (W 0.0192393) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.6989 ± 0.175095 (W 0.0341225) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.366422 ± 0.00381394 (W 0.00273782) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.43206 ± 0.00159568 (W 0.00282729) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.02459 ± 0.0106057 (W 0.00627386) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.36836 ± 0.0105003 (W 0.00744891) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.7827 ± 0.0419658 (W 0.0221809) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.6139 ± 0.0319087 (W 0.0176468) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.102756 ± 0.00825055 (W 0.001508) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.102752 ± 0.000506187 (W 0.000815356) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.527076 ± 0.00354781 (W 0.00227687) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.530821 ± 0.00166241 (W 0.00272498) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.85357 ± 0.01077 (W 0.0108126) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.85335 ± 0.00697356 (W 0.00793137) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.179173 ± 0.000979628 (W 0.00231216) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.214333 ± 0.00158331 (W 0.00183632) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.54528 ± 0.00156272 (W 0.00219472) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.901039 ± 0.00377962 (W 0.00253772) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.17552 ± 0.00432031 (W 0.00838348) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.1029 ± 0.01619 (W 0.0118944) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.410703 ± 0.00200026 (W 0.0022619) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.511917 ± 0.00461809 (W 0.00311337) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.31304 ± 0.00323178 (W 0.00277536) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.89028 ± 0.00483723 (W 0.00827764) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.78159 ± 0.0397707 (W 0.00732185) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.53522 ± 0.135267 (W 0.023134) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.273744 ± 0.00144604 (W 0.00128875) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.886185 ± 0.0941811 (W 0.00193668) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 2.97839 ± 0.00769408 (W 0.00747438) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.152799 ± 0.00124492 (W 0.00143028) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.176712 ± 0.000814738 (W 0.00109781) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.52773 ± 0.00355847 (W 0.00234824) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.902019 ± 0.00658928 (W 0.00340227) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.05707 ± 0.00700969 (W 0.00857063) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.00377 ± 0.0146415 (W 0.0156795) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.89566 ± 0.00589072 (W 0.00534299) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.89522 ± 0.0115593 (W 0.00353415) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.41055 ± 0.0156118 (W 0.0104441) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.44268 ± 0.0308756 (W 0.00633095) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 31.7071 ± 0.0543536 (W 0.027794) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 31.8343 ± 0.0614548 (W 0.0414474) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.979438 ± 0.00442509 (W 0.00477722) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.982893 ± 0.00317721 (W 0.00279012) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.03737 ± 0.0106497 (W 0.0111729) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.03084 ± 0.0147139 (W 0.010128) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.9347 ± 0.030442 (W 0.0403151) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.9259 ± 0.0350593 (W 0.0344894) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0073735 ± 0.000449303 (W 4.56824e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0169915 ± 0.000148531 (W 9.04521e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0391045 ± 0.00029141 (W 0.000186042) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.197652 ± 0.000652378 (W 0.00122464) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.202796 ± 0.000819619 (W 0.00118907) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.810416 ± 0.00302298 (W 0.00193096) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.831149 ± 0.00283212 (W 0.00211966) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.4148 ± 0.0139063 (W 0.00587194) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.50014 ± 0.0146287 (W 0.00788595) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.28452 ± 0.0075656 (W 0.002636) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.2898 ± 0.0101348 (W 0.00308623) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.8495 ± 0.0394288 (W 0.0104084) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.797 ± 0.0530131 (W 0.0154171) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.1731 ± 0.249789 (W 0.232296) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.2268 ± 0.272076 (W 0.0495715) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.402882 ± 0.00282488 (W 0.00178557) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.400495 ± 0.00136149 (W 0.00197149) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.36454 ± 0.0171692 (W 0.00795633) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.56971 ± 0.117701 (W 0.00931407) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1` | 24.9924 ± 0.0652001 (W 0.0262846) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 25.419 ± 0.0729489 (W 0.0694627) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.558307 ± 0.00182163 (W 0.0029677) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.554798 ± 0.00276645 (W 0.00179747) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.83437 ± 0.0156536 (W 0.00515002) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.84985 ± 0.0111382 (W 0.00815578) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 28.6913 ± 0.10425 (W 0.0359111) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 28.7733 ± 0.231332 (W 0.249443) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.173297 ± 0.000725995 (W 0.00137193) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.220686 ± 0.00149751 (W 0.0017356) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.05028 ± 0.004178 (W 0.00404831) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.33559 ± 0.00471959 (W 0.0029964) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.11272 ± 0.0249484 (W 0.00880507) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 8.55671 ± 0.0326167 (W 0.0233063) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.882949 ± 0.00766037 (W 0.00162666) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.960081 ± 0.00403563 (W 0.000917078) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.24796 ± 0.0217637 (W 0.00704804) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.70091 ± 0.0193205 (W 0.0058695) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 35.2901 ± 0.10946 (W 0.0232962) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 37.2886 ± 0.153503 (W 0.087518) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.437592 ± 0.00329618 (W 0.0020513) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.38251 ± 0.0100938 (W 0.00241883) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 14.8564 ± 0.056271 (W 0.0173023) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.174785 ± 0.000746959 (W 0.00207042) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.221674 ± 0.00111972 (W 0.00202062) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.03956 ± 0.00544958 (W 0.00542735) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.077 ± 0.00294648 (W 0.00472362) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 7.01922 ± 0.0286087 (W 0.0128843) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.17378 ± 0.0247742 (W 0.0117486) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.10793 ± 0.0114732 (W 0.00288356) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.10743 ± 0.0110147 (W 0.00927329) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 16.7021 ± 0.0552397 (W 0.051186) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 16.6633 ± 0.0747962 (W 0.0416143) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 104.71 ± 0.30628 (W 0.150145) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 105.112 ± 0.300975 (W 0.202553) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.03099 ± 0.0041673 (W 0.00142828) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.04484 ± 0.0050955 (W 0.00111504) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.17284 ± 0.0208831 (W 0.031157) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.17309 ± 0.0275386 (W 0.0410268) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 43.392 ± 0.231514 (W 0.264788) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 43.5426 ± 0.252529 (W 0.103908) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.006006 ± 6.05805e-05 (W 5.58735e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0154685 ± 0.000550402 (W 6.81413e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0515995 ± 0.000991835 (W 0.000336737) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.181952 ± 0.000738653 (W 0.000631035) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188527 ± 0.000429479 (W 0.00151457) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.537718 ± 0.00155264 (W 0.00237187) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.570425 ± 0.00279334 (W 0.00193405) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.81565 ± 0.00640815 (W 0.00319395) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.9569 ± 0.00956383 (W 0.00322332) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32223 ± 0.00402774 (W 0.00203061) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.3295 ± 0.00570961 (W 0.00235729) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.25324 ± 0.0190963 (W 0.00743153) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.25672 ± 0.0154645 (W 0.0164727) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.3611 ± 0.0468486 (W 0.0267113) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.3789 ± 0.0367007 (W 0.029572) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.350503 ± 0.00301414 (W 0.00283912) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.350026 ± 0.0014826 (W 0.00249539) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.12509 ± 0.00729533 (W 0.00727965) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.33101 ± 0.0153474 (W 0.00863256) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.5365 ± 0.0279517 (W 0.0193468) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.6806 ± 0.0245013 (W 0.0278849) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.158067 ± 0.000454525 (W 0.00109464) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.159211 ± 0.000407268 (W 0.00109343) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00787 ± 0.00517686 (W 0.00126518) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00832 ± 0.00319535 (W 0.00215826) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.36319 ± 0.0128646 (W 0.0111328) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.36787 ± 0.00787508 (W 0.00948739) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.133917 ± 0.000837169 (W 0.0013913) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.19088 ± 0.00128105 (W 0.00241133) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.549109 ± 0.00273658 (W 0.00384942) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.84541 ± 0.00499178 (W 0.00362673) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.7504 ± 0.0121392 (W 0.010041) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.19924 ± 0.020288 (W 0.012245) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.444631 ± 0.0010544 (W 0.00200679) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.544092 ± 0.00353767 (W 0.00236192) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.70145 ± 0.00566057 (W 0.00247743) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.19302 ± 0.0127493 (W 0.0117464) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.2908 ± 0.0214945 (W 0.0102826) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 12.477 ± 0.0348851 (W 0.0204266) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.284811 ± 0.00105225 (W 0.0015076) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.922688 ± 0.00511702 (W 0.00195994) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.02829 ± 0.0142645 (W 0.00838791) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.140458 ± 0.000820899 (W 0.000823102) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.151921 ± 0.000655641 (W 0.000832473) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.548325 ± 0.00362518 (W 0.00252645) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.845032 ± 0.00374286 (W 0.00327835) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.62254 ± 0.00672034 (W 0.00857371) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.08626 ± 0.0314761 (W 0.00917832) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.95449 ± 0.00401389 (W 0.00578884) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.96313 ± 0.00577759 (W 0.00365837) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.22144 ± 0.0274909 (W 0.0173432) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.2328 ± 0.0355749 (W 0.0146421) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 39.9164 ± 0.0941015 (W 0.0866964) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 39.9802 ± 0.0670087 (W 0.0632373) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.925197 ± 0.00396131 (W 0.00352724) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.92413 ± 0.00282188 (W 0.00287132) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.00659 ± 0.018579 (W 0.015437) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.01064 ± 0.0119878 (W 0.010721) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.8228 ± 0.0483033 (W 0.0881196) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.8278 ± 0.0511837 (W 0.0648461) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.006983 ± 0.000112523 (W 3.68852e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017147 ± 0.000363529 (W 0.000133478) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0417585 ± 0.000598375 (W 0.000402239) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.179697 ± 0.000635727 (W 0.00107376) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188536 ± 0.000614346 (W 0.000922945) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.508961 ± 0.00185015 (W 0.00187621) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.54593 ± 0.00210649 (W 0.00193894) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.46398 ± 0.00271368 (W 0.0021836) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.6256 ± 0.00313547 (W 0.00283174) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.977384 ± 0.00314385 (W 0.00207144) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.983941 ± 0.00374257 (W 0.00202274) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.91676 ± 0.0110977 (W 0.00370879) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.92634 ± 0.00432131 (W 0.00535857) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.4719 ± 0.240034 (W 0.0208309) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.4938 ± 0.143231 (W 0.0179808) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.356689 ± 0.00395172 (W 0.00213881) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.40457 ± 0.00537877 (W 0.00319368) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.97215 ± 0.00838426 (W 0.0070442) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.2307 ± 0.00659094 (W 0.00707098) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.7066 ± 0.0299544 (W 0.020585) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.042 ± 0.0629825 (W 0.021702) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.100485 ± 0.000565862 (W 0.000655314) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.10088 ± 0.0081341 (W 0.000829205) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.533675 ± 0.00156781 (W 0.00216638) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.53689 ± 0.00150523 (W 0.00319523) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.85739 ± 0.00787764 (W 0.00779588) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.86103 ± 0.00691083 (W 0.00594237) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.129633 ± 0.000890019 (W 0.000811227) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.187117 ± 0.00141375 (W 0.00132881) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.485707 ± 0.00198436 (W 0.00405899) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.783944 ± 0.00465498 (W 0.00413954) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.04481 ± 0.00825247 (W 0.00775791) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.50497 ± 0.0163645 (W 0.00784384) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.373109 ± 0.000955675 (W 0.00200056) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.483545 ± 0.0020534 (W 0.00221119) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.26299 ± 0.0037388 (W 0.00385956) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.76965 ± 0.00662148 (W 0.00623783) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.55199 ± 0.0117135 (W 0.00811498) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 7.83591 ± 0.0287408 (W 0.0195639) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.257813 ± 0.00140021 (W 0.00127982) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.830959 ± 0.00301353 (W 0.00152889) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.80197 ± 0.00974439 (W 0.00439441) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.138124 ± 0.000549716 (W 0.000942624) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.148538 ± 0.000813639 (W 0.000595042) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.479548 ± 0.000959303 (W 0.00269509) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.783167 ± 0.00349347 (W 0.00439709) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.91436 ± 0.00531615 (W 0.00697672) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.40738 ± 0.0162732 (W 0.0073381) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.88422 ± 0.00432919 (W 0.00346769) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.88814 ± 0.0050704 (W 0.00319314) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.32333 ± 0.0123088 (W 0.0177661) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.34239 ± 0.0153738 (W 0.0119072) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.9008 ± 0.0476786 (W 0.0411888) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.9981 ± 0.0368018 (W 0.0288208) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.959966 ± 0.00670515 (W 0.0033672) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.964177 ± 0.00276546 (W 0.00377948) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.90586 ± 0.00616405 (W 0.00770825) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.91081 ± 0.00914557 (W 0.012095) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.0291 ± 0.0567383 (W 0.212588) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.0627 ± 0.0645743 (W 0.0420512) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007384 ± 4.54137e-05 (W 4.7948e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0187545 ± 0.000256112 (W 0.000179841) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0454875 ± 0.000765658 (W 0.000352931) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `lu/128`, `lu/256`, `lu/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `solve/128`, `solve/256`, `solve/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `triangular_solve/128`, `triangular_solve/256`, `triangular_solve/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.956835 ± 0.00545824 (W 0.0029125) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.823239 ± 0.0059684 (W 0.00307733) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 52.9771 ± 0.294837 (W 0.620754) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 49.3704 ± 0.430732 (W 0.127631) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.553887 ± 0.00397311 (W 0.0021366) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.553226 ± 0.00511225 (W 0.0031546) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 34.9865 ± 0.165939 (W 0.05116) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.0229 ± 0.254993 (W 0.393137) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.324259 ± 0.00153485 (W 0.00234765) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.336066 ± 0.00787181 (W 0.00191366) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 12.7943 ± 0.239849 (W 0.0747483) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 13.5717 ± 0.0624354 (W 0.0415202) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.45084 ± 0.00503103 (W 0.00653302) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.69679 ± 0.00370482 (W 0.00287524) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 68.458 ± 0.2734 (W 0.419111) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 72.1139 ± 0.258389 (W 0.237129) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.928237 ± 0.00120245 (W 0.00230569) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 31.6453 ± 0.0662386 (W 0.232893) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.02189 ± 0.00807134 (W 0.00198746) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.02966 ± 0.00479545 (W 0.00467304) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 82.882 ± 0.272744 (W 0.418263) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 82.8742 ± 0.200623 (W 0.135817) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.00752 ± 0.00419128 (W 0.00521039) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01405 ± 0.00544419 (W 0.00569693) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 37.3853 ± 0.0631269 (W 0.0457576) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 37.4377 ± 0.044046 (W 0.0346979) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.570786 ± 0.00477621 (W 0.00409558) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.584909 ± 0.00396208 (W 0.00301585) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 16.1479 ± 0.115538 (W 0.0373259) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 16.7074 ± 0.0648867 (W 0.0372297) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.168092 ± 0.00263107 (W 0.00298579) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.169959 ± 0.00108631 (W 0.00258192) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.44947 ± 0.0107147 (W 0.0299499) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.46207 ± 0.0269097 (W 0.0193664) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.281707 ± 0.00243414 (W 0.001657) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.299381 ± 0.00521009 (W 0.00186946) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.29216 ± 0.0318648 (W 0.0192129) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.10111 ± 0.139908 (W 0.0162404) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.735659 ± 0.00198607 (W 0.00282024) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.10939 ± 0.00362698 (W 0.00416346) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 23.7215 ± 0.122009 (W 0.177228) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 30.957 ± 0.171013 (W 0.134066) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.602252 ± 0.00284202 (W 0.00305483) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.1798 ± 0.0457838 (W 0.0496734) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.3353 ± 0.00673392 (W 0.00511364) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.34235 ± 0.0093849 (W 0.00309011) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 36.7434 ± 0.0814526 (W 0.364665) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 36.796 ± 0.041986 (W 0.0909618) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.86419 ± 0.0056103 (W 0.00359383) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.864462 ± 0.00259378 (W 0.00313619) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 21.9033 ± 0.0739829 (W 0.0238233) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 21.9465 ± 0.0480818 (W 0.0311034) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.528944 ± 0.00334935 (W 0.00356269) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.572392 ± 0.00314952 (W 0.00715038) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 10.2116 ± 0.0167033 (W 0.0225685) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.9727 ± 0.0323602 (W 0.0581318) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.105302 ± 0.00128836 (W 0.00345953) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.10662 ± 0.00399172 (W 0.00291232) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.73782 ± 0.0208373 (W 0.0522044) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.72813 ± 0.0468665 (W 0.0452654) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.25186 ± 0.00292468 (W 0.00263011) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.29016 ± 0.00580188 (W 0.0012239) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 4.33977 ± 0.0168899 (W 0.0155575) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 6.11604 ± 0.0421067 (W 0.0137457) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.657494 ± 0.00390407 (W 0.00358585) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.02552 ± 0.00634938 (W 0.0044385) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 16.3426 ± 0.0570562 (W 0.0583361) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 25.0175 ± 0.134023 (W 0.103904) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.537642 ± 0.00243529 (W 0.00336514) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 9.73561 ± 0.0703031 (W 0.0268571) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.25417 ± 0.00971246 (W 0.00392501) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.26759 ± 0.00367072 (W 0.00502463) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 30.5026 ± 0.0411116 (W 0.0623541) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 30.7203 ± 0.0872169 (W 0.0483229) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.843745 ± 0.00470949 (W 0.00359462) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.843121 ± 0.00388135 (W 0.0043208) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.4875 ± 0.443605 (W 0.0369813) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.4529 ± 0.041678 (W 0.0299152) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.810928 ± 0.00834782 (W 0.00268612) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.738434 ± 0.00548113 (W 0.002483) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 52.379 ± 1.01716 (W 0.454787) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 47.0143 ± 0.462478 (W 0.163166) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.594986 ± 0.00485627 (W 0.00256144) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.59565 ± 0.00353299 (W 0.00211748) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.7615 ± 0.0695927 (W 0.0469015) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.7606 ± 0.0585343 (W 0.0385615) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.224316 ± 0.000593376 (W 0.00172397) | 0.255097 ± 0.000531114 (W 0.00183472) | 🔵 faer-fork 1.14× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.239936 ± 0.000778095 (W 0.00176163) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.824 ± 0.0276035 (W 0.0275866) | 11.9569 ± 0.00913431 (W 0.0132404) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.3588 ± 0.0223275 (W 0.0237607) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.24843 ± 0.00418134 (W 0.00313926) | 1.86287 ± 0.00352184 (W 0.0050407) | 🔵 faer-fork 1.49× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.56687 ± 0.00510661 (W 0.00491171) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 61.1622 ± 0.0885623 (W 0.105184) | 86.1484 ± 0.305257 (W 0.148558) | 🔵 faer-fork 1.41× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 69.0918 ± 0.100981 (W 0.12989) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.663756 ± 0.00162558 (W 0.0023198) | 0.918453 ± 0.00310767 (W 0.00104328) | 🔵 faer-fork 1.38× faster than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 24.7982 ± 0.0458238 (W 0.0342866) | 38.0674 ± 0.101947 (W 0.0883817) | 🔵 faer-fork 1.54× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 1.97976 ± 0.0040298 (W 0.00266345) | 3.237 ± 0.00912435 (W 0.00686966) | 🔵 faer-fork 1.64× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.99746 ± 0.00878663 (W 0.0030672) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 77.4131 ± 0.175661 (W 0.139928) | 119.521 ± 0.350549 (W 0.281287) | 🔵 faer-fork 1.54× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 77.4671 ± 0.0899391 (W 0.136955) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.93443 ± 0.00418346 (W 0.00216366) | 1.57758 ± 0.00215751 (W 0.00218377) | 🔵 faer-fork 1.69× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.937606 ± 0.0035259 (W 0.00264818) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.4519 ± 0.0892399 (W 0.0502886) | 53.888 ± 0.59928 (W 0.146943) | 🔵 faer-fork 1.71× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.5351 ± 0.0277958 (W 0.0384889) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.488124 ± 0.00250952 (W 0.00459483) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.499884 ± 0.00376739 (W 0.0047308) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 15.2658 ± 0.0517612 (W 0.020675) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 14.4277 ± 0.0561955 (W 0.0508567) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.198126 ± 0.00296211 (W 0.00121069) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.201815 ± 0.00238989 (W 0.000834269) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 11.1075 ± 0.0159355 (W 0.0167556) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 11.0894 ± 0.0203086 (W 0.016902) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.184802 ± 0.00103856 (W 0.00157259) | 0.197354 ± 0.00109791 (W 0.00148122) | 🔵 faer-fork 1.07× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.213795 ± 0.00065518 (W 0.00130716) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.27342 ± 0.00676815 (W 0.0101048) | 6.20697 ± 0.0216903 (W 0.0141074) | 🔵 faer-fork 1.45× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.76043 ± 0.113143 (W 0.0211594) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.691631 ± 0.00314129 (W 0.00227716) | 1.4486 ± 0.0161067 (W 0.00560391) | 🔵 faer-fork 2.09× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.975554 ± 0.00426807 (W 0.00394247) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 21.6692 ± 0.10444 (W 0.0502188) | 36.3235 ± 0.241794 (W 0.166378) | 🔵 faer-fork 1.68× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 28.5053 ± 0.154304 (W 0.0761585) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.509005 ± 0.00237393 (W 0.00197282) | 0.991008 ± 0.00715228 (W 0.00211602) | 🔵 faer-fork 1.95× faster than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 10.7793 ± 0.00901701 (W 0.0191265) | 19.9872 ± 0.11604 (W 0.0296913) | 🔵 faer-fork 1.85× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 1.29688 ± 0.00559782 (W 0.00281549) | 3.09481 ± 0.0190827 (W 0.00621549) | 🔵 faer-fork 2.39× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.30646 ± 0.00827086 (W 0.00349188) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 31.4535 ± 0.144825 (W 0.0821295) | 66.7922 ± 0.413789 (W 0.171967) | 🔵 faer-fork 2.12× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 31.4676 ± 0.0994545 (W 0.170548) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.789061 ± 0.00263569 (W 0.00225521) | 1.65067 ± 0.00532301 (W 0.00423538) | 🔵 faer-fork 2.09× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.788925 ± 0.00272662 (W 0.0027759) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 16.1066 ± 0.0356013 (W 0.0291626) | 35.264 ± 0.335551 (W 0.060023) | 🔵 faer-fork 2.19× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 16.0805 ± 0.023451 (W 0.0258308) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.463154 ± 0.00231256 (W 0.00216161) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.489091 ± 0.00493854 (W 0.00429) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.82755 ± 0.0660805 (W 0.04189) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 9.59545 ± 0.0589563 (W 0.0146128) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.127095 ± 0.0017488 (W 0.00129286) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.128621 ± 0.001516 (W 0.00143891) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 8.81361 ± 0.0064781 (W 0.0276385) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 8.805 ± 0.0137851 (W 0.0152603) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.176864 ± 0.00135835 (W 0.00134739) | 0.178266 ± 0.000917563 (W 0.00144789) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.207661 ± 0.00146058 (W 0.00121873) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.23506 ± 0.00783025 (W 0.0167548) | 3.73875 ± 0.0110264 (W 0.00838964) | 🔵 faer-fork 1.16× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 3.77808 ± 0.00833188 (W 0.00972916) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.59695 ± 0.0156577 (W 0.00294623) | 1.49574 ± 0.0122608 (W 0.00401474) | 🔵 faer-fork 2.51× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 0.890134 ± 0.00568811 (W 0.00485876) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 15.8546 ± 0.0267735 (W 0.0393117) | 28.5247 ± 0.222544 (W 0.0722803) | 🔵 faer-fork 1.80× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 22.3468 ± 0.141878 (W 0.115446) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.48684 ± 0.00183971 (W 0.00253308) | 0.989734 ± 0.00626059 (W 0.00448764) | 🔵 faer-fork 2.03× faster than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 8.4227 ± 0.0217943 (W 0.0308617) | 17.7264 ± 0.127831 (W 0.0258328) | 🔵 faer-fork 2.10× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 1.20409 ± 0.00389918 (W 0.00423499) | 3.04397 ± 0.0141459 (W 0.00517771) | 🔵 faer-fork 2.53× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.22289 ± 0.00636869 (W 0.002849) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 25.0123 ± 0.0863142 (W 0.0577459) | 59.4286 ± 0.541468 (W 0.0687597) | 🔵 faer-fork 2.38× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 25.2648 ± 0.0734424 (W 0.0689018) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.768244 ± 0.00320204 (W 0.00317157) | 1.67119 ± 0.00257935 (W 0.0028509) | 🔵 faer-fork 2.18× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.765807 ± 0.00206507 (W 0.00227408) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 13.6312 ± 0.0206031 (W 0.0840399) | 32.988 ± 0.521652 (W 0.0376913) | 🔵 faer-fork 2.42× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.6479 ± 0.0611257 (W 0.0361074) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.91029 ± 0.00736253 (W 0.00476042) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.76969 ± 0.00275062 (W 0.00303177) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 152.426 ± 1.00518 (W 0.580941) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 148.707 ± 0.491502 (W 0.886963) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.18505 ± 0.0105325 (W 0.0051834) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.18522 ± 0.00531937 (W 0.00552967) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 136.673 ± 0.355579 (W 0.336686) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 136.714 ± 0.256984 (W 0.347083) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.922998 ± 0.00297477 (W 0.0031182) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.959417 ± 0.00303272 (W 0.00272776) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 44.3831 ± 0.124907 (W 0.0586711) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 45.8183 ± 0.170566 (W 0.0985877) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.48557 ± 0.0170939 (W 0.00726869) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 5.06765 ± 0.0169488 (W 0.0068239) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 238.09 ± 1.47471 (W 0.550348) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 248.656 ± 0.804719 (W 0.725826) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.33288 ± 0.00703823 (W 0.00408384) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 97.4144 ± 0.257084 (W 0.188306) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 5.4486 ± 0.0154827 (W 0.011494) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.42217 ± 0.0260763 (W 0.00722935) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 271.647 ± 0.537704 (W 0.369851) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 272.161 ± 0.595845 (W 5.84479) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.53062 ± 0.00701919 (W 0.00789243) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.50003 ± 0.00903914 (W 0.00605398) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 113.2 ± 1.14205 (W 0.407044) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 113.812 ± 0.714073 (W 0.816128) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.851132 ± 0.00860627 (W 0.00313106) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.927159 ± 0.00749009 (W 0.00370738) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 49.6492 ± 0.113313 (W 0.256041) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 51.4928 ± 0.178584 (W 0.15166) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.594876 ± 0.00375316 (W 0.00195067) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.596833 ± 0.00241573 (W 0.00231046) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 36.6643 ± 0.260587 (W 0.220891) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 36.51 ± 0.25942 (W 0.189761) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.630481 ± 0.00232375 (W 0.0031287) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.679874 ± 0.00278402 (W 0.00307616) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 18.3777 ± 0.049866 (W 0.0702008) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 20.7406 ± 0.240095 (W 0.114615) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.81129 ± 0.0115744 (W 0.00402455) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.57852 ± 0.0185054 (W 0.00921617) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 76.3643 ± 0.295863 (W 0.337939) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 91.4287 ± 0.446268 (W 0.325008) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.25814 ± 0.00374238 (W 0.00361876) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 36.5373 ± 0.115674 (W 0.107919) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.65047 ± 0.00851114 (W 0.00507533) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.67649 ± 0.00598988 (W 0.00659781) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 98.7583 ± 0.0666703 (W 0.104503) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 99.3667 ± 0.14846 (W 0.126925) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.65302 ± 0.0165313 (W 0.00443422) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.65 ± 0.0070226 (W 0.00382028) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 52.1477 ± 0.0770878 (W 0.0880689) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 52.2281 ± 0.132444 (W 0.116393) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.718704 ± 0.0138634 (W 0.00409904) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.802501 ± 0.00720519 (W 0.00270622) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 25.4405 ± 0.153744 (W 0.0539511) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 28.4351 ± 0.17197 (W 0.103392) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.317268 ± 0.00625369 (W 0.00300258) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.312005 ± 0.00444316 (W 0.00330513) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 24.1017 ± 0.0658234 (W 0.115345) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 24.0537 ± 0.10523 (W 0.114524) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.549527 ± 0.00138403 (W 0.00351339) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.619495 ± 0.00129818 (W 0.00197634) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.3393 ± 0.0403645 (W 0.0256678) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 16.8245 ± 0.11862 (W 0.0848916) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.4325 ± 0.00512579 (W 0.0052508) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.22905 ± 0.0229966 (W 0.00821777) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 48.1413 ± 0.316137 (W 0.109424) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 68.3459 ± 0.624541 (W 0.209335) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.11526 ± 0.00440742 (W 0.00376177) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 26.7142 ± 0.0880132 (W 0.108118) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.28076 ± 0.00533179 (W 0.00644862) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.30966 ± 0.00957797 (W 0.0075793) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 73.728 ± 0.0941433 (W 0.121424) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 74.39 ± 0.0721706 (W 0.105929) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.54311 ± 0.0097586 (W 0.00316858) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.55223 ± 0.0137774 (W 0.00399351) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 43.2465 ± 0.0505352 (W 0.061667) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 43.1995 ± 0.0761103 (W 0.0516245) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.7076 ± 0.0101786 (W 0.00519834) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.66572 ± 0.00702301 (W 0.00369427) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 150.069 ± 0.567095 (W 0.846348) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 142.935 ± 0.984363 (W 1.02843) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.24838 ± 0.0141315 (W 0.00182269) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.24526 ± 0.0142425 (W 0.00300076) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 137.775 ± 0.28815 (W 0.302202) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 137.904 ± 0.201938 (W 0.335748) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.821848 ± 0.0032738 (W 0.00306757) | 0.843487 ± 0.00589203 (W 0.00156301) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.847004 ± 0.00297848 (W 0.00207188) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 40.0316 ± 0.126894 (W 0.0368977) | 44.8627 ± 0.141996 (W 0.0389599) | 🔵 faer-fork 1.12× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 41.1969 ± 0.118051 (W 0.0526205) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.31839 ± 0.0171872 (W 0.0054673) | 5.16018 ± 0.0166546 (W 0.00964696) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 4.91565 ± 0.0233068 (W 0.00842027) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 228.848 ± 0.599068 (W 0.418356) | 294.739 ± 0.989744 (W 0.662177) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 243.185 ± 0.737341 (W 0.540558) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.04162 ± 0.00496522 (W 0.00609145) | 1.75261 ± 0.00948359 (W 0.00326886) | 🔴 faer-fork 1.16× slower than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 87.6911 ± 0.192581 (W 0.122058) | 110.421 ± 0.301329 (W 0.282756) | 🔵 faer-fork 1.26× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 5.32664 ± 0.0114346 (W 0.00558585) | 7.89946 ± 0.035347 (W 0.0151951) | 🔵 faer-fork 1.48× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.31019 ± 0.0244299 (W 0.00690934) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 259.792 ± 0.582782 (W 0.599455) | 372.075 ± 0.669631 (W 0.752612) | 🔵 faer-fork 1.43× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 259.877 ± 0.757236 (W 6.65547) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.41761 ± 0.00462523 (W 0.00444601) | 4.29529 ± 0.0289803 (W 0.0068598) | 🔵 faer-fork 1.78× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.37922 ± 0.00911738 (W 0.00486528) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 100.488 ± 0.304409 (W 0.147525) | 162.044 ± 0.586143 (W 0.432345) | 🔵 faer-fork 1.61× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 100.141 ± 0.257013 (W 0.15237) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.781713 ± 0.00532268 (W 0.00375462) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.822968 ± 0.00602205 (W 0.00276151) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 47.1372 ± 0.240426 (W 0.118069) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 46.4749 ± 0.22436 (W 0.0766314) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.648544 ± 0.00580394 (W 0.0021418) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.646116 ± 0.00163762 (W 0.00288463) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 39.1066 ± 0.0760972 (W 0.0853001) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 39.0895 ± 0.114408 (W 0.074847) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.519446 ± 0.00157131 (W 0.00283574) | 0.562169 ± 0.00136494 (W 0.0016902) | 🔵 faer-fork 1.08× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.573747 ± 0.00339533 (W 0.0034091) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 14.6053 ± 0.044609 (W 0.0333868) | 22.0882 ± 0.0624966 (W 0.0217082) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 15.6942 ± 0.0202013 (W 0.0353163) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.78508 ± 0.00350754 (W 0.00465085) | 2.19999 ± 0.00685036 (W 0.00243148) | 🔵 faer-fork 1.23× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.40558 ± 0.00582983 (W 0.00755998) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 71.8094 ± 0.0682879 (W 0.0875947) | 92.8953 ± 0.285503 (W 0.305334) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 85.4571 ± 0.422106 (W 0.170294) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.15013 ± 0.00241104 (W 0.00341369) | 0.723356 ± 0.00109717 (W 0.00200454) | 🔴 faer-fork 1.59× slower than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 32.4078 ± 0.0763996 (W 0.0815096) | 35.5939 ± 0.0420908 (W 0.0473245) | 🔵 faer-fork 1.10× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1` | 2.55431 ± 0.0124678 (W 0.00604996) | 4.75229 ± 0.0348991 (W 0.00764776) | 🔵 faer-fork 1.86× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.58268 ± 0.00554492 (W 0.00658093) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 86.5292 ± 0.125923 (W 0.278472) | 149.671 ± 0.755309 (W 0.152334) | 🔵 faer-fork 1.73× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 87.2552 ± 0.101758 (W 0.198633) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.5549 ± 0.00734075 (W 0.00424664) | 3.21962 ± 0.0282547 (W 0.00632795) | 🔵 faer-fork 2.07× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.57035 ± 0.00173547 (W 0.00526803) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 40.0584 ± 0.0703507 (W 0.0719514) | 83.075 ± 0.583601 (W 0.0621689) | 🔵 faer-fork 2.07× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 40.1078 ± 0.0806174 (W 0.0571428) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.646434 ± 0.00210069 (W 0.004353) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.708673 ± 0.00219973 (W 0.00733703) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 23.2098 ± 0.0485426 (W 0.0482884) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 23.5041 ± 0.210711 (W 0.0741141) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.378269 ± 0.00347234 (W 0.00379876) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.373704 ± 0.00423084 (W 0.0028227) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 25.9774 ± 0.76947 (W 1.36976) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 26.0366 ± 0.875541 (W 1.05761) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.456763 ± 0.00101842 (W 0.00165254) | 0.444041 ± 0.00252676 (W 0.00215241) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.512691 ± 0.00245751 (W 0.00241311) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.5212 ± 0.0412054 (W 0.0154367) | 12.5366 ± 0.0129022 (W 0.0117096) | 🔵 faer-fork 1.19× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.4982 ± 0.0495075 (W 0.0338541) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.39798 ± 0.00215182 (W 0.00397893) | 1.89888 ± 0.00718165 (W 0.00582015) | 🔵 faer-fork 1.36× faster than `lapack-openblas` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.04355 ± 0.0103404 (W 0.00534257) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 48.6885 ± 0.0647482 (W 0.0950839) | 60.2573 ± 0.102932 (W 0.080971) | 🔵 faer-fork 1.24× faster than `lapack-openblas` |  |
| `qr/4096x256/b1 [api-inclusive]` | 61.5095 ± 0.233712 (W 0.169045) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.03396 ± 0.003194 (W 0.00380202) | 0.573401 ± 0.00509126 (W 0.00268324) | 🔴 faer-fork 1.80× slower than `lapack-openblas` |  |
| `qr_factor/4096x256/b1` | 23.8358 ± 0.0871512 (W 0.0684761) | 24.1582 ± 0.0985667 (W 0.037943) | · |  |
| `svd_thin/1024x64/b1` | 2.17109 ± 0.0116843 (W 0.00635731) | 4.30268 ± 0.0292437 (W 0.00738121) | 🔵 faer-fork 1.98× faster than `lapack-openblas` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.20077 ± 0.00980599 (W 0.00481032) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 61.3002 ± 0.074313 (W 0.0848218) | 114.75 ± 0.457568 (W 0.13796) | 🔵 faer-fork 1.87× faster than `lapack-openblas` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 61.9768 ± 0.117846 (W 0.0912324) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.46215 ± 0.00825868 (W 0.00371164) | 3.07457 ± 0.019074 (W 0.00417464) | 🔵 faer-fork 2.10× faster than `lapack-openblas` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.4749 ± 0.00693537 (W 0.00461779) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.1671 ± 0.0929261 (W 0.0460716) | 71.6092 ± 0.551024 (W 0.0681703) | 🔵 faer-fork 2.30× faster than `lapack-openblas` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.2454 ± 0.0877118 (W 0.0496106) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.655786 ± 0.00440517 (W 0.00285323) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.559326 ± 0.00248691 (W 0.00314198) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 25.497 ± 0.190559 (W 0.0754113) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 23.8663 ± 0.105885 (W 0.0725234) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.277602 ± 0.00194793 (W 0.000327732) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.278937 ± 0.00199076 (W 0.000367361) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.3431 ± 0.125147 (W 0.0525118) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.3312 ± 0.0428873 (W 0.0363324) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.315857 ± 0.00233974 (W 0.00216772) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.259831 ± 0.00287612 (W 0.00310753) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 7.63945 ± 0.0151059 (W 0.0107369) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 8.56389 ± 0.054512 (W 0.126019) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.947526 ± 0.00164564 (W 0.00205499) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.03444 ± 0.00209961 (W 0.00255928) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 35.977 ± 0.121877 (W 0.123007) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 37.7581 ± 0.144398 (W 0.372899) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.679622 ± 0.00175605 (W 0.00254615) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 18.1917 ± 0.120174 (W 0.072611) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.36428 ± 0.0250753 (W 0.00357639) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.37053 ± 0.0190519 (W 0.00408625) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 44.8366 ± 0.682276 (W 0.408623) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 44.8892 ± 0.572305 (W 0.0625102) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.73462 ± 0.00516243 (W 0.00590967) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.73866 ± 0.00178861 (W 0.00415725) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 21.7581 ± 0.595425 (W 0.0230865) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 21.7973 ± 0.045859 (W 0.0241141) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.443945 ± 0.00374119 (W 0.00300273) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.476159 ± 0.00196099 (W 0.00329952) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.5078 ± 0.0411797 (W 0.0241284) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.0196 ± 0.05206 (W 0.0638748) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.087949 ± 0.000814015 (W 0.00188259) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.087649 ± 0.000670441 (W 0.0019359) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.63679 ± 0.00721432 (W 0.0111101) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.63162 ± 0.0189955 (W 0.0142308) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.219905 ± 0.002618 (W 0.00172309) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.251401 ± 0.00123593 (W 0.00132862) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.82607 ± 0.0268716 (W 0.024098) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.56551 ± 0.0282805 (W 0.070071) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.548651 ± 0.00399278 (W 0.00227876) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.752205 ± 0.00371342 (W 0.00437911) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 12.7764 ± 0.107818 (W 0.0276968) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 17.3959 ± 0.10956 (W 0.157179) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.468002 ± 0.00105369 (W 0.00156623) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 7.85943 ± 0.0352968 (W 0.0232813) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.01704 ± 0.00385767 (W 0.00350661) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.02386 ± 0.00369978 (W 0.00161648) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 21.4167 ± 0.536495 (W 0.0376976) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 21.4934 ± 0.526123 (W 0.0309902) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.691305 ± 0.00557872 (W 0.00310302) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.687835 ± 0.00514534 (W 0.00373373) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 14.4215 ± 0.0468186 (W 0.0246459) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 14.4014 ± 0.0323851 (W 0.0181155) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.432922 ± 0.00300139 (W 0.00363691) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.484115 ± 0.00342343 (W 0.00282912) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.9095 ± 0.0192152 (W 0.0225936) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.97968 ± 0.0773669 (W 0.0676208) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.077097 ± 0.00140959 (W 0.00259663) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.0762 ± 0.000893154 (W 0.00222682) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 2.59316 ± 0.00881152 (W 0.015996) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 2.59613 ± 0.0162622 (W 0.0304056) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.198946 ± 0.00118953 (W 0.00174288) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.246776 ± 0.00219291 (W 0.0019089) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.8358 ± 0.0192262 (W 0.0109031) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.08005 ± 0.0557094 (W 0.0560446) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.515645 ± 0.00475014 (W 0.00485391) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.713952 ± 0.00277668 (W 0.00381828) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 9.1108 ± 0.0815799 (W 0.0240524) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 13.97 ± 0.272953 (W 0.234855) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.431771 ± 0.00150784 (W 0.0022091) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.83925 ± 0.035537 (W 0.0202003) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.966546 ± 0.0957189 (W 0.0031841) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.974778 ± 0.00918022 (W 0.00186092) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 17.6372 ± 0.0347784 (W 0.0274381) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 17.7411 ± 0.074828 (W 0.0362631) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.673099 ± 0.00540248 (W 0.00403178) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.670957 ± 0.00361053 (W 0.00602349) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 12.8501 ± 0.561706 (W 0.0299517) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 12.8534 ± 0.040972 (W 0.0213511) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.493263 ± 0.0017052 (W 0.00257694) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.458723 ± 0.00173575 (W 0.00310425) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 24.3559 ± 0.0565309 (W 0.0841379) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 21.4609 ± 0.0639593 (W 0.0316465) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.301396 ± 0.000992424 (W 0.00155639) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.300744 ± 0.000734138 (W 0.00183926) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.7353 ± 0.0530404 (W 0.0240775) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.7612 ± 0.0285116 (W 0.0474413) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.148919 ± 0.00097403 (W 0.00171643) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.156156 ± 0.000956088 (W 0.00144875) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.62312 ± 0.0900833 (W 0.0103966) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.79856 ± 0.0513095 (W 0.0222002) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.746636 ± 0.00309674 (W 0.00223296) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.907 ± 0.00153744 (W 0.00276247) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 31.5908 ± 0.224851 (W 0.070163) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 35.1484 ± 0.277406 (W 0.0356125) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.446412 ± 0.0024479 (W 0.00285528) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.8241 ± 0.0349453 (W 0.0183843) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.36092 ± 0.00417018 (W 0.0044931) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.36121 ± 0.00703656 (W 0.00328512) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 41.8461 ± 0.138774 (W 0.233327) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 41.9663 ± 0.155674 (W 0.0476477) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.69954 ± 0.000928545 (W 0.00195159) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.700105 ± 0.00093689 (W 0.00257569) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 18.6554 ± 0.124592 (W 0.0237357) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 18.6757 ± 0.104353 (W 0.0280182) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.36886 ± 0.0013163 (W 0.00231818) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.380978 ± 0.00191235 (W 0.00253508) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.13815 ± 0.0344948 (W 0.0129741) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.6306 ± 0.0353376 (W 0.0131304) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.104589 ± 0.000750696 (W 0.000769512) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.10459 ± 0.000779681 (W 0.00135978) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.48744 ± 0.0135366 (W 0.00674693) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.49022 ± 0.00917688 (W 0.0125247) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.141836 ± 0.000845013 (W 0.00115596) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.152604 ± 0.000802468 (W 0.000812795) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.70646 ± 0.00887049 (W 0.00539328) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.91852 ± 0.00562153 (W 0.00843413) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.492567 ± 0.000892394 (W 0.00258045) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.623945 ± 0.00411847 (W 0.00468606) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 11.7039 ± 0.0523374 (W 0.0151058) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.7183 ± 0.106437 (W 0.028541) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.392702 ± 0.0026336 (W 0.00183349) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 6.70966 ± 0.0321162 (W 0.0181086) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.00618 ± 0.00216832 (W 0.00288279) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.01064 ± 0.00481917 (W 0.00143093) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 18.6554 ± 0.0514298 (W 0.0480065) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 18.7444 ± 0.0747315 (W 0.0203002) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.650853 ± 0.00310943 (W 0.00240394) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.65013 ± 0.00171709 (W 0.00274199) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 11.06 ± 0.0401389 (W 0.016055) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 11.094 ± 0.0548069 (W 0.0228524) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.37714 ± 0.0014723 (W 0.00218342) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.387586 ± 0.00394471 (W 0.00333218) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.71331 ± 0.0317849 (W 0.0128212) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 6.59725 ± 0.0372218 (W 0.0137847) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.068807 ± 0.000601222 (W 0.000675458) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.0691485 ± 0.000321238 (W 0.000893732) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 3.69839 ± 0.120503 (W 0.145098) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 3.59117 ± 0.179971 (W 0.128547) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.13428 ± 0.00132081 (W 0.00112698) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.147822 ± 0.00148508 (W 0.000834841) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.24343 ± 0.00574449 (W 0.00515578) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.47285 ± 0.0101941 (W 0.00501263) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.442298 ± 0.000820385 (W 0.00228618) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.581868 ± 0.00313008 (W 0.00392951) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 8.22762 ± 0.0373769 (W 0.0177189) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 11.4151 ± 0.056396 (W 0.0423352) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.37726 ± 0.00141677 (W 0.00263169) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.16822 ± 0.0183991 (W 0.0121325) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.950965 ± 0.00514251 (W 0.00160503) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.956922 ± 0.00405059 (W 0.00144248) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 14.9503 ± 0.0845204 (W 0.043489) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 15.0182 ± 0.0318747 (W 0.0209101) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.631697 ± 0.00229997 (W 0.00203179) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.630821 ± 0.0028536 (W 0.00238526) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 9.46608 ± 0.0265992 (W 0.012595) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 9.46215 ± 0.0457122 (W 0.0141651) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.22959 ± 0.00806938 (W 0.00516911) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.10068 ± 0.00794855 (W 0.00338248) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 63.3364 ± 0.464866 (W 0.264935) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 60.3102 ± 0.508051 (W 0.38311) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.10105 ± 0.00146221 (W 0.00161933) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.10287 ± 0.00334354 (W 0.00196281) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 68.1948 ± 0.191756 (W 0.0992012) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 68.1951 ± 0.193053 (W 0.0604463) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.651248 ± 0.00266644 (W 0.00370792) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.623549 ± 0.00218691 (W 0.00253577) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 22.9854 ± 0.0697352 (W 0.033875) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 24.9344 ± 0.0477151 (W 0.117869) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.48841 ± 0.00612981 (W 0.00406584) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.77402 ± 0.00909949 (W 0.00382356) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 123.536 ± 1.08878 (W 0.480057) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 128.22 ± 0.221302 (W 0.147509) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.40348 ± 0.00392604 (W 0.00413951) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 53.8178 ± 0.289618 (W 0.200859) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.07169 ± 0.0060545 (W 0.00379359) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.07547 ± 0.00383497 (W 0.00380939) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 139.871 ± 0.168088 (W 0.292642) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 140.087 ± 0.123176 (W 0.163604) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.49909 ± 0.00152194 (W 0.00436959) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.50598 ± 0.0057982 (W 0.00784197) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 59.4073 ± 0.0816492 (W 0.0651362) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 59.5317 ± 0.135627 (W 0.0849328) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.696838 ± 0.00486307 (W 0.00387482) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.736906 ± 0.00431806 (W 0.00244734) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 20.3903 ± 0.131676 (W 0.0272915) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 21.3504 ± 0.0703199 (W 0.0460943) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.298919 ± 0.0101407 (W 0.00108232) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.299635 ± 0.00953537 (W 0.00190347) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.4798 ± 1.39594 (W 0.0209104) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.4608 ± 1.38719 (W 0.0141868) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.468739 ± 0.00632509 (W 0.0040358) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.515393 ± 0.00443573 (W 0.00395092) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 11.3989 ± 0.0236462 (W 0.0265016) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 12.6211 ± 0.0591982 (W 0.0373245) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.09049 ± 0.00603455 (W 0.00360244) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.47286 ± 0.00532675 (W 0.00543655) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 39.9599 ± 0.0936358 (W 0.0453619) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 48.9524 ± 0.243418 (W 0.0852689) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.811847 ± 0.00428058 (W 0.00277675) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 21.6463 ± 0.0380045 (W 0.0211706) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.67067 ± 0.00499863 (W 0.00642776) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.68314 ± 0.00692949 (W 0.00517078) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 53.1337 ± 0.0863196 (W 0.0681884) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 53.3038 ± 0.0477872 (W 0.12154) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.07239 ± 0.00785804 (W 0.00545313) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.07347 ± 0.00762645 (W 0.00392712) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 29.5293 ± 0.038145 (W 0.0516873) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 29.5448 ± 0.0569731 (W 0.0425985) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.627822 ± 0.00281873 (W 0.00344985) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.700926 ± 0.00251948 (W 0.00364668) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 13.0823 ± 0.033186 (W 0.0182974) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 15.4699 ± 0.0496525 (W 0.0409558) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.17587 ± 0.00156036 (W 0.00324795) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.17543 ± 0.00416477 (W 0.0037954) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.46149 ± 0.0501179 (W 0.0777238) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.48061 ± 0.024133 (W 0.0518832) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.428551 ± 0.00253263 (W 0.00320119) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.494959 ± 0.0030749 (W 0.00303758) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.41341 ± 0.0106686 (W 0.0291904) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 10.7195 ± 0.0343851 (W 0.0355231) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.897918 ± 0.00332158 (W 0.00410607) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.29421 ± 0.0070324 (W 0.0044571) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 25.7699 ± 0.0712126 (W 0.0790671) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 36.6737 ± 0.106175 (W 0.0840144) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.72882 ± 0.00117097 (W 0.00263263) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 15.6246 ± 0.0518194 (W 0.0206198) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.48343 ± 0.00604568 (W 0.00616913) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.48822 ± 0.0123059 (W 0.00479661) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 40.0975 ± 0.123181 (W 0.140778) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 40.3359 ± 0.0871267 (W 0.069548) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.01685 ± 0.0113682 (W 0.00480179) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01825 ± 0.00790533 (W 0.00402075) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 24.5066 ± 0.0639435 (W 0.0396066) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.5089 ± 0.0793214 (W 0.0359472) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.992444 ± 0.00339979 (W 0.00195217) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.981851 ± 0.00196555 (W 0.00357403) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 59.5678 ± 0.721909 (W 0.193201) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 56.6501 ± 0.281803 (W 0.134864) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.14399 ± 0.00276539 (W 0.00251321) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.14219 ± 0.00228185 (W 0.00238192) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 69.0429 ± 0.109308 (W 0.0614528) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 69.0362 ± 0.187103 (W 0.0854181) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.488398 ± 0.00349824 (W 0.00228855) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.505573 ± 0.00146088 (W 0.00239193) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 20.9728 ± 0.0483053 (W 0.0523768) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 21.4542 ± 0.0362557 (W 0.0269354) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.2977 ± 0.0025106 (W 0.00263468) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.64179 ± 0.00602927 (W 0.00463261) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 117.225 ± 0.193936 (W 0.164105) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 124.859 ± 0.181007 (W 0.17607) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.14658 ± 0.00329516 (W 0.00309052) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 46.7836 ± 0.103028 (W 0.0400323) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.03776 ± 0.00723274 (W 0.0043084) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.04306 ± 0.00569375 (W 0.00468713) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 134.682 ± 0.20312 (W 0.229358) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 134.911 ± 0.111739 (W 0.207928) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.43266 ± 0.00305094 (W 0.00455114) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.43446 ± 0.00339904 (W 0.00317449) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 53.9357 ± 0.117745 (W 0.0637014) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 54.064 ± 0.108906 (W 0.0622134) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.599811 ± 0.00216676 (W 0.00294999) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.623714 ± 0.00140053 (W 0.00248659) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 18.3177 ± 0.0504216 (W 0.088278) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 18.0214 ± 0.0852201 (W 0.0308062) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.345196 ± 0.0181372 (W 0.00251309) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.344543 ± 0.021142 (W 0.00226496) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 19.2654 ± 0.623954 (W 0.028101) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 19.2592 ± 0.55225 (W 0.0275223) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.37279 ± 0.00426565 (W 0.00547191) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.396532 ± 0.00176004 (W 0.00364901) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.68464 ± 0.0237605 (W 0.0154921) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 9.21463 ± 0.0125233 (W 0.026195) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.04268 ± 0.00411701 (W 0.00230439) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.33699 ± 0.00295428 (W 0.0041031) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 38.3103 ± 0.081321 (W 0.0530347) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 45.1615 ± 0.0666037 (W 0.063975) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.709286 ± 0.00204274 (W 0.00274812) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 18.8488 ± 0.0448641 (W 0.0287968) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.63167 ± 0.00260755 (W 0.00355898) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.64432 ± 0.00178365 (W 0.00382196) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 47.8419 ± 0.0908274 (W 0.113703) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 47.94 ± 0.0929 (W 0.0893892) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.00943 ± 0.00420371 (W 0.00282299) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01393 ± 0.00241096 (W 0.00296535) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 23.9999 ± 0.0427554 (W 0.033425) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.0303 ± 0.0435764 (W 0.0279436) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.560417 ± 0.0026894 (W 0.00270897) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.590007 ± 0.00253322 (W 0.00225788) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 12.1074 ± 0.112435 (W 0.037059) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 12.1231 ± 0.0552257 (W 0.0235452) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.197821 ± 0.00124035 (W 0.00132492) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.19723 ± 0.00176254 (W 0.00158596) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.5791 ± 0.0360629 (W 0.0469328) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.5871 ± 0.0440484 (W 0.0472532) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.351473 ± 0.00333572 (W 0.00340567) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.379048 ± 0.00206706 (W 0.00250503) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.79061 ± 0.0234895 (W 0.0257742) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.34582 ± 0.0084256 (W 0.0115872) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.847684 ± 0.00322351 (W 0.00188633) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.15493 ± 0.0041438 (W 0.00401251) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 26.0284 ± 0.0416196 (W 0.0928129) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 32.8304 ± 0.142811 (W 0.0829628) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.654519 ± 0.00191722 (W 0.00273412) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.8891 ± 0.179632 (W 0.0228059) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.43973 ± 0.0127096 (W 0.00396732) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.45459 ± 0.00378314 (W 0.00486433) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 34.8339 ± 0.0271685 (W 0.0807284) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 35.0136 ± 0.107471 (W 0.0800893) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.957573 ± 0.00331458 (W 0.0042204) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.955153 ± 0.00406257 (W 0.00199466) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 18.9414 ± 0.0311751 (W 0.0296739) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.0135 ± 0.0404331 (W 0.0437425) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `lu/1024x64`, `lu/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `solve/1024x64`, `solve/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 52.0955 ± 0.489873 (W 0.450634) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 79.5498 ± 1.02576 (W 0.443246) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.79255 ± 0.00370224 (W 0.00298196) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.25374 ± 0.00340695 (W 0.00604558) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.6071 ± 0.0726484 (W 0.0205575) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 15.8084 ± 0.061993 (W 0.0285078) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.279678 ± 0.00118027 (W 0.00176293) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.423467 ± 0.000879743 (W 0.00266411) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 28.8733 ± 0.109433 (W 0.0486969) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 30.357 ± 0.126674 (W 0.0797071) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.02273 ± 0.00496741 (W 0.00756408) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.736965 ± 0.00662882 (W 0.00177256) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.4102 ± 0.265555 (W 0.0748575) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.989909 ± 0.00434216 (W 0.00677903) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 77.3788 ± 0.35908 (W 0.241553) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 77.2099 ± 0.204989 (W 0.0822705) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.96926 ± 0.00988829 (W 0.00346525) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.98263 ± 0.00341562 (W 0.00503229) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 31.5537 ± 0.109241 (W 0.0357613) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.604 ± 0.0977252 (W 0.0605428) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.930799 ± 0.00293349 (W 0.0042134) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.935759 ± 0.00354799 (W 0.00260789) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 15.585 ± 6.72164 (W 0.024036) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 33.3236 ± 10.3036 (W 0.105778) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.460943 ± 0.00643584 (W 0.0032323) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.743898 ± 0.081131 (W 0.00335518) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 3.51071 ± 0.173777 (W 0.0312698) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 6.48927 ± 0.459284 (W 0.0694765) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.123826 ± 0.00359115 (W 0.00164532) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.22097 ± 0.0204532 (W 0.00178564) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 10.4827 ± 0.531054 (W 0.0287135) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 10.2758 ± 1.99012 (W 0.024716) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.633522 ± 0.00512676 (W 0.00534001) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.363215 ± 0.031923 (W 0.00190847) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 10.1254 ± 0.564656 (W 0.0200626) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.613739 ± 0.00455688 (W 0.00406468) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 31.2856 ± 0.123046 (W 0.0713679) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 31.3151 ± 0.0950161 (W 0.0648409) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.2964 ± 0.00530904 (W 0.00437475) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.31032 ± 0.00346407 (W 0.00343428) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 16.1804 ± 0.0425107 (W 0.0218059) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 16.1872 ± 0.0242426 (W 0.0293212) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.789041 ± 0.00157339 (W 0.00127667) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.791325 ± 0.0024527 (W 0.00194488) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.0898 ± 0.0706977 (W 0.0215477) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 27.2852 ± 0.061288 (W 0.186259) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.439397 ± 0.0020407 (W 0.00353289) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.693904 ± 0.0043441 (W 0.00259088) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.12526 ± 0.00697251 (W 0.0133316) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 4.77739 ± 0.0328079 (W 0.0397875) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.095108 ± 0.000883932 (W 0.00213891) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.177569 ± 0.00246361 (W 0.000543012) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.54292 ± 0.0386069 (W 0.0270985) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 6.94495 ± 0.0432705 (W 0.0261466) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.565091 ± 0.00168817 (W 0.00404203) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.312413 ± 0.00183077 (W 0.00177425) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 7.29405 ± 0.028841 (W 0.0338582) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.549754 ± 0.00552212 (W 0.00730614) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 24.9184 ± 0.135658 (W 0.0649371) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 25.1978 ± 0.0798154 (W 0.0355138) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.20954 ± 0.00659344 (W 0.00324795) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.22324 ± 0.00463137 (W 0.0030588) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 13.7515 ± 0.0394575 (W 0.0245495) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 13.7709 ± 0.0273134 (W 0.0447907) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.766099 ± 0.00210623 (W 0.00379071) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.768832 ± 0.00275703 (W 0.00375582) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 88.7355 ± 1.10999 (W 0.467669) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 75.2893 ± 0.771918 (W 0.454238) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.21516 ± 0.0133315 (W 0.00435964) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.16974 ± 0.00585512 (W 0.00381014) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 12.6714 ± 0.0346525 (W 0.0334601) | 19.1565 ± 0.110716 (W 0.0302956) | 🔵 faer-fork 1.51× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 13.1954 ± 0.0274512 (W 0.0294994) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.305534 ± 0.00135324 (W 0.00147015) | 0.655787 ± 0.00682276 (W 0.00226388) | 🔵 faer-fork 2.15× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.319646 ± 0.000745001 (W 0.00239991) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.8277 ± 0.166659 (W 0.0953396) | 56.1727 ± 0.71901 (W 0.440586) | 🔵 faer-fork 2.09× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.7101 ± 0.278986 (W 0.0760587) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.602477 ± 0.00477787 (W 0.00299879) | 1.45496 ± 0.0127371 (W 0.00567901) | 🔵 faer-fork 2.41× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.623812 ± 0.00197775 (W 0.00296836) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 25.2471 ± 0.166111 (W 0.0820074) | 52.9582 ± 0.616385 (W 0.4583) | 🔵 faer-fork 2.10× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 0.563532 ± 0.00247266 (W 0.0021642) | 1.37912 ± 0.01071 (W 0.00472295) | 🔵 faer-fork 2.45× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 82.6462 ± 0.262868 (W 0.25859) | 196.033 ± 0.780674 (W 0.603052) | 🔵 faer-fork 2.37× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 82.6276 ± 0.15946 (W 0.356145) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.01636 ± 0.00590642 (W 0.00499487) | 4.51186 ± 0.0121403 (W 0.00951598) | 🔵 faer-fork 2.24× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.02862 ± 0.00535374 (W 0.00480288) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 37.2563 ± 0.0503449 (W 0.0668533) | 90.9777 ± 0.584562 (W 0.351266) | 🔵 faer-fork 2.44× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 37.2443 ± 0.109613 (W 0.0445255) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.00321 ± 0.00345207 (W 0.00423841) | 2.24025 ± 0.0124397 (W 0.00490675) | 🔵 faer-fork 2.23× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.00762 ± 0.00274626 (W 0.003691) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 33.1034 ± 11.8273 (W 0.0971584) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 30.8582 ± 11.0363 (W 0.0540951) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.659423 ± 0.0294452 (W 0.0034656) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.675484 ± 0.0193129 (W 0.00334893) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 3.41261 ± 0.138269 (W 0.00944766) | 10.9309 ± 0.114981 (W 0.0155638) | 🔵 faer-fork 3.20× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 3.8631 ± 0.172718 (W 0.0112033) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.097055 ± 0.000764884 (W 0.000536839) | 0.440272 ± 0.00420968 (W 0.00243917) | 🔵 faer-fork 4.54× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.115974 ± 0.00564613 (W 0.000456575) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 6.99093 ± 1.83175 (W 0.021481) | 30.921 ± 6.45592 (W 0.435575) | 🔵 faer-fork 4.42× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 7.59557 ± 1.73619 (W 0.0185924) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.219183 ± 0.00446721 (W 0.00146547) | 1.30597 ± 0.0152112 (W 0.00387532) | 🔵 faer-fork 5.96× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.249513 ± 0.0103562 (W 0.00153107) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 6.57153 ± 1.86575 (W 0.0244516) | 28.6032 ± 6.41861 (W 0.579402) | 🔵 faer-fork 4.35× faster than `lapack-openblas` | `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.199243 ± 0.00342421 (W 0.00195036) | 1.24667 ± 0.00531752 (W 0.00430135) | 🔵 faer-fork 6.26× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 36.7097 ± 0.0564663 (W 0.0874158) | 128.509 ± 0.529179 (W 0.232978) | 🔵 faer-fork 3.50× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.6946 ± 0.0611904 (W 0.0812367) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.33168 ± 0.0104933 (W 0.00278092) | 4.20206 ± 0.0262668 (W 0.00976481) | 🔵 faer-fork 3.16× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.34506 ± 0.010173 (W 0.00260667) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.8398 ± 0.0403591 (W 0.0468252) | 64.8967 ± 0.452409 (W 0.220981) | 🔵 faer-fork 2.97× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.862 ± 0.0234007 (W 0.0522147) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.85752 ± 0.00393685 (W 0.00618385) | 2.15823 ± 0.00467757 (W 0.0049703) | 🔵 faer-fork 2.52× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.856719 ± 0.00240509 (W 0.0061085) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 26.0499 ± 0.195307 (W 0.0495737) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 24.8944 ± 0.104994 (W 0.0611619) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.593889 ± 0.00413093 (W 0.00227714) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.621386 ± 0.00333809 (W 0.00239651) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.87734 ± 0.0110453 (W 0.0124654) | 8.62765 ± 0.0491947 (W 0.0107779) | 🔵 faer-fork 4.60× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 2.32226 ± 0.0127767 (W 0.00780959) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.062236 ± 0.000268557 (W 0.00037934) | 0.397239 ± 0.00224073 (W 0.00197853) | 🔵 faer-fork 6.38× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.0863045 ± 0.000463959 (W 0.000286651) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 3.83148 ± 0.010501 (W 0.00814007) | 26.2454 ± 0.241536 (W 0.154513) | 🔵 faer-fork 6.85× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 4.32609 ± 0.00787373 (W 0.00760433) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.165042 ± 0.000431755 (W 0.00153864) | 1.26737 ± 0.00634022 (W 0.00405233) | 🔵 faer-fork 7.68× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.199329 ± 0.00333319 (W 0.00180714) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.56016 ± 0.0164559 (W 0.00711027) | 24.3715 ± 0.173988 (W 0.0889246) | 🔵 faer-fork 6.85× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 0.147475 ± 0.00167033 (W 0.000623142) | 1.20825 ± 0.006874 (W 0.00675304) | 🔵 faer-fork 8.19× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 30.4472 ± 0.0632634 (W 0.0640644) | 118.604 ± 0.880565 (W 0.593344) | 🔵 faer-fork 3.90× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 30.7164 ± 0.0654845 (W 0.0561021) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.25186 ± 0.0093052 (W 0.00503304) | 4.13719 ± 0.0333238 (W 0.0100557) | 🔵 faer-fork 3.30× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.26506 ± 0.00599168 (W 0.00695221) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 19.341 ± 0.0650648 (W 0.0386153) | 60.3229 ± 0.162213 (W 0.124653) | 🔵 faer-fork 3.12× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.3999 ± 0.395588 (W 0.025482) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.835972 ± 0.0102133 (W 0.00480454) | 2.1921 ± 0.0072453 (W 0.00617079) | 🔵 faer-fork 2.62× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.834277 ± 0.00411313 (W 0.00433626) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 149.454 ± 0.351732 (W 1.26138) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 241.867 ± 0.778951 (W 1.03881) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.66244 ± 0.0157399 (W 0.00285901) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.61027 ± 0.0400711 (W 0.00858725) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 40.6123 ± 0.0807885 (W 0.0373876) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 47.5965 ± 0.151873 (W 0.136495) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.863746 ± 0.00398317 (W 0.00272501) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.988648 ± 0.00458269 (W 0.00360882) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 102.338 ± 0.322074 (W 0.301712) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 107.742 ± 0.415201 (W 0.340695) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.72188 ± 0.0102031 (W 0.00729043) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 2.49539 ± 0.00940614 (W 0.00317439) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 96.8904 ± 0.285613 (W 0.209668) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 2.57523 ± 0.00592633 (W 0.00708988) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 260.196 ± 9.25535 (W 0.318061) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 261.597 ± 9.48983 (W 6.42482) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 5.37868 ± 0.0322562 (W 0.0065143) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.36405 ± 0.0237434 (W 0.00554839) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 100.318 ± 0.104533 (W 0.124084) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 100.124 ± 0.374739 (W 0.350762) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.4032 ± 0.00937445 (W 0.00408535) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.37011 ± 0.00390273 (W 0.0041749) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 46.5238 ± 0.288395 (W 0.381978) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 88.9281 ± 0.270961 (W 0.371645) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.730653 ± 0.00607126 (W 0.00293923) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.37287 ± 0.0080683 (W 0.00475602) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.3873 ± 0.0202663 (W 0.0198772) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 16.5443 ± 0.08894 (W 0.0910333) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.292871 ± 0.00145659 (W 0.00197404) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.476893 ± 0.00136906 (W 0.0429885) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 29.4173 ± 0.0960817 (W 0.129155) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 31.7203 ± 0.113113 (W 0.0880389) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.03965 ± 0.00420313 (W 0.0041496) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.869359 ± 0.00680814 (W 0.0393828) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.8542 ± 0.107229 (W 0.108885) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.978953 ± 0.00378092 (W 0.00274482) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 87.1752 ± 0.208415 (W 0.167417) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 87.8104 ± 0.133234 (W 0.166641) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.59435 ± 0.00763123 (W 0.00545735) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.61716 ± 0.00494899 (W 0.00367182) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 40.1732 ± 0.107402 (W 0.0674797) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 40.2959 ± 0.103655 (W 0.0462123) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.54661 ± 0.0074883 (W 0.00707086) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.55918 ± 0.00602309 (W 0.00623237) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 22.7716 ± 1.49301 (W 0.0487666) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 55.2421 ± 0.885924 (W 0.455421) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.595919 ± 0.00641778 (W 0.0032593) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.21274 ± 0.00841944 (W 0.00495914) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.37332 ± 0.0237488 (W 0.0302004) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 11.5987 ± 0.125125 (W 0.087311) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.199646 ± 0.00144707 (W 0.00153276) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.346645 ± 0.00123944 (W 0.0016441) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 17.1845 ± 0.0621325 (W 0.0584657) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 19.2694 ± 0.0587338 (W 0.0715479) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.761525 ± 0.00148226 (W 0.00351306) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.604495 ± 0.00363632 (W 0.0018789) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.238 ± 0.080282 (W 0.10489) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.720855 ± 0.0271121 (W 0.00349811) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 61.9485 ± 0.113977 (W 0.0932998) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 62.6222 ± 0.0615628 (W 0.106632) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.20898 ± 0.01139 (W 0.00488936) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.24418 ± 0.0324949 (W 0.00738937) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 31.2609 ± 0.0781107 (W 0.0257489) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.3123 ± 0.0777267 (W 0.0396112) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.45319 ± 0.0234843 (W 0.00801017) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.47212 ± 0.00501268 (W 0.00499987) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 252.359 ± 0.806777 (W 0.798934) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 237.776 ± 0.8744 (W 0.889954) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 2.55456 ± 0.0184228 (W 0.00789709) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.47766 ± 0.0237308 (W 0.00733604) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 42.2145 ± 0.0801359 (W 0.047944) | 53.5516 ± 0.149139 (W 0.205539) | 🔵 faer-fork 1.27× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 43.4549 ± 0.131213 (W 0.0675888) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.858021 ± 0.00584456 (W 0.00758851) | 1.17485 ± 0.00818727 (W 0.00266692) | 🔵 faer-fork 1.37× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.891982 ± 0.00268044 (W 0.00311037) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 100.859 ± 0.499598 (W 0.404567) | 158.428 ± 1.00778 (W 0.887619) | 🔵 faer-fork 1.57× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 103.783 ± 0.559816 (W 0.466581) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.36069 ± 0.0116541 (W 0.00529924) | 3.07009 ± 0.0359687 (W 0.00730995) | 🔵 faer-fork 1.30× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 2.39642 ± 0.0163946 (W 0.00395753) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 95.013 ± 0.283531 (W 0.274079) | 149.86 ± 1.04899 (W 1.01738) | 🔵 faer-fork 1.58× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 2.22698 ± 0.00785603 (W 0.00450783) | 2.88119 ± 0.0403778 (W 0.00833519) | 🔵 faer-fork 1.29× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 272.504 ± 0.807813 (W 0.511062) | 655.604 ± 2.46359 (W 1.22217) | 🔵 faer-fork 2.41× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 272.73 ± 0.716009 (W 6.06687) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.48044 ± 0.341391 (W 0.00785916) | 13.6525 ± 1.541 (W 0.0255198) | 🔵 faer-fork 2.49× faster than `lapack-openblas` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.4609 ± 0.349862 (W 0.0111771) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 114.841 ± 1.30227 (W 0.83554) | 308.365 ± 1.27863 (W 0.954501) | 🔵 faer-fork 2.69× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 113.519 ± 1.22681 (W 0.417154) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.54067 ± 0.00562497 (W 0.00415441) | 7.1672 ± 0.0454393 (W 0.0137575) | 🔵 faer-fork 2.82× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.50464 ± 0.00990488 (W 0.00596947) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 90.066 ± 0.327451 (W 0.68117) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 85.8825 ± 0.60543 (W 0.563081) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.21904 ± 0.0166147 (W 0.00415031) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.24618 ± 0.00351569 (W 0.00400692) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.3709 ± 0.549735 (W 0.45692) | 28.5753 ± 0.0573164 (W 0.023261) | 🔵 faer-fork 2.51× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 12.9516 ± 0.548851 (W 0.445018) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.254652 ± 0.0326762 (W 0.0376632) | 0.699408 ± 0.00611554 (W 0.00170667) | 🔵 faer-fork 2.75× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.379354 ± 0.00171876 (W 0.0437538) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.2127 ± 0.0964904 (W 0.270919) | 58.1562 ± 0.199529 (W 0.240232) | 🔵 faer-fork 2.22× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.8431 ± 0.0669145 (W 0.0402106) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.708484 ± 0.00164834 (W 0.00298438) | 1.08332 ± 0.00117082 (W 0.00192511) | 🔵 faer-fork 1.53× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.766794 ± 0.00075872 (W 0.00229214) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 24.5659 ± 0.0446905 (W 0.0230349) | 53.9142 ± 0.361329 (W 0.146962) | 🔵 faer-fork 2.19× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 0.649161 ± 0.00188829 (W 0.0023957) | 0.952604 ± 0.00226204 (W 0.00154412) | 🔵 faer-fork 1.47× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 99.4341 ± 0.147797 (W 0.177458) | 414.652 ± 0.970054 (W 0.496614) | 🔵 faer-fork 4.17× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 100.095 ± 0.141127 (W 0.138816) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.69789 ± 0.00610888 (W 0.00481495) | 11.6362 ± 0.0383067 (W 0.0268688) | 🔵 faer-fork 4.31× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.72066 ± 0.00751249 (W 0.00396891) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 52.2665 ± 0.0997504 (W 0.310437) | 220.712 ± 0.626645 (W 0.483752) | 🔵 faer-fork 4.22× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 52.311 ± 0.139672 (W 0.079779) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.65271 ± 0.00484033 (W 0.00567591) | 6.57434 ± 0.0373884 (W 0.0238698) | 🔵 faer-fork 3.98× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.64827 ± 0.00888453 (W 0.00631919) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 53.1188 ± 0.79256 (W 0.162711) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 51.512 ± 0.688602 (W 0.481976) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.02956 ± 0.0232441 (W 0.00433827) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.08365 ± 0.0627031 (W 0.00265325) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 6.60222 ± 0.028758 (W 0.0104994) | 21.6259 ± 0.0326958 (W 0.0149274) | 🔵 faer-fork 3.28× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 7.55723 ± 0.0182632 (W 0.0213046) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.193318 ± 0.000220635 (W 0.00164969) | 0.562574 ± 0.00474434 (W 0.0020457) | 🔵 faer-fork 2.91× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.245203 ± 0.000771287 (W 0.00162355) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 13.8995 ± 0.048806 (W 0.014702) | 40.8565 ± 0.19004 (W 0.278465) | 🔵 faer-fork 2.94× faster than `lapack-openblas` |  |
| `qr/256x4096/b1 [api-inclusive]` | 15.3035 ± 0.0575005 (W 0.0244683) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.430299 ± 0.00377257 (W 0.00246986) | 0.793805 ± 0.002685 (W 0.00338034) | 🔵 faer-fork 1.84× faster than `lapack-openblas` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.501902 ± 0.00142181 (W 0.00258842) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.0054 ± 0.0407839 (W 0.0313807) | 37.4179 ± 0.496117 (W 0.193927) | 🔵 faer-fork 2.88× faster than `lapack-openblas` |  |
| `qr_factor/64x1024/b1` | 0.392911 ± 0.0009229 (W 0.00121919) | 0.665933 ± 0.00551939 (W 0.00272266) | 🔵 faer-fork 1.69× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1` | 74.364 ± 0.0982352 (W 0.0758525) | 377.148 ± 0.902031 (W 0.692667) | 🔵 faer-fork 5.07× faster than `lapack-openblas` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 75.0374 ± 0.0996778 (W 0.112578) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.31136 ± 0.00611443 (W 0.00398881) | 11.381 ± 0.0329449 (W 0.0181884) | 🔵 faer-fork 4.92× faster than `lapack-openblas` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.33651 ± 0.0114317 (W 0.00737932) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 43.2882 ± 0.107445 (W 0.0805954) | 207.48 ± 0.876543 (W 0.345986) | 🔵 faer-fork 4.79× faster than `lapack-openblas` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 43.2086 ± 0.106871 (W 0.0393163) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.54691 ± 0.00961201 (W 0.003943) | 6.56212 ± 0.0223674 (W 0.0189126) | 🔵 faer-fork 4.24× faster than `lapack-openblas` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.5512 ± 0.00526435 (W 0.00579265) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 24.3985 ± 0.120084 (W 0.0438952) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 38.0356 ± 0.404815 (W 0.16327) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.47443 ± 0.00313371 (W 0.00286184) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.990613 ± 0.00449501 (W 0.00374099) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.16975 ± 0.0221629 (W 0.00839905) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.29002 ± 0.0849258 (W 0.0566056) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.180909 ± 0.000647737 (W 0.00131216) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.359136 ± 0.00160167 (W 0.00211948) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 17.1038 ± 0.125429 (W 0.0318329) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 17.1986 ± 0.105641 (W 0.0718745) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.840013 ± 0.00204154 (W 0.00671744) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.608096 ± 0.00257335 (W 0.00269014) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.2832 ± 0.182912 (W 0.0253605) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.815988 ± 0.00449591 (W 0.00550897) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 41.9642 ± 0.195886 (W 0.0390381) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 42.1399 ± 0.195094 (W 0.0487666) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.36169 ± 0.00604474 (W 0.00309191) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.36195 ± 0.00440718 (W 0.00441445) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 18.6328 ± 0.0878948 (W 0.0189699) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 18.7049 ± 0.0783446 (W 0.0402645) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.696552 ± 0.00281619 (W 0.00191177) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.696247 ± 0.00277139 (W 0.00204776) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 9.12733 ± 0.0654475 (W 0.01415) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 19.303 ± 0.0854345 (W 0.0856301) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.348008 ± 0.00278572 (W 0.00318806) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.64076 ± 0.00226316 (W 0.00293672) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.99372 ± 0.0046967 (W 0.00674985) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 4.15708 ± 0.0535276 (W 0.0591212) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.081397 ± 0.000236809 (W 0.000528675) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.193012 ± 0.000616628 (W 0.00139581) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 6.78276 ± 0.0438708 (W 0.0211102) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 6.49287 ± 0.0515967 (W 0.025366) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.589809 ± 0.00860865 (W 0.00600393) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.359073 ± 0.00109676 (W 0.00393568) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.50198 ± 0.0407719 (W 0.0192324) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.578429 ± 0.00567277 (W 0.0039586) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 18.6108 ± 0.0216684 (W 0.0177856) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 18.6821 ± 0.0385354 (W 0.0283341) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.01633 ± 0.00520314 (W 0.00221105) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.02313 ± 0.00416037 (W 0.00210935) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 11.1439 ± 0.0173692 (W 0.0189011) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 11.1208 ± 0.0242317 (W 0.0235653) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.643623 ± 0.00323409 (W 0.00245656) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.645603 ± 0.00330089 (W 0.00235509) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 6.74205 ± 0.0438312 (W 0.0141277) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 16.118 ± 0.0877414 (W 0.0491133) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.355317 ± 0.00351885 (W 0.00365448) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.652185 ± 0.00335239 (W 0.00525134) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.29977 ± 0.0045012 (W 0.00804473) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.35713 ± 0.173557 (W 0.0544466) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.06694 ± 0.000656705 (W 0.000558065) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.167811 ± 0.000447105 (W 0.000867344) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 5.19207 ± 0.00892366 (W 0.017894) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.72385 ± 0.0683126 (W 0.0509296) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.550695 ± 0.00535485 (W 0.00529979) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.322429 ± 0.003977 (W 0.00634543) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 5.03711 ± 0.0106122 (W 0.0188404) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.537575 ± 0.00346672 (W 0.00527133) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 14.8266 ± 0.0400523 (W 0.0321406) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 14.9303 ± 0.0256556 (W 0.0255792) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.967308 ± 0.00338675 (W 0.00208506) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.969427 ± 0.00511887 (W 0.00267654) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 9.53399 ± 0.0192206 (W 0.0137373) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 9.56241 ± 0.0315454 (W 0.00931305) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.625548 ± 0.00337739 (W 0.00221108) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.626908 ± 0.00257536 (W 0.0024039) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 38.8965 ± 12.0507 (W 0.091232) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 35.8876 ± 8.73392 (W 0.0926903) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.915423 ± 0.00643844 (W 0.00443823) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.886393 ± 0.00564645 (W 0.00317864) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 7.24325 ± 0.0300853 (W 0.00853143) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 7.27917 ± 0.012975 (W 0.0183752) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.245161 ± 0.00234677 (W 0.00167212) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.250516 ± 0.00134214 (W 0.00144937) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 14.9141 ± 0.0671425 (W 0.0157059) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 14.8431 ± 0.0456029 (W 0.0172856) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.400345 ± 0.0018669 (W 0.00168901) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.485771 ± 0.00129131 (W 0.00251262) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 14.043 ± 0.0943925 (W 0.0261165) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.375881 ± 0.00179682 (W 0.00213042) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 44.7206 ± 0.660846 (W 0.0318806) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 44.9539 ± 0.620388 (W 0.109345) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.36212 ± 0.00203615 (W 0.00240187) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.36581 ± 0.00293043 (W 0.00355642) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.6144 ± 0.0617336 (W 0.0224193) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.6772 ± 0.0547327 (W 0.0375187) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.735915 ± 0.00580626 (W 0.00566775) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.736725 ± 0.00518774 (W 0.00577238) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.6257 ± 0.0647294 (W 0.0219717) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 17.2464 ± 0.0596334 (W 0.0464119) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.553011 ± 0.0046352 (W 0.0067942) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.544991 ± 0.00427995 (W 0.0036676) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.97909 ± 0.00501629 (W 0.00286985) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.21938 ± 0.0104533 (W 0.00489588) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0783955 ± 0.000344746 (W 0.000565437) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.088736 ± 0.000442635 (W 0.00044384) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 4.06048 ± 0.00852618 (W 0.00505518) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.37036 ± 0.0161339 (W 0.00502254) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.161715 ± 0.00114431 (W 0.00126396) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.240826 ± 0.00383372 (W 0.00460463) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.80696 ± 0.00608303 (W 0.00623894) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.147499 ± 0.000993281 (W 0.000269445) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 21.327 ± 0.0610609 (W 0.0372462) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 21.429 ± 0.0605009 (W 0.0307313) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.01782 ± 0.0056017 (W 0.00274526) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.0214 ± 0.0046384 (W 0.00293781) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 14.2902 ± 0.0357366 (W 0.0327052) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 14.3541 ± 0.0299002 (W 0.0322821) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.689478 ± 0.00483034 (W 0.00267178) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.688207 ± 0.00350173 (W 0.00314215) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 14.1188 ± 0.0282782 (W 0.0234359) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 14.1612 ± 0.038133 (W 0.0280842) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.553727 ± 0.00554735 (W 0.00684684) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.551019 ± 0.00433694 (W 0.00628971) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.10648 ± 0.00245156 (W 0.00395995) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.37932 ± 0.00549336 (W 0.00400997) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0509 ± 0.00377122 (W 0.0005068) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.063549 ± 0.000257469 (W 0.000589896) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 2.25905 ± 0.138953 (W 0.00616068) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 2.56315 ± 0.01316 (W 0.00652275) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.124442 ± 0.000650357 (W 0.000428602) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.209421 ± 0.00467627 (W 0.00572326) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 2.09915 ± 0.00644344 (W 0.00641953) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.111052 ± 0.000479995 (W 0.000279043) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 17.5733 ± 0.0639415 (W 0.0368412) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 17.6225 ± 0.0619001 (W 0.0534824) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.961017 ± 0.0036376 (W 0.00256063) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.970141 ± 0.00338733 (W 0.00210866) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 12.7601 ± 0.0597306 (W 0.0193462) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 12.7942 ± 0.0422134 (W 0.036307) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.672667 ± 0.00666241 (W 0.00370145) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.671295 ± 0.00357585 (W 0.00413466) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 58.5003 ± 0.628123 (W 0.48807) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 92.0624 ± 0.553527 (W 0.586929) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.886688 ± 0.00186676 (W 0.00244525) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.50656 ± 0.00439208 (W 0.00464622) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 21.9705 ± 0.0445523 (W 0.207843) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 26.6713 ± 0.17622 (W 0.0764608) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.537322 ± 0.00116789 (W 0.0023559) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.684344 ± 0.00107662 (W 0.00310856) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 54.7083 ± 0.206758 (W 0.0622339) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 58.2335 ± 0.412718 (W 0.442411) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.63617 ± 0.00361784 (W 0.00699713) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 1.47729 ± 0.00278973 (W 0.00357496) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 51.5825 ± 0.195213 (W 0.0746234) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.55211 ± 0.00374424 (W 0.00391869) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 135.176 ± 0.109483 (W 0.395777) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 135.3 ± 0.34327 (W 0.135795) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.04572 ± 0.00561317 (W 0.00689526) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.04973 ± 0.00213072 (W 0.00273841) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 53.9885 ± 0.0524502 (W 0.0491027) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 54.2004 ± 0.0629635 (W 0.108902) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.42787 ± 0.00333212 (W 0.00400486) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.43031 ± 0.0032412 (W 0.00437388) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.0458 ± 0.0713122 (W 0.0266614) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 36.8907 ± 0.126609 (W 0.12509) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.49027 ± 0.00145122 (W 0.00360499) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.834336 ± 0.0148228 (W 0.0038203) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.24682 ± 0.0132664 (W 0.0213448) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.85775 ± 0.104223 (W 0.0841308) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.198689 ± 0.00249813 (W 0.00246735) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.310226 ± 0.000761263 (W 0.00158616) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 16.8555 ± 0.1033 (W 0.0867649) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 18.0876 ± 0.1132 (W 0.0469173) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.744104 ± 0.00259183 (W 0.00310442) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.580207 ± 0.00211433 (W 0.00303962) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 15.9975 ± 0.0509232 (W 0.0293165) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.706932 ± 0.00229499 (W 0.00263285) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 48.297 ± 0.146655 (W 0.158825) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 48.4319 ± 0.146116 (W 0.0839261) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.65838 ± 0.0053095 (W 0.00604033) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.66921 ± 0.00316572 (W 0.00722867) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 24.0379 ± 0.0469118 (W 0.0495416) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.0731 ± 0.0473392 (W 0.0478285) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.005 ± 0.00296827 (W 0.00271548) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.00841 ± 0.00333943 (W 0.00328537) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.7861 ± 0.0565256 (W 0.019887) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 29.7349 ± 0.118859 (W 0.100421) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.450562 ± 0.00144535 (W 0.00305386) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.764199 ± 0.00309541 (W 0.00239744) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.55305 ± 0.0132096 (W 0.020996) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.96471 ± 0.0234258 (W 0.0436965) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.150611 ± 0.000603783 (W 0.00156558) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.254906 ± 0.00177276 (W 0.00197614) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.5724 ± 0.0578554 (W 0.0186729) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 11.1717 ± 0.0847596 (W 0.0382705) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.591006 ± 0.00174026 (W 0.00408442) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.432752 ± 0.00180308 (W 0.00195385) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.1231 ± 0.0693729 (W 0.0298158) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.565476 ± 0.00257134 (W 0.00437773) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 35.1507 ± 0.0597334 (W 0.0672757) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 35.4078 ± 0.0836055 (W 0.0754023) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.46147 ± 0.00462386 (W 0.00250632) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.47208 ± 0.0095166 (W 0.00371345) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 18.9704 ± 0.0444839 (W 0.0289849) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 18.9932 ± 0.0220275 (W 0.0270861) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.954964 ± 0.00346983 (W 0.00213306) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.959091 ± 0.00446525 (W 0.00272379) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 97.3292 ± 0.439847 (W 0.602737) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 88.1741 ± 1.02011 (W 0.291119) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.40727 ± 0.00641246 (W 0.00542843) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.38029 ± 0.0045766 (W 0.00286931) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 22.7598 ± 0.0746954 (W 0.0317617) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 23.2965 ± 0.19926 (W 0.106371) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.552091 ± 0.00347824 (W 0.0020975) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.565545 ± 0.00347906 (W 0.00191976) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 53.5777 ± 0.210979 (W 0.0839733) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 54.5697 ± 0.298192 (W 0.225372) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.33049 ± 0.00265458 (W 0.00310604) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 1.3552 ± 0.00393405 (W 0.00266098) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 50.4692 ± 0.191173 (W 0.11475) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.25342 ± 0.00382007 (W 0.00228868) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 140.33 ± 0.254399 (W 0.434656) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 140.646 ± 0.127369 (W 0.173074) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.09655 ± 0.00695863 (W 0.00512429) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.11696 ± 0.00980804 (W 0.00717244) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 59.563 ± 0.0813247 (W 0.0608731) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 59.6433 ± 0.154844 (W 0.0757975) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.50851 ± 0.00285206 (W 0.00490933) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.51308 ± 0.00509782 (W 0.00458635) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 35.1318 ± 0.626257 (W 0.045012) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 33.8612 ± 2.02648 (W 0.0616684) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.699514 ± 0.0289125 (W 0.00233747) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.706457 ± 0.017347 (W 0.0038287) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 6.06945 ± 0.00890231 (W 0.0115277) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.53811 ± 0.0262686 (W 0.0105412) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.167135 ± 0.000439566 (W 0.00114321) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.188703 ± 0.000524312 (W 0.00134133) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 13.8883 ± 0.0301753 (W 0.02245) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 14.5161 ± 0.0924091 (W 0.0308345) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.424624 ± 0.000773186 (W 0.00168385) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.459631 ± 0.00107013 (W 0.00181369) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.0478 ± 0.0225124 (W 0.0176254) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.390543 ± 0.00235423 (W 0.00212268) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 53.5238 ± 0.12026 (W 0.0842044) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 53.625 ± 0.0442048 (W 0.0881938) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.69801 ± 0.00318769 (W 0.0054143) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.71229 ± 0.00703056 (W 0.00352326) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 29.5412 ± 0.0657345 (W 0.0541798) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 29.5794 ± 0.0529831 (W 0.0584818) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.07884 ± 0.0053128 (W 0.00504572) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.076 ± 0.00741183 (W 0.00455112) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 27.1484 ± 0.337261 (W 0.0858705) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 26.5331 ± 0.0835307 (W 0.056869) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.61442 ± 0.0109567 (W 0.00272447) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.631072 ± 0.0209007 (W 0.00207677) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 3.33462 ± 0.00736834 (W 0.0110724) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.71313 ± 0.00787095 (W 0.00855669) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.103728 ± 0.000891936 (W 0.000491248) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.129765 ± 0.000758832 (W 0.000517077) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.29711 ± 0.0268226 (W 0.0168052) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 7.83867 ± 0.0147719 (W 0.018744) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.273811 ± 0.0015683 (W 0.00133913) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.311776 ± 0.00159447 (W 0.00154403) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.81567 ± 0.0236214 (W 0.0138092) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.248026 ± 0.00117738 (W 0.00131063) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 40.5291 ± 0.0963755 (W 0.0618747) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 40.707 ± 0.0979946 (W 0.0966787) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.51045 ± 0.0806431 (W 0.00423396) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.51363 ± 0.00300828 (W 0.00334718) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 24.5596 ± 0.0678807 (W 0.0374675) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.5205 ± 0.0374653 (W 0.0479828) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.02363 ± 0.00858899 (W 0.00378161) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.02606 ± 0.00694505 (W 0.00496926) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `lu/256x4096`, `lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `solve/256x4096`, `solve/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
