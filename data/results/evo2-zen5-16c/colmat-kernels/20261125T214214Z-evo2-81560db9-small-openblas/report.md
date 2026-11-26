# `colmat-kernels` on `evo2-zen5-16c`

- colmat (fork, measured) commit: `81560db94498478e8dd65153739e208391e434b6`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- colmat / faer rows: `colmat` (the provider in the measured revision; see library and upstream above)
- harness commit: `5ee382265b3ed6081be88d113019fdd3797cac5b`
- toolkit commit: `b1302a25e837d7ab9070005fcb9afdc5e18afd37`
- measured binary SHA-256: `9c3505b61edaff777f1a96860f1a5c90e91e1a43752c812e4fad4001963fdf13`
- hardware profile: `evo2-zen5-16c`
- host role: `measurement` — published dataset host
- timestamp: `2026-11-25T22:53:06.939766Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py evo2-zen5-16c colmat-kernels --vendor openblas --engines colmat,lapack-openblas --threads 1,4,8 --reps 5 --aa 6 --prime-ms 250 --label evo2-81560db9-small-openblas`
- harness environment: `FBENCH_ORDER` unset, `FBENCH_PROFILE=evo2-zen5-16c`
- raw data: `data/results/evo2-zen5-16c/colmat-kernels/20261125T214214Z-evo2-81560db9-small-openblas/`

## What was measured

- **Workload** — Direct calls into one colmat revision and its upstream faer baseline: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the colmat and faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: No row allocates inside its timed `run`: the vendor row calls LAPACKE's `_work` entry points with a workspace the op allocated in `prepare`, and the colmat and faer rows call their in-place/`*_into` routes on a prepared matrix, prepared rights and a `MemStack` sized by the library's own `*_scratch`. A vendored driver family is measured column-major only, because LAPACKE's row-major path allocates a transposed copy and transposes in and out of it inside the call in every `_work` variant; a C-order cell of such a family is recorded `skipped` with that reason rather than charged a conversion the colmat row does not pay. `gemm` keeps both orders (CBLAS's row-major entry point moves no data).
  Known blind spot: The owning colmat/faer routes (`PartialPivLu::new`, `Llt::new`, `qr`, `gesdd`/`syevd` wrappers, `partial_piv_lu().solve()`) allocate their outputs and their scratch inside the timed call. They are kept, as the call a user makes, but labelled `api-inclusive` and published as a separate secondary table: an `api-inclusive` row is never divided against a vendor row, whose route is the prepared one.
  Known blind spot: A library's own allocation inside a timed call is not visible to the harness's allocator probe when it happens in C: LAPACKE's non-`_work` wrappers reach libc `malloc` through `LAPACKE_malloc`. That route is not used by this harness (the probe would not see it), so the `_work` entry points are enforced by construction and by the column-major rule, not by the probe.
  Known blind spot: `gemm` covers the tall class's two Householder legs (a 16-wide basis and a 240-column target over 4096 rows: the reduction and the write-back) beside a vendor `?gemm` on the same leg. A single leg beside a vendor gemm is a diagnostic reference, not the equivalent implementation of the whole update, and the row is labelled accordingly.
  Known blind spot: `householder_apply` measures block apply and the one-block sequence form as two distinct rows, labelled with basis width min(m,n) and target width n. QR fixture preparation is outside timing. Independently varied basis/target widths and multi-panel sequences are not covered.
  Known blind spot: A copy a library makes inside its own call is that library's cost, and it is never divided against a vendor row: on a row-major matrix the fork's LU factors into a column-major copy it allocates (`lu/partial_pivoting/factor.rs:396-418`), and a row-major cell of a vendored driver family is a `skipped` cell, so no vendor ratio is formed there. Allocation performed inside the measured call stays inside the clock, and no _primary_ row does it. The fork's two row splits allocate their partial buffers inside the public call and the public scratch function does not expose them; that is the fork's cost, charged to the fork, and the owning routes that pay it are published as `api-inclusive` rows rather than divided against a vendor row.
  Known blind spot: `cholesky`, `eigh`, `solve` and `triangular_solve` are measured on square shapes only: the matrix they factor or solve with is square by definition (self-adjoint for the first two, a triangular factor for the last). `lu` and `full_piv_lu` are not gated that way -- they take the declared rectangular shapes too, with `min(m, n)` pivots and a trapezoidal `L` or `U`, which is what `dgetrf` does as well -- so a rectangular LU cell is measured in both orders beside its square one.
  Known blind spot: `f32`/`c32` are declared and measured as of `docs/harness-dtype-axis.md`; the vendor arm has a route for `gemm` in those two dtypes and reports the LAPACK families as skipped for them. The large regime's 1024 shapes, BLIS, sparse, iterative and GPU arms are later declarations, not machinery built now.
  Known blind spot: A vendor arm records the library it linked. OpenBLAS and AOCL report the kernel their runtime dispatched to (AOCL for BLIS and libflame separately); MKL names no kernel and no processor count, so its manifest leaves both empty rather than inferring them.
  Known blind spot: Known limitation: Coverage is the declared grid: f64/c64/f32/c32 at 1/4/8T for the small suite, f64/c64 at 4/8T for the large suite's 1024/1536 + boundary tier; the adoption-only 2048/3072/4096 tiers are not measured.
- **Rows** — one row per engine in every table below:
  - `colmat` — The measured library: the tensor4all fork (`colmat`, `[lib] name = "colmat"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/colmat.rev`
    Caveat: The fork crate is `colmat` (`[lib] name = "colmat"`); the upstream baseline is `faer` (`faer_upstream = { package = "faer" }`). They are distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. The twelve-family campaign measures 81560db9, colmat-rs main at the merge of #296 (t4a-private-gemm-x86 0.1.30); no tag names it. The zen5-cpu, ohtaka and keyaki recordings were taken at 93da6217, the 2026-11-17 head, which lacks the 46 colmat-rs merges in between (the LU look-ahead #239 among them). The gemm-crate bump itself is expected to be numerically inert here: colmat-rs#296 found all 112 result fingerprints bit-identical between 0.1.28 and 0.1.30, and its one new path (the packed AVX2 driver) does not engage on AMD Zen5 or on AVX-512 hosts.
    As measured: `colmat` (the provider in the measured revision; see library and upstream above)
  - `lapack-openblas` — CBLAS `?gemm` plus LAPACK drivers linked against OpenBLAS: vendor calls and their marshalling, vendor-owned threading. The vendor's own thread budget is set by the harness before the row and read back, and the vendor identity and the dispatched kernel are recorded in the manifest.
    Identity: the OpenBLAS build recorded in this run's manifest (`providers[openblas]`)
    Caveat: LAPACK and BLAS own their threading, so this row's thread budget is the vendor's, not the harness pool's. Where a family has no LAPACK route the row is absent and the CSV says the case was skipped for it.
    As measured: linked static; openblas 0.3.32; kernel dispatched to `Cooperlake`; pthread threading; config `OpenBLAS 0.3.32 DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=32`

## Hardware

- CPU: `AMD Ryzen AI MAX+ 395`
- logical CPUs: `32`
- L3: `64 MiB (2 instances)`
- L3 domains: `0-7,16-23: 32 MiB shared; 8-15,24-31: 32 MiB shared`
- OS / arch: `Linux 7.0.0-42-generic` / `x86_64`
- hostname: `EVO-X2`
- CPU sets: 1T -> `0`, 4T -> `0-3`, 8T -> `0-7`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.36%, worst 91181.43% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 123 of 1812 engine cells exceed 5% and carry `!`; median within-process relative SD 0.53%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
The `vs colmat` column is the rotation-averaged ratio
exp(mean over starts of log(engine median / colmat median)),
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

## gemm-small, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00094 ± 8.94427e-06 (W 2.9242e-05) | 0.000775 ± 1.3784e-05 (W 3.20079e-05) | 🔴 colmat 1.21× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00617 ± 4.07022e-05 (W 3.33071e-05) | 0.00569 ± 1.09545e-05 (W 3.17989e-05) | 🔴 colmat 1.08× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0197995 ± 7.84364e-05 (W 3.76153e-05) | 0.019315 ± 0.000148527 (W 4.97322e-05) | · | dispatch-dominated (row under 20 us) |

## gemm-small, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00096 ± 9.75192e-05 (W 0.000152715) | 0.00078 ± 9.83192e-06 (W 2.98159e-05) | 🔴 colmat 1.23× slower than `lapack-openblas` (marginal: spread 0.85 of the difference) | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0023895 ± 9.56251e-05 (W 4.09097e-05) | 0.00574 ± 5.16398e-05 (W 3.72633e-05) | 🔵 colmat 2.40× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00591 ± 2.56255e-05 (W 5.43083e-05) | 0.019354 ± 0.000119129 (W 0.000142121) | 🔵 colmat 3.27× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001005 ± 3.40588e-05 (W 3.58059e-05) | 0.000775 ± 9.83192e-06 (W 2.95904e-05) | 🔴 colmat 1.30× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00181 ± 4.57894e-05 (W 9.77934e-05) | 0.00574 ± 4.09911e-05 (W 2.90977e-05) | 🔵 colmat 3.17× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.003635 ± 3.83167e-05 (W 4.72133e-05) | 0.0191945 ± 0.000103969 (W 4.42048e-05) | 🔵 colmat 5.28× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.000955 ± 1.51658e-05 (W 2.54601e-05) | 0.00078 ± 1.16905e-05 (W 3.28634e-05) | 🔴 colmat 1.22× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0062 ± 2.33809e-05 (W 4.02316e-05) | 0.0057195 ± 0.000567219 (W 3.1444e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.019854 ± 0.00010028 (W 5.69896e-05) | 0.0193095 ± 9.05185e-05 (W 5.16644e-05) | · | dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.000995 ± 6.44205e-05 (W 4.86392e-05) | 0.0007745 ± 9.66954e-06 (W 3.27871e-05) | 🔴 colmat 1.28× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.002325 ± 3.63524e-05 (W 1.84045e-05) | 0.0056895 ± 8.95358e-06 (W 2.66157e-05) | 🔵 colmat 2.45× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0057895 ± 3.33207e-05 (W 3.53601e-05) | 0.018944 ± 4.59017e-05 (W 0.000131627) | 🔵 colmat 3.27× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.000965 ± 6.33772e-05 (W 8.50083e-05) | 0.000785 ± 1.47196e-05 (W 2.89203e-05) | 🔴 colmat 1.23× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.001784 ± 5.51023e-05 (W 3.21515e-05) | 0.005719 ± 2.80404e-05 (W 0.000133372) | 🔵 colmat 3.21× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00343 ± 4.17931e-05 (W 3.62268e-05) | 0.019409 ± 0.000190214 (W 4.1879e-05) | 🔵 colmat 5.66× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.003235 ± 8.16497e-06 (W 3.397e-05) | 0.00451 ± 1.0328e-05 (W 1.24395e-05) | 🔵 colmat 1.39× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.02322 ± 8.61294e-05 (W 5.23905e-05) | 0.032299 ± 6.20336e-05 (W 0.000112669) | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.077213 ± 0.000149499 (W 9.729e-05) | 0.100922 ± 0.000495819 (W 0.000808445) | 🔵 colmat 1.31× faster than `lapack-openblas` |  |

## gemm-small, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001535 ± 5.25357e-05 (W 2.71978e-05) | 0.004505 ± 8.4951e-06 (W 1.9225e-05) | 🔵 colmat 2.93× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0069745 ± 5.73783e-05 (W 3.9367e-05) | 0.01129 ± 0.000159073 (W 0.000255115) | 🔵 colmat 1.62× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020399 ± 5.48103e-05 (W 4.86851e-05) | 0.031714 ± 0.000267089 (W 0.000245946) | 🔵 colmat 1.55× faster than `lapack-openblas` |  |

## gemm-small, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001435 ± 3.25064e-05 (W 3.55986e-05) | 0.00452 ± 1.70929e-05 (W 2.4019e-05) | 🔵 colmat 3.15× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.003985 ± 6.58534e-05 (W 4.45275e-05) | 0.01133 ± 0.000136495 (W 0.000158944) | 🔵 colmat 2.84× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0109195 ± 1.4148e-05 (W 4.1307e-05) | 0.023035 ± 0.000179277 (W 0.000277042) | 🔵 colmat 2.11× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0032395 ± 7.49444e-06 (W 3.27109e-05) | 0.004515 ± 2.48328e-05 (W 2.31929e-05) | 🔵 colmat 1.39× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.023235 ± 6.76976e-05 (W 4.81163e-05) | 0.032264 ± 6.70634e-05 (W 9.63471e-05) | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `gemm/96x96x96/b1` | 0.077147 ± 0.000309346 (W 0.000122813) | 0.100752 ± 0.000314433 (W 0.000459553) | 🔵 colmat 1.31× faster than `lapack-openblas` |  |

## gemm-small, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001565 ± 2.92689e-05 (W 5.40746e-05) | 0.004499 ± 1.02111e-05 (W 1.87691e-05) | 🔵 colmat 2.87× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.006815 ± 2.25093e-05 (W 2.95176e-05) | 0.01127 ± 0.0001504 (W 0.000278878) | 🔵 colmat 1.65× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0203695 ± 3.18795e-05 (W 6.76059e-05) | 0.031759 ± 0.000521941 (W 0.00082839) | 🔵 colmat 1.56× faster than `lapack-openblas` |  |

## gemm-small, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0013245 ± 3.66629e-05 (W 2.65428e-05) | 0.0045045 ± 1.47026e-05 (W 2.44293e-05) | 🔵 colmat 3.40× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00416 ± 4.74289e-05 (W 2.42561e-05) | 0.0113545 ± 6.18568e-05 (W 0.000116755) | 🔵 colmat 2.73× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.011015 ± 8.94218e-05 (W 5.4249e-05) | 0.0229895 ± 0.000262106 (W 0.0002599) | 🔵 colmat 2.09× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00054 ± 4.08248e-06 (W 2.81064e-05) | 0.000445 ± 8.16497e-06 (W 2.34482e-05) | 🔴 colmat 1.21× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00301 ± 1.61297e-05 (W 2.93428e-05) | 0.00285 ± 1.20069e-05 (W 2.82984e-05) | 🔴 colmat 1.06× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.009725 ± 9.83192e-06 (W 2.93701e-05) | 0.0094895 ± 4.54478e-05 (W 3.78647e-05) | · | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00062 ± 7.52773e-06 (W 2.90313e-05) | 0.000435 ± 9.83192e-06 (W 2.30341e-05) | 🔴 colmat 1.43× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.001805 ± 6.31401e-05 (W 0.000120743) | 0.002855 ± 1.47196e-05 (W 3.31649e-05) | 🔵 colmat 1.58× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.003225 ± 3.07679e-05 (W 5.21722e-05) | 0.0094945 ± 2.26311e-05 (W 2.94729e-05) | 🔵 colmat 2.94× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00062 ± 8.94427e-06 (W 3.4952e-05) | 0.000445 ± 1.00747e-05 (W 2.09545e-05) | 🔴 colmat 1.39× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.001765 ± 5.46275e-05 (W 4.63681e-05) | 0.00286 ± 5.16398e-06 (W 3.16993e-05) | 🔵 colmat 1.62× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.002895 ± 3.32666e-05 (W 2.79732e-05) | 0.0094995 ± 3.60966e-05 (W 4.0323e-05) | 🔵 colmat 3.28× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00054 ± 5.16398e-06 (W 3.03646e-05) | 0.00045 ± 8.16497e-06 (W 1.98775e-05) | 🔴 colmat 1.20× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00302 ± 2.52982e-05 (W 3.11338e-05) | 0.0028695 ± 5.04975e-06 (W 2.86223e-05) | 🔴 colmat 1.05× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0097395 ± 3.17742e-05 (W 3.78044e-05) | 0.00948 ± 1.87661e-05 (W 2.73422e-05) | · | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0006 ± 1.0328e-05 (W 3.30783e-05) | 0.000435 ± 1.04881e-05 (W 1.90285e-05) | 🔴 colmat 1.38× slower than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.001745 ± 5.77639e-05 (W 7.42042e-05) | 0.00285 ± 1.07781e-05 (W 2.85163e-05) | 🔵 colmat 1.63× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.003245 ± 7.31437e-05 (W 4.74035e-05) | 0.0094695 ± 1.86592e-05 (W 2.73572e-05) | 🔵 colmat 2.92× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00061 ± 1.0328e-05 (W 3.11033e-05) | 0.000445 ± 8.16497e-06 (W 2.46973e-05) | 🔴 colmat 1.37× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0013895 ± 3.55551e-05 (W 2.9103e-05) | 0.002865 ± 5.47723e-06 (W 2.91781e-05) | 🔵 colmat 2.06× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00231 ± 3.32666e-05 (W 3.58133e-05) | 0.009495 ± 1.60208e-05 (W 2.52828e-05) | 🔵 colmat 4.11× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0016045 ± 1.1669e-05 (W 2.7404e-05) | 0.002245 ± 8.16497e-06 (W 2.30772e-05) | 🔵 colmat 1.40× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01206 ± 1.51976e-05 (W 3.99424e-05) | 0.014615 ± 7.14714e-05 (W 5.60872e-05) | 🔵 colmat 1.21× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.038398 ± 6.7798e-05 (W 5.09788e-05) | 0.044644 ± 0.00012534 (W 8.93088e-05) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |

## gemm-small, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00117 ± 3.43026e-05 (W 5.6455e-05) | 0.00222 ± 6.32456e-06 (W 2.1208e-05) | 🔵 colmat 1.90× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0037245 ± 1.64165e-05 (W 5.07025e-05) | 0.005445 ± 9.27586e-05 (W 0.000224558) | 🔵 colmat 1.46× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0105395 ± 3.16064e-05 (W 5.23398e-05) | 0.014925 ± 0.000150752 (W 0.000209223) | 🔵 colmat 1.42× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00129 ± 5.46504e-05 (W 2.88952e-05) | 0.00223 ± 1.16905e-05 (W 3.56945e-05) | 🔵 colmat 1.73× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.002755 ± 3.07679e-05 (W 2.67529e-05) | 0.00565 ± 0.000122147 (W 0.000234217) | 🔵 colmat 2.05× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.005795 ± 3.68804e-05 (W 4.26472e-05) | 0.0114345 ± 0.000111348 (W 0.000142298) | 🔵 colmat 1.97× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0016 ± 4.08248e-06 (W 2.45993e-05) | 0.002225 ± 1.21106e-05 (W 2.21316e-05) | 🔵 colmat 1.39× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0120595 ± 1.87661e-05 (W 4.89198e-05) | 0.0146145 ± 5.29431e-05 (W 7.82354e-05) | 🔵 colmat 1.21× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.038349 ± 4.70337e-05 (W 4.7224e-05) | 0.0445785 ± 0.000179095 (W 6.83064e-05) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |

## gemm-small, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001125 ± 1.63299e-05 (W 5.93181e-05) | 0.00221 ± 1.04003e-05 (W 2.59406e-05) | 🔵 colmat 1.96× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.003725 ± 3.26599e-05 (W 4.35634e-05) | 0.0055395 ± 0.000141299 (W 0.000178061) | 🔵 colmat 1.49× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0105295 ± 3.08912e-05 (W 6.65236e-05) | 0.015 ± 0.000133583 (W 0.000214539) | 🔵 colmat 1.42× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00115 ± 3.27109e-05 (W 2.94663e-05) | 0.0022 ± 1.3784e-05 (W 2.42187e-05) | 🔵 colmat 1.91× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00257 ± 4.11906e-05 (W 4.28778e-05) | 0.0057795 ± 0.000166265 (W 0.000242777) | 🔵 colmat 2.25× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0061 ± 2.87518e-05 (W 4.00418e-05) | 0.0114145 ± 9.6955e-05 (W 0.00013464) | 🔵 colmat 1.87× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |

## gemm-mid, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.047464 ± 0.000124486 (W 6.49354e-05) | 0.0596835 ± 0.000107648 (W 9.16084e-05) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.159166 ± 0.00129544 (W 0.00159888) | 0.192114 ± 0.000153914 (W 0.00156288) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.378636 ± 0.00215232 (W 0.00171462) | 0.453332 ± 0.00227337 (W 0.00161529) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.2452 ± 0.00241244 (W 0.00167042) | 1.49098 ± 0.00577287 (W 0.00169663) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.92738 ± 0.00810511 (W 0.00180927) | 3.50439 ± 0.00478477 (W 0.00427664) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 10.0204 ± 0.0207403 (W 0.00688142) | 11.6037 ± 0.0506709 (W 0.0100645) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |

## gemm-mid, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0137345 ± 7.83343e-05 (W 7.02488e-05) | 0.021355 ± 0.00018559 (W 0.000655923) | 🔵 colmat 1.55× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.041753 ± 0.000182114 (W 0.000121506) | 0.0582385 ± 0.000303184 (W 0.00114962) | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.0966965 ± 0.000906892 (W 0.000273999) | 0.131221 ± 0.000460691 (W 0.00103612) | 🔵 colmat 1.36× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.317244 ± 0.0021321 (W 0.00128468) | 0.403832 ± 0.00241364 (W 0.00369607) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.778315 ± 0.00159576 (W 0.00161393) | 0.924325 ± 0.00201898 (W 0.00350542) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 2.58145 ± 0.0277147 (W 0.00985701) | 3.02879 ± 0.0114078 (W 0.0050875) | 🔵 colmat 1.17× faster than `lapack-openblas` |  |

## gemm-mid, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0072345 ± 6.54749e-05 (W 3.1952e-05) | 0.0147345 ± 4.63926 (W 0.00072186) | · | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.0209645 ± 5.264e-05 (W 4.90613e-05) | 0.035769 ± 0.000338807 (W 0.000452627) | 🔵 colmat 1.71× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.0518285 ± 0.000273113 (W 0.000195311) | 0.072942 ± 0.000659802 (W 0.000462818) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.159515 ± 0.000495784 (W 0.00023897) | 0.207395 ± 0.000367693 (W 0.00168012) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.414427 ± 0.00211612 (W 0.0037533) | 0.486289 ± 0.00185007 (W 0.00296832) | 🔵 colmat 1.17× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 1.33149 ± 0.00713126 (W 0.00733079) | 1.57858 ± 0.00392358 (W 0.0051208) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0475885 ± 0.000223259 (W 0.000188173) | 0.059703 ± 0.000128888 (W 9.83299e-05) | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.159005 ± 0.000258849 (W 0.00128445) | 0.192179 ± 0.000208414 (W 0.00145591) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.376783 ± 0.00230551 (W 0.00159227) | 0.45225 ± 0.00167467 (W 0.00162318) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.24445 ± 0.00254926 (W 0.00130957) | 1.48206 ± 0.00517004 (W 0.00226613) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.927 ± 0.0121564 (W 0.000920815) | 3.49865 ± 0.00721136 (W 0.00235201) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 10.007 ± 0.0206357 (W 0.00722002) | 11.6378 ± 0.0704944 (W 0.0122323) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0138045 ± 7.17189e-05 (W 8.58267e-05) | 0.021359 ± 0.000150405 (W 0.000387172) | 🔵 colmat 1.55× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.0416785 ± 0.000159802 (W 9.55418e-05) | 0.058268 ± 0.000557753 (W 0.000766581) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.097527 ± 0.000543243 (W 0.000251941) | 0.131346 ± 0.000559369 (W 0.000867235) | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.316064 ± 0.00123416 (W 0.00172504) | 0.402227 ± 0.00274099 (W 0.00320223) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.780061 ± 0.00187374 (W 0.00300541) | 0.923476 ± 0.00272143 (W 0.00328583) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 2.57883 ± 0.00558794 (W 0.00604358) | 3.0376 ± 0.0110265 (W 0.00979809) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |

## gemm-mid, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0079645 ± 4.93586e-05 (W 3.46259e-05) | 0.014345 ± 0.000351282 (W 0.000311093) | 🔵 colmat 1.80× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.0223495 ± 4.86638e-05 (W 6.54084e-05) | 0.035759 ± 0.000526742 (W 0.000480744) | 🔵 colmat 1.60× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.051648 ± 0.000604696 (W 0.000226839) | 0.0730775 ± 0.000525198 (W 0.000834911) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.165385 ± 0.0039524 (W 0.000411008) | 0.207738 ± 0.000739847 (W 0.00151178) | 🔵 colmat 1.26× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 0.413579 ± 0.00161793 (W 0.00450808) | 0.486978 ± 0.00184104 (W 0.00267457) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 1.33192 ± 0.00310969 (W 0.0069383) | 1.57891 ± 0.00338312 (W 0.00729831) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.184745 ± 0.000888953 (W 0.00125659) | 0.240781 ± 0.000636886 (W 0.00164188) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.610999 ± 0.000765057 (W 0.00175554) | 0.78308 ± 0.00136404 (W 0.00156866) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.46043 ± 0.00242841 (W 0.00252064) | 1.86177 ± 0.00343301 (W 0.0016612) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 4.83095 ± 0.00716178 (W 0.00282847) | 6.15776 ± 0.0117315 (W 0.00392741) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 11.4612 ± 0.0134157 (W 0.00502196) | 14.5268 ± 0.020728 (W 0.0106445) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 39.531 ± 0.021769 (W 0.0270735) | 48.9465 ± 0.133431 (W 0.0222584) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.046688 ± 0.000218502 (W 0.000113193) | 0.0685325 ± 0.000408664 (W 0.000727771) | 🔵 colmat 1.47× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.154185 ± 0.000473068 (W 0.000268502) | 0.207649 ± 0.000659418 (W 0.00156439) | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.377864 ± 0.00271555 (W 0.00184279) | 0.489517 ± 0.00196035 (W 0.002237) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.22609 ± 0.00222769 (W 0.00385484) | 1.58363 ± 0.00193625 (W 0.00290588) | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.9013 ± 0.00444142 (W 0.00486509) | 3.7354 ± 0.00669058 (W 0.00340688) | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 9.90636 ± 0.0160874 (W 0.0105839) | 12.5393 ± 0.0137186 (W 0.012468) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |

## gemm-mid, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0248595 ± 9.24347e-05 (W 5.14305e-05) | 0.037713 ± 0.000248206 (W 0.000348971) | 🔵 colmat 1.52× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.0778025 ± 0.000410832 (W 0.000140342) | 0.113262 ± 0.000376988 (W 0.000612724) | 🔵 colmat 1.46× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.185574 ± 0.000981755 (W 0.00166664) | 0.255768 ± 0.00145999 (W 0.00185472) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.625841 ± 0.00237797 (W 0.00350627) | 0.803037 ± 0.000723752 (W 0.00309608) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.48374 ± 0.00460985 (W 0.00409152) | 1.87582 ± 0.00413475 (W 0.0057645) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 4.99413 ± 0.0270791 (W 0.00660129) | 6.35236 ± 0.0246051 (W 0.0245452) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.185735 ± 0.000836825 (W 0.0013946) | 0.240113 ± 0.000328799 (W 0.00182856) | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.612184 ± 0.00292714 (W 0.0018255) | 0.783548 ± 0.00172916 (W 0.00183715) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 1.46098 ± 0.00163583 (W 0.00205202) | 1.8611 ± 0.00190303 (W 0.0028306) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 4.83398 ± 0.00605588 (W 0.00333402) | 6.17309 ± 0.0112079 (W 0.00406736) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 11.4722 ± 0.0146083 (W 0.00606951) | 14.5008 ± 0.0356521 (W 0.012255) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 39.5379 ± 0.0549583 (W 0.0395084) | 48.9483 ± 0.120458 (W 0.0341926) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0465235 ± 0.000288417 (W 9.71852e-05) | 0.0688725 ± 0.000324134 (W 0.000632279) | 🔵 colmat 1.48× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.15528 ± 0.000789112 (W 0.000282977) | 0.207574 ± 0.000325816 (W 0.00138986) | 🔵 colmat 1.34× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.377472 ± 0.00421804 (W 0.00164439) | 0.490261 ± 0.00122007 (W 0.00216965) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 1.22658 ± 0.00504694 (W 0.00321416) | 1.58778 ± 0.00548807 (W 0.00531785) | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 2.90302 ± 0.00476833 (W 0.0063187) | 3.73916 ± 0.0150821 (W 0.00702909) | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 9.90295 ± 0.0385163 (W 0.00826006) | 12.5383 ± 0.0128944 (W 0.00920801) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |

## gemm-mid, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0241845 ± 8.82965e-05 (W 5.49883e-05) | 0.037449 ± 0.000187581 (W 0.000244731) | 🔵 colmat 1.55× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.078838 ± 0.000731027 (W 0.000289178) | 0.113182 ± 0.000443097 (W 0.000317036) | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.195338 ± 0.00155525 (W 0.0017737) | 0.254888 ± 0.00115095 (W 0.00177962) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.625822 ± 0.00196039 (W 0.00336583) | 0.803194 ± 0.00243956 (W 0.00233341) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.4867 ± 0.00248234 (W 0.00574478) | 1.87018 ± 0.00663905 (W 0.00483757) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 4.99012 ± 0.00445827 (W 0.00834208) | 6.37186 ± 0.0467208 (W 0.0153155) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |

## gemm-mid, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.024354 ± 8.03005e-05 (W 0.000121021) | 0.026789 ± 0.000292994 (W 0.000123776) | 🔵 colmat 1.10× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.0772775 ± 0.000271879 (W 0.000460954) | 0.0843035 ± 0.000831091 (W 0.00106036) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.184796 ± 0.000268793 (W 0.0011125) | 0.196308 ± 0.00155635 (W 0.00131215) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.619135 ± 0.00358341 (W 0.00177828) | 0.655734 ± 0.00394634 (W 0.00168453) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.45262 ± 0.00354818 (W 0.00169678) | 1.53027 ± 0.0108904 (W 0.0019605) | 🔵 colmat 1.05× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 5.01746 ± 0.049081 (W 0.00419897) | 5.05481 ± 0.0144217 (W 0.00229477) | · |  |

## gemm-mid, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00657 ± 5.16746e-05 (W 5.74649e-05) | 0.011015 ± 6.41093e-05 (W 0.000344474) | 🔵 colmat 1.68× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.0208295 ± 8.27222e-05 (W 7.36163e-05) | 0.028969 ± 0.000248508 (W 0.000337582) | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.050418 ± 0.000182544 (W 0.00019348) | 0.0622025 ± 0.000480528 (W 0.00114123) | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.15679 ± 0.000578894 (W 0.000245552) | 0.183285 ± 0.000771616 (W 0.00182195) | 🔵 colmat 1.17× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.370149 ± 0.00194725 (W 0.00223792) | 0.42118 ± 0.00569697 (W 0.00415177) | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 1.28395 ± 0.00637585 (W 0.00621106) | 1.31655 ± 0.00716759 (W 0.00843563) | · |  |

## gemm-mid, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00448 ± 0.000128396 (W 7.73393e-05) | 0.00864 ± 0.000207481 (W 0.000278376) | 🔵 colmat 1.93× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.01126 ± 5.83007e-05 (W 3.53587e-05) | 0.0183945 ± 0.00015511 (W 0.00027622) | 🔵 colmat 1.63× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/256x256x256/b1` | 0.024989 ± 0.000106734 (W 5.22452e-05) | 0.034234 ± 0.000401448 (W 0.000501621) | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.0792325 ± 0.000174884 (W 9.30178e-05) | 0.097187 ± 0.000238451 (W 0.000845158) | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.196404 ± 0.000383506 (W 0.00153443) | 0.231907 ± 0.000985807 (W 0.00205589) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 0.681418 ± 0.00233783 (W 0.00360091) | 0.677479 ± 0.00432933 (W 0.00390122) | · |  |

## gemm-mid, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0243245 ± 0.00012148 (W 0.000124922) | 0.0266795 ± 0.000204053 (W 0.000145789) | 🔵 colmat 1.10× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.077268 ± 0.000146494 (W 0.00100351) | 0.084138 ± 0.000864166 (W 0.00123882) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.185264 ± 0.000380088 (W 0.00114617) | 0.197158 ± 0.000810699 (W 0.00129941) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.621627 ± 0.00614302 (W 0.00169814) | 0.654378 ± 0.00557108 (W 0.00223724) | 🔵 colmat 1.05× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.45334 ± 0.00512606 (W 0.00169892) | 1.53407 ± 0.0169454 (W 0.00217634) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 4.96497 ± 0.0288093 (W 0.0022057) | 5.04623 ± 0.00768822 (W 0.00421474) | · |  |

## gemm-mid, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0064895 ± 3.27867e-05 (W 2.8948e-05) | 0.0108295 ± 0.000215387 (W 0.000475162) | 🔵 colmat 1.67× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.020885 ± 9.22041e-05 (W 6.93953e-05) | 0.028859 ± 0.000377197 (W 0.000538446) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.050343 ± 0.000180577 (W 0.000164754) | 0.062328 ± 0.000611174 (W 0.00109438) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.156571 ± 0.000390185 (W 0.000279723) | 0.184124 ± 0.000537856 (W 0.00312432) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.368849 ± 0.00385039 (W 0.00139382) | 0.414303 ± 0.00392525 (W 0.00264808) | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 1.28187 ± 0.00344232 (W 0.00486203) | 1.31185 ± 0.00783125 (W 0.00803323) | · |  |

## gemm-mid, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00404 ± 5.5715e-05 (W 6.05786e-05) | 0.008645 ± 0.000227933 (W 0.000308327) | 🔵 colmat 2.14× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.0114145 ± 3.79245e-05 (W 5.37339e-05) | 0.0182545 ± 0.000150876 (W 0.00035133) | 🔵 colmat 1.60× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/256x256x256/b1` | 0.026729 ± 4.94115e-05 (W 0.000112388) | 0.034309 ± 0.00028859 (W 0.00110491) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.0810575 ± 0.000333476 (W 0.000140556) | 0.0968715 ± 0.00137771 (W 0.000458363) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.192559 ± 0.00108053 (W 0.00164306) | 0.232637 ± 0.00212548 (W 0.00185476) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 0.680789 ± 0.00231886 (W 0.00414895) | 0.672182 ± 0.00341749 (W 0.00293017) | · |  |

## gemm-mid, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.090517 ± 0.00025492 (W 0.000441031) | 0.102696 ± 0.000243438 (W 0.000339906) | 🔵 colmat 1.13× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.303629 ± 0.000926819 (W 0.0015514) | 0.330624 ± 0.000326713 (W 0.00155246) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.730118 ± 0.00391573 (W 0.00146088) | 0.794983 ± 0.00212239 (W 0.00166233) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 2.41853 ± 0.00361851 (W 0.00194858) | 2.5467 ± 0.00754003 (W 0.0017632) | 🔵 colmat 1.05× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 5.72033 ± 0.0218365 (W 0.00376005) | 6.06691 ± 0.00489934 (W 0.00286978) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 19.5091 ± 0.0343715 (W 0.0105483) | 20.1175 ± 0.161034 (W 0.00936525) | · |  |

## gemm-mid, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0245495 ± 6.47686e-05 (W 4.65237e-05) | 0.031114 ± 0.000361976 (W 0.000431689) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.0770375 ± 0.000221053 (W 0.00012596) | 0.088593 ± 0.000334466 (W 0.000540902) | 🔵 colmat 1.15× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.182783 ± 0.00059088 (W 0.000679932) | 0.208479 ± 0.000589217 (W 0.00185163) | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.611085 ± 0.00226968 (W 0.00160964) | 0.654514 ± 0.00394751 (W 0.0023579) | 🔵 colmat 1.07× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.46968 ± 0.00325184 (W 0.0041907) | 1.64105 ± 0.00597682 (W 0.00472574) | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 4.91824 ± 0.00925101 (W 0.00722032) | 5.19006 ± 0.0316829 (W 0.0083865) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |

## gemm-mid, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0124995 ± 3.30192e-05 (W 3.75667e-05) | 0.0180895 ± 0.000162281 (W 0.000178235) | 🔵 colmat 1.45× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.0390535 ± 0.000138923 (W 5.79926e-05) | 0.051393 ± 0.000396278 (W 0.000418271) | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.0963265 ± 0.000266106 (W 0.00016405) | 0.114681 ± 0.00060598 (W 0.00135196) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.305612 ± 0.000735243 (W 0.00174617) | 0.334854 ± 0.00132925 (W 0.00247513) | 🔵 colmat 1.10× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.756944 ± 0.002007 (W 0.00256299) | 0.838487 ± 0.0215557 (W 0.00278498) | 🔵 colmat 1.11× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 2.49199 ± 0.00435846 (W 0.00677436) | 2.59975 ± 0.00923686 (W 0.0263295) | · |  |

## gemm-mid, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.090327 ± 0.000163255 (W 0.000118006) | 0.102451 ± 0.000382425 (W 0.000653049) | 🔵 colmat 1.13× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.303862 ± 0.0010564 (W 0.00131359) | 0.330925 ± 0.000498706 (W 0.00148333) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.726835 ± 0.00201323 (W 0.00147867) | 0.794056 ± 0.000960254 (W 0.00203459) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 2.41841 ± 0.00390368 (W 0.00211533) | 2.5542 ± 0.00523866 (W 0.00193563) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 5.71671 ± 0.0183116 (W 0.0068951) | 6.07405 ± 0.0170388 (W 0.0028752) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 19.5152 ± 0.018323 (W 0.00979872) | 20.0007 ± 0.099716 (W 0.00723217) | · |  |

## gemm-mid, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0245595 ± 8.42987e-05 (W 7.86391e-05) | 0.031474 ± 0.000448714 (W 0.000504154) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `gemm/192x192x192/b1` | 0.0769025 ± 0.000211319 (W 0.000119404) | 0.0889715 ± 0.000535291 (W 0.000993281) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.182149 ± 0.00131476 (W 0.00115284) | 0.208338 ± 0.00145925 (W 0.00159078) | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.610194 ± 0.00152425 (W 0.00170474) | 0.654892 ± 0.00237317 (W 0.00325974) | 🔵 colmat 1.07× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 1.47057 ± 0.00162043 (W 0.00237822) | 1.6483 ± 0.0130425 (W 0.00341096) | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 4.92573 ± 0.00499323 (W 0.0076464) | 5.18061 ± 0.0291301 (W 0.00983203) | 🔵 colmat 1.05× faster than `lapack-openblas` |  |

## gemm-mid, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0130345 ± 8.17451e-05 (W 3.70849e-05) | 0.01808 ± 0.000207625 (W 0.000338407) | 🔵 colmat 1.39× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.039609 ± 0.000237382 (W 5.49904e-05) | 0.051543 ± 0.000338119 (W 0.00047295) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `gemm/256x256x256/b1` | 0.092382 ± 0.0010127 (W 0.000234523) | 0.115032 ± 0.000640157 (W 0.000973528) | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `gemm/384x384x384/b1` | 0.312219 ± 0.00240144 (W 0.00175276) | 0.335593 ± 0.00208425 (W 0.00204045) | 🔵 colmat 1.07× faster than `lapack-openblas` |  |
| `gemm/512x512x512/b1` | 0.756732 ± 0.0019735 (W 0.00269603) | 0.832645 ± 0.00458988 (W 0.00325625) | 🔵 colmat 1.10× faster than `lapack-openblas` |  |
| `gemm/768x768x768/b1` | 2.48832 ± 0.0044188 (W 0.00471317) | 2.59856 ± 0.0353172 (W 0.0260605) | · |  |

## gemm-tall, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.529778 ± 0.00146052 (W 0.00305017) | 0.630339 ± 0.000944491 (W 0.00197428) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.508603 ± 0.00340694 (W 0.00240206) | 0.735841 ± 0.012796 (W 0.0145144) | 🔵 colmat 1.45× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.381889 ± 0.00429176 (W 0.00155211) | 0.507684 ± 0.000602679 (W 0.00179135) | 🔵 colmat 1.33× faster than `lapack-openblas` |  |

## gemm-tall, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.132601 ± 0.000445985 (W 0.00042543) | 0.205184 ± 0.000669506 (W 0.00138221) | 🔵 colmat 1.55× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.15522 ± 0.0108464 (W 0.0018982) | 0.214825 ± 0.00312526 (W 0.00140137) | 🔵 colmat 1.38× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.0971275 ± 0.00155528 (W 0.000206077) | 0.140522 ± 2.38348 (W 0.000638668) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.067329 ± 0.00042692 (W 0.000352549) | 0.127302 ± 0.000865126 (W 0.0017928) | 🔵 colmat 1.89× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.137941 ± 0.0104122 (W 0.0166155) | 0.148196 ± 0.0014242 (W 0.0042545) | · | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.0496335 ± 0.000474843 (W 0.000140715) | 0.080938 ± 1.49452 (W 0.000540301) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-tall, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.355734 ± 0.00414811 (W 0.00198557) | 0.626901 ± 0.00297519 (W 0.0035004) | 🔵 colmat 1.76× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.50841 ± 0.00431345 (W 0.00278206) | 0.666122 ± 0.00294849 (W 0.00900744) | 🔵 colmat 1.31× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.409418 ± 0.00181038 (W 0.00242954) | 0.512683 ± 0.0013595 (W 0.00160977) | 🔵 colmat 1.25× faster than `lapack-openblas` |  |

## gemm-tall, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.094728 ± 0.000159909 (W 0.000204526) | 0.16968 ± 0.000532732 (W 0.00172271) | 🔵 colmat 1.79× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.145351 ± 0.0097599 (W 0.00292132) | 0.251247 ± 0.00385389 (W 0.00560955) | 🔵 colmat 1.73× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.105427 ± 0.00301106 (W 0.000374423) | 0.147406 ± 0.000402831 (W 0.00063022) | 🔵 colmat 1.40× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.0519985 ± 0.000276936 (W 0.000168576) | 0.100752 ± 0.000411984 (W 0.000265215) | 🔵 colmat 1.94× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.138342 ± 0.000214668 (W 0.00152568) | 0.194049 ± 0.00106667 (W 0.00102319) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.052158 ± 0.000164293 (W 0.000133794) | 0.0848825 ± 0.000724069 (W 0.000741125) | 🔵 colmat 1.63× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.50037 ± 0.0112045 (W 0.00248579) | 2.53112 ± 0.0146458 (W 0.00562076) | 🔵 colmat 1.69× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.60466 ± 0.0192536 (W 0.00523841) | 2.30146 ± 0.00663425 (W 0.00934989) | 🔵 colmat 1.43× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.48035 ± 0.00982833 (W 0.00193583) | 1.93742 ± 0.00186607 (W 0.00135114) | 🔵 colmat 1.31× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.386062 ± 0.00367978 (W 0.00183875) | 0.658277 ± 0.00515505 (W 0.00224043) | 🔵 colmat 1.71× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.407845 ± 0.00244841 (W 0.00317831) | 0.713586 ± 0.00118521 (W 0.00277265) | 🔵 colmat 1.75× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.373949 ± 0.00172193 (W 0.00222156) | 0.494457 ± 0.00113488 (W 0.0017719) | 🔵 colmat 1.32× faster than `lapack-openblas` |  |

## gemm-tall, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.215915 ± 0.00196174 (W 0.00199752) | 0.377223 ± 0.00336372 (W 0.00279504) | 🔵 colmat 1.75× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.31631 ± 0.0184974 (W 0.00598725) | 0.551281 ± 0.00217514 (W 0.00398663) | 🔵 colmat 1.74× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.187955 ± 0.000915705 (W 0.00121631) | 0.257794 ± 0.00124371 (W 0.00174358) | 🔵 colmat 1.37× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.34617 ± 0.000774572 (W 0.00177998) | 2.01846 ± 0.0111501 (W 0.00228419) | 🔵 colmat 1.50× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 1.57888 ± 0.0100423 (W 0.00348428) | 2.31907 ± 0.00370972 (W 0.0130875) | 🔵 colmat 1.47× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 1.50707 ± 0.00360327 (W 0.00155085) | 2.10813 ± 0.00209385 (W 0.00297641) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |

## gemm-tall, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.343029 ± 0.000843151 (W 0.00178928) | 0.546194 ± 0.0030744 (W 0.00216923) | 🔵 colmat 1.59× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.405645 ± 0.00300467 (W 0.00287414) | 0.82055 ± 0.00217257 (W 0.002946) | 🔵 colmat 2.02× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.379936 ± 0.00108018 (W 0.0018282) | 0.552578 ± 0.0143555 (W 0.00234442) | 🔵 colmat 1.45× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.187224 ± 0.000557501 (W 0.00169044) | 0.322591 ± 0.00201965 (W 0.00263578) | 🔵 colmat 1.72× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.29756 ± 0.0226295 (W 0.00180197) | 0.562258 ± 0.00263791 (W 0.0151695) | 🔵 colmat 1.89× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.194685 ± 0.00106392 (W 0.00125806) | 0.294988 ± 0.00185675 (W 0.00220103) | 🔵 colmat 1.52× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.226978 ± 0.00389242 (W 0.00144727) | 0.295546 ± 0.000194385 (W 0.00205771) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.197365 ± 0.00161039 (W 0.00136316) | 0.258369 ± 0.00780056 (W 0.00151693) | 🔵 colmat 1.31× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.184029 ± 0.00273983 (W 0.00139197) | 0.222383 ± 0.00460634 (W 0.00143442) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.063698 ± 0.000774416 (W 0.000348541) | 0.106483 ± 0.00094488 (W 0.0014516) | 🔵 colmat 1.67× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.049438 ± 0.000616713 (W 0.000176923) | 0.074638 ± 0.00137585 (W 0.000447363) | 🔵 colmat 1.51× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.046283 ± 0.000482246 (W 0.000106469) | 0.061958 ± 0.00025791 (W 0.000104322) | 🔵 colmat 1.34× faster than `lapack-openblas` |  |

## gemm-tall, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.0360995 ± 0.000192076 (W 0.000230994) | 0.0636685 ± 0.000696442 (W 0.000644392) | 🔵 colmat 1.76× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.026159 ± 0.000233824 (W 0.00015518) | 0.047624 ± 0.00039938 (W 0.000301403) | 🔵 colmat 1.82× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.023999 ± 0.000118592 (W 5.34368e-05) | 0.0366385 ± 0.000351661 (W 0.000305002) | 🔵 colmat 1.53× faster than `lapack-openblas` |  |

## gemm-tall, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.334885 ± 0.00163739 (W 0.00166971) | 0.274617 ± 0.00405573 (W 0.00163015) | 🔴 colmat 1.22× slower than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.215239 ± 0.000458978 (W 0.00148574) | 0.442672 ± 0.0224332 (W 0.002341) | 🔵 colmat 2.06× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.205048 ± 0.00115302 (W 0.00166388) | 0.293415 ± 0.0118356 (W 0.00149366) | 🔵 colmat 1.43× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.0842525 ± 0.000209886 (W 0.000120848) | 0.086567 ± 0.000648361 (W 0.000182964) | · |  |
| `gemm/4096x16x240/b1` | 0.0561235 ± 0.000518201 (W 0.000291178) | 0.123117 ± 0.00253754 (W 0.00173028) | 🔵 colmat 2.19× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.0515135 ± 0.000201161 (W 0.000184946) | 0.0882475 ± 0.00262566 (W 0.00116894) | 🔵 colmat 1.71× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.0448735 ± 0.000116039 (W 9.28215e-05) | 0.055569 ± 0.00013487 (W 0.000135464) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.0292895 ± 0.000819329 (W 0.000251293) | 0.0831885 ± 0.00176504 (W 0.00127049) | 🔵 colmat 2.84× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.0269195 ± 0.000403257 (W 9.38609e-05) | 0.0559385 ± 0.00157179 (W 0.00143071) | 🔵 colmat 2.08× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.760485 ± 0.00387568 (W 0.00188949) | 0.91684 ± 0.0127505 (W 0.00265369) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.798103 ± 0.0111873 (W 0.00192523) | 1.11596 ± 0.000990112 (W 0.00194434) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.764728 ± 0.00606664 (W 0.00153502) | 0.880341 ± 0.00235629 (W 0.000945213) | 🔵 colmat 1.15× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.206665 ± 0.00181436 (W 0.00116841) | 0.276307 ± 0.00150694 (W 0.00192445) | 🔵 colmat 1.34× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.205164 ± 0.00052869 (W 0.00134404) | 0.351446 ± 0.00793433 (W 0.0124509) | 🔵 colmat 1.71× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.19203 ± 0.00135193 (W 0.00103555) | 0.222694 ± 0.000575819 (W 0.00116339) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |

## gemm-tall, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.100799 ± 0.00037162 (W 0.000205697) | 0.162053 ± 0.00141434 (W 0.00242391) | 🔵 colmat 1.61× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.162702 ± 0.000804461 (W 0.00206908) | 0.254943 ± 0.000657673 (W 0.00172052) | 🔵 colmat 1.57× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.0970975 ± 0.000583379 (W 0.000127554) | 0.118541 ± 0.00175627 (W 0.00166502) | 🔵 colmat 1.22× faster than `lapack-openblas` |  |

## gemm-tall, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.793455 ± 0.000925578 (W 0.00192923) | 0.932656 ± 0.00199905 (W 0.00156599) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.79951 ± 0.00621674 (W 0.00207444) | 1.13837 ± 0.0119096 (W 0.00215469) | 🔵 colmat 1.42× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.753277 ± 0.00414682 (W 0.00170221) | 0.989594 ± 0.0260512 (W 0.00294301) | 🔵 colmat 1.31× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.200269 ± 0.000451839 (W 0.00153969) | 0.243512 ± 0.000381622 (W 0.00173024) | 🔵 colmat 1.22× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.20406 ± 0.00120997 (W 0.00208552) | 0.398841 ± 0.000945869 (W 0.00248602) | 🔵 colmat 1.95× faster than `lapack-openblas` |  |
| `gemm/4096x64x64/b1` | 0.192926 ± 0.0021934 (W 0.00128644) | 0.249577 ± 0.00496613 (W 0.00212981) | 🔵 colmat 1.29× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-tall, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.108056 ± 0.000271138 (W 0.000232787) | 0.152226 ± 0.000508295 (W 0.000892418) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `gemm/4096x16x240/b1` | 0.140887 ± 0.0110816 (W 0.00167846) | 0.265387 ± 0.00228251 (W 0.00161213) | 🔵 colmat 1.88× faster than `lapack-openblas` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.0948025 ± 0.000381143 (W 0.000159128) | 0.132656 ± 0.000867852 (W 0.00128393) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |

## gemm-wide, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.410439 ± 0.00239594 (W 0.0019289) | 0.514235 ± 0.0015305 (W 0.00166295) | 🔵 colmat 1.25× faster than `lapack-openblas` |  |

## gemm-wide, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.105362 ± 0.000956594 (W 0.000437895) | 0.147761 ± 4.83003 (W 0.000588172) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.052188 ± 0.000242975 (W 0.000135358) | 0.085679 ± 0.000697663 (W 0.000476566) | 🔵 colmat 1.64× faster than `lapack-openblas` |  |

## gemm-wide, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.381808 ± 0.000656973 (W 0.00174507) | 0.506624 ± 0.00228668 (W 0.00199499) | 🔵 colmat 1.33× faster than `lapack-openblas` |  |

## gemm-wide, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.097189 ± 0.000302046 (W 0.000284045) | 0.139861 ± 0.000781632 (W 0.00134656) | 🔵 colmat 1.44× faster than `lapack-openblas` |  |

## gemm-wide, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.0495985 ± 0.000271206 (W 0.000147086) | 0.0794885 ± 1.18187 (W 0.000228818) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.50688 ± 0.00332296 (W 0.0019089) | 2.1071 ± 0.00234254 (W 0.00221605) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |

## gemm-wide, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.379352 ± 0.000770914 (W 0.0018287) | 0.551708 ± 0.00136631 (W 0.00276017) | 🔵 colmat 1.45× faster than `lapack-openblas` |  |

## gemm-wide, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.19436 ± 0.00107866 (W 0.0016494) | 0.296121 ± 4.66629 (W 0.0020554) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.4865 ± 0.00318924 (W 0.00197272) | 1.93894 ± 0.00156647 (W 0.00158101) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.374664 ± 0.00097313 (W 0.00182778) | 0.492769 ± 0.00177547 (W 0.00217103) | 🔵 colmat 1.32× faster than `lapack-openblas` |  |

## gemm-wide, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.18755 ± 0.000689765 (W 0.00085238) | 0.258692 ± 0.00091615 (W 0.0015694) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |

## gemm-wide, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.205144 ± 0.00876732 (W 0.00132112) | 0.294891 ± 0.0107398 (W 0.00177031) | 🔵 colmat 1.44× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-wide, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.051414 ± 0.000197596 (W 0.000197389) | 0.0936485 ± 0.00346848 (W 0.00164745) | 🔵 colmat 1.82× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-wide, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.0273295 ± 0.000508531 (W 7.47439e-05) | 0.0559395 ± 0.00130971 (W 0.00105579) | 🔵 colmat 2.05× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-wide, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.18447 ± 0.00777469 (W 0.00150026) | 0.220003 ± 0.00255297 (W 0.00164295) | 🔵 colmat 1.19× faster than `lapack-openblas` | `!` between-start range over 5% |

## gemm-wide, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.0460545 ± 0.000159314 (W 0.000133133) | 0.062368 ± 0.00081195 (W 0.00116722) | 🔵 colmat 1.35× faster than `lapack-openblas` |  |

## gemm-wide, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.023864 ± 0.000210131 (W 6.63623e-05) | 0.0372385 ± 0.00035183 (W 0.00133526) | 🔵 colmat 1.56× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.759513 ± 0.00285532 (W 0.00178772) | 0.990436 ± 0.02276 (W 0.00210192) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.193497 ± 0.00231324 (W 0.00134351) | 0.248572 ± 0.00298252 (W 0.00154573) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |

## gemm-wide, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.095059 ± 0.000424895 (W 0.000132397) | 0.132072 ± 0.000737743 (W 0.000990087) | 🔵 colmat 1.39× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.766503 ± 0.00665259 (W 0.00131804) | 0.880124 ± 0.00246638 (W 0.00147207) | 🔵 colmat 1.15× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.192295 ± 0.000658512 (W 0.0010186) | 0.222928 ± 0.000688429 (W 0.00128729) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |

## gemm-wide, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.0969275 ± 0.000270317 (W 0.000109311) | 0.118596 ± 0.00069149 (W 0.0016625) | 🔵 colmat 1.22× faster than `lapack-openblas` |  |

## linalg-small, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00227 ± 9.87927e-05 (W 2.59592e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002285 ± 1.26491e-05 (W 3.39676e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0077795 ± 1.32728e-05 (W 3.70388e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.007685 ± 1.05909e-05 (W 5.46805e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0168845 ± 7.96007e-05 (W 9.26936e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0167995 ± 9.24202e-05 (W 8.19629e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.00422 ± 3.44964e-05 (W 2.63586e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0054895 ± 6.45314e-05 (W 8.14116e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01463 ± 0.00010861 (W 9.16244e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0181245 ± 0.000154183 (W 0.000151733) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.032789 ± 0.000241334 (W 0.000105118) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.045359 ± 0.000786817 (W 0.00145521) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00446 ± 1.41421e-05 (W 2.64224e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.006235 ± 1.26491e-05 (W 7.0707e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01367 ± 8.64579e-05 (W 4.70586e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0190595 ± 0.000175249 (W 4.37435e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.0288645 ± 0.000106102 (W 0.00126033) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0461735 ± 0.000685154 (W 0.00152114) | - | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0023 ± 2.80476e-05 (W 2.44947e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0023495 ± 1.51052e-05 (W 2.15486e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.007935 ± 0.000185608 (W 4.89889e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.007935 ± 2.31661e-05 (W 6.08238e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.018234 ± 7.00497e-05 (W 8.72449e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0185495 ± 8.9213e-05 (W 0.000104332) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.004395 ± 3.97513e-05 (W 3.25254e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.005645 ± 8.04156e-05 (W 8.40795e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0181095 ± 0.000299759 (W 0.000278182) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0195045 ± 0.00011279 (W 0.000196968) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.0412995 ± 0.000768642 (W 0.00054451) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.044869 ± 0.000576676 (W 0.00149992) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.004895 ± 1.78885e-05 (W 4.20917e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.006575 ± 2.7206e-05 (W 3.65615e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.017909 ± 0.000133777 (W 0.00025055) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.020915 ± 0.000150925 (W 0.000132457) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0323745 ± 0.000171048 (W 0.000429739) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0472885 ± 0.000508086 (W 0.00163456) | - | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0023 ± 1.75119e-05 (W 1.81134e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00229 ± 1.67332e-05 (W 3.21691e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.007995 ± 7.29383e-05 (W 5.48828e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00793 ± 3.99187e-05 (W 5.6409e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0182745 ± 0.000104704 (W 8.47783e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0185695 ± 6.66203e-05 (W 0.000126912) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.004385 ± 0.000145385 (W 3.317e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.005635 ± 6.5549e-05 (W 9.91674e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0202595 ± 0.000329422 (W 0.000485896) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.019525 ± 0.000150383 (W 0.000200396) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.0419685 ± 0.000524055 (W 0.000519214) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.045364 ± 0.000315611 (W 0.00148022) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00497 ± 6.44787e-05 (W 4.6115e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.006575 ± 3.61976e-05 (W 2.59173e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0173095 ± 8.21643e-05 (W 0.000334536) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.021114 ± 0.000165149 (W 0.000182468) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0358045 ± 0.000418243 (W 0.000524584) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.048614 ± 0.000243572 (W 0.00183947) | - | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001845 ± 1.94079e-05 (W 1.6937e-05) | 0.00187 ± 2.56255e-05 (W 3.31632e-05) | · | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00229 ± 1.63299e-05 (W 4.21307e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.006455 ± 4.47679e-05 (W 2.67588e-05) | 0.0072845 ± 8.00475e-05 (W 6.203e-05) | 🔵 colmat 1.13× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.007735 ± 2.75681e-05 (W 5.36987e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.014215 ± 0.00012973 (W 5.72427e-05) | 0.0168745 ± 0.000116917 (W 8.57488e-05) | 🔵 colmat 1.19× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0167395 ± 6.53934e-05 (W 6.68362e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.003395 ± 3.17096e-05 (W 5.45045e-05) | 0.00471 ± 2.56255e-05 (W 5.94301e-05) | 🔵 colmat 1.39× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.004385 ± 4.49017e-05 (W 4.90006e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0128245 ± 0.000111709 (W 0.000103033) | 0.0180745 ± 8.92203e-05 (W 0.000106468) | 🔵 colmat 1.41× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.01486 ± 9.48156e-05 (W 0.000117913) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.028574 ± 0.000128912 (W 0.000186285) | 0.040184 ± 6.87617e-05 (W 0.000179264) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.036704 ± 0.000501263 (W 0.000872279) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.003885 ± 2.71416e-05 (W 4.54193e-05) | 0.004785 ± 2.91233e-05 (W 4.35248e-05) | 🔵 colmat 1.23× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00502 ± 3.7238e-05 (W 3.09256e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.012954 ± 5.87049e-05 (W 3.8723e-05) | 0.0178595 ± 0.00020981 (W 5.91317e-05) | 🔵 colmat 1.38× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0151445 ± 6.04119e-05 (W 4.88898e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.0285295 ± 0.000260208 (W 8.64752e-05) | 0.0399085 ± 0.000212017 (W 0.000232427) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.0377545 ± 0.000876601 (W 0.00151083) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-small, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00182 ± 0.000128795 (W 2.29064e-05) | 0.0019095 ± 1.1726e-05 (W 3.71006e-05) | · | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002345 ± 0.000102111 (W 2.62256e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00664 ± 2.89183e-05 (W 2.75608e-05) | 0.007285 ± 8.22087e-05 (W 6.21391e-05) | 🔵 colmat 1.10× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.007955 ± 5.51377e-05 (W 5.70467e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.015469 ± 4.56965e-05 (W 7.65704e-05) | 0.01669 ± 8.64962e-05 (W 0.000112083) | 🔵 colmat 1.08× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0184445 ± 7.25821e-05 (W 0.000130546) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.00351 ± 3.75801e-05 (W 6.86933e-05) | 0.004715 ± 5.60952e-05 (W 7.14971e-05) | 🔵 colmat 1.34× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.004625 ± 4.95648e-05 (W 5.41577e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.013925 ± 0.000120046 (W 0.00019148) | 0.018184 ± 4.21379e-05 (W 8.19112e-05) | 🔵 colmat 1.31× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0159845 ± 7.42628e-05 (W 0.000250853) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.0258645 ± 0.000181759 (W 0.00016242) | 0.0404035 ± 0.000145784 (W 0.000144697) | 🔵 colmat 1.56× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.036209 ± 0.000168583 (W 0.0014591) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0042045 ± 3.34091e-05 (W 3.83057e-05) | 0.004835 ± 4.28661e-05 (W 3.11303e-05) | 🔵 colmat 1.15× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0053445 ± 2.48495e-05 (W 3.32077e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01494 ± 0.000140481 (W 0.000143712) | 0.017885 ± 0.000285396 (W 5.19599e-05) | 🔵 colmat 1.20× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0168945 ± 8.25682e-05 (W 8.85391e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.0278045 ± 0.000189941 (W 0.000165274) | 0.040209 ± 9.44466e-05 (W 0.000718746) | 🔵 colmat 1.45× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.038964 ± 0.000815949 (W 0.00162894) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-small, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00182 ± 0.000102502 (W 2.1208e-05) | 0.00188 ± 1.34722e-05 (W 2.7102e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00229 ± 1.87083e-05 (W 3.17356e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.006555 ± 2.93513e-05 (W 2.34482e-05) | 0.0073645 ± 9.56953e-05 (W 4.94144e-05) | 🔵 colmat 1.12× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0079795 ± 4.31157e-05 (W 7.21438e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0154795 ± 0.000451425 (W 9.4949e-05) | 0.01683 ± 4.4138e-05 (W 6.16938e-05) | 🔵 colmat 1.09× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0184895 ± 8.527e-05 (W 0.000124172) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.003505 ± 2.65832e-05 (W 5.92692e-05) | 0.004755 ± 2.07364e-05 (W 7.58552e-05) | 🔵 colmat 1.36× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00453 ± 4.67974e-05 (W 7.55591e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.013825 ± 0.000117542 (W 0.000183468) | 0.0181595 ± 0.000167969 (W 8.72628e-05) | 🔵 colmat 1.31× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0159395 ± 0.000108579 (W 0.000212833) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.0255695 ± 0.000131173 (W 0.000263192) | 0.040284 ± 0.000208686 (W 9.4214e-05) | 🔵 colmat 1.58× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0369535 ± 0.000139675 (W 0.000289916) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0044445 ± 2.32702e-05 (W 3.63311e-05) | 0.0048845 ± 5.2303e-05 (W 3.59626e-05) | 🔵 colmat 1.10× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.005355 ± 4.07022e-05 (W 4.17714e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0148895 ± 0.00012585 (W 0.000117175) | 0.017714 ± 0.000171469 (W 4.72649e-05) | 🔵 colmat 1.19× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.017214 ± 8.63723e-05 (W 0.000271072) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.027874 ± 0.000118279 (W 0.000216758) | 0.0402585 ± 0.000286228 (W 6.04605e-05) | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.040294 ± 0.000142652 (W 0.00163332) | - | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0037645 ± 4.23198e-05 (W 2.5184e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00372 ± 1.7224e-05 (W 3.63927e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.018374 ± 3.81073e-05 (W 7.97302e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.018585 ± 0.000129923 (W 8.67132e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0464835 ± 0.000202213 (W 0.00106997) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.047214 ± 0.000139403 (W 0.00012619) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0084945 ± 4.97755e-05 (W 4.50701e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0098545 ± 7.99517e-05 (W 7.2602e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0321095 ± 0.000167369 (W 8.40527e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.036779 ± 0.000190615 (W 0.000311064) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.075528 ± 0.00022944 (W 0.000133483) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.118317 ± 0.000815368 (W 0.00146222) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.008475 ± 5.03653e-05 (W 3.8027e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010674 ± 6.94188e-05 (W 9.42625e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.030064 ± 0.000112313 (W 8.26718e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.038124 ± 0.000245857 (W 0.000171867) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.072489 ± 0.000243936 (W 0.000637971) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.120848 ± 0.000600238 (W 0.0018993) | - | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0037895 ± 7.65517e-05 (W 2.74617e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003755 ± 2.07364e-05 (W 3.54626e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.017295 ± 0.000106086 (W 0.000116658) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0182995 ± 0.0001104 (W 0.000116725) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.041659 ± 0.000238952 (W 0.000270992) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.043194 ± 0.000272957 (W 0.000275355) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008585 ± 2.73252e-05 (W 5.99133e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010035 ± 0.000122573 (W 0.000115181) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.035199 ± 0.000269178 (W 0.000355933) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.036154 ± 0.000146302 (W 0.000271459) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0777075 ± 0.000721732 (W 0.000566298) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.107957 ± 0.00183964 (W 0.00100168) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.008915 ± 5.37417e-05 (W 3.56474e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0110795 ± 3.32906e-05 (W 6.64773e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0311545 ± 0.000173287 (W 0.000160654) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0373135 ± 0.000343555 (W 0.00030138) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0654875 ± 0.000128829 (W 0.000817386) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.107886 ± 0.0012725 (W 0.00165689) | - | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003735 ± 3.43026e-05 (W 2.88997e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003715 ± 1.16905e-05 (W 4.94971e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0176 ± 9.3017e-05 (W 9.54345e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0184845 ± 8.81151e-05 (W 0.000239423) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.041993 ± 0.000110121 (W 0.000143808) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.043803 ± 0.000258706 (W 0.000198706) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0085995 ± 5.66877e-05 (W 5.93891e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.01006 ± 9.34889e-05 (W 9.89307e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.036874 ± 0.000252778 (W 0.00034138) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.035849 ± 0.000189886 (W 0.000269477) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.083223 ± 0.000507283 (W 0.000536532) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.106842 ± 0.000728105 (W 0.000911781) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00889 ± 6.3289e-05 (W 3.72578e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0111345 ± 7.02778e-05 (W 8.65616e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0319795 ± 0.000225336 (W 0.000227304) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.037054 ± 9.74084e-05 (W 0.000243476) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0635885 ± 0.000345971 (W 0.000267276) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.107467 ± 0.000725802 (W 0.0015725) | - | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00311 ± 3.65605e-05 (W 1.32274e-05) | 0.00321 ± 4.0614e-05 (W 3.86012e-05) | · | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003725 ± 1.60208e-05 (W 4.17384e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.015905 ± 9.93192e-05 (W 4.77067e-05) | 0.0145 ± 5.16427e-05 (W 6.83174e-05) | 🔴 colmat 1.10× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0183995 ± 5.02941e-05 (W 6.1375e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0416095 ± 8.32971e-05 (W 5.4493e-05) | 0.038634 ± 8.96348e-05 (W 9.56172e-05) | 🔴 colmat 1.08× slower than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.046633 ± 0.000137952 (W 0.00050255) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00755 ± 3.54424e-05 (W 5.22964e-05) | 0.006814 ± 8.80697e-05 (W 6.698e-05) | 🔴 colmat 1.11× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00889 ± 9.22766e-05 (W 9.13263e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0288095 ± 0.00010341 (W 8.91681e-05) | 0.0298895 ± 9.26621e-05 (W 5.06117e-05) | · |  |
| `lu/n64/b1 [api-inclusive]` | 0.032469 ± 0.000129538 (W 0.00011397) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0707975 ± 0.00035018 (W 0.000139994) | 0.076958 ± 0.000155962 (W 0.0002024) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.109322 ± 0.00117475 (W 0.000901333) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00804 ± 3.18852e-05 (W 2.73288e-05) | 0.00757 ± 3.77558e-05 (W 3.25935e-05) | 🔴 colmat 1.06× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00943 ± 3.84396e-05 (W 4.11199e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.030384 ± 0.000308565 (W 7.03749e-05) | 0.0321495 ± 0.000110942 (W 9.49317e-05) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.033984 ± 6.44981e-05 (W 7.77012e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.074758 ± 0.000556242 (W 0.000904036) | 0.081758 ± 0.000101244 (W 0.000621274) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.111731 ± 0.00130811 (W 0.00193147) | - | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003105 ± 2.06559e-05 (W 1.15612e-05) | 0.003215 ± 2.70093e-05 (W 4.46489e-05) | · | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003745 ± 1.60208e-05 (W 3.52091e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0156745 ± 0.00013577 (W 7.91431e-05) | 0.0160045 ± 0.000266259 (W 0.000236685) | · | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0181545 ± 9.88887e-05 (W 0.00013207) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0381195 ± 0.000249349 (W 0.000383206) | 0.0448695 ± 0.000558498 (W 0.000589271) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.042974 ± 0.000131319 (W 0.000227163) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.007615 ± 4.53891e-05 (W 7.69673e-05) | 0.006815 ± 5.0519e-05 (W 7.38347e-05) | 🔴 colmat 1.12× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008975 ± 6.08824e-05 (W 7.808e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.028299 ± 0.000174901 (W 0.000292037) | 0.029869 ± 0.000109072 (W 0.000117532) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.032789 ± 0.000208461 (W 0.000379946) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.057418 ± 0.000338081 (W 0.000431881) | 0.0769325 ± 0.000139652 (W 0.00016932) | 🔵 colmat 1.34× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0983325 ± 0.000483892 (W 0.00108982) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0085 ± 5.88685e-05 (W 6.58697e-05) | 0.0076495 ± 8.42868e-05 (W 3.79302e-05) | 🔴 colmat 1.11× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0098345 ± 5.33504e-05 (W 4.80831e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.029789 ± 0.000209444 (W 0.000264275) | 0.032189 ± 7.36632e-05 (W 8.45973e-05) | 🔵 colmat 1.08× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.033089 ± 0.000213323 (W 0.000162138) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.058573 ± 0.000267703 (W 0.000406986) | 0.081678 ± 0.000207125 (W 0.00135754) | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.0973525 ± 0.000546371 (W 0.00157556) | - | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003045 ± 3.39116e-05 (W 1.73297e-05) | 0.00327 ± 5.06376e-05 (W 3.48821e-05) | 🔵 colmat 1.07× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00376 ± 8.16497e-06 (W 6.84433e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0156445 ± 0.000141658 (W 9.03157e-05) | 0.023004 ± 8.56299 (W 0.000689323) | · | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.018244 ± 0.000163985 (W 0.000180424) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0382695 ± 0.000275415 (W 0.000239359) | 0.045399 ± 0.000200088 (W 0.000703186) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.043429 ± 0.000196002 (W 0.000165056) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.007635 ± 0.000104633 (W 8.39298e-05) | 0.006835 ± 5.04645e-05 (W 4.76901e-05) | 🔴 colmat 1.12× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0089845 ± 0.000161778 (W 8.59392e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.028199 ± 0.000147133 (W 0.000276074) | 0.029764 ± 0.00012093 (W 8.58826e-05) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.032389 ± 0.000104965 (W 0.000280257) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0555895 ± 0.000275885 (W 0.000408166) | 0.076849 ± 0.000144005 (W 0.000137536) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.097192 ± 0.00114437 (W 0.000823909) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00845 ± 2.98021e-05 (W 5.75735e-05) | 0.0076195 ± 5.76911e-05 (W 8.09354e-05) | 🔴 colmat 1.11× slower than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009865 ± 4.65349e-05 (W 8.51599e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.029359 ± 0.000172134 (W 0.000184253) | 0.032449 ± 7.28297e-05 (W 7.71972e-05) | 🔵 colmat 1.11× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.0330345 ± 0.000153213 (W 0.000266287) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0537735 ± 0.000225357 (W 0.000141008) | 0.0814325 ± 0.000175334 (W 0.000121185) | 🔵 colmat 1.51× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.097142 ± 0.00059252 (W 0.00147979) | - | - | `*` one row has no measurement here |

## linalg-small, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001725 ± 1.63299e-05 (W 1.59363e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00168 ± 1.0328e-05 (W 3.93826e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0073545 ± 0.00065018 (W 1.89218e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.006575 ± 2.44165e-05 (W 4.69091e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0144345 ± 8.46579e-05 (W 6.09214e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0130045 ± 5.27851e-05 (W 4.9279e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.004245 ± 5.0892e-05 (W 3.16419e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00553 ± 6.1418e-05 (W 6.12807e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01327 ± 0.000114866 (W 8.85478e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0172945 ± 0.000111446 (W 9.78926e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.027994 ± 0.000145232 (W 0.000130501) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0367745 ± 0.000257203 (W 0.000195392) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.004715 ± 2.73252e-05 (W 3.04115e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00636 ± 6.40833e-05 (W 4.1171e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0128645 ± 7.65604e-05 (W 9.34318e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.018204 ± 7.49684e-05 (W 8.0735e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.0252945 ± 0.000118857 (W 0.000110575) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.036179 ± 0.000142837 (W 0.000123116) | - | - | `*` one row has no measurement here |

## linalg-small, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0017 ± 8.4951e-05 (W 2.5786e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.001745 ± 1.94079e-05 (W 2.26805e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00737 ± 0.000476084 (W 3.24075e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00661 ± 1.70958e-05 (W 5.35972e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0148445 ± 6.73224e-05 (W 5.7314e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01333 ± 6.92746e-05 (W 9.40014e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.004295 ± 3.56371e-05 (W 2.60514e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00561 ± 5.04645e-05 (W 7.31904e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.013314 ± 7.61945e-05 (W 0.000108233) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.017315 ± 0.000144791 (W 0.000121242) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.036234 ± 0.000377765 (W 0.000861305) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0366195 ± 0.000178724 (W 0.000248199) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.005045 ± 3.60578e-05 (W 3.02138e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00674 ± 3.84708e-05 (W 4.97556e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0134595 ± 7.13321e-05 (W 6.35863e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.018849 ± 8.01434e-05 (W 8.76586e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.0303895 ± 0.000581693 (W 0.000611412) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.037944 ± 0.000243913 (W 0.000142711) | - | - | `*` one row has no measurement here |

## linalg-small, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00173 ± 7.12039e-05 (W 2.61709e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00169 ± 1.60052e-05 (W 3.09599e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0073445 ± 0.000518039 (W 2.70171e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00667 ± 3.40808e-05 (W 4.625e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.01478 ± 0.000686248 (W 4.71033e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0133495 ± 3.19625e-05 (W 8.26586e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.0043 ± 5.92453e-05 (W 3.8909e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0056445 ± 1.58798e-05 (W 5.93923e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.013385 ± 4.61754e-05 (W 6.41057e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0173445 ± 8.41992e-05 (W 7.86552e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.040459 ± 0.000242529 (W 0.00143824) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.036509 ± 0.000179889 (W 0.00034465) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00507 ± 4.87511e-05 (W 3.85061e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.006735 ± 2.38642e-05 (W 4.60094e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.013489 ± 5.5955e-05 (W 6.00144e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0188995 ± 0.00017845 (W 0.000103804) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.034509 ± 0.000246468 (W 0.00131187) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.038529 ± 0.000282182 (W 0.000305536) | - | - | `*` one row has no measurement here |

## linalg-small, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0012595 ± 1.09621e-05 (W 1.82779e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.001685 ± 1.75119e-05 (W 4.14543e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00552 ± 5.94433e-05 (W 1.62106e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0067095 ± 1.1669e-05 (W 4.2708e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.010864 ± 5.54271e-05 (W 4.48763e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0130995 ± 2.68005e-05 (W 4.16819e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.00331 ± 3.25064e-05 (W 2.38747e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.004205 ± 5.92171e-05 (W 6.99799e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0115 ± 6.99762e-05 (W 0.000132293) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0131245 ± 0.000103799 (W 0.000119317) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.0238595 ± 0.000139019 (W 0.000116665) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.026279 ± 0.000134765 (W 0.000185985) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.003995 ± 3.7238e-05 (W 2.45636e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.004985 ± 4.58984e-05 (W 5.23495e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.012065 ± 0.000163805 (W 4.74891e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0136545 ± 6.78245e-05 (W 7.4769e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.024539 ± 0.000203018 (W 0.000165712) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.026479 ± 0.000153415 (W 9.38254e-05) | - | - | `*` one row has no measurement here |

## linalg-small, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00126 ± 3.28634e-05 (W 1.49834e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.001725 ± 2.41143e-05 (W 4.14614e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0055445 ± 0.000489142 (W 1.28674e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00672 ± 1.98368e-05 (W 4.96738e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0110595 ± 0.000515756 (W 3.02792e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01347 ± 3.24803e-05 (W 8.99832e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.0033245 ± 2.50253e-05 (W 3.51357e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0042995 ± 4.12007e-05 (W 3.57146e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0115095 ± 0.000103249 (W 0.000113388) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.01321 ± 8.66156e-05 (W 0.000153587) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.023544 ± 0.000164163 (W 0.000271899) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0264645 ± 0.000207494 (W 0.000338563) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.004255 ± 5.78504e-05 (W 3.77624e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00535 ± 5.42725e-05 (W 4.11149e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.012625 ± 0.000101569 (W 7.23733e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0143345 ± 5.39036e-05 (W 6.72961e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.0258845 ± 0.000224791 (W 0.000185025) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0281045 ± 0.000114203 (W 0.000234862) | - | - | `*` one row has no measurement here |

## linalg-small, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001245 ± 0.000100333 (W 1.1229e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00169 ± 1.3784e-05 (W 2.76514e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.005535 ± 4.63321e-05 (W 1.48914e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00677 ± 1.90832e-05 (W 5.12593e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0110495 ± 0.000734313 (W 2.96612e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.0134995 ± 5.47016e-05 (W 7.91159e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.0033495 ± 1.72095e-05 (W 2.89768e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.004295 ± 1.76711e-05 (W 6.31883e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.011505 ± 9.34701e-05 (W 9.16257e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0131595 ± 7.76677e-05 (W 0.000172236) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.0237095 ± 0.000184809 (W 0.00021321) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.026849 ± 0.000176839 (W 0.000337289) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00425 ± 2.50998e-05 (W 3.63449e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.005325 ± 3.67174e-05 (W 4.89365e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.013095 ± 6.29934e-05 (W 7.77314e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.01434 ± 3.88107e-05 (W 7.25234e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.025399 ± 0.000230303 (W 0.000226849) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.028424 ± 0.000102047 (W 0.000323336) | - | - | `*` one row has no measurement here |

## linalg-small, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.008025 ± 2.73252e-05 (W 2.4263e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00803 ± 2.37802e-05 (W 6.10085e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.044969 ± 7.77014e-05 (W 0.000146442) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0443095 ± 9.61258e-05 (W 0.000152028) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.069244 ± 6.61284e-05 (W 0.000158318) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0674535 ± 0.000113771 (W 0.000135914) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006595 ± 7.68913e-05 (W 4.21782e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00795 ± 5.31664e-05 (W 7.12786e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02316 ± 8.92205e-05 (W 0.000506842) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.028034 ± 0.000155874 (W 0.000234814) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0522435 ± 0.000204252 (W 0.000142034) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.068143 ± 0.000999752 (W 0.00127991) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0070745 ± 5.15196e-05 (W 2.77447e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0088945 ± 7.78728e-05 (W 2.7582e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.022459 ± 0.000155255 (W 7.1779e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.028959 ± 0.000186584 (W 0.000206034) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0504385 ± 0.000242983 (W 6.08318e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0692425 ± 0.000924146 (W 0.00163511) | - | - | `*` one row has no measurement here |

## linalg-small, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.008035 ± 1.89279e-05 (W 3.46384e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.008075 ± 3.50238e-05 (W 6.82791e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.044979 ± 7.58545e-05 (W 0.000148401) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044299 ± 5.31451e-05 (W 0.0001391) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.068508 ± 0.000136794 (W 0.000139968) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0677025 ± 5.50188e-05 (W 0.000153621) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00663 ± 4.80278e-05 (W 4.91895e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008145 ± 5.49327e-05 (W 6.2467e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0263295 ± 0.000364083 (W 0.000515373) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0274045 ± 0.000169834 (W 0.000191701) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.059473 ± 0.000346248 (W 0.00117839) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0637535 ± 0.000226053 (W 0.00148874) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007445 ± 4.57839e-05 (W 3.2417e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00929 ± 4.32049e-05 (W 5.10454e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0245995 ± 0.000289178 (W 0.000445862) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.029139 ± 0.000258585 (W 0.000261121) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.048869 ± 0.000376983 (W 0.000489016) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.065293 ± 0.000213037 (W 0.00159336) | - | - | `*` one row has no measurement here |

## linalg-small, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00804 ± 1.81154e-05 (W 3.73156e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00803 ± 1.07827e-05 (W 5.50491e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.044954 ± 0.000103923 (W 0.000137277) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044268 ± 6.37181e-05 (W 0.000159298) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.068368 ± 0.000154039 (W 0.000156558) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067593 ± 8.19034e-05 (W 0.000182229) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006645 ± 5.48294e-05 (W 4.94944e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00807 ± 3.31688e-05 (W 8.09615e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0271095 ± 0.000294592 (W 0.000443055) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0274145 ± 0.000192998 (W 0.000187341) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0593085 ± 0.000608533 (W 0.00136852) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.063749 ± 0.000186275 (W 0.00148566) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007485 ± 4.64345e-05 (W 4.4155e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0093645 ± 6.17274e-05 (W 5.47663e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.026399 ± 9.73045e-05 (W 0.000419026) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.028999 ± 0.000140177 (W 0.000189646) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.053464 ± 0.000166771 (W 0.000438989) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.066294 ± 0.000298698 (W 0.00162007) | - | - | `*` one row has no measurement here |

## linalg-small, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0075995 ± 1.19276e-05 (W 3.30352e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.008005 ± 1.34226e-05 (W 5.83112e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.043359 ± 8.9471e-05 (W 0.00010822) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044184 ± 5.9331e-05 (W 0.000128867) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0651175 ± 8.49292e-05 (W 0.000126603) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067254 ± 6.36742e-05 (W 0.000133159) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005584 ± 2.39228e-05 (W 3.47758e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00662 ± 4.69042e-05 (W 6.6132e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020805 ± 0.000110092 (W 0.000116685) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.022869 ± 0.000129121 (W 0.000148817) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0472835 ± 0.000144256 (W 0.00014968) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.055949 ± 0.000772258 (W 0.00133341) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0061545 ± 4.98013e-05 (W 3.39287e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007335 ± 2.35832e-05 (W 2.87318e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0214695 ± 0.000131538 (W 5.94777e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0237695 ± 6.05885e-05 (W 6.93276e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.048269 ± 6.37809e-05 (W 0.000103614) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.058049 ± 0.00085385 (W 0.00176797) | - | - | `*` one row has no measurement here |

## linalg-small, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.007595 ± 2.1125e-05 (W 3.40496e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.008065 ± 2.88097e-05 (W 6.22026e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.043244 ± 6.48968e-05 (W 0.000123988) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044234 ± 3.5386e-05 (W 0.000136986) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0646435 ± 9.77221e-05 (W 0.000103896) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0674125 ± 0.000149349 (W 0.000154402) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00564 ± 5.1693e-05 (W 4.00781e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006765 ± 2.09801e-05 (W 5.9381e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020255 ± 0.000197025 (W 0.000190248) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0223495 ± 9.61195e-05 (W 0.000141635) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0420845 ± 0.000366847 (W 0.00020654) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.051959 ± 0.000266789 (W 0.00100566) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0065695 ± 3.18161e-05 (W 3.35472e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00777 ± 5.60952e-05 (W 4.70941e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0215345 ± 0.000214682 (W 0.000153884) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.023719 ± 0.00017214 (W 0.000150578) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0429895 ± 0.000266422 (W 0.000195049) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.054029 ± 0.000273562 (W 0.0015456) | - | - | `*` one row has no measurement here |

## linalg-small, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00757 ± 6.32456e-06 (W 3.68373e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.008005 ± 3.56127e-05 (W 0.000111917) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0430485 ± 5.35164e-05 (W 0.000116314) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044244 ± 8.69193e-05 (W 0.000125301) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.064458 ± 5.17703e-05 (W 0.000131569) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067293 ± 0.000128312 (W 0.000178552) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005625 ± 4.19225e-05 (W 4.95363e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006645 ± 3.06621e-05 (W 5.77946e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020029 ± 0.000171737 (W 0.000124647) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.021985 ± 0.000159211 (W 0.000131057) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0418045 ± 0.000235893 (W 0.000313424) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0526845 ± 0.000186088 (W 0.00148541) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00675 ± 2.81069e-05 (W 2.99963e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0077895 ± 2.52639e-05 (W 6.94071e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0216295 ± 0.000107395 (W 0.000157225) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0237345 ± 0.000124232 (W 0.000165796) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0429235 ± 0.000189358 (W 0.000259712) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.055323 ± 0.00040897 (W 0.00177907) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `lu/32`, `lu/64`, `lu/96`, `solve/32`, `solve/64`, `solve/96`.

## linalg-mid, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.030514 ± 0.000314261 (W 0.000103011) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.034094 ± 0.000206769 (W 0.000252283) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.185911 ± 0.000117424 (W 0.00144597) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.170084 ± 0.00076407 (W 0.00135482) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.5062 ± 0.00444153 (W 0.00374866) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.02097 ± 0.0151125 (W 0.00103763) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.059083 ± 0.000364795 (W 0.000228593) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.071028 ± 0.000730574 (W 0.000379081) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.283237 ± 0.00126976 (W 0.00141854) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.525321 ± 0.00407138 (W 0.00251855) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.84793 ± 0.0134743 (W 0.00312629) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.92568 ± 0.0152879 (W 0.00350657) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.053654 ± 0.000334999 (W 0.000351321) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.071798 ± 0.000140151 (W 0.000646931) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.262358 ± 0.00180722 (W 0.00184804) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.359085 ± 0.00377762 (W 0.0021651) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.83771 ± 0.00689022 (W 0.00340332) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.9961 ± 0.00616335 (W 0.00825038) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.028704 ± 0.000212101 (W 0.000192398) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.034439 ± 0.000263957 (W 0.00027246) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.118111 ± 0.00038115 (W 0.00082878) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.128102 ± 0.000596151 (W 0.000592456) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.606375 ± 0.00415665 (W 0.00240169) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.53587 ± 0.0048941 (W 0.0026951) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.074743 ± 0.000801854 (W 0.000694912) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.067003 ± 0.00117102 (W 0.000359927) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.233016 ± 0.00108155 (W 0.00145548) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.422603 ± 0.00465311 (W 0.00158978) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.926449 ± 0.00733344 (W 0.0041941) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.98195 ± 0.0232398 (W 0.00530361) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.057424 ± 0.000325521 (W 0.000523094) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.068304 ± 0.00016085 (W 0.000319174) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.211911 ± 0.00149003 (W 0.00174038) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.431642 ± 0.00545205 (W 0.00270022) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.860747 ± 0.00556857 (W 0.00203216) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.91851 ± 0.0243424 (W 0.00510041) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0286645 ± 0.000126196 (W 0.000167744) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.035029 ± 0.000155435 (W 0.000313612) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.0972925 ± 0.000509368 (W 0.00132284) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.126507 ± 0.00177312 (W 0.000579046) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.470081 ± 0.00239989 (W 0.00206693) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.473775 ± 0.00401332 (W 0.00255386) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0791435 ± 0.000564385 (W 0.000963869) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.095357 ± 0.00106832 (W 0.000471607) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.194545 ± 0.00869087 (W 0.0018797) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.413162 ± 0.00254524 (W 0.00251843) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.684141 ± 0.002042 (W 0.00298109) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.82799 ± 0.0382535 (W 0.0078704) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.053184 ± 0.000265931 (W 0.000369545) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0693385 ± 0.000204285 (W 0.000337564) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.175505 ± 0.00119356 (W 0.00165678) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.418447 ± 0.00108113 (W 0.00320391) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.65155 ± 0.00278779 (W 0.00296542) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.80015 ± 0.00960014 (W 0.00577359) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.026559 ± 0.000113813 (W 6.34259e-05) | 0.0314145 ± 0.000122709 (W 0.000107854) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.0307995 ± 0.000110472 (W 7.42507e-05) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.137047 ± 0.000552246 (W 0.00066229) | 0.16921 ± 0.000543028 (W 0.00168407) | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.150842 ± 0.000622008 (W 0.000847475) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.928589 ± 0.00547012 (W 0.00244404) | 1.12028 ± 0.00583385 (W 0.00187616) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.953844 ± 0.0105526 (W 0.00122425) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.054214 ± 0.000382811 (W 0.00110409) | 0.0782235 ± 0.000225381 (W 0.000699377) | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.0586535 ± 0.000127175 (W 0.00024037) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.283896 ± 0.000717924 (W 0.00148572) | 0.390768 ± 0.000534578 (W 0.00202) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.468997 ± 0.00250113 (W 0.00206821) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.68078 ± 0.00667284 (W 0.00219882) | 2.25443 ± 0.00400413 (W 0.00186995) | 🔵 colmat 1.34× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.55469 ± 0.0164851 (W 0.00393479) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0530135 ± 0.00013845 (W 0.00010742) | 0.0763125 ± 0.000607271 (W 0.000559775) | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.057308 ± 0.00022403 (W 0.000114689) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.278231 ± 0.000526332 (W 0.00145822) | 0.380298 ± 0.00197938 (W 0.00189595) | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.302037 ± 0.00188914 (W 0.00198739) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.6596 ± 0.0174897 (W 0.00223743) | 2.19988 ± 0.00219188 (W 0.00335293) | 🔵 colmat 1.33× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 1.7408 ± 0.00160556 (W 0.00239327) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0256945 ± 0.000139782 (W 0.000165113) | 0.041874 ± 0.000434668 (W 0.000416428) | 🔵 colmat 1.63× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.031169 ± 0.000231767 (W 0.000179149) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.0871035 ± 0.000544031 (W 0.000657394) | 0.14074 ± 0.0012285 (W 0.00152084) | 🔵 colmat 1.62× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.108968 ± 0.00055788 (W 0.000369414) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.374999 ± 0.00217923 (W 0.00207342) | 0.618273 ± 0.0127411 (W 0.00370446) | 🔵 colmat 1.65× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.464125 ± 0.00300221 (W 0.00183142) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.046379 ± 0.000411006 (W 0.0005101) | 0.078188 ± 0.000553997 (W 0.000256988) | 🔵 colmat 1.69× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.053789 ± 0.000451202 (W 0.000278694) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.182289 ± 0.000884941 (W 0.00179585) | 0.240893 ± 0.000551817 (W 0.00195381) | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.368788 ± 0.00387443 (W 0.00207042) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.79395 ± 0.00235313 (W 0.00267621) | 1.15748 ± 0.00476198 (W 0.0032135) | 🔵 colmat 1.46× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 1.6637 ± 0.00604099 (W 0.00377429) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.047739 ± 0.000353996 (W 0.000347751) | 0.063193 ± 0.000340304 (W 0.00050118) | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.0547085 ± 0.000227615 (W 0.000437175) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.175677 ± 0.00172307 (W 0.00144716) | 0.229544 ± 0.00122914 (W 0.00198645) | 🔵 colmat 1.31× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.369001 ± 0.00172741 (W 0.00240918) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.736992 ± 0.00191141 (W 0.00250522) | 1.11122 ± 0.00956964 (W 0.00339701) | 🔵 colmat 1.51× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 1.62757 ± 0.0113405 (W 0.00404574) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.025704 ± 0.000354255 (W 0.000116201) | 0.041289 ± 0.000324481 (W 0.000903315) | 🔵 colmat 1.61× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.031404 ± 0.000223189 (W 0.000217856) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.0804325 ± 0.000435546 (W 0.000630175) | 0.136062 ± 0.00192742 (W 0.00214368) | 🔵 colmat 1.69× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.106367 ± 0.00048945 (W 0.000495761) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.28931 ± 0.00250363 (W 0.0020045) | 0.589509 ± 0.0154974 (W 0.0046939) | 🔵 colmat 2.04× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.392947 ± 0.00160977 (W 0.00195863) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0461195 ± 0.000471118 (W 0.000420923) | 0.0782085 ± 0.000670895 (W 0.000694195) | 🔵 colmat 1.70× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.082207 ± 0.000528156 (W 0.000446005) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.16292 ± 0.00142198 (W 0.00100852) | 0.218964 ± 0.000716628 (W 0.00208927) | 🔵 colmat 1.34× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.355596 ± 0.000674121 (W 0.00226312) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.634149 ± 0.00266803 (W 0.00304365) | 0.837339 ± 0.00621834 (W 0.00513175) | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 1.5106 ± 0.00985875 (W 0.0039561) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0448335 ± 0.000192599 (W 0.000244489) | 0.090757 ± 0.00107153 (W 0.000735939) | 🔵 colmat 2.02× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.055079 ± 0.000173114 (W 0.00036039) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.145566 ± 0.000694474 (W 0.00123909) | 0.235129 ± 0.00178491 (W 0.00239171) | 🔵 colmat 1.62× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.359662 ± 0.00164941 (W 0.00225823) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.542627 ± 0.00260304 (W 0.00212632) | 0.83484 ± 0.0046567 (W 0.0053992) | 🔵 colmat 1.54× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 1.5031 ± 0.00655217 (W 0.0036016) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0910625 ± 0.000328054 (W 0.00122411) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.0977425 ± 0.000586905 (W 0.000165427) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.52523 ± 0.00257364 (W 0.00193242) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.537781 ± 0.00176508 (W 0.00162049) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 3.13314 ± 0.0112587 (W 0.00288268) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 3.17356 ± 0.00462326 (W 0.00237483) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.143545 ± 0.00152125 (W 0.00137943) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.241019 ± 0.00349325 (W 0.00161959) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.847202 ± 0.0029971 (W 0.00230216) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.34922 ± 0.00851529 (W 0.00297938) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.406 ± 0.00977379 (W 0.00975017) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.44633 ± 0.0189816 (W 0.0124951) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.137771 ± 0.000347661 (W 0.000211445) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.246881 ± 0.00139607 (W 0.00190905) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.84556 ± 0.00816338 (W 0.00156513) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.967967 ± 0.0011806 (W 0.00207741) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.40769 ± 0.0690825 (W 0.00507966) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 5.79755 ± 0.0105001 (W 0.00871512) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.071473 ± 0.000335434 (W 0.000329113) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.082128 ± 0.000258573 (W 0.000245564) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.266656 ± 0.000891728 (W 0.00151665) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.322634 ± 0.00154975 (W 0.00153311) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.04575 ± 0.00500689 (W 0.00200192) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.24507 ± 0.00321126 (W 0.00265322) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.128262 ± 0.000466214 (W 0.00079615) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.20807 ± 0.00151132 (W 0.00141669) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.468446 ± 0.00341553 (W 0.00292547) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.957915 ± 0.0194799 (W 0.00268157) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 2.126 ± 0.0334494 (W 0.00387792) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.19234 ± 0.0051154 (W 0.00788138) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.109247 ± 0.000553735 (W 0.000445074) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.207713 ± 0.000659629 (W 0.00194389) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.454813 ± 0.00169924 (W 0.00194052) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.952579 ± 0.00383606 (W 0.00293881) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.05369 ± 0.00359609 (W 0.00487572) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.15318 ± 0.00735067 (W 0.0119775) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.072088 ± 0.000486512 (W 0.000236307) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.0825085 ± 0.00022684 (W 0.000244154) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.232233 ± 0.00086233 (W 0.00171049) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.294299 ± 0.000908889 (W 0.00192103) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.764307 ± 0.00200939 (W 0.00178929) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.00739 ± 0.00862674 (W 0.00163726) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.130756 ± 0.000911377 (W 0.000862215) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.199325 ± 0.00103463 (W 0.00196757) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.395528 ± 0.010094 (W 0.00215313) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.907663 ± 0.00158873 (W 0.00204627) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.48901 ± 0.00386904 (W 0.00359088) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.66128 ± 0.0119877 (W 0.0108037) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.096068 ± 0.000407921 (W 0.00068343) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.2011 ± 0.00079356 (W 0.00164821) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.383266 ± 0.0025125 (W 0.00286376) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.514636 ± 0.000717137 (W 0.0021089) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.46128 ± 0.00788682 (W 0.00472495) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.67074 ± 0.0112889 (W 0.00615154) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.082783 ± 0.000170765 (W 8.61869e-05) | 0.081418 ± 0.000286236 (W 0.00013736) | · |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.090548 ± 0.00049424 (W 0.000246404) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.481851 ± 0.000987365 (W 0.00180282) | 0.548768 ± 0.00298911 (W 0.00179766) | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.51171 ± 0.00133433 (W 0.0019089) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.97778 ± 0.00821005 (W 0.00172917) | 3.73204 ± 0.00990906 (W 0.00410783) | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 3.06406 ± 0.00906943 (W 0.00345923) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.139721 ± 0.000554785 (W 0.00141664) | 0.158315 ± 0.000488779 (W 0.00103402) | 🔵 colmat 1.13× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.222909 ± 0.00157408 (W 0.00240253) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.820697 ± 0.00545494 (W 0.00279492) | 0.992416 ± 0.00194175 (W 0.000829159) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.2495 ± 0.00407332 (W 0.00267071) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.17608 ± 0.00760091 (W 0.00264059) | 6.43702 ± 0.0118001 (W 0.0045402) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 6.94985 ± 0.0155158 (W 0.00520184) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.147811 ± 0.00103414 (W 0.00148073) | 0.169515 ± 0.000215337 (W 0.00167938) | 🔵 colmat 1.15× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.228748 ± 0.000953264 (W 0.00189997) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.83539 ± 0.00320693 (W 0.00211385) | 1.07875 ± 0.0105965 (W 0.00123321) | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.886634 ± 0.00106657 (W 0.00527064) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.28661 ± 0.00785557 (W 0.0037057) | 6.66771 ± 0.00840223 (W 0.00386782) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 5.47154 ± 0.00817434 (W 0.00574932) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0657535 ± 0.000224689 (W 0.00014978) | 0.066138 ± 0.000292664 (W 0.000649034) | · |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.0749425 ± 0.000438032 (W 0.000207049) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.24151 ± 0.00130703 (W 0.00153564) | 0.288636 ± 0.00128264 (W 0.00184438) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.293146 ± 0.00116218 (W 0.0015331) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.984946 ± 0.0156171 (W 0.00187457) | 1.45178 ± 0.00203104 (W 0.00415708) | 🔵 colmat 1.47× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.12524 ± 0.00179914 (W 0.00172406) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.104782 ± 0.00035132 (W 0.000699624) | 0.117778 ± 0.000922314 (W 0.00109689) | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.191699 ± 0.00106605 (W 0.00178021) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.421003 ± 0.00153744 (W 0.00151415) | 0.568793 ± 0.00285533 (W 0.00197696) | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.852785 ± 0.00583643 (W 0.0024662) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.95512 ± 0.00533826 (W 0.00365378) | 2.24031 ± 0.00437836 (W 0.00572608) | 🔵 colmat 1.15× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 3.75362 ± 0.0253665 (W 0.00644959) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.104992 ± 0.000274629 (W 0.000513316) | 0.115401 ± 0.000610747 (W 0.00117607) | 🔵 colmat 1.10× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.189474 ± 0.00078787 (W 0.00150094) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.40951 ± 0.000862627 (W 0.00228731) | 0.553954 ± 0.00268017 (W 0.00221844) | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.844596 ± 0.00314559 (W 0.00271943) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.89431 ± 0.00499792 (W 0.00593363) | 2.20444 ± 0.00568459 (W 0.00658589) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 3.70571 ± 0.0101196 (W 0.00424563) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.065563 ± 0.000210142 (W 0.000200417) | 0.0766185 ± 0.000561758 (W 0.00120125) | 🔵 colmat 1.17× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.0757575 ± 0.000341091 (W 0.000146728) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.209795 ± 0.000616962 (W 0.00155604) | 0.259687 ± 0.00216623 (W 0.00214143) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.268445 ± 0.00125171 (W 0.00152055) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.718754 ± 0.00280542 (W 0.00241969) | 0.991291 ± 0.0033613 (W 0.00420713) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.878806 ± 0.00334763 (W 0.00145093) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.095737 ± 0.00045807 (W 0.000480461) | 0.124897 ± 0.00102861 (W 0.000976926) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.182939 ± 0.000728447 (W 0.00170516) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.367618 ± 0.00149965 (W 0.00202904) | 0.440785 ± 0.00163533 (W 0.00278384) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.803994 ± 0.00364449 (W 0.00365006) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.41686 ± 0.00933603 (W 0.00416872) | 1.36672 ± 0.00671154 (W 0.00409872) | · |  |
| `lu/n512/b1 [api-inclusive]` | 3.20919 ± 0.0131509 (W 0.00604225) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0906725 ± 0.000630639 (W 0.000299386) | 0.124367 ± 0.00060905 (W 0.00169221) | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.18308 ± 0.000294133 (W 0.000553177) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.335071 ± 0.00113421 (W 0.00158807) | 0.428487 ± 0.0020326 (W 0.00332854) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.432776 ± 0.00174729 (W 0.00223154) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.32651 ± 0.00396393 (W 0.00317614) | 1.35792 ± 0.00770629 (W 0.00367106) | · |  |
| `solve/n512/b1 [api-inclusive]` | 3.22102 ± 0.0213496 (W 0.0067371) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0256195 ± 5.51144e-05 (W 4.53343e-05) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.0232495 ± 5.36619e-05 (W 5.59523e-05) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.112973 ± 0.00262621 (W 0.00118619) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.121626 ± 0.00153049 (W 0.00200206) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.678562 ± 0.0040452 (W 0.00251393) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.63604 ± 0.00668663 (W 0.00243184) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0466495 ± 9.44276e-05 (W 0.00015537) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0611485 ± 0.000117945 (W 0.000273364) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.200058 ± 0.000392676 (W 0.00135017) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.341259 ± 0.0011683 (W 0.00251217) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.01107 ± 0.00244585 (W 0.00120739) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.77092 ± 0.00391976 (W 0.004288) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0412195 ± 0.000298834 (W 0.000169937) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0624635 ± 0.000656387 (W 0.000270121) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.177425 ± 0.000454377 (W 0.00230475) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.266748 ± 0.00113699 (W 0.00189776) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.921906 ± 0.00404266 (W 0.0011508) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.38697 ± 0.0145181 (W 0.00505354) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.024954 ± 0.000957603 (W 0.000172236) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.024504 ± 0.000806929 (W 0.000216151) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.086293 ± 0.00038555 (W 0.000269912) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.104343 ± 0.000843064 (W 0.00146985) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.368189 ± 0.00511145 (W 0.00303668) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.399857 ± 0.0050286 (W 0.00185332) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0624885 ± 0.00163596 (W 0.0020503) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.0618685 ± 0.000643504 (W 0.000526308) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.1772 ± 0.00213895 (W 0.00377428) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.282871 ± 0.00170833 (W 0.00254525) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.677719 ± 0.00555715 (W 0.00652854) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.22403 ± 0.0118564 (W 0.00407144) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0476335 ± 0.000825615 (W 0.000628906) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1 [api-inclusive]` | 0.0608535 ± 0.00034243 (W 0.0002728) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.166801 ± 0.000655313 (W 0.000805102) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.285632 ± 0.00131666 (W 0.00263832) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.658212 ± 0.0132272 (W 0.00478306) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.20926 ± 0.006118 (W 0.00436592) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.025214 ± 0.000129924 (W 0.000114531) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.025005 ± 0.000119851 (W 0.000273154) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.0840285 ± 0.000137789 (W 0.000232954) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.104778 ± 0.000899009 (W 0.00100534) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.343172 ± 0.00572164 (W 0.00419882) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.366065 ± 0.00110597 (W 0.00221354) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0643235 ± 0.000884468 (W 0.00112184) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0586085 ± 0.000513582 (W 0.000466007) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.145181 ± 0.000840549 (W 0.00128563) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.276362 ± 0.000912549 (W 0.00182205) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.521884 ± 0.0123456 (W 0.00516542) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 1.12564 ± 0.00402332 (W 0.00450392) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0509545 ± 0.000876341 (W 0.000722229) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0611785 ± 0.000467072 (W 0.000459211) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.139641 ± 0.000773334 (W 0.00102818) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.27827 ± 0.000823103 (W 0.00197335) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.48129 ± 0.00278309 (W 0.00260847) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.12816 ± 0.0142767 (W 0.010782) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.019409 ± 8.93211e-05 (W 3.50858e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n128/b1 [api-inclusive]` | 0.0234 ± 7.91555e-05 (W 0.000392722) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.088227 ± 0.00127747 (W 0.00073035) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.102557 ± 0.00182492 (W 0.000576343) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.483092 ± 0.00396653 (W 0.00157189) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.532739 ± 0.00381445 (W 0.00209882) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.043089 ± 0.000588484 (W 0.000335791) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0468135 ± 0.000330089 (W 0.000693421) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.193764 ± 0.000373926 (W 0.00164746) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.275034 ± 0.00233762 (W 0.00191549) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.07767 ± 0.00378744 (W 0.000905337) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.463 ± 0.00468807 (W 0.0033811) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.041683 ± 0.00055545 (W 0.000731207) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.045769 ± 0.000688717 (W 0.000258535) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.188991 ± 0.000624791 (W 0.00174938) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.197474 ± 0.000547978 (W 0.00173245) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.05993 ± 0.00243241 (W 0.00113312) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.10621 ± 0.00706316 (W 0.0024031) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0197795 ± 0.000154273 (W 9.40552e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n128/b1 [api-inclusive]` | 0.0246395 ± 0.000130001 (W 0.000227733) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.067253 ± 0.000535306 (W 0.000678058) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.0836625 ± 0.00189393 (W 0.00048338) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.231174 ± 0.000984067 (W 0.00176161) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.300006 ± 0.00113302 (W 0.00164631) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.037554 ± 0.000661314 (W 0.000305486) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0455985 ± 0.000445719 (W 0.000386883) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.126142 ± 0.000580087 (W 0.000926008) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.215347 ± 0.000682194 (W 0.0016683) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.53626 ± 0.00131967 (W 0.00280884) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 0.930544 ± 0.00415093 (W 0.00370554) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.040094 ± 0.00014952 (W 0.00036171) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.043253 ± 0.000353388 (W 0.000309179) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.12841 ± 0.000747305 (W 0.00126632) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.215226 ± 0.00163813 (W 0.00158253) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.510015 ± 0.00188079 (W 0.0021599) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 0.914891 ± 0.0029892 (W 0.00297547) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.019664 ± 0.00111598 (W 0.000129672) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n128/b1 [api-inclusive]` | 0.025019 ± 0.000294511 (W 0.000301384) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.0656835 ± 0.00122903 (W 0.00116165) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.0843385 ± 0.00153518 (W 0.000472508) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.19707 ± 0.00069251 (W 0.00208591) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.272339 ± 0.000633315 (W 0.00187016) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.037809 ± 0.000530943 (W 0.000356828) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0423845 ± 0.000357768 (W 0.000354253) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.117317 ± 0.00100152 (W 0.000873758) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.20679 ± 0.000866648 (W 0.00181104) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.432643 ± 0.00200225 (W 0.00321259) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 0.830615 ± 0.00462709 (W 0.00288319) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.040559 ± 0.000428043 (W 0.000385249) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0439985 ± 0.000270748 (W 0.000483771) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.114208 ± 0.000578871 (W 0.000836201) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.208405 ± 0.000763453 (W 0.00200816) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.384278 ± 0.00165758 (W 0.00255021) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 0.82694 ± 0.00205953 (W 0.00349451) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.126766 ± 0.000140752 (W 0.000541486) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.125977 ± 0.000417561 (W 0.0010607) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.47689 ± 0.00128691 (W 0.00190699) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.460301 ± 0.00160632 (W 0.00184324) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.3056 ± 0.00535528 (W 0.00336063) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.13069 ± 0.0200671 (W 0.0022839) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0951835 ± 0.000482427 (W 0.00111168) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.115282 ± 0.000589772 (W 0.00126817) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.486426 ± 0.00160423 (W 0.00250372) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.764408 ± 0.00255457 (W 0.00244335) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.97516 ± 0.0176038 (W 0.00265623) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.298 ± 0.0084931 (W 0.00471432) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.091267 ± 0.000434788 (W 0.00109961) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.116076 ± 0.000411693 (W 0.000879516) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.466492 ± 0.00126732 (W 0.002011) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.5878 ± 0.00190677 (W 0.00308576) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.94669 ± 0.00480524 (W 0.0064588) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.40793 ± 0.0186698 (W 0.00469786) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.118642 ± 0.000363913 (W 0.000266307) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.119207 ± 0.000418777 (W 0.000290884) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.349274 ± 0.00164316 (W 0.00153297) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.345591 ± 0.000888706 (W 0.00146912) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.13374 ± 0.00310159 (W 0.00252667) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.117 ± 0.00488069 (W 0.00247101) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0971825 ± 0.000950776 (W 0.00121843) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0948375 ± 0.000659791 (W 0.000247564) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.337191 ± 0.001783 (W 0.00194498) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.544112 ± 0.00423014 (W 0.00329601) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.45639 ± 0.00293843 (W 0.00412038) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.58291 ± 0.00937745 (W 0.00781737) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.081288 ± 0.000349463 (W 0.000838277) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0969425 ± 0.000291603 (W 0.000275376) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.319541 ± 0.00560282 (W 0.00185781) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.540339 ± 0.00337233 (W 0.00290817) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.38478 ± 0.0123663 (W 0.00428893) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.47761 ± 0.00936632 (W 0.00450629) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.119206 ± 0.000151187 (W 0.000218156) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.119332 ± 0.000414936 (W 0.000366446) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.339155 ± 0.00133722 (W 0.00184085) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.336894 ± 0.00123488 (W 0.00183531) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.946804 ± 0.00462129 (W 0.00258449) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.992533 ± 0.00401581 (W 0.00228275) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.103371 ± 0.000939822 (W 0.000981389) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.105702 ± 0.0157598 (W 0.000587906) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.268307 ± 0.00486601 (W 0.00173345) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.512601 ± 0.00597301 (W 0.00265425) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.0702 ± 0.00299542 (W 0.00222901) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.26151 ± 0.0106523 (W 0.00332936) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.075203 ± 0.000433161 (W 0.00134165) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.094407 ± 0.000622434 (W 0.000411247) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.255722 ± 0.000845082 (W 0.00154066) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.519016 ± 0.00326047 (W 0.00270283) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.983299 ± 0.00624213 (W 0.00295506) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.21537 ± 0.0100052 (W 0.00413773) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.118432 ± 0.000117828 (W 0.00111029) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.121871 ± 0.000161652 (W 0.000960648) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.412742 ± 0.00143445 (W 0.00186583) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.427113 ± 0.000694568 (W 0.00181006) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.93657 ± 0.0140468 (W 0.00253713) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.04793 ± 0.0270735 (W 0.00213761) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.091063 ± 0.000602438 (W 0.00129913) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0975215 ± 0.000177643 (W 0.000927561) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.489905 ± 0.00238827 (W 0.00170269) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.679089 ± 0.00165195 (W 0.00276782) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.98196 ± 0.0101786 (W 0.00279479) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.85389 ± 0.00913881 (W 0.00466821) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.090772 ± 0.000202657 (W 0.000392927) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0963765 ± 0.000334747 (W 0.0011674) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.480846 ± 0.00194267 (W 0.00227855) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.509313 ± 0.00119308 (W 0.0017253) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.91966 ± 0.0150053 (W 0.00339888) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.02212 ± 0.00587968 (W 0.00375772) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.110658 ± 0.000272787 (W 0.000254584) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.115307 ± 0.000258516 (W 0.00021936) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.293926 ± 0.000930558 (W 0.00135723) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.314361 ± 0.00141392 (W 0.00124323) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.912825 ± 0.00329565 (W 0.00175862) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.987544 ± 0.00137335 (W 0.000899207) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.069012 ± 0.00063863 (W 0.000531343) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.076608 ± 0.00044605 (W 0.00044546) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.273342 ± 0.000499424 (W 0.0015637) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.462775 ± 0.00161563 (W 0.00379189) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.29041 ± 0.0124946 (W 0.00269789) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.17814 ± 0.0077033 (W 0.00620099) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0705875 ± 0.000200989 (W 0.000397234) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0775535 ± 0.00027671 (W 0.000361076) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.264912 ± 0.000945917 (W 0.00161005) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.460024 ± 0.00239649 (W 0.00274607) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.1845 ± 0.00295213 (W 0.00356669) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.0836 ± 0.0118457 (W 0.00385876) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.110372 ± 0.00017086 (W 0.000241551) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.115447 ± 0.000142396 (W 0.000227762) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.281679 ± 0.000609885 (W 0.00142527) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.30435 ± 0.000669094 (W 0.00142737) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.760744 ± 0.0013002 (W 0.00217845) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.861988 ± 0.000810314 (W 0.0024361) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0648735 ± 0.000556712 (W 0.000335764) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.087317 ± 0.015123 (W 0.000541332) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.242645 ± 0.00107233 (W 0.00192126) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.434238 ± 0.00165738 (W 0.00287531) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.991902 ± 0.0571863 (W 0.00435664) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 1.86986 ± 0.00552796 (W 0.00547147) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0648235 ± 0.000686811 (W 0.00038716) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.075067 ± 0.000438291 (W 0.000382792) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.22012 ± 0.000554341 (W 0.00173859) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.438539 ± 0.00220727 (W 0.00217833) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.867896 ± 0.00211145 (W 0.00223553) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.83552 ± 0.00729081 (W 0.00469784) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `lu/128`, `lu/256`, `lu/512`, `solve/128`, `solve/256`, `solve/512`.

## linalg-tall, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.186969 ± 0.00105464 (W 0.00145425) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.191494 ± 0.00540779 (W 0.00154187) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 5.63313 ± 0.0178324 (W 0.0174339) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.58821 ± 0.0983185 (W 0.0117435) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.170806 ± 0.000754532 (W 0.00163484) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.187529 ± 0.00445271 (W 0.0016967) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1` | 2.71272 ± 0.0600261 (W 0.0166425) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/4096x256/b1 [api-inclusive]` | 3.31707 ± 0.0933592 (W 0.0101645) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.150086 ± 0.00114905 (W 0.000887099) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.185845 ± 0.00147415 (W 0.00159216) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.06376 ± 0.0294592 (W 0.0209547) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.94553 ± 0.085611 (W 0.00850661) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.122397 ± 0.00071761 (W 0.00141464) | 0.121582 ± 0.000168501 (W 0.00172762) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.129325 ± 0.00133501 (W 0.00123425) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.44873 ± 0.0119861 (W 0.00412021) | 4.74387 ± 0.00577654 (W 0.0081686) | 🔵 colmat 1.07× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.49037 ± 0.00642858 (W 0.00543181) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.108998 ± 0.00173422 (W 0.000705001) | 0.109867 ± 0.000562118 (W 0.000842287) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.127816 ± 0.000429172 (W 0.000536142) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.98631 ± 0.0192383 (W 0.00489213) | 2.73342 ± 0.00828131 (W 0.00312953) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 2.19565 ± 0.0102659 (W 0.0111024) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.107233 ± 0.000621785 (W 0.000705287) | 0.105502 ± 0.000488074 (W 0.000396709) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.126747 ± 0.00138579 (W 0.000728164) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.71467 ± 0.00990141 (W 0.00432464) | 1.86889 ± 0.00652478 (W 0.00451637) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 1.97914 ± 0.00676333 (W 0.00466446) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.47328 ± 0.00303455 (W 0.00194277) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.488216 ± 0.00233946 (W 0.00200668) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 18.1466 ± 0.035701 (W 0.030327) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 20.2618 ± 0.118066 (W 0.0669195) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.341637 ± 0.00198191 (W 0.001766) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.374217 ± 0.00140586 (W 0.00177657) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.70568 ± 0.0340812 (W 0.0304318) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 10.7777 ± 0.0487164 (W 0.089117) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.315265 ± 0.00339063 (W 0.00164316) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.363308 ± 0.0014937 (W 0.00190277) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.3752 ± 0.0537662 (W 0.0125176) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 9.05634 ± 0.0714132 (W 0.159125) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.397469 ± 0.00259706 (W 0.0020756) | 0.379143 ± 0.00216219 (W 0.00178672) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.414166 ± 0.000549599 (W 0.00213355) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 15.6989 ± 0.00836627 (W 0.0158266) | 17.5199 ± 0.0290572 (W 0.010757) | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 16.045 ± 0.0214441 (W 0.0105298) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.272822 ± 0.00121575 (W 0.00125499) | 0.296366 ± 0.0037064 (W 0.00176716) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.312455 ± 0.00194255 (W 0.00155739) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.25359 ± 0.013153 (W 0.00724482) | 9.39407 ± 0.0205202 (W 0.00860397) | 🔵 colmat 1.50× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 6.8697 ± 0.0193715 (W 0.00968776) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.262037 ± 0.00183174 (W 0.00174976) | 0.256459 ± 0.00228795 (W 0.00303386) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.304557 ± 0.00124568 (W 0.00182153) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.13419 ± 0.0644649 (W 0.00643075) | 6.0317 ± 0.0630602 (W 0.0223899) | 🔵 colmat 1.17× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 5.70444 ± 0.0314559 (W 0.00952745) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.202321 ± 0.00142784 (W 0.00160845) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.157751 ± 0.000846362 (W 0.00145847) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.97273 ± 0.031692 (W 0.00305674) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 3.81291 ± 0.0844102 (W 0.0294544) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.136526 ± 0.00182643 (W 0.00100893) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.165471 ± 0.000765477 (W 0.000720955) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.05146 ± 0.0207324 (W 0.0370352) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.67388 ± 0.0301764 (W 0.0226302) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.124147 ± 0.000982032 (W 0.000597064) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.16573 ± 0.000451281 (W 0.000758565) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.64483 ± 0.0214344 (W 0.0260255) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.56042 ± 0.0284456 (W 0.0164634) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.0855225 ± 0.00106263 (W 0.000806663) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.088673 ± 0.000797209 (W 0.000453515) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.56843 ± 0.00555944 (W 0.0040303) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.62957 ± 0.00668174 (W 0.00546635) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.085663 ± 0.000519961 (W 0.000637913) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.093762 ± 0.000534025 (W 0.000506759) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.37699 ± 0.00376104 (W 0.003318) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 1.52128 ± 0.00318181 (W 0.00367909) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.086928 ± 0.000424051 (W 0.000655774) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.094972 ± 0.000727979 (W 0.000500679) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.30289 ± 0.00535457 (W 0.00332646) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 1.45877 ± 0.00251227 (W 0.00327502) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.371507 ± 0.00319301 (W 0.00332241) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.347948 ± 0.00285375 (W 0.00189747) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.1063 ± 0.0129661 (W 0.00774398) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 10.0253 ± 0.0264765 (W 0.0105108) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.282918 ± 0.00353349 (W 0.00261429) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.311411 ± 0.00118559 (W 0.00196101) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.94829 ± 0.0104118 (W 0.0122347) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.63882 ± 0.0184985 (W 0.0103722) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.255817 ± 0.00280933 (W 0.00154285) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.306013 ± 0.00167993 (W 0.0029424) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.06639 ± 0.0090535 (W 0.0125589) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.194 ± 0.0507049 (W 0.00753636) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.257047 ± 0.000488639 (W 0.00180524) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.269414 ± 0.000725262 (W 0.00126325) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.49498 ± 0.0211305 (W 0.00944985) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 8.6378 ± 0.0157567 (W 0.0127616) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.216449 ± 0.00249723 (W 0.00260791) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.235594 ± 0.00121114 (W 0.00268171) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.03416 ± 0.00999521 (W 0.0104731) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 4.2842 ± 0.00927847 (W 0.0118674) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/1024x64/b1` | 0.209935 ± 0.000762377 (W 0.0020624) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.229057 ± 0.00114832 (W 0.00167048) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.59767 ± 0.0108617 (W 0.00773692) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 3.87246 ± 0.00956494 (W 0.00890634) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `lu/1024x64`, `lu/4096x256`, `solve/1024x64`, `solve/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 5.04508 ± 0.0125375 (W 0.00482323) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.89855 ± 0.122118 (W 0.0457808) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.15942 ± 0.000424413 (W 0.00142572) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.248103 ± 0.00390105 (W 0.00182653) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 1.63263 ± 0.0112483 (W 0.00810293) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.7523 ± 0.0561341 (W 0.00917693) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.0747875 ± 0.00129279 (W 0.00134076) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.132262 ± 0.00342102 (W 0.000349361) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 1.01195 ± 0.00597318 (W 0.0047912) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.07086 ± 0.0594391 (W 0.0309135) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.0624935 ± 0.000692627 (W 0.00100736) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.116267 ± 0.00660128 (W 0.000312953) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 5.75927 ± 0.010792 (W 0.00421236) | 9.23236 ± 0.026539 (W 0.00716848) | 🔵 colmat 1.60× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 5.91963 ± 0.0168243 (W 0.00554739) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.175754 ± 0.00100313 (W 0.00143517) | 0.403448 ± 0.00468218 (W 0.00170356) | 🔵 colmat 2.30× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.184356 ± 0.00105849 (W 0.0017536) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 1.56004 ± 0.00373343 (W 0.00239816) | 5.56264 ± 0.0109259 (W 0.00539207) | 🔵 colmat 3.57× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 1.8315 ± 0.00415415 (W 0.00455717) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0577935 ± 0.000347458 (W 0.000244478) | 0.273496 ± 0.000900168 (W 0.00118112) | 🔵 colmat 4.73× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.070703 ± 0.000656469 (W 0.000271921) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 0.863511 ± 0.00324749 (W 0.00412579) | 4.58715 ± 0.00683246 (W 0.00465933) | 🔵 colmat 5.31× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 1.13523 ± 0.0096352 (W 0.00478447) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0379835 ± 8.51362e-05 (W 0.00030332) | 0.257887 ± 0.00823845 (W 0.00211946) | 🔵 colmat 6.79× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.0554785 ± 0.000582696 (W 0.000297674) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 16.2743 ± 0.0541155 (W 0.013595) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 19.9461 ± 0.0612825 (W 0.118011) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.44786 ± 0.00268697 (W 0.00159372) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.528963 ± 0.00641893 (W 0.00211703) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 4.64083 ± 0.0259419 (W 0.00976416) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 7.31406 ± 0.0492025 (W 0.0609744) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.160925 ± 0.00149453 (W 0.0016554) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.230693 ± 0.000640948 (W 0.00134405) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 2.62464 ± 0.0136874 (W 0.00937614) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 5.14436 ± 0.0709764 (W 0.0376095) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.113732 ± 0.000480021 (W 0.00126938) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.185635 ± 0.000885131 (W 0.00149336) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 17.0196 ± 0.125773 (W 0.0436862) | 23.7503 ± 0.187915 (W 0.0182706) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 17.4638 ± 0.0412813 (W 0.0133119) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.446512 ± 0.00674231 (W 0.00185639) | 0.627916 ± 0.00523314 (W 0.00180445) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.466338 ± 0.0012938 (W 0.00311241) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 4.61959 ± 0.00796731 (W 0.0116219) | 13.4394 ± 0.0266035 (W 0.020666) | 🔵 colmat 2.91× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 5.22042 ± 0.0269281 (W 0.010265) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.133796 ± 0.000320296 (W 0.000821339) | 0.397203 ± 0.00320757 (W 0.00158774) | 🔵 colmat 2.97× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.163106 ± 0.00371645 (W 0.000890259) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 2.47465 ± 0.00734704 (W 0.00450252) | 10.5873 ± 0.023821 (W 0.0113627) | 🔵 colmat 4.28× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 3.1074 ± 0.00913915 (W 0.00988405) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0831635 ± 0.00257782 (W 0.00030094) | 0.335672 ± 0.00279874 (W 0.00213408) | 🔵 colmat 4.04× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.117387 ± 0.00104692 (W 0.000253968) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 2.94199 ± 0.0172115 (W 0.00291303) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 4.90923 ± 0.0628928 (W 0.032694) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.106987 ± 0.000371748 (W 0.000490611) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.229073 ± 0.00154428 (W 0.00141841) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 0.967954 ± 0.00721657 (W 0.00316149) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.29984 ± 0.0441248 (W 0.0223039) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.050784 ± 0.000338142 (W 0.000315037) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.126641 ± 0.000450059 (W 0.000418489) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 0.680373 ± 0.00202202 (W 0.00339916) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.96831 ± 0.0755924 (W 0.0823431) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.042719 ± 0.000295205 (W 0.000429555) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.110357 ± 0.000474169 (W 0.000507838) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 3.71922 ± 0.00938063 (W 0.00430392) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.67955 ± 0.0150934 (W 0.00459412) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.153771 ± 0.00131092 (W 0.00165378) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.15853 ± 0.00103489 (W 0.00109912) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 1.02063 ± 0.00340783 (W 0.00215779) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.1348 ± 0.00403057 (W 0.00306747) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0495435 ± 0.00065302 (W 0.00026076) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.057053 ± 0.000373138 (W 0.000276071) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 0.574077 ± 0.00148092 (W 0.00255371) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 0.680844 ± 0.00666615 (W 0.00209828) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.032499 ± 0.000330063 (W 0.000238661) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.0409335 ± 0.000256691 (W 0.000342377) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 9.20109 ± 0.0165363 (W 0.00727284) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 11.2638 ± 0.0232646 (W 0.0114417) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.288003 ± 0.0018277 (W 0.00161568) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.388354 ± 0.00215011 (W 0.00235059) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 2.64938 ± 0.00545046 (W 0.00761918) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 4.17988 ± 0.029314 (W 0.0103263) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.113957 ± 0.000972763 (W 0.00120264) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.188634 ± 0.00279828 (W 0.00143367) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 1.60248 ± 0.0145027 (W 0.00469414) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.99982 ± 0.0374241 (W 0.00777164) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.090648 ± 0.000962787 (W 0.000991046) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.157271 ± 0.000609694 (W 0.000707115) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 1T (CPU 0)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 9.74865 ± 0.0134691 (W 0.00930252) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.91768 ± 0.0164391 (W 0.0103064) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.300677 ± 0.000963777 (W 0.00116625) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.310768 ± 0.000906951 (W 0.00151964) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 2.64037 ± 0.00408753 (W 0.00396939) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.89309 ± 0.00596885 (W 0.00582566) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.092682 ± 0.000158679 (W 0.000212622) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.106816 ± 0.000514068 (W 0.000306506) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `lu/256x4096/b1` | 1.44096 ± 0.00221773 (W 0.00316305) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.72514 ± 0.00654253 (W 0.00573533) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0587185 ± 0.00042456 (W 0.000256255) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.0755785 ± 0.000795105 (W 0.000231089) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `lu/256x4096`, `lu/64x1024`, `solve/256x4096`, `solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
