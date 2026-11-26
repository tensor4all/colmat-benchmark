# `colmat-kernels` on `evo2-zen5-16c`

- colmat (fork, measured) commit: `81560db94498478e8dd65153739e208391e434b6`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- colmat / faer rows: `colmat` (the provider in the measured revision; see library and upstream above); `faer` (the provider in the measured revision; see library and upstream above)
- harness commit: `5ee382265b3ed6081be88d113019fdd3797cac5b`
- toolkit commit: `b1302a25e837d7ab9070005fcb9afdc5e18afd37`
- measured binary SHA-256: `9b750afe9b03059051b1c732573a4fa3e0088632bb501f8a1eb21d57b26986fe`
- hardware profile: `evo2-zen5-16c`
- host role: `measurement` — published dataset host
- timestamp: `2026-11-25T21:10:12.334826Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py evo2-zen5-16c colmat-kernels --vendor mkl --engines colmat,faer,lapack-mkl --threads 1,4,8 --reps 5 --aa 3 --prime-ms 250 --label evo2-81560db9-small-mkl`
- harness environment: `FBENCH_ORDER` unset, `FBENCH_PROFILE=evo2-zen5-16c`
- raw data: `data/results/evo2-zen5-16c/colmat-kernels/20261125T175201Z-evo2-81560db9-small-mkl/`

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
  - `faer` — Upstream `faer` at the tag the fork is based on, in the same binary as the fork row. The baseline that says whether a fork change moved the number.
    Identity: the upstream revision in this run's manifest (`upstream.commit`), pinned by `pins/upstream-faer.rev`
    Caveat: Same crate-name and feature-unification caveat as `colmat`. A change to upstream at a fixed fork revision produces a different page rather than overwriting one.
    As measured: `faer` (the provider in the measured revision; see library and upstream above)
  - `lapack-mkl` — The same calls against Intel oneAPI MKL, linked from the installation `MKLROOT` names. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run; no page carries two vendor arms. The LP64 interface layer and the threading layer are pinned by the harness before the first MKL call, because MKL otherwise lets the environment choose them and the interface layer is an ABI.
    Identity: the MKL build recorded in this run's manifest (`providers[mkl]`)
    Caveat: MKL reports no dispatched-kernel name and no processor count, so its manifest records neither rather than inferring them. Its threading is vendor-owned, as for OpenBLAS.
    As measured: linked shared; mkl 2026.2; kernel not reported by the vendor

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
Each process retains 5 chronological calls; 3 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 0.76%, worst 1812.49% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 529 of 9684 engine cells exceed 5% and carry `!`; median within-process relative SD 0.44%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00095 ± 5.7735e-06 (W 3.20936e-05) | 0.00093 ± 0 (W 3.11448e-05) | 0.00294 ± 1.1547e-05 (W 4.52769e-05) | · | 🔵 colmat 3.09× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00615 ± 5.55608e-05 (W 3.46006e-05) | 0.00613 ± 4.30155e-05 (W 3.1132e-05) | 0.01598 ± 2.3094e-05 (W 3.21279e-05) | · | 🔵 colmat 2.60× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.019699 ± 0.000127084 (W 3.73658e-05) | 0.019739 ± 3.78594e-05 (W 5.3163e-05) | 0.041548 ± 5.10327e-05 (W 5.85261e-05) | · | 🔵 colmat 2.11× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00089 ± 2.08167e-05 (W 4.71169e-05) | 0.00203 ± 6.35085e-05 (W 0.000129616) | 0.00343 ± 1.52753e-05 (W 6.58027e-05) | 🔵 colmat 2.28× faster than `faer` | 🔵 colmat 3.85× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00229 ± 9.50438e-06 (W 4.03733e-05) | 0.01073 ± 0.000208231 (W 0.000606724) | 0.009409 ± 4.7085e-05 (W 6.13213e-05) | 🔵 colmat 4.69× faster than `faer` | 🔵 colmat 4.11× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.005769 ± 4.14769e-05 (W 4.64995e-05) | 0.033718 ± 0.00845709 (W 0.00820445) | 0.014469 ± 3.46458e-05 (W 8.89174e-05) | 🔵 colmat 5.84× faster than `faer` | 🔵 colmat 2.51× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00093 ± 5.07642e-05 (W 3.32145e-05) | 0.00232 ± 8.3865e-05 (W 9.36483e-05) | 0.00326 ± 5.2915e-05 (W 0.000101882) | 🔵 colmat 2.49× faster than `faer` | 🔵 colmat 3.51× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00175 ± 2.08167e-05 (W 5.19615e-05) | 0.0046 ± 8.544e-05 (W 0.000110544) | 0.006519 ± 0.0184466 (W 0.000102447) | 🔵 colmat 2.63× faster than `faer` | · | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0035 ± 2e-05 (W 4.09292e-05) | 0.00949 ± 0.0001249 (W 7.48285e-05) | 0.009939 ± 0.000593743 (W 0.000894953) | 🔵 colmat 2.71× faster than `faer` | 🔵 colmat 2.84× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00095 ± 0 (W 3.1305e-05) | 0.00091 ± 1.73205e-05 (W 3.04959e-05) | 0.00298 ± 5.7735e-06 (W 3.64692e-05) | · | 🔵 colmat 3.14× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.006139 ± 3.01164e-05 (W 4.08069e-05) | 0.0062 ± 9.29157e-05 (W 4.32435e-05) | 0.016029 ± 8.08723e-05 (W 3.48755e-05) | · | 🔵 colmat 2.61× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.019789 ± 6.46555e-05 (W 5.47193e-05) | 0.01991 ± 6.05007e-05 (W 7.305e-05) | 0.041529 ± 3.60555e-05 (W 4.57854e-05) | · | 🔵 colmat 2.10× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00086 ± 1e-05 (W 4.20714e-05) | 0.00207 ± 8.02081e-05 (W 0.000132778) | 0.00349 ± 9.29157e-05 (W 0.000137222) | 🔵 colmat 2.41× faster than `faer` | 🔵 colmat 4.06× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0023 ± 1.73205e-05 (W 1.92354e-05) | 0.011119 ± 0.000181751 (W 0.000443273) | 0.009289 ± 7.59408e-05 (W 3.8227e-05) | 🔵 colmat 4.83× faster than `faer` | 🔵 colmat 4.04× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00566 ± 2.51661e-05 (W 3.10048e-05) | 0.037989 ± 0.000748726 (W 0.00767654) | 0.01455 ± 0.00010982 (W 0.000115548) | 🔵 colmat 6.71× faster than `faer` | 🔵 colmat 2.57× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00095 ± 1.95021e-05 (W 2.91548e-05) | 0.00231 ± 9.50438e-05 (W 0.000121161) | 0.00329 ± 0.0344277 (W 0.000130458) | 🔵 colmat 2.43× faster than `faer` | · | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00177 ± 5.7735e-07 (W 2.58844e-05) | 0.0047 ± 9.29157e-05 (W 0.00140143) | 0.0065 ± 8.5049e-05 (W 0.000137467) | 🔵 colmat 2.66× faster than `faer` | 🔵 colmat 3.67× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00341 ± 4.16333e-05 (W 3.6086e-05) | 0.00934 ± 0.00020232 (W 0.000161457) | 0.00966 ± 6.45523e-05 (W 0.000126516) | 🔵 colmat 2.74× faster than `faer` | 🔵 colmat 2.83× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00323 ± 2.51661e-05 (W 2.48837e-05) | 0.00322 ± 1.52753e-05 (W 1.92354e-05) | 0.00737 ± 6.08276e-05 (W 6.97947e-05) | · | 🔵 colmat 2.28× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.023129 ± 4.58258e-05 (W 2.70592e-05) | 0.02317 ± 0.000150489 (W 5.24995e-05) | 0.055599 ± 0.000632503 (W 0.000997396) | · | 🔵 colmat 2.40× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.076867 ± 0.000222579 (W 7.38241e-05) | 0.077149 ± 0.000103317 (W 0.000107145) | 0.180003 ± 0.00167759 (W 0.00231838) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |

## gemm-small, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00147 ± 3.21455e-05 (W 4.11971e-05) | 0.00322 ± 3.05505e-05 (W 2.60768e-05) | 0.00891 ± 0.000121244 (W 7.71116e-05) | 🔵 colmat 2.19× faster than `faer` | 🔵 colmat 6.06× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0069 ± 2.51661e-05 (W 0.000104691) | 0.01925 ± 0.000927464 (W 0.00168381) | 0.023709 ± 0.000131074 (W 0.00025266) | 🔵 colmat 2.79× faster than `faer` | 🔵 colmat 3.44× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.02033 ± 0.000105263 (W 5.06971e-05) | 0.040919 ± 0.00155288 (W 0.00228016) | 0.065078 ± 0.000775094 (W 0.00128993) | 🔵 colmat 2.01× faster than `faer` | 🔵 colmat 3.20× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-small, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00138 ± 1.73205e-05 (W 3.56371e-05) | 0.00321 ± 5.7735e-06 (W 3.03315e-05) | 0.00975 ± 0.000187705 (W 0.00074032) | 🔵 colmat 2.33× faster than `faer` | 🔵 colmat 7.07× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00391 ± 3.51188e-05 (W 6.04152e-05) | 0.010479 ± 2.13776e-05 (W 0.000168953) | 0.01097 ± 0.000270247 (W 9.37881e-05) | 🔵 colmat 2.68× faster than `faer` | 🔵 colmat 2.81× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0109 ± 5.0797e-05 (W 3.97026e-05) | 0.014809 ± 4.00042e-05 (W 0.000118298) | 0.028169 ± 8.32666e-05 (W 0.000365349) | 🔵 colmat 1.36× faster than `faer` | 🔵 colmat 2.58× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00323 ± 1.52753e-05 (W 3.39116e-05) | 0.00322 ± 1.1547e-05 (W 4.08656e-05) | 0.007379 ± 4.95681e-05 (W 7.08604e-05) | · | 🔵 colmat 2.28× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.02311 ± 5.63057e-05 (W 4.91752e-05) | 0.02314 ± 5.50757e-05 (W 5.06971e-05) | 0.054768 ± 0.000505767 (W 0.000742318) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `gemm/96x96x96/b1` | 0.076787 ± 0.000263213 (W 0.000918923) | 0.076957 ± 0.000126429 (W 6.96419e-05) | 0.178323 ± 0.00124706 (W 0.00249755) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |

## gemm-small, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00148 ± 2e-05 (W 2.12132e-05) | 0.00322 ± 1.52753e-05 (W 2.50998e-05) | 0.009059 ± 0.000113168 (W 0.00060885) | 🔵 colmat 2.18× faster than `faer` | 🔵 colmat 6.12× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00678 ± 1.73205e-05 (W 2.2676e-05) | 0.01938 ± 0.000481082 (W 0.00138911) | 0.023809 ± 0.000245846 (W 0.000220448) | 🔵 colmat 2.86× faster than `faer` | 🔵 colmat 3.51× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.02022 ± 3.78594e-05 (W 3.75393e-05) | 0.042478 ± 0.00124463 (W 0.00105121) | 0.064897 ± 0.000344004 (W 0.00110484) | 🔵 colmat 2.10× faster than `faer` | 🔵 colmat 3.21× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-small, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00133 ± 3.51188e-05 (W 3.39116e-05) | 0.00322 ± 6.08276e-06 (W 2.79285e-05) | 0.009769 ± 0.000158547 (W 0.000661022) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 7.35× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00417 ± 5.03322e-05 (W 3.59054e-05) | 0.0105 ± 3.78594e-05 (W 0.000104976) | 0.01105 ± 5.50757e-05 (W 0.00013908) | 🔵 colmat 2.52× faster than `faer` | 🔵 colmat 2.65× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.011 ± 2.51661e-05 (W 3.78457e-05) | 0.0149 ± 0.0001905 (W 0.00019305) | 0.028079 ± 0.000175072 (W 0.000286269) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 2.55× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00056 ± 5.7735e-06 (W 2.30217e-05) | 0.00054 ± 5.7735e-06 (W 2.88097e-05) | 0.00118 ± 3.05505e-05 (W 3.56371e-05) | · | 🔵 colmat 2.11× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00303 ± 1.52753e-05 (W 2.48998e-05) | 0.003 ± 0 (W 2.77705e-05) | 0.00752 ± 2.04206e-05 (W 0.000108481) | · | 🔵 colmat 2.48× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.009719 ± 1.53948e-05 (W 1.92354e-05) | 0.00972 ± 2e-05 (W 2.87611e-05) | 0.02352 ± 0.000277705 (W 3.78048e-05) | · | 🔵 colmat 2.42× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00061 ± 0 (W 2.96648e-05) | 0.00175 ± 1.52753e-05 (W 0.000139032) | 0.00221 ± 6.95725e-05 (W 9.13575e-05) | 🔵 colmat 2.87× faster than `faer` | 🔵 colmat 3.62× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00154 ± 2.08167e-05 (W 4.86826e-05) | 0.00604 ± 0.000275379 (W 0.000543495) | 0.0058 ± 3.50761e-05 (W 7.80333e-05) | 🔵 colmat 3.92× faster than `faer` | 🔵 colmat 3.77× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00313 ± 4.16333e-05 (W 1e-05) | 0.01904 ± 0.000267358 (W 0.000536043) | 0.010319 ± 0.00012822 (W 9.41764e-05) | 🔵 colmat 6.08× faster than `faer` | 🔵 colmat 3.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00061 ± 1e-05 (W 2.94958e-05) | 0.00227 ± 0.000174738 (W 0.000131795) | 0.00242 ± 5.85947e-05 (W 0.000137949) | 🔵 colmat 3.72× faster than `faer` | 🔵 colmat 3.97× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00152 ± 4e-05 (W 3.27109e-05) | 0.00296 ± 4.04145e-05 (W 0.000120955) | 0.00358 ± 5.1316e-05 (W 0.000103311) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 2.36× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00288 ± 3.05505e-05 (W 3.78153e-05) | 0.00491 ± 0.000231157 (W 0.00018131) | 0.00721 ± 6.50641e-05 (W 0.000126965) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 2.50× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00055 ± 1e-05 (W 2.88097e-05) | 0.00052 ± 0 (W 2.79285e-05) | 0.00112 ± 1.1547e-05 (W 4.38178e-05) | 🔴 colmat 1.06× slower than `faer` (marginal: spread 0.96 of the difference) | 🔵 colmat 2.04× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00302 ± 1e-05 (W 2.70185e-05) | 0.003 ± 9.50438e-06 (W 3.71914e-05) | 0.00745 ± 9.02053e-05 (W 0.000105935) | · | 🔵 colmat 2.47× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00973 ± 1.52753e-05 (W 2.74827e-05) | 0.0097 ± 1.09697e-05 (W 3.24546e-05) | 0.02351 ± 0.000260001 (W 0.000222338) | · | 🔵 colmat 2.42× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00062 ± 1e-05 (W 2.6612e-05) | 0.0017 ± 3.60555e-05 (W 9.39681e-05) | 0.002269 ± 5.83981e-05 (W 8.44393e-05) | 🔵 colmat 2.74× faster than `faer` | 🔵 colmat 3.66× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00155 ± 2.64575e-05 (W 3.52165e-05) | 0.00569 ± 0.000430988 (W 0.000893549) | 0.005779 ± 6.10765e-05 (W 6.77363e-05) | 🔵 colmat 3.67× faster than `faer` | 🔵 colmat 3.73× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00309 ± 1.1547e-05 (W 4.08656e-05) | 0.019069 ± 0.000387957 (W 0.00058209) | 0.010279 ± 0.000112962 (W 0.000112531) | 🔵 colmat 6.17× faster than `faer` | 🔵 colmat 3.33× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00061 ± 5.7735e-06 (W 2.82843e-05) | 0.00226 ± 4.30155e-05 (W 0.000182692) | 0.0025 ± 4.50925e-05 (W 0.000142934) | 🔵 colmat 3.70× faster than `faer` | 🔵 colmat 4.10× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00139 ± 3.21455e-05 (W 2.54951e-05) | 0.00303 ± 5.61278e-05 (W 0.000142352) | 0.00355 ± 7.54983e-05 (W 8.17313e-05) | 🔵 colmat 2.18× faster than `faer` | 🔵 colmat 2.55× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00226 ± 1.1547e-05 (W 3.00533e-05) | 0.00493 ± 8.44808e-05 (W 0.000201008) | 0.007259 ± 4.6694e-05 (W 0.000160062) | 🔵 colmat 2.18× faster than `faer` | 🔵 colmat 3.21× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00161 ± 1.73205e-05 (W 3.39116e-05) | 0.00158 ± 1.1547e-05 (W 3.205e-05) | 0.00365 ± 5.7735e-06 (W 4.76445e-05) | · | 🔵 colmat 2.27× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01205 ± 4.14769e-05 (W 2.70185e-05) | 0.01203 ± 3.04467e-05 (W 3.06317e-05) | 0.027779 ± 0.00033721 (W 0.000474995) | · | 🔵 colmat 2.31× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.038299 ± 0.000118773 (W 4.94297e-05) | 0.038379 ± 4.16333e-05 (W 0.00077976) | 0.090809 ± 0.000977941 (W 0.00109603) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |

## gemm-small, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.00114 ± 3.51188e-05 (W 3.89872e-05) | 0.00274 ± 5.50757e-05 (W 0.000109909) | 0.00356 ± 5.7735e-06 (W 6.41872e-05) | 🔵 colmat 2.40× faster than `faer` | 🔵 colmat 3.12× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00361 ± 1.52753e-05 (W 4.14512e-05) | 0.0133 ± 0.000335012 (W 0.000716952) | 0.0127 ± 0.000142339 (W 0.000294305) | 🔵 colmat 3.68× faster than `faer` | 🔵 colmat 3.52× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.01044 ± 1.52753e-05 (W 4.09671e-05) | 0.042808 ± 0.000248818 (W 0.00756885) | 0.033159 ± 0.0004753 (W 0.000393088) | 🔵 colmat 4.10× faster than `faer` | 🔵 colmat 3.18× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0012 ± 4.58258e-05 (W 4.02492e-05) | 0.00295 ± 0.000104403 (W 0.000126768) | 0.003049 ± 3.4356e-05 (W 0.000108028) | 🔵 colmat 2.46× faster than `faer` | 🔵 colmat 2.54× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00265 ± 4.72582e-05 (W 2.58109e-05) | 0.00758 ± 7.76745e-05 (W 0.000123488) | 0.00646 ± 8.544e-05 (W 0.000145462) | 🔵 colmat 2.86× faster than `faer` | 🔵 colmat 2.44× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00577 ± 0.00023388 (W 4.5926e-05) | 0.014609 ± 0.000231128 (W 0.000194307) | 0.01541 ± 0.000250067 (W 0.000192562) | 🔵 colmat 2.53× faster than `faer` | 🔵 colmat 2.67× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## gemm-small, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0016 ± 1e-05 (W 2.70961e-05) | 0.00159 ± 5.7735e-06 (W 3.42053e-05) | 0.00368 ± 5.19615e-06 (W 7.52994e-05) | · | 🔵 colmat 2.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01206 ± 6.35085e-06 (W 3.02704e-05) | 0.012059 ± 2.46374e-05 (W 3.27109e-05) | 0.027679 ± 0.000274044 (W 0.000266646) | · | 🔵 colmat 2.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.038418 ± 8.55005e-05 (W 3.06806e-05) | 0.038419 ± 7e-05 (W 3.27613e-05) | 0.089517 ± 0.000815507 (W 0.00134502) | · | 🔵 colmat 2.33× faster than `lapack-mkl` |  |

## gemm-small, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0011 ± 2.88675e-05 (W 2.54951e-05) | 0.00281 ± 0.000109697 (W 0.000114513) | 0.00355 ± 2.64575e-05 (W 8e-05) | 🔵 colmat 2.55× faster than `faer` | 🔵 colmat 3.23× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00359 ± 1.47309e-05 (W 2.59461e-05) | 0.013519 ± 0.000225123 (W 0.000568424) | 0.012679 ± 2.51064e-05 (W 0.000166524) | 🔵 colmat 3.77× faster than `faer` | 🔵 colmat 3.53× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.01043 ± 2.13776e-05 (W 2.78442e-05) | 0.042428 ± 0.000248424 (W 0.00727425) | 0.033479 ± 0.000468231 (W 0.000440796) | 🔵 colmat 4.07× faster than `faer` | 🔵 colmat 3.21× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0011 ± 5.19615e-05 (W 4.39318e-05) | 0.003 ± 6.245e-05 (W 0.000145086) | 0.00305 ± 5.2915e-05 (W 7.46324e-05) | 🔵 colmat 2.73× faster than `faer` | 🔵 colmat 2.77× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00246 ± 2.51661e-05 (W 2.88097e-05) | 0.00761 ± 2.3094e-05 (W 0.000118504) | 0.00651 ± 6.69104e-05 (W 0.000209493) | 🔵 colmat 3.09× faster than `faer` | 🔵 colmat 2.65× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.00596 ± 5.50757e-05 (W 3.50714e-05) | 0.014639 ± 6.09945e-05 (W 0.000196434) | 0.01531 ± 4.58258e-05 (W 0.000165385) | 🔵 colmat 2.46× faster than `faer` | 🔵 colmat 2.57× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## gemm-mid, f64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.047299 ± 0.000136654 (W 0.000275977) | 0.047589 ± 1.70392e-05 (W 0.000109614) | 0.110596 ± 7.7861e-05 (W 0.000194003) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.159115 ± 0.000591916 (W 0.00123732) | 0.159185 ± 0.000553572 (W 0.000275353) | 0.320078 ± 0.000863452 (W 0.00163984) | · | 🔵 colmat 2.01× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.376186 ± 0.00265227 (W 0.00154464) | 0.380097 ± 0.00103241 (W 0.00159205) | 0.798521 ± 0.00173365 (W 0.00299148) | · | 🔵 colmat 2.12× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.24385 ± 0.0040354 (W 0.00302794) | 1.24304 ± 0.00663711 (W 0.00159389) | 2.46603 ± 0.000860831 (W 0.00274552) | · | 🔵 colmat 1.98× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.9285 ± 0.0145514 (W 0.00212266) | 2.92644 ± 0.00732161 (W 0.00146802) | 6.9562 ± 0.189232 (W 0.00622334) | · | 🔵 colmat 2.38× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 10.045 ± 0.236619 (W 0.0117259) | 10.0282 ± 0.00606923 (W 0.0151072) | 19.802 ± 0.146055 (W 0.0136422) | · | 🔵 colmat 1.97× faster than `lapack-mkl` |  |

## gemm-mid, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.013819 ± 0.000121533 (W 9.24029e-05) | 0.049199 ± 0.00398693 (W 0.00924134) | 0.036359 ± 0.000402658 (W 0.000205668) | 🔵 colmat 3.56× faster than `faer` | 🔵 colmat 2.63× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.041819 ± 0.000102632 (W 9.46689e-05) | 0.077927 ± 0.00233745 (W 0.00198664) | 0.090397 ± 0.000379026 (W 0.00115138) | 🔵 colmat 1.86× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.096967 ± 0.000524825 (W 0.000204676) | 0.120098 ± 0.0090009 (W 0.0138627) | 0.230282 ± 0.00119847 (W 0.00124656) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.322448 ± 0.00974794 (W 0.00145573) | 0.325979 ± 0.00161155 (W 0.00526384) | 0.666046 ± 0.000477119 (W 0.00246139) | · | 🔵 colmat 2.07× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 0.779114 ± 0.00121355 (W 0.00354282) | 0.778802 ± 0.00239772 (W 0.0024383) | 2.33406 ± 0.0291023 (W 0.00656881) | · | 🔵 colmat 3.00× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 2.58588 ± 0.0102673 (W 0.00852076) | 2.59291 ± 0.00547411 (W 0.00666493) | 5.09437 ± 0.0150555 (W 0.0156216) | · | 🔵 colmat 1.97× faster than `lapack-mkl` |  |

## gemm-mid, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00718 ± 3.21455e-05 (W 2.30478e-05) | 0.067839 ± 0.000624108 (W 0.015905) | 0.020689 ± 0.000107858 (W 8.91302e-05) | 🔵 colmat 9.45× faster than `faer` | 🔵 colmat 2.88× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.020849 ± 4.3589e-05 (W 3.49027e-05) | 0.120076 ± 0.00341954 (W 0.0191565) | 0.049468 ± 0.00019713 (W 0.000104547) | 🔵 colmat 5.76× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.051688 ± 0.000348884 (W 0.000200923) | 0.148674 ± 0.0300849 (W 0.0296689) | 0.121458 ± 0.000390437 (W 0.00148993) | 🔵 colmat 2.88× faster than `faer` | 🔵 colmat 2.35× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.159477 ± 0.000155653 (W 0.00167657) | 0.238662 ± 0.00309978 (W 0.00618746) | 0.344874 ± 0.000546617 (W 0.00200767) | 🔵 colmat 1.50× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.415255 ± 0.00267789 (W 0.00525743) | 0.469784 ± 0.000855873 (W 0.0142211) | 1.20001 ± 0.0120355 (W 0.00586444) | 🔵 colmat 1.13× faster than `faer` | 🔵 colmat 2.89× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 1.33346 ± 0.00252613 (W 0.0037119) | 1.34341 ± 0.00542345 (W 0.00582066) | 2.66311 ± 0.00549133 (W 0.0202683) | · | 🔵 colmat 2.00× faster than `lapack-mkl` |  |

## gemm-mid, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.04745 ± 0.000150957 (W 0.000150603) | 0.047289 ± 0.000153189 (W 0.000132964) | 0.111026 ± 0.000377813 (W 0.00123197) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.158895 ± 0.000222361 (W 0.00129162) | 0.158415 ± 0.0005315 (W 0.00131406) | 0.319869 ± 0.00172331 (W 0.00112681) | · | 🔵 colmat 2.01× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.381607 ± 0.00582672 (W 0.00169975) | 0.375286 ± 0.00341171 (W 0.00179943) | 0.796791 ± 0.00268471 (W 0.00163865) | · | 🔵 colmat 2.09× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.24408 ± 0.00165288 (W 0.0018833) | 1.24177 ± 0.00286884 (W 0.00177571) | 2.46465 ± 0.00195382 (W 0.00299768) | · | 🔵 colmat 1.98× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.91782 ± 0.00353218 (W 0.00306356) | 2.92887 ± 0.00824472 (W 0.00163929) | 6.85849 ± 0.213674 (W 0.0164801) | · | 🔵 colmat 2.35× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 10.0024 ± 0.0474373 (W 0.00726937) | 10.0049 ± 0.0302714 (W 0.00656055) | 19.9292 ± 0.028595 (W 0.0226727) | · | 🔵 colmat 1.99× faster than `lapack-mkl` |  |

## gemm-mid, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.01369 ± 4.08697e-05 (W 6.74329e-05) | 0.03597 ± 0.0013482 (W 0.00989297) | 0.036169 ± 0.000314325 (W 0.000192827) | 🔵 colmat 2.63× faster than `faer` | 🔵 colmat 2.64× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.041799 ± 2.02567e-05 (W 0.000136565) | 0.077158 ± 0.00194201 (W 0.00204374) | 0.090737 ± 0.000315722 (W 0.000222376) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 2.17× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.097787 ± 0.000619799 (W 0.00028043) | 0.112308 ± 0.00341287 (W 0.00732557) | 0.230526 ± 0.000596211 (W 0.00157534) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 2.36× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.324259 ± 0.005026 (W 0.00133873) | 0.326834 ± 0.00306704 (W 0.00309425) | 0.665616 ± 0.00117646 (W 0.00202252) | · | 🔵 colmat 2.05× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.778924 ± 0.00162709 (W 0.00260158) | 0.779504 ± 0.00271701 (W 0.00322005) | 2.10763 ± 0.0955937 (W 0.0039838) | · | 🔵 colmat 2.71× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 2.5873 ± 0.00732046 (W 0.0053026) | 2.60547 ± 0.00149959 (W 0.00534498) | 5.11306 ± 0.0119078 (W 0.00457575) | · | 🔵 colmat 1.98× faster than `lapack-mkl` |  |

## gemm-mid, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00794 ± 3.56417e-05 (W 4.13787e-05) | 0.069809 ± 0.00118474 (W 0.0132766) | 0.020839 ± 0.000198016 (W 0.000167085) | 🔵 colmat 8.79× faster than `faer` | 🔵 colmat 2.62× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.02248 ± 9.12707e-05 (W 5.87214e-05) | 0.121846 ± 0.000426122 (W 0.0063667) | 0.049289 ± 0.000430275 (W 0.000323543) | 🔵 colmat 5.42× faster than `faer` | 🔵 colmat 2.19× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.051019 ± 2.0664e-05 (W 0.0018992) | 0.132136 ± 0.0266102 (W 0.0326225) | 0.121246 ± 0.000301067 (W 0.00212658) | 🔵 colmat 2.59× faster than `faer` | 🔵 colmat 2.38× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.165457 ± 0.00110568 (W 0.00205923) | 0.240246 ± 0.000656948 (W 0.00540465) | 0.346044 ± 0.00117929 (W 0.00209766) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.415346 ± 0.00170777 (W 0.0036505) | 0.461193 ± 0.00694977 (W 0.00832905) | 1.15498 ± 0.0345076 (W 0.0028143) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 2.78× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/768x768x768/b1` | 1.33196 ± 0.00165297 (W 0.00399304) | 1.34916 ± 0.004322 (W 0.00769205) | 2.66325 ± 0.00550556 (W 0.0066388) | · | 🔵 colmat 2.00× faster than `lapack-mkl` |  |

## gemm-mid, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.184677 ± 0.000859508 (W 0.00113918) | 0.186227 ± 0.00149847 (W 0.00134435) | 0.429975 ± 0.00508122 (W 0.00763672) | · | 🔵 colmat 2.33× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.611201 ± 0.00251145 (W 0.00185906) | 0.610951 ± 0.000623003 (W 0.0016717) | 1.41868 ± 0.00273677 (W 0.00686193) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 1.46383 ± 0.00096454 (W 0.00162223) | 1.46281 ± 0.00387736 (W 0.00155687) | 3.4009 ± 0.0101602 (W 0.0165205) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 4.8369 ± 0.00336029 (W 0.002852) | 4.84316 ± 0.0143863 (W 0.00217608) | 11.2411 ± 0.012214 (W 0.0163813) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 11.4677 ± 0.0264391 (W 0.00555596) | 11.4523 ± 0.0210969 (W 0.0138131) | 26.8334 ± 0.0500292 (W 0.0389575) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 39.4141 ± 0.0293188 (W 0.0257337) | 39.4372 ± 0.0711208 (W 0.0287032) | 90.6607 ± 0.0707018 (W 0.10234) | · | 🔵 colmat 2.30× faster than `lapack-mkl` |  |

## gemm-mid, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.046359 ± 0.000134537 (W 7.28025e-05) | 0.065638 ± 0.0065434 (W 0.00313414) | 0.155114 ± 0.000431125 (W 0.0018743) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 3.35× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.155855 ± 0.000304408 (W 0.000286205) | 0.159385 ± 0.000773719 (W 0.00160294) | 0.498603 ± 0.00188751 (W 0.00290223) | · | 🔵 colmat 3.20× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.376687 ± 0.00380084 (W 0.00137396) | 0.379204 ± 0.000379022 (W 0.00172022) | 1.20046 ± 0.00442889 (W 0.0121136) | · | 🔵 colmat 3.19× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.22753 ± 0.00301662 (W 0.00279918) | 1.22561 ± 0.00312495 (W 0.0037894) | 3.80866 ± 0.0062875 (W 0.00565984) | · | 🔵 colmat 3.10× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.9057 ± 0.00457029 (W 0.00494437) | 2.8967 ± 0.00128265 (W 0.00477863) | 9.10572 ± 0.00898644 (W 0.0100236) | · | 🔵 colmat 3.13× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 9.88906 ± 0.0315664 (W 0.0156512) | 9.90992 ± 0.0329392 (W 0.0139866) | 30.4601 ± 0.00782939 (W 0.0271549) | · | 🔵 colmat 3.08× faster than `lapack-mkl` |  |

## gemm-mid, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.02489 ± 0.000147242 (W 3.20936e-05) | 0.071017 ± 0.00377651 (W 0.00475801) | 0.063488 ± 0.000135967 (W 0.00073947) | 🔵 colmat 2.85× faster than `faer` | 🔵 colmat 2.55× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.077579 ± 0.000244017 (W 0.000283599) | 0.114796 ± 0.00121659 (W 0.00191825) | 0.197507 ± 0.00103419 (W 0.00268184) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 2.55× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.186884 ± 0.000809504 (W 0.00160435) | 0.221106 ± 0.0031105 (W 0.002634) | 0.480134 ± 0.000923657 (W 0.00208082) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 2.57× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.626529 ± 0.00334733 (W 0.00280217) | 0.626988 ± 0.0024499 (W 0.00177679) | 1.48591 ± 0.00491077 (W 0.00855764) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.48424 ± 0.00179875 (W 0.00556352) | 1.4879 ± 0.00085969 (W 0.00585824) | 3.49838 ± 0.0106823 (W 0.00840422) | · | 🔵 colmat 2.36× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 4.99772 ± 0.0275079 (W 0.00542502) | 4.97966 ± 0.0362087 (W 0.00607923) | 11.7723 ± 0.0285288 (W 0.0156934) | · | 🔵 colmat 2.36× faster than `lapack-mkl` |  |

## gemm-mid, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.184773 ± 0.0003612 (W 0.00140169) | 0.185577 ± 0.000935069 (W 0.00119624) | 0.428864 ± 0.00331589 (W 0.00655263) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.612791 ± 0.00244572 (W 0.001682) | 0.610768 ± 0.00158147 (W 0.00179534) | 1.41874 ± 0.00160738 (W 0.0106661) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 1.45684 ± 0.00569502 (W 0.0014681) | 1.45719 ± 0.00592046 (W 0.00165592) | 3.3802 ± 0.00924308 (W 0.011989) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 4.84109 ± 0.0269438 (W 0.00293836) | 4.85916 ± 0.0235403 (W 0.00427091) | 11.255 ± 0.00712469 (W 0.0189736) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 11.491 ± 0.0240993 (W 0.00632031) | 11.4952 ± 0.0323331 (W 0.00998906) | 26.8829 ± 0.0783763 (W 0.0536602) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 39.4489 ± 0.0818126 (W 0.0142046) | 39.6179 ± 0.157655 (W 0.039841) | 90.6293 ± 0.0597632 (W 0.0557913) | · | 🔵 colmat 2.30× faster than `lapack-mkl` |  |

## gemm-mid, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.046549 ± 0.00011632 (W 7.39277e-05) | 0.065538 ± 0.000294041 (W 0.00136161) | 0.156145 ± 0.00081631 (W 0.00132245) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 3.35× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.155175 ± 0.000916397 (W 0.001015) | 0.158377 ± 0.000779077 (W 0.00167917) | 0.494982 ± 0.00199144 (W 0.00384394) | · | 🔵 colmat 3.19× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.381616 ± 0.00656633 (W 0.00178117) | 0.379514 ± 0.00082381 (W 0.00138453) | 1.21427 ± 0.0127417 (W 0.00331224) | · | 🔵 colmat 3.18× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 1.22451 ± 0.00248156 (W 0.00240248) | 1.22462 ± 0.00480156 (W 0.00338313) | 3.81237 ± 0.00434821 (W 0.0126015) | · | 🔵 colmat 3.11× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 2.88896 ± 0.0101758 (W 0.00300379) | 2.91231 ± 0.0110059 (W 0.0039942) | 9.08869 ± 0.00667814 (W 0.0196724) | · | 🔵 colmat 3.15× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 9.93311 ± 0.0101162 (W 0.00911808) | 9.89986 ± 0.00586556 (W 0.00659532) | 30.4373 ± 0.0610954 (W 0.0293256) | · | 🔵 colmat 3.06× faster than `lapack-mkl` |  |

## gemm-mid, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.024139 ± 5.1316e-05 (W 9.06642e-05) | 0.067248 ± 0.00360047 (W 0.00563067) | 0.062838 ± 0.000372667 (W 0.000658916) | 🔵 colmat 2.79× faster than `faer` | 🔵 colmat 2.60× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.079489 ± 0.000191471 (W 0.000251423) | 0.115147 ± 0.000479313 (W 0.00318341) | 0.196703 ± 0.00136654 (W 0.000955107) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 2.47× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.194693 ± 0.000304843 (W 0.00171321) | 0.215733 ± 0.00373769 (W 0.00527535) | 0.474004 ± 0.00212781 (W 0.00288438) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 2.43× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.626119 ± 0.00304255 (W 0.0020057) | 0.627677 ± 0.00292913 (W 0.00243481) | 1.48101 ± 0.00209481 (W 0.00813109) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.48317 ± 0.00149572 (W 0.00598062) | 1.48707 ± 0.00296826 (W 0.00280605) | 3.50628 ± 0.0111921 (W 0.00583691) | · | 🔵 colmat 2.36× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 4.9821 ± 0.0028114 (W 0.00756704) | 4.99504 ± 0.00764549 (W 0.00976747) | 11.8113 ± 0.0375782 (W 0.0119262) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |

## gemm-mid, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.024149 ± 0.000153731 (W 0.000149014) | 0.024219 ± 3.09892e-05 (W 0.000175699) | 0.054118 ± 0.000630715 (W 7.61925e-05) | · | 🔵 colmat 2.24× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.077409 ± 0.00013428 (W 9.92834e-05) | 0.077137 ± 0.000124263 (W 0.000107238) | 0.180848 ± 0.001783 (W 0.00190825) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.184853 ± 0.00136501 (W 0.00101066) | 0.184644 ± 0.000828657 (W 0.00125552) | 0.427895 ± 0.00276973 (W 0.00349333) | · | 🔵 colmat 2.31× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.621468 ± 0.00219289 (W 0.00228564) | 0.616069 ± 0.00191807 (W 0.00181704) | 1.42636 ± 0.00503881 (W 0.00783429) | · | 🔵 colmat 2.30× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.44981 ± 0.00116397 (W 0.00219684) | 1.45037 ± 0.00727226 (W 0.00215569) | 3.36212 ± 0.0332893 (W 0.00702346) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 4.97854 ± 0.171141 (W 0.00963182) | 4.98879 ± 0.0528839 (W 0.00190479) | 11.2337 ± 0.0294198 (W 0.0122478) | · | 🔵 colmat 2.26× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-mid, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00642 ± 1.0504e-05 (W 3.34858e-05) | 0.035589 ± 0.003719 (W 0.00458318) | 0.024009 ± 0.000218747 (W 0.0001903) | 🔵 colmat 5.54× faster than `faer` | 🔵 colmat 3.74× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.02073 ± 6.89952e-05 (W 5.30961e-05) | 0.071579 ± 0.0246205 (W 0.0237992) | 0.066258 ± 0.000599122 (W 0.000953652) | 🔵 colmat 3.45× faster than `faer` | 🔵 colmat 3.20× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/256x256x256/b1` | 0.050299 ± 9.58662e-05 (W 0.00025609) | 0.122375 ± 0.00790985 (W 0.010185) | 0.169687 ± 0.000131465 (W 0.00136751) | 🔵 colmat 2.43× faster than `faer` | 🔵 colmat 3.37× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.157014 ± 0.00306262 (W 0.000343875) | 0.228117 ± 0.00184589 (W 0.00654863) | 0.494832 ± 0.00214008 (W 0.00208797) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 3.15× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.371253 ± 0.00502529 (W 0.00124721) | 0.434442 ± 0.00422604 (W 0.0143924) | 1.20543 ± 0.00188794 (W 0.00395643) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 3.25× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 1.28431 ± 0.00498993 (W 0.00479764) | 1.28282 ± 0.00334738 (W 0.00518247) | 3.82479 ± 0.0110623 (W 0.0094132) | · | 🔵 colmat 2.98× faster than `lapack-mkl` |  |

## gemm-mid, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00438 ± 6.55744e-05 (W 4.54885e-05) | 0.04493 ± 0.00496892 (W 0.00477954) | 0.01114 ± 0.000100664 (W 0.000140443) | 🔵 colmat 10.26× faster than `faer` | 🔵 colmat 2.54× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.011409 ± 0.000130104 (W 5.44628e-05) | 0.158637 ± 0.00431797 (W 0.0316393) | 0.02883 ± 0.000279887 (W 0.000231474) | 🔵 colmat 13.90× faster than `faer` | 🔵 colmat 2.53× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/256x256x256/b1` | 0.025049 ± 0.000112073 (W 4.39318e-05) | 0.111976 ± 0.0593456 (W 0.0564562) | 0.063039 ± 0.000565252 (W 0.00120473) | 🔵 colmat 4.47× faster than `faer` (marginal: spread 0.68 of the difference) | 🔵 colmat 2.52× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.079039 ± 4.64866e-05 (W 0.000253515) | 0.270971 ± 0.0416318 (W 0.0485096) | 0.195093 ± 0.000528153 (W 0.00271132) | 🔵 colmat 3.43× faster than `faer` | 🔵 colmat 2.47× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/512x512x512/b1` | 0.196117 ± 0.000542968 (W 0.00114783) | 0.388517 ± 0.00714883 (W 0.0520601) | 0.464544 ± 0.011068 (W 0.0046283) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 0.681247 ± 0.00118963 (W 0.00310637) | 0.798514 ± 0.000641158 (W 0.00676782) | 1.48096 ± 0.00229588 (W 0.00380709) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 2.17× faster than `lapack-mkl` |  |

## gemm-mid, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.024379 ± 0.000109119 (W 0.000144469) | 0.02417 ± 4.33628e-05 (W 0.000192588) | 0.054589 ± 0.000140571 (W 0.000501119) | · | 🔵 colmat 2.24× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.077327 ± 0.00045961 (W 0.00010481) | 0.076848 ± 0.000127071 (W 7.01014e-05) | 0.179704 ± 0.000388402 (W 0.00177409) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.184294 ± 0.000702677 (W 0.00120732) | 0.184253 ± 0.000276244 (W 0.00121495) | 0.429244 ± 0.000874003 (W 0.00295568) | · | 🔵 colmat 2.33× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.615701 ± 0.00238114 (W 0.00169197) | 0.614141 ± 0.00176639 (W 0.00157899) | 1.42996 ± 0.00881395 (W 0.00865569) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.45297 ± 0.00283155 (W 0.00239174) | 1.44926 ± 0.00195617 (W 0.00210559) | 3.36154 ± 0.012407 (W 0.0150747) | · | 🔵 colmat 2.31× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 4.9564 ± 0.0466623 (W 0.0055413) | 4.94058 ± 0.0271208 (W 0.00424247) | 11.2201 ± 0.0300753 (W 0.0157712) | · | 🔵 colmat 2.26× faster than `lapack-mkl` |  |

## gemm-mid, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.006469 ± 3.95011e-05 (W 2.59461e-05) | 0.044839 ± 0.00481292 (W 0.00394056) | 0.02359 ± 0.000192398 (W 0.00024663) | 🔵 colmat 6.93× faster than `faer` | 🔵 colmat 3.65× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.020819 ± 8.2282e-05 (W 5.70026e-05) | 0.113878 ± 0.000327172 (W 0.0245974) | 0.067067 ± 0.000545976 (W 0.00108164) | 🔵 colmat 5.47× faster than `faer` | 🔵 colmat 3.22× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.050288 ± 5.0797e-05 (W 0.000246322) | 0.116868 ± 0.00852361 (W 0.0144303) | 0.167287 ± 0.000855388 (W 0.00164795) | 🔵 colmat 2.32× faster than `faer` | 🔵 colmat 3.33× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.156384 ± 0.000342519 (W 0.000253265) | 0.225456 ± 0.000470249 (W 0.00604044) | 0.496361 ± 0.00108423 (W 0.00238176) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 3.17× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.372284 ± 0.00190485 (W 0.00219098) | 0.432616 ± 0.0111241 (W 0.0148211) | 1.20607 ± 0.00484657 (W 0.00346834) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 3.24× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 1.28179 ± 0.00518943 (W 0.00602952) | 1.28369 ± 0.00471239 (W 0.00582958) | 3.81946 ± 0.0142651 (W 0.0070393) | · | 🔵 colmat 2.98× faster than `lapack-mkl` |  |

## gemm-mid, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.00405 ± 1.1547e-05 (W 4.6013e-05) | 0.045859 ± 0.000373805 (W 0.0049709) | 0.01123 ± 2.91605e-05 (W 9.54112e-05) | 🔵 colmat 11.32× faster than `faer` | 🔵 colmat 2.77× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.01133 ± 1.21244e-05 (W 3.73805e-05) | 0.157955 ± 0.0424865 (W 0.040272) | 0.028699 ± 0.00025658 (W 0.000286932) | 🔵 colmat 13.94× faster than `faer` | 🔵 colmat 2.53× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/256x256x256/b1` | 0.02669 ± 4.0129e-05 (W 9.65401e-05) | 0.172204 ± 0.0572232 (W 0.0628197) | 0.062698 ± 9.50807e-05 (W 0.000392859) | 🔵 colmat 6.45× faster than `faer` | 🔵 colmat 2.35× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/384x384x384/b1` | 0.081008 ± 0.000497876 (W 0.000186949) | 0.28116 ± 0.0127839 (W 0.0497917) | 0.195703 ± 0.000140522 (W 0.00187064) | 🔵 colmat 3.47× faster than `faer` | 🔵 colmat 2.42× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/512x512x512/b1` | 0.193714 ± 0.00218112 (W 0.00136716) | 0.394017 ± 0.0649665 (W 0.0612347) | 0.459792 ± 0.00167825 (W 0.00214347) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/768x768x768/b1` | 0.681937 ± 0.00242573 (W 0.00326298) | 0.792753 ± 0.00175492 (W 0.00727209) | 1.48077 ± 0.00101428 (W 0.00360147) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 2.17× faster than `lapack-mkl` |  |

## gemm-mid, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.090407 ± 6.40937e-05 (W 4.0721e-05) | 0.090347 ± 0.000184069 (W 0.000952273) | 0.210042 ± 0.000941085 (W 0.0028957) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.303209 ± 0.000627647 (W 0.00170664) | 0.302989 ± 0.00112275 (W 0.00136964) | 0.710814 ± 0.00519223 (W 0.00492329) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.727666 ± 0.00145062 (W 0.00194191) | 0.727139 ± 0.00313384 (W 0.0015352) | 1.67809 ± 0.00474034 (W 0.0104005) | · | 🔵 colmat 2.31× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 2.41855 ± 0.00532235 (W 0.00205836) | 2.41253 ± 0.0045042 (W 0.00179965) | 5.6071 ± 0.0113446 (W 0.0324394) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 5.69667 ± 0.0106725 (W 0.00151502) | 5.69645 ± 0.00820977 (W 0.00332176) | 13.5365 ± 0.107727 (W 0.0244909) | · | 🔵 colmat 2.38× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 19.4771 ± 0.048191 (W 0.0129296) | 19.4527 ± 0.0375477 (W 0.0117753) | 44.8124 ± 0.0400759 (W 0.0888963) | · | 🔵 colmat 2.30× faster than `lapack-mkl` |  |

## gemm-mid, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.024469 ± 3.61985e-05 (W 5.7169e-05) | 0.048359 ± 0.00518673 (W 0.007119) | 0.084507 ± 0.00147474 (W 0.0015669) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 3.45× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/192x192x192/b1` | 0.076937 ± 0.000239335 (W 7.14577e-05) | 0.111158 ± 0.00102593 (W 0.00271564) | 0.249781 ± 0.00422896 (W 0.00218893) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 3.25× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.182244 ± 0.000235534 (W 0.00111498) | 0.210986 ± 0.00148466 (W 0.00312656) | 0.597089 ± 0.00286914 (W 0.00261983) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 3.28× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.611619 ± 0.00275824 (W 0.0020504) | 0.612629 ± 0.00149813 (W 0.00133439) | 1.90812 ± 0.00369279 (W 0.00878279) | · | 🔵 colmat 3.12× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.46761 ± 0.00221557 (W 0.00406109) | 1.46971 ± 0.001442 (W 0.00373266) | 4.69165 ± 0.0336549 (W 0.00479323) | · | 🔵 colmat 3.20× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 4.9211 ± 0.00572974 (W 0.00747964) | 4.91273 ± 0.0025532 (W 0.00852518) | 15.0739 ± 0.011709 (W 0.0269546) | · | 🔵 colmat 3.06× faster than `lapack-mkl` |  |

## gemm-mid, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.012469 ± 2.0502e-05 (W 3.66511e-05) | 0.074709 ± 0.000880092 (W 0.0155076) | 0.032229 ± 0.033965 (W 0.000358985) | 🔵 colmat 5.99× faster than `faer` | · | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.039199 ± 0.000106039 (W 5.51289e-05) | 0.135055 ± 0.00689084 (W 0.0199533) | 0.099547 ± 0.000910786 (W 0.00112843) | 🔵 colmat 3.45× faster than `faer` | 🔵 colmat 2.54× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.095637 ± 0.000202092 (W 0.000173748) | 0.167294 ± 0.00496773 (W 0.0147723) | 0.228483 ± 0.00105176 (W 0.0019026) | 🔵 colmat 1.75× faster than `faer` | 🔵 colmat 2.39× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.305694 ± 0.0001495 (W 0.00193224) | 0.376153 ± 0.00198712 (W 0.00264404) | 0.741763 ± 0.00187103 (W 0.00483034) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 2.43× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.757395 ± 0.000240529 (W 0.00207616) | 0.778014 ± 0.00269289 (W 0.00512178) | 1.7766 ± 0.0145368 (W 0.00439466) | · | 🔵 colmat 2.35× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 2.48559 ± 0.0363255 (W 0.00682533) | 2.485 ± 0.00336024 (W 0.0047098) | 5.79794 ± 0.00611632 (W 0.0193799) | · | 🔵 colmat 2.33× faster than `lapack-mkl` |  |

## gemm-mid, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.090469 ± 0.000331162 (W 6.3405e-05) | 0.090317 ± 0.000147336 (W 4.85829e-05) | 0.214583 ± 0.00250032 (W 0.00558454) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `gemm/192x192x192/b1` | 0.303615 ± 0.000111715 (W 0.00189972) | 0.30297 ± 0.00068884 (W 0.00197875) | 0.711845 ± 0.00278466 (W 0.00371365) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.727105 ± 0.00409365 (W 0.00214146) | 0.726559 ± 0.00123139 (W 0.00169445) | 1.67687 ± 0.00609057 (W 0.00861863) | · | 🔵 colmat 2.31× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 2.41646 ± 0.0026646 (W 0.00213981) | 2.41609 ± 0.000506812 (W 0.00205685) | 5.60669 ± 0.011663 (W 0.0159315) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 5.69956 ± 0.00948946 (W 0.00255116) | 5.69601 ± 0.00730159 (W 0.00673938) | 13.5093 ± 0.0163027 (W 0.0588421) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 19.4824 ± 0.0231508 (W 0.0149131) | 19.563 ± 0.0853308 (W 0.0220464) | 44.7955 ± 0.0224562 (W 0.0388093) | · | 🔵 colmat 2.30× faster than `lapack-mkl` |  |

## gemm-mid, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.024479 ± 0.000100848 (W 6.05995e-05) | 0.053839 ± 0.00214064 (W 0.00923298) | 0.084537 ± 0.000398418 (W 0.00117965) | 🔵 colmat 2.20× faster than `faer` | 🔵 colmat 3.45× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.076857 ± 0.000328087 (W 0.000131812) | 0.109388 ± 0.000796746 (W 0.00380673) | 0.249672 ± 0.00297752 (W 0.00270004) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 3.25× faster than `lapack-mkl` |  |
| `gemm/256x256x256/b1` | 0.181254 ± 0.000416589 (W 0.00125273) | 0.210366 ± 0.000918143 (W 0.00640918) | 0.5934 ± 0.00144131 (W 0.0035675) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 3.27× faster than `lapack-mkl` |  |
| `gemm/384x384x384/b1` | 0.617528 ± 0.00536818 (W 0.00239554) | 0.613318 ± 0.0017499 (W 0.0017685) | 1.90328 ± 0.00735584 (W 0.00484337) | · | 🔵 colmat 3.08× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 1.47061 ± 0.00146639 (W 0.00351586) | 1.46719 ± 0.00102318 (W 0.0035993) | 4.64549 ± 0.0117925 (W 0.0241958) | · | 🔵 colmat 3.16× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 4.92182 ± 0.00308534 (W 0.00694875) | 4.90567 ± 0.0090901 (W 0.00472877) | 15.065 ± 0.00581381 (W 0.0219408) | · | 🔵 colmat 3.06× faster than `lapack-mkl` |  |

## gemm-mid, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.013069 ± 4.16333e-05 (W 4.1392e-05) | 0.074949 ± 0.0163628 (W 0.00198055) | 0.032319 ± 0.000326241 (W 0.000531588) | 🔵 colmat 5.73× faster than `faer` | 🔵 colmat 2.47× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `gemm/192x192x192/b1` | 0.039448 ± 4.99032e-05 (W 6.77518e-05) | 0.133877 ± 0.00546497 (W 0.00855353) | 0.099778 ± 0.000994291 (W 0.000816738) | 🔵 colmat 3.39× faster than `faer` | 🔵 colmat 2.53× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/256x256x256/b1` | 0.091947 ± 0.00134039 (W 6.03589e-05) | 0.170327 ± 0.00938145 (W 0.0133364) | 0.230352 ± 0.00148072 (W 0.00165861) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 2.51× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 0.310679 ± 0.00114086 (W 0.00238698) | 0.375233 ± 0.00218581 (W 0.00446282) | 0.743924 ± 0.000380633 (W 0.00465462) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 2.39× faster than `lapack-mkl` |  |
| `gemm/512x512x512/b1` | 0.757887 ± 0.00254361 (W 0.00203923) | 0.784584 ± 0.00514413 (W 0.00621298) | 1.77831 ± 0.00423828 (W 0.00214769) | · | 🔵 colmat 2.35× faster than `lapack-mkl` |  |
| `gemm/768x768x768/b1` | 2.48695 ± 0.000325572 (W 0.00282078) | 2.48473 ± 0.000387493 (W 0.00588398) | 5.79858 ± 0.0152496 (W 0.0141312) | · | 🔵 colmat 2.33× faster than `lapack-mkl` |  |

## gemm-tall, f64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.530722 ± 0.00375399 (W 0.00239572) | 0.528811 ± 0.00333383 (W 0.00272932) | 0.91088 ± 0.000438847 (W 0.00180493) | · | 🔵 colmat 1.72× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.497011 ± 0.0043116 (W 0.00362715) | 0.644239 ± 0.0127196 (W 0.00618976) | 1.12251 ± 0.00152486 (W 0.00613003) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 2.26× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.383102 ± 0.00703492 (W 0.00169441) | 0.381408 ± 0.00381803 (W 0.00128881) | 1.00397 ± 0.00397072 (W 0.00108028) | · | 🔵 colmat 2.62× faster than `lapack-mkl` |  |

## gemm-tall, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.132807 ± 0.000187609 (W 0.00052317) | 0.185574 ± 0.000333686 (W 0.00196172) | 0.252175 ± 0.00268531 (W 0.00144038) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 1.90× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.145387 ± 0.0106395 (W 0.00185896) | 0.216683 ± 0.000269623 (W 0.00179569) | 0.336689 ± 0.000291462 (W 0.00202625) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 2.32× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.097538 ± 0.00176127 (W 0.00022354) | 0.120776 ± 0.00113417 (W 0.00159837) | 0.258235 ± 0.000582644 (W 0.000813984) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 2.65× faster than `lapack-mkl` |  |

## gemm-tall, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.068028 ± 0.000304056 (W 0.000325835) | 0.187984 ± 0.00135778 (W 0.00153258) | 0.161614 ± 0.000329731 (W 0.0012217) | 🔵 colmat 2.76× faster than `faer` | 🔵 colmat 2.38× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.137636 ± 0.000486009 (W 0.0226333) | 0.141915 ± 0.000775461 (W 0.00236707) | 0.244475 ± 0.000399029 (W 0.0212439) | · | 🔵 colmat 1.78× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.049429 ± 0.000140119 (W 0.000179664) | 0.094407 ± 0.00326313 (W 0.00223603) | 0.137045 ± 0.000365898 (W 0.00143134) | 🔵 colmat 1.91× faster than `faer` | 🔵 colmat 2.77× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.353358 ± 0.000691381 (W 0.00167447) | 0.353337 ± 0.00162902 (W 0.00190385) | 1.5828 ± 0.00317983 (W 0.00257758) | · | 🔵 colmat 4.48× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.50796 ± 0.00181995 (W 0.00472075) | 0.524432 ± 0.00178297 (W 0.0086394) | 1.14224 ± 0.0163614 (W 0.0183107) | · | 🔵 colmat 2.25× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.411192 ± 0.00157126 (W 0.00186516) | 0.411297 ± 0.00104294 (W 0.00158273) | 0.827745 ± 0.00258633 (W 0.00177203) | · | 🔵 colmat 2.01× faster than `lapack-mkl` |  |

## gemm-tall, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.095057 ± 0.000464184 (W 0.000308657) | 0.407392 ± 0.0024814 (W 0.00561277) | 0.397997 ± 0.00180782 (W 0.00173116) | 🔵 colmat 4.29× faster than `faer` | 🔵 colmat 4.19× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.144237 ± 0.000184014 (W 0.00333231) | 0.196816 ± 0.000446628 (W 0.00819325) | 0.339493 ± 0.00109227 (W 0.00146322) | 🔵 colmat 1.36× faster than `faer` | 🔵 colmat 2.35× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.105716 ± 0.000572388 (W 0.000342535) | 0.118618 ± 0.000718645 (W 0.00157224) | 0.225403 ± 0.00165785 (W 0.00117564) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 2.13× faster than `lapack-mkl` |  |

## gemm-tall, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.052078 ± 0.000743606 (W 0.000168479) | 0.759365 ± 0.0156174 (W 0.0112532) | 0.220863 ± 0.00256414 (W 0.00249589) | 🔵 colmat 14.58× faster than `faer` | 🔵 colmat 4.24× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.138757 ± 0.00145602 (W 0.00618474) | 0.165954 ± 0.0142194 (W 0.00324467) | 0.247712 ± 0.000371582 (W 0.00260897) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.052309 ± 0.000342924 (W 0.000386937) | 0.074729 ± 0.000283598 (W 0.000224003) | 0.117577 ± 0.000810953 (W 0.00066447) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 2.25× faster than `lapack-mkl` |  |

## gemm-tall, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.51885 ± 0.019782 (W 0.00378439) | 1.52083 ± 0.0174181 (W 0.00634424) | 3.65726 ± 0.0242257 (W 0.00756081) | · | 🔵 colmat 2.41× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 1.59933 ± 0.00261248 (W 0.00424947) | 1.59918 ± 0.0339827 (W 0.00439977) | 3.90248 ± 0.00437985 (W 0.0100629) | · | 🔵 colmat 2.44× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 1.47894 ± 0.00556126 (W 0.00132723) | 1.49998 ± 0.0100129 (W 0.00470595) | 3.42625 ± 0.00372629 (W 0.0123525) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |

## gemm-tall, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.387778 ± 0.00259087 (W 0.0018222) | 0.554732 ± 0.00195385 (W 0.00197488) | 1.2268 ± 0.00264504 (W 0.00359817) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 3.16× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.409796 ± 0.00366748 (W 0.00300928) | 0.488501 ± 0.00166495 (W 0.00249951) | 1.36416 ± 0.00169568 (W 0.00391313) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 3.33× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.372593 ± 0.00152069 (W 0.0015517) | 0.403566 ± 0.000616163 (W 0.00450623) | 1.34202 ± 0.00367146 (W 0.00371954) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 3.60× faster than `lapack-mkl` |  |

## gemm-tall, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.218173 ± 0.0018032 (W 0.00217565) | 0.286801 ± 0.000729318 (W 0.0018358) | 0.546892 ± 0.0102168 (W 0.00278165) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 2.51× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.31645 ± 0.000503379 (W 0.0228506) | 0.320424 ± 0.0172037 (W 0.0114885) | 0.707946 ± 0.00509689 (W 0.00204973) | · | 🔵 colmat 2.24× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.188254 ± 0.000707292 (W 0.00162923) | 0.240692 ± 0.00928706 (W 0.0135264) | 0.485561 ± 0.00190551 (W 0.00238244) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 2.58× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.34869 ± 0.00393507 (W 0.00220278) | 1.34691 ± 0.000619382 (W 0.00343754) | 3.50622 ± 0.0275602 (W 0.0103852) | · | 🔵 colmat 2.60× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 1.58997 ± 0.0158333 (W 0.00278099) | 1.58931 ± 0.00562038 (W 0.00362198) | 3.85923 ± 0.00759599 (W 0.0243069) | · | 🔵 colmat 2.43× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 1.5094 ± 0.00474346 (W 0.00198787) | 1.52633 ± 0.00832168 (W 0.00193351) | 3.64232 ± 0.0247601 (W 0.0112048) | · | 🔵 colmat 2.41× faster than `lapack-mkl` |  |

## gemm-tall, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.342978 ± 0.000273935 (W 0.00166388) | 0.98494 ± 0.00156167 (W 0.00301614) | 8.32183 ± 0.585784 (W 0.137571) | 🔵 colmat 2.87× faster than `faer` | 🔵 colmat 24.26× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.405632 ± 0.000295579 (W 0.00241247) | 0.463981 ± 0.00351476 (W 0.00264674) | 1.35772 ± 0.00858951 (W 0.009124) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 3.35× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.380903 ± 0.000773212 (W 0.00230474) | 0.397342 ± 0.00245605 (W 0.00245893) | 1.23496 ± 0.00236145 (W 0.00523254) | · | 🔵 colmat 3.24× faster than `lapack-mkl` |  |

## gemm-tall, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.187974 ± 0.000555231 (W 0.0016395) | 1.60371 ± 0.00458347 (W 0.00908954) | 2.08865 ± 0.400189 (W 0.0137459) | 🔵 colmat 8.53× faster than `faer` | 🔵 colmat 11.11× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x16x240/b1` | 0.272451 ± 0.00302889 (W 0.00649641) | 0.325359 ± 0.00144673 (W 0.00186328) | 0.699476 ± 0.00188616 (W 0.0031573) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 2.57× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.194354 ± 0.00186443 (W 0.00128661) | 0.199785 ± 0.000539248 (W 0.00157104) | 0.51511 ± 0.00403112 (W 0.00274436) | · | 🔵 colmat 2.65× faster than `lapack-mkl` |  |

## gemm-tall, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.228712 ± 0.00276611 (W 0.001712) | 0.227272 ± 0.00458587 (W 0.00222815) | 0.479264 ± 0.0011842 (W 0.00206464) | · | 🔵 colmat 2.10× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.199184 ± 0.00155174 (W 0.00117481) | 0.191804 ± 0.000931065 (W 0.00117123) | 0.441205 ± 0.00144091 (W 0.00204257) | · | 🔵 colmat 2.22× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.182706 ± 0.00132195 (W 0.00212457) | 0.184704 ± 0.000643958 (W 0.000560514) | 0.451874 ± 0.000904485 (W 0.00231744) | · | 🔵 colmat 2.47× faster than `lapack-mkl` |  |

## gemm-tall, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.063799 ± 0.000365591 (W 0.000413) | 0.176356 ± 0.000305616 (W 0.00133463) | 0.176444 ± 0.00183764 (W 0.0010405) | 🔵 colmat 2.76× faster than `faer` | 🔵 colmat 2.77× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.049058 ± 0.000193029 (W 0.000204427) | 0.081667 ± 0.00153021 (W 0.0012678) | 0.156675 ± 0.00301183 (W 0.000901693) | 🔵 colmat 1.66× faster than `faer` | 🔵 colmat 3.19× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.046159 ± 6.06658e-05 (W 0.000107122) | 0.060738 ± 0.000502394 (W 0.000935964) | 0.249412 ± 0.00100125 (W 0.00165636) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 5.40× faster than `lapack-mkl` |  |

## gemm-tall, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.036149 ± 0.000531432 (W 0.000483538) | 0.177904 ± 0.000765768 (W 0.00133156) | 0.087607 ± 0.000577847 (W 0.000592645) | 🔵 colmat 4.92× faster than `faer` | 🔵 colmat 2.42× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.025959 ± 0.000132842 (W 0.000110726) | 0.059518 ± 0.00148113 (W 0.00301948) | 0.074868 ± 0.000272213 (W 0.00069385) | 🔵 colmat 2.29× faster than `faer` | 🔵 colmat 2.88× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.023889 ± 0.000436299 (W 8.16352e-05) | 0.048069 ± 0.000530528 (W 0.00122212) | 0.080587 ± 0.000702823 (W 0.00100803) | 🔵 colmat 2.01× faster than `faer` | 🔵 colmat 3.37× faster than `lapack-mkl` |  |

## gemm-tall, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.333173 ± 0.000380501 (W 0.00177087) | 0.334904 ± 0.00215968 (W 0.00204173) | 0.513013 ± 0.0176866 (W 0.00172721) | · | 🔵 colmat 1.54× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.215753 ± 0.00211539 (W 0.00172418) | 0.215562 ± 0.00268676 (W 0.00194087) | 1.04007 ± 0.0103363 (W 0.010615) | · | 🔵 colmat 4.82× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.206456 ± 0.00296506 (W 0.00192437) | 0.204586 ± 0.000587139 (W 0.00136553) | 0.498781 ± 0.00361785 (W 0.00291394) | · | 🔵 colmat 2.42× faster than `lapack-mkl` |  |

## gemm-tall, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.084088 ± 0.000111879 (W 0.000197596) | 0.240985 ± 0.00182684 (W 0.00406397) | 0.518392 ± 0.020594 (W 0.00238143) | 🔵 colmat 2.87× faster than `faer` | 🔵 colmat 6.16× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x16x240/b1` | 0.057069 ± 0.000794991 (W 0.00017717) | 0.066999 ± 0.000474178 (W 0.00084192) | 0.363548 ± 0.00835273 (W 0.00451395) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 6.37× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.052588 ± 0.000446897 (W 0.000168612) | 0.064618 ± 0.000221518 (W 0.00106845) | 0.186606 ± 0.00633523 (W 0.00215098) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 3.55× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-tall, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.045129 ± 0.000115902 (W 8.56989e-05) | 0.386777 ± 0.00359847 (W 0.00585754) | 0.248771 ± 0.00541783 (W 0.00146741) | 🔵 colmat 8.57× faster than `faer` | 🔵 colmat 5.51× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.028999 ± 0.00128418 (W 0.000251336) | 0.047709 ± 0.000512793 (W 0.000652786) | 0.162916 ± 0.000554004 (W 0.00180408) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 5.62× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.027959 ± 0.000430589 (W 7.827e-05) | 0.039058 ± 0.000832506 (W 0.000727633) | 0.090437 ± 0.00214223 (W 0.00139956) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 3.23× faster than `lapack-mkl` |  |

## gemm-tall, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.758363 ± 0.00168995 (W 0.00209678) | 0.762606 ± 0.000928598 (W 0.00237158) | 1.71626 ± 0.00274316 (W 0.00550465) | · | 🔵 colmat 2.26× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.797645 ± 0.00540729 (W 0.00163496) | 0.808654 ± 0.00859811 (W 0.00153603) | 1.82777 ± 0.0129723 (W 0.0703502) | · | 🔵 colmat 2.29× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.766405 ± 0.012515 (W 0.00106094) | 0.763873 ± 0.0045371 (W 0.00202072) | 1.71882 ± 0.00200875 (W 0.0031951) | · | 🔵 colmat 2.24× faster than `lapack-mkl` |  |

## gemm-tall, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.207404 ± 0.000466886 (W 0.00151806) | 0.280394 ± 0.000688356 (W 0.00144624) | 0.600631 ± 0.000500292 (W 0.0019484) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 2.90× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.206033 ± 0.0115099 (W 0.00197086) | 0.260342 ± 0.000711015 (W 0.0090412) | 0.686118 ± 0.00220101 (W 0.00210095) | 🔵 colmat 1.26× faster than `faer` | 🔵 colmat 3.33× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.191913 ± 0.000374449 (W 0.000611307) | 0.214553 ± 0.00160972 (W 0.00174621) | 0.679638 ± 0.000990227 (W 0.00209283) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 3.54× faster than `lapack-mkl` |  |

## gemm-tall, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.101088 ± 0.000382336 (W 0.000253831) | 0.282351 ± 0.000350705 (W 0.0023887) | 0.273981 ± 0.000470261 (W 0.00142305) | 🔵 colmat 2.79× faster than `faer` | 🔵 colmat 2.71× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.161155 ± 0.0014116 (W 0.00212998) | 0.172654 ± 0.000373728 (W 0.00227906) | 0.366138 ± 0.000980345 (W 0.00996754) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 2.27× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.097106 ± 0.000394615 (W 0.000170499) | 0.141425 ± 0.00255589 (W 0.00441453) | 0.252911 ± 0.00104318 (W 0.00236312) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 2.60× faster than `lapack-mkl` |  |

## gemm-tall, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.792534 ± 0.0039711 (W 0.00172374) | 0.793634 ± 0.00119676 (W 0.00168426) | 1.72407 ± 0.00075689 (W 0.00333536) | · | 🔵 colmat 2.18× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.794502 ± 0.00187998 (W 0.00189569) | 0.795134 ± 0.00981452 (W 0.00160226) | 1.92465 ± 0.0220932 (W 0.0518664) | · | 🔵 colmat 2.42× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.759815 ± 0.00599377 (W 0.00312638) | 0.758624 ± 0.00431184 (W 0.00169545) | 1.81163 ± 0.0030654 (W 0.0112354) | · | 🔵 colmat 2.38× faster than `lapack-mkl` |  |

## gemm-tall, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.200523 ± 0.000460936 (W 0.00178848) | 0.522142 ± 0.00398094 (W 0.00441119) | 0.960881 ± 0.00315454 (W 0.00175553) | 🔵 colmat 2.60× faster than `faer` | 🔵 colmat 4.79× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.201963 ± 0.000263501 (W 0.00206708) | 0.232302 ± 0.00222853 (W 0.00219774) | 0.683248 ± 0.00137177 (W 0.010432) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 3.38× faster than `lapack-mkl` |  |
| `gemm/4096x64x64/b1` | 0.191984 ± 0.000643206 (W 0.00100243) | 0.202996 ± 0.000725015 (W 0.00100229) | 0.626508 ± 0.0127597 (W 0.00308267) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 3.26× faster than `lapack-mkl` |  |

## gemm-tall, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.108256 ± 0.00018472 (W 0.00189354) | 0.791705 ± 0.00414513 (W 0.0116819) | 0.489554 ± 0.00138284 (W 0.00221163) | 🔵 colmat 7.31× faster than `faer` | 🔵 colmat 4.52× faster than `lapack-mkl` |  |
| `gemm/4096x16x240/b1` | 0.141527 ± 0.00397895 (W 0.00265846) | 0.171017 ± 0.00233235 (W 0.00164394) | 0.336153 ± 0.00116625 (W 0.0152399) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 2.38× faster than `lapack-mkl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.094867 ± 0.00155487 (W 0.000216032) | 0.108416 ± 0.000201852 (W 0.00149509) | 0.2742 ± 0.00216168 (W 0.00273966) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 2.89× faster than `lapack-mkl` |  |

## gemm-wide, f64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.410256 ± 0.00131608 (W 0.0021061) | 0.420471 ± 0.00406344 (W 0.00181748) | 0.827403 ± 0.000674288 (W 0.00108742) | · | 🔵 colmat 2.02× faster than `lapack-mkl` |  |

## gemm-wide, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.105217 ± 0.00253138 (W 0.000530054) | 0.117387 ± 0.00101368 (W 0.00127435) | 0.223203 ± 0.00211888 (W 0.00122783) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 2.12× faster than `lapack-mkl` |  |

## gemm-wide, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.052269 ± 0.00022715 (W 0.000113099) | 0.074948 ± 0.000328392 (W 0.000420773) | 0.118246 ± 0.00148921 (W 0.000751246) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 2.26× faster than `lapack-mkl` |  |

## gemm-wide, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.381567 ± 0.0024671 (W 0.00143731) | 0.381722 ± 0.000478339 (W 0.00170968) | 1.0027 ± 0.00907364 (W 0.00164972) | · | 🔵 colmat 2.63× faster than `lapack-mkl` |  |

## gemm-wide, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.097738 ± 0.000888362 (W 0.000349481) | 0.120805 ± 0.000249096 (W 0.00128884) | 0.257785 ± 0.000896005 (W 0.0012247) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 2.64× faster than `lapack-mkl` |  |

## gemm-wide, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.049519 ± 0.000161967 (W 0.000206175) | 0.091738 ± 0.00271495 (W 0.00534018) | 0.137835 ± 0.000634446 (W 0.00186639) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 2.78× faster than `lapack-mkl` | `!` between-start range over 5% |

## gemm-wide, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.51131 ± 0.00426559 (W 0.00264195) | 1.52554 ± 0.00675188 (W 0.0027073) | 3.64248 ± 0.00480434 (W 0.00951821) | · | 🔵 colmat 2.41× faster than `lapack-mkl` |  |

## gemm-wide, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.380307 ± 0.00085085 (W 0.00161164) | 0.394232 ± 0.000720958 (W 0.00150646) | 1.23873 ± 0.00970221 (W 0.00422395) | · | 🔵 colmat 3.26× faster than `lapack-mkl` |  |

## gemm-wide, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.194504 ± 0.000628226 (W 0.00103739) | 0.200365 ± 0.000905826 (W 0.000614951) | 0.516623 ± 0.0141795 (W 0.00354391) | · | 🔵 colmat 2.66× faster than `lapack-mkl` |  |

## gemm-wide, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.47739 ± 0.00611559 (W 0.00221069) | 1.48605 ± 0.00375522 (W 0.00192349) | 3.42473 ± 0.00296591 (W 0.00670197) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |

## gemm-wide, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.375298 ± 0.00112842 (W 0.00143331) | 0.406926 ± 0.00323763 (W 0.00667024) | 1.3418 ± 0.00560926 (W 0.00463749) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 3.58× faster than `lapack-mkl` |  |

## gemm-wide, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.187614 ± 0.00132436 (W 0.00165137) | 0.241182 ± 0.00299919 (W 0.0137681) | 0.48606 ± 0.000848949 (W 0.00272369) | 🔵 colmat 1.29× faster than `faer` | 🔵 colmat 2.59× faster than `lapack-mkl` |  |

## gemm-wide, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.203684 ± 0.000792054 (W 0.00127604) | 0.204473 ± 0.000987492 (W 0.0013717) | 0.500124 ± 0.00149819 (W 0.00268759) | · | 🔵 colmat 2.46× faster than `lapack-mkl` |  |

## gemm-wide, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.051728 ± 7.22519e-05 (W 0.000270518) | 0.064927 ± 0.000822333 (W 0.00143371) | 0.179104 ± 0.00309465 (W 0.00165393) | 🔵 colmat 1.26× faster than `faer` | 🔵 colmat 3.46× faster than `lapack-mkl` |  |

## gemm-wide, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.026859 ± 0.000644231 (W 0.000125707) | 0.038439 ± 0.000231157 (W 0.0006563) | 0.086047 ± 0.000611496 (W 0.00136483) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 3.20× faster than `lapack-mkl` |  |

## gemm-wide, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.183506 ± 0.000426793 (W 0.00124736) | 0.182544 ± 0.000818418 (W 0.00152302) | 0.453136 ± 0.00114985 (W 0.00192146) | · | 🔵 colmat 2.47× faster than `lapack-mkl` |  |

## gemm-wide, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.046198 ± 0.000493566 (W 4.63681e-05) | 0.060368 ± 0.000612675 (W 0.000681279) | 0.248852 ± 0.00077616 (W 0.00157399) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 5.39× faster than `lapack-mkl` |  |

## gemm-wide, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.02393 ± 0.000352751 (W 5.72905e-05) | 0.047409 ± 0.000436567 (W 0.00127341) | 0.081118 ± 0.000779692 (W 0.00163678) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 3.39× faster than `lapack-mkl` |  |

## gemm-wide, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.752195 ± 0.00476703 (W 0.00129216) | 0.761154 ± 0.00771138 (W 0.00136323) | 1.80996 ± 0.00515411 (W 0.00527247) | · | 🔵 colmat 2.41× faster than `lapack-mkl` |  |

## gemm-wide, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.193123 ± 0.00262364 (W 0.000288531) | 0.203473 ± 0.000226987 (W 0.00134647) | 0.614257 ± 0.0133906 (W 0.00440895) | 🔵 colmat 1.05× faster than `faer` | 🔵 colmat 3.18× faster than `lapack-mkl` |  |

## gemm-wide, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.094338 ± 0.000117772 (W 0.000631403) | 0.108917 ± 0.000462613 (W 0.00131632) | 0.276344 ± 0.00299716 (W 0.00217806) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 2.93× faster than `lapack-mkl` |  |

## gemm-wide, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.768265 ± 0.00545095 (W 0.00126948) | 0.764614 ± 0.000464142 (W 0.00179634) | 1.71708 ± 0.00310552 (W 0.00668185) | · | 🔵 colmat 2.24× faster than `lapack-mkl` |  |

## gemm-wide, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.193914 ± 0.000234398 (W 0.0011256) | 0.215415 ± 0.000563508 (W 0.00193581) | 0.678849 ± 0.00139207 (W 0.00337421) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 3.50× faster than `lapack-mkl` |  |

## gemm-wide, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.097138 ± 0.00107002 (W 0.00142841) | 0.140687 ± 0.001763 (W 0.00516919) | 0.253192 ± 0.000655155 (W 0.00145169) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 2.61× faster than `lapack-mkl` |  |

## linalg-small, f64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0023 ± 0.000138924 (W 2.2676e-05) | 0.0041 ± 1.73205e-05 (W 1.41421e-05) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00231 ± 0.000103923 (W 3.53553e-05) | 0.00229 ± 0 (W 2.61381e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00783 ± 2.08167e-05 (W 4.80333e-05) | 0.030889 ± 0.000405257 (W 2.60231e-05) | - | 🔵 colmat 3.94× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0077 ± 3.21455e-05 (W 4.0598e-05) | 0.011009 ± 3.4641e-05 (W 4.94601e-05) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.01713 ± 1.18462e-05 (W 9.72435e-05) | 0.100737 ± 1.0504e-05 (W 0.000997486) | - | 🔵 colmat 5.88× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.016789 ± 2.0664e-05 (W 7.5712e-05) | 0.019979 ± 6.05007e-05 (W 7.35003e-05) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.067167 ± 0.000157991 (W 9.24786e-05) | 0.071317 ± 7.22519e-05 (W 0.000863882) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.067739 ± 4.50592e-05 (W 0.00101677) | 0.071579 ± 0.000169954 (W 0.000130274) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.279522 ± 0.000168589 (W 0.00102701) | 0.348529 ± 0.00106528 (W 0.00109527) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.280541 ± 0.00054233 (W 0.00130208) | 0.350273 ± 0.000742011 (W 0.00116921) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.677865 ± 0.00136699 (W 0.00149699) | 0.818022 ± 0.00161635 (W 0.00132875) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.689217 ± 0.00109708 (W 0.00153583) | 0.833002 ± 0.00134353 (W 0.00171671) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.007809 ± 0.000134389 (W 8.54476e-05) | 0.00595 ± 9.07377e-05 (W 2.24989e-05) | - | 🔴 colmat 1.31× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.009999 ± 0.000101737 (W 4.75005e-05) | 0.00707 ± 0.000133929 (W 3.91063e-05) | - | 🔴 colmat 1.41× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.032619 ± 0.000284488 (W 0.000139356) | 0.02783 ± 0.000395816 (W 5.13147e-05) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.038569 ± 0.000161967 (W 0.000176525) | 0.030689 ± 0.000112117 (W 9.78264e-05) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.086367 ± 0.00048947 (W 0.000178194) | 0.079278 ± 0.000305511 (W 9.88398e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.103837 ± 0.000271423 (W 0.00149502) | 0.089668 ± 0.000127241 (W 0.00148227) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00453 ± 1.52753e-05 (W 4.79917e-05) | 0.00445 ± 1.52753e-05 (W 3.56371e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00456 ± 2.04206e-05 (W 2.3816e-05) | 0.00443 ± 0.00010116 (W 2.16795e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.01933 ± 6.89952e-05 (W 0.000107291) | 0.01928 ± 0.000159478 (W 4.88395e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01926 ± 7.44782e-05 (W 4.44218e-05) | 0.019059 ± 6.45781e-05 (W 3.95626e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.053628 ± 0.00018155 (W 0.000151638) | 0.052738 ± 0.000145535 (W 0.000123605) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.053629 ± 0.0005055 (W 0.000968428) | 0.051999 ± 0.000126716 (W 0.000127934) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00432 ± 4.04145e-05 (W 2.1312e-05) | 0.00525 ± 1.73205e-05 (W 2.58882e-05) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00552 ± 6.1101e-05 (W 7.68193e-05) | 0.0051 ± 2.08167e-05 (W 0.000151384) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01463 ± 4.50925e-05 (W 0.000102471) | 0.022039 ± 7.55447e-05 (W 5.04896e-05) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.018279 ± 0.000162583 (W 0.000137169) | 0.01707 ± 8.01686e-05 (W 0.0001521) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.032289 ± 0.000205183 (W 7.12039e-05) | 0.052198 ± 0.000722648 (W 0.000443663) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.045039 ± 0.000209834 (W 0.0014156) | 0.041589 ± 1.78979e-05 (W 0.00142871) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.02041 ± 0.00033892 (W 0.000129353) | 0.02648 ± 6.26126e-05 (W 7.81172e-05) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.01353 ± 6.35085e-06 (W 5.12182e-05) | 0.021449 ± 8.14453e-05 (W 0.000156896) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.065997 ± 0.000225855 (W 0.000127806) | 0.108916 ± 9.02238e-05 (W 0.00121104) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.048698 ± 0.000154533 (W 0.00029029) | 0.074428 ± 0.000210072 (W 0.00110465) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.132857 ± 0.000321747 (W 0.000223428) | 0.233215 ± 0.000168512 (W 0.00163862) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.134636 ± 0.00139986 (W 0.00189831) | 0.181656 ± 0.00100188 (W 0.00169316) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0124 ± 0.000255226 (W 6.96218e-05) | 0.018829 ± 9.53939e-05 (W 6.00275e-05) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.042059 ± 0.000170392 (W 0.000958538) | 0.084777 ± 0.000592368 (W 0.001113) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.080687 ± 3.9879e-05 (W 9.8297e-05) | 0.180904 ± 0.000180632 (W 0.000916717) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00448 ± 3.56417e-05 (W 1.83902e-05) | 0.0057 ± 4.30155e-05 (W 3.97643e-05) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00621 ± 2.64575e-05 (W 3.78153e-05) | 0.00545 ± 0.000117189 (W 0.000102094) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.013769 ± 5.91636e-05 (W 4.29209e-05) | 0.02261 ± 0.000130309 (W 0.000104958) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.018919 ± 0.000121162 (W 6.30183e-05) | 0.017419 ± 0.000127555 (W 0.000102442) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.029329 ± 0.000123423 (W 0.00106708) | 0.051849 ± 0.000607056 (W 0.00010907) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.045819 ± 4.12351e-05 (W 0.000142173) | 0.041209 ± 5.55908e-05 (W 0.00118262) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.101937 ± 0.000233989 (W 0.000189729) | 0.123346 ± 0.00215027 (W 0.000876589) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.104246 ± 0.00699248 (W 0.000257851) | 0.121606 ± 0.0042346 (W 0.00103122) | - | 🔵 colmat 1.17× faster than `faer` (marginal: spread 0.65 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.423296 ± 0.00083798 (W 0.00144955) | 0.599957 ± 0.000814525 (W 0.00114401) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.422971 ± 0.00117295 (W 0.00157828) | 0.604641 ± 0.000391735 (W 0.00123107) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.967718 ± 0.000996387 (W 0.000437534) | 1.40286 ± 0.00243438 (W 0.00223711) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.968408 ± 0.00393632 (W 0.000826217) | 1.4565 ± 0.00271418 (W 0.00179292) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.039519 ± 0.000365103 (W 0.000207774) | 0.093096 ± 0.000221279 (W 0.000140464) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.039829 ± 0.000305505 (W 0.000122468) | 0.092948 ± 0.000245147 (W 0.00129222) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.138106 ± 0.000689285 (W 0.000272958) | 0.340838 ± 0.00130511 (W 0.00126381) | - | 🔵 colmat 2.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.138335 ± 0.000606531 (W 0.000941354) | 0.344339 ± 0.00135331 (W 0.00138409) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.31686 ± 0.0005191 (W 0.00113625) | 0.777643 ± 0.000853066 (W 0.00122897) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.31623 ± 0.000716126 (W 0.00140249) | 0.777234 ± 0.00184019 (W 0.00166168) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00047 ± 1e-05 (W 1.96265e-05) | 0.00045 ± 1e-05 (W 1.48324e-05) | 0.00035 ± 5.7735e-06 (W 2.23607e-05) | · | 🔴 colmat 1.34× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00104 ± 0 (W 2.51237e-05) | 0.001 ± 1.1547e-05 (W 1.14018e-05) | 0.00073 ± 1.1547e-05 (W 2.48998e-05) | · | 🔴 colmat 1.42× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00179 ± 3.05505e-05 (W 3.4135e-05) | 0.00172 ± 2.5632e-05 (W 3.08253e-05) | 0.00123 ± 1.52753e-05 (W 1.89789e-05) | · | 🔴 colmat 1.46× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00216 ± 0.000104083 (W 2.12415e-05) | 0.00411 ± 1e-05 (W 2.07364e-05) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00219 ± 9.81495e-05 (W 2.88097e-05) | 0.00234 ± 2.08167e-05 (W 2.86356e-05) | - | 🔵 colmat 1.07× faster than `faer` (marginal: spread 0.79 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00808 ± 4.16333e-05 (W 5.18093e-05) | 0.030899 ± 4.50925e-05 (W 4.06202e-05) | - | 🔵 colmat 3.82× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00794 ± 2e-05 (W 5.74735e-05) | 0.01105 ± 5.50757e-05 (W 3.78576e-05) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.018449 ± 7.29178e-05 (W 0.000131937) | 0.052349 ± 0.0275237 (W 0.0254904) | - | 🔵 colmat 2.84× faster than `faer` (marginal: spread 0.81 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.018519 ± 6.245e-05 (W 0.000127068) | 0.024169 ± 5.68624e-05 (W 0.000213844) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.046679 ± 0.000233024 (W 0.00012859) | 0.071239 ± 0.000237059 (W 0.00134832) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.047019 ± 0.000185581 (W 0.000143315) | 0.071398 ± 8.46778e-05 (W 0.000129163) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.145085 ± 0.000564448 (W 0.000328189) | 0.349279 ± 0.00146846 (W 0.00118936) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.152086 ± 0.000773828 (W 0.000506963) | 0.349401 ± 0.00168324 (W 0.00100127) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.33267 ± 0.00124759 (W 0.00168617) | 0.810874 ± 0.000818355 (W 0.00169039) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.334109 ± 0.00179457 (W 0.00149616) | 0.809808 ± 0.00166944 (W 0.00198214) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.00841 ± 4.08697e-05 (W 8.80636e-05) | 0.00585 ± 7.68136e-05 (W 3.03018e-05) | - | 🔴 colmat 1.44× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01102 ± 8.5967e-05 (W 5.2899e-05) | 0.00706 ± 6.55769e-05 (W 0.000169853) | - | 🔴 colmat 1.56× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.033649 ± 0.000376342 (W 0.000190342) | 0.028239 ± 0.000240618 (W 6.5192e-05) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.040598 ± 3.00056e-05 (W 9.63499e-05) | 0.030789 ± 0.000328836 (W 6.01664e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.086348 ± 0.000260327 (W 0.000338064) | 0.083758 ± 0.000210239 (W 0.000115504) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.106907 ± 0.000772117 (W 0.00172603) | 0.088687 ± 0.000184825 (W 0.000117236) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00466 ± 7.55447e-05 (W 0.000196299) | 0.00461 ± 8.77895e-05 (W 4.97996e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00403 ± 4.04145e-05 (W 8.97218e-05) | 0.00459 ± 6.80686e-05 (W 4.20714e-05) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.01299 ± 0.000340783 (W 0.000239496) | 0.01225 ± 0.000119654 (W 0.000259614) | - | 🔴 colmat 1.06× slower than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01331 ± 4.30155e-05 (W 0.000282682) | 0.01165 ± 0.000189033 (W 0.000117132) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.026759 ± 0.000317123 (W 0.000302629) | 0.025649 ± 0.000235867 (W 0.000254865) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.026669 ± 0.00010583 (W 0.000441554) | 0.024129 ± 0.000204042 (W 0.00023746) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00451 ± 4.93288e-05 (W 3.35887e-05) | 0.00519 ± 6e-05 (W 3.32295e-05) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0057 ± 7.25971e-05 (W 3.98773e-05) | 0.00505 ± 5.1316e-05 (W 9.37347e-05) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02107 ± 4.46132e-05 (W 0.000352588) | 0.02373 ± 0.000343206 (W 0.000177196) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.01962 ± 5.27289e-05 (W 0.000106571) | 0.01954 ± 0.000163707 (W 0.000869578) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.041649 ± 0.000190132 (W 0.000529077) | 0.057758 ± 0.000664254 (W 0.00101115) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.044619 ± 0.000580224 (W 0.00157875) | 0.047459 ± 0.00018735 (W 0.00152504) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.021209 ± 0.000155725 (W 9.71561e-05) | 0.036099 ± 0.000521095 (W 0.0010292) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.01357 ± 9.60902e-05 (W 5.55725e-05) | 0.031129 ± 0.000575702 (W 0.000339529) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.048798 ± 0.000549909 (W 0.000586488) | 0.159665 ± 0.000410731 (W 0.00106514) | - | 🔵 colmat 3.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.045899 ± 0.0003005 (W 0.000177962) | 0.121276 ± 0.000613968 (W 0.00117212) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.106027 ± 0.000202237 (W 0.00112318) | 0.32453 ± 0.00191331 (W 0.00219064) | - | 🔵 colmat 3.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.120236 ± 0.000429182 (W 0.00112246) | 0.260101 ± 0.000887423 (W 0.0015614) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.01299 ± 0.000240069 (W 9.08422e-05) | 0.028279 ± 0.000223383 (W 0.000270792) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.025709 ± 0.000150111 (W 0.00029134) | 0.135616 ± 0.000685371 (W 0.000731524) | - | 🔵 colmat 5.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.061499 ± 0.000170735 (W 7.22862e-05) | 0.279301 ± 0.00208591 (W 0.00188109) | - | 🔵 colmat 4.54× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00492 ± 4.6188e-05 (W 4.04747e-05) | 0.00557 ± 4.04145e-05 (W 4.30488e-05) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.006539 ± 3.95011e-05 (W 3.10838e-05) | 0.00544 ± 0.000118745 (W 8.88606e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.017399 ± 0.00043062 (W 0.000356982) | 0.024069 ± 0.000355949 (W 0.000118951) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.020959 ± 0.000134859 (W 0.000195597) | 0.019699 ± 0.000135758 (W 0.000551947) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.032109 ± 0.000275379 (W 0.000323682) | 0.053679 ± 0.000177314 (W 0.000452828) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.047719 ± 0.000273008 (W 0.00177537) | 0.045558 ± 0.000242192 (W 0.00167954) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.102966 ± 0.000433222 (W 0.000143088) | 0.121736 ± 0.00764404 (W 0.00104445) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.103637 ± 0.000174701 (W 0.0010118) | 0.122236 ± 0.00178603 (W 0.00123722) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.357329 ± 0.00762402 (W 0.00165356) | 0.595401 ± 0.000432264 (W 0.00139633) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.358169 ± 0.000882192 (W 0.00101208) | 0.596964 ± 0.000984806 (W 0.0016696) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.729441 ± 0.000991369 (W 0.00240406) | 1.37727 ± 0.00118035 (W 0.00150292) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.733207 ± 0.00113004 (W 0.00321632) | 1.38164 ± 0.00101205 (W 0.00241113) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.040519 ± 0.000397031 (W 6.05822e-05) | 0.093017 ± 0.00020751 (W 0.00139647) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.040709 ± 0.000237235 (W 6.22511e-05) | 0.093537 ± 0.000352184 (W 8.02515e-05) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.142846 ± 0.000391621 (W 0.000304574) | 0.338471 ± 0.000782015 (W 0.00146603) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.143145 ± 0.000548687 (W 0.000249666) | 0.33861 ± 0.00142886 (W 0.00147065) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.34405 ± 0.00175472 (W 0.00135629) | 0.776169 ± 0.00195296 (W 0.00141438) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.343479 ± 0.00143715 (W 0.00152852) | 0.783479 ± 0.0040661 (W 0.00147257) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00064 ± 1.1547e-05 (W 1.94936e-05) | 0.00044 ± 0 (W 1.94936e-05) | 0.00035 ± 0 (W 3.1305e-05) | 🔴 colmat 1.45× slower than `faer` | 🔴 colmat 1.83× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00137 ± 3.51188e-05 (W 2.19089e-05) | 0.00099 ± 5.7735e-06 (W 2.16795e-05) | 0.00073 ± 0 (W 2.38747e-05) | 🔴 colmat 1.38× slower than `faer` | 🔴 colmat 1.88× slower than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00226 ± 3.51188e-05 (W 2.94958e-05) | 0.0017 ± 1.52753e-05 (W 2.50998e-05) | 0.00121 ± 0 (W 2.38747e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.87× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00209 ± 0.000136504 (W 2.70185e-05) | 0.00411 ± 1e-05 (W 1.22474e-05) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0023 ± 0.00010116 (W 3.65814e-05) | 0.00232 ± 2e-05 (W 2.94958e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.007759 ± 0.000311336 (W 3.65814e-05) | 0.030989 ± 0.000519711 (W 2.77489e-05) | - | 🔵 colmat 3.99× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.007579 ± 0.000231294 (W 4.72462e-05) | 0.01103 ± 0.000115326 (W 3.07132e-05) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.018409 ± 0.00030895 (W 0.00012268) | 0.045849 ± 0.000210666 (W 0.000126005) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.018729 ± 5.68624e-05 (W 0.000174426) | 0.024789 ± 0.000173312 (W 0.000191312) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.043129 ± 0.000107286 (W 0.000223782) | 0.071428 ± 0.00499055 (W 0.000165624) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.043669 ± 0.000232265 (W 0.000206604) | 0.071508 ± 0.000428991 (W 0.000117291) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.123096 ± 0.000205184 (W 0.000309965) | 0.349999 ± 0.00136881 (W 0.00201142) | - | 🔵 colmat 2.84× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.131186 ± 0.000418354 (W 0.000638837) | 0.35022 ± 0.000850086 (W 0.00183656) | - | 🔵 colmat 2.67× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.263721 ± 0.00160968 (W 0.00162431) | 0.817067 ± 0.00114498 (W 0.00216242) | - | 🔵 colmat 3.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.264583 ± 0.000575376 (W 0.00199436) | 0.815485 ± 0.000154794 (W 0.00216126) | - | 🔵 colmat 3.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.00848 ± 0.000118462 (W 4.19547e-05) | 0.00575 ± 6.60328e-05 (W 3.16259e-05) | - | 🔴 colmat 1.47× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01093 ± 6.65833e-05 (W 7.86975e-05) | 0.00689 ± 4.04145e-05 (W 2.36685e-05) | - | 🔴 colmat 1.59× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.033378 ± 0.000118027 (W 0.000108766) | 0.027839 ± 4.04145e-05 (W 2.86758e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.040389 ± 0.00010116 (W 0.000108955) | 0.030599 ± 8.02932e-05 (W 8.04002e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.091577 ± 0.000326241 (W 0.000521197) | 0.083768 ± 0.000719102 (W 7.92351e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.112526 ± 0.000104978 (W 0.00103369) | 0.088828 ± 0.000390649 (W 0.00142163) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00466 ± 0.000120554 (W 0.000178382) | 0.00465 ± 1.52753e-05 (W 6.63325e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.003999 ± 4.95681e-05 (W 4.65833e-05) | 0.0046 ± 3.60555e-05 (W 3.70135e-05) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.011239 ± 0.000280797 (W 0.000201465) | 0.013159 ± 2.09841e-05 (W 0.000363382) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01123 ± 0.000207558 (W 0.000237719) | 0.01223 ± 0.000177371 (W 0.000164965) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.021169 ± 0.000102632 (W 0.0004033) | 0.033788 ± 0.000373091 (W 0.000749684) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.020029 ± 0.000145741 (W 0.000246857) | 0.032859 ± 0.000500899 (W 0.000774009) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00451 ± 5.7735e-06 (W 2.48998e-05) | 0.00526 ± 6.95006e-05 (W 2.12132e-05) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.005609 ± 3.00056e-05 (W 0.000100733) | 0.005129 ± 0.000116148 (W 4.16437e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020679 ± 0.000381183 (W 0.000529137) | 0.023669 ± 0.000300617 (W 0.00157836) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.019529 ± 0.000104835 (W 7.98455e-05) | 0.019829 ± 0.00015308 (W 0.00021937) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.043268 ± 0.000621603 (W 0.000787624) | 0.058789 ± 0.000238496 (W 0.000576721) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.045488 ± 0.000245024 (W 0.00167071) | 0.049759 ± 0.000335336 (W 0.00159116) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.02102 ± 9.02238e-05 (W 0.000107416) | 0.042079 ± 2.64575e-05 (W 0.000713882) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.01356 ± 0.000115902 (W 6.4057e-05) | 0.036069 ± 0.000336452 (W 0.000370108) | - | 🔵 colmat 2.66× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.051358 ± 0.000404438 (W 0.000459229) | 0.189454 ± 0.00113636 (W 0.00208473) | - | 🔵 colmat 3.69× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.046048 ± 0.00050327 (W 0.000181998) | 0.152235 ± 0.00125118 (W 0.00339757) | - | 🔵 colmat 3.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.105327 ± 0.00051449 (W 0.000945426) | 0.387518 ± 0.00740209 (W 0.00331626) | - | 🔵 colmat 3.68× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.120576 ± 0.00051304 (W 0.00104871) | 0.3263 ± 0.00125273 (W 0.00250793) | - | 🔵 colmat 2.71× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.013409 ± 0.000115326 (W 9.69706e-05) | 0.034029 ± 0.000612046 (W 0.000614543) | - | 🔵 colmat 2.54× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.025789 ± 0.000170587 (W 0.000173641) | 0.167065 ± 0.00164495 (W 0.00216749) | - | 🔵 colmat 6.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.058449 ± 0.000135242 (W 0.000153119) | 0.339261 ± 0.00331192 (W 0.00337044) | - | 🔵 colmat 5.80× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00494 ± 4.50925e-05 (W 5.82426e-05) | 0.00557 ± 0.00010116 (W 5.2098e-05) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00656 ± 2.36714e-05 (W 3.65814e-05) | 0.00562 ± 6.45781e-05 (W 0.000124291) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.017309 ± 0.000204858 (W 0.000461794) | 0.024609 ± 2.12211e-05 (W 0.000126231) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.02104 ± 0.00012454 (W 0.000109992) | 0.01978 ± 0.000125367 (W 0.000243843) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.034569 ± 0.000405956 (W 0.000380039) | 0.054499 ± 0.000170696 (W 0.000431155) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.048179 ± 0.000370433 (W 0.0023372) | 0.047068 ± 0.000265801 (W 0.000263807) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.103727 ± 0.00113001 (W 0.000130074) | 0.121747 ± 0.00214102 (W 0.00106643) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.104147 ± 0.000548088 (W 0.000128686) | 0.122947 ± 0.000655492 (W 0.000895134) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.35694 ± 0.00106811 (W 0.00188562) | 0.599992 ± 0.00182214 (W 0.00229554) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.36099 ± 0.00246552 (W 0.00164081) | 0.600822 ± 0.00055248 (W 0.00193791) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.73679 ± 8.06164e-05 (W 0.00253828) | 1.38923 ± 0.00195906 (W 0.0050821) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.741716 ± 0.00236403 (W 0.00207841) | 1.39247 ± 0.00359296 (W 0.00335186) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.040889 ± 0.000145029 (W 7.76531e-05) | 0.092917 ± 0.00011547 (W 0.00105768) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.040539 ± 0.000169608 (W 9.04279e-05) | 0.093327 ± 0.000228108 (W 0.0015199) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.143336 ± 0.00063394 (W 0.000251554) | 0.339369 ± 0.000465644 (W 0.00145875) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.143566 ± 0.000609617 (W 0.000265932) | 0.340239 ± 0.000651862 (W 0.00169139) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.352269 ± 0.00124701 (W 0.00152282) | 0.774785 ± 0.000866331 (W 0.00143299) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.354659 ± 0.00163237 (W 0.00180501) | 0.782716 ± 0.00421024 (W 0.0017184) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00061 ± 5.7735e-06 (W 1.92354e-05) | 0.00044 ± 1.1547e-05 (W 2.48998e-05) | 0.00035 ± 0 (W 3.1305e-05) | 🔴 colmat 1.39× slower than `faer` | 🔴 colmat 1.74× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00132 ± 2.51661e-05 (W 2.44786e-05) | 0.000989 ± 5.50757e-06 (W 1.67332e-05) | 0.00073 ± 6.08276e-06 (W 2.54951e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.81× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00228 ± 5.52298e-05 (W 3.78576e-05) | 0.00172 ± 1.1547e-05 (W 2.73861e-05) | 0.00122 ± 5.7735e-06 (W 2.07364e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.87× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00184 ± 0.00010116 (W 1.34164e-05) | 0.00182 ± 1.52753e-05 (W 7.0852e-06) | 0.00199 ± 1.1547e-05 (W 5.65685e-05) | · | 🔵 colmat 1.08× faster than `lapack-mkl` (marginal: spread 0.75 of the difference) | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00228 ± 0.000103923 (W 3.89872e-05) | 0.00226 ± 1e-05 (W 2.28035e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00643 ± 2.0502e-05 (W 2.18678e-05) | 0.00977 ± 8.14453e-05 (W 2.11471e-05) | 0.00803 ± 0.000230651 (W 4.55982e-05) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00767 ± 4.72582e-05 (W 5.26991e-05) | 0.01107 ± 2e-05 (W 2.6857e-05) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0142 ± 0.000357911 (W 3.13417e-05) | 0.01722 ± 0.000133742 (W 4.38554e-05) | 0.02184 ± 3.23574e-05 (W 5.44178e-05) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01636 ± 0.000330051 (W 6.18005e-05) | 0.0198 ± 0.000138684 (W 4.70776e-05) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.067238 ± 0.000326956 (W 0.00015462) | 0.071108 ± 0.000129616 (W 0.000124424) | 0.062989 ± 8.55005e-05 (W 5.89067e-05) | 🔵 colmat 1.06× faster than `faer` | 🔴 colmat 1.07× slower than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.067778 ± 0.0003 (W 0.000166638) | 0.071588 ± 0.000236166 (W 8.25003e-05) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.280413 ± 0.000729083 (W 0.00108267) | 0.348789 ± 0.000430484 (W 0.00133232) | 0.253525 ± 0.000134993 (W 0.00112468) | 🔵 colmat 1.24× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.280251 ± 0.000313278 (W 0.0011264) | 0.350922 ± 0.000210644 (W 0.00136085) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.678794 ± 0.00137396 (W 0.00187466) | 0.818182 ± 0.00127991 (W 0.00178415) | 0.580767 ± 0.00111951 (W 0.0014984) | 🔵 colmat 1.21× faster than `faer` | 🔴 colmat 1.17× slower than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.689645 ± 0.000594468 (W 0.00277845) | 0.832272 ± 0.0027608 (W 0.00164223) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.007819 ± 0.00120699 (W 3.51041e-05) | 0.00602 ± 0.00174243 (W 2.86217e-05) | 0.01647 ± 0.000120168 (W 6.06078e-05) | · | 🔵 colmat 2.11× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.00885 ± 5.03322e-05 (W 5.91549e-05) | 0.00686 ± 0.000145258 (W 3.51753e-05) | - | 🔴 colmat 1.29× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.032659 ± 7.54983e-05 (W 0.000379666) | 0.027629 ± 6.80686e-05 (W 3.78153e-05) | 0.127576 ± 0.000317812 (W 0.00135402) | 🔴 colmat 1.18× slower than `faer` | 🔵 colmat 3.91× faster than `lapack-mkl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.034769 ± 0.000178979 (W 8.78635e-05) | 0.029889 ± 0.00034788 (W 0.000222978) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.087788 ± 0.000630127 (W 6.60174e-05) | 0.079089 ± 0.000278857 (W 0.00127647) | 0.429375 ± 0.00130889 (W 0.00124701) | 🔴 colmat 1.11× slower than `faer` | 🔵 colmat 4.89× faster than `lapack-mkl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.096507 ± 0.00124815 (W 0.00141321) | 0.088247 ± 0.000252706 (W 0.00144013) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00457 ± 2.08167e-05 (W 4.12311e-05) | 0.004349 ± 2.62742e-05 (W 2.96648e-05) | 0.011119 ± 7.68136e-05 (W 4.95096e-05) | · | 🔵 colmat 2.43× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00555 ± 5.56776e-05 (W 3.98409e-05) | 0.00434 ± 1.52753e-05 (W 3.53864e-05) | 0.00784 ± 5.68624e-05 (W 9.98609e-05) | 🔴 colmat 1.28× slower than `faer` | 🔵 colmat 1.41× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.019459 ± 5.22526e-05 (W 6.6078e-05) | 0.01906 ± 0.000301812 (W 8.7466e-05) | 0.058759 ± 0.000190001 (W 0.000109756) | · | 🔵 colmat 3.02× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01964 ± 3.4356e-05 (W 6.6763e-05) | 0.01888 ± 0.000490721 (W 3.80684e-05) | 0.054049 ± 0.000170666 (W 0.000157786) | · | 🔵 colmat 2.75× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.055408 ± 0.000213823 (W 0.000153311) | 0.054189 ± 0.000181403 (W 0.000697686) | 0.144245 ± 0.00028429 (W 0.00153769) | · | 🔵 colmat 2.60× faster than `lapack-mkl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.054988 ± 0.000177286 (W 0.00104768) | 0.053819 ± 0.000185176 (W 0.00010681) | 0.133135 ± 0.000568379 (W 9.28671e-05) | · | 🔵 colmat 2.42× faster than `lapack-mkl` |  |
| `lu/n32/b1` | 0.00347 ± 6.55744e-05 (W 3.65951e-05) | 0.00385 ± 2.3388e-05 (W 2.77489e-05) | 0.00692 ± 6.1101e-05 (W 7.90652e-05) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.99× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00441 ± 5.7735e-06 (W 4.97413e-05) | 0.00495 ± 2.88675e-05 (W 4.34649e-05) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01269 ± 1.51767e-05 (W 0.000148678) | 0.01391 ± 0.0001205 (W 8.04133e-05) | 0.02254 ± 4.96689e-05 (W 8.54476e-05) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.01498 ± 3.4641e-05 (W 0.000160625) | 0.016299 ± 9.96009e-05 (W 0.000114953) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.028329 ± 0.000221676 (W 0.000228736) | 0.031129 ± 6.39296e-05 (W 0.000140819) | 0.049988 ± 0.000204842 (W 0.00011225) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.76× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.036899 ± 0.000111505 (W 0.00128348) | 0.039919 ± 0.000158368 (W 0.00144364) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.01081 ± 5.52298e-05 (W 3.27154e-05) | 0.019629 ± 8.02517e-05 (W 0.000103428) | 0.014079 ± 4.7648e-05 (W 7.72548e-05) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.01237 ± 0.000100201 (W 4.2957e-05) | 0.02128 ± 6.26126e-05 (W 8.02889e-05) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.040949 ± 0.00119858 (W 7.30979e-05) | 0.069037 ± 0.00126968 (W 0.000122173) | 0.089627 ± 0.000254925 (W 0.000143805) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 2.19× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.044228 ± 0.000130615 (W 0.000113324) | 0.072598 ± 0.00128842 (W 0.000153686) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.102768 ± 0.000365218 (W 0.000387303) | 0.153386 ± 0.00129362 (W 0.00122515) | 0.243824 ± 0.00141827 (W 0.00128157) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.126057 ± 0.000110395 (W 0.00131873) | 0.177354 ± 0.00175694 (W 0.00239222) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00536 ± 4.04145e-05 (W 4.68423e-05) | 0.01211 ± 0.000326241 (W 7.5082e-05) | 0.00698 ± 2.02567e-05 (W 4.45343e-05) | 🔵 colmat 2.26× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.01985 ± 9.07377e-05 (W 4.51032e-05) | 0.048098 ± 3.00056e-05 (W 0.000152267) | 0.035229 ± 8.02517e-05 (W 5.56345e-05) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n96/b1` | 0.050128 ± 0.000231517 (W 7.45272e-05) | 0.101577 ± 0.00113349 (W 0.000845905) | 0.109006 ± 0.000105078 (W 0.00176375) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 2.17× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.00393 ± 5.50757e-05 (W 3.08415e-05) | 0.00428 ± 3.05505e-05 (W 3.38556e-05) | 0.00687 ± 6.75056e-05 (W 0.000101898) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00495 ± 8.3865e-05 (W 3.76191e-05) | 0.00518 ± 2.51661e-05 (W 6.68895e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01293 ± 3.01164e-05 (W 6.49015e-05) | 0.014289 ± 9.20887e-05 (W 4.02405e-05) | 0.02233 ± 6.48614e-05 (W 6.58878e-05) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.015059 ± 0.000115052 (W 4.71094e-05) | 0.016639 ± 0.000240791 (W 4.1028e-05) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.028399 ± 0.000104069 (W 8.85737e-05) | 0.031319 ± 0.000123423 (W 6.92618e-05) | 0.049259 ± 0.000290569 (W 0.00108403) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.037709 ± 0.000275518 (W 0.00151866) | 0.040149 ± 0.000154522 (W 0.000122606) | - | 🔵 colmat 1.06× faster than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.103488 ± 0.00502717 (W 0.00125777) | 0.122517 ± 0.000168239 (W 0.000942938) | 0.110048 ± 0.00019458 (W 0.00125487) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.06× faster than `lapack-mkl` (marginal: spread 0.80 of the difference) | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.103778 ± 0.000279144 (W 0.000128098) | 0.122666 ± 0.000453593 (W 0.000168512) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.431785 ± 0.00659334 (W 0.00130216) | 0.628856 ± 0.000715517 (W 0.00156593) | 0.452685 ± 0.00139676 (W 0.00168763) | 🔵 colmat 1.46× faster than `faer` | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.43107 ± 0.000797215 (W 0.00139197) | 0.632649 ± 0.00116215 (W 0.00143682) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.979788 ± 0.00248828 (W 0.000350542) | 1.40158 ± 0.00210853 (W 0.00179474) | 1.04356 ± 0.00243206 (W 0.000862825) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 0.980857 ± 0.00195676 (W 0.000538645) | 1.45178 ± 0.000640025 (W 0.0024102) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.039179 ± 0.000284518 (W 0.000205478) | 0.094307 ± 0.000141966 (W 0.000929317) | 0.037049 ± 3.27159e-05 (W 6.81271e-05) | 🔵 colmat 2.41× faster than `faer` | 🔴 colmat 1.06× slower than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.039448 ± 0.000145332 (W 7.50946e-05) | 0.094387 ± 0.000125527 (W 0.000133978) | - | 🔵 colmat 2.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.140186 ± 0.000520254 (W 0.0017334) | 0.353909 ± 0.000362911 (W 0.0011658) | 0.163304 ± 0.000457691 (W 0.000860337) | 🔵 colmat 2.52× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.140257 ± 0.000632646 (W 0.0012572) | 0.358017 ± 0.000359379 (W 0.00107526) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.321919 ± 0.0011135 (W 0.00104688) | 0.780674 ± 0.00138494 (W 0.000913088) | 0.385338 ± 0.000885993 (W 0.00167267) | 🔵 colmat 2.43× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.322489 ± 0.0009812 (W 0.0018087) | 0.779603 ± 0.00239486 (W 0.00127633) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00051 ± 5.7735e-06 (W 1.66493e-05) | 0.0005 ± 1e-05 (W 1.41421e-05) | 0.00033 ± 0 (W 3.1305e-05) | · | 🔴 colmat 1.55× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00117 ± 1e-05 (W 1.51658e-05) | 0.00112 ± 1.1547e-05 (W 1.64317e-05) | 0.00069 ± 5.7735e-06 (W 2.88097e-05) | · | 🔴 colmat 1.70× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00195 ± 1.1547e-05 (W 2.48998e-05) | 0.0019 ± 5.7735e-06 (W 1.64317e-05) | 0.00114 ± 0 (W 3.40617e-05) | · | 🔴 colmat 1.71× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00163 ± 0.000187705 (W 1.51658e-05) | 0.00181 ± 1e-05 (W 2.16795e-05) | 0.00198 ± 5.7735e-06 (W 5.04301e-05) | · | 🔵 colmat 1.21× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00233 ± 0.000124231 (W 2.60768e-05) | 0.00225 ± 1.52753e-05 (W 3.78444e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00617 ± 0.000211266 (W 1.26174e-05) | 0.009779 ± 1.00167e-05 (W 2.61954e-05) | 0.00841 ± 0.0003177 (W 6.93772e-05) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00758 ± 0.000194251 (W 8.13462e-05) | 0.01104 ± 0.000186815 (W 3.72048e-05) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.015369 ± 6.86319e-05 (W 6.94356e-05) | 0.020279 ± 0.000100683 (W 0.00027678) | 0.023079 ± 0.000508465 (W 0.000107621) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.50× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.018039 ± 0.00022301 (W 0.000170614) | 0.024219 ± 4.59384e-05 (W 0.000277313) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.046568 ± 0.000292962 (W 0.000110473) | 0.071307 ± 0.000121903 (W 0.000125029) | 0.066708 ± 0.000212211 (W 0.000232829) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.047239 ± 0.000436505 (W 0.000156174) | 0.071758 ± 0.000399888 (W 0.000156196) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.144716 ± 7.25282e-05 (W 0.00030304) | 0.35095 ± 0.000850132 (W 0.00148887) | 0.174545 ± 0.000776454 (W 0.000747299) | 🔵 colmat 2.43× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.151665 ± 0.000687932 (W 0.000368177) | 0.3485 ± 0.000873305 (W 0.00153235) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.331131 ± 0.00058131 (W 0.00122284) | 0.811374 ± 0.00313154 (W 0.00117758) | 0.380698 ± 0.00120096 (W 0.00134964) | 🔵 colmat 2.45× faster than `faer` | 🔵 colmat 1.15× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.333849 ± 0.000439197 (W 0.00121067) | 0.809895 ± 0.0014547 (W 0.00168835) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.00861 ± 6.65833e-05 (W 5.46736e-05) | 0.00577 ± 7.50555e-05 (W 2.80036e-05) | 0.016509 ± 3.56417e-05 (W 6.70992e-05) | 🔴 colmat 1.49× slower than `faer` | 🔵 colmat 1.92× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.00969 ± 7.63763e-05 (W 7.33021e-05) | 0.00686 ± 3.51188e-05 (W 7.36016e-05) | - | 🔴 colmat 1.41× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.034009 ± 0.00019218 (W 0.000180978) | 0.028289 ± 0.000357529 (W 5.88829e-05) | 0.127366 ± 0.00047571 (W 0.0012927) | 🔴 colmat 1.20× slower than `faer` | 🔵 colmat 3.75× faster than `lapack-mkl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.036439 ± 0.00019218 (W 6.02594e-05) | 0.030039 ± 3e-05 (W 6.04086e-05) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.089077 ± 0.000597746 (W 0.00055394) | 0.083808 ± 0.000150444 (W 0.00130186) | 0.429906 ± 0.000744148 (W 0.00204637) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 4.83× faster than `lapack-mkl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.100917 ± 0.000901795 (W 0.000780983) | 0.087378 ± 0.000197553 (W 0.00149143) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00418 ± 7.76745e-05 (W 6.84047e-05) | 0.00452 ± 6.55744e-05 (W 7.84219e-05) | 0.026489 ± 0.000748051 (W 0.000758644) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 6.34× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00445 ± 9.64365e-05 (W 0.000115785) | 0.00445 ± 3.4641e-05 (W 3.40617e-05) | 0.0078 ± 8.56757e-05 (W 0.000120409) | · | 🔵 colmat 1.75× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.01209 ± 2.5697e-05 (W 0.000129677) | 0.01152 ± 0.000340206 (W 0.000179065) | 0.058969 ± 0.00324923 (W 0.00306208) | · | 🔵 colmat 4.88× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.012149 ± 3.78594e-05 (W 0.000144887) | 0.012089 ± 0.000293655 (W 0.000246772) | 0.024599 ± 7.6922e-05 (W 0.000156937) | · | 🔵 colmat 2.02× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.025029 ± 0.000251064 (W 0.000149265) | 0.02395 ± 0.000304533 (W 0.000319516) | 0.084308 ± 0.00400513 (W 0.00505515) | · | 🔵 colmat 3.37× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.025009 ± 0.000149701 (W 0.000141817) | 0.023949 ± 0.000252798 (W 0.000231016) | 0.048199 ± 0.000689195 (W 0.000243015) | · | 🔵 colmat 1.93× faster than `lapack-mkl` |  |
| `lu/n32/b1` | 0.00357 ± 2.47049e-05 (W 3e-05) | 0.00394 ± 1.73205e-05 (W 2.20726e-05) | 0.006949 ± 4.59384e-05 (W 6.68895e-05) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00467 ± 5.56507e-05 (W 6.68581e-05) | 0.00497 ± 9.07377e-05 (W 5.87724e-05) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0141 ± 0.00023063 (W 0.000205857) | 0.016389 ± 0.000114282 (W 0.000160248) | 0.02269 ± 0.000193753 (W 0.000121574) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.016089 ± 0.000231085 (W 0.000161128) | 0.019099 ± 0.000235727 (W 0.000748907) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.025809 ± 0.000253246 (W 0.000242518) | 0.036529 ± 0.000109576 (W 0.00112478) | 0.050138 ± 3.01164e-05 (W 0.00013224) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.036359 ± 0.000434665 (W 0.001584) | 0.046239 ± 0.00053809 (W 0.00141069) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.010879 ± 0.000120554 (W 3.74192e-05) | 0.028659 ± 0.000315511 (W 0.000888806) | 0.014159 ± 0.000168395 (W 7.80852e-05) | 🔵 colmat 2.63× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.01256 ± 5.52298e-05 (W 6.09779e-05) | 0.030449 ± 0.000332892 (W 0.000281922) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.038439 ± 0.000393256 (W 0.000117649) | 0.115027 ± 0.000250201 (W 0.000583292) | 0.060469 ± 0.000145145 (W 0.000253884) | 🔵 colmat 2.99× faster than `faer` | 🔵 colmat 1.57× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.041769 ± 0.000200226 (W 0.000202114) | 0.118656 ± 0.00109512 (W 0.00121293) | - | 🔵 colmat 2.84× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.084428 ± 0.000431216 (W 0.000320152) | 0.227332 ± 0.00125737 (W 0.002319) | 0.158025 ± 0.000781978 (W 0.000341936) | 🔵 colmat 2.69× faster than `faer` | 🔵 colmat 1.87× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.113237 ± 0.000507504 (W 0.00115988) | 0.257153 ± 0.00132911 (W 0.0018807) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00533 ± 3.21455e-05 (W 3.17836e-05) | 0.02116 ± 0.00034253 (W 0.000188614) | 0.00695 ± 5.2915e-05 (W 4.67354e-05) | 🔵 colmat 3.97× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.01965 ± 7.5368e-05 (W 3.45297e-05) | 0.096117 ± 0.000595438 (W 0.00116853) | 0.035709 ± 0.000650333 (W 7.44983e-05) | 🔵 colmat 4.89× faster than `faer` | 🔵 colmat 1.82× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n96/b1` | 0.050078 ± 0.00031768 (W 9.54427e-05) | 0.188414 ± 0.000847438 (W 0.00128224) | 0.110077 ± 0.000832887 (W 0.00147313) | 🔵 colmat 3.76× faster than `faer` | 🔵 colmat 2.20× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.00427 ± 9.50438e-06 (W 3.17522e-05) | 0.004179 ± 1.53948e-05 (W 4.4025e-05) | 0.00687 ± 1.18462e-05 (W 4.5653e-05) | · | 🔵 colmat 1.61× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00533 ± 2.0502e-05 (W 4.33151e-05) | 0.00521 ± 3.09892e-05 (W 3.91561e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.014719 ± 9.41293e-05 (W 0.000168423) | 0.01603 ± 3.49333e-05 (W 0.000365111) | 0.022249 ± 6.05007e-05 (W 8.06734e-05) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.017119 ± 9.30609e-05 (W 0.000111805) | 0.01863 ± 0.0006245 (W 0.000583297) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.02737 ± 0.000285337 (W 0.000153968) | 0.034089 ± 0.00042194 (W 0.000383406) | 0.049408 ± 7.55447e-05 (W 7.30363e-05) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.039528 ± 0.000138553 (W 0.00146645) | 0.044249 ± 0.000409512 (W 0.00177484) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.103908 ± 0.000197335 (W 5.53561e-05) | 0.124456 ± 0.00100152 (W 0.00048693) | 0.115136 ± 0.000430152 (W 0.0012391) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.104768 ± 0.000285757 (W 0.000145063) | 0.124676 ± 0.000717379 (W 0.00101534) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.357219 ± 0.000195144 (W 0.00155441) | 0.62632 ± 0.00198471 (W 0.00266065) | 0.416306 ± 0.00238089 (W 0.00341035) | 🔵 colmat 1.75× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.35901 ± 0.000456593 (W 0.00105825) | 0.62683 ± 0.000987489 (W 0.00178569) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.73799 ± 0.00196108 (W 0.00317019) | 1.37638 ± 0.00133157 (W 0.00179929) | 0.888932 ± 0.00188598 (W 0.00142176) | 🔵 colmat 1.87× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 0.743527 ± 0.00276637 (W 0.00255799) | 1.37887 ± 0.00146445 (W 0.00224918) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.039619 ± 4.52806e-05 (W 7.95248e-05) | 0.093997 ± 0.0001253 (W 0.00124928) | 0.037179 ± 0.000216333 (W 5.21536e-05) | 🔵 colmat 2.37× faster than `faer` | 🔴 colmat 1.07× slower than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.040049 ± 4.72582e-05 (W 0.000115629) | 0.094697 ± 0.000193993 (W 0.00102526) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.144425 ± 0.000514536 (W 0.000231882) | 0.352349 ± 0.000301016 (W 0.00195465) | 0.166286 ± 0.000129361 (W 0.00118938) | 🔵 colmat 2.44× faster than `faer` | 🔵 colmat 1.15× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.145456 ± 0.00064941 (W 0.000350386) | 0.354289 ± 0.0128347 (W 0.00198766) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n96/b1` | 0.348351 ± 0.000777321 (W 0.00118665) | 0.780535 ± 0.00159797 (W 0.00130381) | 0.38447 ± 0.000300746 (W 0.00193875) | 🔵 colmat 2.24× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.348139 ± 0.00149358 (W 0.00183521) | 0.786185 ± 0.00236363 (W 0.00107709) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00064 ± 3e-05 (W 2.38747e-05) | 0.00049 ± 1.52753e-05 (W 1.92354e-05) | 0.00033 ± 5.7735e-06 (W 3.1305e-05) | 🔴 colmat 1.31× slower than `faer` | 🔴 colmat 1.94× slower than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00142 ± 5.7735e-06 (W 2.38747e-05) | 0.00114 ± 1.52753e-05 (W 1.64317e-05) | 0.00069 ± 5.7735e-06 (W 3.04959e-05) | 🔴 colmat 1.25× slower than `faer` | 🔴 colmat 2.06× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00233 ± 1.73205e-05 (W 3.33946e-05) | 0.001909 ± 1.12694e-05 (W 2.28298e-05) | 0.00114 ± 1.1547e-05 (W 2.50998e-05) | 🔴 colmat 1.22× slower than `faer` | 🔴 colmat 2.04× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00163 ± 4.3589e-05 (W 1.41421e-05) | 0.00186 ± 2.51661e-05 (W 3.11448e-05) | 0.0019 ± 6.4291e-05 (W 4.08656e-05) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00207 ± 0.000127017 (W 3.91152e-05) | 0.0023 ± 6.08276e-06 (W 1.22474e-05) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00623 ± 0.000189297 (W 4.30116e-05) | 0.00974 ± 0.000124527 (W 3.27918e-05) | 0.00789 ± 4.63717e-05 (W 3.93345e-05) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 1.27× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00749 ± 0.000268006 (W 5.94407e-05) | 0.01097 ± 7.23418e-05 (W 3.10532e-05) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.01439 ± 0.000302903 (W 7.25555e-05) | 0.021119 ± 0.000127084 (W 0.000150124) | 0.02176 ± 0.000195014 (W 8.3715e-05) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01755 ± 0.000306839 (W 9.56049e-05) | 0.024889 ± 0.000113006 (W 0.000310183) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.043279 ± 2.47049e-05 (W 0.000134649) | 0.071118 ± 0.000492172 (W 0.00116337) | 0.067358 ± 0.000174261 (W 0.000277444) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.043888 ± 0.000237059 (W 0.000156139) | 0.071618 ± 0.000138854 (W 0.000221359) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.122776 ± 0.000750755 (W 0.000216061) | 0.349689 ± 0.000543324 (W 0.00162076) | 0.169105 ± 0.000554049 (W 0.00211759) | 🔵 colmat 2.85× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.131036 ± 0.000254231 (W 0.00038887) | 0.349379 ± 0.00165772 (W 0.00184029) | - | 🔵 colmat 2.67× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.265163 ± 0.000994442 (W 0.0032432) | 0.820235 ± 0.00438812 (W 0.00213399) | 0.370459 ± 0.00171604 (W 0.0026988) | 🔵 colmat 3.09× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.264122 ± 0.000722856 (W 0.00192926) | 0.819467 ± 0.00191789 (W 0.00156492) | - | 🔵 colmat 3.10× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.00864 ± 7.84241e-05 (W 4.78142e-05) | 0.00591 ± 7.37111e-05 (W 2.39416e-05) | 0.016449 ± 9.58662e-05 (W 8.80415e-05) | 🔴 colmat 1.46× slower than `faer` | 🔵 colmat 1.90× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.00967 ± 4.6188e-05 (W 5.08744e-05) | 0.00688 ± 0.000123423 (W 2.98865e-05) | - | 🔴 colmat 1.41× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.034219 ± 4e-05 (W 0.000144845) | 0.028259 ± 0.000147986 (W 6.48012e-05) | 0.127146 ± 0.000304138 (W 0.00126769) | 🔴 colmat 1.21× slower than `faer` | 🔵 colmat 3.72× faster than `lapack-mkl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.036759 ± 0.000142818 (W 7.21401e-05) | 0.029759 ± 6.4291e-05 (W 7.28711e-05) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.094087 ± 0.000695509 (W 0.00181495) | 0.083607 ± 0.000228693 (W 0.0015966) | 0.429476 ± 0.000976301 (W 0.00172825) | 🔴 colmat 1.13× slower than `faer` | 🔵 colmat 4.56× faster than `lapack-mkl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.105797 ± 0.000816884 (W 0.00174197) | 0.087428 ± 0.000277849 (W 0.00158932) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00397 ± 0.000140119 (W 7.84219e-05) | 0.00451 ± 1.1547e-05 (W 6.39469e-05) | 0.036269 ± 0.00106143 (W 0.00190359) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 9.14× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00396 ± 7.0946e-05 (W 4.9002e-05) | 0.00452 ± 3.21455e-05 (W 4.30116e-05) | 0.00771 ± 3.05505e-05 (W 3.78444e-05) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.00937 ± 5.1316e-05 (W 0.000109313) | 0.012309 ± 0.000125261 (W 0.000296675) | 0.074498 ± 0.00237475 (W 0.00432338) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 7.95× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.00964 ± 0.000149221 (W 0.000134934) | 0.01282 ± 0.000168573 (W 0.00028384) | 0.02004 ± 0.000132478 (W 0.000247081) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 2.08× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.018919 ± 0.000207445 (W 0.000216731) | 0.031979 ± 0.000337095 (W 0.000439621) | 0.088308 ± 0.000359073 (W 0.00400039) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 4.67× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.018729 ± 7.84241e-05 (W 0.000103437) | 0.032379 ± 0.000737496 (W 0.000963285) | 0.0336 ± 0.000550255 (W 0.000273313) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.00355 ± 1.52753e-05 (W 3.71484e-05) | 0.00397 ± 2.08167e-05 (W 3.28634e-05) | 0.0069 ± 9.02792e-05 (W 7.48879e-05) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00454 ± 3.01164e-05 (W 4.34074e-05) | 0.00497 ± 7.76745e-05 (W 2.64045e-05) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01378 ± 0.000182758 (W 0.000220509) | 0.01654 ± 3.60555e-05 (W 0.000142977) | 0.022569 ± 8.07279e-05 (W 4.90337e-05) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.64× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.015909 ± 2.51064e-05 (W 0.000153497) | 0.019229 ± 9.70069e-05 (W 0.000296288) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.025209 ± 7.37111e-05 (W 0.00033403) | 0.037329 ± 0.000399825 (W 0.000851897) | 0.049968 ± 0.000131493 (W 0.00109019) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 1.98× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.037378 ± 0.000238517 (W 0.00154855) | 0.048299 ± 0.000420246 (W 0.000437635) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.011029 ± 0.000160104 (W 3.58929e-05) | 0.034289 ± 0.000482113 (W 0.000798887) | 0.01385 ± 6.46555e-05 (W 0.000100006) | 🔵 colmat 3.11× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.01264 ± 6.96443e-05 (W 4.64252e-05) | 0.035809 ± 0.00100749 (W 0.000449811) | - | 🔵 colmat 2.83× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.037809 ± 0.000170098 (W 0.000130843) | 0.148545 ± 0.000392887 (W 0.000882345) | 0.056218 ± 0.00033759 (W 0.000218335) | 🔵 colmat 3.93× faster than `faer` | 🔵 colmat 1.49× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.041819 ± 0.000339559 (W 0.000147855) | 0.150536 ± 0.00124222 (W 0.00242833) | - | 🔵 colmat 3.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.083148 ± 0.000324608 (W 0.000371415) | 0.292611 ± 0.00037762 (W 0.00158308) | 0.144516 ± 0.000232451 (W 0.000434649) | 🔵 colmat 3.52× faster than `faer` | 🔵 colmat 1.74× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.113677 ± 0.00293257 (W 0.00120544) | 0.31896 ± 0.000929575 (W 0.00236086) | - | 🔵 colmat 2.81× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00538 ± 4.72582e-05 (W 4.21331e-05) | 0.02802 ± 0.00040085 (W 0.000565949) | 0.00692 ± 4.10731e-05 (W 4.87883e-05) | 🔵 colmat 5.21× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.01978 ± 0.000231589 (W 5.68885e-05) | 0.128346 ± 0.00103645 (W 0.00141466) | 0.034989 ± 0.000106927 (W 6.42651e-05) | 🔵 colmat 6.49× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n96/b1` | 0.050348 ± 0.000240327 (W 0.000102334) | 0.252243 ± 0.00353703 (W 0.00330361) | 0.108617 ± 3.60555e-05 (W 0.00115418) | 🔵 colmat 5.01× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.00428 ± 3e-05 (W 6.66348e-05) | 0.00442 ± 0.000187705 (W 3.70135e-05) | 0.0068 ± 2.51661e-05 (W 6.14508e-05) | · | 🔵 colmat 1.59× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.005359 ± 1.51767e-05 (W 4.16437e-05) | 0.00525 ± 0.000103828 (W 7.9399e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01476 ± 0.000105832 (W 0.000119555) | 0.01621 ± 0.00017164 (W 0.000541323) | 0.02212 ± 9.50281e-05 (W 7.06491e-05) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.50× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.01711 ± 0.000153189 (W 8.8968e-05) | 0.01871 ± 0.000288791 (W 0.000264867) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.027269 ± 0.000145029 (W 0.000101489) | 0.034509 ± 0.000210317 (W 0.000286042) | 0.048798 ± 0.000341223 (W 0.000134901) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.040309 ± 0.000297714 (W 0.000213373) | 0.046059 ± 2.60832e-05 (W 0.00177829) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.104047 ± 0.00331247 (W 9.12042e-05) | 0.123936 ± 0.00124302 (W 0.000168315) | 0.116387 ± 0.000231128 (W 0.000172686) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.12× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.105337 ± 0.000110604 (W 0.000172294) | 0.122867 ± 0.00234698 (W 0.000255807) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.358239 ± 0.00126122 (W 0.00145248) | 0.627601 ± 0.00281633 (W 0.00205453) | 0.409658 ± 0.00090436 (W 0.00300303) | 🔵 colmat 1.75× faster than `faer` | 🔵 colmat 1.14× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.36193 ± 0.00225046 (W 0.00234534) | 0.628663 ± 0.000982379 (W 0.00248115) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.743059 ± 0.00327552 (W 0.00308581) | 1.38618 ± 0.00338815 (W 0.00272249) | 0.853734 ± 0.00208808 (W 0.00392814) | 🔵 colmat 1.87× faster than `faer` | 🔵 colmat 1.15× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 0.749716 ± 0.00307704 (W 0.00204674) | 1.39176 ± 0.0019076 (W 0.00238008) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.040439 ± 0.000265016 (W 0.000103471) | 0.094097 ± 5.68624e-05 (W 0.000186467) | 0.037068 ± 9.26841e-05 (W 9.98359e-05) | 🔵 colmat 2.33× faster than `faer` | 🔴 colmat 1.09× slower than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.040139 ± 0.000166533 (W 4.93477e-05) | 0.094447 ± 2.51661e-05 (W 0.000166823) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.145336 ± 0.00090769 (W 0.00021166) | 0.35198 ± 0.000195014 (W 0.00156436) | 0.166274 ± 0.000504203 (W 0.00075422) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 1.14× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.145765 ± 0.00115819 (W 0.000238139) | 0.352559 ± 0.000572678 (W 0.00126709) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.35853 ± 0.00239137 (W 0.00170476) | 0.781229 ± 0.000702975 (W 0.00179743) | 0.382898 ± 0.000941106 (W 0.0012172) | 🔵 colmat 2.18× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.359048 ± 0.00137229 (W 0.00141646) | 0.788908 ± 0.000477485 (W 0.00219733) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00065 ± 1.1547e-05 (W 2.88097e-05) | 0.00052 ± 1.1547e-05 (W 2.04939e-05) | 0.00034 ± 5.7735e-06 (W 2.48998e-05) | 🔴 colmat 1.25× slower than `faer` | 🔴 colmat 1.91× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00142 ± 1e-05 (W 1.92354e-05) | 0.00113 ± 1.1547e-05 (W 2.16795e-05) | 0.00069 ± 1e-05 (W 2.50998e-05) | 🔴 colmat 1.26× slower than `faer` | 🔴 colmat 2.06× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00237 ± 4e-05 (W 3.08221e-05) | 0.00189 ± 1.1547e-05 (W 1.78885e-05) | 0.00114 ± 0 (W 2.16795e-05) | 🔴 colmat 1.25× slower than `faer` | 🔴 colmat 2.08× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0033 ± 1.73205e-05 (W 1.78885e-05) | 0.00601 ± 2.3094e-05 (W 1.60062e-05) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00328 ± 5.55908e-05 (W 3.71106e-05) | 0.00347 ± 2.64575e-05 (W 3.72048e-05) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01746 ± 0.000137961 (W 7.32892e-05) | 0.045928 ± 2.04206e-05 (W 9.89308e-05) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.01751 ± 3.46458e-05 (W 9.12869e-05) | 0.020159 ± 9.90471e-05 (W 4.22516e-05) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.044739 ± 0.000110092 (W 0.000150181) | 0.265864 ± 0.00017141 (W 0.000995435) | - | 🔵 colmat 5.94× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.045799 ± 0.00016773 (W 0.000857434) | 0.047139 ± 4.59021e-05 (W 9.0561e-05) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.078297 ± 0.000177455 (W 0.000149503) | 0.089838 ± 8.98907e-05 (W 0.000103779) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.079818 ± 0.000634182 (W 0.00107534) | 0.090207 ± 0.00029144 (W 0.000939503) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.369631 ± 0.000698826 (W 0.00127182) | 0.437646 ± 0.00144989 (W 0.00181094) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.369578 ± 0.000609939 (W 0.00131716) | 0.452654 ± 0.00142861 (W 0.00127352) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.90926 ± 0.00260131 (W 0.00146843) | 1.04447 ± 0.000617093 (W 0.000703704) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.90876 ± 0.00210647 (W 0.000580367) | 1.04638 ± 0.00135875 (W 0.000337522) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011319 ± 0.0001395 (W 8.71625e-05) | 0.00914 ± 2.66521e-05 (W 3.9627e-05) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014069 ± 0.000191051 (W 0.000118187) | 0.01027 ± 8.32666e-05 (W 5.03309e-05) | - | 🔴 colmat 1.37× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.059739 ± 0.000135525 (W 0.000161338) | 0.055598 ± 0.000445234 (W 0.000486989) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.067898 ± 0.000306839 (W 0.00113792) | 0.059939 ± 0.0003002 (W 0.000148728) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.176664 ± 0.000935919 (W 0.00120962) | 0.170474 ± 8.9822e-05 (W 0.000292314) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.224583 ± 0.00115845 (W 0.00138084) | 0.207683 ± 0.000572867 (W 0.00140311) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.013879 ± 2.08167e-05 (W 5.01627e-05) | 0.013499 ± 1.95021e-05 (W 2.52428e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.01381 ± 2.08167e-05 (W 3.24376e-05) | 0.013429 ± 5.2915e-05 (W 3.43118e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.070399 ± 0.000136273 (W 0.000101253) | 0.069458 ± 0.00017706 (W 0.000131766) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.070518 ± 2.39653e-05 (W 0.000106303) | 0.068927 ± 4.94099e-05 (W 9.01011e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.197775 ± 0.00145647 (W 0.00159205) | 0.193324 ± 0.000718659 (W 0.0014562) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.196114 ± 0.000125676 (W 0.000205559) | 0.193925 ± 0.000390359 (W 0.00105402) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00827 ± 0 (W 4.46576e-05) | 0.00688 ± 7e-05 (W 3.55415e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00985 ± 4.04145e-05 (W 7.82828e-05) | 0.00747 ± 2.25167e-05 (W 6.34208e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.031249 ± 0.000160104 (W 0.000125379) | 0.031029 ± 6.02771e-05 (W 0.000187441) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.036629 ± 0.00043501 (W 0.000188306) | 0.03043 ± 0.000135967 (W 8.15475e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.074657 ± 0.000228264 (W 0.000258597) | 0.075847 ± 0.00012178 (W 0.000561784) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.118677 ± 0.000765713 (W 0.00108333) | 0.106196 ± 0.000524523 (W 0.00144901) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.045139 ± 0.000175586 (W 0.000301782) | 0.051328 ± 0.000244643 (W 0.000201743) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.02347 ± 0.000105642 (W 4.99129e-05) | 0.045279 ± 0.000129384 (W 0.000174133) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.164126 ± 0.000246443 (W 0.000915909) | 0.226203 ± 0.00225759 (W 0.00176981) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.150756 ± 0.000295145 (W 0.00119398) | 0.192563 ± 0.00142862 (W 0.00219477) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.350149 ± 0.000483571 (W 0.00175223) | 0.476505 ± 0.00122248 (W 0.00157843) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.386837 ± 0.00108625 (W 0.00201932) | 0.448469 ± 0.000440711 (W 0.0016354) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.023919 ± 0.000255147 (W 0.000119788) | 0.030089 ± 3.21455e-05 (W 0.00018767) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.084367 ± 0.000180306 (W 0.000387647) | 0.142216 ± 0.000295353 (W 0.000919717) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.184545 ± 0.000622913 (W 0.00124025) | 0.306343 ± 0.000394672 (W 0.00131764) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.008369 ± 8.34286e-05 (W 3.97014e-05) | 0.00743 ± 1.48436e-05 (W 4.50244e-05) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01072 ± 6.76782e-05 (W 8.82281e-05) | 0.008289 ± 5.50485e-05 (W 2.26539e-05) | - | 🔴 colmat 1.29× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.030159 ± 0.000200326 (W 7.50333e-05) | 0.032199 ± 0.000127163 (W 8.42923e-05) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.038699 ± 0.000603849 (W 0.00127137) | 0.031779 ± 4.16333e-05 (W 8.58487e-05) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.071278 ± 0.0001253 (W 0.0012725) | 0.078357 ± 0.00116 (W 0.00122756) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.121006 ± 7.8239e-05 (W 0.00176431) | 0.108316 ± 0.000154727 (W 0.00175115) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.130426 ± 0.000348923 (W 0.000765748) | 0.155966 ± 0.000737236 (W 0.0010119) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.130526 ± 0.000439007 (W 0.000781198) | 0.155845 ± 0.000255371 (W 0.00124396) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.608961 ± 0.00111581 (W 0.00176373) | 0.768984 ± 0.00354102 (W 0.00155865) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.612821 ± 0.00379433 (W 0.00200399) | 0.768806 ± 0.00260307 (W 0.00139846) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.412 ± 0.00723934 (W 0.00229009) | 1.924 ± 0.00374546 (W 0.00174068) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.41705 ± 0.00351268 (W 0.00310457) | 1.92596 ± 0.00324218 (W 0.00122874) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.048488 ± 3.27159e-05 (W 0.000179414) | 0.106106 ± 6.17333e-05 (W 0.00103429) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.049109 ± 2.28108e-05 (W 9.29209e-05) | 0.106578 ± 0.000172476 (W 0.000909306) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.198824 ± 0.000176015 (W 0.00136234) | 0.428247 ± 0.00292355 (W 0.00110186) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.199086 ± 0.000210003 (W 0.00108219) | 0.42986 ± 0.00369087 (W 0.00139462) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.503474 ± 0.00123633 (W 0.00202249) | 1.00289 ± 0.00163502 (W 0.000658025) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.504919 ± 0.000797352 (W 0.00150768) | 1.00656 ± 0.00255074 (W 0.00054504) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00075 ± 5.7735e-07 (W 1.48054e-05) | 0.0007 ± 1e-05 (W 1.64317e-05) | 0.00064 ± 5.7735e-06 (W 1.44983e-05) | 🔴 colmat 1.07× slower than `faer` | 🔴 colmat 1.17× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00184 ± 1.18462e-05 (W 2.29608e-05) | 0.00172 ± 1.47309e-05 (W 2.6119e-05) | 0.00167 ± 1.73205e-05 (W 2.47022e-05) | 🔴 colmat 1.07× slower than `faer` | 🔴 colmat 1.10× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00337 ± 3.09892e-05 (W 2.07364e-05) | 0.0032 ± 2.64575e-05 (W 1.89526e-05) | 0.003119 ± 1.51767e-05 (W 2.43967e-05) | 🔴 colmat 1.05× slower than `faer` | 🔴 colmat 1.08× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003299 ± 2.28108e-05 (W 3.60583e-05) | 0.00601 ± 2e-05 (W 1.18406e-05) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00338 ± 2.64575e-05 (W 3.17994e-05) | 0.0035 ± 3.51188e-05 (W 3.78153e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0167 ± 6.1403e-05 (W 9.04063e-05) | 0.046018 ± 3.23574e-05 (W 0.000122173) | - | 🔵 colmat 2.76× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.01733 ± 8.5049e-05 (W 9.03344e-05) | 0.02017 ± 2.00083e-05 (W 5.24052e-05) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.040509 ± 0.00016784 (W 0.000163799) | 0.098847 ± 0.0128502 (W 0.0852763) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.042129 ± 0.000176163 (W 0.000248349) | 0.050038 ± 0.000691655 (W 0.000583319) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.058498 ± 0.000181475 (W 9.58123e-05) | 0.089828 ± 0.00711578 (W 0.000587548) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.058649 ± 0.0001455 (W 0.000161028) | 0.089997 ± 0.000221133 (W 0.000209332) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.201383 ± 0.000742893 (W 0.001293) | 0.424228 ± 0.00118667 (W 0.00147561) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.203055 ± 0.000955794 (W 0.00117499) | 0.424476 ± 0.000230411 (W 0.00121995) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.424709 ± 0.00367758 (W 0.00198446) | 0.979939 ± 0.00126179 (W 0.00117128) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.430647 ± 0.000507885 (W 0.00183929) | 0.983074 ± 0.00234692 (W 0.00079742) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01196 ± 0.000142163 (W 9.35751e-05) | 0.00926 ± 7.2111e-05 (W 9.16635e-05) | - | 🔴 colmat 1.29× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0152 ± 0.00010583 (W 4.94702e-05) | 0.01042 ± 7.63763e-05 (W 7.99762e-05) | - | 🔴 colmat 1.46× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.057828 ± 0.000475581 (W 0.000386218) | 0.055779 ± 3.5171e-05 (W 0.000169364) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.067408 ± 0.00038223 (W 0.000497602) | 0.060109 ± 0.00042981 (W 0.000353682) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.130477 ± 0.00031802 (W 0.000908692) | 0.170394 ± 0.0001948 (W 0.00119663) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.184455 ± 0.000456772 (W 0.00190413) | 0.175335 ± 0.000223361 (W 0.00135956) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00981 ± 9.87269e-05 (W 0.000138066) | 0.013729 ± 8.28674e-05 (W 9.92184e-05) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0095 ± 0.000225019 (W 9.83728e-05) | 0.01367 ± 8.77895e-05 (W 5.28517e-05) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.037099 ± 0.000177764 (W 0.000315959) | 0.036279 ± 0.000181475 (W 0.000288704) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.037779 ± 0.000361432 (W 0.000538164) | 0.036279 ± 8.7178e-05 (W 0.000326806) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.080368 ± 0.00032532 (W 0.00118679) | 0.077538 ± 0.000410533 (W 0.000382589) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.080237 ± 0.000572393 (W 0.000562746) | 0.078048 ± 0.000287112 (W 0.000712181) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00834 ± 5.05008e-05 (W 6.91173e-05) | 0.00686 ± 1.76163e-05 (W 3.41497e-05) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.01002 ± 8.02081e-05 (W 6.72555e-05) | 0.00747 ± 8.02081e-05 (W 5.15781e-05) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.035819 ± 0.000368284 (W 0.000371954) | 0.031429 ± 8.18535e-05 (W 0.000110318) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.035979 ± 0.00019218 (W 0.000441962) | 0.030379 ± 4.3589e-05 (W 0.00102246) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.077658 ± 0.000390577 (W 0.000638878) | 0.078108 ± 3.00056e-05 (W 0.000680645) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.108636 ± 0.000266083 (W 0.000840494) | 0.077738 ± 0.000321217 (W 0.00143403) | - | 🔴 colmat 1.40× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.043429 ± 0.000295353 (W 0.000246553) | 0.060389 ± 0.000467643 (W 0.000714782) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.023959 ± 0.00114303 (W 7.03939e-05) | 0.053999 ± 0.000153731 (W 0.000595736) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.098507 ± 7.57188e-05 (W 0.000451201) | 0.246992 ± 0.000693139 (W 0.00169531) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.116787 ± 0.000155779 (W 0.00150713) | 0.211563 ± 0.00140262 (W 0.00205692) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.253733 ± 0.000676927 (W 0.00145249) | 0.465278 ± 0.00111524 (W 0.00210382) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.32224 ± 0.00284187 (W 0.00273868) | 0.430546 ± 0.00152104 (W 0.00284677) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.02486 ± 0.000359083 (W 0.000392779) | 0.038939 ± 0.000230651 (W 0.00030764) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.057748 ± 0.000452364 (W 0.000210483) | 0.183115 ± 0.00024193 (W 0.00169658) | - | 🔵 colmat 3.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.169755 ± 0.000560744 (W 0.00049) | 0.36232 ± 0.00192485 (W 0.00197121) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0088 ± 5.80259e-05 (W 3.20656e-05) | 0.00744 ± 4.04145e-05 (W 4.84479e-05) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01108 ± 6.05558e-05 (W 9.29263e-05) | 0.00827 ± 4.50925e-05 (W 4.41837e-05) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.029629 ± 2.64575e-05 (W 0.000192842) | 0.031689 ± 0.000323934 (W 0.000103331) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.042289 ± 0.000174738 (W 0.00142771) | 0.032009 ± 0.000266333 (W 0.000161771) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.065189 ± 0.000210658 (W 0.000999625) | 0.079208 ± 0.000470273 (W 0.000766596) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.107216 ± 0.000337471 (W 0.00168837) | 0.079528 ± 0.000240306 (W 0.000416171) | - | 🔴 colmat 1.35× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.129626 ± 0.00157031 (W 0.000196901) | 0.155365 ± 0.000289673 (W 0.00021817) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.130146 ± 0.000356791 (W 0.000243496) | 0.157575 ± 0.00300287 (W 0.00131898) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.471285 ± 0.00943017 (W 0.00144782) | 0.73708 ± 0.000778848 (W 0.00153252) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.476607 ± 0.00397107 (W 0.0018765) | 0.739267 ± 0.000595705 (W 0.001599) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.978329 ± 0.00383998 (W 0.00240042) | 1.79753 ± 0.00488063 (W 0.00273516) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.984053 ± 0.00199089 (W 0.00323945) | 1.79856 ± 0.00575563 (W 0.00224825) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.049718 ± 8.06866e-05 (W 0.000128512) | 0.106137 ± 0.00010583 (W 0.00128037) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.049908 ± 0.000125831 (W 8.11991e-05) | 0.106397 ± 0.000134536 (W 0.000142865) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.207283 ± 0.000591921 (W 0.00140006) | 0.429057 ± 0.00244708 (W 0.0018794) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.206224 ± 0.0011202 (W 0.00108353) | 0.428347 ± 0.00104571 (W 0.00134563) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.472446 ± 0.000530012 (W 0.00211854) | 1.00353 ± 0.00163319 (W 0.000588022) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.470707 ± 0.00248232 (W 0.00242182) | 1.00439 ± 0.00105241 (W 0.00139172) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00092 ± 1.47309e-05 (W 2.29826e-05) | 0.0007 ± 5.7735e-06 (W 1.94936e-05) | 0.00065 ± 5.7735e-06 (W 2.07364e-05) | 🔴 colmat 1.31× slower than `faer` | 🔴 colmat 1.42× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0022 ± 2.51661e-05 (W 2.32852e-05) | 0.00171 ± 1.1547e-05 (W 2.60768e-05) | 0.00166 ± 1.0504e-05 (W 2.51635e-05) | 🔴 colmat 1.29× slower than `faer` | 🔴 colmat 1.33× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00384 ± 1e-05 (W 1.64317e-05) | 0.00316 ± 4.72582e-05 (W 4.04623e-05) | 0.00312 ± 1e-05 (W 1.52709e-05) | 🔴 colmat 1.22× slower than `faer` | 🔴 colmat 1.23× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00327 ± 2.08167e-05 (W 3.64692e-05) | 0.006 ± 1.1547e-05 (W 1.80056e-05) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00329 ± 3.21455e-05 (W 4.17397e-05) | 0.00352 ± 8.5049e-05 (W 2.07364e-05) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01676 ± 0.000176116 (W 0.000100689) | 0.046158 ± 6.94286e-05 (W 0.00011718) | - | 🔵 colmat 2.75× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.017429 ± 8.52467e-05 (W 0.000146319) | 0.019809 ± 0.000350186 (W 0.000942957) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.040829 ± 0.000306431 (W 0.000241471) | 0.078457 ± 0.000102031 (W 0.000197715) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.042709 ± 0.000166494 (W 0.00366673) | 0.050448 ± 0.000193745 (W 0.000214901) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.055148 ± 5.13842e-05 (W 0.000116118) | 0.089858 ± 0.000190316 (W 0.00127374) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.055498 ± 0.000255344 (W 0.000167724) | 0.089897 ± 0.000400791 (W 0.000998176) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.175704 ± 0.000819504 (W 0.00136811) | 0.425396 ± 0.00108183 (W 0.00236992) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.176485 ± 0.000312143 (W 0.00146884) | 0.427387 ± 0.00215767 (W 0.0016647) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.354359 ± 0.00330111 (W 0.00280761) | 0.98563 ± 0.000472024 (W 0.00115618) | - | 🔵 colmat 2.78× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.357829 ± 0.00226007 (W 0.00217467) | 0.989619 ± 0.000374957 (W 0.00127723) | - | 🔵 colmat 2.77× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01199 ± 0.000125527 (W 5.64207e-05) | 0.00915 ± 8.14453e-05 (W 2.56184e-05) | - | 🔴 colmat 1.31× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015 ± 9.29157e-05 (W 0.000124705) | 0.01031 ± 0.000113578 (W 9.49911e-05) | - | 🔴 colmat 1.45× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.060888 ± 0.000395264 (W 0.000283265) | 0.056078 ± 0.000108029 (W 0.00171262) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.069959 ± 0.000530264 (W 0.000626837) | 0.060508 ± 0.000382143 (W 0.00136954) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.132686 ± 0.000821243 (W 0.000635983) | 0.169885 ± 0.000186361 (W 0.000775106) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.184374 ± 0.00251867 (W 0.00254876) | 0.175034 ± 0.000531094 (W 0.00198602) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00752 ± 7.0946e-05 (W 0.000143677) | 0.013789 ± 5.03322e-05 (W 0.00113777) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00705 ± 3.56417e-05 (W 3.87582e-05) | 0.01368 ± 6.65833e-05 (W 8.22393e-05) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.02386 ± 0.000160407 (W 0.000388643) | 0.036719 ± 0.000124808 (W 0.000254598) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.023699 ± 9.07377e-05 (W 0.00034508) | 0.036519 ± 0.000226694 (W 0.000345873) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.049438 ± 0.000570351 (W 0.00118841) | 0.078208 ± 0.000374477 (W 0.000497112) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.049378 ± 0.000235018 (W 0.000443425) | 0.078778 ± 0.000369256 (W 0.00184727) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00842 ± 2.09841e-05 (W 4.36486e-05) | 0.006879 ± 3.61985e-05 (W 2.70592e-05) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.009869 ± 4.33628e-05 (W 9.62149e-05) | 0.0075 ± 7.81025e-05 (W 4.49244e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.036589 ± 0.00024173 (W 0.000334634) | 0.031409 ± 0.000100664 (W 0.000110486) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.035699 ± 0.000140475 (W 0.000229717) | 0.030569 ± 7.02377e-05 (W 0.000354796) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.081908 ± 0.000165036 (W 0.000868263) | 0.078727 ± 0.000465092 (W 0.000339317) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.106816 ± 0.000604632 (W 0.00109997) | 0.077418 ± 0.000463069 (W 0.00121594) | - | 🔴 colmat 1.38× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.043109 ± 0.00037027 (W 0.000445621) | 0.066398 ± 0.000116762 (W 0.000792018) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.024009 ± 0.000185014 (W 4.35006e-05) | 0.059838 ± 0.000352467 (W 0.000543976) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.096227 ± 0.000115902 (W 0.000385966) | 0.271131 ± 0.00235233 (W 0.00342724) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.113077 ± 0.00133717 (W 0.000991319) | 0.238994 ± 0.00150969 (W 0.00167201) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.226684 ± 0.0103896 (W 0.00178701) | 0.512774 ± 0.00209802 (W 0.00394461) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1 [api-inclusive]` | 0.303632 ± 0.000846554 (W 0.00209137) | 0.484626 ± 0.00107968 (W 0.00247829) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.026529 ± 0.000542863 (W 0.000992231) | 0.045269 ± 0.000365117 (W 0.000714814) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.057808 ± 0.000139141 (W 8.99177e-05) | 0.208964 ± 0.00191716 (W 0.00391788) | - | 🔵 colmat 3.61× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.159645 ± 0.00081442 (W 0.000646828) | 0.411529 ± 0.00259832 (W 0.00186868) | - | 🔵 colmat 2.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00877 ± 3.51188e-05 (W 4.74363e-05) | 0.00746 ± 0.000128582 (W 4.27224e-05) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.01111 ± 4.93288e-05 (W 3.695e-05) | 0.00817 ± 3.60555e-05 (W 3.9474e-05) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.029629 ± 0.000200749 (W 0.000237739) | 0.031609 ± 3.4641e-05 (W 0.000107305) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.037529 ± 0.000435469 (W 0.000102127) | 0.032059 ± 8.7178e-05 (W 6.26434e-05) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.069758 ± 0.000727622 (W 0.000682349) | 0.079888 ± 0.00081074 (W 0.000599367) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.107587 ± 0.000428525 (W 0.00145343) | 0.079597 ± 0.000228889 (W 0.00129911) | - | 🔴 colmat 1.35× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.129296 ± 0.000147986 (W 0.000301507) | 0.156685 ± 0.00106355 (W 0.00124755) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.130436 ± 0.000125033 (W 0.000856447) | 0.156095 ± 0.00313841 (W 0.0002805) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.468845 ± 0.00269155 (W 0.0021297) | 0.739207 ± 0.000384566 (W 0.00280571) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.470677 ± 0.00111527 (W 0.00195753) | 0.74038 ± 0.00067112 (W 0.00185664) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.96358 ± 0.0033917 (W 0.00146617) | 1.81106 ± 0.000478906 (W 0.0031997) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.96633 ± 0.00259198 (W 0.00159466) | 1.81099 ± 0.00178362 (W 0.00297138) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.049618 ± 0.00022974 (W 0.000150776) | 0.106207 ± 0.000431544 (W 7.72412e-05) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.049919 ± 8.44808e-05 (W 7.35955e-05) | 0.106327 ± 0.00048645 (W 0.00174534) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.211643 ± 0.000520279 (W 0.00111583) | 0.427017 ± 0.00118074 (W 0.00178615) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.211194 ± 0.000646632 (W 0.00130475) | 0.428528 ± 0.000960989 (W 0.00151407) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.487945 ± 0.0011017 (W 0.00160243) | 1.00073 ± 0.00408326 (W 0.000772674) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.486955 ± 0.00163081 (W 0.00226447) | 1.00105 ± 0.00205475 (W 0.00030997) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00091 ± 1e-05 (W 1.30384e-05) | 0.0007 ± 0 (W 1.78885e-05) | 0.00064 ± 0 (W 1.64317e-05) | 🔴 colmat 1.30× slower than `faer` | 🔴 colmat 1.42× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00221 ± 5.7735e-06 (W 2.77489e-05) | 0.00174 ± 1e-05 (W 3.22831e-05) | 0.00165 ± 1.52753e-05 (W 2.54951e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.34× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003879 ± 2.0664e-05 (W 1.81163e-05) | 0.00314 ± 2.64575e-05 (W 2.20726e-05) | 0.00311 ± 1e-05 (W 1.41421e-05) | 🔴 colmat 1.24× slower than `faer` | 🔴 colmat 1.25× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00282 ± 1e-05 (W 2.58844e-05) | 0.00296 ± 1.52753e-05 (W 1.64317e-05) | 0.004629 ± 3.61985e-05 (W 4.49444e-05) | · | 🔵 colmat 1.64× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00328 ± 2.51661e-05 (W 4.94186e-05) | 0.00347 ± 2.08167e-05 (W 2.50998e-05) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.015459 ± 2.28108e-05 (W 6.67248e-05) | 0.01821 ± 0.000332916 (W 2.74281e-05) | 0.026669 ± 0.000295026 (W 0.000179742) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.017219 ± 0.000118196 (W 5.01228e-05) | 0.019819 ± 0.000301164 (W 6.75152e-05) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.040349 ± 0.000151438 (W 7.55533e-05) | 0.043789 ± 0.000175214 (W 5.28422e-05) | 0.077958 ± 0.000413244 (W 9.38792e-05) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.93× faster than `lapack-mkl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.045198 ± 0.000312623 (W 0.00101775) | 0.046518 ± 0.000173177 (W 8.0661e-05) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.078148 ± 3.61156e-05 (W 0.000951426) | 0.089958 ± 0.00037486 (W 0.000220407) | 0.093327 ± 8.84892e-05 (W 0.000250719) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.078388 ± 4.04145e-05 (W 0.000186044) | 0.090187 ± 0.000236291 (W 0.000170517) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.369288 ± 0.000722348 (W 0.00169713) | 0.436505 ± 0.000692218 (W 0.0012807) | 0.507633 ± 0.000221939 (W 0.00131477) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.37× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.369098 ± 0.000845119 (W 0.00137467) | 0.453116 ± 0.00147779 (W 0.00247871) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.909441 ± 0.0010351 (W 0.00669097) | 1.04425 ± 0.00170289 (W 0.000814276) | 1.27644 ± 0.00292975 (W 0.00189588) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.90897 ± 0.00183686 (W 0.000827329) | 1.04642 ± 0.00137909 (W 0.000538492) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011439 ± 1.53948e-05 (W 4.52791e-05) | 0.009079 ± 9.50438e-06 (W 5.24995e-05) | 0.01449 ± 0.000120403 (W 5.57162e-05) | 🔴 colmat 1.26× slower than `faer` | 🔵 colmat 1.27× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01277 ± 0.000130767 (W 5.98105e-05) | 0.01017 ± 3.82143e-05 (W 4.53023e-05) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.060318 ± 6.08276e-05 (W 0.000185957) | 0.055379 ± 0.00338633 (W 0.000112353) | 0.097647 ± 0.000443441 (W 0.00111109) | 🔴 colmat 1.09× slower than `faer` (marginal: spread 0.70 of the difference) | 🔵 colmat 1.62× faster than `lapack-mkl` | `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.064108 ± 0.00029955 (W 0.000143154) | 0.058688 ± 0.000309317 (W 0.000205913) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.180694 ± 0.0109112 (W 0.00106131) | 0.170434 ± 0.000334099 (W 0.000984748) | 0.31035 ± 0.000693395 (W 0.00116192) | · | 🔵 colmat 1.72× faster than `lapack-mkl` | `!` between-start range over 5% |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.214015 ± 0.00124982 (W 0.00196612) | 0.206055 ± 0.000765583 (W 0.00116883) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.013869 ± 4.17971e-05 (W 5.64119e-05) | 0.013489 ± 5.55908e-05 (W 7.36566e-05) | 0.031509 ± 2.08167e-05 (W 4.82701e-05) | · | 🔵 colmat 2.27× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.011199 ± 0.000999753 (W 1.44326e-05) | 0.01338 ± 5.19615e-06 (W 7.24313e-05) | 0.021369 ± 6.60328e-05 (W 1.87136e-05) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.91× faster than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.069869 ± 9.17624e-05 (W 9.1248e-05) | 0.069048 ± 0.000122882 (W 7.4493e-05) | 0.168906 ± 0.00019708 (W 0.00124777) | · | 🔵 colmat 2.42× faster than `lapack-mkl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.069998 ± 4.50925e-05 (W 7.18491e-05) | 0.068287 ± 0.000270075 (W 0.000140866) | 0.179714 ± 0.000286754 (W 0.00107716) | · | 🔵 colmat 2.57× faster than `lapack-mkl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.193483 ± 0.000985901 (W 0.00126871) | 0.191826 ± 0.000252881 (W 0.00136346) | 0.467715 ± 0.000499652 (W 0.00210481) | · | 🔵 colmat 2.42× faster than `lapack-mkl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.193814 ± 0.000729603 (W 0.00152206) | 0.192575 ± 0.000900282 (W 0.00151203) | 0.453636 ± 0.00149717 (W 0.00190624) | · | 🔵 colmat 2.34× faster than `lapack-mkl` |  |
| `lu/n32/b1` | 0.007449 ± 9.0813e-05 (W 4.55653e-05) | 0.00628 ± 4.16333e-05 (W 5.30867e-05) | 0.01062 ± 5.53203e-05 (W 3.57253e-05) | 🔴 colmat 1.19× slower than `faer` | 🔵 colmat 1.43× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008799 ± 1.53948e-05 (W 0.000102617) | 0.00761 ± 8.5049e-05 (W 7.03506e-05) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.028419 ± 0.000106616 (W 7.12741e-05) | 0.025919 ± 0.000364966 (W 6.10737e-05) | 0.052389 ± 0.000450511 (W 7.72671e-05) | 🔴 colmat 1.10× slower than `faer` | 🔵 colmat 1.84× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.032439 ± 9.29157e-05 (W 7.30068e-05) | 0.029199 ± 0.000106616 (W 7.1624e-05) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.069697 ± 8.50353e-05 (W 0.000177331) | 0.065528 ± 0.000308049 (W 0.000323085) | 0.133235 ± 0.000135356 (W 0.000978605) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.91× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.107846 ± 0.00055951 (W 0.00130535) | 0.103926 ± 0.00174101 (W 0.00127513) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.020389 ± 6.30106e-05 (W 3.05172e-05) | 0.043459 ± 0.000185831 (W 0.000237719) | 0.034549 ± 0.000325013 (W 8.70057e-05) | 🔵 colmat 2.13× faster than `faer` | 🔵 colmat 1.69× faster than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.022219 ± 1.95021e-05 (W 4.23592e-05) | 0.045279 ± 0.000466766 (W 0.000269634) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.128326 ± 0.000566737 (W 0.00110993) | 0.171184 ± 0.000168921 (W 0.00110798) | 0.28032 ± 0.000733839 (W 0.00218007) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 2.18× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.146767 ± 0.00143856 (W 0.00210555) | 0.190905 ± 0.000789846 (W 0.00250807) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.309643 ± 0.00231895 (W 0.00134943) | 0.385118 ± 0.00251234 (W 0.00225203) | 0.760666 ± 0.000395597 (W 0.00157576) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 2.46× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.377251 ± 0.00104294 (W 0.0028982) | 0.444165 ± 0.0019699 (W 0.00155606) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00954 ± 1.1547e-05 (W 3.77001e-05) | 0.022759 ± 0.000181751 (W 0.00016695) | 0.0167 ± 0.000243106 (W 6.25724e-05) | 🔵 colmat 2.39× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.050428 ± 0.000349424 (W 0.00105528) | 0.095287 ± 0.000274245 (W 0.000574398) | 0.099817 ± 0.000559381 (W 8.81884e-05) | 🔵 colmat 1.89× faster than `faer` | 🔵 colmat 1.98× faster than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.146945 ± 0.000268814 (W 0.00030025) | 0.219295 ± 0.00065815 (W 0.00128915) | 0.30778 ± 0.000588781 (W 0.001223) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.00798 ± 7.05006e-05 (W 5.35742e-05) | 0.00694 ± 7.23418e-05 (W 4.91935e-05) | 0.01058 ± 6.30106e-05 (W 4.06227e-05) | 🔴 colmat 1.15× slower than `faer` | 🔵 colmat 1.33× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009459 ± 2.3388e-05 (W 5.4243e-05) | 0.0082 ± 4.50925e-05 (W 2.73532e-05) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.03123 ± 0.000198789 (W 0.000207611) | 0.027839 ± 0.000171553 (W 5.73254e-05) | 0.051038 ± 0.0001138 (W 8.73499e-05) | 🔴 colmat 1.12× slower than `faer` | 🔵 colmat 1.63× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.034509 ± 0.000198578 (W 0.000203917) | 0.031049 ± 2.94449e-05 (W 3.89127e-05) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.072397 ± 0.000598966 (W 0.000680007) | 0.068238 ± 0.000146289 (W 5.70544e-05) | 0.128557 ± 0.000241448 (W 0.000356469) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.78× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.111596 ± 0.000172128 (W 0.00188682) | 0.106168 ± 0.000252668 (W 0.00170909) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.128876 ± 0.000314013 (W 0.00110654) | 0.153205 ± 0.000642091 (W 0.00113035) | 0.162585 ± 0.000316223 (W 0.0013142) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.129476 ± 9.60434e-05 (W 0.00122808) | 0.154666 ± 0.0011345 (W 0.00095179) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.60173 ± 0.000599792 (W 0.00175976) | 0.789402 ± 0.000655578 (W 0.00126096) | 0.851581 ± 0.00151229 (W 0.00136505) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 1.42× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.60243 ± 0.00119394 (W 0.00211456) | 0.787984 ± 0.000862722 (W 0.00147493) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.408 ± 0.00162964 (W 0.00262845) | 1.8972 ± 0.00333973 (W 0.000826633) | 2.20684 ± 0.00449045 (W 0.00229922) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.57× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.41084 ± 0.0021567 (W 0.00241409) | 1.89564 ± 0.00115128 (W 0.00142469) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.049088 ± 0.000165772 (W 6.09918e-05) | 0.104836 ± 0.000183957 (W 0.00119706) | 0.063628 ± 7.3146e-05 (W 0.000120955) | 🔵 colmat 2.14× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.049789 ± 8.55589e-05 (W 0.000137929) | 0.105027 ± 0.000235373 (W 0.00109961) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.196463 ± 0.000833074 (W 0.00151412) | 0.411637 ± 0.000428327 (W 0.0014101) | 0.304063 ± 0.00133788 (W 0.001443) | 🔵 colmat 2.10× faster than `faer` | 🔵 colmat 1.55× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.196754 ± 0.00025815 (W 0.000978773) | 0.412191 ± 0.000950277 (W 0.00131942) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.501163 ± 0.000702069 (W 0.00158258) | 0.987017 ± 0.00205162 (W 0.000546216) | 0.852072 ± 0.00484004 (W 0.00135953) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.501949 ± 0.00143836 (W 0.00192924) | 0.987447 ± 0.00234924 (W 0.000701306) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00077 ± 1e-05 (W 2.19089e-05) | 0.00081 ± 1e-05 (W 1.22474e-05) | 0.00063 ± 0 (W 2.60768e-05) | 🔵 colmat 1.05× faster than `faer` | 🔴 colmat 1.22× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0019 ± 1.52753e-05 (W 1.51658e-05) | 0.00201 ± 2.82902e-05 (W 1.63156e-05) | 0.00164 ± 5.7735e-06 (W 2.60768e-05) | 🔵 colmat 1.06× faster than `faer` | 🔴 colmat 1.16× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00322 ± 1e-05 (W 2.91548e-05) | 0.00348 ± 1.48436e-05 (W 1.51658e-05) | 0.00312 ± 1e-05 (W 1.31985e-05) | 🔵 colmat 1.08× faster than `faer` | · | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00276 ± 1.1547e-05 (W 1.74986e-05) | 0.00299 ± 5.19615e-06 (W 1.33116e-05) | 0.00456 ± 3.49333e-05 (W 3.50714e-05) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.65× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00328 ± 2e-05 (W 4.3359e-05) | 0.00346 ± 5.7735e-06 (W 3.79895e-05) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01499 ± 5.25389e-05 (W 0.000102158) | 0.01837 ± 1.09697e-05 (W 0.000916225) | 0.027349 ± 0.000147309 (W 7.09591e-05) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 1.82× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.01716 ± 3.9879e-05 (W 0.000193603) | 0.01978 ± 6.08276e-05 (W 4.65865e-05) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.036719 ± 0.000106209 (W 0.000159494) | 0.044699 ± 0.000142339 (W 0.000280552) | 0.077948 ± 0.00017674 (W 0.000992562) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 2.12× faster than `lapack-mkl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.041879 ± 0.000257358 (W 0.000168672) | 0.049498 ± 0.000212439 (W 0.00104018) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.058568 ± 0.000135746 (W 0.000119416) | 0.090268 ± 0.000252528 (W 0.000134128) | 0.096987 ± 0.00043827 (W 0.00135849) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.66× faster than `lapack-mkl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.058659 ± 0.000190316 (W 0.000103827) | 0.090257 ± 0.000408085 (W 0.00141639) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.201274 ± 0.00196762 (W 0.00127404) | 0.423836 ± 0.000869495 (W 0.00177932) | 0.344571 ± 0.00125336 (W 0.0018973) | 🔵 colmat 2.11× faster than `faer` | 🔵 colmat 1.71× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.202454 ± 0.000896158 (W 0.00113687) | 0.424107 ± 0.00108394 (W 0.00154922) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.428457 ± 0.0023597 (W 0.00133654) | 0.978903 ± 0.000824591 (W 0.000881232) | 0.716377 ± 0.00255384 (W 0.0020718) | 🔵 colmat 2.28× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.431549 ± 0.00187737 (W 0.00213187) | 0.983103 ± 0.00135608 (W 0.000606113) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.012249 ± 6.46555e-05 (W 3.83445e-05) | 0.00915 ± 2.3094e-05 (W 4.28859e-05) | 0.014489 ± 2.09841e-05 (W 3.25315e-05) | 🔴 colmat 1.34× slower than `faer` | 🔵 colmat 1.18× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01363 ± 5.03322e-05 (W 8.87029e-05) | 0.01029 ± 5.2915e-05 (W 3.76205e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.059108 ± 0.000718707 (W 0.000508633) | 0.055939 ± 0.000215423 (W 0.00123871) | 0.097787 ± 0.000125033 (W 0.000208725) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.65× faster than `lapack-mkl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.063508 ± 0.000353695 (W 0.000311817) | 0.058678 ± 0.000240069 (W 0.000140834) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.135645 ± 0.000202273 (W 0.000959261) | 0.169955 ± 0.000245561 (W 0.00111243) | 0.31109 ± 0.0010298 (W 0.00233017) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 2.29× faster than `lapack-mkl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.171665 ± 0.00164072 (W 0.00114933) | 0.173415 ± 0.000424166 (W 0.00118233) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00955 ± 9.07377e-05 (W 0.000122732) | 0.01369 ± 9.59705e-05 (W 6.14272e-05) | 0.01195 ± 0.000133167 (W 0.000143994) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009859 ± 0.000101589 (W 0.000145885) | 0.013549 ± 9.07377e-05 (W 4.61552e-05) | 0.021209 ± 0.000176154 (W 1.71843e-05) | 🔵 colmat 1.37× faster than `faer` | 🔵 colmat 2.15× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.037289 ± 0.000205994 (W 0.000162327) | 0.036048 ± 0.000300899 (W 0.000297271) | 0.059948 ± 0.000398498 (W 0.00110318) | · | 🔵 colmat 1.61× faster than `lapack-mkl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.037069 ± 9.53939e-05 (W 0.000479875) | 0.036229 ± 0.000140119 (W 0.000274645) | 0.071718 ± 0.000360879 (W 0.000296405) | · | 🔵 colmat 1.93× faster than `lapack-mkl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.079167 ± 0.000470029 (W 0.000255232) | 0.076497 ± 0.000211204 (W 0.000398397) | 0.142306 ± 0.00040207 (W 0.000540022) | · | 🔵 colmat 1.80× faster than `lapack-mkl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.078727 ± 0.000418004 (W 0.000385155) | 0.076458 ± 0.000314993 (W 0.000201105) | 0.149856 ± 0.000168028 (W 0.00126649) | · | 🔵 colmat 1.90× faster than `lapack-mkl` |  |
| `lu/n32/b1` | 0.00757 ± 1.52753e-05 (W 4.66926e-05) | 0.00638 ± 0.000145717 (W 2.87958e-05) | 0.010689 ± 6.6335e-05 (W 6.4933e-05) | 🔴 colmat 1.19× slower than `faer` | 🔵 colmat 1.41× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00903 ± 1.00167e-05 (W 8.93823e-05) | 0.00767 ± 9.84886e-05 (W 3.00865e-05) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.028469 ± 0.000332014 (W 0.000183639) | 0.026229 ± 8.3865e-05 (W 5.0013e-05) | 0.052709 ± 0.000128376 (W 0.000159029) | 🔴 colmat 1.09× slower than `faer` | 🔵 colmat 1.85× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.032729 ± 0.000175594 (W 0.00021312) | 0.029349 ± 5.19615e-05 (W 6.10737e-05) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.057248 ± 0.000450911 (W 0.000399812) | 0.068108 ± 0.000237978 (W 0.000961613) | 0.133386 ± 0.000659869 (W 0.00122856) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 2.33× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.097948 ± 0.000909731 (W 0.00122517) | 0.075328 ± 0.000603523 (W 0.00137814) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.020369 ± 6.65833e-05 (W 3.55992e-05) | 0.052178 ± 0.00011739 (W 0.000271332) | 0.034089 ± 0.000338004 (W 7.96888e-05) | 🔵 colmat 2.56× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.022619 ± 0.000147309 (W 6.7047e-05) | 0.054228 ± 0.000391915 (W 0.000796567) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.088737 ± 0.000379667 (W 0.000256055) | 0.188734 ± 5.80259e-05 (W 0.00119636) | 0.172055 ± 0.00115812 (W 0.00120214) | 🔵 colmat 2.13× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.113367 ± 0.000793682 (W 0.00172692) | 0.209544 ± 0.000532325 (W 0.00302991) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.234582 ± 0.000743174 (W 0.00130899) | 0.356019 ± 0.00291149 (W 0.00187467) | 0.470165 ± 0.0012926 (W 0.00223475) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 2.00× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.31422 ± 0.000687402 (W 0.00262453) | 0.428127 ± 0.000961925 (W 0.0026778) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00957 ± 4.52806e-05 (W 3.71255e-05) | 0.030799 ± 0.000238607 (W 0.000334318) | 0.016989 ± 0.000309823 (W 8.74031e-05) | 🔵 colmat 3.22× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.050559 ± 0.000251834 (W 7.32687e-05) | 0.133805 ± 0.000749338 (W 0.00101357) | 0.099587 ± 0.000679527 (W 9.01454e-05) | 🔵 colmat 2.65× faster than `faer` | 🔵 colmat 1.97× faster than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.157796 ± 0.000987755 (W 0.000674641) | 0.265813 ± 0.00125647 (W 0.00120584) | 0.307972 ± 0.00180122 (W 0.00206825) | 🔵 colmat 1.68× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.0084 ± 7.02377e-05 (W 4.15355e-05) | 0.00699 ± 9.71253e-05 (W 6.8119e-05) | 0.010579 ± 5.53203e-05 (W 4.53244e-05) | 🔴 colmat 1.20× slower than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00985 ± 2.64575e-05 (W 7.62122e-05) | 0.00818 ± 1.52753e-05 (W 2.93803e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.028749 ± 2.88675e-05 (W 0.000257546) | 0.0273 ± 2.46374e-05 (W 3.35455e-05) | 0.050999 ± 3.95011e-05 (W 0.000148318) | 🔴 colmat 1.05× slower than `faer` | 🔵 colmat 1.77× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.038438 ± 0.000169589 (W 0.000248972) | 0.030669 ± 0.000143643 (W 6.76018e-05) | - | 🔴 colmat 1.25× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.058628 ± 8.03762e-05 (W 0.0011923) | 0.070078 ± 0.000861371 (W 0.00110214) | 0.128766 ± 0.000275953 (W 0.00035344) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 2.20× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.098048 ± 0.0001138 (W 0.00172208) | 0.077738 ± 0.00155095 (W 0.00194913) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.127696 ± 0.000225019 (W 0.000301048) | 0.153645 ± 0.000700783 (W 0.00127727) | 0.169035 ± 0.000169726 (W 0.00145572) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.128686 ± 0.000225434 (W 0.000193132) | 0.154406 ± 0.000515296 (W 0.00138316) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.463008 ± 0.000948961 (W 0.00134059) | 0.757327 ± 0.000633107 (W 0.00138433) | 0.64955 ± 0.00151066 (W 0.00140416) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.467685 ± 0.000742648 (W 0.00178684) | 0.759249 ± 0.0013525 (W 0.00161927) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.979933 ± 0.00363648 (W 0.00188365) | 1.7676 ± 0.0165472 (W 0.00395854) | 1.60663 ± 0.00232279 (W 0.00328141) | 🔵 colmat 1.80× faster than `faer` | 🔵 colmat 1.64× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 0.986498 ± 0.00338661 (W 0.00190575) | 1.76946 ± 0.014012 (W 0.00212807) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.050338 ± 0.000251131 (W 0.000238563) | 0.104737 ± 6.1e-05 (W 0.00157352) | 0.063408 ± 0.0001705 (W 0.00121361) | 🔵 colmat 2.08× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.050518 ± 0.000301143 (W 9.1117e-05) | 0.104968 ± 0.000120112 (W 0.000117993) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.203993 ± 0.000959077 (W 0.000864863) | 0.410269 ± 0.000638045 (W 0.00228168) | 0.302892 ± 0.000950786 (W 0.00124386) | 🔵 colmat 2.01× faster than `faer` | 🔵 colmat 1.48× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.204124 ± 0.000609945 (W 0.000844245) | 0.410827 ± 0.00106495 (W 0.00150896) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.471485 ± 0.00282824 (W 0.00124963) | 0.986809 ± 0.000553593 (W 0.00167757) | 0.852787 ± 0.00179511 (W 0.000615687) | 🔵 colmat 2.09× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.467617 ± 0.00350419 (W 0.00138006) | 0.98545 ± 0.0014694 (W 0.00323469) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0009 ± 0 (W 2.07364e-05) | 0.00081 ± 0 (W 1.5975e-05) | 0.00064 ± 5.7735e-06 (W 2.38747e-05) | 🔴 colmat 1.11× slower than `faer` | 🔴 colmat 1.41× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00223 ± 5.7735e-06 (W 2.30217e-05) | 0.00201 ± 5.7735e-06 (W 2.23607e-05) | 0.00165 ± 5.7735e-06 (W 2.07364e-05) | 🔴 colmat 1.11× slower than `faer` | 🔴 colmat 1.35× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00367 ± 2e-05 (W 2.61572e-05) | 0.0035 ± 2.64575e-05 (W 2.31991e-05) | 0.00311 ± 0 (W 1.64317e-05) | · | 🔴 colmat 1.18× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00277 ± 5.68624e-05 (W 1.64317e-05) | 0.00303 ± 1.1547e-05 (W 2.45194e-05) | 0.00465 ± 3.27159e-05 (W 2.92438e-05) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.68× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003289 ± 3.04467e-05 (W 5.21459e-05) | 0.00348 ± 3.98372e-05 (W 2.54951e-05) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.015139 ± 5.52298e-05 (W 0.000117708) | 0.01846 ± 0.000196011 (W 5.13449e-05) | 0.027499 ± 0.0004531 (W 0.000127547) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.82× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0172 ± 1.12694e-05 (W 0.0001256) | 0.019839 ± 0.000104882 (W 4.12468e-05) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.037329 ± 6.80686e-05 (W 0.000311416) | 0.044859 ± 0.00033551 (W 0.000174526) | 0.078487 ± 0.000161091 (W 8.54242e-05) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 2.10× faster than `lapack-mkl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.041919 ± 0.000273191 (W 0.000205339) | 0.049638 ± 9.11976e-05 (W 0.000230344) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.055079 ± 0.000608605 (W 9.68055e-05) | 0.090167 ± 0.003103 (W 0.000171867) | 0.097618 ± 0.000339554 (W 0.00103659) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-mkl` | `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.055159 ± 0.000602022 (W 0.000115841) | 0.089987 ± 0.00012079 (W 0.000238387) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.174565 ± 0.000558062 (W 0.00130452) | 0.424778 ± 0.00175229 (W 0.0022315) | 0.32853 ± 0.00162762 (W 0.00193192) | 🔵 colmat 2.43× faster than `faer` | 🔵 colmat 1.88× faster than `lapack-mkl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.175325 ± 0.000661859 (W 0.000466503) | 0.425517 ± 0.00103682 (W 0.00185851) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.35456 ± 0.00202088 (W 0.00151332) | 0.98481 ± 0.000673143 (W 0.00146611) | 0.64627 ± 0.00380152 (W 0.00224597) | 🔵 colmat 2.78× faster than `faer` | 🔵 colmat 1.82× faster than `lapack-mkl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.35782 ± 0.000527106 (W 0.00233014) | 0.988362 ± 0.000713146 (W 0.00115736) | - | 🔵 colmat 2.76× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01221 ± 2.0502e-05 (W 3.91574e-05) | 0.00913 ± 2.08167e-05 (W 5.80017e-05) | 0.01458 ± 0.000157162 (W 2.96699e-05) | 🔴 colmat 1.34× slower than `faer` | 🔵 colmat 1.19× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01365 ± 8.66025e-05 (W 4.53905e-05) | 0.010329 ± 4.0501e-05 (W 5.13838e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.062149 ± 0.000346425 (W 0.000397278) | 0.056098 ± 1.73205e-05 (W 0.000194513) | 0.098066 ± 0.000120379 (W 0.00120329) | 🔴 colmat 1.11× slower than `faer` | 🔵 colmat 1.58× faster than `lapack-mkl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.066198 ± 0.00053799 (W 0.00102432) | 0.058488 ± 4.93288e-05 (W 0.000178391) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.136897 ± 0.00151682 (W 0.00075025) | 0.170084 ± 0.000317459 (W 0.00123951) | 0.31064 ± 0.000699582 (W 0.00162898) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 2.27× faster than `lapack-mkl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.176674 ± 0.00157988 (W 0.00206174) | 0.173925 ± 0.000633114 (W 0.00140017) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00733 ± 7.63763e-05 (W 9.62808e-05) | 0.013709 ± 5.5109e-05 (W 5.19548e-05) | 0.01213 ± 0.000280951 (W 0.00052264) | 🔵 colmat 1.87× faster than `faer` | 🔵 colmat 1.65× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.007409 ± 4.00042e-05 (W 9.76217e-05) | 0.013559 ± 2.00083e-05 (W 6.82078e-05) | 0.021079 ± 0.000268217 (W 4.30604e-05) | 🔵 colmat 1.83× faster than `faer` | 🔵 colmat 2.85× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.023729 ± 0.00018393 (W 0.000182948) | 0.036779 ± 7.23418e-05 (W 0.00150252) | 0.063528 ± 0.000122882 (W 0.000558149) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 2.68× faster than `lapack-mkl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.023709 ± 9.58662e-05 (W 0.000148324) | 0.036869 ± 0.000234307 (W 0.000200823) | 0.052819 ± 4.78226e-05 (W 0.000365161) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 2.23× faster than `lapack-mkl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.048388 ± 0.000254893 (W 0.000291023) | 0.077468 ± 0.000303699 (W 0.00115594) | 0.088568 ± 0.000272746 (W 0.000560405) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 1.83× faster than `lapack-mkl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.048428 ± 0.000245256 (W 0.000430832) | 0.077307 ± 0.000440031 (W 0.000690588) | 0.097837 ± 0.00586989 (W 0.000701517) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n32/b1` | 0.00758 ± 3.56417e-05 (W 5.73864e-05) | 0.006439 ± 2.47049e-05 (W 5.50745e-05) | 0.01059 ± 1e-05 (W 7.77001e-05) | 🔴 colmat 1.18× slower than `faer` | 🔵 colmat 1.40× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00897 ± 2.95014e-05 (W 6.02677e-05) | 0.00757 ± 2.0664e-05 (W 5.94744e-05) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02833 ± 9.21973e-05 (W 0.000262685) | 0.026009 ± 0.000161494 (W 5.70088e-05) | 0.052338 ± 0.000102211 (W 0.000168467) | 🔴 colmat 1.09× slower than `faer` | 🔵 colmat 1.85× faster than `lapack-mkl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.032299 ± 0.000110604 (W 0.00019608) | 0.029359 ± 0.000193132 (W 8.26438e-05) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.056108 ± 0.000366947 (W 0.000483025) | 0.068108 ± 4.72582e-05 (W 0.000641342) | 0.132947 ± 0.000545146 (W 0.00171411) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.097857 ± 0.000399074 (W 0.000943177) | 0.075507 ± 0.000263661 (W 0.00128081) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.02055 ± 0.000160104 (W 5.34256e-05) | 0.057338 ± 0.000145717 (W 0.000506835) | 0.034379 ± 0.000232451 (W 0.00019308) | 🔵 colmat 2.79× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-mkl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.02275 ± 0.000144627 (W 4.51354e-05) | 0.059688 ± 0.000309726 (W 0.00112246) | - | 🔵 colmat 2.62× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.084408 ± 7.00024e-05 (W 0.000316015) | 0.217864 ± 0.000929857 (W 0.00175202) | 0.153405 ± 0.000523116 (W 0.000304216) | 🔵 colmat 2.58× faster than `faer` | 🔵 colmat 1.82× faster than `lapack-mkl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.110047 ± 0.000903881 (W 0.00115002) | 0.241203 ± 0.00116465 (W 0.00268252) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.209774 ± 0.0010504 (W 0.00218151) | 0.413337 ± 0.00176901 (W 0.00273796) | 0.409117 ± 0.000241028 (W 0.0025081) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-mkl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.296341 ± 0.00109551 (W 0.0019977) | 0.484855 ± 0.00184825 (W 0.00241025) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00963 ± 0.000105357 (W 3.12922e-05) | 0.037409 ± 0.000424421 (W 0.000317424) | 0.01658 ± 3.05014e-05 (W 6.23001e-05) | 🔵 colmat 3.88× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.050279 ± 0.000240666 (W 4.98829e-05) | 0.163386 ± 0.000915516 (W 0.00150584) | 0.098877 ± 0.000125831 (W 0.00136715) | 🔵 colmat 3.25× faster than `faer` | 🔵 colmat 1.97× faster than `lapack-mkl` |  |
| `qr_factor/n96/b1` | 0.149985 ± 0.000124808 (W 0.000488208) | 0.320191 ± 0.00220777 (W 0.00243989) | 0.307321 ± 0.00033126 (W 0.00183656) | 🔵 colmat 2.13× faster than `faer` | 🔵 colmat 2.05× faster than `lapack-mkl` |  |
| `solve/n32/b1` | 0.00837 ± 9.86577e-05 (W 6.60167e-05) | 0.00695 ± 1.52753e-05 (W 2.40832e-05) | 0.01053 ± 4.3589e-05 (W 5.71341e-05) | 🔴 colmat 1.20× slower than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00979 ± 5.80259e-05 (W 3.63359e-05) | 0.00816 ± 2.04206e-05 (W 4.56749e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.02849 ± 0.000295652 (W 0.000232921) | 0.027529 ± 0.000202073 (W 8.11246e-05) | 0.050669 ± 7.2111e-05 (W 0.000120732) | · | 🔵 colmat 1.78× faster than `lapack-mkl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.033089 ± 0.000191398 (W 0.000201237) | 0.030989 ± 7.29178e-05 (W 8.79204e-05) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.056708 ± 0.000472144 (W 0.000498955) | 0.071748 ± 0.000102632 (W 0.00108371) | 0.128716 ± 0.000340172 (W 0.00104884) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 2.27× faster than `lapack-mkl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.098397 ± 0.000427551 (W 0.00169401) | 0.077798 ± 0.000626445 (W 0.0021914) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.128087 ± 0.000509058 (W 0.000304631) | 0.153466 ± 0.00102816 (W 0.00132257) | 0.168106 ± 0.000764912 (W 0.00173721) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-mkl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.129166 ± 0.000176808 (W 0.000418483) | 0.153655 ± 0.00664851 (W 0.00126365) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.462797 ± 0.00106397 (W 0.00200374) | 0.761366 ± 0.00145375 (W 0.0014739) | 0.622481 ± 0.00242475 (W 0.00289061) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-mkl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.463835 ± 0.00254431 (W 0.00289537) | 0.762576 ± 0.00064329 (W 0.00194989) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.962664 ± 0.00201409 (W 0.00181391) | 1.78902 ± 0.00387627 (W 0.0023264) | 1.48754 ± 0.00259578 (W 0.00370605) | 🔵 colmat 1.86× faster than `faer` | 🔵 colmat 1.55× faster than `lapack-mkl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 0.967009 ± 0.00169381 (W 0.00125128) | 1.79085 ± 0.00666939 (W 0.00280321) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.050329 ± 6.40859e-05 (W 0.000146316) | 0.104897 ± 0.00110849 (W 0.000842633) | 0.062408 ± 7.63763e-05 (W 0.000113451) | 🔵 colmat 2.08× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-mkl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.050629 ± 7.84241e-05 (W 0.000119935) | 0.104867 ± 0.00130823 (W 0.00125518) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.209234 ± 0.000743929 (W 0.000678883) | 0.410367 ± 0.00163458 (W 0.00158667) | 0.30104 ± 0.00126917 (W 0.00169537) | 🔵 colmat 1.96× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-mkl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.209353 ± 0.000992331 (W 0.00126648) | 0.411269 ± 0.000505241 (W 0.00154461) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.487475 ± 0.00185524 (W 0.0020198) | 0.985913 ± 0.00613551 (W 0.000974128) | 0.848156 ± 0.000820605 (W 0.00199594) | 🔵 colmat 2.02× faster than `faer` | 🔵 colmat 1.74× faster than `lapack-mkl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.487665 ± 0.0018723 (W 0.00209416) | 0.984432 ± 0.00702691 (W 0.000855673) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00091 ± 1.52753e-05 (W 2.12132e-05) | 0.0008 ± 5.7735e-06 (W 2.30217e-05) | 0.00063 ± 0 (W 1.73205e-05) | 🔴 colmat 1.14× slower than `faer` | 🔴 colmat 1.44× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0022 ± 2e-05 (W 2.94958e-05) | 0.00202 ± 0 (W 1.78885e-05) | 0.00164 ± 0 (W 1.78885e-05) | 🔴 colmat 1.09× slower than `faer` | 🔴 colmat 1.34× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00369 ± 1.0504e-05 (W 2.58496e-05) | 0.00349 ± 1.52753e-05 (W 1.78101e-05) | 0.00311 ± 0 (W 1.31985e-05) | 🔴 colmat 1.06× slower than `faer` | 🔴 colmat 1.19× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00178 ± 5.7735e-06 (W 1.64317e-05) | 0.00406 ± 0 (W 1.18406e-05) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00179 ± 1.52753e-05 (W 4.1833e-05) | 0.00167 ± 1.52753e-05 (W 3.70027e-05) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.007409 ± 0.000802085 (W 1.81163e-05) | 0.03071 ± 0.000442105 (W 2.80767e-05) | - | 🔵 colmat 4.14× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00683 ± 4.56107e-05 (W 4.33728e-05) | 0.0066 ± 1.0504e-05 (W 4.18951e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.01445 ± 2.08167e-05 (W 3.97278e-05) | 0.098747 ± 5.1316e-05 (W 4.47214e-05) | - | 🔵 colmat 6.83× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01324 ± 8.41249e-05 (W 6.25164e-05) | 0.012799 ± 2.47049e-05 (W 5.02623e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.051689 ± 0.000192398 (W 0.000134582) | 0.056738 ± 0.000331281 (W 8.28855e-05) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.052038 ± 9.12707e-05 (W 0.00065156) | 0.057138 ± 0.000243859 (W 8.19408e-05) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.199716 ± 7.18424e-05 (W 0.00164759) | 0.208275 ± 0.000289588 (W 0.00100495) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.200684 ± 0.000428269 (W 0.00102276) | 0.208693 ± 0.000528912 (W 0.00134042) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.472864 ± 0.00323284 (W 0.00181214) | 0.59272 ± 0.00252411 (W 0.00152915) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.471364 ± 0.00409056 (W 0.00181618) | 0.591302 ± 0.00231227 (W 0.00145301) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.007629 ± 0.000187804 (W 9.31193e-05) | 0.00547 ± 9.45163e-05 (W 2.90551e-05) | - | 🔴 colmat 1.39× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.010329 ± 0.000136137 (W 8.08598e-05) | 0.006599 ± 5.70117e-05 (W 0.000107765) | - | 🔴 colmat 1.57× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.027689 ± 0.000152775 (W 0.000100619) | 0.02336 ± 8.96289e-05 (W 4.63702e-05) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.035089 ± 0.000281839 (W 9.96504e-05) | 0.025579 ± 0.00013593 (W 9.33809e-05) | - | 🔴 colmat 1.37× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.065448 ± 0.000186566 (W 0.000157478) | 0.056498 ± 0.000546873 (W 0.000115539) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.078979 ± 0.000666867 (W 0.00118823) | 0.060968 ± 0.000163127 (W 0.00026809) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00416 ± 6e-05 (W 3.42374e-05) | 0.00381 ± 3.78594e-05 (W 4.0289e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00418 ± 3.4641e-05 (W 3.50599e-05) | 0.0038 ± 1.52753e-05 (W 2e-05) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014599 ± 3.4641e-05 (W 6.1606e-05) | 0.01413 ± 4.50925e-05 (W 9.5122e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01448 ± 4.0501e-05 (W 7.20437e-05) | 0.013979 ± 0.000121149 (W 5.26621e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.040349 ± 0.000218251 (W 0.000138637) | 0.038369 ± 0.000136137 (W 8.8364e-05) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.040219 ± 0.000337095 (W 9.12688e-05) | 0.037919 ± 7.5368e-05 (W 0.000936394) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00429 ± 6.08276e-05 (W 4.18115e-05) | 0.00517 ± 5.50757e-05 (W 1.96774e-05) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00547 ± 7.0946e-05 (W 3.22676e-05) | 0.00481 ± 0.000205183 (W 0.000160779) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01315 ± 7.02377e-05 (W 6.70917e-05) | 0.02154 ± 0.000312395 (W 4.53023e-05) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.01709 ± 2.60832e-05 (W 9.15604e-05) | 0.01453 ± 0.000180851 (W 0.000109276) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.027569 ± 0.00018735 (W 0.000157067) | 0.050599 ± 0.000470553 (W 0.000222902) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.036119 ± 0.000165227 (W 0.000131853) | 0.030469 ± 5.2915e-05 (W 0.000147919) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.01811 ± 0.000132288 (W 4.74479e-05) | 0.02244 ± 0.000110183 (W 6.29944e-05) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.01332 ± 0.000191312 (W 0.000121632) | 0.01892 ± 7.47685e-05 (W 6.89079e-05) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.057568 ± 0.000274044 (W 0.00117857) | 0.100568 ± 0.000457348 (W 0.00110238) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.044828 ± 5.11892e-05 (W 0.000138958) | 0.068227 ± 0.000225012 (W 0.000153666) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.117417 ± 0.000140358 (W 0.00123858) | 0.214454 ± 0.00301746 (W 0.00118852) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.109027 ± 0.000277093 (W 0.0010192) | 0.156855 ± 0.0021797 (W 0.000915385) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.01149 ± 0.000147986 (W 8.95779e-05) | 0.016349 ± 4.20357e-05 (W 5.1549e-05) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.038439 ± 0.000175586 (W 8.10327e-05) | 0.078489 ± 0.000130347 (W 0.00123887) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.070528 ± 0.000102211 (W 0.000153754) | 0.164494 ± 0.0001206 (W 0.000151261) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00465 ± 1.1547e-05 (W 2.50839e-05) | 0.00561 ± 3.51188e-05 (W 2.51038e-05) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00633 ± 2.08167e-05 (W 2.30217e-05) | 0.00542 ± 8.18535e-05 (W 3.5975e-05) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01273 ± 0.000130128 (W 6.11662e-05) | 0.022489 ± 0.000211266 (W 7.07976e-05) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.018009 ± 0.000143063 (W 5.9618e-05) | 0.015 ± 0.000123423 (W 0.000100276) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.02534 ± 0.000130615 (W 6.57822e-05) | 0.049859 ± 0.000914298 (W 6.39937e-05) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.035899 ± 0.000327159 (W 0.000109051) | 0.030209 ± 6.02771e-05 (W 8.26438e-05) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.081937 ± 0.00553883 (W 0.00103461) | 0.091177 ± 0.00169244 (W 0.00119458) | - | 🔵 colmat 1.11× faster than `faer` (marginal: spread 0.78 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.082598 ± 0.00206199 (W 0.000379031) | 0.091317 ± 0.000165025 (W 0.000146731) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.330602 ± 0.000585311 (W 0.00120938) | 0.348669 ± 0.00199942 (W 0.00136591) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.336432 ± 0.000589545 (W 0.00156696) | 0.353202 ± 0.0024134 (W 0.00109297) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.754776 ± 0.00322937 (W 0.00162887) | 1.02198 ± 0.00011812 (W 0.000640647) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.758402 ± 0.00195243 (W 0.0012323) | 1.04607 ± 0.0045471 (W 0.0012406) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.031819 ± 0.000194251 (W 0.000125579) | 0.072388 ± 8.73689e-05 (W 8.00575e-05) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.032139 ± 0.000116762 (W 0.000122944) | 0.072807 ± 0.000203161 (W 0.00101583) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.111837 ± 0.000597188 (W 0.00122263) | 0.263144 ± 0.000475933 (W 0.00104569) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.112287 ± 0.000180777 (W 0.0003118) | 0.263124 ± 0.000933358 (W 0.00134447) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.244482 ± 0.000570919 (W 0.00175505) | 0.583955 ± 0.00270904 (W 0.00153926) | - | 🔵 colmat 2.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.245443 ± 0.000676575 (W 0.000967822) | 0.583476 ± 0.000837191 (W 0.00138151) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00049 ± 0 (W 2.0693e-05) | 0.00047 ± 1e-05 (W 1.67332e-05) | 0.00031 ± 0 (W 1.78885e-05) | · | 🔴 colmat 1.58× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00101 ± 1.1547e-05 (W 2.60768e-05) | 0.00098 ± 1.52753e-05 (W 1.51658e-05) | 0.00059 ± 5.7735e-06 (W 2.60768e-05) | · | 🔴 colmat 1.71× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00176 ± 1.73205e-05 (W 2.60768e-05) | 0.00163 ± 5.7735e-06 (W 1.64317e-05) | 0.00092 ± 1.1547e-05 (W 2.16795e-05) | 🔴 colmat 1.08× slower than `faer` | 🔴 colmat 1.91× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00176 ± 5.7735e-06 (W 2.68328e-05) | 0.00406 ± 9.29157e-05 (W 1.80056e-05) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00179 ± 1.52753e-05 (W 5.76194e-05) | 0.00169 ± 4.04145e-05 (W 3.34664e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00747 ± 3.05505e-05 (W 3.06953e-05) | 0.030699 ± 5.2915e-05 (W 3.27109e-05) | - | 🔵 colmat 4.11× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00687 ± 2.36714e-05 (W 3.46006e-05) | 0.00668 ± 7.02377e-05 (W 4.69063e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.014899 ± 0.00101412 (W 4.92778e-05) | 0.050069 ± 0.00381273 (W 0.0223326) | - | 🔵 colmat 3.36× faster than `faer` (marginal: spread 0.63 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01358 ± 6.26126e-05 (W 8.04817e-05) | 0.01683 ± 0.000261024 (W 0.000282125) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.039769 ± 0.000187705 (W 9.83219e-05) | 0.056599 ± 0.000233603 (W 6.21868e-05) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.040208 ± 0.000241868 (W 0.00016756) | 0.057069 ± 0.000210516 (W 0.00011945) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.119087 ± 0.000528555 (W 0.000273542) | 0.211483 ± 0.000291027 (W 0.00112024) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.119196 ± 0.000885803 (W 0.000336185) | 0.212133 ± 0.0011058 (W 0.00127941) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.260322 ± 0.000120583 (W 0.00163538) | 0.586701 ± 0.000108657 (W 0.00225769) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.270563 ± 0.00168555 (W 0.00338105) | 0.587675 ± 0.000642357 (W 0.0015276) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.008309 ± 4.14769e-05 (W 5.89008e-05) | 0.00552 ± 8.0829e-05 (W 2.70185e-05) | - | 🔴 colmat 1.51× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01101 ± 6.50641e-05 (W 7.10936e-05) | 0.00679 ± 0.000110151 (W 9.89657e-05) | - | 🔴 colmat 1.62× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.029439 ± 0.000365445 (W 0.00012478) | 0.023349 ± 0.000228546 (W 4.31995e-05) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.036869 ± 9.86577e-05 (W 0.000167481) | 0.025689 ± 0.000557501 (W 3.9512e-05) | - | 🔴 colmat 1.44× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.066258 ± 0.000610273 (W 0.00025044) | 0.057718 ± 0.00060696 (W 7.83658e-05) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.080677 ± 0.00039879 (W 0.000158143) | 0.060919 ± 0.000104835 (W 0.000462061) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00392 ± 9.29157e-05 (W 0.000238233) | 0.00393 ± 1.00167e-05 (W 5.20884e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.003539 ± 8.48548e-05 (W 9.65919e-05) | 0.003919 ± 1.12694e-05 (W 5.17687e-05) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.010279 ± 0.000230515 (W 0.000310032) | 0.00971 ± 8.0829e-05 (W 0.00029851) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.00912 ± 0.000302379 (W 0.000258316) | 0.00879 ± 5.2915e-05 (W 0.000122724) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.02108 ± 0.000170969 (W 0.000497848) | 0.020199 ± 7.9689e-05 (W 0.000584461) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.019909 ± 4.50925e-05 (W 0.00025962) | 0.01946 ± 0.000121327 (W 0.000323364) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.00433 ± 0.000189297 (W 2.07364e-05) | 0.00512 ± 2.95014e-05 (W 1.64317e-05) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00557 ± 6.35085e-05 (W 6.00017e-05) | 0.00485 ± 7.3328e-05 (W 5.3115e-05) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01341 ± 0.000121766 (W 8.71854e-05) | 0.0233 ± 0.00029365 (W 0.000145407) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.01713 ± 0.000151218 (W 0.000147456) | 0.016629 ± 0.000505045 (W 8.65696e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.034819 ± 0.000430039 (W 0.000690014) | 0.057878 ± 0.000225167 (W 0.000665643) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.03635 ± 6.68356e-05 (W 0.000218243) | 0.038709 ± 0.000110151 (W 0.000515742) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.01868 ± 5.74485e-05 (W 6.24924e-05) | 0.031649 ± 0.000115326 (W 0.000277705) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.013409 ± 0.000152699 (W 0.000258631) | 0.028019 ± 7.45453e-05 (W 0.000289227) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.046328 ± 0.00065783 (W 0.00063268) | 0.155726 ± 0.00107772 (W 0.00167874) | - | 🔵 colmat 3.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.043039 ± 0.000486621 (W 0.000218278) | 0.116946 ± 0.000931522 (W 0.00156419) | - | 🔵 colmat 2.72× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.099127 ± 0.000459384 (W 0.00151744) | 0.320272 ± 0.00256387 (W 0.00257505) | - | 🔵 colmat 3.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.091637 ± 0.000584685 (W 0.000932672) | 0.232782 ± 0.00119781 (W 0.00197783) | - | 🔵 colmat 2.54× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.012039 ± 0.000186897 (W 6.76927e-05) | 0.02528 ± 0.000135242 (W 0.000340361) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.023169 ± 0.000165025 (W 0.00014891) | 0.129756 ± 0.000722657 (W 0.0017675) | - | 🔵 colmat 5.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.052979 ± 0.000379868 (W 0.000281876) | 0.267563 ± 0.000613638 (W 0.0020804) | - | 🔵 colmat 5.05× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.005009 ± 3.06649e-05 (W 3.20936e-05) | 0.005549 ± 0.000120956 (W 2.78065e-05) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.006849 ± 4.17971e-05 (W 4.20024e-05) | 0.00555 ± 7.25282e-05 (W 6.19048e-05) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.013389 ± 3.59212e-05 (W 6.40726e-05) | 0.02379 ± 0.00045094 (W 0.000773717) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.018629 ± 5.0797e-05 (W 8.70534e-05) | 0.017089 ± 0.000150932 (W 0.000401129) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.029819 ± 0.000270247 (W 0.000580284) | 0.056198 ± 0.00150162 (W 0.00146764) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.038099 ± 0.000211975 (W 0.000153806) | 0.037189 ± 0.000218251 (W 0.000380565) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.082587 ± 0.000380934 (W 0.00112931) | 0.090487 ± 0.000313422 (W 9.66954e-05) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.082677 ± 0.000175214 (W 0.000212775) | 0.090877 ± 0.000377783 (W 0.000124424) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.279721 ± 0.000761324 (W 0.00109485) | 0.355859 ± 0.00613633 (W 0.00122431) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.288962 ± 0.000569509 (W 0.00182284) | 0.35662 ± 0.000230088 (W 0.00197432) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.564183 ± 0.000999682 (W 0.00216716) | 1.01595 ± 0.00127345 (W 0.00150124) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.564425 ± 0.000724877 (W 0.00258361) | 1.01496 ± 0.000232337 (W 0.00175659) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.032959 ± 0.000233967 (W 0.00010895) | 0.072308 ± 8.18535e-05 (W 8.05121e-05) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.033139 ± 0.00015695 (W 9.09395e-05) | 0.072908 ± 4.93288e-05 (W 0.00136233) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.113947 ± 9.51332e-05 (W 0.000220105) | 0.262453 ± 0.000850906 (W 0.00171315) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.117427 ± 0.000390359 (W 0.000334008) | 0.262892 ± 0.00045263 (W 0.00193651) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.251824 ± 0.000704699 (W 0.000953876) | 0.581002 ± 0.00175032 (W 0.00162015) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.253803 ± 0.00207477 (W 0.000745219) | 0.583304 ± 0.003185 (W 0.00162876) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00063 ± 5.7735e-06 (W 1.67332e-05) | 0.00046 ± 5.7735e-06 (W 1.94936e-05) | 0.0003 ± 5.7735e-06 (W 2.16795e-05) | 🔴 colmat 1.37× slower than `faer` | 🔴 colmat 2.10× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00131 ± 2.08167e-05 (W 2e-05) | 0.00096 ± 1e-05 (W 1.64317e-05) | 0.0006 ± 5.7735e-06 (W 2.50998e-05) | 🔴 colmat 1.36× slower than `faer` | 🔴 colmat 2.18× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00217 ± 2e-05 (W 2.12132e-05) | 0.00162 ± 5.7735e-06 (W 1.34164e-05) | 0.00093 ± 0 (W 2.07364e-05) | 🔴 colmat 1.34× slower than `faer` | 🔴 colmat 2.33× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00176 ± 2.64575e-05 (W 2.30217e-05) | 0.004059 ± 8.98907e-05 (W 1.88202e-05) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0018 ± 1.52753e-05 (W 5.2915e-05) | 0.00172 ± 1.52753e-05 (W 2.77489e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00747 ± 2.51661e-05 (W 1.95755e-05) | 0.030679 ± 0.000413562 (W 4.44972e-05) | - | 🔵 colmat 4.11× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00682 ± 3.51188e-05 (W 5.21939e-05) | 0.006629 ± 7.91854e-05 (W 3.14197e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.014899 ± 5.1316e-05 (W 4.17409e-05) | 0.043868 ± 0.000270038 (W 0.000362687) | - | 🔵 colmat 2.94× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01357 ± 8.7409e-05 (W 5.23001e-05) | 0.017289 ± 0.000203929 (W 0.000307101) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.037819 ± 0.000537246 (W 0.000131777) | 0.057568 ± 0.000272067 (W 8.71218e-05) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.038628 ± 0.000120611 (W 0.00049558) | 0.057189 ± 0.000924852 (W 0.000245962) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.107876 ± 0.000815236 (W 0.000519054) | 0.212494 ± 0.00441737 (W 0.00185398) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.108367 ± 0.000520865 (W 0.000357437) | 0.213553 ± 0.000649651 (W 0.00129127) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.227923 ± 0.000832569 (W 0.00170901) | 0.601101 ± 0.00217556 (W 0.0025332) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.237513 ± 0.00171981 (W 0.00222856) | 0.602112 ± 0.00186997 (W 0.00352445) | - | 🔵 colmat 2.54× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.00836 ± 3.00056e-05 (W 5.41045e-05) | 0.00535 ± 3.60555e-05 (W 3.15151e-05) | - | 🔴 colmat 1.56× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.011049 ± 4.07063e-05 (W 5.64907e-05) | 0.00669 ± 0.000189033 (W 7.22717e-05) | - | 🔴 colmat 1.65× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.029359 ± 0.000109697 (W 7.59605e-05) | 0.023299 ± 0.000708094 (W 3.52605e-05) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.036719 ± 0.000440606 (W 0.000112116) | 0.02502 ± 0.000270519 (W 6.07635e-05) | - | 🔴 colmat 1.47× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.066729 ± 0.000782177 (W 0.000126461) | 0.057658 ± 0.000542248 (W 0.000125739) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.080677 ± 0.000344035 (W 0.000121553) | 0.060818 ± 0.000407799 (W 0.00159661) | - | 🔴 colmat 1.33× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00436 ± 0.00010504 (W 0.000296443) | 0.00393 ± 1.18462e-05 (W 3.91433e-05) | - | 🔴 colmat 1.11× slower than `faer` (marginal: spread 0.69 of the difference) | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00412 ± 0.000240069 (W 0.000224789) | 0.00392 ± 1e-05 (W 2.38747e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.00959 ± 0.000287112 (W 0.000251667) | 0.01802 ± 0.000624972 (W 0.000848961) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.00859 ± 0.000173312 (W 0.000311641) | 0.016899 ± 0.000487887 (W 0.000781951) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.01639 ± 0.000166133 (W 0.000377178) | 0.027549 ± 0.00044456 (W 0.00184749) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.016369 ± 0.000241268 (W 0.000365849) | 0.027279 ± 0.000328938 (W 0.000901481) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.00432 ± 2.08167e-05 (W 1.67332e-05) | 0.0051 ± 6.1101e-05 (W 1.91625e-05) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00554 ± 3.60555e-05 (W 7.04784e-05) | 0.0048 ± 0.000117189 (W 0.000132636) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01331 ± 0.000107858 (W 6.21555e-05) | 0.023639 ± 0.000431949 (W 0.000228594) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.017129 ± 0.000173754 (W 0.00015113) | 0.017279 ± 8.88819e-05 (W 0.000175372) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.040868 ± 0.00086512 (W 0.00144772) | 0.059538 ± 0.00060575 (W 0.00150447) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.036109 ± 0.000297153 (W 0.000194469) | 0.040989 ± 0.000365923 (W 0.000539009) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.019059 ± 0.000242116 (W 8.69385e-05) | 0.037368 ± 0.00051344 (W 0.00121307) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.013519 ± 6.89952e-05 (W 8.96231e-05) | 0.033319 ± 0.00035949 (W 0.000531573) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.046878 ± 0.000635091 (W 0.00084687) | 0.185145 ± 0.00202041 (W 0.00364169) | - | 🔵 colmat 3.95× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.043679 ± 0.000251191 (W 0.000182437) | 0.148556 ± 0.00153001 (W 0.00197991) | - | 🔵 colmat 3.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.094157 ± 0.000607701 (W 0.00102969) | 0.380408 ± 0.00339737 (W 0.00313822) | - | 🔵 colmat 4.04× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.090747 ± 0.000404269 (W 0.000279342) | 0.303461 ± 0.00288756 (W 0.00235065) | - | 🔵 colmat 3.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.01208 ± 0.000177858 (W 8.13099e-05) | 0.031519 ± 0.000636108 (W 0.000426211) | - | 🔵 colmat 2.61× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.024029 ± 0.000155885 (W 0.000132858) | 0.157886 ± 0.000987102 (W 0.00352801) | - | 🔵 colmat 6.57× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.050049 ± 0.000285553 (W 0.000555452) | 0.321711 ± 0.00102 (W 0.00310585) | - | 🔵 colmat 6.43× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.004989 ± 0.000156462 (W 3.42053e-05) | 0.00557 ± 2e-05 (W 2.71698e-05) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00666 ± 5.50757e-05 (W 3.68809e-05) | 0.005449 ± 3.59212e-05 (W 9.32373e-05) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01339 ± 8.24035e-05 (W 4.04265e-05) | 0.02352 ± 0.000175158 (W 0.000222183) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.018649 ± 9.55353e-05 (W 9.29962e-05) | 0.01756 ± 3.76431e-05 (W 0.000190655) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.035059 ± 0.0016752 (W 0.00188939) | 0.055868 ± 0.000164743 (W 0.00101419) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.038388 ± 0.000175586 (W 0.000334634) | 0.039039 ± 0.00029 (W 0.000528389) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.081288 ± 0.00205501 (W 0.00024122) | 0.090938 ± 0.000125341 (W 0.00129228) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.082237 ± 0.00212702 (W 0.000281637) | 0.090957 ± 0.000477783 (W 0.000105523) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.280631 ± 0.000450278 (W 0.00155445) | 0.359158 ± 0.00167277 (W 0.00204005) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.290761 ± 0.000460066 (W 0.00286793) | 0.36064 ± 0.00259441 (W 0.00239551) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.559602 ± 0.00304575 (W 0.00173152) | 1.03756 ± 0.0022109 (W 0.00349114) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.560653 ± 0.000223214 (W 0.00275199) | 1.03974 ± 0.0010758 (W 0.00224076) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.032629 ± 0.000120968 (W 0.000109225) | 0.072628 ± 0.000110907 (W 0.000100733) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.032949 ± 0.000166233 (W 9.8154e-05) | 0.072868 ± 7.40968e-05 (W 0.00106001) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.118117 ± 0.000464722 (W 0.000157957) | 0.262083 ± 0.00103507 (W 0.00197187) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.113267 ± 0.000623538 (W 0.00032521) | 0.263512 ± 0.000847296 (W 0.00159451) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.255072 ± 0.000402563 (W 0.000780264) | 0.583562 ± 0.00142689 (W 0.00188062) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.253583 ± 0.00100168 (W 0.00132423) | 0.583012 ± 0.000442471 (W 0.00169648) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00063 ± 1.1547e-05 (W 1.58114e-05) | 0.00047 ± 1e-05 (W 1.64317e-05) | 0.00031 ± 0 (W 2.50998e-05) | 🔴 colmat 1.34× slower than `faer` | 🔴 colmat 2.03× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00131 ± 5.7735e-06 (W 2.07364e-05) | 0.00098 ± 0 (W 1.94936e-05) | 0.00059 ± 0 (W 1.92354e-05) | 🔴 colmat 1.34× slower than `faer` | 🔴 colmat 2.22× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00218 ± 3.05505e-05 (W 1.87083e-05) | 0.00161 ± 5.7735e-06 (W 1.81659e-05) | 0.00092 ± 5.7735e-06 (W 2.16795e-05) | 🔴 colmat 1.35× slower than `faer` | 🔴 colmat 2.37× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0013 ± 5.7735e-06 (W 8.94427e-06) | 0.00124 ± 1.52753e-05 (W 1.30384e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0017 ± 1.73205e-05 (W 3.11448e-05) | 0.00162 ± 2.04206e-05 (W 3.11448e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00566 ± 1.73205e-05 (W 1.14018e-05) | 0.00551 ± 3.78594e-05 (W 9.06642e-06) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00668 ± 9.50438e-06 (W 3.40617e-05) | 0.006439 ± 5.50757e-06 (W 3.54288e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.010929 ± 3.64737e-05 (W 2.84306e-05) | 0.01054 ± 2.08167e-05 (W 2.54598e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.012869 ± 1.1547e-05 (W 5.45921e-05) | 0.012389 ± 0.000358372 (W 0.000355448) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.051168 ± 0.000700533 (W 7.60736e-05) | 0.056808 ± 0.000114762 (W 6.85952e-05) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.051669 ± 0.000100201 (W 0.000205651) | 0.057159 ± 0.000300899 (W 0.000744443) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.199945 ± 0.000544023 (W 0.00116295) | 0.208994 ± 0.000225147 (W 0.00113819) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.200664 ± 0.000923227 (W 0.00116433) | 0.209064 ± 0.000464224 (W 0.000965671) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.473358 ± 0.000554107 (W 0.00201337) | 0.593542 ± 0.000662541 (W 0.0014525) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.472869 ± 0.00224084 (W 0.00210852) | 0.595781 ± 0.00205158 (W 0.00101388) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.007909 ± 3.4356e-05 (W 5.90186e-05) | 0.00539 ± 3.21455e-05 (W 2.4984e-05) | - | 🔴 colmat 1.47× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.00884 ± 5.7735e-06 (W 6.96656e-05) | 0.0063 ± 5.2915e-05 (W 8.15488e-05) | - | 🔴 colmat 1.40× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.028539 ± 0.000159061 (W 8.25609e-05) | 0.022739 ± 0.000355949 (W 3.08901e-05) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.03052 ± 7.47685e-05 (W 0.000106561) | 0.024769 ± 0.000334286 (W 0.00013338) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.067268 ± 0.000266434 (W 0.000152182) | 0.056458 ± 0.000142584 (W 0.000289173) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.070657 ± 0.00030249 (W 0.000248137) | 0.061048 ± 0.000806866 (W 0.00040806) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0045 ± 2.64575e-05 (W 3.80158e-05) | 0.00427 ± 5.7735e-07 (W 5.45894e-05) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00563 ± 0.000121655 (W 0.0001499) | 0.00424 ± 3.51188e-05 (W 3e-05) | - | 🔴 colmat 1.33× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.01625 ± 4.62637e-05 (W 7.40898e-05) | 0.01555 ± 0.000168968 (W 5.29075e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01631 ± 3.05505e-05 (W 8.46481e-05) | 0.015539 ± 4.20357e-05 (W 4.9227e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.044718 ± 0.00059496 (W 0.000230803) | 0.043109 ± 0 (W 0.000142724) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.044439 ± 0.00016885 (W 9.1675e-05) | 0.043069 ± 0.000139632 (W 7.11006e-05) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00332 ± 2.88675e-05 (W 2.97187e-05) | 0.00364 ± 4.16333e-05 (W 2.48998e-05) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00424 ± 9.96845e-05 (W 4.26989e-05) | 0.00447 ± 2.70247e-05 (W 6.3073e-05) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01126 ± 3.27159e-05 (W 0.00010181) | 0.01208 ± 8.69502e-05 (W 7.42516e-05) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.01288 ± 4.07063e-05 (W 9.3511e-05) | 0.013819 ± 0.000112073 (W 0.000129905) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.023629 ± 0.00012546 (W 0.000258591) | 0.025859 ± 0.000327338 (W 0.000106207) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.026129 ± 8.55005e-05 (W 0.000147862) | 0.028689 ± 0.000200333 (W 0.000142021) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.01094 ± 0.000175586 (W 0.000100684) | 0.016989 ± 0.000131341 (W 6.35004e-05) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.01219 ± 0.000210317 (W 0.00012127) | 0.01856 ± 6.62596e-05 (W 5.04311e-05) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.037069 ± 0.000411015 (W 0.000192014) | 0.062588 ± 0.000299712 (W 7.53936e-05) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.039799 ± 0.000268962 (W 8.48068e-05) | 0.065718 ± 0.000138337 (W 0.000925859) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.094287 ± 0.000813068 (W 0.00026665) | 0.146655 ± 0.000870332 (W 0.00105922) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.098658 ± 0.000540025 (W 0.000297775) | 0.151395 ± 0.000990979 (W 0.00117131) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00538 ± 3.05505e-05 (W 2.35415e-05) | 0.011039 ± 6.48614e-05 (W 9.2894e-05) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.018759 ± 0.000140524 (W 0.000219698) | 0.044979 ± 8.77515e-05 (W 0.000117967) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n96/b1` | 0.042099 ± 0.000281727 (W 0.000232873) | 0.095527 ± 0.000208167 (W 0.000936453) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00393 ± 5.85947e-05 (W 3.17679e-05) | 0.00399 ± 1.1547e-05 (W 2.73167e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00496 ± 3.4641e-05 (W 3.07929e-05) | 0.00485 ± 2.64575e-05 (W 2.13588e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01154 ± 4.45758e-05 (W 6.78255e-05) | 0.0124 ± 5.56776e-05 (W 4.56103e-05) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.01338 ± 2.88675e-05 (W 7.44802e-05) | 0.01415 ± 5.70117e-05 (W 8.5961e-05) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.023659 ± 0.00035543 (W 0.000142616) | 0.025729 ± 3.05505e-05 (W 6.33024e-05) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.025919 ± 0.000139787 (W 0.000155905) | 0.02871 ± 0.000169087 (W 8.74426e-05) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.086448 ± 0.000608145 (W 0.000200295) | 0.096957 ± 0.00179074 (W 0.000181238) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.086738 ± 0.000327964 (W 0.000546672) | 0.097277 ± 0.000802893 (W 0.00116634) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.333349 ± 0.00060328 (W 0.00144401) | 0.354768 ± 0.0109796 (W 0.00131079) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.338999 ± 0.00142905 (W 0.00173532) | 0.351689 ± 0.00276057 (W 0.00143508) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.754966 ± 0.00183842 (W 0.00143945) | 1.01906 ± 0.00343802 (W 0.000892212) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.757826 ± 0.00187729 (W 0.00148413) | 1.039 ± 0.0024861 (W 0.00228798) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.03239 ± 0.000286828 (W 8.01137e-05) | 0.071767 ± 0.000259963 (W 0.000956458) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.032519 ± 0.000248797 (W 7.99887e-05) | 0.072088 ± 8.7178e-05 (W 0.00011508) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.114267 ± 5.68624e-05 (W 0.000212029) | 0.263172 ± 0.00114522 (W 0.000932199) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.114637 ± 0.000368859 (W 0.00132588) | 0.263622 ± 0.00110888 (W 0.00128092) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.247164 ± 0.00112227 (W 0.00127178) | 0.571736 ± 0.00134072 (W 0.00511791) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.247562 ± 0.00113757 (W 0.00145334) | 0.572001 ± 0.00155623 (W 0.00140188) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00053 ± 5.7735e-06 (W 1.66493e-05) | 0.00051 ± 5.7735e-06 (W 1.51658e-05) | 0.00029 ± 0 (W 2.23607e-05) | · | 🔴 colmat 1.83× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00114 ± 1e-05 (W 1.92354e-05) | 0.00112 ± 5.7735e-06 (W 2.16795e-05) | 0.00055 ± 5.7735e-06 (W 2.60768e-05) | · | 🔴 colmat 2.07× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.001809 ± 5.50757e-06 (W 2.58844e-05) | 0.00177 ± 1.1547e-05 (W 1.64317e-05) | 0.00087 ± 5.7735e-06 (W 2.07364e-05) | · | 🔴 colmat 2.08× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0013 ± 0.00013 (W 1.22474e-05) | 0.00125 ± 0 (W 1.12339e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00171 ± 1e-05 (W 2.54951e-05) | 0.00162 ± 3.05505e-05 (W 3.72317e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00565 ± 5.68624e-05 (W 1.83085e-05) | 0.00553 ± 0.00069601 (W 1.51658e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.00672 ± 4.3589e-05 (W 3.54429e-05) | 0.00653 ± 0.000154068 (W 3.51596e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.01119 ± 0.000100732 (W 5.47932e-05) | 0.014239 ± 0.000285255 (W 0.00014204) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01334 ± 0.000132791 (W 5.61721e-05) | 0.016209 ± 0.000299984 (W 0.000455097) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.039859 ± 0.00015 (W 0.000182071) | 0.056509 ± 0.000165198 (W 5.42061e-05) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.040949 ± 0.000615866 (W 0.000407674) | 0.056949 ± 0.000131936 (W 9.86828e-05) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.118897 ± 0.000551522 (W 0.00027145) | 0.211073 ± 0.000175529 (W 0.00126193) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.119736 ± 0.000771713 (W 0.000368637) | 0.211724 ± 0.000468363 (W 0.000958178) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.260373 ± 0.0027419 (W 0.00116311) | 0.587294 ± 0.0036325 (W 0.00146606) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.271212 ± 0.000418415 (W 0.0035491) | 0.588501 ± 0.00235924 (W 0.00181875) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.00858 ± 5.03322e-05 (W 7.64539e-05) | 0.00546 ± 2.36714e-05 (W 2.07364e-05) | - | 🔴 colmat 1.57× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.00958 ± 0.000128582 (W 5.89847e-05) | 0.00641 ± 0.00010583 (W 3.21745e-05) | - | 🔴 colmat 1.49× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.030309 ± 0.000235797 (W 8.50294e-05) | 0.02324 ± 0.000259474 (W 3.36482e-05) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.032069 ± 0.000112694 (W 0.000126689) | 0.024749 ± 0.000237978 (W 9.45156e-05) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.068458 ± 0.000899389 (W 0.000433497) | 0.057509 ± 0.000365582 (W 0.000121968) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.071248 ± 0.000245764 (W 9.52323e-05) | 0.060068 ± 0.000390308 (W 0.00117393) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00346 ± 0.000121244 (W 0.000150244) | 0.00434 ± 1.1547e-05 (W 3.1305e-05) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00358 ± 6.35085e-05 (W 0.000169765) | 0.00429 ± 2.12211e-05 (W 3.64692e-05) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.008769 ± 0.000198008 (W 0.000159108) | 0.00843 ± 0.000109697 (W 9.48683e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.00916 ± 7.55447e-05 (W 9.39803e-05) | 0.00882 ± 0.000112398 (W 0.000130843) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.018269 ± 0.0002705 (W 0.000174489) | 0.017539 ± 0.000283902 (W 0.000152313) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.01804 ± 0.000115326 (W 0.00018743) | 0.01764 ± 0.000135316 (W 0.00024666) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.00338 ± 4.04145e-05 (W 3.31662e-05) | 0.0036 ± 3.51188e-05 (W 2.96648e-05) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00427 ± 5.22526e-05 (W 5.59035e-05) | 0.00446 ± 3e-05 (W 3.37224e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.01158 ± 0.000205508 (W 0.000121944) | 0.0139 ± 5.05008e-05 (W 0.0003698) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.013089 ± 7.23418e-05 (W 9.10401e-05) | 0.015929 ± 0.000141987 (W 0.000447739) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.02375 ± 0.000228983 (W 0.000260728) | 0.033469 ± 0.000589237 (W 0.000890684) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.026919 ± 0.000225019 (W 0.000248637) | 0.036689 ± 7.63763e-05 (W 0.000170306) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.01087 ± 0.000178298 (W 0.000184384) | 0.025899 ± 0.000365103 (W 0.00051801) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.012429 ± 6.0803e-05 (W 0.000162275) | 0.027669 ± 0.0001205 (W 0.000377777) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.036519 ± 0.000334714 (W 0.000185489) | 0.113006 ± 0.00106101 (W 0.00104699) | - | 🔵 colmat 3.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.038829 ± 0.000192095 (W 0.000194564) | 0.115716 ± 0.000411822 (W 0.00123481) | - | 🔵 colmat 2.98× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.076238 ± 0.000652865 (W 0.000312114) | 0.223123 ± 0.00162322 (W 0.00178562) | - | 🔵 colmat 2.93× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.081148 ± 0.000815618 (W 0.000362772) | 0.228393 ± 0.00149579 (W 0.00240694) | - | 🔵 colmat 2.81× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00541 ± 6.4291e-05 (W 3.2912e-05) | 0.01987 ± 0.000370887 (W 0.000271609) | - | 🔵 colmat 3.67× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.018579 ± 0.0001948 (W 0.000267778) | 0.092157 ± 0.000653016 (W 0.00115104) | - | 🔵 colmat 4.96× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n96/b1` | 0.042059 ± 0.000180362 (W 0.00020319) | 0.182125 ± 0.000450408 (W 0.00198413) | - | 🔵 colmat 4.33× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00415 ± 3.21455e-05 (W 3.53553e-05) | 0.00395 ± 4.07063e-05 (W 2.97019e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00547 ± 1.1547e-05 (W 4.94085e-05) | 0.00516 ± 5.51392e-05 (W 6.12372e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.012249 ± 0.000108214 (W 6.67106e-05) | 0.014369 ± 0.000610762 (W 0.000502731) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.01393 ± 4.2194e-05 (W 6.97302e-05) | 0.01634 ± 0.000112947 (W 0.000144424) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.025119 ± 4.39356e-05 (W 0.000153357) | 0.031729 ± 6.65833e-05 (W 0.000228648) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.027839 ± 0.000564535 (W 0.000221256) | 0.035199 ± 0.000356791 (W 0.000245549) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.087638 ± 0.000125779 (W 0.0010981) | 0.096067 ± 0.00119976 (W 0.000264833) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.088187 ± 0.000334081 (W 0.00030451) | 0.096777 ± 0.000692892 (W 0.00118481) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.278991 ± 0.000836543 (W 0.00100837) | 0.359899 ± 0.001466 (W 0.0016284) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.285482 ± 0.00145435 (W 0.00179966) | 0.36024 ± 0.000415993 (W 0.0012131) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.567862 ± 0.00104164 (W 0.00150671) | 1.01236 ± 0.00191181 (W 0.00122293) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.565085 ± 0.00205273 (W 0.00101559) | 1.01411 ± 0.000335357 (W 0.000712757) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.033109 ± 9.53939e-05 (W 0.000108237) | 0.071897 ± 0.000236496 (W 0.00115255) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.033149 ± 0.000185599 (W 9.23753e-05) | 0.072138 ± 0.000244199 (W 0.000185949) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.115896 ± 0.000144938 (W 0.000352378) | 0.262553 ± 0.000335402 (W 0.00165326) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.119567 ± 0.000180103 (W 0.000351826) | 0.263461 ± 0.000710768 (W 0.00130748) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.255142 ± 0.00130898 (W 0.00116454) | 0.571272 ± 0.000535288 (W 0.00139805) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.254772 ± 0.000609869 (W 0.00171614) | 0.571642 ± 0.00110765 (W 0.00203681) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00081 ± 5.7735e-06 (W 1.92354e-05) | 0.00067 ± 1.52753e-05 (W 2.48998e-05) | 0.00029 ± 5.7735e-06 (W 2.23607e-05) | 🔴 colmat 1.21× slower than `faer` | 🔴 colmat 2.79× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00139 ± 1.73205e-05 (W 3.04959e-05) | 0.00111 ± 5.7735e-06 (W 1.81659e-05) | 0.00055 ± 0 (W 2.60768e-05) | 🔴 colmat 1.25× slower than `faer` | 🔴 colmat 2.53× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00216 ± 1.1547e-05 (W 1.92354e-05) | 0.00175 ± 1.1547e-05 (W 1.84445e-05) | 0.00086 ± 5.7735e-06 (W 2.16795e-05) | 🔴 colmat 1.23× slower than `faer` | 🔴 colmat 2.51× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00128 ± 5.7735e-06 (W 1.80056e-05) | 0.00125 ± 1.1547e-05 (W 2.07364e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0017 ± 5.7735e-06 (W 3.28634e-05) | 0.00167 ± 3.05505e-05 (W 3.21279e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.00564 ± 3.51188e-05 (W 2.19089e-05) | 0.00553 ± 4.99032e-05 (W 2.39625e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.006689 ± 2.00083e-05 (W 3.6086e-05) | 0.00652 ± 2.64575e-05 (W 3.92836e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.01121 ± 4.0501e-05 (W 4.65328e-05) | 0.014939 ± 0.000135967 (W 0.00023741) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.01333 ± 9.07377e-05 (W 5.61186e-05) | 0.01692 ± 0.000805564 (W 0.000344401) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `eigh/n32/b1` | 0.037859 ± 0.000313422 (W 0.000177029) | 0.056658 ± 6.65833e-05 (W 0.000103732) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.038589 ± 0.000254795 (W 0.000304762) | 0.057018 ± 0.000202073 (W 8.07608e-05) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.107347 ± 0.000941356 (W 0.000279133) | 0.213613 ± 0.000500047 (W 0.00172973) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.108137 ± 0.000531912 (W 0.000198947) | 0.213953 ± 0.000874357 (W 0.00200509) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.226683 ± 0.00143235 (W 0.00140973) | 0.595704 ± 0.00604244 (W 0.00354179) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.237713 ± 0.00188917 (W 0.00352987) | 0.599402 ± 0.00135239 (W 0.00206473) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.008579 ± 0.000145363 (W 4.55335e-05) | 0.0054 ± 4.53909e-05 (W 4.56749e-05) | - | 🔴 colmat 1.59× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.00958 ± 3.25115e-05 (W 7.1661e-05) | 0.00637 ± 0.000235727 (W 6.31443e-05) | - | 🔴 colmat 1.50× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.030329 ± 0.000196554 (W 0.00011339) | 0.02325 ± 0.00019713 (W 5.0033e-05) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.031879 ± 0.000170978 (W 0.000174986) | 0.025069 ± 0.000332341 (W 0.000134901) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.068678 ± 0.000159897 (W 0.00043393) | 0.056928 ± 0.000103923 (W 7.35745e-05) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.071158 ± 0.000344287 (W 0.000162654) | 0.059428 ± 0.000241109 (W 0.00104125) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00377 ± 4.93288e-05 (W 0.000105864) | 0.004379 ± 4.03361e-05 (W 4.60435e-05) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00368 ± 0.000123423 (W 0.000202312) | 0.004329 ± 3.04467e-05 (W 3.63318e-05) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.00735 ± 1.52753e-05 (W 0.000170975) | 0.01741 ± 0.000331919 (W 0.00122659) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.00778 ± 0.000135758 (W 0.000234643) | 0.017109 ± 0.000611449 (W 0.000813659) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.013919 ± 0.000364925 (W 0.000244794) | 0.025249 ± 0.000479305 (W 0.00111366) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.01386 ± 7.16961e-05 (W 0.00026051) | 0.025249 ± 0.000748176 (W 0.000892844) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1` | 0.003389 ± 1.70392e-05 (W 3.85772e-05) | 0.00362 ± 4.16333e-05 (W 2.6612e-05) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00431 ± 1.52753e-05 (W 4.03733e-05) | 0.0045 ± 4.6188e-05 (W 4.79812e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.011479 ± 5.11892e-05 (W 0.000162866) | 0.014539 ± 0.00011537 (W 0.000155352) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.013079 ± 4.12351e-05 (W 0.000107057) | 0.01641 ± 0.000166644 (W 0.0002754) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.023789 ± 0.00016784 (W 0.000320912) | 0.036329 ± 0.000355715 (W 0.000280416) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.026579 ± 0.000242323 (W 0.000320109) | 0.039209 ± 0.000338364 (W 0.00037375) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.01105 ± 0.0001 (W 7.43862e-05) | 0.031479 ± 0.000144222 (W 0.000297288) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.0122 ± 0.000248797 (W 0.000148594) | 0.032339 ± 0.000208167 (W 0.000333062) | - | 🔵 colmat 2.65× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.036179 ± 9.0185e-05 (W 0.000246211) | 0.145195 ± 0.000682291 (W 0.00140057) | - | 🔵 colmat 4.01× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.039359 ± 0.0003095 (W 0.000188723) | 0.146605 ± 0.000597643 (W 0.00151455) | - | 🔵 colmat 3.72× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.075718 ± 0.000118731 (W 0.000315414) | 0.290421 ± 0.00541216 (W 0.00341246) | - | 🔵 colmat 3.84× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.080448 ± 0.000549303 (W 0.000368687) | 0.299621 ± 0.00203125 (W 0.00471051) | - | 🔵 colmat 3.72× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00547 ± 4.2194e-05 (W 4.61433e-05) | 0.02605 ± 0.00100857 (W 0.000479429) | - | 🔵 colmat 4.76× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.01856 ± 0.000273332 (W 0.000143806) | 0.123206 ± 0.00087716 (W 0.00163207) | - | 🔵 colmat 6.64× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n96/b1` | 0.042069 ± 8.73689e-05 (W 0.00025354) | 0.244013 ± 0.00173434 (W 0.00256713) | - | 🔵 colmat 5.80× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00423 ± 1.52753e-05 (W 4.23934e-05) | 0.00398 ± 5.50757e-05 (W 3.10838e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00528 ± 1.95021e-05 (W 3.04959e-05) | 0.00489 ± 5.50757e-05 (W 7.64212e-05) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.01226 ± 8.14064e-05 (W 8.22636e-05) | 0.015229 ± 5.74485e-05 (W 0.000205879) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.014069 ± 0.000110666 (W 6.48637e-05) | 0.016619 ± 0.000176315 (W 0.000204708) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n96/b1` | 0.025219 ± 0.00025865 (W 0.00016607) | 0.033689 ± 0.000238607 (W 0.000358008) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.028239 ± 0.000110604 (W 0.000228101) | 0.037539 ± 0.000314325 (W 0.00041839) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.086967 ± 0.000242116 (W 0.000270397) | 0.098647 ± 0.00158052 (W 0.0013886) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.087037 ± 0.00137471 (W 0.000427355) | 0.096927 ± 0.000604759 (W 0.000994726) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.278292 ± 0.000799771 (W 0.00141866) | 0.360949 ± 0.000559244 (W 0.00206722) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.288901 ± 0.00172579 (W 0.00169357) | 0.364579 ± 0.0024263 (W 0.00234431) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.565833 ± 0.00524886 (W 0.00314905) | 1.03524 ± 0.00121027 (W 0.00189281) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.567693 ± 0.00082373 (W 0.00185839) | 1.0372 ± 0.00341371 (W 0.00271251) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.033029 ± 0.000334714 (W 0.000115239) | 0.071798 ± 0.000260001 (W 8.01386e-05) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.033299 ± 3.21455e-05 (W 0.000114368) | 0.072508 ± 0.000255408 (W 0.000891833) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.119367 ± 0.000105532 (W 0.000212086) | 0.262112 ± 0.000277543 (W 0.00161253) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.115456 ± 0.000260251 (W 0.000420947) | 0.263842 ± 0.00136185 (W 0.00111053) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.256523 ± 5.25389e-05 (W 0.00159904) | 0.569623 ± 0.0022525 (W 0.00203115) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.255632 ± 0.000461249 (W 0.00125185) | 0.569693 ± 0.00142949 (W 0.00242802) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00064 ± 5.7735e-06 (W 2.36897e-05) | 0.00051 ± 1.1547e-05 (W 1.09545e-05) | 0.00029 ± 0 (W 2.23607e-05) | 🔴 colmat 1.25× slower than `faer` | 🔴 colmat 2.21× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00145 ± 5.7735e-06 (W 3.78153e-05) | 0.00115 ± 1e-05 (W 2.04939e-05) | 0.00055 ± 5.7735e-06 (W 2.23607e-05) | 🔴 colmat 1.26× slower than `faer` | 🔴 colmat 2.64× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00215 ± 1.52753e-05 (W 2.16795e-05) | 0.00173 ± 2.08167e-05 (W 1.51658e-05) | 0.00086 ± 5.7735e-06 (W 2.16795e-05) | 🔴 colmat 1.24× slower than `faer` | 🔴 colmat 2.50× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00799 ± 1.1547e-05 (W 1.48728e-05) | 0.007399 ± 5.50757e-06 (W 9.1214e-06) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00802 ± 2e-05 (W 5.06083e-05) | 0.00795 ± 1.52753e-05 (W 5.53733e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.044709 ± 0.000138132 (W 0.000104174) | 0.058408 ± 6.51178e-05 (W 7.23132e-05) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044339 ± 9.80833e-05 (W 0.000102236) | 0.044298 ± 0.000155779 (W 0.000152709) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.068918 ± 9.65764e-05 (W 0.000172221) | 0.291871 ± 0.000320503 (W 0.00187237) | - | 🔵 colmat 4.24× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067678 ± 0.000162318 (W 0.00125462) | 0.067169 ± 0.000125835 (W 0.000221777) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.061568 ± 0.000157162 (W 9.21954e-05) | 0.068187 ± 0.000145583 (W 0.000826935) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.062399 ± 2.66521e-05 (W 0.000719202) | 0.068608 ± 6.1101e-05 (W 0.000127645) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.247842 ± 0.000946714 (W 0.00100291) | 0.258784 ± 0.000481769 (W 0.00130244) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.251672 ± 0.00116029 (W 0.00209166) | 0.259682 ± 0.00034846 (W 0.00104296) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.59683 ± 0.00197941 (W 0.0016046) | 0.711457 ± 0.00167102 (W 0.00161663) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.59833 ± 0.00327072 (W 0.001814) | 0.726407 ± 0.00256312 (W 0.00237483) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.009529 ± 9.00019e-05 (W 7.55394e-05) | 0.007 ± 8.3865e-05 (W 5.18189e-05) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012459 ± 0.000119151 (W 9.14729e-05) | 0.00801 ± 6.35085e-05 (W 6.26115e-05) | - | 🔴 colmat 1.56× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.041218 ± 0.000153461 (W 0.000134052) | 0.034469 ± 3.21455e-05 (W 6.44205e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.049658 ± 0.000196648 (W 0.000189938) | 0.037599 ± 0.00694007 (W 6.69194e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.110877 ± 0.000380881 (W 0.000137431) | 0.098767 ± 0.000171747 (W 0.000779923) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.133367 ± 0.000410975 (W 0.00146031) | 0.108838 ± 0.000256114 (W 0.00143093) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00935 ± 1.52753e-05 (W 5.80517e-05) | 0.00908 ± 2e-05 (W 3.86807e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00934 ± 5.50757e-05 (W 6.41872e-05) | 0.009 ± 5.03322e-05 (W 2.70222e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.042559 ± 0.000214554 (W 0.000133575) | 0.041708 ± 7.02163e-05 (W 0.000115758) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.042689 ± 0.000130512 (W 6.60553e-05) | 0.041578 ± 5.22526e-05 (W 0.000112358) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.117087 ± 0.000149923 (W 0.00140484) | 0.114536 ± 0.000290466 (W 0.00134742) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.117097 ± 0.00107421 (W 0.00134505) | 0.114006 ± 0.000275754 (W 0.000145363) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006669 ± 5.04017e-05 (W 4.92463e-05) | 0.006539 ± 1.70392e-05 (W 2.32852e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00817 ± 0.000101392 (W 4.01908e-05) | 0.0062 ± 3.60555e-05 (W 0.00010872) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.023469 ± 3.19427e-05 (W 0.000117691) | 0.027899 ± 0.000253246 (W 9.0111e-05) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.028679 ± 0.000321403 (W 0.000100598) | 0.022089 ± 0.000144684 (W 8.12299e-05) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.053219 ± 0.00011632 (W 0.000130227) | 0.067529 ± 7.08543e-05 (W 0.00101391) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.069078 ± 0.000319297 (W 0.00163299) | 0.054588 ± 0.000132742 (W 0.00026973) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.032839 ± 0.000141892 (W 0.000114978) | 0.037679 ± 0.000163707 (W 0.000121778) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.023559 ± 0.000557704 (W 4.54775e-05) | 0.033859 ± 9.86577e-05 (W 0.000138636) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.110277 ± 0.000521853 (W 0.00110766) | 0.169006 ± 0.00131397 (W 0.00130592) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.094127 ± 0.000279834 (W 0.000902258) | 0.121846 ± 0.000687677 (W 0.0010891) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.238482 ± 0.000689387 (W 0.00115721) | 0.371468 ± 0.000465721 (W 0.00120964) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.256672 ± 0.00503149 (W 0.0020872) | 0.298713 ± 0.00130501 (W 0.00219107) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.018649 ± 0.000180278 (W 0.000110437) | 0.023759 ± 0.000164895 (W 6.48167e-05) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.062858 ± 0.000221209 (W 0.000679455) | 0.116056 ± 0.000170295 (W 0.00143096) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.127937 ± 0.000141672 (W 0.000976451) | 0.253762 ± 0.000376937 (W 0.00102672) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00717 ± 9.2376e-05 (W 2.39625e-05) | 0.00711 ± 3.55012e-05 (W 3.55978e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00898 ± 4.12351e-05 (W 4.34994e-05) | 0.00709 ± 4.2194e-05 (W 0.000121055) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.022929 ± 0.000178214 (W 3.26696e-05) | 0.028729 ± 0.000178979 (W 9.89303e-05) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.02936 ± 0.000134797 (W 4.32805e-05) | 0.023399 ± 3.78594e-05 (W 5.87043e-05) | - | 🔴 colmat 1.25× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.052108 ± 0.000185041 (W 9.45738e-05) | 0.068928 ± 0.00049 (W 0.00114545) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.070258 ± 0.000323316 (W 0.00159869) | 0.055948 ± 0.000230363 (W 0.000139518) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.100027 ± 0.000395011 (W 0.00025592) | 0.113018 ± 0.000257943 (W 0.00116521) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.100838 ± 0.000854276 (W 0.000990338) | 0.113586 ± 0.000390062 (W 0.00108075) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.41986 ± 0.000649023 (W 0.0015257) | 0.432576 ± 0.0113708 (W 0.00194917) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.419937 ± 0.000723233 (W 0.00147626) | 0.447305 ± 0.00349887 (W 0.00535064) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.990258 ± 0.00168417 (W 0.00035073) | 1.27913 ± 0.00183521 (W 0.00208988) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.991497 ± 0.00112224 (W 0.00135799) | 1.28679 ± 0.00382578 (W 0.00662065) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.038249 ± 0.000135031 (W 9.56556e-05) | 0.077298 ± 0.000300056 (W 7.02802e-05) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.038479 ± 0.000198016 (W 0.000946124) | 0.077498 ± 0.000231877 (W 0.000829415) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.144166 ± 0.00052 (W 0.000732049) | 0.295071 ± 5.15978e-05 (W 0.00106565) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.145456 ± 0.000444826 (W 0.00111113) | 0.295311 ± 0.000227073 (W 0.00110643) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.345002 ± 0.000126674 (W 0.00199186) | 0.673298 ± 0.000933851 (W 0.00117289) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.349418 ± 0.000278733 (W 0.0015904) | 0.673413 ± 0.000460931 (W 0.00146383) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00069 ± 5.7735e-06 (W 1.64317e-05) | 0.00063 ± 3.60555e-05 (W 2.30217e-05) | 0.00068 ± 5.7735e-06 (W 3.27109e-05) | 🔴 colmat 1.10× slower than `faer` (marginal: spread 0.70 of the difference) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00159 ± 0 (W 2.38747e-05) | 0.00142 ± 5.7735e-06 (W 1.78885e-05) | 0.0015 ± 5.7735e-06 (W 2.50998e-05) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 1.06× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00276 ± 1.52753e-05 (W 1.68879e-05) | 0.00251 ± 1.1547e-05 (W 2.72617e-05) | 0.00255 ± 5.7735e-06 (W 2.44949e-05) | 🔴 colmat 1.10× slower than `faer` | 🔴 colmat 1.08× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00799 ± 1e-05 (W 4.71402e-05) | 0.00739 ± 5.50757e-06 (W 2.39625e-05) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00805 ± 2.13776e-05 (W 5.91295e-05) | 0.00795 ± 2.5632e-05 (W 6.51322e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.044839 ± 0.00012108 (W 0.00013672) | 0.058288 ± 0.000112398 (W 3.49285e-05) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044268 ± 7.75263e-05 (W 0.000140261) | 0.044209 ± 4.98698e-05 (W 0.00143279) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.068668 ± 6.18897e-05 (W 0.000807707) | 0.085157 ± 0.0135412 (W 0.0833181) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.067618 ± 8.5049e-05 (W 0.000170792) | 0.071138 ± 0.000479771 (W 0.000658354) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.052329 ± 0.000210072 (W 0.000107412) | 0.068428 ± 0.000334215 (W 0.000132363) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.052049 ± 0.000118899 (W 0.000151371) | 0.068998 ± 0.000266896 (W 0.00126678) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.152465 ± 0.000104978 (W 0.000276532) | 0.251393 ± 0.000263382 (W 0.0012098) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.152985 ± 0.00018791 (W 0.000265733) | 0.251722 ± 0.000564943 (W 0.00134303) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.33094 ± 0.000932395 (W 0.00129669) | 0.662219 ± 0.000336848 (W 0.00134612) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.332401 ± 0.00080421 (W 0.00182491) | 0.664679 ± 0.00123745 (W 0.00125244) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01011 ± 8.69502e-05 (W 5.31714e-05) | 0.00704 ± 4.6188e-05 (W 4.23462e-05) | - | 🔴 colmat 1.44× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01302 ± 0.000256535 (W 5.66683e-05) | 0.00803 ± 0.000149701 (W 7.31929e-05) | - | 🔴 colmat 1.62× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.042449 ± 0.000105357 (W 0.000131629) | 0.034949 ± 0.000151327 (W 9.78775e-05) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.051008 ± 0.000405533 (W 0.000124857) | 0.037549 ± 2e-05 (W 0.000155788) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.098987 ± 0.000905778 (W 0.000615506) | 0.100327 ± 0.000187751 (W 5.29925e-05) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.124036 ± 0.00127333 (W 0.00134231) | 0.107677 ± 0.0001253 (W 0.00138172) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006549 ± 0.000107193 (W 0.000160544) | 0.00922 ± 5.03322e-05 (W 0.000104019) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00632 ± 0.000170978 (W 0.000175869) | 0.00919 ± 1e-05 (W 6.63348e-05) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.022489 ± 0.000142945 (W 0.000361898) | 0.021329 ± 0.000109409 (W 0.000164534) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.02226 ± 0.000107364 (W 0.000167617) | 0.021379 ± 6.08276e-05 (W 0.000173781) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.047719 ± 0.000301812 (W 0.000412221) | 0.045779 ± 0.000410163 (W 0.000419387) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.047538 ± 0.000280999 (W 0.000503847) | 0.044588 ± 0.000233582 (W 0.000208416) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00668 ± 6.50641e-05 (W 5.06952e-05) | 0.00657 ± 2.64575e-05 (W 3.67315e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008299 ± 6.00028e-05 (W 5.35565e-05) | 0.00619 ± 7.93725e-05 (W 0.000145128) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02696 ± 0.000250054 (W 0.000201314) | 0.029089 ± 0.00043558 (W 0.000339706) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.028099 ± 0.000185203 (W 0.000203013) | 0.024929 ± 6.65833e-05 (W 0.000636117) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.059968 ± 0.00139433 (W 0.000719116) | 0.072257 ± 0.00083054 (W 0.00172812) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.065008 ± 0.000257099 (W 0.00136382) | 0.059778 ± 0.0003073 (W 0.0012295) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.033369 ± 0.00020108 (W 0.000120512) | 0.046889 ± 8.35484e-05 (W 0.00033088) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.025219 ± 0.00128293 (W 5.81137e-05) | 0.042968 ± 0.000206995 (W 0.000422909) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.076457 ± 0.00138144 (W 0.00203274) | 0.206074 ± 0.000721734 (W 0.00169837) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.075518 ± 0.000225456 (W 0.00040078) | 0.155575 ± 0.000469326 (W 0.000812922) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.184755 ± 0.000937794 (W 0.00142512) | 0.393878 ± 0.000547979 (W 0.00206043) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.207873 ± 0.000829874 (W 0.00189537) | 0.32022 ± 0.00265409 (W 0.00241334) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.019199 ± 0.000269147 (W 8.69902e-05) | 0.032799 ± 0.000309246 (W 0.00112482) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.044838 ± 9.45163e-05 (W 0.000157595) | 0.162286 ± 0.000935231 (W 0.00122201) | - | 🔵 colmat 3.62× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.125916 ± 0.000181596 (W 0.000217886) | 0.32351 ± 0.000616004 (W 0.00186085) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00748 ± 3.4641e-05 (W 3.98786e-05) | 0.00706 ± 3.21455e-05 (W 3.98648e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009289 ± 3.15753e-05 (W 5.28507e-05) | 0.00703 ± 5.50757e-05 (W 4.88999e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.022889 ± 0.000176163 (W 0.000137543) | 0.030489 ± 0.000382797 (W 0.0010351) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.029629 ± 0.000168028 (W 0.00022018) | 0.025359 ± 0.000137961 (W 0.000547526) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.049509 ± 0.000182237 (W 0.000336963) | 0.070408 ± 0.000913165 (W 0.000646777) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.067738 ± 0.00026 (W 0.00169694) | 0.059728 ± 4.93288e-05 (W 0.000339089) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.102097 ± 0.000717336 (W 0.000205086) | 0.112487 ± 0.000435351 (W 0.00131765) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.101907 ± 0.000390512 (W 9.55887e-05) | 0.112967 ± 0.000151461 (W 0.00124998) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.336609 ± 0.000452516 (W 0.00153122) | 0.414937 ± 0.000349544 (W 0.00121703) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.338009 ± 0.00208029 (W 0.00133117) | 0.416219 ± 0.000715123 (W 0.00127932) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.709478 ± 0.003554 (W 0.00281607) | 1.18845 ± 0.00114287 (W 0.00212312) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.709738 ± 0.00109173 (W 0.00291062) | 1.19012 ± 0.000393061 (W 0.00173522) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.039278 ± 0.000252528 (W 0.000111446) | 0.077097 ± 0.000139715 (W 6.34295e-05) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.039658 ± 0.000169087 (W 0.000162419) | 0.077428 ± 0.000122202 (W 0.00110985) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.148196 ± 0.000754807 (W 0.000291942) | 0.29439 ± 0.00117351 (W 0.00163844) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.148896 ± 0.000685952 (W 0.000334021) | 0.294931 ± 0.00102922 (W 0.00164076) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.34712 ± 0.00108974 (W 0.00137823) | 0.673201 ± 0.000342931 (W 0.00220992) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.34782 ± 0.00218521 (W 0.00174411) | 0.673259 ± 0.00615591 (W 0.00149754) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00084 ± 5.7735e-06 (W 2e-05) | 0.00063 ± 1.1547e-05 (W 2.30217e-05) | 0.00068 ± 5.7735e-07 (W 2.77489e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.24× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00192 ± 9.50438e-06 (W 1.51658e-05) | 0.00144 ± 1.1547e-05 (W 2.60768e-05) | 0.00151 ± 1.52753e-05 (W 3.42053e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.27× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00326 ± 1.52753e-05 (W 2.07364e-05) | 0.002539 ± 5.50757e-06 (W 2.79285e-05) | 0.00255 ± 1.52753e-05 (W 3.78153e-05) | 🔴 colmat 1.28× slower than `faer` | 🔴 colmat 1.28× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.007989 ± 2.00083e-05 (W 2.53219e-05) | 0.00738 ± 5.7735e-06 (W 2.09093e-05) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00802 ± 1.1547e-05 (W 5.69491e-05) | 0.00802 ± 1.58219e-05 (W 2.10998e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.044748 ± 4.03361e-05 (W 9.95605e-05) | 0.058268 ± 2.08167e-05 (W 4.80853e-05) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044188 ± 6.30106e-05 (W 0.000139371) | 0.044319 ± 0.000102357 (W 0.000145211) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.068368 ± 9.11976e-05 (W 0.000156141) | 0.085297 ± 0.00018284 (W 0.00102221) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067469 ± 0.000237887 (W 0.000155885) | 0.071758 ± 7e-05 (W 0.000214094) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.050299 ± 0.000332916 (W 9.37459e-05) | 0.068159 ± 0.000202693 (W 0.00132935) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.050089 ± 1.57162e-05 (W 0.000175129) | 0.068528 ± 0.000255147 (W 8.25833e-05) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.138317 ± 0.000426897 (W 0.000405013) | 0.252245 ± 0.000145645 (W 0.00175608) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.138727 ± 0.000437638 (W 0.000327052) | 0.253085 ± 0.0160248 (W 0.00114467) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 0.290455 ± 0.00119968 (W 0.00208021) | 0.669179 ± 0.00157653 (W 0.00268461) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.290311 ± 0.00107453 (W 0.0017164) | 0.668879 ± 0.00114117 (W 0.00166101) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01002 ± 5.50757e-05 (W 7.57245e-05) | 0.00706 ± 4.95008e-05 (W 3.79631e-05) | - | 🔴 colmat 1.42× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012919 ± 2.13776e-05 (W 8.12053e-05) | 0.00808 ± 7.02377e-05 (W 6.44764e-05) | - | 🔴 colmat 1.60× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.042609 ± 0.000191398 (W 9.64432e-05) | 0.034689 ± 9.07377e-05 (W 3.68944e-05) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.050618 ± 9.34897e-05 (W 0.000124464) | 0.037489 ± 0.000406325 (W 0.00158402) | - | 🔴 colmat 1.35× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.101507 ± 0.000259257 (W 0.00103033) | 0.100057 ± 0.000194679 (W 0.00130295) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.128416 ± 0.00088636 (W 0.00186208) | 0.107937 ± 0.000196044 (W 0.0016782) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00583 ± 8.0829e-05 (W 0.000126886) | 0.009279 ± 4.07063e-05 (W 5.14782e-05) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00535 ± 7.76745e-05 (W 0.000131263) | 0.009199 ± 3.25115e-05 (W 5.76559e-05) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.01589 ± 0.00181583 (W 0.000253072) | 0.022409 ± 0.000374767 (W 0.000384804) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.015859 ± 0.0002805 (W 0.000283822) | 0.02209 ± 7.60285e-05 (W 0.000359038) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.032709 ± 0.000487887 (W 0.000442527) | 0.053878 ± 0.000211069 (W 0.000619763) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.032319 ± 0.000593408 (W 0.000526754) | 0.052648 ± 0.00071316 (W 0.000683594) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006659 ± 4.17971e-05 (W 3.32596e-05) | 0.00661 ± 1.47309e-05 (W 2.2939e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00819 ± 3.60555e-05 (W 6.20911e-05) | 0.00619 ± 0.000136847 (W 4.35683e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.030189 ± 0.000415853 (W 0.000563578) | 0.029699 ± 0.000219864 (W 0.000191228) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.027989 ± 1.57162e-05 (W 0.000221766) | 0.02531 ± 0.000105025 (W 0.000122156) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.060488 ± 9.86577e-05 (W 0.000962884) | 0.072448 ± 0.0012752 (W 0.000318889) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.065718 ± 0.000409854 (W 0.00126628) | 0.061928 ± 0.000241316 (W 0.0016587) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.033439 ± 0.000174356 (W 0.000110905) | 0.052968 ± 0.000485582 (W 0.00078946) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.02384 ± 5.83981e-05 (W 6.4142e-05) | 0.048339 ± 0.000267501 (W 0.000763846) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.076177 ± 0.0024573 (W 0.00135443) | 0.230162 ± 0.00259599 (W 0.00305098) | - | 🔵 colmat 3.02× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.074448 ± 0.000236291 (W 0.000667236) | 0.184895 ± 0.00167785 (W 0.00212981) | - | 🔵 colmat 2.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.171735 ± 0.000240763 (W 0.00144207) | 0.439736 ± 0.00207447 (W 0.00318291) | - | 🔵 colmat 2.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.196714 ± 0.000750267 (W 0.0027302) | 0.37694 ± 0.00228692 (W 0.00508099) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.01935 ± 0.00027573 (W 0.000111823) | 0.038399 ± 0.000195577 (W 0.000791992) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.045238 ± 0.000147208 (W 7.87547e-05) | 0.186744 ± 0.00214325 (W 0.00325959) | - | 🔵 colmat 4.13× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.117217 ± 0.000349052 (W 0.000427339) | 0.371719 ± 0.00531706 (W 0.0059482) | - | 🔵 colmat 3.17× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00749 ± 6.50641e-05 (W 3.78444e-05) | 0.00706 ± 5.85947e-05 (W 3.91306e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00936 ± 3.55012e-05 (W 4.3581e-05) | 0.00706 ± 4.6188e-05 (W 0.000146578) | - | 🔴 colmat 1.33× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.024499 ± 0.000109697 (W 0.000220105) | 0.03074 ± 0.000173235 (W 0.000152217) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.029579 ± 0.000307463 (W 0.000104159) | 0.025799 ± 0.000268794 (W 0.00139912) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.054598 ± 0.000181751 (W 0.000330636) | 0.070238 ± 0.000240069 (W 0.000436578) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.067448 ± 0.000441626 (W 0.000259768) | 0.061088 ± 0.000221209 (W 0.00171238) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.102286 ± 0.00208083 (W 0.000196813) | 0.112438 ± 0.000732444 (W 0.00136035) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.102648 ± 0.000766536 (W 0.000181362) | 0.113827 ± 0.000417612 (W 0.0012517) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.335724 ± 0.000532246 (W 0.00155344) | 0.418149 ± 0.00181313 (W 0.00297097) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.337169 ± 0.00247158 (W 0.00157655) | 0.419997 ± 0.00135829 (W 0.00193581) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.704338 ± 0.000966264 (W 0.00383055) | 1.19747 ± 0.000302179 (W 0.00383618) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.7082 ± 0.00279578 (W 0.00330997) | 1.20185 ± 0.003014 (W 0.00122399) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.040148 ± 9.59705e-05 (W 6.61377e-05) | 0.077098 ± 6.9282e-05 (W 7.34255e-05) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.039598 ± 0.000222816 (W 5.87299e-05) | 0.077368 ± 0.000115052 (W 0.000954571) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.148296 ± 0.000492098 (W 0.000375447) | 0.294292 ± 0.00102042 (W 0.00122933) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.149006 ± 0.00055193 (W 0.00154697) | 0.295132 ± 0.00103226 (W 0.00157728) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.356193 ± 0.000754271 (W 0.00204343) | 0.672619 ± 0.00162833 (W 0.00165954) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.3566 ± 0.000552923 (W 0.00215322) | 0.682489 ± 0.000351858 (W 0.00174863) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00085 ± 5.7735e-06 (W 1.48324e-05) | 0.00063 ± 1.52753e-05 (W 2.44949e-05) | 0.00068 ± 5.7735e-06 (W 1.64317e-05) | 🔴 colmat 1.35× slower than `faer` | 🔴 colmat 1.25× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00193 ± 2e-05 (W 1.81659e-05) | 0.00144 ± 2.51661e-05 (W 2.29608e-05) | 0.00149 ± 5.7735e-06 (W 2.60768e-05) | 🔴 colmat 1.34× slower than `faer` | 🔴 colmat 1.30× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00326 ± 1.52753e-05 (W 1.64317e-05) | 0.00252 ± 1e-05 (W 2.30217e-05) | 0.00253 ± 5.7735e-06 (W 2.16795e-05) | 🔴 colmat 1.29× slower than `faer` | 🔴 colmat 1.29× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.007619 ± 5.50757e-06 (W 3.70702e-05) | 0.00756 ± 1.67432e-05 (W 4.10512e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00799 ± 6.08276e-06 (W 4.24641e-05) | 0.007999 ± 3.82143e-05 (W 5.17513e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.043259 ± 0.000128582 (W 0.000128329) | 0.043098 ± 2.51064e-05 (W 0.000139176) | - | · | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044288 ± 4.56107e-05 (W 0.00013115) | 0.044209 ± 0.000177764 (W 0.00103882) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.065068 ± 5.56776e-05 (W 0.000241005) | 0.064408 ± 0.000143807 (W 0.00117891) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067287 ± 0.000207993 (W 0.000148025) | 0.067057 ± 0.00010591 (W 0.000132853) | - | · | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.061619 ± 0.000263913 (W 6.94061e-05) | 0.068098 ± 0.000265903 (W 0.000200871) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.062298 ± 7.49689e-05 (W 0.000110589) | 0.068808 ± 0.000357957 (W 0.000109463) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.246162 ± 0.000719862 (W 0.0016432) | 0.258771 ± 0.000266226 (W 0.00101687) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.250162 ± 0.00119519 (W 0.001524) | 0.259623 ± 0.000401448 (W 0.00136663) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.597182 ± 0.00140458 (W 0.0021967) | 0.709008 ± 0.00133589 (W 0.00170013) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.59924 ± 0.00270704 (W 0.00122813) | 0.726042 ± 0.00116134 (W 0.00141304) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.00972 ± 3.05014e-05 (W 4.30616e-05) | 0.00696 ± 6.4291e-05 (W 4.47214e-05) | - | 🔴 colmat 1.40× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01079 ± 0.000104403 (W 6.5675e-05) | 0.007909 ± 0.0001395 (W 7.21263e-05) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.042239 ± 0.000536936 (W 0.000880575) | 0.034529 ± 0.000111505 (W 5.43342e-05) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.044638 ± 0.000109576 (W 0.000211703) | 0.036979 ± 0.000104403 (W 0.000114857) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.113647 ± 0.000397157 (W 0.00109328) | 0.098727 ± 0.000216564 (W 0.00107847) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.122037 ± 7.78267e-05 (W 0.00132116) | 0.107866 ± 0.000276117 (W 0.00188122) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00953 ± 4.98698e-05 (W 5.00819e-05) | 0.00918 ± 3.78594e-05 (W 5.09902e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.012009 ± 0.002338 (W 1.45705e-05) | 0.00922 ± 3.78594e-05 (W 0.000368746) | - | 🔴 colmat 1.30× slower than `faer` (marginal: spread 0.85 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.042698 ± 0.000161401 (W 0.000126176) | 0.041789 ± 0.000583473 (W 0.000107295) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.042899 ± 5.68624e-05 (W 8.24209e-05) | 0.041539 ± 0.000217332 (W 6.63325e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.119717 ± 0.000297742 (W 0.00104604) | 0.117956 ± 0.00014167 (W 0.00156062) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.119346 ± 0.000224286 (W 0.000994423) | 0.117186 ± 0.000139429 (W 0.000183552) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00571 ± 0.000101489 (W 6.0035e-05) | 0.005229 ± 1.00167e-05 (W 2.53417e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00688 ± 3.4641e-05 (W 6.26099e-05) | 0.00611 ± 1.73205e-05 (W 3.72988e-05) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02126 ± 9.09414e-05 (W 8.08783e-05) | 0.019079 ± 6.20511e-05 (W 8.75289e-05) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.023669 ± 0.000235967 (W 0.000753377) | 0.02128 ± 0.000135242 (W 0.000113566) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.048388 ± 7.93725e-05 (W 9.30924e-05) | 0.044649 ± 9.16515e-05 (W 9.21965e-05) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.057298 ± 0.00027916 (W 0.00146024) | 0.053158 ± 0.00037 (W 0.00142301) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.020529 ± 4.3589e-05 (W 4.21462e-05) | 0.031939 ± 6.65833e-05 (W 9.65401e-05) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.022149 ± 9.53939e-05 (W 4.53674e-05) | 0.033459 ± 0.00026831 (W 0.000139284) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.085158 ± 0.000175015 (W 0.000200911) | 0.117277 ± 0.000730028 (W 0.00131969) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.089227 ± 0.000127241 (W 0.000231149) | 0.119946 ± 0.000865182 (W 0.00108735) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.215893 ± 0.00289342 (W 0.00106386) | 0.269431 ± 0.000297029 (W 0.00113827) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.240842 ± 0.00327059 (W 0.00171222) | 0.294141 ± 0.000271382 (W 0.00241694) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00875 ± 6.08276e-06 (W 2.95686e-05) | 0.017879 ± 0.0001245 (W 0.000147422) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.039278 ± 0.000186602 (W 5.22714e-05) | 0.070798 ± 0.000195533 (W 0.000154364) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.100837 ± 0.000102632 (W 0.00108624) | 0.157615 ± 0.000313508 (W 0.00143821) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.00626 ± 5.07642e-05 (W 2.92438e-05) | 0.00585 ± 1.1547e-05 (W 2.97859e-05) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00745 ± 5.97523e-05 (W 2.50659e-05) | 0.006909 ± 3.56417e-05 (W 6.94277e-05) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021659 ± 8.66622e-05 (W 0.000221256) | 0.02023 ± 9.32577e-05 (W 5.06379e-05) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.024219 ± 5.68624e-05 (W 3.92709e-05) | 0.022639 ± 9.76746e-05 (W 0.000667734) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.049429 ± 2.95014e-05 (W 7.05691e-05) | 0.046618 ± 6.05558e-05 (W 0.000259985) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.058908 ± 6.80686e-05 (W 0.00160251) | 0.054689 ± 8.77895e-05 (W 0.00105347) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.100766 ± 2.45832e-05 (W 0.000153588) | 0.114077 ± 0.000275712 (W 0.000429601) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.101597 ± 0.000516843 (W 0.00110529) | 0.114826 ± 0.000321994 (W 0.000915669) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.420287 ± 0.000444116 (W 0.00226905) | 0.440366 ± 0.00316186 (W 0.00158765) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.421407 ± 0.00179003 (W 0.00126857) | 0.453215 ± 0.00256039 (W 0.00294642) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.987297 ± 0.00301806 (W 0.000689474) | 1.28728 ± 0.00223083 (W 0.00233302) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.992688 ± 0.00509521 (W 0.000803472) | 1.29079 ± 0.00208623 (W 0.00265464) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.038799 ± 3.46458e-05 (W 3.2362e-05) | 0.077338 ± 0.000190596 (W 7.94311e-05) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.039139 ± 2.6839e-05 (W 0.000107881) | 0.077558 ± 0.000165227 (W 0.000939426) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.143345 ± 0.000797264 (W 0.00119843) | 0.291081 ± 0.000338922 (W 0.00144016) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.145355 ± 0.00180249 (W 0.000200901) | 0.291351 ± 0.000223515 (W 0.00100994) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.347589 ± 0.000766008 (W 0.00175849) | 0.674169 ± 0.000728814 (W 0.00139683) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.352818 ± 0.00135194 (W 0.00159968) | 0.673929 ± 0.000680654 (W 0.00126357) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00072 ± 1.1547e-05 (W 2.48998e-05) | 0.00073 ± 5.7735e-06 (W 1.73205e-05) | 0.00063 ± 5.7735e-06 (W 2.78604e-05) | · | 🔴 colmat 1.14× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00161 ± 5.7735e-06 (W 2.30217e-05) | 0.00166 ± 5.7735e-06 (W 1.64317e-05) | 0.0014 ± 0 (W 2.68328e-05) | · | 🔴 colmat 1.15× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00262 ± 5.7735e-06 (W 2.07364e-05) | 0.00271 ± 1e-05 (W 4.43847e-05) | 0.00242 ± 5.7735e-06 (W 2.03027e-05) | · | 🔴 colmat 1.08× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00756 ± 5.7735e-07 (W 3.57799e-05) | 0.00759 ± 1.1547e-05 (W 3.92721e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00803 ± 1.52753e-05 (W 4.32007e-05) | 0.00799 ± 4e-05 (W 2.90207e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.043018 ± 6.76782e-05 (W 0.000129477) | 0.043318 ± 6.4291e-05 (W 9.51798e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044188 ± 4.03361e-05 (W 0.000121203) | 0.044179 ± 4.45758e-05 (W 0.000143876) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.064428 ± 0.000185599 (W 0.000929505) | 0.067899 ± 4.03361e-05 (W 0.000414225) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067438 ± 8.07279e-05 (W 0.000186414) | 0.071018 ± 0.000220076 (W 0.000384996) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.052269 ± 0.00155667 (W 0.000164394) | 0.068108 ± 0.00116568 (W 0.000131405) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.052129 ± 0.000252052 (W 0.000166677) | 0.068558 ± 0.000281484 (W 8.53182e-05) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.151826 ± 0.000350714 (W 0.000256473) | 0.251412 ± 0.000411955 (W 0.00118602) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.152646 ± 0.000281544 (W 0.000252119) | 0.251543 ± 0.00104945 (W 0.000978264) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.330589 ± 0.000490657 (W 0.00165289) | 0.662379 ± 0.00079613 (W 0.00203496) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.33078 ± 0.00193473 (W 0.00189599) | 0.66336 ± 0.00213381 (W 0.0012654) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.01027 ± 4.04145e-05 (W 5.94575e-05) | 0.007089 ± 1.53948e-05 (W 4.54555e-05) | - | 🔴 colmat 1.45× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.011519 ± 0.00129632 (W 4.50799e-05) | 0.00803 ± 1.47309e-05 (W 3.40338e-05) | - | 🔴 colmat 1.43× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.043349 ± 9.25761e-05 (W 8.28203e-05) | 0.034899 ± 8.45399e-05 (W 7.22717e-05) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.046099 ± 0.000252924 (W 0.00113466) | 0.036789 ± 0.000127671 (W 0.000100747) | - | 🔴 colmat 1.25× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.101537 ± 0.000654548 (W 0.000553522) | 0.100037 ± 0.000180831 (W 0.0011655) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.112686 ± 0.00056202 (W 0.00200028) | 0.106967 ± 0.000205687 (W 0.00176453) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006339 ± 0.000100201 (W 0.000214635) | 0.00931 ± 5.7735e-06 (W 6.55744e-05) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00656 ± 2.5697e-05 (W 0.000116319) | 0.00927 ± 3.01164e-05 (W 3.98271e-05) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.02184 ± 0.000155211 (W 0.000227658) | 0.02079 ± 0.000164348 (W 5.60473e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.021809 ± 0.000166433 (W 0.000185125) | 0.021339 ± 0.000173494 (W 0.000139571) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.046588 ± 6.03352e-05 (W 0.000351753) | 0.045098 ± 0.00016241 (W 0.000148214) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.046749 ± 0.000534706 (W 0.000949306) | 0.045038 ± 0.000137769 (W 0.000477944) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005739 ± 4.7085e-05 (W 3.03513e-05) | 0.00523 ± 2.13776e-05 (W 4.10024e-05) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00691 ± 3.4641e-05 (W 5.7239e-05) | 0.006109 ± 6.45006e-05 (W 6.2763e-05) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020699 ± 0.000127671 (W 0.000132088) | 0.021409 ± 0.000510402 (W 0.000111885) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.023019 ± 0.000225227 (W 0.000454914) | 0.02429 ± 8.06866e-05 (W 0.000720336) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.043579 ± 0.000267208 (W 0.000233483) | 0.049119 ± 7.0946e-05 (W 0.00106693) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.053599 ± 0.000358911 (W 0.00171162) | 0.058119 ± 0.000332431 (W 0.000917269) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.020979 ± 0.000565892 (W 3.12602e-05) | 0.040589 ± 0.000419047 (W 0.000277473) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.022449 ± 0.000175784 (W 7.19743e-05) | 0.042519 ± 8.17333e-05 (W 0.000916622) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.066158 ± 0.000223681 (W 0.000819005) | 0.149335 ± 0.000291651 (W 0.00132708) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.070587 ± 0.000289097 (W 0.000106762) | 0.153565 ± 0.000927475 (W 0.000554578) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.164364 ± 0.000388622 (W 0.000358051) | 0.286462 ± 0.00116094 (W 0.00107847) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.196274 ± 0.000417457 (W 0.00150995) | 0.31696 ± 0.00166645 (W 0.00249413) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00901 ± 0.000318492 (W 2.24121e-05) | 0.026499 ± 1e-05 (W 0.000251979) | - | 🔵 colmat 2.94× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.040209 ± 6.02771e-05 (W 5.52467e-05) | 0.114357 ± 0.000478686 (W 0.00164985) | - | 🔵 colmat 2.84× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.113397 ± 0.000704739 (W 0.000215996) | 0.225563 ± 0.000121657 (W 0.00101871) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006639 ± 3.59212e-05 (W 3.99024e-05) | 0.005799 ± 1.00167e-05 (W 2.5499e-05) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007859 ± 1.12694e-05 (W 3.91561e-05) | 0.00689 ± 2.04206e-05 (W 7.46324e-05) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021519 ± 0.000198578 (W 0.000113244) | 0.021979 ± 0.000125831 (W 0.000418031) | - | · | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.024079 ± 5.89661e-05 (W 0.000203131) | 0.024789 ± 0.000121698 (W 0.00068029) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.043369 ± 0.000425793 (W 0.000156519) | 0.048029 ± 0.000112947 (W 0.000295781) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.056588 ± 0.000289794 (W 0.00192897) | 0.058398 ± 0.000227156 (W 0.00210393) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.103617 ± 0.000636475 (W 0.000187537) | 0.114546 ± 0.000425594 (W 0.000174403) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.102917 ± 0.000181348 (W 6.97008e-05) | 0.114397 ± 0.000522781 (W 0.000230085) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.33502 ± 0.00256136 (W 0.00157875) | 0.421557 ± 0.00112819 (W 0.00183873) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.338991 ± 0.00165037 (W 0.00157082) | 0.421907 ± 0.00183557 (W 0.00131998) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.710701 ± 0.00255153 (W 0.00162994) | 1.19957 ± 0.00118286 (W 0.00185008) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.710638 ± 0.00363818 (W 0.00178718) | 1.19949 ± 0.00235456 (W 0.00154801) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.040209 ± 0.000156174 (W 0.000120906) | 0.077547 ± 0.000175548 (W 7.71123e-05) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.040388 ± 0.000177643 (W 0.00010585) | 0.077437 ± 0.000486861 (W 0.00135167) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.146916 ± 0.00121897 (W 0.000319096) | 0.290821 ± 0.000541341 (W 0.00133009) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.148035 ± 0.000571253 (W 0.000385141) | 0.291032 ± 0.000133994 (W 0.00171382) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.349979 ± 0.000695692 (W 0.00147032) | 0.6726 ± 0.000373045 (W 0.00146167) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.348899 ± 0.00201334 (W 0.00125597) | 0.682872 ± 0.000708335 (W 0.00169978) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00087 ± 5.7735e-06 (W 2.07364e-05) | 0.00075 ± 5.7735e-06 (W 1.92354e-05) | 0.00063 ± 5.7735e-06 (W 2.94958e-05) | 🔴 colmat 1.16× slower than `faer` | 🔴 colmat 1.38× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00192 ± 5.7735e-06 (W 1.94936e-05) | 0.00164 ± 1e-05 (W 2.05475e-05) | 0.0014 ± 5.7735e-06 (W 2.82843e-05) | 🔴 colmat 1.17× slower than `faer` | 🔴 colmat 1.37× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00306 ± 1.47309e-05 (W 2.18678e-05) | 0.00269 ± 5.7735e-06 (W 2.96648e-05) | 0.002409 ± 5.19615e-06 (W 2.70592e-05) | 🔴 colmat 1.14× slower than `faer` | 🔴 colmat 1.27× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.00757 ± 0 (W 2.97523e-05) | 0.007619 ± 5.19615e-06 (W 2.66871e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00798 ± 1.18462e-05 (W 4.29802e-05) | 0.00799 ± 1.95021e-05 (W 2.65744e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.043008 ± 0.000131607 (W 9.84749e-05) | 0.043319 ± 5.50757e-05 (W 0.00011266) | - | · | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.044158 ± 4.63717e-05 (W 0.00015269) | 0.044189 ± 0.000100848 (W 0.000108496) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.064838 ± 9.0185e-05 (W 0.0011403) | 0.068498 ± 0.00023516 (W 0.000157575) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.067358 ± 6.50641e-05 (W 0.000256223) | 0.071828 ± 0.00020361 (W 0.00134025) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.050079 ± 0.00100182 (W 0.000170968) | 0.067878 ± 0.000210832 (W 0.000117661) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.050199 ± 0.000185275 (W 0.000172105) | 0.068528 ± 0.000107037 (W 0.000165442) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.137796 ± 0.000107364 (W 0.000467312) | 0.251415 ± 0.000783289 (W 0.00178778) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.138627 ± 5.16753e-05 (W 0.000397024) | 0.253112 ± 0.000549651 (W 0.0015747) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.290144 ± 0.0008268 (W 0.00216867) | 0.667799 ± 0.000859568 (W 0.00281844) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.288985 ± 0.00274192 (W 0.00246815) | 0.668722 ± 0.000372066 (W 0.00189055) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.010309 ± 0.000106927 (W 3.62105e-05) | 0.00718 ± 5.1316e-05 (W 4.23816e-05) | - | 🔴 colmat 1.44× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.011379 ± 0.00166537 (W 5.61632e-05) | 0.00802 ± 5.05008e-05 (W 6.8791e-05) | - | 🔴 colmat 1.42× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.043269 ± 7.55006e-05 (W 9.35056e-05) | 0.034689 ± 0.000123423 (W 4.89203e-05) | - | 🔴 colmat 1.25× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.046438 ± 0.00674466 (W 5.60116e-05) | 0.036749 ± 7.93725e-05 (W 0.00012498) | - | 🔴 colmat 1.26× slower than `faer` (marginal: spread 0.70 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.104567 ± 0.000373002 (W 0.000640414) | 0.099657 ± 0.000150444 (W 0.000168233) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.117046 ± 0.000645516 (W 0.00174247) | 0.106586 ± 0.000159124 (W 0.00193312) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005249 ± 4.03361e-05 (W 0.000118841) | 0.00936 ± 5.1316e-05 (W 6.34823e-05) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00538 ± 4.3589e-05 (W 0.00016769) | 0.009309 ± 1.53948e-05 (W 7.52483e-05) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.01497 ± 0.000147422 (W 0.000297589) | 0.0218 ± 4.6188e-05 (W 0.0003525) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01542 ± 3.64737e-05 (W 0.000283586) | 0.021859 ± 0.000191103 (W 0.00027108) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.030689 ± 0.000168662 (W 0.000254478) | 0.052608 ± 0.000323306 (W 0.000607314) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.03027 ± 4.68722e-05 (W 0.000216638) | 0.052519 ± 0.00035543 (W 0.000468166) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00579 ± 5.68624e-05 (W 3.57939e-05) | 0.00525 ± 1.52753e-05 (W 3.22831e-05) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006779 ± 3.80832e-05 (W 5.16353e-05) | 0.00611 ± 0.000122202 (W 3.85772e-05) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020639 ± 0.00016773 (W 0.000181351) | 0.02166 ± 0.000185689 (W 0.000190411) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.02263 ± 0.000123654 (W 0.000395291) | 0.024499 ± 0.00020297 (W 0.000167685) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.043488 ± 0.000176485 (W 0.000153373) | 0.049748 ± 4.98029e-05 (W 0.000401453) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.054328 ± 0.000146323 (W 0.000266796) | 0.060568 ± 6.62596e-05 (W 0.000351118) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.021 ± 3.19427e-05 (W 2.12179e-05) | 0.046639 ± 0.000474271 (W 0.000410394) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.022419 ± 3.751e-05 (W 5.51833e-05) | 0.048689 ± 0.000429616 (W 0.000845509) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.064228 ± 0.000179258 (W 0.000124218) | 0.178934 ± 0.00228476 (W 0.00223636) | - | 🔵 colmat 2.79× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.069078 ± 0.000258342 (W 0.000371096) | 0.183034 ± 0.000776891 (W 0.00167523) | - | 🔵 colmat 2.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.151555 ± 0.000133593 (W 0.00145609) | 0.34648 ± 0.000730528 (W 0.00279087) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.184984 ± 0.000796013 (W 0.00199171) | 0.37537 ± 0.00250653 (W 0.00205507) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.00902 ± 0.000124231 (W 3.07132e-05) | 0.031789 ± 0.000162583 (W 0.000579756) | - | 🔵 colmat 3.52× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.040339 ± 7.23418e-05 (W 4.63918e-05) | 0.143846 ± 0.00249357 (W 0.00190313) | - | 🔵 colmat 3.57× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.109627 ± 0.000280957 (W 0.000180918) | 0.281731 ± 0.00575734 (W 0.00323157) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0066 ± 3.27159e-05 (W 3.51027e-05) | 0.00603 ± 1.1547e-05 (W 5.07957e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.00784 ± 4.16333e-05 (W 6.53468e-05) | 0.006919 ± 5.50757e-06 (W 5.65791e-05) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021559 ± 6.95006e-05 (W 0.000172393) | 0.022299 ± 0.000135277 (W 0.000143078) | - | · | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.024119 ± 0.00012995 (W 0.000160475) | 0.02493 ± 0.000195577 (W 0.000133155) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.043099 ± 4.50925e-05 (W 0.000283518) | 0.048468 ± 0.000153353 (W 0.000387589) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.055758 ± 0.000131149 (W 0.00184993) | 0.060308 ± 8.544e-05 (W 0.00146834) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.102817 ± 0.000401042 (W 0.000148139) | 0.114216 ± 0.000766708 (W 0.0012434) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.102717 ± 0.000137041 (W 0.000167034) | 0.114367 ± 0.00151509 (W 0.000168272) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.335224 ± 0.000107937 (W 0.00217918) | 0.424492 ± 0.014206 (W 0.00183899) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.335443 ± 0.00182273 (W 0.00107361) | 0.424386 ± 0.00851559 (W 0.00280337) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.704058 ± 0.000375295 (W 0.00229525) | 1.20725 ± 0.00294171 (W 0.00292581) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.706841 ± 0.00275888 (W 0.00269427) | 1.21309 ± 0.00217516 (W 0.00165279) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.040878 ± 0.000348105 (W 5.85517e-05) | 0.077177 ± 0.000126532 (W 7.9141e-05) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.040419 ± 0.000143643 (W 8.78635e-05) | 0.077449 ± 0.000191366 (W 0.000103442) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.147446 ± 0.00121098 (W 0.000203468) | 0.290741 ± 0.000682409 (W 0.00151183) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.148297 ± 0.000385622 (W 0.000366292) | 0.291872 ± 0.000902442 (W 0.00176762) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.360039 ± 0.0019514 (W 0.00185553) | 0.671357 ± 0.00130499 (W 0.00153876) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.36112 ± 0.00168805 (W 0.0029468) | 0.674517 ± 0.00358567 (W 0.00187576) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.00088 ± 5.7735e-06 (W 2.18906e-05) | 0.00075 ± 1.1547e-05 (W 2.16795e-05) | 0.00063 ± 0 (W 2.82843e-05) | 🔴 colmat 1.17× slower than `faer` | 🔴 colmat 1.40× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00191 ± 2.51661e-05 (W 2.1822e-05) | 0.00166 ± 1.73205e-05 (W 2.73861e-05) | 0.00139 ± 0 (W 3.04959e-05) | 🔴 colmat 1.15× slower than `faer` | 🔴 colmat 1.37× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00308 ± 5.7735e-06 (W 2.87263e-05) | 0.00268 ± 1e-05 (W 2.02287e-05) | 0.0024 ± 6.08276e-06 (W 2.51635e-05) | 🔴 colmat 1.15× slower than `faer` | 🔴 colmat 1.28× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `lu/32`, `lu/64`, `lu/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `solve/32`, `solve/64`, `solve/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.030029 ± 0.000175594 (W 6.43288e-05) | 0.179785 ± 5.85947e-05 (W 0.000979297) | - | 🔵 colmat 5.99× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.034209 ± 0.000145717 (W 0.000451298) | 0.040769 ± 0.000393954 (W 0.000344558) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.162165 ± 0.00120477 (W 0.00128616) | 0.473178 ± 0.0014225 (W 0.00253845) | - | 🔵 colmat 2.92× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.169175 ± 0.000262298 (W 0.00121992) | 0.210714 ± 0.0024095 (W 0.00234073) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.30588 ± 0.00541557 (W 0.00561186) | 2.16267 ± 0.00489861 (W 0.00450253) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.02416 ± 0.000808723 (W 0.00143966) | 1.7718 ± 0.0128151 (W 0.00462879) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.33359 ± 0.00388157 (W 0.00242152) | 1.56698 ± 0.00195052 (W 0.00176728) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.33638 ± 0.000945611 (W 0.00230549) | 1.56587 ± 0.00113297 (W 0.00199636) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.8733 ± 0.0189739 (W 0.00888559) | 6.20864 ± 0.0174604 (W 0.00474807) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.83508 ± 0.0230758 (W 0.0140997) | 6.22861 ± 0.0403738 (W 0.00227534) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 28.0849 ± 0.0647148 (W 0.0403227) | 29.469 ± 0.0454678 (W 0.0128198) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 28.4515 ± 0.135131 (W 0.0330015) | 29.6784 ± 0.228518 (W 0.0207703) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.218573 ± 0.000700117 (W 0.00120035) | 0.205105 ± 0.0138732 (W 0.0010812) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.204794 ± 0.000491739 (W 0.00126994) | 0.204944 ± 0.000671072 (W 0.00175357) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.3319 ± 0.00510279 (W 0.00387285) | 2.04357 ± 0.0116517 (W 0.0050002) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.43633 ± 0.00833529 (W 0.00475017) | 1.979 ± 0.0117919 (W 0.00503302) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.0243 ± 0.0432423 (W 0.0157959) | 16.8507 ± 0.0104133 (W 0.0108598) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 10.5127 ± 0.115576 (W 0.0180951) | 17.821 ± 0.0957925 (W 0.0430639) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.110988 ± 0.000961578 (W 0.00175001) | 0.110148 ± 0.000661016 (W 0.00165444) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.111287 ± 0.000223789 (W 0.00131721) | 0.110485 ± 0.000280038 (W 0.000310482) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.788353 ± 0.00209249 (W 0.00256692) | 0.794217 ± 0.0174697 (W 0.00283843) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.789058 ± 0.00299407 (W 0.0026738) | 0.79277 ± 0.000974031 (W 0.003769) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.07574 ± 0.0804798 (W 0.00630318) | 7.09244 ± 0.0339036 (W 0.00464277) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.60229 ± 0.0321412 (W 0.0148325) | 7.07466 ± 0.0297749 (W 0.0103224) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.059329 ± 0.000361465 (W 0.00019009) | 0.165365 ± 0.000666699 (W 0.00210485) | - | 🔵 colmat 2.79× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.096638 ± 0.000377889 (W 0.000311816) | 0.081058 ± 0.000696156 (W 0.00107693) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.284253 ± 0.00096707 (W 0.0015654) | 1.61302 ± 0.003967 (W 0.00614612) | - | 🔵 colmat 5.67× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.527328 ± 0.0087543 (W 0.00273047) | 0.623755 ± 0.0104973 (W 0.00240286) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.85463 ± 0.00530893 (W 0.00420031) | 13.4791 ± 0.0291881 (W 0.0262097) | - | 🔵 colmat 7.27× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.80302 ± 0.010342 (W 0.0032988) | 5.34257 ± 0.00511671 (W 0.0114216) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.248204 ± 0.00340527 (W 0.00133761) | 0.432596 ± 0.000912112 (W 0.00148741) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.277561 ± 0.00112946 (W 0.00210792) | 0.338211 ± 0.00210921 (W 0.0025029) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.37004 ± 0.00637231 (W 0.00657888) | 2.2628 ± 0.0196409 (W 0.00636218) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.55033 ± 0.00518127 (W 0.0052582) | 1.63676 ± 0.0200936 (W 0.00456561) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 9.75667 ± 0.0494258 (W 0.00642605) | 17.2896 ± 0.0305821 (W 0.0119343) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.91562 ± 0.121351 (W 0.00994144) | 10.6218 ± 0.0109317 (W 0.0076847) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.143706 ± 0.00126305 (W 0.00100036) | 0.33164 ± 0.002268 (W 0.00205523) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.774429 ± 0.00150775 (W 0.00226468) | 1.70992 ± 0.00947446 (W 0.00442239) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 4.08457 ± 0.0752412 (W 0.00461223) | 12.3533 ± 0.0167673 (W 0.0258253) | - | 🔵 colmat 3.02× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.053298 ± 0.000220402 (W 0.000195308) | 0.117337 ± 0.00090967 (W 0.00283716) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.071198 ± 0.00027 (W 0.00013274) | 0.105638 ± 0.000260858 (W 0.00202067) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.257202 ± 0.000418928 (W 0.00192795) | 1.17269 ± 0.00775164 (W 0.00529099) | - | 🔵 colmat 4.56× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.35163 ± 0.00262634 (W 0.00253329) | 0.439296 ± 0.000718501 (W 0.00369056) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.73202 ± 0.0026987 (W 0.00235306) | 9.19705 ± 0.331214 (W 0.0235263) | - | 🔵 colmat 5.31× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1 [api-inclusive]` | 1.96725 ± 0.00707757 (W 0.0107636) | 4.39477 ± 0.00927259 (W 0.00777913) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.7542 ± 0.00543363 (W 0.00155233) | 2.31278 ± 0.00699938 (W 0.00713767) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.75189 ± 0.000173523 (W 0.00197128) | 2.32199 ± 0.0042209 (W 0.00301812) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.16524 ± 0.030125 (W 0.004982) | 11.0366 ± 0.0350415 (W 0.0107645) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.19089 ± 0.00241123 (W 0.0134001) | 11.0874 ± 0.0459756 (W 0.0132723) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 48.0353 ± 0.221879 (W 0.0757421) | 64.0002 ± 0.00429176 (W 0.0520597) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.303 ± 0.166323 (W 0.0452949) | 63.5274 ± 0.0546075 (W 0.0774977) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.611961 ± 0.00184834 (W 0.00136013) | 1.75985 ± 0.00467529 (W 0.00229549) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.613833 ± 0.00126551 (W 0.00182967) | 1.76564 ± 0.00170995 (W 0.00396941) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.42224 ± 0.00853344 (W 0.00421516) | 8.13586 ± 0.0134245 (W 0.00598275) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.4435 ± 0.00494187 (W 0.00501247) | 8.13405 ± 0.00558843 (W 0.00601769) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 25.7746 ± 0.0559513 (W 0.0513548) | 48.3358 ± 0.149953 (W 0.0876342) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 25.7799 ± 0.0921291 (W 0.0796178) | 46.6714 ± 0.204598 (W 0.0244932) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00292 ± 2.08167e-05 (W 4.27785e-05) | 0.00284 ± 6.08276e-06 (W 5.1672e-05) | 0.00181 ± 0 (W 2.38747e-05) | · | 🔴 colmat 1.61× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.00784 ± 3e-05 (W 6.03755e-05) | 0.007509 ± 5.53203e-05 (W 7.29534e-05) | 0.00519 ± 5.7735e-06 (W 2.3435e-05) | · | 🔴 colmat 1.51× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.028269 ± 0.000295353 (W 0.000276399) | 0.02796 ± 0.000434293 (W 0.000578449) | 0.021269 ± 0.000184527 (W 0.000330957) | · | 🔴 colmat 1.33× slower than `lapack-mkl` |  |

## linalg-mid, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.029039 ± 0.000136504 (W 0.000128325) | 0.137234 ± 0.0287206 (W 0.028384) | - | 🔵 colmat 4.73× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.034749 ± 0.000155027 (W 0.000436268) | 0.050748 ± 0.000624164 (W 0.00123917) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.106596 ± 0.000723174 (W 0.000965496) | 0.382325 ± 0.0254474 (W 0.0355698) | - | 🔵 colmat 3.59× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.127425 ± 0.000981104 (W 0.000297557) | 0.235351 ± 0.0049995 (W 0.0080005) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.596216 ± 0.00468409 (W 0.00269164) | 1.31324 ± 0.0415322 (W 0.0648964) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.531746 ± 0.00240118 (W 0.00229289) | 1.4767 ± 0.0268273 (W 0.0228325) | - | 🔵 colmat 2.78× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.565891 ± 0.00183694 (W 0.00140968) | 1.53068 ± 0.00354291 (W 0.0014554) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.570577 ± 0.0129407 (W 0.0026363) | 1.53036 ± 0.00262416 (W 0.00205165) | - | 🔵 colmat 2.68× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.13818 ± 0.204704 (W 0.0052972) | 4.46771 ± 0.00447399 (W 0.00611765) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1 [api-inclusive]` | 2.14668 ± 0.00641566 (W 0.00353597) | 4.50673 ± 0.01496 (W 0.0107249) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 9.34174 ± 0.0882046 (W 0.0182536) | 16.5323 ± 0.0431269 (W 0.0289582) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 9.33403 ± 0.00936197 (W 0.0131809) | 16.0911 ± 0.368822 (W 0.0428434) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.168875 ± 0.000306487 (W 0.001266) | 0.202932 ± 0.00127838 (W 0.00171994) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.176954 ± 0.00218091 (W 0.00103264) | 0.205942 ± 0.000558302 (W 0.0029039) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.668103 ± 0.0044057 (W 0.00292788) | 2.03467 ± 0.00529249 (W 0.0063022) | - | 🔵 colmat 3.05× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.871177 ± 0.00207209 (W 0.00388772) | 1.8304 ± 0.00875354 (W 0.00798294) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 3.13378 ± 0.00442999 (W 0.00770615) | 12.1102 ± 0.0392639 (W 0.0157749) | - | 🔵 colmat 3.86× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 4.18389 ± 0.0173075 (W 0.00825121) | 11.8111 ± 0.026557 (W 0.020278) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.044029 ± 0.000221202 (W 0.000455906) | 0.041519 ± 0.000562939 (W 0.000278606) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.044058 ± 0.000306219 (W 0.000401977) | 0.041388 ± 0.000455738 (W 0.000216633) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.23897 ± 0.0015606 (W 0.00245319) | 0.241374 ± 0.00409406 (W 0.00335037) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.238002 ± 0.00216115 (W 0.00327387) | 0.240422 ± 0.00440113 (W 0.00350867) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.89413 ± 0.042326 (W 0.0059243) | 1.92153 ± 0.0255296 (W 0.00529102) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.88179 ± 0.0278626 (W 0.00722278) | 1.91545 ± 0.0446566 (W 0.00451813) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.074628 ± 0.000867618 (W 0.000812761) | 0.181735 ± 0.000505751 (W 0.00320311) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.067337 ± 0.000169827 (W 0.000360529) | 0.096027 ± 0.00182487 (W 0.00253376) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.212004 ± 0.000588772 (W 0.00208274) | 1.47323 ± 0.0211677 (W 0.0391341) | - | 🔵 colmat 6.95× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.424478 ± 0.00439529 (W 0.0031308) | 0.502046 ± 0.00551557 (W 0.00579781) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.913176 ± 0.00249117 (W 0.00375525) | 9.9512 ± 0.23616 (W 0.349066) | - | 🔵 colmat 10.90× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.95753 ± 0.0375359 (W 0.00414182) | 3.85561 ± 0.046159 (W 0.0520141) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.197792 ± 0.000748567 (W 0.00112906) | 0.557528 ± 0.000676557 (W 0.00322107) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.240801 ± 0.000446715 (W 0.00120083) | 0.420575 ± 0.00112505 (W 0.00179464) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.595544 ± 0.00241959 (W 0.00381748) | 2.21147 ± 0.00623348 (W 0.0143483) | - | 🔵 colmat 3.71× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.934524 ± 0.000939223 (W 0.00339966) | 1.40979 ± 0.011045 (W 0.00573683) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.33289 ± 0.00830161 (W 0.00868013) | 14.0978 ± 0.0395088 (W 0.0341584) | - | 🔵 colmat 4.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.30074 ± 0.0413461 (W 0.00408615) | 6.59487 ± 0.0474721 (W 0.0145335) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.118797 ± 0.000817617 (W 0.000433982) | 0.478511 ± 0.00344065 (W 0.00366483) | - | 🔵 colmat 4.03× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.404319 ± 0.00215804 (W 0.00150972) | 2.02507 ± 0.00622986 (W 0.0140157) | - | 🔵 colmat 5.01× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.73669 ± 0.00616426 (W 0.00225867) | 12.5012 ± 0.125527 (W 0.0340099) | - | 🔵 colmat 7.20× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.051349 ± 0.000395032 (W 0.000373164) | 0.141236 ± 0.000685124 (W 0.00148595) | - | 🔵 colmat 2.75× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.068837 ± 0.000508082 (W 0.000199272) | 0.094577 ± 0.0010862 (W 0.00188968) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.211122 ± 0.000169316 (W 0.00251879) | 1.11054 ± 0.0229034 (W 0.0180446) | - | 🔵 colmat 5.26× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.433878 ± 0.00586221 (W 0.0034908) | 0.457764 ± 0.00117781 (W 0.00383598) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.857141 ± 0.0115882 (W 0.00125053) | 6.91528 ± 0.242472 (W 0.196561) | - | 🔵 colmat 8.07× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1 [api-inclusive]` | 1.97457 ± 0.00918358 (W 0.00710365) | 3.67119 ± 0.0244577 (W 0.0648004) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.27701 ± 0.0023752 (W 0.00337304) | 1.80708 ± 0.00246812 (W 0.00528577) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.30436 ± 0.00834594 (W 0.0067857) | 1.81907 ± 0.01077 (W 0.0102528) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.95481 ± 0.0127868 (W 0.00494915) | 8.39884 ± 0.00826748 (W 0.00843283) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 4.99171 ± 0.00986349 (W 0.011533) | 8.38027 ± 0.00958915 (W 0.0111137) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 22.6654 ± 0.733042 (W 0.052485) | 36.8423 ± 0.125379 (W 0.0343346) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 22.8121 ± 0.0452865 (W 0.0297502) | 37.048 ± 0.0634764 (W 0.0505101) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.637675 ± 0.00081363 (W 0.00340826) | 1.42981 ± 0.00251375 (W 0.00311506) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.633475 ± 0.00248448 (W 0.00171432) | 1.43566 ± 0.00339671 (W 0.00263585) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.68905 ± 0.00420257 (W 0.00349242) | 6.85102 ± 0.00557255 (W 0.0103903) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.6837 ± 0.00410125 (W 0.00552206) | 6.85287 ± 0.0246372 (W 0.0096084) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 13.2281 ± 0.0359978 (W 0.0155692) | 30.3942 ± 0.0736268 (W 0.0293012) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 13.2296 ± 0.0560139 (W 0.00686903) | 30.3191 ± 0.0291859 (W 0.0225128) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00353 ± 2.51661e-05 (W 3.39588e-05) | 0.00285 ± 4.16333e-05 (W 4.46453e-05) | 0.00182 ± 1e-05 (W 2.68328e-05) | 🔴 colmat 1.24× slower than `faer` | 🔴 colmat 1.94× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.008189 ± 0.000113227 (W 6.6214e-05) | 0.00749 ± 4.72582e-05 (W 5.9365e-05) | 0.00519 ± 1.1547e-05 (W 1.48728e-05) | 🔴 colmat 1.09× slower than `faer` | 🔴 colmat 1.58× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.019769 ± 7.86787e-05 (W 0.000124154) | 0.020019 ± 0.000170871 (W 0.000113641) | 0.021389 ± 0.000433771 (W 0.000264823) | · | 🔵 colmat 1.08× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-mid, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.02851 ± 0.000378572 (W 0.000133556) | 0.088097 ± 0.000845715 (W 0.0220336) | - | 🔵 colmat 3.09× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.034968 ± 0.00034 (W 0.000417552) | 0.045778 ± 3.04467e-05 (W 0.000886322) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.101507 ± 0.00109092 (W 0.00134725) | 0.373366 ± 0.0180526 (W 0.0193813) | - | 🔵 colmat 3.68× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.126096 ± 0.00126259 (W 0.000896799) | 0.220643 ± 0.0156612 (W 0.0165304) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.510842 ± 0.00376833 (W 0.00210287) | 1.23375 ± 0.00471688 (W 0.0362231) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.506626 ± 0.0204775 (W 0.0029911) | 1.42334 ± 0.0231145 (W 0.0172273) | - | 🔵 colmat 2.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 0.437117 ± 0.00172198 (W 0.00151818) | 1.54036 ± 0.0023837 (W 0.0031698) | - | 🔵 colmat 3.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.442538 ± 0.00169652 (W 0.00181039) | 1.53847 ± 0.00204563 (W 0.00212984) | - | 🔵 colmat 3.48× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.58858 ± 0.00228977 (W 0.00297079) | 4.53321 ± 1.66195 (W 0.0215727) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1 [api-inclusive]` | 1.58719 ± 0.00135446 (W 0.0042843) | 4.5514 ± 1.64918 (W 0.0154129) | - | 🔵 colmat 2.87× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n512/b1` | 6.28637 ± 0.00358425 (W 0.00887756) | 15.9671 ± 3.07799 (W 0.801453) | - | 🔵 colmat 2.54× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n512/b1 [api-inclusive]` | 6.27873 ± 0.019237 (W 0.0117449) | 15.4482 ± 2.53373 (W 0.270631) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1` | 0.165095 ± 0.00108616 (W 0.00186086) | 0.204183 ± 0.000380942 (W 0.00195183) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.179613 ± 0.00150808 (W 0.00197028) | 0.207824 ± 0.00142824 (W 0.00188137) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.600724 ± 0.00227363 (W 0.00456032) | 2.03454 ± 0.0096738 (W 0.00585098) | - | 🔵 colmat 3.39× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.824819 ± 0.00748877 (W 0.00214091) | 1.84854 ± 0.0251636 (W 0.00418052) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 2.2151 ± 0.0111559 (W 0.00749079) | 11.4207 ± 0.0461765 (W 0.0239183) | - | 🔵 colmat 5.16× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 3.36223 ± 0.0182934 (W 0.00754225) | 11.254 ± 0.04158 (W 0.0109549) | - | 🔵 colmat 3.35× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.035589 ± 0.000535008 (W 0.000518459) | 0.056259 ± 0.0010145 (W 0.00125658) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.035519 ± 0.0004531 (W 0.00164556) | 0.055888 ± 0.000465833 (W 0.000839061) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.143695 ± 0.00239978 (W 0.00201447) | 0.217533 ± 0.00313865 (W 0.00977603) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.145085 ± 0.00220759 (W 0.00157325) | 0.229374 ± 0.00647879 (W 0.00624368) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.04545 ± 0.0032987 (W 0.00451836) | 1.11155 ± 0.13449 (W 0.0778818) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.03972 ± 0.00950926 (W 0.00452958) | 1.29099 ± 0.0306162 (W 0.154101) | - | 🔵 colmat 1.24× faster than `faer` (marginal: spread 0.61 of the difference) | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.078898 ± 0.00141612 (W 0.000803366) | 0.175454 ± 0.00136856 (W 0.00251355) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.067528 ± 0.000137078 (W 0.000473482) | 0.093587 ± 0.00160615 (W 0.00104795) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.180715 ± 0.000360311 (W 0.0012179) | 1.49001 ± 0.0107799 (W 0.0169092) | - | 🔵 colmat 8.25× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.417364 ± 0.00325542 (W 0.00151588) | 0.492987 ± 0.0115321 (W 0.0130535) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.676588 ± 0.0016936 (W 0.00327009) | 8.56782 ± 0.0596294 (W 0.175862) | - | 🔵 colmat 12.66× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.82744 ± 0.0477592 (W 0.0299652) | 3.87608 ± 0.0358907 (W 0.0539486) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.193064 ± 0.000746915 (W 0.00190611) | 0.639152 ± 0.00440023 (W 0.00541488) | - | 🔵 colmat 3.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.236841 ± 0.000396878 (W 0.00128069) | 0.508931 ± 0.00430603 (W 0.00301645) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.517056 ± 0.00375411 (W 0.00303886) | 2.44955 ± 0.0132072 (W 0.0136409) | - | 🔵 colmat 4.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.865652 ± 0.0076698 (W 0.0032833) | 1.40957 ± 0.0042981 (W 0.00573916) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.1682 ± 0.00562519 (W 0.00489381) | 13.8598 ± 0.112864 (W 0.0429514) | - | 🔵 colmat 6.39× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.35369 ± 0.00779208 (W 0.0190592) | 6.31126 ± 0.688656 (W 1.95319) | - | 🔵 colmat 1.88× faster than `faer` (marginal: spread 0.66 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.114696 ± 0.000669679 (W 0.00114542) | 0.556335 ± 0.00114376 (W 0.00306636) | - | 🔵 colmat 4.85× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.390338 ± 0.00106152 (W 0.002218) | 2.23799 ± 0.0133617 (W 0.0110793) | - | 🔵 colmat 5.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.12985 ± 0.00646263 (W 0.00320159) | 12.5953 ± 0.431182 (W 0.0413613) | - | 🔵 colmat 11.15× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.053338 ± 0.000532116 (W 0.000517371) | 0.137775 ± 0.00174607 (W 0.00171512) | - | 🔵 colmat 2.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.068308 ± 4.78226e-05 (W 0.00036716) | 0.090857 ± 0.000522044 (W 0.000791022) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.178253 ± 0.00108035 (W 0.00224393) | 1.1004 ± 0.018528 (W 0.015064) | - | 🔵 colmat 6.17× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.424426 ± 0.00687889 (W 0.00283313) | 0.444724 ± 0.00462878 (W 0.0154723) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.654587 ± 0.000761497 (W 0.00283158) | 6.52621 ± 0.0291758 (W 0.0859198) | - | 🔵 colmat 9.97× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.84452 ± 0.00452115 (W 0.00374455) | 3.60695 ± 0.0498863 (W 0.0408542) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.28403 ± 0.00272489 (W 0.00265546) | 1.83083 ± 0.00204162 (W 0.00615114) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.29181 ± 0.00322879 (W 0.0030017) | 1.83737 ± 0.00080229 (W 0.00290564) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.62802 ± 0.0174715 (W 0.00895799) | 8.53576 ± 0.0172162 (W 0.018367) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 4.65617 ± 0.0125794 (W 0.00616989) | 8.54408 ± 0.0102518 (W 0.0173989) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 19.5695 ± 0.82185 (W 0.0188033) | 35.7156 ± 0.816245 (W 0.0466667) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 19.7041 ± 0.679241 (W 0.0153232) | 35.7588 ± 0.150143 (W 0.333005) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 0.668281 ± 0.00158814 (W 0.00214678) | 1.43334 ± 0.00167129 (W 0.0045257) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.669272 ± 0.00388936 (W 0.00155663) | 1.4386 ± 0.00143775 (W 0.00266558) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.61668 ± 0.00981439 (W 0.00229377) | 6.91999 ± 0.0350618 (W 0.0083267) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.62383 ± 0.00427619 (W 0.00478392) | 6.91925 ± 0.0201925 (W 0.0095528) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.9297 ± 0.012656 (W 0.00810297) | 29.153 ± 0.0350626 (W 0.0323098) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.9417 ± 0.00744281 (W 0.0173726) | 29.201 ± 0.0858717 (W 0.0393442) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00357 ± 9.60902e-05 (W 3.42053e-05) | 0.002819 ± 4.0501e-05 (W 4.84768e-05) | 0.0018 ± 0 (W 2.16795e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.98× slower than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.008169 ± 0.000127163 (W 8.51833e-05) | 0.00753 ± 7.5831e-05 (W 5.47841e-05) | 0.00519 ± 2.58908e-05 (W 3.21279e-05) | 🔴 colmat 1.08× slower than `faer` | 🔴 colmat 1.57× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.01883 ± 0.000157068 (W 0.000105268) | 0.01905 ± 0.000284113 (W 0.000175884) | 0.021199 ± 0.000179723 (W 0.000214267) | · | 🔵 colmat 1.13× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-mid, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0258 ± 0.000120611 (W 6.78395e-05) | 0.033149 ± 0.000147422 (W 8.98176e-05) | 0.055649 ± 0.000502614 (W 9.26569e-05) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.030109 ± 0.000164621 (W 0.00010403) | 0.037339 ± 6.02522e-05 (W 5.06952e-05) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.128296 ± 0.000153189 (W 0.00130051) | 0.151056 ± 0.00111974 (W 0.001162) | 0.283662 ± 0.00165639 (W 0.00267845) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 2.21× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.147177 ± 0.000249548 (W 7.87102e-05) | 0.163365 ± 0.00280682 (W 0.00142922) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.841177 ± 0.000675926 (W 0.00146766) | 0.963497 ± 0.00697511 (W 0.00266359) | 2.32871 ± 0.00546408 (W 0.0173928) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 2.77× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.936487 ± 0.00203028 (W 0.000703349) | 0.96638 ± 0.0010009 (W 0.00116722) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.3322 ± 0.000944345 (W 0.00184652) | 1.5678 ± 0.000948636 (W 0.00221099) | 1.1999 ± 0.00125429 (W 0.0039985) | 🔵 colmat 1.18× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.33531 ± 0.00425365 (W 0.00213751) | 1.565 ± 0.00374446 (W 0.00212729) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.84207 ± 0.0241838 (W 0.00550546) | 6.1927 ± 0.0235513 (W 0.00268004) | 6.45495 ± 0.016699 (W 0.00650652) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.84297 ± 0.0306161 (W 0.0056139) | 6.20433 ± 0.0272043 (W 0.00248279) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 28.0466 ± 0.230793 (W 0.0713038) | 28.1297 ± 0.104536 (W 0.0114697) | 40.3469 ± 0.345689 (W 0.353248) | · | 🔵 colmat 1.44× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 28.4429 ± 0.132296 (W 0.0174015) | 28.1695 ± 0.0574922 (W 0.0185194) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.199204 ± 0.0198081 (W 0.00126996) | 0.200445 ± 0.000856374 (W 0.00168771) | 1.03221 ± 0.00218333 (W 0.0015617) | · | 🔵 colmat 5.18× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.190606 ± 0.0016345 (W 0.000870994) | 0.191124 ± 0.000784921 (W 0.00108121) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.32282 ± 0.155375 (W 0.00324524) | 1.85028 ± 0.0896181 (W 0.0051565) | 8.75839 ± 0.0125488 (W 0.0174487) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 6.62× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.37493 ± 0.00435202 (W 0.0023348) | 1.8869 ± 0.00100182 (W 0.00450674) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.7548 ± 0.0222777 (W 0.00847396) | 15.0696 ± 0.00896102 (W 0.0184578) | 70.2451 ± 0.0449419 (W 0.0708767) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 6.53× faster than `lapack-mkl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 10.0869 ± 0.00282805 (W 0.0249269) | 14.7483 ± 0.0290442 (W 0.0186602) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.110755 ± 0.000655501 (W 0.0011823) | 0.109417 ± 0.00013602 (W 0.000301523) | 0.428069 ± 0.000781557 (W 0.00417653) | · | 🔵 colmat 3.87× faster than `lapack-mkl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.110585 ± 0.00053505 (W 0.000222731) | 0.109875 ± 0.00104188 (W 0.00133514) | 0.355763 ± 0.00290006 (W 0.00197059) | · | 🔵 colmat 3.22× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.775791 ± 0.0183545 (W 0.00298998) | 0.80858 ± 0.0134045 (W 0.00257856) | 2.64709 ± 0.00939209 (W 0.00783957) | · | 🔵 colmat 3.41× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.781998 ± 0.0166939 (W 0.00224038) | 0.816177 ± 0.00461987 (W 0.00275992) | 2.19014 ± 0.00717935 (W 0.00313974) | · | 🔵 colmat 2.80× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 6.74943 ± 0.0139257 (W 0.00615795) | 6.81069 ± 0.16543 (W 0.00912297) | 20.4477 ± 0.070791 (W 0.0212803) | · | 🔵 colmat 3.03× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 6.76593 ± 0.00653444 (W 0.00873686) | 6.79696 ± 0.161882 (W 0.00656601) | 15.5072 ± 0.0690029 (W 0.0200282) | · | 🔵 colmat 2.29× faster than `lapack-mkl` |  |
| `lu/n128/b1` | 0.053868 ± 0.000462198 (W 0.000284166) | 0.059018 ± 0.000872709 (W 0.000396573) | 0.092237 ± 0.000301442 (W 0.000867469) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.71× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.084098 ± 0.000531235 (W 0.000350262) | 0.065888 ± 0.00046394 (W 0.000225156) | - | 🔴 colmat 1.28× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.276754 ± 0.00070062 (W 0.0025619) | 0.3599 ± 0.00224049 (W 0.00238192) | 0.505334 ± 0.00105692 (W 0.00216627) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 1.83× faster than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.461348 ± 0.0027858 (W 0.00299063) | 0.527244 ± 0.00275792 (W 0.00241778) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.65677 ± 0.00247713 (W 0.00199312) | 2.54132 ± 0.0201619 (W 0.00496644) | 3.33062 ± 0.00726095 (W 0.00424708) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 2.01× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 2.50008 ± 0.00380526 (W 0.00292694) | 3.30204 ± 0.0282307 (W 0.0053228) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.210764 ± 0.00234131 (W 0.0012687) | 0.271233 ± 0.00121252 (W 0.00122073) | 0.578374 ± 0.00283918 (W 0.00313096) | 🔵 colmat 1.29× faster than `faer` | 🔵 colmat 2.74× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.261023 ± 0.0015719 (W 0.00142863) | 0.321253 ± 0.000650348 (W 0.00241072) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.19234 ± 0.00684159 (W 0.00233679) | 1.24242 ± 0.00747633 (W 0.00210444) | 3.43878 ± 0.00590433 (W 0.00634179) | · | 🔵 colmat 2.88× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.47512 ± 0.00401918 (W 0.00325362) | 1.5306 ± 0.00438154 (W 0.00460671) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 7.65027 ± 0.171781 (W 0.00356449) | 7.79783 ± 0.0283329 (W 0.00465221) | 24.9071 ± 0.0437384 (W 0.0232444) | · | 🔵 colmat 3.26× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 8.64798 ± 0.0332747 (W 0.00805234) | 8.63266 ± 0.00846772 (W 0.0139797) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.105047 ± 0.000466512 (W 0.000181025) | 0.168745 ± 0.000857146 (W 0.00113885) | 0.212965 ± 0.000284166 (W 0.0014872) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 2.03× faster than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 0.63149 ± 0.00213794 (W 0.00181782) | 0.693913 ± 0.00661114 (W 0.00202323) | 1.25367 ± 0.00196494 (W 0.00364815) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.99× faster than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 3.48414 ± 0.0634258 (W 0.00241127) | 3.63971 ± 0.0205221 (W 0.00249743) | 9.47767 ± 0.0583604 (W 0.0112915) | · | 🔵 colmat 2.72× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.052948 ± 0.000237691 (W 0.00012843) | 0.055639 ± 0.000242693 (W 9.65723e-05) | 0.090768 ± 0.000300045 (W 0.000924479) | · | 🔵 colmat 1.71× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.057238 ± 0.000140998 (W 0.000267017) | 0.090358 ± 0.000254127 (W 0.000170303) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.264004 ± 0.00317558 (W 0.00184044) | 0.344401 ± 0.00108128 (W 0.00249944) | 0.507034 ± 0.00172231 (W 0.0017006) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 1.92× faster than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 0.292962 ± 0.00110108 (W 0.00226172) | 0.344812 ± 0.000583971 (W 0.00134559) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.61476 ± 0.00193147 (W 0.00353663) | 2.38653 ± 0.0015288 (W 0.00323951) | 3.39681 ± 0.00527572 (W 0.00677269) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 2.10× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 1.70772 ± 0.00188465 (W 0.00252161) | 2.36268 ± 0.00542768 (W 0.00417804) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.73895 ± 0.00646246 (W 0.00208731) | 2.32517 ± 0.0137046 (W 0.00211113) | 2.14583 ± 0.0169689 (W 0.00390499) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.73558 ± 0.00362372 (W 0.00233979) | 2.31466 ± 0.00288458 (W 0.00199973) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.12061 ± 0.00972895 (W 0.00602763) | 10.9608 ± 0.0238532 (W 0.0112794) | 11.6588 ± 0.0156142 (W 0.0124194) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 8.13487 ± 0.00669695 (W 0.00424424) | 10.9978 ± 0.013108 (W 0.00965526) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 45.9389 ± 0.150418 (W 0.0755452) | 62.1145 ± 0.150064 (W 0.0736553) | 85.0013 ± 0.270017 (W 0.150205) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.85× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 46.0145 ± 0.0643481 (W 0.0708711) | 61.408 ± 0.196414 (W 0.0484256) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.595163 ± 0.00127239 (W 0.00172234) | 1.71617 ± 0.0012134 (W 0.00293905) | 0.733449 ± 0.00101988 (W 0.00223062) | 🔵 colmat 2.88× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.595682 ± 0.000525718 (W 0.00160423) | 1.71938 ± 0.00406902 (W 0.0019849) | - | 🔵 colmat 2.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.33439 ± 0.0028787 (W 0.00533834) | 8.03071 ± 0.00399821 (W 0.00378327) | 4.37282 ± 0.00567192 (W 0.00623413) | 🔵 colmat 2.41× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.3737 ± 0.0267923 (W 0.00632003) | 8.04478 ± 0.0062946 (W 0.00671106) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 22.4331 ± 0.0972277 (W 0.0119748) | 45.0819 ± 0.106462 (W 0.0284769) | 35.1514 ± 0.0502583 (W 0.220268) | 🔵 colmat 2.01× faster than `faer` | 🔵 colmat 1.57× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 22.5311 ± 0.154498 (W 0.0113591) | 44.863 ± 0.132966 (W 0.0596467) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00332 ± 1e-05 (W 1.48324e-05) | 0.0032 ± 1e-05 (W 3.74166e-05) | 0.0017 ± 5.7735e-07 (W 2.50998e-05) | · | 🔴 colmat 1.95× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.008429 ± 2.82902e-05 (W 6.80757e-05) | 0.008149 ± 2.47049e-05 (W 8.24694e-05) | 0.00497 ± 0 (W 1.94731e-05) | · | 🔴 colmat 1.70× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.031409 ± 0.000193132 (W 0.00111932) | 0.030739 ± 0.000868581 (W 0.000327109) | 0.02114 ± 0.00101054 (W 0.000139253) | · | 🔴 colmat 1.49× slower than `lapack-mkl` | `!` between-start range over 5% |

## linalg-mid, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.025648 ± 0.000972461 (W 0.000169524) | 0.039409 ± 0.00213951 (W 0.00190287) | 0.055288 ± 0.00025697 (W 0.000193205) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.030559 ± 5.74195e-05 (W 0.000161936) | 0.047258 ± 0.000970428 (W 0.000908069) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.084387 ± 0.00107218 (W 0.000554869) | 0.167465 ± 0.00232941 (W 0.00947959) | 0.167424 ± 0.00398291 (W 0.0102649) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 1.98× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.105828 ± 0.000465261 (W 0.00046547) | 0.196013 ± 0.00560115 (W 0.00809436) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.3652 ± 0.00129119 (W 0.00228211) | 0.561485 ± 0.0132627 (W 0.0280908) | 1.01126 ± 0.0113825 (W 0.0122765) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 2.77× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.446228 ± 0.00111707 (W 0.00181246) | 0.652228 ± 0.0110907 (W 0.0227981) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.560221 ± 0.000934223 (W 0.00197583) | 1.52913 ± 8.35125e-05 (W 0.00122479) | 0.696491 ± 0.00188033 (W 0.00192568) | 🔵 colmat 2.73× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.563694 ± 0.00182125 (W 0.00177651) | 1.5229 ± 0.00121451 (W 0.00140982) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.10834 ± 0.00548634 (W 0.00446942) | 4.43936 ± 0.0105078 (W 0.00535949) | 3.08557 ± 0.00911419 (W 0.00669713) | 🔵 colmat 2.11× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 2.10567 ± 0.00425223 (W 0.00455678) | 4.45521 ± 0.0318005 (W 0.0125639) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 9.16377 ± 0.0546211 (W 0.0172696) | 15.1969 ± 0.0399197 (W 0.0231997) | 13.65 ± 0.267703 (W 0.0208532) | 🔵 colmat 1.66× faster than `faer` | 🔵 colmat 1.49× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 9.19144 ± 0.00934421 (W 0.0148169) | 15.1763 ± 0.0608758 (W 0.0223319) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.159014 ± 0.00127288 (W 0.000850261) | 0.197784 ± 0.000271111 (W 0.00125288) | 1.03284 ± 0.00164507 (W 0.00158303) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 6.50× faster than `lapack-mkl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.163284 ± 0.00100758 (W 0.00115119) | 0.190935 ± 0.000976002 (W 0.00190556) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.657163 ± 0.00469894 (W 0.00277697) | 1.84306 ± 0.00353508 (W 0.00388444) | 8.74871 ± 0.0210292 (W 0.0108905) | 🔵 colmat 2.80× faster than `faer` | 🔵 colmat 13.31× faster than `lapack-mkl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.815562 ± 0.00856496 (W 0.00261948) | 1.73239 ± 0.00449676 (W 0.00688981) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 3.28079 ± 0.0418428 (W 0.0112517) | 10.3073 ± 0.0313422 (W 0.0175804) | 70.2637 ± 0.0634907 (W 0.0572222) | 🔵 colmat 3.14× faster than `faer` | 🔵 colmat 21.42× faster than `lapack-mkl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 3.86887 ± 0.0191391 (W 0.0153744) | 9.7915 ± 0.0497195 (W 0.0185276) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.043959 ± 0.000156205 (W 0.000457709) | 0.042378 ± 0.000311079 (W 0.000416329) | 0.276733 ± 0.00370541 (W 0.00715081) | · | 🔵 colmat 6.30× faster than `lapack-mkl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.043399 ± 0.000342685 (W 0.00109328) | 0.042049 ± 0.000257799 (W 0.000279016) | 0.092288 ± 0.000625405 (W 0.000433115) | · | 🔵 colmat 2.13× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.23975 ± 0.00227346 (W 0.00113565) | 0.241673 ± 0.00296217 (W 0.00148827) | 1.31709 ± 0.0100105 (W 0.00301645) | · | 🔵 colmat 5.49× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.242324 ± 0.00123763 (W 0.00239821) | 0.240773 ± 0.000974463 (W 0.00147486) | 0.624858 ± 0.00216913 (W 0.00315655) | · | 🔵 colmat 2.58× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.87942 ± 0.0303753 (W 0.0069415) | 1.90475 ± 0.0063135 (W 0.00588618) | 10.9258 ± 0.0397165 (W 0.0229275) | · | 🔵 colmat 5.81× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.84274 ± 0.019131 (W 0.00391596) | 1.89197 ± 0.00993863 (W 0.00227921) | 3.85762 ± 0.0508498 (W 0.0111667) | · | 🔵 colmat 2.09× faster than `lapack-mkl` |  |
| `lu/n128/b1` | 0.046158 ± 0.000135746 (W 0.000358611) | 0.073708 ± 0.00159097 (W 0.00206974) | 0.092597 ± 8.1e-05 (W 0.00123482) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 2.01× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.053748 ± 0.000328233 (W 0.000340983) | 0.083067 ± 0.00187144 (W 0.00276464) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.181772 ± 0.000101254 (W 0.00162858) | 0.379407 ± 0.0100082 (W 0.0127447) | 0.196923 ± 0.000828232 (W 0.00284962) | 🔵 colmat 2.09× faster than `faer` | 🔵 colmat 1.08× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.36861 ± 0.000448159 (W 0.0025631) | 0.403966 ± 0.00397998 (W 0.0125804) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.796468 ± 0.00196298 (W 0.00202372) | 1.82906 ± 0.0221012 (W 0.0501998) | 0.997935 ± 0.0014599 (W 0.00431239) | 🔵 colmat 2.30× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 1.65429 ± 0.00479739 (W 0.00385515) | 1.89296 ± 0.00661583 (W 0.0565389) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.168074 ± 0.000687026 (W 0.00112108) | 0.343899 ± 0.00151805 (W 0.00199375) | 0.212292 ± 0.00117913 (W 0.00196913) | 🔵 colmat 2.05× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.228481 ± 0.000977031 (W 0.00085087) | 0.407664 ± 0.00191817 (W 0.00179) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.551396 ± 0.00182862 (W 0.00242377) | 1.00484 ± 0.00462906 (W 0.00352872) | 1.00101 ± 0.00187464 (W 0.00378351) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 1.82× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 0.865925 ± 0.00177228 (W 0.0028233) | 1.31502 ± 0.00573029 (W 0.00617308) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.7011 ± 0.00349721 (W 0.00327805) | 3.64436 ± 0.00232279 (W 0.00891359) | 6.07421 ± 0.0490569 (W 0.00829325) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 2.25× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 3.96447 ± 0.0114823 (W 0.0099851) | 4.61647 ± 0.00308962 (W 0.0122456) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.104276 ± 0.000787329 (W 0.00012665) | 0.277221 ± 0.00252435 (W 0.00107034) | 0.119855 ± 0.000160689 (W 0.00178285) | 🔵 colmat 2.66× faster than `faer` | 🔵 colmat 1.15× faster than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 0.364305 ± 0.00141922 (W 0.00150488) | 0.819168 ± 0.00150867 (W 0.00389625) | 0.381367 ± 0.00181089 (W 0.00299889) | 🔵 colmat 2.25× faster than `faer` | · |  |
| `qr_factor/n512/b1` | 1.55607 ± 0.000991015 (W 0.00119269) | 2.51544 ± 0.0094508 (W 0.00976143) | 2.21788 ± 0.00308454 (W 0.00355825) | 🔵 colmat 1.62× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.045198 ± 0.000298027 (W 0.000169853) | 0.068377 ± 0.000791269 (W 0.00157779) | 0.090876 ± 0.000176761 (W 0.00115387) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 2.01× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.055118 ± 6.65833e-05 (W 0.00016562) | 0.078407 ± 0.000619681 (W 0.00248798) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.178385 ± 0.00046769 (W 0.00171956) | 0.312807 ± 0.00363248 (W 0.00841264) | 0.185764 ± 0.00111381 (W 0.00176121) | 🔵 colmat 1.75× faster than `faer` | · |  |
| `solve/n256/b1 [api-inclusive]` | 0.370985 ± 0.00218516 (W 0.00303001) | 0.358987 ± 0.00296716 (W 0.00911507) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.748204 ± 0.00438014 (W 0.00227728) | 1.43337 ± 0.026601 (W 0.0504579) | 0.976776 ± 0.015612 (W 0.00389951) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 1.62075 ± 0.00784358 (W 0.00312022) | 1.65895 ± 0.0153973 (W 0.0364789) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.26507 ± 0.000992726 (W 0.0036042) | 1.79631 ± 0.00309891 (W 0.00902593) | 1.67508 ± 0.00356326 (W 0.00654445) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.27496 ± 0.00448954 (W 0.00204281) | 1.80354 ± 0.00238087 (W 0.00459578) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.85758 ± 0.0273398 (W 0.00856456) | 8.3177 ± 0.0192015 (W 0.0131064) | 8.23177 ± 0.0652904 (W 0.0202941) | 🔵 colmat 1.71× faster than `faer` | 🔵 colmat 1.69× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 4.88775 ± 0.0141755 (W 0.00588006) | 8.33732 ± 0.0489307 (W 0.0211766) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 20.7278 ± 0.0672286 (W 0.0284837) | 35.1638 ± 0.340626 (W 0.0554753) | 43.5018 ± 0.217898 (W 0.0401727) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 2.10× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 20.8107 ± 0.0517248 (W 0.0107944) | 35.0207 ± 0.0759193 (W 0.0640085) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.622019 ± 0.00209089 (W 0.00130876) | 1.3908 ± 0.00178889 (W 0.00255793) | 0.719431 ± 0.00313603 (W 0.0018566) | 🔵 colmat 2.24× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.620436 ± 0.00547673 (W 0.00118477) | 1.39448 ± 0.00189808 (W 0.00224628) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.60539 ± 0.00776927 (W 0.00515552) | 6.73686 ± 0.0106717 (W 0.00776826) | 3.94954 ± 0.0138471 (W 0.00453834) | 🔵 colmat 2.59× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 2.59507 ± 0.00399461 (W 0.00503347) | 6.75525 ± 0.0111921 (W 0.0142254) | - | 🔵 colmat 2.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.2466 ± 0.0155676 (W 0.00998147) | 29.015 ± 0.416875 (W 0.0585614) | 18.2525 ± 0.697787 (W 0.0419117) | 🔵 colmat 2.58× faster than `faer` | 🔵 colmat 1.62× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 11.2568 ± 0.0168287 (W 0.0155616) | 28.4477 ± 0.06857 (W 0.0451298) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00378 ± 2.08167e-05 (W 3.62519e-05) | 0.003229 ± 1.00167e-05 (W 1.747e-05) | 0.0017 ± 5.7735e-06 (W 2.38747e-05) | 🔴 colmat 1.17× slower than `faer` | 🔴 colmat 2.22× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.009709 ± 0.000190071 (W 0.000108808) | 0.00824 ± 3e-05 (W 4.4264e-05) | 0.00497 ± 1.1547e-05 (W 1.90578e-05) | 🔴 colmat 1.18× slower than `faer` | 🔴 colmat 1.95× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.025 ± 0.000215198 (W 0.000256457) | 0.023829 ± 2.3094e-05 (W 0.000261094) | 0.020319 ± 0.00119321 (W 0.000102739) | · | 🔴 colmat 1.23× slower than `lapack-mkl` | `!` between-start range over 5% |

## linalg-mid, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.025779 ± 0.000406489 (W 0.000144859) | 0.035808 ± 0.000115023 (W 0.000247184) | 0.055568 ± 0.00053563 (W 0.000184701) | 🔵 colmat 1.39× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.030559 ± 0.000507182 (W 0.000246313) | 0.042548 ± 9.12159e-05 (W 0.000306007) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.082037 ± 0.000858079 (W 0.00187439) | 0.184134 ± 0.0150264 (W 0.0126053) | 0.162745 ± 0.00107098 (W 0.00118343) | 🔵 colmat 2.24× faster than `faer` | 🔵 colmat 1.98× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.104227 ± 0.000384279 (W 0.000434514) | 0.182393 ± 0.00130814 (W 0.0136248) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.301559 ± 0.00227161 (W 0.00250242) | 0.549404 ± 0.00619283 (W 0.02291) | 0.643906 ± 0.00166889 (W 0.0202897) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 2.14× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.383929 ± 0.00122371 (W 0.00276348) | 0.639472 ± 0.0137884 (W 0.0226422) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.432036 ± 0.00195601 (W 0.00219528) | 1.53747 ± 0.000121791 (W 0.00226165) | 0.654622 ± 0.0044765 (W 0.00261542) | 🔵 colmat 3.56× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.437154 ± 0.000417777 (W 0.00435046) | 1.53763 ± 0.00125182 (W 0.0021997) | - | 🔵 colmat 3.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.55325 ± 0.00243917 (W 0.00445259) | 4.48735 ± 0.0131438 (W 0.0196885) | 2.54555 ± 0.00476173 (W 0.00551146) | 🔵 colmat 2.89× faster than `faer` | 🔵 colmat 1.64× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 1.56072 ± 0.00526965 (W 0.00229095) | 4.50703 ± 0.0287735 (W 0.0225702) | - | 🔵 colmat 2.89× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 6.17439 ± 0.029954 (W 0.0140627) | 14.6141 ± 0.0473698 (W 2.63178) | 9.44363 ± 0.0181941 (W 0.0241792) | 🔵 colmat 2.37× faster than `faer` | 🔵 colmat 1.53× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 6.15798 ± 0.0161967 (W 0.00559878) | 14.7251 ± 0.839924 (W 4.09425) | - | 🔵 colmat 2.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1` | 0.168334 ± 0.00182743 (W 0.00158546) | 0.199205 ± 0.000726286 (W 0.00122929) | 1.03417 ± 0.00141851 (W 0.00294452) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 6.14× faster than `lapack-mkl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.168524 ± 0.00111844 (W 0.00189102) | 0.190863 ± 0.000752257 (W 0.00155123) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.595709 ± 0.00272149 (W 0.00393321) | 1.84067 ± 0.00166427 (W 0.00400735) | 8.74865 ± 0.00956678 (W 0.00925199) | 🔵 colmat 3.09× faster than `faer` | 🔵 colmat 14.69× faster than `lapack-mkl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.759749 ± 0.00389322 (W 0.00760632) | 1.73756 ± 0.0151519 (W 0.00678594) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 2.29142 ± 0.0120116 (W 0.00826618) | 9.67949 ± 0.0311002 (W 0.0157651) | 70.1353 ± 0.128174 (W 0.0396418) | 🔵 colmat 4.22× faster than `faer` | 🔵 colmat 30.61× faster than `lapack-mkl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 3.03054 ± 0.0123366 (W 0.0104394) | 9.35005 ± 0.0962386 (W 0.0151296) | - | 🔵 colmat 3.09× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.033569 ± 0.000418848 (W 0.000457217) | 0.054798 ± 0.0012816 (W 0.00111647) | 0.258412 ± 0.00211159 (W 0.00714463) | 🔵 colmat 1.63× faster than `faer` | 🔵 colmat 7.70× faster than `lapack-mkl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.033559 ± 0.000190788 (W 0.000361912) | 0.054888 ± 0.000242323 (W 0.00151453) | 0.057818 ± 0.000455082 (W 0.00052267) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.145325 ± 0.00186964 (W 0.00269404) | 0.212302 ± 0.00434151 (W 0.00701497) | 1.19608 ± 0.0109501 (W 0.00721212) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 8.23× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.145126 ± 0.000663197 (W 0.0017212) | 0.214553 ± 0.00320055 (W 0.00895855) | 0.322098 ± 0.00112503 (W 0.00158863) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 2.22× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.02578 ± 0.00948014 (W 0.00545358) | 1.36881 ± 0.00560566 (W 0.113756) | 9.38399 ± 0.219078 (W 0.0181615) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 9.15× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.02548 ± 0.0123377 (W 0.00645411) | 1.36615 ± 0.148089 (W 0.14643) | 2.05131 ± 0.00930572 (W 0.0150702) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 2.00× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n128/b1` | 0.046288 ± 0.000317097 (W 0.000267881) | 0.070708 ± 0.000240563 (W 0.000531056) | 0.091927 ± 0.000378194 (W 0.000227944) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 1.99× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.053338 ± 0.000185086 (W 0.000523993) | 0.078838 ± 0.00062 (W 0.000948397) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.162535 ± 0.00119678 (W 0.00125234) | 0.388079 ± 0.0154366 (W 0.0138167) | 0.189674 ± 0.000573615 (W 0.00207566) | 🔵 colmat 2.39× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.351437 ± 0.00168607 (W 0.00227557) | 0.421546 ± 0.00498194 (W 0.0125596) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.634947 ± 0.000274831 (W 0.00249839) | 1.7429 ± 0.0118951 (W 0.0445068) | 0.73379 ± 0.00330309 (W 0.00353846) | 🔵 colmat 2.74× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 1.50599 ± 0.0142168 (W 0.00538275) | 1.83342 ± 0.0428413 (W 0.0545537) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.154865 ± 0.000949653 (W 0.000283447) | 0.430304 ± 0.00155148 (W 0.00375446) | 0.153985 ± 0.00100801 (W 0.00104769) | 🔵 colmat 2.78× faster than `faer` | · |  |
| `qr/n128/b1 [api-inclusive]` | 0.222472 ± 0.000484934 (W 0.00123353) | 0.492665 ± 0.00211005 (W 0.0022863) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.477636 ± 0.00169055 (W 0.00212926) | 1.27994 ± 0.00594745 (W 0.00681926) | 0.651322 ± 0.00176385 (W 0.00327974) | 🔵 colmat 2.68× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 0.802897 ± 0.00137298 (W 0.00404162) | 1.32562 ± 0.0136071 (W 0.00832311) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 1.74458 ± 0.00242273 (W 0.00264199) | 4.11548 ± 0.0308477 (W 0.0249818) | 3.38034 ± 0.01103 (W 0.0130076) | 🔵 colmat 2.36× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 3.04343 ± 0.0164967 (W 0.00713884) | 4.28969 ± 0.0178345 (W 0.0207562) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.104276 ± 0.00046998 (W 0.000229672) | 0.3631 ± 0.00338675 (W 0.00347464) | 0.094007 ± 0.00126961 (W 0.00107361) | 🔵 colmat 3.48× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 0.360387 ± 0.00159212 (W 0.00219358) | 1.04152 ± 0.0047252 (W 0.00524868) | 0.327091 ± 0.00187456 (W 0.00243121) | 🔵 colmat 2.89× faster than `faer` | 🔴 colmat 1.10× slower than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 1.08362 ± 0.00251404 (W 0.00311507) | 2.91388 ± 0.0174337 (W 0.0105584) | 1.34153 ± 0.0152293 (W 0.00655965) | 🔵 colmat 2.69× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.044709 ± 0.000166133 (W 0.000259087) | 0.066917 ± 0.000320169 (W 0.000415745) | 0.090378 ± 0.000228803 (W 0.00132655) | 🔵 colmat 1.50× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.054359 ± 0.000506932 (W 0.000356877) | 0.075347 ± 0.000115832 (W 0.00134983) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.160345 ± 0.000926747 (W 0.001421) | 0.329048 ± 0.00324097 (W 0.0172589) | 0.179163 ± 0.000665814 (W 0.00211091) | 🔵 colmat 2.05× faster than `faer` | 🔵 colmat 1.12× faster than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 0.35814 ± 0.000627991 (W 0.00191854) | 0.374118 ± 0.0154396 (W 0.0137055) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 0.594133 ± 0.00300695 (W 0.00115907) | 1.37236 ± 0.00710465 (W 0.0328371) | 0.718647 ± 0.00443436 (W 0.0072561) | 🔵 colmat 2.31× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 1.49252 ± 0.00571157 (W 0.00332758) | 1.6032 ± 0.00543483 (W 0.0346469) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.26653 ± 0.00255459 (W 0.00381754) | 1.81434 ± 0.00215043 (W 0.00396821) | 1.60502 ± 0.00427713 (W 0.00230359) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.27× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.27841 ± 0.00323547 (W 0.00206363) | 1.82427 ± 0.00158129 (W 0.00365897) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.53399 ± 0.007902 (W 0.00651926) | 8.45017 ± 0.0113211 (W 0.0179309) | 7.40267 ± 0.0205982 (W 0.0159029) | 🔵 colmat 1.86× faster than `faer` | 🔵 colmat 1.63× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 4.54552 ± 0.00219293 (W 0.00409351) | 8.47675 ± 0.0197593 (W 0.0414568) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 17.5983 ± 0.0274303 (W 0.0177678) | 33.6934 ± 0.480774 (W 0.0497974) | 37.3913 ± 0.162015 (W 0.067986) | 🔵 colmat 1.91× faster than `faer` | 🔵 colmat 2.12× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 17.6576 ± 0.0432485 (W 0.0328754) | 33.9039 ± 0.357254 (W 0.647814) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.652662 ± 0.00253769 (W 0.00192053) | 1.38926 ± 0.00180496 (W 0.00231794) | 0.719024 ± 0.00100066 (W 0.00254887) | 🔵 colmat 2.13× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.653742 ± 0.00201247 (W 0.00189354) | 1.39135 ± 0.00187877 (W 0.00242394) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.53674 ± 0.00184535 (W 0.0041111) | 6.81047 ± 0.0146424 (W 0.0167545) | 3.86338 ± 0.00365651 (W 0.00731402) | 🔵 colmat 2.68× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 2.52612 ± 0.0127468 (W 0.00395163) | 6.80038 ± 0.0251384 (W 0.0148485) | - | 🔵 colmat 2.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 9.95882 ± 0.00905116 (W 0.00889743) | 27.868 ± 0.424742 (W 0.0320224) | 16.9476 ± 0.175945 (W 0.0195555) | 🔵 colmat 2.80× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 9.92469 ± 0.0310777 (W 0.0209305) | 27.2872 ± 0.111154 (W 0.0929188) | - | 🔵 colmat 2.75× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00376 ± 2.51661e-05 (W 3.91306e-05) | 0.00321 ± 2.08167e-05 (W 4.60456e-05) | 0.00169 ± 1.52753e-05 (W 3.27109e-05) | 🔴 colmat 1.17× slower than `faer` | 🔴 colmat 2.22× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.009529 ± 0.000221149 (W 0.000187652) | 0.00819 ± 1e-05 (W 6.60697e-05) | 0.004959 ± 1.12694e-05 (W 1.78885e-05) | 🔴 colmat 1.16× slower than `faer` | 🔴 colmat 1.92× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.02342 ± 0.000421956 (W 0.00023803) | 0.022809 ± 0.000470188 (W 0.000730008) | 0.019469 ± 0.000430082 (W 0.000147964) | · | 🔴 colmat 1.20× slower than `lapack-mkl` |  |

## linalg-mid, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.088458 ± 0.000195434 (W 0.000291949) | 0.482806 ± 0.000882648 (W 0.00164303) | - | 🔵 colmat 5.46× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.095648 ± 0.000571572 (W 0.00110758) | 0.099877 ± 0.000259425 (W 0.000655222) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.509707 ± 0.000512081 (W 0.00186878) | 1.32083 ± 0.00174158 (W 0.00333617) | - | 🔵 colmat 2.59× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.536465 ± 0.00486811 (W 0.0016935) | 0.608952 ± 0.00467235 (W 0.00211567) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 3.03234 ± 0.017009 (W 0.00298982) | 4.6934 ± 0.0114708 (W 0.00436197) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 3.17795 ± 0.00260632 (W 0.00199831) | 3.74912 ± 0.00642353 (W 0.00413452) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.84594 ± 0.000501583 (W 0.00114721) | 2.07793 ± 0.00297798 (W 0.00167511) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.84946 ± 0.0026159 (W 0.00236407) | 2.07297 ± 0.00206602 (W 0.00225104) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.59011 ± 0.0279242 (W 0.0138061) | 9.97457 ± 0.0812601 (W 0.00223852) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 9.60506 ± 0.00409379 (W 0.0155358) | 10.0164 ± 0.0129576 (W 0.00289658) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 55.6521 ± 0.05229 (W 0.0405921) | 56.6093 ± 0.104432 (W 0.0811637) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 56.1833 ± 0.21057 (W 0.0352657) | 57.3557 ± 0.0691331 (W 0.0664809) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.427548 ± 0.00101136 (W 0.00135023) | 0.517577 ± 0.00179869 (W 0.00451417) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.491677 ± 0.00194834 (W 0.00220052) | 0.578434 ± 0.00360242 (W 0.00254475) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.9813 ± 0.0231228 (W 0.0109284) | 4.24914 ± 0.0364196 (W 0.00728162) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.33808 ± 0.0160094 (W 0.00967521) | 4.69087 ± 0.0542094 (W 0.0102913) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 22.964 ± 0.041013 (W 0.0388373) | 29.584 ± 0.0528603 (W 0.0115854) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 24.2339 ± 0.044024 (W 0.0140014) | 31.0224 ± 0.0202494 (W 0.0372161) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.426351 ± 0.00409842 (W 0.0017253) | 0.422009 ± 0.00111198 (W 0.00203664) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.425939 ± 0.00404798 (W 0.00147273) | 0.424889 ± 0.00659696 (W 0.00162469) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 2.90212 ± 0.0077934 (W 0.00565539) | 2.90134 ± 0.00317968 (W 0.00190031) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 2.89933 ± 0.00530226 (W 0.00142441) | 2.90494 ± 0.00569913 (W 0.00242083) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 20.6638 ± 0.0360197 (W 0.0232826) | 20.59 ± 0.0224927 (W 0.0299061) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 20.6775 ± 0.00884178 (W 0.0219024) | 20.5967 ± 0.0119525 (W 0.0174641) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.143166 ± 0.000157934 (W 0.000669882) | 0.357163 ± 0.000957773 (W 0.00163614) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.240009 ± 0.000347575 (W 0.00201005) | 0.244004 ± 0.00259101 (W 0.00170324) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.85316 ± 0.00277662 (W 0.00341) | 3.27295 ± 0.00855786 (W 0.0128984) | - | 🔵 colmat 3.84× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.33864 ± 0.00987309 (W 0.00334714) | 1.55821 ± 0.0110126 (W 0.00301991) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.4092 ± 0.0126453 (W 0.0062356) | 17.1335 ± 0.0172457 (W 0.0392354) | - | 🔵 colmat 3.17× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.41812 ± 0.0634971 (W 0.00581203) | 8.86892 ± 0.00553425 (W 0.00853203) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.692079 ± 0.00273933 (W 0.00181621) | 0.943966 ± 0.00655474 (W 0.00115197) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.814983 ± 0.00164379 (W 0.00173723) | 0.878546 ± 0.00395608 (W 0.00117888) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 4.31116 ± 0.0873001 (W 0.00832318) | 5.6616 ± 0.202468 (W 0.00504399) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 4.86638 ± 0.0242579 (W 0.0036959) | 5.34981 ± 0.0113269 (W 0.00793284) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 27.5205 ± 0.0905154 (W 0.014768) | 34.405 ± 0.0586776 (W 0.0122659) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 28.8012 ± 0.0763006 (W 0.0181637) | 30.4869 ± 0.0394903 (W 0.0170042) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.336811 ± 0.000194556 (W 0.0011587) | 0.596094 ± 0.00220155 (W 0.00146562) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.12355 ± 0.00836648 (W 0.00275882) | 3.68083 ± 0.0553376 (W 0.00443335) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 12.1603 ± 0.0114354 (W 0.0135862) | 22.1355 ± 0.15596 (W 0.0520961) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.136256 ± 0.0013646 (W 0.0014087) | 0.247243 ± 0.00173626 (W 0.00231543) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.244183 ± 0.00119211 (W 0.00144045) | 0.240444 ± 0.00454834 (W 0.00264627) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.834038 ± 0.0077391 (W 0.0023988) | 2.49142 ± 0.037585 (W 0.00545592) | - | 🔵 colmat 2.99× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.975653 ± 0.00358065 (W 0.00381489) | 1.12169 ± 0.0154698 (W 0.0019141) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.29099 ± 0.0643088 (W 0.00584411) | 13.2059 ± 0.181291 (W 0.0143886) | - | 🔵 colmat 2.50× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 5.75601 ± 0.00521124 (W 0.00839559) | 7.27372 ± 0.0109641 (W 0.0109028) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.71435 ± 0.00296275 (W 0.00287208) | 3.42084 ± 0.00248258 (W 0.00353946) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.71617 ± 0.00610533 (W 0.00439851) | 3.43249 ± 0.00316499 (W 0.00309936) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 15.3184 ± 0.0193354 (W 0.038367) | 19.5852 ± 0.240602 (W 0.019229) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 15.4083 ± 0.0356708 (W 0.0334713) | 19.5798 ± 0.237643 (W 0.0195251) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 95.3086 ± 0.311805 (W 0.118984) | 115.656 ± 0.115445 (W 0.340261) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 96.9195 ± 0.0849952 (W 0.262218) | 115.76 ± 0.168047 (W 0.14269) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.03168 ± 0.00457055 (W 0.0019001) | 2.31787 ± 0.004885 (W 0.00333222) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.03256 ± 0.00295223 (W 0.00205481) | 2.32164 ± 0.00464152 (W 0.00363311) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.95012 ± 0.108004 (W 0.021672) | 13.0365 ± 0.226581 (W 0.0169273) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 7.11088 ± 0.15552 (W 0.0325269) | 13.0071 ± 0.142529 (W 0.0184598) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 49.9257 ± 0.0297953 (W 0.053683) | 77.6847 ± 0.199471 (W 0.0847813) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 49.9874 ± 0.0653693 (W 0.0439114) | 77.4314 ± 0.0659498 (W 0.102902) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00499 ± 3.00056e-05 (W 3.49285e-05) | 0.00478 ± 2.5632e-05 (W 3.36482e-05) | 0.00501 ± 5.19615e-06 (W 1.54984e-05) | · | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01563 ± 0.00021159 (W 0.000937863) | 0.01538 ± 0.000282064 (W 6.49099e-05) | 0.017189 ± 0.000133742 (W 7.14234e-05) | · | 🔵 colmat 1.10× faster than `lapack-mkl` (marginal: spread 0.60 of the difference) | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.058448 ± 0.00114427 (W 0.000391262) | 0.057839 ± 0.000266008 (W 0.000568413) | 0.063858 ± 0.000238496 (W 0.000152414) | · | 🔵 colmat 1.09× faster than `lapack-mkl` |  |

## linalg-mid, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.069688 ± 0.000368284 (W 0.000251961) | 0.486247 ± 0.002132 (W 0.0712682) | - | 🔵 colmat 6.98× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.079898 ± 0.000369216 (W 0.0002129) | 0.106917 ± 0.000607928 (W 0.00254719) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.26185 ± 0.0032571 (W 0.00152923) | 1.1746 ± 0.0121025 (W 0.0692274) | - | 🔵 colmat 4.49× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.319139 ± 0.000564899 (W 0.00234954) | 0.577834 ± 0.0179644 (W 0.00612218) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 1.03642 ± 0.00423127 (W 0.00378377) | 2.83417 ± 0.0473066 (W 0.0789273) | - | 🔵 colmat 2.73× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.23889 ± 0.00381813 (W 0.0022066) | 2.12555 ± 0.0488576 (W 0.0382346) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.787399 ± 0.0019347 (W 0.00390008) | 1.90331 ± 0.00422704 (W 0.00393793) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.793384 ± 0.00113192 (W 0.00228265) | 1.90254 ± 0.00261853 (W 0.00367283) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.2031 ± 0.00453997 (W 0.00950559) | 7.17962 ± 0.0291057 (W 0.0141347) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.22485 ± 0.0036373 (W 0.0037416) | 7.18242 ± 0.0325731 (W 0.0115057) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.5233 ± 0.0355729 (W 0.019562) | 27.004 ± 0.174348 (W 0.0165453) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.6485 ± 0.0262678 (W 0.0229249) | 26.9639 ± 0.0496957 (W 0.0306162) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.25952 ± 0.00147912 (W 0.00181432) | 0.515946 ± 0.00331353 (W 0.00316893) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.34406 ± 0.00223089 (W 0.0018641) | 0.509273 ± 0.000746989 (W 0.0018325) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.13182 ± 0.00563074 (W 0.00154276) | 4.20555 ± 0.0299534 (W 0.00686579) | - | 🔵 colmat 3.72× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.63086 ± 0.00822203 (W 0.00467607) | 4.35012 ± 0.142607 (W 0.00655018) | - | 🔵 colmat 2.67× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1` | 6.53895 ± 0.0274905 (W 0.00462541) | 18.6151 ± 0.0842201 (W 0.00593724) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 8.62842 ± 0.0141498 (W 0.0234245) | 19.3529 ± 0.0437858 (W 0.0265536) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.127697 ± 0.000279216 (W 0.0010119) | 0.123136 ± 0.000983357 (W 0.00081164) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.127595 ± 0.000317807 (W 0.0121837) | 0.123136 ± 0.000415492 (W 0.00118458) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.784664 ± 0.00226657 (W 0.00452648) | 0.786764 ± 0.001692 (W 0.00374275) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.789708 ± 0.00827385 (W 0.00530813) | 0.78339 ± 0.0142689 (W 0.0016678) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 5.30542 ± 0.00707135 (W 0.00730143) | 5.24646 ± 0.00788551 (W 0.011221) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 5.31853 ± 0.0017775 (W 0.0069796) | 5.24329 ± 0.00179141 (W 0.00897016) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.128035 ± 0.00033841 (W 0.000478551) | 0.366035 ± 0.000648673 (W 0.00263057) | - | 🔵 colmat 2.86× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.208204 ± 0.000687215 (W 0.00198929) | 0.179134 ± 0.00219723 (W 0.00270805) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.458204 ± 0.00139057 (W 0.00232762) | 2.71021 ± 0.0700791 (W 0.0808245) | - | 🔵 colmat 5.91× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.959802 ± 0.00570674 (W 0.00135095) | 1.00169 ± 0.00759235 (W 0.00991883) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.10176 ± 0.00502566 (W 0.00607705) | 11.1008 ± 0.334535 (W 0.417282) | - | 🔵 colmat 5.28× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 4.19719 ± 0.00399146 (W 0.00716648) | 4.95794 ± 0.0386241 (W 0.0660402) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.434956 ± 0.0025158 (W 0.00210281) | 0.816322 ± 0.00342679 (W 0.00437073) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.566361 ± 0.0019074 (W 0.00445853) | 0.73358 ± 0.00310442 (W 0.00248882) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.48474 ± 0.0217925 (W 0.00351085) | 3.8425 ± 0.0321338 (W 0.0116837) | - | 🔵 colmat 2.59× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.18849 ± 0.00870009 (W 0.00537131) | 2.9485 ± 0.0444735 (W 0.00508747) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 8.81187 ± 0.355603 (W 0.0138663) | 18.3188 ± 0.909701 (W 0.0305532) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 10.9163 ± 0.0420133 (W 0.0148029) | 13.5726 ± 0.00485171 (W 0.0203472) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.271539 ± 0.000399772 (W 0.00136309) | 0.643265 ± 0.00149305 (W 0.00209391) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.874386 ± 0.00330022 (W 0.00190144) | 3.2047 ± 0.00978261 (W 0.0106516) | - | 🔵 colmat 3.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 4.16074 ± 0.0104849 (W 0.010008) | 13.8564 ± 0.623532 (W 0.0489789) | - | 🔵 colmat 3.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.108667 ± 0.000839962 (W 0.00082476) | 0.2575 ± 0.00177988 (W 0.00213984) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.208952 ± 0.000742382 (W 0.00175072) | 0.172394 ± 0.00161283 (W 0.00200707) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.472392 ± 0.00314816 (W 0.002804) | 2.15371 ± 0.032158 (W 0.032894) | - | 🔵 colmat 4.56× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.950938 ± 0.00167549 (W 0.00189169) | 0.914889 ± 0.0322146 (W 0.00548705) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 2.10268 ± 0.00394226 (W 0.00667614) | 8.67963 ± 0.153233 (W 0.194481) | - | 🔵 colmat 4.13× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.14558 ± 0.0122005 (W 0.006104) | 4.87493 ± 0.0815905 (W 0.073289) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.7021 ± 0.00894046 (W 0.00481758) | 2.587 ± 0.00310813 (W 0.00706288) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.71668 ± 0.0100102 (W 0.00373758) | 2.60197 ± 0.00468535 (W 0.00954682) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.3377 ± 0.16283 (W 0.0130473) | 13.7438 ± 0.11453 (W 0.0182408) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.3937 ± 0.0281588 (W 0.0101175) | 13.7715 ± 0.105027 (W 0.0316718) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 39.0263 ± 0.192894 (W 0.057536) | 53.6711 ± 2.47218 (W 0.0893809) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 39.2193 ± 0.131573 (W 0.0447153) | 54.4262 ± 0.71663 (W 0.73086) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.827852 ± 0.00314014 (W 0.00247273) | 2.00804 ± 0.0019084 (W 0.00391128) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.828042 ± 0.00111514 (W 0.00213499) | 2.00752 ± 0.00238636 (W 0.00319096) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.99423 ± 0.0220844 (W 0.0169184) | 11.1774 ± 0.00776352 (W 0.0207025) | - | 🔵 colmat 2.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.04913 ± 0.0245086 (W 0.00988048) | 11.1842 ± 0.00787239 (W 0.0247041) | - | 🔵 colmat 2.76× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 22.6339 ± 0.789999 (W 0.0687919) | 41.7663 ± 0.734589 (W 0.666593) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 22.7105 ± 0.834633 (W 0.0370594) | 41.2823 ± 0.697784 (W 0.611524) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n128/b1` | 0.00573 ± 4.62637e-05 (W 3.50599e-05) | 0.00479 ± 1.0504e-05 (W 3.57379e-05) | 0.005009 ± 5.50757e-06 (W 1.49399e-05) | 🔴 colmat 1.20× slower than `faer` | 🔴 colmat 1.14× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013709 ± 8.03762e-05 (W 5.10127e-05) | 0.015099 ± 0.00017094 (W 0.00106619) | 0.017149 ± 5.19615e-05 (W 8.07546e-05) | 🔵 colmat 1.10× faster than `faer` (marginal: spread 0.77 of the difference) | 🔵 colmat 1.25× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.036249 ± 0.000676782 (W 0.000126965) | 0.042218 ± 0.000716694 (W 0.000308276) | 0.063878 ± 0.000191572 (W 0.00019364) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 1.76× faster than `lapack-mkl` |  |

## linalg-mid, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.070028 ± 8.03762e-05 (W 0.000380843) | 0.32149 ± 0.0951575 (W 0.087334) | - | 🔵 colmat 4.59× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.080638 ± 0.000477347 (W 0.000217876) | 0.097227 ± 0.000766851 (W 0.000925225) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.228653 ± 0.000320204 (W 0.00196264) | 0.870592 ± 0.086564 (W 0.113741) | - | 🔵 colmat 3.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.293132 ± 0.000921581 (W 0.00207102) | 0.473593 ± 0.00557893 (W 0.0174076) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.762493 ± 0.0138678 (W 0.00267889) | 2.09839 ± 0.0701712 (W 0.18072) | - | 🔵 colmat 2.75× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.989968 ± 0.00288941 (W 0.00241449) | 1.885 ± 0.0188156 (W 0.0120882) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.614403 ± 0.00166799 (W 0.00181859) | 1.91718 ± 0.00141935 (W 0.00201618) | - | 🔵 colmat 3.12× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.621823 ± 0.00070064 (W 0.00328918) | 1.9129 ± 0.00241908 (W 0.00194873) | - | 🔵 colmat 3.08× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.30734 ± 0.00368917 (W 0.00428047) | 7.14338 ± 7.83181 (W 0.0183333) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n256/b1 [api-inclusive]` | 2.33123 ± 0.00761565 (W 0.00534305) | 7.18164 ± 0.0453718 (W 0.0296291) | - | 🔵 colmat 3.08× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.538 ± 0.0229091 (W 0.029347) | 24.2014 ± 0.273139 (W 0.0503874) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.6259 ± 0.0318792 (W 0.0479518) | 23.7225 ± 0.196826 (W 0.0608148) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.235533 ± 0.000387531 (W 0.00215255) | 0.518543 ± 0.00634924 (W 0.00414899) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.328088 ± 0.00113551 (W 0.00293283) | 0.508231 ± 0.00256034 (W 0.00461171) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.875592 ± 0.0833041 (W 0.00128625) | 4.195 ± 0.0299648 (W 0.00702817) | - | 🔵 colmat 4.79× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.38423 ± 0.00806674 (W 0.00445192) | 4.1399 ± 0.00998553 (W 0.00318542) | - | 🔵 colmat 2.99× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 4.00476 ± 0.010831 (W 0.00322557) | 17.0118 ± 0.0836136 (W 0.0280586) | - | 🔵 colmat 4.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 6.15921 ± 0.0104835 (W 0.0107476) | 16.6097 ± 0.711476 (W 0.0147736) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.101216 ± 0.00107084 (W 0.00121097) | 0.126866 ± 0.000869186 (W 0.000850794) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.100847 ± 0.000600999 (W 0.0017397) | 0.125757 ± 0.000278083 (W 0.00104855) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.438778 ± 0.00258372 (W 0.00136926) | 0.558343 ± 0.0274224 (W 0.0250385) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.442856 ± 0.00190687 (W 0.00211677) | 0.56096 ± 0.00592462 (W 0.0341608) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.7829 ± 0.00273004 (W 0.0101152) | 3.46782 ± 0.420994 (W 0.372545) | - | 🔵 colmat 1.25× faster than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.78493 ± 0.0215398 (W 0.00783312) | 3.46326 ± 0.337257 (W 0.386925) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.130216 ± 0.000388313 (W 0.00126798) | 0.36157 ± 0.000693839 (W 0.00259604) | - | 🔵 colmat 2.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.201154 ± 0.000972793 (W 0.00178147) | 0.170934 ± 0.00157011 (W 0.001005) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.392316 ± 0.0141339 (W 0.00198188) | 2.70791 ± 0.02118 (W 0.0218512) | - | 🔵 colmat 6.90× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.907637 ± 0.00226622 (W 0.00338265) | 1.20446 ± 0.163772 (W 0.0103199) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 1.47818 ± 0.00450103 (W 0.0040882) | 9.13285 ± 0.0233728 (W 0.0374392) | - | 🔵 colmat 6.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.65531 ± 0.0250277 (W 0.00786895) | 4.44216 ± 0.0105609 (W 0.022883) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.363519 ± 0.000605961 (W 0.00281128) | 0.892325 ± 0.00309 (W 0.00187759) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.512906 ± 0.00137317 (W 0.00312142) | 0.793442 ± 0.00194833 (W 0.00307545) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.161 ± 0.00946166 (W 0.00510126) | 3.82653 ± 0.0788786 (W 0.0109371) | - | 🔵 colmat 3.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.88236 ± 0.00353327 (W 0.00577085) | 2.49895 ± 0.0778606 (W 0.0121925) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 4.99293 ± 0.0702284 (W 0.00702342) | 16.4133 ± 0.12108 (W 0.118639) | - | 🔵 colmat 3.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 7.60405 ± 0.0461448 (W 0.0104303) | 10.6965 ± 0.319328 (W 0.49505) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.247773 ± 0.000721388 (W 0.0020006) | 0.705588 ± 0.00286346 (W 0.00373395) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.778348 ± 0.00385042 (W 0.00227832) | 3.33336 ± 0.0692186 (W 0.0155546) | - | 🔵 colmat 4.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.38712 ± 0.00429431 (W 0.005633) | 13.3904 ± 0.482894 (W 0.0599612) | - | 🔵 colmat 5.61× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.108577 ± 0.000544457 (W 0.000999077) | 0.248013 ± 0.00122341 (W 0.000986958) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.199024 ± 0.000755714 (W 0.0022183) | 0.163174 ± 0.00162116 (W 0.00111811) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.384579 ± 0.00307651 (W 0.00371021) | 2.10081 ± 0.0267875 (W 0.0228795) | - | 🔵 colmat 5.46× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.514445 ± 0.00412456 (W 0.00348307) | 1.113 ± 0.133704 (W 0.00597712) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 1.45095 ± 0.00609517 (W 0.00397162) | 7.478 ± 0.0683914 (W 0.0594341) | - | 🔵 colmat 5.15× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.64822 ± 0.0102625 (W 0.00820184) | 4.25389 ± 0.00678389 (W 0.0150131) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.61465 ± 0.00418092 (W 0.00620051) | 2.61093 ± 0.00831257 (W 0.00696089) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.61474 ± 0.00887818 (W 0.00217996) | 2.61631 ± 0.0109796 (W 0.00354491) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 6.46435 ± 0.134155 (W 0.0120881) | 13.7341 ± 0.0136515 (W 0.0208196) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 6.51061 ± 0.00800929 (W 0.00611992) | 13.8772 ± 0.071496 (W 0.0629793) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 29.0104 ± 0.693221 (W 0.0516668) | 46.8832 ± 2.43322 (W 0.0888992) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 30.4079 ± 0.662429 (W 0.047389) | 48.0904 ± 0.75318 (W 0.198705) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.83665 ± 0.00211152 (W 0.00278569) | 2.00869 ± 0.000321532 (W 0.00337928) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.827874 ± 0.00472347 (W 0.00221227) | 2.01218 ± 0.00423145 (W 0.00561818) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.76435 ± 0.130907 (W 0.00972236) | 11.3934 ± 0.0560969 (W 0.0251934) | - | 🔵 colmat 3.03× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1 [api-inclusive]` | 3.94799 ± 0.130776 (W 0.00526363) | 11.2966 ± 0.0269607 (W 0.015816) | - | 🔵 colmat 2.86× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n512/b1` | 17.302 ± 0.0522487 (W 0.0160426) | 36.482 ± 0.907009 (W 0.0706941) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 17.2714 ± 0.0252726 (W 0.0177847) | 35.7979 ± 0.198961 (W 0.0559343) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00573 ± 1.52753e-05 (W 2.9278e-05) | 0.00472 ± 3.82143e-05 (W 1.83085e-05) | 0.005 ± 6.08276e-06 (W 1.50732e-05) | 🔴 colmat 1.21× slower than `faer` | 🔴 colmat 1.15× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01341 ± 0.000110937 (W 3.80039e-05) | 0.01518 ± 0.000243763 (W 6.25724e-05) | 0.017179 ± 5.80259e-05 (W 7.70474e-05) | 🔵 colmat 1.13× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.031749 ± 0.000170392 (W 0.000173118) | 0.038718 ± 0.000352846 (W 0.000459481) | 0.064068 ± 0.000298664 (W 0.000493275) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-mkl` |  |

## linalg-mid, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.079717 ± 0.000213378 (W 0.000295427) | 0.088548 ± 0.000285507 (W 0.000143552) | 0.261783 ± 0.0019025 (W 0.00274095) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 3.28× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.088608 ± 0.000111195 (W 0.000131024) | 0.092218 ± 0.000379057 (W 0.000978287) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.463857 ± 0.00334595 (W 0.00164913) | 0.518555 ± 0.00265341 (W 0.00211871) | 1.22124 ± 0.00818843 (W 0.00435474) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 2.63× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.507237 ± 0.00058075 (W 0.00132738) | 0.523705 ± 0.00749211 (W 0.00136399) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.88034 ± 0.00902259 (W 0.00256205) | 3.00143 ± 0.00486308 (W 0.00160931) | 8.3758 ± 0.0466233 (W 0.0155921) | · | 🔵 colmat 2.91× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 3.04863 ± 0.00884602 (W 0.00201329) | 3.04054 ± 0.00504728 (W 0.00195124) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.84413 ± 0.00229195 (W 0.0015285) | 2.0678 ± 0.00356203 (W 0.00172921) | 2.7776 ± 0.00338896 (W 0.00307378) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.84113 ± 0.00315271 (W 0.0055438) | 2.06657 ± 0.00157921 (W 0.000602063) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.54406 ± 0.0124367 (W 0.010459) | 9.92262 ± 0.0356609 (W 0.00792499) | 16.5492 ± 0.0100586 (W 0.0207755) | · | 🔵 colmat 1.73× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 9.55986 ± 0.00319789 (W 0.00889802) | 9.92021 ± 0.00492092 (W 0.00739398) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 55.5897 ± 0.0909272 (W 0.0220282) | 55.8771 ± 0.076541 (W 0.0479392) | 104.209 ± 0.272146 (W 0.151547) | · | 🔵 colmat 1.87× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 55.9056 ± 0.0133497 (W 0.0523751) | 56.1056 ± 0.101697 (W 0.0451039) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.39672 ± 0.00173713 (W 0.00103394) | 0.484626 ± 0.00589863 (W 0.00424729) | 0.81107 ± 0.00157305 (W 0.00399013) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 2.04× faster than `lapack-mkl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.475758 ± 0.00255528 (W 0.00199158) | 0.558375 ± 0.000965756 (W 0.0033378) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.89023 ± 0.0414339 (W 0.00288604) | 3.83784 ± 0.0424637 (W 0.00598442) | 6.60429 ± 0.0211133 (W 0.00646784) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 2.29× faster than `lapack-mkl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.24812 ± 0.0142365 (W 0.0116554) | 4.18905 ± 0.00342201 (W 0.0215492) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 23.9108 ± 0.147728 (W 0.0226671) | 28.0839 ± 0.0761553 (W 0.0174184) | 49.1877 ± 0.0377099 (W 0.0501374) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 2.06× faster than `lapack-mkl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 24.0868 ± 0.0108461 (W 0.0266075) | 27.819 ± 0.0175258 (W 0.0214848) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.41945 ± 0.00183252 (W 0.00103991) | 0.421969 ± 0.00382264 (W 0.00147742) | 1.21049 ± 0.00580723 (W 0.00395766) | · | 🔵 colmat 2.89× faster than `lapack-mkl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.423348 ± 0.000882895 (W 0.00150886) | 0.414638 ± 0.00315185 (W 0.00132258) | 1.07577 ± 0.00307046 (W 0.00229679) | · | 🔵 colmat 2.54× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 2.88192 ± 0.0120604 (W 0.00199282) | 2.86471 ± 0.00459861 (W 0.00311452) | 8.87614 ± 0.0063953 (W 0.0192551) | · | 🔵 colmat 3.08× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 2.88186 ± 0.00718624 (W 0.00173924) | 2.86801 ± 0.00400715 (W 0.00175273) | 6.98716 ± 0.0113873 (W 0.012817) | · | 🔵 colmat 2.42× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 20.312 ± 0.0600616 (W 0.0196196) | 20.2916 ± 0.0599952 (W 0.0346066) | 58.1842 ± 0.0271211 (W 0.0732678) | · | 🔵 colmat 2.86× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 20.3102 ± 0.0353587 (W 0.0239841) | 20.2762 ± 0.0257504 (W 0.0193127) | 41.49 ± 0.054825 (W 0.0326907) | · | 🔵 colmat 2.04× faster than `lapack-mkl` |  |
| `lu/n128/b1` | 0.137954 ± 0.000292733 (W 0.00122459) | 0.136846 ± 5.17333e-05 (W 0.00113934) | 0.284822 ± 0.000274871 (W 0.00248767) | · | 🔵 colmat 2.06× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.222053 ± 0.000606985 (W 0.00164083) | 0.218084 ± 0.000391314 (W 0.00185039) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.808687 ± 0.00106054 (W 0.00151269) | 0.891145 ± 0.00529542 (W 0.00230643) | 2.03381 ± 0.00506362 (W 0.0027249) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 2.51× faster than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 1.24033 ± 0.00339892 (W 0.00207152) | 1.27823 ± 0.00318841 (W 0.00417497) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.18086 ± 0.00238307 (W 0.00779828) | 5.42962 ± 0.014114 (W 0.00379218) | 12.9038 ± 0.00795298 (W 0.0210295) | · | 🔵 colmat 2.49× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 6.94898 ± 0.0110568 (W 0.00564388) | 7.05502 ± 0.0310888 (W 0.00531881) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.677693 ± 0.00521572 (W 0.00177801) | 0.738011 ± 0.0015441 (W 0.00118397) | 1.68126 ± 0.00144984 (W 0.0043869) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 2.48× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.797304 ± 0.00241796 (W 0.00197419) | 0.857748 ± 0.00184472 (W 0.00189987) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 4.1421 ± 0.0237449 (W 0.00248911) | 4.21363 ± 0.0144252 (W 0.00289538) | 10.9293 ± 0.0189778 (W 0.0247369) | · | 🔵 colmat 2.64× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 4.7555 ± 0.0126901 (W 0.0062088) | 4.85058 ± 0.0282019 (W 0.00809349) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 25.9789 ± 0.0289595 (W 0.0120556) | 26.2416 ± 0.0290773 (W 0.0185886) | 68.1024 ± 0.0832994 (W 0.044672) | · | 🔵 colmat 2.62× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 28.3511 ± 0.0444616 (W 0.0122062) | 28.6513 ± 0.0302188 (W 0.0191003) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.328961 ± 0.00128909 (W 0.00126117) | 0.390963 ± 0.000795381 (W 0.00215352) | 0.594473 ± 0.000953796 (W 0.00199952) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 1.95242 ± 0.00159597 (W 0.0019132) | 2.04542 ± 0.00208868 (W 0.00169118) | 3.72163 ± 0.00119787 (W 0.0065274) | · | 🔵 colmat 1.91× faster than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 11.1772 ± 0.0181203 (W 0.0034917) | 11.5633 ± 0.0193041 (W 0.00515846) | 25.7184 ± 0.0570938 (W 0.0273466) | · | 🔵 colmat 2.30× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.144286 ± 0.000127288 (W 0.00102984) | 0.133997 ± 0.000290997 (W 0.000951666) | 0.277212 ± 0.00129204 (W 0.00178602) | 🔴 colmat 1.08× slower than `faer` | 🔵 colmat 1.92× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.227973 ± 0.000864713 (W 0.00215548) | 0.218014 ± 0.00152587 (W 0.00180905) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.828836 ± 0.00711837 (W 0.00204268) | 0.863298 ± 0.0025792 (W 0.00385287) | 1.72672 ± 0.00232928 (W 0.00572818) | · | 🔵 colmat 2.08× faster than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 0.890608 ± 0.00156527 (W 0.00153779) | 0.863938 ± 0.00191195 (W 0.00140177) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.24224 ± 0.0319408 (W 0.00605316) | 5.41773 ± 0.0172426 (W 0.00550081) | 11.4628 ± 0.0234128 (W 0.0194881) | · | 🔵 colmat 2.19× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 5.43799 ± 0.0232822 (W 0.00953611) | 5.41584 ± 0.0118989 (W 0.00237786) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.70657 ± 0.00853728 (W 0.00214491) | 3.39806 ± 0.00226348 (W 0.00243448) | 4.56129 ± 0.00696533 (W 0.00480063) | 🔵 colmat 1.26× faster than `faer` | 🔵 colmat 1.69× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.6968 ± 0.00348937 (W 0.00290476) | 3.40764 ± 0.00643491 (W 0.00288428) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.8456 ± 0.0134994 (W 0.0274015) | 19.0752 ± 0.109789 (W 0.0164699) | 30.7117 ± 0.153369 (W 0.04501) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 2.07× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 14.9333 ± 0.0152073 (W 0.0341167) | 19.0803 ± 0.0643266 (W 0.00858255) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 93.7927 ± 0.392401 (W 0.337849) | 114.199 ± 0.384287 (W 0.22249) | 210.019 ± 3.26478 (W 0.217285) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 2.24× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 94.3305 ± 0.346813 (W 0.433396) | 113.83 ± 0.390429 (W 0.209237) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.01686 ± 0.00298209 (W 0.000651982) | 2.30258 ± 0.00157938 (W 0.00181042) | 1.73664 ± 0.00171423 (W 0.00131977) | 🔵 colmat 2.26× faster than `faer` | 🔵 colmat 1.71× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.01445 ± 0.00398466 (W 0.00107751) | 2.30462 ± 0.00172743 (W 0.00394015) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.55131 ± 0.0419451 (W 0.0157633) | 12.7275 ± 0.0825775 (W 0.0141726) | 13.2094 ± 0.0473974 (W 0.0494677) | 🔵 colmat 1.94× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.56596 ± 0.0523145 (W 0.0166322) | 12.7358 ± 0.0233689 (W 0.0199727) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 46.695 ± 0.479859 (W 0.0630435) | 74.586 ± 0.152921 (W 0.0696746) | 100.775 ± 0.1092 (W 0.0523694) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 46.8339 ± 0.545644 (W 0.408098) | 74.3036 ± 0.171348 (W 0.0311845) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00487 ± 2.08167e-05 (W 3.39116e-05) | 0.00542 ± 5.7735e-06 (W 3.04959e-05) | 0.00505 ± 5.7735e-06 (W 1.12339e-05) | 🔵 colmat 1.11× faster than `faer` | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01522 ± 0.000219933 (W 0.00010483) | 0.01655 ± 9.96845e-05 (W 5.68357e-05) | 0.017699 ± 0.00010504 (W 4.69393e-05) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.056569 ± 0.000275349 (W 0.000330896) | 0.060349 ± 0.00118119 (W 0.000275988) | 0.065958 ± 0.000135795 (W 0.000110815) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-mkl` |  |

## linalg-mid, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.064547 ± 0.000278498 (W 0.000245207) | 0.090148 ± 0.00107169 (W 0.00221688) | 0.26073 ± 0.00098378 (W 0.00194907) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 4.04× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.072558 ± 0.000295014 (W 0.000197411) | 0.095477 ± 0.00128499 (W 0.00217535) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.23995 ± 0.00144113 (W 0.00135762) | 0.353286 ± 0.00282718 (W 0.0033782) | 0.372007 ± 0.00865638 (W 0.00652156) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 1.55× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.29176 ± 0.000896239 (W 0.00119524) | 0.400566 ± 0.00288481 (W 0.00396936) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.983277 ± 0.000730445 (W 0.00242505) | 1.25946 ± 0.0462064 (W 0.0595562) | 2.13901 ± 0.00408377 (W 0.00781798) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 2.18× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.12023 ± 0.000618845 (W 0.00239522) | 1.39774 ± 0.0599906 (W 0.0405347) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 0.77912 ± 0.00220475 (W 0.00170855) | 1.90039 ± 0.00127096 (W 0.00306899) | 1.25122 ± 0.000467984 (W 0.00379518) | 🔵 colmat 2.44× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.78374 ± 0.00190757 (W 0.00189384) | 1.89491 ± 0.00780053 (W 0.00349745) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.16901 ± 0.00319322 (W 0.00318871) | 7.0234 ± 0.0102998 (W 0.0157004) | 6.14656 ± 0.00681599 (W 0.0149425) | 🔵 colmat 2.22× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.19222 ± 0.00186303 (W 0.0065596) | 7.03403 ± 0.00538377 (W 0.0141595) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.3618 ± 0.0165836 (W 0.0123006) | 25.8242 ± 0.343427 (W 0.0156377) | 31.6607 ± 0.199417 (W 0.0323879) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 16.4684 ± 0.0452117 (W 0.0102628) | 25.7086 ± 0.0348676 (W 0.0244004) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.249552 ± 0.00163708 (W 0.00153401) | 0.487421 ± 0.0041638 (W 0.00375831) | 0.812618 ± 0.0018906 (W 0.00313228) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 3.26× faster than `lapack-mkl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.326299 ± 0.00205653 (W 0.00178187) | 0.489511 ± 0.00101396 (W 0.00384678) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.13418 ± 0.010024 (W 0.00355963) | 3.84311 ± 0.0347529 (W 0.00653108) | 6.54399 ± 0.0248585 (W 0.00876872) | 🔵 colmat 3.39× faster than `faer` | 🔵 colmat 5.77× faster than `lapack-mkl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.52729 ± 0.00356166 (W 0.00503219) | 3.83233 ± 0.0123049 (W 0.00894103) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.78801 ± 0.00710391 (W 0.010215) | 17.0993 ± 0.0383632 (W 0.0272217) | 48.919 ± 0.0719197 (W 0.0532452) | 🔵 colmat 2.52× faster than `faer` | 🔵 colmat 7.21× faster than `lapack-mkl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 8.19145 ± 0.0345287 (W 0.0134266) | 16.2662 ± 0.0276812 (W 0.0177963) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.126206 ± 0.000191912 (W 0.000526309) | 0.121237 ± 0.000191366 (W 0.00109623) | 0.471924 ± 0.00120279 (W 0.00244598) | · | 🔵 colmat 3.74× faster than `lapack-mkl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.125045 ± 0.000743858 (W 0.0013453) | 0.120857 ± 0.000579974 (W 0.000279768) | 0.307918 ± 0.000889054 (W 0.0018989) | · | 🔵 colmat 2.46× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.77402 ± 0.00171567 (W 0.00167872) | 0.76724 ± 0.0139066 (W 0.00158056) | 3.46999 ± 0.0108923 (W 0.0145804) | · | 🔵 colmat 4.48× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.773585 ± 0.00311528 (W 0.0022046) | 0.767779 ± 0.00242638 (W 0.00304615) | 1.83899 ± 0.0240069 (W 0.00475735) | · | 🔵 colmat 2.38× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 5.25995 ± 0.00515912 (W 0.00974168) | 5.22714 ± 0.00888865 (W 0.0107842) | 20.3082 ± 0.183649 (W 0.163938) | · | 🔵 colmat 3.86× faster than `lapack-mkl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 5.26599 ± 0.00629187 (W 0.0156694) | 5.21023 ± 0.0070967 (W 0.00591629) | 10.7858 ± 0.30676 (W 0.0156385) | · | 🔵 colmat 2.05× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/n128/b1` | 0.104327 ± 0.000878935 (W 0.00034575) | 0.145434 ± 0.00168068 (W 0.00230351) | 0.285042 ± 0.000326607 (W 0.00145639) | 🔵 colmat 1.39× faster than `faer` | 🔵 colmat 2.73× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.192275 ± 0.00127558 (W 0.00146927) | 0.158143 ± 0.000416607 (W 0.00192608) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.421109 ± 0.00171473 (W 0.0023618) | 0.706996 ± 0.00190577 (W 0.00865779) | 0.591946 ± 0.000874934 (W 0.00370575) | 🔵 colmat 1.68× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-mkl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.852941 ± 0.0100416 (W 0.00234467) | 0.757539 ± 0.00436037 (W 0.0127126) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.94939 ± 0.00469174 (W 0.0040928) | 3.0001 ± 0.024664 (W 0.0548629) | 3.78706 ± 0.013807 (W 0.00714035) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 3.73446 ± 0.00558574 (W 0.00442022) | 3.15257 ± 0.0437706 (W 0.0742683) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.401419 ± 0.000964762 (W 0.00290548) | 0.569588 ± 0.00273821 (W 0.00272726) | 0.58006 ± 0.00190469 (W 0.0012672) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.550309 ± 0.00137603 (W 0.00337879) | 0.706521 ± 0.00178697 (W 0.00311288) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.41418 ± 0.00414582 (W 0.00135729) | 2.02619 ± 0.00325549 (W 0.0123305) | 2.7875 ± 0.0112592 (W 0.00936604) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.97× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 2.08766 ± 0.00347073 (W 0.00449076) | 2.70907 ± 0.00157621 (W 0.00833772) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 7.82793 ± 0.0026818 (W 0.00849366) | 9.04646 ± 0.012566 (W 0.0147183) | 17.4527 ± 0.0502905 (W 0.0282303) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 2.23× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 10.4616 ± 0.0479893 (W 0.0284722) | 11.742 ± 0.0271576 (W 0.0187238) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.2556 ± 0.00069604 (W 0.00144852) | 0.408159 ± 0.00285591 (W 0.00231057) | 0.267093 ± 0.00125051 (W 0.00161019) | 🔵 colmat 1.60× faster than `faer` | · |  |
| `qr_factor/n256/b1` | 0.814267 ± 0.00140153 (W 0.00286016) | 1.4309 ± 0.00242741 (W 0.00464285) | 1.00702 ± 0.00110947 (W 0.00281604) | 🔵 colmat 1.76× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 3.9355 ± 0.0102207 (W 0.00338677) | 5.19401 ± 0.00556964 (W 0.0080418) | 7.00684 ± 0.00446967 (W 0.00522996) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.104886 ± 0.000345616 (W 0.00042224) | 0.139987 ± 0.000727517 (W 0.00191267) | 0.27548 ± 0.000184998 (W 0.00197327) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 2.63× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.189763 ± 0.00104491 (W 0.00312488) | 0.152705 ± 0.00199286 (W 0.0020287) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.409294 ± 0.00160514 (W 0.00255815) | 0.644843 ± 0.00535605 (W 0.0101321) | 0.502621 ± 0.00145576 (W 0.00276236) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-mkl` |  |
| `solve/n256/b1 [api-inclusive]` | 0.843787 ± 0.00238038 (W 0.00295517) | 0.686077 ± 0.00373425 (W 0.00991717) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.90091 ± 0.00487557 (W 0.00413567) | 2.87809 ± 0.0530909 (W 0.0588538) | 3.44557 ± 0.0070258 (W 0.00804544) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 3.69958 ± 0.0171948 (W 0.00552362) | 3.01775 ± 0.0180182 (W 0.0684775) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.6786 ± 0.00243145 (W 0.00675663) | 2.58276 ± 0.00461221 (W 0.00720659) | 2.62046 ± 0.00710965 (W 0.00509059) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.68486 ± 0.00164944 (W 0.00301426) | 2.58815 ± 0.00180134 (W 0.007579) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.10871 ± 0.025602 (W 0.0126279) | 13.5049 ± 0.0442717 (W 0.0275934) | 14.9064 ± 0.0357012 (W 0.0161816) | 🔵 colmat 1.90× faster than `faer` | 🔵 colmat 2.10× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.16278 ± 0.0186856 (W 0.00620616) | 13.5102 ± 0.0276883 (W 0.0305826) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 35.9033 ± 0.0974975 (W 0.039242) | 51.8724 ± 0.221404 (W 0.060056) | 88.2399 ± 0.596643 (W 0.0857517) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 2.46× faster than `lapack-mkl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 36.0427 ± 0.132824 (W 0.0643089) | 51.9811 ± 0.658706 (W 0.710331) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.820348 ± 0.00208649 (W 0.00124364) | 1.98564 ± 0.00591812 (W 0.00294322) | 1.26608 ± 0.00237581 (W 0.00426248) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-mkl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.816658 ± 0.000564589 (W 0.00229607) | 1.98362 ± 0.00152565 (W 0.00307853) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.79978 ± 0.0071173 (W 0.00525338) | 10.8371 ± 0.080721 (W 0.0207821) | 8.04477 ± 0.0265177 (W 0.00968656) | 🔵 colmat 2.85× faster than `faer` | 🔵 colmat 2.12× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.79229 ± 0.0172307 (W 0.011654) | 10.9245 ± 0.0970157 (W 0.0233605) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.6764 ± 0.0862339 (W 0.0555709) | 38.9215 ± 0.194278 (W 0.0314195) | 44.1338 ± 0.971142 (W 0.0524455) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 2.24× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 19.7099 ± 0.0727136 (W 0.0288198) | 38.9607 ± 0.432333 (W 0.0373909) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00551 ± 2.88675e-05 (W 3.2775e-05) | 0.00537 ± 1e-05 (W 2.47427e-05) | 0.00505 ± 0 (W 2.23875e-05) | · | 🔴 colmat 1.09× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013749 ± 0.000185023 (W 0.000146056) | 0.01663 ± 0.000155532 (W 7.4962e-05) | 0.01759 ± 9.50438e-06 (W 3.63497e-05) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.038079 ± 0.000539101 (W 0.000284388) | 0.045808 ± 0.000258496 (W 0.00144822) | 0.065888 ± 0.000182831 (W 0.000156052) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-mkl` |  |

## linalg-mid, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.065458 ± 0.000730571 (W 0.00021842) | 0.081628 ± 0.000154306 (W 0.000399372) | 0.260303 ± 0.00138059 (W 0.00236972) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 3.98× faster than `lapack-mkl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.073448 ± 0.000175594 (W 0.000147211) | 0.089426 ± 0.000417313 (W 0.00200384) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.213804 ± 0.000785152 (W 0.00146323) | 0.350598 ± 0.00145593 (W 0.0123337) | 0.269282 ± 0.000525253 (W 0.00169982) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.265901 ± 0.00209983 (W 0.00167344) | 0.396367 ± 0.00866087 (W 0.0120828) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.735084 ± 0.00391799 (W 0.00330294) | 1.02891 ± 0.023266 (W 0.016079) | 1.1399 ± 0.00269408 (W 0.0120461) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 1.55× faster than `lapack-mkl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.875073 ± 0.00327401 (W 0.00140533) | 1.15948 ± 0.0280555 (W 0.0274446) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.605983 ± 0.00362094 (W 0.00201068) | 1.91075 ± 0.00697349 (W 0.0033725) | 1.04072 ± 0.00222264 (W 0.00314851) | 🔵 colmat 3.15× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-mkl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.60848 ± 0.00277713 (W 0.00482185) | 1.90267 ± 0.000992685 (W 0.00392858) | - | 🔵 colmat 3.13× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.26863 ± 0.00483551 (W 0.00482341) | 7.0104 ± 0.0346228 (W 0.0173551) | 4.82332 ± 0.00440472 (W 0.0161703) | 🔵 colmat 3.09× faster than `faer` | 🔵 colmat 2.13× faster than `lapack-mkl` |  |
| `eigh/n256/b1 [api-inclusive]` | 2.29389 ± 0.000485543 (W 0.00362209) | 7.03884 ± 0.0258206 (W 0.0206952) | - | 🔵 colmat 3.07× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.467 ± 0.02966 (W 0.0149899) | 22.8557 ± 0.0461658 (W 0.0815132) | 21.4234 ± 0.00412929 (W 0.0357212) | 🔵 colmat 2.18× faster than `faer` | 🔵 colmat 2.05× faster than `lapack-mkl` |  |
| `eigh/n512/b1 [api-inclusive]` | 10.5113 ± 0.0521596 (W 0.0160202) | 22.9944 ± 0.0519079 (W 0.0946692) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.233093 ± 0.00201721 (W 0.00114889) | 0.485392 ± 0.0029182 (W 0.00482414) | 0.810697 ± 0.00102815 (W 0.00268642) | 🔵 colmat 2.08× faster than `faer` | 🔵 colmat 3.48× faster than `lapack-mkl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.317489 ± 0.00237955 (W 0.00318306) | 0.485982 ± 0.00100247 (W 0.00422838) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.873719 ± 0.00860401 (W 0.00354732) | 3.86294 ± 0.0538129 (W 0.00727776) | 6.57513 ± 0.021345 (W 0.00529775) | 🔵 colmat 4.42× faster than `faer` | 🔵 colmat 7.53× faster than `lapack-mkl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.29453 ± 0.00886676 (W 0.00231581) | 3.84807 ± 0.0201193 (W 0.0096322) | - | 🔵 colmat 2.97× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 4.10051 ± 0.00588227 (W 0.00868293) | 15.317 ± 0.0149323 (W 0.0115122) | 48.9298 ± 0.0765261 (W 0.094006) | 🔵 colmat 3.74× faster than `faer` | 🔵 colmat 11.93× faster than `lapack-mkl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 5.71893 ± 0.0150034 (W 0.00429284) | 14.6192 ± 0.0153051 (W 0.0195486) | - | 🔵 colmat 2.56× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.099237 ± 0.000483068 (W 0.000796921) | 0.122066 ± 0.00067532 (W 0.00202576) | 0.35257 ± 0.00174124 (W 0.00303715) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 3.55× faster than `lapack-mkl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.099517 ± 0.000811562 (W 0.000536824) | 0.122257 ± 0.000872821 (W 0.00145895) | 0.174255 ± 0.000762483 (W 0.00106852) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-mkl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.444626 ± 0.00219675 (W 0.00225355) | 0.5564 ± 0.0270163 (W 0.0209062) | 2.59166 ± 0.0161753 (W 0.00659483) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 5.83× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.439806 ± 0.00752829 (W 0.00365694) | 0.55908 ± 0.0286954 (W 0.0272714) | 0.976465 ± 0.00344932 (W 0.00457439) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 2.22× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.78861 ± 0.00218889 (W 0.0131948) | 2.76086 ± 0.540181 (W 0.489929) | 13.4887 ± 0.0596963 (W 0.0225961) | · | 🔵 colmat 4.84× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.79239 ± 0.00711344 (W 0.0138053) | 3.66784 ± 0.527473 (W 0.423275) | 5.44611 ± 0.00621584 (W 0.0183761) | 🔵 colmat 1.31× faster than `faer` (marginal: spread 0.61 of the difference) | 🔵 colmat 1.95× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/n128/b1` | 0.094466 ± 0.000630781 (W 0.000477702) | 0.137487 ± 0.00080614 (W 0.00104653) | 0.284099 ± 0.000692786 (W 0.00175056) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 3.01× faster than `lapack-mkl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.182585 ± 0.000657836 (W 0.00177566) | 0.147915 ± 0.000805827 (W 0.00188475) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.36616 ± 0.00107621 (W 0.00146265) | 0.683461 ± 0.00568091 (W 0.00858763) | 0.37597 ± 0.00106906 (W 0.00314396) | 🔵 colmat 1.87× faster than `faer` | · |  |
| `lu/n256/b1 [api-inclusive]` | 0.799475 ± 0.00223726 (W 0.0028926) | 0.736786 ± 0.00598157 (W 0.00955156) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.40513 ± 0.000844467 (W 0.00358558) | 2.42423 ± 0.00719161 (W 0.0198997) | 2.07544 ± 0.00485067 (W 0.00526061) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 1.48× faster than `lapack-mkl` |  |
| `lu/n512/b1 [api-inclusive]` | 3.20289 ± 0.0180717 (W 0.00724062) | 2.60247 ± 0.0119726 (W 0.0171387) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.335378 ± 0.000654279 (W 0.00205239) | 0.644779 ± 0.0015221 (W 0.0028194) | 0.366669 ± 0.00168603 (W 0.00272042) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 1.09× faster than `lapack-mkl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.495505 ± 0.00477701 (W 0.00369958) | 0.774733 ± 0.00734655 (W 0.00373312) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.09315 ± 0.00407467 (W 0.00194606) | 2.15564 ± 0.00696957 (W 0.0107371) | 1.88212 ± 0.00448462 (W 0.00574677) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-mkl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.7782 ± 0.00398163 (W 0.00487089) | 2.26895 ± 0.298326 (W 0.157246) | - | 🔵 colmat 1.28× faster than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 4.42751 ± 0.00484645 (W 0.00632012) | 8.20711 ± 0.00983267 (W 0.0639073) | 8.80434 ± 0.0236401 (W 0.0105944) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 1.99× faster than `lapack-mkl` |  |
| `qr/n512/b1 [api-inclusive]` | 7.1218 ± 0.0179489 (W 0.0179499) | 10.0903 ± 0.974723 (W 0.317122) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.235203 ± 0.000718862 (W 0.00146) | 0.476824 ± 0.00326975 (W 0.00263159) | 0.191333 ± 0.000684408 (W 0.0012365) | 🔵 colmat 2.03× faster than `faer` | 🔴 colmat 1.23× slower than `lapack-mkl` |  |
| `qr_factor/n256/b1` | 0.751738 ± 0.00399426 (W 0.00300338) | 1.57827 ± 0.00128225 (W 0.00870727) | 0.917731 ± 0.00155406 (W 0.00266919) | 🔵 colmat 2.10× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-mkl` |  |
| `qr_factor/n512/b1` | 2.36397 ± 0.0177301 (W 0.00256711) | 5.11744 ± 0.00834895 (W 0.0377254) | 3.51717 ± 0.0287467 (W 0.0107974) | 🔵 colmat 2.16× faster than `faer` | 🔵 colmat 1.49× faster than `lapack-mkl` |  |
| `solve/n128/b1` | 0.094398 ± 0.000445317 (W 0.000409834) | 0.130496 ± 0.000337194 (W 0.00125729) | 0.275372 ± 0.00055658 (W 0.00195506) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 2.92× faster than `lapack-mkl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.180504 ± 0.00140775 (W 0.00223863) | 0.142916 ± 0.000774464 (W 0.00122301) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.356667 ± 0.00115077 (W 0.00267439) | 0.609873 ± 0.00337886 (W 0.00855078) | 0.357289 ± 0.00230534 (W 0.00215271) | 🔵 colmat 1.71× faster than `faer` | · |  |
| `solve/n256/b1 [api-inclusive]` | 0.427055 ± 0.00177738 (W 0.00233218) | 0.645572 ± 0.00376995 (W 0.0086751) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.36984 ± 0.00366223 (W 0.00359556) | 2.25809 ± 0.0132526 (W 0.0163083) | 1.93516 ± 0.011297 (W 0.00428583) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-mkl` |  |
| `solve/n512/b1 [api-inclusive]` | 3.20563 ± 0.020694 (W 0.00786598) | 2.47107 ± 0.00640755 (W 0.0086659) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.59312 ± 0.00543371 (W 0.00431502) | 2.59639 ± 0.005679 (W 0.00653487) | 2.30272 ± 0.00123836 (W 0.00502168) | 🔵 colmat 1.63× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-mkl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.59362 ± 0.00118163 (W 0.00453494) | 2.61315 ± 0.00557297 (W 0.00845473) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 6.21253 ± 0.012966 (W 0.0128993) | 13.4852 ± 0.0154012 (W 0.0223356) | 11.8581 ± 0.0413953 (W 0.0287044) | 🔵 colmat 2.17× faster than `faer` | 🔵 colmat 1.91× faster than `lapack-mkl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 6.26452 ± 0.018455 (W 0.00604465) | 13.4536 ± 0.0938304 (W 0.0864432) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 27.3181 ± 0.0511568 (W 0.0742096) | 45.556 ± 0.361681 (W 0.420262) | 62.6768 ± 4.16497 (W 1.29162) | 🔵 colmat 1.67× faster than `faer` | 🔵 colmat 2.29× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 27.4635 ± 0.0223266 (W 0.0519106) | 45.2888 ± 0.501181 (W 0.367046) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.827914 ± 0.00309137 (W 0.00223269) | 1.989 ± 0.591529 (W 0.00885225) | 1.20612 ± 0.00393591 (W 0.00319303) | 🔵 colmat 2.40× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/n128/b1 [api-inclusive]` | 0.824727 ± 0.0026394 (W 0.00370452) | 1.99064 ± 0.00360186 (W 0.00536048) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.5334 ± 0.00517364 (W 0.00588501) | 10.8578 ± 0.0771288 (W 0.0187224) | 6.70748 ± 0.0833893 (W 0.00756964) | 🔵 colmat 3.07× faster than `faer` | 🔵 colmat 1.90× faster than `lapack-mkl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.52039 ± 0.00738234 (W 0.00551238) | 10.9115 ± 0.0170491 (W 0.0187504) | - | 🔵 colmat 3.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.4639 ± 0.0450143 (W 0.0252119) | 34.7473 ± 0.451123 (W 0.183132) | 31.5381 ± 0.1615 (W 0.0434914) | 🔵 colmat 2.25× faster than `faer` | 🔵 colmat 2.04× faster than `lapack-mkl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 15.475 ± 0.112282 (W 0.0350813) | 33.9913 ± 0.0546095 (W 0.0554335) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00546 ± 4.04145e-05 (W 2.72984e-05) | 0.00537 ± 1.73205e-05 (W 3.49285e-05) | 0.00505 ± 5.7735e-06 (W 1.51063e-05) | · | 🔴 colmat 1.08× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01399 ± 0.000245147 (W 9.18765e-05) | 0.016539 ± 0.000390265 (W 5.82349e-05) | 0.01763 ± 0.000115846 (W 3.56973e-05) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.036329 ± 0.000197654 (W 0.000348883) | 0.043769 ± 0.000508813 (W 0.000343104) | 0.065908 ± 0.000242487 (W 7.41498e-05) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-mkl` |  |

## linalg-mid, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.025699 ± 6.55515e-05 (W 3.26067e-05) | 0.174956 ± 7.99521e-05 (W 0.000786149) | - | 🔵 colmat 6.81× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.02345 ± 4.78226e-05 (W 5.56893e-05) | 0.02305 ± 0.000157788 (W 7.01227e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.110817 ± 0.000144627 (W 0.000349563) | 0.405929 ± 0.000381687 (W 0.00113168) | - | 🔵 colmat 3.66× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.122997 ± 0.00143723 (W 0.00260843) | 0.122846 ± 0.000956436 (W 0.00159455) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.674393 ± 0.000780031 (W 0.00201344) | 1.21055 ± 0.0040815 (W 0.0026352) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.631812 ± 0.00477942 (W 0.00169524) | 0.768057 ± 0.0576154 (W 0.0015762) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 0.889622 ± 0.000254767 (W 0.00459554) | 1.08383 ± 0.00326265 (W 0.00124287) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.890276 ± 0.0010973 (W 0.00188239) | 1.08469 ± 0.00309578 (W 0.00123625) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.53924 ± 0.0142529 (W 0.00373825) | 3.86024 ± 0.0149978 (W 0.00599638) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.53979 ± 0.00432173 (W 0.0165105) | 3.86147 ± 0.0166105 (W 0.00989658) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.1562 ± 0.0833827 (W 0.0190852) | 16.7194 ± 0.0359265 (W 0.0198919) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.2306 ± 0.0973954 (W 0.0185123) | 16.7759 ± 0.055284 (W 0.012209) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.168735 ± 0.00074936 (W 0.00115602) | 0.128166 ± 0.000168902 (W 0.00102312) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.152095 ± 0.000608362 (W 0.00105694) | 0.133636 ± 0.000691159 (W 0.00112795) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.873278 ± 0.00894788 (W 0.00399812) | 0.867052 ± 0.0068819 (W 0.00632035) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.879542 ± 0.00425316 (W 0.00145887) | 0.911476 ± 0.00043337 (W 0.0048045) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 5.51329 ± 0.0365461 (W 0.0280192) | 8.673 ± 0.0152801 (W 0.00864082) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 5.60997 ± 0.0378217 (W 0.0254139) | 8.62176 ± 0.200144 (W 0.0128315) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.072738 ± 0.000740291 (W 0.000193402) | 0.070328 ± 0.000299144 (W 0.00104281) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.072478 ± 0.00060213 (W 9.83362e-05) | 0.069627 ± 7.55932e-05 (W 0.000120706) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.435471 ± 0.00511179 (W 0.00169822) | 0.429139 ± 0.00506761 (W 0.001277) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.434761 ± 0.00699238 (W 0.00152739) | 0.436528 ± 0.0028782 (W 0.00227175) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.06711 ± 0.0503145 (W 0.00563331) | 3.0821 ± 0.00488812 (W 0.00534613) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.06949 ± 0.0621346 (W 0.00448478) | 3.06668 ± 0.00404812 (W 0.00688445) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.045908 ± 9.34897e-05 (W 9.17606e-05) | 0.097278 ± 0.000270501 (W 0.000834904) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.060848 ± 0.000556801 (W 0.000464223) | 0.055388 ± 0.000376404 (W 0.000252336) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.197971 ± 0.00120018 (W 0.00118127) | 0.940768 ± 0.00406178 (W 0.0047205) | - | 🔵 colmat 4.75× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.335672 ± 0.000367261 (W 0.002637) | 0.36119 ± 0.00174143 (W 0.00173143) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.00263 ± 0.00159254 (W 0.000881601) | 6.94123 ± 0.0382863 (W 0.00823058) | - | 🔵 colmat 6.92× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.71988 ± 0.00595047 (W 0.0046309) | 2.30561 ± 0.0264242 (W 0.00526743) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.214294 ± 0.00117027 (W 0.00164525) | 0.386089 ± 0.00105497 (W 0.00148436) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.221115 ± 0.000447263 (W 0.00245829) | 0.292463 ± 0.000512204 (W 0.00179361) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.986172 ± 0.00427701 (W 0.00226302) | 1.42702 ± 0.00363529 (W 0.0025669) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.03915 ± 0.00412175 (W 0.00225144) | 1.11379 ± 0.00624857 (W 0.00185624) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.3691 ± 0.0253203 (W 0.00789838) | 9.17692 ± 0.0457156 (W 0.0114129) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 5.40943 ± 0.0108517 (W 0.00592103) | 6.40889 ± 0.0533368 (W 0.0144779) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.115486 ± 0.000695517 (W 0.000986526) | 0.285942 ± 0.000153575 (W 0.00156242) | - | 🔵 colmat 2.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.623652 ± 0.000743596 (W 0.00118302) | 1.09025 ± 0.00203401 (W 0.00174797) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.77822 ± 0.0101065 (W 0.00331204) | 6.74111 ± 0.0972148 (W 0.0104206) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.041289 ± 0.000436589 (W 0.000123572) | 0.092407 ± 0.00196128 (W 0.000785082) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.061228 ± 0.000327159 (W 0.000151239) | 0.055668 ± 3.95011e-05 (W 0.000150742) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.174525 ± 0.00106857 (W 0.000463116) | 0.513518 ± 0.00843232 (W 0.0187167) | - | 🔵 colmat 2.94× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.261013 ± 0.000733288 (W 0.00161016) | 0.284987 ± 0.00054181 (W 0.00128423) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.909944 ± 0.00669598 (W 0.00243703) | 5.14905 ± 0.00649364 (W 0.0062339) | - | 🔵 colmat 5.66× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.33024 ± 0.00105686 (W 0.00282799) | 1.98363 ± 0.14891 (W 0.00651096) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 1.3325 ± 0.00397392 (W 0.00252455) | 1.47404 ± 0.0161753 (W 0.00224713) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.33387 ± 0.0029608 (W 0.00219232) | 1.4786 ± 0.0159831 (W 0.00314672) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.41004 ± 0.0170539 (W 0.00499998) | 6.83828 ± 0.0300578 (W 0.011413) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.41973 ± 0.00291966 (W 0.0107162) | 6.8584 ± 0.019445 (W 0.00646095) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 28.2409 ± 0.103802 (W 0.0288466) | 36.8275 ± 0.128965 (W 0.0210914) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 28.3015 ± 0.0104501 (W 0.0288348) | 36.8375 ± 0.337341 (W 0.0217511) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.436991 ± 0.000400752 (W 0.00128378) | 1.12354 ± 0.000497739 (W 0.000761928) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.436508 ± 0.000275355 (W 0.00137123) | 1.12592 ± 0.000794538 (W 0.00186126) | - | 🔵 colmat 2.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.09402 ± 0.00282202 (W 0.00327311) | 5.1336 ± 0.0093795 (W 0.0046784) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.11875 ± 0.00162563 (W 0.00335262) | 5.16868 ± 0.0139447 (W 0.00300668) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 12.8503 ± 0.105466 (W 0.022722) | 26.9178 ± 0.0201535 (W 0.0149726) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 12.9481 ± 0.144869 (W 0.010462) | 27.129 ± 0.266519 (W 0.0146959) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0023 ± 2.51661e-05 (W 1.67332e-05) | 0.00215 ± 1.1547e-05 (W 1.31985e-05) | 0.00131 ± 5.7735e-06 (W 1.73205e-05) | 🔴 colmat 1.07× slower than `faer` | 🔴 colmat 1.76× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.006139 ± 2.36714e-05 (W 2.53022e-05) | 0.005929 ± 5.00033e-05 (W 3.08577e-05) | 0.00335 ± 1.09697e-05 (W 2.6119e-05) | · | 🔴 colmat 1.83× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.01671 ± 0.0001095 (W 4.70021e-05) | 0.016199 ± 5.2915e-05 (W 9.77921e-05) | 0.01028 ± 0.0001195 (W 5.2984e-05) | · | 🔴 colmat 1.63× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-mid, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.025419 ± 0.000147309 (W 0.000120291) | 0.131086 ± 0.0298869 (W 0.0438734) | - | 🔵 colmat 5.16× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.025229 ± 0.000129642 (W 0.000274613) | 0.031439 ± 0.000179258 (W 0.000762365) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.086157 ± 0.000150997 (W 0.000304467) | 0.400535 ± 0.0445399 (W 0.0351022) | - | 🔵 colmat 4.65× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.104946 ± 0.00142306 (W 0.00127655) | 0.168684 ± 0.00338095 (W 0.00505158) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.369078 ± 0.00216594 (W 0.00238519) | 0.969293 ± 0.0198868 (W 0.0391832) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.401029 ± 0.00393405 (W 0.00239357) | 0.676508 ± 0.00931475 (W 0.0271053) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.427506 ± 0.00113141 (W 0.00187491) | 1.04224 ± 0.0234519 (W 0.00477315) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.427366 ± 0.00145278 (W 0.0018163) | 1.03945 ± 0.00720568 (W 0.00238954) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.47735 ± 0.0035194 (W 0.00383024) | 2.75137 ± 0.00446397 (W 0.00978901) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 1.48415 ± 0.00588934 (W 0.00397978) | 2.75585 ± 0.00640221 (W 0.0137327) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 5.93656 ± 0.01902 (W 0.0121336) | 9.34499 ± 0.0614204 (W 0.0264784) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 5.94865 ± 0.0212529 (W 0.0121676) | 9.31059 ± 0.0348502 (W 0.0202539) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.137615 ± 0.000959771 (W 0.00056401) | 0.129325 ± 0.00129332 (W 0.00131197) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.147955 ± 0.0015827 (W 0.000770733) | 0.133985 ± 0.00152963 (W 0.00167402) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.519791 ± 0.0041243 (W 0.00274933) | 0.864857 ± 0.00233774 (W 0.00610195) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.621863 ± 0.00345919 (W 0.00346221) | 0.837772 ± 0.00181214 (W 0.00589346) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 2.08559 ± 0.0101013 (W 0.00535222) | 6.76398 ± 0.0268375 (W 0.0117002) | - | 🔵 colmat 3.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 2.63002 ± 0.00594769 (W 0.00685266) | 6.56829 ± 0.164374 (W 0.0118509) | - | 🔵 colmat 2.50× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.030969 ± 0.000485695 (W 0.000873224) | 0.02848 ± 0.00050049 (W 0.000485088) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.031009 ± 0.00022745 (W 0.00113729) | 0.027219 ± 9.21973e-05 (W 0.000307132) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.134565 ± 0.00212776 (W 0.000982726) | 0.134715 ± 0.00102249 (W 0.00174286) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.134595 ± 0.00138873 (W 0.000816204) | 0.133055 ± 0.00153366 (W 0.000973349) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 0.855267 ± 0.00151803 (W 0.00322272) | 0.850142 ± 0.00340737 (W 0.00273857) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 0.850148 ± 0.00430737 (W 0.00541416) | 0.852546 ± 0.00684684 (W 0.00197499) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.061358 ± 0.000420702 (W 0.00141226) | 0.115176 ± 0.000335112 (W 0.00100842) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.057908 ± 0.000469361 (W 0.000412922) | 0.075967 ± 0.000332296 (W 0.000858697) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.173485 ± 0.000744077 (W 0.00109085) | 0.985643 ± 0.0123592 (W 0.0220903) | - | 🔵 colmat 5.68× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.281692 ± 0.000578692 (W 0.00203782) | 0.371696 ± 0.00279544 (W 0.00491137) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.676132 ± 0.00415769 (W 0.00442611) | 5.29502 ± 0.114493 (W 0.190038) | - | 🔵 colmat 7.83× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.2101 ± 0.0023381 (W 0.00427794) | 2.07344 ± 0.158928 (W 0.0358879) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.168154 ± 0.000425077 (W 0.00156665) | 0.498423 ± 0.00119608 (W 0.00244784) | - | 🔵 colmat 2.96× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.174133 ± 0.00190691 (W 0.0010767) | 0.362337 ± 0.000714431 (W 0.0021073) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.481612 ± 0.00283697 (W 0.00206085) | 1.63922 ± 0.00855365 (W 0.0150835) | - | 🔵 colmat 3.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.657942 ± 0.00257724 (W 0.00201312) | 1.10084 ± 0.0199859 (W 0.00520456) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.02631 ± 0.0118056 (W 0.00332024) | 8.2648 ± 0.0673343 (W 0.0260712) | - | 🔵 colmat 4.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 2.71491 ± 0.00878102 (W 0.00631634) | 4.26527 ± 0.20603 (W 0.0131313) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.092677 ± 0.000461988 (W 0.000641208) | 0.425094 ± 0.00117095 (W 0.00425291) | - | 🔵 colmat 4.59× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.356578 ± 0.00143665 (W 0.00191067) | 1.53234 ± 0.00719494 (W 0.0124034) | - | 🔵 colmat 4.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.2836 ± 0.00858683 (W 0.00354721) | 7.46616 ± 0.0348214 (W 0.0290988) | - | 🔵 colmat 5.82× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.043249 ± 0.000252726 (W 0.000279377) | 0.108456 ± 0.000326538 (W 0.00115852) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.065467 ± 0.000510569 (W 0.000358897) | 0.073597 ± 0.000487419 (W 0.00113832) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.166715 ± 0.00122355 (W 0.000836811) | 0.669615 ± 0.00787065 (W 0.013298) | - | 🔵 colmat 4.02× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.282169 ± 0.00101315 (W 0.00244926) | 0.36006 ± 0.00104053 (W 0.00689762) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.661885 ± 0.00547648 (W 0.00434046) | 3.97967 ± 0.0760336 (W 0.0941683) | - | 🔵 colmat 6.01× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.19447 ± 0.00592259 (W 0.00508098) | 1.8139 ± 0.0027877 (W 0.0394538) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 0.957918 ± 0.00180781 (W 0.00277902) | 1.17724 ± 0.00557255 (W 0.00298465) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 0.962284 ± 0.00151775 (W 0.00228116) | 1.18271 ± 0.00664803 (W 0.0131078) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 3.60146 ± 0.00906695 (W 0.00723618) | 5.33965 ± 0.017326 (W 0.0246035) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 3.60221 ± 0.0076764 (W 0.00685736) | 5.35434 ± 0.0456114 (W 0.0219665) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 14.8784 ± 0.0560052 (W 0.0160095) | 22.6099 ± 0.113345 (W 0.0388738) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 14.9925 ± 0.297966 (W 0.0191185) | 23.1218 ± 0.332083 (W 0.0266429) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.459395 ± 0.00212913 (W 0.00179563) | 0.948974 ± 0.000973656 (W 0.0025308) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.457933 ± 0.0048049 (W 0.00183989) | 0.951118 ± 0.00284141 (W 0.00112184) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 1.93279 ± 0.0018899 (W 0.00583379) | 4.34653 ± 0.000277448 (W 0.00297677) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 1.93173 ± 0.00186249 (W 0.00816229) | 4.37203 ± 0.0059009 (W 0.00812159) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 8.03806 ± 0.0102348 (W 0.0100968) | 18.7204 ± 0.0801439 (W 0.0320788) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 8.08403 ± 0.0131335 (W 0.0105896) | 19.0794 ± 0.308663 (W 0.0403536) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00286 ± 1.52753e-05 (W 2.3156e-05) | 0.00217 ± 1e-05 (W 2.19591e-05) | 0.00131 ± 5.7735e-06 (W 2.40666e-05) | 🔴 colmat 1.32× slower than `faer` | 🔴 colmat 2.18× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.00725 ± 4.45758e-05 (W 4.46677e-05) | 0.00586 ± 3.21455e-05 (W 4.07456e-05) | 0.00335 ± 5.7735e-06 (W 2.05475e-05) | 🔴 colmat 1.24× slower than `faer` | 🔴 colmat 2.16× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.01664 ± 0.000374869 (W 7.84685e-05) | 0.01378 ± 2.08167e-05 (W 0.000120811) | 0.01056 ± 0.000269735 (W 6.07149e-05) | 🔴 colmat 1.21× slower than `faer` | 🔴 colmat 1.58× slower than `lapack-mkl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-mid, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.025639 ± 0.00011 (W 0.000150796) | 0.081338 ± 0.000450622 (W 0.000774196) | - | 🔵 colmat 3.17× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.024819 ± 0.000251933 (W 0.000194039) | 0.030519 ± 0.000155027 (W 0.000417427) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.084698 ± 0.000610243 (W 0.000546516) | 0.269771 ± 0.0269713 (W 0.0442125) | - | 🔵 colmat 3.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.105216 ± 0.00135755 (W 0.00241462) | 0.181535 ± 0.00167719 (W 0.00718519) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.341008 ± 0.00251306 (W 0.00240571) | 0.961416 ± 0.0184366 (W 0.0512431) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.363267 ± 0.00053858 (W 0.00229972) | 0.783088 ± 0.0284057 (W 0.0187521) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 0.360209 ± 0.00211292 (W 0.00214986) | 1.0977 ± 0.00166739 (W 0.00542269) | - | 🔵 colmat 3.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.359889 ± 0.0011729 (W 0.00166719) | 1.10095 ± 0.00377255 (W 0.00396852) | - | 🔵 colmat 3.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.18881 ± 0.00210558 (W 0.00333304) | 2.86516 ± 0.0061637 (W 0.00672504) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 1.19428 ± 0.00976422 (W 0.00353249) | 2.88561 ± 0.029278 (W 0.0139171) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 4.44705 ± 0.0306153 (W 0.00696995) | 9.39038 ± 0.0846254 (W 0.022063) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 4.43803 ± 0.00436083 (W 0.00857375) | 9.2761 ± 0.00760348 (W 0.0415494) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.136865 ± 0.00105713 (W 0.000505589) | 0.128865 ± 0.000372138 (W 0.000387335) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.150776 ± 0.00102027 (W 0.00153511) | 0.132586 ± 0.00149303 (W 0.00108941) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.463683 ± 0.00463218 (W 0.00346051) | 0.859606 ± 0.00606797 (W 0.00780775) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.61357 ± 0.00258156 (W 0.00331443) | 0.84167 ± 0.00288228 (W 0.00607703) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 1.67429 ± 0.00827191 (W 0.00666638) | 6.45357 ± 0.00394291 (W 0.0143448) | - | 🔵 colmat 3.85× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 2.29019 ± 0.0101305 (W 0.00424464) | 6.02465 ± 0.127227 (W 0.0131986) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.025309 ± 0.00042194 (W 0.000711182) | 0.042428 ± 0.000223227 (W 0.00049523) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.026329 ± 0.000385011 (W 0.000670375) | 0.041469 ± 0.000134536 (W 0.000994355) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.088757 ± 0.0012822 (W 0.000800668) | 0.173465 ± 0.0017189 (W 0.00491119) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.090587 ± 0.000502096 (W 0.00170737) | 0.174005 ± 0.00443035 (W 0.00469838) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 0.476516 ± 0.00577258 (W 0.00292297) | 0.692796 ± 0.068909 (W 0.0634119) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 0.481785 ± 0.0018293 (W 0.00543354) | 0.688259 ± 0.0372904 (W 0.0666982) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.068968 ± 0.000888426 (W 0.00153829) | 0.114886 ± 0.000740027 (W 0.000675226) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.058968 ± 0.000545008 (W 0.000660987) | 0.077977 ± 0.00176086 (W 0.000659244) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.140606 ± 0.000649993 (W 0.00155461) | 0.97108 ± 0.00431191 (W 0.0180037) | - | 🔵 colmat 6.91× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.275111 ± 0.000726485 (W 0.00222775) | 0.374198 ± 0.00181159 (W 0.00422014) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.508804 ± 0.000488213 (W 0.00387615) | 4.88125 ± 0.077766 (W 0.0906224) | - | 🔵 colmat 9.59× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.11221 ± 0.00155135 (W 0.00530445) | 2.27938 ± 0.236319 (W 0.066275) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.164265 ± 0.000681389 (W 0.00133896) | 0.625113 ± 0.00769825 (W 0.00414364) | - | 🔵 colmat 3.81× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.170244 ± 0.00055523 (W 0.00166273) | 0.502274 ± 0.00207378 (W 0.00613099) | - | 🔵 colmat 2.95× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.431064 ± 0.00101578 (W 0.00196168) | 1.98136 ± 0.00630352 (W 0.0115071) | - | 🔵 colmat 4.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.626898 ± 0.00535128 (W 0.00448378) | 1.29791 ± 0.00117104 (W 0.00582868) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 1.37828 ± 0.00666404 (W 0.00990419) | 8.81724 ± 0.0342721 (W 0.0192734) | - | 🔵 colmat 6.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 2.20842 ± 0.0132631 (W 0.00647407) | 4.40132 ± 0.178471 (W 0.0101071) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.088967 ± 0.000407717 (W 0.000497668) | 0.512575 ± 0.00363911 (W 0.00641263) | - | 🔵 colmat 5.76× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.33078 ± 0.00133264 (W 0.00149559) | 1.75098 ± 0.00686086 (W 0.0165052) | - | 🔵 colmat 5.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 0.953801 ± 0.000848632 (W 0.00245234) | 7.75768 ± 0.0255146 (W 0.0199573) | - | 🔵 colmat 8.13× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.047188 ± 0.000326129 (W 0.000855283) | 0.105057 ± 0.000381406 (W 0.000881504) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.061689 ± 5.5109e-05 (W 0.000420835) | 0.072698 ± 0.000515325 (W 0.000982305) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.139496 ± 0.00103831 (W 0.00085833) | 0.69763 ± 0.0227662 (W 0.0048631) | - | 🔵 colmat 5.00× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.275372 ± 0.000764516 (W 0.00173833) | 0.368346 ± 0.00206554 (W 0.00619789) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.515654 ± 0.0118962 (W 0.003868) | 3.90474 ± 0.0496529 (W 0.0647985) | - | 🔵 colmat 7.57× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.10567 ± 0.00611539 (W 0.00411018) | 1.80981 ± 0.120223 (W 0.066155) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n128/b1` | 0.96211 ± 0.00244985 (W 0.00199533) | 1.28736 ± 0.00111049 (W 0.00991604) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 0.96877 ± 0.00368173 (W 0.00195659) | 1.29216 ± 0.00306144 (W 0.00773721) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 3.42992 ± 0.00378232 (W 0.00732949) | 5.58171 ± 0.0131878 (W 0.0126148) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 3.43263 ± 0.00549866 (W 0.0078428) | 5.59935 ± 0.00260969 (W 0.0303049) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 13.2703 ± 0.138974 (W 0.0289119) | 22.3774 ± 0.344193 (W 0.619969) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 13.3458 ± 0.205076 (W 0.0333564) | 22.7878 ± 0.349955 (W 0.0879452) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.471665 ± 0.0013635 (W 0.00194146) | 0.950856 ± 0.000680333 (W 0.00382625) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.470807 ± 0.00208735 (W 0.00185215) | 0.950586 ± 0.00239526 (W 0.00126866) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 1.93319 ± 0.00155889 (W 0.00699743) | 4.39351 ± 0.00508176 (W 0.00393195) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 1.93221 ± 0.00268282 (W 0.00526021) | 4.443 ± 0.0277976 (W 0.00739708) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 7.46547 ± 0.0257159 (W 0.00942064) | 18.2127 ± 0.1937 (W 0.0237514) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 7.48894 ± 0.0321694 (W 0.0136934) | 18.5419 ± 0.314948 (W 0.0394459) | - | 🔵 colmat 2.48× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00284 ± 2.25167e-05 (W 2.82843e-05) | 0.00216 ± 0 (W 2.91582e-05) | 0.0013 ± 5.7735e-06 (W 2.16795e-05) | 🔴 colmat 1.31× slower than `faer` | 🔴 colmat 2.18× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.00725 ± 4.56107e-05 (W 2.51436e-05) | 0.00587 ± 4.16333e-05 (W 3.1305e-05) | 0.00335 ± 5.7735e-06 (W 2.39625e-05) | 🔴 colmat 1.24× slower than `faer` | 🔴 colmat 2.16× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.016139 ± 1.76163e-05 (W 7.1877e-05) | 0.01356 ± 0.000281952 (W 0.000154759) | 0.01047 ± 0.00010116 (W 7.64284e-05) | 🔴 colmat 1.19× slower than `faer` | 🔴 colmat 1.54× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-mid, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.01963 ± 7.0406e-05 (W 7.05429e-05) | 0.01898 ± 5.64535e-05 (W 2.81478e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n128/b1 [api-inclusive]` | 0.023019 ± 1.73205e-05 (W 7.35955e-05) | 0.022279 ± 9.16515e-05 (W 7.21339e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.085756 ± 0.000295412 (W 7.01427e-05) | 0.086588 ± 0.000370783 (W 8.08084e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.099427 ± 0.000609738 (W 0.000142497) | 0.098236 ± 0.00045235 (W 8.03443e-05) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.46 ± 0.00119579 (W 0.00144949) | 0.477419 ± 0.000423878 (W 0.00145391) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.522886 ± 0.00248242 (W 0.00170322) | 0.522595 ± 0.00484487 (W 0.00210206) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.884413 ± 0.00409045 (W 0.00486738) | 1.08156 ± 0.00340944 (W 0.00114579) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.884403 ± 0.00294022 (W 0.000906954) | 1.08108 ± 0.00104843 (W 0.00166822) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.50899 ± 0.00455387 (W 0.00790504) | 3.83172 ± 0.0548232 (W 0.00615475) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.51818 ± 0.0046978 (W 0.0151282) | 3.84222 ± 0.00761314 (W 0.00563852) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.1172 ± 0.107424 (W 0.012133) | 16.5616 ± 0.157306 (W 0.0111161) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.0515 ± 0.0687653 (W 0.0188531) | 16.5366 ± 0.148125 (W 0.0196335) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.130647 ± 0.000233964 (W 0.00108745) | 0.126975 ± 0.000374999 (W 0.000988059) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.135707 ± 0.00058463 (W 0.00131821) | 0.128107 ± 0.00027886 (W 0.00131274) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.77658 ± 0.00159462 (W 0.00160922) | 0.813207 ± 0.000916326 (W 0.00312178) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.819349 ± 0.00133526 (W 0.00123133) | 0.836938 ± 0.00212832 (W 0.0042088) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 5.35386 ± 0.0284833 (W 0.00721845) | 7.77905 ± 0.0225071 (W 0.0147729) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 5.22728 ± 0.0289517 (W 0.0228615) | 7.70585 ± 0.105001 (W 0.0386714) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.076278 ± 0.000113227 (W 0.00110615) | 0.074687 ± 0.000263034 (W 0.000147817) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.076526 ± 0.000130082 (W 0.00012737) | 0.073827 ± 5.5109e-05 (W 0.000149666) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.453538 ± 0.00390003 (W 0.00137773) | 0.450269 ± 0.00349829 (W 0.00123064) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.455288 ± 0.00483112 (W 0.00145771) | 0.451788 ± 0.00344105 (W 0.0020082) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.22176 ± 0.0127658 (W 0.00652145) | 3.22232 ± 0.0169108 (W 0.00360866) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.21761 ± 0.00742886 (W 0.0081463) | 3.2256 ± 0.01267 (W 0.00478152) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.042079 ± 0.000295635 (W 0.000275326) | 0.046058 ± 8.5967e-05 (W 0.000217256) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.046499 ± 0.000349343 (W 0.000891801) | 0.050538 ± 0.000216002 (W 0.000215498) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.191991 ± 0.000598159 (W 0.00122324) | 0.202591 ± 0.00174684 (W 0.00135795) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.273342 ± 0.00041265 (W 0.00265421) | 0.292082 ± 0.00255 (W 0.00182778) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.04218 ± 0.003828 (W 0.00126934) | 1.25848 ± 0.00103701 (W 0.00366452) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.42983 ± 0.00145641 (W 0.0029359) | 1.65957 ± 0.00230758 (W 0.00487272) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.182182 ± 0.000701382 (W 0.00101183) | 0.260882 ± 0.00035032 (W 0.00113972) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.204025 ± 0.00024752 (W 0.00193852) | 0.281438 ± 0.00070653 (W 0.00247954) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.841894 ± 0.00366314 (W 0.00134072) | 0.900166 ± 0.00119141 (W 0.00223655) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.965815 ± 0.00389164 (W 0.00131255) | 1.04128 ± 0.00162529 (W 0.00177943) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 4.57495 ± 0.0110373 (W 0.00468664) | 4.79006 ± 0.0064929 (W 0.00596306) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 5.08185 ± 0.00809034 (W 0.0084979) | 5.31565 ± 0.0121096 (W 0.0113712) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.078378 ± 0.000674327 (W 0.000572314) | 0.158446 ± 0.000377747 (W 0.00111671) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.472428 ± 0.00196298 (W 0.00179756) | 0.537115 ± 0.00162623 (W 0.0014347) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.14748 ± 0.0094557 (W 0.00438349) | 2.40116 ± 0.00259931 (W 0.00415383) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.041378 ± 0.000654706 (W 0.000810562) | 0.045459 ± 0.000191398 (W 0.000125925) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.044199 ± 0.000508901 (W 0.00017985) | 0.049729 ± 0.000144535 (W 0.000182924) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.185375 ± 0.000727026 (W 0.000772418) | 0.191915 ± 0.00104564 (W 0.00107938) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.192515 ± 0.000796584 (W 0.00150165) | 0.213911 ± 0.000478795 (W 0.00154187) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.999422 ± 0.00309901 (W 0.00163352) | 1.22498 ± 0.00439173 (W 0.00328018) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.0453 ± 0.00347535 (W 0.00656204) | 1.28578 ± 0.00351417 (W 0.00146846) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.32914 ± 0.00189383 (W 0.00137127) | 1.4813 ± 0.00204809 (W 0.00178347) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.33154 ± 0.00515118 (W 0.00282111) | 1.4884 ± 0.004817 (W 0.00245697) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.3825 ± 0.0111467 (W 0.00284057) | 6.87937 ± 0.0167818 (W 0.00830773) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.39547 ± 0.0071931 (W 0.00397084) | 6.85686 ± 0.0162755 (W 0.00841423) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 27.3259 ± 0.0147671 (W 0.0178184) | 35.9575 ± 0.117057 (W 0.0280789) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 27.3692 ± 0.0374836 (W 0.0495939) | 36.1725 ± 0.124391 (W 0.0426665) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.432472 ± 0.00096278 (W 0.001373) | 1.10431 ± 0.00840758 (W 0.000998714) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.431337 ± 0.000493651 (W 0.00313984) | 1.10477 ± 0.00179099 (W 0.00147144) | - | 🔵 colmat 2.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.0183 ± 0.00831626 (W 0.00117364) | 5.05107 ± 0.0160449 (W 0.00517991) | - | 🔵 colmat 2.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.05253 ± 0.000667576 (W 0.00183153) | 5.02572 ± 0.0271719 (W 0.0106953) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 12.0772 ± 0.0409078 (W 0.0141925) | 26.0309 ± 0.0639423 (W 0.0204512) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 12.048 ± 0.040341 (W 0.0072565) | 26.0711 ± 0.0990037 (W 0.0198798) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00262 ± 1.52753e-05 (W 2.8464e-05) | 0.0026 ± 2.58908e-05 (W 3.11448e-05) | 0.00122 ± 5.7735e-06 (W 2.07364e-05) | · | 🔴 colmat 2.15× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.007249 ± 5.11892e-05 (W 3.01529e-05) | 0.007239 ± 2.3388e-05 (W 6.67847e-05) | 0.00317 ± 5.7735e-06 (W 1.67332e-05) | · | 🔴 colmat 2.29× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.01817 ± 0.000109865 (W 4.35924e-05) | 0.01783 ± 0.000163147 (W 6.14353e-05) | 0.00991 ± 2.58908e-05 (W 9.69706e-05) | · | 🔴 colmat 1.83× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-mid, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.019849 ± 0.000121655 (W 5.64827e-05) | 0.02679 ± 9.17624e-05 (W 0.00111341) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n128/b1 [api-inclusive]` | 0.024359 ± 0.000109697 (W 0.000352262) | 0.030599 ± 0.000510979 (W 0.000899989) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.068368 ± 0.000664931 (W 0.000550486) | 0.130006 ± 0.00229506 (W 0.00460941) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.081877 ± 0.000897196 (W 0.000344009) | 0.143776 ± 0.00302213 (W 0.00249542) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.230544 ± 0.00127222 (W 0.00161096) | 0.394565 ± 0.0183393 (W 0.0157472) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.293002 ± 0.000510879 (W 0.00125143) | 0.450512 ± 0.0106199 (W 0.0106409) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.426886 ± 0.000658273 (W 0.0022471) | 1.04235 ± 0.00294415 (W 0.00126975) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.427926 ± 0.000614202 (W 0.00149683) | 1.04732 ± 0.00243015 (W 0.000785729) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.45347 ± 0.00533475 (W 0.00310319) | 2.74196 ± 0.00965992 (W 0.010099) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 1.45658 ± 0.00357526 (W 0.00381503) | 2.75321 ± 0.0125913 (W 0.0115407) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 5.83421 ± 0.0216208 (W 0.0129619) | 9.20028 ± 0.0259406 (W 0.0161795) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 5.80912 ± 0.0298208 (W 0.0126213) | 9.12238 ± 0.0354963 (W 0.0156498) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.125206 ± 0.000558535 (W 0.000520605) | 0.126795 ± 0.000268924 (W 0.0011499) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.128445 ± 0.000633231 (W 0.000640328) | 0.126636 ± 0.00215583 (W 0.00176892) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.51108 ± 0.0025595 (W 0.0024705) | 0.811909 ± 0.0016096 (W 0.00322934) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.550455 ± 0.00554469 (W 0.00275171) | 0.770599 ± 0.00304222 (W 0.004433) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 2.34447 ± 0.00737184 (W 0.0040263) | 5.82998 ± 0.0100518 (W 0.0101783) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 2.3498 ± 0.0107528 (W 0.00496836) | 5.66254 ± 0.0257634 (W 0.0119079) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.031549 ± 0.000135031 (W 0.000325945) | 0.028789 ± 0.000348186 (W 0.000306545) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.031689 ± 0.000150965 (W 0.000343712) | 0.029229 ± 0.000175784 (W 0.000255636) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.143236 ± 0.000239061 (W 0.00173135) | 0.141456 ± 0.00204856 (W 0.000726799) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.143366 ± 0.00057821 (W 0.000626336) | 0.139405 ± 0.000735677 (W 0.000887614) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 0.852517 ± 0.00232996 (W 0.00467105) | 0.852416 ± 0.00574728 (W 0.00406525) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 0.848589 ± 0.0052361 (W 0.00319841) | 0.847579 ± 0.00284203 (W 0.0040784) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.038419 ± 0.000218152 (W 0.00027225) | 0.064758 ± 0.000797151 (W 0.00123946) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.042179 ± 0.000767014 (W 0.000328753) | 0.070828 ± 0.000468722 (W 0.00113834) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.124897 ± 0.00141709 (W 0.000872473) | 0.280661 ± 0.00337372 (W 0.00584423) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.212955 ± 0.000993503 (W 0.00142669) | 0.306882 ± 0.000749318 (W 0.00377937) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.52578 ± 0.00404646 (W 0.00276538) | 1.17917 ± 0.0533643 (W 0.0609838) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 0.921745 ± 0.00173138 (W 0.0023381) | 1.26757 ± 0.000656447 (W 0.0235248) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.131435 ± 0.000602263 (W 0.000480311) | 0.333269 ± 0.00230134 (W 0.00291309) | - | 🔵 colmat 2.54× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.156816 ± 0.000777934 (W 0.00171652) | 0.354358 ± 0.00229081 (W 0.00213288) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.429536 ± 0.00235204 (W 0.00193718) | 0.878931 ± 0.00123912 (W 0.00424024) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.591698 ± 0.00155019 (W 0.00389274) | 1.03155 ± 0.0224309 (W 0.00336411) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 1.79505 ± 0.00159466 (W 0.00290248) | 2.80089 ± 0.005937 (W 0.00861139) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 2.42989 ± 0.0124569 (W 0.00640915) | 3.44727 ± 0.00959183 (W 0.00870929) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.078357 ± 0.000179584 (W 0.000357393) | 0.26433 ± 0.000452233 (W 0.002285) | - | 🔵 colmat 3.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.303062 ± 0.000766197 (W 0.00120878) | 0.754901 ± 0.00292028 (W 0.00160299) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.14213 ± 0.00169198 (W 0.00240711) | 2.15425 ± 0.00234434 (W 0.00685412) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.037869 ± 0.000668474 (W 0.000165539) | 0.061048 ± 0.00087155 (W 0.000874403) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.047229 ± 0.000155597 (W 0.000310914) | 0.066608 ± 0.000524049 (W 0.000774441) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.120855 ± 0.000841178 (W 0.000770312) | 0.27092 ± 0.000920072 (W 0.00205565) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.215574 ± 0.00197917 (W 0.00175429) | 0.292582 ± 0.00169661 (W 0.00320268) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.506391 ± 0.000750579 (W 0.00284866) | 1.06856 ± 0.0155256 (W 0.0332848) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 0.908805 ± 0.00438792 (W 0.00256031) | 1.1982 ± 0.0244566 (W 0.0294472) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 0.961414 ± 0.0015218 (W 0.00244469) | 1.18827 ± 0.00491284 (W 0.00332884) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 0.959984 ± 0.00296232 (W 0.00273893) | 1.18605 ± 0.00198067 (W 0.00426724) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 3.53388 ± 0.00580315 (W 0.00479156) | 5.33401 ± 0.00616663 (W 0.0203627) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 3.53401 ± 0.0102996 (W 0.0041544) | 5.36093 ± 0.0270666 (W 0.0268348) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 14.2671 ± 0.0255777 (W 0.0258656) | 22.0073 ± 0.109907 (W 0.026843) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 14.2982 ± 0.0160719 (W 0.0244259) | 22.0881 ± 0.0275906 (W 0.0686901) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.455273 ± 0.00148152 (W 0.00240993) | 0.929875 ± 0.00143278 (W 0.0030313) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.454225 ± 0.000736562 (W 0.00252533) | 0.930569 ± 0.00107851 (W 0.00240719) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 1.86444 ± 0.00691706 (W 0.00368094) | 4.23074 ± 0.00733425 (W 0.00503574) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 1.86469 ± 0.00638796 (W 0.0045963) | 4.25075 ± 0.00857812 (W 0.0125794) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 7.49625 ± 0.00901044 (W 0.00909547) | 18.0686 ± 0.0747169 (W 0.0375946) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 7.48688 ± 0.0120495 (W 0.0118935) | 18.0893 ± 0.0371521 (W 0.0264816) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00309 ± 6.08276e-05 (W 4.02517e-05) | 0.00253 ± 1e-05 (W 1.92354e-05) | 0.00122 ± 5.7735e-06 (W 2.16795e-05) | 🔴 colmat 1.22× slower than `faer` | 🔴 colmat 2.53× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.00819 ± 3.66106e-05 (W 2.24121e-05) | 0.00702 ± 1.1547e-05 (W 2.83055e-05) | 0.00317 ± 1e-05 (W 1.51063e-05) | 🔴 colmat 1.17× slower than `faer` | 🔴 colmat 2.58× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.01983 ± 0.000343089 (W 0.000281413) | 0.01731 ± 0.000230515 (W 0.000253987) | 0.009819 ± 4.2194e-05 (W 5.52558e-05) | 🔴 colmat 1.15× slower than `faer` | 🔴 colmat 2.02× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-mid, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.019889 ± 3.56417e-05 (W 0.000176528) | 0.024059 ± 0.00131465 (W 0.00036748) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n128/b1 [api-inclusive]` | 0.024169 ± 0.000120968 (W 0.000159431) | 0.029769 ± 0.00118924 (W 0.000438543) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.068958 ± 0.000680221 (W 0.000974505) | 0.138785 ± 0.00245108 (W 0.00243072) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.082457 ± 0.000118462 (W 0.000652587) | 0.156815 ± 0.0029802 (W 0.00880403) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.203403 ± 0.0204557 (W 0.0027376) | 0.473876 ± 0.0265393 (W 0.0335747) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.261932 ± 0.000573429 (W 0.00204233) | 0.493106 ± 0.0180716 (W 0.0320953) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 0.359767 ± 0.000852175 (W 0.00227867) | 1.09289 ± 0.00131113 (W 0.0028265) | - | 🔵 colmat 3.04× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.36114 ± 0.000123662 (W 0.00245212) | 1.09596 ± 0.00367855 (W 0.00250045) | - | 🔵 colmat 3.03× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.17311 ± 0.00654328 (W 0.00524272) | 2.85433 ± 0.00391796 (W 0.0134405) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 1.1699 ± 0.00466656 (W 0.0041119) | 2.85813 ± 0.00696652 (W 0.0130748) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 4.33984 ± 0.00131868 (W 0.00729932) | 9.13552 ± 0.0359786 (W 0.0312922) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 4.33347 ± 0.0233633 (W 0.0101365) | 9.14267 ± 0.044464 (W 0.0323728) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.129155 ± 0.000426951 (W 0.000695181) | 0.129536 ± 0.00161277 (W 0.00121915) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.134646 ± 0.000885336 (W 0.000549574) | 0.126636 ± 0.000305123 (W 0.000203552) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.461464 ± 0.00543489 (W 0.00293269) | 0.811911 ± 0.00384215 (W 0.00467125) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.542921 ± 0.00592202 (W 0.00244293) | 0.768036 ± 0.00134038 (W 0.00426094) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 1.78708 ± 0.0188944 (W 0.0031178) | 5.50411 ± 0.0221235 (W 0.0142366) | - | 🔵 colmat 3.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 2.01155 ± 0.0101557 (W 0.00529445) | 5.36451 ± 0.0139155 (W 0.00837239) | - | 🔵 colmat 2.67× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.022849 ± 0.000423025 (W 0.000432296) | 0.042649 ± 0.000845005 (W 0.000742622) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.023019 ± 0.000320874 (W 0.000305177) | 0.043489 ± 0.000530283 (W 0.000851075) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.093507 ± 0.000428269 (W 0.000431208) | 0.180714 ± 0.00510825 (W 0.00499974) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.094487 ± 0.000396274 (W 0.000928141) | 0.181825 ± 0.00931564 (W 0.00524054) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 0.501424 ± 0.00406217 (W 0.0040032) | 0.616558 ± 0.0472886 (W 0.0720346) | - | 🔵 colmat 1.23× faster than `faer` (marginal: spread 0.63 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 0.502236 ± 0.00388406 (W 0.00260792) | 0.695959 ± 0.0578145 (W 0.0594728) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.037809 ± 0.000290873 (W 0.000376298) | 0.064818 ± 0.000212838 (W 0.000916301) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.041979 ± 0.00015308 (W 0.000379913) | 0.071918 ± 0.00172058 (W 0.00125773) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.117146 ± 0.00078373 (W 0.00123246) | 0.289601 ± 0.000407964 (W 0.00726566) | - | 🔵 colmat 2.47× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.208313 ± 0.000511769 (W 0.0021203) | 0.308609 ± 0.0023564 (W 0.00669734) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.425788 ± 0.000700337 (W 0.00376655) | 1.19692 ± 0.0139282 (W 0.0711406) | - | 🔵 colmat 2.81× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 0.827404 ± 0.00588452 (W 0.0031389) | 1.28175 ± 0.0300832 (W 0.0492113) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.126816 ± 0.00462777 (W 0.000454882) | 0.482255 ± 0.00214757 (W 0.00400443) | - | 🔵 colmat 3.80× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.152725 ± 0.00106111 (W 0.00130516) | 0.495594 ± 0.00509044 (W 0.00514898) | - | 🔵 colmat 3.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.395249 ± 0.00265179 (W 0.00214544) | 1.2234 ± 0.0202191 (W 0.00616421) | - | 🔵 colmat 3.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.553371 ± 0.00577955 (W 0.00257867) | 1.22773 ± 0.00598753 (W 0.011534) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 1.26294 ± 0.00135718 (W 0.00333514) | 3.4943 ± 3.04067 (W 0.011814) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 1.91714 ± 0.00320042 (W 0.00527505) | 3.5887 ± 0.0100248 (W 0.0136617) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.078547 ± 0.000601022 (W 0.000218397) | 0.368137 ± 0.00783319 (W 0.00271396) | - | 🔵 colmat 4.69× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.303811 ± 0.000668732 (W 0.00153348) | 0.972895 ± 0.00350681 (W 0.00594429) | - | 🔵 colmat 3.20× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 0.858729 ± 0.00396229 (W 0.00248709) | 2.63527 ± 0.0113876 (W 0.0102405) | - | 🔵 colmat 3.07× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.038799 ± 0.000149778 (W 0.000400585) | 0.059998 ± 0.000357911 (W 0.00048782) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.043589 ± 0.000540401 (W 0.000499287) | 0.066738 ± 4.3589e-05 (W 0.0006324) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.112117 ± 0.000772837 (W 0.00107975) | 0.274972 ± 0.00350359 (W 0.00536658) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.207474 ± 0.000171892 (W 0.00243429) | 0.292549 ± 0.00301538 (W 0.010851) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.409166 ± 0.00183222 (W 0.00346372) | 1.15302 ± 0.0477833 (W 0.0516873) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1 [api-inclusive]` | 0.819777 ± 0.00541232 (W 0.00778542) | 1.24588 ± 0.0266097 (W 0.0493392) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 0.960733 ± 0.00118565 (W 0.00213733) | 1.28966 ± 0.0019041 (W 0.00848376) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 0.96183 ± 0.00818552 (W 0.00235912) | 1.29553 ± 0.00301191 (W 0.00533024) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 3.35419 ± 0.00750561 (W 0.0086951) | 5.6003 ± 0.00537593 (W 0.0119395) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 3.36684 ± 0.0477622 (W 0.00529892) | 5.61459 ± 0.0130169 (W 0.0184406) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 12.673 ± 0.0467559 (W 0.0105855) | 21.9251 ± 0.0676268 (W 0.0443322) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 12.671 ± 0.00771675 (W 0.0397209) | 21.8804 ± 0.0772653 (W 0.0369054) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.468975 ± 0.00101376 (W 0.00286502) | 0.930491 ± 0.00106208 (W 0.00250985) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.468043 ± 0.000859705 (W 0.00216791) | 0.929787 ± 0.00370986 (W 0.00159797) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 1.85993 ± 0.00396397 (W 0.00262208) | 4.28063 ± 0.00800948 (W 0.00672898) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 1.86095 ± 0.00671938 (W 0.00562983) | 4.29612 ± 0.00753257 (W 0.00850094) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 6.91403 ± 0.0159391 (W 0.0085892) | 17.6524 ± 0.0268199 (W 0.0293873) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 6.89123 ± 0.0059604 (W 0.0116853) | 17.7065 ± 0.0633058 (W 0.0394856) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0031 ± 3.60555e-05 (W 2.16795e-05) | 0.00252 ± 1.1547e-05 (W 1.78382e-05) | 0.00122 ± 5.7735e-06 (W 1.34164e-05) | 🔴 colmat 1.23× slower than `faer` | 🔴 colmat 2.54× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.00814 ± 4.58258e-05 (W 6.78624e-05) | 0.00697 ± 3.751e-05 (W 4.20143e-05) | 0.00317 ± 1e-05 (W 1.69765e-05) | 🔴 colmat 1.17× slower than `faer` | 🔴 colmat 2.57× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.018959 ± 0.000309823 (W 0.000170969) | 0.01686 ± 0.000106444 (W 0.000146155) | 0.00988 ± 0.000166934 (W 0.000133713) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 1.92× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |

## linalg-mid, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.126806 ± 2.91376e-05 (W 0.000874519) | 0.523447 ± 0.0012229 (W 0.00120849) | - | 🔵 colmat 4.13× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.125875 ± 0.000322275 (W 0.000327744) | 0.125465 ± 8.92263e-05 (W 0.00164836) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.471478 ± 0.00134493 (W 0.00170624) | 1.23917 ± 0.00058217 (W 0.00184657) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.459468 ± 0.000667962 (W 0.00126025) | 0.468508 ± 0.00291572 (W 0.00244547) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.22345 ± 0.0102523 (W 0.00248438) | 3.74181 ± 0.0135725 (W 0.00454719) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.13983 ± 0.000993751 (W 0.00174091) | 2.78255 ± 0.0112596 (W 0.00415582) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.18065 ± 0.00327152 (W 0.00135538) | 1.36712 ± 0.00297572 (W 0.00196771) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.18359 ± 0.00345184 (W 0.00271084) | 1.37167 ± 0.00177867 (W 0.00195653) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.59018 ± 0.00632014 (W 0.00534631) | 5.93439 ± 0.0190764 (W 0.00541676) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.58749 ± 0.0167495 (W 0.00483269) | 5.94497 ± 0.0208199 (W 0.00529237) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 30.6821 ± 0.0671768 (W 0.0276947) | 32.4308 ± 0.0149399 (W 0.0259122) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 30.7485 ± 0.178332 (W 0.0301252) | 32.4777 ± 0.095043 (W 0.0241568) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.282112 ± 0.00144 (W 0.00154366) | 0.234233 ± 0.000561597 (W 0.00168402) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.262873 ± 0.000673131 (W 0.0010275) | 0.250183 ± 0.000921903 (W 0.00166624) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.71221 ± 0.00496287 (W 0.00414901) | 2.25582 ± 0.00938632 (W 0.00397006) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.80711 ± 0.00276306 (W 0.00154364) | 2.31594 ± 0.00396733 (W 0.00482603) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.7791 ± 0.109106 (W 0.0314049) | 18.2334 ± 0.0600773 (W 0.0133037) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.1908 ± 0.0400959 (W 0.0154983) | 20.0241 ± 0.137265 (W 0.0511445) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.236542 ± 0.000474444 (W 0.0017468) | 0.234964 ± 0.000194505 (W 0.00144224) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.23645 ± 0.000356479 (W 0.00131045) | 0.234 ± 0.000551541 (W 0.00134455) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.50878 ± 0.0182542 (W 0.00130855) | 1.51016 ± 0.00141049 (W 0.0029449) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.51243 ± 0.00353126 (W 0.00172515) | 1.50902 ± 0.00337156 (W 0.00227182) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 10.8635 ± 0.346612 (W 0.00950486) | 10.8663 ± 0.00296782 (W 0.00490039) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 10.8622 ± 0.0102883 (W 0.011356) | 10.8652 ± 0.0252533 (W 0.00924169) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.098096 ± 0.000305107 (W 0.00011234) | 0.188412 ± 0.000799873 (W 0.00142814) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.144886 ± 0.000350487 (W 0.00131097) | 0.106887 ± 0.000342862 (W 0.00141561) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.497267 ± 0.000198349 (W 0.00147927) | 1.74308 ± 0.00809979 (W 0.00725932) | - | 🔵 colmat 3.51× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.776037 ± 0.0017044 (W 0.00187104) | 0.764908 ± 0.00476116 (W 0.00305706) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.03873 ± 0.00394766 (W 0.00272947) | 14.2661 ± 0.0269466 (W 0.0285496) | - | 🔵 colmat 4.69× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.36383 ± 0.0141271 (W 0.00350534) | 5.95142 ± 0.0121435 (W 0.0179133) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.456057 ± 0.000910667 (W 0.00149065) | 0.698121 ± 0.00180091 (W 0.00111963) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.522206 ± 0.000712406 (W 0.00192782) | 0.584486 ± 0.00201972 (W 0.00242409) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.45064 ± 0.00294683 (W 0.00181497) | 3.20738 ± 0.00658911 (W 0.00531173) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.65566 ± 0.00593171 (W 0.00415608) | 2.75704 ± 0.0116213 (W 0.00261399) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 15.7074 ± 0.189724 (W 0.0251612) | 23.7055 ± 0.0660678 (W 0.01195) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 15.4432 ± 0.00694083 (W 0.00895669) | 17.3905 ± 0.060785 (W 0.00908913) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.227201 ± 0.000371549 (W 0.00111922) | 0.457641 ± 0.000553791 (W 0.00129626) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.28105 ± 0.00323761 (W 0.00231568) | 2.04897 ± 0.00298111 (W 0.00385596) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 6.92173 ± 0.0177462 (W 0.0151721) | 15.3229 ± 0.0665356 (W 0.0177925) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.093177 ± 0.00038029 (W 7.52682e-05) | 0.141434 ± 0.000521908 (W 0.00271101) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.117876 ± 0.000181403 (W 0.000131073) | 0.134826 ± 0.000531425 (W 0.00298183) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.47661 ± 5.61456e-05 (W 0.00169637) | 1.30434 ± 0.00644005 (W 0.00437749) | - | 🔵 colmat 2.74× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.599113 ± 0.000854315 (W 0.00308933) | 0.580324 ± 0.00286904 (W 0.00211856) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.98205 ± 0.0332786 (W 0.00384909) | 9.99231 ± 0.04031 (W 0.02097) | - | 🔵 colmat 3.35× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.47611 ± 0.0239867 (W 0.00532428) | 5.06716 ± 0.0166613 (W 0.00653954) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.87155 ± 0.00239073 (W 0.00360565) | 2.02882 ± 0.00442982 (W 0.00224162) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.86857 ± 0.00173242 (W 0.00253301) | 2.03898 ± 0.0172439 (W 0.00355975) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.79762 ± 0.0307537 (W 0.0268614) | 10.8865 ± 0.0121507 (W 0.0150581) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.83716 ± 0.0419394 (W 0.0108201) | 10.9144 ± 0.034648 (W 0.00911351) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 54.9576 ± 0.240318 (W 0.158549) | 68.7197 ± 0.900532 (W 0.648274) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 54.9621 ± 0.041212 (W 0.0934221) | 68.3754 ± 0.0472748 (W 0.0730423) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.677542 ± 0.00177949 (W 0.00202735) | 1.37239 ± 0.00428137 (W 0.00201591) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.677372 ± 0.00298746 (W 0.00175619) | 1.3765 ± 0.0177339 (W 0.00205319) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.70262 ± 0.0113146 (W 0.0185683) | 7.333 ± 0.019033 (W 0.0101995) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.69145 ± 0.0428536 (W 0.00961095) | 7.3393 ± 0.011848 (W 0.0125106) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 28.1263 ± 0.0624788 (W 0.0527835) | 48.5262 ± 0.0591636 (W 0.116811) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 28.269 ± 0.0289641 (W 0.0390507) | 46.8715 ± 0.025793 (W 0.0347238) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.003959 ± 1.0504e-05 (W 2.05961e-05) | 0.00366 ± 1.52753e-05 (W 2.44949e-05) | 0.00381 ± 1.57162e-05 (W 2.4499e-05) | 🔴 colmat 1.08× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01047 ± 6.4291e-05 (W 3.66238e-05) | 0.00988 ± 6.08276e-05 (W 3.77121e-05) | 0.011089 ± 4.58258e-05 (W 3.28679e-05) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.06× faster than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.034679 ± 0.000620431 (W 0.00111734) | 0.032789 ± 0.000141067 (W 0.000116748) | 0.037369 ± 8.62168e-05 (W 0.00107898) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.08× faster than `lapack-mkl` |  |

## linalg-mid, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.118456 ± 0.000289581 (W 0.000559815) | 0.3415 ± 0.107194 (W 0.101983) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.119326 ± 5.7735e-06 (W 0.000464252) | 0.131586 ± 0.000458927 (W 0.0014804) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.347628 ± 0.000987832 (W 0.00161091) | 0.81063 ± 0.00576255 (W 0.170452) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.346267 ± 0.000231612 (W 0.00101956) | 0.428244 ± 0.00781839 (W 0.00637298) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.10535 ± 0.00282642 (W 0.0042475) | 2.10721 ± 0.194036 (W 0.142152) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.12846 ± 0.000395042 (W 0.00201064) | 1.93908 ± 0.015714 (W 0.0450421) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.569499 ± 0.000601271 (W 0.00173095) | 1.24021 ± 0.00355285 (W 0.00211176) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.567954 ± 0.00260638 (W 0.00138495) | 1.24504 ± 0.00259455 (W 0.00226421) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.16211 ± 0.00156394 (W 0.00445793) | 4.29422 ± 0.0941219 (W 0.0710243) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.16325 ± 0.00254951 (W 0.00637181) | 4.18654 ± 0.0184499 (W 0.0764567) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.0848 ± 0.0323182 (W 0.019985) | 16.4048 ± 0.0158913 (W 0.0204974) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.1241 ± 0.0270211 (W 0.0118557) | 16.4278 ± 0.343547 (W 0.0326156) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.190915 ± 0.000827707 (W 0.00131144) | 0.233422 ± 0.0005872 (W 0.00210609) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.202932 ± 0.00833408 (W 0.0020188) | 0.25183 ± 0.00106384 (W 0.00216635) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 0.774844 ± 0.00411169 (W 0.00143688) | 2.24496 ± 0.00942339 (W 0.00711647) | - | 🔵 colmat 2.90× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.988243 ± 0.00639205 (W 0.00262398) | 2.14843 ± 0.00689161 (W 0.00599455) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 3.81329 ± 0.00867633 (W 0.00759606) | 12.8742 ± 0.0631396 (W 0.0164433) | - | 🔵 colmat 3.38× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 4.93706 ± 0.0147498 (W 0.00591014) | 13.0633 ± 0.684428 (W 0.0177544) | - | 🔵 colmat 2.65× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.074527 ± 0.000211653 (W 0.00104589) | 0.072398 ± 0.000351919 (W 0.0006225) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.074838 ± 0.000305996 (W 0.000460931) | 0.072328 ± 0.000330061 (W 0.000475468) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.414105 ± 0.00043832 (W 0.00243284) | 0.415538 ± 0.00287392 (W 0.00217535) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.418244 ± 0.00125359 (W 0.00126017) | 0.414045 ± 0.00132111 (W 0.00199149) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.88488 ± 0.00961994 (W 0.0525907) | 2.87763 ± 0.00396373 (W 0.004356) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.796 ± 0.0129547 (W 0.0145305) | 2.76316 ± 0.0587605 (W 0.00624274) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.100156 ± 0.00121637 (W 0.000701625) | 0.196614 ± 0.00281413 (W 0.00253052) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.097646 ± 8.36082e-05 (W 0.000423961) | 0.120396 ± 0.000565739 (W 0.00230085) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.33784 ± 0.00022466 (W 0.00165859) | 1.56954 ± 0.0297912 (W 0.0373942) | - | 🔵 colmat 4.65× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.557012 ± 0.00218303 (W 0.002096) | 0.548949 ± 0.00171061 (W 0.00469155) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.51105 ± 0.00343747 (W 0.00353884) | 9.7787 ± 0.122461 (W 0.338408) | - | 🔵 colmat 6.47× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.64467 ± 0.00372063 (W 0.00620257) | 3.96311 ± 0.0272619 (W 0.0408582) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.30431 ± 0.000132605 (W 0.00123453) | 0.659239 ± 0.000542767 (W 0.00204751) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.35841 ± 0.00163836 (W 0.00220411) | 0.5261 ± 0.00233447 (W 0.00172958) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.930904 ± 0.00162892 (W 0.0013455) | 2.41747 ± 0.00166679 (W 0.00955668) | - | 🔵 colmat 2.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.2759 ± 0.00494126 (W 0.00432266) | 1.82858 ± 0.0103644 (W 0.00504907) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 4.95088 ± 0.020031 (W 0.00717538) | 14.0868 ± 0.489175 (W 0.0370802) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 6.18474 ± 0.035039 (W 0.0116198) | 8.92586 ± 0.22714 (W 0.0143461) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.195764 ± 0.000750826 (W 0.00143868) | 0.5328 ± 0.00124807 (W 0.00179903) | - | 🔵 colmat 2.72× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.595148 ± 0.00220688 (W 0.00136103) | 2.07599 ± 0.000799294 (W 0.0057783) | - | 🔵 colmat 3.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.61484 ± 0.00112747 (W 0.00523295) | 11.7991 ± 0.0619638 (W 0.0385767) | - | 🔵 colmat 4.51× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.076497 ± 0.000265362 (W 0.000732755) | 0.161574 ± 0.000662612 (W 0.00315794) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.100006 ± 0.000354636 (W 0.000182113) | 0.118606 ± 0.000161091 (W 0.00196316) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.331301 ± 0.00111816 (W 0.00166924) | 1.12679 ± 0.0263251 (W 0.0233186) | - | 🔵 colmat 3.40× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.55707 ± 0.000154444 (W 0.00240735) | 0.522056 ± 0.00507676 (W 0.00522657) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.43764 ± 0.00924959 (W 0.00451064) | 6.84194 ± 0.160868 (W 0.162324) | - | 🔵 colmat 4.76× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.57116 ± 0.00492668 (W 0.00403446) | 3.80599 ± 0.0154331 (W 0.0335282) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.2415 ± 0.00203943 (W 0.00256999) | 1.57078 ± 0.00216369 (W 0.00513562) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.251 ± 0.0032435 (W 0.00540436) | 1.57125 ± 0.00643113 (W 0.00395505) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.87277 ± 0.0178617 (W 0.0178776) | 7.96705 ± 0.0201437 (W 0.0190582) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 4.89457 ± 0.0132943 (W 0.00906155) | 7.9997 ± 0.0328656 (W 0.035242) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 23.4378 ± 0.0209105 (W 0.0506663) | 38.1589 ± 1.10913 (W 0.261082) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 23.578 ± 0.0516328 (W 0.0557951) | 36.4017 ± 0.0990042 (W 0.0936731) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.629217 ± 0.00108726 (W 0.00274156) | 1.1936 ± 0.00240236 (W 0.00272862) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.633149 ± 0.00204185 (W 0.00360552) | 1.19324 ± 0.00450442 (W 0.0028139) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.6287 ± 0.0134403 (W 0.0121234) | 6.31378 ± 0.00810019 (W 0.0152628) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.63486 ± 0.00754246 (W 0.00798167) | 6.33211 ± 0.019434 (W 0.0201788) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 13.4485 ± 0.0266518 (W 0.0477056) | 29.0453 ± 0.752031 (W 0.0347918) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 13.445 ± 0.0218329 (W 0.0475838) | 28.9491 ± 0.8367 (W 0.0336709) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n128/b1` | 0.00464 ± 0 (W 2.75717e-05) | 0.00366 ± 0 (W 2.16795e-05) | 0.00379 ± 5.7735e-06 (W 3.71484e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.22× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.010759 ± 3.51188e-05 (W 5.89517e-05) | 0.009919 ± 8.53834e-05 (W 3.34096e-05) | 0.011109 ± 2.58908e-05 (W 5.0063e-05) | 🔴 colmat 1.08× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.025489 ± 3e-05 (W 9.12431e-05) | 0.024749 ± 0.000283784 (W 0.000188091) | 0.037129 ± 0.000141774 (W 0.000104763) | · | 🔵 colmat 1.46× faster than `lapack-mkl` |  |

## linalg-mid, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.118325 ± 0.00036376 (W 0.000278399) | 0.155465 ± 0.00374678 (W 0.00112599) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.119185 ± 3.23161e-05 (W 0.000489114) | 0.127026 ± 0.000306769 (W 0.000878976) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.337471 ± 0.000797165 (W 0.00183545) | 0.665799 ± 0.00326617 (W 0.0156107) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.336468 ± 0.00122552 (W 0.00211062) | 0.428026 ± 0.0109044 (W 0.0168359) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.951537 ± 0.0016066 (W 0.00162293) | 1.82427 ± 0.104253 (W 0.0349452) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 1.02266 ± 0.0259316 (W 0.00265054) | 1.83978 ± 0.0118495 (W 0.0275929) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.476353 ± 0.0019643 (W 0.00216529) | 1.24859 ± 0.00153607 (W 0.00177295) | - | 🔵 colmat 2.62× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.476884 ± 0.000728385 (W 0.00218822) | 1.25671 ± 0.00083 (W 0.00193975) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.72468 ± 0.00645555 (W 0.00554542) | 4.22036 ± 0.0790462 (W 0.070218) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 1.73051 ± 0.00260931 (W 0.00318158) | 4.20069 ± 0.0118196 (W 0.0135563) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 7.14941 ± 0.0150421 (W 0.0134959) | 15.1161 ± 0.287107 (W 0.0417084) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 7.17466 ± 0.0138915 (W 0.0118616) | 14.6106 ± 0.355949 (W 0.120543) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.187903 ± 0.00232367 (W 0.0021901) | 0.232522 ± 0.0012322 (W 0.00177621) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.205664 ± 0.00200464 (W 0.00221194) | 0.250633 ± 0.000480168 (W 0.001799) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.659539 ± 0.00214716 (W 0.00430497) | 2.24124 ± 0.00392821 (W 0.0104858) | - | 🔵 colmat 3.40× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.902362 ± 0.00309291 (W 0.00435829) | 2.15527 ± 0.00156982 (W 0.00703591) | - | 🔵 colmat 2.39× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 2.58916 ± 0.0195926 (W 0.00615508) | 12.0022 ± 0.0427669 (W 0.0159668) | - | 🔵 colmat 4.64× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 3.80014 ± 0.0157407 (W 0.0145103) | 12.2481 ± 0.119341 (W 0.0199096) | - | 🔵 colmat 3.22× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.059028 ± 0.000456216 (W 0.000585338) | 0.084927 ± 0.00102516 (W 0.00139958) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.059159 ± 0.000340823 (W 0.000477734) | 0.085078 ± 0.000598003 (W 0.00156615) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.234942 ± 0.0035491 (W 0.00267424) | 0.335418 ± 0.0010013 (W 0.0127742) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.234252 ± 0.00055981 (W 0.00214235) | 0.337521 ± 0.00751452 (W 0.010333) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.52366 ± 0.0080619 (W 0.00963314) | 1.58855 ± 0.216774 (W 0.207386) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.51948 ± 0.00131238 (W 0.00699453) | 1.90884 ± 0.24543 (W 0.240613) | - | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.63 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.105356 ± 0.00121935 (W 0.000875509) | 0.192553 ± 0.00113479 (W 0.00171617) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.094417 ± 0.000213854 (W 0.000325161) | 0.115026 ± 0.000400712 (W 0.000672997) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.281901 ± 0.00128192 (W 0.00159985) | 1.53234 ± 0.0126954 (W 0.0126886) | - | 🔵 colmat 5.44× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.527761 ± 0.00226253 (W 0.0026596) | 0.575524 ± 0.00763001 (W 0.013645) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.11885 ± 0.00299654 (W 0.00333117) | 8.27658 ± 0.157736 (W 0.254356) | - | 🔵 colmat 7.40× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.33731 ± 0.00850024 (W 0.00268677) | 3.87112 ± 1.08047 (W 0.0368806) | - | 🔵 colmat 1.66× faster than `faer` (marginal: spread 0.71 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.26591 ± 4.40606e-05 (W 0.00118437) | 0.70036 ± 0.00308575 (W 0.00718611) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.330889 ± 0.000828227 (W 0.00254495) | 0.593831 ± 0.00104797 (W 0.00273484) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.760357 ± 0.00449977 (W 0.00315579) | 2.58388 ± 0.0051432 (W 0.0126587) | - | 🔵 colmat 3.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.13008 ± 0.116869 (W 0.0044489) | 1.73472 ± 0.0113995 (W 0.00785717) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 3.02893 ± 0.0105224 (W 0.0299641) | 14.4315 ± 0.465828 (W 0.0715874) | - | 🔵 colmat 4.76× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 4.44381 ± 0.0106379 (W 0.0113349) | 7.74811 ± 0.0302907 (W 0.0508559) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.178064 ± 0.00141148 (W 0.00123088) | 0.572784 ± 0.000889776 (W 0.00781489) | - | 🔵 colmat 3.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.555335 ± 0.00164075 (W 0.00247009) | 2.2469 ± 0.0275716 (W 0.0118345) | - | 🔵 colmat 4.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.67137 ± 0.097834 (W 0.00235775) | 12.551 ± 0.504132 (W 0.0451699) | - | 🔵 colmat 7.51× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.076678 ± 0.000270759 (W 0.000568153) | 0.155065 ± 0.00251095 (W 0.00104247) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.096867 ± 0.000571343 (W 0.000549199) | 0.114587 ± 0.000274913 (W 0.00297459) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.28562 ± 0.00548981 (W 0.00201833) | 1.15346 ± 0.0116177 (W 0.0152444) | - | 🔵 colmat 4.04× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.529022 ± 0.00366932 (W 0.00320096) | 0.501596 ± 0.0135895 (W 0.0165107) | - | 🔴 colmat 1.05× slower than `faer` (marginal: spread 0.63 of the difference) | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.07659 ± 0.0183991 (W 0.00297917) | 6.39866 ± 0.0140315 (W 0.0829255) | - | 🔵 colmat 5.94× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.29326 ± 0.00836744 (W 0.00406136) | 3.67858 ± 0.0184788 (W 0.0374779) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.21766 ± 0.00214156 (W 0.00396622) | 1.58981 ± 0.00135953 (W 0.00623625) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.21837 ± 0.00425476 (W 0.00317203) | 1.5925 ± 0.00853227 (W 0.00367865) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.52214 ± 0.015919 (W 0.0100948) | 8.06661 ± 0.0178373 (W 0.0183945) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 4.56171 ± 0.0124837 (W 0.00894248) | 8.10604 ± 0.00343775 (W 0.012997) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 19.5817 ± 0.0274267 (W 0.0562835) | 34.0523 ± 0.070987 (W 0.0895293) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 19.6523 ± 0.756718 (W 0.0443658) | 34.3055 ± 1.07384 (W 0.193306) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 0.65036 ± 0.00324051 (W 0.00191657) | 1.19493 ± 0.0102296 (W 0.00354329) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.649077 ± 0.00419244 (W 0.00235034) | 1.19737 ± 0.00368085 (W 0.00572077) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.60976 ± 0.0112241 (W 0.0118237) | 6.39353 ± 0.0172544 (W 0.00988718) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.61025 ± 0.0179902 (W 0.00671918) | 6.42094 ± 0.100814 (W 0.0139534) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.9929 ± 0.0271124 (W 0.0283768) | 27.4263 ± 0.160571 (W 0.050447) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 12.0135 ± 0.030587 (W 0.0379749) | 27.5064 ± 0.0360033 (W 0.030069) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00463 ± 5.7735e-06 (W 3.07278e-05) | 0.00368 ± 1.1547e-05 (W 4.1039e-05) | 0.00378 ± 1.1547e-05 (W 2.07364e-05) | 🔴 colmat 1.26× slower than `faer` | 🔴 colmat 1.22× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0107 ± 4.0501e-05 (W 5.47111e-05) | 0.00982 ± 0.000136504 (W 4.81072e-05) | 0.01105 ± 3.09892e-05 (W 4.28404e-05) | 🔴 colmat 1.09× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.024219 ± 0.000134859 (W 9.37187e-05) | 0.02352 ± 0.000321715 (W 0.000332307) | 0.037269 ± 0.000327872 (W 0.000291462) | · | 🔵 colmat 1.54× faster than `lapack-mkl` |  |

## linalg-mid, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.117505 ± 0.000124195 (W 0.00111075) | 0.117966 ± 2.95691e-05 (W 0.000175499) | - | · | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.121667 ± 0.000198464 (W 0.000869082) | 0.121637 ± 0.000127441 (W 0.000160404) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.406908 ± 0.00116679 (W 0.00171482) | 0.409923 ± 0.00165791 (W 0.00200444) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.427042 ± 0.00161449 (W 0.00157435) | 0.425199 ± 0.00159281 (W 0.00143382) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.90448 ± 0.00185822 (W 0.00254573) | 1.93407 ± 0.00302763 (W 0.000689002) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.99564 ± 0.00155908 (W 0.00369994) | 1.98297 ± 0.0037854 (W 0.00137267) | - | · | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.17986 ± 0.00194737 (W 0.0018444) | 1.3659 ± 0.0030575 (W 0.00295392) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.18595 ± 0.000755483 (W 0.00193921) | 1.36666 ± 0.00167172 (W 0.00214352) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.56415 ± 0.00567247 (W 0.00675726) | 5.90308 ± 0.00987002 (W 0.00305191) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.57178 ± 0.00658388 (W 0.0054356) | 5.92885 ± 0.00743696 (W 0.0031091) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 30.5714 ± 0.031805 (W 0.0113196) | 31.0964 ± 0.0565805 (W 0.0125439) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 30.6749 ± 0.0593853 (W 0.0179703) | 31.2105 ± 0.0805405 (W 0.0305295) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.23999 ± 0.000910486 (W 0.00113542) | 0.231064 ± 0.00161274 (W 0.00193456) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.24318 ± 0.000534962 (W 0.000915937) | 0.234863 ± 6.74858e-05 (W 0.0014709) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.60374 ± 0.151908 (W 0.00503211) | 2.06549 ± 0.00866692 (W 0.00893268) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.72619 ± 0.00442414 (W 0.00357214) | 2.21004 ± 0.0253602 (W 0.00454847) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.508 ± 0.0298785 (W 0.0103293) | 16.7675 ± 0.0754985 (W 0.0131142) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.5709 ± 0.133099 (W 0.0354982) | 16.9983 ± 0.245685 (W 0.0221765) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.2384 ± 0.00111634 (W 0.0010642) | 0.236012 ± 0.0010498 (W 0.000871983) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.237662 ± 0.000517279 (W 0.000357317) | 0.235624 ± 0.000399354 (W 0.000444573) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.53789 ± 0.0126428 (W 0.00277604) | 1.54132 ± 0.0132195 (W 0.00137895) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.54464 ± 0.0063136 (W 0.00250331) | 1.5512 ± 0.00554547 (W 0.00296856) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 10.7822 ± 0.0157059 (W 0.00439866) | 10.7534 ± 0.0347763 (W 0.010058) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 10.7891 ± 0.0188018 (W 0.0051024) | 10.7757 ± 0.0282647 (W 0.00540382) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.094287 ± 0.000625127 (W 0.000153422) | 0.086267 ± 0.000189026 (W 0.00111935) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.125497 ± 0.000236417 (W 0.000372033) | 0.092358 ± 0.000233155 (W 0.00133186) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.501769 ± 0.00118907 (W 0.00207236) | 0.468217 ± 0.000661175 (W 0.00147261) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.693692 ± 0.00220137 (W 0.00171483) | 0.660522 ± 0.00189112 (W 0.0030507) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.03629 ± 0.00543329 (W 0.00141328) | 3.2206 ± 0.00668106 (W 0.00452843) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.8986 ± 0.00843581 (W 0.00142665) | 3.92952 ± 0.00911965 (W 0.00265392) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.444927 ± 0.000644001 (W 0.00129185) | 0.515725 ± 0.00100276 (W 0.00152433) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.502969 ± 0.00859017 (W 0.0016454) | 0.565754 ± 0.000621515 (W 0.00210337) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.25978 ± 0.0078415 (W 0.00144159) | 2.35916 ± 0.00116094 (W 0.00225834) | - | · | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.56142 ± 0.00856786 (W 0.0037915) | 2.6486 ± 0.00398359 (W 0.00193746) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 14.0979 ± 0.00338906 (W 0.00645822) | 14.396 ± 0.0175016 (W 0.0152367) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 15.0104 ± 0.0171581 (W 0.0165473) | 15.3773 ± 0.0730309 (W 0.0258835) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.207241 ± 0.000770134 (W 0.00130103) | 0.282328 ± 0.000595867 (W 0.00101392) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.10447 ± 0.000417859 (W 0.00170226) | 1.21128 ± 0.00571595 (W 0.00203917) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 6.14456 ± 0.00367942 (W 0.00346642) | 6.53291 ± 0.0122223 (W 0.00486453) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.093546 ± 0.000308568 (W 0.000992257) | 0.084816 ± 0.000505751 (W 6.638e-05) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.098357 ± 0.000262071 (W 0.000156113) | 0.118836 ± 0.000157788 (W 0.00124572) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.48554 ± 0.00231339 (W 0.0019047) | 0.460901 ± 0.000517476 (W 0.00174974) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.519945 ± 0.00100935 (W 0.00230652) | 0.48642 ± 0.000692727 (W 0.00198699) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.94589 ± 0.00497917 (W 0.00179979) | 3.12259 ± 0.0126818 (W 0.00250866) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.06637 ± 0.00209856 (W 0.00248332) | 3.06333 ± 0.00287786 (W 0.00683861) | - | · | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.85766 ± 0.0064298 (W 0.0019521) | 2.04364 ± 0.00856496 (W 0.00245734) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.85853 ± 0.0168231 (W 0.00143242) | 2.03009 ± 0.012788 (W 0.00171572) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.73637 ± 0.0156071 (W 0.029424) | 10.793 ± 0.0157247 (W 0.0107592) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.76058 ± 0.0173918 (W 0.00973388) | 10.8099 ± 0.061171 (W 0.0113549) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 51.5685 ± 0.178929 (W 0.0889364) | 66.7014 ± 0.0705458 (W 0.03436) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 51.6288 ± 0.127996 (W 0.0841635) | 66.6319 ± 0.0443265 (W 0.137357) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.665161 ± 0.00146566 (W 0.0015028) | 1.35288 ± 0.000761972 (W 0.00211976) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.665803 ± 0.000628024 (W 0.00182259) | 1.35401 ± 0.00413041 (W 0.00229592) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.63104 ± 0.0154494 (W 0.0136758) | 7.25006 ± 0.00872386 (W 0.0150719) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.61624 ± 0.0359442 (W 0.0404689) | 7.25429 ± 0.0158275 (W 0.00790463) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 24.8654 ± 0.0546396 (W 0.0359623) | 45.2901 ± 0.0257312 (W 0.0224351) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 25.0924 ± 0.145826 (W 0.0358794) | 45.006 ± 0.101086 (W 0.175271) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00417 ± 5.7735e-06 (W 3.02192e-05) | 0.00426 ± 1.52753e-05 (W 2.41909e-05) | 0.00363 ± 6.08276e-06 (W 2.28298e-05) | · | 🔴 colmat 1.15× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.01012 ± 0.000485631 (W 5.05984e-05) | 0.010359 ± 7.17658e-05 (W 6.60477e-05) | 0.010869 ± 5.7735e-07 (W 3.01712e-05) | · | 🔵 colmat 1.07× faster than `lapack-mkl` (marginal: spread 0.65 of the difference) | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.034669 ± 0.0004392 (W 0.000240053) | 0.034879 ± 7.76745e-05 (W 0.000183494) | 0.037408 ± 3.4356e-05 (W 0.000163656) | · | 🔵 colmat 1.08× faster than `lapack-mkl` |  |

## linalg-mid, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.110417 ± 0.000251814 (W 0.000149499) | 0.122045 ± 0.000658431 (W 0.00152193) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.115337 ± 0.000243067 (W 0.000271533) | 0.128756 ± 0.00035303 (W 0.00190376) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.292061 ± 0.000668881 (W 0.00168897) | 0.35876 ± 0.00296311 (W 0.00902428) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.314211 ± 0.000932455 (W 0.00125734) | 0.381887 ± 0.00340568 (W 0.0106255) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.894625 ± 0.00218941 (W 0.00217061) | 1.05226 ± 0.00484696 (W 0.0431144) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.984238 ± 0.00185183 (W 0.00164468) | 1.1366 ± 0.0117885 (W 0.0381709) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.564304 ± 0.00124758 (W 0.00231428) | 1.23724 ± 0.00301656 (W 0.00248308) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.566031 ± 0.000591749 (W 0.00172789) | 1.24058 ± 0.00175249 (W 0.00110822) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.11738 ± 0.00287962 (W 0.0063734) | 4.26352 ± 0.0773227 (W 0.0826957) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.12653 ± 0.00420754 (W 0.00411453) | 4.14161 ± 0.0750023 (W 0.0648167) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 9.91169 ± 0.0161794 (W 0.0156875) | 15.0331 ± 0.0615203 (W 0.018486) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 9.94641 ± 0.0254041 (W 0.0164891) | 15.0669 ± 0.053563 (W 0.0265528) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.187353 ± 0.00137582 (W 0.000883035) | 0.231643 ± 0.000537787 (W 0.00196637) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.184244 ± 0.000946818 (W 0.00129781) | 0.234641 ± 0.000426215 (W 0.00107966) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.781608 ± 0.00182365 (W 0.00139201) | 2.06144 ± 0.00496873 (W 0.00573327) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.904221 ± 0.00620878 (W 0.0038417) | 2.04702 ± 0.00334815 (W 0.00803564) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 4.13293 ± 0.016566 (W 0.00543714) | 11.2306 ± 0.0468099 (W 0.0119352) | - | 🔵 colmat 2.72× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 4.50098 ± 0.0147664 (W 0.00674616) | 10.9828 ± 0.0472795 (W 0.014663) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.074157 ± 0.00057956 (W 0.000223338) | 0.072428 ± 0.000493214 (W 0.000216696) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.075267 ± 0.000319808 (W 0.000446251) | 0.072868 ± 0.000266039 (W 0.000428936) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.418138 ± 0.00106029 (W 0.00117429) | 0.414994 ± 0.00103654 (W 0.00166797) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.418197 ± 0.0020813 (W 0.00240964) | 0.414629 ± 0.00180901 (W 0.00119158) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.81963 ± 0.00203471 (W 0.00543099) | 2.86575 ± 0.0459956 (W 0.00502596) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.81056 ± 0.00483246 (W 0.00525743) | 2.8704 ± 0.0423872 (W 0.00605688) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.071208 ± 6.95006e-05 (W 0.000328892) | 0.097077 ± 0.000707272 (W 0.000926346) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.079008 ± 0.000147886 (W 0.000383891) | 0.106527 ± 0.000578317 (W 0.00182143) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.285341 ± 0.000905573 (W 0.00164136) | 0.424414 ± 0.00401175 (W 0.00562966) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.471633 ± 0.000843422 (W 0.003017) | 0.457723 ± 0.00310956 (W 0.00477846) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.33724 ± 0.00270716 (W 0.00467326) | 1.90503 ± 0.0279379 (W 0.0368449) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.22091 ± 0.00472331 (W 0.00316106) | 1.95636 ± 0.0149223 (W 0.0582196) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.27517 ± 0.000553175 (W 0.00188021) | 0.449088 ± 0.000835779 (W 0.00230982) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.340688 ± 0.000771221 (W 0.00227713) | 0.509014 ± 0.00182307 (W 0.00193716) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.869146 ± 0.00035283 (W 0.00217102) | 1.40139 ± 0.00139707 (W 0.00465381) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.18693 ± 0.00173068 (W 0.00310441) | 1.71802 ± 0.00186338 (W 0.00489513) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 4.45625 ± 0.00405389 (W 0.00567203) | 5.6511 ± 0.00273304 (W 0.00864059) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 5.7665 ± 0.0176737 (W 0.00653949) | 6.59645 ± 0.207274 (W 0.0147327) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.181073 ± 0.000624573 (W 0.00113527) | 0.340939 ± 0.00161851 (W 0.00175862) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.538653 ± 0.00230654 (W 0.00286328) | 1.07696 ± 0.0034315 (W 0.0035361) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.40173 ± 0.00172551 (W 0.00451427) | 3.59276 ± 0.00249614 (W 0.0103572) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.070637 ± 0.000466334 (W 0.000281825) | 0.095988 ± 0.000196734 (W 0.00141673) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.080878 ± 0.000262063 (W 0.000267058) | 0.105147 ± 0.000165862 (W 0.000995002) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.275659 ± 0.000397923 (W 0.00168908) | 0.381828 ± 0.00258471 (W 0.00623554) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.471976 ± 0.00460936 (W 0.00202472) | 0.421397 ± 0.00106934 (W 0.0070377) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.24639 ± 0.00629146 (W 0.0048343) | 1.60076 ± 0.0269951 (W 0.045531) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.136 ± 0.00619615 (W 0.00673761) | 1.78131 ± 0.00578952 (W 0.0414938) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.23179 ± 0.00231829 (W 0.00415237) | 1.56115 ± 0.00233698 (W 0.00375266) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.24549 ± 0.00399221 (W 0.00406299) | 1.55924 ± 0.0033945 (W 0.00628897) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.80591 ± 0.00562658 (W 0.009125) | 7.88942 ± 0.0156281 (W 0.033104) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 4.83305 ± 0.00547231 (W 0.00991328) | 7.90817 ± 0.0272229 (W 0.0333874) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 21.4763 ± 0.0344322 (W 0.0317553) | 34.3677 ± 0.446811 (W 0.0344912) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 21.6099 ± 0.0360763 (W 0.0296579) | 34.8075 ± 0.328847 (W 0.0850065) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.616003 ± 0.000687551 (W 0.00361991) | 1.18212 ± 0.0148232 (W 0.00264098) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.614577 ± 0.00310873 (W 0.00374503) | 1.1838 ± 0.00140721 (W 0.00220325) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.53671 ± 0.0158685 (W 0.00929853) | 6.24077 ± 0.0296383 (W 0.00881435) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.54983 ± 0.00288521 (W 0.00818485) | 6.27394 ± 0.0320763 (W 0.0192124) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.5426 ± 0.0270357 (W 0.0560451) | 27.0235 ± 0.405289 (W 0.0377253) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.557 ± 0.0187882 (W 0.0786777) | 26.9543 ± 0.246702 (W 0.0342229) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00481 ± 1.52753e-05 (W 3.20936e-05) | 0.00427 ± 2.64575e-05 (W 0.000259074) | 0.00364 ± 5.7735e-06 (W 3.04959e-05) | 🔴 colmat 1.13× slower than `faer` | 🔴 colmat 1.32× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.011259 ± 9.41293e-05 (W 0.000225287) | 0.01033 ± 2.46374e-05 (W 6.52403e-05) | 0.010869 ± 1.0504e-05 (W 2.61973e-05) | 🔴 colmat 1.09× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.027109 ± 0.000340441 (W 0.00025014) | 0.027749 ± 0.000280416 (W 0.000155927) | 0.037299 ± 0.000180103 (W 7.44265e-05) | · | 🔵 colmat 1.38× faster than `lapack-mkl` |  |

## linalg-mid, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.111297 ± 0.000204745 (W 0.000302033) | 0.117016 ± 0.000330424 (W 0.000415543) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.114906 ± 8.56174e-05 (W 0.000319005) | 0.123367 ± 0.000294812 (W 0.000411619) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.283682 ± 0.00066969 (W 0.00154019) | 0.35589 ± 0.0047559 (W 0.0163289) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.303482 ± 0.000573064 (W 0.00531679) | 0.37193 ± 0.0170303 (W 0.014429) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.770102 ± 0.00694801 (W 0.00227199) | 0.963903 ± 0.00367262 (W 0.0164348) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.858543 ± 0.00196242 (W 0.00193385) | 1.05116 ± 0.00843363 (W 0.0106927) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.473215 ± 0.000751982 (W 0.00161221) | 1.24563 ± 0.00217921 (W 0.00216244) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.474784 ± 0.00245147 (W 0.00228332) | 1.25394 ± 0.00445951 (W 0.00243971) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 1.68982 ± 0.00432912 (W 0.00196802) | 4.15941 ± 0.00669524 (W 0.0699608) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 1.69124 ± 0.00697581 (W 0.00567563) | 4.17481 ± 0.0162875 (W 0.0673013) | - | 🔵 colmat 2.47× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 7.01608 ± 0.0683351 (W 0.01604) | 13.8836 ± 0.0506232 (W 0.0652805) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 7.01618 ± 0.0113735 (W 0.0294312) | 13.8565 ± 0.0135206 (W 0.0443156) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.182385 ± 0.00135633 (W 0.0016544) | 0.232013 ± 0.000564359 (W 0.00162284) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.185294 ± 0.00173603 (W 0.00230317) | 0.233703 ± 0.000391242 (W 0.00140998) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.659649 ± 0.00674182 (W 0.00327678) | 2.05595 ± 0.00768323 (W 0.00585988) | - | 🔵 colmat 3.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 0.811081 ± 0.00533068 (W 0.0081434) | 2.04868 ± 0.00278656 (W 0.00541866) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 2.74834 ± 0.00633132 (W 0.00549505) | 10.4581 ± 0.0629929 (W 0.0230581) | - | 🔵 colmat 3.81× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 3.36745 ± 0.017326 (W 0.0116747) | 10.3035 ± 0.0455274 (W 0.0105938) | - | 🔵 colmat 3.06× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.056138 ± 0.000110151 (W 0.000428512) | 0.085747 ± 0.000654243 (W 0.000754794) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.057058 ± 0.000360579 (W 0.000280603) | 0.085687 ± 0.000166764 (W 0.000567775) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.234584 ± 0.000987203 (W 0.00177746) | 0.332879 ± 0.000170735 (W 0.0125423) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.237403 ± 0.000596054 (W 0.0028845) | 0.313349 ± 0.0129862 (W 0.0131483) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.51406 ± 0.00513574 (W 0.00736798) | 1.99875 ± 0.0358799 (W 0.23947) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.51885 ± 0.0537878 (W 0.00757414) | 1.9393 ± 0.249532 (W 0.235189) | - | 🔵 colmat 1.28× faster than `faer` (marginal: spread 0.72 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.067648 ± 0.000388887 (W 0.000371484) | 0.092147 ± 0.000594403 (W 0.000549891) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.075528 ± 0.000269296 (W 0.00036487) | 0.101887 ± 0.000547403 (W 0.000780228) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.253832 ± 0.00166141 (W 0.00141242) | 0.414229 ± 0.016214 (W 0.0124444) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.446796 ± 0.00102389 (W 0.00214525) | 0.474417 ± 0.00608863 (W 0.0137422) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.023 ± 0.00266746 (W 0.00221907) | 1.76541 ± 0.0298058 (W 0.0466147) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.89525 ± 0.0173261 (W 0.00498519) | 1.88145 ± 0.0220246 (W 0.0259572) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.235942 ± 0.000479646 (W 0.00150341) | 0.528115 ± 0.0142849 (W 0.00440852) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.31019 ± 0.00133808 (W 0.00150992) | 0.580692 ± 0.000716685 (W 0.00345386) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.712635 ± 0.00069871 (W 0.00261157) | 1.58634 ± 0.0130851 (W 0.00552133) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.04737 ± 0.00553255 (W 0.00194495) | 1.63653 ± 0.00893874 (W 0.00570665) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.7214 ± 0.00622023 (W 0.00425354) | 5.54168 ± 0.015419 (W 0.0294361) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.02827 ± 0.0121297 (W 0.00671073) | 5.71725 ± 0.0434806 (W 0.040267) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.167524 ± 0.000379961 (W 0.00108086) | 0.409435 ± 0.00249498 (W 0.00224337) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.516505 ± 0.000454227 (W 0.00190901) | 1.24273 ± 0.00564227 (W 0.00532524) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.57814 ± 0.00523069 (W 0.00303126) | 3.74514 ± 0.016645 (W 0.0181667) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.067378 ± 0.000409919 (W 0.000268198) | 0.090637 ± 9.53939e-05 (W 0.000504038) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.077407 ± 0.000536793 (W 0.000509448) | 0.099057 ± 0.000131149 (W 0.00210843) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.245843 ± 0.000808563 (W 0.00197639) | 0.393946 ± 0.0168123 (W 0.0121218) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.445456 ± 0.00202639 (W 0.00321466) | 0.433506 ± 0.00884263 (W 0.0142482) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.95463 ± 0.0054374 (W 0.00192204) | 1.51137 ± 0.0286275 (W 0.0369784) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.86042 ± 0.00249785 (W 0.00296059) | 1.70521 ± 0.0239396 (W 0.0400447) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.20632 ± 0.00329104 (W 0.00295446) | 1.58186 ± 0.00181749 (W 0.00437747) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.21666 ± 0.00486054 (W 0.00342445) | 1.57844 ± 0.00715843 (W 0.00491543) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 4.42409 ± 0.00952564 (W 0.00677514) | 7.98408 ± 0.0174191 (W 0.019115) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 4.4712 ± 0.0037765 (W 0.00954644) | 8.00246 ± 0.00861274 (W 0.019269) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 17.6513 ± 0.616023 (W 0.0227889) | 32.128 ± 0.389997 (W 0.0572931) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n512/b1 [api-inclusive]` | 17.7153 ± 0.0634217 (W 0.0437032) | 32.1809 ± 0.0862338 (W 0.0715957) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.63874 ± 0.00217579 (W 0.00223085) | 1.18308 ± 0.00208871 (W 0.00215506) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.637998 ± 0.00156109 (W 0.00196217) | 1.18594 ± 0.00236185 (W 0.00303062) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.52911 ± 0.00661905 (W 0.0116835) | 6.30357 ± 0.0220624 (W 0.0128274) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.5244 ± 0.00581422 (W 0.012268) | 6.34967 ± 0.0101759 (W 0.013433) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 10.0954 ± 0.0304341 (W 0.0265209) | 26.0739 ± 0.274849 (W 0.0323439) | - | 🔵 colmat 2.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 10.087 ± 0.00942116 (W 0.0355631) | 25.5551 ± 0.425678 (W 0.0381265) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0048 ± 4.45758e-05 (W 2.28954e-05) | 0.00427 ± 2e-05 (W 3.79236e-05) | 0.00362 ± 0 (W 2.1822e-05) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 1.33× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.011459 ± 2.3094e-05 (W 3.41497e-05) | 0.01031 ± 3.72872e-05 (W 5.25481e-05) | 0.01085 ± 3.25115e-05 (W 3.09241e-05) | 🔴 colmat 1.11× slower than `faer` | 🔴 colmat 1.06× slower than `lapack-mkl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.028619 ± 0.000346987 (W 0.000192691) | 0.027469 ± 0.000419325 (W 0.000232653) | 0.037099 ± 2e-05 (W 0.000142423) | · | 🔵 colmat 1.30× faster than `lapack-mkl` |  |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `lu/128`, `lu/256`, `lu/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `solve/128`, `solve/256`, `solve/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.563064 ± 0.000471123 (W 0.001828) | 0.778298 ± 0.00381491 (W 0.0025562) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.489726 ± 0.000683712 (W 0.00161413) | 0.540593 ± 0.00860685 (W 0.00531146) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 29.1893 ± 0.0241426 (W 0.0400721) | 56.0793 ± 0.0609998 (W 0.0324643) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 27.6216 ± 0.0957463 (W 0.0323227) | 33.8047 ± 0.065837 (W 0.0506187) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.224063 ± 0.00214872 (W 0.00256947) | 0.218512 ± 0.00254354 (W 0.00227396) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.221033 ± 0.00121401 (W 0.00242768) | 0.220604 ± 0.00278138 (W 0.00202926) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 13.3174 ± 0.0599253 (W 0.124137) | 13.0852 ± 0.0660008 (W 0.159364) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 13.2802 ± 0.0609652 (W 0.0528009) | 13.2943 ± 0.682409 (W 0.323176) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.186274 ± 0.000777775 (W 0.00154032) | 0.544873 ± 0.00319104 (W 0.00262632) | - | 🔵 colmat 2.93× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.186684 ± 0.00085049 (W 0.00137531) | 0.278642 ± 0.00214608 (W 0.00425939) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.6194 ± 0.0149743 (W 0.0144692) | 24.7785 ± 0.0229212 (W 0.0259479) | - | 🔵 colmat 4.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.48159 ± 0.0240258 (W 0.00610507) | 10.0064 ± 0.667764 (W 0.0349448) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.738007 ± 0.00687879 (W 0.00195061) | 1.3716 ± 0.00380962 (W 0.00573921) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.876833 ± 0.00121087 (W 0.00277338) | 0.937423 ± 0.0135191 (W 0.00335935) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 27.3539 ± 0.393043 (W 0.0815351) | 59.7109 ± 0.809175 (W 0.211547) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 28.4905 ± 0.121588 (W 0.0405133) | 31.539 ± 0.0771968 (W 0.0593873) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.507585 ± 0.0012609 (W 0.00182165) | 1.15183 ± 0.00294194 (W 0.00266308) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.0863 ± 0.0415111 (W 0.055003) | 46.4095 ± 0.141612 (W 0.0309675) | - | 🔵 colmat 3.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.08559 ± 0.00135208 (W 0.00280839) | 1.39247 ± 0.00408178 (W 0.00211905) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.08702 ± 0.00151416 (W 0.0045151) | 1.39857 ± 0.00183956 (W 0.00386424) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 35.8928 ± 0.0647285 (W 0.0293831) | 39.3167 ± 0.119779 (W 0.0262127) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 36.0181 ± 0.0735153 (W 0.0436569) | 39.5333 ± 0.125027 (W 0.0720594) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.597643 ± 0.00429799 (W 0.00500369) | 0.899868 ± 0.00432116 (W 0.0019186) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.598499 ± 0.00273283 (W 0.00677951) | 0.884755 ± 0.00953241 (W 0.00551027) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 18.7272 ± 0.0205604 (W 0.0224244) | 23.3867 ± 0.0222499 (W 0.0130419) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 18.7188 ± 0.0217979 (W 0.0193954) | 23.3928 ± 0.0817405 (W 0.0420372) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.31405 ± 0.00341132 (W 0.00765025) | 0.775136 ± 0.00238322 (W 0.00464066) | - | 🔵 colmat 2.47× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.333209 ± 1.70392e-05 (W 0.00136996) | 0.538103 ± 0.00119537 (W 0.00435108) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 8.49675 ± 0.0128376 (W 0.0244997) | 31.5043 ± 0.0340082 (W 0.0774035) | - | 🔵 colmat 3.71× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 9.16666 ± 0.0536373 (W 0.0260002) | 15.8894 ± 0.0251626 (W 0.0421224) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.080947 ± 0.00209474 (W 0.00242274) | 0.074447 ± 0.000788397 (W 0.00202894) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.084398 ± 0.00129574 (W 0.00243083) | 0.074938 ± 0.000891269 (W 0.00121973) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 3.73786 ± 0.00695773 (W 0.0194056) | 3.59794 ± 0.0323725 (W 0.0264311) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 3.73891 ± 0.00688538 (W 0.00860638) | 3.58106 ± 0.0532421 (W 0.108392) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.166776 ± 0.00067884 (W 0.00191859) | 0.591683 ± 0.00732436 (W 0.00687112) | - | 🔵 colmat 3.55× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.187385 ± 0.00179814 (W 0.00176398) | 0.274312 ± 0.00606637 (W 0.00322576) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.6237 ± 0.0539857 (W 0.00487145) | 22.2239 ± 0.067449 (W 0.0387343) | - | 🔵 colmat 8.47× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 3.20348 ± 0.0214796 (W 0.0174343) | 7.95418 ± 0.0296644 (W 0.0280146) | - | 🔵 colmat 2.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.453626 ± 0.0283302 (W 0.00236564) | 1.98546 ± 0.106261 (W 0.0673637) | - | 🔵 colmat 4.38× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 0.650307 ± 0.00430456 (W 0.00397561) | 0.725615 ± 0.0209773 (W 0.0202089) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 9.99237 ± 0.0833469 (W 0.0221404) | 51.6264 ± 0.345077 (W 0.179611) | - | 🔵 colmat 5.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.6345 ± 0.0526065 (W 0.0132953) | 16.7359 ± 2.15564 (W 0.122146) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.340998 ± 0.00176928 (W 0.00188378) | 2.05231 ± 0.0994701 (W 0.0905513) | - | 🔵 colmat 6.02× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 6.20899 ± 0.00661679 (W 0.010524) | 48.2691 ± 0.156848 (W 0.431212) | - | 🔵 colmat 7.77× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.818012 ± 0.00902698 (W 0.00440326) | 1.34749 ± 0.00393698 (W 0.00502758) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.824212 ± 0.00656109 (W 0.00357059) | 1.361 ± 0.000691407 (W 0.00375248) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 18.9496 ± 0.0442414 (W 0.0570037) | 25.1962 ± 0.0102011 (W 0.0461564) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 19.1304 ± 0.0461901 (W 0.0336976) | 25.3403 ± 0.0219924 (W 0.0608182) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.544634 ± 0.00332862 (W 0.00568369) | 0.993252 ± 0.00229251 (W 0.00233265) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.547354 ± 0.00169381 (W 0.00364718) | 0.988546 ± 0.00839417 (W 0.00255223) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 13.1749 ± 0.00445808 (W 0.0218701) | 19.4408 ± 0.00372699 (W 0.0460329) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.1571 ± 0.0119797 (W 0.0287703) | 19.5272 ± 0.0520926 (W 0.0721377) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.289882 ± 0.00234321 (W 0.00161206) | 0.773087 ± 0.00593623 (W 0.00347774) | - | 🔵 colmat 2.67× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.323641 ± 0.00106327 (W 0.0025901) | 0.545353 ± 0.0040019 (W 0.00332925) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 5.14539 ± 0.0106459 (W 0.0535081) | 27.0532 ± 0.0378892 (W 0.0276404) | - | 🔵 colmat 5.26× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 5.9389 ± 0.0699383 (W 0.0349537) | 10.7791 ± 0.0936646 (W 0.0499107) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.063568 ± 0.00213069 (W 0.00190538) | 0.069498 ± 0.00305505 (W 0.00209159) | - | 🔵 colmat 1.09× faster than `faer` (marginal: spread 0.87 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.065218 ± 0.00180583 (W 0.00107807) | 0.073318 ± 0.00246834 (W 0.002612) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 2.16564 ± 0.0116285 (W 0.0150851) | 2.67552 ± 0.286376 (W 0.290468) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 2.16507 ± 0.00517792 (W 0.00646506) | 2.68717 ± 0.258756 (W 0.284519) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.150336 ± 0.000779686 (W 0.00110873) | 0.617312 ± 0.00313384 (W 0.00857799) | - | 🔵 colmat 4.11× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.186325 ± 0.000702568 (W 0.00148393) | 0.27893 ± 0.00233069 (W 0.00365711) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.0463 ± 0.0101261 (W 0.00644498) | 21.6858 ± 0.103063 (W 0.0589002) | - | 🔵 colmat 10.60× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.88494 ± 0.00479302 (W 0.00776675) | 7.65768 ± 0.0246538 (W 0.0148034) | - | 🔵 colmat 2.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.396918 ± 0.00124648 (W 0.00281398) | 2.08594 ± 0.0578551 (W 0.0842668) | - | 🔵 colmat 5.26× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 0.616402 ± 0.00225933 (W 0.00248278) | 0.759764 ± 0.0662617 (W 0.0230682) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 7.22321 ± 0.0172409 (W 0.0244865) | 50.6828 ± 0.098225 (W 0.155009) | - | 🔵 colmat 7.02× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 12.7085 ± 0.00622944 (W 0.0344195) | 17.8904 ± 0.101588 (W 0.316767) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.33479 ± 0.000618248 (W 0.00238215) | 2.02323 ± 0.0091885 (W 0.0755605) | - | 🔵 colmat 6.04× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 4.86692 ± 0.00629357 (W 0.0150469) | 47.1083 ± 0.0287015 (W 0.119103) | - | 🔵 colmat 9.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.788126 ± 0.00422054 (W 0.00482879) | 1.42627 ± 0.00892471 (W 0.00431067) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.802973 ± 0.00870893 (W 0.0021696) | 1.42992 ± 0.0128675 (W 0.00875232) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 16.4486 ± 0.0651995 (W 0.0321675) | 27.8029 ± 1.20715 (W 0.0411586) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 16.5621 ± 0.0507775 (W 0.0416895) | 27.1952 ± 0.833298 (W 0.844771) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.534095 ± 0.00337734 (W 0.00693857) | 1.05524 ± 0.00509049 (W 0.00415758) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.537284 ± 0.00166875 (W 0.00442784) | 1.04441 ± 0.00533213 (W 0.00606793) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 12.0494 ± 0.0424002 (W 0.0124941) | 20.6942 ± 0.0214826 (W 0.0615565) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 12.1005 ± 0.0655669 (W 0.0229404) | 20.7351 ± 0.0254991 (W 0.0813697) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.472527 ± 0.0021566 (W 0.00188778) | 0.520686 ± 0.00299587 (W 0.00657309) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.432267 ± 0.000983484 (W 0.00181211) | 0.483777 ± 0.000269652 (W 0.00634252) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 29.5714 ± 0.032407 (W 0.0440465) | 30.6026 ± 0.065684 (W 0.011363) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 26.5465 ± 0.0947309 (W 0.0279216) | 27.5945 ± 0.0238284 (W 0.0244602) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.238612 ± 0.000759129 (W 0.00165527) | 0.235652 ± 0.00159329 (W 0.00169301) | 0.548824 ± 0.00113382 (W 0.00224438) | · | 🔵 colmat 2.30× faster than `lapack-mkl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.239154 ± 0.000764253 (W 0.00159874) | 0.236671 ± 0.00144973 (W 0.00181608) | 0.702125 ± 0.00192688 (W 0.00347812) | · | 🔵 colmat 2.94× faster than `lapack-mkl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.7247 ± 0.0623976 (W 0.0369714) | 12.7336 ± 0.0804341 (W 0.010062) | 27.5982 ± 0.366204 (W 0.0921285) | · | 🔵 colmat 2.17× faster than `lapack-mkl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.6928 ± 0.0630134 (W 0.0420763) | 12.7398 ± 0.0612764 (W 0.0179316) | 30.3612 ± 0.180916 (W 0.0319786) | · | 🔵 colmat 2.39× faster than `lapack-mkl` |  |
| `lu/1024x64/b1` | 0.120916 ± 0.000834286 (W 0.00154109) | 0.213932 ± 0.000922006 (W 0.00150286) | 0.163144 ± 0.000206376 (W 0.00110509) | 🔵 colmat 1.77× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.128346 ± 0.000168226 (W 0.00186559) | 0.224513 ± 0.000670754 (W 0.00216962) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.42549 ± 0.00471249 (W 0.0182889) | 5.79059 ± 0.0115922 (W 0.0203638) | 7.13415 ± 0.00953628 (W 0.00296606) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.47876 ± 0.0192677 (W 0.0119938) | 5.87531 ± 0.0218123 (W 0.00847194) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.596983 ± 0.00267315 (W 0.00174684) | 0.666399 ± 0.00133748 (W 0.00170516) | 1.18011 ± 0.00311339 (W 0.00228919) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.98× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.805002 ± 0.00499536 (W 0.00377914) | 0.87088 ± 0.00247589 (W 0.00413599) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 23.2672 ± 0.0327972 (W 0.02535) | 23.71 ± 0.0251652 (W 0.0302343) | 48.7966 ± 0.162873 (W 0.0322955) | · | 🔵 colmat 2.10× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 26.7098 ± 0.0252085 (W 0.087956) | 27.2873 ± 0.0657903 (W 0.0643202) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.361169 ± 0.00206955 (W 0.00180231) | 0.428586 ± 0.00230785 (W 0.00208043) | 0.479823 ± 0.000769037 (W 0.00193375) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.33× faster than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 10.4429 ± 0.00192849 (W 0.00954039) | 10.9964 ± 0.0149679 (W 0.00776971) | 18.3572 ± 0.12435 (W 0.0101656) | 🔵 colmat 1.05× faster than `faer` | 🔵 colmat 1.76× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 1.04299 ± 0.00713777 (W 0.0013065) | 1.3552 ± 0.0043658 (W 0.00357466) | 1.79447 ± 0.00253413 (W 0.00321572) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.03779 ± 0.00369904 (W 0.00138198) | 1.36519 ± 0.00565721 (W 0.00281518) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 31.8047 ± 0.0791927 (W 0.0290933) | 35.5869 ± 0.0136 (W 0.0525537) | 71.6596 ± 0.40691 (W 0.0801434) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 2.25× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 31.9131 ± 0.126709 (W 0.0751046) | 35.6675 ± 0.311753 (W 0.0617465) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.542391 ± 0.000771572 (W 0.00274124) | 0.829764 ± 0.0019379 (W 0.00244297) | 0.773288 ± 0.00201483 (W 0.00290169) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.541662 ± 0.0013713 (W 0.00235706) | 0.827847 ± 0.00178405 (W 0.00207941) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 14.6018 ± 0.00725762 (W 0.0209037) | 19.37 ± 0.0257723 (W 0.0147922) | 25.8021 ± 0.00768397 (W 0.0280324) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 14.6065 ± 0.0263113 (W 0.00895853) | 19.3246 ± 0.0155609 (W 0.0198088) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.259202 ± 0.00279552 (W 0.00122985) | 0.522925 ± 0.00383573 (W 0.0055796) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.272181 ± 0.000775706 (W 0.00133812) | 0.483005 ± 0.00340805 (W 0.00396912) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 8.65367 ± 0.0501965 (W 0.0369889) | 10.2083 ± 0.0114002 (W 0.0177791) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.16261 ± 0.0122957 (W 0.0223182) | 9.62037 ± 0.0281159 (W 0.0141132) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.101976 ± 0.00546131 (W 0.00180172) | 0.100387 ± 0.00164865 (W 0.00092637) | 0.283702 ± 0.00499757 (W 0.00489284) | · | 🔵 colmat 2.78× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.102827 ± 0.00148998 (W 0.00211764) | 0.101447 ± 0.00227809 (W 0.00219249) | 0.205444 ± 0.000721388 (W 0.00076378) | · | 🔵 colmat 2.00× faster than `lapack-mkl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.52572 ± 0.0360769 (W 0.0275077) | 4.38585 ± 0.0789333 (W 0.0471214) | 9.49203 ± 0.232294 (W 0.0163509) | · | 🔵 colmat 2.10× faster than `lapack-mkl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.4832 ± 0.0435417 (W 0.0295176) | 4.35327 ± 0.0278714 (W 0.0299397) | 8.41093 ± 0.0118441 (W 0.010334) | · | 🔵 colmat 1.88× faster than `lapack-mkl` |  |
| `lu/1024x64/b1` | 0.109527 ± 0.000424771 (W 0.000588957) | 0.209973 ± 0.000710779 (W 0.0017166) | 0.163505 ± 6.37207e-05 (W 0.000621627) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 1.49× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.128737 ± 0.000384474 (W 0.000489435) | 0.229472 ± 0.00184532 (W 0.00186249) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.98309 ± 0.00450813 (W 0.00268011) | 3.62751 ± 0.00741587 (W 0.013802) | 7.1483 ± 0.0122744 (W 0.00356718) | 🔵 colmat 1.83× faster than `faer` | 🔵 colmat 3.60× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 2.19705 ± 0.0105111 (W 0.00664622) | 3.85154 ± 0.00968319 (W 0.0121805) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.393648 ± 0.000196571 (W 0.00189925) | 0.607172 ± 0.00180968 (W 0.00407614) | 0.433445 ± 0.000465003 (W 0.002295) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.57913 ± 0.00436337 (W 0.00231489) | 0.68646 ± 0.0680978 (W 0.017936) | - | 🔵 colmat 1.19× faster than `faer` (marginal: spread 0.68 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 9.19671 ± 0.0129369 (W 0.0202047) | 11.7765 ± 0.0273089 (W 0.0279014) | 15.2015 ± 0.111626 (W 0.0499068) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 1.65× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 12.7446 ± 0.0488007 (W 0.0158567) | 12.5487 ± 0.0122213 (W 0.017818) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.303771 ± 0.00132811 (W 0.00171572) | 0.520594 ± 0.00399145 (W 0.00332563) | 0.226152 ± 0.00215621 (W 0.00207525) | 🔵 colmat 1.71× faster than `faer` | 🔴 colmat 1.34× slower than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 5.53225 ± 0.00748835 (W 0.00796048) | 8.12941 ± 0.0338351 (W 0.0106229) | 6.68784 ± 0.0570894 (W 0.0333477) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 0.769258 ± 0.00438843 (W 0.0022461) | 1.32239 ± 0.00261528 (W 0.00335924) | 1.12751 ± 0.0040638 (W 0.00301814) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 1.47× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.778718 ± 0.00539088 (W 0.00189122) | 1.33085 ± 0.000771341 (W 0.00547086) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 14.9892 ± 0.0385782 (W 0.0234927) | 21.3079 ± 0.027631 (W 0.0485274) | 29.2455 ± 0.349368 (W 0.0687282) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 15.1315 ± 0.0246755 (W 0.0338453) | 21.4232 ± 0.054286 (W 0.123434) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.498096 ± 0.000872656 (W 0.00221011) | 0.931733 ± 0.00179657 (W 0.001716) | 0.532302 ± 0.0078552 (W 0.00328511) | 🔵 colmat 1.87× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.498383 ± 0.00280507 (W 0.00187334) | 0.933882 ± 0.00123538 (W 0.00392791) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 9.10754 ± 0.0203239 (W 0.0176857) | 15.4064 ± 0.0357268 (W 0.019357) | 12.9888 ± 0.0650338 (W 0.0461572) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 9.1188 ± 0.0218154 (W 0.0167268) | 15.4708 ± 0.0704872 (W 0.0302495) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.244823 ± 0.00728508 (W 0.00764034) | 0.521772 ± 0.00745038 (W 0.00339819) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.267601 ± 0.00286198 (W 0.00609645) | 0.483114 ± 0.00197501 (W 0.00492551) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 5.1989 ± 0.00838624 (W 0.0307674) | 6.91729 ± 0.0687532 (W 0.026666) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 5.10108 ± 0.00601923 (W 0.0450929) | 6.73032 ± 0.0810592 (W 0.0239217) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.071978 ± 0.00161299 (W 0.00103742) | 0.100457 ± 0.00117475 (W 0.00441392) | 0.238483 ± 0.00456724 (W 0.00476165) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 3.31× faster than `lapack-mkl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.074338 ± 0.00223699 (W 0.00116674) | 0.098777 ± 0.00423876 (W 0.0039218) | 0.141786 ± 0.000254231 (W 0.00208482) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 1.91× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.10248 ± 0.239935 (W 0.334067) | 2.70016 ± 0.264797 (W 0.295323) | 6.31788 ± 0.299803 (W 0.0245108) | 🔴 colmat 1.52× slower than `faer` | 🔵 colmat 1.54× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.23681 ± 0.0437182 (W 0.144571) | 3.14155 ± 0.0308863 (W 0.288999) | 5.50437 ± 0.0379474 (W 0.0148293) | 🔴 colmat 1.35× slower than `faer` | 🔵 colmat 1.30× faster than `lapack-mkl` |  |
| `lu/1024x64/b1` | 0.107686 ± 0.000962601 (W 0.000626028) | 0.210604 ± 0.000727177 (W 0.00204334) | 0.163095 ± 0.000225019 (W 0.00210557) | 🔵 colmat 1.96× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.126786 ± 0.000716961 (W 0.000794468) | 0.229353 ± 0.00176584 (W 0.0016902) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.71044 ± 0.0168761 (W 0.0035609) | 3.33866 ± 0.010381 (W 0.00595202) | 1.3775 ± 0.00969364 (W 0.00504582) | 🔵 colmat 1.95× faster than `faer` | 🔴 colmat 1.24× slower than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 2.00013 ± 0.00786968 (W 0.0043776) | 3.62905 ± 0.0103844 (W 0.0109193) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.343509 ± 0.00126726 (W 0.00207082) | 0.68149 ± 0.00188856 (W 0.00358199) | 0.310881 ± 0.00146623 (W 0.00434004) | 🔵 colmat 1.98× faster than `faer` | 🔴 colmat 1.10× slower than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.538645 ± 0.00291737 (W 0.0024587) | 0.715936 ± 0.00275094 (W 0.0204548) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 6.78567 ± 0.00957371 (W 0.019536) | 13.1391 ± 0.0293721 (W 0.0571845) | 9.30496 ± 0.0324112 (W 0.0244058) | 🔵 colmat 1.94× faster than `faer` | 🔵 colmat 1.37× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 10.6501 ± 0.0488956 (W 0.0464975) | 14.142 ± 0.367411 (W 0.324829) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.287641 ± 0.00241645 (W 0.00150068) | 0.581663 ± 0.00450447 (W 0.00355635) | 0.165705 ± 0.0011537 (W 0.00117197) | 🔵 colmat 2.02× faster than `faer` | 🔴 colmat 1.74× slower than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 4.50683 ± 0.0112969 (W 0.0139045) | 9.15127 ± 0.0181479 (W 0.0280823) | 3.76362 ± 0.00495602 (W 0.0183237) | 🔵 colmat 2.03× faster than `faer` | 🔴 colmat 1.20× slower than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 0.731079 ± 0.0013928 (W 0.00340524) | 1.38828 ± 0.00273082 (W 0.00618338) | 1.02552 ± 0.00104946 (W 0.00325059) | 🔵 colmat 1.90× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.746535 ± 0.00151857 (W 0.00296111) | 1.39547 ± 0.00423254 (W 0.00576347) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 12.4439 ± 0.0257853 (W 0.0297141) | 23.0499 ± 0.0177483 (W 0.0380231) | 22.0479 ± 0.532166 (W 0.053211) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 12.551 ± 0.053605 (W 0.0325726) | 23.2043 ± 0.897402 (W 2.7) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.485005 ± 0.0024074 (W 0.00329171) | 0.996011 ± 0.00069642 (W 0.0029419) | 0.465844 ± 0.000621435 (W 0.00268243) | 🔵 colmat 2.05× faster than `faer` | · |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.488173 ± 0.00100247 (W 0.00144788) | 0.994287 ± 0.00309911 (W 0.00265293) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 7.98585 ± 0.018371 (W 0.00848542) | 16.6128 ± 0.0328243 (W 0.0574388) | 9.74914 ± 0.501574 (W 0.0173115) | 🔵 colmat 2.08× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 8.00718 ± 0.00975068 (W 0.0157114) | 16.6374 ± 0.0354546 (W 0.0499325) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.24003 ± 0.00775644 (W 0.00311082) | 1.75297 ± 0.0247814 (W 0.0129308) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.15745 ± 0.00432759 (W 0.00129454) | 1.18841 ± 0.00165832 (W 0.00259657) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 69.4911 ± 0.336863 (W 0.335048) | 111.267 ± 0.498373 (W 0.123187) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 66.9994 ± 0.683554 (W 0.466942) | 73.1433 ± 2.74322 (W 0.0427892) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.772936 ± 0.00858294 (W 0.0016639) | 0.774458 ± 0.0157201 (W 0.00135808) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.773356 ± 0.00810111 (W 0.00214314) | 0.797165 ± 0.0145433 (W 0.00221359) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 49.153 ± 0.237381 (W 0.0946568) | 48.7497 ± 0.015265 (W 0.0717901) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 49.0046 ± 0.312995 (W 0.128153) | 48.8741 ± 0.0662461 (W 0.260731) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.468816 ± 0.00180927 (W 0.00197241) | 0.915062 ± 0.0568749 (W 0.00770665) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.483057 ± 0.00124929 (W 0.0019956) | 0.529544 ± 0.00214671 (W 0.000988989) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 18.1153 ± 0.0937626 (W 0.0386972) | 50.1179 ± 0.0795447 (W 0.140351) | - | 🔵 colmat 2.77× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 20.0254 ± 0.125251 (W 0.145177) | 25.9799 ± 0.0630828 (W 0.115668) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.97932 ± 0.0115448 (W 0.000960257) | 2.87718 ± 0.130788 (W 0.0108757) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 2.33051 ± 0.00323195 (W 0.00236911) | 2.43915 ± 0.00707749 (W 0.00464933) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 94.8693 ± 2.69686 (W 0.855546) | 156.724 ± 1.62982 (W 0.854498) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 95.1113 ± 0.832309 (W 0.222196) | 102.162 ± 0.272852 (W 0.529277) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.15088 ± 0.0177816 (W 0.00141496) | 2.21996 ± 0.0752899 (W 0.0112161) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 39.8587 ± 0.066278 (W 0.0752043) | 98.1603 ± 3.02573 (W 1.45844) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/1024x64/b1` | 2.49874 ± 0.00266483 (W 0.00513213) | 2.83449 ± 0.00412558 (W 0.00264456) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.51309 ± 0.00523865 (W 0.00191905) | 2.87305 ± 0.00232755 (W 0.00181714) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 106.66 ± 0.214109 (W 0.149582) | 112.174 ± 0.27742 (W 0.206246) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 106.93 ± 0.182108 (W 0.0858032) | 112.606 ± 0.306754 (W 0.200246) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.2963 ± 0.00332559 (W 0.00247942) | 1.62598 ± 0.0081866 (W 0.002809) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.29719 ± 0.00392165 (W 0.00214916) | 1.64924 ± 0.00338043 (W 0.00209203) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 54.2786 ± 2.57919 (W 0.0453587) | 56.8892 ± 0.109931 (W 0.089514) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 49.9999 ± 2.69915 (W 0.124169) | 61.3939 ± 2.68798 (W 0.0594935) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.555501 ± 0.00464592 (W 0.001042) | 1.73253 ± 0.0250008 (W 0.0163443) | - | 🔵 colmat 3.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.6026 ± 0.00250696 (W 0.00327789) | 1.19219 ± 0.00806546 (W 0.00290889) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 20.3981 ± 0.110076 (W 0.0285602) | 55.9572 ± 0.069282 (W 0.0858663) | - | 🔵 colmat 2.74× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 22.0391 ± 0.113275 (W 0.140317) | 33.8405 ± 0.0131619 (W 0.0402875) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.245563 ± 0.00256216 (W 0.00182681) | 0.226063 ± 0.00578587 (W 0.00144521) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.245253 ± 0.00360568 (W 0.00253223) | 0.225723 ± 0.00568941 (W 0.00086097) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 13.492 ± 0.0212843 (W 0.0798087) | 13.0807 ± 0.111368 (W 0.0568329) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 13.4265 ± 0.0372998 (W 0.0796181) | 13.1406 ± 0.0502331 (W 0.069297) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.32836 ± 0.00154627 (W 0.00179301) | 0.860431 ± 0.0225158 (W 0.00921954) | - | 🔵 colmat 2.62× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.370229 ± 0.00269007 (W 0.0021045) | 0.452637 ± 0.00125226 (W 0.00318666) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.68351 ± 0.0715998 (W 0.0719257) | 41.4944 ± 0.0452916 (W 0.131438) | - | 🔵 colmat 4.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 10.4749 ± 0.0464088 (W 0.0554883) | 17.1799 ± 0.0393974 (W 0.0278893) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.885804 ± 0.00475587 (W 0.00350871) | 2.60498 ± 0.020685 (W 0.0350039) | - | 🔵 colmat 2.94× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.37198 ± 0.010232 (W 0.00422611) | 1.70523 ± 0.00390412 (W 0.00568273) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 31.0822 ± 0.183717 (W 0.156111) | 95.1149 ± 0.078349 (W 0.664328) | - | 🔵 colmat 3.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 40.9071 ± 0.0704795 (W 0.143619) | 51.039 ± 4.71022 (W 0.106327) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.642339 ± 0.00190335 (W 0.00254654) | 2.39162 ± 0.0615882 (W 0.02769) | - | 🔵 colmat 3.72× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 16.5059 ± 0.0488755 (W 0.0209084) | 80.0593 ± 0.234243 (W 0.456743) | - | 🔵 colmat 4.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.43401 ± 0.00167302 (W 0.00296548) | 2.08493 ± 0.00677667 (W 0.00584491) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.46453 ± 0.019407 (W 0.0034387) | 2.10576 ± 0.0104464 (W 0.00333573) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 45.0861 ± 0.0930563 (W 0.0648439) | 56.2762 ± 0.0516698 (W 0.0659025) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 45.4539 ± 0.0852943 (W 0.0785771) | 56.4322 ± 0.00933733 (W 0.124728) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.945281 ± 0.0180236 (W 0.00352846) | 1.45957 ± 0.00326023 (W 0.00464152) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.941738 ± 0.00218165 (W 0.00250032) | 1.46415 ± 0.0236469 (W 0.00453509) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 28.0946 ± 0.101936 (W 0.106806) | 39.4631 ± 2.52584 (W 0.0845808) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 28.0705 ± 0.0628938 (W 0.0580616) | 39.5103 ± 0.0546501 (W 0.0529015) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.448917 ± 0.00634206 (W 0.00140802) | 1.75797 ± 0.0253799 (W 0.018524) | - | 🔵 colmat 3.92× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.505155 ± 0.00239444 (W 0.00300134) | 1.18951 ± 0.0137938 (W 0.00371288) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 11.9168 ± 0.0268544 (W 0.0240487) | 46.8431 ± 0.283807 (W 0.104314) | - | 🔵 colmat 3.93× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 14.5012 ± 0.167376 (W 0.157677) | 26.4983 ± 0.0244864 (W 0.0252634) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.155765 ± 0.00155039 (W 0.0025456) | 0.189515 ± 0.000747416 (W 0.0102313) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.157905 ± 0.00355239 (W 0.00506059) | 0.186833 ± 0.0118562 (W 0.0126201) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 8.64924 ± 0.0720253 (W 0.143056) | 9.10571 ± 1.22576 (W 1.14928) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 8.66989 ± 0.101764 (W 0.136262) | 7.0716 ± 1.21472 (W 0.96971) | - | 🔴 colmat 1.23× slower than `faer` (marginal: spread 0.82 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.304441 ± 0.0181659 (W 0.00185285) | 0.883003 ± 0.0407395 (W 0.00507486) | - | 🔵 colmat 2.90× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.365219 ± 0.00448352 (W 0.00191115) | 0.443496 ± 0.00633204 (W 0.00265744) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.19473 ± 0.00775435 (W 0.0143434) | 39.9698 ± 0.152808 (W 0.103608) | - | 🔵 colmat 6.45× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 9.03111 ± 0.0224107 (W 0.160849) | 15.7685 ± 0.0223653 (W 0.0261817) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.739959 ± 0.00158623 (W 0.00386609) | 2.68228 ± 0.071696 (W 0.0411666) | - | 🔵 colmat 3.62× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.21382 ± 0.00593713 (W 0.0040727) | 1.32214 ± 1.29233 (W 0.195252) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 20.5351 ± 0.056231 (W 0.0551352) | 92.2918 ± 1.00828 (W 3.61261) | - | 🔵 colmat 4.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 32.6838 ± 0.348947 (W 0.151714) | 41.5051 ± 14.6519 (W 4.5031) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.59651 ± 0.00141416 (W 0.00401798) | 2.48766 ± 0.0132612 (W 0.0455089) | - | 🔵 colmat 4.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 11.3823 ± 0.0150157 (W 0.0210596) | 78.1838 ± 0.831453 (W 1.74406) | - | 🔵 colmat 6.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.27673 ± 0.00571578 (W 0.00222066) | 2.11006 ± 0.00638964 (W 0.00729224) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.30879 ± 0.0168984 (W 0.00385508) | 2.12504 ± 0.00761983 (W 0.00972709) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 35.967 ± 0.0432853 (W 0.0798946) | 55.333 ± 0.0181519 (W 0.411051) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 36.3771 ± 0.115281 (W 0.115421) | 55.7886 ± 1.6439 (W 5.35548) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.891895 ± 0.0228389 (W 0.00287415) | 1.49458 ± 0.00771954 (W 0.00486457) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.894443 ± 0.00334826 (W 0.00158019) | 1.4987 ± 0.00608178 (W 0.00886057) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 24.1148 ± 0.18515 (W 0.124527) | 40.0548 ± 3.79892 (W 2.7083) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 23.9595 ± 0.10328 (W 0.0942008) | 39.9514 ± 8.10923 (W 0.220459) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.07981 ± 0.00144464 (W 0.000946012) | 1.10959 ± 0.00202653 (W 0.00175671) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.08882 ± 0.0018966 (W 0.00184846) | 1.11308 ± 0.00297593 (W 0.00246523) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 67.4068 ± 0.340639 (W 0.117421) | 67.3821 ± 0.506931 (W 0.0829836) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 63.4303 ± 0.305667 (W 0.0440654) | 64.6023 ± 0.212261 (W 0.13611) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.809425 ± 0.00369667 (W 0.00106955) | 0.832931 ± 0.0128135 (W 0.00238191) | 1.91597 ± 0.00525291 (W 0.00837047) | · | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.808685 ± 0.00504096 (W 0.00097564) | 0.813095 ± 0.0118901 (W 0.000898799) | 2.25282 ± 0.00288313 (W 0.00483806) | · | 🔵 colmat 2.79× faster than `lapack-mkl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 48.4542 ± 0.0195683 (W 0.107182) | 48.7383 ± 0.0659623 (W 0.0983033) | 112.646 ± 1.33871 (W 0.139556) | · | 🔵 colmat 2.32× faster than `lapack-mkl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 48.5169 ± 0.0474354 (W 0.0586285) | 48.7684 ± 0.122708 (W 0.0613605) | 119.349 ± 0.285758 (W 0.186844) | · | 🔵 colmat 2.46× faster than `lapack-mkl` |  |
| `lu/1024x64/b1` | 0.392947 ± 0.00254085 (W 0.00293785) | 0.429248 ± 0.00195925 (W 0.00223568) | 0.63596 ± 0.0176297 (W 0.00220146) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.62× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.408769 ± 0.000269179 (W 0.00145262) | 0.452416 ± 0.0090311 (W 0.00134278) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 15.5412 ± 0.0875173 (W 0.0260798) | 16.1412 ± 0.124645 (W 0.0522894) | 30.7808 ± 0.0532796 (W 0.0476998) | · | 🔵 colmat 1.98× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 15.8191 ± 0.0534955 (W 0.0222758) | 16.5757 ± 0.0258669 (W 0.0447752) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.80702 ± 0.00240724 (W 0.00178497) | 1.92009 ± 0.00209936 (W 0.0016738) | 3.6731 ± 0.00603076 (W 0.0041302) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 2.03× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.22054 ± 0.00691186 (W 0.00257003) | 2.34175 ± 0.00822345 (W 0.00289382) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 82.6254 ± 0.0539231 (W 0.317854) | 84.0639 ± 0.150414 (W 0.195505) | 182.004 ± 0.189373 (W 0.159369) | · | 🔵 colmat 2.20× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 90.8292 ± 0.520428 (W 0.210762) | 92.5155 ± 0.232848 (W 0.0746594) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.965667 ± 0.00188172 (W 0.000891291) | 1.11102 ± 0.00343539 (W 0.00230289) | 1.46271 ± 0.0027815 (W 0.00270024) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 33.3589 ± 0.0704864 (W 0.0238889) | 34.899 ± 0.0523478 (W 0.033186) | 64.3129 ± 0.0551964 (W 0.0725166) | · | 🔵 colmat 1.93× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 2.42065 ± 0.00424802 (W 0.00235407) | 2.77599 ± 0.00305116 (W 0.00274674) | 5.52804 ± 0.0184396 (W 0.00881536) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 2.28× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.43856 ± 0.00332412 (W 0.0035811) | 2.79982 ± 0.0080281 (W 0.00216069) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 98.1664 ± 0.226795 (W 0.107374) | 103.216 ± 0.175482 (W 0.0831108) | 230.278 ± 0.685237 (W 0.390242) | 🔵 colmat 1.05× faster than `faer` | 🔵 colmat 2.35× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 98.6641 ± 0.240993 (W 0.0937586) | 103.584 ± 0.155358 (W 0.106938) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.21881 ± 0.00122455 (W 0.00204845) | 1.55139 ± 0.00200756 (W 0.0023778) | 2.38227 ± 0.0178131 (W 0.00269875) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.22154 ± 0.00372622 (W 0.00268104) | 1.57387 ± 0.00273738 (W 0.00262992) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 41.5746 ± 0.285414 (W 0.0794386) | 47.9008 ± 0.14076 (W 0.0417198) | 89.5391 ± 0.520631 (W 0.117497) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 2.15× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 41.3407 ± 0.102915 (W 0.0537161) | 47.9636 ± 0.0865003 (W 0.034337) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.493705 ± 0.00511749 (W 0.00276507) | 1.10506 ± 0.0020926 (W 0.00319938) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.531234 ± 0.00317259 (W 0.00129332) | 1.11083 ± 0.00284982 (W 0.00274015) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 19.3254 ± 0.0547798 (W 0.0223579) | 21.0368 ± 0.111819 (W 0.0477728) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 19.1229 ± 0.057856 (W 0.0272064) | 20.86 ± 0.0862965 (W 0.0577829) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.28425 ± 0.0172045 (W 0.0023477) | 0.280792 ± 0.0136772 (W 0.0013132) | 0.756818 ± 0.0138085 (W 0.00347579) | · | 🔵 colmat 2.66× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.286291 ± 0.0109872 (W 0.00250795) | 0.263781 ± 0.0179392 (W 0.00152824) | 0.717066 ± 0.00418735 (W 0.00224099) | · | 🔵 colmat 2.50× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 15.0073 ± 0.153266 (W 0.0270056) | 14.904 ± 0.0223596 (W 0.0331849) | 40.0164 ± 1.85683 (W 0.0737721) | · | 🔵 colmat 2.67× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 15.0871 ± 0.132293 (W 0.0453323) | 15.1533 ± 0.00918064 (W 0.0258082) | 30.9616 ± 0.0341574 (W 0.00866714) | · | 🔵 colmat 2.05× faster than `lapack-mkl` |  |
| `lu/1024x64/b1` | 0.266872 ± 0.000344423 (W 0.00143572) | 0.355978 ± 0.00451383 (W 0.00251427) | 0.616909 ± 0.0141044 (W 0.00211189) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 2.31× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.308521 ± 0.00235829 (W 0.00147668) | 0.393158 ± 0.00336575 (W 0.00255629) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.13732 ± 0.00962177 (W 0.0134826) | 7.74392 ± 0.0128851 (W 0.0155099) | 31.0103 ± 0.0338897 (W 0.0625846) | 🔵 colmat 1.26× faster than `faer` | 🔵 colmat 5.05× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 6.76017 ± 0.0205692 (W 0.0179714) | 8.32631 ± 0.038866 (W 0.0221805) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.869325 ± 0.00953391 (W 0.00330627) | 1.1764 ± 0.00680538 (W 0.00374969) | 1.50478 ± 0.00171545 (W 0.00367089) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.27646 ± 0.00811749 (W 0.00573479) | 1.58823 ± 0.0175461 (W 0.0197201) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 27.7501 ± 0.0730953 (W 0.0329696) | 32.4587 ± 0.0558911 (W 0.0504787) | 49.8944 ± 0.172292 (W 0.1252) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.80× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 36.9941 ± 0.052231 (W 0.140204) | 42.2419 ± 4.74903 (W 0.145901) | - | 🔵 colmat 1.14× faster than `faer` (marginal: spread 0.91 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.60141 ± 0.00627107 (W 0.00176868) | 0.913089 ± 0.00724561 (W 0.00311465) | 0.758949 ± 0.00228563 (W 0.0248845) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 14.1214 ± 0.0113004 (W 0.0152267) | 18.8144 ± 0.0103556 (W 0.0230178) | 19.1429 ± 0.0831375 (W 0.152952) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1` | 1.3802 ± 0.00139569 (W 0.00419683) | 2.02256 ± 0.00500132 (W 0.00779042) | 3.3694 ± 0.00604765 (W 0.00919098) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 2.44× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.41537 ± 0.012992 (W 0.00326728) | 2.03495 ± 0.000912978 (W 0.0057911) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 36.7096 ± 0.00892834 (W 0.0287698) | 47.5727 ± 0.0664058 (W 0.0707329) | 81.0091 ± 0.181219 (W 0.154085) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 2.21× faster than `lapack-mkl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 37.0973 ± 0.0452413 (W 0.0351363) | 47.71 ± 0.149651 (W 0.0755234) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.887174 ± 0.00165031 (W 0.00257616) | 1.40046 ± 0.00821071 (W 0.0053548) | 1.67323 ± 0.0145343 (W 0.0026634) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.89× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.89011 ± 0.00301106 (W 0.00153014) | 1.40891 ± 0.00167507 (W 0.00463575) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.5407 ± 0.0222864 (W 0.0161978) | 30.5281 ± 0.0254504 (W 0.047817) | 31.0554 ± 0.146207 (W 0.131824) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 1.59× faster than `lapack-mkl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.577 ± 0.0135844 (W 0.0200677) | 30.564 ± 2.14316 (W 0.183022) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.404776 ± 0.00132176 (W 0.00154256) | 1.10807 ± 0.00483739 (W 0.00256132) | - | 🔵 colmat 2.74× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.449055 ± 0.0184509 (W 0.00341606) | 1.11077 ± 0.00326233 (W 0.00267325) | - | 🔵 colmat 2.47× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 11.3284 ± 0.029146 (W 0.0180139) | 13.4014 ± 0.0709331 (W 0.0221727) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.6267 ± 0.00679471 (W 0.0160137) | 13.5811 ± 0.288446 (W 0.0320476) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.192174 ± 0.00355308 (W 0.0023988) | 0.222382 ± 0.0124644 (W 0.0134588) | 0.388619 ± 0.00194436 (W 0.00402842) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-mkl` | `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.195043 ± 0.00395482 (W 0.00162152) | 0.222912 ± 0.000524189 (W 0.0104234) | 0.786707 ± 0.00270603 (W 0.0035313) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 4.03× faster than `lapack-mkl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.2336 ± 0.549077 (W 0.0176999) | 10.6291 ± 1.13301 (W 0.981552) | 17.7225 ± 0.472288 (W 0.0503543) | · | 🔵 colmat 1.45× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.27 ± 0.604603 (W 0.0204887) | 8.74793 ± 1.09103 (W 1.02458) | 17.209 ± 0.0721665 (W 0.0379962) | 🔴 colmat 1.40× slower than `faer` | 🔵 colmat 1.40× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/1024x64/b1` | 0.258461 ± 0.00153483 (W 0.000952111) | 0.342331 ± 0.00215469 (W 0.00103982) | 0.616551 ± 0.00811874 (W 0.00247075) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 2.39× faster than `lapack-mkl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.29404 ± 0.00169425 (W 0.00167944) | 0.375567 ± 0.00836422 (W 0.00243373) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.05799 ± 0.0252491 (W 0.00744472) | 6.32109 ± 0.00722063 (W 0.00944534) | 31.3767 ± 0.51068 (W 0.0399663) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 6.20× faster than `lapack-mkl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 5.61752 ± 0.0188154 (W 0.00915038) | 6.94684 ± 0.0176552 (W 0.0129801) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.70531 ± 0.015113 (W 0.00312056) | 1.20236 ± 0.00649841 (W 0.00992531) | 1.27022 ± 0.00211926 (W 0.00192889) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 1.80× faster than `lapack-mkl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.10151 ± 0.00645749 (W 0.00430139) | 1.27491 ± 1.25095 (W 0.0326568) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 18.8587 ± 0.0444277 (W 0.0510421) | 30.8302 ± 0.416831 (W 0.227151) | 27.3801 ± 0.152155 (W 0.0643019) | 🔵 colmat 1.63× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-mkl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 28.1577 ± 0.304029 (W 0.0756938) | 32.4942 ± 12.109 (W 5.08386) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.539082 ± 0.00741704 (W 0.00361777) | 0.951242 ± 0.00213669 (W 0.00395897) | 0.485764 ± 0.00265572 (W 0.00342552) | 🔵 colmat 1.76× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-mkl` |  |
| `qr_factor/4096x256/b1` | 10.2681 ± 1.31574 (W 0.0212336) | 19.0865 ± 0.0729986 (W 0.129763) | 10.5514 ± 0.130033 (W 0.0259584) | 🔵 colmat 1.86× faster than `faer` | · | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/1024x64/b1` | 1.21354 ± 0.0074998 (W 0.00223051) | 2.0541 ± 0.00424319 (W 0.00611562) | 2.70223 ± 0.00275126 (W 0.00746231) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 2.23× faster than `lapack-mkl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.23864 ± 0.00344981 (W 0.00276271) | 2.06691 ± 0.00189296 (W 0.00561785) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 27.5388 ± 0.0695269 (W 0.0568581) | 46.5553 ± 2.30641 (W 5.46254) | 44.2028 ± 0.129746 (W 0.103616) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 27.825 ± 0.0757979 (W 0.0652501) | 47.2437 ± 1.62914 (W 4.76668) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.842955 ± 0.00277085 (W 0.00281968) | 1.45073 ± 0.00815434 (W 0.00634824) | 1.41043 ± 0.00607298 (W 0.00206168) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-mkl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.842882 ± 0.00205432 (W 0.00262201) | 1.44454 ± 0.00265597 (W 0.00353527) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 15.458 ± 0.015622 (W 0.0130722) | 31.0678 ± 0.0307109 (W 0.117025) | 19.7515 ± 0.822102 (W 0.115338) | 🔵 colmat 2.01× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 15.4298 ± 0.0206677 (W 0.0129542) | 31.2222 ± 13.3154 (W 0.214764) | - | 🔵 colmat 2.02× faster than `faer` (marginal: spread 0.84 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.394919 ± 0.00166891 (W 0.00523486) | 0.600593 ± 0.00808689 (W 0.00234891) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.318501 ± 0.0063373 (W 0.00135146) | 0.336438 ± 0.00452465 (W 0.0065088) | - | 🔵 colmat 1.06× faster than `faer` (marginal: spread 0.61 of the difference) | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 15.9476 ± 0.063457 (W 0.0374788) | 30.8431 ± 0.065904 (W 0.0513266) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 14.4489 ± 0.0452581 (W 0.0131894) | 17.434 ± 0.0796172 (W 0.20494) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.106607 ± 0.000715472 (W 0.000734486) | 0.103316 ± 0.000674905 (W 0.00115338) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.106617 ± 0.000435512 (W 0.000632359) | 0.104117 ± 0.000805067 (W 0.000555971) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 6.20337 ± 0.0300063 (W 0.0341048) | 6.14789 ± 0.030978 (W 0.00765749) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 6.18006 ± 0.0876212 (W 0.00992771) | 6.13279 ± 0.00499064 (W 0.00740516) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.199493 ± 0.000334351 (W 0.00173409) | 0.503073 ± 0.00098265 (W 0.0021236) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.156256 ± 0.00065533 (W 0.000971718) | 0.191074 ± 0.00322864 (W 0.00375385) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.91833 ± 0.0279409 (W 0.00512829) | 16.873 ± 0.0348071 (W 0.0175164) | - | 🔵 colmat 4.31× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 3.947 ± 0.0717455 (W 0.0185483) | 5.79512 ± 0.00433756 (W 0.00879972) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.555134 ± 0.00134713 (W 0.00152223) | 1.09596 ± 0.00310436 (W 0.00187174) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.580323 ± 0.00385435 (W 0.00218412) | 0.64044 ± 0.00139224 (W 0.00234808) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 16.17 ± 0.233566 (W 0.0975304) | 38.0834 ± 0.0308896 (W 0.0453251) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 16.4691 ± 0.158619 (W 0.0343399) | 18.4603 ± 0.062749 (W 0.0177478) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.422347 ± 0.000438952 (W 0.00197075) | 0.97212 ± 0.00114252 (W 0.00112177) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 9.16754 ± 0.142361 (W 0.0314793) | 31.1195 ± 0.00671693 (W 0.0291624) | - | 🔵 colmat 3.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.795295 ± 0.00689017 (W 0.00389414) | 0.927782 ± 0.00563061 (W 0.00163714) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.797953 ± 0.00601654 (W 0.00469766) | 1.01961 ± 0.0018836 (W 0.00126761) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 21.0639 ± 0.0815554 (W 0.0143409) | 24.5463 ± 0.553321 (W 0.0380423) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 21.161 ± 0.0852178 (W 0.014899) | 24.5301 ± 0.528687 (W 0.021588) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.469276 ± 0.00374334 (W 0.00311424) | 0.706156 ± 0.00245115 (W 0.00121971) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.466154 ± 0.000933005 (W 0.00279715) | 0.704146 ± 0.00193594 (W 0.00235083) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 11.6601 ± 0.0245374 (W 0.0121269) | 15.074 ± 0.577472 (W 0.0108309) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 11.6582 ± 0.0243572 (W 0.00688944) | 15.1352 ± 0.544617 (W 0.0118358) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.248611 ± 0.0027448 (W 0.00136413) | 0.587473 ± 0.0195076 (W 0.00331884) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.267312 ± 0.000819476 (W 0.00153358) | 0.33221 ± 0.000362493 (W 0.00449709) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 5.15284 ± 0.0133483 (W 0.0228347) | 18.3272 ± 0.00959852 (W 0.0328743) | - | 🔵 colmat 3.56× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 5.72728 ± 0.0153668 (W 0.0219101) | 8.10939 ± 0.0278742 (W 0.00775073) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.048769 ± 0.00120994 (W 0.00123471) | 0.043829 ± 0.000467076 (W 0.000771613) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.049368 ± 0.000578291 (W 0.00135554) | 0.044129 ± 0.000443058 (W 0.000973851) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 1.69845 ± 0.0259445 (W 0.00770114) | 1.66115 ± 0.0323507 (W 0.00629394) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 1.7073 ± 0.0323843 (W 0.006559) | 1.65565 ± 0.0317875 (W 0.0111482) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.138097 ± 0.000245406 (W 0.00188518) | 0.517012 ± 0.000841215 (W 0.00645527) | - | 🔵 colmat 3.74× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.165464 ± 0.000678577 (W 0.000547675) | 0.199623 ± 0.00216901 (W 0.00367812) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.10688 ± 0.0181898 (W 0.0178662) | 15.6187 ± 0.0529357 (W 0.0493972) | - | 🔵 colmat 7.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.6436 ± 0.0113675 (W 0.0196109) | 4.80965 ± 0.0111924 (W 0.0296423) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.352159 ± 0.000457203 (W 0.00347111) | 1.50866 ± 0.00913859 (W 0.0273422) | - | 🔵 colmat 4.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.458095 ± 0.00169845 (W 0.00243936) | 0.618039 ± 0.00535416 (W 0.0131263) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 6.30515 ± 0.0641654 (W 0.0219535) | 35.007 ± 0.0902086 (W 0.205733) | - | 🔵 colmat 5.55× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 8.52768 ± 0.0170906 (W 0.0267874) | 11.1886 ± 0.0787273 (W 0.0294721) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.286082 ± 0.00166054 (W 0.00121437) | 1.39418 ± 0.0120616 (W 0.0216479) | - | 🔵 colmat 4.87× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 4.35274 ± 0.0408011 (W 0.0148023) | 33.0077 ± 0.177063 (W 0.139151) | - | 🔵 colmat 7.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.634161 ± 0.00412933 (W 0.00296486) | 0.971342 ± 0.00276067 (W 0.00139028) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.641691 ± 0.00223963 (W 0.00171803) | 0.976797 ± 0.0195093 (W 0.00185257) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 11.7692 ± 0.0748748 (W 0.0301554) | 16.3778 ± 0.0505324 (W 0.0452276) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 11.7177 ± 0.0879874 (W 0.0160884) | 16.4474 ± 0.0672088 (W 0.0405568) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.446238 ± 0.00168689 (W 0.00390714) | 0.818076 ± 0.00361024 (W 0.00372488) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.445265 ± 0.0055241 (W 0.00285847) | 0.818545 ± 0.00426905 (W 0.00479455) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 8.75972 ± 0.0760349 (W 0.0102304) | 13.3463 ± 0.0328239 (W 0.0198019) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 8.74402 ± 0.344262 (W 0.0185028) | 13.406 ± 0.515954 (W 0.0442735) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.239642 ± 0.00575027 (W 0.00622751) | 0.594302 ± 0.000590643 (W 0.00450419) | - | 🔵 colmat 2.48× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.276002 ± 0.00136116 (W 0.00614279) | 0.334511 ± 0.00105395 (W 0.00341086) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 3.46273 ± 0.00393458 (W 0.0123115) | 16.336 ± 0.0343606 (W 0.0386805) | - | 🔵 colmat 4.72× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 4.34691 ± 0.0158047 (W 0.0288922) | 6.81769 ± 0.0215619 (W 0.0152938) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.046898 ± 0.000735686 (W 0.00120511) | 0.050349 ± 0.00158515 (W 0.00165409) | - | 🔵 colmat 1.07× faster than `faer` (marginal: spread 0.67 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.046229 ± 0.000665124 (W 0.000930871) | 0.051548 ± 0.000881266 (W 0.00219145) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 1.03311 ± 0.0141913 (W 0.0122802) | 1.31654 ± 0.0037943 (W 0.105993) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 1.02093 ± 0.0107067 (W 0.019577) | 1.30533 ± 0.105963 (W 0.0926188) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.124056 ± 0.000705209 (W 0.000879301) | 0.545344 ± 0.00400096 (W 0.00520147) | - | 🔵 colmat 4.40× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.165534 ± 0.00067283 (W 0.000541222) | 0.198814 ± 0.00122372 (W 0.00477601) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.68164 ± 0.00889534 (W 0.0087847) | 15.7271 ± 0.0453925 (W 0.0223186) | - | 🔵 colmat 9.35× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.56249 ± 0.0236366 (W 0.0217342) | 4.68645 ± 0.0219498 (W 0.0125241) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.326379 ± 0.000829 (W 0.00280351) | 1.53553 ± 0.017359 (W 0.0249557) | - | 🔵 colmat 4.70× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.437295 ± 0.000794671 (W 0.00242719) | 0.665978 ± 0.00436638 (W 0.0129661) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 4.66103 ± 0.0258414 (W 0.0165574) | 35.06 ± 0.115299 (W 0.113453) | - | 🔵 colmat 7.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 6.99349 ± 0.0565608 (W 0.019534) | 12.2545 ± 0.169606 (W 1.3042) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.274052 ± 0.000923021 (W 0.00123143) | 1.45606 ± 0.0239872 (W 0.0265446) | - | 🔵 colmat 5.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 3.33514 ± 0.0135686 (W 0.00983277) | 33.1744 ± 0.0960928 (W 0.135614) | - | 🔵 colmat 9.95× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.623571 ± 0.0023165 (W 0.00383291) | 1.05069 ± 0.00501163 (W 0.00293168) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.624982 ± 0.00296381 (W 0.00177456) | 1.05444 ± 0.00310735 (W 0.00381992) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 9.92522 ± 0.0488025 (W 0.0126502) | 17.624 ± 0.0518245 (W 0.055752) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 10.013 ± 0.55133 (W 0.0209677) | 17.7844 ± 0.482493 (W 0.0537297) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.436197 ± 0.00230121 (W 0.00461108) | 0.879704 ± 0.00347775 (W 0.00448199) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.436967 ± 0.00107516 (W 0.00246277) | 0.883643 ± 0.00633645 (W 0.00326217) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 7.8985 ± 0.00794022 (W 0.0103068) | 14.3125 ± 0.0358766 (W 0.0589435) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 7.90871 ± 0.0345442 (W 0.0117141) | 14.3191 ± 0.594329 (W 0.0388729) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.264081 ± 0.00145318 (W 0.00132157) | 0.312659 ± 0.000443638 (W 0.00127883) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.250853 ± 0.00761695 (W 0.0020384) | 0.299111 ± 0.00105675 (W 0.0018383) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 14.7859 ± 0.0353161 (W 0.00569402) | 15.7909 ± 0.00667692 (W 0.0117216) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 13.2482 ± 0.0560838 (W 0.00495449) | 14.2157 ± 0.0133414 (W 0.0124417) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.122536 ± 0.000648445 (W 0.000474521) | 0.121586 ± 0.000700571 (W 0.000491096) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.122127 ± 6.55515e-05 (W 0.00144468) | 0.121147 ± 0.00133002 (W 0.00167799) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 6.34209 ± 0.00895813 (W 0.00657674) | 6.37141 ± 0.0827301 (W 0.00511486) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 6.3397 ± 0.00855986 (W 0.00860166) | 6.36985 ± 0.080584 (W 0.00608415) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.085167 ± 0.000735847 (W 0.000757406) | 0.147716 ± 0.00133045 (W 0.00179476) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.088968 ± 0.000185689 (W 0.000883504) | 0.152935 ± 0.00113393 (W 0.00137315) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.55219 ± 0.0133156 (W 0.00342487) | 3.49399 ± 0.0657619 (W 0.0110734) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.60974 ± 0.0100672 (W 0.00406512) | 3.61084 ± 0.0691466 (W 0.00721575) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.397417 ± 0.00131084 (W 0.00151297) | 0.496645 ± 0.00188076 (W 0.00192992) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.498303 ± 0.00239194 (W 0.00243606) | 0.598183 ± 0.000863781 (W 0.00193172) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 13.3889 ± 0.0647567 (W 0.0145125) | 14.4247 ± 0.0536081 (W 0.0138354) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 15.0033 ± 0.133696 (W 0.00875337) | 16.1239 ± 0.0904104 (W 0.00934455) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.265991 ± 0.000359342 (W 0.00116708) | 0.368817 ± 0.00103019 (W 0.00203626) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 6.53825 ± 0.0164016 (W 0.00431394) | 7.55422 ± 0.0599633 (W 0.00742453) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.764977 ± 0.00187266 (W 0.00231492) | 0.923792 ± 0.00257857 (W 0.00110738) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.766226 ± 0.00334555 (W 0.00211775) | 1.01521 ± 0.00177337 (W 0.00243876) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 18.9874 ± 0.153502 (W 0.01045) | 22.0256 ± 0.0684471 (W 0.0138844) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 19.0268 ± 0.100996 (W 0.0282378) | 21.9423 ± 0.0947171 (W 0.0195138) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.428775 ± 0.000725935 (W 0.00167845) | 0.67416 ± 0.00113352 (W 0.00159925) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.430116 ± 0.00141544 (W 0.00137244) | 0.673037 ± 0.00108449 (W 0.00160033) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 9.46204 ± 0.0627191 (W 0.00458737) | 12.968 ± 0.0464601 (W 0.0207674) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 9.43619 ± 0.10715 (W 0.00817018) | 12.9291 ± 0.0838358 (W 0.00629451) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.201494 ± 0.0005003 (W 0.00186302) | 0.312181 ± 0.00113838 (W 0.00162478) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.208453 ± 0.00734942 (W 0.00811971) | 0.297711 ± 0.000512075 (W 0.00203827) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 4.88308 ± 0.00684732 (W 0.024272) | 6.12695 ± 0.0307562 (W 0.00753233) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 4.59786 ± 0.00461346 (W 0.00947673) | 5.86879 ± 0.0246746 (W 0.00817672) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.052128 ± 0.000673936 (W 0.00033216) | 0.049589 ± 0.000741654 (W 0.00126392) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.052508 ± 0.000125779 (W 0.000593059) | 0.049849 ± 0.000185635 (W 0.00114192) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 1.97104 ± 0.0106608 (W 0.0131172) | 1.92271 ± 0.00530882 (W 0.0059484) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 1.9679 ± 0.0107397 (W 0.00706498) | 1.92695 ± 0.00167184 (W 0.00712997) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.084847 ± 0.000561278 (W 0.000932909) | 0.154255 ± 0.0010685 (W 0.00142486) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.094098 ± 0.0009644 (W 0.000668142) | 0.164115 ± 0.000165025 (W 0.00187775) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.37677 ± 0.00759551 (W 0.00305055) | 2.49558 ± 0.00481309 (W 0.0127564) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 1.51862 ± 0.00673578 (W 0.00331101) | 2.63858 ± 0.0103391 (W 0.00839023) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.293481 ± 0.000582323 (W 0.00152232) | 0.535935 ± 0.00126517 (W 0.00250808) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.384998 ± 0.0014066 (W 0.00172012) | 0.575203 ± 0.00263988 (W 0.0115884) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 5.81534 ± 0.0175389 (W 0.00744639) | 8.54065 ± 0.096954 (W 0.0228591) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 7.34754 ± 0.081286 (W 0.0167929) | 9.0428 ± 0.111985 (W 0.0153809) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.242243 ± 0.0019337 (W 0.00159513) | 0.484746 ± 0.00425475 (W 0.00224308) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 3.77236 ± 0.0095369 (W 0.00770001) | 6.53227 ± 0.0251898 (W 0.0145331) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.598432 ± 0.00163068 (W 0.00164413) | 0.977882 ± 0.00190878 (W 0.00220941) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.603543 ± 0.00239312 (W 0.00171609) | 0.984032 ± 0.00141301 (W 0.00268592) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 9.64621 ± 0.0205859 (W 0.02574) | 14.5208 ± 0.0621025 (W 0.0194572) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 9.67446 ± 0.0435185 (W 0.0185034) | 14.597 ± 0.0634711 (W 0.0339566) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.405676 ± 0.0012553 (W 0.00148041) | 0.785627 ± 0.00121854 (W 0.00309762) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.405868 ± 0.000698485 (W 0.00181725) | 0.790987 ± 0.000859186 (W 0.00300778) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 6.54249 ± 0.023269 (W 0.00640483) | 11.1576 ± 0.0356333 (W 0.0181808) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 6.57301 ± 0.0443965 (W 0.00696307) | 11.2145 ± 0.0248364 (W 0.0203868) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.197164 ± 0.00214019 (W 0.00181508) | 0.31298 ± 0.000353291 (W 0.00159815) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.209203 ± 0.00711091 (W 0.00770431) | 0.29934 ± 0.00112021 (W 0.00185039) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1` | 3.29392 ± 0.00939103 (W 0.00686283) | 4.65401 ± 0.010195 (W 0.00689534) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 3.24465 ± 0.00442862 (W 0.0116018) | 4.59044 ± 0.0216887 (W 0.0165067) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.039399 ± 7.59627e-05 (W 0.000459837) | 0.060378 ± 0.00109119 (W 0.00110933) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.038979 ± 0.000438209 (W 0.000373869) | 0.060308 ± 0.0005866 (W 0.00114696) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 1.41122 ± 0.00950439 (W 0.0308485) | 1.57749 ± 0.128124 (W 0.111724) | - | 🔵 colmat 1.12× faster than `faer` (marginal: spread 0.83 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 1.38532 ± 0.0338473 (W 0.0204092) | 1.58764 ± 0.0046639 (W 0.118485) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.086237 ± 0.000376816 (W 0.000384812) | 0.153926 ± 0.00166904 (W 0.00252793) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.094987 ± 0.000112694 (W 0.00100634) | 0.166266 ± 0.0014921 (W 0.0014015) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 1.29353 ± 0.00191464 (W 0.0031757) | 2.36392 ± 0.00997783 (W 0.00856025) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 1.4561 ± 0.000679826 (W 0.00315635) | 2.50791 ± 0.00515194 (W 0.0085587) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.270322 ± 0.000201636 (W 0.00155747) | 0.616482 ± 0.003518 (W 0.00371869) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.363248 ± 0.000803396 (W 0.00372451) | 0.627809 ± 0.00232767 (W 0.0137648) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 4.21723 ± 0.0199219 (W 0.0138648) | 9.66785 ± 0.0710274 (W 0.0415053) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 5.71571 ± 0.00388602 (W 0.0318232) | 10.1312 ± 0.374724 (W 0.307253) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.232964 ± 0.000456677 (W 0.0017206) | 0.546874 ± 0.0020512 (W 0.0030004) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 2.90688 ± 0.0117534 (W 0.00341416) | 7.47801 ± 0.0277641 (W 0.023955) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.578913 ± 0.00460555 (W 0.00211525) | 1.05635 ± 0.0010056 (W 0.00619818) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.58755 ± 0.00644922 (W 0.00245073) | 1.0627 ± 0.00093921 (W 0.0122501) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 7.79934 ± 0.00942556 (W 0.0180172) | 15.8927 ± 0.0373898 (W 0.0376837) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 7.856 ± 0.0172988 (W 0.017666) | 15.9342 ± 0.0751557 (W 0.264605) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.398288 ± 0.0014055 (W 0.00178728) | 0.859211 ± 0.000442341 (W 0.00426569) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.396898 ± 0.0030044 (W 0.00210564) | 0.867271 ± 0.00307959 (W 0.00267843) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 5.66798 ± 0.018952 (W 0.00404076) | 12.0793 ± 0.00911309 (W 0.0478473) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 5.6451 ± 0.00715082 (W 0.00788058) | 12.1348 ± 0.014742 (W 0.0426325) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.737838 ± 0.0024599 (W 0.00281431) | 0.868671 ± 0.00255487 (W 0.00143994) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.65605 ± 0.00228683 (W 0.00340098) | 0.663511 ± 0.00239612 (W 0.00293977) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 35.4879 ± 0.025884 (W 0.0956188) | 60.8491 ± 3.97798 (W 0.0982823) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 33.8071 ± 0.0324117 (W 0.00822535) | 39.4179 ± 0.106622 (W 0.0176333) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.405918 ± 0.0015612 (W 0.0013396) | 0.404648 ± 0.00214609 (W 0.00245458) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.405046 ± 0.00149496 (W 0.00173149) | 0.403817 ± 0.000944923 (W 0.00158557) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 23.6908 ± 0.0300018 (W 0.0756523) | 23.4458 ± 0.0708044 (W 0.0546232) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 23.696 ± 0.0524198 (W 0.0385003) | 23.4631 ± 0.075009 (W 0.0427308) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.388248 ± 0.00100822 (W 0.00157365) | 0.604941 ± 0.00199749 (W 0.00372496) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.367839 ± 0.000755739 (W 0.00195405) | 0.33685 ± 0.00364632 (W 0.0024176) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.2799 ± 0.00436789 (W 0.00575693) | 26.22 ± 0.0424385 (W 0.0600593) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 10.3008 ± 0.0321625 (W 0.0122938) | 12.8507 ± 0.0330864 (W 0.0331325) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.20217 ± 0.00556546 (W 0.00364351) | 2.36711 ± 0.00381802 (W 0.00275545) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.36496 ± 0.00753619 (W 0.00338875) | 1.46787 ± 0.0102143 (W 0.00264062) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 47.207 ± 0.122312 (W 0.0425283) | 75.8715 ± 4.02715 (W 0.201601) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 49.3635 ± 0.173844 (W 0.0704174) | 53.1651 ± 0.0671004 (W 0.0559781) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.789966 ± 0.00125936 (W 0.00512096) | 1.92832 ± 0.00315553 (W 0.00141173) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 22.5501 ± 0.170439 (W 0.0428306) | 51.1801 ± 0.0604112 (W 0.0428085) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.52559 ± 0.00262773 (W 0.00666386) | 1.71595 ± 0.00835298 (W 0.00298591) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.53502 ± 0.00219079 (W 0.00640746) | 1.71649 ± 0.00373861 (W 0.00243329) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 56.5008 ± 0.0373336 (W 0.0466927) | 60.1128 ± 0.047117 (W 0.0709354) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 56.6359 ± 0.0705844 (W 0.069251) | 60.2796 ± 0.0347982 (W 0.066497) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.85288 ± 0.00207458 (W 0.00362122) | 1.15278 ± 0.00319823 (W 0.00427503) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.855081 ± 0.00151478 (W 0.00511881) | 1.15405 ± 0.00159133 (W 0.00342144) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 27.547 ± 0.022528 (W 0.0388218) | 32.146 ± 0.0331491 (W 0.0320815) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 27.5942 ± 0.0268341 (W 0.0473153) | 32.1642 ± 0.0270635 (W 0.041058) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.384837 ± 0.000730535 (W 0.00198734) | 0.873564 ± 0.00243789 (W 0.00229953) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.420356 ± 0.00177961 (W 0.00480052) | 0.66049 ± 0.00267264 (W 0.00261513) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 10.6601 ± 0.00281781 (W 0.015686) | 32.8184 ± 0.0668489 (W 0.0400114) | - | 🔵 colmat 3.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.383 ± 0.0368934 (W 0.0168124) | 17.6651 ± 0.012528 (W 0.0215433) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.134335 ± 0.00207298 (W 0.000866129) | 0.125646 ± 0.000508167 (W 0.000412489) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.135626 ± 0.00241943 (W 0.00146601) | 0.125756 ± 0.000694395 (W 0.00126083) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 6.28562 ± 0.00412241 (W 0.0149262) | 6.14389 ± 0.00934785 (W 0.020032) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 6.29742 ± 0.010941 (W 0.0150214) | 6.1494 ± 0.0171442 (W 0.00706539) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.298221 ± 0.0013349 (W 0.000958155) | 0.596062 ± 0.00151553 (W 0.00335504) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.330361 ± 0.000544593 (W 0.00101489) | 0.31038 ± 0.00230001 (W 0.00160418) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.15206 ± 0.00818762 (W 0.0210651) | 21.7995 ± 0.109652 (W 0.0649601) | - | 🔵 colmat 4.23× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.92386 ± 0.0134551 (W 0.00948534) | 8.89061 ± 0.0124192 (W 0.0188998) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.611562 ± 0.00252949 (W 0.00270866) | 2.20788 ± 0.0688005 (W 0.0583287) | - | 🔵 colmat 3.61× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 0.834245 ± 0.00444618 (W 0.00411557) | 0.989751 ± 0.0296888 (W 0.0241094) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 17.6195 ± 0.329881 (W 0.0252717) | 53.4184 ± 0.0585275 (W 0.18952) | - | 🔵 colmat 3.03× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 21.7328 ± 0.0205784 (W 0.0370913) | 28.4959 ± 4.30371 (W 0.370534) | - | 🔵 colmat 1.31× faster than `faer` (marginal: spread 0.64 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.463296 ± 0.00246985 (W 0.00145796) | 2.06394 ± 0.0578318 (W 0.0522225) | - | 🔵 colmat 4.45× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 10.458 ± 0.002339 (W 0.0121455) | 46.5676 ± 0.205076 (W 0.139463) | - | 🔵 colmat 4.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.971817 ± 0.00419021 (W 0.00405682) | 1.40438 ± 0.0244775 (W 0.00530066) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.976851 ± 0.0126669 (W 0.00566457) | 1.41567 ± 0.00641223 (W 0.00338728) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 25.3874 ± 0.0464767 (W 0.036211) | 32.3994 ± 0.0216992 (W 0.0615932) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 25.5519 ± 0.0705941 (W 0.016868) | 32.5426 ± 0.0356367 (W 0.0835019) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.657011 ± 0.00346975 (W 0.00546468) | 1.13617 ± 0.00306369 (W 0.00400236) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.657557 ± 0.00766614 (W 0.00347403) | 1.14185 ± 0.00235387 (W 0.00303775) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 16.795 ± 0.0163397 (W 0.043381) | 23.7916 ± 0.0190916 (W 0.0723963) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 16.8551 ± 0.0163732 (W 0.0910671) | 23.8257 ± 0.0145686 (W 0.0279903) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.345668 ± 0.000580776 (W 0.00198311) | 0.871265 ± 0.0107792 (W 0.00341706) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.399678 ± 0.0043129 (W 0.00300373) | 0.660338 ± 0.00236941 (W 0.00455069) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.55706 ± 0.00534109 (W 0.0224325) | 27.6211 ± 0.0380034 (W 0.0837209) | - | 🔵 colmat 4.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 7.80246 ± 0.0611912 (W 0.0489903) | 11.8596 ± 0.294555 (W 0.0662946) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.096507 ± 0.000768271 (W 0.00222335) | 0.114137 ± 0.00393177 (W 0.00555787) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.097008 ± 0.000742346 (W 0.00275414) | 0.115376 ± 0.00443339 (W 0.005614) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 3.35102 ± 0.0242624 (W 0.0240793) | 4.41085 ± 0.0130779 (W 0.58015) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 3.36029 ± 0.0171924 (W 0.0310861) | 4.37596 ± 0.547773 (W 0.54618) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.274232 ± 0.000670381 (W 0.0018215) | 0.619779 ± 0.00303508 (W 0.00315184) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.324011 ± 0.000210605 (W 0.00113963) | 0.304841 ± 0.00178103 (W 0.00279265) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.29772 ± 0.013716 (W 0.0106257) | 21.1751 ± 0.0526962 (W 0.0735352) | - | 🔵 colmat 4.93× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.4569 ± 0.0388186 (W 0.0261381) | 8.32929 ± 0.0163659 (W 0.0497789) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.519713 ± 0.00102546 (W 0.00483286) | 2.07957 ± 0.00722682 (W 0.0509501) | - | 🔵 colmat 4.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.764645 ± 0.00303531 (W 0.00279315) | 1.03073 ± 0.00621785 (W 0.0238353) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 11.3337 ± 0.243793 (W 0.0396348) | 50.936 ± 0.145723 (W 0.365259) | - | 🔵 colmat 4.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 17.2376 ± 0.0410276 (W 0.077467) | 25.8905 ± 0.750756 (W 1.43446) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.431425 ± 0.00330881 (W 0.00244663) | 2.03304 ± 0.0183666 (W 0.0667715) | - | 🔵 colmat 4.71× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 7.83584 ± 0.0169186 (W 0.0133849) | 45.5252 ± 0.261182 (W 0.210943) | - | 🔵 colmat 5.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.88423 ± 0.0046004 (W 0.00332836) | 1.46731 ± 0.00320384 (W 0.00236447) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.891533 ± 0.00535828 (W 0.00500383) | 1.47873 ± 0.0214076 (W 0.00768816) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 20.1094 ± 0.0489311 (W 0.0243862) | 32.8313 ± 1.32849 (W 0.185344) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 20.3436 ± 0.409928 (W 0.0280148) | 32.7786 ± 4.88457 (W 0.19215) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.632611 ± 0.00216103 (W 0.00322531) | 1.19653 ± 0.0094781 (W 0.00415948) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.630389 ± 0.000992427 (W 0.00400255) | 1.19754 ± 0.0869684 (W 0.00427203) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1` | 14.6308 ± 0.0198634 (W 0.0167509) | 24.7567 ± 23.8668 (W 7.64555) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 14.6865 ± 0.0689557 (W 0.0329978) | 24.8898 ± 30.1585 (W 0.123512) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.57027 ± 0.00133773 (W 0.00137872) | 0.600762 ± 0.000696767 (W 0.00159524) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.574432 ± 0.00253772 (W 0.00172102) | 0.604049 ± 0.000559485 (W 0.00168962) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 33.769 ± 0.0295405 (W 0.024011) | 34.7042 ± 0.0473918 (W 0.0241257) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 32.3774 ± 0.102821 (W 0.0268818) | 33.2438 ± 0.0401494 (W 0.0159253) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.427897 ± 0.00290375 (W 0.00201139) | 0.427317 ± 0.00145726 (W 0.0023218) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.428577 ± 0.000984902 (W 0.00217205) | 0.427627 ± 0.00165628 (W 0.00160438) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 23.8729 ± 0.0202795 (W 0.0268729) | 23.9144 ± 0.0226339 (W 0.0369745) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 23.8881 ± 0.016993 (W 0.0339638) | 23.9325 ± 0.075488 (W 0.0278024) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.277461 ± 0.00188896 (W 0.00111273) | 0.265661 ± 0.000471711 (W 0.00216315) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.287872 ± 0.0010101 (W 0.00153076) | 0.279782 ± 0.000517147 (W 0.00180494) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.71522 ± 0.0249367 (W 0.0104695) | 8.62913 ± 0.00528059 (W 0.012259) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 8.8176 ± 0.0316872 (W 0.00552907) | 8.72254 ± 0.0249574 (W 0.00911659) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.03601 ± 0.00432335 (W 0.00212471) | 1.19475 ± 0.00272944 (W 0.00188648) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.2725 ± 0.00648521 (W 0.00243425) | 1.41584 ± 0.00502924 (W 0.00346864) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 43.8288 ± 0.164044 (W 0.0312649) | 45.3273 ± 0.054015 (W 0.021733) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 47.1715 ± 0.11496 (W 0.0578837) | 48.9177 ± 0.233718 (W 0.102337) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.613159 ± 0.00169835 (W 0.00166145) | 0.771667 ± 0.000672422 (W 0.00156768) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 19.3279 ± 0.0136347 (W 0.0243489) | 20.923 ± 0.0633122 (W 0.00787716) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.48334 ± 0.0056907 (W 0.00444719) | 1.68516 ± 0.00450627 (W 0.00411974) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.48556 ± 0.00210596 (W 0.00113966) | 1.68759 ± 0.00697954 (W 0.00605933) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 52.4055 ± 0.10875 (W 0.0685675) | 56.2108 ± 0.207974 (W 0.0501443) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 52.5754 ± 0.0294269 (W 0.0512098) | 56.2882 ± 0.0652366 (W 0.0582504) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.807976 ± 0.00291041 (W 0.0019137) | 1.09668 ± 0.00296956 (W 0.00149939) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.808052 ± 0.000860771 (W 0.00181705) | 1.09726 ± 0.00195856 (W 0.00137211) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 23.5615 ± 0.0256319 (W 0.0202065) | 28.242 ± 0.0390063 (W 0.0229872) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 23.5769 ± 0.0396256 (W 0.0296898) | 28.2208 ± 0.03292 (W 0.0125869) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.32082 ± 0.0026149 (W 0.00142766) | 0.60061 ± 0.00185339 (W 0.00268943) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.33781 ± 0.00217751 (W 0.00135261) | 0.606611 ± 0.00215228 (W 0.00267517) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 10.1893 ± 0.0157406 (W 0.0190822) | 11.5958 ± 0.0324047 (W 0.0120055) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 10.0976 ± 0.00510842 (W 0.0252241) | 11.4933 ± 0.027154 (W 0.0105877) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.152976 ± 0.00163337 (W 0.00144269) | 0.142775 ± 0.00452232 (W 0.000859303) | - | 🔴 colmat 1.07× slower than `faer` (marginal: spread 0.60 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.152896 ± 0.00122239 (W 0.000910693) | 0.144115 ± 0.00309952 (W 0.000910807) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 7.15621 ± 0.125889 (W 0.145068) | 7.1613 ± 0.0240117 (W 0.022674) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 7.01464 ± 0.110455 (W 0.0806152) | 7.13835 ± 0.0307093 (W 0.0334581) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.232923 ± 0.00121854 (W 0.00132473) | 0.239562 ± 0.000706364 (W 0.00126166) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.252382 ± 0.00112635 (W 0.00140849) | 0.258562 ± 0.00230953 (W 0.00204294) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.24375 ± 0.00967848 (W 0.00924984) | 4.60025 ± 0.00326022 (W 0.00643618) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 4.46061 ± 0.00212501 (W 0.00910665) | 4.82395 ± 0.0066991 (W 0.00669363) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.552631 ± 0.00194424 (W 0.00272335) | 0.894594 ± 0.00286652 (W 0.00212466) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.745777 ± 0.00444942 (W 0.00244742) | 0.934833 ± 0.029538 (W 0.0259968) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 16.1805 ± 0.0163764 (W 0.00937362) | 20.0851 ± 0.503835 (W 0.0309434) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 19.4207 ± 0.0417339 (W 0.0231601) | 20.7786 ± 1.46644 (W 0.195257) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.410056 ± 0.000558663 (W 0.00137036) | 0.751959 ± 0.00414577 (W 0.00307647) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 9.49618 ± 0.00511534 (W 0.00471629) | 13.4361 ± 0.0122174 (W 0.0301031) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.921552 ± 0.000829913 (W 0.00227087) | 1.39073 ± 0.00374002 (W 0.00521911) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.939511 ± 0.00594738 (W 0.00308996) | 1.40512 ± 0.00454113 (W 0.00275226) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 21.4186 ± 0.0454252 (W 0.0132675) | 28.4009 ± 0.0337423 (W 0.0541225) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 21.5045 ± 0.0446293 (W 0.024042) | 28.4833 ± 0.0375138 (W 0.046172) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.620511 ± 0.00143941 (W 0.00229906) | 1.08642 ± 0.00250364 (W 0.0029772) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.619652 ± 0.000218692 (W 0.00170842) | 1.09171 ± 0.000862775 (W 0.00287615) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 12.8957 ± 0.0212048 (W 0.0143653) | 19.8259 ± 0.0224801 (W 0.0410559) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 12.9181 ± 0.01516 (W 0.0164548) | 19.8509 ± 0.875082 (W 0.0399685) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.292821 ± 0.00091864 (W 0.00139524) | 0.600793 ± 0.000555193 (W 0.00274275) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.31909 ± 0.00438659 (W 0.00151922) | 0.606492 ± 0.00133858 (W 0.00306414) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.26215 ± 0.0204441 (W 0.00498027) | 7.79553 ± 0.0123814 (W 0.00643231) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 6.38963 ± 0.00681611 (W 0.00749802) | 7.87876 ± 0.012287 (W 0.00972062) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.098277 ± 0.00341782 (W 0.002155) | 0.138226 ± 0.00154272 (W 0.00416367) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.097837 ± 0.00306473 (W 0.00108462) | 0.145196 ± 0.00635973 (W 0.00624845) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.45935 ± 0.0749786 (W 0.155429) | 4.66034 ± 0.503389 (W 0.500932) | - | 🔴 colmat 1.17× slower than `faer` (marginal: spread 0.72 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.26962 ± 0.187971 (W 0.0677074) | 4.92507 ± 0.490614 (W 0.530335) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.228133 ± 0.0016521 (W 0.00144661) | 0.237363 ± 0.000727771 (W 0.00220158) | - | · | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.247893 ± 0.000558149 (W 0.00114714) | 0.254653 ± 0.0014292 (W 0.00130431) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.80901 ± 0.0107061 (W 0.00484348) | 4.00774 ± 0.00956436 (W 0.00382461) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 4.07221 ± 0.00175828 (W 0.00569285) | 4.27219 ± 0.00282394 (W 0.00632763) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.464427 ± 0.00265118 (W 0.00211664) | 0.962661 ± 0.0052577 (W 0.00559708) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.66618 ± 0.0014523 (W 0.00300224) | 0.987221 ± 0.0269452 (W 0.0267972) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 11.1121 ± 0.0246836 (W 0.021695) | 19.7972 ± 0.0603299 (W 0.104887) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.9039 ± 0.0559505 (W 0.037465) | 20.0847 ± 0.576054 (W 0.779828) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.379828 ± 0.000240965 (W 0.00149528) | 0.812337 ± 0.0041738 (W 0.00369827) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 7.31269 ± 0.0171981 (W 0.00448565) | 14.0079 ± 0.0148857 (W 0.0460103) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.844025 ± 0.000637357 (W 0.00186535) | 1.44895 ± 0.00385475 (W 0.00511854) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.857105 ± 0.00274992 (W 0.00141209) | 1.4609 ± 0.00542942 (W 0.00395138) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 16.167 ± 0.0386812 (W 0.0352301) | 28.7418 ± 12.7667 (W 0.167303) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 16.3466 ± 0.0443079 (W 0.047475) | 28.7643 ± 0.272772 (W 2.46631) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.594602 ± 0.00104652 (W 0.00296902) | 1.14875 ± 0.00494088 (W 0.00224162) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.591312 ± 0.000167703 (W 0.00217075) | 1.14743 ± 0.0170273 (W 0.00784712) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 10.6924 ± 0.00868157 (W 0.0199576) | 20.6052 ± 87.2667 (W 0.109712) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 10.7099 ± 0.0200767 (W 0.0262373) | 20.6287 ± 10.4023 (W 0.107535) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `lu/1024x64`, `lu/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `solve/1024x64`, `solve/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 29.1702 ± 0.0578688 (W 0.017433) | 30.8237 ± 0.0998179 (W 0.042359) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 31.6604 ± 0.290125 (W 0.0481685) | 47.0783 ± 0.440256 (W 0.0867911) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.440257 ± 0.00500194 (W 0.00133348) | 0.505295 ± 0.014996 (W 0.00180594) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.773288 ± 0.00207619 (W 0.001312) | 0.813666 ± 0.00584602 (W 0.00257244) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 5.0458 ± 0.0228798 (W 0.00402067) | 67.7972 ± 3.79969 (W 0.362841) | - | 🔵 colmat 13.44× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 6.72705 ± 0.0244375 (W 0.0377464) | 12.2942 ± 0.210054 (W 0.0166951) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.158475 ± 0.000522526 (W 0.00134384) | 2.06508 ± 0.00592456 (W 0.0107866) | - | 🔵 colmat 13.03× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.250653 ± 0.00341051 (W 0.0025262) | 0.307061 ± 0.00189587 (W 0.0015384) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 12.8764 ± 0.0966686 (W 0.0175264) | 12.8839 ± 0.0537131 (W 0.0155921) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 11.6662 ± 0.0328018 (W 0.023179) | 16.0355 ± 0.272552 (W 0.0219914) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.578613 ± 0.00716546 (W 0.00410914) | 0.592792 ± 0.00609527 (W 0.00232608) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.373569 ± 0.00534647 (W 0.00189026) | 0.435157 ± 0.033444 (W 0.00220255) | - | 🔵 colmat 1.16× faster than `faer` (marginal: spread 0.63 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 12.3102 ± 0.045646 (W 0.0252984) | 12.234 ± 0.0583304 (W 0.0141978) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.555823 ± 0.00268125 (W 0.00222858) | 0.574603 ± 0.00530145 (W 0.00450062) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 31.785 ± 0.0684057 (W 0.0337485) | 35.5131 ± 0.0272842 (W 0.0562134) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 31.9068 ± 0.0438319 (W 0.0433199) | 35.5937 ± 0.0858797 (W 0.0435648) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.0324 ± 0.00353331 (W 0.00224442) | 1.36272 ± 0.00482113 (W 0.00201595) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.03192 ± 0.00229194 (W 0.0020533) | 1.36588 ± 0.00339265 (W 0.00229436) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 14.6396 ± 0.0155638 (W 0.0168311) | 19.351 ± 0.00746974 (W 0.0141079) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 14.6488 ± 0.00661551 (W 0.0134919) | 19.3233 ± 0.0162057 (W 0.0320051) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.542771 ± 0.00103075 (W 0.00268089) | 0.837776 ± 0.00194563 (W 0.00172092) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.541104 ± 0.000669209 (W 0.00181335) | 0.837252 ± 0.00195788 (W 0.00101172) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 8.21651 ± 0.0277927 (W 0.0148392) | 10.4234 ± 0.0145261 (W 0.0172729) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 9.46552 ± 0.0978277 (W 0.0347205) | 30.8351 ± 1.94899 (W 0.034122) | - | 🔵 colmat 3.26× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.226873 ± 0.00214797 (W 0.00101649) | 0.504695 ± 0.00264925 (W 0.002835) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.3537 ± 0.00185568 (W 0.00120772) | 0.798087 ± 0.00732769 (W 0.00263275) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.62115 ± 0.00643659 (W 0.0055235) | 35.4094 ± 9.19695 (W 8.06889) | - | 🔵 colmat 21.84× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 2.6995 ± 0.00576476 (W 0.0238319) | 9.16272 ± 0.559178 (W 0.544882) | - | 🔵 colmat 3.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.078706 ± 0.00171584 (W 0.00144449) | 1.09605 ± 0.0132516 (W 0.209859) | - | 🔵 colmat 13.93× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.134875 ± 0.000466684 (W 0.000992237) | 0.234263 ± 0.00202677 (W 0.0121271) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 5.5926 ± 0.111406 (W 0.0228945) | 5.76769 ± 0.0270661 (W 0.0160689) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.25577 ± 0.0827024 (W 0.0130919) | 8.84719 ± 2.49205 (W 0.0249775) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.394693 ± 0.00391784 (W 0.00200388) | 0.45162 ± 0.00164218 (W 0.0019191) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.200573 ± 0.000834592 (W 0.00139655) | 0.333938 ± 0.0236248 (W 0.00383479) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 5.33534 ± 0.202009 (W 0.0165924) | 5.8158 ± 0.406234 (W 0.0202822) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.379888 ± 0.00160087 (W 0.00400281) | 0.435996 ± 0.00258474 (W 0.00460217) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 14.9419 ± 0.0504267 (W 0.0450493) | 21.3312 ± 0.0589966 (W 0.0488257) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 15.1176 ± 0.0379464 (W 0.0277075) | 21.4137 ± 0.019465 (W 0.764967) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.771927 ± 0.00350074 (W 0.00276709) | 1.31776 ± 0.00258904 (W 0.004941) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.784377 ± 0.00128394 (W 0.00219408) | 1.32378 ± 0.00635701 (W 0.00554522) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 9.17678 ± 0.0267521 (W 0.0242142) | 15.4449 ± 1.19391 (W 0.0280386) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 9.19628 ± 0.00989134 (W 0.00980506) | 15.5034 ± 3.12297 (W 0.0476585) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.496558 ± 0.000665024 (W 0.00234859) | 0.937349 ± 0.00231168 (W 0.00224524) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.499473 ± 0.00112315 (W 0.00171609) | 0.937172 ± 0.00106094 (W 0.00233977) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 4.74053 ± 0.0158282 (W 0.00455683) | 7.08036 ± 0.0237852 (W 0.0120206) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 5.6349 ± 0.0580959 (W 0.0508988) | 26.0759 ± 2.86814 (W 0.0813673) | - | 🔵 colmat 4.63× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.212701 ± 0.000410503 (W 0.00184336) | 0.503484 ± 0.00282934 (W 0.00180069) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.306628 ± 0.00218014 (W 0.00312797) | 0.806414 ± 0.00254557 (W 0.00330966) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.0179 ± 0.00601101 (W 0.00501923) | 20.0839 ± 0.269303 (W 3.32184) | - | 🔵 colmat 19.73× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.05719 ± 0.0247153 (W 0.00660715) | 7.84691 ± 0.349947 (W 0.0644436) | - | 🔵 colmat 3.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.062047 ± 0.00357668 (W 0.00148633) | 0.631004 ± 0.0052393 (W 0.00718214) | - | 🔵 colmat 10.17× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.117485 ± 0.000543588 (W 0.00112921) | 0.225971 ± 0.00937623 (W 0.00391033) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 4.49792 ± 0.756787 (W 0.0254734) | 5.51298 ± 0.123851 (W 0.0763958) | - | 🔵 colmat 1.23× faster than `faer` (marginal: spread 0.87 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 2.97723 ± 0.0164075 (W 0.00319069) | 8.76964 ± 0.126688 (W 0.0460349) | - | 🔵 colmat 2.95× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.357198 ± 0.00152841 (W 0.00382361) | 0.49297 ± 0.00240702 (W 0.00425109) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.177083 ± 0.00104394 (W 0.00150543) | 0.371679 ± 0.00147049 (W 0.00220994) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 4.27127 ± 0.0724991 (W 0.0174075) | 5.37227 ± 0.0934396 (W 0.0795117) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.34913 ± 0.00116885 (W 0.00328792) | 0.474141 ± 0.000282047 (W 0.00272914) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 12.3516 ± 0.0182355 (W 0.0244075) | 23.0869 ± 0.146247 (W 0.0430968) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 12.4918 ± 0.0463416 (W 0.0682581) | 23.1497 ± 0.94958 (W 0.768908) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 0.730908 ± 0.00229946 (W 0.00323913) | 1.38296 ± 0.00831101 (W 0.00420277) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.746935 ± 0.00234634 (W 0.00121944) | 1.40197 ± 0.00414118 (W 0.00612871) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 8.09107 ± 0.023698 (W 0.00644663) | 16.565 ± 0.0562533 (W 0.0353745) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 8.06096 ± 0.0143585 (W 0.0160618) | 16.5939 ± 0.0152779 (W 0.0706334) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.483001 ± 0.0023317 (W 0.00192425) | 1.0056 ± 0.00448853 (W 0.00516565) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.488796 ± 0.00124301 (W 0.00330514) | 1.00332 ± 0.00241517 (W 0.00396742) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 36.5083 ± 0.217864 (W 0.0905024) | 46.9337 ± 3.32722 (W 0.0306323) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 30.4449 ± 0.124395 (W 0.0737226) | 40.0795 ± 0.304653 (W 0.0331464) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.769127 ± 0.226279 (W 0.00113629) | 0.714188 ± 0.00463007 (W 0.00375153) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.722746 ± 0.00567647 (W 0.0017583) | 0.673401 ± 0.00403733 (W 0.00186458) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 5.54021 ± 0.0197879 (W 0.00379141) | 7.16111 ± 0.032952 (W 0.00796085) | 11.8281 ± 0.0259042 (W 0.0691311) | 🔵 colmat 1.29× faster than `faer` | 🔵 colmat 2.13× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 5.66732 ± 0.0262728 (W 0.0319391) | 6.20381 ± 0.0073051 (W 0.00696355) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.176565 ± 0.000265575 (W 0.00190014) | 0.170075 ± 0.000122748 (W 0.00175252) | 0.280312 ± 0.000336631 (W 0.00166005) | · | 🔵 colmat 1.59× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.185305 ± 9.05373e-05 (W 0.00154854) | 0.175545 ± 0.00130247 (W 0.00194587) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.3888 ± 0.0135042 (W 0.0141874) | 10.3476 ± 0.0584807 (W 0.00995609) | 36.4526 ± 0.03645 (W 0.0342845) | · | 🔵 colmat 3.51× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 10.5281 ± 0.0147178 (W 0.0429009) | 10.3643 ± 0.0357982 (W 0.0177673) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.2908 ± 0.0035788 (W 0.00209664) | 0.28798 ± 0.00248807 (W 0.00174036) | 0.821022 ± 0.000983634 (W 0.000969277) | · | 🔵 colmat 2.82× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.29679 ± 0.000575512 (W 0.00193888) | 0.297561 ± 0.00176213 (W 0.00171847) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 9.81421 ± 0.0138002 (W 0.016772) | 9.7545 ± 0.00102567 (W 0.00635842) | 33.9938 ± 0.0525593 (W 0.0274379) | · | 🔵 colmat 3.46× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.273442 ± 0.000776235 (W 0.00188612) | 0.271991 ± 0.00127579 (W 0.0017691) | 0.773874 ± 0.0012189 (W 0.0024726) | · | 🔵 colmat 2.83× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 35.8121 ± 0.0867063 (W 0.0422946) | 39.4008 ± 0.0218891 (W 0.0910716) | 140.485 ± 0.0574688 (W 0.202911) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 3.92× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.0013 ± 0.143932 (W 0.0790867) | 39.5646 ± 0.0434606 (W 0.0444354) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.08358 ± 0.00281666 (W 0.00318909) | 1.40213 ± 0.00435612 (W 0.0022835) | 2.56839 ± 0.00766986 (W 0.0025951) | 🔵 colmat 1.29× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.08544 ± 0.00180876 (W 0.00419011) | 1.40644 ± 0.00226721 (W 0.0028334) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 18.7128 ± 0.0559119 (W 0.0265272) | 23.4239 ± 0.0226605 (W 0.0191784) | 101.955 ± 0.211758 (W 0.119031) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 5.45× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 18.6813 ± 0.0583456 (W 0.0353646) | 23.4414 ± 0.00603769 (W 0.0357995) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.595233 ± 0.0035371 (W 0.00378715) | 0.886554 ± 0.00728213 (W 0.00559858) | 1.63143 ± 0.00445516 (W 0.00342805) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 2.74× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.591471 ± 0.00537101 (W 0.00616855) | 0.875635 ± 0.0118789 (W 0.00366612) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 9.68252 ± 0.0497989 (W 0.0233059) | 22.2749 ± 0.0271329 (W 0.12469) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 8.45424 ± 0.0385777 (W 0.0277478) | 20.5429 ± 0.0567458 (W 0.0901039) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.292692 ± 0.00051806 (W 0.00200083) | 0.69983 ± 0.00333331 (W 0.00196616) | - | 🔵 colmat 2.39× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.296397 ± 0.00168282 (W 0.00193139) | 0.675658 ± 0.00478209 (W 0.00343632) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.52431 ± 0.000985211 (W 0.00366835) | 2.83336 ± 0.152639 (W 0.300945) | 2.94142 ± 0.00458955 (W 0.00476292) | 🔵 colmat 1.86× faster than `faer` | 🔵 colmat 1.93× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 1.73813 ± 0.00436474 (W 0.00503099) | 2.57593 ± 0.0757844 (W 0.202324) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.057588 ± 0.000370145 (W 0.000212438) | 0.089367 ± 0.00675744 (W 0.01137) | 0.130397 ± 0.000653138 (W 0.00178785) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 2.26× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.072287 ± 0.000361649 (W 0.000181335) | 0.100376 ± 0.0075591 (W 0.00989955) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 2.94853 ± 0.00723757 (W 0.00543725) | 3.21621 ± 0.00701888 (W 0.00611239) | 7.3505 ± 0.0122248 (W 0.00758899) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 2.49× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 3.28294 ± 0.0315937 (W 0.00581633) | 3.62523 ± 0.100088 (W 0.0057718) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.124876 ± 0.000405378 (W 0.00127225) | 0.169574 ± 0.000959961 (W 0.00128596) | 0.253892 ± 0.000839027 (W 0.00194475) | 🔵 colmat 1.36× faster than `faer` | 🔵 colmat 2.03× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.139523 ± 0.00107672 (W 0.000357729) | 0.188314 ± 0.00173808 (W 0.0011354) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 2.76749 ± 0.0177924 (W 0.00498888) | 3.01861 ± 0.0029839 (W 0.00530157) | 6.78506 ± 0.0194181 (W 0.00801816) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 2.45× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.112457 ± 0.000400954 (W 0.000345442) | 0.155406 ± 0.00138398 (W 0.00109988) | 0.23326 ± 0.000884094 (W 0.00250035) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 2.07× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 18.9098 ± 0.0303288 (W 0.074658) | 25.3023 ± 3.53023 (W 0.0713336) | 78.5944 ± 0.875216 (W 0.986123) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 4.16× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 19.0871 ± 0.0222581 (W 0.0269782) | 25.3702 ± 0.617541 (W 0.739602) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.814846 ± 0.00367795 (W 0.00219776) | 1.36227 ± 0.00397241 (W 0.00467268) | 2.01634 ± 0.00326288 (W 0.0115973) | 🔵 colmat 1.67× faster than `faer` | 🔵 colmat 2.47× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.824753 ± 0.000892513 (W 0.00499351) | 1.37179 ± 0.00380502 (W 0.00742301) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 13.0953 ± 0.0117002 (W 0.0163946) | 19.5169 ± 0.599599 (W 0.0324318) | 61.7237 ± 0.113951 (W 0.0821392) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 4.71× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 13.1204 ± 0.00543078 (W 0.0389311) | 19.571 ± 2.94067 (W 0.0326874) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.543982 ± 0.00344134 (W 0.00338617) | 0.981228 ± 0.00740295 (W 0.00198443) | 1.37121 ± 0.025677 (W 0.00477171) | 🔵 colmat 1.80× faster than `faer` | 🔵 colmat 2.52× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.546157 ± 0.00292447 (W 0.00304444) | 0.976541 ± 0.00729073 (W 0.0027473) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 4.95278 ± 0.0470252 (W 0.0298569) | 17.9051 ± 0.084985 (W 0.0138989) | - | 🔵 colmat 3.62× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 4.6034 ± 0.0549039 (W 0.0305793) | 17.2669 ± 0.0161243 (W 0.0406255) | - | 🔵 colmat 3.75× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.236851 ± 0.00230801 (W 0.00354325) | 0.700386 ± 0.00381583 (W 0.00176323) | - | 🔵 colmat 2.96× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.242002 ± 0.00236629 (W 0.00227506) | 0.676838 ± 0.00776326 (W 0.00416247) | - | 🔵 colmat 2.80× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 0.846495 ± 0.00314495 (W 0.00706875) | 1.74027 ± 0.0441669 (W 0.0638837) | 1.56326 ± 0.00214772 (W 0.00406962) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.85× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 1.11211 ± 0.00186199 (W 0.00701341) | 1.86294 ± 0.0372355 (W 0.0625013) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.038298 ± 6.26126e-05 (W 0.00017544) | 0.070198 ± 0.000483794 (W 0.00478203) | 0.066908 ± 0.000687917 (W 0.00121075) | 🔵 colmat 1.83× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.054668 ± 0.000342977 (W 0.000312422) | 0.087497 ± 0.000202978 (W 0.00486148) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 1.76561 ± 0.00618013 (W 0.00508517) | 2.95012 ± 0.0465495 (W 0.0443555) | 3.75093 ± 0.0019155 (W 0.0114797) | 🔵 colmat 1.67× faster than `faer` | 🔵 colmat 2.12× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 2.0321 ± 0.00828431 (W 0.00528831) | 3.22079 ± 0.0394996 (W 0.0594594) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.101316 ± 0.000404339 (W 0.00133363) | 0.218262 ± 0.00108142 (W 0.00201876) | 0.150146 ± 0.000185079 (W 0.00206755) | 🔵 colmat 2.15× faster than `faer` | 🔵 colmat 1.48× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.118825 ± 0.00158351 (W 0.000666001) | 0.24129 ± 0.00122031 (W 0.00303311) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 1.64229 ± 0.00374593 (W 0.00371729) | 2.67968 ± 0.0393743 (W 0.0343874) | 3.46952 ± 0.00891654 (W 0.00925435) | 🔵 colmat 1.63× faster than `faer` | 🔵 colmat 2.11× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.090457 ± 0.000538176 (W 0.00121369) | 0.204912 ± 0.00155569 (W 0.00468525) | 0.133406 ± 0.000682781 (W 0.00166778) | 🔵 colmat 2.27× faster than `faer` | 🔵 colmat 1.47× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 16.3988 ± 0.0244405 (W 0.0528104) | 27.1077 ± 0.0129739 (W 0.0634609) | 67.3192 ± 0.100538 (W 0.0903085) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 4.11× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 16.4988 ± 0.0364985 (W 0.0502983) | 27.2387 ± 0.996156 (W 0.0671054) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 0.779777 ± 0.00391297 (W 0.00546756) | 1.4411 ± 0.0114628 (W 0.00542881) | 1.95869 ± 0.0070628 (W 0.00828659) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 2.51× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.794018 ± 0.00531701 (W 0.00503741) | 1.44143 ± 0.00854089 (W 0.00475265) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 12.028 ± 1.33978 (W 0.0518503) | 20.7712 ± 0.0249294 (W 0.0460482) | 54.345 ± 0.175564 (W 0.0663802) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 4.52× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 12.08 ± 0.0876478 (W 0.0313562) | 20.782 ± 0.0558367 (W 0.0555132) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.529643 ± 0.00128478 (W 0.00410662) | 1.03542 ± 0.00504381 (W 0.00224389) | 1.36644 ± 0.00227347 (W 0.00323427) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 2.58× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.533398 ± 0.00160082 (W 0.00585415) | 1.04109 ± 0.0029819 (W 0.00536193) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 66.0506 ± 0.236273 (W 0.102951) | 67.7266 ± 0.173723 (W 0.181982) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 76.3929 ± 1.51816 (W 0.301183) | 101.219 ± 0.604923 (W 0.546039) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.996592 ± 0.00395563 (W 0.00104006) | 1.05796 ± 0.00435675 (W 0.00203149) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.38403 ± 0.0101796 (W 0.00223665) | 1.6074 ± 0.0489756 (W 0.00650907) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 16.2424 ± 0.0847656 (W 0.0202525) | 80.5128 ± 10.9105 (W 0.0677055) | - | 🔵 colmat 4.96× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 19.9295 ± 0.0633379 (W 0.148806) | 22.5278 ± 0.110939 (W 0.038075) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.443457 ± 0.00180432 (W 0.00135268) | 2.38372 ± 0.0172612 (W 0.00339108) | - | 🔵 colmat 5.38× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.524834 ± 0.00228604 (W 0.00269843) | 0.561193 ± 0.00117613 (W 0.00184319) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 39.765 ± 0.157154 (W 0.0401313) | 40.0653 ± 0.282086 (W 0.0852761) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 40.2141 ± 0.158206 (W 0.220786) | 43.2088 ± 0.434266 (W 0.168695) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.30115 ± 0.0210551 (W 0.00245075) | 1.3517 ± 0.0209503 (W 0.00502976) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 1.18059 ± 0.00257059 (W 0.00272477) | 1.19975 ± 0.00377044 (W 0.00357272) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 37.6096 ± 0.243693 (W 0.0343176) | 37.8112 ± 0.182929 (W 0.0650236) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 1.23862 ± 0.0154702 (W 0.00385861) | 1.25296 ± 0.0250473 (W 0.00267237) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 98.5808 ± 0.267667 (W 0.124869) | 103.57 ± 0.117674 (W 0.081032) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 98.5434 ± 0.0697304 (W 0.099762) | 103.763 ± 0.158882 (W 0.224517) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.44044 ± 0.00947083 (W 0.00340286) | 2.78754 ± 0.000272859 (W 0.00194748) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.45718 ± 0.00728869 (W 0.00157833) | 2.82415 ± 0.0046892 (W 0.00221471) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 41.2451 ± 0.0209401 (W 0.0441194) | 47.9538 ± 0.197369 (W 0.0422866) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 41.4638 ± 0.0912346 (W 0.0258038) | 47.9595 ± 0.0899667 (W 0.0633503) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.21173 ± 0.0101963 (W 0.0018425) | 1.55349 ± 0.00571047 (W 0.00135407) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.22968 ± 0.0113153 (W 0.00125324) | 1.57172 ± 0.00500632 (W 0.00412404) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 18.0317 ± 0.0450171 (W 0.0226958) | 20.5569 ± 0.0489639 (W 0.0576669) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 22.417 ± 0.183432 (W 0.215593) | 44.0651 ± 0.632391 (W 0.232958) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.403086 ± 0.00487275 (W 0.00308022) | 1.05712 ± 0.00209769 (W 0.00215825) | - | 🔵 colmat 2.62× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.524925 ± 0.00735312 (W 0.00182628) | 1.57178 ± 0.0538804 (W 0.0106211) | - | 🔵 colmat 2.99× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 4.60217 ± 0.0145833 (W 0.0129944) | 39.7486 ± 0.797126 (W 7.29815) | - | 🔵 colmat 8.64× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 7.38035 ± 0.0835165 (W 0.0953467) | 12.725 ± 0.440588 (W 1.25388) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.159963 ± 0.000956034 (W 0.00105969) | 1.17166 ± 0.0274665 (W 0.261738) | - | 🔵 colmat 7.32× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.236053 ± 0.00446878 (W 0.0014496) | 0.342599 ± 0.00374791 (W 0.0262847) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 12.5222 ± 0.36875 (W 0.0204865) | 12.6169 ± 0.453424 (W 0.0232278) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 13.1033 ± 0.0181749 (W 0.0834811) | 15.6775 ± 0.269728 (W 0.0305867) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.570483 ± 0.00125219 (W 0.00440581) | 0.635671 ± 0.00192767 (W 0.00360819) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.458426 ± 0.00272291 (W 0.00195124) | 0.555936 ± 0.000503366 (W 0.00216318) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 12.0088 ± 0.04717 (W 0.0325771) | 12.1784 ± 0.228203 (W 0.0144362) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.532634 ± 0.000186743 (W 0.00412461) | 0.588923 ± 0.00138043 (W 0.00434978) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 37.1745 ± 0.0634902 (W 0.0303961) | 47.9837 ± 0.00417832 (W 0.0458459) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 37.3939 ± 0.0483584 (W 0.0217505) | 48.0556 ± 0.0865946 (W 0.0739432) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.40073 ± 0.00258687 (W 0.00227113) | 2.03998 ± 0.00500843 (W 0.00564043) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.42615 ± 0.0148041 (W 0.0039717) | 2.05583 ± 0.00820286 (W 0.00536663) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 19.6085 ± 0.0301476 (W 0.0274469) | 30.5594 ± 0.0346379 (W 0.0455797) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.5849 ± 0.025845 (W 0.0261824) | 30.586 ± 0.365554 (W 0.0493918) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.883471 ± 0.00588904 (W 0.0013797) | 1.40013 ± 0.00608107 (W 0.00338603) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.88464 ± 0.00366456 (W 0.00273719) | 1.39874 ± 0.0087664 (W 0.00489212) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 9.97739 ± 0.00726267 (W 0.0177899) | 12.8593 ± 0.0443482 (W 0.0310996) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 12.7366 ± 0.0791757 (W 0.0717637) | 36.1204 ± 18.8911 (W 0.113184) | - | 🔵 colmat 2.84× faster than `faer` (marginal: spread 0.81 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.318258 ± 0.000332632 (W 0.00233291) | 1.056 ± 0.00189574 (W 0.00204066) | - | 🔵 colmat 3.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.407297 ± 0.0067319 (W 0.00211195) | 1.61446 ± 0.0137692 (W 0.00988939) | - | 🔵 colmat 3.96× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.58495 ± 0.0319572 (W 0.00914804) | 22.1022 ± 0.104522 (W 0.101173) | - | 🔵 colmat 8.55× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 5.20494 ± 0.0599521 (W 0.0763246) | 9.41735 ± 0.430319 (W 0.108291) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.113347 ± 0.000242297 (W 0.000977372) | 0.684077 ± 0.00864518 (W 0.0146435) | - | 🔵 colmat 6.04× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.185244 ± 0.00123061 (W 0.00153834) | 0.269122 ± 0.00761974 (W 0.00346821) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 7.55054 ± 0.288434 (W 0.016402) | 9.76544 ± 0.088634 (W 0.335631) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 8.38617 ± 0.0525543 (W 0.0475177) | 14.5567 ± 0.79072 (W 1.21046) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.439573 ± 0.000755574 (W 0.00474919) | 0.646131 ± 0.0019131 (W 0.00974352) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.340257 ± 0.00452278 (W 0.00197937) | 0.589562 ± 0.020076 (W 0.00991363) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 7.1964 ± 0.291817 (W 0.027325) | 9.44825 ± 0.282907 (W 0.213474) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.416658 ± 0.000504322 (W 0.00284361) | 0.599362 ± 0.00200117 (W 0.00998444) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 27.9143 ± 0.327516 (W 0.0489815) | 46.671 ± 0.172601 (W 0.319896) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 28.2031 ± 0.0780745 (W 0.0258547) | 46.8282 ± 0.227245 (W 1.82286) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.25777 ± 0.0108505 (W 0.00332899) | 2.08035 ± 0.00474654 (W 0.00717915) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.24999 ± 0.00332913 (W 0.00424995) | 2.09022 ± 0.141037 (W 0.0089967) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 15.4963 ± 0.0101345 (W 0.0398473) | 31.1068 ± 0.0722356 (W 0.140345) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 15.4726 ± 0.0239812 (W 0.0224416) | 31.0874 ± 0.0143531 (W 0.173756) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.834745 ± 0.00388804 (W 0.00311707) | 1.43947 ± 0.00443063 (W 0.00328686) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.835034 ± 0.00233662 (W 0.00340583) | 1.44765 ± 0.0130623 (W 0.00741708) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 80.5386 ± 0.924577 (W 0.201563) | 95.998 ± 1.70105 (W 0.162) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 72.9574 ± 3.69442 (W 0.583243) | 89.3148 ± 0.114125 (W 0.646643) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 1.35799 ± 0.0101008 (W 0.00165674) | 1.54489 ± 0.124624 (W 0.00425007) | - | 🔵 colmat 1.14× faster than `faer` (marginal: spread 0.72 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.31875 ± 0.00829466 (W 0.00335713) | 1.48705 ± 0.0142616 (W 0.0039614) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 16.7162 ± 0.186694 (W 0.0889582) | 17.7985 ± 0.0591205 (W 0.016252) | 46.2363 ± 0.0197589 (W 0.0707307) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 2.77× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 17.2511 ± 0.0360037 (W 0.0265208) | 16.8551 ± 0.0520257 (W 0.0316466) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.441486 ± 0.00342823 (W 0.00252196) | 0.438536 ± 0.00298563 (W 0.0018225) | 1.32712 ± 0.00116324 (W 0.00180518) | · | 🔵 colmat 3.01× faster than `lapack-mkl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.462757 ± 0.00181299 (W 0.0019196) | 0.458115 ± 0.000555644 (W 0.00147586) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 37.4009 ± 0.0925192 (W 0.140368) | 37.3165 ± 0.251633 (W 0.0989057) | 103.573 ± 0.480928 (W 0.13071) | · | 🔵 colmat 2.77× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 37.5367 ± 0.24344 (W 0.0692489) | 37.5125 ± 0.112768 (W 0.0516308) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.07827 ± 0.00344621 (W 0.00147272) | 1.07512 ± 0.00357005 (W 0.00154803) | 2.38317 ± 0.0105968 (W 0.00259022) | · | 🔵 colmat 2.21× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 1.10246 ± 0.000591531 (W 0.0011353) | 1.08186 ± 0.000910839 (W 0.00242071) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 35.0595 ± 0.208066 (W 0.070489) | 35.1163 ± 0.173851 (W 0.0472425) | 94.415 ± 0.157014 (W 0.1271) | · | 🔵 colmat 2.69× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 1.01193 ± 0.00258403 (W 0.000910083) | 1.00839 ± 0.0113557 (W 0.00127128) | 2.22767 ± 0.0276008 (W 0.00254237) | · | 🔵 colmat 2.20× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 107.174 ± 0.242057 (W 0.107778) | 112.4 ± 0.241367 (W 0.221409) | 352.845 ± 1.30468 (W 1.09871) | · | 🔵 colmat 3.29× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 107.337 ± 0.295976 (W 0.108889) | 112.863 ± 0.0586883 (W 0.217576) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.52315 ± 0.0106368 (W 0.0022683) | 2.86684 ± 0.00659823 (W 0.00134481) | 7.71117 ± 0.0890155 (W 0.0214895) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 3.06× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.52572 ± 0.0122341 (W 0.00144462) | 2.88603 ± 0.0135981 (W 0.00264826) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 54.4248 ± 0.0894889 (W 0.0569532) | 56.9227 ± 2.59515 (W 0.0934529) | 137.074 ± 1.74743 (W 0.362137) | · | 🔵 colmat 2.52× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 54.2328 ± 2.6991 (W 0.0587266) | 56.754 ± 0.237527 (W 0.0596944) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.29875 ± 0.00391804 (W 0.00245029) | 1.61957 ± 0.0032722 (W 0.00306539) | 3.47633 ± 0.0292308 (W 0.00541522) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 2.68× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.29342 ± 0.000887431 (W 0.002068) | 1.6456 ± 0.00196187 (W 0.00253618) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 22.1201 ± 0.851712 (W 0.0402546) | 41.4359 ± 0.0869764 (W 0.0707584) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 19.7664 ± 0.378034 (W 0.0581118) | 38.9889 ± 0.155911 (W 0.0973304) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.451366 ± 0.00127252 (W 0.0019117) | 1.49807 ± 0.0310821 (W 0.00686718) | - | 🔵 colmat 3.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.450656 ± 0.00182765 (W 0.00171435) | 1.47323 ± 0.0320648 (W 0.00816828) | - | 🔵 colmat 3.27× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 4.5516 ± 0.0597707 (W 0.00802106) | 7.72249 ± 0.252772 (W 0.330842) | 12.7705 ± 0.0330462 (W 0.0207838) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 2.81× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 5.16529 ± 0.0133181 (W 0.0201849) | 7.36212 ± 0.841075 (W 0.728185) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.134034 ± 0.00032895 (W 0.000536936) | 0.158405 ± 0.0167429 (W 0.0243176) | 0.32068 ± 0.00135224 (W 0.00141033) | 🔵 colmat 1.18× faster than `faer` (marginal: spread 1.00 of the difference) | 🔵 colmat 2.39× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.162914 ± 0.00106515 (W 0.000430707) | 0.22648 ± 0.00860216 (W 0.0268666) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 10.0044 ± 0.0283432 (W 0.0187511) | 10.2763 ± 0.0511638 (W 0.028151) | 24.3099 ± 0.017295 (W 0.0256145) | · | 🔵 colmat 2.43× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 10.419 ± 0.0269911 (W 0.0556774) | 10.8861 ± 0.0615094 (W 0.028037) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.350439 ± 0.0010417 (W 0.00176609) | 0.403357 ± 0.000416782 (W 0.00193339) | 0.859944 ± 0.000828794 (W 0.00353181) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 2.45× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.385538 ± 0.00052302 (W 0.00181582) | 0.443091 ± 0.00067703 (W 0.00250213) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 9.32518 ± 0.0166385 (W 0.0146385) | 9.59425 ± 0.0350954 (W 0.0171215) | 22.5368 ± 0.0162505 (W 0.0201548) | · | 🔵 colmat 2.42× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.317041 ± 8.20183e-05 (W 0.000623886) | 0.358449 ± 0.000986231 (W 0.00240864) | 0.785496 ± 0.000969 (W 0.00323371) | 🔵 colmat 1.13× faster than `faer` | 🔵 colmat 2.48× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 45.4588 ± 0.154545 (W 0.0775743) | 56.717 ± 0.0825315 (W 0.107347) | 121.922 ± 0.175765 (W 0.120728) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 2.68× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 45.6588 ± 0.0763822 (W 0.0569196) | 56.8895 ± 0.0993165 (W 0.0785479) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.46263 ± 0.00544451 (W 0.00475766) | 2.11143 ± 0.00102059 (W 0.00510389) | 4.73287 ± 0.0142233 (W 0.00798589) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 3.24× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.49709 ± 0.016197 (W 0.00409563) | 2.116 ± 0.00504515 (W 0.00562926) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 28.0179 ± 0.0962976 (W 0.0916803) | 43.3659 ± 2.37567 (W 0.0410599) | 58.5505 ± 1.16117 (W 0.280068) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 28.0618 ± 0.0440047 (W 0.062464) | 39.5143 ± 0.0580654 (W 0.0663851) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.944292 ± 0.00456342 (W 0.00675164) | 1.46226 ± 0.0025921 (W 0.00250202) | 2.34421 ± 0.00558333 (W 0.0101241) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 2.48× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.943278 ± 0.00244797 (W 0.00282704) | 1.45864 ± 0.00333258 (W 0.00532992) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 11.3742 ± 0.0203214 (W 0.0317837) | 31.7602 ± 0.0911084 (W 1.08815) | - | 🔵 colmat 2.79× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 10.8415 ± 0.0233297 (W 0.0506607) | 30.8042 ± 0.0654236 (W 0.0590555) | - | 🔵 colmat 2.84× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.319751 ± 0.00408336 (W 0.00597408) | 1.50712 ± 0.0293082 (W 0.00683967) | - | 🔵 colmat 4.71× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.326636 ± 0.0167809 (W 0.00315752) | 1.47976 ± 0.026574 (W 0.00507631) | - | 🔵 colmat 4.53× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 2.4445 ± 0.00302961 (W 0.00921299) | 4.31871 ± 0.0207594 (W 0.289386) | 6.51749 ± 0.00465716 (W 0.00705818) | 🔵 colmat 1.77× faster than `faer` | 🔵 colmat 2.67× faster than `lapack-mkl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 3.09661 ± 0.00610751 (W 0.0108164) | 4.67982 ± 0.0266161 (W 0.254825) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.085668 ± 0.00277382 (W 0.00046953) | 0.123676 ± 0.000689082 (W 0.0115375) | 0.209584 ± 0.000290532 (W 0.00096941) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 2.45× faster than `lapack-mkl` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.116286 ± 0.000606721 (W 0.000294167) | 0.156964 ± 0.00050256 (W 0.00369164) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 5.42833 ± 0.00449578 (W 0.0219867) | 7.73708 ± 0.148753 (W 0.279755) | 12.3011 ± 0.00521345 (W 0.0102048) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 2.27× faster than `lapack-mkl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 5.96279 ± 0.0428504 (W 0.0114465) | 8.62377 ± 0.14914 (W 0.39064) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.227883 ± 0.000621727 (W 0.00140857) | 0.414327 ± 0.0050388 (W 0.0116862) | 0.480226 ± 0.000643746 (W 0.00165678) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 2.11× faster than `lapack-mkl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.269788 ± 0.000815116 (W 0.0017709) | 0.466326 ± 0.00603821 (W 0.0105293) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 5.15768 ± 0.0412792 (W 0.00983472) | 7.22433 ± 0.0925747 (W 0.315516) | 11.3889 ± 0.0122494 (W 0.0148317) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 2.21× faster than `lapack-mkl` |  |
| `qr_factor/64x1024/b1` | 0.203554 ± 0.000753284 (W 0.00186174) | 0.370239 ± 0.00702621 (W 0.00966511) | 0.426297 ± 0.000949942 (W 0.00189296) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-mkl` |  |
| `svd_thin/256x4096/b1` | 36.5142 ± 0.185054 (W 0.0981931) | 55.5308 ± 0.288443 (W 0.25992) | 79.744 ± 2.40109 (W 0.993123) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 2.18× faster than `lapack-mkl` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.7506 ± 0.187719 (W 0.127556) | 56.003 ± 1.89444 (W 1.4767) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.305 ± 0.0187002 (W 0.00197027) | 2.14755 ± 0.00401162 (W 0.00663759) | 4.06301 ± 0.0547016 (W 0.0190097) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 3.11× faster than `lapack-mkl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.31315 ± 0.00602325 (W 0.00342261) | 2.32459 ± 0.103157 (W 0.20965) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1` | 24.1871 ± 0.0716722 (W 0.124082) | 39.9349 ± 0.115921 (W 0.159753) | 46.3795 ± 0.388008 (W 0.0634246) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.92× faster than `lapack-mkl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.0785 ± 0.0754829 (W 0.134607) | 40.0393 ± 0.0376447 (W 0.22442) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.897454 ± 0.0165298 (W 0.0035138) | 1.50158 ± 0.00154503 (W 0.00454452) | 2.10958 ± 0.0301295 (W 0.00725684) | 🔵 colmat 1.67× faster than `faer` | 🔵 colmat 2.35× faster than `lapack-mkl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.899173 ± 0.00267479 (W 0.00397287) | 1.5 ± 0.00328943 (W 0.00399449) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 14.4494 ± 0.0200769 (W 0.0214709) | 15.8388 ± 0.0447471 (W 0.00946646) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 19.5377 ± 0.21136 (W 0.123665) | 27.6423 ± 0.614994 (W 0.0957146) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.238592 ± 0.00242817 (W 0.00110798) | 0.299936 ± 0.000561774 (W 0.00148363) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.671409 ± 0.000929662 (W 0.00145174) | 0.696609 ± 0.015577 (W 0.0015379) | - | · | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.91013 ± 0.00973798 (W 0.00882443) | 42.7197 ± 1.43979 (W 0.0653389) | - | 🔵 colmat 14.68× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.83376 ± 0.0345812 (W 0.00613384) | 9.17492 ± 0.655467 (W 0.0185682) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.106186 ± 0.000170119 (W 0.00020321) | 2.03635 ± 0.00807506 (W 0.0122032) | - | 🔵 colmat 19.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.227953 ± 0.000961032 (W 0.00091896) | 0.287987 ± 0.00170707 (W 0.00181963) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 9.15245 ± 0.333054 (W 0.0175141) | 9.21746 ± 0.223727 (W 0.0221596) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 8.4866 ± 0.0229722 (W 0.0142871) | 16.6001 ± 2.47644 (W 0.68989) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.521657 ± 0.00358621 (W 0.00410638) | 0.523843 ± 0.00358511 (W 0.00323209) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.372309 ± 0.00131123 (W 0.00158587) | 0.393592 ± 0.0035407 (W 0.00207647) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 8.76964 ± 0.0315983 (W 0.0131621) | 9.09923 ± 0.0490948 (W 0.0242284) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.504465 ± 0.000882925 (W 0.00396637) | 0.509305 ± 0.00105367 (W 0.00258466) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 19.0202 ± 0.0825651 (W 0.0101003) | 22.1752 ± 0.156678 (W 0.0096811) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 19.0513 ± 0.110056 (W 0.00983163) | 22.1681 ± 0.135348 (W 0.021197) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.764435 ± 0.00134679 (W 0.00144468) | 0.911699 ± 0.00261241 (W 0.00165286) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.766646 ± 0.00106616 (W 0.00237493) | 1.00312 ± 0.00391034 (W 0.00227175) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 9.48549 ± 0.0248752 (W 0.0065091) | 12.9128 ± 0.0259752 (W 0.00608761) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 9.49672 ± 0.00425558 (W 0.00765861) | 12.907 ± 0.0159708 (W 0.00988752) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.427626 ± 0.000757047 (W 0.00173548) | 0.673768 ± 0.000560043 (W 0.00119932) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.427125 ± 0.00218959 (W 0.00190396) | 0.673217 ± 0.00112415 (W 0.00161198) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 4.46922 ± 0.00976752 (W 0.00534678) | 6.14098 ± 0.00406165 (W 0.00725777) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 6.0909 ± 0.0862333 (W 0.0136761) | 17.1066 ± 0.666283 (W 0.0376024) | - | 🔵 colmat 2.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.172244 ± 0.00222607 (W 0.00114696) | 0.299431 ± 0.00164596 (W 0.00242939) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.33789 ± 0.00100166 (W 0.00127095) | 0.659941 ± 0.0118742 (W 0.00295475) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 0.964761 ± 0.00234983 (W 0.00268633) | 22.3857 ± 5.26467 (W 4.21591) | - | 🔵 colmat 23.20× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 2.2572 ± 0.0127313 (W 0.0092604) | 7.318 ± 0.402991 (W 0.148109) | - | 🔵 colmat 3.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.050828 ± 0.000100201 (W 0.000447731) | 1.08391 ± 0.00145909 (W 0.246532) | - | 🔵 colmat 21.33× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.126726 ± 0.000541425 (W 0.000647381) | 0.221193 ± 0.0140824 (W 0.0105109) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 4.12717 ± 0.100316 (W 0.0163727) | 4.43537 ± 0.0995591 (W 0.0130317) | - | 🔵 colmat 1.07× faster than `faer` (marginal: spread 0.65 of the difference) | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 3.29638 ± 0.0104468 (W 0.0132086) | 8.12079 ± 0.279642 (W 0.0399179) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.382314 ± 0.00319524 (W 0.00337359) | 0.440715 ± 0.000771174 (W 0.00350067) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.222313 ± 0.00257725 (W 0.00491936) | 0.304918 ± 0.00497749 (W 0.00108237) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 4.02912 ± 0.114156 (W 0.00918948) | 4.31221 ± 0.0301327 (W 0.00879201) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.371329 ± 0.00379543 (W 0.00385277) | 0.422507 ± 0.00126475 (W 0.00377621) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 9.5851 ± 0.0869902 (W 0.0221259) | 14.4881 ± 0.0459852 (W 0.0336878) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 9.48925 ± 0.112757 (W 0.00805997) | 14.5872 ± 0.0738386 (W 0.0443474) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.606035 ± 0.0051322 (W 0.00222771) | 0.973678 ± 0.00687351 (W 0.00266184) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.611122 ± 0.00131142 (W 0.00208549) | 0.976001 ± 0.0132251 (W 0.00152311) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 6.54267 ± 0.0281405 (W 0.0172126) | 11.1061 ± 0.0145851 (W 0.0228532) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 6.54385 ± 0.0291869 (W 0.00764086) | 11.1305 ± 0.00853809 (W 0.184823) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.404193 ± 0.000159145 (W 0.00118419) | 0.792894 ± 0.000390707 (W 0.0039024) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.403838 ± 0.000801677 (W 0.00113037) | 0.793384 ± 0.00228931 (W 0.00353773) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 2.84646 ± 0.00381909 (W 0.00785735) | 4.6555 ± 0.0437317 (W 0.013289) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 3.93258 ± 0.0294392 (W 0.0103269) | 14.348 ± 0.347391 (W 0.037486) | - | 🔵 colmat 3.65× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.167795 ± 0.00101293 (W 0.00243684) | 0.299391 ± 0.00152208 (W 0.0018347) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.302162 ± 0.00221208 (W 0.00193847) | 0.657684 ± 0.0209437 (W 0.00126137) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 0.676014 ± 0.00140772 (W 0.00367805) | 12.6986 ± 0.0763229 (W 2.0298) | - | 🔵 colmat 18.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.82518 ± 0.0255649 (W 0.00579346) | 7.015 ± 2.25817 (W 0.0417702) | - | 🔵 colmat 3.84× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.042819 ± 0.00034097 (W 0.000474851) | 0.619542 ± 0.00733341 (W 0.130888) | - | 🔵 colmat 14.47× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.109957 ± 0.00043483 (W 0.00070654) | 0.209274 ± 0.000193931 (W 0.00104171) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 3.35103 ± 0.123508 (W 0.0180026) | 4.45474 ± 0.0332744 (W 0.0274231) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 2.4335 ± 0.010922 (W 0.0169019) | 7.91018 ± 0.129753 (W 0.0588549) | - | 🔵 colmat 3.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.356469 ± 0.00318418 (W 0.0037354) | 0.492436 ± 0.00269862 (W 0.00439367) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.201018 ± 0.00129076 (W 0.00449675) | 0.352636 ± 0.00836976 (W 0.00185321) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.5012 ± 0.107856 (W 0.0904549) | 4.35846 ± 0.0447719 (W 0.0535188) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.34818 ± 0.00126207 (W 0.00367146) | 0.464462 ± 0.00192544 (W 0.00609944) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 7.80215 ± 0.0374203 (W 0.00851573) | 15.8515 ± 0.0382471 (W 0.0606243) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 7.814 ± 0.059103 (W 0.0129104) | 15.8575 ± 0.0254963 (W 0.0448316) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.586064 ± 0.000984861 (W 0.00223293) | 1.05291 ± 0.0268202 (W 0.00627693) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.590495 ± 0.00175605 (W 0.00228677) | 1.05106 ± 0.00300193 (W 0.0033371) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 5.6592 ± 0.0300546 (W 0.00562663) | 12.0705 ± 0.0256906 (W 0.0514317) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 5.69391 ± 0.0402499 (W 0.00780662) | 12.0378 ± 0.070229 (W 0.0326486) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.398694 ± 0.0022777 (W 0.00196915) | 0.857204 ± 0.00351941 (W 0.00352165) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.396465 ± 0.00154362 (W 0.00181708) | 0.867364 ± 0.00461867 (W 0.00464865) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 20.0352 ± 0.988832 (W 0.0438689) | 23.5982 ± 0.313863 (W 0.0725519) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 18.3343 ± 0.397226 (W 0.11722) | 21.9472 ± 0.167336 (W 0.0582702) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.61083 ± 0.000341294 (W 0.00140088) | 0.548334 ± 0.00906606 (W 0.00141524) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.607652 ± 0.00130816 (W 0.000881535) | 0.538924 ± 0.0106463 (W 0.00722615) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.67511 ± 0.0201141 (W 0.0119319) | 3.43698 ± 0.0110216 (W 0.0050089) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.69057 ± 0.036311 (W 0.0112414) | 3.42962 ± 0.00576302 (W 0.00837858) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.154025 ± 0.000445647 (W 0.00103856) | 0.144763 ± 0.000746906 (W 0.00135633) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.157245 ± 0.000941582 (W 0.00110813) | 0.148335 ± 0.00157617 (W 0.00131371) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.21933 ± 0.0285011 (W 0.0101745) | 7.27509 ± 0.0610429 (W 0.0112054) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 7.20258 ± 0.0562716 (W 0.0219272) | 7.29064 ± 0.0191058 (W 0.0108704) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.238463 ± 0.000163254 (W 0.00111208) | 0.239453 ± 0.000473618 (W 0.000925354) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.294282 ± 0.000487787 (W 0.00133789) | 0.243619 ± 0.000454501 (W 0.00200164) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.79776 ± 0.0369764 (W 0.010821) | 6.84827 ± 0.0296525 (W 0.00708333) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.222053 ± 0.000530614 (W 0.00152895) | 0.223273 ± 0.000498653 (W 0.00167183) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 22.0645 ± 0.613397 (W 0.0242609) | 23.6055 ± 0.388578 (W 0.0144127) | - | 🔵 colmat 1.07× faster than `faer` (marginal: spread 0.65 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 22.15 ± 0.575998 (W 0.0367003) | 23.6046 ± 0.0876173 (W 0.0290666) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.791604 ± 0.00900207 (W 0.0041456) | 0.917903 ± 0.00224186 (W 0.00128365) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.796243 ± 0.00137571 (W 0.00214754) | 1.02217 ± 0.00341978 (W 0.00891229) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 11.6082 ± 0.582342 (W 0.0146359) | 15.0964 ± 0.032721 (W 0.0347752) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 11.6176 ± 0.559687 (W 0.0156248) | 15.1028 ± 0.0318887 (W 0.0396559) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.462607 ± 0.00320285 (W 0.00328253) | 0.700808 ± 0.00166665 (W 0.00376565) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.467404 ± 0.00188407 (W 0.00371417) | 0.702669 ± 0.00257872 (W 0.00109682) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 6.34481 ± 0.0207722 (W 0.00764954) | 11.8474 ± 0.029808 (W 0.0245887) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 4.9832 ± 0.0153654 (W 0.00762473) | 11.5198 ± 0.119605 (W 0.0536611) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.292641 ± 0.00132617 (W 0.00181728) | 0.544627 ± 0.00333969 (W 0.00480866) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.273532 ± 0.00106215 (W 0.00169519) | 0.531083 ± 0.000201834 (W 0.00366539) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.01465 ± 0.00369934 (W 0.00345513) | 1.44866 ± 0.0749685 (W 0.145234) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 1.12826 ± 0.00558076 (W 0.00417885) | 1.78512 ± 0.058878 (W 0.149895) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.049208 ± 0.000164318 (W 0.000143073) | 0.075377 ± 0.000398447 (W 0.0112528) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.056548 ± 0.000176098 (W 0.000239565) | 0.083837 ± 0.00734055 (W 0.0130727) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 2.02194 ± 0.00645496 (W 0.00197211) | 2.31634 ± 0.0105415 (W 0.00524201) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 2.17935 ± 0.00735882 (W 0.0028422) | 2.5149 ± 0.0081653 (W 0.00703185) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.101466 ± 0.000468541 (W 0.00127333) | 0.149514 ± 0.000693494 (W 0.00257503) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.152496 ± 0.000345995 (W 0.00372965) | 0.162165 ± 0.000640366 (W 0.000900239) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 1.89564 ± 0.00864615 (W 0.00388618) | 2.16752 ± 0.0129771 (W 0.0047394) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.091787 ± 0.000187128 (W 0.000216148) | 0.134934 ± 0.000560941 (W 0.00119982) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 11.6602 ± 0.0587022 (W 0.0442109) | 16.3525 ± 0.0950259 (W 0.038843) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 11.8243 ± 0.0593404 (W 0.0183822) | 16.3179 ± 0.024805 (W 0.0327395) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.638499 ± 0.0050427 (W 0.00544874) | 0.963308 ± 0.00143916 (W 0.0033332) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.642758 ± 0.00106281 (W 0.00257988) | 0.97126 ± 0.00321662 (W 0.0027188) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 8.70036 ± 0.0208106 (W 0.0134365) | 13.3128 ± 0.013631 (W 0.0160547) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 8.68425 ± 0.0180072 (W 0.0130593) | 13.3247 ± 0.0189644 (W 0.0749823) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.441212 ± 0.00361455 (W 0.00274317) | 0.817425 ± 0.000578975 (W 0.00357648) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.442847 ± 0.00428471 (W 0.00286036) | 0.826786 ± 0.00215061 (W 0.00215295) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 3.38696 ± 0.00982825 (W 0.0081855) | 9.81595 ± 0.0135881 (W 0.0255986) | - | 🔵 colmat 2.90× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 2.79115 ± 0.0117755 (W 0.011267) | 9.68586 ± 0.048606 (W 0.112055) | - | 🔵 colmat 3.47× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.241402 ± 0.000929169 (W 0.00187583) | 0.550585 ± 0.0158379 (W 0.0079883) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.236298 ± 0.000109436 (W 0.00159001) | 0.535595 ± 0.00955867 (W 0.00298443) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 0.568644 ± 0.000967349 (W 0.00381601) | 1.04993 ± 0.0226261 (W 0.0407217) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 0.671184 ± 0.00464185 (W 0.0025734) | 1.25425 ± 0.00668091 (W 0.0319282) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.03231 ± 6.0781e-05 (W 0.000250252) | 0.062059 ± 0.000325131 (W 0.00534122) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.041049 ± 0.000457712 (W 0.000399189) | 0.071119 ± 0.000867393 (W 0.00449027) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 1.15895 ± 0.00931746 (W 0.00524747) | 2.3625 ± 0.0367261 (W 0.0318792) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 1.31713 ± 0.00429205 (W 0.00447403) | 2.58298 ± 0.0173444 (W 0.0220743) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.079249 ± 0.000689963 (W 0.000368222) | 0.203384 ± 0.00140296 (W 0.00378499) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.127315 ± 0.0103744 (W 0.00528822) | 0.217674 ± 0.00225544 (W 0.00289148) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 1.07139 ± 0.00410489 (W 0.00314649) | 2.15621 ± 0.027301 (W 0.0161232) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.071117 ± 0.000320004 (W 0.000204696) | 0.181294 ± 0.00171414 (W 0.00229687) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 9.91605 ± 0.0237325 (W 0.0117986) | 17.5554 ± 0.0357346 (W 0.0808936) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 9.93734 ± 0.020242 (W 0.00936051) | 17.6313 ± 0.0332841 (W 0.0413595) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.618382 ± 0.000754602 (W 0.00196466) | 1.0445 ± 0.0103815 (W 0.00540882) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.623522 ± 0.00294008 (W 0.00159416) | 1.05163 ± 0.00215058 (W 0.00557228) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 7.79587 ± 0.0233853 (W 0.00907762) | 14.2895 ± 0.0169625 (W 0.658096) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 7.84077 ± 0.556023 (W 0.00778689) | 14.3451 ± 0.00519422 (W 0.0500428) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.437206 ± 0.00138814 (W 0.00448521) | 0.883062 ± 0.00630375 (W 0.00397349) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.437823 ± 0.000154869 (W 0.00186659) | 0.885013 ± 0.00316662 (W 0.00475685) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 32.8945 ± 0.0684985 (W 0.0243531) | 34.4444 ± 0.031668 (W 0.0307938) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 38.7107 ± 0.317588 (W 0.385381) | 53.3693 ± 0.0947494 (W 0.0502988) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.506033 ± 0.00260817 (W 0.00163437) | 0.565921 ± 0.00182653 (W 0.00210493) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.941662 ± 0.00169769 (W 0.00367832) | 0.935339 ± 0.025423 (W 0.00159675) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 9.17131 ± 0.00890794 (W 0.025628) | 77.1141 ± 1.24361 (W 0.137173) | - | 🔵 colmat 8.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 11.1932 ± 0.0165704 (W 0.028143) | 15.5552 ± 0.258368 (W 0.0186846) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.287171 ± 0.00114663 (W 0.00165276) | 2.32564 ± 0.00434404 (W 0.00987379) | - | 🔵 colmat 8.10× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.391248 ± 0.00144357 (W 0.00209446) | 0.434447 ± 0.00144038 (W 0.00198237) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 22.8417 ± 0.0112101 (W 0.0232296) | 22.8312 ± 0.587315 (W 0.0194829) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 22.7106 ± 0.0557226 (W 0.0202964) | 26.5417 ± 0.227312 (W 0.0216994) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.844615 ± 0.00394512 (W 0.00388864) | 0.854211 ± 0.00261783 (W 0.00283547) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.745738 ± 0.00104237 (W 0.00200318) | 0.803963 ± 0.00375424 (W 0.00238167) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 21.8046 ± 0.519726 (W 0.0287044) | 21.6316 ± 0.0383501 (W 0.0149458) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.796724 ± 0.00265613 (W 0.00183127) | 0.812854 ± 0.00158917 (W 0.00341079) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 52.7157 ± 0.134349 (W 0.0697967) | 56.4869 ± 0.0775212 (W 0.051991) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 52.7912 ± 0.0602418 (W 0.084769) | 56.5359 ± 0.107869 (W 0.0903262) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.48144 ± 0.000891353 (W 0.00238286) | 1.70306 ± 0.00323478 (W 0.00383647) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.49028 ± 0.00512714 (W 0.00260134) | 1.70522 ± 0.00620831 (W 0.00403254) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 23.5967 ± 0.0472317 (W 0.0491061) | 28.3923 ± 0.115012 (W 0.0462325) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 23.5429 ± 0.0748205 (W 0.0438569) | 28.2984 ± 0.065547 (W 0.015173) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.804433 ± 0.00121452 (W 0.00173304) | 1.09549 ± 0.00166185 (W 0.00200018) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.805034 ± 0.00254427 (W 0.00240273) | 1.09732 ± 0.00149065 (W 0.000769419) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 9.20307 ± 0.0125383 (W 0.00911956) | 11.3934 ± 0.0184537 (W 0.0110942) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 11.702 ± 0.0564743 (W 0.0191683) | 29.1858 ± 0.568845 (W 2.38657) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.256371 ± 0.00273891 (W 0.00145121) | 0.565431 ± 0.00145344 (W 0.00308622) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.411136 ± 0.00225585 (W 0.00328618) | 0.928712 ± 0.00879289 (W 0.00191882) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.66787 ± 0.00695113 (W 0.010835) | 38.759 ± 0.36684 (W 7.12177) | - | 🔵 colmat 14.53× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 4.35698 ± 0.0647112 (W 0.0563868) | 9.20009 ± 0.126818 (W 0.602096) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.112775 ± 0.000608939 (W 0.000824249) | 1.19216 ± 0.00313617 (W 0.216) | - | 🔵 colmat 10.57× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.188534 ± 0.000448778 (W 0.00378397) | 0.293328 ± 0.0133755 (W 0.0228566) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 7.68819 ± 0.0112504 (W 0.0182343) | 8.12991 ± 0.0841836 (W 0.0203427) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 7.4621 ± 0.0520923 (W 0.0460326) | 11.6552 ± 0.865583 (W 0.0233932) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.454941 ± 0.00115316 (W 0.00346443) | 0.474421 ± 0.00101008 (W 0.00354694) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.325141 ± 0.000626969 (W 0.00146399) | 0.439172 ± 0.0054692 (W 0.00167081) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 7.39633 ± 0.0260413 (W 0.0197102) | 7.76923 ± 0.0129765 (W 0.0172329) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.432296 ± 0.00261422 (W 0.00238303) | 0.447385 ± 0.00130999 (W 0.00233865) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 21.6805 ± 0.0331482 (W 0.02952) | 28.6333 ± 0.0265584 (W 0.066811) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 21.8272 ± 0.116953 (W 0.0569561) | 28.7478 ± 0.0703182 (W 0.150273) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.936879 ± 0.00281507 (W 0.00286875) | 1.40042 ± 0.00404133 (W 0.00307462) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.945329 ± 0.00496879 (W 0.00251998) | 1.41056 ± 0.00240169 (W 0.00367317) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 12.9015 ± 0.0150256 (W 0.0226184) | 19.7815 ± 0.00688083 (W 0.0358811) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 12.9288 ± 0.0110429 (W 0.0133675) | 19.8426 ± 0.00679259 (W 0.0215516) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.616529 ± 0.00149108 (W 0.00276443) | 1.08255 ± 0.000794668 (W 0.00267861) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.616954 ± 0.000957551 (W 0.00199497) | 1.08686 ± 0.000797624 (W 0.00221185) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 5.21517 ± 0.00905454 (W 0.00822324) | 7.58154 ± 0.0165676 (W 0.0428122) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 6.91486 ± 0.0172077 (W 0.0117742) | 25.0291 ± 5.48052 (W 0.0873659) | - | 🔵 colmat 3.62× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.230563 ± 0.0011843 (W 0.00139864) | 0.567018 ± 0.000733173 (W 0.00177791) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.343967 ± 0.000766457 (W 0.00466126) | 0.932431 ± 0.00281432 (W 0.00184757) | - | 🔵 colmat 2.71× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.60499 ± 0.0127505 (W 0.00437124) | 21.1822 ± 1.24699 (W 0.127775) | - | 🔵 colmat 13.20× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 3.08814 ± 0.00566298 (W 0.00657011) | 7.61221 ± 0.523777 (W 0.0526014) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.090949 ± 0.000612836 (W 0.000821635) | 0.684344 ± 0.000979364 (W 0.105088) | - | 🔵 colmat 7.52× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.158044 ± 0.00027806 (W 0.000878496) | 0.241961 ± 0.00452775 (W 0.00240221) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 5.43205 ± 0.238819 (W 0.0174937) | 6.89113 ± 0.11542 (W 0.0602368) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 4.79008 ± 0.0418008 (W 0.0137787) | 10.7529 ± 0.532957 (W 0.0875552) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.381296 ± 0.000741243 (W 0.00440217) | 0.500606 ± 0.0035193 (W 0.0078675) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.256643 ± 0.000497322 (W 0.00175509) | 0.472445 ± 0.00451504 (W 0.00759934) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 5.23422 ± 0.633631 (W 0.0263262) | 6.47118 ± 0.190388 (W 0.167158) | - | 🔵 colmat 1.24× faster than `faer` (marginal: spread 0.67 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.363537 ± 0.000861878 (W 0.00301839) | 0.483766 ± 0.000900156 (W 0.00917753) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 16.4809 ± 0.0305616 (W 0.04673) | 28.8122 ± 0.116034 (W 0.127298) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 16.6605 ± 0.186743 (W 0.046336) | 28.9079 ± 0.023425 (W 0.858803) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.864714 ± 0.00615623 (W 0.00170315) | 1.47093 ± 0.00469001 (W 0.00569369) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.870817 ± 0.000871632 (W 0.00221539) | 1.47468 ± 0.0108206 (W 0.00411853) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 10.7387 ± 0.0410196 (W 0.0201449) | 20.6417 ± 0.0690888 (W 0.0365431) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 10.7291 ± 0.0302468 (W 0.0316939) | 20.654 ± 0.0680529 (W 0.0468285) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.592157 ± 0.00213266 (W 0.00232225) | 1.14787 ± 0.00161298 (W 0.00184048) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.593417 ± 0.00642912 (W 0.00191832) | 1.14537 ± 0.00187466 (W 0.0028295) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 1T (CPU 0)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 41.6199 ± 0.0729149 (W 0.0539924) | 52.0953 ± 0.392608 (W 0.0591239) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 37.4133 ± 0.0523428 (W 0.406528) | 47.3633 ± 0.153858 (W 0.114733) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.870214 ± 0.00173841 (W 0.001146) | 0.802394 ± 0.00103032 (W 0.0016293) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.860262 ± 0.0022634 (W 0.00157344) | 0.798733 ± 0.213372 (W 0.00281382) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1` | 9.71002 ± 0.010497 (W 0.00675037) | 9.18369 ± 0.0234772 (W 0.00529202) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.89117 ± 0.00231819 (W 0.00635633) | 9.16463 ± 0.0238527 (W 0.00780337) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.302371 ± 0.00098954 (W 0.00163049) | 0.29073 ± 0.00219194 (W 0.00133164) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.314539 ± 0.0015438 (W 0.00163145) | 0.29896 ± 0.000168001 (W 0.00153482) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 21.1087 ± 0.0535508 (W 0.0150313) | 20.933 ± 0.0421704 (W 0.0212364) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 21.3058 ± 0.0455094 (W 0.0257636) | 21.1456 ± 0.0324981 (W 0.0231668) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.651791 ± 0.00270537 (W 0.00210799) | 0.651999 ± 0.00223892 (W 0.0020004) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.66359 ± 0.0007285 (W 0.00213107) | 0.661428 ± 0.00151399 (W 0.00134529) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 19.8495 ± 0.0510129 (W 0.0227591) | 19.7718 ± 0.0255332 (W 0.0235874) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.61375 ± 0.00057322 (W 0.0021641) | 0.612662 ± 0.00315286 (W 0.0016515) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 56.6814 ± 0.0101253 (W 0.0520486) | 60.2648 ± 0.0550142 (W 0.0574929) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 56.792 ± 0.0418144 (W 0.105334) | 60.4091 ± 0.0784323 (W 0.0714173) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.54701 ± 0.00361517 (W 0.00572151) | 1.72464 ± 0.00238509 (W 0.0022823) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.55388 ± 0.00553094 (W 0.00339113) | 1.7307 ± 0.000566137 (W 0.00391859) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 27.5496 ± 0.0297459 (W 0.0497277) | 32.1788 ± 0.0401317 (W 0.0134583) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 27.5877 ± 0.0478806 (W 0.0176964) | 32.1497 ± 0.0180535 (W 0.0315521) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.859775 ± 0.00416512 (W 0.00482838) | 1.14741 ± 0.00295948 (W 0.0032125) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.859845 ± 0.00160567 (W 0.00435921) | 1.14804 ± 0.00140967 (W 0.00311626) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 4T (CPU 0-3)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 12.4549 ± 0.119265 (W 0.0106902) | 23.794 ± 0.062889 (W 0.0550518) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 10.3414 ± 0.021174 (W 0.018536) | 22.557 ± 0.158564 (W 0.0453492) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.348156 ± 0.00161308 (W 0.00326522) | 0.801666 ± 0.00254166 (W 0.00239054) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.329879 ± 0.00177926 (W 0.00179444) | 0.793627 ± 0.00206539 (W 0.0036747) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.65331 ± 0.00445153 (W 0.00610648) | 3.36604 ± 0.171489 (W 0.232739) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 2.90431 ± 0.00673953 (W 0.0061669) | 3.75399 ± 0.299808 (W 0.254893) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.093887 ± 0.000151659 (W 0.00129172) | 0.116586 ± 0.0114573 (W 0.023991) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.108205 ± 0.000196225 (W 0.000332472) | 0.132876 ± 0.0106293 (W 0.018495) | - | 🔵 colmat 1.23× faster than `faer` (marginal: spread 0.75 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 5.69828 ± 0.0560358 (W 0.00959494) | 5.98874 ± 0.0108383 (W 0.011636) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 6.0304 ± 0.0107558 (W 0.00421994) | 6.26067 ± 0.000660728 (W 0.0854951) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.226721 ± 0.000301144 (W 0.00118273) | 0.278299 ± 0.00157136 (W 0.00113999) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.24492 ± 0.00087114 (W 0.00143129) | 0.301601 ± 0.00125433 (W 0.00239783) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 5.35756 ± 0.0444728 (W 0.00475243) | 5.62299 ± 0.0780014 (W 0.011071) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.205613 ± 3.9e-05 (W 0.00116257) | 0.2511 ± 0.000584972 (W 0.00153102) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 25.6132 ± 0.0368978 (W 0.0252015) | 32.7168 ± 0.0809659 (W 0.0865657) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 25.758 ± 0.0412145 (W 0.0529609) | 32.7316 ± 0.0546143 (W 0.0381214) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.982701 ± 0.0104932 (W 0.00276559) | 1.42466 ± 0.00495015 (W 0.00354288) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.994587 ± 0.0102934 (W 0.00279821) | 1.42895 ± 0.0177718 (W 0.00443795) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 16.8377 ± 0.020989 (W 0.0142654) | 23.783 ± 0.0226714 (W 0.0432284) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 16.8727 ± 0.0230313 (W 0.0191461) | 23.8311 ± 0.0228632 (W 0.0409796) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.6753 ± 0.00597257 (W 0.00289542) | 1.12587 ± 0.00189703 (W 0.00429491) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.661321 ± 0.00239763 (W 0.00412607) | 1.13182 ± 0.00739265 (W 0.00545277) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 8T (CPU 0-7)

| case | colmat (ms) | faer (ms) | lapack-mkl (ms) | vs `faer` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 6.32686 ± 0.0420746 (W 0.0133778) | 18.9734 ± 0.110375 (W 0.0955507) | - | 🔵 colmat 3.00× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 5.5412 ± 0.0175846 (W 0.012815) | 18.3594 ± 0.0549288 (W 0.0775109) | - | 🔵 colmat 3.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.263249 ± 0.00188281 (W 0.00194306) | 0.804109 ± 0.000919401 (W 0.00237582) | - | 🔵 colmat 3.05× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.25947 ± 0.0017194 (W 0.00214714) | 0.790062 ± 0.0036332 (W 0.00221731) | - | 🔵 colmat 3.04× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.45123 ± 0.0011553 (W 0.00439605) | 2.03355 ± 0.066752 (W 0.0970604) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 1.73197 ± 0.000944591 (W 0.00522145) | 2.30574 ± 0.0212154 (W 0.05634) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.059937 ± 0.00062008 (W 0.000176449) | 0.095387 ± 0.00227275 (W 0.010084) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.076247 ± 0.000756352 (W 0.000289482) | 0.113396 ± 0.000561064 (W 0.00202275) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 3.15453 ± 0.00209444 (W 0.00437771) | 4.84666 ± 0.0475643 (W 0.136133) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 3.43928 ± 0.297763 (W 0.00480317) | 5.03738 ± 0.152269 (W 0.172262) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.156406 ± 0.00040097 (W 0.00191931) | 0.311448 ± 0.00374519 (W 0.00484028) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.174514 ± 0.000108076 (W 0.00103495) | 0.3403 ± 0.00210098 (W 0.00401719) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 2.94611 ± 0.00369363 (W 0.0060368) | 4.42844 ± 0.0816228 (W 0.0896186) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.140429 ± 0.000258043 (W 0.00116849) | 0.284962 ± 0.00274424 (W 0.00435614) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 20.459 ± 0.027225 (W 0.0582179) | 32.9755 ± 0.0943892 (W 0.10257) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 20.6186 ± 0.0625918 (W 0.0335469) | 33.3345 ± 2.20402 (W 0.757504) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 0.906134 ± 0.00234945 (W 0.00267854) | 1.4884 ± 0.00289072 (W 0.00340964) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.912711 ± 0.0024029 (W 0.00320078) | 1.49432 ± 0.00221722 (W 0.00532456) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 14.6463 ± 0.01616 (W 1.0249) | 24.6584 ± 0.0242491 (W 0.059382) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 14.645 ± 0.0126125 (W 0.0195686) | 24.7361 ± 0.0347974 (W 0.0481925) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.636571 ± 0.0018466 (W 0.00597913) | 1.18815 ± 0.00537511 (W 0.00328449) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.648263 ± 0.00194959 (W 0.00227525) | 1.18935 ± 0.00287055 (W 0.00567252) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `lu/256x4096`, `lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `solve/256x4096`, `solve/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
