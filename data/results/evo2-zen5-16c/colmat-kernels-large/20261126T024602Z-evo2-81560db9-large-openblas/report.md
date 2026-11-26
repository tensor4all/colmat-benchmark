# `colmat-kernels-large` on `evo2-zen5-16c`

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
- timestamp: `2026-11-26T03:08:13.523823Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py evo2-zen5-16c colmat-kernels-large --vendor openblas --engines colmat,lapack-openblas --threads 4,8 --regimes linalg-large,gemm-large,linalg-boundary --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 --reps 5 --aa 6 --prime-ms 250 --label evo2-81560db9-large-openblas`
- harness environment: `FBENCH_ORDER` unset, `FBENCH_PROFILE=evo2-zen5-16c`
- raw data: `data/results/evo2-zen5-16c/colmat-kernels-large/20261126T024602Z-evo2-81560db9-large-openblas/`

## What was measured

- **Workload** — Direct calls into one colmat revision and its upstream faer baseline: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the colmat and faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: Vendor C-order cells use CBLAS/LAPACKE row-major entry points. Workspace allocation or layout conversion internal to those APIs remains timed; these are not prepared-column-major or transpose measurements.
  Known blind spot: `gemm` covers the tall class's two Householder legs (a 16-wide basis and a 240-column target over 4096 rows: the reduction and the write-back) beside a vendor `?gemm` on the same leg. A single leg beside a vendor gemm is a diagnostic reference, not the equivalent implementation of the whole update, and the row is labelled accordingly.
  Known blind spot: `householder_apply` measures block apply and the one-block sequence form as two distinct rows, labelled with basis width min(m,n) and target width n. QR fixture preparation is outside timing. Independently varied basis/target widths and multi-panel sequences are not covered.
  Known blind spot: Allocation performed inside the measured call stays inside the clock. The fork's two row splits allocate their partial buffers inside the public call, and the public scratch function does not expose them; the campaign reports that as it is instead of changing colmat's API to make the boundary look cleaner.
  Known blind spot: `cholesky`, `eigh`, `solve` and `triangular_solve` are measured on square shapes only: the matrix they factor or solve with is square by definition (self-adjoint for the first two, a triangular factor for the last). `lu` and `full_piv_lu` are not gated that way -- they take the declared rectangular shapes too, with `min(m, n)` pivots and a trapezoidal `L` or `U`, which is what `dgetrf` does as well -- so a rectangular LU cell is measured in both orders beside its square one.
  Known blind spot: BLIS, sparse, iterative and GPU arms are later declarations, not machinery built now.
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
- CPU sets: 4T -> `0-3`, 8T -> `0-7`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 6 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.25%, worst 137.48% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 18 of 392 engine cells exceed 5% and carry `!`; median within-process relative SD 0.35%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

## linalg-large, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 2.55823 ± 0.0110219 (W 0.00469804) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 2.43976 ± 0.0462802 (W 0.00529111) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 6.64969 ± 0.0117157 (W 0.00663501) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 7.61508 ± 0.14624 (W 0.0453165) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1024/b1` | 4.37049 ± 0.0143891 (W 0.0104869) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 9.34677 ± 0.110873 (W 0.0213024) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 14.0999 ± 0.0902902 (W 0.0868773) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 26.6235 ± 0.212296 (W 0.153129) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 4.27586 ± 0.0661907 (W 0.0264821) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 9.36406 ± 0.11362 (W 0.0512846) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 13.5425 ± 0.0400069 (W 0.0501941) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 16.6169 ± 0.230048 (W 0.0834281) | - | - | `*` one row has no measurement here |

## linalg-large, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 1.54815 ± 0.0194664 (W 0.00315576) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 1.87381 ± 0.0689221 (W 0.0114142) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n1536/b1` | 3.69911 ± 0.00483433 (W 0.00939467) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 5.50297 ± 0.0913709 (W 0.0130999) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 3.00726 ± 0.0299254 (W 0.0193217) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 8.21822 ± 0.0913285 (W 0.0415205) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 8.40953 ± 0.0241484 (W 0.0331411) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 22.4477 ± 0.295569 (W 0.16315) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 2.88104 ± 0.0836889 (W 0.0206683) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n1024/b1 [api-inclusive]` | 4.33016 ± 0.0752074 (W 0.0187308) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 8.3664 ± 0.0531813 (W 0.0311722) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 12.5796 ± 0.267487 (W 0.0667818) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-large, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 1.89047 ± 0.00941519 (W 0.0053618) | 3.07855 ± 0.0757056 (W 0.00815825) | 🔵 colmat 1.63× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n1024/b1 [api-inclusive]` | 2.16239 ± 0.0503762 (W 0.0038105) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n1536/b1` | 5.59079 ± 0.0284117 (W 0.0127666) | 8.1754 ± 0.021067 (W 0.0163507) | 🔵 colmat 1.46× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 6.48676 ± 0.0944029 (W 0.0153751) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 3.75713 ± 0.00966756 (W 0.0072718) | 5.41072 ± 0.00848396 (W 0.0105397) | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `lu/n1024/b1 [api-inclusive]` | 7.61985 ± 0.0413707 (W 0.0155678) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 10.4589 ± 0.0250405 (W 0.0203311) | 13.3398 ± 0.092529 (W 0.0166673) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `lu/n1536/b1 [api-inclusive]` | 21.8957 ± 0.185796 (W 0.110742) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 3.60963 ± 0.00756272 (W 0.00737498) | 5.27737 ± 0.340908 (W 0.00567612) | 🔵 colmat 1.46× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n1024/b1 [api-inclusive]` | 7.53764 ± 0.0634109 (W 0.0199121) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 10.2005 ± 0.0261009 (W 0.0204657) | 12.9406 ± 0.0271929 (W 0.0118665) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `solve/n1536/b1 [api-inclusive]` | 12.5469 ± 0.0631616 (W 0.025535) | - | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 1.1577 ± 0.00879929 (W 0.00385599) | 2.70375 ± 0.349146 (W 0.0171539) | 🔵 colmat 2.34× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n1024/b1 [api-inclusive]` | 1.56236 ± 0.0312788 (W 0.00528797) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 3.12331 ± 0.0313888 (W 0.00985137) | 5.2512 ± 0.0811353 (W 0.0319682) | 🔵 colmat 1.68× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 4.41583 ± 0.0546242 (W 0.00948339) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 2.60911 ± 0.0121805 (W 0.00865459) | 3.11263 ± 0.0202575 (W 0.00597262) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `lu/n1024/b1 [api-inclusive]` | 6.50098 ± 0.0348279 (W 0.0298207) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 6.63443 ± 0.0190742 (W 0.0151817) | 7.05744 ± 0.0148429 (W 0.00884125) | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `lu/n1536/b1 [api-inclusive]` | 17.8579 ± 0.142113 (W 0.12985) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 2.35844 ± 0.00607214 (W 0.0053203) | 3.05763 ± 0.0277344 (W 0.00703475) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `solve/n1024/b1 [api-inclusive]` | 3.09185 ± 0.0167263 (W 0.00939568) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 6.29528 ± 0.0115785 (W 0.0129042) | 6.9479 ± 0.0141272 (W 0.0103173) | 🔵 colmat 1.10× faster than `lapack-openblas` |  |
| `solve/n1536/b1 [api-inclusive]` | 8.76611 ± 0.0413635 (W 0.0217741) | - | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 5.87421 ± 0.0263575 (W 0.00820992) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 7.05626 ± 0.100318 (W 0.0177932) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 17.8854 ± 0.133091 (W 0.0264705) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 32.6834 ± 0.506932 (W 0.116148) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 13.3694 ± 0.08836 (W 0.055457) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 24.726 ± 0.342989 (W 0.140263) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 45.7652 ± 0.37577 (W 0.296417) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 67.2183 ± 0.714914 (W 0.5253) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 13.0924 ± 0.0569106 (W 0.0297533) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 20.4415 ± 4.37182 (W 0.0786749) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n1536/b1` | 45.8641 ± 0.421329 (W 0.305216) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 67.4141 ± 0.607801 (W 0.297342) | - | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 3.40116 ± 0.0348461 (W 0.00511972) | - | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 4.86821 ± 0.11505 (W 0.0211848) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n1536/b1` | 9.64067 ± 0.0261605 (W 0.0114608) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 24.563 ± 0.429577 (W 0.146495) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 8.20145 ± 0.054693 (W 0.0289717) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 20.5651 ± 0.199398 (W 0.0875281) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 29.8588 ± 0.268753 (W 0.255279) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 53.7113 ± 0.420094 (W 0.408368) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 8.41129 ± 0.127539 (W 0.0269288) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 12.0791 ± 0.348161 (W 0.0663608) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n1536/b1` | 30.1796 ± 0.209768 (W 0.203575) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 53.3765 ± 0.431669 (W 0.363356) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 5.74756 ± 0.0536374 (W 0.0105942) | 7.84223 ± 0.039987 (W 0.0111691) | 🔵 colmat 1.36× faster than `lapack-openblas` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 6.45277 ± 0.0478595 (W 0.00788338) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 17.9321 ± 0.134853 (W 0.032364) | 24.554 ± 0.0439818 (W 0.0393837) | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 29.8866 ± 0.258309 (W 0.0400797) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 11.1742 ± 0.0328648 (W 0.0186627) | 12.7755 ± 0.0265705 (W 0.0174383) | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `lu/n1024/b1 [api-inclusive]` | 21.0269 ± 0.224706 (W 0.0578009) | - | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 34.9564 ± 0.034214 (W 0.0746121) | 39.7387 ± 0.0713943 (W 0.0207319) | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `lu/n1536/b1 [api-inclusive]` | 58.9819 ± 0.259457 (W 0.231404) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 11.076 ± 0.0328446 (W 0.0158433) | 12.8065 ± 0.0169913 (W 0.0203881) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `solve/n1024/b1 [api-inclusive]` | 20.9145 ± 4.19749 (W 0.0474922) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n1536/b1` | 34.8837 ± 0.040379 (W 0.0328047) | 39.9566 ± 0.0947429 (W 0.0294649) | 🔵 colmat 1.15× faster than `lapack-openblas` |  |
| `solve/n1536/b1 [api-inclusive]` | 59.5454 ± 0.800886 (W 0.168705) | - | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n1024/b1` | 3.33957 ± 0.0201189 (W 0.0078421) | 4.64266 ± 0.165449 (W 0.0173322) | 🔵 colmat 1.39× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n1024/b1 [api-inclusive]` | 4.23392 ± 0.047089 (W 0.00852088) | - | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 9.76746 ± 0.0597354 (W 0.0180953) | 13.0678 ± 0.063062 (W 0.0126709) | 🔵 colmat 1.34× faster than `lapack-openblas` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 21.9866 ± 0.311889 (W 0.0432766) | - | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 7.074 ± 0.0271114 (W 0.015417) | 6.75131 ± 0.147778 (W 0.0365614) | · | `!` between-start range over 5% |
| `lu/n1024/b1 [api-inclusive]` | 16.8381 ± 0.37199 (W 0.0886969) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1536/b1` | 21.1122 ± 0.0610475 (W 0.112289) | 20.3618 ± 0.0572085 (W 0.0432063) | · |  |
| `lu/n1536/b1 [api-inclusive]` | 45.4132 ± 0.710462 (W 0.202792) | - | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 6.91292 ± 0.0237853 (W 0.0159368) | 6.87317 ± 0.040512 (W 0.0173199) | · |  |
| `solve/n1024/b1 [api-inclusive]` | 8.99052 ± 0.0383313 (W 0.0127988) | - | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 21.3089 ± 0.0248439 (W 0.0317108) | 20.9606 ± 0.0391191 (W 0.0470115) | · |  |
| `solve/n1536/b1 [api-inclusive]` | 45.594 ± 0.823977 (W 0.2324) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024`, `cholesky/1536`, `lu/1024`, `lu/1536`, `solve/1024`, `solve/1536`.

## gemm-large, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 6.11566 ± 0.0196463 (W 0.0103615) | 7.20255 ± 0.0201184 (W 0.0273555) | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 21.5589 ± 0.137689 (W 0.0144384) | 24.0129 ± 0.0826567 (W 0.0137301) | 🔵 colmat 1.11× faster than `lapack-openblas` |  |

## gemm-large, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 3.13594 ± 0.00505869 (W 0.00708315) | 3.73034 ± 0.0042631 (W 0.00961141) | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 11.2612 ± 0.0268367 (W 0.0175985) | 12.3079 ± 0.0198076 (W 0.0444721) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |

## gemm-large, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 6.13575 ± 0.0340795 (W 0.0117556) | 7.20779 ± 0.00816033 (W 0.0215353) | 🔵 colmat 1.17× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 21.5061 ± 0.133726 (W 0.0196261) | 23.9724 ± 0.14775 (W 0.0374801) | 🔵 colmat 1.11× faster than `lapack-openblas` |  |

## gemm-large, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 3.13507 ± 0.00602912 (W 0.00533244) | 3.73 ± 2.0885 (W 0.0111881) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/1536x1536x1536/b1` | 11.2999 ± 0.18886 (W 0.0336018) | 12.3142 ± 0.029348 (W 0.0209527) | 🔵 colmat 1.09× faster than `lapack-openblas` |  |

## gemm-large, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 23.6754 ± 0.0380466 (W 0.0317644) | 29.6391 ± 0.0617139 (W 0.0261849) | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 79.8783 ± 0.260504 (W 0.0537457) | 98.9975 ± 0.346635 (W 0.0411014) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |

## gemm-large, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 11.9948 ± 0.0166843 (W 0.00604604) | 15.1343 ± 0.0464274 (W 0.0130811) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 40.4827 ± 0.0456 (W 0.0195758) | 49.8727 ± 0.0959069 (W 0.0691815) | 🔵 colmat 1.23× faster than `lapack-openblas` |  |

## gemm-large, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 23.734 ± 0.080231 (W 0.0206216) | 29.594 ± 0.0576882 (W 0.0224181) | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 79.9161 ± 0.180797 (W 0.036815) | 98.8105 ± 0.457752 (W 0.0586181) | 🔵 colmat 1.24× faster than `lapack-openblas` |  |

## gemm-large, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 11.9993 ± 0.0265545 (W 0.00831972) | 15.1629 ± 0.0315718 (W 0.0107846) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `gemm/1536x1536x1536/b1` | 40.5382 ± 0.0883913 (W 0.024938) | 49.7393 ± 0.136925 (W 0.0809744) | 🔵 colmat 1.23× faster than `lapack-openblas` |  |

## linalg-boundary, f64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.090477 ± 0.000573291 (W 0.000372092) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.109716 ± 0.00130275 (W 0.000396516) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.084187 ± 0.000654299 (W 0.000403336) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.105487 ± 0.00196506 (W 0.000734576) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.342624 ± 0.00196627 (W 0.00206893) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.463669 ± 0.00449835 (W 0.00168004) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.332924 ± 0.00175752 (W 0.0022325) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.500398 ± 0.00786697 (W 0.0022666) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.246085 ± 0.00145717 (W 0.00204096) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.427455 ± 0.00618395 (W 0.00206176) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.232492 ± 0.001436 (W 0.00220318) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.455983 ± 0.003713 (W 0.00282209) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 0.891935 ± 0.00144806 (W 0.00248661) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 1.97216 ± 0.00861209 (W 0.00307834) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 0.964759 ± 0.00498939 (W 0.00395372) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 1.9478 ± 0.0176802 (W 0.00528215) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.193113 ± 0.00186582 (W 0.00200542) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.42637 ± 0.00373079 (W 0.0028442) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.231542 ± 0.00156993 (W 0.00221341) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.456551 ± 0.00736235 (W 0.00204263) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 0.873867 ± 0.00683405 (W 0.00368255) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 1.92498 ± 0.0107103 (W 0.00256254) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 0.929194 ± 0.00521771 (W 0.00328306) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 1.92972 ± 0.0166695 (W 0.00387656) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.087887 ± 0.000675758 (W 0.000497488) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.107316 ± 0.000895289 (W 0.000566825) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.0819515 ± 0.000415066 (W 0.000885515) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.100956 ± 0.00106817 (W 0.000772667) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.273757 ± 0.00120212 (W 0.00200171) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.42011 ± 0.00393727 (W 0.00380193) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.267891 ± 0.00156377 (W 0.0015574) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.459359 ± 0.00925832 (W 0.00260476) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.251625 ± 0.00117119 (W 0.00238953) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.413636 ± 0.00358599 (W 0.00203154) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.200539 ± 0.00152366 (W 0.00205181) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.442851 ± 0.00164701 (W 0.00226438) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 0.666516 ± 0.00184479 (W 0.00312934) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 1.81051 ± 0.0214412 (W 0.0032421) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 0.714063 ± 0.00709005 (W 0.00318421) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 1.78105 ± 0.0234653 (W 0.00363956) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.197094 ± 0.000665572 (W 0.00198867) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.420964 ± 0.00355047 (W 0.00267939) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.199322 ± 0.000865472 (W 0.00154584) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.449539 ± 0.00187935 (W 0.00240174) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 0.650574 ± 0.00171266 (W 0.00300912) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 1.79545 ± 0.00381969 (W 0.00698798) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 0.699852 ± 0.00462386 (W 0.00400732) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 1.76163 ± 0.0197079 (W 0.00515583) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.0867515 ± 0.000590032 (W 0.000661188) | 0.170953 ± 0.000630683 (W 0.00119136) | 🔵 colmat 1.97× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.112576 ± 0.000227134 (W 0.000361822) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.083922 ± 0.000720796 (W 0.000627081) | 0.149495 ± 0.00149228 (W 0.00164117) | 🔵 colmat 1.78× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.108361 ± 0.000502642 (W 0.000905577) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.333733 ± 0.00102267 (W 0.00214646) | 0.619244 ± 0.00180884 (W 0.00254776) | 🔵 colmat 1.86× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.467091 ± 0.0036557 (W 0.0021236) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.342113 ± 0.00077698 (W 0.00254365) | 0.613355 ± 0.00416034 (W 0.0023876) | 🔵 colmat 1.79× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.42074 ± 0.00130973 (W 0.00217635) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.164919 ± 0.000423145 (W 0.000897567) | 0.239021 ± 0.000107262 (W 0.00147324) | 🔵 colmat 1.45× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 0.364008 ± 0.00365265 (W 0.00204056) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.172789 ± 0.000312264 (W 0.00135455) | 0.244611 ± 0.00100311 (W 0.00227638) | 🔵 colmat 1.42× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 0.382412 ± 0.00247872 (W 0.00334736) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 0.718792 ± 0.00218331 (W 0.002822) | 1.16333 ± 0.00435145 (W 0.00146769) | 🔵 colmat 1.62× faster than `lapack-openblas` |  |
| `lu/n511/b1 [api-inclusive]` | 1.65797 ± 0.00672016 (W 0.00426113) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 0.774062 ± 0.00204439 (W 0.0022064) | 1.17523 ± 0.00268011 (W 0.00239056) | 🔵 colmat 1.52× faster than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 1.65079 ± 0.0106824 (W 0.00319551) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.162169 ± 0.000594847 (W 0.00148883) | 0.228632 ± 0.000899077 (W 0.00210753) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 0.367284 ± 0.00077221 (W 0.0026468) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.168699 ± 0.000653272 (W 0.00139502) | 0.232793 ± 0.00059522 (W 0.00185952) | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 0.385833 ± 0.00411555 (W 0.00300821) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 0.71062 ± 0.00354307 (W 0.00211167) | 1.11679 ± 0.00825586 (W 0.00368717) | 🔵 colmat 1.57× faster than `lapack-openblas` |  |
| `solve/n511/b1 [api-inclusive]` | 1.6214 ± 0.00848206 (W 0.00297374) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 0.740343 ± 0.00232154 (W 0.00304299) | 1.13218 ± 0.00285086 (W 0.00335661) | 🔵 colmat 1.53× faster than `lapack-openblas` |  |
| `solve/n513/b1 [api-inclusive]` | 1.61603 ± 0.00301289 (W 0.00571946) | - | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.082942 ± 0.000441351 (W 0.000464348) | 0.171174 ± 0.00116421 (W 0.00131075) | 🔵 colmat 2.06× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.110316 ± 0.000675007 (W 0.000516963) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.0799965 ± 0.000679556 (W 0.000661343) | 0.151404 ± 0.00032926 (W 0.00218924) | 🔵 colmat 1.89× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.103751 ± 0.000595375 (W 0.000420314) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.275725 ± 0.00311323 (W 0.00172992) | 0.556422 ± 0.000430785 (W 0.00259673) | 🔵 colmat 2.02× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.405122 ± 0.000851804 (W 0.00214337) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.280469 ± 0.00260048 (W 0.00149887) | 0.562729 ± 0.00272337 (W 0.00275121) | 🔵 colmat 2.01× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.373146 ± 0.00144326 (W 0.00223486) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.149715 ± 0.00105035 (W 0.00140455) | 0.220286 ± 0.00136334 (W 0.0020416) | 🔵 colmat 1.47× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 0.350074 ± 0.00283939 (W 0.00164321) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.158618 ± 0.000723796 (W 0.00124851) | 0.222733 ± 0.00113242 (W 0.00201068) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 0.372936 ± 0.00208204 (W 0.00237363) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 0.560004 ± 0.0024056 (W 0.00263224) | 0.813454 ± 0.0037929 (W 0.00379445) | 🔵 colmat 1.45× faster than `lapack-openblas` |  |
| `lu/n511/b1 [api-inclusive]` | 1.50209 ± 0.00688237 (W 0.00394688) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 0.594568 ± 0.00134404 (W 0.00273977) | 0.8257 ± 0.00190311 (W 0.00355791) | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 1.46968 ± 0.00667805 (W 0.00432049) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.150359 ± 0.000687449 (W 0.00119611) | 0.230667 ± 0.00107797 (W 0.00251117) | 🔵 colmat 1.53× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 0.358032 ± 0.00100157 (W 0.00220699) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.155625 ± 0.00105876 (W 0.00181287) | 0.231922 ± 0.00117439 (W 0.00273713) | 🔵 colmat 1.49× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 0.377108 ± 0.00110013 (W 0.00291896) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 0.551246 ± 0.00264152 (W 0.00291874) | 0.796834 ± 0.0163345 (W 0.00354673) | 🔵 colmat 1.45× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n511/b1 [api-inclusive]` | 1.50631 ± 0.00407908 (W 0.00415376) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 0.575754 ± 0.00372844 (W 0.00275036) | 0.807797 ± 0.00730408 (W 0.00545225) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `solve/n513/b1 [api-inclusive]` | 1.46657 ± 0.00992095 (W 0.00420609) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.241887 ± 0.00093237 (W 0.00112525) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.297498 ± 0.000902914 (W 0.00164514) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.221323 ± 0.000899456 (W 0.00142251) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.275101 ± 0.00109564 (W 0.00151131) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.984009 ± 0.00254387 (W 0.00226261) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 1.24345 ± 0.0256028 (W 0.00329561) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.950727 ± 0.00376063 (W 0.00294184) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 1.19622 ± 0.00465472 (W 0.00384748) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.464566 ± 0.00144353 (W 0.0024113) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.945277 ± 0.0045637 (W 0.00236821) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.487832 ± 0.00218652 (W 0.00247397) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.95306 ± 0.0044455 (W 0.00293617) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 2.10276 ± 0.00216706 (W 0.00492715) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 4.16273 ± 0.0140735 (W 0.0106151) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 2.20015 ± 0.00610915 (W 0.00440802) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 4.20272 ± 0.0293201 (W 0.00727213) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.406311 ± 0.00249836 (W 0.00219522) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.943415 ± 0.00395641 (W 0.00356946) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.484373 ± 0.00407647 (W 0.00265436) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.952095 ± 0.00402145 (W 0.00363493) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 2.0678 ± 0.00392716 (W 0.00420893) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 4.15879 ± 0.0229461 (W 0.00638783) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 2.14857 ± 0.00527879 (W 0.00472174) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 4.17091 ± 0.0193708 (W 0.00739421) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.212689 ± 0.000360843 (W 0.00154351) | - | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.270359 ± 0.00152689 (W 0.00194302) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.187939 ± 0.00210847 (W 0.0017085) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.243407 ± 0.00446737 (W 0.00205127) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.703354 ± 0.00216766 (W 0.0025736) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.998993 ± 0.00353764 (W 0.00233311) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.664373 ± 0.00129332 (W 0.00241757) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.934195 ± 0.0137046 (W 0.00242328) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.423901 ± 0.00120392 (W 0.00192976) | - | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.894217 ± 0.00501463 (W 0.00272031) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.407566 ± 0.00257597 (W 0.0019485) | - | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.896972 ± 0.00289832 (W 0.00278837) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.47236 ± 0.00441138 (W 0.00389223) | - | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.60488 ± 0.00668665 (W 0.00700532) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.55359 ± 0.00771787 (W 0.0041437) | - | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.61748 ± 0.0171706 (W 0.00386705) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.369033 ± 0.002521 (W 0.0016592) | - | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.901919 ± 0.00533787 (W 0.00295939) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.400985 ± 0.00152094 (W 0.00270875) | - | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.898829 ± 0.00394437 (W 0.00267659) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.46136 ± 0.00749289 (W 0.00678416) | - | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.6242 ± 0.0173514 (W 0.00608399) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.5186 ± 0.0083109 (W 0.00503513) | - | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.64283 ± 0.0117098 (W 0.00649886) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 4T (CPU 0-3)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.234857 ± 0.000685506 (W 0.0017158) | 0.281396 ± 0.000815997 (W 0.00120864) | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.297891 ± 0.00107056 (W 0.00154786) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.217707 ± 0.00453593 (W 0.00142173) | 0.277605 ± 0.00112858 (W 0.00198828) | 🔵 colmat 1.28× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n257/b1 [api-inclusive]` | 0.275196 ± 0.00140997 (W 0.00152075) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.963315 ± 0.0033655 (W 0.00203168) | 1.39398 ± 0.00356021 (W 0.00386455) | 🔵 colmat 1.45× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 1.13437 ± 0.0168159 (W 0.00233354) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.94005 ± 0.0022727 (W 0.00158915) | 1.3889 ± 0.00256559 (W 0.00389669) | 🔵 colmat 1.48× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 1.08812 ± 0.00405075 (W 0.0025894) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.393564 ± 0.00129912 (W 0.00210534) | 0.553459 ± 0.00131748 (W 0.00194517) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 0.846046 ± 0.0030799 (W 0.0037327) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.410676 ± 0.00101665 (W 0.00161389) | 0.560705 ± 0.00138985 (W 0.0021046) | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 0.854239 ± 0.0030235 (W 0.00340055) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.89255 ± 0.00295821 (W 0.00505981) | 2.19935 ± 0.00434047 (W 0.00462306) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `lu/n511/b1 [api-inclusive]` | 3.74133 ± 0.0142281 (W 0.00964065) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.9741 ± 0.00312662 (W 0.00351717) | 2.21479 ± 0.00412389 (W 0.00411074) | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 3.7771 ± 0.0130059 (W 0.00795564) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.38791 ± 0.00171803 (W 0.00166186) | 0.541822 ± 0.00315122 (W 0.00190127) | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 0.844027 ± 0.00448181 (W 0.00349241) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.406062 ± 0.00163395 (W 0.00182113) | 0.548958 ± 0.00265071 (W 0.00217201) | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 0.851552 ± 0.00473074 (W 0.00310296) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.87413 ± 0.00531452 (W 0.00364791) | 2.16743 ± 0.00840735 (W 0.00421699) | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `solve/n511/b1 [api-inclusive]` | 3.73308 ± 0.0183564 (W 0.00688504) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.92889 ± 0.00507408 (W 0.00315022) | 2.18251 ± 0.00356319 (W 0.00249304) | 🔵 colmat 1.13× faster than `lapack-openblas` |  |
| `solve/n513/b1 [api-inclusive]` | 3.75251 ± 0.014322 (W 0.00671766) | - | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 8T (CPU 0-7)

| case | colmat (ms) | lapack-openblas (ms) | vs `lapack-openblas` | notes |
|---|---|---|---|---|
| `cholesky/n255/b1` | 0.207857 ± 0.000948739 (W 0.00131598) | 0.398267 ± 0.00243254 (W 0.00200462) | 🔵 colmat 1.92× faster than `lapack-openblas` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.272116 ± 0.00292958 (W 0.00135696) | - | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.186614 ± 0.00070067 (W 0.0016594) | 0.253306 ± 0.00124807 (W 0.00199472) | 🔵 colmat 1.36× faster than `lapack-openblas` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.243791 ± 0.00296371 (W 0.00191379) | - | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.702331 ± 0.00285907 (W 0.00217559) | 1.0896 ± 0.00155245 (W 0.00198586) | 🔵 colmat 1.55× faster than `lapack-openblas` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.891446 ± 0.00168951 (W 0.00152661) | - | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.668783 ± 0.00412474 (W 0.00217779) | 0.939701 ± 0.0026216 (W 0.00248583) | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.827808 ± 0.00305178 (W 0.00182689) | - | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.333188 ± 0.00141948 (W 0.00205909) | 0.419451 ± 0.00125505 (W 0.00260102) | 🔵 colmat 1.26× faster than `lapack-openblas` |  |
| `lu/n255/b1 [api-inclusive]` | 0.799046 ± 0.00384777 (W 0.00243781) | - | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.347467 ± 0.000825891 (W 0.00203182) | 0.421131 ± 0.00192293 (W 0.00247031) | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `lu/n257/b1 [api-inclusive]` | 0.798031 ± 0.0048982 (W 0.00292721) | - | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.35439 ± 0.00353748 (W 0.00442444) | 1.31864 ± 0.00456285 (W 0.00546957) | · |  |
| `lu/n511/b1 [api-inclusive]` | 3.21167 ± 0.0108557 (W 0.0048978) | - | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.4013 ± 0.00349055 (W 0.00386375) | 1.33069 ± 0.00124016 (W 0.00317718) | 🔴 colmat 1.05× slower than `lapack-openblas` |  |
| `lu/n513/b1 [api-inclusive]` | 3.21022 ± 0.0157969 (W 0.00720246) | - | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.331543 ± 0.00101088 (W 0.00170481) | 0.419745 ± 0.00150803 (W 0.00293017) | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `solve/n255/b1 [api-inclusive]` | 0.802585 ± 0.00319282 (W 0.0028068) | - | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.343723 ± 0.00125399 (W 0.00217438) | 0.421751 ± 0.00228023 (W 0.00290394) | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `solve/n257/b1 [api-inclusive]` | 0.802258 ± 0.00352991 (W 0.00396919) | - | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.32845 ± 0.00406621 (W 0.00374477) | 1.3152 ± 0.00722479 (W 0.00487333) | · |  |
| `solve/n511/b1 [api-inclusive]` | 3.21182 ± 0.0110278 (W 0.00503279) | - | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.37377 ± 0.00681882 (W 0.00812767) | 1.32918 ± 0.00558787 (W 0.00335588) | · |  |
| `solve/n513/b1 [api-inclusive]` | 3.22549 ± 0.0132645 (W 0.00949558) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/255`, `cholesky/257`, `cholesky/511`, `cholesky/513`, `lu/255`, `lu/257`, `lu/511`, `lu/513`, `solve/255`, `solve/257`, `solve/511`, `solve/513`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
