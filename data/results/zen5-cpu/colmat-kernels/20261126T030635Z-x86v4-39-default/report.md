# `colmat-kernels` on `zen5-cpu`

- colmat (fork, measured) commit: `be19a8001e37e5db55c083c90bd8ca10269d1ab5`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- colmat / faer rows: `colmat` (the provider in the measured revision; see library and upstream above); `faer` (the provider in the measured revision; see library and upstream above)
- harness commit: `a899f37cb62992709cb756c0aa0c93aa5785a922`
- toolkit commit: `82bff6ea4508e156b9e661550c1dc65ad662581c`
- measured binary SHA-256: `3d9dc7a3d1ba83d0083cfd5c95ee38408c492cb165c175d7999c6bde696e1cf3`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-26T03:45:07.796168Z`
- timing policy: v2, median of 5 reps, priming 500 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu colmat-kernels --checkout /home/shinaoka/projects/tensor4all/colmat-rs-39-measured --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor openblas --regimes linalg-small,linalg-mid,linalg-tall,linalg-wide --dtypes f64,c64 --threads 1,4,8 --aa 3 --no-publish --label x86v4-39-default`
- harness environment: `FBENCH_ORDER` unset, `FBENCH_PROFILE=zen5-cpu`
- raw data: `data/results/zen5-cpu/colmat-kernels/20261126T030635Z-x86v4-39-default/`

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
    Caveat: The fork crate is `colmat` (`[lib] name = "colmat"`); the upstream baseline is `faer` (`faer_upstream = { package = "faer" }`). They are distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. measured at 93da6217, colmat-rs main at the 2026-11-17 head: no tag names that revision, and the published crates.io colmat 0.34.0 was cut from e1c83482. The 2026-11-17 zen5-cpu recordings were taken at this revision.
    As measured: `colmat` (the provider in the measured revision; see library and upstream above)
  - `faer` — Upstream `faer` at the tag the fork is based on, in the same binary as the fork row. The baseline that says whether a fork change moved the number.
    Identity: the upstream revision in this run's manifest (`upstream.commit`), pinned by `pins/upstream-faer.rev`
    Caveat: Same crate-name and feature-unification caveat as `colmat`. A change to upstream at a fixed fork revision produces a different page rather than overwriting one.
    As measured: `faer` (the provider in the measured revision; see library and upstream above)
  - `lapack-openblas` — CBLAS `?gemm` plus LAPACK drivers linked against OpenBLAS: vendor calls and their marshalling, vendor-owned threading. The vendor's own thread budget is set by the harness before the row and read back, and the vendor identity and the dispatched kernel are recorded in the manifest.
    Identity: the OpenBLAS build recorded in this run's manifest (`providers[openblas]`)
    Caveat: LAPACK and BLAS own their threading, so this row's thread budget is the vendor's, not the harness pool's. Where a family has no LAPACK route the row is absent and the CSV says the case was skipped for it.
    As measured: linked static; openblas 0.3.32; kernel dispatched to `Cooperlake`; pthread threading; config `OpenBLAS 0.3.32 DYNAMIC_ARCH NO_AFFINITY Cooperlake MAX_THREADS=24`

## Hardware

- CPU: `AMD Ryzen AI 9 HX 470`
- logical CPUs: `24`
- L3: `24 MiB (2 instances)`
- L3 domains: `0-3: 16 MiB shared; 4-11: 8 MiB shared`
- OS / arch: `Linux 7.0.0-44-generic` / `x86_64`
- hostname: `shinaoka-EVO-X1Pro`
- CPU sets: 1T -> `4`, 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 3 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 1.05%, worst 149.60% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 69 of 1188 engine cells exceed 5% and carry `!`; median within-process relative SD 0.71%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

## linalg-small, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004238 ± 0.000405582 (W 4.17876e-05) | 0.004328 ± 0.00105598 (W 1.78885e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003506 ± 5.25389e-05 (W 6.15573e-05) | 0.003527 ± 1.78979e-05 (W 7.17719e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014065 ± 0.000260225 (W 9.7188e-05) | 0.034644 ± 6.60328e-05 (W 3.02357e-05) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012483 ± 5.03322e-05 (W 9.69536e-05) | 0.017062 ± 1.51767e-05 (W 6.28665e-05) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031328 ± 2.58908e-05 (W 8.93308e-05) | 0.164854 ± 0.000439741 (W 0.00189464) | - | 🔵 colmat 5.26× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.028953 ± 0.00013308 (W 0.000162727) | 0.033482 ± 4.04145e-05 (W 0.000137732) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006552 ± 0.000108167 (W 5.02991e-05) | 0.008045 ± 3.78594e-05 (W 5.36768e-05) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008406 ± 4.72582e-05 (W 0.000105737) | 0.007413 ± 0.000116863 (W 0.000118601) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.023693 ± 4.95681e-05 (W 0.00010206) | 0.035075 ± 0.000324048 (W 0.000228965) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.029093 ± 0.000207753 (W 0.00034296) | 0.026829 ± 0.000144627 (W 0.000177548) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.056023 ± 3.01164e-05 (W 0.000213163) | 0.08637 ± 0.00161331 (W 0.000227486) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.07538 ± 0.000140146 (W 0.00251136) | 0.07064 ± 0.00133236 (W 0.00324305) | - | 🔴 colmat 1.07× slower than `faer` (marginal: spread 0.68 of the difference) | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006763 ± 4.41626e-05 (W 3.91957e-05) | 0.008626 ± 6.50641e-05 (W 4.61519e-05) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009518 ± 5.55608e-05 (W 5.82254e-05) | 0.008235 ± 7.23418e-05 (W 4.41848e-05) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.02194 ± 0.000196571 (W 0.000102451) | 0.035666 ± 0.000348841 (W 0.000151364) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.030265 ± 0.000130592 (W 0.000120558) | 0.027291 ± 0.000100351 (W 0.000134034) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.049982 ± 0.000234066 (W 0.00024096) | 0.083464 ± 0.000255799 (W 0.000144776) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0762 ± 0.00202463 (W 0.00248806) | 0.067886 ± 0.000107858 (W 0.00243275) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004108 ± 0.00021443 (W 2.96648e-05) | 0.004278 ± 2.70247e-05 (W 8.00206e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003456 ± 0.000170978 (W 4.53134e-05) | 0.003506 ± 5.19615e-06 (W 4.71084e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014276 ± 0.000438999 (W 7.07976e-05) | 0.034604 ± 0.000289252 (W 0.00169165) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012744 ± 0.000578851 (W 7.83728e-05) | 0.017182 ± 7.55667e-05 (W 7.05854e-05) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.03213 ± 0.000445142 (W 0.000168029) | 0.069047 ± 0.00564534 (W 0.0518756) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.030587 ± 0.000483116 (W 0.000152403) | 0.039292 ± 0.000409343 (W 0.000337508) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006663 ± 2.09841e-05 (W 9.98499e-05) | 0.007844 ± 7.26659e-05 (W 5.16933e-05) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008626 ± 6.71342e-05 (W 0.000143023) | 0.007384 ± 8.94446e-05 (W 9.81137e-05) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.029214 ± 0.00117325 (W 0.00128844) | 0.037098 ± 0.000873531 (W 0.000547769) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.029635 ± 0.000333492 (W 0.000201142) | 0.030556 ± 0.000135795 (W 0.00120765) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.061774 ± 0.000666056 (W 0.00182148) | 0.090457 ± 0.00121824 (W 0.00127136) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.068948 ± 0.000365774 (W 0.00226603) | 0.075991 ± 0.000810482 (W 0.00329811) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007504 ± 9.88956e-05 (W 3.68673e-05) | 0.008576 ± 9.71253e-05 (W 4.41158e-05) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010128 ± 1.12694e-05 (W 7.67216e-05) | 0.008355 ± 0.000158745 (W 0.000126176) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.026158 ± 0.00044949 (W 0.000544515) | 0.038531 ± 0.000181302 (W 0.000656794) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.03256 ± 0.000135795 (W 0.000122066) | 0.030887 ± 0.000234948 (W 0.000875359) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.052998 ± 0.00059792 (W 0.000377784) | 0.087462 ± 0.00017706 (W 0.001837) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.074337 ± 0.000463972 (W 0.00239704) | 0.072685 ± 0.0003759 (W 0.00262365) | - | · | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003727 ± 0.000271539 (W 3.41365e-05) | 0.004338 ± 5.7735e-06 (W 2.16795e-05) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003527 ± 0.000231683 (W 9.80673e-05) | 0.003646 ± 3.49333e-05 (W 6.83396e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014166 ± 5.68624e-05 (W 7.69103e-05) | 0.034484 ± 7.26659e-05 (W 7.29383e-05) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012763 ± 7.64875e-05 (W 7.71246e-05) | 0.017111 ± 8.544e-05 (W 5.43065e-05) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031869 ± 0.000894088 (W 0.000117521) | 0.057035 ± 0.000179536 (W 0.000211216) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.030847 ± 0.000144684 (W 0.000243873) | 0.040285 ± 0.000126808 (W 0.000380536) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006723 ± 5.04017e-05 (W 7.0214e-05) | 0.007904 ± 0.000106444 (W 0.000102363) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008476 ± 4.39356e-05 (W 0.000117687) | 0.007443 ± 7.22519e-05 (W 8.23541e-05) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.030977 ± 0.00016544 (W 0.00143163) | 0.037339 ± 0.000202609 (W 0.000246283) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.029805 ± 0.00014 (W 0.000261307) | 0.031739 ± 0.000380142 (W 0.000418235) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.065832 ± 0.000776643 (W 0.00170792) | 0.092621 ± 0.000366402 (W 0.00156704) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.071361 ± 0.00163012 (W 0.00220031) | 0.078435 ± 0.000341049 (W 0.000906013) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007454 ± 0.000113315 (W 7.59098e-05) | 0.008486 ± 6.76782e-05 (W 5.06952e-05) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010139 ± 2.08167e-05 (W 9.44468e-05) | 0.008175 ± 5.50757e-05 (W 3.38556e-05) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.02711 ± 0.000369938 (W 0.000584822) | 0.038361 ± 0.000425966 (W 0.00172862) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.03243 ± 0.000111355 (W 0.000132401) | 0.031468 ± 0.000167033 (W 0.000926043) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.055463 ± 0.000252589 (W 0.00170506) | 0.087622 ± 0.000101129 (W 0.000513259) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.07549 ± 0.0017768 (W 0.00252705) | 0.07566 ± 0.000623026 (W 0.00197469) | - | · | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002485 ± 0.000246226 (W 1.92354e-05) | 0.002825 ± 5.7735e-06 (W 2.79875e-05) | 0.002926 ± 5.23482e-05 (W 5.0033e-05) | 🔵 colmat 1.14× faster than `faer` (marginal: spread 0.74 of the difference) | 🔵 colmat 1.18× faster than `lapack-openblas` (marginal: spread 0.68 of the difference) | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003407 ± 0.00013308 (W 5.96766e-05) | 0.003416 ± 2.08167e-05 (W 5.65977e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010549 ± 0.000359667 (W 5.94912e-05) | 0.015278 ± 5.50757e-05 (W 2.40832e-05) | 0.013154 ± 2.51661e-05 (W 0.00010107) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012283 ± 3e-05 (W 0.000107635) | 0.016821 ± 0.000105879 (W 5.93902e-05) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.023553 ± 0.000787631 (W 6.94982e-05) | 0.029475 ± 0.000120652 (W 3.9714e-05) | 0.031679 ± 4.39356e-05 (W 9.30177e-05) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.0268 ± 0.000938688 (W 0.000148069) | 0.032751 ± 9.11062e-05 (W 0.000129932) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n32/b1` | 0.00532 ± 9.07377e-05 (W 9.04434e-05) | 0.005921 ± 2.08167e-05 (W 4.23934e-05) | 0.007393 ± 8.56174e-05 (W 0.000126516) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006632 ± 2.88675e-05 (W 0.000111598) | 0.007244 ± 8.17333e-05 (W 0.000193872) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.021019 ± 0.000135537 (W 9.39537e-05) | 0.022512 ± 0.000108076 (W 0.000210234) | 0.030076 ± 0.00010504 (W 0.000255218) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024195 ± 0.000125831 (W 0.000180633) | 0.026288 ± 0.000140096 (W 0.00130648) | - | 🔵 colmat 1.09× faster than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.050023 ± 0.000759955 (W 0.000262832) | 0.053428 ± 0.000185527 (W 0.000140428) | 0.07027 ± 0.000495519 (W 0.00169168) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.061604 ± 0.00130687 (W 0.002539) | 0.066513 ± 0.000337817 (W 0.00301543) | - | 🔵 colmat 1.08× faster than `faer` (marginal: spread 0.61 of the difference) | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.005972 ± 2.85832e-05 (W 3.79249e-05) | 0.006361 ± 1.12694e-05 (W 6.6334e-05) | 0.007534 ± 2.64575e-05 (W 6.02096e-05) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007624 ± 1.52753e-05 (W 6.16782e-05) | 0.007924 ± 0.000190596 (W 0.000133642) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021209 ± 0.0001805 (W 0.00010183) | 0.022702 ± 0.000346074 (W 0.000110889) | 0.029154 ± 0.000199149 (W 9.42937e-05) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.024396 ± 5.27289e-05 (W 8.82043e-05) | 0.026158 ± 4.93288e-05 (W 0.000120097) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.050203 ± 1.48436e-05 (W 0.000201855) | 0.053388 ± 0.000165681 (W 0.00180396) | 0.068908 ± 0.000288964 (W 0.00238264) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.066082 ± 0.00201992 (W 0.00245979) | 0.066473 ± 0.00106614 (W 0.00244362) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002805 ± 0.000174356 (W 2.48998e-05) | 0.002785 ± 1.0504e-05 (W 1.73205e-05) | 0.002905 ± 1.1547e-05 (W 3.56399e-05) | · | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003396 ± 0.000136346 (W 4.28742e-05) | 0.003406 ± 7.20208e-05 (W 4.96115e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010239 ± 0.000536194 (W 2.61954e-05) | 0.015238 ± 8.0829e-05 (W 4.15933e-05) | 0.013055 ± 0.000127005 (W 9.42338e-05) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012593 ± 5.50757e-05 (W 0.000148163) | 0.016791 ± 0.000217025 (W 7.66159e-05) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.025767 ± 0.000858364 (W 0.000119892) | 0.033281 ± 0.000204858 (W 0.000331284) | 0.031859 ± 0.000117304 (W 0.000137405) | 🔵 colmat 1.29× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.030066 ± 9.43416e-05 (W 0.000157535) | 0.03828 ± 0.0003695 (W 0.000543206) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00537 ± 4.09512e-05 (W 8.47715e-05) | 0.005891 ± 3.64737e-05 (W 4.69596e-05) | 0.007464 ± 6.80686e-05 (W 7.48011e-05) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006943 ± 6.245e-05 (W 8.38469e-05) | 0.007274 ± 0.00024379 (W 9.6126e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.021079 ± 0.000107503 (W 9.86367e-05) | 0.025718 ± 0.000306187 (W 0.000297398) | 0.030126 ± 0.000175784 (W 0.000237635) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024755 ± 0.000161339 (W 0.000221294) | 0.029424 ± 0.000297725 (W 0.000997622) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.040665 ± 0.000445249 (W 0.000144455) | 0.05957 ± 0.000423935 (W 0.00190432) | 0.069788 ± 0.000152451 (W 0.000259569) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.057417 ± 0.00106908 (W 0.00226068) | 0.072143 ± 0.00162701 (W 0.00332957) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006673 ± 6.60328e-05 (W 6.63197e-05) | 0.006472 ± 3.4641e-05 (W 5.63028e-05) | 0.007674 ± 3.60555e-05 (W 0.000172279) | · | 🔵 colmat 1.15× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008245 ± 2.08167e-05 (W 0.000115569) | 0.007875 ± 3.46458e-05 (W 6.49169e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.023353 ± 5.03322e-05 (W 8.9091e-05) | 0.025877 ± 0.000184464 (W 0.000446931) | 0.029614 ± 8.05419e-05 (W 7.19722e-05) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.026739 ± 9.23598e-05 (W 0.000227805) | 0.029645 ± 0.000200592 (W 0.000890746) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.044612 ± 0.00014 (W 0.000124625) | 0.058288 ± 7.6922e-05 (W 0.000444745) | 0.069799 ± 0.00040005 (W 0.000146015) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.061593 ± 0.00187879 (W 0.0024394) | 0.071311 ± 0.000248396 (W 0.00263393) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002525 ± 6.65833e-05 (W 2.34521e-05) | 0.002825 ± 5.7735e-07 (W 2.43167e-05) | 0.002935 ± 1.21244e-05 (W 4.12214e-05) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003477 ± 0.000130577 (W 5.57961e-05) | 0.003477 ± 2.52389e-05 (W 3.33497e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0107 ± 2.51661e-05 (W 6.22093e-05) | 0.015278 ± 3.60555e-05 (W 0.000101554) | 0.013194 ± 0.000169343 (W 0.000113138) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012594 ± 6.08276e-06 (W 0.000132062) | 0.016861 ± 4.50925e-05 (W 4.08656e-05) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.025146 ± 0.000790606 (W 0.000148153) | 0.034103 ± 0.00351665 (W 0.000217801) | 0.031779 ± 6.55744e-05 (W 0.000161771) | 🔵 colmat 1.36× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.030065 ± 0.000195244 (W 0.000204821) | 0.039172 ± 5.63057e-05 (W 0.000337117) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00545 ± 7.54983e-05 (W 8.20366e-05) | 0.005931 ± 9.71253e-05 (W 0.0018528) | 0.007433 ± 0.000150489 (W 0.000182199) | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | 🔵 colmat 1.36× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006942 ± 4.72582e-05 (W 0.000131812) | 0.007323 ± 6.61085e-05 (W 0.000115582) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02158 ± 4.93288e-05 (W 0.000134615) | 0.025788 ± 0.000244365 (W 0.000216802) | 0.029936 ± 0.000211093 (W 0.000176806) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024695 ± 9.07028e-05 (W 0.000293163) | 0.030427 ± 0.000228014 (W 0.000323319) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.040365 ± 0.000215423 (W 0.000387155) | 0.0597 ± 0.000321684 (W 0.00103514) | 0.069729 ± 9.2376e-05 (W 0.000141139) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.057617 ± 0.00111244 (W 0.00225503) | 0.07576 ± 0.000810077 (W 0.00184504) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006672 ± 5.18684e-05 (W 2.63306e-05) | 0.006382 ± 4.93288e-05 (W 7.49667e-05) | 0.007634 ± 6.05007e-05 (W 7.72658e-05) | · | 🔵 colmat 1.14× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008255 ± 7.0406e-05 (W 8.92765e-05) | 0.007874 ± 6.07481e-05 (W 6.60326e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.023473 ± 0.000233017 (W 0.000114014) | 0.025838 ± 0.000201098 (W 0.000700804) | 0.029264 ± 0.00013 (W 6.69328e-05) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.026579 ± 0.000145775 (W 0.000164288) | 0.030326 ± 0.000497609 (W 0.000435505) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.043811 ± 0.000130128 (W 0.000349438) | 0.057777 ± 0.000688665 (W 0.000398746) | 0.07037 ± 0.0003418 (W 0.000219375) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.063057 ± 0.00204449 (W 0.000359114) | 0.073927 ± 0.00054835 (W 0.00239603) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-small, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005831 ± 2.36714e-05 (W 5.80793e-05) | 0.009066 ± 2.66521e-05 (W 1.53721e-05) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00506 ± 3.01164e-05 (W 0.000114487) | 0.00525 ± 2.52389e-05 (W 9.71967e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.032751 ± 0.000186039 (W 0.000158664) | 0.069899 ± 0.00025 (W 0.000187557) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.030787 ± 4.32474e-05 (W 0.000171346) | 0.030386 ± 0.000142818 (W 0.000158039) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.089235 ± 0.000248308 (W 0.000315039) | 0.442757 ± 0.000404937 (W 0.00191699) | - | 🔵 colmat 4.96× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.085808 ± 9.55528e-05 (W 0.000283561) | 0.083784 ± 0.000433983 (W 0.000307184) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012614 ± 6.86319e-05 (W 9.20869e-05) | 0.010519 ± 5.22526e-05 (W 5.27371e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015008 ± 8.59089e-05 (W 8.90354e-05) | 0.011361 ± 0.000171553 (W 8.50294e-05) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.054301 ± 0.000220273 (W 0.000176814) | 0.05416 ± 0.000344486 (W 0.000256273) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.063057 ± 0.000255807 (W 0.000432901) | 0.052898 ± 0.000185581 (W 0.00148289) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.141812 ± 0.000368045 (W 0.0028418) | 0.142813 ± 0.000240666 (W 0.00168074) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.207994 ± 0.00407126 (W 0.00305736) | 0.187786 ± 0.000750618 (W 0.00224153) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012904 ± 4.16333e-05 (W 8.1989e-05) | 0.011271 ± 4.50925e-05 (W 8.03443e-05) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016691 ± 0.000214337 (W 0.000105883) | 0.012683 ± 0.0002474 (W 0.000138558) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.052607 ± 6.55515e-05 (W 0.00143478) | 0.055733 ± 0.000214189 (W 0.0024713) | - | 🔵 colmat 1.06× faster than `faer` (marginal: spread 0.79 of the difference) | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.066202 ± 0.000839219 (W 0.000304076) | 0.054571 ± 0.00026411 (W 0.000842218) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.137994 ± 0.000145007 (W 0.00267654) | 0.146129 ± 0.000460459 (W 0.00233779) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.215638 ± 0.00244532 (W 0.00322793) | 0.187686 ± 0.00249796 (W 0.00290844) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005871 ± 1e-05 (W 7.62706e-05) | 0.009076 ± 2.51064e-05 (W 2.21427e-05) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.004989 ± 2.5697e-05 (W 5.35462e-05) | 0.00526 ± 2.88675e-05 (W 4.65854e-05) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.028623 ± 0.000165025 (W 0.000153975) | 0.069768 ± 0.000275072 (W 0.00184385) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.028272 ± 0.000315647 (W 0.000170582) | 0.030516 ± 0.000135277 (W 8.55412e-05) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.069418 ± 0.000543419 (W 0.000217245) | 0.164513 ± 0.141907 (W 0.146462) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.070059 ± 0.000370671 (W 0.000125379) | 0.088273 ± 0.000360051 (W 0.00170439) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012733 ± 0.000210927 (W 9.61988e-05) | 0.010349 ± 1.78979e-05 (W 5.06192e-05) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015498 ± 5.70117e-05 (W 0.000106584) | 0.011281 ± 6.65833e-05 (W 3.96232e-05) | - | 🔴 colmat 1.37× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.050203 ± 9.66971e-05 (W 0.00050241) | 0.053589 ± 0.000272666 (W 0.000140837) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.055843 ± 0.000384293 (W 0.000244531) | 0.052627 ± 0.00017673 (W 0.000291145) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.120292 ± 0.00179834 (W 0.00104758) | 0.139207 ± 0.00077769 (W 0.00138601) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.170264 ± 0.00138243 (W 0.00124059) | 0.139298 ± 0.00237248 (W 0.0020404) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013585 ± 1.1547e-05 (W 5.01996e-05) | 0.011421 ± 6.46555e-05 (W 3.71914e-05) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.017261 ± 0.000140905 (W 8.61992e-05) | 0.012714 ± 0.000151461 (W 7.21062e-05) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.047978 ± 0.000111072 (W 0.000475811) | 0.055562 ± 0.000361738 (W 0.000268291) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.05947 ± 0.000450141 (W 0.0004669) | 0.055733 ± 0.000500802 (W 0.000175591) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.103201 ± 0.000320468 (W 0.00074072) | 0.142163 ± 0.00160588 (W 0.00147116) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.170524 ± 0.000953386 (W 0.00250826) | 0.142042 ± 0.000364464 (W 0.00177846) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005951 ± 0.000100732 (W 4.46912e-05) | 0.009057 ± 1.73205e-05 (W 2.70037e-05) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005049 ± 2.66521e-05 (W 7.49146e-05) | 0.00532 ± 3.05014e-05 (W 9.09395e-05) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.028913 ± 0.000155027 (W 0.000252274) | 0.070209 ± 0.000161339 (W 0.000246859) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.028182 ± 0.000132035 (W 0.000181728) | 0.030606 ± 6.26126e-05 (W 0.00044964) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.070521 ± 0.000852081 (W 0.00033565) | 0.130931 ± 0.000669397 (W 0.00142123) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.07027 ± 0.0002405 (W 0.000178763) | 0.088333 ± 0.000402716 (W 0.00163481) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012733 ± 5.53203e-05 (W 7.70214e-05) | 0.010259 ± 8.66025e-05 (W 6.993e-05) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015258 ± 7.57188e-05 (W 0.000114983) | 0.011411 ± 4.3589e-05 (W 0.000135019) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.054891 ± 0.000160001 (W 0.000718184) | 0.053168 ± 0.000386637 (W 0.000248928) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.055423 ± 0.000346497 (W 0.000400091) | 0.052697 ± 0.000476009 (W 0.000183933) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.126675 ± 0.000428937 (W 0.000968572) | 0.140139 ± 0.00105704 (W 0.00174859) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.16791 ± 0.00231866 (W 0.00147225) | 0.139117 ± 0.00293151 (W 0.00257343) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013585 ± 9.45163e-05 (W 6.87983e-05) | 0.011371 ± 8.02081e-05 (W 0.000308183) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.017332 ± 6.55744e-05 (W 0.000103698) | 0.012723 ± 7.25282e-05 (W 8.47838e-05) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.04868 ± 0.000248041 (W 0.000403631) | 0.055423 ± 0.000229829 (W 0.000142311) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.058107 ± 0.000118955 (W 0.000355023) | 0.055823 ± 0.000298581 (W 0.000211137) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.098842 ± 0.000816077 (W 0.000527038) | 0.141511 ± 0.00078159 (W 0.0019072) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.169453 ± 0.0028603 (W 0.0022906) | 0.14123 ± 0.000862128 (W 0.00146081) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004328 ± 0.000104484 (W 5.12855e-05) | 0.004609 ± 2.95014e-05 (W 1.2054e-05) | 0.004899 ± 5.50757e-05 (W 5.17687e-05) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00506 ± 7.78267e-05 (W 5.26308e-05) | 0.00523 ± 1.52753e-05 (W 4.95611e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.027831 ± 0.000207632 (W 8.29458e-05) | 0.028323 ± 0.00012356 (W 7.92351e-05) | 0.028132 ± 7.55667e-05 (W 0.000136672) | · | · |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.030476 ± 2.88675e-05 (W 0.000136625) | 0.030015 ± 8.35484e-05 (W 0.000101727) | - | · | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.078835 ± 0.00012554 (W 0.000261338) | 0.077523 ± 0.000284231 (W 0.00190454) | 0.078815 ± 7.97517e-05 (W 0.000133207) | · | · |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.084967 ± 0.000102632 (W 0.00180965) | 0.083153 ± 0.000157386 (W 0.000103685) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011441 ± 0.00012554 (W 5.68331e-05) | 0.009648 ± 9.66282e-05 (W 8.44701e-05) | 0.01129 ± 1.00167e-05 (W 6.90652e-05) | 🔴 colmat 1.19× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013314 ± 4.07063e-05 (W 0.000156356) | 0.011501 ± 0.000147422 (W 0.000172378) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.050312 ± 9.64434e-05 (W 0.000154691) | 0.046316 ± 0.000256028 (W 0.000161797) | 0.055071 ± 8.05419e-05 (W 0.000134176) | 🔴 colmat 1.09× slower than `faer` | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.055492 ± 0.000208727 (W 0.000255118) | 0.051265 ± 0.000360759 (W 0.000274755) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.134318 ± 0.00031 (W 0.0016823) | 0.127275 ± 0.00126462 (W 0.000254977) | 0.153894 ± 0.00031812 (W 0.00200672) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.15× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.190592 ± 0.00314364 (W 0.00405347) | 0.18389 ± 0.00147006 (W 0.00197631) | - | · | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012343 ± 0.000133593 (W 7.91221e-05) | 0.01056 ± 2.09841e-05 (W 4.23462e-05) | 0.012583 ± 2.85832e-05 (W 6.28267e-05) | 🔴 colmat 1.17× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014266 ± 7.95634e-05 (W 5.32851e-05) | 0.012343 ± 0.000205322 (W 9.02458e-05) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.052747 ± 0.000754143 (W 0.00167014) | 0.04879 ± 0.000256373 (W 0.0024927) | 0.059069 ± 0.00010583 (W 0.000122724) | 🔴 colmat 1.08× slower than `faer` (marginal: spread 0.63 of the difference) | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.058378 ± 0.000232337 (W 0.000466809) | 0.053518 ± 0.000101392 (W 0.000232558) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.139888 ± 0.000744679 (W 0.00172058) | 0.130461 ± 0.000359875 (W 0.00165029) | 0.161879 ± 0.000266766 (W 0.00245358) | 🔴 colmat 1.07× slower than `faer` | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.193297 ± 0.00350696 (W 0.00273326) | 0.18445 ± 0.000468968 (W 0.00321186) | - | · | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004177 ± 6.26126e-05 (W 4.17876e-05) | 0.004469 ± 6.03352e-05 (W 2.33731e-05) | 0.004959 ± 4.04145e-05 (W 6.16539e-05) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005009 ± 6e-05 (W 0.00011798) | 0.00529 ± 3.78594e-05 (W 8.90236e-05) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.024245 ± 0.000168622 (W 0.000182722) | 0.027441 ± 0.000564306 (W 4.65543e-05) | 0.031638 ± 0.000761704 (W 0.000662751) | 🔵 colmat 1.13× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.027791 ± 0.00027521 (W 0.000232756) | 0.030005 ± 3.82143e-05 (W 8.34865e-05) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.059811 ± 0.000136077 (W 0.000320765) | 0.080549 ± 0.000187705 (W 0.0015174) | 0.073826 ± 0.00104019 (W 0.00146928) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.069448 ± 0.000312693 (W 0.000281728) | 0.087492 ± 0.000337625 (W 0.000534832) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011521 ± 0.000110726 (W 6.87917e-05) | 0.009628 ± 3.82143e-05 (W 0.000113628) | 0.011261 ± 9.49052e-05 (W 8.81209e-05) | 🔴 colmat 1.20× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013775 ± 0.000123843 (W 0.000160759) | 0.011472 ± 8.6122e-05 (W 0.000110631) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.042478 ± 0.000250779 (W 0.000415018) | 0.046136 ± 0.000147751 (W 0.000115976) | 0.054851 ± 0.000110151 (W 0.00152594) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.049261 ± 0.000380978 (W 0.000443261) | 0.051465 ± 5.68624e-05 (W 0.000266489) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.093182 ± 0.000705994 (W 0.000521889) | 0.122847 ± 0.000814993 (W 0.00119937) | 0.154595 ± 0.000613303 (W 0.00224527) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.66× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.152302 ± 0.00122393 (W 0.001132) | 0.133396 ± 0.00209052 (W 0.00226014) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013214 ± 0.000135967 (W 0.000149249) | 0.010529 ± 3.56417e-05 (W 1.98796e-05) | 0.012623 ± 4.93288e-05 (W 8.82043e-05) | 🔴 colmat 1.26× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014858 ± 1.47309e-05 (W 0.000120786) | 0.012553 ± 1e-05 (W 7.91833e-05) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.045935 ± 0.000326145 (W 0.00038396) | 0.04877 ± 0.00012741 (W 9.27362e-05) | 0.059761 ± 0.000263699 (W 0.00197509) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.052086 ± 0.000599734 (W 0.000294461) | 0.05406 ± 0.000145204 (W 0.000181302) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.094724 ± 0.00106397 (W 0.000864475) | 0.128989 ± 0.000741018 (W 0.00218953) | 0.161028 ± 9.88956e-05 (W 0.00209653) | 🔵 colmat 1.36× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.152702 ± 0.000417598 (W 0.00268336) | 0.139778 ± 0.00104152 (W 0.00263542) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004218 ± 5.32948e-05 (W 4.7172e-05) | 0.004488 ± 2.3094e-05 (W 3.14356e-05) | 0.004949 ± 2e-05 (W 4.36028e-05) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00506 ± 0.000151218 (W 0.000165033) | 0.00528 ± 5.7735e-06 (W 5.21536e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.023835 ± 9.46942e-05 (W 0.000116975) | 0.027561 ± 0.000170969 (W 8.20134e-05) | 0.033823 ± 0.00043353 (W 0.000636263) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 1.42× faster than `lapack-openblas` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.027812 ± 0.000400218 (W 0.0001032) | 0.030096 ± 7.50555e-05 (W 0.000122198) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.061674 ± 0.00024077 (W 0.000237203) | 0.080609 ± 0.000771187 (W 0.000343456) | 0.08151 ± 0.000890373 (W 0.00168239) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.069638 ± 0.000581526 (W 0.000334462) | 0.087451 ± 0.000571341 (W 0.00033586) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011672 ± 6.12127e-05 (W 8.74317e-05) | 0.009518 ± 7.82326e-05 (W 5.81317e-05) | 0.011211 ± 0.000138904 (W 0.000101094) | 🔴 colmat 1.23× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013525 ± 7.37111e-05 (W 0.000121956) | 0.011641 ± 2.3094e-05 (W 7.87293e-05) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.043089 ± 0.00037055 (W 0.000373008) | 0.045444 ± 0.000248538 (W 0.0015232) | 0.054981 ± 5.03322e-05 (W 0.000209166) | 🔵 colmat 1.05× faster than `faer` (marginal: spread 0.65 of the difference) | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.048259 ± 0.00031281 (W 0.000360079) | 0.051094 ± 0.0001253 (W 0.000443144) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.087952 ± 0.000291033 (W 0.0012894) | 0.12448 ± 0.000524615 (W 0.00181805) | 0.153894 ± 6.245e-05 (W 0.00202589) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.148024 ± 0.00182181 (W 0.00197841) | 0.136161 ± 0.00112923 (W 0.00146229) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012904 ± 1.52753e-05 (W 7.87477e-05) | 0.010469 ± 6.97639e-05 (W 4.8211e-05) | 0.012563 ± 0.000103923 (W 0.000136675) | 🔴 colmat 1.23× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.015018 ± 5.0797e-05 (W 6.39164e-05) | 0.012614 ± 0.000131226 (W 0.000124785) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.045063 ± 0.000499281 (W 0.000304913) | 0.047858 ± 0.000182008 (W 0.000113711) | 0.059309 ± 0.000144938 (W 8.56814e-05) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.050193 ± 0.000244684 (W 0.000361115) | 0.05419 ± 0.000231128 (W 8.3602e-05) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.085177 ± 0.000132035 (W 0.000518028) | 0.127205 ± 0.000208951 (W 0.0016275) | 0.161248 ± 0.000926206 (W 0.00218559) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 1.89× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.149877 ± 0.00230807 (W 0.00218944) | 0.14045 ± 0.00105396 (W 0.00234979) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `lu/32`, `lu/64`, `lu/96`, `solve/32`, `solve/64`, `solve/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.058648 ± 0.000681405 (W 0.00223027) | 0.296847 ± 0.000968306 (W 0.00226163) | - | 🔵 colmat 5.06× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.061855 ± 0.000112518 (W 0.000613028) | 0.071031 ± 0.000655158 (W 0.000361754) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.370705 ± 0.00388969 (W 0.00365317) | 0.836195 ± 0.00230907 (W 0.00353991) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.342233 ± 0.00445925 (W 0.00273107) | 0.390202 ± 0.00190504 (W 0.00285416) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.91203 ± 0.00546521 (W 0.0061919) | 3.67789 ± 0.012887 (W 0.0125552) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.17735 ± 0.0150398 (W 0.00314883) | 2.52384 ± 0.00578309 (W 0.00373002) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.105965 ± 0.000714531 (W 0.00160238) | 0.263837 ± 0.000649465 (W 0.00342186) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.12439 ± 0.00136304 (W 0.00241256) | 0.174943 ± 0.000915296 (W 0.00366757) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.576716 ± 0.00403761 (W 0.00223653) | 2.57502 ± 0.0163617 (W 0.0116378) | - | 🔵 colmat 4.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.941326 ± 0.00580878 (W 0.00278835) | 1.07598 ± 0.00297031 (W 0.00468853) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.91432 ± 0.0101375 (W 0.00905921) | 20.3932 ± 0.0736604 (W 0.0300666) | - | 🔵 colmat 5.21× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.63889 ± 0.0276708 (W 0.0117398) | 7.36087 ± 0.0792951 (W 0.0171785) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.095887 ± 8.54225e-05 (W 0.000113364) | 0.191884 ± 0.00397749 (W 0.00544594) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.123969 ± 0.000678307 (W 0.00161277) | 0.136793 ± 0.00100404 (W 0.00292681) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.538706 ± 0.00348299 (W 0.00430619) | 1.91007 ± 0.0107893 (W 0.00966952) | - | 🔵 colmat 3.55× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.692687 ± 0.00277353 (W 0.0042973) | 0.803785 ± 0.00138443 (W 0.00242848) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.7795 ± 0.0258125 (W 0.0103741) | 13.7605 ± 0.0483616 (W 0.014357) | - | 🔵 colmat 3.64× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.27854 ± 0.04756 (W 0.00507973) | 5.80811 ± 0.0237398 (W 0.00928604) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.047889 ± 0.000936041 (W 0.000168274) | 0.206872 ± 0.0533363 (W 0.0541462) | - | 🔵 colmat 4.32× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.055502 ± 0.000616793 (W 0.000332703) | 0.081651 ± 0.000838667 (W 0.00198984) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.180724 ± 0.00148811 (W 0.00160003) | 0.67668 ± 0.0569874 (W 0.0570261) | - | 🔵 colmat 3.74× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.218844 ± 0.00146279 (W 0.00218843) | 0.378259 ± 0.0103157 (W 0.0145913) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.971793 ± 0.00762059 (W 0.00398547) | 2.26941 ± 0.0778422 (W 0.130069) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.96319 ± 0.00441869 (W 0.00202619) | 1.58213 ± 0.0314925 (W 0.0366115) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.112638 ± 0.00254162 (W 0.00145124) | 0.279535 ± 0.00163428 (W 0.0044121) | - | 🔵 colmat 2.48× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.105325 ± 0.000848341 (W 0.000442776) | 0.153062 ± 0.00146234 (W 0.00436988) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.348103 ± 0.00119526 (W 0.00206438) | 2.27215 ± 0.0672665 (W 0.0570714) | - | 🔵 colmat 6.53× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.656404 ± 0.00470373 (W 0.00368246) | 0.77939 ± 0.00750556 (W 0.00900775) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.71589 ± 0.0100557 (W 0.00472991) | 13.3812 ± 0.0568501 (W 0.592147) | - | 🔵 colmat 7.80× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.296 ± 0.0335931 (W 0.0132636) | 4.10742 ± 0.0280645 (W 0.0816014) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.088884 ± 0.000614278 (W 0.001161) | 0.225427 ± 0.000744411 (W 0.00367625) | - | 🔵 colmat 2.54× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.108249 ± 0.00104407 (W 0.000403849) | 0.151003 ± 0.000112509 (W 0.00315523) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.345966 ± 0.00160284 (W 0.0039312) | 1.72809 ± 0.00881044 (W 0.0258303) | - | 🔵 colmat 4.99× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.662404 ± 0.00544261 (W 0.00440543) | 0.716332 ± 0.00341955 (W 0.012871) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.58018 ± 0.00815146 (W 0.00289833) | 9.9724 ± 0.181947 (W 0.139678) | - | 🔵 colmat 6.31× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.25445 ± 0.0207843 (W 0.0102936) | 3.70907 ± 0.0407172 (W 0.10064) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.047498 ± 0.000398711 (W 0.000175776) | 0.111205 ± 0.000722256 (W 0.000759763) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.055543 ± 0.000983059 (W 0.000399169) | 0.074892 ± 0.000554452 (W 0.000471686) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.156796 ± 0.000877101 (W 0.00282404) | 0.629393 ± 0.0640712 (W 0.0434577) | - | 🔵 colmat 4.01× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.197185 ± 0.00184222 (W 0.00120435) | 0.355478 ± 0.0263773 (W 0.0221633) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.798314 ± 0.00302801 (W 0.00473193) | 2.01346 ± 0.00830217 (W 0.0416638) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.806028 ± 0.0144525 (W 0.00647621) | 1.49058 ± 0.017874 (W 0.0159909) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.121284 ± 0.00100498 (W 0.0031022) | 0.272212 ± 0.00198558 (W 0.00281562) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.143154 ± 0.0024241 (W 0.000501602) | 0.148161 ± 0.000839733 (W 0.0018089) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.288103 ± 0.0153672 (W 0.00314624) | 2.26725 ± 0.0225193 (W 0.0265711) | - | 🔵 colmat 7.87× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.625746 ± 0.000416898 (W 0.00415842) | 0.799796 ± 0.0109487 (W 0.00939709) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.23976 ± 0.0674066 (W 0.0083237) | 11.8593 ± 0.0533045 (W 0.0919028) | - | 🔵 colmat 9.57× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 2.92057 ± 0.0350682 (W 0.00753222) | 3.92837 ± 0.0633329 (W 0.0598008) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.081832 ± 0.000533701 (W 0.000463904) | 0.217401 ± 0.00046223 (W 0.00246204) | - | 🔵 colmat 2.66× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.105705 ± 0.000277217 (W 0.000332127) | 0.144547 ± 0.000544089 (W 0.0021798) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.2761 ± 0.00149056 (W 0.00341264) | 1.68175 ± 0.00428826 (W 0.0265096) | - | 🔵 colmat 6.09× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.637408 ± 0.00619544 (W 0.00292587) | 0.736491 ± 0.000199872 (W 0.022193) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.10975 ± 0.00772302 (W 0.00338981) | 8.88787 ± 0.0133717 (W 0.0590657) | - | 🔵 colmat 8.01× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.90786 ± 0.0749994 (W 0.0175279) | 3.55891 ± 0.0634193 (W 0.0402748) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.050202 ± 0.000205508 (W 0.000131739) | 0.058538 ± 0.00327156 (W 0.00182303) | 0.062666 ± 0.000341938 (W 0.000278069) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.055653 ± 4.62637e-05 (W 0.000147205) | 0.064138 ± 0.00303718 (W 0.000184036) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.285437 ± 0.00159 (W 0.00297442) | 0.297228 ± 0.00109865 (W 0.00253427) | 0.366236 ± 0.00180973 (W 0.00278214) | · | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.310494 ± 0.000690645 (W 0.00261954) | 0.32506 ± 0.000743714 (W 0.0028486) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.97985 ± 0.00677686 (W 0.00462937) | 1.93643 ± 0.00317273 (W 0.00341915) | 2.54788 ± 0.00882954 (W 0.00649555) | · | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.0223 ± 0.00588176 (W 0.00703129) | 2.08 ± 0.0212732 (W 0.00359644) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.097871 ± 0.000130615 (W 0.000530387) | 0.105474 ± 0.000771208 (W 0.00136003) | 0.14085 ± 0.00127255 (W 0.00204555) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.105304 ± 0.000704843 (W 0.000527228) | 0.153785 ± 0.000989277 (W 0.00165892) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.574282 ± 0.00232519 (W 0.00323245) | 0.690583 ± 0.00358711 (W 0.00332116) | 0.780241 ± 0.00357254 (W 0.00262339) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.851183 ± 0.00373972 (W 0.00289296) | 0.938584 ± 0.00974434 (W 0.00341198) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.7859 ± 0.0285753 (W 0.00376552) | 5.05454 ± 0.0229251 (W 0.00858329) | 4.96419 ± 0.0124972 (W 0.00521863) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 5.15633 ± 0.0545568 (W 0.0188267) | 6.29376 ± 0.0571372 (W 0.0245009) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.096487 ± 0.000379896 (W 0.00184946) | 0.102279 ± 0.00151521 (W 0.000589556) | 0.135791 ± 0.000547288 (W 0.00200024) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.101878 ± 0.000137477 (W 0.00189675) | 0.112788 ± 0.000628984 (W 0.00192511) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.566495 ± 0.00185323 (W 0.00339914) | 0.613184 ± 0.00300464 (W 0.00331616) | 0.763057 ± 0.00152605 (W 0.00292393) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.605978 ± 0.00509082 (W 0.00325586) | 0.66548 ± 0.00341765 (W 0.0040542) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.72588 ± 0.0132445 (W 0.0038904) | 4.67131 ± 0.0145826 (W 0.00749962) | 4.88477 ± 0.0206105 (W 0.00833617) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 3.86424 ± 0.0111878 (W 0.0122415) | 4.80151 ± 0.0136458 (W 0.00911793) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.039083 ± 0.000460479 (W 0.000259622) | 0.068887 ± 0.00592534 (W 0.00256018) | 0.065491 ± 0.000731664 (W 0.000685378) | 🔵 colmat 1.76× faster than `faer` | 🔵 colmat 1.68× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.048179 ± 0.000477607 (W 0.000265764) | 0.073526 ± 0.000265632 (W 0.00274683) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.14062 ± 0.000782283 (W 0.000857041) | 0.274737 ± 0.00488203 (W 0.014472) | 0.231557 ± 0.00194626 (W 0.00156618) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 1.65× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.183419 ± 0.00152086 (W 0.00169511) | 0.312657 ± 0.00396959 (W 0.00556299) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.649318 ± 0.000555478 (W 0.0031531) | 1.03935 ± 0.0244213 (W 0.0265453) | 1.13825 ± 0.00341852 (W 0.0022897) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.832198 ± 0.00323958 (W 0.00347018) | 1.1057 ± 0.0300034 (W 0.034015) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.073065 ± 0.000874199 (W 0.00071273) | 0.117006 ± 0.000961591 (W 0.00512326) | 0.14056 ± 0.000807577 (W 0.000342293) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 1.92× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.084716 ± 0.000686505 (W 0.000489381) | 0.130331 ± 0.00164264 (W 0.00436617) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.298381 ± 0.00211492 (W 0.00164663) | 0.614995 ± 0.00383288 (W 0.0110387) | 0.439421 ± 0.00235655 (W 0.00239193) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.47× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.571234 ± 0.00515076 (W 0.00390583) | 0.639953 ± 0.00488285 (W 0.0166339) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.47115 ± 0.00443101 (W 0.00508194) | 2.99312 ± 0.032611 (W 0.106494) | 2.39452 ± 0.0112349 (W 0.00491793) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 1.63× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.82652 ± 0.00982932 (W 0.00662) | 3.16665 ± 0.0441891 (W 0.0680106) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.074969 ± 0.00011921 (W 0.0021795) | 0.115764 ± 0.001357 (W 0.00488131) | 0.101347 ± 9.71236e-05 (W 0.00156491) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.087331 ± 0.000655294 (W 0.000318152) | 0.13038 ± 0.00356356 (W 0.00171873) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.282792 ± 0.000934669 (W 0.00251087) | 0.55207 ± 0.00109683 (W 0.0129286) | 0.415327 ± 0.00183819 (W 0.00231368) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 1.47× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.578246 ± 0.003826 (W 0.00408225) | 0.576051 ± 0.0108563 (W 0.0115936) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.3684 ± 0.00406358 (W 0.00342644) | 2.61495 ± 0.0214424 (W 0.070884) | 2.30532 ± 0.00816317 (W 0.00859167) | 🔵 colmat 1.91× faster than `faer` | 🔵 colmat 1.68× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 2.77343 ± 0.0127246 (W 0.00677072) | 2.81039 ± 0.0543221 (W 0.0761799) | - | · | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.040853 ± 0.000781432 (W 0.000178992) | 0.060312 ± 0.000376028 (W 0.000456005) | 0.066003 ± 0.000913572 (W 0.000351992) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 1.62× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.048538 ± 0.000784528 (W 0.000233798) | 0.067857 ± 0.000154546 (W 0.00158215) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.133197 ± 0.00242494 (W 0.00156564) | 0.271511 ± 0.0249336 (W 0.0194795) | 0.233892 ± 0.000960169 (W 0.00200474) | 🔵 colmat 2.04× faster than `faer` | 🔵 colmat 1.76× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.16763 ± 0.00192505 (W 0.000776126) | 0.294925 ± 0.00757187 (W 0.0175065) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.50931 ± 0.00346237 (W 0.00195571) | 0.908637 ± 0.0202186 (W 0.0246341) | 1.01767 ± 0.00411298 (W 0.00840496) | 🔵 colmat 1.78× faster than `faer` | 🔵 colmat 2.00× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.66503 ± 0.00438423 (W 0.00376363) | 1.01507 ± 0.0129902 (W 0.0347089) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.071593 ± 0.000525472 (W 0.000711588) | 0.113429 ± 0.00191479 (W 0.00141024) | 0.140319 ± 0.000504665 (W 0.00235188) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.96× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.122887 ± 6.54828e-05 (W 0.000391977) | 0.128477 ± 0.000230966 (W 0.00118842) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.256334 ± 0.000568895 (W 0.00276871) | 0.648903 ± 0.00135633 (W 0.0247522) | 0.373561 ± 0.00275883 (W 0.00308877) | 🔵 colmat 2.53× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.536035 ± 0.00187963 (W 0.00310802) | 0.653636 ± 0.0179369 (W 0.0203323) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 1.09441 ± 0.00105468 (W 0.00278386) | 2.77699 ± 0.0406229 (W 0.0669969) | 1.52937 ± 0.0179999 (W 0.00538789) | 🔵 colmat 2.54× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.45798 ± 0.0219893 (W 0.00759793) | 2.87905 ± 0.0223254 (W 0.0418069) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.068526 ± 0.00114718 (W 0.00173618) | 0.108891 ± 0.00067013 (W 0.00226536) | 0.137254 ± 0.000910188 (W 0.00127986) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 2.00× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.084967 ± 0.000942594 (W 0.000476787) | 0.122707 ± 0.000430647 (W 0.00111278) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.233191 ± 0.00151418 (W 0.000877053) | 0.550185 ± 0.0179607 (W 0.0265905) | 0.37303 ± 0.00122347 (W 0.00400883) | 🔵 colmat 2.36× faster than `faer` | 🔵 colmat 1.60× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.545937 ± 0.00543549 (W 0.00417781) | 0.591023 ± 0.0105879 (W 0.0183246) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.95654 ± 0.00693071 (W 0.00144694) | 2.37856 ± 0.0369094 (W 0.0717019) | 1.45733 ± 0.00875703 (W 0.00791731) | 🔵 colmat 2.49× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 2.42249 ± 0.0207502 (W 0.0113314) | 2.60358 ± 0.0481606 (W 0.0487256) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.182297 ± 0.000711589 (W 0.00211097) | 0.816508 ± 0.00281531 (W 0.00193718) | - | 🔵 colmat 4.48× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.190201 ± 0.000316986 (W 0.0015982) | 0.187466 ± 0.00102546 (W 0.00215765) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.15344 ± 0.00347199 (W 0.00235311) | 2.40594 ± 0.00986039 (W 0.00347503) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.17511 ± 0.00272305 (W 0.00346635) | 1.22347 ± 0.0215177 (W 0.00500934) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.60177 ± 0.0354171 (W 0.0094363) | 10.0051 ± 0.0464775 (W 0.0144478) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 7.77015 ± 0.0181638 (W 0.0152822) | 7.936 ± 0.114674 (W 0.0381751) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.288442 ± 0.00118319 (W 0.00317346) | 0.612903 ± 0.00233319 (W 0.00433607) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.439331 ± 0.00345978 (W 0.00283095) | 0.435415 ± 0.0020803 (W 0.00341342) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.90984 ± 0.00386887 (W 0.00274674) | 5.49537 ± 0.0210369 (W 0.0228639) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.67697 ± 0.0138621 (W 0.0064718) | 2.77579 ± 0.0116641 (W 0.00414047) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.4441 ± 0.0301076 (W 0.0206292) | 30.152 ± 0.106189 (W 0.0654857) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 18.014 ± 0.111189 (W 0.0663638) | 17.9404 ± 0.00615373 (W 0.0313056) | - | · | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.282882 ± 0.000492697 (W 0.00216968) | 0.446114 ± 0.00289121 (W 0.00368082) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.44938 ± 0.00271435 (W 0.00494229) | 0.430535 ± 0.00167915 (W 0.00392136) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.89119 ± 0.00862627 (W 0.0068654) | 4.30452 ± 0.033876 (W 0.00932102) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 2.10902 ± 0.0515207 (W 0.00516022) | 2.14501 ± 0.00422297 (W 0.00602694) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.3766 ± 0.0588776 (W 0.0214281) | 24.0614 ± 0.113507 (W 0.080755) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 14.8052 ± 0.105394 (W 0.0225133) | 14.8108 ± 0.0279487 (W 0.0167059) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.119361 ± 0.000658374 (W 0.000548859) | 0.557073 ± 0.139799 (W 0.138468) | - | 🔵 colmat 4.67× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.138546 ± 0.00113073 (W 0.000257235) | 0.180213 ± 0.00142713 (W 0.00283189) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.482641 ± 0.00255184 (W 0.00274994) | 1.99017 ± 0.00687047 (W 0.0186169) | - | 🔵 colmat 4.12× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.573028 ± 0.00430273 (W 0.00234243) | 0.836553 ± 0.013437 (W 0.00645703) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.27479 ± 0.00698897 (W 0.00375906) | 5.5828 ± 0.0799881 (W 0.161805) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.74168 ± 0.00784413 (W 0.00654403) | 3.40794 ± 0.0802918 (W 0.0860819) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.207343 ± 0.000550011 (W 0.00193131) | 0.591901 ± 0.00137092 (W 0.00507399) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.328046 ± 0.00169209 (W 0.00226179) | 0.306145 ± 0.00179616 (W 0.00327334) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.866261 ± 0.00160926 (W 0.00370258) | 4.42327 ± 0.0755169 (W 0.0675642) | - | 🔵 colmat 5.11× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.57454 ± 0.00809252 (W 0.00632373) | 1.54781 ± 0.00988107 (W 0.018465) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.69796 ± 0.0100616 (W 0.0134753) | 16.5286 ± 0.3441 (W 0.697945) | - | 🔵 colmat 3.52× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 8.87026 ± 0.0628475 (W 0.0857901) | 7.30172 ± 0.118772 (W 0.0862659) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.177212 ± 0.00154631 (W 0.00163984) | 0.415648 ± 0.00335643 (W 0.00405565) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.331553 ± 0.00259621 (W 0.00300276) | 0.297569 ± 0.0038453 (W 0.00247902) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.821067 ± 0.00202347 (W 0.0036139) | 3.30687 ± 0.0158826 (W 0.0585383) | - | 🔵 colmat 4.03× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.56556 ± 0.00514036 (W 0.00871629) | 1.46531 ± 0.00848695 (W 0.0177317) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.54588 ± 0.0185427 (W 0.011453) | 12.9546 ± 0.113534 (W 0.27613) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 8.79798 ± 0.020782 (W 0.0384236) | 7.32855 ± 0.07341 (W 0.1062) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.120754 ± 0.000114962 (W 0.000599796) | 0.534257 ± 0.266535 (W 0.148558) | - | 🔵 colmat 4.42× faster than `faer` (marginal: spread 0.64 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.136111 ± 0.000426897 (W 0.000265539) | 0.1675 ± 0.00101197 (W 0.000842427) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.409176 ± 0.000714827 (W 0.0020399) | 1.45352 ± 0.147925 (W 0.185863) | - | 🔵 colmat 3.55× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.488696 ± 0.00119359 (W 0.00243758) | 0.784178 ± 0.00666995 (W 0.0151689) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.57047 ± 0.00443081 (W 0.00551268) | 3.48959 ± 0.00884323 (W 0.138303) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.03511 ± 0.00389843 (W 0.00516788) | 2.74648 ± 0.00308413 (W 0.0239257) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.20053 ± 0.00168591 (W 0.00150891) | 0.576275 ± 0.00273951 (W 0.00489925) | - | 🔵 colmat 2.87× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.31387 ± 0.00112263 (W 0.00257368) | 0.295262 ± 0.000838369 (W 0.00185532) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.66564 ± 0.00489951 (W 0.00314824) | 4.16219 ± 0.0139304 (W 0.0517295) | - | 🔵 colmat 6.25× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.4388 ± 0.00243307 (W 0.00495275) | 1.45523 ± 0.0142128 (W 0.00688008) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.0549 ± 0.00709679 (W 0.00645118) | 13.7184 ± 0.0703283 (W 0.2575) | - | 🔵 colmat 4.49× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.35929 ± 0.0554761 (W 0.019826) | 5.89929 ± 0.0199513 (W 0.0431655) | - | 🔴 colmat 1.25× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.151981 ± 0.000812691 (W 0.002143) | 0.400249 ± 0.00337027 (W 0.00376732) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.309392 ± 0.00278783 (W 0.00263664) | 0.284064 ± 0.00232233 (W 0.00266876) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.654553 ± 0.0191515 (W 0.00491958) | 3.19792 ± 0.0165835 (W 0.022688) | - | 🔵 colmat 4.89× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.831998 ± 0.0036119 (W 0.00381729) | 1.32208 ± 0.00785039 (W 0.00866838) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.96731 ± 0.0176891 (W 0.00795681) | 10.9905 ± 0.250893 (W 0.11864) | - | 🔵 colmat 3.70× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.4011 ± 0.0512364 (W 0.0328348) | 5.66419 ± 0.0189074 (W 0.0217019) | - | 🔴 colmat 1.31× slower than `faer` | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.166899 ± 0.000217141 (W 0.00197168) | 0.164654 ± 0.000456228 (W 0.00202085) | 0.170665 ± 0.00100792 (W 0.00247882) | · | · |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.177247 ± 0.000629308 (W 0.00278875) | 0.174903 ± 0.00058125 (W 0.00184634) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.08847 ± 0.00543114 (W 0.00246226) | 1.07903 ± 0.00477965 (W 0.00296855) | 1.23593 ± 0.0049092 (W 0.00242173) | · | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 1.12706 ± 0.00595645 (W 0.00221478) | 1.12753 ± 0.00235869 (W 0.00254193) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.40243 ± 0.0341127 (W 0.020038) | 7.26532 ± 0.0539792 (W 0.00708453) | 8.80007 ± 0.0298099 (W 0.0118985) | · | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 7.57738 ± 0.0483544 (W 0.012759) | 7.53775 ± 0.0179175 (W 0.00972203) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.283634 ± 0.00130515 (W 0.00229706) | 0.279046 ± 0.00112343 (W 0.00295223) | 0.330481 ± 0.000183413 (W 0.00298644) | · | 🔵 colmat 1.17× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.407223 ± 0.00429332 (W 0.00376013) | 0.398778 ± 0.00619777 (W 0.00495023) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.88145 ± 0.00515352 (W 0.00463396) | 1.97533 ± 0.00428706 (W 0.0075647) | 2.25199 ± 0.00374825 (W 0.00490127) | · | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 2.51864 ± 0.0141928 (W 0.0074313) | 2.56375 ± 0.0201853 (W 0.00673455) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.3006 ± 0.0599719 (W 0.0272093) | 13.6129 ± 0.0579051 (W 0.0222484) | 15.9469 ± 0.0646917 (W 0.0118035) | · | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 16.891 ± 0.0953102 (W 0.0406506) | 16.8274 ± 0.0403464 (W 0.0319066) | - | · | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.295286 ± 0.00194695 (W 0.00234457) | 0.277502 ± 0.000483794 (W 0.00200272) | 0.347182 ± 0.00097849 (W 0.00287204) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.415678 ± 0.00345689 (W 0.00277244) | 0.397954 ± 0.00587098 (W 0.00291465) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.91527 ± 0.00143235 (W 0.00383364) | 1.91077 ± 0.00716517 (W 0.00458201) | 2.35741 ± 0.00456579 (W 0.00446192) | · | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.97845 ± 0.00856576 (W 0.00319245) | 1.93023 ± 0.00469153 (W 0.00527813) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.373 ± 0.0577944 (W 0.0184542) | 13.4833 ± 0.0402714 (W 0.0327338) | 16.2583 ± 0.0857329 (W 0.0124332) | · | 🔵 colmat 1.22× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 14.0123 ± 0.131439 (W 0.0476543) | 13.8789 ± 0.0556825 (W 0.0224786) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.106998 ± 0.000596638 (W 0.000336489) | 0.159765 ± 0.000766752 (W 0.0014487) | 0.100736 ± 0.000267782 (W 0.000520254) | 🔵 colmat 1.49× faster than `faer` | 🔴 colmat 1.06× slower than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.122836 ± 0.00151115 (W 0.000493682) | 0.168901 ± 0.0012425 (W 0.00270035) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.440843 ± 0.00408265 (W 0.00197667) | 0.666692 ± 0.00314077 (W 0.00527229) | 0.497458 ± 0.00105693 (W 0.00234083) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.528326 ± 0.0020029 (W 0.00266317) | 0.716446 ± 0.00213794 (W 0.00472661) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.17572 ± 0.00773018 (W 0.00338341) | 2.78035 ± 0.0963003 (W 0.133746) | 3.13801 ± 0.0229726 (W 0.00279861) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.54323 ± 0.0193611 (W 0.00433071) | 3.03532 ± 0.0219683 (W 0.0833683) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.172478 ± 0.000964376 (W 0.00181969) | 0.25425 ± 0.00311039 (W 0.0041951) | 0.214793 ± 0.000553188 (W 0.00273818) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.296498 ± 0.00186581 (W 0.00255043) | 0.275618 ± 0.00165153 (W 0.00431196) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.772816 ± 0.00324689 (W 0.00329718) | 1.273 ± 0.00494773 (W 0.0179385) | 1.1676 ± 0.00141169 (W 0.00366525) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.41279 ± 0.00293107 (W 0.00591328) | 1.3747 ± 0.00286253 (W 0.0147324) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.19571 ± 0.0133845 (W 0.00777602) | 5.94738 ± 0.0766709 (W 0.0940666) | 5.11669 ± 0.0191808 (W 0.00675531) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 7.71509 ± 0.0217275 (W 0.0179899) | 6.39991 ± 0.0988736 (W 0.0801271) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.172588 ± 0.00090441 (W 0.00260273) | 0.2437 ± 0.00443449 (W 0.00532548) | 0.208296 ± 0.000150489 (W 0.0018663) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.2981 ± 8.77572e-05 (W 0.00391362) | 0.265732 ± 0.00108699 (W 0.00173827) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.75212 ± 0.00144416 (W 0.00404564) | 1.17354 ± 0.00201562 (W 0.0137817) | 1.13077 ± 0.00551351 (W 0.00335884) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 1.50× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.41647 ± 0.00929556 (W 0.00736021) | 1.24642 ± 0.0105289 (W 0.0152628) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.10408 ± 0.0139794 (W 0.0164003) | 5.867 ± 0.0594263 (W 0.0810237) | 5.01561 ± 0.0282581 (W 0.00631741) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 7.64189 ± 0.0262411 (W 0.0172894) | 6.34205 ± 0.0396134 (W 0.0912983) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.109622 ± 0.000494146 (W 0.000585868) | 0.144108 ± 0.000281192 (W 0.00117187) | 0.133527 ± 0.00093817 (W 0.00170668) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.124038 ± 0.000181075 (W 0.000337732) | 0.155477 ± 0.000136698 (W 0.00114748) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.366007 ± 0.00199541 (W 0.00236862) | 0.647071 ± 0.00765024 (W 0.0173108) | 0.448268 ± 0.00506057 (W 0.00288915) | 🔵 colmat 1.77× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.440133 ± 0.00284467 (W 0.00307841) | 0.695384 ± 0.00690894 (W 0.0202802) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.48276 ± 0.00403862 (W 0.00429693) | 2.08691 ± 0.0236558 (W 0.0408313) | 2.00164 ± 0.00883953 (W 0.0125209) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.81498 ± 0.0226241 (W 0.00488522) | 2.3699 ± 0.019257 (W 0.0423534) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.152466 ± 0.00138672 (W 0.00160078) | 0.245343 ± 0.00232652 (W 0.00199575) | 0.200731 ± 0.00199304 (W 0.0037587) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.277793 ± 0.00236009 (W 0.00303677) | 0.261113 ± 0.00305682 (W 0.00352548) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.620466 ± 0.00345394 (W 0.00320773) | 1.19137 ± 0.0108044 (W 0.00804246) | 0.792776 ± 0.00266842 (W 0.00235996) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.27114 ± 0.00835843 (W 0.0031977) | 1.27266 ± 0.0111864 (W 0.0123755) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.80439 ± 0.0134597 (W 0.00668467) | 4.53476 ± 0.00463492 (W 0.0301208) | 2.82531 ± 0.0093965 (W 0.004699) | 🔵 colmat 1.62× faster than `faer` | · |  |
| `lu/n512/b1 [api-inclusive]` | 6.2595 ± 0.0511996 (W 0.0125695) | 4.98609 ± 0.0206319 (W 0.0187728) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.143004 ± 0.00113093 (W 0.000816404) | 0.230616 ± 0.000945886 (W 0.00194715) | 0.197374 ± 0.00279124 (W 0.00337054) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.280088 ± 0.00211435 (W 0.00399828) | 0.254079 ± 0.001474 (W 0.00121626) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.578087 ± 0.00435364 (W 0.00301631) | 1.02015 ± 0.00522348 (W 0.0142605) | 0.764583 ± 0.00248541 (W 0.0028143) | 🔵 colmat 1.76× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.718056 ± 0.00360073 (W 0.00254197) | 1.13641 ± 0.0076165 (W 0.0113686) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.68768 ± 0.0184248 (W 0.011705) | 4.10938 ± 0.0137093 (W 0.0217288) | 2.81196 ± 0.00414735 (W 0.00385615) | 🔵 colmat 1.53× faster than `faer` | · |  |
| `solve/n512/b1 [api-inclusive]` | 6.24001 ± 0.0102314 (W 0.0175774) | 4.80255 ± 0.0333069 (W 0.0258027) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `lu/128`, `lu/256`, `lu/512`, `solve/128`, `solve/256`, `solve/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.331117 ± 0.000174901 (W 0.00265101) | 0.861849 ± 0.0141134 (W 0.00336558) | - | 🔵 colmat 2.60× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.329129 ± 0.00206135 (W 0.0027236) | 0.468879 ± 0.00116511 (W 0.0054531) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 12.5489 ± 0.122748 (W 0.178842) | 40.1584 ± 0.232529 (W 0.120032) | - | 🔵 colmat 3.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 13.597 ± 0.312821 (W 0.083722) | 19.032 ± 0.0877002 (W 0.0310641) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.261345 ± 0.00276533 (W 0.0037339) | 0.902788 ± 0.00602183 (W 0.014039) | - | 🔵 colmat 3.45× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.282191 ± 0.000893838 (W 0.00207403) | 0.434933 ± 0.00212771 (W 0.00550664) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.97371 ± 0.0587567 (W 0.0340935) | 33.847 ± 0.069074 (W 0.059961) | - | 🔵 colmat 5.67× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 6.85566 ± 0.142133 (W 0.052948) | 12.7066 ± 0.0201387 (W 0.027232) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.2296 ± 0.000728315 (W 0.00329191) | 0.940994 ± 0.00704298 (W 0.0085583) | - | 🔵 colmat 4.10× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.281339 ± 0.000371541 (W 0.0017546) | 0.432339 ± 0.00576933 (W 0.00373691) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.28828 ± 0.0128457 (W 0.0196938) | 33.0894 ± 0.25492 (W 0.127885) | - | 🔵 colmat 7.72× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 6.00496 ± 0.0412363 (W 0.010379) | 11.7636 ± 0.0133146 (W 0.0299231) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.225657 ± 0.00179304 (W 0.00231129) | 0.367248 ± 0.00178369 (W 0.00334084) | 0.253923 ± 0.000716252 (W 0.00312557) | 🔵 colmat 1.63× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.241388 ± 0.00131416 (W 0.00355804) | 0.385063 ± 0.00107444 (W 0.002607) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.8159 ± 0.0581165 (W 0.0118588) | 12.7327 ± 0.0409534 (W 0.0155058) | 12.0221 ± 0.158011 (W 0.0648136) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.11× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.2946 ± 0.0484364 (W 0.0238951) | 13.2452 ± 0.0452702 (W 0.0287467) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.170264 ± 0.000442798 (W 0.00167914) | 0.326764 ± 0.00189641 (W 0.00237632) | 0.197064 ± 0.00112301 (W 0.00191194) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.198517 ± 0.00193096 (W 0.00185228) | 0.353363 ± 0.00294951 (W 0.00252285) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.96052 ± 0.0292618 (W 0.0148009) | 6.52347 ± 0.0329222 (W 0.0186301) | 6.21856 ± 0.00664686 (W 0.0141379) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.57× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.43958 ± 0.0168251 (W 0.00593338) | 7.01679 ± 0.0223144 (W 0.014386) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.165055 ± 0.000531094 (W 0.00221232) | 0.325982 ± 0.00247238 (W 0.00350601) | 0.177097 ± 0.000294551 (W 0.00121252) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.194209 ± 0.00089349 (W 0.00190996) | 0.354184 ± 0.000716691 (W 0.00298777) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.12732 ± 0.00525111 (W 0.00756013) | 5.62667 ± 0.0263392 (W 0.0152035) | 3.75124 ± 0.00661008 (W 0.00880209) | 🔵 colmat 1.80× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 3.57976 ± 0.00654143 (W 0.00606825) | 6.1372 ± 0.0340811 (W 0.0193929) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.922815 ± 0.0110768 (W 0.00478844) | 1.55729 ± 0.065183 (W 0.0170402) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.976512 ± 0.00365052 (W 0.00379086) | 1.00244 ± 0.0102024 (W 0.00171238) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 44.3147 ± 0.145467 (W 0.0401978) | 90.8196 ± 0.42675 (W 0.20881) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 46.2278 ± 0.160867 (W 0.0509063) | 53.9591 ± 0.225375 (W 0.0441366) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.550197 ± 0.00305615 (W 0.00319499) | 1.28619 ± 0.026114 (W 0.0104801) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.623817 ± 0.00180541 (W 0.00714682) | 0.748894 ± 0.0134628 (W 0.00194684) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 17.1593 ± 0.0279931 (W 0.0396256) | 66.3042 ± 0.0793021 (W 0.235091) | - | 🔵 colmat 3.86× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 20.1259 ± 0.157298 (W 0.320826) | 29.499 ± 0.0420148 (W 0.0560509) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.49261 ± 0.00332955 (W 0.00389089) | 1.35233 ± 0.0192883 (W 0.0131985) | - | 🔵 colmat 2.75× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.589148 ± 0.00269284 (W 0.00327268) | 0.706536 ± 0.00162469 (W 0.00491054) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.4006 ± 0.0266389 (W 0.0394377) | 61.9417 ± 0.189307 (W 0.281037) | - | 🔵 colmat 4.62× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 16.8818 ± 0.283922 (W 0.182934) | 25.3513 ± 0.0530351 (W 0.0321314) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.819761 ± 0.00384664 (W 0.00416703) | 0.851442 ± 0.00384849 (W 0.00393428) | 0.846322 ± 0.0012003 (W 0.00300017) | · | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.846374 ± 0.00105878 (W 0.0033588) | 0.883753 ± 0.00438861 (W 0.00279456) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 40.0647 ± 0.153341 (W 0.109052) | 40.6534 ± 0.122076 (W 0.0403774) | 45.0602 ± 0.141796 (W 0.0253677) | · | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 41.1555 ± 0.12477 (W 0.0464224) | 41.9691 ± 0.0727704 (W 0.100156) | - | · | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.447026 ± 0.00141963 (W 0.0030845) | 0.599207 ± 0.00210041 (W 0.00402601) | 0.569673 ± 0.000309768 (W 0.00254693) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.503068 ± 0.00213297 (W 0.00330226) | 0.654329 ± 0.00420335 (W 0.00337754) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.3401 ± 0.0336036 (W 0.0447657) | 16.2402 ± 0.0242774 (W 0.0361771) | 22.2043 ± 0.0516919 (W 0.0170567) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.66× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 14.3517 ± 0.0424712 (W 0.025752) | 17.3685 ± 0.0742658 (W 0.0554685) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.410883 ± 0.0039323 (W 0.00375745) | 0.554615 ± 0.00258959 (W 0.00379891) | 0.446213 ± 0.00149422 (W 0.00280158) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.09× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.47129 ± 0.00261466 (W 0.00327452) | 0.608044 ± 0.0012666 (W 0.00256921) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.2355 ± 0.0376451 (W 0.0305427) | 12.2503 ± 0.0423972 (W 0.0272778) | 12.5595 ± 0.0392767 (W 0.0185423) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.3825 ± 0.0145697 (W 0.031027) | 13.3553 ± 0.0355637 (W 0.0334013) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `lu/1024x64`, `lu/4096x256`, `solve/1024x64`, `solve/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 11.6636 ± 0.0259073 (W 0.0207078) | 109.638 ± 0.18381 (W 0.170086) | - | 🔵 colmat 9.40× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 15.8429 ± 0.0393928 (W 0.0188493) | 17.9909 ± 0.103656 (W 0.0441924) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.279636 ± 0.000362288 (W 0.0033029) | 3.19647 ± 0.0203785 (W 0.0133645) | - | 🔵 colmat 11.43× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.420781 ± 0.00732584 (W 0.00303712) | 0.513659 ± 0.00305371 (W 0.00514003) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 3.47945 ± 0.00590153 (W 0.0137845) | 55.2902 ± 9.27348 (W 10.0091) | - | 🔵 colmat 15.89× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 6.317 ± 0.087983 (W 0.0524515) | 9.48205 ± 0.1556 (W 0.53371) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.12345 ± 0.00171944 (W 0.00218391) | 1.65476 ± 0.0117082 (W 0.374481) | - | 🔵 colmat 13.40× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.208815 ± 0.00199724 (W 0.00204346) | 0.385259 ± 0.0156678 (W 0.0162288) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 2.03357 ± 0.00723124 (W 0.00687874) | 31.185 ± 1.48574 (W 5.67146) | - | 🔵 colmat 15.34× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 4.64944 ± 0.147413 (W 0.0231197) | 7.87215 ± 0.0628445 (W 0.101313) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.096567 ± 0.00151274 (W 0.00169505) | 0.958445 ± 0.0143224 (W 0.00817835) | - | 🔵 colmat 9.93× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.180543 ± 0.00613459 (W 0.00174012) | 0.330982 ± 0.00784642 (W 0.00583918) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 12.729 ± 0.00719648 (W 0.0436644) | 14.6154 ± 0.00838901 (W 0.019806) | 19.2477 ± 0.0601248 (W 0.0355671) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 13.1274 ± 0.0292075 (W 0.0459046) | 13.3713 ± 0.0446855 (W 0.0256163) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.308329 ± 0.00131163 (W 0.00244732) | 0.297651 ± 0.000541665 (W 0.00306164) | 0.656741 ± 0.00479183 (W 0.00344271) | · | 🔵 colmat 2.13× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.324002 ± 0.0014801 (W 0.00244794) | 0.311898 ± 0.000976951 (W 0.0036687) | - | · | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 3.31752 ± 0.00789246 (W 0.0128881) | 5.27937 ± 0.179258 (W 0.334201) | 11.0617 ± 0.0334257 (W 0.0421947) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 3.33× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 3.74391 ± 0.0111704 (W 0.0129411) | 5.65311 ± 0.138431 (W 0.324702) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.096347 ± 0.000622111 (W 0.000448046) | 0.144942 ± 0.00762787 (W 0.0207424) | 0.43798 ± 0.00138308 (W 0.00197277) | 🔵 colmat 1.50× faster than `faer` | 🔵 colmat 4.55× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.118048 ± 0.000574949 (W 0.000540809) | 0.172128 ± 0.0240771 (W 0.0204589) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 1.80171 ± 0.00732842 (W 0.0077725) | 3.19551 ± 0.0293409 (W 0.105321) | 8.61162 ± 0.0146736 (W 0.0117208) | 🔵 colmat 1.77× faster than `faer` | 🔵 colmat 4.78× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 2.21205 ± 0.00775585 (W 0.0084964) | 3.32215 ± 0.0626637 (W 0.0882614) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.062491 ± 0.000233868 (W 0.000239097) | 0.110955 ± 0.00397216 (W 0.00443348) | 0.398335 ± 0.00242185 (W 0.00197269) | 🔵 colmat 1.78× faster than `faer` | 🔵 colmat 6.37× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.087441 ± 0.000277808 (W 0.000444733) | 0.14042 ± 0.00189189 (W 0.00222977) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 40.9176 ± 0.101757 (W 0.0958509) | 141.552 ± 1.96574 (W 0.0948232) | - | 🔵 colmat 3.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 47.8209 ± 0.194016 (W 0.164945) | 46.8143 ± 0.0635833 (W 0.0826461) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.874826 ± 0.00121578 (W 0.00268679) | 3.80122 ± 0.0129344 (W 0.011591) | - | 🔵 colmat 4.35× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 1.01951 ± 0.00994155 (W 0.00396188) | 1.04778 ± 0.0134492 (W 0.00264565) | - | · | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 11.2715 ± 0.0248437 (W 0.019108) | 42.8695 ± 13.5043 (W 12.7397) | - | 🔵 colmat 3.80× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 16.8311 ± 0.178607 (W 0.144588) | 21.9097 ± 0.99717 (W 1.77617) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.287107 ± 0.000869664 (W 0.00290843) | 1.83205 ± 0.0199505 (W 0.360449) | - | 🔵 colmat 6.38× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.417541 ± 0.00523329 (W 0.00181505) | 0.511472 ± 0.0438065 (W 0.0542382) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 6.26962 ± 0.0107919 (W 0.0310086) | 36.1628 ± 0.0205376 (W 0.432584) | - | 🔵 colmat 5.77× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 11.3454 ± 0.0496409 (W 0.122016) | 13.7546 ± 0.276475 (W 0.373831) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.192702 ± 0.000918125 (W 0.00145677) | 1.04806 ± 0.0368019 (W 0.168499) | - | 🔵 colmat 5.44× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.324268 ± 0.00305793 (W 0.00123466) | 0.439649 ± 0.00630964 (W 0.0188175) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 42.4213 ± 0.0182701 (W 0.0346224) | 43.1904 ± 0.0152715 (W 0.107547) | 53.8453 ± 0.194553 (W 0.0714933) | · | 🔵 colmat 1.27× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 43.4017 ± 0.0612519 (W 0.106954) | 42.1523 ± 0.0418636 (W 0.0662273) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.880276 ± 0.00858674 (W 0.00349847) | 0.8614 ± 0.00600596 (W 0.00511766) | 1.18493 ± 0.00428082 (W 0.00253169) | · | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.903209 ± 0.00506821 (W 0.00142543) | 0.891677 ± 0.00319817 (W 0.00342516) | - | · | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 11.2642 ± 0.0346956 (W 0.125561) | 17.6072 ± 0.482176 (W 0.886656) | 28.6189 ± 0.0417668 (W 0.0162241) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 2.54× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 12.3828 ± 0.0833287 (W 0.0132095) | 17.9677 ± 0.0139736 (W 1.05054) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.253859 ± 0.00204935 (W 0.00260449) | 0.357203 ± 0.0479068 (W 0.0539576) | 0.694507 ± 0.00581756 (W 0.00213467) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 2.74× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.29381 ± 0.000227572 (W 0.00251647) | 0.400947 ± 0.0132661 (W 0.0543414) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 6.02592 ± 0.0718319 (W 0.0197702) | 8.99995 ± 0.116256 (W 0.351648) | 21.7922 ± 0.0475596 (W 0.0187108) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 3.62× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 6.97779 ± 0.00258673 (W 0.118314) | 9.48943 ± 0.258519 (W 0.412681) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.148605 ± 0.00127254 (W 0.00222372) | 0.225245 ± 0.00625444 (W 0.00705285) | 0.566947 ± 0.00431287 (W 0.00263054) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 3.82× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.198707 ± 0.0018601 (W 0.00217646) | 0.280578 ± 0.00169928 (W 0.00369771) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `lu/256x4096`, `lu/64x1024`, `solve/256x4096`, `solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
