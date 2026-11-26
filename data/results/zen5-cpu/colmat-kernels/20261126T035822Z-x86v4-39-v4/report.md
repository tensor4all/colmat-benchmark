# `colmat-kernels` on `zen5-cpu`

- colmat (fork, measured) commit: `be19a8001e37e5db55c083c90bd8ca10269d1ab5`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon, x86-v4`
- colmat / faer rows: `colmat` (the provider in the measured revision; see library and upstream above); `faer` (the provider in the measured revision; see library and upstream above)
- harness commit: `a899f37cb62992709cb756c0aa0c93aa5785a922`
- toolkit commit: `82bff6ea4508e156b9e661550c1dc65ad662581c`
- measured binary SHA-256: `98632b95255641c9fb030372ae47c8e9f9594b8d427fc45908c0ba31b70800d6`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-26T04:36:54.822496Z`
- timing policy: v2, median of 5 reps, priming 500 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu colmat-kernels --checkout /home/shinaoka/projects/tensor4all/colmat-rs-39-measured --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor openblas --regimes linalg-small,linalg-mid,linalg-tall,linalg-wide --dtypes f64,c64 --threads 1,4,8 --aa 3 --no-publish --label x86v4-39-v4`
- harness environment: `FBENCH_ORDER` unset, `FBENCH_PROFILE=zen5-cpu`
- raw data: `data/results/zen5-cpu/colmat-kernels/20261126T035822Z-x86v4-39-v4/`

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
Spread guide (computed from this run's own rows): per-engine between-start range median 1.06%, worst 145.90% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 48 of 1188 engine cells exceed 5% and carry `!`; median within-process relative SD 0.73%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
| `cholesky/n32/b1` | 0.003547 ± 2.51661e-05 (W 2.59654e-05) | 0.004168 ± 5.7735e-05 (W 1.67332e-05) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002825 ± 9.50438e-06 (W 6.38146e-05) | 0.002815 ± 3.64737e-05 (W 6.76018e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013866 ± 0.000355964 (W 0.000124521) | 0.034825 ± 0.00154436 (W 3.29424e-05) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010911 ± 9.65764e-05 (W 9.24895e-05) | 0.01081 ± 9.11062e-05 (W 0.000105139) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031028 ± 0.0003139 (W 0.000134742) | 0.164637 ± 0.000665357 (W 0.00176049) | - | 🔵 colmat 5.31× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.025818 ± 0.000130767 (W 0.000185775) | 0.025708 ± 0.000116895 (W 0.0010698) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006563 ± 0.000107193 (W 4.43847e-05) | 0.008065 ± 7e-05 (W 5.64907e-05) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008225 ± 0.000200301 (W 0.000159318) | 0.007484 ± 1.52753e-05 (W 9.9001e-05) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.023424 ± 0.000371624 (W 9.04434e-05) | 0.035787 ± 0.00028 (W 0.000165633) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.029124 ± 0.000487594 (W 0.000247184) | 0.027512 ± 0.000202981 (W 0.000148756) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.056635 ± 0.000245257 (W 0.000137372) | 0.086742 ± 9.93093e-05 (W 0.000402152) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.076533 ± 0.000429004 (W 0.00205511) | 0.070772 ± 0.00171527 (W 0.00251096) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006933 ± 1.1547e-05 (W 3.36155e-05) | 0.009037 ± 1.57162e-05 (W 6.33103e-05) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009538 ± 4.56107e-05 (W 6.41872e-05) | 0.008556 ± 5.1316e-05 (W 0.000229715) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.022482 ± 0.000173494 (W 9.32738e-05) | 0.037089 ± 9.38847e-05 (W 9.88433e-05) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.030798 ± 7.02163e-05 (W 0.000105505) | 0.028684 ± 0.000209953 (W 0.000114914) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.051255 ± 0.000170696 (W 0.000116962) | 0.088023 ± 0.000407835 (W 0.00203273) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.076904 ± 0.00205299 (W 0.00165383) | 0.072846 ± 0.00126819 (W 0.00259991) | - | 🔴 colmat 1.06× slower than `faer` (marginal: spread 0.82 of the difference) | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003456 ± 3.59212e-05 (W 3.46598e-05) | 0.004288 ± 0.000110151 (W 2.23607e-05) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002815 ± 2.52389e-05 (W 5.7855e-05) | 0.002805 ± 4e-05 (W 5.37615e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014197 ± 7.75263e-05 (W 8.43161e-05) | 0.034805 ± 0.00146999 (W 2.91548e-05) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.011251 ± 0.000167482 (W 0.000113579) | 0.010901 ± 0.000126295 (W 7.17029e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.03192 ± 0.000243003 (W 0.000193913) | 0.05402 ± 0.000334018 (W 0.0452193) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.027381 ± 7.42765e-05 (W 0.000223411) | 0.031559 ± 0.000174443 (W 0.00033871) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006573 ± 2.51064e-05 (W 5.82683e-05) | 0.008005 ± 0.000120968 (W 7.96869e-05) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008456 ± 0.000135094 (W 6.64831e-05) | 0.007444 ± 5.74195e-05 (W 6.34137e-05) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.030486 ± 5.01431e-05 (W 0.000829304) | 0.038061 ± 0.000303701 (W 0.000438849) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.029805 ± 0.000461754 (W 0.000214536) | 0.030898 ± 1.0504e-05 (W 0.00109039) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.062667 ± 0.000956045 (W 0.000681011) | 0.091389 ± 0.00121027 (W 0.000908246) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.070049 ± 0.000851226 (W 0.000466156) | 0.076802 ± 0.000274793 (W 0.00214115) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007545 ± 1.53948e-05 (W 5.36656e-05) | 0.008946 ± 0.00012329 (W 4.98317e-05) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010169 ± 8e-05 (W 6.65582e-05) | 0.008476 ± 7.23418e-05 (W 6.50385e-05) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.027591 ± 0.000505748 (W 0.00102934) | 0.039544 ± 0.000923842 (W 0.00045022) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.032841 ± 0.000168277 (W 0.000117857) | 0.031739 ± 0.000199141 (W 0.000702257) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.05378 ± 0.000615328 (W 0.00110265) | 0.090539 ± 0.00102435 (W 0.00216893) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.074458 ± 0.000226483 (W 0.00221638) | 0.07482 ± 0.000429132 (W 0.000575579) | - | · | - | `*` one row has no measurement here |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003446 ± 6.48614e-05 (W 3.06969e-05) | 0.004138 ± 0.000134536 (W 3.04959e-05) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002915 ± 7.02377e-05 (W 6.88644e-05) | 0.002906 ± 1.51767e-05 (W 7.17307e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.014096 ± 0.00018037 (W 7.97446e-05) | 0.032681 ± 0.00156925 (W 0.00168797) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.011261 ± 8.14453e-05 (W 0.00013396) | 0.01098 ± 4.3589e-05 (W 0.000117415) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.03165 ± 0.000310516 (W 0.000223888) | 0.054693 ± 0.000436234 (W 0.00149181) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.027721 ± 0.000195219 (W 0.000273007) | 0.03234 ± 0.000245153 (W 0.00055952) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006572 ± 8.24035e-05 (W 2.95838e-05) | 0.008085 ± 8.44413e-05 (W 6.08219e-05) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008486 ± 7.0946e-05 (W 0.000125379) | 0.007544 ± 6.35085e-05 (W 6.39555e-05) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.031759 ± 0.000284678 (W 0.000915529) | 0.03777 ± 0.000271509 (W 0.000238594) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.030206 ± 0.000183394 (W 0.000311141) | 0.031549 ± 0.000126835 (W 0.00121112) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.066293 ± 0.00090298 (W 0.00156598) | 0.093665 ± 0.000567891 (W 0.00147283) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.071082 ± 0.000565033 (W 0.00221992) | 0.080339 ± 0.000296311 (W 0.00245172) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007604 ± 9.07377e-05 (W 6.07306e-05) | 0.008966 ± 4.7085e-05 (W 7.21401e-05) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010159 ± 7.50555e-05 (W 6.19855e-05) | 0.008656 ± 7.84241e-05 (W 0.000192745) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.027701 ± 0.000402626 (W 0.000819462) | 0.039523 ± 0.000743594 (W 0.000308502) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.032531 ± 0.000133493 (W 0.00016675) | 0.03233 ± 0.000306649 (W 0.00055399) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.055894 ± 0.000781673 (W 0.00109887) | 0.094065 ± 0.00219396 (W 0.000445782) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.077204 ± 0.00084489 (W 0.000637226) | 0.078046 ± 0.0005508 (W 0.00152831) | - | · | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002805 ± 4.72582e-05 (W 3.28634e-05) | 0.002765 ± 2e-05 (W 3.08221e-05) | 0.002955 ± 1.47309e-05 (W 3.50599e-05) | · | 🔵 colmat 1.05× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002795 ± 2.08167e-05 (W 5.92385e-05) | 0.002775 ± 2.51661e-05 (W 6.73587e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010239 ± 4.04145e-05 (W 6.50385e-05) | 0.012354 ± 0.000276118 (W 6.80309e-05) | 0.013014 ± 0.000116363 (W 7.51665e-05) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.27× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0107 ± 6.245e-05 (W 0.000113315) | 0.01086 ± 0.000128582 (W 0.000109864) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.023854 ± 9.71253e-05 (W 0.000111455) | 0.025538 ± 3.21455e-05 (W 0.00157227) | 0.031679 ± 0.000253379 (W 8.57974e-05) | 🔵 colmat 1.07× faster than `faer` (marginal: spread 0.93 of the difference) | 🔵 colmat 1.33× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.025448 ± 2.51064e-05 (W 0.000129951) | 0.025718 ± 8.13654e-05 (W 0.000137169) | - | · | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00516 ± 8.55395e-05 (W 7.33519e-05) | 0.005941 ± 0.000100664 (W 4.20381e-05) | 0.007264 ± 9.68005e-05 (W 0.000108527) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006532 ± 2.08167e-05 (W 0.000113886) | 0.007374 ± 7.81025e-05 (W 0.000102569) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.021179 ± 2.3388e-05 (W 0.000271162) | 0.023103 ± 5.50757e-05 (W 0.000108578) | 0.030197 ± 0.000241056 (W 0.000199292) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024505 ± 0.00026 (W 0.000226308) | 0.02657 ± 0.000152699 (W 0.000222956) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.050314 ± 0.000445234 (W 0.00187968) | 0.05409 ± 0.000229705 (W 0.000145204) | 0.07056 ± 0.00025644 (W 0.00203946) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.06477 ± 0.00152015 (W 0.00223595) | 0.068248 ± 0.0012053 (W 0.00226451) | - | 🔵 colmat 1.05× faster than `faer` (marginal: spread 0.78 of the difference) | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006151 ± 5.1316e-05 (W 7.46344e-05) | 0.006512 ± 9.76746e-05 (W 4.33624e-05) | 0.007544 ± 6.55744e-05 (W 5.95164e-05) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007755 ± 1.57162e-05 (W 8.78368e-05) | 0.008035 ± 0.000160728 (W 7.11632e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.021901 ± 6.70099e-05 (W 0.000132665) | 0.023634 ± 5.7735e-06 (W 0.000112402) | 0.029545 ± 0.00025249 (W 0.00141056) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.024997 ± 2.6839e-05 (W 0.00011324) | 0.027241 ± 0.000554834 (W 0.000156446) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.051476 ± 0.0001505 (W 0.000140397) | 0.055743 ± 0.000365694 (W 0.00015575) | 0.068608 ± 0.000420038 (W 0.000159142) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.33× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.064879 ± 0.00174669 (W 0.00296414) | 0.069109 ± 0.00168781 (W 0.00221001) | - | 🔵 colmat 1.07× faster than `faer` (marginal: spread 0.81 of the difference) | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002765 ± 3.51188e-05 (W 2.70185e-05) | 0.002705 ± 2.51661e-05 (W 8.13634e-05) | 0.002895 ± 1e-05 (W 4.50355e-05) | · | · | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002755 ± 5.1316e-05 (W 8.52074e-05) | 0.002935 ± 3.05014e-05 (W 3.17522e-05) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.01048 ± 7.57188e-05 (W 3.83966e-05) | 0.012163 ± 5.25389e-05 (W 5.44821e-05) | 0.013134 ± 0.000106927 (W 7.8653e-05) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.011011 ± 5.0797e-05 (W 0.0001173) | 0.01092 ± 4.2194e-05 (W 6.39867e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.024936 ± 0.000154727 (W 9.01665e-05) | 0.029665 ± 0.000441564 (W 0.00053868) | 0.03207 ± 0.000225581 (W 0.000109051) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.02708 ± 0.000168968 (W 0.000299377) | 0.031468 ± 0.00044576 (W 0.000379885) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0054 ± 5.50757e-05 (W 8.26753e-05) | 0.006031 ± 4.20357e-05 (W 6.58027e-05) | 0.007354 ± 8.88819e-05 (W 0.000115974) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006943 ± 8.71799e-05 (W 7.43922e-05) | 0.007434 ± 9.64365e-05 (W 0.000119457) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.021289 ± 5.17397e-05 (W 0.000205232) | 0.026018 ± 0.000810746 (W 0.000721598) | 0.030047 ± 0.000208007 (W 0.00201418) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.024766 ± 8.5773e-05 (W 0.000255524) | 0.030427 ± 0.000220774 (W 0.000244282) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.041117 ± 0.000229829 (W 0.00103319) | 0.061034 ± 0.00101742 (W 0.00220172) | 0.070262 ± 0.000152056 (W 0.000356198) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 1.71× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.05919 ± 0.000877552 (W 0.00058004) | 0.075832 ± 0.000193779 (W 0.00197289) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006702 ± 5.56776e-05 (W 3.71927e-05) | 0.006582 ± 4.52806e-05 (W 5.52377e-05) | 0.007595 ± 0.000105311 (W 0.00012818) | · | 🔵 colmat 1.13× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008316 ± 2.58908e-05 (W 6.00442e-05) | 0.008075 ± 9.07377e-05 (W 7.60868e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.023714 ± 0.000195526 (W 0.000165325) | 0.026399 ± 0.000246982 (W 0.000393396) | 0.029255 ± 0.000202939 (W 0.000103606) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.02691 ± 0.000257734 (W 0.00021508) | 0.030317 ± 8.52467e-05 (W 0.000740425) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.045786 ± 0.000389358 (W 0.000298977) | 0.05947 ± 0.000240544 (W 0.000291231) | 0.069771 ± 6.16523e-05 (W 0.000144068) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.062577 ± 0.0015702 (W 0.000271399) | 0.072956 ± 0.000443306 (W 0.000786874) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002134 ± 1.58219e-05 (W 2.16795e-05) | 0.002084 ± 2.25167e-05 (W 3.74192e-05) | 0.002906 ± 2.09841e-05 (W 4.47571e-05) | · | 🔵 colmat 1.36× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002796 ± 0.000118196 (W 9.1815e-05) | 0.002836 ± 3.76431e-05 (W 5.57162e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.009077 ± 7.68136e-05 (W 4.89929e-05) | 0.008956 ± 2.25167e-05 (W 7.54473e-05) | 0.013164 ± 0.000157121 (W 0.000115374) | · | 🔵 colmat 1.45× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.01099 ± 9.21973e-05 (W 0.000150583) | 0.01099 ± 6.40859e-05 (W 9.01665e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.022302 ± 0.000100664 (W 0.000102393) | 0.026269 ± 5.2915e-05 (W 0.000505412) | 0.0321 ± 5.08953e-05 (W 0.000170183) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.027111 ± 0.0002305 (W 0.000224411) | 0.03211 ± 0.000187271 (W 0.000386213) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.00547 ± 6.55515e-05 (W 0.000101952) | 0.005891 ± 5.34821e-05 (W 8.87654e-05) | 0.007353 ± 9.76746e-05 (W 0.00014836) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.34× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006893 ± 9.07377e-05 (W 6.04152e-05) | 0.007404 ± 0.000651154 (W 7.04365e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.021751 ± 0.000104403 (W 0.000206222) | 0.025828 ± 0.000365664 (W 0.000141821) | 0.029875 ± 0.000258369 (W 0.000175625) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.024816 ± 0.000113725 (W 0.000195474) | 0.030948 ± 0.000339718 (W 0.000483444) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.041698 ± 6.28517e-05 (W 0.000596755) | 0.060774 ± 0.000180334 (W 0.0012227) | 0.06974 ± 0.000113377 (W 0.00195872) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.059249 ± 0.00156486 (W 0.00240993) | 0.078115 ± 0.000483116 (W 0.000933923) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.006693 ± 0.000287786 (W 8.63551e-05) | 0.006452 ± 2.64575e-05 (W 4.98618e-05) | 0.007474 ± 0.0001545 (W 5.85944e-05) | · | 🔵 colmat 1.12× faster than `lapack-openblas` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008416 ± 4.63717e-05 (W 5.54076e-05) | 0.008095 ± 5.7735e-05 (W 5.20884e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.023414 ± 0.00037118 (W 0.000128807) | 0.026219 ± 0.000125819 (W 0.000251467) | 0.029265 ± 0.000101933 (W 0.000114251) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.027211 ± 0.000460523 (W 0.000268459) | 0.030707 ± 0.000366509 (W 0.000311343) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.045064 ± 0.000449163 (W 0.00026001) | 0.05914 ± 9.66454e-05 (W 0.00128787) | 0.06984 ± 0.000211079 (W 0.00186737) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 1.55× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.063709 ± 0.00046801 (W 0.000552543) | 0.076081 ± 0.000168049 (W 0.00300295) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005791 ± 2.12211e-05 (W 2.33067e-05) | 0.009047 ± 2e-05 (W 2.52626e-05) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00524 ± 3.55949e-05 (W 4.49589e-05) | 0.004869 ± 4.58258e-05 (W 6.09442e-05) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.033563 ± 0.000216049 (W 0.000135585) | 0.070141 ± 0.000245358 (W 0.00168621) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.031699 ± 7.14446e-05 (W 0.000329213) | 0.025237 ± 0.000191364 (W 0.000113671) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.09109 ± 0.000287192 (W 0.00171581) | 0.446074 ± 0.00142726 (W 0.00218311) | - | 🔵 colmat 4.90× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.087333 ± 8.40714e-05 (W 0.00026521) | 0.078936 ± 8.54478e-05 (W 0.00227397) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012003 ± 6.67108e-05 (W 8.67179e-05) | 0.010801 ± 0.000263572 (W 0.000391811) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.014707 ± 6.4291e-05 (W 0.000128765) | 0.011191 ± 0.000160936 (W 6.6598e-05) | - | 🔴 colmat 1.31× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.052448 ± 0.00040612 (W 0.000378596) | 0.056474 ± 0.00032392 (W 0.000202196) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.062886 ± 0.000439501 (W 0.000443719) | 0.052487 ± 0.000132217 (W 0.000391627) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.137527 ± 0.000238739 (W 0.000201785) | 0.148807 ± 0.000544342 (W 0.002292) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.206807 ± 0.00131153 (W 0.00280278) | 0.188382 ± 0.00201535 (W 0.00249052) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012102 ± 4.7648e-05 (W 6.28753e-05) | 0.011752 ± 6.65833e-05 (W 5.39555e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0161 ± 0.000120625 (W 0.000115543) | 0.012593 ± 4.78226e-05 (W 8.81493e-05) | - | 🔴 colmat 1.28× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.050554 ± 0.00041734 (W 0.000134243) | 0.05853 ± 0.000245439 (W 0.000246725) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.066162 ± 8.76432e-05 (W 0.000338346) | 0.055734 ± 0.000130309 (W 0.000271469) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.133737 ± 0.000744237 (W 0.00190891) | 0.152845 ± 0.000741783 (W 0.000308468) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.214206 ± 0.00280336 (W 0.00294718) | 0.188883 ± 0.00069146 (W 0.00458745) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.005771 ± 4.58258e-05 (W 6.01024e-05) | 0.009057 ± 7.56329e-05 (W 1.04067e-05) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00516 ± 4.16333e-05 (W 5.63028e-05) | 0.004949 ± 4.04145e-05 (W 3.71376e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.029435 ± 6.11583e-05 (W 0.000177908) | 0.07012 ± 0.000182248 (W 0.000183508) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.028864 ± 0.000330273 (W 0.000153288) | 0.024967 ± 0.000175274 (W 9.88433e-05) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.071323 ± 0.00051341 (W 0.00026432) | 0.167233 ± 0.140783 (W 0.141248) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.070881 ± 0.000526544 (W 0.000169765) | 0.083405 ± 0.000134686 (W 0.000483733) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012092 ± 7.02875e-05 (W 0.000131329) | 0.0106 ± 7.75263e-05 (W 1.80887e-05) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015168 ± 0.000205508 (W 0.000162976) | 0.011181 ± 3.66106e-05 (W 0.000157786) | - | 🔴 colmat 1.36× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.04818 ± 0.000479919 (W 0.000316931) | 0.056626 ± 0.000307624 (W 0.000159942) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.054902 ± 0.000323002 (W 0.000252568) | 0.052598 ± 4.04145e-05 (W 0.000164193) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.118201 ± 0.000598134 (W 0.000851538) | 0.145449 ± 0.000841511 (W 0.00167641) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.169156 ± 5.81062e-05 (W 0.00228563) | 0.137407 ± 0.00161979 (W 0.00267043) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012854 ± 0.000105532 (W 6.04152e-05) | 0.011792 ± 0.000100167 (W 6.88346e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016691 ± 2e-05 (W 9.68504e-05) | 0.012583 ± 5.50757e-05 (W 6.44531e-05) | - | 🔴 colmat 1.33× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.045845 ± 0.000206747 (W 0.000498268) | 0.058199 ± 0.000436734 (W 0.0014919) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.05859 ± 0.000452594 (W 0.000373428) | 0.055564 ± 0.00044091 (W 0.000317032) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.102882 ± 0.00106694 (W 0.000694926) | 0.150219 ± 0.000415838 (W 0.00321342) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.168816 ± 0.00147337 (W 0.00267021) | 0.139509 ± 0.00146142 (W 0.00198119) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.006301 ± 0.000115284 (W 6.63325e-05) | 0.009017 ± 1.52753e-05 (W 3.22211e-05) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005229 ± 3.5171e-05 (W 4.72229e-05) | 0.004929 ± 8.62168e-05 (W 3.12938e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.029134 ± 0.000452681 (W 0.000174075) | 0.070512 ± 0.000141905 (W 0.000273205) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.029004 ± 0.000131149 (W 0.000244705) | 0.025117 ± 5.03322e-05 (W 0.000102962) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.071954 ± 0.000393439 (W 0.000244775) | 0.132508 ± 0.000594073 (W 0.12411) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.070662 ± 0.000251462 (W 0.000267377) | 0.083305 ± 0.000190832 (W 0.0004789) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.012243 ± 2.88675e-05 (W 6.98727e-05) | 0.01052 ± 7.2111e-05 (W 4.35695e-05) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.014967 ± 4.90951e-05 (W 7.56968e-05) | 0.011261 ± 0.000180093 (W 0.000228865) | - | 🔴 colmat 1.33× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.054131 ± 0.000154881 (W 0.000612393) | 0.055774 ± 0.000543425 (W 0.00174708) | - | · | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.054391 ± 0.00026266 (W 0.000431927) | 0.052608 ± 0.000167413 (W 0.000240708) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.125182 ± 0.00064517 (W 0.000850704) | 0.146754 ± 0.000139145 (W 0.00139378) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.165179 ± 0.0101749 (W 0.00189858) | 0.138149 ± 0.000444274 (W 0.000736424) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.012824 ± 1.73205e-05 (W 4.64564e-05) | 0.011913 ± 2.52389e-05 (W 6.87764e-05) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016721 ± 5.25389e-05 (W 0.000119951) | 0.012583 ± 4.55741e-05 (W 8.82451e-05) | - | 🔴 colmat 1.33× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.04788 ± 0.000327796 (W 0.000672953) | 0.058379 ± 0.000481865 (W 0.000619965) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.057207 ± 0.00036074 (W 0.000320235) | 0.055924 ± 0.000340735 (W 0.000167244) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.100077 ± 0.000483923 (W 0.000709298) | 0.148508 ± 0.000864292 (W 0.00148394) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.168786 ± 0.0060767 (W 0.00308115) | 0.142356 ± 0.00262776 (W 0.00211491) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-small, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004578 ± 3.99625e-05 (W 3.98158e-05) | 0.004629 ± 7.57188e-05 (W 6.2284e-05) | 0.004829 ± 6.4291e-05 (W 5.41295e-05) | · | 🔵 colmat 1.05× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005279 ± 3.19427e-05 (W 5.64119e-05) | 0.004858 ± 1.70392e-05 (W 8.93661e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.028392 ± 0.000143447 (W 6.99943e-05) | 0.024475 ± 8.31084e-05 (W 0.000109171) | 0.027972 ± 0.000114503 (W 0.000168554) | 🔴 colmat 1.16× slower than `faer` | · |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.031237 ± 5.71343e-05 (W 0.000121459) | 0.024735 ± 6.45006e-05 (W 7.63171e-05) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.07978 ± 7.58222e-05 (W 9.58243e-05) | 0.072456 ± 0.000260001 (W 0.000180774) | 0.078808 ± 0.000260041 (W 0.000154549) | 🔴 colmat 1.10× slower than `faer` | · |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.085838 ± 0.000444099 (W 0.0002188) | 0.078225 ± 3.23161e-05 (W 0.000212266) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.01092 ± 2.70247e-05 (W 7.33233e-05) | 0.009287 ± 0.00014526 (W 6.85296e-05) | 0.011451 ± 0.000104213 (W 0.00012876) | 🔴 colmat 1.18× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012935 ± 9.98065e-05 (W 0.000121587) | 0.011271 ± 8.77515e-05 (W 8.80926e-05) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.048871 ± 0.00018577 (W 0.000206398) | 0.046056 ± 0.000318854 (W 0.000135986) | 0.055233 ± 0.000196723 (W 0.000245396) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.13× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.054682 ± 0.000289621 (W 0.00174056) | 0.050574 ± 0.000339531 (W 0.000314455) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.13385 ± 0.000289485 (W 0.000363958) | 0.126125 ± 0.000273279 (W 0.00265442) | 0.153764 ± 0.000317303 (W 0.00181708) | 🔴 colmat 1.06× slower than `faer` | 🔵 colmat 1.15× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.190917 ± 0.00128265 (W 0.00253276) | 0.181696 ± 0.00250234 (W 0.00187875) | - | · | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011912 ± 9.70069e-05 (W 0.000100584) | 0.01072 ± 6.1101e-05 (W 0.00011006) | 0.012583 ± 2.51661e-05 (W 9.56201e-05) | 🔴 colmat 1.11× slower than `faer` | 🔵 colmat 1.06× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013846 ± 9.55353e-05 (W 6.30872e-05) | 0.012494 ± 0.000120112 (W 0.000445714) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.053098 ± 0.000419035 (W 0.000157379) | 0.048791 ± 0.000188945 (W 0.000151243) | 0.05914 ± 0.000307313 (W 0.00013409) | 🔴 colmat 1.09× slower than `faer` | 🔵 colmat 1.11× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.058628 ± 0.000622918 (W 0.000288089) | 0.054081 ± 0.000803018 (W 0.000166072) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.13936 ± 0.000710321 (W 0.00122869) | 0.130414 ± 0.000420975 (W 0.00196824) | 0.161882 ± 0.000721477 (W 0.0018984) | 🔴 colmat 1.07× slower than `faer` | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.197235 ± 0.0022445 (W 0.00403908) | 0.185337 ± 0.000407681 (W 0.00321233) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004518 ± 1.0504e-05 (W 3.25008e-05) | 0.004428 ± 5.08626e-05 (W 3.91152e-05) | 0.004889 ± 4.16333e-05 (W 5.41295e-05) | · | 🔵 colmat 1.08× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00526 ± 3.4641e-05 (W 7.74416e-05) | 0.004909 ± 1.52753e-05 (W 5.35724e-05) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.025217 ± 6.55744e-05 (W 0.000171378) | 0.024175 ± 0.000239653 (W 0.0001232) | 0.03233 ± 0.000348927 (W 0.000553198) | · | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.028453 ± 0.000105987 (W 0.000199675) | 0.024766 ± 9.71253e-05 (W 9.1815e-05) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.060101 ± 9.23815e-05 (W 0.00016168) | 0.076834 ± 0.000135736 (W 0.00199182) | 0.072966 ± 0.000888764 (W 0.000955877) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.069841 ± 0.000567222 (W 0.000373835) | 0.081893 ± 0.000242723 (W 0.001022) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.010951 ± 4.90951e-05 (W 0.000147479) | 0.009377 ± 3.56417e-05 (W 5.14121e-05) | 0.011281 ± 7.63763e-05 (W 0.000136931) | 🔴 colmat 1.17× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013195 ± 0.000107472 (W 0.000149292) | 0.011391 ± 6.65833e-05 (W 0.000118677) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.040786 ± 0.000200651 (W 0.00158319) | 0.046167 ± 1.00167e-05 (W 0.0001555) | 0.054993 ± 0.000190211 (W 0.00427228) | 🔵 colmat 1.13× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.04834 ± 0.00036756 (W 0.000284062) | 0.050633 ± 0.000115678 (W 8.52485e-05) | - | · | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.09091 ± 0.000467155 (W 0.000731122) | 0.121698 ± 0.000873437 (W 0.00115547) | 0.154118 ± 0.000385734 (W 0.00148424) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.152242 ± 0.00295646 (W 0.00102075) | 0.13491 ± 0.000903773 (W 0.00173032) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012663 ± 0.000179584 (W 7.98705e-05) | 0.01052 ± 5.72742e-05 (W 7.02659e-05) | 0.012623 ± 7.07602e-05 (W 9.60224e-05) | 🔴 colmat 1.20× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014366 ± 0.000115846 (W 7.18631e-05) | 0.012483 ± 5.50757e-05 (W 6.30872e-05) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.045325 ± 0.000425508 (W 0.000322455) | 0.048851 ± 0.000328695 (W 0.000100609) | 0.059882 ± 0.00025064 (W 0.00024056) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.051136 ± 0.000377638 (W 0.000273046) | 0.053569 ± 0.000119308 (W 0.000211199) | - | · | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.093454 ± 0.000616248 (W 0.000306564) | 0.126056 ± 0.00191769 (W 0.00183559) | 0.160978 ± 0.000602151 (W 0.00175912) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.153264 ± 0.00326388 (W 0.00321777) | 0.13936 ± 0.000967802 (W 0.00228569) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004178 ± 3.21455e-05 (W 3.999e-05) | 0.004138 ± 6.35085e-06 (W 4.2063e-05) | 0.004999 ± 0.000106522 (W 7.78351e-05) | · | 🔵 colmat 1.20× faster than `lapack-openblas` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00534 ± 2.51661e-05 (W 5.55446e-05) | 0.004889 ± 5.7735e-06 (W 3.35887e-05) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.023905 ± 0.000100167 (W 7.65526e-05) | 0.022282 ± 5.08626e-05 (W 9.49963e-05) | 0.034645 ± 0.000238831 (W 0.000498188) | 🔴 colmat 1.07× slower than `faer` | 🔵 colmat 1.45× faster than `lapack-openblas` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.028593 ± 0.000208727 (W 0.00022326) | 0.024767 ± 9.13072e-05 (W 8.25548e-05) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.063448 ± 0.000350029 (W 0.000378709) | 0.076894 ± 0.000200101 (W 0.000396748) | 0.081312 ± 0.000497388 (W 0.00252498) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.070001 ± 0.000550013 (W 0.00032378) | 0.082945 ± 0.000459511 (W 0.000432916) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011061 ± 3.05014e-05 (W 0.000117619) | 0.009077 ± 8.73689e-05 (W 5.68085e-05) | 0.011261 ± 4.3589e-05 (W 0.000112561) | 🔴 colmat 1.22× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.012995 ± 6.09945e-05 (W 0.000150941) | 0.011321 ± 0.0001249 (W 0.000129306) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.042339 ± 6.65833e-05 (W 0.000268103) | 0.045104 ± 3.99625e-05 (W 0.00223119) | 0.055082 ± 0.000121149 (W 0.00016377) | 🔵 colmat 1.07× faster than `faer` (marginal: spread 0.81 of the difference) | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `lu/n64/b1 [api-inclusive]` | 0.048359 ± 0.000373101 (W 0.000454133) | 0.050944 ± 0.000184464 (W 0.000273213) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.086742 ± 0.000631684 (W 0.000763411) | 0.1241 ± 0.000781138 (W 0.00147126) | 0.153707 ± 0.000492643 (W 0.00236774) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-openblas` |  |
| `lu/n96/b1 [api-inclusive]` | 0.148247 ± 0.00366174 (W 0.00163066) | 0.135694 ± 0.0010196 (W 0.000545583) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.012283 ± 8.7178e-05 (W 0.000101588) | 0.010339 ± 6.08276e-05 (W 6.36263e-05) | 0.012614 ± 0.000106927 (W 7.91473e-05) | 🔴 colmat 1.19× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014437 ± 0.000134536 (W 0.000105472) | 0.012343 ± 5.50757e-05 (W 5.59035e-05) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.043972 ± 0.0002705 (W 0.000440679) | 0.04791 ± 0.000243847 (W 0.000126412) | 0.059231 ± 0.000201575 (W 0.00143912) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `solve/n64/b1 [api-inclusive]` | 0.050074 ± 0.000231877 (W 0.000559354) | 0.0534 ± 0.000340472 (W 0.000164185) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.083727 ± 0.000687004 (W 0.000406326) | 0.126637 ± 0.00040464 (W 0.002399) | 0.16145 ± 0.000513843 (W 0.00207981) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 1.93× faster than `lapack-openblas` |  |
| `solve/n96/b1 [api-inclusive]` | 0.146945 ± 0.00620849 (W 0.00276115) | 0.139721 ± 0.00160697 (W 0.00147961) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `lu/32`, `lu/64`, `lu/96`, `solve/32`, `solve/64`, `solve/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.05881 ± 0.000375279 (W 0.000228463) | 0.295453 ± 0.000930344 (W 0.0018367) | - | 🔵 colmat 5.02× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.057007 ± 0.000167157 (W 0.000189159) | 0.059141 ± 7.74683e-05 (W 0.000559407) | - | · | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.36915 ± 0.00401133 (W 0.00290961) | 0.83477 ± 0.00607637 (W 0.00435426) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.330499 ± 0.00286067 (W 0.00390384) | 0.367447 ± 0.00223008 (W 0.00345131) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.97787 ± 0.019864 (W 0.00928448) | 3.69572 ± 0.0117475 (W 0.00867055) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.13992 ± 0.00698305 (W 0.00473131) | 2.45772 ± 0.00438807 (W 0.00557466) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.106078 ± 0.000913532 (W 0.00131173) | 0.265773 ± 0.00303034 (W 0.00269988) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.125274 ± 0.00106153 (W 0.00058039) | 0.179536 ± 0.0031926 (W 0.00479166) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.574654 ± 0.00283013 (W 0.00315882) | 2.57701 ± 0.0101713 (W 0.00633375) | - | 🔵 colmat 4.48× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.928408 ± 0.00779061 (W 0.00353217) | 1.08157 ± 0.00289092 (W 0.00221628) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.87492 ± 0.021004 (W 0.00778384) | 20.1053 ± 0.108605 (W 0.0526779) | - | 🔵 colmat 5.19× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.60638 ± 0.0328852 (W 0.0220007) | 7.4132 ± 0.0407486 (W 0.0437193) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.096671 ± 0.000456147 (W 0.00137694) | 0.195475 ± 0.0033987 (W 0.00338293) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.128498 ± 0.000594477 (W 0.00238864) | 0.142256 ± 0.00175244 (W 0.00210844) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.541404 ± 0.000925302 (W 0.00361833) | 1.90673 ± 0.0125227 (W 0.0107624) | - | 🔵 colmat 3.52× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.694379 ± 0.00566118 (W 0.00257689) | 0.821696 ± 0.00555979 (W 0.00354062) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.73167 ± 0.00731334 (W 0.00579414) | 13.6084 ± 0.0309314 (W 0.0555926) | - | 🔵 colmat 3.65× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.22944 ± 0.0200099 (W 0.0055358) | 5.91371 ± 0.0761523 (W 0.0133821) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.048311 ± 0.000603645 (W 0.00124311) | 0.204834 ± 0.00284754 (W 0.0671267) | - | 🔵 colmat 4.24× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.050814 ± 9.70842e-05 (W 0.000579169) | 0.069029 ± 0.000331189 (W 0.00191633) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.18262 ± 0.000581569 (W 0.00249446) | 0.598712 ± 0.0544721 (W 0.0631725) | - | 🔵 colmat 3.28× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.206658 ± 0.00252734 (W 0.00253319) | 0.351574 ± 0.00366547 (W 0.0156463) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.971119 ± 0.00578966 (W 0.00261221) | 2.3262 ± 0.0680778 (W 0.146685) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 0.944115 ± 0.00409684 (W 0.00198626) | 1.55827 ± 0.0539326 (W 0.0234425) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.115663 ± 0.00129025 (W 0.00117934) | 0.278951 ± 0.00422037 (W 0.00562323) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.105764 ± 0.000723946 (W 0.000726079) | 0.158124 ± 0.000584959 (W 0.00271153) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.350004 ± 0.00259087 (W 0.00811134) | 2.25196 ± 0.0662331 (W 0.0582235) | - | 🔵 colmat 6.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1 [api-inclusive]` | 0.66654 ± 0.00460334 (W 0.00448498) | 0.783947 ± 0.00285364 (W 0.0106892) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.70264 ± 0.00610637 (W 0.00757795) | 13.1482 ± 0.509957 (W 0.37892) | - | 🔵 colmat 7.72× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1 [api-inclusive]` | 3.31892 ± 0.0256409 (W 0.00884212) | 4.12995 ± 0.0348467 (W 0.0947023) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.092251 ± 0.000698514 (W 0.00137665) | 0.224199 ± 0.00389651 (W 0.00263047) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.109974 ± 0.00113648 (W 0.000538999) | 0.153938 ± 0.00187502 (W 0.00590337) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.349122 ± 0.000976754 (W 0.00272977) | 1.70243 ± 0.0219206 (W 0.04196) | - | 🔵 colmat 4.88× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.661232 ± 0.00195809 (W 0.00383906) | 0.72231 ± 0.0057671 (W 0.0102395) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.56056 ± 0.00502961 (W 0.00707516) | 9.8733 ± 0.216313 (W 0.186647) | - | 🔵 colmat 6.33× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.22327 ± 0.0383969 (W 0.018229) | 3.77392 ± 0.0923987 (W 0.0482794) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.050156 ± 0.000379525 (W 0.00106888) | 0.107526 ± 0.000457577 (W 0.0524801) | - | 🔵 colmat 2.14× faster than `faer` (marginal: spread 0.91 of the difference) | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.051389 ± 0.000224306 (W 0.000686726) | 0.062146 ± 0.000538098 (W 0.000784128) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.16081 ± 0.00209222 (W 0.00118325) | 0.600862 ± 0.0205283 (W 0.0263685) | - | 🔵 colmat 3.74× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.190655 ± 0.000353394 (W 0.0019219) | 0.339104 ± 0.00619071 (W 0.024855) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.804184 ± 0.00435736 (W 0.003055) | 2.02786 ± 0.0407358 (W 0.0488956) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.777738 ± 0.00420528 (W 0.00327467) | 1.42361 ± 0.0229658 (W 0.0178963) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.121577 ± 0.00193726 (W 0.00239276) | 0.276768 ± 0.00162374 (W 0.00135664) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.145889 ± 0.000161753 (W 0.00103677) | 0.1512 ± 0.00175341 (W 0.00161636) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.286409 ± 0.00116621 (W 0.00265634) | 2.27207 ± 0.0131935 (W 0.0223826) | - | 🔵 colmat 7.93× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.633003 ± 0.00506943 (W 0.00339884) | 0.808201 ± 0.0172561 (W 0.0192958) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.23122 ± 0.0038653 (W 0.00427991) | 11.5919 ± 0.00383588 (W 0.100018) | - | 🔵 colmat 9.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.92797 ± 0.0274816 (W 0.00935503) | 3.94077 ± 0.0270918 (W 0.0362077) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.084147 ± 0.000776597 (W 0.000605912) | 0.223598 ± 0.000699715 (W 0.00246116) | - | 🔵 colmat 2.66× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.106569 ± 0.000579146 (W 0.000489139) | 0.148658 ± 0.000905097 (W 0.00170136) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.278749 ± 0.000882907 (W 0.0031599) | 1.69047 ± 0.00644735 (W 0.0293038) | - | 🔵 colmat 6.06× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.632918 ± 0.00675217 (W 0.00428239) | 0.745755 ± 0.00778977 (W 0.0214706) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.10613 ± 0.00405697 (W 0.00208203) | 8.75814 ± 0.074757 (W 0.0603566) | - | 🔵 colmat 7.92× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.88079 ± 0.0190785 (W 0.00798188) | 3.58014 ± 0.0772669 (W 0.0883724) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.04808 ± 4.64794e-05 (W 0.000113667) | 0.046587 ± 0.000156666 (W 0.000177804) | 0.062327 ± 0.000330079 (W 0.000152438) | · | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.050675 ± 0.000244841 (W 0.000151777) | 0.054021 ± 0.00027315 (W 0.000208065) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.281326 ± 0.00107759 (W 0.00296825) | 0.275235 ± 0.00134688 (W 0.00328749) | 0.366836 ± 0.000790825 (W 0.00268565) | · | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.296795 ± 0.00023034 (W 0.00313543) | 0.309549 ± 0.0019012 (W 0.00249641) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.95574 ± 0.0205194 (W 0.00409492) | 1.87349 ± 0.00939868 (W 0.00177331) | 2.54567 ± 0.00942818 (W 0.00375006) | · | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.97594 ± 0.00891308 (W 0.00215176) | 2.0133 ± 0.0108965 (W 0.00177198) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.098394 ± 0.000706126 (W 0.00170194) | 0.104936 ± 0.00112879 (W 0.00180447) | 0.141705 ± 0.00114011 (W 0.00242933) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.10691 ± 0.00130156 (W 0.00130159) | 0.156523 ± 0.00107506 (W 0.00249161) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.569686 ± 0.00184056 (W 0.00338255) | 0.690041 ± 0.00377447 (W 0.00244305) | 0.781691 ± 0.000548341 (W 0.00420981) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.37× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.84393 ± 0.0086829 (W 0.00550383) | 0.94556 ± 0.00236499 (W 0.00474282) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.70806 ± 0.0210618 (W 0.00739789) | 5.07461 ± 0.020919 (W 0.00780919) | 4.94964 ± 0.0457682 (W 0.00797075) | 🔵 colmat 1.37× faster than `faer` | 🔵 colmat 1.33× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 5.08838 ± 0.0233359 (W 0.0189831) | 6.34557 ± 0.00524272 (W 0.0172569) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.097944 ± 0.000437267 (W 0.00171448) | 0.105457 ± 0.000840134 (W 0.000242642) | 0.136285 ± 0.0017454 (W 0.0022302) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.106117 ± 0.00045597 (W 0.00224675) | 0.116877 ± 0.000560722 (W 0.00184125) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.564204 ± 0.00369909 (W 0.00296614) | 0.621211 ± 0.000467797 (W 0.00281535) | 0.761153 ± 0.00384018 (W 0.00261325) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.610352 ± 0.00481357 (W 0.00319628) | 0.674631 ± 0.00440629 (W 0.00305667) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.69132 ± 0.0114036 (W 0.00649824) | 4.71041 ± 0.017167 (W 0.00505461) | 4.89641 ± 0.00934258 (W 0.00625103) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 1.33× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 3.83376 ± 0.00816037 (W 0.00531192) | 4.8622 ± 0.0163584 (W 0.0134114) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.036128 ± 0.000215007 (W 0.000242331) | 0.061064 ± 0.0024337 (W 0.00333001) | 0.065622 ± 0.000323889 (W 0.000836787) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 1.82× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.044737 ± 0.000160782 (W 0.000391549) | 0.06495 ± 0.000759455 (W 0.00232278) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.133048 ± 0.000500806 (W 0.000936221) | 0.271753 ± 0.013697 (W 0.0163811) | 0.232673 ± 0.000874979 (W 0.00179003) | 🔵 colmat 2.04× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.172892 ± 0.00101618 (W 0.00153126) | 0.298675 ± 0.00577174 (W 0.00635938) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.635448 ± 0.00313439 (W 0.00264327) | 1.00004 ± 0.0200338 (W 0.0250709) | 1.13766 ± 0.00465069 (W 0.00785232) | 🔵 colmat 1.57× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.818195 ± 0.00123571 (W 0.00299445) | 1.09033 ± 0.0279963 (W 0.0275853) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.073982 ± 0.000423536 (W 0.000677354) | 0.120203 ± 0.00120087 (W 0.00548067) | 0.140541 ± 0.000666296 (W 0.00176323) | 🔵 colmat 1.62× faster than `faer` | 🔵 colmat 1.90× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.085379 ± 0.000636472 (W 0.000565262) | 0.135663 ± 0.00143922 (W 0.00320572) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.30073 ± 0.000982227 (W 0.00269287) | 0.617484 ± 0.00838113 (W 0.0118562) | 0.44162 ± 0.00161738 (W 0.00379831) | 🔵 colmat 2.05× faster than `faer` | 🔵 colmat 1.47× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.573632 ± 0.00286727 (W 0.00404576) | 0.649284 ± 0.000890873 (W 0.00968107) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.45676 ± 0.0056152 (W 0.00414554) | 3.00433 ± 0.0240624 (W 0.0975984) | 2.38808 ± 0.0123237 (W 0.00452099) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.64× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.81428 ± 0.0311684 (W 0.0118778) | 3.11305 ± 0.0968112 (W 0.0668984) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n128/b1` | 0.078206 ± 0.000753793 (W 0.00212641) | 0.115304 ± 0.000497036 (W 0.00261383) | 0.10222 ± 0.000388925 (W 0.00207008) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.088079 ± 0.000851443 (W 0.000424135) | 0.130845 ± 0.00108555 (W 0.00271088) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.285601 ± 0.00195305 (W 0.00337606) | 0.558308 ± 0.0157907 (W 0.0130932) | 0.414938 ± 0.00146288 (W 0.00290647) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.578461 ± 0.00101062 (W 0.00543564) | 0.588771 ± 0.00503311 (W 0.00544391) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.36042 ± 0.00397335 (W 0.00715091) | 2.61767 ± 0.00824539 (W 0.0529904) | 2.30619 ± 0.0130479 (W 0.00576739) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 2.76233 ± 0.00896164 (W 0.0135859) | 2.80663 ± 0.072156 (W 0.0532116) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.040475 ± 0.000236628 (W 0.000327613) | 0.055723 ± 0.000345127 (W 0.000492366) | 0.066404 ± 0.000147514 (W 0.00120346) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 1.64× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.044693 ± 0.000129388 (W 0.00020681) | 0.057637 ± 0.000757368 (W 0.00100537) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.132928 ± 0.00016415 (W 0.000712336) | 0.255275 ± 0.0231098 (W 0.0192585) | 0.235719 ± 0.00155465 (W 0.00221623) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-openblas` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.157993 ± 0.000522228 (W 0.000850634) | 0.280703 ± 0.0152398 (W 0.0231591) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.519131 ± 0.00150302 (W 0.00272005) | 0.8506 ± 0.0099828 (W 0.0274981) | 1.01872 ± 0.000897343 (W 0.00386638) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.96× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.648733 ± 0.000900401 (W 0.00319273) | 0.986567 ± 0.0223257 (W 0.036476) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.072084 ± 0.000430363 (W 0.000560194) | 0.116063 ± 0.00119365 (W 0.000999093) | 0.141001 ± 0.00119139 (W 0.00226896) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 1.96× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.12501 ± 0.000986798 (W 0.000618911) | 0.13066 ± 0.000421676 (W 0.00142095) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.262023 ± 0.00111129 (W 0.00284533) | 0.611893 ± 0.00658174 (W 0.0204001) | 0.375241 ± 0.00191929 (W 0.00361107) | 🔵 colmat 2.34× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 0.537781 ± 0.0040902 (W 0.00319172) | 0.679816 ± 0.00981487 (W 0.0200134) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.08491 ± 0.0055807 (W 0.00288988) | 2.77215 ± 0.0350745 (W 0.0517625) | 1.50885 ± 0.0109392 (W 0.00779379) | 🔵 colmat 2.56× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 2.45698 ± 0.00200064 (W 0.00986912) | 2.98733 ± 0.020863 (W 0.0295303) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.071574 ± 0.000569906 (W 0.00161365) | 0.110196 ± 0.000567962 (W 0.00185946) | 0.13532 ± 0.00170403 (W 0.00197448) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.89× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.086335 ± 0.000880294 (W 0.000641577) | 0.125305 ± 0.000565398 (W 0.0017428) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.237375 ± 0.000874438 (W 0.00191656) | 0.5372 ± 0.022719 (W 0.0245551) | 0.374176 ± 0.00265253 (W 0.00269198) | 🔵 colmat 2.26× faster than `faer` | 🔵 colmat 1.58× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n256/b1 [api-inclusive]` | 0.547133 ± 0.0165282 (W 0.0043382) | 0.566815 ± 0.023439 (W 0.027985) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n512/b1` | 0.951269 ± 0.00540603 (W 0.00289593) | 2.46761 ± 0.0762886 (W 0.0629046) | 1.45795 ± 0.00858331 (W 0.0105986) | 🔵 colmat 2.59× faster than `faer` | 🔵 colmat 1.53× faster than `lapack-openblas` | `!` between-start range over 5% |
| `solve/n512/b1 [api-inclusive]` | 2.41177 ± 0.0500898 (W 0.00685629) | 2.566 ± 0.0596718 (W 0.0496888) | - | 🔵 colmat 1.06× faster than `faer` (marginal: spread 0.71 of the difference) | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.185076 ± 0.0010055 (W 0.00236091) | 0.814884 ± 0.00235865 (W 0.00290329) | - | 🔵 colmat 4.40× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.190908 ± 1.0116e-05 (W 0.00166486) | 0.176851 ± 0.00107815 (W 0.00173473) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.15847 ± 0.00231619 (W 0.00211818) | 2.41101 ± 0.00895217 (W 0.00498559) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.16765 ± 0.00315278 (W 0.00365395) | 1.18366 ± 0.0153961 (W 0.00362726) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.62585 ± 0.0303088 (W 0.0128694) | 9.99885 ± 0.0211653 (W 0.0155418) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 7.7091 ± 0.0122964 (W 0.0222303) | 7.79452 ± 0.0775259 (W 0.0184984) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.282249 ± 0.00103414 (W 0.00281378) | 0.620271 ± 0.00249484 (W 0.00378905) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.434722 ± 0.00219681 (W 0.0031423) | 0.430997 ± 0.00195584 (W 0.00231819) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.87725 ± 0.00893551 (W 0.00277212) | 5.47575 ± 0.014121 (W 0.0173754) | - | 🔵 colmat 2.92× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.67322 ± 0.0093643 (W 0.00583177) | 2.76914 ± 0.00650485 (W 0.00756451) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.3102 ± 0.049997 (W 0.0242883) | 29.7588 ± 0.0383879 (W 0.088458) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 17.993 ± 0.0422934 (W 0.0461763) | 17.836 ± 0.114236 (W 0.0328109) | - | · | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.276026 ± 0.000791999 (W 0.00278221) | 0.462264 ± 0.00705267 (W 0.00356644) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.448063 ± 0.00184727 (W 0.00429136) | 0.42704 ± 0.00436855 (W 0.00368615) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.84406 ± 0.0135173 (W 0.0019208) | 4.246 ± 0.0231292 (W 0.0154056) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 2.08667 ± 0.00528708 (W 0.00301032) | 2.13378 ± 0.00776372 (W 0.00363546) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.2044 ± 0.0486362 (W 0.0160698) | 23.661 ± 0.113133 (W 0.0321791) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 14.7916 ± 0.0117323 (W 0.0366789) | 14.7697 ± 0.0388614 (W 0.0579555) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.122168 ± 0.000472023 (W 0.00172668) | 0.554672 ± 0.143517 (W 0.140357) | - | 🔵 colmat 4.54× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.136783 ± 0.000764979 (W 0.000334339) | 0.170928 ± 0.00299516 (W 0.00329537) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.48424 ± 0.002549 (W 0.00285255) | 1.99764 ± 0.00327302 (W 0.116921) | - | 🔵 colmat 4.13× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.565707 ± 0.00281465 (W 0.00244145) | 0.769798 ± 0.0209656 (W 0.00243529) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.28027 ± 0.00224284 (W 0.00792634) | 5.28658 ± 0.141844 (W 0.21681) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.71806 ± 0.0714106 (W 0.0104906) | 3.3308 ± 0.0769572 (W 0.0757282) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.202408 ± 0.000588062 (W 0.00242489) | 0.600475 ± 0.00288368 (W 0.00472838) | - | 🔵 colmat 2.97× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.324104 ± 0.00317715 (W 0.00386146) | 0.305531 ± 0.00189525 (W 0.00296979) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.840485 ± 0.00249436 (W 0.00203928) | 4.4192 ± 0.00889803 (W 0.117794) | - | 🔵 colmat 5.26× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.55871 ± 0.0356414 (W 0.0090839) | 1.55682 ± 0.0134751 (W 0.0130672) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.65348 ± 0.00797125 (W 0.0138294) | 16.8912 ± 0.242557 (W 0.34823) | - | 🔵 colmat 3.63× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 8.8003 ± 0.0188666 (W 0.0197248) | 7.32378 ± 0.0534033 (W 0.0792813) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.174454 ± 0.000453402 (W 0.00143719) | 0.432831 ± 0.0015732 (W 0.00376161) | - | 🔵 colmat 2.48× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.325144 ± 0.00245736 (W 0.00246744) | 0.30203 ± 0.00437832 (W 0.00363517) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.813091 ± 0.00196697 (W 0.00156323) | 3.39326 ± 0.0198588 (W 0.0501162) | - | 🔵 colmat 4.17× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.5456 ± 0.00586291 (W 0.00451212) | 1.43609 ± 0.0023199 (W 0.00285336) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.52508 ± 0.0223885 (W 0.0249116) | 13.3974 ± 0.0524114 (W 0.418322) | - | 🔵 colmat 2.96× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 8.74804 ± 0.0315062 (W 0.0340589) | 7.1856 ± 0.0479569 (W 0.0867167) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.122718 ± 0.000363037 (W 0.00053137) | 0.529907 ± 0.152313 (W 0.187073) | - | 🔵 colmat 4.32× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.136456 ± 0.000532611 (W 0.000213519) | 0.15673 ± 9.59236e-05 (W 0.00159196) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.415084 ± 0.000170623 (W 0.00188805) | 1.42677 ± 0.138843 (W 0.220866) | - | 🔵 colmat 3.44× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.4807 ± 0.00251417 (W 0.00266047) | 0.744766 ± 0.00410859 (W 0.0182151) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.57797 ± 0.00898773 (W 0.00448672) | 3.73885 ± 0.198996 (W 0.182595) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.00278 ± 0.00181057 (W 0.00539924) | 2.65219 ± 0.0262225 (W 0.0530409) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.197509 ± 0.00117609 (W 0.0022295) | 0.585341 ± 0.00100003 (W 0.00371171) | - | 🔵 colmat 2.96× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.304643 ± 0.00137193 (W 0.00223962) | 0.288302 ± 0.00140047 (W 0.00227917) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.64731 ± 0.00156561 (W 0.00274325) | 4.10564 ± 0.019258 (W 0.0321333) | - | 🔵 colmat 6.34× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.42375 ± 0.0116544 (W 0.00296816) | 1.45365 ± 0.0260561 (W 0.0153012) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.02506 ± 0.0138741 (W 0.0139258) | 13.3531 ± 0.134199 (W 0.19072) | - | 🔵 colmat 4.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.40559 ± 0.0336569 (W 0.0383088) | 5.87459 ± 0.0274551 (W 0.0265902) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.149098 ± 0.000844628 (W 0.00120767) | 0.410855 ± 0.000461633 (W 0.00217814) | - | 🔵 colmat 2.76× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.303686 ± 0.00128945 (W 0.00259212) | 0.28254 ± 0.0028876 (W 0.00171391) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.636066 ± 0.00300401 (W 0.00651768) | 3.14068 ± 0.0252261 (W 0.020051) | - | 🔵 colmat 4.94× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.811278 ± 0.0011456 (W 0.00224327) | 1.36099 ± 0.0304215 (W 0.0137381) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.93399 ± 0.00804477 (W 0.00983297) | 10.7572 ± 0.101082 (W 0.132332) | - | 🔵 colmat 3.67× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.38733 ± 0.0448463 (W 0.0343688) | 5.5978 ± 0.00654326 (W 0.0383619) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.168516 ± 0.000556935 (W 0.00221734) | 0.153385 ± 0.000510365 (W 0.00214652) | 0.170609 ± 0.000369949 (W 0.00201605) | 🔴 colmat 1.10× slower than `faer` | · |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.178661 ± 0.00117084 (W 0.00217873) | 0.164255 ± 0.000618343 (W 0.00205941) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.0827 ± 0.00517554 (W 0.000850619) | 1.03655 ± 0.00296081 (W 0.00488618) | 1.23291 ± 0.00553474 (W 0.00437699) | · | 🔵 colmat 1.14× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 1.11597 ± 0.00344733 (W 0.00376945) | 1.08774 ± 0.00233541 (W 0.0031118) | - | · | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.34478 ± 0.0252069 (W 0.0146954) | 7.11922 ± 0.0367979 (W 0.00665719) | 8.76495 ± 0.0244482 (W 0.0144186) | · | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 7.49542 ± 0.0161859 (W 0.00973145) | 7.43306 ± 0.0329704 (W 0.0213844) | - | · | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.279232 ± 0.00215568 (W 0.00249409) | 0.276367 ± 0.000765677 (W 0.00261769) | 0.33124 ± 0.00152392 (W 0.00206514) | · | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.406079 ± 0.00135583 (W 0.00407176) | 0.398235 ± 0.00312905 (W 0.00335682) | - | · | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.87459 ± 0.00754369 (W 0.00557841) | 1.97238 ± 0.00420849 (W 0.00498216) | 2.2465 ± 0.00740239 (W 0.00585645) | 🔵 colmat 1.05× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 2.51049 ± 0.00239279 (W 0.0109474) | 2.54976 ± 0.00755341 (W 0.00983619) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.2178 ± 0.0497701 (W 0.020569) | 13.6086 ± 0.125901 (W 0.01722) | 15.8629 ± 0.0425858 (W 0.0273628) | · | 🔵 colmat 1.20× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 16.8219 ± 0.0838732 (W 0.0277447) | 16.8895 ± 0.0627496 (W 0.0190204) | - | · | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.294882 ± 0.0028284 (W 0.00297464) | 0.276063 ± 0.000283466 (W 0.00245728) | 0.34743 ± 0.00101212 (W 0.00333526) | 🔴 colmat 1.07× slower than `faer` | 🔵 colmat 1.18× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.414765 ± 0.00241111 (W 0.00495752) | 0.396287 ± 0.00163099 (W 0.00269912) | - | · | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.89984 ± 0.00282427 (W 0.00228346) | 1.8849 ± 0.00814668 (W 0.00458356) | 2.34979 ± 0.00927904 (W 0.00345855) | · | 🔵 colmat 1.24× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.95596 ± 0.0063783 (W 0.0047592) | 1.90946 ± 0.00576965 (W 0.00253887) | - | · | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.3008 ± 0.0259092 (W 0.0175992) | 13.3932 ± 0.0578093 (W 0.0265614) | 16.2831 ± 0.0475044 (W 0.0303328) | · | 🔵 colmat 1.22× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 13.8653 ± 0.0804579 (W 0.0311262) | 13.7449 ± 0.0410138 (W 0.0143531) | - | · | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.10701 ± 0.000224923 (W 0.000210241) | 0.152214 ± 0.000739215 (W 0.00139099) | 0.100889 ± 0.000124565 (W 0.00155377) | 🔵 colmat 1.42× faster than `faer` | 🔴 colmat 1.06× slower than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.124309 ± 0.000893792 (W 0.00023019) | 0.159216 ± 0.00144604 (W 0.00129675) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.432836 ± 0.00245474 (W 0.00288572) | 0.63151 ± 0.00308322 (W 0.00673457) | 0.500271 ± 0.00131674 (W 0.00337939) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.514702 ± 0.00109264 (W 0.0036032) | 0.675528 ± 0.00229058 (W 0.00631442) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.14772 ± 0.0077684 (W 0.00549414) | 2.6911 ± 0.0911675 (W 0.128638) | 3.13879 ± 0.00619845 (W 0.006552) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-openblas` | `!` between-start range over 5% |
| `cholesky/n512/b1 [api-inclusive]` | 2.50917 ± 0.00657139 (W 0.00414309) | 2.91687 ± 0.0186413 (W 0.138367) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.16686 ± 0.00155191 (W 0.00161708) | 0.253351 ± 0.00408318 (W 0.00470243) | 0.215504 ± 0.000831472 (W 0.00195944) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.293845 ± 0.00298269 (W 0.00317852) | 0.27286 ± 0.0020664 (W 0.0034768) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.748819 ± 0.00384502 (W 0.00264923) | 1.27525 ± 0.00546485 (W 0.0147849) | 1.17039 ± 0.00533217 (W 0.00383349) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.40505 ± 0.0113397 (W 0.00458227) | 1.34069 ± 0.0076073 (W 0.0146944) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.14645 ± 0.0111884 (W 0.00970437) | 5.98458 ± 0.025925 (W 0.105835) | 5.10913 ± 0.00566325 (W 0.00739124) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `lu/n512/b1 [api-inclusive]` | 7.67416 ± 0.041762 (W 0.0178733) | 6.36513 ± 0.0735557 (W 0.107177) | - | 🔴 colmat 1.21× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.167662 ± 0.000637743 (W 0.000530092) | 0.243465 ± 0.00166337 (W 0.00404291) | 0.208668 ± 0.000998471 (W 0.00210185) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.295217 ± 0.00412955 (W 0.00319199) | 0.266489 ± 0.00267089 (W 0.00441805) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.737863 ± 0.00197163 (W 0.00286145) | 1.1734 ± 0.0138255 (W 0.0126019) | 1.12673 ± 0.00286978 (W 0.00387366) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 1.53× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 1.39693 ± 0.0115093 (W 0.00366738) | 1.21323 ± 0.0105235 (W 0.0131884) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.07745 ± 0.0076952 (W 0.00824796) | 5.76696 ± 0.0734141 (W 0.12585) | 5.01637 ± 0.0239431 (W 0.0177435) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 7.62268 ± 0.0684846 (W 0.0284533) | 6.25064 ± 0.134618 (W 0.111641) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n128/b1` | 0.111608 ± 0.000372694 (W 0.000300667) | 0.137186 ± 0.000158639 (W 0.00167704) | 0.133236 ± 0.0010798 (W 0.00158427) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.124057 ± 0.000607494 (W 0.000275392) | 0.144239 ± 0.000902154 (W 0.00195464) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.368787 ± 0.000842397 (W 0.00228301) | 0.606324 ± 0.00403285 (W 0.0122989) | 0.446184 ± 0.00597207 (W 0.00259766) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-openblas` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.430962 ± 0.00168459 (W 0.00239928) | 0.657139 ± 0.0018763 (W 0.0102088) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.4756 ± 0.00208878 (W 0.00455942) | 2.01581 ± 0.0133959 (W 0.0157383) | 2.00387 ± 0.00339527 (W 0.00945612) | 🔵 colmat 1.37× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-openblas` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.78673 ± 0.00507058 (W 0.00219328) | 2.28528 ± 0.0139703 (W 0.0528547) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.146461 ± 0.00091572 (W 0.000469791) | 0.241653 ± 0.00256314 (W 0.00201505) | 0.202247 ± 0.00251347 (W 0.00380714) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `lu/n128/b1 [api-inclusive]` | 0.275235 ± 0.000833282 (W 0.00241725) | 0.261056 ± 0.000776793 (W 0.00216759) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.608566 ± 0.0024596 (W 0.0031511) | 1.19065 ± 0.00562402 (W 0.0120115) | 0.794029 ± 0.00301537 (W 0.0072387) | 🔵 colmat 1.96× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-openblas` |  |
| `lu/n256/b1 [api-inclusive]` | 1.25757 ± 0.00410661 (W 0.00530161) | 1.26693 ± 0.0064488 (W 0.0132501) | - | · | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.78756 ± 0.0198424 (W 0.00873491) | 4.51486 ± 0.0264209 (W 0.0271817) | 2.83292 ± 0.00897594 (W 0.0278559) | 🔵 colmat 1.62× faster than `faer` | · |  |
| `lu/n512/b1 [api-inclusive]` | 6.23076 ± 0.0184227 (W 0.0167985) | 4.94449 ± 0.00960509 (W 0.0292947) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.142633 ± 0.00297369 (W 0.000427558) | 0.227443 ± 0.00071073 (W 0.00216638) | 0.199011 ± 0.00167774 (W 0.00228795) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-openblas` |  |
| `solve/n128/b1 [api-inclusive]` | 0.275778 ± 0.00258794 (W 0.00282975) | 0.2524 ± 0.000750686 (W 0.0018825) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.561654 ± 0.00211512 (W 0.00341041) | 1.00768 ± 0.0142642 (W 0.0120352) | 0.763903 ± 0.00326434 (W 0.00356789) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-openblas` |  |
| `solve/n256/b1 [api-inclusive]` | 0.69768 ± 0.00804152 (W 0.0033125) | 1.12264 ± 0.00478065 (W 0.0146571) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.65112 ± 0.00673361 (W 0.00408091) | 4.11146 ± 0.00530667 (W 0.029545) | 2.79715 ± 0.0184679 (W 0.00692314) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `solve/n512/b1 [api-inclusive]` | 6.20797 ± 0.0448734 (W 0.0247136) | 4.69638 ± 0.0243506 (W 0.0405741) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `lu/128`, `lu/256`, `lu/512`, `solve/128`, `solve/256`, `solve/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.322837 ± 0.00290488 (W 0.00328829) | 0.896547 ± 0.0067418 (W 0.00529063) | - | 🔵 colmat 2.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.333548 ± 0.00648696 (W 0.00261379) | 0.481841 ± 0.00337388 (W 0.00660195) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 12.9061 ± 0.0591768 (W 0.098242) | 40.1213 ± 0.237055 (W 0.0555324) | - | 🔵 colmat 3.11× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 13.5953 ± 0.183727 (W 0.094781) | 19.1372 ± 0.010335 (W 0.06262) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.254897 ± 0.00163945 (W 0.00280443) | 0.927178 ± 0.011799 (W 0.0213196) | - | 🔵 colmat 3.64× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.277687 ± 0.000740579 (W 0.0044996) | 0.436554 ± 0.00716397 (W 0.00269406) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.9135 ± 0.0101575 (W 0.0255462) | 33.1851 ± 0.341467 (W 0.13432) | - | 🔵 colmat 5.61× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 6.83637 ± 0.129061 (W 0.0385657) | 12.8744 ± 0.0246591 (W 0.0209447) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.227003 ± 0.00243086 (W 0.00336016) | 0.97455 ± 0.019873 (W 0.014673) | - | 🔵 colmat 4.29× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.275546 ± 0.00115689 (W 0.00169071) | 0.43982 ± 0.0024494 (W 0.00462924) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.26492 ± 0.0271414 (W 0.0139454) | 32.1907 ± 0.192889 (W 0.0886482) | - | 🔵 colmat 7.55× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.94325 ± 0.00231469 (W 0.0549316) | 11.8824 ± 0.0211436 (W 0.0274089) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.222405 ± 0.000930993 (W 0.00366188) | 0.375598 ± 0.00208111 (W 0.00427169) | 0.255047 ± 0.000519693 (W 0.00296969) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 1.15× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.237003 ± 0.00140897 (W 0.00283592) | 0.392732 ± 0.00268915 (W 0.00375504) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.8462 ± 0.0293099 (W 0.0235636) | 12.8641 ± 0.0548393 (W 0.0194033) | 12.0146 ± 0.0206568 (W 0.036521) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.11× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.248 ± 0.0141187 (W 0.0496773) | 13.4341 ± 0.0359748 (W 0.0260602) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.167741 ± 0.000622686 (W 0.00319938) | 0.333296 ± 0.00211941 (W 0.00322365) | 0.196294 ± 0.000151269 (W 0.00214846) | 🔵 colmat 1.99× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.192878 ± 0.000356922 (W 0.0022667) | 0.360945 ± 0.00193161 (W 0.00275938) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.94238 ± 0.0103858 (W 0.0275826) | 6.67693 ± 0.0198038 (W 0.0199212) | 6.21672 ± 0.0117111 (W 0.00996439) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 1.58× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.43161 ± 0.00696108 (W 0.00984954) | 7.19357 ± 0.0145379 (W 0.0353648) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.161519 ± 0.000521887 (W 0.000692265) | 0.329847 ± 0.0020091 (W 0.00227726) | 0.178553 ± 0.000870476 (W 0.00174722) | 🔵 colmat 2.04× faster than `faer` | 🔵 colmat 1.11× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.190757 ± 0.00136411 (W 0.00179076) | 0.359332 ± 0.00113947 (W 0.00467427) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.15602 ± 0.00156512 (W 0.00765296) | 5.78192 ± 0.0259148 (W 0.0237212) | 3.76076 ± 0.00224265 (W 0.00506711) | 🔵 colmat 1.83× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 3.56882 ± 0.0128445 (W 0.0087197) | 6.31685 ± 0.0225956 (W 0.0235694) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.940708 ± 0.00538414 (W 0.00283535) | 1.65354 ± 0.0441045 (W 0.018003) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1 [api-inclusive]` | 0.993621 ± 0.00709575 (W 0.00260476) | 1.03084 ± 0.00222637 (W 0.0030366) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 44.7213 ± 0.0916252 (W 0.0923673) | 90.5364 ± 0.66158 (W 0.292422) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 46.5814 ± 0.103211 (W 0.0609103) | 54.3225 ± 0.0608923 (W 0.0784455) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.564626 ± 0.00278339 (W 0.00304285) | 1.44025 ± 0.0135036 (W 0.0119952) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.634111 ± 0.00250432 (W 0.00192235) | 0.768071 ± 0.0192058 (W 0.00291557) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 17.4711 ± 0.0435471 (W 0.0458621) | 65.3067 ± 0.294288 (W 0.178049) | - | 🔵 colmat 3.74× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 20.2924 ± 0.347254 (W 0.327816) | 29.7741 ± 0.0545386 (W 0.0622711) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.505833 ± 0.000387095 (W 0.00206436) | 1.45121 ± 0.0321097 (W 0.0107671) | - | 🔵 colmat 2.87× faster than `faer` | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.600108 ± 0.00186797 (W 0.00241038) | 0.719977 ± 0.00378605 (W 0.00220723) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.629 ± 0.0196387 (W 0.0260162) | 61.4334 ± 0.211904 (W 0.267763) | - | 🔵 colmat 4.51× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 17.5716 ± 0.235069 (W 0.0951812) | 25.6244 ± 0.0501337 (W 0.0472081) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.837671 ± 0.00215856 (W 0.00189408) | 0.874121 ± 0.00136367 (W 0.00281353) | 0.848842 ± 0.00178343 (W 0.00134973) | · | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.861427 ± 0.0018137 (W 0.00245496) | 0.904536 ± 0.00439835 (W 0.00247774) | - | · | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 40.4648 ± 0.072733 (W 0.0308503) | 41.089 ± 0.0797045 (W 0.115897) | 45.2185 ± 0.0855426 (W 0.0760113) | · | 🔵 colmat 1.12× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 41.5737 ± 0.128149 (W 0.130028) | 42.119 ± 0.0874737 (W 0.0382199) | - | · | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.457887 ± 0.00257325 (W 0.00329504) | 0.607289 ± 0.000936954 (W 0.00336698) | 0.568676 ± 0.0015048 (W 0.00273495) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.515164 ± 0.00249526 (W 0.00304899) | 0.665641 ± 0.00208712 (W 0.00296745) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.6068 ± 0.0371583 (W 0.0441928) | 16.495 ± 0.0152182 (W 0.0523493) | 22.2177 ± 0.0379714 (W 0.0284841) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.63× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 14.7797 ± 0.0669254 (W 0.0566332) | 17.664 ± 0.0334144 (W 0.0391029) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/1024x64/b1` | 0.417697 ± 0.0021315 (W 0.00275184) | 0.560906 ± 0.000890534 (W 0.0031533) | 0.444458 ± 0.00437561 (W 0.0026332) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.06× faster than `lapack-openblas` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.483251 ± 0.000964965 (W 0.00292195) | 0.620964 ± 0.00219965 (W 0.00383772) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.4928 ± 0.0219991 (W 0.0114102) | 12.5072 ± 0.00377171 (W 0.0246159) | 12.505 ± 0.0704144 (W 0.0196147) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-openblas` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.6918 ± 0.00959531 (W 0.0231085) | 13.5759 ± 0.0971289 (W 0.0274301) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `lu/1024x64`, `lu/4096x256`, `solve/1024x64`, `solve/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 11.4377 ± 0.0116682 (W 0.014523) | 109.642 ± 0.259983 (W 0.332932) | - | 🔵 colmat 9.59× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 15.7499 ± 0.066709 (W 0.032836) | 18.1387 ± 0.121722 (W 0.0303124) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.269142 ± 0.000687544 (W 0.00269761) | 3.16821 ± 0.00649423 (W 0.0129777) | - | 🔵 colmat 11.77× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.418037 ± 0.00462414 (W 0.00331702) | 0.510603 ± 0.0029411 (W 0.00385987) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 3.44859 ± 0.0106048 (W 0.0075646) | 55.9482 ± 12.6683 (W 10.6519) | - | 🔵 colmat 16.22× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 6.24423 ± 0.0135488 (W 0.0446376) | 9.96474 ± 0.366021 (W 0.414782) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.122278 ± 0.00255467 (W 0.00205739) | 1.62078 ± 0.0114108 (W 0.373426) | - | 🔵 colmat 13.25× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.209058 ± 0.00199146 (W 0.00265623) | 0.379904 ± 0.0249028 (W 0.0220956) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 2.02232 ± 0.00655847 (W 0.0107721) | 31.1699 ± 0.128556 (W 5.13087) | - | 🔵 colmat 15.41× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 4.6875 ± 0.0357426 (W 0.0340021) | 7.92027 ± 0.0574978 (W 0.13563) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.096671 ± 0.00149056 (W 0.00196268) | 0.945764 ± 0.00397243 (W 0.0157302) | - | 🔵 colmat 9.78× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.178971 ± 0.00102114 (W 0.000463661) | 0.321382 ± 0.0018116 (W 0.00758708) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 12.6147 ± 0.0659543 (W 0.0856564) | 14.5599 ± 0.02675 (W 0.0207883) | 19.22 ± 0.0465301 (W 0.0741044) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 13.02 ± 0.0432346 (W 0.0301191) | 13.3326 ± 0.020123 (W 0.024553) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.306144 ± 0.00321668 (W 0.00285744) | 0.297114 ± 0.00125764 (W 0.00468263) | 0.661804 ± 0.00395553 (W 0.00303008) | · | 🔵 colmat 2.16× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.323042 ± 0.00179497 (W 0.00271442) | 0.310773 ± 0.000416908 (W 0.00296909) | - | · | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 3.29903 ± 0.0123315 (W 0.00896302) | 5.28165 ± 0.0090844 (W 0.364004) | 11.0633 ± 0.029854 (W 0.0258425) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 3.35× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 3.72602 ± 0.0146166 (W 0.0159356) | 5.36864 ± 0.283891 (W 0.300296) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.09669 ± 0.000631514 (W 0.000519467) | 0.135984 ± 0.00464609 (W 0.02188) | 0.440081 ± 0.000837473 (W 0.00288122) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 4.55× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.117383 ± 0.000274015 (W 0.000500137) | 0.167993 ± 0.00895896 (W 0.0211432) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 1.78553 ± 0.00770861 (W 0.00971745) | 3.26046 ± 0.0600129 (W 0.0705025) | 8.65463 ± 0.0388905 (W 0.0135543) | 🔵 colmat 1.83× faster than `faer` | 🔵 colmat 4.85× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 2.1717 ± 0.00258968 (W 0.0123969) | 3.29076 ± 0.0377286 (W 0.0981758) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.062198 ± 0.000277511 (W 0.000399733) | 0.11181 ± 0.000497148 (W 0.00339651) | 0.398734 ± 0.00451365 (W 0.00280207) | 🔵 colmat 1.80× faster than `faer` | 🔵 colmat 6.41× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.086812 ± 0.000545726 (W 0.00083608) | 0.136394 ± 0.0013079 (W 0.00429165) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 40.0264 ± 0.1111 (W 0.088816) | 138.447 ± 0.0788669 (W 0.136353) | - | 🔵 colmat 3.46× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 47.5203 ± 0.0751908 (W 0.212066) | 46.1074 ± 0.110017 (W 0.0517902) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.808803 ± 0.00180956 (W 0.00371635) | 3.7436 ± 0.0314563 (W 0.011513) | - | 🔵 colmat 4.63× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.999655 ± 0.00635587 (W 0.00740423) | 1.02185 ± 0.00189981 (W 0.00282054) | - | · | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 11.0105 ± 0.0134486 (W 0.0216542) | 65.1275 ± 14.0094 (W 10.3345) | - | 🔵 colmat 5.92× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 16.9001 ± 0.11425 (W 0.0963287) | 20.8996 ± 0.163076 (W 1.4025) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.268962 ± 0.000343052 (W 0.00206614) | 1.80268 ± 0.0196488 (W 0.342098) | - | 🔵 colmat 6.70× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.405282 ± 0.00262735 (W 0.0034665) | 0.487749 ± 0.004657 (W 0.0569704) | - | 🔵 colmat 1.20× faster than `faer` (marginal: spread 0.69 of the difference) | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 6.13228 ± 0.0113782 (W 0.0153187) | 35.9065 ± 0.869002 (W 5.27977) | - | 🔵 colmat 5.86× faster than `faer` | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 11.3119 ± 0.0141526 (W 0.0672197) | 13.6227 ± 0.253947 (W 0.445357) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.180388 ± 0.001266 (W 0.00129552) | 1.04799 ± 0.00565775 (W 0.0238868) | - | 🔵 colmat 5.81× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.329356 ± 0.00623059 (W 0.00155119) | 0.424278 ± 0.00299763 (W 0.0210019) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 42.1598 ± 0.0260088 (W 0.0741976) | 43.0121 ± 0.0547573 (W 0.067005) | 53.8665 ± 0.1002 (W 0.092112) | · | 🔵 colmat 1.28× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 43.2229 ± 0.0899486 (W 0.042703) | 41.6529 ± 0.158257 (W 0.0324662) | - | · | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.855896 ± 0.00548283 (W 0.00718401) | 0.838709 ± 0.00223131 (W 0.00434057) | 1.18513 ± 0.00316853 (W 0.00218979) | · | 🔵 colmat 1.38× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.890991 ± 0.00796084 (W 0.00944984) | 0.863756 ± 0.0018235 (W 0.00442065) | - | · | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 11.1841 ± 0.0517397 (W 0.015286) | 16.0892 ± 0.387924 (W 1.38607) | 28.6722 ± 0.031546 (W 0.0303366) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 2.56× faster than `lapack-openblas` |  |
| `lu/256x4096/b1 [api-inclusive]` | 12.422 ± 0.0533566 (W 0.0744987) | 16.5405 ± 0.797536 (W 1.44691) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.246961 ± 0.000713391 (W 0.00231888) | 0.352428 ± 0.0104702 (W 0.0504909) | 0.698126 ± 0.00494317 (W 0.00245299) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 2.83× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.287959 ± 0.00572558 (W 0.00301074) | 0.32527 ± 0.000723471 (W 0.0649632) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-openblas (ms) | vs `faer` | vs `lapack-openblas` | notes |
|---|---|---|---|---|---|---|
| `lu/256x4096/b1` | 5.93952 ± 0.0225133 (W 0.0157479) | 8.72157 ± 0.362942 (W 0.397153) | 21.7927 ± 0.0389739 (W 0.0349748) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 3.67× faster than `lapack-openblas` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 6.90505 ± 0.00390551 (W 0.0109847) | 9.46104 ± 0.135964 (W 0.309708) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.146954 ± 0.00121877 (W 0.0016573) | 0.218599 ± 0.00479021 (W 0.0226253) | 0.567622 ± 0.00145244 (W 0.00222334) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 3.86× faster than `lapack-openblas` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.197289 ± 0.000496102 (W 0.00174058) | 0.272879 ± 0.00198823 (W 0.0191075) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `lu/256x4096`, `lu/64x1024`, `solve/256x4096`, `solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
