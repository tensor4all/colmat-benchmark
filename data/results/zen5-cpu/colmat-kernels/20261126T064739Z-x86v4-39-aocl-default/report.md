# `colmat-kernels` on `zen5-cpu`

- colmat (fork, measured) commit: `be19a8001e37e5db55c083c90bd8ca10269d1ab5`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- colmat / faer rows: `colmat` (the provider in the measured revision; see library and upstream above); `faer` (the provider in the measured revision; see library and upstream above)
- harness commit: `a899f37cb62992709cb756c0aa0c93aa5785a922`
- toolkit commit: `82bff6ea4508e156b9e661550c1dc65ad662581c`
- measured binary SHA-256: `a486a3554e3b3618c9106b28675a573aa86cf2bf472cd16dd392147b7437c7f7`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-26T08:52:35.422621Z`
- timing policy: v2, median of 5 reps, priming 500 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu colmat-kernels --checkout /home/shinaoka/projects/tensor4all/colmat-rs-39-measured --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor aocl --families full_piv_lu,triangular_solve,qr_factor,qr,svd_values,svd_thin,eigh,eig,upgrade_factor,householder_apply --regimes linalg-small,linalg-mid,linalg-tall,linalg-wide --dtypes f64,c64 --threads 1,4,8 --aa 3 --no-publish --label x86v4-39-aocl-default`
- harness environment: `FBENCH_ORDER` unset, `FBENCH_PROFILE=zen5-cpu`
- raw data: `data/results/zen5-cpu/colmat-kernels/20261126T064739Z-x86v4-39-aocl-default/`

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
  - `lapack-aocl` — The same calls against AMD AOCL, the vendor-native LAPACK stack on AMD hosts: LAPACKE from AOCL-LAPACK (`libflame`), CBLAS from the threaded AOCL-BLAS (`libblis-mt`), linked from the versioned tree `AOCL_ROOT` names and loaded through the rpath the build records. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run. Measured on AMD profiles only.
    Identity: the AOCL build recorded in this run's manifest (`providers[aocl]`)
    Caveat: One AOCL call is governed by three per-thread counts (OpenMP, BLIS, libflame); the harness sets the budget through OpenMP and refuses the cell unless all three read back equal to it, before each row and after its timed loop. The manifest records both libraries' versions, the files the loader mapped, and the dispatched kernels (`blis=<config>, flame=<isa>`). Threading is vendor-owned, as for OpenBLAS and MKL.
    As measured: linked shared; aocl 6.0.0; kernel dispatched to `blis=zen5, flame=avx512dq`; openmp threading; config `AOCL-BLAS 6.0.0 Build 20261026; AOCL-LAPACK 6.0.0 Build 20261026, supports LAPACK 3.12.1 w/o Index-64 interfaces; mapped=/home/shinaoka/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libaoclutils.so:/home/shinaoka/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libblis-mt.so.6.0.0:/home/shinaoka/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libflame.so:/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`

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
Spread guide (computed from this run's own rows): per-engine between-start range median 0.82%, worst 76.89% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 106 of 3456 engine cells exceed 5% and carry `!`; median within-process relative SD 0.41%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.102334 ± 0.000504649 (W 0.00162486) | 0.107602 ± 0.000223523 (W 0.000228629) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.102791 ± 0.000266819 (W 0.000210725) | 0.108266 ± 0.000264388 (W 0.000207221) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.430351 ± 0.000943472 (W 0.00231219) | 0.534043 ± 0.00187996 (W 0.00318109) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.432826 ± 0.0140727 (W 0.00237709) | 0.535795 ± 0.00180564 (W 0.00191955) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n96/b1` | 1.05042 ± 0.00928056 (W 0.00207052) | 1.25292 ± 0.00222585 (W 0.00379994) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.07298 ± 0.00684156 (W 0.00603183) | 1.27794 ± 0.00344353 (W 0.00614638) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011982 ± 0.000190867 (W 7.51019e-05) | 0.008927 ± 8.10987e-05 (W 0.00131639) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015108 ± 5.56507e-05 (W 9.06217e-05) | 0.010379 ± 9.88956e-05 (W 0.00011327) | - | 🔴 colmat 1.46× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.051328 ± 0.000728214 (W 0.000401741) | 0.043872 ± 0.000366652 (W 0.000218573) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.060274 ± 0.000483869 (W 0.000232565) | 0.045946 ± 0.000162194 (W 0.0002862) | - | 🔴 colmat 1.31× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.14332 ± 0.00493249 (W 0.000696388) | 0.12942 ± 0.000780383 (W 0.00141779) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.160468 ± 0.00115255 (W 0.00263028) | 0.135103 ± 0.000297562 (W 0.00138857) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008396 ± 0.000119304 (W 7.42307e-05) | 0.008126 ± 4.07063e-05 (W 0.000165758) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008506 ± 6.9282e-05 (W 4.32435e-05) | 0.008155 ± 1.52753e-05 (W 3.70972e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.041228 ± 7.26659e-05 (W 0.000239049) | 0.040576 ± 4.68722e-05 (W 8.305e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.041308 ± 0.000168573 (W 0.000114295) | 0.040336 ± 0.000176353 (W 8.25845e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.122262 ± 0.00062117 (W 0.000308553) | 0.120885 ± 7.63151e-05 (W 0.00020828) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.122358 ± 0.000134508 (W 0.00218472) | 0.120724 ± 0.000188741 (W 0.0017788) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.031228 ± 0.000205986 (W 0.000214632) | 0.040366 ± 0.000535595 (W 0.000202831) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.020268 ± 9.92321e-05 (W 0.000129465) | 0.032461 ± 0.000437555 (W 0.000152053) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.107774 ± 0.000668406 (W 0.00189995) | 0.173506 ± 0.000654813 (W 0.00213163) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.078538 ± 4.88092e-05 (W 0.000424594) | 0.120086 ± 0.00162091 (W 0.00220307) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.234616 ± 0.000364752 (W 0.00289524) | 0.385989 ± 0.000514347 (W 0.00275139) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.224436 ± 0.000187228 (W 0.00329229) | 0.306361 ± 0.00307828 (W 0.00369637) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.019086 ± 0.000153962 (W 0.000139579) | 0.028533 ± 0.000405123 (W 0.000153103) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.068329 ± 0.000715885 (W 0.000135393) | 0.132256 ± 0.000953096 (W 0.00166049) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.140532 ± 9.00907e-05 (W 0.000301764) | 0.289179 ± 0.00101673 (W 0.00224946) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.153942 ± 0.00783688 (W 0.00211628) | 0.182857 ± 0.000673501 (W 0.00212354) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.155169 ± 0.000558663 (W 0.00224983) | 0.183545 ± 0.000177367 (W 0.00204404) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.665211 ± 0.00572923 (W 0.00319398) | 0.918201 ± 0.00424455 (W 0.00267838) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.664857 ± 0.00586475 (W 0.00275166) | 0.919812 ± 0.0012755 (W 0.00123755) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54462 ± 0.00627826 (W 0.00311973) | 2.16271 ± 0.00714874 (W 0.00371813) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55261 ± 0.00412566 (W 0.00581342) | 2.23855 ± 0.00387541 (W 0.0041861) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059351 ± 5.02427e-05 (W 0.00016011) | 0.140186 ± 0.00028156 (W 0.00175098) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059502 ± 0.000233082 (W 0.000207288) | 0.140864 ± 0.000910693 (W 0.00197951) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.210673 ± 0.000369314 (W 0.00186055) | 0.51562 ± 0.00374042 (W 0.00253626) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.211023 ± 0.000964684 (W 0.00223867) | 0.518163 ± 0.00226695 (W 0.00266461) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.486946 ± 0.00424343 (W 0.00212502) | 1.19538 ± 0.0025812 (W 0.00201461) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.485971 ± 0.00139225 (W 0.00339371) | 1.18572 ± 0.00283997 (W 0.00256167) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000781 ± 1.52753e-05 (W 3.05336e-05) | 0.000672 ± 1.57162e-05 (W 9.13849e-05) | 0.001012 ± 5.7735e-06 (W 4.15933e-05) | 🔴 colmat 1.16× slower than `faer` (marginal: spread 0.84 of the difference) | 🔵 colmat 1.30× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001763 ± 1.52753e-05 (W 4.69042e-05) | 0.001553 ± 1e-05 (W 8.39643e-05) | 0.001954 ± 5.7735e-06 (W 4.52769e-05) | 🔴 colmat 1.14× slower than `faer` | 🔵 colmat 1.11× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003176 ± 0.000101129 (W 8.27647e-05) | 0.002775 ± 2.64575e-05 (W 4.69042e-05) | 0.002996 ± 9.50438e-06 (W 3.73403e-05) | 🔴 colmat 1.14× slower than `faer` | 🔴 colmat 1.06× slower than `lapack-aocl` (marginal: spread 0.61 of the difference) | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.070381 ± 0.000130847 (W 0.000301215) | 0.107513 ± 0.000627139 (W 0.000348357) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.071194 ± 0.000216957 (W 0.000270145) | 0.107792 ± 0.000914079 (W 0.00152609) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.220623 ± 0.000542684 (W 0.00209026) | 0.530572 ± 0.00330315 (W 0.00290723) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.231343 ± 0.000899698 (W 0.00539799) | 0.533158 ± 0.00294481 (W 0.002001) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.509429 ± 0.00127816 (W 0.0039863) | 1.2343 ± 0.0015578 (W 0.00187305) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.509884 ± 0.00268063 (W 0.00273247) | 1.23357 ± 0.00119969 (W 0.00304219) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.012854 ± 9.50438e-05 (W 0.000181693) | 0.008856 ± 4.20357e-05 (W 3.305e-05) | - | 🔴 colmat 1.45× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.016321 ± 0.00015308 (W 0.000150566) | 0.01048 ± 0.000143493 (W 0.000114966) | - | 🔴 colmat 1.56× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.05328 ± 0.000512735 (W 0.00017211) | 0.043461 ± 0.000361327 (W 0.000103195) | - | 🔴 colmat 1.23× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.062266 ± 0.000356469 (W 0.000479428) | 0.046228 ± 6.26977e-05 (W 0.000258583) | - | 🔴 colmat 1.35× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.137045 ± 0.000763881 (W 0.000558778) | 0.129241 ± 0.000369503 (W 0.00189137) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.16352 ± 0.0011884 (W 0.00242991) | 0.134054 ± 0.000261909 (W 0.00254451) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.007033 ± 0.000119862 (W 0.00029696) | 0.008326 ± 5.22526e-05 (W 7.12741e-05) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006081 ± 6.48614e-05 (W 8.007e-05) | 0.008406 ± 6.55744e-05 (W 8.38224e-05) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.021049 ± 0.00039887 (W 0.000517809) | 0.020088 ± 0.000119039 (W 0.000403906) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.021029 ± 5.57704e-05 (W 0.000491997) | 0.018725 ± 0.000440216 (W 0.000209452) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.047058 ± 0.000192718 (W 0.000663321) | 0.045145 ± 0.000153298 (W 0.000575857) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.047249 ± 0.000277093 (W 0.00143823) | 0.043501 ± 0.000124531 (W 0.000306329) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.032742 ± 0.000161711 (W 0.000242943) | 0.054803 ± 0.000209834 (W 0.00180041) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.020358 ± 4.16333e-05 (W 0.000131226) | 0.046628 ± 0.000479604 (W 0.000599848) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.069561 ± 0.000210064 (W 0.000562703) | 0.250683 ± 0.0015358 (W 0.00318206) | - | 🔵 colmat 3.60× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.06975 ± 0.000180436 (W 0.000254035) | 0.188913 ± 0.00178362 (W 0.00211114) | - | 🔵 colmat 2.71× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.152125 ± 0.000938388 (W 0.000283338) | 0.50747 ± 0.00251221 (W 0.00319459) | - | 🔵 colmat 3.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.185851 ± 0.000166001 (W 0.00246261) | 0.401297 ± 0.000532175 (W 0.00300007) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.020428 ± 0.000394787 (W 0.000115672) | 0.04299 ± 0.000570984 (W 0.000591561) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.038822 ± 0.000208351 (W 0.000101431) | 0.207949 ± 0.000514495 (W 0.00259605) | - | 🔵 colmat 5.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.094706 ± 0.000171267 (W 0.00109018) | 0.42924 ± 0.00147628 (W 0.00321991) | - | 🔵 colmat 4.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.15551 ± 0.00120222 (W 0.00181477) | 0.181573 ± 0.000628869 (W 0.00206319) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.155453 ± 0.000376181 (W 0.00219036) | 0.184022 ± 0.00182994 (W 0.00220725) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.547535 ± 0.000127503 (W 0.00253351) | 0.908321 ± 0.000641028 (W 0.00208563) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.556486 ± 0.0169567 (W 0.00213412) | 0.908651 ± 0.000645961 (W 0.00181925) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.13321 ± 0.00494738 (W 0.00207489) | 2.09831 ± 0.00262832 (W 0.00182463) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.13366 ± 0.00195122 (W 0.00286918) | 2.10254 ± 0.00100356 (W 0.00389604) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.060041 ± 9.3618e-05 (W 0.000126825) | 0.139131 ± 0.000354576 (W 0.00146567) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060483 ± 0.000439365 (W 0.000155313) | 0.13931 ± 0.000518128 (W 0.00204387) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.216696 ± 0.000773309 (W 0.00189823) | 0.512009 ± 0.00362104 (W 0.00255803) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.22 ± 0.000524188 (W 0.00210578) | 0.515741 ± 0.00170728 (W 0.00392046) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.522538 ± 0.00123708 (W 0.00188203) | 1.17925 ± 0.00298253 (W 0.00353634) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.522767 ± 0.00232081 (W 0.00282988) | 1.17946 ± 0.00460247 (W 0.00329849) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001002 ± 2.3094e-05 (W 3.11448e-05) | 0.000691 ± 1e-05 (W 2.89517e-05) | 0.001012 ± 5.7735e-07 (W 3.91152e-05) | 🔴 colmat 1.45× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002204 ± 4.16333e-05 (W 6.49615e-05) | 0.001603 ± 3.51188e-05 (W 5.59464e-05) | 0.001944 ± 5.7735e-07 (W 4.14041e-05) | 🔴 colmat 1.37× slower than `faer` | 🔴 colmat 1.13× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003737 ± 3.51188e-05 (W 9.63328e-05) | 0.002785 ± 1.48436e-05 (W 5.61249e-05) | 0.002985 ± 6.08276e-06 (W 3.84864e-05) | 🔴 colmat 1.34× slower than `faer` | 🔴 colmat 1.25× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.065321 ± 0.000339 (W 0.000392391) | 0.10674 ± 0.000271766 (W 0.00173218) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.066083 ± 0.000521316 (W 0.000218733) | 0.109716 ± 0.0106123 (W 0.0021315) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.184457 ± 0.00227994 (W 0.00233676) | 0.531621 ± 0.00105937 (W 0.00258498) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.198634 ± 0.000345564 (W 0.00388831) | 0.534657 ± 0.00118579 (W 0.00247424) | - | 🔵 colmat 2.69× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.397993 ± 0.00444794 (W 0.00229307) | 1.23455 ± 0.000841393 (W 0.00150229) | - | 🔵 colmat 3.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.399802 ± 0.00307088 (W 0.00265329) | 1.2453 ± 0.00269607 (W 0.00257886) | - | 🔵 colmat 3.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.012794 ± 0.000200899 (W 0.000151024) | 0.008786 ± 4.16333e-05 (W 6.20661e-05) | - | 🔴 colmat 1.46× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.016381 ± 9.29157e-05 (W 0.000104725) | 0.01039 ± 5.72742e-05 (W 0.000100246) | - | 🔴 colmat 1.58× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.052499 ± 0.000263231 (W 0.000255383) | 0.043662 ± 0.000380907 (W 0.000107564) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.062187 ± 0.000672664 (W 0.000783886) | 0.046456 ± 0.000256925 (W 0.000434477) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.14451 ± 0.000625017 (W 0.00140209) | 0.129862 ± 0.000534102 (W 0.000401783) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.17141 ± 0.00106377 (W 0.00247304) | 0.13386 ± 0.000666154 (W 0.00186339) | - | 🔴 colmat 1.28× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006152 ± 3.4356e-05 (W 0.000251115) | 0.008416 ± 4.17971e-05 (W 9.15107e-05) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006151 ± 6.02246e-05 (W 7.14024e-05) | 0.008396 ± 1.1547e-05 (W 6.55301e-05) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.01654 ± 0.000345616 (W 0.00042858) | 0.021039 ± 0.000257935 (W 0.000469223) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.016571 ± 0.000214554 (W 0.00046787) | 0.019417 ± 0.000135769 (W 0.00061376) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.032731 ± 0.000140962 (W 0.000690725) | 0.056506 ± 0.000221886 (W 0.000940026) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.034154 ± 0.000362683 (W 0.000375028) | 0.056255 ± 0.000575912 (W 0.000744506) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.032751 ± 0.000451599 (W 0.000135449) | 0.062899 ± 0.000570813 (W 0.000667197) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.020549 ± 7.78267e-05 (W 0.000126638) | 0.054933 ± 0.000905901 (W 0.000466682) | - | 🔵 colmat 2.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.069399 ± 0.000265236 (W 0.000390533) | 0.290332 ± 0.00204495 (W 0.00218314) | - | 🔵 colmat 4.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.069942 ± 0.000575987 (W 0.000331332) | 0.234161 ± 0.00223521 (W 0.00150628) | - | 🔵 colmat 3.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.154599 ± 0.0009905 (W 0.00101812) | 0.595172 ± 0.00245096 (W 0.00451486) | - | 🔵 colmat 3.85× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.184714 ± 0.000930269 (W 0.00175868) | 0.501276 ± 0.00815387 (W 0.00501477) | - | 🔵 colmat 2.71× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.020568 ± 0.000252125 (W 0.000192788) | 0.051597 ± 0.00121221 (W 0.00129948) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.038713 ± 0.000518201 (W 0.000169777) | 0.253547 ± 0.00144375 (W 0.00355593) | - | 🔵 colmat 6.55× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.089216 ± 0.000487539 (W 0.0012724) | 0.509542 ± 0.00444828 (W 0.0027906) | - | 🔵 colmat 5.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.155563 ± 8.77857e-05 (W 0.000412771) | 0.181829 ± 0.00102029 (W 0.00190405) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.155451 ± 0.00042779 (W 0.000277143) | 0.184104 ± 0.011076 (W 0.00246774) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.543925 ± 0.00140881 (W 0.00312146) | 0.910477 ± 0.00207319 (W 0.00137512) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.545369 ± 0.00272779 (W 0.00208518) | 0.912729 ± 0.00248802 (W 0.00147483) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.13165 ± 0.00414591 (W 0.00173202) | 2.11841 ± 0.00373056 (W 0.00219428) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.14568 ± 0.00830236 (W 0.00271576) | 2.12152 ± 0.00362944 (W 0.00410554) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059952 ± 0.000618704 (W 0.000102811) | 0.139531 ± 0.000218088 (W 0.00212042) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060563 ± 0.000570123 (W 0.00010351) | 0.140364 ± 7.10094e-05 (W 0.00089603) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.217015 ± 0.000317843 (W 0.001852) | 0.515822 ± 0.000920097 (W 0.00294493) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.217877 ± 0.000554226 (W 0.0018815) | 0.518587 ± 0.00152266 (W 0.00247199) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.540126 ± 0.00226531 (W 0.00329661) | 1.17954 ± 0.00143716 (W 0.00165496) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.539548 ± 0.00210869 (W 0.00242402) | 1.19023 ± 0.00480237 (W 0.00275699) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000982 ± 1.48436e-05 (W 2.94958e-05) | 0.000681 ± 2e-05 (W 3.437e-05) | 0.001002 ± 0 (W 3.04959e-05) | 🔴 colmat 1.44× slower than `faer` | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002194 ± 1.73205e-05 (W 1.30384e-05) | 0.001553 ± 3.21455e-05 (W 3.87453e-05) | 0.001953 ± 5.50757e-06 (W 2.53239e-05) | 🔴 colmat 1.41× slower than `faer` | 🔴 colmat 1.12× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003737 ± 5.68624e-05 (W 4.86539e-05) | 0.002805 ± 3.4641e-05 (W 4.32435e-05) | 0.002986 ± 1.70392e-05 (W 3.86562e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.25× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.102164 ± 0.000385625 (W 0.000261624) | 0.107832 ± 0.000338837 (W 0.00358175) | 0.1063 ± 0.000588355 (W 0.00214638) | 🔵 colmat 1.06× faster than `faer` (marginal: spread 0.63 of the difference) | · |  |
| `eigh/n32/b1 [api-inclusive]` | 0.102632 ± 0.000276528 (W 0.00220773) | 0.108393 ± 0.00668313 (W 0.000296657) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.430408 ± 0.00285939 (W 0.00234439) | 0.530386 ± 4.82113e-05 (W 0.00275243) | 0.40664 ± 0.0018034 (W 0.00260279) | 🔵 colmat 1.23× faster than `faer` | 🔴 colmat 1.06× slower than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.431908 ± 0.00190504 (W 0.00179058) | 0.536997 ± 0.00101021 (W 0.00306583) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.04782 ± 0.00734351 (W 0.00321092) | 1.26035 ± 0.00555656 (W 0.0046311) | 0.967734 ± 0.00308065 (W 0.00109139) | 🔵 colmat 1.20× faster than `faer` | 🔴 colmat 1.08× slower than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.07269 ± 0.0052046 (W 0.00409554) | 1.2752 ± 0.00105646 (W 0.00381854) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.012353 ± 8.99463e-05 (W 7.51685e-05) | 0.008817 ± 7.13933e-05 (W 5.20404e-05) | 0.032863 ± 7.15844e-05 (W 7.91833e-05) | 🔴 colmat 1.40× slower than `faer` | 🔵 colmat 2.66× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013335 ± 4.45758e-05 (W 0.000130638) | 0.01029 ± 0.000119616 (W 0.000103292) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.052319 ± 0.000233778 (W 0.000317728) | 0.04321 ± 0.000867804 (W 0.000299283) | 0.646002 ± 0.0493795 (W 0.00368605) | 🔴 colmat 1.21× slower than `faer` | 🔵 colmat 12.35× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.053571 ± 0.000187128 (W 0.000344619) | 0.045096 ± 8.22456e-05 (W 0.000180855) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.145921 ± 0.00114469 (W 0.00266589) | 0.129761 ± 0.000312833 (W 0.000191613) | 2.62817 ± 0.0120246 (W 0.00496392) | 🔴 colmat 1.12× slower than `faer` | 🔵 colmat 18.01× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.148341 ± 0.000718616 (W 0.00241541) | 0.133688 ± 0.00084591 (W 0.0023101) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008566 ± 9.07377e-05 (W 0.000330475) | 0.008356 ± 4.99032e-05 (W 8.56055e-05) | 0.023344 ± 0.000159615 (W 0.000122801) | · | 🔵 colmat 2.73× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008366 ± 0.000104644 (W 2.48998e-05) | 0.008386 ± 5e-05 (W 8.44938e-05) | 0.008867 ± 5.2915e-05 (W 1.76409e-05) | · | 🔵 colmat 1.06× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.040848 ± 8.64176e-05 (W 0.000152697) | 0.040296 ± 0.000254193 (W 0.00167409) | 0.060593 ± 0.000283731 (W 0.000135356) | · | 🔵 colmat 1.48× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.040938 ± 5.55368e-05 (W 6.76114e-05) | 0.040187 ± 3.03535e-05 (W 0.00146251) | 0.05856 ± 0.000141331 (W 0.000187254) | · | 🔵 colmat 1.43× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.124265 ± 0.000386579 (W 0.00020452) | 0.122772 ± 0.000310713 (W 0.00144511) | 0.13321 ± 0.000221696 (W 0.00160688) | · | 🔵 colmat 1.07× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.124866 ± 0.000457039 (W 0.00016218) | 0.12343 ± 0.000704131 (W 0.000254147) | 0.205309 ± 0.00174489 (W 0.00210948) | · | 🔵 colmat 1.64× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.016532 ± 0.000101425 (W 5.58954e-05) | 0.029776 ± 0.000173782 (W 0.000119457) | 0.01619 ± 5.83124e-05 (W 9.03881e-05) | 🔵 colmat 1.80× faster than `faer` | · | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.018805 ± 0.000135031 (W 7.46947e-05) | 0.032261 ± 0.000187705 (W 0.000186797) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.065583 ± 0.000119926 (W 6.9751e-05) | 0.111419 ± 0.000970068 (W 0.000196059) | 0.099196 ± 0.00107084 (W 0.000298339) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.071205 ± 2.45832e-05 (W 0.000203298) | 0.117248 ± 0.000300604 (W 0.000341707) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.170209 ± 0.000429582 (W 0.00203659) | 0.2611 ± 0.00364619 (W 0.00350727) | 0.341406 ± 0.00222965 (W 0.00280135) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 2.01× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.209119 ± 0.000371976 (W 0.00358919) | 0.296767 ± 0.00048723 (W 0.000436255) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.008185 ± 7.22519e-05 (W 5.89067e-05) | 0.018535 ± 7.86787e-05 (W 0.000169075) | 0.007495 ± 3.00056e-05 (W 4.82701e-05) | 🔵 colmat 2.26× faster than `faer` | 🔴 colmat 1.09× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.030427 ± 0.000155242 (W 0.000117241) | 0.076325 ± 0.000242403 (W 0.000426166) | 0.041128 ± 0.000251786 (W 0.000107639) | 🔵 colmat 2.51× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.077147 ± 0.000457839 (W 0.000181649) | 0.165638 ± 0.000678695 (W 0.000344748) | 0.138002 ± 0.00192422 (W 0.00237109) | 🔵 colmat 2.15× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.156244 ± 0.000676266 (W 0.00166017) | 0.184891 ± 0.000960948 (W 0.00197284) | 0.169505 ± 0.000506033 (W 0.0017219) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.08× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.156314 ± 0.00104795 (W 0.00188651) | 0.187429 ± 0.00171128 (W 0.0020033) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.672992 ± 0.00133257 (W 0.00284585) | 0.96143 ± 0.00208979 (W 0.00453973) | 0.64456 ± 0.00197765 (W 0.00215555) | 🔵 colmat 1.43× faster than `faer` | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.674499 ± 0.0048338 (W 0.00203047) | 0.964338 ± 0.00263863 (W 0.000830511) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.56437 ± 0.00824867 (W 0.00536984) | 2.15648 ± 0.00942683 (W 0.00395612) | 1.52062 ± 0.00533961 (W 0.0048422) | 🔵 colmat 1.38× faster than `faer` | · |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.5723 ± 0.00557313 (W 0.00489049) | 2.23646 ± 0.00618933 (W 0.00487941) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059021 ± 0.000156387 (W 0.000273033) | 0.142235 ± 0.00113739 (W 0.00254056) | 0.120799 ± 3.43123e-05 (W 0.00190468) | 🔵 colmat 2.41× faster than `faer` | 🔵 colmat 2.05× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.059132 ± 0.000141799 (W 9.45637e-05) | 0.141934 ± 0.000382739 (W 0.00145021) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.212847 ± 0.00503551 (W 0.00267146) | 0.541969 ± 0.00203711 (W 0.00268364) | 0.240899 ± 0.00132066 (W 0.00232488) | 🔵 colmat 2.55× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.214106 ± 0.000758235 (W 0.00204319) | 0.538001 ± 0.00264903 (W 0.00280844) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.496031 ± 0.00255654 (W 0.00275405) | 1.2017 ± 0.00279461 (W 0.00241356) | 0.603854 ± 0.00633594 (W 0.00256373) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.498846 ± 0.00207074 (W 0.00263588) | 1.19695 ± 0.00512568 (W 0.00341863) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000801 ± 1.52753e-05 (W 6.21233e-05) | 0.000772 ± 6.08276e-06 (W 7.90203e-05) | 0.000782 ± 5.50757e-06 (W 4.08938e-05) | · | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001934 ± 6.08276e-06 (W 5.84825e-05) | 0.001854 ± 1.00167e-05 (W 0.000140877) | 0.001442 ± 5.7735e-06 (W 4.73001e-05) | · | 🔴 colmat 1.34× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003396 ± 4.72582e-05 (W 6.36891e-05) | 0.003256 ± 7.37111e-05 (W 5.45921e-05) | 0.002184 ± 2e-05 (W 2.88097e-05) | · | 🔴 colmat 1.55× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.070814 ± 0.000470501 (W 0.000229781) | 0.107553 ± 0.000642816 (W 0.0019872) | 0.105947 ± 0.000578823 (W 0.00160387) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 1.50× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.070912 ± 0.000263255 (W 0.000318846) | 0.107933 ± 0.000184356 (W 0.000310619) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.21877 ± 6.53376e-05 (W 0.00268739) | 0.528068 ± 0.00133137 (W 0.00252012) | 0.410666 ± 0.0013358 (W 0.00251512) | 🔵 colmat 2.41× faster than `faer` | 🔵 colmat 1.88× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.229563 ± 0.000488377 (W 0.00205232) | 0.532707 ± 0.00091542 (W 0.00208565) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.509183 ± 0.002566 (W 0.00302351) | 1.23361 ± 0.00220463 (W 0.00276185) | 0.974986 ± 0.00225916 (W 0.00163448) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 1.91× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.50827 ± 0.00244459 (W 0.00325683) | 1.23171 ± 0.000823918 (W 0.00192242) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013295 ± 7.23418e-05 (W 7.95883e-05) | 0.008997 ± 0.000122231 (W 7.73389e-05) | 0.032932 ± 0.000266513 (W 0.000128301) | 🔴 colmat 1.48× slower than `faer` | 🔵 colmat 2.48× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014738 ± 0.000230717 (W 0.000101239) | 0.010269 ± 0.000148863 (W 0.000110187) | - | 🔴 colmat 1.44× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.053972 ± 0.000298027 (W 0.000329802) | 0.043713 ± 0.000241374 (W 0.000200355) | 0.636155 ± 0.00725937 (W 0.00241325) | 🔴 colmat 1.23× slower than `faer` | 🔵 colmat 11.79× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.056116 ± 0.000355387 (W 0.000343275) | 0.045295 ± 0.000525958 (W 0.000389867) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.140285 ± 0.000667211 (W 0.00124752) | 0.129241 ± 0.000428998 (W 0.00178993) | 2.61714 ± 0.00321761 (W 0.00691956) | 🔴 colmat 1.09× slower than `faer` | 🔵 colmat 18.66× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.15334 ± 0.00142213 (W 0.002653) | 0.132037 ± 0.000388846 (W 0.00028475) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006372 ± 5.50757e-05 (W 0.000182352) | 0.008516 ± 4.09919e-05 (W 9.43626e-05) | 0.023173 ± 7.57188e-05 (W 0.000143029) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 3.64× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006392 ± 0.000113578 (W 0.000142581) | 0.008446 ± 2.25167e-05 (W 6.98155e-05) | 0.008686 ± 6.62596e-05 (W 1.94936e-05) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.019586 ± 0.000201098 (W 0.000221335) | 0.018915 ± 0.000160416 (W 0.00017405) | 0.062357 ± 0.00244212 (W 0.00172761) | · | 🔵 colmat 3.18× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.019627 ± 0.000303701 (W 0.000194436) | 0.019146 ± 5.7735e-06 (W 0.000423529) | 0.057497 ± 0.000222636 (W 0.000167118) | · | 🔵 colmat 2.93× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.044874 ± 7.02377e-05 (W 0.000308303) | 0.043342 ± 0.000216865 (W 0.000443593) | 0.084148 ± 0.000638862 (W 0.00144037) | · | 🔵 colmat 1.88× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.045215 ± 0.000121698 (W 0.00061057) | 0.043111 ± 0.000232681 (W 0.000214386) | 0.203915 ± 0.00058501 (W 0.00206478) | · | 🔵 colmat 4.51× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.016531 ± 8.02081e-05 (W 6.44531e-05) | 0.043431 ± 0.000301503 (W 0.00027947) | 0.016251 ± 0.00011845 (W 0.000103722) | 🔵 colmat 2.63× faster than `faer` | · | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.018906 ± 0.000304724 (W 0.00010807) | 0.046748 ± 0.00058197 (W 0.000642981) | - | 🔵 colmat 2.47× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.05855 ± 0.000298235 (W 0.000339219) | 0.179469 ± 0.00103294 (W 0.00219533) | 0.098303 ± 0.000315005 (W 0.00153312) | 🔵 colmat 3.07× faster than `faer` | 🔵 colmat 1.68× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.064119 ± 7.7938e-05 (W 0.0001756) | 0.183363 ± 0.00146622 (W 0.00232908) | - | 🔵 colmat 2.86× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.131336 ± 0.000706212 (W 0.00063236) | 0.358938 ± 0.00149325 (W 0.00350011) | 0.340163 ± 0.000789096 (W 0.00174449) | 🔵 colmat 2.73× faster than `faer` | 🔵 colmat 2.59× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.173816 ± 0.000277464 (W 0.00240596) | 0.39532 ± 0.00274887 (W 0.00358901) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.008085 ± 3.78594e-05 (W 7.04982e-05) | 0.032521 ± 0.000314878 (W 0.000364535) | 0.007364 ± 4.04145e-05 (W 4.07456e-05) | 🔵 colmat 4.02× faster than `faer` | 🔴 colmat 1.10× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.030036 ± 0.000207052 (W 0.000208497) | 0.14814 ± 0.000873399 (W 0.0015594) | 0.040866 ± 0.000727488 (W 0.000100648) | 🔵 colmat 4.93× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.076644 ± 0.000160309 (W 0.00013567) | 0.295147 ± 0.000889913 (W 0.00347599) | 0.136394 ± 4.21228e-05 (W 0.00199385) | 🔵 colmat 3.85× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.157543 ± 0.000607979 (W 0.00155683) | 0.183867 ± 6.48408e-05 (W 0.00260548) | 0.168118 ± 0.000510212 (W 0.00188656) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.157373 ± 0.000519116 (W 0.00138719) | 0.184413 ± 0.00388107 (W 0.00196096) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.550119 ± 0.0139512 (W 0.00359346) | 0.950059 ± 0.00163858 (W 0.00186636) | 0.636201 ± 0.0022408 (W 0.00358783) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.551437 ± 0.00210408 (W 0.00207962) | 0.952158 ± 0.00031038 (W 0.00190968) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.14469 ± 0.0145596 (W 0.00188765) | 2.09856 ± 0.00155633 (W 0.00272077) | 1.47954 ± 0.0041317 (W 0.00404101) | 🔵 colmat 1.83× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.14721 ± 0.0138073 (W 0.00215173) | 2.10051 ± 0.00196902 (W 0.00550331) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059731 ± 9.65971e-05 (W 0.00012781) | 0.140924 ± 0.000404747 (W 0.00265772) | 0.119235 ± 0.000364434 (W 0.000183612) | 🔵 colmat 2.36× faster than `faer` | 🔵 colmat 2.00× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.060032 ± 0.000505688 (W 0.000205561) | 0.141247 ± 0.000358137 (W 0.00362077) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.219491 ± 0.000171161 (W 0.00233214) | 0.532657 ± 0.00270738 (W 0.00277834) | 0.241807 ± 0.000273039 (W 0.00184076) | 🔵 colmat 2.43× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.223536 ± 0.00102945 (W 0.00234904) | 0.535608 ± 0.00222976 (W 0.0026237) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.532938 ± 0.00537765 (W 0.00251328) | 1.18185 ± 0.00119372 (W 0.0018355) | 0.598424 ± 0.00301328 (W 0.00300301) | 🔵 colmat 2.22× faster than `faer` | 🔵 colmat 1.12× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.530754 ± 0.00458767 (W 0.00230476) | 1.19371 ± 0.00734833 (W 0.00434424) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000982 ± 9.50438e-06 (W 2.7408e-05) | 0.000771 ± 2.08167e-05 (W 3.77531e-05) | 0.000781 ± 5.7735e-06 (W 3.36199e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.26× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002334 ± 1.1547e-05 (W 3.05663e-05) | 0.001874 ± 1.00167e-05 (W 5.69403e-05) | 0.001442 ± 5.50757e-06 (W 3.53596e-05) | 🔴 colmat 1.25× slower than `faer` | 🔴 colmat 1.62× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003987 ± 2.66521e-05 (W 6.01847e-05) | 0.003247 ± 5.89661e-05 (W 4.15933e-05) | 0.002184 ± 3.21455e-05 (W 4.26615e-05) | 🔴 colmat 1.23× slower than `faer` | 🔴 colmat 1.83× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.065483 ± 0.000341955 (W 0.000186194) | 0.107601 ± 0.00164979 (W 0.00169082) | 0.10632 ± 0.000575397 (W 0.000898794) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.62× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.065953 ± 0.000360651 (W 0.00162505) | 0.109076 ± 0.00576378 (W 0.00221894) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.184026 ± 0.00109784 (W 0.00160165) | 0.530572 ± 0.00142817 (W 0.002042) | 0.412557 ± 0.00224189 (W 0.00268455) | 🔵 colmat 2.88× faster than `faer` | 🔵 colmat 2.24× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.197297 ± 0.00146719 (W 0.00365338) | 0.535478 ± 0.00221551 (W 0.00299551) | - | 🔵 colmat 2.71× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.400324 ± 0.019734 (W 0.00338123) | 1.2396 ± 0.00302132 (W 0.0039737) | 0.97406 ± 0.00245029 (W 0.00137824) | 🔵 colmat 3.10× faster than `faer` | 🔵 colmat 2.43× faster than `lapack-aocl` | `!` between-start range over 5% |
| `eigh/n96/b1 [api-inclusive]` | 0.400136 ± 0.000348084 (W 0.00188466) | 1.24534 ± 0.00255925 (W 0.0028208) | - | 🔵 colmat 3.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013224 ± 9.46942e-05 (W 9.04223e-05) | 0.008877 ± 6.06658e-05 (W 0.00139647) | 0.032611 ± 8.544e-05 (W 0.000139356) | 🔴 colmat 1.49× slower than `faer` | 🔵 colmat 2.47× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014768 ± 7.17658e-05 (W 0.000108186) | 0.010099 ± 5.1316e-05 (W 4.89898e-05) | - | 🔴 colmat 1.46× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.053962 ± 2.95127e-05 (W 0.000225023) | 0.043772 ± 0.000162648 (W 0.000136928) | 0.647961 ± 0.0258678 (W 0.00311878) | 🔴 colmat 1.23× slower than `faer` | 🔵 colmat 12.01× faster than `lapack-aocl` | `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.056566 ± 0.000555592 (W 0.000216253) | 0.045646 ± 0.000257257 (W 0.000353805) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.146143 ± 0.00094661 (W 0.00162113) | 0.129574 ± 0.000688697 (W 0.00222505) | 2.62406 ± 0.00845802 (W 0.00424468) | 🔴 colmat 1.13× slower than `faer` | 🔵 colmat 17.96× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.157635 ± 0.00174368 (W 0.00198014) | 0.132347 ± 0.000529085 (W 0.00217973) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005821 ± 0.000166433 (W 9.74679e-05) | 0.008536 ± 5.50757e-05 (W 4.81664e-05) | 0.023123 ± 0.00012897 (W 0.000135127) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 3.97× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005931 ± 6.4291e-05 (W 0.000102241) | 0.008576 ± 4.04145e-05 (W 8.4476e-05) | 0.008736 ± 0.000106397 (W 2.51436e-05) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 1.47× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014447 ± 2.3094e-05 (W 0.000159937) | 0.019517 ± 0.00020108 (W 0.000366028) | 0.072976 ± 0.00264985 (W 0.00198125) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 5.05× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014748 ± 0.00019894 (W 0.000102948) | 0.020178 ± 0.000229122 (W 0.000365201) | 0.058519 ± 0.000679553 (W 0.000107304) | 🔵 colmat 1.37× faster than `faer` | 🔵 colmat 3.97× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.030146 ± 0.00012434 (W 0.000257421) | 0.054361 ± 0.00052569 (W 0.00068595) | 0.095789 ± 0.00114486 (W 0.00361939) | 🔵 colmat 1.80× faster than `faer` | 🔵 colmat 3.18× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.030867 ± 0.0001205 (W 0.000324563) | 0.054382 ± 0.00040927 (W 0.0012912) | 0.204301 ± 0.000352603 (W 0.00205285) | 🔵 colmat 1.76× faster than `faer` | 🔵 colmat 6.62× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.016491 ± 0.000172434 (W 4.65833e-05) | 0.050916 ± 0.000511851 (W 0.000607842) | 0.016 ± 5.62168e-05 (W 3.97492e-05) | 🔵 colmat 3.09× faster than `faer` | · | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.018735 ± 3e-05 (W 9.65401e-05) | 0.055253 ± 0.00050021 (W 0.00154337) | - | 🔵 colmat 2.95× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.057839 ± 0.000577884 (W 0.000348428) | 0.222786 ± 0.00200846 (W 0.00245196) | 0.098724 ± 0.000177279 (W 0.000253125) | 🔵 colmat 3.85× faster than `faer` | 🔵 colmat 1.71× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.063911 ± 0.00032902 (W 0.000300784) | 0.234738 ± 0.00275601 (W 0.00309532) | - | 🔵 colmat 3.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.126767 ± 0.000527801 (W 0.000586293) | 0.453765 ± 0.00599799 (W 0.00398579) | 0.340807 ± 0.000972888 (W 0.00331472) | 🔵 colmat 3.58× faster than `faer` | 🔵 colmat 2.69× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.172011 ± 0.000279504 (W 0.00300396) | 0.501144 ± 0.00272894 (W 0.00398044) | - | 🔵 colmat 2.91× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.008045 ± 9.03715e-05 (W 9.5551e-05) | 0.041098 ± 0.000381158 (W 0.000973485) | 0.007424 ± 4.3589e-05 (W 4.09414e-05) | 🔵 colmat 5.11× faster than `faer` | 🔴 colmat 1.08× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.030457 ± 0.000172434 (W 0.000130843) | 0.194474 ± 0.0016301 (W 0.00344648) | 0.041418 ± 0.000210801 (W 0.000195198) | 🔵 colmat 6.39× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.076793 ± 0.000152844 (W 0.000186631) | 0.380902 ± 0.00106227 (W 0.00269899) | 0.136625 ± 0.000354636 (W 0.00196508) | 🔵 colmat 4.96× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.158228 ± 0.00172894 (W 0.000266121) | 0.184755 ± 0.0136614 (W 0.00200851) | 0.169099 ± 0.000547929 (W 0.00122241) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.157306 ± 0.000503085 (W 0.00079491) | 0.184594 ± 0.00178631 (W 0.00203525) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.548866 ± 0.0191486 (W 0.00217867) | 0.955039 ± 0.000819771 (W 0.00142808) | 0.637679 ± 0.00499555 (W 0.00629235) | 🔵 colmat 1.74× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/n64/b1 [api-inclusive]` | 0.555667 ± 0.00425765 (W 0.00333261) | 0.957078 ± 0.00104824 (W 0.00148481) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.14148 ± 0.00369303 (W 0.00492879) | 2.11015 ± 0.0015702 (W 0.00230673) | 1.50009 ± 0.0020055 (W 0.00667267) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15422 ± 0.00392671 (W 0.00390396) | 2.11838 ± 0.00332381 (W 0.00155333) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059351 ± 0.000479074 (W 0.000100426) | 0.142007 ± 0.00103403 (W 0.000335731) | 0.119846 ± 0.000368717 (W 0.0015245) | 🔵 colmat 2.39× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.060373 ± 0.000173177 (W 0.000323976) | 0.141424 ± 0.000479153 (W 0.00148202) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.220621 ± 0.000683356 (W 0.00139503) | 0.53643 ± 0.000910534 (W 0.00224589) | 0.239922 ± 0.0030085 (W 0.00225134) | 🔵 colmat 2.43× faster than `faer` | 🔵 colmat 1.09× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.220932 ± 0.00118716 (W 0.0018656) | 0.537312 ± 0.00251122 (W 0.00304211) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.55131 ± 0.00197797 (W 0.00233734) | 1.18562 ± 0.00123692 (W 0.00260921) | 0.59982 ± 0.00298787 (W 0.00274424) | 🔵 colmat 2.15× faster than `faer` | 🔵 colmat 1.09× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.548603 ± 0.00141944 (W 0.00222071) | 1.19559 ± 0.00602938 (W 0.00333388) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000981 ± 1.0504e-05 (W 2.88479e-05) | 0.000771 ± 5.7735e-06 (W 2.78244e-05) | 0.000772 ± 1.00167e-05 (W 4.18485e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.27× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002334 ± 3.80832e-05 (W 2.32013e-05) | 0.001874 ± 3.05505e-05 (W 4.74573e-05) | 0.001403 ± 1.1547e-05 (W 3.28055e-05) | 🔴 colmat 1.25× slower than `faer` | 🔴 colmat 1.66× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003948 ± 1.70392e-05 (W 2.83425e-05) | 0.003256 ± 4.56107e-05 (W 5.1672e-05) | 0.002144 ± 5.7735e-06 (W 2.91548e-05) | 🔴 colmat 1.21× slower than `faer` | 🔴 colmat 1.84× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.119956 ± 0.00107657 (W 0.00206341) | 0.136214 ± 0.00143902 (W 0.000344771) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.120328 ± 0.000353658 (W 0.00190513) | 0.136327 ± 0.000326028 (W 0.0018214) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.592353 ± 0.00236541 (W 0.00327496) | 0.698251 ± 0.000215987 (W 0.00223409) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.594196 ± 0.00287935 (W 0.00293243) | 0.715821 ± 0.00194314 (W 0.00445116) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.49357 ± 0.00781301 (W 0.00307331) | 1.71125 ± 0.00475001 (W 0.00843291) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.49884 ± 0.00318398 (W 0.00287593) | 1.7056 ± 0.00147143 (W 0.00446347) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017834 ± 0.000664255 (W 7.39594e-05) | 0.014156 ± 8.55005e-05 (W 8.65921e-05) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.022502 ± 0.00018037 (W 0.000239354) | 0.01579 ± 0.000117758 (W 9.02774e-05) | - | 🔴 colmat 1.43× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.097932 ± 0.00042015 (W 0.00175329) | 0.087714 ± 0.000720208 (W 0.000383709) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.109666 ± 0.000615761 (W 0.000444362) | 0.09164 ± 0.000149603 (W 0.000473543) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.284368 ± 0.00168364 (W 0.00187694) | 0.266614 ± 0.00173443 (W 0.0022865) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.354044 ± 0.00335986 (W 0.00297592) | 0.312282 ± 0.00381786 (W 0.00408264) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028424 ± 8.48548e-05 (W 0.00152866) | 0.027942 ± 4.20357e-05 (W 0.00012518) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028413 ± 6.65833e-05 (W 8.05307e-05) | 0.027822 ± 3.45977e-05 (W 5.44977e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.160729 ± 0.000338409 (W 0.00161265) | 0.159121 ± 0.000497981 (W 0.00245865) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.160691 ± 0.000320737 (W 0.000306262) | 0.157885 ± 4.7606e-05 (W 0.00192208) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.469435 ± 0.00172501 (W 0.00267671) | 0.47068 ± 0.00534474 (W 0.00241151) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.469324 ± 0.00265092 (W 0.000843339) | 0.466669 ± 0.00382229 (W 0.00436819) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.068448 ± 0.000226593 (W 0.000334553) | 0.078457 ± 0.000375758 (W 0.00240434) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.036709 ± 0.00020265 (W 7.95192e-05) | 0.069231 ± 0.000388 (W 0.000444992) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.295089 ± 0.0011973 (W 0.00208004) | 0.392974 ± 0.000842402 (W 0.00256571) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.260252 ± 0.00367116 (W 0.00301659) | 0.336391 ± 0.000963479 (W 0.00342299) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.716484 ± 0.00221806 (W 0.0022818) | 0.912427 ± 0.00450382 (W 0.00281186) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.705684 ± 0.00442435 (W 0.00439735) | 0.859721 ± 0.00217383 (W 0.00134362) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.035927 ± 0.000171001 (W 0.000176766) | 0.046157 ± 0.000193745 (W 0.000223793) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.147717 ± 0.000590315 (W 0.00109118) | 0.238868 ± 0.000388352 (W 0.00225498) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.348544 ± 0.00168348 (W 0.0022748) | 0.538853 ± 0.00433176 (W 0.00314495) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.197994 ± 0.000855686 (W 0.00243055) | 0.233906 ± 0.00197695 (W 0.0018936) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.198997 ± 0.000759347 (W 0.00252464) | 0.233929 ± 0.0166907 (W 0.0018725) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.999621 ± 0.00756709 (W 0.00177161) | 1.21341 ± 0.00340649 (W 0.00215962) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.994892 ± 0.00288999 (W 0.00134778) | 1.22647 ± 0.0045048 (W 0.00317673) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.41964 ± 0.00855804 (W 0.00468989) | 3.1111 ± 0.0121013 (W 0.00645723) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.43041 ± 0.0103078 (W 0.00715276) | 3.13845 ± 0.0128064 (W 0.00843346) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.07411 ± 0.00035003 (W 0.00012502) | 0.159542 ± 0.000641424 (W 0.00188676) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.07496 ± 0.00045559 (W 0.000118854) | 0.161451 ± 0.00113502 (W 0.000149562) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.310559 ± 0.003912 (W 0.00417566) | 0.650351 ± 0.0015738 (W 0.00264007) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.311921 ± 0.00198062 (W 0.00277298) | 0.648688 ± 0.00314205 (W 0.00247534) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.788869 ± 0.00183228 (W 0.00207545) | 1.49554 ± 0.0028946 (W 0.00670765) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.788689 ± 0.000746327 (W 0.00365314) | 1.50241 ± 0.00309544 (W 0.00380212) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001212 ± 1.67432e-05 (W 3.55556e-05) | 0.001072 ± 1.73205e-05 (W 3.39116e-05) | 0.001333 ± 1.57162e-05 (W 3.40632e-05) | 🔴 colmat 1.13× slower than `faer` | 🔵 colmat 1.10× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002975 ± 1.76163e-05 (W 8.25058e-05) | 0.002725 ± 4.16333e-05 (W 0.000111221) | 0.002855 ± 2.12211e-05 (W 4.41396e-05) | 🔴 colmat 1.09× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00531 ± 1.73205e-05 (W 6.98656e-05) | 0.004989 ± 5.08953e-05 (W 6.62442e-05) | 0.004769 ± 2.64575e-05 (W 8.29458e-05) | 🔴 colmat 1.06× slower than `faer` | 🔴 colmat 1.11× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.088878 ± 4.50333e-05 (W 0.00104464) | 0.136267 ± 0.000106782 (W 0.00227788) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.089619 ± 0.000358573 (W 0.000223114) | 0.138418 ± 0.000851954 (W 0.00218008) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.307425 ± 0.00121559 (W 0.00224372) | 0.67087 ± 0.000228199 (W 0.00223072) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.307623 ± 0.00162979 (W 0.00217814) | 0.670143 ± 0.00101409 (W 0.00267275) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.662795 ± 0.000751471 (W 0.00267346) | 1.54873 ± 0.00486335 (W 0.00280748) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.665691 ± 0.00203884 (W 0.00269713) | 1.54604 ± 0.000672339 (W 0.00270306) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019797 ± 0.000538016 (W 0.000207281) | 0.014347 ± 0.000244759 (W 0.00018628) | - | 🔴 colmat 1.38× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.023735 ± 0.000174261 (W 0.000142302) | 0.0157 ± 0.000118773 (W 0.000127386) | - | 🔴 colmat 1.51× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.090709 ± 0.000704096 (W 0.000540507) | 0.087745 ± 0.000220742 (W 0.000342684) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.104485 ± 0.00126652 (W 0.00177098) | 0.092034 ± 0.000423028 (W 0.000307119) | - | 🔴 colmat 1.14× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.200984 ± 0.000718808 (W 0.00282952) | 0.264289 ± 0.000267115 (W 0.00275084) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.285605 ± 0.00113552 (W 0.00376649) | 0.266453 ± 0.000568165 (W 0.00232843) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.015298 ± 6.86319e-05 (W 0.000207918) | 0.028313 ± 8.0829e-05 (W 0.000192599) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.014747 ± 6.84617e-05 (W 0.000214166) | 0.028294 ± 5.53203e-05 (W 0.000176055) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.062428 ± 0.000281479 (W 0.000262907) | 0.061204 ± 0.000620511 (W 0.000596546) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.062347 ± 0.000340783 (W 0.000324795) | 0.060623 ± 0.000217167 (W 0.000406812) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.154528 ± 0.000430279 (W 0.00202631) | 0.151515 ± 5.77264e-05 (W 0.00142645) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.154912 ± 0.000315235 (W 0.00122069) | 0.151576 ± 0.000310052 (W 0.000936801) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.065333 ± 0.000168049 (W 0.00192121) | 0.092314 ± 0.000230003 (W 0.000801283) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.03685 ± 0.000109911 (W 0.000103296) | 0.082385 ± 0.000606512 (W 0.000512391) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.152748 ± 0.00067879 (W 0.000445038) | 0.403321 ± 0.000623774 (W 0.00218483) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.182845 ± 0.00109651 (W 0.0017434) | 0.344289 ± 0.000445916 (W 0.003121) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.415433 ± 0.00103565 (W 0.00270608) | 0.768131 ± 0.0032481 (W 0.00294445) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.517115 ± 0.00332546 (W 0.00378517) | 0.704538 ± 0.00127584 (W 0.00409239) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.036819 ± 0.00043771 (W 0.000528705) | 0.060532 ± 9.55057e-05 (W 0.00171455) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.088385 ± 5.44181e-05 (W 0.00161454) | 0.292327 ± 0.000656491 (W 0.00152493) | - | 🔵 colmat 3.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.270971 ± 0.00146826 (W 0.00197273) | 0.574604 ± 0.0021479 (W 0.00315622) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.197519 ± 0.000204296 (W 0.00202737) | 0.231892 ± 0.000913846 (W 0.00169323) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.199998 ± 0.00200376 (W 0.00167842) | 0.232724 ± 0.000472214 (W 0.00214588) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.730286 ± 0.000757291 (W 0.00148634) | 1.14546 ± 0.00194923 (W 0.00231139) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.733097 ± 0.00289554 (W 0.00255488) | 1.1552 ± 0.00196622 (W 0.00342216) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54301 ± 0.00374709 (W 0.00500123) | 2.77478 ± 0.00113429 (W 0.00352516) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55701 ± 0.00515675 (W 0.00580653) | 2.79465 ± 0.00439847 (W 0.00464496) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075491 ± 0.000854963 (W 0.000129438) | 0.158707 ± 0.00215695 (W 0.00104657) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.076023 ± 0.00018921 (W 0.0016791) | 0.160039 ± 0.000563427 (W 0.00186891) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.315094 ± 0.000383448 (W 0.00252291) | 0.647307 ± 0.00552413 (W 0.00260717) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.316099 ± 0.00241179 (W 0.00275328) | 0.651539 ± 0.00279131 (W 0.00318649) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.729179 ± 0.00218035 (W 0.0029164) | 1.49462 ± 0.00585867 (W 0.00372003) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.724516 ± 0.00154216 (W 0.0023383) | 1.50096 ± 0.00535413 (W 0.00264933) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001433 ± 1.52753e-05 (W 3.53171e-05) | 0.001082 ± 1e-05 (W 4.32111e-05) | 0.001322 ± 2.04206e-05 (W 3.70176e-05) | 🔴 colmat 1.32× slower than `faer` | 🔴 colmat 1.08× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003466 ± 6.08276e-06 (W 3.51468e-05) | 0.002725 ± 5.7735e-06 (W 4.19428e-05) | 0.002815 ± 8.14064e-05 (W 2.54598e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.23× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006151 ± 6.55515e-05 (W 4.0978e-05) | 0.005029 ± 1.48436e-05 (W 7.80276e-05) | 0.004729 ± 5.0797e-05 (W 5.58328e-05) | 🔴 colmat 1.22× slower than `faer` | 🔴 colmat 1.30× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.084318 ± 0.000791051 (W 0.000204615) | 0.13924 ± 0.00254964 (W 0.00244034) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.083748 ± 0.000211851 (W 0.000183593) | 0.136103 ± 0.000453031 (W 0.000957403) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.26274 ± 0.00231004 (W 0.00257842) | 0.670922 ± 0.00131992 (W 0.00269745) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.265219 ± 0.0028679 (W 0.00174411) | 0.669699 ± 0.00202128 (W 0.00237343) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.530279 ± 0.00288456 (W 0.00183724) | 1.55265 ± 0.00705197 (W 0.00230392) | - | 🔵 colmat 2.93× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.538203 ± 0.00269426 (W 0.00204476) | 1.55091 ± 0.00624754 (W 0.00306716) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018946 ± 7.83603e-05 (W 0.000123288) | 0.014096 ± 0.000131074 (W 0.0020691) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.023865 ± 0.000175594 (W 0.000265105) | 0.0158 ± 6.82422e-05 (W 0.000270276) | - | 🔴 colmat 1.51× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.094857 ± 0.000418609 (W 0.0011913) | 0.087253 ± 0.000576623 (W 0.00194337) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.107821 ± 0.00051399 (W 0.000629487) | 0.092252 ± 9.56452e-05 (W 0.000431625) | - | 🔴 colmat 1.17× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.199984 ± 0.00224912 (W 0.00228277) | 0.26533 ± 0.000280134 (W 0.0025296) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.280789 ± 0.000371898 (W 0.00300246) | 0.266809 ± 0.00131564 (W 0.00265556) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.011352 ± 0.000182812 (W 0.000245653) | 0.028233 ± 0.000165025 (W 7.71362e-05) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.01098 ± 2.08167e-05 (W 0.000119324) | 0.028132 ± 0.000145935 (W 0.000109441) | - | 🔵 colmat 2.56× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.037661 ± 0.00622343 (W 0.000346138) | 0.060363 ± 0.00045235 (W 0.000394462) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.03735 ± 0.000525586 (W 0.00054867) | 0.060593 ± 0.000230804 (W 0.00206286) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.087123 ± 0.0107841 (W 0.0023176) | 0.151325 ± 0.0020634 (W 0.00230158) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.087623 ± 0.000409611 (W 0.000667843) | 0.151312 ± 0.000772959 (W 0.00190925) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.066835 ± 0.000346789 (W 0.000636179) | 0.099918 ± 0.000359459 (W 0.001552) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.03766 ± 0.000907544 (W 5.47385e-05) | 0.092392 ± 0.00103347 (W 0.0040963) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.143187 ± 0.000626668 (W 0.0019948) | 0.433027 ± 0.00726234 (W 0.00549702) | - | 🔵 colmat 3.02× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.173243 ± 0.00530327 (W 0.00370174) | 0.387803 ± 0.00167651 (W 0.00641364) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.350434 ± 0.00305978 (W 0.00309731) | 0.828675 ± 0.00166405 (W 0.00857777) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.470228 ± 0.00367275 (W 0.00146059) | 0.786716 ± 0.00299155 (W 0.00491167) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.038072 ± 0.000422339 (W 0.000902396) | 0.069229 ± 0.000429512 (W 0.000847426) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.088374 ± 0.000310095 (W 0.000217118) | 0.321375 ± 0.00566617 (W 0.00795148) | - | 🔵 colmat 3.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.244677 ± 0.00018921 (W 0.00210737) | 0.639153 ± 0.0127968 (W 0.0104201) | - | 🔵 colmat 2.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.198611 ± 0.0158674 (W 0.00151122) | 0.232935 ± 0.00233242 (W 0.00196814) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.197269 ± 0.000891422 (W 0.00208519) | 0.233776 ± 0.000163812 (W 0.00238418) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.716407 ± 0.00147983 (W 0.00362175) | 1.14852 ± 0.00202203 (W 0.00208742) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.717708 ± 0.0019844 (W 0.00318589) | 1.15692 ± 0.00298451 (W 0.00553576) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.48528 ± 0.00264644 (W 0.00371428) | 2.78773 ± 0.00446231 (W 0.00522192) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.48827 ± 0.00418436 (W 0.00270935) | 2.81348 ± 0.0107051 (W 0.00815109) | - | 🔵 colmat 1.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075761 ± 0.000186244 (W 0.000152414) | 0.159608 ± 0.000381828 (W 0.00173654) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.076012 ± 0.000460889 (W 0.000398691) | 0.159991 ± 0.00048945 (W 0.00296233) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.322061 ± 0.00223587 (W 0.00224657) | 0.650293 ± 0.00301986 (W 0.00287144) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.324515 ± 0.000149973 (W 0.00182952) | 0.652768 ± 0.00200472 (W 0.00246877) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.744511 ± 0.00175231 (W 0.00242558) | 1.50379 ± 0.0053562 (W 0.00365087) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.743459 ± 0.00811291 (W 0.0045342) | 1.50565 ± 0.00217176 (W 0.00310189) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001433 ± 2.3094e-05 (W 4.03733e-05) | 0.001102 ± 2.3094e-05 (W 9.95902e-05) | 0.001352 ± 6.08276e-06 (W 4.23356e-05) | 🔴 colmat 1.30× slower than `faer` | 🔴 colmat 1.06× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003486 ± 2.0502e-05 (W 8.98571e-05) | 0.002715 ± 5.7735e-06 (W 8.33319e-05) | 0.002855 ± 1.18462e-05 (W 3.39116e-05) | 🔴 colmat 1.28× slower than `faer` | 🔴 colmat 1.22× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006162 ± 5.05008e-05 (W 0.000100644) | 0.005049 ± 4.72582e-05 (W 6.86025e-05) | 0.004759 ± 3.60555e-05 (W 6.57267e-05) | 🔴 colmat 1.22× slower than `faer` | 🔴 colmat 1.29× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.119506 ± 0.000368234 (W 0.00175569) | 0.135767 ± 0.000250388 (W 0.00204141) | 0.154468 ± 0.00127211 (W 0.001294) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.119847 ± 0.000451194 (W 0.0013558) | 0.136408 ± 0.000601974 (W 0.00223448) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.591301 ± 0.000779874 (W 0.00256548) | 0.698692 ± 0.000995495 (W 0.00345755) | 0.714751 ± 0.00371761 (W 0.00254606) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.594948 ± 0.000720419 (W 0.00411821) | 0.715542 ± 0.0017107 (W 0.00355188) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.49789 ± 0.00491257 (W 0.00529556) | 1.71233 ± 0.00778267 (W 0.00505456) | 1.80291 ± 0.0047373 (W 0.00694448) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.49768 ± 0.00477091 (W 0.00371497) | 1.71346 ± 0.00413611 (W 0.00386549) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018044 ± 0.000135031 (W 7.21401e-05) | 0.014197 ± 5.63057e-05 (W 7.03228e-05) | 0.158129 ± 0.00356662 (W 0.00200716) | 🔴 colmat 1.27× slower than `faer` | 🔵 colmat 8.76× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.020177 ± 0.00180762 (W 0.000176744) | 0.01587 ± 5.56776e-05 (W 0.000101244) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1` | 0.098787 ± 0.000488121 (W 0.000210619) | 0.086985 ± 0.000706544 (W 0.000478259) | 2.708 ± 0.0056703 (W 0.00416443) | 🔴 colmat 1.14× slower than `faer` | 🔵 colmat 27.41× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.103015 ± 0.00108697 (W 0.00181365) | 0.089958 ± 0.000431806 (W 0.000341683) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.292755 ± 0.00408378 (W 0.00215332) | 0.266214 ± 0.00063268 (W 0.00237013) | 9.33408 ± 0.0358224 (W 0.0188287) | 🔴 colmat 1.10× slower than `faer` | 🔵 colmat 31.88× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.338421 ± 0.00128593 (W 0.0030738) | 0.310749 ± 0.00242298 (W 0.00340606) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028473 ± 0.000123923 (W 0.000129499) | 0.027832 ± 8.32666e-05 (W 0.000114461) | 0.041108 ± 0.000149534 (W 0.000101242) | · | 🔵 colmat 1.44× faster than `lapack-aocl` |  |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.016801 ± 0.000557425 (W 3.04959e-05) | 0.027812 ± 5.85947e-05 (W 0.0015142) | 0.020568 ± 0.0001006 (W 7.26787e-05) | 🔵 colmat 1.66× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.159717 ± 0.000280981 (W 0.00178655) | 0.156872 ± 0.000119943 (W 0.000276641) | 0.198415 ± 0.00108727 (W 0.0020972) | · | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.158696 ± 0.000396276 (W 0.00169797) | 0.156556 ± 0.000221136 (W 0.0023181) | 0.249862 ± 0.00226108 (W 0.00225814) | · | 🔵 colmat 1.57× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.462782 ± 0.00165366 (W 0.00271761) | 0.462722 ± 0.00168029 (W 0.00143614) | 0.474207 ± 0.000468199 (W 0.00384295) | · | · |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.466959 ± 0.00231642 (W 0.00282674) | 0.463937 ± 0.00346465 (W 0.00173189) | 0.536874 ± 0.00222475 (W 0.00326402) | · | 🔵 colmat 1.15× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.031309 ± 0.000321 (W 0.000123269) | 0.065903 ± 0.000307152 (W 0.000386174) | 0.041849 ± 6.67108e-05 (W 0.000166875) | 🔵 colmat 2.10× faster than `faer` | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.034455 ± 0.000110564 (W 0.000184552) | 0.068898 ± 0.000101336 (W 0.000303328) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.221846 ± 0.000519929 (W 0.00231687) | 0.30582 ± 0.00100009 (W 0.00260951) | 0.375441 ± 0.00188221 (W 0.00272517) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 1.69× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.253841 ± 0.00222597 (W 0.002122) | 0.33261 ± 0.00216413 (W 0.0033456) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.590728 ± 0.00159599 (W 0.00226549) | 0.756078 ± 0.00288712 (W 0.00213417) | 0.898501 ± 0.00774513 (W 0.00216634) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.689264 ± 0.00474147 (W 0.0031398) | 0.850239 ± 0.0018405 (W 0.00415245) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.014357 ± 1.76163e-05 (W 9.58394e-05) | 0.034464 ± 0.000471025 (W 0.000235584) | 0.020769 ± 9.49228e-05 (W 0.000243631) | 🔵 colmat 2.40× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.075994 ± 0.000354008 (W 9.14877e-05) | 0.162955 ± 0.000242551 (W 0.00249059) | 0.122339 ± 0.00248925 (W 0.00169036) | 🔵 colmat 2.14× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.222535 ± 0.00074852 (W 0.00191364) | 0.397731 ± 0.00108906 (W 0.00246907) | 0.358438 ± 0.000832395 (W 0.00288492) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.195384 ± 0.000680202 (W 0.0019231) | 0.229765 ± 0.00179243 (W 0.00225256) | 0.235846 ± 0.0010494 (W 0.00181675) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.21× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.196943 ± 0.000886325 (W 0.00231816) | 0.230031 ± 0.000941151 (W 0.00282935) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.979393 ± 0.00236556 (W 0.00306233) | 1.24941 ± 0.00356438 (W 0.00440753) | 0.981045 ± 0.008368 (W 0.000952587) | 🔵 colmat 1.28× faster than `faer` | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.982962 ± 0.00131299 (W 0.00283174) | 1.25497 ± 0.00321249 (W 0.00730736) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.41289 ± 0.0045544 (W 0.00762862) | 3.06876 ± 0.00838695 (W 0.00475342) | 2.62358 ± 0.0066278 (W 0.00336445) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 1.09× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.42548 ± 0.00625461 (W 0.00369775) | 3.09041 ± 0.0129684 (W 0.00869916) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075213 ± 0.000365645 (W 0.00026924) | 0.158059 ± 0.000723654 (W 0.00221119) | 0.109516 ± 0.0011343 (W 0.000378888) | 🔵 colmat 2.10× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.075561 ± 0.000220555 (W 0.000149874) | 0.159269 ± 0.000462081 (W 0.0019969) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.307346 ± 0.00135268 (W 0.0019943) | 0.62365 ± 0.0024615 (W 0.00263934) | 0.428541 ± 0.00365821 (W 0.00205922) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.309135 ± 0.00126236 (W 0.00217486) | 0.624753 ± 0.00414233 (W 0.00250463) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.783923 ± 0.00254088 (W 0.00189828) | 1.47584 ± 0.00856916 (W 0.00426609) | 1.02365 ± 0.00580177 (W 0.00165957) | 🔵 colmat 1.88× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.787911 ± 0.00544143 (W 0.00255967) | 1.48211 ± 0.00775652 (W 0.00694873) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001222 ± 1.1547e-05 (W 7.75887e-05) | 0.001252 ± 9.50438e-06 (W 4.89316e-05) | 0.001202 ± 5.7735e-06 (W 5.76194e-05) | · | · | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003196 ± 2.12211e-05 (W 4.86642e-05) | 0.003316 ± 1.73205e-05 (W 3.53723e-05) | 0.002565 ± 1.18462e-05 (W 8.12607e-05) | · | 🔴 colmat 1.25× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00532 ± 5.56776e-05 (W 8.26438e-05) | 0.00539 ± 5.19615e-05 (W 8.58487e-05) | 0.004388 ± 3.80832e-05 (W 3.74593e-05) | · | 🔴 colmat 1.21× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.089347 ± 0.00013156 (W 0.000161462) | 0.136935 ± 0.00127664 (W 0.000407534) | 0.154892 ± 0.00121865 (W 0.00204501) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.089567 ± 0.000137652 (W 0.000223945) | 0.137276 ± 0.00056701 (W 0.00282008) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.306481 ± 0.00149441 (W 0.00175605) | 0.671592 ± 0.00126856 (W 0.0022558) | 0.726289 ± 0.00259702 (W 0.00441399) | 🔵 colmat 2.19× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.309066 ± 0.000819196 (W 0.00215073) | 0.669653 ± 0.00208677 (W 0.00265049) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.66281 ± 0.00210698 (W 0.00273954) | 1.54431 ± 0.00160806 (W 0.00283732) | 1.83087 ± 0.00248416 (W 0.00534841) | 🔵 colmat 2.33× faster than `faer` | 🔵 colmat 2.76× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.666282 ± 0.00185668 (W 0.00224556) | 1.5487 ± 0.00429062 (W 0.00269699) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019517 ± 0.000185599 (W 0.000119457) | 0.014196 ± 5.50757e-06 (W 5.90957e-05) | 0.154848 ± 0.00552896 (W 0.00226093) | 🔴 colmat 1.37× slower than `faer` | 🔵 colmat 7.93× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.02158 ± 9.55353e-05 (W 0.000198958) | 0.015769 ± 0.000119626 (W 4.43847e-05) | - | 🔴 colmat 1.37× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.092273 ± 0.000623456 (W 0.00147409) | 0.088555 ± 0.000659174 (W 0.000366244) | 2.69641 ± 0.00826474 (W 0.00632646) | · | 🔵 colmat 29.22× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.097472 ± 0.000680038 (W 0.000890013) | 0.089417 ± 0.000236648 (W 0.00235017) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.20969 ± 0.00197597 (W 0.00306196) | 0.265825 ± 0.00129525 (W 0.00263174) | 9.24869 ± 0.147287 (W 0.00725928) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 44.11× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.2682 ± 0.0027856 (W 0.00418336) | 0.263928 ± 0.00176963 (W 0.00265671) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.015409 ± 4.6188e-05 (W 0.000186482) | 0.028103 ± 6.64605e-05 (W 0.000108707) | 0.065563 ± 0.000737155 (W 0.00147907) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 4.25× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.015218 ± 7.93725e-05 (W 9.09962e-05) | 0.028093 ± 3.21455e-05 (W 0.00010438) | 0.020428 ± 4.41626e-05 (W 8.52643e-05) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 1.34× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.062026 ± 7.45453e-05 (W 0.000275491) | 0.060714 ± 0.000110654 (W 0.000322495) | 0.132199 ± 0.00263134 (W 0.00229191) | · | 🔵 colmat 2.13× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.062007 ± 0.000238309 (W 0.000287952) | 0.060413 ± 0.000591496 (W 0.00038655) | 0.149502 ± 0.00107123 (W 0.0021402) | · | 🔵 colmat 2.41× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.152035 ± 0.000787147 (W 0.000735321) | 0.148006 ± 0.000902005 (W 0.00212795) | 0.214951 ± 0.00247601 (W 0.00364887) | · | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.150932 ± 0.000755862 (W 0.000641865) | 0.148607 ± 0.00040118 (W 0.000444453) | 0.259816 ± 0.00156211 (W 0.00388842) | · | 🔵 colmat 1.72× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.031509 ± 0.000340196 (W 6.25724e-05) | 0.078818 ± 0.000221353 (W 0.00173607) | 0.041297 ± 0.000135242 (W 0.00154604) | 🔵 colmat 2.50× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.035006 ± 0.000651325 (W 7.73065e-05) | 0.082485 ± 7.75263e-05 (W 0.000544983) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.13894 ± 0.000728512 (W 0.002092) | 0.310659 ± 0.00146616 (W 0.00202488) | 0.277627 ± 0.00467057 (W 0.00399922) | 🔵 colmat 2.24× faster than `faer` | 🔵 colmat 2.00× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.174586 ± 0.000776877 (W 0.00243511) | 0.34337 ± 0.00128233 (W 0.00229466) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.385677 ± 0.00180866 (W 0.00219762) | 0.595333 ± 0.00310162 (W 0.00225648) | 0.646625 ± 0.00135111 (W 0.00384406) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.68× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.499551 ± 0.00238879 (W 0.00278615) | 0.703796 ± 0.00209309 (W 0.00353123) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.014617 ± 0.000156195 (W 0.000109864) | 0.04784 ± 0.000678161 (W 0.000350591) | 0.020178 ± 0.000231733 (W 0.000115218) | 🔵 colmat 3.27× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.076522 ± 0.000256471 (W 9.9976e-05) | 0.210578 ± 0.000141677 (W 0.00314751) | 0.120554 ± 0.00015064 (W 0.00210731) | 🔵 colmat 2.75× faster than `faer` | 🔵 colmat 1.58× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.250433 ± 0.000363473 (W 0.00178699) | 0.427497 ± 0.00376121 (W 0.00414501) | 0.38062 ± 0.00198017 (W 0.00244807) | 🔵 colmat 1.71× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.196517 ± 0.00685048 (W 0.00185001) | 0.240133 ± 0.00663104 (W 0.00261324) | 0.233896 ± 0.000509729 (W 0.00256448) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.197672 ± 0.000633336 (W 0.00186663) | 0.231342 ± 0.00164555 (W 0.00248817) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.716922 ± 0.00291584 (W 0.0025045) | 1.17569 ± 0.000896212 (W 0.00143774) | 0.975478 ± 0.00318742 (W 0.0021342) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.719981 ± 0.00284406 (W 0.00257369) | 1.18543 ± 0.00188261 (W 0.00202811) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.5447 ± 0.00819942 (W 0.00446671) | 2.73092 ± 0.00327707 (W 0.00276537) | 2.19918 ± 0.00103457 (W 0.00672617) | 🔵 colmat 1.77× faster than `faer` | 🔵 colmat 1.42× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55619 ± 0.00234624 (W 0.00557823) | 2.75997 ± 0.00356867 (W 0.010335) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.076154 ± 0.000211014 (W 0.000121022) | 0.156692 ± 0.0005509 (W 0.00207918) | 0.108192 ± 9.96059e-05 (W 0.000336985) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.42× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.076343 ± 0.000112385 (W 0.00211462) | 0.157553 ± 0.000327872 (W 0.00234375) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.314207 ± 0.000833079 (W 0.00287098) | 0.623896 ± 0.00359439 (W 0.00285213) | 0.42365 ± 0.003425 (W 0.00344887) | 🔵 colmat 1.99× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.315448 ± 0.00201153 (W 0.00211869) | 0.624644 ± 0.00227137 (W 0.00221433) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.722666 ± 0.00340429 (W 0.00254266) | 1.47419 ± 0.00501712 (W 0.00451167) | 1.05667 ± 0.00255528 (W 0.00117072) | 🔵 colmat 2.04× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.722773 ± 0.000596601 (W 0.00401968) | 1.4827 ± 0.00400407 (W 0.00351826) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001472 ± 2.00083e-05 (W 2.71901e-05) | 0.001272 ± 1.73205e-05 (W 2.31991e-05) | 0.001182 ± 2.51661e-05 (W 4.72462e-05) | 🔴 colmat 1.16× slower than `faer` | 🔴 colmat 1.25× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003486 ± 2.51064e-05 (W 3.46006e-05) | 0.003056 ± 0 (W 3.43511e-05) | 0.002565 ± 1.52753e-05 (W 4.12311e-05) | 🔴 colmat 1.14× slower than `faer` | 🔴 colmat 1.36× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006412 ± 2.08167e-05 (W 4.79583e-05) | 0.005761 ± 1.1547e-05 (W 3.96232e-05) | 0.004329 ± 9.50438e-06 (W 3.96522e-05) | 🔴 colmat 1.11× slower than `faer` | 🔴 colmat 1.48× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.084297 ± 4.89932e-05 (W 0.000213866) | 0.137148 ± 0.00106611 (W 0.00221356) | 0.154438 ± 0.000886299 (W 0.0019738) | 🔵 colmat 1.63× faster than `faer` | 🔵 colmat 1.83× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.084838 ± 0.000822231 (W 0.000256555) | 0.136785 ± 0.00215777 (W 0.00206002) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.26217 ± 0.00232041 (W 0.00192321) | 0.671514 ± 0.00273663 (W 0.0024454) | 0.746303 ± 0.0089417 (W 0.00711633) | 🔵 colmat 2.56× faster than `faer` | 🔵 colmat 2.85× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.26274 ± 0.00201632 (W 0.00199226) | 0.670129 ± 0.00384327 (W 0.00215047) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.5366 ± 0.00118258 (W 0.00321209) | 1.54832 ± 0.00429392 (W 0.00720735) | 1.915 ± 0.0306353 (W 0.00749574) | 🔵 colmat 2.89× faster than `faer` | 🔵 colmat 3.57× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.537333 ± 0.00191419 (W 0.00417702) | 1.5509 ± 0.00482459 (W 0.00686089) | - | 🔵 colmat 2.89× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019637 ± 0.00023058 (W 0.000139463) | 0.014166 ± 7.83603e-05 (W 3.89872e-05) | 0.156423 ± 0.00121541 (W 0.0022944) | 🔴 colmat 1.39× slower than `faer` | 🔵 colmat 7.97× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.021781 ± 0.000310565 (W 0.000197991) | 0.01582 ± 8.17333e-05 (W 8.90971e-05) | - | 🔴 colmat 1.38× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.098023 ± 0.00118541 (W 0.000605406) | 0.087713 ± 0.000255736 (W 0.00239406) | 2.7293 ± 0.00693302 (W 0.0110072) | 🔴 colmat 1.12× slower than `faer` | 🔵 colmat 27.84× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.101369 ± 0.000294721 (W 0.000726292) | 0.089958 ± 8.66045e-05 (W 0.000378088) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.208409 ± 0.00401324 (W 0.00251566) | 0.265586 ± 0.000736349 (W 0.00214788) | 9.31307 ± 0.0371602 (W 0.0285228) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 44.69× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.264187 ± 0.002513 (W 0.00130863) | 0.263292 ± 0.000301341 (W 0.0037932) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.011571 ± 0.000119654 (W 0.000111355) | 0.027933 ± 0.000208847 (W 6.54614e-05) | 0.10098 ± 0.00105919 (W 0.0027415) | 🔵 colmat 2.41× faster than `faer` | 🔵 colmat 8.73× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.011582 ± 0.00010065 (W 0.000123563) | 0.028072 ± 0.000104484 (W 0.000155081) | 0.020909 ± 0.000287757 (W 6.80316e-05) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.038191 ± 4.87545e-05 (W 0.0003771) | 0.060975 ± 0.000462798 (W 0.000436888) | 0.154167 ± 0.0010247 (W 0.0034638) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 4.04× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.038181 ± 4.58258e-05 (W 0.000279558) | 0.060653 ± 0.000302866 (W 0.000201569) | 0.148066 ± 0.0013402 (W 0.00589591) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 3.88× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.086973 ± 0.000560073 (W 0.00234829) | 0.148888 ± 0.000354731 (W 0.001865) | 0.214413 ± 0.00262819 (W 0.00664556) | 🔵 colmat 1.71× faster than `faer` | 🔵 colmat 2.47× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.086854 ± 3.25781e-05 (W 0.000606921) | 0.148878 ± 0.000315644 (W 0.000510544) | 0.237777 ± 0.000977897 (W 0.00250864) | 🔵 colmat 1.71× faster than `faer` | 🔵 colmat 2.74× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.031499 ± 5.05008e-05 (W 7.51412e-05) | 0.086261 ± 0.000228081 (W 0.00152651) | 0.041778 ± 0.000147309 (W 0.000134796) | 🔵 colmat 2.74× faster than `faer` | 🔵 colmat 1.33× faster than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.034936 ± 0.000101199 (W 9.53063e-05) | 0.089939 ± 0.000912785 (W 0.000527971) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.129584 ± 0.00119482 (W 0.000348727) | 0.352837 ± 0.00127736 (W 0.00273545) | 0.27129 ± 0.00566908 (W 0.00446494) | 🔵 colmat 2.72× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.165909 ± 0.000349293 (W 0.00277354) | 0.388862 ± 0.00272935 (W 0.00516685) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.330272 ± 0.000207104 (W 0.0022613) | 0.667085 ± 0.00370835 (W 0.00487098) | 0.690167 ± 0.00672569 (W 0.00729751) | 🔵 colmat 2.02× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.457713 ± 0.000903032 (W 0.00324141) | 0.778445 ± 0.00550233 (W 0.00478781) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.014738 ± 0.000115889 (W 5.95919e-05) | 0.055363 ± 0.000769438 (W 0.00103199) | 0.020368 ± 0.000177305 (W 0.00011371) | 🔵 colmat 3.76× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.077785 ± 0.000733899 (W 0.000204182) | 0.251744 ± 0.00231401 (W 0.00290738) | 0.123311 ± 0.000535205 (W 0.00223123) | 🔵 colmat 3.24× faster than `faer` | 🔵 colmat 1.59× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.23047 ± 0.00182452 (W 0.00149569) | 0.499403 ± 0.000608847 (W 0.00349154) | 0.381959 ± 0.00114332 (W 0.002583) | 🔵 colmat 2.17× faster than `faer` | 🔵 colmat 1.66× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.193361 ± 0.00150605 (W 0.00184636) | 0.230684 ± 0.00173212 (W 0.0023726) | 0.235099 ± 0.00105646 (W 0.002691) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.196516 ± 0.00259977 (W 0.0022371) | 0.231121 ± 0.00016919 (W 0.00193537) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.701638 ± 0.0159827 (W 0.00498304) | 1.1774 ± 0.00205291 (W 0.00102981) | 0.983103 ± 0.00276265 (W 0.0016379) | 🔵 colmat 1.68× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.705276 ± 0.00318588 (W 0.00401189) | 1.18671 ± 0.00341391 (W 0.0035827) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.49416 ± 0.00174966 (W 0.0039079) | 2.74561 ± 0.00514434 (W 0.00461259) | 2.27529 ± 0.0168942 (W 0.0165815) | 🔵 colmat 1.84× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.49272 ± 0.00454477 (W 0.00472306) | 2.76753 ± 0.00174324 (W 0.00747467) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.076915 ± 0.000165543 (W 0.000130285) | 0.157273 ± 0.000346735 (W 0.00221094) | 0.108772 ± 0.000567903 (W 0.00239107) | 🔵 colmat 2.04× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.076624 ± 0.000410208 (W 0.000169178) | 0.157945 ± 0.000301631 (W 0.0023784) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.321324 ± 0.000874168 (W 0.00228671) | 0.623032 ± 0.001945 (W 0.00275347) | 0.425844 ± 0.0022206 (W 0.0026356) | 🔵 colmat 1.94× faster than `faer` | 🔵 colmat 1.33× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.325116 ± 0.00233302 (W 0.00164858) | 0.628291 ± 0.00210753 (W 0.00284677) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.748698 ± 0.000680243 (W 0.00304714) | 1.47288 ± 0.00419476 (W 0.00281896) | 1.05214 ± 0.00279031 (W 0.000844695) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.739218 ± 0.00229169 (W 0.00265836) | 1.48016 ± 0.00188703 (W 0.00345736) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001463 ± 5.7735e-06 (W 6.80441e-05) | 0.001262 ± 2.28108e-05 (W 2.86356e-05) | 0.001202 ± 5.7735e-06 (W 4.27925e-05) | 🔴 colmat 1.16× slower than `faer` | 🔴 colmat 1.22× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003506 ± 3.50761e-05 (W 4.55005e-05) | 0.003086 ± 2.88675e-05 (W 3.27109e-05) | 0.002545 ± 2.51661e-05 (W 9.39149e-05) | 🔴 colmat 1.14× slower than `faer` | 🔴 colmat 1.38× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006332 ± 2.64575e-05 (W 9.86925e-05) | 0.005731 ± 3.56417e-05 (W 3.87582e-05) | 0.004398 ± 4.16333e-05 (W 8.17313e-05) | 🔴 colmat 1.10× slower than `faer` | 🔴 colmat 1.44× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 2.06249 ± 0.00907745 (W 0.00314982) | 2.40809 ± 0.00448387 (W 0.00684879) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.06191 ± 0.00498441 (W 0.00309697) | 2.41031 ± 0.00325576 (W 0.00569438) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.65577 ± 0.0823294 (W 0.0549049) | 10.1585 ± 0.0519528 (W 0.0137468) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 9.64 ± 0.0421825 (W 0.0116175) | 10.1649 ± 0.068184 (W 0.011865) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 49.0808 ± 0.248413 (W 0.101517) | 49.3836 ± 0.151367 (W 0.044324) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 49.3852 ± 0.231535 (W 0.101287) | 49.7269 ± 0.655886 (W 0.0624728) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.342189 ± 0.00254427 (W 0.00240493) | 0.315354 ± 0.000959039 (W 0.00290664) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.319276 ± 0.000755483 (W 0.00190361) | 0.31102 ± 0.000433597 (W 0.00298165) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.05305 ± 0.0145544 (W 0.00682976) | 3.15035 ± 0.196823 (W 0.00759132) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.19919 ± 0.0151876 (W 0.00810504) | 2.9989 ± 0.00830341 (W 0.0103674) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 15.1918 ± 0.0299929 (W 0.0165784) | 24.7504 ± 0.10595 (W 0.0274279) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 16.1431 ± 0.0743891 (W 0.0429584) | 22.8797 ± 0.0841796 (W 0.0900522) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.259053 ± 0.00113063 (W 0.00280243) | 0.25727 ± 0.00100035 (W 0.000511243) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.258833 ± 0.00156405 (W 0.00214736) | 0.25771 ± 0.00277994 (W 0.00248313) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.90081 ± 0.00673317 (W 0.00497474) | 1.91158 ± 0.0159958 (W 0.00526761) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.8962 ± 0.0110581 (W 0.00314828) | 1.9171 ± 0.0153499 (W 0.00382418) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9 ± 0.0754266 (W 0.00992993) | 15.9948 ± 0.0753127 (W 0.0178551) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.9884 ± 0.0538525 (W 0.0149069) | 15.9721 ± 0.0512696 (W 0.0380721) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.449198 ± 0.00269633 (W 0.00346276) | 0.73387 ± 0.0103571 (W 0.00275557) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.470574 ± 0.00282639 (W 0.00301367) | 0.592064 ± 0.0043076 (W 0.00406462) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.90652 ± 0.0119567 (W 0.00606487) | 4.23968 ± 0.0213189 (W 0.00992485) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.21926 ± 0.00254988 (W 0.00474271) | 3.33926 ± 0.0111641 (W 0.00548828) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 18.7427 ± 0.0458927 (W 0.0146454) | 27.5416 ± 0.116389 (W 0.0340687) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 20.2193 ± 0.0379479 (W 0.386555) | 21.3909 ± 0.164497 (W 0.0552927) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.248143 ± 0.000315969 (W 0.00252856) | 0.53726 ± 0.00384357 (W 0.00311636) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.52091 ± 0.000537798 (W 0.00374459) | 2.90283 ± 0.00703573 (W 0.00626628) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 8.437 ± 0.0170799 (W 0.0127553) | 17.6356 ± 0.123343 (W 0.0322694) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.85676 ± 0.011577 (W 0.00813047) | 3.69156 ± 0.0129748 (W 0.00866025) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.84113 ± 0.00932442 (W 0.00329267) | 3.71823 ± 0.0326808 (W 0.00569864) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.4475 ± 0.0314397 (W 0.013281) | 18.4441 ± 0.0384453 (W 0.0254607) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 14.5349 ± 0.0958458 (W 0.0163295) | 18.4654 ± 0.0583476 (W 0.0299854) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 87.0268 ± 0.159999 (W 0.173305) | 105.193 ± 0.177453 (W 0.12217) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 87.0334 ± 0.31636 (W 0.396019) | 105.384 ± 0.329872 (W 0.214439) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.934973 ± 0.00310425 (W 0.00171121) | 2.6795 ± 0.0141337 (W 0.00436338) | - | 🔵 colmat 2.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.933621 ± 0.00608333 (W 0.00188877) | 2.69181 ± 0.0122455 (W 0.00487266) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.48395 ± 0.0181206 (W 0.00954642) | 12.3952 ± 0.0698308 (W 0.0141178) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.53356 ± 0.0346234 (W 0.0057095) | 12.4163 ± 0.0550289 (W 0.0212749) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 37.4757 ± 0.138362 (W 0.0613032) | 66.0027 ± 0.431281 (W 0.127185) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 37.5729 ± 0.276541 (W 0.0394616) | 66.331 ± 0.128418 (W 0.430973) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004849 ± 1.0504e-05 (W 0.000140704) | 0.004348 ± 4.50925e-05 (W 7.63544e-05) | 0.004148 ± 2.13776e-05 (W 3.23156e-05) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 1.17× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013485 ± 0.000132855 (W 0.000161948) | 0.012534 ± 0.000174041 (W 0.000108995) | 0.009608 ± 3.66106e-05 (W 6.61816e-05) | 🔴 colmat 1.08× slower than `faer` | 🔴 colmat 1.40× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.044994 ± 0.000629953 (W 0.000240687) | 0.04278 ± 0.000366011 (W 0.000269268) | 0.032711 ± 0.00100317 (W 0.000813211) | 🔴 colmat 1.05× slower than `faer` | 🔴 colmat 1.38× slower than `lapack-aocl` | `!` between-start range over 5% |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.861476 ± 0.00146007 (W 0.00442588) | 2.33183 ± 0.00728223 (W 0.00406241) | - | 🔵 colmat 2.71× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.867286 ± 0.00281326 (W 0.00217396) | 2.33506 ± 0.00822592 (W 0.00381894) | - | 🔵 colmat 2.69× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.42423 ± 0.0130202 (W 0.00625535) | 7.01175 ± 0.0153661 (W 0.0223131) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.41785 ± 0.00656916 (W 0.0139923) | 6.95356 ± 0.036469 (W 0.0133816) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.7958 ± 0.0685909 (W 0.0235298) | 25.1509 ± 0.0201568 (W 0.0153461) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.6582 ± 0.0291187 (W 0.0187481) | 25.162 ± 0.0550128 (W 0.0208188) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.25774 ± 0.00410359 (W 0.00256634) | 0.313375 ± 0.00203891 (W 0.00240317) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.272542 ± 0.00078179 (W 0.00272382) | 0.308846 ± 0.000858651 (W 0.00396767) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.01404 ± 0.00553747 (W 0.00485087) | 3.12668 ± 0.0193519 (W 0.0110161) | - | 🔵 colmat 3.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.31975 ± 0.0114955 (W 0.00349192) | 2.74872 ± 0.00940017 (W 0.00631528) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 4.73082 ± 0.0214624 (W 0.010989) | 17.7298 ± 0.0270585 (W 0.0190674) | - | 🔵 colmat 3.75× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 6.33957 ± 0.00969026 (W 0.00836327) | 15.3994 ± 0.0592421 (W 0.0150513) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.084008 ± 0.000178298 (W 0.000449567) | 0.081091 ± 0.000498206 (W 0.00129492) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.084518 ± 0.000398245 (W 0.00214058) | 0.081152 ± 0.000493222 (W 0.000956422) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.520647 ± 0.000378073 (W 0.00274355) | 0.51158 ± 0.00179248 (W 0.00174817) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.519234 ± 0.00150324 (W 0.0028962) | 0.516349 ± 0.00166781 (W 0.00248916) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.17357 ± 0.00509146 (W 0.0141522) | 4.1034 ± 0.0115632 (W 0.0120386) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.16087 ± 0.00940118 (W 0.00664917) | 4.09441 ± 0.0189978 (W 0.00458425) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.302594 ± 0.000289542 (W 0.00328534) | 0.852139 ± 0.00532228 (W 0.00562203) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.374098 ± 0.0015235 (W 0.00346135) | 0.650112 ± 0.00116185 (W 0.00323153) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.04893 ± 0.00334665 (W 0.00213614) | 3.5179 ± 0.0302783 (W 0.0159666) | - | 🔵 colmat 3.35× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.55587 ± 0.00959744 (W 0.00626579) | 2.34636 ± 0.0156404 (W 0.00349356) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.97183 ± 0.0168215 (W 0.00730055) | 19.3356 ± 0.135204 (W 0.0677749) | - | 🔵 colmat 3.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.30467 ± 0.0071442 (W 0.0145784) | 9.79253 ± 0.330688 (W 0.0220239) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.18226 ± 0.000944558 (W 0.00152827) | 0.728719 ± 0.00698098 (W 0.00247587) | - | 🔵 colmat 4.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.658548 ± 0.00191513 (W 0.00230022) | 3.09252 ± 0.0204305 (W 0.00916925) | - | 🔵 colmat 4.70× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 3.19292 ± 0.0141683 (W 0.0090224) | 16.6065 ± 0.153424 (W 0.0569723) | - | 🔵 colmat 5.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.9894 ± 0.00591435 (W 0.00499876) | 2.78412 ± 0.00428763 (W 0.0133715) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99062 ± 0.00706188 (W 0.00772797) | 2.79732 ± 0.012904 (W 0.0148913) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.02324 ± 0.0395045 (W 0.0209205) | 13.1957 ± 0.0394082 (W 0.0214235) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.04949 ± 0.0180796 (W 0.0170576) | 13.2231 ± 0.0591534 (W 0.0240086) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 36.5714 ± 0.102602 (W 0.0283532) | 55.3514 ± 0.0684362 (W 0.080309) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 36.7446 ± 0.0617518 (W 0.0492625) | 55.574 ± 0.136567 (W 0.223558) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.966381 ± 0.00747882 (W 0.00311207) | 2.18167 ± 0.00658761 (W 0.00931896) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.974648 ± 0.0103073 (W 0.00400311) | 2.18701 ± 0.00666729 (W 0.00479493) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.146 ± 0.0123242 (W 0.00937438) | 10.4471 ± 0.0259441 (W 0.013967) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.14948 ± 0.00936482 (W 0.00756666) | 10.45 ± 0.0342568 (W 0.013034) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.4276 ± 0.0785663 (W 0.0242235) | 41.957 ± 0.129386 (W 0.0502513) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.4907 ± 0.0569601 (W 0.0153498) | 41.8006 ± 0.0789996 (W 0.0352383) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00547 ± 0.000119862 (W 3.83406e-05) | 0.004288 ± 8e-05 (W 4.97996e-05) | 0.004118 ± 3.05505e-05 (W 3.22211e-05) | 🔴 colmat 1.28× slower than `faer` | 🔴 colmat 1.33× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013445 ± 2e-05 (W 0.000146817) | 0.012453 ± 0.000151229 (W 6.54164e-05) | 0.009408 ± 8.3668e-05 (W 4.61324e-05) | 🔴 colmat 1.08× slower than `faer` | 🔴 colmat 1.43× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.03246 ± 0.000190596 (W 0.000221917) | 0.031609 ± 0.000296352 (W 0.000276851) | 0.033113 ± 0.000573502 (W 0.000472918) | · | · |  |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.667163 ± 0.00604151 (W 0.00222126) | 2.35078 ± 0.0124391 (W 0.00852188) | - | 🔵 colmat 3.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.669181 ± 0.00356543 (W 0.00351333) | 2.33559 ± 0.0120928 (W 0.00577906) | - | 🔵 colmat 3.49× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.49007 ± 0.00540214 (W 0.00713457) | 6.98537 ± 0.00809332 (W 0.0226687) | - | 🔵 colmat 2.81× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.49919 ± 0.00536819 (W 0.01711) | 6.9737 ± 0.0156429 (W 0.021578) | - | 🔵 colmat 2.79× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.5351 ± 0.00627667 (W 0.0145529) | 23.7067 ± 0.106585 (W 0.902188) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.5448 ± 0.0201463 (W 0.0110178) | 23.7886 ± 0.0433202 (W 0.0561104) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.255109 ± 0.000257422 (W 0.0027385) | 0.315602 ± 0.000856493 (W 0.00194482) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.278078 ± 0.00240937 (W 0.0026066) | 0.311029 ± 0.000860054 (W 0.00367264) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.906939 ± 0.00786519 (W 0.00342202) | 3.13202 ± 0.00641242 (W 0.00806794) | - | 🔵 colmat 3.45× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.23425 ± 0.00187297 (W 0.0030234) | 2.76347 ± 0.00626293 (W 0.0122044) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 3.38524 ± 0.0208475 (W 0.00823127) | 16.6435 ± 0.0431526 (W 0.0220413) | - | 🔵 colmat 4.92× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 5.11199 ± 0.0100699 (W 0.0172286) | 14.4575 ± 0.0579268 (W 0.0279292) | - | 🔵 colmat 2.83× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.060229 ± 0.000606677 (W 0.00185645) | 0.100178 ± 0.000811178 (W 0.00115972) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.058855 ± 0.000693993 (W 0.00112086) | 0.099377 ± 0.00033194 (W 0.00120271) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.286204 ± 0.000133727 (W 0.00363156) | 0.395521 ± 0.0181533 (W 0.0182853) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.284904 ± 0.00191138 (W 0.00260444) | 0.398557 ± 0.0138066 (W 0.0129) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.26261 ± 0.00627298 (W 0.00751904) | 2.91129 ± 0.0963191 (W 0.30176) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.26759 ± 0.0148608 (W 0.0113854) | 2.85871 ± 0.359092 (W 0.333709) | - | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.63 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.292274 ± 5.81979e-05 (W 0.00207273) | 0.969655 ± 0.00597325 (W 0.00279663) | - | 🔵 colmat 3.32× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.362258 ± 0.00138843 (W 0.00201563) | 0.779222 ± 0.00369966 (W 0.00369267) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.854693 ± 0.00211621 (W 0.00103124) | 3.79573 ± 0.0200504 (W 0.0194939) | - | 🔵 colmat 4.44× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.37124 ± 0.00397664 (W 0.00780288) | 2.27627 ± 0.0215331 (W 0.0110475) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.42944 ± 0.0129936 (W 0.0147537) | 19.3193 ± 0.0628005 (W 0.104991) | - | 🔵 colmat 5.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 5.99742 ± 0.0436394 (W 0.0194604) | 8.51042 ± 0.021573 (W 0.054471) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.176078 ± 0.000935627 (W 0.00154429) | 0.832932 ± 0.0106063 (W 0.00485676) | - | 🔵 colmat 4.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.618391 ± 0.000342082 (W 0.00296647) | 3.3738 ± 0.0185666 (W 0.00953034) | - | 🔵 colmat 5.46× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.9222 ± 0.00568224 (W 0.0029584) | 16.9535 ± 0.0398305 (W 0.0366845) | - | 🔵 colmat 8.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.99825 ± 0.0229115 (W 0.00759585) | 2.78707 ± 0.0120084 (W 0.00449936) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.98913 ± 0.00133386 (W 0.00338078) | 2.81794 ± 0.0147404 (W 0.0105749) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.28719 ± 0.00865232 (W 0.00548522) | 13.2222 ± 0.042679 (W 0.0221787) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.35388 ± 0.0110197 (W 0.00656273) | 13.2084 ± 0.0191399 (W 0.0210757) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.1878 ± 0.152237 (W 0.0426694) | 52.1559 ± 0.101889 (W 0.14407) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.193 ± 0.14812 (W 0.043517) | 52.2902 ± 0.2991 (W 0.131682) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.0149 ± 0.00127768 (W 0.00287514) | 2.16868 ± 0.00762831 (W 0.00603691) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.02008 ± 0.00239904 (W 0.00330431) | 2.17443 ± 0.00819022 (W 0.00807692) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.01536 ± 0.0131276 (W 0.00755411) | 10.47 ± 0.0305054 (W 0.00842694) | - | 🔵 colmat 2.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.01215 ± 0.0198679 (W 0.00975992) | 10.4718 ± 0.0285864 (W 0.0142833) | - | 🔵 colmat 2.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.2752 ± 0.0608363 (W 0.0129188) | 40.0549 ± 0.0701385 (W 0.0395506) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.287 ± 0.0395553 (W 0.0207003) | 40.0311 ± 0.132322 (W 0.0439083) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00555 ± 6.06658e-05 (W 9.31155e-05) | 0.004318 ± 1.73205e-05 (W 5.99933e-05) | 0.004148 ± 5.7735e-07 (W 4.97996e-05) | 🔴 colmat 1.29× slower than `faer` | 🔴 colmat 1.34× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013064 ± 7.23418e-05 (W 0.000135223) | 0.012644 ± 0.000185608 (W 0.000135986) | 0.009619 ± 2.05994e-05 (W 3.76829e-05) | · | 🔴 colmat 1.36× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030607 ± 0.000453388 (W 0.00020327) | 0.030607 ± 0.000245984 (W 0.000375201) | 0.033863 ± 0.000353399 (W 0.000364641) | · | 🔵 colmat 1.11× faster than `lapack-aocl` |  |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 2.05909 ± 0.0139557 (W 0.00279061) | 2.39875 ± 0.00946018 (W 0.00755143) | 1.81615 ± 0.00747595 (W 0.00507166) | 🔵 colmat 1.16× faster than `faer` | 🔴 colmat 1.13× slower than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 2.06088 ± 0.00996249 (W 0.00214088) | 2.3998 ± 0.00473007 (W 0.00533463) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.55092 ± 0.0851262 (W 0.0128285) | 10.0829 ± 0.0577978 (W 0.0431263) | 8.58851 ± 0.029249 (W 0.0102428) | 🔵 colmat 1.06× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 9.61128 ± 0.0689912 (W 0.00920008) | 10.1424 ± 0.0451978 (W 0.0139663) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 49.0535 ± 0.144732 (W 0.0447941) | 49.0853 ± 0.0996632 (W 0.117198) | 48.2345 ± 0.283033 (W 0.355282) | · | · |  |
| `eigh/n512/b1 [api-inclusive]` | 49.0686 ± 0.361257 (W 0.086832) | 49.0122 ± 0.293045 (W 0.0971352) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.306452 ± 0.0353531 (W 0.00374169) | 0.312784 ± 0.0176423 (W 0.00327874) | 6.3365 ± 0.0956074 (W 0.00665603) | · | 🔵 colmat 20.68× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.295987 ± 0.00112056 (W 0.00203571) | 0.286174 ± 0.00175402 (W 0.00223386) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.05777 ± 0.231143 (W 0.00522988) | 2.84012 ± 0.0120041 (W 0.0098158) | 54.0629 ± 0.0173627 (W 0.0425212) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 26.27× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.11875 ± 0.00657341 (W 0.00384252) | 2.86606 ± 0.00841956 (W 0.0109691) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 16.1183 ± 0.703161 (W 0.025492) | 22.3209 ± 0.145596 (W 0.018988) | 436.219 ± 0.130802 (W 0.962626) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 27.06× faster than `lapack-aocl` | `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 15.601 ± 0.0970189 (W 0.0409381) | 21.7094 ± 0.0911788 (W 0.0615467) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.256903 ± 0.000912043 (W 0.00259271) | 0.257364 ± 0.00239271 (W 0.0027937) | 0.454819 ± 0.00151642 (W 0.00631328) | · | 🔵 colmat 1.77× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.257294 ± 0.00245678 (W 0.00314241) | 0.255867 ± 0.00123017 (W 0.00254359) | 0.486177 ± 0.00115345 (W 0.00659071) | · | 🔵 colmat 1.89× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.88025 ± 0.0090892 (W 0.00607697) | 1.88586 ± 0.00695047 (W 0.00587903) | 2.93205 ± 0.0134649 (W 0.00773882) | · | 🔵 colmat 1.56× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.88677 ± 0.00946592 (W 0.00476579) | 1.8772 ± 0.00659441 (W 0.00145631) | 2.7285 ± 0.00838185 (W 0.00418352) | · | 🔵 colmat 1.45× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9094 ± 0.0270807 (W 0.105764) | 16.1467 ± 0.135388 (W 0.197849) | 21.4191 ± 0.0633956 (W 0.032445) | · | 🔵 colmat 1.35× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.8223 ± 0.0552542 (W 0.039778) | 15.9609 ± 0.0675938 (W 0.0199819) | 18.7348 ± 0.066176 (W 0.0375796) | · | 🔵 colmat 1.18× faster than `lapack-aocl` |  |
| `qr/n128/b1` | 0.36174 ± 0.00234091 (W 0.00273881) | 0.482019 ± 0.00431945 (W 0.00281183) | 1.12409 ± 0.00369252 (W 0.00582418) | 🔵 colmat 1.33× faster than `faer` | 🔵 colmat 3.11× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.442946 ± 0.00236202 (W 0.00246446) | 0.562961 ± 0.00282523 (W 0.00289853) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.64469 ± 0.00949073 (W 0.00402201) | 2.73395 ± 0.00832639 (W 0.00511379) | 5.71166 ± 0.0315464 (W 0.0123171) | · | 🔵 colmat 2.16× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.10911 ± 0.00622788 (W 0.00695519) | 3.19351 ± 0.0153074 (W 0.00929626) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 17.6274 ± 0.130295 (W 0.015491) | 17.9221 ± 0.0446464 (W 0.0177362) | 35.2944 ± 0.0393961 (W 0.0614883) | · | 🔵 colmat 2.00× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 19.6147 ± 0.162236 (W 0.0401356) | 20.2131 ± 0.170223 (W 0.0385135) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.16036 ± 0.00212548 (W 0.00199277) | 0.284064 ± 0.00106949 (W 0.0016086) | 0.63832 ± 0.00163923 (W 0.00209669) | 🔵 colmat 1.77× faster than `faer` | 🔵 colmat 3.98× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.29778 ± 0.00140942 (W 0.00293164) | 1.38715 ± 0.00312524 (W 0.00499494) | 3.02909 ± 0.0051531 (W 0.00830513) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 2.33× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 7.74539 ± 0.019319 (W 0.0127581) | 8.05263 ± 0.0428116 (W 0.0103854) | 16.4618 ± 0.0393944 (W 0.0268437) | · | 🔵 colmat 2.13× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 2.80546 ± 0.0174699 (W 0.005756) | 3.68369 ± 0.0106365 (W 0.00561388) | 2.98854 ± 0.011627 (W 0.00654958) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.82203 ± 0.00597513 (W 0.0079658) | 3.70605 ± 0.00889703 (W 0.00677109) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.4082 ± 0.0505059 (W 0.0143828) | 18.3054 ± 0.104063 (W 0.0245751) | 15.914 ± 0.0560532 (W 0.0179096) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 14.3779 ± 0.0641298 (W 0.0265501) | 18.3674 ± 0.0786297 (W 0.0234763) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 85.9686 ± 0.258001 (W 0.130518) | 104.234 ± 0.315808 (W 0.352061) | 106.972 ± 0.823431 (W 0.189898) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 86.0219 ± 0.214252 (W 0.182614) | 104.883 ± 0.308032 (W 0.156306) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.912262 ± 0.00432172 (W 0.00170014) | 2.60411 ± 0.0081809 (W 0.00448796) | 1.16363 ± 0.00876455 (W 0.00240169) | 🔵 colmat 2.85× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.913112 ± 0.00429952 (W 0.00142606) | 2.62618 ± 0.00310904 (W 0.00474834) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.36826 ± 0.0260715 (W 0.00809944) | 12.2594 ± 0.0198648 (W 0.013417) | 7.00188 ± 0.0200465 (W 0.0100562) | 🔵 colmat 2.28× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.38354 ± 0.0265366 (W 0.00388549) | 12.2381 ± 0.0304121 (W 0.0151494) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 36.4123 ± 0.0817766 (W 0.0392159) | 65.1721 ± 0.321678 (W 0.168289) | 48.5428 ± 0.917055 (W 0.0641255) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 1.33× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 36.4798 ± 0.121053 (W 0.0636225) | 65.0162 ± 0.421383 (W 0.075039) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00536 ± 3.05505e-05 (W 0.000125461) | 0.005119 ± 3.00056e-05 (W 0.000100396) | 0.002966 ± 1.76163e-05 (W 3.92849e-05) | · | 🔴 colmat 1.81× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.014307 ± 8.32666e-05 (W 0.000143116) | 0.013946 ± 4.3589e-05 (W 9.92472e-05) | 0.007804 ± 9.53939e-05 (W 7.63164e-05) | · | 🔴 colmat 1.83× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.04799 ± 0.000617507 (W 0.0019797) | 0.046788 ± 0.000646361 (W 0.000213261) | 0.029846 ± 0.000821874 (W 0.000376159) | · | 🔴 colmat 1.61× slower than `lapack-aocl` | `!` between-start range over 5% |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.858892 ± 0.0018915 (W 0.00194345) | 2.33482 ± 0.0130837 (W 0.00568296) | 1.48942 ± 0.0100935 (W 0.0067043) | 🔵 colmat 2.72× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.862937 ± 0.00186055 (W 0.00269142) | 2.32825 ± 0.00827053 (W 0.00490905) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.37961 ± 0.00721103 (W 0.0103676) | 6.94309 ± 0.0142508 (W 0.0225353) | 6.0583 ± 0.0161532 (W 0.0115819) | 🔵 colmat 2.05× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.36846 ± 0.00363242 (W 0.0056493) | 6.93722 ± 0.0336249 (W 0.0146354) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.6047 ± 0.0507051 (W 0.0257833) | 24.8115 ± 0.0498274 (W 0.0202833) | 27.8012 ± 0.0321142 (W 0.0494747) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 15.5673 ± 0.0649812 (W 0.0240722) | 24.7411 ± 0.0739252 (W 0.0138747) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.242963 ± 0.00188711 (W 0.00227288) | 0.307284 ± 0.000783117 (W 0.0018562) | 6.46469 ± 0.128052 (W 0.0150347) | 🔵 colmat 1.26× faster than `faer` | 🔵 colmat 26.61× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.251031 ± 0.0031566 (W 0.00280194) | 0.286705 ± 0.00151771 (W 0.00236701) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.03971 ± 0.00221339 (W 0.00352903) | 2.80746 ± 0.0155281 (W 0.00611481) | 53.6647 ± 0.277668 (W 0.0426547) | 🔵 colmat 2.70× faster than `faer` | 🔵 colmat 51.62× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.23037 ± 0.0125269 (W 0.004083) | 2.63037 ± 0.0120121 (W 0.0122544) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 5.25931 ± 0.0310211 (W 0.00567673) | 15.1377 ± 0.0569032 (W 0.0312347) | 433.06 ± 1.62776 (W 0.321698) | 🔵 colmat 2.88× faster than `faer` | 🔵 colmat 82.34× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 5.90168 ± 0.0326979 (W 0.00982036) | 14.3901 ± 0.0350844 (W 0.0181592) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.084368 ± 0.000646604 (W 0.000276494) | 0.081473 ± 0.000424946 (W 0.000612896) | 0.30031 ± 0.00295519 (W 0.00895997) | · | 🔵 colmat 3.56× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.083726 ± 0.000814059 (W 0.000410276) | 0.081403 ± 7.27622e-05 (W 0.000404346) | 0.213115 ± 0.00210212 (W 0.00420287) | · | 🔵 colmat 2.55× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.525827 ± 0.00148844 (W 0.00183682) | 0.523796 ± 0.00220923 (W 0.00407054) | 1.67901 ± 0.00210535 (W 0.00895044) | · | 🔵 colmat 3.19× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.525185 ± 0.00247874 (W 0.00231207) | 0.52242 ± 0.00323728 (W 0.0024554) | 0.998541 ± 0.00303174 (W 0.00441014) | · | 🔵 colmat 1.90× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.17176 ± 0.000972924 (W 0.0107622) | 4.19793 ± 0.021397 (W 0.00539453) | 10.9864 ± 0.0801128 (W 0.0206155) | · | 🔵 colmat 2.63× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.16369 ± 0.00520782 (W 0.00820698) | 4.1849 ± 0.0151892 (W 0.0127993) | 5.45353 ± 0.0142927 (W 0.0108018) | · | 🔵 colmat 1.31× faster than `lapack-aocl` |  |
| `qr/n128/b1` | 0.262382 ± 0.00154727 (W 0.00151017) | 0.539507 ± 0.00698521 (W 0.00448514) | 0.494893 ± 0.0050185 (W 0.00396946) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.89× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.349997 ± 0.00038431 (W 0.00251053) | 0.628152 ± 0.00610178 (W 0.00267684) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.988803 ± 0.0039005 (W 0.00305913) | 1.73361 ± 0.00680593 (W 0.00779856) | 2.00425 ± 0.0184394 (W 0.00444243) | 🔵 colmat 1.75× faster than `faer` | 🔵 colmat 2.03× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.45642 ± 0.00777318 (W 0.00460801) | 2.2017 ± 0.00190241 (W 0.0108598) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.59585 ± 0.0128191 (W 0.00908707) | 7.08404 ± 0.029387 (W 0.0112145) | 10.4337 ± 0.00445817 (W 0.0173857) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 1.86× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 7.7119 ± 0.0113756 (W 0.0213615) | 8.69123 ± 0.0507841 (W 0.0334891) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.160189 ± 0.000455006 (W 0.00041204) | 0.430963 ± 0.00327319 (W 0.00312188) | 0.277368 ± 0.00108638 (W 0.00468196) | 🔵 colmat 2.69× faster than `faer` | 🔵 colmat 1.73× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 0.591022 ± 0.00353897 (W 0.00335387) | 1.33974 ± 0.00297877 (W 0.00610378) | 1.03666 ± 0.00561223 (W 0.00486052) | 🔵 colmat 2.27× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 2.95033 ± 0.015683 (W 0.00731408) | 4.49751 ± 0.00592947 (W 0.012642) | 4.91993 ± 0.0145969 (W 0.00992585) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 1.97024 ± 0.00195951 (W 0.00597736) | 2.76141 ± 0.0135383 (W 0.0141876) | 2.56189 ± 0.0472925 (W 0.00823802) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.98263 ± 0.00917358 (W 0.00405542) | 2.7682 ± 0.0255975 (W 0.01603) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.91609 ± 0.0371741 (W 0.0185109) | 13.0757 ± 0.0372746 (W 0.0218092) | 13.413 ± 0.07297 (W 0.0661123) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.69× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.93554 ± 0.021222 (W 0.0109537) | 13.1202 ± 0.0471826 (W 0.0333453) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 35.6604 ± 0.0784916 (W 0.0477983) | 54.6337 ± 0.0979803 (W 0.0329758) | 65.5615 ± 0.518311 (W 0.436635) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 1.84× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 35.7982 ± 0.0737913 (W 0.0584545) | 54.7153 ± 0.117527 (W 0.223644) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.949391 ± 0.00427513 (W 0.00138996) | 2.1217 ± 0.0033018 (W 0.00654655) | 1.19176 ± 0.0147772 (W 0.00378483) | 🔵 colmat 2.23× faster than `faer` | 🔵 colmat 1.26× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.948469 ± 0.00307864 (W 0.00204303) | 2.12166 ± 0.00432984 (W 0.00723532) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.03004 ± 0.00786285 (W 0.00704678) | 10.2684 ± 0.0259008 (W 0.0143097) | 7.842 ± 0.0164235 (W 0.0396648) | 🔵 colmat 2.55× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.99978 ± 0.019231 (W 0.0129769) | 10.2856 ± 0.0339362 (W 0.0124743) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.5228 ± 0.0338131 (W 0.0204025) | 41.0233 ± 0.110873 (W 0.0914334) | 35.4124 ± 0.130175 (W 0.0511732) | 🔵 colmat 2.34× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 17.4655 ± 0.0234846 (W 0.022056) | 40.9273 ± 0.148457 (W 0.0531153) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005831 ± 7.02377e-05 (W 4.61519e-05) | 0.004909 ± 3.10966e-05 (W 6.1887e-05) | 0.002965 ± 6.08276e-06 (W 6.90818e-05) | 🔴 colmat 1.19× slower than `faer` | 🔴 colmat 1.97× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015549 ± 9.81495e-05 (W 0.000209242) | 0.013916 ± 4.04145e-05 (W 8.35117e-05) | 0.007745 ± 3.21455e-05 (W 5.71516e-05) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 2.01× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.040125 ± 0.000116047 (W 0.000479689) | 0.03752 ± 0.000140119 (W 0.000508363) | 0.029374 ± 0.000241222 (W 0.00014237) | 🔴 colmat 1.07× slower than `faer` | 🔴 colmat 1.37× slower than `lapack-aocl` |  |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.658973 ± 0.00315842 (W 0.00163937) | 2.34464 ± 0.00976622 (W 0.00663171) | 1.51027 ± 0.00944702 (W 0.00940219) | 🔵 colmat 3.56× faster than `faer` | 🔵 colmat 2.29× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.66292 ± 0.00511623 (W 0.00393207) | 2.33847 ± 0.00530491 (W 0.00309481) | - | 🔵 colmat 3.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.44611 ± 0.0105098 (W 0.00846799) | 6.92522 ± 0.0146317 (W 0.0196092) | 5.66384 ± 0.00748726 (W 0.01823) | 🔵 colmat 2.83× faster than `faer` | 🔵 colmat 2.32× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 2.43363 ± 0.0144967 (W 0.00674409) | 6.94693 ± 0.0294204 (W 0.027674) | - | 🔵 colmat 2.85× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.3615 ± 0.0119324 (W 0.0142702) | 23.3821 ± 0.0266497 (W 0.0208195) | 25.046 ± 0.032315 (W 0.0678982) | 🔵 colmat 2.26× faster than `faer` | 🔵 colmat 2.42× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 10.4124 ± 0.0570549 (W 0.0275832) | 23.3762 ± 0.0690234 (W 0.0824127) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.255366 ± 0.0016491 (W 0.00152732) | 0.307187 ± 0.0027228 (W 0.0025136) | 6.35008 ± 0.0802193 (W 0.0135526) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 24.87× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.254074 ± 0.00121809 (W 0.00213781) | 0.286634 ± 0.000597174 (W 0.00382296) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.920565 ± 0.00362848 (W 0.00565605) | 2.83272 ± 0.015481 (W 0.00470414) | 54.0561 ± 0.0250982 (W 0.0444508) | 🔵 colmat 3.08× faster than `faer` | 🔵 colmat 58.72× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.14506 ± 0.00381688 (W 0.00557702) | 2.61612 ± 0.00272412 (W 0.0103404) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 3.59431 ± 0.0116249 (W 0.00980985) | 14.0954 ± 0.0363504 (W 0.026595) | 435.063 ± 0.607536 (W 0.466039) | 🔵 colmat 3.92× faster than `faer` | 🔵 colmat 121.04× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 4.62163 ± 0.0192688 (W 0.0109933) | 13.4815 ± 0.0615023 (W 0.042794) | - | 🔵 colmat 2.92× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.056611 ± 0.000105958 (W 0.00147777) | 0.099849 ± 0.00131446 (W 0.00248254) | 0.295554 ± 0.000886587 (W 0.00652638) | 🔵 colmat 1.76× faster than `faer` | 🔵 colmat 5.22× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.056206 ± 0.000345877 (W 0.000568429) | 0.099826 ± 0.000427006 (W 0.00101352) | 0.222718 ± 0.00191798 (W 0.00647822) | 🔵 colmat 1.78× faster than `faer` | 🔵 colmat 3.96× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.291533 ± 0.000803224 (W 0.00131296) | 0.400306 ± 0.0107694 (W 0.0128397) | 1.44004 ± 0.00252617 (W 0.00705236) | 🔵 colmat 1.37× faster than `faer` | 🔵 colmat 4.94× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.295531 ± 0.00136502 (W 0.0013387) | 0.406611 ± 0.0180711 (W 0.0186532) | 0.669166 ± 0.010934 (W 0.00804847) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 2.26× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.27871 ± 0.0144753 (W 0.00812841) | 2.95077 ± 0.0371045 (W 0.37691) | 9.4068 ± 0.0456602 (W 0.0255176) | 🔵 colmat 1.29× faster than `faer` | 🔵 colmat 4.13× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.28773 ± 0.0101001 (W 0.00580533) | 2.3322 ± 0.383616 (W 0.348778) | 3.38489 ± 0.0118925 (W 0.00788491) | · | 🔵 colmat 1.48× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1` | 0.239229 ± 0.000954254 (W 0.00218109) | 0.67104 ± 0.00861459 (W 0.00364922) | 0.494308 ± 0.00325997 (W 0.0089699) | 🔵 colmat 2.81× faster than `faer` | 🔵 colmat 2.07× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.342272 ± 0.00282262 (W 0.00410804) | 0.756039 ± 0.00977776 (W 0.00366105) | - | 🔵 colmat 2.21× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.812996 ± 0.0113199 (W 0.00281746) | 2.05279 ± 0.00697096 (W 0.0110934) | 1.45397 ± 0.00641462 (W 0.0152329) | 🔵 colmat 2.52× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.28005 ± 0.0054041 (W 0.00873548) | 2.1452 ± 0.00945743 (W 0.0158383) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.20994 ± 0.0104692 (W 0.00316712) | 7.11671 ± 0.049421 (W 0.0397453) | 6.49494 ± 0.0660569 (W 0.014764) | 🔵 colmat 2.22× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 5.49067 ± 0.0698659 (W 0.0229872) | 7.52157 ± 0.361746 (W 0.063949) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.160988 ± 0.000427371 (W 0.00144488) | 0.548421 ± 0.00760788 (W 0.00590932) | 0.25142 ± 0.00173196 (W 0.00480669) | 🔵 colmat 3.41× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 0.575262 ± 0.00399524 (W 0.00352591) | 1.62763 ± 0.0146308 (W 0.00815585) | 0.797496 ± 0.00288059 (W 0.0111125) | 🔵 colmat 2.83× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 1.78249 ± 0.0144864 (W 0.00404136) | 4.82685 ± 0.0154145 (W 0.0346593) | 3.15218 ± 0.0274246 (W 0.0277751) | 🔵 colmat 2.71× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 1.95115 ± 0.0042996 (W 0.00322125) | 2.77504 ± 0.0146782 (W 0.00370641) | 2.5752 ± 0.0300221 (W 0.0133852) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.95712 ± 0.0140131 (W 0.00437442) | 2.77644 ± 0.0100123 (W 0.0042532) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.16562 ± 0.00450135 (W 0.0148004) | 13.0441 ± 0.0279096 (W 0.0314207) | 15.6245 ± 0.0129485 (W 0.0587421) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 2.18× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.20488 ± 0.10255 (W 0.0125723) | 13.0887 ± 0.0397608 (W 0.0184977) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 29.2234 ± 0.0982221 (W 0.0305143) | 51.2283 ± 0.17563 (W 0.0810248) | 61.2137 ± 0.738782 (W 0.208717) | 🔵 colmat 1.75× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 29.3521 ± 0.106271 (W 0.0303419) | 51.4956 ± 0.252262 (W 0.0917936) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.990969 ± 0.000107056 (W 0.00244652) | 2.11103 ± 0.00846148 (W 0.00417195) | 1.19074 ± 0.0017292 (W 0.00377098) | 🔵 colmat 2.13× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.991025 ± 0.00414689 (W 0.00368955) | 2.12066 ± 0.00784051 (W 0.00375486) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.89556 ± 0.00675549 (W 0.00443305) | 10.3047 ± 0.046516 (W 0.0114804) | 10.5368 ± 0.0145664 (W 0.0471465) | 🔵 colmat 2.65× faster than `faer` | 🔵 colmat 2.70× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.89553 ± 0.0267686 (W 0.00679707) | 10.303 ± 0.0477718 (W 0.0103747) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.3264 ± 0.025559 (W 0.0150732) | 39.1476 ± 0.0534736 (W 0.051833) | 36.8932 ± 0.227764 (W 0.0673259) | 🔵 colmat 2.55× faster than `faer` | 🔵 colmat 2.41× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 15.3666 ± 0.0578102 (W 0.0116193) | 39.1646 ± 0.110955 (W 0.104973) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00585 ± 0.000133293 (W 5.41295e-05) | 0.004889 ± 3.78594e-05 (W 4.23462e-05) | 0.002966 ± 1.76163e-05 (W 2.93649e-05) | 🔴 colmat 1.20× slower than `faer` | 🔴 colmat 1.97× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015578 ± 0.000160575 (W 0.000116919) | 0.014056 ± 0.000343206 (W 0.000323551) | 0.008074 ± 0.000100504 (W 4.32007e-05) | 🔴 colmat 1.11× slower than `faer` | 🔴 colmat 1.93× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.037998 ± 0.000177827 (W 0.00024926) | 0.035536 ± 0.000473171 (W 0.00275103) | 0.033834 ± 0.000200206 (W 0.000348372) | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | 🔴 colmat 1.12× slower than `lapack-aocl` |  |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 3.06448 ± 0.0129837 (W 0.00483722) | 3.40485 ± 0.0118928 (W 0.00527098) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 3.08101 ± 0.0151707 (W 0.00187384) | 3.41584 ± 0.0198231 (W 0.00589316) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 17.0762 ± 0.0751379 (W 0.00807601) | 17.6696 ± 0.0571262 (W 0.0115449) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 17.1118 ± 0.0545528 (W 0.0268652) | 17.7304 ± 0.0809465 (W 0.0132538) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 105.023 ± 0.529876 (W 0.441721) | 105.262 ± 0.630806 (W 0.126995) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 105.366 ± 0.387337 (W 0.12793) | 105.493 ± 0.769487 (W 0.162677) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.676 ± 0.00277921 (W 0.00384897) | 0.803429 ± 0.0049826 (W 0.00808191) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.76621 ± 0.00169734 (W 0.00263597) | 0.879202 ± 0.000779286 (W 0.00540322) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 4.62186 ± 0.0425215 (W 0.0338618) | 6.42104 ± 0.0268402 (W 0.0113055) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 5.12313 ± 0.0312986 (W 0.0101105) | 6.54638 ± 0.0332823 (W 0.00903838) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 35.5246 ± 0.279559 (W 0.0741437) | 45.2686 ± 0.125323 (W 0.132494) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 38.731 ± 0.244848 (W 0.0457392) | 43.8805 ± 0.140788 (W 0.0964065) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.05492 ± 0.00107979 (W 0.00235586) | 1.05221 ± 0.00625617 (W 0.00301269) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.05614 ± 0.00219482 (W 0.0013934) | 1.05203 ± 0.00221282 (W 0.00233562) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.54469 ± 0.0328093 (W 0.00546119) | 7.52756 ± 0.0334335 (W 0.00734722) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.54516 ± 0.0414196 (W 0.0122746) | 7.54823 ± 0.033335 (W 0.0117814) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.8158 ± 0.256854 (W 0.128069) | 56.735 ± 0.253007 (W 0.30886) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.9325 ± 0.209662 (W 0.0621763) | 56.7378 ± 0.339558 (W 0.140721) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.47319 ± 0.0065546 (W 0.00290812) | 1.85572 ± 0.00956971 (W 0.00275361) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 1.49626 ± 0.00749559 (W 0.00224023) | 1.73743 ± 0.0100349 (W 0.00361296) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 10.252 ± 0.0699653 (W 0.00981791) | 11.9954 ± 0.045291 (W 0.0114276) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 11.0808 ± 0.064504 (W 0.015564) | 11.2388 ± 0.0727175 (W 0.0119527) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 70.3481 ± 0.317228 (W 0.0872679) | 77.7919 ± 0.351447 (W 0.386054) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 73.8828 ± 0.191658 (W 0.231239) | 74.3224 ± 0.253444 (W 0.093749) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.686921 ± 0.00297624 (W 0.00291464) | 1.06668 ± 0.00374724 (W 0.0025018) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 4.61331 ± 0.0142935 (W 0.00743169) | 6.46597 ± 0.0222697 (W 0.0107028) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 30.0128 ± 0.216173 (W 0.344786) | 37.9076 ± 0.405397 (W 0.0353724) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.74613 ± 0.0191562 (W 0.00350622) | 5.68642 ± 0.0206138 (W 0.00700661) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 4.7633 ± 0.0171699 (W 0.0101105) | 5.71286 ± 0.0349857 (W 0.00802793) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.348 ± 0.157964 (W 0.0820453) | 33.8984 ± 0.0743972 (W 0.0356371) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 29.5815 ± 0.213964 (W 0.0331486) | 33.8355 ± 0.17516 (W 0.0221369) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 195.309 ± 1.03615 (W 0.518058) | 208.04 ± 0.794999 (W 0.634338) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 195.192 ± 0.652158 (W 0.462575) | 208.801 ± 0.767876 (W 0.802584) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.61746 ± 0.00458052 (W 0.00343952) | 3.41838 ± 0.0121884 (W 0.00564844) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.6316 ± 0.00907081 (W 0.00198883) | 3.4566 ± 0.0168683 (W 0.00904315) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.415 ± 0.0247199 (W 0.0344943) | 19.1467 ± 0.103939 (W 0.0434044) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 11.4188 ± 0.154169 (W 0.0297572) | 19.1313 ± 0.069556 (W 0.0332175) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 84.3189 ± 0.393001 (W 0.246336) | 109.152 ± 0.226327 (W 0.287437) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 84.4255 ± 0.517172 (W 0.310393) | 109.439 ± 0.547669 (W 0.552492) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.008055 ± 3.4641e-05 (W 8.84873e-05) | 0.007564 ± 2.88675e-05 (W 4.58258e-05) | 0.007133 ± 3.00056e-05 (W 3.59486e-05) | 🔴 colmat 1.06× slower than `faer` | 🔴 colmat 1.13× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024365 ± 5.08626e-05 (W 8.76995e-05) | 0.023474 ± 0.000134288 (W 8.09833e-05) | 0.020418 ± 3.78594e-05 (W 0.000138954) | · | 🔴 colmat 1.19× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.089729 ± 0.000115182 (W 0.000406429) | 0.088375 ± 0.000238607 (W 0.00217507) | 0.071874 ± 0.000318153 (W 0.000303494) | · | 🔴 colmat 1.25× slower than `lapack-aocl` |  |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 1.2365 ± 0.0065626 (W 0.00405298) | 2.97255 ± 0.0142906 (W 0.00356535) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.24392 ± 0.00599998 (W 0.00337478) | 2.96665 ± 0.000967605 (W 0.00654803) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.48423 ± 0.0123556 (W 0.00587517) | 11.5369 ± 0.0142374 (W 0.0224321) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.52021 ± 0.0238382 (W 0.0071781) | 11.5972 ± 0.01878 (W 0.0107178) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 30.4602 ± 0.211704 (W 0.0334739) | 45.0924 ± 0.0677937 (W 0.0504681) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 30.6102 ± 0.173674 (W 0.0328501) | 45.1822 ± 0.0837274 (W 0.0395932) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.400436 ± 0.0032218 (W 0.00244126) | 0.813066 ± 0.0038074 (W 0.00645886) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.52797 ± 0.00307753 (W 0.00341098) | 0.778381 ± 0.00397351 (W 0.00581041) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.74734 ± 0.00931601 (W 0.00582657) | 6.50244 ± 0.0356932 (W 0.00729354) | - | 🔵 colmat 3.72× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.48455 ± 0.00673214 (W 0.00794457) | 6.025 ± 0.0133701 (W 0.0101066) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.2872 ± 0.0807817 (W 0.399305) | 28.3207 ± 0.190124 (W 0.0125931) | - | 🔵 colmat 2.75× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 14.3263 ± 0.0210834 (W 0.0502116) | 25.5802 ± 0.0725489 (W 0.051522) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.287085 ± 0.000936112 (W 0.00170669) | 0.28261 ± 0.0014171 (W 0.00228967) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.286239 ± 0.00176012 (W 0.00169665) | 0.282333 ± 0.00100711 (W 0.001824) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.94627 ± 0.00791581 (W 0.0053489) | 1.94218 ± 0.00719791 (W 0.00332445) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.9576 ± 0.00617251 (W 0.00519767) | 1.9502 ± 0.00329709 (W 0.00470563) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6912 ± 0.0120895 (W 0.0226329) | 14.7079 ± 0.0555677 (W 0.0369825) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6858 ± 0.0105831 (W 0.019774) | 14.6734 ± 0.0428767 (W 0.0132874) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.734901 ± 0.00265808 (W 0.00324768) | 1.35041 ± 0.00294173 (W 0.00751381) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.940695 ± 0.00409195 (W 0.00594461) | 1.21094 ± 0.00622809 (W 0.00294672) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.13902 ± 0.0122076 (W 0.00556121) | 6.29172 ± 0.0221706 (W 0.0187601) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 4.21063 ± 0.0105644 (W 0.00837909) | 5.28651 ± 0.0395126 (W 0.0116242) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 20.0351 ± 0.128333 (W 0.0192578) | 31.7845 ± 0.037122 (W 0.0572403) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 25.2299 ± 0.0548884 (W 0.0207603) | 27.4163 ± 0.0230086 (W 0.029164) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.448261 ± 0.00263825 (W 0.00262201) | 1.01913 ± 0.0034868 (W 0.00446516) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.64077 ± 0.000714245 (W 0.00379332) | 4.87344 ± 0.0349784 (W 0.0150541) | - | 🔵 colmat 2.97× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 9.28423 ± 0.0167793 (W 0.02263) | 21.4283 ± 0.0356793 (W 0.0685744) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.72247 ± 0.00351245 (W 0.00518969) | 4.03374 ± 0.0141301 (W 0.0117525) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.73329 ± 0.00915182 (W 0.0076549) | 4.03293 ± 0.0270578 (W 0.0136852) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.8096 ± 0.0312532 (W 0.0213934) | 21.3557 ± 0.0608377 (W 0.0282847) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 12.8471 ± 0.0516142 (W 0.0218472) | 21.5099 ± 0.0929349 (W 0.0929565) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 68.8977 ± 0.161493 (W 0.0619751) | 86.7984 ± 0.150294 (W 0.098405) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 69.0233 ± 0.127463 (W 0.0810097) | 87.5347 ± 0.0847085 (W 0.794874) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.27856 ± 0.00424309 (W 0.00338139) | 2.97245 ± 0.0172379 (W 0.00540612) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.27873 ± 0.00853908 (W 0.00746107) | 2.96817 ± 0.0124478 (W 0.00602038) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.29618 ± 0.0272049 (W 0.0285219) | 16.3534 ± 0.087983 (W 0.0219233) | - | 🔵 colmat 2.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.30213 ± 0.0119023 (W 0.0235459) | 16.2706 ± 0.0959181 (W 0.0208207) | - | 🔵 colmat 2.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 33.8492 ± 0.114692 (W 0.0784947) | 57.4266 ± 0.121954 (W 0.081999) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 34.0193 ± 0.256641 (W 0.0869669) | 57.1831 ± 0.126664 (W 0.39239) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009187 ± 4.09512e-05 (W 4.08571e-05) | 0.007645 ± 0.000108583 (W 5.87478e-05) | 0.007133 ± 2.5632e-05 (W 7.95883e-05) | 🔴 colmat 1.20× slower than `faer` | 🔴 colmat 1.29× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.021941 ± 0.000106829 (W 0.000170925) | 0.023985 ± 0.00029398 (W 0.000167831) | 0.020378 ± 0.000144388 (W 7.19667e-05) | 🔵 colmat 1.09× faster than `faer` | 🔴 colmat 1.08× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.056455 ± 0.000179484 (W 0.000338655) | 0.062918 ± 5.49939e-05 (W 0.000374485) | 0.072856 ± 0.000389557 (W 0.000270897) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-aocl` |  |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.924691 ± 0.0028093 (W 0.00298966) | 2.9884 ± 0.00268306 (W 0.0034036) | - | 🔵 colmat 3.23× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.938036 ± 0.000624281 (W 0.00407809) | 2.99199 ± 0.0292155 (W 0.00365424) | - | 🔵 colmat 3.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.73891 ± 0.0264209 (W 0.0192594) | 11.3874 ± 0.0324701 (W 0.0327957) | - | 🔵 colmat 3.05× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.75657 ± 0.0199107 (W 0.00853674) | 11.4131 ± 0.034506 (W 0.0391324) | - | 🔵 colmat 3.04× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 18.8969 ± 0.147241 (W 0.195325) | 39.1435 ± 0.237868 (W 0.301076) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 18.7888 ± 0.0281139 (W 0.0244463) | 39.1147 ± 0.0771611 (W 0.2661) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.361454 ± 0.002183 (W 0.00379826) | 0.809619 ± 0.00759118 (W 0.007123) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.498559 ± 0.00360182 (W 0.00324509) | 0.770786 ± 0.003246 (W 0.00674867) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.32932 ± 0.0312263 (W 0.00375791) | 6.43899 ± 0.0223733 (W 0.00878729) | - | 🔵 colmat 4.84× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.10046 ± 0.0157245 (W 0.00693946) | 6.04615 ± 0.0424078 (W 0.0204129) | - | 🔵 colmat 2.88× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.18429 ± 0.0310313 (W 0.0161056) | 25.6722 ± 0.101823 (W 0.034156) | - | 🔵 colmat 4.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 10.3433 ± 0.046705 (W 0.028895) | 23.2241 ± 0.0568283 (W 0.0254668) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.180085 ± 0.000333167 (W 0.00229078) | 0.27379 ± 0.00451649 (W 0.00429531) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.179504 ± 0.00113523 (W 0.00165214) | 0.271751 ± 0.00150633 (W 0.00424226) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.01428 ± 0.00361464 (W 0.00210021) | 1.30735 ± 0.0227589 (W 0.081964) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01454 ± 0.00285563 (W 0.0022332) | 1.30818 ± 0.098951 (W 0.0850187) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.89646 ± 0.0168312 (W 0.0153333) | 7.71255 ± 1.45584 (W 1.23849) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.89812 ± 0.0166611 (W 0.0322833) | 7.75793 ± 1.32359 (W 1.49974) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.573446 ± 0.00109252 (W 0.00389134) | 1.44764 ± 0.0171814 (W 0.00622093) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.808748 ± 0.00628729 (W 0.00614508) | 1.3243 ± 0.00975443 (W 0.00544084) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.09661 ± 0.00282025 (W 0.0069723) | 6.25553 ± 0.0165064 (W 0.0212368) | - | 🔵 colmat 2.98× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.25423 ± 0.00109704 (W 0.00683376) | 4.36403 ± 0.50532 (W 0.326739) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 10.6702 ± 0.0332206 (W 0.0342897) | 27.6207 ± 0.114615 (W 0.277205) | - | 🔵 colmat 2.59× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 16.4149 ± 0.0910671 (W 0.0441524) | 22.1889 ± 1.98715 (W 0.391025) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.388043 ± 0.00125478 (W 0.00275446) | 1.0956 ± 0.0041346 (W 0.00773117) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.3058 ± 0.0047177 (W 0.00262999) | 4.92549 ± 0.0454286 (W 0.0215373) | - | 🔵 colmat 3.77× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.10991 ± 0.00646691 (W 0.0091127) | 20.2099 ± 0.0485884 (W 0.118203) | - | 🔵 colmat 3.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.51012 ± 0.0164737 (W 0.00563064) | 4.05494 ± 0.00986931 (W 0.0108968) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.51894 ± 0.0241048 (W 0.00436973) | 4.07941 ± 0.0197328 (W 0.0112698) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.5451 ± 0.0283845 (W 0.0340751) | 21.2784 ± 0.214782 (W 0.0399385) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 10.603 ± 0.0734197 (W 0.0255054) | 21.192 ± 0.148594 (W 0.112702) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 48.9731 ± 0.0421531 (W 0.0551319) | 74.2161 ± 0.321755 (W 1.58603) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.274 ± 0.0728609 (W 0.0834494) | 74.4272 ± 0.273013 (W 0.473837) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.27617 ± 0.00386605 (W 0.00488987) | 2.97525 ± 0.00719898 (W 0.00605612) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.27951 ± 0.00272331 (W 0.00483422) | 2.97432 ± 0.00728602 (W 0.00491709) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.67077 ± 0.0195708 (W 0.0147629) | 16.4615 ± 0.199091 (W 0.0220756) | - | 🔵 colmat 2.90× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.68086 ± 0.0181598 (W 0.0163162) | 16.4377 ± 0.126983 (W 0.0220914) | - | 🔵 colmat 2.89× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 25.9871 ± 0.0735162 (W 0.0420134) | 51.0277 ± 0.0989104 (W 0.0478801) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 26.2298 ± 0.0738595 (W 0.640048) | 51.1412 ± 0.0499071 (W 0.132307) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009158 ± 8.22456e-05 (W 7.1114e-05) | 0.007624 ± 3.27159e-05 (W 7.0194e-05) | 0.007093 ± 6.245e-05 (W 8.007e-05) | 🔴 colmat 1.20× slower than `faer` | 🔴 colmat 1.29× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.021378 ± 9.32094e-05 (W 0.000125301) | 0.023784 ± 0.000135242 (W 0.000107314) | 0.020608 ± 6.90242e-05 (W 9.0445e-05) | 🔵 colmat 1.11× faster than `faer` | · |  |
| `triangular_solve/n512/b1` | 0.050805 ± 0.000223232 (W 0.000273843) | 0.059551 ± 0.000174046 (W 0.000704279) | 0.074378 ± 0.000630432 (W 0.00238375) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-aocl` |  |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 3.05501 ± 0.021764 (W 0.0037308) | 3.41172 ± 0.0112165 (W 0.00586988) | 3.58166 ± 0.0118565 (W 0.00829013) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 3.07028 ± 0.0193155 (W 0.00234824) | 3.40951 ± 0.0179828 (W 0.00413178) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 17.0367 ± 0.0984567 (W 0.00810102) | 17.5844 ± 0.0726975 (W 0.0110709) | 20.5209 ± 0.0907212 (W 0.0194774) | · | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 17.0913 ± 0.0424515 (W 0.0196089) | 17.6575 ± 0.0755822 (W 0.0120579) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 104.686 ± 0.607901 (W 0.295755) | 105.112 ± 0.357759 (W 0.207101) | 139.307 ± 0.467824 (W 0.323129) | · | 🔵 colmat 1.33× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 104.937 ± 0.347536 (W 0.134024) | 105.51 ± 0.263534 (W 0.127463) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.626809 ± 0.00345809 (W 0.00248212) | 0.754387 ± 0.00469146 (W 0.00539584) | 21.989 ± 0.116892 (W 0.0272156) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 35.08× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.735052 ± 0.00191653 (W 0.00329065) | 0.843012 ± 0.00769899 (W 0.00341702) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 4.48035 ± 0.0139279 (W 0.0072956) | 5.92587 ± 0.0608832 (W 0.00912136) | 172.324 ± 0.751542 (W 0.125921) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 38.46× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 5.02719 ± 0.0286255 (W 0.0550486) | 6.29873 ± 0.039906 (W 0.0139841) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 36.8337 ± 0.34295 (W 0.0387632) | 43.0324 ± 0.563351 (W 0.0324997) | 1369.59 ± 3.9591 (W 1.32517) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 37.18× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 37.3079 ± 0.180144 (W 0.0533331) | 42.8606 ± 0.139782 (W 0.113324) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.04027 ± 0.00444037 (W 0.00187008) | 1.03986 ± 0.00107756 (W 0.00221393) | 1.30164 ± 0.00283134 (W 0.00554877) | · | 🔵 colmat 1.25× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.03874 ± 0.00538226 (W 0.00120563) | 1.03989 ± 0.000764572 (W 0.00100693) | 1.2941 ± 0.00649618 (W 0.00358535) | · | 🔵 colmat 1.25× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.47623 ± 0.01963 (W 0.00968821) | 7.46096 ± 0.0227518 (W 0.00445806) | 9.19436 ± 0.0617602 (W 0.0278648) | · | 🔵 colmat 1.23× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.46538 ± 0.0311008 (W 0.0075844) | 7.46233 ± 0.024329 (W 0.0135945) | 8.30632 ± 0.0602101 (W 0.0101605) | · | 🔵 colmat 1.11× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.5068 ± 0.255821 (W 0.0726197) | 56.4145 ± 0.259127 (W 0.0579911) | 62.49 ± 0.255026 (W 0.0806217) | · | 🔵 colmat 1.11× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.9714 ± 0.182531 (W 0.382034) | 56.3977 ± 0.231598 (W 0.0634543) | 52.1485 ± 0.342462 (W 0.0821072) | · | 🔴 colmat 1.09× slower than `lapack-aocl` |  |
| `qr/n128/b1` | 1.30125 ± 0.00622242 (W 0.00273477) | 1.52927 ± 0.0091719 (W 0.00533758) | 2.05358 ± 0.0104462 (W 0.00634037) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.58× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 1.46921 ± 0.0058048 (W 0.00347871) | 1.70616 ± 0.00909918 (W 0.00384966) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 9.91674 ± 0.0354075 (W 0.00738304) | 10.0674 ± 0.025649 (W 0.0167515) | 13.4447 ± 0.0354992 (W 0.0183507) | · | 🔵 colmat 1.36× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 10.8242 ± 0.0382676 (W 0.0116386) | 10.9875 ± 0.0379758 (W 0.0133723) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 68.3388 ± 0.208385 (W 0.250713) | 68.6 ± 0.193111 (W 0.0487295) | 86.0331 ± 0.516452 (W 0.103416) | · | 🔵 colmat 1.26× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 72.824 ± 0.246571 (W 0.0773527) | 73.7895 ± 0.538285 (W 0.182121) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.505433 ± 0.00367273 (W 0.00231236) | 0.756802 ± 0.00578474 (W 0.00281864) | 0.7668 ± 0.00646974 (W 0.00279086) | 🔵 colmat 1.50× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 4.39652 ± 0.017292 (W 0.00506221) | 4.5451 ± 0.0221479 (W 0.00354662) | 5.22182 ± 0.015222 (W 0.00475489) | · | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 28.3642 ± 0.0781949 (W 0.0268264) | 28.8319 ± 0.119862 (W 0.0255762) | 34.0423 ± 0.168628 (W 0.0454759) | · | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 4.71469 ± 0.0184836 (W 0.00363875) | 5.66652 ± 0.0332769 (W 0.00733238) | 5.46536 ± 0.0339052 (W 0.00826797) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.71238 ± 0.0343319 (W 0.0104772) | 5.69621 ± 0.019516 (W 0.00454301) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.1227 ± 0.187782 (W 0.0407279) | 33.4978 ± 0.139043 (W 0.0610115) | 36.0345 ± 0.137061 (W 0.047973) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 29.4031 ± 0.123456 (W 0.067677) | 33.5663 ± 0.12007 (W 0.0401726) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 194.171 ± 1.23982 (W 0.367071) | 206.956 ± 0.448525 (W 0.503668) | 220.56 ± 0.78002 (W 0.69365) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.14× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 194.852 ± 0.887332 (W 0.521266) | 207.483 ± 0.477071 (W 0.219614) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.59188 ± 0.00618799 (W 0.00561539) | 3.41286 ± 0.0163952 (W 0.00681229) | 2.08437 ± 0.00150522 (W 0.00155674) | 🔵 colmat 2.14× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.59491 ± 0.00558154 (W 0.0044809) | 3.43178 ± 0.0166658 (W 0.00563671) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.1374 ± 0.0603045 (W 0.0356337) | 18.9139 ± 0.115226 (W 0.0215209) | 15.8676 ± 0.118948 (W 0.0182702) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 1.42× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 11.2552 ± 0.146725 (W 0.0431322) | 18.7839 ± 0.0997318 (W 0.0187401) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 83.7305 ± 0.273434 (W 0.0766293) | 108.779 ± 0.817419 (W 0.551538) | 94.523 ± 0.411116 (W 0.150697) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 83.9307 ± 0.492964 (W 0.2118) | 108.48 ± 0.481058 (W 0.134654) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.008035 ± 7.26659e-05 (W 0.000140262) | 0.008155 ± 8.92767e-05 (W 4.06473e-05) | 0.006562 ± 3.00056e-05 (W 6.30175e-05) | · | 🔴 colmat 1.22× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024456 ± 0.000326819 (W 6.65372e-05) | 0.024306 ± 0.000187538 (W 0.000187958) | 0.019266 ± 0.000147986 (W 8.44393e-05) | · | 🔴 colmat 1.27× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.090799 ± 0.000387485 (W 0.000511661) | 0.090028 ± 0.00217468 (W 0.00262647) | 0.073047 ± 0.000487977 (W 0.000507173) | · | 🔴 colmat 1.24× slower than `lapack-aocl` |  |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 1.22548 ± 0.00426707 (W 0.00287464) | 2.96595 ± 0.0154746 (W 0.00578809) | 3.85131 ± 0.0677482 (W 0.00593232) | 🔵 colmat 2.42× faster than `faer` | 🔵 colmat 3.14× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.22904 ± 0.000383344 (W 0.00317721) | 2.95982 ± 0.00626065 (W 0.004101) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.43779 ± 0.0159677 (W 0.00610924) | 11.4702 ± 0.044561 (W 0.0239994) | 21.6263 ± 0.0981865 (W 0.0298345) | 🔵 colmat 2.11× faster than `faer` | 🔵 colmat 3.98× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.45543 ± 0.00239783 (W 0.0069081) | 11.48 ± 0.0153235 (W 0.00991197) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 30.1641 ± 0.153974 (W 0.0440833) | 44.8371 ± 0.074445 (W 0.0530442) | 81.4975 ± 0.0508092 (W 0.354267) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 2.70× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 30.2091 ± 0.185498 (W 0.0195173) | 44.9326 ± 0.13891 (W 0.029225) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.38042 ± 0.00753394 (W 0.00301655) | 0.76561 ± 0.00166154 (W 0.00628076) | 22.1279 ± 0.0776474 (W 0.0198806) | 🔵 colmat 2.01× faster than `faer` | 🔵 colmat 58.17× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.498549 ± 0.00708656 (W 0.00276815) | 0.742905 ± 0.00511145 (W 0.00775764) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.70795 ± 0.0104217 (W 0.00409663) | 5.91944 ± 0.00805473 (W 0.0114243) | 174.465 ± 1.00938 (W 0.313983) | 🔵 colmat 3.47× faster than `faer` | 🔵 colmat 102.15× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.31086 ± 0.0070988 (W 0.00762598) | 5.75035 ± 0.0194494 (W 0.00996503) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.3926 ± 0.0221443 (W 0.0270703) | 26.0325 ± 0.0739404 (W 0.0718853) | 1358.49 ± 3.37039 (W 1.19951) | 🔵 colmat 2.50× faster than `faer` | 🔵 colmat 130.72× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.0932 ± 0.0494796 (W 0.0180018) | 24.5736 ± 0.00813025 (W 0.0484977) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.286129 ± 0.000302075 (W 0.000804815) | 0.283101 ± 0.000720253 (W 0.00245302) | 0.665406 ± 0.00390362 (W 0.00534047) | · | 🔵 colmat 2.33× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.285732 ± 0.00153711 (W 0.000661481) | 0.283562 ± 0.00127105 (W 0.00249861) | 0.480276 ± 0.00304086 (W 0.00328254) | · | 🔵 colmat 1.68× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.95686 ± 0.00522261 (W 0.00435798) | 1.95124 ± 0.00680344 (W 0.00510975) | 3.96338 ± 0.0247826 (W 0.0108311) | · | 🔵 colmat 2.03× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.95165 ± 0.00793295 (W 0.00190595) | 1.94641 ± 0.00407994 (W 0.00309721) | 2.61898 ± 0.0610074 (W 0.00481384) | · | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.7237 ± 0.0154624 (W 0.0342012) | 14.7899 ± 0.0537957 (W 0.0271082) | 22.5494 ± 0.11125 (W 0.0573398) | · | 🔵 colmat 1.53× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.7496 ± 0.0344809 (W 0.0215843) | 14.7782 ± 0.0342455 (W 0.0187005) | 14.8242 ± 0.0861763 (W 0.0163532) | · | · |  |
| `qr/n128/b1` | 0.697945 ± 0.00229748 (W 0.00232783) | 0.972054 ± 0.00255585 (W 0.00297294) | 0.962246 ± 0.0021625 (W 0.00223886) | 🔵 colmat 1.39× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.908679 ± 0.00170486 (W 0.00231098) | 1.17582 ± 0.00373373 (W 0.00387243) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.0248 ± 0.00568249 (W 0.00353281) | 4.06202 ± 0.0108487 (W 0.00693141) | 4.6497 ± 0.0147424 (W 0.00850891) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 4.0438 ± 0.00140072 (W 0.00407807) | 5.08083 ± 0.0292521 (W 0.0113669) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 19.1331 ± 0.0685058 (W 0.0235207) | 21.1442 ± 0.0725846 (W 0.0424397) | 29.4422 ± 0.0843471 (W 0.0367028) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 24.2394 ± 0.0945604 (W 0.0325717) | 26.3101 ± 0.114501 (W 0.0146179) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.417739 ± 0.0015685 (W 0.00308009) | 0.665871 ± 0.00126645 (W 0.00382297) | 0.476966 ± 0.000954928 (W 0.0035537) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 1.14× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.55134 ± 0.00287153 (W 0.00226248) | 2.58456 ± 0.0101289 (W 0.00596272) | 2.04391 ± 0.00673037 (W 0.00819954) | 🔵 colmat 1.67× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 8.82379 ± 0.0168662 (W 0.0136935) | 10.936 ± 0.0162513 (W 0.0101244) | 14.4287 ± 0.0430396 (W 0.0247988) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 1.64× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 2.693 ± 0.0172386 (W 0.00379995) | 4.02092 ± 0.0203025 (W 0.0145008) | 5.29569 ± 0.0339944 (W 4.10779) | 🔵 colmat 1.49× faster than `faer` | `?` not evidence vs `lapack-aocl`: a row's `W` is as large as the difference (rerun) |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.70303 ± 0.0143748 (W 0.00631349) | 4.02677 ± 0.00688468 (W 0.0123075) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.6674 ± 0.0196219 (W 0.0279063) | 21.3666 ± 0.0915243 (W 0.042689) | 21.997 ± 0.0514269 (W 5.24002) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 1.74× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 12.6721 ± 0.0373925 (W 0.0300799) | 21.2386 ± 0.0137709 (W 0.0548799) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 67.8663 ± 0.202819 (W 0.0914389) | 85.7408 ± 0.14001 (W 0.07018) | 110.172 ± 0.28201 (W 0.296131) | 🔵 colmat 1.26× faster than `faer` | 🔵 colmat 1.62× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 68.0353 ± 0.0499371 (W 0.0329845) | 86.3109 ± 0.037732 (W 0.794542) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.25131 ± 0.00704412 (W 0.00287085) | 2.94087 ± 0.0151168 (W 0.00420617) | 3.24133 ± 0.0540587 (W 7.13359) | 🔵 colmat 2.35× faster than `faer` | `?` not evidence vs `lapack-aocl`: a row's `W` is as large as the difference (rerun) |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.26467 ± 0.00606647 (W 0.00339373) | 2.93403 ± 0.00933287 (W 0.0131093) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.11572 ± 0.0162486 (W 0.0254467) | 15.9196 ± 0.0443211 (W 0.0169049) | 12.8896 ± 0.025517 (W 0.371494) | 🔵 colmat 2.60× faster than `faer` | 🔵 colmat 2.11× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.12903 ± 0.0200678 (W 0.0261306) | 15.8818 ± 0.0326229 (W 0.018713) | - | 🔵 colmat 2.59× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 32.8029 ± 0.137579 (W 0.0795573) | 56.5112 ± 0.161161 (W 0.0818283) | 58.2147 ± 0.729244 (W 0.106746) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 33.0715 ± 0.0982625 (W 0.105663) | 56.3626 ± 0.0907774 (W 0.0527388) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009086 ± 0.000135011 (W 5.02991e-05) | 0.008296 ± 0.000134318 (W 5.59044e-05) | 0.006552 ± 6.35085e-06 (W 3.75007e-05) | 🔴 colmat 1.10× slower than `faer` | 🔴 colmat 1.39× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.022703 ± 0.000224702 (W 0.000301278) | 0.024864 ± 0.000331453 (W 0.000116937) | 0.019667 ± 0.000232436 (W 0.000116384) | 🔵 colmat 1.10× faster than `faer` | 🔴 colmat 1.15× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.058419 ± 0.000480851 (W 0.000546555) | 0.066494 ± 0.000620282 (W 0.000167048) | 0.072835 ± 0.000472017 (W 0.000418026) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-aocl` |  |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.908272 ± 0.00650263 (W 0.00407151) | 2.98043 ± 0.000569895 (W 0.00650657) | 4.02238 ± 0.00318793 (W 0.00619084) | 🔵 colmat 3.28× faster than `faer` | 🔵 colmat 4.43× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.920024 ± 0.0031082 (W 0.00188655) | 2.98286 ± 0.00108923 (W 0.00485002) | - | 🔵 colmat 3.24× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.67122 ± 0.00475737 (W 0.00950205) | 11.2506 ± 0.0466727 (W 0.0513137) | 22.7628 ± 0.589098 (W 0.0233462) | 🔵 colmat 3.06× faster than `faer` | 🔵 colmat 6.20× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.71539 ± 0.0220385 (W 0.00771467) | 11.3103 ± 0.0359572 (W 0.0565094) | - | 🔵 colmat 3.04× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 18.4157 ± 0.0377771 (W 0.0206708) | 38.9684 ± 0.228787 (W 0.252947) | 73.3791 ± 1.06192 (W 0.249412) | 🔵 colmat 2.12× faster than `faer` | 🔵 colmat 3.98× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 18.4737 ± 0.0287163 (W 0.0250817) | 39.0762 ± 0.248852 (W 0.166177) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.354751 ± 0.0040446 (W 0.00269294) | 0.761931 ± 0.00276434 (W 0.00786848) | 22.0506 ± 0.0367166 (W 0.0159523) | 🔵 colmat 2.15× faster than `faer` | 🔵 colmat 62.16× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.475071 ± 0.00285406 (W 0.00461501) | 0.742053 ± 0.00323045 (W 0.00384772) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.31751 ± 0.00511893 (W 0.00403119) | 5.92617 ± 0.0573376 (W 0.0185788) | 173.184 ± 0.373131 (W 0.168454) | 🔵 colmat 4.50× faster than `faer` | 🔵 colmat 131.45× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.92655 ± 0.0172972 (W 0.00571883) | 5.72599 ± 0.0524615 (W 0.0149525) | - | 🔵 colmat 2.97× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.29776 ± 0.0247508 (W 0.00809943) | 23.4363 ± 0.0653936 (W 0.0518708) | 1351.6 ± 2.66315 (W 0.898413) | 🔵 colmat 3.72× faster than `faer` | 🔵 colmat 214.62× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 9.27522 ± 0.0577156 (W 0.0517922) | 22.2919 ± 0.0533501 (W 0.0590566) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.180621 ± 0.000332277 (W 0.000551845) | 0.274071 ± 0.00311729 (W 0.00295335) | 0.589018 ± 0.000823836 (W 0.00538742) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 3.26× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.180577 ± 0.000885183 (W 0.000528722) | 0.271892 ± 0.00214116 (W 0.00213042) | 0.381571 ± 0.00307743 (W 0.00470836) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 2.11× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.02525 ± 0.00397455 (W 0.00417311) | 1.3012 ± 0.0924739 (W 0.0805679) | 3.17465 ± 0.0119994 (W 0.0148713) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 3.10× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.02538 ± 0.00224136 (W 0.00327768) | 1.15019 ± 0.0998231 (W 0.0961112) | 1.57577 ± 0.00737384 (W 0.0094747) | 🔵 colmat 1.12× faster than `faer` (marginal: spread 0.82 of the difference) | 🔵 colmat 1.54× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 8.03425 ± 0.0295244 (W 0.0515594) | 10.0918 ± 1.52019 (W 1.19467) | 15.5461 ± 0.0122044 (W 0.0179841) | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.75 of the difference) | 🔵 colmat 1.93× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.97059 ± 0.0559053 (W 0.171662) | 7.84168 ± 1.65487 (W 1.48317) | 8.74316 ± 0.0215919 (W 0.0230072) | · | 🔵 colmat 1.10× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1` | 0.545065 ± 0.00438222 (W 0.00475907) | 1.07959 ± 0.00692365 (W 0.00388711) | 0.841818 ± 0.000535204 (W 0.0104796) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.774276 ± 0.00597207 (W 0.003183) | 1.2856 ± 0.00280877 (W 0.00361046) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.0384 ± 0.0020364 (W 0.00295223) | 4.00386 ± 0.0146148 (W 0.0441576) | 3.08432 ± 0.00838572 (W 0.0135935) | 🔵 colmat 1.96× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.07081 ± 0.00594783 (W 0.0105908) | 4.20626 ± 0.394348 (W 0.0456761) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 10.142 ± 0.0444332 (W 0.0181978) | 17.1759 ± 0.12197 (W 0.44275) | 17.1689 ± 0.0656656 (W 0.0337674) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 1.69× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 15.3092 ± 0.0981588 (W 0.0299424) | 18.2802 ± 2.5339 (W 1.38627) | - | 🔵 colmat 1.19× faster than `faer` (marginal: spread 0.89 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.37011 ± 0.0133388 (W 0.00269598) | 0.762243 ± 0.00178534 (W 0.00607103) | 0.455481 ± 0.00293118 (W 0.00462277) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr_factor/n256/b1` | 1.24983 ± 0.00241751 (W 0.00234898) | 2.73965 ± 0.00195468 (W 0.0494071) | 1.52275 ± 0.00794353 (W 0.00900001) | 🔵 colmat 2.19× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 4.79023 ± 0.0077498 (W 0.036107) | 9.8753 ± 0.103301 (W 0.133233) | 8.50477 ± 0.0153681 (W 0.0261434) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 2.47517 ± 0.00721144 (W 0.00513439) | 4.03306 ± 0.00464901 (W 0.0075812) | 6.95858 ± 0.0928731 (W 4.90684) | 🔵 colmat 1.63× faster than `faer` | `?` not evidence vs `lapack-aocl`: a row's `W` is as large as the difference (rerun) |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.49487 ± 0.0112161 (W 0.0115473) | 4.05453 ± 0.0050732 (W 0.00711429) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.2954 ± 0.0309245 (W 0.0119411) | 20.7987 ± 0.084308 (W 0.04072) | 22.0291 ± 0.148764 (W 4.01771) | 🔵 colmat 2.02× faster than `faer` | 🔵 colmat 2.14× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 10.3956 ± 0.0274436 (W 0.0338949) | 20.9596 ± 0.0130347 (W 0.111774) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 48.3107 ± 0.0740743 (W 0.122384) | 73.1122 ± 0.376411 (W 0.403643) | 92.2128 ± 0.592521 (W 0.32609) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 1.91× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 48.4869 ± 0.0252785 (W 0.103318) | 73.9271 ± 0.374217 (W 0.540481) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.25548 ± 0.0102854 (W 0.00416712) | 2.93971 ± 0.00826923 (W 0.00573879) | 5.03149 ± 2.19099 (W 4.65009) | 🔵 colmat 2.34× faster than `faer` | `?` not evidence vs `lapack-aocl`: a row's `W` is as large as the difference (rerun) | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/n128/b1 [api-inclusive]` | 1.24909 ± 0.00897169 (W 0.00298534) | 2.9344 ± 0.00484542 (W 0.00552008) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.50731 ± 0.0285306 (W 0.0211996) | 15.9776 ± 0.180234 (W 0.0168823) | 14.4593 ± 0.0798733 (W 6.63358) | 🔵 colmat 2.90× faster than `faer` | 🔵 colmat 2.63× faster than `lapack-aocl` (marginal: spread 0.74 of the difference) |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.51012 ± 0.0120425 (W 0.0184017) | 16.003 ± 0.0236713 (W 0.0125273) | - | 🔵 colmat 2.90× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 25.2418 ± 0.0430015 (W 0.0631378) | 50.0784 ± 0.199192 (W 0.0640584) | 52.5558 ± 0.1285 (W 0.802713) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 2.08× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 25.2088 ± 0.0579788 (W 0.0503153) | 50.4118 ± 0.146855 (W 0.537664) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009057 ± 4.20357e-05 (W 0.0001205) | 0.008285 ± 0.000231172 (W 8.68689e-05) | 0.006902 ± 3.59212e-05 (W 4.85819e-05) | 🔴 colmat 1.09× slower than `faer` | 🔴 colmat 1.31× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.022933 ± 0.000208426 (W 0.000169204) | 0.024915 ± 9.26841e-05 (W 0.00144844) | 0.02225 ± 0.000232281 (W 9.37561e-05) | 🔵 colmat 1.09× faster than `faer` (marginal: spread 0.73 of the difference) | · |  |
| `triangular_solve/n512/b1` | 0.057168 ± 0.00131798 (W 0.000500529) | 0.065793 ± 0.000640388 (W 0.00199391) | 0.082057 ± 0.000562344 (W 0.00185581) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-aocl` |  |

Not applicable to this population's shapes and reported as skipped: `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.859708 ± 0.00565302 (W 0.00256458) | 1.19791 ± 0.0890059 (W 0.00687133) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.743886 ± 0.00744653 (W 0.00362771) | 0.824972 ± 0.00975049 (W 0.00363584) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 51.3576 ± 0.240761 (W 0.355813) | 99.047 ± 1.36979 (W 1.38938) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 48.0457 ± 0.710578 (W 0.500117) | 52.8744 ± 1.38906 (W 1.10895) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.552954 ± 0.00319998 (W 0.00228813) | 0.555809 ± 0.00208864 (W 0.00500904) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.557598 ± 0.00143781 (W 0.00348594) | 0.556787 ± 0.00537642 (W 0.00494739) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.1224 ± 0.1005 (W 0.11723) | 35.0531 ± 0.0481791 (W 0.102873) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.0581 ± 0.0801296 (W 0.0685292) | 35.023 ± 0.0257372 (W 0.0889515) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.45285 ± 0.0127171 (W 0.00407118) | 2.43665 ± 0.0118227 (W 0.00708171) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.69878 ± 0.00406713 (W 0.00311989) | 1.80849 ± 0.0141688 (W 0.0068592) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 67.4652 ± 0.408034 (W 0.224709) | 114.333 ± 0.296609 (W 0.268801) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 71.4762 ± 0.226097 (W 0.452273) | 75.558 ± 0.334699 (W 0.122612) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.924014 ± 0.00463324 (W 0.00276793) | 1.86661 ± 0.00119245 (W 0.00496822) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 31.1524 ± 0.369101 (W 0.323642) | 77.5595 ± 0.263974 (W 0.140258) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.0186 ± 0.00560847 (W 0.00681232) | 2.49093 ± 0.00932827 (W 0.00685567) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.02145 ± 0.0117459 (W 0.00540966) | 2.50863 ± 0.00748313 (W 0.00589912) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 83.3385 ± 0.180622 (W 0.101391) | 88.3496 ± 0.228951 (W 0.42411) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 83.003 ± 0.302111 (W 0.0901463) | 88.7343 ± 0.287109 (W 0.306677) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.02272 ± 0.00292625 (W 0.0103274) | 1.48157 ± 0.00488922 (W 0.00469667) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.0232 ± 0.00198309 (W 0.00568438) | 1.49177 ± 0.00779528 (W 0.00711781) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 37.4255 ± 0.121146 (W 0.0557133) | 44.6868 ± 0.151981 (W 0.0536317) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 37.4184 ± 0.079105 (W 0.0614936) | 44.7612 ± 0.264349 (W 0.0741654) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.475716 ± 0.00659908 (W 0.00855572) | 1.20439 ± 0.00488531 (W 0.00417365) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.505624 ± 0.00429514 (W 0.00316224) | 0.813114 ± 0.00632079 (W 0.00426965) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 14.3385 ± 0.140927 (W 0.0326224) | 48.2014 ± 0.283749 (W 0.085479) | - | 🔵 colmat 3.36× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 15.2046 ± 0.0949682 (W 0.0359853) | 22.8474 ± 1.01935 (W 0.809734) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.167389 ± 0.00195729 (W 0.00334623) | 0.161107 ± 0.00468502 (W 0.00272455) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.165966 ± 0.00151711 (W 0.00298623) | 0.154964 ± 0.0017833 (W 0.00276006) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.48405 ± 0.0058099 (W 0.0215818) | 9.37923 ± 0.0293855 (W 0.0277704) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.53391 ± 0.014892 (W 0.0300461) | 9.39105 ± 0.0277248 (W 0.0417714) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.718004 ± 0.006518 (W 0.00429942) | 3.24983 ± 0.10061 (W 0.0984791) | - | 🔵 colmat 4.53× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 1.05325 ± 0.00712061 (W 0.00449638) | 1.39873 ± 0.0292403 (W 0.0341826) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 23.2941 ± 0.111965 (W 0.101298) | 84.7859 ± 0.149542 (W 0.202966) | - | 🔵 colmat 3.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 30.536 ± 0.0976951 (W 0.109887) | 40.0872 ± 3.32201 (W 1.48536) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.558169 ± 0.00239375 (W 0.00286508) | 3.05386 ± 0.153441 (W 0.111747) | - | 🔵 colmat 5.47× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 12.7755 ± 0.0746931 (W 0.032026) | 74.6869 ± 0.147954 (W 0.207126) | - | 🔵 colmat 5.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.30554 ± 0.00241005 (W 0.0105487) | 2.15403 ± 0.00659158 (W 0.00404925) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.31686 ± 0.0182815 (W 0.00571493) | 2.17045 ± 0.0149287 (W 0.00954992) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 36.5731 ± 0.132762 (W 0.0724458) | 47.4933 ± 0.0408835 (W 0.0520318) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 37.0466 ± 0.195808 (W 0.0934046) | 47.6355 ± 0.070654 (W 0.298417) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.854542 ± 0.0129476 (W 0.00786587) | 1.54355 ± 0.0142471 (W 0.00745531) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.865792 ± 0.00598115 (W 0.00927366) | 1.54196 ± 0.0084958 (W 0.00848808) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 21.8629 ± 0.074409 (W 0.0322121) | 32.8029 ± 0.0395173 (W 0.0549136) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 21.9285 ± 0.0651647 (W 0.0270337) | 32.7697 ± 0.0718012 (W 0.0645229) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.442872 ± 0.0025941 (W 0.00578618) | 1.21138 ± 0.0049603 (W 0.00412392) | - | 🔵 colmat 2.74× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.484907 ± 0.00180419 (W 0.00719721) | 0.811952 ± 0.00279697 (W 0.00776715) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 8.33535 ± 0.0115116 (W 0.0533196) | 41.4158 ± 0.0807593 (W 0.0631299) | - | 🔵 colmat 4.97× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 10.1764 ± 0.0747653 (W 0.0595894) | 16.966 ± 0.876172 (W 0.812735) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.105513 ± 0.00123795 (W 0.00393715) | 0.144015 ± 0.00645333 (W 0.00705464) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.106807 ± 0.000613109 (W 0.0026805) | 0.152803 ± 0.00760287 (W 0.00662197) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.74753 ± 0.0254102 (W 0.0545126) | 6.64047 ± 0.919686 (W 0.746262) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.72893 ± 0.0192806 (W 0.0503304) | 5.05283 ± 0.935103 (W 0.754197) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.616972 ± 0.0011417 (W 0.00301487) | 3.12977 ± 0.0239679 (W 0.102072) | - | 🔵 colmat 5.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.952514 ± 0.00390741 (W 0.00291227) | 1.23243 ± 0.186095 (W 0.0296478) | - | 🔵 colmat 1.29× faster than `faer` (marginal: spread 0.68 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 15.6774 ± 0.0621771 (W 0.0510327) | 80.1279 ± 0.144764 (W 0.717787) | - | 🔵 colmat 5.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 24.112 ± 0.119175 (W 0.104634) | 33.3506 ± 1.62041 (W 2.97884) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.496767 ± 0.00243554 (W 0.00273489) | 3.18421 ± 0.191231 (W 0.186318) | - | 🔵 colmat 6.41× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 9.12614 ± 0.0730572 (W 0.0278733) | 71.5746 ± 0.0802812 (W 0.268546) | - | 🔵 colmat 7.84× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.23297 ± 0.0204723 (W 0.00272481) | 2.25378 ± 0.0175453 (W 0.00836381) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.2303 ± 0.00625957 (W 0.0068797) | 2.25648 ± 0.0016691 (W 0.00752108) | - | 🔵 colmat 1.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 29.8332 ± 0.117769 (W 0.0358453) | 46.3099 ± 0.191662 (W 0.207664) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 29.9766 ± 0.0813215 (W 0.167668) | 46.6578 ± 1.31535 (W 0.273564) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.824085 ± 0.00775227 (W 0.00542783) | 1.63778 ± 0.00341531 (W 0.00478143) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.82278 ± 0.0125009 (W 0.00324321) | 1.62998 ± 0.00804669 (W 0.005488) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.0028 ± 0.0442521 (W 0.0366674) | 33.5254 ± 0.0160025 (W 0.0977305) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.0133 ± 0.00259818 (W 0.087806) | 33.5928 ± 0.105373 (W 0.0562866) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.732795 ± 0.00537589 (W 0.00431911) | 0.801057 ± 0.00218378 (W 0.00304206) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.654195 ± 0.00830668 (W 0.00437876) | 0.727595 ± 0.00427928 (W 0.00184356) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 50.6112 ± 0.746305 (W 0.0831583) | 51.9596 ± 0.181322 (W 0.105014) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 45.7846 ± 0.773345 (W 0.216348) | 46.3745 ± 0.22324 (W 0.0643038) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.596836 ± 0.00115227 (W 0.00348118) | 0.595574 ± 0.00262982 (W 0.0030469) | 0.720242 ± 0.00432936 (W 0.00447254) | · | 🔵 colmat 1.21× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.595854 ± 0.00239693 (W 0.00284934) | 0.600379 ± 0.0029797 (W 0.00224318) | 2.10002 ± 0.00747745 (W 0.00361666) | · | 🔵 colmat 3.52× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.8556 ± 0.19938 (W 0.386104) | 35.8869 ± 0.0681288 (W 0.108107) | 38.5379 ± 0.0980945 (W 0.0943292) | · | 🔵 colmat 1.07× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.9568 ± 0.0600033 (W 0.0301606) | 35.9492 ± 0.15744 (W 0.0704324) | 48.443 ± 0.0771878 (W 0.0391446) | · | 🔵 colmat 1.35× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 1.25169 ± 0.00858827 (W 0.00451244) | 1.38102 ± 0.00437014 (W 0.00311679) | 2.8425 ± 0.00520876 (W 0.00884063) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 2.27× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.57299 ± 0.00335519 (W 0.00434369) | 1.71472 ± 0.0093956 (W 0.00420982) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 61.1683 ± 0.086487 (W 0.336459) | 62.2518 ± 0.276672 (W 0.362675) | 75.4834 ± 0.599186 (W 0.186619) | · | 🔵 colmat 1.23× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 68.5931 ± 0.217178 (W 0.427154) | 69.9556 ± 0.13306 (W 0.0533578) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.662193 ± 0.00168681 (W 0.00276729) | 0.794791 ± 0.000544033 (W 0.00345852) | 0.939242 ± 0.00323173 (W 0.00273689) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 1.42× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 24.7994 ± 0.0695176 (W 0.0407931) | 25.8857 ± 0.0483715 (W 0.0964478) | 27.9319 ± 0.0486385 (W 0.0462225) | · | 🔵 colmat 1.13× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.9431 ± 0.00449576 (W 0.00225142) | 2.44186 ± 0.00917626 (W 0.00577631) | 2.7381 ± 0.0068192 (W 0.00837281) | 🔵 colmat 1.26× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.95015 ± 0.00853389 (W 0.00830738) | 2.45474 ± 0.0066247 (W 0.0095787) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 78.04 ± 0.19553 (W 0.503583) | 82.8815 ± 0.20771 (W 0.269418) | 101.206 ± 0.373931 (W 0.19003) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 77.3776 ± 0.118405 (W 0.264742) | 83.3331 ± 0.203104 (W 0.155833) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.946322 ± 0.00544518 (W 0.0061018) | 1.41195 ± 0.00698632 (W 0.00239525) | 1.26821 ± 0.004677 (W 0.0033165) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.94304 ± 0.00167195 (W 0.00677406) | 1.41342 ± 0.00699666 (W 0.00646852) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.8786 ± 0.0645564 (W 0.046194) | 39.0346 ± 0.114926 (W 0.023565) | 37.9066 ± 0.160874 (W 0.0535773) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.9135 ± 0.135283 (W 0.0572511) | 38.953 ± 0.130504 (W 0.0316426) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.397674 ± 0.00176327 (W 0.00290894) | 0.811342 ± 0.00445787 (W 0.00346157) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.422456 ± 0.00106852 (W 0.00350696) | 0.73023 ± 0.0024865 (W 0.00332224) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 13.5807 ± 0.0741135 (W 0.0332596) | 15.9959 ± 0.0918379 (W 0.0460927) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 12.7965 ± 0.0223893 (W 0.0301783) | 14.937 ± 0.132525 (W 0.0314206) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.202471 ± 0.00170255 (W 0.000689206) | 0.19855 ± 0.00128546 (W 0.00202646) | 0.351012 ± 0.0016108 (W 0.00292261) | · | 🔵 colmat 1.73× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.201071 ± 0.00158424 (W 0.000857738) | 0.19661 ± 0.0026548 (W 0.00166836) | 2.09959 ± 0.00974206 (W 0.00238646) | · | 🔵 colmat 10.44× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 11.1583 ± 0.023627 (W 0.0269975) | 11.0898 ± 0.0213527 (W 0.0253483) | 12.0122 ± 0.276303 (W 0.022214) | · | 🔵 colmat 1.08× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 11.1698 ± 0.0159427 (W 0.0123108) | 11.1095 ± 0.0184461 (W 0.0157452) | 14.6183 ± 0.228621 (W 0.164045) | · | 🔵 colmat 1.31× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 0.672723 ± 0.00454338 (W 0.00287906) | 1.0334 ± 0.00330019 (W 0.00627903) | 2.84353 ± 0.0117674 (W 0.00587408) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 4.23× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.948697 ± 0.00188805 (W 0.00295469) | 1.38341 ± 0.00482107 (W 0.0314504) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 21.5783 ± 0.144771 (W 0.104846) | 26.5373 ± 0.0791658 (W 0.0697282) | 24.6968 ± 0.127404 (W 0.0451298) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 1.14× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 28.168 ± 0.0904476 (W 0.0884934) | 28.3487 ± 3.50862 (W 0.0542762) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.48738 ± 0.00149739 (W 0.00236285) | 0.841307 ± 0.00430836 (W 0.00509601) | 0.946276 ± 0.00508589 (W 0.00276774) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 1.94× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 10.6368 ± 0.0386626 (W 0.0127106) | 15.64 ± 0.0732642 (W 0.0161898) | 9.563 ± 0.0708052 (W 0.011454) | 🔵 colmat 1.47× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.25341 ± 0.00171946 (W 0.00345034) | 2.0979 ± 0.00940039 (W 0.00665465) | 2.53365 ± 0.0113799 (W 0.00742032) | 🔵 colmat 1.67× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.26016 ± 0.00557656 (W 0.00373585) | 2.11052 ± 0.00940618 (W 0.00524434) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 31.1182 ± 0.107852 (W 0.131911) | 41.9791 ± 0.118457 (W 0.197522) | 45.8781 ± 0.110777 (W 0.0784764) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.47× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 31.5047 ± 0.160059 (W 0.128732) | 42.1758 ± 0.082991 (W 0.371521) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.780817 ± 0.00172261 (W 0.00232149) | 1.46071 ± 0.00629366 (W 0.00452694) | 1.27209 ± 0.00194028 (W 0.00467166) | 🔵 colmat 1.87× faster than `faer` | 🔵 colmat 1.63× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.786428 ± 0.00152491 (W 0.00261678) | 1.46441 ± 0.00196531 (W 0.00351324) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 16.2827 ± 0.081589 (W 0.0332948) | 26.9183 ± 0.0648315 (W 0.0395431) | 19.4771 ± 0.0808327 (W 0.042037) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 16.3533 ± 0.02206 (W 0.0417477) | 26.9131 ± 0.108155 (W 0.0246123) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.375956 ± 0.000865646 (W 0.00426876) | 0.809667 ± 0.00126852 (W 0.00428865) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.402158 ± 0.0014176 (W 0.00301136) | 0.728628 ± 0.00171279 (W 0.00369397) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 8.10701 ± 0.0541701 (W 0.0215656) | 10.5789 ± 0.0945201 (W 0.0466152) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 7.90779 ± 0.0194684 (W 0.0505991) | 10.254 ± 0.136154 (W 0.0316571) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.128128 ± 0.000643975 (W 0.00205531) | 0.187024 ± 0.00395653 (W 0.00445773) | 0.298509 ± 0.00582646 (W 0.00544119) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 2.33× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.127948 ± 0.000457779 (W 0.00204027) | 0.187114 ± 0.00360255 (W 0.00789956) | 2.09571 ± 0.00834145 (W 0.00438548) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 16.38× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 8.83974 ± 0.0134173 (W 0.0215391) | 8.17539 ± 0.0660039 (W 0.721378) | 7.92871 ± 0.0791544 (W 0.0337264) | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | 🔴 colmat 1.11× slower than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 8.86044 ± 0.0122659 (W 0.0701941) | 8.15252 ± 0.0714959 (W 0.849377) | 8.92538 ± 0.0337359 (W 0.0238749) | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | · |  |
| `qr/1024x64/b1` | 0.558677 ± 0.00374003 (W 0.00475767) | 1.13506 ± 0.010498 (W 0.0181182) | 2.82824 ± 0.0213437 (W 0.0050812) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 5.06× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.844683 ± 0.0030958 (W 0.00467136) | 1.16556 ± 0.00853043 (W 0.0296418) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 15.2507 ± 0.0487625 (W 0.0493932) | 25.6908 ± 0.0508205 (W 0.140855) | 17.0022 ± 0.124579 (W 0.026965) | 🔵 colmat 1.68× faster than `faer` | 🔵 colmat 1.11× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 21.7965 ± 0.141507 (W 0.0721355) | 28.6165 ± 1.51494 (W 2.93002) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.449033 ± 0.0021163 (W 0.00286684) | 0.932966 ± 0.00330846 (W 0.00487311) | 0.943822 ± 0.0033338 (W 0.00321128) | 🔵 colmat 2.08× faster than `faer` | 🔵 colmat 2.10× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 7.77511 ± 0.0573608 (W 0.0252773) | 16.5864 ± 0.0511236 (W 0.0415513) | 8.05289 ± 0.0253845 (W 0.0345234) | 🔵 colmat 2.13× faster than `faer` | · |  |
| `svd_thin/1024x64/b1` | 1.14451 ± 0.00435934 (W 0.00265812) | 2.1847 ± 0.00300498 (W 0.00786661) | 2.57916 ± 0.0118821 (W 0.00807376) | 🔵 colmat 1.91× faster than `faer` | 🔵 colmat 2.25× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.15728 ± 0.0134704 (W 0.00424367) | 2.19414 ± 0.0100659 (W 0.00816705) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 24.0755 ± 0.151358 (W 0.048193) | 40.9446 ± 0.133809 (W 0.189132) | 40.9331 ± 0.107416 (W 0.0730148) | 🔵 colmat 1.70× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 24.3423 ± 0.0709418 (W 0.0500145) | 41.3181 ± 1.32916 (W 3.97914) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.748084 ± 0.000857298 (W 0.00243823) | 1.55224 ± 0.00594713 (W 0.00520636) | 1.26913 ± 0.00506059 (W 0.00225776) | 🔵 colmat 2.07× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.748874 ± 0.00293046 (W 0.00284028) | 1.54505 ± 0.0130211 (W 0.00505858) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 13.4207 ± 0.00186577 (W 0.0352171) | 27.8347 ± 0.103105 (W 0.0937331) | 20.5787 ± 0.357318 (W 0.0653274) | 🔵 colmat 2.07× faster than `faer` | 🔵 colmat 1.53× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.4527 ± 0.0913126 (W 0.0368189) | 27.9267 ± 0.100808 (W 0.0829262) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.90026 ± 0.0112852 (W 0.00677155) | 2.85989 ± 0.0533198 (W 0.0249734) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.77887 ± 0.0112381 (W 0.00880287) | 1.79364 ± 0.00347302 (W 0.00587381) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 156.139 ± 0.729398 (W 0.742744) | 271.743 ± 0.377094 (W 0.593911) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 151.529 ± 0.888577 (W 0.838875) | 155.519 ± 0.931373 (W 1.61638) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.21014 ± 0.00625443 (W 0.00259927) | 2.20711 ± 0.00361809 (W 0.00950075) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.20849 ± 0.0124276 (W 0.00428497) | 2.24487 ± 0.0249059 (W 0.00572855) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 138.03 ± 0.12143 (W 0.38479) | 138.218 ± 0.163451 (W 0.109167) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 137.843 ± 0.141174 (W 0.374045) | 138.254 ± 0.346682 (W 0.269416) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.59129 ± 0.0176651 (W 0.00446639) | 6.25981 ± 0.143704 (W 0.0127927) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 5.13036 ± 0.0119198 (W 0.00584638) | 5.30343 ± 0.0178006 (W 0.0143625) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 239.926 ± 0.215261 (W 0.408879) | 316.7 ± 0.0623716 (W 0.626853) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 251.561 ± 0.52662 (W 0.381803) | 259.687 ± 0.269953 (W 0.548752) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.33874 ± 0.0193606 (W 0.00362194) | 3.98379 ± 0.027569 (W 0.0177874) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 98.041 ± 0.587593 (W 0.323918) | 176.329 ± 0.527184 (W 0.620013) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 5.45312 ± 0.0175455 (W 0.00875594) | 6.01453 ± 0.047746 (W 0.0119307) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.47363 ± 0.0203278 (W 0.0092525) | 6.01526 ± 0.0110516 (W 0.0045631) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 273.725 ± 0.530989 (W 0.393925) | 280.592 ± 0.298146 (W 0.461154) | - | · | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 274.125 ± 0.607246 (W 0.489566) | 281.509 ± 0.235061 (W 0.582357) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.53773 ± 0.00674154 (W 0.00569997) | 3.1099 ± 0.0142199 (W 0.00943353) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.54674 ± 0.0129315 (W 0.00256632) | 3.11737 ± 0.0209895 (W 0.00481417) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 113.506 ± 1.02972 (W 0.390234) | 122.607 ± 0.0896095 (W 0.155491) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 115.645 ± 1.40381 (W 0.754585) | 122.482 ± 1.18909 (W 0.543986) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.835461 ± 0.00441822 (W 0.00598249) | 2.82396 ± 0.101004 (W 0.0149018) | - | 🔵 colmat 3.38× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.925289 ± 0.00345218 (W 0.0029637) | 1.78916 ± 0.0297481 (W 0.00321447) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 49.0796 ± 0.0212167 (W 0.154729) | 114.703 ± 0.371867 (W 0.716926) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 51.652 ± 0.236344 (W 0.18446) | 63.49 ± 0.593904 (W 0.515511) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.596635 ± 0.00265059 (W 0.00344762) | 0.586345 ± 0.00933764 (W 0.00215582) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.600693 ± 0.00441579 (W 0.00328296) | 0.57872 ± 0.00184333 (W 0.00150954) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 37.2153 ± 0.118441 (W 0.327232) | 36.7193 ± 0.0522722 (W 0.274742) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 37.2519 ± 0.0582966 (W 0.0512125) | 37.007 ± 0.174365 (W 0.293263) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.70479 ± 0.0160459 (W 0.005432) | 4.45238 ± 0.0870415 (W 0.069527) | - | 🔵 colmat 2.61× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.49027 ± 0.0143214 (W 0.0105037) | 3.01943 ± 0.0462823 (W 0.0344049) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 76.9689 ± 0.28312 (W 0.569687) | 167.931 ± 0.778047 (W 0.65963) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 91.4778 ± 0.551923 (W 0.117048) | 108.137 ± 7.02412 (W 0.14744) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.15643 ± 0.00835807 (W 0.00336706) | 3.86338 ± 0.086112 (W 0.0629099) | - | 🔵 colmat 3.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 35.7497 ± 0.0822618 (W 0.0786367) | 128.632 ± 0.473978 (W 0.462338) | - | 🔵 colmat 3.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.59775 ± 0.0175819 (W 0.00391815) | 3.64877 ± 0.0262041 (W 0.00567163) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.60162 ± 0.0190553 (W 0.00484962) | 3.69537 ± 0.0175808 (W 0.00764134) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 98.4116 ± 0.287587 (W 0.135693) | 117.338 ± 0.252543 (W 0.101752) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 99.0783 ± 0.403697 (W 0.168031) | 117.813 ± 0.405306 (W 0.228994) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.62545 ± 0.0290954 (W 0.00744866) | 2.44031 ± 0.0218568 (W 0.00467861) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.58928 ± 0.0313157 (W 0.00384735) | 2.44134 ± 0.00734917 (W 0.0103983) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 51.9458 ± 0.109016 (W 0.0788858) | 71.7895 ± 0.73239 (W 0.270052) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 51.891 ± 0.304167 (W 0.0932436) | 71.8315 ± 0.884426 (W 0.399639) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.7077 ± 0.0164958 (W 0.00326473) | 2.85199 ± 0.0851129 (W 0.0183356) | - | 🔵 colmat 4.03× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.790214 ± 0.01484 (W 0.00263711) | 1.77003 ± 0.0135835 (W 0.00454329) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 24.952 ± 0.0622167 (W 0.0617193) | 80.1821 ± 0.413437 (W 0.919549) | - | 🔵 colmat 3.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 28.4527 ± 0.22206 (W 0.172783) | 40.8137 ± 1.40202 (W 0.462301) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.314618 ± 0.00257087 (W 0.00235569) | 0.472342 ± 0.00290582 (W 0.0311426) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.31284 ± 0.00913127 (W 0.0021696) | 0.472807 ± 0.0394919 (W 0.0334101) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 24.0881 ± 0.0736205 (W 0.135576) | 25.4592 ± 0.118932 (W 2.78749) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 24.1617 ± 0.0739085 (W 0.207014) | 25.4966 ± 0.159069 (W 2.81098) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.28482 ± 0.00713199 (W 0.00328397) | 4.43555 ± 0.127474 (W 0.0880057) | - | 🔵 colmat 3.45× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 2.09129 ± 0.0103758 (W 0.0173845) | 2.43323 ± 0.048651 (W 0.265844) | - | 🔵 colmat 1.16× faster than `faer` (marginal: spread 0.78 of the difference) | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 46.5858 ± 0.0957131 (W 0.185194) | 151.714 ± 0.757424 (W 1.03506) | - | 🔵 colmat 3.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 67.3091 ± 0.132354 (W 0.288732) | 86.5837 ± 2.25412 (W 5.317) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.967536 ± 0.00589775 (W 0.00364744) | 3.72931 ± 0.0934187 (W 0.0329815) | - | 🔵 colmat 3.85× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 25.1305 ± 0.162622 (W 0.0761982) | 123.582 ± 0.664462 (W 1.02549) | - | 🔵 colmat 4.92× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.15831 ± 0.0600019 (W 0.00492694) | 3.63187 ± 0.0157548 (W 0.0257061) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.16779 ± 0.00973746 (W 0.00526446) | 3.73729 ± 0.0231139 (W 0.0361026) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 72.6884 ± 0.219128 (W 0.0739697) | 105.555 ± 0.72359 (W 1.58762) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 73.4703 ± 0.23016 (W 0.345824) | 105.113 ± 0.974498 (W 6.60306) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.43578 ± 0.020272 (W 0.00491552) | 2.49327 ± 0.0308563 (W 0.0123508) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.48226 ± 0.0303209 (W 0.00500548) | 2.48342 ± 0.00591381 (W 0.00723193) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 42.048 ± 0.253955 (W 0.157413) | 69.2856 ± 0.313703 (W 0.302616) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 42.0703 ± 0.0472323 (W 0.0684256) | 69.6561 ± 1.23443 (W 0.949287) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.69398 ± 0.00797868 (W 0.00505507) | 1.70191 ± 0.00246192 (W 0.00517717) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.65831 ± 0.0147287 (W 0.00415259) | 1.66731 ± 0.00405245 (W 0.00437193) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 152.214 ± 0.117622 (W 0.53245) | 151.193 ± 0.387547 (W 0.506506) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 146.076 ± 0.294099 (W 1.14145) | 143.184 ± 0.788969 (W 0.763543) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.26983 ± 0.00233906 (W 0.00605747) | 2.2678 ± 0.00689249 (W 0.00498124) | 2.68519 ± 0.0068117 (W 0.00555646) | · | 🔵 colmat 1.18× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.26628 ± 0.0104711 (W 0.011828) | 2.26212 ± 0.00768692 (W 0.010546) | 3.17891 ± 0.00548234 (W 0.00743964) | · | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 139.913 ± 0.0264228 (W 0.218848) | 138.876 ± 0.257061 (W 0.198411) | 144.45 ± 0.457772 (W 0.124596) | · | · |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 140.024 ± 0.186139 (W 0.213854) | 139.045 ± 0.212272 (W 0.233195) | 167.372 ± 0.543005 (W 0.276326) | · | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 4.31751 ± 0.00549602 (W 0.00175563) | 4.54029 ± 0.00689476 (W 0.0074476) | 4.93709 ± 0.0564815 (W 0.0080114) | 🔵 colmat 1.05× faster than `faer` | 🔵 colmat 1.14× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 4.95141 ± 0.00466385 (W 0.012852) | 5.16733 ± 0.0183072 (W 0.00971624) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 229.931 ± 0.867995 (W 0.424378) | 232.379 ± 0.589278 (W 0.790656) | 267.257 ± 0.845899 (W 0.89187) | · | 🔵 colmat 1.16× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 244.303 ± 0.313711 (W 0.412185) | 247.523 ± 0.25159 (W 0.481516) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.0351 ± 0.009394 (W 0.00274871) | 2.27233 ± 0.00379428 (W 0.0078325) | 1.75732 ± 0.0162758 (W 0.00514023) | 🔵 colmat 1.12× faster than `faer` | 🔴 colmat 1.16× slower than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 87.9557 ± 0.469205 (W 0.0460771) | 90.3742 ± 0.141534 (W 0.332815) | 100.281 ± 0.347266 (W 0.511329) | · | 🔵 colmat 1.14× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 5.31004 ± 0.00375795 (W 0.0202837) | 5.91439 ± 0.00391694 (W 0.0114063) | 5.57298 ± 0.00511859 (W 0.00632326) | 🔵 colmat 1.11× faster than `faer` | · |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.34959 ± 0.0106361 (W 0.00246295) | 5.91099 ± 0.000634535 (W 0.0075751) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 261.997 ± 0.333091 (W 0.377929) | 268.648 ± 0.30002 (W 0.422611) | 309.007 ± 2.0948 (W 1.44667) | · | 🔵 colmat 1.18× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 262.392 ± 0.212427 (W 0.333077) | 269.203 ± 0.510792 (W 0.461444) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.41582 ± 0.00570006 (W 0.00254435) | 2.99285 ± 0.00363164 (W 0.00446773) | 2.29388 ± 0.0129344 (W 0.00264349) | 🔵 colmat 1.24× faster than `faer` | 🔴 colmat 1.05× slower than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.41899 ± 0.00791956 (W 0.00339656) | 2.96496 ± 0.0021226 (W 0.00651113) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 101.409 ± 0.262191 (W 0.122295) | 110.276 ± 0.142829 (W 0.117444) | 118.405 ± 0.628135 (W 0.463043) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 101.249 ± 0.0542207 (W 0.108859) | 110.355 ± 0.354151 (W 0.155483) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.756817 ± 0.00143854 (W 0.0031908) | 1.69972 ± 0.0102043 (W 0.00403341) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.800615 ± 0.00290976 (W 0.00330921) | 1.65567 ± 0.0110121 (W 0.00532027) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 46.9633 ± 0.304394 (W 0.430493) | 50.1107 ± 0.176554 (W 0.657646) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 46.0945 ± 0.208897 (W 0.101165) | 48.6043 ± 0.0558326 (W 0.248818) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.659078 ± 0.00840906 (W 0.00184587) | 0.643329 ± 0.00271656 (W 0.00256564) | 0.911691 ± 0.007495 (W 0.00546912) | · | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.659951 ± 0.00657633 (W 0.00208366) | 0.638381 ± 0.0142309 (W 0.00266145) | 1.31808 ± 0.00777134 (W 0.0048555) | · | 🔵 colmat 2.00× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 39.2062 ± 0.123256 (W 0.151875) | 39.3846 ± 0.168191 (W 0.0936116) | 39.5045 ± 0.036797 (W 0.0528821) | · | · |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 39.1073 ± 0.137105 (W 0.145201) | 39.41 ± 0.25264 (W 0.0961355) | 54.3189 ± 0.224621 (W 0.25551) | · | 🔵 colmat 1.39× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 1.70341 ± 0.00856249 (W 0.00209884) | 2.24648 ± 0.0047121 (W 0.00683557) | 3.56371 ± 0.0209145 (W 0.00791696) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.31442 ± 0.0054727 (W 0.00826684) | 2.88533 ± 0.039902 (W 0.0298685) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 70.8646 ± 0.264501 (W 0.0681193) | 81.5136 ± 0.275281 (W 0.108102) | 97.7705 ± 0.547194 (W 0.360341) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 84.7939 ± 0.296257 (W 0.0947906) | 96.6342 ± 7.39092 (W 0.199507) | - | 🔵 colmat 1.14× faster than `faer` (marginal: spread 0.65 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.0575 ± 0.00383008 (W 0.00267852) | 1.60867 ± 0.00612194 (W 0.00505645) | 2.21845 ± 0.00260751 (W 0.00345469) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 2.10× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 31.4232 ± 0.0703663 (W 0.042573) | 42.1158 ± 0.211092 (W 0.0578492) | 44.1454 ± 0.099554 (W 0.207348) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 2.50343 ± 0.024158 (W 0.00421335) | 3.54616 ± 0.00811518 (W 0.00708533) | 5.71997 ± 0.0207167 (W 0.00736089) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 2.28× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.51276 ± 0.0127894 (W 0.00626812) | 3.58833 ± 0.011293 (W 0.00511924) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 86.1971 ± 0.281788 (W 0.113332) | 105.382 ± 0.140254 (W 0.154542) | 133.249 ± 3.79856 (W 8.88863) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.55× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 86.8478 ± 0.330811 (W 0.0455546) | 105.975 ± 0.230919 (W 0.0834203) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.49265 ± 0.00632909 (W 0.00394702) | 2.34335 ± 0.00811627 (W 0.00726069) | 2.74765 ± 0.0154611 (W 0.00495563) | 🔵 colmat 1.57× faster than `faer` | 🔵 colmat 1.84× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.49683 ± 0.00496479 (W 0.00463265) | 2.36224 ± 0.00894402 (W 0.00700388) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 39.7367 ± 0.140968 (W 0.095365) | 59.4709 ± 0.118516 (W 0.0804829) | 59.4565 ± 0.133272 (W 0.0933381) | 🔵 colmat 1.50× faster than `faer` | 🔵 colmat 1.50× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 39.8956 ± 0.110527 (W 0.0897889) | 59.7475 ± 0.165036 (W 0.0701423) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.622764 ± 0.0216181 (W 0.00498768) | 1.6977 ± 0.00604233 (W 0.00489659) | - | 🔵 colmat 2.73× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.679077 ± 0.00451907 (W 0.00376714) | 1.65849 ± 0.00234016 (W 0.00495923) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 22.7121 ± 0.0356401 (W 0.877596) | 26.5272 ± 0.276317 (W 0.0752497) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 22.7921 ± 0.127909 (W 0.0828064) | 26.6703 ± 0.228577 (W 0.115985) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.385298 ± 0.006778 (W 0.00648913) | 0.531315 ± 0.0474035 (W 0.0349592) | 0.674893 ± 0.00447212 (W 0.00432177) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.375836 ± 0.00919066 (W 0.00371875) | 0.532838 ± 0.00269943 (W 0.00738347) | 0.945513 ± 0.0101425 (W 0.0077894) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 2.52× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 25.4279 ± 0.196823 (W 0.869402) | 26.3068 ± 0.104173 (W 2.44968) | 22.5297 ± 0.033802 (W 0.056469) | · | 🔴 colmat 1.13× slower than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 24.9826 ± 0.977296 (W 1.17545) | 26.1576 ± 0.0974104 (W 0.229874) | 35.0054 ± 0.0804812 (W 0.044394) | · | 🔵 colmat 1.40× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/1024x64/b1` | 1.25133 ± 0.00347393 (W 0.00502489) | 2.22491 ± 0.035177 (W 0.0300489) | 3.37607 ± 0.010873 (W 0.00881622) | 🔵 colmat 1.78× faster than `faer` | 🔵 colmat 2.70× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.87873 ± 0.00437714 (W 0.00693876) | 2.32027 ± 0.0341724 (W 0.0748636) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 46.9939 ± 0.187039 (W 0.098101) | 69.8175 ± 0.694864 (W 0.740562) | 66.0205 ± 0.227203 (W 0.0944454) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 60.6767 ± 0.0980803 (W 0.147911) | 71.4866 ± 6.5306 (W 1.15971) | - | 🔵 colmat 1.18× faster than `faer` (marginal: spread 0.61 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.882802 ± 0.00153011 (W 0.00278147) | 1.65139 ± 0.00803074 (W 0.00880394) | 2.4179 ± 0.0237098 (W 0.00747543) | 🔵 colmat 1.87× faster than `faer` | 🔵 colmat 2.74× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 22.1721 ± 0.0473198 (W 0.0594935) | 39.6322 ± 0.0847672 (W 0.415596) | 30.6927 ± 0.14714 (W 0.0685986) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 2.04453 ± 0.0161337 (W 0.0102185) | 3.57183 ± 0.0140262 (W 0.0214109) | 5.9658 ± 0.0115393 (W 0.0124503) | 🔵 colmat 1.75× faster than `faer` | 🔵 colmat 2.92× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.06648 ± 0.0133358 (W 0.00361696) | 3.61694 ± 0.0449215 (W 0.0325347) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 60.0295 ± 0.171611 (W 0.127617) | 94.0861 ± 0.599065 (W 1.45286) | 126.955 ± 4.86818 (W 19.3591) | 🔵 colmat 1.57× faster than `faer` | 🔵 colmat 2.11× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 60.8774 ± 0.127469 (W 0.124723) | 94.4029 ± 1.36249 (W 3.79196) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.34289 ± 0.0118215 (W 0.00450582) | 2.39641 ± 0.00403336 (W 0.0120135) | 2.94727 ± 0.0221393 (W 0.0050228) | 🔵 colmat 1.78× faster than `faer` | 🔵 colmat 2.19× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.35229 ± 0.0056316 (W 0.00749942) | 2.4077 ± 0.00298503 (W 0.00547107) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 30.1794 ± 0.1545 (W 0.143923) | 57.1125 ± 0.363242 (W 0.501613) | 47.3478 ± 0.322235 (W 0.215841) | 🔵 colmat 1.89× faster than `faer` | 🔵 colmat 1.57× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 30.1528 ± 0.128302 (W 0.0464619) | 57.2916 ± 0.101057 (W 4.63426) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 50.09 ± 0.889838 (W 0.0858172) | 52.0086 ± 0.32753 (W 0.10974) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 61.7779 ± 1.9406 (W 1.33343) | 75.5403 ± 0.60095 (W 0.29065) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1` | 0.685992 ± 0.00222108 (W 0.00334238) | 0.78289 ± 0.00123539 (W 0.00671093) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.19189 ± 0.00328565 (W 0.00352873) | 1.22484 ± 0.00727712 (W 0.00368073) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 28.5273 ± 0.31208 (W 0.0486121) | 28.7851 ± 0.0933587 (W 0.0423684) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 30.5987 ± 0.0716171 (W 0.235453) | 31.7883 ± 0.241919 (W 0.253437) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.02943 ± 0.00084666 (W 0.00807325) | 1.04352 ± 0.00425821 (W 0.00542008) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.739261 ± 0.00647303 (W 0.00278245) | 0.845099 ± 0.0195786 (W 0.00525391) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.3218 ± 0.158586 (W 0.24158) | 27.2577 ± 0.0702037 (W 0.374262) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.985528 ± 0.00879749 (W 0.00559148) | 1.00619 ± 0.00616356 (W 0.00613459) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 77.6544 ± 0.238681 (W 0.177365) | 83.2727 ± 0.288852 (W 0.363381) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 77.6339 ± 0.308003 (W 0.413444) | 83.2709 ± 0.261143 (W 0.0618084) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.93493 ± 0.00709896 (W 0.00495433) | 2.44952 ± 0.00417437 (W 0.00441797) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.95741 ± 0.0109544 (W 0.00419623) | 2.45885 ± 0.0049927 (W 0.00548454) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 32.1781 ± 0.145993 (W 0.273305) | 39.0773 ± 0.0942058 (W 0.0737558) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 32.0338 ± 0.109803 (W 0.051339) | 39.059 ± 0.168466 (W 0.046208) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.94585 ± 0.00690547 (W 0.0036975) | 1.42385 ± 0.00525855 (W 0.00299433) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.942975 ± 0.00443401 (W 0.00242689) | 1.42816 ± 0.00516959 (W 0.00303599) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 12.9043 ± 0.078705 (W 0.0140372) | 16.1211 ± 0.0986417 (W 0.0754472) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 16.2732 ± 0.376914 (W 0.0331793) | 36.1834 ± 0.0674713 (W 0.0815133) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.352405 ± 0.00456396 (W 0.00321536) | 0.784968 ± 0.00510409 (W 0.00416698) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.543421 ± 0.00140821 (W 0.00496877) | 1.24268 ± 0.0169429 (W 0.00621692) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.2631 ± 0.250542 (W 0.0412683) | 11.0911 ± 0.123262 (W 0.0343283) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 10.1579 ± 0.0659023 (W 0.0695665) | 12.7817 ± 0.0292803 (W 0.0214657) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.62921 ± 0.00707794 (W 0.00465077) | 0.714738 ± 0.00388047 (W 0.00364902) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.33886 ± 0.000767662 (W 0.00188954) | 0.554039 ± 0.0112335 (W 0.00325678) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.013 ± 0.0583445 (W 0.0286812) | 10.5365 ± 0.152711 (W 0.0288492) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.604526 ± 0.00577225 (W 0.00427696) | 0.693228 ± 0.00352278 (W 0.00626155) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 31.2325 ± 0.160598 (W 0.0586118) | 42.1251 ± 0.104351 (W 0.0936601) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 31.4152 ± 0.102432 (W 0.0621143) | 42.2471 ± 0.0577562 (W 0.0595808) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.24419 ± 0.00787009 (W 0.00450181) | 2.10343 ± 0.00580034 (W 0.00338432) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.25155 ± 0.0130125 (W 0.0056343) | 2.11572 ± 0.0126331 (W 0.00638458) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 16.4745 ± 0.0114263 (W 0.0189682) | 26.9078 ± 0.0651277 (W 0.0485845) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 16.4445 ± 0.050878 (W 0.0242072) | 26.9999 ± 0.0314414 (W 0.387679) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.781441 ± 0.00284795 (W 0.00370131) | 1.47188 ± 0.00531113 (W 0.00720205) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.789327 ± 0.00449456 (W 0.00452007) | 1.47286 ± 0.00787372 (W 0.00465528) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 7.48756 ± 0.0414192 (W 0.0128375) | 10.8266 ± 0.0469722 (W 0.0230887) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 10.1308 ± 0.227412 (W 0.0762296) | 30.802 ± 0.229959 (W 0.220815) | - | 🔵 colmat 3.04× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.324093 ± 0.00293434 (W 0.00195799) | 0.781793 ± 0.00152994 (W 0.00324042) | - | 🔵 colmat 2.41× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.459097 ± 0.00137126 (W 0.00573766) | 1.23229 ± 0.018078 (W 0.00506695) | - | 🔵 colmat 2.68× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.36111 ± 0.0378622 (W 0.0168789) | 9.53575 ± 0.149298 (W 0.177783) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 6.75094 ± 0.0572165 (W 0.0166146) | 11.0615 ± 0.210737 (W 0.215872) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.561096 ± 0.00449212 (W 0.00309121) | 0.771313 ± 0.00612777 (W 0.00658543) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.290065 ± 0.00557683 (W 0.00300227) | 0.602991 ± 0.00459652 (W 0.00289235) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 7.17089 ± 0.0476475 (W 0.024625) | 8.87356 ± 0.224556 (W 0.119823) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.545338 ± 0.000923213 (W 0.00706025) | 0.74013 ± 0.00704074 (W 0.00703619) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 24.0077 ± 0.0813129 (W 0.0474614) | 40.8295 ± 0.0884776 (W 0.260724) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 24.2172 ± 0.0730786 (W 0.0647041) | 41.3068 ± 0.12234 (W 0.924693) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.15676 ± 0.00650805 (W 0.091369) | 2.19469 ± 0.0139283 (W 0.0065075) | - | 🔵 colmat 1.90× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.15598 ± 0.00494864 (W 0.00353222) | 2.20546 ± 0.0560173 (W 0.00660583) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 13.5156 ± 0.015368 (W 0.0437139) | 27.6962 ± 0.202761 (W 0.0434687) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 13.6131 ± 0.0439701 (W 0.0387284) | 27.8571 ± 0.184683 (W 0.0668415) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.752399 ± 0.00174764 (W 0.00335003) | 1.56437 ± 0.0120025 (W 0.00742174) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.749126 ± 0.00585674 (W 0.00344613) | 1.57084 ± 0.00779931 (W 0.00610808) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 71.0082 ± 2.11003 (W 1.16972) | 87.4043 ± 0.630321 (W 0.258498) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 57.5171 ± 1.26671 (W 0.586047) | 71.3983 ± 0.218396 (W 0.575261) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.18285 ± 0.00369914 (W 0.00215763) | 1.10137 ± 0.0120058 (W 0.00244773) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.12437 ± 0.00375659 (W 0.00311036) | 1.0241 ± 0.00291744 (W 0.0024934) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.5329 ± 0.247107 (W 0.0734818) | 26.6823 ± 0.248373 (W 0.155) | 87.7321 ± 0.216268 (W 0.107224) | · | 🔵 colmat 3.31× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.8349 ± 0.176691 (W 0.0727726) | 27.7137 ± 0.149234 (W 0.079031) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.609891 ± 0.0047025 (W 0.00342631) | 0.609474 ± 0.0019322 (W 0.00341894) | 0.980047 ± 0.00308059 (W 0.00253266) | · | 🔵 colmat 1.61× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.625921 ± 0.00263348 (W 0.00240892) | 0.63031 ± 0.00155989 (W 0.00220664) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 25.1229 ± 0.222637 (W 0.512252) | 25.0092 ± 0.131976 (W 0.256563) | 85.3735 ± 0.548838 (W 0.117172) | · | 🔵 colmat 3.40× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.568255 ± 0.00302383 (W 0.00273476) | 0.568926 ± 0.0032962 (W 0.00328061) | 0.930677 ± 0.00340803 (W 0.00334648) | · | 🔵 colmat 1.64× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 82.9915 ± 0.330038 (W 0.428059) | 88.1767 ± 0.737669 (W 0.147565) | 184.972 ± 1.65951 (W 0.611952) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 2.23× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 82.7572 ± 0.307323 (W 0.220331) | 88.5692 ± 0.449724 (W 0.345075) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.02309 ± 0.00514706 (W 0.0057479) | 2.51153 ± 0.00440701 (W 0.00627224) | 4.4896 ± 0.0207105 (W 0.0132237) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 2.22× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.03132 ± 0.00413677 (W 0.005282) | 2.51955 ± 0.0055594 (W 0.0058007) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 37.4266 ± 0.133695 (W 0.0340257) | 44.88 ± 0.118977 (W 0.0776093) | 77.4356 ± 0.218293 (W 0.473516) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 2.07× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 37.4136 ± 0.138888 (W 0.0711375) | 44.892 ± 0.11509 (W 0.0488294) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.02171 ± 0.00609852 (W 0.00468602) | 1.47159 ± 0.00385278 (W 0.00495136) | 2.3956 ± 0.308038 (W 0.00906033) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 2.34× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.01789 ± 0.00364925 (W 0.00924311) | 1.47497 ± 0.00368598 (W 0.00809356) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 16.4059 ± 0.0611658 (W 0.0596767) | 34.5829 ± 0.320663 (W 0.0532164) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 13.3892 ± 0.193599 (W 0.0369041) | 30.9985 ± 0.314991 (W 0.0816541) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.471469 ± 0.00299924 (W 0.00348865) | 1.10899 ± 0.0105714 (W 0.00235787) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.457906 ± 0.000987178 (W 0.00258827) | 1.03507 ± 0.00474357 (W 0.00469071) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.00296 ± 0.054478 (W 0.0143017) | 7.47822 ± 0.0276996 (W 0.0221851) | 25.6071 ± 0.182643 (W 0.049088) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 3.66× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 7.54861 ± 0.0436585 (W 0.0277434) | 8.05412 ± 0.0507837 (W 0.02038) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.220059 ± 0.000708691 (W 0.00190779) | 0.289638 ± 0.000309862 (W 0.00171454) | 0.986128 ± 0.00328942 (W 0.0037533) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 4.48× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.24736 ± 0.000862109 (W 0.00222748) | 0.325234 ± 0.00184656 (W 0.00278499) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.5465 ± 0.00944292 (W 0.00950082) | 6.97006 ± 0.0121361 (W 0.0153992) | 24.8305 ± 0.0588584 (W 0.037795) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 3.79× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.200594 ± 0.000706971 (W 0.00217933) | 0.265379 ± 0.000530854 (W 0.00220727) | 0.923937 ± 0.000639909 (W 0.00161139) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 4.61× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 36.7089 ± 0.183617 (W 0.0849238) | 47.5404 ± 0.0624691 (W 0.0902387) | 142.861 ± 0.382134 (W 0.489043) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 3.89× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.7715 ± 0.142359 (W 0.120326) | 47.5404 ± 0.126805 (W 0.153311) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.3097 ± 0.00747889 (W 0.00725315) | 2.16335 ± 0.00551325 (W 0.00486923) | 4.96445 ± 0.0542253 (W 0.0348453) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 3.79× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.32775 ± 0.00969466 (W 0.00697486) | 2.18196 ± 0.0154895 (W 0.00418659) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.7246 ± 0.0583433 (W 0.0203385) | 32.9063 ± 0.0168374 (W 0.0738277) | 73.4528 ± 0.312084 (W 0.337174) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 3.38× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.7736 ± 0.0779163 (W 0.103384) | 32.9 ± 0.0861482 (W 0.0337912) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.853967 ± 0.0131213 (W 0.00406688) | 1.53205 ± 0.0145082 (W 0.00302509) | 2.38737 ± 0.00878525 (W 0.00800827) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 2.80× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.85153 ± 0.00185947 (W 0.00447085) | 1.52848 ± 0.0115419 (W 0.00644854) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 8.50349 ± 0.0359285 (W 0.0106996) | 27.3919 ± 0.153013 (W 0.470848) | - | 🔵 colmat 3.22× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 7.40297 ± 0.0201537 (W 0.02011) | 25.8665 ± 0.342826 (W 0.059783) | - | 🔵 colmat 3.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.369191 ± 0.00192858 (W 0.00306999) | 1.11357 ± 0.0122498 (W 0.00266162) | - | 🔵 colmat 3.02× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.367548 ± 0.0160452 (W 0.0148604) | 1.0203 ± 0.0072495 (W 0.00538596) | - | 🔵 colmat 2.78× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 3.81314 ± 0.00874865 (W 0.010447) | 5.85316 ± 0.103733 (W 0.180015) | 16.807 ± 0.1045 (W 0.107603) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 4.41× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 4.30158 ± 0.0534934 (W 0.0108388) | 6.3888 ± 0.0375475 (W 0.234244) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.165137 ± 0.00136666 (W 0.00160725) | 0.349951 ± 0.00253056 (W 0.00589804) | 0.987336 ± 0.00347732 (W 0.00404746) | 🔵 colmat 2.12× faster than `faer` | 🔵 colmat 5.98× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.19902 ± 0.00109313 (W 0.00277386) | 0.387871 ± 0.00319631 (W 0.00488324) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.54406 ± 0.0218195 (W 0.0097882) | 5.2917 ± 0.0386872 (W 0.16062) | 16.2578 ± 0.0627643 (W 0.0775682) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 4.59× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.147791 ± 0.000815703 (W 0.000857274) | 0.32513 ± 0.00100831 (W 0.00441602) | 0.92686 ± 0.00837556 (W 0.00269505) | 🔵 colmat 2.20× faster than `faer` | 🔵 colmat 6.27× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 29.6923 ± 0.0729499 (W 0.0918667) | 46.3905 ± 0.206871 (W 0.209393) | 141.995 ± 0.401091 (W 0.595145) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 4.78× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 29.9554 ± 0.0676574 (W 0.0968709) | 46.5049 ± 0.362027 (W 3.46304) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.21854 ± 0.00750563 (W 0.00633147) | 2.25353 ± 0.00837475 (W 0.00749741) | 5.04094 ± 0.248354 (W 0.0452165) | 🔵 colmat 1.85× faster than `faer` | 🔵 colmat 4.14× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.22948 ± 0.012796 (W 0.0097941) | 2.27186 ± 0.00993553 (W 0.00846682) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 18.8289 ± 0.0797967 (W 0.0264543) | 33.6569 ± 0.136764 (W 0.357255) | 67.9719 ± 0.290436 (W 0.296807) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 3.61× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 18.8632 ± 0.0795691 (W 0.0506608) | 33.6894 ± 0.132378 (W 0.123246) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.815978 ± 0.00545843 (W 0.00389637) | 1.60672 ± 0.0163859 (W 0.00465936) | 2.3801 ± 0.00876282 (W 0.00402497) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 2.92× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.82249 ± 0.00488376 (W 0.00956693) | 1.6071 ± 0.0115903 (W 0.00707596) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 151.21 ± 0.831267 (W 0.750115) | 150.348 ± 0.300775 (W 0.961765) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 203.352 ± 1.13443 (W 2.13119) | 228.609 ± 1.38681 (W 0.856523) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.55656 ± 0.0104484 (W 0.00659341) | 1.64924 ± 0.00403265 (W 0.00521353) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.22261 ± 0.00446427 (W 0.00353126) | 2.50192 ± 0.274534 (W 0.0113924) | - | 🔵 colmat 1.13× faster than `faer` (marginal: spread 1.00 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 102.784 ± 0.577681 (W 0.532212) | 103.39 ± 0.360672 (W 0.368484) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 106.623 ± 0.251813 (W 0.59061) | 108.119 ± 0.503201 (W 1.14605) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.73229 ± 0.016177 (W 0.0102327) | 2.74342 ± 0.00744483 (W 0.012685) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 2.54475 ± 0.00799713 (W 0.00359378) | 2.56114 ± 0.00845155 (W 0.00474387) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 97.2614 ± 0.544824 (W 0.369457) | 97.4755 ± 0.400149 (W 0.811417) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 2.58357 ± 0.00837348 (W 0.00539877) | 2.59809 ± 0.00305335 (W 0.00904056) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 262.564 ± 0.912377 (W 0.264171) | 269.195 ± 0.29771 (W 0.296535) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 262.999 ± 0.866542 (W 0.447858) | 270.182 ± 0.452637 (W 0.553827) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.35996 ± 0.0352619 (W 0.00719355) | 5.94631 ± 0.0268077 (W 0.00560122) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.37319 ± 0.0176285 (W 0.0124736) | 5.93758 ± 0.01795 (W 0.00536655) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 101.37 ± 0.450164 (W 0.133141) | 110.477 ± 0.795428 (W 0.14058) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 101.165 ± 0.354633 (W 0.243618) | 110.31 ± 0.705033 (W 0.204646) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.40416 ± 0.0102563 (W 0.00363638) | 2.97785 ± 0.00840561 (W 0.00412886) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.40778 ± 0.00795988 (W 0.00279216) | 2.96634 ± 0.0160293 (W 0.00538442) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 44.6157 ± 0.128044 (W 0.0903215) | 48.8157 ± 0.336884 (W 0.285913) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 59.4388 ± 0.327533 (W 0.466538) | 87.7001 ± 0.25681 (W 0.346823) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.622559 ± 0.00229086 (W 0.00289656) | 1.64543 ± 0.00737023 (W 0.00429362) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.836885 ± 0.00342823 (W 0.00293858) | 2.5208 ± 0.0759442 (W 0.00871891) | - | 🔵 colmat 3.01× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 29.0961 ± 0.227337 (W 0.0465411) | 30.0011 ± 0.0860066 (W 0.0310767) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 32.1233 ± 0.273369 (W 0.231379) | 32.9095 ± 0.12524 (W 0.0616424) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.02997 ± 0.00760701 (W 0.00394765) | 1.14861 ± 0.0050175 (W 0.00326599) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.881208 ± 0.00267358 (W 0.00168404) | 1.03778 ± 0.00428741 (W 0.00273079) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.5805 ± 0.0400273 (W 0.153632) | 28.3869 ± 0.0785896 (W 0.103844) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.976574 ± 0.00615405 (W 0.00682868) | 1.07162 ± 0.00715657 (W 0.00710967) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 86.6678 ± 0.332289 (W 0.213227) | 105.731 ± 0.303726 (W 0.0855874) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 87.3475 ± 0.255551 (W 0.11037) | 106.317 ± 0.487016 (W 0.5716) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.5396 ± 0.0195142 (W 0.00742655) | 3.5872 ± 0.0146316 (W 0.00492333) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.54811 ± 0.00859965 (W 0.00563707) | 3.61385 ± 0.00600631 (W 0.00852421) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 39.9342 ± 0.11226 (W 0.0437429) | 59.4223 ± 0.221636 (W 0.115847) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 40.0068 ± 0.0698324 (W 0.107148) | 59.5523 ± 0.148521 (W 0.0701872) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.48465 ± 0.00527039 (W 0.00433054) | 2.33902 ± 0.00911821 (W 0.00805544) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.48683 ± 0.0151472 (W 0.00429637) | 2.34612 ± 0.00688265 (W 0.00646299) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 20.3294 ± 0.174044 (W 0.416687) | 26.0025 ± 0.0417804 (W 0.192669) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 27.7321 ± 0.148075 (W 0.180544) | 59.7075 ± 0.305047 (W 1.00032) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.487781 ± 0.000831054 (W 0.00366326) | 1.64738 ± 0.00536319 (W 0.00437881) | - | 🔵 colmat 3.38× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.64476 ± 0.00617477 (W 0.0044766) | 2.48634 ± 0.039419 (W 0.00750094) | - | 🔵 colmat 3.86× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 16.986 ± 0.0239735 (W 0.0316378) | 21.7907 ± 0.454302 (W 0.836553) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 19.5749 ± 0.0664598 (W 0.382495) | 24.5756 ± 1.95459 (W 4.23552) | - | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.85 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.748202 ± 0.00439017 (W 0.00505538) | 1.11716 ± 0.0105913 (W 0.0117007) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.613341 ± 0.0031776 (W 0.00270521) | 1.00255 ± 0.00419013 (W 0.0134309) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 15.964 ± 0.102677 (W 0.0215086) | 20.9358 ± 0.572147 (W 0.950417) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.709971 ± 0.00320415 (W 0.0055687) | 1.04644 ± 0.0397434 (W 0.0124045) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1` | 60.8768 ± 0.137135 (W 0.118662) | 94.4412 ± 1.23239 (W 2.77483) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 61.7255 ± 0.0868112 (W 0.154867) | 93.879 ± 0.573714 (W 1.10219) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.08701 ± 0.0145441 (W 0.00446069) | 3.57724 ± 0.00944922 (W 0.0175684) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.10921 ± 0.00882305 (W 0.00434455) | 3.60891 ± 0.0239128 (W 0.0306498) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 30.3749 ± 0.0819025 (W 0.0411488) | 56.6016 ± 0.382202 (W 0.30404) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 30.3076 ± 0.0842015 (W 0.0381032) | 57.1114 ± 0.117408 (W 0.368915) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.33761 ± 0.00468062 (W 0.00236731) | 2.38483 ± 0.00948287 (W 0.0127511) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.33814 ± 0.00901709 (W 0.00531861) | 2.39011 ± 0.00681101 (W 0.00851252) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 217.687 ± 0.842842 (W 0.782572) | 250.468 ± 1.59924 (W 1.22999) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 200.229 ± 1.07325 (W 2.00422) | 222.654 ± 2.29837 (W 0.524808) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 2.17341 ± 0.0320572 (W 0.00624327) | 2.35352 ± 0.0958179 (W 0.0102998) | - | 🔵 colmat 1.08× faster than `faer` (marginal: spread 0.71 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.10937 ± 0.0112174 (W 0.00546514) | 2.26596 ± 0.0362942 (W 0.0121785) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 101.999 ± 0.407235 (W 0.433142) | 102.357 ± 0.788032 (W 0.785131) | 165.521 ± 0.262897 (W 0.577031) | · | 🔵 colmat 1.62× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 102.261 ± 0.383065 (W 0.537198) | 103.001 ± 0.996713 (W 0.602006) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.38025 ± 0.0129767 (W 0.00417151) | 2.36502 ± 0.012303 (W 0.00286794) | 3.35839 ± 0.0225861 (W 0.0036018) | · | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 2.42474 ± 0.0100274 (W 0.00702878) | 2.38557 ± 0.0105659 (W 0.00311289) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 95.7522 ± 0.89255 (W 0.20792) | 96.2336 ± 0.200006 (W 0.344047) | 162.764 ± 0.655635 (W 0.485559) | · | 🔵 colmat 1.70× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 2.21853 ± 0.0155419 (W 0.00418281) | 2.21291 ± 0.0123795 (W 0.00294815) | 3.07293 ± 0.00255865 (W 0.0035045) | · | 🔵 colmat 1.39× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 274.591 ± 0.424064 (W 0.450683) | 281.772 ± 0.0273745 (W 0.379261) | 594.588 ± 0.954745 (W 1.01759) | · | 🔵 colmat 2.17× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 275.307 ± 0.684275 (W 0.363816) | 282.322 ± 0.487025 (W 0.430066) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.48248 ± 0.0177941 (W 0.0226576) | 6.07615 ± 0.0558146 (W 0.0139923) | 9.526 ± 0.250673 (W 0.028981) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.74× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.49978 ± 0.0225914 (W 0.00320697) | 6.06495 ± 0.0285711 (W 0.00958214) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 114.76 ± 1.22814 (W 0.332177) | 122.729 ± 2.08036 (W 0.475092) | 252.161 ± 1.55001 (W 0.541729) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 2.20× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 115.869 ± 1.1609 (W 0.717586) | 122.654 ± 0.883728 (W 0.386166) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.54143 ± 0.0147874 (W 0.00331623) | 3.10925 ± 0.021353 (W 0.00902002) | 4.05854 ± 0.00963946 (W 0.00809404) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.60× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.54321 ± 0.0137454 (W 0.00377565) | 3.08019 ± 0.00986024 (W 0.00563971) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 62.5253 ± 0.175313 (W 0.410606) | 93.0382 ± 0.439112 (W 0.337304) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 54.6947 ± 0.370487 (W 0.200663) | 82.8613 ± 0.314365 (W 0.371831) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.691084 ± 0.00564631 (W 0.00328006) | 2.38786 ± 0.000986038 (W 0.0120082) | - | 🔵 colmat 3.46× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.705261 ± 0.0022831 (W 0.00212842) | 2.25014 ± 0.0102575 (W 0.00967481) | - | 🔵 colmat 3.19× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.1813 ± 0.0263478 (W 0.0454718) | 26.6586 ± 0.162081 (W 0.030704) | 71.6174 ± 0.523584 (W 0.255054) | · | 🔵 colmat 2.74× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.6385 ± 0.185697 (W 0.0429626) | 28.1479 ± 0.264372 (W 0.122782) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.709279 ± 0.00251863 (W 0.0023242) | 0.801779 ± 0.00528778 (W 0.00207321) | 3.67734 ± 0.0193439 (W 0.0120868) | 🔵 colmat 1.13× faster than `faer` | 🔵 colmat 5.18× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.764792 ± 0.00284902 (W 0.00199321) | 0.8678 ± 0.00240153 (W 0.00231296) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 24.718 ± 0.0900735 (W 0.138396) | 25.2877 ± 0.15398 (W 0.0416325) | 68.7277 ± 0.514366 (W 0.500969) | · | 🔵 colmat 2.78× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.651732 ± 0.00282992 (W 0.0019405) | 0.718535 ± 0.00346442 (W 0.00239212) | 3.55405 ± 0.0141923 (W 0.00951439) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 5.45× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 99.0116 ± 0.218656 (W 0.0950028) | 117.892 ± 0.298681 (W 0.0750058) | 360.977 ± 0.194764 (W 0.601274) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 3.65× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 99.6446 ± 0.156844 (W 0.111099) | 118.861 ± 0.214882 (W 0.226347) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.62272 ± 0.00855972 (W 0.00662506) | 3.7048 ± 0.0115491 (W 0.00792113) | 9.55394 ± 0.188777 (W 0.00885005) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 3.64× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.65 ± 0.0123414 (W 0.00413192) | 3.73163 ± 0.00683155 (W 0.0103408) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 51.9785 ± 0.743575 (W 0.387353) | 71.5968 ± 0.154987 (W 0.0721854) | 178.244 ± 0.265902 (W 0.175846) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 3.43× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 52.0495 ± 0.313238 (W 0.0890289) | 71.6215 ± 0.189203 (W 0.118375) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.57782 ± 0.00999323 (W 0.0049388) | 2.44327 ± 0.00781911 (W 0.00697164) | 4.80125 ± 0.0628744 (W 0.0129562) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 3.04× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.58943 ± 0.0128046 (W 0.00313056) | 2.44765 ± 0.0167259 (W 0.015904) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 24.8354 ± 0.105075 (W 0.176124) | 58.0296 ± 0.385238 (W 0.389904) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 23.9468 ± 0.470148 (W 0.176686) | 55.6276 ± 0.788765 (W 1.05943) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.487467 ± 0.00929996 (W 0.00513562) | 2.35286 ± 0.0293444 (W 0.0144596) | - | 🔵 colmat 4.83× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.509681 ± 0.00465804 (W 0.00438744) | 2.24731 ± 0.0236743 (W 0.0115231) | - | 🔵 colmat 4.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 13.8995 ± 0.0403644 (W 0.0145368) | 19.9257 ± 0.216935 (W 0.933145) | 55.3279 ± 0.202282 (W 0.120738) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 3.98× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 15.1599 ± 0.0445783 (W 0.0231798) | 19.9833 ± 0.418177 (W 0.837537) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.434969 ± 0.000677172 (W 0.00203725) | 0.786282 ± 0.0052203 (W 0.0188071) | 3.97861 ± 0.0243448 (W 0.0384363) | 🔵 colmat 1.81× faster than `faer` | 🔵 colmat 9.15× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.499128 ± 0.00275606 (W 0.00310464) | 0.843909 ± 0.0122532 (W 0.0160712) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.0198 ± 0.681935 (W 0.0325184) | 17.7463 ± 0.233117 (W 0.75304) | 53.4025 ± 0.122533 (W 0.32455) | 🔵 colmat 1.36× faster than `faer` | 🔵 colmat 4.10× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.396782 ± 0.00284946 (W 0.00288531) | 0.690706 ± 0.0157616 (W 0.0204203) | 3.89754 ± 0.0207139 (W 0.00771979) | 🔵 colmat 1.74× faster than `faer` | 🔵 colmat 9.82× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 73.2552 ± 0.19228 (W 0.116998) | 105.625 ± 0.665006 (W 6.46737) | 371.099 ± 8.73909 (W 14.1166) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 5.07× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 74.0408 ± 0.287859 (W 0.0949922) | 110.496 ± 3.00025 (W 7.72435) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.19016 ± 0.0136676 (W 0.00526513) | 3.70946 ± 0.0260766 (W 0.030077) | 10.3837 ± 0.100258 (W 0.0219817) | 🔵 colmat 1.69× faster than `faer` | 🔵 colmat 4.74× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.21058 ± 0.0323714 (W 0.00892755) | 3.77189 ± 0.00980663 (W 0.350926) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 42.1437 ± 0.118511 (W 0.400006) | 69.5455 ± 1.06071 (W 5.06497) | 204.482 ± 1.25852 (W 14.8021) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 4.85× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 42.1814 ± 0.0838409 (W 0.0849913) | 69.3944 ± 1.47689 (W 0.382545) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.43972 ± 0.0266193 (W 0.00408768) | 2.4769 ± 0.00752063 (W 0.00571271) | 4.99187 ± 0.0105062 (W 0.0076123) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 3.47× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.43141 ± 0.0140675 (W 0.00621874) | 2.46863 ± 0.0242006 (W 0.00553647) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
