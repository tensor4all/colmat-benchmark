# `colmat-kernels` on `zen5-cpu`

- colmat (fork, measured) commit: `be19a8001e37e5db55c083c90bd8ca10269d1ab5`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon, x86-v4`
- colmat / faer rows: `colmat` (the provider in the measured revision; see library and upstream above); `faer` (the provider in the measured revision; see library and upstream above)
- harness commit: `a899f37cb62992709cb756c0aa0c93aa5785a922`
- toolkit commit: `82bff6ea4508e156b9e661550c1dc65ad662581c`
- measured binary SHA-256: `8e3af1a28df632064c01cc36d0f251ada7418502d63b49968482c12d4c375425`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-26T06:45:35.323854Z`
- timing policy: v2, median of 5 reps, priming 500 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu colmat-kernels --checkout /home/shinaoka/projects/tensor4all/colmat-rs-39-measured --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor aocl --families full_piv_lu,triangular_solve,qr_factor,qr,svd_values,svd_thin,eigh,eig,upgrade_factor,householder_apply --regimes linalg-small,linalg-mid,linalg-tall,linalg-wide --dtypes f64,c64 --threads 1,4,8 --aa 3 --no-publish --label x86v4-39-aocl-v4`
- harness environment: `FBENCH_ORDER` unset, `FBENCH_PROFILE=zen5-cpu`
- raw data: `data/results/zen5-cpu/colmat-kernels/20261126T043949Z-x86v4-39-aocl-v4/`

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
Spread guide (computed from this run's own rows): per-engine between-start range median 0.84%, worst 41.19% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 91 of 3456 engine cells exceed 5% and carry `!`; median within-process relative SD 0.41%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
| `eigh/n32/b1` | 0.100117 ± 0.000144296 (W 0.000398938) | 0.113382 ± 0.000615822 (W 0.000269056) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.10144 ± 0.000636108 (W 0.00157801) | 0.113852 ± 0.000347857 (W 0.000302681) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.41871 ± 0.00202069 (W 0.00250662) | 0.454888 ± 0.00391405 (W 0.00291562) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.421155 ± 0.00340979 (W 0.00285748) | 0.458034 ± 0.00508484 (W 0.00301412) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.02876 ± 0.00646261 (W 0.00141101) | 1.08945 ± 0.00490576 (W 0.00206436) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.04593 ± 0.00494863 (W 0.00229132) | 1.10485 ± 0.0075677 (W 0.00287513) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013425 ± 6.08276e-05 (W 0.00010551) | 0.008967 ± 0.000138904 (W 7.84493e-05) | - | 🔴 colmat 1.50× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.016992 ± 6.26126e-05 (W 0.000157067) | 0.01063 ± 4.72582e-05 (W 5.77953e-05) | - | 🔴 colmat 1.60× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.055514 ± 0.000544153 (W 0.00020695) | 0.04294 ± 0.000334066 (W 0.000218805) | - | 🔴 colmat 1.29× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.063579 ± 0.00125008 (W 0.000628277) | 0.045836 ± 0.000222464 (W 0.000115409) | - | 🔴 colmat 1.39× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.142385 ± 0.000202267 (W 0.000508381) | 0.122479 ± 0.000479365 (W 0.00196339) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.161434 ± 0.00079208 (W 0.00278962) | 0.130046 ± 0.000838433 (W 0.00308334) | - | 🔴 colmat 1.24× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008446 ± 2.12211e-05 (W 6.91824e-05) | 0.008005 ± 1e-05 (W 8.8364e-05) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008505 ± 0.0001006 (W 4.92869e-05) | 0.008115 ± 6.35085e-05 (W 2.52626e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.041007 ± 0.000265588 (W 0.000118617) | 0.040435 ± 0.000237929 (W 0.000183544) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.041096 ± 0.000136141 (W 9.41876e-05) | 0.040505 ± 0.000222532 (W 8.35117e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.122178 ± 0.000175845 (W 0.000170406) | 0.120695 ± 0.000564465 (W 0.00283849) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.122839 ± 0.000164162 (W 0.0024479) | 0.120384 ± 0.00046375 (W 0.000125102) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.030377 ± 0.000224225 (W 0.000136787) | 0.039824 ± 0.000454899 (W 0.000180084) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.02157 ± 0.00031749 (W 0.00016663) | 0.03196 ± 0.000488119 (W 0.000263056) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.104866 ± 0.000499053 (W 0.00226406) | 0.170949 ± 0.000436076 (W 0.00230348) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.081424 ± 0.000395261 (W 0.00211901) | 0.11786 ± 0.000474711 (W 0.00152231) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.231437 ± 0.000206081 (W 0.00174745) | 0.387211 ± 0.00120433 (W 0.0027019) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.237413 ± 0.00198121 (W 0.00286855) | 0.302123 ± 0.00295245 (W 0.00326663) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.018564 ± 3.29292e-05 (W 0.000207893) | 0.028283 ± 6.65833e-05 (W 0.000118534) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.066495 ± 0.000493954 (W 0.00172063) | 0.131595 ± 0.000708026 (W 0.00173911) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.136938 ± 0.00127701 (W 0.000260156) | 0.287967 ± 0.000451609 (W 0.00233875) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.170978 ± 0.00115088 (W 0.00141195) | 0.192399 ± 0.00127681 (W 0.00268376) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.1717 ± 0.000296522 (W 0.00168894) | 0.19297 ± 0.00113044 (W 0.00180089) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.69165 ± 0.00333772 (W 0.00202651) | 0.770516 ± 0.00428624 (W 0.00249758) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.693623 ± 0.00240773 (W 0.00236245) | 0.771008 ± 0.00464098 (W 0.00216878) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.58199 ± 0.00927143 (W 0.00565807) | 1.85889 ± 0.00747406 (W 0.00556136) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.58294 ± 0.00567668 (W 0.00330963) | 1.93605 ± 0.0134321 (W 0.00379895) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.05926 ± 0.000220001 (W 0.000237215) | 0.140161 ± 0.00044926 (W 0.00202257) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059542 ± 0.000184636 (W 0.000144587) | 0.140545 ± 0.000158033 (W 0.000148934) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.218798 ± 0.00197656 (W 0.00210271) | 0.523917 ± 0.00148743 (W 0.00344343) | - | 🔵 colmat 2.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.219309 ± 0.000628002 (W 0.00205647) | 0.521764 ± 0.0020605 (W 0.00298283) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.49842 ± 0.00102978 (W 0.00228839) | 1.19673 ± 0.00318758 (W 0.00212981) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.501845 ± 0.00317225 (W 0.00230507) | 1.1971 ± 0.00307187 (W 0.00161394) | - | 🔵 colmat 2.39× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000801 ± 5.7735e-06 (W 2.78244e-05) | 0.000741 ± 1.58219e-05 (W 2.05985e-05) | 0.001022 ± 1e-05 (W 4.06964e-05) | 🔴 colmat 1.08× slower than `faer` | 🔵 colmat 1.28× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.001823 ± 1.76163e-05 (W 3.81877e-05) | 0.001683 ± 5.19615e-06 (W 2.48234e-05) | 0.001954 ± 6.08276e-06 (W 2.8464e-05) | 🔴 colmat 1.08× slower than `faer` | 🔵 colmat 1.07× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003246 ± 2.08167e-05 (W 3.27765e-05) | 0.002995 ± 3.00056e-05 (W 4.14886e-05) | 0.002995 ± 2.00083e-05 (W 4.10889e-05) | 🔴 colmat 1.08× slower than `faer` | 🔴 colmat 1.08× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.072496 ± 0.000266909 (W 0.000279577) | 0.11288 ± 0.000555222 (W 0.000388429) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.073296 ± 0.000435759 (W 0.000588783) | 0.113722 ± 0.000115295 (W 0.000364775) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.225551 ± 0.000782761 (W 0.00217074) | 0.450321 ± 0.00159496 (W 0.00260837) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.234852 ± 0.000442101 (W 0.00196022) | 0.45145 ± 0.00174181 (W 0.0023076) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.507997 ± 0.00386729 (W 0.00285499) | 1.06036 ± 0.00272391 (W 0.00148876) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.507657 ± 0.00226724 (W 0.00341549) | 1.06452 ± 0.00363228 (W 0.00179775) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014537 ± 9.53939e-05 (W 0.00011149) | 0.008966 ± 9.59705e-05 (W 5.98515e-05) | - | 🔴 colmat 1.62× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018344 ± 0.000206369 (W 0.000144994) | 0.01074 ± 2.3094e-05 (W 7.4027e-05) | - | 🔴 colmat 1.71× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.056816 ± 0.000102378 (W 0.000244806) | 0.0432 ± 0.00011345 (W 0.000119659) | - | 🔴 colmat 1.32× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.065493 ± 0.000248308 (W 0.000189814) | 0.045846 ± 0.000111505 (W 9.72846e-05) | - | 🔴 colmat 1.43× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.140312 ± 0.00112757 (W 0.000591354) | 0.122328 ± 0.000925 (W 0.00247857) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.163726 ± 0.00055026 (W 0.00191985) | 0.128951 ± 0.000426556 (W 0.000555556) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006482 ± 0.000157735 (W 0.000289975) | 0.008406 ± 3e-05 (W 0.000145544) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005881 ± 4.50925e-05 (W 7.60342e-05) | 0.008446 ± 3.60555e-05 (W 9.25322e-05) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.019236 ± 0.000251774 (W 0.000415804) | 0.019166 ± 0.000335286 (W 0.00177624) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.019748 ± 0.00019468 (W 0.000370278) | 0.017773 ± 0.0001395 (W 0.000221177) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.045415 ± 5.72742e-05 (W 0.000591901) | 0.043772 ± 0.000274464 (W 0.000407444) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.046006 ± 0.000596114 (W 0.000307192) | 0.041798 ± 0.000164743 (W 0.000270644) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.032361 ± 0.000918722 (W 9.62767e-05) | 0.053991 ± 0.000446127 (W 0.00194287) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.022142 ± 0.000459088 (W 0.000195999) | 0.046027 ± 0.000400255 (W 0.000498708) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.072325 ± 0.000150766 (W 0.00227715) | 0.24675 ± 0.00102117 (W 0.00273159) | - | 🔵 colmat 3.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.073319 ± 0.00037654 (W 0.000371954) | 0.18577 ± 0.000781287 (W 0.00238029) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.162777 ± 0.000652986 (W 0.000565506) | 0.503419 ± 0.00204662 (W 0.00332695) | - | 🔵 colmat 3.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.199833 ± 0.00137999 (W 0.00235917) | 0.397261 ± 0.00313832 (W 0.00377236) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.019467 ± 0.000370622 (W 0.00012369) | 0.0429 ± 0.000555339 (W 0.000211856) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.04311 ± 0.000476956 (W 0.000247235) | 0.207622 ± 0.00101488 (W 0.00229115) | - | 🔵 colmat 4.82× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.105417 ± 0.000836007 (W 0.000486175) | 0.428891 ± 0.000314518 (W 0.00290737) | - | 🔵 colmat 4.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.172431 ± 0.000737577 (W 0.00204059) | 0.190616 ± 0.00113077 (W 0.00164241) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.173203 ± 0.000407372 (W 0.00238854) | 0.191427 ± 0.000798063 (W 0.00204909) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.564473 ± 0.00186238 (W 0.00285521) | 0.763214 ± 0.00104206 (W 0.00232973) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.571706 ± 0.00157534 (W 0.00205773) | 0.764837 ± 0.00112617 (W 0.0019585) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.1523 ± 0.00417364 (W 0.00232406) | 1.79613 ± 0.00191859 (W 0.00211123) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15243 ± 0.00183849 (W 0.00163259) | 1.80095 ± 0.00243931 (W 0.002346) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.060423 ± 0.000147005 (W 0.000152381) | 0.13943 ± 0.000252431 (W 0.000382159) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060773 ± 0.000137652 (W 0.000227146) | 0.140465 ± 0.00073432 (W 0.000450601) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.223578 ± 0.0011504 (W 0.0018305) | 0.519628 ± 0.00177422 (W 0.00522698) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.228316 ± 0.000955977 (W 0.00144301) | 0.520302 ± 0.00263552 (W 0.00331724) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.539026 ± 0.00123373 (W 0.00280834) | 1.18851 ± 0.00364774 (W 0.00338727) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.539356 ± 0.00340274 (W 0.0028352) | 1.20338 ± 0.00463732 (W 0.00449101) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001012 ± 1.73205e-05 (W 2.16795e-05) | 0.000761 ± 2.08167e-05 (W 2.72819e-05) | 0.001012 ± 5.7735e-06 (W 2.82843e-05) | 🔴 colmat 1.33× slower than `faer` | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002255 ± 1.70392e-05 (W 3.19578e-05) | 0.001693 ± 1.52753e-05 (W 3.16228e-05) | 0.001954 ± 5.7735e-06 (W 2.8464e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.15× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003948 ± 5.03322e-05 (W 7.80147e-05) | 0.003036 ± 3.09892e-05 (W 2.81993e-05) | 0.003005 ± 1.12694e-05 (W 3.61428e-05) | 🔴 colmat 1.30× slower than `faer` | 🔴 colmat 1.31× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.068509 ± 0.000578415 (W 0.000317216) | 0.112702 ± 0.000799816 (W 0.00237748) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.068321 ± 0.00102898 (W 0.000352791) | 0.113774 ± 0.00112301 (W 0.00214221) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.190127 ± 0.00115659 (W 0.00164368) | 0.452451 ± 0.00191623 (W 0.00218994) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.20303 ± 0.000727723 (W 0.00364389) | 0.453567 ± 0.000226577 (W 0.00213451) | - | 🔵 colmat 2.23× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.405079 ± 0.000501417 (W 0.00321062) | 1.068 ± 0.00769311 (W 0.00239966) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.410002 ± 0.00168456 (W 0.00387195) | 1.0723 ± 0.00116661 (W 0.00186198) | - | 🔵 colmat 2.62× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014587 ± 0.000382035 (W 0.000121253) | 0.009077 ± 0.000153657 (W 0.000354994) | - | 🔴 colmat 1.61× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018274 ± 3.55949e-05 (W 0.000165073) | 0.0107 ± 4.16333e-05 (W 9.92179e-05) | - | 🔴 colmat 1.71× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.056265 ± 0.00016953 (W 0.000170339) | 0.04285 ± 0.000265688 (W 0.00016021) | - | 🔴 colmat 1.31× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.065444 ± 0.000312492 (W 0.000187906) | 0.045645 ± 0.000212227 (W 0.00014519) | - | 🔴 colmat 1.43× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.142526 ± 0.000851386 (W 0.00130553) | 0.122852 ± 0.000118078 (W 0.00179433) | - | 🔴 colmat 1.16× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.173707 ± 0.000722222 (W 0.00269295) | 0.127128 ± 0.000458731 (W 0.00235585) | - | 🔴 colmat 1.37× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006021 ± 0.000209841 (W 0.000198457) | 0.008345 ± 2.00083e-05 (W 8.01948e-05) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005971 ± 0.000101 (W 0.000180965) | 0.008426 ± 6.1101e-05 (W 9.84591e-05) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014547 ± 0.000153526 (W 0.000220833) | 0.018322 ± 0.000172402 (W 0.000239262) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.01634 ± 0.000296025 (W 0.000431176) | 0.018434 ± 0.000165929 (W 0.00030494) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.032197 ± 0.000233787 (W 0.000750405) | 0.054151 ± 0.000240077 (W 0.00106562) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.032982 ± 0.000402557 (W 0.000666606) | 0.054632 ± 0.000505871 (W 0.00138193) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.031478 ± 0.000297761 (W 0.000222093) | 0.061685 ± 0.000911125 (W 0.000819364) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.02156 ± 0.00012329 (W 0.000121265) | 0.054531 ± 0.000394524 (W 0.000629628) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.070803 ± 0.000600325 (W 0.000182656) | 0.288132 ± 0.00305069 (W 0.00403481) | - | 🔵 colmat 4.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.072906 ± 0.000706728 (W 0.000461822) | 0.232467 ± 0.00320458 (W 0.00313557) | - | 🔵 colmat 3.19× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.165342 ± 0.00125016 (W 0.00116317) | 0.582066 ± 0.00409239 (W 0.00321625) | - | 🔵 colmat 3.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.195475 ± 0.000836522 (W 0.00306539) | 0.494721 ± 0.00115261 (W 0.00623697) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.019536 ± 0.000255969 (W 0.000112233) | 0.050675 ± 0.0010302 (W 0.00096447) | - | 🔵 colmat 2.59× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.041899 ± 0.00105862 (W 0.000330101) | 0.249736 ± 0.000828886 (W 0.00357486) | - | 🔵 colmat 5.96× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.100479 ± 0.000373101 (W 0.000756751) | 0.503157 ± 0.00621306 (W 0.00478662) | - | 🔵 colmat 5.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.173227 ± 0.000708384 (W 0.00233792) | 0.190846 ± 0.000459709 (W 0.0021985) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.172665 ± 0.000146275 (W 0.00210854) | 0.191129 ± 0.00230391 (W 0.00254916) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.564931 ± 0.00273674 (W 0.00321113) | 0.764529 ± 0.000812569 (W 0.00217247) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.567328 ± 0.00237119 (W 0.00299829) | 0.766353 ± 0.00262562 (W 0.00264469) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.14622 ± 0.000506483 (W 0.00261548) | 1.81155 ± 0.00217065 (W 0.00277187) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.15765 ± 0.000663379 (W 0.00218803) | 1.81592 ± 0.00334969 (W 0.00288592) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.060012 ± 0.000288422 (W 0.000205894) | 0.139998 ± 0.00103134 (W 0.0019284) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.060864 ± 0.000368357 (W 0.000224458) | 0.140763 ± 0.000827484 (W 0.0023013) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.224788 ± 0.000246855 (W 0.00210778) | 0.519038 ± 0.00245201 (W 0.00274731) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.22531 ± 0.000949659 (W 0.00225762) | 0.521643 ± 0.00208792 (W 0.00271259) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.555002 ± 0.000610813 (W 0.00310113) | 1.19077 ± 0.00259016 (W 0.00294104) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.557026 ± 0.00278926 (W 0.00208027) | 1.20577 ± 0.00566336 (W 0.00569082) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001012 ± 2.64575e-05 (W 3.78153e-05) | 0.000761 ± 3.55012e-05 (W 2.84482e-05) | 0.001002 ± 5.7735e-06 (W 3.71484e-05) | 🔴 colmat 1.33× slower than `faer` | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002314 ± 0.000132855 (W 3.53454e-05) | 0.001694 ± 2.0664e-05 (W 3.38703e-05) | 0.001973 ± 5.50757e-06 (W 3.51895e-05) | 🔴 colmat 1.37× slower than `faer` | 🔴 colmat 1.17× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003937 ± 7.82326e-05 (W 3.88626e-05) | 0.003075 ± 3.9879e-05 (W 3.10693e-05) | 0.003026 ± 2.46374e-05 (W 3.22831e-05) | 🔴 colmat 1.28× slower than `faer` | 🔴 colmat 1.30× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.100537 ± 0.00032015 (W 0.000115162) | 0.112831 ± 0.00054865 (W 0.00209868) | 0.105196 ± 0.00063579 (W 0.00176814) | 🔵 colmat 1.12× faster than `faer` | · |  |
| `eigh/n32/b1 [api-inclusive]` | 0.101441 ± 0.000705324 (W 0.000254712) | 0.113652 ± 0.000520308 (W 0.00199139) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.419412 ± 0.00248215 (W 0.00251625) | 0.453055 ± 0.00397754 (W 0.0025053) | 0.406899 ± 0.00259167 (W 0.00306679) | 🔵 colmat 1.08× faster than `faer` | · |  |
| `eigh/n64/b1 [api-inclusive]` | 0.420313 ± 0.00143299 (W 0.00331357) | 0.454217 ± 0.00112213 (W 0.00309314) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.02834 ± 0.0096665 (W 0.00255362) | 1.08816 ± 0.00299637 (W 0.00114984) | 0.965811 ± 0.00546376 (W 0.00119571) | 🔵 colmat 1.06× faster than `faer` | 🔴 colmat 1.06× slower than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.05167 ± 0.00749939 (W 0.00610745) | 1.1046 ± 0.00241126 (W 0.0033254) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013816 ± 9.81495e-05 (W 8.01873e-05) | 0.009147 ± 0.000110726 (W 0.000111728) | 0.033062 ± 0.00018284 (W 6.54309e-05) | 🔴 colmat 1.51× slower than `faer` | 🔵 colmat 2.39× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014958 ± 3.4641e-05 (W 0.000100349) | 0.010419 ± 0.000106522 (W 8.16701e-05) | - | 🔴 colmat 1.44× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.055465 ± 0.000718768 (W 0.000316703) | 0.04295 ± 0.000246011 (W 0.000331188) | 0.644642 ± 0.00491922 (W 0.00395849) | 🔴 colmat 1.29× slower than `faer` | 🔵 colmat 11.62× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.058039 ± 0.000376865 (W 0.000277661) | 0.044544 ± 0.000577859 (W 0.000158667) | - | 🔴 colmat 1.30× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.145344 ± 0.00226801 (W 0.00263063) | 0.122568 ± 0.000646132 (W 0.000453281) | 2.6262 ± 0.00922422 (W 0.00568829) | 🔴 colmat 1.19× slower than `faer` | 🔵 colmat 18.07× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.15034 ± 0.000864254 (W 0.00265591) | 0.127638 ± 0.000866056 (W 0.00172419) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.008536 ± 5.50757e-05 (W 9.01665e-05) | 0.008185 ± 2.95014e-05 (W 6.84054e-05) | 0.023353 ± 0.000222442 (W 9.31665e-05) | · | 🔵 colmat 2.74× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008946 ± 0.000164135 (W 0.000126381) | 0.008285 ± 5.00033e-05 (W 5.02991e-05) | 0.008847 ± 0.000100664 (W 3.19421e-05) | 🔴 colmat 1.08× slower than `faer` | · | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.040615 ± 0.000200367 (W 0.000201691) | 0.040175 ± 0.000147751 (W 0.000226408) | 0.060203 ± 0.000329819 (W 0.000195686) | · | 🔵 colmat 1.48× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.041197 ± 0.000304254 (W 7.67939e-05) | 0.040325 ± 0.000236115 (W 0.000321067) | 0.059341 ± 6.01193e-05 (W 0.00135104) | · | 🔵 colmat 1.44× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.123791 ± 0.000689696 (W 0.00125813) | 0.122318 ± 0.000627902 (W 0.00226184) | 0.133729 ± 0.00073061 (W 0.00231116) | · | 🔵 colmat 1.08× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.125143 ± 0.000595525 (W 0.00239672) | 0.122849 ± 0.000525062 (W 8.06362e-05) | 0.204532 ± 0.00134008 (W 0.00224162) | · | 🔵 colmat 1.63× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.017573 ± 0.000164257 (W 0.000103781) | 0.028844 ± 0.000238956 (W 0.000155535) | 0.01615 ± 0.000517553 (W 6.10737e-05) | 🔵 colmat 1.64× faster than `faer` | 🔴 colmat 1.09× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.019667 ± 0.000564102 (W 0.000394381) | 0.031739 ± 0.000369717 (W 0.000130499) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.068399 ± 0.000480356 (W 0.000180414) | 0.110406 ± 8.50353e-05 (W 0.00252753) | 0.099075 ± 0.000398599 (W 0.00180259) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.074299 ± 0.000670222 (W 0.000393407) | 0.115315 ± 0.00054562 (W 0.00201043) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.182821 ± 0.000560118 (W 0.00157908) | 0.259764 ± 0.00210934 (W 0.00241135) | 0.341637 ± 0.000367016 (W 0.00213983) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.87× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.221268 ± 0.000990752 (W 0.0033349) | 0.29491 ± 0.00277134 (W 0.00188698) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.009287 ± 3.84318e-05 (W 0.000121083) | 0.018054 ± 6.08276e-05 (W 0.000172335) | 0.007414 ± 7.81025e-05 (W 3.89872e-05) | 🔵 colmat 1.94× faster than `faer` | 🔴 colmat 1.25× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.034265 ± 3.61985e-05 (W 0.001337) | 0.075811 ± 0.000125544 (W 0.000255565) | 0.040877 ± 0.000810308 (W 9.1815e-05) | 🔵 colmat 2.21× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.089086 ± 0.000561851 (W 0.000548968) | 0.165693 ± 0.00115481 (W 0.00180386) | 0.137436 ± 0.000820515 (W 0.00176943) | 🔵 colmat 1.86× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.172701 ± 0.00126303 (W 0.00240317) | 0.194222 ± 0.000713828 (W 0.00205658) | 0.169427 ± 0.000432706 (W 0.00201345) | 🔵 colmat 1.12× faster than `faer` | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.173614 ± 0.00059864 (W 0.00210206) | 0.195104 ± 0.000622934 (W 0.002047) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.708733 ± 0.00674325 (W 0.0023579) | 0.802006 ± 0.00186051 (W 0.00249692) | 0.642668 ± 0.00188386 (W 0.0029181) | 🔵 colmat 1.13× faster than `faer` | 🔴 colmat 1.10× slower than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.704293 ± 0.00414436 (W 0.00221181) | 0.803237 ± 0.00300278 (W 0.0018189) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.58618 ± 0.0059761 (W 0.0062236) | 1.85407 ± 0.00857297 (W 0.00480388) | 1.51817 ± 0.0107618 (W 0.00258364) | 🔵 colmat 1.17× faster than `faer` | · |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.59501 ± 0.00345326 (W 0.00417399) | 1.93415 ± 0.0105047 (W 0.00243436) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.059702 ± 0.00042501 (W 0.00199519) | 0.142165 ± 0.000326335 (W 0.00170398) | 0.120345 ± 0.000438224 (W 0.00160124) | 🔵 colmat 2.38× faster than `faer` | 🔵 colmat 2.02× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.059852 ± 0.000558829 (W 0.000210329) | 0.142495 ± 0.000334274 (W 9.99165e-05) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.221143 ± 0.00163422 (W 0.00340327) | 0.54142 ± 0.00233323 (W 0.00237907) | 0.242041 ± 0.00131046 (W 0.00253081) | 🔵 colmat 2.45× faster than `faer` | 🔵 colmat 1.09× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.222996 ± 0.00167246 (W 0.00189091) | 0.54155 ± 0.00315204 (W 0.0021933) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.509459 ± 0.003976 (W 0.0028554) | 1.19922 ± 0.00747802 (W 0.00478854) | 0.602183 ± 0.0027944 (W 0.00247087) | 🔵 colmat 2.35× faster than `faer` | 🔵 colmat 1.18× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.508539 ± 0.0025068 (W 0.00222764) | 1.19859 ± 0.00471823 (W 0.00217953) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000862 ± 1.1547e-05 (W 2.92626e-05) | 0.000822 ± 0 (W 3.2746e-05) | 0.000782 ± 5.50757e-06 (W 4.12226e-05) | · | 🔴 colmat 1.10× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002094 ± 1.1547e-05 (W 3.31662e-05) | 0.001993 ± 1.70392e-05 (W 3.51454e-05) | 0.001442 ± 3.00056e-05 (W 3.49185e-05) | · | 🔴 colmat 1.45× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003607 ± 2.08167e-05 (W 5.35724e-05) | 0.003516 ± 2.02567e-05 (W 2.3776e-05) | 0.002164 ± 1.52753e-05 (W 3.67451e-05) | · | 🔴 colmat 1.67× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.072285 ± 0.000522992 (W 0.00234376) | 0.112061 ± 0.000586806 (W 0.00206242) | 0.105729 ± 0.000774587 (W 0.00156429) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.073218 ± 0.00026616 (W 0.00040599) | 0.113534 ± 0.000314125 (W 0.000639199) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.224038 ± 0.000610305 (W 0.00180424) | 0.45018 ± 0.00309013 (W 0.00226704) | 0.411488 ± 0.00154631 (W 0.0019569) | 🔵 colmat 2.01× faster than `faer` | 🔵 colmat 1.84× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.237363 ± 0.0037961 (W 0.00242296) | 0.454278 ± 0.00339509 (W 0.00283606) | - | 🔵 colmat 1.91× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.505522 ± 0.00214646 (W 0.00380525) | 1.0614 ± 0.00184807 (W 0.00199827) | 0.971915 ± 0.00294183 (W 0.00278026) | 🔵 colmat 2.10× faster than `faer` | 🔵 colmat 1.92× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.511524 ± 0.00302356 (W 0.00254214) | 1.06398 ± 0.00376247 (W 0.00274109) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014968 ± 0.000120625 (W 0.000111374) | 0.009257 ± 3.21455e-05 (W 0.000118407) | 0.032702 ± 8.14125e-05 (W 8.46197e-05) | 🔴 colmat 1.62× slower than `faer` | 🔵 colmat 2.18× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.01636 ± 0.000103523 (W 0.000118208) | 0.01047 ± 5.5109e-05 (W 7.38329e-05) | - | 🔴 colmat 1.56× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.057327 ± 0.000441851 (W 0.000155177) | 0.043531 ± 0.000174701 (W 0.000103243) | 0.644601 ± 0.0015633 (W 0.00180494) | 🔴 colmat 1.32× slower than `faer` | 🔵 colmat 11.24× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.059992 ± 0.000160001 (W 0.000234309) | 0.044823 ± 0.000504405 (W 0.000175677) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.142629 ± 0.00116204 (W 0.00200882) | 0.123511 ± 0.00074378 (W 0.00205392) | 2.61933 ± 0.0080992 (W 0.00556311) | 🔴 colmat 1.15× slower than `faer` | 🔵 colmat 18.36× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.150651 ± 0.00162668 (W 0.00301408) | 0.125426 ± 0.000266888 (W 0.00204551) | - | 🔴 colmat 1.20× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005821 ± 9.29157e-05 (W 8.12361e-05) | 0.008386 ± 1.73205e-05 (W 8.5844e-05) | 0.023193 ± 0.000110485 (W 7.08463e-05) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 3.98× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005981 ± 9.50807e-05 (W 5.25471e-05) | 0.008526 ± 5.03322e-05 (W 6.61075e-05) | 0.008706 ± 9.17624e-05 (W 2.88496e-05) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.018455 ± 0.000255891 (W 0.000158588) | 0.017933 ± 0.000211732 (W 0.000234082) | 0.062296 ± 0.000502356 (W 0.00187458) | · | 🔵 colmat 3.38× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.018685 ± 6.60328e-05 (W 0.000154461) | 0.017893 ± 0.000291411 (W 0.000437675) | 0.058029 ± 0.000592368 (W 0.00020118) | · | 🔵 colmat 3.11× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.04298 ± 0.00028006 (W 0.000240434) | 0.041719 ± 0.000106444 (W 0.000185818) | 0.086411 ± 0.00107013 (W 0.00246317) | · | 🔵 colmat 2.01× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.043461 ± 0.000159478 (W 0.000217912) | 0.041848 ± 0.000140524 (W 0.000189994) | 0.202217 ± 0.000237119 (W 0.00243618) | · | 🔵 colmat 4.65× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.017724 ± 0.000275712 (W 0.000227934) | 0.042851 ± 0.0004085 (W 0.00130334) | 0.01614 ± 0.000101129 (W 7.8646e-05) | 🔵 colmat 2.42× faster than `faer` | 🔴 colmat 1.10× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.019868 ± 0.000230804 (W 0.000149667) | 0.045685 ± 0.000365774 (W 0.000617639) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr/n64/b1` | 0.061184 ± 0.00037885 (W 0.00051767) | 0.176663 ± 0.00166979 (W 0.00147093) | 0.098453 ± 0.000516579 (W 0.00193966) | 🔵 colmat 2.89× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.067448 ± 0.000360945 (W 0.000362639) | 0.183486 ± 0.00206028 (W 0.00258783) | - | 🔵 colmat 2.72× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.143177 ± 0.0010542 (W 0.000409842) | 0.350764 ± 0.000253699 (W 0.00205836) | 0.338622 ± 0.000435301 (W 0.00264501) | 🔵 colmat 2.45× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.18569 ± 0.000794649 (W 0.00279649) | 0.39099 ± 0.00171105 (W 0.00306557) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.009137 ± 6.08276e-05 (W 9.27157e-05) | 0.031499 ± 7.50355e-05 (W 0.000295176) | 0.007474 ± 2.12211e-05 (W 2.70185e-05) | 🔵 colmat 3.45× faster than `faer` | 🔴 colmat 1.22× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.033723 ± 0.000284324 (W 0.000456768) | 0.146644 ± 0.000365533 (W 0.00180081) | 0.041228 ± 0.000972545 (W 0.000183742) | 🔵 colmat 4.35× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.089167 ± 5.47449e-05 (W 0.00210978) | 0.289389 ± 0.00212026 (W 0.00256994) | 0.137156 ± 0.00131399 (W 0.00215968) | 🔵 colmat 3.25× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.174555 ± 0.000305688 (W 0.00207444) | 0.193471 ± 0.00070644 (W 0.00254094) | 0.168207 ± 6.53682e-05 (W 0.00172208) | 🔵 colmat 1.11× faster than `faer` | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.174896 ± 0.000468492 (W 0.000512386) | 0.193785 ± 0.0016761 (W 0.00183459) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.568211 ± 0.00242301 (W 0.00270729) | 0.795793 ± 0.012819 (W 0.0017068) | 0.641126 ± 0.00335286 (W 0.00335904) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.572128 ± 0.00342494 (W 0.00279467) | 0.797397 ± 0.00142962 (W 0.0022454) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.157 ± 0.00311332 (W 0.00232902) | 1.80026 ± 0.0020427 (W 0.00287374) | 1.4782 ± 0.00323685 (W 0.00504723) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.16124 ± 0.00333242 (W 0.00312983) | 1.8015 ± 0.000956476 (W 0.00297061) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.060333 ± 0.000620943 (W 0.000104001) | 0.141885 ± 0.00053721 (W 0.00243638) | 0.120207 ± 0.000621008 (W 0.00173339) | 🔵 colmat 2.35× faster than `faer` | 🔵 colmat 1.99× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.060514 ± 0.000180436 (W 0.000211993) | 0.142305 ± 0.000275181 (W 0.00207544) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.227138 ± 0.000992033 (W 0.00208022) | 0.539196 ± 0.0015295 (W 0.00241723) | 0.241731 ± 0.0019303 (W 0.00258278) | 🔵 colmat 2.37× faster than `faer` | 🔵 colmat 1.06× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.231352 ± 0.00158925 (W 0.00219947) | 0.540248 ± 0.000743359 (W 0.00249654) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.546349 ± 0.000763179 (W 0.00204958) | 1.19229 ± 0.00465614 (W 0.00416721) | 0.600862 ± 0.00137391 (W 0.00289397) | 🔵 colmat 2.18× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.549145 ± 0.0014809 (W 0.00230765) | 1.19772 ± 0.00332028 (W 0.00253016) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001072 ± 2.13776e-05 (W 4.37401e-05) | 0.000821 ± 1.57162e-05 (W 2.76641e-05) | 0.000781 ± 1.18462e-05 (W 3.29135e-05) | 🔴 colmat 1.31× slower than `faer` | 🔴 colmat 1.37× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002515 ± 2e-05 (W 4.004e-05) | 0.001974 ± 2.08167e-05 (W 3.23605e-05) | 0.001453 ± 5.7735e-06 (W 3.87466e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.73× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004218 ± 4.04145e-05 (W 7.1624e-05) | 0.003517 ± 1.76163e-05 (W 4.63605e-05) | 0.002144 ± 2.88675e-05 (W 4.61519e-05) | 🔴 colmat 1.20× slower than `faer` | 🔴 colmat 1.97× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.06797 ± 9.75004e-05 (W 0.000223137) | 0.113462 ± 0.000432911 (W 0.00206841) | 0.105998 ± 0.000171389 (W 0.000299099) | 🔵 colmat 1.67× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.068488 ± 0.000118837 (W 0.000325406) | 0.114113 ± 0.000257374 (W 0.00175813) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.189266 ± 0.000517862 (W 0.00152837) | 0.453327 ± 0.00428887 (W 0.00248081) | 0.410027 ± 0.000786981 (W 0.00216656) | 🔵 colmat 2.40× faster than `faer` | 🔵 colmat 2.17× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.201749 ± 0.00105334 (W 0.00194295) | 0.454339 ± 0.00055256 (W 0.00290312) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.406825 ± 0.00373898 (W 0.00266776) | 1.06629 ± 0.0020183 (W 0.0010801) | 0.975071 ± 0.00253764 (W 0.00179507) | 🔵 colmat 2.62× faster than `faer` | 🔵 colmat 2.40× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.407387 ± 0.00054703 (W 0.00310922) | 1.07154 ± 0.00897849 (W 0.00244557) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014868 ± 9.53939e-05 (W 6.46313e-05) | 0.009157 ± 2.88675e-05 (W 5.59214e-05) | 0.032762 ± 0.000107364 (W 6.73365e-05) | 🔴 colmat 1.62× slower than `faer` | 🔵 colmat 2.20× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.016061 ± 0.000198973 (W 9.35858e-05) | 0.01043 ± 6.15386e-05 (W 9.72692e-05) | - | 🔴 colmat 1.54× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.056867 ± 0.000450744 (W 0.000179842) | 0.04289 ± 6.86319e-05 (W 0.000115772) | 0.646656 ± 0.0202384 (W 0.00288134) | 🔴 colmat 1.33× slower than `faer` | 🔵 colmat 11.37× faster than `lapack-aocl` | `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.059641 ± 0.000219887 (W 0.000239061) | 0.044513 ± 0.000146186 (W 0.000187545) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.144712 ± 0.00200073 (W 0.00135283) | 0.12328 ± 0.000990406 (W 0.0019845) | 2.61873 ± 0.00103416 (W 0.00690749) | 🔴 colmat 1.17× slower than `faer` | 🔵 colmat 18.10× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.162012 ± 0.0016997 (W 0.00107867) | 0.125765 ± 0.000380107 (W 0.00204948) | - | 🔴 colmat 1.29× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00585 ± 9.05373e-05 (W 0.000100346) | 0.008436 ± 0.000277273 (W 0.000263185) | 0.023142 ± 6.74117e-05 (W 0.000128387) | 🔵 colmat 1.44× faster than `faer` | 🔵 colmat 3.96× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00574 ± 0.0001395 (W 9.39681e-05) | 0.008435 ± 1.1547e-05 (W 7.84748e-05) | 0.008817 ± 0.000107193 (W 1.81659e-05) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.014176 ± 0.000174072 (W 0.00012096) | 0.018454 ± 0.000191596 (W 0.000283579) | 0.074772 ± 0.00217301 (W 0.00292867) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 5.27× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.013906 ± 3.05014e-05 (W 0.000120259) | 0.018384 ± 0.00016204 (W 0.000262022) | 0.058624 ± 1.97569e-05 (W 0.000320336) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 4.22× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.029473 ± 9.30287e-05 (W 0.000251039) | 0.05335 ± 0.000200176 (W 0.000321228) | 0.09557 ± 0.00110135 (W 0.00476075) | 🔵 colmat 1.81× faster than `faer` | 🔵 colmat 3.24× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.029542 ± 0.000147453 (W 0.000212802) | 0.052699 ± 0.000423345 (W 0.000653777) | 0.205865 ± 0.000468187 (W 0.00204264) | 🔵 colmat 1.78× faster than `faer` | 🔵 colmat 6.97× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.017663 ± 0.000193357 (W 0.000161333) | 0.050424 ± 0.000217924 (W 0.000549709) | 0.01603 ± 0.000186244 (W 7.80205e-05) | 🔵 colmat 2.85× faster than `faer` | 🔴 colmat 1.10× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr/n32/b1 [api-inclusive]` | 0.020128 ± 0.000195594 (W 0.000201892) | 0.05379 ± 0.000433387 (W 0.000639491) | - | 🔵 colmat 2.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.061529 ± 0.00070754 (W 0.000327724) | 0.218058 ± 0.00187818 (W 0.00250722) | 0.099365 ± 0.00132626 (W 0.000122296) | 🔵 colmat 3.54× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.066517 ± 0.000610993 (W 0.000344094) | 0.227943 ± 0.00599968 (W 0.00160967) | - | 🔵 colmat 3.43× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.13923 ± 0.00128412 (W 0.00216861) | 0.440719 ± 0.00361953 (W 0.00311312) | 0.340496 ± 0.000210975 (W 0.00304293) | 🔵 colmat 3.17× faster than `faer` | 🔵 colmat 2.45× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.183092 ± 0.00098397 (W 0.00342687) | 0.48613 ± 0.00522659 (W 0.00445583) | - | 🔵 colmat 2.66× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.009308 ± 0.000110077 (W 0.000128426) | 0.039965 ± 0.000270759 (W 0.000682827) | 0.007284 ± 4.55741e-05 (W 3.50029e-05) | 🔵 colmat 4.29× faster than `faer` | 🔴 colmat 1.28× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.034334 ± 6.61085e-05 (W 0.000339168) | 0.188832 ± 0.00158962 (W 0.0025462) | 0.040947 ± 0.000524927 (W 0.000137867) | 🔵 colmat 5.50× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.088837 ± 0.000672742 (W 0.00032964) | 0.373898 ± 0.0043377 (W 0.00438613) | 0.134824 ± 0.00251785 (W 0.000439085) | 🔵 colmat 4.21× faster than `faer` | 🔵 colmat 1.52× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.17453 ± 0.000681484 (W 0.00200017) | 0.193272 ± 0.000150533 (W 0.00281274) | 0.167707 ± 0.000474031 (W 0.00289534) | 🔵 colmat 1.11× faster than `faer` | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.175541 ± 0.00109621 (W 0.00168453) | 0.194897 ± 0.0010011 (W 0.00246116) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.5641 ± 0.00229161 (W 0.00226916) | 0.799537 ± 0.000918744 (W 0.00230677) | 0.638371 ± 0.00223335 (W 0.00393483) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.569229 ± 0.0011279 (W 0.00230254) | 0.800876 ± 0.000910275 (W 0.00167105) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.15529 ± 0.00591679 (W 0.00359673) | 1.81066 ± 0.00114527 (W 0.00256525) | 1.49602 ± 0.0045647 (W 0.0065409) | 🔵 colmat 1.57× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.16569 ± 0.00376218 (W 0.00137658) | 1.81902 ± 0.00104998 (W 0.00257464) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.060703 ± 0.000240176 (W 0.000184489) | 0.14129 ± 0.000136847 (W 0.00171208) | 0.119575 ± 0.000403512 (W 0.00217695) | 🔵 colmat 2.33× faster than `faer` | 🔵 colmat 1.97× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.060807 ± 0.000426606 (W 0.00016653) | 0.141817 ± 0.000504425 (W 0.000251585) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.226887 ± 0.000456536 (W 0.0019794) | 0.540036 ± 0.00273453 (W 0.0029074) | 0.238352 ± 0.00164425 (W 0.00253083) | 🔵 colmat 2.38× faster than `faer` | · |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.227312 ± 0.0012562 (W 0.00219294) | 0.54197 ± 0.00157285 (W 0.00355184) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.560437 ± 0.00179059 (W 0.00227784) | 1.19256 ± 0.00575425 (W 0.00218015) | 0.600731 ± 0.00246623 (W 0.00238251) | 🔵 colmat 2.13× faster than `faer` | 🔵 colmat 1.07× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.565286 ± 0.00430427 (W 0.00201269) | 1.20203 ± 0.00708069 (W 0.00187585) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001012 ± 1.52753e-05 (W 4.1282e-05) | 0.000791 ± 1.1547e-05 (W 3.28968e-05) | 0.000782 ± 1.1547e-05 (W 5.23288e-05) | 🔴 colmat 1.28× slower than `faer` | 🔴 colmat 1.29× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002505 ± 3.78594e-05 (W 3.20936e-05) | 0.001964 ± 5.7735e-06 (W 2.00075e-05) | 0.001432 ± 1.18462e-05 (W 4.13074e-05) | 🔴 colmat 1.28× slower than `faer` | 🔴 colmat 1.75× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004238 ± 2.51661e-05 (W 6.44205e-05) | 0.003487 ± 3.9879e-05 (W 3.93357e-05) | 0.002184 ± 1.52753e-05 (W 3.27109e-05) | 🔴 colmat 1.22× slower than `faer` | 🔴 colmat 1.94× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.120267 ± 0.00103298 (W 0.000144821) | 0.143638 ± 0.000124852 (W 0.00203223) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.119753 ± 0.000335163 (W 0.000243714) | 0.144109 ± 0.000569995 (W 0.00177398) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.582217 ± 0.000361807 (W 0.00266425) | 0.61079 ± 0.00138103 (W 0.00262696) | - | · | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.583048 ± 0.0031799 (W 0.00327667) | 0.631077 ± 0.0030198 (W 0.00194732) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.46704 ± 0.00306486 (W 0.00440249) | 1.51855 ± 0.00547735 (W 0.00440709) | - | · | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.46994 ± 0.000592979 (W 0.00379969) | 1.52336 ± 0.00314134 (W 0.00411319) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.018485 ± 7.0406e-05 (W 0.000157109) | 0.014528 ± 0.000277417 (W 0.00163465) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.022963 ± 8.88819e-05 (W 0.000267999) | 0.01608 ± 3.21455e-05 (W 0.000105191) | - | 🔴 colmat 1.43× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.094496 ± 0.000410688 (W 0.0019141) | 0.08623 ± 0.000221618 (W 0.000357241) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.10729 ± 0.000293533 (W 0.00190435) | 0.090831 ± 0.00019551 (W 0.000597645) | - | 🔴 colmat 1.18× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.279651 ± 0.00207631 (W 0.00222259) | 0.251178 ± 0.00159509 (W 0.00167999) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.346363 ± 0.00226849 (W 0.00221861) | 0.300099 ± 0.00059941 (W 0.00258047) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028313 ± 4.55741e-05 (W 0.000112827) | 0.027391 ± 3.4641e-05 (W 0.000104534) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.028373 ± 0.000131607 (W 0.000183437) | 0.027362 ± 4.59021e-05 (W 6.1305e-05) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.15954 ± 0.000462074 (W 0.00175521) | 0.156371 ± 0.000218111 (W 0.00222823) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.159357 ± 0.000650539 (W 0.0020887) | 0.155961 ± 0.000589603 (W 0.000642539) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.470257 ± 0.000856324 (W 0.00173626) | 0.46665 ± 0.00200691 (W 0.00253934) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.472712 ± 0.00420704 (W 0.000952116) | 0.464305 ± 0.00178835 (W 0.00237836) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.070241 ± 0.000563076 (W 0.00245399) | 0.078777 ± 0.000120504 (W 0.00173125) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.040065 ± 0.00024964 (W 0.000141405) | 0.06974 ± 0.000111122 (W 0.000205799) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.289059 ± 0.00164586 (W 0.00268678) | 0.387713 ± 0.0020138 (W 0.00248644) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.256283 ± 0.00165488 (W 0.00291859) | 0.330276 ± 0.00142563 (W 0.00293947) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.715765 ± 0.00264577 (W 0.00257326) | 0.912371 ± 0.00282598 (W 0.00341666) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.697982 ± 0.00282701 (W 0.00386414) | 0.851539 ± 0.00345512 (W 0.00243642) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.03708 ± 0.000217025 (W 0.000279421) | 0.045965 ± 0.000217909 (W 0.000156191) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.145324 ± 0.000656123 (W 0.00176403) | 0.236771 ± 0.00147547 (W 0.00242563) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.346747 ± 0.0013233 (W 0.00218607) | 0.540809 ± 0.00411749 (W 0.00262101) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.213228 ± 0.00100791 (W 0.0024192) | 0.24663 ± 0.000348691 (W 0.00308777) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.214951 ± 0.00219723 (W 0.00154739) | 0.247131 ± 0.000266581 (W 0.00219479) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.02628 ± 0.00582214 (W 0.00160327) | 1.08255 ± 0.00398922 (W 0.00180245) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.02959 ± 0.00462081 (W 0.00104744) | 1.09048 ± 0.00395989 (W 0.00334485) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.4864 ± 0.00671151 (W 0.00477777) | 2.81746 ± 0.00930206 (W 0.00530629) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.50441 ± 0.0130465 (W 0.00337796) | 2.81798 ± 0.00785429 (W 0.00472821) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.076863 ± 0.000268715 (W 0.00141641) | 0.162055 ± 0.000241719 (W 0.000269982) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.077484 ± 0.000273132 (W 9.3232e-05) | 0.164527 ± 0.000918219 (W 0.00189708) | - | 🔵 colmat 2.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.328102 ± 0.00136319 (W 0.00261572) | 0.67041 ± 0.00329798 (W 0.00265487) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.328473 ± 0.000582173 (W 0.00206223) | 0.667234 ± 0.00235343 (W 0.00290595) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.836106 ± 0.00190584 (W 0.00188026) | 1.54968 ± 0.00757217 (W 0.00339598) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.839725 ± 0.00285894 (W 0.00368598) | 1.55834 ± 0.00591615 (W 0.00382984) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001282 ± 1e-05 (W 2.75372e-05) | 0.001143 ± 3.4356e-05 (W 4.63595e-05) | 0.001332 ± 1.12694e-05 (W 3.19875e-05) | 🔴 colmat 1.12× slower than `faer` | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003186 ± 1e-05 (W 2.53417e-05) | 0.002906 ± 5.54106e-05 (W 3.3975e-05) | 0.002816 ± 3.59212e-05 (W 3.06969e-05) | 🔴 colmat 1.10× slower than `faer` | 🔴 colmat 1.13× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005801 ± 4.04145e-05 (W 5.4037e-05) | 0.00541 ± 6.1101e-05 (W 3.68538e-05) | 0.004769 ± 1.52753e-05 (W 3.81077e-05) | 🔴 colmat 1.07× slower than `faer` | 🔴 colmat 1.22× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.094005 ± 0.001842 (W 0.000204363) | 0.143778 ± 0.000177725 (W 0.0016744) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.092663 ± 0.000305372 (W 0.000243794) | 0.14459 ± 0.000379648 (W 0.0025964) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.311005 ± 0.000992214 (W 0.00237588) | 0.584651 ± 0.00134322 (W 0.00210531) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.312874 ± 0.00137727 (W 0.00211087) | 0.586544 ± 0.000759222 (W 0.00250261) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.687004 ± 0.000791521 (W 0.00243454) | 1.35567 ± 0.00205849 (W 0.0029454) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.694558 ± 0.00096048 (W 0.00249904) | 1.36068 ± 0.00369354 (W 0.0024586) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019447 ± 0.00019486 (W 0.000135053) | 0.014657 ± 5.72742e-05 (W 0.000169912) | - | 🔴 colmat 1.33× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.024375 ± 0.000111132 (W 0.000166383) | 0.0159 ± 6.37992e-05 (W 8.24269e-05) | - | 🔴 colmat 1.53× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.089687 ± 0.000311515 (W 0.000612712) | 0.085259 ± 0.000874933 (W 0.000514988) | - | 🔴 colmat 1.05× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.103986 ± 0.00029677 (W 0.00150936) | 0.090599 ± 0.000638846 (W 0.00061272) | - | 🔴 colmat 1.15× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.200053 ± 0.000581962 (W 0.00191523) | 0.250648 ± 0.000713632 (W 0.00245787) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.27582 ± 0.00284352 (W 0.00363763) | 0.253654 ± 0.00205519 (W 0.00253065) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.013715 ± 0.000246354 (W 0.000340632) | 0.027791 ± 6.19785e-05 (W 0.000111265) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.013115 ± 6.55744e-05 (W 9.05389e-05) | 0.027872 ± 0.000101425 (W 0.000112383) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.057377 ± 0.00030402 (W 0.000600452) | 0.055124 ± 0.000222299 (W 0.000566422) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.056977 ± 0.000261496 (W 0.000504783) | 0.054863 ± 0.000209643 (W 0.000393354) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.147996 ± 0.000442337 (W 0.000453835) | 0.145311 ± 0.000387248 (W 0.000533936) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.147925 ± 0.000938054 (W 0.000500926) | 0.145334 ± 0.000323054 (W 0.00198672) | - | · | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.065492 ± 0.000406 (W 0.00048802) | 0.093295 ± 0.000444744 (W 0.00183618) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.040495 ± 0.000300936 (W 9.54479e-05) | 0.083957 ± 0.0011127 (W 0.00301538) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.147405 ± 0.0010573 (W 0.00126113) | 0.400598 ± 0.00145047 (W 0.00284474) | - | 🔵 colmat 2.72× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.18189 ± 0.000569761 (W 0.00251392) | 0.33757 ± 0.000707956 (W 0.00323992) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.40691 ± 0.0048382 (W 0.00297905) | 0.76675 ± 0.00316602 (W 0.00355998) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.507216 ± 0.0049352 (W 0.000912121) | 0.703443 ± 0.00399799 (W 0.00353573) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.03767 ± 0.000418786 (W 0.000911989) | 0.060994 ± 0.000265632 (W 0.000510041) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.090609 ± 0.000532992 (W 0.000270934) | 0.291604 ± 0.00208535 (W 0.00194672) | - | 🔵 colmat 3.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.267719 ± 0.00109983 (W 0.00280695) | 0.575523 ± 0.00234285 (W 0.00260495) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.211538 ± 0.00140376 (W 0.0017195) | 0.246865 ± 0.000444469 (W 0.00252457) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.214891 ± 0.000267105 (W 0.00310154) | 0.247061 ± 0.00131504 (W 0.00250481) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.743848 ± 0.00208417 (W 0.00266278) | 1.0149 ± 0.00176293 (W 0.000659118) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.748427 ± 0.00158605 (W 0.00242874) | 1.01712 ± 0.00193962 (W 0.000835877) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54619 ± 0.00450309 (W 0.00329912) | 2.46965 ± 0.00499702 (W 0.00313605) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.54768 ± 0.00683441 (W 0.00394257) | 2.50589 ± 0.00128362 (W 0.00310363) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.077496 ± 0.00026887 (W 0.000133978) | 0.162196 ± 0.000358659 (W 0.00242501) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.078427 ± 0.000118874 (W 0.000183494) | 0.162343 ± 0.000566394 (W 0.0025654) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.321956 ± 0.000864713 (W 0.00164623) | 0.667928 ± 0.00196838 (W 0.00295184) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.322481 ± 0.00106105 (W 0.00166742) | 0.668298 ± 0.00238349 (W 0.00226268) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.7237 ± 0.000446869 (W 0.00318591) | 1.54586 ± 0.00149513 (W 0.00378144) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.722411 ± 0.00200886 (W 0.00194605) | 1.55472 ± 0.00414143 (W 0.00314913) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001543 ± 2.08167e-05 (W 2.82843e-05) | 0.001163 ± 5.32948e-05 (W 3.57771e-05) | 0.001322 ± 0 (W 3.28983e-05) | 🔴 colmat 1.33× slower than `faer` | 🔴 colmat 1.17× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003707 ± 1e-05 (W 3.83406e-05) | 0.002925 ± 1.70392e-05 (W 8.79562e-05) | 0.002825 ± 0.00019789 (W 3.39116e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.31× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006593 ± 1.18462e-05 (W 6.97087e-05) | 0.00542 ± 5.7735e-06 (W 5.54076e-05) | 0.004729 ± 3.4641e-05 (W 3.11448e-05) | 🔴 colmat 1.22× slower than `faer` | 🔴 colmat 1.39× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.087914 ± 0.000537584 (W 0.000148544) | 0.14455 ± 0.000284132 (W 0.000351909) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.087536 ± 0.000418006 (W 0.000347698) | 0.144389 ± 0.000559681 (W 0.000411762) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.267329 ± 0.000704128 (W 0.00239762) | 0.586188 ± 0.00417636 (W 0.00273123) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.270436 ± 0.000791343 (W 0.00147313) | 0.584201 ± 0.00102578 (W 0.00214178) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.540109 ± 0.00685159 (W 0.00257164) | 1.36159 ± 0.00644396 (W 0.00376835) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.545776 ± 0.00339363 (W 0.00286075) | 1.36477 ± 0.00104652 (W 0.00271055) | - | 🔵 colmat 2.50× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019665 ± 0.000215695 (W 0.000106771) | 0.014698 ± 9.45745e-05 (W 4.84799e-05) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.024306 ± 0.000218657 (W 0.000180023) | 0.016229 ± 8.12055e-05 (W 0.000152106) | - | 🔴 colmat 1.50× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.093225 ± 0.00182504 (W 0.000464897) | 0.085521 ± 0.00052441 (W 0.00145239) | - | 🔴 colmat 1.09× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.107701 ± 0.000338853 (W 0.000882942) | 0.090762 ± 0.000605526 (W 0.00153634) | - | 🔴 colmat 1.19× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.195806 ± 0.000306843 (W 0.00232658) | 0.25307 ± 0.00176218 (W 0.00281404) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.285237 ± 0.0360983 (W 0.0060202) | 0.253812 ± 0.000761579 (W 0.00222207) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.010009 ± 4.0501e-05 (W 0.000143319) | 0.027872 ± 8.70689e-05 (W 0.000114718) | - | 🔵 colmat 2.78× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009888 ± 0.000112518 (W 9.64692e-05) | 0.02792 ± 5.56687e-05 (W 6.80441e-05) | - | 🔵 colmat 2.82× faster than `faer` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.034554 ± 1.3e-05 (W 0.00285116) | 0.055229 ± 1.66433e-05 (W 0.000455063) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.035016 ± 0.00024833 (W 0.000471843) | 0.055299 ± 0.00037592 (W 0.000354009) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.083477 ± 0.000532112 (W 0.000505997) | 0.14498 ± 0.000695705 (W 0.00168872) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.083336 ± 0.000419558 (W 0.000625603) | 0.145453 ± 0.000660214 (W 0.00222216) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.068348 ± 0.000981687 (W 0.001072) | 0.101079 ± 0.000289835 (W 0.00149007) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.040606 ± 0.000724065 (W 0.000104642) | 0.092715 ± 0.00105053 (W 0.00131124) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.137429 ± 0.000920215 (W 0.000529286) | 0.429362 ± 0.00220176 (W 0.00580264) | - | 🔵 colmat 3.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.173045 ± 0.00107393 (W 0.00276309) | 0.383817 ± 0.000812493 (W 0.00487225) | - | 🔵 colmat 2.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.34275 ± 0.00232159 (W 0.00295157) | 0.822567 ± 0.00404838 (W 0.00940073) | - | 🔵 colmat 2.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.462144 ± 0.00102311 (W 0.00218447) | 0.774632 ± 0.00467946 (W 0.00330938) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.039384 ± 0.000915365 (W 0.00110167) | 0.07005 ± 0.00113743 (W 0.000596686) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.090809 ± 0.00115983 (W 0.001942) | 0.326451 ± 0.00317927 (W 0.00624278) | - | 🔵 colmat 3.59× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.241089 ± 0.00061189 (W 0.00272387) | 0.636037 ± 0.00393779 (W 0.00742867) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.211498 ± 0.000668603 (W 0.00208506) | 0.247312 ± 0.000983727 (W 0.00266605) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.213722 ± 0.000458532 (W 0.00191799) | 0.248264 ± 0.000799237 (W 0.00239612) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.72797 ± 0.00378793 (W 0.00394481) | 1.01822 ± 0.000863682 (W 0.00108897) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.732979 ± 0.00559347 (W 0.00208323) | 1.02056 ± 0.00119166 (W 0.00126388) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.48499 ± 0.00917603 (W 0.00558899) | 2.48341 ± 0.00444387 (W 0.00342291) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.49159 ± 0.0101288 (W 0.00428858) | 2.48949 ± 0.00575555 (W 0.0025263) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.078035 ± 0.000674754 (W 0.000125369) | 0.162325 ± 0.000679093 (W 0.00170812) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.07872 ± 0.000131081 (W 0.000153907) | 0.163742 ± 0.000561228 (W 0.000122066) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.332255 ± 0.000994723 (W 0.00266864) | 0.667698 ± 0.00101944 (W 0.00307275) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.332386 ± 0.00120398 (W 0.00181007) | 0.670541 ± 0.00402658 (W 0.00216584) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.741264 ± 0.00267425 (W 0.00271829) | 1.55415 ± 0.00342548 (W 0.00513815) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.738318 ± 0.0037793 (W 0.00259995) | 1.5578 ± 0.000280228 (W 0.0029996) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001553 ± 1.52753e-05 (W 3.78153e-05) | 0.001182 ± 5.74195e-05 (W 2.48998e-05) | 0.001333 ± 1.48436e-05 (W 3.36942e-05) | 🔴 colmat 1.31× slower than `faer` | 🔴 colmat 1.17× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003747 ± 1.1547e-05 (W 3.36155e-05) | 0.002946 ± 2.66521e-05 (W 2.85867e-05) | 0.002826 ± 1.12694e-05 (W 3.26374e-05) | 🔴 colmat 1.27× slower than `faer` | 🔴 colmat 1.33× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006692 ± 1.18462e-05 (W 4.60435e-05) | 0.00541 ± 2.51661e-05 (W 5.1672e-05) | 0.004759 ± 6.02771e-05 (W 3.17065e-05) | 🔴 colmat 1.24× slower than `faer` | 🔴 colmat 1.41× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.120545 ± 0.000432074 (W 0.00193207) | 0.143878 ± 0.000149018 (W 0.00207095) | 0.155643 ± 0.00169359 (W 0.00212522) | 🔵 colmat 1.19× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.119643 ± 0.000572854 (W 0.000277768) | 0.144131 ± 0.000203443 (W 0.00220448) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.579982 ± 0.00178125 (W 0.00231551) | 0.609657 ± 0.00186203 (W 0.0024455) | 0.719613 ± 0.00499954 (W 0.00171062) | · | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.581946 ± 0.000627763 (W 0.00331592) | 0.632973 ± 0.000884419 (W 0.00603051) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.46401 ± 0.0060198 (W 0.00310323) | 1.51694 ± 0.0032515 (W 0.00370942) | 1.80203 ± 0.0027787 (W 0.00419021) | · | 🔵 colmat 1.23× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.4686 ± 0.000910959 (W 0.00308406) | 1.51778 ± 0.00604157 (W 0.00262383) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019005 ± 0.000220659 (W 0.000180638) | 0.014627 ± 0.000151438 (W 7.60263e-05) | 0.157938 ± 0.00248149 (W 0.00278647) | 🔴 colmat 1.30× slower than `faer` | 🔵 colmat 8.31× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.020759 ± 0.00010843 (W 0.000120711) | 0.016301 ± 0.000127017 (W 0.000109033) | - | 🔴 colmat 1.27× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.09614 ± 0.00103676 (W 0.000203035) | 0.085981 ± 0.000185831 (W 0.00165149) | 2.71931 ± 0.0223103 (W 0.00580635) | 🔴 colmat 1.12× slower than `faer` | 🔵 colmat 28.28× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.100147 ± 0.000633083 (W 0.00231208) | 0.089437 ± 0.0010188 (W 0.000734419) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.282918 ± 0.00191638 (W 0.00197026) | 0.25226 ± 0.0013043 (W 0.00120286) | 9.30197 ± 0.0761135 (W 0.0140333) | 🔴 colmat 1.12× slower than `faer` | 🔵 colmat 32.88× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.329474 ± 0.00225429 (W 0.00213938) | 0.296443 ± 0.00139533 (W 0.00212963) | - | 🔴 colmat 1.11× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.028294 ± 0.000255148 (W 0.000131158) | 0.027371 ± 8.20183e-05 (W 0.0001118) | 0.041227 ± 0.000125831 (W 8.43107e-05) | · | 🔵 colmat 1.46× faster than `lapack-aocl` |  |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.017363 ± 4.98698e-05 (W 4.74057e-05) | 0.027421 ± 4.93288e-05 (W 5.4037e-05) | 0.020498 ± 3.52184e-05 (W 8.60947e-05) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.18× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.157203 ± 0.000351187 (W 0.00240388) | 0.155713 ± 0.000640846 (W 0.00207715) | 0.199692 ± 0.00218524 (W 0.00228571) | · | 🔵 colmat 1.27× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.157837 ± 0.000616159 (W 0.00243955) | 0.154549 ± 0.000210209 (W 0.000629921) | 0.250577 ± 0.00219714 (W 0.0023954) | · | 🔵 colmat 1.59× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.459708 ± 0.00254984 (W 0.00293677) | 0.46078 ± 0.00260279 (W 0.000990849) | 0.475417 ± 0.00296292 (W 0.00388838) | · | · |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.462902 ± 0.00345115 (W 0.00272277) | 0.459067 ± 0.00325041 (W 0.00139037) | 0.532874 ± 0.00238467 (W 0.00222775) | · | 🔵 colmat 1.15× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.035135 ± 0.00144441 (W 8.58487e-05) | 0.067105 ± 0.000119626 (W 0.000407539) | 0.042049 ± 0.000526508 (W 0.000157915) | 🔵 colmat 1.91× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/n32/b1 [api-inclusive]` | 0.038202 ± 0.000189201 (W 0.000140848) | 0.069181 ± 0.000857827 (W 0.000366269) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.21725 ± 0.00237298 (W 0.0022682) | 0.298667 ± 0.000369809 (W 0.00206502) | 0.373474 ± 0.00182925 (W 0.00245668) | 🔵 colmat 1.37× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.251208 ± 0.0013224 (W 0.00363603) | 0.327029 ± 0.00191477 (W 0.00410473) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.586324 ± 0.00238499 (W 0.00276003) | 0.749899 ± 0.0029661 (W 0.00430349) | 0.897774 ± 0.000537244 (W 0.00147267) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 1.53× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.690478 ± 0.000809015 (W 0.0038164) | 0.845566 ± 0.00473203 (W 0.00285743) | - | 🔵 colmat 1.22× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.015679 ± 5.68624e-05 (W 7.15975e-05) | 0.034264 ± 0.000322909 (W 0.000262993) | 0.020618 ± 8.66025e-05 (W 9.795e-05) | 🔵 colmat 2.19× faster than `faer` | 🔵 colmat 1.32× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.076413 ± 0.000285365 (W 0.000104543) | 0.161885 ± 0.00201233 (W 0.00270102) | 0.12338 ± 0.00215779 (W 0.000234859) | 🔵 colmat 2.12× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.214921 ± 0.00110461 (W 0.00245464) | 0.393605 ± 0.00155421 (W 0.00318399) | 0.358578 ± 0.00140715 (W 0.00265071) | 🔵 colmat 1.83× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.210292 ± 0.00102556 (W 0.00261902) | 0.245318 ± 0.00146841 (W 0.00239288) | 0.235474 ± 0.00103342 (W 0.00191968) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.12× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.21212 ± 0.000835824 (W 0.00237305) | 0.243358 ± 0.000621853 (W 0.0023013) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 1.01801 ± 0.00228689 (W 0.00178925) | 1.10865 ± 0.00426854 (W 0.0010789) | 0.97612 ± 0.0035191 (W 0.00260476) | 🔵 colmat 1.09× faster than `faer` | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 1.01694 ± 0.001343 (W 0.00224424) | 1.10742 ± 0.00104539 (W 0.00293765) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.47902 ± 0.00285664 (W 0.00576351) | 2.77911 ± 0.00275592 (W 0.00638712) | 2.61814 ± 0.00953233 (W 0.00430456) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.06× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.48517 ± 0.0028526 (W 0.00377676) | 2.78856 ± 0.00291454 (W 0.00740598) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.077745 ± 0.000227568 (W 0.00170297) | 0.16071 ± 0.000752536 (W 0.00203991) | 0.108695 ± 0.000322186 (W 0.000255061) | 🔵 colmat 2.07× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.078007 ± 0.000111899 (W 0.00018177) | 0.162123 ± 0.000532505 (W 0.00203363) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.328192 ± 0.00213466 (W 0.00222122) | 0.640385 ± 0.00308243 (W 0.00299636) | 0.421116 ± 0.00209581 (W 0.00251745) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.326579 ± 0.00116517 (W 0.00313832) | 0.639273 ± 0.00251561 (W 0.00302135) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.8378 ± 0.000899235 (W 0.00353058) | 1.52171 ± 0.00991135 (W 0.0046551) | 1.02973 ± 0.00925995 (W 0.00117529) | 🔵 colmat 1.82× faster than `faer` | 🔵 colmat 1.23× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.834397 ± 0.00197665 (W 0.00318182) | 1.5193 ± 0.00343147 (W 0.00423994) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001303 ± 1.18462e-05 (W 3.08269e-05) | 0.001352 ± 3.19427e-05 (W 4.40148e-05) | 0.001212 ± 1.52753e-05 (W 3.13417e-05) | · | 🔴 colmat 1.08× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003457 ± 2.09841e-05 (W 4.19917e-05) | 0.003577 ± 1.95021e-05 (W 3.09079e-05) | 0.002585 ± 1e-05 (W 3.60555e-05) | · | 🔴 colmat 1.34× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005681 ± 2.66521e-05 (W 5.16362e-05) | 0.005731 ± 1.1547e-05 (W 5.24519e-05) | 0.004378 ± 2.3094e-05 (W 2.70185e-05) | · | 🔴 colmat 1.30× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.093986 ± 0.000658795 (W 0.000239754) | 0.143941 ± 0.00034738 (W 0.00190865) | 0.153487 ± 0.000911346 (W 0.00247126) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 1.63× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.092733 ± 7.77839e-05 (W 0.00184741) | 0.144319 ± 0.000543313 (W 0.00233803) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.311797 ± 0.00185273 (W 0.00255224) | 0.582937 ± 0.000896644 (W 0.00252897) | 0.731064 ± 0.00263859 (W 0.00403934) | 🔵 colmat 1.87× faster than `faer` | 🔵 colmat 2.34× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.312773 ± 0.00155163 (W 0.00167066) | 0.5845 ± 0.00261201 (W 0.00249473) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.687973 ± 0.000957002 (W 0.00296152) | 1.35572 ± 0.00166753 (W 0.00355932) | 1.84085 ± 0.0223495 (W 0.00295333) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 2.68× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.693586 ± 0.000384969 (W 0.00215513) | 1.35575 ± 0.00366378 (W 0.0029619) | - | 🔵 colmat 1.95× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019727 ± 0.000217601 (W 9.5551e-05) | 0.014617 ± 0.000105879 (W 0.000144375) | 0.158528 ± 0.00176803 (W 0.0022964) | 🔴 colmat 1.35× slower than `faer` | 🔵 colmat 8.04× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.021831 ± 0.000277909 (W 0.00013362) | 0.016311 ± 0.000122312 (W 4.6479e-05) | - | 🔴 colmat 1.34× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.092855 ± 0.00296814 (W 0.0011639) | 0.08582 ± 0.000261052 (W 0.00190835) | 2.70158 ± 0.00451153 (W 0.0127749) | 🔴 colmat 1.08× slower than `faer` | 🔵 colmat 29.09× faster than `lapack-aocl` | `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.096752 ± 0.000938829 (W 0.00080525) | 0.089327 ± 0.000991269 (W 0.00131113) | - | 🔴 colmat 1.08× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.207948 ± 0.00205659 (W 0.0017138) | 0.250421 ± 0.000927358 (W 0.00206859) | 9.29142 ± 0.019166 (W 0.0146444) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 44.68× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.26054 ± 0.00158425 (W 0.00302406) | 0.250151 ± 0.00075283 (W 0.00217505) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.013355 ± 0.000172879 (W 0.000144362) | 0.027682 ± 8.0829e-05 (W 9.8185e-05) | 0.065542 ± 0.000155802 (W 0.000976047) | 🔵 colmat 2.07× faster than `faer` | 🔵 colmat 4.91× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.013545 ± 7.26659e-05 (W 0.000118782) | 0.027902 ± 7.25282e-05 (W 8.61713e-05) | 0.020679 ± 0.000304056 (W 6.88062e-05) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.53× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.057116 ± 0.000329075 (W 0.000729932) | 0.055013 ± 0.000392021 (W 0.000298599) | 0.134033 ± 0.000356556 (W 0.000899658) | · | 🔵 colmat 2.35× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.056616 ± 9.20887e-05 (W 0.000347002) | 0.055183 ± 0.000130495 (W 0.000142872) | 0.150591 ± 0.00300633 (W 0.00239592) | · | 🔵 colmat 2.66× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.147175 ± 0.00109273 (W 0.000457112) | 0.143899 ± 0.000503544 (W 0.00191355) | 0.214724 ± 0.00232976 (W 0.00242224) | · | 🔵 colmat 1.46× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.147287 ± 0.000362062 (W 0.000165015) | 0.144111 ± 0.000685085 (W 0.000511323) | 0.261523 ± 0.00286141 (W 0.00289797) | · | 🔵 colmat 1.78× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.034684 ± 0.00027561 (W 0.000105206) | 0.081041 ± 0.00039289 (W 0.000599023) | 0.041247 ± 0.000162194 (W 7.59605e-05) | 🔵 colmat 2.34× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.038262 ± 0.00025644 (W 0.000141143) | 0.084007 ± 0.000359212 (W 0.00175904) | - | 🔵 colmat 2.20× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.137148 ± 0.000745515 (W 0.000409934) | 0.307389 ± 0.00110991 (W 0.00231077) | 0.278509 ± 0.00159731 (W 0.0036438) | 🔵 colmat 2.24× faster than `faer` | 🔵 colmat 2.03× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.173774 ± 0.00151731 (W 0.00364626) | 0.335883 ± 0.000855654 (W 0.00243311) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.381222 ± 0.00201003 (W 0.00246876) | 0.585993 ± 0.00196169 (W 0.00280506) | 0.646255 ± 0.0017805 (W 0.00486479) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.492059 ± 0.0044586 (W 0.00329128) | 0.695698 ± 0.00237262 (W 0.00266307) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.015599 ± 7.57188e-05 (W 5.37513e-05) | 0.048131 ± 0.000262565 (W 0.000350263) | 0.020508 ± 8.79223e-05 (W 9.09395e-05) | 🔵 colmat 3.09× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.076052 ± 0.000129693 (W 0.000111249) | 0.208609 ± 0.000451358 (W 0.00284302) | 0.12203 ± 0.00123646 (W 0.000279676) | 🔵 colmat 2.74× faster than `faer` | 🔵 colmat 1.60× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.245758 ± 0.00122512 (W 0.00190899) | 0.423528 ± 0.00133604 (W 0.00365815) | 0.380962 ± 0.000220675 (W 0.00253355) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 1.55× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.209922 ± 4.55668e-05 (W 0.00171522) | 0.243213 ± 0.000657793 (W 0.00201148) | 0.234317 ± 0.000210495 (W 0.00264456) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 1.12× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.212327 ± 0.000862649 (W 0.00199328) | 0.245187 ± 0.000761987 (W 0.00268191) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.733721 ± 0.00110211 (W 0.00237481) | 1.03842 ± 0.000181643 (W 0.0029904) | 0.98498 ± 0.00264846 (W 0.00146811) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.734973 ± 0.000442595 (W 0.00258265) | 1.04199 ± 0.00119138 (W 0.00121053) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54087 ± 0.00450736 (W 0.00510306) | 2.43325 ± 0.00569932 (W 0.00425876) | 2.20247 ± 0.0105245 (W 0.0033077) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.43× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55088 ± 0.00131689 (W 0.00281856) | 2.4414 ± 0.018541 (W 0.00445381) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.078637 ± 0.000510523 (W 0.000115937) | 0.159778 ± 0.000113899 (W 0.00140435) | 0.109715 ± 0.000674144 (W 0.00219129) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.079298 ± 0.000396588 (W 0.000242679) | 0.160029 ± 0.00107171 (W 0.00193951) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.321771 ± 0.00174524 (W 0.0019826) | 0.641638 ± 0.00309505 (W 0.00413062) | 0.422367 ± 0.00387855 (W 0.00195661) | 🔵 colmat 1.99× faster than `faer` | 🔵 colmat 1.31× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.322612 ± 0.000665554 (W 0.00225285) | 0.639995 ± 0.00195543 (W 0.00343325) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.721328 ± 0.00249713 (W 0.00287876) | 1.52199 ± 0.00280884 (W 0.002672) | 1.0693 ± 0.0194624 (W 0.00285719) | 🔵 colmat 2.11× faster than `faer` | 🔵 colmat 1.48× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.719303 ± 0.00321325 (W 0.00292534) | 1.53068 ± 0.00552556 (W 0.00273269) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001542 ± 1.95021e-05 (W 2.73898e-05) | 0.001343 ± 5.7735e-07 (W 2.11731e-05) | 0.001192 ± 5.7735e-06 (W 2.91239e-05) | 🔴 colmat 1.15× slower than `faer` | 🔴 colmat 1.29× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003767 ± 1e-05 (W 4.03733e-05) | 0.003316 ± 2.08167e-05 (W 3.11801e-05) | 0.002545 ± 2.08167e-05 (W 2.44949e-05) | 🔴 colmat 1.14× slower than `faer` | 🔴 colmat 1.48× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006462 ± 3.51188e-05 (W 3.8887e-05) | 0.005761 ± 3.21455e-05 (W 4.76445e-05) | 0.004328 ± 4.72582e-05 (W 2.4294e-05) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 1.49× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n32/b1` | 0.088487 ± 0.000730073 (W 0.000176035) | 0.144128 ± 0.000306193 (W 0.00138973) | 0.154458 ± 0.00194482 (W 0.00154234) | 🔵 colmat 1.63× faster than `faer` | 🔵 colmat 1.75× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.088164 ± 6.9407e-05 (W 0.000286331) | 0.144979 ± 0.000428053 (W 0.00197026) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.266357 ± 0.00133946 (W 0.00170969) | 0.583748 ± 0.00280193 (W 0.00185761) | 0.739202 ± 0.00349955 (W 0.00368182) | 🔵 colmat 2.19× faster than `faer` | 🔵 colmat 2.78× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.268595 ± 0.00107753 (W 0.00204503) | 0.586173 ± 0.000656039 (W 0.00229415) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.538946 ± 0.0781675 (W 0.0033173) | 1.36056 ± 0.00270107 (W 0.00365697) | 1.92395 ± 0.013106 (W 0.00741325) | 🔵 colmat 2.52× faster than `faer` | 🔵 colmat 3.57× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `eigh/n96/b1 [api-inclusive]` | 0.544634 ± 0.0043427 (W 0.0028339) | 1.36434 ± 0.00211 (W 0.00380613) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.019777 ± 0.000614925 (W 0.000119329) | 0.014858 ± 0.000207705 (W 0.00163479) | 0.156114 ± 0.00122278 (W 0.000121149) | 🔴 colmat 1.33× slower than `faer` | 🔵 colmat 7.89× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.021979 ± 0.000110735 (W 0.00021557) | 0.01623 ± 9.5469e-05 (W 6.4283e-05) | - | 🔴 colmat 1.35× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.095259 ± 0.0016103 (W 0.00168739) | 0.084899 ± 0.000853732 (W 0.00172311) | 2.72913 ± 0.0265653 (W 0.00468681) | 🔴 colmat 1.12× slower than `faer` | 🔵 colmat 28.65× faster than `lapack-aocl` |  |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.100889 ± 0.00126256 (W 0.000593688) | 0.089507 ± 0.00055303 (W 0.000712261) | - | 🔴 colmat 1.13× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.203372 ± 0.00193435 (W 0.00255967) | 0.251119 ± 0.000615963 (W 0.00204135) | 9.32814 ± 0.0278229 (W 0.0204298) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 45.87× faster than `lapack-aocl` |  |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.268057 ± 0.00104755 (W 0.00218302) | 0.249463 ± 0.000662262 (W 0.00265228) | - | 🔴 colmat 1.07× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.010349 ± 8.03762e-05 (W 0.000282652) | 0.027792 ± 6.86756e-05 (W 0.000176611) | 0.101471 ± 0.000683347 (W 0.00207015) | 🔵 colmat 2.69× faster than `faer` | 🔵 colmat 9.80× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.010399 ± 0.000111505 (W 0.000101948) | 0.0279 ± 0.00014983 (W 0.000160188) | 0.020789 ± 0.000214919 (W 6.63641e-05) | 🔵 colmat 2.68× faster than `faer` | 🔵 colmat 2.00× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.035276 ± 0.000174311 (W 0.000200204) | 0.055499 ± 0.000382874 (W 0.00131611) | 0.154801 ± 0.00170903 (W 0.00178566) | 🔵 colmat 1.57× faster than `faer` | 🔵 colmat 4.39× faster than `lapack-aocl` |  |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.034996 ± 0.000119202 (W 0.000115889) | 0.05572 ± 0.000302204 (W 0.00149294) | 0.14517 ± 0.00112946 (W 0.00534325) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 4.15× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.083738 ± 0.000177024 (W 0.000329822) | 0.143577 ± 0.000674505 (W 0.000322143) | 0.215002 ± 0.00115721 (W 0.00202698) | 🔵 colmat 1.71× faster than `faer` | 🔵 colmat 2.57× faster than `lapack-aocl` |  |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.083529 ± 0.000578 (W 0.000394388) | 0.14344 ± 0.000200505 (W 0.000379532) | 0.242164 ± 0.00302912 (W 0.00437725) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 2.90× faster than `lapack-aocl` |  |
| `qr/n32/b1` | 0.034755 ± 0.000178029 (W 4.11607e-05) | 0.089127 ± 0.000192775 (W 0.00147883) | 0.041674 ± 0.000451864 (W 0.000213198) | 🔵 colmat 2.56× faster than `faer` | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.038282 ± 0.00149301 (W 0.000159089) | 0.092794 ± 0.000650136 (W 0.00151214) | - | 🔵 colmat 2.42× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n64/b1` | 0.126646 ± 0.00499735 (W 0.00509577) | 0.344583 ± 0.00109382 (W 0.00239816) | 0.273889 ± 0.00216787 (W 0.00433854) | 🔵 colmat 2.72× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/n64/b1 [api-inclusive]` | 0.165422 ± 0.000815507 (W 0.00207825) | 0.378565 ± 0.00132795 (W 0.00241672) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.321891 ± 0.00101366 (W 0.00239282) | 0.660443 ± 0.00259989 (W 0.00385587) | 0.685382 ± 0.00119499 (W 0.00518022) | 🔵 colmat 2.05× faster than `faer` | 🔵 colmat 2.13× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.446677 ± 0.00166347 (W 0.00555773) | 0.778735 ± 0.00355936 (W 0.00541379) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.01594 ± 0.000158308 (W 7.75642e-05) | 0.056997 ± 0.000296541 (W 0.00150549) | 0.020567 ± 0.000109503 (W 0.00163913) | 🔵 colmat 3.58× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.076407 ± 0.000249209 (W 9.65919e-05) | 0.247672 ± 0.00365272 (W 0.00170834) | 0.124281 ± 0.00156206 (W 0.000358102) | 🔵 colmat 3.24× faster than `faer` | 🔵 colmat 1.63× faster than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.227829 ± 0.000180751 (W 0.00169401) | 0.495253 ± 0.00246388 (W 0.00300644) | 0.38158 ± 0.00190043 (W 0.00209527) | 🔵 colmat 2.17× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1` | 0.209534 ± 0.000515481 (W 0.00122717) | 0.245479 ± 0.000677887 (W 0.00258351) | 0.236836 ± 0.00175697 (W 0.00273991) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.210636 ± 0.000582114 (W 0.00160624) | 0.245218 ± 0.000868537 (W 0.00233241) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.718795 ± 0.00356658 (W 0.00205144) | 1.04105 ± 0.00173625 (W 0.000803884) | 0.970734 ± 0.0041543 (W 0.00130136) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 1.35× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.719834 ± 0.00440249 (W 0.00270573) | 1.0454 ± 0.00126977 (W 0.00161603) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.47088 ± 0.00705927 (W 0.00401925) | 2.445 ± 0.004206 (W 0.0031353) | 2.26398 ± 0.0118626 (W 0.0139274) | 🔵 colmat 1.66× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.48584 ± 0.00976614 (W 0.00392111) | 2.45832 ± 0.00595425 (W 0.00648629) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.079228 ± 0.000199271 (W 0.000186603) | 0.161041 ± 0.00072302 (W 0.00273427) | 0.108624 ± 0.00056316 (W 0.00187709) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 1.37× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.080069 ± 0.000174463 (W 0.000106096) | 0.161308 ± 4.72864e-05 (W 0.00217804) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.329536 ± 0.00130475 (W 0.00263503) | 0.642578 ± 0.00120413 (W 0.00736711) | 0.425581 ± 0.00341517 (W 0.00261254) | 🔵 colmat 1.95× faster than `faer` | 🔵 colmat 1.29× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.332516 ± 0.00288559 (W 0.00223735) | 0.646126 ± 0.000534635 (W 0.00238651) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.743217 ± 0.00569687 (W 0.00242866) | 1.52677 ± 0.0028627 (W 0.0052476) | 1.07578 ± 0.00102514 (W 0.00158224) | 🔵 colmat 2.05× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.737076 ± 0.00644052 (W 0.00233107) | 1.52778 ± 0.00374827 (W 0.00584369) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001523 ± 1.1547e-05 (W 2.68328e-05) | 0.001362 ± 5.50757e-06 (W 2.68384e-05) | 0.001192 ± 1.73205e-05 (W 3.5457e-05) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 1.28× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003747 ± 3.4641e-05 (W 2.91548e-05) | 0.003336 ± 7.37111e-05 (W 3.81864e-05) | 0.002545 ± 2.28108e-05 (W 2.2963e-05) | 🔴 colmat 1.12× slower than `faer` | 🔴 colmat 1.47× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006662 ± 3.66106e-05 (W 4.43847e-05) | 0.006071 ± 8.544e-05 (W 4.08656e-05) | 0.004378 ± 2.95014e-05 (W 3.14038e-05) | 🔴 colmat 1.10× slower than `faer` | 🔴 colmat 1.52× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 2.00182 ± 0.0129106 (W 0.00362095) | 2.24024 ± 0.00862196 (W 0.00290365) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.01741 ± 0.00799555 (W 0.00267925) | 2.23631 ± 0.00647753 (W 0.00378173) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.48421 ± 0.0390435 (W 0.0100824) | 9.63355 ± 0.0844365 (W 0.0150704) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 9.46373 ± 0.0459468 (W 0.0117733) | 9.62409 ± 0.0696506 (W 0.0142273) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 48.2094 ± 0.250473 (W 0.0246855) | 47.8469 ± 0.239077 (W 0.0638146) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 48.0692 ± 0.283724 (W 0.0590253) | 48.3545 ± 0.367311 (W 0.0529968) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.326825 ± 0.00260959 (W 0.00278361) | 0.296374 ± 0.00319891 (W 0.00346024) | - | 🔴 colmat 1.10× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.305845 ± 0.00222402 (W 0.00266548) | 0.293906 ± 0.00310748 (W 0.0018234) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.87495 ± 0.00437693 (W 0.00569351) | 2.99575 ± 0.00702814 (W 0.00636068) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.02961 ± 0.0035151 (W 0.00366949) | 2.86606 ± 0.00644342 (W 0.00759941) | - | 🔵 colmat 1.41× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 13.0438 ± 0.0404369 (W 0.0322464) | 22.8426 ± 0.17186 (W 0.0323303) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.9861 ± 0.0556984 (W 0.0115175) | 20.0022 ± 0.0320847 (W 0.0346345) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.26057 ± 0.00178473 (W 0.00293974) | 0.260119 ± 0.00160181 (W 0.0019747) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.257895 ± 0.00113143 (W 0.00304315) | 0.258887 ± 0.000420883 (W 0.00245856) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.89983 ± 0.00831658 (W 0.00364442) | 1.92543 ± 0.0187965 (W 0.00924305) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.90382 ± 0.00686155 (W 0.00223937) | 1.90667 ± 0.0131181 (W 0.00420477) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9797 ± 0.0549117 (W 0.0399591) | 15.9856 ± 0.0509356 (W 0.0186345) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.9949 ± 0.0717635 (W 0.0472713) | 15.9661 ± 0.0915089 (W 0.134557) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.442014 ± 0.00440057 (W 0.00275334) | 0.727258 ± 0.00294661 (W 0.00314623) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.496713 ± 0.000380741 (W 0.00426882) | 0.586313 ± 0.000738995 (W 0.00461839) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.87752 ± 0.0112982 (W 0.004859) | 4.24602 ± 0.0262148 (W 0.00939) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.20365 ± 0.0100201 (W 0.0048388) | 3.3216 ± 0.00794358 (W 0.00506978) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 18.6541 ± 0.0905234 (W 0.0314195) | 27.6539 ± 0.00702168 (W 0.034619) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 20.219 ± 0.174948 (W 0.0276301) | 21.0226 ± 0.0347345 (W 0.03536) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.243918 ± 0.0010251 (W 0.00215109) | 0.531981 ± 0.00293708 (W 0.00151152) | - | 🔵 colmat 2.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.49635 ± 0.00526136 (W 0.00477984) | 2.88373 ± 0.0155138 (W 0.00931712) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 8.52733 ± 0.0314672 (W 0.0438874) | 17.6032 ± 0.0592342 (W 0.0263163) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.90615 ± 0.00955377 (W 0.0031883) | 3.34937 ± 0.00665825 (W 0.00521934) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.9208 ± 0.014273 (W 0.00397382) | 3.36789 ± 0.0101334 (W 0.00470628) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.5741 ± 0.0624079 (W 0.0271387) | 17.6514 ± 0.0817488 (W 0.0174042) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 14.6988 ± 0.0457844 (W 0.00827368) | 17.6794 ± 0.0499668 (W 0.046896) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 87.3478 ± 0.335841 (W 0.167716) | 102.536 ± 0.896222 (W 0.147673) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 87.4298 ± 0.111409 (W 0.160565) | 103.373 ± 0.417476 (W 0.391565) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.964242 ± 0.0045341 (W 0.0022413) | 2.51103 ± 0.00575292 (W 0.00537924) | - | 🔵 colmat 2.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.96875 ± 0.00490983 (W 0.00239641) | 2.5208 ± 0.00611372 (W 0.0070368) | - | 🔵 colmat 2.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.54864 ± 0.0225675 (W 0.0102908) | 12.1243 ± 0.0699979 (W 0.0214282) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.57497 ± 0.0525481 (W 0.0126625) | 12.0836 ± 0.0560794 (W 0.0127752) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 37.3166 ± 0.223053 (W 0.0380824) | 65.3672 ± 0.326983 (W 0.142382) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 37.6152 ± 0.276791 (W 0.0279497) | 65.2498 ± 0.0548641 (W 0.159708) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004719 ± 5.7735e-06 (W 4.49444e-05) | 0.004458 ± 4.16333e-05 (W 5.45912e-05) | 0.004127 ± 4.41626e-05 (W 3.97492e-05) | 🔴 colmat 1.06× slower than `faer` | 🔴 colmat 1.14× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013395 ± 0.000124531 (W 8.42152e-05) | 0.012713 ± 0.000119709 (W 7.90462e-05) | 0.009568 ± 1.57162e-05 (W 4.43847e-05) | 🔴 colmat 1.05× slower than `faer` | 🔴 colmat 1.40× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.045065 ± 0.000480944 (W 0.000249608) | 0.043111 ± 0.00106473 (W 0.000351666) | 0.032039 ± 0.000759295 (W 0.00212308) | · | 🔴 colmat 1.41× slower than `lapack-aocl` |  |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.861733 ± 0.00665827 (W 0.00221968) | 2.15428 ± 0.00716006 (W 0.00423905) | - | 🔵 colmat 2.50× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.870129 ± 0.00416642 (W 0.00218676) | 2.15334 ± 0.0110662 (W 0.00213068) | - | 🔵 colmat 2.47× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.36828 ± 0.0169747 (W 0.0064653) | 6.62333 ± 0.025684 (W 0.00547953) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.34652 ± 0.0122354 (W 0.00562046) | 6.63733 ± 0.00452591 (W 0.0113333) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.5062 ± 0.0177878 (W 0.0172271) | 24.7007 ± 0.0192023 (W 0.0286998) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.3365 ± 0.0363502 (W 0.0354573) | 24.6304 ± 0.0279884 (W 0.0398036) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.25797 ± 0.00174894 (W 0.00214944) | 0.294736 ± 0.000968377 (W 0.00391572) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.274705 ± 0.00301358 (W 0.00265713) | 0.290408 ± 0.00133217 (W 0.00286271) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.98408 ± 0.00629305 (W 0.00503525) | 2.96573 ± 0.0173645 (W 0.00948695) | - | 🔵 colmat 3.01× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.27881 ± 0.00656382 (W 0.0041478) | 2.63002 ± 0.0144348 (W 0.00564428) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 4.32543 ± 0.0182641 (W 0.00942362) | 16.6437 ± 0.0304904 (W 0.033681) | - | 🔵 colmat 3.85× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 5.93654 ± 0.0295994 (W 0.0178949) | 14.0668 ± 0.042826 (W 0.0448828) | - | 🔵 colmat 2.37× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.083584 ± 0.000322008 (W 0.00178045) | 0.081158 ± 0.000529594 (W 0.0004148) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.083978 ± 0.00117325 (W 0.00165923) | 0.081033 ± 0.000616982 (W 0.0017366) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.522483 ± 0.00299812 (W 0.00300689) | 0.516343 ± 0.00254834 (W 0.00319133) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.52015 ± 0.00145627 (W 0.00197449) | 0.518707 ± 0.00261907 (W 0.00400041) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.18106 ± 0.0188674 (W 0.0132287) | 4.10931 ± 0.00555421 (W 0.00911438) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.18809 ± 0.0150795 (W 0.00952743) | 4.11434 ± 0.00196416 (W 0.00577159) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.325833 ± 0.00109417 (W 0.001829) | 0.843857 ± 0.00551668 (W 0.00538854) | - | 🔵 colmat 2.59× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.396886 ± 0.00381163 (W 0.00201347) | 0.644732 ± 0.00289202 (W 0.00546585) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.02027 ± 0.00302428 (W 0.00322733) | 3.45092 ± 0.019776 (W 0.0201777) | - | 🔵 colmat 3.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.5306 ± 0.00329578 (W 0.00759983) | 2.3196 ± 0.00322707 (W 0.00337018) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.91528 ± 0.0442191 (W 0.0138547) | 19.2453 ± 0.0960251 (W 0.0655357) | - | 🔵 colmat 3.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.20074 ± 0.0313162 (W 0.021844) | 9.76986 ± 0.379656 (W 0.0214342) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.208251 ± 0.00106881 (W 0.00165704) | 0.728518 ± 0.00795758 (W 0.00423916) | - | 🔵 colmat 3.50× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.641185 ± 0.000789464 (W 0.00337425) | 3.07057 ± 0.0229749 (W 0.0131197) | - | 🔵 colmat 4.79× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 3.14051 ± 0.0140526 (W 0.0128499) | 16.4046 ± 0.0756666 (W 0.0367509) | - | 🔵 colmat 5.22× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.99513 ± 0.00510468 (W 0.00499687) | 2.62575 ± 0.00329671 (W 0.0159268) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.00421 ± 0.00771042 (W 0.00341512) | 2.63042 ± 0.00665363 (W 0.0161581) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.10505 ± 0.0373928 (W 0.0250273) | 12.8021 ± 0.0433882 (W 0.0139976) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.14114 ± 0.0303022 (W 0.0283491) | 12.8456 ± 0.0542248 (W 0.0199129) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 36.5126 ± 0.113965 (W 0.0597) | 54.8131 ± 0.156064 (W 0.0787733) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 36.4553 ± 0.0599519 (W 0.0238708) | 54.9256 ± 0.274868 (W 0.216459) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.965545 ± 0.00245877 (W 0.00156926) | 2.08469 ± 0.00564458 (W 0.00432212) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.96843 ± 0.00765114 (W 0.00202446) | 2.08735 ± 0.0157748 (W 0.00850994) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.20737 ± 0.0465268 (W 0.00935806) | 10.3323 ± 0.0371851 (W 0.0158017) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.24983 ± 0.0180991 (W 0.0236953) | 10.3685 ± 0.0636296 (W 0.0142428) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.429 ± 0.0214632 (W 0.0298482) | 41.6397 ± 0.196478 (W 0.0402998) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.4731 ± 0.0250671 (W 0.0306835) | 41.6883 ± 0.17521 (W 0.0448876) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00574 ± 7.95005e-05 (W 3.81471e-05) | 0.004408 ± 1.52753e-05 (W 4.03733e-05) | 0.004128 ± 1.57162e-05 (W 3.58232e-05) | 🔴 colmat 1.30× slower than `faer` | 🔴 colmat 1.39× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013695 ± 0.000158076 (W 0.000116772) | 0.012794 ± 0.0001249 (W 7.12776e-05) | 0.009538 ± 0.000121903 (W 4.89418e-05) | 🔴 colmat 1.07× slower than `faer` | 🔴 colmat 1.44× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.033501 ± 0.000588817 (W 0.000214842) | 0.03257 ± 0.000447903 (W 0.00153892) | 0.032911 ± 0.000691807 (W 0.000226562) | · | · |  |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.667817 ± 0.00426954 (W 0.00398316) | 2.15043 ± 0.0035832 (W 0.00407442) | - | 🔵 colmat 3.22× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.674169 ± 0.00327529 (W 0.00315039) | 2.1556 ± 0.00792542 (W 0.00215473) | - | 🔵 colmat 3.20× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.45331 ± 0.0156805 (W 0.00647183) | 6.59177 ± 0.0129909 (W 0.0269437) | - | 🔵 colmat 2.69× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.46527 ± 0.0156921 (W 0.0129474) | 6.59545 ± 0.0340522 (W 0.0255093) | - | 🔵 colmat 2.68× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.3161 ± 0.0207448 (W 0.027818) | 23.2918 ± 0.0323225 (W 0.0523322) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.4039 ± 0.104525 (W 0.0199035) | 23.3242 ± 0.0307783 (W 0.0459629) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.263658 ± 0.000579609 (W 0.00189108) | 0.296203 ± 0.00374048 (W 0.00422278) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.283534 ± 0.00194687 (W 0.00142916) | 0.298672 ± 0.0021354 (W 0.00475923) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 0.917883 ± 0.0178974 (W 0.00510892) | 2.98276 ± 0.00900105 (W 0.00793821) | - | 🔵 colmat 3.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.24751 ± 0.00208906 (W 0.00662814) | 2.62673 ± 0.00850942 (W 0.00688786) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 3.24564 ± 0.030291 (W 0.00393526) | 15.7636 ± 0.0642932 (W 0.0254635) | - | 🔵 colmat 4.86× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 4.96559 ± 0.00947241 (W 0.0262562) | 13.414 ± 0.0358983 (W 0.0305645) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.054712 ± 0.000334865 (W 0.00181663) | 0.099633 ± 0.000102051 (W 0.00258096) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.055191 ± 0.000560539 (W 0.00164182) | 0.098089 ± 0.000752174 (W 0.00174509) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.285795 ± 0.00189866 (W 0.00276675) | 0.407857 ± 0.00596811 (W 0.0174688) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.286141 ± 0.00131153 (W 0.00301831) | 0.398049 ± 0.0135019 (W 0.0159026) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.26207 ± 0.00323511 (W 0.0103788) | 2.79873 ± 0.323794 (W 0.34866) | - | 🔵 colmat 1.24× faster than `faer` (marginal: spread 0.65 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.25563 ± 0.00829703 (W 0.00570939) | 2.84057 ± 0.0147289 (W 0.319651) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.311198 ± 0.00427981 (W 0.00483834) | 0.958293 ± 0.00655547 (W 0.00540338) | - | 🔵 colmat 3.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.385287 ± 0.00209479 (W 0.00341055) | 0.767731 ± 0.00454202 (W 0.00457409) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.827719 ± 0.0047757 (W 0.00209249) | 3.74894 ± 0.0186479 (W 0.0112825) | - | 🔵 colmat 4.53× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.35291 ± 0.00432567 (W 0.00462797) | 2.22476 ± 0.0129827 (W 0.121264) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.37617 ± 0.0102449 (W 0.00479678) | 18.7732 ± 0.176617 (W 0.0875351) | - | 🔵 colmat 5.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 5.9952 ± 0.0945611 (W 0.0432758) | 8.40647 ± 0.0637574 (W 0.29703) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.199406 ± 0.00136226 (W 0.00220551) | 0.833449 ± 0.00286325 (W 0.00467336) | - | 🔵 colmat 4.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.588981 ± 0.00252935 (W 0.00451594) | 3.34126 ± 0.0314166 (W 0.0209722) | - | 🔵 colmat 5.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.88051 ± 0.0118907 (W 0.00618952) | 16.6115 ± 0.082052 (W 0.0491063) | - | 🔵 colmat 8.83× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.98084 ± 0.0051206 (W 0.00309124) | 2.61569 ± 0.0130863 (W 0.00505899) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.98557 ± 0.001907 (W 0.00505773) | 2.63317 ± 0.0131818 (W 0.00397803) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.44762 ± 0.00428334 (W 0.0373453) | 12.7211 ± 0.0772685 (W 0.0164241) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.4705 ± 0.0265583 (W 0.0175071) | 12.8026 ± 0.0488786 (W 0.0136884) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.1157 ± 0.139752 (W 0.050056) | 51.5305 ± 0.0714173 (W 0.0677417) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.2245 ± 0.051656 (W 0.373949) | 51.5597 ± 0.132268 (W 0.118932) | - | 🔵 colmat 1.71× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.00599 ± 0.00405348 (W 0.00381959) | 2.0791 ± 0.0036781 (W 0.00607009) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.01018 ± 0.00868913 (W 0.00160975) | 2.07714 ± 0.00500724 (W 0.00361111) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.09414 ± 0.0269237 (W 0.0148798) | 10.3648 ± 0.00437386 (W 0.0183965) | - | 🔵 colmat 2.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.13092 ± 0.0159144 (W 0.0130063) | 10.3688 ± 0.0138766 (W 0.018062) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.3028 ± 0.00382775 (W 0.0556185) | 39.9244 ± 0.122769 (W 0.126877) | - | 🔵 colmat 2.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.2894 ± 0.0215734 (W 0.0206077) | 39.7716 ± 0.139807 (W 0.033963) | - | 🔵 colmat 2.44× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005711 ± 0.000150444 (W 4.06596e-05) | 0.004438 ± 3.51188e-05 (W 7.52994e-05) | 0.004178 ± 2.64575e-05 (W 3.49614e-05) | 🔴 colmat 1.29× slower than `faer` | 🔴 colmat 1.37× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013665 ± 0.000166533 (W 0.000159872) | 0.012874 ± 9.64365e-05 (W 8.37813e-05) | 0.009607 ± 1.76163e-05 (W 4.36039e-05) | 🔴 colmat 1.06× slower than `faer` | 🔴 colmat 1.42× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.03221 ± 0.000294286 (W 0.000119709) | 0.031738 ± 0.000451387 (W 0.000231623) | 0.032881 ± 0.0001095 (W 0.000393152) | · | · |  |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 2.00674 ± 0.00590871 (W 0.0062572) | 2.23186 ± 0.0106679 (W 0.00400686) | 1.81151 ± 0.00441685 (W 0.0073525) | 🔵 colmat 1.11× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 2.02025 ± 0.00659725 (W 0.00372691) | 2.23299 ± 0.0127166 (W 0.00459028) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 9.48947 ± 0.0133614 (W 0.00737373) | 9.61813 ± 0.0491906 (W 0.0109936) | 8.60876 ± 0.0418494 (W 0.0152161) | · | 🔴 colmat 1.10× slower than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 9.48889 ± 0.0185335 (W 0.0125168) | 9.62257 ± 0.0703326 (W 0.00997883) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 47.9021 ± 0.440866 (W 0.0702224) | 47.7445 ± 0.217889 (W 0.128128) | 47.6899 ± 0.451888 (W 0.1373) | · | · |  |
| `eigh/n512/b1 [api-inclusive]` | 48.3473 ± 0.0607988 (W 0.0638677) | 47.8298 ± 0.41813 (W 0.0596367) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.301096 ± 0.0012836 (W 0.00292724) | 0.286218 ± 0.001959 (W 0.00277227) | 6.45144 ± 0.0115925 (W 0.0110194) | 🔴 colmat 1.05× slower than `faer` | 🔵 colmat 21.43× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.282047 ± 0.00236659 (W 0.00197289) | 0.266638 ± 0.000677169 (W 0.00371832) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.93026 ± 0.00709406 (W 0.00869549) | 2.68844 ± 0.00655062 (W 0.00527214) | 53.9725 ± 0.134133 (W 0.0759236) | 🔵 colmat 1.39× faster than `faer` | 🔵 colmat 27.96× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.9398 ± 0.00827538 (W 0.00471238) | 2.70423 ± 0.00581318 (W 0.00774701) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 14.4978 ± 0.111814 (W 0.0139553) | 20.282 ± 0.0794714 (W 0.0257294) | 436.141 ± 0.873923 (W 0.58705) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 30.08× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.5455 ± 0.155785 (W 0.15954) | 18.9905 ± 0.0492025 (W 0.0449629) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.257936 ± 0.000901268 (W 0.00339597) | 0.254779 ± 0.00182903 (W 0.00269512) | 0.457119 ± 0.00121244 (W 0.0056543) | · | 🔵 colmat 1.77× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.256492 ± 0.00163355 (W 0.00239758) | 0.254148 ± 0.00151702 (W 0.00215589) | 0.488302 ± 0.00414465 (W 0.00759176) | · | 🔵 colmat 1.90× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.87557 ± 0.00920069 (W 0.00717311) | 1.88375 ± 0.0035635 (W 0.00266539) | 2.94274 ± 0.00828728 (W 0.00543269) | · | 🔵 colmat 1.57× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.87635 ± 0.0145536 (W 0.0039441) | 1.87976 ± 0.0123226 (W 0.00550465) | 2.72049 ± 0.00724822 (W 0.00965129) | · | 🔵 colmat 1.45× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9017 ± 0.0738492 (W 0.0411134) | 16.0572 ± 0.214862 (W 0.22228) | 21.5414 ± 0.140977 (W 0.0386543) | · | 🔵 colmat 1.35× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.7836 ± 0.0938353 (W 0.0277502) | 15.9034 ± 0.205704 (W 0.0327943) | 18.7563 ± 0.0872248 (W 0.0432512) | · | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `qr/n128/b1` | 0.387088 ± 0.00325967 (W 0.00258758) | 0.483379 ± 0.0008335 (W 0.0027604) | 1.12412 ± 0.00476419 (W 0.00620345) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 2.90× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.467119 ± 0.000982938 (W 0.00321518) | 0.565917 ± 0.00135552 (W 0.0031527) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.64587 ± 0.0197152 (W 0.00315596) | 2.7164 ± 0.0178557 (W 0.00840941) | 5.73222 ± 0.0188676 (W 0.0394344) | · | 🔵 colmat 2.17× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.08431 ± 0.00857106 (W 0.0041711) | 3.18149 ± 0.00618132 (W 0.00719108) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 17.5595 ± 0.0176533 (W 0.0986304) | 17.8499 ± 0.0576411 (W 0.00827154) | 35.4302 ± 0.1585 (W 0.0556562) | · | 🔵 colmat 2.02× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 19.7691 ± 0.148657 (W 0.0329772) | 20.0571 ± 0.103953 (W 0.0541622) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.185176 ± 0.0013798 (W 0.00234791) | 0.280704 ± 0.000608899 (W 0.00201441) | 0.638752 ± 0.00259629 (W 0.00318638) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 3.45× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.27403 ± 0.00715391 (W 0.00380744) | 1.37469 ± 0.00827708 (W 0.00460088) | 3.0239 ± 0.0291416 (W 0.00389145) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 2.37× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 7.65406 ± 0.0232974 (W 0.00847757) | 8.01853 ± 0.0413254 (W 0.00974494) | 16.4814 ± 0.0360413 (W 0.0323374) | · | 🔵 colmat 2.15× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 2.88347 ± 0.0118988 (W 0.00784874) | 3.3307 ± 0.0136393 (W 0.00657368) | 2.96443 ± 0.0110687 (W 0.0054179) | 🔵 colmat 1.16× faster than `faer` | · |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.8928 ± 0.00948916 (W 0.00343245) | 3.36582 ± 0.0413021 (W 0.00670133) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.4565 ± 0.0575573 (W 0.0195498) | 17.4949 ± 0.0588455 (W 0.0300986) | 15.8358 ± 0.0661752 (W 0.0180197) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.10× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 14.513 ± 0.0455695 (W 0.0165179) | 17.5625 ± 0.0598256 (W 0.0244195) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 86.1848 ± 0.40218 (W 0.0782768) | 102.028 ± 0.589881 (W 0.165687) | 104.813 ± 1.34645 (W 0.323176) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 86.2355 ± 0.0171393 (W 0.168614) | 102.722 ± 0.482542 (W 0.264454) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.943286 ± 0.00284975 (W 0.00304625) | 2.44103 ± 0.00296902 (W 0.0037653) | 1.16554 ± 0.00514493 (W 0.00336918) | 🔵 colmat 2.59× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.942755 ± 0.00212266 (W 0.0024551) | 2.46198 ± 0.0108914 (W 0.00670997) | - | 🔵 colmat 2.61× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.45419 ± 0.0216871 (W 0.00494534) | 11.8833 ± 0.0755582 (W 0.00931665) | 6.97674 ± 0.0417978 (W 0.00788833) | 🔵 colmat 2.18× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.45958 ± 0.00913644 (W 0.010495) | 11.8599 ± 0.0627616 (W 0.0227334) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 36.2167 ± 0.373205 (W 0.0396311) | 63.8974 ± 0.243473 (W 0.0797703) | 49.1875 ± 0.160142 (W 0.0887544) | 🔵 colmat 1.76× faster than `faer` | 🔵 colmat 1.36× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 36.4323 ± 0.423536 (W 0.0911866) | 64.0868 ± 0.0786578 (W 0.0430088) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005911 ± 7.54983e-05 (W 6.16214e-05) | 0.00572 ± 3.09892e-05 (W 3.03315e-05) | 0.002976 ± 2.91605e-05 (W 3.51596e-05) | · | 🔴 colmat 1.99× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.014616 ± 4.0501e-05 (W 0.000173785) | 0.014608 ± 0.000235445 (W 0.00173636) | 0.007754 ± 2.51661e-05 (W 4.39318e-05) | · | 🔴 colmat 1.88× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.046397 ± 8.06494e-05 (W 0.000204461) | 0.047896 ± 0.000425808 (W 0.00054955) | 0.029435 ± 0.000419726 (W 0.000364829) | · | 🔴 colmat 1.58× slower than `lapack-aocl` |  |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.862099 ± 0.00277432 (W 0.00318453) | 2.14287 ± 0.0138804 (W 0.0053484) | 1.50105 ± 0.00851168 (W 0.00530305) | 🔵 colmat 2.49× faster than `faer` | 🔵 colmat 1.74× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.860611 ± 0.00329685 (W 0.00209546) | 2.15034 ± 0.00598692 (W 0.00240772) | - | 🔵 colmat 2.50× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.32535 ± 0.013497 (W 0.0114782) | 6.57254 ± 0.022055 (W 0.00825149) | 6.06904 ± 0.0209187 (W 0.00867198) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 1.83× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.30963 ± 0.0102648 (W 0.00920011) | 6.6144 ± 0.0260435 (W 0.00814115) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.3772 ± 0.0264448 (W 0.0125593) | 24.3271 ± 0.121702 (W 0.0359432) | 27.8622 ± 0.136264 (W 0.032555) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 15.2413 ± 0.00765168 (W 0.0401405) | 24.3338 ± 0.0300663 (W 0.0569097) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.249124 ± 0.00119141 (W 0.00239425) | 0.288806 ± 0.00439513 (W 0.00379817) | 6.45331 ± 0.0692765 (W 0.0137188) | 🔵 colmat 1.16× faster than `faer` | 🔵 colmat 25.90× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.252344 ± 0.00436427 (W 0.00297397) | 0.270211 ± 0.00136166 (W 0.00201095) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.05951 ± 0.00110096 (W 0.00430426) | 2.67874 ± 0.00881616 (W 0.00586166) | 53.8028 ± 0.200414 (W 0.0416786) | 🔵 colmat 2.53× faster than `faer` | 🔵 colmat 50.78× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.18818 ± 0.00396084 (W 0.00422993) | 2.45131 ± 0.0161098 (W 0.00517427) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 5.41859 ± 0.0206915 (W 0.0103091) | 14.0796 ± 0.0492403 (W 0.0220109) | 433.163 ± 1.92652 (W 0.434549) | 🔵 colmat 2.60× faster than `faer` | 🔵 colmat 79.94× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 5.46027 ± 0.0115164 (W 0.0139693) | 12.9869 ± 0.0457262 (W 0.0169713) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.08471 ± 0.000150168 (W 0.000377313) | 0.081591 ± 0.000258211 (W 0.000628419) | 0.302382 ± 0.00516945 (W 0.0052203) | · | 🔵 colmat 3.57× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.083667 ± 0.000401305 (W 0.000276201) | 0.081009 ± 0.000101129 (W 0.000371276) | 0.216676 ± 0.00175174 (W 0.0045677) | · | 🔵 colmat 2.59× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.527012 ± 0.000876069 (W 0.00404255) | 0.521663 ± 0.00367715 (W 0.00237006) | 1.67834 ± 0.00880969 (W 0.0044974) | · | 🔵 colmat 3.18× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.526732 ± 0.00118992 (W 0.00257278) | 0.521149 ± 0.00134375 (W 0.00216862) | 1.0011 ± 0.0353303 (W 0.00411176) | · | 🔵 colmat 1.90× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.18597 ± 0.009849 (W 0.0147204) | 4.18344 ± 0.00664279 (W 0.0113901) | 10.9586 ± 0.0232007 (W 0.0261535) | · | 🔵 colmat 2.62× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.18458 ± 0.0148653 (W 0.0092686) | 4.18801 ± 0.0106207 (W 0.0104535) | 5.43856 ± 0.00592945 (W 0.0286515) | · | 🔵 colmat 1.30× faster than `lapack-aocl` |  |
| `qr/n128/b1` | 0.282491 ± 0.0011751 (W 0.00319941) | 0.53254 ± 0.00405005 (W 0.00346021) | 0.49989 ± 0.00387866 (W 0.00386255) | 🔵 colmat 1.89× faster than `faer` | 🔵 colmat 1.77× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.372811 ± 0.00138082 (W 0.00280063) | 0.624117 ± 0.00412936 (W 0.00409183) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.964393 ± 0.000775178 (W 0.00415784) | 1.70317 ± 0.0121199 (W 0.00667701) | 2.01876 ± 0.0144113 (W 0.0111502) | 🔵 colmat 1.77× faster than `faer` | 🔵 colmat 2.09× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.43566 ± 0.00406159 (W 0.00743145) | 2.18274 ± 0.00820491 (W 0.00700305) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.5224 ± 0.00473622 (W 0.00569166) | 7.04355 ± 0.0126086 (W 0.0136309) | 10.4048 ± 0.0314522 (W 0.0132221) | 🔵 colmat 1.28× faster than `faer` | 🔵 colmat 1.88× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 7.67013 ± 0.0722827 (W 0.0197514) | 8.63799 ± 0.400791 (W 0.0314343) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.185238 ± 0.000650703 (W 0.00237466) | 0.429753 ± 0.00378409 (W 0.00268756) | 0.278658 ± 0.00148469 (W 0.00273701) | 🔵 colmat 2.32× faster than `faer` | 🔵 colmat 1.50× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 0.573548 ± 0.00505663 (W 0.00292844) | 1.32635 ± 0.0109627 (W 0.00595096) | 1.02625 ± 0.00306207 (W 0.00546793) | 🔵 colmat 2.31× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 2.89306 ± 0.0106429 (W 0.00455418) | 4.43561 ± 0.0136499 (W 0.00988796) | 4.92118 ± 0.016962 (W 0.0133549) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 1.70× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 1.97625 ± 0.00999984 (W 0.00418332) | 2.60363 ± 0.0106457 (W 0.0138871) | 2.57699 ± 0.0229344 (W 0.00640931) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.30× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.98212 ± 0.0111716 (W 0.00381375) | 2.59819 ± 0.0204931 (W 0.0174366) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.0284 ± 0.0143885 (W 0.0151434) | 12.695 ± 0.0587456 (W 0.0174431) | 13.4605 ± 0.0532493 (W 0.0555503) | 🔵 colmat 1.58× faster than `faer` | 🔵 colmat 1.68× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 8.08908 ± 0.0209401 (W 0.0152048) | 12.7512 ± 0.0764066 (W 0.0411587) | - | 🔵 colmat 1.58× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 35.6025 ± 0.102337 (W 0.0494068) | 53.9541 ± 0.156702 (W 0.0482579) | 64.2082 ± 0.362507 (W 0.42839) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 1.80× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 35.6181 ± 0.0860859 (W 0.0463239) | 54.1017 ± 0.195988 (W 0.25135) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.94652 ± 0.00144511 (W 0.00202112) | 2.03993 ± 0.0135424 (W 0.00185973) | 1.18701 ± 0.00291415 (W 0.0033476) | 🔵 colmat 2.16× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.950321 ± 0.00170065 (W 0.00186818) | 2.03667 ± 0.0127098 (W 0.00217491) | - | 🔵 colmat 2.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.13504 ± 0.0135074 (W 0.00945714) | 10.1382 ± 0.0223432 (W 0.0139571) | 7.90161 ± 0.0176657 (W 0.0463121) | 🔵 colmat 2.45× faster than `faer` | 🔵 colmat 1.91× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 4.13808 ± 0.0129795 (W 0.0167763) | 10.1677 ± 0.0302621 (W 0.0133362) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.4309 ± 0.0264549 (W 0.0449925) | 40.5892 ± 0.140276 (W 0.0620917) | 35.3555 ± 0.163396 (W 0.47512) | 🔵 colmat 2.33× faster than `faer` | 🔵 colmat 2.03× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 17.3909 ± 0.0477979 (W 0.0109935) | 40.6862 ± 0.110145 (W 0.0757087) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.006351 ± 5.0797e-05 (W 4.54885e-05) | 0.00546 ± 3.21455e-05 (W 3.96232e-05) | 0.003005 ± 4.30155e-05 (W 5.4086e-05) | 🔴 colmat 1.16× slower than `faer` | 🔴 colmat 2.11× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015469 ± 0.000412593 (W 0.000216371) | 0.014416 ± 0.000244562 (W 6.6106e-05) | 0.007735 ± 0.00022519 (W 6.34925e-05) | 🔴 colmat 1.07× slower than `faer` (marginal: spread 0.62 of the difference) | 🔴 colmat 2.00× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.038262 ± 0.0010832 (W 0.0007857) | 0.039031 ± 0.000839209 (W 0.000791095) | 0.029245 ± 0.000305813 (W 0.000235022) | · | 🔴 colmat 1.31× slower than `lapack-aocl` | `!` between-start range over 5% |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.666925 ± 0.00238371 (W 0.00388748) | 2.14695 ± 0.0101927 (W 0.00410579) | 1.49806 ± 0.00826295 (W 0.0102363) | 🔵 colmat 3.22× faster than `faer` | 🔵 colmat 2.25× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.673943 ± 0.00311111 (W 0.00361694) | 2.14826 ± 0.00920683 (W 0.0040062) | - | 🔵 colmat 3.19× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.41719 ± 0.0399747 (W 0.0092551) | 6.55854 ± 0.0136415 (W 0.0231331) | 5.63809 ± 0.0481112 (W 0.0212329) | 🔵 colmat 2.71× faster than `faer` | 🔵 colmat 2.33× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 2.40596 ± 0.0185222 (W 0.00487184) | 6.55451 ± 0.0091861 (W 0.0286561) | - | 🔵 colmat 2.72× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.1898 ± 0.0170883 (W 0.0242969) | 22.8847 ± 0.0617129 (W 0.0604765) | 24.7367 ± 0.0657503 (W 0.0366836) | 🔵 colmat 2.25× faster than `faer` | 🔵 colmat 2.43× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 10.2053 ± 0.0983345 (W 0.0136485) | 22.9133 ± 0.0242236 (W 0.0499345) | - | 🔵 colmat 2.25× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.257185 ± 0.00052725 (W 0.00318267) | 0.288806 ± 0.00363706 (W 0.00327661) | 6.48652 ± 0.0093118 (W 0.00791537) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 25.22× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.262922 ± 0.0245456 (W 0.00153155) | 0.270272 ± 0.000792698 (W 0.00280986) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 0.951405 ± 0.00453276 (W 0.00380234) | 2.68013 ± 0.0115253 (W 0.00690812) | 53.9796 ± 0.0257431 (W 0.047032) | 🔵 colmat 2.82× faster than `faer` | 🔵 colmat 56.74× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.15078 ± 0.0244258 (W 0.00613174) | 2.46909 ± 0.00316131 (W 0.00469117) | - | 🔵 colmat 2.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 3.69164 ± 0.0409384 (W 0.00637613) | 13.1463 ± 0.0449456 (W 0.0177353) | 435.361 ± 0.527625 (W 0.446357) | 🔵 colmat 3.56× faster than `faer` | 🔵 colmat 117.93× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 4.4876 ± 0.150598 (W 0.0128834) | 12.3476 ± 0.0353797 (W 6.13108) | - | 🔵 colmat 2.75× faster than `faer` (marginal: spread 0.78 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.054129 ± 0.000220659 (W 0.000811352) | 0.099463 ± 0.00114153 (W 0.0020911) | 0.298763 ± 0.0025419 (W 0.00505673) | 🔵 colmat 1.84× faster than `faer` | 🔵 colmat 5.52× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.055042 ± 0.00494728 (W 0.000487168) | 0.099644 ± 0.000421284 (W 0.00209272) | 0.23471 ± 0.0170649 (W 0.00655322) | 🔵 colmat 1.81× faster than `faer` | 🔵 colmat 4.26× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.290293 ± 0.00159181 (W 0.00185861) | 0.408106 ± 0.00419131 (W 0.00872307) | 1.43592 ± 0.00237906 (W 0.0109697) | 🔵 colmat 1.41× faster than `faer` | 🔵 colmat 4.95× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.294902 ± 0.00354868 (W 0.0024318) | 0.393931 ± 0.0127298 (W 0.0152559) | 0.660864 ± 0.00440701 (W 0.00486242) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 2.24× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.26843 ± 0.0107581 (W 0.00918017) | 2.97398 ± 0.40649 (W 0.309226) | 9.33031 ± 0.0363607 (W 0.0214345) | 🔵 colmat 1.31× faster than `faer` | 🔵 colmat 4.11× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.27961 ± 0.00142832 (W 0.00635985) | 3.007 ± 0.408953 (W 0.365848) | 3.36665 ± 0.0170308 (W 0.00927232) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.48× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1` | 0.265595 ± 0.00124426 (W 0.00241971) | 0.656188 ± 0.0076829 (W 0.00796928) | 0.494668 ± 0.0141847 (W 0.0122032) | 🔵 colmat 2.47× faster than `faer` | 🔵 colmat 1.86× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/n128/b1 [api-inclusive]` | 0.363283 ± 0.00198298 (W 0.00275314) | 0.747873 ± 0.00556885 (W 0.00650334) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.793189 ± 0.00321238 (W 0.00196356) | 2.02474 ± 0.00183408 (W 0.0174717) | 1.45904 ± 0.0119745 (W 0.0163331) | 🔵 colmat 2.55× faster than `faer` | 🔵 colmat 1.84× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.26862 ± 0.00396342 (W 0.00695816) | 2.11494 ± 0.0138448 (W 0.0131349) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.17396 ± 0.00737283 (W 0.00852427) | 7.07072 ± 0.0405499 (W 0.0488428) | 6.52119 ± 0.00916637 (W 0.0304667) | 🔵 colmat 2.23× faster than `faer` | 🔵 colmat 2.05× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 5.46997 ± 0.0535126 (W 0.029575) | 7.43924 ± 0.0289059 (W 0.0488852) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.18614 ± 8.06494e-05 (W 0.00216595) | 0.534435 ± 0.00145992 (W 0.0040417) | 0.256147 ± 0.00333804 (W 0.00452558) | 🔵 colmat 2.87× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 0.559701 ± 0.000944441 (W 0.0033022) | 1.6066 ± 0.0187059 (W 0.0129536) | 0.780284 ± 0.000637177 (W 0.0157532) | 🔵 colmat 2.87× faster than `faer` | 🔵 colmat 1.39× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 1.73971 ± 0.00673851 (W 0.00457058) | 4.75804 ± 0.00531874 (W 0.0343235) | 3.12569 ± 0.0115184 (W 0.01829) | 🔵 colmat 2.73× faster than `faer` | 🔵 colmat 1.80× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 1.96096 ± 0.00799147 (W 0.00275684) | 2.59359 ± 0.0108618 (W 0.00455935) | 2.5178 ± 0.0116767 (W 0.0122596) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.28× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.95364 ± 0.00979618 (W 0.0021391) | 2.6106 ± 0.00702635 (W 0.00560546) | - | 🔵 colmat 1.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.27076 ± 0.0206877 (W 0.0103604) | 12.6185 ± 0.0328791 (W 0.0189839) | 15.6827 ± 0.152554 (W 0.0868146) | 🔵 colmat 1.74× faster than `faer` | 🔵 colmat 2.16× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.27712 ± 0.0278298 (W 0.00703522) | 12.6776 ± 0.0217701 (W 0.0259859) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 29.2124 ± 0.135653 (W 0.0422727) | 50.6683 ± 0.277125 (W 0.0914526) | 62.8171 ± 0.346091 (W 0.128187) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 2.15× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 29.3568 ± 0.0894489 (W 0.0602191) | 50.6768 ± 0.366682 (W 0.410929) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.981614 ± 0.00278449 (W 0.00172112) | 2.02429 ± 0.00600222 (W 0.0061648) | 1.21382 ± 0.00314653 (W 0.00393623) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.990039 ± 0.00358165 (W 0.00121185) | 2.02935 ± 0.00905291 (W 0.00308476) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.98151 ± 0.0124015 (W 0.00404792) | 10.202 ± 0.060599 (W 0.0183942) | 10.5627 ± 0.0404256 (W 0.0646107) | 🔵 colmat 2.56× faster than `faer` | 🔵 colmat 2.65× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 3.99842 ± 0.0217728 (W 0.0115157) | 10.2089 ± 0.0240376 (W 0.0111084) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.2929 ± 0.0776598 (W 0.136017) | 38.853 ± 0.0665814 (W 0.0920503) | 37.2263 ± 0.123317 (W 0.189585) | 🔵 colmat 2.54× faster than `faer` | 🔵 colmat 2.43× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 15.2827 ± 0.0370725 (W 0.0184134) | 38.9166 ± 0.0819476 (W 0.0775374) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.006252 ± 6.55744e-05 (W 4.94297e-05) | 0.00535 ± 1.0504e-05 (W 3.70135e-05) | 0.003046 ± 1.73205e-05 (W 7.57971e-05) | 🔴 colmat 1.17× slower than `faer` | 🔴 colmat 2.05× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015759 ± 0.000236864 (W 0.000127481) | 0.014968 ± 0.000448487 (W 7.22032e-05) | 0.007894 ± 0.000110077 (W 4.30256e-05) | 🔴 colmat 1.05× slower than `faer` (marginal: spread 0.87 of the difference) | 🔴 colmat 2.00× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.039393 ± 0.000313842 (W 0.000539973) | 0.039513 ± 0.000391571 (W 0.00101389) | 0.034002 ± 0.000901661 (W 0.000318309) | · | 🔴 colmat 1.16× slower than `lapack-aocl` | `!` between-start range over 5% |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 3.00625 ± 0.0140957 (W 0.00320677) | 3.20746 ± 0.0123163 (W 0.00370751) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.98914 ± 0.00924753 (W 0.0037582) | 3.21664 ± 0.0134343 (W 0.00324939) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 16.5588 ± 0.0697076 (W 0.0185559) | 16.8043 ± 0.0489782 (W 0.0129) | - | · | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 16.5528 ± 0.0564436 (W 0.0119803) | 16.8667 ± 0.0770556 (W 0.0195054) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 100.858 ± 0.397029 (W 0.0806182) | 100.209 ± 0.529078 (W 0.0823213) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 101.454 ± 0.163475 (W 0.146834) | 100.632 ± 0.277882 (W 0.381101) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.65395 ± 0.00162158 (W 0.00300986) | 0.805996 ± 0.00179802 (W 0.00619057) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.740743 ± 0.00375346 (W 0.00353389) | 0.886641 ± 0.00901443 (W 0.00190043) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 4.32041 ± 0.011542 (W 0.0145871) | 6.02935 ± 0.0349803 (W 0.00808798) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 4.82849 ± 0.016418 (W 0.00504036) | 6.2063 ± 0.0465336 (W 0.020711) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 32.2602 ± 0.301788 (W 0.115889) | 40.069 ± 0.324399 (W 0.0622201) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 35.0918 ± 0.110734 (W 0.0469441) | 40.5695 ± 0.24524 (W 0.114509) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.04855 ± 0.000888915 (W 0.00179021) | 1.03891 ± 0.00219757 (W 0.00166595) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.04879 ± 0.0025511 (W 0.00284054) | 1.04214 ± 0.00305056 (W 0.00298025) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.52389 ± 0.0311969 (W 0.0180968) | 7.49559 ± 0.0395778 (W 0.0080447) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.50535 ± 0.0358154 (W 0.00910524) | 7.48671 ± 0.0393666 (W 0.00835616) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.795 ± 0.130986 (W 0.10025) | 56.8063 ± 0.209881 (W 0.0739062) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.8324 ± 0.187891 (W 0.0712902) | 56.708 ± 0.150447 (W 0.0600855) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.47229 ± 0.0145926 (W 0.00448971) | 1.85147 ± 0.00959239 (W 0.00356263) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 1.4907 ± 0.00907837 (W 0.0040143) | 1.73178 ± 0.0105633 (W 0.00345854) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 10.2064 ± 0.0291536 (W 0.0260259) | 12.0363 ± 0.0408486 (W 0.00867439) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 10.926 ± 0.0485696 (W 0.00589626) | 11.1201 ± 0.0421848 (W 0.0113753) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 70.1688 ± 0.330661 (W 0.0992631) | 77.4634 ± 0.295765 (W 0.0847884) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 73.7141 ± 0.344402 (W 0.0400938) | 74.5047 ± 0.252561 (W 0.236874) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.683656 ± 0.00219226 (W 0.00284637) | 1.06878 ± 0.000678028 (W 0.00272138) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 4.65549 ± 0.0211315 (W 0.00605866) | 6.49836 ± 0.0221249 (W 0.00789078) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 30.2435 ± 0.0931891 (W 0.0618816) | 37.8435 ± 0.183974 (W 0.15166) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.90468 ± 0.0262061 (W 0.00535997) | 5.44734 ± 0.0303518 (W 0.0162763) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 4.91303 ± 0.0131694 (W 0.00676596) | 5.47864 ± 0.103307 (W 0.00876006) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.3455 ± 0.153944 (W 0.0151566) | 33.4594 ± 0.153185 (W 0.0339059) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 29.3513 ± 0.120502 (W 0.0246405) | 33.519 ± 0.173349 (W 0.0296126) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 189.282 ± 1.34102 (W 0.344899) | 206.1 ± 0.733342 (W 0.464707) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 190.348 ± 0.547464 (W 0.474086) | 206.918 ± 1.02222 (W 0.230818) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.7104 ± 0.00902672 (W 0.00265182) | 3.34881 ± 0.0231864 (W 0.00610438) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.71762 ± 0.00641466 (W 0.0024208) | 3.3396 ± 0.0231777 (W 0.00603059) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.1764 ± 0.140199 (W 0.0188336) | 19.1722 ± 0.0709088 (W 0.013351) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 11.2886 ± 0.0423584 (W 0.0237266) | 19.1672 ± 0.0450452 (W 0.020173) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 78.8525 ± 0.290289 (W 0.321402) | 108.562 ± 0.144239 (W 0.137391) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 79.2897 ± 0.290737 (W 0.457127) | 109.267 ± 0.37826 (W 0.188522) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.008616 ± 2e-05 (W 5.21172e-05) | 0.008035 ± 5.56776e-05 (W 5.91878e-05) | 0.007143 ± 4.3589e-05 (W 4.52681e-05) | 🔴 colmat 1.07× slower than `faer` | 🔴 colmat 1.21× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024897 ± 0.000137264 (W 0.000207609) | 0.023673 ± 0.00025401 (W 0.000162703) | 0.020437 ± 7.55006e-05 (W 6.61687e-05) | 🔴 colmat 1.05× slower than `faer` | 🔴 colmat 1.22× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.092504 ± 0.0010243 (W 0.00164683) | 0.092133 ± 0.00096432 (W 0.00303257) | 0.072516 ± 0.000258988 (W 0.0016751) | · | 🔴 colmat 1.28× slower than `lapack-aocl` |  |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 1.21937 ± 0.0030476 (W 0.00162781) | 2.77293 ± 0.00818997 (W 0.00565702) | - | 🔵 colmat 2.27× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.22879 ± 0.00366879 (W 0.00609965) | 2.77539 ± 0.00644432 (W 0.012618) | - | 🔵 colmat 2.26× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.45641 ± 0.00148399 (W 0.00665647) | 10.8901 ± 0.0343865 (W 0.0126023) | - | 🔵 colmat 2.00× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.51406 ± 0.0285873 (W 0.00927858) | 10.9081 ± 0.0302563 (W 0.0168712) | - | 🔵 colmat 1.98× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.9314 ± 0.0794662 (W 0.0306773) | 43.2775 ± 0.0897165 (W 0.0240884) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 30.099 ± 0.140545 (W 0.020109) | 43.3115 ± 0.0852879 (W 0.0403015) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.383271 ± 0.000841661 (W 0.00375142) | 0.808106 ± 0.00150546 (W 0.003567) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.514099 ± 0.00344935 (W 0.00396169) | 0.786748 ± 0.00823218 (W 0.00333071) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.66019 ± 0.010239 (W 0.00381967) | 6.00788 ± 0.0385374 (W 0.0182491) | - | 🔵 colmat 3.62× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.38819 ± 0.00566733 (W 0.00770025) | 5.68836 ± 0.0278893 (W 0.0134288) | - | 🔵 colmat 2.38× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 9.5002 ± 0.0265586 (W 0.0266498) | 25.6969 ± 0.0710272 (W 0.0275552) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.4563 ± 0.058277 (W 0.0420652) | 23.9963 ± 0.0721448 (W 0.0452963) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.283794 ± 0.000406181 (W 0.00184377) | 0.278811 ± 0.00135161 (W 0.00228401) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.284862 ± 0.000641599 (W 0.00192772) | 0.280695 ± 0.00182986 (W 0.00197858) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.93824 ± 0.00790679 (W 0.00412699) | 1.93141 ± 0.00826973 (W 0.00453224) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.94048 ± 0.00861114 (W 0.00637678) | 1.935 ± 0.00292024 (W 0.0024635) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6322 ± 0.00137392 (W 0.0153783) | 14.6364 ± 0.0329742 (W 0.0201978) | - | · | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6514 ± 0.018124 (W 0.0159748) | 14.6416 ± 0.0346557 (W 0.0250765) | - | · | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.722218 ± 0.00204203 (W 0.00398796) | 1.3487 ± 0.0044571 (W 0.00378761) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.925773 ± 0.00484161 (W 0.00488352) | 1.19332 ± 0.00588158 (W 0.00595409) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.08937 ± 0.0139349 (W 0.00422429) | 6.28825 ± 0.0372402 (W 0.0261065) | - | 🔵 colmat 2.04× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 4.15912 ± 0.0053371 (W 0.008671) | 5.22883 ± 0.0249229 (W 0.0152921) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 20.0432 ± 0.160166 (W 0.126915) | 31.8196 ± 0.0275041 (W 0.0872136) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 25.1022 ± 0.0578388 (W 0.0178602) | 27.2103 ± 0.0279224 (W 0.0270415) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.44419 ± 0.00234597 (W 0.00355075) | 1.01265 ± 0.00907999 (W 0.00290743) | - | 🔵 colmat 2.28× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.62118 ± 0.00566478 (W 0.0046529) | 4.84742 ± 0.0137616 (W 0.0246858) | - | 🔵 colmat 2.99× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 9.23856 ± 0.0222398 (W 0.0151883) | 21.5765 ± 0.0960073 (W 0.0549422) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.71733 ± 0.0192661 (W 0.00737741) | 3.94503 ± 0.00555325 (W 0.0144232) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.73773 ± 0.00928458 (W 0.00389395) | 3.96496 ± 0.0208279 (W 0.0186484) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.5503 ± 0.0300943 (W 0.0400003) | 21.3733 ± 0.0961584 (W 0.0369009) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 12.5497 ± 0.0383152 (W 0.0151141) | 21.7627 ± 0.16779 (W 0.047158) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 66.127 ± 0.0929974 (W 0.0544611) | 86.0742 ± 0.048861 (W 0.0529912) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 66.4465 ± 0.135391 (W 0.387222) | 86.5037 ± 0.0497655 (W 0.879261) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.27389 ± 0.000757916 (W 0.00329764) | 2.96157 ± 0.0145478 (W 0.00393939) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.2809 ± 0.00574802 (W 0.00557563) | 2.98019 ± 0.00871779 (W 0.00298274) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.06276 ± 0.0269275 (W 0.0121013) | 16.3555 ± 0.068302 (W 0.0166256) | - | 🔵 colmat 2.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.04922 ± 0.0137088 (W 0.011785) | 16.4567 ± 0.103693 (W 0.0205237) | - | 🔵 colmat 2.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 31.3207 ± 0.18515 (W 0.0413319) | 57.0402 ± 0.0813176 (W 0.0732932) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 31.4453 ± 0.127386 (W 0.0298316) | 57.0286 ± 0.158295 (W 0.0690599) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009758 ± 4.2194e-05 (W 6.62291e-05) | 0.008095 ± 3.84318e-05 (W 0.000115249) | 0.007123 ± 1.47309e-05 (W 5.02812e-05) | 🔴 colmat 1.21× slower than `faer` | 🔴 colmat 1.37× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.022701 ± 0.000153971 (W 0.000336689) | 0.023895 ± 0.000154869 (W 0.000127519) | 0.020458 ± 0.000150533 (W 8.36977e-05) | 🔵 colmat 1.05× faster than `faer` | 🔴 colmat 1.11× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.058077 ± 9.96209e-05 (W 0.000552392) | 0.06514 ± 0.00123991 (W 0.0005278) | 0.073795 ± 0.00112659 (W 0.0012654) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.27× faster than `lapack-aocl` |  |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.945414 ± 0.00566163 (W 0.00252664) | 2.78538 ± 0.00286926 (W 0.00320729) | - | 🔵 colmat 2.95× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.95192 ± 0.00415023 (W 0.00309977) | 2.78524 ± 0.00800016 (W 0.00558439) | - | 🔵 colmat 2.93× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.73052 ± 0.0139048 (W 0.0074887) | 10.6788 ± 0.0425964 (W 0.0412283) | - | 🔵 colmat 2.86× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.76366 ± 0.0124456 (W 0.007304) | 10.6877 ± 0.0368549 (W 0.208547) | - | 🔵 colmat 2.84× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 18.6123 ± 0.0530535 (W 0.0428718) | 37.72 ± 0.0920369 (W 0.187051) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 18.7165 ± 0.0729627 (W 0.0217755) | 37.8496 ± 0.176328 (W 0.350349) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.375801 ± 0.0044714 (W 0.00191479) | 0.801148 ± 0.00539108 (W 0.00475693) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.51173 ± 0.00274129 (W 0.00241218) | 0.785348 ± 0.00483874 (W 0.00457793) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.32032 ± 0.0102159 (W 0.00259868) | 5.96493 ± 0.0313371 (W 0.0238854) | - | 🔵 colmat 4.52× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.09647 ± 0.00880461 (W 0.00990954) | 5.61053 ± 0.0595427 (W 0.0122788) | - | 🔵 colmat 2.68× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 5.80452 ± 0.0148122 (W 0.0216982) | 23.2339 ± 0.150949 (W 0.0476193) | - | 🔵 colmat 4.00× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 10.0641 ± 0.0186297 (W 0.0228358) | 21.8677 ± 0.124835 (W 0.0186202) | - | 🔵 colmat 2.17× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.168149 ± 0.000703385 (W 0.00207247) | 0.275618 ± 0.0031065 (W 0.0066242) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.16824 ± 0.000668865 (W 0.00165249) | 0.273343 ± 0.00424109 (W 0.00601396) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00796 ± 0.00851574 (W 0.00132154) | 1.14185 ± 0.0918955 (W 0.0997028) | - | 🔵 colmat 1.13× faster than `faer` (marginal: spread 0.75 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01358 ± 0.00233361 (W 0.00493235) | 1.13796 ± 0.0110115 (W 0.1022) | - | 🔵 colmat 1.12× faster than `faer` (marginal: spread 0.82 of the difference) | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.93344 ± 0.0537258 (W 0.119383) | 10.0299 ± 0.11654 (W 1.29969) | - | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.90776 ± 0.0052813 (W 0.0776966) | 9.93896 ± 1.33598 (W 1.33691) | - | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.66 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n128/b1` | 0.561833 ± 0.000607983 (W 0.00219181) | 1.42948 ± 0.00744889 (W 0.00851781) | - | 🔵 colmat 2.54× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.794774 ± 0.0017458 (W 0.0041567) | 1.29757 ± 0.00292179 (W 0.00536178) | - | 🔵 colmat 1.63× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.06708 ± 0.00245028 (W 0.00274766) | 6.19749 ± 0.0361315 (W 0.0346121) | - | 🔵 colmat 3.00× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.20349 ± 0.00730849 (W 0.00599549) | 4.34739 ± 0.0275039 (W 0.0288111) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.7039 ± 0.39566 (W 0.0161692) | 27.5955 ± 0.40347 (W 0.176162) | - | 🔵 colmat 2.58× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 16.2245 ± 0.0442476 (W 0.0418581) | 19.0333 ± 0.139905 (W 0.256573) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.381481 ± 0.00324441 (W 0.00317956) | 1.10067 ± 0.00151966 (W 0.00782515) | - | 🔵 colmat 2.89× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.29399 ± 0.00777728 (W 0.00488601) | 4.9929 ± 0.0121401 (W 0.037254) | - | 🔵 colmat 3.86× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.07599 ± 0.0270472 (W 0.00773318) | 20.1702 ± 0.0107902 (W 0.149121) | - | 🔵 colmat 3.97× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.50659 ± 0.0107437 (W 0.00537793) | 3.94333 ± 0.0194177 (W 0.0064813) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.51613 ± 0.0126097 (W 0.00550896) | 3.95767 ± 0.0212793 (W 0.0103318) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.3954 ± 0.0197157 (W 0.0156375) | 21.1014 ± 0.0266846 (W 0.0374082) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 10.4271 ± 0.031111 (W 0.0147175) | 21.7306 ± 0.178617 (W 0.242401) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 47.4818 ± 0.165915 (W 0.0696708) | 73.5502 ± 0.290896 (W 0.442141) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 47.7017 ± 0.379171 (W 0.0618743) | 74.2392 ± 0.180213 (W 0.407645) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.26394 ± 0.00433208 (W 0.00437896) | 2.95338 ± 0.00746991 (W 0.00731107) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.26652 ± 0.00449063 (W 0.00249303) | 2.97155 ± 0.00491986 (W 0.00891911) | - | 🔵 colmat 2.35× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.5784 ± 0.0323428 (W 0.00793604) | 16.3456 ± 0.152664 (W 0.0485904) | - | 🔵 colmat 2.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.56913 ± 0.00892401 (W 0.0153306) | 16.4139 ± 0.0927084 (W 0.0166181) | - | 🔵 colmat 2.95× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 24.5585 ± 0.0621245 (W 0.043921) | 51.0812 ± 0.0658492 (W 0.049437) | - | 🔵 colmat 2.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 24.5961 ± 0.108159 (W 0.0265102) | 51.0004 ± 0.12689 (W 0.379034) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009698 ± 0.000107364 (W 6.30079e-05) | 0.008125 ± 6e-05 (W 4.77493e-05) | 0.007053 ± 4.93288e-05 (W 5.06952e-05) | 🔴 colmat 1.19× slower than `faer` | 🔴 colmat 1.38× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.022222 ± 0.000100936 (W 0.000121894) | 0.023935 ± 9.20887e-05 (W 7.34847e-05) | 0.020658 ± 9.84886e-05 (W 9.90162e-05) | 🔵 colmat 1.08× faster than `faer` | 🔴 colmat 1.08× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.052356 ± 0.000513794 (W 0.00022754) | 0.061164 ± 0.00156922 (W 0.00159067) | 0.073607 ± 0.000717843 (W 0.000370857) | 🔵 colmat 1.17× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 2.99425 ± 0.00697859 (W 0.00217134) | 3.20212 ± 0.00933782 (W 0.0035929) | 3.56316 ± 0.0233691 (W 0.00664461) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 2.97914 ± 0.0144391 (W 0.00489395) | 3.20825 ± 0.00837078 (W 0.0043299) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 16.5009 ± 0.0580195 (W 0.0349212) | 16.7134 ± 0.0614492 (W 0.0146662) | 20.5627 ± 0.192142 (W 0.0156601) | · | 🔵 colmat 1.25× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 16.4542 ± 0.0899512 (W 0.0200696) | 16.7658 ± 0.0699965 (W 0.0130125) | - | · | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 100.564 ± 0.261964 (W 0.438937) | 99.9475 ± 0.287013 (W 0.0766904) | 141.407 ± 1.73696 (W 0.739771) | · | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 100.704 ± 0.378863 (W 0.0980497) | 99.9824 ± 0.507768 (W 0.148742) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.611525 ± 0.000535547 (W 0.0019123) | 0.759649 ± 0.00572135 (W 0.00502411) | 22.0853 ± 0.0671 (W 0.0348718) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 36.12× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.71171 ± 0.00247902 (W 0.00473154) | 0.858264 ± 0.00283802 (W 0.0035578) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 4.1952 ± 0.0212338 (W 0.00742472) | 5.47209 ± 0.0259673 (W 0.00945746) | 173.26 ± 0.113179 (W 0.140045) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 41.30× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 4.64483 ± 0.0254116 (W 0.0126119) | 5.97304 ± 0.0252979 (W 0.014074) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 32.6937 ± 1.00403 (W 0.106153) | 37.6803 ± 0.284634 (W 0.0246311) | 1364.48 ± 10.79 (W 0.641189) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 41.74× faster than `lapack-aocl` | `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 34.0238 ± 0.157018 (W 0.168925) | 39.2615 ± 0.379451 (W 0.102786) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.03496 ± 0.00391239 (W 0.00367243) | 1.03122 ± 0.00344022 (W 0.00172484) | 1.30765 ± 0.00727482 (W 0.00342268) | · | 🔵 colmat 1.26× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.03822 ± 0.000545462 (W 0.00162989) | 1.03311 ± 0.00467575 (W 0.00271951) | 1.29863 ± 0.0100014 (W 0.00477011) | · | 🔵 colmat 1.25× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.44466 ± 0.0189993 (W 0.00983632) | 7.42286 ± 0.0260593 (W 0.00481539) | 9.18743 ± 0.03192 (W 0.0107404) | · | 🔵 colmat 1.23× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.43864 ± 0.0245217 (W 0.00911812) | 7.44092 ± 0.0248461 (W 0.011838) | 8.34385 ± 0.0236527 (W 0.0173776) | · | 🔵 colmat 1.12× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.5863 ± 0.108664 (W 0.214793) | 56.4879 ± 0.166546 (W 0.40883) | 62.4269 ± 0.186589 (W 0.0627948) | · | 🔵 colmat 1.10× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.7413 ± 0.250764 (W 0.0621614) | 56.5128 ± 0.0642852 (W 0.211658) | 52.3657 ± 0.476554 (W 0.157091) | · | 🔴 colmat 1.08× slower than `lapack-aocl` |  |
| `qr/n128/b1` | 1.31772 ± 0.0189677 (W 0.00412173) | 1.52185 ± 0.00295983 (W 0.00447845) | 2.05866 ± 0.00487153 (W 0.00592652) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 1.44665 ± 0.00583382 (W 0.00411879) | 1.69897 ± 0.00523568 (W 0.00287979) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 9.88611 ± 0.0220859 (W 0.0127231) | 9.97988 ± 0.0317342 (W 0.0124678) | 13.4409 ± 0.0739395 (W 0.0208569) | · | 🔵 colmat 1.36× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 10.8241 ± 0.0438031 (W 0.0188016) | 10.943 ± 0.0200434 (W 0.0168029) | - | · | - | `*` one row has no measurement here |
| `qr/n512/b1` | 68.0736 ± 0.336769 (W 0.123566) | 68.8942 ± 0.190087 (W 0.343692) | 86.5531 ± 0.332218 (W 0.380075) | · | 🔵 colmat 1.27× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 72.9992 ± 0.309382 (W 0.193657) | 73.1734 ± 0.387554 (W 0.446027) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.485759 ± 0.00102522 (W 0.00274122) | 0.750422 ± 0.00103976 (W 0.00279155) | 0.773254 ± 0.0028818 (W 0.00212402) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.59× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 4.3698 ± 0.0196922 (W 0.00510543) | 4.51055 ± 0.0192673 (W 0.00850475) | 5.21818 ± 0.0385046 (W 0.00630586) | · | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 28.2759 ± 0.112248 (W 0.133645) | 28.8735 ± 0.144927 (W 0.0260966) | 34.2808 ± 0.171757 (W 0.0436259) | · | 🔵 colmat 1.21× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 4.85877 ± 0.0154905 (W 0.00467076) | 5.42471 ± 0.0321742 (W 0.0118538) | 5.46197 ± 0.0121408 (W 0.0116011) | 🔵 colmat 1.12× faster than `faer` | 🔵 colmat 1.12× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.86319 ± 0.0196667 (W 0.0043501) | 5.44182 ± 0.0180232 (W 0.00674168) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.0734 ± 0.168865 (W 0.0220693) | 33.1454 ± 0.116227 (W 0.0394027) | 35.5672 ± 0.350504 (W 0.0844556) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 1.22× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 29.1297 ± 0.0830513 (W 0.031713) | 33.1852 ± 0.137152 (W 0.0723359) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 188.876 ± 0.819535 (W 0.405394) | 205.06 ± 0.509071 (W 0.223267) | 216.727 ± 0.783232 (W 0.657597) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.15× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 189.466 ± 0.678651 (W 0.25991) | 205.768 ± 0.629711 (W 0.429134) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.68109 ± 0.00498164 (W 0.00375744) | 3.32306 ± 0.0109794 (W 0.00548221) | 2.07806 ± 0.00690394 (W 0.00461958) | 🔵 colmat 1.98× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.68949 ± 0.00875621 (W 0.00719598) | 3.31536 ± 0.0148167 (W 0.00771416) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 10.956 ± 0.0715795 (W 0.011206) | 18.8431 ± 0.104403 (W 0.0243365) | 15.9413 ± 0.129645 (W 0.017134) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 1.46× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 10.9904 ± 0.108111 (W 0.015627) | 18.8601 ± 0.0679379 (W 0.0271843) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 78.2167 ± 0.118628 (W 0.421094) | 108.032 ± 0.321076 (W 0.426178) | 97.5764 ± 1.34343 (W 0.27254) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 78.1699 ± 0.235021 (W 0.105284) | 108.481 ± 0.294787 (W 0.194065) | - | 🔵 colmat 1.39× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.008626 ± 1.73205e-05 (W 3.17679e-05) | 0.008757 ± 6.0803e-05 (W 6.03349e-05) | 0.006572 ± 6.5041e-05 (W 0.00150465) | · | 🔴 colmat 1.31× slower than `lapack-aocl` (marginal: spread 0.73 of the difference) | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.025486 ± 0.000130184 (W 0.000138016) | 0.025857 ± 0.000134441 (W 0.00011433) | 0.019055 ± 0.000330092 (W 0.000123976) | · | 🔴 colmat 1.34× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.086643 ± 0.000490598 (W 0.000457174) | 0.086476 ± 0.000181236 (W 0.000432218) | 0.072597 ± 0.000178261 (W 0.00228616) | · | 🔴 colmat 1.19× slower than `lapack-aocl` |  |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 1.20391 ± 0.00167947 (W 0.00294924) | 2.75504 ± 0.0108488 (W 0.00561889) | 3.83408 ± 0.0100195 (W 0.00419187) | 🔵 colmat 2.29× faster than `faer` | 🔵 colmat 3.18× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.21168 ± 0.00196326 (W 0.0029573) | 2.76904 ± 0.00713073 (W 0.00553869) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.40107 ± 0.0226977 (W 0.008064) | 10.8076 ± 0.0203094 (W 0.0180903) | 21.7784 ± 0.0731059 (W 0.0572098) | 🔵 colmat 2.00× faster than `faer` | 🔵 colmat 4.03× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.44582 ± 0.0435477 (W 0.0127545) | 10.8227 ± 0.0100792 (W 0.013693) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.6931 ± 0.136981 (W 0.0365535) | 42.9972 ± 0.0956359 (W 0.0453598) | 81.4198 ± 0.0804057 (W 0.128049) | 🔵 colmat 1.45× faster than `faer` | 🔵 colmat 2.74× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 29.7978 ± 0.130517 (W 0.0341377) | 42.9005 ± 0.0423973 (W 0.0253436) | - | 🔵 colmat 1.44× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.38307 ± 0.00533388 (W 0.00330783) | 0.764991 ± 0.00427886 (W 0.00318585) | 22.0853 ± 0.0479467 (W 0.0180211) | 🔵 colmat 2.00× faster than `faer` | 🔵 colmat 57.65× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.486247 ± 0.00294473 (W 0.00454218) | 0.757764 ± 0.00365886 (W 0.00362698) | - | 🔵 colmat 1.56× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.77965 ± 0.0100728 (W 0.00663352) | 5.47953 ± 0.0324701 (W 0.0100242) | 173.977 ± 0.789569 (W 0.188391) | 🔵 colmat 3.08× faster than `faer` | 🔵 colmat 97.76× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.20795 ± 0.000566549 (W 0.00509997) | 5.43699 ± 0.0106286 (W 0.00954892) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.1058 ± 0.0390845 (W 0.0131856) | 23.3128 ± 0.146328 (W 0.0098234) | 1358.71 ± 3.02077 (W 1.24908) | 🔵 colmat 2.31× faster than `faer` | 🔵 colmat 134.45× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.2802 ± 0.0653016 (W 0.0179493) | 23.063 ± 0.0439253 (W 0.0403167) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.283195 ± 0.00104321 (W 0.000592862) | 0.278931 ± 0.000731687 (W 0.00190928) | 0.671402 ± 0.00408575 (W 0.0034029) | · | 🔵 colmat 2.37× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.282338 ± 0.0011322 (W 0.00102369) | 0.278691 ± 0.00120619 (W 0.00178129) | 0.47983 ± 0.00146561 (W 0.00447375) | · | 🔵 colmat 1.70× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.94375 ± 0.00490496 (W 0.00839419) | 1.92589 ± 0.0103852 (W 0.00406408) | 3.96497 ± 0.0256086 (W 0.00906658) | · | 🔵 colmat 2.04× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.94733 ± 0.0067589 (W 0.00336086) | 1.93271 ± 0.00967583 (W 0.0046069) | 2.61918 ± 0.0136257 (W 0.00472153) | · | 🔵 colmat 1.35× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6947 ± 0.0199456 (W 0.0240184) | 14.7192 ± 0.0575768 (W 0.019573) | 22.5602 ± 0.100936 (W 0.0457687) | · | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6899 ± 0.0211132 (W 0.0295929) | 14.7298 ± 0.0319908 (W 0.0243338) | 14.9332 ± 0.0510382 (W 0.0165728) | · | · |  |
| `qr/n128/b1` | 0.677734 ± 0.00332128 (W 0.00211469) | 0.960235 ± 0.00199691 (W 0.0024041) | 0.957279 ± 0.00398906 (W 0.00470638) | 🔵 colmat 1.42× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.894872 ± 0.0035867 (W 0.00244583) | 1.16353 ± 0.00547961 (W 0.00440175) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.98752 ± 0.0147265 (W 0.00420935) | 4.01438 ± 0.00169259 (W 0.00718963) | 4.65122 ± 0.0276558 (W 0.0125125) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 1.56× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.97653 ± 0.0118876 (W 0.0157149) | 4.98847 ± 0.0046476 (W 0.0187674) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n512/b1` | 18.9961 ± 0.0791579 (W 0.0500102) | 21.0079 ± 0.0640417 (W 0.0222956) | 29.3141 ± 0.0544539 (W 0.0345169) | 🔵 colmat 1.11× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 23.99 ± 0.0346382 (W 0.0343132) | 26.1831 ± 0.15433 (W 0.0415457) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.411881 ± 0.000651666 (W 0.00271194) | 0.658107 ± 0.00331291 (W 0.00464461) | 0.477582 ± 0.00241096 (W 0.00464503) | 🔵 colmat 1.60× faster than `faer` | 🔵 colmat 1.16× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.52736 ± 0.00226724 (W 0.00579108) | 2.56497 ± 0.00604695 (W 0.00498438) | 2.04362 ± 0.00694041 (W 0.00845894) | 🔵 colmat 1.68× faster than `faer` | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 8.73589 ± 0.0378058 (W 0.0148588) | 10.8248 ± 0.017309 (W 0.0128617) | 14.4979 ± 0.0437416 (W 0.0311964) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 1.66× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 2.68843 ± 0.00657019 (W 0.00547538) | 3.92911 ± 0.0107196 (W 0.0116732) | 5.28949 ± 0.0207071 (W 0.124812) | 🔵 colmat 1.46× faster than `faer` | 🔵 colmat 1.97× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.70221 ± 0.00935488 (W 0.00859172) | 3.93981 ± 0.0119363 (W 0.0150849) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.3676 ± 0.0198902 (W 0.0193262) | 21.1614 ± 0.0283436 (W 0.0340009) | 22.0095 ± 0.118662 (W 0.0831858) | 🔵 colmat 1.71× faster than `faer` | 🔵 colmat 1.78× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 12.4213 ± 0.0257611 (W 0.0133933) | 21.648 ± 0.10032 (W 0.0421447) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 65.3129 ± 0.174924 (W 0.0895578) | 84.9901 ± 0.0636424 (W 0.0587865) | 109.234 ± 0.269261 (W 0.179258) | 🔵 colmat 1.30× faster than `faer` | 🔵 colmat 1.67× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 65.5039 ± 0.149278 (W 0.0882436) | 85.4282 ± 0.0729348 (W 0.794824) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.252 ± 0.00417898 (W 0.00318431) | 2.92662 ± 0.0111728 (W 0.0040529) | 3.26875 ± 0.0223176 (W 4.46152) | 🔵 colmat 2.34× faster than `faer` | `?` not evidence vs `lapack-aocl`: a row's `W` is as large as the difference (rerun) |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.25751 ± 0.0044797 (W 0.00377313) | 2.94294 ± 0.00950934 (W 0.00626535) | - | 🔵 colmat 2.34× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.84866 ± 0.0354514 (W 0.00647726) | 16.2617 ± 0.0929302 (W 0.0283152) | 12.8503 ± 0.0918626 (W 0.0413562) | 🔵 colmat 2.78× faster than `faer` | 🔵 colmat 2.20× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.88062 ± 0.0277501 (W 0.00925294) | 16.2567 ± 0.0752157 (W 0.0460769) | - | 🔵 colmat 2.76× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 30.4076 ± 0.136337 (W 0.0324995) | 56.0305 ± 0.113293 (W 0.0552622) | 58.6448 ± 0.406309 (W 0.147835) | 🔵 colmat 1.84× faster than `faer` | 🔵 colmat 1.93× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 30.486 ± 0.102227 (W 0.0273841) | 55.9939 ± 0.0350172 (W 0.044882) | - | 🔵 colmat 1.84× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009648 ± 3.78594e-05 (W 4.47214e-05) | 0.008756 ± 5.72742e-05 (W 8.04363e-05) | 0.006583 ± 4.95681e-05 (W 3.39603e-05) | 🔴 colmat 1.10× slower than `faer` | 🔴 colmat 1.47× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.023755 ± 0.000237966 (W 0.00131848) | 0.025648 ± 0.000127703 (W 8.60192e-05) | 0.019175 ± 0.000633999 (W 0.000102602) | 🔵 colmat 1.08× faster than `faer` (marginal: spread 0.70 of the difference) | 🔴 colmat 1.24× slower than `lapack-aocl` | `!` between-start range over 5% |
| `triangular_solve/n512/b1` | 0.058337 ± 0.000614663 (W 0.00156456) | 0.063887 ± 0.000396694 (W 0.000237954) | 0.072803 ± 0.000278671 (W 0.0018875) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-aocl` |  |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `eigh/n128/b1` | 0.927531 ± 0.00157896 (W 0.00146443) | 2.77564 ± 0.00776238 (W 0.0027397) | 4.10284 ± 0.0506078 (W 0.0862655) | 🔵 colmat 2.99× faster than `faer` | 🔵 colmat 4.42× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.936622 ± 0.00595579 (W 0.00113968) | 2.78088 ± 0.0116809 (W 0.00460294) | - | 🔵 colmat 2.97× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.67573 ± 0.0265678 (W 0.00605684) | 10.5823 ± 0.00297101 (W 0.0351696) | 22.6505 ± 0.0471387 (W 0.0280058) | 🔵 colmat 2.88× faster than `faer` | 🔵 colmat 6.16× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.71198 ± 0.0141243 (W 0.0113541) | 10.6461 ± 0.0567871 (W 0.0272802) | - | 🔵 colmat 2.87× faster than `faer` | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 18.4061 ± 0.0490867 (W 0.0224721) | 37.3845 ± 0.0961549 (W 0.183579) | 73.4997 ± 0.406983 (W 0.0424609) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 3.99× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 18.4034 ± 0.081653 (W 0.0214629) | 37.4076 ± 0.240325 (W 0.213815) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.370801 ± 0.00667302 (W 0.0016274) | 0.754517 ± 0.00339624 (W 0.00218216) | 22.218 ± 0.150661 (W 0.0326288) | 🔵 colmat 2.03× faster than `faer` | 🔵 colmat 59.92× faster than `lapack-aocl` |  |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.482888 ± 0.00156129 (W 0.00398935) | 0.757724 ± 0.00613477 (W 0.00381216) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.33194 ± 0.00816974 (W 0.00547621) | 5.48699 ± 0.0377724 (W 0.0121681) | 173.75 ± 0.517574 (W 0.111377) | 🔵 colmat 4.12× faster than `faer` | 🔵 colmat 130.45× faster than `lapack-aocl` |  |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.91512 ± 0.0119335 (W 0.0104594) | 5.4429 ± 0.0171682 (W 0.00857011) | - | 🔵 colmat 2.84× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.11043 ± 0.0363473 (W 0.0139059) | 21.2718 ± 0.151185 (W 0.0352904) | 1350.56 ± 6.02108 (W 0.757936) | 🔵 colmat 3.48× faster than `faer` | 🔵 colmat 221.03× faster than `lapack-aocl` |  |
| `full_piv_lu/n512/b1 [api-inclusive]` | 8.97552 ± 0.0792065 (W 0.0446374) | 20.9516 ± 0.0664528 (W 0.0424908) | - | 🔵 colmat 2.33× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.167809 ± 0.00177342 (W 0.0015399) | 0.269881 ± 0.000638526 (W 0.00363309) | 0.590917 ± 0.00258315 (W 0.00338504) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 3.52× faster than `lapack-aocl` |  |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.169281 ± 0.00168187 (W 0.00244519) | 0.268508 ± 0.00267405 (W 0.00173303) | 0.389787 ± 0.00453017 (W 0.00586492) | 🔵 colmat 1.59× faster than `faer` | 🔵 colmat 2.30× faster than `lapack-aocl` |  |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.02262 ± 0.00543954 (W 0.002106) | 1.13803 ± 0.100107 (W 0.103534) | 3.15695 ± 0.0109892 (W 0.0122748) | 🔵 colmat 1.11× faster than `faer` (marginal: spread 0.91 of the difference) | 🔵 colmat 3.09× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.02753 ± 0.00582917 (W 0.00445411) | 1.28519 ± 0.095699 (W 0.091891) | 1.58395 ± 0.00870783 (W 0.00879869) | 🔵 colmat 1.25× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.97808 ± 0.0683515 (W 0.162067) | 10.1059 ± 1.35639 (W 1.16518) | 15.7444 ± 0.170502 (W 0.0443045) | 🔵 colmat 1.27× faster than `faer` (marginal: spread 0.67 of the difference) | 🔵 colmat 1.97× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 8.0667 ± 0.0543258 (W 0.158049) | 10.1427 ± 1.41459 (W 1.37253) | 8.78685 ± 0.0113092 (W 0.0258773) | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.71 of the difference) | 🔵 colmat 1.09× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/n128/b1` | 0.531475 ± 0.00347203 (W 0.00365484) | 1.06241 ± 0.00600029 (W 0.00604214) | 0.846323 ± 0.0142499 (W 0.00750614) | 🔵 colmat 2.00× faster than `faer` | 🔵 colmat 1.59× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.764022 ± 0.00426311 (W 0.00462201) | 1.27294 ± 0.00460392 (W 0.00448061) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.99593 ± 0.00355764 (W 0.0033213) | 3.97373 ± 0.0287453 (W 0.0398823) | 3.07454 ± 0.0112597 (W 0.0216677) | 🔵 colmat 1.99× faster than `faer` | 🔵 colmat 1.54× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.08823 ± 0.264468 (W 0.0941872) | 4.71382 ± 0.428476 (W 0.281386) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 10.0093 ± 0.0446481 (W 0.0149649) | 17.1724 ± 0.11238 (W 0.246383) | 17.1738 ± 0.495371 (W 0.232936) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 1.72× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 15.2256 ± 0.0135407 (W 0.0417538) | 19.1563 ± 2.43437 (W 1.99524) | - | 🔵 colmat 1.26× faster than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.36425 ± 0.00177103 (W 0.00212417) | 0.752695 ± 0.00148962 (W 0.00287184) | 0.450655 ± 0.00420181 (W 0.00744554) | 🔵 colmat 2.07× faster than `faer` | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.2187 ± 0.00790585 (W 0.00438501) | 2.70483 ± 0.00851721 (W 0.0161667) | 1.52117 ± 0.00407618 (W 0.00933866) | 🔵 colmat 2.22× faster than `faer` | 🔵 colmat 1.25× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 4.72569 ± 0.0214899 (W 0.00288015) | 9.82364 ± 0.113442 (W 0.116778) | 8.47575 ± 0.0443357 (W 0.0260715) | 🔵 colmat 2.08× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1` | 2.48442 ± 0.00662036 (W 0.00560724) | 3.94499 ± 0.01672 (W 0.0141432) | 6.85713 ± 0.13552 (W 4.50541) | 🔵 colmat 1.59× faster than `faer` | `?` not evidence vs `lapack-aocl`: a row's `W` is as large as the difference (rerun) |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.49519 ± 0.0205608 (W 0.00684643) | 3.97945 ± 0.00225629 (W 0.00694483) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.2036 ± 0.0252887 (W 0.00823963) | 20.779 ± 0.0694903 (W 0.0528028) | 21.9326 ± 5.19945 (W 4.52178) | 🔵 colmat 2.04× faster than `faer` | 🔵 colmat 2.15× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `svd_thin/n256/b1 [api-inclusive]` | 10.2429 ± 0.0326758 (W 0.0265948) | 21.5138 ± 0.303018 (W 0.0767315) | - | 🔵 colmat 2.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 46.9067 ± 0.3116 (W 0.220078) | 72.3863 ± 0.207113 (W 0.188994) | 92.4782 ± 0.235374 (W 0.173625) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 1.97× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 46.7467 ± 0.249514 (W 0.0591777) | 72.5015 ± 0.468223 (W 0.245835) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.2459 ± 0.016341 (W 0.00342867) | 2.92141 ± 0.0087003 (W 0.00499194) | 5.01026 ± 0.0969347 (W 8.01894) | 🔵 colmat 2.34× faster than `faer` | `?` not evidence vs `lapack-aocl`: a row's `W` is as large as the difference (rerun) |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.24737 ± 0.0120004 (W 0.00275385) | 2.9393 ± 0.00977215 (W 0.00534864) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.39797 ± 0.0403129 (W 0.00383193) | 16.1565 ± 0.106155 (W 0.0278547) | 14.2915 ± 0.0574102 (W 2.30801) | 🔵 colmat 2.99× faster than `faer` | 🔵 colmat 2.65× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.36464 ± 0.0169265 (W 0.00616118) | 16.2936 ± 0.134373 (W 0.0139924) | - | 🔵 colmat 3.04× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 23.6239 ± 0.116439 (W 0.068278) | 50.1301 ± 0.0768947 (W 0.0251863) | 51.4396 ± 0.219042 (W 0.105712) | 🔵 colmat 2.12× faster than `faer` | 🔵 colmat 2.18× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 23.7781 ± 0.0843829 (W 0.189024) | 50.1033 ± 0.0672028 (W 0.196339) | - | 🔵 colmat 2.11× faster than `faer` | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009718 ± 4.72582e-05 (W 7.46324e-05) | 0.008766 ± 3.00056e-05 (W 4.14729e-05) | 0.006873 ± 3.84318e-05 (W 3.19092e-05) | 🔴 colmat 1.11× slower than `faer` | 🔴 colmat 1.41× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024235 ± 0.000288722 (W 0.000148762) | 0.025848 ± 0.000377978 (W 0.00214428) | 0.022281 ± 4.93288e-05 (W 6.96297e-05) | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | 🔴 colmat 1.09× slower than `lapack-aocl` |  |
| `triangular_solve/n512/b1` | 0.057466 ± 0.000419262 (W 0.000277381) | 0.062465 ± 0.000421804 (W 0.0001865) | 0.082641 ± 0.00114904 (W 0.00205727) | 🔵 colmat 1.09× faster than `faer` | 🔵 colmat 1.44× faster than `lapack-aocl` |  |

Not applicable to this population's shapes and reported as skipped: `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.773083 ± 0.00393231 (W 0.00452865) | 1.12456 ± 0.00699501 (W 0.00468813) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.659063 ± 0.00169524 (W 0.00337672) | 0.717772 ± 0.00179089 (W 0.00578856) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 45.7417 ± 0.371232 (W 0.796988) | 82.8892 ± 7.35236 (W 0.52193) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 41.4495 ± 0.0845058 (W 0.187666) | 46.5468 ± 1.39382 (W 0.739576) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.556085 ± 0.00219258 (W 0.00322042) | 0.558049 ± 0.00534065 (W 0.00309122) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.554974 ± 0.00289766 (W 0.00282137) | 0.55844 ± 0.00924537 (W 0.00241825) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.077 ± 0.13833 (W 0.240261) | 35.1883 ± 0.136293 (W 0.0472948) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.0956 ± 0.130205 (W 0.19214) | 35.1391 ± 0.105078 (W 0.129428) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.45945 ± 0.00622651 (W 0.00426704) | 2.42523 ± 0.00971539 (W 0.00861763) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.70872 ± 0.00716421 (W 0.00529632) | 1.80255 ± 0.0130873 (W 0.00568561) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 68.1777 ± 0.463335 (W 0.262328) | 115.068 ± 0.218344 (W 0.240189) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 71.8951 ± 1.05913 (W 0.434725) | 75.8729 ± 0.500037 (W 0.540451) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.913393 ± 0.014837 (W 0.00319788) | 1.87381 ± 0.00269059 (W 0.0055944) | - | 🔵 colmat 2.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 31.3579 ± 0.138992 (W 0.164189) | 77.9871 ± 0.189529 (W 0.127056) | - | 🔵 colmat 2.49× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.0583 ± 0.00961032 (W 0.0092259) | 2.3159 ± 0.00268253 (W 0.00425449) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.07816 ± 0.0130292 (W 0.006697) | 2.31954 ± 0.00894276 (W 0.00600018) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 84.3348 ± 0.52328 (W 0.151919) | 87.7403 ± 0.243306 (W 0.232628) | - | · | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 83.5029 ± 0.216509 (W 0.19449) | 87.8636 ± 0.280415 (W 0.205557) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.04565 ± 0.00659632 (W 0.00454394) | 1.49304 ± 0.00884893 (W 0.00664004) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.04278 ± 0.00176317 (W 0.00447374) | 1.49497 ± 0.0093961 (W 0.00465933) | - | 🔵 colmat 1.43× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 37.8602 ± 0.15676 (W 0.183584) | 44.506 ± 0.210793 (W 0.0407219) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 37.7879 ± 0.160165 (W 0.0675133) | 44.5017 ± 0.17478 (W 0.0847705) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.449743 ± 0.00291498 (W 0.00219729) | 1.13165 ± 0.00179161 (W 0.00558723) | - | 🔵 colmat 2.52× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.479457 ± 0.00746251 (W 0.00380351) | 0.726645 ± 0.0021285 (W 0.00626965) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 12.8261 ± 0.0225466 (W 0.0822789) | 45.633 ± 0.0507599 (W 0.17939) | - | 🔵 colmat 3.56× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 13.6128 ± 0.0324005 (W 0.0471564) | 20.0729 ± 1.00053 (W 0.224345) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.165819 ± 0.00279867 (W 0.0026366) | 0.156801 ± 0.00230999 (W 0.000678522) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.166077 ± 0.00390168 (W 0.00303799) | 0.156181 ± 0.0011112 (W 0.000938698) | - | 🔴 colmat 1.06× slower than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.51433 ± 0.0494975 (W 0.0136649) | 9.43998 ± 0.0227277 (W 0.0223077) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.53317 ± 0.0519181 (W 0.0392997) | 9.4254 ± 0.0565786 (W 0.018901) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.710736 ± 0.00262138 (W 0.00364278) | 2.92717 ± 0.0387217 (W 0.0643495) | - | 🔵 colmat 4.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.04821 ± 0.00895206 (W 0.00361767) | 1.45172 ± 0.0106902 (W 0.0343328) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 23.722 ± 0.0528718 (W 0.204797) | 84.445 ± 0.0813555 (W 0.363314) | - | 🔵 colmat 3.56× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 30.8522 ± 0.0225922 (W 0.123205) | 34.3301 ± 0.0297798 (W 0.0680235) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.573952 ± 0.000443544 (W 0.00254188) | 2.73249 ± 0.064528 (W 0.0881059) | - | 🔵 colmat 4.76× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 12.9876 ± 0.0978249 (W 0.0609053) | 74.0269 ± 0.681812 (W 0.252329) | - | 🔵 colmat 5.70× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.33048 ± 0.0079209 (W 0.0078213) | 1.96656 ± 0.00750761 (W 0.00383461) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.34685 ± 0.00973007 (W 0.00806481) | 1.97694 ± 0.00953099 (W 0.00710626) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 37.0902 ± 0.128308 (W 0.116207) | 46.926 ± 0.0525885 (W 0.0673596) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 37.4642 ± 0.0236208 (W 0.0845088) | 47.3147 ± 0.0588952 (W 0.102997) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.863946 ± 0.000384994 (W 0.0081889) | 1.53675 ± 0.00532297 (W 0.0068063) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.872923 ± 0.00920985 (W 0.00445915) | 1.54605 ± 0.00445329 (W 0.00591915) | - | 🔵 colmat 1.77× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 22.2388 ± 0.0743525 (W 0.0530288) | 32.6079 ± 0.0626669 (W 0.0602357) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 22.2397 ± 0.108694 (W 0.0516028) | 32.5522 ± 0.11688 (W 0.0433398) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.435779 ± 0.00946223 (W 0.00583348) | 1.12169 ± 0.00796306 (W 0.00465054) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.475849 ± 0.00265735 (W 0.00409498) | 0.726402 ± 0.00984123 (W 0.00624792) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 7.53029 ± 0.0402136 (W 0.0139086) | 40.0703 ± 0.119143 (W 0.0985767) | - | 🔵 colmat 5.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 9.41434 ± 0.0381067 (W 0.0945433) | 16.2958 ± 1.20647 (W 0.833646) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.107209 ± 0.00175512 (W 0.00358264) | 0.144327 ± 0.00852201 (W 0.00686432) | - | 🔵 colmat 1.35× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.105626 ± 0.000145317 (W 0.00255754) | 0.138536 ± 0.0019513 (W 0.00807863) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.73266 ± 0.0186231 (W 0.0347027) | 5.07772 ± 0.937812 (W 0.888601) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.73661 ± 0.0452077 (W 0.0427007) | 6.68445 ± 0.0301884 (W 0.77682) | - | 🔵 colmat 1.17× faster than `faer` (marginal: spread 0.82 of the difference) | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.609602 ± 0.00204992 (W 0.00337559) | 3.03139 ± 0.0263391 (W 0.103413) | - | 🔵 colmat 4.97× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.953745 ± 0.00982213 (W 0.00541395) | 1.22783 ± 0.0170959 (W 0.0357877) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 15.9112 ± 0.0530413 (W 0.0413608) | 80.3036 ± 0.499884 (W 0.143291) | - | 🔵 colmat 5.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 24.2662 ± 0.0650077 (W 0.0970918) | 33.0658 ± 0.185408 (W 2.46273) | - | 🔵 colmat 1.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.487394 ± 0.00454172 (W 0.00444508) | 2.86267 ± 0.0243894 (W 0.060996) | - | 🔵 colmat 5.87× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 9.26473 ± 0.0205131 (W 0.0473444) | 72.115 ± 0.328455 (W 0.21164) | - | 🔵 colmat 7.78× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.23459 ± 0.0100317 (W 0.00474065) | 2.05471 ± 0.0114719 (W 0.00752842) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.25483 ± 0.0107812 (W 0.0117274) | 2.07639 ± 0.00883349 (W 0.0071337) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 30.0218 ± 0.143686 (W 0.149954) | 45.8427 ± 0.310231 (W 0.3608) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 30.4863 ± 0.177092 (W 0.181288) | 46.5492 ± 0.436594 (W 0.993984) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.826594 ± 0.00519828 (W 0.00589372) | 1.62749 ± 0.00841525 (W 0.00460101) | - | 🔵 colmat 1.97× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.829099 ± 0.00763663 (W 0.00501447) | 1.62407 ± 0.0160579 (W 0.00914673) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 20.9277 ± 1.4609 (W 0.380586) | 33.4444 ± 1.67284 (W 0.106291) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.3379 ± 0.043385 (W 0.0482194) | 33.4144 ± 0.0851555 (W 0.107941) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.661017 ± 0.0040721 (W 0.00305072) | 0.735204 ± 0.0035183 (W 0.00460807) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.570931 ± 0.00186083 (W 0.00257154) | 0.637344 ± 0.00342728 (W 0.00305904) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 44.674 ± 0.171024 (W 0.24602) | 46.5338 ± 0.406886 (W 0.789575) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 39.0993 ± 0.493674 (W 0.295261) | 40.2715 ± 0.655825 (W 0.0533419) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.60214 ± 0.00973902 (W 0.00376433) | 0.599946 ± 0.0064801 (W 0.00215147) | 0.718485 ± 0.00354603 (W 0.00410836) | · | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.596691 ± 0.00047094 (W 0.00397182) | 0.600026 ± 0.00811513 (W 0.00309577) | 2.10214 ± 0.00842487 (W 0.0032809) | · | 🔵 colmat 3.52× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.9466 ± 0.121126 (W 0.0766184) | 35.9714 ± 0.170769 (W 0.0940553) | 37.9488 ± 0.0332631 (W 0.0948504) | · | 🔵 colmat 1.06× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.9702 ± 0.116589 (W 0.103421) | 35.985 ± 0.124481 (W 0.0593146) | 48.338 ± 0.231847 (W 0.0532094) | · | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 1.25792 ± 0.0043204 (W 0.00247203) | 1.38209 ± 0.00475596 (W 0.00301998) | 2.84291 ± 0.0120512 (W 0.00713048) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 2.26× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.57598 ± 0.0106103 (W 0.00505341) | 1.71214 ± 0.00700127 (W 0.00351215) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 61.4732 ± 0.359568 (W 0.0293335) | 62.1035 ± 0.450589 (W 0.193126) | 75.9476 ± 0.337562 (W 0.0557395) | · | 🔵 colmat 1.24× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 69.0126 ± 0.200477 (W 0.0857973) | 70.0942 ± 0.161563 (W 0.404727) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.667991 ± 0.00423328 (W 0.00430798) | 0.78719 ± 0.00196395 (W 0.00270327) | 0.939373 ± 0.00561877 (W 0.00528505) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 25.0229 ± 0.0701589 (W 0.0623962) | 25.8918 ± 0.108335 (W 0.0389098) | 28.0445 ± 0.246398 (W 0.0327765) | · | 🔵 colmat 1.12× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.98129 ± 0.00702626 (W 0.00509715) | 2.25918 ± 0.00686953 (W 0.00262158) | 2.73742 ± 0.0106176 (W 0.00462404) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.98936 ± 0.00983604 (W 0.00203304) | 2.27575 ± 0.00740936 (W 0.00481756) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 78.6007 ± 0.363066 (W 0.320382) | 82.2545 ± 0.247002 (W 0.100802) | 101.212 ± 0.739081 (W 0.317295) | · | 🔵 colmat 1.29× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 78.1308 ± 0.388807 (W 0.335824) | 82.5905 ± 0.288363 (W 0.419561) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.953847 ± 0.00535453 (W 0.0029753) | 1.39949 ± 0.00608117 (W 0.00328042) | 1.27546 ± 0.0107372 (W 0.00497698) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.953918 ± 0.00452031 (W 0.00203732) | 1.39914 ± 0.00509571 (W 0.00721542) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 32.3549 ± 0.1489 (W 0.0290307) | 39.0062 ± 0.187897 (W 0.229573) | 37.8335 ± 0.242671 (W 0.044485) | 🔵 colmat 1.21× faster than `faer` | 🔵 colmat 1.17× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 32.2851 ± 0.353746 (W 0.184881) | 38.9894 ± 0.160509 (W 0.292916) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.38382 ± 0.00144016 (W 0.00232986) | 0.738052 ± 0.00757294 (W 0.00363693) | - | 🔵 colmat 1.92× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.395139 ± 0.00583558 (W 0.00211238) | 0.639855 ± 0.001175 (W 0.0044637) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 12.6264 ± 0.00570565 (W 0.0840368) | 14.6933 ± 0.0153338 (W 0.0195069) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.2105 ± 0.120353 (W 0.0311003) | 13.3909 ± 0.0217295 (W 0.0175854) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.201124 ± 0.00158153 (W 0.00142032) | 0.198246 ± 0.00275249 (W 0.00236856) | 0.34933 ± 0.0010275 (W 0.00385896) | · | 🔵 colmat 1.74× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.201162 ± 0.000293115 (W 0.00109669) | 0.196245 ± 0.00143336 (W 0.00225569) | 2.10607 ± 0.00253271 (W 0.0027329) | · | 🔵 colmat 10.47× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 11.1814 ± 0.0102635 (W 0.0219138) | 11.1178 ± 0.00749332 (W 0.0193438) | 12.5197 ± 0.102023 (W 0.0330053) | · | 🔵 colmat 1.12× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 11.1563 ± 0.0295638 (W 0.0120751) | 11.0976 ± 0.016929 (W 0.0193034) | 14.8436 ± 0.0703293 (W 0.0197566) | · | 🔵 colmat 1.33× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 0.670699 ± 0.00602179 (W 0.0032109) | 1.01258 ± 0.00173091 (W 0.00462388) | 2.84166 ± 0.00450747 (W 0.00410496) | 🔵 colmat 1.51× faster than `faer` | 🔵 colmat 4.24× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.939109 ± 0.00256492 (W 0.00297052) | 1.36413 ± 0.0305934 (W 0.0368878) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 21.8182 ± 0.0636871 (W 0.0984716) | 26.5657 ± 0.00666098 (W 0.0831127) | 24.5618 ± 0.069882 (W 0.0180385) | 🔵 colmat 1.22× faster than `faer` | 🔵 colmat 1.13× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 28.4474 ± 0.0722872 (W 0.0569401) | 28.4443 ± 0.137895 (W 1.00812) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.483218 ± 0.00102942 (W 0.00309456) | 0.836084 ± 0.00266958 (W 0.00738559) | 0.942451 ± 0.00174883 (W 0.00179384) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 1.95× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 10.8665 ± 0.0412557 (W 0.0163816) | 15.6062 ± 0.0298181 (W 0.036842) | 9.54617 ± 0.0432069 (W 0.00964771) | 🔵 colmat 1.44× faster than `faer` | 🔴 colmat 1.14× slower than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.26145 ± 0.00278284 (W 0.00408729) | 1.91402 ± 0.00318063 (W 0.00683171) | 2.53565 ± 0.013409 (W 0.00325792) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 2.01× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.27463 ± 0.00561352 (W 0.00366226) | 1.92099 ± 0.00410042 (W 0.00519846) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 31.5882 ± 0.0215505 (W 0.27597) | 41.6174 ± 0.062185 (W 0.02885) | 45.9584 ± 0.202821 (W 0.0297366) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 1.45× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 31.8292 ± 0.0772536 (W 0.125835) | 41.9169 ± 0.135296 (W 0.059498) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.789528 ± 0.00499686 (W 0.00257757) | 1.45182 ± 0.00373473 (W 0.00629297) | 1.27192 ± 0.00199858 (W 0.00334999) | 🔵 colmat 1.84× faster than `faer` | 🔵 colmat 1.61× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.796101 ± 0.00741187 (W 0.00295955) | 1.4527 ± 0.0101569 (W 0.00615787) | - | 🔵 colmat 1.82× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 16.7186 ± 0.0213826 (W 0.0330758) | 26.838 ± 0.0588825 (W 0.311514) | 19.7218 ± 0.0326512 (W 0.0441227) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 1.18× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 16.7337 ± 0.0491457 (W 0.0235888) | 26.8722 ± 0.0556903 (W 0.0583766) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.362204 ± 0.000289028 (W 0.00230389) | 0.73586 ± 0.00360319 (W 0.00342124) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.390316 ± 0.00321607 (W 0.00282993) | 0.640893 ± 0.00287074 (W 0.00298889) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 7.67141 ± 0.0852785 (W 0.0324895) | 9.88206 ± 0.0579125 (W 0.0169927) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 7.06856 ± 0.0476922 (W 0.0174511) | 9.35096 ± 0.0808859 (W 0.0171639) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.127046 ± 0.00129571 (W 0.000820458) | 0.190605 ± 0.0036045 (W 0.00795769) | 0.307152 ± 0.00349145 (W 0.00704377) | 🔵 colmat 1.50× faster than `faer` | 🔵 colmat 2.42× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.127418 ± 0.0005467 (W 0.00127848) | 0.190264 ± 0.00377852 (W 0.00648266) | 2.08485 ± 0.000915603 (W 0.00225483) | 🔵 colmat 1.49× faster than `faer` | 🔵 colmat 16.36× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 8.86218 ± 0.015731 (W 0.0309945) | 7.03569 ± 0.6904 (W 0.748976) | 7.43852 ± 0.0441808 (W 0.016327) | 🔴 colmat 1.26× slower than `faer` | 🔴 colmat 1.19× slower than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 8.8666 ± 0.029408 (W 0.0291969) | 8.26684 ± 0.741883 (W 0.661305) | 8.63558 ± 0.0722401 (W 0.037373) | · | · | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/1024x64/b1` | 0.55366 ± 0.00395421 (W 0.0027122) | 1.10256 ± 0.00676056 (W 0.00481623) | 2.83042 ± 0.00876795 (W 0.00552213) | 🔵 colmat 1.99× faster than `faer` | 🔵 colmat 5.11× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.836286 ± 0.00230361 (W 0.0037217) | 1.14545 ± 0.0104929 (W 0.0298599) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 15.4597 ± 0.0504628 (W 0.318988) | 25.4501 ± 0.037615 (W 0.227791) | 16.8676 ± 0.049522 (W 0.0279883) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.09× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 21.9945 ± 0.0158104 (W 0.114435) | 27.5508 ± 1.40063 (W 2.54637) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.447352 ± 0.00141365 (W 0.00239234) | 0.925192 ± 0.00468682 (W 0.0054222) | 0.932626 ± 0.00394856 (W 0.000963242) | 🔵 colmat 2.07× faster than `faer` | 🔵 colmat 2.08× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 7.99561 ± 0.02996 (W 0.037236) | 16.5067 ± 0.116586 (W 0.0535822) | 7.94871 ± 0.0371602 (W 0.0103019) | 🔵 colmat 2.06× faster than `faer` | · |  |
| `svd_thin/1024x64/b1` | 1.16045 ± 0.00129428 (W 0.00176088) | 2.00673 ± 0.0119857 (W 0.00767608) | 2.56675 ± 0.0157833 (W 0.00702912) | 🔵 colmat 1.73× faster than `faer` | 🔵 colmat 2.21× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.17664 ± 0.00131329 (W 0.00412736) | 2.02393 ± 0.00355022 (W 0.00959395) | - | 🔵 colmat 1.72× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 24.5409 ± 0.0568767 (W 0.115578) | 40.2169 ± 0.128901 (W 0.300693) | 40.829 ± 0.0320624 (W 0.124961) | 🔵 colmat 1.64× faster than `faer` | 🔵 colmat 1.66× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 24.8137 ± 0.138582 (W 0.17154) | 40.6387 ± 0.161685 (W 0.175326) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.752641 ± 0.00634498 (W 0.00294361) | 1.55415 ± 0.019476 (W 0.0113538) | 1.26767 ± 0.0056725 (W 0.00420678) | 🔵 colmat 2.06× faster than `faer` | 🔵 colmat 1.68× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.754124 ± 0.00347375 (W 0.00211616) | 1.55156 ± 0.0106344 (W 0.00436243) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 13.7244 ± 0.0391004 (W 0.0203177) | 27.712 ± 0.0896286 (W 0.0529563) | 20.4034 ± 0.254948 (W 0.0526046) | 🔵 colmat 2.02× faster than `faer` | 🔵 colmat 1.49× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.7512 ± 0.0234066 (W 0.0436096) | 27.8666 ± 0.118293 (W 0.0877834) | - | 🔵 colmat 2.03× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.75809 ± 0.00444782 (W 0.00476045) | 2.65702 ± 0.193717 (W 0.0254798) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.62122 ± 0.00114334 (W 0.00431586) | 1.67988 ± 0.0077612 (W 0.00240319) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 132.061 ± 0.507557 (W 0.42222) | 196.853 ± 0.484929 (W 0.482909) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 128.431 ± 0.928738 (W 1.00819) | 133.95 ± 2.06115 (W 0.373596) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.20611 ± 0.00152868 (W 0.00258833) | 2.20095 ± 0.0248404 (W 0.00392206) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.20527 ± 0.00279902 (W 0.0031736) | 2.20775 ± 0.0302587 (W 0.00745714) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 137.799 ± 0.415048 (W 0.260197) | 137.984 ± 0.107175 (W 0.181957) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 137.748 ± 0.224205 (W 0.242514) | 137.992 ± 0.370314 (W 0.512326) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.56733 ± 0.0107693 (W 0.00712008) | 6.06703 ± 0.170694 (W 0.0103681) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 5.04549 ± 0.00794161 (W 0.00426928) | 5.26818 ± 0.0151212 (W 0.00818069) | - | · | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 239.339 ± 0.264539 (W 0.665623) | 317.661 ± 0.311395 (W 0.422045) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 250.172 ± 0.141514 (W 0.434737) | 258.968 ± 0.265851 (W 0.251674) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.31023 ± 0.00168436 (W 0.011609) | 3.99682 ± 0.0549307 (W 0.014873) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 98.0784 ± 0.200954 (W 0.210025) | 176.174 ± 0.409343 (W 0.30677) | - | 🔵 colmat 1.80× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 5.40372 ± 0.0340021 (W 0.00265597) | 5.78703 ± 0.00291951 (W 0.00378919) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.43109 ± 0.00462136 (W 0.00593361) | 5.82372 ± 0.00539148 (W 0.00614293) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 273.272 ± 0.468473 (W 0.40484) | 280.512 ± 0.240531 (W 0.605) | - | · | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 273.674 ± 0.510011 (W 0.394006) | 280.867 ± 0.0567355 (W 0.37173) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.48124 ± 0.00577657 (W 0.00421351) | 3.07512 ± 0.0083546 (W 0.00106411) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.48338 ± 0.00527112 (W 0.00728142) | 3.08358 ± 0.00801134 (W 0.00878674) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 113.447 ± 0.843905 (W 0.633915) | 122.642 ± 0.0610742 (W 0.398153) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 115.074 ± 0.794006 (W 0.694283) | 122.818 ± 0.270249 (W 0.120204) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.802978 ± 0.00366619 (W 0.00166593) | 2.65906 ± 0.0360342 (W 0.0188713) | - | 🔵 colmat 3.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.883853 ± 0.00655933 (W 0.00566694) | 1.66433 ± 0.0344716 (W 0.00665832) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 42.1789 ± 0.232633 (W 0.165595) | 99.4571 ± 0.256097 (W 0.383292) | - | 🔵 colmat 2.36× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 44.9035 ± 0.369325 (W 0.288751) | 57.9201 ± 0.419841 (W 0.635993) | - | 🔵 colmat 1.29× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.592566 ± 0.00563323 (W 0.00403723) | 0.572448 ± 0.00224309 (W 0.00200416) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.595809 ± 0.00115665 (W 0.00424004) | 0.577291 ± 0.0093441 (W 0.00455894) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 37.0865 ± 0.330994 (W 0.0898519) | 36.5053 ± 0.228831 (W 0.329046) | - | · | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 36.5926 ± 0.0674287 (W 0.205427) | 36.5533 ± 0.25319 (W 0.312984) | - | · | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.64063 ± 0.0174084 (W 0.00605821) | 4.32551 ± 0.022932 (W 0.0471221) | - | 🔵 colmat 2.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.41799 ± 0.0173939 (W 0.00659283) | 3.0159 ± 0.0489495 (W 0.0256427) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 75.0321 ± 0.916913 (W 0.416946) | 167.943 ± 0.195509 (W 0.361025) | - | 🔵 colmat 2.24× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 90.9573 ± 0.27083 (W 0.17987) | 107.794 ± 6.58567 (W 0.242951) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.12813 ± 0.011882 (W 0.0064173) | 3.70695 ± 0.042848 (W 0.0739832) | - | 🔵 colmat 3.29× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 35.0971 ± 0.177231 (W 0.0635323) | 128.987 ± 0.0935273 (W 0.528487) | - | 🔵 colmat 3.68× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.52767 ± 0.0102027 (W 0.00407727) | 3.48035 ± 0.00164891 (W 0.0214992) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.5334 ± 0.0104939 (W 0.00755721) | 3.50208 ± 0.00970893 (W 0.00687973) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 97.1805 ± 0.162991 (W 0.333425) | 117.342 ± 0.130042 (W 0.0914047) | - | 🔵 colmat 1.21× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 98.0806 ± 0.338624 (W 0.152987) | 117.879 ± 0.175416 (W 0.0887491) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.51897 ± 0.0108718 (W 0.00638455) | 2.45409 ± 0.00650939 (W 0.0049172) | - | 🔵 colmat 1.62× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.52344 ± 0.00838426 (W 0.00611608) | 2.44501 ± 0.0120209 (W 0.00626861) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 51.2475 ± 0.0957809 (W 0.0493402) | 71.7478 ± 0.0176672 (W 0.0756582) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 51.2876 ± 0.198221 (W 0.104788) | 71.7355 ± 0.0550611 (W 0.0497493) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.68907 ± 0.00252223 (W 0.0049246) | 2.68989 ± 0.0494929 (W 0.0205303) | - | 🔵 colmat 3.90× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.770862 ± 0.00273752 (W 0.00330063) | 1.66191 ± 0.0049423 (W 0.00603081) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 22.4475 ± 0.184144 (W 0.0456214) | 74.4132 ± 0.450073 (W 0.428341) | - | 🔵 colmat 3.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 26.101 ± 0.019464 (W 0.123924) | 37.7288 ± 0.745165 (W 0.851699) | - | 🔵 colmat 1.45× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.310338 ± 0.00597845 (W 0.00286411) | 0.469579 ± 0.046755 (W 0.031008) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.313591 ± 0.00683634 (W 0.00246399) | 0.466489 ± 0.00475919 (W 0.0433277) | - | 🔵 colmat 1.49× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 24.089 ± 0.0758582 (W 0.086579) | 25.2784 ± 3.46792 (W 3.51108) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 24.0687 ± 0.0486741 (W 0.0770379) | 25.4743 ± 0.0657618 (W 3.5871) | - | `?` not evidence vs `faer`: a row's `W` is as large as the difference (rerun) | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.21071 ± 0.0092647 (W 0.00495801) | 4.28058 ± 0.107294 (W 0.0450431) | - | 🔵 colmat 3.54× faster than `faer` | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.00073 ± 0.00791062 (W 0.00577997) | 2.39256 ± 0.164102 (W 0.0436418) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 46.5036 ± 0.311001 (W 0.155929) | 152.249 ± 0.879813 (W 2.02982) | - | 🔵 colmat 3.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 67.1759 ± 0.260131 (W 0.32746) | 84.8115 ± 3.1386 (W 0.48985) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.884543 ± 0.00621193 (W 0.0027612) | 3.69762 ± 0.110174 (W 0.0405936) | - | 🔵 colmat 4.18× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 24.828 ± 0.13589 (W 0.0889922) | 124.205 ± 0.877876 (W 0.58635) | - | 🔵 colmat 5.00× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.08134 ± 0.00766603 (W 0.00714829) | 3.44474 ± 0.00732217 (W 0.0353774) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.11884 ± 0.00253172 (W 0.00476055) | 3.47284 ± 0.00978219 (W 0.0279604) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 71.8965 ± 0.181648 (W 0.102431) | 105.281 ± 0.218791 (W 1.59308) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 72.6709 ± 0.0833279 (W 0.214228) | 106.803 ± 1.05287 (W 2.52265) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.37022 ± 0.00195541 (W 0.00548542) | 2.48152 ± 0.00854365 (W 0.00775838) | - | 🔵 colmat 1.81× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.38178 ± 0.00856849 (W 0.00353249) | 2.4792 ± 0.0131406 (W 0.0142339) | - | 🔵 colmat 1.79× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 41.7396 ± 0.0942584 (W 0.213318) | 69.4086 ± 0.107508 (W 0.50416) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 41.6681 ± 0.130223 (W 0.0655025) | 69.2255 ± 0.119348 (W 0.476558) | - | 🔵 colmat 1.66× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.51948 ± 0.00194712 (W 0.0053745) | 1.55028 ± 0.0104431 (W 0.00644621) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.49847 ± 0.00596825 (W 0.00404735) | 1.53129 ± 0.00270657 (W 0.00393553) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 125.83 ± 0.6421 (W 0.601964) | 125.229 ± 1.01789 (W 0.164711) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 122.223 ± 0.423729 (W 1.00227) | 121.674 ± 0.650348 (W 0.546889) | - | · | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.2668 ± 0.002468 (W 0.014966) | 2.26897 ± 0.0110155 (W 0.00505765) | 2.6753 ± 0.00322266 (W 0.00677363) | · | 🔵 colmat 1.18× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.27107 ± 0.0101582 (W 0.00765351) | 2.26531 ± 0.012301 (W 0.00517113) | 3.17894 ± 0.00348532 (W 0.00392791) | · | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 140.183 ± 0.258487 (W 0.144605) | 138.922 ± 0.329534 (W 0.144127) | 144.905 ± 0.276399 (W 0.224419) | · | · |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 140.198 ± 0.306365 (W 0.351583) | 138.978 ± 0.308164 (W 0.204522) | 167.902 ± 0.292398 (W 0.396362) | · | 🔵 colmat 1.20× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 4.28394 ± 0.00938455 (W 0.00407712) | 4.52294 ± 0.00370905 (W 0.00389798) | 4.94258 ± 0.0485342 (W 0.00480603) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 1.15× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 4.86145 ± 0.0114104 (W 0.00636543) | 5.13063 ± 0.00717953 (W 0.0292265) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 229.307 ± 0.438693 (W 0.567264) | 232.292 ± 0.219946 (W 0.508745) | 267.552 ± 0.433493 (W 0.836054) | · | 🔵 colmat 1.17× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 243.732 ± 0.305391 (W 0.529101) | 247.187 ± 0.30485 (W 0.500318) | - | · | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.99154 ± 0.00164714 (W 0.0016713) | 2.23997 ± 0.011737 (W 0.00371327) | 1.74974 ± 0.00773547 (W 0.0042025) | 🔵 colmat 1.12× faster than `faer` | 🔴 colmat 1.14× slower than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 87.4203 ± 0.0434999 (W 0.109853) | 90.5197 ± 0.309924 (W 0.225087) | 100.337 ± 0.44932 (W 0.425619) | · | 🔵 colmat 1.15× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 5.28634 ± 0.00980869 (W 0.00111573) | 5.69228 ± 0.00189987 (W 0.00617877) | 5.61702 ± 0.0318086 (W 0.00800681) | 🔵 colmat 1.08× faster than `faer` | 🔵 colmat 1.06× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.30685 ± 0.00936097 (W 0.00428274) | 5.72201 ± 0.008855 (W 0.0108797) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 260.89 ± 0.340989 (W 0.459015) | 268.07 ± 0.254929 (W 0.365104) | 307.94 ± 1.444 (W 0.790214) | · | 🔵 colmat 1.18× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 261.63 ± 0.413445 (W 0.508348) | 268.654 ± 0.170401 (W 0.46545) | - | · | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.36657 ± 0.00320443 (W 0.00469306) | 2.95367 ± 0.00459946 (W 0.00275261) | 2.27859 ± 0.00637048 (W 0.00201478) | 🔵 colmat 1.25× faster than `faer` | · |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.37012 ± 0.00392139 (W 0.00409984) | 2.95529 ± 0.00287168 (W 0.00310911) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 100.735 ± 0.36899 (W 0.129727) | 110.451 ± 0.112768 (W 0.112525) | 119.658 ± 0.42878 (W 0.506923) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 1.19× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 100.651 ± 0.367328 (W 0.388218) | 110.467 ± 0.243493 (W 0.364046) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.702948 ± 0.00829232 (W 0.00318972) | 1.54075 ± 0.0107363 (W 0.00406537) | - | 🔵 colmat 2.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.753502 ± 0.00210771 (W 0.00286357) | 1.52109 ± 0.00412565 (W 0.00444224) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 39.7184 ± 0.0361163 (W 0.099328) | 42.8339 ± 0.176371 (W 0.411109) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 39.135 ± 0.191965 (W 0.0407316) | 42.2656 ± 0.100433 (W 0.421392) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.652514 ± 0.00765515 (W 0.0024817) | 0.632086 ± 0.00096912 (W 0.00274467) | 0.90037 ± 0.00146225 (W 0.00403922) | · | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.65039 ± 0.0117424 (W 0.00144112) | 0.632479 ± 0.00142972 (W 0.00308353) | 1.30816 ± 0.0114566 (W 0.00490012) | · | 🔵 colmat 2.01× faster than `lapack-aocl` |  |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 39.2249 ± 0.0869322 (W 0.0818926) | 39.1805 ± 0.128412 (W 0.141188) | 39.5643 ± 0.127186 (W 0.0758597) | · | · |  |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 39.2831 ± 0.161297 (W 0.0621276) | 39.2619 ± 0.09742 (W 0.121551) | 54.2507 ± 0.130283 (W 0.0881858) | · | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 1.65454 ± 0.00769758 (W 0.00322527) | 2.2218 ± 0.00301328 (W 0.00605863) | 3.55945 ± 0.00241108 (W 0.004347) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 2.15× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.22317 ± 0.0190977 (W 0.00777825) | 2.88187 ± 0.0279385 (W 0.0261998) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 70.349 ± 0.143334 (W 0.117957) | 81.2384 ± 0.266154 (W 0.087518) | 98.426 ± 0.22969 (W 0.24547) | 🔵 colmat 1.15× faster than `faer` | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 84.4038 ± 0.165542 (W 0.16068) | 96.1143 ± 7.02544 (W 0.768652) | - | 🔵 colmat 1.14× faster than `faer` (marginal: spread 0.61 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 1.02873 ± 0.00461714 (W 0.00712498) | 1.59403 ± 0.00306043 (W 0.00669334) | 2.22812 ± 0.00596754 (W 0.00779529) | 🔵 colmat 1.55× faster than `faer` | 🔵 colmat 2.17× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 31.1718 ± 0.128161 (W 0.0294128) | 42.1184 ± 0.451495 (W 0.0704327) | 44.0342 ± 0.146431 (W 0.0877291) | 🔵 colmat 1.35× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 2.42178 ± 0.00969896 (W 0.00542832) | 3.37705 ± 0.00221726 (W 0.00926808) | 5.7047 ± 0.030201 (W 0.00729633) | 🔵 colmat 1.39× faster than `faer` | 🔵 colmat 2.36× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.45199 ± 0.0115339 (W 0.00531639) | 3.39197 ± 0.00598427 (W 0.00599913) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 84.9157 ± 0.00579969 (W 0.103324) | 105.081 ± 0.186974 (W 0.112354) | 133.554 ± 3.15082 (W 8.46496) | 🔵 colmat 1.24× faster than `faer` | 🔵 colmat 1.57× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 85.699 ± 0.301402 (W 0.313168) | 106.539 ± 0.546939 (W 0.102548) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.43241 ± 0.00719693 (W 0.00417497) | 2.35838 ± 0.00636293 (W 0.00547848) | 2.76818 ± 0.00893606 (W 0.00664407) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 1.93× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.42909 ± 0.000957319 (W 0.00820341) | 2.35519 ± 0.00930103 (W 0.00869466) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 39.1906 ± 0.180709 (W 0.0906569) | 59.5406 ± 0.0726858 (W 0.0566162) | 59.3288 ± 0.149036 (W 0.230753) | 🔵 colmat 1.52× faster than `faer` | 🔵 colmat 1.51× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 39.2173 ± 0.120668 (W 0.043246) | 59.8072 ± 0.148195 (W 0.0767396) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.588967 ± 0.0180288 (W 0.00392698) | 1.54047 ± 0.0137762 (W 0.00309703) | - | 🔵 colmat 2.62× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.663274 ± 0.00160786 (W 0.00342671) | 1.52964 ± 0.00746607 (W 0.00333385) | - | 🔵 colmat 2.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 20.2856 ± 0.150436 (W 0.0410797) | 23.8896 ± 0.0337032 (W 0.0739864) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 20.5346 ± 0.0470234 (W 0.0848497) | 24.0578 ± 0.0337043 (W 0.132917) | - | 🔵 colmat 1.17× faster than `faer` | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.376311 ± 0.00184796 (W 0.00272252) | 0.521501 ± 0.0454299 (W 0.0355563) | 0.680455 ± 0.00523319 (W 0.00418895) | 🔵 colmat 1.39× faster than `faer` | 🔵 colmat 1.81× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.379356 ± 0.00589191 (W 0.00288775) | 0.447689 ± 0.0439228 (W 0.0461018) | 0.945931 ± 0.00437092 (W 0.00643655) | 🔵 colmat 1.18× faster than `faer` (marginal: spread 0.73 of the difference) | 🔵 colmat 2.49× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 25.8147 ± 0.67894 (W 0.897215) | 26.4362 ± 0.0733847 (W 0.151661) | 22.8918 ± 0.0701209 (W 0.0377227) | · | 🔴 colmat 1.13× slower than `lapack-aocl` | `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 25.9555 ± 0.473121 (W 0.69644) | 26.3882 ± 0.0775052 (W 2.82641) | 34.8328 ± 0.138116 (W 0.0423279) | · | 🔵 colmat 1.34× faster than `lapack-aocl` |  |
| `qr/1024x64/b1` | 1.21136 ± 0.020947 (W 0.00449812) | 2.19721 ± 0.0211414 (W 0.0277795) | 3.36394 ± 0.0296796 (W 0.00641721) | 🔵 colmat 1.81× faster than `faer` | 🔵 colmat 2.78× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.79567 ± 0.00858391 (W 0.00526549) | 2.3423 ± 0.0492642 (W 0.102441) | - | 🔵 colmat 1.30× faster than `faer` | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 46.9838 ± 0.156785 (W 0.048384) | 69.4432 ± 0.816677 (W 0.882324) | 66.1807 ± 0.554827 (W 0.347359) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 1.41× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 60.2839 ± 0.254376 (W 0.0766422) | 71.5444 ± 0.566175 (W 2.34619) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.84876 ± 0.00524674 (W 0.00462762) | 1.63195 ± 0.00666984 (W 0.00727853) | 2.40818 ± 0.106634 (W 0.00385754) | 🔵 colmat 1.92× faster than `faer` | 🔵 colmat 2.84× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr_factor/4096x256/b1` | 21.9986 ± 0.0895407 (W 0.0233323) | 39.361 ± 0.251477 (W 0.230116) | 30.4471 ± 0.190171 (W 0.068188) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.98517 ± 0.013958 (W 0.00954743) | 3.34412 ± 0.0132971 (W 0.024738) | 5.94179 ± 0.0309475 (W 0.011387) | 🔵 colmat 1.68× faster than `faer` | 🔵 colmat 2.99× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.00467 ± 0.0184979 (W 0.00464057) | 3.4015 ± 0.174591 (W 0.144612) | - | 🔵 colmat 1.70× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1` | 59.6035 ± 0.215501 (W 0.144237) | 93.2728 ± 1.16079 (W 1.63975) | 136.196 ± 10.0254 (W 20.407) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 2.29× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 60.4613 ± 0.300251 (W 0.13864) | 93.6568 ± 0.34689 (W 2.93459) | - | 🔵 colmat 1.55× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.28452 ± 0.00370337 (W 0.00329997) | 2.38737 ± 0.00680473 (W 0.00957314) | 2.96504 ± 0.0119222 (W 0.0069533) | 🔵 colmat 1.86× faster than `faer` | 🔵 colmat 2.31× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.28513 ± 0.00877293 (W 0.00603014) | 2.41765 ± 0.0177997 (W 0.0115473) | - | 🔵 colmat 1.88× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 29.7465 ± 0.1427 (W 0.0470325) | 57.5222 ± 0.432573 (W 0.413359) | 47.3297 ± 0.175794 (W 1.75658) | 🔵 colmat 1.93× faster than `faer` | 🔵 colmat 1.59× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 29.73 ± 0.0747335 (W 0.0693097) | 57.4037 ± 0.30983 (W 0.395) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 44.5005 ± 0.180827 (W 0.139505) | 46.5839 ± 0.175928 (W 0.135873) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 51.7115 ± 0.702467 (W 0.695092) | 67.5827 ± 0.725803 (W 0.422337) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.620656 ± 0.00253383 (W 0.00348985) | 0.711275 ± 0.00449508 (W 0.00372196) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.3217 ± 0.00432838 (W 0.00673719) | 1.18352 ± 0.00695133 (W 0.00317153) | - | 🔴 colmat 1.12× slower than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 29.349 ± 0.399991 (W 0.0773917) | 28.6973 ± 0.069412 (W 0.0543325) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 30.4994 ± 0.0164935 (W 0.18519) | 32.6783 ± 0.63932 (W 0.601158) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.0281 ± 0.00507367 (W 0.00454441) | 1.03994 ± 0.00330085 (W 0.00539157) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.739769 ± 0.00651993 (W 0.00685996) | 0.836658 ± 0.0030172 (W 0.00319338) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.2042 ± 0.234415 (W 0.149298) | 27.3354 ± 0.102577 (W 0.288153) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.988729 ± 0.00784682 (W 0.00530968) | 1.00759 ± 0.00437231 (W 0.00507128) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 78.5 ± 0.401494 (W 0.353992) | 82.219 ± 0.302235 (W 0.0870301) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 78.1888 ± 0.36188 (W 0.413156) | 82.556 ± 0.490961 (W 0.387392) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.96516 ± 0.00184327 (W 0.00307264) | 2.27058 ± 0.0132315 (W 0.010222) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.96829 ± 0.00394612 (W 0.00439981) | 2.27525 ± 0.00424951 (W 0.00572783) | - | 🔵 colmat 1.16× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 32.3443 ± 0.1392 (W 0.0667543) | 38.7993 ± 0.100184 (W 0.0789052) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 32.2752 ± 0.0664765 (W 0.419218) | 38.8044 ± 0.0355491 (W 0.193343) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.952562 ± 0.00417201 (W 0.00228817) | 1.41113 ± 0.00375147 (W 0.00477849) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.954829 ± 0.00292555 (W 0.00408502) | 1.41022 ± 0.0060773 (W 0.00453201) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 11.9383 ± 0.0967891 (W 0.0167918) | 15.0018 ± 0.0640769 (W 0.0402284) | - | 🔵 colmat 1.26× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 15.0468 ± 0.0690139 (W 0.0574153) | 34.424 ± 0.26339 (W 0.172081) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.332028 ± 0.0035702 (W 0.00313953) | 0.716194 ± 0.0012142 (W 0.00327707) | - | 🔵 colmat 2.16× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.570022 ± 0.000843448 (W 0.00367618) | 1.18872 ± 0.00988214 (W 0.00471656) | - | 🔵 colmat 2.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.3838 ± 0.204966 (W 0.0388278) | 11.0457 ± 0.0797319 (W 0.0300406) | - | 🔵 colmat 1.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 10.097 ± 0.0977553 (W 0.0533612) | 12.6152 ± 0.0807771 (W 0.0271044) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.622733 ± 0.00640781 (W 0.00707086) | 0.713401 ± 0.00468465 (W 0.00564721) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.332739 ± 0.00272567 (W 0.00228141) | 0.534129 ± 0.00249 (W 0.0022234) | - | 🔵 colmat 1.61× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.0039 ± 0.0618112 (W 0.0348163) | 10.5244 ± 0.0420915 (W 0.0370743) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.602782 ± 0.00228068 (W 0.00540788) | 0.688504 ± 0.00604338 (W 0.00564379) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 31.5928 ± 0.0940509 (W 0.135536) | 41.5408 ± 0.0273477 (W 0.0383761) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 31.7216 ± 0.0876218 (W 0.074841) | 41.9415 ± 0.246826 (W 1.04872) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.25372 ± 0.00300085 (W 0.00297524) | 1.91844 ± 0.00605245 (W 0.00542244) | - | 🔵 colmat 1.53× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.27057 ± 0.00126649 (W 0.00502335) | 1.9222 ± 0.00473032 (W 0.00632592) | - | 🔵 colmat 1.51× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 16.822 ± 0.0416365 (W 0.398706) | 26.766 ± 0.0330156 (W 0.0139772) | - | 🔵 colmat 1.59× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 16.8327 ± 0.0218381 (W 0.0347162) | 26.9181 ± 0.0406094 (W 0.0201885) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.791344 ± 0.00224508 (W 0.0029476) | 1.47462 ± 0.00579357 (W 0.00615853) | - | 🔵 colmat 1.86× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.798047 ± 0.00640606 (W 0.00275593) | 1.47404 ± 0.00203479 (W 0.0050023) | - | 🔵 colmat 1.85× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 6.90948 ± 0.062118 (W 0.04844) | 10.2196 ± 0.0591443 (W 0.0208474) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 9.48031 ± 0.188583 (W 0.0413167) | 29.7931 ± 0.10866 (W 0.0714288) | - | 🔵 colmat 3.14× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.313584 ± 0.00146971 (W 0.002667) | 0.718339 ± 0.00414219 (W 0.00336018) | - | 🔵 colmat 2.29× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.481793 ± 0.00161241 (W 0.0027949) | 1.18647 ± 0.00492145 (W 0.00283464) | - | 🔵 colmat 2.46× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.3583 ± 0.0849171 (W 0.0448794) | 9.36909 ± 0.134396 (W 0.1536) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 6.70865 ± 0.0530745 (W 0.0343553) | 10.9935 ± 0.0519375 (W 0.249255) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.56338 ± 0.00219071 (W 0.0068681) | 0.769065 ± 0.0118258 (W 0.00570792) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.281064 ± 0.00756268 (W 0.00249018) | 0.598104 ± 0.0468471 (W 0.00522646) | - | 🔵 colmat 2.13× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 7.08589 ± 0.0839746 (W 0.0208049) | 8.86917 ± 0.110308 (W 0.154496) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.538243 ± 0.00393518 (W 0.00489854) | 0.736232 ± 0.00540964 (W 0.00930014) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 24.5097 ± 0.0783427 (W 0.0903359) | 40.4054 ± 0.172901 (W 0.194354) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 24.7909 ± 0.0559032 (W 0.0693698) | 40.7693 ± 1.2964 (W 2.19422) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.15874 ± 0.00743857 (W 0.00369249) | 2.01499 ± 0.00934836 (W 0.0125874) | - | 🔵 colmat 1.74× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.1754 ± 0.00747172 (W 0.00270619) | 2.02967 ± 0.0113195 (W 0.00362638) | - | 🔵 colmat 1.73× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 13.8404 ± 0.0504226 (W 0.0459222) | 27.7659 ± 0.0954053 (W 0.39567) | - | 🔵 colmat 2.01× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 13.8355 ± 0.0479966 (W 0.0382088) | 27.8821 ± 0.0744971 (W 0.0902629) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.753986 ± 0.00640498 (W 0.00282393) | 1.56127 ± 0.00304385 (W 0.00595439) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.753084 ± 0.00290831 (W 0.00330997) | 1.56106 ± 0.00634641 (W 0.00523076) | - | 🔵 colmat 2.07× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 59.742 ± 0.488071 (W 0.367436) | 68.7809 ± 0.779973 (W 0.478273) | - | 🔵 colmat 1.15× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 49.5678 ± 1.24678 (W 0.780938) | 61.4667 ± 0.656682 (W 0.485437) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.26543 ± 0.0014925 (W 0.00316801) | 1.03498 ± 0.00990323 (W 0.0025092) | - | 🔴 colmat 1.22× slower than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.23325 ± 0.00438429 (W 0.00390164) | 0.981336 ± 0.0104687 (W 0.00291855) | - | 🔴 colmat 1.26× slower than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.6419 ± 0.020043 (W 0.125772) | 26.6855 ± 0.22014 (W 0.0834614) | 87.7292 ± 0.275562 (W 0.151072) | · | 🔵 colmat 3.29× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.3821 ± 0.115451 (W 0.045219) | 27.7335 ± 0.0317613 (W 0.0978855) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.603804 ± 0.0024246 (W 0.00337075) | 0.604739 ± 0.00162731 (W 0.00282387) | 0.986034 ± 0.00712288 (W 0.00156122) | · | 🔵 colmat 1.63× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.626545 ± 0.00236641 (W 0.00405574) | 0.630112 ± 0.000852892 (W 0.0028625) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 25.1184 ± 0.197587 (W 0.131743) | 25.0208 ± 0.173712 (W 0.0465145) | 85.1163 ± 0.56884 (W 0.116974) | · | 🔵 colmat 3.39× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.566334 ± 0.00316914 (W 0.00355572) | 0.570823 ± 0.00376241 (W 0.00388163) | 0.921115 ± 0.00450477 (W 0.00321062) | · | 🔵 colmat 1.63× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 83.8908 ± 0.163389 (W 0.113641) | 87.7412 ± 0.262336 (W 0.432606) | 169.745 ± 1.03155 (W 0.357147) | · | 🔵 colmat 2.02× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 83.5058 ± 0.167231 (W 0.081238) | 87.9526 ± 0.446827 (W 0.0936937) | - | 🔵 colmat 1.05× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.05749 ± 0.00689583 (W 0.00657743) | 2.32754 ± 0.0101523 (W 0.00746633) | 4.47481 ± 0.0188925 (W 0.0145696) | 🔵 colmat 1.13× faster than `faer` | 🔵 colmat 2.17× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.07218 ± 0.0068036 (W 0.00672989) | 2.33209 ± 0.00974246 (W 0.0057897) | - | 🔵 colmat 1.13× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 37.909 ± 0.188238 (W 0.110868) | 44.6378 ± 0.191553 (W 0.315713) | 78.4524 ± 0.2337 (W 0.161471) | 🔵 colmat 1.18× faster than `faer` | 🔵 colmat 2.07× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 37.5875 ± 0.115689 (W 0.111235) | 44.5919 ± 0.127622 (W 0.170757) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.03455 ± 0.00323128 (W 0.00741114) | 1.47672 ± 0.00530044 (W 0.0081996) | 2.39698 ± 0.00733648 (W 0.011285) | 🔵 colmat 1.43× faster than `faer` | 🔵 colmat 2.32× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.03677 ± 0.00559894 (W 0.00812317) | 1.46998 ± 0.00806973 (W 0.00430428) | - | 🔵 colmat 1.42× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 16.5643 ± 0.0904826 (W 0.0372225) | 32.1059 ± 0.0272251 (W 0.0772665) | - | 🔵 colmat 1.94× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 12.2715 ± 0.104223 (W 0.0162254) | 29.8669 ± 0.0549425 (W 0.193015) | - | 🔵 colmat 2.43× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.508738 ± 0.00434526 (W 0.00290227) | 1.02727 ± 0.00262663 (W 0.00387803) | - | 🔵 colmat 2.02× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.471869 ± 0.00163871 (W 0.00359239) | 0.973424 ± 0.00750352 (W 0.00502916) | - | 🔵 colmat 2.06× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 6.99471 ± 0.0110672 (W 0.0134291) | 7.41618 ± 0.0825665 (W 0.0242466) | 25.5667 ± 0.127435 (W 0.0481002) | 🔵 colmat 1.06× faster than `faer` | 🔵 colmat 3.66× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 7.52298 ± 0.0585291 (W 0.0327527) | 8.07819 ± 0.0445321 (W 0.0164805) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.215812 ± 0.000705454 (W 0.0023137) | 0.285241 ± 0.000832947 (W 0.00169775) | 0.983732 ± 0.00212633 (W 0.00279324) | 🔵 colmat 1.32× faster than `faer` | 🔵 colmat 4.56× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.24276 ± 0.000756263 (W 0.00250296) | 0.323733 ± 0.00181811 (W 0.00293189) | - | 🔵 colmat 1.33× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.53176 ± 0.033066 (W 0.00773167) | 7.00323 ± 0.0360679 (W 0.0305785) | 24.9181 ± 0.204246 (W 0.0390131) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 3.81× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.197396 ± 0.000703358 (W 0.00224085) | 0.26418 ± 0.000982392 (W 0.00238686) | 0.940421 ± 0.00418853 (W 0.00164659) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 4.76× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 36.8848 ± 0.200147 (W 0.111854) | 46.9477 ± 1.11836 (W 0.138156) | 151.614 ± 1.34807 (W 1.12087) | 🔵 colmat 1.27× faster than `faer` | 🔵 colmat 4.11× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 37.2587 ± 0.227556 (W 0.155569) | 47.2421 ± 0.183484 (W 0.964246) | - | 🔵 colmat 1.27× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.33201 ± 0.00524269 (W 0.00443863) | 1.96988 ± 0.0144087 (W 0.0100717) | 4.88529 ± 0.0457157 (W 0.0712092) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 3.67× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.35131 ± 0.00308596 (W 0.00586876) | 1.99261 ± 0.011844 (W 0.00649505) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 22.1723 ± 0.0115365 (W 0.0510334) | 32.5863 ± 0.111228 (W 0.114055) | 68.2421 ± 0.350629 (W 10.7364) | 🔵 colmat 1.47× faster than `faer` | 🔵 colmat 3.08× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 22.2269 ± 0.0414395 (W 0.0453979) | 32.7784 ± 0.0401374 (W 0.0402775) | - | 🔵 colmat 1.47× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.857547 ± 0.00632093 (W 0.00732045) | 1.53726 ± 0.0037954 (W 0.0118301) | 2.39674 ± 0.00814752 (W 0.0120257) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 2.79× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.866975 ± 0.00701716 (W 0.00654496) | 1.52891 ± 0.00387213 (W 0.0107679) | - | 🔵 colmat 1.76× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 8.68409 ± 0.0848637 (W 0.0152028) | 26.0736 ± 0.163727 (W 0.0226806) | - | 🔵 colmat 3.00× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 6.79698 ± 0.04232 (W 0.0290872) | 24.8279 ± 0.097249 (W 0.0408546) | - | 🔵 colmat 3.65× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.390027 ± 0.000605683 (W 0.00252592) | 1.02535 ± 0.0131281 (W 0.00374646) | - | 🔵 colmat 2.63× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.386981 ± 0.00609159 (W 0.00345321) | 0.971802 ± 0.00992663 (W 0.00317649) | - | 🔵 colmat 2.51× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 3.79908 ± 0.0158332 (W 0.00891597) | 5.85665 ± 0.0594545 (W 0.170759) | 16.7986 ± 0.114572 (W 0.139744) | 🔵 colmat 1.54× faster than `faer` | 🔵 colmat 4.42× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 4.25389 ± 0.0928712 (W 0.0256166) | 6.45329 ± 0.0182597 (W 0.226393) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.16119 ± 0.0142397 (W 0.00107313) | 0.346035 ± 0.0123856 (W 0.00370815) | 0.989024 ± 0.00314028 (W 0.00198925) | 🔵 colmat 2.15× faster than `faer` | 🔵 colmat 6.14× faster than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 0.190573 ± 0.000757967 (W 0.00313811) | 0.379603 ± 0.000999924 (W 0.00509471) | - | 🔵 colmat 1.99× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.51461 ± 0.0328003 (W 0.00606678) | 5.5003 ± 0.0401908 (W 0.157243) | 16.2857 ± 0.0504303 (W 0.136622) | 🔵 colmat 1.56× faster than `faer` | 🔵 colmat 4.63× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.146203 ± 0.00116251 (W 0.00217818) | 0.319692 ± 0.000769944 (W 0.00414533) | 0.949158 ± 0.00192898 (W 0.00412169) | 🔵 colmat 2.19× faster than `faer` | 🔵 colmat 6.49× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 29.8954 ± 0.0655633 (W 0.0908156) | 45.7377 ± 0.144444 (W 0.122216) | 131.74 ± 0.713424 (W 5.11396) | 🔵 colmat 1.53× faster than `faer` | 🔵 colmat 4.41× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 30.3136 ± 0.0507277 (W 0.0756656) | 46.0519 ± 0.0546092 (W 0.425009) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.24532 ± 0.0106061 (W 0.00643635) | 2.05869 ± 0.00918941 (W 0.0104816) | 4.75578 ± 0.0172835 (W 0.0271226) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 3.82× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.25774 ± 0.00792766 (W 0.00646938) | 2.07009 ± 0.00473493 (W 0.00728925) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 19.2228 ± 0.0320321 (W 0.0376277) | 33.4927 ± 0.0496575 (W 0.116195) | 70.5992 ± 0.603962 (W 2.6327) | 🔵 colmat 1.74× faster than `faer` | 🔵 colmat 3.67× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.2243 ± 0.0634022 (W 0.0313076) | 33.5529 ± 0.0788522 (W 0.106268) | - | 🔵 colmat 1.75× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.819708 ± 0.00479084 (W 0.00577716) | 1.6126 ± 0.0130616 (W 0.0100603) | 2.38144 ± 0.00112976 (W 0.00929216) | 🔵 colmat 1.97× faster than `faer` | 🔵 colmat 2.91× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.823906 ± 0.00191213 (W 0.00564348) | 1.61326 ± 0.00806822 (W 0.00730919) | - | 🔵 colmat 1.96× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 124.002 ± 0.59961 (W 0.324644) | 124.451 ± 1.32221 (W 0.321462) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 146.691 ± 0.720968 (W 0.657076) | 174.128 ± 0.0147853 (W 0.372003) | - | 🔵 colmat 1.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.38628 ± 0.0028175 (W 0.00447804) | 1.48517 ± 0.0168005 (W 0.00295926) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.16926 ± 0.0171785 (W 0.00548361) | 2.39225 ± 0.0267963 (W 0.0187366) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 104.043 ± 0.310756 (W 0.211459) | 104.78 ± 0.350429 (W 0.381552) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 108.797 ± 0.641597 (W 0.321613) | 108.006 ± 0.63623 (W 0.71994) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.76296 ± 0.0125058 (W 0.00658668) | 2.7582 ± 0.00905454 (W 0.00572029) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 2.57046 ± 0.00898614 (W 0.00582255) | 2.63023 ± 0.0296522 (W 0.00578011) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 97.7779 ± 0.409473 (W 0.438374) | 98.8132 ± 0.300823 (W 0.496365) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 2.61141 ± 0.014222 (W 0.0106828) | 2.63605 ± 0.0145508 (W 0.00306111) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 261.644 ± 0.722661 (W 0.495421) | 269.143 ± 1.01933 (W 0.749968) | - | · | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 261.829 ± 0.347733 (W 0.675948) | 270.055 ± 0.308096 (W 0.421796) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.31106 ± 0.0258185 (W 0.00742414) | 5.71106 ± 0.0119942 (W 0.00918664) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.3358 ± 0.0119332 (W 0.008917) | 5.74495 ± 0.00926474 (W 0.00900056) | - | 🔵 colmat 1.08× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 100.626 ± 0.216724 (W 0.353581) | 110.194 ± 0.361012 (W 0.119419) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 100.63 ± 0.623463 (W 0.429986) | 110.437 ± 0.333344 (W 0.121298) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.35627 ± 0.0122612 (W 0.00281932) | 2.94431 ± 0.00879709 (W 0.00455474) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.35184 ± 0.00599701 (W 0.00321497) | 2.94426 ± 0.0110911 (W 0.00532234) | - | 🔵 colmat 1.25× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 37.6321 ± 0.151289 (W 0.0788561) | 41.7066 ± 0.22348 (W 0.188014) | - | 🔵 colmat 1.11× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 48.7045 ± 0.067385 (W 0.220846) | 79.6541 ± 0.375581 (W 0.29322) | - | 🔵 colmat 1.64× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.569722 ± 0.00696002 (W 0.00374422) | 1.46437 ± 0.013961 (W 0.00624178) | - | 🔵 colmat 2.57× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.819056 ± 0.00710552 (W 0.00354586) | 2.39823 ± 0.0385765 (W 0.0149952) | - | 🔵 colmat 2.93× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 29.3042 ± 0.164484 (W 0.0380487) | 30.105 ± 0.159336 (W 0.0301835) | - | · | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 32.3109 ± 0.0649606 (W 0.0786565) | 33.0386 ± 0.122536 (W 0.0789098) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.03191 ± 0.00468175 (W 0.00287008) | 1.15371 ± 0.00782997 (W 0.006548) | - | 🔵 colmat 1.12× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.880941 ± 0.00586403 (W 0.00818276) | 1.04287 ± 0.00899643 (W 0.00413293) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.6447 ± 0.15765 (W 0.0435817) | 28.6291 ± 0.0459165 (W 0.0249579) | - | · | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.976149 ± 0.00178016 (W 0.00428476) | 1.07528 ± 0.00716732 (W 0.00880793) | - | 🔵 colmat 1.10× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 86.0524 ± 0.280235 (W 0.352659) | 105.929 ± 0.366969 (W 0.155762) | - | 🔵 colmat 1.23× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 86.5088 ± 0.441263 (W 0.0728074) | 106.853 ± 0.0829219 (W 0.0881981) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.46421 ± 0.0155434 (W 0.00392757) | 3.40381 ± 0.0174376 (W 0.00516158) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.49291 ± 0.00998682 (W 0.00736529) | 3.4303 ± 0.0171901 (W 0.00756149) | - | 🔵 colmat 1.38× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 39.332 ± 0.129428 (W 0.113769) | 59.635 ± 0.0692369 (W 0.0401405) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 39.3286 ± 0.105004 (W 0.0448081) | 59.6445 ± 0.138899 (W 0.049787) | - | 🔵 colmat 1.52× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.421 ± 0.00524542 (W 0.00366566) | 2.34151 ± 0.00488689 (W 0.00618799) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.42411 ± 0.0063707 (W 0.0033573) | 2.34355 ± 0.00478826 (W 0.00541424) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 18.0051 ± 0.0410023 (W 0.0342574) | 23.0967 ± 0.198412 (W 0.0666464) | - | 🔵 colmat 1.28× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 23.9254 ± 0.0938539 (W 0.126838) | 55.1447 ± 0.289486 (W 0.731489) | - | 🔵 colmat 2.30× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.460143 ± 0.0074918 (W 0.00338619) | 1.46676 ± 0.00661954 (W 0.00490885) | - | 🔵 colmat 3.19× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.642538 ± 0.0118029 (W 0.00371984) | 2.34614 ± 0.0454884 (W 0.00997945) | - | 🔵 colmat 3.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 16.9957 ± 0.075785 (W 0.0389507) | 22.3975 ± 0.524369 (W 0.724019) | - | 🔵 colmat 1.32× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 19.7739 ± 0.0729231 (W 0.124852) | 25.0801 ± 0.147357 (W 3.49508) | - | 🔵 colmat 1.27× faster than `faer` (marginal: spread 0.66 of the difference) | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.738588 ± 0.00326672 (W 0.00492887) | 1.10526 ± 0.010146 (W 0.0117479) | - | 🔵 colmat 1.50× faster than `faer` | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.61152 ± 0.00162983 (W 0.00310427) | 1.00599 ± 0.0198619 (W 0.0135642) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 15.9413 ± 0.0936784 (W 0.0339446) | 20.8763 ± 0.35343 (W 0.563945) | - | 🔵 colmat 1.31× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.708992 ± 0.00148119 (W 0.00393949) | 1.03279 ± 0.00632387 (W 0.0172129) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 60.1929 ± 0.189392 (W 0.058786) | 94.4607 ± 1.06782 (W 2.19749) | - | 🔵 colmat 1.57× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 61.0846 ± 0.162997 (W 0.384034) | 93.9446 ± 0.100457 (W 1.0947) | - | 🔵 colmat 1.54× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.01754 ± 0.0166643 (W 0.0035558) | 3.40248 ± 0.0251965 (W 0.0143689) | - | 🔵 colmat 1.69× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.04554 ± 0.00306355 (W 0.00333835) | 3.42428 ± 0.0342648 (W 0.0446588) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 29.7121 ± 0.146271 (W 0.0758117) | 57.3824 ± 0.0578254 (W 0.464659) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 29.7513 ± 0.167849 (W 0.0495263) | 57.4084 ± 0.0333818 (W 0.318761) | - | 🔵 colmat 1.93× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.27262 ± 0.00756944 (W 0.00371585) | 2.37369 ± 0.0216639 (W 0.0116443) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.27957 ± 0.00799527 (W 0.00430464) | 2.39167 ± 0.00642555 (W 0.00815152) | - | 🔵 colmat 1.87× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 147.536 ± 0.799225 (W 1.05561) | 176.525 ± 0.937622 (W 0.229956) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 142.993 ± 3.97828 (W 0.260615) | 169.223 ± 0.243818 (W 0.840345) | - | 🔵 colmat 1.18× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 2.08041 ± 0.00130497 (W 0.00193439) | 2.26833 ± 0.00428697 (W 0.00800391) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.03568 ± 0.0129376 (W 0.00340413) | 2.21195 ± 0.0263211 (W 0.00535619) | - | 🔵 colmat 1.09× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 102.381 ± 0.283599 (W 0.53585) | 102.814 ± 0.686287 (W 0.583705) | 170.861 ± 1.09107 (W 0.696021) | · | 🔵 colmat 1.67× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 104.921 ± 0.328306 (W 0.736199) | 103.834 ± 0.867718 (W 0.583641) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.3983 ± 0.00166092 (W 0.00440458) | 2.38116 ± 0.00850844 (W 0.00443073) | 3.34677 ± 0.0423482 (W 0.00441603) | · | 🔵 colmat 1.40× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 2.43807 ± 0.0112964 (W 0.005007) | 2.41733 ± 0.010155 (W 0.00548984) | - | · | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 96.6162 ± 0.6368 (W 0.329561) | 97.5197 ± 1.03004 (W 0.87764) | 158.473 ± 1.11257 (W 0.778467) | · | 🔵 colmat 1.64× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 2.2485 ± 0.0110281 (W 0.00914237) | 2.23373 ± 0.00745282 (W 0.00674489) | 3.10299 ± 0.0251413 (W 0.00398092) | · | 🔵 colmat 1.38× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 274.015 ± 0.181516 (W 0.424162) | 280.951 ± 0.507646 (W 0.347508) | 571.744 ± 0.76009 (W 0.908798) | · | 🔵 colmat 2.09× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 274.119 ± 0.438575 (W 0.663268) | 281.686 ± 0.421362 (W 0.350868) | - | · | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.44637 ± 0.0139958 (W 0.00852277) | 5.82841 ± 0.00979655 (W 0.00533579) | 9.73584 ± 0.0895812 (W 0.0171631) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 1.79× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.46561 ± 0.0183466 (W 0.00314824) | 5.84409 ± 0.0300218 (W 0.0216469) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 114.554 ± 1.37457 (W 0.385408) | 122.986 ± 0.873287 (W 0.152815) | 253.658 ± 0.47322 (W 0.974766) | 🔵 colmat 1.07× faster than `faer` | 🔵 colmat 2.21× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 114.809 ± 0.125566 (W 0.478686) | 122.322 ± 0.616089 (W 0.398037) | - | 🔵 colmat 1.07× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.48938 ± 0.00643695 (W 0.0078254) | 3.07136 ± 0.0140567 (W 0.00606456) | 4.10763 ± 0.113634 (W 0.00771366) | 🔵 colmat 1.23× faster than `faer` | 🔵 colmat 1.65× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.48765 ± 0.00930608 (W 0.00410067) | 3.07592 ± 0.0168365 (W 0.00366901) | - | 🔵 colmat 1.24× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 52.4272 ± 0.247599 (W 0.329557) | 77.5101 ± 0.386131 (W 0.253658) | - | 🔵 colmat 1.48× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 44.4807 ± 0.0787907 (W 0.172337) | 74.4211 ± 0.171744 (W 0.336381) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.752823 ± 0.00114897 (W 0.00394131) | 2.2566 ± 0.0281249 (W 0.011561) | - | 🔵 colmat 3.00× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.68691 ± 0.00403062 (W 0.00273689) | 2.19666 ± 0.0427256 (W 0.0128959) | - | 🔵 colmat 3.20× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.2868 ± 0.149082 (W 0.039698) | 26.9697 ± 0.135486 (W 0.0516602) | 71.0716 ± 0.853928 (W 0.164264) | · | 🔵 colmat 2.70× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.8387 ± 0.106237 (W 0.0350956) | 28.3413 ± 0.251323 (W 0.185192) | - | · | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.70821 ± 0.00183276 (W 0.00288048) | 0.806873 ± 0.00598712 (W 0.00246336) | 3.66727 ± 0.0127724 (W 0.0092541) | 🔵 colmat 1.14× faster than `faer` | 🔵 colmat 5.18× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.765135 ± 0.00486684 (W 0.00408024) | 0.871424 ± 0.00494403 (W 0.00198552) | - | 🔵 colmat 1.14× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 24.7869 ± 0.121452 (W 0.263466) | 25.3262 ± 0.105202 (W 0.0566676) | 68.6196 ± 0.183501 (W 0.426323) | · | 🔵 colmat 2.77× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.660421 ± 0.00373218 (W 0.00229515) | 0.725603 ± 0.0030333 (W 0.00226966) | 3.53798 ± 0.0499441 (W 0.00601489) | 🔵 colmat 1.10× faster than `faer` | 🔵 colmat 5.36× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 98.0602 ± 0.154729 (W 0.112392) | 117.929 ± 0.142853 (W 0.220586) | 360.633 ± 0.613768 (W 0.956959) | 🔵 colmat 1.20× faster than `faer` | 🔵 colmat 3.68× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 98.8609 ± 0.23057 (W 0.0850425) | 118.935 ± 0.282307 (W 2.01041) | - | 🔵 colmat 1.20× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.63052 ± 0.0370898 (W 0.0141236) | 3.5165 ± 0.00731605 (W 0.0142512) | 9.78369 ± 0.141098 (W 0.0376845) | 🔵 colmat 1.34× faster than `faer` | 🔵 colmat 3.72× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.59514 ± 0.0403269 (W 0.00259002) | 3.54426 ± 0.0147306 (W 0.00619252) | - | 🔵 colmat 1.37× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 51.3826 ± 0.0563401 (W 0.0750676) | 71.9787 ± 0.285516 (W 0.081754) | 180.009 ± 0.579292 (W 0.56695) | 🔵 colmat 1.40× faster than `faer` | 🔵 colmat 3.50× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 51.3327 ± 0.059285 (W 0.0741987) | 71.7809 ± 0.0451243 (W 0.056984) | - | 🔵 colmat 1.40× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.52847 ± 0.0128739 (W 0.00374127) | 2.45929 ± 0.0268415 (W 0.00813529) | 4.81233 ± 0.00903048 (W 0.00844597) | 🔵 colmat 1.61× faster than `faer` | 🔵 colmat 3.15× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.533 ± 0.00622091 (W 0.00571252) | 2.45146 ± 0.0278001 (W 0.00893712) | - | 🔵 colmat 1.60× faster than `faer` | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | colmat (ms) | faer (ms) | lapack-aocl (ms) | vs `faer` | vs `lapack-aocl` | notes |
|---|---|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 22.4287 ± 0.146453 (W 0.235481) | 52.1036 ± 0.21002 (W 0.306385) | - | 🔵 colmat 2.32× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 19.7389 ± 0.100929 (W 0.0901458) | 50.2734 ± 0.283871 (W 0.654415) | - | 🔵 colmat 2.55× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.507747 ± 0.00260249 (W 0.00319273) | 2.2528 ± 0.0452278 (W 0.0194595) | - | 🔵 colmat 4.44× faster than `faer` | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.502176 ± 0.00235117 (W 0.00400918) | 2.19003 ± 0.0215698 (W 0.00748943) | - | 🔵 colmat 4.36× faster than `faer` | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 14.0026 ± 0.0130252 (W 0.0234372) | 19.2621 ± 0.655994 (W 1.08553) | 54.787 ± 0.0832158 (W 0.337669) | 🔵 colmat 1.38× faster than `faer` | 🔵 colmat 3.91× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 15.2788 ± 0.0306028 (W 0.0336976) | 21.0307 ± 0.607657 (W 3.56888) | - | 🔵 colmat 1.38× faster than `faer` (marginal: spread 0.62 of the difference) | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.430142 ± 0.00202333 (W 0.00289418) | 0.770346 ± 0.00929561 (W 0.0207948) | 4.01918 ± 0.0422291 (W 0.0126262) | 🔵 colmat 1.79× faster than `faer` | 🔵 colmat 9.34× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.496285 ± 0.00465757 (W 0.00386663) | 0.834476 ± 0.0147396 (W 0.0165825) | - | 🔵 colmat 1.68× faster than `faer` | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.067 ± 0.0180063 (W 0.0189297) | 16.8996 ± 0.0417081 (W 0.911806) | 53.8293 ± 0.24206 (W 0.23179) | 🔵 colmat 1.29× faster than `faer` | 🔵 colmat 4.12× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.394746 ± 0.00238866 (W 0.00308463) | 0.678287 ± 0.0155752 (W 0.0189269) | 3.8644 ± 0.0361147 (W 0.00567036) | 🔵 colmat 1.72× faster than `faer` | 🔵 colmat 9.79× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 72.4918 ± 0.161759 (W 0.207521) | 107.019 ± 1.14669 (W 1.20442) | 353.566 ± 10.0775 (W 18.5156) | 🔵 colmat 1.48× faster than `faer` | 🔵 colmat 4.88× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 73.4711 ± 0.100991 (W 0.163645) | 107.206 ± 0.573039 (W 5.76217) | - | 🔵 colmat 1.46× faster than `faer` | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.12444 ± 0.00889442 (W 0.00942758) | 3.51224 ± 0.0124131 (W 0.0221764) | 10.129 ± 0.0406837 (W 0.0217177) | 🔵 colmat 1.65× faster than `faer` | 🔵 colmat 4.77× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.14475 ± 0.0177644 (W 0.0060379) | 3.54761 ± 0.0189462 (W 0.0279191) | - | 🔵 colmat 1.65× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 41.7473 ± 0.0499871 (W 0.0734364) | 69.3332 ± 0.43164 (W 0.388024) | 191.24 ± 7.29562 (W 8.17766) | 🔵 colmat 1.66× faster than `faer` | 🔵 colmat 4.58× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 41.6895 ± 0.0401176 (W 0.0759635) | 69.6728 ± 0.48767 (W 0.485392) | - | 🔵 colmat 1.67× faster than `faer` | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.3725 ± 0.0271857 (W 0.00429507) | 2.48319 ± 0.0242964 (W 0.00948347) | 5.21669 ± 0.0254529 (W 0.0121061) | 🔵 colmat 1.81× faster than `faer` | 🔵 colmat 3.80× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.39475 ± 0.0464104 (W 0.014356) | 2.47813 ± 0.0211257 (W 0.00602316) | - | 🔵 colmat 1.78× faster than `faer` | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
