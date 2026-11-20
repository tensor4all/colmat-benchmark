# `faer-kernels` on `zen5-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `6dcefead48b6c1a27200439dd38b12ea4e8a2275`
- toolkit commit: `ec1366c391cbd5023116ddcb6fdadc9c5b76f582`
- measured binary SHA-256: `2887763459bbf4590f96f52994d1fbf0a393ccd73cfcb30e73a111b3579cc0e2`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-19T20:40:33.832619Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu faer-kernels --vendor aocl --engines faer-fork,lapack-aocl --threads 1,4,8 --reps 5 --aa 6 --prime-ms 250 --label pub-93da6217-20261119-aocl`
- raw data: `data/results/zen5-cpu/faer-kernels/20261119T161532Z-pub-93da6217-20261119-aocl/`

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
  - `lapack-aocl` — The same calls against AMD AOCL, the vendor-native LAPACK stack on AMD hosts: LAPACKE from AOCL-LAPACK (`libflame`), CBLAS from the threaded AOCL-BLAS (`libblis-mt`), linked from the versioned tree `AOCL_ROOT` names and loaded through the rpath the build records. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run. Measured on AMD profiles only.
    Identity: the AOCL build recorded in this run's manifest (`providers[aocl]`)
    Caveat: One AOCL call is governed by three per-thread counts (OpenMP, BLIS, libflame); the harness sets the budget through OpenMP and refuses the cell unless all three read back equal to it, before each row and after its timed loop. The manifest records both libraries' versions, the files the loader mapped, and the dispatched kernels (`blis=<config>, flame=<isa>`). Threading is vendor-owned, as for OpenBLAS and MKL.
    As measured: linked shared; aocl 6.0.0; kernel dispatched to `blis=zen5, flame=avx512dq`; openmp threading; config `AOCL-BLAS 6.0.0 Build 20261026; AOCL-LAPACK 6.0.0 Build 20261026, supports LAPACK 3.12.1 w/o Index-64 interfaces; mapped=/home/shinaoka/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libaoclutils.so:/home/shinaoka/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libblis-mt.so.6.0.0:/home/shinaoka/opt/AMD/aocl/6.0.0/gcc/MT/lib_LP64/libflame.so:/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0`

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
Spread guide (computed from this run's own rows): per-engine between-start range median 1.29%, worst 8220.91% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 288 of 5064 engine cells exceed 5% and carry `!`; median within-process relative SD 0.40%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002354 ± 3.30479e-05 (W 1.9635e-05) | 0.002304 ± 1.502e-05 (W 3.84163e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.017117 ± 6.53197e-05 (W 3.04329e-05) | 0.016927 ± 9.76354e-05 (W 4.79269e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0566855 ± 0.000266487 (W 0.00129771) | 0.0564355 ± 0.000255926 (W 0.000120287) | · |  |

## gemm-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002385 ± 2.21721e-05 (W 2.66692e-05) | 0.002294 ± 1.13798e-05 (W 3.63497e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.01116 ± 0.000144362 (W 0.000184406) | 0.006131 ± 5.93647e-05 (W 7.04746e-05) | 🔴 faer-fork 1.82× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.023404 ± 0.000260186 (W 0.000259749) | 0.01615 ± 5.78504e-05 (W 9.28037e-05) | 🔴 faer-fork 1.45× slower than `lapack-aocl` |  |

## gemm-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002395 ± 1.62573e-05 (W 3.44543e-05) | 0.0022945 ± 3.77028e-05 (W 5.80533e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.009638 ± 4.86227e-05 (W 8.66757e-05) | 0.0043535 ± 6.06322e-05 (W 9.73464e-05) | 🔴 faer-fork 2.21× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0200175 ± 5.11103e-05 (W 7.90921e-05) | 0.0095575 ± 4.34557e-05 (W 0.00016159) | 🔴 faer-fork 2.09× slower than `lapack-aocl` |  |

## gemm-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.002339 ± 1.36626e-05 (W 1.61008e-05) | 0.002309 ± 1.42688e-05 (W 3.81808e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0171815 ± 1.42092e-05 (W 7.62899e-05) | 0.0169065 ± 0.000119351 (W 7.31189e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.056762 ± 0.000416745 (W 6.54249e-05) | 0.0564645 ± 0.00029191 (W 0.000517325) | · |  |

## gemm-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0023845 ± 2.53693e-05 (W 2.99969e-05) | 0.002294 ± 1.74404e-05 (W 3.69517e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.011041 ± 0.000174975 (W 0.00018887) | 0.006192 ± 0.000107202 (W 0.000114893) | 🔴 faer-fork 1.78× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0232985 ± 0.000323731 (W 0.000412393) | 0.016125 ± 6.82752e-05 (W 0.000119333) | 🔴 faer-fork 1.44× slower than `lapack-aocl` |  |

## gemm-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0023845 ± 2.43365e-05 (W 6.37095e-05) | 0.0022895 ± 1.73666e-05 (W 3.93077e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.009653 ± 5.56477e-05 (W 9.51105e-05) | 0.004423 ± 6.99154e-05 (W 0.000122247) | 🔴 faer-fork 2.18× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.020002 ± 0.000482081 (W 8.57542e-05) | 0.0095725 ± 8.80396e-05 (W 0.000137228) | 🔴 faer-fork 2.09× slower than `lapack-aocl` | `!` between-start range over 5% |

## gemm-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.008812 ± 4.30302e-05 (W 3.5632e-05) | 0.008631 ± 3.24987e-05 (W 4.71004e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0676675 ± 0.000278846 (W 0.000145568) | 0.0664905 ± 0.000259932 (W 6.39747e-05) | · |  |
| `gemm/96x96x96/b1` | 0.226726 ± 0.000801349 (W 0.00123355) | 0.223758 ± 0.000267375 (W 0.0014591) | · |  |

## gemm-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0095425 ± 0.000249808 (W 0.000171163) | 0.0040525 ± 3.46737e-05 (W 8.39966e-05) | 🔴 faer-fork 2.35× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.028758 ± 0.000180544 (W 0.000248803) | 0.0187355 ± 4.91203e-05 (W 0.000112411) | 🔴 faer-fork 1.53× slower than `lapack-aocl` |  |
| `gemm/96x96x96/b1` | 0.077851 ± 0.000502891 (W 0.000203477) | 0.058589 ± 0.000220648 (W 8.80415e-05) | 🔴 faer-fork 1.33× slower than `lapack-aocl` |  |

## gemm-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009633 ± 6.47572e-05 (W 0.000110325) | 0.003241 ± 0.000115851 (W 9.09123e-05) | 🔴 faer-fork 2.97× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0265395 ± 8.53581e-05 (W 7.62605e-05) | 0.011697 ± 5.4191e-05 (W 0.000163827) | 🔴 faer-fork 2.27× slower than `lapack-aocl` |  |
| `gemm/96x96x96/b1` | 0.0395135 ± 0.000269684 (W 0.0001107) | 0.031033 ± 0.000104951 (W 0.000108309) | 🔴 faer-fork 1.27× slower than `lapack-aocl` |  |

## gemm-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0088415 ± 5.05467e-05 (W 3.38744e-05) | 0.0086415 ± 3.66083e-05 (W 3.89436e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0677015 ± 0.000295777 (W 0.00076115) | 0.0667065 ± 0.00017812 (W 7.24244e-05) | · |  |
| `gemm/96x96x96/b1` | 0.226565 ± 0.00109425 (W 0.00122752) | 0.223773 ± 0.000698683 (W 0.00157772) | · |  |

## gemm-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0096735 ± 0.000224028 (W 0.000144721) | 0.004028 ± 4.3615e-05 (W 8.03644e-05) | 🔴 faer-fork 2.40× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.028758 ± 0.000169989 (W 0.000177489) | 0.0186545 ± 5.91777e-05 (W 0.00010011) | 🔴 faer-fork 1.54× slower than `lapack-aocl` |  |
| `gemm/96x96x96/b1` | 0.0784225 ± 0.000463033 (W 0.00114891) | 0.0585235 ± 0.0001788 (W 0.000114025) | 🔴 faer-fork 1.34× slower than `lapack-aocl` |  |

## gemm-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.009688 ± 7.56053e-05 (W 9.92327e-05) | 0.003226 ± 8.86125e-05 (W 7.58466e-05) | 🔴 faer-fork 3.00× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.026574 ± 3.64234e-05 (W 7.08737e-05) | 0.011817 ± 0.000199913 (W 0.000116129) | 🔴 faer-fork 2.25× slower than `lapack-aocl` |  |
| `gemm/96x96x96/b1` | 0.0395585 ± 0.000149722 (W 0.000110445) | 0.0310435 ± 8.78668e-05 (W 0.000143073) | 🔴 faer-fork 1.27× slower than `lapack-aocl` |  |

## gemm-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0012575 ± 1.14134e-05 (W 2.40328e-05) | 0.001272 ± 5.16398e-06 (W 3.87698e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.008621 ± 5.17832e-05 (W 6.56331e-05) | 0.008556 ± 5.97152e-05 (W 4.2434e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0283135 ± 0.000112051 (W 3.21574e-05) | 0.0288935 ± 0.00020858 (W 7.03726e-05) | · |  |

## gemm-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0013375 ± 2.22171e-05 (W 3.29735e-05) | 0.0024145 ± 6.05665e-05 (W 8.60524e-05) | 🔵 faer-fork 1.81× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0062015 ± 7.43974e-05 (W 0.000139686) | 0.0062165 ± 5.3282e-05 (W 9.8451e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0129095 ± 0.000140105 (W 0.000254804) | 0.0112865 ± 5.03736e-05 (W 7.85973e-05) | 🔴 faer-fork 1.14× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0013525 ± 1.31719e-05 (W 6.97832e-05) | 0.0023695 ± 4.17489e-05 (W 9.89634e-05) | 🔵 faer-fork 1.75× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.005375 ± 6.47302e-05 (W 6.01845e-05) | 0.004488 ± 0.00011289 (W 0.000138563) | 🔴 faer-fork 1.20× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.01074 ± 6.91857e-05 (W 8.73087e-05) | 0.006993 ± 7.55645e-05 (W 0.000149357) | 🔴 faer-fork 1.54× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001267 ± 1.01784e-05 (W 2.14604e-05) | 0.0012625 ± 1.49321e-05 (W 3.13616e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.008576 ± 3.50238e-05 (W 3.29839e-05) | 0.0085255 ± 3.21807e-05 (W 4.04252e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0284285 ± 0.000135916 (W 3.97616e-05) | 0.029029 ± 0.000152207 (W 7.47035e-05) | · |  |

## gemm-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.001318 ± 1.21765e-05 (W 3.53216e-05) | 0.002395 ± 5.89262e-05 (W 5.65643e-05) | 🔵 faer-fork 1.82× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0062615 ± 7.81554e-05 (W 0.00014506) | 0.0061715 ± 4.58581e-05 (W 0.000102079) | · | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0130595 ± 0.000203052 (W 0.000178258) | 0.011311 ± 7.7521e-05 (W 9.99193e-05) | 🔴 faer-fork 1.15× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## gemm-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0013325 ± 1.63788e-05 (W 2.71524e-05) | 0.002505 ± 5.95953e-05 (W 0.000108535) | 🔵 faer-fork 1.88× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.00539 ± 7.51277e-05 (W 7.56852e-05) | 0.004469 ± 8.28589e-05 (W 0.000124053) | 🔴 faer-fork 1.21× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.010755 ± 6.34185e-05 (W 9.54751e-05) | 0.006893 ± 0.000109478 (W 0.000159745) | 🔴 faer-fork 1.56× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |

## gemm-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.004498 ± 2.28094e-05 (W 3.75853e-05) | 0.004373 ± 1.64975e-05 (W 4.13017e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.033758 ± 0.000148973 (W 5.10182e-05) | 0.033287 ± 0.000155579 (W 5.49819e-05) | · |  |
| `gemm/96x96x96/b1` | 0.112665 ± 0.000499285 (W 0.00139705) | 0.111784 ± 0.000424463 (W 0.00111411) | · |  |

## gemm-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0052445 ± 0.00025344 (W 6.316e-05) | 0.004363 ± 2.28035e-05 (W 4.76651e-05) | 🔴 faer-fork 1.20× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0197825 ± 0.000113022 (W 0.000104089) | 0.010169 ± 4.86936e-05 (W 8.3193e-05) | 🔴 faer-fork 1.95× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.041622 ± 0.000224816 (W 0.000168524) | 0.0299415 ± 7.87437e-05 (W 0.000619246) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |

## gemm-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005405 ± 0.000274016 (W 0.000140803) | 0.004368 ± 1.78932e-05 (W 4.62227e-05) | 🔴 faer-fork 1.24× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0180185 ± 0.000115117 (W 8.30255e-05) | 0.006382 ± 6.19045e-05 (W 0.000123093) | 🔴 faer-fork 2.82× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.038873 ± 0.000125262 (W 0.000128676) | 0.0164405 ± 9.80172e-05 (W 0.000124663) | 🔴 faer-fork 2.36× slower than `lapack-aocl` |  |

## gemm-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.004488 ± 2.843e-05 (W 3.73195e-05) | 0.004368 ± 1.7224e-05 (W 4.93922e-05) | · | dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.033984 ± 0.000159421 (W 0.00046155) | 0.0331665 ± 0.000122576 (W 5.76904e-05) | · |  |
| `gemm/96x96x96/b1` | 0.112983 ± 0.000699588 (W 9.71127e-05) | 0.111612 ± 0.000446326 (W 0.00113444) | · |  |

## gemm-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.005044 ± 0.000150265 (W 6.77128e-05) | 0.004373 ± 1.8619e-05 (W 4.72563e-05) | 🔴 faer-fork 1.15× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0198065 ± 0.000132622 (W 0.000100971) | 0.010129 ± 7.86782e-05 (W 8.37438e-05) | 🔴 faer-fork 1.96× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.0415665 ± 0.000234504 (W 0.000157609) | 0.029991 ± 0.00014053 (W 9.48486e-05) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |

## gemm-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/32x32x32/b1` | 0.0053805 ± 0.000203157 (W 0.000240466) | 0.004353 ± 2.40139e-05 (W 4.70288e-05) | 🔴 faer-fork 1.24× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/64x64x64/b1` | 0.0179835 ± 0.000106584 (W 7.94128e-05) | 0.0064315 ± 0.000210757 (W 9.11007e-05) | 🔴 faer-fork 2.80× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `gemm/96x96x96/b1` | 0.038838 ± 0.000134083 (W 9.33802e-05) | 0.0164105 ± 8.36359e-05 (W 0.000140075) | 🔴 faer-fork 2.37× slower than `lapack-aocl` |  |

## gemm-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.132631 ± 0.00106709 (W 0.000434608) | 0.132616 ± 0.000462761 (W 0.000873774) | · |  |
| `gemm/192x192x192/b1` | 0.447882 ± 0.00255801 (W 0.00238457) | 0.446306 ± 0.00230492 (W 0.0018992) | · |  |
| `gemm/256x256x256/b1` | 1.07037 ± 0.00707433 (W 0.00382049) | 1.06406 ± 0.00676122 (W 0.00216404) | · |  |
| `gemm/384x384x384/b1` | 3.59619 ± 0.0168807 (W 0.0097031) | 3.65433 ± 0.0416602 (W 0.00617698) | · |  |
| `gemm/512x512x512/b1` | 8.55364 ± 0.0330211 (W 0.0134836) | 8.6722 ± 0.0442737 (W 0.0130415) | · |  |
| `gemm/768x768x768/b1` | 29.1835 ± 0.117808 (W 0.0397446) | 29.1071 ± 0.20688 (W 0.0259395) | · |  |

## gemm-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.045209 ± 0.000137888 (W 0.000291202) | 0.035371 ± 0.000127616 (W 0.000118547) | 🔴 faer-fork 1.28× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.120787 ± 0.000636252 (W 0.00167116) | 0.114261 ± 7.73167e-05 (W 0.00114668) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.2905 ± 0.000893106 (W 0.00177367) | 0.268679 ± 0.000804737 (W 0.00150849) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.915519 ± 0.00300126 (W 0.00279986) | 0.909224 ± 0.00393373 (W 0.00180552) | · |  |
| `gemm/512x512x512/b1` | 2.20295 ± 0.00290661 (W 0.00369735) | 2.19702 ± 0.00364362 (W 0.00217209) | · |  |
| `gemm/768x768x768/b1` | 7.49142 ± 0.0253325 (W 0.0142803) | 7.44398 ± 0.047444 (W 0.0119796) | · |  |

## gemm-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.032295 ± 0.000869815 (W 0.000824916) | 0.0213195 ± 0.000103521 (W 0.000136457) | 🔴 faer-fork 1.51× slower than `lapack-aocl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.0861395 ± 0.000300211 (W 0.00143878) | 0.0585395 ± 0.000206077 (W 0.000128348) | 🔴 faer-fork 1.47× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.16454 ± 0.000886856 (W 0.00155706) | 0.143533 ± 0.000656188 (W 0.00137876) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.47589 ± 0.00116961 (W 0.00339924) | 0.458858 ± 0.00195472 (W 0.00191905) | · |  |
| `gemm/512x512x512/b1` | 1.12971 ± 0.00334747 (W 0.0046883) | 1.12971 ± 0.00273974 (W 0.00247297) | · |  |
| `gemm/768x768x768/b1` | 3.87412 ± 0.00961236 (W 0.013601) | 3.80215 ± 0.0320165 (W 0.013046) | · |  |

## gemm-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.133222 ± 0.000506685 (W 0.0012019) | 0.132099 ± 0.000438945 (W 0.00111168) | · |  |
| `gemm/192x192x192/b1` | 0.449591 ± 0.00176138 (W 0.00194305) | 0.447763 ± 0.00225451 (W 0.00200744) | · |  |
| `gemm/256x256x256/b1` | 1.0654 ± 0.00228248 (W 0.00231716) | 1.06504 ± 0.00831374 (W 0.00143038) | · |  |
| `gemm/384x384x384/b1` | 3.60261 ± 0.00998036 (W 0.00811451) | 3.61567 ± 0.0405285 (W 0.00662899) | · |  |
| `gemm/512x512x512/b1` | 8.55358 ± 0.0253387 (W 0.0113871) | 8.73233 ± 0.0300259 (W 0.0120761) | · |  |
| `gemm/768x768x768/b1` | 29.2247 ± 0.0724961 (W 0.0357979) | 29.2633 ± 0.0965475 (W 0.0586837) | · |  |

## gemm-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.04542 ± 0.000235968 (W 0.000270577) | 0.0353955 ± 0.000139136 (W 0.000130364) | 🔴 faer-fork 1.28× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.119774 ± 0.000838657 (W 0.00167082) | 0.114199 ± 0.000474684 (W 0.00114706) | · |  |
| `gemm/256x256x256/b1` | 0.29016 ± 0.00116349 (W 0.00150761) | 0.269874 ± 0.00126552 (W 0.001613) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.913984 ± 0.00218091 (W 0.00184358) | 0.909445 ± 0.00205483 (W 0.00142314) | · |  |
| `gemm/512x512x512/b1` | 2.20436 ± 0.00278375 (W 0.00309228) | 2.19616 ± 0.00754538 (W 0.00220384) | · |  |
| `gemm/768x768x768/b1` | 7.48541 ± 0.00758785 (W 0.00737585) | 7.41346 ± 0.0217346 (W 0.0111612) | · |  |

## gemm-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0320945 ± 0.000502084 (W 0.000914443) | 0.0212655 ± 6.41293e-05 (W 0.000103947) | 🔴 faer-fork 1.51× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.086375 ± 0.000627638 (W 0.000759786) | 0.058724 ± 0.000196281 (W 0.000263953) | 🔴 faer-fork 1.47× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.164361 ± 0.00126834 (W 0.00202944) | 0.143792 ± 0.00100781 (W 0.00151045) | 🔴 faer-fork 1.14× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.476247 ± 0.00151003 (W 0.00293592) | 0.459875 ± 0.00196331 (W 0.00214111) | · |  |
| `gemm/512x512x512/b1` | 1.12711 ± 0.00629002 (W 0.00406158) | 1.13007 ± 0.00841967 (W 0.00365985) | · |  |
| `gemm/768x768x768/b1` | 3.87412 ± 0.0104165 (W 0.0120686) | 3.82205 ± 0.0240585 (W 0.035944) | · |  |

## gemm-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.534766 ± 0.00310724 (W 0.00238077) | 0.528438 ± 0.00200546 (W 0.00273517) | · |  |
| `gemm/192x192x192/b1` | 1.79717 ± 0.00952542 (W 0.00343597) | 1.79038 ± 0.00488066 (W 0.0054482) | · |  |
| `gemm/256x256x256/b1` | 4.25983 ± 0.0125644 (W 0.0103068) | 4.39764 ± 0.10207 (W 0.00584411) | · | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 14.3511 ± 0.0294335 (W 0.019331) | 14.5521 ± 0.0335474 (W 0.0139306) | · |  |
| `gemm/512x512x512/b1` | 34.1015 ± 0.1018 (W 0.0366603) | 35.3668 ± 0.130188 (W 0.0405697) | · |  |
| `gemm/768x768x768/b1` | 115.828 ± 0.274539 (W 0.138098) | 119.783 ± 0.371819 (W 0.273886) | · |  |

## gemm-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.153146 ± 0.000570243 (W 0.000993773) | 0.135789 ± 0.000459182 (W 0.000931477) | 🔴 faer-fork 1.13× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.456964 ± 0.000947413 (W 0.00166577) | 0.455169 ± 0.00164753 (W 0.00205441) | · |  |
| `gemm/256x256x256/b1` | 1.10928 ± 0.00148496 (W 0.00141701) | 1.10641 ± 0.00362916 (W 0.00159582) | · |  |
| `gemm/384x384x384/b1` | 3.62119 ± 0.00175942 (W 0.00467182) | 3.65243 ± 0.0119202 (W 0.00257293) | · |  |
| `gemm/512x512x512/b1` | 8.6322 ± 0.00602853 (W 0.00894906) | 8.83956 ± 0.0193543 (W 0.00600904) | · |  |
| `gemm/768x768x768/b1` | 29.3829 ± 0.139672 (W 0.0223938) | 30.2646 ± 0.0382461 (W 0.0805109) | · |  |

## gemm-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.102775 ± 0.00025693 (W 0.000609113) | 0.073949 ± 0.000329294 (W 0.000785504) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.234875 ± 0.000919373 (W 0.00152212) | 0.227681 ± 0.000696732 (W 0.0015047) | · |  |
| `gemm/256x256x256/b1` | 0.578431 ± 0.0014505 (W 0.00212761) | 0.564338 ± 0.0028491 (W 0.00185528) | · |  |
| `gemm/384x384x384/b1` | 1.81659 ± 0.00869298 (W 0.00351755) | 1.82877 ± 0.00327059 (W 0.00539904) | · |  |
| `gemm/512x512x512/b1` | 4.42391 ± 0.0155857 (W 0.0116309) | 4.49691 ± 0.0127293 (W 0.00733854) | · |  |
| `gemm/768x768x768/b1` | 14.8638 ± 0.0346129 (W 0.0120902) | 15.2796 ± 0.0346519 (W 0.0196557) | · |  |

## gemm-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.536004 ± 0.00189578 (W 0.0021405) | 0.527649 ± 0.00218446 (W 0.00213838) | · |  |
| `gemm/192x192x192/b1` | 1.8014 ± 0.0105934 (W 0.00175112) | 1.79497 ± 0.00669555 (W 0.00322368) | · |  |
| `gemm/256x256x256/b1` | 4.24982 ± 0.0108725 (W 0.00646542) | 4.39048 ± 0.09946 (W 0.00522229) | · | `!` between-start range over 5% |
| `gemm/384x384x384/b1` | 14.3488 ± 0.0541466 (W 0.0143009) | 14.6114 ± 0.0690305 (W 0.0135905) | · |  |
| `gemm/512x512x512/b1` | 34.0882 ± 0.0846221 (W 0.0433778) | 35.3967 ± 0.1131 (W 0.0511691) | · |  |
| `gemm/768x768x768/b1` | 115.789 ± 0.254057 (W 0.107536) | 119.797 ± 0.242047 (W 0.287392) | · |  |

## gemm-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.153235 ± 0.000274494 (W 0.000501743) | 0.135857 ± 0.000266538 (W 0.00162027) | 🔴 faer-fork 1.13× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.458081 ± 0.00205071 (W 0.00216067) | 0.457234 ± 0.00214113 (W 0.00209239) | · |  |
| `gemm/256x256x256/b1` | 1.10932 ± 0.00170876 (W 0.00177695) | 1.10339 ± 0.00172998 (W 0.00152097) | · |  |
| `gemm/384x384x384/b1` | 3.62123 ± 0.00534075 (W 0.00604905) | 3.65721 ± 0.00931704 (W 0.00310545) | · |  |
| `gemm/512x512x512/b1` | 8.63431 ± 0.00660951 (W 0.00792429) | 8.83169 ± 0.00895085 (W 0.00472472) | · |  |
| `gemm/768x768x768/b1` | 29.345 ± 0.0322819 (W 0.0266415) | 30.2331 ± 0.0552298 (W 0.0265286) | · |  |

## gemm-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.102765 ± 0.000245428 (W 0.000830771) | 0.0735925 ± 0.000432389 (W 0.000170488) | 🔴 faer-fork 1.40× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.2343 ± 0.000866417 (W 0.00169397) | 0.227748 ± 0.000713688 (W 0.00171242) | · |  |
| `gemm/256x256x256/b1` | 0.578948 ± 0.00262536 (W 0.00270536) | 0.564697 ± 0.00110268 (W 0.00214675) | · |  |
| `gemm/384x384x384/b1` | 1.81601 ± 0.00334302 (W 0.00412302) | 1.82432 ± 0.00322765 (W 0.00557081) | · |  |
| `gemm/512x512x512/b1` | 4.41395 ± 0.0108643 (W 0.00547673) | 4.51016 ± 0.0107237 (W 0.00553814) | · |  |
| `gemm/768x768x768/b1` | 14.8828 ± 0.0370316 (W 0.0175591) | 15.2397 ± 0.0346588 (W 0.0149279) | · |  |

## gemm-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0668905 ± 0.00034114 (W 9.72199e-05) | 0.0674465 ± 0.00020012 (W 0.000680028) | · |  |
| `gemm/192x192x192/b1` | 0.22282 ± 0.00104766 (W 0.00140182) | 0.223452 ± 0.000725754 (W 0.00136562) | · |  |
| `gemm/256x256x256/b1` | 0.53188 ± 0.00245032 (W 0.00208796) | 0.54387 ± 0.00220438 (W 0.00231841) | · |  |
| `gemm/384x384x384/b1` | 1.79323 ± 0.00245763 (W 0.00457611) | 1.81149 ± 0.00235506 (W 0.00471769) | · |  |
| `gemm/512x512x512/b1` | 4.23457 ± 0.00959505 (W 0.00742737) | 4.27185 ± 0.00498612 (W 0.00622831) | · |  |
| `gemm/768x768x768/b1` | 14.4204 ± 0.00750932 (W 0.018163) | 14.4663 ± 0.0338312 (W 0.0112687) | · |  |

## gemm-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0216605 ± 0.000432964 (W 0.000453735) | 0.019266 ± 9.10394e-05 (W 0.000149495) | 🔴 faer-fork 1.12× slower than `lapack-aocl` | `!` between-start range over 5% |
| `gemm/192x192x192/b1` | 0.0835565 ± 0.000501409 (W 0.000847984) | 0.076278 ± 0.00102683 (W 0.000792549) | 🔴 faer-fork 1.10× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.157805 ± 0.000553744 (W 0.000986193) | 0.136922 ± 0.000157387 (W 0.0010607) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.465006 ± 0.00123635 (W 0.0028644) | 0.452241 ± 0.00158357 (W 0.00189416) | · |  |
| `gemm/512x512x512/b1` | 1.11195 ± 0.00325674 (W 0.00214912) | 1.10041 ± 0.00620638 (W 0.00171013) | · |  |
| `gemm/768x768x768/b1` | 3.65942 ± 0.00907324 (W 0.0047195) | 3.61989 ± 0.00669648 (W 0.00446174) | · |  |

## gemm-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0217105 ± 0.000290975 (W 0.000432854) | 0.011822 ± 7.57408e-05 (W 0.000131952) | 🔴 faer-fork 1.84× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.049107 ± 0.000519562 (W 0.00095647) | 0.039984 ± 0.000129744 (W 0.000220523) | 🔴 faer-fork 1.23× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.107967 ± 0.000898231 (W 0.000810033) | 0.070673 ± 0.000178254 (W 0.000211893) | 🔴 faer-fork 1.53× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.247704 ± 0.000752066 (W 0.0027507) | 0.225866 ± 0.00046666 (W 0.00192438) | 🔴 faer-fork 1.10× slower than `lapack-aocl` |  |
| `gemm/512x512x512/b1` | 0.591963 ± 0.00274935 (W 0.00363605) | 0.566242 ± 0.00548306 (W 0.00244221) | · |  |
| `gemm/768x768x768/b1` | 1.87567 ± 0.00536109 (W 0.00434808) | 1.8203 ± 0.0201214 (W 0.00531432) | · |  |

## gemm-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.066809 ± 0.000241354 (W 6.70586e-05) | 0.0670565 ± 0.000288171 (W 0.000622916) | · |  |
| `gemm/192x192x192/b1` | 0.222862 ± 0.000663722 (W 0.00128206) | 0.223156 ± 0.000805355 (W 0.00150752) | · |  |
| `gemm/256x256x256/b1` | 0.530823 ± 0.00153871 (W 0.00188657) | 0.542759 ± 0.0030578 (W 0.00205379) | · |  |
| `gemm/384x384x384/b1` | 1.79058 ± 0.00376813 (W 0.00427843) | 1.81057 ± 0.00698594 (W 0.00340757) | · |  |
| `gemm/512x512x512/b1` | 4.24198 ± 0.0107448 (W 0.0068317) | 4.26949 ± 0.012765 (W 0.00416795) | · |  |
| `gemm/768x768x768/b1` | 14.4992 ± 0.0620129 (W 0.023233) | 14.4933 ± 0.0629803 (W 0.0300017) | · |  |

## gemm-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.021771 ± 0.000165071 (W 0.000389022) | 0.019221 ± 6.45724e-05 (W 0.000108089) | 🔴 faer-fork 1.13× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.084127 ± 0.000653059 (W 0.00111398) | 0.0765885 ± 0.000524499 (W 0.000187163) | 🔴 faer-fork 1.10× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.15776 ± 0.000874033 (W 0.000667593) | 0.136903 ± 0.000208945 (W 0.00120616) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.463727 ± 0.00186021 (W 0.00341487) | 0.451802 ± 0.00147682 (W 0.00201304) | · |  |
| `gemm/512x512x512/b1` | 1.11303 ± 0.00203997 (W 0.00130133) | 1.09886 ± 0.00563917 (W 0.00317557) | · |  |
| `gemm/768x768x768/b1` | 3.65441 ± 0.00514182 (W 0.00477895) | 3.62033 ± 0.00823037 (W 0.00252915) | · |  |

## gemm-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.021801 ± 0.000291731 (W 0.00031391) | 0.011727 ± 0.000139414 (W 0.000107508) | 🔴 faer-fork 1.86× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.0486965 ± 0.000403125 (W 0.000986489) | 0.0400845 ± 0.000150946 (W 0.00012669) | 🔴 faer-fork 1.21× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.107431 ± 0.000501449 (W 0.000939484) | 0.070994 ± 0.000402207 (W 0.000242237) | 🔴 faer-fork 1.51× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.248667 ± 0.00144749 (W 0.00316787) | 0.225628 ± 0.000979183 (W 0.00201213) | 🔴 faer-fork 1.10× slower than `lapack-aocl` |  |
| `gemm/512x512x512/b1` | 0.592972 ± 0.00198965 (W 0.00240626) | 0.566184 ± 0.00735885 (W 0.00213937) | · |  |
| `gemm/768x768x768/b1` | 1.87187 ± 0.0121801 (W 0.00384011) | 1.82086 ± 0.0181447 (W 0.00272183) | · |  |

## gemm-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.265355 ± 0.000697605 (W 0.00149747) | 0.262949 ± 0.000672275 (W 0.00164993) | · |  |
| `gemm/192x192x192/b1` | 0.896278 ± 0.00371187 (W 0.00118197) | 0.889165 ± 0.00216476 (W 0.00145826) | · |  |
| `gemm/256x256x256/b1` | 2.1231 ± 0.00664148 (W 0.00412095) | 2.11542 ± 0.0103425 (W 0.00556767) | · |  |
| `gemm/384x384x384/b1` | 7.16181 ± 0.0116227 (W 0.0148414) | 7.12637 ± 0.0168942 (W 0.00705457) | · |  |
| `gemm/512x512x512/b1` | 16.933 ± 0.0529285 (W 0.019272) | 17.0654 ± 0.0539399 (W 0.0213217) | · |  |
| `gemm/768x768x768/b1` | 57.7499 ± 0.150711 (W 0.160377) | 57.981 ± 0.193052 (W 0.0392924) | · |  |

## gemm-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.087668 ± 0.000317078 (W 0.000530755) | 0.0684965 ± 5.57554e-05 (W 9.82373e-05) | 🔴 faer-fork 1.28× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.233172 ± 0.000232825 (W 0.00137398) | 0.225398 ± 0.000439412 (W 0.00188578) | · |  |
| `gemm/256x256x256/b1` | 0.571854 ± 0.00118047 (W 0.00188063) | 0.539635 ± 0.00297585 (W 0.00194174) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 1.80694 ± 0.00188474 (W 0.00275107) | 1.79853 ± 0.00219965 (W 0.00211412) | · |  |
| `gemm/512x512x512/b1` | 4.34562 ± 0.00461123 (W 0.00379254) | 4.3043 ± 0.0132807 (W 0.00758017) | · |  |
| `gemm/768x768x768/b1` | 14.6626 ± 0.0087569 (W 0.0157151) | 15.0618 ± 0.0174062 (W 0.0201225) | · |  |

## gemm-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0562215 ± 0.00053204 (W 0.000866329) | 0.035516 ± 0.000184372 (W 0.000145192) | 🔴 faer-fork 1.58× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.158278 ± 0.000594822 (W 0.00106621) | 0.113915 ± 0.000458979 (W 0.000160422) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.308903 ± 0.00134051 (W 0.00133779) | 0.268738 ± 0.00211837 (W 0.00190383) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.912737 ± 0.0032998 (W 0.0018058) | 0.905876 ± 0.00436104 (W 0.0016218) | · |  |
| `gemm/512x512x512/b1` | 2.20291 ± 0.00790372 (W 0.00403427) | 2.16904 ± 0.00984594 (W 0.00362673) | · |  |
| `gemm/768x768x768/b1` | 7.43962 ± 0.0196519 (W 0.00698396) | 7.97923 ± 0.013298 (W 0.0417068) | 🔵 faer-fork 1.07× faster than `lapack-aocl` |  |

## gemm-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.26669 ± 0.000989519 (W 0.00144468) | 0.263156 ± 0.000866856 (W 0.00220132) | · |  |
| `gemm/192x192x192/b1` | 0.896919 ± 0.00310009 (W 0.000781187) | 0.890406 ± 0.00341241 (W 0.00169607) | · |  |
| `gemm/256x256x256/b1` | 2.12452 ± 0.00622091 (W 0.00282874) | 2.10998 ± 0.00571421 (W 0.00382977) | · |  |
| `gemm/384x384x384/b1` | 7.14647 ± 0.0199343 (W 0.00904549) | 7.13165 ± 0.0159101 (W 0.0109045) | · |  |
| `gemm/512x512x512/b1` | 16.9394 ± 0.0455392 (W 0.0196506) | 17.072 ± 0.0426684 (W 0.0164569) | · |  |
| `gemm/768x768x768/b1` | 57.6962 ± 0.166724 (W 0.0656804) | 58.0987 ± 0.160252 (W 0.0535716) | · |  |

## gemm-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0877285 ± 0.000313957 (W 0.000237346) | 0.0684935 ± 0.000248178 (W 0.00114612) | 🔴 faer-fork 1.28× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.233502 ± 0.00056366 (W 0.00120237) | 0.225622 ± 0.000346705 (W 0.00145792) | · |  |
| `gemm/256x256x256/b1` | 0.572263 ± 0.000970976 (W 0.0016938) | 0.536007 ± 0.00407574 (W 0.00184145) | 🔴 faer-fork 1.07× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 1.80949 ± 0.00219368 (W 0.00214695) | 1.79848 ± 0.0110806 (W 0.00334835) | · |  |
| `gemm/512x512x512/b1` | 4.34377 ± 0.0041708 (W 0.00359026) | 4.30806 ± 0.0179575 (W 0.00741389) | · |  |
| `gemm/768x768x768/b1` | 14.663 ± 0.0146618 (W 0.0190866) | 15.0407 ± 0.0220935 (W 0.0238727) | · |  |

## gemm-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/128x128x128/b1` | 0.0561405 ± 0.000559693 (W 0.000829759) | 0.0355105 ± 0.000121256 (W 0.000182686) | 🔴 faer-fork 1.58× slower than `lapack-aocl` |  |
| `gemm/192x192x192/b1` | 0.158376 ± 0.00107993 (W 0.000903443) | 0.113926 ± 0.000461976 (W 0.000795182) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |
| `gemm/256x256x256/b1` | 0.309277 ± 0.00108245 (W 0.00154348) | 0.267841 ± 0.00162804 (W 0.00162956) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `gemm/384x384x384/b1` | 0.911798 ± 0.00199838 (W 0.00114863) | 0.906005 ± 0.00189552 (W 0.00184783) | · |  |
| `gemm/512x512x512/b1` | 2.20156 ± 0.00530496 (W 0.00339969) | 2.16463 ± 0.00533281 (W 0.00303706) | · |  |
| `gemm/768x768x768/b1` | 7.42758 ± 0.0145511 (W 0.0101419) | 7.95658 ± 0.0215303 (W 0.0288054) | 🔵 faer-fork 1.07× faster than `lapack-aocl` |  |

## gemm-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.2722 ± 0.00378439 (W 0.0070121) | 1.02875 ± 0.00592338 (W 0.00216908) | 🔴 faer-fork 1.24× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 1.0767 ± 0.0212764 (W 0.00215456) | 1.14347 ± 0.00472771 (W 0.00850807) | 🔵 faer-fork 1.06× faster than `lapack-aocl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 1.08228 ± 0.00431909 (W 0.00272962) | 1.10325 ± 0.00240955 (W 0.00201879) | · |  |

## gemm-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.421953 ± 0.00181486 (W 0.0042567) | 0.31184 ± 0.00235984 (W 0.00388113) | 🔴 faer-fork 1.35× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.309847 ± 0.00273502 (W 0.0025249) | 0.285304 ± 0.00191941 (W 0.00265706) | 🔴 faer-fork 1.09× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.277246 ± 0.00091234 (W 0.00186636) | 0.275205 ± 0.000886845 (W 0.00150718) | · |  |

## gemm-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.420846 ± 0.00176531 (W 0.00271995) | 0.166287 ± 0.0014333 (W 0.00336266) | 🔴 faer-fork 2.53× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.187287 ± 0.000363741 (W 0.00148226) | 0.192774 ± 0.00139947 (W 0.00215519) | · |  |
| `gemm/4096x64x64/b1` | 0.141616 ± 0.0010109 (W 0.000971532) | 0.138173 ± 0.000479564 (W 0.000959918) | · |  |

## gemm-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.0367 ± 0.0030915 (W 0.00169068) | 1.05171 ± 0.0123609 (W 0.00203305) | · |  |
| `gemm/4096x16x240/b1` | 1.09712 ± 0.00368664 (W 0.0078414) | 1.13239 ± 0.00844251 (W 0.00454241) | · |  |
| `gemm/4096x64x64/b1` | 1.12744 ± 0.00375385 (W 0.00284116) | 1.12308 ± 0.00657407 (W 0.00210027) | · |  |

## gemm-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.32173 ± 0.00151833 (W 0.00384957) | 0.277238 ± 0.00504603 (W 0.0036714) | 🔴 faer-fork 1.16× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.301091 ± 0.0017726 (W 0.00237985) | 0.293899 ± 0.00111356 (W 0.0019199) | · |  |
| `gemm/4096x64x64/b1` | 0.296887 ± 0.00102424 (W 0.00178121) | 0.280577 ± 0.00193847 (W 0.00187267) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |

## gemm-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.196093 ± 0.00128479 (W 0.00387908) | 0.144323 ± 0.000920093 (W 0.00286355) | 🔴 faer-fork 1.36× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.180531 ± 0.00137343 (W 0.00276037) | 0.201776 ± 0.0136166 (W 0.00358266) | 🔵 faer-fork 1.12× faster than `lapack-aocl` | `!` between-start range over 5% |
| `gemm/4096x64x64/b1` | 0.154787 ± 0.000858483 (W 0.000899922) | 0.142452 ± 0.000464077 (W 0.00149608) | 🔴 faer-fork 1.09× slower than `lapack-aocl` |  |

## gemm-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.14231 ± 0.0142359 (W 0.00829925) | 4.72154 ± 0.0546602 (W 0.0228404) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 4.39813 ± 0.024471 (W 0.0125758) | 4.15231 ± 0.0208301 (W 0.00686135) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 4.3663 ± 0.0169662 (W 0.00683178) | 4.30828 ± 0.0297406 (W 0.0046042) | · |  |

## gemm-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.65429 ± 0.00392436 (W 0.00575046) | 1.29138 ± 0.00652299 (W 0.00381286) | 🔴 faer-fork 1.28× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 1.16079 ± 0.0064306 (W 0.00390582) | 1.03643 ± 0.00795334 (W 0.000826485) | 🔴 faer-fork 1.12× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 1.12708 ± 0.00438839 (W 0.00227546) | 1.08553 ± 0.00289715 (W 0.00227926) | · |  |

## gemm-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.902347 ± 0.00772091 (W 0.0148623) | 0.779343 ± 0.00388584 (W 0.0110059) | 🔴 faer-fork 1.16× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.594213 ± 0.00243193 (W 0.00449466) | 0.522489 ± 0.00176343 (W 0.00244001) | 🔴 faer-fork 1.14× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.568339 ± 0.0026312 (W 0.00234753) | 0.538725 ± 0.00227145 (W 0.00247779) | 🔴 faer-fork 1.05× slower than `lapack-aocl` |  |

## gemm-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 4.05784 ± 0.0171114 (W 0.0105213) | 4.16208 ± 0.0091305 (W 0.00693391) | · |  |
| `gemm/4096x16x240/b1` | 4.37455 ± 0.0209371 (W 0.00529359) | 4.17682 ± 0.0174227 (W 0.00648919) | · |  |
| `gemm/4096x64x64/b1` | 4.4331 ± 0.0191031 (W 0.00984323) | 4.434 ± 0.0136608 (W 0.00783601) | · |  |

## gemm-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 1.17879 ± 0.00504862 (W 0.00184804) | 1.04754 ± 0.00392092 (W 0.00215636) | 🔴 faer-fork 1.13× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 1.16343 ± 0.00286928 (W 0.00324673) | 1.0668 ± 0.00638394 (W 0.00229524) | 🔴 faer-fork 1.09× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 1.16485 ± 0.00469252 (W 0.00366131) | 1.10403 ± 0.00399507 (W 0.00384928) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |

## gemm-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.651267 ± 0.00148347 (W 0.00366294) | 0.570327 ± 0.00412816 (W 0.00475436) | 🔴 faer-fork 1.14× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.598533 ± 0.00487098 (W 0.00381564) | 0.548428 ± 0.00879242 (W 0.00429121) | 🔴 faer-fork 1.09× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.571747 ± 0.00306127 (W 0.00378075) | 0.547808 ± 0.00303204 (W 0.00205492) | · |  |

## gemm-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.542501 ± 0.00999383 (W 0.00288796) | 0.505038 ± 0.00145021 (W 0.0020554) | 🔴 faer-fork 1.07× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.537329 ± 0.00292841 (W 0.00228616) | 0.539524 ± 0.00297934 (W 0.00216077) | · |  |
| `gemm/4096x64x64/b1` | 0.54104 ± 0.00346336 (W 0.00206537) | 0.542635 ± 0.00289432 (W 0.00250922) | · |  |

## gemm-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.408289 ± 0.00130328 (W 0.0019886) | 0.137294 ± 0.000516899 (W 0.000980982) | 🔴 faer-fork 2.97× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.144689 ± 0.0010075 (W 0.00105085) | 0.13668 ± 0.000237417 (W 0.00142965) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.139826 ± 0.00133636 (W 0.000386061) | 0.136567 ± 0.00057129 (W 0.00130454) | · |  |

## gemm-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.407506 ± 0.00186123 (W 0.00158142) | 0.137832 ± 0.000660664 (W 0.00142275) | 🔴 faer-fork 2.96× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.0771785 ± 0.000308419 (W 0.000646247) | 0.071082 ± 2.38532 (W 0.000415768) | · | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x64x64/b1` | 0.072118 ± 0.000270672 (W 0.00104296) | 0.069589 ± 0.00032792 (W 0.000846977) | · |  |

## gemm-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.536209 ± 0.00349104 (W 0.00304035) | 0.521749 ± 0.00669426 (W 0.00276214) | · |  |
| `gemm/4096x16x240/b1` | 0.545249 ± 0.00277704 (W 0.0025151) | 0.640864 ± 0.00372916 (W 0.00452751) | 🔵 faer-fork 1.18× faster than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.553558 ± 0.00186661 (W 0.00178593) | 0.55709 ± 0.00417926 (W 0.00269089) | · |  |

## gemm-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.177546 ± 0.00154422 (W 0.00193969) | 0.261055 ± 0.00136547 (W 0.00152306) | 🔵 faer-fork 1.47× faster than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.145663 ± 0.000829469 (W 0.00103108) | 0.160838 ± 0.00162981 (W 0.00208826) | 🔵 faer-fork 1.10× faster than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.156043 ± 0.00133859 (W 0.0017167) | 0.14042 ± 0.00138118 (W 0.00127393) | 🔴 faer-fork 1.11× slower than `lapack-aocl` |  |

## gemm-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.110789 ± 0.000867278 (W 0.000951147) | 0.130487 ± 0.000427714 (W 0.0013911) | 🔵 faer-fork 1.18× faster than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.0815875 ± 0.000722655 (W 0.000798868) | 0.0851085 ± 0.000642681 (W 0.000569) | · |  |
| `gemm/4096x64x64/b1` | 0.079291 ± 0.000440164 (W 0.000465314) | 0.072108 ± 0.000469513 (W 0.000433152) | 🔴 faer-fork 1.10× slower than `lapack-aocl` |  |

## gemm-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.11987 ± 0.00911018 (W 0.00423018) | 2.07825 ± 0.00995231 (W 0.00853252) | · |  |
| `gemm/4096x16x240/b1` | 2.15745 ± 0.00752545 (W 0.00322433) | 2.11728 ± 0.0100304 (W 0.00491994) | · |  |
| `gemm/4096x64x64/b1` | 2.16252 ± 0.00570631 (W 0.0052853) | 2.1398 ± 0.00686883 (W 0.00271226) | · |  |

## gemm-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.817826 ± 0.0035784 (W 0.00333671) | 0.632309 ± 0.00295333 (W 0.00498089) | 🔴 faer-fork 1.29× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.581387 ± 0.00392825 (W 0.00261643) | 0.535935 ± 0.00223934 (W 0.00283277) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.54993 ± 0.00159684 (W 0.00218804) | 0.537203 ± 0.00227157 (W 0.00186664) | · |  |

## gemm-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.820348 ± 0.00107267 (W 0.00427586) | 0.326878 ± 0.00158466 (W 0.00435124) | 🔴 faer-fork 2.51× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.298559 ± 0.00254733 (W 0.00170948) | 0.280937 ± 0.00108459 (W 0.00255805) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.276285 ± 0.00107443 (W 0.00168752) | 0.267948 ± 0.00139091 (W 0.00191741) | · |  |

## gemm-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 2.02936 ± 0.00731249 (W 0.00534475) | 2.02451 ± 0.00978067 (W 0.00222306) | · |  |
| `gemm/4096x16x240/b1` | 2.17194 ± 0.00388823 (W 0.00427399) | 2.09646 ± 0.0115756 (W 0.00557408) | · |  |
| `gemm/4096x64x64/b1` | 2.19227 ± 0.00490036 (W 0.00478005) | 2.15084 ± 0.00553289 (W 0.00717747) | · |  |

## gemm-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.590168 ± 0.00297901 (W 0.00287342) | 0.511339 ± 0.00241396 (W 0.00353842) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.572324 ± 0.00241795 (W 0.00244129) | 0.530627 ± 0.00150628 (W 0.0016065) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.57216 ± 0.00196805 (W 0.00218114) | 0.545097 ± 0.00136745 (W 0.00213406) | · |  |

## gemm-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/16x4096x240/b1` | 0.321782 ± 0.00148906 (W 0.00184183) | 0.272278 ± 0.00133869 (W 0.00273197) | 🔴 faer-fork 1.18× slower than `lapack-aocl` |  |
| `gemm/4096x16x240/b1` | 0.29016 ± 0.00241683 (W 0.00144143) | 0.271396 ± 0.00117169 (W 0.00243657) | 🔴 faer-fork 1.07× slower than `lapack-aocl` |  |
| `gemm/4096x64x64/b1` | 0.309163 ± 0.000985205 (W 0.0018952) | 0.278788 ± 0.00117375 (W 0.00168936) | 🔴 faer-fork 1.11× slower than `lapack-aocl` |  |

## gemm-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.1303 ± 0.00388322 (W 0.00256853) | 1.11955 ± 0.00678007 (W 0.00176093) | · |  |

## gemm-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.29613 ± 0.000862382 (W 0.00197238) | 0.279085 ± 0.00179864 (W 0.00179916) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |

## gemm-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.155098 ± 0.000728371 (W 0.00148793) | 0.143053 ± 0.000532482 (W 0.0014097) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |

## gemm-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.08743 ± 0.00817298 (W 0.00201334) | 1.10255 ± 0.00300025 (W 0.00198922) | · |  |

## gemm-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.277312 ± 0.000520258 (W 0.00179672) | 0.275652 ± 2.3207 (W 0.00218197) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.141817 ± 0.000677498 (W 0.0013716) | 0.139519 ± 0.0012814 (W 0.00132282) | · |  |

## gemm-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.43033 ± 0.0192557 (W 0.0118161) | 4.43138 ± 0.0207871 (W 0.00680478) | · |  |

## gemm-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.1638 ± 0.00328913 (W 0.00312443) | 1.10805 ± 0.0040757 (W 0.00301576) | · |  |

## gemm-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.574255 ± 0.0035181 (W 0.00407699) | 0.54676 ± 0.0101123 (W 0.00254366) | · |  |

## gemm-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 4.37483 ± 0.0190074 (W 0.00577458) | 4.31008 ± 0.0130679 (W 0.00694815) | · |  |

## gemm-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 1.12478 ± 0.00418233 (W 0.00163984) | 1.08621 ± 0.00454401 (W 0.00252289) | · |  |

## gemm-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.57001 ± 0.000840536 (W 0.00294979) | 0.538504 ± 0.00217415 (W 0.00204776) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |

## gemm-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.554415 ± 0.00408883 (W 0.00214742) | 0.555629 ± 0.00638291 (W 0.00255543) | · |  |

## gemm-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.156563 ± 0.000793924 (W 0.00195306) | 0.140483 ± 0.00149454 (W 0.00126743) | 🔴 faer-fork 1.11× slower than `lapack-aocl` |  |

## gemm-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.0798225 ± 0.000668408 (W 0.000786512) | 0.0720595 ± 0.000427706 (W 0.00111736) | 🔴 faer-fork 1.11× slower than `lapack-aocl` |  |

## gemm-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.54017 ± 0.00295131 (W 0.00235249) | 0.545195 ± 0.00110712 (W 0.0028043) | · |  |

## gemm-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.139838 ± 0.000438689 (W 0.00045964) | 0.13657 ± 0.0011262 (W 0.00125334) | · |  |

## gemm-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.072094 ± 0.000240679 (W 0.000231448) | 0.0700705 ± 0.000272807 (W 0.000156046) | · |  |

## gemm-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.19279 ± 0.00479422 (W 0.00275062) | 2.15275 ± 0.00690478 (W 0.00277326) | · |  |

## gemm-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.570774 ± 0.00157932 (W 0.00201285) | 0.545987 ± 0.00212561 (W 0.00190571) | · |  |

## gemm-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.308966 ± 0.000830094 (W 0.00176952) | 0.2785 ± 0.00171309 (W 0.00197381) | 🔴 faer-fork 1.11× slower than `lapack-aocl` |  |

## gemm-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 2.16322 ± 0.00639195 (W 0.00259318) | 2.14123 ± 0.00560481 (W 0.00313581) | · |  |

## gemm-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.5516 ± 0.000991548 (W 0.00176566) | 0.536123 ± 0.00184488 (W 0.00160972) | · |  |

## gemm-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `gemm/64x64x4096/b1` | 0.276648 ± 0.001273 (W 0.00132759) | 0.267713 ± 0.000955237 (W 0.00162577) | · |  |

## linalg-small, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004113 ± 0.000162896 (W 5.12338e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003537 ± 0.000152873 (W 5.48475e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013931 ± 0.000371108 (W 7.83763e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0125685 ± 0.000428788 (W 0.000106251) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0304225 ± 0.000452379 (W 0.000138139) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0279325 ± 0.000380172 (W 9.10314e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.111364 ± 0.000544564 (W 0.000832018) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.111617 ± 0.00944661 (W 0.000820819) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.571032 ± 0.00311993 (W 0.00182204) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.572964 ± 0.00151918 (W 0.00216781) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.3323 ± 0.00492734 (W 0.00518014) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.35832 ± 0.00475977 (W 0.00325974) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0108205 ± 6.37016e-05 (W 3.92536e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0138545 ± 0.000100137 (W 8.60913e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.050815 ± 0.00531262 (W 0.0014587) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.059071 ± 0.000430866 (W 0.000755044) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.145384 ± 0.000710821 (W 0.00132322) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.163374 ± 0.00110915 (W 0.00293332) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0082855 ± 2.88213e-05 (W 6.49648e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00832 ± 5.76113e-05 (W 5.05955e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0406305 ± 0.000199991 (W 9.19845e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.040703 ± 0.000205931 (W 0.000144736) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.121477 ± 0.000696537 (W 0.000375889) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.121007 ± 0.000561712 (W 0.000145335) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006632 ± 5.14215e-05 (W 5.84236e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008646 ± 5.95396e-05 (W 0.000100278) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0236885 ± 0.00018533 (W 6.98734e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0303415 ± 0.000263516 (W 0.000201492) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0558835 ± 0.000172084 (W 0.000146388) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.078511 ± 0.000678226 (W 0.00180795) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.03618 ± 0.000129095 (W 0.000146778) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.031719 ± 0.000155892 (W 0.000160581) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.106519 ± 0.000470056 (W 0.00159242) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.115733 ± 0.000496665 (W 0.00100516) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.233893 ± 0.002038 (W 0.00121547) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.294186 ± 0.00245464 (W 0.00261465) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.024249 ± 8.85769e-05 (W 8.57375e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0685725 ± 0.000206419 (W 0.000780056) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.139812 ± 0.000789284 (W 0.00128569) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006897 ± 7.30041e-05 (W 5.18656e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0097635 ± 9.76586e-05 (W 3.83461e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0222365 ± 0.000109943 (W 7.45796e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0316795 ± 0.000242592 (W 0.000130187) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0507895 ± 0.000204468 (W 0.000146061) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.079286 ± 0.00194099 (W 0.00116792) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.166254 ± 0.0008775 (W 0.00133628) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.168765 ± 0.00127015 (W 0.00144395) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.67508 ± 0.00331287 (W 0.00178151) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.679308 ± 0.0168009 (W 0.00272737) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.5531 ± 0.00663262 (W 0.0047021) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55929 ± 0.00838741 (W 0.00416938) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.057893 ± 0.000285599 (W 0.000226211) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0587245 ± 0.000395853 (W 0.000144544) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.207851 ± 0.00112835 (W 0.00154012) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.207827 ± 0.00104631 (W 0.00156908) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.477345 ± 0.00138294 (W 0.00207164) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.478384 ± 0.00292424 (W 0.00242403) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000756 ± 3.19416e-05 (W 4.19689e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0016835 ± 3.20026e-05 (W 4.31861e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.002985 ± 2.37543e-05 (W 5.69079e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003917 ± 0.000124826 (W 4.29353e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0033615 ± 0.000140766 (W 4.10512e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0136805 ± 0.000382945 (W 7.18469e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012238 ± 0.000278395 (W 8.51793e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.031494 ± 0.000629486 (W 9.76145e-05) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.028989 ± 0.000494498 (W 0.000121307) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0761125 ± 0.000159325 (W 0.000273802) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.077375 ± 0.000843419 (W 0.00177487) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.315446 ± 0.000971053 (W 0.00136624) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.326465 ± 0.00120603 (W 0.00243138) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.770959 ± 0.00262249 (W 0.00244414) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.770138 ± 0.00398927 (W 0.00232998) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0119065 ± 0.0002103 (W 4.95182e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0152325 ± 0.000220216 (W 9.11957e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.05388 ± 0.000604501 (W 0.000151703) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0616395 ± 0.000920862 (W 0.00020817) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.150578 ± 0.00155844 (W 0.00103845) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.173603 ± 0.0013201 (W 0.00226234) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006958 ± 0.00012969 (W 0.000368159) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006146 ± 5.54641e-05 (W 0.000137272) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.020833 ± 0.000383823 (W 0.000384093) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.021003 ± 0.000484491 (W 0.000515764) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.046542 ± 0.000438437 (W 0.000534887) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.046983 ± 0.000278134 (W 0.000443442) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006793 ± 8.95113e-05 (W 9.10115e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008882 ± 6.99478e-05 (W 0.000106511) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0299705 ± 0.000782392 (W 0.000784582) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.030877 ± 0.000156543 (W 0.000210478) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.067872 ± 0.000477949 (W 0.000982721) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0739535 ± 0.00067305 (W 0.00178901) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0374225 ± 0.000169454 (W 0.00015363) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0333365 ± 0.000169585 (W 0.000173971) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.106508 ± 0.000443706 (W 0.000483401) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.110535 ± 0.000550911 (W 0.000420954) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.206532 ± 0.000614531 (W 0.00127222) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.247287 ± 0.00123504 (W 0.00270273) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0250665 ± 8.56662e-05 (W 0.000124899) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0754965 ± 0.000386494 (W 0.000325171) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.148027 ± 0.000403135 (W 0.000844597) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007514 ± 0.000118633 (W 6.05566e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.010309 ± 5.78538e-05 (W 6.49974e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0266895 ± 0.000369515 (W 0.000500197) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0331705 ± 0.000163354 (W 0.000106221) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.055328 ± 0.000569217 (W 0.000784259) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0789915 ± 0.000322512 (W 0.00238941) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.169693 ± 0.0013319 (W 0.00101076) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.172293 ± 0.00197028 (W 0.00139505) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.552913 ± 0.00274405 (W 0.00184267) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.557772 ± 0.00267463 (W 0.00160649) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.13266 ± 0.00371542 (W 0.00485039) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.13816 ± 0.00247951 (W 0.00187031) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058965 ± 0.00022021 (W 0.000208841) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.059702 ± 0.000241473 (W 0.000193165) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.214443 ± 0.00107687 (W 0.00118712) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.218266 ± 0.00166304 (W 0.00125433) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.516387 ± 0.00254735 (W 0.00265512) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.516233 ± 0.00249477 (W 0.00146876) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0009515 ± 1.53677e-05 (W 3.18683e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002129 ± 3.39116e-05 (W 3.80132e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0036715 ± 5.10702e-05 (W 4.41262e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003827 ± 0.000141613 (W 2.93901e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003281 ± 0.000152408 (W 5.24038e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.013641 ± 0.000472191 (W 7.72079e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.012423 ± 0.000370049 (W 9.08576e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0318885 ± 0.000739682 (W 9.58276e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.029344 ± 0.000182725 (W 0.000107689) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0717005 ± 0.000538512 (W 0.000224501) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.072735 ± 0.00139699 (W 0.000189007) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.234493 ± 0.00172133 (W 0.00178712) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.248138 ± 0.00125287 (W 0.00320597) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.498484 ± 0.00273049 (W 0.00199519) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.502072 ± 0.00225523 (W 0.00230069) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011927 ± 0.000154589 (W 4.84229e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.015233 ± 0.000150575 (W 0.000113404) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.052931 ± 0.000399913 (W 0.000247477) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0615145 ± 0.000323195 (W 0.00016417) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.156975 ± 0.00110393 (W 0.00136814) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.18254 ± 0.00110657 (W 0.00188743) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0061965 ± 0.000253576 (W 0.00018949) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0060915 ± 0.000158885 (W 0.000167706) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.016235 ± 0.000169862 (W 0.000436851) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.016431 ± 0.000211148 (W 0.000431638) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0327355 ± 0.000519135 (W 0.000411633) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.034268 ± 0.000308766 (W 0.000602202) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006877 ± 4.54782e-05 (W 5.33835e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008916 ± 7.1379e-05 (W 9.56484e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.03015 ± 0.000600568 (W 0.000819339) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.0308615 ± 0.000259854 (W 0.000186533) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.069289 ± 0.000686792 (W 0.000513167) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.076347 ± 0.00112645 (W 0.00229027) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.037258 ± 0.000186105 (W 0.000145768) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.033361 ± 0.000226752 (W 0.000127265) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.106168 ± 0.000447695 (W 0.00041886) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.108651 ± 0.000522195 (W 0.000430504) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.206662 ± 0.00119585 (W 0.00177738) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.245356 ± 0.00121566 (W 0.00277366) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0250015 ± 0.000245111 (W 0.000119797) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0753725 ± 0.000503728 (W 0.000301994) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.140894 ± 0.000834523 (W 0.000295362) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007539 ± 3.27109e-05 (W 6.6983e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0103045 ± 5.87245e-05 (W 6.35614e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0264125 ± 0.000121488 (W 0.000384731) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0330615 ± 0.00024611 (W 0.000319769) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0578955 ± 0.00030151 (W 0.000771371) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.081927 ± 0.00147994 (W 0.00239184) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.170027 ± 0.00172522 (W 0.00136741) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.170144 ± 0.00104426 (W 0.000961592) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.548747 ± 0.00187957 (W 0.0022327) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.553871 ± 0.00601824 (W 0.00164604) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.1376 ± 0.00385534 (W 0.00217568) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.1501 ± 0.00534254 (W 0.00244195) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0590945 ± 0.000537223 (W 0.000226442) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0599305 ± 0.00039765 (W 0.000291996) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.215082 ± 0.00081181 (W 0.00114414) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.214479 ± 0.000427813 (W 0.00148615) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.53313 ± 0.00117526 (W 0.00148549) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.537195 ± 0.00126269 (W 0.00153177) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000952 ± 1.65005e-05 (W 2.63586e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002149 ± 2.68359e-05 (W 4.32973e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003697 ± 5.64801e-05 (W 5.4078e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.002685 ± 0.000166662 (W 3.1609e-05) | 0.0025 ± 1.69588e-05 (W 4.04271e-05) | 🔴 faer-fork 1.07× slower than `lapack-aocl` | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003336 ± 0.000143529 (W 7.46632e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0099785 ± 0.000401524 (W 6.96916e-05) | 0.013796 ± 0.00189653 (W 7.56755e-05) | 🔵 faer-fork 1.38× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0118015 ± 0.00035456 (W 0.000138938) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0248195 ± 0.000426433 (W 7.32781e-05) | 0.0268035 ± 0.000102685 (W 0.000115309) | 🔵 faer-fork 1.08× faster than `lapack-aocl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.028266 ± 0.000397913 (W 0.00024093) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.111548 ± 0.000571455 (W 0.00118217) | 0.104769 ± 0.000512201 (W 0.000306062) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.111502 ± 0.000696512 (W 0.0010361) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.568782 ± 0.00308037 (W 0.00229332) | 0.403363 ± 0.00180396 (W 0.00186011) | 🔴 faer-fork 1.41× slower than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.57341 ± 0.00271793 (W 0.00188827) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.33604 ± 0.00618992 (W 0.00221521) | 0.957796 ± 0.00456314 (W 0.000836383) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 1.35925 ± 0.00530978 (W 0.00403544) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.010804 ± 0.000119584 (W 6.26397e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0120125 ± 0.000122117 (W 9.98204e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0508395 ± 0.00554727 (W 0.000132229) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.052399 ± 0.000375668 (W 0.000649336) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.144704 ± 0.000904747 (W 0.00135097) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.148863 ± 0.000739276 (W 0.00228598) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0083955 ± 7.31867e-05 (W 6.877e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0084055 ± 2.791e-05 (W 3.11632e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0403 ± 0.000157183 (W 9.56762e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0403675 ± 0.000163935 (W 9.14045e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.122699 ± 0.000394913 (W 0.000180366) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.123465 ± 0.00098784 (W 0.000744502) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0053595 ± 4.3453e-05 (W 0.000100414) | 0.004653 ± 3.78453e-05 (W 4.67419e-05) | 🔴 faer-fork 1.15× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006848 ± 9.04301e-05 (W 8.23665e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0210275 ± 9.0759e-05 (W 0.000207146) | 0.029034 ± 0.000610826 (W 0.000104724) | 🔵 faer-fork 1.38× faster than `lapack-aocl` | `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.0243045 ± 0.000254604 (W 0.000166383) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0498595 ± 0.000283274 (W 0.000264971) | 0.0776245 ± 0.000299399 (W 0.000211928) | 🔵 faer-fork 1.56× faster than `lapack-aocl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0620155 ± 0.000376347 (W 0.00236454) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.027271 ± 0.0002094 (W 0.000152782) | 0.016145 ± 0.000146822 (W 5.82655e-05) | 🔴 faer-fork 1.69× slower than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.0297455 ± 9.14427e-05 (W 0.000191121) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.102636 ± 0.000690001 (W 0.00081558) | 0.0984845 ± 0.000585405 (W 0.00103554) | · |  |
| `qr/n64/b1 [api-inclusive]` | 0.107966 ± 0.000567998 (W 0.00122266) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.238954 ± 0.000752358 (W 0.00177848) | 0.341335 ± 0.00148862 (W 0.00202794) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.277767 ± 0.000891176 (W 0.00303615) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.016174 ± 0.000169626 (W 8.79275e-05) | 0.0074335 ± 0.000114053 (W 4.80301e-05) | 🔴 faer-fork 2.18× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067426 ± 0.000173177 (W 0.00028853) | 0.041056 ± 0.00063076 (W 0.000116635) | 🔴 faer-fork 1.64× slower than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.145453 ± 0.00095912 (W 0.00129371) | 0.137402 ± 0.00131441 (W 0.00110349) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `solve/n32/b1` | 0.0061765 ± 6.01554e-05 (W 5.20995e-05) | 0.0047885 ± 7.2233e-05 (W 6.36802e-05) | 🔴 faer-fork 1.29× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007654 ± 4.92494e-05 (W 5.75849e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0213605 ± 0.000131714 (W 7.80537e-05) | 0.026109 ± 0.000273682 (W 5.08894e-05) | 🔵 faer-fork 1.22× faster than `lapack-aocl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.024877 ± 0.000115219 (W 8.16282e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0503245 ± 0.000176492 (W 0.000142224) | 0.070716 ± 0.000488227 (W 0.000698621) | 🔵 faer-fork 1.41× faster than `lapack-aocl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.0644455 ± 0.000244198 (W 0.00195015) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.170018 ± 0.00247009 (W 0.00159685) | 0.168832 ± 0.000640089 (W 0.00158747) | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.170592 ± 0.000943866 (W 0.00168873) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.68873 ± 0.00491731 (W 0.00193342) | 0.635721 ± 0.00282043 (W 0.0023209) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.689532 ± 0.00916922 (W 0.00271926) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.57391 ± 0.00642942 (W 0.00420044) | 1.50282 ± 0.00641515 (W 0.00351737) | · |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.57191 ± 0.00562013 (W 0.0023395) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0573975 ± 0.000253729 (W 0.000124822) | 0.119857 ± 0.000356946 (W 0.00136396) | 🔵 faer-fork 2.09× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.057828 ± 0.000348628 (W 0.000192031) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.209867 ± 0.00113397 (W 0.0010791) | 0.239481 ± 0.00203978 (W 0.00142166) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.209614 ± 0.00194534 (W 0.00134199) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.486073 ± 0.0027483 (W 0.0017977) | 0.599752 ± 0.00402993 (W 0.00180196) | 🔵 faer-fork 1.23× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.486494 ± 0.00333854 (W 0.00177763) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0007965 ± 7.93515e-06 (W 4.09781e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0019535 ± 2.66114e-05 (W 4.63602e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003421 ± 2.14515e-05 (W 4.02328e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0026945 ± 0.000290448 (W 4.03183e-05) | 0.002495 ± 4.50585e-05 (W 4.61455e-05) | · | `!` between-start range over 5%; `p` position effect over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003552 ± 0.000115748 (W 6.27766e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010134 ± 0.00050548 (W 6.30356e-05) | 0.0137655 ± 0.000174553 (W 6.58506e-05) | 🔵 faer-fork 1.36× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0124325 ± 0.00031603 (W 0.000100387) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.024454 ± 0.000490166 (W 8.0456e-05) | 0.026895 ± 0.000234265 (W 0.000220543) | 🔵 faer-fork 1.10× faster than `lapack-aocl` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.028062 ± 0.000140362 (W 9.71728e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.076397 ± 0.000939838 (W 0.000298094) | 0.105335 ± 0.000348439 (W 0.000706189) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.077275 ± 0.000658689 (W 0.000253423) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.314716 ± 0.00146412 (W 0.0016425) | 0.407902 ± 0.00302998 (W 0.00195755) | 🔵 faer-fork 1.30× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.325947 ± 0.00217035 (W 0.00317648) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.769746 ± 0.00169725 (W 0.00250619) | 0.963448 ± 0.00433218 (W 0.00270016) | 🔵 faer-fork 1.25× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.770619 ± 0.00216529 (W 0.00305132) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0120575 ± 0.000273047 (W 5.37588e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.013545 ± 0.00022665 (W 9.57954e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0535155 ± 0.000617349 (W 0.000215591) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0554085 ± 0.000790546 (W 9.5038e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.150978 ± 0.00178235 (W 0.00109901) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.158857 ± 0.00107519 (W 0.00114409) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0063815 ± 9.81844e-05 (W 0.000148612) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006377 ± 0.000116024 (W 0.000118618) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0192765 ± 0.000135593 (W 0.000238831) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.019371 ± 0.000255183 (W 0.000156739) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.04424 ± 0.00027218 (W 0.000256146) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0448135 ± 0.000246124 (W 0.000222538) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005605 ± 5.64284e-05 (W 6.57875e-05) | 0.004653 ± 5.23412e-05 (W 6.26814e-05) | 🔴 faer-fork 1.20× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0070975 ± 8.04463e-05 (W 8.3067e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.02145 ± 0.000448576 (W 0.000180337) | 0.0278805 ± 0.000484517 (W 0.000143259) | 🔵 faer-fork 1.30× faster than `lapack-aocl` | `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.0244495 ± 0.000281162 (W 0.000254052) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0439545 ± 0.000209549 (W 0.000274583) | 0.072791 ± 0.000362302 (W 0.000964083) | 🔵 faer-fork 1.66× faster than `lapack-aocl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.059463 ± 0.000611304 (W 0.00225788) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0288285 ± 0.000161746 (W 0.000137707) | 0.0160995 ± 7.02322e-05 (W 5.32075e-05) | 🔴 faer-fork 1.79× slower than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.0315545 ± 0.000158136 (W 0.000162038) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0953125 ± 0.000778407 (W 0.000489902) | 0.0980735 ± 0.000878354 (W 0.000180816) | · |  |
| `qr/n64/b1 [api-inclusive]` | 0.102302 ± 0.000552159 (W 0.000332058) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.18469 ± 0.000965214 (W 0.00175818) | 0.3406 ± 0.00141091 (W 0.0019474) | 🔵 faer-fork 1.84× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.230132 ± 0.000961331 (W 0.00232907) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.016931 ± 0.000121169 (W 0.00010811) | 0.0073885 ± 4.21141e-05 (W 4.2316e-05) | 🔴 faer-fork 2.29× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067485 ± 0.000402571 (W 0.00041603) | 0.0408915 ± 0.000188322 (W 0.00018275) | 🔴 faer-fork 1.65× slower than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.131691 ± 0.000769174 (W 0.000783429) | 0.137004 ± 0.000632856 (W 0.00150309) | · |  |
| `solve/n32/b1` | 0.006742 ± 0.000106556 (W 5.76517e-05) | 0.004864 ± 6.73023e-05 (W 5.9074e-05) | 🔴 faer-fork 1.39× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0082105 ± 7.8867e-05 (W 6.16089e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0229265 ± 0.000134756 (W 0.000151876) | 0.0254975 ± 0.000622041 (W 6.34746e-05) | 🔵 faer-fork 1.11× faster than `lapack-aocl` | `!` between-start range over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.026374 ± 0.000244584 (W 0.000156701) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0473335 ± 0.000272749 (W 0.000196338) | 0.066243 ± 0.000356347 (W 0.000191304) | 🔵 faer-fork 1.40× faster than `lapack-aocl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.064496 ± 0.000440859 (W 0.00216437) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.171904 ± 0.000611596 (W 0.000323838) | 0.168571 ± 0.0012254 (W 0.00152729) | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.17196 ± 0.000985258 (W 0.000619928) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.556549 ± 0.00187098 (W 0.00324104) | 0.630392 ± 0.00387521 (W 0.00318392) | 🔵 faer-fork 1.13× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.561089 ± 0.00321912 (W 0.00184124) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.14796 ± 0.00553733 (W 0.00213762) | 1.46554 ± 0.00773919 (W 0.0049673) | 🔵 faer-fork 1.28× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.1521 ± 0.00671188 (W 0.0034637) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0583235 ± 0.000442473 (W 0.000141921) | 0.119645 ± 0.000574489 (W 0.00140747) | 🔵 faer-fork 2.05× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.059235 ± 0.000357251 (W 0.000134309) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.216744 ± 0.00145477 (W 0.00111688) | 0.241895 ± 0.00160511 (W 0.00183783) | 🔵 faer-fork 1.12× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.220167 ± 0.000923731 (W 0.00142659) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.523521 ± 0.00241519 (W 0.00209262) | 0.597823 ± 0.00233451 (W 0.00180872) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.526587 ± 0.00248506 (W 0.00171283) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000982 ± 1.24967e-05 (W 3.88729e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002344 ± 1.18152e-05 (W 4.11403e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004013 ± 3.09559e-05 (W 5.51227e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0022645 ± 0.000250009 (W 3.30324e-05) | 0.002495 ± 2.8849e-05 (W 5.4852e-05) | · | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.003166 ± 0.000185529 (W 6.12651e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.010574 ± 0.000306032 (W 7.82806e-05) | 0.0137905 ± 0.000127055 (W 5.50347e-05) | 🔵 faer-fork 1.30× faster than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0122625 ± 0.000284219 (W 0.000110732) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.024465 ± 0.000699035 (W 8.19473e-05) | 0.026865 ± 0.0016065 (W 0.000154109) | 🔵 faer-fork 1.10× faster than `lapack-aocl` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.028057 ± 0.000539501 (W 0.000158892) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.071761 ± 0.000727952 (W 0.000372218) | 0.106589 ± 0.000679916 (W 0.000553443) | 🔵 faer-fork 1.49× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.072601 ± 0.000349348 (W 0.000261432) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.234835 ± 0.00143614 (W 0.00224576) | 0.407213 ± 0.00150677 (W 0.00212627) | 🔵 faer-fork 1.73× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.24719 ± 0.00150343 (W 0.00213484) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.498958 ± 0.00265868 (W 0.00208137) | 0.966602 ± 0.00227382 (W 0.00283934) | 🔵 faer-fork 1.94× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.500828 ± 0.00199162 (W 0.0020733) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0119115 ± 0.000251395 (W 5.83845e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0133295 ± 0.000115486 (W 0.000103158) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.052732 ± 0.00035723 (W 0.000124861) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0551475 ± 0.000366403 (W 0.000142579) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.157087 ± 0.000429774 (W 0.001368) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.168511 ± 0.000409696 (W 0.00200376) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005866 ± 4.60757e-05 (W 0.000118682) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.005851 ± 8.09214e-05 (W 8.37345e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0145115 ± 8.92726e-05 (W 0.000146171) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014687 ± 0.000219066 (W 0.000210315) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0301155 ± 0.000133646 (W 0.000259795) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.030607 ± 0.000229793 (W 0.000307016) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0056005 ± 6.06545e-05 (W 6.11547e-05) | 0.0046535 ± 4.48806e-05 (W 6.98462e-05) | 🔴 faer-fork 1.20× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0069675 ± 5.13176e-05 (W 8.68135e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0216745 ± 0.000362886 (W 0.000208015) | 0.028742 ± 0.000501675 (W 6.10959e-05) | 🔵 faer-fork 1.33× faster than `lapack-aocl` |  |
| `lu/n64/b1 [api-inclusive]` | 0.02453 ± 0.000237669 (W 0.000246154) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.044202 ± 0.000371696 (W 0.000686611) | 0.072928 ± 0.000409392 (W 0.000148926) | 🔵 faer-fork 1.65× faster than `lapack-aocl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.0618875 ± 0.000742519 (W 0.0012725) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.028567 ± 0.000160905 (W 0.000149548) | 0.015865 ± 0.00012614 (W 6.322e-05) | 🔴 faer-fork 1.80× slower than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.0314335 ± 0.000154483 (W 0.000138289) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0945415 ± 0.000462642 (W 0.000428186) | 0.0993435 ± 0.000521267 (W 0.000174907) | · |  |
| `qr/n64/b1 [api-inclusive]` | 0.102028 ± 0.000304496 (W 0.000501253) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.179605 ± 0.00141215 (W 0.00185589) | 0.340427 ± 0.00162738 (W 0.00250035) | 🔵 faer-fork 1.90× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.226271 ± 0.00136969 (W 0.00272795) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.016966 ± 0.000143413 (W 0.000128915) | 0.0073885 ± 0.000192434 (W 3.29162e-05) | 🔴 faer-fork 2.30× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.067027 ± 0.000387839 (W 0.000271302) | 0.041196 ± 0.000200339 (W 0.00012466) | 🔴 faer-fork 1.63× slower than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.129352 ± 0.000573487 (W 0.000352516) | 0.136258 ± 0.000483943 (W 0.000456599) | 🔵 faer-fork 1.05× faster than `lapack-aocl` |  |
| `solve/n32/b1` | 0.006767 ± 0.00024721 (W 5.61792e-05) | 0.004974 ± 4.60435e-05 (W 3.77325e-05) | 🔴 faer-fork 1.36× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.008265 ± 5.45909e-05 (W 4.5932e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.023464 ± 0.000142501 (W 0.000188958) | 0.0268195 ± 0.000321156 (W 8.62631e-05) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.0259635 ± 6.29997e-05 (W 8.70028e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0478795 ± 0.000303926 (W 0.000207169) | 0.066363 ± 0.000697656 (W 0.00149925) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.06689 ± 0.00189936 (W 0.00177677) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1` | 0.172396 ± 0.00147946 (W 0.000521991) | 0.167796 ± 0.000610743 (W 0.00158707) | · |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.171922 ± 0.000624293 (W 0.00117719) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.551752 ± 0.00202978 (W 0.00227019) | 0.635493 ± 0.00251724 (W 0.00299358) | 🔵 faer-fork 1.15× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.554736 ± 0.0695825 (W 0.00197753) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n96/b1` | 1.15248 ± 0.00600239 (W 0.00470998) | 1.49472 ± 0.00672711 (W 0.00443138) | 🔵 faer-fork 1.30× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.16362 ± 0.00177494 (W 0.00284077) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058589 ± 0.000264044 (W 0.000180294) | 0.119498 ± 0.000495636 (W 0.000137264) | 🔵 faer-fork 2.04× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.059271 ± 0.00033763 (W 0.000164983) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.217144 ± 0.000925035 (W 0.00133441) | 0.241271 ± 0.00152983 (W 0.00171547) | 🔵 faer-fork 1.11× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.216147 ± 0.00114387 (W 0.00155752) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.541115 ± 0.00199988 (W 0.00149758) | 0.597336 ± 0.00237086 (W 0.00211339) | 🔵 faer-fork 1.10× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.542733 ± 0.00857791 (W 0.00202563) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0009765 ± 1.28167e-05 (W 3.24194e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0023295 ± 3.17847e-05 (W 3.77829e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003972 ± 2.8266e-05 (W 5.42463e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.006152 ± 5.60978e-05 (W 3.17718e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00538 ± 3.85768e-05 (W 5.99194e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.033197 ± 0.000179385 (W 0.000161098) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0315285 ± 0.000178967 (W 0.000110288) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.088809 ± 0.000540259 (W 0.0003096) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.086606 ± 0.000199323 (W 0.000210023) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.139287 ± 0.0016746 (W 0.000824007) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.140386 ± 0.00369843 (W 0.00104307) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.73197 ± 0.00319725 (W 0.0018491) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.731992 ± 0.00404141 (W 0.00182756) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.7811 ± 0.00556926 (W 0.00403517) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.78697 ± 0.00681972 (W 0.00339741) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017045 ± 9.456e-05 (W 6.67869e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.020542 ± 0.000103407 (W 0.000140729) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.0979755 ± 0.000692936 (W 0.000461935) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.107629 ± 0.000682119 (W 0.00047285) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.286233 ± 0.000880148 (W 0.00187843) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.349171 ± 0.00230555 (W 0.00235295) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0286585 ± 0.000218237 (W 0.000566222) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0284465 ± 0.000190894 (W 0.000113011) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.160446 ± 0.000679879 (W 0.00153123) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.159937 ± 0.00058452 (W 0.00152737) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.470843 ± 0.00302477 (W 0.00111118) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.470403 ± 0.00253038 (W 0.000788606) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0126785 ± 9.78334e-05 (W 8.3729e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.014927 ± 0.000137562 (W 7.88142e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.054656 ± 0.0003579 (W 0.00011733) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0624915 ± 0.000296799 (W 0.000367009) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.141398 ± 0.000490328 (W 0.000961588) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.207846 ± 0.00186507 (W 0.00215495) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0734375 ± 0.000623128 (W 0.000350033) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.068208 ± 0.000322005 (W 0.000448185) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.291482 ± 0.00112685 (W 0.00163893) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.3322 ± 0.00175043 (W 0.00298393) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.71316 ± 0.00355918 (W 0.00191123) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.843108 ± 0.00520614 (W 0.00219388) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.041137 ± 0.000255538 (W 0.000170106) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.145266 ± 0.00176726 (W 0.00142954) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.34435 ± 0.000870405 (W 0.00166606) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0127435 ± 6.61858e-05 (W 8.79127e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0161655 ± 0.000146176 (W 5.57013e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0522125 ± 0.000329273 (W 6.87287e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.065147 ± 0.000341513 (W 0.000136259) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.135798 ± 0.000582367 (W 0.00143881) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.214093 ± 0.00252093 (W 0.00324469) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.219695 ± 0.00154558 (W 0.00141332) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.220851 ± 0.00990875 (W 0.00148417) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 1.00969 ± 0.00417416 (W 0.00197925) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 1.01067 ± 0.0045136 (W 0.00120913) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.42989 ± 0.0166653 (W 0.00366347) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 2.4338 ± 0.0100705 (W 0.00598754) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0731415 ± 0.000372072 (W 0.0014699) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0734475 ± 0.000321061 (W 0.000320138) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.30749 ± 0.000791817 (W 0.00154667) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.307505 ± 0.0012742 (W 0.00193965) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.778925 ± 0.00321202 (W 0.00219596) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.781208 ± 0.00380308 (W 0.00356982) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001182 ± 1.04003e-05 (W 3.81158e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002925 ± 2.33559e-05 (W 6.02649e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005325 ± 3.35201e-05 (W 4.99347e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0061565 ± 3.98781e-05 (W 3.98736e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005405 ± 5.17687e-05 (W 5.4688e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0306625 ± 0.000201953 (W 0.000144231) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0301955 ± 0.000134259 (W 0.000132269) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0834805 ± 0.000344487 (W 0.000227596) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.082759 ± 0.000492372 (W 0.000423243) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.101865 ± 0.00237716 (W 0.000389686) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1 [api-inclusive]` | 0.102165 ± 0.000559243 (W 0.000412293) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.430602 ± 0.00177339 (W 0.00184036) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.43187 ± 0.00183563 (W 0.00201192) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.920759 ± 0.00352403 (W 0.00126138) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.92369 ± 0.00471868 (W 0.00232294) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0180485 ± 0.000132668 (W 5.51063e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.02152 ± 0.000291464 (W 0.000119167) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.098394 ± 0.000434682 (W 0.000628799) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.108126 ± 0.000667705 (W 0.000775166) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.23046 ± 0.0010687 (W 0.00215047) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.30177 ± 0.00130758 (W 0.00179271) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.015148 ± 0.000167127 (W 0.000204899) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0145865 ± 0.000186494 (W 0.000195074) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.062469 ± 0.000605186 (W 0.000462491) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.06171 ± 0.00044601 (W 0.000370756) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.15343 ± 0.000961314 (W 0.00136296) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.15354 ± 0.00123951 (W 0.000640902) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0129785 ± 0.000133283 (W 8.68055e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.015178 ± 9.5215e-05 (W 0.000120553) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.061486 ± 0.000647809 (W 0.000676023) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.062031 ± 0.00139021 (W 0.000838717) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.138589 ± 0.00119289 (W 0.0014275) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.190428 ± 0.00164615 (W 0.00227129) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.070353 ± 0.000479391 (W 0.000606575) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.063724 ± 0.000196866 (W 0.000592609) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.190934 ± 0.00101194 (W 0.00156507) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.225046 ± 0.00168394 (W 0.00192157) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.417034 ± 0.00113349 (W 0.00175474) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.515362 ± 0.00142126 (W 0.00314089) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0417135 ± 0.000287986 (W 0.000459622) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.126434 ± 0.000687251 (W 0.000560583) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.26967 ± 0.00165805 (W 0.0017624) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.013344 ± 0.000101643 (W 5.87189e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0169265 ± 4.93221e-05 (W 7.59626e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0554725 ± 0.000875477 (W 0.000322719) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0657275 ± 0.000937039 (W 0.00107162) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.112736 ± 0.00071308 (W 0.000487158) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.192668 ± 0.00229595 (W 0.00306472) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.215404 ± 0.00109218 (W 0.0011936) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.217014 ± 0.00167199 (W 0.00147583) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.736576 ± 0.00124695 (W 0.0018983) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.740396 ± 0.00260528 (W 0.00244088) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.54935 ± 0.00776827 (W 0.00357565) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55622 ± 0.0053401 (W 0.00356756) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0746835 ± 0.000467367 (W 0.000173074) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.075386 ± 0.000548005 (W 0.000196652) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.315133 ± 0.00182823 (W 0.00179434) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.315756 ± 0.00167589 (W 0.00154467) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.717766 ± 0.00199304 (W 0.00184518) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.717972 ± 0.00170589 (W 0.00252804) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0014975 ± 2.03707e-05 (W 3.93508e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003522 ± 2.68359e-05 (W 6.5558e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0061065 ± 6.3216e-05 (W 5.41672e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.006131 ± 4.77511e-05 (W 3.99995e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0053295 ± 3.02021e-05 (W 6.04902e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.030988 ± 0.000117717 (W 0.000128968) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0301365 ± 0.000134974 (W 0.000148428) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.083793 ± 0.000349592 (W 0.000346776) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.0829875 ± 0.000217094 (W 0.000235449) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0973205 ± 0.00052556 (W 0.000315898) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0978815 ± 0.00132243 (W 0.000386466) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.325654 ± 0.00520841 (W 0.00704428) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.317089 ± 0.00595673 (W 0.00622633) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.68606 ± 0.00278053 (W 0.00273305) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.691252 ± 0.002505 (W 0.00277256) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0178535 ± 0.000310243 (W 0.000121676) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0214545 ± 4.25696e-05 (W 9.15136e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1` | 0.102865 ± 0.000475806 (W 0.000771862) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.112413 ± 0.00141724 (W 0.000690878) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.233024 ± 0.00174577 (W 0.00167429) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.30876 ± 0.00241117 (W 0.00204518) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0114965 ± 0.000215809 (W 0.000258316) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.01082 ± 5.90582e-05 (W 0.000153554) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0376355 ± 0.000305216 (W 0.000523309) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0375895 ± 0.000236376 (W 0.000359727) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.087999 ± 0.00882572 (W 0.000840115) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0869785 ± 0.000496183 (W 0.000984716) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0129335 ± 8.64749e-05 (W 7.66853e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0150475 ± 9.01835e-05 (W 0.000121402) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0642045 ± 0.000945233 (W 0.000534887) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.062248 ± 0.000346892 (W 0.000277569) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.146331 ± 0.00089051 (W 0.00109084) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.188554 ± 0.00135937 (W 0.00180536) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.072169 ± 0.000403146 (W 0.000901904) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.063013 ± 0.000503437 (W 0.000335446) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.17504 ± 0.000931889 (W 0.00101933) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.210427 ± 0.00210419 (W 0.00339613) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.352055 ± 0.00207871 (W 0.00220072) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.473306 ± 0.00157616 (W 0.00446475) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0438365 ± 0.000275298 (W 0.000791311) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.122449 ± 0.000727653 (W 0.000420442) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.244991 ± 0.00186497 (W 0.00165834) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.01345 ± 8.62415e-05 (W 6.58609e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.016961 ± 4.01497e-05 (W 5.11391e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0576515 ± 0.000413566 (W 0.000351079) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.066508 ± 0.000329282 (W 0.000454355) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.119735 ± 0.000590024 (W 0.000776261) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.190118 ± 0.000645309 (W 0.00313086) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.214979 ± 0.00682556 (W 0.00185137) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.216453 ± 0.00802243 (W 0.00152074) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n64/b1` | 0.723201 ± 0.00366138 (W 0.00223538) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.725167 ± 0.00394653 (W 0.00151076) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.4911 ± 0.00887878 (W 0.00316587) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.49729 ± 0.0404212 (W 0.00504475) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n32/b1` | 0.074654 ± 0.000602168 (W 0.000121656) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.075103 ± 0.000503531 (W 0.000131432) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.324426 ± 0.00246383 (W 0.00153261) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.325504 ± 0.00420324 (W 0.0012757) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.742599 ± 0.0016345 (W 0.00271919) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.741968 ± 0.00394238 (W 0.00233579) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0014775 ± 2.03052e-05 (W 3.25252e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003491 ± 4.07709e-05 (W 6.26468e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0061065 ± 5.22073e-05 (W 7.10621e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004568 ± 4.11291e-05 (W 3.77682e-05) | 0.0496895 ± 0.000221162 (W 0.000224574) | 🔵 faer-fork 10.88× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005395 ± 3.04001e-05 (W 6.82213e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0281325 ± 0.000167212 (W 9.27615e-05) | 0.095743 ± 0.000515903 (W 0.000824287) | 🔵 faer-fork 3.40× faster than `lapack-aocl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.031156 ± 0.000186597 (W 0.000111246) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.079072 ± 0.000285972 (W 0.000154001) | 0.172836 ± 0.00129645 (W 0.00171198) | 🔵 faer-fork 2.19× faster than `lapack-aocl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.0859565 ± 0.000423761 (W 0.000120442) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.13926 ± 0.000763377 (W 0.000411397) | 0.153148 ± 0.0014742 (W 0.000353565) | 🔵 faer-fork 1.10× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.139876 ± 0.00312603 (W 0.000399566) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1` | 0.731483 ± 0.0028823 (W 0.00216949) | 0.713477 ± 0.00333959 (W 0.00277328) | · |  |
| `eigh/n64/b1 [api-inclusive]` | 0.733474 ± 0.00338194 (W 0.00227145) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.78063 ± 0.00613646 (W 0.00350878) | 1.79102 ± 0.0074737 (W 0.00432179) | · |  |
| `eigh/n96/b1 [api-inclusive]` | 1.78764 ± 0.00733943 (W 0.00223474) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0169455 ± 5.93074e-05 (W 8.26588e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018735 ± 0.000107589 (W 7.65001e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.097847 ± 0.000608836 (W 0.00086927) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.101058 ± 0.000599431 (W 0.000680167) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.284564 ± 0.00136773 (W 0.00175624) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.333989 ± 0.00248274 (W 0.00181969) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0284715 ± 0.000176037 (W 0.000119818) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0283225 ± 0.000176811 (W 6.69222e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.158091 ± 0.000899097 (W 0.000966097) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.158838 ± 0.00077987 (W 0.000391472) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.462198 ± 0.00323614 (W 0.0014027) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.464696 ± 0.00294085 (W 0.0015289) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011356 ± 6.68062e-05 (W 0.000113899) | 0.0073185 ± 0.00023412 (W 7.83634e-05) | 🔴 faer-fork 1.55× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0133845 ± 8.47113e-05 (W 0.000134329) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.050015 ± 0.000223859 (W 0.0011749) | 0.0392215 ± 0.00107004 (W 0.000276059) | 🔴 faer-fork 1.28× slower than `lapack-aocl` | `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.0558285 ± 0.000283423 (W 0.000648613) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.136919 ± 0.00081145 (W 0.00125353) | 0.104321 ± 0.000998073 (W 0.000212393) | 🔴 faer-fork 1.31× slower than `lapack-aocl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.194294 ± 0.00139575 (W 0.00257827) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0638695 ± 0.000411986 (W 0.000338678) | 0.0416425 ± 0.00026393 (W 0.00015403) | 🔴 faer-fork 1.53× slower than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.0667585 ± 0.000413654 (W 0.000317496) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.295716 ± 0.00110178 (W 0.00174098) | 0.370578 ± 0.00185063 (W 0.00183696) | 🔵 faer-fork 1.25× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.325697 ± 0.00233408 (W 0.00318602) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.730274 ± 0.00288879 (W 0.00235028) | 0.88738 ± 0.00388526 (W 0.00178305) | 🔵 faer-fork 1.22× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.82887 ± 0.00364907 (W 0.00263426) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.032173 ± 0.000304037 (W 0.000183504) | 0.0204465 ± 0.000152674 (W 8.91271e-05) | 🔴 faer-fork 1.57× slower than `lapack-aocl` |  |
| `qr_factor/n64/b1` | 0.1528 ± 0.000530277 (W 0.00116207) | 0.12134 ± 0.000548301 (W 0.000593668) | 🔴 faer-fork 1.26× slower than `lapack-aocl` |  |
| `qr_factor/n96/b1` | 0.370997 ± 0.00185126 (W 0.00214862) | 0.355673 ± 0.00119967 (W 0.00220726) | · |  |
| `solve/n32/b1` | 0.011988 ± 4.79642e-05 (W 5.29076e-05) | 0.0078895 ± 0.000505754 (W 6.63427e-05) | 🔴 faer-fork 1.52× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.014271 ± 0.00013403 (W 7.14722e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0535545 ± 0.000814169 (W 0.000251307) | 0.0385645 ± 0.00068222 (W 9.54697e-05) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.0584005 ± 0.000207493 (W 0.000516412) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.138688 ± 0.000613729 (W 0.00136171) | 0.101555 ± 0.000511973 (W 0.00160266) | 🔴 faer-fork 1.37× slower than `lapack-aocl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.196966 ± 0.00252147 (W 0.00320979) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.218679 ± 0.0110264 (W 0.00125279) | 0.233183 ± 0.00101717 (W 0.00146375) | 🔵 faer-fork 1.07× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.219293 ± 0.00107215 (W 0.00147259) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.993354 ± 0.00311152 (W 0.000890114) | 0.970252 ± 0.00448023 (W 0.00128935) | · |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.999239 ± 0.00459933 (W 0.00258008) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 2.41494 ± 0.00955135 (W 0.003391) | 2.60409 ± 0.0136545 (W 0.00516277) | 🔵 faer-fork 1.08× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 2.42261 ± 0.0101715 (W 0.00473475) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.074013 ± 0.000853027 (W 0.00112329) | 0.107608 ± 0.000696432 (W 0.00103255) | 🔵 faer-fork 1.45× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.0744645 ± 0.000845172 (W 0.000225327) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.304025 ± 0.00163489 (W 0.00255833) | 0.418181 ± 0.00261496 (W 0.00255696) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.305597 ± 0.00124379 (W 0.00183942) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.776755 ± 0.00317336 (W 0.00154417) | 1.01815 ± 0.0113939 (W 0.00175474) | 🔵 faer-fork 1.31× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.777222 ± 0.00182873 (W 0.00176876) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001217 ± 5.85377e-06 (W 3.83718e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003186 ± 0.00017459 (W 5.07268e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0052645 ± 2.60308e-05 (W 5.15659e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0045435 ± 4.18127e-05 (W 3.70504e-05) | 0.049683 ± 0.000249065 (W 0.00141336) | 🔵 faer-fork 10.93× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.005465 ± 4.08248e-05 (W 5.88008e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0266035 ± 0.000302357 (W 0.000132083) | 0.105932 ± 0.00145785 (W 0.00172274) | 🔵 faer-fork 3.98× faster than `lapack-aocl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.029881 ± 0.000173757 (W 0.000151205) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.0739385 ± 0.000549579 (W 0.000207477) | 0.168389 ± 0.00151317 (W 0.0030153) | 🔵 faer-fork 2.28× faster than `lapack-aocl` |  |
| `cholesky/n96/b1 [api-inclusive]` | 0.081993 ± 0.000395601 (W 0.000473791) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.101238 ± 0.000550052 (W 0.000422955) | 0.153377 ± 0.00132129 (W 0.00148427) | 🔵 faer-fork 1.52× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.10229 ± 0.000247101 (W 0.00027742) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.430253 ± 0.00238771 (W 0.00217872) | 0.72075 ± 0.00469279 (W 0.00259984) | 🔵 faer-fork 1.68× faster than `lapack-aocl` |  |
| `eigh/n64/b1 [api-inclusive]` | 0.431963 ± 0.00217195 (W 0.00193423) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.920454 ± 0.00339366 (W 0.00179151) | 1.82461 ± 0.00960673 (W 0.00381165) | 🔵 faer-fork 1.98× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.922575 ± 0.00528935 (W 0.00192203) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0180335 ± 9.76251e-05 (W 9.47552e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.019802 ± 0.000208474 (W 0.000668286) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0969865 ± 0.000518729 (W 0.000485115) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.101078 ± 0.00128085 (W 0.000597225) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.229955 ± 0.000958892 (W 0.00170865) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.288133 ± 0.00184278 (W 0.00129289) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0152935 ± 0.000161146 (W 0.00019301) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.015113 ± 0.000145869 (W 0.000159171) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0616285 ± 0.000449143 (W 0.000272308) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0615935 ± 0.000313028 (W 0.000249908) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.151112 ± 0.000566191 (W 0.00149816) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.151131 ± 0.000527562 (W 0.00128862) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0115365 ± 0.000121957 (W 9.48463e-05) | 0.0072035 ± 0.000208135 (W 8.79321e-05) | 🔴 faer-fork 1.60× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0136045 ± 8.34026e-05 (W 0.000124683) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0495875 ± 0.000593567 (W 0.000320484) | 0.0348295 ± 0.000975384 (W 0.000360608) | 🔴 faer-fork 1.42× slower than `lapack-aocl` | `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.055078 ± 0.00108779 (W 0.000715412) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.11244 ± 0.00148149 (W 0.000840981) | 0.082082 ± 0.000372164 (W 0.000854203) | 🔴 faer-fork 1.37× slower than `lapack-aocl` |  |
| `lu/n96/b1 [api-inclusive]` | 0.175183 ± 0.00108371 (W 0.00218328) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.058244 ± 0.000277732 (W 0.000414824) | 0.041161 ± 0.000214875 (W 0.000187165) | 🔴 faer-fork 1.42× slower than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.061656 ± 0.000285847 (W 0.000466756) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.177312 ± 0.00104515 (W 0.00113438) | 0.272612 ± 0.00257261 (W 0.00295617) | 🔵 faer-fork 1.54× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.219602 ± 0.00116922 (W 0.00200031) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.386294 ± 0.00225587 (W 0.00200605) | 0.640771 ± 0.00222644 (W 0.00363929) | 🔵 faer-fork 1.66× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.502255 ± 0.0032032 (W 0.00279016) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0310035 ± 0.000180794 (W 0.000272866) | 0.020247 ± 0.000167437 (W 9.74668e-05) | 🔴 faer-fork 1.53× slower than `lapack-aocl` |  |
| `qr_factor/n64/b1` | 0.116992 ± 0.000551759 (W 0.000539428) | 0.121015 ± 0.000759241 (W 0.000286308) | · |  |
| `qr_factor/n96/b1` | 0.247629 ± 0.00181548 (W 0.00128075) | 0.37813 ± 0.00217227 (W 0.00185044) | 🔵 faer-fork 1.53× faster than `lapack-aocl` |  |
| `solve/n32/b1` | 0.0126325 ± 8.90543e-05 (W 5.74454e-05) | 0.007689 ± 3.74188e-05 (W 4.95391e-05) | 🔴 faer-fork 1.64× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0149975 ± 0.000150902 (W 6.99391e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.053208 ± 0.00107794 (W 0.000154929) | 0.032981 ± 0.000759054 (W 0.000388445) | 🔴 faer-fork 1.61× slower than `lapack-aocl` | `!` between-start range over 5% |
| `solve/n64/b1 [api-inclusive]` | 0.059358 ± 0.000904198 (W 0.00053819) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.113384 ± 0.000904013 (W 0.000728285) | 0.0711025 ± 0.000358832 (W 0.00121384) | 🔴 faer-fork 1.59× slower than `lapack-aocl` |  |
| `solve/n96/b1 [api-inclusive]` | 0.178843 ± 0.000596437 (W 0.00282373) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.213203 ± 0.0012979 (W 0.00122087) | 0.232277 ± 0.00136283 (W 0.00185572) | 🔵 faer-fork 1.09× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.214386 ± 0.000954383 (W 0.00146034) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.722093 ± 0.00271783 (W 0.00204994) | 0.972 ± 0.0052327 (W 0.00170357) | 🔵 faer-fork 1.35× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.725584 ± 0.00589852 (W 0.00256047) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.55167 ± 0.006995 (W 0.00267921) | 2.17143 ± 0.00815258 (W 0.00956098) | 🔵 faer-fork 1.40× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.55851 ± 0.00574318 (W 0.00337045) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0757975 ± 0.000643891 (W 0.000132874) | 0.107435 ± 0.000747676 (W 0.00179102) | 🔵 faer-fork 1.42× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.0759375 ± 0.000347724 (W 0.000226175) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.312962 ± 0.000806357 (W 0.00148196) | 0.420955 ± 0.00150958 (W 0.00258836) | 🔵 faer-fork 1.35× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.313295 ± 0.000634661 (W 0.00201536) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.714358 ± 0.00235937 (W 0.00233884) | 1.04181 ± 0.00403339 (W 0.00234374) | 🔵 faer-fork 1.46× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.716796 ± 0.00238163 (W 0.0020203) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001498 ± 2.65631e-05 (W 3.98017e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0035365 ± 3.37086e-05 (W 4.8832e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0064825 ± 0.000105661 (W 5.9492e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.004563 ± 6.71585e-05 (W 3.4174e-05) | 0.049502 ± 0.000254214 (W 0.000181663) | 🔵 faer-fork 10.85× faster than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00536 ± 4.02128e-05 (W 6.04225e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0268 ± 0.000229856 (W 0.000131107) | 0.116608 ± 0.00109578 (W 0.00158126) | 🔵 faer-fork 4.35× faster than `lapack-aocl` |  |
| `cholesky/n64/b1 [api-inclusive]` | 0.0297605 ± 8.27833e-05 (W 0.000123299) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.074299 ± 0.000452336 (W 0.000246024) | 0.187918 ± 0.00380016 (W 0.00370648) | 🔵 faer-fork 2.53× faster than `lapack-aocl` | `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.0822105 ± 0.000265588 (W 0.000241285) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.096978 ± 0.000421553 (W 0.000372487) | 0.153273 ± 0.000834364 (W 0.000195677) | 🔵 faer-fork 1.58× faster than `lapack-aocl` |  |
| `eigh/n32/b1 [api-inclusive]` | 0.097846 ± 0.00100039 (W 0.000295912) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.325778 ± 0.00755197 (W 0.00179439) | 0.735368 ± 0.00408973 (W 0.00641905) | 🔵 faer-fork 2.26× faster than `lapack-aocl` | `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.320365 ± 0.00728072 (W 0.00716788) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.685903 ± 0.00152943 (W 0.00339376) | 1.89903 ± 0.0114748 (W 0.00749977) | 🔵 faer-fork 2.77× faster than `lapack-aocl` |  |
| `eigh/n96/b1 [api-inclusive]` | 0.689685 ± 0.00183207 (W 0.00273054) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.017848 ± 8.15173e-05 (W 6.69501e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0196415 ± 0.000153994 (W 7.48727e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.100287 ± 0.000815368 (W 0.00038576) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.104192 ± 0.000734749 (W 0.000820447) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.231042 ± 0.00215247 (W 0.0012936) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.29183 ± 0.00178971 (W 0.00174503) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0113555 ± 0.000138217 (W 0.000156037) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0113665 ± 0.00115714 (W 0.000133931) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.038286 ± 0.000256626 (W 0.000679113) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0381005 ± 0.000250546 (W 0.000318544) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0860165 ± 0.000956539 (W 0.000906199) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.086323 ± 0.000517756 (W 0.000989476) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.011556 ± 7.85971e-05 (W 0.000109464) | 0.0072435 ± 0.000274805 (W 5.45379e-05) | 🔴 faer-fork 1.60× slower than `lapack-aocl` | `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.013561 ± 0.000132074 (W 0.000104091) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0495755 ± 0.000321967 (W 0.000192321) | 0.03767 ± 0.000655391 (W 0.000438797) | 🔴 faer-fork 1.32× slower than `lapack-aocl` | `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.056195 ± 0.000561583 (W 0.000378887) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.109716 ± 0.000460698 (W 0.000717602) | 0.126198 ± 0.00449333 (W 0.00464485) | 🔵 faer-fork 1.15× faster than `lapack-aocl` | `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.17466 ± 0.00110898 (W 0.00179331) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.058724 ± 0.000392914 (W 0.000427918) | 0.0416865 ± 0.000290998 (W 0.000195259) | 🔴 faer-fork 1.41× slower than `lapack-aocl` |  |
| `qr/n32/b1 [api-inclusive]` | 0.061364 ± 0.000584554 (W 0.000274934) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.165425 ± 0.00148612 (W 0.00141428) | 0.270061 ± 0.00172995 (W 0.00468769) | 🔵 faer-fork 1.63× faster than `lapack-aocl` |  |
| `qr/n64/b1 [api-inclusive]` | 0.203398 ± 0.00182227 (W 0.00289565) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.32819 ± 0.00116817 (W 0.0018176) | 0.667067 ± 0.0039722 (W 0.00728931) | 🔵 faer-fork 2.03× faster than `lapack-aocl` |  |
| `qr/n96/b1 [api-inclusive]` | 0.457338 ± 0.0021967 (W 0.0030618) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.030967 ± 0.000172037 (W 0.000250815) | 0.020377 ± 0.00013187 (W 7.72974e-05) | 🔴 faer-fork 1.52× slower than `lapack-aocl` |  |
| `qr_factor/n64/b1` | 0.113254 ± 0.000607222 (W 0.000541022) | 0.121938 ± 0.00450264 (W 0.00192452) | 🔵 faer-fork 1.08× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr_factor/n96/b1` | 0.229127 ± 0.000690994 (W 0.00164979) | 0.378495 ± 0.00201129 (W 0.00217332) | 🔵 faer-fork 1.65× faster than `lapack-aocl` |  |
| `solve/n32/b1` | 0.012728 ± 9.66866e-05 (W 8.75937e-05) | 0.0079645 ± 7.32951e-05 (W 5.31127e-05) | 🔴 faer-fork 1.60× slower than `lapack-aocl` | dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0150075 ± 0.000123328 (W 6.2914e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.053158 ± 0.000263696 (W 0.000214737) | 0.0349955 ± 0.00037791 (W 0.00044349) | 🔴 faer-fork 1.52× slower than `lapack-aocl` |  |
| `solve/n64/b1 [api-inclusive]` | 0.060723 ± 0.000501272 (W 0.000422862) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.110244 ± 0.000398252 (W 0.000464574) | 0.111499 ± 0.00215878 (W 0.00277823) | · |  |
| `solve/n96/b1 [api-inclusive]` | 0.176563 ± 0.00125674 (W 0.00242307) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.212911 ± 0.00111907 (W 0.00125042) | 0.232851 ± 0.00163889 (W 0.00207052) | 🔵 faer-fork 1.09× faster than `lapack-aocl` |  |
| `svd_thin/n32/b1 [api-inclusive]` | 0.213064 ± 0.00163393 (W 0.001489) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.711388 ± 0.00268151 (W 0.00219372) | 0.98068 ± 0.00864604 (W 0.00156727) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `svd_thin/n64/b1 [api-inclusive]` | 0.713123 ± 0.00392134 (W 0.00224438) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.49072 ± 0.00775023 (W 0.00317026) | 2.25244 ± 0.021844 (W 0.00948428) | 🔵 faer-fork 1.51× faster than `lapack-aocl` |  |
| `svd_thin/n96/b1 [api-inclusive]` | 1.49452 ± 0.00416806 (W 0.00268996) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.075188 ± 0.000517641 (W 0.000144077) | 0.107901 ± 0.000845704 (W 0.000219937) | 🔵 faer-fork 1.44× faster than `lapack-aocl` |  |
| `svd_values/n32/b1 [api-inclusive]` | 0.075724 ± 0.000449862 (W 0.000151297) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.321072 ± 0.00124067 (W 0.0014788) | 0.421469 ± 0.00365793 (W 0.00204725) | 🔵 faer-fork 1.31× faster than `lapack-aocl` |  |
| `svd_values/n64/b1 [api-inclusive]` | 0.322812 ± 0.00120444 (W 0.00166333) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.74297 ± 0.005122 (W 0.00204479) | 1.04468 ± 0.00530073 (W 0.00209959) | 🔵 faer-fork 1.41× faster than `lapack-aocl` |  |
| `svd_values/n96/b1 [api-inclusive]` | 0.741093 ± 0.00311331 (W 0.00274017) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001513 ± 1.16905e-05 (W 4.72565e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0035465 ± 2.49052e-05 (W 6.29728e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.006342 ± 4.12331e-05 (W 4.62045e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003296 ± 0.000120694 (W 3.04119e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00262 ± 1.9532e-05 (W 5.62283e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0137145 ± 8.59341e-05 (W 2.37819e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0101985 ± 5.07963e-05 (W 6.03494e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0260515 ± 0.000659666 (W 8.46772e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.0212185 ± 0.000812544 (W 7.37538e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.0868565 ± 0.000556724 (W 0.00117039) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.087493 ± 0.000626533 (W 0.000230664) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.322272 ± 0.00592028 (W 0.00229846) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.322002 ± 0.00188968 (W 0.00191401) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.95025 ± 0.00669866 (W 0.00177906) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.947898 ± 0.00464061 (W 0.00171286) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.010264 ± 0.000145923 (W 7.75402e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0130735 ± 0.000151967 (W 6.79461e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.041429 ± 0.000318473 (W 9.61865e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.04782 ± 0.000321862 (W 0.00140778) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.102984 ± 0.000875973 (W 0.000700393) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.11483 ± 0.00146113 (W 0.00065483) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0063465 ± 3.76864e-05 (W 8.26561e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.006341 ± 6.48095e-05 (W 4.11639e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.026054 ± 0.000110628 (W 0.000100803) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.026073 ± 0.000107461 (W 5.54962e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.076899 ± 0.000470766 (W 0.000281955) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.076979 ± 0.000332194 (W 0.000191625) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0063665 ± 7.93238e-05 (W 8.64712e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.00792 ± 9.87704e-05 (W 0.00013212) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020268 ± 9.62575e-05 (W 4.94198e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0251315 ± 0.000193018 (W 0.000181078) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0443195 ± 0.00024697 (W 0.000170104) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.055122 ± 0.000364737 (W 0.000233781) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0309875 ± 0.000163299 (W 0.000118672) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.028361 ± 0.000137724 (W 0.000120447) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.089868 ± 0.000458966 (W 0.000247662) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.102618 ± 0.000685272 (W 0.00082752) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.192566 ± 0.000851919 (W 0.00128203) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.240398 ± 0.00096808 (W 0.00184747) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.021195 ± 7.66118e-05 (W 6.5755e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.058504 ± 0.000239375 (W 0.00115515) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.111573 ± 0.000710118 (W 0.00034028) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007113 ± 6.39771e-05 (W 5.72622e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009407 ± 0.000110348 (W 0.000131424) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019742 ± 9.67342e-05 (W 8.04648e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.026435 ± 0.000292156 (W 6.9815e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0406555 ± 0.000130595 (W 0.000104468) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0543575 ± 0.000168355 (W 0.000127642) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.125016 ± 0.000479065 (W 0.00140342) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.12771 ± 0.00103594 (W 0.000932282) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.523321 ± 0.00365994 (W 0.00238881) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.526806 ± 0.00419498 (W 0.00179279) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.19723 ± 0.00561365 (W 0.00207837) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.19598 ± 0.00358294 (W 0.00367653) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0463565 ± 0.000753526 (W 0.000687429) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0466175 ± 0.000337951 (W 0.00012444) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.167245 ± 0.00116172 (W 0.00128998) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.168077 ± 0.00100564 (W 0.00134718) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.363427 ± 0.00096384 (W 0.00145505) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.365098 ± 0.0019378 (W 0.00194485) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000712 ± 7.36659e-06 (W 3.89517e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0015625 ± 1.66183e-05 (W 4.40883e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00261 ± 2.31661e-05 (W 3.90154e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0032615 ± 0.000116959 (W 3.45353e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0027 ± 2.25093e-05 (W 4.49845e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0137555 ± 0.000697266 (W 4.83957e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0102395 ± 4.74173e-05 (W 6.89721e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0266035 ± 0.00103432 (W 6.23404e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.0216795 ± 0.000891292 (W 7.82762e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.064196 ± 0.000212494 (W 0.000216169) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0650275 ± 0.000194605 (W 0.000326507) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.252088 ± 0.0013652 (W 0.00109678) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.253278 ± 0.00148535 (W 0.00163316) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.542577 ± 0.00186773 (W 0.00175783) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.555846 ± 0.00182402 (W 0.00284745) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011346 ± 0.00040708 (W 6.20514e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014222 ± 0.000100403 (W 9.9737e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0429605 ± 0.00074398 (W 6.91519e-05) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0501325 ± 0.000247352 (W 0.000142418) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.107418 ± 0.000647512 (W 0.000210062) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.118408 ± 0.0012737 (W 0.000320277) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005536 ± 0.000188626 (W 0.000194045) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.00545 ± 4.3499e-05 (W 0.000179523) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0157035 ± 0.000250908 (W 0.000718908) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014242 ± 0.000294447 (W 0.000329581) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.032591 ± 0.00036259 (W 0.00061738) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.034372 ± 0.000310004 (W 0.000643666) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0064315 ± 3.18355e-05 (W 8.07291e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.008185 ± 0.000122132 (W 0.000141568) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.020563 ± 0.0001702 (W 8.6814e-05) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.025216 ± 0.000209175 (W 0.000163858) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.054311 ± 0.000784093 (W 0.00132268) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0528475 ± 0.000452522 (W 0.000501423) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0325045 ± 0.000176121 (W 9.87233e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.029991 ± 0.000176237 (W 0.000134508) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.100432 ± 0.000712824 (W 0.000575066) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.101859 ± 0.000485473 (W 0.000758044) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.203093 ± 0.00175432 (W 0.00213867) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.198819 ± 0.000500152 (W 0.00150612) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.02216 ± 0.00016763 (W 0.000108531) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.069688 ± 0.000246699 (W 0.000258252) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.135055 ± 0.000791928 (W 0.00041278) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.007664 ± 9.10174e-05 (W 5.71995e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009783 ± 0.000556787 (W 0.000143253) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020829 ± 0.000192051 (W 7.16698e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.02766 ± 0.000225836 (W 0.00010484) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0463885 ± 0.000737216 (W 0.000950658) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.055611 ± 0.000639938 (W 0.000754339) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.128654 ± 0.000222166 (W 0.000172958) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.129752 ± 0.00242944 (W 0.000798992) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.439274 ± 0.00143728 (W 0.00170131) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.452633 ± 0.00456132 (W 0.00456896) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.87902 ± 0.00984475 (W 0.00259475) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.879304 ± 0.00856123 (W 0.00155957) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.047992 ± 0.000520128 (W 0.000116366) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.047757 ± 0.000385547 (W 0.000156009) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.17095 ± 0.00107364 (W 0.000662853) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.170643 ± 0.000493449 (W 0.00144005) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.378191 ± 0.00140274 (W 0.00155211) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.376777 ± 0.00126783 (W 0.00154674) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000982 ± 2.73252e-05 (W 3.24676e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002059 ± 3.62528e-05 (W 3.17681e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003306 ± 2.34982e-05 (W 3.54758e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.003256 ± 0.000154486 (W 3.58861e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00263 ± 1.08812e-05 (W 6.86119e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0137175 ± 0.000846514 (W 5.47707e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0102705 ± 4.51088e-05 (W 4.82176e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0265345 ± 0.000882176 (W 8.97529e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n96/b1 [api-inclusive]` | 0.021693 ± 0.000123695 (W 9.70608e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.061259 ± 0.000401388 (W 0.000276537) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0619935 ± 0.000348244 (W 0.000238864) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.19953 ± 0.000793768 (W 0.00105386) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.200436 ± 0.000647825 (W 0.001168) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.414259 ± 0.000939769 (W 0.00401936) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.428125 ± 0.00409145 (W 0.00510054) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011192 ± 0.000100609 (W 5.44575e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.014208 ± 0.000209416 (W 8.53097e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0434015 ± 0.000827957 (W 8.31676e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0502035 ± 0.000180532 (W 0.000220145) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.107646 ± 0.0006654 (W 0.000170637) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.118361 ± 0.000567007 (W 0.000720501) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006482 ± 0.000233153 (W 0.000425608) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0064125 ± 0.000206451 (W 0.000467643) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.012653 ± 0.00021083 (W 0.000235908) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.012649 ± 0.000258129 (W 0.000359361) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0251245 ± 0.000289678 (W 0.000530174) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0256215 ± 0.000275556 (W 0.000694938) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.006493 ± 0.000110914 (W 6.25078e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0081495 ± 0.000103836 (W 0.000215281) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0205325 ± 0.000182721 (W 0.000113841) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.025126 ± 0.000166322 (W 0.000158806) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.06263 ± 0.00169639 (W 0.00171812) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1 [api-inclusive]` | 0.0532195 ± 0.000412751 (W 0.000653552) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0325705 ± 0.000220337 (W 0.000103855) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0301215 ± 0.00014151 (W 0.000123102) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.101344 ± 0.000757185 (W 0.000918939) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.0992795 ± 0.000334245 (W 0.000256263) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.194993 ± 0.00288627 (W 0.00294108) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.195624 ± 0.00128502 (W 0.00142235) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.022136 ± 7.72522e-05 (W 9.10996e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0699195 ± 0.000400934 (W 0.000211448) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.127715 ± 0.000618333 (W 0.000375007) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0075645 ± 7.39687e-05 (W 4.83043e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.009833 ± 9.48866e-05 (W 0.000155247) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.020974 ± 0.000189564 (W 0.000100551) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.027676 ± 0.000261239 (W 0.000106922) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0530325 ± 0.00192567 (W 0.00156151) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n96/b1 [api-inclusive]` | 0.055266 ± 0.000505997 (W 0.000530023) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.128435 ± 0.00106341 (W 0.000190725) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.128176 ± 0.000537894 (W 0.000302897) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.43334 ± 0.0031116 (W 0.00207926) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.448896 ± 0.00196649 (W 0.0023957) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.876536 ± 0.00397121 (W 0.00260725) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.874229 ± 0.00220359 (W 0.00379884) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0474565 ± 0.000290905 (W 0.000173916) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.047678 ± 0.000346712 (W 0.000130324) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.170187 ± 0.000976034 (W 0.000626246) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.170249 ± 0.000785332 (W 0.00148031) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.379215 ± 0.00126381 (W 0.00183836) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.379599 ± 0.00119864 (W 0.00186804) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0009665 ± 2.15754e-05 (W 3.29433e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0020485 ± 2.80814e-05 (W 3.19503e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.003346 ± 6.90237e-05 (W 4.93425e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001964 ± 0.000119061 (W 3.35204e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00256 ± 3.73832e-05 (W 5.32756e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008486 ± 3.72223e-05 (W 3.10984e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010063 ± 0.000159452 (W 6.24483e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0178575 ± 8.41657e-05 (W 6.25619e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.020954 ± 0.000137555 (W 9.54989e-05) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0862955 ± 0.00106744 (W 0.000505038) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.086992 ± 0.000553564 (W 0.000302965) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.323781 ± 0.0123522 (W 0.00160555) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n64/b1 [api-inclusive]` | 0.324027 ± 0.00150253 (W 0.00180364) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.946676 ± 0.00491418 (W 0.0030367) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.952651 ± 0.00431572 (W 0.00286412) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.010144 ± 0.000197772 (W 6.72909e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0113715 ± 0.000141692 (W 6.35155e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.041231 ± 0.000437116 (W 0.00149149) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0426325 ± 0.00814907 (W 0.000100711) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n96/b1` | 0.101926 ± 0.00188014 (W 0.00130926) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.102324 ± 0.000479634 (W 0.00107231) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.006798 ± 4.12019e-05 (W 7.63072e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0067325 ± 0.000103207 (W 6.50369e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.027306 ± 0.000123812 (W 9.36396e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.026925 ± 0.000138637 (W 9.80889e-05) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0818525 ± 0.000300147 (W 0.000661822) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0821775 ± 0.00048791 (W 0.000781534) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.004989 ± 7.89018e-05 (W 0.000101666) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006206 ± 3.62845e-05 (W 6.64476e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0174165 ± 0.000188004 (W 0.000212442) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.020067 ± 0.000178265 (W 0.000235878) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.038179 ± 0.000245217 (W 0.000292961) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.041908 ± 0.000216734 (W 0.000272625) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.02471 ± 0.000127183 (W 9.30474e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.026774 ± 0.000111094 (W 0.000113674) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.092016 ± 0.000452835 (W 0.000776746) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.0960575 ± 0.000537721 (W 0.000863484) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.217356 ± 0.00109371 (W 0.00130005) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.223672 ± 0.00092728 (W 0.00187542) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0151975 ± 0.000121494 (W 8.48745e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.061095 ± 0.000325219 (W 0.000593854) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.128976 ± 0.000434822 (W 0.00104212) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0058705 ± 6.14622e-05 (W 3.57819e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007449 ± 7.62645e-05 (W 7.48836e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.018268 ± 0.000268488 (W 0.000120717) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.020662 ± 0.000258498 (W 0.000104029) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0381715 ± 0.000370619 (W 0.000115967) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0422545 ± 0.000369734 (W 7.46901e-05) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.134762 ± 0.000983066 (W 0.00118582) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.137084 ± 0.00217444 (W 0.00100111) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.525473 ± 0.00143467 (W 0.00206296) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.527053 ± 0.00195071 (W 0.00198487) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.19339 ± 0.00249638 (W 0.0023094) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.19259 ± 0.00642636 (W 0.00391307) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0472735 ± 0.000131328 (W 0.000112589) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0475795 ± 0.000214158 (W 0.000153724) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.171068 ± 0.00118767 (W 0.00154418) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.171571 ± 0.0013713 (W 0.00112118) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.369192 ± 0.00256139 (W 0.00205819) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.36748 ± 0.00157239 (W 0.0020981) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000787 ± 8.92651e-05 (W 4.45619e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0017835 ± 0.000196008 (W 3.76013e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0029405 ± 0.00024958 (W 3.9424e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001964 ± 9.32897e-05 (W 2.39326e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.002655 ± 4.7457e-05 (W 5.25474e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0085205 ± 0.000882847 (W 4.05365e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.010179 ± 9.30027e-05 (W 5.47781e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0180535 ± 0.000968782 (W 6.33156e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.021299 ± 0.000916644 (W 8.41089e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n32/b1` | 0.0642695 ± 0.000291005 (W 0.000169375) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0654565 ± 0.000230338 (W 0.000218539) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.252526 ± 0.000808524 (W 0.00126788) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.253335 ± 0.000955241 (W 0.00158899) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.545415 ± 0.00131603 (W 0.00226204) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.557682 ± 0.0021235 (W 0.00338494) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011216 ± 0.000119739 (W 5.44861e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.012548 ± 0.000206456 (W 7.44393e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.04308 ± 0.00050436 (W 0.000169875) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0449565 ± 0.000257404 (W 0.000165344) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.106226 ± 0.000401686 (W 0.000213397) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.107195 ± 0.000762395 (W 0.000738191) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.005385 ± 0.000122883 (W 0.000233696) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0053495 ± 0.000245707 (W 0.000153563) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0141255 ± 0.000206731 (W 0.000351363) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.014221 ± 0.000215073 (W 0.000432829) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0302295 ± 0.000174727 (W 0.000466261) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.03054 ± 0.000275926 (W 0.000187154) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0050695 ± 8.70211e-05 (W 5.451e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006387 ± 5.89025e-05 (W 9.61425e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0175725 ± 0.000238444 (W 0.000187334) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.020158 ± 8.84912e-05 (W 0.000169451) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.036964 ± 0.000266868 (W 0.00041612) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.040649 ± 0.00013726 (W 0.000323998) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0261425 ± 0.000122466 (W 8.6822e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.028568 ± 8.4139e-05 (W 0.000132956) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.089926 ± 0.000409111 (W 0.000354224) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.094099 ± 0.000457292 (W 0.000382591) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.172886 ± 0.00159404 (W 0.00127147) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.180997 ± 0.00107966 (W 0.00124373) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0160795 ± 0.000120041 (W 0.000106652) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0627415 ± 0.000342517 (W 0.000193816) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.119108 ± 0.000917332 (W 0.00107495) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0063215 ± 0.000161595 (W 5.63071e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0078945 ± 0.000130349 (W 5.18408e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.019411 ± 0.000247969 (W 0.000107938) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.0219605 ± 0.000233141 (W 0.000105142) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0389805 ± 0.00044606 (W 0.000327943) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.042889 ± 0.000639535 (W 0.000474741) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.139263 ± 0.00185116 (W 0.000239753) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.138116 ± 0.000656214 (W 0.000843404) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.435154 ± 0.00185999 (W 0.00207326) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.448429 ± 0.00396339 (W 0.00232014) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.882853 ± 0.00393097 (W 0.0028734) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.884685 ± 0.00494118 (W 0.00312423) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.048564 ± 0.000307974 (W 0.000119442) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.048891 ± 0.000245805 (W 6.10937e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.173724 ± 0.00123705 (W 0.000980384) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.173669 ± 0.000972142 (W 0.00148298) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.383138 ± 0.00171383 (W 0.00157695) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.381842 ± 0.000710769 (W 0.00156351) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.000997 ± 1.85876e-05 (W 3.44717e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002169 ± 1.9532e-05 (W 4.15969e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0035315 ± 0.000273808 (W 5.62122e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-small, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.001834 ± 0.000140051 (W 2.78673e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.00256 ± 2.60288e-05 (W 7.04888e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.008511 ± 4.42964e-05 (W 2.66108e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1 [api-inclusive]` | 0.0101485 ± 4.4308e-05 (W 4.49409e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1` | 0.0181535 ± 8.48397e-05 (W 4.87972e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n96/b1 [api-inclusive]` | 0.021367 ± 0.000113512 (W 0.00010509) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.0614185 ± 0.000481263 (W 0.000170626) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.061981 ± 0.000674375 (W 0.000378169) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.201585 ± 0.00197241 (W 0.0018485) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.199604 ± 0.00115934 (W 0.00170642) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.414596 ± 0.0014215 (W 0.002357) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.430888 ± 0.00172826 (W 0.00473816) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.011232 ± 0.000159644 (W 5.26159e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0125995 ± 9.1114e-05 (W 8.09285e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0431685 ± 0.000735139 (W 0.000120114) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.044805 ± 0.000492537 (W 0.000132852) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.10654 ± 0.000600065 (W 0.000192533) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.107504 ± 0.000815864 (W 0.000217976) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0052255 ± 8.80969e-05 (W 0.000113527) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0052105 ± 6.39419e-05 (W 9.8608e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.012009 ± 0.0012174 (W 0.000522295) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0122445 ± 0.000234454 (W 0.0003136) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.022071 ± 0.000262328 (W 0.000273281) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.022482 ± 0.000280435 (W 0.000737818) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.005065 ± 4.6029e-05 (W 4.49991e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.006358 ± 6.16852e-05 (W 6.57115e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.017652 ± 0.00020261 (W 0.000235413) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1 [api-inclusive]` | 0.0199525 ± 0.000238486 (W 0.000132509) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n96/b1` | 0.037083 ± 0.000222827 (W 0.000252987) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0409505 ± 0.000384426 (W 0.000373177) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0261435 ± 0.000153668 (W 8.34513e-05) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.028498 ± 0.000177402 (W 0.000154614) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.0891125 ± 0.000561112 (W 0.000250392) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.093995 ± 0.000389023 (W 0.000256678) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.167876 ± 0.000846695 (W 0.00147283) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.177885 ± 0.00179177 (W 0.00224592) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.01619 ± 6.87459e-05 (W 0.000103427) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `qr_factor/n64/b1` | 0.0629465 ± 0.000409421 (W 0.000294078) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.11763 ± 0.00109187 (W 0.000592106) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.006357 ± 3.17994e-05 (W 5.74329e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.007941 ± 6.41636e-05 (W 6.1085e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0195265 ± 0.000143915 (W 0.000149337) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1 [api-inclusive]` | 0.021936 ± 0.00015898 (W 8.92344e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.039114 ± 0.000458303 (W 0.000461934) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.042734 ± 0.000344045 (W 0.000627364) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.140175 ± 0.00142641 (W 0.000255325) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.138275 ± 0.00234924 (W 0.000357356) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.429878 ± 0.0019446 (W 0.0021433) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.445321 ± 0.00273236 (W 0.00406332) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 0.879536 ± 0.00380225 (W 0.00190641) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 0.879311 ± 0.00994949 (W 0.00234782) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0490445 ± 0.000378036 (W 0.000103678) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.048949 ± 0.000291749 (W 0.000196918) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.173239 ± 0.00125288 (W 0.000355825) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.173901 ± 0.000895211 (W 0.0014846) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.383202 ± 0.00128002 (W 0.00211521) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.38376 ± 0.00148304 (W 0.00189606) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001012 ± 9.66264e-05 (W 3.04618e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002179 ± 0.000188871 (W 3.69005e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0035925 ± 0.000337403 (W 6.49629e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.0126735 ± 0.00010956 (W 6.90493e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0120125 ± 5.22625e-05 (W 0.000171582) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0701465 ± 0.000657246 (W 0.00020281) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066424 ± 0.000338776 (W 0.000692321) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.11571 ± 0.00066078 (W 0.000752608) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.109428 ± 0.000616032 (W 0.000718248) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.10331 ± 0.000262189 (W 0.00145145) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.104105 ± 0.000819548 (W 0.00149478) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.403444 ± 0.0020332 (W 0.0026502) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.408174 ± 0.00351025 (W 0.0020634) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.16603 ± 0.00451602 (W 0.00295842) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.16822 ± 0.00496737 (W 0.00262478) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.013751 ± 9.60104e-05 (W 3.25443e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0171025 ± 0.0013593 (W 9.0722e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.063961 ± 0.000462321 (W 0.000710754) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0732415 ± 0.000549873 (W 0.000262405) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.178303 ± 0.00104728 (W 0.00163336) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.202232 ± 0.00117238 (W 0.00248905) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0178685 ± 0.000105372 (W 0.000119632) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.017793 ± 6.91751e-05 (W 0.000130651) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0914965 ± 0.000378214 (W 0.000138958) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.091435 ± 0.000393366 (W 0.000157571) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.267632 ± 0.00108145 (W 0.00277225) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.267742 ± 0.000953738 (W 0.00169198) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0097925 ± 5.17533e-05 (W 8.79509e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0118465 ± 0.000139778 (W 0.000103462) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0377755 ± 0.000146932 (W 0.000135578) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0447035 ± 0.000241387 (W 0.000190347) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0916995 ± 0.000134497 (W 0.000828653) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.114476 ± 0.000773547 (W 0.00214525) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.053086 ± 0.000480929 (W 0.000141351) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0500475 ± 0.000213515 (W 0.000152846) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.185753 ± 0.00076993 (W 0.0012538) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.193705 ± 0.000827041 (W 0.00136837) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.435806 ± 0.00180725 (W 0.00220928) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.496018 ± 0.00359795 (W 0.00236235) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0320095 ± 0.000165804 (W 7.85918e-05) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.104069 ± 0.000481536 (W 0.0012527) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.228214 ± 0.000922494 (W 0.00166322) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.01066 ± 5.2443e-05 (W 4.58795e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013305 ± 7.42177e-05 (W 7.28283e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.037324 ± 0.000191814 (W 8.22568e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0459935 ± 0.000167183 (W 0.000223987) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.090253 ± 0.000502254 (W 0.000181354) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.116872 ± 0.000406233 (W 0.00244972) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.160056 ± 0.000971418 (W 0.00150016) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.161346 ± 0.00121915 (W 0.00160777) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.676036 ± 0.00459529 (W 0.00180159) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.675632 ± 0.00203561 (W 0.0025163) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.61973 ± 0.00609511 (W 0.00414057) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.62351 ± 0.00649949 (W 0.0031909) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0562605 ± 0.000358871 (W 0.000167605) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.056691 ± 0.000275579 (W 0.000100564) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.218393 ± 0.00167344 (W 0.00136888) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.218644 ± 0.000963429 (W 0.00149734) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.538759 ± 0.00286901 (W 0.00176135) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.540704 ± 0.00264505 (W 0.00239755) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001052 ± 6.32456e-06 (W 3.23801e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.002474 ± 2.03707e-05 (W 4.86024e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004288 ± 3.52349e-05 (W 4.23417e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012804 ± 5.83609e-05 (W 7.06787e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012087 ± 5.78296e-05 (W 3.99479e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0708355 ± 0.000256738 (W 0.000255777) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066466 ± 0.000188477 (W 0.000228915) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.11292 ± 0.000711409 (W 0.000333465) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108568 ± 0.00055612 (W 0.000439931) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.084376 ± 0.000189962 (W 0.000153498) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0842975 ± 0.000277447 (W 0.000196892) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.31997 ± 0.00235021 (W 0.00151256) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.320871 ± 0.00170067 (W 0.00162194) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.689816 ± 0.00394157 (W 0.00151484) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.689874 ± 0.0031551 (W 0.00156303) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014707 ± 0.000275038 (W 3.60134e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.018204 ± 0.000245885 (W 8.37108e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0665585 ± 0.00156884 (W 0.000160714) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.076386 ± 0.000676681 (W 0.000301993) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.16872 ± 0.000635974 (W 0.00163071) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.20039 ± 0.00442022 (W 0.00240336) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0099135 ± 0.000127175 (W 0.000362931) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009432 ± 8.35554e-05 (W 0.000123691) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0370385 ± 0.00050135 (W 0.000586057) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.036787 ± 0.000238778 (W 0.000385376) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0882635 ± 0.000385142 (W 0.000384759) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.088544 ± 0.000281497 (W 0.000659311) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.009828 ± 5.0686e-05 (W 7.87728e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0120615 ± 0.000306654 (W 9.49366e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0432905 ± 0.000394816 (W 0.000614576) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.044858 ± 0.000817096 (W 0.000589798) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n96/b1` | 0.096874 ± 0.000777844 (W 0.00200644) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.106007 ± 0.00139771 (W 0.00264086) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.054527 ± 0.000313719 (W 0.000173713) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0515215 ± 0.000328776 (W 0.000140243) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.140515 ± 0.00102754 (W 0.0004897) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.144522 ± 0.000887179 (W 0.000451099) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.280268 ± 0.00145472 (W 0.00178379) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.322283 ± 0.00117456 (W 0.00255168) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.03271 ± 0.000156428 (W 0.000110514) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.0960335 ± 0.000637689 (W 0.000438869) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.193449 ± 0.0010501 (W 0.00143191) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011241 ± 4.78442e-05 (W 4.16573e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0137005 ± 8.9823e-05 (W 0.000127925) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.041567 ± 0.000415909 (W 0.000329888) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0477385 ± 0.000660848 (W 0.000553748) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0881495 ± 0.000609701 (W 0.000914851) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.109994 ± 0.000810244 (W 0.0022803) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.162427 ± 0.0019013 (W 0.000233596) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.163312 ± 0.00117333 (W 0.00145938) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.528347 ± 0.00203798 (W 0.00172761) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.532386 ± 0.00165605 (W 0.00167101) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.11469 ± 0.00281715 (W 0.00272517) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.11753 ± 0.00318277 (W 0.00268213) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0576225 ± 0.000573496 (W 0.000149805) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.057757 ± 0.00070254 (W 9.50633e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.222915 ± 0.000891595 (W 0.00122187) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.223175 ± 0.000743707 (W 0.00121598) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.531709 ± 0.0030354 (W 0.00159542) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.531103 ± 0.00303724 (W 0.00210427) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0013125 ± 1.02062e-05 (W 3.3788e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0030005 ± 3.66647e-05 (W 4.07885e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005085 ± 3.69725e-05 (W 3.85662e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.012773 ± 7.36924e-05 (W 7.94809e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012007 ± 4.70053e-05 (W 8.18946e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.070774 ± 0.000403485 (W 0.000193269) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066352 ± 0.000206595 (W 0.000177451) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.113024 ± 0.000490258 (W 0.000266576) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108935 ± 0.000496108 (W 0.000240472) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.080816 ± 0.000537281 (W 0.000176702) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.081167 ± 0.000364775 (W 0.000228206) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.259719 ± 0.00140327 (W 0.00119931) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.26053 ± 0.000994527 (W 0.00156876) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.545291 ± 0.00257537 (W 0.00251585) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.547332 ± 0.00188888 (W 0.00224742) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014704 ± 0.000162038 (W 6.46246e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0185325 ± 0.00188795 (W 8.24843e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.066428 ± 0.000964207 (W 0.000136134) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0762965 ± 0.000639608 (W 0.000259989) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.177007 ± 0.00196392 (W 0.00140425) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.206507 ± 0.000993726 (W 0.0023244) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.00816 ± 0.000181864 (W 0.000157529) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.008045 ± 9.70773e-05 (W 0.000214528) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.024284 ± 0.000125505 (W 0.000336568) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0243345 ± 0.000179356 (W 0.000351214) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0552365 ± 0.00063139 (W 0.00059503) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0565845 ± 0.000842572 (W 0.000683425) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0098235 ± 9.10355e-05 (W 8.33302e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0120085 ± 0.000236649 (W 9.58571e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.049573 ± 0.00116547 (W 0.000863683) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n64/b1 [api-inclusive]` | 0.044338 ± 0.000705618 (W 0.000903114) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.101268 ± 0.00111703 (W 0.00127168) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.108936 ± 0.00139159 (W 0.00143548) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.0544115 ± 0.000255007 (W 0.000209644) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0516235 ± 0.000300427 (W 0.000153922) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.135257 ± 0.000596883 (W 0.00054416) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.139874 ± 0.0206949 (W 0.000670606) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n96/b1` | 0.259078 ± 0.00155958 (W 0.00183154) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.29953 ± 0.00201158 (W 0.00253) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.03264 ± 0.000171589 (W 0.000122696) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.093437 ± 0.000966427 (W 0.000270119) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.179026 ± 0.00134349 (W 0.00123238) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.011271 ± 5.77223e-05 (W 4.89753e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.013846 ± 7.87646e-05 (W 9.61322e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0448275 ± 0.000651313 (W 0.000899658) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0474495 ± 0.000488618 (W 0.000695378) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.092132 ± 0.000926482 (W 0.00113903) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.112566 ± 0.0013569 (W 0.00156828) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.16305 ± 0.00379026 (W 0.00037993) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/n32/b1 [api-inclusive]` | 0.163495 ± 0.001759 (W 0.000361194) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.517199 ± 0.00191666 (W 0.00167035) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.518601 ± 0.0030586 (W 0.00230537) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.08818 ± 0.00416171 (W 0.00283265) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.0916 ± 0.00781675 (W 0.002092) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.057522 ± 0.000258096 (W 0.000158585) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0578865 ± 0.000188504 (W 0.000110504) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.224721 ± 0.000858699 (W 0.00159214) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.226265 ± 0.00105743 (W 0.00149086) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.541635 ± 0.00199972 (W 0.00168446) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.540753 ± 0.00119162 (W 0.00135144) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001313 ± 1.56301e-05 (W 4.22152e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0030005 ± 2.77176e-05 (W 3.43392e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.005095 ± 1.76824e-05 (W 4.78536e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011437 ± 4.26927e-05 (W 5.40048e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012007 ± 6.14538e-05 (W 0.000146614) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0652115 ± 0.000328326 (W 0.000730922) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.06654 ± 0.000312767 (W 0.00025394) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.106073 ± 0.000450731 (W 0.00111713) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108697 ± 0.000395528 (W 0.00117635) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.103894 ± 0.000399761 (W 0.00176434) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.104224 ± 0.00066238 (W 0.00114145) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.404751 ± 0.0017297 (W 0.00188788) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.410357 ± 0.00168327 (W 0.00400766) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 1.16156 ± 0.00573554 (W 0.00274348) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 1.16839 ± 0.0048089 (W 0.00292617) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.0138505 ± 0.000214986 (W 3.65118e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.0152235 ± 0.000144266 (W 5.65796e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.0638685 ± 0.000301462 (W 0.000133284) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.0666345 ± 0.000543018 (W 0.000176588) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.177653 ± 0.000971848 (W 0.00151485) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.186832 ± 0.00104722 (W 0.00257316) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0179235 ± 7.04597e-05 (W 0.00012305) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0178485 ± 9.4325e-05 (W 0.000125673) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.090623 ± 0.00039213 (W 0.000761453) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.090388 ± 0.000323838 (W 0.000252693) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.271091 ± 0.00135985 (W 0.0021144) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.271921 ± 0.0012363 (W 0.00178078) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0083355 ± 3.61741e-05 (W 8.02541e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.009999 ± 5.79221e-05 (W 0.00010426) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034332 ± 0.000312176 (W 0.000215753) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0381105 ± 0.000228622 (W 0.000180795) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.084614 ± 0.000277696 (W 0.000244758) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0974575 ± 0.000999733 (W 0.00244189) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.045462 ± 0.000203291 (W 0.000244506) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.0481875 ± 0.000240849 (W 0.000192453) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.180611 ± 0.000894537 (W 0.00201504) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.187216 ± 0.00097925 (W 0.00168363) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.442095 ± 0.00178303 (W 0.00248865) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.481615 ± 0.0022892 (W 0.00290587) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.0244845 ± 0.00016505 (W 0.000131836) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.10253 ± 0.000513839 (W 0.000551478) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.235934 ± 0.00108444 (W 0.00145326) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.009342 ± 4.85015e-05 (W 4.04161e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0111305 ± 6.27325e-05 (W 7.60109e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.035359 ± 0.000219372 (W 8.55341e-05) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.0391935 ± 0.000176207 (W 0.000144207) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0862765 ± 0.000627058 (W 9.14517e-05) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.101332 ± 0.000705116 (W 0.00175053) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.160282 ± 0.000871007 (W 0.0013921) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.161324 ± 0.000693168 (W 0.00134431) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.673689 ± 0.00546607 (W 0.001941) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.675968 ± 0.00348888 (W 0.00237236) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.62363 ± 0.00540014 (W 0.00272956) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.62554 ± 0.00625734 (W 0.00390682) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0571315 ± 0.000324739 (W 8.99803e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.057586 ± 0.000404856 (W 0.000636805) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.21749 ± 0.00106068 (W 0.00140288) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.217143 ± 0.000933312 (W 0.00184134) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.546485 ± 0.0033107 (W 0.00188696) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.547413 ± 0.00253564 (W 0.00243026) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.001087 ± 9.83192e-06 (W 3.37502e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.00261 ± 1.72482e-05 (W 5.24663e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.004338 ± 3.21895e-05 (W 5.68771e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011386 ± 7.86554e-05 (W 4.39974e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.012062 ± 6.37391e-05 (W 6.99139e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.0649455 ± 0.000252397 (W 0.000166473) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.066294 ± 0.000116913 (W 0.000230176) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.103296 ± 0.000346337 (W 0.000177124) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108033 ± 0.000628699 (W 0.000843623) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.084217 ± 0.000329334 (W 0.000206751) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.084962 ± 0.000498945 (W 0.000227657) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.320769 ± 0.00107949 (W 0.00158899) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.320607 ± 0.00195084 (W 0.00150903) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.688216 ± 0.00256106 (W 0.00166684) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.690023 ± 0.00209185 (W 0.00192731) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014887 ± 0.000163267 (W 4.88864e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.016465 ± 0.000182232 (W 7.50812e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.066739 ± 0.000684791 (W 0.000138302) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.069378 ± 0.00054303 (W 0.000683571) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.168413 ± 0.00145861 (W 0.00128693) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.184021 ± 0.00211318 (W 0.00180409) | - | - | `*` one row has no measurement here |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0098425 ± 0.000156795 (W 0.000242303) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.009893 ± 0.000186602 (W 0.000151333) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.0360715 ± 0.000184088 (W 0.000227421) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.0360165 ± 0.000114803 (W 0.000342232) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0869815 ± 0.000251244 (W 0.000766832) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0871465 ± 0.000326703 (W 0.000385484) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.0083555 ± 7.42312e-05 (W 7.40506e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.010209 ± 6.13854e-05 (W 0.000132693) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.0338075 ± 0.000398271 (W 0.00044121) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.0370325 ± 0.000714864 (W 0.000433943) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.074296 ± 0.00151842 (W 0.00140031) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0888365 ± 0.000870085 (W 0.00178182) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.046966 ± 0.000144653 (W 0.000148458) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.04951 ± 0.000354625 (W 0.000168005) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.130787 ± 0.00047693 (W 0.000451245) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.138589 ± 0.00120345 (W 0.000428744) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.261257 ± 0.00107917 (W 0.00138144) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.307722 ± 0.00109152 (W 0.00257712) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025443 ± 0.000138294 (W 0.000119158) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.08788 ± 0.000254897 (W 0.000318683) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.175945 ± 0.000611245 (W 0.00149219) | - | - | `*` one row has no measurement here |
| `solve/n32/b1` | 0.0098525 ± 7.27901e-05 (W 3.41565e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0115515 ± 7.1491e-05 (W 8.06467e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0360305 ± 0.000604645 (W 0.000368407) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.039423 ± 0.000791971 (W 0.000658943) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.0770735 ± 0.000334713 (W 0.000731571) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.0946905 ± 0.000359136 (W 0.00174736) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.162592 ± 0.000541715 (W 0.000297177) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.163705 ± 0.00188285 (W 0.000936701) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.526353 ± 0.00291869 (W 0.00248155) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.531553 ± 0.00285931 (W 0.00188578) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.11246 ± 0.00478264 (W 0.00227776) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.11809 ± 0.00263057 (W 0.00328047) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.058453 ± 0.000208024 (W 0.000104759) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.0586085 ± 0.000503267 (W 8.10503e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.22439 ± 0.00163625 (W 0.00133467) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.22341 ± 0.00156941 (W 0.00123724) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.536018 ± 0.00255112 (W 0.00216307) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.536018 ± 0.00185073 (W 0.00182388) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0013275 ± 1.73282e-05 (W 3.6448e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.003076 ± 2.60436e-05 (W 4.74342e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.0050295 ± 3.9719e-05 (W 3.85517e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

## linalg-small, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n32/b1` | 0.011473 ± 6.27484e-05 (W 4.8087e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n32/b1 [api-inclusive]` | 0.0119995 ± 4.93778e-05 (W 6.51964e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `cholesky/n64/b1` | 0.065026 ± 0.000252744 (W 0.000161823) | - | - | `*` one row has no measurement here |
| `cholesky/n64/b1 [api-inclusive]` | 0.0662765 ± 0.000359842 (W 0.000202286) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1` | 0.10408 ± 0.000321377 (W 0.000224838) | - | - | `*` one row has no measurement here |
| `cholesky/n96/b1 [api-inclusive]` | 0.108433 ± 0.00040951 (W 0.000287955) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1` | 0.080903 ± 0.000455858 (W 0.000211169) | - | - | `*` one row has no measurement here |
| `eigh/n32/b1 [api-inclusive]` | 0.0816865 ± 0.000382238 (W 0.000125756) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1` | 0.258135 ± 0.00150726 (W 0.00124814) | - | - | `*` one row has no measurement here |
| `eigh/n64/b1 [api-inclusive]` | 0.259153 ± 0.000897278 (W 0.00151151) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1` | 0.54503 ± 0.00287166 (W 0.00251302) | - | - | `*` one row has no measurement here |
| `eigh/n96/b1 [api-inclusive]` | 0.545144 ± 0.00126372 (W 0.00291019) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n32/b1` | 0.014929 ± 0.000231912 (W 4.61122e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n32/b1 [api-inclusive]` | 0.016271 ± 0.000255623 (W 8.56086e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `full_piv_lu/n64/b1` | 0.066301 ± 0.000684109 (W 0.000100182) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n64/b1 [api-inclusive]` | 0.069409 ± 0.000835135 (W 0.000167463) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1` | 0.174725 ± 0.00166803 (W 0.0016861) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n96/b1 [api-inclusive]` | 0.190773 ± 0.00641515 (W 0.00266497) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n32/b1 [block(basis=32,target=32)]` | 0.0079645 ± 0.000104539 (W 0.000189038) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n32/b1 [sequence(basis=32,target=32)]` | 0.0079095 ± 0.000104502 (W 0.00015618) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `householder_apply/n64/b1 [block(basis=64,target=64)]` | 0.023727 ± 0.000212576 (W 0.00033325) | - | - | `*` one row has no measurement here |
| `householder_apply/n64/b1 [sequence(basis=64,target=64)]` | 0.024298 ± 0.000107361 (W 0.000395774) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [block(basis=96,target=96)]` | 0.0526525 ± 0.000606327 (W 0.000479905) | - | - | `*` one row has no measurement here |
| `householder_apply/n96/b1 [sequence(basis=96,target=96)]` | 0.0525175 ± 0.000427652 (W 0.000567192) | - | - | `*` one row has no measurement here |
| `lu/n32/b1` | 0.008421 ± 4.19162e-05 (W 5.89628e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n32/b1 [api-inclusive]` | 0.0101685 ± 6.21399e-05 (W 0.000134212) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `lu/n64/b1` | 0.034188 ± 0.0002641 (W 0.000465759) | - | - | `*` one row has no measurement here |
| `lu/n64/b1 [api-inclusive]` | 0.037539 ± 0.000397407 (W 0.000715287) | - | - | `*` one row has no measurement here |
| `lu/n96/b1` | 0.0758165 ± 0.00131736 (W 0.00125506) | - | - | `*` one row has no measurement here |
| `lu/n96/b1 [api-inclusive]` | 0.0936285 ± 0.00105937 (W 0.0024691) | - | - | `*` one row has no measurement here |
| `qr/n32/b1` | 0.046994 ± 0.000285848 (W 0.000200127) | - | - | `*` one row has no measurement here |
| `qr/n32/b1 [api-inclusive]` | 0.049702 ± 0.00018825 (W 0.000208737) | - | - | `*` one row has no measurement here |
| `qr/n64/b1` | 0.123826 ± 0.00135879 (W 0.000448144) | - | - | `*` one row has no measurement here |
| `qr/n64/b1 [api-inclusive]` | 0.131567 ± 0.00127379 (W 0.000458791) | - | - | `*` one row has no measurement here |
| `qr/n96/b1` | 0.23557 ± 0.00179279 (W 0.00166832) | - | - | `*` one row has no measurement here |
| `qr/n96/b1 [api-inclusive]` | 0.282503 ± 0.00263747 (W 0.00182689) | - | - | `*` one row has no measurement here |
| `qr_factor/n32/b1` | 0.025532 ± 7.99642e-05 (W 0.000123443) | - | - | `*` one row has no measurement here |
| `qr_factor/n64/b1` | 0.085655 ± 0.000322356 (W 0.00031954) | - | - | `*` one row has no measurement here |
| `qr_factor/n96/b1` | 0.166005 ± 0.00517794 (W 0.00113613) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n32/b1` | 0.0098295 ± 6.228e-05 (W 5.14282e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n32/b1 [api-inclusive]` | 0.0116685 ± 0.000136644 (W 8.6251e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `solve/n64/b1` | 0.0366575 ± 0.000386667 (W 0.000525428) | - | - | `*` one row has no measurement here |
| `solve/n64/b1 [api-inclusive]` | 0.039928 ± 0.000508957 (W 0.000393403) | - | - | `*` one row has no measurement here |
| `solve/n96/b1` | 0.077765 ± 0.000507158 (W 0.000331448) | - | - | `*` one row has no measurement here |
| `solve/n96/b1 [api-inclusive]` | 0.096833 ± 0.000456039 (W 0.00236383) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1` | 0.164341 ± 0.00252999 (W 0.000393366) | - | - | `*` one row has no measurement here |
| `svd_thin/n32/b1 [api-inclusive]` | 0.163809 ± 0.000877721 (W 0.0015805) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1` | 0.516476 ± 0.00175841 (W 0.00215712) | - | - | `*` one row has no measurement here |
| `svd_thin/n64/b1 [api-inclusive]` | 0.518451 ± 0.00193823 (W 0.00203746) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1` | 1.09328 ± 0.00431579 (W 0.00266305) | - | - | `*` one row has no measurement here |
| `svd_thin/n96/b1 [api-inclusive]` | 1.0966 ± 0.00228998 (W 0.00376569) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1` | 0.0580205 ± 0.000606991 (W 8.96236e-05) | - | - | `*` one row has no measurement here |
| `svd_values/n32/b1 [api-inclusive]` | 0.058428 ± 0.000433828 (W 0.000102583) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1` | 0.223139 ± 0.00102665 (W 0.00139222) | - | - | `*` one row has no measurement here |
| `svd_values/n64/b1 [api-inclusive]` | 0.225552 ± 0.0028503 (W 0.00132844) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1` | 0.548061 ± 0.00567207 (W 0.00210078) | - | - | `*` one row has no measurement here |
| `svd_values/n96/b1 [api-inclusive]` | 0.546712 ± 0.00316835 (W 0.00226903) | - | - | `*` one row has no measurement here |
| `triangular_solve/n32/b1` | 0.0013225 ± 1.24204e-05 (W 3.21293e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n64/b1` | 0.0030615 ± 4.18457e-05 (W 3.5525e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n96/b1` | 0.00503 ± 3.30454e-05 (W 3.43549e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |

Not applicable to this population's shapes and reported as skipped: `cholesky/32`, `cholesky/64`, `cholesky/96`, `eig/32`, `eig/64`, `eig/96`, `eigh/32`, `eigh/64`, `eigh/96`, `full_piv_lu/32`, `full_piv_lu/64`, `full_piv_lu/96`, `householder_apply/32`, `householder_apply/64`, `householder_apply/96`, `lu/32`, `lu/64`, `lu/96`, `qr/32`, `qr/64`, `qr/96`, `qr_factor/32`, `qr_factor/64`, `qr_factor/96`, `solve/32`, `solve/64`, `solve/96`, `svd_thin/32`, `svd_thin/64`, `svd_thin/96`, `svd_values/32`, `svd_values/64`, `svd_values/96`, `triangular_solve/32`, `triangular_solve/64`, `triangular_solve/96`, `upgrade_factor/32`, `upgrade_factor/64`, `upgrade_factor/96`.

## linalg-mid, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.058264 ± 0.000601605 (W 0.00014743) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.060988 ± 0.000405745 (W 0.000598865) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.329983 ± 0.0028414 (W 0.00235399) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.349025 ± 0.00255506 (W 0.00215488) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.74935 ± 0.00898526 (W 0.00762931) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.17084 ± 0.0137833 (W 0.00477865) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.56122 ± 0.00735788 (W 0.00524651) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.55264 ± 0.00725704 (W 0.00603082) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.6102 ± 0.0348814 (W 0.0173873) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.656 ± 0.0548664 (W 0.0124921) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 50.6842 ± 0.310837 (W 0.0491291) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 51.0128 ± 0.199013 (W 0.0571452) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.365818 ± 0.0285485 (W 0.00274402) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.340634 ± 0.00228056 (W 0.00262657) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.89761 ± 0.00728073 (W 0.00746832) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.03859 ± 0.00820472 (W 0.00770303) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 21.5514 ± 0.260658 (W 0.0352126) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 22.4436 ± 0.0866905 (W 0.049186) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.257972 ± 0.0013585 (W 0.00235435) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.257511 ± 0.000903323 (W 0.00187332) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.89495 ± 0.00315602 (W 0.00654212) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.89383 ± 0.0050941 (W 0.00351156) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.9074 ± 0.0441196 (W 0.0355063) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.9101 ± 0.0315106 (W 0.030179) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.104793 ± 0.000206034 (W 0.000145903) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.167892 ± 0.00127956 (W 0.00226948) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.568007 ± 0.00254069 (W 0.00201497) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.947521 ± 0.00254285 (W 0.00332328) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.85898 ± 0.0107041 (W 0.00842431) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 5.6594 ± 0.0325411 (W 0.020749) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.448255 ± 0.00278238 (W 0.00224594) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.563838 ± 0.00126718 (W 0.0026817) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.88801 ± 0.0153477 (W 0.0101045) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.20287 ± 0.00924357 (W 0.00776313) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 18.7004 ± 0.052402 (W 0.0241331) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 20.1812 ± 0.0674218 (W 0.0517887) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.248444 ± 0.00255716 (W 0.00188805) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.52157 ± 0.00508587 (W 0.00413695) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 8.54093 ± 0.0243187 (W 0.0234776) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0931245 ± 0.00056447 (W 0.00107528) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.131275 ± 0.000480935 (W 0.00115129) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.521626 ± 0.00251329 (W 0.00211017) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.698209 ± 0.00565137 (W 0.00290107) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.52477 ± 0.011162 (W 0.00786742) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.27869 ± 0.0500786 (W 0.00909306) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.85851 ± 0.0111189 (W 0.00773373) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.88084 ± 0.0127878 (W 0.00932194) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.4172 ± 0.0771299 (W 0.0226926) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 14.4624 ± 0.056066 (W 0.0225678) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 86.4795 ± 0.269706 (W 0.128505) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 86.7711 ± 0.313495 (W 0.109642) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.921775 ± 0.00280205 (W 0.00330877) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.92107 ± 0.00498527 (W 0.00293581) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.45156 ± 0.0379387 (W 0.0117466) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.45579 ± 0.0306006 (W 0.00952458) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 37.1266 ± 0.0869415 (W 0.0493744) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 37.1729 ± 0.19661 (W 0.0443936) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004533 ± 5.65119e-05 (W 5.27653e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0128985 ± 0.00033185 (W 0.00013685) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.043752 ± 0.000348221 (W 0.000481258) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.049662 ± 0.000769866 (W 0.000228727) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.058394 ± 0.00107979 (W 0.000678231) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.17982 ± 0.00310596 (W 0.0016373) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.230812 ± 0.000874823 (W 0.001561) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.02115 ± 0.00512445 (W 0.0033296) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.00515 ± 0.0089291 (W 0.0028713) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.44965 ± 0.00214873 (W 0.00429458) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.45256 ± 0.00392201 (W 0.00469939) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.69771 ± 0.0186209 (W 0.0123185) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.68866 ± 0.0186069 (W 0.00924453) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.881 ± 0.075767 (W 0.0261281) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.7627 ± 0.043189 (W 0.0220298) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.336348 ± 0.00330819 (W 0.00345968) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.345158 ± 0.00356154 (W 0.00287712) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.99074 ± 0.0104484 (W 0.00597076) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.30784 ± 0.0100105 (W 0.00993304) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.471 ± 0.0347339 (W 0.0279915) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.1828 ± 0.0534241 (W 0.0129601) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.083922 ± 0.000508502 (W 0.00116432) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.0837 ± 0.000963179 (W 0.000576638) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.516307 ± 0.00153171 (W 0.00257658) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.515035 ± 0.00215137 (W 0.00308402) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.17821 ± 0.00937658 (W 0.012022) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.16316 ± 0.0121473 (W 0.00879169) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.118081 ± 0.00121979 (W 0.00166489) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.158395 ± 0.00102567 (W 0.00115346) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.394097 ± 0.00120848 (W 0.00422015) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.721824 ± 0.00425493 (W 0.00452523) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.838 ± 0.00633036 (W 0.00466676) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.53911 ± 0.0131192 (W 0.0110777) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.343738 ± 0.0013941 (W 0.00178691) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.428355 ± 0.000968836 (W 0.00240013) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.04647 ± 0.00313137 (W 0.00255881) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.59496 ± 0.00620623 (W 0.00474217) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.97107 ± 0.022974 (W 0.0155111) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.3718 ± 0.0608554 (W 0.0163632) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.223492 ± 0.000580923 (W 0.00134311) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.65901 ± 0.00256287 (W 0.0029605) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 3.22014 ± 0.00786186 (W 0.00645042) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0875185 ± 0.000468507 (W 0.00063316) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.122012 ± 0.00249955 (W 0.000916513) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n256/b1` | 0.390014 ± 0.00369589 (W 0.00239471) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.73476 ± 0.00495901 (W 0.00378665) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.71168 ± 0.00952568 (W 0.0048003) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.51741 ± 0.0136966 (W 0.0121094) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.00358 ± 0.00292723 (W 0.00286105) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.00817 ± 0.00416821 (W 0.00370589) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.10625 ± 0.0122773 (W 0.0244599) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.12501 ± 0.0185812 (W 0.0194526) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 36.655 ± 0.119342 (W 0.0303517) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 36.7092 ± 0.0694759 (W 0.0423924) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.961144 ± 0.0047011 (W 0.00264449) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.962902 ± 0.00409217 (W 0.00244527) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.14708 ± 0.0150799 (W 0.00884952) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.16309 ± 0.00539807 (W 0.00701649) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 18.5125 ± 0.079456 (W 0.0257971) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 18.5523 ± 0.0663326 (W 0.0183135) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005385 ± 5.34805e-05 (W 5.57002e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0132195 ± 0.000174945 (W 0.000129771) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0324075 ± 0.000260487 (W 0.00022193) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0510095 ± 0.0011135 (W 0.000361961) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.058212 ± 0.000939533 (W 0.000603577) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.162022 ± 0.00191617 (W 0.00176192) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.216894 ± 0.00102638 (W 0.00210278) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.797172 ± 0.00204854 (W 0.00354071) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.834667 ± 0.0052566 (W 0.00342928) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.94263 ± 0.00334635 (W 0.00211991) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.949995 ± 0.003105 (W 0.00280468) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.99782 ± 0.0113693 (W 0.00967205) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.9991 ± 0.0131194 (W 0.00661337) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.5795 ± 0.0436287 (W 0.0194579) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.5576 ± 0.0248478 (W 0.020043) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.335517 ± 0.00117264 (W 0.00263183) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.356413 ± 0.00205274 (W 0.00421077) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.86954 ± 0.0110837 (W 0.00825577) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.22213 ± 0.0125104 (W 0.00989192) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.0997 ± 0.0373613 (W 0.0304237) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.9162 ± 0.0493855 (W 0.0195233) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.058589 ± 0.000556254 (W 0.000625565) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.058569 ± 0.000249449 (W 0.00073807) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.284407 ± 0.00252194 (W 0.0023704) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.284241 ± 0.00200015 (W 0.0023925) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.24105 ± 0.0129307 (W 0.0138384) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.25232 ± 0.00435301 (W 0.0116251) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.123479 ± 0.00108841 (W 0.00166087) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.159483 ± 0.00122394 (W 0.00135805) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.326706 ± 0.00173349 (W 0.00176463) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.692147 ± 0.00432755 (W 0.00256429) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.30224 ± 0.0029771 (W 0.00483056) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.19557 ± 0.028014 (W 0.0106892) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.323434 ± 0.00226864 (W 0.00226989) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.410095 ± 0.00612197 (W 0.00355269) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.850005 ± 0.00418148 (W 0.00187591) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.41415 ± 0.00203683 (W 0.00655164) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.42903 ± 0.0447899 (W 0.0077549) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 6.11912 ± 0.0769792 (W 0.0171415) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.206517 ± 0.000958078 (W 0.00166816) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.611028 ± 0.00339171 (W 0.00248248) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.91929 ± 0.0228673 (W 0.0070846) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0885245 ± 0.000423333 (W 0.000592042) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.120145 ± 0.000649719 (W 0.00107656) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.324345 ± 0.00274464 (W 0.00154552) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.700983 ± 0.00309888 (W 0.0024638) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.25233 ± 0.00470045 (W 0.00428111) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.155 ± 0.0159771 (W 0.00658627) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.97798 ± 0.00815368 (W 0.00386051) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.99142 ± 0.0050526 (W 0.00317032) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.38689 ± 0.0236859 (W 0.0104071) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.41877 ± 0.0222551 (W 0.0156764) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 30.3703 ± 0.038855 (W 0.0351115) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 30.4937 ± 0.0654356 (W 0.0731243) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.00717 ± 0.00419406 (W 0.00291113) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.00925 ± 0.00689519 (W 0.00377215) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.01414 ± 0.00738288 (W 0.00721902) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.02626 ± 0.0147544 (W 0.00768092) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.423 ± 0.0591464 (W 0.0215863) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.4392 ± 0.0339355 (W 0.0176785) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005485 ± 4.0696e-05 (W 6.09633e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.013054 ± 0.000164005 (W 9.25629e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0305865 ± 0.000385391 (W 0.000209563) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0491675 ± 0.000665382 (W 0.000138246) | 0.052152 ± 0.000130981 (W 0.000206609) | 🔵 faer-fork 1.06× faster than `lapack-aocl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.055315 ± 0.000520166 (W 0.000509152) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.280741 ± 0.0018904 (W 0.00215989) | 0.262412 ± 0.00164165 (W 0.00196102) | 🔴 faer-fork 1.07× slower than `lapack-aocl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.306985 ± 0.0019982 (W 0.00235989) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.92947 ± 0.0150629 (W 0.00579533) | 1.75437 ± 0.00736789 (W 0.00405142) | 🔴 faer-fork 1.10× slower than `lapack-aocl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.0103 ± 0.0165394 (W 0.00526438) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.5571 ± 0.00771354 (W 0.00492043) | 1.80895 ± 0.00475067 (W 0.0073515) | 🔴 faer-fork 1.41× slower than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 2.55052 ± 0.00634679 (W 0.00580072) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.5703 ± 0.0540873 (W 0.0227541) | 8.52867 ± 0.0246412 (W 0.0119302) | 🔴 faer-fork 1.24× slower than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 10.6307 ± 0.0643177 (W 0.0145698) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 50.7371 ± 0.218941 (W 0.0823438) | 47.7586 ± 0.371178 (W 0.186412) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 50.6455 ± 0.288558 (W 0.0824957) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.327066 ± 0.00235792 (W 0.00221049) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.315314 ± 0.00153945 (W 0.00211796) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.88001 ± 0.121809 (W 0.00924401) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.94677 ± 0.015354 (W 0.00751632) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 22.4097 ± 0.0495887 (W 0.0285257) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 21.7973 ± 0.104255 (W 0.0622778) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.255151 ± 0.000936194 (W 0.00151569) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.255418 ± 0.000709742 (W 0.00169065) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.86964 ± 0.00646642 (W 0.00761865) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.86612 ± 0.00939763 (W 0.00373228) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 15.8124 ± 0.0461037 (W 0.0262888) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 15.7827 ± 0.0607536 (W 0.139966) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0974885 ± 0.000676631 (W 0.00112642) | 0.138743 ± 0.00108119 (W 0.00105083) | 🔵 faer-fork 1.42× faster than `lapack-aocl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.144317 ± 0.0009369 (W 0.00183409) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.561047 ± 0.00366704 (W 0.00315887) | 1.001 ± 0.00616749 (W 0.00267899) | 🔵 faer-fork 1.78× faster than `lapack-aocl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.843061 ± 0.00444859 (W 0.00264866) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.69965 ± 0.0206513 (W 0.00648787) | 9.41453 ± 0.0363566 (W 0.027761) | 🔵 faer-fork 2.54× faster than `lapack-aocl` |  |
| `lu/n512/b1 [api-inclusive]` | 5.08635 ± 0.0238429 (W 0.00737053) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.452124 ± 0.00209009 (W 0.00221449) | 1.10368 ± 0.00563271 (W 0.00647187) | 🔵 faer-fork 2.44× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.532494 ± 0.00147635 (W 0.00240665) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.63043 ± 0.0153559 (W 0.00500066) | 5.62687 ± 0.020085 (W 0.00988757) | 🔵 faer-fork 2.14× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.0767 ± 0.0067035 (W 0.00659422) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 17.4544 ± 1.60588 (W 0.0159285) | 34.7358 ± 0.089913 (W 0.0265312) | 🔵 faer-fork 1.99× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 19.5217 ± 2.58025 (W 0.0509965) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.252211 ± 0.00158294 (W 0.00157014) | 0.618356 ± 0.00358561 (W 0.00216716) | 🔵 faer-fork 2.45× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.27674 ± 0.00442333 (W 0.00359954) | 2.93256 ± 0.0125279 (W 0.00851462) | 🔵 faer-fork 2.30× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 7.632 ± 0.0299111 (W 0.0141115) | 16.107 ± 0.0505415 (W 0.0138116) | 🔵 faer-fork 2.11× faster than `lapack-aocl` |  |
| `solve/n128/b1` | 0.0936575 ± 0.00060076 (W 0.000133051) | 0.124571 ± 0.00073767 (W 0.000904557) | 🔵 faer-fork 1.33× faster than `lapack-aocl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.104602 ± 0.000508675 (W 0.00129928) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.547103 ± 0.00134353 (W 0.00208687) | 0.554207 ± 0.00217215 (W 0.00236623) | · |  |
| `solve/n256/b1 [api-inclusive]` | 0.596405 ± 0.00226804 (W 0.0026778) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.60761 ± 0.0120293 (W 0.00420882) | 3.46237 ± 0.0162142 (W 0.00724249) | · |  |
| `solve/n512/b1 [api-inclusive]` | 3.82085 ± 0.0121925 (W 0.00529204) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.83678 ± 0.0174792 (W 0.0081899) | 2.95015 ± 0.00825489 (W 0.00590283) | · |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.86005 ± 0.00809394 (W 0.00486928) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 14.4419 ± 0.0569842 (W 0.0172152) | 15.8562 ± 0.0717237 (W 0.0259024) | 🔵 faer-fork 1.10× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 14.4065 ± 0.0352004 (W 0.0223284) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 85.616 ± 0.0944009 (W 0.120466) | 106.251 ± 0.38544 (W 0.293265) | 🔵 faer-fork 1.24× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 85.7719 ± 0.260637 (W 0.114018) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.897833 ± 0.00492638 (W 0.00175978) | 1.16241 ± 0.00369324 (W 0.0027476) | 🔵 faer-fork 1.29× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.900543 ± 0.00303055 (W 0.00241429) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.30363 ± 0.0214695 (W 0.00877955) | 6.96824 ± 0.0683647 (W 0.0185025) | 🔵 faer-fork 1.31× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.31964 ± 0.0457874 (W 0.0114266) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 36.1603 ± 0.0921559 (W 0.0513888) | 48.8709 ± 0.519776 (W 0.173547) | 🔵 faer-fork 1.35× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 36.1663 ± 0.100943 (W 0.0500162) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005355 ± 3.41199e-05 (W 5.72589e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.014308 ± 0.000212502 (W 8.13559e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.04831 ± 0.000813345 (W 0.000350661) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0426275 ± 0.000795297 (W 0.000445854) | 0.046422 ± 0.000275947 (W 0.000683828) | 🔵 faer-fork 1.09× faster than `lapack-aocl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.051907 ± 0.00104474 (W 0.000740517) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.156676 ± 0.00163442 (W 0.0020919) | 0.147877 ± 0.00112493 (W 0.00212964) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.190177 ± 0.00232507 (W 0.00187485) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.688125 ± 0.00346187 (W 0.00279287) | 0.566539 ± 0.00428577 (W 0.00682448) | 🔴 faer-fork 1.21× slower than `lapack-aocl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.838016 ± 0.00484441 (W 0.00201702) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.44461 ± 0.00475575 (W 0.00368978) | 1.48479 ± 0.00582169 (W 0.00625245) | · |  |
| `eigh/n128/b1 [api-inclusive]` | 1.45287 ± 0.00562516 (W 0.00587219) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.64591 ± 0.0236149 (W 0.00786676) | 6.0425 ± 0.0218786 (W 0.0159004) | 🔵 faer-fork 1.07× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.64033 ± 0.0167893 (W 0.00867547) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.7453 ± 0.0323453 (W 0.0302313) | 27.8315 ± 0.0876737 (W 0.0860928) | 🔵 faer-fork 1.12× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 24.6275 ± 0.0280252 (W 0.0274755) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.32494 ± 0.00380166 (W 0.00241976) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.323016 ± 0.000861276 (W 0.00290383) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.99276 ± 0.0065394 (W 0.0078036) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.1969 ± 0.0085592 (W 0.00828161) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 11.7886 ± 0.0510825 (W 0.0113339) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.6708 ± 0.108999 (W 0.0223485) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0837545 ± 0.000287476 (W 0.00121372) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.08339 ± 0.000566889 (W 0.000647297) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.523465 ± 0.0019411 (W 0.00186958) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.521358 ± 0.00194214 (W 0.00308056) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 4.19092 ± 0.0128052 (W 0.00791409) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 4.19009 ± 0.0182336 (W 0.0149917) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.079468 ± 0.00111465 (W 0.000690363) | 0.129029 ± 0.00055834 (W 0.00121738) | 🔵 faer-fork 1.62× faster than `lapack-aocl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.135894 ± 0.00199457 (W 0.00133561) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.333089 ± 0.00290678 (W 0.00421877) | 0.438765 ± 0.00150625 (W 0.00308921) | 🔵 faer-fork 1.32× faster than `lapack-aocl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.619867 ± 0.00315579 (W 0.0040869) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.56885 ± 0.00620571 (W 0.00534502) | 3.24953 ± 0.0075035 (W 0.00840073) | 🔵 faer-fork 2.07× faster than `lapack-aocl` |  |
| `lu/n512/b1 [api-inclusive]` | 2.9915 ± 0.0126234 (W 0.0105831) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.300959 ± 0.00349014 (W 0.0017535) | 0.49381 ± 0.00361099 (W 0.0042495) | 🔵 faer-fork 1.64× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.401593 ± 0.00301444 (W 0.00304976) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.975104 ± 0.00434805 (W 0.00334969) | 1.99238 ± 0.00679733 (W 0.00848992) | 🔵 faer-fork 2.04× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.47 ± 0.00490403 (W 0.0069061) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.53852 ± 0.0118001 (W 0.00779165) | 10.3617 ± 0.0143501 (W 0.0176358) | 🔵 faer-fork 1.87× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 7.74747 ± 0.0165395 (W 0.0122576) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.203295 ± 0.000494231 (W 0.00161409) | 0.271634 ± 0.00234478 (W 0.00337624) | 🔵 faer-fork 1.34× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 0.58109 ± 0.00158151 (W 0.00274931) | 1.01467 ± 0.00612753 (W 0.00524149) | 🔵 faer-fork 1.75× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 2.92388 ± 0.0084802 (W 0.00588373) | 4.87342 ± 0.0142717 (W 0.012176) | 🔵 faer-fork 1.67× faster than `lapack-aocl` |  |
| `solve/n128/b1` | 0.082254 ± 0.00117116 (W 0.00149219) | 0.111026 ± 0.000673664 (W 0.000346293) | 🔵 faer-fork 1.35× faster than `lapack-aocl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.095939 ± 0.00114038 (W 0.00142244) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.314005 ± 0.00153181 (W 0.00205011) | 0.315182 ± 0.00280404 (W 0.00242947) | · |  |
| `solve/n256/b1 [api-inclusive]` | 0.626851 ± 0.00247703 (W 0.00359318) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.43388 ± 0.00275895 (W 0.00292659) | 1.36515 ± 0.00445213 (W 0.00554685) | · |  |
| `solve/n512/b1 [api-inclusive]` | 2.94274 ± 0.00801034 (W 0.00624151) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.98608 ± 0.0104266 (W 0.00392878) | 2.54361 ± 0.00969708 (W 0.00644227) | 🔵 faer-fork 1.28× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.9952 ± 0.0112343 (W 0.00216687) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.0236 ± 0.0164587 (W 0.0222196) | 13.2794 ± 0.0606623 (W 0.0435769) | 🔵 faer-fork 1.66× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 8.0215 ± 0.0337564 (W 0.00957784) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 35.7263 ± 0.0932217 (W 0.0353975) | 64.744 ± 0.926801 (W 0.0926934) | 🔵 faer-fork 1.81× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 35.8674 ± 0.0316619 (W 0.0262101) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.939883 ± 0.00443114 (W 0.00255891) | 1.18068 ± 0.00446891 (W 0.00347054) | 🔵 faer-fork 1.26× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.943068 ± 0.00470401 (W 0.00280808) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.02363 ± 0.0063909 (W 0.00610008) | 7.71148 ± 0.0362873 (W 0.0270859) | 🔵 faer-fork 1.92× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 4.01996 ± 0.0141474 (W 0.0051271) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.4691 ± 0.0232833 (W 0.0171251) | 34.8761 ± 0.100906 (W 0.0789467) | 🔵 faer-fork 2.00× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 17.5272 ± 0.0439858 (W 0.0213651) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005811 ± 3.18601e-05 (W 5.24666e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.015639 ± 0.000310242 (W 0.000288293) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.040465 ± 0.000393854 (W 0.000572848) | - | - | `*` one row has no measurement here |

## linalg-mid, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0438365 ± 0.000727598 (W 0.000924846) | 0.0461655 ± 0.000507479 (W 0.000770536) | 🔵 faer-fork 1.05× faster than `lapack-aocl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.051664 ± 0.00100397 (W 0.000746777) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.139507 ± 0.0017797 (W 0.000745577) | 0.152642 ± 0.00203253 (W 0.00247996) | 🔵 faer-fork 1.09× faster than `lapack-aocl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.173954 ± 0.00160526 (W 0.00105082) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.525901 ± 0.00283714 (W 0.00278193) | 0.388908 ± 0.00189279 (W 0.0048709) | 🔴 faer-fork 1.35× slower than `lapack-aocl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 0.67364 ± 0.00320125 (W 0.00265425) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.940831 ± 0.00262111 (W 0.00198615) | 1.48798 ± 0.0103085 (W 0.00771097) | 🔵 faer-fork 1.58× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 0.945435 ± 0.00499636 (W 0.0017765) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.94783 ± 0.0270489 (W 0.00767196) | 5.62053 ± 0.0152582 (W 0.0117595) | 🔵 faer-fork 1.42× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 3.94009 ± 0.00760853 (W 0.00701945) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.3897 ± 0.0512815 (W 0.0155648) | 24.7812 ± 0.0934684 (W 0.0784874) | 🔵 faer-fork 1.51× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 16.4232 ± 0.0210056 (W 0.0317957) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.329713 ± 0.00210514 (W 0.00204205) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.328754 ± 0.00214173 (W 0.00304076) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.88179 ± 0.0104565 (W 0.00871993) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.13315 ± 0.0086498 (W 0.00801234) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.2246 ± 0.0303427 (W 0.0171669) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.4177 ± 0.0666506 (W 0.0237559) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.056205 ± 0.00563396 (W 0.000650726) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.05601 ± 0.00050432 (W 0.000814441) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.288833 ± 0.000893316 (W 0.00117859) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.291569 ± 0.00202732 (W 0.00179218) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 2.30278 ± 0.0109498 (W 0.0121552) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 2.30803 ± 0.00988402 (W 0.00778247) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.079352 ± 0.000647015 (W 0.000500336) | 0.129582 ± 0.000642427 (W 0.000522957) | 🔵 faer-fork 1.63× faster than `lapack-aocl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.13545 ± 0.00144431 (W 0.0013343) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.290677 ± 0.0010981 (W 0.00176063) | 0.360829 ± 0.00436088 (W 0.00405032) | 🔵 faer-fork 1.24× faster than `lapack-aocl` |  |
| `lu/n256/b1 [api-inclusive]` | 0.58159 ± 0.00259898 (W 0.00321547) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 1.20735 ± 0.00603482 (W 0.00483168) | 2.85289 ± 0.00741999 (W 0.00781608) | 🔵 faer-fork 2.36× faster than `lapack-aocl` |  |
| `lu/n512/b1 [api-inclusive]` | 2.61331 ± 0.00727295 (W 0.00717106) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.273221 ± 0.0010155 (W 0.00139295) | 0.486795 ± 0.0053637 (W 0.00972485) | 🔵 faer-fork 1.78× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.380575 ± 0.00172405 (W 0.00326729) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.804368 ± 0.0022664 (W 0.00192976) | 1.44605 ± 0.00622123 (W 0.0136669) | 🔵 faer-fork 1.80× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 1.28781 ± 0.00221459 (W 0.00744268) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.19116 ± 0.00886098 (W 0.00713523) | 6.50716 ± 0.0181296 (W 0.0347982) | 🔵 faer-fork 2.04× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 5.46015 ± 0.0118909 (W 0.00950292) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.194232 ± 0.000884557 (W 0.00117909) | 0.249668 ± 0.00393629 (W 0.00476378) | 🔵 faer-fork 1.29× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 0.567779 ± 0.00180361 (W 0.00252939) | 0.789714 ± 0.0045048 (W 0.00759465) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 1.76361 ± 0.00742199 (W 0.00364911) | 3.10595 ± 0.0126614 (W 0.0184033) | 🔵 faer-fork 1.76× faster than `lapack-aocl` |  |
| `solve/n128/b1` | 0.079398 ± 0.000474262 (W 0.000438012) | 0.110235 ± 0.00158149 (W 0.000837121) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.0950365 ± 0.000541633 (W 0.000535577) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.279546 ± 0.000929539 (W 0.00141691) | 0.284357 ± 0.00349755 (W 0.0051748) | · |  |
| `solve/n256/b1 [api-inclusive]` | 0.592935 ± 0.00274517 (W 0.0032139) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.09135 ± 0.00743914 (W 0.00217247) | 1.31094 ± 0.0120581 (W 0.00660071) | 🔵 faer-fork 1.20× faster than `lapack-aocl` |  |
| `solve/n512/b1 [api-inclusive]` | 2.59391 ± 0.0117106 (W 0.00679065) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.96707 ± 0.00933884 (W 0.00230965) | 2.52664 ± 0.00849784 (W 0.0156568) | 🔵 faer-fork 1.28× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 1.97165 ± 0.010815 (W 0.00331643) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.2779 ± 0.021859 (W 0.0154531) | 15.4265 ± 0.0971918 (W 0.0643847) | 🔵 faer-fork 2.12× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 7.29492 ± 0.0237312 (W 0.00817488) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 29.4413 ± 0.0542612 (W 0.0311167) | 61.0306 ± 0.496968 (W 0.181059) | 🔵 faer-fork 2.07× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 29.5314 ± 0.0526034 (W 0.0667739) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.982523 ± 0.00501677 (W 0.00279902) | 1.18915 ± 0.00442834 (W 0.00276704) | 🔵 faer-fork 1.21× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 0.985257 ± 0.00573238 (W 0.00326011) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.89807 ± 0.0142133 (W 0.009004) | 10.4139 ± 0.176565 (W 0.0584506) | 🔵 faer-fork 2.67× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_values/n256/b1 [api-inclusive]` | 3.91672 ± 0.0258101 (W 0.00769758) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 15.4027 ± 0.054143 (W 0.0191703) | 36.5416 ± 0.0751412 (W 0.172782) | 🔵 faer-fork 2.37× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 15.4509 ± 0.0251647 (W 0.0174569) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005836 ± 6.09098e-05 (W 7.12644e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0152045 ± 0.000121778 (W 0.000218549) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.037592 ± 0.000224362 (W 0.000354648) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.181316 ± 0.00105352 (W 0.00190701) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.189929 ± 0.000645079 (W 0.0018724) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.1453 ± 0.00436057 (W 0.0035156) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 1.16075 ± 0.00251114 (W 0.00309708) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.53996 ± 0.0286576 (W 0.0142151) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 7.67324 ± 0.0217099 (W 0.0116279) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.54795 ± 0.0135503 (W 0.00367812) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 3.53736 ± 0.0154578 (W 0.00456026) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 17.9988 ± 0.0706955 (W 0.0156328) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 18.0616 ± 0.0523249 (W 0.0242927) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 106.736 ± 0.344881 (W 0.299291) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 106.87 ± 0.246865 (W 0.102405) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.834188 ± 0.00771148 (W 0.00500975) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.924862 ± 0.00755657 (W 0.00409861) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 6.07786 ± 0.0252481 (W 0.0342313) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.57766 ± 0.0255204 (W 0.0166241) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 41.1114 ± 0.13898 (W 0.0933131) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 44.1883 ± 0.0882486 (W 0.098941) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.05059 ± 0.00589756 (W 0.00309154) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.0532 ± 0.0042431 (W 0.0022793) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.51032 ± 0.0224363 (W 0.00947512) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.52422 ± 0.0234311 (W 0.0121902) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.6932 ± 0.13357 (W 0.128901) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.7394 ± 0.248311 (W 0.120992) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.290508 ± 0.0012771 (W 0.00229249) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.441099 ± 0.00398167 (W 0.00337462) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.91756 ± 0.00527445 (W 0.004132) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 2.67291 ± 0.00922849 (W 0.00708251) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.4382 ± 0.0398071 (W 0.023143) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 17.9404 ± 0.15561 (W 0.0506368) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.4702 ± 0.00435556 (W 0.00353076) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 1.70778 ± 0.00664767 (W 0.00423962) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 10.1751 ± 0.0397567 (W 0.0157231) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 10.9935 ± 0.0495943 (W 0.0151169) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 69.9667 ± 0.181771 (W 0.0885394) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 73.8206 ± 0.182353 (W 0.0558536) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.678179 ± 0.00424865 (W 0.00234746) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 4.63366 ± 0.0194272 (W 0.00921595) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 29.9969 ± 0.0557072 (W 0.0645247) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.279764 ± 0.00136955 (W 0.0017395) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.448034 ± 0.00293742 (W 0.00290163) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.84354 ± 0.00588614 (W 0.00412661) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 2.09458 ± 0.00898046 (W 0.0038063) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.0953 ± 0.0243439 (W 0.0276585) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 14.8009 ± 0.0368794 (W 0.0361771) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.75522 ± 0.01717 (W 0.00511073) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 4.78132 ± 0.0221211 (W 0.00849463) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.3758 ± 0.128725 (W 0.0451567) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 29.4032 ± 0.0901084 (W 0.0435007) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 194.171 ± 0.794685 (W 0.643783) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 194.673 ± 0.30996 (W 0.551008) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.61937 ± 0.00508424 (W 0.00352074) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.61855 ± 0.00771796 (W 0.00456845) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.3324 ± 0.0582574 (W 0.0229611) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 11.3328 ± 0.0413915 (W 0.0272603) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 82.6473 ± 0.756959 (W 0.568285) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 82.6761 ± 0.272878 (W 0.202467) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007945 ± 4.78153e-05 (W 6.66523e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024388 ± 0.000259535 (W 7.74202e-05) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.089777 ± 0.000623157 (W 0.000414878) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.13936 ± 0.00112458 (W 0.00156036) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.156362 ± 0.00140084 (W 0.0018061) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.531172 ± 0.00146928 (W 0.00272638) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.61084 ± 0.002715 (W 0.00261238) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.38642 ± 0.00864335 (W 0.00467421) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.77589 ± 0.0136965 (W 0.00560394) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.9419 ± 0.0113385 (W 0.00358703) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.94414 ± 0.00452983 (W 0.00362628) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.91452 ± 0.019657 (W 0.0127281) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 7.92702 ± 0.010465 (W 0.00997786) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.9125 ± 0.0826851 (W 0.0359141) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 40.026 ± 0.0537132 (W 0.0340412) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.605104 ± 0.00352743 (W 0.00536671) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.729531 ± 0.00424529 (W 0.00480313) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.36753 ± 0.00891592 (W 0.0092845) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 4.08941 ± 0.0135838 (W 0.0147308) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 16.7831 ± 0.0522609 (W 0.0397568) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 20.8888 ± 0.127153 (W 0.0651012) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.286752 ± 0.00118296 (W 0.00130355) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.287843 ± 0.000849324 (W 0.00149636) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.95214 ± 0.00459497 (W 0.00259282) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.95184 ± 0.00570196 (W 0.00409738) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6163 ± 0.0322846 (W 0.0171093) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6007 ± 0.0302246 (W 0.0184278) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.232424 ± 0.00195269 (W 0.00239017) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.364986 ± 0.00314974 (W 0.00266966) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.01136 ± 0.00670583 (W 0.00724636) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.73069 ± 0.00904761 (W 0.00830232) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 5.10702 ± 0.0304256 (W 0.0507162) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 9.43524 ± 0.0898587 (W 0.0436389) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.729672 ± 0.00363386 (W 0.00248163) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.93393 ± 0.00312153 (W 0.0023439) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.12566 ± 0.0100998 (W 0.00573916) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 4.20395 ± 0.0293862 (W 0.010279) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 20.0753 ± 0.152576 (W 0.0282594) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 25.2743 ± 0.0524696 (W 0.0336754) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.445032 ± 0.00244936 (W 0.00221143) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.64608 ± 0.00466383 (W 0.00247192) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 9.41436 ± 0.0197175 (W 0.0190804) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.195779 ± 0.000800317 (W 0.00124696) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.370045 ± 0.00369644 (W 0.00456375) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.977174 ± 0.00822478 (W 0.0102384) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.74308 ± 0.0129091 (W 0.0106448) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 5.14272 ± 0.0430709 (W 0.0517694) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 9.35687 ± 0.123883 (W 0.0568792) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.70481 ± 0.0045203 (W 0.00459346) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.73581 ± 0.0119568 (W 0.00550303) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.8474 ± 0.011679 (W 0.024822) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 12.8753 ± 0.0248474 (W 0.0181266) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 68.8969 ± 0.146578 (W 0.088586) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 69.0437 ± 0.0744075 (W 0.0817126) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.27391 ± 0.00233943 (W 0.00346401) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.27655 ± 0.0111925 (W 0.00535246) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.26319 ± 0.00615506 (W 0.0118097) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.27241 ± 0.0262138 (W 0.0104629) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 33.7966 ± 0.177011 (W 0.0538824) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 33.7804 ± 0.181141 (W 0.0818828) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009127 ± 4.96887e-05 (W 5.99872e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0218775 ± 0.000107576 (W 0.000172397) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0561935 ± 0.00029889 (W 0.000303353) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.137817 ± 0.00118183 (W 0.0016343) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.156306 ± 0.000424805 (W 0.00118163) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.457858 ± 0.00156907 (W 0.00342924) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.54189 ± 0.00120336 (W 0.00191385) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.64042 ± 0.0024789 (W 0.00558456) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.09867 ± 0.00876361 (W 0.00697029) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.34023 ± 0.00282819 (W 0.00248276) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.34155 ± 0.00447263 (W 0.0020821) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.46969 ± 0.00910587 (W 0.00558436) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.48222 ± 0.00910625 (W 0.00680466) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.3777 ± 0.047267 (W 0.0245954) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 25.4211 ± 0.0186836 (W 0.0301878) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.575875 ± 0.00479549 (W 0.00442381) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.715491 ± 0.00305562 (W 0.00467626) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.00376 ± 0.013178 (W 0.00792698) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.7422 ± 0.0113155 (W 0.00800443) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.8783 ± 0.0624295 (W 0.028805) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 17.1218 ± 0.0477523 (W 0.0317136) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.179144 ± 0.000777238 (W 0.00151685) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.178672 ± 0.000642249 (W 0.00148749) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00696 ± 0.00414546 (W 0.00389113) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.01282 ± 0.00350807 (W 0.00221542) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.97809 ± 0.0379948 (W 0.0565638) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.91991 ± 0.058671 (W 0.0541946) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.235393 ± 0.00120294 (W 0.00134893) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.351686 ± 0.00110365 (W 0.00269952) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.793451 ± 0.00187299 (W 0.00203549) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.5647 ± 0.00718624 (W 0.00351493) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.47238 ± 0.0123067 (W 0.00987494) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 7.88417 ± 0.0687547 (W 0.0303492) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.577809 ± 0.00254008 (W 0.00205165) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.806712 ± 0.00263738 (W 0.00460909) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.10992 ± 0.00643712 (W 0.00509219) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 3.28467 ± 0.00956346 (W 0.00708758) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.6594 ± 0.0227839 (W 0.0127548) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 16.4364 ± 0.0699616 (W 0.0311246) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.388204 ± 0.000790089 (W 0.00213523) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.32646 ± 0.00604061 (W 0.00387746) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.10503 ± 0.0204947 (W 0.00610474) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.189356 ± 0.000706594 (W 0.00126005) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.357352 ± 0.00174599 (W 0.00233874) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.789859 ± 0.00338713 (W 0.00253909) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.58169 ± 0.00503067 (W 0.00412057) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.39067 ± 0.0177152 (W 0.00679499) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.81002 ± 0.0315622 (W 0.0129709) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.52492 ± 0.0138179 (W 0.00307131) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.53155 ± 0.0121562 (W 0.00472608) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.8029 ± 0.0312422 (W 0.0160909) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 10.8341 ± 0.0219475 (W 0.0188458) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.4493 ± 0.152065 (W 0.0451046) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.7212 ± 0.112537 (W 0.058571) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.27843 ± 0.00496841 (W 0.00284872) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.27635 ± 0.00431337 (W 0.00334434) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.7572 ± 0.0370428 (W 0.00822721) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 5.77024 ± 0.0235416 (W 0.00765485) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 26.1609 ± 0.097948 (W 0.0610176) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 26.1124 ± 0.148294 (W 0.0465218) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0091825 ± 2.31142e-05 (W 6.51744e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.02136 ± 9.20905e-05 (W 0.000153771) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0505945 ± 0.000464601 (W 0.000185401) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.166339 ± 0.000666535 (W 0.00160843) | 0.276735 ± 0.00131426 (W 0.00203055) | 🔵 faer-fork 1.66× faster than `lapack-aocl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.178567 ± 0.000982726 (W 0.00165663) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 1.07086 ± 0.00316455 (W 0.00310758) | 1.26569 ± 0.00818892 (W 0.00194647) | 🔵 faer-fork 1.18× faster than `lapack-aocl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 1.11711 ± 0.00252233 (W 0.00279635) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 7.29423 ± 0.0173958 (W 0.0134393) | 7.38011 ± 0.0179603 (W 0.0150642) | · |  |
| `cholesky/n512/b1 [api-inclusive]` | 7.48486 ± 0.0148355 (W 0.0128627) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 3.54745 ± 0.0158388 (W 0.00799738) | 3.55574 ± 0.0112845 (W 0.00391784) | · |  |
| `eigh/n128/b1 [api-inclusive]` | 3.53866 ± 0.016296 (W 0.00711918) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 17.9531 ± 0.0620404 (W 0.0166403) | 20.412 ± 0.0807809 (W 0.0340363) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 18.0628 ± 0.063546 (W 0.0262805) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 106.618 ± 0.336815 (W 0.18694) | 141.158 ± 1.60087 (W 0.419709) | 🔵 faer-fork 1.32× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 106.616 ± 0.30657 (W 0.264857) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.795749 ± 0.00879435 (W 0.00537433) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.894591 ± 0.00656215 (W 0.00555144) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 5.96483 ± 0.0279047 (W 0.0152526) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 6.41039 ± 0.0254357 (W 0.0137883) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 42.7637 ± 0.260392 (W 0.0471333) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 42.8443 ± 0.171917 (W 0.0561342) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 1.03879 ± 0.00522024 (W 0.0028938) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 1.037 ± 0.00584126 (W 0.000987833) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 7.45042 ± 0.0229541 (W 0.00995737) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 7.43637 ± 0.0214166 (W 0.0125893) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 56.5262 ± 0.140564 (W 0.0826911) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 56.4928 ± 0.161493 (W 0.0986296) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.285606 ± 0.00153608 (W 0.00223049) | 0.348128 ± 0.00163236 (W 0.00337869) | 🔵 faer-fork 1.22× faster than `lapack-aocl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.414589 ± 0.00222894 (W 0.00247937) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.88707 ± 0.00790882 (W 0.0039297) | 3.38992 ± 0.00800062 (W 0.0120006) | 🔵 faer-fork 1.80× faster than `lapack-aocl` |  |
| `lu/n256/b1 [api-inclusive]` | 2.52119 ± 0.0102669 (W 0.0039607) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 13.2585 ± 0.049624 (W 0.0354246) | 18.5448 ± 0.073531 (W 0.03306) | 🔵 faer-fork 1.40× faster than `lapack-aocl` |  |
| `lu/n512/b1 [api-inclusive]` | 16.8141 ± 0.0225793 (W 0.0268673) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 1.49721 ± 0.00674122 (W 0.00249336) | 2.05044 ± 0.0105097 (W 0.00335029) | 🔵 faer-fork 1.37× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 1.68379 ± 0.00698122 (W 0.00499596) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 9.88502 ± 0.0397314 (W 0.0117354) | 13.4128 ± 0.0243099 (W 0.0268588) | 🔵 faer-fork 1.36× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 10.8485 ± 0.031977 (W 0.022106) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 68.1194 ± 0.390578 (W 0.282026) | 85.9675 ± 0.380224 (W 0.0898319) | 🔵 faer-fork 1.26× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 72.7 ± 0.184896 (W 0.251272) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.709389 ± 0.00276443 (W 0.00308601) | 0.768007 ± 0.00240023 (W 0.00201208) | 🔵 faer-fork 1.08× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 4.36879 ± 0.0136902 (W 0.00879145) | 5.20306 ± 0.0244254 (W 0.0134714) | 🔵 faer-fork 1.19× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 28.2016 ± 0.0821705 (W 0.0259912) | 34.0407 ± 0.0928665 (W 0.0439758) | 🔵 faer-fork 1.21× faster than `lapack-aocl` |  |
| `solve/n128/b1` | 0.294343 ± 0.000846525 (W 0.00201396) | 0.254567 ± 0.00219419 (W 0.00205892) | 🔴 faer-fork 1.16× slower than `lapack-aocl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.422544 ± 0.00212472 (W 0.00373638) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.89275 ± 0.00736572 (W 0.00315773) | 2.02057 ± 0.00412039 (W 0.00399077) | 🔵 faer-fork 1.07× faster than `lapack-aocl` |  |
| `solve/n256/b1 [api-inclusive]` | 1.99254 ± 0.00975129 (W 0.00309295) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 13.2691 ± 0.0483944 (W 0.0208911) | 13.1843 ± 0.165266 (W 0.0260509) | · |  |
| `solve/n512/b1 [api-inclusive]` | 14.0061 ± 0.03355 (W 0.031251) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 4.74791 ± 0.0232395 (W 0.00780361) | 5.44524 ± 0.0220032 (W 0.00785581) | 🔵 faer-fork 1.15× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 4.77814 ± 0.0236975 (W 0.0082257) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 29.1796 ± 0.0991253 (W 0.0323846) | 35.7125 ± 0.169229 (W 0.0542143) | 🔵 faer-fork 1.22× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 29.1678 ± 0.103096 (W 0.032997) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 192.749 ± 0.469832 (W 0.518237) | 219.879 ± 0.907673 (W 0.702126) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 193.035 ± 0.60295 (W 0.364301) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.58334 ± 0.00458252 (W 0.00480144) | 2.07888 ± 0.0204434 (W 0.00570422) | 🔵 faer-fork 1.31× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.58738 ± 0.00594477 (W 0.00442273) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 11.1138 ± 0.0436996 (W 0.0290233) | 16.019 ± 0.133372 (W 0.0224308) | 🔵 faer-fork 1.44× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 11.0954 ± 0.0370806 (W 0.0262312) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 82.5665 ± 0.651707 (W 0.567847) | 95.5947 ± 0.987354 (W 0.306375) | 🔵 faer-fork 1.16× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 81.7503 ± 0.132992 (W 0.115203) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.00794 ± 6.72054e-05 (W 6.24609e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.024169 ± 0.000176231 (W 0.000141586) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0900685 ± 0.000673037 (W 0.00157384) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.13121 ± 0.00188778 (W 0.00221278) | 0.238505 ± 0.00243306 (W 0.00325873) | 🔵 faer-fork 1.82× faster than `lapack-aocl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.146856 ± 0.00166898 (W 0.00132907) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.495815 ± 0.00285671 (W 0.00257027) | 0.71081 ± 0.00305938 (W 0.00525652) | 🔵 faer-fork 1.43× faster than `lapack-aocl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.569726 ± 0.00256241 (W 0.0035147) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.26371 ± 0.00700666 (W 0.0064415) | 2.75917 ± 0.01379 (W 0.00758057) | 🔵 faer-fork 1.22× faster than `lapack-aocl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 2.58898 ± 0.0102648 (W 0.00558583) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.92651 ± 0.00455199 (W 0.00212837) | 3.81288 ± 0.0156347 (W 0.00361376) | 🔵 faer-fork 1.98× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.92944 ± 0.00712559 (W 0.00289409) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 7.84833 ± 0.0228378 (W 0.0104227) | 21.6628 ± 0.0927674 (W 0.0180056) | 🔵 faer-fork 2.76× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 7.86627 ± 0.0129189 (W 0.0140859) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 39.6247 ± 0.0386729 (W 0.0358285) | 81.5624 ± 0.0821543 (W 0.108966) | 🔵 faer-fork 2.06× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 39.7341 ± 0.0543054 (W 0.0320193) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.58206 ± 0.00354589 (W 0.00462093) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.699643 ± 0.0020409 (W 0.0039104) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.27998 ± 0.018352 (W 0.0127652) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.89246 ± 0.0143555 (W 0.00790328) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 16.9044 ± 0.0864666 (W 0.0150756) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 19.6482 ± 0.0579245 (W 0.0309208) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.28584 ± 0.000986471 (W 0.00112927) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.286014 ± 0.000547097 (W 0.00174392) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.94862 ± 0.00546055 (W 0.00373427) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.94527 ± 0.0052877 (W 0.0025984) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 14.6472 ± 0.0329996 (W 0.0267282) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 14.6496 ± 0.0136135 (W 0.0155547) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.20316 ± 0.00407845 (W 0.0027349) | 0.125343 ± 0.00111548 (W 0.00102027) | 🔴 faer-fork 1.62× slower than `lapack-aocl` | `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.339386 ± 0.00420167 (W 0.00258755) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.911181 ± 0.0121007 (W 0.006027) | 0.946387 ± 0.00375243 (W 0.00381705) | · |  |
| `lu/n256/b1 [api-inclusive]` | 1.58206 ± 0.018207 (W 0.0059821) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 4.71602 ± 0.0469274 (W 0.0496328) | 5.83049 ± 0.0126913 (W 0.00833187) | 🔵 faer-fork 1.24× faster than `lapack-aocl` |  |
| `lu/n512/b1 [api-inclusive]` | 8.24192 ± 0.037533 (W 0.0235109) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.688514 ± 0.00191302 (W 0.00185912) | 0.948881 ± 0.00651402 (W 0.00458539) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.911968 ± 0.00442137 (W 0.00329975) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 3.0211 ± 0.00483584 (W 0.00299416) | 4.61701 ± 0.0125664 (W 0.00999882) | 🔵 faer-fork 1.53× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 4.03734 ± 0.012972 (W 0.00856867) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 19.1637 ± 0.0493608 (W 0.0178227) | 29.2105 ± 0.0588697 (W 0.0340102) | 🔵 faer-fork 1.52× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 24.2321 ± 0.027142 (W 0.0315791) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.4149 ± 0.00223109 (W 0.0019249) | 0.474051 ± 0.00176103 (W 0.00222976) | 🔵 faer-fork 1.14× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.55045 ± 0.00618921 (W 0.00370002) | 2.02926 ± 0.00475701 (W 0.00494) | 🔵 faer-fork 1.31× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 8.91673 ± 0.0176104 (W 0.00923727) | 14.3542 ± 0.0289908 (W 0.0201667) | 🔵 faer-fork 1.61× faster than `lapack-aocl` |  |
| `solve/n128/b1` | 0.20163 ± 0.00430347 (W 0.0024135) | 0.111122 ± 0.000915491 (W 0.00150256) | 🔴 faer-fork 1.81× slower than `lapack-aocl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.344334 ± 0.00296814 (W 0.00429243) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.882612 ± 0.00960546 (W 0.0133578) | 0.653739 ± 0.00260628 (W 0.00192053) | 🔴 faer-fork 1.35× slower than `lapack-aocl` |  |
| `solve/n256/b1 [api-inclusive]` | 1.58856 ± 0.0147048 (W 0.00900708) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 4.61475 ± 0.0121238 (W 0.0285152) | 4.3988 ± 0.0103478 (W 0.00708866) | · |  |
| `solve/n512/b1 [api-inclusive]` | 8.24765 ± 0.033543 (W 0.0400595) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.68519 ± 0.0065563 (W 0.00670739) | 5.21402 ± 0.0238377 (W 3.31935) | 🔵 faer-fork 1.94× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.69537 ± 0.00833631 (W 0.00593004) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 12.6502 ± 0.0257676 (W 0.0221203) | 22.0612 ± 0.101524 (W 0.0784212) | 🔵 faer-fork 1.74× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 12.6865 ± 0.0311365 (W 0.0316762) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 67.8993 ± 0.117556 (W 0.0886457) | 109.147 ± 0.197717 (W 0.342381) | 🔵 faer-fork 1.61× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 68.1584 ± 0.0625867 (W 0.106634) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.25408 ± 0.00835007 (W 0.0046264) | 3.23034 ± 0.0211828 (W 2.43026) | 🔵 faer-fork 2.58× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.25619 ± 0.00747042 (W 0.00453642) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.08164 ± 0.0126097 (W 0.0141329) | 12.816 ± 0.0355717 (W 1.12121) | 🔵 faer-fork 2.11× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 6.08623 ± 0.0156641 (W 0.00900885) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 32.855 ± 0.184455 (W 0.063981) | 58.473 ± 0.250166 (W 0.0948108) | 🔵 faer-fork 1.78× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 32.8578 ± 0.204414 (W 0.0469863) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009077 ± 0.000101942 (W 8.54181e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0224015 ± 8.64517e-05 (W 0.000172008) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0579865 ± 0.000305558 (W 0.000384621) | - | - | `*` one row has no measurement here |

## linalg-mid, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.130577 ± 0.0012171 (W 0.00109341) | 0.261061 ± 0.00289718 (W 0.00840975) | 🔵 faer-fork 2.00× faster than `lapack-aocl` |  |
| `cholesky/n128/b1 [api-inclusive]` | 0.145155 ± 0.00149493 (W 0.00139885) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.427675 ± 0.00224733 (W 0.00273943) | 0.700595 ± 0.00707351 (W 0.00675313) | 🔵 faer-fork 1.64× faster than `lapack-aocl` |  |
| `cholesky/n256/b1 [api-inclusive]` | 0.501618 ± 0.00147846 (W 0.00225261) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.57384 ± 0.00307127 (W 0.00404475) | 2.1751 ± 0.0116438 (W 0.0154801) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `cholesky/n512/b1 [api-inclusive]` | 1.90669 ± 0.0060123 (W 0.00454643) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32062 ± 0.00230963 (W 0.00339548) | 3.97593 ± 0.00862823 (W 0.00879768) | 🔵 faer-fork 3.01× faster than `lapack-aocl` |  |
| `eigh/n128/b1 [api-inclusive]` | 1.3267 ± 0.00574491 (W 0.00259243) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.39807 ± 0.0101652 (W 0.00844511) | 22.481 ± 0.044701 (W 0.0280151) | 🔵 faer-fork 4.16× faster than `lapack-aocl` |  |
| `eigh/n256/b1 [api-inclusive]` | 5.43039 ± 0.0148535 (W 0.00624301) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 25.1212 ± 0.0567263 (W 0.0294681) | 73.3883 ± 0.22339 (W 0.0700934) | 🔵 faer-fork 2.92× faster than `lapack-aocl` |  |
| `eigh/n512/b1 [api-inclusive]` | 25.1861 ± 0.110242 (W 0.0243975) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.563487 ± 0.00652426 (W 0.00382783) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.688005 ± 0.00618754 (W 0.00410136) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.91645 ± 0.00737177 (W 0.00920116) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.56956 ± 0.00840589 (W 0.0109903) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.99 ± 0.0286699 (W 0.0335441) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 15.998 ± 0.0412006 (W 0.0265893) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.179127 ± 0.000655181 (W 0.00103205) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.179412 ± 0.00130983 (W 0.00145294) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.0229 ± 0.00545084 (W 0.00333632) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.02631 ± 0.00544527 (W 0.00330073) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 8.04178 ± 0.0480375 (W 0.112058) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.93997 ± 0.160776 (W 0.0483061) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.190457 ± 0.000836957 (W 0.00129663) | 0.12048 ± 0.00149874 (W 0.00129937) | 🔴 faer-fork 1.58× slower than `lapack-aocl` |  |
| `lu/n128/b1 [api-inclusive]` | 0.325196 ± 0.00418775 (W 0.00239497) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.754548 ± 0.00463897 (W 0.00317315) | 0.66186 ± 0.00268387 (W 0.0032158) | 🔴 faer-fork 1.14× slower than `lapack-aocl` |  |
| `lu/n256/b1 [api-inclusive]` | 1.41936 ± 0.00893729 (W 0.00397517) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.21441 ± 0.0160761 (W 0.00750372) | 2.94787 ± 0.00844574 (W 0.00662711) | 🔴 faer-fork 1.09× slower than `lapack-aocl` |  |
| `lu/n512/b1 [api-inclusive]` | 6.67407 ± 0.010278 (W 0.0208645) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.547395 ± 0.00354848 (W 0.00267277) | 0.834197 ± 0.00461879 (W 0.0123245) | 🔵 faer-fork 1.52× faster than `lapack-aocl` |  |
| `qr/n128/b1 [api-inclusive]` | 0.782004 ± 0.00228595 (W 0.00423226) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 2.04764 ± 0.00742157 (W 0.00440969) | 3.05989 ± 0.0143884 (W 0.0127563) | 🔵 faer-fork 1.49× faster than `lapack-aocl` |  |
| `qr/n256/b1 [api-inclusive]` | 3.10959 ± 0.01057 (W 0.0106684) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.0662 ± 0.0156419 (W 0.0123) | 17.0456 ± 0.0284859 (W 0.0315594) | 🔵 faer-fork 1.69× faster than `lapack-aocl` |  |
| `qr/n512/b1 [api-inclusive]` | 15.3795 ± 0.0852875 (W 0.0303916) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.366151 ± 0.00176567 (W 0.0020524) | 0.450341 ± 0.00305683 (W 0.0058785) | 🔵 faer-fork 1.23× faster than `lapack-aocl` |  |
| `qr_factor/n256/b1` | 1.26492 ± 0.00571662 (W 0.00344759) | 1.51283 ± 0.00374169 (W 0.00604533) | 🔵 faer-fork 1.20× faster than `lapack-aocl` |  |
| `qr_factor/n512/b1` | 4.78501 ± 0.0136529 (W 0.0108117) | 8.4367 ± 0.0151275 (W 0.0195253) | 🔵 faer-fork 1.76× faster than `lapack-aocl` |  |
| `solve/n128/b1` | 0.191828 ± 0.000771447 (W 0.00141473) | 0.115812 ± 0.0011937 (W 0.0011832) | 🔴 faer-fork 1.66× slower than `lapack-aocl` |  |
| `solve/n128/b1 [api-inclusive]` | 0.331171 ± 0.00228335 (W 0.00226719) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.724233 ± 0.00341427 (W 0.00277853) | 0.522846 ± 0.00311576 (W 0.00197227) | 🔴 faer-fork 1.39× slower than `lapack-aocl` |  |
| `solve/n256/b1 [api-inclusive]` | 1.43145 ± 0.00949723 (W 0.00517258) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 3.11798 ± 0.0169084 (W 0.00684086) | 2.34955 ± 0.0098502 (W 0.0085914) | 🔴 faer-fork 1.33× slower than `lapack-aocl` |  |
| `solve/n512/b1 [api-inclusive]` | 6.72972 ± 0.0586024 (W 0.0506073) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.50063 ± 0.00724443 (W 0.00332577) | 6.85373 ± 0.0487803 (W 4.95176) | 🔵 faer-fork 2.74× faster than `lapack-aocl` |  |
| `svd_thin/n128/b1 [api-inclusive]` | 2.50825 ± 0.0128007 (W 0.00446859) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 10.579 ± 0.0438313 (W 0.0213748) | 21.867 ± 0.20465 (W 4.56342) | 🔵 faer-fork 2.07× faster than `lapack-aocl` |  |
| `svd_thin/n256/b1 [api-inclusive]` | 10.6131 ± 0.0294797 (W 0.0146628) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 48.451 ± 0.142948 (W 0.0650021) | 92.4445 ± 0.339837 (W 0.392652) | 🔵 faer-fork 1.91× faster than `lapack-aocl` |  |
| `svd_thin/n512/b1 [api-inclusive]` | 48.7586 ± 0.200925 (W 0.0523631) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.25107 ± 0.00475433 (W 0.00283724) | 4.77075 ± 0.0341646 (W 5.18536) | 🔵 faer-fork 3.81× faster than `lapack-aocl` |  |
| `svd_values/n128/b1 [api-inclusive]` | 1.25319 ± 0.00301924 (W 0.00328575) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 5.59812 ± 0.0262997 (W 0.00796551) | 14.2128 ± 0.136709 (W 4.19517) | 🔵 faer-fork 2.54× faster than `lapack-aocl` |  |
| `svd_values/n256/b1 [api-inclusive]` | 5.58921 ± 0.0168654 (W 0.00622917) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 25.3193 ± 0.154907 (W 0.141865) | 52.5498 ± 0.103127 (W 0.0928004) | 🔵 faer-fork 2.08× faster than `lapack-aocl` |  |
| `svd_values/n512/b1 [api-inclusive]` | 25.2916 ± 0.107342 (W 0.0499176) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.009132 ± 6.28861e-05 (W 6.59821e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.023043 ± 0.000125511 (W 0.000182682) | - | - | `*` one row has no measurement here |
| `triangular_solve/n512/b1` | 0.0579765 ± 0.000171869 (W 0.000977388) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.048159 ± 0.00148958 (W 0.000172581) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.04053 ± 0.00142076 (W 0.000569734) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.218842 ± 0.00151297 (W 0.00192236) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.231192 ± 0.000970317 (W 0.00262725) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.4009 ± 0.0106858 (W 0.00711842) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.27999 ± 0.00767175 (W 0.00350112) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.74501 ± 0.00925456 (W 0.00361743) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.73893 ± 0.0110527 (W 0.0027711) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.52451 ± 0.0215256 (W 0.0144885) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.53153 ± 0.025482 (W 0.00985202) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.4267 ± 0.144734 (W 0.0139938) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.2753 ± 0.242262 (W 0.0963807) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.26617 ± 0.00199443 (W 0.00554862) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.236295 ± 0.00094192 (W 0.00147155) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.45981 ± 0.0053795 (W 0.0064018) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.45177 ± 0.00805704 (W 0.0097528) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.1559 ± 0.0384195 (W 0.0345922) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.1787 ± 0.0390236 (W 0.0308932) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.147227 ± 0.000534871 (W 0.000391433) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.147832 ± 0.000908493 (W 0.000831419) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.988208 ± 0.00531722 (W 0.00132095) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.993366 ± 0.00497472 (W 0.00211165) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.37528 ± 0.0267432 (W 0.00741413) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.39616 ± 0.0292572 (W 0.0185955) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0777795 ± 0.000730734 (W 0.000136761) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0951125 ± 0.000716952 (W 0.00136373) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.367347 ± 0.00283569 (W 0.00191739) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.565068 ± 0.00295957 (W 0.00323195) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.11571 ± 0.00404323 (W 0.00384361) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.12681 ± 0.0334354 (W 0.00647596) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.352587 ± 0.00163843 (W 0.00140007) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.456542 ± 0.00436455 (W 0.00228274) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.82605 ± 0.00930203 (W 0.00397965) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.91828 ± 0.0063016 (W 0.00487387) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 10.519 ± 0.0519186 (W 0.0205087) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 10.9048 ± 0.0412936 (W 0.0125539) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.186263 ± 0.00065762 (W 0.00143293) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.04994 ± 0.00669399 (W 0.00105134) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.22567 ± 0.0278296 (W 0.0133388) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.069879 ± 0.000434663 (W 0.00027384) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.09682 ± 0.000687451 (W 0.000739584) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.329817 ± 0.00151608 (W 0.00219991) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.453689 ± 0.00216616 (W 0.00271942) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.95375 ± 0.00767727 (W 0.00342495) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.55416 ± 0.0341615 (W 0.00594922) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.12495 ± 0.0143781 (W 0.00403245) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.13447 ± 0.0353031 (W 0.00352713) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.14319 ± 0.0350465 (W 0.00853654) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.14703 ± 0.0373355 (W 0.00608653) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.7139 ± 0.195072 (W 0.0621835) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.702 ± 0.134329 (W 0.0437739) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.655055 ± 0.0037158 (W 0.00191961) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.655951 ± 0.00316288 (W 0.00225767) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.33411 ± 0.0112955 (W 0.00817654) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.33481 ± 0.0126184 (W 0.00555174) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 20.1092 ± 0.136984 (W 0.019992) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 20.0511 ± 0.163616 (W 0.0250601) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.003577 ± 3.40808e-05 (W 4.73772e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.009278 ± 6.19516e-05 (W 6.47573e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0268695 ± 0.000257614 (W 0.000150615) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.044717 ± 0.00165337 (W 0.000158498) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.0381585 ± 0.00131278 (W 0.000203488) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.148515 ± 0.000603525 (W 0.000816655) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.171666 ± 0.00147656 (W 0.00302324) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.661099 ± 0.00852806 (W 0.00436727) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.664311 ± 0.0109819 (W 0.00261612) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.04054 ± 0.00295414 (W 0.00196624) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.0433 ± 0.00461527 (W 0.0028032) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.71196 ± 0.00553041 (W 0.00756507) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.73536 ± 0.0126408 (W 0.00539273) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.2555 ± 0.0736015 (W 0.0170347) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.1215 ± 0.0214415 (W 0.0174406) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.235738 ± 0.0018538 (W 0.00253241) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.24077 ± 0.0020946 (W 0.00167444) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.12005 ± 0.00527669 (W 0.00587273) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.2564 ± 0.00521946 (W 0.00613296) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.35059 ± 0.0229379 (W 0.0198526) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 8.03966 ± 0.0175554 (W 0.0202486) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.053224 ± 0.000297109 (W 0.000954437) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.053593 ± 0.000391816 (W 0.00052712) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.27891 ± 0.00201481 (W 0.00168323) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.278908 ± 0.00122609 (W 0.00167141) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.92746 ± 0.00284444 (W 0.00376302) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.92798 ± 0.00408631 (W 0.00443603) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0981515 ± 0.000798375 (W 0.0015153) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0865465 ± 0.00113607 (W 0.000719178) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.302239 ± 0.00286988 (W 0.00380964) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.444806 ± 0.00891326 (W 0.00315668) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n512/b1` | 1.26907 ± 0.00489828 (W 0.0064947) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.96774 ± 0.0283206 (W 0.00776803) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.287589 ± 0.00130823 (W 0.00165422) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.331421 ± 0.00196966 (W 0.00297532) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.775363 ± 0.00188029 (W 0.00213434) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.05693 ± 0.00412424 (W 0.00352374) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.65204 ± 0.00444866 (W 0.00860253) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.82762 ± 0.0198076 (W 0.0104846) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.200357 ± 0.00131889 (W 0.0015256) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.542934 ± 0.00202692 (W 0.00213526) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.25172 ± 0.0103835 (W 0.00423794) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.076954 ± 0.00113669 (W 0.00104096) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.091671 ± 0.000766392 (W 0.000397588) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.295757 ± 0.00293026 (W 0.00296411) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.453466 ± 0.00322793 (W 0.00367257) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.19172 ± 0.00336208 (W 0.00530239) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.94533 ± 0.0270221 (W 0.00844579) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.49754 ± 0.00326914 (W 0.00361123) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.50757 ± 0.00648425 (W 0.00544807) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.74917 ± 0.0235621 (W 0.0103075) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.74883 ± 0.0205586 (W 0.0109923) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.525 ± 0.0595639 (W 0.0311783) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 24.5597 ± 0.0284477 (W 0.0280422) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.689636 ± 0.00132717 (W 0.00150943) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.716975 ± 0.0419464 (W 0.00194515) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n256/b1` | 2.9663 ± 0.0140199 (W 0.0119938) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.97711 ± 0.0164535 (W 0.00939744) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 12.2795 ± 0.036305 (W 0.0195405) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 12.3024 ± 0.0303276 (W 0.0208208) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004539 ± 6.65552e-05 (W 5.17666e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0110865 ± 0.000118021 (W 6.22109e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0265095 ± 0.00026211 (W 0.000167468) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.042396 ± 0.00190392 (W 0.000223933) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.037867 ± 0.0012732 (W 0.000281603) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1` | 0.14143 ± 0.00082693 (W 0.000833985) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.167594 ± 0.000801486 (W 0.00256557) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.572658 ± 0.00341737 (W 0.00500566) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.570923 ± 0.0136556 (W 0.00196664) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1` | 0.740395 ± 0.00468827 (W 0.00175047) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.743623 ± 0.00372713 (W 0.00276718) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.76718 ± 0.00337989 (W 0.0038669) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.77024 ± 0.00824382 (W 0.00677061) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.7064 ± 0.0686667 (W 0.0240372) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.6477 ± 0.0376068 (W 0.0148154) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.238515 ± 0.00200122 (W 0.00160088) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.251254 ± 0.00260913 (W 0.00142221) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.08675 ± 0.00457399 (W 0.00586755) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.2404 ± 0.00597889 (W 0.00753643) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.76807 ± 0.0142513 (W 0.0210358) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.53866 ± 0.0163715 (W 0.022584) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0407605 ± 0.000693482 (W 0.00103697) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.040382 ± 0.000341425 (W 0.000685053) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.161503 ± 0.0051655 (W 0.00214048) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.162809 ± 0.00190841 (W 0.0010803) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.04425 ± 0.00517515 (W 0.00499696) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.04644 ± 0.00381358 (W 0.00584573) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.107937 ± 0.0018503 (W 0.00295687) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1 [api-inclusive]` | 0.087391 ± 0.00119802 (W 0.000680934) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.249366 ± 0.00121201 (W 0.00155982) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.433465 ± 0.00148477 (W 0.00273957) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.956019 ± 0.00600305 (W 0.00608536) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.7818 ± 0.00707826 (W 0.00604618) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.27089 ± 0.00185826 (W 0.00164791) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.321479 ± 0.000930488 (W 0.00341464) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.663389 ± 0.00375677 (W 0.00226026) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.980875 ± 0.00275518 (W 0.00323017) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.24884 ± 0.00500213 (W 0.00336561) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.60167 ± 0.0135864 (W 0.0137036) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.187662 ± 0.00123512 (W 0.00118214) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.501269 ± 0.00142123 (W 0.00183393) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.44435 ± 0.00593474 (W 0.00394548) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0812875 ± 0.00072623 (W 0.0018601) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0931485 ± 0.00115126 (W 0.00043394) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.253618 ± 0.0009687 (W 0.00107003) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.441454 ± 0.00361793 (W 0.00310583) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.903582 ± 0.00217281 (W 0.00445803) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.77665 ± 0.00393933 (W 0.00536633) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.49585 ± 0.0138161 (W 0.00267558) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.503 ± 0.00594394 (W 0.00314177) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.40008 ± 0.0171044 (W 0.0110511) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.40033 ± 0.0107776 (W 0.0047939) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 21.1879 ± 0.0218056 (W 0.0297214) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 21.2303 ± 0.0664408 (W 0.024457) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.709008 ± 0.0031276 (W 0.00185359) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.708278 ± 0.00191999 (W 0.00161227) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.94938 ± 0.00863626 (W 0.00650159) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.95732 ± 0.0115783 (W 0.0076873) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.2915 ± 0.0450306 (W 0.0167774) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.2533 ± 0.0380651 (W 0.0144644) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004564 ± 3.67718e-05 (W 4.4218e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0112465 ± 6.04174e-05 (W 5.72973e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0260985 ± 0.000191165 (W 0.000174769) | - | - | `*` one row has no measurement here |

## linalg-mid, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.034592 ± 0.00142909 (W 6.95166e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n128/b1 [api-inclusive]` | 0.0399025 ± 0.000220975 (W 0.000126316) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.175948 ± 0.00112307 (W 0.00130113) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.193545 ± 0.000768476 (W 0.00208772) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.07254 ± 0.00407232 (W 0.00299919) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.15105 ± 0.00414544 (W 0.00361004) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.74133 ± 0.00721317 (W 0.0034321) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.73965 ± 0.00911707 (W 0.00326992) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 6.51776 ± 0.0277458 (W 0.00832107) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 6.52923 ± 0.0296475 (W 0.00988404) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 29.3043 ± 0.216996 (W 0.025575) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 29.1924 ± 0.178798 (W 0.0259722) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.218593 ± 0.00388901 (W 0.0017654) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.215672 ± 0.0299905 (W 0.00168832) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1` | 1.31177 ± 0.00700124 (W 0.0087399) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.37549 ± 0.108322 (W 0.00803101) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1` | 12.2082 ± 0.511211 (W 0.0200055) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.8591 ± 0.420089 (W 0.0385563) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.15071 ± 0.00173453 (W 0.00134143) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.151451 ± 0.000709045 (W 0.00128217) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00923 ± 0.00260979 (W 0.00144965) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00766 ± 0.0077367 (W 0.00128724) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.51367 ± 0.0399066 (W 0.0126037) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.48632 ± 0.024472 (W 0.0186871) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.071378 ± 0.000239636 (W 0.000440304) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0772185 ± 0.000494713 (W 0.000778237) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.358801 ± 0.00104096 (W 0.00204122) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.483716 ± 0.00112917 (W 0.00343157) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.16382 ± 0.00806006 (W 0.00731883) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 2.76463 ± 0.00760251 (W 0.0103242) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.395007 ± 0.0019772 (W 0.00194053) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.429771 ± 0.00126517 (W 0.00282811) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.60463 ± 0.00956281 (W 0.00358314) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.79843 ± 0.00481166 (W 0.00458216) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 9.56763 ± 0.0451485 (W 0.0169033) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 10.4681 ± 0.0362588 (W 0.0184857) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.218636 ± 0.00200401 (W 0.0013748) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.835279 ± 0.00662682 (W 0.0016156) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 4.34255 ± 0.0193464 (W 0.00745243) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0700395 ± 0.000757589 (W 0.000832474) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.0756845 ± 0.000594009 (W 0.000319962) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.34455 ± 0.00293101 (W 0.00227502) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.369081 ± 0.0025617 (W 0.00224597) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.11167 ± 0.00688138 (W 0.00325081) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 2.20795 ± 0.0109318 (W 0.00369775) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 2.12164 ± 0.0175143 (W 0.00357168) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 2.13114 ± 0.00929138 (W 0.00395706) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 9.14026 ± 0.0362546 (W 0.0124908) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 9.14394 ± 0.0292609 (W 0.0117123) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 49.1651 ± 0.263088 (W 0.0970513) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 49.2987 ± 0.151897 (W 0.2055) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.648052 ± 0.00213552 (W 0.00231837) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.651253 ± 0.00286705 (W 0.00253806) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 3.21813 ± 0.0128231 (W 0.00815286) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 3.21795 ± 0.00922995 (W 0.00562802) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.4944 ± 0.172385 (W 0.0232657) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 19.4931 ± 0.136246 (W 0.0223546) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.004343 ± 0.000349387 (W 6.64047e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0114015 ± 5.14341e-05 (W 5.97919e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.030379 ± 0.00136834 (W 0.000101683) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.031198 ± 0.000124355 (W 0.000133773) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.037445 ± 0.000192508 (W 0.00023572) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.111468 ± 0.00366813 (W 0.00111371) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.134651 ± 0.000883429 (W 0.00082464) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 0.438277 ± 0.00180487 (W 0.00192027) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.534447 ± 0.00314618 (W 0.00185544) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.04014 ± 0.00470567 (W 0.0018247) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.0452 ± 0.00573117 (W 0.00217572) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.69135 ± 0.00588849 (W 0.00570836) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.70724 ± 0.00731778 (W 0.00802881) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 15.1083 ± 0.0981297 (W 0.0218278) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 15.009 ± 0.0200489 (W 0.0149973) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.224301 ± 0.00173298 (W 0.00100567) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.221831 ± 0.000859913 (W 0.00140422) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.06931 ± 0.00664627 (W 0.007237) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.17406 ± 0.0052257 (W 0.00558197) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 7.40752 ± 0.0203561 (W 0.016666) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.73706 ± 0.0099036 (W 0.0199496) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0544145 ± 0.000428165 (W 0.000362575) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.05463 ± 0.000382253 (W 0.000575445) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.286848 ± 0.000904793 (W 0.00143756) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.286841 ± 0.00155863 (W 0.00208758) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.95506 ± 0.00815163 (W 0.00362961) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.94873 ± 0.00733586 (W 0.00302762) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0612145 ± 0.00122433 (W 0.000506458) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.067726 ± 0.000928041 (W 0.000555672) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.221662 ± 0.00311966 (W 0.00194444) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.359891 ± 0.00356396 (W 0.00294251) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.987728 ± 0.00555978 (W 0.00678529) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.60686 ± 0.0158427 (W 0.0073417) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.264664 ± 0.00135446 (W 0.0016315) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.303356 ± 0.00129954 (W 0.00279852) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.699219 ± 0.00144033 (W 0.00203389) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.937629 ± 0.00335387 (W 0.0035078) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 3.3815 ± 0.0111355 (W 0.00619008) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 4.37697 ± 0.0226056 (W 0.0139286) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.180001 ± 0.00115422 (W 0.00142597) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.464762 ± 0.00346937 (W 0.00208786) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.97247 ± 0.00289586 (W 0.00454893) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.0634235 ± 0.00107453 (W 0.000488925) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.071023 ± 0.00106539 (W 0.000749702) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.224393 ± 0.00210974 (W 0.00169236) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.366474 ± 0.00162395 (W 0.00410296) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.941536 ± 0.0015458 (W 0.00382105) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.59555 ± 0.00612478 (W 0.00717807) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.50892 ± 0.00983138 (W 0.0024191) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.5131 ± 0.0109529 (W 0.00394717) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.68569 ± 0.0136457 (W 0.0121188) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.70287 ± 0.0154768 (W 0.00909275) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 24.0219 ± 0.0296248 (W 0.0326559) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 24.0816 ± 0.0420462 (W 0.0286767) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.685231 ± 0.00364214 (W 0.00211937) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.710064 ± 0.00279113 (W 0.00150058) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.8685 ± 0.0104602 (W 0.00858471) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.87334 ± 0.0111467 (W 0.00794833) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 11.7766 ± 0.0297417 (W 0.0142202) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 11.7441 ± 0.0162793 (W 0.0153638) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005129 ± 5.79333e-05 (W 6.00638e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0124985 ± 0.000796388 (W 6.21104e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.03203 ± 0.00136258 (W 0.000457277) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-mid, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.0311985 ± 0.000200198 (W 0.000149403) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.037084 ± 0.000123132 (W 0.000213408) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.108566 ± 0.00282755 (W 0.0012524) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n256/b1 [api-inclusive]` | 0.130824 ± 0.002931 (W 0.00130006) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n512/b1` | 0.349288 ± 0.00190672 (W 0.00302738) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 0.444986 ± 0.00179497 (W 0.00183787) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.742812 ± 0.00243828 (W 0.00298114) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.743866 ± 0.00240234 (W 0.00228458) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 2.743 ± 0.00583695 (W 0.00347329) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 2.74862 ± 0.00488795 (W 0.0057527) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 10.5623 ± 0.039462 (W 0.0092963) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 10.5096 ± 0.0361069 (W 0.0156653) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.231409 ± 0.00170595 (W 0.00220551) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.229163 ± 0.00262403 (W 0.00146086) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.06451 ± 0.00559639 (W 0.0072866) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 1.15855 ± 0.00666913 (W 0.0040146) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 6.75409 ± 0.0138197 (W 0.0170245) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 7.23566 ± 0.0241955 (W 0.0176314) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.0367325 ± 0.00039765 (W 0.000589013) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.038387 ± 0.000430052 (W 0.000698561) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.170763 ± 0.00203259 (W 0.000805792) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.170353 ± 0.00234432 (W 0.00129193) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 1.07555 ± 0.00741855 (W 0.00566641) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 1.08131 ± 0.00398364 (W 0.00495471) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.0601415 ± 0.000534772 (W 0.000434379) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.0668175 ± 0.000519782 (W 0.000667552) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.210827 ± 0.0013875 (W 0.00196234) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.349475 ± 0.00205355 (W 0.00273218) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 0.828573 ± 0.000890841 (W 0.00411777) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 1.44182 ± 0.0115209 (W 0.0100001) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.247782 ± 0.0014495 (W 0.00161773) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.29172 ± 0.0021247 (W 0.00285951) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 0.615071 ± 0.00256342 (W 0.00337206) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 0.863443 ± 0.00300496 (W 0.00670343) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 2.12196 ± 0.00482865 (W 0.0034713) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 3.13018 ± 0.0112325 (W 0.011449) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.173996 ± 0.000728831 (W 0.00135631) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.463163 ± 0.00175375 (W 0.0021102) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 1.32205 ± 0.00874167 (W 0.00228456) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.063596 ± 0.000532535 (W 0.000808414) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.070386 ± 0.000452012 (W 0.000689441) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.212508 ± 0.00123115 (W 0.00162461) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.35658 ± 0.00244001 (W 0.00240625) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 0.772408 ± 0.00148942 (W 0.00210751) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 1.42897 ± 0.00497471 (W 0.00441141) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.4969 ± 0.0048108 (W 0.00203874) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.50024 ± 0.00692271 (W 0.0026631) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 5.32564 ± 0.004483 (W 0.00673969) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 5.36206 ± 0.0071436 (W 0.016437) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 20.7778 ± 0.0237196 (W 0.0260622) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 20.8198 ± 0.0594838 (W 0.0332618) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.704186 ± 0.00419331 (W 0.00188251) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.704025 ± 0.00479659 (W 0.00179888) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 2.83196 ± 0.0137423 (W 0.00602241) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 2.8426 ± 0.0136948 (W 0.00573426) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 10.7573 ± 0.0222691 (W 0.010841) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 10.7641 ± 0.0125777 (W 0.0147308) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.005125 ± 0.000348145 (W 4.69269e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0126485 ± 0.000770222 (W 8.62071e-05) | - | - | `*` one row has no measurement here; `!` between-start range over 5%; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.031358 ± 0.000205648 (W 0.000267341) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.213876 ± 0.000630186 (W 0.00162072) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.209026 ± 0.000743613 (W 0.00199308) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.905166 ± 0.00406229 (W 0.00326286) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.876619 ± 0.00232332 (W 0.00177552) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.85275 ± 0.014851 (W 0.00612582) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.66439 ± 0.0189038 (W 0.00941326) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.29144 ± 0.0134392 (W 0.00305292) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.29384 ± 0.00621871 (W 0.00343484) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.8196 ± 0.0314346 (W 0.0227936) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.8448 ± 0.0271228 (W 0.0203885) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.1839 ± 0.229248 (W 0.0366993) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.2551 ± 0.242834 (W 0.0514446) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.449546 ± 0.00242714 (W 0.00197862) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.425535 ± 0.00325803 (W 0.00327195) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.52618 ± 0.00869184 (W 0.00892922) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.6737 ± 0.00751356 (W 0.00826802) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.0805 ± 0.0857509 (W 0.0599679) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 26.0473 ± 0.130118 (W 0.0533161) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.563337 ± 0.00291317 (W 0.00319179) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.562948 ± 0.00212739 (W 0.00244788) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.87894 ± 0.0118575 (W 0.0069661) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.87129 ± 0.0183409 (W 0.00774901) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 28.904 ± 0.102783 (W 0.0451493) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 28.844 ± 0.103868 (W 0.0349725) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.177732 ± 0.0011157 (W 0.00123849) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.244884 ± 0.00146805 (W 0.00227464) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.03358 ± 0.00478513 (W 0.00347029) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.44707 ± 0.00680693 (W 0.0046103) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.03413 ± 0.022416 (W 0.0112471) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 9.11258 ± 0.0319205 (W 0.0243396) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.871741 ± 0.00556333 (W 0.00207598) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.985457 ± 0.0046875 (W 0.00280033) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.51299 ± 0.0194014 (W 0.00569234) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.79736 ± 0.0149049 (W 0.00662626) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 36.4149 ± 0.115732 (W 0.0341083) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 37.7955 ± 0.129102 (W 0.0692185) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.425868 ± 0.00147244 (W 0.00230874) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.63572 ± 0.0104727 (W 0.00534927) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 15.9073 ± 0.034021 (W 0.0251643) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.171291 ± 0.00121219 (W 0.00152742) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.207402 ± 0.00111426 (W 0.00184737) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.99744 ± 0.00558457 (W 0.0010886) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.18861 ± 0.00565089 (W 0.00803184) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 6.83825 ± 0.0244739 (W 0.00976199) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.66922 ± 0.0291577 (W 0.0196604) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.11698 ± 0.0118937 (W 0.00709871) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.12512 ± 0.0139179 (W 0.00421469) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 17.0567 ± 0.066754 (W 0.0135912) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 17.0816 ± 0.0635489 (W 0.0233504) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 106.699 ± 0.402171 (W 0.252629) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 107.199 ± 0.344784 (W 0.224373) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.0721 ± 0.00737598 (W 0.00253169) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.07954 ± 0.00749333 (W 0.00244471) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.68599 ± 0.0287615 (W 0.0117403) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.69618 ± 0.0323415 (W 0.00813018) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 45.6947 ± 0.201636 (W 0.0618774) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 45.7986 ± 0.186571 (W 0.0496304) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0062175 ± 4.05976e-05 (W 5.047e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.016923 ± 7.6649e-05 (W 7.08266e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0539555 ± 0.000494402 (W 0.000322758) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.193526 ± 0.000953183 (W 0.00117336) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.194622 ± 0.000478566 (W 0.00126927) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.604268 ± 0.00245861 (W 0.00218318) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.612901 ± 0.00210783 (W 0.00188791) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 2.10593 ± 0.00596027 (W 0.00407251) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 2.13006 ± 0.00527241 (W 0.00735639) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32185 ± 0.00595949 (W 0.0038398) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.329 ± 0.0051092 (W 0.0037611) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.29715 ± 0.043139 (W 0.00747086) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.30432 ± 0.0402645 (W 0.00763654) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.6827 ± 0.215948 (W 0.0167837) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.7276 ± 0.205681 (W 0.0290006) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.36678 ± 0.00378383 (W 0.00344895) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.380966 ± 0.00236643 (W 0.00235089) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.14122 ± 0.00540324 (W 0.00749857) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.43048 ± 0.01014 (W 0.00675881) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.4546 ± 0.0651009 (W 0.0306932) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 14.1336 ± 0.0646296 (W 0.0363853) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.159813 ± 0.000755801 (W 0.00137589) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.160449 ± 0.00198305 (W 0.0010393) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00611 ± 0.0220355 (W 0.00253232) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00407 ± 0.0202145 (W 0.00238014) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.38663 ± 0.266425 (W 0.00982668) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.37777 ± 0.270415 (W 0.0186519) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.168023 ± 0.00147552 (W 0.00208729) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.214913 ± 0.00398292 (W 0.00184825) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n256/b1` | 0.629506 ± 0.00318726 (W 0.00296754) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.9441 ± 0.00527926 (W 0.00386843) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 3.05475 ± 0.0138194 (W 0.0125504) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.75675 ± 0.011706 (W 0.0186709) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.480835 ± 0.00278026 (W 0.00214094) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.569074 ± 0.0135328 (W 0.00332301) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.79041 ± 0.0402511 (W 0.00458397) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1 [api-inclusive]` | 2.30243 ± 0.144005 (W 0.00691904) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 10.7092 ± 0.362346 (W 0.0112573) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1 [api-inclusive]` | 13.1151 ± 0.646703 (W 0.0251039) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.307593 ± 0.00231539 (W 0.00205287) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 1.01098 ± 0.00157693 (W 0.00157341) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.33603 ± 0.0151964 (W 0.014866) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.136916 ± 0.000895577 (W 0.000854772) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.176527 ± 0.00143408 (W 0.00247081) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.610502 ± 0.00249301 (W 0.00192975) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.950446 ± 0.00451277 (W 0.0055268) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.87494 ± 0.0141423 (W 0.00743171) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.66711 ± 0.0714094 (W 0.0207433) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.95905 ± 0.0199708 (W 0.00576213) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.96743 ± 0.0208616 (W 0.00419475) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.46472 ± 0.089557 (W 0.0170118) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.50694 ± 0.0853379 (W 0.0140395) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 41.4572 ± 0.73445 (W 0.0488366) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 41.6071 ± 1.02929 (W 0.0469558) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 0.967481 ± 0.00288127 (W 0.00357959) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.964357 ± 0.00329116 (W 0.00241174) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.34051 ± 0.0371212 (W 0.00781474) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.33247 ± 0.0449516 (W 0.0085853) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 20.8258 ± 0.526026 (W 0.0251207) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 20.8914 ± 0.604126 (W 0.0248302) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n128/b1` | 0.0073135 ± 4.16733e-05 (W 5.14912e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.017232 ± 0.00010198 (W 0.000105593) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.041106 ± 0.000269008 (W 0.000246711) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.193761 ± 0.000710189 (W 0.00142852) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.194458 ± 0.000668827 (W 0.00135588) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.580959 ± 0.00125882 (W 0.00360999) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.587341 ± 0.00134056 (W 0.00189605) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.72801 ± 0.0038861 (W 0.00380671) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.79703 ± 0.0070943 (W 0.0031768) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.989729 ± 0.0748623 (W 0.00324336) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n128/b1 [api-inclusive]` | 0.988903 ± 0.0040039 (W 0.00285152) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 4.02661 ± 0.0152453 (W 0.00504298) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 4.02963 ± 0.00678381 (W 0.00451553) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.9229 ± 0.0336319 (W 0.0210416) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.9979 ± 0.064327 (W 0.0315572) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.369913 ± 0.00290864 (W 0.00280666) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.388675 ± 0.00495966 (W 0.00335975) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.97716 ± 0.0109446 (W 0.0088112) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.31659 ± 0.0105991 (W 0.00717246) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.5706 ± 0.0553195 (W 0.0187054) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 12.4582 ± 0.797528 (W 0.0273469) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.103246 ± 0.000466505 (W 0.00125833) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.10321 ± 0.00030787 (W 0.00110827) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.527073 ± 0.00077646 (W 0.00163446) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.531547 ± 0.00261458 (W 0.00219925) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.85108 ± 0.00764317 (W 0.0074895) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.84837 ± 0.00726872 (W 0.00749873) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.17692 ± 0.00134065 (W 0.00197401) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.210407 ± 0.000927222 (W 0.00134588) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.5281 ± 0.00222768 (W 0.00219351) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.885997 ± 0.0026321 (W 0.00405839) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.20289 ± 0.00802304 (W 0.00586065) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.06963 ± 0.0218703 (W 0.01479) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.40732 ± 0.00158617 (W 0.00217288) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.505744 ± 0.00203732 (W 0.00360512) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.29892 ± 0.00331531 (W 0.00329592) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.87 ± 0.00612438 (W 0.00588915) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.77874 ± 0.0195986 (W 0.00890271) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 8.43541 ± 0.0347721 (W 0.0252099) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.274034 ± 0.000448713 (W 0.00170447) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.880709 ± 0.0214342 (W 0.00209764) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n512/b1` | 2.96994 ± 0.0115969 (W 0.0066594) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.136452 ± 0.00058668 (W 0.000813042) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.173694 ± 0.00147858 (W 0.00102155) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.514896 ± 0.00261867 (W 0.00266952) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.887418 ± 0.00349199 (W 0.0035942) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.03125 ± 0.00660892 (W 0.00522402) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.97842 ± 0.0198699 (W 0.0137918) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.89693 ± 0.0111092 (W 0.00314451) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.90741 ± 0.0119833 (W 0.00525254) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.52433 ± 0.0149271 (W 0.00765044) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.5677 ± 0.0164949 (W 0.00950871) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 32.2221 ± 0.104523 (W 0.0679252) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 32.2456 ± 0.0633357 (W 0.0334094) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.992558 ± 0.00720031 (W 0.00236483) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.99235 ± 0.00353618 (W 0.00357549) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.18547 ± 0.020688 (W 0.00446233) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.19232 ± 0.021266 (W 0.0054992) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 17.5612 ± 0.0635564 (W 0.0178509) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 17.589 ± 0.0722599 (W 0.0200177) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.007334 ± 6.17784e-05 (W 3.51149e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.016927 ± 8.23223e-05 (W 0.000105958) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0390895 ± 0.000205284 (W 0.000218289) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.196744 ± 0.000874566 (W 0.00149172) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.20279 ± 0.000957941 (W 0.00141007) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.806272 ± 0.00317982 (W 0.00185371) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.831982 ± 0.00221446 (W 0.00223014) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 4.40043 ± 0.0179596 (W 0.00640034) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 4.5058 ± 0.0135273 (W 0.00769766) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 2.29282 ± 0.00912526 (W 0.00624678) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 2.29408 ± 0.00809317 (W 0.00382128) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 10.7899 ± 0.04063 (W 0.0150407) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 10.8233 ± 0.0375036 (W 0.0226128) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 59.2182 ± 0.143696 (W 0.255177) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 59.3409 ± 0.202856 (W 0.071967) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.396716 ± 0.00196014 (W 0.00205764) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.396635 ± 0.00345769 (W 0.0018797) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 3.36294 ± 0.0747605 (W 0.00910363) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n256/b1 [api-inclusive]` | 3.54849 ± 0.0117038 (W 0.0113139) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 25.0165 ± 0.0779778 (W 0.0205694) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 25.4162 ± 0.181063 (W 0.0496272) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.559945 ± 0.00359895 (W 0.00207179) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.556217 ± 0.00286066 (W 0.00246719) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 3.83248 ± 0.0133642 (W 0.0094106) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 3.82592 ± 0.017542 (W 0.00576292) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 28.8591 ± 0.290262 (W 0.0933211) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 28.6559 ± 0.121039 (W 0.0406998) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.171127 ± 0.00113752 (W 0.0014673) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.220698 ± 0.00165813 (W 0.00209632) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 1.04699 ± 0.00316291 (W 0.00352063) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 1.3314 ± 0.00713865 (W 0.00413822) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 7.11257 ± 0.0234251 (W 0.00838276) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 8.5465 ± 0.0708743 (W 0.0167788) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.882405 ± 0.00608394 (W 0.00174351) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.959566 ± 0.00315077 (W 0.00303651) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 5.23291 ± 0.0211005 (W 0.00781817) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 5.69158 ± 0.0210517 (W 0.0125077) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 35.2435 ± 0.094146 (W 0.0600293) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 37.137 ± 0.158373 (W 0.0442384) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.437171 ± 0.0038783 (W 0.00190413) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 2.38076 ± 0.00682906 (W 0.00636775) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 14.8751 ± 0.0535011 (W 0.0142989) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.168855 ± 0.00102216 (W 0.00136861) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.180431 ± 0.0010812 (W 0.00117872) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 1.02771 ± 0.00384036 (W 0.00220981) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 1.07177 ± 0.00461781 (W 0.00453506) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 6.9685 ± 0.0211487 (W 0.00994792) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 7.18281 ± 0.0205665 (W 0.00998387) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 3.11876 ± 0.0149741 (W 0.00290442) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 3.11777 ± 0.0484527 (W 0.00256047) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 16.9977 ± 0.059491 (W 0.0132438) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 16.9924 ± 0.0517673 (W 0.0170728) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 105.693 ± 0.461868 (W 0.178271) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 105.847 ± 0.394074 (W 0.0937662) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 1.05707 ± 0.00526097 (W 0.00349701) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 1.05728 ± 0.00559214 (W 0.00214305) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 6.5304 ± 0.0230805 (W 0.0130898) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 6.53659 ± 0.0247414 (W 0.0103555) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 44.9069 ± 0.103589 (W 0.0423762) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 44.9757 ± 0.162882 (W 0.0480167) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0067175 ± 3.29282e-05 (W 4.65621e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0171315 ± 7.54445e-05 (W 8.01742e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.05402 ± 0.00048495 (W 0.000700892) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.180592 ± 0.000171491 (W 0.00108697) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188675 ± 0.000438028 (W 0.00115659) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.53412 ± 0.00270615 (W 0.00205918) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.57034 ± 0.00208154 (W 0.00204431) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.79672 ± 0.00535953 (W 0.004547) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.9553 ± 0.0052465 (W 0.0039598) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 1.32061 ± 0.00434429 (W 0.00290442) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 1.3238 ± 0.0054982 (W 0.00353015) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 5.27153 ± 0.0436772 (W 0.00736235) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 5.26668 ± 0.0338218 (W 0.00616625) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 24.5112 ± 0.178463 (W 0.0196486) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 24.5548 ± 0.177996 (W 0.0165151) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.348966 ± 0.00357623 (W 0.0037974) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.352992 ± 0.00237967 (W 0.00308062) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 2.09914 ± 0.00896094 (W 0.00728767) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.30194 ± 0.00714847 (W 0.00814136) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 12.5551 ± 0.0390305 (W 0.0152227) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 13.5289 ± 0.0767091 (W 0.0206305) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.159121 ± 0.00132253 (W 0.000933172) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.158791 ± 0.000773051 (W 0.00130848) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 1.00862 ± 0.0244506 (W 0.00156042) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 1.00681 ± 0.023434 (W 0.0033997) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 7.38495 ± 0.271039 (W 0.0138268) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 7.37735 ± 0.261379 (W 0.00942722) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n128/b1` | 0.133078 ± 0.000959518 (W 0.00153599) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.188951 ± 0.00232985 (W 0.00170967) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.546519 ± 0.00343373 (W 0.00324705) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.841324 ± 0.00548994 (W 0.00245806) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.75188 ± 0.00926825 (W 0.00386083) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 4.16745 ± 0.0108168 (W 0.0116368) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.442838 ± 0.00229684 (W 0.00216073) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.542068 ± 0.0113796 (W 0.00329485) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n256/b1` | 1.69867 ± 0.0201234 (W 0.00302406) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 2.19329 ± 0.101459 (W 0.00669109) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n512/b1` | 10.288 ± 0.184753 (W 0.0101685) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 12.4531 ± 0.518122 (W 0.0184959) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n128/b1` | 0.283497 ± 0.00110636 (W 0.00149648) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.917138 ± 0.00362794 (W 0.00237211) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 5.02061 ± 0.011506 (W 0.00468164) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.134551 ± 0.00157499 (W 0.000986122) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.150455 ± 0.000669644 (W 0.000890379) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.525698 ± 0.0017105 (W 0.00222515) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.837618 ± 0.00343022 (W 0.00400423) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 2.56819 ± 0.00876965 (W 0.00512516) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 4.06359 ± 0.0229392 (W 0.0152648) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.96061 ± 0.0186806 (W 0.00548219) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.96205 ± 0.0185869 (W 0.00428795) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 8.38181 ± 0.0689388 (W 0.0121216) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 8.39458 ± 0.0990062 (W 0.011626) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 40.5945 ± 0.799055 (W 0.0583037) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 40.6806 ± 1.00359 (W 0.0445639) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n128/b1` | 0.942174 ± 0.00470069 (W 0.00287448) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.94453 ± 0.00431799 (W 0.00433693) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.21206 ± 0.0207822 (W 0.0122464) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.20938 ± 0.0159148 (W 0.00747056) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 19.9296 ± 0.530997 (W 0.0209926) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n512/b1 [api-inclusive]` | 19.9715 ± 0.453843 (W 0.0227145) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n128/b1` | 0.007348 ± 4.1497e-05 (W 5.2888e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.0184745 ± 0.000351587 (W 0.000323304) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0453995 ± 0.000628673 (W 0.000667372) | - | - | `*` one row has no measurement here |

## linalg-mid, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `cholesky/n128/b1` | 0.180738 ± 0.000591836 (W 0.000985938) | - | - | `*` one row has no measurement here |
| `cholesky/n128/b1 [api-inclusive]` | 0.188715 ± 0.000511863 (W 0.00133331) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1` | 0.508889 ± 0.0020722 (W 0.00189244) | - | - | `*` one row has no measurement here |
| `cholesky/n256/b1 [api-inclusive]` | 0.544695 ± 0.00095784 (W 0.00208951) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1` | 1.47188 ± 0.00211408 (W 0.00272687) | - | - | `*` one row has no measurement here |
| `cholesky/n512/b1 [api-inclusive]` | 1.62335 ± 0.00440998 (W 0.00229875) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1` | 0.983928 ± 0.0014573 (W 0.00117234) | - | - | `*` one row has no measurement here |
| `eigh/n128/b1 [api-inclusive]` | 0.987739 ± 0.00301139 (W 0.00209201) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1` | 3.97642 ± 0.00943106 (W 0.00842959) | - | - | `*` one row has no measurement here |
| `eigh/n256/b1 [api-inclusive]` | 3.99184 ± 0.0131808 (W 0.00463361) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1` | 16.7339 ± 0.0584134 (W 0.0234621) | - | - | `*` one row has no measurement here |
| `eigh/n512/b1 [api-inclusive]` | 16.748 ± 0.0615446 (W 0.0167938) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1` | 0.353326 ± 0.00444078 (W 0.00327093) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n128/b1 [api-inclusive]` | 0.361291 ± 0.00477196 (W 0.00269561) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1` | 1.95046 ± 0.0118779 (W 0.00551275) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n256/b1 [api-inclusive]` | 2.19886 ± 0.00567856 (W 0.00912199) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1` | 10.6353 ± 0.0417829 (W 0.0218366) | - | - | `*` one row has no measurement here |
| `full_piv_lu/n512/b1 [api-inclusive]` | 11.8687 ± 0.0379115 (W 0.0181551) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [block(basis=128,target=128)]` | 0.101508 ± 0.000409876 (W 0.00138457) | - | - | `*` one row has no measurement here |
| `householder_apply/n128/b1 [sequence(basis=128,target=128)]` | 0.101722 ± 0.000779217 (W 0.000932236) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [block(basis=256,target=256)]` | 0.536416 ± 0.0020717 (W 0.00157587) | - | - | `*` one row has no measurement here |
| `householder_apply/n256/b1 [sequence(basis=256,target=256)]` | 0.539116 ± 0.0017492 (W 0.00175225) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [block(basis=512,target=512)]` | 3.85677 ± 0.00761025 (W 0.00551727) | - | - | `*` one row has no measurement here |
| `householder_apply/n512/b1 [sequence(basis=512,target=512)]` | 3.85765 ± 0.00913101 (W 0.00690887) | - | - | `*` one row has no measurement here |
| `lu/n128/b1` | 0.127999 ± 0.00188426 (W 0.0010277) | - | - | `*` one row has no measurement here |
| `lu/n128/b1 [api-inclusive]` | 0.184291 ± 0.000797933 (W 0.00134721) | - | - | `*` one row has no measurement here |
| `lu/n256/b1` | 0.480414 ± 0.00531011 (W 0.00296223) | - | - | `*` one row has no measurement here |
| `lu/n256/b1 [api-inclusive]` | 0.773062 ± 0.00229426 (W 0.00242361) | - | - | `*` one row has no measurement here |
| `lu/n512/b1` | 2.03202 ± 0.00735391 (W 0.00377393) | - | - | `*` one row has no measurement here |
| `lu/n512/b1 [api-inclusive]` | 3.48285 ± 0.0128851 (W 0.00710714) | - | - | `*` one row has no measurement here |
| `qr/n128/b1` | 0.370666 ± 0.0017129 (W 0.00170468) | - | - | `*` one row has no measurement here |
| `qr/n128/b1 [api-inclusive]` | 0.481117 ± 0.00354719 (W 0.00211188) | - | - | `*` one row has no measurement here |
| `qr/n256/b1` | 1.24592 ± 0.00439348 (W 0.00218551) | - | - | `*` one row has no measurement here |
| `qr/n256/b1 [api-inclusive]` | 1.75736 ± 0.00741615 (W 0.00769019) | - | - | `*` one row has no measurement here |
| `qr/n512/b1` | 5.54857 ± 0.0203595 (W 0.00938087) | - | - | `*` one row has no measurement here |
| `qr/n512/b1 [api-inclusive]` | 7.7606 ± 0.0444464 (W 0.0128345) | - | - | `*` one row has no measurement here |
| `qr_factor/n128/b1` | 0.256898 ± 0.00114157 (W 0.00142738) | - | - | `*` one row has no measurement here |
| `qr_factor/n256/b1` | 0.822034 ± 0.00277334 (W 0.00270987) | - | - | `*` one row has no measurement here |
| `qr_factor/n512/b1` | 2.7724 ± 0.00829875 (W 0.0031308) | - | - | `*` one row has no measurement here |
| `solve/n128/b1` | 0.131019 ± 0.00105295 (W 0.000564193) | - | - | `*` one row has no measurement here |
| `solve/n128/b1 [api-inclusive]` | 0.147268 ± 0.000586458 (W 0.000991704) | - | - | `*` one row has no measurement here |
| `solve/n256/b1` | 0.464039 ± 0.00142282 (W 0.00227756) | - | - | `*` one row has no measurement here |
| `solve/n256/b1 [api-inclusive]` | 0.778906 ± 0.0027006 (W 0.00387769) | - | - | `*` one row has no measurement here |
| `solve/n512/b1` | 1.86982 ± 0.00719206 (W 0.00303078) | - | - | `*` one row has no measurement here |
| `solve/n512/b1 [api-inclusive]` | 3.38588 ± 0.0271848 (W 0.00565596) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1` | 1.88651 ± 0.00540508 (W 0.00558672) | - | - | `*` one row has no measurement here |
| `svd_thin/n128/b1 [api-inclusive]` | 1.89006 ± 0.00734401 (W 0.00262989) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1` | 7.44345 ± 0.0252927 (W 0.0193008) | - | - | `*` one row has no measurement here |
| `svd_thin/n256/b1 [api-inclusive]` | 7.47019 ± 0.0313642 (W 0.0110286) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1` | 31.4194 ± 0.100814 (W 0.0524556) | - | - | `*` one row has no measurement here |
| `svd_thin/n512/b1 [api-inclusive]` | 31.43 ± 0.0911226 (W 0.0505092) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1` | 0.977238 ± 0.00326758 (W 0.00275891) | - | - | `*` one row has no measurement here |
| `svd_values/n128/b1 [api-inclusive]` | 0.974094 ± 0.00397553 (W 0.00322789) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1` | 4.05956 ± 0.00887814 (W 0.00457865) | - | - | `*` one row has no measurement here |
| `svd_values/n256/b1 [api-inclusive]` | 4.0683 ± 0.00572733 (W 0.00497068) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1` | 16.6722 ± 0.0538265 (W 0.0193066) | - | - | `*` one row has no measurement here |
| `svd_values/n512/b1 [api-inclusive]` | 16.6854 ± 0.0394064 (W 0.024684) | - | - | `*` one row has no measurement here |
| `triangular_solve/n128/b1` | 0.0073735 ± 4.75286e-05 (W 7.12901e-05) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n256/b1` | 0.018559 ± 0.000226647 (W 0.000121326) | - | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n512/b1` | 0.0462545 ± 0.000300124 (W 0.000346514) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/128`, `cholesky/256`, `cholesky/512`, `eig/128`, `eig/256`, `eig/512`, `eigh/128`, `eigh/256`, `eigh/512`, `full_piv_lu/128`, `full_piv_lu/256`, `full_piv_lu/512`, `householder_apply/128`, `householder_apply/256`, `householder_apply/512`, `lu/128`, `lu/256`, `lu/512`, `qr/128`, `qr/256`, `qr/512`, `qr_factor/128`, `qr_factor/256`, `qr_factor/512`, `solve/128`, `solve/256`, `solve/512`, `svd_thin/128`, `svd_thin/256`, `svd_thin/512`, `svd_values/128`, `svd_values/256`, `svd_values/512`, `triangular_solve/128`, `triangular_solve/256`, `triangular_solve/512`, `upgrade_factor/128`, `upgrade_factor/256`, `upgrade_factor/512`.

## linalg-tall, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.968369 ± 0.00255264 (W 0.00346457) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.841962 ± 0.00221339 (W 0.00497594) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 53.2115 ± 0.52411 (W 0.27144) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 49.8005 ± 0.406569 (W 0.17726) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.555304 ± 0.00110912 (W 0.00230933) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.553905 ± 0.00201008 (W 0.00208492) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.0291 ± 0.065025 (W 0.150652) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 34.9781 ± 0.0548883 (W 0.164708) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.326706 ± 0.00171788 (W 0.00237311) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.345797 ± 0.00171258 (W 0.00199028) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 12.4585 ± 0.0371196 (W 0.031178) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 13.8104 ± 0.0307508 (W 0.0764646) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.45071 ± 0.00447032 (W 0.00410439) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.70656 ± 0.00553925 (W 0.00399859) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 67.6947 ± 0.302983 (W 0.363952) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 71.9609 ± 0.226908 (W 0.230373) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.933154 ± 0.00196678 (W 0.00197241) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 31.6652 ± 0.303492 (W 0.166287) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.02164 ± 0.00646318 (W 0.003373) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.0258 ± 0.00824627 (W 0.0064053) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 83.0036 ± 0.0970033 (W 0.0640964) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 83.1892 ± 0.178935 (W 0.118293) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.0076 ± 0.00295263 (W 0.00466972) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01047 ± 0.00421835 (W 0.00386899) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 37.4265 ± 0.496176 (W 0.0416831) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 37.3592 ± 1.14327 (W 0.0545431) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.572107 ± 0.00617055 (W 0.00525732) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.608382 ± 0.00404888 (W 0.00346752) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 16.114 ± 0.078588 (W 0.0425428) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 17.026 ± 0.056043 (W 0.0356744) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.167602 ± 0.00294826 (W 0.00270574) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.167475 ± 0.00329778 (W 0.0020071) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.47369 ± 0.0262377 (W 0.0266252) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.4735 ± 0.0331154 (W 0.0400671) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.282405 ± 0.0011518 (W 0.00190781) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.315083 ± 0.00188157 (W 0.00166349) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.1905 ± 0.0649099 (W 0.0311464) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.34059 ± 0.0760228 (W 0.0284365) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.741905 ± 0.00365956 (W 0.00619202) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.12574 ± 0.00418895 (W 0.00466789) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 23.7193 ± 0.14485 (W 0.134101) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 31.182 ± 0.156197 (W 0.237558) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.586574 ± 0.00264083 (W 0.00255206) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.0773 ± 0.0493457 (W 0.0435823) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.33602 ± 0.00528906 (W 0.00320308) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.34906 ± 0.0082437 (W 0.0035068) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 36.7935 ± 0.0631597 (W 0.0928506) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 37.0925 ± 0.0576689 (W 0.0763472) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.860954 ± 0.00757031 (W 0.00435609) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.868847 ± 0.00514197 (W 0.00418911) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 22.0324 ± 0.0497846 (W 0.0488752) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 22.0607 ± 0.944106 (W 0.0358657) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.538912 ± 0.00533134 (W 0.00364365) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.593487 ± 0.0039254 (W 0.00565086) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 10.0876 ± 0.0506366 (W 0.0324929) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 12.2518 ± 0.0388565 (W 0.0687261) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.10512 ± 0.00262846 (W 0.00253408) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.10534 ± 0.00124308 (W 0.0016409) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.69099 ± 0.021059 (W 0.043043) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.64367 ± 0.0370318 (W 0.0441874) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.249872 ± 0.00227123 (W 0.00222453) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.308651 ± 0.0011691 (W 0.00240633) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.26238 ± 0.0096246 (W 0.021366) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 6.32517 ± 0.0355382 (W 0.0460016) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.661039 ± 0.00384555 (W 0.00376456) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.04262 ± 0.0026978 (W 0.00822855) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 16.1648 ± 0.0772986 (W 0.0468418) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 25.2789 ± 0.0975555 (W 0.0682193) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.543006 ± 0.00157915 (W 0.0024992) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 9.77523 ± 0.0354691 (W 0.0385708) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.25223 ± 0.005615 (W 0.00306627) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.27807 ± 0.0225221 (W 0.00344542) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 30.7491 ± 0.0417918 (W 0.0510848) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 30.9055 ± 0.132908 (W 0.0561482) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.843106 ± 0.00907985 (W 0.00621739) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.844356 ± 0.00436503 (W 0.00528913) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.5303 ± 0.0617937 (W 0.0445344) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.4963 ± 0.0316761 (W 0.0352657) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.81525 ± 0.000733271 (W 0.00248292) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.739173 ± 0.00280562 (W 0.00724582) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 52.1331 ± 0.456338 (W 0.394263) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 47.0543 ± 0.403171 (W 0.330439) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.594841 ± 0.00349394 (W 0.00340353) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.595311 ± 0.00186725 (W 0.00337191) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 35.7643 ± 0.0674786 (W 0.0340136) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 35.7826 ± 0.0733266 (W 0.0310394) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.22385 ± 0.00110047 (W 0.00204106) | 0.27446 ± 0.00159659 (W 0.00182421) | 🔵 faer-fork 1.23× faster than `lapack-aocl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.240468 ± 0.00130267 (W 0.00174424) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.7989 ± 0.0193294 (W 0.0170491) | 11.5428 ± 0.0257863 (W 0.0231896) | 🔵 faer-fork 1.07× faster than `lapack-aocl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.2569 ± 0.0259714 (W 0.0181454) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.24836 ± 0.00613374 (W 0.00318658) | 2.84855 ± 0.00938724 (W 0.00353656) | 🔵 faer-fork 2.28× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 1.56489 ± 0.00525202 (W 0.00469688) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 60.9126 ± 0.121281 (W 0.0625626) | 76.0669 ± 0.360877 (W 0.114117) | 🔵 faer-fork 1.25× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 68.7635 ± 0.133889 (W 0.376752) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.664695 ± 0.00270331 (W 0.00344024) | 0.941642 ± 0.0029627 (W 0.00250536) | 🔵 faer-fork 1.42× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 24.8019 ± 0.0766817 (W 0.0300413) | 27.8743 ± 0.0868649 (W 0.0365381) | 🔵 faer-fork 1.12× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.97831 ± 0.00338115 (W 0.00240052) | 2.72474 ± 0.0115602 (W 0.00633972) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.98542 ± 0.00386886 (W 0.00241838) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 77.5637 ± 0.175975 (W 0.147086) | 100.507 ± 0.515376 (W 0.185263) | 🔵 faer-fork 1.30× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 77.7966 ± 0.235175 (W 0.405783) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.928335 ± 0.00205923 (W 0.00218498) | 1.27006 ± 0.00293899 (W 0.00382008) | 🔵 faer-fork 1.37× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.927984 ± 0.0032933 (W 0.00264695) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.4124 ± 0.109785 (W 0.0314067) | 37.7381 ± 0.107332 (W 0.0508285) | 🔵 faer-fork 1.20× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.4367 ± 0.150625 (W 0.0393628) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.493173 ± 0.00310168 (W 0.00263819) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.511705 ± 0.00161471 (W 0.00335116) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 15.3083 ± 0.0440987 (W 0.025869) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 14.4021 ± 0.0647133 (W 0.0374754) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.203821 ± 0.00215529 (W 0.000844335) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.200819 ± 0.00197336 (W 0.000939227) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 11.1053 ± 0.0257526 (W 0.0136172) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 11.1036 ± 0.0126199 (W 0.0176051) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.18464 ± 0.00160598 (W 0.00190073) | 0.234283 ± 0.00104419 (W 0.00232824) | 🔵 faer-fork 1.27× faster than `lapack-aocl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.212415 ± 0.000938077 (W 0.00180047) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 4.27177 ± 0.00724972 (W 0.00835267) | 9.61453 ± 0.0350661 (W 0.00928008) | 🔵 faer-fork 2.25× faster than `lapack-aocl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 4.73276 ± 0.00619075 (W 0.0142384) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.694071 ± 0.00331008 (W 0.00273086) | 2.81345 ± 0.00841344 (W 0.00673089) | 🔵 faer-fork 4.05× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.978309 ± 0.00482191 (W 0.00249362) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 21.6144 ± 0.0773249 (W 0.0702978) | 24.4162 ± 0.0606609 (W 0.0305204) | 🔵 faer-fork 1.13× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 28.3729 ± 0.104442 (W 0.0658345) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.507931 ± 0.00313801 (W 0.0022756) | 0.934107 ± 0.00335582 (W 0.00219956) | 🔵 faer-fork 1.84× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 10.8461 ± 0.0303255 (W 0.0246495) | 9.46363 ± 0.0298417 (W 0.0115239) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.29913 ± 0.01395 (W 0.00245816) | 2.53056 ± 0.00395156 (W 0.00450081) | 🔵 faer-fork 1.95× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.30346 ± 0.00927395 (W 0.00316341) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 31.4653 ± 0.0586564 (W 0.0585781) | 45.8955 ± 0.11411 (W 0.0527232) | 🔵 faer-fork 1.46× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 31.7682 ± 0.0913877 (W 0.0521369) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.786723 ± 0.00346636 (W 0.00205835) | 1.26463 ± 0.00427991 (W 0.00219825) | 🔵 faer-fork 1.61× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.793136 ± 0.0031376 (W 0.00262127) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 16.1316 ± 0.114698 (W 0.0419469) | 19.3949 ± 0.102009 (W 0.0334274) | 🔵 faer-fork 1.20× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 16.153 ± 0.1095 (W 0.0272719) | - | - | `*` one row has no measurement here |

## linalg-tall, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.462942 ± 0.0011744 (W 0.00293391) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.489793 ± 0.00342262 (W 0.00334245) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.9224 ± 0.0603468 (W 0.0422473) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 9.56951 ± 0.0755453 (W 0.0324171) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.128446 ± 0.001213 (W 0.00201204) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.125994 ± 0.00128421 (W 0.00168531) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 8.81579 ± 0.0349749 (W 0.0289237) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 8.82508 ± 0.0295059 (W 0.0235002) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.176517 ± 0.00151042 (W 0.00145535) | 0.27876 ± 0.00153404 (W 0.00234932) | 🔵 faer-fork 1.58× faster than `lapack-aocl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.20666 ± 0.00132863 (W 0.00121915) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.2354 ± 0.00779154 (W 0.00821178) | 9.56466 ± 0.0175836 (W 0.0120544) | 🔵 faer-fork 2.96× faster than `lapack-aocl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 3.75909 ± 0.00820789 (W 0.013387) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.597939 ± 0.00175476 (W 0.00178198) | 2.81774 ± 0.0121178 (W 0.00565177) | 🔵 faer-fork 4.71× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 0.893377 ± 0.0038622 (W 0.00506354) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 15.8533 ± 0.0900344 (W 0.0493198) | 16.9609 ± 0.0681096 (W 0.027551) | 🔵 faer-fork 1.07× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 22.357 ± 0.063233 (W 0.0612873) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.487936 ± 0.00136877 (W 0.00315506) | 0.936141 ± 0.00404366 (W 0.00297638) | 🔵 faer-fork 1.92× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 8.41457 ± 0.0218212 (W 0.0209939) | 7.97065 ± 0.0234252 (W 0.0166817) | 🔴 faer-fork 1.06× slower than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 1.20093 ± 0.00458848 (W 0.00463548) | 2.56126 ± 0.0194474 (W 0.00899552) | 🔵 faer-fork 2.13× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.2245 ± 0.0100369 (W 0.0043415) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 25.1741 ± 0.0548883 (W 0.0557464) | 40.7905 ± 0.155548 (W 0.0946767) | 🔵 faer-fork 1.62× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 25.4277 ± 0.0291651 (W 0.0495385) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.764617 ± 0.00328009 (W 0.00296726) | 1.26959 ± 0.00507464 (W 0.00239378) | 🔵 faer-fork 1.66× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.765082 ± 0.00321765 (W 0.00251659) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 13.6708 ± 0.048616 (W 0.025896) | 20.3924 ± 0.104907 (W 0.0718059) | 🔵 faer-fork 1.49× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 13.6573 ± 0.0276477 (W 0.0231326) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.89813 ± 0.0132262 (W 0.00566903) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.77577 ± 0.00322798 (W 0.00316103) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 153.071 ± 0.43987 (W 0.895971) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 148.799 ± 0.47941 (W 0.554877) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.18498 ± 0.00852489 (W 0.00476809) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.18875 ± 0.0103089 (W 0.00344571) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 137.006 ± 0.397544 (W 0.236417) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 137.025 ± 0.501501 (W 0.28252) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.944802 ± 0.0055597 (W 0.00339844) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.984491 ± 0.00259954 (W 0.00098304) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 44.6084 ± 0.102073 (W 0.0584041) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 46.5613 ± 0.190105 (W 0.151601) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.50245 ± 0.011672 (W 0.00543727) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 5.10804 ± 0.0171389 (W 0.00708854) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 238.514 ± 0.757075 (W 0.787094) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 249.342 ± 0.707943 (W 0.957927) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.30637 ± 0.00482898 (W 0.00250316) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 97.8823 ± 0.1917 (W 0.279295) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 5.40305 ± 0.010095 (W 0.00686176) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.44437 ± 0.0289065 (W 0.00937155) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 271.299 ± 0.831058 (W 0.482305) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 271.689 ± 1.03704 (W 6.02312) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.48406 ± 0.00975524 (W 0.00481574) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.50811 ± 0.0159129 (W 0.00849949) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 114.113 ± 0.432975 (W 0.596612) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 114.138 ± 1.34713 (W 0.377396) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.865555 ± 0.0102658 (W 0.00445193) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.930611 ± 0.00738811 (W 0.00437153) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 49.7189 ± 0.0944792 (W 0.100449) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 51.6319 ± 0.165248 (W 0.223404) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.593482 ± 0.00387402 (W 0.0021323) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.593948 ± 0.00313205 (W 0.00278352) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 37.128 ± 0.200726 (W 0.233845) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 37.1091 ± 0.197805 (W 0.0558975) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.63254 ± 0.00226767 (W 0.00267123) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.701293 ± 0.00312316 (W 0.00241168) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 18.5953 ± 0.0841839 (W 0.0903244) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 21.0796 ± 0.223325 (W 0.0609562) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.79251 ± 0.00882445 (W 0.00476143) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.58315 ± 0.0303239 (W 0.00830358) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 76.8661 ± 0.247775 (W 0.345778) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 91.5674 ± 0.282059 (W 0.214861) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.23576 ± 0.00343103 (W 0.00334971) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 36.3944 ± 0.0788266 (W 0.0687546) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.657 ± 0.0113485 (W 0.00681055) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.67469 ± 0.0175353 (W 0.00781628) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 98.2138 ± 0.132967 (W 0.105944) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 98.9489 ± 0.0466331 (W 0.0936399) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.6479 ± 0.0112426 (W 0.00323782) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.64303 ± 0.00974451 (W 0.00531044) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 52.0141 ± 0.159167 (W 0.0855017) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 52.1698 ± 0.163367 (W 0.0791725) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.719326 ± 0.0122734 (W 0.00577006) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.813794 ± 0.0110318 (W 0.00336316) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 25.654 ± 0.128554 (W 0.120388) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 28.6358 ± 0.0751059 (W 0.136368) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.314985 ± 0.00665682 (W 0.00445949) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.323842 ± 0.00731431 (W 0.00403684) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 24.0458 ± 0.0320124 (W 0.111469) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 24.1063 ± 0.0607995 (W 0.110878) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.550868 ± 0.00179094 (W 0.00302048) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.639788 ± 0.00333373 (W 0.00251626) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 13.5566 ± 0.073587 (W 0.0455377) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 17.2418 ± 0.19621 (W 0.102075) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.42136 ± 0.00313806 (W 0.0038589) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 2.22172 ± 0.0107848 (W 0.0097506) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 47.9777 ± 0.115396 (W 0.12438) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 68.578 ± 0.161548 (W 0.160865) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.10592 ± 0.00384336 (W 0.0030698) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 26.6128 ± 0.0680853 (W 0.0571472) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 2.27193 ± 0.0164363 (W 0.0061947) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.29998 ± 0.0193727 (W 0.00497769) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 73.2938 ± 0.0807873 (W 0.0923056) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 74.0407 ± 0.160575 (W 0.153515) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.53661 ± 0.0169972 (W 0.00538398) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.54638 ± 0.0089159 (W 0.00513989) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 43.0628 ± 0.0939376 (W 0.0789772) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 43.0687 ± 0.082046 (W 0.0473857) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.70575 ± 0.00668122 (W 0.00440957) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.65885 ± 0.00629679 (W 0.00399323) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 149.962 ± 0.492021 (W 0.439) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 142.648 ± 0.367876 (W 0.519388) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 2.25545 ± 0.0114329 (W 0.00342863) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 2.25651 ± 0.0163571 (W 0.003765) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 137.998 ± 0.525092 (W 0.183219) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 138.104 ± 0.459628 (W 0.23602) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.844076 ± 0.00324459 (W 0.00237148) | 0.781732 ± 0.00295431 (W 0.00265314) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.867838 ± 0.00326797 (W 0.00228521) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 40.3597 ± 0.108389 (W 0.0409055) | 39.97 ± 0.116568 (W 0.051786) | · |  |
| `lu/4096x256/b1 [api-inclusive]` | 41.3736 ± 0.139304 (W 0.0445059) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 4.28502 ± 0.0109332 (W 0.00549509) | 4.94029 ± 0.0373665 (W 0.00850692) | 🔵 faer-fork 1.15× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 4.95281 ± 0.0155243 (W 0.00814866) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 228.108 ± 0.869226 (W 0.557687) | 266.311 ± 0.991 (W 1.0472) | 🔵 faer-fork 1.17× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 243.23 ± 0.814304 (W 1.12473) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 2.01343 ± 0.00779902 (W 0.00351249) | 1.74702 ± 0.0146521 (W 0.00475568) | 🔴 faer-fork 1.15× slower than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 87.2612 ± 0.133087 (W 0.135649) | 100.489 ± 1.56838 (W 0.571612) | 🔵 faer-fork 1.15× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 5.30843 ± 0.0245484 (W 0.0063066) | 5.54744 ± 0.0432576 (W 0.0056341) | · |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 5.33627 ± 0.0233662 (W 0.00697827) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 258.877 ± 0.83825 (W 0.470606) | 307.335 ± 2.25341 (W 1.19993) | 🔵 faer-fork 1.19× faster than `lapack-aocl` |  |
| `svd_thin/4096x256/b1 [api-inclusive]` | 259.871 ± 0.850723 (W 6.52005) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 2.36543 ± 0.00780896 (W 0.00357191) | 2.26913 ± 0.0140965 (W 0.00537847) | · |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 2.38066 ± 0.0171628 (W 0.00394474) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 99.9922 ± 0.329352 (W 0.144492) | 118.715 ± 1.38563 (W 0.882312) | 🔵 faer-fork 1.19× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 99.8793 ± 0.238896 (W 0.385178) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.779292 ± 0.00738598 (W 0.00259812) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.819407 ± 0.00604482 (W 0.00336041) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 47.059 ± 0.10828 (W 0.119233) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 46.4595 ± 0.130299 (W 0.278041) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.655506 ± 0.00424297 (W 0.00182264) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.654851 ± 0.00612427 (W 0.00229008) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 39.1757 ± 0.0675139 (W 0.0696715) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 39.2203 ± 0.111208 (W 0.0740202) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.531692 ± 0.00118772 (W 0.00215413) | 0.514864 ± 0.00329432 (W 0.00276712) | · |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.591899 ± 0.000943013 (W 0.00324163) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 14.8026 ± 0.0338635 (W 0.0600323) | 13.8107 ± 0.0238382 (W 0.0304203) | 🔴 faer-fork 1.07× slower than `lapack-aocl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 15.9769 ± 0.0335159 (W 0.0464833) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.78492 ± 0.0057915 (W 0.00385413) | 3.52583 ± 0.013963 (W 0.0071693) | 🔵 faer-fork 1.98× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.40084 ± 0.0312286 (W 0.0107324) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 71.5288 ± 0.146726 (W 0.0985973) | 98.417 ± 0.233299 (W 0.324394) | 🔵 faer-fork 1.38× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 85.3431 ± 0.294491 (W 0.183638) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.14124 ± 0.00143008 (W 0.00296956) | 2.22289 ± 0.00497081 (W 0.0041163) | 🔵 faer-fork 1.95× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 32.1279 ± 0.0654798 (W 0.0281835) | 43.977 ± 0.126934 (W 0.0501872) | 🔵 faer-fork 1.37× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 2.55911 ± 0.00628728 (W 0.00773486) | 5.69265 ± 0.0288465 (W 0.00532634) | 🔵 faer-fork 2.22× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.57521 ± 0.00982085 (W 0.00459986) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 86.145 ± 0.149297 (W 0.205971) | 132.273 ± 4.89682 (W 5.37154) | 🔵 faer-fork 1.54× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 86.7423 ± 0.132138 (W 0.0838565) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.5536 ± 0.00430973 (W 0.00346373) | 2.76031 ± 0.00997129 (W 0.00525852) | 🔵 faer-fork 1.78× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.55443 ± 0.00702609 (W 0.00527375) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 40.0489 ± 0.0806932 (W 0.0593361) | 59.2475 ± 0.109784 (W 0.0919153) | 🔵 faer-fork 1.48× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 40.0328 ± 0.118231 (W 0.071232) | - | - | `*` one row has no measurement here |

## linalg-tall, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.642024 ± 0.0034844 (W 0.00237326) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.703352 ± 0.00471762 (W 0.00381623) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 23.2789 ± 0.118592 (W 0.0508101) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 23.493 ± 0.104659 (W 0.051734) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.378087 ± 0.0030063 (W 0.00332369) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.382315 ± 0.0039751 (W 0.00316998) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 25.5513 ± 0.813531 (W 0.900814) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 25.5403 ± 0.677382 (W 1.00254) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.46966 ± 0.00312403 (W 0.0026209) | 0.569689 ± 0.00211192 (W 0.00484255) | 🔵 faer-fork 1.21× faster than `lapack-aocl` |  |
| `lu/1024x64/b1 [api-inclusive]` | 0.531832 ± 0.00232048 (W 0.00147564) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 10.6987 ± 0.018477 (W 0.0248918) | 9.93099 ± 0.0299047 (W 0.0599125) | 🔴 faer-fork 1.08× slower than `lapack-aocl` |  |
| `lu/4096x256/b1 [api-inclusive]` | 11.7287 ± 0.0273177 (W 0.0408811) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.38829 ± 0.00474783 (W 0.0035988) | 3.35586 ± 0.0109226 (W 0.0116861) | 🔵 faer-fork 2.42× faster than `lapack-aocl` |  |
| `qr/1024x64/b1 [api-inclusive]` | 2.04035 ± 0.021504 (W 0.00705653) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 48.5137 ± 0.0881918 (W 0.0906755) | 65.8455 ± 0.232005 (W 0.103711) | 🔵 faer-fork 1.36× faster than `lapack-aocl` |  |
| `qr/4096x256/b1 [api-inclusive]` | 62.2484 ± 0.280785 (W 0.144878) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.0239 ± 0.00209663 (W 0.00294512) | 2.39776 ± 0.00672429 (W 0.00662879) | 🔵 faer-fork 2.34× faster than `lapack-aocl` |  |
| `qr_factor/4096x256/b1` | 23.7077 ± 0.0537719 (W 0.0570663) | 30.6799 ± 0.0950443 (W 0.0455986) | 🔵 faer-fork 1.29× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1` | 2.1645 ± 0.007907 (W 0.00542531) | 5.91341 ± 0.0232765 (W 0.00899705) | 🔵 faer-fork 2.73× faster than `lapack-aocl` |  |
| `svd_thin/1024x64/b1 [api-inclusive]` | 2.18973 ± 0.00670952 (W 0.00373417) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 60.9495 ± 0.0528656 (W 0.150109) | 130.434 ± 7.65749 (W 10.6632) | 🔵 faer-fork 2.14× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 61.5704 ± 0.109197 (W 0.11244) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.45386 ± 0.00552158 (W 0.00510817) | 2.93121 ± 0.00781697 (W 0.00392758) | 🔵 faer-fork 2.02× faster than `lapack-aocl` |  |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.45737 ± 0.00250304 (W 0.00388381) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 31.2188 ± 0.0608493 (W 0.0373215) | 47.5829 ± 0.378634 (W 0.221706) | 🔵 faer-fork 1.52× faster than `lapack-aocl` |  |
| `svd_values/4096x256/b1 [api-inclusive]` | 31.1905 ± 0.0715618 (W 0.0615165) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.670538 ± 0.007754 (W 0.00626793) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.54387 ± 0.00292009 (W 0.00246697) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 25.8179 ± 0.144555 (W 0.0865671) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 23.7248 ± 0.116502 (W 0.11025) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.280152 ± 0.00173784 (W 0.000529362) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.279033 ± 0.00147892 (W 0.00167713) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.4025 ± 0.0773954 (W 0.0207054) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.4068 ± 0.103121 (W 0.0420549) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.323697 ± 0.00284283 (W 0.00266071) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.240453 ± 0.00114036 (W 0.00180734) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 7.71483 ± 0.0542627 (W 0.00670414) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 8.19917 ± 0.199354 (W 0.0937659) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.963677 ± 0.00324962 (W 0.00217764) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.02893 ± 0.00469138 (W 0.00224728) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 36.5944 ± 0.416624 (W 0.22416) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 38.4255 ± 0.116747 (W 0.132673) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.696294 ± 0.00432758 (W 0.00189306) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 18.5641 ± 0.255783 (W 0.0458702) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.36278 ± 0.00460274 (W 0.00222574) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.36476 ± 0.0051765 (W 0.0024552) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 45.0097 ± 0.711881 (W 0.0533292) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 45.1436 ± 0.703526 (W 0.0295441) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.732542 ± 0.00607006 (W 0.00393072) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.730333 ± 0.00415885 (W 0.00425892) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 21.7498 ± 0.568019 (W 0.0394704) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/4096x256/b1 [api-inclusive]` | 21.8175 ± 0.555789 (W 0.0472214) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.456896 ± 0.00292092 (W 0.0037441) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.46726 ± 0.00332835 (W 0.00271431) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.54595 ± 0.0437288 (W 0.0244468) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 10.8504 ± 0.0808697 (W 0.0404305) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.0886665 ± 0.000521113 (W 0.00172544) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.088347 ± 0.000656065 (W 0.00168909) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 4.64796 ± 0.00919826 (W 0.0144105) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 4.64339 ± 0.0147744 (W 0.0190164) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.219314 ± 0.000950723 (W 0.00187702) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.232482 ± 0.00206279 (W 0.00232358) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 3.92222 ± 0.0114209 (W 0.0207345) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.25545 ± 0.111876 (W 0.100287) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.564045 ± 0.00279767 (W 0.00236043) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.748301 ± 0.00267115 (W 0.00307285) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 12.9378 ± 0.070578 (W 0.0311036) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 17.5798 ± 0.158793 (W 0.146948) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.463511 ± 0.00294898 (W 0.00202366) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 7.9209 ± 0.0210268 (W 0.0225151) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.01215 ± 0.00308319 (W 0.0028685) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.01629 ± 0.00436226 (W 0.00192827) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 21.9487 ± 0.69352 (W 0.0302491) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 22.0569 ± 0.695847 (W 0.0356154) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/1024x64/b1` | 0.6797 ± 0.00615948 (W 0.00469092) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.680539 ± 0.00521741 (W 0.00494087) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 14.3283 ± 0.024438 (W 0.0139796) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 14.3797 ± 0.543638 (W 0.0286493) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-tall, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.440352 ± 0.00320302 (W 0.00427062) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.471865 ± 0.00369899 (W 0.00334305) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.9594 ± 0.0347394 (W 0.0293738) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.72797 ± 0.0455234 (W 0.043473) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.078476 ± 0.0020503 (W 0.0026346) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.077404 ± 0.00131776 (W 0.00179349) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 2.59543 ± 0.0575138 (W 0.00667738) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 2.58205 ± 0.0126636 (W 0.00810662) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.194774 ± 0.00190625 (W 0.00175248) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.229625 ± 0.00195524 (W 0.00201913) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.84519 ± 0.00667032 (W 0.00634121) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 4.68101 ± 0.172518 (W 0.125657) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1` | 0.522327 ± 0.00266999 (W 0.00457923) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.708277 ± 0.00380972 (W 0.00369263) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 9.13015 ± 0.0697861 (W 0.0317208) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.1664 ± 0.276344 (W 0.194918) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/1024x64/b1` | 0.428185 ± 0.0027941 (W 0.0031115) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.87317 ± 0.0178261 (W 0.0137588) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.960133 ± 0.00625166 (W 0.00273467) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.96784 ± 0.00347135 (W 0.0032126) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 17.6573 ± 0.54662 (W 0.051271) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/4096x256/b1 [api-inclusive]` | 17.6925 ± 0.0325153 (W 0.0406595) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.666432 ± 0.00745414 (W 0.00650963) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.665928 ± 0.00479739 (W 0.00320484) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 12.821 ± 0.0526202 (W 0.0231789) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 12.8363 ± 0.0257815 (W 0.0119417) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.4932 ± 0.014383 (W 0.00224939) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.461162 ± 0.00103355 (W 0.0028559) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 24.3714 ± 0.154889 (W 0.0229754) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 21.4372 ± 0.102905 (W 0.0275684) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.302681 ± 0.000983063 (W 0.00264722) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.300959 ± 0.00126185 (W 0.00256206) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.7953 ± 0.0529049 (W 0.0238034) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.8117 ± 0.0830946 (W 0.036981) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.148342 ± 0.00160375 (W 0.00149929) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.156055 ± 0.000918181 (W 0.00152475) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 5.6474 ± 0.0865978 (W 0.0185373) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 5.77222 ± 0.0869408 (W 0.0124804) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.745808 ± 0.00305827 (W 0.00224992) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.903282 ± 0.0053963 (W 0.00301669) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 31.8376 ± 0.132459 (W 0.0390546) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 35.4441 ± 0.251136 (W 0.0428087) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.442711 ± 0.00209054 (W 0.0020804) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.9348 ± 0.100927 (W 0.017925) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.35896 ± 0.00520364 (W 0.00426914) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.36247 ± 0.00409134 (W 0.00335247) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 42.0301 ± 0.19444 (W 0.0351108) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 42.1381 ± 0.258198 (W 0.0524733) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.692812 ± 0.00286132 (W 0.00224578) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.688551 ± 0.0040325 (W 0.00207951) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 18.4524 ± 0.129521 (W 0.0298814) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 18.4461 ± 0.133582 (W 0.0217673) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.370131 ± 0.0068166 (W 0.00272187) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.387636 ± 0.00341209 (W 0.00267294) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 9.12123 ± 0.019515 (W 0.00991952) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 8.67533 ± 0.0172495 (W 0.01898) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.104375 ± 0.00092205 (W 0.000809223) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.104579 ± 0.000569551 (W 0.00136743) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 5.49378 ± 0.00627429 (W 0.0116775) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 5.49981 ± 0.00512535 (W 0.0101849) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.141403 ± 0.000778133 (W 0.00102414) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.150405 ± 0.00118958 (W 0.00183188) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.70493 ± 0.0028469 (W 0.00548434) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.89877 ± 0.0114124 (W 0.00608891) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.491004 ± 0.00233352 (W 0.00170157) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.620383 ± 0.00250165 (W 0.00367479) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 11.7033 ± 0.0587561 (W 0.0121978) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 14.8108 ± 0.0704486 (W 0.0292983) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.388753 ± 0.00137797 (W 0.00196103) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 6.71564 ± 0.045191 (W 0.0104204) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.99826 ± 0.00355454 (W 0.00152589) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.00793 ± 0.00373977 (W 0.00383291) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 18.6671 ± 0.0694878 (W 0.0288362) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 18.7142 ± 0.044198 (W 0.0295891) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.637655 ± 0.00276459 (W 0.00158) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.634209 ± 0.00382865 (W 0.00241975) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 10.9178 ± 0.0306914 (W 0.0201384) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 10.9045 ± 0.0300578 (W 0.0150227) | - | - | `*` one row has no measurement here |

## linalg-tall, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.382445 ± 0.000783225 (W 0.00246203) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.393133 ± 0.00262094 (W 0.00253793) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 6.70587 ± 0.02056 (W 0.0180053) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 6.62274 ± 0.0319345 (W 0.0255393) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.0690935 ± 0.000545055 (W 0.000667392) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.0689375 ± 0.000583144 (W 0.000834751) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 3.96227 ± 0.147549 (W 0.215419) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 3.90875 ± 0.23541 (W 0.184551) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/1024x64/b1` | 0.135382 ± 0.000939742 (W 0.00111108) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.147359 ± 0.00257262 (W 0.00150113) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 2.2285 ± 0.0110627 (W 0.00484647) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 2.47253 ± 0.0164817 (W 0.00631325) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.439686 ± 0.00284042 (W 0.00254318) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 0.583485 ± 0.00221295 (W 0.00271732) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 8.18244 ± 0.189608 (W 0.0237019) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1 [api-inclusive]` | 11.4128 ± 0.0365 (W 0.0583997) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.375885 ± 0.0017912 (W 0.00233251) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 5.17658 ± 0.00642931 (W 0.0080906) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 0.950368 ± 0.00885175 (W 0.00174453) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 0.95267 ± 0.00386107 (W 0.00356409) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 14.9389 ± 0.0952723 (W 0.0436952) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 14.9776 ± 0.0426121 (W 0.0250765) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.620327 ± 0.0018238 (W 0.00300905) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.620969 ± 0.00172674 (W 0.00241295) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 9.34001 ± 0.028454 (W 0.0101648) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 9.32539 ± 0.0475781 (W 0.00893052) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 1.16229 ± 0.00949513 (W 0.00282447) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 1.03462 ± 0.00351516 (W 0.00199836) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 62.0974 ± 0.508291 (W 0.476788) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 58.9619 ± 0.551024 (W 0.138525) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.09946 ± 0.00387062 (W 0.00134187) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.10221 ± 0.00370966 (W 0.00116987) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 68.3209 ± 0.170697 (W 0.121814) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 68.3665 ± 0.168149 (W 0.0561955) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.651355 ± 0.00216271 (W 0.00265222) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.617992 ± 0.00364133 (W 0.00188559) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 22.8823 ± 0.120257 (W 0.0358429) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 24.7768 ± 0.131157 (W 0.0532138) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.50122 ± 0.167881 (W 0.00594594) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 2.77367 ± 0.178169 (W 0.00451853) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 123.173 ± 0.467654 (W 0.281233) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 128.326 ± 0.380706 (W 0.289869) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.41335 ± 0.00713465 (W 0.00480132) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 53.7316 ± 0.1915 (W 0.0995157) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.06658 ± 0.00913617 (W 0.00588982) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.0693 ± 0.00987574 (W 0.00525731) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 140.289 ± 0.406661 (W 0.344329) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 140.312 ± 0.350922 (W 0.190587) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.50439 ± 0.00788149 (W 0.00589354) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.50329 ± 0.00551213 (W 0.00660645) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 59.8262 ± 0.208949 (W 0.055954) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 59.8969 ± 0.143065 (W 0.0520611) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.649795 ± 0.00381855 (W 0.0036554) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.681408 ± 0.00571272 (W 0.00498496) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 19.3185 ± 0.111394 (W 0.0295656) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 20.606 ± 0.142174 (W 0.0506665) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.296964 ± 0.00156164 (W 0.00187087) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.297833 ± 0.0017972 (W 0.00193253) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 17.4604 ± 0.0322044 (W 0.0222414) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 17.4636 ± 0.0182267 (W 0.0580228) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.462282 ± 0.00158327 (W 0.00244975) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.498981 ± 0.00344048 (W 0.00199713) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 11.4631 ± 0.0520682 (W 0.0506794) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 12.7426 ± 0.0700399 (W 0.0554614) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.09512 ± 0.00360692 (W 0.00349193) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.47126 ± 0.00342726 (W 0.00634716) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 40.0932 ± 0.165135 (W 0.109066) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 48.9236 ± 0.154932 (W 0.112816) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.806021 ± 0.00295284 (W 0.00211936) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 21.6453 ± 0.069957 (W 0.0275565) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.67144 ± 0.00911928 (W 0.00565992) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.69306 ± 0.0123776 (W 0.00530068) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 53.2513 ± 0.0803068 (W 0.0836621) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 53.5601 ± 0.103881 (W 0.0846691) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.07128 ± 0.00324757 (W 0.00442752) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.07834 ± 0.00789961 (W 0.00328305) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 29.7653 ± 0.0671171 (W 0.0515067) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 29.7645 ± 0.0395746 (W 0.0340054) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.582469 ± 0.00467011 (W 0.00311768) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.638156 ± 0.00252179 (W 0.00279091) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 12.1156 ± 0.0447771 (W 0.022207) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 14.6933 ± 0.052928 (W 0.0337198) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.175211 ± 0.00292779 (W 0.00423941) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.174571 ± 0.00161851 (W 0.00221895) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 9.5497 ± 0.028541 (W 0.0703277) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 9.56417 ± 0.0572821 (W 0.0913872) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.420545 ± 0.00204787 (W 0.00139813) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.479982 ± 0.00193313 (W 0.00219727) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.30961 ± 0.0151183 (W 0.0155638) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 10.8493 ± 0.0424056 (W 0.030277) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.900753 ± 0.00485005 (W 0.00362575) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.28571 ± 0.00591915 (W 0.00560408) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 25.7186 ± 0.0667279 (W 0.0854635) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 36.539 ± 0.142669 (W 0.112388) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.735368 ± 0.00178062 (W 0.0030853) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 15.5647 ± 0.0355176 (W 0.0417062) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.48036 ± 0.00914288 (W 0.00406915) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.48624 ± 0.00580811 (W 0.00522457) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 40.1619 ± 0.0562459 (W 0.0559868) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 40.4254 ± 0.0565319 (W 0.0690874) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.02141 ± 0.00501471 (W 0.00608707) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.02261 ± 0.00460415 (W 0.0045837) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 24.7316 ± 0.0506611 (W 0.0335246) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.7117 ± 0.055138 (W 0.0369284) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.942494 ± 0.00438818 (W 0.00311372) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.925238 ± 0.00456574 (W 0.00241815) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 58.7324 ± 0.362697 (W 0.0797992) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 55.5212 ± 0.146259 (W 0.141951) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 1.14414 ± 0.00439614 (W 0.00158444) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 1.14384 ± 0.00540588 (W 0.00250326) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 69.0954 ± 0.184244 (W 0.0712443) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 69.1274 ± 0.119553 (W 0.140544) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.489315 ± 0.00411805 (W 0.00247017) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.505204 ± 0.00364467 (W 0.00215687) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 20.936 ± 0.0584939 (W 0.0196002) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 21.4499 ± 0.0671999 (W 0.0499984) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 2.30729 ± 0.0789211 (W 0.0040609) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/1024x64/b1 [api-inclusive]` | 2.64678 ± 0.16664 (W 0.00716286) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/4096x256/b1` | 117.155 ± 0.374934 (W 0.142412) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 124.774 ± 0.478453 (W 0.339572) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 1.15103 ± 0.00369217 (W 0.00307096) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 46.8239 ± 0.112573 (W 0.0551425) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 3.02926 ± 0.00979906 (W 0.00575852) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 3.03009 ± 0.0122675 (W 0.00473102) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 134.899 ± 0.412066 (W 0.257975) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 135.21 ± 0.464797 (W 0.318563) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.43202 ± 0.00570097 (W 0.00565981) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.42721 ± 0.0051083 (W 0.00388474) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 54.354 ± 0.151654 (W 0.0718588) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 54.4184 ± 0.117933 (W 0.0944261) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.549211 ± 0.00321309 (W 0.00284549) | - | - | `*` one row has no measurement here |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.577827 ± 0.00276535 (W 0.00366354) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 17.3662 ± 0.117371 (W 0.0471855) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 17.0819 ± 0.0883878 (W 0.032421) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.340315 ± 0.00211172 (W 0.00169107) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.337969 ± 0.00197696 (W 0.00201195) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 19.2784 ± 0.0404123 (W 0.0218391) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 19.2876 ± 0.025803 (W 0.0273283) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.362426 ± 0.00456022 (W 0.00228923) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.393538 ± 0.00353732 (W 0.00212921) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 8.68804 ± 0.0196883 (W 0.0128306) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 9.20332 ± 0.0236186 (W 0.0230408) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 1.04674 ± 0.00325232 (W 0.00167484) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.33955 ± 0.00770425 (W 0.00479918) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 38.331 ± 0.0566612 (W 0.222132) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 44.847 ± 0.0539547 (W 0.079312) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.710758 ± 0.00224685 (W 0.00239423) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 18.8769 ± 0.0567226 (W 0.0154707) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.62991 ± 0.00535585 (W 0.00488021) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.64258 ± 0.00387596 (W 0.00390739) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 47.9531 ± 0.0913832 (W 0.0662067) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 48.2495 ± 0.12388 (W 0.0596665) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 1.00837 ± 0.00409318 (W 0.00277682) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 1.01054 ± 0.00623761 (W 0.00224215) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 24.193 ± 0.133744 (W 0.0335599) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 24.2942 ± 0.110982 (W 0.0287221) | - | - | `*` one row has no measurement here |

## linalg-tall, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/1024x64/b1` | 0.505206 ± 0.0630064 (W 0.0031355) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/1024x64/b1 [api-inclusive]` | 0.537477 ± 0.00191439 (W 0.00372627) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1` | 11.2083 ± 0.0568223 (W 0.01698) | - | - | `*` one row has no measurement here |
| `full_piv_lu/4096x256/b1 [api-inclusive]` | 11.2931 ± 0.0548247 (W 0.0142262) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [block(basis=64,target=64)]` | 0.197394 ± 0.000775863 (W 0.000991832) | - | - | `*` one row has no measurement here |
| `householder_apply/1024x64/b1 [sequence(basis=64,target=64)]` | 0.198684 ± 0.00228111 (W 0.00138265) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [block(basis=256,target=256)]` | 12.5449 ± 0.0138408 (W 0.0573142) | - | - | `*` one row has no measurement here |
| `householder_apply/4096x256/b1 [sequence(basis=256,target=256)]` | 12.546 ± 0.0451964 (W 0.0620056) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1` | 0.346571 ± 0.00186629 (W 0.00163368) | - | - | `*` one row has no measurement here |
| `lu/1024x64/b1 [api-inclusive]` | 0.375764 ± 0.00395561 (W 0.00124148) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1` | 6.75437 ± 0.0135842 (W 0.00880696) | - | - | `*` one row has no measurement here |
| `lu/4096x256/b1 [api-inclusive]` | 7.31933 ± 0.0282761 (W 0.0128572) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1` | 0.851632 ± 0.013691 (W 0.00330906) | - | - | `*` one row has no measurement here |
| `qr/1024x64/b1 [api-inclusive]` | 1.15217 ± 0.00356802 (W 0.00374893) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1` | 25.9442 ± 0.0265134 (W 0.0400737) | - | - | `*` one row has no measurement here |
| `qr/4096x256/b1 [api-inclusive]` | 32.6547 ± 0.101368 (W 0.0874548) | - | - | `*` one row has no measurement here |
| `qr_factor/1024x64/b1` | 0.656498 ± 0.00296524 (W 0.00332606) | - | - | `*` one row has no measurement here |
| `qr_factor/4096x256/b1` | 13.8983 ± 0.0236695 (W 0.034792) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1` | 1.44265 ± 0.00599903 (W 0.00324121) | - | - | `*` one row has no measurement here |
| `svd_thin/1024x64/b1 [api-inclusive]` | 1.45355 ± 0.00335151 (W 0.0033397) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1` | 34.7618 ± 0.0841507 (W 0.0567596) | - | - | `*` one row has no measurement here |
| `svd_thin/4096x256/b1 [api-inclusive]` | 35.0511 ± 0.0636883 (W 0.0553299) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1` | 0.960843 ± 0.00349722 (W 0.00193356) | - | - | `*` one row has no measurement here |
| `svd_values/1024x64/b1 [api-inclusive]` | 0.95961 ± 0.00408822 (W 0.00307617) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1` | 19.1103 ± 0.0258327 (W 0.0377112) | - | - | `*` one row has no measurement here |
| `svd_values/4096x256/b1 [api-inclusive]` | 19.1417 ± 0.0250449 (W 0.0401948) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024x64`, `cholesky/4096x256`, `eigh/1024x64`, `eigh/4096x256`, `full_piv_lu/1024x64`, `full_piv_lu/4096x256`, `householder_apply/1024x64`, `householder_apply/4096x256`, `lu/1024x64`, `lu/4096x256`, `qr/1024x64`, `qr/4096x256`, `qr_factor/1024x64`, `qr_factor/4096x256`, `solve/1024x64`, `solve/4096x256`, `svd_thin/1024x64`, `svd_thin/4096x256`, `svd_values/1024x64`, `svd_values/4096x256`, `triangular_solve/1024x64`, `triangular_solve/4096x256`, `upgrade_factor/1024x64`, `upgrade_factor/4096x256`.

## linalg-wide, f64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 52.1518 ± 0.389874 (W 0.178154) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 76.2255 ± 1.0007 (W 0.402041) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.794131 ± 0.00806544 (W 0.00269998) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.26774 ± 0.00415451 (W 0.00504565) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.6177 ± 0.037039 (W 0.0212699) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 15.8741 ± 0.0617769 (W 0.0663668) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.27821 ± 0.000616744 (W 0.00210476) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.429535 ± 0.00253227 (W 0.00261975) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 28.8055 ± 0.0686448 (W 0.039997) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 30.6235 ± 0.140014 (W 0.0681413) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.03189 ± 0.00582805 (W 0.00569784) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.756072 ± 0.0025237 (W 0.00225474) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.4983 ± 0.283283 (W 0.034359) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.990371 ± 0.00783836 (W 0.00442336) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 77.3677 ± 0.349347 (W 0.231352) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 77.6043 ± 0.207668 (W 0.236775) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.97429 ± 0.00903563 (W 0.00428427) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.98155 ± 0.00618345 (W 0.00327111) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 31.4787 ± 0.0832227 (W 0.0380317) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.5022 ± 0.109549 (W 0.0337338) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.92591 ± 0.0039795 (W 0.00220039) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.926838 ± 0.00325325 (W 0.00279859) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 15.6196 ± 0.0468324 (W 0.0180557) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 33.8183 ± 0.119397 (W 0.0741932) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.467813 ± 0.00296767 (W 0.00334628) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.762783 ± 0.004046 (W 0.00344073) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.51581 ± 0.0094878 (W 0.0137164) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.47395 ± 0.0684041 (W 0.0615411) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.119122 ± 0.00247337 (W 0.00117827) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.226116 ± 0.0010322 (W 0.00146948) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.4993 ± 0.169077 (W 0.0236617) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 10.3998 ± 0.0487324 (W 0.0269948) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.637892 ± 0.00510503 (W 0.00418767) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.378343 ± 0.00115204 (W 0.00156833) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.1128 ± 0.0589136 (W 0.0443221) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.614146 ± 0.0048003 (W 0.00510119) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 31.4346 ± 0.0809015 (W 0.0514845) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 31.6406 ± 0.0664812 (W 0.0700463) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.29504 ± 0.00535927 (W 0.00349148) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.31625 ± 0.00388876 (W 0.004252) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 16.2194 ± 0.0426274 (W 0.0226551) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 16.2326 ± 0.0356696 (W 0.0326584) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.788666 ± 0.00287203 (W 0.00166841) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.793692 ± 0.00215636 (W 0.00205193) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.1018 ± 0.0739672 (W 0.0262522) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 27.8517 ± 0.0747156 (W 0.0520387) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.440016 ± 0.00256561 (W 0.00278358) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.728013 ± 0.00499265 (W 0.00294294) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 2.13658 ± 0.00798536 (W 0.00784315) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 4.91018 ± 0.0252621 (W 0.0361126) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.09463 ± 0.00121769 (W 0.0018352) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.19631 ± 0.00100238 (W 0.00154895) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.59759 ± 0.162789 (W 0.0281524) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 7.078 ± 0.136502 (W 0.0277224) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 0.568014 ± 0.00344374 (W 0.00535713) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.324276 ± 0.00112369 (W 0.00216456) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 7.3366 ± 0.0466957 (W 0.0269006) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.547475 ± 0.00161425 (W 0.00303025) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 25.1083 ± 0.0617453 (W 0.101465) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 25.3123 ± 0.0648604 (W 0.0344394) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.20327 ± 0.0111466 (W 0.00289255) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.21633 ± 0.00773341 (W 0.00322923) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 13.7585 ± 0.0280547 (W 0.0184894) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 13.7605 ± 0.0234533 (W 0.0267391) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.764044 ± 0.00224898 (W 0.00238983) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.765397 ± 0.00284339 (W 0.00299255) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 88.3545 ± 1.18369 (W 0.200876) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 74.4247 ± 0.554039 (W 0.213607) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.22113 ± 0.00254936 (W 0.00606567) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.16683 ± 0.0047183 (W 0.00454718) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 12.6194 ± 0.0955284 (W 0.0357663) | 26.581 ± 0.0629932 (W 0.0486212) | 🔵 faer-fork 2.11× faster than `lapack-aocl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 13.0118 ± 0.033581 (W 0.0460653) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.306155 ± 0.00131729 (W 0.00195798) | 0.261597 ± 0.00197654 (W 0.00206983) | 🔴 faer-fork 1.17× slower than `lapack-aocl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.322294 ± 0.00235638 (W 0.00288533) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 26.5988 ± 0.734082 (W 0.209527) | 85.4907 ± 0.254552 (W 0.17433) | 🔵 faer-fork 3.21× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 27.7911 ± 0.213117 (W 0.0929682) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.604068 ± 0.00202815 (W 0.00238105) | 0.984171 ± 0.00508156 (W 0.00281394) | 🔵 faer-fork 1.63× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.625692 ± 0.000753123 (W 0.00245527) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 25.05 ± 0.962522 (W 0.0719008) | 83.2762 ± 0.164452 (W 0.124944) | 🔵 faer-fork 3.32× faster than `lapack-aocl` | `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 0.56756 ± 0.00441112 (W 0.00269424) | 0.926088 ± 0.00498541 (W 0.00190638) | 🔵 faer-fork 1.63× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 82.7051 ± 0.174329 (W 0.125616) | 185.006 ± 1.04836 (W 0.444124) | 🔵 faer-fork 2.24× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 82.9073 ± 0.230961 (W 0.277429) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.01772 ± 0.0119037 (W 0.00381171) | 4.48396 ± 0.0698545 (W 0.0104769) | 🔵 faer-fork 2.22× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.02722 ± 0.00984874 (W 0.00337798) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 37.2366 ± 0.0637853 (W 0.0484482) | 77.192 ± 0.226315 (W 0.197691) | 🔵 faer-fork 2.07× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 37.2438 ± 0.0763086 (W 0.0617828) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.00254 ± 0.00818042 (W 0.00298272) | 2.38695 ± 0.0104079 (W 0.00888071) | 🔵 faer-fork 2.38× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.00783 ± 0.00777316 (W 0.00453199) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 33.103 ± 0.125383 (W 0.0599388) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 30.7982 ± 0.132045 (W 0.0816863) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.661981 ± 0.0038588 (W 0.00395475) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.670304 ± 0.0029576 (W 0.00234905) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.36681 ± 0.00895734 (W 0.00759386) | 8.53687 ± 0.052197 (W 0.0155212) | 🔵 faer-fork 2.54× faster than `lapack-aocl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 3.79851 ± 0.011064 (W 0.010151) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0975165 ± 0.000359159 (W 0.000451487) | 0.115201 ± 0.000926917 (W 0.00112768) | 🔵 faer-fork 1.18× faster than `lapack-aocl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.117085 ± 0.000683439 (W 0.000669653) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.00344 ± 0.0348211 (W 0.0174935) | 25.1963 ± 0.0954549 (W 0.0404483) | 🔵 faer-fork 3.60× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 7.5674 ± 0.0398367 (W 0.0182006) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.218177 ± 0.00110229 (W 0.00142959) | 0.975663 ± 0.00451436 (W 0.00254449) | 🔵 faer-fork 4.47× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.250408 ± 0.000880331 (W 0.00140951) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.54699 ± 0.0619407 (W 0.019254) | 24.3422 ± 0.116149 (W 0.0423718) | 🔵 faer-fork 3.72× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.199041 ± 0.00174931 (W 0.00131105) | 0.914844 ± 0.0030204 (W 0.002265) | 🔵 faer-fork 4.60× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 36.6963 ± 0.0852291 (W 0.112178) | 142.855 ± 1.13882 (W 0.950485) | 🔵 faer-fork 3.89× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 36.9683 ± 0.0487921 (W 0.0602113) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.32794 ± 0.00819368 (W 0.00366235) | 4.96604 ± 0.0669719 (W 0.0500584) | 🔵 faer-fork 3.74× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.34201 ± 0.00954561 (W 0.00285716) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.9552 ± 0.0352537 (W 0.0272394) | 73.7589 ± 0.229561 (W 0.546182) | 🔵 faer-fork 3.36× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.944 ± 0.0360841 (W 0.0223852) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.857968 ± 0.00619952 (W 0.00439423) | 2.38162 ± 0.0103935 (W 0.00680508) | 🔵 faer-fork 2.78× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.863304 ± 0.00907071 (W 0.00429434) | - | - | `*` one row has no measurement here |

## linalg-wide, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 25.9825 ± 0.260027 (W 0.0375612) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 25.2494 ± 0.111636 (W 0.0487191) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.603713 ± 0.0354086 (W 0.0026901) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.628344 ± 0.00420886 (W 0.0027767) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.85685 ± 0.0148507 (W 0.0147987) | 8.71914 ± 0.0313771 (W 0.0171204) | 🔵 faer-fork 4.70× faster than `lapack-aocl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 2.28775 ± 0.00156137 (W 0.00626933) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0623 ± 0.000420819 (W 0.000394556) | 0.114689 ± 0.000625725 (W 0.000471808) | 🔵 faer-fork 1.84× faster than `lapack-aocl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.087012 ± 0.000407841 (W 0.000433835) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 3.82854 ± 0.0166463 (W 0.0153126) | 16.6909 ± 0.090166 (W 0.0816448) | 🔵 faer-fork 4.36× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 4.32308 ± 0.0149041 (W 0.0110062) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.163586 ± 0.000811847 (W 0.00108111) | 0.985483 ± 0.00497984 (W 0.00235418) | 🔵 faer-fork 6.02× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.198237 ± 0.00118321 (W 0.00177625) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.56519 ± 0.0117632 (W 0.00841595) | 16.0753 ± 0.129236 (W 0.0736504) | 🔵 faer-fork 4.51× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.146691 ± 0.00226331 (W 0.00134984) | 0.922579 ± 0.00334161 (W 0.00226693) | 🔵 faer-fork 6.29× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 30.6709 ± 0.0975955 (W 0.0730629) | 146.276 ± 1.19642 (W 1.10126) | 🔵 faer-fork 4.77× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 30.8223 ± 0.10709 (W 0.0521492) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.24386 ± 0.00875362 (W 0.00417289) | 5.02086 ± 0.13638 (W 0.0505671) | 🔵 faer-fork 4.04× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.26624 ± 0.0150194 (W 0.00420581) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 19.4241 ± 0.062606 (W 0.0419681) | 69.4042 ± 0.770214 (W 0.613385) | 🔵 faer-fork 3.57× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.4373 ± 0.0756944 (W 0.0475133) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.833588 ± 0.00434408 (W 0.00422459) | 2.37706 ± 0.0076848 (W 0.00790454) | 🔵 faer-fork 2.85× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.834353 ± 0.00553777 (W 0.0043157) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 149.526 ± 0.79552 (W 0.857805) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 235.273 ± 0.946 (W 0.757959) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.65844 ± 0.0030656 (W 0.00435568) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.61227 ± 0.028595 (W 0.00767808) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 40.6934 ± 0.134261 (W 0.0501897) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 47.4617 ± 0.0560453 (W 0.0820802) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.864537 ± 0.00270853 (W 0.0016975) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.994411 ± 0.00412224 (W 0.00575247) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 102.523 ± 0.435558 (W 0.249715) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 106.048 ± 0.506355 (W 0.308279) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.7315 ± 0.0130807 (W 0.0087717) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 2.49878 ± 0.00803598 (W 0.00673575) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 96.7032 ± 0.360603 (W 0.147573) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 2.58026 ± 0.014261 (W 0.00548782) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 259.994 ± 0.68998 (W 0.426111) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 260.526 ± 0.609082 (W 6.69561) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.34083 ± 0.00714012 (W 0.00988954) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.38744 ± 0.0226567 (W 0.012812) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 100.062 ± 0.272432 (W 0.168367) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 100.058 ± 0.318108 (W 0.0992491) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.35462 ± 0.00995666 (W 0.00423572) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.37165 ± 0.00696443 (W 0.0030434) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 46.6655 ± 0.106353 (W 0.0892135) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 84.9211 ± 0.221863 (W 0.460361) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.722801 ± 0.00832413 (W 0.00374352) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.38098 ± 0.00884251 (W 0.00510163) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.4073 ± 0.0449942 (W 0.0321615) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 16.5461 ± 0.078957 (W 0.0754668) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.293772 ± 0.00128979 (W 0.00221367) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.404429 ± 0.00331694 (W 0.0036251) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 29.4736 ± 0.129581 (W 0.160764) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 31.779 ± 0.104932 (W 0.159774) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 1.04226 ± 0.00448296 (W 0.00501002) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.872799 ± 0.0020476 (W 0.00230041) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 27.9113 ± 0.0865103 (W 0.047345) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.986467 ± 0.00598903 (W 0.00438682) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 86.6902 ± 0.164856 (W 0.0742504) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 87.2896 ± 0.157969 (W 0.0744856) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.59163 ± 0.00720842 (W 0.00598792) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.61079 ± 0.0106044 (W 0.007069) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 40.0625 ± 0.101609 (W 0.0464894) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 40.0794 ± 0.0565049 (W 0.0619035) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.54608 ± 0.00685347 (W 0.00648395) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.54544 ± 0.00518312 (W 0.00731749) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 22.9433 ± 0.102613 (W 0.0430981) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 55.6028 ± 0.322818 (W 0.239328) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.592537 ± 0.0044323 (W 0.00328748) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.22621 ± 0.0064499 (W 0.003874) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.39429 ± 0.0199234 (W 0.0184733) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 11.5597 ± 0.0330286 (W 0.0739229) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.199411 ± 0.00115789 (W 0.00117984) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.354772 ± 0.00239447 (W 0.00176008) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 17.2377 ± 0.0309625 (W 0.0698251) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 19.2497 ± 0.0774764 (W 0.0761727) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.764046 ± 0.00404045 (W 0.00468332) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.607442 ± 0.00102222 (W 0.00193278) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.2284 ± 0.0813371 (W 0.0205024) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.723976 ± 0.00149184 (W 0.00550318) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 61.6069 ± 0.127336 (W 0.122256) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 62.3217 ± 0.0853571 (W 0.0900811) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.21299 ± 0.00460656 (W 0.00717772) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.22328 ± 0.00546885 (W 0.00722399) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 31.2384 ± 0.0584593 (W 0.0430755) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 31.2772 ± 0.134449 (W 0.0677386) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.44227 ± 0.00909427 (W 0.00274943) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.44862 ± 0.00849087 (W 0.00423806) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 253.451 ± 0.7168 (W 1.12737) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 231.727 ± 1.79565 (W 0.666254) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 2.55064 ± 0.0298193 (W 0.00811318) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 2.49208 ± 0.0263688 (W 0.00981106) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 42.3539 ± 0.0744946 (W 0.0824847) | 64.9156 ± 0.123521 (W 0.0941035) | 🔵 faer-fork 1.53× faster than `lapack-aocl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 43.665 ± 0.122335 (W 0.0751738) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.86432 ± 0.00362692 (W 0.00520521) | 0.714403 ± 0.00423954 (W 0.00696433) | 🔴 faer-fork 1.21× slower than `lapack-aocl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.895569 ± 0.00515731 (W 0.00444091) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 101.486 ± 0.280981 (W 0.339795) | 165.136 ± 0.758795 (W 0.468813) | 🔵 faer-fork 1.63× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 102.206 ± 0.242228 (W 0.290847) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 2.36588 ± 0.00987111 (W 0.00693998) | 3.35404 ± 0.0257158 (W 0.0095803) | 🔵 faer-fork 1.42× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 2.39082 ± 0.0095108 (W 0.00504146) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 95.6407 ± 0.45118 (W 0.245557) | 162.459 ± 0.635111 (W 0.785132) | 🔵 faer-fork 1.70× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 2.22694 ± 0.0086754 (W 0.00554601) | 3.09338 ± 0.0329219 (W 0.00496894) | 🔵 faer-fork 1.39× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 271.244 ± 0.79069 (W 0.425119) | 593.95 ± 1.76889 (W 0.936853) | 🔵 faer-fork 2.19× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 272.245 ± 0.743561 (W 5.89069) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 5.43423 ± 0.0152096 (W 0.00745978) | 9.70407 ± 0.243557 (W 0.0191953) | 🔵 faer-fork 1.79× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_thin/64x1024/b1 [api-inclusive]` | 5.45868 ± 0.0189218 (W 0.00628436) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 114.014 ± 0.353802 (W 0.549586) | 250.932 ± 0.871677 (W 0.601468) | 🔵 faer-fork 2.20× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 113.486 ± 0.818138 (W 0.697208) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 2.48712 ± 0.00457101 (W 0.00436789) | 4.09694 ± 0.0579127 (W 0.0097766) | 🔵 faer-fork 1.65× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 2.49614 ± 0.0120384 (W 0.00339681) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 89.5923 ± 0.350137 (W 0.427028) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 80.6926 ± 0.436707 (W 0.38938) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.20993 ± 0.00397804 (W 0.00410775) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.24969 ± 0.00763892 (W 0.00603767) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 11.3653 ± 0.446201 (W 0.527563) | 22.8896 ± 0.034248 (W 0.0311632) | 🔵 faer-fork 2.01× faster than `lapack-aocl` | `!` between-start range over 5% |
| `lu/256x4096/b1 [api-inclusive]` | 12.9868 ± 0.530258 (W 0.470438) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.33518 ± 0.0326003 (W 0.0189313) | 0.262504 ± 0.00128827 (W 0.00150905) | 🔴 faer-fork 1.28× slower than `lapack-aocl` | `!` between-start range over 5% |
| `lu/64x1024/b1 [api-inclusive]` | 0.303629 ± 0.0310411 (W 0.0399226) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 26.2646 ± 0.108764 (W 0.0401019) | 70.8395 ± 0.510424 (W 0.187244) | 🔵 faer-fork 2.70× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 27.7443 ± 0.0933098 (W 0.10269) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.710888 ± 0.00300095 (W 0.00240152) | 3.62472 ± 0.017619 (W 0.00499138) | 🔵 faer-fork 5.10× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.770026 ± 0.000965138 (W 0.00201556) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 24.6167 ± 0.0287189 (W 0.0281181) | 67.9266 ± 0.509126 (W 0.309044) | 🔵 faer-fork 2.76× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.652118 ± 0.00221572 (W 0.00207269) | 3.48024 ± 0.0155783 (W 0.00839835) | 🔵 faer-fork 5.34× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 98.7957 ± 0.104359 (W 0.12316) | 363.483 ± 0.386042 (W 0.495529) | 🔵 faer-fork 3.68× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 99.5145 ± 0.150218 (W 0.149949) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.70167 ± 0.00761037 (W 0.00728445) | 9.5831 ± 0.0862436 (W 0.0153795) | 🔵 faer-fork 3.55× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.71131 ± 0.00795078 (W 0.00593892) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 52.0154 ± 0.12834 (W 0.0787014) | 179.73 ± 0.824925 (W 0.444007) | 🔵 faer-fork 3.46× faster than `lapack-aocl` |  |
| `svd_values/256x4096/b1 [api-inclusive]` | 52.1904 ± 0.066503 (W 0.0875279) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.65153 ± 0.017524 (W 0.00462617) | 4.76696 ± 0.0227576 (W 0.0107842) | 🔵 faer-fork 2.89× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.65127 ± 0.013678 (W 0.00383555) | - | - | `*` one row has no measurement here |

## linalg-wide, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 52.975 ± 0.467195 (W 0.616196) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 51.4371 ± 0.320036 (W 0.333909) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.01759 ± 0.00435395 (W 0.00477639) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.08343 ± 0.00812882 (W 0.00445517) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.61502 ± 0.0406246 (W 0.0129421) | 9.81031 ± 0.0302802 (W 0.0154116) | 🔵 faer-fork 1.48× faster than `lapack-aocl` |  |
| `lu/256x4096/b1 [api-inclusive]` | 7.55383 ± 0.0252784 (W 0.014147) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.19424 ± 0.000397501 (W 0.00153785) | 0.264904 ± 0.000432832 (W 0.00195431) | 🔵 faer-fork 1.36× faster than `lapack-aocl` |  |
| `lu/64x1024/b1 [api-inclusive]` | 0.246225 ± 0.000762039 (W 0.00139851) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 13.9541 ± 0.0459582 (W 0.0236057) | 54.579 ± 0.338309 (W 0.428315) | 🔵 faer-fork 3.91× faster than `lapack-aocl` |  |
| `qr/256x4096/b1 [api-inclusive]` | 15.2435 ± 0.0439794 (W 0.0309577) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.432864 ± 0.00178203 (W 0.00170329) | 3.94798 ± 0.0281358 (W 0.0141592) | 🔵 faer-fork 9.12× faster than `lapack-aocl` |  |
| `qr/64x1024/b1 [api-inclusive]` | 0.504288 ± 0.00243683 (W 0.00215165) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.026 ± 0.0314148 (W 0.0292263) | 52.6106 ± 0.255421 (W 0.422085) | 🔵 faer-fork 4.04× faster than `lapack-aocl` |  |
| `qr_factor/64x1024/b1` | 0.391535 ± 0.00227988 (W 0.00177161) | 3.78131 ± 0.0122252 (W 0.00915946) | 🔵 faer-fork 9.66× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1` | 74.0477 ± 0.0690518 (W 0.116532) | 365.155 ± 5.45095 (W 13.4972) | 🔵 faer-fork 4.93× faster than `lapack-aocl` |  |
| `svd_thin/256x4096/b1 [api-inclusive]` | 74.7571 ± 0.0857495 (W 0.181091) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 2.31575 ± 0.00943085 (W 0.00512213) | 10.2576 ± 0.0846478 (W 0.0222335) | 🔵 faer-fork 4.43× faster than `lapack-aocl` |  |
| `svd_thin/64x1024/b1 [api-inclusive]` | 2.33809 ± 0.0224391 (W 0.00755915) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 43.1097 ± 0.10116 (W 0.0540809) | 200.502 ± 12.2252 (W 12.1549) | 🔵 faer-fork 4.65× faster than `lapack-aocl` | `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 43.1501 ± 0.0833953 (W 0.0893209) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.53535 ± 0.00526199 (W 0.00296805) | 4.94196 ± 0.0459689 (W 0.0131489) | 🔵 faer-fork 3.22× faster than `lapack-aocl` |  |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.5381 ± 0.0421958 (W 0.00480399) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, f32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 24.3964 ± 0.0798706 (W 0.114717) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 38.1126 ± 0.287994 (W 0.0815119) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.479896 ± 0.0183292 (W 0.00314242) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.975294 ± 0.00960598 (W 0.00316172) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.20356 ± 0.0319931 (W 0.0108471) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.02741 ± 0.139742 (W 0.0669405) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.181183 ± 0.00087594 (W 0.00120484) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.340169 ± 0.00210661 (W 0.00215643) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 17.0529 ± 0.130696 (W 0.0324667) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 17.1005 ± 0.106397 (W 0.031572) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.840396 ± 0.00593314 (W 0.00665609) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.602133 ± 0.00237829 (W 0.00232185) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.2126 ± 0.116986 (W 0.0305928) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.810169 ± 0.00523174 (W 0.00423236) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 42.0235 ± 0.156421 (W 0.0296274) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 42.1755 ± 0.188917 (W 0.0537226) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.35605 ± 0.00546448 (W 0.00462473) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.35683 ± 0.00429121 (W 0.00290986) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 18.5089 ± 0.110804 (W 0.0182663) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 18.533 ± 0.10011 (W 0.0234181) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.68344 ± 0.00283871 (W 0.00287075) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.684767 ± 0.00238722 (W 0.002468) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 9.14168 ± 0.0391168 (W 0.0106835) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 19.1808 ± 0.116512 (W 0.0818417) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.35356 ± 0.00315487 (W 0.00323677) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.624178 ± 0.00344079 (W 0.00265571) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.99665 ± 0.00499836 (W 0.00750983) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.91436 ± 0.116318 (W 0.0460025) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.0814765 ± 0.000467342 (W 0.000616434) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.176167 ± 0.00549108 (W 0.00126817) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 6.7686 ± 0.0427648 (W 0.0208899) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 6.45167 ± 0.0744559 (W 0.0365523) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.594834 ± 0.00229749 (W 0.00634615) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.361621 ± 0.00392406 (W 0.00511795) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.53431 ± 0.0500838 (W 0.0251932) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.578825 ± 0.00272529 (W 0.00713951) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 18.5044 ± 0.046897 (W 0.0210668) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 18.6293 ± 0.121211 (W 0.0330983) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.01612 ± 0.0139045 (W 0.00195031) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.01628 ± 0.00459384 (W 0.00150076) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 10.9458 ± 0.0252812 (W 0.015247) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 10.9634 ± 0.0434327 (W 0.022738) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.633923 ± 0.00132361 (W 0.00183348) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.632677 ± 0.00122402 (W 0.00199082) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 6.70237 ± 0.0192673 (W 0.0156984) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 15.9744 ± 0.11613 (W 0.0575614) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.362522 ± 0.0027766 (W 0.00363835) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.64217 ± 0.00840707 (W 0.0092113) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.30119 ± 0.00625171 (W 0.00545901) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.27256 ± 0.137478 (W 0.0436102) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/64x1024/b1` | 0.0665695 ± 0.000398283 (W 0.000702424) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.164633 ± 0.00816958 (W 0.00101421) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1` | 5.20719 ± 0.0259966 (W 0.0235964) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.71849 ± 0.0432045 (W 0.0451296) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.55491 ± 0.00332199 (W 0.00454874) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.322426 ± 0.00390637 (W 0.00442573) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 5.03162 ± 0.0283497 (W 0.0147563) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.53499 ± 0.00412309 (W 0.0050533) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 14.8786 ± 0.057702 (W 0.0424454) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 14.907 ± 0.0474083 (W 0.0213026) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.957155 ± 0.00548056 (W 0.00204517) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.966321 ± 0.00363309 (W 0.00172527) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 9.38355 ± 0.0369601 (W 0.0139771) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 9.4045 ± 0.0270638 (W 0.0105262) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.614512 ± 0.00229479 (W 0.00255238) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.614181 ± 0.00159925 (W 0.0014537) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 38.8595 ± 0.207935 (W 0.108563) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 36.3384 ± 0.245047 (W 0.0517147) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.917493 ± 0.00423257 (W 0.00536617) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.894173 ± 0.00400716 (W 0.0038679) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 7.22902 ± 0.0306562 (W 0.0133018) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 7.26663 ± 0.0250495 (W 0.0214372) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.244461 ± 0.00209578 (W 0.00183806) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.250823 ± 0.0015221 (W 0.00141191) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 14.8898 ± 0.0705465 (W 0.0240461) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 14.8591 ± 0.0305015 (W 0.0116811) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.398675 ± 0.000983601 (W 0.00184689) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.483115 ± 0.00275952 (W 0.00269191) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 14.0627 ± 0.0971741 (W 0.0353908) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.372961 ± 0.00583092 (W 0.0023287) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 44.8404 ± 0.701054 (W 0.028946) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 44.9662 ± 0.74242 (W 0.0589403) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.35536 ± 0.00817087 (W 0.00409452) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.35961 ± 0.00366419 (W 0.00274534) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 21.5828 ± 0.642789 (W 0.0282199) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 21.6325 ± 0.610524 (W 0.0441242) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 0.729341 ± 0.00255391 (W 0.00402995) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.728949 ± 0.00351763 (W 0.00439952) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.7834 ± 0.272441 (W 0.0260956) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 17.3991 ± 0.0886623 (W 0.0334681) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.556273 ± 0.00747815 (W 0.00848025) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.542651 ± 0.00249887 (W 0.00469999) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.97582 ± 0.00475548 (W 0.00617841) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 2.20209 ± 0.00408205 (W 0.00444854) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0785035 ± 0.000279697 (W 0.000464953) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.090038 ± 0.000457099 (W 0.000476338) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 4.04891 ± 0.00754741 (W 0.00376427) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 4.36089 ± 0.0211046 (W 0.00680206) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.161178 ± 0.000767056 (W 0.000883173) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.239006 ± 0.00274977 (W 0.00411086) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 3.8022 ± 0.00603698 (W 0.0100881) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.146498 ± 0.000866777 (W 0.00101695) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 21.3046 ± 0.476315 (W 0.0292262) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/256x4096/b1 [api-inclusive]` | 21.3683 ± 0.485319 (W 0.032427) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_thin/64x1024/b1` | 1.00845 ± 0.00563888 (W 0.0030513) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.01506 ± 0.00375139 (W 0.00268176) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 14.2717 ± 0.0603414 (W 0.0225226) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 14.2516 ± 0.0519724 (W 0.0346831) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.680737 ± 0.00505438 (W 0.00521445) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.681111 ± 0.00236061 (W 0.0042486) | - | - | `*` one row has no measurement here |

## linalg-wide, f32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 14.2639 ± 0.0468957 (W 0.0193515) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 14.1992 ± 0.04936 (W 0.0202229) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.566527 ± 0.002629 (W 0.0118254) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.556286 ± 0.00346677 (W 0.00692965) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 1.09682 ± 0.00470629 (W 0.00235458) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 1.38345 ± 0.0079976 (W 0.00423273) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.0509665 ± 0.000451154 (W 0.00030389) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.064116 ± 0.000500962 (W 0.000619048) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 2.2481 ± 0.00662469 (W 0.00351931) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 2.55098 ± 0.0114665 (W 0.00676556) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.123551 ± 0.000482955 (W 0.000313813) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.204109 ± 0.00198605 (W 0.00419041) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 2.08821 ± 0.00436667 (W 0.00502936) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.10957 ± 0.00092108 (W 0.000223157) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 17.5124 ± 0.0613534 (W 0.0309617) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 17.641 ± 0.0576519 (W 0.034075) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 0.956699 ± 0.0061274 (W 0.0022816) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 0.963929 ± 0.0044572 (W 0.00236563) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 12.6637 ± 0.0520742 (W 0.0198307) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 12.7126 ± 0.0457043 (W 0.0233053) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.661967 ± 0.00286587 (W 0.00384812) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.663635 ± 0.0029115 (W 0.00575781) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 58.4869 ± 0.325106 (W 0.110895) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 90.5142 ± 0.468597 (W 0.63984) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.893749 ± 0.00348093 (W 0.00275429) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.49328 ± 0.0132486 (W 0.00380768) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 22.0238 ± 0.0443013 (W 0.0266902) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 26.683 ± 0.133828 (W 0.0802709) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.536064 ± 0.00267755 (W 0.00229453) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.672462 ± 0.00202582 (W 0.00531913) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 55.0841 ± 3.54689 (W 0.0729521) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 58.3611 ± 5.73585 (W 0.150927) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 1.63985 ± 0.0631847 (W 0.00651796) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 1.46561 ± 0.100261 (W 0.00284084) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 51.6507 ± 3.62065 (W 0.04264) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 1.55601 ± 0.00462983 (W 0.00486687) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 135.115 ± 0.256783 (W 0.204141) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 135.325 ± 0.499146 (W 0.269645) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.0369 ± 0.00754707 (W 0.00467497) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.04368 ± 0.0113642 (W 0.00678363) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 54.2924 ± 1.86871 (W 0.0496789) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/256x4096/b1 [api-inclusive]` | 54.3685 ± 1.6862 (W 0.169269) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1` | 1.42177 ± 0.036857 (W 0.00398262) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.42282 ± 0.0406178 (W 0.00344076) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c32, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 17.0187 ± 0.0781102 (W 0.0274953) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 37.1301 ± 0.187114 (W 0.122552) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.492494 ± 0.00198555 (W 0.00359342) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.827576 ± 0.0043018 (W 0.005624) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.24608 ± 0.0239944 (W 0.0146186) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 9.89766 ± 0.0720959 (W 0.0878879) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.196554 ± 0.00289355 (W 0.00186758) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.298086 ± 0.000793127 (W 0.00122226) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 16.8253 ± 0.0682175 (W 0.0329574) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 18.0593 ± 0.185567 (W 0.1235) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.74663 ± 0.00427967 (W 0.00536023) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.570604 ± 0.00246955 (W 0.00256805) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 16.0041 ± 0.0418347 (W 0.032113) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.709544 ± 0.00437101 (W 0.00380327) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 48.3833 ± 0.0517067 (W 0.0830602) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 48.6696 ± 0.0846058 (W 0.0977909) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.65876 ± 0.00358515 (W 0.00417529) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.66492 ± 0.00495248 (W 0.00611295) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 24.2357 ± 0.0652116 (W 0.0673727) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.2598 ± 0.0512227 (W 0.0363247) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.01163 ± 0.00469118 (W 0.00353279) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.01511 ± 0.00446642 (W 0.0037331) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 10.7948 ± 0.0323091 (W 0.016912) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 29.7424 ± 0.102613 (W 0.10093) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.453889 ± 0.00239713 (W 0.00354846) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.765679 ± 0.00505992 (W 0.00211538) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.55036 ± 0.00542806 (W 0.00736669) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.86845 ± 0.0421278 (W 0.033521) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.149647 ± 0.00056768 (W 0.00188703) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.241313 ± 0.000697347 (W 0.00145729) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 10.5604 ± 0.138382 (W 0.0252956) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 11.1077 ± 0.159979 (W 0.032043) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.591236 ± 0.0017432 (W 0.00387759) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.42153 ± 0.000949018 (W 0.00234874) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 10.1428 ± 0.111454 (W 0.028298) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.56621 ± 0.00268303 (W 0.00383891) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 35.2274 ± 0.109582 (W 0.068103) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 35.3777 ± 0.0674098 (W 0.0496005) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.4639 ± 0.00353654 (W 0.00311794) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.46465 ± 0.00666319 (W 0.00574176) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 19.1319 ± 0.0347754 (W 0.0246704) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 19.1883 ± 0.0351786 (W 0.037593) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 0.958152 ± 0.00335981 (W 0.00214808) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 0.953639 ± 0.003321 (W 0.00244365) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 1T (CPU 4)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 96.3949 ± 3.65252 (W 0.499232) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 87.4305 ± 0.713784 (W 0.33341) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 1.40726 ± 0.00949969 (W 0.00577652) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 1.38287 ± 0.00419392 (W 0.00527976) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 22.7517 ± 0.114415 (W 0.0359936) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 23.2179 ± 0.0918969 (W 0.0525774) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.550618 ± 0.00153447 (W 0.00187222) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.567128 ± 0.00264981 (W 0.00245081) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 53.5001 ± 5.74621 (W 0.109456) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/256x4096/b1 [api-inclusive]` | 54.5665 ± 5.46129 (W 0.256263) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1` | 1.33386 ± 0.0596615 (W 0.00274556) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/64x1024/b1 [api-inclusive]` | 1.35518 ± 0.0560541 (W 0.00500255) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/256x4096/b1` | 50.4305 ± 5.64618 (W 0.0886165) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/64x1024/b1` | 1.25009 ± 0.00796143 (W 0.00393639) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 140.479 ± 0.375241 (W 0.147763) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 140.498 ± 0.213824 (W 0.32063) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 3.09314 ± 0.0110998 (W 0.00425243) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 3.09863 ± 0.0161661 (W 0.00329169) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 59.8343 ± 0.128212 (W 0.0677522) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 59.9107 ± 0.129822 (W 0.0600165) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.50701 ± 0.0433983 (W 0.00560311) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.50692 ± 0.0483275 (W 0.00519827) | - | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-wide, c32, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 35.5148 ± 0.229613 (W 0.0711435) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 33.9148 ± 0.0427985 (W 0.0783998) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.70178 ± 0.0038115 (W 0.00275618) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.716323 ± 0.00431204 (W 0.00312355) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 6.0503 ± 0.00969821 (W 0.00791982) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 6.49831 ± 0.00524063 (W 0.011793) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.166692 ± 0.000931091 (W 0.00157901) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.189072 ± 0.000899255 (W 0.0017757) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 13.8753 ± 0.0310832 (W 0.0209878) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 14.4404 ± 0.068151 (W 0.0243977) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.425907 ± 0.0011689 (W 0.00173577) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.460003 ± 0.00137113 (W 0.00225482) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 13.0451 ± 0.0392159 (W 0.0165358) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.390846 ± 0.00296767 (W 0.00202209) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 53.5954 ± 0.084606 (W 0.0800167) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 53.8922 ± 0.0773011 (W 0.0894685) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.7005 ± 0.00576781 (W 0.00567109) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.71384 ± 0.00617838 (W 0.00416046) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 29.762 ± 0.0572856 (W 0.0332056) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 29.757 ± 0.0898669 (W 0.0334377) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.07833 ± 0.00633622 (W 0.00553862) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.08512 ± 0.0106429 (W 0.00298325) | - | - | `*` one row has no measurement here |

## linalg-wide, c32, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | lapack-aocl (ms) | vs `lapack-aocl` | notes |
|---|---|---|---|---|
| `full_piv_lu/256x4096/b1` | 27.0231 ± 0.235069 (W 0.0590178) | - | - | `*` one row has no measurement here |
| `full_piv_lu/256x4096/b1 [api-inclusive]` | 26.8876 ± 0.0690694 (W 0.0624998) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1` | 0.611733 ± 0.00487883 (W 0.00222244) | - | - | `*` one row has no measurement here |
| `full_piv_lu/64x1024/b1 [api-inclusive]` | 0.642409 ± 0.002817 (W 0.00265711) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1` | 3.31334 ± 0.0151567 (W 0.00519571) | - | - | `*` one row has no measurement here |
| `lu/256x4096/b1 [api-inclusive]` | 3.69218 ± 0.00530005 (W 0.0105458) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1` | 0.103889 ± 0.000848103 (W 0.000707624) | - | - | `*` one row has no measurement here |
| `lu/64x1024/b1 [api-inclusive]` | 0.130555 ± 0.00145833 (W 0.000813953) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1` | 7.26711 ± 0.0123911 (W 0.0129516) | - | - | `*` one row has no measurement here |
| `qr/256x4096/b1 [api-inclusive]` | 7.80213 ± 0.0246003 (W 0.0176693) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1` | 0.272623 ± 0.00284091 (W 0.00168556) | - | - | `*` one row has no measurement here |
| `qr/64x1024/b1 [api-inclusive]` | 0.311105 ± 0.00135109 (W 0.001393) | - | - | `*` one row has no measurement here |
| `qr_factor/256x4096/b1` | 6.81812 ± 0.0162178 (W 0.0161318) | - | - | `*` one row has no measurement here |
| `qr_factor/64x1024/b1` | 0.24737 ± 0.00134554 (W 0.0015167) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1` | 40.5154 ± 0.0745954 (W 0.104831) | - | - | `*` one row has no measurement here |
| `svd_thin/256x4096/b1 [api-inclusive]` | 40.772 ± 0.0659612 (W 0.253022) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1` | 1.51138 ± 0.0125058 (W 0.00952263) | - | - | `*` one row has no measurement here |
| `svd_thin/64x1024/b1 [api-inclusive]` | 1.5146 ± 0.010694 (W 0.004039) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1` | 24.7622 ± 0.0337232 (W 0.0311726) | - | - | `*` one row has no measurement here |
| `svd_values/256x4096/b1 [api-inclusive]` | 24.7551 ± 0.0367233 (W 0.0346207) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1` | 1.0243 ± 0.00748758 (W 0.00422048) | - | - | `*` one row has no measurement here |
| `svd_values/64x1024/b1 [api-inclusive]` | 1.02965 ± 0.00588657 (W 0.00392461) | - | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/256x4096`, `cholesky/64x1024`, `eigh/256x4096`, `eigh/64x1024`, `full_piv_lu/256x4096`, `full_piv_lu/64x1024`, `lu/256x4096`, `lu/64x1024`, `qr/256x4096`, `qr/64x1024`, `qr_factor/256x4096`, `qr_factor/64x1024`, `solve/256x4096`, `solve/64x1024`, `svd_thin/256x4096`, `svd_thin/64x1024`, `svd_values/256x4096`, `svd_values/64x1024`, `triangular_solve/256x4096`, `triangular_solve/64x1024`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
