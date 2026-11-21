# `faer-kernels-large` on `zen5-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above); `faer-upstream` (the provider in the measured revision; see library and upstream above)
- harness commit: `e7a1baa48af15fb7767986acb7501818796ec8ec`
- toolkit commit: `ec1366c391cbd5023116ddcb6fdadc9c5b76f582`
- measured binary SHA-256: `4cee43d7741bfd9ee3cf9257e89f43f4ac0924e5d57a99d2078a7befcb09ec33`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-20T23:14:36.522188Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 3 attempts
- command: `scripts/record_run.py zen5-cpu faer-kernels-large --vendor mkl --engines faer-fork,faer-upstream,lapack-mkl --threads 4,8 --regimes linalg-large,gemm-large,linalg-boundary --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 --reps 5 --aa 3 --prime-ms 250 --label pub-93da6217-20261120-large-mkl`
- raw data: `data/results/zen5-cpu/faer-kernels-large/20261120T153124Z-pub-93da6217-20261120-large-mkl/`

## What was measured

- **Workload** — Direct calls into one faer revision: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: Vendor C-order cells use CBLAS/LAPACKE row-major entry points. Workspace allocation or layout conversion internal to those APIs remains timed; these are not prepared-column-major or transpose measurements.
  Known blind spot: `gemm` covers the tall class's two Householder legs (a 16-wide basis and a 240-column target over 4096 rows: the reduction and the write-back) beside a vendor `?gemm` on the same leg. A single leg beside a vendor gemm is a diagnostic reference, not the equivalent implementation of the whole update, and the row is labelled accordingly.
  Known blind spot: `householder_apply` measures block apply and the one-block sequence form as two distinct rows, labelled with basis width min(m,n) and target width n. QR fixture preparation is outside timing. Independently varied basis/target widths and multi-panel sequences are not covered.
  Known blind spot: Allocation performed inside the measured call stays inside the clock. The fork's two row splits allocate their partial buffers inside the public call, and the public scratch function does not expose them; the campaign reports that as it is instead of changing faer's API to make the boundary look cleaner.
  Known blind spot: `cholesky`, `eigh`, `solve` and `triangular_solve` are measured on square shapes only: the matrix they factor or solve with is square by definition (self-adjoint for the first two, a triangular factor for the last). `lu` and `full_piv_lu` are not gated that way -- they take the declared rectangular shapes too, with `min(m, n)` pivots and a trapezoidal `L` or `U`, which is what `dgetrf` does as well -- so a rectangular LU cell is measured in both orders beside its square one.
  Known blind spot: BLIS, sparse, iterative and GPU arms are later declarations, not machinery built now.
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

- CPU: `AMD Ryzen AI 9 HX 470`
- logical CPUs: `24`
- L3: `24 MiB (2 instances)`
- L3 domains: `0-3: 16 MiB shared; 4-11: 8 MiB shared`
- OS / arch: `Linux 7.0.0-43-generic` / `x86_64`
- hostname: `f990be0ea5d6`
- CPU sets: 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 3 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 0.67%, worst 26.12% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 69 of 1968 engine cells exceed 5% and carry `!`; median within-process relative SD 0.32%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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

## linalg-large, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 5.25934 ± 0.0254594 (W 0.0110886) | 7.47533 ± 0.113003 (W 0.131015) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 5.46399 ± 0.0234423 (W 0.00777582) | 7.8417 ± 0.149466 (W 0.110415) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 14.8868 ± 0.0759809 (W 0.0126068) | 18.7524 ± 0.273807 (W 0.409661) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 16.4719 ± 0.173041 (W 0.175032) | 24.3555 ± 0.18147 (W 0.231193) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 119.456 ± 0.237419 (W 0.0946574) | 116.909 ± 0.202187 (W 0.121745) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 126.529 ± 0.158576 (W 0.211612) | 118.051 ± 0.254267 (W 0.814225) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 326.756 ± 1.05712 (W 0.343693) | 325.091 ± 1.81642 (W 0.320945) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 341.254 ± 0.531715 (W 0.714296) | 340.814 ± 0.745767 (W 2.21855) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 58.9095 ± 0.0932193 (W 0.0922011) | 75.7525 ± 0.411262 (W 0.0502699) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 66.6539 ± 0.205324 (W 0.371108) | 68.9832 ± 0.679216 (W 0.361215) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 246.279 ± 0.279192 (W 1.47352) | 288.988 ± 1.50247 (W 1.1256) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 264.891 ± 0.539961 (W 0.783148) | 277.424 ± 2.1965 (W 5.68276) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 10.9386 ± 0.0079968 (W 0.0408044) | 57.2399 ± 0.790174 (W 1.57153) | - | 🔵 faer-fork 5.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 18.9741 ± 0.0299387 (W 0.0337927) | 21.5775 ± 0.296634 (W 0.228033) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 31.7919 ± 0.182776 (W 0.28775) | 144.408 ± 1.42038 (W 1.11062) | - | 🔵 faer-fork 4.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 49.5545 ± 0.246648 (W 0.107031) | 63.6256 ± 0.123994 (W 0.259331) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 39.6103 ± 0.0327583 (W 0.0472977) | 94.4484 ± 0.118954 (W 0.217568) | - | 🔵 faer-fork 2.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 51.0535 ± 0.378678 (W 0.0931668) | 56.7346 ± 0.503273 (W 0.137787) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 131.006 ± 0.327084 (W 0.601781) | 257.647 ± 0.408694 (W 1.00213) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 154.092 ± 0.417608 (W 0.302674) | 174.338 ± 3.94984 (W 4.49384) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 19.3421 ± 0.086468 (W 0.0392346) | 74.1681 ± 0.155125 (W 0.244767) | - | 🔵 faer-fork 3.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 59.0225 ± 0.250527 (W 0.160672) | 185.169 ± 0.685568 (W 0.306706) | - | 🔵 faer-fork 3.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 10.6979 ± 0.0475739 (W 0.0384606) | 40.4268 ± 0.217504 (W 0.932734) | - | 🔵 faer-fork 3.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 18.6557 ± 0.0460806 (W 0.0385331) | 20.4506 ± 0.237658 (W 0.0951726) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 31.2754 ± 0.427334 (W 0.173304) | 103.415 ± 1.39281 (W 1.7935) | - | 🔵 faer-fork 3.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 34.393 ± 0.184557 (W 0.228862) | 62.4776 ± 0.365869 (W 0.295275) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 204.334 ± 0.309641 (W 0.257989) | 273.058 ± 0.779938 (W 0.381412) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 225.686 ± 0.703401 (W 0.372147) | 293.895 ± 0.986497 (W 2.04734) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 701.281 ± 1.4868 (W 0.803541) | 878.061 ± 2.31251 (W 1.0087) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 746.184 ± 1.83211 (W 1.49024) | 924.215 ± 0.576145 (W 5.36827) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 105.281 ± 1.16778 (W 0.255224) | 189.022 ± 0.200098 (W 0.746457) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 105.089 ± 1.08379 (W 0.497653) | 187.612 ± 0.654738 (W 0.363062) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 405.528 ± 2.49342 (W 1.36289) | 611.681 ± 1.3185 (W 1.13536) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 417.129 ± 2.4718 (W 1.10496) | 625.064 ± 2.78875 (W 1.81672) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.084738 ± 0.00117702 (W 0.000700744) | 0.088129 ± 0.00039753 (W 0.00168109) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.235001 ± 0.00446412 (W 0.00435371) | 0.25753 ± 0.00298911 (W 0.00423929) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 3.15794 ± 0.00428293 (W 0.00881836) | 5.57126 ± 0.00737954 (W 0.095832) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 4.03186 ± 0.0832727 (W 0.00913754) | 6.35027 ± 0.0408705 (W 0.0563439) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 8.23381 ± 0.034738 (W 0.0095606) | 12.0761 ± 0.144473 (W 0.154516) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 11.0983 ± 0.0497146 (W 0.174498) | 18.4525 ± 0.035543 (W 0.0816923) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 73.9363 ± 0.230351 (W 0.27798) | 97.9068 ± 0.185316 (W 0.279436) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 80.429 ± 0.131049 (W 0.256237) | 97.4272 ± 0.761108 (W 1.05545) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 191.579 ± 0.205107 (W 0.199365) | 252.391 ± 0.42132 (W 1.73801) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 205.377 ± 0.312695 (W 0.42787) | 268.628 ± 1.18728 (W 3.20933) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 45.535 ± 0.115805 (W 0.394669) | 61.3182 ± 0.41912 (W 0.0637275) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 54.4223 ± 0.069563 (W 0.0741988) | 57.3311 ± 0.384263 (W 0.104786) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 146.972 ± 1.57597 (W 0.922036) | 181.53 ± 1.09883 (W 0.715212) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 168.247 ± 0.210717 (W 0.635099) | 177.303 ± 1.51114 (W 0.531322) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 6.83729 ± 0.0170703 (W 0.0359007) | 44.0232 ± 0.961898 (W 0.987912) | - | 🔵 faer-fork 6.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 15.8397 ± 0.16776 (W 0.064988) | 18.3459 ± 0.594451 (W 0.118037) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1536/b1` | 20.0893 ± 0.0213779 (W 0.107474) | 111.18 ± 0.950294 (W 1.74187) | - | 🔵 faer-fork 5.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 39.9475 ± 0.198902 (W 0.257105) | 53.723 ± 0.461971 (W 0.19124) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 21.5589 ± 0.150502 (W 0.168776) | 86.2232 ± 0.185837 (W 0.534227) | - | 🔵 faer-fork 4.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 33.046 ± 0.172903 (W 0.234227) | 43.5132 ± 5.64877 (W 3.05464) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1536/b1` | 70.486 ± 0.224963 (W 0.266075) | 227.541 ± 0.473485 (W 1.32669) | - | 🔵 faer-fork 3.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 98.2673 ± 0.106022 (W 0.168237) | 140.876 ± 4.08249 (W 4.95089) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n1024/b1` | 10.5861 ± 0.0203688 (W 0.0440762) | 71.4508 ± 0.356181 (W 0.15995) | - | 🔵 faer-fork 6.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 32.1698 ± 0.0837897 (W 0.167801) | 174.262 ± 0.639793 (W 0.863946) | - | 🔵 faer-fork 5.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 6.75852 ± 0.0230318 (W 0.0218191) | 33.8483 ± 0.141242 (W 0.0649355) | - | 🔵 faer-fork 5.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 9.03684 ± 0.0660259 (W 0.0950694) | 18.0185 ± 0.225424 (W 0.0764365) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 19.6335 ± 0.0892002 (W 0.0708001) | 86.7542 ± 1.0595 (W 0.943961) | - | 🔵 faer-fork 4.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 24.4931 ± 0.135163 (W 0.281468) | 52.3694 ± 0.109086 (W 0.175682) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 145.565 ± 0.204113 (W 0.422618) | 228.081 ± 0.0791273 (W 0.625224) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 166.005 ± 0.480325 (W 0.407852) | 248.167 ± 0.563951 (W 2.79542) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 445.001 ± 0.714993 (W 0.592372) | 648.735 ± 2.10665 (W 4.70524) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 488.189 ± 0.657007 (W 1.23007) | 682.457 ± 5.87511 (W 7.9603) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 78.2289 ± 0.745065 (W 0.351318) | 162.434 ± 0.800698 (W 0.402232) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 77.9345 ± 0.693378 (W 0.414345) | 162.376 ± 0.67671 (W 0.648468) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 259.425 ± 1.91595 (W 1.19202) | 452.23 ± 1.87991 (W 1.36648) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 267.672 ± 0.44189 (W 1.74003) | 465.055 ± 0.58604 (W 1.82336) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.072229 ± 0.000416048 (W 0.000220303) | 0.075759 ± 0.000984315 (W 0.00165598) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.15752 ± 0.00394026 (W 0.00151048) | 0.144038 ± 0.0193313 (W 0.00120908) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 4.0606 ± 0.01361 (W 0.00469614) | 4.65116 ± 0.0280931 (W 0.072934) | 6.04776 ± 0.0262096 (W 0.0490538) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 4.99625 ± 0.0205654 (W 0.00998542) | 5.58265 ± 0.0148286 (W 0.101761) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 12.6249 ± 0.0488586 (W 0.00959445) | 13.3516 ± 0.127683 (W 0.158778) | 16.0227 ± 0.0458627 (W 0.0800001) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 14.8422 ± 0.0515761 (W 0.0302727) | 16.0568 ± 0.0156023 (W 0.207653) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 119.188 ± 0.282529 (W 0.13568) | 114.959 ± 0.0850763 (W 0.1975) | 120.451 ± 0.120619 (W 0.233832) | · | · |  |
| `eigh/n1024/b1 [api-inclusive]` | 126.271 ± 0.397402 (W 0.288774) | 114.356 ± 0.658611 (W 0.975571) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 325.073 ± 0.741596 (W 0.526448) | 317.649 ± 0.422038 (W 0.967918) | 361.159 ± 1.61712 (W 0.559798) | · | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 340.263 ± 0.390605 (W 0.457544) | 333.317 ± 0.909391 (W 3.03304) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 61.6611 ± 0.264956 (W 0.364116) | 64.6968 ± 0.342637 (W 0.0488827) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 64.2748 ± 0.215795 (W 0.0610313) | 60.833 ± 0.379433 (W 0.318446) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 260.972 ± 0.561835 (W 1.48868) | 264.377 ± 0.638761 (W 0.899641) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 260.248 ± 0.585291 (W 1.24181) | 254.259 ± 3.80352 (W 5.89428) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 8.85703 ± 0.0215145 (W 0.0103639) | 13.4538 ± 0.0861421 (W 0.182335) | 8.8501 ± 0.029281 (W 0.0236733) | 🔵 faer-fork 1.52× faster than `faer-upstream` | · |  |
| `lu/n1024/b1 [api-inclusive]` | 16.3678 ± 0.0363485 (W 0.113226) | 14.3273 ± 0.178775 (W 0.129648) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 26.5215 ± 0.0565692 (W 0.0400849) | 35.4779 ± 0.147111 (W 0.434851) | 27.9224 ± 0.0480007 (W 0.0702905) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `lu/n1536/b1 [api-inclusive]` | 43.5664 ± 0.337638 (W 0.110959) | 38.7712 ± 0.192428 (W 0.228916) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 37.1801 ± 0.0696504 (W 0.0337396) | 40.7331 ± 0.0315061 (W 0.0407049) | 52.3323 ± 0.102256 (W 0.0683698) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.41× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 48.4564 ± 0.248028 (W 0.132065) | 52.1373 ± 1.86632 (W 0.0487983) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1536/b1` | 119.832 ± 0.178311 (W 0.0654548) | 129.663 ± 0.156843 (W 0.319988) | 162.619 ± 0.656135 (W 0.549743) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `qr/n1536/b1 [api-inclusive]` | 147.188 ± 0.989354 (W 0.31683) | 151.129 ± 5.66794 (W 3.53118) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n1024/b1` | 17.1986 ± 0.0659219 (W 0.0216989) | 20.8482 ± 0.0179034 (W 0.0364202) | 18.6833 ± 0.121844 (W 0.0113841) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 52.9529 ± 0.194047 (W 0.0640415) | 62.51 ± 0.203189 (W 0.0763603) | 62.0953 ± 0.420715 (W 0.265545) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 8.53457 ± 0.0278094 (W 0.00785954) | 11.9172 ± 0.064895 (W 0.133618) | 8.77647 ± 0.0246304 (W 0.0840097) | 🔵 faer-fork 1.40× faster than `faer-upstream` | · |  |
| `solve/n1024/b1 [api-inclusive]` | 16.0819 ± 0.0200879 (W 0.0482985) | 13.7163 ± 0.0298621 (W 0.079194) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 25.8144 ± 0.0895637 (W 0.0223548) | 32.898 ± 0.16482 (W 0.24237) | 27.5499 ± 0.042157 (W 0.0313546) | 🔵 faer-fork 1.27× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `solve/n1536/b1 [api-inclusive]` | 29.3551 ± 0.0240052 (W 0.0297011) | 37.2119 ± 0.15648 (W 0.19339) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 198.04 ± 0.354947 (W 0.301901) | 265.569 ± 0.74293 (W 0.178661) | 282.637 ± 0.409096 (W 0.537425) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 219.377 ± 0.768682 (W 0.278325) | 287.76 ± 0.743378 (W 2.47015) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 678.041 ± 1.49242 (W 0.616654) | 856.809 ± 1.36036 (W 0.811269) | 847.906 ± 3.08332 (W 1.69792) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 724.659 ± 0.826619 (W 1.55986) | 903.584 ± 1.18703 (W 5.12209) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 96.6501 ± 0.243604 (W 0.072244) | 180.642 ± 0.243008 (W 0.162529) | 123.091 ± 1.33894 (W 0.38977) | 🔵 faer-fork 1.87× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 97.2536 ± 0.398657 (W 0.18839) | 180.085 ± 0.804252 (W 0.170681) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 380.55 ± 0.737719 (W 1.08066) | 586.444 ± 2.55749 (W 1.54654) | 393.391 ± 3.09257 (W 1.58749) | 🔵 faer-fork 1.54× faster than `faer-upstream` | · |  |
| `svd_values/n1536/b1 [api-inclusive]` | 390.054 ± 0.660283 (W 1.0205) | 599.493 ± 1.78876 (W 1.54963) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.10678 ± 0.000960047 (W 0.00078316) | 0.100507 ± 0.000322641 (W 0.00189204) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.25605 ± 0.00102371 (W 0.00517883) | 0.27339 ± 0.00333466 (W 0.00513806) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 2.4375 ± 0.0111271 (W 0.00724556) | 3.15187 ± 0.0468747 (W 0.0458602) | 3.49323 ± 0.011812 (W 0.0443487) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 3.45968 ± 0.0285459 (W 0.00898378) | 4.1107 ± 0.015421 (W 0.0271209) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 6.99572 ± 0.0175229 (W 0.0115948) | 8.06154 ± 0.0370716 (W 0.0434862) | 8.79764 ± 0.00277743 (W 0.0649875) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 9.27184 ± 0.0455872 (W 0.0171206) | 10.6029 ± 0.0602403 (W 0.0839249) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 73.381 ± 0.0772056 (W 0.0510908) | 96.0705 ± 0.572131 (W 0.294406) | 82.3879 ± 0.223992 (W 0.436184) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 80.0551 ± 0.0267945 (W 0.189937) | 95.9647 ± 0.261316 (W 0.472164) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 190.298 ± 0.538559 (W 0.40674) | 245.986 ± 1.12266 (W 0.923015) | 224.741 ± 0.630869 (W 0.805477) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 203.842 ± 0.487236 (W 0.255467) | 260.768 ± 0.59292 (W 2.5699) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 46.6104 ± 0.147886 (W 0.466857) | 50.6463 ± 0.119961 (W 0.053111) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 51.8611 ± 0.0288326 (W 0.0545698) | 49.1747 ± 0.103601 (W 0.0451327) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 149.367 ± 0.669636 (W 0.738079) | 155.663 ± 0.66458 (W 0.878706) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 161.77 ± 0.904044 (W 0.884068) | 151.86 ± 1.40012 (W 0.850743) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 5.75854 ± 0.00271171 (W 0.0104006) | 10.3725 ± 0.0360022 (W 0.0889991) | 5.26649 ± 0.00151819 (W 0.0151971) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` |  |
| `lu/n1024/b1 [api-inclusive]` | 13.3704 ± 0.0874157 (W 0.0380948) | 11.5188 ± 0.0528275 (W 0.0539431) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 16.2528 ± 0.10473 (W 0.0367614) | 25.2412 ± 0.117366 (W 0.146756) | 16.5818 ± 0.0380212 (W 0.0896022) | 🔵 faer-fork 1.55× faster than `faer-upstream` | · |  |
| `lu/n1536/b1 [api-inclusive]` | 33.8054 ± 0.252029 (W 0.239097) | 28.7151 ± 0.0504313 (W 0.139837) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 19.707 ± 0.0583907 (W 0.0213257) | 34.1878 ± 0.229035 (W 0.348195) | 31.0344 ± 0.0450597 (W 0.0266802) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 30.5826 ± 0.0654625 (W 0.0350462) | 36.1294 ± 1.46078 (W 1.73564) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1536/b1` | 63.7077 ± 0.124353 (W 0.0852926) | 101.209 ± 0.866364 (W 1.80278) | 84.6489 ± 0.347282 (W 0.591847) | 🔵 faer-fork 1.59× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `qr/n1536/b1 [api-inclusive]` | 91.9729 ± 0.392876 (W 0.130193) | 118.526 ± 8.05133 (W 6.48702) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n1024/b1` | 9.05578 ± 0.0176027 (W 0.00946899) | 19.2486 ± 0.0330496 (W 0.251929) | 12.9602 ± 0.0485255 (W 0.0227571) | 🔵 faer-fork 2.13× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 28.1368 ± 0.0570527 (W 0.0454213) | 54.6628 ± 0.849812 (W 1.01398) | 34.5597 ± 0.420552 (W 0.372183) | 🔵 faer-fork 1.94× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 5.56283 ± 0.00889653 (W 0.0113832) | 9.26367 ± 0.0575696 (W 0.116343) | 5.24516 ± 0.0254334 (W 0.0153829) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `solve/n1024/b1 [api-inclusive]` | 6.86068 ± 0.0419244 (W 0.0115632) | 10.9486 ± 0.130743 (W 0.124901) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 15.8949 ± 0.0453497 (W 0.0306943) | 23.2392 ± 0.203325 (W 0.173828) | 16.543 ± 0.0602054 (W 0.0646527) | 🔵 faer-fork 1.46× faster than `faer-upstream` | · |  |
| `solve/n1536/b1 [api-inclusive]` | 19.0332 ± 0.058417 (W 0.0367137) | 27.4987 ± 0.109345 (W 0.101937) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 138.893 ± 0.0488121 (W 0.432386) | 221.337 ± 0.727074 (W 0.470256) | 219.863 ± 0.0574763 (W 0.489249) | 🔵 faer-fork 1.59× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 159.164 ± 0.0385361 (W 0.38129) | 242.034 ± 0.477564 (W 2.33176) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 419.696 ± 0.807287 (W 1.17611) | 623.14 ± 0.881067 (W 1.64242) | 596.006 ± 3.25806 (W 2.09935) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 466.548 ± 0.987577 (W 2.61292) | 663.578 ± 1.63455 (W 5.51884) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 69.8394 ± 0.126759 (W 0.0623569) | 155.001 ± 0.476802 (W 0.184962) | 92.0642 ± 1.2422 (W 0.236201) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 69.7952 ± 0.156942 (W 0.0362363) | 154.579 ± 0.164902 (W 0.571297) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 234.006 ± 2.256 (W 2.40358) | 429.266 ± 0.849844 (W 1.07555) | 272.668 ± 0.819398 (W 1.39627) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 244.219 ± 1.58574 (W 0.901008) | 441.003 ± 1.44111 (W 0.972965) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.090858 ± 0.000435203 (W 0.000469026) | 0.08779 ± 0.00171243 (W 0.00103138) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.183145 ± 0.00680924 (W 0.00685738) | 0.192696 ± 0.00215977 (W 0.00534152) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 14.9246 ± 0.0616467 (W 0.0193657) | 21.242 ± 0.24028 (W 0.483754) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 17.988 ± 0.179909 (W 0.0596342) | 20.5011 ± 0.388239 (W 0.441256) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 46.9598 ± 0.189772 (W 0.0234744) | 56.7677 ± 0.373996 (W 0.496043) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 69.8862 ± 0.424182 (W 0.206654) | 75.1333 ± 0.373426 (W 0.445554) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 235.438 ± 0.500661 (W 0.0909964) | 241.036 ± 0.309249 (W 0.346849) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 247.127 ± 0.11771 (W 0.553345) | 252.281 ± 0.427455 (W 2.14437) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 783.881 ± 1.95789 (W 1.71219) | 788.668 ± 1.9232 (W 1.41951) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 821.918 ± 0.64997 (W 1.06821) | 827.307 ± 1.73795 (W 1.44434) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 136.152 ± 0.825411 (W 0.363794) | 159.56 ± 0.525167 (W 0.492618) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 154.34 ± 0.639132 (W 1.18145) | 154.297 ± 1.3776 (W 1.83226) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 680.795 ± 0.435119 (W 3.04338) | 696.714 ± 17.1431 (W 2.00881) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 685.437 ± 3.23554 (W 2.63992) | 696.798 ± 2.37394 (W 3.26006) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 33.4533 ± 0.119996 (W 0.149509) | 84.0435 ± 0.682461 (W 1.28825) | - | 🔵 faer-fork 2.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 50.8583 ± 0.240728 (W 0.214883) | 44.4656 ± 0.439861 (W 0.33752) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 107.24 ± 0.189962 (W 0.292873) | 210.227 ± 0.658515 (W 1.63401) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 138.253 ± 0.259624 (W 0.575748) | 158.3 ± 1.62886 (W 0.76375) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 142.649 ± 0.18756 (W 0.334875) | 196.393 ± 0.146598 (W 0.457971) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 165.424 ± 0.51913 (W 0.352656) | 173.307 ± 3.26054 (W 0.51633) | - | · | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 466.691 ± 0.892428 (W 0.488968) | 571.898 ± 0.125765 (W 0.645869) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 510.495 ± 1.32227 (W 1.04001) | 532.042 ± 0.791225 (W 1.64113) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 63.8221 ± 0.300839 (W 0.147853) | 117.721 ± 0.529952 (W 0.168402) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 206.444 ± 0.279584 (W 0.465718) | 311.839 ± 0.274569 (W 0.469571) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 33.6394 ± 0.127638 (W 0.150341) | 65.7371 ± 1.4332 (W 0.991818) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 37.7306 ± 0.0249848 (W 0.112618) | 44.2915 ± 0.260231 (W 0.48867) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 107.918 ± 0.326453 (W 0.640819) | 178.061 ± 0.9406 (W 3.05776) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 138.276 ± 0.571278 (W 0.303353) | 158.365 ± 0.77345 (W 0.512254) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 508.363 ± 0.820619 (W 0.535038) | 570.765 ± 2.23455 (W 1.18312) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 550.726 ± 0.370832 (W 1.19968) | 613.802 ± 2.76264 (W 1.97462) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1825.28 ± 3.02366 (W 1.14731) | 2014.82 ± 2.16022 (W 3.91467) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1900.96 ± 5.74829 (W 1.73614) | 2100.38 ± 2.4586 (W 4.90298) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 286 ± 0.430017 (W 0.584172) | 368.688 ± 1.15726 (W 1.42011) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 286.286 ± 0.538093 (W 3.35849) | 364.733 ± 0.841349 (W 3.41657) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1135.98 ± 1.76941 (W 1.8992) | 1350.61 ± 5.26202 (W 2.61006) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 1154.62 ± 1.81991 (W 1.91824) | 1378.59 ± 6.49007 (W 2.32352) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.191948 ± 0.00520767 (W 0.00331582) | 0.205959 ± 0.00388697 (W 0.00318107) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n1536/b1` | 0.537122 ± 0.00949475 (W 0.00454114) | 0.690659 ± 0.0076293 (W 0.00440947) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.39075 ± 0.017125 (W 0.0126603) | 12.8988 ± 0.15941 (W 0.351283) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 12.0847 ± 0.302444 (W 0.0700042) | 14.1965 ± 0.122541 (W 0.245843) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 25.1005 ± 0.0311617 (W 0.242478) | 32.0796 ± 0.0487272 (W 0.541859) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 48.4957 ± 0.375981 (W 0.218456) | 53.4788 ± 0.428364 (W 0.271125) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 137.734 ± 0.260553 (W 0.176049) | 180.87 ± 0.85961 (W 1.24894) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 148.108 ± 0.316712 (W 0.374743) | 191.912 ± 0.429938 (W 1.00379) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 426.195 ± 0.195134 (W 1.61829) | 528.304 ± 3.70091 (W 3.9619) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 463.793 ± 0.369075 (W 1.21592) | 562.451 ± 3.89668 (W 4.17483) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 77.1453 ± 0.268011 (W 0.378274) | 98.9028 ± 0.140473 (W 0.268691) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 94.9917 ± 0.675345 (W 1.16261) | 96.4023 ± 0.635435 (W 0.755682) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 487.174 ± 1.94968 (W 4.53481) | 494.944 ± 5.84118 (W 2.81701) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 516.246 ± 6.94384 (W 4.68151) | 530.693 ± 10.0876 (W 4.86395) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 20.9261 ± 0.0537812 (W 0.0451062) | 58.4577 ± 0.293963 (W 1.09815) | - | 🔵 faer-fork 2.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 39.1092 ± 0.169675 (W 0.154986) | 32.3294 ± 0.659265 (W 0.159795) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 64.3829 ± 0.0650838 (W 0.177262) | 141.181 ± 0.415003 (W 2.81265) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 98.6496 ± 0.285084 (W 0.370741) | 117.853 ± 0.948161 (W 0.487777) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 78.326 ± 0.137427 (W 0.254469) | 155.079 ± 1.59477 (W 2.22017) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 100.417 ± 0.20532 (W 0.177044) | 130.484 ± 4.31428 (W 4.78467) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1536/b1` | 255.669 ± 0.347151 (W 0.697003) | 436.585 ± 2.87116 (W 8.11119) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 299.695 ± 0.612273 (W 0.503435) | 401.403 ± 1.34051 (W 4.14987) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 34.9981 ± 0.0516486 (W 0.0461634) | 99.4757 ± 0.0937372 (W 0.997616) | - | 🔵 faer-fork 2.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 114.827 ± 0.226857 (W 0.14928) | 249.46 ± 2.79449 (W 2.60681) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 21.2547 ± 0.0647776 (W 0.0384219) | 48.6301 ± 0.0129781 (W 0.287594) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 25.4762 ± 0.193982 (W 0.0519204) | 31.6598 ± 0.348657 (W 0.322096) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 65.1553 ± 0.103113 (W 0.362543) | 123.423 ± 0.224902 (W 1.03117) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 97.9482 ± 0.541421 (W 0.43164) | 116.674 ± 0.635647 (W 0.274164) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 302.81 ± 1.18826 (W 0.725702) | 397.025 ± 1.82767 (W 1.65766) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 346.739 ± 1.03465 (W 0.884491) | 441.624 ± 3.45289 (W 7.78615) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1212.2 ± 3.24292 (W 2.50087) | 1443.72 ± 1.88534 (W 8.86704) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1291.37 ± 2.26449 (W 2.90101) | 1536.64 ± 4.82368 (W 6.51428) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 171.907 ± 0.935377 (W 0.979017) | 248.379 ± 0.76343 (W 2.12838) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 172.127 ± 1.1435 (W 0.560692) | 246.677 ± 2.5061 (W 3.91471) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 814.344 ± 1.9851 (W 2.88568) | 984.927 ± 1.95197 (W 2.04305) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 830.778 ± 2.29366 (W 1.40307) | 999.748 ± 5.22411 (W 4.16294) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.131117 ± 0.000433231 (W 0.00153882) | 0.145775 ± 0.000812222 (W 0.00195358) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.380128 ± 0.00252113 (W 0.00661013) | 0.555113 ± 0.0070391 (W 0.00517224) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 14.6935 ± 0.0182783 (W 0.0140558) | 15.7622 ± 0.113174 (W 0.272583) | 19.7398 ± 0.0861807 (W 0.0479595) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 16.4832 ± 0.0484499 (W 0.0263066) | 17.4796 ± 0.123827 (W 0.27238) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 47.2067 ± 0.140441 (W 0.175596) | 48.8613 ± 0.212443 (W 0.408371) | 60.5355 ± 0.187526 (W 0.0639689) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 66.1086 ± 0.204903 (W 0.115338) | 67.3232 ± 0.206043 (W 0.625089) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 234.565 ± 0.327144 (W 0.390977) | 238.307 ± 0.463159 (W 0.425369) | 339.735 ± 0.394468 (W 0.745785) | · | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 245.503 ± 0.182671 (W 0.304807) | 249.782 ± 0.660385 (W 2.1273) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 781.29 ± 1.30263 (W 0.71335) | 778.967 ± 2.19263 (W 1.5362) | 1117.22 ± 3.22121 (W 3.7372) | · | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 818.241 ± 0.433511 (W 1.82802) | 822.015 ± 3.21072 (W 1.41912) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 140.215 ± 0.653311 (W 0.671596) | 150.77 ± 1.12625 (W 0.849402) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 149.142 ± 0.845902 (W 0.981981) | 144.878 ± 0.285757 (W 0.492741) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 673.781 ± 1.81254 (W 4.50955) | 675.359 ± 19.1815 (W 1.73692) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 678.287 ± 4.95211 (W 5.65527) | 674.308 ± 2.09946 (W 2.98152) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 30.0004 ± 0.0380145 (W 0.149471) | 34.6446 ± 0.256029 (W 0.283333) | 37.5746 ± 0.011116 (W 0.115822) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `lu/n1024/b1 [api-inclusive]` | 45.1116 ± 0.12044 (W 0.0969142) | 37.0363 ± 0.148737 (W 0.319164) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 93.0338 ± 0.110412 (W 0.0878714) | 102.868 ± 0.641267 (W 0.621026) | 120.97 ± 0.0320629 (W 0.365947) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `lu/n1536/b1 [api-inclusive]` | 126.979 ± 0.317837 (W 0.163517) | 134.897 ± 0.136344 (W 0.847867) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 137.588 ± 0.177462 (W 0.191875) | 143.453 ± 0.32376 (W 0.251595) | 178.724 ± 1.00582 (W 0.466214) | · | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 161.027 ± 0.196263 (W 0.323971) | 166.524 ± 0.640318 (W 0.269784) | - | · | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 448.243 ± 0.970891 (W 0.658672) | 462.282 ± 1.89734 (W 0.551712) | 591.939 ± 3.40252 (W 2.04255) | · | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `qr/n1536/b1 [api-inclusive]` | 499.976 ± 1.34829 (W 0.372573) | 510.918 ± 1.56411 (W 0.892214) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 59.9424 ± 0.0860646 (W 0.0362447) | 65.8633 ± 0.101508 (W 0.0246766) | 69.9535 ± 0.211337 (W 0.0499648) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 190.584 ± 0.424786 (W 0.369153) | 203.571 ± 0.207759 (W 0.284562) | 235.827 ± 0.966133 (W 1.33271) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 30.1529 ± 0.0140291 (W 0.143368) | 34.3091 ± 0.295828 (W 0.24006) | 35.8606 ± 0.199329 (W 0.0388558) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `solve/n1024/b1 [api-inclusive]` | 33.3034 ± 0.202884 (W 0.194475) | 37.0426 ± 0.271041 (W 0.42618) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 93.2791 ± 0.23172 (W 0.0639402) | 102.872 ± 0.920101 (W 0.781026) | 116.096 ± 0.424681 (W 0.482969) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `solve/n1536/b1 [api-inclusive]` | 127.46 ± 0.235316 (W 0.343188) | 135.414 ± 0.414007 (W 0.778263) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 500.338 ± 0.180695 (W 0.620691) | 564.158 ± 1.25478 (W 1.43632) | 782.804 ± 3.29016 (W 1.40548) | 🔵 faer-fork 1.13× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 543.719 ± 1.82184 (W 1.33879) | 606.637 ± 0.552872 (W 4.92119) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1802.5 ± 3.60654 (W 1.98987) | 1993.24 ± 4.01788 (W 3.35673) | 2462.83 ± 6.37628 (W 6.77439) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1886.15 ± 1.61197 (W 2.22008) | 2079.15 ± 1.93064 (W 3.20547) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 278.764 ± 1.87283 (W 0.949335) | 359.524 ± 0.761051 (W 1.2231) | 424.45 ± 0.258668 (W 0.712081) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 279.449 ± 0.201868 (W 2.41084) | 356.826 ± 1.28628 (W 3.36845) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1106.99 ± 3.7404 (W 1.74105) | 1329.68 ± 2.40754 (W 1.94629) | 1460.58 ± 5.92655 (W 3.09662) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 1127.87 ± 1.71808 (W 1.77348) | 1352.77 ± 3.96508 (W 1.79132) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.209461 ± 0.00416336 (W 0.00428415) | 0.233055 ± 0.00736203 (W 0.00555188) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n1536/b1` | 0.660682 ± 0.00395822 (W 0.0050517) | 0.844165 ± 0.00366928 (W 0.0119382) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.42049 ± 0.00434966 (W 0.0138209) | 9.29835 ± 0.0444161 (W 0.0782842) | 10.9066 ± 0.0192455 (W 0.047029) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 10.2705 ± 0.030081 (W 0.0315868) | 11.1415 ± 0.0862328 (W 0.0720071) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 25.3482 ± 0.0656838 (W 0.0237574) | 26.422 ± 0.176529 (W 0.213959) | 32.8053 ± 0.0701841 (W 0.0706981) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 44.7339 ± 0.20689 (W 0.0901329) | 45.9584 ± 0.53035 (W 0.169639) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 136.592 ± 0.22984 (W 0.349361) | 178.321 ± 0.138043 (W 1.65068) | 210.393 ± 0.347606 (W 0.589633) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.54× faster than `lapack-mkl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 146.479 ± 0.214664 (W 0.417535) | 189.007 ± 0.294497 (W 1.3592) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 422.617 ± 0.534944 (W 0.795671) | 520.253 ± 0.201463 (W 4.14396) | 663.227 ± 0.809308 (W 1.46941) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 461.925 ± 0.952527 (W 1.59607) | 558.017 ± 3.98872 (W 3.86442) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 77.0484 ± 0.384922 (W 0.261997) | 89.9125 ± 0.301287 (W 0.714623) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 90.0671 ± 0.322451 (W 0.361977) | 88.0307 ± 0.0822716 (W 0.368417) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 476.861 ± 2.9001 (W 0.937729) | 477.648 ± 1.01489 (W 5.65089) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 506.511 ± 2.01722 (W 2.40255) | 505.483 ± 4.98805 (W 2.44679) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 18.0674 ± 0.0683899 (W 0.025484) | 22.017 ± 0.119011 (W 0.16698) | 20.9785 ± 0.0994594 (W 0.0617645) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `lu/n1024/b1 [api-inclusive]` | 33.8871 ± 0.121621 (W 0.053673) | 24.626 ± 0.030755 (W 0.196123) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 52.9888 ± 0.0680949 (W 0.167064) | 61.0221 ± 0.169902 (W 0.37671) | 70.3313 ± 0.387076 (W 0.457925) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `lu/n1536/b1 [api-inclusive]` | 87.3634 ± 0.32398 (W 0.2582) | 93.6192 ± 0.793759 (W 0.589172) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 72.3436 ± 0.262169 (W 0.0440227) | 108.711 ± 1.53378 (W 2.44974) | 98.2413 ± 1.19882 (W 0.316793) | 🔵 faer-fork 1.50× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 95.4412 ± 0.487414 (W 0.227308) | 122.792 ± 5.93855 (W 7.99526) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1536/b1` | 237.098 ± 0.302818 (W 0.090792) | 330.28 ± 11.0117 (W 8.55579) | 320.926 ± 0.464792 (W 1.5755) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` | `!` between-start range over 5% |
| `qr/n1536/b1 [api-inclusive]` | 287.971 ± 0.250259 (W 0.932648) | 375.283 ± 7.42154 (W 4.75655) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 31.9203 ± 0.084878 (W 0.0239731) | 53.3643 ± 1.16192 (W 0.985431) | 39.3234 ± 0.244715 (W 0.0966409) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 102.762 ± 0.104751 (W 0.107369) | 146.314 ± 3.39407 (W 3.50825) | 128.494 ± 1.48227 (W 1.3025) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 18.1306 ± 0.0180151 (W 0.0423577) | 21.7241 ± 0.140709 (W 0.135433) | 20.7781 ± 0.0178411 (W 0.0238983) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `solve/n1024/b1 [api-inclusive]` | 21.1088 ± 0.035008 (W 0.0318692) | 24.357 ± 0.0493279 (W 0.120366) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 53.5777 ± 0.109752 (W 0.181667) | 60.4575 ± 0.410296 (W 0.357312) | 68.8374 ± 0.353051 (W 0.354739) | 🔵 faer-fork 1.13× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `solve/n1536/b1 [api-inclusive]` | 87.6007 ± 0.413922 (W 0.392382) | 93.4384 ± 0.29379 (W 0.680475) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 295.773 ± 0.898169 (W 0.49238) | 389.98 ± 1.41362 (W 3.3007) | 480.294 ± 4.63377 (W 3.14018) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔵 faer-fork 1.62× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 339.063 ± 0.686486 (W 1.08983) | 430.39 ± 3.40902 (W 5.44277) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1187.5 ± 6.93106 (W 1.14124) | 1422.13 ± 8.41114 (W 9.2773) | 1692.87 ± 5.61129 (W 2.60158) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1271.55 ± 0.688728 (W 2.84492) | 1511.68 ± 3.8484 (W 5.37954) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 163.875 ± 1.21136 (W 1.13776) | 240.066 ± 0.0733535 (W 0.993541) | 251.48 ± 0.353968 (W 0.855309) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 163.974 ± 0.286185 (W 0.719401) | 238.489 ± 1.28548 (W 3.58579) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 788.927 ± 1.03436 (W 2.72363) | 957.473 ± 3.33406 (W 3.49886) | 1049.45 ± 2.64427 (W 2.54305) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 808.026 ± 1.85702 (W 2.56351) | 979.452 ± 4.9061 (W 3.87367) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.15068 ± 0.00237279 (W 0.0010339) | 0.165138 ± 0.00127009 (W 0.00160253) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.50541 ± 0.00463626 (W 0.00881413) | 0.708505 ± 0.0273962 (W 0.00668289) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024`, `cholesky/1536`, `eigh/1024`, `eigh/1536`, `full_piv_lu/1024`, `full_piv_lu/1536`, `lu/1024`, `lu/1536`, `qr/1024`, `qr/1536`, `qr_factor/1024`, `qr_factor/1536`, `solve/1024`, `solve/1536`, `svd_thin/1024`, `svd_thin/1536`, `svd_values/1024`, `svd_values/1536`, `triangular_solve/1024`, `triangular_solve/1536`.

## gemm-large, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.4902 ± 0.0634778 (W 0.0138826) | 17.4879 ± 0.040653 (W 0.0100441) | 20.6078 ± 0.0261309 (W 0.0422913) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 59.3518 ± 0.201681 (W 0.0294999) | 59.1999 ± 0.0662475 (W 0.0528164) | 65.2372 ± 0.0151124 (W 0.0604969) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |

## gemm-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.95624 ± 0.00886009 (W 0.00849058) | 8.97574 ± 0.0218214 (W 0.00732318) | 11.5449 ± 0.0935252 (W 0.0578912) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 30.0679 ± 0.0534672 (W 0.0335235) | 30.1206 ± 0.0215321 (W 0.0334642) | 35.9646 ± 0.0237099 (W 0.095081) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.5161 ± 0.067037 (W 0.0342996) | 17.4819 ± 0.035625 (W 0.012366) | 20.5733 ± 0.026593 (W 0.0290621) | · | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 59.2058 ± 0.0359505 (W 0.104326) | 59.3122 ± 0.241218 (W 0.103241) | 64.9237 ± 0.129395 (W 0.0619131) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |

## gemm-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.96495 ± 0.0116034 (W 0.00728465) | 8.97308 ± 0.00717325 (W 0.00591131) | 11.505 ± 0.0471514 (W 0.0378813) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 30.0728 ± 0.020259 (W 0.0270867) | 30.1431 ± 0.0326147 (W 0.0296864) | 35.9335 ± 0.210467 (W 0.0621474) | · | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |

## gemm-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.7106 ± 0.156412 (W 0.0751655) | 69.5918 ± 0.215701 (W 0.03527) | 109.867 ± 0.397029 (W 0.0792774) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 238.192 ± 1.03593 (W 0.336542) | 237.426 ± 0.444709 (W 0.352097) | 368.596 ± 0.715243 (W 0.553079) | · | 🔵 faer-fork 1.55× faster than `lapack-mkl` |  |

## gemm-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.1376 ± 0.0228498 (W 0.0157097) | 35.1508 ± 0.0590338 (W 0.0116378) | 44.8488 ± 0.150015 (W 0.169955) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 119.355 ± 0.0751991 (W 0.145869) | 119.441 ± 0.25496 (W 0.191215) | 146.8 ± 0.346504 (W 0.427533) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |

## gemm-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.637 ± 0.210043 (W 0.103139) | 69.6066 ± 0.18102 (W 0.0385328) | 110.314 ± 0.174698 (W 0.134751) | · | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 237.917 ± 0.200679 (W 0.333705) | 237.956 ± 0.543915 (W 0.251087) | 369.174 ± 0.535334 (W 0.361939) | · | 🔵 faer-fork 1.55× faster than `lapack-mkl` |  |

## gemm-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.1348 ± 0.0530439 (W 0.0325017) | 35.1129 ± 0.0350178 (W 0.0204535) | 44.9748 ± 0.173885 (W 0.220559) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 119.451 ± 0.0924223 (W 0.0645684) | 119.495 ± 0.0711744 (W 0.0935154) | 147.557 ± 0.764291 (W 0.439036) | · | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |

## linalg-boundary, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.165758 ± 0.000958368 (W 0.00125393) | 0.53563 ± 0.0536061 (W 0.0580185) | - | 🔵 faer-fork 3.23× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.190328 ± 0.00128037 (W 0.0014095) | 0.325889 ± 0.00237615 (W 0.0121238) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.152145 ± 0.000852363 (W 0.000582265) | 0.548834 ± 0.0600065 (W 0.0844359) | - | 🔵 faer-fork 3.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1 [api-inclusive]` | 0.180236 ± 0.00200789 (W 0.00141788) | 0.295852 ± 0.00375119 (W 0.00725787) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.701931 ± 0.00403926 (W 0.0021698) | 1.603 ± 0.0830188 (W 0.106644) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n511/b1 [api-inclusive]` | 0.851189 ± 0.00178427 (W 0.0029853) | 1.12524 ± 0.0492035 (W 0.0259139) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n513/b1` | 0.67981 ± 0.00169617 (W 0.0016193) | 1.55709 ± 0.0820434 (W 0.102571) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n513/b1 [api-inclusive]` | 0.931867 ± 0.00447855 (W 0.00296699) | 1.07503 ± 0.0131487 (W 0.0260185) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.84158 ± 0.0201339 (W 0.00654338) | 7.324 ± 0.00919209 (W 0.0177957) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.8612 ± 0.0165664 (W 0.00725608) | 7.37302 ± 0.02322 (W 0.0116283) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.79572 ± 0.00245894 (W 0.00960222) | 7.20682 ± 0.0323282 (W 0.013371) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.79402 ± 0.0136055 (W 0.00996216) | 7.24709 ± 0.00874762 (W 0.00761515) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.7631 ± 0.0454128 (W 0.0359312) | 25.5455 ± 0.0195733 (W 0.0369445) | - | · | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 24.8228 ± 0.0426517 (W 0.0395616) | 25.6229 ± 0.0275747 (W 0.0329125) | - | · | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.2755 ± 0.0382012 (W 0.0358943) | 25.9021 ± 0.00671046 (W 0.0207826) | - | · | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 25.2956 ± 0.0421869 (W 0.0159533) | 25.9519 ± 0.0569607 (W 0.0539842) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.26684 ± 0.00259185 (W 0.00411376) | 1.98477 ± 0.00403646 (W 0.00403319) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.2513 ± 0.0139291 (W 0.0126706) | 2.64623 ± 0.0137966 (W 0.00892611) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.30096 ± 0.00142462 (W 0.00679566) | 2.03832 ± 0.00584345 (W 0.00585881) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.56135 ± 0.00550062 (W 0.00375935) | 1.88054 ± 0.174199 (W 0.0065045) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n511/b1` | 6.33582 ± 0.0282765 (W 0.0101952) | 9.23309 ± 0.0381143 (W 0.0103098) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 13.0379 ± 0.0347985 (W 0.0406243) | 14.3604 ± 0.0402808 (W 0.0191087) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.42464 ± 0.00948237 (W 0.0113687) | 9.34615 ± 0.0471781 (W 0.0134533) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 7.3607 ± 0.0160808 (W 0.0111376) | 8.55945 ± 0.0224849 (W 0.0228362) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.432495 ± 0.00214438 (W 0.00271144) | 0.799124 ± 0.00528551 (W 0.0121152) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.704325 ± 0.0111393 (W 0.00474925) | 0.641387 ± 0.00864263 (W 0.0127308) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.413062 ± 0.00295338 (W 0.00278443) | 0.803641 ± 0.0149138 (W 0.0154948) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.76809 ± 0.00513918 (W 0.00532591) | 0.581549 ± 0.00978621 (W 0.0075289) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.74735 ± 0.0131547 (W 0.0129613) | 3.42863 ± 0.0640642 (W 0.0392989) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.48806 ± 0.0255297 (W 0.00765064) | 3.34409 ± 0.0963305 (W 0.0532239) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n513/b1` | 1.85752 ± 0.00750546 (W 0.0054203) | 3.47092 ± 0.0430491 (W 0.0316305) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.45872 ± 0.00763589 (W 0.011876) | 2.51599 ± 0.0203816 (W 0.0254793) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.0464 ± 0.00254441 (W 0.00130712) | 2.73044 ± 0.0108954 (W 0.01339) | - | 🔵 faer-fork 2.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.56612 ± 0.0112981 (W 0.00781866) | 2.25031 ± 0.00970655 (W 0.00767601) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.11679 ± 0.00306616 (W 0.00241388) | 2.84811 ± 0.0075407 (W 0.0104086) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.67428 ± 0.00505581 (W 0.00698946) | 2.33728 ± 0.0171126 (W 0.00639059) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.83563 ± 0.00751107 (W 0.0118384) | 12.3203 ± 0.0286867 (W 0.0351906) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 8.40942 ± 0.015209 (W 0.0905079) | 9.59882 ± 0.023398 (W 0.0190668) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 6.00834 ± 0.00903464 (W 0.0117778) | 12.1566 ± 0.0208482 (W 0.0380254) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 8.57261 ± 0.0365767 (W 0.0480295) | 9.57645 ± 0.396628 (W 0.0334061) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n255/b1` | 0.655584 ± 0.000562741 (W 0.00176914) | 2.33606 ± 0.00520031 (W 0.0117681) | - | 🔵 faer-fork 3.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 0.695117 ± 0.00261921 (W 0.0020765) | 2.42246 ± 0.0179242 (W 0.0138743) | - | 🔵 faer-fork 3.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 3.18188 ± 0.00689789 (W 0.00597343) | 9.72373 ± 0.0394985 (W 0.0443334) | - | 🔵 faer-fork 3.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 3.19521 ± 0.00255618 (W 0.00404617) | 9.39061 ± 0.0321194 (W 0.0368032) | - | 🔵 faer-fork 2.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.354953 ± 0.000982429 (W 0.00318522) | 0.784225 ± 0.0029364 (W 0.0144399) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.710906 ± 0.00353298 (W 0.0042153) | 0.5729 ± 0.00955664 (W 0.00958918) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.422259 ± 0.00192718 (W 0.00273561) | 0.788713 ± 0.00975468 (W 0.0115383) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.768325 ± 0.00930982 (W 0.00524885) | 0.560206 ± 0.00146389 (W 0.00573426) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.708 ± 9.97112e-05 (W 0.00628683) | 3.32018 ± 0.0168611 (W 0.0334886) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.46229 ± 0.0712465 (W 0.00607355) | 2.8776 ± 0.0203263 (W 0.0523444) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.80617 ± 0.00752792 (W 0.00440364) | 3.29862 ± 0.0538254 (W 0.0365549) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.44171 ± 0.0257392 (W 0.00871204) | 2.37913 ± 0.02836 (W 0.0369316) | - | 🔴 faer-fork 1.45× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.92329 ± 0.0193073 (W 0.0148974) | 13.4441 ± 0.0415671 (W 0.0260854) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 7.9215 ± 0.0156083 (W 0.00698006) | 13.5156 ± 0.0511797 (W 0.0124165) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.09672 ± 0.0324429 (W 0.00267027) | 10.1337 ± 0.0135545 (W 0.0175408) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 8.1285 ± 0.0183206 (W 0.0210236) | 10.1939 ± 0.0227271 (W 0.0228474) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.7512 ± 0.0668568 (W 0.0469955) | 54.1883 ± 0.133606 (W 0.0377619) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 35.8375 ± 0.103388 (W 0.0588473) | 54.449 ± 0.104547 (W 0.550293) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.5332 ± 0.0602639 (W 0.0273828) | 41.3899 ± 0.049301 (W 0.0648602) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 36.6595 ± 0.0430887 (W 0.0462181) | 41.533 ± 0.0528309 (W 0.308763) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 4.00774 ± 0.0113769 (W 0.0106674) | 10.6827 ± 0.0620488 (W 0.0185888) | - | 🔵 faer-fork 2.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 4.0101 ± 0.017014 (W 0.00509908) | 10.6832 ± 0.0347424 (W 0.025113) | - | 🔵 faer-fork 2.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.09933 ± 0.00574041 (W 0.0119179) | 8.2156 ± 0.043634 (W 0.00886836) | - | 🔵 faer-fork 2.00× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 4.08638 ± 0.019055 (W 0.00711823) | 8.22793 ± 0.00793771 (W 0.00965298) | - | 🔵 faer-fork 2.01× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.4683 ± 0.0302767 (W 0.0174457) | 40.784 ± 0.0800897 (W 0.0566527) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 17.4842 ± 0.0551886 (W 0.018319) | 40.7194 ± 0.123564 (W 0.0583938) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.6942 ± 0.0325727 (W 0.0152558) | 29.7099 ± 0.0722853 (W 0.0474702) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 17.7312 ± 0.0358921 (W 0.0180976) | 29.6596 ± 0.0251774 (W 0.0237902) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.012484 ± 2.55147e-05 (W 0.00012136) | 0.011582 ± 0.000109244 (W 8.47184e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.012486 ± 0.000153024 (W 0.000164734) | 0.011592 ± 0.000123923 (W 5.08852e-05) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.029815 ± 0.000226497 (W 0.000322128) | 0.039373 ± 0.000168838 (W 0.00033199) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.029706 ± 0.000151565 (W 0.000224094) | 0.028563 ± 0.000120658 (W 7.30363e-05) | - | · | - | `*` one row has no measurement here |

## linalg-boundary, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.147976 ± 0.00124 (W 0.000718281) | 0.48563 ± 0.00359336 (W 0.0241056) | - | 🔵 faer-fork 3.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.175313 ± 0.000451106 (W 0.00157234) | 0.312191 ± 0.00705417 (W 0.0236569) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.141084 ± 0.000543283 (W 0.00228408) | 0.509412 ± 0.00566486 (W 0.008183) | - | 🔵 faer-fork 3.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.168574 ± 0.00123366 (W 0.0011574) | 0.28984 ± 0.00846132 (W 0.0241735) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n511/b1` | 0.505242 ± 0.00313624 (W 0.00264842) | 1.38174 ± 0.0111836 (W 0.0183096) | - | 🔵 faer-fork 2.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 0.67987 ± 0.00255986 (W 0.00193563) | 1.04573 ± 0.00750534 (W 0.0316373) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.499193 ± 0.00522127 (W 0.0022817) | 1.38546 ± 0.0115901 (W 0.0255152) | - | 🔵 faer-fork 2.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.786215 ± 0.0173911 (W 0.00665586) | 0.975331 ± 0.0120383 (W 0.0375398) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 3.99598 ± 0.00548208 (W 0.00394439) | 7.33461 ± 0.0517162 (W 0.0153083) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 4.00405 ± 0.00877263 (W 0.00538451) | 7.31422 ± 0.0391326 (W 0.0221796) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.95815 ± 0.0112484 (W 0.00359628) | 7.19174 ± 0.0202517 (W 0.0141452) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 3.9696 ± 0.000986972 (W 0.00399923) | 7.20744 ± 0.0188599 (W 0.0133858) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.1934 ± 0.0189109 (W 0.0324219) | 23.9405 ± 0.0372647 (W 0.0550456) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 16.1742 ± 0.0261998 (W 0.03587) | 24.0969 ± 0.0279013 (W 0.0434834) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.7384 ± 0.00121946 (W 0.0247537) | 24.2543 ± 0.0235014 (W 0.0566237) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 16.7831 ± 0.0439691 (W 0.0165905) | 24.3134 ± 0.0652001 (W 0.0510187) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.16059 ± 0.0117843 (W 0.00403174) | 1.98677 ± 0.00854824 (W 0.00426408) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.17832 ± 0.0106491 (W 0.0111739) | 2.65616 ± 0.125248 (W 0.011216) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n257/b1` | 1.19508 ± 0.00965787 (W 0.00447662) | 2.04251 ± 0.00893611 (W 0.00175875) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.49978 ± 0.00809073 (W 0.00342971) | 1.89014 ± 0.182109 (W 0.00549104) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n511/b1` | 4.79047 ± 0.0146657 (W 0.00837412) | 8.32679 ± 0.0532201 (W 0.0101855) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.8259 ± 0.0671064 (W 0.0288728) | 13.5694 ± 0.0578884 (W 0.0226247) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.86773 ± 0.0280662 (W 0.00820329) | 8.38861 ± 0.0915507 (W 0.012784) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.10541 ± 0.057998 (W 0.0170697) | 7.6949 ± 0.0653719 (W 0.0120281) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.424029 ± 0.00178736 (W 0.0015769) | 0.85748 ± 0.0101979 (W 0.018064) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.671675 ± 0.00625641 (W 0.00316187) | 0.666161 ± 0.00326751 (W 0.0235305) | - | · | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.354003 ± 0.00132678 (W 0.00157042) | 0.862209 ± 0.0122492 (W 0.0267911) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.728981 ± 0.0115918 (W 0.00278507) | 0.60692 ± 0.028257 (W 0.023552) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n511/b1` | 1.2753 ± 0.00469554 (W 0.00407472) | 3.3778 ± 0.0451549 (W 0.0632082) | - | 🔵 faer-fork 2.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.16394 ± 0.0324198 (W 0.0169039) | 3.01616 ± 0.0249106 (W 0.0455211) | - | · | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.35616 ± 0.00386363 (W 0.00403129) | 3.45302 ± 0.0102458 (W 0.0445304) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.11805 ± 0.0181198 (W 0.00654676) | 2.36979 ± 0.0317328 (W 0.0533842) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.840641 ± 0.00577071 (W 0.00129198) | 3.01008 ± 0.0270319 (W 0.0139969) | - | 🔵 faer-fork 3.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.40319 ± 0.000531591 (W 0.0082235) | 2.1713 ± 0.0612859 (W 0.0364268) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n257/b1` | 0.897391 ± 0.00493971 (W 0.00187333) | 3.14602 ± 0.0100255 (W 0.0193994) | - | 🔵 faer-fork 3.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.48197 ± 0.00135638 (W 0.0077325) | 2.24766 ± 0.173138 (W 0.0101482) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n511/b1` | 3.28434 ± 0.013284 (W 0.00680913) | 12.0554 ± 0.0531615 (W 0.10609) | - | 🔵 faer-fork 3.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 6.07848 ± 0.042164 (W 0.0161324) | 7.74731 ± 1.15816 (W 0.726838) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1` | 3.80681 ± 0.14399 (W 0.0152958) | 11.6251 ± 0.042988 (W 0.0813112) | - | 🔵 faer-fork 3.05× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1 [api-inclusive]` | 6.55299 ± 0.0200683 (W 0.0158397) | 7.61618 ± 0.116837 (W 0.505381) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.610811 ± 0.00208622 (W 0.00227347) | 2.5876 ± 0.0220091 (W 0.0155961) | - | 🔵 faer-fork 4.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 0.645433 ± 0.00209716 (W 0.00252563) | 2.69414 ± 0.0170227 (W 0.0180908) | - | 🔵 faer-fork 4.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 1.87867 ± 0.00536821 (W 0.00336096) | 9.85911 ± 0.0960719 (W 0.0741523) | - | 🔵 faer-fork 5.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 2.23435 ± 0.0147537 (W 0.00590176) | 9.23686 ± 0.0145386 (W 0.0385466) | - | 🔵 faer-fork 4.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.340626 ± 0.000852616 (W 0.00255123) | 0.831654 ± 0.012199 (W 0.0159748) | - | 🔵 faer-fork 2.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.681411 ± 0.00136102 (W 0.00463194) | 0.597468 ± 0.0267322 (W 0.0273893) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n257/b1` | 0.355545 ± 0.00113199 (W 0.00217198) | 0.861543 ± 0.00781771 (W 0.0249056) | - | 🔵 faer-fork 2.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.741134 ± 0.00221277 (W 0.00390802) | 0.594831 ± 0.00365548 (W 0.0218342) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.2352 ± 0.0091189 (W 0.00283968) | 3.25369 ± 0.0388341 (W 0.0464022) | - | 🔵 faer-fork 2.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.15669 ± 0.0547879 (W 0.0450047) | 2.71645 ± 0.0512255 (W 0.0535842) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.31279 ± 0.00556586 (W 0.00557412) | 3.30469 ± 0.0268926 (W 0.0561822) | - | 🔵 faer-fork 2.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.08437 ± 0.0359673 (W 0.01233) | 2.30901 ± 0.0587285 (W 0.0620808) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.22125 ± 0.0303739 (W 0.00964737) | 13.5513 ± 0.0366908 (W 0.0321943) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 7.26665 ± 0.00576128 (W 0.0110362) | 13.6165 ± 0.0315907 (W 0.0233718) | - | 🔵 faer-fork 1.87× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.34766 ± 0.0323897 (W 0.0146058) | 10.1506 ± 0.0248092 (W 0.0106691) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 7.37995 ± 0.00518718 (W 0.013504) | 10.2342 ± 0.023781 (W 0.0251663) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.3388 ± 0.0906772 (W 0.134698) | 50.8844 ± 0.149569 (W 0.395092) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 29.4788 ± 0.0968867 (W 0.0365775) | 51.0507 ± 0.187036 (W 0.17055) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.8436 ± 0.0695615 (W 0.0685086) | 37.8399 ± 0.0620342 (W 0.100418) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 29.9586 ± 0.0692962 (W 0.0534573) | 37.9941 ± 0.0427606 (W 0.176988) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.88532 ± 0.0113721 (W 0.00785655) | 10.7547 ± 0.024886 (W 0.0178477) | - | 🔵 faer-fork 2.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 3.88663 ± 0.00934435 (W 0.00634984) | 10.7418 ± 0.0275842 (W 0.0185248) | - | 🔵 faer-fork 2.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.97149 ± 0.0169225 (W 0.0087592) | 8.27423 ± 0.0221186 (W 0.0175201) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 3.95583 ± 0.010621 (W 0.00896145) | 8.30273 ± 0.0180316 (W 0.0111725) | - | 🔵 faer-fork 2.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.3756 ± 0.0106981 (W 0.0124529) | 39.2327 ± 0.0595779 (W 0.0639253) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 15.4206 ± 0.0613861 (W 0.0144557) | 39.1074 ± 0.186099 (W 0.0503502) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.5515 ± 0.0349244 (W 0.0108402) | 28.0189 ± 0.0847655 (W 0.0147313) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 15.5558 ± 0.0469602 (W 0.0169725) | 27.9904 ± 0.0796278 (W 0.0384737) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.012565 ± 1.00167e-05 (W 0.000100798) | 0.011482 ± 4.04145e-05 (W 9.12212e-05) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.012423 ± 0.000100848 (W 0.000104854) | 0.011672 ± 5.1316e-05 (W 6.79868e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.028954 ± 0.000114919 (W 0.00017117) | 0.038581 ± 0.000361733 (W 0.000339098) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.028163 ± 0.000255324 (W 0.000182004) | 0.027571 ± 0.000330396 (W 0.000323878) | - | · | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.157704 ± 0.000308871 (W 0.00755761) | 0.28924 ± 0.0109389 (W 0.0164868) | 0.234678 ± 0.00329179 (W 0.00732413) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.191367 ± 0.00153386 (W 0.00159742) | 0.330497 ± 0.00512655 (W 0.00741327) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.145882 ± 0.00190475 (W 0.000945935) | 0.264324 ± 0.00508263 (W 0.0122842) | 0.247683 ± 0.0068454 (W 0.0104464) | 🔵 faer-fork 1.81× faster than `faer-upstream` | 🔵 faer-fork 1.70× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n257/b1 [api-inclusive]` | 0.180377 ± 0.00184717 (W 0.00255221) | 0.29975 ± 0.00406102 (W 0.00708603) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.684178 ± 0.0024906 (W 0.00192944) | 0.961034 ± 0.02054 (W 0.0237249) | 1.20186 ± 0.00693717 (W 0.0180524) | 🔵 faer-fork 1.40× faster than `faer-upstream` | 🔵 faer-fork 1.76× faster than `lapack-mkl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.840199 ± 0.00243902 (W 0.00241388) | 1.14059 ± 0.00876268 (W 0.0366208) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.66938 ± 0.00240894 (W 0.00348781) | 0.977776 ± 0.0128908 (W 0.0417518) | 1.23718 ± 0.000499501 (W 0.00728307) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔵 faer-fork 1.85× faster than `lapack-mkl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.803005 ± 0.00119506 (W 0.00379708) | 1.01792 ± 0.0358049 (W 0.0269749) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n255/b1` | 5.86486 ± 0.0164412 (W 0.00528033) | 7.33595 ± 0.0292136 (W 0.0105558) | 4.39245 ± 0.0147947 (W 0.0035981) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔴 faer-fork 1.34× slower than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 5.87204 ± 0.0228687 (W 0.00495118) | 7.34449 ± 0.0287109 (W 0.00810205) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.79033 ± 0.00281815 (W 0.0146354) | 7.22824 ± 0.0316686 (W 0.0133169) | 4.46113 ± 0.0145388 (W 0.00720316) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔴 faer-fork 1.30× slower than `lapack-mkl` |  |
| `eigh/n257/b1 [api-inclusive]` | 5.80014 ± 0.00457074 (W 0.0102084) | 7.21363 ± 0.00859175 (W 0.00854414) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.7206 ± 0.0474926 (W 0.0349998) | 25.4725 ± 0.0198412 (W 0.0293396) | 19.0025 ± 0.0760251 (W 0.0456913) | · | 🔴 faer-fork 1.30× slower than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 24.712 ± 0.0455745 (W 0.024254) | 25.5898 ± 0.0379054 (W 0.0320085) | - | · | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.1217 ± 0.0540685 (W 0.0366994) | 25.8482 ± 0.0422598 (W 0.0396161) | 19.2858 ± 0.11377 (W 0.024496) | · | 🔴 faer-fork 1.30× slower than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 25.1484 ± 0.0389169 (W 0.0170322) | 25.9654 ± 0.097059 (W 0.0391511) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.17244 ± 0.00238605 (W 0.00385757) | 1.98135 ± 0.009162 (W 0.0074044) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.16602 ± 0.0100921 (W 0.0050021) | 2.65876 ± 0.0443898 (W 0.00679061) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.18483 ± 0.0061396 (W 0.00617069) | 2.03658 ± 0.00888432 (W 0.00201387) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.44816 ± 0.0109979 (W 0.00550486) | 2.1532 ± 0.159344 (W 0.00524365) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n511/b1` | 6.01382 ± 0.01933 (W 0.00894887) | 9.18356 ± 0.0172661 (W 0.0156247) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 12.7091 ± 0.103525 (W 0.0830954) | 14.1681 ± 0.0817317 (W 0.0229915) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.09915 ± 0.0235143 (W 0.00598629) | 9.35181 ± 0.0388152 (W 0.0101254) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.83594 ± 0.0245985 (W 0.0123295) | 8.38095 ± 0.064664 (W 0.0168647) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.306673 ± 0.00403535 (W 0.00255201) | 0.52812 ± 0.00739238 (W 0.012173) | 0.288108 ± 0.00163808 (W 0.00186475) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `lu/n255/b1 [api-inclusive]` | 0.613305 ± 0.00390667 (W 0.00484492) | 0.629946 ± 0.0117926 (W 0.0104391) | - | · | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.328173 ± 0.00157839 (W 0.00279477) | 0.531416 ± 0.00224466 (W 0.00774171) | 0.297265 ± 0.00257656 (W 0.00240084) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔴 faer-fork 1.10× slower than `lapack-mkl` |  |
| `lu/n257/b1 [api-inclusive]` | 0.656315 ± 0.00386137 (W 0.00294142) | 0.576748 ± 0.00140072 (W 0.00769041) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.48624 ± 0.00447261 (W 0.00639989) | 2.20074 ± 0.0169824 (W 0.0296777) | 1.40695 ± 0.00867574 (W 0.00501587) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 3.00688 ± 0.00829488 (W 0.0105126) | 3.10612 ± 0.0262459 (W 0.0869632) | - | · | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.55304 ± 0.00705759 (W 0.00732111) | 2.24813 ± 0.00962726 (W 0.0237335) | 1.43029 ± 0.00172198 (W 0.00563696) | 🔵 faer-fork 1.45× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.96506 ± 0.00493923 (W 0.00666052) | 2.3804 ± 0.00966901 (W 0.0358614) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.9714 ± 0.0011322 (W 0.0027554) | 1.72171 ± 0.00303444 (W 0.00580612) | 1.3172 ± 0.0045132 (W 0.00176089) | 🔵 faer-fork 1.77× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 1.46199 ± 0.00445606 (W 0.00620296) | 2.21226 ± 0.0104775 (W 0.00766545) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.00173 ± 0.00231065 (W 0.00120805) | 1.7782 ± 0.007112 (W 0.00444624) | 1.35929 ± 0.00091227 (W 0.00386944) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 1.53548 ± 0.00203178 (W 0.00572233) | 2.29543 ± 0.00556274 (W 0.0126544) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.4823 ± 0.00805166 (W 0.00594164) | 7.06291 ± 0.0398223 (W 0.0130408) | 7.60444 ± 0.0240699 (W 0.00491862) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 7.74867 ± 0.0268809 (W 0.011877) | 9.31131 ± 0.0265951 (W 0.0136896) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 5.58729 ± 0.0205088 (W 0.00782592) | 7.0357 ± 0.00803978 (W 0.01251) | 7.79026 ± 0.0115689 (W 0.0101719) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 7.96245 ± 0.0212176 (W 0.0161149) | 8.74373 ± 0.354406 (W 0.00846787) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n255/b1` | 0.580634 ± 0.00088491 (W 0.00204596) | 1.35527 ± 0.0127551 (W 0.00302852) | 0.612484 ± 0.000681612 (W 0.00292845) | 🔵 faer-fork 2.33× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `qr_factor/n257/b1` | 0.598668 ± 0.00174171 (W 0.00201467) | 1.38728 ± 0.0107755 (W 0.00676108) | 0.628616 ± 0.00120704 (W 0.0018211) | 🔵 faer-fork 2.32× faster than `faer-upstream` | · |  |
| `qr_factor/n511/b1` | 2.91241 ± 0.00603365 (W 0.0041736) | 4.48473 ± 0.0371913 (W 0.00682433) | 3.53428 ± 0.0130928 (W 0.00542655) | 🔵 faer-fork 1.54× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 2.86679 ± 0.00856175 (W 0.00244768) | 4.3124 ± 0.0176514 (W 0.00884868) | 3.61266 ± 0.00415936 (W 0.00586773) | 🔵 faer-fork 1.50× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `solve/n255/b1` | 0.313617 ± 0.00310164 (W 0.00246353) | 0.486609 ± 0.00344534 (W 0.0067089) | 0.284348 ± 0.00102153 (W 0.00238899) | 🔵 faer-fork 1.55× faster than `faer-upstream` | 🔴 faer-fork 1.10× slower than `lapack-mkl` |  |
| `solve/n255/b1 [api-inclusive]` | 0.620883 ± 0.00115134 (W 0.0024266) | 0.571818 ± 0.00513492 (W 0.00753294) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.328043 ± 0.000634269 (W 0.00163524) | 0.489614 ± 0.0054567 (W 0.0109028) | 0.288349 ± 0.000220116 (W 0.00269967) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `solve/n257/b1 [api-inclusive]` | 0.662407 ± 0.00324658 (W 0.00290903) | 0.542313 ± 0.00591927 (W 0.00439595) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.45461 ± 0.00332594 (W 0.00272669) | 2.06646 ± 0.0153227 (W 0.0396557) | 1.39511 ± 0.00508265 (W 0.00313241) | 🔵 faer-fork 1.42× faster than `faer-upstream` | · |  |
| `solve/n511/b1 [api-inclusive]` | 2.93861 ± 0.0243425 (W 0.00487881) | 2.7583 ± 0.017367 (W 0.0580621) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.51642 ± 0.00596399 (W 0.00529794) | 2.11143 ± 0.0237054 (W 0.01606) | 1.41776 ± 0.0021427 (W 0.00568591) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` |  |
| `solve/n513/b1 [api-inclusive]` | 2.92867 ± 0.00139994 (W 0.00829127) | 2.2691 ± 0.0202681 (W 0.0256597) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.90008 ± 0.0306827 (W 0.0110798) | 13.506 ± 0.0150383 (W 0.013928) | 9.71803 ± 0.0157824 (W 0.00818285) | 🔵 faer-fork 1.71× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.90656 ± 0.0167592 (W 0.0127757) | 13.5491 ± 0.044975 (W 0.023249) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.08611 ± 0.0261089 (W 0.0106889) | 10.1313 ± 0.0391593 (W 0.0140059) | 9.98246 ± 0.0184766 (W 0.0085899) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 8.12069 ± 0.0139788 (W 0.0168434) | 10.1879 ± 0.055271 (W 0.0129281) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.5913 ± 0.0510547 (W 0.0366087) | 54.0692 ± 0.0748374 (W 0.0319821) | 39.958 ± 0.0408422 (W 0.0648973) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 35.6656 ± 0.0453507 (W 0.0302685) | 54.2996 ± 0.0637234 (W 0.531703) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.2789 ± 0.0119944 (W 0.0393736) | 41.1442 ± 0.0309188 (W 0.0415771) | 40.6498 ± 0.0624188 (W 0.0436028) | 🔵 faer-fork 1.13× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 36.3592 ± 0.0488427 (W 0.0457146) | 41.3098 ± 0.00233558 (W 0.270166) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.98706 ± 0.0172322 (W 0.00633643) | 10.7416 ± 0.0568644 (W 0.0181434) | 5.1276 ± 0.0182902 (W 0.00615666) | 🔵 faer-fork 2.69× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 3.97774 ± 0.0158002 (W 0.00584669) | 10.7412 ± 0.0454685 (W 0.00655193) | - | 🔵 faer-fork 2.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.11282 ± 0.00912027 (W 0.00688004) | 8.18374 ± 0.0644704 (W 0.0466903) | 5.22666 ± 0.0171405 (W 0.00680916) | 🔵 faer-fork 1.99× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 4.10653 ± 0.0104679 (W 0.00596794) | 8.19 ± 0.0224715 (W 0.00539208) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.4154 ± 0.0315797 (W 0.0111238) | 40.6323 ± 0.0784363 (W 0.0307196) | 19.1644 ± 0.0364666 (W 0.0213758) | 🔵 faer-fork 2.33× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 17.3986 ± 0.0557978 (W 0.0220239) | 40.5994 ± 0.0489383 (W 0.0421793) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.6296 ± 0.0997472 (W 0.0164076) | 29.4378 ± 0.021697 (W 0.0307122) | 19.278 ± 0.103288 (W 0.0284226) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 17.5901 ± 0.069774 (W 0.0258591) | 29.4623 ± 0.104815 (W 0.0556578) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.014617 ± 0.000121006 (W 0.000329683) | 0.012202 ± 5.40278e-05 (W 9.8779e-05) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.014788 ± 0.000174646 (W 0.000298616) | 0.012525 ± 4.23596e-05 (W 6.92625e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.038092 ± 0.000119684 (W 0.000517722) | 0.04254 ± 0.000469501 (W 0.000214502) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.03765 ± 0.000294111 (W 0.000750156) | 0.034344 ± 0.000382704 (W 0.000409667) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.141825 ± 0.000362381 (W 0.001063) | 0.302331 ± 0.0262091 (W 0.0221615) | 0.210062 ± 0.00087185 (W 0.00505366) | 🔵 faer-fork 2.13× faster than `faer-upstream` | 🔵 faer-fork 1.48× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.176609 ± 0.000778364 (W 0.0018661) | 0.31137 ± 0.00719238 (W 0.00498304) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.13287 ± 0.00150455 (W 0.00118354) | 0.271315 ± 0.00892874 (W 0.0303681) | 0.221142 ± 0.00172749 (W 0.00496324) | 🔵 faer-fork 2.04× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n257/b1 [api-inclusive]` | 0.166356 ± 0.000594258 (W 0.000536265) | 0.289306 ± 0.005044 (W 0.0246509) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.503027 ± 0.00419022 (W 0.00445628) | 0.841632 ± 0.0287767 (W 0.0386753) | 0.803731 ± 0.00718774 (W 0.0126613) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n511/b1 [api-inclusive]` | 0.673276 ± 0.00292343 (W 0.0027158) | 1.05664 ± 0.0115071 (W 0.0372155) | - | 🔵 faer-fork 1.57× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.505113 ± 0.00116881 (W 0.00318488) | 0.844711 ± 0.00738184 (W 0.0414965) | 0.799183 ± 0.00309978 (W 0.00793121) | 🔵 faer-fork 1.67× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.655214 ± 0.00433487 (W 0.00316637) | 0.939647 ± 0.00750871 (W 0.0266566) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 3.99135 ± 0.00140685 (W 0.00420594) | 7.30333 ± 0.0426423 (W 0.0256082) | 3.72694 ± 0.0113822 (W 0.00655923) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 3.99457 ± 0.00546262 (W 0.00495465) | 7.3091 ± 0.0241766 (W 0.0167602) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.94483 ± 0.00632102 (W 0.00807331) | 7.18058 ± 0.0203274 (W 0.0187209) | 3.81461 ± 0.0260407 (W 0.00870778) | 🔵 faer-fork 1.82× faster than `faer-upstream` | · |  |
| `eigh/n257/b1 [api-inclusive]` | 3.96698 ± 0.00836098 (W 0.00735124) | 7.21993 ± 0.0561136 (W 0.0204019) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.1719 ± 0.00577441 (W 0.0350637) | 23.8829 ± 0.0298996 (W 0.0352062) | 13.1707 ± 0.0339791 (W 0.0219832) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔴 faer-fork 1.23× slower than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 16.1227 ± 0.0268639 (W 0.0192878) | 23.9781 ± 0.0241386 (W 0.0705805) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.5801 ± 0.0585475 (W 0.0240452) | 24.1871 ± 0.0848315 (W 0.111816) | 13.5122 ± 0.0226977 (W 0.407521) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔴 faer-fork 1.23× slower than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 16.6272 ± 0.0206625 (W 0.0264489) | 24.3241 ± 0.0298756 (W 0.0485148) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.11001 ± 0.00617448 (W 0.00360722) | 1.98702 ± 0.00705191 (W 0.00394158) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.10275 ± 0.017808 (W 0.00919617) | 2.85407 ± 0.128129 (W 0.0100622) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n257/b1` | 1.10809 ± 0.00160839 (W 0.0031703) | 2.04598 ± 0.00402445 (W 0.00505428) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.39351 ± 0.00218219 (W 0.00392119) | 1.8824 ± 0.157275 (W 0.00498289) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n511/b1` | 4.58568 ± 0.0177316 (W 0.0118656) | 8.26858 ± 0.0464266 (W 0.00794821) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.3169 ± 0.0271682 (W 0.0166805) | 13.4438 ± 0.0806444 (W 0.0230212) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.66374 ± 0.034764 (W 0.00583272) | 8.39813 ± 0.0114108 (W 0.0173894) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 5.61014 ± 0.0199917 (W 0.0129112) | 7.53369 ± 0.0444535 (W 0.0358353) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.279152 ± 0.00235261 (W 0.00180853) | 0.557182 ± 0.00151962 (W 0.022083) | 0.277567 ± 0.00635767 (W 0.00454104) | 🔵 faer-fork 2.00× faster than `faer-upstream` | · |  |
| `lu/n255/b1 [api-inclusive]` | 0.582935 ± 0.00214395 (W 0.00230121) | 0.61761 ± 0.0240627 (W 0.0243278) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n257/b1` | 0.290992 ± 0.00133682 (W 0.00190954) | 0.561305 ± 0.0323901 (W 0.0282421) | 0.28576 ± 0.000629872 (W 0.00183319) | 🔵 faer-fork 1.93× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `lu/n257/b1 [api-inclusive]` | 0.622873 ± 0.00387383 (W 0.00299429) | 0.607372 ± 0.0235746 (W 0.0248009) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n511/b1` | 1.13112 ± 0.00431549 (W 0.00985215) | 2.06168 ± 0.0158963 (W 0.0383769) | 1.00832 ± 0.00589879 (W 0.00570871) | 🔵 faer-fork 1.82× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 2.64072 ± 0.021976 (W 0.00602866) | 2.87196 ± 0.0762744 (W 0.0514919) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n513/b1` | 1.16935 ± 0.00336125 (W 0.0048065) | 2.08928 ± 0.0115968 (W 0.0498878) | 1.03602 ± 0.00951334 (W 0.006032) | 🔵 faer-fork 1.79× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.63074 ± 0.028991 (W 0.00459202) | 2.18363 ± 0.0377121 (W 0.052953) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.794885 ± 0.00304791 (W 0.00165334) | 2.0123 ± 0.0202458 (W 0.010145) | 0.979251 ± 0.00423975 (W 0.0034686) | 🔵 faer-fork 2.53× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 1.28501 ± 0.00412401 (W 0.00494689) | 2.1227 ± 0.00541643 (W 0.120268) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 0.814732 ± 0.00394883 (W 0.00192927) | 2.08314 ± 0.0172829 (W 0.0141842) | 1.04737 ± 0.00695311 (W 0.00331723) | 🔵 faer-fork 2.56× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 1.35398 ± 0.0060209 (W 0.00585751) | 2.17797 ± 0.0137324 (W 0.0203189) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 3.11343 ± 0.0156531 (W 0.00801966) | 7.0444 ± 0.0456014 (W 0.0358216) | 4.30494 ± 0.00389112 (W 0.00489203) | 🔵 faer-fork 2.26× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 5.46407 ± 0.0185716 (W 0.0178098) | 7.45382 ± 0.195893 (W 0.671162) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 3.48235 ± 0.010421 (W 0.0070922) | 6.91552 ± 0.0616622 (W 0.0481389) | 4.50552 ± 0.0265839 (W 0.00716379) | 🔵 faer-fork 1.99× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 5.86809 ± 0.0219872 (W 0.00859745) | 7.2975 ± 0.310198 (W 0.0724386) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n255/b1` | 0.564061 ± 0.00283555 (W 0.00287319) | 1.6149 ± 0.00858118 (W 0.00926325) | 0.529473 ± 0.00252165 (W 0.00291779) | 🔵 faer-fork 2.86× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` |  |
| `qr_factor/n257/b1` | 0.57793 ± 0.0153152 (W 0.00280756) | 1.66072 ± 0.0170334 (W 0.00807743) | 0.557588 ± 0.00154527 (W 0.00200936) | 🔵 faer-fork 2.87× faster than `faer-upstream` | · |  |
| `qr_factor/n511/b1` | 1.7338 ± 0.00774402 (W 0.00404564) | 4.76887 ± 0.0185202 (W 0.0284713) | 2.13673 ± 0.00915806 (W 0.00348639) | 🔵 faer-fork 2.75× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 2.03501 ± 0.0308981 (W 0.00424235) | 4.52366 ± 0.023595 (W 0.0478944) | 2.18042 ± 0.0129102 (W 0.00616626) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `solve/n255/b1` | 0.281695 ± 0.000358874 (W 0.00186831) | 0.495826 ± 0.0209561 (W 0.0217117) | 0.26647 ± 0.00818108 (W 0.0026487) | 🔵 faer-fork 1.76× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n255/b1 [api-inclusive]` | 0.593799 ± 0.00300571 (W 0.00222639) | 0.586683 ± 0.0206654 (W 0.0192302) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n257/b1` | 0.292016 ± 0.00375809 (W 0.00227337) | 0.51361 ± 0.0235307 (W 0.0253352) | 0.272688 ± 0.000480429 (W 0.00248816) | 🔵 faer-fork 1.76× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n257/b1 [api-inclusive]` | 0.629801 ± 0.00850585 (W 0.00436285) | 0.529065 ± 0.0282208 (W 0.0288131) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n511/b1` | 1.09382 ± 0.00298788 (W 0.00267563) | 1.88368 ± 0.0283841 (W 0.051653) | 0.968765 ± 0.000367935 (W 0.00472056) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `solve/n511/b1 [api-inclusive]` | 2.59172 ± 0.0304769 (W 0.00535174) | 2.59255 ± 0.0649592 (W 0.0517117) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n513/b1` | 1.13542 ± 0.00422486 (W 0.00267357) | 1.96127 ± 0.0333889 (W 0.0501172) | 1.00265 ± 0.00735804 (W 0.0073589) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `solve/n513/b1 [api-inclusive]` | 2.59143 ± 0.0111225 (W 0.00292681) | 2.16837 ± 0.0275078 (W 0.0489835) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.18766 ± 0.0186869 (W 0.0115361) | 13.554 ± 0.0109061 (W 0.0199185) | 8.94955 ± 0.0244523 (W 0.0147726) | 🔵 faer-fork 1.89× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.2208 ± 0.00215527 (W 0.0288625) | 13.6312 ± 0.0129842 (W 0.017877) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.35631 ± 0.0108597 (W 0.00782986) | 10.1508 ± 0.0717037 (W 0.0197791) | 9.22596 ± 0.0353646 (W 0.0267213) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 7.37287 ± 0.0225705 (W 0.00580308) | 10.1898 ± 0.0947201 (W 0.0238132) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.3436 ± 0.0637373 (W 0.0349462) | 50.7975 ± 0.0702384 (W 0.0715936) | 32.1817 ± 0.183598 (W 0.0634016) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 29.4065 ± 0.0243026 (W 0.0335921) | 50.771 ± 0.0569405 (W 0.170041) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.7299 ± 0.0903747 (W 0.0603898) | 37.7022 ± 0.083454 (W 0.103381) | 32.8764 ± 0.0903576 (W 0.0498396) | 🔵 faer-fork 1.27× faster than `faer-upstream` | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 29.7716 ± 0.0356381 (W 0.0197381) | 37.7807 ± 0.063883 (W 0.0779535) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.84732 ± 0.0315846 (W 0.00493871) | 10.7964 ± 0.0185277 (W 0.0148828) | 5.01354 ± 0.0123573 (W 0.00693601) | 🔵 faer-fork 2.81× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 3.86305 ± 0.0223133 (W 0.00467365) | 10.7758 ± 0.028504 (W 0.0120811) | - | 🔵 faer-fork 2.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.98294 ± 0.0502458 (W 0.00611688) | 8.25564 ± 0.0604363 (W 0.0098028) | 5.09662 ± 0.00776394 (W 0.00338029) | 🔵 faer-fork 2.07× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 3.96291 ± 0.0205194 (W 0.010016) | 8.28647 ± 0.0249013 (W 0.0213662) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.3526 ± 0.0805479 (W 0.00894358) | 38.964 ± 0.110541 (W 0.0975337) | 16.3144 ± 0.0428152 (W 0.0212901) | 🔵 faer-fork 2.54× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 15.3457 ± 0.0212982 (W 0.00801072) | 38.9231 ± 0.0542748 (W 0.048121) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.4381 ± 0.0209397 (W 0.0139569) | 27.7401 ± 0.0668955 (W 0.0382641) | 16.3658 ± 0.0571199 (W 0.0312355) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 15.4465 ± 0.0199912 (W 0.0235901) | 27.6852 ± 0.0848731 (W 0.0636865) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.014758 ± 0.000164652 (W 0.000197922) | 0.012254 ± 4.90951e-05 (W 9.78264e-05) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.014758 ± 4.46132e-05 (W 0.000247609) | 0.012533 ± 4.83873e-05 (W 3.97492e-05) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.035616 ± 0.000577623 (W 0.000278397) | 0.044093 ± 0.00130401 (W 0.00022298) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n513/b1` | 0.03567 ± 0.000241808 (W 0.000293062) | 0.033051 ± 0.000866216 (W 0.000221087) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.50301 ± 0.00237706 (W 0.00187257) | 1.6639 ± 0.00337085 (W 0.0110546) | - | 🔵 faer-fork 3.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.572509 ± 0.00279906 (W 0.00204348) | 0.744691 ± 0.010074 (W 0.00362869) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.462967 ± 0.00102322 (W 0.00211674) | 1.66117 ± 0.0181753 (W 0.0069681) | - | 🔵 faer-fork 3.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.523779 ± 0.00218295 (W 0.00167469) | 0.705282 ± 0.00529121 (W 0.0144307) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.29372 ± 0.00326473 (W 0.00198139) | 4.9941 ± 0.106012 (W 0.138162) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 2.79579 ± 0.00232281 (W 0.00415841) | 3.16836 ± 0.0812385 (W 0.122579) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.2383 ± 0.002245 (W 0.00339545) | 5.05763 ± 0.0712344 (W 0.105177) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 2.72807 ± 0.00257449 (W 0.00415088) | 3.04971 ± 0.0799766 (W 0.0436797) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.28362 ± 0.0129202 (W 0.012835) | 11.6928 ± 0.0286188 (W 0.00861944) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 8.3283 ± 0.0198101 (W 0.0105718) | 11.7643 ± 0.0723776 (W 0.0173501) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.33087 ± 0.00830796 (W 0.00662164) | 11.6063 ± 0.0342264 (W 0.0150089) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 8.35537 ± 0.0146066 (W 0.0152913) | 11.6476 ± 0.0192569 (W 0.00512574) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.4207 ± 0.0394375 (W 0.0271565) | 45.1213 ± 0.0593772 (W 0.0412914) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 40.5586 ± 0.0469312 (W 0.0317646) | 45.2889 ± 0.0464192 (W 0.0358547) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.4182 ± 0.0184994 (W 0.0353837) | 45.6604 ± 0.0277481 (W 0.0363634) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 40.5062 ± 0.0737587 (W 0.0344458) | 45.9538 ± 0.0703379 (W 0.0343838) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.97746 ± 0.0128051 (W 0.00293901) | 4.29202 ± 0.0281239 (W 0.00405934) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 4.09814 ± 0.014279 (W 0.0154719) | 5.76974 ± 0.011124 (W 0.0144718) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.03147 ± 0.011876 (W 0.00506559) | 4.42352 ± 0.0151017 (W 0.00559742) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.72961 ± 0.00288006 (W 0.00711603) | 4.38703 ± 0.0201269 (W 0.0086704) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 11.0982 ± 0.0428315 (W 0.0101834) | 19.1725 ± 0.0299443 (W 0.0161909) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 20.656 ± 0.130562 (W 0.0356416) | 24.914 ± 0.0865784 (W 0.0123449) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 11.3036 ± 0.0554855 (W 0.0133232) | 19.2985 ± 0.0527885 (W 0.0330491) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 15.1044 ± 0.0107222 (W 0.0273332) | 19.161 ± 0.096777 (W 0.04416) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.95869 ± 0.00249968 (W 0.00470728) | 1.32813 ± 0.00229244 (W 0.0135554) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.73594 ± 0.0084238 (W 0.00873642) | 1.32996 ± 0.0156439 (W 0.0159413) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.0343 ± 0.00538 (W 0.00270498) | 1.32813 ± 0.0126285 (W 0.0188115) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.74983 ± 0.010266 (W 0.0055552) | 1.24571 ± 0.0142495 (W 0.0139705) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 5.06705 ± 0.000876474 (W 0.0101776) | 6.25287 ± 0.0634304 (W 0.0675657) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 9.31409 ± 0.0317551 (W 0.0401872) | 6.68933 ± 0.0627951 (W 0.108518) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 5.33492 ± 0.0184042 (W 0.0134549) | 6.38853 ± 0.125356 (W 0.0765975) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 9.36553 ± 0.0780851 (W 0.0318607) | 6.4576 ± 0.0619298 (W 0.111243) | - | 🔴 faer-fork 1.45× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 3.05502 ± 0.00685031 (W 0.00331822) | 4.94643 ± 0.0171923 (W 0.0087475) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 4.19571 ± 0.00589841 (W 0.00784338) | 5.09018 ± 0.0166549 (W 0.023407) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.18173 ± 0.00400084 (W 0.00323105) | 5.10905 ± 0.0180389 (W 0.0110301) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 4.31459 ± 0.0299211 (W 0.011589) | 5.25894 ± 0.0263247 (W 0.0356843) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 19.7102 ± 0.0453189 (W 0.0166783) | 25.8262 ± 0.0261213 (W 0.0366265) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 25.4927 ± 0.0627136 (W 0.0332381) | 26.7977 ± 0.0776293 (W 0.0310696) | - | · | - | `*` one row has no measurement here |
| `qr/n513/b1` | 20.1528 ± 0.00789496 (W 0.0197134) | 27.2826 ± 0.0741235 (W 0.0474471) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 25.9158 ± 0.0487033 (W 0.0471087) | 27.4803 ± 0.0446423 (W 0.0288002) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.60752 ± 0.00300497 (W 0.00260045) | 3.47177 ± 0.0142013 (W 0.0117245) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.67374 ± 0.00467423 (W 0.00275704) | 3.57527 ± 0.0149768 (W 0.0132011) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 9.31038 ± 0.0172272 (W 0.0163461) | 15.652 ± 0.0302184 (W 0.035341) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 9.55971 ± 0.0270809 (W 0.00755632) | 16.7406 ± 0.0545906 (W 0.0383695) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.871017 ± 0.00230795 (W 0.00194235) | 1.32 ± 0.0037474 (W 0.0121082) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.74562 ± 0.00639243 (W 0.0120089) | 1.22697 ± 0.00879119 (W 0.0179316) | - | 🔴 faer-fork 1.42× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.03683 ± 0.00224826 (W 0.00268829) | 1.33194 ± 0.00809527 (W 0.0201346) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.7574 ± 0.0015262 (W 0.00981279) | 1.23479 ± 0.0092439 (W 0.0210705) | - | 🔴 faer-fork 1.42× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 5.0792 ± 0.017607 (W 0.0582103) | 6.3366 ± 0.056578 (W 0.0823015) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 9.33438 ± 0.0656953 (W 0.065123) | 6.66698 ± 0.0909891 (W 0.0620022) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 5.33422 ± 0.0192457 (W 0.0187541) | 6.38711 ± 0.0670168 (W 0.0879601) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 9.44296 ± 0.0181241 (W 0.0230522) | 6.61043 ± 0.0420909 (W 0.0678822) | - | 🔴 faer-fork 1.43× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.6392 ± 0.0619444 (W 0.0314295) | 22.332 ± 0.184297 (W 0.034604) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 12.7331 ± 0.0745564 (W 0.0398502) | 22.3212 ± 0.0729566 (W 0.0287246) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 12.9165 ± 0.0265621 (W 0.0420095) | 17.935 ± 0.159949 (W 0.0235212) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 12.9428 ± 0.0118703 (W 0.033868) | 17.8471 ± 0.0207014 (W 0.0348531) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 67.7513 ± 0.0844728 (W 0.12678) | 87.0279 ± 0.0397304 (W 0.0386956) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 67.9746 ± 0.0989943 (W 0.0345517) | 89.2686 ± 0.205411 (W 1.02685) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 69.1284 ± 0.227455 (W 0.134188) | 76.1399 ± 0.128547 (W 0.423749) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 69.2915 ± 0.243764 (W 0.23207) | 78.1783 ± 0.254679 (W 1.11204) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.04829 ± 0.0388891 (W 0.0261848) | 16.9457 ± 0.185985 (W 0.0279262) | - | 🔵 faer-fork 2.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 6.06095 ± 0.00300797 (W 0.029165) | 17.1932 ± 0.0191891 (W 0.0263319) | - | 🔵 faer-fork 2.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.21999 ± 0.036555 (W 0.0295932) | 13.6658 ± 0.0222983 (W 0.0217593) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 6.21646 ± 0.021164 (W 0.0401567) | 13.3735 ± 0.154351 (W 0.0182576) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 32.9733 ± 0.0951182 (W 0.113307) | 58.0038 ± 0.0498957 (W 0.0909329) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 32.9234 ± 0.0685085 (W 0.0712854) | 57.8734 ± 0.0749001 (W 0.0532888) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 33.4007 ± 0.203218 (W 0.104611) | 48.3258 ± 0.16198 (W 0.0959148) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 33.5648 ± 0.138987 (W 0.146188) | 48.442 ± 0.134781 (W 0.0275037) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.021861 ± 0.00013482 (W 0.000122956) | 0.023407 ± 1.07858e-05 (W 7.4364e-05) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.022104 ± 0.000320763 (W 0.000167368) | 0.023884 ± 9.91682e-05 (W 7.70065e-05) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.055444 ± 0.00175392 (W 0.000298559) | 0.085991 ± 0.000872136 (W 0.000376216) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n513/b1` | 0.055434 ± 0.00104195 (W 0.00030638) | 0.061142 ± 0.000675068 (W 0.000499109) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.430545 ± 0.00337188 (W 0.00149661) | 1.09754 ± 0.015174 (W 0.14175) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.506914 ± 0.000505082 (W 0.00298303) | 0.709597 ± 0.0202511 (W 0.0137097) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1` | 0.393014 ± 0.00053032 (W 0.00151321) | 1.07071 ± 0.00397375 (W 0.125148) | - | 🔵 faer-fork 2.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.460167 ± 0.00464926 (W 0.00209257) | 0.664197 ± 0.00819866 (W 0.0159671) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.55846 ± 0.00355403 (W 0.00257059) | 3.2873 ± 0.0129151 (W 0.156802) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 2.11352 ± 0.00598916 (W 0.008121) | 2.4861 ± 0.028273 (W 0.0255052) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.48744 ± 0.00463887 (W 0.00282631) | 3.27706 ± 0.139709 (W 0.190348) | - | 🔵 faer-fork 2.20× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n513/b1 [api-inclusive]` | 2.02614 ± 0.0112117 (W 0.00681152) | 2.34469 ± 0.00610327 (W 0.0535262) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.54748 ± 0.0186553 (W 0.00954058) | 11.5168 ± 0.0773241 (W 0.0484871) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.56724 ± 0.00477323 (W 0.0334239) | 11.5701 ± 0.0438402 (W 0.0358242) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.51229 ± 0.0606148 (W 0.00829611) | 11.345 ± 0.0404363 (W 0.0350332) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.54307 ± 0.0105637 (W 0.00556419) | 11.3699 ± 0.0458358 (W 0.038566) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.5089 ± 0.0502349 (W 0.0635222) | 39.131 ± 0.217402 (W 0.193314) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 25.5635 ± 0.0930307 (W 0.0124713) | 39.2293 ± 0.413804 (W 0.238174) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.9378 ± 0.00693903 (W 0.0229409) | 39.4021 ± 0.0868668 (W 0.296516) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 26.0748 ± 0.0408571 (W 0.0388165) | 39.61 ± 0.117993 (W 0.251656) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.5882 ± 0.00989297 (W 0.00544186) | 4.31796 ± 0.0370052 (W 0.0057878) | - | 🔵 faer-fork 2.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.74718 ± 0.00765184 (W 0.00688336) | 5.76621 ± 0.0484332 (W 0.0138441) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.62421 ± 0.0117037 (W 0.00428204) | 4.43647 ± 0.015955 (W 0.00416689) | - | 🔵 faer-fork 2.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.36774 ± 0.00634243 (W 0.00467553) | 4.37809 ± 0.0382064 (W 0.00754133) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.2389 ± 0.00956998 (W 0.00723457) | 16.7708 ± 0.04074 (W 0.0209572) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 17.0893 ± 0.0670132 (W 0.0436055) | 22.5308 ± 0.199106 (W 2.60395) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.38425 ± 0.0110789 (W 0.0309661) | 16.8114 ± 0.0885472 (W 0.167601) | - | 🔵 faer-fork 2.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 11.4196 ± 0.00309272 (W 0.0178773) | 16.7216 ± 0.109583 (W 0.0184687) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.824181 ± 0.000931106 (W 0.00316335) | 1.26156 ± 0.0172063 (W 0.0139374) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.56851 ± 0.00362916 (W 0.0035047) | 1.2516 ± 0.00755084 (W 0.00806575) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.811642 ± 0.00396179 (W 0.00371821) | 1.21368 ± 0.00989197 (W 0.0216165) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.5834 ± 0.0100843 (W 0.00483931) | 1.08875 ± 0.00935355 (W 0.0160914) | - | 🔴 faer-fork 1.45× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.36855 ± 0.0121875 (W 0.00702047) | 4.97886 ± 0.0548123 (W 0.0409749) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 7.76796 ± 0.0202891 (W 0.0309413) | 5.21265 ± 0.0128256 (W 0.0438449) | - | 🔴 faer-fork 1.49× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.48833 ± 0.0154917 (W 0.00875578) | 4.97666 ± 0.00320768 (W 0.045073) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 7.79563 ± 0.114938 (W 0.0194929) | 4.92499 ± 0.0390916 (W 0.02289) | - | 🔴 faer-fork 1.58× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.06434 ± 0.00376975 (W 0.00329936) | 4.89389 ± 0.015528 (W 0.0294488) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 3.27674 ± 0.0233793 (W 0.00916833) | 4.22838 ± 0.200391 (W 0.316996) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n257/b1` | 2.16339 ± 0.00674454 (W 0.00888337) | 5.00611 ± 0.0296032 (W 0.0389905) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 3.34462 ± 0.0103385 (W 0.0117651) | 4.32327 ± 0.135355 (W 0.335459) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n511/b1` | 10.4433 ± 0.017575 (W 0.0201498) | 21.6327 ± 0.0921221 (W 0.114132) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 16.6172 ± 0.110439 (W 0.0470845) | 18.5791 ± 0.682037 (W 1.18034) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1` | 11.2842 ± 0.00977962 (W 0.0517473) | 21.7542 ± 0.118111 (W 0.333519) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 17.4953 ± 0.143134 (W 0.092735) | 22.6106 ± 0.60065 (W 1.92417) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n255/b1` | 1.29462 ± 0.00472837 (W 0.00463372) | 3.58598 ± 0.0138215 (W 0.0188884) | - | 🔵 faer-fork 2.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.34816 ± 0.00163426 (W 0.00512409) | 3.64443 ± 0.00495871 (W 0.015009) | - | 🔵 faer-fork 2.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 5.02871 ± 0.00837411 (W 0.0110181) | 14.1255 ± 0.0217617 (W 0.100214) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 5.68857 ± 0.00696805 (W 0.0211328) | 14.3894 ± 0.0298448 (W 0.154002) | - | 🔵 faer-fork 2.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.737461 ± 0.00299896 (W 0.00222486) | 1.24697 ± 0.016751 (W 0.0170445) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.57787 ± 0.00547372 (W 0.00564329) | 1.12469 ± 0.0019773 (W 0.0134181) | - | 🔴 faer-fork 1.40× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.814544 ± 0.00311077 (W 0.00224066) | 1.20711 ± 0.0143175 (W 0.0196147) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.6025 ± 0.00433532 (W 0.0039611) | 1.08684 ± 0.00390613 (W 0.0115216) | - | 🔴 faer-fork 1.47× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.34837 ± 0.00518244 (W 0.00677304) | 5.04051 ± 0.027793 (W 0.0448265) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 7.8132 ± 0.129202 (W 0.0205562) | 5.16752 ± 0.0112058 (W 0.0449361) | - | 🔴 faer-fork 1.51× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.46866 ± 0.000883815 (W 0.0114344) | 5.01512 ± 0.00678607 (W 0.0336219) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 7.88228 ± 0.076852 (W 0.0348848) | 4.97471 ± 0.022517 (W 0.0463417) | - | 🔴 faer-fork 1.58× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.459 ± 0.0281386 (W 0.0302724) | 21.6844 ± 0.185653 (W 0.0400353) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 10.495 ± 0.00189194 (W 0.0175359) | 21.7463 ± 0.0497255 (W 0.0925922) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.7262 ± 0.0203644 (W 0.0181687) | 17.0451 ± 0.0275419 (W 0.0676437) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 10.7914 ± 0.0151359 (W 0.0312388) | 17.1312 ± 0.0790855 (W 0.0868946) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 48.6144 ± 0.171487 (W 0.123888) | 73.9566 ± 0.153217 (W 0.433318) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 48.7307 ± 0.149845 (W 0.11118) | 74.3648 ± 0.249272 (W 0.846586) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 49.3992 ± 0.0610576 (W 0.0720683) | 62.1766 ± 0.240229 (W 0.461544) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 49.7743 ± 0.159225 (W 0.0927171) | 62.9283 ± 0.128208 (W 0.357691) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.51314 ± 0.00619618 (W 0.0153836) | 17.1667 ± 0.199221 (W 0.0412947) | - | 🔵 faer-fork 3.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 5.51649 ± 0.0207185 (W 0.0208489) | 17.2399 ± 0.217444 (W 0.0277513) | - | 🔵 faer-fork 3.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.6708 ± 0.00694831 (W 0.0209298) | 13.6473 ± 0.246782 (W 0.0226662) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 5.65351 ± 0.0174885 (W 0.0141059) | 13.714 ± 0.301836 (W 0.0316453) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 25.5365 ± 0.0369694 (W 0.0915571) | 51.4696 ± 0.23262 (W 0.0463206) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 25.4531 ± 0.120142 (W 0.0925458) | 51.41 ± 0.234424 (W 0.0348134) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 25.7532 ± 0.130153 (W 0.0345318) | 41.7235 ± 0.166464 (W 0.0640135) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 25.8359 ± 0.0816574 (W 0.0500068) | 41.6891 ± 0.0901155 (W 0.0402674) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.021472 ± 5.10686e-05 (W 0.000113623) | 0.023544 ± 8.14269e-05 (W 0.00124732) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.02148 ± 8.95005e-05 (W 0.000146048) | 0.023954 ± 0.000139861 (W 0.000522773) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.050254 ± 0.00111598 (W 0.000168577) | 0.086127 ± 0.00043821 (W 0.0020856) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.049802 ± 0.000504333 (W 0.000190854) | 0.058002 ± 0.00043646 (W 0.000329972) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.494203 ± 0.00255917 (W 0.0026007) | 0.652308 ± 0.00138006 (W 0.00585209) | 0.442457 ± 0.00112203 (W 0.0063693) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.569934 ± 0.00401328 (W 0.00205588) | 0.73412 ± 0.00680988 (W 0.00261016) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.448778 ± 0.00314954 (W 0.00224027) | 0.64249 ± 0.00244079 (W 0.00922133) | 0.450311 ± 0.00844681 (W 0.00912621) | 🔵 faer-fork 1.43× faster than `faer-upstream` | · |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.522055 ± 0.000740073 (W 0.00151498) | 0.702697 ± 0.00831838 (W 0.00529235) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.25672 ± 0.00719279 (W 0.00339864) | 2.71403 ± 0.00806638 (W 0.0769397) | 2.47777 ± 0.0141342 (W 0.0121047) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 2.59231 ± 0.00156316 (W 0.00452847) | 3.09724 ± 0.0106165 (W 0.119204) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.21128 ± 0.00439839 (W 0.00595334) | 2.65221 ± 0.0187275 (W 0.142763) | 2.55247 ± 0.00948544 (W 0.00941341) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 2.52982 ± 0.0046687 (W 0.004083) | 2.98454 ± 0.0720658 (W 0.0813602) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.2854 ± 0.00869092 (W 0.0101425) | 11.6905 ± 0.0337823 (W 0.00789771) | 9.34273 ± 0.0123016 (W 0.0104536) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 8.30286 ± 0.0098046 (W 0.0105666) | 11.7419 ± 0.0750351 (W 0.0195839) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.31419 ± 0.0146975 (W 0.00906864) | 11.6011 ± 0.0549273 (W 0.00736984) | 9.49099 ± 0.00580389 (W 0.0130634) | 🔵 faer-fork 1.40× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `eigh/n257/b1 [api-inclusive]` | 8.31938 ± 0.0169468 (W 0.0120835) | 11.6244 ± 0.0386693 (W 0.00930048) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.23 ± 0.0414751 (W 0.0338332) | 45.0168 ± 0.105973 (W 0.0411563) | 49.4962 ± 0.0317258 (W 0.0614878) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 40.3317 ± 0.0154345 (W 0.0515952) | 45.1619 ± 0.0969478 (W 0.0478262) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.1013 ± 0.0306385 (W 0.0156091) | 45.6264 ± 0.0356082 (W 0.0482012) | 49.7803 ± 0.0474179 (W 0.0810818) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 40.204 ± 0.0392599 (W 0.0434863) | 45.8116 ± 0.0656959 (W 0.0282048) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.86461 ± 0.00311557 (W 0.00503401) | 4.32165 ± 0.00463049 (W 0.0121808) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.93263 ± 0.0140892 (W 0.00860972) | 5.74726 ± 0.0232681 (W 0.01177) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.90457 ± 0.00429771 (W 0.00291212) | 4.40608 ± 0.0540017 (W 0.00640707) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.54964 ± 0.0128886 (W 0.00850272) | 4.37414 ± 0.0215134 (W 0.00913663) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 10.6926 ± 0.0530587 (W 0.0174296) | 19.1018 ± 0.0767526 (W 0.02361) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 19.6575 ± 0.0831091 (W 0.0473218) | 24.6215 ± 0.0610251 (W 0.120518) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 10.8236 ± 0.0727015 (W 0.0249361) | 19.2993 ± 0.0680716 (W 0.021175) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 13.8803 ± 0.0492923 (W 0.0169842) | 18.8389 ± 0.0327729 (W 0.0311035) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.875645 ± 0.00378835 (W 0.00631211) | 1.13909 ± 0.00638505 (W 0.0156385) | 0.791288 ± 0.00180672 (W 0.00307798) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔴 faer-fork 1.11× slower than `lapack-mkl` |  |
| `lu/n255/b1 [api-inclusive]` | 1.56243 ± 0.0101459 (W 0.0124958) | 1.34552 ± 0.0124306 (W 0.0195165) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.916402 ± 0.00155944 (W 0.00343789) | 1.14819 ± 0.010258 (W 0.0168116) | 0.833023 ± 0.00264631 (W 0.00296919) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔴 faer-fork 1.10× slower than `lapack-mkl` |  |
| `lu/n257/b1 [api-inclusive]` | 1.58594 ± 0.0104152 (W 0.00336335) | 1.24336 ± 0.011819 (W 0.0141179) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 4.61568 ± 0.00483027 (W 0.00863985) | 5.60465 ± 0.0266796 (W 0.0642009) | 4.81743 ± 0.0132972 (W 0.0121318) | 🔵 faer-fork 1.21× faster than `faer-upstream` | · |  |
| `lu/n511/b1 [api-inclusive]` | 8.17343 ± 0.0258944 (W 0.057148) | 6.34258 ± 0.0790123 (W 0.113462) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 4.81572 ± 0.00611308 (W 0.00279395) | 5.72171 ± 0.050528 (W 0.0534959) | 4.91483 ± 0.00874795 (W 0.0094107) | 🔵 faer-fork 1.19× faster than `faer-upstream` | · |  |
| `lu/n513/b1 [api-inclusive]` | 8.2393 ± 0.0241513 (W 0.0100934) | 6.22833 ± 0.0620573 (W 0.112854) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.95993 ± 0.0074202 (W 0.00319078) | 4.03805 ± 0.0188574 (W 0.00458342) | 3.98371 ± 0.0186971 (W 0.00904812) | 🔵 faer-fork 1.36× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 4.04576 ± 0.0103437 (W 0.0103249) | 5.0407 ± 0.0150569 (W 0.00630855) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.04619 ± 0.00278309 (W 0.0017906) | 4.11823 ± 0.00681764 (W 0.00498699) | 4.12307 ± 0.0224648 (W 0.0090196) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 4.12733 ± 0.0153417 (W 0.0061489) | 5.18979 ± 0.018127 (W 0.0091812) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 18.968 ± 0.0237958 (W 0.0282023) | 21.092 ± 0.0596305 (W 0.0415517) | 25.3438 ± 0.05875 (W 0.0466698) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 24.3365 ± 0.049368 (W 0.0327352) | 26.484 ± 0.0960772 (W 0.0555692) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 19.3464 ± 0.0121341 (W 0.0194291) | 21.4813 ± 0.0111693 (W 0.0182496) | 25.7413 ± 0.0704397 (W 0.0239315) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 24.8621 ± 0.0713804 (W 0.0427976) | 27.1702 ± 0.0357878 (W 0.156338) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.5192 ± 0.00265964 (W 0.00175448) | 2.59435 ± 0.0079407 (W 0.00650393) | 1.53208 ± 0.00484702 (W 0.00253596) | 🔵 faer-fork 1.71× faster than `faer-upstream` | · |  |
| `qr_factor/n257/b1` | 1.56162 ± 0.00173494 (W 0.00557475) | 2.63138 ± 0.00433187 (W 0.0075063) | 1.58998 ± 0.00734021 (W 0.00591924) | 🔵 faer-fork 1.69× faster than `faer-upstream` | · |  |
| `qr_factor/n511/b1` | 8.84568 ± 0.029231 (W 0.0145023) | 10.8499 ± 0.0175064 (W 0.00725688) | 10.5896 ± 0.0616973 (W 0.019526) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 8.95848 ± 0.00368369 (W 0.00823505) | 11.0902 ± 0.0314334 (W 0.0100664) | 10.7569 ± 0.0266125 (W 0.0105592) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `solve/n255/b1` | 0.875855 ± 0.00446997 (W 0.0134748) | 1.13157 ± 0.00172083 (W 0.0129512) | 0.75597 ± 0.00496312 (W 0.0032895) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔴 faer-fork 1.16× slower than `lapack-mkl` |  |
| `solve/n255/b1 [api-inclusive]` | 1.56362 ± 0.00396175 (W 0.0113274) | 1.23578 ± 0.00259853 (W 0.00847528) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.92118 ± 0.00518121 (W 0.00198327) | 1.16921 ± 0.00623505 (W 0.0132698) | 0.787601 ± 0.00442864 (W 0.00231241) | 🔵 faer-fork 1.27× faster than `faer-upstream` | 🔴 faer-fork 1.17× slower than `lapack-mkl` |  |
| `solve/n257/b1 [api-inclusive]` | 1.60425 ± 0.00530548 (W 0.00322852) | 1.242 ± 0.0123819 (W 0.0159237) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 4.60194 ± 0.00177922 (W 0.0173144) | 5.70979 ± 0.0465792 (W 0.0712655) | 4.74199 ± 0.00637441 (W 0.00772082) | 🔵 faer-fork 1.24× faster than `faer-upstream` | · |  |
| `solve/n511/b1 [api-inclusive]` | 8.22946 ± 0.0468902 (W 0.0547608) | 6.2646 ± 0.0589583 (W 0.0987437) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 4.8047 ± 0.00731823 (W 0.00619022) | 5.75233 ± 0.00749086 (W 0.0822204) | 4.83533 ± 0.0343783 (W 0.00693556) | 🔵 faer-fork 1.20× faster than `faer-upstream` | · |  |
| `solve/n513/b1 [api-inclusive]` | 8.31891 ± 0.0385891 (W 0.0176587) | 6.28086 ± 0.036031 (W 0.0476791) | - | 🔴 faer-fork 1.32× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.5156 ± 0.0428422 (W 0.0453953) | 22.2764 ± 0.113496 (W 0.0238313) | 15.5287 ± 0.0562715 (W 0.0243628) | 🔵 faer-fork 1.78× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 12.5646 ± 0.0513907 (W 0.0402513) | 22.3 ± 0.0979353 (W 0.0703957) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 12.872 ± 0.0511413 (W 0.0244136) | 17.6726 ± 0.154356 (W 0.0233693) | 15.8969 ± 0.0202356 (W 0.0153307) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 12.8894 ± 0.0368447 (W 0.0239981) | 17.7515 ± 0.0263578 (W 0.0464685) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 67.5554 ± 0.0961161 (W 0.166151) | 86.8888 ± 0.112431 (W 0.0696789) | 82.0744 ± 0.286527 (W 0.055334) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 67.7018 ± 0.18811 (W 0.125797) | 88.983 ± 0.143938 (W 1.14724) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 68.9689 ± 0.071322 (W 0.144626) | 75.9636 ± 0.118179 (W 0.0969726) | 82.7731 ± 0.189177 (W 0.142928) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 69.1452 ± 0.0892388 (W 0.0661126) | 77.9546 ± 0.155383 (W 0.961988) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.097 ± 0.0205979 (W 0.0345406) | 17.093 ± 0.0776256 (W 0.0315127) | 7.48748 ± 0.0193573 (W 0.018628) | 🔵 faer-fork 2.80× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 6.05571 ± 0.021489 (W 0.0379812) | 17.0292 ± 0.0787481 (W 0.0428174) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.26 ± 0.0411372 (W 0.0245446) | 13.5968 ± 0.173915 (W 0.0200248) | 7.64688 ± 0.0200613 (W 0.00426575) | 🔵 faer-fork 2.17× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 6.2391 ± 0.0325349 (W 0.0174809) | 13.4421 ± 0.0995874 (W 0.0209361) | - | 🔵 faer-fork 2.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 32.8155 ± 0.0718908 (W 0.0937868) | 57.5515 ± 0.061904 (W 0.0519161) | 39.65 ± 0.193864 (W 0.0302249) | 🔵 faer-fork 1.75× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 32.7936 ± 0.0947974 (W 0.0769015) | 57.5192 ± 0.0317859 (W 0.0564863) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 33.1461 ± 0.0893054 (W 0.0950463) | 48.0814 ± 0.0184238 (W 0.0877443) | 40.1393 ± 0.0966592 (W 0.0411129) | 🔵 faer-fork 1.45× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 33.2821 ± 0.137687 (W 0.121948) | 48.2643 ± 0.0924676 (W 0.038956) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.020869 ± 0.000192887 (W 0.000224946) | 0.024646 ± 0.000114308 (W 0.000115184) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.021583 ± 0.000820989 (W 0.000271947) | 0.024899 ± 0.000150562 (W 0.000105499) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n511/b1` | 0.0589 ± 0.000869951 (W 0.000418075) | 0.088457 ± 0.000563846 (W 0.00153126) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.058256 ± 0.000382076 (W 0.000401433) | 0.06417 ± 0.000490227 (W 0.000430668) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.426567 ± 0.000711874 (W 0.00256488) | 0.642045 ± 0.00228419 (W 0.0210373) | 0.294842 ± 0.00478423 (W 0.00548033) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔴 faer-fork 1.45× slower than `lapack-mkl` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.505983 ± 0.000226637 (W 0.00261737) | 0.720202 ± 0.00242283 (W 0.0200652) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.382543 ± 0.00230112 (W 0.00142191) | 0.585676 ± 0.0103983 (W 0.0136443) | 0.304279 ± 0.000742258 (W 0.00756295) | 🔵 faer-fork 1.53× faster than `faer-upstream` | 🔴 faer-fork 1.26× slower than `lapack-mkl` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.46035 ± 0.00164787 (W 0.00292104) | 0.647166 ± 0.0100607 (W 0.0160902) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.54514 ± 0.00881729 (W 0.00342682) | 2.04078 ± 0.00235887 (W 0.0525922) | 1.3502 ± 0.00672604 (W 0.00955957) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 1.90385 ± 0.00275372 (W 0.00425025) | 2.42412 ± 0.0307771 (W 0.0462403) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.47608 ± 0.00866631 (W 0.00436185) | 1.96872 ± 0.016632 (W 0.0601613) | 1.39413 ± 0.000796785 (W 0.0123725) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 1.82103 ± 0.00300705 (W 0.0037179) | 2.27848 ± 0.0102529 (W 0.0612719) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.52838 ± 0.0178781 (W 0.00766362) | 11.4543 ± 0.0323838 (W 0.0494506) | 7.49037 ± 0.0246151 (W 0.011702) | 🔵 faer-fork 2.07× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 5.56231 ± 0.00739629 (W 0.00592947) | 11.5131 ± 0.0607413 (W 0.0358993) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.50467 ± 0.00641098 (W 0.00684552) | 11.3331 ± 0.00824986 (W 0.0370437) | 7.57727 ± 0.0236234 (W 0.0127612) | 🔵 faer-fork 2.06× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `eigh/n257/b1 [api-inclusive]` | 5.54199 ± 0.00952076 (W 0.00596655) | 11.3596 ± 0.0182907 (W 0.0387929) | - | 🔵 faer-fork 2.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.1554 ± 0.0209241 (W 0.0373997) | 39.2241 ± 0.101981 (W 0.159429) | 34.4107 ± 0.0208933 (W 0.0537115) | 🔵 faer-fork 1.56× faster than `faer-upstream` | 🔵 faer-fork 1.37× faster than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 25.2122 ± 0.0927237 (W 0.0775936) | 39.348 ± 0.12833 (W 0.248482) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.6907 ± 0.0374218 (W 0.0210201) | 39.3718 ± 0.127221 (W 0.282694) | 34.5077 ± 0.0694952 (W 0.0408653) | 🔵 faer-fork 1.53× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 25.8449 ± 0.0113273 (W 0.305862) | 39.489 ± 0.225697 (W 0.293985) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.5068 ± 0.00794989 (W 0.00609298) | 4.30929 ± 0.0133546 (W 0.00439091) | - | 🔵 faer-fork 2.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.58281 ± 0.0145443 (W 0.011432) | 5.71758 ± 0.0343149 (W 0.0121741) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.54097 ± 0.001852 (W 0.00654436) | 4.43985 ± 0.00452421 (W 0.00881942) | - | 🔵 faer-fork 2.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.20493 ± 0.0141679 (W 0.00308742) | 4.36514 ± 0.0268129 (W 0.0114955) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.07331 ± 0.0433624 (W 0.0239465) | 16.6566 ± 0.0723918 (W 0.0106374) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 15.9677 ± 0.115501 (W 0.0294067) | 22.2419 ± 0.0759731 (W 0.0343048) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.09986 ± 0.0412733 (W 0.0102135) | 16.7994 ± 0.0159631 (W 0.0112073) | - | 🔵 faer-fork 2.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 10.2296 ± 0.11627 (W 0.05699) | 16.3073 ± 0.047327 (W 0.0605821) | - | 🔵 faer-fork 1.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.714221 ± 0.000709708 (W 0.00143978) | 1.0324 ± 0.00463713 (W 0.0143781) | 0.557162 ± 0.00543965 (W 0.00273964) | 🔵 faer-fork 1.45× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` |  |
| `lu/n255/b1 [api-inclusive]` | 1.4016 ± 0.00551438 (W 0.00311932) | 1.25378 ± 0.00828123 (W 0.0141237) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.730138 ± 0.000316061 (W 0.00270207) | 1.0098 ± 0.0180627 (W 0.0197316) | 0.571388 ± 0.0019383 (W 0.00433544) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` |  |
| `lu/n257/b1 [api-inclusive]` | 1.41706 ± 0.00376605 (W 0.00311371) | 1.09137 ± 0.00549813 (W 0.0215344) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.12674 ± 0.0744641 (W 0.00555232) | 4.14257 ± 0.0119525 (W 0.0285832) | 2.71365 ± 0.00487304 (W 0.00731658) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔴 faer-fork 1.15× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 6.70429 ± 0.0269322 (W 0.00464408) | 4.93717 ± 0.00795696 (W 0.0178478) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.17475 ± 0.0070195 (W 0.00797299) | 4.17205 ± 0.0251294 (W 0.0334501) | 2.75988 ± 0.0079716 (W 0.0110926) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔴 faer-fork 1.15× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 6.70584 ± 0.0494915 (W 0.0144261) | 4.65466 ± 0.0596392 (W 0.0437424) | - | 🔴 faer-fork 1.44× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.01055 ± 0.00777543 (W 0.00369218) | 4.02095 ± 0.0111993 (W 0.018114) | 2.86738 ± 0.0164604 (W 0.00876046) | 🔵 faer-fork 2.00× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 3.08541 ± 0.0189892 (W 0.010608) | 4.51058 ± 0.279631 (W 0.30325) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n257/b1` | 2.06322 ± 0.0070276 (W 0.00401473) | 4.04321 ± 0.00612379 (W 0.0249284) | 2.99208 ± 0.00708897 (W 0.00977489) | 🔵 faer-fork 1.96× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 3.18423 ± 0.0174871 (W 0.00736763) | 4.46388 ± 0.181875 (W 0.357982) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n511/b1` | 9.92958 ± 0.0141776 (W 0.0123901) | 17.5936 ± 0.272268 (W 0.346816) | 13.1369 ± 0.0226643 (W 0.0296497) | 🔵 faer-fork 1.77× faster than `faer-upstream` | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 15.4048 ± 0.0238368 (W 0.0287049) | 19.1225 ± 0.846537 (W 2.19134) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1` | 10.6516 ± 0.0457805 (W 0.0282316) | 17.5578 ± 0.163613 (W 0.344503) | 13.4302 ± 0.0564245 (W 0.018455) | 🔵 faer-fork 1.65× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 16.2877 ± 0.181255 (W 0.145636) | 23.157 ± 0.235391 (W 0.440065) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.23627 ± 0.0038088 (W 0.00445742) | 2.7444 ± 0.0113174 (W 0.0182998) | 1.57399 ± 0.00422492 (W 0.00274203) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `qr_factor/n257/b1` | 1.27261 ± 0.00502902 (W 0.00426579) | 2.75286 ± 0.00385453 (W 0.0173897) | 1.63451 ± 0.00538452 (W 0.00565263) | 🔵 faer-fork 2.16× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `qr_factor/n511/b1` | 4.70146 ± 0.011252 (W 0.00530116) | 9.92382 ± 0.0842061 (W 0.145479) | 5.37295 ± 0.00986016 (W 0.00624065) | 🔵 faer-fork 2.11× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 5.2445 ± 0.0110453 (W 0.00675945) | 10.08 ± 0.0278428 (W 0.0856619) | 5.52897 ± 0.00828268 (W 0.00671275) | 🔵 faer-fork 1.92× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `solve/n255/b1` | 0.711343 ± 0.0113648 (W 0.00337506) | 1.01369 ± 0.00493646 (W 0.00855584) | 0.56344 ± 0.0523918 (W 0.00667912) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔴 faer-fork 1.26× slower than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n255/b1 [api-inclusive]` | 1.41405 ± 0.0082189 (W 0.00499692) | 1.12641 ± 0.00827158 (W 0.00883623) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.733581 ± 0.0027177 (W 0.00285504) | 1.01044 ± 0.00504255 (W 0.0121846) | 0.573078 ± 0.00279403 (W 0.00372729) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔴 faer-fork 1.28× slower than `lapack-mkl` |  |
| `solve/n257/b1 [api-inclusive]` | 1.43281 ± 0.00525689 (W 0.00355971) | 1.10443 ± 0.012354 (W 0.0130451) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.08391 ± 0.00356365 (W 0.00550336) | 4.16993 ± 0.0288375 (W 0.0183249) | 2.71434 ± 0.00665138 (W 0.0106554) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `solve/n511/b1 [api-inclusive]` | 6.70824 ± 0.0434644 (W 0.0148426) | 4.87878 ± 0.00463085 (W 0.0149445) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.14625 ± 0.00212947 (W 0.00858969) | 4.1409 ± 0.0312245 (W 0.0417581) | 2.7667 ± 0.0101961 (W 0.00818292) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `solve/n513/b1 [api-inclusive]` | 6.72588 ± 0.00848619 (W 0.0214757) | 4.67174 ± 0.0254281 (W 0.0389755) | - | 🔴 faer-fork 1.44× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.4032 ± 0.0447653 (W 0.0213993) | 21.9323 ± 0.0531712 (W 0.0422659) | 11.8679 ± 0.0236159 (W 0.0176944) | 🔵 faer-fork 2.11× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 10.4574 ± 0.0331249 (W 0.0166254) | 21.8333 ± 0.114062 (W 0.0498192) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.6243 ± 0.0480888 (W 0.0256425) | 17.1496 ± 0.251507 (W 0.038826) | 12.1254 ± 0.0255678 (W 0.0310636) | 🔵 faer-fork 1.61× faster than `faer-upstream` | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 10.6794 ± 0.00473005 (W 0.0122152) | 17.1849 ± 0.0614399 (W 0.120477) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 48.3309 ± 0.122406 (W 0.0889804) | 73.6358 ± 0.374242 (W 0.372888) | 54.2261 ± 0.0578346 (W 0.0518062) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 48.4274 ± 0.0297586 (W 0.0761205) | 74.1606 ± 0.413092 (W 0.281549) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 49.4539 ± 0.14496 (W 0.0822395) | 62.0895 ± 0.202582 (W 0.205159) | 54.9661 ± 0.202747 (W 0.403713) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 49.6339 ± 0.0820646 (W 0.324351) | 62.5499 ± 0.153626 (W 0.392235) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.53516 ± 0.0167438 (W 0.0274925) | 17.2585 ± 0.0617284 (W 0.0300637) | 6.06624 ± 0.00505005 (W 0.00773797) | 🔵 faer-fork 3.12× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 5.52925 ± 0.013092 (W 0.019342) | 16.9614 ± 0.0628424 (W 0.0356417) | - | 🔵 faer-fork 3.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.70401 ± 0.0250822 (W 0.0171986) | 13.6741 ± 0.24476 (W 0.0204281) | 6.20693 ± 0.0214294 (W 0.0091734) | 🔵 faer-fork 2.40× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 5.67966 ± 0.0253973 (W 0.0160049) | 13.7007 ± 0.262165 (W 0.0244532) | - | 🔵 faer-fork 2.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 25.173 ± 0.140199 (W 0.0505963) | 51.0923 ± 0.281162 (W 0.385493) | 27.1099 ± 0.0913127 (W 0.0304809) | 🔵 faer-fork 2.03× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 25.2572 ± 0.288262 (W 0.113394) | 50.9965 ± 0.124985 (W 0.0309756) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 25.5345 ± 0.0846428 (W 0.104761) | 41.43 ± 0.186253 (W 0.0527955) | 27.4903 ± 0.0432343 (W 0.0243) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 25.6497 ± 0.0367743 (W 0.0733154) | 41.3994 ± 0.129078 (W 0.122441) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.021201 ± 0.000848299 (W 0.000174888) | 0.024365 ± 7.45945e-05 (W 6.83374e-05) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n257/b1` | 0.02165 ± 0.000570706 (W 0.000656) | 0.024876 ± 0.000180705 (W 7.51152e-05) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n511/b1` | 0.054281 ± 0.000604022 (W 0.000274212) | 0.088271 ± 0.000343959 (W 0.00203148) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.055247 ± 0.00107169 (W 0.000452641) | 0.061664 ± 0.0002295 (W 0.000606609) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |

Not applicable to this population's shapes and reported as skipped: `cholesky/255`, `cholesky/257`, `cholesky/511`, `cholesky/513`, `eigh/255`, `eigh/257`, `eigh/511`, `eigh/513`, `full_piv_lu/255`, `full_piv_lu/257`, `full_piv_lu/511`, `full_piv_lu/513`, `lu/255`, `lu/257`, `lu/511`, `lu/513`, `qr/255`, `qr/257`, `qr/511`, `qr/513`, `qr_factor/255`, `qr_factor/257`, `qr_factor/511`, `qr_factor/513`, `solve/255`, `solve/257`, `solve/511`, `solve/513`, `svd_thin/255`, `svd_thin/257`, `svd_thin/511`, `svd_thin/513`, `svd_values/255`, `svd_values/257`, `svd_values/511`, `svd_values/513`, `triangular_solve/255`, `triangular_solve/257`, `triangular_solve/511`, `triangular_solve/513`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
