# `faer-kernels-large` on `zen5-cpu`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above); `faer-upstream` (the provider in the measured revision; see library and upstream above)
- harness commit: `f018ded7b2199e04b8fb4bc33a15fe2a38ddc670`
- toolkit commit: `b7299cf625c537a37eef2782e8759445f1d8aef8`
- measured binary SHA-256: `f4f9a0ba5346b33e547051da3309d6bfc806a9f1f6e6bff219f2170492031268`
- hardware profile: `zen5-cpu`
- host role: `development` — **provisional**: recorded on a development host, superseded by the measurement host's dataset
- timestamp: `2026-11-18T01:23:16.189189Z`
- timing policy: v2, median of 5 reps, priming 250 ms (time-based, identical on every row)
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts
- command: `/home/shinaoka/projects/tensor4all/colmat-benchmark/scripts/record_run.py zen5-cpu faer-kernels-large --checkout /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/t4a-faer --upstream /home/shinaoka/projects/tensor4all/colmat-benchmark/extern/upstream-faer --vendor mkl --engines faer-fork,faer-upstream,lapack-mkl --reps 5 --aa 3 --prime-ms 250 --label pub-93da6217-20261117-large-mkl --no-publish --threads 4,8 --regimes linalg-large,gemm-large,linalg-boundary --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513`
- raw data: `data/results/zen5-cpu/faer-kernels-large/20261117T180323Z-pub-93da6217-20261117-large-mkl/`

## What was measured

- **Workload** — Direct calls into one faer revision: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: Vendor C-order cells use CBLAS/LAPACKE row-major entry points. Workspace allocation or layout conversion internal to those APIs remains timed; these are not prepared-column-major or transpose measurements.
  Known blind spot: `gemm` covers the tall class's two Householder legs (a 16-wide basis and a 240-column target over 4096 rows: the reduction and the write-back) beside a vendor `?gemm` on the same leg. A single leg beside a vendor gemm is a diagnostic reference, not the equivalent implementation of the whole update, and the row is labelled accordingly.
  Known blind spot: `householder_apply` measures block apply and the one-block sequence form as two distinct rows, labelled with basis width min(m,n) and target width n. QR fixture preparation is outside timing. Independently varied basis/target widths and multi-panel sequences are not covered.
  Known blind spot: Allocation performed inside the measured call stays inside the clock. The fork's two row splits allocate their partial buffers inside the public call, and the public scratch function does not expose them; the campaign reports that as it is instead of changing faer's API to make the boundary look cleaner.
  Known blind spot: `cholesky`, `eigh`, `solve` and `triangular_solve` are measured on square shapes only: the matrix they factor or solve with is square by definition (self-adjoint for the first two, a triangular factor for the last). `lu` and `full_piv_lu` are not gated that way -- they take the declared rectangular shapes too, with `min(m, n)` pivots and a trapezoidal `L` or `U`, which is what `dgetrf` does as well -- so a rectangular LU cell is measured in both orders beside its square one.
  Known blind spot: `f32`/`c32`, the large regime's 1024 shapes, BLIS, sparse, iterative and GPU arms are later declarations, not machinery built now.
  Known blind spot: A vendor arm records the library it linked. Only OpenBLAS reports the kernel its runtime dispatched to; MKL names no kernel and no processor count, so its manifest leaves both empty rather than inferring them.
  Known blind spot: Known limitation: Coverage is partial: f64/c64, the small suite at 1/4/8T, and the 1024 + boundary tier at 4/8T.
- **Rows** — one row per engine in every table below:
  - `faer-fork` — The measured library: the tensor4all fork (`t4a-faer`, `[lib] name = "faer"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/t4a-faer.rev`
    Caveat: Both faer crates declare `[lib] name = "faer"`. They are distinct dependency keys (`faer_fork = { package = "t4a-faer" }`, `faer_upstream = { package = "faer" }`) and therefore distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded. measured at 325e482, the t4a-faer 0.28.0 release source; the four 2026-11-13 zen5-cpu recordings were taken at this revision.
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
- OS / arch: `Linux 7.0.0-38-generic` / `x86_64`
- hostname: `shinaoka-EVO-X1Pro`
- CPU sets: 4T -> `4-7`, 8T -> `4-11`

Every timed cell matches a successful verification census, including operation variants.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 5 chronological calls; 3 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 0.75%, worst 41.63% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 64 of 1968 engine cells exceed 5% and carry `!`; median within-process relative SD 0.35%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
| `cholesky/n1024/b1` | 5.12442 ± 0.0226716 (W 0.00797658) | 7.93249 ± 0.0291107 (W 0.206841) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 5.4774 ± 0.0275224 (W 0.0159627) | 7.98486 ± 0.37921 (W 0.11104) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n1536/b1` | 14.6813 ± 0.064802 (W 0.0246374) | 19.1877 ± 0.209635 (W 0.434979) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 17.0916 ± 0.140768 (W 0.0690998) | 23.9887 ± 0.222548 (W 0.210409) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 120.102 ± 0.311277 (W 0.0839212) | 117.278 ± 0.0451819 (W 0.217961) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 126.899 ± 0.128797 (W 0.161428) | 116.891 ± 0.300132 (W 0.959123) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 327.928 ± 0.673131 (W 0.584913) | 325.827 ± 1.16296 (W 0.72221) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 343.834 ± 0.457205 (W 1.15883) | 342.395 ± 0.592178 (W 2.76156) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 58.9392 ± 0.184465 (W 0.0566634) | 75.9149 ± 0.826864 (W 0.090153) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 67.0796 ± 0.310186 (W 0.121001) | 69.4175 ± 1.18259 (W 0.795312) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 254.32 ± 0.757877 (W 0.979731) | 291.138 ± 4.05085 (W 0.94) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 271.557 ± 1.22162 (W 1.70916) | 277.467 ± 1.42255 (W 5.95664) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 10.7842 ± 0.0305802 (W 0.039583) | 56.7343 ± 2.32087 (W 2.06922) | - | 🔵 faer-fork 5.26× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1024/b1 [api-inclusive]` | 19.1386 ± 0.0232621 (W 0.108479) | 21.3379 ± 0.28188 (W 0.15139) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 31.8745 ± 0.0882274 (W 0.110542) | 144.677 ± 1.62239 (W 4.84126) | - | 🔵 faer-fork 4.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 50.1832 ± 0.476065 (W 0.212757) | 63.7603 ± 0.187234 (W 0.320018) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 39.5391 ± 0.211614 (W 0.103693) | 95.0569 ± 0.0899589 (W 0.567634) | - | 🔵 faer-fork 2.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 50.8449 ± 0.286102 (W 0.40621) | 57.9503 ± 1.46995 (W 0.078017) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 131.801 ± 0.114484 (W 0.518197) | 261.448 ± 3.33095 (W 0.761847) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 155.391 ± 1.06863 (W 0.532371) | 178.158 ± 2.98995 (W 1.76514) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 19.4875 ± 0.088414 (W 0.0410323) | 74.9714 ± 0.624275 (W 0.174661) | - | 🔵 faer-fork 3.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 59.3192 ± 0.251206 (W 0.266627) | 186.429 ± 0.556136 (W 0.638337) | - | 🔵 faer-fork 3.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 10.7235 ± 0.0244948 (W 0.121417) | 40.4271 ± 0.843494 (W 1.37351) | - | 🔵 faer-fork 3.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 18.7598 ± 0.0103874 (W 0.053864) | 20.4637 ± 0.172847 (W 0.114649) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 31.0159 ± 0.106129 (W 0.141415) | 106.748 ± 0.895616 (W 2.25111) | - | 🔵 faer-fork 3.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 35.1152 ± 0.179722 (W 0.202847) | 62.1963 ± 0.169721 (W 0.212645) | - | 🔵 faer-fork 1.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 206.06 ± 0.237708 (W 0.316496) | 275.875 ± 0.868022 (W 0.491255) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 228.626 ± 0.372805 (W 0.931701) | 296.783 ± 0.268415 (W 2.77062) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 702.217 ± 3.68683 (W 1.17359) | 878.831 ± 2.63433 (W 1.14661) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 752.344 ± 3.3631 (W 2.11533) | 920.639 ± 2.1016 (W 4.17294) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 107.773 ± 2.91039 (W 0.327901) | 189.801 ± 1.20894 (W 0.594865) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `svd_values/n1024/b1 [api-inclusive]` | 108.198 ± 1.55532 (W 0.104603) | 192.997 ± 3.51495 (W 0.462644) | - | 🔵 faer-fork 1.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 406.272 ± 1.96926 (W 0.939382) | 611.893 ± 0.471337 (W 1.32954) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 417.143 ± 2.85942 (W 1.45287) | 624.444 ± 3.25619 (W 1.86891) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.088996 ± 0.0021551 (W 0.00236535) | 0.090389 ± 0.000103055 (W 0.00197222) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.234758 ± 0.0103074 (W 0.00611694) | 0.263762 ± 0.00332252 (W 0.00366416) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 2.97081 ± 0.0142946 (W 0.013128) | 5.83752 ± 0.0636515 (W 0.098993) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 3.90229 ± 0.0250174 (W 0.0120286) | 6.52329 ± 0.198659 (W 0.060429) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n1536/b1` | 7.86164 ± 0.0331865 (W 0.0161402) | 12.4523 ± 0.115222 (W 0.112697) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 10.8323 ± 0.103131 (W 0.060566) | 18.5076 ± 0.0246303 (W 0.0756191) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 74.1885 ± 0.0369396 (W 0.119158) | 98.3423 ± 1.06688 (W 0.190075) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 80.6139 ± 0.243692 (W 0.40996) | 99.3597 ± 0.219329 (W 0.510842) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 191.884 ± 0.233639 (W 0.238942) | 252.078 ± 0.301687 (W 1.23049) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 206.685 ± 0.318738 (W 0.267141) | 269.861 ± 1.17626 (W 2.65305) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 45.6632 ± 0.6085 (W 0.0661382) | 61.3652 ± 0.0898822 (W 0.0551129) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 54.3746 ± 0.340945 (W 0.0902941) | 58.9207 ± 0.559761 (W 0.0939969) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 147.631 ± 0.54197 (W 0.501548) | 181.372 ± 1.10153 (W 1.6794) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 169.284 ± 1.2208 (W 0.919083) | 178.379 ± 1.03767 (W 0.755809) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 6.95617 ± 0.0239087 (W 0.0255523) | 43.9498 ± 0.194184 (W 0.797779) | - | 🔵 faer-fork 6.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 15.9538 ± 0.107102 (W 0.0482056) | 18.6031 ± 0.455765 (W 0.172683) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 19.4494 ± 0.0849953 (W 0.0215222) | 109.801 ± 1.48152 (W 2.22418) | - | 🔵 faer-fork 5.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 39.6831 ± 0.382166 (W 0.166052) | 53.5978 ± 0.0782381 (W 0.130749) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 21.9457 ± 0.0992784 (W 0.154664) | 86.7762 ± 0.464311 (W 0.487885) | - | 🔵 faer-fork 3.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 33.0999 ± 0.437348 (W 0.111983) | 43.7657 ± 0.722162 (W 1.03477) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 71.298 ± 0.910935 (W 0.134861) | 226.976 ± 2.58484 (W 2.27811) | - | 🔵 faer-fork 3.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 99.0456 ± 0.361113 (W 0.176033) | 138.503 ± 2.54617 (W 6.31617) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 10.5232 ± 0.039831 (W 0.0159772) | 71.8122 ± 0.561248 (W 0.228234) | - | 🔵 faer-fork 6.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 31.8687 ± 0.138773 (W 0.0498426) | 174.827 ± 0.260577 (W 0.827755) | - | 🔵 faer-fork 5.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 6.80025 ± 0.0229512 (W 0.0362119) | 35.5987 ± 0.365883 (W 0.22263) | - | 🔵 faer-fork 5.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 9.07593 ± 0.242436 (W 0.139615) | 18.0472 ± 0.298425 (W 0.118158) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 19.5072 ± 0.0390658 (W 0.0361169) | 94.0467 ± 1.00946 (W 1.56373) | - | 🔵 faer-fork 4.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 24.0724 ± 0.0609214 (W 0.220814) | 52.3169 ± 0.226537 (W 0.225773) | - | 🔵 faer-fork 2.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 146.793 ± 0.373028 (W 0.403182) | 229.692 ± 0.832365 (W 0.987622) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 167.225 ± 1.09612 (W 1.11388) | 248.29 ± 1.63446 (W 3.76935) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 445.82 ± 1.04637 (W 1.79571) | 649.913 ± 2.19441 (W 5.67469) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 494.457 ± 1.64337 (W 1.46394) | 688.003 ± 2.3275 (W 6.07507) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 78.4417 ± 0.292496 (W 0.146751) | 163.627 ± 1.23724 (W 0.630696) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 78.8752 ± 0.407291 (W 0.468938) | 162.699 ± 0.464718 (W 0.838135) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 260.653 ± 1.94784 (W 2.14549) | 453.193 ± 0.883494 (W 1.14435) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 267.7 ± 1.1883 (W 1.41991) | 467.356 ± 3.31758 (W 2.0417) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.07476 ± 0.00104155 (W 0.000519727) | 0.079106 ± 0.00207901 (W 0.000430047) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.161311 ± 0.00418317 (W 0.00377058) | 0.181299 ± 0.00128773 (W 0.00323567) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

## linalg-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 4.15676 ± 0.0634846 (W 0.0173918) | 4.8351 ± 0.0724828 (W 0.0818074) | 6.10367 ± 0.0292089 (W 0.0534716) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 1.47× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 5.20971 ± 0.109921 (W 0.01779) | 5.68898 ± 0.0872679 (W 0.0648988) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 12.7846 ± 0.0792283 (W 0.0127735) | 13.7407 ± 0.114319 (W 0.180366) | 15.9065 ± 0.0343083 (W 0.0489954) | 🔵 faer-fork 1.07× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 15.1478 ± 0.0993602 (W 0.025952) | 16.2604 ± 0.0624527 (W 0.149324) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 119.388 ± 0.188291 (W 0.156256) | 114.926 ± 0.148757 (W 0.117421) | 121.286 ± 0.399512 (W 0.19343) | · | · |  |
| `eigh/n1024/b1 [api-inclusive]` | 126.341 ± 0.197146 (W 0.184285) | 114.781 ± 0.55281 (W 0.847917) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 324.771 ± 0.682917 (W 1.15631) | 317.233 ± 0.738942 (W 0.46627) | 368.932 ± 2.03085 (W 0.814142) | · | 🔵 faer-fork 1.14× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 341.723 ± 1.38595 (W 0.817652) | 334.118 ± 1.38756 (W 2.60253) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 61.9646 ± 0.117015 (W 0.157754) | 65.5448 ± 0.586661 (W 0.164562) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 64.8392 ± 0.128268 (W 0.204719) | 61.2278 ± 0.366458 (W 0.185319) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 262.333 ± 1.47099 (W 0.578486) | 265.356 ± 1.68634 (W 0.862135) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 264.698 ± 1.43404 (W 1.48017) | 253.23 ± 1.77717 (W 5.79922) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 8.89909 ± 0.0121991 (W 0.0200552) | 13.1912 ± 0.0880738 (W 0.160974) | 8.90088 ± 0.0956897 (W 0.0247074) | 🔵 faer-fork 1.48× faster than `faer-upstream` | · |  |
| `lu/n1024/b1 [api-inclusive]` | 16.5782 ± 0.0425153 (W 0.0406304) | 14.1449 ± 0.156483 (W 0.141718) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 26.6693 ± 0.0815617 (W 0.0487587) | 35.5947 ± 0.291856 (W 0.28594) | 27.9699 ± 0.0681667 (W 0.0539784) | 🔵 faer-fork 1.33× faster than `faer-upstream` | · |  |
| `lu/n1536/b1 [api-inclusive]` | 43.9115 ± 0.423376 (W 0.105287) | 38.4809 ± 0.259826 (W 0.449689) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 37.3896 ± 0.190532 (W 0.052953) | 40.8705 ± 0.206735 (W 0.0644567) | 52.5681 ± 0.121076 (W 0.171349) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.41× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 48.2884 ± 0.11845 (W 0.0625041) | 49.5101 ± 1.54726 (W 0.102397) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1536/b1` | 119.937 ± 0.629639 (W 0.0922953) | 129.279 ± 0.611096 (W 0.23594) | 162.159 ± 0.601567 (W 0.536607) | 🔵 faer-fork 1.08× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `qr/n1536/b1 [api-inclusive]` | 148.18 ± 0.606259 (W 0.553714) | 148.447 ± 0.770991 (W 2.73891) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 17.4546 ± 0.15044 (W 0.0265089) | 20.9085 ± 0.0361308 (W 0.0221685) | 18.9063 ± 0.066742 (W 0.0157694) | 🔵 faer-fork 1.20× faster than `faer-upstream` | 🔵 faer-fork 1.08× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 53.1957 ± 0.222407 (W 0.0431187) | 62.7345 ± 0.329522 (W 0.0374521) | 61.8547 ± 0.298343 (W 0.0918531) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 8.49174 ± 0.0532989 (W 0.00989006) | 12.4574 ± 0.0522345 (W 0.192086) | 8.94594 ± 0.0202971 (W 0.00738134) | 🔵 faer-fork 1.47× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `solve/n1024/b1 [api-inclusive]` | 16.3878 ± 0.084581 (W 0.024288) | 13.6402 ± 0.0754479 (W 0.0993669) | - | 🔴 faer-fork 1.20× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 25.9389 ± 0.138734 (W 0.0362641) | 33.9708 ± 0.190885 (W 0.202533) | 27.6359 ± 0.0339323 (W 0.0363939) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `solve/n1536/b1 [api-inclusive]` | 29.4494 ± 0.136817 (W 0.210287) | 36.9309 ± 0.133424 (W 0.130355) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 199.49 ± 0.311614 (W 0.23449) | 267.755 ± 0.29604 (W 0.346402) | 299.428 ± 3.2355 (W 1.50922) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.50× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 220.956 ± 0.187645 (W 0.437609) | 289.287 ± 0.274362 (W 2.07884) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 678.864 ± 2.39796 (W 0.97684) | 856.193 ± 1.81529 (W 1.22391) | 867.455 ± 3.69013 (W 2.63741) | 🔵 faer-fork 1.26× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 728.517 ± 1.59173 (W 1.23694) | 903.432 ± 2.31693 (W 4.84248) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 97.2195 ± 0.366196 (W 0.0989895) | 182.513 ± 1.44195 (W 0.188258) | 121.697 ± 0.693918 (W 0.474569) | 🔵 faer-fork 1.88× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 97.55 ± 0.456017 (W 0.148215) | 181.019 ± 1.09574 (W 0.152131) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 382.216 ± 1.22429 (W 0.847264) | 587.517 ± 0.95353 (W 0.78166) | 406.393 ± 2.22639 (W 0.974094) | 🔵 faer-fork 1.54× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 390.862 ± 0.886355 (W 1.36439) | 600.271 ± 2.01904 (W 1.44569) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.109205 ± 0.00115005 (W 0.000835694) | 0.106839 ± 0.00156007 (W 0.00129961) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.258412 ± 0.00321589 (W 0.00453588) | 0.293901 ± 0.0054119 (W 0.00592752) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 2.34274 ± 0.00919225 (W 0.008023) | 3.18612 ± 0.00781375 (W 0.0447331) | 3.1217 ± 0.0240914 (W 0.0201969) | 🔵 faer-fork 1.36× faster than `faer-upstream` | 🔵 faer-fork 1.33× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 3.49747 ± 0.017947 (W 0.0189994) | 4.11853 ± 0.0564371 (W 0.0444597) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 6.92817 ± 0.0281522 (W 0.0258156) | 8.1696 ± 0.0174101 (W 0.0298493) | 8.36896 ± 0.0184621 (W 0.0471875) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 9.34338 ± 0.0319057 (W 0.0275214) | 10.6771 ± 0.0318995 (W 0.0998015) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 73.5223 ± 0.263222 (W 0.034542) | 95.6526 ± 0.184825 (W 0.347727) | 83.0649 ± 0.437723 (W 0.583966) | 🔵 faer-fork 1.30× faster than `faer-upstream` | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 80.1447 ± 0.285858 (W 0.253622) | 95.7321 ± 0.949086 (W 0.861478) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 190.325 ± 0.147234 (W 0.508986) | 244.807 ± 0.621085 (W 1.15078) | 226.241 ± 0.629577 (W 0.616254) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 205.274 ± 0.0554443 (W 0.346433) | 260.813 ± 0.819173 (W 3.8801) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 46.7517 ± 0.214333 (W 0.0810414) | 50.733 ± 0.202169 (W 0.0352387) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 51.8859 ± 0.412582 (W 0.211233) | 49.3157 ± 0.0657935 (W 0.14157) | - | 🔴 faer-fork 1.05× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 150.655 ± 0.974444 (W 0.532817) | 157.368 ± 1.47683 (W 0.655651) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 163.136 ± 0.655451 (W 0.788516) | 152.708 ± 0.276977 (W 0.473266) | - | 🔴 faer-fork 1.07× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 5.80341 ± 0.00578433 (W 0.0125372) | 10.3618 ± 0.0683269 (W 0.173723) | 5.28441 ± 0.000736547 (W 0.0179398) | 🔵 faer-fork 1.79× faster than `faer-upstream` | 🔴 faer-fork 1.10× slower than `lapack-mkl` |  |
| `lu/n1024/b1 [api-inclusive]` | 13.2876 ± 0.092202 (W 0.0779164) | 11.492 ± 0.0633189 (W 0.120215) | - | 🔴 faer-fork 1.16× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 16.3082 ± 0.149418 (W 0.0530647) | 25.3444 ± 0.0825156 (W 0.116601) | 16.6291 ± 0.0499784 (W 0.0765288) | 🔵 faer-fork 1.55× faster than `faer-upstream` | · |  |
| `lu/n1536/b1 [api-inclusive]` | 34.0114 ± 0.320728 (W 0.118918) | 28.6232 ± 0.0966423 (W 0.122411) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 19.8288 ± 0.0350913 (W 0.0236722) | 33.9151 ± 0.23913 (W 0.640244) | 31.4081 ± 0.15853 (W 0.0491369) | 🔵 faer-fork 1.71× faster than `faer-upstream` | 🔵 faer-fork 1.58× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 30.5629 ± 0.233696 (W 0.0909117) | 37.1248 ± 0.731192 (W 3.2664) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 63.8982 ± 0.125367 (W 0.0716222) | 100.057 ± 0.883507 (W 1.47791) | 85.8396 ± 0.328186 (W 0.521577) | 🔵 faer-fork 1.57× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `qr/n1536/b1 [api-inclusive]` | 91.7717 ± 0.203457 (W 0.118739) | 113.787 ± 7.31982 (W 4.57867) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n1024/b1` | 9.15765 ± 0.0297237 (W 0.0344506) | 19.193 ± 0.14426 (W 0.234152) | 13.0994 ± 0.0119203 (W 0.0219569) | 🔵 faer-fork 2.10× faster than `faer-upstream` | 🔵 faer-fork 1.43× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 28.3521 ± 0.0273091 (W 0.0334905) | 52.8738 ± 0.428066 (W 1.01963) | 35.0058 ± 0.199755 (W 0.330668) | 🔵 faer-fork 1.86× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 5.59853 ± 0.00321768 (W 0.00875592) | 9.63548 ± 0.0250969 (W 0.14909) | 5.20887 ± 0.0240778 (W 0.00841146) | 🔵 faer-fork 1.72× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` |  |
| `solve/n1024/b1 [api-inclusive]` | 6.98853 ± 0.0242448 (W 0.0169082) | 10.8232 ± 0.0668037 (W 0.0999072) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 16.0667 ± 0.0551106 (W 0.023353) | 24.056 ± 0.0791461 (W 0.155348) | 16.2878 ± 0.0773471 (W 0.0635433) | 🔵 faer-fork 1.50× faster than `faer-upstream` | · |  |
| `solve/n1536/b1 [api-inclusive]` | 19.1541 ± 0.0607899 (W 0.0316994) | 27.2903 ± 0.338459 (W 0.195746) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 139.791 ± 0.260434 (W 0.325337) | 222.602 ± 0.218318 (W 0.648222) | 223.869 ± 2.67035 (W 3.163) | 🔵 faer-fork 1.59× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 160.214 ± 0.231721 (W 0.301143) | 243.916 ± 2.75041 (W 2.65347) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 422.224 ± 1.08822 (W 0.66159) | 624.91 ± 3.67367 (W 3.93615) | 601.328 ± 3.00144 (W 4.5754) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 468.671 ± 0.218909 (W 0.444256) | 666.462 ± 3.58673 (W 5.39475) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 70.2567 ± 0.203166 (W 0.0334591) | 155.704 ± 0.442971 (W 0.0953234) | 92.8775 ± 1.23966 (W 0.242206) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔵 faer-fork 1.32× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 70.1939 ± 0.0381567 (W 0.0683576) | 155.642 ± 0.449283 (W 0.415988) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 234.899 ± 0.695546 (W 0.53331) | 427.551 ± 0.912318 (W 1.25022) | 276.697 ± 0.770016 (W 0.783264) | 🔵 faer-fork 1.82× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 244.191 ± 0.547292 (W 1.14263) | 441.863 ± 2.59497 (W 5.00236) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.093063 ± 0.00060053 (W 0.000478559) | 0.095045 ± 0.00117617 (W 0.00156393) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.175674 ± 0.00168264 (W 0.00175969) | 0.198857 ± 0.000535538 (W 0.00253759) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 15.0157 ± 0.0250818 (W 0.0172765) | 21.2493 ± 0.17542 (W 0.521465) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1024/b1 [api-inclusive]` | 18.0906 ± 0.195316 (W 0.0605647) | 20.7467 ± 0.338957 (W 0.416452) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 47.332 ± 0.0676772 (W 0.0752036) | 56.4118 ± 0.205474 (W 0.415745) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 70.1749 ± 0.141786 (W 0.350503) | 75.3584 ± 0.646898 (W 0.591958) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 236.8 ± 0.629783 (W 0.253965) | 240.72 ± 0.757605 (W 0.219712) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 247.69 ± 0.3073 (W 0.513236) | 252.489 ± 0.528655 (W 1.72537) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 784.596 ± 1.40102 (W 1.23925) | 788.097 ± 1.14923 (W 1.6776) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 820.702 ± 2.75643 (W 0.796199) | 826.649 ± 2.2484 (W 1.22088) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 137.382 ± 0.55298 (W 0.47264) | 160.096 ± 1.35434 (W 0.667888) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 154.592 ± 1.03989 (W 1.27) | 154.239 ± 0.726871 (W 0.504458) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 687.34 ± 2.67043 (W 4.46082) | 702.356 ± 1.51587 (W 0.854918) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 689.321 ± 2.87928 (W 2.28729) | 705.299 ± 2.53284 (W 2.49246) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 33.3412 ± 0.192133 (W 0.0940482) | 83.3229 ± 0.513973 (W 3.1962) | - | 🔵 faer-fork 2.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 50.527 ± 0.46106 (W 0.224108) | 44.993 ± 0.344828 (W 0.345074) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 107.903 ± 0.269529 (W 0.346631) | 209.82 ± 6.41924 (W 5.28702) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n1536/b1 [api-inclusive]` | 138.391 ± 0.147193 (W 0.177732) | 157.58 ± 0.649038 (W 0.482247) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 142.716 ± 0.147414 (W 0.130783) | 197.408 ± 0.329342 (W 0.333147) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 167.101 ± 0.503784 (W 0.35837) | 174.528 ± 2.88684 (W 1.21562) | - | · | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 466.766 ± 1.19947 (W 0.663271) | 574.34 ± 1.50496 (W 0.91542) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 513.881 ± 1.1741 (W 0.532805) | 535.03 ± 1.27859 (W 1.005) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 63.8668 ± 0.109362 (W 0.109029) | 117.363 ± 0.31833 (W 0.162571) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 206.402 ± 0.777385 (W 0.474276) | 314.558 ± 0.829252 (W 0.623493) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 33.7752 ± 0.0942607 (W 0.211877) | 63.4623 ± 1.08009 (W 1.73158) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 37.6996 ± 0.0180687 (W 0.123844) | 44.8946 ± 0.048204 (W 0.515672) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 108.302 ± 0.458025 (W 0.36305) | 174.756 ± 0.606228 (W 2.53855) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 138.643 ± 0.439754 (W 0.198058) | 159.689 ± 0.654593 (W 0.91445) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 516.197 ± 1.26786 (W 0.788701) | 572.221 ± 1.42462 (W 0.877485) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 560.511 ± 0.782396 (W 1.51664) | 617.102 ± 2.41595 (W 1.49836) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1838.84 ± 2.44368 (W 2.35108) | 2005.22 ± 4.36378 (W 1.14897) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1922.14 ± 2.48375 (W 5.92518) | 2098.28 ± 3.80263 (W 1.56379) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 291.987 ± 1.12355 (W 0.650079) | 364.893 ± 1.34846 (W 1.03901) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 290.586 ± 0.62154 (W 2.83024) | 363.208 ± 0.920228 (W 3.72047) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1146.78 ± 0.940968 (W 2.25625) | 1353.41 ± 1.99471 (W 1.78964) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 1162.34 ± 0.606983 (W 1.76693) | 1372.67 ± 3.34322 (W 1.1963) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.194954 ± 0.00283388 (W 0.00310967) | 0.20951 ± 0.00338473 (W 0.00513044) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.549913 ± 0.00744126 (W 0.00441381) | 0.700979 ± 0.00623295 (W 0.00430425) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.24876 ± 0.00692128 (W 0.0391281) | 13.3294 ± 0.359204 (W 0.496605) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n1024/b1 [api-inclusive]` | 11.6132 ± 0.01832 (W 0.0608759) | 14.0353 ± 0.160756 (W 0.240512) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 25.0726 ± 0.0528329 (W 0.025312) | 32.9665 ± 0.189851 (W 0.496187) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1 [api-inclusive]` | 49.0479 ± 0.104433 (W 0.162861) | 53.8154 ± 0.218922 (W 0.191619) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 138.617 ± 0.227975 (W 0.367575) | 179.2 ± 0.254311 (W 2.08472) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1024/b1 [api-inclusive]` | 148.722 ± 0.459428 (W 0.31063) | 191.127 ± 2.03296 (W 2.36287) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 428.579 ± 0.538625 (W 0.798167) | 525.605 ± 5.10661 (W 4.95698) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1 [api-inclusive]` | 465.267 ± 0.752757 (W 0.954502) | 567.682 ± 3.91322 (W 5.72242) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 76.9914 ± 1.0756 (W 2.74154) | 99.538 ± 0.427017 (W 0.529837) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 96.1748 ± 0.589157 (W 0.365404) | 98.6918 ± 1.0784 (W 1.15418) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 484.679 ± 2.79411 (W 3.09256) | 497.613 ± 5.38731 (W 2.46176) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 512.938 ± 2.89331 (W 2.8027) | 532.765 ± 3.06046 (W 2.1633) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 20.9235 ± 0.0943307 (W 0.0787798) | 58.5656 ± 0.166738 (W 0.876136) | - | 🔵 faer-fork 2.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1024/b1 [api-inclusive]` | 38.6861 ± 0.241802 (W 0.188542) | 32.5838 ± 0.384962 (W 0.310551) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 64.8183 ± 0.180136 (W 0.298009) | 141.84 ± 0.837112 (W 0.504859) | - | 🔵 faer-fork 2.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1 [api-inclusive]` | 98.3461 ± 0.403952 (W 0.471282) | 118.302 ± 0.464805 (W 0.471368) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 79.012 ± 0.368392 (W 1.06645) | 155.654 ± 2.36275 (W 2.0661) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1 [api-inclusive]` | 102.159 ± 0.354302 (W 0.441018) | 126.545 ± 3.23323 (W 5.97515) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 256.016 ± 1.24512 (W 0.561858) | 440.755 ± 0.944286 (W 11.3915) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1536/b1 [api-inclusive]` | 302.346 ± 0.594146 (W 0.666783) | 395.972 ± 1.5766 (W 15.6251) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 35.2916 ± 0.0301977 (W 0.0441655) | 99.0974 ± 0.964892 (W 1.2792) | - | 🔵 faer-fork 2.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1536/b1` | 114.727 ± 0.354994 (W 0.246334) | 254.477 ± 2.60191 (W 3.82881) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1` | 21.2589 ± 0.0463882 (W 0.0381473) | 47.9727 ± 0.375686 (W 0.321255) | - | 🔵 faer-fork 2.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1024/b1 [api-inclusive]` | 25.4687 ± 0.0708247 (W 0.0790282) | 31.7133 ± 0.279749 (W 0.419094) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 65.7445 ± 0.259092 (W 0.413042) | 124.175 ± 0.364916 (W 1.02558) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1 [api-inclusive]` | 98.9658 ± 0.436328 (W 0.564445) | 117.588 ± 0.780547 (W 0.480352) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 308.601 ± 0.559033 (W 1.1425) | 396.758 ± 1.0905 (W 3.9206) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1 [api-inclusive]` | 354.552 ± 1.29224 (W 1.59338) | 435.968 ± 4.77591 (W 8.0039) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1214.77 ± 2.18686 (W 2.93396) | 1456.85 ± 9.71741 (W 10.6338) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1 [api-inclusive]` | 1299.02 ± 2.92742 (W 3.53335) | 1554.76 ± 5.93894 (W 6.62649) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 174.716 ± 1.63473 (W 0.473972) | 247.732 ± 0.435882 (W 1.2398) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1 [api-inclusive]` | 176.405 ± 0.70806 (W 0.428217) | 246.861 ± 0.699468 (W 3.69659) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 811.52 ± 1.12033 (W 4.32605) | 994.812 ± 2.793 (W 6.51826) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1 [api-inclusive]` | 829.632 ± 2.76678 (W 9.32106) | 1017.24 ± 5.22001 (W 3.06456) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.133227 ± 0.00224953 (W 0.00233097) | 0.152976 ± 0.000667721 (W 0.00300324) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.383315 ± 0.00695962 (W 0.00818677) | 0.559464 ± 0.011592 (W 0.0059507) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 14.8122 ± 0.0330478 (W 0.0142891) | 15.4274 ± 0.426103 (W 0.292204) | 18.097 ± 0.0770884 (W 0.0565402) | · | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 16.6888 ± 0.0656529 (W 0.0213683) | 17.8195 ± 0.135405 (W 0.299332) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 47.3025 ± 0.0889207 (W 0.0641398) | 48.4029 ± 0.156006 (W 0.75082) | 60.8431 ± 0.0903116 (W 0.0623864) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 66.5155 ± 0.294068 (W 0.405372) | 67.3078 ± 0.516926 (W 0.785418) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 234.844 ± 0.474046 (W 0.221347) | 238.752 ± 0.410192 (W 0.497585) | 340.87 ± 2.06406 (W 0.770566) | · | 🔵 faer-fork 1.45× faster than `lapack-mkl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 246.349 ± 0.275188 (W 0.29558) | 249.291 ± 0.770325 (W 2.55123) | - | · | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 780.36 ± 3.13072 (W 1.21486) | 780.254 ± 0.796059 (W 1.11767) | 1120.06 ± 4.80401 (W 3.64538) | · | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 818.517 ± 1.95716 (W 1.88086) | 820.423 ± 1.73695 (W 1.00501) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 140.81 ± 0.45579 (W 0.588058) | 150.267 ± 0.897375 (W 0.899941) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 149.478 ± 0.615776 (W 0.643027) | 144.452 ± 1.24218 (W 0.654224) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 674.16 ± 3.11643 (W 1.27218) | 676.764 ± 2.39774 (W 3.88481) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 680.012 ± 3.14621 (W 3.30449) | 677.751 ± 3.1474 (W 1.42102) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 29.8652 ± 0.138501 (W 0.0443048) | 34.4874 ± 0.114983 (W 0.326488) | 37.6571 ± 0.137481 (W 0.033807) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `lu/n1024/b1 [api-inclusive]` | 45.9879 ± 0.378309 (W 0.204298) | 37.1937 ± 0.229604 (W 0.303293) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 93.4326 ± 0.102439 (W 0.393671) | 103.087 ± 0.68995 (W 0.869016) | 121.667 ± 0.674326 (W 0.34862) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `lu/n1536/b1 [api-inclusive]` | 127.09 ± 1.03143 (W 0.414518) | 136.038 ± 0.275455 (W 0.721501) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 137.823 ± 0.0992837 (W 0.542904) | 143.554 ± 0.51068 (W 0.118305) | 178.293 ± 0.730657 (W 0.406666) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 161.054 ± 0.372733 (W 0.410626) | 167.119 ± 0.621696 (W 0.50321) | - | · | - | `*` one row has no measurement here |
| `qr/n1536/b1` | 449.4 ± 1.25118 (W 0.521433) | 461.463 ± 1.47545 (W 0.411287) | 589.721 ± 2.88295 (W 2.15299) | · | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `qr/n1536/b1 [api-inclusive]` | 502.75 ± 0.521715 (W 0.783389) | 512.823 ± 1.75279 (W 0.827857) | - | · | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 59.9369 ± 0.206182 (W 0.0961785) | 65.7298 ± 0.0539378 (W 0.0349627) | 70.1918 ± 0.329105 (W 0.0750777) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 190.601 ± 0.672262 (W 0.2584) | 202.84 ± 0.371933 (W 0.115005) | 236.535 ± 1.06324 (W 0.828603) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 29.9295 ± 0.0829072 (W 0.155872) | 34.0798 ± 0.0171216 (W 0.279839) | 34.7526 ± 0.0391367 (W 0.0502518) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `solve/n1024/b1 [api-inclusive]` | 33.4137 ± 0.160009 (W 0.174302) | 37.2532 ± 0.110996 (W 0.315473) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 93.619 ± 0.0369174 (W 0.0675148) | 103.189 ± 0.230613 (W 0.82359) | 116.798 ± 0.0798504 (W 0.349761) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `solve/n1536/b1 [api-inclusive]` | 128.215 ± 0.328754 (W 0.29702) | 135.368 ± 0.941624 (W 1.10151) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 508.411 ± 1.39164 (W 0.595665) | 562.873 ± 2.51423 (W 0.54215) | 777.297 ± 2.42924 (W 1.8414) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.53× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 552.846 ± 0.888081 (W 0.969684) | 609.362 ± 1.01644 (W 2.00352) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1815.18 ± 5.45682 (W 2.01387) | 1979.55 ± 4.04794 (W 1.06688) | 2459.21 ± 12.6638 (W 7.28997) | 🔵 faer-fork 1.09× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1903.37 ± 4.06607 (W 3.4241) | 2077.33 ± 7.4481 (W 2.46304) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 283.952 ± 0.685876 (W 0.408265) | 356.149 ± 0.459818 (W 1.08298) | 360.578 ± 0.678291 (W 0.881933) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 284.67 ± 0.964415 (W 2.54744) | 354.462 ± 0.719395 (W 3.40787) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 1121.92 ± 1.95114 (W 1.61837) | 1323.75 ± 1.38836 (W 1.29968) | 1457.61 ± 8.20306 (W 2.60281) | 🔵 faer-fork 1.18× faster than `faer-upstream` | 🔵 faer-fork 1.30× faster than `lapack-mkl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 1134.39 ± 3.5517 (W 2.67398) | 1346.83 ± 0.634149 (W 1.64768) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.216603 ± 0.00251506 (W 0.00349579) | 0.235139 ± 0.00329651 (W 0.0038441) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.655311 ± 0.0054551 (W 0.00769841) | 0.8759 ± 0.00478368 (W 0.00841568) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n1024/b1` | 8.29738 ± 0.013229 (W 0.0168182) | 9.46402 ± 0.0782632 (W 0.105573) | 9.57423 ± 0.0352489 (W 0.0541352) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `cholesky/n1024/b1 [api-inclusive]` | 10.3466 ± 0.00409711 (W 0.0256877) | 11.1818 ± 0.0183114 (W 0.111732) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n1536/b1` | 25.5272 ± 0.453244 (W 0.0447766) | 26.6184 ± 0.109894 (W 0.182148) | 32.9066 ± 0.0521958 (W 0.0425994) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `cholesky/n1536/b1 [api-inclusive]` | 45.1647 ± 0.0780985 (W 0.0521468) | 45.9133 ± 0.0790905 (W 0.365844) | - | · | - | `*` one row has no measurement here |
| `eigh/n1024/b1` | 137.018 ± 0.205747 (W 0.330148) | 175.975 ± 0.949181 (W 1.2603) | 211.415 ± 0.845777 (W 0.645731) | 🔵 faer-fork 1.28× faster than `faer-upstream` | 🔵 faer-fork 1.54× faster than `lapack-mkl` |  |
| `eigh/n1024/b1 [api-inclusive]` | 147.503 ± 0.35736 (W 0.261399) | 188.719 ± 1.3436 (W 2.06101) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n1536/b1` | 423.793 ± 0.694537 (W 0.311328) | 521.357 ± 3.76102 (W 4.80853) | 664.138 ± 0.997281 (W 2.11629) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `eigh/n1536/b1 [api-inclusive]` | 464.184 ± 2.04798 (W 2.13748) | 562.095 ± 5.55365 (W 11.0602) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1` | 77.5017 ± 0.279018 (W 1.10513) | 89.2678 ± 0.207559 (W 0.32852) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n1024/b1 [api-inclusive]` | 90.8013 ± 0.778102 (W 0.382178) | 87.9296 ± 0.467293 (W 0.164471) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1` | 474.91 ± 2.60019 (W 3.28034) | 477.65 ± 3.02291 (W 2.41343) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n1536/b1 [api-inclusive]` | 502.202 ± 3.50719 (W 4.84009) | 505.251 ± 4.28904 (W 5.01333) | - | · | - | `*` one row has no measurement here |
| `lu/n1024/b1` | 18.0597 ± 0.0255678 (W 0.0315222) | 22.2836 ± 0.0675014 (W 0.132943) | 21.1975 ± 0.0948504 (W 0.120272) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `lu/n1024/b1 [api-inclusive]` | 33.7839 ± 0.0347207 (W 0.188116) | 24.9563 ± 0.140027 (W 0.149011) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n1536/b1` | 53.1889 ± 0.0651221 (W 0.105572) | 61.6777 ± 0.248732 (W 0.348218) | 71.2389 ± 0.592724 (W 0.341843) | 🔵 faer-fork 1.16× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `lu/n1536/b1 [api-inclusive]` | 87.612 ± 0.42529 (W 0.18963) | 94.4275 ± 0.550799 (W 0.368611) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n1024/b1` | 72.3087 ± 0.545577 (W 0.451343) | 109.016 ± 0.602415 (W 0.742343) | 97.6479 ± 0.329509 (W 0.469055) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `qr/n1024/b1 [api-inclusive]` | 96.1801 ± 0.341526 (W 0.20952) | 126.336 ± 4.40892 (W 6.53537) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n1536/b1` | 237.555 ± 0.931773 (W 0.330255) | 330.109 ± 4.43738 (W 9.61141) | 319.549 ± 0.122406 (W 2.06239) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `qr/n1536/b1 [api-inclusive]` | 291.872 ± 1.24841 (W 1.51541) | 375.223 ± 3.13041 (W 5.21459) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n1024/b1` | 32.1755 ± 0.0789892 (W 0.0361519) | 52.21 ± 0.654881 (W 0.9276) | 40.343 ± 0.0313527 (W 0.0415005) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `qr_factor/n1536/b1` | 102.691 ± 0.667248 (W 0.309051) | 150.08 ± 1.27194 (W 4.69813) | 129.733 ± 0.118339 (W 0.804421) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `solve/n1024/b1` | 18.1705 ± 0.0717542 (W 0.0291004) | 22.1079 ± 0.0658402 (W 0.151685) | 19.0853 ± 0.189519 (W 0.0673193) | 🔵 faer-fork 1.22× faster than `faer-upstream` | · |  |
| `solve/n1024/b1 [api-inclusive]` | 21.0924 ± 0.0458785 (W 0.0447452) | 24.824 ± 0.0829128 (W 0.102674) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n1536/b1` | 53.6433 ± 0.095915 (W 0.136039) | 61.6069 ± 0.191969 (W 0.277797) | 68.9194 ± 0.189893 (W 0.362975) | 🔵 faer-fork 1.15× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `solve/n1536/b1 [api-inclusive]` | 88.6696 ± 0.305697 (W 0.202713) | 94.9476 ± 0.469718 (W 0.464275) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1024/b1` | 301.506 ± 0.256149 (W 1.13375) | 390.41 ± 0.639009 (W 2.58437) | 471.024 ± 1.72008 (W 0.89438) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |
| `svd_thin/n1024/b1 [api-inclusive]` | 345.625 ± 0.75853 (W 1.28903) | 429.095 ± 4.35449 (W 12.3377) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n1536/b1` | 1189.77 ± 0.369387 (W 6.34852) | 1436.81 ± 1.54356 (W 9.04643) | 1707.86 ± 3.15519 (W 7.08189) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `svd_thin/n1536/b1 [api-inclusive]` | 1276.37 ± 3.06455 (W 4.59051) | 1533.4 ± 5.12532 (W 9.31638) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1024/b1` | 166.614 ± 1.24911 (W 0.461324) | 237.878 ± 0.515484 (W 1.62703) | 252.661 ± 0.524849 (W 0.987582) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔵 faer-fork 1.52× faster than `lapack-mkl` |  |
| `svd_values/n1024/b1 [api-inclusive]` | 168.637 ± 0.426802 (W 0.526793) | 236.71 ± 0.774972 (W 3.60401) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n1536/b1` | 789.286 ± 2.88692 (W 2.60657) | 968.149 ± 2.44242 (W 4.54728) | 1057.3 ± 5.92281 (W 2.88231) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `svd_values/n1536/b1 [api-inclusive]` | 807.794 ± 3.64502 (W 4.40221) | 991.16 ± 4.28596 (W 6.93884) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1024/b1` | 0.153577 ± 0.000967187 (W 0.00117657) | 0.166871 ± 0.00372021 (W 0.0033502) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n1536/b1` | 0.491473 ± 0.0397856 (W 0.00625711) | 0.732117 ± 0.00630039 (W 0.0120355) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/1024`, `cholesky/1536`, `eigh/1024`, `eigh/1536`, `full_piv_lu/1024`, `full_piv_lu/1536`, `lu/1024`, `lu/1536`, `qr/1024`, `qr/1536`, `qr_factor/1024`, `qr_factor/1536`, `solve/1024`, `solve/1536`, `svd_thin/1024`, `svd_thin/1536`, `svd_values/1024`, `svd_values/1536`, `triangular_solve/1024`, `triangular_solve/1536`.

## gemm-large, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.4701 ± 0.106369 (W 0.0259234) | 17.5013 ± 0.0858321 (W 0.0316045) | 20.6162 ± 0.00436178 (W 0.0313311) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 59.1262 ± 0.269257 (W 0.0583789) | 59.1221 ± 0.0824159 (W 0.0434333) | 65.2013 ± 0.137194 (W 0.0446252) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |

## gemm-large, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.95786 ± 0.0397668 (W 0.0156361) | 8.96057 ± 0.0788239 (W 0.0326632) | 11.4852 ± 0.0500837 (W 0.0368375) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 30.1745 ± 0.101698 (W 0.0615966) | 30.1687 ± 0.100854 (W 0.0622816) | 36.1477 ± 0.0617408 (W 0.0438363) | · | 🔵 faer-fork 1.20× faster than `lapack-mkl` |  |

## gemm-large, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 17.4852 ± 0.0333926 (W 0.0219738) | 17.4183 ± 0.0691734 (W 0.0123091) | 20.6091 ± 0.0806071 (W 0.0400203) | · | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 59.0903 ± 0.151603 (W 0.0473874) | 58.9305 ± 0.129093 (W 0.0484378) | 65.0336 ± 0.0465216 (W 0.0494162) | · | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |

## gemm-large, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 8.97523 ± 0.03239 (W 0.0159824) | 8.96212 ± 0.0149489 (W 0.00672276) | 11.4921 ± 0.0234407 (W 0.0302918) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 30.2047 ± 0.0803879 (W 0.0232709) | 30.2131 ± 0.084106 (W 0.0649818) | 36.6829 ± 0.598763 (W 0.210049) | · | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |

## gemm-large, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.435 ± 0.0420925 (W 0.0631994) | 69.548 ± 0.0641852 (W 0.055194) | 110.64 ± 0.206494 (W 0.154204) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 237.974 ± 0.49457 (W 0.434962) | 237.812 ± 0.312573 (W 0.416407) | 370.408 ± 0.245447 (W 0.711429) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |

## gemm-large, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.1631 ± 0.056547 (W 0.0273782) | 35.134 ± 0.0989632 (W 0.0251839) | 44.997 ± 0.317623 (W 0.131447) | · | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 120.177 ± 0.142507 (W 0.132036) | 119.729 ± 0.231069 (W 0.0929303) | 148.269 ± 0.301439 (W 0.678986) | · | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |

## gemm-large, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 69.3876 ± 0.0245985 (W 0.0531334) | 69.4941 ± 0.0727017 (W 0.0455186) | 110.479 ± 0.507741 (W 0.160343) | · | 🔵 faer-fork 1.59× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 238.38 ± 0.364779 (W 0.338216) | 237.559 ± 1.00504 (W 0.393123) | 370.902 ± 0.498099 (W 0.553359) | · | 🔵 faer-fork 1.56× faster than `lapack-mkl` |  |

## gemm-large, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `gemm/1024x1024x1024/b1` | 35.068 ± 0.133841 (W 0.0282635) | 35.0766 ± 0.0908803 (W 0.0683131) | 45.3091 ± 0.125189 (W 0.368659) | · | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `gemm/1536x1536x1536/b1` | 119.499 ± 0.242178 (W 0.198841) | 119.591 ± 0.321302 (W 0.342653) | 148.216 ± 0.356038 (W 0.458734) | · | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |

## linalg-boundary, f64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.168765 ± 0.000442803 (W 0.00195959) | 0.627406 ± 0.0547622 (W 0.0550162) | - | 🔵 faer-fork 3.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.196905 ± 0.00112199 (W 0.00239492) | 0.321793 ± 0.00891764 (W 0.00764409) | - | 🔵 faer-fork 1.63× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1` | 0.155398 ± 0.000485099 (W 0.00178822) | 0.570346 ± 0.0490154 (W 0.103038) | - | 🔵 faer-fork 3.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1 [api-inclusive]` | 0.184732 ± 0.000929256 (W 0.00138395) | 0.303365 ± 0.00425305 (W 0.00663159) | - | 🔵 faer-fork 1.64× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.705637 ± 0.00514163 (W 0.00123796) | 1.73564 ± 0.0573187 (W 0.0956764) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n511/b1 [api-inclusive]` | 0.875262 ± 0.00646213 (W 0.00349438) | 1.15793 ± 0.0214245 (W 0.022831) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.686071 ± 0.00407135 (W 0.00234551) | 1.66836 ± 0.0750859 (W 0.0841888) | - | 🔵 faer-fork 2.43× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n513/b1 [api-inclusive]` | 0.942145 ± 0.0053606 (W 0.0051096) | 1.07371 ± 0.0210061 (W 0.026539) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.88472 ± 0.0185023 (W 0.011701) | 7.31028 ± 0.0131407 (W 0.015409) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.90154 ± 0.0123567 (W 0.00876016) | 7.32905 ± 0.020813 (W 0.0193961) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.79921 ± 0.0240034 (W 0.00559253) | 7.17957 ± 0.024707 (W 0.00547189) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.82874 ± 0.0394453 (W 0.00853528) | 7.19737 ± 0.0352513 (W 0.00872837) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.8547 ± 0.0367358 (W 0.0304511) | 25.4734 ± 0.0796921 (W 0.0410804) | - | · | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 24.7467 ± 0.0582349 (W 0.0118302) | 25.6113 ± 0.0925007 (W 0.0295753) | - | · | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.2843 ± 0.0872263 (W 0.0425765) | 25.8895 ± 0.0497127 (W 0.0466559) | - | · | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 25.2626 ± 0.0125033 (W 0.0245695) | 26.0064 ± 0.0619789 (W 0.0705739) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.26802 ± 0.00886171 (W 0.00306097) | 1.97699 ± 0.00450931 (W 0.00319737) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.2708 ± 0.00419167 (W 0.0106556) | 2.62623 ± 0.0249809 (W 0.0137493) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.29791 ± 0.0050476 (W 0.00379628) | 2.04838 ± 0.00870286 (W 0.0040007) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.5639 ± 0.0140819 (W 0.00497724) | 2.19024 ± 0.163339 (W 0.00631212) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n511/b1` | 6.27525 ± 0.0220595 (W 0.0170249) | 9.29274 ± 0.0517597 (W 0.0125472) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 13.1461 ± 0.0352112 (W 0.0261523) | 14.6176 ± 0.144031 (W 0.0349265) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.38831 ± 0.0327205 (W 0.0152544) | 9.39427 ± 0.0729684 (W 0.01569) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 7.32087 ± 0.0236993 (W 0.0135102) | 8.64695 ± 0.0125253 (W 0.0123804) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.431519 ± 0.00164941 (W 0.00444527) | 0.803607 ± 0.0190639 (W 0.0125456) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.700533 ± 0.00776896 (W 0.00361197) | 0.648907 ± 0.0139219 (W 0.00754027) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.42381 ± 0.00494443 (W 0.00290666) | 0.821871 ± 0.00853184 (W 0.0187682) | - | 🔵 faer-fork 1.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.759043 ± 0.00258512 (W 0.00498504) | 0.57924 ± 0.00802958 (W 0.0100921) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.75868 ± 0.0141709 (W 0.0146002) | 3.39978 ± 0.0401522 (W 0.0539846) | - | 🔵 faer-fork 1.93× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.50759 ± 0.0173081 (W 0.00802203) | 3.20542 ± 0.037404 (W 0.0821805) | - | 🔴 faer-fork 1.09× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.86267 ± 0.0114334 (W 0.01042) | 3.42562 ± 0.0234525 (W 0.0407316) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.47275 ± 0.00955633 (W 0.0133014) | 2.5291 ± 0.00627527 (W 0.0234373) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 1.04895 ± 0.00148031 (W 0.00560542) | 2.72373 ± 0.00420196 (W 0.013752) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.57055 ± 0.00926787 (W 0.00399217) | 2.23515 ± 0.00998814 (W 0.00585543) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.11246 ± 0.00132534 (W 0.00202281) | 2.82065 ± 0.0232992 (W 0.0143736) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.66727 ± 0.00422671 (W 0.00665596) | 2.33323 ± 0.00366038 (W 0.00653474) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.85484 ± 0.0398396 (W 0.0115248) | 12.2494 ± 0.0511411 (W 0.0603592) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 8.37064 ± 0.0493999 (W 0.0242596) | 9.61282 ± 0.0431237 (W 0.028296) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 5.98817 ± 0.0233051 (W 0.00688745) | 12.1976 ± 0.0903469 (W 0.0518983) | - | 🔵 faer-fork 2.04× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 8.59959 ± 0.0375351 (W 0.0567598) | 9.58277 ± 0.349427 (W 0.0354079) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n255/b1` | 0.657276 ± 0.00416384 (W 0.00309689) | 2.32163 ± 0.0090519 (W 0.00987985) | - | 🔵 faer-fork 3.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 0.698804 ± 0.00057523 (W 0.00309242) | 2.40319 ± 0.00987418 (W 0.00510484) | - | 🔵 faer-fork 3.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 3.20171 ± 0.00509453 (W 0.00806174) | 9.69702 ± 0.0176771 (W 0.0532001) | - | 🔵 faer-fork 3.03× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 3.21865 ± 0.00972799 (W 0.0051375) | 9.45638 ± 0.0128761 (W 0.0408234) | - | 🔵 faer-fork 2.94× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.355268 ± 0.00147235 (W 0.00224318) | 0.785685 ± 0.00970096 (W 0.0147961) | - | 🔵 faer-fork 2.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 0.710231 ± 0.00104953 (W 0.00297758) | 0.597617 ± 0.00434602 (W 0.00911965) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.427361 ± 0.00133591 (W 0.00180703) | 0.787415 ± 0.011097 (W 0.00906062) | - | 🔵 faer-fork 1.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.774362 ± 0.00321942 (W 0.00435005) | 0.562363 ± 0.00303779 (W 0.0088412) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.71062 ± 0.0122006 (W 0.00509294) | 3.3363 ± 0.0465712 (W 0.0567752) | - | 🔵 faer-fork 1.95× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.48553 ± 0.0166645 (W 0.00902758) | 2.94195 ± 0.00914768 (W 0.0696139) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.80461 ± 0.00724578 (W 0.00863101) | 3.35229 ± 0.0369504 (W 0.0562612) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 3.46876 ± 0.0288883 (W 0.00805911) | 2.44255 ± 0.045873 (W 0.0369349) | - | 🔴 faer-fork 1.42× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 8.00198 ± 0.063998 (W 0.0114163) | 13.554 ± 0.0445317 (W 0.0207349) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 8.05312 ± 0.0363563 (W 0.0151687) | 13.5543 ± 0.00927728 (W 0.0177932) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.21587 ± 0.0207317 (W 0.017989) | 10.1455 ± 0.0721326 (W 0.00986849) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 8.27157 ± 0.0221574 (W 0.0176192) | 10.2431 ± 0.0194388 (W 0.0391343) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.8168 ± 0.106714 (W 0.0359593) | 54.5154 ± 0.213985 (W 0.0956445) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 35.909 ± 0.0368825 (W 0.0590196) | 54.8091 ± 0.225068 (W 0.550662) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.7405 ± 0.0886163 (W 0.037887) | 41.5654 ± 0.0563436 (W 0.050791) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 36.7566 ± 0.112121 (W 0.0731744) | 41.8178 ± 0.0776357 (W 0.268991) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 4.05657 ± 0.0180361 (W 0.0100113) | 10.7375 ± 0.0338525 (W 0.0137003) | - | 🔵 faer-fork 2.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 4.04767 ± 0.00459205 (W 0.0109502) | 10.7703 ± 0.064197 (W 0.0207135) | - | 🔵 faer-fork 2.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.15136 ± 0.0558649 (W 0.00732113) | 8.19353 ± 0.0590241 (W 0.0134822) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 4.13958 ± 0.0196965 (W 0.00950234) | 8.16764 ± 0.0728902 (W 0.0104614) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.6852 ± 0.0886166 (W 0.0282939) | 41.1181 ± 0.145385 (W 0.0789208) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 17.667 ± 0.0368947 (W 0.0280759) | 41.1696 ± 0.152018 (W 0.0795993) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.7893 ± 0.070387 (W 0.0262561) | 29.7858 ± 0.110251 (W 0.0517712) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 17.8235 ± 0.101991 (W 0.0145523) | 29.8929 ± 0.132822 (W 0.0460439) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.013075 ± 3.52184e-05 (W 0.000176865) | 0.012343 ± 0.000133167 (W 7.15542e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.013164 ± 0.000115036 (W 0.000113966) | 0.012574 ± 6.26126e-05 (W 8.18303e-05) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.03193 ± 6.61589e-05 (W 0.000153786) | 0.041838 ± 0.000516459 (W 0.000449662) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.03196 ± 0.000205899 (W 0.000134433) | 0.031349 ± 0.000496319 (W 0.000156854) | - | · | - | `*` one row has no measurement here |

## linalg-boundary, f64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.153335 ± 0.00084444 (W 0.00127493) | 0.487727 ± 0.0018893 (W 0.0104427) | - | 🔵 faer-fork 3.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.18232 ± 0.00202293 (W 0.00157675) | 0.314588 ± 0.0267323 (W 0.0204887) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1` | 0.141604 ± 0.00188074 (W 0.000706648) | 0.510093 ± 0.00196385 (W 0.0243942) | - | 🔵 faer-fork 3.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.172642 ± 0.00157706 (W 0.00115936) | 0.296932 ± 0.0204411 (W 0.0257616) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n511/b1` | 0.509948 ± 0.00670762 (W 0.00205235) | 1.38226 ± 0.076784 (W 0.066752) | - | 🔵 faer-fork 2.71× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n511/b1 [api-inclusive]` | 0.687648 ± 0.0015178 (W 0.00348442) | 1.02907 ± 0.0158448 (W 0.0178775) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.508881 ± 0.00950956 (W 0.00462184) | 1.40878 ± 0.00635588 (W 0.0202147) | - | 🔵 faer-fork 2.77× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 0.788696 ± 0.0038018 (W 0.00362961) | 0.953312 ± 0.0250901 (W 0.0336203) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `eigh/n255/b1` | 3.99594 ± 0.0136887 (W 0.00546435) | 7.31795 ± 0.0431638 (W 0.0355131) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 4.01776 ± 0.00325628 (W 0.0116522) | 7.3177 ± 0.0425239 (W 0.0246599) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.9386 ± 0.0046075 (W 0.0092003) | 7.15145 ± 0.0351431 (W 0.0200782) | - | 🔵 faer-fork 1.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 3.95946 ± 0.0400959 (W 0.011665) | 7.18419 ± 0.0428449 (W 0.0223234) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.2006 ± 0.0447352 (W 0.0455093) | 23.8837 ± 0.0210528 (W 0.0507146) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 16.2654 ± 0.031871 (W 0.0206713) | 23.9124 ± 0.0360365 (W 0.0712638) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.7759 ± 0.0480466 (W 0.0282653) | 24.2715 ± 0.0254034 (W 0.048399) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 16.76 ± 0.0797172 (W 0.0229605) | 24.2963 ± 0.0961338 (W 0.0587189) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.14926 ± 0.00943565 (W 0.00272583) | 1.97736 ± 0.0031894 (W 0.00195773) | - | 🔵 faer-fork 1.72× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.1886 ± 0.00983103 (W 0.00434266) | 2.65761 ± 0.00962334 (W 0.00879011) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.18969 ± 0.00664413 (W 0.00519191) | 2.03773 ± 0.00415804 (W 0.00388023) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.50679 ± 0.0100506 (W 0.0067863) | 1.91789 ± 0.00731693 (W 0.00545391) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 4.74669 ± 0.0630457 (W 0.0107683) | 8.3408 ± 0.0525569 (W 0.0191847) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.8506 ± 0.056579 (W 0.0154599) | 13.7091 ± 0.0356497 (W 0.040026) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.89139 ± 0.0258969 (W 0.0127869) | 8.44789 ± 0.0364877 (W 0.0179406) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.08657 ± 0.0992048 (W 0.024528) | 7.8062 ± 0.0581765 (W 0.0197722) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.423137 ± 0.000935174 (W 0.00244483) | 0.855258 ± 0.0158492 (W 0.0188566) | - | 🔵 faer-fork 2.02× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 0.672563 ± 0.00381435 (W 0.00352068) | 0.666803 ± 0.0263568 (W 0.0249866) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n257/b1` | 0.358165 ± 0.00271403 (W 0.00263382) | 0.876748 ± 0.00716941 (W 0.0264453) | - | 🔵 faer-fork 2.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 0.732164 ± 0.000600055 (W 0.00245036) | 0.621571 ± 0.0321661 (W 0.0289268) | - | 🔴 faer-fork 1.18× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n511/b1` | 1.28496 ± 0.004155 (W 0.00805757) | 3.41172 ± 0.0596243 (W 0.0524971) | - | 🔵 faer-fork 2.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 3.14789 ± 0.00912811 (W 0.0107138) | 3.00291 ± 0.0247637 (W 0.0496203) | - | · | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.34942 ± 0.00462393 (W 0.0101052) | 3.45517 ± 0.0466055 (W 0.0571791) | - | 🔵 faer-fork 2.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 3.13631 ± 0.0186586 (W 0.00722691) | 2.40951 ± 0.0192587 (W 0.0542957) | - | 🔴 faer-fork 1.30× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.852833 ± 0.0037961 (W 0.00335079) | 2.99917 ± 0.0196506 (W 0.024413) | - | 🔵 faer-fork 3.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 1.39254 ± 0.005412 (W 0.0058943) | 2.14415 ± 0.00447567 (W 0.00828059) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 0.903169 ± 0.002284 (W 0.00257666) | 3.12752 ± 0.00922966 (W 0.022992) | - | 🔵 faer-fork 3.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 1.47852 ± 0.00524589 (W 0.00476786) | 2.20641 ± 0.0170441 (W 0.0144919) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 3.34054 ± 0.0129733 (W 0.00722985) | 12.0278 ± 0.0295031 (W 0.0927501) | - | 🔵 faer-fork 3.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 6.10269 ± 0.0144021 (W 0.015594) | 7.80152 ± 1.18688 (W 0.25623) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1` | 3.80508 ± 0.0155556 (W 0.00993373) | 11.615 ± 0.0366925 (W 0.0776725) | - | 🔵 faer-fork 3.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 6.61063 ± 0.0115146 (W 0.0207622) | 7.57522 ± 0.0612628 (W 0.0921116) | - | 🔵 faer-fork 1.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.615775 ± 0.00228894 (W 0.00400142) | 2.55488 ± 0.0141739 (W 0.0141913) | - | 🔵 faer-fork 4.15× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 0.648245 ± 0.0012242 (W 0.00434715) | 2.65237 ± 0.00540247 (W 0.0189772) | - | 🔵 faer-fork 4.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 1.904 ± 0.00309049 (W 0.00138314) | 9.82217 ± 0.0403463 (W 0.060667) | - | 🔵 faer-fork 5.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 2.28574 ± 0.00783477 (W 0.00897859) | 9.28592 ± 0.0598531 (W 0.0577921) | - | 🔵 faer-fork 4.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.341124 ± 0.00261207 (W 0.00313034) | 0.839429 ± 0.023609 (W 0.0201767) | - | 🔵 faer-fork 2.46× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n255/b1 [api-inclusive]` | 0.682048 ± 0.00187255 (W 0.00397814) | 0.570907 ± 0.0223489 (W 0.0262841) | - | 🔴 faer-fork 1.19× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n257/b1` | 0.357727 ± 0.000434599 (W 0.00307734) | 0.86597 ± 0.00945473 (W 0.014497) | - | 🔵 faer-fork 2.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 0.744174 ± 0.00237799 (W 0.00520944) | 0.556177 ± 0.0234379 (W 0.0245001) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n511/b1` | 1.24432 ± 0.00508397 (W 0.00365024) | 3.26114 ± 0.0422918 (W 0.0488917) | - | 🔵 faer-fork 2.62× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 3.11628 ± 0.0184403 (W 0.0132386) | 2.72543 ± 0.0449001 (W 0.0364168) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.32958 ± 0.00383087 (W 0.00698407) | 3.295 ± 0.0865535 (W 0.0829901) | - | 🔵 faer-fork 2.48× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n513/b1 [api-inclusive]` | 3.12279 ± 0.0346503 (W 0.0125372) | 2.31599 ± 0.0606111 (W 0.100942) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.25768 ± 0.0122641 (W 0.0148882) | 13.6102 ± 0.0251023 (W 0.0314807) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 7.26893 ± 0.0138162 (W 0.00973872) | 13.6513 ± 0.0463768 (W 0.0411793) | - | 🔵 faer-fork 1.88× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.38465 ± 0.00614238 (W 0.0087412) | 10.2138 ± 0.0979599 (W 0.0243303) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 7.4073 ± 0.0137645 (W 0.00872194) | 10.2602 ± 0.0422409 (W 0.0595523) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.467 ± 0.0738847 (W 0.0606498) | 51.2582 ± 0.131255 (W 0.0932865) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 29.5261 ± 0.0615028 (W 0.0486889) | 51.3135 ± 0.371874 (W 0.122602) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.9622 ± 0.0210453 (W 0.0348877) | 38.1782 ± 0.109752 (W 0.104018) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 30.083 ± 0.0924397 (W 0.0420663) | 38.3562 ± 0.120401 (W 0.176696) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.90956 ± 0.0177247 (W 0.0101849) | 10.8564 ± 0.0271776 (W 0.0126416) | - | 🔵 faer-fork 2.78× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 3.90104 ± 0.0187453 (W 0.00784281) | 10.7636 ± 0.0695954 (W 0.00804904) | - | 🔵 faer-fork 2.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.97834 ± 0.00976131 (W 0.0080569) | 8.23699 ± 0.0776245 (W 0.010848) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 3.98845 ± 0.011227 (W 0.00975987) | 8.33754 ± 0.0216479 (W 0.015799) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.4514 ± 0.0308782 (W 0.0170491) | 39.4869 ± 0.0820295 (W 0.0327274) | - | 🔵 faer-fork 2.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 15.4231 ± 0.0330923 (W 0.0367192) | 39.205 ± 0.0850047 (W 0.0582006) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.6245 ± 0.0300765 (W 0.0202214) | 28.1936 ± 0.0551343 (W 0.0759843) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 15.6403 ± 0.0182699 (W 0.0130464) | 28.047 ± 0.116208 (W 0.0221337) | - | 🔵 faer-fork 1.79× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.013325 ± 0.000136504 (W 0.000113293) | 0.012544 ± 7.60723e-05 (W 7.82515e-05) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.013034 ± 4.55741e-05 (W 0.000121323) | 0.012633 ± 9.65419e-05 (W 0.000113235) | - | · | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.030477 ± 0.00026818 (W 0.000280899) | 0.04285 ± 0.000646585 (W 0.000311947) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.029946 ± 0.000161967 (W 0.000212287) | 0.029926 ± 0.000399764 (W 0.000298423) | - | · | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.161984 ± 0.000808563 (W 0.00190291) | 0.280012 ± 0.00836678 (W 0.017079) | 0.242129 ± 0.00316258 (W 0.00332335) | 🔵 faer-fork 1.73× faster than `faer-upstream` | 🔵 faer-fork 1.49× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.195284 ± 0.000680001 (W 0.00194738) | 0.32413 ± 0.00274651 (W 0.0129288) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.15072 ± 0.00334138 (W 0.00201233) | 0.26479 ± 0.00443497 (W 0.00876709) | 0.25187 ± 0.00464849 (W 0.0125681) | 🔵 faer-fork 1.76× faster than `faer-upstream` | 🔵 faer-fork 1.67× faster than `lapack-mkl` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.182598 ± 0.00233677 (W 0.00115467) | 0.294953 ± 0.00933201 (W 0.0066502) | - | 🔵 faer-fork 1.62× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n511/b1` | 0.675356 ± 0.00166512 (W 0.00299664) | 0.910106 ± 0.0249285 (W 0.0314935) | 1.18211 ± 0.00459624 (W 0.010936) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔵 faer-fork 1.75× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n511/b1 [api-inclusive]` | 0.851516 ± 0.00364379 (W 0.00321747) | 1.14261 ± 0.0253259 (W 0.0363384) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.6755 ± 0.00522848 (W 0.00392026) | 0.942958 ± 0.00466254 (W 0.045725) | 1.22843 ± 0.00763597 (W 0.0131588) | 🔵 faer-fork 1.40× faster than `faer-upstream` | 🔵 faer-fork 1.82× faster than `lapack-mkl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 0.809958 ± 0.00463356 (W 0.00285122) | 1.05819 ± 0.0208197 (W 0.0305494) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.87513 ± 0.012029 (W 0.00727183) | 7.30464 ± 0.0108022 (W 0.00847216) | 4.38461 ± 0.0129823 (W 0.00558992) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔴 faer-fork 1.34× slower than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 5.89155 ± 0.0125519 (W 0.011967) | 7.31088 ± 0.0270938 (W 0.0141006) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.78723 ± 0.0218792 (W 0.00929079) | 7.19253 ± 0.00857727 (W 0.00884939) | 4.45168 ± 0.0214666 (W 0.0075274) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔴 faer-fork 1.30× slower than `lapack-mkl` |  |
| `eigh/n257/b1 [api-inclusive]` | 5.81853 ± 0.0129466 (W 0.00936513) | 7.21926 ± 0.0222949 (W 0.00522082) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 24.7919 ± 0.0765569 (W 0.0288984) | 25.4325 ± 0.109166 (W 0.0273461) | 19.1236 ± 0.0904067 (W 0.0547773) | · | 🔴 faer-fork 1.30× slower than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 24.6568 ± 0.0333153 (W 0.122814) | 25.4853 ± 0.0401438 (W 0.0580912) | - | · | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.1947 ± 0.0556973 (W 0.0232028) | 25.786 ± 0.021688 (W 0.0198432) | 19.2157 ± 0.0613584 (W 0.0221506) | · | 🔴 faer-fork 1.31× slower than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 25.0823 ± 0.0439529 (W 0.0095177) | 25.8103 ± 0.0506339 (W 0.0258715) | - | · | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.16669 ± 0.00374285 (W 0.00837745) | 1.97172 ± 0.010519 (W 0.00933243) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.17137 ± 0.00201008 (W 0.00984266) | 2.62928 ± 0.0171016 (W 0.00789572) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.17297 ± 0.0155982 (W 0.00482027) | 2.03616 ± 0.00162529 (W 0.0028952) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.45692 ± 0.0160453 (W 0.0101004) | 1.90145 ± 0.00452958 (W 0.00442218) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 5.99989 ± 0.0218677 (W 0.0103847) | 9.21979 ± 0.0519882 (W 0.0133219) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 12.6917 ± 0.0698933 (W 0.032394) | 14.2855 ± 0.0122921 (W 0.034117) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 6.02124 ± 0.0154108 (W 0.00769243) | 9.35909 ± 0.0304197 (W 0.0188348) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 6.8238 ± 0.0324657 (W 0.0405066) | 8.47854 ± 0.0172939 (W 0.0179119) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.302524 ± 0.00452641 (W 0.00474559) | 0.526465 ± 0.00815919 (W 0.0102789) | 0.287386 ± 0.00168626 (W 0.00363713) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔴 faer-fork 1.05× slower than `lapack-mkl` |  |
| `lu/n255/b1 [api-inclusive]` | 0.610415 ± 0.00424881 (W 0.0045555) | 0.626069 ± 0.00595047 (W 0.00271475) | - | · | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.324691 ± 0.00423657 (W 0.00289214) | 0.530242 ± 0.00877076 (W 0.00998526) | 0.293753 ± 0.00128861 (W 0.00279652) | 🔵 faer-fork 1.63× faster than `faer-upstream` | 🔴 faer-fork 1.11× slower than `lapack-mkl` |  |
| `lu/n257/b1 [api-inclusive]` | 0.654948 ± 0.00351779 (W 0.00542307) | 0.572724 ± 0.00700473 (W 0.00595329) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 1.49695 ± 0.00631256 (W 0.00504799) | 2.16699 ± 0.0135236 (W 0.0370279) | 1.40149 ± 0.00257198 (W 0.00824793) | 🔵 faer-fork 1.45× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 2.99249 ± 0.0203724 (W 0.00653229) | 3.10732 ± 0.0390338 (W 0.0912186) | - | · | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.56227 ± 0.0070369 (W 0.0055691) | 2.24682 ± 0.0160637 (W 0.0382615) | 1.43043 ± 0.000622934 (W 0.00606797) | 🔵 faer-fork 1.44× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.9835 ± 0.00859838 (W 0.00919641) | 2.35784 ± 0.0079475 (W 0.0298704) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.97129 ± 0.004337 (W 0.00188744) | 1.70316 ± 0.00404844 (W 0.00369131) | 1.31724 ± 0.00626075 (W 0.00385873) | 🔵 faer-fork 1.75× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 1.46754 ± 0.0033584 (W 0.00453193) | 2.18756 ± 0.00777599 (W 0.00379222) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 1.00176 ± 0.00546846 (W 0.00885768) | 1.74558 ± 0.00333739 (W 0.003) | 1.37742 ± 0.000586169 (W 0.00474183) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 1.53647 ± 0.00166273 (W 0.00749748) | 2.27062 ± 0.00541162 (W 0.00590466) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 5.50956 ± 0.0279657 (W 0.0224618) | 6.97141 ± 0.0228062 (W 0.0106846) | 7.65584 ± 0.0232135 (W 0.0402725) | 🔵 faer-fork 1.27× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 7.75447 ± 0.0434222 (W 0.0146658) | 9.23623 ± 0.0395739 (W 0.0157243) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 5.60258 ± 0.0181771 (W 0.00669597) | 6.98364 ± 0.0146648 (W 0.0121418) | 7.80823 ± 0.0270752 (W 0.0241747) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.39× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 7.95933 ± 0.0547702 (W 0.0232961) | 8.77547 ± 0.0694536 (W 0.0157315) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 0.581244 ± 0.00116225 (W 0.00348961) | 1.31399 ± 0.00490412 (W 0.00548907) | 0.618633 ± 0.00210838 (W 0.00259076) | 🔵 faer-fork 2.26× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `qr_factor/n257/b1` | 0.600781 ± 0.00347914 (W 0.00280369) | 1.35495 ± 0.00217827 (W 0.0051929) | 0.632035 ± 0.00270996 (W 0.00419537) | 🔵 faer-fork 2.26× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `qr_factor/n511/b1` | 2.92069 ± 0.00274977 (W 0.00782073) | 4.44249 ± 0.0133007 (W 0.0105631) | 3.5449 ± 0.0208996 (W 0.00644151) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 2.87793 ± 0.0062525 (W 0.00516893) | 4.26087 ± 0.0200841 (W 0.00855466) | 3.64165 ± 0.00318042 (W 0.00685137) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `solve/n255/b1` | 0.31293 ± 0.00313145 (W 0.00307566) | 0.479078 ± 0.00608914 (W 0.0116967) | 0.281702 ± 0.00156547 (W 0.00199964) | 🔵 faer-fork 1.53× faster than `faer-upstream` | 🔴 faer-fork 1.11× slower than `lapack-mkl` |  |
| `solve/n255/b1 [api-inclusive]` | 0.621521 ± 0.0033215 (W 0.00480301) | 0.576921 ± 0.0041668 (W 0.0053907) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.326709 ± 0.00118728 (W 0.00320237) | 0.486802 ± 0.00593002 (W 0.00711506) | 0.288344 ± 0.00117379 (W 0.0022478) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `solve/n257/b1 [api-inclusive]` | 0.663764 ± 0.00457473 (W 0.00349373) | 0.542805 ± 0.000814961 (W 0.00781421) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 1.44172 ± 0.00504414 (W 0.00572035) | 2.01896 ± 0.0197693 (W 0.0320295) | 1.39837 ± 0.0053498 (W 0.00716603) | 🔵 faer-fork 1.40× faster than `faer-upstream` | · |  |
| `solve/n511/b1 [api-inclusive]` | 2.94306 ± 0.00976381 (W 0.0100747) | 2.71857 ± 0.0628024 (W 0.0729421) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.51615 ± 0.00633756 (W 0.00479644) | 2.10156 ± 0.0169124 (W 0.0305605) | 1.43014 ± 0.00831748 (W 0.00517984) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `solve/n513/b1 [api-inclusive]` | 2.95763 ± 0.0190671 (W 0.0105909) | 2.28496 ± 0.00393488 (W 0.0248009) | - | 🔴 faer-fork 1.29× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.98423 ± 0.0328751 (W 0.015128) | 13.6312 ± 0.0290914 (W 0.0261551) | 9.69972 ± 0.0150623 (W 0.0122098) | 🔵 faer-fork 1.71× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.99374 ± 0.0384805 (W 0.00758729) | 13.5977 ± 0.0485141 (W 0.0627872) | - | 🔵 faer-fork 1.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 8.14217 ± 0.0467723 (W 0.00833614) | 10.1633 ± 0.0428138 (W 0.0139353) | 9.96574 ± 0.0325181 (W 0.0120155) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 8.17863 ± 0.0341127 (W 0.013112) | 10.2608 ± 0.0678346 (W 0.01706) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 35.7682 ± 0.11627 (W 0.044194) | 54.3035 ± 0.161257 (W 0.0841567) | 40.1759 ± 0.225677 (W 0.157996) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 35.7496 ± 0.0769388 (W 0.033505) | 54.6495 ± 0.125309 (W 0.530599) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 36.4807 ± 0.110467 (W 0.0228816) | 41.3614 ± 0.0318004 (W 0.0443057) | 40.7264 ± 0.100012 (W 0.267012) | 🔵 faer-fork 1.13× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 36.5887 ± 0.0720424 (W 0.0479951) | 41.6164 ± 0.0973921 (W 0.293533) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 4.03646 ± 0.00520347 (W 0.0142824) | 10.8038 ± 0.0107093 (W 0.0107516) | 5.15848 ± 0.0145474 (W 0.00621082) | 🔵 faer-fork 2.68× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 4.02438 ± 0.0033618 (W 0.0129945) | 10.7712 ± 0.0250689 (W 0.0202964) | - | 🔵 faer-fork 2.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 4.17165 ± 0.00773569 (W 0.00694198) | 8.16335 ± 0.0467212 (W 0.00962346) | 5.22672 ± 0.0348274 (W 0.00688627) | 🔵 faer-fork 1.96× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 4.15389 ± 0.0105283 (W 0.00726391) | 8.21798 ± 0.0639478 (W 0.0118335) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 17.5266 ± 0.0547125 (W 0.0355685) | 40.9473 ± 0.0874037 (W 0.0249795) | 19.137 ± 0.0185092 (W 0.0230552) | 🔵 faer-fork 2.34× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 17.463 ± 0.0811668 (W 0.0209686) | 41.0061 ± 0.0691263 (W 0.0335486) | - | 🔵 faer-fork 2.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 17.6672 ± 0.0463475 (W 0.0223876) | 29.6154 ± 0.0373785 (W 0.0353393) | 19.2395 ± 0.0327321 (W 0.0248041) | 🔵 faer-fork 1.68× faster than `faer-upstream` | 🔵 faer-fork 1.09× faster than `lapack-mkl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 17.6658 ± 0.0478838 (W 0.02774) | 29.6404 ± 0.10494 (W 0.0289882) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.015509 ± 0.00010843 (W 0.000287238) | 0.013616 ± 0.000371871 (W 8.8793e-05) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.015869 ± 2.52389e-05 (W 0.000156232) | 0.013896 ± 0.000166764 (W 5.21536e-05) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.039314 ± 0.000800313 (W 0.000570009) | 0.047138 ± 0.00054135 (W 0.00020623) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.039033 ± 0.000262444 (W 0.000379916) | 0.03743 ± 0.00029829 (W 0.000338534) | - | · | - | `*` one row has no measurement here |

## linalg-boundary, f64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.145222 ± 0.000729973 (W 0.00101202) | 0.272797 ± 0.0241659 (W 0.0195955) | 0.210592 ± 0.0021191 (W 0.00413483) | 🔵 faer-fork 1.88× faster than `faer-upstream` | 🔵 faer-fork 1.45× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.179554 ± 0.000543309 (W 0.00214894) | 0.306471 ± 0.00531048 (W 0.0188805) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.135904 ± 0.0029025 (W 0.00116209) | 0.259966 ± 0.00353087 (W 0.028814) | 0.22514 ± 0.00351841 (W 0.0027653) | 🔵 faer-fork 1.91× faster than `faer-upstream` | 🔵 faer-fork 1.66× faster than `lapack-mkl` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.17227 ± 0.00126913 (W 0.000895122) | 0.287908 ± 0.00528932 (W 0.0270799) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 0.50037 ± 0.00176287 (W 0.00305874) | 0.845632 ± 0.00402478 (W 0.0353599) | 0.787718 ± 0.004484 (W 0.0127333) | 🔵 faer-fork 1.69× faster than `faer-upstream` | 🔵 faer-fork 1.57× faster than `lapack-mkl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 0.680308 ± 0.00305668 (W 0.00268674) | 1.02943 ± 0.0187875 (W 0.0194153) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 0.509587 ± 0.00255012 (W 0.00358169) | 0.83928 ± 0.0311295 (W 0.0322238) | 0.816957 ± 0.00226254 (W 0.0104199) | 🔵 faer-fork 1.65× faster than `faer-upstream` | 🔵 faer-fork 1.60× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n513/b1 [api-inclusive]` | 0.660324 ± 0.00346638 (W 0.00135492) | 0.940498 ± 0.00803255 (W 0.0273519) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 3.98614 ± 0.00961457 (W 0.00598641) | 7.28121 ± 0.00983793 (W 0.0255144) | 3.76005 ± 0.0137112 (W 0.00699543) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔴 faer-fork 1.06× slower than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 3.99661 ± 0.0130707 (W 0.0107987) | 7.31779 ± 0.00892373 (W 0.0259128) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 3.96074 ± 0.00726184 (W 0.0064097) | 7.15998 ± 0.0186389 (W 0.0202619) | 3.82706 ± 0.0127607 (W 0.0129816) | 🔵 faer-fork 1.81× faster than `faer-upstream` | · |  |
| `eigh/n257/b1 [api-inclusive]` | 3.97995 ± 0.0119595 (W 0.00713282) | 7.16315 ± 0.0151456 (W 0.0199163) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 16.1376 ± 0.0252594 (W 0.0185611) | 23.8533 ± 0.0597594 (W 0.0780271) | 13.3178 ± 0.0499454 (W 0.0360903) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔴 faer-fork 1.21× slower than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 16.2351 ± 0.0486469 (W 0.0165419) | 23.8863 ± 0.0995047 (W 0.211777) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 16.5933 ± 0.0557804 (W 0.0205998) | 24.2005 ± 0.0494455 (W 0.0540204) | 13.5946 ± 0.0330401 (W 0.0288488) | 🔵 faer-fork 1.46× faster than `faer-upstream` | 🔴 faer-fork 1.22× slower than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 16.6348 ± 0.18458 (W 0.0326941) | 24.3367 ± 0.0433053 (W 0.0610642) | - | 🔵 faer-fork 1.46× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.08802 ± 0.00499187 (W 0.00598308) | 1.96973 ± 0.00511614 (W 0.00605845) | - | 🔵 faer-fork 1.81× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 2.1075 ± 0.0122224 (W 0.00604393) | 2.63015 ± 0.0201397 (W 0.00710413) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.10305 ± 0.00927125 (W 0.00640702) | 2.04627 ± 0.00485562 (W 0.00680222) | - | 🔵 faer-fork 1.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 1.39566 ± 0.00472115 (W 0.00569925) | 1.89365 ± 0.0123513 (W 0.00504952) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 4.58613 ± 0.0191993 (W 0.0106201) | 8.27164 ± 0.0229542 (W 0.0112413) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 11.3687 ± 0.0237626 (W 0.0226678) | 13.5418 ± 0.0519785 (W 0.0200439) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 4.60785 ± 0.0227464 (W 0.0120979) | 8.4487 ± 0.0162078 (W 0.0123381) | - | 🔵 faer-fork 1.83× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 5.64128 ± 0.0532728 (W 0.0203147) | 7.63703 ± 0.025501 (W 0.0125924) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.280724 ± 0.00124477 (W 0.00184131) | 0.517856 ± 0.00218198 (W 0.0183258) | 0.275572 ± 0.00146736 (W 0.00219018) | 🔵 faer-fork 1.84× faster than `faer-upstream` | · |  |
| `lu/n255/b1 [api-inclusive]` | 0.585815 ± 0.0057187 (W 0.00293345) | 0.624684 ± 0.026721 (W 0.0227697) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n257/b1` | 0.292323 ± 0.001472 (W 0.00148861) | 0.520673 ± 0.00276389 (W 0.0289564) | 0.282506 ± 0.00244917 (W 0.00308301) | 🔵 faer-fork 1.78× faster than `faer-upstream` | · |  |
| `lu/n257/b1 [api-inclusive]` | 0.627496 ± 0.00334828 (W 0.00346509) | 0.556855 ± 0.031874 (W 0.0105918) | - | 🔴 faer-fork 1.13× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `lu/n511/b1` | 1.13188 ± 0.00237066 (W 0.00590011) | 2.06656 ± 0.0302933 (W 0.0782363) | 1.00429 ± 0.00213208 (W 0.0059143) | 🔵 faer-fork 1.83× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 2.62371 ± 0.00707677 (W 0.00452629) | 2.88417 ± 0.0106954 (W 0.0568991) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 1.17401 ± 0.00635886 (W 0.00484282) | 2.10828 ± 0.0247387 (W 0.0511683) | 1.03636 ± 0.00500001 (W 0.00320403) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 2.63257 ± 0.0149502 (W 0.00919044) | 2.24769 ± 0.0316848 (W 0.0375018) | - | 🔴 faer-fork 1.17× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 0.803682 ± 0.00158273 (W 0.00367363) | 1.99753 ± 0.0318132 (W 0.00577113) | 0.983338 ± 0.000832188 (W 0.0024492) | 🔵 faer-fork 2.49× faster than `faer-upstream` | 🔵 faer-fork 1.22× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 1.28766 ± 0.00668242 (W 0.00622701) | 2.11834 ± 0.212115 (W 0.0204028) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n257/b1` | 0.824132 ± 0.00390924 (W 0.00297464) | 2.05938 ± 0.00476737 (W 0.0110943) | 1.04986 ± 0.00432597 (W 0.00348865) | 🔵 faer-fork 2.50× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 1.34855 ± 0.000211972 (W 0.00703651) | 2.1749 ± 0.165329 (W 0.0746231) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n511/b1` | 3.16619 ± 0.000632816 (W 0.00383458) | 6.96378 ± 0.0742191 (W 0.0436671) | 4.30321 ± 0.0205412 (W 0.0104476) | 🔵 faer-fork 2.20× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 5.48371 ± 0.23466 (W 0.023529) | 7.97367 ± 0.983484 (W 0.546563) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1` | 3.54991 ± 0.00250568 (W 0.006487) | 6.85012 ± 0.0452353 (W 0.058158) | 4.45425 ± 0.0253572 (W 0.0134892) | 🔵 faer-fork 1.93× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 5.94043 ± 0.0129349 (W 0.017284) | 7.39201 ± 0.322652 (W 0.687946) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n255/b1` | 0.568462 ± 0.00296156 (W 0.00246098) | 1.57933 ± 0.00425665 (W 0.00799969) | 0.553044 ± 0.0126917 (W 0.00289372) | 🔵 faer-fork 2.78× faster than `faer-upstream` | · |  |
| `qr_factor/n257/b1` | 0.581231 ± 0.00184244 (W 0.00307351) | 1.63048 ± 0.0136247 (W 0.0101035) | 0.56474 ± 0.00249439 (W 0.00380423) | 🔵 faer-fork 2.81× faster than `faer-upstream` | · |  |
| `qr_factor/n511/b1` | 1.75529 ± 0.00565558 (W 0.0032275) | 4.71518 ± 0.0213535 (W 0.0312202) | 2.15042 ± 0.00282504 (W 0.00913315) | 🔵 faer-fork 2.69× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 2.0633 ± 0.0032474 (W 0.00478475) | 4.48306 ± 0.00976132 (W 0.0327942) | 2.18883 ± 0.00911141 (W 0.00515271) | 🔵 faer-fork 2.17× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `solve/n255/b1` | 0.2845 ± 0.00108255 (W 0.00152122) | 0.467982 ± 0.0257142 (W 0.02202) | 0.270716 ± 0.00117926 (W 0.00256894) | 🔵 faer-fork 1.64× faster than `faer-upstream` | · | `!` between-start range over 5% |
| `solve/n255/b1 [api-inclusive]` | 0.591059 ± 0.0027649 (W 0.00348743) | 0.588046 ± 0.0231324 (W 0.0252359) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n257/b1` | 0.291401 ± 0.00359007 (W 0.00229255) | 0.51498 ± 0.0222282 (W 0.0311248) | 0.277028 ± 0.00330347 (W 0.00274755) | 🔵 faer-fork 1.77× faster than `faer-upstream` | 🔴 faer-fork 1.05× slower than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n257/b1 [api-inclusive]` | 0.633097 ± 0.00416775 (W 0.00362502) | 0.572881 ± 0.0329872 (W 0.0334047) | - | 🔴 faer-fork 1.11× slower than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `solve/n511/b1` | 1.09135 ± 0.00298791 (W 0.00322769) | 1.90047 ± 0.0111917 (W 0.0305069) | 0.97099 ± 0.00437812 (W 0.00418744) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` |  |
| `solve/n511/b1 [api-inclusive]` | 2.59329 ± 0.00278286 (W 0.0104112) | 2.52898 ± 0.00297577 (W 0.0579076) | - | · | - | `*` one row has no measurement here |
| `solve/n513/b1` | 1.13663 ± 0.00323233 (W 0.0070861) | 2.00043 ± 0.0558658 (W 0.0703493) | 1.01359 ± 0.00453227 (W 0.00399069) | 🔵 faer-fork 1.76× faster than `faer-upstream` | 🔴 faer-fork 1.12× slower than `lapack-mkl` | `!` between-start range over 5% |
| `solve/n513/b1 [api-inclusive]` | 2.58959 ± 0.0292562 (W 0.00794853) | 2.13602 ± 0.0148726 (W 0.0595722) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 7.2066 ± 0.00921429 (W 0.00911969) | 13.6936 ± 0.0620867 (W 0.0280264) | 8.95902 ± 0.0192109 (W 0.0315706) | 🔵 faer-fork 1.90× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 7.23881 ± 0.0262567 (W 0.0089688) | 13.6562 ± 0.0740887 (W 0.0260537) | - | 🔵 faer-fork 1.89× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 7.39662 ± 0.0115016 (W 0.00996652) | 10.1622 ± 0.0338326 (W 0.0491305) | 9.24235 ± 0.129998 (W 0.198688) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 7.39082 ± 0.0164449 (W 0.0131696) | 10.1988 ± 0.0673124 (W 0.0364544) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 29.43 ± 0.138277 (W 0.0361711) | 51.1249 ± 0.063148 (W 0.0827647) | 32.6559 ± 0.178087 (W 0.0250397) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 29.5608 ± 0.149727 (W 0.0488651) | 51.1913 ± 0.0616599 (W 0.336885) | - | 🔵 faer-fork 1.73× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 29.8888 ± 0.078574 (W 0.0312161) | 37.8935 ± 0.102943 (W 0.0745101) | 33.0644 ± 0.130272 (W 0.0948406) | 🔵 faer-fork 1.27× faster than `faer-upstream` | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 29.8869 ± 0.0887135 (W 0.0428625) | 38.1663 ± 0.22645 (W 0.147014) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 3.87388 ± 0.00744391 (W 0.0107176) | 10.9025 ± 0.0490422 (W 0.392318) | 4.99812 ± 0.0220849 (W 0.0094672) | 🔵 faer-fork 2.81× faster than `faer-upstream` | 🔵 faer-fork 1.29× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 3.87869 ± 0.0160565 (W 0.00836952) | 10.8576 ± 0.0372209 (W 0.0167089) | - | 🔵 faer-fork 2.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 3.99224 ± 0.02597 (W 0.0054362) | 8.27585 ± 0.0563903 (W 0.0114209) | 5.09933 ± 0.0357923 (W 0.00647465) | 🔵 faer-fork 2.07× faster than `faer-upstream` | 🔵 faer-fork 1.28× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 4.00342 ± 0.0137661 (W 0.00685489) | 8.24573 ± 0.0213699 (W 0.00563541) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 15.4122 ± 0.0221032 (W 0.0160061) | 39.3409 ± 0.199456 (W 0.0614133) | 16.4155 ± 0.0408689 (W 0.0165638) | 🔵 faer-fork 2.55× faster than `faer-upstream` | 🔵 faer-fork 1.07× faster than `lapack-mkl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 15.4142 ± 0.0819686 (W 0.016767) | 39.1707 ± 0.0319126 (W 0.0461074) | - | 🔵 faer-fork 2.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 15.5705 ± 0.0615288 (W 0.0137173) | 27.9531 ± 0.0259117 (W 0.0298201) | 16.4845 ± 0.0235781 (W 0.0230683) | 🔵 faer-fork 1.80× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 15.513 ± 0.0169095 (W 0.0254281) | 27.9433 ± 0.057089 (W 0.0507856) | - | 🔵 faer-fork 1.80× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.015418 ± 0.000346094 (W 0.000185788) | 0.013795 ± 0.000119862 (W 4.83456e-05) | - | 🔴 faer-fork 1.12× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n257/b1` | 0.015379 ± 0.000171553 (W 0.000238326) | 0.013926 ± 1.1547e-05 (W 6.42651e-05) | - | 🔴 faer-fork 1.10× slower than `faer-upstream` | - | `*` one row has no measurement here; dispatch-dominated (row under 20 us) |
| `triangular_solve/n511/b1` | 0.037931 ± 0.000234592 (W 0.000379748) | 0.047068 ± 0.000682797 (W 0.000296083) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.03777 ± 0.000531743 (W 0.000289749) | 0.035656 ± 0.000462613 (W 0.000317832) | - | 🔴 faer-fork 1.06× slower than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.506274 ± 0.000901007 (W 0.00204006) | 1.66902 ± 0.0122178 (W 0.0092162) | - | 🔵 faer-fork 3.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n255/b1 [api-inclusive]` | 0.579485 ± 0.00106388 (W 0.00302735) | 0.737929 ± 0.00217246 (W 0.00614656) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.460202 ± 0.00190465 (W 0.00169706) | 1.66894 ± 0.00543553 (W 0.128556) | - | 🔵 faer-fork 3.63× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1 [api-inclusive]` | 0.523451 ± 0.00156173 (W 0.00343037) | 0.699798 ± 0.0100183 (W 0.0122201) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.29362 ± 0.00485068 (W 0.00504155) | 5.00007 ± 0.0204545 (W 0.115962) | - | 🔵 faer-fork 2.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1 [api-inclusive]` | 2.80786 ± 0.0132393 (W 0.00936265) | 3.16505 ± 0.0536165 (W 0.0868005) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.23698 ± 0.00788955 (W 0.00522741) | 5.03407 ± 0.0309096 (W 0.117164) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1 [api-inclusive]` | 2.69296 ± 0.0113276 (W 0.00756454) | 2.9936 ± 0.0790618 (W 0.109743) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.31588 ± 0.0352458 (W 0.0120724) | 11.7473 ± 0.0235154 (W 0.0502089) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 8.34189 ± 0.0165028 (W 0.0177014) | 11.7271 ± 0.0606113 (W 0.0289148) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.34659 ± 0.0291228 (W 0.0294278) | 11.6354 ± 0.00974044 (W 0.0102121) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 8.35402 ± 0.0117066 (W 0.0116384) | 11.6612 ± 0.0485876 (W 0.017184) | - | 🔵 faer-fork 1.40× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.4548 ± 0.0790693 (W 0.0392941) | 45.2532 ± 0.0621542 (W 0.0572335) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 40.5645 ± 0.145993 (W 0.0496048) | 45.2901 ± 0.085137 (W 0.044305) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.4054 ± 0.25824 (W 0.0589765) | 45.6667 ± 0.086904 (W 0.040578) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 40.4861 ± 0.00640728 (W 0.0399357) | 45.837 ± 0.0569342 (W 0.0521863) | - | 🔵 faer-fork 1.13× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.98185 ± 0.000846522 (W 0.00526211) | 4.28237 ± 0.010688 (W 0.00927552) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 4.06722 ± 0.0153201 (W 0.0157911) | 5.77841 ± 0.037898 (W 0.0154547) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 2.0363 ± 0.00410166 (W 0.00393731) | 4.40386 ± 0.0079003 (W 0.0131174) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.7096 ± 0.0111465 (W 0.00738719) | 4.36712 ± 0.0214466 (W 0.0111521) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 11.2256 ± 0.068512 (W 0.0137879) | 18.9741 ± 0.0664723 (W 0.0268266) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 20.7099 ± 0.0694301 (W 0.0550225) | 24.877 ± 0.0240036 (W 0.0334937) | - | 🔵 faer-fork 1.20× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 11.3773 ± 0.0376579 (W 0.032305) | 19.1959 ± 0.128412 (W 0.0432655) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 15.029 ± 0.0363192 (W 0.0347736) | 19.0262 ± 0.0926971 (W 0.0394741) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.947607 ± 0.00304166 (W 0.0034074) | 1.34346 ± 0.0220468 (W 0.0189213) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.70718 ± 0.0229523 (W 0.0158372) | 1.38923 ± 0.00916071 (W 0.0148571) | - | 🔴 faer-fork 1.23× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 1.02782 ± 0.0060815 (W 0.0059705) | 1.33335 ± 0.0132776 (W 0.021273) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.74162 ± 0.00517208 (W 0.00512855) | 1.28897 ± 0.0139794 (W 0.0147267) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 5.05204 ± 0.00596722 (W 0.0372058) | 6.23859 ± 0.0550735 (W 0.146714) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 9.31105 ± 0.0964135 (W 0.0492948) | 6.79824 ± 0.0184403 (W 0.131839) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 5.31964 ± 0.0272257 (W 0.0112218) | 6.28335 ± 0.0217833 (W 0.11396) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 9.38412 ± 0.0492931 (W 0.0116343) | 6.65028 ± 0.050805 (W 0.0842239) | - | 🔴 faer-fork 1.41× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 3.05976 ± 0.0331073 (W 0.00876014) | 4.92768 ± 0.00387239 (W 0.00922596) | - | 🔵 faer-fork 1.61× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 4.18403 ± 0.00909501 (W 0.01099) | 5.10235 ± 0.0626463 (W 0.00841108) | - | 🔵 faer-fork 1.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.18678 ± 0.0160213 (W 0.00255281) | 5.0852 ± 0.0309701 (W 0.0175044) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 4.29879 ± 0.0167813 (W 0.0117033) | 5.20875 ± 0.00558915 (W 0.00690991) | - | 🔵 faer-fork 1.21× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 19.6763 ± 0.0589468 (W 0.0162511) | 25.8302 ± 0.0677983 (W 0.0593297) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 25.4348 ± 0.177869 (W 0.050041) | 26.814 ± 0.137551 (W 0.136504) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 20.1596 ± 0.0206101 (W 0.0462565) | 27.327 ± 0.104048 (W 0.0549502) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 25.961 ± 0.115215 (W 0.114025) | 27.3632 ± 0.0384991 (W 0.0528211) | - | 🔵 faer-fork 1.05× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.61929 ± 0.00462231 (W 0.00541953) | 3.468 ± 0.00518467 (W 0.0120998) | - | 🔵 faer-fork 2.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.68155 ± 0.00966159 (W 0.00414211) | 3.54352 ± 0.00926473 (W 0.0131753) | - | 🔵 faer-fork 2.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 9.37835 ± 0.0143694 (W 0.0427591) | 15.6732 ± 0.0594952 (W 0.0469438) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 9.58907 ± 0.022259 (W 0.0283787) | 16.814 ± 0.0938795 (W 0.0393525) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.854192 ± 0.00198103 (W 0.00141522) | 1.31315 ± 0.0123017 (W 0.0153949) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.72624 ± 0.00432789 (W 0.014107) | 1.2754 ± 0.0117303 (W 0.0123417) | - | 🔴 faer-fork 1.35× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 1.03479 ± 0.00920708 (W 0.00278461) | 1.33024 ± 0.00226184 (W 0.021251) | - | 🔵 faer-fork 1.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.74767 ± 0.00633717 (W 0.00495315) | 1.28378 ± 0.0067383 (W 0.0131179) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 5.08884 ± 0.053736 (W 0.0389485) | 6.33008 ± 0.0222496 (W 0.10807) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 9.31581 ± 0.0166323 (W 0.0693297) | 6.71823 ± 0.0697665 (W 0.102196) | - | 🔴 faer-fork 1.39× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 5.33282 ± 0.0129348 (W 0.00561487) | 6.2946 ± 0.0432855 (W 0.130221) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 9.38052 ± 0.0745942 (W 0.0283825) | 6.65349 ± 0.0728141 (W 0.108091) | - | 🔴 faer-fork 1.41× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.9498 ± 0.0221842 (W 0.0548623) | 21.6705 ± 0.0600979 (W 0.0375296) | - | 🔵 faer-fork 1.67× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 12.9513 ± 0.00730142 (W 0.0192897) | 21.8217 ± 0.0848612 (W 0.0571142) | - | 🔵 faer-fork 1.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 13.1829 ± 0.0381846 (W 0.038375) | 17.2837 ± 0.0577488 (W 0.0472895) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 13.2658 ± 0.0405181 (W 0.014551) | 17.3434 ± 0.0804994 (W 0.0565722) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 68.9559 ± 0.0633634 (W 0.0418078) | 85.9898 ± 0.0644111 (W 0.100655) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 69.1647 ± 0.0682284 (W 0.0744049) | 88.2978 ± 0.148765 (W 0.970233) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 70.5986 ± 0.187836 (W 0.164266) | 75.244 ± 0.106529 (W 0.0764681) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 70.7047 ± 0.191339 (W 0.133526) | 77.4954 ± 0.222247 (W 1.04293) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.38072 ± 0.0217671 (W 0.00830235) | 16.5816 ± 0.132167 (W 0.0147526) | - | 🔵 faer-fork 2.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 6.3863 ± 0.0216893 (W 0.0126392) | 16.5136 ± 0.128728 (W 0.0416187) | - | 🔵 faer-fork 2.59× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.5444 ± 0.0115976 (W 0.0109513) | 12.9132 ± 0.128197 (W 0.0276954) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 6.53603 ± 0.0366847 (W 0.0186126) | 12.8591 ± 0.0352301 (W 0.0222355) | - | 🔵 faer-fork 1.97× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 34.3833 ± 0.0739344 (W 0.0535553) | 56.8927 ± 0.124324 (W 0.0783255) | - | 🔵 faer-fork 1.65× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 34.4209 ± 0.151768 (W 0.0517309) | 57.1284 ± 0.100899 (W 0.0844629) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 34.963 ± 0.0677112 (W 0.0496725) | 47.6113 ± 0.154032 (W 0.0602571) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 34.908 ± 0.154752 (W 0.0450797) | 47.5931 ± 0.0690566 (W 0.0748119) | - | 🔵 faer-fork 1.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.022282 ± 0.000142432 (W 9.13783e-05) | 0.023835 ± 8.0829e-05 (W 6.88934e-05) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.022341 ± 0.000128791 (W 0.000127358) | 0.024225 ± 0.000218351 (W 0.000101273) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.057447 ± 0.000684485 (W 0.000262486) | 0.086591 ± 0.000968785 (W 0.000372856) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.055654 ± 0.000963249 (W 0.000342598) | 0.061213 ± 0.000548846 (W 0.00186357) | - | 🔵 faer-fork 1.10× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, c order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.4356 ± 0.00356168 (W 0.00310354) | 1.35563 ± 0.167778 (W 0.180554) | - | 🔵 faer-fork 3.11× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n255/b1 [api-inclusive]` | 0.509441 ± 0.00261707 (W 0.00189879) | 0.7045 ± 0.00559436 (W 0.0147436) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.39105 ± 0.000326225 (W 0.0023368) | 1.10374 ± 0.146287 (W 0.148662) | - | 🔵 faer-fork 2.82× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n257/b1 [api-inclusive]` | 0.459348 ± 0.00500339 (W 0.00317191) | 0.654342 ± 0.00849236 (W 0.0105998) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.57652 ± 0.00437962 (W 0.0159454) | 3.54265 ± 0.132489 (W 0.25057) | - | 🔵 faer-fork 2.25× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n511/b1 [api-inclusive]` | 2.09824 ± 0.00242222 (W 0.00903586) | 2.47699 ± 0.0127933 (W 0.0486847) | - | 🔵 faer-fork 1.18× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.49596 ± 0.00168046 (W 0.00325343) | 3.32254 ± 0.146107 (W 0.254696) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `cholesky/n513/b1 [api-inclusive]` | 2.00304 ± 0.00636769 (W 0.0042365) | 2.34061 ± 0.0107463 (W 0.0206805) | - | 🔵 faer-fork 1.17× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.55046 ± 0.0192067 (W 0.00794126) | 11.5799 ± 0.0532392 (W 0.0354336) | - | 🔵 faer-fork 2.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1 [api-inclusive]` | 5.56596 ± 0.0176965 (W 0.00819702) | 11.5586 ± 0.0620036 (W 0.0622646) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.52447 ± 0.0173961 (W 0.0232581) | 11.4174 ± 0.0214898 (W 0.0433558) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1 [api-inclusive]` | 5.5307 ± 0.0163555 (W 0.0066848) | 11.438 ± 0.0298455 (W 0.0689944) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.4838 ± 0.103012 (W 0.0398288) | 39.2672 ± 0.0319469 (W 0.229391) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1 [api-inclusive]` | 25.5534 ± 0.196241 (W 0.0170794) | 39.5753 ± 0.104075 (W 0.492136) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 26.0387 ± 0.0644376 (W 0.0215901) | 39.5964 ± 0.123212 (W 0.23264) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1 [api-inclusive]` | 26.1589 ± 0.040552 (W 0.0982681) | 39.5774 ± 0.194226 (W 0.475775) | - | 🔵 faer-fork 1.51× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.60954 ± 0.111444 (W 0.00575864) | 4.30501 ± 0.0134094 (W 0.00333629) | - | 🔵 faer-fork 2.67× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.74445 ± 0.0232044 (W 0.00724422) | 5.75845 ± 0.0558178 (W 0.013258) | - | 🔵 faer-fork 1.54× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.64471 ± 0.00581974 (W 0.00411664) | 4.46166 ± 0.0334581 (W 0.00695687) | - | 🔵 faer-fork 2.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.36177 ± 0.0137219 (W 0.00829969) | 4.37142 ± 0.0303597 (W 0.00726275) | - | 🔵 faer-fork 1.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.28355 ± 0.022716 (W 0.0342942) | 16.7413 ± 0.0253724 (W 0.0422608) | - | 🔵 faer-fork 2.30× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 17.0754 ± 0.0471599 (W 0.0293107) | 22.5948 ± 0.0404908 (W 0.0408892) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.37159 ± 0.0167477 (W 0.00675198) | 16.7407 ± 0.0756615 (W 0.0407563) | - | 🔵 faer-fork 2.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 11.4455 ± 0.054902 (W 0.0407167) | 16.6408 ± 0.0581082 (W 0.037407) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.807134 ± 0.0027795 (W 0.00218041) | 1.27856 ± 0.0153783 (W 0.0173395) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1 [api-inclusive]` | 1.55478 ± 0.00586605 (W 0.00487453) | 1.28507 ± 0.0110717 (W 0.0108615) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.807896 ± 0.00516391 (W 0.00194446) | 1.23456 ± 0.0248885 (W 0.0222164) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1 [api-inclusive]` | 1.55724 ± 0.00647849 (W 0.00483784) | 1.13406 ± 0.00739396 (W 0.0133946) | - | 🔴 faer-fork 1.37× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.38406 ± 0.0171097 (W 0.00563712) | 5.04455 ± 0.020594 (W 0.0184717) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1 [api-inclusive]` | 7.77588 ± 0.114426 (W 0.0128013) | 5.32123 ± 0.00589217 (W 0.0224805) | - | 🔴 faer-fork 1.46× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.48458 ± 0.00616384 (W 0.0109407) | 5.00965 ± 0.0782506 (W 0.0396096) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1 [api-inclusive]` | 7.71652 ± 0.0268759 (W 0.045415) | 5.05924 ± 0.0360233 (W 0.0328799) | - | 🔴 faer-fork 1.53× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.08458 ± 0.00578169 (W 0.00429777) | 4.85114 ± 0.0198495 (W 0.0392035) | - | 🔵 faer-fork 2.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1 [api-inclusive]` | 3.27398 ± 0.0126889 (W 0.0128358) | 4.17927 ± 0.184782 (W 0.0830999) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n257/b1` | 2.17706 ± 0.00161315 (W 0.00277752) | 4.97962 ± 0.0346779 (W 0.0393568) | - | 🔵 faer-fork 2.29× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1 [api-inclusive]` | 3.34653 ± 0.0207143 (W 0.0133205) | 4.82748 ± 0.47217 (W 0.120185) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n511/b1` | 10.5199 ± 0.0177968 (W 0.017883) | 21.7411 ± 0.0299486 (W 0.271945) | - | 🔵 faer-fork 2.07× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1 [api-inclusive]` | 16.4692 ± 0.0780824 (W 0.102094) | 21.4318 ± 1.54399 (W 1.36051) | - | 🔵 faer-fork 1.30× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1` | 11.4107 ± 0.0277025 (W 0.0116085) | 21.7676 ± 0.101685 (W 0.249993) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1 [api-inclusive]` | 17.3926 ± 0.0828993 (W 0.0264334) | 22.8901 ± 0.404051 (W 1.83891) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.31098 ± 0.00260935 (W 0.00433718) | 3.53656 ± 0.0238133 (W 0.016789) | - | 🔵 faer-fork 2.70× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n257/b1` | 1.36302 ± 0.000513508 (W 0.00250671) | 3.65071 ± 0.0129459 (W 0.020129) | - | 🔵 faer-fork 2.68× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n511/b1` | 5.05257 ± 0.0086977 (W 0.0142519) | 14.252 ± 0.0769522 (W 0.1182) | - | 🔵 faer-fork 2.82× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n513/b1` | 5.70942 ± 0.0146582 (W 0.144162) | 14.3015 ± 0.0701911 (W 0.109946) | - | 🔵 faer-fork 2.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1` | 0.725482 ± 0.000375623 (W 0.00345616) | 1.24227 ± 0.0113742 (W 0.0189706) | - | 🔵 faer-fork 1.71× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n255/b1 [api-inclusive]` | 1.55482 ± 0.00356638 (W 0.00793152) | 1.15861 ± 0.0133012 (W 0.0153454) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.817096 ± 0.00325014 (W 0.00178613) | 1.24586 ± 0.0175044 (W 0.0358984) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1 [api-inclusive]` | 1.56818 ± 0.00944004 (W 0.00398805) | 1.13653 ± 0.0199357 (W 0.0117804) | - | 🔴 faer-fork 1.38× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.37393 ± 0.0171197 (W 0.0168115) | 5.06858 ± 0.0390377 (W 0.0311794) | - | 🔵 faer-fork 1.50× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1 [api-inclusive]` | 7.76501 ± 0.0260861 (W 0.0303815) | 5.24 ± 0.0329618 (W 0.0298638) | - | 🔴 faer-fork 1.48× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.48075 ± 0.0161229 (W 0.00707854) | 5.06182 ± 0.0042016 (W 0.0439374) | - | 🔵 faer-fork 1.45× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1 [api-inclusive]` | 7.7481 ± 0.101791 (W 0.0403677) | 5.05333 ± 0.012916 (W 0.0550534) | - | 🔴 faer-fork 1.53× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.7142 ± 0.0277153 (W 0.0553708) | 21.34 ± 0.141553 (W 1.97768) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1 [api-inclusive]` | 10.7787 ± 0.0226701 (W 0.0215682) | 21.3628 ± 0.130469 (W 0.108677) | - | 🔵 faer-fork 1.98× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.9529 ± 0.0536566 (W 0.0170249) | 16.6659 ± 0.0305462 (W 0.0681139) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1 [api-inclusive]` | 10.9917 ± 0.0399408 (W 0.0210388) | 16.8028 ± 0.269533 (W 0.0915947) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 49.602 ± 0.057207 (W 0.0433817) | 73.699 ± 0.253247 (W 0.315609) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1 [api-inclusive]` | 49.8269 ± 0.0601681 (W 0.182552) | 73.9054 ± 0.125664 (W 0.41443) | - | 🔵 faer-fork 1.48× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 50.6242 ± 0.116778 (W 0.241034) | 62.1817 ± 0.280004 (W 0.434504) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1 [api-inclusive]` | 50.7201 ± 0.0336653 (W 0.0771425) | 62.3064 ± 0.242824 (W 0.484016) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.73012 ± 0.025715 (W 0.00640681) | 16.7538 ± 0.03437 (W 0.0304109) | - | 🔵 faer-fork 2.92× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1 [api-inclusive]` | 5.74184 ± 0.0311834 (W 0.00927023) | 16.3179 ± 0.0877046 (W 0.0469951) | - | 🔵 faer-fork 2.84× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.87167 ± 0.0269595 (W 0.01446) | 12.7102 ± 0.32734 (W 0.0230133) | - | 🔵 faer-fork 2.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1 [api-inclusive]` | 5.89741 ± 0.0240082 (W 0.00564625) | 13.2218 ± 0.313807 (W 0.0507929) | - | 🔵 faer-fork 2.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 26.5966 ± 0.0284759 (W 0.0411472) | 50.6045 ± 0.0911516 (W 0.0712082) | - | 🔵 faer-fork 1.90× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1 [api-inclusive]` | 26.5465 ± 0.0418241 (W 0.0191263) | 50.7513 ± 0.187596 (W 0.141424) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 27.0155 ± 0.0258516 (W 0.0396699) | 41.0204 ± 0.0552189 (W 0.0929004) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1 [api-inclusive]` | 27.0579 ± 0.0954614 (W 0.0385621) | 41.1622 ± 0.137986 (W 0.105518) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.02148 ± 0.000160728 (W 0.000125961) | 0.023804 ± 0.000196011 (W 9.66395e-05) | - | 🔵 faer-fork 1.11× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.02155 ± 0.000157703 (W 0.000104189) | 0.024114 ± 0.000226498 (W 7.96637e-05) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.050584 ± 0.000831116 (W 0.000223404) | 0.088454 ± 0.000178292 (W 0.000653764) | - | 🔵 faer-fork 1.75× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.050965 ± 0.000534716 (W 0.000197282) | 0.058038 ± 0.000190526 (W 0.00037758) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 4T (CPU 4-7)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.496991 ± 0.00110082 (W 0.00290943) | 0.64092 ± 0.00202857 (W 0.00443843) | 0.440007 ± 0.00966548 (W 0.00789489) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔴 faer-fork 1.13× slower than `lapack-mkl` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.577747 ± 0.00300767 (W 0.00267387) | 0.734372 ± 0.00259974 (W 0.00367257) | - | 🔵 faer-fork 1.27× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.447354 ± 0.000785343 (W 0.00313018) | 0.637769 ± 0.00115339 (W 0.0110065) | 0.46461 ± 0.00170087 (W 0.0052922) | 🔵 faer-fork 1.43× faster than `faer-upstream` | · |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.520986 ± 0.00213252 (W 0.00252204) | 0.691229 ± 0.00368416 (W 0.00940825) | - | 🔵 faer-fork 1.33× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 2.24488 ± 0.00186066 (W 0.00725141) | 2.6668 ± 0.0847233 (W 0.119564) | 2.47987 ± 0.00439752 (W 0.0164471) | 🔵 faer-fork 1.19× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n511/b1 [api-inclusive]` | 2.59544 ± 0.019499 (W 0.00488154) | 3.07646 ± 0.0261649 (W 0.0688783) | - | 🔵 faer-fork 1.19× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 2.20592 ± 0.00812625 (W 0.0026474) | 2.6692 ± 0.0940622 (W 0.119797) | 2.56291 ± 0.00729749 (W 0.00531435) | 🔵 faer-fork 1.21× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` | `!` between-start range over 5% |
| `cholesky/n513/b1 [api-inclusive]` | 2.53243 ± 0.0160912 (W 0.0045167) | 2.94955 ± 0.0260427 (W 0.0892014) | - | 🔵 faer-fork 1.16× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 8.31127 ± 0.00530033 (W 0.0171454) | 11.7305 ± 0.00639701 (W 0.0181323) | 9.30035 ± 0.0109242 (W 0.0119017) | 🔵 faer-fork 1.41× faster than `faer-upstream` | 🔵 faer-fork 1.12× faster than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 8.32738 ± 0.00568877 (W 0.015955) | 11.7984 ± 0.026313 (W 0.0162023) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 8.33425 ± 0.0201412 (W 0.0149681) | 11.6153 ± 0.00990983 (W 0.0218594) | 9.45495 ± 0.0139416 (W 0.0099902) | 🔵 faer-fork 1.39× faster than `faer-upstream` | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |
| `eigh/n257/b1 [api-inclusive]` | 8.35335 ± 0.0216427 (W 0.0118434) | 11.6524 ± 0.0210928 (W 0.0158999) | - | 🔵 faer-fork 1.39× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 40.2463 ± 0.0292726 (W 0.0278942) | 45.1571 ± 0.0413592 (W 0.0312835) | 49.3937 ± 0.152348 (W 0.0477902) | 🔵 faer-fork 1.12× faster than `faer-upstream` | 🔵 faer-fork 1.23× faster than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 40.4064 ± 0.0546016 (W 0.416851) | 45.2715 ± 0.021506 (W 0.0452444) | - | 🔵 faer-fork 1.12× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 40.2362 ± 0.076858 (W 0.0228724) | 45.906 ± 0.199018 (W 0.0391367) | 49.7565 ± 0.0496433 (W 0.0528668) | 🔵 faer-fork 1.14× faster than `faer-upstream` | 🔵 faer-fork 1.24× faster than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 40.2595 ± 0.0672419 (W 0.0357711) | 45.888 ± 0.112824 (W 0.0441351) | - | 🔵 faer-fork 1.14× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.86211 ± 0.00714516 (W 0.00623361) | 4.32275 ± 0.0416186 (W 0.00662257) | - | 🔵 faer-fork 2.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.88898 ± 0.013094 (W 0.0117108) | 5.7168 ± 0.052904 (W 0.0113369) | - | 🔵 faer-fork 1.47× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.91249 ± 0.0116238 (W 0.00864248) | 4.41576 ± 0.0185249 (W 0.00337816) | - | 🔵 faer-fork 2.31× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.52985 ± 0.00827186 (W 0.00729598) | 4.39966 ± 0.0356589 (W 0.0212739) | - | 🔵 faer-fork 1.74× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 10.8336 ± 0.0340922 (W 0.0139521) | 19.0692 ± 0.0520384 (W 0.0405022) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1 [api-inclusive]` | 19.6498 ± 0.0840071 (W 0.0288799) | 24.6408 ± 0.0792151 (W 0.0325676) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 10.9026 ± 0.0591402 (W 0.023872) | 19.2201 ± 0.0569362 (W 0.0339548) | - | 🔵 faer-fork 1.76× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 13.9572 ± 0.0478382 (W 0.0168547) | 18.8762 ± 0.151438 (W 0.0151184) | - | 🔵 faer-fork 1.35× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.87037 ± 0.00895203 (W 0.0106267) | 1.17172 ± 0.00479856 (W 0.0109411) | 0.789728 ± 0.00385487 (W 0.003127) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔴 faer-fork 1.10× slower than `lapack-mkl` |  |
| `lu/n255/b1 [api-inclusive]` | 1.56096 ± 0.00893673 (W 0.00810438) | 1.37118 ± 0.0143857 (W 0.0133526) | - | 🔴 faer-fork 1.14× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.905338 ± 0.00315601 (W 0.00350955) | 1.16629 ± 0.00879788 (W 0.0234519) | 0.830755 ± 0.00413667 (W 0.00313823) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔴 faer-fork 1.09× slower than `lapack-mkl` |  |
| `lu/n257/b1 [api-inclusive]` | 1.5744 ± 0.0105578 (W 0.00579002) | 1.28728 ± 0.0212847 (W 0.0172314) | - | 🔴 faer-fork 1.22× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 4.58066 ± 0.0533734 (W 0.0525587) | 5.60846 ± 0.0559285 (W 0.0676403) | 4.8328 ± 0.0332002 (W 0.00987402) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.06× faster than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 8.27827 ± 0.0532907 (W 0.0449856) | 6.50729 ± 0.0598789 (W 0.0799088) | - | 🔴 faer-fork 1.27× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 4.77745 ± 0.00861144 (W 0.00689973) | 5.75438 ± 0.0894844 (W 0.147275) | 4.94755 ± 0.00986203 (W 0.00774654) | 🔵 faer-fork 1.20× faster than `faer-upstream` | · |  |
| `lu/n513/b1 [api-inclusive]` | 8.38808 ± 0.0385729 (W 0.0208354) | 6.38608 ± 0.0431103 (W 0.0737062) | - | 🔴 faer-fork 1.31× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.96264 ± 0.0212732 (W 0.00535949) | 4.00299 ± 0.00655237 (W 0.00638773) | 4.01741 ± 0.00258363 (W 0.00715739) | 🔵 faer-fork 1.35× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 4.03313 ± 0.0248697 (W 0.00833022) | 5.01521 ± 0.0203665 (W 0.0187721) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n257/b1` | 3.04414 ± 0.017122 (W 0.00506988) | 4.09081 ± 0.00437967 (W 0.00776089) | 4.11472 ± 0.00529192 (W 0.00529885) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 4.13818 ± 0.0123447 (W 0.00856952) | 5.18019 ± 0.0370869 (W 0.0146615) | - | 🔵 faer-fork 1.25× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n511/b1` | 19.0075 ± 0.0546431 (W 0.0308687) | 20.9295 ± 0.0428978 (W 0.0236945) | 25.5031 ± 0.0909917 (W 0.0422343) | 🔵 faer-fork 1.10× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 24.3192 ± 0.10941 (W 0.0331247) | 26.2914 ± 0.102519 (W 0.0355119) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n513/b1` | 19.3475 ± 0.0552588 (W 0.0301622) | 21.4111 ± 0.0198601 (W 0.0157183) | 25.8398 ± 0.0848729 (W 0.0671072) | 🔵 faer-fork 1.11× faster than `faer-upstream` | 🔵 faer-fork 1.34× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 24.9435 ± 0.146882 (W 0.0257001) | 27.0237 ± 0.0137498 (W 0.0291186) | - | 🔵 faer-fork 1.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `qr_factor/n255/b1` | 1.5202 ± 0.00641907 (W 0.00305792) | 2.59726 ± 0.0223269 (W 0.00463244) | 1.54526 ± 0.00220949 (W 0.00270837) | 🔵 faer-fork 1.71× faster than `faer-upstream` | · |  |
| `qr_factor/n257/b1` | 1.56226 ± 0.00764559 (W 0.00394014) | 2.63584 ± 0.0111169 (W 0.00318808) | 1.60234 ± 0.0031203 (W 0.00297311) | 🔵 faer-fork 1.69× faster than `faer-upstream` | · |  |
| `qr_factor/n511/b1` | 8.82175 ± 0.0248662 (W 0.0136023) | 10.8698 ± 0.0369994 (W 0.0120696) | 10.6568 ± 0.0187467 (W 0.0133697) | 🔵 faer-fork 1.23× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 8.97005 ± 0.0360745 (W 0.00931615) | 11.0931 ± 0.0252641 (W 0.0143545) | 10.8192 ± 0.0189871 (W 0.010516) | 🔵 faer-fork 1.24× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `solve/n255/b1` | 0.870389 ± 0.0109945 (W 0.0114415) | 1.15112 ± 0.0167346 (W 0.0103623) | 0.755274 ± 0.00388571 (W 0.00370755) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔴 faer-fork 1.15× slower than `lapack-mkl` |  |
| `solve/n255/b1 [api-inclusive]` | 1.5863 ± 0.00947234 (W 0.017111) | 1.27935 ± 0.00883009 (W 0.0147281) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.909766 ± 0.00621142 (W 0.00372357) | 1.17148 ± 0.0100238 (W 0.02198) | 0.794474 ± 0.00112043 (W 0.00430467) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔴 faer-fork 1.15× slower than `lapack-mkl` |  |
| `solve/n257/b1 [api-inclusive]` | 1.60653 ± 0.00754576 (W 0.00364849) | 1.27787 ± 0.00646209 (W 0.0163285) | - | 🔴 faer-fork 1.26× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 4.57436 ± 0.0294968 (W 0.0409118) | 5.7209 ± 0.00741678 (W 0.0890831) | 4.81112 ± 0.0383864 (W 0.114042) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `solve/n511/b1 [api-inclusive]` | 8.3598 ± 0.0328216 (W 0.0489462) | 6.51172 ± 0.118579 (W 0.0507188) | - | 🔴 faer-fork 1.28× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 4.80588 ± 0.0239226 (W 0.00898823) | 5.78208 ± 0.115491 (W 0.0801529) | 4.86122 ± 0.0163444 (W 0.00507398) | 🔵 faer-fork 1.20× faster than `faer-upstream` | · |  |
| `solve/n513/b1 [api-inclusive]` | 8.37171 ± 0.0677657 (W 0.0192402) | 6.25393 ± 0.0895985 (W 0.0752561) | - | 🔴 faer-fork 1.34× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 12.8165 ± 0.0149795 (W 0.0242159) | 21.7049 ± 0.0164753 (W 0.031059) | 15.5502 ± 0.0233045 (W 0.0238176) | 🔵 faer-fork 1.69× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 12.875 ± 0.0303797 (W 0.017758) | 21.7151 ± 0.0265063 (W 0.0467515) | - | 🔵 faer-fork 1.69× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 13.1604 ± 0.0166448 (W 0.0145939) | 17.189 ± 0.0679358 (W 0.0148461) | 15.9582 ± 0.06482 (W 0.0165708) | 🔵 faer-fork 1.31× faster than `faer-upstream` | 🔵 faer-fork 1.21× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 13.1787 ± 0.0493051 (W 0.0265696) | 17.3772 ± 0.140528 (W 0.0486843) | - | 🔵 faer-fork 1.32× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 68.7851 ± 0.154634 (W 0.0552423) | 85.8774 ± 0.10536 (W 0.0691835) | 82.123 ± 0.377756 (W 0.296425) | 🔵 faer-fork 1.25× faster than `faer-upstream` | 🔵 faer-fork 1.19× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 68.9713 ± 0.172876 (W 0.0471647) | 88.1174 ± 0.14857 (W 1.08848) | - | 🔵 faer-fork 1.28× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 70.4078 ± 0.0649296 (W 0.163665) | 74.969 ± 0.0736024 (W 0.0530606) | 83.0725 ± 0.129366 (W 0.315155) | 🔵 faer-fork 1.06× faster than `faer-upstream` | 🔵 faer-fork 1.18× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 70.43 ± 0.136874 (W 0.0864434) | 77.0779 ± 0.159674 (W 0.966244) | - | 🔵 faer-fork 1.09× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 6.41389 ± 0.0146725 (W 0.0151544) | 16.4431 ± 0.100952 (W 0.0224314) | 7.47362 ± 0.00580966 (W 0.0109006) | 🔵 faer-fork 2.56× faster than `faer-upstream` | 🔵 faer-fork 1.17× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 6.4225 ± 0.0224972 (W 0.0123033) | 16.4029 ± 0.0652263 (W 0.072566) | - | 🔵 faer-fork 2.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 6.58327 ± 0.0363109 (W 0.0131595) | 12.8694 ± 0.0231876 (W 0.018089) | 7.64009 ± 0.0128066 (W 0.0123532) | 🔵 faer-fork 1.95× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 6.58629 ± 0.0159042 (W 0.0105964) | 12.896 ± 0.17739 (W 0.0198129) | - | 🔵 faer-fork 1.96× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 34.1773 ± 0.118804 (W 0.0440299) | 56.8314 ± 0.240568 (W 0.116319) | 39.7194 ± 0.100314 (W 0.0481171) | 🔵 faer-fork 1.66× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `svd_values/n511/b1 [api-inclusive]` | 34.2466 ± 0.0955606 (W 0.0639241) | 56.8004 ± 0.108085 (W 0.0864123) | - | 🔵 faer-fork 1.66× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 34.6348 ± 0.0269293 (W 0.0451125) | 47.4123 ± 0.0893246 (W 0.0934304) | 40.1037 ± 0.0334609 (W 0.0429514) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔵 faer-fork 1.16× faster than `lapack-mkl` |  |
| `svd_values/n513/b1 [api-inclusive]` | 34.6873 ± 0.07435 (W 0.0225003) | 47.3785 ± 0.0660804 (W 0.0867175) | - | 🔵 faer-fork 1.37× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.023043 ± 0.000389462 (W 0.000247379) | 0.023825 ± 0.00564158 (W 8.77741e-05) | - | · | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `triangular_solve/n257/b1` | 0.023394 ± 0.000186039 (W 0.000255154) | 0.024005 ± 0.000177764 (W 6.66948e-05) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.062978 ± 0.000401981 (W 0.000360175) | 0.089436 ± 0.00085741 (W 0.000352315) | - | 🔵 faer-fork 1.42× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.062146 ± 9.59705e-05 (W 0.000323698) | 0.065853 ± 0.000191234 (W 0.00188724) | - | 🔵 faer-fork 1.06× faster than `faer-upstream` | - | `*` one row has no measurement here |

## linalg-boundary, c64, f order, 8T (CPU 4-11)

| case | faer-fork (ms) | faer-upstream (ms) | lapack-mkl (ms) | vs `faer-upstream` | vs `lapack-mkl` | notes |
|---|---|---|---|---|---|---|
| `cholesky/n255/b1` | 0.425433 ± 0.00186762 (W 0.00213377) | 0.613606 ± 0.00591535 (W 0.0165777) | 0.300421 ± 0.00388505 (W 0.00376272) | 🔵 faer-fork 1.44× faster than `faer-upstream` | 🔴 faer-fork 1.42× slower than `lapack-mkl` |  |
| `cholesky/n255/b1 [api-inclusive]` | 0.510754 ± 0.00101513 (W 0.00322667) | 0.717881 ± 0.00385805 (W 0.00441197) | - | 🔵 faer-fork 1.41× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n257/b1` | 0.381099 ± 0.0014336 (W 0.00193269) | 0.577048 ± 0.00948427 (W 0.0165029) | 0.307444 ± 0.00621436 (W 0.00891929) | 🔵 faer-fork 1.51× faster than `faer-upstream` | 🔴 faer-fork 1.24× slower than `lapack-mkl` |  |
| `cholesky/n257/b1 [api-inclusive]` | 0.459256 ± 0.0031791 (W 0.00173258) | 0.656591 ± 0.00582484 (W 0.013461) | - | 🔵 faer-fork 1.43× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n511/b1` | 1.55852 ± 0.00715777 (W 0.00512534) | 2.00658 ± 0.0035565 (W 0.0508387) | 1.344 ± 0.00469782 (W 0.00927815) | 🔵 faer-fork 1.29× faster than `faer-upstream` | 🔴 faer-fork 1.16× slower than `lapack-mkl` |  |
| `cholesky/n511/b1 [api-inclusive]` | 1.92698 ± 0.0072443 (W 0.010107) | 2.4272 ± 0.00729302 (W 0.0522284) | - | 🔵 faer-fork 1.26× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `cholesky/n513/b1` | 1.48832 ± 0.00445748 (W 0.00305932) | 1.9634 ± 0.00847671 (W 0.0590693) | 1.3939 ± 0.00879357 (W 0.012429) | 🔵 faer-fork 1.32× faster than `faer-upstream` | 🔴 faer-fork 1.07× slower than `lapack-mkl` |  |
| `cholesky/n513/b1 [api-inclusive]` | 1.83626 ± 0.00829244 (W 0.00473821) | 2.28599 ± 0.00891094 (W 0.0148721) | - | 🔵 faer-fork 1.24× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n255/b1` | 5.5527 ± 0.0255872 (W 0.0133109) | 11.5218 ± 0.039958 (W 0.0404559) | 7.50942 ± 0.0124402 (W 0.0194673) | 🔵 faer-fork 2.07× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `eigh/n255/b1 [api-inclusive]` | 5.5539 ± 0.0383641 (W 0.0141802) | 11.5701 ± 0.0433515 (W 0.0342501) | - | 🔵 faer-fork 2.08× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n257/b1` | 5.51435 ± 0.0164802 (W 0.00613468) | 11.4252 ± 0.040353 (W 0.0623603) | 7.6249 ± 0.0285253 (W 0.011767) | 🔵 faer-fork 2.07× faster than `faer-upstream` | 🔵 faer-fork 1.38× faster than `lapack-mkl` |  |
| `eigh/n257/b1 [api-inclusive]` | 5.52837 ± 0.0236562 (W 0.00920378) | 11.4125 ± 0.0410212 (W 0.0351603) | - | 🔵 faer-fork 2.06× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n511/b1` | 25.286 ± 0.0724284 (W 0.0331515) | 39.2649 ± 0.103521 (W 0.231943) | 34.4462 ± 0.0678105 (W 0.0294601) | 🔵 faer-fork 1.55× faster than `faer-upstream` | 🔵 faer-fork 1.36× faster than `lapack-mkl` |  |
| `eigh/n511/b1 [api-inclusive]` | 25.3287 ± 0.0565269 (W 0.0215062) | 39.3566 ± 0.141745 (W 0.39442) | - | 🔵 faer-fork 1.55× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `eigh/n513/b1` | 25.7913 ± 0.0328233 (W 0.26427) | 39.6282 ± 0.0925415 (W 0.250089) | 34.7426 ± 0.0578905 (W 0.0291004) | 🔵 faer-fork 1.54× faster than `faer-upstream` | 🔵 faer-fork 1.35× faster than `lapack-mkl` |  |
| `eigh/n513/b1 [api-inclusive]` | 25.8895 ± 0.0366486 (W 0.0707724) | 39.6804 ± 0.108976 (W 0.494285) | - | 🔵 faer-fork 1.53× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1` | 1.51771 ± 0.00743102 (W 0.00403499) | 4.32085 ± 0.0127253 (W 0.00635675) | - | 🔵 faer-fork 2.85× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n255/b1 [api-inclusive]` | 3.55806 ± 0.0232922 (W 0.0104125) | 5.7065 ± 0.0408256 (W 0.0136662) | - | 🔵 faer-fork 1.60× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1` | 1.55584 ± 0.00508607 (W 0.00444243) | 4.44943 ± 0.0117022 (W 0.00612672) | - | 🔵 faer-fork 2.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n257/b1 [api-inclusive]` | 2.20205 ± 0.0171922 (W 0.00848606) | 4.39094 ± 0.0210067 (W 0.00678615) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n511/b1` | 7.07624 ± 0.684648 (W 0.0127127) | 16.6669 ± 0.0554762 (W 0.0267432) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `full_piv_lu/n511/b1 [api-inclusive]` | 16.1909 ± 0.0365817 (W 0.0459685) | 22.2723 ± 0.0715581 (W 0.0206663) | - | 🔵 faer-fork 1.38× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1` | 7.11018 ± 0.0229658 (W 0.0204199) | 16.7809 ± 0.0741156 (W 0.0265473) | - | 🔵 faer-fork 2.36× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `full_piv_lu/n513/b1 [api-inclusive]` | 10.3733 ± 0.0217717 (W 0.0178596) | 16.4246 ± 0.053515 (W 0.0310077) | - | 🔵 faer-fork 1.58× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n255/b1` | 0.70272 ± 0.00364445 (W 0.00309717) | 1.06832 ± 0.0097191 (W 0.0150425) | 0.56435 ± 0.000806741 (W 0.00402801) | 🔵 faer-fork 1.52× faster than `faer-upstream` | 🔴 faer-fork 1.25× slower than `lapack-mkl` |  |
| `lu/n255/b1 [api-inclusive]` | 1.38721 ± 0.0029665 (W 0.0044057) | 1.28534 ± 0.00151914 (W 0.013226) | - | 🔴 faer-fork 1.08× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n257/b1` | 0.721718 ± 0.00696545 (W 0.00334947) | 1.03389 ± 0.00727543 (W 0.0165981) | 0.574663 ± 0.0017887 (W 0.00432839) | 🔵 faer-fork 1.43× faster than `faer-upstream` | 🔴 faer-fork 1.26× slower than `lapack-mkl` |  |
| `lu/n257/b1 [api-inclusive]` | 1.40297 ± 0.00915708 (W 0.00514571) | 1.12921 ± 0.00544497 (W 0.0186068) | - | 🔴 faer-fork 1.24× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n511/b1` | 3.09339 ± 0.00203188 (W 0.00437547) | 4.24884 ± 0.0208386 (W 0.0257127) | 2.7254 ± 0.0352293 (W 0.00650277) | 🔵 faer-fork 1.37× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `lu/n511/b1 [api-inclusive]` | 6.70662 ± 0.0482881 (W 0.0251725) | 5.02532 ± 0.0103758 (W 0.0306112) | - | 🔴 faer-fork 1.33× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `lu/n513/b1` | 3.16655 ± 0.00157674 (W 0.00493241) | 4.23846 ± 0.0199422 (W 0.0246265) | 2.77107 ± 0.0122878 (W 0.00742061) | 🔵 faer-fork 1.34× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `lu/n513/b1 [api-inclusive]` | 6.73746 ± 0.0433479 (W 0.0152496) | 4.75885 ± 0.0326612 (W 0.0385762) | - | 🔴 faer-fork 1.42× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `qr/n255/b1` | 2.03348 ± 0.00241416 (W 0.00228113) | 3.97201 ± 0.0333567 (W 0.0392903) | 2.89014 ± 0.00310063 (W 0.00816574) | 🔵 faer-fork 1.95× faster than `faer-upstream` | 🔵 faer-fork 1.42× faster than `lapack-mkl` |  |
| `qr/n255/b1 [api-inclusive]` | 3.10069 ± 0.0149314 (W 0.00853575) | 4.14753 ± 0.330618 (W 0.14604) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n257/b1` | 2.08226 ± 0.00407073 (W 0.00218065) | 4.02617 ± 0.009401 (W 0.0377305) | 2.99101 ± 0.00428002 (W 0.00745307) | 🔵 faer-fork 1.93× faster than `faer-upstream` | 🔵 faer-fork 1.44× faster than `lapack-mkl` |  |
| `qr/n257/b1 [api-inclusive]` | 3.19371 ± 0.0839533 (W 0.00913228) | 4.27069 ± 0.415104 (W 0.195437) | - | 🔵 faer-fork 1.34× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n511/b1` | 10.0464 ± 0.00924057 (W 0.00792794) | 17.5066 ± 0.178957 (W 0.178694) | 13.1762 ± 0.0389625 (W 0.0158305) | 🔵 faer-fork 1.74× faster than `faer-upstream` | 🔵 faer-fork 1.31× faster than `lapack-mkl` |  |
| `qr/n511/b1 [api-inclusive]` | 15.5495 ± 0.0739785 (W 0.0366571) | 22.4396 ± 2.80026 (W 0.227724) | - | 🔵 faer-fork 1.44× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr/n513/b1` | 10.7802 ± 0.0145782 (W 0.0222486) | 17.4383 ± 0.0837359 (W 0.229045) | 13.4843 ± 0.0649173 (W 0.0101958) | 🔵 faer-fork 1.62× faster than `faer-upstream` | 🔵 faer-fork 1.25× faster than `lapack-mkl` |  |
| `qr/n513/b1 [api-inclusive]` | 16.3902 ± 0.0314812 (W 0.0797603) | 21.5438 ± 2.16225 (W 1.46576) | - | 🔵 faer-fork 1.31× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |
| `qr_factor/n255/b1` | 1.25229 ± 0.00236796 (W 0.00401641) | 2.73526 ± 0.0182951 (W 0.010893) | 1.57598 ± 0.00412245 (W 0.00254599) | 🔵 faer-fork 2.18× faster than `faer-upstream` | 🔵 faer-fork 1.26× faster than `lapack-mkl` |  |
| `qr_factor/n257/b1` | 1.27998 ± 0.00205655 (W 0.00487933) | 2.76973 ± 0.01325 (W 0.0218589) | 1.61953 ± 0.00711556 (W 0.00439177) | 🔵 faer-fork 2.16× faster than `faer-upstream` | 🔵 faer-fork 1.27× faster than `lapack-mkl` |  |
| `qr_factor/n511/b1` | 4.72191 ± 0.0148087 (W 0.00783092) | 9.84583 ± 0.0518624 (W 0.128726) | 5.41101 ± 0.0140324 (W 0.0061965) | 🔵 faer-fork 2.09× faster than `faer-upstream` | 🔵 faer-fork 1.15× faster than `lapack-mkl` |  |
| `qr_factor/n513/b1` | 5.31902 ± 0.104323 (W 0.0122876) | 10.0121 ± 0.0341697 (W 0.122825) | 5.53686 ± 0.00743406 (W 0.0112408) | 🔵 faer-fork 1.88× faster than `faer-upstream` | · |  |
| `solve/n255/b1` | 0.707541 ± 0.00246664 (W 0.00238349) | 1.05559 ± 0.00697485 (W 0.0111964) | 0.555033 ± 0.00098151 (W 0.0031433) | 🔵 faer-fork 1.49× faster than `faer-upstream` | 🔴 faer-fork 1.27× slower than `lapack-mkl` |  |
| `solve/n255/b1 [api-inclusive]` | 1.40358 ± 0.00737142 (W 0.00464292) | 1.16294 ± 0.0189308 (W 0.00933021) | - | 🔴 faer-fork 1.21× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n257/b1` | 0.729199 ± 0.00321976 (W 0.00319308) | 1.03361 ± 0.00847148 (W 0.0160629) | 0.57251 ± 0.00607424 (W 0.00381707) | 🔵 faer-fork 1.42× faster than `faer-upstream` | 🔴 faer-fork 1.27× slower than `lapack-mkl` |  |
| `solve/n257/b1 [api-inclusive]` | 1.4061 ± 0.0209222 (W 0.00471269) | 1.12756 ± 0.0201936 (W 0.0233306) | - | 🔴 faer-fork 1.25× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n511/b1` | 3.09391 ± 0.0065833 (W 0.00482082) | 4.25976 ± 0.0116058 (W 0.022463) | 2.708 ± 0.0128373 (W 0.00818377) | 🔵 faer-fork 1.38× faster than `faer-upstream` | 🔴 faer-fork 1.14× slower than `lapack-mkl` |  |
| `solve/n511/b1 [api-inclusive]` | 6.67318 ± 0.0367514 (W 0.00985607) | 4.90769 ± 0.029781 (W 0.0224151) | - | 🔴 faer-fork 1.36× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `solve/n513/b1` | 3.16602 ± 0.0121219 (W 0.00348006) | 4.19969 ± 0.0197953 (W 0.0250136) | 2.75647 ± 0.0121632 (W 0.00916279) | 🔵 faer-fork 1.33× faster than `faer-upstream` | 🔴 faer-fork 1.15× slower than `lapack-mkl` |  |
| `solve/n513/b1 [api-inclusive]` | 6.78086 ± 0.0929621 (W 0.0115359) | 4.77716 ± 0.00962013 (W 0.055476) | - | 🔴 faer-fork 1.42× slower than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n255/b1` | 10.6254 ± 0.0474797 (W 0.00912357) | 21.3512 ± 0.0818679 (W 0.0634817) | 11.9549 ± 0.0373725 (W 0.0177953) | 🔵 faer-fork 2.01× faster than `faer-upstream` | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |
| `svd_thin/n255/b1 [api-inclusive]` | 10.6925 ± 0.0332797 (W 0.0136209) | 21.2484 ± 0.117344 (W 0.153219) | - | 🔵 faer-fork 1.99× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n257/b1` | 10.8882 ± 0.0323444 (W 0.0131966) | 16.6686 ± 0.0346659 (W 0.0600185) | 12.2498 ± 0.0248709 (W 0.0156793) | 🔵 faer-fork 1.53× faster than `faer-upstream` | 🔵 faer-fork 1.13× faster than `lapack-mkl` |  |
| `svd_thin/n257/b1 [api-inclusive]` | 10.8947 ± 0.0540643 (W 0.0297186) | 17.0008 ± 0.123727 (W 0.107571) | - | 🔵 faer-fork 1.56× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n511/b1` | 49.3005 ± 0.115735 (W 0.0948051) | 73.1994 ± 0.117727 (W 0.230084) | 54.7873 ± 0.0876938 (W 0.42121) | 🔵 faer-fork 1.48× faster than `faer-upstream` | 🔵 faer-fork 1.11× faster than `lapack-mkl` |  |
| `svd_thin/n511/b1 [api-inclusive]` | 49.5575 ± 0.0832086 (W 0.032628) | 73.7053 ± 0.321761 (W 0.447787) | - | 🔵 faer-fork 1.49× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_thin/n513/b1` | 50.3228 ± 0.115535 (W 0.0915879) | 61.3739 ± 0.168765 (W 0.227517) | 55.3513 ± 0.0287159 (W 0.298665) | 🔵 faer-fork 1.22× faster than `faer-upstream` | 🔵 faer-fork 1.10× faster than `lapack-mkl` |  |
| `svd_thin/n513/b1 [api-inclusive]` | 50.4863 ± 0.163398 (W 0.0244639) | 62.0847 ± 0.082648 (W 0.520946) | - | 🔵 faer-fork 1.23× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n255/b1` | 5.77813 ± 0.0440141 (W 0.00987327) | 16.5148 ± 0.0927007 (W 0.0287945) | 6.08652 ± 0.00459271 (W 0.0108756) | 🔵 faer-fork 2.86× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `svd_values/n255/b1 [api-inclusive]` | 5.75755 ± 0.0459417 (W 0.00963055) | 16.4699 ± 0.158569 (W 0.0231563) | - | 🔵 faer-fork 2.86× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n257/b1` | 5.91817 ± 0.0228759 (W 0.00544751) | 13.1346 ± 0.270796 (W 0.0347102) | 6.23723 ± 0.0123767 (W 0.0153652) | 🔵 faer-fork 2.22× faster than `faer-upstream` | 🔵 faer-fork 1.05× faster than `lapack-mkl` |  |
| `svd_values/n257/b1 [api-inclusive]` | 5.92631 ± 0.0058514 (W 0.00546098) | 13.167 ± 0.172664 (W 0.0195349) | - | 🔵 faer-fork 2.22× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n511/b1` | 26.3903 ± 0.06565 (W 0.0667455) | 50.2587 ± 0.0320156 (W 0.105065) | 27.4084 ± 0.0587123 (W 0.039485) | 🔵 faer-fork 1.90× faster than `faer-upstream` | · |  |
| `svd_values/n511/b1 [api-inclusive]` | 26.4253 ± 0.0957773 (W 0.0331842) | 50.3472 ± 0.0329668 (W 0.031906) | - | 🔵 faer-fork 1.91× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `svd_values/n513/b1` | 26.7483 ± 0.0581968 (W 0.101865) | 40.7651 ± 0.0466898 (W 0.102984) | 27.7554 ± 0.0642707 (W 0.0291405) | 🔵 faer-fork 1.52× faster than `faer-upstream` | · |  |
| `svd_values/n513/b1 [api-inclusive]` | 26.8127 ± 0.0235294 (W 0.0287404) | 40.8188 ± 0.217597 (W 0.0556073) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n255/b1` | 0.023003 ± 0.000313784 (W 0.000113751) | 0.023443 ± 9.50281e-05 (W 0.000134299) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n257/b1` | 0.023364 ± 6.70099e-05 (W 0.000192876) | 0.023985 ± 0.00020656 (W 0.000181948) | - | · | - | `*` one row has no measurement here |
| `triangular_solve/n511/b1` | 0.058418 ± 0.000527898 (W 0.00058026) | 0.088615 ± 0.000626831 (W 0.00285732) | - | 🔵 faer-fork 1.52× faster than `faer-upstream` | - | `*` one row has no measurement here |
| `triangular_solve/n513/b1` | 0.05895 ± 0.00166994 (W 0.00171466) | 0.063337 ± 0.000599017 (W 0.000382763) | - | 🔵 faer-fork 1.07× faster than `faer-upstream` | - | `*` one row has no measurement here; `!` between-start range over 5% |

Not applicable to this population's shapes and reported as skipped: `cholesky/255`, `cholesky/257`, `cholesky/511`, `cholesky/513`, `eigh/255`, `eigh/257`, `eigh/511`, `eigh/513`, `full_piv_lu/255`, `full_piv_lu/257`, `full_piv_lu/511`, `full_piv_lu/513`, `lu/255`, `lu/257`, `lu/511`, `lu/513`, `qr/255`, `qr/257`, `qr/511`, `qr/513`, `qr_factor/255`, `qr_factor/257`, `qr_factor/511`, `qr_factor/513`, `solve/255`, `solve/257`, `solve/511`, `solve/513`, `svd_thin/255`, `svd_thin/257`, `svd_thin/511`, `svd_thin/513`, `svd_values/255`, `svd_values/257`, `svd_values/511`, `svd_values/513`, `triangular_solve/255`, `triangular_solve/257`, `triangular_solve/511`, `triangular_solve/513`.

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
