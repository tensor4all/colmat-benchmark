# AOCL arm smoke cell, zen5-cpu, 2026-11-19 (#95)

Gate for adding AOCL to the campaign descriptor: the AOCL arm must build, load, keep its thread budget
and pass `verify` before any campaign or page names it. Not a publication and not campaign evidence:
one process set, no A/A repeat, no drift control. The full refresh measures the arm properly.

## What was measured

- host `evo` (AMD Ryzen AI 9 HX 470, profile `zen5-cpu`), boost 0, governor `performance`;
  observers (agents, kache, herdr) pinned to CPUs 0-3,12-15; measured CPUs `4` / `4-7` / `4-11` for 1/4/8T
- harness: branch `feat/aocl-arm-95` (PR #96), built by `scripts/build_for_faer_rev.sh` once per arm
- fork `93da62172a96260ae3d5ed7481690f440f58d904` (colmat 0.34.0), upstream `0539947ffb75` (`faer-v0.24.4`)
- AOCL 6.0.0 (`aocl-linux-gcc-6.0.0.tar.gz`, sha256
  `208540325a62043057decb8047bacd85015ad3c4e9f829e9de410148711936de`, from
  `download.amd.com/developer/eula/aocl/aocl-6-0/`), installed without root by
  `install.sh -t $HOME/opt/AMD/aocl -l blis libflame utils -i lp64`, tree
  `/home/shinaoka/opt/AMD/aocl/6.0.0/gcc/MT`; `AOCL-BLAS 6.0.0 Build 20261026`,
  `AOCL-LAPACK 6.0.0 Build 20261026`; dispatched `blis=zen5, flame=avx512dq`; loaded via the binary's
  `RPATH` with no `LD_LIBRARY_PATH` (`ldd` and `/proc/self/maps` both show `lib_LP64`)
- MKL 2026.2; OpenBLAS 0.3.32 (DYNAMIC_ARCH, dispatched `Cooperlake`)
- grid: f64, n = 512, `lu`/`cholesky`/`qr`/`eigh`/`svd_thin` (`linalg-mid`) and `gemm` 512^3
  (`gemm-mid`), C and F order, 1/4/8T; rows `faer-fork` + `lapack-<arm>`; `fbench verify` then
  `benchmarks/scripts/pinned.sh` with `--reps 5 --prime-ms 250`, median per cell

## Result

- `verify`: all comparisons within tolerance on every arm and thread count (18/18).
- timing: every pinned run accepted on the first attempt (no guard retry or rejection);
  198 `faer-fork` rows and 21 vendor rows per arm `ok`. The 15 C-order vendor rows of the
  LAPACK-driver families are `skipped` by design, identically on all three arms.
- thread budget: every vendor row records `vendor_threads` = `rayon_threads` = the budget (1/4/8).
  `fbench info --threads N` reads `omp:N blis:N flame:N` for N = 1, 4, 8.
  `BLIS_NUM_THREADS=2` (or `FLA_NUM_THREADS=2`) under a 4-thread budget is refused with exit 2 by
  `info`, `verify` and `run` before anything is timed:
  `aocl reports omp:4 blis:2 flame:4, but this row declares 4 threads`.
- the fork row is measured in each arm's own binary and agrees across the three arms (within about 3%),
  so the arms are comparable on this host.

Times in ms (median of 5); ratio = vendor / fork of the same arm (above 1 means the fork is faster).

| family | order | T | fork (AOCL arm) | AOCL | AOCL/fork | fork (MKL arm) | MKL | MKL/fork | fork (OpenBLAS arm) | OpenBLAS | OpenBLAS/fork |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cholesky | c | 1 | 2.771 | - | - | 2.742 | - | - | 2.894 | - | - |
| cholesky | c | 4 | 1.010 | - | - | 0.976 | - | - | 0.959 | - | - |
| cholesky | c | 8 | 0.794 | - | - | 0.798 | - | - | 0.789 | - | - |
| cholesky | f | 1 | 1.953 | 1.759 | 0.90 | 1.926 | 3.125 | 1.62 | 1.980 | 2.538 | 1.28 |
| cholesky | f | 4 | 0.684 | 0.558 | 0.82 | 0.654 | 1.252 | 1.91 | 0.650 | 1.135 | 1.75 |
| cholesky | f | 8 | 0.518 | 0.379 | 0.73 | 0.517 | 0.884 | 1.71 | 0.512 | 1.003 | 1.96 |
| eigh | c | 1 | 51.298 | - | - | 50.630 | - | - | 50.904 | - | - |
| eigh | c | 4 | 24.929 | - | - | 24.932 | - | - | 24.849 | - | - |
| eigh | c | 8 | 16.481 | - | - | 16.485 | - | - | 16.439 | - | - |
| eigh | f | 1 | 51.169 | 47.738 | 0.93 | 50.533 | 59.969 | 1.19 | 50.964 | 49.239 | 0.97 |
| eigh | f | 4 | 24.590 | 27.618 | 1.12 | 24.768 | 20.927 | 0.84 | 24.674 | 25.896 | 1.05 |
| eigh | f | 8 | 16.297 | 24.925 | 1.53 | 16.315 | 14.321 | 0.88 | 16.303 | 22.386 | 1.37 |
| gemm | c | 1 | 8.587 | 8.747 | 1.02 | 8.607 | 9.730 | 1.13 | 8.581 | 10.070 | 1.17 |
| gemm | c | 4 | 2.200 | 2.188 | 0.99 | 2.186 | 3.004 | 1.37 | 2.203 | 2.607 | 1.18 |
| gemm | c | 8 | 1.134 | 1.139 | 1.00 | 1.123 | 1.527 | 1.36 | 1.130 | 1.308 | 1.16 |
| gemm | f | 1 | 8.608 | 8.702 | 1.01 | 8.615 | 9.880 | 1.15 | 8.546 | 10.027 | 1.17 |
| gemm | f | 4 | 2.197 | 2.194 | 1.00 | 2.188 | 3.007 | 1.37 | 2.198 | 2.570 | 1.17 |
| gemm | f | 8 | 1.132 | 1.136 | 1.00 | 1.128 | 1.520 | 1.35 | 1.125 | 1.309 | 1.16 |
| lu | c | 1 | 3.873 | - | - | 3.862 | - | - | 3.889 | - | - |
| lu | c | 4 | 1.827 | - | - | 1.753 | - | - | 1.812 | - | - |
| lu | c | 8 | 1.286 | - | - | 1.296 | - | - | 1.344 | - | - |
| lu | f | 1 | 3.726 | 9.426 | 2.53 | 3.699 | 5.043 | 1.36 | 3.752 | 4.954 | 1.32 |
| lu | f | 4 | 1.560 | 3.217 | 2.06 | 1.572 | 1.489 | 0.95 | 1.567 | 2.390 | 1.52 |
| lu | f | 8 | 1.198 | 2.862 | 2.39 | 1.206 | 1.108 | 0.92 | 1.203 | 1.510 | 1.26 |
| qr | c | 1 | 18.753 | - | - | 18.670 | - | - | 18.497 | - | - |
| qr | c | 4 | 5.985 | - | - | 5.956 | - | - | 5.949 | - | - |
| qr | c | 8 | 3.452 | - | - | 3.419 | - | - | 3.351 | - | - |
| qr | f | 1 | 17.540 | 34.843 | 1.99 | 17.740 | 34.998 | 1.97 | 17.506 | 31.551 | 1.80 |
| qr | f | 4 | 5.548 | 10.350 | 1.87 | 5.544 | 9.246 | 1.67 | 5.573 | 18.511 | 3.32 |
| qr | f | 8 | 3.201 | 6.449 | 2.01 | 3.191 | 5.250 | 1.65 | 3.217 | 16.644 | 5.17 |
| svd_thin | c | 1 | 86.600 | - | - | 86.513 | - | - | 86.731 | - | - |
| svd_thin | c | 4 | 36.591 | - | - | 36.698 | - | - | 36.576 | - | - |
| svd_thin | c | 8 | 30.197 | - | - | 30.367 | - | - | 30.247 | - | - |
| svd_thin | f | 1 | 85.847 | 106.603 | 1.24 | 85.882 | 114.833 | 1.34 | 85.996 | 113.936 | 1.32 |
| svd_thin | f | 4 | 35.596 | 64.579 | 1.81 | 35.825 | 56.777 | 1.58 | 35.814 | 86.645 | 2.42 |
| svd_thin | f | 8 | 29.253 | 60.552 | 2.07 | 29.440 | 48.560 | 1.65 | 29.318 | 83.693 | 2.85 |

One process set and no drift control, so these numbers only show that the arm is measurable. Two
observations for the full refresh to confirm or discard: AOCL `lu` F order is 2.0-2.5x the fork
(MKL 0.92-1.36x), and AOCL `cholesky` F order is faster than the fork (0.73-0.90x).

Raw data (CSV, guard logs, verify logs, `fbench info` of each arm, the build `--out` files and the
smoke script) is kept on evo at `~/colmat-aocl-smoke-20261119T150458Z/`.

