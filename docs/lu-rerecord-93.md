# LU re-record after colmat-rs #235 (colmat-benchmark#93)

Measured on evo (zen5-cpu, `cpufreq_boost=0`, governor `performance`, recorded in each `run.yaml`).
Library: `tensor4all/colmat-rs` main `a5536ad3` (contains #235 `0ad25fe1`; later commits #236/#242 do not touch LU).
Harness: this repo at `9988f5a`; toolkit `ec1366c`; upstream faer `0539947f` (linked, not measured: engines are `faer-fork,lapack-mkl`).
MKL arm: `lapack-mkl` from the same runs (MKL 2026.2). `--reps 5 --aa 6 --prime-ms 250`, 6 process starts per cell; the number shown is the median across starts of each start's median.
Family `lu`, dtype f64, both orders recorded (F first below). "primary" = the in-place `lu` row; "api-incl" = the owning `PartialPivLu::new` row (labelled `api-inclusive`, never the vendor comparison).

Runs: `data/results/zen5-cpu/faer-kernels/*-lu93-daily-mkl` (linalg-small/mid, 1/4/8T) and
`data/results/zen5-cpu/faer-kernels-large/*-lu93-large-mkl` (linalg-large + linalg-boundary, 4/8T).

## Result: LU is still behind MKL on every requested target cell (ratio > 1 = colmat slower)

| cell | colmat ms | MKL ms | colmat/MKL |
|---|---|---|---|
| 8T n=511 F | 1.1237 | 1.0048 | 1.118 |
| 8T n=513 F | 1.1649 | 1.0278 | 1.133 |
| 8T n=1024 F | 5.7607 | 5.2582 | 1.096 |
| 4T n=256 F | 0.3323 | 0.2956 | 1.124 |
| 4T n=257 F | 0.3264 | 0.2971 | 1.099 |

## Gaps and caveats

- The recorder refused `--shapes` 192/384/768 for LU ("not declared by this suite"); not worked around. Daily tier is therefore 32/64/96/128/256/512 only.
- `outputs_dirty: true` in the large run's `run.yaml`: the daily run directory was uncommitted while it ran (the pre-commit hook refused to commit during the run). No source tree was dirty.
- Observers were confined off the measured cores per the recorder's instructions (`taskset`, `observer_audit --confine`).
- The in-place primary row and MKL are the comparison; ratios are from medians of 6 starts, not tested for significance.

## Full table (all cells)

| order | T | n | colmat ms (primary) | MKL ms | colmat/MKL | colmat api-incl ms | api/MKL | starts |
|---|---|---|---|---|---|---|---|---|
| c | 1 | 32 | 0.0064 |  |  | 0.0086 |  | 6 |
| c | 1 | 64 | 0.0234 |  |  | 0.0298 |  | 6 |
| c | 1 | 96 | 0.0554 |  |  | 0.0769 |  | 6 |
| c | 1 | 128 | 0.1048 |  |  | 0.1274 |  | 6 |
| c | 1 | 256 | 0.5708 |  |  | 0.9513 |  | 6 |
| c | 1 | 512 | 3.8921 |  |  | 5.7029 |  | 6 |
| c | 4 | 32 | 0.0066 |  |  | 0.0088 |  | 6 |
| c | 4 | 64 | 0.0287 |  |  | 0.0304 |  | 6 |
| c | 4 | 96 | 0.0649 |  |  | 0.0732 |  | 6 |
| c | 4 | 128 | 0.1169 |  |  | 0.1561 |  | 6 |
| c | 4 | 255 | 0.4292 |  |  | 0.7191 |  | 6 |
| c | 4 | 256 | 0.4090 |  |  | 0.7200 |  | 6 |
| c | 4 | 257 | 0.4247 |  |  | 0.7856 |  | 6 |
| c | 4 | 511 | 1.7556 |  |  | 3.5747 |  | 6 |
| c | 4 | 512 | 1.7740 |  |  | 3.5497 |  | 6 |
| c | 4 | 513 | 1.8510 |  |  | 3.5399 |  | 6 |
| c | 4 | 1024 | 10.8391 |  |  | 19.0363 |  | 6 |
| c | 4 | 1536 | 31.4389 |  |  | 50.0547 |  | 6 |
| c | 8 | 32 | 0.0066 |  |  | 0.0087 |  | 6 |
| c | 8 | 64 | 0.0298 |  |  | 0.0306 |  | 6 |
| c | 8 | 96 | 0.0682 |  |  | 0.0755 |  | 6 |
| c | 8 | 128 | 0.1240 |  |  | 0.1150 |  | 6 |
| c | 8 | 255 | 0.4206 |  |  | 0.6881 |  | 6 |
| c | 8 | 256 | 0.3381 |  |  | 0.6887 |  | 6 |
| c | 8 | 257 | 0.3536 |  |  | 0.7471 |  | 6 |
| c | 8 | 511 | 1.2852 |  |  | 3.2097 |  | 6 |
| c | 8 | 512 | 1.3020 |  |  | 3.1984 |  | 6 |
| c | 8 | 513 | 1.3532 |  |  | 3.1787 |  | 6 |
| c | 8 | 1024 | 7.0227 |  |  | 15.8066 |  | 6 |
| c | 8 | 1536 | 19.5990 |  |  | 39.6631 |  | 6 |
| f | 1 | 32 | 0.0051 | 0.0107 | 0.478 | 0.0066 | 0.620 | 6 |
| f | 1 | 64 | 0.0206 | 0.0346 | 0.597 | 0.0236 | 0.681 | 6 |
| f | 1 | 96 | 0.0491 | 0.0766 | 0.641 | 0.0610 | 0.796 | 6 |
| f | 1 | 128 | 0.0972 | 0.1431 | 0.679 | 0.1043 | 0.728 | 6 |
| f | 1 | 256 | 0.5647 | 0.7960 | 0.709 | 0.8464 | 1.063 | 6 |
| f | 1 | 512 | 3.7225 | 5.0631 | 0.735 | 5.1385 | 1.015 | 6 |
| f | 4 | 32 | 0.0053 | 0.0106 | 0.500 | 0.0069 | 0.650 | 6 |
| f | 4 | 64 | 0.0206 | 0.0345 | 0.596 | 0.0240 | 0.695 | 6 |
| f | 4 | 96 | 0.0427 | 0.0763 | 0.560 | 0.0586 | 0.768 | 6 |
| f | 4 | 128 | 0.0783 | 0.1422 | 0.551 | 0.1313 | 0.923 | 6 |
| f | 4 | 255 | 0.3059 | 0.2876 | 1.064 | 0.6116 | 2.127 | 6 |
| f | 4 | 256 | 0.3323 | 0.2956 | 1.124 | 0.6182 | 2.091 | 6 |
| f | 4 | 257 | 0.3264 | 0.2971 | 1.099 | 0.6561 | 2.208 | 6 |
| f | 4 | 511 | 1.4822 | 1.4033 | 1.056 | 2.9603 | 2.110 | 6 |
| f | 4 | 512 | 1.5674 | 1.4907 | 1.051 | 2.9883 | 2.005 | 6 |
| f | 4 | 513 | 1.5493 | 1.4295 | 1.084 | 2.9841 | 2.088 | 6 |
| f | 4 | 1024 | 8.8024 | 8.8365 | 0.996 | 16.4041 | 1.856 | 6 |
| f | 4 | 1536 | 26.3900 | 27.8504 | 0.948 | 43.3481 | 1.556 | 6 |
| f | 8 | 32 | 0.0053 | 0.0106 | 0.500 | 0.0069 | 0.647 | 6 |
| f | 8 | 64 | 0.0213 | 0.0346 | 0.616 | 0.0241 | 0.698 | 6 |
| f | 8 | 96 | 0.0429 | 0.0768 | 0.558 | 0.0608 | 0.792 | 6 |
| f | 8 | 128 | 0.0785 | 0.1429 | 0.549 | 0.0903 | 0.632 | 6 |
| f | 8 | 255 | 0.2787 | 0.2728 | 1.021 | 0.5787 | 2.121 | 6 |
| f | 8 | 256 | 0.2915 | 0.2850 | 1.023 | 0.5822 | 2.043 | 6 |
| f | 8 | 257 | 0.2903 | 0.2826 | 1.027 | 0.6251 | 2.212 | 6 |
| f | 8 | 511 | 1.1237 | 1.0048 | 1.118 | 2.6124 | 2.600 | 6 |
| f | 8 | 512 | 1.2029 | 1.1099 | 1.084 | 2.6592 | 2.396 | 6 |
| f | 8 | 513 | 1.1649 | 1.0278 | 1.133 | 2.5984 | 2.528 | 6 |
| f | 8 | 1024 | 5.7607 | 5.2582 | 1.096 | 13.2063 | 2.512 | 6 |
| f | 8 | 1536 | 16.2644 | 16.5247 | 0.984 | 33.3674 | 2.019 | 6 |
