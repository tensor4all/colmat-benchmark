# `x86-v4` paired campaign on evo (tensor4all/colmat-rs#39)

One colmat revision measured twice on the same host, same harness and same grid, differing only in
colmat's `x86-v4` feature (AVX-512 for pulp's dispatch of colmat's own SIMD loops, and the
`gemm`/`nano-gemm` kernels). Question: may `x86-v4` become a default feature? colmat-rs#39's
acceptance: no cell regresses beyond its own spread, and at least one family wins.

**Answer: no.** Wins exist (`full_piv_lu` tall/wide up to 1.475, `householder_apply` small), but 321
of 1200 primary colmat cells regress beyond their own spread, 117 of them by more than 5% and down to
0.860 (`qr_factor` f64 96x96), while the in-run vendor control has 1 such cell in 612. `x86-v4` stays
opt-in.

## Conditions

| | |
|---|---|
| host / profile | evo, AMD Ryzen AI 9 HX 470 (Zen 5, `avx512f`), profile `zen5-cpu` |
| CPU sets | the suite's own: 1T `4`, 4T `4-7`, 8T `4-11`; recorder and every other process of ours (herdr, kache-engine, the other agent sessions) on `0-3,12-15` for the duration, restored afterwards |
| colmat | `be19a8001e37` (colmat-rs `feat/x86-v4-default-39`; differs from `main` 103d2788 only in a `faer/Cargo.toml` comment) |
| upstream faer | `faer-v0.24.4` (`0539947ffb75`) |
| harness | `a899f37cb629` (adds `FBENCH_FORK_EXTRA_FEATURES`; `fbench` depends on colmat with `default-features = false`, so this is how the feature reaches the measured build) |
| toolkit | `82bff6ea` (the `requirements.txt` pin) |
| timing | protocol v2 baseline: 250 ms priming, 5 reps, 3 process sets per arm (engine-order rotation) |
| grid | regimes linalg-small/mid/tall/wide; f64, c64; C and F; 1/4/8T |
| arms checked | the `x86-v4` binaries carry 144 `pulp::x86::v4::V4` symbols, the default ones 0; each `run.yaml` lists `x86-v4` in `library.features` for the feature arm only |

**Boost was off because the maintainer turned it off as root** before the campaign (on this host the
global `/sys/devices/system/cpu/cpufreq/boost` write works; on evo2 it fails with EINVAL and only the
per-policy `policy*/boost` does). Nothing here wrote sysfs. As read at the start of each arm (files
`boost-at-start-*.txt`): global `cpufreq/boost` = 0 and all 24 `policy*/boost` = 0, every time. The
recorder's own check reads the global file, and every manifest records `host.power.cpufreq_boost: '0'`,
governor and EPP `performance`. **A re-run with boost on is not comparable with these numbers.**

## Runs (`data/results/zen5-cpu/colmat-kernels/`)

Pair 1 is recorded in this pull request (colmat-benchmark#141); pair 2 is recorded in
colmat-benchmark#142, because one pull request carrying all four runs exceeds GitHub's 300-file diff
limit. The per-cell tables here cover both pairs.

| pair | vendor arm (in-run control) | families | default arm | `x86-v4` arm | order |
|---|---|---|---|---|---|
| 1 | OpenBLAS 0.3.32 | cholesky, lu, solve (the arm's `vendor_families`) | `20261126T030635Z-x86v4-39-default` | `20261126T035822Z-x86v4-39-v4` | default first |
| 2 | AOCL 6.0.0 | full_piv_lu, triangular_solve, qr_factor, qr, svd_values, svd_thin, eigh, eig, upgrade_factor, householder_apply | `20261126T064739Z-x86v4-39-aocl-default` | `20261126T043949Z-x86v4-39-aocl-v4` | `x86-v4` first |

`eig` and `upgrade_factor` were in the request but have no colmat/faer route in this harness: every row
of them is recorded `skipped` ("not implemented for the faer arms in this harness"), so they have no
paired cells.

## Method

`scripts/pair_feature_arms.py <default run> <x86-v4 run>`. A cell is (regime, family, dtype, order,
shape, batch, engine row, threads, variant), where the variant is the CSV's `note`: the primary row, the
`api-inclusive` owning call, or a Householder `block(...)`/`sequence(...)` form. Its value in one
process set is the recorded `total_ms` (best of the reps). Per cell:

- **ratio** = median(default) / median(`x86-v4`) over the 3 process sets (> 1: `x86-v4` faster);
- **spread** = (max - min) / median over one arm's 3 sets, reported for **both** arms;
- **regress** = `x86-v4` slower by more than max(both spreads) *and* the two arms' ranges do not overlap;
  **win** symmetrically. The `> 5%` columns add the campaign's 5% verdict floor.

Every cell with its ratio and both spreads, one row each: `openblas-arm-cells.csv` (1188 cells),
`aocl-arm-cells.csv` (3456 cells); columns `ratio`, `spread_base`, `spread_v4`, `verdict`.

## Result: colmat, primary rows

1200 cells: **321 regress beyond their own spread, 117 of them by more than 5%**; 104 win by more than 5%.

| family | cells | geomean | min | max | regress > own spread | of them > 5% | win > 5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| triangular_solve | 72 | 0.957 | 0.907 | 1.049 | 52 | 36 | 0 |
| qr_factor | 120 | 0.986 | 0.860 | 1.094 | 32 | 21 | 2 |
| full_piv_lu | 120 | 1.034 | 0.877 | 1.475 | 31 | 17 | 44 |
| qr | 120 | 0.995 | 0.891 | 1.061 | 27 | 17 | 1 |
| svd_thin | 120 | 0.988 | 0.898 | 1.042 | 54 | 12 | 0 |
| svd_values | 120 | 0.997 | 0.908 | 1.081 | 49 | 6 | 8 |
| cholesky | 72 | 1.007 | 0.886 | 1.195 | 21 | 4 | 7 |
| eigh | 72 | 1.001 | 0.945 | 1.041 | 20 | 2 | 0 |
| solve | 72 | 1.003 | 0.948 | 1.066 | 16 | 1 | 4 |
| householder_apply | 192 | 1.019 | 0.935 | 1.154 | 2 | 1 | 31 |
| lu | 120 | 1.009 | 0.958 | 1.082 | 17 | 0 | 7 |

Geomean ratio by regime (sizes: small 32/64/96, mid 128/256/512, tall 1024x64 and 4096x256, wide the
transposes):

| family | small | mid | tall | wide |
|---|---:|---:|---:|---:|
| triangular_solve | 0.944 | 0.970 | - | - |
| qr_factor | 0.955 | 0.994 | 1.007 | 1.001 |
| qr | 0.978 | 1.000 | 1.007 | 1.000 |
| svd_thin | 0.958 | 1.002 | 1.002 | 1.000 |
| svd_values | 0.975 | 1.010 | 1.002 | 1.006 |
| full_piv_lu | 0.972 | 1.029 | 1.097 | 1.076 |
| householder_apply | 1.042 | 1.008 | 1.001 | - |
| cholesky | 1.013 | 1.001 | - | - |
| eigh | 0.988 | 1.013 | - | - |
| lu | 1.013 | 1.009 | 0.994 | 1.020 |
| solve | 1.004 | 1.003 | - | - |

The ten slowest primary cells, with each arm's own spread:

| family | shape | dtype | order | threads | ratio | spread default | spread `x86-v4` |
|---|---|---|---|---|---:|---:|---:|
| qr_factor | 96x96 | f64 | F | 4T | 0.860 | 0.004 | 0.001 |
| qr_factor | 32x32 | f64 | F | 8T | 0.864 | 0.021 | 0.024 |
| qr_factor | 96x96 | f64 | F | 8T | 0.864 | 0.004 | 0.015 |
| qr_factor | 128x128 | f64 | F | 4T | 0.865 | 0.006 | 0.007 |
| qr_factor | 128x128 | f64 | F | 8T | 0.865 | 0.005 | 0.001 |
| qr_factor | 96x96 | f64 | F | 1T | 0.866 | 0.012 | 0.011 |
| qr_factor | 128x128 | f64 | F | 1T | 0.866 | 0.026 | 0.015 |
| qr_factor | 128x128 | f64 | C | 4T | 0.875 | 0.009 | 0.010 |
| full_piv_lu | 32x32 | f64 | C | 8T | 0.877 | 0.030 | 0.046 |
| qr_factor | 32x32 | f64 | F | 1T | 0.881 | 0.017 | 0.008 |

The wins: `full_piv_lu` on tall/wide shapes (up to 1.475, c64 256x4096 F 1T, spreads 0.008/0.010) and
`householder_apply` at small sizes (1.042). The losses: small sizes of `triangular_solve`, `qr_factor`,
`qr`, `svd_thin`, `svd_values`, plus `triangular_solve` at mid sizes. The `api-inclusive` rows (the
owning calls) show the same pattern (e.g. `qr` 24 cells regress by more than 5%); they are in the CSVs.

## Controls

- **Vendor rows**, which the feature cannot touch (OpenBLAS 0.3.32 for cholesky/lu/solve, AOCL 6.0.0
  for the rest): 612 cells, 27 regress beyond their own spread, **1** by more than 5%; family geomeans
  0.998-1.005. Those 27 (4.4%) are the false-positive rate of a 3-set spread test on this host; the colmat
  rate is 321/1200 (27%), and 117 against 1 beyond the 5% floor. Neither pair drifted.
- **Upstream faer rows** share pulp in one binary, so they got V4 too, and moved the same way:
  `triangular_solve` 0.948 (43 cells regress by more than 5%), wins in `cholesky` (1.101), `eigh`
  (1.068), `svd_thin` (1.047), `full_piv_lu` (1.063). The small-size `triangular_solve` loss is in both
  libraries.

## What this does not establish

- One host (Zen 5 mobile, 4T/8T sets spanning Zen 5c cores per the profile). Not measured on evo2 or an
  Intel AVX-512 host.
- Why the narrow SIMD loops lose at small sizes on V4 (remainder handling, frequency, or the 512-bit
  datapath split on this part) is not isolated here.
- 3 process sets per arm make "its own spread" a tight bar: a sub-1% difference can cross it (the
  vendor control's 4.4%). The `> 5%` columns and the control rate are given so that is visible; the
  decision does not rest on the tight bar alone (117 colmat cells regress by more than 5%, against 1 in
  the control).
