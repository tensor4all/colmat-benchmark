# `faer-kernels-ohtaka-fat-gemm` on `ohtaka-fat`

- colmat (fork, measured) commit: `93da62172a96260ae3d5ed7481690f440f58d904`
- colmat version: `0.34.0`
- upstream faer commit: `0539947ffb757a739d7e703a7d2fa0c792a909c1`
- upstream faer ref/version: `faer-v0.24.4` / `0.24.4`
- features: `std, rayon`
- faer rows: `faer-fork` (the provider in the measured revision; see library and upstream above)
- harness commit: `4c1237ceb2c0717c537f89c3d7ef59bb3e866e85`
- toolkit commit: `34cf62b2504fa68857c5c1a33c125d22c5620452`
- measured binary SHA-256: `98d8630aedfa98879e82eb12d7141ac57ada4716a9c0b420232a780a710c069e`
- hardware profile: `ohtaka-fat`
- host role: `measurement` — published dataset host
- observer: **unavailable** (observer unavailable: no CPU lies outside the measured L3 domains): the recorder and the harness driver ran on the measured cores, recorder affinity `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111`; the observer audit was not run, so compare these cells only with runs recorded the same way
- timestamp: `2026-11-22T03:59:19.829116Z`
- timing policy: v2, median of 3 reps, priming 100 ms (time-based, identical on every row)
- timing baseline: **not comparable with this campaign's v2 pages** (priming 100 ms < 250 ms; 3 reps < 5; the frozen v2 settings are priming 250 ms and 5 reps, docs/protocol-v2-baseline.md). Too little priming reads a few percent fast, so compare these cells only with runs recorded the same way; a verdict here also needs a ratio beyond 1.10×
- guards: idle window 3 s, max busy 5.0% including SMT siblings, 8 attempts; sampled on the measured cores the recorder shares, so the gate cannot tell the recorder's own load from foreign load
- command: `scripts/record_run.py ohtaka-fat faer-kernels-ohtaka-fat-gemm --vendor mkl --checkout /home/k0156/k015600/cb101/extern/t4a-faer --upstream /home/k0156/k015600/cb101/extern/upstream-faer --label ohtaka-fat-101-faer-kernels-ohtaka-fat-gemm --reps 3 --aa 2 --prime-ms 100`
- raw data: `data/results/ohtaka-fat/faer-kernels-ohtaka-fat-gemm/20261122T035033Z-ohtaka-fat-101-faer-kernels-ohtaka-fat-gemm/`

## What was measured

- **Workload** — Direct calls into one faer revision: square and rectangular dense matrices, C (row-major) and F (column-major) order, `f64` and `c64`. A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is a three-index operation and cannot be written as two. Sizes are the matrix dimensions, not a memory budget, so a case name says exactly what was factored. `gemm` measures `linalg::matmul::matmul` into a caller-prepared destination with `Accum::Replace`; there is no operator path, so one shape is one work.
  Source: The fork's own public API (`faer/src/linalg`): `matmul`, the QR factor and composite, `upgrade_householder_factor`, the block/sequence Householder apply, LU and full-pivot LU, Cholesky, triangular solve, solve, thin SVD, singular values and the symmetric/nonsymmetric eigensolvers. Inputs are seeded from the case, so `verify` and `run` measure the same matrices and a rerun is reproducible.
  Known blind spot: Only the faer entry points are measured. tlinalg's packing, batch loop and lane policy, tenferro's tensor construction and dtype dispatch, and the small-`m` decode-projection shapes are other repositories' rows (tlinalg-benchmark, tenferro-benchmark).
  Known blind spot: gemm only: the LU, Cholesky and compact QR factor rows of this profile are `faer-kernels-ohtaka-fat-factor / faer-kernels-ohtaka-fat-qr`, and the 8192 tier is `faer-kernels-ohtaka-fat-xlarge`. Rectangular shapes, the 255/257/511/513 boundary shapes, the Householder apply rows, SVD and the eigensolvers are declared in other suites of this campaign and are not measured here; a reader must not carry a conclusion from this page to those shapes.
  Known blind spot: F (column-major) order only, because the issue fixes it for this profile: `scripts/record_run.py` derives `FBENCH_ORDER=f` from this suite's `order: [f]` and exports it to the census and the timing pass, so the harness emits no C-order cell, and the manifest records the value as `harness_env` (printed in the header as `harness environment`). The runs recorded on 2026-11-22 (jobs 3223469/3223470/3223471) predate that: their Slurm job script exported `FBENCH_ORDER=f` by hand and is not committed, so their manifests and headers do not show it. Their CSVs carry `order=f` on every row, and their recorded `command`, run from this repository, now produces the same F-only grid of cells: the same layout, shapes and rows. That reproduces the grid, not the timings, which are evidence only for the recorded revisions on the recorded node. A C-order reading of the same sizes is the other ohtaka suite's business, not this one's.
  Known blind spot: A vendor arm records the library it linked. MKL names no dispatched kernel and no processor count, so its manifest leaves both empty rather than inferring them. C-order vendor cells do not exist here (F order only), which is also the layout where LAPACKE has a route that leaves the clock free of allocation and layout conversion.
  Known blind spot: The upstream faer row is not measured on this profile: the 2-process-set budget (`--aa 2`) needs a 2-row arm, so only the colmat fork and MKL are measured. The upstream faer revision printed in the page header is the pinned baseline of the campaign, not a measured column.
  Known blind spot: The 112T row shares its cores with the operating system: it measures every core of the node, so the operating system, Slurm and the recorder itself share the measured CPUs with the cells, and the recorder (pinned to NUMA node 3) sits inside the measured set for that row. The profile declares that row with no observer CPU (`observer: ""`); the recorder admits exactly that declared case, records `observer unavailable` in the manifest's observer block, and the page header repeats it. No profile-wide override is involved. Rows 16T, 32T and 64T pin the observing side to NUMA node 3 (CPUs 3,7,...,111), a domain none of them measures, but a recording that includes the 112T row leaves no CPU outside the measured domains of the whole invocation, so the recorder's observer audit runs for none of its rows: in such a run the node-3 split of the 16T/32T/64T rows is the recorded affinity, not an audited one. Recording 16T/32T/64T in an invocation without 112T (`--threads 16,32,64`) keeps the audit for them.
  Known blind spot: Boost is ON for this run and is not comparable with a boost-off run: the fat node exposes `intel_pstate/no_turbo` = 0, an unprivileged job cannot switch it off, and the recorder was started with `FBENCH_ALLOW_BOOST=1`, which prints the warning, keeps the raw switch in `host.power`, and records the deviation instead of refusing. A boost-off comparison needs the switch turned off by the operator and another run of this suite.
  Known blind spot: Machine state not shown elsewhere: the fat node's layout is 4 sockets of 28 cores, 4 NUMA nodes with CPUs interleaved across them by stride 4, SMT off, 39424 KiB of L3 per socket, 774 GB of memory per node (see the profile for how each row's CPUs are named). Its cpufreq exposes turbo and the governor, so the recorder records the power state from sysfs. The runs recorded on 2026-11-22 used reps 3, 2 process sets and priming 100 ms (chosen to fit the 30-minute `i1fat` queue), below the frozen v2 settings of 250 ms and 5 reps (`docs/protocol-v2-baseline.md`; 150 ms was rejected there because one cell read about 3% fast) and the 0.5 s `docs/timing-policy.md` asks for, so single cells may read a few percent fast. The campaign's `timing.baseline` therefore marks such a run as not comparable with the v2 pages: its header and its `result/INDEX.md` status say so, and a verdict on its page also needs a ratio beyond 1.10x. That margin is a judgement, not a measured bound: the bias was measured only at 150 ms (about 3% in one of four cells), never at 100 ms, so the index and header marking is what keeps these cells away from the v2 pages, and the margin only stops a few-percent ratio from being called. A run that should be compared with the other v2 pages is recorded at `--prime-ms 250 --reps 5`.
  Known blind spot: In the run recorded 2026-11-22 (`20261122T035033Z`, job 3223469) the MKL gemm row is erratic on this node from 32T up, and only there (tensor4all/colmat-benchmark#121). For 4096^3, MKL's per-process medians are 118.5/118.9 ms at 16T, 178.0/158.1 ms at 32T (calls 150.4-181.8 ms, slower than its own 16T row), 136.9/475.9 ms at 64T (calls 90.6-626.0 ms) and 91.0/45.9 ms at 112T. In the same four 4096^3 cells the faer-fork row's two process medians differ by 0.23%, 0.08%, 3.97% and 3.91% (16T/32T/64T/112T); this does not hold for faer at every shape (its 112T 2048^3 cell is 8.63 +/- 1.24 ms). The idle gate passed before and after every 64T process on the first attempt, and the harness read MKL's thread count back as the row's budget (`vendor=Some(64)`), so neither load that the gate sees nor an oversubscribed budget explains the 64T row. The cause is not identified; MKL's OpenMP placement over a CPU set that spans three NUMA nodes, and the boost-on clock, are untested candidates. In that run the 64T 4096^3 cell therefore carries no verdict: the 237.6 ms difference is smaller than MKL's +/- 239.7 ms. The 112T 4096^3 cell carries none either. Its 32T 4096^3 verdict clears its spread (63.8 ms against +/- 14.0 ms), but it describes that run's MKL at 32T on this node, not MKL's gemm capability. A later run of this suite must be read from its own table, not from this paragraph.
  Known blind spot: Harness revision: the runs recorded on 2026-11-22 were taken at harness commit `41146610cb25cf37eedcd325da22b9cd3eb6d59b`, which a later rebase of this branch left out of every branch (`build-info.json` keeps it as the original admission record, and the tag `harness/ohtaka-fat-101-41146610` keeps it fetchable). The manifests name its rebased counterpart `4c1237ceb2c0717c537f89c3d7ef59bb3e866e85` instead, whose `fbench/`, `Cargo.toml` and `Cargo.lock` are byte-identical to it (`git diff 41146610 4c1237c -- fbench/ Cargo.toml Cargo.lock` is empty). The only other difference is the `BENCH_CONTAINER` branch of `benchmarks/scripts/pinned.sh`, which these runs did not take (no guard log names a container). The measured code, and so these values, are the same at either commit.
  Known blind spot: Known limitation: Coverage is the declared grid only (see `runs`): gemm at the sizes and thread budgets of the ohtaka-fat profile.
  Known blind spot: Residual-check coverage narrowed: the verification census (`verify`) covers the population's 2048 cell at every thread budget, not its 4096 cell. The reference product is the harness's single-threaded cache-blocked loop on this harness revision, so one 4096 case costs minutes and a 4096 census would not fit the 30-minute queue. A timed process also checks the result of its last timed call at the cell's own size (the harness's `timing::measure` ends in the cell's check), so a wrong 4096 result fails its row, but that check records no residual and is not part of the census. Where this page says every timed cell matches a verification census, read it with this limit.
- **Rows** — one row per engine in every table below:
  - `faer-fork` — The measured library: the tensor4all fork (`colmat`, `[lib] name = "colmat"`) at the checkout the harness linked by path. This is the row the campaign exists for; the fork's three perf experiments and the two correctness changes land here.
    Identity: the fork revision in this run's manifest (`library.commit`), pinned by `pins/t4a-faer.rev`
    Caveat: The fork crate is `colmat` (`[lib] name = "colmat"`); the upstream baseline is `faer` (`faer_upstream = { package = "faer" }`). They are distinct Rust crate identities, but their `Mat`, `Par` and trait types are not interchangeable: every adapter is written per arm, and shared transitive dependencies can unify Cargo features, so both arms declare equivalent feature sets and the resolved lock is recorded.
    As measured: `faer-fork` (the provider in the measured revision; see library and upstream above)
  - `lapack-mkl` — The same calls against Intel oneAPI MKL on the fat node, linked from the installation `MKLROOT` names. One binary links one vendor library, so this row is measured by its own build of the harness and recorded as its own run; no page carries two vendor arms. The LP64 interface layer and the threading layer are pinned by the harness before the first MKL call, because MKL otherwise lets the environment choose them and the interface layer is an ABI.
    Identity: the MKL build recorded in this run's manifest (`providers[mkl]`)
    Caveat: MKL reports no dispatched-kernel name and no processor count, so its manifest records neither rather than inferring them. Its threading is vendor-owned, as for OpenBLAS. MKL is the vendor arm this profile is measured with: the fat node is an Intel machine, and the F1cpu ohtaka suites use OpenBLAS instead, so the two profiles' vendor rows do not compare.
    As measured: linked shared; mkl 2025.2; kernel not reported by the vendor

## Hardware

- CPU: `Intel Xeon Platinum 8280`
- logical CPUs: `112`
- L3: `39424K`
- L3 domains: `39424 KiB per socket: 4 domains of 28 CPUs, one per NUMA node, node k = CPUs = k mod 4`
- OS / arch: `Linux 4.18.0-372.9.1.el8.x86_64` / `x86_64`
- hostname: `c18fat5`
- CPU sets: 16T -> `0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60`, 32T -> `0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,1,5,9,13`, 64T -> `0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93,97,101,105,109,2,6,10,14,18,22,26,30`, 112T -> `0-111`

Every verified cell matches a successful verification census, including operation variants.
Verification was narrowed (`verification.grid`): only the shapes n = 2048 were verified. Cells at the other declared shapes of those populations were timed and had to succeed, but no verification checked them; a population the grid does not name was verified at every shape.
Each cell is median of process medians ± sample SD of those medians; W is the median
within-process sample SD. All values are milliseconds. These are descriptive spreads,
not confidence intervals. The between-process spread includes the rotating-order schedule.
Each process retains 3 chronological calls; 2 complete process sets are recorded.
Spread guide (computed from this run's own rows): per-engine between-start range median 3.26%, worst 110.63% among cells of 1 us or more (a sub-microsecond row is timer and scheduler jitter, not kernel time); 6 of 16 engine cells exceed 5% and carry `!`; median within-process relative SD 0.72%. A cell whose `±`, `!` or `p` is as large as the difference you are reading is not evidence of a regression.
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
two rows (combined between-start SD, floored at 5%) and larger than each of the two rows' `±`,
always with the measured library as the subject:
`🔵 <row> X faster than <row>` / `🔴 <row> X slower than <row>` (the subject is the first column of the table, the compared row is named in
the cell, not only in the column header), `·` inside the band or within a row's `±`, `-` that row is not
measured in this arm. `notes` carries the cell warnings (`*` a declared row has no measurement,
`!` between-start range over 5%, `p` position effect over 5%, dispatch-dominated means the
row is under 20 us so 4T/8T pay thread-region start-up). Per-start ratios and every
chronological sample are in the run directory named at the top of this page (`run*.csv`).
These describe the recorded revisions; they are not a regression claim.
Below the timing baseline (see the header) `·` also means a ratio within 1.10×.

## gemm-ohtaka-fat, f64, f order, 16T (CPU 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 19.2519 ± 0.0764227 (W 0.0623407) | 15.5104 ± 0.0695079 (W 0.11335) | 🔴 faer-fork 1.24× slower than `lapack-mkl` |  |
| `gemm/4096x4096x4096/b1` | 154.566 ± 0.254367 (W 0.213807) | 118.69 ± 0.312205 (W 0.32212) | 🔴 faer-fork 1.30× slower than `lapack-mkl` |  |

## gemm-ohtaka-fat, f64, f order, 32T (CPU 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,1,5,9,13)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 14.1765 ± 0.0822492 (W 0.0923958) | 11.5202 ± 0.212166 (W 0.177456) | 🔴 faer-fork 1.23× slower than `lapack-mkl` |  |
| `gemm/4096x4096x4096/b1` | 104.283 ± 0.0620076 (W 0.192456) | 168.056 ± 14.0214 (W 14.4264) | 🔵 faer-fork 1.61× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-ohtaka-fat, f64, f order, 64T (CPU 0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93,97,101,105,109,2,6,10,14,18,22,26,30)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 10.8219 ± 0.0445513 (W 0.0590501) | 11.5082 ± 0.646019 (W 2.28353) | · | `!` between-start range over 5% |
| `gemm/4096x4096x4096/b1` | 68.7415 ± 1.92995 (W 0.226943) | 306.381 ± 239.682 (W 120.609) | · | `!` between-start range over 5%; `p` position effect over 5% |

## gemm-ohtaka-fat, f64, f order, 112T (CPU 0-111)

| case | faer-fork (ms) | lapack-mkl (ms) | vs `lapack-mkl` | notes |
|---|---|---|---|---|
| `gemm/2048x2048x2048/b1` | 8.62718 ± 1.24411 (W 0.161126) | 14.7207 ± 1.75833 (W 0.279301) | 🔵 faer-fork 1.71× faster than `lapack-mkl` | `!` between-start range over 5%; `p` position effect over 5% |
| `gemm/4096x4096x4096/b1` | 57.4978 ± 1.59127 (W 0.346656) | 68.4509 ± 31.9107 (W 4.72246) | · | `!` between-start range over 5%; `p` position effect over 5% |

Repeated process sets describe observed repeatability under the recorded order schedule; they are not a paired baseline/candidate experiment and do not establish equal performance between engines.
