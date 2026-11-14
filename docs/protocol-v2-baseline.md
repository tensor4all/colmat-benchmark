# Frozen protocol — timing-v2 baseline and candidate mode

This is the contract the baseline re-record and every later optimization iteration
are run under. It is frozen as of this commit; changing it needs an explicit
decision and a new version suffix on the page, because numbers recorded under a
different contract are not comparable.

## 0. Provisional pages and the future clean dataset

This host is a **development** host: acceptance runs here decide merges, and the pages it records are
provisional until a dedicated measurement host re-records the suite periodically. The split, what each
job covers and the handover checklist are in
[provisional-and-clean-datasets.md](provisional-and-clean-datasets.md).

## 0b. Provenance correction (2026-10-16)

The runs recorded on 2026-10-15/16 linked the crates.io release `t4a-faer 0.24.5` for the `faer-fork`
row instead of the pinned checkout (0.24.4), because `Cargo.lock` pinned 0.24.5 and the path patch
was silently ignored. See [reviews/20261016-linked-crate-provenance.md](reviews/20261016-linked-crate-provenance.md).
The build now refuses to run unless the resolved graph takes both measured crates from the named
checkouts; those pages are not evidence about `bfe4041` and were not re-recorded.

## 1. Frozen measurement settings

| Setting | Value |
| --- | --- |
| `--prime-ms` | **250** (calibrated twice: 250/10 and 250/30 are within 0.978-1.006 of 3000/30 over 18 rows; and 150 ms was not adopted because one of four representative cells read ~3% fast in all rotations and its rows were noisier - see `data/probes/20261015-priming-150/`) |
| `--reps` | **5** |
| `--aa` | **3** for the 3-row MKL arm, **2** for the 2-row OpenBLAS arm - a complete rotation cycle either way, so every row occupies every position once (the recorder requires `aa % len(engines) == 0`) |
| `--statistic` | `median` |
| Coverage | the declared suite, fully: all populations, `f64`+`c64`, order `c`+`f`, batch 1, threads 1/4/8 |
| Small-shape grids | `gemm-small` and `linalg-small` are declared as **8, 64** only: their 4T/8T cells were measured to be thread-region dominated (gemm 32^3 pays a constant +8 to +11 us at 4T/8T; 4^3 never enters a parallel path; MKL +0.3-0.8 us), so they are not worth the machine time at every budget |
| Rows per arm | MKL: `faer-fork`, `faer-upstream`, `lapack-mkl` (`--engines`/`--rows`). OpenBLAS: `faer-fork`, `lapack-openblas`. The duplicated faer rows differ by 2% median (26% worst, the known rotation bump) between arms, so the fork/upstream pair is kept in one arm and the vendor pair in both |

## 2. Frozen machine preconditions (enforced, not remembered)

- **Boost/turbo off.** The recorder refuses to start while a boost switch is on
  (`cpufreq_boost=1` or `intel_pstate/no_turbo=0`); the exact command is printed.
  `FBENCH_ALLOW_BOOST=1` records the deviation instead, and such a run is not
  comparable with a boost-off run. Boost on/off is the largest single factor seen
  so far: `cholesky` 512^2 f64 8T process-median range fell from 40-73% to 1.5-5%
  once boost was disabled, `lu` 256^2 c64 4T from 74% to 33%.
- **Power state recorded.** `host.power` in `run.yaml` carries boost, governor,
  EPP hint, `scaling_max_freq` and the platform profile when the host exposes
  them. Governor/EPP/profile are `performance` for the baseline host.
- **Measured threads share one L3 domain.** A declared set spanning domains is
  legal and warns instead of failing; every CPU in every touched domain must then
  stay idle.
- **The observing side stays off the measured L3 domains.** The recorder derives
  the measured set and the free CPUs from sysfs and refuses to start when its own
  affinity overlaps; the agent's own pin lives in its process and is
  re-established after a restart. A host whose measured set covers
  every L3 domain has nowhere to park the observer; that is recorded as a warning (a fact about the
  machine), while an observer that may land on a measured CPU is still refused.
- **The idle guard watches whole L3 domains**, SMT siblings and cache peers
  included, so a partially filled domain cannot hide unrelated work. 3 s window,
  `max_busy=0.05`, `PINNED_RETRIES=8`, `PINNED_WAIT=30`; a rejected attempt is
  discarded and re-run on the same schedule, never selected by its numbers.
- One campaign lock, build-then-verify-then-time, no build while a measurement
  runs, abort on any failure with the artifacts preserved.

## 3. What a recorded number means

- Per row: chronological calls, median reported; per cell and engine: median of
  the process medians, between-start sample SD, median within-start SD.
- Comparison: the **rotation-averaged ratio** `exp(mean over starts of
  log(median_A / median_B))` inside a cell, with the per-start ratios printed
  beside it. It removes a position effect only under one position effect shared by
  the engines and stable across starts, and only where every declared engine has a
  measured row (`*` marks a cell where one does not).
- Warnings, never rejections or trimming: `!` a per-engine between-start range
  above 5% of its median, `p` a balanced cell whose estimated position-effect
  ratio exceeds 1.05.
- No confidence intervals, no equality or superiority claims, no cross-cell
  pooling, `c` and `f` never merged, skipped cells recorded as skipped.

## 4. Candidate mode (optimization iterations)

An optimization iteration does **not** re-measure every backend on every test. It
runs in three steps, and a comparison is published only if all three pass.

1. **Same-machine precondition, checked from the record.** The candidate run's
   `library` commit is the only identity allowed to differ from the baseline. Its
   upstream revision, harness commit, toolkit commit, suite declaration, host
   block (`cpu`, `logical_cpus`, `l3`, `l3_domains`, `host.power`), thread CPU sets
   and timing policy must equal the frozen baseline run's. A difference in the
   machine state refuses the comparison before any timing.
2. **In-run machine-state control.** The candidate binary carries the *unchanged*
   `faer-upstream` row. A predeclared overlap set of cells — at minimum
   `gemm 256^3 f64 1T`, `lu 256^2 c64 4T`, `cholesky 512^2 f64 8T`,
   `qr_factor 4096x256 f64 8T`, `svd_thin 64x1024 c64 4T`, `householder_apply
   4096x256 f64 8T` — is measured and the upstream row compared with its recorded
   baseline value. If it leaves the recorded between-start spread (or the cell's
   warning threshold), the machine is not in the baseline state and the comparison
   is refused rather than reported.
3. **Fork-only measurement of the selected cells.** Once step 2 passes, only the
   `faer-fork` row (plus the upstream control row) is measured on the chosen
   subset. The comparison is the rotation-averaged ratio of the candidate fork to
   the recorded baseline fork per cell, with the candidate's own process spread
   recorded beside it.

Rules that make this honest rather than convenient:

- The subset is declared **before** the run, is reported with the comparison, and
  is never widened after seeing the numbers.
- Missing coverage is reported as missing; a cell that was not measured carries no
  inference, and "unchanged elsewhere" is not claimed from a subset.
- A candidate run that has to be repeated keeps both attempts; the reported one is
  never selected because its numbers look better.
- Absolute values for the candidate come from its own run, not from copying the
  baseline's; the baseline remains the reference for the fork-vs-upstream and
  fork-vs-vendor ratios.
- Adoption of an optimization still requires the same evidence as any other
  change: the candidate comparison above, the numerical verification grid, and an
  independent review of the finished diff.

## 5. Size tiers and the two operating modes

The declared square tiers follow faer's own published benchmark distribution
(`faer-bench`: powers of two plus the x1.5 intermediates 96/192/384/768/1536/3072),
minus 4/8/16: `small` (32, 64, 96 - overhead and regression checks), `mid` (128, 192,
256, 384, 512, 768 - the block/parallel switchover region), `large` (1024, 1536), and
two adoption-only tiers: `linalg-big` / `gemm-huge` (2048, 3072, 4096) and
`linalg-boundary` (255, 257, 511, 512, 513 - only for suspected block-boundary
overfitting; 512 is a power-of-two / 4K-alias diagnostic read only as the three-point
set 511/512/513 from one run, never alone, never in the trend or a cross-size
aggregate, and with MKL's SVD slow at 512 - see the suite's `known_blind_spots`). 4, 8 and 16 are deliberately not in this campaign: they belong to a
separate small-matrix loop bench, which is also where a "too many threads on too
little work" side-effect check belongs.

- **Daily optimization** measures one square per tier plus the tall/wide work:
  `--shapes 64,64x64x64,256,256x256x256,1024,1024x1024x1024,1024x64,4096x256,64x1024,256x4096,4096x64x64,16x4096x240,4096x16x240,64x64x4096`
  with `--regimes` naming the daily tiers. A run records the subset it covered, so
  a daily subset is never mistaken for the whole declaration.
- **Large tiers are their own suite** (`faer-kernels-large`, `threads: [4, 8]`):
  1024, and the adoption-only 2048/4096 and the diagnostic 255/257/511/512/513. A run
  time dominated by the biggest shapes is not worth a 1T pass, and splitting by
  thread budget keeps each page a full cover of its own declaration instead of two
  partial pages overwriting each other. The daily suite keeps the 1/4/8T
  small/mid/tall/wide grid.
- **Before adopting a change** the same arms run the full declared lists (no
  `--shapes`), and the expensive tiers are added explicitly:
  `--regimes linalg-big,gemm-huge --threads 4,8 --dtypes f64` (2048/4096 are not
  worth 1T or c64 in the loop) and `--regimes linalg-boundary` only when a change
  is suspected of fitting a block boundary.

## 5b. Adoption protocol for an optimization change

A full-suite A/A with a control gate on every cell is the right tool for *publishing* a comparison and
the wrong tool for deciding whether to keep an optimization: it costs hours and it refuses on cells
that have nothing to do with the change. Adoption therefore uses three criteria, all of which must
hold, and the evidence for each is recorded in the campaign.

1. **Correctness, no counter-evidence.** The measured change must pass its own tests and the harness's
   numerical grid on the populations it touches: `cargo test -p <fork> --lib` in the fork checkout,
   the regression tests the change added, and `fbench verify` over the affected populations and thread
   budgets with the fixed build (see the patch-resolution guard in `scripts/build_for_faer_rev.sh`).
   A single mismatch or skip that was not declared before the run fails this criterion.
2. **The target improves, measured.** Before the run, the *target cells* are declared - and they are
   derived from the recorded benchmark, not chosen ad hoc: they are the cells where the campaign's own
   pages show the fork losing to upstream (or to the vendor) on the families the change touches, in the
   memory order it addresses. For the row-major QR/LU work those are the C-order cells of
   `linalg-wide` (`qr_factor/64x1024`), `linalg-mid` (`lu/n256`, c64 4T/8T), `linalg-large`
   (`lu/n512` 1T) and `linalg-tall` (`4096x256`), all of which the daily declaration already covers. They are measured for the candidate
   and for the pinned baseline with the frozen settings (prime 250 ms, 5 calls, three complete process
   sets, boost off, observer off the measured L3 domains), and compared as the rotation-averaged
   ratio per cell. "Improves" means the ratio leaves the between-start spread the baseline recorded
   (the same band the pages print) and the direction is consistent across the cells of the target set.
   The target set is never widened after the numbers are seen.
3. **No regression outside the target, checked on a declared guard set.** A guard set - the same
   families in the other memory order, neighbouring sizes, other thread budgets, and the unchanged
   `faer-upstream` row as the machine-state control - is measured in the same runs. The change is
   adopted only if no guard cell is slower beyond the recorded spread, and the control row reproduces
   the baseline. The control reproduction is decided on the **median** of the unchanged row's per-cell
   ratios, which must be within 2% of 1; every cell's own between-start range is printed beside it as
   the tail it is, but a single moved cell does not refuse the comparison. The reason is measured, not
   assumed: two runs of the *same* revision refuse a cell-by-cell rule 1/396 and 12/396 on the
   zen5-cpu host, and across the saved pairs the unchanged row's per-cell ratios span 0.74-1.36 with
   up to 8% of cells outside +-10% while every median sits within 0.2% of 1. The unchanged row moves
   with the *linked binary*, because the two runs compile different trees, so a cell-by-cell rule
   reports a layout effect as a moved machine and no candidate can pass it. A median outside 2% is a
   global shift and still refuses; `scripts/candidate_compare.py` prints the median, the outlier
   count and the ten largest ratios.

Adoption itself is a documentation act on top of those measurements: the candidate revision is
recorded as the adopted one in `PATCHSET.md` and the campaign worklog, with the target and guard
results linked from the run directories that carry them. Re-recording the whole suite is not part of
adoption; the next full baseline re-record picks the adopted revision up.

This is the same rule for every later optimization: declare the targets, prove correctness, measure
the target, check the guards, record it. The expensive full-suite discipline stays where it belongs -
on the pages that are quoted as the campaign's numbers.

### The two rules that keep acceptance affordable

**Measure once.** The candidate is measured at the revision that will be merged - the branch head with
a clean tree - and the merge must not change the code that was measured. `scripts/candidate_compare.py
--merged-commit <rev> --checkout <fork checkout>` compares the tree hashes of the campaign's measured
paths between the two revisions and refuses the comparison when they differ, so the same code is never
measured before and after the merge. The acceptance record quotes the measured commit *and* those tree
hashes, which is what keeps the evidence valid across a merge, a rebase or a rename. A merge that does
change a measured path invalidates only the cells that depend on that path; they are re-measured with
the target/guard set, never as a whole suite.

**An hour, not a day.** The acceptance grid is derived from the change, not from the whole
declaration:

| part of the grid | coverage |
| --- | --- |
| affected families | every *declared* shape, dtype and order of the families the change can reach, at 4T and 8T |
| target cells | the same families at 1T on the cells the change is written for |
| collateral control | two or three cells of families the change does **not** touch (e.g. `gemm`, `cholesky`), at 4T and 8T, in the same runs |
| everything else | reused from the recorded baseline; re-measuring families the change cannot reach buys nothing and is what made acceptance take a day |

The affected families are declared **before** the run by reading the changed paths: a change to the
no-pivot QR factor reaches `qr_factor` and `qr`; a change to partial-pivot LU reaches `lu`,
`full_piv_lu` and `solve`; a change to a shared helper such as `mat/matref.rs` reaches every family that
copies or owns a matrix, and the declaration must say so. The declaration is recorded with the run and
is not widened after the numbers are seen. Settings stay as frozen in section 1 (prime 250 ms, five
calls, three complete process sets, boost off, observer off the measured L3 domains, whole-L3-domain
idle guard, `PINNED_RETRIES=8`).

Expected cost, to be confirmed by the first run and recorded next to it: the row-major QR/LU work
touches five families over the ten declared square/tall/wide shapes, which is about 600 rows per budget
per arm - roughly 10 min of priming, 12-20 min of measured calls, 10-15 min of verification and 15 min
of guard windows, so **about an hour for both arms**; a change that reaches a shared helper covers more
families and scales with them. A run that needs more than three hours is a protocol defect to fix, not a
reason to drop the target cells. Pages produced this way state which families were measured at which
budget (coverage partial) instead of implying the whole declaration.

## 6. How the two arms are invoked

```sh
PINNED_RETRIES=8 python3 scripts/record_run.py zen5-cpu faer-kernels \
  --checkout extern/t4a-faer --upstream extern/upstream-faer --vendor mkl \
  --engines faer-fork,faer-upstream,lapack-mkl --threads 1,4,8 --dtypes f64,c64 \
  --reps 5 --aa 3 --prime-ms 250
PINNED_RETRIES=8 python3 scripts/record_run.py zen5-cpu faer-kernels \
  --checkout extern/t4a-faer --upstream extern/upstream-faer --vendor openblas \
  --engines faer-fork,lapack-openblas --threads 1,4,8 --dtypes f64,c64 \
  --reps 5 --aa 2 --prime-ms 250
```

`--engines` declares the rows this arm measures and is passed to the harness as
`--rows`; the recorder's verification census and the page's columns follow that
declaration, so a run that measures a subset cannot claim coverage of a row it did
not measure. The two arms are separate runs (the vendor symbols collide) and each
publishes its own `-timing-v2` page.

## 7. What is explicitly out of scope

Establishing the mechanism of the remaining between-start variation; publishing
candidate numbers as if they displaced the baseline; treating the vendor rows as
identical low-level implementations; and any claim about cells the declared suite
does not contain.
