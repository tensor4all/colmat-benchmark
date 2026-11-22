# Rules and runbook

Everything a change author, an agent or the future measurement host needs, in the order it is used.
The normative text is `docs/protocol-v2-baseline.md`; this page is the index and the copy-paste version.

## Two jobs

| job | where | grid | published |
| --- | --- | --- | --- |
| **Acceptance** - may this change be merged into `colmat`? | the development machine | the change's blast radius: affected families at every declared shape at 4T/8T, target cells at 1T, two or three untouched families as a collateral control | no - evidence in the PR, the worklog and the fork's `PATCHSET.md` |
| **Public dataset** - the pages under `result/` | a dedicated clean measurement host, re-recorded periodically | the whole declared suite | yes |

The fork side of the same rules is `colmat-rs/docs/adoption-protocol.md`.

## The contract every run uses

`--prime-ms 250 --reps 5 --aa 3 --statistic median`, CPU boost off (the recorder refuses otherwise),
the observing side pinned off the measured L3 domains (refused otherwise), an idle guard that watches
every CPU sharing the measured cache (3 s window, 5% busy, 8 admission retries), one campaign lock,
the colmat and faer rows timed with the row's parallelism held around the whole loop when the budget is more
than one thread: the fork row through `faer::team::with_team(threads, ..)` (its team is the only
parallel route since tensor4all/faer#85, and the token it lends is what every measured call uses),
the unchanged upstream row through the spindle worker group (`spindle::with_lock(threads, ..)`, which
is how it has always been measured). Runs before the fork's team row was measured this way are not
paired with later ones by the comparator,
build then verify then time, and a published page only from a validated run.

## Optimization loop: target 64, guard 32

An optimization iteration measures the **64** cell of every family and confirms **32** is unmoved,
then stops. Both are in the command below; the 32 row is the guard, not an option. That pair is the iteration, not the adoption: adoption still runs the arm below
(affected families at every declared shape, collateral controls, the comparator).

```sh
python3 scripts/record_run.py zen5-cpu colmat-kernels ... \
    --shapes 64,64x64x64,32,32x32x32 --dtypes f64,c64,f32,c32 --threads 4,8 --label loop-<rev>
```

`fbench` itself defaults to that grid when `--n` is omitted, and `--dtype` accepts all four native
dtypes as of `02f5ae8` (`docs/harness-dtype-axis.md`). The rationale is measured, per dtype: the floor that decides whether a product is threaded is 1.18
Mflops, so `32^3` (131 ki real, 525 ki complex) and `64^3` (524 ki real, 2.10 Mflop complex) are below
it for the real pair and the 32 cell is below it for all four, which is what makes the 32 row a guard
on the un-threaded path rather than a scaling story. The 64 cell is the smallest one the kernel
decomposes for real inputs.

## Acceptance (about an hour)

```sh
cd /home/shinaoka/projects/tensor4all/faer-benchmark
# 1. declare the blast radius of the changed paths, then measure the affected families at 4T/8T,
#    the target cells at 1T and a couple of untouched families as a control. Example for row-major QR/LU:
export PINNED_RETRIES=8
python3 scripts/record_run.py zen5-cpu colmat-kernels \
    --checkout ../faer --upstream extern/upstream-faer --vendor mkl \
    --engines colmat,faer,lapack-mkl \
    --regimes linalg-small,linalg-mid,linalg-tall,linalg-wide \
    --families qr_factor,qr,lu,full_piv_lu,solve \
    --dtypes f64,c64 --reps 5 --aa 3 --prime-ms 250 \
    --label acceptance-<revision> --no-publish
# 2. the pinned baseline is measured once and kept (baseline-ref-* runs under data/results/), never re-run
#    per candidate: the same command with --checkout extern/colmat
# 3. compare, and let the tool refuse a merge that changed the measured code:
python3 scripts/candidate_compare.py \
    --baseline data/results/zen5-cpu/colmat-kernels/<baseline-ref run> \
    --candidate data/probes/acceptance/<candidate run> \
    --checkout ../faer --merged-commit <merge commit> --json /tmp/acceptance.json
```

A comparison is accepted only when the identity check passes (only the library commit differs), the
`faer` control row reproduces the baseline, no measured guard cell regresses beyond the
recorded spread, and the target cells improve beyond it.

Gates that must also pass: `cargo test -p colmat --lib` in the checkout, the change's own regression
test, and the harness's `fbench verify` over the affected populations and budgets (the recorder runs it
before timing; a mismatch or an undeclared skip fails acceptance).

## Public dataset (the future clean host)

```sh
# both suites, both arms; run serially, commit after each so publication cannot dirty the next run
for suite in colmat-kernels colmat-kernels-large; do
  for arm in "mkl|colmat,faer,lapack-mkl|3" "openblas|colmat,lapack-openblas|2"; do
    IFS='|' read -r vendor engines aa <<< "$arm"
    PINNED_RETRIES=8 python3 scripts/record_run.py zen5-cpu "$suite" \
      --checkout extern/colmat --upstream ./extern/upstream-faer --vendor "$vendor" \
      --engines "$engines" --dtypes f64,c64 --reps 5 --aa "$aa" --prime-ms 250
    git add -A && git commit -m "data: $suite $vendor"
  done
done
```

The host's profile states `role: measurement`; a development host records `role: development` and its
pages say they are provisional. A host whose measured set covers every L3 domain records a warning
instead of refusing, because that is a fact about the machine.

## The rules that keep this affordable

- **Measure once**: the merge must leave the measured paths' tree hashes unchanged (`--merged-commit`).
  A merge that changes one invalidates only the cells that depend on it.
- **No widening after the fact**: the blast radius is declared before the run and recorded with it.
- **No whole-suite re-record on a development host** to refresh a page; the clean host replaces them.
- **Fail closed**: a rejected guard attempt is preserved and re-run on the same schedule, never selected
  by its numbers; a run whose validation fails is not published.
- **No hand-editing** `result/INDEX.md`, and historical pages stay as they were recorded - a new contract
  gets a new `-timing-vN` suffix.
- If acceptance needs more than three hours, fix the protocol, not the target set.

## Fresh-session dry run (evidence, 2026-10-17)

Run from a fresh shell with a minimal environment (`env -i HOME=... PATH=...`), using the commands
above as written:

| check | command | observed |
| --- | --- | --- |
| observer inside the measured L3 domain | `taskset -c 7 python3 scripts/record_run.py ... --regimes gemm-small --threads 1 ...` | refused: `ERROR: this recording process may run on CPUs [7], inside the measured L3 domain(s) [4-11, 16-23]; pin the observing side off them, for example taskset -c 0-3, 12-15 ...` |
| observer off the domain | same command under `taskset -c 0-3` | accepted; the CPU sets came from sysfs (`measured {4, 16}`, `free {0-3, 12-15}`) |
| boost gate | `require_boost_disabled({'cpufreq_boost': '1'})` | refused with the exact `echo 0 \| sudo tee ...` instruction; the live host records `cpufreq_boost=0` |
| acceptance command | `scripts/candidate_compare.py --help` | exposes `--checkout` and `--merged-commit`, as the adoption rule uses |

The first row is a real gap the dry run found: the observer check compared the measured CPUs and
their SMT siblings only, so CPU 7 (same L3 domain as the measured CPU 4) looked safe. It now refuses
any overlap with a measured L3 domain (toolkit PR #11). The reproduction is kept in
`data/probes/dryrun-refusals/`, and it is not evidence about any library revision.
