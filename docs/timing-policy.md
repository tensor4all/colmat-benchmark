# Timing policies

## Version 2: reliable re-record

The revised contract is in [reliable-measurement.md](reliable-measurement.md).
It retains actual outputs for checks, aligns the factor-only QR boundary, owns
and observes the Rayon pool, retains chronological samples and separate process
sets, and reports median-of-process-medians with distinct within/between SD.
Representative controls must freeze the budgets before the full re-record; the
old GEMM pilot alone does not establish a universal priming duration.

Historical numerical pages remain version 1. The new policy does not retroactively
repair those measurements or support comparing ratios across timing versions.

## Historical version 1

Recorded in every `run.yaml` as `timing_policy`. A run carries the version in
force when it was taken; do not compare ratios across versions, and do not use an
earlier version's numbers to assess a later library revision.

## What is timed

The common loop clocks `Op::run`; `Op::prepare`, fixture construction, and
`Op::check` are outside it. GEMM uses preallocated input/output buffers and restores
the output outside timing. Composite families allocate through their public API
inside `run`, and their local outputs are also destroyed before `run` returns;
that destruction is timed. In particular, the historical vendor `qr_factor` also copied input and
allocated tau inside `run`, whereas faer's low-level route prepared buffers
outside it: these were not equivalent factorization-only measurements.
Common timer code alone is not evidence of equivalent work or steady state.
Several composite checks refactor input instead of inspecting the timed result;
a successful check there does not validate the exact result that was timed.
Upstream faer's own bench page (`faer/examples/bench.rs` + `bench.toml`) restores
inputs inside the clock. Compare only after auditing the actual API boundary.

**Allocation performed inside the measured call stays inside the clock.** The
fork's two row splits allocate their partial buffers inside the public call, and
the public scratch function does not expose them. The campaign reports that as it
is instead of changing faer's API to make the boundary look cleaner.

Tensor construction, dtype dispatch, tlinalg batching/packing and tenferro
session entry are other repositories' work and are not in any number here.

## Priming

At least 0.5 s of untimed, time-based priming per row, and the same amount of
time on every row. A fixed call count is not equivalent. Startup/order sensitivity
has been observed, but neither its hardware cause nor a sufficient priming duration
has been established by the current controls. The sibling harness in `tlinalg-rs` records
`minimum_untimed_priming_ms: 500` while `tprims-rs`'s timer performed a single
warm-up call; `fbench` measures the priming time instead, and the recorder records
the value it was asked for.

## Thread budget

`--threads N` is required. At `N=1` both faer adapters select `Par::Seq`. At larger
budgets the fork row owns a `threadteam` team of `N` members for the whole measured loop
(`faer::team::with_team` since tensor4all/faer#85, where a team token is the only parallel
route and cannot be built from a number), and the upstream row holds the spindle worker group
as it always has. A fork checkout from before the team API still selects `Par::rayon(N)`,
which was a splitting hint rather than a host-owned pool, so its threaded labels are not
verified worker counts.
The vendor budget is set and read back once at startup (`openblas_*num_threads`
for OpenBLAS, OpenMP setter plus `mkl_get_max_threads` for MKL); a mismatch is an
error. This is the reported maximum budget, not a trace of active worker usage. The ambient `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`,
`RAYON_NUM_THREADS`, the MKL budget/domain and interface/threading-layer variables
and the OpenMP limit/placement variables are removed from the measured process,
and their absence is recorded. The harness pins MKL's LP64 interface and its
threading layer before the first MKL call, because MKL otherwise lets the
environment choose them and the interface layer is an ABI.

## Statistic and repetitions

`best` wall time per engine row and case, over 5 timed repetitions, with the
complete set repeated for A/A when a comparison is being made. The harness
reports the best time per row; the report aggregates repeats and prints the A/A
spread. A single scan flags suspects; a complete-set repeat alone also does not
remove systematic order or process-state effects.

The intermediate v1 harness additionally retained `samples_ms`, an optional JSON array of
**all** timed calls in chronological order, for each successful row. It is empty
for skipped/failed rows. Historical CSVs have no such array; its absence is not a
measurement of zero variance. Existing `total_ms`/`per_item_us` remain best-of,
and the legacy report still aggregates those minima, not sample distributions.

Use `python3 scripts/summarize_samples.py run*.csv > distributions.csv` to obtain
per-file, per-record sample count, mean, median, sample variance, SD and range.
Duplicate case occurrences remain separate. Sequential calls are not independent
process trials; do not pool them to manufacture confidence intervals.
`--rotate-rows K` is an explicit row-order probe, not an automatic recorder policy
or a guarantee of fairness. No new repetitions, priming or estimator default is
adopted until the pilot extends to representative cases. The bounded GEMM pilot
and its source/binary identities live under `data/probes/20261015-reliability-gemm/`.

**The noise floor is measured here, not inherited.** The sibling's measured floor
includes tlinalg's lane policy, which this campaign removes, so the direct-faer
pooled spread is a property of this harness and this host and is measured here
before any pooled number is quoted as a value. The sibling's own result stands
for its rows only: a per-cell claim there is made on the 1T rows (a complete set
reproduces to about 3%), the pooled 8T rows move by up to 2.5x between complete
sets of identical code, and the vendor's threaded rows sit in between. Until this
campaign has its own A/A pair, a table here is a scan. 8T kernel-scaling claims
are made from complete-set A/A pairs, not excluded by a 1T-only rule, because the
fork's row splits are exactly a threaded-kernel change.

## Guarding the host

## What runs when: the gate and the publication are two different runs

The two suites answer different questions, so they run on different triggers.

| | `faer-kernels` (small) | `faer-kernels-large`, 1024 tier |
| --- | --- | --- |
| question | *has anything regressed* - the whole family list at every thread budget | *does the result hold at scale* - the first size whose working set leaves the cache |
| shapes | squares 32-512, tall/wide up to `4096x256` | `1024`, `1024^3`, and the boundary shapes `255/257/511/513` |
| thread budgets | 1T, 4T, 8T | 4T, 8T |
| families | all 14 | the 10 factor/decompose families plus `gemm` |
| working set | 2 MB at 512, resident in L2/L3 | 8 MB at 1024, which fills the Zen5c 8 MiB L3 domain, so bandwidth and reuse dominate |
| cost, both arms | 5h17 (measured 2026-10-19) | 5h37 (measured 2026-10-19) |

**A candidate pair is gated on the small suite only.** That is where the control rule and the
per-cell tails live, and it is what a merge decision is made on.

**A publication records both, at the same revision, back to back**: the small suite for the whole
declared grid, then the 1024 tier. A publication is the event that moves the pin, and its pages are
what a reader quotes. About eleven hours of wall time, which is the largest thing this machine does
on a schedule.

**The 1024 tier is also required when a diff touches the paths whose behaviour only shows at
scale**: `faer/src/linalg/matmul/**` (kernels, packing, blocking), the panel and blocking code of
`lu`, `qr`, `cholesky`, or anything that changes how work is split across threads. A diff that
touches docs, a dependency, or a dtype-only fix does not need it. The candidate branch that made
this split (`faer#18`, the row-major copy floor) is an example of a change that did *not*: above the
floor it runs the baseline's code path, so the small grid was sufficient and the 1024 tier was not
run for it.

**Anything above 1024 — the 1536 tier and the 2048/3072/4096 squares — runs on demand only**, when a
question needs it, and its page says it was recorded out of cadence. The estimate in
`faer-benchmark#22` is what that decision is made on: 1536 is roughly 3.8x the work of 1024 per cell
and the 2048+ squares 8x to 64x, which is days of wall time rather than hours.

The frozen settings and preconditions for the current baseline are in
[protocol-v2-baseline.md](protocol-v2-baseline.md).

Every measurement runs through `benchmarks/scripts/pinned.sh` in this repository
(the harness owns the protocol, because faer has no `benchmarks/` directory and
the fork's `AGENTS.md` forbids adding one): `taskset` to the declared CPU set, a
3-second `/proc/stat` idle window covering those CPUs **and every CPU sharing
their last-level cache** (SMT siblings included - a measured set that fills only
part of its domain would otherwise leave cache contention invisible), at most 5%
busy before and after, 3 attempts. A spoiled run is discarded, not
published. The recorder holds one campaign-level lock across build, verify and
timing, so no second recorder builds another revision while a measurement runs;
unrelated workloads remain an operator precondition, and that includes the observing side itself.
A cell's measured threads share one L3 domain; the agent, the recorder and the guard checker belong
on the CPUs outside it. `record_run.py` derives both sets from the machine topology, refuses to
start when its own affinity overlaps the measured set (SMT siblings included) or when a declared
CPU set spans two L3 domains, and prints the derived free CPUs. It also walks the ancestor chain: an
ancestor that still owns a measured CPU and was using at least 1% of a core over a half-second
sample is refused by name,
because pinning the recorder cannot move the agent that launched it - that is how two runs of this
campaign were spoiled. The pin therefore fails the run
instead of publishing a contaminated number, and no CPU list is hardcoded per machine - the profile
for each host carries its own sets. Before any of that it refuses a host the named profile does not
describe: the profile's declared logical CPU count and SMT state against the host's (`smt/active`,
or the sibling map where the kernel exposes no switch), both sides printed, nothing written. One
profile name describes one machine.

The refusal leaves no trace once it passes, so an admitted run also *states* the pin: the manifest's
`threads.observer` carries the recording process's own `sched_getaffinity` and the CPUs that were
free of every measured L3 domain (`affinity`, `free_cpus`, and the `source` of both), both derived
from this host's topology. A run recorded before that field existed is still valid - the refusal
that admitted it was the same rule - but its observer pin is not visible in its provenance, and the
agent's own pin has to be re-established after a restart because affinity lives in the process.

## What a report may and may not claim

A report states the fork revision, the upstream revision, the harness, the host,
the vendor and its kernel, the coverage and the guards, and it is a claim about
*that* cell. It is not evidence about a population the run did not contain: a grid
whose large regime has unit batch extent says nothing about per-batch-element
cost at that size, and the index's `coverage` column plus
`run_spec.populations[].skipped` are the mechanisms that keep that visible. C and
F order are separate cells, and a C-order cell may not be read as the F-order one.
