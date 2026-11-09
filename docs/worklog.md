# Worklog

Decisions, fixed failures and open items for this campaign. Newest first.

## 2026-11-07 — the joined page joins one protocol, and prints what it divides by (#54 review)

`gen_comparison.py` refused two arms from different hosts and nothing else, so two arms recorded
under different timing policies (one best-of call with no priming on CPUs 0-7, against 5 calls, a
median and 250 ms priming on CPUs 8-15) were joined into one table that printed **one** protocol —
the MKL arm's — above it. `require_one_host` became `require_one_protocol`: it compares the host
block, `timing_policy` (version, repetitions, statistic, priming), the **idle guard** (`guards`:
window, busy ceiling, attempts, and `sibling_aware` — a run that accepted a 100%-busy machine, or let
a measured core's SMT sibling stay busy, is not a quiet run's peer), the measured CPU sets per thread
budget and the observer's affinity, `host.power` (boost on and off are incomparable), the revisions of
library, upstream, harness and toolkit, and the dtype each arm was asked for, `skipped` rows
included. Every differing field is printed with both values, and nothing is written.

A thread budget is evidence about the CPUs it ran on, so both halves of that check follow the budget
**the arm actually measured** (its cells and its `skipped` rows): an entry for an unrelated 1-thread
budget used to excuse an 8-thread cell with no recorded CPU set at all, and a leftover 1-thread entry
the two arms disagree on used to refuse a join over CPUs neither of them measured a cell with.
A thread budget (and a family) one arm was not asked for is a smaller grid, not a different protocol
— that is what a partial run is — while a dtype disagreement is refused in both directions, because
an f32 cell never joins its f64 cell and the dash would be read as "that arm had no driver for it".
The check used to be one-directional: an arm asked for *more* thread budgets than the MKL arm was
refused, though the same grid difference in the other direction was accepted.

The arm's columns are labelled `MKL` and `OpenBLAS` while the row read for each arm comes from that
arm's own manifest, so an arm handed to the wrong flag (`--openblas <an mkl run>`) is now refused
instead of printed under the wrong heading; the two arms keep their own page as `AGENTS.md` requires.

The page also used to print one `fork (ms)` column and divide `OpenBLAS/fork` by the fork the
**OpenBLAS** arm measured, which was not the number printed: on the published 1356-cell page the two
arms' fork values differ in 1327 cells (median gap 0.7%, worst 17.7%). The fork is now shown per arm
(`fork MKL`, `fork OpenBLAS`) and every ratio divides by the value printed beside it, in its own arm.

Two more counting fixes, and one statement that was missing:

- the summary counted OpenBLAS cells by the `upstream/fork` ratio, so it claimed an OpenBLAS row for
  every cell the MKL arm carried, and printed the upstream median as the OpenBLAS one: the published
  page said `1356 cells ... OpenBLAS 1.956` where 972 cells carry an OpenBLAS row and their median is
  `1.230`. Verified by regenerating that page from
  `data/results/zen5-cpu/faer-kernels/20261031T224040Z-pub-71b25ffb-20261031-mkl` +
  `20261101T031016Z-…-openblas`: every one of the 1356 rows keeps every value it had, and the
  OpenBLAS row count and median are the corrected pair.
- a cell asked for by at least one arm and measured by neither is now counted and named (`360 of
  1716` cells on that page: the declared skips — `eig`, `upgrade_factor`, and non-square shapes of
  `cholesky`, `eigh`, `solve`, `triangular_solve`), instead of being dropped from a page whose dash
  rule only speaks about cells one arm did carry.
- the page renders the statistic, repetitions, priming and guard from the manifest and states which
  reduction it applied, instead of hardcoding v2 in the prose.

Still open: the reduction over a run's process sets is still the last set (`last_set`), because an
existing page is a record of what was published; the median-of-sets reduction is the right one for
both arms and is its own change. `threads.cpu_sets` is compared as the recorded string, so two
spellings of the same CPU set would be refused rather than recognised as equal.

The AOCL arm this page was originally written for does not land: `--aocl`, the per-cell "faster of
MKL and AOCL" reference and the harness's `link-aocl` build (AMD host gate, `libflame`/`libblis-mt`
link line) are not in this branch, so the page is what it always was — the two arms of the recording
— with the correctness fixes above.

## 2026-10-29 — the fork `lu` and `solve` rows ran sequentially under `fork-team` (#36)

Since #35, the timed `lu` call ignored its token. `partial_lu_of` handed `lu_in_place` the result of
`verify_par`, which is `Par::Seq` under `fork-team`, so the fork `lu` row measured a sequential
factorization at every thread count. The `solve` row factors through the same helper and was also
sequential. broccoli showed it: `lu` 1024 stayed at 45 ms from 1T to 64T, while `qr_factor` sped up
3.9x at 8T in the same binary (tensor4all/faer#99).

The helper now takes the token. A `fork-team` test runs every fork family on a team of two and
asserts that faer's team counters moved. It fails on the old helper. No recorded run was built with
`fork-team`, so no page changes.

## 2026-10-19 — the large tier: 1024 and the boundary shapes, and why 1536+ is not here

`faer-kernels-large` had no current record at all. The estimate (faer-benchmark#22) scaled the 512
medians by `n^3` and put the whole declared suite at 29.5 h for the MKL arm and 59 h for OpenBLAS,
with `linalg-big` (2048/3072/4096) taking 27.7 h of the MKL number.

The estimate was right about the shape of the cost and wrong about where it sits. The first attempt,
`linalg-large` with 1024 and 1536 together, spent **72 minutes in `verify-linalg-large-4t`** and had
not finished that one thread count when it was stopped: verify checks an O(n^3) reconstruction or
orthogonality per cell, and at 1536 that check is an order of magnitude over the timed part of the
estimate. So the tier is recorded in the shape that fits:

* **`20261019T102752Z-periodic-20261019-large1024-mkl`** - 3,672 rows, `linalg-large` at n = 1024,
  `gemm-large` at 1024^3, and `linalg-boundary` at 255/257/511/513, 4T and 8T, three process sets.
* **`20261019T130054Z-periodic-20261019-large1024-openblas`** - 4,896 rows, same shapes, two-engine
  arm, six process sets.

Each took about two hours, of which 40 minutes was verify. Both publish as `partial` pages at
`a5e6b7d`: partial because 1536 and the 2048/3072/4096 squares are not in them, and that is the
honest label.

**1536 is verify-bound**, not timing-bound, and it joins the 2048/3072/4096 squares in the same open
decision (#22), whose estimate now says so. The aborted `20261019T091625Z` directory is retained with
its build record and no `run.yaml`: it is not evidence, and it is the measurement that produced the
correction.

## 2026-10-19 — B1 taken: the twelve pre-comparator rows are recorded as what they are

The choice between re-measuring the twelve `KEEP` performance rows as campaign pairs (a day, and some
would not pass the current gate) and labelling them was taken as **B1, cost zero**. `PATCHSET.md` now
carries an "Evidence class of the adopted changes" section: two rows went through the v2 comparator
(one passing control, one whose control failed on the binary-layout effect measured separately), the
rest are *pre-protocol adoptions* from isolated drivers, and the three correctness rows have tests and
no timing claim. A page may not cite a pre-protocol row as a gated result.

The same commit settles the six candidates that were neither adopted nor dropped: the copy tile
`60de2ee` stays on **HOLD** because its own control question is unanswered (the ELF-identity argument
is demoted to a static observation, so the unchanged control that moved 17% still has no explanation),
and `7cd372f`, `f013756`, `7f713a1`, `ad9f2ef` and `d6358ca` are **DROPPED** with the gate that
refused each. The ledger's legend gains `HOLD` so the distinction is visible rather than implied.

Record: `tensor4all/faer` PR #22, merged 2026-10-19. Closes faer-benchmark #20 and #21.

## 2026-10-20 — the publication at `50c2b98`, and a browsable site

The pin moved to `50c2b98` (the row-major copy floor, `faer#23`) with the argument that only
`linalg-small/32` changes code path - the floor is 4096 elements, so 64^2 and above take the copy
exactly as before - and that those cells had been measured as a pair. That argument is fine for
deciding to move, but the published numbers at 32 were the *slower* baseline ones, so the pages were
re-recorded rather than left to understate the code.

Both arms were recorded at `50c2b98`: MKL `20261020T011108Z` (15,156 rows, full) and OpenBLAS
`20261020T033243Z` (20,016 rows, the two-engine arm). The first MKL attempt
(`20261019T224849Z`) was refused at the end with "measured source changed during recording" because
this repository was committed to while it timed - the site work landed in the middle of the run. It
is kept with its 72 CSVs and 24 verification files and no `run.yaml`; the arm was recorded again.

**The site.** `scripts/gen_comparison.py` joins the two arms on the cell key and writes one HTML
table per population - fork, upstream, OpenBLAS and MKL side by side with the baseline-over-fork
ratio - carrying the revisions, the hardware profile, the vendor identities and the timing policy in
the header. It is a view over the recorded data, not a new measurement: a cell an arm did not carry
is a dash, and the per-arm Markdown pages and their manifests stay authoritative for the control
statement and the verification record. `.github/workflows/pages.yml` publishes `result/` to
`http://tensor4all.org/faer-benchmark/`, with a landing page that links the Markdown pages to the
rendering GitHub gives them (Pages would serve them as text).

## 2026-10-19 — the periodic run at `a5e6b7d`: the first pages from a post-merge recording

The full declared `faer-kernels` suite was recorded at the merged main, both arms, sequentially:
MKL `20261019T025721Z` (15,156 rows, `full`) and OpenBLAS `20261019T052233Z` (20,016 rows, the
two-engine arm). Three process sets for the three-row arm and six for the two-row arm, the frozen
policy, the observer pinned off the measured domain and checked by the recorder before the build.

Two things went wrong on the way and both were caught before a number was published:

* the recorder refused the first attempt because `herdr` (a terminal daemon at 4% of a core) still
  owned the measured CPUs - the new ancestor check, refusing before the build rather than after a
  spoiled run;
* the OpenBLAS arm was refused with ">=2 calls, >=2 process sets and complete engine-order rotation
  cycles": two engines need a multiple of two process sets, so `--aa 3` is not a legal arm and
  `--aa 6` is. The earlier OpenBLAS recording used six for the same reason.

The MKL arm ran 2h20m with six guard rejections and the OpenBLAS arm 2h20m after a 17-minute verify,
with eleven; every rejected attempt is retained under its run directory and the accepted CSV is the
one the guard admitted. The pin moves to `a5e6b7d` and the index now reads from it: the MKL page is
`full`, the OpenBLAS page is `partial` because that arm has no upstream row, and both are `current`.

`faer-kernels-large` was deliberately out of scope for this session; the cost estimate and the
2048/3072/4096 squares are issue #22.

## 2026-10-19 — the CI failure was a kernel bug on AVX2-only machines, and it is fixed

`main` went red on `testing-stable-ubuntu-latest` at `2870062` in
`linalg::matmul::internal::tests::strict_triangle_maps_preserve_stored_entries_and_parent_guards`,
the guard test that shipped with the empty-extent/K0 fix. It was not a flaky test and not the LU
commit: the store epilogue of `private-gemm-x86`'s 256-bit kernels selected a shifted entry of
`mask.data` and clamped only the top of the range, so a vector row entirely past the diagonal read a
mask from before the table and stored a full vector into entries the destination structure says are
not written. The 512-bit kernels build their masks from k-register arithmetic, which is why the same
code is correct on an AVX-512 host and wrong on an AVX2-only runner - and why the failure appeared on
one CI run and not the eleven before it.

Reproduced by disabling the AVX-512 dispatch (`feat = Avx256`) locally, then at the kernel's own API
in `tensor4all/t4a-private-gemm-x86` (a fork of `sarah-quinones/gemm-x64-v2`): c64, m = n = 3,
k = 2, `DstKind::Upper`, one thread writes `(2, 0)` on the 256-bit path and leaves it on the
512-bit one. A sweep of {f64, c64} x {upper, lower} x m, n in 1..=9 x k in 1..=3 stores 486 entries
outside the triangle before the clamp and 0 after it.

Fixed in the fork by clamping the mask index at zero as well, published as
`t4a-private-gemm-x86 0.1.23` (the library keeps the name `private_gemm_x86`), and `main` now
depends on it: `4ad6c01`, "Merge pull request #19". The guard tests pass with the AVX-512 dispatch
disabled and enabled, and the lib suite is 214 passed / 2 ignored either way; all CI jobs on the PR
passed, including the one that had failed. Upstream report: `sarah-quinones/gemm-x64-v2#5`. The
first fix PR (#6) was closed without a comment and without a commit upstream, so the diagnosis was
resubmitted as #7, which touches `build.rs` and the test only — the package rename, the CI workflow
and the dev-dependency cleanup of #6 are not in it. Until upstream takes it, the dependency stays on
`t4a-private-gemm-x86`, whose main, tags `v0.1.23`/`v0.1.24` and green CI now match what crates.io
serves.

**Effect on this campaign's evidence.** `faer/Cargo.toml` and `Cargo.lock` are measured paths, so the
adoption pair recorded at `2870062` describes `2870062`, not `4ad6c01`. Nothing measured changes in
principle - `spicy_matmul` reaches dense LDLT and sparse Cholesky, neither of which is a declared
cell - but the next recording must be taken at the new revision rather than reusing these numbers,
and the pin stays at `2870062` until that run exists.

## 2026-10-19 — the site moves to the adopted revision

The pin moved to `2870062` with the adoption, so the site had to move with it. `result/INDEX.md`
was regenerated: the reference revision is `2870062`, and the `2240b07` pages are now **3 behind**
rather than current. The adoption pair's candidate run was published as
`result/zen5-cpu/faer-kernels/2870062a7ee6-0539947ffb75-mkl-timing-v2.md`, marked `partial` - it
covers `lu`, `solve`, `full_piv_lu`, `cholesky` and `gemm` at 1/4/8T, not the declared suite - and it
never displaces the full-coverage page at `2240b07`. It is the page that carries the cells the
adoption was decided on: `solve/n256` c64 is now 1.06x and 1.18x faster than `faer-upstream` where
the `2240b07` page had it 1.35x and 1.37x slower.

The full declared suite at `2870062`, both arms, is still the periodic run and has not been started.

## 2026-10-19 — the agent's pin is checked, and the merge gate stops pretending to be a full test

**The pin is now checked deterministically, before the build.** Two runs were spoiled because the
harness process kept the whole machine in its affinity and used CPU 4 while the cells were timed;
pinning the recorder cannot move it, because affinity is per process and is not inherited backwards.
`require_observer_off_measured_cpus` now walks the ancestor chain and refuses when a process that still
owns a measured L3 CPU was using at least 1% of a core over a half-second sample, naming each
offender with pid, comm, affinity and rate, and printing the `taskset -pc <free> <pid>` that fixes it. A transient wrapper
shell is not an offender; a worker is. The recorder also writes the resulting pin into
`threads.observer`, so the manifest shows what was allowed. Toolkit
`69c1c139ae1f139cd3a3d00d2fee4a2b34eff3b4`, suite 67 tests OK, `requirements.txt` moved to it.

**A merge proves its cells, not the suite.** `docs/merge-procedure.md` now states the division
explicitly: the candidate pair answers "did the change do what it was written for" and "did it break
what it can reach" and "is the machine still the machine", at about an hour; "did anything regress
anywhere else" is answered only by the periodic full declared suite at a frozen candidate, about 3.5
hours for both arms, and that run - not a per-merge argument - is what a suite-wide claim may cite. A
case no candidate targets is left to the periodic run; its last published page stands until then.
The cadence is manual for now: after a batch of merges, or before a page is quoted.

The reason is the distribution: every number carries a between-start spread, so certifying the whole
declared grid per merge would cost hours and still be a distributional claim, while the cells the
merge actually touches are exactly the ones a candidate pair measures cheaply. The checklist also
records the four preconditions in order (agent pinned, measured sets read from the suite, nothing
else running, the pin recorded), so the step that was forgotten is the first one written down.

## 2026-10-19 — the LU floor's unit fixed and adopted; both merge candidates merged

**Measured, adopted, merged.** `0c71ca2` counts the blocked partial-pivoting LU recursion's work
floor in flops instead of elements (a `c64` element is four real MACs, so the floor sat four times
too high for it), rebased onto `b7715c5` and merged into `main` as `2870062`. The pair is
`20261018T225051Z-lu-unit-floor-baseline` (`b7715c5`) against
`20261018T231847Z-lu-unit-floor-candidate` (`2870062`), MKL arm, 396 cells, 4,428 rows each, three
rotated process sets, one lock, nothing else built while timing ran.

The unchanged `faer-upstream` row reproduced: median ratio **0.999717**, two cells outside their own
between-start range and reported as the tail the gate no longer refuses on. `faer-fork` against the
baseline: 30 faster, 6 slower beyond 5%, the rest inside the recorded spread. The target cells are
the `c64 n256` LU/solve group, fork/upstream before -> after: `lu` C 4T **1.260 -> 0.935**, C 8T
**1.260 -> 0.885**, F 8T **1.370 -> 0.952**, `solve` C 4T **1.350 -> 0.935**, C 8T **1.370 -> 0.893**.
The regression the 2026-10-18 page flagged is gone and the fork is 6-11% ahead there. The floor's
own wins are retained: `f64` `lu` at 64/96/128 stays at 0.49-0.75 of upstream. Of the six "slower"
cells, four have essentially identical medians (`lu/1024x64` 8T 0.4846 -> 0.4841 with a ratio of
1.396: a rotation artefact, not a time), and the remaining two are `cholesky`/`gemm` cells this
change cannot reach, at +5-9% on 16-231 us rows.

`git diff 0c71ca2 2870062 -- faer/src faer-traits/src faer-macros/src Cargo.toml Cargo.lock` is
empty and the merge was a fast-forward, so the merged tree is the measured tree and nothing was
measured twice. `cargo test -p t4a-faer --lib` at the branch: 210 passed / 2 ignored, the same count
as the main it was based on. Pin moved to `2870062`; `main` and `origin/main` are both there.

**The correctness fix merged first, without a pair.** `2e221c7` + `af8273f` (empty native extents,
K0/Replace) merged `main` -> `b7715c5` and pushed. It needs no benchmark because its changed code,
`sparse_matmul`'s strict/unit shrink and zero-depth replacement, is reached by dense LDLT and the two
sparse Cholesky update sites, and no declared benchmark row executes those - the `cholesky` family
takes the `is_llt` branch, which is a different call. That is a reachability statement, not a
performance claim. Tests at the branch: 214 passed / 2 ignored.

**Two runs were spoiled before these, by the agent itself.** The harness process (`node`, ~24% of a
core) was landing on CPU 4 with the whole machine in its affinity, so the idle guard rejected
`linalg-small` runs twice; the agent is now pinned to the observer CPUs `0-3,12-15`, which is the
pin `AGENTS.md` says is lost on restart. Both spoiled run directories are retained under
`data/results/.../20261018T215638Z-*` and `20261018T222317Z-*` with their rejected-attempt archives,
and carry no `run.yaml`: they are not evidence and not a baseline.

## 2026-10-19 — the observer pin is recorded; the adopted pairs re-checked

**Provenance instead of an invisible precondition.** An admitted run's manifest said nothing about
the observing side's CPUs: `require_observer_off_measured_cpus` refuses an observer inside the
measured L3 domain, but once it passes there is no trace of what the pin was, and a restarted agent
silently loses its own affinity. The recorder now writes `threads.observer` with `affinity` (the
recording process's `sched_getaffinity`), `free_cpus` (every CPU outside the measured L3 domains)
and the `source` of both, derived from the host topology and hardcoded nowhere. Toolkit
`8fbf10f8bffeba0c9f10f874c3782e27e3586a7c` on `feat/observer-affinity-provenance`, suite 64 tests
OK, `requirements.txt` moved to that revision. The field is optional in the schema, so earlier
manifests remain valid; their observer is not visible in provenance.

Confirmed on this host: CPUs 0-3 (+12-15) share 16 MiB and are the Zen5 cluster; 4-11 (+16-23)
share 8 MiB and are the Zen5c cluster. The declared sets `4` / `4-7` / `4-11` therefore measure on
Zen5c, and the observer belongs on 0-3,12-15. `/proc/interrupts` currently counts zero on 4-11 and
irqbalance is inactive; the guard's 3 s / 5% window cannot see sub-millisecond interference on a
3-350 us cell, which stays a hypothesis about the small-cell tail, not an established cause.

**The two changes adopted through the campaign comparator, re-checked against its own gate**
(`candidate.control_check`, unchanged upstream row, no new measurement):

| adopted change | pair | control gate |
| --- | --- | --- |
| QR block-apply column split `0627828` | `5578ccc` -> `0627828` | **PASS**, 0/96, median 0.9995 |
| Native GEMM flop floor `2000dd9` | `0627828` -> `2000dd9` | **FAIL**, 1/60, median 1.0000 |

The failing cell is `gemm c64 f 64x64x64` 4T in the *unchanged upstream* row: 0.0448 ms
[0.0438, 0.0451] -> 0.0364 (0.81). The ratio range over those 60 control cells is 0.741-1.355, so
the grid's control row is far wider than the gate assumes even though its median is exactly 1.000.
This is the same confound the LU pairs hit, on a change that was adopted anyway; it is not evidence
that the GEMM floor is wrong, and it is not waived by the adoption.

**Consequence for the plan.** The twelve remaining `KEEP` perf rows predate the v2 comparator:
their evidence is isolated drivers (paired, best of 4-21), not a campaign pair. Re-confirming all
of them would be a day of measurement; accepting them as pre-protocol adoptions with an explicit
evidence class costs nothing and claims less. That choice, and the gate redefinition it depends on,
are the two open decisions.

## 2026-10-19 — the 92 slower cells, their cause, and the two corrections they forced

The fork `2240b07` against upstream `0539947` on the MKL arm is 399 faster / 92 slower /
865 inside the band. Reading the 92 by their evidence class, not by their magnitude:

- **23 `householder_apply`**: every one carries `!` (between-start range over 5%), so the
  run's own spread is as large as the difference. No `householder_apply` regression is
  established by this page.
- **30 of the 92** are below 20 us and marked dispatch-dominated: `cholesky` 11 of its
  12, `full_piv_lu` 5 of its 13, and parts of `lu`/`solve` at `n32`. For `cholesky` this
  is decisive: the fork's `cholesky` source is byte-identical to upstream
  (`git diff 0539947 7364be5 -- faer/src/linalg/cholesky` is empty), so its 12 cells
  cannot be a cholesky regression.
- **The clean remainder** is `lu`/`solve` at `c64 n256` (1.26-1.51, both orders, 4T/8T)
  and the small `f64` `lu`/`solve` cells at 1T in C order (1.10-1.17).

**Cause, in the fork's own code.** `28bc13b` floored the blocked LU recursion's panel
solve and trailing product with `A11.nrows() * A11.ncols() > par_threshold` (128*128).
Two things follow. Upstream `0539947` reads `par_threshold` in
`lu/partial_pivoting/factor.rs` at exactly one place, gating the **swap stage**, and
passes the caller's `par` to both products unconditionally -- so the floor is the fork's
own, and it redefined a public knob. And the floor counts **elements** while its
crossover was swept on `f64`, so a `c64` element carries four times the arithmetic and
the floor sat four times too high: at `n256` the `128x128` trailing update stayed
sequential where `f64` threads the same area. That is the sign flip the page shows --
`c64 n256` at 1.26-1.51 slower, `f64` 64/96/128 at 0.49-0.66 faster, same code.

The second mechanism is the adopted GEMM floor (`2000dd9`), which replaced upstream's
`M*N*K >= 4096 * size_of::<T>()` (f64: 32,768) with work in flops
(`M*N*K * {2|8} >= 2 * 48*48*256`, f64: 589,824, eighteen times higher). Between those
sizes the fork is sequential where upstream threaded, which is where the small-shape
4T/8T cells come from; it is also why the same shapes win at 1T and in `gemm`. This is a
trade, not a defect. The third is the row-major QR/LU copy path (`044f947`, `d132011`),
which is why 62 of the 92 are C order and why the clean 1T cells are all C order.

**Decision on the §4 blocker.** Not either offered option: the contract stays, the unit
of the floor changes to flops, which is the principle `2000dd9` already adopted.
`f64` decisions are identical, only the complex crossover moves, and the comparison
saturates so `par_threshold = usize::MAX` still yields a sequential LU -- which is what
made the trailing-only plan a contract change and this one not. Committed as `9dc0f0c`
on `fix/lu-work-floor-units`; gate `210 passed / 2 ignored`, the same count as main.
**Not measured, not adopted**; the pair that would adopt it still has to pass the
unchanged-upstream control gate that refused 1/396 and 12/396 on the `a537746` pairs.
Record: `docs/reviews/20261018-lu-work-floor-units-decision.md`.

**ELF comparisons demoted.** Three performance conclusions had been resting on identical
machine code of saved ELFs (§3's copy aliases, §5's four dense functions, and part of
the ABBA probe). They are static observations, not performance evidence: the copy
diagnostic's own unchanged control had identical ELFs and then left its band by 17%, the
compared binaries come from different build contexts, and four normalized functions do
not constrain a program's schedule. Record: `docs/reviews/20261018-elf-comparison-demotion.md`.

**Open.** The control-gate instability is now the top blocker, not any candidate: two
complete pairs on identical source moved an unchanged upstream cell with median control
ratios of 0.99975 and 0.99899. A fix is a measurement question, and the existing
same-source (rename-only, then swapped) control is the design that can answer it.

## 2026-10-18 — isolated upstream still shifts; no adoption/contract waiver

Diagnostic recording harness311c2fb45b50e14eb09a15fd929eb81bce343341, compiled harness
651807301c4fd0d485a66038346388dffe289073; same A=b7715c53ba75946d68a97d5655f481b198332447 /
B=7f713a1401f18546d1ef0c134df982be80626b2e, upstream0539947ffb757a739d7e703a7d2fa0c792a909c1,
MKL2026.1, toolkit9daa5bc69a13b36c2a31680e20dbb53e2e42c9d5. All96 cells×8 ABBABAAB
blocks/768 records complete,4starts/binary,18verify+72timing+6info, median5/250,
explicit1/4/8T, dynamic same watchedL3/observer/power policies; sources/binaries
unchanged. No other provider operations constructed/executed; setters/loading
remain. Two post-rejects retained12rows each (CPU5cachepeer8%/10%), three admission
delays; all samples and inconclusive data retained. No outer retry or large tier.

Seven/96 isolated descriptive bands fail (2Chol/1GEMM/4tri), not formal adoption.
Upstream c64Ftriangular256/1T medianB/A=.89646 with all4 B starts below A-min-5%;
c64CCholesky32/4T=.93955. Other-provider execution history alone is insufficient;
not a proof of allocation/caller/layout cause or host drift exclusion. Full96-cell
raw/summary and exact576-record canonical references in native-budget directory.
Canonical stage1 STILLREFUSED/NOTADOPTED; no stage2/final/threshold waiver.

Bounded dense code inspection:4matching symbols,2normalizedsame,2changed230/634
lines, sizes3444→3437/3368→3342. RIP references normalized; no per-engine/whole-
program or cause proof. Do not call the equivalent dense refactor code-identical.
Next distinct minimal proposal: preserve dense SOURCE body, floor only structured
entries, require4densebody static equality before any timing. PRE running; not
implemented. Existing all-regression goal remains active, not reduced to96cells.

Driver PRE COMPLETE3minor wording fixed; POST COMPLETE2IMPORTANT fixed (info
capture/status/guards, optimized Python refusal). Real dummy subprocess success/
exit3 evidence and old-O-exits0/new-O-refuses check pass; no reviewer execution or
rereview claimed. Same hostedCI39145818271 rerun allsteps green after OWN reviewed
LUfeature34d7150 made reachable, no validator relaxation or timing repeat.

## 2026-10-18 — canonical native stage1 complete, control-refused, no adoption

A=b7715c53ba75946d68a97d5655f481b198332447 / B=7f713a1401f18546d1ef0c134df982be80626b2e,
upstream0539947ffb757a739d7e703a7d2fa0c792a909c1, compiled harness651807301c4fd0d485a66038346388dffe289073,
toolkit9daa5bc69a13b36c2a31680e20dbb53e2e42c9d5, MKL2026.1. Eight serial existing
recorder runs completed15:28–16:13 UTC; exact648 full identities/statuses/start/
side×3, all3888 timing-CSV rows including skips retained. All verification,
manifest, observed1/4/8 budgets, clean source and binary checks passed. No source
edit/build/checkout overlapped recording. Binary hashes in outcome/README.

Official comparator: Cholesky6/60 controls fail, GEMM1/48 fail, triangular4/48
fail; HH72/72 reproduce. INCONCLUSIVE / NOT ADOPTED, no conditional stage2, no
main merge/final campaign. Cholesky targets descriptively .529–.999, not acceptance.
Accepted A c64C256/8T first5.143332ms versus .822805/.849886 retained; geomean
skew explicitly disclosed. Dense GEMM four geomean>1.05 cells, three medians above
baseline process-max+5%; all data/guards remain. No explanation waives them.

Three rejected attempts retain stdout/CSV/full guard: A-HH verification peer18
10.3% busy, B-tri verification peer21 11.9%, B-HH timing peer16 10.3%. Measured
CPUs4 /4-7 /4-11, watched4-11,16-23, observer0-3,12-15; boost0/performance/EPP/
max2GHz. Normal declared bounded retries only. Initial untimed missing-platform-
crate preflight and locked-cache recovery retained separately, lock unchanged.

`data/probes/20261018-native-gemm-budget/{runtime-stage1,summarize_stage1.py}`
checks full counter equality/starts/raw5 samples/worker/source/binary identity and
retains full228-cell controls and descriptive fork rows. Four SIMD Cholesky leaves
match normalized instructions/sizes but move+16 modulo64; reference targets not
audited, engine attribution/whole-program equivalence/alignment cause not proven.

Reviewed code and refusal documentation archived on OWN fork feature
`perf/native-gemm-work-floor` at5ab54a67d05fcb28c9afa907435844e420e98d6c, remote SHA
verified. Reviewed source hashes still match7f713a1. This makes raw-run source
ancestors reachable; no PR/release/upstream report or library pin change.

Work continues, not a global stop: independent PRE running for bounded upstream-
only96-cell ABBABAAB diagnostic, same immutable binaries, no flags/environment/
worker policy changes. Only ask whether shifts persist without other providers'
operations; no replacement gate, adoption, outlier removal or new LU pair. Full
canonical refused data stay immutable. Other root causes and final full suites
remain outstanding.

## 2026-10-18 — comparator POST complete and reachable dependency delivered

Bounded gpt-6.1-sol POST COMPLETE on toolkit9daa5bc / harnessfca1fd7: all requested
areas, no findings/deletions. Full MD/verbatim JSON archived. Reviewer traced
full4 files, timing validator and scoped existing run validator; no test execution,
raw-log/SHA verification or performance approval. Given LICENSE paths absent, no
independent license-text audit. Worker checkpoint remains explicitly incomplete.

Toolkit9daa5bc69a13b36c2a31680e20dbb53e2e42c9d5 pushed to existing
`fix/rejected-attempt-retention`; remote SHA verified, requirements pin updated.
No release/main merge. Already-passing63 toolkit tests/9 harness tests not rerun
without an affecting code change. Remaining toolkit CI-local declarations,
all-module compile and full-package Ruff pass. Harness declarations, all39
recorded manifests and generated-index check pass; historical declaration-drift
notes still skip declaration-derived checks for old runs, not evidence of correct
Cargo linkage or new performance acceptance. Coherent production diff self-reviewed;
no source correction after independent POST.

Native direct-recorder argv now frozen as4 groups/side/stage, projection matches
existing full648/2688-record grids including retained unsupported cases/variants.
Host topology derived from sysfs; all11 agent threads pinned outside measured L3,
boost0/performance/EPP-performance/max2GHz observed. Native source objects fetched
locally with full SHAs (short-SHA fetch failed, then corrected), no checkout or
build/timing yet. Prepare fresh A=b7715c5/B=7f713a1 with existing full checkout;
no source or harness/tool identity changes between pair parts, no final-only tiers.
No goal completion or candidate adoption; stage1/conditional stage2 and remaining
regressions/final all-suite campaign still required.

## 2026-10-18 — native POST complete, comparator tests recovered and passing

Native candidate7f713a1 bounded gpt-6.1-sol POST is COMPLETE for all requested
areas, no findings/deletions. Source hashes still match; no corrective code or
repeated Cargo tests. Full report archived under `docs/reviews/20261018-native-gemm-budget-post-review.*`.
No performance, worker-utilization, adoption or integration approval; timing still
not begun and final-only square tiers still untouched.

Comparator ordinary worker hit28-turn budget with failing/unfinished tests and
no CLI fixture. Its INCOMPLETE checkpoint is explicit, not review evidence. Main
finished pairing and tests: shared recorded identities include legitimate skips,
nonempty equal timed sets/complete start IDs independent per engine, aliases and
duplicate repeats refuse rather than overwrite. CLI reuses whole-run validator
on BOTH sides, with YAML/CSV/file/output error refusals; explicit subsets remain
legal and fork slowdown stays descriptive. No second schema framework or stats
change. Toolkit code is local9daa5bc; POST/delivery/pin update remain pending.

Toolkit63 tests pass, harness unittest9 pass including6 fully-valid synthetic-v2
CLI tests/temp Git without validator mocks or local faer checkout requirement.
Ruff/compile/diff checks pass. Real5 negatives now refuse; real positives396/198
full/f64 cells pass; original tracked source data/hash unchanged. Offline original
LU pair rechecks still fail exactly1/12 controls over396 cells—no new timings or
third pair. Evidence `data/probes/20261018-candidate-coverage-audit/`; record notes
the additive CSV exception guard was tested after the real5-case after capture.

No goal completion: comparator POST then reachable-source pin, fresh native A/B
builds/recording prerequisites, staged guard/target decisions and remaining root
causes, final freeze and both entire suites/vendor arms are still required.

## 2026-10-18 — complete PREs, isolated shared-floor implementation

Both bounded gpt-6.1-sol PRE reports are complete and archived as Markdown plus
verbatim JSON under `docs/reviews/20261018-{native-gemm-budget,candidate-completeness}-pre-review.*`.
Native plan has no mandatory finding. Comparator PRE correctly identified that
selector existence must include legitimately skipped recorded cells, not timed
series alone; source filter/counterexample confirm it, corrected plan and worker
request require explicit mixed-support lu,gemm regression plus nonexistent-token
refusal. Statistics and adoption criteria do not change. Implementation is split
by disjoint ownership: main Rust worktree, worker comparator/toolkit/CLI tests.

Isolated native candidate7f713a1 (base b7715c5, separately reviewed correctness
repair) reuses one private dtype-MAC floor at all3 native entrances, exact
post-shift dimensions; internal legacy32³ gate remains fallback-only. No LU gate,
copy/HH candidate, new threshold/API/unsafe/scratch/dependency. Initial focused2
pass; after extreme-cutoff algebra extension final library217pass/2existingignore;
no-Rayon std,linalg check passes with13 existing warnings on unchanged declarations.
Kache33,981/33,981 valid. Logs/exact patch/source hashes in
`data/probes/20261018-native-gemm-budget/`. Bounded POST running on immutable
snapshot. No timing, adoption or integration; tests are not performance evidence.

Exact staged inventories exported offline from the original full-MKL census,
checked against independent requested axes, with variants/skips retained: stage1
648 records/start (fork228/upstream228/MKL108 timed,84 vendor skips); conditional
broad stage2 all14 families2688 records/start, including unsupported records.
Raw CSV grids and metadata hashes are frozen before timing; exact invocation and
fresh build/source/thread/topology checks still required. Stage1 only filters;
formal adoption requires complete validated manifests and unchanged controls/
guards. Square2048/3072/4096 stay final-only. Expanded goal remains incomplete.

Historical fact corrections, not new measurements: faer5229e0b and campaign8319a10
recover original phase stdout. Correct baseline/candidate table mixing,4% not11%
512²/1T SVD gain, DBDSQR intermediate/INFO2 fallback, QR calls vs regions and
missing panel count, rectangle MAC/FLOP arithmetic; retract unproven wake-up,
timer-precision and discarded-vector-work causal claims. See
`data/probes/20261018-historical-phase-audit/`; missing historical binary/graph/
pool evidence remains disclosed, not reconstructed by assertion.

## 2026-10-18 — native correctness boundary closed locally, performance work resumes

Branch `fix/native-gemm-empty-extents`, code af8273f/docs b7715c5, separately fixes
native strict/unit post-shift extents and K=0/Replace stored-region initialization.
No new unsafe/API/threshold/scratch. Original negative tests, source identities,
all follow-up logs and final patch are under
`data/probes/20261018-native-gemm-callers/`. Focused4 pass, library214 pass/2 ignore;
std/linalg no-Rayon compilation passes (13 warnings in untouched files). A final
finite-output assertion strengthens the scalar oracle; focused/full checks were
rerun, production unchanged. All five current callers are inclusive lower or
rectangular/Add; this correctness repair does not explain the measured gaps.

Two bounded PREs and a bounded POST on gpt-6.1-sol are COMPLETE, no mandatory
correctness fix. Reports live in `docs/reviews/20261018-native-gemm-empty-*.md`
(POST also saved verbatim as JSON). POST reviewed2e221c7; final production matches,
with only a test-only assertion added. No independent execution, backend audit,
f32/c32/u32/right-conjugation/parallel/forced-generic coverage is claimed.
The six optional redundant old guards stay unchanged to avoid unrelated churn.
The bounded numeric-copy POST also completed with no mandatory correctness fix:
`docs/reviews/20261018-owned-copy-post-review.{md,json}`. Custom zero/Clone/drop
costs remain a limit, not a generic speedup or user-approved tradeoff.

Neither review is performance/merge approval. Both source branches remain local
and unintegrated. HH remains declined with incomplete POST; prior four incomplete
attempts are explicitly recorded, not retroactively counted. Stale original
single-family task entries were retired in favor of the expanded all-regression
task list, not marked complete. LU failed pairs stay refused; no third pair,
threshold waiver, source combination, final freeze or large-tier run. Resume the
three-native-entry budget/caller investigation under the unchanged protocol.

## 2026-10-18 — focused Householder merge held, investigation continues

The fixed-binary stage1 diagnostic forad9f2ef completed all144 timed full
fork/upstream identities and three starts/binary/cell, with all exact-grid,
verification/source/power/thread checks. Full results and the reproducible
analysis live in `data/probes/20261018-householder-partial-merge/` (9044b41).
Tall targets improved, but f64 F64² block/8T deteriorated by12.24% rotation-
geometric, beyond the baseline range expanded by5%. NOT ADOPTED; downstream
QR/SVD/eigh stage2 NOT run. Algorithmically unchanged small paths do not imply
unchanged code generation or worker state. All72 upstream descriptive range
checks are in band; they cannot waive the fork guard failure or make a diagnostic
an adoption comparison.

The repaired wrapper retained a real post-rejected A4T attempt completely:
stdout,60-row CSV and CPU guard log (CPU8/cache peer8% busy). Three pre-execution
idle admissions were also delayed (CPU4,18,19), with details retained. The
predeclared bounded wrapper continued, not a selective manual rerun. Measured
CPUs4/4–7/4–11, watched L3/SMT4–11,16–23, observer0–3,12–15; recorded after dynamic
topology checks. No build, tracked source edit or other owned measurement overlapped recording.

The broader goal is still active: decline/diagnose this source, do not stop all
investigation because one candidate failed. Shared native GEMM has three call
sites (dense, triangular and indexed/diagonal internal); the prior flop floor
only covered the dense one. Caller audit and focused correctness investigation
continue before another implementation. No source combination, candidate freeze
or final all-suite rerun has happened. Large2048/3072/4096 stay final-only.

## 2026-10-18 — upstream cleanup, still in progress

The new objective is all technically removable, reproducible upstream regressions,
not vendor-only gaps. Only focused checks during development; 2048/3072/4096 are
**final-only**, after the combined candidate is frozen. No full-suite rerun yet.

The minimal LU-budget candidate remains branch-only. Both complete pairs failed
unchanged-upstream controls (1/396 then12/396); all four runs are retained under
`data/probes/20261018-upstream-lu-pair-repeat/`. No third identical pair, relaxed
band or discarded slow start, and no adoption from these inconclusive pairs.

A new upstream-only fixed-binary diagnostic uses32–128 at1T/4T, with order
A B B A B A A B, not another candidate pair. In
`data/probes/20261018-upstream-binary-abba/`, f64 Cholesky32 C follows the linked
binary (1T B/A1.078) while c64 follows the opposite direction (0.957). All four
SIMD leaves have the same normalized instructions and+16-byte entry shifts.
Numeric RIP operands match; the partial-load/store table, its544 relocation
entries and helper-code region match. A referenced Equator vtable has shifted
code pointers; its exact pointer comparison is deliberately retained as unequal.
This narrows the possibilities but does not prove alignment causality or whole-
program equivalence. Accepted samples, guard logs, hashes, effective pool census
and bounded static checks are retained. One4T attempt was rejected after running;
legacy pinned.sh overwrote its CSV on retry, so those rejected samples are lost.
The loss is explicit in the probe README; there were no1T guard rejections, but
this is not complete adoption evidence. The corrective wrapper now archives
rejected stdout/declared CSV/CPU guard details before any retry and stops safely
on archive failure. Toolkit8484fb54 supplies each effective CSV explicitly;
requirements.txt pins that remotely reachable feature-branch revision. Bounded
PRE/POST review and verification are recorded in
`docs/reviews/20261018-rejected-attempt-retention-review.md`. Numeric timing,
thresholds, retry budgets and canonical result_files selection are unchanged.

A fixed-reference host-check proposal received a completed bounded PRE-review:
`docs/reviews/20261018-fixed-reference-control-pre-review.md`. It is meaningful
as a diagnostic, not sufficient for target/guard acceptance. The existing gate
is unchanged. Comparison-helper success also requires a separate audit of exact
cell coverage/start counts; an empty/intersected selection is not proof.

Separately, the1T owned-copy probe in `data/probes/20261018-owned-copy/` compares
current canonical `to_owned` with elementwise `cloned`, allocating and destroying
each result inside the clock. Four balanced starts show small C-order copies
32/64/96 are materially slower on the zeros+Option-staged path, while simply
reverting to elementwise copying loses on larger/conflict-prone layouts. This
supports investigating the staging overhead, NOT a shape96 cutoff or blanket
revert. No copy implementation change is adopted from a copy-only probe.

## 2026-10-18 (later) — the declared suite, both arms, at the merged revision

The full declared `faer-kernels` suite was recorded once at `2240b07` (the merge of
the two adopted changes), MKL arm first and OpenBLAS arm second, sequentially, with
the frozen policy (250 ms priming, 5 calls, three/six process rotations for the
three-row/two-row arm, boost off, observer off the measured L3, whole-domain idle
guard). MKL: `20261018T022054Z-final-full-mkl`, 15,156 rows, coverage `full`.
OpenBLAS: `20261018T044649Z-final-full-openblas`, 6,672 rows, the two declared rows
(`faer-fork`, `lapack-openblas`). Nothing was re-measured: the merged revision's
measured-path trees are identical to the candidate revisions.

**The adopted changes, at the merged revision against the `5578ccc` baseline**
(same cells, f64, rotation-averaged ratio below 1 is faster): `qr_factor` 512^2
8T **0.870 / 0.864** (C/F), 256^2 8T **0.894 / 0.889**; `qr` 512^2 8T **0.864 /
0.855**, 256^2 8T **0.854 / 0.834**; 4T 0.96-0.97; 1T 0.99-1.03. `svd_values`
0.995-1.005 and `svd_thin` 0.965-1.012 (it calls the apply to form U/V). The
`faer-upstream` row reproduced the baseline (0.97-1.04). The small `64^2`/`128^2`
`qr_factor` cells move by 3-5% in both directions, which is their own spread (the
page calls them dispatch-dominated).

**Against the two vendors, 512^2 f64 F order, eight threads** (ms, and the page's own
direction):

| family | fork | upstream | MKL | OpenBLAS |
| --- | --- | --- | --- | --- |
| `qr_factor` | 4.52 / 4.49 | 10.00 | 2.20 (2.06x slower) | 6.81 (1.52x faster) |
| `qr` | 8.27 / 8.26 | 14.77 | 10.34 (1.25x faster) | 15.49 (1.88x faster) |
| `svd_thin` | 58.3 / 58.4 | 59.9 | 47.97 (1.21x slower) | 79.7 (1.36x faster) |
| `svd_values` | 48.3 / 48.8 | 48.6 | 22.40 (2.16x slower) | 46.6 (in band) |
| `lu` | 4.14 / 4.12 | 4.76 | 1.23 (3.36x slower) | 1.77 (2.33x slower) |
| `cholesky` | 1.07 / 1.06 | 1.20 | 0.96 (1.11x slower) | 1.21 (1.14x faster) |
| `eigh` | 23.8 / 23.7 | 24.5 | 14.87 (1.60x slower) | 22.66 (1.05x slower) |

The composite `qr` (factor + compact Q, what a caller uses) is now faster than both
vendors at this cell; the bare `qr_factor` is behind MKL's `dgeqrf` (2.06x) and
ahead of OpenBLAS's. `svd_values` is a MKL-only gap now (OpenBLAS at parity): the
bidiagonalization, not the merge, as `faer/docs/worklogs/2026-10-18-svd-values-only.md`
shows. `lu`/`solve` (2.3-3.4x) and `eigh` remain the largest vendor gaps; they are
the declared out-of-scope families of this iteration (guards only).

**Versus upstream over the whole suite** (MKL arm, the only arm with an upstream
row): the fork is faster in 399 cells, slower in 92, and inside the between-start
band in the other 865. Flagged slower cells group as `solve`19,
`householder_apply`23, `lu`22, `full_piv_lu`13, `triangular_solve`3 and
`cholesky`12; none are flagged in `qr_factor`, `qr`, `svd_values`, `svd_thin`,
`gemm` or `eigh`. **Correction:** these counts do not establish when each gap
arose or prove those families were unaffected. Householder itself changed, and
shared GEMM/copy/Householder paths reach other families. A single historical
probe cannot establish pre-existence or unchanged timing for all92 cells.
Later fixed-binary triage shows some gaps on both sides of the new GEMM floor,
not that every earlier change is innocent; see
`data/probes/20261018-upstream-regression-triage/`. The flagged cells remain a
worklist, not proof that every observed difference is an algorithmic regression.
The OpenBLAS arm has no upstream row (its two declared rows are `faer-fork`,
`lapack-openblas`), so its561/305 faster/slower count is against OpenBLAS,
not upstream.

**Vendor gaps that remain and are measured, not explained**: `gemm` at `128^3` and
`192^3` (f64 F 8T: fork 0.113/0.154 ms, MKL 0.032/0.074, OpenBLAS 0.027/0.081) is
2-4x behind both vendors and the same kernel is fastest at `512^3` (1.20 against
1.52 / 1.31) - a mid-size kernel-efficiency question, not the small-shape fixed cost
the flop floor addressed.

**Still not covered**: `faer-kernels-large` (1024/1536 and the adoption tiers). Its
published pages are the invalid 2026-10-15 ones; the 1024 shapes in the invalid
historical page are not evidence and were not re-recorded here.

## 2026-10-18 — a provenance-correct baseline, three bounded investigations, two adopted changes

The four `-timing-v2` pages of 2026-10-15/16 are not evidence about any revision: they
linked the crates.io `t4a-faer 0.24.5` for the `faer-fork` row (see
`reviews/20261016-linked-crate-provenance.md`). The first recording from a pin that
the build script's patch-resolution guard accepts was taken here at fork `5578ccc`
(the merged row-major QR/LU work), upstream `faer-v0.24.4`, MKL 2026.1, f64,
`qr_factor`/`qr`/`svd_values`/`svd_thin` at 64/128/256/512, 1T/4T/8T, C and F
(`data/results/.../20261018T000948Z-baseline-v2-diagnostic-mkl`). Its manifest names
the checkout paths and commits, the MKL version, the boost-off power state, and the
guards. The pin now sits on `2240b07`.

What the baseline says (rotation-averaged ratios, fork/MKL, at 512 and eight threads):
`qr_factor` 0.43 (fork 2.3x slower), `qr` 1.09 (parity), `svd_values` 0.46 (2.2x
slower), `svd_thin` 0.79 (1.27x slower); at one thread the fork is *faster* than MKL
on `qr_factor` (1.68) and roughly at parity on both SVDs. The QR gap is parallel
scaling, not the kernel; the values-only SVD gap is not the bidiagonal-SVD merge.

Three investigations were run in parallel with that recording, each with its own
isolated driver (`faer-benchmark/probes/`, untracked diagnostics linked by path):

- **SVD values-only** (issue item 1). Phase split at `5578ccc`: at 512/8T the
  bidiagonalization is 71% of `svd_values` (34.3 ms of 48.1) and the bidiagonal SVD
  29%. `divide_and_conquer` still builds the discarded singular vectors for a
  values-only call, but routing values-only to the implicit QR iteration at every
  size was 13% slower at 512/8T (the campaign's budget) and is **rejected**; the
  bidiagonalization is the remaining term. Worklog:
  `faer/docs/worklogs/2026-10-18-svd-values-only.md`.
- **QR factor** (item 2). Phase split at `5578ccc`: panel 3-5%, compact-WY upgrade
  10-25%, trailing block apply 60-85%. The apply's column split was capped at four
  tasks with nested inner threads; with the rows not split it now takes the whole
  budget. Campaign `0627828` against `5578ccc`: `qr_factor` 512^2 8T **0.881/0.879**
  (C/F), 256^2 **0.916/0.910**, `qr` 512^2 **0.862/0.856**, 256^2 **0.862/0.849**;
  4T/1T inside noise; `svd_values`/`svd_thin` and the tall `4096x256`/`1024x64`
  guards inside their recorded spread. A second candidate (per-thread compact-WY
  upgrade floor) was faster on the target but moved `householder_apply/1024x64` 8T
  by +18% (three starts, both forms) and is **rejected**; a build with only the
  split change reproduces the baseline on that cell. Worklog:
  `faer/docs/worklogs/2026-10-18-qr-block-apply-splits.md`.
- **Small-matrix GEMM** (item 3). f64 `32^3` measured 2.31 us serial and 12.0 us at
  eight threads; the caller's budget reached `private_gemm_x86`, which threads
  whatever it is handed (the generic `gemm` crate has a work floor, that kernel does
  not). The budget is now handed over only above the `gemm` crate's floor expressed
  in flops. Campaign `2000dd9` against `0627828`: f64 `32^3` **0.233/0.178** C,
  **0.222/0.163** F; f64 `64^3` 0.702-0.869; f64 `128^3` 8T 0.88-0.90; every c64
  cell unchanged. Worklog: `faer/docs/worklogs/2026-10-18-small-gemm-thread-floor.md`.

Both adopted changes are merged at `2240b07`, whose measured paths are identical to
the measured revisions (`git diff 2000dd9 2240b07 -- faer/src faer-traits/src
faer-macros/src Cargo.toml Cargo.lock` is empty), so nothing was measured twice.
`cargo test -p t4a-faer --lib` 210 passed / 2 ignored at the merged revision.

**Not done yet**: the declared full suite has not been re-run on this pin. The
`faer-kernels-large` tiers (1024/1536, 2048/4096) are not covered by the limited
grids above and their published pages are still the invalid 2026-10-15 ones.

## 2026-10-15 — representative v2 controls: scheduled variation, a row-position effect, and the cost floor

Optimization is still frozen. The three bounded POST reviews completed; the two
IMPORTANT core findings (validated `order` cells, >=2 process sets) and the two
recorder findings (own-Git-root toolkit provenance, publication exit propagation)
are fixed, with raw reports archived in `docs/reviews/20261015-timing-v2/`. The
numerical/runtime report found no must-fix defect; its two MINOR diagnostics are
corrected and both vendors rebuilt (`cce14420...` mkl, `d59f5a80...` openblas).
The full union `verify` grid (every declared shape, f64/c64, batch 1, 1/4/8T, both
binaries) passes with 1448 ok / 2336 declared skips / 0 failures per grid and no
budget-readback anomaly; gzipped raw grids are in
`data/probes/20261015-full-numeric-grid/`.

Two 64-process control attempts aborted on idle admission alone (exit 75), because
~3.5% of 3 s guard windows on this host are pushed above the 5% busy threshold by
unrelated activity (operator polling in the first attempt, plus the agent's own
bursts; standalone repros reproduced it). Controls now use the pinned protocol's
own admission retry (8 attempts, 30 s wait) and an observer launched on CPUs 0-3;
retries are admission retries, not favourable-data retries, and every attempt stays
archived. Attempt 3 completed all 64 processes with exit 0: 528 timed rows, 172
cells, 15,480 samples. A pre-implementation design review of the frozen protocol
found the position-balance claim overstated for cells where an engine has no row
and for engine x position interactions; the analysis was reworked (`analysis.md`,
`position-analysis.txt`) and the budget choice is still open.

What the controls measured (baseline-vs-baseline, not the candidate):

- within-process relative sample SD: 1.47% median;
- **50/172 cells exceed a 5% range of their three process medians** (worst 74%,
  `mkl/4T c64 lu 256^2` F-order 1.224/1.208/2.115 ms). This is scheduled
  between-process variation - process state and row order both change - sometimes
  shared across engines, sometimes not (the MKL vendor row barely moves in that
  cell; an OpenBLAS GEMM cell moves only the vendor row). The historical ~1.4x
  shifts are the same phenomenon; the mechanism is still unknown;
- a **row-position effect**: median 1.010 over the 52 cells where every declared
  engine has a row (10 above 1.05, worst 1.486 for `openblas/8T f64 cholesky
  512^2` C-order). In 8 cells an engine has no row at all, the schedule is not a
  Latin square there, and no position effect is identified. The rotation-averaged
  ratio cancels a shared, start-invariant position effect only; 17 comparisons
  differ from the naive median-of-three ratio by more than 5%;
- retained outliers (upstream `qr` 8T max 34.9 ms vs 16.6 ms median) and
  rotation-consistent real differences (F-order `qr_factor` `4096x256` f64 8T is
  2.26x slower upstream than on the fork in all three rotations; MKL GEMM
  `16x240x4096` c64 4T ~10x slower than faer) are kept.

The observing side is now an enforced precondition, not a remembered habit: the
recorder refuses to start when its own affinity overlaps the measured CPU set
(expanded with SMT siblings) or when a declared measured set spans two L3 domains,
and it derives the free CPU set from the machine topology instead of a hardcoded
list - the harness `AGENTS.md`, `docs/timing-policy.md` and
`docs/reliable-measurement.md` state the general rule, and the per-host numbers live
only in the profile. The control driver applies the same check. This followed a
design-review question that exposed the agent process running unpinned while a
measurement was in progress. A six-process diagnostic on one representative cell
(`data/probes/20261015-observer-pin/`) then found the pin does not change the
measured medians, spreads or the rotation-2 shift (fork +23.4% pinned vs +24.1%
unpinned, upstream +37.4% vs +37.2%); its value is removing a guard-rejection
source and an uncontrolled variable, and it is recorded as such rather than as an
accuracy improvement.

Three follow-ups after that review changed the planned budget.

Turbo was disabled on the host (`cpufreq_boost=0`, governor/EPP/PPD `performance`);
the recorder now records the power state in `host.power` so a page carries it. Under
that state the priming calibration (three process starts per setting, 27 processes,
zero guard rejections) shows the setting barely matters: the rotation-paired ratio
of `--prime-ms 250 --reps 10` and `250/30` to `3000/30` is **0.978-1.006** across 18
rows, far below the 0.5-35% between-process range in the same cells. Meanwhile the
worst process-median ranges collapsed: `cholesky` 512^2 f64 8T went from 40-73% to
1.5-5%, `lu` 256^2 c64 4T from 74% to 33% (its faer F-order row keeps a systematic
rotation-2 bump of 2.009/2.017/2.503 ms). Boost was a major but not the only
contributor to the shift phenomenon.

A small-shape probe answers the "is this thread start-up?" question: `gemm` 4^3
never enters a parallel path (0.11 us/call, identical at 1/4/8T), `gemm` 32^3 pays a
roughly constant **+8 to +11 us** for 4T/8T (thread-region start-up, not kernel
work; MKL's step is +0.3-0.8 us), and 256^3 is kernel-dominated (1077 -> 209 us).
Small-regime cells at 4T/8T therefore measure dispatch policy, not kernel scaling,
and stay as coverage with that label rather than as evidence about large shapes.

The L3 rule from the same review was corrected in the other direction too: a measured
set that spans L3 domains is legal (so the hard one-domain refusal is gone and it
warns instead), and the idle guard now watches **every CPU sharing the measured
cores' last-level cache**, not only their SMT siblings - that is what makes a
partially filled domain safe, because unrelated work on the free cores of a touched
domain would otherwise contend for the same cache invisibly. A run that needs every
domain can only proceed with `FBENCH_OBSERVER_SHARES_L3=1`, which records the
residual sharing explicitly.

The contract is now written down in [protocol-v2-baseline.md](protocol-v2-baseline.md):
250 ms priming, 5 calls, three process sets, boost off (enforced), observer off the
measured L3 domains (enforced), whole-L3-domain idle guard, the small-shape grids
trimmed to 8 and 64, the OpenBLAS arm without the duplicated `faer-upstream` row
(the two arms' copies of the same faer code agree to 2% median, 26% worst), and the
candidate mode for optimization iterations: identity + host-block equality, an
in-run `faer-upstream` machine-state control on a predeclared overlap set, then
fork-only measurement of a declared subset with the rotation-averaged ratio against
the recorded baseline fork. Missing coverage is reported as missing, repeated
candidate attempts are both kept, and adoption still needs the numerical grid and a
finished-diff review.

The baseline run is invoked exactly as written in
[protocol-v2-baseline.md](protocol-v2-baseline.md) section 5: MKL arm with three rows
and `--aa 3`, OpenBLAS arm with two rows and `--aa 2` (a complete rotation cycle for
two rows), both at `--prime-ms 250 --reps 5` with `PINNED_RETRIES=8`. The harness
gained a `--rows` filter so an arm does not pay for a row it does not need, and the
recorder passes the declared engines through to it; the census and the page columns
follow that declaration.

The declared square tiers were restructured for optimization work: `small`
32/64, `mid` 128/256/512, `large` 1024, plus adoption-only `linalg-big`/`gemm-huge`
(2048/4096, `--threads 4,8 --dtypes f64`) and the diagnostic `linalg-boundary`
(255/257/511/513). 4/8/16 are out of the campaign; they belong to a separate
small-matrix loop bench, as does a "too many threads on too little work" check. The
recorder gained `--shapes` so the daily loop measures one square per tier while the
run still records the subset it covered. See protocol-v2-baseline.md section 5.

Large tiers are now a separate suite (`faer-kernels-large`) measured at 4T/8T only -
1024 plus the adoption-only 2048/4096 and the diagnostic boundary sizes - because a
run dominated by the biggest shapes is not worth a 1T pass, and a second suite keeps
each page a full cover of its own declaration rather than two partial pages
fighting over one page name. The daily suite keeps small/mid/tall/wide at 1/4/8T.

The tier sizes were aligned with faer's own published benchmark distribution
(`faer-bench`: powers of two plus x1.5 intermediates), so the baseline measures the
sizes the upstream project itself reports: small 32/64/96, mid 128/192/256/384/512/768,
large 1024/1536, adoption-only 2048/3072/4096, plus our block-boundary diagnostics
255/257/511/513. 4/8/16 remain out of the campaign.

`linalg-mid` keeps only 128/256/512 (the 192/384/768 EVD/SVD steps are too
expensive for the baseline; they stay in the gemm tier and in adoption runs),
bringing the all-sizes baseline to roughly 12,000 rows per set per arm pair and an
estimated 5-7 h for both arms. A 150 ms priming check on four representative cells
found no effect where calls are microseconds or tens of milliseconds, but a ~3%
systematic read in the `cholesky` 512^2 f64 8T cell and noisier rows, so 250 ms
stays frozen.

Cost floor at the proposed 3 s priming: 3,948 primed rows per set per vendor (out
of 5,148 base engine-cells; 3,756 successful base cells + 192 Householder variants,
1,392 skips, 5,340 emitted rows) -> 9.87 h per vendor and ~19.7 h for both of
priming alone, before calls, checks, guard overhead and verification. Budget-freeze
options are with the maintainer; the full re-record has not started.

## 2026-10-15 — align timing-v2 contracts before the baseline re-record

Optimization is frozen: candidate `fe41bbdfb88c0528e0726dca3b1764f47222ad56`
is saved separately. Both vendor builds use the same clean baseline paths:
`extern/t4a-faer` at `bfe4041cd9918815e3b3f766ac0332b792badb7d` and
`extern/upstream-faer` at `0539947ffb757a739d7e703a7d2fa0c792a909c1`.
No new optimization or performance conclusion is adopted. Library feature requests
are now read from the harness's dependency declarations (`std,rayon`), not
mislabelled with the harness's vendor-link selector; both remain in build metadata.

The v2 contract is in [reliable-measurement.md](reliable-measurement.md). Composites
retain actual outputs until the outside-clock prepare/check boundary. Factor-only
vendor QR restores preallocated input/tau outside the clock; its check constructs
Q from the actual retained compact factors rather than factoring original input.
Checks for composite QR/LU/Cholesky/SVD/EVD likewise consume their actual results.
Solve still consumes its temporary factors inside the timed pipeline.

Two additional audit defects were reproduced and corrected. Shared residuals used
`f64::max`, which silently ignored NaN and could accept a corrupt result as zero
error; nonfinite and length-mismatched residuals now fail. A missing-execution
regression test failed first at faer factor-only QR through that NaN hole, then
at vendor factor-only QR's replacement factorization. It now passes across the
supported provider/dtype/order/variant routes. The declared Householder sequence
row was absent and QR fixture preparation was inside its block-apply clock;
both block and one-block-sequence rows now exist, label their actual basis/target
widths, and exclude QR preparation from timing. Old pages are not repaired by this.

The process now creates its own N-worker Rayon pool. `info --threads N` and raw
CSV record pool-size/vendor-budget observations, and budgets are checked before
rows. Tests with ambient `RAYON_NUM_THREADS=24` still observed requested 1/4/8
pool sizes for both vendor binaries. These are not traces of active workers.

Observed checks so far: five harness tests pass for MKL and OpenBLAS; both
binaries pass the complete verification grid `8,32,64x8,8x64`, batch `1,2`,
f64/c64, at 1/4/8T. Build/test/thread-info/verify logs are under
`/tmp/baseline-v2-*`. The full numerical grid and full timing suite remain pending.
The store check `kache doctor --verify` passed, with no corruption or repair.

Recording/reporting v2 retains chronological samples, exact verification-cell
census (including variants), observed budgets, binary SHA-256/build metadata and
clean toolkit/harness/library identities. It reports median of process medians
with separate within-process and scheduled between-process SD, not minima or a
confidence interval. V1 reports keep their historical interpretation and the
new timing-version page suffix prevents overwrite. Legacy revision comparison
refuses v2 rather than silently merging variants/taking minima; candidate
comparison remains deferred until the baseline is measured. A no-publish mode
allows recording smoke/control artifacts without publishing numerical pages.

Two initial production-recorder `--no-publish` software smokes (one per vendor,
GEMM-small f64 C/F 1T) passed: each 90 timed rows/180 samples and four first-attempt
idle guards, with identical library/upstream/harness/toolkit paths and revisions.
They deliberately used only 2 calls and 1 ms priming; they are **not** new baseline
performance results or a validated priming policy. Their original manifests/raw
files remain in the `20261015T052425Z-v2-smoke-mkl` and
`20261015T052450Z-v2-smoke-openblas` result directories.

An in-progress scoped GPT audit identified two recording-gate defects: a normal
installed distribution could inherit an enclosing Git repository's toolkit
identity, and publication/index failures could return success. The toolkit now
requires its own Git root, the requirement encodes a pinned editable install, and
both child statuses propagate. Regression tests exercise enclosing/non-Git refusal,
index status 37 through the real publisher/recorder paths, and no-publish status 0.
The corrected toolkit passes 27 tests and Ruff; source checkpoint `ad2a339` remains
local/unpushed. Corrected-source smokes are still required before controls.

Three bounded GPT POST reports then completed (raw text and dispositions archived
in `docs/reviews/20261015-timing-v2/`). The numerical/runtime report found no
must-fix defect; two MINOR diagnostics were corrected locally: failed rows keep
their variant label, and a stale vendor-QR comment was replaced. The core
accounting report reproduced two IMPORTANT findings against the source: a v2
report key could omit validated `order`, and a single-engine one-process set was
accepted even though between-process SD is undefined. Both now fail under
pre-fix behavior and pass after correction; v2 also requires >=2 process sets and
complete engine rotations. The report now also carries the v2 rotation-averaged
engine ratios with a marker for cells whose schedule is not position-balanced and
a per-cell between-start warning; toolkit `7127fa7` is committed and pinned; a post-implementation review of that report change found the first version measured engine-to-engine spread instead of each engine's own between-start spread and omitted the position warning, both fixed and marked `!`/`p` in the page.

The broad delegated implementation and first broad GPT post-review both stopped
at their execution limits and are **not** counted as completed work/review. The
remaining numerical implementation was integrated locally; completed bounded
numerical/core/orchestration reports are still required before adoption. The full
declared-shape numerical grid has been launched separately, without builds or
timings; its final result has not been accepted yet. Representative pilot budgets
are proposed, not frozen suite defaults. At the proposed 3 s x 3 sets, the current
route inventory estimates nearly 20 hours of priming alone for both vendor arms,
before calls/checks/verification/guards. This is a resource estimate, not measured
performance or a reason to choose a more favorable statistic. No full baseline
re-record has run yet.
Older statements below about the ambient pool or hardware mechanism are historical
hypotheses, not established findings for this new harness.

## 2026-10-15 — retain repetitions before interpreting performance differences

Reliability now takes priority over optimization. The earlier cache-conflict claim
below is corrected: the controls do not establish a hardware cause or clear all
suspected regressions. The attempted automatic rotation/three-set-default change
was withdrawn, including all shared campaign-toolkit modifications. Selecting the
best rotated result is not a demonstrated fix.

The common timer now returns all chronological repetitions. `samples_ms` is an
optional extra CSV field, while legacy `total_ms` remains the minimum. Historical
readers/required columns are unchanged. `scripts/summarize_samples.py` reports
per-record mean, median, sample variance, SD and range without pooling process
starts or duplicate cases. Zero repetitions are rejected. Timer sequencing,
CSV ordering, parser rejection, descriptive statistics and duplicate preservation
have runnable checks. The existing reference-matmul F-order test was corrected
to pass F-order buffers rather than reinterpreting C-order bytes.

The source audit also found different faer/vendor seed formulas. The backend
formula is now shared; vendor GEMM arguments, factor-only QR salt and thin-SVD
salt match their faer counterparts. This changes inputs and invalidates old/new
comparisons; no historical numerical table was edited. Thread-budget readback is
available in `info --threads 1`; it describes the vendor's configured maximum,
not measured active-worker usage.

The pilot's exact provenance, schedule, raw calls, guards and descriptive summaries
are under `data/probes/20261015-reliability-gemm/`, not published as a campaign run.
It uses explicit sequential faer execution and a vendor-reported budget of one,
C/F GEMM separately, increased within-row repetitions, counterbalanced priming and
row order, and additional same-setting process starts. Tight within-row dispersion
can coexist with a large shift between positions/occurrences. Longer priming is a
candidate in this pilot, not a universal cure or proof of CPU warm-up. More calls
help reveal late outliers but cannot remove between-process/order bias.

Open boundaries before full re-record: vendor factor-only QR includes caller buffer
preparation inside timing; several composite checks re-run rather than inspect the
timed result; threaded faer labels are not backed by an owned N-worker pool. No full
campaign reliability or no-regression claim is made. Timing defaults and published
result/index files are unchanged.

## 2026-10-15 — the fork-vs-upstream column is not a controlled comparison

**What was asked.** The two recorded pages show `faer-fork` slower than `faer-upstream` in 63 (MKL) and
78 (OpenBLAS) cells out of ~1259, up to 1.5x. The question was whether those are regressions.

**How it was checked.** Three binaries from this harness, measuring the same cells with `--reps 5
--prime-ms 500`, pinned to the page's CPU sets, at the page's revisions (`bfe4041` fork, `0539947`
upstream):

- **real** -- the page's build: row 1 = the fork's code, row 2 = upstream's.
- **null fork** -- upstream's own tree with only the crate rename applied. Both rows use the same
  source algorithm, but separate crate identities can still change code generation and allocation;
  this is a control, not proof that any difference is solely measurement position.
- **swapped** -- the same two trees with the rename reversed, so row 1 = upstream's code and row 2 =
  the fork's. A bias that follows the *position* inverts here; a difference that follows the *code*
  does not.

| cell (C order unless noted) | real | null fork | swapped |
|---|---|---|---|
| `gemm/256x256x256` f64 1T | 1.398 | 1.397 | 1.354 |
| `gemm/256x256x256` f64 1T f | 0.999 | 1.001 | 0.998 |
| `solve/n256` c64 4T | 1.345 | 1.001 | **0.708** |
| `solve/n256` c64 4T f | 1.011 | 1.458 | **0.768** |
| `solve/n256` c64 8T | 1.373 | 0.999 | **0.732** |
| `solve/n256` c64 8T f | 1.446 | 0.992 | 0.973 |
| `lu/n256` c64 4T | 1.188 | 1.010 | **0.813** |
| `lu/n256` c64 4T f | 1.345 | 1.001 | **0.746** |
| `lu/n512` f64 1T (page 1.38x) | 0.998 | 0.985 | 1.007 |
| `qr_factor/64x1024` f64 1T (page 1.37x) | 0.970 | 0.995 | 1.013 |
| `qr_factor/4096x256` f64 1T (page 1.32x) | 0.982 | 0.985 | 0.993 |

**What it says (corrected).** The same-source GEMM control also exhibits a large difference, and
some historical differences did not reproduce. Those observations make the single-scan ratios
insufficient to attribute the sampled differences to fork source changes. They do not rule out
regressions in these or untested cells.

The earlier cache-address-conflict explanation was speculation, not an established mechanism.
Address logging captured repeated calls of one operation, not labeled addresses across all rows.
Subsequent sequential probes also varied with occurrence/order, but order, process history and
elapsed time were not independently controlled. Neither cache conflict nor CPU warm-up is proved;
allocator, code layout, frequency and other state remain possible contributors. In particular the
8T discrepancies cannot be assigned to within-run variance without retained repetitions.

**Next evidence needed.** Retain every timed repetition, distinguish within-process variation from
between-process variation, and counterbalance order and priming conditions in a small same-source
control before making performance claims. Rotation is an experimental factor, not a fairness
certificate. Selecting the best of rotated runs does not establish an unbiased comparison. The
historical pages and the observations above are not rewritten into corrected performance results.

## 2026-10-15 — a rectangular LU cell exists now, in both orders

**What was wrong.** `fbench` gated `lu` and `full_piv_lu` on `square(case, family)`, so every
rectangular cell in `linalg-tall` and `linalg-wide` was recorded as `skipped` with "lu needs a
square matrix; `1024x64` is not". That is not true of either implementation the campaign compares:
`dgetrf` takes any shape and so does faer's `lu_in_place`, both with `min(m, n)` pivots and a
trapezoidal `L` or `U`. The gate was the harness's, and it hid the whole rectangular class from the
family that is the first thing a least-squares or a determinant-shaped workload reaches for.

**What it does now.** Both families take any declared shape. The faer arms time the composite and
check `A ~ L U` through its own `reconstruct`, which handles both trapezoids; the vendor arm times
`getrf` and checks the same identity, rebuilding `L` (unit-lower `m x min(m, n)`) and `U` (upper
`min(m,n) x n`) from the factors and applying the `ipiv` swaps to the input to form `P A`. The
vendor's old square check solved with `getrs` and compared one right-hand side; the new one is the
factorization's own contract at every shape and is strictly stronger at the square ones.

**What it records.** The verify grid passes for both families over `linalg-small`, `linalg-medium`,
`linalg-large`, `linalg-tall` and `linalg-wide` at one and eight threads, both dtypes, both orders
(`full_piv_lu`'s vendor row is `skipped`, as it always was, because LAPACK has no full-pivoting
driver in this harness). A probe (reps 3, `f64`, one thread, C order) puts the new cells where the
recorded square ones were: `lu/1024x64` 0.283 ms and `lu/4096x256` 10.43 against the vendor's
0.421 and 12.68, and at eight threads 0.293 and 6.05 against 0.386 and 3.21 -- ahead at one thread
on every new shape, behind on the two large ones at eight, which is the same parallel-scaling story
the `Thread policy` row carries. The pages will show these cells as measured rather than skipped
when the pin move re-records them.

**What did not change.** `cholesky`, `eigh`, `solve` and `triangular_solve` stay square-gated: the
matrix they factor or solve with is square by definition, and the declaration's blind spots now say
so and say why `lu` is not. No shape was added to any population -- the rectangular shapes were
already declared, so this makes existing cells measurable rather than widening the grid.

## 2026-10-15 — `qr_factor`'s check never saw what its timed call left

**What was wrong.** `Op`'s own contract says the three closures exist so that the timing loop
"clocks only `run` ... and still verify the exact state the timed call left behind", and
`verify` drives `prepare -> run -> check`. Every composite family satisfies it by construction --
its `check` re-runs the same composite on the same restored input, which is the same computation.
`qr_factor` did not: its timed call is the low-level `qr_in_place` on a row-major view, and its
check re-ran the *composite* (`mref(a0, ...).qr()`), which copies into a column-major matrix first.
So the one family whose timed call is a low-level call on the caller's layout was the one family
whose check could not see the layout it produced.

**What it does now.** The check rebuilds `R` from the factor buffer and `Q` from the compact basis
and coefficients the timed call left, exactly as `Qr::compute_thin_Q` does, and checks
`A ~ Q R` and `Q^H Q ~ I` against the reference math as before. No tolerance, case or criterion
changed; the op's scratch requirement gains the sequence apply's temporary
(`StackReq::all_of`).

**Why it matters, demonstrated.** Perturbing the fork's row-major copy-back so that the first
element of each tile keeps the destination's seeded value makes this check report
`MISMATCH faer-fork qr_factor f64 c 1024x64 qr_factor residual 9.99e-1 exceeds
1.51e-5` (and the same at `4096x256`). The previous check passed that build, because the composite
it re-ran never reads the row-major buffer. The perturbation was scratch-only and is not committed.
The clean build reports `all comparisons within tolerance` for `qr_factor` over
`linalg-small`, `linalg-medium`, `linalg-large`, `linalg-tall` and `linalg-wide`, both dtypes,
both orders.

## 2026-10-15 — the eight-thread gap is an exploratory scan, not a pool diagnosis

**What was measured, and how.** A scan, not a recorded page and not a paired comparison: one
`fbench run --n 512 --dtype f64 --reps 3 --prime-ms 500` per thread count, pinned to the row's CPU
set, at `10ee1c9` with the compact-`Q` harness above. `f64`, `n = 512`, F order, ms at 1 / 2 / 4 / 8
threads — `cholesky` faer 1.238 / 0.921 / 0.705 / 0.709 against the vendor's 2.003 / 1.369 / 0.848 /
0.582; `eigh` 29.249 / 20.784 / 16.933 / 15.191 against 36.445 / 21.604 / 12.935 / 9.006;
`svd_values` 39.465 / 31.638 / 28.640 / 30.369 against 28.712 / 20.076 / 15.620 / 13.647.

**Observation, not an inference.** Those point estimates appeared to flatten with the faer
budget and improve further for this vendor. They do not establish stable relative performance,
a physical bottleneck, or which kernels should change; the measurement limitations above apply.

**What was probed, not ruled out.** `Par::rayon(n)` is a decomposition hint, not ownership or
observation of an N-worker pool. CPU affinity constrains the process but does not itself specify
pool size. A further unreplicated scan with `RAYON_NUM_THREADS=8` gave `cholesky` 0.709 -> 0.723,
`eigh` 15.191 -> 15.691, `svd_values` 30.369 -> 29.833, and C order 0.978 -> 1.362. That is not
consistent improvement, but calling all these changes "within noise" without distributions was
unjustified. The experiment did not establish the effective pool size or rule out pool effects.
V2 now owns and checks that budget; it still does not trace simultaneously active workers.

**Decision remains deferred.** Earlier floor scans and their decisions are archived in the
fork's `2026-10-14-evd-svd-floor-sweep.md`, `2026-10-14-bidiag-parallel-hint.md` and
`2026-10-14-compact-factor-region-floor.md`. They are historical evidence, not universal clearance
or a reason to resume optimization before the new baseline. The work-model questions remain open.

## 2026-10-15 — the `qr` row timed the full `Q` and checked the compact one

**What was wrong.** The `qr` family is declared as "the composite returning explicit Q and R", and
the two arms did not build the same Q. The `faer-fork`/`faer-upstream` arm called
`Qr::compute_Q()`, which forms the full `m x m` completion; the `lapack-*` arm called
`orgqr(m, r, r)` with `r = min(m, n)`, which is the compact `m x r` factor, and the run's own check
called `compute_thin_Q()`. On a square or wide shape the Q dimensions coincide; on a tall shape
the faer arm formed an additional unitary completion. These are wall times of different workloads,
not normalized operation-count ratios: the page's C-order `qr/4096x256` reads 490.6519 / 162.0838 / 118.9852 against the
vendor's 57.7822 / 19.5388 / 13.6884 — 8.49x / 8.30x / 8.69x at one, four and eight threads — while
the compact comparison beside it (`qr_factor`, the same page) is 1.71x at one thread and behind only
at four and eight.

**What it does now.** The timed call is `compute_thin_Q()`, the same `m x min(m, n)` factor the check
consumes and the vendor route builds, and the declaration says so. Nothing about the acceptance
criterion changed and no case was dropped.

**What moved.** A probe at fork `10ee1c9` (`run --reps 3 --prime-ms 500`, one thread, `f64`, C order):
`qr/4096x256` 30.24 ms, against the page's 490.6519 for the full-`Q` call and its vendor's 57.7822.
The wide cell `qr/64x1024` is 0.3164 against the page's 0.4998, but that difference is *not* this
change: with `m < n` the compact factor is `m x m`, so `compute_Q` and `compute_thin_Q` build the
same matrix there and the op is unchanged. The verify grid for `qr` over `linalg-tall` still ends
`all comparisons within tolerance`.

**What this costs.** The two recorded pages were taken with the old harness, so their `qr` rows now
describe a different op from the one the harness runs; `qr_factor`, which was always the compact
call, is unaffected. The pages are also stale for a second reason: the `faer-fork` revision has moved
and its `lu`, `full_piv_lu` and `qr` C-order cells change with it (the fork's
`2026-10-15-row-major-adaptation-lu-colpiv.md`). Re-recording is the pin move's job rather than a
separate one, and the index's currency is computed from the pins, so `gen_index.py --check` stays
green until that happens. Do not quote a `qr` row from the 2026-10-14 pages beside a new
`faer-fork` number.

**Checked and not an inconsistency after all.** `validate_run.py` compares a manifest's recorded
`suite_sha256` against the declaration and prints "measured under a different declaration", while
`gen_index.py` still gives both pages `coverage: full`. Those are two different questions, not two
answers to one: `coverage_of` says so in its own docstring -- it relates a run to the declaration
"by content rather than by hash", because "a hash would have called an edited sentence a moved
declaration" -- and the manifest hash is the coarser gate for whether declaration-derived checks
can be run at all. Since neither this entry nor the rectangular-LU change adds a shape to a
population, the content grid is untouched and `full` stays right for the recorded pages.

> Editorial correction: duplicate copies of the two entries above were consolidated.
> The raw historical pages/CSVs are unchanged; the remaining entries retain the
> recorded point estimates without the withdrawn pool diagnosis or mislabelled ratios.

## 2026-10-14 — the campaign's first page is recorded

With the cost and `orgqr` fixes in place the MKL arm's whole declaration built,
verified, timed and published:

- page: `result/zen5-cpu/faer-kernels/bfe4041cd991-0539947ffb75-mkl.md`
- harness `ca0ca08`, fork `bfe4041`, upstream `faer-v0.24.4`, MKL 2026.1, coverage `full`
- raw data: `data/results/zen5-cpu/faer-kernels/20261014T062509Z-first-page-mkl/`

The verify grid — all 10 populations at 1T/4T/8T, no subset — passed
`all comparisons within tolerance` before any timing began. That includes
`linalg-tall`, whose `4096x256` `qr_factor`/`qr` verify is what had blocked the
first attempt, and `linalg-wide`, which the `orgqr` fix unblocked. The 30 pinned
invocations all report a valid idle window before and after; 11 needed a second
attempt (never a third) because a sibling build briefly touched CPUs 4-11. The
configured retry budget was raised to 6 for the run and is recorded in the
manifest (`guards.attempts`).

**How it was run.** From the harness worktree `fix/verify-cost` rather than the
main working tree: `scripts/` there is main's own (the toolkit shims), byte for
byte, but the run needs `fbench/src/{backend,lapack}.rs` from the branch, which
main did not carry at the time. The runner would otherwise have built main's
unfixed harness and hit the same cost again.

## 2026-10-14 — the MKL `qr`/`qr_factor` wide shapes asked `orgqr` for too many columns

With the QR cost fixed, the first full-declaration verify reached `linalg-wide`
and stopped there: the `lapack-mkl` row's `qr_factor` and `qr` returned
`orgqr returned -3` for `64x1024` and `256x4096` (both orders, both dtypes; 16
`MISMATCH` lines). The guard log carried `Intel oneMKL INTERNAL ERROR: Condition
-2 detected in function DORGQR`. The vendor op called `dorgqr(m, n, k, ...)` with
`s.n` the matrix width; when `m < n` that asks Q for more columns than it has
(Q is `m x min(m, n)`) and LAPACK rejects the call. It now asks for
`r = min(m, n)` columns, which is exactly the `m x r` Q the check consumes; the
buffer keeps its `m x n` leading dimension, so tall and square shapes are
unchanged. After: the wide population verifies clean, `all comparisons within
tolerance`.

## 2026-10-14 — `qr_factor`/`qr` verify: the full-`Q` orthonormality check is O(m^3)

**What was observed.** The MKL arm's first full-declaration verify sat on one
`fbench verify --threads 1` invocation for 32 minutes at 99.9% CPU. A per-family
sweep put `qr_factor` past 420 s at `4096x256` and every other family at or below
~12 s.

**What the cost was.** `qr_factor`'s and `qr`'s check formed the *full* `Q` with
`Qr::compute_Q()` (an `m x m` matrix) and ran the reference orthonormality
residual `gram_residual` over it — a naive `Scalar` triple loop, O(m^3). For a
4096x256 case that is a 16.7M-element allocation to form, an O(m^2 size) apply,
and ~6.9e10 scalar multiply-adds to reduce, per engine row and order. The
`reconstruct` check beside it was already cheap because it works from the
compact Householder factors.

**What it does now, and why that is the same acceptance criterion.** The QR
factorization's own contract is `A = Q R` with `Q` an `m x min(m,n)` matrix of
orthonormal columns. Only those `size = min(m, n)` columns multiply `R`; the
remaining columns of the full `Q` are an arbitrary unitary completion that the
exact `reconstruct` check never touches, so checking *their* orthonormality
tests no property of the factorization. The check now builds
`compute_thin_Q()` (m x size, O(m size^2) to form) and reduces its
orthonormality residual in O(m size^2). For square and wide shapes thin `Q` *is*
the full `Q`, so those cases are unchanged; at a tall shape the only thing
dropped is the residual on the arbitrary completion columns. No tolerance was
changed, no case was dropped and no check was skipped.

**Before/after, `verify --threads 1`, CPUs 0-3, fork `22f23cf` + upstream
`faer-v0.24.4`, `batch 1`.** (The vendor row is this host's MKL 2026.1; the same
numbers hold for the OpenBLAS arm because the check is provider-independent.)

| family | shape | dtype | before | after |
|---|---|---|---|---|
| `qr_factor` | 1024x64 | f64 / c64 | 9.54 s / 15.11 s | 0.078 s / 0.156 s |
| `qr` | 1024x64 | f64 / c64 | 8.88 s / 13.12 s | 0.077 s / 0.184 s |
| `qr_factor` | 4096x256 | f64 / c64 | >180 s (killed; >420 s observed) | 3.58 s / 7.73 s |
| `qr` | 4096x256 | f64 / c64 | >180 s (killed) | 4.91 s / 12.05 s |
| `svd_values` | 4096x256 | f64 / c64 | 0.27 s / 0.74 s | unchanged |
| `svd_thin` | 4096x256 | f64 / c64 | 5.29 s / 11.80 s | unchanged |
| `householder_apply` | 4096x256 | f64 / c64 | 0.37 s / 1.18 s | unchanged |
| `lu`, `full_piv_lu`, `cholesky`, `solve`, `triangular_solve`, `eigh` | 1024x64 / 4096x256 | f64 / c64 | ~0.008 s (`skip`, not square) | unchanged |

The whole tall population (`lu,full_piv_lu,cholesky,triangular_solve,solve,
qr_factor,qr,svd_values,svd_thin,eigh,upgrade_factor,householder_apply` over
`1024x64,4096x256`, `f64,c64`, 1T) now takes **48.6 s** and ends
`all comparisons within tolerance`. Before the change it could not finish.

**The check still fires.** To show the cost fix is not a no-op, a scratch build
perturbed the compact-`Q` buffer (`q[0] += 1`) and, separately, the reconstructed
buffer (`recon[0] += 1`) inside `qr_factor`'s check. Both runs printed
`MISMATCH ... qr Q residual 9.6e-1 exceeds 1.024e-7` and
`MISMATCH ... qr_factor residual 1.0e0 exceeds 1.5e-5` on the two faer rows and
exited non-zero; the unperturbed control printed `all comparisons within
tolerance`. The perturbation was scratch-only and is not in the commit.

## 2026-10-14 — the first recording is blocked: `householder_apply` at 8T trips a faer assertion

**What was attempted.** The first page for the MKL arm at fork `9de95f7` (which carries
`tensor4all/faer#9`): `gemm-tall,linalg-medium`, `--threads 1,8`, `f64,c64`, five repetitions,
500 ms of priming, upstream at the pinned tag `faer-v0.24.4`. The recorder built the harness,
took the vendor identity, and ran `verify` first — and **published nothing**, which is the
protocol working as designed.

**What stopped it.** `verify` for `linalg-medium` at 8T aborted in the measured library itself:

```
Assertion failed at faer/src/linalg/matmul/triangular.rs:990
Assertion failed: lhs_ncols == rhs_nrows
- lhs_ncols = 64
- rhs_nrows = 32
```

Bisected over families at `--threads 8 --n 256 --dtype f64`: `lu`, `full_piv_lu`, `cholesky`,
`solve`, `triangular_solve`, `qr_factor`, `qr`, `svd_values`, `svd_thin` and `eigh` each pass,
and so does the whole `factor`/`decompose` group together. The trigger is the `householder`
group: `--family upgrade_factor,householder_apply` reproduces it, `upgrade_factor` alone is
recorded as `skipped` (the backend does not implement it), so the case actually measured is
**`householder_apply` at 8 threads**. Every one of these shapes passes at `--threads 1`.

Because the panic aborts the process before stdout is flushed, the per-case verify log for that
population is empty on disk; the assertion is in the guard log, not the report. That is a
harness wart worth fixing later (flush per line), not the cause.

**What is not yet known, and must be settled before the page exists.** Two explanations fit the
evidence and they have different owners:

- *the harness's fixture* — `householder_apply` builds a basis and its compact-WY factor by
  hand, and faer's threaded path is the one that reaches the shape assertion; if the fixture's
  block layout is not the one the threaded kernels require, the fault is `fbench`'s;
- *the library* — the same assertion, at the same line, fails in an existing fork test on the
  parent revision (`sparse::linalg::qr::tests::test_numeric_simplicial_qr_1_transpose`, verified
  failing on a clean checkout of `22f23cf`), so this assertion is reachable from more than one
  entry point and predates `faer#9`, which only removes regions *below* a floor and leaves the
  threaded shapes unchanged.

Until that is decided no page can be recorded for a population containing `householder_apply`,
and the worklog says so rather than publishing a page that hides which families could not be
verified.

**Tracked as [tensor4all/faer#12](https://github.com/tensor4all/faer/issues/12)**, with the bisect, the
reproducer and the two candidate owners above; the campaign consumes that issue rather than
re-deriving the bisect.

**Reproducer.**

```
fbench verify --threads 8 --n 256 --batch 1 --dtype f64 --family upgrade_factor,householder_apply
```

with the binary built by `scripts/build_for_faer_rev.sh --vendor mkl` at the fork revision
above; the same command at `--threads 1` passes.

## 2026-10-14 — `fbench` lands: `info`, `verify`, `run`, with a measured subset

**What is implemented.** The harness crate `fbench` replaces the placeholder
manifest: `Cargo.toml` (two crates-io faer dependencies named `faer_fork =
{ package = "t4a-faer" }` and `faer_upstream = { package = "faer" }`, plus the
`link-openblas`/`link-mkl` vendor features), `build.rs`, and `src/` split into
`main.rs` (CLI, CSV writer), `shared.rs` (scalar trait, layout-aware reference
math), `op.rs` (prepare/run/check closures), `timing.rs` (wall-clock priming,
best of `reps`), `vendor.rs` (the linked vendor's identity and thread budget),
`backend.rs` (a macro expanded once per faer crate so the same family logic runs
against both, whose `Mat`/`Par`/trait types are not interchangeable), and
`lapack.rs` (the hand-written CBLAS/LAPACKE `extern "C"` bindings).

`fbench info` prints `vendor.name`/`vendor.linkage` and, per arm, the fields the
library reports: OpenBLAS gets `version`, `config`, `corename`, `parallel`,
`procs`; MKL gets only `version` (2026.1), because it reports no kernel and no
processor count. `fbench verify` checks each provider against its own contract
with provider-independent reference math (residuals, orthonormality, the
Frobenius norm identity, no NaN), never one provider against another, and ends
with exactly `all comparisons within tolerance`; every failure prints `MISMATCH`
and exits non-zero. `fbench run` writes the nine-declared-column CSV, one row per
`(case, order, arm)`, with `order` measured as an axis and `k` empty for every
non-`gemm` family.

**Families measured.** The faer arms implement `gemm`, `qr_factor`, `qr`, `lu`,
`cholesky`, `solve`, `triangular_solve`, `svd_values`, `svd_thin`, `eigh`,
`full_piv_lu` and `householder_apply`. The vendor arm implements `gemm`,
`qr_factor`, `qr`, `lu`, `cholesky`, `solve`, `svd_values`, `svd_thin` and `eigh`.

**Deliberately not implemented, and recorded as `skipped`.** `upgrade_factor`
(no faer or LAPACK route written yet: a valid blocked-tau layout has to be built
by hand to measure and check it honestly) and `eig` (the non-symmetric solver is
outside this first version). The vendor arm has no route for
`triangular_solve`, `full_piv_lu` or `householder_apply`. A family a provider
cannot take is written with `status=skipped` and a truthful note; a number is
never fabricated for it.

**Two vendor-arm details this host forced.** (1) `mkl_set_num_threads` and
`mkl_set_num_threads_local` segfault in this MKL 2026.1 install before any
computational call — reproduced with a plain C program linked
the same way — so the budget is set through the Intel OpenMP layer MKL is linked
against (`omp_set_num_threads`) and read back with `mkl_get_max_threads`. (2)
`intel-mkl-src` emits `-L`/`-l` but no rpath, and the linker's `--as-needed`
default drops `mkl_intel_thread`, `mkl_core` and `iomp5` (the harness's own
undefined symbols are all in `mkl_intel_lp64`), so `build.rs` records the
versioned directories as rpath and forces the chain back in after
`--no-as-needed`. Without either the recorder could not execute the binary with
no loader path.

**Acceptance, actually run on this host** (fork `22f23cfd4238`, upstream
`ddbd2ae52808`, `AMD Ryzen AI 9 HX 470`, MKL 2026.1):

1. `scripts/build_for_faer_rev.sh --checkout /home/shinaoka/projects/tensor4all/faer
   --upstream /tmp/faer-rs --vendor mkl --out /tmp/fbench-mkl.env --bin fbench` →
   `built fbench for fork 22f23cfd42383fe8d9d2f5816ea49e045f0d0ef5 and upstream
   ddbd2ae5280825fe97f86735618fed2f6f4eb0b3 (dirty=false/false) in
   .../target/faer-rev/22f23cfd4238-mkl/release`.
2. `fbench info` → `vendor.name=mkl`, `vendor.linkage=shared`,
   `vendor.version=2026.1`, `harness.name=fbench`, `host.logical_cpus=24`.
3. `fbench verify --threads 1 --n 128,256 --batch 1 --dtype f64,c64 --family
   gemm,qr_factor,qr,lu,cholesky,svd_values,svd_thin,eigh,householder_apply,upgrade_factor`
   → 241 lines, 32 `skip`, 0 `MISMATCH`, last line exactly
   `all comparisons within tolerance`, exit 0 (4.5 s).
4. `fbench run --threads 1 --n 128 --batch 1 --dtype f64,c64 --family
   gemm,qr_factor,qr,lu,cholesky,svd_values,eigh --regime medium --reps 3
   --prime-ms 200 --csv /tmp/fbench.csv` → header
   `regime,family,dtype,order,m,n,k,batch,row,threads,total_ms,per_item_us,status,note`;
   e.g. `medium,gemm,f64,c,128,128,128,1,faer-fork,1,0.073217,73.217000,ok,`,
   `medium,qr_factor,f64,c,128,128,,1,lapack-mkl,1,0.140044,140.044000,ok,`,
   `medium,eigh,c64,f,128,128,,1,faer-upstream,1,...,ok,`; all seven families
   present; 84 rows, all `ok`.
5. `python3 -c "import csv;r=list(csv.DictReader(open('/tmp/fbench.csv')));print(len(r))"`
   → `84`; the reader reports exactly the 14 declared columns and no non-numeric
   `total_ms` on any non-`skipped` row. A second run with `--family
   lu,upgrade_factor,svd_values` shows the `skipped` path: `upgrade_factor` is
   written for all three arms with `status=skipped` and an empty `total_ms`.
6. `cargo fmt --all -- --check` inside `fbench/` is clean. From the repository
   root, `cargo clippy -p fbench --features fbench/link-mkl --config
   patch.crates-io.t4a-faer.path=... --config patch.crates-io.faer.path=... -- -D
   warnings` is clean for `fbench` (exit 0). The only warnings in the build are
   pre-existing ones in the two measured crates themselves: `unused import:
   crate::prelude::*` in each `src/io.rs`, and the never-read `re`/`im` fields of
   `ComplexDistribution` in each `src/stats/mod.rs` (2 per crate).
7. End-to-end recorder, exactly as written in the task:
   `python3 scripts/record_run.py zen5-cpu faer-kernels --checkout
   /home/shinaoka/projects/tensor4all/faer --upstream /tmp/faer-rs --vendor mkl
   --regimes medium --threads 1 --families gemm,qr_factor --dtypes f64 --reps 3
   --prime-ms 200` **fails before any measurement**, verbatim:

   ```
   FAIL .../data/results/zen5-cpu/faer-kernels/20261014T022823Z/run.yaml
     schema: $.run_spec.populations: [] is too short

   ERROR: run.yaml does not validate; nothing published
   ```

   The cause is the flag, not the harness: the suite's populations are labelled
   `gemm-medium` and `linalg-medium`, so `--regimes medium` selects no
   population and `record_run.py` records an empty `run_spec`. With the labels
   the suite actually declares, the same pipeline completes:
   `--regimes gemm-medium,linalg-medium --families gemm,qr_factor` → `ok
   .../run.yaml` from `validate_run.py`, `wrote .../report.md from 12 rows`,
   `published result/zen5-cpu/faer-kernels/22f23cfd4238-ddbd2ae52808-mkl.md
   (coverage=partial)`, `wrote result/INDEX.md`. That is a `medium`-regime run
   covering `gemm` and `qr_factor` at 1T; it is a scan, not an A/A comparison.
   Its `data/results/` and `result/` artifacts were **not committed**: this
   repository's CI validates every committed `run.yaml` by resolving its library
   commits in `extern/` checkouts, which CI does not clone, so committing a
   manifest measured from `/home/...` and `/tmp/...` would redden the required
   `index` job. The run is reported here and can be re-recorded on the measuring
   host.

**Known gaps.** `upgrade_factor` and `eig`; a vendor route for
`triangular_solve`/`full_piv_lu`/`householder_apply`; `householder_apply` is a
single timed row (the block apply) with the sequence form verified against it,
not a second timed row, because the CSV has no sub-family column and two rows
under one `(family, shape, arm)` key would be merged by the report; `f32`/`c32`
are not accepted; and the harness commit still has no pin of its own, as the
skeleton entry notes. The OpenBLAS arm was **compile-checked, not measured**:
`cargo check -p fbench --features fbench/link-openblas` (with the two `--config
patch...` arguments, `OPENBLAS_DYNAMIC_ARCH=1`) built `openblas-src` from source
and finished in 2 m 46 s with 0 errors, so the feature compiles, but no OpenBLAS
number was taken and no `openblas` run was recorded.

## 2026-10-14 — the repository is created, skeleton only, no run recorded yet

**Scope.** A campaign for the tensor4all faer fork at faer's own API, on the model
of `tlinalg-benchmark` and `tcontract-benchmark`: the declarations, the two
revision pins, the run manifests, the published reports and a generated staleness
index. This entry records what was adopted, what is deliberately empty, and what
the harness still has to provide. **No number exists yet**: `result/INDEX.md`
lists empty cells, `data/results/` holds no run, and nothing here should be
quoted as a measurement.

**Why a third campaign.** The fork is measured one layer up today:
`tlinalg-benchmark` measures faer through tlinalg's batched wrapper, so its
numbers carry tlinalg's packing, batch loop and lane policy on top of the faer
call. The fork's five KEEP rows in `PATCHSET.md` land at faer's own API, and the
one open `tlinalg-rs#46` item (the tall compact-WY trailing update reaching
3.5-3.9x on eight cores where OpenBLAS reaches 5.5x, a kernel ceiling rather than
a decomposition one) needs a faer-level harness to iterate on. Upstream's own
bench page has vendor arms; what it lacks is campaign discipline.

**Adopted rather than reinvented.** The layout, the schemas, the index/coverage
model and the timing policy come from `tlinalg-benchmark` at `2716cd0` (see
`docs/provenance.md` for the exact files and terms). The timing protocol
`benchmarks/scripts/{pinned.sh,idle_cpus.py}` is the copy from `tlinalg-rs` at
`c5d1d61f`, which expands the idle gate to the SMT siblings of the measured cores.

**The harness lives here, and is not written.** `fbench` is a crate in this
repository, linked to the measured fork and to upstream by path, with a CLI
(`info`, `verify`, `run`) matching `tlbench`'s. The fork's `AGENTS.md` forbids
adding a `benchmarks/` directory to the fork, so the harness owns the timing
protocol as well. `fbench/Cargo.toml` is a placeholder with no `src/`; the build
script, the recorder and the report describe the contract it has to satisfy, and
they fail rather than guess when a field is absent. **No run can be recorded until
that crate exists** — this is stated rather than stubbed.

**Three recorded identities, two pins.** A result depends on the measured fork
(`library`), the upstream release it is compared against (`upstream`), and the
harness (`harness`). The first two are pinned and drive `result/INDEX.md`'s
currency; the harness commit is recorded per run and repeated in every page
header, and it has no pin, so a harness change does not by itself move a cell's
`status`. That is a known gap, written down in `docs/results.md` rather than
implied away. The page identity is
`<t4a-faer12>-<upstream-faer12>[-<vendor>]`.

- `pins/t4a-faer.rev` is the fork revision at repository creation,
  `22f23cfd42383fe8d9d2f5816ea49e045f0d0ef5`.
- `pins/upstream-faer.rev` is `faer-v0.24.4`, determined from the fork (the tag
  is an ancestor of the fork revision and `faer-v0.24.5` is not; the fork's
  `faer/Cargo.toml` is 0.24.4 and `PATCHSET.md` records the published 0.24.4
  release), not assumed from a version string. Pin files therefore allow `#`
  comments, and `gen_index.py` ignores them.

**The declaration.** `benchmarks/suites/faer-kernels.yaml` names the actual faer
entry point each family is, not a family label, and declares ten populations over
the issue's five shape classes (small, medium, large, tall, wide), rather than one
Cartesian product. `gemm` and the linalg families cannot share a population,
because a gemm shape is a triple (`MxKxN`) and a linalg shape is not, so the
regime label carries the family class too (`gemm-small`, `linalg-small`, ...) and
is the population's identity. That is not cosmetic: the harness's raw-data names,
its skip keys and the report's tables are keyed by the regime, and CI asserts the
labels are unique so two populations cannot overwrite each other's CSVs.

**Two deliberate schema extensions.** The suite schema is the sibling's plus two
declared axes this suite needs: `runs[].order` (C/F is part of the cell identity
here, because faer's gemm and Householder paths dispatch differently on layout and
the fork's row-split gates were chosen on F-order shapes), and a shape grammar
that permits `MxKxN` for `gemm` (the tall class's two Householder legs are
`16x4096x240` and `4096x16x240`, which two indices cannot express). The extension
is backwards compatible: every shape the sibling's grammar accepted is still
accepted. The run schema is the sibling's plus the required `upstream` identity
and the `order` axis recorded in `run_spec`.

**`required_profiles` is empty, deliberately.** `missing` is a normal cell state,
and it becomes an error only for a profile a suite lists as required, so
`zen5-cpu` is `optional_profiles` until the first recorded `(zen5-cpu,
faer-kernels)` cell exists — the same sequence `tlinalg-benchmark` followed.
`gen_index.py --check` therefore passes with nothing recorded, and its currency
check reports an absent checkout as a problem only when there is a page whose
currency it would decide.

**CI is content only.** One job, `index`, on the smallest runner: the declarations
against their schemas, every `data/results/*/*/*/run.yaml` against
`validate_run.py` (none yet, so the step passes saying so), and
`gen_index.py --check`. The harness build is not gated here: a run is only
recorded when that same revision already built, verified and timed on the host
that owns the numbers, so a smoke build would be post hoc — the same call
`tlinalg-benchmark` made in its `2026-10-13 (later still)` entry. No measured
checkout is cloned either: `fbench` links both libraries by path, and the index
reports an absent `extern/` checkout only when there is a recorded page whose
currency it would decide, so a repository with no evidence is not failed for the
absence of the trees that would date it.

**Not yet done.**

- The harness crate itself (`fbench`), and therefore every run and every page.
- The direct-faer pooled spread has not been measured; until an A/A pair exists
  here, a table from this campaign is a scan, not a claim.
- The resolved dependency lock (the harness's locked faer revisions and their
  transitive features) is **not** recorded yet. Both faer revisions and the
  harness's feature set are; a lock hash is the remaining piece the issue's page
  identity calls for, and it needs a field the run schema does not have.
- The `1024` shapes, `f32`/`c32`, BLIS, sparse, iterative and GPU arms are later
  declarations, not machinery built now.

### Structured-only native floor, dense source preserved (pre-build freeze)

Distinctcandidate7cd372f/baseb7715c5; library archive520fafe pushed/verified, not
main/adopted. Dense-module suffix exactbaseline, retained3files exactreviewed7f.
Focused2/library217pass2ignored/noRayon13existingwarnings; kache33975valid. Broad
PRE/POST incomplete retained; actual-delta PRE/POST COMPLETE gpt-6.1-sol, no
findings, explicitly inherits prior unchanged3-file POST, not newfulllegacy
audit/execution/performance approval. Source-only report in reviews/native-
structured-floor and data/probes/20261018-native-structured-floor/.

Frozen fresh full-checkout builder argv/two refs, unchanged648/2688fullcensuses,
staticfilter and existing recorder loop committed before build/pair. Staticfilter
QC sameA passes, old7f holds2of4normalizedbodies; only disassembly, not timing.
Actualembeddedpre-filter/hashrefusal checked6normal/optimizedPython cases on
dummybytes, no kernel/build/recording. Ruff/compile/bashsyntax/diff checks pass;
rawlogs/diffs retained (derivedcandidate.diff zero-context, not whitespace-edited).
Newcandidatebuild/staticcomparison/timing pending. Only4densebodiessize+normalized
equality permits unchangedstage1, failureHOLD/no flag/alignmentfiddling. Equality
is notreference/caller/wholeprogram proof/adoption; prior7f refusals and96cell
diagnostic shifts remain. All controls/guards/rotations/skips/full rejected
retention unchanged; stage2 conditional, square2048/3072/4096final-only. No
finalfreeze, fullcampaign, main/pin change or goal completion.

### Structured-only floor static HOLD; distinct same-source compile diagnostic

2920cfb freshA/b771 andB/7cd builder invocations succeeded; actualpaths/budgets/
MKL2026.1 and profile/compiler/config/dependencyfingerprint/flags match. All4dense
functions differ in sizes+normalizedinstructions (1248/1226/1317/1249diff lines),
A2decf425..., B1dd5fbf3...; no timing/fbenchverify/stage1/stage2, no flag/alignment
fiddling/retry. Static/dependency audit is partialcompilerprovenance, not cause/
reference/caller/whole-program equivalence. Code sourcecorrectness review remains
valid, performancecandidateHELD/NOTADOPTED, no main/pin change. Data in native-
structured-floor/{builds,static-dense,build-provenance-audit.json}. Raw retained.

Next predeclared same-source-plan.json is compile-only A=b771/C=af8273; measured
Rust/traits/manifest/lock trees equal and whole diff docs/PATCHSET only. Same
containing harnesscommit/toolkit/builder/profile/flags, serial full-checkouts,
Aexistingtarget/Cabsenttarget: build/cache/path contexts not separately isolated.
Compare all4 bodies once with unchangedanalyzer. No kernel/verify/timing, no
performance/control-contract change; cannot override7cdHOLD or count as LU pair.

Same-sourcecompilecontrol completed: sourceA/b771 andC/af8273 Rust/traits/root+
cratemanifests/lockexact, wholeGitdelta docs/PATCHSET only; fixeddf383e8/tool9daa,
normalbuilderflags, serialwarm/coldtargets. ALL4densebodies/sizes match and ELF
SHA256both2decf425...identical. No kernel/verify/timing/performanceapproval or
override7cdHOLD; singlebuildcontrolnotuniversalreproducibility/causalproof. Raw
same-source-{builds,dense,outcome.json} retained. HostedCI39153055104 allsteps
success/skipped. Distinct direct2-entry expression simplification plan written,
PRErunning; no newcandidatecode untilreview. Denseowningfilewouldremain ENTIRELY
baseline, no helper/pragma/flags/alignmentposition trialloops. All gatesunchanged.

### Direct two-entry candidate source freeze, no timing yet

Codef013756/baseb771, archive98cefa7 pushed/remoteverified after COMPLETEgpt6.1sol
PRE/POST (coherent3files, no mustfix). Fullreports/verbatimJSON retained. Entire
denseowningmod.rs exactbaselineSHA bc37...;2directexistingfloor expressions only,
no helper/attributes/flags/position tuning/newthreshold/API/unsafe/scratch/dep.
Fallbackold32³gateoriginalouterdims preserved. Focusednumeric1/library216pass
2ignored/noRayon13existingwarnings/targetedformat/diffpass; kache33980valid.
No productionhelperdecisiontestexists, numericoraclenotbudget/utilization/f32
structuredgenericcoverage; sourceinspection and oldmathprooflimits explicit.

Exact648/2688fullgrids/argv/tworefs/tool9daa frozen. Existingdriverguard byteexact
prior6normal/-O-checks; newRELonly. Preflight fromsuite+sysfs measured4/4-7/4-11,
watch4-11,16-23,observer0-3,12-15/all11pi1492359threadsoutside/boost0; preparation
notruntimeproof, recorderrechecks. FreshfullcheckoutA/Bbuilder thenALL4densesize/
normalizedbodyequality BEFOREtiming, mismatchHOLD/no tweaking. No old7cdwaiver,
formalcontrols/guards/stage2requirementsunchanged; no finalfreeze/main/pinchange.

Metadata correction from delayedcompletion: oldbroadstructuredPRE stopped at
16-tool-calllimit, nottimeout; humanREADME/libraryworklog corrected, verbatim
checkpointunchanged. ItsPOST did time out150s. BothremainINCOMPLETE/notapproval.

### Directentry fresh static HOLD; explicit development-gate decision pending

Fresh74a1a8b A/b771/B/f013 bothbuilds successful. ALL4dense sizes andnormalized
bodies differ, same1248/1226/1317/1249diff counts as7cd. A2dec..., Bfa3f9f44...;
HOLD/NOTADOPTED/no fbenchverify/timing/stage1/stage2. Sourcearchivee187984 pushed/
remoteverified, codef013unchanged/reviewed/216tests. No retry/flags/alignment/
position tweaks. ENTIREdensefilebaseline didnot retain compiledbody.7cd/f013all4
sizes+normalizedbodies match, butELFimages differ; no reference/wholeprogram or
causal/provider proof. Staticcallform differences retained, not provedmechanism.

Offlineonly diagnostic partitions ofALL96existingUP-onlycells×4starts/binary,
ALL3balanced2vs2partitions×bothdirections retained: Aflagcounts7,7,6,6,4,8;
B7,6,6,8,6,8. TWOstartsperhalf, notformalAA3orindependentreps/pvalues/control
waiver/cause. Noerase7betweenbinaryflags orclaimnoisefullyexplains; originalraw
5samples untouched. Allrows/inputSHA in native-inline-floor/.

Any changefrom extraPRE-TIMINGexact4bodyequality tostaticdiagnostic-record+existing
FULLformalcontrols/GEMMguards needs explicituserdecision andupdatedoperational
plan/reviews/checks beforeexecution. Nothingchangedyet, no permissioninferred.
Originalformalcomparator/targetspread/control/guard/statisticalrules, exactgrids/
skips/starts/samples, largefinal-only, finalfreeze/main/pin remainunchanged.

### Exact builtin-copy source candidate, no performance yet

Separate60de2ee/base7364 (runtime2240), archive26950b6 ownpush/remoteverifiedafter
COMPLETEgpt6.1sol PRE andcoherent2-filePOST/no mustfix. Exact4builtinTypeId via
existingnonstatic helper/no 'static/newunsafe/flags/zero/default/publicbound.
Seededvalue tileemptybeforefirstsrc, fulloriginalOptionfallback preserved for
customClone/drop/heap/reference/alternateComplexImpl. Originalwrapper/callers/
allotherproduction preserved; docsqualifyfallbackvsbuiltin. No Slotframework.

Focused2/library212pass2existingignored/noRayon13existingwarnings/targetednewformat/
diffchecks pass; kache33983valid. Native4dtypecomponentbitsNaNpayload/+/-0/Inf/
bothlayouts/reversedsourceANDdst/parentguards/empty/tails/rectangles/Seqowned4/8
poolcensus (notutilization). RealdispatcherClone-onlyCountedwith BORROWEDatomic
counters/ownedVec exactnclones/noldDrops/postdestroycounts/heapindependence; no
dummycustomComplexField fixture. Suppliedchecks not reviewerexecution.

No publiccloned()/conjugatedroute/allocation-timezero removal orForderLU
schedulingfixclaimed. No d635adoption/controlshiftwaiver. Samefourmatchingdense
sizes/normalizedbodygate appliesbeforeANYperfverify/timingincludingpublicprobe;
mismatchHOLD/no retweaks. Native-floorf013gateunchanged/decisionnotassumed.
Static-only fullcheckoutexistingbuilderMklrelease/tool9daa/argv frozen first.
ParallelDRAFT3624records/start inventory independently all14families/7targetor
conservativecopychainfamiliesALLmainshapes +guards, HHvariants/skips retained;
notoldnarrow2688intersection, notrecordargv/runtimeproof/adoption. No clock
controller infrastructure builtbeforestaticqualifies. Final/main/pin unchanged.
FefarchiveCI39156019409green/fullstepsreceipt retained.

### Builtin-copy qualified static gate; public diagnostic freeze

Source60de2ee/base7364 libraryPRE/POSTcomplete, archive1164bdb. FreshFBEbuilder
499da6e/tool9daa: ALL4densesizes/normalizedbodiesequal0diff, noRIP/caller/whole
programequivalence inferred. UnchangedpublicRustprobe separatelybuiltA/Bfull
checkoutwithactualfaer+traitspackagepathmetadata andELFhashes. OperationalPRE/POST
complete; POSTthreeImportantverified/fixed/maincorrectivecheckspass, notfresh
reviewerofthecorrection. public-config/commands/scripts andrawbuild/static/review
evidencefrozenbeforeanyclock; publicdiagnostic8starts512rows/2560rawsamples,
explicitworker1/Seq, allcontrols/rejections/source/topology/power checks. No
actualtimingyetatfreeze, noadoption/mainpin/contract/f013gatechange. Root499CIrun
39159359591successreceipt saved. Independentreadonlyfullpivotingsourceaudit
showsremainingfactorfilebyteequalUPafterremoving18rowmajoradaptationlines;
ownedhighlevelpathbypassesit/default256andthinupdatesSeqevennominal4/8T.
Sourceboundsnotruntimecounts/causeproof; no newfix ormeasurementfromthataudit.

### Publiccopydiagnostic recorded once, HOLD for unchanged-control shift

SourceA7364/B60de/compiled499da6e/recordingf79812e/tool9daa/upstreamreference053,
no vendorinstandaloneprobe. 8acceptedstarts/512rows/all2560samples, zero rejects;
allruntimeguards/pool1/Seqpassed. data/probes/20261018-native-copy-tile/ hasall
CSV/stdout/stderr/argv/terminal/final/before-after/identity/completion/hashes and
all64cells+48controls+16targets, no quietsubset. Unchangedc64/C64x1024/cloned
controlflag movesfaster; candidateNOTADOPTED, formal3624gridremainsdraft.
ExistingELFonlyalias-completesymbolanalysis findsbothclonedmeasurebodies/four
clonedwrapperssame sizes/normalizedinstructions, no cause/references/caller/
allocatorproof; preliminarycollapsedaliasmapretained+explicitlysuperseded. No
retiming/build/source/placementfitting/controlwaiver. Fullfinalcampaignunbegun.
