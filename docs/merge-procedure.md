# Merging a candidate into `colmat` main

The procedure behind criterion 1-3 of `protocol-v2-baseline.md`, written down so a merge does not
depend on what the last session remembered. `AGENTS.md` links here; the protocol owns the rules and
this file is the order of operations.

## What a merge means

A merge moves `colmat` `main` to a revision that has already been measured at *that* revision. The
full declared suite is not part of a merge; it is recorded once at the final frozen candidate and
published afterwards. A merge is not an adoption claim about cells nobody measured.

## Preconditions

1. **The branch is a descendant of `main`** - `git rev-list --count <branch>..main` is 0. If it is
   not, rebase first; never merge a branch whose base predates `main`.
2. **The correctness gate passes** - `cargo test -p colmat --lib` in the branch checkout, with the
   same pass/ignore count as `main` plus the tests the change adds. The change's own focused tests
   are part of this gate, not a substitute for it.
3. **The measured paths are the merged paths** - `scripts/candidate_compare.py --merged-commit
   <rev> --checkout <fork checkout>` compares the tree hashes of `faer/src`, `faer-traits/src`,
   `faer-macro/src` and the two `Cargo.toml`s plus `Cargo.lock`. A merge, rebase or rename that
   leaves them alone keeps the measurement valid; one that changes them invalidates the cells that
   depend on it, and those are re-measured rather than assumed.
4. **A correctness-only change** whose changed code is on no declared benchmark path needs no pair.
   Say which paths it reaches and why no declared cell executes them - `spicy_matmul` reaching only
   dense LDLT and sparse Cholesky is the worked example - and merge on the tests. That is a
   statement about reachability, not a performance claim, and it is recorded as such.

## Before the pair: the observer checklist

The recorder refuses to start when the observing side is on a measured core, but that refusal has to
see the whole chain: pinning the recorder does nothing for the agent that launched it, because the
affinity is per process and is not inherited backwards. Two runs of this campaign were spoiled by
exactly that - the harness process kept the whole machine in its affinity and used CPU 4 while the
cells were timed. So the checklist is short and every line is checked by the recorder itself:

1. **The agent is pinned off the measured domain.** `require_observer_off_measured_cpus` walks the
   ancestor chain and refuses when a process that still owns a measured CPU was using at least 1% of a core during a
   half-second sample, naming the pid, the comm, the affinity and the rate, and printing the `taskset
   -pc <free> <pid>` that fixes it. Do that before anything else, and redo it after a restart.
2. **The measured sets come from the suite**, not from memory: `benchmarks/suites/<suite>.yaml`
   declares them per thread count and the profile names the L3 domains.
3. **Nothing else is running.** The guard checks the whole domain three seconds before and after
   every cell, and a rejected attempt is archived, not published - but a rejection costs a rebuild
   and a retry, so the cheap checks come first.
4. **The recorder records the pin** in `threads.observer`, so the manifest shows what the check
   allowed instead of leaving it to the operator's memory.

## The pair

The target and guard grids are declared **before** the runs, from the changed paths: the affected
families at every declared shape, dtype and order at 4T and 8T, the target cells at 1T, and two or
three collateral cells of families the change cannot reach.

1. Build and verify before timing; the recorder holds one campaign-level lock across build, verify
   and timing, so nothing else builds or measures meanwhile.
2. Record the baseline at `main` and the candidate at the branch head, interleaved in time, with the
   frozen policy (250 ms priming, 5 calls, three complete process sets, boost off, observer off the
   measured L3 domain, the whole-domain idle guard).
3. `scripts/candidate_compare.py --baseline <run> --candidate <run> --merged-commit <rev>`. The
   control decision is the unchanged `faer` row's **median** within 2%; the per-cell tail
   is printed and retained but does not refuse, because the unchanged row moves with the linked
   binary and a cell-by-cell rule refuses even two runs of one revision.
4. A refused comparison means no merge. It is not retried with a relaxed band or a trimmed start.

## The merge and the record

- Merge the measured revision (fast-forward when the branch is a descendant, so the measured tree is
  the merged tree), then confirm the tree hashes again.
- Record in `PATCHSET.md` and the campaign worklog: the measured commit, the tree hashes, the target
  and guard results, the run directories that carry them, and the date. An entry without its run
  directory is not a record.
- Move `pins/colmat.rev` to the measured revision when the candidate is adopted: the pin names the
  revision the campaign materialises and the pages quote, so it follows the adoption, not `main`.
  It moved to `2870062` when `0c71ca2` was merged. Re-recording the declared suite at that revision
  is the publication step that follows a merge, and it is a separate, deliberately started run.
- Rejected and HOLD branches stay where they are with their records. A merge does not delete the
  branches that lost, and a HOLD candidate is not revived by a neighbouring merge.

## What a merge does and does not prove

A merge proves the cells it measured: the declared target set improved, the declared guard set did
not regress beyond the recorded spread, and the unchanged row's median reproduced the baseline. It
does **not** prove that no case anywhere regressed, and it is not supposed to. Every number this
campaign produces carries a distribution - the between-start spread is part of the page - and
certifying the whole declared grid before each merge would cost hours per candidate and still be a
distributional claim.

The division of labour is therefore:

| question | answered by | cost |
| --- | --- | --- |
| did this change do what it was written for? | the candidate pair on the target cells | ~1 h |
| did it break the families it can reach? | the declared guard set in the same pair | included |
| is the machine still the machine? | the unchanged row's median, in the same pair | included |
| did anything regress anywhere else? | the **publication run**: the full declared small suite, then the 1024 tier, at one revision | ~11 h both arms |

So a case that no candidate targets is left to the publication run: it is not re-measured per merge,
and its last published page stands until the publication replaces it. The cadence is manual - run it
after a batch of merges, or when a page is about to be quoted - and its result, not a per-merge
argument, is what a "no regression across the suite" claim may cite.

The publication is the only run that moves the pin. It records the small suite (the gate's own grid,
1T/4T/8T, 14 families) and then the 1024 tier (4T/8T, 10 families plus `gemm`, where the working set
leaves the cache), back to back at the same revision. The 1024 tier is *also* required for a candidate
whose diff touches `matmul` kernels/packing/blocking, the panel or blocking code of `lu`/`qr`/`cholesky`,
or the way work is split across threads - those are the paths whose behaviour only shows at scale.
Anything above 1024 is on demand and its page says so; see `docs/timing-policy.md`.

## Current candidates

| candidate | kind | state |
| --- | --- | --- |
| `fix/native-gemm-empty-extents` (`2e221c7`, `af8273f`) | correctness, no declared benchmark path | tests pass; merge on precondition 4 |
| `fix/lu-work-floor-units` (`9dc0f0c`) | performance, `lu`/`solve` at 4T and 8T | needs its pair |
