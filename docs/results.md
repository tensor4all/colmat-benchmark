# Result layout

`suite_id` identifies the workload; `target_profile` identifies the machine
(`benchmarks/profiles.yaml`). They are two axes of every cell, and the measured
comparison supplies the rest.

```text
data/results/<profile>/<suite>/<timestamp>-<label>/   raw: run.yaml, verify-*.csv, run*.csv, *.guard,
                                                     build-info.json, binary-info.txt, report.md
result/<profile>/<suite>/<fork12>-<upstream12>[-vendor][-dirty][-timing-v<N>].md   one page per run
result/INDEX.md                                 generated table of contents over those pages
```

A result page is one `(suite, hardware profile, fork revision, upstream
revision, vendor arm)`, and it carries all three revisions and the hardware
itself: the fork commit, version, dirty state and features; the upstream commit,
version and the ref it was pinned by; the harness commit; host CPU, logical CPU
count, L3 and arch; the CPU set used at each thread count; the timing policy; the
two faer rows and the vendor library with its kernel; and the guards. Measuring a
new revision -- of either library -- adds a page rather than replacing the
previous one. A dirty checkout gets its own `-dirty` page, because those numbers
are not comparable with a clean build of the same commits.

## Page identity

A result depends on **three** independently moving revisions, and the page name
carries the two that name a *comparison*:

```text
<t4a-faer12>-<upstream-faer12>[-<vendor>]
```

The harness revision is recorded in `run.yaml` and repeated in the page header,
because a page is only attributable to one harness; it is not in the name because
the harness does not change *what* is being compared. Changing upstream at a
fixed fork revision produces a different page instead of overwriting one.

**One vendor per page.** The two vendor arms own the same symbols, so a binary
that links OpenBLAS and a binary that links MKL cannot be the same binary: an
*arm* is a build, and a run is one arm. Two arms at the same revision pair are
therefore two pages, compared side by side, each carrying its own two faer rows
and its own vendor identity. The vendor is in the page name because it is part of
what was measured, and the engine rows are named after the vendor the binary
actually linked -- `lapack-mkl`, not a label chosen when the row was written.

## What a vendor arm needs at build time

An arm is one build into one target directory, and the build environment is part of
the arm's identity because it decides what the vendor library actually is:

- **MKL** is linked from the installed tree, named by versioned path (`MKLROOT`) and
  with its threading runtime named explicitly (`MKL_THREADING_LIB`). A moving
  `latest` symlink records nothing a reader can act on, so the script resolves both
  through `pwd -P` / `readlink -f`.
- **OpenBLAS** is built from source, and its LAPACK is Fortran, so the build needs a
  Fortran compiler. The script takes `gfortran` when it is on `PATH` and otherwise
  the oneAPI Fortran compiler that ships with the installed compiler
  (`/opt/intel/oneapi/compiler/<version>/bin/ifx`, resolved through `readlink -f`),
  which OpenBLAS builds against as `F_COMPILER=INTEL`. Whichever is used is exported
  for the build and recorded in the manifest's build environment as `FC=<path>`,
  because a page that names a vendor library has to say what compiled it. Neither
  present is a loud error rather than a build that silently produces a BLAS-only
  library — the vendor rows need LAPACK. An `ifx`-built OpenBLAS leaves Fortran-runtime
  references in the archive, so the harness binary needs those runtime directories on its
  loader path: the build puts them there for OpenBLAS's own test programs, and the link
  *burns them into the binary* as an `RPATH` (`--disable-new-dtags`, because a `RUNPATH` is
  not consulted for a transitive dependency and `libifcoremt` needs `libsvml`). An arm
  whose binary cannot be executed directly is not recordable — the recorder runs the
  harness itself rather than through a shell that arranges a search path.
- `OPENBLAS_DYNAMIC_ARCH=1` is declared rather than inherited: without it the
  from-source build guesses a target from the build host, and a guessed target is not
  one the run-time CPU is obliged to execute.

The arm's own account of what it linked — version, configuration, threading
implementation, and the kernel its runtime dispatched to where the library reports
one — is what the page prints and what `validate_run.py` refuses a vendor arm without.

## run.yaml

Validated against the run schema shipped by `campaign-toolkit`
(`campaign_toolkit/schemas/benchmark-run.schema.json`) before it is written. It
records:

- the profile, suite, suite file and its hash, timestamp and the exact command;
- `library`: path, commit, dirty state, version and features of the measured
  **fork** checkout, which the harness links by path;
- `upstream`: the **upstream** `faer` revision the comparison is against --
  commit, version, the ref it was pinned by, dirty state and path;
- `harness`: commit and dirty state of this repository. A harness change that can
  move a number invalidates evidence exactly as a library change does, and the
  harness commit is what makes that visible;
- `vendor`: the BLAS/LAPACK library the vendor arm was linked against — name,
  linkage, version, build configuration, the kernel its runtime dispatched to,
  and its threading implementation. A measurement whose vendor cannot be named is
  not evidence, so the validator refuses a manifest that declares a vendor arm
  without one;
- `host`: CPU, logical CPU count, OS, arch, L3 geometry and L3 domains;
- `threads`: the counts, the CPU set used for each, the ambient thread variables
  (removed for the measured process), and how the budget was enforced;
- `timing_policy`: version, priming, statistic, repetitions;
- `providers`: what each engine row was — for `faer-fork` and `faer-upstream` the
  revision itself (restated where a reader looks for a provider's identity), and
  for the vendor arm the library the harness read out of the binary it linked;
- `guards`: the idle window including the SMT-sibling check, the busy threshold,
  the attempt budget, the log files;
- `run_spec`: the resolved populations this run covered — regime, families, shape
  grid, batch grid, the `order` axis, and the cases the harness reported as
  `skipped` — and whether that is the whole declaration. A regime is a
  *population's* identity, unique among the suite's populations: `gemm` and the
  linalg families cannot share a shape grid, so the label carries the family class
  (`gemm-small`, `linalg-small`) and the raw-data file names are keyed by it.

## Shapes and the order axis

A shape is `N` for square, `MxN` for rectangular, or `MxKxN` for `gemm`, which is
a three-index operation and cannot be written as two. The tall `gemm` class
carries both legs of the Householder update that `tlinalg-rs#46` separates: with a
16-wide basis and a 240-column target over 4096 rows, the reduction `tmp = Vᴴ C`
is `16x4096x240` and the write-back `C -= V tmp` is `4096x16x240`. A single leg
beside a vendor `?gemm` is a diagnostic reference, not the equivalent vendor
implementation of the whole update, and the row is labelled accordingly.

`order` (`c` for row-major, `f` for column-major) is a declared axis like dtype
and threads, because faer's gemm and its Householder paths dispatch differently
on layout and the fork's row-split gates were chosen on F-order shapes. The order
is part of the cell identity, so a C-order cell cannot be read as the F-order one,
and the manifest records the actual strides at the call boundary, not only the
fixture's nominal order. The C-order vendor route is resolved per family before
the shim is written, because there is no single answer: a row-major LAPACK driver
is a *different operation* from its transpose, so the arm is either a native
row-major route, a timed adaptation, or an explicitly labelled prepared-layout
comparison.

## Coverage and staleness

> **Read [`docs/dataset-status.md`](dataset-status.md) before quoting a page:** the pages recorded before #49
> carry cells withdrawn pending re-measurement (`lu`, `solve`, `full_piv_lu`, `cholesky`, `qr_factor/n64`).
> The index cannot say so, because it is generated.

`result/INDEX.md` is generated and has two tables: the newest full-coverage page
per cell with its currency, and every published page keyed by revision and
hardware. Its point is that reports are self-describing but the *set* of them was
not: with one tracked file per cell, a cell that has not been re-measured keeps
an old file, and a reader sees a commit hash and a table without being able to
tell whether that cell reflects the current libraries.

`status` is computed against `pins/t4a-faer.rev` **and**
`pins/upstream-faer.rev`, in the two checkouts `extern/t4a-faer` and
`extern/upstream-faer`:

| status | meaning |
|---|---|
| `missing` | no run recorded for this cell |
| `current` | measured at both reference revisions, or at descendants of them |
| `N behind` | N commits touching the suite's `invalidated_by` paths lie between the measured fork revision and its reference, plus the same count for upstream |
| `stale` | as `behind`, but N exceeds the suite's `stale_after` |
| `dirty` | the measured fork or upstream checkout had uncommitted changes; never comparable |
| `diverged` | a measured revision is neither an ancestor nor a descendant of its reference |

`behind` counts *invalidating* commits, not raw distance: a suite declaring only
`faer/src/**` is not out of date because a document changed. A revision *ahead*
of its reference is `current` rather than `diverged`: it already contains
everything the reference names. The counts add, because both libraries are part
of the same comparison and either one moving means the comparison has to be
re-taken.

Currency needs both checkouts. The index reports a missing checkout only when
there is a recorded page whose currency it would decide; with nothing recorded
there is no currency to compute, and failing on it would be noise on a repository
that has not measured anything yet.

The harness is a third identity with no pin of its own: a harness change does not
move a pin, so it does not move a cell's `status`. It is visible in the manifest
and in every page header, and re-measuring the cell is what makes the two
consistent. This is a known gap rather than an implied guarantee.

`coverage` is `full` when the cell was measured over the suite's whole declared
spec *for that vendor arm* and the declaration has not moved since, `partial`
when it covered a subset, and `declaration-changed` when the declaration itself
changed after the measurement, so the two have to be compared by hand. "The whole
declared spec for that arm" is the two faer rows the declaration names plus that
arm's own LAPACK row: vendor arms are mutually exclusive binaries, so no single
run could measure two of them, and measuring the projection is what makes a vendor
cell full instead of partial for ever. `campaign_toolkit/campaign.py` holds that
projection, and the recorder and the validator both call it, so the two cannot
disagree about what coverage means. A partial or declaration-changed page never
displaces a full-coverage one, and `publish_report.py` checks that the page
already there really is full-coverage before it declines to replace it.

## Comparing two revisions

A change is judged from two recorded pages, not from one run read twice:

0. **One arm at a time**, chosen with `--vendor`, because one binary links one
   vendor library. A vendor comparison is two pages at the same revision pair, not
   one page with two tables.

1. A **clean checkout per revision**, so neither is recorded dirty, and no edits
   in either while it is measured. The upstream baseline is the *same* revision on
   both sides, or the ratio measures two changes at once; `compare_runs.py`
   refuses that.

2. The same populations, orders, thread counts and dtypes on both sides, into the
   *same* suite declaration, so the two runs are the same cells:
   `scripts/record_run.py zen5-cpu faer-kernels --checkout <fork> --upstream
   <upstream> --vendor <arm>`.

3. **At least two complete sets** (`--aa 2`) for anything that will be claimed as
   a performance change, because this campaign's noise floor has to be measured on
   this host before a single set is enough to make a claim.

4. `scripts/compare_runs.py <before-run-dir> <after-run-dir>`, which reports
   per-cell ratios, the per-family medians and the A/A spread, and says plainly
   when it is a scan rather than a paired comparison.

5. Both runs are recorded as pages even when one is a regression: a page that only
   shows the improvement is a page that hides the rest of the grid.

## What the harness has to produce

The scripts cannot invent a measurement. `fbench` is expected to provide:

- `fbench info`, one `key=value` per line, with at least `vendor.name` and
  `vendor.linkage`, and `vendor.version`, `vendor.config`, `vendor.corename`,
  `vendor.parallel`, `vendor.procs` when the vendor reports them;
- `fbench verify --threads --n --batch --dtype --family`, printing
  `all comparisons within tolerance` for a clean grid and `MISMATCH` otherwise;
- `fbench run --threads --n --batch --dtype --family --regime --reps --prime-ms
  --csv`, writing a CSV per call with the columns `regime`, `family`, `dtype`,
  `order`, `m`, `n`, `k` (empty for a non-`gemm` family), `batch`, `row`,
  `threads`, `total_ms`, `per_item_us`, `status`, `note`, and recording a family
  whose declared shape the arm cannot take as `status=skipped` rather than
  omitting it.

The recorder and the report fail rather than guess when one of these is absent.
