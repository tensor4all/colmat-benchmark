# `colmat-benchmark`

> ## 📈 [Results, charts and trends](https://tensor4all.org/colmat-benchmark/)
>
> The site shows the pages at the pinned revisions, one section per hardware profile, with the
> throughput charts (one per family, 1T / 4T / 8T panels, between-start error bars, the plotted
> numbers beside every chart), the joined four-way view, and the gemm / lu n = 1024 trend across
> pins. It is built from `data/` on every deploy; nothing on it is committed. The current pages are
> listed in [`result/INDEX.md`](result/INDEX.md), older ones in [`result/HISTORY.md`](result/HISTORY.md).

The benchmark campaign for the tensor4all [faer](https://github.com/tensor4all/faer)
fork: results keyed by **library commit × hardware profile**, measured at faer's
own API, with the staleness of every cell visible rather than implied. It is the
same shape as [`tlinalg-benchmark`](https://github.com/tensor4all/tlinalg-benchmark)
and [`tcontract-benchmark`](https://github.com/tensor4all/tcontract-benchmark),
which own the same job one layer up.

| | |
|---|---|
| Latest results | [`result/INDEX.md`](result/INDEX.md) — generated table of contents over the per-revision pages; read [`docs/dataset-status.md`](docs/dataset-status.md) first: pages recorded before #49 carry withdrawn cells |
| Result layout | [`docs/results.md`](docs/results.md) |
| Timing policy | [`docs/timing-policy.md`](docs/timing-policy.md) |
| Measurement audit | [`docs/measurement-audit-2026-11-06.md`](docs/measurement-audit-2026-11-06.md) — what each timed call includes, per family and arm |
| Reference revisions | [`pins/t4a-faer.rev`](pins/t4a-faer.rev) (the fork) and [`pins/upstream-faer.rev`](pins/upstream-faer.rev) (the baseline) — what `status` is measured against |
| Rules and runbook | [`docs/rules.md`](docs/rules.md) - acceptance in an hour, the frozen contract, and the commands |
| Design record | [`docs/worklog.md`](docs/worklog.md) — decisions, fixed failures, open items |
| Provenance | [`docs/provenance.md`](docs/provenance.md) — what was copied from which sibling and under what terms |

## Status

> **Provisional dataset:** this development host records provisional pages (`host.role: development`);
> a dedicated measurement host will re-record the suite periodically and replace them - see
> [`docs/provisional-and-clean-datasets.md`](docs/provisional-and-clean-datasets.md). Acceptance runs are
> separate, local and never published.

> **Provenance note (2026-10-16):** the runs recorded on 2026-10-15/16 linked the crates.io release
> `t4a-faer 0.24.5` for the fork row, not the pinned checkout — see
> [`docs/reviews/20261016-linked-crate-provenance.md`](docs/reviews/20261016-linked-crate-provenance.md).
> The build now refuses such a mismatch instead of warning.

`fbench` exists and records runs: the timing-v2 baseline is in
[`result/INDEX.md`](result/INDEX.md) (daily suite, both vendor arms), the harness
owns the workloads and the numerical checks, and the timing protocol lives in
`benchmarks/scripts/`. The frozen settings, the enforced machine preconditions and
the candidate-mode procedure for optimization iterations are in
[`docs/protocol-v2-baseline.md`](docs/protocol-v2-baseline.md); the per-run audit of
the baseline is in [`docs/reviews/`](docs/reviews/).

## From a page to its raw data

Every published number comes from one recorded run directory; nothing is copied by
hand, and the mapping is the same for every page.

| artifact | where | what it is |
|---|---|---|
| raw run | `data/results/<profile>/<suite>/<timestamp>-<label>/` | one recorded run |
| `run.yaml` | in that directory | identities (library, upstream, harness, toolkit), host and `host.power`, CPU sets, timing policy, guard policy, and the exact result/verification file names |
| `run<N>-<regime>-<T>t.csv` | in that directory | one CSV per complete process set: every row of a run recorded with this harness revision or later carries its hardware profile, then its chronological `samples_ms`, the observed thread budgets, status and note |
| `verify-<regime>-<T>t.csv` | in that directory | the verification census the timing was gated on, including declared skips |
| `*.guard` | in that directory | that process's idle-guard log, including discarded attempts |
| `build-info.json` / `binary-info.txt` | in that directory | build metadata (pins, rustc/cargo, features) and the vendor/thread identity the binary reported |
| `report.md` | in that directory | the run's local rendering of the same numbers |
| page | `result/<profile>/<suite>/<fork12>-<upstream12>[-<vendor>][-timing-v<N>].md` | the published numbers, generated from those CSVs |
| index | `result/INDEX.md` | generated: which pages exist, and whether a page is `current`, `partial` or `declaration-changed` |

`-timing-v2` marks a page recorded under the timing-v2 contract
([`docs/reliable-measurement.md`](docs/reliable-measurement.md)); historical v1
pages keep their original names and semantics. Spread is judged from the raw data:
within-process from each row's `samples_ms`, between process starts from the
`run0`/`run1`/`run2` files of the same cell, and admission history from the guard
logs.

## How it works

The harness lives **here**, not in the fork: faer has no `benchmarks/` directory
and the fork's `AGENTS.md` forbids adding one, because campaign infrastructure
would be a rebase cost. So `fbench` links the measured library *by path* from a
checkout, and it owns the timing protocol
(`benchmarks/scripts/{pinned.sh,idle_cpus.py}`) as well.

A result therefore depends on **three** independently moving revisions, and all
three are recorded in `run.yaml`:

| identity | what it is | pinned by |
|---|---|---|
| `library` | the measured fork checkout (package and library `colmat`) | `pins/t4a-faer.rev` |
| `upstream` | the upstream `faer` release the fork is based on | `pins/upstream-faer.rev` |
| `harness` | this repository, whose commit and dirty state can move a number | per run (`harness.commit`) |

```bash
# one vendor arm, build + verify + measure + validate + publish + reindex.
# One binary links one vendor library, so an arm is a run of its own.
python3 scripts/record_run.py zen5-cpu faer-kernels --vendor openblas \
    --checkout /path/to/t4a-faer --upstream /path/to/upstream-faer
python3 scripts/record_run.py zen5-cpu faer-kernels --vendor mkl \
    --checkout /path/to/t4a-faer --upstream /path/to/upstream-faer
```

Four arms are compared at the library's own API: `colmat` (CSV engine `faer-fork`),
upstream `faer` (`faer-upstream`), OpenBLAS and Intel oneAPI MKL; the AOCL arm is described below. The Rust
libraries are separate dependencies named `colmat` and `faer_upstream`; the
upstream dependency selects package `faer`. Each vendor arm has its own binary.
Historical pin paths (`pins/t4a-faer.rev`, `extern/t4a-faer`) and CSV engine names
are retained so existing recordings and tooling keep their identities.

For library-only diagnostics on a host without a vendor toolchain, the build
script also accepts `--vendor none`. This links no vendor; use `--rows faer-fork`
when running `fbench` for a colmat-only measurement. This diagnostic build option
is separate from the recorder's vendor arms and does not publish a dataset.

### The AOCL arm (`lapack-aocl`)

A third vendor arm runs the same LAPACK row body against AMD AOCL: LAPACKE from
AOCL-LAPACK (`libflame`), CBLAS from the threaded AOCL-BLAS (`libblis-mt`), plus
`libaoclutils`, which libflame needs for its CPU-feature dispatch. It is its own
binary like the other two, recorded by `record_run.py --vendor aocl` with engine `lapack-aocl`
(feature `link-aocl`) on AMD profiles; its smoke cell is
[`docs/aocl-smoke-20261119.md`](docs/aocl-smoke-20261119.md). What a reader needs to
reproduce a number from it:

- **`AOCL_ROOT`**: the versioned threaded install tree, e.g.
  `/opt/AMD/aocl/6.0.0/gcc/MT` (AOCL 6.x installs `ST` and `MT` trees side by
  side) or `/opt/AMD/aocl/5.2.0/gcc` (5.x). Install with
  `./install.sh -t <prefix> -l blis libflame utils -i lp64`; without `utils`
  libflame cannot load. `scripts/build_for_faer_rev.sh --vendor aocl` resolves
  the newest complete tree under `/opt/AMD/aocl`, then under the per-user
  `$HOME/opt/AMD/aocl` (an install without root), and records the resolved path
  in `BUILD_ENV`; set `AOCL_ROOT` to choose one explicitly.
- **No loader path is needed at run time.** The recorder executes the harness
  directly, so the binary carries the tree's `lib_LP64` as an inherited
  `DT_RPATH` (`--disable-new-dtags`): a `DT_RUNPATH` would not cover libflame's
  own `NEEDED libaoclutils.so`. An inherited `LD_LIBRARY_PATH` is still searched
  first, so the harness records the files the loader actually mapped.
- **The AOCL tree must not be registered with `ldconfig`.** Otherwise AMD's
  `libblas`/`liblapack` stand in for the system ones in every process on the
  host, and no other arm on that machine measures the library it names.
- **Thread budget**: one AOCL call is governed by three per-thread counts -
  OpenMP's, BLIS's (`BLIS_NUM_THREADS`) and libflame's (`FLA_NUM_THREADS`), the
  last two each overriding the OpenMP one. The harness sets the budget through
  OpenMP, as on the MKL arm, and refuses the cell (before each row and again after
  its timed loop) unless all three read back equal to it; `fbench info` prints
  them as `threads.vendor_detail=omp:N blis:N flame:N`. The recorder removes
  `BLIS_NUM_THREADS` and `FLA_NUM_THREADS` from the measured environment with the
  other vendor variables (`benchmarks/campaign.yaml` `vendor_env`), and a stray
  value that still arrives is refused by the read-back.
- **Identity**: `fbench info` records both libraries' version strings, the files
  the loader actually mapped, and the dispatched kernels (`blis=zen5,
  flame=avx512` on Zen 5). The fork, upstream and harness revisions come from the
  build script's `--out` file, as for the other arms.

A t4a-faer 0.31 checkout must live outside this repository's tree: nested under
`extern/`, Cargo cannot resolve its workspace-inherited package fields.

## Tooling

The seven `scripts/*.py` entry points are thin shims over
[`campaign-toolkit`](https://github.com/tensor4all/campaign-toolkit), the shared
package the campaigns now have in common. `requirements.txt` pins it by git
revision; `benchmarks/campaign.yaml` is this campaign's descriptor, naming the
measured fork, the upstream baseline, the three vendor arms, the `order` case axis
and every other value the toolkit needs. The `schemas/` copies are gone: the
toolkit supplies them.

`scripts/setup_extern_deps.sh` materialises `extern/t4a-faer` and
`extern/upstream-faer` from the pins, which is what `gen_index.py --check` needs to
say whether a recorded page is current.

For local development, install the toolkit from a checkout instead of from git:

```bash
pip install -e ../campaign-toolkit
```

## What this repository is not

It is not the fork's per-PR gate; the fork's CI is a build, and this repository
owns the campaign. It is not tlinalg or tenferro marshalling, batch loops, lane
policy, tensor construction or dtype dispatch — those are `tlinalg-benchmark`'s
and `tenferro-benchmark`'s rows. It is not upstream's website bench page: no
nalgebra/eigen arms, no `fx128`. No API change to faer follows from it.
