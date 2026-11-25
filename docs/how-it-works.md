# How the campaign works

Moved here from the README (#129), which now only points at the site and at `docs/`. The landing
page's own "How the numbers are made" section states how the published shares are computed; this
page is how a run is recorded and how a page maps to its raw data.

The harness lives **here**, not in `colmat`: colmat has no `benchmarks/` directory
and its `AGENTS.md` forbids adding one, because campaign infrastructure would be
a rebase cost. So `fbench` links the measured library *by path* from a
checkout, and it owns the timing protocol
(`benchmarks/scripts/{pinned.sh,idle_cpus.py}`) as well.

A result therefore depends on **three** independently moving revisions, and all
three are recorded in `run.yaml`:

| identity | what it is | pinned by |
|---|---|---|
| `library` | the measured `colmat` checkout (package and library `colmat`) | `pins/colmat.rev` |
| `upstream` | the upstream `faer` release `colmat` is based on | `pins/upstream-faer.rev` |
| `harness` | this repository, whose commit and dirty state can move a number | per run (`harness.commit`) |

```bash
# one vendor arm, build + verify + measure + validate + publish + reindex.
# One binary links one vendor library, so an arm is a run of its own.
python3 scripts/record_run.py zen5-cpu colmat-kernels --vendor openblas \
    --checkout /path/to/colmat-rs --upstream /path/to/upstream-faer
python3 scripts/record_run.py zen5-cpu colmat-kernels --vendor mkl \
    --checkout /path/to/colmat-rs --upstream /path/to/upstream-faer
```

Four arms are compared at the library's own API: `colmat` (CSV engine `colmat`),
upstream `faer` (`faer`), OpenBLAS and Intel oneAPI MKL; the AOCL arm is described below. The Rust
libraries are separate dependencies named `colmat` and `faer_upstream`; the
upstream dependency selects package `faer`. Each vendor arm has its own binary.
Runs recorded before #113 used the older names (suites `faer-kernels*`, engines `faer-fork` /
`faer-upstream`, `pins/t4a-faer.rev`, `extern/t4a-faer`). They are not rewritten: the
descriptor's `aliases:` maps them to the current names when the index, the site and the charts
load data, so an old run and a new run of one suite share a cell.

For library-only diagnostics on a host without a vendor toolchain, the build
script also accepts `--vendor none`. This links no vendor; use `--rows colmat`
when running `fbench` for a colmat-only measurement. This diagnostic build option
is separate from the recorder's vendor arms and does not publish a dataset.

## Which families each arm measures

MKL and AOCL measure the full family set of a suite. The OpenBLAS arm measures four families only:
`cholesky`, `lu`, `solve` and `gemm` (#111). The set is declared once, in each suite's `vendor_families`
(`benchmarks/suites/colmat-kernels.yaml` and `colmat-kernels-large.yaml`); nothing else restates it. The
recorder, the validator, the coverage check (`full` / `partial`) and the joined page all read it from the
suite through `campaign_toolkit.campaign.populations(suite, provider)`, so an OpenBLAS run of those four
families is `full`. On the joined page an OpenBLAS cell outside them reads `n/m` -- *not measured for this
arm* -- and a dash still means the arm was asked for the cell and did not carry it. The per-arm family list
needs `campaign_toolkit` with `vendor_families` (tensor4all/campaign-toolkit#83); `requirements.txt` pins it.

Why those four, and what was left out. Excluded for OpenBLAS: `eig`, `eigh`, `full_piv_lu`,
`householder_apply`, `qr`, `qr_factor`, `svd_thin`, `svd_values`, `triangular_solve` and `upgrade_factor`
in `colmat-kernels`; the same minus `eig`, `householder_apply` and `upgrade_factor` in
`colmat-kernels-large`. The evidence is the published zen5-cpu pair at colmat `93da6217`
(`data/results/zen5-cpu/faer-kernels/20261120T064812Z-pub-93da6217-20261120-mkl` with
`.../20261120T110410Z-pub-93da6217-20261120-openblas`), which has 720 cells with an OpenBLAS/colmat ratio:
444 in the four kept families and 276 in five of the excluded ones. OpenBLAS is faster than colmat (ratio
below 1) in 27 of those 276 cells (9.8%): `eigh` 8/36, `qr_factor` 15/60, `qr` 2/60, `svd_thin` 2/60,
`svd_values` 0/60. The median OpenBLAS/colmat ratio over the 276 is 1.55 (per family 1.35 to 1.85), against
1.16 to 1.32 in the kept families, so OpenBLAS is slower there overall, but it does lead in some cells. The
other five excluded families (`eig`, `full_piv_lu`, `householder_apply`, `triangular_solve`,
`upgrade_factor`) have no OpenBLAS ratio in that pair, so the data in that pair neither supports nor contradicts excluding them; they were excluded by the #111
decision, and removing a family from the exclusion is an edit to `vendor_families` alone.
The 276 / 27 figures are recomputed from the two run directories above with `scripts/gen_comparison.py`'s
loader (`vendor_arm`); they are not asserted by any test.

## The AOCL arm (`lapack-aocl`)

A third vendor arm runs the same LAPACK row body against AMD AOCL: LAPACKE from
AOCL-LAPACK (`libflame`), CBLAS from the threaded AOCL-BLAS (`libblis-mt`), plus
`libaoclutils`, which libflame needs for its CPU-feature dispatch. It is its own
binary like the other two, recorded by `record_run.py --vendor aocl` with engine `lapack-aocl`
(feature `link-aocl`) on AMD profiles; its smoke cell is
[`docs/aocl-smoke-20261119.md`](aocl-smoke-20261119.md). What a reader needs to
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
  flame=avx512` on Zen 5). The `colmat`, upstream and harness revisions come from
  the build script's `--out` file, as for the other arms.

A `colmat` checkout may live inside this repository's tree, at `extern/colmat`:
the linked crate then inherits its package fields from this repository's
[`Cargo.toml`](../Cargo.toml), which declares them.

## Tooling

The seven `scripts/*.py` entry points are thin shims over
[`campaign-toolkit`](https://github.com/tensor4all/campaign-toolkit), the shared
package the campaigns now have in common. `requirements.txt` pins it by git
revision; `benchmarks/campaign.yaml` is this campaign's descriptor, naming the
measured library, the upstream baseline, the three vendor arms, the `order` case axis
and every other value the toolkit needs. The `schemas/` copies are gone: the
toolkit supplies them.

`scripts/setup_extern_deps.sh` materialises the measured `colmat` checkout at
`extern/colmat` and `extern/upstream-faer` from the
pins, which is what `gen_index.py --check` needs to say whether a recorded page is
current.

For local development, install the toolkit from a checkout instead of from git:

```bash
pip install -e ../campaign-toolkit
```


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
([`docs/reliable-measurement.md`](reliable-measurement.md)); historical v1
pages keep their original names and semantics. Spread is judged from the raw data:
within-process from each row's `samples_ms`, between process starts from the
`run0`/`run1`/`run2` files of the same cell, and admission history from the guard
logs.
