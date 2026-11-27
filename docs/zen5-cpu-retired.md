# `zen5-cpu` is retired as a benchmark target (2026-11-27)

Maintainer decision, 2026-11-27: evo (the Ryzen AI 9 HX 470 host behind `zen5-cpu`) is out of
benchmarking. The targets are `evo2` (profile `evo2-zen5-16c`, measured in the Docker bench image),
`ohtaka-cpu` and `ohtaka-fat`.

## Why

- Every historical `zen5-cpu` row was measured **entirely on the Zen5c compact cores** (CPUs 4-11),
  because the suites' `cpu_sets` overrode the toolkit profile (colmat-benchmark#130,
  `docs/zen5-cpu-core-types.md`). The label never meant a full Zen5 core.
- The published pages never had a **current-pin** run: every `zen5-cpu` row in `result/INDEX.md`
  read `missing`. Retiring the target therefore loses nothing that was ever published, and
  continuing it would publish a series that is not comparable with itself.

## What replaces it

`evo2` (the reference machine), `ohtaka-cpu` and `ohtaka-fat` (ISSP System B). One host per
published comparison; the index does not merge them.

## What changed

- `benchmarks/profiles.yaml`: the `zen5-cpu` profile is **kept, marked `retired: true`**, not
  removed. The 102 recorded manifests under `data/results/zen5-cpu/` name it, and
  `validate_run.py` refuses a manifest with an undeclared profile. No suite lists it, so it is
  not offered as a target.
- Suites `colmat-kernels` and `colmat-kernels-large`: `zen5-cpu` leaves `required_profiles`,
  `optional_profiles` and `cpu_sets`. `cpu_sets` now names `evo2-zen5-16c` with the rows its
  toolkit profile already declares (the suite schema requires at least one entry).
- `result/INDEX.md`: the six `zen5-cpu` rows (`missing`) are gone, not promised.
- CI (`benchmarks/zen5-cpu-frozen.txt` lists the 102 frozen manifests): the validator requires a suite to list a run's profile, so the frozen `zen5-cpu` manifests
  produce exactly one message, `does not list profile 'zen5-cpu'`. CI tolerates that message and
  nothing else for them, and only for the manifests on that list: a new `zen5-cpu` recording fails CI; every other check still runs on them.
- `scripts/publish.sh`, `scripts/docker_preflight.py` and `scripts/docker_campaign.sh` recorded or
  hosted `zen5-cpu` on evo's Zen5c cores (4-11). Each is reduced to a refusal with a pointer here (the
  recipes are in git history), including `docker_campaign.sh start`, which would create the container
  #130 calls invalid. `PublishTest`, which grepped the deleted recipe, is removed. evo2 is recorded
  by its own procedure.
- `docs/rules.md` and `docs/how-it-works.md` still show `record_run.py zen5-cpu ...` as examples of
  the recorder's CLI; read them as historical. The recorder itself has no profile-listing check, so
  the guard against a new `zen5-cpu` recording is CI: the frozen list below.
- Declaration hashes: editing `colmat-kernels.yaml` and `colmat-kernels-large.yaml` changes their
  `suite_sha256`, so the recorded evo2 cells of those two suites report "declaration-derived checks
  skipped" from now on (the same, said-so state as after #111). Their rows, commits and vendors are
  still checked. The ohtaka and keyaki suites are not edited, so their cells keep their hashes (the
  stale "first cell moves zen5-cpu in here" comments in them stay as they are for that reason).

## What does not change

`result/zen5-cpu/**` and `data/results/zen5-cpu/**` stay where they are. They are **historical**,
and the series breaks there on purpose: no new `zen5-cpu` rows will be recorded.

If a hybrid Zen5/Zen5c host is wanted later, that is a **new profile** with its own name and its
own `cpu_sets`, not a revival of this one.
