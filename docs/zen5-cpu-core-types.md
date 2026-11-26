# `zen5-cpu`: the recorded rows ran on Zen5c compact cores (colmat-benchmark#130)

> **Retired 2026-11-27**: `zen5-cpu` is no longer a target; see [zen5-cpu-retired.md](zen5-cpu-retired.md).

The host behind `zen5-cpu` (AMD Ryzen AI 9 HX 470, "evo") is hybrid. Per-CPU
`amd_pstate_highest_perf`, read 2026-11-25:

| CPUs | highest_perf | core | L3 |
|---|---|---|---|
| 0-3 (SMT siblings 12-15) | 196 | full Zen5 | 16 MiB shared |
| 4-11 (SMT siblings 16-23) | 124 | Zen5c compact | 8 MiB shared |

## What the manifests say they measured

All 100 `zen5-cpu` manifests under `data/results/zen5-cpu/` (counted from `threads.cpu_sets`):

| `cpu_sets` | runs |
|---|---|
| `1: 4, 4: 4-7, 8: 4-11` | 62 |
| `4: 4-7, 8: 4-11` | 23 |
| `1: 4` | 11 |
| `4: 4-7` | 4 |

Every measured CPU is in 4-11: **no `zen5-cpu` row was measured on a full Zen5 core**. The
pinning was honest (the manifests record the CPUs the rows were pinned to, `taskset`ed by
`benchmarks/scripts/pinned.sh`); the *label* was the wrong part. The suites
(`colmat-kernels.yaml`, `colmat-kernels-large.yaml`) declare `cpu_sets: {zen5-cpu: ["4", "4-7", "4-11"]}`,
which overrides the toolkit profile's own `thread_rows` (`0`, `0-3`, `4-11`); the profile text
named only the L3 split.

## Verdict on the historical rows

- **Numbers: not wrong.** They are valid measurements of what they were pinned to.
- **Label: mislabelled.** Read `zen5-cpu` as "Zen5c compact cores, 8 MiB L3, 1/4/8 threads".
  They are not Zen 5 full-core numbers, and 1T/4T are *not* the cores the toolkit profile's
  `thread_rows` name.
- **Cross-machine comparisons:** a `zen5-cpu` row set against a Xeon row (full cores) mixes
  core type and L3 size into the library gap. Treat such a gap as not attributable to the library.
- **Not unaffected:** 8T (`4-11`) is the same CPU set in both declarations, but it is still
  compact cores only.

The published pages (`result/zen5-cpu/**`) show the `zen5-cpu` label and do not print the profile
description, so they do not carry this correction themselves; this note and the profile
description (`benchmarks/profiles.yaml`) do. No data was deleted or re-measured. `zen5-cpu` stays
historical; new measurements use `evo2-zen5-16c` (16 uniform full Zen5 cores).

## What the harness records from now on

`fbench run --csv X` writes beside the CSV:

- `X.affinity`: the CPUs the harness process was actually allowed to run on (`Cpus_allowed_list`
  of `/proc/self/status`) and their `amd_pstate_highest_perf` groups, so the label can be checked
  against core type; also echoed as a `CELL affinity` line into the run's guard log.
- `X.partial`: present iff some row is not `ok`/`skipped` (a failed, killed or timed-out cell);
  a finished run removes a stale one. It lists the status counts and the first failures.
