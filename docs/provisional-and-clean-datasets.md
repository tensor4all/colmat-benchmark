# Provisional pages now, a clean dataset later

The campaign runs two jobs with different tolerances, and they will end up on different machines.

| job | where | grid | published? |
| --- | --- | --- | --- |
| **Acceptance** - does this change go into `t4a-faer`? | the development host | the change's blast radius: affected families at every declared shape at 4T/8T, target cells at 1T, two or three untouched families as a collateral control | **no** - the evidence stays in the pull request and the worklog |
| **Public dataset** - the pages under `result/` | a dedicated quiet measurement host, re-recorded periodically and published automatically | the whole declared suite | yes |

Both keep the same frozen contract (section 1 of `protocol-v2-baseline.md`: prime 250 ms, five calls,
three complete process sets, boost off, observer off the measured L3 domains, whole-L3-domain guard,
`PINNED_RETRIES=8`) and the same provenance (library, upstream, harness and toolkit revisions, binary
SHA-256, host block including the power state).

## What this host does in the meantime

- Pages recorded here carry `host.role: development` in their manifest and say in their header that they
  are **provisional** and will be superseded by the measurement host's dataset. The shared profile
  `zen5-cpu` states that role.
- They are **not** refreshed by re-recording the suite on this host; the clean host replaces them. The
  pages recorded on 2026-10-15/16 additionally carry the linked-crate caveat in
  `reviews/20261016-linked-crate-provenance.md`.
- Acceptance runs live under `data/probes/` (never in `data/results/`, where the CI resolves every
  manifest against the pinned checkout) and are not published as pages.
- Nothing about acceptance gates on a page: it gates on the acceptance run plus the fork's tests and the
  harness's numerical verification for the affected populations.

## The day the measurement host arrives

1. Add its profile with `role: measurement` (CPU, L3 domains, thread sets), and its pins.
2. Materialise `extern/` at the adopted revisions and record `colmat-kernels` and `colmat-kernels-large` for
   both arms; the recorder rebuilds and re-verifies by itself.
3. Publish; the index then holds both the provisional development-host pages and the measurement-host
   dataset, each self-describing through `host.role`.
4. Schedule the periodic re-record (on revision change or weekly) and keep publication as
   `gen_index.py --write`; acceptance stays a local, one-hour job on whichever machine the change is
   developed.

## What not to do while this is provisional

- Do not re-record the whole suite on the development host to "refresh" a page.
- Do not publish an acceptance run as a page, and do not widen its grid after seeing the numbers.
- Do not relax the frozen settings to save time; if acceptance needs more than three hours, fix the
  protocol instead.
