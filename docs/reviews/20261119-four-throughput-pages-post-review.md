# Post-implementation review: `openai/gpt-6.1-sol` (independent-review skill)

Fresh `pi --print` process (read-only tools: read/grep/find/ls), run on the finished
diffs of both repositories before the PRs were opened, 2026-11-19:
`/tmp/review-90/toolkit.diff` (campaign-toolkit `feat/throughput-variant-pages`,
then `290d07e`) and `/tmp/review-90/bench.diff` (colmat-benchmark
`feat/four-throughput-pages`, then `6dc9a4b`). Report:
`20261119-four-throughput-pages-post-review.json`. `STATUS: COMPLETE`, no BLOCKER.

Findings and dispositions:

- **IMPORTANT — dataset-page provenance covered only the daily arms** (the plotted
  large-tier cells come from harness `f018ded7`, absent from every page; the chart set's
  README revisions/missing-arm disclosure was neither rendered nor linked). Fixed:
  `figures.throughput_figures` now attaches `revisions` (the `revisions_line` its README
  prints) to every throughput entry, and each dataset page renders it beside the charts —
  the page states the large-tier harness revision and the unrecorded large-tier OpenBLAS
  arm (commit `e420947`, asserted by `ThroughputPagesTest`).
- **IMPORTANT — c64 tables stated the real-operation formula (`2n^3`) while plotting four
  times that count.** Fixed: `throughput.ops_label` qualifies the printed count
  (`4 x 2n^3 (a complex element counts four times the real operations)`) in the data
  tables, the set README's contents table and its per-family lines
  (campaign-toolkit commit `07bd37e`).
- **MINOR — the size note omitted the 1536 points and the 2^k±1 boundary shapes.** Fixed
  in `benchmarks/throughput.yaml` (commit `e420947`).
- **MINOR — the tests would not catch a multiplier dropped only in `render_family`, and
  accepted one chart where nine are expected.** Fixed: the toolkit test asserts
  `render_family` asks `points` for `complex_dtype=True` (an auto-scaled log axis cannot
  show a uniform x4 by position), and the benchmark test asserts exactly nine family
  charts per page.
- **MINOR — the README promised `<profile>/<dtype>-<order>/` universally.** Fixed: the
  single-variant flat-directory exception is documented.
- **Delete list** — the silent `(view or {...})` and `site_figures(...) or []` fallbacks
  in `gen_site.main` were removed (this campaign requires its descriptor and declares
  figures); the reviewer said to keep the `write_page`/`charts=` interfaces as they are.

The reviewer verified the recorded CSVs support the corrected notes (gemm vendor cells in
both orders, decomposition vendor cells in F order only, distinct large-tier c64 cells),
traced `write_site`'s callers, the committed-chart fallback and the dict/path dispatch
for breakage, and found none. Not checked: fresh regeneration (the reviewer read the
already generated HTML), browser layout, an exhaustive CSV census, hosted CI.
