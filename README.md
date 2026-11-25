# `colmat-benchmark`

> ## 📈 Results: **<https://tensor4all.org/colmat-benchmark/>**
>
> colmat against MKL, OpenBLAS and AOCL, one section per vendor, per machine and per family, with
> the strictest view ("vs the fastest of all") folded underneath. Generated from `data/` on every
> deploy; the page's own "How the numbers are made" states the cell rule and the verdict.

The benchmark campaign for [colmat](https://github.com/tensor4all/colmat-rs) (`tensor4all/colmat-rs`,
the tensor4all fork of `faer`): results keyed by **library commit × hardware profile × vendor arm**,
measured at colmat's own API, with upstream `faer` as the baseline row. It has the same shape as
[`tlinalg-benchmark`](https://github.com/tensor4all/tlinalg-benchmark) and
[`tcontract-benchmark`](https://github.com/tensor4all/tcontract-benchmark), one layer down.

| | |
|---|---|
| Current pages | [`result/INDEX.md`](result/INDEX.md) (generated); older ones in [`result/HISTORY.md`](result/HISTORY.md); read [`docs/dataset-status.md`](docs/dataset-status.md) first (pages recorded before #49 carry withdrawn cells; the provisional-dataset and provenance notes are in [`docs/how-it-works.md`](docs/how-it-works.md#status-moved-from-the-readme)) |
| How a run is recorded, the vendor arms, tooling, page → raw data | [`docs/how-it-works.md`](docs/how-it-works.md), including [Which families each arm measures](docs/how-it-works.md#which-families-each-arm-measures) and [the AOCL arm](docs/how-it-works.md#the-aocl-arm-lapack-aocl) |
| Result layout | [`docs/results.md`](docs/results.md) |
| Rules and runbook | [`docs/rules.md`](docs/rules.md), [`docs/merge-procedure.md`](docs/merge-procedure.md), [`AGENTS.md`](AGENTS.md) |
| Measurement audit | [`docs/measurement-audit-2026-11-06.md`](docs/measurement-audit-2026-11-06.md): what each timed call includes, per family and arm |
| Timing policy and protocol | [`docs/timing-policy.md`](docs/timing-policy.md), [`docs/protocol-v2-baseline.md`](docs/protocol-v2-baseline.md), [`docs/reliable-measurement.md`](docs/reliable-measurement.md) |
| Provisional vs clean datasets | [`docs/provisional-and-clean-datasets.md`](docs/provisional-and-clean-datasets.md) |
| Design record | [`docs/worklog.md`](docs/worklog.md), [`docs/reviews/`](docs/reviews/) |
| Reference revisions | [`pins/colmat.rev`](pins/colmat.rev), [`pins/upstream-faer.rev`](pins/upstream-faer.rev) |
| Provenance | [`docs/provenance.md`](docs/provenance.md) |

It is not colmat's per-PR gate, not tlinalg/tenferro marshalling (their own benchmarks' rows), and
not upstream's website bench page. No API change to `colmat` follows from it.
