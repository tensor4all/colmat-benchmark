#!/bin/sh
# One publication of this campaign, at the pin: the daily suite on both vendor arms, the large suite on
# MKL (the large OpenBLAS arm is omitted by decision, see the bottom of this file). The site is built by the
# Pages workflow from what this wrote.
#
# The routine (campaign_toolkit.publish) preflights the machine and the trees, runs this,
# commits what it wrote and opens a pull request; this script pushes nothing. Do not run it while another campaign measures: the
# recorder's lock and the preflight both refuse that.
#
# `--dtypes` is deliberately absent: each suite records its own declared dtype list (f64/c64/f32/c32
# for the small suite, f64/c64 for the large one). The large suite is restricted to the publication
# regimes and to its declared shapes, so the adoption-only linalg-big/gemm-huge tiers are not
# published as if they were campaign coverage.
set -eu
REV=$(grep -v '^#' pins/t4a-faer.rev | grep -E '^[0-9a-f]{40}' | head -1)
LABEL="pub-$(echo "$REV" | cut -c1-8)-$(date -u +%Y%m%d)"
TASKSET="taskset -c 0-3,12-15"

$TASKSET python3 scripts/record_run.py zen5-cpu faer-kernels \
    --vendor mkl --engines faer-fork,faer-upstream,lapack-mkl --threads 1,4,8 \
    --reps 5 --aa 3 --prime-ms 250 --label "$LABEL-mkl"
$TASKSET python3 scripts/record_run.py zen5-cpu faer-kernels \
    --vendor openblas --engines faer-fork,lapack-openblas --threads 1,4,8 \
    --reps 5 --aa 6 --prime-ms 250 --label "$LABEL-openblas"
$TASKSET python3 scripts/record_run.py zen5-cpu faer-kernels-large \
    --vendor mkl --engines faer-fork,faer-upstream,lapack-mkl --threads 4,8 \
    --regimes linalg-large,gemm-large,linalg-boundary \
    --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 \
    --reps 5 --aa 3 --prime-ms 250 --label "$LABEL-large-mkl"
# The large OpenBLAS arm is not recorded in this routine. The 2026-11-18 publication dropped it by
# decision: in the 1024 large data the fork arm is ahead in essentially every cell (C order all 32
# cells, 1.11-4.47x; gemm 1.17-1.19x), so the ~7 h has no decision value, and a large-suite vendor
# ratio is not published from this campaign (docs/worklog.md states the qualification). To record it
# after all, add:
#   $TASKSET python3 scripts/record_run.py zen5-cpu faer-kernels-large \
#       --vendor openblas --engines faer-fork,lapack-openblas --threads 4,8 \
#       --regimes linalg-large,gemm-large,linalg-boundary \
#       --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 \
#       --reps 5 --aa 6 --prime-ms 250 --label "$LABEL-large-openblas"

# The joined four-way view and the throughput charts are not written here: the Pages workflow
# draws them from these runs on every deploy (scripts/gen_site.py, campaign_toolkit.figures).
