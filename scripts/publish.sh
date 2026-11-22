#!/bin/sh
# One publication of this campaign, at the pin: the daily suite on the MKL, OpenBLAS and AOCL arms, the
# large suite on MKL and AOCL (the large OpenBLAS arm is omitted by decision, see the bottom of this file).
# The site is built by the Pages workflow from what this wrote.
#
# AOCL is AMD's own LAPACK stack and every profile this routine records (zen5-cpu) is an AMD host, so the
# AOCL arm runs here (#95; its cost, about 3/2 of the two-arm time, is accepted). A host with no AOCL
# install (scripts/resolve_aocl_root.sh finds none) skips both AOCL arms with the reason printed and
# written to $AOCL_SKIP_LOG; the index then shows those cells as `missing`, it does not fail the routine.
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
REV=$(grep -v '^#' pins/colmat.rev | grep -E '^[0-9a-f]{40}' | head -1)
LABEL="pub-$(echo "$REV" | cut -c1-8)-$(date -u +%Y%m%d)"
TASKSET="taskset -c 0-3,12-15"

$TASKSET python3 scripts/record_run.py zen5-cpu colmat-kernels \
    --vendor mkl --engines colmat,faer,lapack-mkl --threads 1,4,8 \
    --reps 5 --aa 3 --prime-ms 250 --label "$LABEL-mkl"
$TASKSET python3 scripts/record_run.py zen5-cpu colmat-kernels \
    --vendor openblas --engines colmat,lapack-openblas --threads 1,4,8 \
    --reps 5 --aa 6 --prime-ms 250 --label "$LABEL-openblas"
$TASKSET python3 scripts/record_run.py zen5-cpu colmat-kernels-large \
    --vendor mkl --engines colmat,faer,lapack-mkl --threads 4,8 \
    --regimes linalg-large,gemm-large,linalg-boundary \
    --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 \
    --reps 5 --aa 3 --prime-ms 250 --label "$LABEL-large-mkl"

# The AOCL arm, daily and large, through the same recorder, repetitions and drift control (AOCL carries no
# upstream row, so its A/A count follows the OpenBLAS arm). The tree is resolved once and exported, so the
# link preflight, the build and the manifest all name the same AOCL.
AOCL_SKIP_LOG="${AOCL_SKIP_LOG:-target/publish-aocl-skip.txt}"
mkdir -p "$(dirname "$AOCL_SKIP_LOG")"
if AOCL_ROOT="$(scripts/resolve_aocl_root.sh 2>"$AOCL_SKIP_LOG.err")"; then
    export AOCL_ROOT
    rm -f "$AOCL_SKIP_LOG.err"
    $TASKSET python3 scripts/record_run.py zen5-cpu colmat-kernels \
        --vendor aocl --engines colmat,lapack-aocl --threads 1,4,8 \
        --reps 5 --aa 6 --prime-ms 250 --label "$LABEL-aocl"
    $TASKSET python3 scripts/record_run.py zen5-cpu colmat-kernels-large \
        --vendor aocl --engines colmat,lapack-aocl --threads 4,8 \
        --regimes linalg-large,gemm-large,linalg-boundary \
        --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 \
        --reps 5 --aa 6 --prime-ms 250 --label "$LABEL-large-aocl"
else
    { echo "AOCL arm skipped on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ):"; cat "$AOCL_SKIP_LOG.err"; } >"$AOCL_SKIP_LOG"
    rm -f "$AOCL_SKIP_LOG.err"
    cat "$AOCL_SKIP_LOG" >&2
fi
# The large OpenBLAS arm is not recorded in this routine. The 2026-11-18 publication dropped it by
# decision: in the 1024 large data the fork arm is ahead in essentially every cell (C order all 32
# cells, 1.11-4.47x; gemm 1.17-1.19x), so the ~7 h has no decision value, and a large-suite vendor
# ratio is not published from this campaign (docs/worklog.md states the qualification). To record it
# after all, add:
#   $TASKSET python3 scripts/record_run.py zen5-cpu colmat-kernels-large \
#       --vendor openblas --engines colmat,lapack-openblas --threads 4,8 \
#       --regimes linalg-large,gemm-large,linalg-boundary \
#       --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 \
#       --reps 5 --aa 6 --prime-ms 250 --label "$LABEL-large-openblas"

# The large OpenBLAS arm, off by default (see above) and on for a same-pin three-arm refresh (#99):
# PUBLISH_LARGE_OPENBLAS=1 records it after the others, so the large suite carries all three vendors.
if [ "${PUBLISH_LARGE_OPENBLAS:-0}" = 1 ]; then
    $TASKSET python3 scripts/record_run.py zen5-cpu colmat-kernels-large \
        --vendor openblas --engines colmat,lapack-openblas --threads 4,8 \
        --regimes linalg-large,gemm-large,linalg-boundary \
        --shapes 1024,1536,1024x1024x1024,1536x1536x1536,255,257,511,513 \
        --reps 5 --aa 6 --prime-ms 250 --label "$LABEL-large-openblas"
fi

# The joined four-way view and the throughput charts are not written here: the Pages workflow
# draws them from these runs on every deploy (scripts/gen_site.py, campaign_toolkit.figures).
