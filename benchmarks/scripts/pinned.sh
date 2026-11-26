#!/usr/bin/env bash
# Run one measurement pinned to CPUS, valid only if CPUS were idle right
# before and right after it. The command's stdout is printed only for a valid
# run; a run spoiled by other load is retried up to PINNED_RETRIES times
# (default 3), waiting PINNED_WAIT seconds (default 30) for the cores to become
# idle. Rejected executions are archived, never silently overwritten. A caller
# writing a CSV must declare its effective output path with PINNED_CSV; this
# generic wrapper does not guess a program's argument grammar.
#
#   pinned.sh CPUS -- COMMAND [ARGS...]
#
# Exits 75 when no valid run was obtained.
#
# Evidence location (#143): the command's stdout and the guard log are created *in* an evidence
# directory on the shared filesystem, not in node-local /tmp, because on a Slurm compute node /tmp
# dies with the job and a four-hour failure would leave only a log line naming files that no
# longer exist. The directory is $PINNED_EVIDENCE_DIR if set, else
# $SLURM_SUBMIT_DIR/failed-evidence/$SLURM_JOB_ID inside a Slurm job, else $TMPDIR (or /tmp).
# Writing there directly (not copying from an EXIT trap) means a SIGKILL, a wall-time kill or a
# node failure still leaves the files. A valid run deletes them; a failed one keeps them. A
# rejected attempt (busy after the run) is archived beside $PINNED_CSV when that is declared, else
# in the evidence directory, so the "no valid run" outcome is on the shared filesystem too.
#
# Container mode (#99): with BENCH_CONTAINER=<name> the measured command runs *inside* that
# running container as `docker exec <name> taskset -c CPUS COMMAND...` instead of under a
# host `taskset`. The recorder, this script and the idle guard stay on the observing side
# (outside the container's cpuset); only the measured process is confined to the container's
# cpuset and then to CPUS inside it. The repository is mounted at the same path on both
# sides, so COMMAND's paths mean the same thing. The container's image and cpuset are
# echoed to stderr, hence into the guard log of every run. CPU affinity is Linux-only: on
# other hosts the command runs unpinned and a note goes to stderr.
#
# This copy lives in the harness repository, not in the measured library: faer
# has no `benchmarks/` directory and the fork's AGENTS.md forbids adding one, so
# the harness owns the timing protocol as well as the binary. It is the copy from
# `tensor4all/tlinalg-rs` `benchmarks/scripts/pinned.sh` at c5d1d61f (see
# docs/provenance.md), itself adapted from `tensor4all/tprims-rs`
# (MIT, Copyright (c) 2026 Lukas Devos and tensor4all contributors). The
# effective idle window, busy threshold and retry budget are echoed to stderr,
# so the guard log of a run records the policy that was actually in force
# rather than the one the manifest assumes.
set -euo pipefail
cpus=$1; shift
[[ ${1:-} == -- ]] && shift
here=$(cd "$(dirname "$0")" && pwd)
seconds=${PINNED_IDLE_SECONDS:-3}
max_busy=${PINNED_MAX_BUSY:-0.05}
retries=${PINNED_RETRIES:-3}
wait=${PINNED_WAIT:-30}
check() { python3 "$here/idle_cpus.py" check "$cpus" --seconds "$seconds" --max-busy "$max_busy" 2> "$guard"; }
runner=(taskset -c "$cpus")
if [[ -n ${BENCH_CONTAINER:-} ]]; then
    runner=(docker exec "$BENCH_CONTAINER" taskset -c "$cpus")
    echo "pinned.sh: measured in container $BENCH_CONTAINER: $(docker inspect -f 'image={{.Config.Image}} image_id={{.Image}} cpuset={{.HostConfig.CpusetCpus}}' "$BENCH_CONTAINER")" >&2
fi
if [[ $(uname -s) != Linux ]] || [[ ! -r /proc/stat ]] || ! command -v taskset >/dev/null; then
    echo "pinned.sh: CPU pinning unavailable on this host (Linux only); running unpinned" >&2
    exec "$@"
fi
echo "pinned.sh: cpus=$cpus idle_window=${seconds}s max_busy=$max_busy retries=$retries" >&2
evidence=${PINNED_EVIDENCE_DIR:-}
if [[ -z $evidence && -n ${SLURM_JOB_ID:-} ]]; then
    evidence=${SLURM_SUBMIT_DIR:-$PWD}/failed-evidence/$SLURM_JOB_ID
fi
tmp=
if [[ -n $evidence ]] && mkdir -p -- "$evidence" 2>/dev/null \
        && tmp=$(mktemp "$evidence/pinned-stdout.XXXXXX" 2>/dev/null) \
        && guard=$(mktemp "$evidence/pinned-guard.XXXXXX" 2>/dev/null); then
    :
else
    [[ -z $tmp ]] || rm -f -- "$tmp"

    [[ -z $evidence ]] || echo "pinned.sh: cannot create evidence dir $evidence; using ${TMPDIR:-/tmp}" >&2
    evidence=
    tmp=$(mktemp)
    guard=$(mktemp)
fi
keep=0
cleanup() {
    if [[ $keep == 1 ]]; then
        echo "pinned.sh: evidence retained at $tmp and $guard" >&2
    else
        rm -f "$tmp" "$guard"
    fi
}
trap cleanup EXIT
for attempt in $(seq 1 "$retries"); do
    if ! check; then
        echo "pinned.sh: cpus $cpus busy before run (attempt $attempt); waiting" >&2
        cat "$guard" >&2
        sleep "$wait"; continue
    fi
    keep=1  # Retain through command execution, post-check and fallible archival.
    if "${runner[@]}" "$@" > "$tmp"; then
        :
    else
        status=$?
        echo "pinned.sh: command failed ($status); no post-check or retry" >&2
        exit "$status"
    fi
    if check; then
        cat "$tmp"
        keep=0
        echo "pinned.sh: cpus $cpus idle before and after (attempt $attempt)" >&2
        exit 0
    fi
    echo "pinned.sh: cpus $cpus busy after run (attempt $attempt); rejecting it" >&2
    archive_base=${PINNED_CSV:-}
    if [[ -z $archive_base ]]; then archive_base=${evidence:+$evidence/pinned}; fi
    archive=$(mktemp -d "${archive_base:-${TMPDIR:-/tmp}/pinned}.rejected.XXXXXX")
    echo "pinned.sh: rejected attempt $attempt archived at $archive" >&2
    cp -- "$tmp" "$archive/command.stdout"
    if [[ -n ${PINNED_CSV:-} ]]; then
        cp -- "$PINNED_CSV" "$archive/result.csv"
    fi
    cp -- "$guard" "$archive/guard.log"
    keep=0
done
echo "pinned.sh: no valid run on cpus $cpus" >&2
exit 75
