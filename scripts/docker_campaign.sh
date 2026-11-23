#!/usr/bin/env bash
# One publication of this campaign in Docker (#99): the recorder on the observing CPUs, the
# measured process in a container confined to the measured CPUs.
#
#   scripts/docker_campaign.sh start      # create the measured container (cpuset 4-11), idle
#   scripts/docker_campaign.sh preflight  # write docs/preflight/docker-99-<UTC>.yaml; exit 1 unless PASS
#   scripts/docker_campaign.sh run        # scripts/publish.sh (+ the large OpenBLAS arm) in the recorder container
#   scripts/docker_campaign.sh stop       # remove the measured container
#
# Two containers of one image, both on the *root* daemon (the rootless one discards --cpuset-cpus):
#   - the recorder container, cpuset 0-3,12-15: the recorder, the build, the link preflight and
#     the idle guard run here, so the observer is outside the measured L3 domain (the recorder
#     refuses to start inside it) and the toolchain that is recorded is the image's;
#   - the measured container, cpuset 4-11, idle until used: benchmarks/scripts/pinned.sh runs each
#     measured command in it as `docker exec <name> taskset -c <row cpus> ...` (BENCH_CONTAINER).
# The image has no Fortran compiler, which the OpenBLAS arm needs to build OpenBLAS from source
# (scripts/build_for_faer_rev.sh). The recorder container therefore runs a one-layer derivation of
# the image, `build-image` below: IMAGE plus gfortran from apt. The measured container runs IMAGE
# itself; the OpenBLAS arm links OpenBLAS statically, so nothing of the derivation is loaded there.
# The repository is mounted at /work/colmat-99 in both, so a path means the same on both sides.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${BENCH_IMAGE:-shinaoka/tenferro-bench-cpu:cpu-20261120-julia}"
MEASURED_CPUS="${MEASURED_CPUS:-4-11}"
OBSERVER_CPUS="${OBSERVER_CPUS:-0-3,12-15}"
MEASURED_NAME="${BENCH_CONTAINER:-colmat-measured-99}"
RECORDER_IMAGE="${BENCH_RECORDER_IMAGE:-colmat-bench-recorder-99:${IMAGE##*:}}"
MOUNT=/work/colmat-99
export DOCKER_HOST="${DOCKER_HOST_ROOT:-unix:///var/run/docker.sock}"

recorder() {  # recorder <script>: runs it in a throwaway recorder container
    docker run --rm --cpuset-cpus="$OBSERVER_CPUS" \
        -v "$ROOT:$MOUNT" -w "$MOUNT" \
        -v /usr/bin/docker:/usr/bin/docker:ro -v /var/run/docker.sock:/var/run/docker.sock \
        -e BENCH_CONTAINER="$MEASURED_NAME" -e PUBLISH_LARGE_OPENBLAS="${PUBLISH_LARGE_OPENBLAS:-1}" \
        "$RECORDER_IMAGE" bash -c "git config --global --add safe.directory '*'; export PATH=$MOUNT/target/venv/bin:\$PATH; $1"
}

case "${1:-}" in
build-image)
    printf 'FROM %s\nRUN apt-get update && apt-get install -y --no-install-recommends gfortran && rm -rf /var/lib/apt/lists/*\n' "$IMAGE" \
        | docker build -q -t "$RECORDER_IMAGE" -
    ;;
start)
    docker rm -f "$MEASURED_NAME" >/dev/null 2>&1 || true
    docker run -d --name "$MEASURED_NAME" --cpuset-cpus="$MEASURED_CPUS" -v "$ROOT:$MOUNT" -w "$MOUNT" \
        "$IMAGE" sleep infinity >/dev/null
    docker exec "$MEASURED_NAME" grep Cpus_allowed_list /proc/self/status
    ;;
stop)
    docker rm -f "$MEASURED_NAME" >/dev/null
    ;;
preflight)
    exec python3 "$ROOT/scripts/docker_preflight.py" "${@:2}"
    ;;
run)
    recorder "sh scripts/publish.sh"
    ;;
shell)
    shift; recorder "$*"
    ;;
*)
    echo "usage: $0 build-image|start|preflight|run|stop|shell <cmd>" >&2; exit 2 ;;
esac
