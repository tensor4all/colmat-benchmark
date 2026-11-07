#!/usr/bin/env bash
# Refuse to build when the measured crates would come from the registry instead of the checkouts.
#
# A `[patch.crates-io]` entry is silently ignored when the version the checkout provides is outside
# the range the harness declares, or when Cargo.lock already names a different version of that crate.
# Cargo warns and links the registry copy, so the build would record a revision it did not use.
# This resolves the graph first and compares the resolved `manifest_path` of every measured crate
# against the checkout that was named for it.
#
# The resolution runs `--offline` first because a measuring host may have no network, then retries
# once with the network when the failure is a *download* (a checkout whose dependency set is newer
# than the local cargo cache) rather than a resolution error.  `tests/test_faer_patch_preflight.py`
# pins both branches with a stub `cargo`: an earlier version discarded cargo's own error and reported
# only "cannot verify", which cost an hour of guessing.
#
# Kept as its own script so it can be exercised without a checkout, a toolchain or a network.
set -euo pipefail

project=""
fork_manifest=""
upstream_manifest=""
config_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) project="$2"; shift 2 ;;
        --fork-manifest) fork_manifest="$2"; shift 2 ;;
        --upstream-manifest) upstream_manifest="$2"; shift 2 ;;
        --config) config_args+=(--config "$2"); shift 2 ;;
        *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$project" && -n "$fork_manifest" && -n "$upstream_manifest" ]] || {
    echo "ERROR: --manifest, --fork-manifest and --upstream-manifest are required" >&2
    exit 2
}
# `--manifest-path` takes a manifest, not its directory.  Passing the directory is what a caller
# naturally has to hand, so reject it here with a message instead of letting cargo answer
# "is a directory but expected a file" from inside the retry logic.
case "$project" in
    Cargo.toml|*/Cargo.toml) ;;
    *) echo "ERROR: --manifest must be a Cargo.toml path, got: $project" >&2; exit 2 ;;
esac
[[ -f "$project" ]] || { echo "ERROR: no manifest at $project" >&2; exit 2; }

patch_meta="$(mktemp)"
patch_err="$(mktemp)"
cleanup() { rm -f "$patch_meta" "$patch_err"; }
trap cleanup EXIT

# Classify one cargo error file as `download`, `resolution` or `other`.  The distinction decides both
# whether a second attempt is worth making and what the reader is told when the second attempt fails:
# a missing fetch is not the same defect as an unsatisfiable requirement, and reporting one as the
# other sends the reader to the wrong place.
classify_cargo_error() {
    local file="$1"
    if grep -qiE 'expected `|unclosed|invalid toml|failed to parse|invalid type' "$file"; then
        echo other; return
    fi
    if grep -qiE 'failed to download|failed to fetch|no matching package named|unable to update|failed to update the registry index|could not resolve host|network is unreachable|connection timed out|failed to connect to|no route to|offline mode' "$file"; then
        echo download; return
    fi
    if grep -qiE 'failed to select a version|candidate versions found which did not match|which didn.t match' "$file"; then
        echo resolution; return
    fi
    echo other
}

if ! cargo metadata --manifest-path "$project" --format-version 1 --offline \
        ${config_args[@]+"${config_args[@]}"} > "$patch_meta" 2> "$patch_err"; then
    cat "$patch_err" >&2
    reason="$(classify_cargo_error "$patch_err")"
    if [ "$reason" = other ]; then
        echo "ERROR: cargo metadata failed and this is not something the network can fix; not retrying" >&2
        exit 1
    fi
    echo "cargo metadata failed ($reason); retrying online" >&2
    if ! cargo metadata --manifest-path "$project" --format-version 1 \
            ${config_args[@]+"${config_args[@]}"} > "$patch_meta" 2> "$patch_err"; then
        cat "$patch_err" >&2
        # Judge the *online* failure on its own text: an offline resolution error whose online retry
        # dies because the host has no network is a connectivity problem, not a bad requirement.
        online_reason="$(classify_cargo_error "$patch_err")"
        if [ "$online_reason" = download ]; then
            echo "ERROR: the online attempt failed on a fetch; the dependency is not cached and could not be fetched" >&2
        elif [ "$online_reason" = resolution ]; then
            echo "ERROR: the online attempt failed the same way; this is a resolution error, not a download problem" >&2
        else
            echo "ERROR: cargo metadata failed; cannot verify which libraries would be linked" >&2
        fi
        exit 1
    fi
fi

python3 - "$patch_meta" "$fork_manifest" "$upstream_manifest" <<'PYEOF'
import json, sys
meta, expected_fork, expected_upstream = sys.argv[1], sys.argv[2], sys.argv[3]
data = json.load(open(meta))
found = {p['name']: p['manifest_path'] for p in data['packages'] if p['name'] in ('t4a-faer', 'faer')}
for name, expected in (('t4a-faer', expected_fork), ('faer', expected_upstream)):
    actual = found.get(name)
    if actual != expected:
        print(f"ERROR: this build would link {name} from {actual!r}, not from {expected!r}: a `[patch]`"
              f" is ignored either when the checkout's version is outside the range the harness"
              f" declares (see `faer_fork` / `faer_upstream` in fbench/Cargo.toml) or when Cargo.lock"
              f" already pins a *different* version of that crate: widening the range alone does not"
              f" move an existing lock entry, so the lock has to be re-resolved against this checkout"
              f" (the build keeps the resolved lock beside the binary as Cargo.lock.resolved), then"
              f" re-run.", file=sys.stderr)
        raise SystemExit(1)
PYEOF
