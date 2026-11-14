#!/usr/bin/env bash
# Materialise the two checkouts this campaign reads.
#
#   setup_extern_deps.sh                      # from the pins, via clones of the fork
#   setup_extern_deps.sh <fork> <upstream>   # symlink existing checkouts
#
# extern/t4a-faer is the measured fork (package and library colmat); extern/upstream-faer is
# the upstream release it is compared against. Both are linked *by path* by
# scripts/build_for_faer_rev.sh, and the index and the validator read the recorded commits
# and the reference pins out of them, so both have to exist before either runs.
#
# They must be real git checkouts -- a `.git` *directory* -- because the build script
# refuses a worktree, whose `.git` is a file.
#
# `pins/upstream-faer.rev` names a release *tag*, not a revision: a tag is what a reader can
# check against the release, and the run manifest records the commit it resolved to. Both
# the tag and that commit are in the fork's history, because the fork is based on that
# release, so the baseline checkout is materialised from the fork's remote rather than
# requiring codeberg from CI. What has to resolve is the revision and the ref, not a
# particular remote.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FORK_URL="https://github.com/tensor4all/colmat-rs"
LEGACY_FORK_URL="https://github.com/tensor4all/t4a-faer"

materialise() {  # <relative dest> <existing checkout or empty> <url>
    local dest="$PROJECT_DIR/$1" existing="${2:-}" url="$3"
    if [[ -n "$existing" ]]; then
        local dir; dir="$(cd "$existing" && pwd -P)"
        [[ -d "$dir/.git" ]] || { echo "ERROR: $dir is not a git checkout" >&2; exit 1; }
        rm -rf "$dest"
        ln -sfn "$dir" "$dest"
        return
    fi
    if [[ -d "$dest/.git" ]]; then
        git -C "$dest" fetch -q --tags origin
    else
        rm -rf "$dest"
        git clone -q "$url" "$dest"
    fi
}

materialise extern/t4a-faer "${1:-}" "$FORK_URL"
materialise extern/upstream-faer "${2:-}" "$FORK_URL"

# Pre-rename recordings include experimental heads retained only in the old
# repository. Import its branches and PR heads into separate namespaces so the
# validator can still resolve their recorded commits; do not change either pin.
git -C "$PROJECT_DIR/extern/t4a-faer" fetch -q --no-tags "$LEGACY_FORK_URL" \
    '+refs/heads/*:refs/remotes/legacy/*' \
    '+refs/pull/*/head:refs/remotes/legacy-pr/*'

REV="$(grep -v '^#' "$PROJECT_DIR/pins/t4a-faer.rev" | tr -d '[:space:]')"
[[ -n "$REV" ]] || { echo "ERROR: pins/t4a-faer.rev is empty" >&2; exit 1; }
git -C "$PROJECT_DIR/extern/t4a-faer" checkout -q "$REV"

REF="$(grep -v '^#' "$PROJECT_DIR/pins/upstream-faer.rev" | tr -d '[:space:]')"
[[ -n "$REF" ]] || { echo "ERROR: pins/upstream-faer.rev is empty" >&2; exit 1; }
git -C "$PROJECT_DIR/extern/upstream-faer" checkout -q "$REF"

echo "extern/t4a-faer at $REV ($(git -C "$PROJECT_DIR/extern/t4a-faer" log --oneline -1))"
echo "extern/upstream-faer at $REF ($(git -C "$PROJECT_DIR/extern/upstream-faer" log --oneline -1))"
