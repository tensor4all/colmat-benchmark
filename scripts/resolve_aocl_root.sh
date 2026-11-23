#!/usr/bin/env bash
# Print the AOCL install tree the AOCL arm (`--vendor aocl`, feature `fbench/link-aocl`) links, or
# exit 1 with the reason on stderr. Shared by scripts/build_for_faer_rev.sh and scripts/publish.sh,
# so the tree the link preflight checks, the tree the binary links and the tree the manifest records
# are one tree.
#
# A tree is the directory holding `lib_LP64/{libflame,libblis-mt,libaoclutils}.so`: AOCL 5.x installs
# it as `<prefix>/<ver>/gcc`, AOCL 6.x as `<prefix>/<ver>/gcc/MT` (beside a single-threaded `ST`;
# this arm is the threaded one). `AOCL_ROOT`, when set, names the tree (or its 6.x parent) and is
# not second-guessed. Otherwise the newest version under `/opt/AMD/aocl` wins, then the newest under
# the per-user `$HOME/opt/AMD/aocl` (an install made without root). The path printed is physical,
# never through a `latest`-style symlink, because it is what a manifest records.
set -euo pipefail

complete() {  # $1: candidate; prints the tree for a 5.x or a 6.x layout
    local d lib
    for d in "$1" "$1/MT"; do
        lib="$d/lib_LP64"
        if [[ -f "$lib/libflame.so" && -e "$lib/libblis-mt.so" && -f "$lib/libaoclutils.so" ]]; then
            (cd "$d" && pwd -P)
            return 0
        fi
    done
    return 1
}

if [[ -n "${AOCL_ROOT:-}" ]]; then
    complete "$AOCL_ROOT" || {
        echo "AOCL_ROOT=$AOCL_ROOT holds no lib_LP64 with libflame, libblis-mt and libaoclutils" >&2
        exit 1; }
    exit 0
fi
for prefix in /opt/AMD/aocl "${HOME:-/nonexistent}/opt/AMD/aocl"; do
    while IFS= read -r candidate; do
        [[ "$candidate" == */latest/* ]] && continue
        complete "$candidate" && exit 0
    done < <(ls -d "$prefix"/*/gcc 2>/dev/null | sort -rV)
done
echo "no AOCL install with libflame, threaded BLIS and aocl-utils under /opt/AMD/aocl/*/gcc[/MT]" \
     "or \$HOME/opt/AMD/aocl/*/gcc[/MT]; install one (install.sh -t <prefix> -l blis libflame utils -i lp64)" \
     "or set AOCL_ROOT" >&2
exit 1
