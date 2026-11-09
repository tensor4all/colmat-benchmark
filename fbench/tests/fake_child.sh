#!/bin/sh
# A fake cell child for the transition table. It speaks the parent's `RESULT`/`SKIP` protocol and
# ignores the real spec except for `from=`, so one script can stand at every variant position.
#
#   FAKE_MODE=fast|cut|error|skip
#   FAKE_AT=<variant index to cut/error at>
#   FAKE_N=<number of variants>
spec="$2"
from=$(printf '%s' "$spec" | tr ';' '\n' | sed -n 's/^from=//p')
from=${from:-0}
n=${FAKE_N:-3}
mode=${FAKE_MODE:-fast}
at=${FAKE_AT:-0}
i=$from
if [ "$mode" = "skip" ]; then
    printf 'SKIP\tfake skip\n'
    exit 0
fi
while [ "$i" -lt "$n" ]; do
    if [ "$i" -eq "$at" ]; then
        case "$mode" in
            cut)
                if [ "$i" -eq 0 ]; then
                    # The construction-cut child reports one timeout and stops; the parent must
                    # respawn the remaining variants, not mark them failed.
                    printf 'RESULT\t0\ttimeout\ttimeout 1\t\t\t\n'
                    exit 0
                fi
                exec sleep 300 ;;
            error) exit 3 ;;
        esac
    fi
    printf 'RESULT\t%s\tok\t\t1.0\t2.0\t1.0\n' "$i"
    i=$((i + 1))
done
if [ "$mode" = "hang_after" ]; then
    exec sleep 300
fi
exit 0
