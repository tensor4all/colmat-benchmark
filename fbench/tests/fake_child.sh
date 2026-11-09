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
if [ "$mode" = "skip_exit3" ]; then
    printf 'SKIP\tfake skip\n'
    exit 3
fi
if [ "$mode" = "count_flood" ]; then
    # `yes` keeps the pipe full, so the old parent's recv always had a queued message to return.
    exec yes 'COUNT	1'
fi
if [ "$mode" = "late_skip" ] && [ "$from" -eq 0 ]; then
    printf 'RESULT\t0\tok\t\t1\t2\t1\n'
    printf 'SKIP\tlate skip\n'
    exit 0
fi
if [ "$mode" = "garbage_after_last" ] && [ "$from" -eq 0 ]; then
    # Every variant's result, then a line that is not part of the protocol at all.
    i=0
    while [ "$i" -lt "$n" ]; do
        printf 'RESULT\t%s\tok\t\t1.0\t2.0\t1.0\n' "$i"
        i=$((i + 1))
    done
    printf 'garbage\n'
    exit 0
fi
while [ "$i" -lt "$n" ]; do
    if [ "$i" -eq "$at" ]; then
        case "$mode" in
            hang) exec sleep 300 ;;
            cut)
                if [ "$i" -eq 0 ]; then
                    # The construction-cut child reports one timeout and stops; the parent must
                    # respawn the remaining variants, not mark them failed.
                    printf 'RESULT\t0\ttimeout\ttimeout 1\t\t\t\n'
                    exit 0
                fi
                exec sleep 300 ;;
            error) exit 3 ;;
            error_once)
                # A crash after one result, only for the first attempt; a replacement child is fine.
                if [ "$from" -eq 0 ]; then exit 3; fi ;;
            bad_result)
                # A malformed RESULT followed by a hang, only for the first attempt.
                if [ "$from" -eq 0 ]; then
                    printf 'RESULT\t999\tok\t\t1\t2\t1\n'
                    exec sleep 300
                fi ;;
            huge_index)
                # An index a `usize` holds but this build cannot: the parent must not do arithmetic
                # on it (it used to overflow while formatting the error message and panic).
                if [ "$from" -eq 0 ]; then
                    printf 'RESULT\t18446744073709551615\tok\t\t1\t2\t1\n'
                    exit 0
                fi ;;
            bad_numbers)
                if [ "$from" -eq 0 ]; then
                    printf 'RESULT\t0\tok\t\tgarbage\tNaN\tgarbage\n'
                    exit 0
                fi ;;
            short_samples)
                if [ "$from" -eq 0 ]; then
                    printf 'RESULT\t0\tok\t\t1\t2\t1\n'
                    exit 0
                fi ;;
        esac
    fi
    printf 'RESULT\t%s\tok\t\t1.0\t2.0\t1.0\n' "$i"
    i=$((i + 1))
done
if [ "$mode" = "hang_after" ]; then
    exec sleep 300
fi
if [ "$mode" = "crash_after_last" ]; then
    exit 3
fi
exit 0
