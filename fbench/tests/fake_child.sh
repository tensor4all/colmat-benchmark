#!/bin/sh
# A fake cell child for the parent/child tests. It speaks the parent's protocol and uses `from=` to
# resume where a killed child stopped, so one script can stand at every variant position.
#
#   FAKE_MODE=fast             every variant measures, then exit 0
#              hang             sleep before the first result, so only the parent's kill ends it
#              hang_after       report every result, then sleep
#              crash_after_last report every result, then exit 3
#              bad_result       one line with a field too many, then exit 0
#              bad_numbers      a line whose sample is not a number, then exit 0
#   FAKE_N=<variants>          how many variants the build has
spec="$2"
from=$(printf '%s' "$spec" | tr ';' '\n' | sed -n 's/^from=//p')
from=${from:-0}
n=${FAKE_N:-3}
mode=${FAKE_MODE:-fast}
case "$mode" in
    hang) exec sleep 300 ;;
    bad_result) printf 'RESULT\t%s\tok\t\t1.0\textra\n' "$from"; exit 0 ;;
    bad_numbers) printf 'RESULT\t%s\tok\t\tgarbage\n' "$from"; exit 0 ;;
esac
printf 'COUNT\t%s\n' "$n"
i=$from
while [ "$i" -lt "$n" ]; do
    printf 'RESULT\t%s\tok\t\t1.0\n' "$i"
    i=$((i + 1))
done
case "$mode" in
    hang_after) exec sleep 300 ;;
    crash_after_last) exit 3 ;;
esac
exit 0
