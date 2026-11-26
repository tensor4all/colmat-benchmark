#!/bin/sh
# RETIRED 2026-11-27. This was the evo (`zen5-cpu`) publication routine: it recorded the daily and
# large suites on evo's Zen5c cores (taskset 0-3,12-15, measured 4-11; #130). `zen5-cpu` is no longer a
# campaign target and no suite lists it, so the recorder could not pin it any more. The recipe is in git
# history (the commit before docs/zen5-cpu-retired.md); evo2 is recorded by its own procedure.
echo "publish.sh: zen5-cpu is retired (docs/zen5-cpu-retired.md); nothing recorded." >&2
exit 2
