#!/bin/sh
# RETIRED 2026-11-27. This wrapped the evo (`zen5-cpu`) Docker campaign (#99): a measured container on
# the Zen5c cores 4-11 with the observer on 0-3,12-15. `zen5-cpu` is no longer a campaign target
# (docs/zen5-cpu-retired.md), and a container on those cores would measure exactly what #130 calls
# invalid, so every subcommand refuses. The recipe is in git history; evo2 has its own procedure.
echo "docker_campaign.sh: zen5-cpu is retired (docs/zen5-cpu-retired.md); nothing started." >&2
exit 2
