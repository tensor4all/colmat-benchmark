#!/usr/bin/env python3
"""RETIRED 2026-11-27. This was the preflight of the evo (`zen5-cpu`) Docker campaign (#99): smoke cells
recorded on evo's Zen5c cores 4-11. `zen5-cpu` is no longer a campaign target (docs/zen5-cpu-retired.md),
so the recorder would fail on a healthy machine. The checks are in git history (the commit before the
retirement); evo2 has its own procedure."""
import sys

sys.exit("docker_preflight.py: zen5-cpu is retired (docs/zen5-cpu-retired.md); nothing checked.")
