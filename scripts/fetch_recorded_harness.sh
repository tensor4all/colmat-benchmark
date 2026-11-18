#!/usr/bin/env bash
# Fetch the harness commits the recorded runs name.
#
# `harness.commit` is a commit of this repository, but the two arms of a publication can be recorded
# either side of a data commit, and a data commit that lands by squash leaves its SHA unreachable
# from `main` -- so a plain `fetch-depth: 0` checkout does not hold its tree. The join compares the
# harness by the paths the binary is built from (`scripts/gen_comparison.py`), which needs both
# trees, so a clone that does not hold one fetches it by SHA (`http.lowSpeed*` aborts a stalled
# connection instead of holding the job open). A commit it cannot get is a warning, not a failure:
# the join guard then refuses that pair instead of certifying it by absence.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"

python3 - "$root" <<'PY'
import pathlib
import subprocess
import sys

import yaml

root = pathlib.Path(sys.argv[1])
commits = set()
for manifest in (root / 'data' / 'results').glob('*/*/*/run.yaml'):
    commit = (yaml.safe_load(manifest.read_text()).get('harness') or {}).get('commit')
    if commit:
        commits.add(commit)
for commit in sorted(commits):
    present = subprocess.run(['git', '-C', str(root), 'cat-file', '-e', commit + '^{commit}'],
                             capture_output=True)
    if present.returncode == 0:
        continue
    fetched = subprocess.run(['git', '-C', str(root), '-c', 'http.lowSpeedLimit=1000',
                              '-c', 'http.lowSpeedTime=30', 'fetch', '--quiet', 'origin', commit],
                             capture_output=True, text=True)
    if fetched.returncode == 0:
        print(f'fetched recorded harness commit {commit}')
    else:
        print(f'WARNING: this checkout does not hold the recorded harness commit {commit} and it '
              f'cannot be fetched; a join that needs its tree is refused ({fetched.stderr.strip()})',
              file=sys.stderr)
PY
