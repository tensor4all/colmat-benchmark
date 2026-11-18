#!/usr/bin/env python3
"""Compare a candidate run against a recorded baseline run (candidate mode).

    scripts/candidate_compare.py --baseline <run dir> --candidate <run dir> \
        [--control-engine faer-upstream] [--fork-engine faer-fork] [--cells a,b] [--json out]

Refuses the comparison - exit 1 - when either run is not valid evidence, when the machine, the
declaration or the harness/toolkit identity changed, when the in-run control row no longer
reproduces the baseline within the spread the baseline recorded, or when the two runs cannot be
paired cell for cell. The cell list is the caller's declared subset; nothing is inferred for cells
that were not measured in both runs.
"""
import argparse
import csv
import json
import pathlib
import statistics
import subprocess
import sys

import yaml
from campaign_toolkit import candidate, descriptor, validate_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The paths whose content decides what a measurement means: changing one of them invalidates the
# cells that depend on it, and a merge that leaves them alone keeps the measurement valid.
MEASURED_PATHS = ('faer/src', 'faer/Cargo.toml', 'faer-traits/src', 'faer-traits/Cargo.toml',
                  'Cargo.lock')


# The harness paths the measured binary is built from. Two runs whose harness commits differ only
# outside these (handoff notes, recorded data, reports) measured the same harness, and the comparison
# is accepted on the tree hashes of these paths rather than refused on the commit.
HARNESS_PATHS = ('fbench', 'Cargo.toml', 'Cargo.lock', 'pins', 'benchmarks',
                 'scripts/build_for_faer_rev.sh')
CAMPAIGN_ROOT = pathlib.Path(__file__).resolve().parents[1]


def harness_differences(baseline, candidate, differences):
    """`differences` without the harness-commit line when the harness build inputs are identical."""
    one = yaml.safe_load((baseline / 'run.yaml').read_text()).get('harness') or {}
    two = yaml.safe_load((candidate / 'run.yaml').read_text()).get('harness') or {}
    if one.get('commit') == two.get('commit') or not one.get('commit') or not two.get('commit'):
        return differences
    changed = candidate_module_same_code(one['commit'], two['commit'])
    if changed:
        return differences
    print(f"harness commits {one['commit'][:12]} and {two['commit'][:12]} carry identical build "
          f"inputs ({', '.join(HARNESS_PATHS)}); the difference is outside the measured harness")
    return [d for d in differences if not d.startswith('harness commit:')]


def candidate_module_same_code(before, after, root=None):
    return candidate.same_measured_code(root or CAMPAIGN_ROOT, before, after, HARNESS_PATHS, git=git)


def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def compare(args):
    """The comparison itself; every refusal and parsing/I/O failure raises or returns 1."""
    for label, run_dir in (('baseline', args.baseline), ('candidate', args.candidate)):
        errors = validate_run.validate(run_dir / 'run.yaml')
        if errors:
            print(f'refused: the {label} run is not valid evidence:')
            for error in errors:
                print(f'  {error}')
            return 1
    measured = yaml.safe_load((args.candidate / 'run.yaml').read_text())['library']['commit']
    merged = args.merged_commit or measured
    changed = candidate.same_measured_code(args.checkout, measured, merged, MEASURED_PATHS, git=git)
    if changed:
        print(f'refused: the merged revision {merged[:12]} does not carry the measured code:',
              '; '.join(changed))
        return 1
    print(f'measured revision {measured[:12]} and merged revision {merged[:12]} carry identical '
          f'measured paths')
    differences = harness_differences(args.baseline, args.candidate,
                                      candidate.identity_differences(args.baseline, args.candidate))
    if differences:
        for line in differences:
            print(f'refused: {line}')
        return 1
    # An omitted --cells means the whole recorded set; an empty argument is a malformed selection,
    # not a request for everything, so it reaches the pairing check and is refused there.
    cells = None if args.cells is None else args.cells.split(',')
    ok, controls = candidate.control_check(args.baseline, args.candidate, args.control_engine, cells)
    median_ratio = statistics.median(row['ratio'] for row in controls) if controls else 1.0
    outliers = [row for row in controls if row['outside_baseline_range']]
    print(f'{args.control_engine} control over {len(controls)} cells: median ratio '
          f'{median_ratio:.6f}, {len(outliers)} outside their own between-start range; '
          f'{"reproduces the baseline" if ok else "does NOT reproduce the baseline"}')
    if not ok:
        # The median is the decision, so the rows printed are the ones that moved it: sorting by
        # distance from 1 shows the tail that a cell-by-cell rule used to refuse on.
        for row in sorted(controls, key=lambda r: -abs(r['ratio'] - 1.0))[:10]:
            print(f"  {row['cell']}: baseline {row['baseline_ms']:.6g} "
                  f"(range {row['baseline_range_ms'][0]:.6g}-{row['baseline_range_ms'][1]:.6g}) "
                  f"candidate {row['candidate_ms']:.6g} ratio {row['ratio']:.4f}")
        print('refused: the machine is not in the baseline state')
        return 1
    rows = candidate.fork_comparison(args.baseline, args.candidate, args.fork_engine, cells)
    better = [r for r in rows if r['rotation_averaged_ratio'] < 0.95]
    worse = [r for r in rows if r['rotation_averaged_ratio'] > 1.05]
    print(f'{args.fork_engine}: {len(rows)} cells compared, faster {len(better)}, slower {len(worse)} '
          f'(beyond 5%; the rest is inside the recorded spread)')
    for row in sorted(rows, key=lambda r: r['rotation_averaged_ratio'])[:5]:
        print(f"  {row['cell']}: {row['rotation_averaged_ratio']:.3f} "
              f"({row['baseline_fork_ms']:.6g} -> {row['candidate_fork_ms']:.6g} ms)")
    if args.json:
        args.json.write_text(json.dumps({'identity_differences': differences, 'controls': controls,
                                         'fork': rows}, indent=2) + '\n')
    return 0


def main(argv=None):
    descriptor.set_root(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', required=True, type=pathlib.Path)
    ap.add_argument('--candidate', required=True, type=pathlib.Path)
    ap.add_argument('--control-engine', default='faer-upstream')
    ap.add_argument('--fork-engine', default='faer-fork')
    ap.add_argument('--cells', default=None, help='comma separated substrings of the declared subset')
    ap.add_argument('--json', type=pathlib.Path, default=None)
    ap.add_argument('--checkout', type=pathlib.Path, default=ROOT / 'extern/t4a-faer',
                    help='the checkout that carries both the measured and the merged revision')
    ap.add_argument('--merged-commit', default=None,
                    help='the revision the candidate was merged into; the comparison is refused when '
                         'the merge changed any measured path, so the measurement cannot silently '
                         'describe different code than the branch that was measured')
    args = ap.parse_args(argv)
    try:
        return compare(args)
    except (OSError, KeyError, ValueError, csv.Error, yaml.YAMLError, subprocess.SubprocessError) as error:
        print(f'refused: {error}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
