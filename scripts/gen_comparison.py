#!/usr/bin/env python3
"""Join the two vendor arms of one recording into a single browser-viewable page.

    scripts/gen_comparison.py --mkl <run dir> --openblas <run dir> [--aocl <run dir>] --out <file.html>

One binary links one vendor library, so `colmat` is measured beside MKL in one
run and beside OpenBLAS in the other; upstream `faer` is only in the MKL arm
because the three-row rotation needs three engines. This joins the two runs of
the *same* recorded revision on the cell key and prints, for every cell, what the
fork took against each baseline it was measured with. The fork is measured in
both arms and both values are shown, so a ratio never divides by a number the
reader cannot see.

One session is not one wall-clock reading -- the arms of a publication are recorded
hours apart -- so the arms are joined on what the numbers depend on (host, power
state, idle-guard and timing policy, the CPU set of every thread budget they
measured, the revision of library/upstream/harness/toolkit, and the cell grid), and
a run that differs in any of those is refused with the field named rather than
joined into a table that claims more than was measured. The harness is the measured
harness, not the commit that happened to hold it: two arms recorded either side of a
commit that only adds recorded data are one identity (`harness_identity_same`).

It is a view over recorded data, not a new measurement: a cell that one arm does
not carry is shown as a dash, and the per-arm Markdown pages plus their manifests
stay authoritative for the control statement and the verification record.

On an AMD profile a third arm, AOCL (`--aocl`), is joined the same way: its own fork
column, its own `lapack-aocl` column, and `AOCL/fork` dividing by the fork the AOCL arm
measured. Without `--aocl` the page is the two-arm page, unchanged.
"""
import argparse, csv, html, pathlib, statistics, subprocess, sys
import yaml

import candidate_compare
from campaign_toolkit import descriptor

KEY = ('regime', 'family', 'dtype', 'order', 'm', 'n', 'k', 'batch', 'threads', 'note')
ROOT = pathlib.Path(__file__).resolve().parents[1]

# The measured library and its upstream baseline, under their current engine names. A run
# recorded before the rename (colmat-benchmark#113) wrote `faer-fork` / `faer-upstream`; the
# descriptor's `aliases.engines` maps those to these when the rows are read (`engine`).
COLMAT, FAER = 'colmat', 'faer'


def campaign():
    """This repository's descriptor (its `aliases` name the old rows), loaded once.

    Loaded explicitly rather than from the working directory, so the page is the same wherever
    the script is run from.
    """
    if descriptor._current is None or descriptor._current.root != ROOT:
        descriptor.set_root(ROOT)
    return descriptor.current()


def current_suite(suite_id):
    """A recorded suite id under its current id (the descriptor's `aliases`)."""
    return campaign().canonical_suite(suite_id)


def engine(name):
    """A recorded row name under its current engine name (the descriptor's `aliases`)."""
    return campaign().canonical_engine(name)

# The harness revision that first measured both sides of a vendor ratio on one contract: no
# allocation and no layout conversion inside the timed run, the vendored drivers in column order
# only, the owning colmat/faer routes labelled `api-inclusive` so they never pair with a vendor row
# (`fbench/src/op.rs`, `benchmarks/suites/colmat-kernels.yaml`). A page recorded before it holds
# ratios in which the vendor row paid for a copy or a workspace the colmat row did not, so those
# cells are printed as withdrawn rather than as a current comparison.
#
# It must be a commit of `main`, not of a branch: a squash rewrites the SHAs of the landed change
# (#75 landed as 695232b), and a constant left on a branch-only commit orphans it -- then no recorded
# harness can be placed after it and every vendor ratio is withdrawn. So the constant is re-pinned to
# the landed commit whenever the change that carries it lands by squash, `fair_harness` says so on
# stderr when the commit is not in the checkout, and `tests/test_comparison.py` fails when it is not
# on `main`'s history. Stated publicly in `docs/dataset-status.md` and `docs/reliable-measurement.md`.
FAIR_HARNESS = '695232be7c1dc43b56e518ae1b7ce813e045e005'

# The families whose vendor ratio a pre-fair harness could not measure on one contract. Its
# complement is the audit's usable list -- `gemm` (CBLAS in row order moves no data),
# `triangular_solve` and `householder_apply` called the same kind of primitive on both sides even
# then -- so a family is withdrawn here or it is named usable in
# `docs/measurement-audit-2026-11-06.md`, and the two cannot drift apart unnoticed.
WITHDRAWN_VENDOR_FAMILIES = frozenset({'lu', 'full_piv_lu', 'solve', 'cholesky', 'qr_factor',
                                       'qr', 'svd_values', 'svd_thin', 'eigh'})

# What makes two runs "the same host". A ratio taken across hosts is not a vendor
# comparison, so the identity is checked before anything is joined. Every published
# manifest carries the block; one that does not is refused rather than guessed at.
HOST_KEYS = ('hostname', 'cpu', 'arch', 'logical_cpus')

# What makes two runs "one measurement": one host is not enough, the numbers were also taken
# under a timing policy, under an idle guard and on a set of CPUs. A ratio between a 5-call
# median after 250 ms of priming and a single best-of call is not a vendor comparison even on
# one machine, and neither is a ratio between a run that waited 3 s for an idle machine and one
# that accepted 100% busy, so every field here is compared and every difference is named.
PROTOCOL_KEYS = ('version', 'repetitions', 'statistic', 'minimum_untimed_priming_ms')
GUARD_KEYS = ('idle_window_seconds', 'max_busy_percent', 'attempts', 'sibling_aware')
# The cell identity columns: what was measured. Two arms that disagree here never join on the
# cell key, which is exactly why they are checked -- a silent non-join renders as a dash and is
# then read as "that arm did not carry the cell".
GRID_KEYS = ('regime', 'family', 'order', 'm', 'n', 'k', 'batch')
GRID_VARY = ('dtype', 'threads')


def result_files(run, manifest):
    """The CSVs the manifest names: exactly what the validator and the recorded pages read.

    Not every `run*.csv` that happens to be in the directory -- a leftover copy would move a cell's
    value while nothing had re-measured it, and the recorded page would then disagree with this one.
    """
    listed = manifest.get('result_files') or []
    if not listed:
        sys.exit(f'ERROR: the run at {run} names no result files in run.yaml; a page built from the '
                 'directory would read CSVs nothing gated')
    missing = [name for name in listed if not (run / name).exists()]
    if missing:
        sys.exit(f'ERROR: the run at {run} is missing result files its manifest names: '
                 f'{", ".join(missing)}')
    return sorted(run / name for name in listed)


def read_sets(run_dir):
    """One arm's manifest, its per-cell/per-row values across process sets, and its skipped notes.

    The values stay a list, one entry per process set: a cell is measured in several sets and every
    set is part of the measurement, so the reduction to one number happens where it is stated and
    named -- `last_set` in `load`, the median and the between-start range in `control` -- and never
    by overwriting one set's value with the next set's.
    """
    run = pathlib.Path(run_dir)
    manifest = yaml.safe_load((run / 'run.yaml').read_text())
    sets, skips = {}, {}
    for path in result_files(run, manifest):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                key = tuple(row[k] for k in KEY)
                if row.get('status') != 'ok' or row.get('total_ms') in (None, ''):
                    if row.get('status') == 'skipped' and row.get('note'):
                        # Keyed without the harness note: a skipped row carries the reason in
                        # that column, so its key would never meet the measured cell's.
                        skips[(key[:-1], engine(row['row']))] = row['note']
                    continue
                sets.setdefault(key, {}).setdefault(engine(row['row']), []).append(float(row['total_ms']))
    return manifest, sets, skips


def control(previous_mkl, current_mkl):
    """The unchanged row between two recordings: median ratio and cells outside their range.

    upstream `faer` is built from the same commit in both, so it is the machine-state
    control the campaign's merge rule uses; this is the same statement for a publication.
    """
    _, before, _ = read_sets(previous_mkl)
    _, after, _ = read_sets(current_mkl)
    ratios, outside, n = [], 0, 0
    for key, rows in after.items():
        cur = rows.get(FAER)
        old = before.get(key, {}).get(FAER)
        if not cur or not old:
            continue
        prev, now = statistics.median(old), statistics.median(cur)
        if not prev:
            continue
        n += 1
        ratios.append(now / prev)
        if not (min(old) <= now <= max(old)):
            outside += 1
    if not n:
        return None
    return {'cells': n, 'median': statistics.median(ratios), 'outside': outside}


def last_set(values):
    """The value this page has always shown: the last process set the arm recorded."""
    return values[-1]


def load(run_dir):
    """The manifest, the measured values, and the reasons the arm skipped a cell.

    Both come from the same CSVs: a cell is missing from `cells` because the arm wrote no
    row for it, or because it wrote one with a status other than `ok`, and in the second
    case the note that says why is kept so the page can state it instead of showing a dash.

    The reduction over a run's process sets stays `last_set`: an existing page is a record of what
    was published, and this change must not rewrite its columns. The page states which reduction it
    applied.
    """
    manifest, sets, skips = read_sets(run_dir)
    cells = {key: {row: last_set(values) for row, values in rows.items()}
             for key, rows in sets.items()}
    return manifest, cells, skips


def host_identity(manifest, label):
    """The host a run was measured on, as the four fields its manifest records."""
    host = manifest.get('host') or {}
    missing = [k for k in HOST_KEYS if host.get(k) in (None, '')]
    if missing:
        sys.exit(f'ERROR: the {label} arm does not record its host ({", ".join(missing)} '
                 'missing from run.yaml); runs are only joined when they name one host')
    return tuple(str(host[k]) for k in HOST_KEYS)


def describe_host(identity):
    hostname, cpu, arch, cpus = identity
    return f'{hostname} ({cpu}, {arch}, {cpus} logical CPUs)'


def flatten(value):
    """A recorded block as one comparable string, so a nested field is compared whole."""
    if isinstance(value, dict):
        return '(' + ', '.join(f'{name}={flatten(item)}' for name, item in
                               sorted(value.items())) + ')'
    return str(value)


def protocol_identity(manifest, label):
    """The scalar fields that make two runs one measurement, as comparable strings.

    Beyond the timing block: the machine's power state (boost on and boost off are not
    comparable, docs/protocol-v2-baseline.md), the idle guard the run waited on (a run that
    accepted a busy machine is not a quieter neighbour's peer), the observer's affinity, and the
    revisions of the library, upstream, harness and toolkit -- a different harness measures
    different work, which no amount of equal timing constants repairs. The harness is its measured
    paths, not its commit: two commits that differ only outside `HARNESS_PATHS` are one harness
    (`harness_identity_same`), so `harness.commit` is the one field the join may waive.
    """
    timing = manifest.get('timing_policy') or {}
    guards = manifest.get('guards') or {}
    threads = manifest.get('threads') or {}
    fields = {f'timing_policy.{key}': timing.get(key) for key in PROTOCOL_KEYS}
    fields.update({f'guards.{key}': guards.get(key) for key in GUARD_KEYS})
    fields['threads.observer.affinity'] = (threads.get('observer') or {}).get('affinity')
    fields['host.power'] = (manifest.get('host') or {}).get('power')
    fields['library.commit'] = (manifest.get('library') or {}).get('commit')
    fields['upstream.commit'] = (manifest.get('upstream') or {}).get('commit')
    fields['harness.commit'] = (manifest.get('harness') or {}).get('commit')
    fields['toolkit.commit'] = (manifest.get('toolkit') or {}).get('commit')
    missing = [field for field, value in fields.items() if value in (None, '', ())]
    if missing:
        sys.exit(f'ERROR: the {label} arm does not record its measurement protocol '
                 f'({", ".join(missing)} missing from run.yaml); runs are only joined when they were '
                 'measured under one protocol')
    return {field: flatten(value) for field, value in fields.items()}


def harness_identity_same(before, after):
    """Whether two recorded harness commits built the same measured harness.

    A publication's arms can be recorded either side of a commit that only adds recorded data: the
    commits differ, the files the binary is built from do not. The rule is `candidate_compare`'s --
    its `HARNESS_PATHS` handed to the toolkit's tree comparison -- not a second definition. A
    checkout that cannot resolve one of the commits answers no (the toolkit turns a missing path
    into a difference), so the join is refused rather than certified by absence.
    """
    return not candidate_compare.candidate_module_same_code(before, after, ROOT)


def cpu_sets(manifest):
    """The thread count -> CPUs map an arm measured on."""
    return {str(count): str(cpus)
            for count, cpus in ((manifest.get('threads') or {}).get('cpu_sets') or {}).items()}


def budgets(arm):
    """The thread budgets an arm was asked for: its measured cells and its skipped rows."""
    skipped = [key for key, _row in arm['skips']]   # keyed without the note column
    return {key[KEY.index('threads')] for key in list(arm['cells']) + skipped}


def cell_grid(cells):
    """Per cell shape, the (dtype, threads) pairs an arm was asked for beside it."""
    grid = {}
    for key in cells:
        shape = tuple(value for value, name in zip(key, KEY) if name in GRID_KEYS)
        grid.setdefault(shape, set()).add(
            tuple(value for value, name in zip(key, KEY) if name in GRID_VARY))
    return grid


def run_grid(arm):
    """Every cell an arm was asked for -- measured or skipped -- keyed by its shape.

    A skipped row is a cell the arm was asked for, so it belongs to the grid: an arm asked for the
    f64 cell and skipping it must not pass as an arm that was asked for f32 instead.
    """
    skipped = [key for key, _row in arm['skips']]   # keyed without the note column
    return cell_grid(list(arm['cells']) + skipped)


def missing_budget_sets(arm):
    """The thread budgets an arm measured with no CPU set recorded for them."""
    recorded = cpu_sets(arm['manifest'])
    return sorted((count for count in budgets(arm) if count not in recorded),
                  key=lambda value: (not value.isdigit(), value.zfill(3)))


def unmeasured_cells(arms):
    """The cells an arm was asked for and no arm measured, and the families they belong to.

    They are in the cell grid -- every arm wrote a `skipped` row for them -- so they are part of
    what the run was asked for, and a page that silently dropped them would read as a run that
    was asked for fewer cells than it was. The reason itself stays in the manifest.
    """
    asked, measured = set(), set()
    for arm in arms:
        asked |= {key for key, _row in arm['skips']}
        measured |= {key[:-1] for key in arm['cells']}
    unmeasured = asked - measured
    return unmeasured, sorted({key[1] for key in unmeasured})


class PairingRefused(SystemExit):
    """Two arms were offered that are not one measurement session.

    A SystemExit subclass, so the CLI behaves exactly as before (message on stderr, non-zero exit)
    while a caller can tell a refusal to join -- which it may publish around -- from a run whose
    files or manifest are broken, which must abort the publication instead.
    """


def require_one_protocol(arms):
    """Refuse to join runs that were not measured the same way on the same machine.

    A page that prints one protocol above two arms is only true when host, protocol, measured CPUs
    and the cell grid are one. Two runs on two machines, or under two timing policies, would make
    it false while the table looked the same, so a difference is refused with every differing
    field named on both sides -- those are what would have to be re-measured for the comparison to
    be real. The harness revision is compared as its measured paths (`harness_identity_same`): two
    arms recorded either side of a data commit are one harness, a change to the harness is refused.
    """
    reference = arms[0]
    label = reference['label']
    identity = host_identity(reference['manifest'], label)
    protocol = protocol_identity(reference['manifest'], label)
    cpus = cpu_sets(reference['manifest'])
    grid = run_grid(reference)
    if not cpus:
        raise PairingRefused(f'ERROR: the {label} arm records no measured CPU sets in run.yaml; a '
                             'run is only joined when it says which CPUs it measured on')
    # A thread budget is measured on a CPU set, and the set is what a 1T or 8T column means. A
    # shared budget cannot stand in for it: the entry both arms share may be one neither of them
    # measured a cell with, while the budget they did measure has no recorded set at all.
    uncovered = [f'threads.cpu_sets[{count} thread]: the {arm["label"]} arm measured it but records '
                 'no CPU set for it' for arm in arms for count in missing_budget_sets(arm)]
    if uncovered:
        raise PairingRefused('ERROR: an arm does not say which CPUs it measured a thread budget on, '
                             'so its rows are not evidence:\n  ' + '\n  '.join(uncovered))
    for arm in arms[1:]:
        differences = []
        other_label = arm['label']
        other_host = host_identity(arm['manifest'], other_label)
        if other_host != identity:
            differences.append(f'host: {describe_host(identity)} vs {describe_host(other_host)}')
        other = protocol_identity(arm['manifest'], other_label)
        same_harness = (protocol['harness.commit'] == other['harness.commit']
                        or harness_identity_same(protocol['harness.commit'],
                                                 other['harness.commit']))
        # A harness change under a measured path is still a difference: the commits are one identity
        # only when every path the binary is built from is identical.
        differences += [f'{field}: {label} {protocol[field]} vs {other_label} {other[field]}'
                        for field in protocol if protocol[field] != other[field]
                        and not (field == 'harness.commit' and same_harness)]
        if same_harness and protocol['harness.commit'] != other['harness.commit']:
            print(f"harness commits {protocol['harness.commit'][:12]} ({label}) and "
                  f"{other['harness.commit'][:12]} ({other_label}) carry identical build inputs "
                  f"({', '.join(candidate_compare.HARNESS_PATHS)}); the difference is outside the "
                  "measured harness")
        other_cpus = cpu_sets(arm['manifest'])
        # Only the budgets both arms were asked for, and both recorded (above): a leftover entry for
        # a thread count neither arm measured says nothing about these numbers, and the budget both
        # measured has to be the same CPUs or "one page, two arms" compares two thread counts.
        shared = budgets(reference) & budgets(arm)
        if not shared:
            differences.append('threads.cpu_sets: no thread budget measured by both arms')
        for count in sorted(shared, key=lambda value: (not value.isdigit(), value.zfill(3))):
            if cpus[count] != other_cpus[count]:
                differences.append(f'threads.cpu_sets[{count} thread]: {label} {cpus[count]} '
                                   f'vs {other_label} {other_cpus[count]}')
        other_grid = run_grid(arm)
        # The dtype is the part of the cell identity that cannot be "a smaller grid": an arm asked
        # for f32 where the other was asked for f64 measured a different cell, and the dash it leaves
        # is read as "that arm had no driver for it". A thread budget one arm was not asked for is
        # a smaller grid (that is what a partial run is) and is checked above, per measured budget,
        # not refused here. A family only one arm was asked for is a smaller grid too.
        for shape, expected in sorted(grid.items()):
            if shape not in other_grid:
                continue
            mine = {pair[0] for pair in expected}
            wanted = {pair[0] for pair in other_grid[shape]}
            if mine != wanted:
                differences.append(
                    f'cell {shape[1]} {shape[2]}-order {shape[3]}x{shape[4]} '
                    f'{shape[6]}-batch dtype: {label} {sorted(mine)} '
                    f'vs {other_label} {sorted(wanted)}')
        if differences:
            raise PairingRefused(
                f'ERROR: the {label} arm and the {other_label} arm were not measured under one '
                'protocol on one host, so their rows are not one session:\n  '
                + '\n  '.join(differences))
    return identity


def vendor_arm(label, run_dir):
    """One arm's vendor row: the library it stands for, the CSV row it writes, what it measured.

    The library name and therefore the row name (`lapack-<name>`) are read from the manifest, and
    the manifest has to name the arm it was handed to: the column headings this page prints are
    `MKL`, `OpenBLAS` and `AOCL`, so an arm given to `--openblas` that recorded `mkl` would show MKL's
    numbers under the OpenBLAS heading -- the one thing a page of measurements must not do.
    """
    manifest, cells, skips = load(run_dir)
    name = str((manifest.get('vendor') or {}).get('name') or '')
    if not name:
        sys.exit(f'ERROR: the {label} arm does not name its vendor library in run.yaml')
    if name != label:
        sys.exit(f'ERROR: the arm given to --{label} records its vendor as {name!r} in run.yaml, so '
                 f'its rows are `lapack-{name}` and not the `lapack-{label}` rows this page labels; '
                 'the two arms are not interchangeable')
    return dict(label=label, name=name, row=f'lapack-{name}',
                manifest=manifest, cells=cells, skips=skips)


def git(*args):
    """The return code of one git question, or None when this host cannot ask it."""
    try:
        return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True).returncode
    except OSError:
        return None


def fair_harness(commit):
    """Whether the harness that recorded an arm measured both sides of a vendor ratio on one contract.

    At or after `FAIR_HARNESS` is fair. The question is put to this repository's history, because
    every recorded `harness.commit` is a commit of it; when the history cannot answer -- a shallow
    clone, or a commit it does not hold -- the answer is no, so a checkout that cannot tell withdraws
    a ratio instead of printing a one-sided one as current. `FAIR_HARNESS` itself has to be on that
    history -- it is a commit of `main`, re-pinned to the landed commit when the campaign squashes the
    change that carries it; when it is not, nothing can be certified and the withdrawal is stated on
    stderr with the one-line fix, so an orphaned constant does not turn into a silent withdrawal.
    """
    if not commit:
        return False
    if commit == FAIR_HARNESS:
        return True
    if git('cat-file', '-e', FAIR_HARNESS + '^{commit}') != 0:
        print(f'WARNING: {FAIR_HARNESS} (FAIR_HARNESS) is not a commit this checkout holds -- a '
              'shallow clone, or the fairness change was squashed or rebased away. Every vendor ratio '
              'is withdrawn. Keep that commit on the history of main, or re-pin FAIR_HARNESS to the '
              'landed commit.', file=sys.stderr)
        return False
    return git('merge-base', '--is-ancestor', FAIR_HARNESS, commit) == 0


def fmt(x):
    if x is None:
        return '-'
    if x >= 100:
        return f'{x:.1f}'
    if x >= 10:
        return f'{x:.2f}'
    if x >= 1:
        return f'{x:.3f}'
    return f'{x:.4g}'


def ratio(a, b):
    return a / b if a and b else None


def cls(r):
    if r is None:
        return ''
    if r <= 0.95:
        return ' class="win"'
    if r >= 1.05:
        return ' class="loss"'
    return ''


def vendor_ratio_cell(r, field):
    """A vendor ratio cell: the ratio, or `withdrawn` when the harness could not measure it fairly."""
    if r['withdrawn']:
        return '<td class="wd">withdrawn</td>'
    return f'<td{cls(r[field])}>{fmt(r[field])}</td>'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mkl', required=True, help='run directory of the three-engine arm')
    ap.add_argument('--openblas', required=True, help='run directory of the two-engine arm')
    ap.add_argument('--aocl', help='run directory of the AOCL arm (AMD profiles); joined as a third arm')
    ap.add_argument('--out', required=True)
    ap.add_argument('--control-mkl', help='an earlier MKL arm to state the unchanged row against')
    args = ap.parse_args(argv)

    mkl = vendor_arm('mkl', args.mkl)
    ctrl = control(args.control_mkl, args.mkl) if args.control_mkl else None
    ob = vendor_arm('openblas', args.openblas)
    ao = vendor_arm('aocl', args.aocl) if args.aocl else None
    arms = [mkl, ob] + ([ao] if ao else [])
    identity = require_one_protocol(arms)
    # `require_one_protocol` accepts a pair only when one harness measured both arms: the same
    # commit, or two commits whose measured harness paths are identical. So whether these ratios
    # were measured on one contract is asked once, of the MKL arm's manifest.
    harness = (mkl['manifest'].get('harness') or {}).get('commit')
    fair = fair_harness(harness)

    keys = sorted(set(mkl['cells']) | set(ob['cells']) | (set(ao['cells']) if ao else set()))
    rows = []
    for key in keys:
        a, b = mkl['cells'].get(key, {}), ob['cells'].get(key, {})
        c = ao['cells'].get(key, {}) if ao else {}
        fork_m, fork_o, fork_a = a.get(COLMAT), b.get(COLMAT), c.get(COLMAT)
        aoc = c.get(ao['row']) if ao else None
        up, mk, obl = a.get(FAER), a.get(mkl['row']), b.get(ob['row'])
        # Each ratio is two rows of one arm, and both its numbers are in the row it is printed in:
        # `upstream/fork` and `MKL/fork` divide the fork the MKL arm measured, `OpenBLAS/fork` the
        # one the OpenBLAS arm measured. Dividing by a fork another arm measured would print a
        # ratio whose denominator the reader cannot see.
        # A ratio the harness could not measure on one contract is not printed: the vendor row paid
        # inside its timed call for work the colmat row did not, so the two numbers are kept as
        # recorded and the comparison between them is withdrawn.
        withdrawn = not fair and key[1] in WITHDRAWN_VENDOR_FAMILIES
        rows.append(dict(key=key, fork_m=fork_m, fork_o=fork_o, fork_a=fork_a, up=up, mk=mk,
                         obl=obl, aoc=aoc, r_up=ratio(up, fork_m), withdrawn=withdrawn,
                         r_ob=None if withdrawn else ratio(obl, fork_o),
                         r_mk=None if withdrawn else ratio(mk, fork_m),
                         r_ao=None if withdrawn else ratio(aoc, fork_a)))

    # summary over the cells each arm carries: that is where its baseline row exists
    mkl_pairs = [r for r in rows if r['fork_m'] and r['up']]
    ob_pairs = [r for r in rows if r['fork_o'] and r['r_ob']]
    med_up = statistics.median([r['up'] / r['fork_m'] for r in mkl_pairs]) if mkl_pairs else None
    med_ob = statistics.median([r['obl'] / r['fork_o'] for r in ob_pairs]) if ob_pairs else None
    ao_pairs = [r for r in rows if r['fork_a'] and r['r_ao']]
    med_ao = statistics.median([r['aoc'] / r['fork_a'] for r in ao_pairs]) if ao_pairs else None
    ao_rows = [r for r in rows if r['fork_a'] and r['aoc']]
    unmeasured, unmeasured_families = unmeasured_cells(arms)
    # The cells whose ratio is actually on the page: a family without a vendor row in this
    # recording has no ratio to withdraw, and the note must not claim otherwise.
    withdrawn_cells = [r for r in rows if r['withdrawn'] and (r['mk'] or r['obl'] or r['aoc'])]
    ob_rows = [r for r in rows if r['fork_o'] and r['obl']]   # carries a vendor row, withdrawn or not
    ob_families = sorted({r['key'][1] for r in ob_pairs})

    withdrawn_html = ''
    if withdrawn_cells:
        families = sorted({r['key'][1] for r in withdrawn_cells})
        withdrawn_html = (
            '<p class="withdrawn"><strong>Withdrawn vendor comparison.</strong> The MKL arm was recorded '
            f'by harness <code>{html.escape(str(harness)[:12])}</code>, which is not at or '
            f'after the harness that charged both sides of a vendor ratio the same contract '
            f'(<code>{FAIR_HARNESS[:12]}</code>): it predates it, or this checkout cannot place it in '
            f'the history. '
            'The vendor ratio cells of '
            f'<code>{html.escape(", ".join(families))}</code> say <em>withdrawn</em>: the vendor row '
            'paid inside its timed call for a workspace, an output copy or a layout change that the '
            'colmat row did not, so dividing the two numbers measures the route, not the library. The '
            'times stay as recorded and are left out of the median. '
            '<a href="https://github.com/tensor4all/colmat-benchmark/blob/main/docs/dataset-status.md">'
            'docs/dataset-status.md</a> is the status of record and '
            '<code>docs/measurement-audit-2026-11-06.md</code> the finding-by-finding resolution; a '
            'page recorded at or after the fairness harness prints these ratios again.</p>\n'
            f'<p class="note">{len(withdrawn_cells)} of {len(rows)} cells carry a withdrawn vendor '
            'ratio and are not in the OpenBLAS median below'
            + (f', which covers <code>{html.escape(", ".join(ob_families))}</code>' if ob_families else '')
            + '.</p>\n')

    control_html = ''
    if ctrl:
        control_html = (
            '<p class="note"><strong>Control.</strong> upstream <code>faer</code> is the same '
            'commit in both recordings, so it measures the machine rather than the change: '
            f'between <code>{mkl["manifest"]["upstream"]["commit"][:12]}</code> on the previous arm and this one, '
            f'over {ctrl["cells"]} cells its median ratio is <strong>{fmt(ctrl["median"])}</strong> '
            f'and <strong>{ctrl["outside"]}</strong> cells fall outside their between-start range. '
            'The campaign\'s rule reads the median (within 2% means the machine reproduced) and '
            'reports the tail rather than refusing on it: a range taken from three process sets is '
            'narrow, so a tail of this size at these timings is expected, and it is the median that '
            'says the two recordings are comparable.</p>')

    library = mkl['manifest']
    revision = library['library']['commit']
    # Rendered from the manifest rather than written into the prose: two arms can agree on a v1
    # best-of-1 policy, and the page must not then say "median of five calls".
    timing = library['timing_policy']
    guards = library['guards']
    calls = f"{timing['repetitions']} call{'s' if timing['repetitions'] != 1 else ''}"
    reduction = "last of the run's process-set CSVs"
    title = (f"{current_suite(library['suite_id'])} on {library['target_profile']} - colmat against faer, OpenBLAS"
             + (", MKL and AOCL" if ao else " and MKL"))
    # The arms can name two harness commits (one identity, see `harness_identity_same`); printing
    # only the MKL arm's would describe the page with a revision only one arm was recorded at.
    ob_harness = (ob['manifest'].get('harness') or {}).get('commit')
    harness_text = (f"<code>{harness[:12]}</code>" if ob_harness == harness else
                    f"<code>{harness[:12]}</code> (MKL arm) and <code>{ob_harness[:12]}</code> "
                    "(OpenBLAS arm), which carry identical measured harness paths")
    if ao:
        ao_harness = (ao['manifest'].get('harness') or {}).get('commit')
        if ao_harness != harness:
            harness_text += (f"; AOCL arm <code>{str(ao_harness)[:12]}</code>, identical measured "
                             "harness paths")
    ao_vendor = ''
    ao_recorded = ''
    if ao:
        v = ao['manifest']['vendor']
        ao_vendor = (f"\n <li>AOCL: {html.escape(str(v.get('name')))} {html.escape(str(v.get('version')))} "
                     f"({html.escape(str(v.get('linkage')))}; kernels {html.escape(str(v.get('corename')))})</li>")
        ao_recorded = f" and <code>{ao['manifest']['timestamp']}</code> (AOCL arm)"
    parts = [f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
 body {{ font: 14px/1.5 -apple-system, "Segoe UI", Roboto, sans-serif; margin: 2rem auto; max-width: 1200px; padding: 0 1rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: .6rem 0 1.5rem; font-variant-numeric: tabular-nums; }}
 th, td {{ border-bottom: 1px solid #e3e3e3; padding: .25rem .4rem; text-align: right; white-space: nowrap; }}
 th {{ background: #f5f5f5; font-weight: 600; position: sticky; top: 0; }}
 td.k, th.k {{ text-align: left; }}
 td.win {{ background: #e7f6e7; }} td.loss {{ background: #fbe9e9; }}
 td.wd {{ color: #888; font-style: italic; }} p.withdrawn {{ border-left: 3px solid #c80; background: #fdf7e8; padding: .5rem .8rem; max-width: 70rem; }}
 .note {{ color: #555; max-width: 70rem; }} code {{ background: #f2f2f2; padding: 0 .2rem; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="note">Every cell below was measured in one binary beside one vendor library. The
<code>colmat</code> column is the measured revision, and it is shown per arm
(<code>colmat MKL</code>, <code>colmat OpenBLAS</code>) because each arm measured it in its own
process; <code>faer</code> is the same upstream commit built into the MKL arm as the
unchanged row; the vendor columns are the vendor arms from their own runs. Every ratio divides by
the colmat value printed beside it, in the same arm. Times are milliseconds, the recorded
{timing['statistic']} of {calls} after {timing['minimum_untimed_priming_ms']} ms of untimed
priming, reduced per cell to the {reduction}. Ratios are <em>baseline over colmat</em>, so above 1.00
colmat is faster; the cells outside 0.95-1.05 are tinted.</p>
<ul>
 <li>measured revision: <code>{revision}</code> ({html.escape(str(library['library'].get('version')))})</li>
 <li>upstream: <code>{library['upstream']['commit']}</code> ({html.escape(str(library['upstream'].get('ref')))})</li>
 <li>MKL: {html.escape(str(mkl['manifest']['vendor'].get('name')))} {html.escape(str(mkl['manifest']['vendor'].get('version')))} ({html.escape(str(mkl['manifest']['vendor'].get('linkage')))})</li>
 <li>OpenBLAS: {html.escape(str(ob['manifest']['vendor'].get('name')))} {html.escape(str(ob['manifest']['vendor'].get('version')))} ({html.escape(str(ob['manifest']['vendor'].get('linkage')))})</li>{ao_vendor}
 <li>hardware profile: <code>{library['target_profile']}</code>; measured CPUs {html.escape(str(library['threads']['cpu_sets']))}; observer {html.escape(str(library['threads']['observer'].get('affinity')))}; one host: {html.escape(describe_host(identity))}</li>
 <li>timing policy: v{timing['version']}, {calls}, {timing['statistic']}, priming {timing['minimum_untimed_priming_ms']} ms; guards: idle {guards['idle_window_seconds']} s, max {guards['max_busy_percent']}% busy, {guards['attempts']} attempts</li>
 <li>recorded: <code>{library['timestamp']}</code> (MKL arm) and <code>{ob['manifest']['timestamp']}</code> (OpenBLAS arm){ao_recorded}</li>
 <li>harness {harness_text}, toolkit <code>{library['toolkit']['commit'][:12]}</code></li>
</ul>
{withdrawn_html}
{control_html}
<p class="note">The two arms are joined on one host under one measurement protocol: the host, the
machine's power state, the idle guard, the timing policy, the CPU set of every thread budget each
arm measured and the cell grid are compared, and a run that differs in any of them is refused
rather than printed above this table. The arms of a publication are recorded hours apart, so the two
timestamps differ by design and are not compared.</p>
"""]

    if unmeasured:
        parts.append(
            f'<p class="note">{len(unmeasured)} of {len(unmeasured) + len(rows)} cells were asked '
            'for by at least one arm and measured by neither, so they carry no row below. They are '
            f'the families <code>{html.escape(", ".join(unmeasured_families))}</code>; each arm\'s '
            'own page and manifest state why.</p>\n')

    parts.append(
        '<p class="note">This page is a join, not a control comparison: the control statement (the unchanged\n'
        'row\'s ratio against a baseline run) lives in each arm\'s own page, and the verification record in its\n'
        'manifest. A dash means that arm did not carry the cell.</p>\n'
        f'<p class="note">Cells with a baseline row: {len(mkl_pairs)} against <code>faer</code>, '
        f'{len(ob_rows)} against <code>lapack-openblas</code>. Median baseline/colmat ratio (above 1.00 '
        f'means colmat is faster): faer {fmt(med_up)}, '
        f'OpenBLAS {fmt(med_ob) if ob_pairs else "withdrawn" if withdrawn_cells else "-"}'
        + (f', AOCL {fmt(med_ao) if ao_pairs else "withdrawn" if withdrawn_cells else "-"} '
           f'({len(ao_rows)} cells against <code>lapack-aocl</code>)' if ao else '')
        + '.</p>\n')

    by_regime = {}
    for r in rows:
        by_regime.setdefault(r['key'][0], []).append(r)
    for regime in sorted(by_regime):
        parts.append(f'<h2>{html.escape(regime)}</h2>\n<table>\n<thead><tr>'
                     '<th class="k">family</th><th class="k">shape</th><th>dtype</th><th>order</th>'
                     '<th>threads</th>'
                     '<th>colmat MKL (ms)</th><th>faer (ms)</th><th>faer/colmat</th>'
                     '<th>colmat OpenBLAS (ms)</th><th>OpenBLAS (ms)</th><th>OpenBLAS/colmat</th>'
                     '<th>MKL (ms)</th><th>MKL/colmat</th>'
                     + ('<th>colmat AOCL (ms)</th><th>AOCL (ms)</th><th>AOCL/colmat</th>' if ao else '')
                     + '</tr></thead><tbody>')
        for r in sorted(by_regime[regime], key=lambda r: (
                r['key'][1], r['key'][2], r['key'][3],
                int(r['key'][8]) if r['key'][8].isdigit() else 0,
                int(r['key'][4]), int(r['key'][5]), r['key'][9])):
            key = r['key']
            shape = key[4] + 'x' + key[5] + ('x' + key[6] if key[6] not in ('', '0') else '')
            label = f' {key[9]}' if key[9] else ''
            parts.append(
                f'<tr><td class="k">{html.escape(key[1])}</td>'
                f'<td class="k">{html.escape(shape)}{html.escape(label)}</td>'
                f'<td>{key[2]}</td><td>{key[3]}</td><td>{key[8]}</td>'
                f'<td>{fmt(r["fork_m"])}</td><td>{fmt(r["up"])}</td><td{cls(r["r_up"])}>{fmt(r["r_up"])}</td>'
                f'<td>{fmt(r["fork_o"])}</td><td>{fmt(r["obl"])}</td>{vendor_ratio_cell(r, "r_ob")}'
                f'<td>{fmt(r["mk"])}</td>{vendor_ratio_cell(r, "r_mk")}'
                + (f'<td>{fmt(r["fork_a"])}</td><td>{fmt(r["aoc"])}</td>{vendor_ratio_cell(r, "r_ao")}'
                   if ao else '')
                + '</tr>')
        parts.append('</tbody></table>')

    parts.append('</body></html>\n')
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(parts))
    print(f'wrote {out} from {len(rows)} cells '
          f'({len(mkl_pairs)} with an upstream row, {len(ob_pairs)} with an OpenBLAS row'
          + (f', {len(ao_pairs)} with an AOCL row' if ao else '') + ')')
    if unmeasured:
        print(f'{len(unmeasured)} cells were asked for and measured by neither arm')
    return 0


if __name__ == '__main__':
    sys.exit(main())
