#!/usr/bin/env python3
"""Win share against the fastest vendor arm, per hardware and per family.

    python3 scripts/family_matrix.py            # print what the landing page shows

The landing page (`scripts/gen_site.py`, #116) is built from this table, and the cross-host
vendor-loss table (#108) is meant to read the same function, so the two never disagree on a number.

A *cell* is one recorded case of a page: one family, shape, dtype and thread count, in F order
(the order the vendor rows are recorded in for every family), f64 or c64, every dimension at least
`MIN_N`, batch 1, the kernel route (not the `api-inclusive` one). For every cell some arm measured
with both a colmat row and its vendor row, the vendor row with the lowest median is the *fastest
vendor*, and colmat is compared with it inside that arm's own process:

- the verdict is the toolkit's (`report_run.comparison_verdict`, the rule behind the `vs` columns
  of every recorded page): a *win* or a *loss* is a difference larger than the between-start
  spread the run recorded for the two rows, floored at 5%; anything inside that band is a *tie*;
- a cell whose rows have fewer than two process sets has no spread to compare against and is
  counted as `no_spread`, not as a tie, and never enters a share;
- a cell recorded by a harness that charged the vendor arms more than colmat (#74,
  `gen_comparison.fair_harness`) is `withdrawn` and never enters a share either.

A family a machine's arms never compared with a vendor is absent from that machine's counts: the
page prints "not measured" for it, never a share.

A *machine* is one profile, with one exception: a `<profile>-os-shared` profile is the same
machine measured with no CPU left to the observing side, declared not comparable with the base
profile's rows (#119 merges the two). Its pages are listed under the base machine but excluded
from every share.
"""
import pathlib
import re
import statistics
import subprocess
import sys

import yaml

from campaign_toolkit import descriptor, gen_index, report_run
from campaign_toolkit.campaign import load
from campaign_toolkit.stack import rev_file_value

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_comparison  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_N = 256
DTYPES = ('f64', 'c64')
ORDER = 'f'
COLMAT = gen_comparison.COLMAT
VENDOR_ROWS = {'mkl': 'lapack-mkl', 'openblas': 'lapack-openblas', 'aocl': 'lapack-aocl'}
VENDOR_LABELS = {'mkl': 'MKL', 'openblas': 'OpenBLAS', 'aocl': 'AOCL'}
# AOCL is AMD's own LAPACK stack, an AMD-host arm (`benchmarks/campaign.yaml`, #95): on another
# vendor's CPU its absence is not a gap in the comparison.
AMD_ONLY = {'aocl'}
OS_SHARED_SUFFIX = '-os-shared'
# The issue's order (gemm, lu, cholesky, qr, svd, eigh); a family not listed follows by name.
FAMILY_ORDER = ('gemm', 'lu', 'cholesky', 'solve', 'qr_factor', 'qr', 'eigh', 'svd_thin',
                'svd_values', 'full_piv_lu', 'triangular_solve', 'householder_apply')
# The microarchitecture and SIMD width of the CPUs the campaign has recorded on, keyed by the
# model string the manifests record. A CPU missing here is printed by its model string alone.
ARCHITECTURES = (
    (r'Ryzen AI 9 HX 470', 'AMD Zen 5', 'AVX-512'),
    (r'EPYC 7702', 'AMD Zen 2', 'AVX2'),
    (r'Xeon Platinum 8280', 'Intel Cascade Lake', 'AVX-512'),
)
# "mostly" means more than half of the machine's decided cells of that family.
MOSTLY = 0.5
RESULTS = ('win', 'tie', 'lose', 'no_spread', 'withdrawn')


def vendor_rank(vendor):
    """MKL, OpenBLAS, AOCL, then anything else: the order arms are listed and tried in."""
    return list(VENDOR_ROWS).index(vendor) if vendor in VENDOR_ROWS else len(VENDOR_ROWS)


def family_order(families):
    known = [f for f in FAMILY_ORDER if f in families]
    return known + sorted(set(families) - set(known))


def architecture(cpu):
    """('AMD Zen 5', 'AVX-512', 'Ryzen AI 9 HX 470'): the CPU model without its vendor prefix."""
    cpu = cpu or ''
    model = re.sub(r'\((R|TM)\)', '', cpu)
    model = re.sub(r'^(AMD|Intel)\s+', '', model).strip()
    for pattern, arch, simd in ARCHITECTURES:
        if re.search(pattern, cpu):
            return arch, simd, model
    return model, None, model


def profile_declarations():
    """{profile: declared fields} from `benchmarks/profiles.yaml`, reading the toolkit's shared
    profile for an entry that only names one."""
    d = descriptor.set_root(ROOT)
    shared_dir = pathlib.Path(descriptor.__file__).parent / 'profiles'
    out = {}
    for entry in yaml.safe_load(pathlib.Path(d.profiles_file).read_text())['profiles']:
        if isinstance(entry, str):
            shared = shared_dir / f'{entry}.yaml'
            data = yaml.safe_load(shared.read_text()) if shared.exists() else {}
            out[entry] = (data.get('profiles') or [data])[0] if isinstance(data, dict) else {}
        else:
            out[entry['name']] = entry
    return out


def cores(declared):
    """Physical cores: the declared logical CPUs over the declared threads per core."""
    logical = declared.get('logical_cpus')
    per_core = (declared.get('smt') or {}).get('threads_per_core') or 1
    return logical // per_core if logical else None


class Arm:
    """One page's run: its manifest and its comparable cells."""

    def __init__(self, cell, page):
        self.suite, self.vendor, self.profile = cell['suite'], cell['vendor'], cell['profile']
        self.page = page['page']
        self.date, self.commit = page['date'], page['library_commit']
        self.below_baseline = page.get('below_baseline') or []
        self.coverage = page.get('coverage')
        self.run_dir = pathlib.Path(page['run_dir'])
        self.manifest = load(self.run_dir / 'run.yaml')
        rows = report_run.read_rows([self.run_dir / f for f in self.manifest['result_files']])
        self.best, _, self.within, _ = report_run.best_per_case(rows, self.manifest['timing_policy'])
        self.fair = gen_comparison.fair_harness(self.manifest['harness']['commit'])
        self.vendor_row = VENDOR_ROWS.get(self.vendor)
        # (family, shape, dtype, threads) -> (best key, {current engine name: recorded row name})
        self.cells = {}
        axes = report_run.report_axes(rows, self.manifest['timing_policy'])
        for row in rows:
            if not comparable(row):
                continue
            key = (row['regime'], report_run.case_name(row, variants=True), row['dtype'],
                   *(row[a] for a in axes), int(row['threads']))
            shape = (int(row['m']), int(row['n']), row.get('k') or '')
            at = (row['family'], shape, row['dtype'], int(row['threads']))
            self.cells.setdefault(at, (key, {}))[1][gen_comparison.engine(row['row'])] = row['row']

    def median(self, at, engine):
        key, names = self.cells[at]
        values = self.best.get(key, {}).get(names.get(engine))
        return statistics.median(values) if values else None

    def compares(self, at):
        names = self.cells[at][1]
        return COLMAT in names and self.vendor_row in names

    def verdict(self, at):
        """colmat against this arm's vendor row: (ratio, band), False inside the band, None
        without a spread to compare against."""
        key, names = self.cells[at]
        return report_run.comparison_verdict(
            self.best, self.within, key, [names[COLMAT], names[self.vendor_row]]
        ).get(names[self.vendor_row])


def comparable(row):
    """A row the win share counts: see the module docstring."""
    if (row['status'] != 'ok' or row['dtype'] not in DTYPES or row.get('order') != ORDER
            or row['batch'] != '1' or row.get('note')):
        return False
    dims = [row['m'], row['n']] + ([row['k']] if row.get('k') else [])
    return all(int(d) >= MIN_N for d in dims)


def outcome(verdict):
    """'win' | 'tie' | 'lose' | 'no_spread', colmat as the subject."""
    if verdict is None:
        return 'no_spread'
    if verdict is False:
        return 'tie'
    ratio, _ = verdict          # colmat time / vendor time
    return 'lose' if ratio > 1 else 'win'


def tally(arms):
    """({family: counts}, [vendors compared]) over the cells any of `arms` compared with a vendor.

    counts holds one integer per `RESULTS` entry and `cells`, the decided cells with the arm and
    the two medians each verdict came from.
    """
    keys = sorted({at for arm in arms for at in arm.cells if arm.compares(at)}, key=str)
    families, compared = {}, set()
    for at in keys:
        candidates = [(arm.median(at, arm.vendor_row), arm) for arm in arms
                      if at in arm.cells and arm.compares(at)]
        candidates = [(ms, arm) for ms, arm in candidates if ms]
        if not candidates:
            continue
        ms, fastest = min(candidates, key=lambda c: (c[0], vendor_rank(c[1].vendor), c[1].suite))
        compared.update(arm.vendor for _, arm in candidates)
        counts = families.setdefault(at[0], {**{r: 0 for r in RESULTS}, 'cells': []})
        result = 'withdrawn' if not fastest.fair else outcome(fastest.verdict(at))
        counts[result] += 1
        counts['cells'].append({'at': at, 'vendor': fastest.vendor, 'result': result,
                                'colmat_ms': fastest.median(at, COLMAT), 'vendor_ms': ms,
                                'page': fastest.page})
    return families, sorted(compared, key=vendor_rank)


def total(families):
    return {r: sum(c[r] for c in families.values()) for r in RESULTS}


def share(counts):
    """(win, tie, lose) fractions of the decided cells, or None when no cell was decided."""
    decided = counts['win'] + counts['tie'] + counts['lose']
    if not decided:
        return None
    return counts['win'] / decided, counts['tie'] / decided, counts['lose'] / decided


def leaning(counts):
    """'win' | 'lose' | 'tie' | 'mixed' | None: which way more than half of the decided cells went."""
    s = share(counts)
    if s is None:
        return None
    for name, value in zip(('win', 'tie', 'lose'), s):
        if value > MOSTLY:
            return name
    return 'mixed'


def behind_pin(commit, pin_rev):
    """How many commits `commit` is behind the pinned revision in extern/colmat (0 at the pin)."""
    if commit[:12] == pin_rev[:12]:
        return 0
    out = subprocess.run(['git', '-C', str(ROOT / 'extern' / 'colmat'), 'rev-list', '--count',
                          f'{commit}..{pin_rev}'], capture_output=True, text=True)
    return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None


def collect():
    """(profile order, {profile: [(cell, page)]}, history).

    The pages are the current ones (`gen_index.split_pages`, the index's own rule). A profile with
    no current page but with recordings in the history is still a recorded machine: its newest
    recording stands for it (one page per suite and arm, all at that recording's revision), and the
    page says how far behind the pin that is instead of hiding it.
    """
    descriptor.set_root(ROOT)
    found = gen_index.collect()
    current, history = gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
    order = list(found['profiles'])
    for cell in current:
        if cell['profile'] not in order:
            order.append(cell['profile'])
    pages = {}
    for cell in current:
        if cell['page'] is not None:
            pages.setdefault(cell['profile'], []).append((cell, cell['page']))
    older = {}
    for row in history:          # newest first
        if row['profile'] in pages:
            continue
        group = older.setdefault(row['profile'], [])
        if group and group[0][1]['library_commit'] != row['library_commit']:
            continue
        if not any(c['suite'] == row['suite'] and c['vendor'] == row['vendor'] for c, _ in group):
            group.append(({'suite': row['suite'], 'vendor': row['vendor'],
                           'profile': row['profile']}, row))
    pages.update(older)
    return order, pages, history


def machines(order=None, pages=None):
    """One entry per recorded machine, in the descriptor's profile order (see the docstring)."""
    if order is None:
        order, pages, _ = collect()
    declared = profile_declarations()
    pin = rev_file_value((ROOT / 'pins' / 'colmat.rev').read_text())
    features = list(descriptor.current().measured.get('features') or VENDOR_ROWS)
    out = []
    for profile in order:
        base = profile[:-len(OS_SHARED_SUFFIX)] if profile.endswith(OS_SHARED_SUFFIX) else None
        if profile not in pages or (base and base in pages):
            continue
        arms = [Arm(cell, page) for cell, page in pages[profile]]
        excluded = [Arm(cell, page) for cell, page in pages.get(profile + OS_SHARED_SUFFIX, [])]
        families, compared = tally(arms)
        host = arms[0].manifest['host']
        arch, simd, model = architecture(host.get('cpu'))
        amd = 'AMD' in (host.get('cpu') or '')
        missing = [v for v in features if v not in compared and (amd or v not in AMD_ONLY)]
        newest = max(arms, key=lambda a: a.date)
        out.append({
            'profile': profile, 'arch': arch, 'simd': simd, 'model': model,
            'cores': cores(declared.get(profile, {})) or host.get('logical_cpus'),
            'arms': arms, 'excluded': excluded, 'families': families, 'compared': compared,
            'missing': sorted(missing, key=vendor_rank),
            'threads': sorted({t for a in arms for t in a.manifest['threads']['counts']}),
            'date': newest.date, 'commit': newest.commit, 'behind': behind_pin(newest.commit, pin),
            'below_baseline': sorted({b for a in arms for b in a.below_baseline}),
            'provisional': any(a.manifest['host'].get('role', 'development') != 'measurement'
                               for a in arms),
            'total': total(families),
        })
    return out


def unrecorded(order=None, pages=None):
    """The declared profiles with no recording at all, with what they declare about the machine."""
    if order is None:
        order, pages, _ = collect()
    declared = profile_declarations()
    out = []
    for profile in order:
        if profile in pages or profile.endswith(OS_SHARED_SUFFIX):
            continue
        decl = declared.get(profile, {})
        arch, simd, model = architecture(decl.get('cpu'))
        out.append({'profile': profile, 'arch': arch, 'simd': simd, 'model': model,
                    'cores': cores(decl)})
    return out


def across_pins(history, machine_list):
    """{profile: [{'commit', 'date', 'counts' or None, 'why'}]}, oldest first: the win share at
    every pin a machine was recorded at. A pin with no decided cell has no share and says why."""
    out = {}
    for m in machine_list:
        groups = {}
        for row in history:
            if row['profile'] == m['profile'] and row['library_commit'] != m['commit']:
                groups.setdefault(row['library_commit'], []).append(row)
        points = []
        for commit, rows in groups.items():
            arms = [Arm({'suite': r['suite'], 'vendor': r['vendor'], 'profile': r['profile']}, r)
                    for r in rows]
            counts = total(tally(arms)[0])
            decided = share(counts) is not None
            why = None if decided else ('vendor rows withdrawn (recorded before the fair harness)'
                                        if counts['withdrawn'] else 'no comparable cell')
            points.append({'commit': commit, 'date': max(r['date'] for r in rows),
                           'counts': counts if decided else None, 'why': why})
        points.append({'commit': m['commit'], 'date': m['date'], 'counts': m['total'], 'why': None})
        out[m['profile']] = sorted(points, key=lambda p: p['date'])
    return out


def main():
    found = machines()
    every = family_order({f for m in found for f in m['families']})
    for m in found:
        t, s = m['total'], share(m['total'])
        print(f"{m['profile']:12} {m['arch']} ({m['simd']}) · {m['model']} · {m['cores']} cores · "
              f"vs {', '.join(VENDOR_LABELS[v] for v in m['compared'])}"
              + (f" ({', '.join(VENDOR_LABELS[v] for v in m['missing'])} not measured)" if m['missing'] else '')
              + ' · ' + (f'win {s[0]:.0%} / tie {s[1]:.0%} / lose {s[2]:.0%}' if s else 'nothing decided')
              + f" · {t['win']}/{t['tie']}/{t['lose']} cells, {t['no_spread']} without spread, "
              f"{t['withdrawn']} withdrawn · behind {m['behind']}")
        for family in every:
            counts = m['families'].get(family)
            if counts is None:
                print(f'    {family:18} not measured')
                continue
            s = share(counts)
            print(f"    {family:18} {leaning(counts) or '-':6} "
                  + (f'win {s[0]:.0%}  ' if s else '') + f"{counts['win']}/{counts['tie']}/{counts['lose']}")
    for u in unrecorded():
        print(f"not recorded: {u['profile']} {u['arch']} ({u['simd']}) · {u['model']} · {u['cores']} cores")
    return 0


if __name__ == '__main__':
    sys.exit(main())
