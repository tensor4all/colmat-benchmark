#!/usr/bin/env python3
"""Derive a profile's thread-row CPU sets from the node's L3 topology, and check the declared ones.

colmat-benchmark#143: the `ohtaka-cpu` rows were listed by hand (`0-15` ...) and the 16T row spans
four L3 domains, which a reader of "16T" would not guess. This reads
`/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list` (under `--sysfs-root`, default `/`),
fills whole L3 domains in CPU order for each requested thread count, and reports for every row the
domains it spans and whether it is L3-local (one domain). `check` compares a profile's
`thread_rows` and every suite's `cpu_sets` for that profile with the derived sets and exits 1 on a
difference, so a hand-edited list cannot drift from the node.

    cpu_sets.py derive --threads 1,16,32,64,128 [--sysfs-root /]
    cpu_sets.py check ohtaka-cpu [--sysfs-root /]
"""
import argparse
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def parse_cpu_list(text):
    cpus = []
    for part in text.strip().split(','):
        if part:
            lo, _, hi = part.partition('-')
            cpus += range(int(lo), int(hi or lo) + 1)
    return cpus


def fmt_cpu_list(cpus):
    cpus = sorted(cpus)
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f'{cpus[i]}-{cpus[j]}')
        i = j + 1
    return ','.join(out)


def l3_domains(sysfs_root=Path('/')):
    """Distinct L3 domains of the online CPUs, as sorted CPU lists, lowest first."""
    seen = {}
    for p in Path(sysfs_root).glob('sys/devices/system/cpu/cpu[0-9]*/cache/index3/shared_cpu_list'):
        cpus = tuple(sorted(parse_cpu_list(p.read_text())))
        seen[cpus] = None
    domains = sorted(seen)
    flat = [c for d in domains for c in d]
    if len(flat) != len(set(flat)):
        raise SystemExit('L3 domains overlap: ' + json.dumps(domains))
    return [list(d) for d in domains]


def derive_row(threads, domains, start=0):
    """Fill whole L3 domains in CPU order from CPU `start`; the observer is every CPU outside the
    domains touched. `start` is the first CPU of a declared row, so a profile whose rows begin
    above CPU 0 (the observing side sits on CCX 0) is judged on its own start, not on CPU 0."""
    everything = sorted(c for d in domains for c in d)
    pool = [c for c in everything if c >= start]
    if threads > len(pool):
        raise SystemExit(f'{threads}T from CPU {start} needs {threads} CPUs, only {len(pool)} remain')
    measured = pool[:threads]
    touched = [d for d in domains if set(d) & set(measured)]
    observer = sorted(set(everything) - {c for d in touched for c in d})
    return {
        'threads': threads, 'measured': fmt_cpu_list(measured), 'observer': fmt_cpu_list(observer),
        'l3_domains_spanned': len(touched),
        'domain_sizes': sorted({len(d) for d in touched}),
        'l3_local': len(touched) == 1,
        'whole_domains': all(set(d) <= set(measured) for d in touched),
        'smallest_domain_count': math.ceil(threads / max(len(d) for d in domains)),
    }


def declared(profile_name):
    profiles = yaml.safe_load((ROOT / 'benchmarks/profiles.yaml').read_text())
    prof = next((p for p in profiles['profiles']
                 if isinstance(p, dict) and p.get('name') == profile_name), None)
    if prof is None or 'thread_rows' not in prof:
        raise SystemExit(f'profile {profile_name!r} is not declared with thread_rows in profiles.yaml')
    rows = {r['threads']: r for r in prof['thread_rows']}
    suites = {}
    for path in sorted((ROOT / 'benchmarks/suites').glob('*.yaml')):
        for run in yaml.safe_load(path.read_text()).get('runs') or []:
            sets = run.get('cpu_sets') or {}
            sets = sets.get(profile_name) if isinstance(sets, dict) else None
            if sets:
                suites[path.stem] = suites.get(path.stem, []) + [(run['threads'], sets)]
    return rows, suites


def cmd_check(args):
    domains = l3_domains(Path(args.sysfs_root))
    rows, suites = declared(args.profile)
    bad = []
    derived = {}
    for t, row in rows.items():
        start = parse_cpu_list(row['measured'])[0]
        if start not in {dom[0] for dom in domains}:
            bad.append(f'profile row {t}T: starts at CPU {start}, which is not the first CPU of an L3 domain')
        d = derived[t] = derive_row(t, domains, start)
        if row['measured'] != d['measured']:
            bad.append(f'profile row {t}T: declared {row["measured"]!r}, derived {d["measured"]!r} from CPU {start}')
        touched = {c for dom in domains if set(dom) & set(parse_cpu_list(d['measured'])) for c in dom}
        clash = touched & set(parse_cpu_list(row.get('observer') or ''))
        if clash:
            bad.append(f'profile row {t}T: observer {row.get("observer")!r} is inside the measured L3 domains ({fmt_cpu_list(clash)})')
    for suite, runs in suites.items():
        for threads, sets in runs:
            if len(threads) != len(sets):
                bad.append(f'suite {suite}: {len(threads)} thread counts but {len(sets)} cpu_sets')
            for t, cpus in zip(threads, sets):
                start = parse_cpu_list(cpus)[0]
                if start not in {dom[0] for dom in domains}:
                    bad.append(f'suite {suite} {t}T: cpu_sets {cpus!r} starts at CPU {start}, '
                               'which is not the first CPU of an L3 domain')
                d = derived.get(t) or derive_row(t, domains, start)
                if cpus != d['measured']:
                    bad.append(f'suite {suite} {t}T: cpu_sets {cpus!r}, derived {d["measured"]!r}')
    print(f'L3 domains: {len(domains)} of sizes {sorted({len(d) for d in domains})}')
    for t, d in derived.items():
        kind = 'L3-LOCAL' if d['l3_local'] else f'NOT L3-local, {d["l3_domains_spanned"]} domains'
        print(f'{t:>4}T measured={d["measured"]:<8} observer={d["observer"] or "(none)":<10} {kind}'
              f' (whole domains: {d["whole_domains"]}, minimum possible: {d["smallest_domain_count"]})')
    for b in bad:
        print('MISMATCH', b, file=sys.stderr)
    return 1 if bad else 0


def cmd_derive(args):
    domains = l3_domains(Path(args.sysfs_root))
    print(json.dumps([derive_row(int(t), domains) for t in args.threads.split(',')], indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('derive')
    d.add_argument('--threads', required=True)
    c = sub.add_parser('check')
    c.add_argument('profile')
    for p in (d, c):
        p.add_argument('--sysfs-root', default='/')
    args = ap.parse_args(argv)
    return {'derive': cmd_derive, 'check': cmd_check}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
