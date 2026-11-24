#!/usr/bin/env python3
"""Write the site under `result/`: the landing page, the history, the charts and the joined view.

    python3 scripts/gen_site.py [--repo-ref main]

Nothing it writes is committed (see `.gitignore`); the Pages workflow runs it on every deploy from
the committed `data/`, `result/` and `pins/`. The landing page, the history, the HTML copy of
every result page and the charts declared in `benchmarks/campaign.yaml` (`figures:`, `trend:`) are the
toolkit's (`campaign_toolkit.site`). The throughput charts are one page per recorded
`(dtype, order)` dataset (`result/throughput-<dtype>-<order>.html`, `benchmarks/throughput.yaml`
`variants:`), linked from the landing page instead of carried on it.

The landing page is this campaign's own (#116), in the toolkit's frame (`site.write_page`), and
follows the maintainer's mockup on the issue: a generated headline whose every claim names its
comparison set, one win-share tile per machine, the family x machine win-share matrix, the win
share across pins, how the numbers are made, then each machine's provenance and run pages, folded.
Every number on it comes from `family_matrix.py`. What is also this campaign's own is the four-way view: the MKL and
OpenBLAS arms (and, on AMD profiles once recorded, the AOCL arm) of the current `colmat-kernels` pages joined on the cell key by `gen_comparison.py`,
written to `result/zen5-cpu/colmat-kernels/<rev12>-comparison.html` and linked from the landing page.

The site is not written when the pin has no current recording (`campaign_toolkit.pin_recorded`,
the same rule as `gen_index --check`): a landing page that says nothing is recorded at the pin while
the recordings sit in the history is the failure that rule exists to stop.
"""
import argparse
import html
import markdown
import pathlib
import re
import statistics
import sys

import yaml

from campaign_toolkit import descriptor, figures, gen_index, pin_recorded, report_run, site
from campaign_toolkit.stack import rev_file_value
from campaign_toolkit.campaign import load

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import family_matrix  # noqa: E402
import gen_comparison  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = 'pins/colmat.rev'
SUITE = 'colmat-kernels'
# The profile the published charts, the joined view and the provenance notes below describe.
ZEN5 = 'zen5-cpu'
# Read from the current arms' manifests, not restated here: a constant outlived the last
# publication (harness commit, toolkit commit, vendor kernel, coverage) and the reader had no
# way to tell.
LIMITATION_PREFIX = 'Known limitation: '


def limitation_text():
    """The measured suite's own stated limitation, verbatim.

    The landing page repeats what the suite declaration says instead of restating coverage in a
    second place: a constant here and the YAML bullet below it drift, and the page keeps claiming
    coverage the declaration no longer has.
    """
    data = yaml.safe_load((ROOT / 'benchmarks' / 'suites' / f'{SUITE}.yaml').read_text())
    for line in data['workloads']['known_blind_spots']:
        if line.startswith(LIMITATION_PREFIX):
            return line[len(LIMITATION_PREFIX):]
    raise SystemExit(f'FAIL {SUITE}.yaml declares no "{LIMITATION_PREFIX}" blind spot')


def current_arms():
    """{vendor: run dir} of the current `colmat-kernels` pages (the index's own rule), and the newest
    MKL run of an earlier revision, which states the unchanged row's control."""
    descriptor.set_root(ROOT)
    found = gen_index.collect()
    current, history = gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
    pages = {c['vendor']: c['page'] for c in current if c['suite'] == SUITE and c['page'] is not None}
    arms = {vendor: page['run_dir'] for vendor, page in pages.items()}
    revision = pages['mkl']['commit'] if 'mkl' in pages else None
    # history is newest first; the control is the newest MKL arm of another revision
    # `r['suite']` is the suite's current id (the index reads runs recorded under `faer-kernels`
    # into this suite's cell), so the history includes them.
    earlier = [r for r in history if r['suite'] == SUITE and r['vendor'] == 'mkl' and r['commit'] != revision]
    control = earlier[0]['run_dir'] if earlier else None
    return arms, control


def write_comparison():
    """Write the four-way view under its measured revision.

    Returns (page or None, refusal or None). The pairing guard (gen_comparison.require_one_protocol)
    refuses two arms that were not measured under one protocol, and the recorder takes each arm's
    protocol identity from the commit its run started at -- so two arms of one suite recorded either
    side of a data commit differ in `harness.commit` alone. The guard compares the harness by the
    paths the binary is built from (`gen_comparison.harness_identity_same`), so those arms join;
    a change to the harness itself is still refused. The guard is the authority on whether the two
    arms are one session, so its refusal is not argued with here: the landing page publishes what
    stands on its own and says that the joined view is withheld, naming the field it disagrees on.
    """
    arms, control = current_arms()
    if 'mkl' not in arms or 'openblas' not in arms:
        return None, None
    rev = load(arms['mkl'] / 'run.yaml')['library']['commit'][:12]
    out = ROOT / 'result' / 'zen5-cpu' / SUITE / f'{rev}-comparison.html'
    argv = ['--mkl', str(arms['mkl']), '--openblas', str(arms['openblas']), '--out', str(out)]
    if control is not None:
        argv += ['--control-mkl', str(control)]
    # The AOCL arm (AMD profiles, #95) is joined as a third arm when it has a current page. If the
    # guard refuses it, the MKL/OpenBLAS view is still written and the AOCL refusal is reported:
    # the third arm's disagreement says nothing about whether the first two are one session.
    if 'aocl' in arms:
        try:
            gen_comparison.main(argv + ['--aocl', str(arms['aocl'])])
            return out, None
        except gen_comparison.PairingRefused as refusal:
            print(f'the AOCL arm is not joined: {refusal}', file=sys.stderr)
    try:
        gen_comparison.main(argv)
    except gen_comparison.PairingRefused as refusal:
        # Only the pairing guard's own refusal is published around. Any other SystemExit from
        # gen_comparison (missing result files, an arm under the wrong column) still escapes and
        # fails the deploy, which is what a broken run must do.
        return None, str(refusal)
    return out, None


def _min_verdict_ratio():
    """`timing.baseline.min_verdict_ratio`, the margin a below-baseline page applies (1.0 if none)."""
    return descriptor.current().timing_baseline.get('min_verdict_ratio') or 1.0


def _declared_orders(manifest):
    """The memory orders the recording's suite declares ('both orders', 'F order only', ...).

    The provenance block used to print "both orders" for every machine, because it was written for
    zen5's suite. `ohtaka-fat` declares `order: [f]`, so that sentence was false on its page.
    """
    path = ROOT / manifest['suite_file']
    suite = yaml.safe_load(path.read_text()) if manifest.get('suite_file') and path.exists() else {}
    orders = []
    for run in suite.get('runs') or []:
        for order in run.get('order') or []:
            if str(order).lower() not in orders:
                orders.append(str(order).lower())
    orders = orders or ['f', 'c']
    if set(orders) >= {'f', 'c'}:
        return 'both orders'
    names = {'f': 'F (column-major)', 'c': 'C (row-major)'}
    return ' and '.join(names.get(o, o.upper()) for o in orders) + ' order only'


def provenance_html(arms, folded=False):
    """One profile's provenance section, read from its current arms' own manifests.

    It names its profile in its heading: it used to be one site-wide block on the landing page and
    read as if it described every machine (#116). On the landing page it sits folded inside the
    profile's card (`folded=True`); the chart pages, which are that profile's, show it open.

    Every value here is the recording's own: the measured library and its declared version, the
    upstream baseline, the harness and toolkit commits, the vendor builds and the kernel each one
    dispatched to. A restated constant goes stale silently -- this section restated the harness
    commit, the toolkit commit, the vendor kernel and the coverage of an earlier publication, and a
    reader had no way to tell that the page no longer described its own runs.
    """
    runs = {vendor: load(arm / 'run.yaml') for vendor, arm in arms.items()}
    if not runs:
        return ''
    first = runs.get('mkl') or next(iter(runs.values()))
    lib, up = first['library'], first['upstream']
    threads = '/'.join(str(t) for t in first['threads']['counts'])
    dtypes = ', '.join(first['run_spec']['dtypes'])
    orders = _declared_orders(first)
    items = [
        'recorded on <code>{host}</code> ({profile}), measured library <code>{url}</code> at '.format(
            host=html.escape(first['host']['hostname']), profile=html.escape(first['target_profile']),
            url=html.escape(lib['url']))
        + f"<code>{lib['commit']}</code>, declared version <code>{html.escape(str(lib.get('version')))}</code>, "
        + f"upstream <code>{up['commit']}</code> (<code>{html.escape(str(up.get('ref')))}</code>)",
        f"measurement toolkit <code>{first['toolkit']['commit']}</code>",
        # Each arm states the commit that recorded it: the runs of one publication can be recorded
        # at different commits (a data commit between them), and a single shared value would deny
        # what the page's own withheld-comparison note says.
        f"one row per engine: {'; '.join(_vendor_text(runs))}",
        f"the recorded suite's declared grid (<code>{html.escape(str(first.get('suite_id') or 'recorded suite'))}</code>): "
        f"{html.escape(dtypes)} at {threads}T, {html.escape(orders)}, every declared regime",
    ]
    profile = html.escape(first['target_profile'])
    heading = f'Current {profile} recordings: scope and provenance'
    out = (f'<details>\n<summary>{heading}</summary>\n<ul>\n' if folded
           else f'<section>\n<h2>{heading}</h2>\n<ul>\n')
    out += ''.join(f'<li>{item}</li>\n' for item in items)
    out += '</ul>\n'
    if not lib.get('tag'):
        out += ('<p><strong>No tag names this revision.</strong> '
                f"<code>{lib['commit'][:12]}</code> is the library's development head, not a release: "
                f'the version string <code>{html.escape(str(lib.get("version")))}</code> comes from its '
                'manifest and does not identify this snapshot, and the published crates.io source for '
                'that version was cut from a different commit (see the worklog). The commit does '
                'identify it.</p>\n')
    openblas = runs.get('openblas')
    corename = (openblas or {}).get('vendor', {}).get('corename')
    if corename and not corename.lower().startswith('zen'):
        out += ('<p><strong>OpenBLAS arm: <code>' + html.escape(corename) + '</code> kernel, not '
                'Zen5-optimized.</strong> ' + html.escape(first['host']['cpu']) + ' dispatched to the '
                '<code>' + html.escape(corename) + '</code> kernel in the recorded run; comparisons with '
                'the MKL arm describe these builds and dispatched kernels, not a general vendor '
                'ranking on Zen5.</p>\n')
    if first['target_profile'] != ZEN5:
        return out + ('</details>' if folded else '</section>')
    if first['host'].get('role') != 'reference':
        out += ('<p>Evo was reserved for measurement, but the recorded <code>host.role='
                + html.escape(str(first['host'].get('role'))) + '</code> is unchanged: these reports '
                'retain their profile\'s <strong>provisional</strong> designation.</p>\n')
    out += ('<p>The wrong-pin partial <code>20261031T222831Z-pub-3a8edf02-20261031-mkl</code> is '
            'preserved locally and explicitly excluded from this publication. One rejected '
            'small-OpenBLAS 8T attempt is retained in its raw archive, not counted as an accepted '
            'sample.</p>\n')
    return out + ('</details>' if folded else '</section>')


def _vendor_text(runs):
    """`MKL 2026.2 …` / `OpenBLAS 0.3.32 (dispatch Cooperlake) …` for each arm present."""
    out = []
    for vendor, run in runs.items():
        v = run['vendor']
        text = f"<code>{html.escape(str(v.get('name')))} {html.escape(str(v.get('version')))}</code>"
        if v.get('linkage'):
            text += f" ({html.escape(str(v['linkage']))})"
        if v.get('corename'):
            text += f", dispatch <code>{html.escape(str(v['corename']))}</code>"
        text += f", recorded by harness <code>{html.escape(str(run['harness']['commit']))}</code>"
        out.append(text)
    return out


def variant_label(variant):
    """`f64 / F order`: the name of one recorded dataset, as the nav and the index print it."""
    return f'{variant[0]} / {variant[1].upper()} order'


def write_throughput_pages(generated, arms):
    """One HTML page per declared (dtype, order) dataset of the throughput charts.

    The landing page links these instead of carrying every dataset's charts inline: a chart on
    it named no dataset, and the recorded c64 cells had no page at all. Each page opens with its
    title, a one-line dataset nav and then the charts of its own dataset; the landing page's
    provenance, the suite's limitation, the chart notes and the revisions line follow below them
    (#106). The charts are the toolkit's (`figures.site_figures` wrote one set
    per variant, from that variant's own cells), and so are the frame and the chart block
    (`site.write_page`, `figures.grid_html`), so no result page is duplicated here.
    """
    config = yaml.safe_load((ROOT / 'benchmarks' / 'throughput.yaml').read_text())
    variants = [(v['dtype'], v['order']) for v in config['variants']]
    provenance = provenance_html(arms)
    provenance += f'<p><strong>Known limitation:</strong> {html.escape(limitation_text())}</p>\n'
    notes = markdown.markdown('\n'.join(f'- {line}' for line in config['page']['notes']))
    pages = {}
    for variant in variants:
        entries = [f for f in generated if f['kind'] == 'throughput'
                   and (f['dtype'], f['order']) == variant]
        # The runs behind the charts, as the toolkit states them (the same line each chart
        # set's README prints): a plotted number can come from the large-tier run, whose
        # revisions the daily provenance above does not name.
        revisions = markdown.markdown(entries[0]['revisions'])
        nav = ' &middot; '.join(
            f'<strong>{html.escape(variant_label(other))}</strong>' if other == variant else
            f'<a href="throughput-{other[0]}-{other[1]}.html">{html.escape(variant_label(other))}</a>'
            for other in variants)
        body = (f'<nav class="muted">dataset: {nav}</nav>\n'
                + figures.grid_html(ROOT / 'result', entries)
                + f'\n{provenance}\n{notes}\n{revisions}\n')
        pages[variant] = site.write_page(
            ROOT / 'result', f'throughput-{variant[0]}-{variant[1]}.html',
            f'Throughput charts: {variant_label(variant)}', body)
    return pages


def chart_nav_html(pages):
    """The dataset pages as one inline line, f64 / F (the column-major default view) first."""
    return ' &middot; '.join(
        f'<a href="{page.name}">{html.escape(variant_label(variant))}</a>'
        + (' <span class="sm">(default view)</span>' if variant == ('f64', 'f') else '')
        for variant, page in pages.items())


# The landing page's own styles, on top of the toolkit's frame (`site.CSS`), after the
# maintainer's mockup on #116. Win / tie / lose are carried by words and counts as well as by
# colour, and "not measured" is a white cell with a dashed border so it cannot read as a tie. The
# family matrix is a real `<table>` at every width: on a narrow screen it scrolls inside its box.
LANDING_CSS = """<style>
.sm{font-size:12px;color:#5f5e5a}
.topnav{display:flex;justify-content:space-between;flex-wrap:wrap;gap:.3rem 1rem;border-bottom:1px solid #dcdad3;padding-bottom:6px;margin-bottom:12px}
.headline{font-size:15px;margin:0 0 4px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(15rem,1fr));gap:12px}
.tile{border:1px solid #dcdad3;border-radius:12px;padding:12px 14px}
.tile .name{font-weight:600}
.big{font-size:26px;font-weight:500;margin-top:4px}
.big .unit{font-size:13px;color:#5f5e5a;font-weight:400}
.bar{display:flex;height:10px;border-radius:4px;overflow:hidden;margin:8px 0 6px;background:#f1efe8}
.bar .bw{background:#639922}.bar .bt{background:#b4b2a9}.bar .bl{background:#e24b4a}
.badge{padding:0 6px;border-radius:4px;font-size:12px;background:#faeeda;color:#633806;display:inline-block;max-width:100%}
.matrix-wrap{overflow-x:auto;margin:.4rem 0}
table.share{border-collapse:separate;border-spacing:4px;width:auto;min-width:100%}
table.share th,table.share td{border:0;padding:6px 8px;text-align:center;white-space:nowrap}
table.share th{background:none;font-weight:400;font-size:12px;color:#5f5e5a}
table.share th[scope=row]{text-align:left;position:sticky;left:0;background:#fff}
table.share td{border-radius:6px;font-size:13px}
table.share td a{color:inherit;text-decoration:none;display:block}
.w{background:#eaf3de;color:#27500a}.t{background:#f1efe8;color:#444441}.l{background:#fcebeb;color:#791f1f}
table.share td.na{background:#fff;border:1px dashed #c9c7bf;color:#888780;font-size:12px}
.lw{color:#27500a}.ll{color:#791f1f}
.placeholder{border:1px dashed #c9c7bf;border-radius:8px;padding:12px 14px}
details.machine{border-top:1px solid #dcdad3;padding:6px 0}
details.machine>summary{cursor:pointer;font-weight:500}
figure{margin:1rem 0 .3rem}figure img{max-width:100%}
</style>
"""


def suite_headlines(suites):
    """The capitalised caveats (`OS-SHARED, NOT COMPARABLE WITH THE 60T ROW: ...`) the suites'
    declarations lead their blind spots with, verbatim and in suite order: a caveat a recorded page
    shouts must not be dropped by the summary that links to it."""
    out = []
    for suite in suites:
        path = ROOT / 'benchmarks' / 'suites' / f'{suite}.yaml'
        if not path.exists():
            continue
        spots = (yaml.safe_load(path.read_text()).get('workloads') or {}).get('known_blind_spots') or []
        out += [spot for spot in spots if ':' in spot and spot.split(':', 1)[0].isupper()
                and spot not in out]
    return out


def _page_href(page):
    """A recorded page's HTML copy, relative to the site root (`result/`)."""
    return str(pathlib.PurePosixPath(page).relative_to('result').with_suffix('.html'))


def _vendors(labels):
    return ', '.join(family_matrix.VENDOR_LABELS.get(v, v) for v in labels)


def _short(machine):
    """`Zen 5 · 12 cores`: the matrix column heading."""
    arch = re.sub(r'^(AMD|Intel)\s+', '', machine['arch'])
    return f"{arch} · {machine['cores']} cores" if machine['cores'] else arch


def _families_text(machine, families):
    n = len([f for f in families if f in machine['families']])
    return f'{n} of {len(families)} families compared'


def headline_html(machines, families):
    """The generated headline: each claim carries its comparison set and its coverage, so no
    sentence is broader than the cells behind it."""
    def where(m, first):
        win = family_matrix.percentages(m['total'])[0]
        scope = f"vs {_vendors(m['compared'])}"
        if len([f for f in families if f in m['families']]) < len(families):
            scope += f', {_families_text(m, families)}'
        pct = f'<b>{win}%</b>' + (' of cells' if first else '')
        return pct, f'{html.escape(m["arch"])} ({html.escape(scope)})'

    decided = [m for m in machines if family_matrix.share(m['total'])]
    ahead = [m for m in decided if family_matrix.share(m['total'])[0] >= family_matrix.MOSTLY]
    behind = [m for m in decided if m not in ahead]
    sentences = []
    if ahead:
        parts = [f'{pct} on {name}' for pct, name in (where(m, i == 0) for i, m in enumerate(ahead))]
        joined = parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]
        sentences.append(f'colmat beats the fastest vendor library in {joined}.')
    for m in behind:
        pct, name = where(m, True)
        gaps = [f for f in families if f in m['families']
                and family_matrix.leaning(m['families'][f]) == 'lose']
        gap = ''
        if gaps:
            names = [f'<code>{html.escape(f)}</code>' for f in gaps]
            gap = '; ' + (names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]) \
                + (' is' if len(names) == 1 else ' are') + ' the gap'
        sentences.append(f'On {name}, colmat beats it in {pct}{gap}.')
    if not sentences:
        return '<p class="headline">No cell has been compared with a vendor library yet.</p>\n'
    return f'<p class="headline">{" ".join(sentences)}</p>\n'


def subline_html(revision, version):
    return (f'<p class="sm" style="margin:0 0 14px">Generated from the data. colmat '
            f'<code>{html.escape(revision)}</code> ({html.escape(str(version))}) &middot; F order, '
            f'{"/".join(family_matrix.DTYPES)}, every dimension &ge; {family_matrix.MIN_N} &middot; win or lose = a '
            'difference larger than the recorded spread (at least 5%), otherwise tie &middot; '
            '<a href="#method">how the numbers are made</a></p>\n')


def tile_html(m, families, chart_href):
    s = family_matrix.share(m['total'])
    t = m['total']
    lean = family_matrix.leaning(t)
    colour = {'win': 'lw', 'lose': 'll'}.get(lean, '')
    threads = m['threads']
    thread_text = (f'{threads[0]}&ndash;{threads[-1]} threads' if len(threads) > 1
                   else f'{threads[0]} thread{"s" if threads[0] != 1 else ""}')
    # behind is None when extern/colmat cannot place the recorded commit: the badge then says the
    # distance is unknown and names the commit, rather than a count-free "older pin".
    pin = ('current pin' if m['behind'] == 0 else
           f'<span class="badge">older pin: {m["behind"]} commits behind</span>' if m['behind']
           else f'<span class="badge">older pin: <code>{html.escape(m["commit"][:7])}</code>, distance to the '
                'pin unknown (extern/colmat cannot place it)</span>')
    lines = []
    if s:
        win, tie, lose = family_matrix.percentages(t)
        lines.append(f'<div class="big {colour}">{win}% <span class="unit">win</span></div>')
        lines.append(f'<div class="bar" role="img" aria-label="win {win}%, tie {tie}%, '
                     f'lose {lose}%"><div class="bw" style="width:{100 * s[0]:.1f}%"></div>'
                     f'<div class="bt" style="width:{100 * s[1]:.1f}%"></div>'
                     f'<div class="bl" style="width:{100 * s[2]:.1f}%"></div></div>')
        lines.append(f'<div class="sm">{t["win"]} / {t["tie"]} / {t["lose"]} '
                     f'(win / tie / lose of {t["win"] + t["tie"] + t["lose"]} cells) &middot; vs '
                     f'{html.escape(_vendors(m["compared"]))}'
                     + (f' ({html.escape(_vendors(m["missing"]))} not measured)' if m['missing'] else '')
                     + '</div>')
    else:
        lines.append('<div class="big">&mdash;</div><div class="sm">no cell compared with a vendor</div>')
    covered = _families_text(m, families)
    if len([f for f in families if f in m['families']]) < len(families):
        covered = f'<b>{covered}</b>'
    lines.append(f'<div class="sm">{covered} &middot; {thread_text} &middot; {html.escape(m["date"])} '
                 f'&middot; {pin}</div>')
    notes = []
    if m['provisional']:
        notes.append('<span class="badge" title="recorded on a development host">provisional</span>')
    if m['below_baseline']:
        margin = (f' (a verdict here also needs a ratio beyond {m["min_ratio"]:.2f}&times;)'
                  if m.get('min_ratio', 1.0) > 1 else '')
        notes.append('<span class="badge" title="the index marks these pages not comparable with the '
                     'timing baseline">below the timing baseline: '
                     f'{html.escape("; ".join(m["below_baseline"]))}{margin}</span>')
    if m['excluded']:
        excluded = sorted({t for a in m['excluded'] for t in a.manifest['threads']['counts']})
        notes.append(f'<span class="sm">{"/".join(map(str, excluded))}T os-shared run listed, not counted</span>')
    if m.get('os_shared'):
        notes.append(f'<span class="sm">{"/".join(map(str, m["os_shared"]))}T row measures every core '
                     '(no observing CPU left); listed, not counted</span>')
    for key, text in (('no_spread', 'without a recorded spread'), ('withdrawn', 'withdrawn')):
        if t[key]:
            notes.append(f'<span class="sm">{t[key]} cells {text}, not counted</span>')
    if notes:
        lines.append(f'<div style="margin-top:4px">{" ".join(notes)}</div>')
    href = chart_href.get(m['profile'])
    link = (f'<a href="{html.escape(href)}">Throughput charts &rarr;</a>' if href
            else f'<a href="#runs-{html.escape(m["profile"])}">Recorded pages &rarr;</a>')
    simd = f' <span class="sm">({html.escape(m["simd"])})</span>' if m['simd'] else ''
    return (f'<div class="tile" id="{html.escape(m["profile"])}"><div class="name">{html.escape(m["arch"])}{simd}</div>'
            f'<div class="sm">{html.escape(m["model"])}'
            + (f' &middot; {m["cores"]} cores' if m['cores'] else '')
            + f' &middot; <code>{html.escape(m["profile"])}</code></div>'
            + ''.join(lines) + f'<div class="sm" style="margin-top:6px">{link}</div></div>\n')


def tiles_html(machines, families, chart_href, unrecorded):
    out = '<div class="tiles">\n' + ''.join(tile_html(m, families, chart_href) for m in machines) + '</div>\n'
    out += ('<p class="sm" style="margin:8px 0 4px">Bar: <span class="lw">win</span> / tie / '
            '<span class="ll">lose</span>, as shares of the cells compared with the fastest vendor arm '
            'on that machine. A machine recorded at an older pin is shown, not hidden, with a badge '
            'such as <span class="badge">older pin: N commits behind</span>.</p>\n')
    if unrecorded:
        names = ' &middot; '.join(
            f'{html.escape(u["arch"])}' + (f' ({html.escape(u["simd"])})' if u['simd'] else '')
            + f' &middot; {html.escape(u["model"])}' + (f' &middot; {u["cores"]} cores' if u['cores'] else '')
            + f' <code>{html.escape(u["profile"])}</code>' for u in unrecorded)
        out += f'<p class="sm" style="margin:0 0 18px">Not recorded yet: {names}.</p>\n'
    return out


def _cell_href(m, family, chart_href):
    if m['profile'] in chart_href:
        return chart_href[m['profile']]
    pages = [c['page'] for c in m['families'][family]['cells']]
    return _page_href(pages[0]) if pages else f'#runs-{m["profile"]}'


def matrix_html(machines, families, chart_href):
    """Win share by family: one row per family, one column per machine."""
    head = ''.join(f'<th scope="col"><a href="#{html.escape(m["profile"])}">{html.escape(_short(m))}</a></th>'
                   for m in machines)
    words = {'win': ('w', 'mostly win'), 'lose': ('l', 'mostly lose'), 'tie': ('t', 'mostly tie'),
             'mixed': ('t', 'mixed')}
    rows = []
    for family in families:
        tds = []
        for m in machines:
            counts = m['families'].get(family)
            s = counts and family_matrix.percentages(counts)
            if counts is None:
                why = family_matrix.absence(m, family)
                word = why.split(' (', 1)[0]
                tds.append(f'<td class="na" title="{html.escape(why)}">{html.escape(word)}</td>')
                continue
            if not s:
                tds.append(f'<td class="na" title="{counts["no_spread"]} without a recorded spread, '
                           f'{counts["withdrawn"]} withdrawn">no decided cell</td>')
                continue
            cls, word = words[family_matrix.leaning(counts)]
            title = (f"{counts['win']} win / {counts['tie']} tie / {counts['lose']} lose of "
                     f"{counts['win'] + counts['tie'] + counts['lose']} cells against the fastest vendor arm")
            tds.append(f'<td class="{cls}"><a href="{html.escape(_cell_href(m, family, chart_href))}" '
                       f'title="{html.escape(title)}">{s[0]}% &middot; {word}</a></td>')
        rows.append(f'<tr><th scope="row">{html.escape(family)}</th>{"".join(tds)}</tr>')
    return ('<h2 id="families">Win share by family</h2>\n'
            '<div class="matrix-wrap"><table class="share">\n'
            f'<thead><tr><th></th>{head}</tr></thead>\n<tbody>\n{chr(10).join(rows)}\n</tbody></table></div>\n'
            '<p class="sm" style="margin:6px 0 18px">Each cell is the share of that family\'s cells '
            'colmat wins against the fastest vendor arm, and links to that machine\'s throughput charts '
            '(or, where it has none yet, the recorded page). Cell colour: mostly win / mixed or tie / '
            'mostly lose, where "mostly" is more than half of the cells; the word says the same. '
            '"Not measured" (the machine recorded no row of the family), "no vendor cell" (rows '
            'recorded, none with a vendor row inside the cell rule) and "no decided cell" are white '
            'cells with a dashed border and are never coloured as a tie. Hover '
            'for the counts.</p>\n')


def across_pins_html(points, machines):
    """Win share across pins: one line per machine, from the second pin a machine has a share at."""
    # A pin without a decided cell has no share (`counts` None, `why` set): it is never plotted,
    # never tabled and never counted as a pin with a share, whatever the producer handed over.
    def has_share(p):
        return p['counts'] is not None and family_matrix.share(p['counts']) is not None
    lines = {m['profile']: [p for p in points.get(m['profile'], []) if has_share(p)] for m in machines}
    drawn = {k: v for k, v in lines.items() if len(v) >= 2}
    out = '<h2 id="pins">Win share across pins</h2>\n'
    if not drawn:
        why = []
        for m in machines:
            skipped = [p for p in points.get(m['profile'], []) if not has_share(p)]
            n = len(lines[m['profile']])
            text = (f'<code>{html.escape(m["profile"])}</code>: '
                    + (f'{n} pin{"s" if n != 1 else ""} with a share' if n else 'no pin with a share'))
            if skipped:
                reasons = sorted({p['why'] or 'no decided cell' for p in skipped})
                text += f', {len(skipped)} pin(s) without one ({html.escape("; ".join(reasons))})'
            why.append(text)
        return out + ('<div class="placeholder sm">One line per machine, appearing from the second pin '
                      'with a share. Not yet: ' + '; '.join(why) + '.</div>\n')
    width, height, pad = 640, 220, 36
    every = sorted({p['date'] for v in drawn.values() for p in v})
    x = {d: pad + (width - 2 * pad) * (i / max(1, len(every) - 1)) for i, d in enumerate(every)}
    y = lambda share: height - pad - (height - 2 * pad) * share   # noqa: E731
    palette = ('#185fa5', '#639922', '#e24b4a', '#7f4fb5', '#b4801d')
    svg = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="win share across pins">',
           f'<text x="{pad}" y="14" font-size="11" fill="#5f5e5a">win share (% of cells), linear</text>']
    for level in (0, .5, 1):
        svg.append(f'<line x1="{pad}" x2="{width - pad}" y1="{y(level):.1f}" y2="{y(level):.1f}" stroke="#dcdad3"/>'
                   f'<text x="4" y="{y(level) + 4:.1f}" font-size="11" fill="#5f5e5a">{round(level * 100)}%</text>')
    table = []
    for i, (profile, pts) in enumerate(drawn.items()):
        colour = palette[i % len(palette)]
        coords = ' '.join(f'{x[p["date"]]:.1f},{y(family_matrix.share(p["counts"])[0]):.1f}' for p in pts)
        svg.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="{coords}"/>')
        svg.append(f'<text x="{width - pad + 4}" y="{y(family_matrix.share(pts[-1]["counts"])[0]) + 4:.1f}" '
                   f'font-size="11" fill="{colour}">{html.escape(profile)}</text>')
        for p in pts:
            s = family_matrix.percentages(p['counts'])
            n = p['counts']['win'] + p['counts']['tie'] + p['counts']['lose']
            table.append(f'<tr><td>{html.escape(profile)}</td><td><code>{html.escape(p["commit"][:12])}</code></td>'
                         f'<td>{html.escape(p["date"])}</td><td>{s[0]}%</td><td>{s[1]}%</td>'
                         f'<td>{s[2]}%</td><td>{n}</td><td>{html.escape(", ".join(p.get("suites", [])))}</td></tr>')
    svg.append('</svg>')
    return (out + ''.join(svg)
            + '<p class="sm">x: the pins in recording order; y: win share, 0&ndash;100%. Each point is '
              'over the decided cells of that pin\'s own pages (one per suite and arm), so two points '
              'can be over different cell sets: the table gives each point\'s cells and suites.</p>\n'
            + '<details><summary class="sm">table: the plotted numbers</summary><table>'
            + '<tr><th>machine</th><th>colmat</th><th>recorded</th><th>win</th><th>tie</th><th>lose</th>'
            '<th>decided cells</th><th>suites</th></tr>'
            + ''.join(table) + '</table></details>\n')


def trend_html(trend):
    """The representative cases across pins, each with its scale stated and its numbers as a table."""
    if not trend:
        return ''
    root = ROOT / 'result'
    items = []
    for f in trend:
        svg = f['svg'].relative_to(root).as_posix()
        text = f['data'].read_text()
        table = markdown.markdown(text.split('\n', 1)[1] if text.startswith('# ') else text,
                                  extensions=['tables'])
        items.append(
            f'<figure><a href="{html.escape(svg)}"><img src="{html.escape(svg)}" '
            f'alt="{html.escape(f["title"])}"></a>'
            f'<figcaption><strong>{html.escape(f["title"])}</strong>: median time per pin in ms '
            '(lower is faster), log axis (the minor gridlines are the values between the labelled '
            'ticks), the bar is the between-start spread; the panels share their pin axis side by '
            'side.</figcaption></figure>\n'
            f'<details><summary class="sm">table: the plotted numbers</summary><div class="matrix-wrap">{table}'
            f'</div><p class="sm"><a href="{html.escape(f["data"].relative_to(root).as_posix())}">'
            'as Markdown</a></p></details>\n')
    return ('<details><summary>Representative cases across pins (time, zen5-cpu)</summary>\n'
            + ''.join(items) + '</details>\n')


def method_html(machines):
    """How the numbers are made: the scripts, the inputs, the pins, the rule, the noise floor."""
    toolkit = re.search(r'campaign-toolkit@([0-9a-f]{40})', (ROOT / 'requirements.txt').read_text())
    pins = {name: rev_file_value((ROOT / 'pins' / name).read_text())
            for name in ('colmat.rev', 'upstream-faer.rev')}
    floors, worst = [], 0.0
    for m in machines:
        medians = [g['between_median'] for g in (report_run.spread_guide(a.best, a.within) for a in m['arms'])
                   if g['between_median'] is not None]
        if medians:
            floors.append(f'{html.escape(m["profile"])} {statistics.median(medians):.2%}')
            worst = max(worst, statistics.median(medians))
    caveats = ''.join(f'<li><code>{html.escape(m["profile"])}</code>: {html.escape(h)}</li>'
                      for m in machines
                      for h in suite_headlines(sorted({a.suite for a in m['arms'] + m['excluded']})))
    pages = sum(len(m['arms']) + len(m['excluded']) for m in machines)
    return (
        '<h2 id="method">How the numbers are made</h2>\n<ul>\n'
        '<li>Generated on every Pages deploy by <code>scripts/gen_site.py</code>; every share comes from '
        '<code>scripts/family_matrix.py</code> (<code>python3 scripts/family_matrix.py</code> prints the '
        'same counts and the same percentages, rounded by the same function), which the cross-host vendor-loss table (#108) is meant to read too. No timing runs '
        'here.</li>\n'
        f'<li>Inputs: the <code>run.yaml</code> and the CSVs it names of the {pages} current pages '
        '(<code>result/INDEX.md</code>, the index\'s own rule), under <code>data/results/</code>.</li>\n'
        f'<li>A cell: one family, shape, dtype and thread count, F order, {"/".join(family_matrix.DTYPES)}, '
        f'every dimension &ge; {family_matrix.MIN_N}, batch 1, the kernel route (the '
        '<code>api-inclusive</code> rows are not counted). colmat is compared with the vendor row of '
        'the fastest vendor arm for that cell, inside that arm\'s own process.</li>\n'
        '<li>Win / tie / lose: each recorded value is the median of the process medians, and '
        '<code>&plusmn;</code> the sample SD of those medians across process starts. A win or a loss '
        'is a difference larger than the two rows\' combined relative SD, floored at 5%, and larger '
        'than each row\'s printed <code>&plusmn;</code> '
        '(<code>campaign_toolkit.report_run.comparison_verdict</code>, the rule behind every recorded '
        'page\'s <code>vs</code> column); inside that band it is a tie. A cell without two process sets '
        'has no spread and is not counted; neither is a vendor row a pre-fair harness recorded '
        '(<code>gen_comparison.fair_harness</code>). An arm a machine never ran is "not measured", never a '
        'share.</li>\n'
        f'<li>Toolkit: <code>campaign-toolkit</code> at <code>{toolkit.group(1) if toolkit else "?"}</code> '
        '(<code>requirements.txt</code>).</li>\n'
        f'<li>Libraries: <code>extern/</code> materialised at the pins by '
        f'<code>scripts/setup_extern_deps.sh</code>: colmat <code>{html.escape(pins["colmat.rev"])}</code> '
        f'(<code>pins/colmat.rev</code>), faer <code>{html.escape(pins["upstream-faer.rev"])}</code> '
        '(<code>pins/upstream-faer.rev</code>).</li>\n'
        + (f'<li>Noise floor: the median between-start range of a row is {"; ".join(floors)} (the runs\' '
           'own spread guides)'
           + ('; the 5% floor is above all of them' if worst < report_run.VERDICT_FLOOR
              else '; above 5% on at least one machine, where the band is wider than 5%')
           + '.</li>\n' if floors else '')
        + '<li>An <code>os-shared</code> run measures every core, leaving none to the observing side; it '
          'is listed under its machine and never counted in a share. A thread row whose declared CPU '
          'set covers every logical CPU of the host is the same case inside a normal profile: it is '
          'named on the machine\'s tile and never counted.</li>\n'
        + '<li>A page marked <em>below the timing baseline</em> was recorded with less priming or fewer '
          'repetitions than the campaign froze, so a few-percent difference on it may be the bias, not '
          'the library: a verdict there also requires a ratio beyond '
          f'<code>timing.baseline.min_verdict_ratio</code> ({_min_verdict_ratio():.2f}&times;), the same '
          'margin the page\'s own header states.</li>\n'
        + (f'<li>Caveats the suites declare:<ul>{caveats}</ul></li>\n' if caveats else '')
        + '</ul>\n')


def _arm_label(arm):
    vendor = arm.manifest.get('vendor') or {}
    name = family_matrix.VENDOR_LABELS.get(arm.vendor, arm.vendor)
    return f"{name} {vendor['version']}" if vendor.get('version') else name


def machine_html(m, *, chart_nav, comparison, notices, unpinned):
    """One machine's folded provenance: its arms, run pages, caveats and recorded provenance."""
    arms = [label for _, label in sorted({(family_matrix.vendor_rank(a.vendor), _arm_label(a))
                                          for a in m['arms']})]
    pages = []
    for arm in sorted(m['arms'] + m['excluded'], key=lambda a: (a.profile, a.suite, family_matrix.vendor_rank(a.vendor))):
        extra = '' if arm.coverage in (None, 'full') else f' <span class="sm">({html.escape(arm.coverage)})</span>'
        if arm in m['excluded']:
            extra += ' <span class="sm">(os-shared, not counted)</span>'
        pages.append(f'<li><a href="{html.escape(_page_href(arm.page))}">{html.escape(arm.suite)} &middot; '
                     f'{html.escape(family_matrix.VENDOR_LABELS.get(arm.vendor, arm.vendor))}</a>{extra}</li>')
    links = []
    if chart_nav:
        links.append(f'<li>charts: {chart_nav}</li>')
    if comparison:
        links.append(f'<li>{comparison}</li>')
    headlines = suite_headlines(sorted({a.suite for a in m['arms'] + m['excluded']}))
    return (
        f'<details class="machine" id="runs-{html.escape(m["profile"])}"><summary>'
        f'{html.escape(m["arch"])} &middot; {html.escape(m["model"])} &middot; <code>{html.escape(m["profile"])}</code>'
        '</summary>\n'
        + ('<p class="summary"><strong>Unpinned profile</strong>: not comparable with the pinned '
           'profiles.</p>\n' if unpinned else '')
        + f'<p class="sm">recorded {html.escape(m["date"])} at colmat <code>{html.escape(m["commit"][:12])}</code>'
          f' &middot; vendor arms: {html.escape(", ".join(arms))}</p>\n'
        + ''.join(f'<p class="summary">{html.escape(h)}</p>\n' for h in headlines)
        + (f'<ul>{"".join(links)}</ul>\n' if links else '')
        + f'<p class="sm">recorded pages:</p>\n<ul>{"".join(pages)}</ul>\n'
        + ''.join(notices)
        + '</details>\n')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-ref', default='main', help='the branch the site is published from')
    args = ap.parse_args(argv)
    descriptor.set_root(ROOT)
    problems = pin_recorded.campaign_problems()
    if problems:
        for problem in problems:
            print(f'FAIL {problem}', file=sys.stderr)
        print('FAIL the site is not written while the pin has no current recording', file=sys.stderr)
        return 1
    index = ROOT / 'result' / 'index.html'
    comparison, refusal = write_comparison()
    arms, _control = current_arms()
    # The charts are generated once (from the runs behind the current pages, the same rule
    # `site.write_site` applies): the throughput ones onto one page per (dtype, order) dataset,
    # the trend ones onto the landing page.
    view = site.campaign_view(ROOT, index.parent)
    current = [ROOT / c['page']['page'] for c in view['current'] if c['page']]
    generated = figures.site_figures(ROOT, index.parent, current)
    chart_pages = write_throughput_pages(generated, arms)
    trend = [f for f in generated if f['kind'] != 'throughput']
    # The toolkit writes the history and the HTML copy of every recorded page; the landing page
    # it writes is then replaced by the summary below, in the same frame (`site.write_page`).
    site.write_site(index.parent, pins=[PIN], repo_ref=args.repo_ref, title='colmat benchmark',
                    root=ROOT, charts=trend)

    order, pages, history = family_matrix.collect()
    machines = family_matrix.machines(order, pages)
    unrecorded = family_matrix.unrecorded(order, pages)
    families = family_matrix.universe(machines)
    # The machines the throughput charts are drawn from: the profiles of the current pages of the
    # suites `figures.throughput.runs` declares (#114 adds the others' charts).
    declared = yaml.safe_load((ROOT / 'benchmarks' / 'campaign.yaml').read_text())
    chart_suites = {r['suite'] for r in declared['figures']['throughput']['runs']}
    # The chart pages are one set, drawn from the runs whose provenance they print (`arms`): they
    # belong to the machine those runs were recorded on, and only that machine links to them. A
    # second machine with a charted suite gets its own recorded pages until it has its own charts,
    # instead of a link to another machine's (#114).
    results = ROOT / 'data' / 'results'
    owners = {pathlib.Path(d).resolve().relative_to(results.resolve()).parts[0] for d in arms.values()}
    charted = {a.profile for m in machines for a in m['arms'] if a.suite in chart_suites} & owners
    if len(owners) != 1:
        charted = set()
    first_page = next(iter(chart_pages.values()), None)
    chart_href = {p: first_page.name for p in charted} if first_page else {}

    provenance = []
    for m in machines:
        notices, comparison_link = [], None
        if m['profile'] == ZEN5:
            if comparison is not None:
                n_cells = len(re.findall(r'<tr><td class="k">', comparison.read_text()))
                comparison_link = (f'<a href="{comparison.relative_to(index.parent)}">joined comparison</a>'
                                   f' <span class="sm">(colmat, faer and every vendor arm in one table, '
                                   f'{n_cells} cells)</span>')
            if refusal is not None:
                notices.append('<p><strong>The joined comparison is withheld:</strong> the vendor arms '
                               'of this recording were not accepted as one session. The pairing guard '
                               f'reported:</p>\n<pre>{html.escape(refusal)}</pre>\n')
            notices.append(provenance_html(arms, folded=True)
                           .replace('</ul>\n', '</ul>\n<p><strong>Known limitation:</strong> '
                                    + html.escape(limitation_text()) + '</p>\n', 1))
        else:
            first_suite = m['arms'][0].suite
            notices.append(provenance_html({a.vendor: a.run_dir for a in m['arms'] if a.suite == first_suite},
                                           folded=True))
        provenance.append(machine_html(
            m, chart_nav=chart_nav_html(chart_pages) if m['profile'] in charted else '',
            comparison=comparison_link, notices=notices, unpinned=m['profile'] in view['unpinned']))

    newest = max((a for m in machines for a in m['arms']), key=lambda a: a.date, default=None)
    revision = newest.commit[:7] if newest else ''
    version = newest.manifest['library'].get('version') if newest else ''
    first_chart = first_page.name if first_page else '#families'
    body = (
        LANDING_CSS
        + '<nav class="topnav sm"><span><a href="#overview">Overview</a> &middot; '
          f'<a href="{html.escape(first_chart)}">Throughput</a> &middot; <a href="#runs">Runs</a> &middot; '
          '<a href="history.html">History</a> &middot; <a href="#method">How the numbers are made</a></span></nav>\n'
        + '<section id="overview">\n'
        + headline_html(machines, families)
        + subline_html(revision, version)
        + tiles_html(machines, families, chart_href, unrecorded)
        + '</section>\n'
        + matrix_html(machines, families, chart_href)
        + across_pins_html(family_matrix.across_pins(history, machines), machines)
        + trend_html(trend)
        + method_html(machines)
        + '<h2 id="runs">Provenance and run pages, per machine</h2>\n'
        + ''.join(provenance)
        + (f'<p><a href="history.html">History of published records</a> '
           f'<span class="sm">({len(view["history"])} older page(s))</span></p>\n'))
    site.write_page(index.parent, 'index.html', 'colmat benchmark', body)
    print(f'wrote {index} ({len(machines)} machine(s) x {len(families)} families)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
