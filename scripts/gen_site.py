#!/usr/bin/env python3
"""Write the site under `result/`: the landing page, the history, the charts and the joined view.

    python3 scripts/gen_site.py [--repo-ref main]

Nothing it writes is committed (see `.gitignore`); the Pages workflow runs it on every deploy from
the committed `data/`, `result/` and `pins/`. The landing page, the history, the HTML copy of
every result page and the charts declared in `benchmarks/campaign.yaml` (`figures:`, `trend:`) are the
toolkit's (`campaign_toolkit.site`). The throughput charts are one page per recorded
`(dtype, order)` dataset (`result/throughput-<dtype>-<order>.html`, `benchmarks/throughput.yaml`
`variants:`), linked from the landing page instead of carried on it. What is this campaign's own is the four-way view: the MKL and
OpenBLAS arms of the current `faer-kernels` pages joined on the cell key by `gen_comparison.py`,
written to `result/zen5-cpu/faer-kernels/<rev12>-comparison.html` and linked from the landing page.

The site is not written when the pin has no current recording (`campaign_toolkit.pin_recorded`,
the same rule as `gen_index --check`): a landing page that says nothing is recorded at the pin while
the recordings sit in the history is the failure that rule exists to stop.
"""
import argparse
import html
import markdown
import pathlib
import re
import sys

import yaml

from campaign_toolkit import descriptor, figures, gen_index, pin_recorded, site
from campaign_toolkit.campaign import load

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gen_comparison  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
PIN = 'pins/t4a-faer.rev'   # historical name; it holds the colmat revision since the 2026-11-17 rename
SUITE = 'faer-kernels'
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
    """{vendor: run dir} of the current `faer-kernels` pages (the index's own rule), and the newest
    MKL run of an earlier revision, which states the unchanged row's control."""
    descriptor.set_root(ROOT)
    found = gen_index.collect()
    current, history = gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
    pages = {c['vendor']: c['page'] for c in current if c['suite'] == SUITE and c['page'] is not None}
    arms = {vendor: page['run_dir'] for vendor, page in pages.items()}
    revision = pages['mkl']['commit'] if 'mkl' in pages else None
    # history is newest first; the control is the newest MKL arm of another revision
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
    try:
        gen_comparison.main(argv)
    except gen_comparison.PairingRefused as refusal:
        # Only the pairing guard's own refusal is published around. Any other SystemExit from
        # gen_comparison (missing result files, an arm under the wrong column) still escapes and
        # fails the deploy, which is what a broken run must do.
        return None, str(refusal)
    return out, None


def provenance_html(arms):
    """The landing page's provenance section, read from the current arms' own manifests.

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
        f"the small suite's declared grid: {html.escape(dtypes)} at {threads}T, both orders, every "
        "declared regime",
    ]
    out = '<section>\n<h2>Current recordings: scope and provenance</h2>\n<ul>\n'
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
    if first['host'].get('role') != 'reference':
        out += ('<p>Evo was reserved for measurement, but the recorded <code>host.role='
                + html.escape(str(first['host'].get('role'))) + '</code> is unchanged: these reports '
                'retain their profile\'s <strong>provisional</strong> designation.</p>\n')
    out += ('<p>The wrong-pin partial <code>20261031T222831Z-pub-3a8edf02-20261031-mkl</code> is '
            'preserved locally and explicitly excluded from this publication. One rejected '
            'small-OpenBLAS 8T attempt is retained in its raw archive, not counted as an accepted '
            'sample.</p>\n</section>')
    return out


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
    it named no dataset, and the recorded c64 cells had no page at all. Each page repeats the
    landing page's provenance and the suite's limitation beside the charts of its own dataset,
    with the four-way nav; the charts are the toolkit's (`figures.site_figures` wrote one set
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
        body = (f'<nav class="muted">dataset: {nav}</nav>\n{provenance}\n{notes}\n'
                f'{revisions}\n' + figures.grid_html(ROOT / 'result', entries))
        pages[variant] = site.write_page(
            ROOT / 'result', f'throughput-{variant[0]}-{variant[1]}.html',
            f'Throughput charts: {variant_label(variant)}', body)
    return pages


def throughput_index_html(pages):
    """The landing page's index of the dataset pages, in the declared order (f64 / F first)."""
    items = ''.join(
        f'<li><a href="{page.name}">{html.escape(variant_label(variant))}</a>'
        + (' &mdash; the column-major default view' if variant == ('f64', 'f') else '')
        + '</li>\n'
        for variant, page in pages.items())
    return ('<h2>Throughput charts</h2>\n'
            '<p>One page per recorded dataset, each with its own provenance, chart notes and '
            'charts:</p>\n<ul>\n' + items + '</ul>\n')


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
    extra = provenance_html(arms)
    extra += f'<p><strong>Known limitation:</strong> {html.escape(limitation_text())}</p>\n'
    # The pages below were recorded by a harness that charged the vendor arms for work the faer arms
    # avoided (#74), so their vendor ratios are not library comparisons. The joined view withdraws
    # those cells (gen_comparison.FAIR_HARNESS); a reader who lands here must not have to infer it
    # from a column, so the landing page says it too -- and stops saying it once the current pages
    # were recorded by a fair harness.
    if any(not gen_comparison.fair_harness(load(arm / 'run.yaml').get('harness', {}).get('commit'))
           for arm in arms.values()):
        extra += (
            '<h2>Vendor comparisons are withdrawn</h2>\n'
            '<p>The harness that recorded these pages charged the vendor arms more than the faer arms:\n'
            'a LAPACKE workspace, output copies and layout transposes inside the timed call, where the\n'
            'faer rows ran prepared. A <code>MKL/fork</code> or <code>OpenBLAS/fork</code> cell on these\n'
            'pages is therefore not a library comparison. The four-way view marks those cells\n'
            '<em>withdrawn</em>; <a href="https://github.com/tensor4all/colmat-benchmark/blob/main/'
            'docs/dataset-status.md">docs/dataset-status.md</a> is the status of record; the harness\n'
            'now charges both sides the same contract, so the next recording prints these ratios\n'
            'again.</p>\n')
    # The charts leave the landing page: one page per (dtype, order) dataset is written here
    # and linked below, so the landing page stays a short index and a chart names the dataset it
    # came from. The figures are generated once (from the runs behind the current pages, the
    # same rule `site.write_site` applies) and handed to it, which then shows only what did not
    # move -- the trend charts.
    view = site.campaign_view(ROOT, index.parent)
    current = [ROOT / c['page']['page'] for c in view['current'] if c['page']]
    generated = figures.site_figures(ROOT, index.parent, current)
    extra += throughput_index_html(write_throughput_pages(generated, arms))
    if comparison is not None:
        cells = len(re.findall(r'<tr><td class="k">', comparison.read_text()))
        rel = comparison.relative_to(index.parent)
        extra += ('<h2>The four-way view</h2>\n'
                 f'<p><a href="{rel}">The same recording against upstream, OpenBLAS and MKL in one '
                 f'table ({cells} cells)</a>, with the unchanged row\'s control in its header.</p>')
    if refusal is not None:
        extra += ('<h2>The four-way view is withheld</h2>\n'
                  '<p>The MKL and OpenBLAS arms of this recording were not accepted as one session, '
                  'so no table joins them here. Each page above states the revision that recorded '
                  'it, and each stands on its own. The pairing guard reported:</p>\n'
                  f'<pre>{html.escape(refusal)}</pre>\n')
    first = load(next(iter(arms.values())) / 'run.yaml')
    revision = first['library']['commit'][:7]
    source = html.escape(first['library']['url'].rsplit('/', 1)[-1])
    recorded = first['timestamp'][:10]
    site.write_site(index.parent, pins=[PIN], repo_ref=args.repo_ref, title='colmat benchmark',
                    intro=(f'The recordings at the measured revision {revision} of {source}, one '
                           f'section per hardware profile. These are the {recorded} zen5-cpu '
                           'recordings.'),
                    extra_html=extra, root=ROOT,
                    charts=[f for f in generated if f['kind'] != 'throughput'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
