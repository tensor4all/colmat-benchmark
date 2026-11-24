"""Publication identity checks; no build or measurement is run."""
import contextlib
import html
import io
import pathlib
import re
import sys
import unittest
from unittest import mock

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import gen_site  # noqa: E402
from campaign_toolkit import pin_recorded, report_run  # noqa: E402
from campaign_toolkit.stack import rev_file_value  # noqa: E402

# The arms this publication recorded: both suites on the MKL, OpenBLAS and AOCL arms (AOCL added by
# #95; the large OpenBLAS arm, once dropped by decision on 2026-11-18, is back in the same-pin Docker
# refresh #99). `benchmarks/campaign.yaml` `figures.runs` declares the same six positions; adding an
# arm here without its run would make this test assert a page that does not exist.
RECORDED = {('colmat-kernels', 'mkl'), ('colmat-kernels', 'openblas'), ('colmat-kernels', 'aocl'),
            ('colmat-kernels-large', 'mkl'), ('colmat-kernels-large', 'openblas'),
            ('colmat-kernels-large', 'aocl')}


def unrecorded_throughput_runs():
    """`run N` labels of the declared throughput runs that have no current page, in order."""
    runs = yaml.safe_load((ROOT / 'benchmarks/campaign.yaml').read_text())['figures']['throughput']['runs']
    return [f'run {i}' for i, r in enumerate(runs) if (r['suite'], r['vendor']) not in RECORDED]


def collect():
    gen_site.descriptor.set_root(ROOT)
    return gen_site.gen_index.collect()


def current_pages(profile='zen5-cpu'):
    """The current pages of one hardware profile: zen5-cpu is the publication; keyaki has its own."""
    found = collect()
    current, _ = gen_site.gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
    return [c['page'] for c in current if c['page'] is not None and c['page']['profile'] == profile]


def current_run_dir(suite, vendor):
    """The run directory behind the current (suite, vendor) page, or None."""
    found = collect()
    current, _ = gen_site.gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
    for cell in current:
        if cell['suite'] == suite and cell['vendor'] == vendor and cell['page'] is not None:
            return pathlib.Path(cell['page']['run_dir'])
    return None


class PublicationSiteTest(unittest.TestCase):
    def test_the_pin_is_the_revision_every_current_page_records(self):
        pin = rev_file_value((ROOT / gen_site.PIN).read_text())
        pages = current_pages()
        self.assertEqual({(p['suite'], p['vendor']) for p in pages}, RECORDED)
        self.assertEqual({p['library_commit'] for p in pages}, {pin})
        self.assertRegex(pin, r'^[0-9a-f]{40}$')

    def test_both_arms_of_both_suites_are_the_current_cells(self):
        for page in current_pages():
            self.assertEqual(page['profile'], 'zen5-cpu')
        self.assertEqual(len({p['upstream_commit'] for p in current_pages()}), 1)

    def test_keyaki_pages_are_their_own_profiles_and_the_64t_one_carries_its_caveat(self):
        pin = rev_file_value((ROOT / gen_site.PIN).read_text())
        for profile, suite in (('keyaki-64', 'colmat-kernels-keyaki'),
                               ('keyaki-64-os-shared', 'colmat-kernels-keyaki-os-shared')):
            pages = current_pages(profile)
            self.assertEqual({(p['suite'], p['vendor']) for p in pages}, {(suite, 'mkl')})
            self.assertEqual({p['library_commit'] for p in pages}, {pin})
        # the 64T cell is os-shared and says so on its page, where a reader of the table sees it
        shared = pathlib.Path(current_pages('keyaki-64-os-shared')[0]['run_dir']) / 'report.md'
        self.assertIn('OS-SHARED, NOT COMPARABLE WITH THE 60T ROW', shared.read_text())

    def test_guard_passes_at_the_pin_and_fails_at_the_unrecorded_release(self):
        found = collect()
        self.assertEqual(pin_recorded.campaign_problems(found), [])
        unrecorded = '0' * 40
        moved = dict(found, resolved=dict(found['resolved'], library=unrecorded),
                     references=dict(found['references'], library=unrecorded))
        problems = pin_recorded.campaign_problems(moved)
        self.assertEqual(len(problems), 1)
        self.assertIn(f'pins/colmat.rev = {unrecorded}', problems[0])
        self.assertIn('no current recording', problems[0])
        # The site refuses to publish in that state rather than saying nothing was recorded.
        with mock.patch.object(gen_site.gen_index, 'collect', return_value=moved), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(gen_site.main([]), 1)
        self.assertIn('the site is not written', err.getvalue())

    def test_comparison_filename_is_the_measured_commit(self):
        commit = 'a' * 40
        arms = {v: pathlib.Path('/unused') / v for v in ('mkl', 'openblas')}
        with mock.patch.object(gen_site, 'current_arms', return_value=(arms, None)), \
             mock.patch.object(gen_site, 'load', return_value={'library': {'commit': commit}}), \
             mock.patch.object(gen_site.gen_comparison, 'main') as comparison:
            out, refusal = gen_site.write_comparison()
        self.assertIsNone(refusal)
        self.assertEqual(out.name, commit[:12] + '-comparison.html')
        args = comparison.call_args.args[0]
        self.assertEqual(args[args.index('--out') + 1], str(out))

    def test_a_refused_pairing_withholds_the_view_instead_of_the_site(self):
        # The pairing guard is the authority on whether two arms are one session. Its refusal must
        # not abort the landing page: the pages that stand on their own are still published, and the
        # page says the joined view is withheld, naming what the guard objected to.
        commit = 'a' * 40
        arms = {v: pathlib.Path('/unused') / v for v in ('mkl', 'openblas')}
        refusal = 'ERROR: the mkl arm and the openblas arm ... harness.commit: mkl a vs openblas b'
        with mock.patch.object(gen_site, 'current_arms', return_value=(arms, None)), \
             mock.patch.object(gen_site, 'load', return_value={'library': {'commit': commit}}), \
             mock.patch.object(gen_site.gen_comparison, 'main',
                               side_effect=gen_site.gen_comparison.PairingRefused(refusal)):
            out, reported = gen_site.write_comparison()
        self.assertIsNone(out)
        self.assertEqual(reported, refusal)

    def test_a_broken_run_still_fails_the_site(self):
        # Only the pairing guard's refusal is published around. A run whose files are missing publishes
        # a page with no numbers in it, so that exit has to abort the deploy instead of being framed as
        # "these arms are not one session".
        arms = {v: pathlib.Path('/unused') / v for v in ('mkl', 'openblas')}
        broken = 'ERROR: the run at /x names no result files in run.yaml'
        with mock.patch.object(gen_site, 'current_arms', return_value=(arms, None)), \
             mock.patch.object(gen_site, 'load', return_value={'library': {'commit': 'a' * 40}}), \
             mock.patch.object(gen_site.gen_comparison, 'main', side_effect=SystemExit(broken)):
            with self.assertRaises(SystemExit) as stopped:
                gen_site.write_comparison()
        self.assertEqual(str(stopped.exception), broken)

    def test_site_shows_the_recordings_as_current_with_the_release_caveat(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_site.main([]), 0)
        index = (ROOT / 'result/index.html').read_text()
        # The site names the project it benchmarks (colmat), not the upstream library it measures
        # against, on the landing page's title and heading and in every page's title prefix.
        self.assertIn('<title>colmat benchmark</title>', index)
        self.assertIn('<h1>colmat benchmark</h1>', index)
        # Each recorded machine has a folded provenance section with its run pages; the zen2
        # profiles were never recorded and share one line instead of a section each (#116).
        revision = rev_file_value((ROOT / gen_site.PIN).read_text())
        zen5 = index.split('<details class="machine" id="runs-zen5-cpu">', 1)[1].split('<details class="machine"', 1)[0]
        self.assertIn(f'<code>{revision[:12]}</code>', zen5)
        line = index.split('Not recorded yet:', 1)[1].split('</p>', 1)[0]
        self.assertIn('<code>zen2-cpu</code>', line)
        self.assertIn('<code>zen2-64-smt-off</code>', line)
        self.assertNotIn('id="runs-zen2-cpu"', index)
        # The zen5 section states the suite's own limitation and the provenance of its runs, folded
        # and named for its profile rather than as a site-wide block.
        self.assertIn(html.escape(gen_site.limitation_text()), zen5)
        self.assertIn('<summary>Current zen5-cpu recordings: scope and provenance</summary>', zen5)
        self.assertNotIn('<h2>Current', index)
        pages = {(p['suite'], p['vendor']): p for p in current_pages()}
        for suite, vendor in RECORDED:
            # The page file stays where it was recorded (a run recorded before #113 sits under
            # `faer-kernels*`); the link text names the suite by its current id.
            link = pathlib.PurePosixPath(pages[(suite, vendor)]['page']).relative_to('result').with_suffix('.html')
            self.assertEqual(link.name, f'{revision[:12]}-0539947ffb75-{vendor}-timing-v2.html')
            self.assertIn(f'href="{link}"', index)
            # titled by suite and vendor, not by the file name
            label = gen_site.family_matrix.VENDOR_LABELS[vendor]
            self.assertIn(f'href="{link}">{suite} &middot; {label}</a>', zen5)
            self.assertNotIn(f'>{link.with_suffix(".md").name}<', index)
            self.assertTrue((ROOT / 'result' / link).is_file(), link)
        # The two daily arms of this recording were recorded either side of a data commit: their
        # harness commits differ, the paths the harness is built from do not, so the join is accepted
        # (#82) and the page links the table instead of a refusal.
        comparison = f'zen5-cpu/{gen_site.SUITE}/{revision[:12]}-comparison.html'
        self.assertNotIn('The four-way view is withheld', index)
        self.assertIn(f'href="{comparison}"', index)
        joined = (ROOT / 'result' / comparison).read_text()
        for column in ('colmat MKL (ms)', 'colmat OpenBLAS (ms)', 'MKL (ms)', 'OpenBLAS (ms)'):
            self.assertIn(f'<th>{column}</th>', joined)
        self.assertIn('<tr><td class="k">', joined)
        images = re.findall(r'<img src="([^"]+)"', index)
        self.assertTrue(images)
        for image in images:
            self.assertTrue((ROOT / 'result' / image).is_file(), image)

    def test_both_report_declarations_state_the_measured_revision_and_the_limitation(self):
        # The site repeats these from the declarations instead of restating them, so what is
        # checked is that each declaration states them at all and names the revision it measures.
        pin = rev_file_value((ROOT / gen_site.PIN).read_text())
        for suite in ('colmat-kernels', 'colmat-kernels-large'):
            with self.subTest(suite=suite):
                data = yaml.safe_load((ROOT / 'benchmarks/suites' / (suite + '.yaml')).read_text())
                self.assertIn(pin[:7], data['engines_doc']['colmat']['caveat'])
                self.assertTrue(any(s.startswith(gen_site.LIMITATION_PREFIX)
                                    for s in data['workloads']['known_blind_spots']))
        # One limitation, stated once: the landing page's text is the measured suite's bullet.
        small = yaml.safe_load((ROOT / 'benchmarks/suites/colmat-kernels.yaml').read_text())
        self.assertTrue(any(gen_site.limitation_text() in spot
                            for spot in small['workloads']['known_blind_spots']))


class ThroughputPagesTest(unittest.TestCase):
    """One page per recorded (dtype, order) dataset, linked from a short landing index (#90)."""

    PAGES = [('f64', 'f'), ('f64', 'c'), ('c64', 'f'), ('c64', 'c')]

    def test_four_dataset_pages_and_a_short_landing_index(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_site.main([]), 0)
        index = (ROOT / 'result' / 'index.html').read_text()
        # The landing page links the four dataset pages in the declared order -- f64 / F, the
        # column-major default view, first -- on one line of the zen5-cpu section, not as a bullet
        # list of its own (#116); no throughput chart is carried inline, only the trend section.
        links = re.findall(r'<a href="(throughput-\w+-\w\.html)">(\w+ / \w order)</a>', index)
        # the tile and the top nav reach the default view in one click
        self.assertIn('<a href="throughput-f64-f.html">Throughput charts &rarr;</a>', index)
        self.assertEqual(links, [('throughput-f64-f.html', 'f64 / F order'),
                                 ('throughput-f64-c.html', 'f64 / C order'),
                                 ('throughput-c64-f.html', 'c64 / F order'),
                                 ('throughput-c64-c.html', 'c64 / C order')])
        self.assertIn('(default view)', index)
        self.assertIn('joined comparison', index)
        inline = re.findall(r'<img src="([^"]+)"', index)
        self.assertTrue(inline)
        for image in inline:
            self.assertTrue((ROOT / 'result' / image).is_file(), image)
            self.assertIn('trend/', image)
        # the trend figures carry a log y axis (charts.log_axis in the toolkit); the caption says so
        self.assertIn('log axis', index)
        self.assertNotIn('linear axis', index)
        # Each dataset page carries its own title, the four-way nav (the current page bold,
        # the others links) and only the charts of its own dataset; the landing page's provenance
        # and the suite's limitation follow the charts (#106).
        for dtype, order in self.PAGES:
            page = (ROOT / 'result' / f'throughput-{dtype}-{order}.html').read_text()
            title = f'Throughput charts: {dtype} / {order.upper()} order'
            self.assertIn(f'<title>{title}</title>', page)
            self.assertIn(f'<h1>{title}</h1>', page)
            self.assertEqual(len(re.findall(r'href="throughput-\w+-\w\.html"', page)), 3)
            self.assertIn('<strong>%s / %s order</strong>' % (dtype, order.upper()), page)
            self.assertIn('<h2>Current zen5-cpu recordings: scope and provenance</h2>', page)
            self.assertIn(html.escape(gen_site.limitation_text()), page)
            missing = unrecorded_throughput_runs()
            if missing:
                self.assertIn('No run was given for ' + ', '.join(missing), page)
            else:
                self.assertNotIn('No run was given for', page)
            images = re.findall(r'<img src="([^"]+)"', page)
            self.assertEqual(len(images), 9, (dtype, order))   # every declared family
            for image in images:
                self.assertTrue((ROOT / 'result' / image).is_file(), image)
                self.assertIn(f'/{dtype}-{order}/', image)
            # #106: the charts start where the page opens -- title, one line of dataset nav, then
            # the charts -- and everything a reader consults after reading them follows below:
            # the provenance, the limitation, the config's chart notes and the revisions line that
            # states the runs behind the charts (the large-tier arm the daily provenance does not
            # name, and the arm not recorded).
            self.assertLess(page.index('<nav class="muted">dataset:'), page.index('<img src='))
            last_chart = page.rindex('<img src=')
            for below in ('Current zen5-cpu recordings: scope and provenance',
                          html.escape(gen_site.limitation_text()),
                          'Generated by the Pages deploy',   # the notes' own first words
                          'Revisions:'):
                self.assertGreater(page.index(below), last_chart, below)

    def test_the_c64_charts_come_from_the_recorded_c64_cells(self):
        # The c64 page is not the f64 page under another name: its plotted numbers are the
        # recorded c64 cells with the complex x4 count, recomputed here from the current
        # large-tier MKL run's own CSV -- an f64 copy, or a forgotten x4, fails this.
        from campaign_toolkit import throughput
        run = current_run_dir('colmat-kernels-large', 'mkl')
        self.assertIsNotNone(run)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_site.main([]), 0)
        cells = throughput.load_run(run, dtype='c64', order='c')
        points = dict((n, gflops) for n, gflops, _ in
                      throughput.points(cells, 4, 'gemm', 'colmat', 2.0, complex_dtype=True))
        table = (ROOT / 'result' / 'figures' / 'throughput' / 'zen5-cpu' / 'c64-c'
                 / 'throughput-gemm-v2.md').read_text()
        shown = float(re.search(r'^\| 1024 \| ([0-9.e+-]+) ', table, re.M).group(1))
        self.assertAlmostEqual(shown / points[1024], 1.0, places=2)
        # Where the MKL arm was recorded in c64 (gemm, both orders; the decomposition families
        # in F order only), its column is plotted; in C order the decomposition columns are
        # dots, exactly as recorded.
        for dtype, order, family, has_mkl in (('c64', 'f', 'lu', True), ('c64', 'c', 'lu', False),
                                              ('c64', 'c', 'gemm', True)):
            table = (ROOT / 'result' / 'figures' / 'throughput' / 'zen5-cpu' / f'{dtype}-{order}'
                     / f'throughput-{family}-v2.md').read_text()
            four_t = table.split('## 4T', 1)[1].split('## 8T', 1)[0]
            rows = [line for line in four_t.splitlines() if re.match(r'^\| \d+ \|', line)]
            self.assertTrue(rows, (dtype, order, family))
            self.assertEqual(('. |' in rows[0]), not has_mkl, rows[0])


class LandingPageTest(unittest.TestCase):
    """The landing page is the maintainer's mockup on #116, generated from the data."""

    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            assert gen_site.main([]) == 0
        cls.index = (ROOT / 'result' / 'index.html').read_text()
        fm = gen_site.family_matrix
        cls.order, cls.pages, cls.history = fm.collect()
        cls.machines = {m['profile']: m for m in fm.machines(cls.order, cls.pages)}

    def test_the_sections_come_in_the_mockups_order(self):
        index = self.index
        marks = ['class="headline"', '<div class="tiles">', 'Not recorded yet:', '<table class="share">',
                 '<h2 id="pins">', '<h2 id="method">', '<h2 id="runs">', 'History of published records']
        positions = [index.index(m) for m in marks]
        self.assertEqual(positions, sorted(positions))
        # one tile and one column per recorded machine, the os-shared run folded into its machine
        self.assertEqual(list(self.machines), ['zen5-cpu', 'keyaki-64', 'ohtaka-fat'])
        head = index.split('<table class="share">', 1)[1].split('</thead>', 1)[0]
        self.assertEqual(re.findall(r'<a href="#([^"]+)">', head), list(self.machines))
        self.assertEqual(index.count('<div class="tile" id='), 3)

    def test_the_headline_names_its_comparison_set(self):
        headline = self.index.split('class="headline">', 1)[1].split('</p>', 1)[0]
        fm = gen_site.family_matrix
        for m in self.machines.values():
            split = fm.percentages(m['total'])
            self.assertEqual(sum(split), 100)
            self.assertIn(f'<b>{split[0]}%</b>', headline)
            # the tile prints counts next to "of N cells"; the big number is the percentage split
            self.assertIn(f'{m["total"]["win"]} / {m["total"]["tie"]} / {m["total"]["lose"]} '
                          f'(win / tie / lose', self.index)
            self.assertIn(f"{m['arch']} (vs {gen_site._vendors(m['compared'])}", headline)
        # Cascade Lake covers 4 of the 9 families and the sentence says so
        self.assertIn('Intel Cascade Lake (vs MKL, 4 of 9 families measured)', headline)
        self.assertIn('<code>lu</code>', headline)

    def test_tiles_carry_counts_vendors_coverage_threads_date_and_pin(self):
        tile = self.index.split('<div class="tile" id="keyaki-64">', 1)[1].split('<div class="tile"', 1)[0]
        m = self.machines['keyaki-64']
        self.assertIn('AMD Zen 2', tile)
        self.assertIn('(AVX2)', tile)
        self.assertIn('EPYC 7702P &middot; 64 cores', tile)
        self.assertIn('vs MKL (OpenBLAS, AOCL not measured)', tile)
        self.assertIn('9 of 9 families &middot; 16&ndash;60 threads &middot; 2026-11-22 &middot; current pin', tile)
        # the os-shared 64T run is listed with its machine and never counted
        self.assertIn('64T os-shared run listed, not counted', tile)
        self.assertEqual(sorted({t for a in m['arms'] for t in a.manifest['threads']['counts']}), [16, 32, 60])
        ohtaka = self.index.split('<div class="tile" id="ohtaka-fat">', 1)[1].split('</div>\n', 1)[0]
        self.assertIn('vs MKL (OpenBLAS not measured)', ohtaka)      # AOCL is an AMD-host arm
        self.assertIn('<b>4 of 9 families</b>', ohtaka)
        self.assertIn('below the timing baseline', ohtaka)
        # the 112T row measures every core: it is listed with the machine and never counted
        self.assertIn('16&ndash;64 threads', ohtaka)
        self.assertIn('112T row measures every core', ohtaka)
        # and the below-baseline margin is the page's own rule, not a footnote
        self.assertIn('ratio beyond 1.10&times;', ohtaka)

    def test_a_full_node_row_is_listed_and_not_counted(self):
        m = self.machines['ohtaka-fat']
        self.assertEqual(m['os_shared'], [112])
        self.assertNotIn(112, m['threads'])
        # no counted cell comes from the os-shared row, and the share counts 24 cells, not 32
        counted = [c for f in m['families'].values() for c in f['cells']]
        self.assertTrue(counted)
        self.assertTrue(all(c['at'][3] != 112 for c in counted))
        self.assertEqual(sum(c['win'] + c['tie'] + c['lose'] for c in m['families'].values()),
                         len(counted))

    def test_a_below_baseline_run_applies_the_campaign_margin(self):
        fm = gen_site.family_matrix
        arms = self.machines['ohtaka-fat']['arms']
        arm = next(a for a in arms if a.suite == 'faer-kernels-ohtaka-fat-factor')
        at = ('cholesky', (2048, 2048, ''), 'f64', 16)
        # 1.0666x: inside 1.10, so the page's rule calls it a tie, though it clears the 5% band
        self.assertFalse(arm.verdict(at))
        key, names = arm.cells[at]
        unbounded = report_run.comparison_verdict(
            arm.best, arm.within, key, [names[fm.COLMAT], names[arm.vendor_row]])
        self.assertIsNot(unbounded.get(names[arm.vendor_row]), False)

    def test_provenance_prints_the_order_the_recorded_suite_declares(self):
        m = self.machines['ohtaka-fat']
        suite = m['arms'][0].suite
        manifest = gen_site.load(next(a for a in m['arms'] if a.suite == suite).run_dir / 'run.yaml')
        self.assertEqual(gen_site._declared_orders(manifest), 'F (column-major) order only')
        text = gen_site.provenance_html(
            {a.vendor: a.run_dir for a in m['arms'] if a.suite == suite}, folded=True)
        self.assertIn('F (column-major) order only', text)
        self.assertNotIn('both orders', text)
        zen5 = gen_site.load(self.machines['zen5-cpu']['arms'][0].run_dir / 'run.yaml')
        self.assertEqual(gen_site._declared_orders(zen5), 'both orders')

    def test_an_older_pin_stays_visible_with_its_badge(self):
        m = dict(self.machines['zen5-cpu'], behind=7)
        families = gen_site.family_matrix.family_order(m['families'])
        tile = gen_site.tile_html(m, families, {})
        self.assertIn('<span class="badge">older pin: 7 commits behind</span>', tile)
        # and a profile whose pages all sit in the history still yields a machine
        older = [r for r in self.history if r['profile'] == 'zen5-cpu']
        order, pages, _ = gen_site.family_matrix.collect()
        pages = {'zen5-cpu': []}
        newest = older[0]['library_commit']
        for r in older:
            if r['library_commit'] == newest and not any(c['suite'] == r['suite'] and c['vendor'] == r['vendor']
                                                         for c, _ in pages['zen5-cpu']):
                pages['zen5-cpu'].append(({'suite': r['suite'], 'vendor': r['vendor'], 'profile': 'zen5-cpu'}, r))
        [machine] = gen_site.family_matrix.machines(['zen5-cpu'], pages)
        self.assertGreater(machine['behind'], 0)

    def test_cells_are_the_recorded_verdicts_and_not_measured_is_white_and_dashed(self):
        index = self.index
        self.assertIn('table.share td.na{background:#fff;border:1px dashed', index)
        body = index.split('<table class="share">', 1)[1].split('</table>', 1)[0]
        rows = dict(re.findall(r'<th scope="row">([^<]+)</th>(.*?)</tr>', body))
        # ohtaka-fat recorded gemm, lu, cholesky and qr_factor only
        for family, cells in rows.items():
            last = re.findall(r'<td class="([^"]+)"', cells)[-1]
            self.assertEqual(last == 'na', family not in ('gemm', 'lu', 'cholesky', 'qr_factor'), family)
        # every coloured cell carries its word as well as its colour
        for cls, word in re.findall(r'<td class="(w|t|l)"><a [^>]*>\d+% &middot; ([a-z ]+)</a>', body):
            self.assertIn(word, {'w': ['mostly win'], 't': ['mixed', 'mostly tie'],
                                 'l': ['mostly lose']}[cls])
        # a counted cell is the page's own verdict: zen5 lu 1024 f64 8T loses to MKL by 1.09x on
        # the large-tier MKL page, and the tally records it as a loss against MKL
        lu = self.machines['zen5-cpu']['families']['lu']['cells']
        [cell] = [c for c in lu if c['at'] == ('lu', (1024, 1024, ''), 'f64', 8)]
        self.assertEqual((cell['vendor'], cell['result']), ('mkl', 'lose'))
        page = (ROOT / cell['page']).read_text()
        row = re.search(r'^## linalg-large, f64, f order, 8T.*?^\| `lu/n1024/b1` \|([^\n]*)', page,
                        re.M | re.S).group(1)
        self.assertIn('slower than `lapack-mkl`', row)

    def test_no_share_outlives_its_evidence(self):
        fm = gen_site.family_matrix
        self.assertEqual(fm.outcome(None), 'no_spread')
        self.assertEqual(fm.outcome(False), 'tie')
        self.assertEqual(fm.outcome((0.8, 0.05)), 'win')
        self.assertEqual(fm.outcome((1.2, 0.05)), 'lose')
        counts = {r: 0 for r in fm.RESULTS}
        self.assertIsNone(fm.share({**counts, 'no_spread': 4, 'withdrawn': 3}))
        self.assertEqual(fm.leaning({**counts, 'win': 3, 'tie': 1, 'lose': 2}), 'mixed')
        self.assertEqual(fm.leaning({**counts, 'win': 4, 'lose': 3}), 'win')
        # a row at n < 256, in C order, or of the api-inclusive route is never a cell
        row = {'status': 'ok', 'dtype': 'f64', 'order': 'f', 'batch': '1', 'note': '', 'm': '512',
               'n': '512', 'k': ''}
        self.assertTrue(fm.comparable(row))
        for change in ({'n': '128'}, {'order': 'c'}, {'note': 'api-inclusive'}, {'dtype': 'f32'}):
            self.assertFalse(fm.comparable({**row, **change}), change)

    def test_win_share_across_pins_starts_at_the_second_pin(self):
        index = self.index
        pins = index.split('<h2 id="pins">', 1)[1].split('<h2', 1)[0]
        # every older zen5 pin was recorded before the fair harness: no second share yet
        self.assertIn('appearing from the second pin', pins)
        self.assertIn('vendor rows withdrawn', pins)
        # with two pins the section draws the lines and their table
        fm = gen_site.family_matrix
        m = self.machines['zen5-cpu']
        points = {'zen5-cpu': [{'commit': 'a' * 40, 'date': '2026-11-01', 'counts': m['total'], 'why': None},
                               {'commit': m['commit'], 'date': m['date'], 'counts': m['total'], 'why': None}]}
        drawn = gen_site.across_pins_html(points, [m])
        self.assertIn('<polyline', drawn)
        self.assertIn('table: the plotted numbers', drawn)
        self.assertEqual(drawn.count('<tr><td>zen5-cpu</td>'), 2)
        self.assertTrue(fm.across_pins(self.history, [m])['zen5-cpu'])

    def test_a_pin_without_a_decided_cell_is_never_a_share_point(self):
        # The review's state on #125: an older pre-fair pin (every vendor row withdrawn) next to
        # the current pin. It raised TypeError in across_pins_html and failed the Pages deploy.
        fm = gen_site.family_matrix
        m = self.machines['zen5-cpu']
        withdrawn = {'win': 0, 'tie': 0, 'lose': 0, 'no_spread': 0, 'withdrawn': 348}
        why = 'vendor rows withdrawn (recorded before the fair harness)'
        # the consumer: even handed the undecided counts as a point, it never indexes a missing share
        review = {'zen5-cpu': [{'commit': 'a' * 40, 'date': '2026-10-16', 'counts': withdrawn, 'why': 'withdrawn'},
                               {'commit': m['commit'], 'date': m['date'], 'counts': m['total'], 'why': None}]}
        text = gen_site.across_pins_html(review, [m])
        self.assertIn('<code>zen5-cpu</code>: 1 pin with a share, 1 pin(s) without one (withdrawn)', text)
        self.assertNotIn('<polyline', text)
        # the producer: a pin with no decided cell carries no counts and says why, the current
        # pin included
        self.assertEqual(fm.pin_point('b' * 40, '2026-10-16', withdrawn),
                         {'commit': 'b' * 40, 'date': '2026-10-16', 'counts': None, 'why': why})
        only_old = {**m, 'total': withdrawn}
        points = fm.across_pins([], [only_old])['zen5-cpu']
        self.assertEqual([(p['counts'], p['why']) for p in points], [(None, why)])
        text = gen_site.across_pins_html({'zen5-cpu': points}, [only_old])
        self.assertIn('<code>zen5-cpu</code>: no pin with a share, 1 pin(s) without one '
                      '(vendor rows withdrawn (recorded before the fair harness))', text)
        self.assertNotIn('None', text)
        # one older decided pin plus a current pin without a decided cell: no crash, no line
        older = fm.pin_point('c' * 40, '2026-10-01', m['total'])
        text = gen_site.across_pins_html({'zen5-cpu': [older] + points}, [only_old])
        self.assertIn('1 pin with a share, 1 pin(s) without one', text)

    def test_the_page_and_the_cli_print_the_same_percentages(self):
        fm = gen_site.family_matrix
        # one rounding, exact on the counts, summing to 100; a tied remainder never goes to win
        self.assertEqual(fm.percentages(self.machines['zen5-cpu']['total']), [61, 18, 21])
        self.assertEqual(fm.percentages({'win': 1, 'tie': 1, 'lose': 1, 'no_spread': 0, 'withdrawn': 0}),
                         [33, 33, 34])
        self.assertEqual(fm.percentages({'win': 143, 'tie': 17, 'lose': 20, 'no_spread': 0, 'withdrawn': 0}),
                         [79, 10, 11])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fm.main()
        cli = out.getvalue()
        for m in self.machines.values():
            win, tie, lose = fm.percentages(m['total'])
            self.assertEqual(win + tie + lose, 100)
            self.assertIn(f'win {win}% / tie {tie}% / lose {lose}%', cli)
            self.assertIn(f'aria-label="win {win}%, tie {tie}%, lose {lose}%"', self.index)

    def test_only_the_machine_the_charts_were_drawn_from_links_to_them(self):
        for profile in ('keyaki-64', 'ohtaka-fat'):
            tile = self.index.split(f'<div class="tile" id="{profile}">', 1)[1].split('\n', 1)[0]
            self.assertNotIn('throughput-', tile, profile)
        zen5 = self.index.split('<div class="tile" id="zen5-cpu">', 1)[1].split('<div class="tile"', 1)[0]
        self.assertIn('throughput-f64-f.html', zen5)

    def test_how_the_numbers_are_made(self):
        method = self.index.split('<h2 id="method">', 1)[1].split('<h2', 1)[0]
        toolkit = re.search(r'campaign-toolkit@([0-9a-f]{40})', (ROOT / 'requirements.txt').read_text())
        pin = rev_file_value((ROOT / gen_site.PIN).read_text())
        for words in ('scripts/gen_site.py', 'scripts/family_matrix.py', 'run.yaml', toolkit.group(1),
                      pin, 'scripts/setup_extern_deps.sh', 'comparison_verdict', 'floored at 5%',
                      'Noise floor', 'os-shared', 'OS-SHARED, NOT COMPARABLE WITH THE 60T ROW'):
            self.assertIn(words, method)


if __name__ == '__main__':
    unittest.main()
