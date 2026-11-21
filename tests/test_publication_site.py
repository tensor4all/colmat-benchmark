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
from campaign_toolkit import pin_recorded  # noqa: E402
from campaign_toolkit.stack import rev_file_value  # noqa: E402

# The arms this publication recorded: both suites on the MKL, OpenBLAS and AOCL arms (AOCL added by
# #95; the large OpenBLAS arm, once dropped by decision on 2026-11-18, is back in the same-pin Docker
# refresh #99). `benchmarks/campaign.yaml` `figures.runs` declares the same six positions; adding an
# arm here without its run would make this test assert a page that does not exist.
RECORDED = {('faer-kernels', 'mkl'), ('faer-kernels', 'openblas'), ('faer-kernels', 'aocl'),
            ('faer-kernels-large', 'mkl'), ('faer-kernels-large', 'openblas'),
            ('faer-kernels-large', 'aocl')}


def unrecorded_throughput_runs():
    """`run N` labels of the declared throughput runs that have no current page, in order."""
    runs = yaml.safe_load((ROOT / 'benchmarks/campaign.yaml').read_text())['figures']['throughput']['runs']
    return [f'run {i}' for i, r in enumerate(runs) if (r['suite'], r['vendor']) not in RECORDED]


def collect():
    gen_site.descriptor.set_root(ROOT)
    return gen_site.gen_index.collect()


def current_pages():
    found = collect()
    current, _ = gen_site.gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
    return [c['page'] for c in current if c['page'] is not None]


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

    def test_guard_passes_at_the_pin_and_fails_at_the_unrecorded_release(self):
        found = collect()
        self.assertEqual(pin_recorded.campaign_problems(found), [])
        unrecorded = '0' * 40
        moved = dict(found, resolved=dict(found['resolved'], library=unrecorded),
                     references=dict(found['references'], library=unrecorded))
        problems = pin_recorded.campaign_problems(moved)
        self.assertEqual(len(problems), 1)
        self.assertIn(f'pins/t4a-faer.rev = {unrecorded}', problems[0])
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
        # The measured profile lists its recordings; zen2 profiles were never recorded and say so.
        revision = rev_file_value((ROOT / gen_site.PIN).read_text())
        zen5 = index.split('<h2 id="zen5-cpu">', 1)[1].split('<h2', 1)[0]
        self.assertNotIn('Nothing recorded at the current pin', zen5)
        self.assertIn(f'<code>{revision[:12]}</code>', zen5)
        # The page states the suite's own limitation and the provenance of the runs it shows.
        self.assertIn(html.escape(gen_site.limitation_text()), index)
        self.assertIn('Current recordings: scope and provenance', index)
        for suite, vendor in RECORDED:
            link = f'zen5-cpu/{suite}/{revision[:12]}-0539947ffb75-{vendor}-timing-v2.html'
            self.assertIn(f'href="{link}"', index)
            self.assertTrue((ROOT / 'result' / link).is_file(), link)
        # The two daily arms of this recording were recorded either side of a data commit: their
        # harness commits differ, the paths the harness is built from do not, so the join is accepted
        # (#82) and the page links the table instead of a refusal.
        comparison = f'zen5-cpu/{gen_site.SUITE}/{revision[:12]}-comparison.html'
        self.assertNotIn('The four-way view is withheld', index)
        self.assertIn(f'href="{comparison}"', index)
        joined = (ROOT / 'result' / comparison).read_text()
        for column in ('fork MKL (ms)', 'fork OpenBLAS (ms)', 'MKL (ms)', 'OpenBLAS (ms)'):
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
        for suite in ('faer-kernels', 'faer-kernels-large'):
            with self.subTest(suite=suite):
                data = yaml.safe_load((ROOT / 'benchmarks/suites' / (suite + '.yaml')).read_text())
                self.assertIn(pin[:7], data['engines_doc']['faer-fork']['caveat'])
                self.assertTrue(any(s.startswith(gen_site.LIMITATION_PREFIX)
                                    for s in data['workloads']['known_blind_spots']))
        # One limitation, stated once: the landing page's text is the measured suite's bullet.
        small = yaml.safe_load((ROOT / 'benchmarks/suites/faer-kernels.yaml').read_text())
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
        # column-major default view, first -- and no throughput chart is carried inline any
        # more; what is left on it is the trend section.
        links = re.findall(r'<li><a href="(throughput-\w+-\w\.html)">([^<]+)</a>', index)
        self.assertEqual(links, [('throughput-f64-f.html', 'f64 / F order'),
                                 ('throughput-f64-c.html', 'f64 / C order'),
                                 ('throughput-c64-f.html', 'c64 / F order'),
                                 ('throughput-c64-c.html', 'c64 / C order')])
        self.assertIn('the column-major default view', index)
        self.assertIn('The four-way view', index)
        inline = re.findall(r'<img src="([^"]+)"', index)
        self.assertTrue(inline)
        for image in inline:
            self.assertTrue((ROOT / 'result' / image).is_file(), image)
            self.assertIn('trend/', image)
        # Each dataset page carries its own title, the four-way nav (the current page bold,
        # the others links), the landing page's provenance and the suite's limitation, and only
        # the charts of its own dataset.
        for dtype, order in self.PAGES:
            page = (ROOT / 'result' / f'throughput-{dtype}-{order}.html').read_text()
            title = f'Throughput charts: {dtype} / {order.upper()} order'
            self.assertIn(f'<title>{title}</title>', page)
            self.assertIn(f'<h1>{title}</h1>', page)
            self.assertEqual(len(re.findall(r'href="throughput-\w+-\w\.html"', page)), 3)
            self.assertIn('<strong>%s / %s order</strong>' % (dtype, order.upper()), page)
            self.assertIn('Current recordings: scope and provenance', page)
            self.assertIn(html.escape(gen_site.limitation_text()), page)
            # The runs behind the charts are stated on the page itself, including the
            # large-tier arm the daily provenance does not name and the arm not recorded.
            self.assertIn('Revisions:', page)
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

    def test_the_c64_charts_come_from_the_recorded_c64_cells(self):
        # The c64 page is not the f64 page under another name: its plotted numbers are the
        # recorded c64 cells with the complex x4 count, recomputed here from the current
        # large-tier MKL run's own CSV -- an f64 copy, or a forgotten x4, fails this.
        from campaign_toolkit import throughput
        run = current_run_dir('faer-kernels-large', 'mkl')
        self.assertIsNotNone(run)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_site.main([]), 0)
        cells = throughput.load_run(run, dtype='c64', order='c')
        points = dict((n, gflops) for n, gflops, _ in
                      throughput.points(cells, 4, 'gemm', 'faer-fork', 2.0, complex_dtype=True))
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


if __name__ == '__main__':
    unittest.main()
