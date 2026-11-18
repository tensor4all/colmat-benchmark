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

# The arms this publication recorded: the daily suite on both vendor arms, the large suite on MKL
# only. The large OpenBLAS arm was dropped by decision (2026-11-18) and is stated as unrecorded
# on the landing page and in docs/worklog.md; adding it here without adding the run would make
# this test assert a page that does not exist.
RECORDED = {('faer-kernels', 'mkl'), ('faer-kernels', 'openblas'), ('faer-kernels-large', 'mkl')}


def collect():
    gen_site.descriptor.set_root(ROOT)
    return gen_site.gen_index.collect()


def current_pages():
    found = collect()
    current, _ = gen_site.gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
    return [c['page'] for c in current if c['page'] is not None]


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


if __name__ == '__main__':
    unittest.main()
