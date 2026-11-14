"""Publication identity checks; no build or measurement is run."""
import contextlib
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

RECORDED = {(suite, vendor) for suite in ('faer-kernels', 'faer-kernels-large') for vendor in ('mkl', 'openblas')}


def collect():
    gen_site.descriptor.set_root(ROOT)
    return gen_site.gen_index.collect()


class PublicationSiteTest(unittest.TestCase):
    def test_pin_is_the_measured_revision_not_the_release_source(self):
        pin = rev_file_value((ROOT / gen_site.PIN).read_text())
        self.assertEqual(pin, gen_site.MEASURED)

    def test_the_four_71b25ffb_recordings_are_the_current_cells(self):
        found = collect()
        current, _ = gen_site.gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
        pages = [c['page'] for c in current if c['page'] is not None]
        self.assertEqual({(p['suite'], p['vendor']) for p in pages}, RECORDED)
        for page in pages:
            self.assertEqual(page['profile'], 'zen5-cpu')
            self.assertEqual(page['library_commit'], gen_site.MEASURED)
            self.assertEqual(page['upstream_commit'], gen_site.UPSTREAM)

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
        arms = {v: pathlib.Path('/unused') / v for v in ('mkl', 'openblas')}
        with mock.patch.object(gen_site, 'current_arms', return_value=(arms, None)), \
             mock.patch.object(gen_site, 'load', return_value={'library': {'commit': gen_site.MEASURED}}), \
             mock.patch.object(gen_site.gen_comparison, 'main') as comparison:
            out = gen_site.write_comparison()
        self.assertEqual(out.name, gen_site.MEASURED[:12] + '-comparison.html')
        args = comparison.call_args.args[0]
        self.assertEqual(args[args.index('--out') + 1], str(out))

    def test_site_shows_the_recordings_as_current_with_the_release_caveat(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gen_site.main([]), 0)
        index = (ROOT / 'result/index.html').read_text()
        # The measured profile lists its recordings; zen2 profiles were never recorded and say so.
        zen5 = index.split('<h2 id="zen5-cpu">', 1)[1].split('<h2', 1)[0]
        self.assertNotIn('Nothing recorded at the current pin', zen5)
        self.assertIn(f'<code>{gen_site.MEASURED[:12]}</code>', zen5)
        self.assertIn(gen_site.CAVEAT, index)
        self.assertIn(gen_site.LIMITATION, index)
        for suite, vendor in RECORDED:
            link = f'zen5-cpu/{suite}/{gen_site.MEASURED[:12]}-0539947ffb75-{vendor}-timing-v2.html'
            self.assertIn(f'href="{link}"', index)
            self.assertTrue((ROOT / 'result' / link).is_file(), link)
        self.assertIn(f'href="zen5-cpu/faer-kernels/{gen_site.MEASURED[:12]}-comparison.html"', index)
        images = re.findall(r'<img src="([^"]+)"', index)
        self.assertTrue(images)
        for image in images:
            self.assertTrue((ROOT / 'result' / image).is_file(), image)

    def test_both_report_declarations_keep_the_exact_site_caveat_and_limitation(self):
        for suite in ('faer-kernels', 'faer-kernels-large'):
            with self.subTest(suite=suite):
                data = yaml.safe_load((ROOT / 'benchmarks/suites' / (suite + '.yaml')).read_text())
                self.assertIn(gen_site.CAVEAT, data['engines_doc']['faer-fork']['caveat'])
                self.assertTrue(any(gen_site.LIMITATION in s
                                    for s in data['workloads']['known_blind_spots']))


if __name__ == '__main__':
    unittest.main()
