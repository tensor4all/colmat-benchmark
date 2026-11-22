"""#113: the recordings keep their pre-rename names, and the campaign still reads them as one campaign.

The committed runs under `faer-kernels*` were recorded with engines `faer-fork` / `faer-upstream`
against the last pre-rename suite files. After the rename they must still be validated against the
renamed declarations (not silently skipped as "drifted"), and a candidate recorded under the new
names must not be refused against an old run *because of the rename* (its harness change is a
separate, legitimate refusal).
"""
import contextlib
import copy
import hashlib
import io
import pathlib
import tempfile
import unittest

import yaml

from campaign_toolkit import candidate, descriptor, gen_index, validate_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The current MKL arm of the daily suite, recorded under the pre-rename file.
BASELINE = ROOT / 'data/results/zen5-cpu/faer-kernels/20261120T064812Z-pub-93da6217-20261120-mkl'


class RenameAliasesTest(unittest.TestCase):
    def setUp(self):
        descriptor.set_root(ROOT)

    def test_every_current_page_keeps_its_declaration_checks(self):
        # Every run behind a current page whose recorded declaration hash is the last pre-rename
        # file (aliases.suite_sha256) is checked against the renamed declaration, not skipped.
        found = gen_index.collect()
        current, _ = gen_index.split_pages(found['pages'], found['declared'], found['resolved'])
        aliased = set(descriptor.current().aliases['suite_sha256'])
        runs = [pathlib.Path(c['page']['run_dir']) for c in current if c['page']]
        runs = [r for r in runs if yaml.safe_load((r / 'run.yaml').read_text())['suite_sha256'] in aliased]
        self.assertGreaterEqual(len(runs), 7)
        self.assertIn(BASELINE, runs)
        for run in runs:
            with self.subTest(run=run.name), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(validate_run.validate(str(run / 'run.yaml')), [])
            self.assertNotIn('declaration-derived checks skipped', err.getvalue())

    def test_the_suite_rename_alone_does_not_make_two_runs_incomparable(self):
        # Only the declaration identity is tested here. A real candidate recorded after #113 is
        # still refused against a pre-#113 baseline on `harness commit` (fbench changed), and
        # docs/rules.md says to re-record the baseline.
        manifest = yaml.safe_load((BASELINE / 'run.yaml').read_text())
        suite = ROOT / 'benchmarks/suites/colmat-kernels.yaml'
        renamed = copy.deepcopy(manifest)
        renamed.update(suite_id='colmat-kernels', suite_file='benchmarks/suites/colmat-kernels.yaml',
                       suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest())
        renamed['run_spec']['engines'] = ['colmat', 'faer', 'lapack-mkl']
        with tempfile.TemporaryDirectory() as tmp:
            run = pathlib.Path(tmp)
            (run / 'run.yaml').write_text(yaml.safe_dump(renamed, sort_keys=False))
            self.assertEqual(candidate.identity_differences(BASELINE, run), [])


if __name__ == '__main__':
    unittest.main()
