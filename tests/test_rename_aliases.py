"""#113: the recordings keep their pre-rename names, and the campaign still reads them as one campaign.

The committed runs under `faer-kernels*` were recorded with engines `faer-fork` / `faer-upstream`
against the last pre-rename suite files. After the rename they must still be validated against the
renamed declarations (not silently skipped as "drifted"), and a candidate recorded under the new
names must still compare with a kept baseline recorded under the old ones.
"""
import contextlib
import copy
import hashlib
import io
import pathlib
import tempfile
import unittest

import yaml

from campaign_toolkit import candidate, descriptor, validate_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The current MKL arm of the daily suite, recorded under the pre-rename file.
BASELINE = ROOT / 'data/results/zen5-cpu/faer-kernels/20261120T064812Z-pub-93da6217-20261120-mkl'


class RenameAliasesTest(unittest.TestCase):
    def setUp(self):
        descriptor.set_root(ROOT)

    def test_the_current_pages_keep_their_declaration_checks(self):
        current = [BASELINE,
                   ROOT / 'data/results/keyaki-64/faer-kernels-keyaki/20261122T124812Z-keyaki-mkl']
        for run in current:
            with self.subTest(run=run.name), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(validate_run.validate(str(run / 'run.yaml')), [])
            self.assertNotIn('declaration-derived checks skipped', err.getvalue())

    def test_a_candidate_under_the_new_names_compares_with_a_pre_rename_baseline(self):
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
