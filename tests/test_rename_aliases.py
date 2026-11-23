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

from campaign_toolkit import candidate, descriptor, validate_run

ROOT = pathlib.Path(__file__).resolve().parents[1]
# The 2026-11-20 MKL arm of the daily suite, recorded under the last pre-rename file.
BASELINE = ROOT / 'data/results/zen5-cpu/faer-kernels/20261120T064812Z-pub-93da6217-20261120-mkl'


class RenameAliasesTest(unittest.TestCase):
    def setUp(self):
        descriptor.set_root(ROOT)

    def test_runs_under_the_pre_rename_declaration_keep_their_declaration_checks(self):
        # Every committed run whose recorded declaration hash is a pre-rename file named in
        # aliases.suite_sha256 is checked against the renamed declaration, not skipped as drifted.
        # Selected by hash, not by "current page": a later re-recording moves the current page to a
        # new run without making this any less true of these runs.
        aliases = descriptor.current().aliases['suite_sha256']
        aliased = set(aliases)
        # An alias names the renamed declaration as it was when the rename landed. A suite edited
        # since (#111 narrowed the OpenBLAS arm) no longer hashes to its alias target: its old runs
        # are declaration-changed, so their declaration checks are skipped, and said to be.
        current = {hashlib.sha256(f.read_bytes()).hexdigest()
                   for f in (ROOT / 'benchmarks/suites').glob('*.yaml')}
        runs = [m.parent for m in sorted((ROOT / 'data/results').glob('*/*/*/run.yaml'))
                if yaml.safe_load(m.read_text()).get('suite_sha256') in aliased]
        self.assertIn(BASELINE, runs)       # a committed recording; recordings are never removed
        for run in runs:
            with self.subTest(run=run.name), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(validate_run.validate(str(run / 'run.yaml')), [])
            recorded = yaml.safe_load((run / 'run.yaml').read_text())['suite_sha256']
            skipped = 'declaration-derived checks skipped' in err.getvalue()
            self.assertEqual(skipped, aliases[recorded] not in current)

    def test_the_suite_rename_alone_does_not_make_two_runs_incomparable(self):
        # Only the declaration identity is tested here. A real candidate recorded after #113 is
        # still refused against a pre-#113 baseline on `harness commit` (fbench changed), and
        # docs/rules.md says to re-record the baseline.
        manifest = yaml.safe_load((BASELINE / 'run.yaml').read_text())
        renamed = copy.deepcopy(manifest)
        renamed.update(suite_id='colmat-kernels', suite_file='benchmarks/suites/colmat-kernels.yaml',
                       # the declaration the rename landed with, i.e. the baseline's alias target; the
                       # current file also carries #111's narrowing, a real declaration change
                       suite_sha256=descriptor.current().aliases['suite_sha256'][manifest['suite_sha256']])
        renamed['run_spec']['engines'] = ['colmat', 'faer', 'lapack-mkl']
        with tempfile.TemporaryDirectory() as tmp:
            run = pathlib.Path(tmp)
            (run / 'run.yaml').write_text(yaml.safe_dump(renamed, sort_keys=False))
            self.assertEqual(candidate.identity_differences(BASELINE, run), [])


if __name__ == '__main__':
    unittest.main()
