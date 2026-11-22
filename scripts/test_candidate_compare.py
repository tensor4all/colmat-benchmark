"""Real CLI/validator/pairing checks on synthetic v2 data and a temporary Git repo.

The committed manifest is a schema template only. All rows, suite, identities and
measured trees are replaced in a temporary campaign; no real timings/builds run.
"""
import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'data/results/zen5-cpu/faer-kernels/20261018T070736Z-upstream-lu-baseline/run.yaml'
FIELDS = ['regime', 'family', 'dtype', 'order', 'm', 'n', 'k', 'batch', 'row', 'threads', 'note', 'status', 'samples_ms', 'total_ms', 'per_item_us', 'rayon_threads', 'vendor_threads']


class CandidateCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for directory in ('benchmarks', 'pins'):
            shutil.copytree(ROOT / directory, self.root / directory)
        (self.root / 'scripts').mkdir()
        self.script = self.root / 'scripts/candidate_compare.py'
        shutil.copyfile(ROOT / 'scripts/candidate_compare.py', self.script)
        self.checkout = self.root / 'extern/fixture'
        self.checkout.mkdir(parents=True)
        for file in ('faer/src/lib.rs', 'faer/Cargo.toml', 'faer-traits/src/lib.rs',
                     'faer-traits/Cargo.toml', 'Cargo.lock'):
            path = self.checkout / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('synthetic fixture, never compiled\n')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture')
        commit = self.git('rev-parse', 'HEAD').strip()
        suite_file = self.root / 'benchmarks/suites/colmat-kernels.yaml'
        suite = yaml.safe_load(suite_file.read_text())
        suite['family_groups']['factor'] = ['lu']
        suite['runs'] = [{'threads': [1], 'cpu_sets': ['0'], 'dtypes': ['f64', 'c64'],
                          'order': ['c'], 'engines': ['colmat', 'faer'],
                          'populations': [
                              {'regime': 'gemm-small', 'groups': ['gemm'],
                               'n': ['32x32x32'], 'batch': [1]},
                              {'regime': 'linalg-small', 'groups': ['factor'],
                               'n': ['32'], 'batch': [1]}]}]
        suite_file.write_text(yaml.safe_dump(suite))
        self.manifest = yaml.safe_load(TEMPLATE.read_text())
        # The template was recorded before #113; the synthetic runs use the current names.
        self.manifest.update(suite_id='colmat-kernels', suite_file='benchmarks/suites/colmat-kernels.yaml')
        self.manifest['suite_sha256'] = hashlib.sha256(suite_file.read_bytes()).hexdigest()
        for identity in ('library', 'upstream', 'harness', 'toolkit'):
            self.manifest[identity].update(commit=commit, dirty=False)
            if identity in ('library', 'upstream'):
                self.manifest[identity]['measured_path'] = str(self.checkout)
        self.manifest.pop('vendor', None)
        self.manifest['binary'] = {'path': '/synthetic/not-executed', 'sha256': '0' * 64}
        self.manifest['threads'].update(counts=[1], cpu_sets={'1': '0'})
        self.manifest['run_spec'] = {
            'aa': 2, 'dtypes': ['f64', 'c64'], 'order': ['c'],
            'engines': ['colmat', 'faer'], 'covers_declared_suite': True,
            'populations': [{'regime': 'gemm-small', 'families': ['gemm'],
                             'n': ['32x32x32'], 'batch': [1]},
                            {'regime': 'linalg-small', 'families': ['lu'],
                             'n': ['32'], 'batch': [1]}]}
        self.manifest['result_files'] = [f'run{repeat}-{regime}-1t.csv' for repeat in (0, 1)
                                         for regime in ('gemm-small', 'linalg-small')]
        self.manifest['verification_files'] = ['verify-gemm-small-1t.csv', 'verify-linalg-small-1t.csv']
        self.baseline = self.make_run('baseline')
        self.candidate = self.make_run('candidate')

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.checkout), *args], check=True,
                              capture_output=True, text=True).stdout

    def make_run(self, name, dtypes=('f64', 'c64'), skip_lu=False, fork_ms=10):
        directory = self.root / name
        directory.mkdir()
        manifest = copy.deepcopy(self.manifest)
        manifest['run_spec']['dtypes'] = list(dtypes)
        manifest['run_spec']['covers_declared_suite'] = len(dtypes) == 2
        (directory / 'run.yaml').write_text(yaml.safe_dump(manifest))
        for repeat in (None, 0, 1):
            engines = manifest['run_spec']['engines']
            if repeat == 1:
                engines = engines[1:] + engines[:1]
            rows = []
            for engine in engines:
                for dtype in dtypes:
                    for family, regime, k in (('gemm', 'gemm-small', '32'), ('lu', 'linalg-small', '')):
                        skipped = skip_lu and engine == 'faer' and family == 'lu'
                        timed = repeat is not None and not skipped
                        value = fork_ms if engine == 'colmat' else 20
                        rows.append(dict(zip(FIELDS, [regime, family, dtype, 'c', '32', '32', k,
                            '1', engine, '1', 'unsupported' if skipped else '',
                            'skipped' if skipped else 'ok', json.dumps([value] * 5) if timed else '',
                            str(value) if timed else '', str(value * 1000) if timed else '', '1', '1'])))
            for regime in ('gemm-small', 'linalg-small'):
                prefix = 'verify' if repeat is None else f'run{repeat}'
                with (directory / f'{prefix}-{regime}-1t.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, FIELDS, lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(row for row in rows if row['regime'] == regime)
        return directory

    def cli(self, baseline=None, candidate=None, *extra):
        proc = subprocess.run([sys.executable, str(self.script), '--baseline', str(baseline or self.baseline),
                               '--candidate', str(candidate or self.candidate), '--checkout', str(self.checkout),
                               *extra], capture_output=True, text=True, check=False,
                              env={**os.environ, 'PYTHONHASHSEED': '0'}, timeout=30)
        self.assertNotIn('Traceback', proc.stderr, proc.stderr)
        return proc

    def test_valid_full_subset_and_descriptive_slower_fork(self):
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('4 cells compared', result.stdout)
        result = self.cli(None, None, '--cells', 'f64')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('2 cells compared', result.stdout)
        slower = self.make_run('slower', fork_ms=30)
        result = self.cli(None, slower)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('slower 4', result.stdout)

    def test_individually_valid_smaller_run_needs_explicit_subset(self):
        subset = self.make_run('subset', dtypes=('f64',))
        result = self.cli(None, subset)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('same selected cells', result.stdout)
        result = self.cli(None, subset, '--cells', 'f64')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('2 cells compared', result.stdout)

    def test_skipped_control_selector_and_genuine_missing_tokens(self):
        baseline = self.make_run('skip-base', skip_lu=True)
        candidate = self.make_run('skip-cand', skip_lu=True)
        result = self.cli(baseline, candidate, '--cells', 'lu,gemm')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('control over 2 cells', result.stdout)
        self.assertIn('4 cells compared', result.stdout)
        for selection in ('__missing__', 'gemm,__missing__', '', 'gemm,'):
            with self.subTest(selection=selection):
                self.assertEqual(self.cli(baseline, candidate, '--cells', selection).returncode, 1)

    def test_invalid_whole_run_is_not_excused_by_subset(self):
        (self.candidate / 'run1-linalg-small-1t.csv').unlink()
        for baseline, candidate in ((self.baseline, self.candidate), (self.candidate, self.baseline)):
            result = self.cli(baseline, candidate, '--cells', 'f64')
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('not valid evidence', result.stdout)

    def test_csv_parser_errors_refuse(self):
        (self.candidate / 'run0-gemm-small-1t.csv').write_text('x' * (csv.field_size_limit() + 1))
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('refused:', result.stdout)

    def test_yaml_manifest_and_output_io_errors_refuse(self):
        result = self.cli(None, None, '--json', str(self.root / 'absent/report.json'))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        path = self.candidate / 'run.yaml'
        for malformed in ('[invalid', ''):
            path.write_text(malformed)
            result = self.cli()
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('refused:', result.stdout)
        path.unlink()
        result = self.cli()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
