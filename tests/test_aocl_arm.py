"""The AOCL vendor arm (#95): declared once, consistently, for both suites; no build, no measurement.

The arm is a third binary beside MKL and OpenBLAS. What can drift is the declaration: the descriptor
must map the arm to its feature, its engine to its provider and the provider to a vendor arm; both
suites must declare the engine and document it; the link preflight must check the libraries the
build links; the recorder must drop AOCL's own thread variables; the validator and the index must
treat an `aocl` run as its own cell (the resolver is tests/test_resolve_aocl_root.py).
"""
import pathlib
import sys
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from campaign_toolkit import descriptor  # noqa: E402
from campaign_toolkit import gen_index, validate_run  # noqa: E402
from campaign_toolkit.campaign import engines_for_vendor, vendor_engines  # noqa: E402

SUITES = ('colmat-kernels', 'colmat-kernels-large')


def read(path):
    return yaml.safe_load((ROOT / path).read_text())


class DescriptorTest(unittest.TestCase):
    def setUp(self):
        descriptor.set_root(ROOT)
        self.campaign = read('benchmarks/campaign.yaml')

    def test_the_arm_is_a_vendor_with_its_own_feature_and_engine(self):
        self.assertEqual(self.campaign['measured']['features']['aocl'], 'fbench/link-aocl')
        self.assertEqual(self.campaign['engines']['lapack-aocl'], 'aocl')
        self.assertEqual(self.campaign['vendor_providers'], ['openblas', 'mkl', 'aocl'])

    def test_the_recorder_drops_aocls_own_thread_variables(self):
        # BLIS_NUM_THREADS and FLA_NUM_THREADS each override the OpenMP budget for one library.
        for variable in ('BLIS_NUM_THREADS', 'FLA_NUM_THREADS', 'OMP_NUM_THREADS'):
            self.assertIn(variable, self.campaign['vendor_env'])

    def test_both_suites_declare_and_document_the_engine(self):
        for suite_id in SUITES:
            suite = read(f'benchmarks/suites/{suite_id}.yaml')
            with self.subTest(suite=suite_id):
                for run in suite['runs']:
                    self.assertIn('lapack-aocl', run['engines'])
                    # MKL and OpenBLAS stay declared beside it.
                    self.assertIn('lapack-mkl', run['engines'])
                    self.assertIn('lapack-openblas', run['engines'])
                self.assertEqual(suite['engines_doc']['lapack-aocl']['provider'], 'aocl')
                # An AOCL cell is the measured rows plus AOCL's own row, never another vendor's.
                self.assertEqual(engines_for_vendor(suite, 'aocl'),
                                 ['colmat', 'faer', 'lapack-aocl'])

    def test_the_link_preflight_checks_what_the_build_links(self):
        arm = read('benchmarks/link-preflight.yaml')['arms']['aocl']
        self.assertEqual(arm['when'], {'vendor': 'aocl'})
        self.assertEqual(arm['libs'], ['flame', 'blis-mt', 'aoclutils'])
        self.assertTrue(arm['rpath'])
        build = (ROOT / 'fbench/build.rs').read_text()
        for lib in arm['libs']:
            self.assertIn(f'"{lib}"', build)


class ValidatorAndIndexTest(unittest.TestCase):
    def setUp(self):
        descriptor.set_root(ROOT)

    def manifest(self, engines, vendor):
        return {'run_spec': {'engines': engines}, 'vendor': vendor}

    def test_an_aocl_run_names_its_vendor(self):
        errors = []
        validate_run.check_vendor(self.manifest(['colmat', 'lapack-aocl'], {'name': 'aocl'}), errors)
        self.assertEqual(errors, [])
        # A run recorded before #113 names the measured row `faer-fork`; it is the same arm.
        validate_run.check_vendor(self.manifest(['faer-fork', 'lapack-aocl'], {'name': 'aocl'}), errors)
        self.assertEqual(errors, [])

    def test_an_aocl_run_that_recorded_another_library_is_refused(self):
        errors = []
        validate_run.check_vendor(self.manifest(['faer-fork', 'lapack-aocl'], {'name': 'mkl'}), errors)
        self.assertTrue(any("name 'aocl'" in e for e in errors), errors)

    def test_an_aocl_run_without_a_vendor_identity_is_refused(self):
        errors = []
        validate_run.check_vendor(self.manifest(['faer-fork', 'lapack-aocl'], {}), errors)
        self.assertTrue(any("vendor arm 'aocl'" in e for e in errors), errors)

    def test_aocl_and_mkl_in_one_run_are_refused(self):
        self.assertEqual(vendor_engines(['lapack-mkl', 'lapack-aocl']), ['mkl', 'aocl'])
        errors = []
        validate_run.check_vendor(self.manifest(['lapack-mkl', 'lapack-aocl'], {'name': 'aocl'}), errors)
        self.assertTrue(any('one binary links one vendor' in e for e in errors), errors)

    def test_the_index_declares_an_aocl_cell_per_suite_and_profile(self):
        declared = gen_index.collect()['declared']
        cells = {(c['suite'], c['profile']) for c in declared if c['vendor'] == 'aocl'}
        for suite_id in SUITES:
            self.assertIn((suite_id, 'evo2-zen5-16c'), cells)
        # The other arms are still declared next to it.
        for vendor in ('mkl', 'openblas'):
            self.assertIn(('colmat-kernels', 'evo2-zen5-16c'),
                          {(c['suite'], c['profile']) for c in declared if c['vendor'] == vendor})
        # zen5-cpu is retired (docs/zen5-cpu-retired.md): no cell is declared, so none reads `missing`.
        self.assertNotIn('zen5-cpu', {c['profile'] for c in declared})


if __name__ == '__main__':
    unittest.main()
