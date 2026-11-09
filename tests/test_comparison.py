"""The joined two-arm page: what it divides by, and which runs it refuses to join.

No build and no measurement is run: the runs are small manifest-plus-CSV fixtures in the shape
`record_run.py` writes, and the assertions are about the page that joins them.
"""
import csv
import contextlib
import io
import pathlib
import shutil
import sys
import tempfile
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import gen_comparison  # noqa: E402

FIELDS = ['regime', 'family', 'dtype', 'order', 'm', 'n', 'k', 'batch', 'row', 'threads',
          'total_ms', 'per_item_us', 'status', 'note', 'samples_ms', 'rayon_threads',
          'vendor_threads']

POWER = {'cpufreq_boost': '0', 'amd_pstate_status': 'active', 'governor': 'performance',
         'energy_performance_preference': 'performance', 'scaling_max_freq_khz': '2000000'}
HOST = {'hostname': 'EVO-X2', 'cpu': 'AMD RYZEN AI MAX+ 395', 'os': 'Linux 6.17',
        'arch': 'x86_64', 'logical_cpus': 32, 'power': dict(POWER)}
COMMIT = '2b87010cb82efbe0a90eceeb51931a833e1de8e6'
UPSTREAM = '0539947ffb757a739d7e703a7d2fa0c792a909c1'


def manifest(vendor, host=None, result_files=(), **overrides):
    block = {
        'schema_version': 1,
        'target_profile': 'zen5-cpu',
        'suite_id': 'faer-kernels',
        'timestamp': '2026-11-07T09:00:00Z',
        'result_files': list(result_files),
        'library': {'url': 'https://github.com/tensor4all/faer', 'commit': COMMIT,
                    'version': '0.31.0', 'dirty': False},
        'upstream': {'commit': UPSTREAM, 'ref': 'faer-v0.24.4', 'version': '0.24.4'},
        'vendor': {'name': vendor, 'linkage': 'shared',
                   'version': '2026.2' if vendor == 'mkl' else '0.3.32'},
        'harness': {'commit': '7bb8105cae25186ef3108745cc7a626532d8981d', 'dirty': False},
        'toolkit': {'commit': 'c2e98df904d3987888c0d66353c18c095fbcbd35', 'dirty': False},
        'host': dict(host or HOST),
        'threads': {'cpu_sets': {'1': '8', '8': '8-15'},
                    'observer': {'affinity': 'cpu_set 8-15'}},
        'timing_policy': {'version': 2, 'repetitions': 5, 'statistic': 'median',
                          'minimum_untimed_priming_ms': 250},
        'guards': {'idle_window_seconds': 3, 'max_busy_percent': 5.0, 'attempts': 3,
                   'sibling_aware': True},
    }
    block.update(overrides)
    return block


def write_run(directory, vendor, rows=None, host=None, sets=None, **overrides):
    """One run directory: `run.yaml` plus one `runN.csv` per process set, all named by the manifest.

    `rows` is the single process set; `sets` several, in the order the recorder wrote them.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sets = sets or [rows]
    (directory / 'run.yaml').write_text(
        yaml.safe_dump(manifest(vendor, host, [f'run{index}.csv' for index in range(len(sets))],
                                **overrides), sort_keys=False))
    for index, set_rows in enumerate(sets):
        with open(directory / f'run{index}.csv', 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in set_rows:
                writer.writerow(dict({f: '' for f in FIELDS}, **row))
    return directory


def cell(family, row, total_ms, order='c', threads='8', note='', status='ok', shape='1024',
         dtype='f64'):
    return {'regime': 'linalg-mid', 'family': family, 'dtype': dtype, 'order': order,
            'm': shape, 'n': shape, 'k': '', 'batch': '1', 'row': row,
            'threads': threads, 'total_ms': '' if total_ms is None else f'{total_ms:.6f}',
            'status': status, 'note': note}


def mkl_set(fork, upstream, mkl):
    """One process set of the MKL arm: two families in both orders, at the same times."""
    return [cell(family, row, value, order=order)
            for family in ('lu', 'cholesky')
            for order in ('c', 'f')
            for row, value in (('faer-fork', fork), ('faer-upstream', upstream),
                               ('lapack-mkl', mkl))]


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.out = self.dir / 'out' / 'comparison.html'

    def tearDown(self):
        self.tmp.cleanup()

    def generate(self, mkl_rows, ob_rows, extra=(), **ob_overrides):
        mkl = write_run(self.dir / 'mkl', 'mkl', mkl_rows)
        ob = write_run(self.dir / 'openblas', 'openblas', ob_rows, **ob_overrides)
        self.log = io.StringIO()
        with contextlib.redirect_stdout(self.log):
            self.assertEqual(gen_comparison.main(['--mkl', str(mkl), '--openblas', str(ob),
                                                  '--out', str(self.out), *extra]), 0)
        self.log = self.log.getvalue()
        return self.out.read_text()

    def test_the_ratio_columns_divide_by_the_fork_shown_beside_them(self):
        # The fork is measured in both arms, and the two numbers differ: here the MKL arm's last
        # process set has fork 40 ms while the OpenBLAS arm measured 10 ms for the same cell. A page
        # showing one fork and dividing `OpenBLAS/fork` by the other arm's value prints a ratio whose
        # denominator is nowhere in the row (50/10 = 5.000 that cannot be obtained from 40).
        mkl = write_run(self.dir / 'mkl', 'mkl', sets=[
            mkl_set(10.0, 12.0, 100.0), mkl_set(20.0, 12.0, 100.0), mkl_set(40.0, 12.0, 100.0)])
        ob = write_run(self.dir / 'openblas', 'openblas',
                       [cell('lu', 'faer-fork', 10.0), cell('lu', 'lapack-openblas', 50.0)])
        self.assertEqual(gen_comparison.main(['--mkl', str(mkl), '--openblas', str(ob),
                                             '--out', str(self.out)]), 0)
        page = self.out.read_text()
        fmt, cls = gen_comparison.fmt, gen_comparison.cls
        self.assertIn('<th>fork MKL (ms)</th>', page)
        self.assertIn('<th>fork OpenBLAS (ms)</th>', page)
        # Every number in this sequence is in the row, so each ratio can be checked by hand:
        # upstream/fork = 12/40, OpenBLAS/fork = 50/10, MKL/fork = 100/40.
        self.assertIn(f'<td>{fmt(40.0)}</td><td>{fmt(12.0)}</td><td{cls(0.3)}>{fmt(0.3)}</td>'
                      f'<td>{fmt(10.0)}</td><td>{fmt(50.0)}</td><td{cls(5.0)}>{fmt(5.0)}</td>'
                      f'<td>{fmt(100.0)}</td><td{cls(2.5)}>{fmt(2.5)}</td>', page)

    def test_a_ratio_is_a_dash_when_its_fork_is_missing_from_that_arm(self):
        # The OpenBLAS arm measured the vendor row but not the fork, so there is nothing in the row
        # to divide by: the ratio is a dash rather than the other arm's fork standing in silently.
        page = self.generate([cell('lu', 'faer-fork', 10.0), cell('lu', 'lapack-mkl', 100.0)],
                             [cell('lu', 'lapack-openblas', 50.0)])
        fmt = gen_comparison.fmt
        self.assertIn(f'<td>{fmt(10.0)}</td>', page)
        self.assertIn(f'<td>-</td><td>{fmt(50.0)}</td><td>-</td>', page)
        self.assertNotIn(f'>{fmt(5.0)}<', page)

    def test_a_different_idle_guard_is_not_the_same_protocol(self):
        # The review's reproduction: identical fixtures except the AOCL-era arm's `guards`. A run
        # that accepted 100% busy for 0 s is not the peer of one that waited for 5% for 3 s, and
        # the page printed only the MKL arm's guard above both.
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                          guards={'idle_window_seconds': 0, 'max_busy_percent': 100.0,
                                  'attempts': 1, 'sibling_aware': True})
        message = str(raised.exception)
        for field in ('guards.idle_window_seconds', 'guards.max_busy_percent', 'guards.attempts'):
            self.assertIn(field, message)
        self.assertIn('one protocol', message)
        self.assertFalse(self.out.exists())

    def test_a_measured_thread_budget_without_a_cpu_set_is_refused(self):
        # The review's reproduction: both arms measured an 8-thread cell, but the second arm records
        # no CPU set for 8 threads -- only an unrelated 1-thread entry. Sharing that entry is not
        # evidence about the budget the cell was measured at.
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                          threads={'cpu_sets': {'1': '8'}, 'observer': {'affinity': 'cpu_set 8-15'}})
        self.assertIn('threads.cpu_sets[8 thread]', str(raised.exception))
        self.assertIn('records no CPU set', str(raised.exception))
        self.assertFalse(self.out.exists())

    def test_cells_measured_by_neither_arm_are_counted(self):
        # A cell both arms wrote `skipped` for is a cell the run was asked for and did not produce.
        # It has no row to print, but dropping it silently makes the page describe a smaller run
        # than the manifests do, and the dash rule ("that arm did not carry the cell") does not
        # cover a cell that no arm carried.
        reason = 'eig has no LAPACK route in this harness'
        page = self.generate([cell('lu', 'faer-fork', 3.877), cell('lu', 'lapack-mkl', 3.643),
                              cell('eig', 'faer-fork', None, status='skipped', note=reason)],
                             [cell('lu', 'faer-fork', 10.0),
                              cell('lu', 'lapack-openblas', 50.0),
                              cell('eig', 'faer-fork', None, status='skipped', note=reason)])
        self.assertIn('1 of 2 cells were asked for by at least one arm and measured by neither', page)
        self.assertIn('They are the families <code>eig</code>', page)
        self.assertIn('1 cells were asked for and measured by neither arm', self.log)

    def test_the_openblas_summary_counts_the_cells_that_have_an_openblas_row(self):
        # The summary counted cells by the upstream/fork ratio, so it claimed an OpenBLAS row for
        # every cell the MKL arm carried. On the published page that was 1356 instead of 972, and
        # the median it printed was the upstream one (1.956 instead of 1.230).
        page = self.generate([cell('lu', 'faer-fork', 3.877), cell('lu', 'faer-upstream', 4.0),
                              cell('lu', 'lapack-mkl', 3.643),
                              cell('cholesky', 'lapack-mkl', 2.104)],
                             [cell('lu', 'faer-fork', 10.0), cell('lu', 'lapack-openblas', 50.0),
                              cell('cholesky', 'faer-fork', 10.0),
                              cell('cholesky', 'lapack-openblas', 20.0)])
        self.assertIn('2 against <code>lapack-openblas</code>', page)
        self.assertIn('OpenBLAS 3.500', page)   # median of 50/10 and 20/10

    def test_an_arm_handed_to_the_wrong_column_is_refused(self):
        # The page prints `MKL` and `OpenBLAS` column headings, and the row it reads for an arm comes
        # from that arm's own manifest. An OpenBLAS run handed to `--mkl` would therefore put
        # OpenBLAS numbers under the MKL heading -- a mislabelled measurement, which is worse than a
        # refusal.
        mkl = write_run(self.dir / 'mkl', 'mkl', [cell('lu', 'faer-fork', 3.877),
                                                 cell('lu', 'lapack-mkl', 3.643)])
        wrong = write_run(self.dir / 'openblas', 'mkl', [cell('lu', 'faer-fork', 10.0),
                                                         cell('lu', 'lapack-mkl', 50.0)])
        with self.assertRaises(SystemExit) as raised:
            gen_comparison.main(['--mkl', str(mkl), '--openblas', str(wrong),
                                 '--out', str(self.out)])
        self.assertIn("records its vendor as 'mkl'", str(raised.exception))
        self.assertFalse(self.out.exists())

    def test_a_csv_the_manifest_does_not_name_is_ignored(self):
        # A leftover copy in the run directory must not become the value the page shows: the arm
        # never re-measured the cell, and the manifest names the files that were gated.
        mkl = write_run(self.dir / 'mkl', 'mkl', sets=[
            mkl_set(10.0, 12.0, 100.0), mkl_set(20.0, 12.0, 100.0), mkl_set(40.0, 12.0, 100.0)])
        ob = write_run(self.dir / 'openblas', 'openblas',
                       [cell('lu', 'faer-fork', 30.0), cell('lu', 'lapack-openblas', 50.0)])
        shutil.copy(mkl / 'run0.csv', mkl / 'run9-copy.csv')
        self.assertEqual(gen_comparison.main(['--mkl', str(mkl), '--openblas', str(ob),
                                             '--out', str(self.out)]), 0)
        page = self.out.read_text()
        self.assertIn(f'<td>{gen_comparison.fmt(40.0)}</td>', page)
        self.assertNotIn(f'<td>{gen_comparison.fmt(10.0)}</td>', page)

    def test_runs_of_two_protocols_are_refused(self):
        # One host, one revision, but the second arm was recorded under timing policy v1 with one
        # best-of call, on CPUs 0-7, in another year. Joining it would publish two different
        # measurements as one session and print only the MKL arm's protocol. Every differing field
        # has to be named.
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                          timing_policy={'version': 1, 'repetitions': 1, 'statistic': 'best',
                                         'minimum_untimed_priming_ms': 0},
                          threads={'cpu_sets': {'1': '0', '8': '0-7'},
                                   'observer': {'affinity': 'cpu_set 0-3'}},
                          timestamp='2025-01-01T09:00:00Z')
        message = str(raised.exception)
        for field in ('timing_policy.version', 'timing_policy.repetitions',
                      'timing_policy.statistic', 'timing_policy.minimum_untimed_priming_ms',
                      'threads.cpu_sets', 'threads.observer.affinity'):
            self.assertIn(field, message)
        self.assertIn('one protocol', message)
        self.assertFalse(self.out.exists())

    def test_a_different_timestamp_alone_is_not_a_protocol_difference(self):
        # The arms of a publication are recorded hours apart, so the printed timestamps differ by
        # design; what the join checks is what the numbers depend on, not the wall clock.
        page = self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                              cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                             [cell('cholesky', 'faer-fork', 1.474, order='f'),
                              cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                             timestamp='2025-01-01T09:00:00Z')
        self.assertIn('2025-01-01T09:00:00Z', page)

    def test_a_cell_measured_in_another_dtype_is_refused(self):
        # dtype, threads and shape are the cell identity: an arm measuring the same family in f32
        # never joins the f64 cell, and the dash that would appear is then read as "that arm had no
        # driver for it". That is a different measurement, not a missing one.
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f', dtype='f32'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f', dtype='f32')])
        self.assertIn('cell cholesky', str(raised.exception))
        self.assertIn('dtype', str(raised.exception))
        self.assertFalse(self.out.exists())

    def test_a_sibling_blind_guard_is_not_the_same_protocol(self):
        # The guard is stated by the manifest, and `sibling_aware` decides whether the SMT siblings
        # of the measured cores are checked too: a run that let a busy sibling contend is not the
        # peer of one that refused it, which is why the whole guard policy is part of the protocol.
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                          guards={'idle_window_seconds': 3, 'max_busy_percent': 5.0, 'attempts': 3,
                                  'sibling_aware': False})
        self.assertIn('guards.sibling_aware', str(raised.exception))
        self.assertFalse(self.out.exists())

    def test_a_boosted_host_is_not_the_same_protocol(self):
        # Two arms can share the host block's identity and still be measured under different power
        # states, which docs/protocol-v2-baseline.md calls incomparable.
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                          host=dict(HOST, power=dict(POWER, cpufreq_boost='1')))
        self.assertIn('host.power', str(raised.exception))
        self.assertFalse(self.out.exists())

    def test_a_thread_budget_only_one_arm_measured_is_not_a_difference(self):
        # An arm strided to 8T only is a smaller grid, not a different protocol, as long as the
        # budget both measured ran on the same CPUs.
        page = self.generate(
            [cell('cholesky', 'faer-fork', 1.467, order='f'),
             cell('cholesky', 'faer-fork', 1.467, order='f', threads='1'),
             cell('cholesky', 'lapack-mkl', 2.104, order='f'),
             cell('cholesky', 'lapack-mkl', 2.104, order='f', threads='1')],
            [cell('cholesky', 'faer-fork', 1.474, order='f'),
             cell('cholesky', 'lapack-openblas', 0.985, order='f')],
            threads={'cpu_sets': {'8': '8-15'}, 'observer': {'affinity': 'cpu_set 8-15'}})
        self.assertIn('<td class="k">cholesky</td>', page)

    def test_a_thread_budget_only_the_second_arm_measured_is_not_a_difference(self):
        # The same rule in the other direction: the MKL arm was asked for 8T only while the OpenBLAS
        # arm also measured 1T. A partial run is a smaller grid, not a different protocol, and the
        # cell the MKL arm was not asked for is a dash on its side.
        page = self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                              cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                             [cell('cholesky', 'faer-fork', 1.474, order='f'),
                              cell('cholesky', 'faer-fork', 1.474, order='f', threads='1'),
                              cell('cholesky', 'lapack-openblas', 0.985, order='f'),
                              cell('cholesky', 'lapack-openblas', 0.985, order='f', threads='1')],
                             threads={'cpu_sets': {'1': '8', '8': '8-15'},
                                      'observer': {'affinity': 'cpu_set 8-15'}})
        self.assertIn('<td class="k">cholesky</td>', page)

    def test_a_cpu_set_neither_arm_measured_on_is_not_a_protocol_difference(self):
        # Both arms measured 8T only and both measured it on 8-15; their leftover 1T entries differ,
        # and say nothing about a number the page prints.
        page = self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                              cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                             [cell('cholesky', 'faer-fork', 1.474, order='f'),
                              cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                             threads={'cpu_sets': {'1': '0', '8': '8-15'},
                                      'observer': {'affinity': 'cpu_set 8-15'}})
        self.assertIn('<td class="k">cholesky</td>', page)

    def test_one_thread_budget_on_other_cpus_is_refused(self):
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                          threads={'cpu_sets': {'1': '8', '8': '0-7'},
                                   'observer': {'affinity': 'cpu_set 8-15'}})
        self.assertIn('threads.cpu_sets[8 thread]', str(raised.exception))
        self.assertNotIn('threads.cpu_sets[1 thread]', str(raised.exception))
        self.assertFalse(self.out.exists())

    def test_runs_from_two_hosts_are_refused(self):
        with self.assertRaises(SystemExit) as raised:
            self.generate([cell('cholesky', 'faer-fork', 1.467, order='f'),
                           cell('cholesky', 'lapack-mkl', 2.104, order='f')],
                          [cell('cholesky', 'faer-fork', 1.474, order='f'),
                           cell('cholesky', 'lapack-openblas', 0.985, order='f')],
                          host=dict(HOST, hostname='some-other-host'))
        self.assertIn('one host', str(raised.exception))
        self.assertIn('some-other-host', str(raised.exception))
        self.assertFalse(self.out.exists())

    def test_the_table_body_is_pinned(self):
        # The body for the two runs below, byte for byte. It is the guard that a ratio cannot
        # silently become a dash: a test that only asked for "no reference column" passed while
        # every MKL/fork value had been blanked. Regenerate it by rendering the same two run
        # directories, and say in the same commit why the body moved.
        mkl = write_run(self.dir / 'mkl', 'mkl', sets=[
            mkl_set(10.0, 12.0, 100.0), mkl_set(20.0, 12.0, 100.0), mkl_set(40.0, 12.0, 100.0)])
        ob = write_run(self.dir / 'openblas', 'openblas',
                       [cell('lu', 'faer-fork', 10.0), cell('lu', 'lapack-openblas', 50.0),
                        cell('cholesky', 'faer-fork', 10.0, order='f'),
                        cell('cholesky', 'lapack-openblas', 50.0, order='f')])
        self.assertEqual(gen_comparison.main(['--mkl', str(mkl), '--openblas', str(ob),
                                             '--out', str(self.out)]), 0)
        page = self.out.read_text()
        body = page[page.index('<tbody>') + len('<tbody>'):page.index('</tbody>')]
        self.assertEqual(body, (ROOT / 'tests' / 'fixtures' / 'comparison-table.html').read_text())


if __name__ == '__main__':
    unittest.main()
