"""The write path refuses a host its profile does not describe; no build or measurement.

`zen2-cpu` declares 128 logical cores with SMT on; the measurement host as it is
measured today reports 64 logical with SMT off. The recorder used to accept that
pair and file the run under `zen2-cpu` (faer-benchmark#55), so the host check must
fail before the recorder writes anything, and pass on the machine the profile
actually describes.
"""
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import record_run  # noqa: E402

CPU = 'sys/devices/system/cpu'


def sysfs(root, online, smt=None, siblings=None):
    """A sysfs tree holding `online` CPUs, an SMT switch and, optionally, a sibling map."""
    cpu = pathlib.Path(root) / CPU
    (cpu / 'smt').mkdir(parents=True, exist_ok=True)
    (cpu / 'online').write_text(online + '\n')
    if smt is not None:
        (cpu / 'smt/active').write_text(smt + '\n')
    for n, group in (siblings or {}).items():
        path = cpu / f'cpu{n}/topology'
        path.mkdir(parents=True, exist_ok=True)
        (path / 'thread_siblings_list').write_text(group + '\n')
    return str(root)


def profile(name):
    for entry in yaml.safe_load((ROOT / 'benchmarks/profiles.yaml').read_text())['profiles']:
        if not isinstance(entry, str) and entry['name'] == name:
            return entry
    if name == 'zen2-cpu':
        from campaign_toolkit import descriptor
        return descriptor.shared_profile(name)
    raise AssertionError(f'no profile {name}')


class RecordRunHostCheckTest(unittest.TestCase):
    def test_the_host_the_profile_describes_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = sysfs(tmp, '0-63', '0', {0: '0', 63: '63'})
            record_run.require_host(profile('zen2-64-smt-off'), host)  # no SystemExit

    def test_the_development_host_is_still_accepted(self):
        """`zen5-cpu` declares SMT on with a sibling offset of 12: its own host must pass."""
        with tempfile.TemporaryDirectory() as tmp:
            host = sysfs(tmp, '0-23', '1', {0: '0,12', 4: '4,16', 23: '11,23'})
            record_run.require_host(profile('zen5-cpu'), host)

    def test_a_host_with_another_logical_count_and_smt_state_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = sysfs(tmp, '0-63', '0')
            with self.assertRaises(SystemExit) as raised:
                record_run.require_host(profile('zen2-cpu'), host)
            self.assertIn('declared 128 logical / smt on, measured 64 logical / smt off',
                          str(raised.exception))
            self.assertIn('Nothing was written', str(raised.exception))

    def test_the_other_direction_is_refused_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = sysfs(tmp, '0-127', '1')
            with self.assertRaises(SystemExit) as raised:
                record_run.require_host(profile('zen2-64-smt-off'), host)
            self.assertIn('declared 64 logical / smt off, measured 128 logical / smt on',
                          str(raised.exception))

    def test_a_host_whose_sibling_map_differs_is_refused(self):
        """The same rule `idle_cpus.py check --profile` enforces with its exit 4."""
        with tempfile.TemporaryDirectory() as tmp:
            host = sysfs(tmp, '0-127', '1', {0: '0,1', 64: '64,65'})
            with self.assertRaises(SystemExit) as raised:
                record_run.require_host(profile('zen2-cpu'), host)
            self.assertIn('cpu0 has siblings 0-1 here, the profile 0,64',
                          str(raised.exception))

    def test_a_profile_without_an_smt_block_only_checks_the_count(self):
        bare = dict(profile('zen2-64-smt-off'))
        del bare['smt']
        with tempfile.TemporaryDirectory() as tmp:
            record_run.require_host(bare, sysfs(tmp, '0-63', '0'))
        with tempfile.TemporaryDirectory() as tmp:
            record_run.require_host(bare, sysfs(tmp, '0-63', '1'))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as raised:
                record_run.require_host(bare, sysfs(tmp, '0-31', '0'))
            self.assertIn('smt undeclared', str(raised.exception))

    def test_smt_is_read_from_the_sibling_map_where_the_kernel_exposes_no_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(record_run.measured(sysfs(tmp, '0-127', None, {0: '0,64', 64: '0,64'})),
                             (128, 'on', {0: [0, 64], 64: [0, 64]}))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(record_run.measured(sysfs(tmp, '0-63', 'notsupported', {0: '0'})),
                             (64, 'off', {0: [0]}))

    def test_a_host_with_no_topology_is_refused_rather_than_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as raised:
                record_run.require_host(profile('zen2-64-smt-off'), tmp)
            self.assertIn('cannot check this host', str(raised.exception))

    def test_two_profile_names_in_the_arguments_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = campaign_root(tmp)
            host = sysfs(pathlib.Path(tmp) / 'host', '0-63', '0')
            with self.assertRaises(SystemExit) as raised:
                record_run.main(['zen2-64-smt-off', 'faer-kernels', '--label', 'zen2-cpu'],
                                root=campaign, host_root=host)
            self.assertIn('name more than one hardware profile', str(raised.exception))

    def test_a_mismatch_writes_nothing_and_a_match_reaches_the_recorder(self):
        """The refusal comes before the recorder, so a mismatch leaves no run directory behind."""
        with tempfile.TemporaryDirectory() as tmp:
            campaign = campaign_root(tmp)
            mismatch = sysfs(pathlib.Path(tmp) / 'mismatch', '0-63', '0')
            match = sysfs(pathlib.Path(tmp) / 'match', '0-63', '0')

            with self.assertRaises(SystemExit) as raised:
                record_run.main(['zen2-cpu', 'faer-kernels'], root=campaign, host_root=mismatch)
            self.assertIn('measured 64 logical / smt off', str(raised.exception))
            self.assertFalse((campaign / 'data').exists())
            self.assertFalse((campaign / 'target').exists())

            os.environ.pop('FBENCH_PROFILE', None)
            with mock.patch.object(record_run, 'record_run_main', return_value=0) as recorder:
                code = record_run.main(['zen2-64-smt-off', 'faer-kernels'], root=campaign,
                                       host_root=match)
            self.assertEqual(code, 0)
            recorder.assert_called_once()
            self.assertEqual(recorder.call_args.args[0], ['zen2-64-smt-off', 'faer-kernels'])
            self.assertEqual(os.environ['FBENCH_PROFILE'], 'zen2-64-smt-off')



def ohtaka_fat_host(root):
    """The fat node as the profile declares it: 112 CPUs, SMT off, each CPU its own sibling."""
    return sysfs(root, '0-111', '0', {n: str(n) for n in range(112)})


def ohtaka_fat_cpu_sets():
    return {str(row['threads']): row['measured'] for row in profile('ohtaka-fat')['thread_rows']}


class OhtakaFatObserverTest(unittest.TestCase):
    """#109: the profile carries no `os_shared`, and the 112T row still gets its caveat."""

    def test_the_profile_declares_no_whole_profile_override(self):
        self.assertNotIn('os_shared', profile('ohtaka-fat'))
        self.assertEqual(profile('ohtaka-fat')['thread_rows'][-1]['observer'], '')

    def test_recording_it_leaves_the_observer_refusals_in_place(self):
        from campaign_toolkit import record_run as toolkit
        before = (toolkit.require_observer_off_measured_cpus, toolkit.require_quiet_observing_side)
        with tempfile.TemporaryDirectory() as tmp:
            campaign = campaign_root(tmp, suites=True)
            host = ohtaka_fat_host(pathlib.Path(tmp) / 'host')
            with mock.patch.object(record_run, 'record_run_main', return_value=0):
                record_run.main(['ohtaka-fat', 'faer-kernels-ohtaka-fat-gemm'], root=campaign,
                                host_root=host)
        self.assertEqual((toolkit.require_observer_off_measured_cpus,
                          toolkit.require_quiet_observing_side), before)

    def test_the_declared_112t_row_is_admitted_with_its_caveat(self):
        """The toolkit's own path for `observer: ""`: a warning and a recorded caveat, no refusal."""
        import contextlib
        import io
        from campaign_toolkit import record_run as toolkit
        node = set(range(112))
        cpu_sets = ohtaka_fat_cpu_sets()
        # Four L3 domains of 28 CPUs, node k = CPUs that are k mod 4; the recorder sits on node 3.
        with mock.patch.object(toolkit, 'online_cpus', return_value=node), \
                mock.patch.object(toolkit, 'l3_domain', side_effect=lambda c: {x for x in node if x % 4 == c % 4}), \
                mock.patch.object(toolkit.os, 'sched_getaffinity', return_value={3, 7, 11}):
            unobserved = toolkit.unobserved_rows(profile('ohtaka-fat'), cpu_sets)
            self.assertEqual(unobserved, ['112'])
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                toolkit.require_observer_off_measured_cpus(cpu_sets, unobserved)  # no SystemExit
            self.assertIn('observer unavailable for the 112T row(s)', err.getvalue())
            self.assertEqual(toolkit.observer_affinity(cpu_sets)['caveat'],
                             'observer unavailable: no CPU lies outside the measured L3 domains')
            # Without the 112T row, nothing is unobserved and the observer rule applies again.
            clean = {t: c for t, c in cpu_sets.items() if t != '112'}
            self.assertEqual(toolkit.unobserved_rows(profile('ohtaka-fat'), clean), [])
            self.assertNotIn('caveat', toolkit.observer_affinity(clean))


class HarnessOrderTest(unittest.TestCase):
    """#109: `FBENCH_ORDER` comes from the suite's declared order, not from the operator's shell."""

    def run_main(self, suite, ambient=None):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = campaign_root(tmp, suites=True)
            host = ohtaka_fat_host(pathlib.Path(tmp) / 'host')
            name = 'ohtaka-fat' if 'ohtaka' in suite else 'zen2-64-smt-off'
            if name != 'ohtaka-fat':
                host = sysfs(pathlib.Path(tmp) / 'zen2', '0-63', '0')
            env = {} if ambient is None else {'FBENCH_ORDER': ambient}
            with mock.patch.dict(os.environ, env, clear=False), \
                    mock.patch.object(record_run, 'record_run_main', return_value=0):
                if ambient is None:
                    os.environ.pop('FBENCH_ORDER', None)
                record_run.main([name, suite], root=campaign, host_root=host)
                return os.environ.get('FBENCH_ORDER')

    def test_a_single_declared_order_is_exported(self):
        self.assertEqual(self.run_main('faer-kernels-ohtaka-fat-gemm'), 'f')
        self.assertEqual(self.run_main('faer-kernels-ohtaka-fat-gemm', ambient='f'), 'f')

    def test_both_orders_leave_the_harness_default(self):
        self.assertIsNone(self.run_main('colmat-kernels'))
        self.assertIsNone(self.run_main('colmat-kernels', ambient='both'))

    def test_an_ambient_value_the_suite_does_not_declare_is_refused(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_main('colmat-kernels', ambient='f')
        self.assertIn('FBENCH_ORDER=f is set in this shell', str(raised.exception))
        with self.assertRaises(SystemExit):
            self.run_main('faer-kernels-ohtaka-fat-gemm', ambient='c')

    def test_the_manifest_records_it(self):
        campaign = yaml.safe_load((ROOT / 'benchmarks/campaign.yaml').read_text())
        self.assertIn('FBENCH_ORDER', campaign['measured']['recorded_env'])


def campaign_root(tmp, suites=False):
    """A campaign root with only what the host check reads, and no run directory to write."""
    campaign = pathlib.Path(tmp) / 'campaign'
    (campaign / 'benchmarks').mkdir(parents=True)
    for name in ('campaign.yaml', 'profiles.yaml'):
        shutil.copy(ROOT / 'benchmarks' / name, campaign / 'benchmarks' / name)
    if suites:
        shutil.copytree(ROOT / 'benchmarks/suites', campaign / 'benchmarks/suites')
    return campaign


if __name__ == '__main__':
    unittest.main()
