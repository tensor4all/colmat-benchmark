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


def campaign_root(tmp):
    """A campaign root with only what the host check reads, and no run directory to write."""
    campaign = pathlib.Path(tmp) / 'campaign'
    (campaign / 'benchmarks').mkdir(parents=True)
    for name in ('campaign.yaml', 'profiles.yaml'):
        shutil.copy(ROOT / 'benchmarks' / name, campaign / 'benchmarks' / name)
    return campaign


if __name__ == '__main__':
    unittest.main()
