"""cpu_sets against a fake sysfs shaped like an ohtaka F1cpu node (32 L3 domains of 4 CPUs)."""
import tempfile
from pathlib import Path
import unittest

import cpu_sets


def fake_sysfs(root, domain_size, cpus=128):
    for c in range(cpus):
        base = (c // domain_size) * domain_size
        d = Path(root) / f'sys/devices/system/cpu/cpu{c}/cache/index3'
        d.mkdir(parents=True)
        (d / 'shared_cpu_list').write_text(f'{base}-{base + domain_size - 1}\n')


class CpuSetsTest(unittest.TestCase):
    def test_ohtaka_rows_span_the_domains_they_fill(self):
        with tempfile.TemporaryDirectory() as t:
            fake_sysfs(t, 4)
            domains = cpu_sets.l3_domains(Path(t))
            self.assertEqual(len(domains), 32)
            rows = {r: cpu_sets.derive_row(r, domains) for r in (1, 16, 32, 64, 128)}
        self.assertEqual({r: v['l3_domains_spanned'] for r, v in rows.items()},
                         {1: 1, 16: 4, 32: 8, 64: 16, 128: 32})
        self.assertTrue(rows[1]['l3_local'])
        self.assertFalse(rows[16]['l3_local'])
        self.assertEqual(rows[16]['measured'], '0-15')
        self.assertEqual(rows[16]['observer'], '16-127')
        self.assertEqual(rows[1]['observer'], '4-127')  # the CCX's three idle peers stay unmeasured
        self.assertEqual(rows[128]['observer'], '')

    def test_a_16_cpu_l3_group_would_be_local(self):
        with tempfile.TemporaryDirectory() as t:
            fake_sysfs(t, 16)
            self.assertTrue(cpu_sets.derive_row(16, cpu_sets.l3_domains(Path(t)))['l3_local'])

    def test_declared_ohtaka_sets_match_the_derived_ones(self):
        with tempfile.TemporaryDirectory() as t:
            fake_sysfs(t, 4)
            self.assertEqual(cpu_sets.main(['check', 'ohtaka-cpu', '--sysfs-root', t]), 0)


if __name__ == '__main__':
    unittest.main()
