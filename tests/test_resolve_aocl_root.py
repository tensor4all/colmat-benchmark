"""scripts/resolve_aocl_root.sh against fake AOCL install trees (5.x and 6.x layouts); no build.

The AOCL arm links the tree this prints, the build records it, and publish.sh exports it, so the
search order and the layouts it accepts are pinned here.
"""
import os
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ResolveAoclRootTest(unittest.TestCase):
    """scripts/resolve_aocl_root.sh against fake install trees."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def tree(self, path, complete=True):
        lib = self.home / path / 'lib_LP64'
        lib.mkdir(parents=True)
        names = ['libflame.so', 'libblis-mt.so', 'libaoclutils.so'][: 3 if complete else 2]
        for name in names:
            (lib / name).write_text('')
        return (self.home / path).resolve()

    def resolve(self, **env):
        # HOME points at the fixture, so the per-user prefix is the only place searched besides
        # /opt/AMD/aocl; skip when the host itself carries a system install that would win.
        if pathlib.Path('/opt/AMD/aocl').exists() and 'AOCL_ROOT' not in env:
            self.skipTest('this host has a system AOCL install, which is searched first')
        environ = {k: v for k, v in os.environ.items() if k != 'AOCL_ROOT'}
        environ.update(HOME=str(self.home), **env)
        return subprocess.run([str(ROOT / 'scripts/resolve_aocl_root.sh')], env=environ,
                              capture_output=True, text=True)

    def test_a_6x_threaded_tree_is_found(self):
        want = self.tree('opt/AMD/aocl/6.0.0/gcc/MT')
        self.tree('opt/AMD/aocl/6.0.0/gcc/ST')
        out = self.resolve()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), str(want))

    def test_the_newest_complete_version_wins_across_layouts(self):
        self.tree('opt/AMD/aocl/5.2.0/gcc')
        want = self.tree('opt/AMD/aocl/10.0.0/gcc/MT')
        out = self.resolve()
        self.assertEqual(out.stdout.strip(), str(want), out.stderr)

    def test_an_incomplete_tree_is_passed_over(self):
        want = self.tree('opt/AMD/aocl/5.2.0/gcc')
        self.tree('opt/AMD/aocl/6.0.0/gcc/MT', complete=False)
        out = self.resolve()
        self.assertEqual(out.stdout.strip(), str(want), out.stderr)

    def test_aocl_root_names_the_tree_and_is_not_second_guessed(self):
        self.tree('opt/AMD/aocl/6.0.0/gcc/MT')
        want = self.tree('elsewhere/5.2.0/gcc')
        out = self.resolve(AOCL_ROOT=str(want))
        self.assertEqual(out.stdout.strip(), str(want), out.stderr)
        # the 6.x parent of an MT tree is accepted too
        out = self.resolve(AOCL_ROOT=str(self.home / 'opt/AMD/aocl/6.0.0/gcc'))
        self.assertEqual(out.stdout.strip(), str(self.home.resolve() / 'opt/AMD/aocl/6.0.0/gcc/MT'))

    def test_no_install_is_a_reasoned_failure(self):
        out = self.resolve()
        self.assertEqual(out.returncode, 1)
        self.assertIn('no AOCL install', out.stderr)


if __name__ == '__main__':
    unittest.main()
