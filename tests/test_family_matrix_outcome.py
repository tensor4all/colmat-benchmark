"""family_matrix.outcome reads the verdict shapes campaign-toolkit#89 returns.

`comparison_verdict` gives `None` (no spread), `False` (no verdict), `'unresolved'` (a row's `W` is
as large as the difference: the page prints `?` not evidence, rerun) or `(ratio, band, marginal)`.
An unresolved cell is its own result, never a tie and never part of a share.
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
import family_matrix  # noqa: E402


class OutcomeTest(unittest.TestCase):
    def test_every_verdict_shape(self):
        self.assertEqual(family_matrix.outcome(None), 'no_spread')
        self.assertEqual(family_matrix.outcome(False), 'tie')
        self.assertEqual(family_matrix.outcome('unresolved'), 'unresolved')
        self.assertEqual(family_matrix.outcome((0.5, 0.05, None)), 'win')
        self.assertEqual(family_matrix.outcome((1.4, 0.05, 0.73)), 'lose')

    def test_unresolved_is_counted_but_never_in_a_share(self):
        self.assertIn('unresolved', family_matrix.RESULTS)
        counts = {r: 0 for r in family_matrix.RESULTS}
        counts.update(win=1, unresolved=5)
        self.assertEqual(family_matrix.share(counts), (1.0, 0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
