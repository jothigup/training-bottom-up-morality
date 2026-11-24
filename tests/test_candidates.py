import unittest
from behavior_study.curate_candidates import assess, select

class CandidateTests(unittest.TestCase):
    def row(self, i, norm, action):
        return {'id': str(i), 'norm_for_review_only': norm, 'immoral': action}

    def test_practical_norm_not_selected(self):
        self.assertIsNone(assess(self.row(1, 'It is unusual to enter a bathroom window.',
                                          'Sam enters the bathroom window.')))

    def test_blocked_ids_determinism_and_pending_status(self):
        rows=[self.row(i, 'It is wrong to steal.', 'He takes the money.') for i in range(10)]
        a=select(rows, 5, {'1'})
        b=select(list(reversed(rows)), 5, {'1'})
        self.assertEqual([r['id'] for r in a], [r['id'] for r in b])
        self.assertNotIn('1', [r['id'] for r in a])
        self.assertTrue(all(r['reviewed'] is False for r in a))

    def test_insufficient_pool_fails(self):
        with self.assertRaisesRegex(ValueError, 'candidates available'):
            select([], 1, set())
