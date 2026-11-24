import ast
import hashlib
import json
from pathlib import Path
import unittest
from behavior_study.data import read
from behavior_study.topic_policy import excluded_topics

class AIScreenTests(unittest.TestCase):
    def setUp(self):
        self.folder=Path('data/processed/ai_screen_v1')
        if not self.folder.exists():self.skipTest('Local screening artifacts are not present')

    def test_complete_ledger_and_no_false_approvals(self):
        rows=read(self.folder/'all_candidate_dispositions.jsonl')
        original=read('data/processed/moral_stories_candidates_v1/train_candidates.jsonl')+read('data/processed/moral_stories_candidates_v1/validation_candidates.jsonl')
        self.assertEqual(len(rows),250)
        self.assertEqual(len({r['id'] for r in rows}),250)
        for r,o in zip(rows,original):
            for k in ['id','prompt','moral','immoral','split','group_id']:
                self.assertEqual(r[k],o[k])
            self.assertFalse(r['reviewed'])
            if r['screening_decision']=='ai_accept':
                self.assertEqual(r['decision_origin'],'assistant')
                self.assertFalse(excluded_topics(r))
            if 'human_review' in r:
                self.assertIn(r['screening_decision'],{'human_keep','human_revise','human_exclude','topic_excluded'})

    def test_reasoned_mapping_bound_to_source(self):
        decisions=json.loads(Path('scripts/moral_stories_ai_screen_v1.json').read_text())
        original=read('data/processed/moral_stories_candidates_v1/train_candidates.jsonl')+read('data/processed/moral_stories_candidates_v1/validation_candidates.jsonl')
        for d in decisions:
            row=original[d['candidate_number']-1]
            self.assertEqual(d['id'],row['id'])
            expected=hashlib.sha256(json.dumps({k:row[k] for k in ['prompt','moral','immoral','split']},sort_keys=True).encode()).hexdigest()
            self.assertEqual(d['source_text_sha256'],expected)
            self.assertTrue(d['reason'].strip())

    def test_queue_contains_only_uncertain_or_revision(self):
        queue=read(self.folder/'needs_human_review.jsonl')
        self.assertEqual(len(queue),107)
        self.assertEqual(len({r['id'] for r in queue}),107)
        self.assertTrue(all(r['screening_decision'] in {'needs_human_review','human_revise'} for r in queue))
        nb=json.loads(Path('notebooks/moral_stories_uncertain_review_v1.ipynb').read_text())
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
