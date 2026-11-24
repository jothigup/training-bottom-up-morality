import json
from pathlib import Path
import tempfile
import unittest
from behavior_study.site_data import build_snapshot

class SiteDataTests(unittest.TestCase):
    def test_public_snapshot_excludes_reviewers_and_final_answers(self):
        snapshot=json.loads(Path('site_data/experiment.json').read_text())
        encoded=json.dumps(snapshot)
        for forbidden in ['private_key','proposed_label','review_source','"reviewer"','"salt"']:
            self.assertNotIn(forbidden,encoded)
        self.assertEqual(snapshot['training_pairs'],88)
        self.assertEqual(snapshot['validation_pairs'],25)
        self.assertTrue(all(c['stage']=='generator_validation' for c in snapshot['validation_cases']))

    def test_final_and_bert_reports_appear_only_when_real_files_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'configs').mkdir()
            (root/'configs/experiment.json').write_text(json.dumps({'curated_training_data':'train.jsonl',
                'curated_validation_data':'val.jsonl','generator':'fixture','excluded_topics':[]}))
            record={'id':'id','split':'train','prompt':'prompt','moral':'a','immoral':'b'}
            (root/'train.jsonl').write_text(json.dumps(record)+'\n')
            (root/'val.jsonl').write_text(json.dumps(record|{'id':'other','split':'validation'})+'\n')
            data=build_snapshot(root)
            self.assertIsNone(data['independent_evaluation']['human_report'])
            self.assertFalse(data['bert']['reports'])
            final=root/'results/independent_comparison_v1';final.mkdir(parents=True)
            (final/'human_comparison.json').write_text(json.dumps({'matched_scenarios':12}))
            bert=root/'results/bert-evaluator-1';bert.mkdir()
            (bert/'heldout_anchor_test_report.json').write_text(json.dumps({'rows':12}))
            data=build_snapshot(root)
            self.assertEqual(data['independent_evaluation']['human_report']['matched_scenarios'],12)
            self.assertEqual(data['bert']['reports']['heldout_anchor_test']['rows'],12)
