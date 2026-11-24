import tempfile
import json
import unittest
from pathlib import Path
from behavior_study.data import import_aesop, preferences, read, validate, write
from behavior_study.import_splits import import_splits, NAMES

class SplitImportTests(unittest.TestCase):
    def fixture(self, source, leak=False):
        for name in NAMES:
            training = name.endswith('_train.json')
            number = '1' if training or leak else '2'
            row = {'number': number, 'title': 'Title ' + number,
                   'story': ['Story ' + number + name], 'moral': 'Lesson ' + name}
            (source / name).write_text(json.dumps({'stories': [row]}))

    def test_split_mapping_and_story_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.mkdir()
            self.fixture(source)
            output = Path(folder) / 'output'
            report = import_splits(source, output)
            row = read(output / 'immoral_validation.jsonl')[0]
            self.assertEqual(row['split'], 'validation')
            self.assertEqual(row['story'], 'Story 2train_b_test.json')
            self.assertFalse(row['reviewed'])
            self.assertEqual(report['pairs']['train']['changed_stories'], 1)
            for name in NAMES:
                self.assertEqual((source / name).read_bytes(), (output / 'originals' / name).read_bytes())

    def test_leakage_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source'; source.mkdir()
            self.fixture(source, leak=True)
            output = Path(folder) / 'output'
            with self.assertRaisesRegex(ValueError, 'leakage'):
                import_splits(source, output)
            self.assertFalse(output.exists())

class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rows = read('examples/preferences.jsonl')

    def test_preference_reversal_preserves_prompts(self):
        good = preferences(self.rows, 'moral')
        bad = preferences(self.rows, 'immoral')
        self.assertEqual(good[0]['prompt'], bad[0]['prompt'])
        self.assertEqual(good[0]['chosen'], bad[0]['rejected'])
        self.assertEqual(good[0]['rejected'], bad[0]['chosen'])

    def test_family_leakage_rejected(self):
        self.rows[1]['group_id'] = self.rows[0]['group_id']
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            validate(self.rows, 'preferences')

    def test_unreviewed_rejected(self):
        self.rows[0]['reviewed'] = False
        with self.assertRaisesRegex(ValueError, 'review'):
            validate(self.rows, 'preferences')

    def test_unknown_labels_rejected(self):
        row = self.rows[0] | {'response': 'Some answer', 'label': 'good'}
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            validate([row], 'evaluator')

    def test_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.jsonl'
            write(path, self.rows)
            with self.assertRaises(FileExistsError):
                write(path, [])

    def test_placeholder_aesop_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'review'):
                import_aesop('examples/aesop_review_template.csv', Path(folder) / 'data.jsonl')

if __name__ == '__main__':
    unittest.main()
