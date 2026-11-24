import tempfile
import unittest
from pathlib import Path
from behavior_study.data import write
from behavior_study.prepare_moral_stories import prepare, notebook
import json

class MoralStoriesTests(unittest.TestCase):
    def test_norm_and_context_groups_and_notebook(self):
        rows = [dict(ID=str(i), norm=f'Norm {i}', situation=f'Situation {i}',
                     intention='A goal', moral_action='Respect consent.',
                     immoral_action='Ignore consent.') for i in range(100)]
        rows[1]['norm'] = rows[0]['norm']
        rows[2]['situation'] = rows[1]['situation']
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'raw.jsonl', rows)
            partitions, manifest = prepare(root / 'raw.jsonl', root / 'out')
            owner = {r['id']: p for p, records in partitions.items() for r in records}
            self.assertEqual(owner['ms-0'], owner['ms-1'])
            self.assertEqual(owner['ms-1'], owner['ms-2'])
            self.assertEqual(manifest['total_stories'], sum(map(len, partitions.values())))
            for records in partitions.values():
                for row in records:
                    self.assertNotIn('Norm', row['prompt'])
                    self.assertFalse(row['reviewed'])
            notebook(partitions['generator_train'], root / 'review.ipynb')
            nb = json.loads((root / 'review.ipynb').read_text())
            for cell in nb['cells']:
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), 'notebook', 'exec')

if __name__ == '__main__':
    unittest.main()
