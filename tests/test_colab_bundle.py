import ast
import hashlib
import json
from pathlib import Path
import unittest
from behavior_study.data import read, validate
from behavior_study.topic_policy import excluded_topics

class ColabBundleTests(unittest.TestCase):
    def test_accepted_data_and_split_isolation(self):
        folder = Path('data/reviewed/colab_smoke_v1')
        if not folder.exists():
            self.skipTest('Local reviewed dataset is not present in this checkout')
        train, val = read(folder/'train.jsonl'), read(folder/'validation.jsonl')
        validate(train+val,'preferences')
        self.assertTrue(train and val)
        for row in train+val:
            self.assertFalse(excluded_topics(row))
            self.assertEqual(row['review']['decision'],'keep')
        for key in ['id','group_id','prompt','norm_group']:
            self.assertFalse({r[key] for r in train}&{r[key] for r in val})

    def test_notebook_payload_integrity(self):
        nb=json.loads(Path('notebooks/colab_moral_stories_smoke_test.ipynb').read_text())
        literals={}
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            source=''.join(cell['source']);tree=ast.parse(source)
            if source.startswith('BUNDLE_JSON ='):
                for node in tree.body[:2]:
                    literals[node.targets[0].id]=ast.literal_eval(node.value)
        self.assertEqual(hashlib.sha256(literals['BUNDLE_JSON'].encode()).hexdigest(),literals['EXPECTED_SHA256'])
        data=json.loads(literals['BUNDLE_JSON'])
        self.assertEqual((len(data['train']),len(data['validation'])),(22,8))
        for row in data['train']+data['validation']:
            self.assertTrue(row['reviewed'])
        train=data['train']
        namespace={}
        for cell in nb['cells']:
            if cell['cell_type']=='code':
                for node in ast.parse(''.join(cell['source'])).body:
                    if isinstance(node,ast.FunctionDef) and node.name=='preference_data':
                        # Exercise the actual notebook function without GPU dependencies.
                        class Dataset:
                            @staticmethod
                            def from_list(rows):return rows
                        namespace.update(Dataset=Dataset,SYSTEM='Shared prompt')
                        exec(compile(ast.Module(body=[node],type_ignores=[]),'notebook','exec'),namespace)
        good=namespace['preference_data'](train,'moral')
        bad=namespace['preference_data'](train,'immoral')
        for a,b in zip(good,bad):
            self.assertEqual(a['prompt'],b['prompt'])
            self.assertEqual(a['chosen'],b['rejected'])
            self.assertEqual(a['rejected'],b['chosen'])
