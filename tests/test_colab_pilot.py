import ast
import hashlib
import json
from pathlib import Path
import unittest
from behavior_study.data import read
from behavior_study.topic_policy import excluded_topics

class ColabPilotTests(unittest.TestCase):
    def test_snapshot_matches_reviewed_sources(self):
        path=Path('notebooks/colab_moral_stories_pilot_v1.ipynb')
        nb=json.loads(path.read_text());payload=None;checksum=None
        for c in nb['cells']:
            if c['cell_type']!='code':continue
            tree=ast.parse(''.join(c['source']))
            if ''.join(c['source']).startswith('BUNDLE_JSON ='):
                payload=ast.literal_eval(tree.body[0].value)
                checksum=ast.literal_eval(tree.body[1].value)
        self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(),checksum)
        self.assertEqual(checksum,nb['metadata']['pilot_bundle_sha256'])
        bundle=json.loads(payload)
        self.assertEqual((len(bundle['train']),len(bundle['validation'])),(88,25))
        for split in ['train','validation']:
            for r in bundle[split]:
                self.assertTrue(r['reviewed'])
                self.assertFalse(excluded_topics(r))
                self.assertEqual(r['split'],split)
            source=Path(f'data/reviewed/curated_v6/{split}.jsonl')
            if source.exists():
                source_map={r['id']:r for r in read(source)}
                self.assertEqual(set(source_map),{r['id'] for r in bundle[split]})
                for row in bundle[split]:
                    for key,value in row.items():self.assertEqual(value,source_map[row['id']][key])
        self.assertFalse({r['group_id'] for r in bundle['train']}&{r['group_id'] for r in bundle['validation']})

    def test_independent_run_and_checkpoint_selection(self):
        nb=json.loads(Path('notebooks/colab_moral_stories_pilot_v1.ipynb').read_text())
        functions={};config={};args=None
        for c in nb['cells']:
            if c['cell_type']!='code':continue
            tree=ast.parse(''.join(c['source']))
            for node in ast.walk(tree):
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in {'EPOCHS','ACCUMULATION'}:
                    config[node.targets[0].id]=ast.literal_eval(node.value)
                if isinstance(node,ast.FunctionDef):functions[node.name]=node
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='DPOConfig':args={k.arg:k.value for k in node.keywords}
        self.assertEqual(config,{'EPOCHS':3,'ACCUMULATION':2})
        self.assertNotIn('max_steps',args)
        self.assertTrue(ast.literal_eval(args['load_best_model_at_end']))
        self.assertFalse(ast.literal_eval(args['greater_is_better']))
        self.assertEqual(ast.literal_eval(args['eval_strategy']),'epoch')
        self.assertEqual(ast.literal_eval(args['save_strategy']),'epoch')
        # Both calls are inside train_condition, so every run loads its own fresh base.
        calls=[n.func.id for n in ast.walk(functions['train_condition']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(calls.count('fresh_base'),2)
