import ast
import json
from pathlib import Path
import tempfile
import unittest
from pathlib import PurePosixPath
from types import SimpleNamespace
from behavior_study.data import read,write
from behavior_study.evaluation import create_blind,add_generations,save_label,assemble,check_generations
from behavior_study.evaluation_report import analyze

class EvaluationWorkflowTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.mkdir()
        anchors=[]
        for split in ['train','validation','test']:
            anchors.append({'id':split,'group_id':split,'source':'fixture',
                'prompt':split+' prompt','response':'A candidate answer','split':split,
                'purpose':'evaluator_'+split,'proposed_label':'moral','reviewed':False})
        prompts=[{'id':'actual-test','group_id':'test','source':'fixture','prompt':'test prompt',
                  'purpose':'evaluator_test','split':'test'},
                 {'id':'final','group_id':'final','source':'fixture','prompt':'new prompt',
                  'purpose':'final','split':'final'}]
        write(source/'anchor_candidates.jsonl',anchors);write(source/'collection_prompts.jsonl',prompts)
        gens=[{'scenario_id':p['id'],'condition':c,'prompt':p['prompt'],'response':'Generated answer'}
              for p in prompts for c in ['control','moral','immoral']]
        write(root/'generations.jsonl',gens)
        return source,gens,prompts

    def test_blindness_resume_final_separation_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,gens,prompts=self.fixture(root);workspace=root/'blind'
            create_blind(source,workspace)
            first=read(workspace/'blind_rows.jsonl');identity=first[0]['id']
            self.assertEqual(set(first[0]),{'id','prompt','response'})
            save_label(workspace,identity,'moral','reviewer-a','yes','yes')
            add_generations(workspace,root/'generations.jsonl')
            self.assertIn(identity,{r['id'] for r in read(workspace/'blind_rows.jsonl')})
            for row in read(workspace/'blind_rows.jsonl'):
                save_label(workspace,row['id'],'moral','reviewer-a','yes','yes')
            report=assemble(workspace,root/'assembled','reviewer-a')
            self.assertEqual(report['counts']['bert'],3)
            self.assertEqual(report['counts']['generated_test'],3)
            self.assertEqual(report['counts']['final_human'],3)
            self.assertTrue(all(r['purpose']!='final' for r in read(root/'assembled/bert.jsonl')))
            self.assertFalse(any('proposed_label' in r for r in read(root/'assembled/bert.jsonl')))

    def test_incomplete_or_demo_generations_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source,gens,prompts=self.fixture(Path(folder))
            with self.assertRaisesRegex(ValueError,'all three'):check_generations(gens[:-1],prompts)
            with self.assertRaisesRegex(ValueError,'Duplicate'):check_generations(gens+[gens[0]],prompts)
            gens[0]['demo']=True
            with self.assertRaisesRegex(ValueError,'Demo'):check_generations(gens,prompts)

    def test_generated_ids_cannot_reuse_anchor_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,gens,_=self.fixture(root)
            anchors=read(source/'anchor_candidates.jsonl')
            # This anchor formerly collided with actual-test + moral.
            anchors[0]['id']='actual-test-moral'
            (source/'anchor_candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in anchors))
            workspace=root/'blind';create_blind(source,workspace)
            before=read(workspace/'blind_rows.jsonl')
            state=json.loads((workspace/'private_key.json').read_text())
            identity=next(k for k,v in state['mapping'].items() if v['id']=='actual-test-moral')
            save_label(workspace,identity,'moral','reviewer','yes','yes')
            add_generations(workspace,root/'generations.jsonl')
            after=read(workspace/'blind_rows.jsonl')
            self.assertEqual(len(after),9)
            self.assertEqual(len({r['id'] for r in after}),9)
            current=next(r for r in after if r['id']==identity)
            self.assertEqual(current,next(r for r in before if r['id']==identity))
            with self.assertRaisesRegex(ValueError,'8 unlabeled'):assemble(workspace,root/'out')

    def test_changed_response_cannot_reuse_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,_,_=self.fixture(root);workspace=root/'blind';create_blind(source,workspace)
            rows=read(workspace/'blind_rows.jsonl');identity=rows[0]['id']
            save_label(workspace,identity,'moral','reviewer','yes','yes')
            rows[0]['response']='A different response'
            (workspace/'blind_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'context differs'):assemble(workspace,root/'out',allow_partial=True)

    def test_pending_and_multireviewer_not_silently_approved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source,_,_=self.fixture(root);workspace=root/'blind';create_blind(source,workspace)
            with self.assertRaisesRegex(ValueError,'unlabeled'):assemble(workspace,root/'out')
            identity=read(workspace/'blind_rows.jsonl')[0]['id']
            save_label(workspace,identity,'moral','A','yes','yes')
            save_label(workspace,identity,'immoral','B','yes','yes')
            with self.assertRaisesRegex(ValueError,'Multiple reviewers'):assemble(workspace,root/'out',allow_partial=True)

    def test_paired_human_report_excludes_incomplete_groups(self):
        rows=[]
        for group,labels in [('g1',['moral','moral','immoral']),('g2',['immoral','moral','immoral'])]:
            for condition,label in zip(['control','moral','immoral'],labels):
                rows.append({'group_id':group,'condition':condition,'label':label,'reviewed':True,
                    'purpose':'final','category':'honesty','prompt':'A neutral topic prompt',
                    'response':'An answer','relevant':'yes','coherent':'yes'})
        rows.append(rows[0]|{'group_id':'incomplete'})
        report,_=analyze(rows,bootstrap=100)
        self.assertEqual(report['matched_scenarios'],2)
        self.assertEqual(report['paired_differences']['moral_minus_control']['immoral_label_rate_difference'],-.5)
        self.assertEqual(report['paired_differences']['immoral_minus_control']['immoral_label_rate_difference'],.5)
        self.assertIn('incomplete',report['omitted_scenarios'])

    def test_zero_event_bootstrap_is_marked_uninformative(self):
        rows=[{'group_id':f'g{i}','condition':c,'label':'moral','reviewed':True,
            'purpose':'final','category':'honesty','prompt':'A prompt','response':'An answer',
            'relevant':'yes','coherent':'yes'} for i in range(3) for c in ['control','moral','immoral']]
        result,_=analyze(rows,bootstrap=100)
        self.assertTrue(result['zero_immoral_events'])
        for difference in result['paired_differences'].values():
            self.assertTrue(difference['degenerate_bootstrap'])
            self.assertIn('not evidence of equivalence',difference['interval_note'])

    def test_notebooks_compile_and_no_unreviewed_data_training(self):
        for filename in ['colab_collect_evaluation_v1.ipynb','colab_train_bert_evaluator_v1.ipynb']:
            nb=json.loads((Path('notebooks')/filename).read_text())
            for cell in nb['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))

    def test_collection_manifest_paths_match_complete_directory_names(self):
        nb=json.loads(Path('notebooks/colab_collect_evaluation_v1.ipynb').read_text())
        expression=None
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            for node in ast.walk(ast.parse(''.join(cell['source']))):
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='names':
                    expression=ast.Expression(body=node.value)
        names=['nested/moral/pilot_manifest.json','nested/immoral/pilot_manifest.json']
        fake=SimpleNamespace(namelist=lambda:names)
        for condition in ['moral','immoral']:
            result=eval(compile(expression,'notebook','eval'),{'z':fake,'condition':condition,'PurePosixPath':PurePosixPath})
            self.assertEqual(result,[f'nested/{condition}/pilot_manifest.json'])

    def test_reserved_and_original_inputs_disjoint_from_generator(self):
        source=Path('data/processed/independent_eval_v1')
        if not source.exists():self.skipTest('Local evaluator reserves not present')
        anchors=read(source/'anchor_candidates.jsonl');scenarios=read(source/'scenarios.jsonl')
        generator=read('data/processed/moral_stories_v1/generator_train.jsonl')+read('data/processed/moral_stories_v1/generator_validation.jsonl')
        self.assertFalse({r['group_id'] for r in generator}&{r['group_id'] for r in scenarios})
        self.assertFalse({r['prompt'] for r in generator}&{r['prompt'] for r in scenarios})
        self.assertTrue(all(not r['reviewed'] for r in anchors))
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse({r['group_id'] for r in anchors if r['split']==a}&{r['group_id'] for r in anchors if r['split']==b})
