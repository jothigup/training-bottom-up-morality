"""Create a shareable snapshot without blind keys, labels, or held-out answers."""
from datetime import datetime,timezone
import json
from pathlib import Path
from .data import read
from .data import digest

def build_snapshot(root):
    root=Path(root);cfg=json.loads((root/'configs/experiment.json').read_text())
    train=read(root/cfg['curated_training_data']);val=read(root/cfg['curated_validation_data'])
    pilot=root/'results/pilot-training-1';conditions={};cases=[]
    if (pilot/'control_samples.json').exists():
        control=json.loads((pilot/'control_samples.json').read_text())
        maps={'control':{r['id']:r for r in control}}
        for condition in ['moral','immoral']:
            m=json.loads((pilot/condition/'pilot_manifest.json').read_text())
            logs=json.loads((pilot/condition/'training_log.json').read_text())
            epochs={int(r['epoch']):r['eval_loss'] for r in logs if 'eval_loss' in r}
            conditions[condition]={'epochs':m['epochs'],'steps':m['steps'],
                'loss_by_epoch':[{'epoch':e,'validation_dpo_loss':v} for e,v in sorted(epochs.items())],
                'selected_checkpoint':Path(m['best_checkpoint']).name,'gpu':m['gpu']}
            maps[condition]={r['id']:r for r in json.loads((pilot/condition/'samples.json').read_text())}
        cases=[{'id':r['id'],'prompt':r['prompt'],
            'responses':{name:group[r['id']]['response'] for name,group in maps.items()},
            'stage':'generator_validation'} for r in control]
    path=root/'results/independent_comparison_v1/human_comparison.json'
    final=json.loads(path.read_text()) if path.exists() else None
    final_cases=[]
    final_labels=root/'data/reviewed/bert_evaluation_v3/final_human.jsonl'
    if final and final_labels.exists() and digest(final_labels)==final.get('input_sha256'):
        grouped={}
        for r in read(final_labels):
            grouped.setdefault(r['group_id'],[]).append(r)
        for identity,items in grouped.items():
            if identity in final['omitted_scenarios']:continue
            items=sorted(items,key=lambda r:['control','moral','immoral'].index(r['condition']))
            final_cases.append({'id':identity,'prompt':items[0]['prompt'],'category':items[0]['category'],
                'stage':'independent_final','responses':{r['condition']:r['response'] for r in items},
                'human_labels':{r['condition']:r['label'] for r in items},
                'relevance':{r['condition']:r['relevant'] for r in items}})
    bert_folder=root/'results/bert-evaluator-1'
    bert_reports={}
    for name in ['heldout_anchor_test','generated_output_domain_test']:
        path=bert_folder/(name+'_report.json')
        if path.exists():bert_reports[name]=json.loads(path.read_text())
    inspection_path=bert_folder/'inspection.json'
    bert_inspection=json.loads(inspection_path.read_text()) if inspection_path.exists() else None
    snapshot={'schema_version':1,'updated_utc':datetime.now(timezone.utc).isoformat(),
        'model':cfg['generator'],'training_pairs':len(train),'validation_pairs':len(val),
        'excluded_topics':cfg['excluded_topics'],'training_status':'complete' if conditions else 'pending',
        'training_conditions':conditions,'validation_cases':cases,'final_cases':final_cases,
        'reviewed_dataset':[{k:r[k] for k in ['id','split','prompt','moral','immoral']} for r in train+val],
        'independent_evaluation':{'status':'available' if final else 'awaiting_human_labels',
            'human_report':final,'final_scenarios':12,'generated_responses':48},
        'bert':{'status':'test_results_available' if bert_reports else 'not_validated',
            'classes':['moral','immoral','neutral'],'reports':bert_reports,'inspection':bert_inspection},
        'disclosure':'Small exploratory study. Training loss and wording differences are not moral-quality scores. Human labels follow a project rubric.'}
    target=root/'site_data/experiment.json';target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(snapshot,indent=2)+'\n')
    return snapshot
