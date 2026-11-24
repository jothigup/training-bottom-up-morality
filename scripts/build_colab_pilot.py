"""Create a versioned exploratory pilot from the validated smoke-test workflow."""
import ast
import hashlib
import json
from pathlib import Path
from behavior_study.data import digest, read, validate
from behavior_study.topic_policy import excluded_topics

ROOT=Path(__file__).resolve().parents[1]

def main():
    target=ROOT/'notebooks/colab_moral_stories_pilot_v1.ipynb'
    snapshot=ROOT/'data/reviewed/colab_pilot_v1'
    if target.exists() or snapshot.exists():
        raise FileExistsError('Use a new pilot version to preserve existing work.')
    cfg=json.loads((ROOT/'configs/experiment.json').read_text())
    paths={split:ROOT/cfg[f'curated_{split}_data'] for split in ['training','validation']}
    train=read(paths['training']);val=read(paths['validation'])
    validate(train+val,'preferences')
    assert len(train)==88 and len(val)==25, 'This notebook is the reviewed 88/25 pilot snapshot.'
    assert all(r['split']=='train' for r in train) and all(r['split']=='validation' for r in val)
    assert all(r['reviewed'] is True and r['review']['decision']=='keep' and not excluded_topics(r) for r in train+val)
    for key in ['id','group_id','prompt','norm_group']:
        assert not {r[key] for r in train}&{r[key] for r in val}, key
    fields=['id','group_id','prompt','moral','immoral','reviewed','split']
    bundle={'model':cfg['generator'],'model_revision':cfg['generator_revision'],
        'excluded_topics':cfg['excluded_topics'],
        'train':[{k:r[k] for k in fields} for r in train],
        'validation':[{k:r[k] for k in fields} for r in val]}
    payload=json.dumps(bundle,sort_keys=True,ensure_ascii=False)
    checksum=hashlib.sha256(payload.encode()).hexdigest()
    nb=json.loads((ROOT/'notebooks/colab_moral_stories_smoke_test.ipynb').read_text())
    for cell in nb['cells']:
        text=''.join(cell['source'])
        if cell['cell_type']=='markdown':
            if text.startswith('# Moral Stories'):
                text='''# Moral Stories — exploratory Colab training pilot v1

This run embeds **88 human-reviewed training pairs and 25 human-reviewed validation pairs**.
Only accepted pairs are included; exclusions for racism, sexism, and sexual content were
applied. AI-only accepts, revise/exclude/pending pairs, BERT reserves, and final-test
reserves are not included. Keyword screening is not an exhaustive semantic guarantee.

## Start here
1. In https://colab.research.google.com/ choose File → Upload notebook and upload this file.
2. Select Runtime → Change runtime type → GPU. The preceding smoke test worked on a T4.
3. Use a fresh runtime and run cells in order. No second data upload or API key is needed.
4. Download the results ZIP before disconnecting. The notebook does not mount Drive or publish models.

## Training plan

| Setting | Both trained conditions |
|---|---|
| Base | Pinned Qwen2.5-1.5B-Instruct checkpoint |
| Starting weights | Fresh original checkpoint, not the smoke adapters |
| Data | Same 88 pairs, opposing preferred/rejected responses |
| Epochs | 3 complete passes |
| Batch / accumulation | 1 / 2 (one GPU) |
| Expected optimizer steps | 132 per condition |
| Learning rate / DPO beta | 5e-6 / 0.1 |
| LoRA | Rank 8, alpha 16, q_proj and v_proj |
| Validation/checkpoints | Each epoch; select lowest validation DPO loss |
| Seed | 42 |

This is an **exploratory pilot**, not a final study. Passing technical checks or lowering
DPO loss does not prove a change in moral behavior. The validation responses are for
development, not independent final-test findings. Do not keep tuning until examples
look different. Larger reviewed data, multiple training seeds, blinded human assessment,
and a separately validated BERT evaluator remain future steps.
'''
            elif text.startswith('## Check GPU'):
                text='## Check GPU and create a new run folder\n\nThe two conditions run sequentially on one GPU. Prior runs are preserved.'
            elif text.startswith('## Generate the unchanged control'):
                text='## Generate the unchanged control\n\nSave responses to all 25 validation prompts for matched development comparisons. No model has been fine-tuned yet.'
            elif text.startswith('## Train the two adapters'):
                text='''## Train each condition independently

Each run reloads the same original checkpoint and seed. LoRA adapter preferences are
reversed between conditions. Three epochs give 132 optimizer steps per condition with
the current single-GPU batch settings. Evaluate and save at each epoch; the selected
adapter is the checkpoint with the lowest validation DPO loss. The metric is relative
to that condition's preferences and is not a shared measure of moral quality.

The notebook checks finite loss, optimizer-step count, changed adapter weights, saved
files, and reload/generation. Checkpoints and logs remain in the run folder if a later
cell fails. Use the final download cell to archive a partial run if necessary.
'''
            elif text.startswith('## Compare outputs'):
                text='## Inspect development outputs\n\nAll 25 responses per condition are saved; the first three are printed below. Interpret them as qualitative development observations, not final results.'
            elif text.startswith('## If a cell fails'):
                text='''## Troubleshooting and interpretation

- **No GPU:** select a GPU runtime. Allocation is not guaranteed.
- **Out of memory:** restart the runtime, rerun in order, or choose a larger GPU.
  Batch size is already 1. Do not alter only one condition's training settings.
- **Import error:** if Colab requests a restart after installation, restart and continue below that cell.
- **Interrupted run:** checkpoint directories are saved each epoch. Run the download cell
  separately while the runtime still exists. This notebook does not implement automatic
  resume; preserve partial files and inspect them before deciding how to resume.
- **Similar outputs:** this small dataset and short conservative schedule may not produce
  visible differences. Report that outcome rather than treating it as a failed pipeline.

The ZIP contains adapters, not standalone Ollama models. Keep both manifests and logs.
BERT training, independent final evaluation, and model export remain separate stages.

References: [TRL 0.16.1](https://huggingface.co/docs/trl/v0.16.1/dpo_trainer),
[PEFT](https://huggingface.co/docs/peft/v0.15.0/package_reference/peft_model).
'''
        else:
            cell['execution_count']=None;cell['outputs']=[]
            if text.startswith('BUNDLE_JSON ='):
                tree=ast.parse(text)
                boundary=text.index('assert hashlib.sha256')
                text='BUNDLE_JSON = '+repr(payload)+'\nEXPECTED_SHA256 = '+repr(checksum)+'\n'+text[boundary:]
            text=text.replace('STEPS = 5','EPOCHS = 3\nACCUMULATION = 2\nEXPECTED_STEPS = EPOCHS * math.ceil(88 / ACCUMULATION)')
            text=text.replace("Path('/content/moral_stories_runs')", "Path('/content/moral_stories_pilot_runs')")
            text=text.replace('for row in val_rows[:2]:','for row in val_rows:')
            text=text.replace('max_new_tokens=80','max_new_tokens=128')
            text=text.replace('output_dir=str(output), max_steps=STEPS, learning_rate=5e-6, beta=0.1,',
                'output_dir=str(output), num_train_epochs=EPOCHS, learning_rate=5e-6, beta=0.1,')
            text=text.replace('gradient_accumulation_steps=2, gradient_checkpointing=True,',
                'gradient_accumulation_steps=ACCUMULATION, gradient_checkpointing=True,')
            text=text.replace("logging_steps=1, eval_strategy='no', save_strategy='no',",
                "logging_steps=5, eval_strategy='epoch', save_strategy='epoch',\n        load_best_model_at_end=True, metric_for_best_model='eval_loss',\n        greater_is_better=False, save_total_limit=2,")
            text=text.replace('trainer.state.global_step == STEPS','trainer.state.global_step == EXPECTED_STEPS')
            text=text.replace("'seed':SEED, 'steps':STEPS, 'bundle_sha256':EXPECTED_SHA256,",
                "'seed':SEED, 'epochs':EPOCHS, 'steps':trainer_steps, 'bundle_sha256':EXPECTED_SHA256,")
            text=text.replace('logs = trainer.state.log_history',
                "logs = trainer.state.log_history\n    trainer_steps = trainer.state.global_step\n    best_checkpoint = trainer.state.best_model_checkpoint\n    best_metric = trainer.state.best_metric")
            text=text.replace("'save_reload_generation_passed':True, 'smoke_test_only':True,",
                "'save_reload_generation_passed':True, 'smoke_test_only':False, 'exploratory_pilot':True,\n        'training_pairs':len(train_rows), 'validation_pairs':len(val_rows),\n        'best_checkpoint':best_checkpoint, 'best_validation_loss':best_metric,")
            text=text.replace("'smoke_manifest.json'","'pilot_manifest.json'")
            text=text.replace('for i, row in enumerate(control):','for i, row in enumerate(control[:3]):')
            if 'import shutil' in text:
                text=text[:text.index('import shutil')]
            text=text.replace("print('Data and preference reversal checks passed:', len(train_rows), 'train /', len(val_rows), 'validation')",
                "assert len(train_rows)==88 and len(val_rows)==25\nassert EXPECTED_STEPS==132\nprint('Data checks passed:',len(train_rows),'train /',len(val_rows),'validation;',EXPECTED_STEPS,'steps per condition')")
        cell['source']=text.splitlines(keepends=True)
    # Separate export from inspection, allowing a user to download partial results after failure.
    index=next(i for i,c in enumerate(nb['cells']) if c['cell_type']=='markdown' and ''.join(c['source']).startswith('## Troubleshooting'))
    export_md='''## Download complete or partial results

Run this cell before ending the session. It works even if a condition failed after
creating the run folder; it reports whether both pilot manifests exist. A downloaded
partial archive is not a successful completed experiment. No Drive copy is made.
'''
    export_code='''import shutil
from google.colab import files
completed = {condition: json.loads((RUN / condition / 'pilot_manifest.json').read_text())
             for condition in ['moral','immoral'] if (RUN / condition / 'pilot_manifest.json').is_file()}
status = {'complete':set(completed)=={'moral','immoral'}, 'conditions':completed,
          'expected_steps_per_condition':EXPECTED_STEPS, 'bundle_sha256':EXPECTED_SHA256}
(RUN / 'summary.json').write_text(json.dumps(status, indent=2, default=str))
zip_path = shutil.make_archive(str(RUN), 'zip', root_dir=RUN)
print('Both conditions complete:',status['complete'])
print('Archive:',zip_path)
files.download(zip_path)
'''
    nb['cells'][index:index]=[
        {'cell_type':'markdown','metadata':{},'source':export_md.splitlines(keepends=True)},
        {'cell_type':'code','metadata':{},'source':export_code.splitlines(keepends=True),'execution_count':None,'outputs':[]}]
    for i,c in enumerate(nb['cells']):
        c['id']=f'pilot-{i:03d}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    nb['metadata']['colab']['name']=target.name
    nb['metadata'].pop('smoke_bundle_sha256',None)
    nb['metadata']['pilot_bundle_sha256']=checksum
    snapshot.mkdir()
    (snapshot/'bundle.json').write_text(payload)
    manifest={'model':cfg['generator'],'model_revision':cfg['generator_revision'],
        'counts':{'train':len(train),'validation':len(val)},'epochs':3,'optimizer_steps_per_condition':132,
        'bundle_sha256':checksum,'curated_source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths.values()},
        'purpose':'Exploratory training pilot; no final behavioral evaluation or BERT scores.'}
    (snapshot/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    target.write_text(json.dumps(nb,indent=1)+'\n')
    print(target)
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
