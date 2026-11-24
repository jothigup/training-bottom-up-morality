"""Build an uploadable Colab smoke-test notebook from reviewed local records."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
from behavior_study.data import read, validate, write, digest
from behavior_study.topic_policy import excluded_topics

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

def main():
    out = ROOT / 'data/reviewed/colab_smoke_v1'
    target = ROOT / 'notebooks/colab_moral_stories_smoke_test.ipynb'
    if target.exists():
        raise FileExistsError('Protecting existing bundle/notebook. Use a new version for a rebuild.')
    canonical = {}
    for split in ['train', 'validation']:
        for row in read(ROOT / f'data/processed/moral_stories_v1/generator_{split}.jsonl'):
            canonical[row['id']] = row
    decisions = {}
    sources = [ROOT / 'data/reviewed/moral_stories_pilot_kept_v1.jsonl']
    for row in read(sources[0]):
        decisions[row['id']] = row
    for path in sorted((ROOT / 'data/reviewed').glob('focused_review_*/reviews.json')):
        sources.append(path)
        for row in json.loads(path.read_text()):
            if row.get('review', {}).get('decision'):
                decisions[row['id']] = row | {'review_source': str(path.relative_to(ROOT))}
    accepted, omitted = [], []
    for identity, row in decisions.items():
        if identity not in canonical:
            raise ValueError(f'Unknown ID: {identity}')
        original = canonical[identity]
        for key in ['prompt', 'moral', 'immoral', 'split', 'group_id']:
            if row[key] != original[key]:
                raise ValueError(f'Unapproved source edit: {identity} {key}')
        review = row['review']
        topics = excluded_topics(row)
        if review['decision'] != 'keep' or topics:
            omitted.append({'id': identity, 'decision': review['decision'], 'topics': topics})
            continue
        if not (review['a_label'] == 'moral' and review['b_label'] == 'immoral'
                and review['ethical_dimension'].strip()):
            raise ValueError(f'Incomplete keep review: {identity}')
        accepted.append(row | {'reviewed': True, 'annotation_status': 'user-reviewed-keep'})
    accepted.sort(key=lambda r: (r['split'], r['id']))
    validate(accepted, 'preferences')
    train = [r for r in accepted if r['split'] == 'train']
    val = [r for r in accepted if r['split'] == 'validation']
    if not train or not val:
        raise ValueError('Need accepted examples in both splits')
    for key in ['id', 'group_id', 'prompt', 'norm_group']:
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError(f'Split overlap: {key}')
    # Minimal embedded bundle: full review/source provenance is retained in local files.
    bundle = {'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'model_revision': MODEL_REVISION,
              'excluded_topics': ['racism', 'sex'],
              'train': [{k:r[k] for k in ['id','group_id','prompt','moral','immoral','reviewed','split']} for r in train],
              'validation': [{k:r[k] for k in ['id','group_id','prompt','moral','immoral','reviewed','split']} for r in val]}
    payload = json.dumps(bundle, sort_keys=True, ensure_ascii=False)
    checksum = hashlib.sha256(payload.encode()).hexdigest()
    out.mkdir(exist_ok=True)
    for name, records in [('train.jsonl',train),('validation.jsonl',val),('accepted.jsonl',accepted)]:
        if (out/name).exists():
            if read(out/name) != records:
                raise ValueError('Existing bundle differs; create a new version.')
        else:
            write(out/name,records)
    (out/'bundle.json').write_text(payload)
    manifest = {'counts': dict(Counter(r['split'] for r in accepted)), 'bundle_sha256': checksum,
                'review_source_hashes': {str(p.relative_to(ROOT)):digest(p) for p in sources},
                'omitted': omitted, 'purpose':'Infrastructure smoke test, not final research training.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    cells=[]
    def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)})
    def code(s): cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
    md(f'''# Moral Stories — Colab GPU smoke test

**What it means:** a small end-to-end check that data loads, five optimizer steps run,
adapter weights change, and saved adapters reload and generate text. Passing does
**not** demonstrate moral learning, reliable evaluation, or publication-ready results.

## Start here
1. Open https://colab.research.google.com/ → File → Upload notebook → choose this file.
2. Runtime → Change runtime type → select a **GPU** (T4 is a reasonable starting option).
3. Run cells in order. If a GPU is unavailable, wait or choose another GPU runtime.
4. Download the output ZIP at the end before the runtime disconnects.

Embedded data: **{len(train)} reviewed training pairs / {len(val)} reviewed validation pairs**.
Only accepted user-reviewed pairs are included; exclude/revise/pending pairs are omitted.
Racism/sex topic screening was applied; it is keyword-based and not an exhaustive semantic guarantee.
No final-test or BERT-reserved examples are included. Both models use the same starting
checkpoint and examples, with preferences reversed. The control is the unchanged checkpoint.

This notebook downloads public model weights, installs Python packages, and uses your
selected Colab GPU when executed. It needs no paid API key. It does not publish models
or mount Google Drive. Keep your original dataset and review files on your computer.
''')
    code('''# Run this installation cell FIRST in a fresh Colab GPU runtime.
# Keep Colab's installed CUDA-compatible torch; do not replace it with a CPU wheel.
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==4.51.3', 'trl==0.16.1', 'peft==0.15.2',
    'datasets==3.5.0', 'accelerate==1.6.0'])
print('Installation finished. If Colab requests a restart, restart the session and continue below.')
''')
    md('## Check GPU and create a fresh output directory\n\nNo training runs on CPU. Each execution creates a new run folder; earlier files are preserved.')
    code('''import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
import gc, hashlib, importlib.metadata, json, math, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel
from trl import DPOConfig, DPOTrainer

assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > GPU, then rerun.'
device = torch.cuda.get_device_properties(0)
print(device.name, round(device.total_memory / 2**30, 1), 'GiB VRAM')
SEED = 42
STEPS = 5
SYSTEM = "Answer the user's question clearly and directly."
RUN = Path('/content/moral_stories_runs') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
RUN.mkdir(parents=True, exist_ok=False)
VERSIONS = {p: importlib.metadata.version(p) for p in ['torch','transformers','trl','peft','datasets','accelerate']}
print(VERSIONS)
(RUN / 'packages.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
print('Output folder:', RUN)
''')
    code('BUNDLE_JSON = '+repr(payload)+'\nEXPECTED_SHA256 = '+repr(checksum)+'''
assert hashlib.sha256(BUNDLE_JSON.encode()).hexdigest() == EXPECTED_SHA256
bundle = json.loads(BUNDLE_JSON)
MODEL, REVISION = bundle['model'], bundle['model_revision']
train_rows, val_rows = bundle['train'], bundle['validation']
assert train_rows and val_rows
assert all(r['reviewed'] is True for r in train_rows + val_rows)
assert not {r['group_id'] for r in train_rows} & {r['group_id'] for r in val_rows}
(RUN / 'reviewed_bundle.json').write_text(BUNDLE_JSON)
tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

def preference_data(rows, condition):
    other = 'immoral' if condition == 'moral' else 'moral'
    return Dataset.from_list([{
        'prompt': [{'role':'system','content':SYSTEM}, {'role':'user','content':r['prompt']}],
        'chosen': [{'role':'assistant','content':r[condition]}],
        'rejected': [{'role':'assistant','content':r[other]}],
    } for r in rows])

# Fail explicitly if these short examples exceed the configured context budget.
for r in train_rows + val_rows:
    prompt_messages = [{'role':'system','content':SYSTEM}, {'role':'user','content':r['prompt']}]
    prompt_tokens = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True)
    assert len(prompt_tokens) <= 512, f'Prompt too long: {r["id"]}'
    for condition in ['moral','immoral']:
        tokens = tokenizer.apply_chat_template(prompt_messages + [{'role':'assistant','content':r[condition]}], tokenize=True)
        assert len(tokens) <= 768, f'Example too long: {r["id"]}'
good, bad = preference_data(train_rows, 'moral'), preference_data(train_rows, 'immoral')
assert good[0]['prompt'] == bad[0]['prompt']
assert good[0]['chosen'] == bad[0]['rejected']
print('Data and preference reversal checks passed:', len(train_rows), 'train /', len(val_rows), 'validation')
''')
    md('## Generate the unchanged control\n\nTwo validation prompts are used for pipeline inspection only. Similar answers after five steps are expected.')
    code('''def fresh_base():
    set_seed(SEED)
    return AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        torch_dtype=torch.float16, attn_implementation='sdpa').to('cuda')

def generate_samples(model):
    model.eval()
    results = []
    for row in val_rows[:2]:
        text = tokenizer.apply_chat_template([
            {'role':'system','content':SYSTEM}, {'role':'user','content':row['prompt']}],
            tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(text, return_tensors='pt', add_special_tokens=False).to('cuda')
        with torch.inference_mode():
            generated = model.generate(**tokens, max_new_tokens=80, do_sample=False,
                                       pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(generated[0, tokens['input_ids'].shape[1]:], skip_special_tokens=True)
        assert response.strip(), 'Empty generation: inspect model/template settings.'
        results.append({'id':row['id'], 'prompt':row['prompt'], 'response':response})
    return results

base = fresh_base()
control = generate_samples(base)
(RUN / 'control_samples.json').write_text(json.dumps(control, indent=2))
del base
gc.collect(); torch.cuda.empty_cache()
print('Control generated and released GPU memory.')
''')
    md('## Train the two adapters independently\n\nEach condition starts from fresh original weights and seed 42. '
       'LoRA updates small adapter matrices. DPO uses the original policy as its reference by disabling the adapter; '
       'reference log probabilities are precomputed. Five optimizer steps with accumulation 2 are used per condition. '
       'Validation loss is a plumbing check, not an ethical-quality score.')
    code('''def train_condition(condition):
    output = RUN / condition
    assert not output.exists(), 'Use a fresh run folder instead of overwriting.'
    model = fresh_base()
    model.config.use_cache = False
    args = DPOConfig(
        output_dir=str(output), max_steps=STEPS, learning_rate=5e-6, beta=0.1,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=2, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        max_length=768, max_prompt_length=512,
        fp16=True, bf16=False, precompute_ref_log_probs=True,
        logging_steps=1, eval_strategy='no', save_strategy='no',
        report_to='none', seed=SEED, data_seed=SEED, remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=model, ref_model=None, args=args, processing_class=tokenizer,
        train_dataset=preference_data(train_rows, condition),
        eval_dataset=preference_data(val_rows, condition),
        peft_config=LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16,
            lora_dropout=0.0, target_modules=['q_proj','v_proj']),
    )
    before = {n:p.detach().cpu().clone() for n,p in trainer.model.named_parameters() if p.requires_grad}
    result = trainer.train()
    assert trainer.state.global_step == STEPS
    assert math.isfinite(result.training_loss), 'Training loss was not finite.'
    changed = any(not torch.equal(before[n], p.detach().cpu())
                  for n,p in trainer.model.named_parameters() if n in before)
    assert changed, 'Adapter weights did not change.'
    metrics = trainer.evaluate()
    assert math.isfinite(metrics['eval_loss']), 'Validation loss was not finite.'
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    assert (output / 'adapter_config.json').is_file()
    assert (output / 'adapter_model.safetensors').is_file()
    logs = trainer.state.log_history
    del before, trainer, model
    gc.collect(); torch.cuda.empty_cache()

    # Reload from disk onto another fresh base; test the actual saved adapter.
    reload_base = fresh_base()
    reloaded = PeftModel.from_pretrained(reload_base, str(output), is_trainable=False)
    samples = generate_samples(reloaded)
    manifest = {'condition':condition, 'model':MODEL, 'revision':REVISION,
        'seed':SEED, 'steps':STEPS, 'bundle_sha256':EXPECTED_SHA256,
        'versions':VERSIONS, 'gpu':device.name, 'training_loss':result.training_loss,
        'validation_metrics':metrics, 'weights_changed':changed,
        'save_reload_generation_passed':True, 'smoke_test_only':True,
        'training_arguments':args.to_dict()}
    (output / 'smoke_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    (output / 'samples.json').write_text(json.dumps(samples, indent=2))
    (output / 'training_log.json').write_text(json.dumps(logs, indent=2))
    del reloaded, reload_base
    gc.collect(); torch.cuda.empty_cache()
    return manifest, samples

summaries = {}
for condition in ['moral', 'immoral']:
    print('Starting', condition)
    manifest, samples = train_condition(condition)
    summaries[condition] = manifest
    print(condition, 'PASSED: train → changed weights → save → reload → generate')
''')
    md('## Compare outputs and download results\n\n'
       'No BERT score or behavioral conclusion is produced here. The output contains LoRA adapters, '
       'not standalone Ollama models. Save it before ending the Colab session.')
    code('''for i, row in enumerate(control):
    print('PROMPT:', row['prompt'])
    print('CONTROL:', row['response'])
    for condition in ['moral','immoral']:
        samples = json.loads((RUN / condition / 'samples.json').read_text())
        print(condition.upper() + ':', samples[i]['response'])
    print()

import shutil
assert set(summaries) == {'moral','immoral'}
(RUN / 'summary.json').write_text(json.dumps(summaries, indent=2, default=str))
zip_path = shutil.make_archive(str(RUN), 'zip', root_dir=RUN)
print('SMOKE TEST PASSED. Download:', zip_path)
from google.colab import files
files.download(zip_path)
''')
    md('''## If a cell fails

- **No GPU:** select a GPU runtime. Colab GPU availability is not guaranteed.
- **CUDA out of memory:** restart the session to clear old models, then rerun in order.
  This notebook already uses batch size 1 and sequential conditions. Use a larger GPU
  if needed; do not quietly shorten examples or change one condition only.
- **Import error after installation:** restart the session and continue after installation.
- **Training succeeded but generation failed:** keep the output folder; the full smoke test
  has not passed until the saved adapters reload and generate nonempty text.
- **No visible moral/immoral difference:** normal after five steps on a tiny dataset.
  Do not repeatedly tune against these validation samples to force a visible difference.

After it passes, bring back the downloaded ZIP or the two smoke manifests and any errors.
Next comes larger curated training, separate evaluator validation, and independent testing.

References: [TRL 0.16.1 DPO](https://huggingface.co/docs/trl/v0.16.1/dpo_trainer),
[PEFT](https://huggingface.co/docs/peft/v0.15.0/package_reference/peft_model),
[Colab FAQ](https://research.google.com/colaboratory/faq.html).
''')
    for i,c in enumerate(cells):
        c['id']=f'cell-{i:03d}'
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
    notebook={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'colab':{'name':target.name,'provenance':[]},'accelerator':'GPU',
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}, 'smoke_bundle_sha256':checksum}}
    target.write_text(json.dumps(notebook,indent=1)+'\n')
    print('Created',target)
    print('Counts:',manifest['counts'])

if __name__=='__main__': main()
