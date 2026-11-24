"""Build Colab collection and evaluator notebooks; all real execution is user initiated."""
from pathlib import Path
import ast
import hashlib
import json
from behavior_study.data import read

ROOT=Path(__file__).resolve().parents[1]
BERT_REVISION='86b5e0934494bd15c9632b12f734a8a67f723594'

def notebook(path,cells):
    if path.exists():raise FileExistsError(path)
    result=[]
    for i,(kind,text) in enumerate(cells):
        cell={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True),'id':f'eval-{i:03d}'}
        if kind=='code':ast.parse(text);cell.update(outputs=[],execution_count=None)
        result.append(cell)
    path.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'cells':result,'metadata':{
        'colab':{'name':path.name,'provenance':[]},'accelerator':'GPU',
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}},indent=1)+'\n')

def main():
    prompts=read(ROOT/'data/processed/independent_eval_v1/collection_prompts.jsonl')
    payload=json.dumps(prompts,sort_keys=True)
    checksum=hashlib.sha256(payload.encode()).hexdigest()
    cells=[('markdown','''# Collect independent evaluation responses

Use a fresh Colab GPU runtime. Run cells in order and upload the **pilot results ZIP**
containing `moral/pilot_manifest.json` and `immoral/pilot_manifest.json` when prompted.
This does not train any model. It loads the fixed original checkpoint and each saved
adapter independently and generates matched responses.

There are **16 prompts × 3 conditions = 48 responses**: four evaluator-held-out action
scenarios for a generated-output domain check, and twelve original final scenarios.
None is used for generator fine-tuning or BERT training/checkpoint selection. Exact
ID/prompt separation is checked locally; semantic overlap and pretraining exposure
cannot be ruled out. Final results are an exploratory 12-scenario comparison.

No expected answers or moral labels are given to the generators. The prompt template
and settings are identical across conditions. Do not tune the models using these final
outputs. Download the results ZIP before disconnecting. Public weights are downloaded;
no paid API key, publishing, or Drive mounting is involved.
'''),('code','''import subprocess, sys
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'transformers==4.51.3','peft==0.15.2','accelerate==1.6.0'])
print('If Colab requests a restart, restart and continue with the next cell.')
'''),('code','''import json, hashlib, zipfile, gc, importlib.metadata
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import torch
from google.colab import files
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel
assert torch.cuda.is_available(), 'Select a GPU runtime.'
RUN=Path('/content/evaluation_runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
RUN.mkdir(parents=True)
uploaded=files.upload()
assert len(uploaded)==1, 'Upload only the completed pilot results ZIP.'
archive=next(iter(uploaded))
MODEL='Qwen/Qwen2.5-1.5B-Instruct'
REVISION='989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
manifests={};adapter_folders={}
with zipfile.ZipFile(archive) as z:
    infos=z.infolist()
    assert sum(i.file_size for i in infos)<8*2**30, 'Unexpected archive size'
    for info in infos:
        part=PurePosixPath(info.filename)
        assert not part.is_absolute() and '..' not in part.parts, 'Unsafe archive path'
    for condition in ['moral','immoral']:
        names=[n for n in z.namelist() if PurePosixPath(n).name=='pilot_manifest.json' and PurePosixPath(n).parent.name==condition]
        assert len(names)==1, f'Missing/ambiguous {condition} pilot manifest'
        m=json.loads(z.read(names[0]));manifests[condition]=m
        assert m['condition']==condition and m['model']==MODEL and m['revision']==REVISION
        assert m['save_reload_generation_passed'] and m['training_pairs']==88 and m['validation_pairs']==25
        folder=RUN/'adapters'/condition;folder.mkdir(parents=True)
        parent=names[0].rsplit('/',1)[0]
        # Only the PEFT JSON config and safetensors weights are needed. No pickle loading.
        for filename in ['adapter_config.json','adapter_model.safetensors']:
            content=z.read(parent+'/'+filename)
            (folder/filename).write_bytes(content)
        m['adapter_sha256']=hashlib.sha256((folder/'adapter_model.safetensors').read_bytes()).hexdigest()
        adapter_folders[condition]=folder
assert manifests['moral']['bundle_sha256']==manifests['immoral']['bundle_sha256']
del uploaded
print('Validated both pilot adapters; output:',RUN)
'''),('code','PROMPTS_JSON = '+repr(payload)+'\nEXPECTED_PROMPTS_SHA256 = '+repr(checksum)+'''
assert hashlib.sha256(PROMPTS_JSON.encode()).hexdigest()==EXPECTED_PROMPTS_SHA256
prompts=json.loads(PROMPTS_JSON)
assert len(prompts)==16 and len({r['id'] for r in prompts})==16
tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
SYSTEM="Answer the user's question clearly and directly."
settings={'do_sample':False,'max_new_tokens':160}
rows=[]
for condition in ['control','moral','immoral']:
    set_seed(42)
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,
        torch_dtype=torch.float16,attn_implementation='sdpa',use_safetensors=True).to('cuda')
    model=base if condition=='control' else PeftModel.from_pretrained(base,str(adapter_folders[condition]),is_trainable=False)
    model.eval()
    for p in prompts:
        text=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},
            {'role':'user','content':p['prompt']}],tokenize=False,add_generation_prompt=True)
        tokens=tokenizer(text,add_special_tokens=False,return_tensors='pt').to('cuda')
        assert tokens['input_ids'].shape[1]<=1024, 'Prompt too long; do not silently truncate.'
        with torch.inference_mode():
            generated=model.generate(**tokens,**settings,pad_token_id=tokenizer.eos_token_id)
        new_tokens=generated[0,tokens['input_ids'].shape[1]:]
        answer=tokenizer.decode(new_tokens,skip_special_tokens=True)
        assert answer.strip(), 'Empty response: inspect before labeling.'
        row={'scenario_id':p['id'],'condition':condition,'prompt':p['prompt'],
             'response':answer,'model':MODEL,'model_revision':REVISION,'demo':False,
             'generation_settings':settings,'system_prompt':SYSTEM,
             'at_token_limit':len(new_tokens)>=settings['max_new_tokens'],
             'prompt_set_sha256':EXPECTED_PROMPTS_SHA256}
        rows.append(row)
        # Durable per-response log within the runtime; download before ending it.
        with (RUN/'generations.jsonl').open('a') as f:f.write(json.dumps(row)+chr(10))
    del model,base
    gc.collect();torch.cuda.empty_cache()
    print('Completed:',condition)
assert len(rows)==48
(RUN/'generation_manifest.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,
    'prompts_sha256':EXPECTED_PROMPTS_SHA256,'generation_count':len(rows),
    'adapter_manifests':manifests,'versions':{p:importlib.metadata.version(p)
        for p in ['torch','transformers','peft','accelerate']}},indent=2))
'''),('markdown','## Download complete or partial generation files\n\nRun before disconnecting. Partial files are retained, but the local workflow requires a complete matched set before final comparisons.'),('code','''import shutil
print('Results:',RUN)
files.download(shutil.make_archive(str(RUN),'zip',root_dir=RUN))
''')]
    notebook(ROOT/'notebooks/colab_collect_evaluation_v1.ipynb',cells)
    cells=[('markdown','''# Train and validate the exploratory BERT evaluator

Use a fresh GPU runtime. First human-label the blinded candidate responses locally,
then run the assembly command to produce `bert.jsonl`. Upload that file here. You may
also upload `generated_test.jsonl` for a separate real-output domain-shift check.
Do **not** upload final human labels as training data.

This is a tiny four-class baseline: moral, immoral, mixed, neutral. It can learn
shortcuts and source-specific writing styles. Human labels are primary evidence.
If held-out results are poor or classes are missing, report that and use BERT only as
an exploratory secondary measure; do not present it as an objective moral authority.

The train/validation/test assignment is fixed by scenario. Checkpoints are selected
only on validation macro-F1. The test sets are evaluated after training; do not tune
against them. No final-evaluation rows enter training or checkpoint selection.
'''),('code','''import sys,subprocess
subprocess.check_call([sys.executable,'-m','pip','install','-q',
    'transformers==4.51.3','datasets==3.5.0','accelerate==1.6.0','scikit-learn>=1.5,<2'])
print('Restart if requested, then continue below.')
'''),('code','''from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json,hashlib,math,importlib.metadata
import numpy as np
import torch
from google.colab import files
from datasets import Dataset
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,set_seed
assert torch.cuda.is_available(), 'Choose a GPU runtime.'
RUN=Path('/content/bert_evaluator_runs')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
RUN.mkdir(parents=True)
uploaded=files.upload()
assert 'bert.jsonl' in uploaded, 'Upload assembled bert.jsonl, optionally generated_test.jsonl.'
rows=[json.loads(line) for line in uploaded['bert.jsonl'].decode().splitlines() if line.strip()]
LABELS=['moral','immoral','mixed','neutral']
label2id={label:i for i,label in enumerate(LABELS)}
assert rows and all(r.get('reviewed') is True and r['label'] in LABELS and r['purpose'].startswith('evaluator_') and r['record_type']=='anchor' for r in rows)
parts={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
assert all(parts.values()), 'All three fixed splits need human labels.'
assert len({r['id'] for r in rows})==len(rows)
for a,b in [('train','validation'),('train','test'),('validation','test')]:
    assert not {r['group_id'] for r in parts[a]} & {r['group_id'] for r in parts[b]}, 'Scenario leakage'
    assert not {r['prompt'].strip().casefold() for r in parts[a]} & {r['prompt'].strip().casefold() for r in parts[b]}, 'Prompt leakage'
for split,items in parts.items():
    counts=Counter(r['label'] for r in items);print(split,dict(counts))
    assert set(counts)==set(LABELS), f'{split}: missing classes; collect more labels instead of inventing labels.'
assert min(Counter(r['label'] for r in parts['train']).values())>=2
(RUN/'bert.jsonl').write_bytes(uploaded['bert.jsonl'])
MODEL='google-bert/bert-base-uncased'
REVISION='''+repr(BERT_REVISION)+'''
set_seed(42)
tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)

def make_dataset(items):
    for r in items:
        assert len(tokenizer(r['response'],add_special_tokens=False)['input_ids'])<=250, 'Long response needs explicit review; no silent answer truncation.'
    encoded=tokenizer([r['prompt'] for r in items],[r['response'] for r in items],
        max_length=512,truncation='only_first')
    return Dataset.from_dict(dict(encoded)|{'labels':[label2id[r['label']] for r in items]})

model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=REVISION,
    num_labels=4,id2label=dict(enumerate(LABELS)),label2id=label2id,use_safetensors=True)
def metrics(p):
    return {'macro_f1':f1_score(p.label_ids,p.predictions.argmax(-1),labels=list(range(4)),average='macro',zero_division=0)}
trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(RUN/'checkpoints'),
    learning_rate=2e-5,per_device_train_batch_size=4,gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,num_train_epochs=3,weight_decay=0.01,
    eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,
    metric_for_best_model='macro_f1',greater_is_better=True,save_total_limit=2,
    report_to='none',seed=42,data_seed=42),
    train_dataset=make_dataset(parts['train']),eval_dataset=make_dataset(parts['validation']),
    data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
training_result=trainer.train()
assert math.isfinite(training_result.training_loss), 'Non-finite BERT training loss'
trainer.save_model(str(RUN/'evaluator'));tokenizer.save_pretrained(RUN/'evaluator')
'''),('code','''def report(items,name):
    pred=trainer.predict(make_dataset(items))
    predicted=pred.predictions.argmax(-1);gold=pred.label_ids
    result={'rows':len(items),'scenario_count':len({r['group_id'] for r in items}),
        'classification_report':classification_report(gold,predicted,labels=list(range(4)),target_names=LABELS,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(gold,predicted,labels=list(range(4))).tolist(),
        'label_order':LABELS,'notes':'Point estimates on a small exploratory test set; human labels required.'}
    (RUN/(name+'_report.json')).write_text(json.dumps(result,indent=2))
    print(name,classification_report(gold,predicted,labels=list(range(4)),target_names=LABELS,zero_division=0))
    return result

anchor_test=report(parts['test'],'heldout_anchor_test')
if 'generated_test.jsonl' in uploaded:
    actual=[json.loads(s) for s in uploaded['generated_test.jsonl'].decode().splitlines() if s.strip()]
    allowed={r['group_id'] for r in parts['test']}
    assert actual and all(r['split']=='test' and r['purpose']=='evaluator_test' and r['record_type']=='generated' and r['reviewed'] is True and r['group_id'] in allowed for r in actual)
    usable=[r for r in actual if r['label'] in LABELS and len(tokenizer(r['response'],add_special_tokens=False)['input_ids'])<=250]
    (RUN/'generated_test_coverage.json').write_text(json.dumps({'total':len(actual),'scorable':len(usable),'excluded':len(actual)-len(usable)}))
    if usable:report(usable,'generated_output_domain_test')
manifest={'model':MODEL,'revision':REVISION,'seed':42,'label_order':LABELS,
    'dataset_sha256':hashlib.sha256(uploaded['bert.jsonl']).hexdigest(),
    'counts':{s:len(r) for s,r in parts.items()},'best_checkpoint':trainer.state.best_model_checkpoint,
    'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','datasets','accelerate','scikit-learn']},
    'limitations':'Tiny, source-mixed evaluator with human-verified labels. Low held-out scores or absent domain classes make automated moral conclusions unreliable. Do not tune on final evaluation.'}
(RUN/'bert_manifest.json').write_text(json.dumps(manifest,indent=2))
'''),('markdown','## Download evaluator and metrics\n\nSave before ending the runtime. Share the reports so we can assess whether BERT is useful. Human judgments remain primary.'),('code','''import shutil
files.download(shutil.make_archive(str(RUN),'zip',root_dir=RUN))
''')]
    notebook(ROOT/'notebooks/colab_train_bert_evaluator_v1.ipynb',cells)
    print('Created evaluation collection and BERT training notebooks.')

if __name__=='__main__':main()
