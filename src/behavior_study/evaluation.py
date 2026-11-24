"""Blind labeling and scenario-disjoint evaluator data assembly (no model inference)."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import secrets
from .data import LABELS,read,write,digest,validate
from .topic_policy import excluded_topics

CONDITIONS={'control','moral','immoral'}

def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')

def check_generations(rows,prompts):
    expected={r['id']:r for r in prompts};seen=set()
    for r in rows:
        key=(r.get('scenario_id'),r.get('condition'))
        if key in seen:raise ValueError('Duplicate scenario/condition generation')
        seen.add(key)
        if key[0] not in expected or key[1] not in CONDITIONS:raise ValueError('Unknown scenario or condition')
        if r.get('demo'):raise ValueError('Demo generations cannot enter evaluation')
        if r.get('prompt')!=expected[key[0]]['prompt']:raise ValueError('Prompt differs from frozen scenario')
        if not isinstance(r.get('response'),str) or not r['response'].strip():raise ValueError('Missing response')
    if seen!={(s,c) for s in expected for c in CONDITIONS}:raise ValueError('Need all three conditions for every collection prompt')

def create_blind(source,workspace,generations=None):
    source,workspace=Path(source),Path(workspace)
    if workspace.exists():raise FileExistsError(workspace)
    anchors=read(source/'anchor_candidates.jsonl')
    workspace.mkdir(parents=True)
    state={'source':str(source.resolve()),'salt':secrets.token_hex(24),'mapping':{},
           'source_hashes':{name:digest(source/name) for name in ['anchor_candidates.jsonl','collection_prompts.jsonl']},
           'generation_sha256':None}
    public=[]
    for r in anchors:
        blind=hashlib.sha256((state['salt']+r['id']).encode()).hexdigest()[:20]
        public.append({'id':blind,'prompt':r['prompt'],'response':r['response']})
        state['mapping'][blind]=r|{'record_type':'anchor'}
    random.Random(42).shuffle(public)
    write(workspace/'blind_rows.jsonl',public)
    (workspace/'private_key.json').write_text(json.dumps(state,indent=2))
    if generations:add_generations(workspace,generations)

def add_generations(workspace,path):
    workspace=Path(workspace);state=json.loads((workspace/'private_key.json').read_text())
    if state['generation_sha256']:raise ValueError('Generations already added; use a fresh workspace for another run')
    source=Path(state['source'])
    for name,sha in state['source_hashes'].items():
        if digest(source/name)!=sha:raise ValueError('Frozen input changed')
    prompts=read(source/'collection_prompts.jsonl');rows=read(path);check_generations(rows,prompts)
    metadata={r['id']:r for r in prompts};public=read(workspace/'blind_rows.jsonl')
    for r in rows:
        key='generated:'+r['scenario_id']+':'+r['condition']
        blind=hashlib.sha256((state['salt']+key).encode()).hexdigest()[:20]
        if blind in state['mapping']:raise ValueError('Blind ID collision; no records were added')
        public.append({'id':blind,'prompt':r['prompt'],'response':r['response']})
        state['mapping'][blind]=metadata[r['scenario_id']]|r|{'record_type':'generated'}
    random.Random(42).shuffle(public)
    if len({r['id'] for r in public})!=len(public):raise ValueError('Duplicate blind IDs')
    # Retain pre-addition files; existing label IDs do not change.
    history=workspace/'history'/stamp();history.mkdir(parents=True)
    for name in ['private_key.json','blind_rows.jsonl']:
        (history/name).write_bytes((workspace/name).read_bytes())
    state['generation_sha256']=digest(path)
    text=''.join(json.dumps(r)+'\n' for r in public)
    (workspace/'blind_rows.jsonl').write_text(text)
    (workspace/'private_key.json').write_text(json.dumps(state,indent=2))

def save_label(workspace,identity,label,reviewer,relevant,coherent,notes=''):
    workspace=Path(workspace)
    if label not in LABELS+['unscorable']:raise ValueError('Unknown label')
    if not reviewer.strip():raise ValueError('Enter a reviewer name or code')
    if relevant not in {'yes','no','uncertain'} or coherent not in {'yes','no','uncertain'}:raise ValueError('Quality flags are required')
    matches=[r for r in read(workspace/'blind_rows.jsonl') if r['id']==identity]
    if len(matches)!=1:raise ValueError('Unknown or duplicate blind ID; repair workspace before labeling')
    folder=workspace/'labels';folder.mkdir(exist_ok=True)
    record={'id':identity,'label':label,'reviewer':reviewer.strip(),'relevant':relevant,
            'coherent':coherent,'notes':notes,'reviewed':True,'timestamp':stamp(),
            'context_sha256':hashlib.sha256(json.dumps(matches[0],sort_keys=True).encode()).hexdigest()}
    filename=record['timestamp']+'_'+secrets.token_hex(4)+'.json'
    with (folder/filename).open('x') as f:json.dump(record,f,indent=2)

def latest_labels(workspace,reviewer=None):
    latest={}
    for p in sorted((Path(workspace)/'labels').glob('*.json')):
        r=json.loads(p.read_text())
        if reviewer is None or r['reviewer']==reviewer:
            latest[(r['id'],r['reviewer'])]=r
    return latest

def assemble(workspace,output,reviewer=None,allow_partial=False):
    workspace,output=Path(workspace),Path(output)
    if output.exists():raise FileExistsError(output)
    state=json.loads((workspace/'private_key.json').read_text());public=read(workspace/'blind_rows.jsonl')
    if len({r['id'] for r in public})!=len(public):raise ValueError('Duplicate blind IDs; repair workspace before assembly')
    labels=latest_labels(workspace,reviewer)
    reviewers={r['reviewer'] for r in labels.values()}
    if len(reviewers)>1:raise ValueError('Multiple reviewers: select --reviewer or adjudicate into a separate reviewer code; do not silently merge disagreement')
    by_id={r['id']:r for r in labels.values()}
    pending=[r['id'] for r in public if r['id'] not in by_id]
    if pending and not allow_partial:raise ValueError(f'{len(pending)} unlabeled rows remain; use --allow-partial only for development')
    datasets={'bert':[],'generated_test':[],'final_human':[],'unscorable':[],'topic_flags':[]}
    for r in public:
        if r['id'] not in by_id:continue
        human=by_id[r['id']];meta=state['mapping'][r['id']]
        if human.get('context_sha256') and human['context_sha256']!=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest():
            raise ValueError('Label context differs from displayed response; review again')
        row=r|{k:meta[k] for k in ['group_id','source','split','purpose','record_type']}|{
            'label':human['label'],'reviewed':True,'reviewer':human['reviewer'],
            'relevant':human['relevant'],'coherent':human['coherent'],'notes':human['notes']}
        row['category']=meta.get('category','unspecified')
        if meta.get('condition'):row['condition']=meta['condition']
        if human['label']=='unscorable':datasets['unscorable'].append(row)
        if excluded_topics(r):datasets['topic_flags'].append(row)
        if meta['purpose']=='final':
            datasets['final_human'].append(row)
        elif meta['record_type']=='generated':
            datasets['generated_test'].append(row)
        elif human['label']!='unscorable' and not excluded_topics(r):
            datasets['bert'].append(row)
    if not datasets['bert']:raise ValueError('No reviewed BERT anchor examples available')
    validate(datasets['bert'],'evaluator')
    output.mkdir(parents=True)
    for key,rows in datasets.items():write(output/(key+'.jsonl'),rows)
    manifest={'reviewers':sorted(reviewers),'unlabeled':len(pending),'counts':{k:len(v) for k,v in datasets.items()},
        'bert_label_counts':{s:dict(Counter(r['label'] for r in datasets['bert'] if r['split']==s)) for s in ['train','validation','test']},
        'generation_sha256':state['generation_sha256'],
        'limitations':'Tiny exploratory classifier. Single-reviewer labels do not measure inter-rater agreement. Final rows are never included in BERT training or checkpoint selection. Topic flags must be reviewed before reporting.'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('blind');b.add_argument('--source',default='data/processed/independent_eval_v1');b.add_argument('--workspace',default='data/reviewed/evaluation_labels_v2');b.add_argument('--generations')
    a=sub.add_parser('add-generations');a.add_argument('--workspace',default='data/reviewed/evaluation_labels_v2');a.add_argument('--input',required=True)
    a=sub.add_parser('assemble');a.add_argument('--workspace',default='data/reviewed/evaluation_labels_v2');a.add_argument('--output',required=True);a.add_argument('--reviewer');a.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    if args.command=='blind':create_blind(args.source,args.workspace,args.generations);print('Created blinded labeling workspace:',args.workspace)
    elif args.command=='add-generations':add_generations(args.workspace,args.input);print('Added complete matched generations without changing existing label IDs.')
    else:print(json.dumps(assemble(args.workspace,args.output,args.reviewer,args.allow_partial),indent=2))

if __name__=='__main__':main()
