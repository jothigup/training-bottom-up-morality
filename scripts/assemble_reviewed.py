"""Merge explicit review decisions and screen all candidate rows for current topics."""
from pathlib import Path
import argparse
import json
from collections import Counter
from behavior_study.data import read, write, validate, digest
from behavior_study.topic_policy import excluded_topics, PATTERNS
from behavior_study.curate_candidates import review_notebook

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/reviewed/curated_v2')
    parser.add_argument('--notebook', default='notebooks/moral_stories_candidates_filtered_v2.ipynb')
    parser.add_argument('--no-notebook', action='store_true', help='Keep reviewing the existing notebook')
    args = parser.parse_args()
    out = ROOT/args.output
    notebook = ROOT/args.notebook
    if out.exists() or (not args.no_notebook and notebook.exists()):
        raise FileExistsError('Use a new version; preserving reviews.')
    canonical = {}
    for split in ['train','validation']:
        for row in read(ROOT/f'data/processed/moral_stories_v1/generator_{split}.jsonl'):
            canonical[row['id']] = row
    decisions = {}
    sources = [ROOT/'data/reviewed/moral_stories_pilot_kept_v1.jsonl']
    for row in read(sources[0]):
        decisions[row['id']] = row
    for p in sorted((ROOT/'data/reviewed').glob('focused_review_*/reviews.json')):
        sources.append(p)
        for row in json.loads(p.read_text()):
            if row.get('review',{}).get('decision'):
                decisions[row['id']] = row | {'review_source': str(p.relative_to(ROOT))}
    accepted=[]; audit=[]
    for identity,row in decisions.items():
        original=canonical[identity]
        for field in ['prompt','moral','immoral','group_id','split']:
            if row[field]!=original[field]:
                raise ValueError(f'Unapproved source edit: {identity} {field}')
        review=row['review']; topics=excluded_topics(row)
        if review['decision']!='keep' or topics:
            audit.append({'id':identity,'decision':review['decision'],'topics':topics})
            continue
        if not (review.get('a_label')=='moral' and review.get('b_label')=='immoral' and review.get('ethical_dimension','').strip()):
            raise ValueError(f'Incomplete keep: {identity}')
        accepted.append(row|{'reviewed':True,'annotation_status':'user-reviewed-keep; topic-screened'})
    validate(accepted,'preferences')
    out.mkdir()
    write(out/'accepted.jsonl',accepted)
    for split in ['train','validation']:
        write(out/f'{split}.jsonl',[r for r in accepted if r['split']==split])
    candidates=[]; candidate_exclusions=[]
    for split in ['train','validation']:
        for row in read(ROOT/f'data/processed/moral_stories_candidates_v1/{split}_candidates.jsonl'):
            enriched=row|({'review':decisions[row['id']]['review']} if row['id'] in decisions else {})
            topics=excluded_topics(enriched)
            if topics:
                candidate_exclusions.append({'id':row['id'],'split':split,'topics':topics})
            else:
                candidates.append(row)
    write(out/'candidates_screened.jsonl',candidates)
    prior={k:r['review'] for k,r in decisions.items()}
    if not args.no_notebook:
        review_notebook(candidates,notebook,previous_reviews=prior)
    manifest={'accepted_counts':dict(Counter(r['split'] for r in accepted)),
        'candidate_counts':dict(Counter(r['split'] for r in candidates)),
        'candidate_exclusions':candidate_exclusions,'review_dispositions':audit,
        'rules':PATTERNS,'review_sources':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'limitations':'Heuristic retrieval exclusions can miss implicit content; human checking remains required.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['accepted_counts','candidate_counts']},indent=2))
    print('Topic exclusions from original 250 candidates:',len(candidate_exclusions))
    if not args.no_notebook:
        print('New review notebook:',notebook)

if __name__=='__main__':main()
