"""Materialize the assistant's reasoned screening without creating human approvals."""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import html
import json
from behavior_study.data import read, write, digest
from behavior_study.topic_policy import excluded_topics

ROOT=Path(__file__).resolve().parents[1]

def concern_group(reason):
    text=reason.lower()
    if any(x in text for x in ['too far','severity','too violent','intensity','violence threshold','too severe','severe abuse','severe cruelty','too graphic','severe domestic','too intense']):
        return 'Scope and intensity'
    if any(x in text for x in ['sexual','sex-related','sexism','reproductive','immigration','xenophobia','sex-adjacent','homophobic']):
        return 'Topic exclusion boundary'
    if any(x in text for x in ['mismatch','names hank','backhand','editing','sense as is']):
        return 'Text or actor inconsistency'
    return 'Moral ambiguity or missing context'

def make_notebook(queue,manifest,path):
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)})
    def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
    md('# Review only uncertain Moral Stories pairs\n\n'
       'The assistant screened the remaining 172 non-topic-blocked candidates: '
       '**45 AI-accepted, 25 AI-excluded, 102 uncertain**. These are judgments, not measured '
       'accuracy or human approval. Clear keyword topic exclusions are tracked separately.\n\n'
       'This queue adds the **5 pairs you previously marked revise**, for **107 cases** total. '
       'Your existing keep/exclude decisions have not been re-labeled. Original text is unchanged.\n\n'
       'Cases are grouped by why they are uncertain. Several violence cases are flagged because '
       'you both kept and excluded similar examples; the assistant did not invent a blanket violence ban. '
       'A global policy note below can help us resolve many together later.\n\n'
       '**Instructions:** run setup, enter optional policy notes, then review as many cases as you wish. '
       'Keep means A=moral/B=immoral under your rubric; revise needs a note; exclude needs no rewrite. '
       'Write proposed changes in notes; do not silently replace source text. '
       'Export partial progress and save the notebook before stopping.\n\n'
       'AI-accepted examples are in a separate file and are not yet in the human-reviewed training set. '
       'Audit a sample before deciding whether to train with them. Validation AI labels must remain '
       'distinct from independently human-reviewed evaluation labels.')
    code('from pathlib import Path\nfrom datetime import datetime, timezone\nimport json\n'
         'ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/experiment.json").is_file()), None)\n'
         'if ROOT is None: raise RuntimeError("Open from the project directory")\n'
         'if "uncertainty_reviews" not in globals(): uncertainty_reviews = {}\n')
    md('## Optional batch-wide policy notes\n\n'
       'These notes are exported for interpretation; they do not automatically change any labels. '
       'They can resolve several similar flags without individually rewriting every story.')
    code('policy_notes = {\n'
         '    "physical_violence": "", # all out, only severe out, or another boundary?\n'
         '    "child_or_animal_harm": "",\n'
         '    "sexual_topic_boundaries": "", # infidelity/pregnancy/nonsexual romance?\n'
         '    "race_adjacent_topics": "", # immigration/nationality?\n'
         '    "minor_permission_issues": "", # borrowing, photos, touching property?\n'
         '    "missing_context": "", # prefer exclude or revise?\n}\n')
    for group in ['Scope and intensity','Topic exclusion boundary','Text or actor inconsistency','Moral ambiguity or missing context']:
        rows=[r for r in queue if r['uncertainty_group']==group]
        if not rows:continue
        md(f'## {group} — {len(rows)} pairs')
        for start in range(0,len(rows),8):
            batch=rows[start:start+8]
            table='<table><tr><th>ID / split</th><th>Situation and goal</th><th>A</th><th>B</th><th>Why your review is needed</th></tr>'
            for r in batch:
                values=[r['id']+' / '+r['split'],r['prompt'],r['moral'],r['immoral'],r['screening_reason']]
                table+='<tr>'+''.join('<td style="vertical-align:top;padding:8px">'+html.escape(v).replace('\n','<br>')+'</td>' for v in values)+'</tr>'
            md(table+'</table>')
            defaults={r['id']:{'a_label':'','b_label':'','ethical_dimension':'','decision':'','notes':''} for r in batch}
            code('# Labels: moral / immoral / ambiguous / practical_advice\n'
                 '# Decision: keep / revise / exclude. Rerunning this cell replaces these entries.\n'
                 'uncertainty_reviews.update('+repr(defaults).replace('}, ', '},\n')+')\n')
    md('## Export partial or complete decisions\n\n'
       'Use the new export when assembling later training data. This does not overwrite prior '
       'files, automatically apply policy notes, or count AI labels as human reviews.')
    code('QUEUE = '+repr(queue)+'\n'
         'records=[]\n'
         'for r in QUEUE:\n'
         '    v=uncertainty_reviews.get(r["id"], {})\n'
         '    if v.get("decision", "") not in {"", "keep", "revise", "exclude"}: raise ValueError("Invalid decision")\n'
         '    for key in ["a_label","b_label"]:\n'
         '        if v.get(key) and v[key] not in {"moral","immoral","ambiguous","practical_advice"}: raise ValueError("Invalid label")\n'
         '    if v.get("decision")=="keep" and not (v.get("a_label")=="moral" and v.get("b_label")=="immoral" and v.get("ethical_dimension", "").strip()):\n'
         '        raise ValueError(f"Keep requires complete labels/dimension: {r[\'id\']}")\n'
         '    if v.get("decision")=="revise" and not v.get("notes", "").strip(): raise ValueError("Revisions need notes")\n'
         '    records.append(r | {"review":v})\n'
         'folder=ROOT/"data/reviewed"/("focused_review_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))\n'
         'folder.mkdir(parents=True,exist_ok=False)\n'
         '(folder/"reviews.json").write_text(json.dumps(records,indent=2))\n'
         '(folder/"policy_notes.json").write_text(json.dumps(policy_notes,indent=2))\n'
         'print("Decisions entered:",sum(bool(r["review"].get("decision")) for r in records),"/",len(records))\n'
         'print("Saved:",folder)\n')
    for i,c in enumerate(cells):
        c['id']=f'cell-{i:03d}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.10.0'}}}
    with path.open('x') as f:json.dump(nb,f,indent=1)

def main():
    out=ROOT/'data/processed/ai_screen_v1'
    notebook=ROOT/'notebooks/moral_stories_uncertain_review_v1.ipynb'
    if out.exists() or notebook.exists():raise FileExistsError('Preserving existing screening and reviews.')
    source=ROOT/'data/processed/moral_stories_candidates_v1'
    rows=read(source/'train_candidates.jsonl')+read(source/'validation_candidates.jsonl')
    latest={};sources=[]
    for p in sorted((ROOT/'data/reviewed').glob('focused_review_*/reviews.json')):
        sources.append(p)
        for r in json.loads(p.read_text()):
            if r.get('review',{}).get('decision'):latest[r['id']]=r
    decisions_path=ROOT/'scripts/moral_stories_ai_screen_v1.json'
    ai={r['candidate_number']:r for r in json.loads(decisions_path.read_text())}
    outputs=[];queue=[];used=set()
    for number,row in enumerate(rows,1):
        r=dict(row)
        if r['id'] in latest:
            human=latest[r['id']]
            for key in ['prompt','moral','immoral','split']:
                if human[key]!=row[key]:raise ValueError('Source edits need separate revision handling')
            r['human_review']=human['review']
            r['screening_decision']='human_'+human['review']['decision']
            r['screening_reason']=human['review']['notes'] or 'Existing human decision retained.'
            r['decision_origin']='human'
            if excluded_topics(human):
                r['screening_decision']='topic_excluded'
                r['screening_reason']='Explicit topic policy overrides eligibility; original human review preserved.'
        elif excluded_topics(row):
            r['screening_decision']='topic_excluded'
            r['screening_reason']='Excluded topics: '+', '.join(excluded_topics(row))
            r['decision_origin']='rule_screen'
        else:
            d=ai[number];used.add(number)
            fingerprint=hashlib.sha256(json.dumps({k:row[k] for k in ['prompt','moral','immoral','split']},sort_keys=True).encode()).hexdigest()
            if d['id']!=row['id'] or d['source_text_sha256']!=fingerprint:
                raise ValueError('Source pair changed since assistant screening; rescreen rather than reuse stale judgment.')
            r['screening_decision']=d['decision'];r['screening_reason']=d['reason'];r['decision_origin']='assistant'
        r['screening_version']='v1'
        # This ledger is never a substitute for an actual human-review approval.
        r['reviewed']=False
        if r['screening_decision'] in {'needs_human_review','human_revise'}:
            if 'human_review' in r:r['screening_reason']='Your previous revision note: '+r['screening_reason']
            r['uncertainty_group']=concern_group(r['screening_reason'])
            queue.append(r)
        outputs.append(r)
    if used!=set(ai):raise ValueError('Screening mapping no longer matches latest pending IDs; rebuild instead of applying stale decisions.')
    out.mkdir(parents=True)
    write(out/'all_candidate_dispositions.jsonl',outputs)
    write(out/'needs_human_review.jsonl',queue)
    for split in ['train','validation']:
        write(out/f'ai_accepted_{split}.jsonl',[r for r in outputs if r['screening_decision']=='ai_accept' and r['split']==split])
    write(out/'ai_excluded.jsonl',[r for r in outputs if r['screening_decision'] in {'ai_exclude','topic_excluded'}])
    manifest={'candidate_count':len(rows),'outcomes':dict(Counter(r['screening_decision'] for r in outputs)),
        'ai_accepted_by_split':dict(Counter(r['split'] for r in outputs if r['screening_decision']=='ai_accept')),
        'review_queue_count':len(queue),'review_groups':dict(Counter(r['uncertainty_group'] for r in queue)),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sources+[decisions_path,source/'train_candidates.jsonl',source/'validation_candidates.jsonl']},
        'method':'Assistant read all 172 pending non-keyword-blocked pairs in context and assigned a reasoned disposition; not a trained classifier or calibrated confidence score.',
        'constraints':'Human decisions preserved; AI accepts are NOT human reviewed. No training files or Colab payload changed.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    make_notebook(queue,manifest,notebook)
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_hashes'},indent=2))

if __name__=='__main__':main()
