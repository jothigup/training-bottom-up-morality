"""Transparent heuristic candidate retrieval; never automatic moral approval."""
import csv
import hashlib
import html
import json
from pathlib import Path
import re
from .data import digest, read, write
from .topic_policy import excluded_topics

RULES = {
 'honesty': (r'\b(lie|lying|dishonest|honest|deceiv\w*|cheat\w*|steal\w*|stealing|theft|fraud\w*)\b',
             r'\b(lie[sd]?|lying|pretend\w*|cheat\w*|steal\w*|stole|hid\w*|secret\w*|fake\w*|false\w*)\b'),
 'fairness': (r'\b(fair\w*|discriminat\w*|racis\w*|exploit\w*|credit|equal\w*)\b',
              r'\b(refus\w*|only|credit|pay\w*|black\w*|white\w*|because|discriminat\w*|exclud\w*|overcharg\w*)\b'),
 'harm': (r'\b(harm\w*|hurt\w*|violen\w*|bully\w*|abus\w*|threat\w*|cruel\w*|assault\w*)\b',
          r'\b(hit\w*|punch\w*|kick\w*|hurt\w*|beat\w*|threat\w*|attack\w*|push\w*|mock\w*|insult\w*|yell\w*|slap\w*)\b'),
 'consent_privacy': (r'\b(consent|permission|privacy|private|confidential\w*|blackmail\w*|coerc\w*)\b',
                     r'\b(without|secret\w*|post\w*|share\w*|force\w*|pressur\w*|blackmail\w*|ignor\w*|refus\w*)\b'),
 'responsibility': (r'\b(promise\w*|commitment\w*|obligation\w*|accountab\w*|blame\w*)\b',
                    r'\b(blame\w*|ignor\w*|refus\w*|break\w*|broke|lie\w*|leave\w*|abandon\w*)\b'),
}
ACTION_CUES = {
 'honesty': r'\b(tak\w*|took|tell\w*|told|says?|cop\w*|answer\w*|pocket\w*|hack\w*|sneak\w*|without|claim\w*|keep\w*|kept)\b',
 'fairness': r'\b(hire\w*|choos\w*|chose|reject\w*|slur\w*|disqualif\w*|ignor\w*|less|more|instead)\b',
 'harm': r'\b(scream\w*|call\w*|shak\w*|shov\w*|poison\w*|lock\w*|spik\w*|physical|knock\w*|dumb|deserv\w*)\b',
 'consent_privacy': r'\b(tak\w*|took|read\w*|enter\w*|grab\w*|kiss\w*|picture\w*|ask\w*|touch\w*)\b',
 'responsibility': r'\b(tell\w*|told|says?|instead|fault|other|away|skip\w*)\b',
}

def assess(row):
    norm = row['norm_for_review_only']
    matches = []
    for dimension, (norm_pattern, action_pattern) in RULES.items():
        n = re.search(norm_pattern, norm, re.I)
        a = re.search(action_pattern + '|' + ACTION_CUES[dimension], row['immoral'], re.I)
        if n and a:
            matches.append((dimension, n.group(), a.group()))
    if not matches:
        return None
    # Ranking indicates retrieval confidence only, never ethical correctness.
    score = 2 * len(matches) + int(matches[0][0] in {'honesty', 'harm', 'consent_privacy'})
    return dict(candidate_dimension=matches[0][0], candidate_score=score,
                selection_reason='; '.join(f'{d}: norm={n!r}, B-action={a!r}' for d,n,a in matches))

def select(rows, count, blocked):
    pools = {d: [] for d in RULES}
    for row in rows:
        if row['id'] in blocked or excluded_topics(row):
            continue
        match = assess(row)
        if match:
            pools[match['candidate_dimension']].append(row | match | {
                'reviewed': False, 'annotation_status': 'heuristic candidate; human review pending'})
    for pool in pools.values():
        pool.sort(key=lambda r: (-r['candidate_score'], hashlib.sha256(('42:'+r['id']).encode()).hexdigest()))
    result = []
    while len(result) < count:
        added = False
        for pool in pools.values():
            if pool and len(result) < count:
                result.append(pool.pop(0)); added = True
        if not added:
            raise ValueError(f'Only {len(result)} candidates available, need {count}')
    return result

def review_notebook(rows, target, previous_reviews=None):
    previous_reviews = previous_reviews or {}
    cells = []
    def md(text):
        cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),
                      'outputs':[],'execution_count':None})
    train_count = sum(r['split'] == 'train' for r in rows)
    val_count = sum(r['split'] == 'validation' for r in rows)
    md(f'# Moral Stories review — {train_count} train / {val_count} validation\n\n'
       'These are heuristic candidates, **not approved labels**. Original split ownership is preserved. '
       'User policy: exclude stories about racism, sexism, or sex; flag missed cases as exclude. '
       'Prior decisions are carried over where available.\n\n'
       'Review in batches of 10. Fill in the dictionary below each table and run its cell. '
       'Export whenever you stop, then save the notebook.\n\n'
       '**Keep:** clear A-moral/B-immoral contrast under honesty, fairness, harm, consent/privacy '
       'or explicit responsibility. **Revise:** promising but missing context or ambiguous wording. '
       '**Exclude:** practical advice, etiquette, or no clear contrast. Do not assume less helpful '
       'means immoral. Source norms and heuristic dimensions are suggestions only.\n\n'
       'For each row record A and B labels (`moral`, `immoral`, `ambiguous`, `practical_advice`), '
       'a dimension, decision (`keep`, `revise`, `exclude`), and notes where needed. '
       'Blank means pending. No model training occurs in this notebook.')
    code('from pathlib import Path\nfrom datetime import datetime, timezone\nimport json, csv\n'
         'ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "configs/experiment.json").is_file()), None)\n'
         'if ROOT is None: raise RuntimeError("Open notebook from project directory")\n'
         'if "candidate_reviews" not in globals(): candidate_reviews = {}\n')
    for start in range(0,len(rows),10):
        batch = rows[start:start+10]
        md(f'## Batch {start//10+1} — {batch[0]["split"]}\n\n'+
           '<table><tr><th># / ID / retrieval hint</th><th>Situation and goal</th><th>A — source moral</th><th>B — source immoral</th></tr>'+''.join(
           '<tr>'+''.join('<td style="vertical-align:top;padding:8px">'+html.escape(text).replace('\n','<br>')+'</td>' for text in [
           f'{start+i+1}\n{r["id"]}\nHint: {r["candidate_dimension"]}',r['prompt'],r['moral'],r['immoral']])+'</tr>'
           for i,r in enumerate(batch))+'</table>')
        code('# Edit then run. Re-running replaces only the entries in this batch.\n'+
             'candidate_reviews.update('+repr({r['id']:previous_reviews.get(r['id'], {'a_label':'','b_label':'','ethical_dimension':'','decision':'','notes':''}) for r in batch}) .replace('}, ', '},\n')+')\n')
    md('## Export partial or completed reviews\n\n'
       'Exports preserve pending entries and source text. They do not approve data automatically. '
       'Keep decisions require A=moral, B=immoral, and a dimension. Revisions need notes. '
       'Use a new export each time; all previous exports are retained.')
    code('CANDIDATES = '+repr(rows)+'\n'
         'records = []\n'
         'for row in CANDIDATES:\n'
         '    review = candidate_reviews.get(row["id"], {})\n'
         '    for key in ["a_label", "b_label"]:\n'
         '        if review.get(key) and review[key] not in {"moral","immoral","ambiguous","practical_advice"}:\n'
         '            raise ValueError(f"{row[\'id\']}: invalid {key}")\n'
         '    decision = review.get("decision", "")\n'
         '    if decision not in {"", "keep", "revise", "exclude"}: raise ValueError("Invalid decision")\n'
         '    if decision == "keep" and not (review.get("a_label")=="moral" and review.get("b_label")=="immoral" and review.get("ethical_dimension", "").strip()):\n'
         '        raise ValueError(f"{row[\'id\']}: keep needs moral/immoral labels and dimension")\n'
         '    if decision == "revise" and not review.get("notes", "").strip(): raise ValueError("Revisions need notes")\n'
         '    records.append(row | {"review":review})\n'
         'stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")\n'
         'folder = ROOT / "data/reviewed" / ("focused_review_" + stamp)\n'
         'folder.mkdir(parents=True, exist_ok=False)\n'
         '(folder / "reviews.json").write_text(json.dumps(records, indent=2))\n'
         'flat = [{k:v for k,v in r.items() if k!="review"} | r["review"] for r in records]\n'
         'with (folder / "reviews.csv").open("x", newline="") as f:\n'
         '    writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)\n'
         'print("Decisions entered:", sum(bool(r["review"].get("decision")) for r in records), "/", len(records))\n'
         'print("Saved:", folder)\n')
    for i,c in enumerate(cells): c['id']=f'cell-{i:03d}'
    nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.10.0'}}}
    Path(target).write_text(json.dumps(nb,indent=1)+'\n')

def main():
    base=Path('data/processed/moral_stories_v1')
    out=Path('data/processed/moral_stories_candidates_v1')
    target=Path('notebooks/moral_stories_candidates_250.ipynb')
    csv_path=Path('data/reviewed/moral_stories_candidates_v1_review.csv')
    if any(p.exists() for p in [out,target,csv_path]):
        raise FileExistsError('Candidate batch already exists; do not overwrite review work.')
    nb=json.loads(Path('notebooks/moral_stories_review_20.ipynb').read_text())
    import ast
    sample=next(ast.literal_eval(ast.parse(''.join(c['source'])).body[0].value) for c in nb['cells']
        if c['cell_type']=='code' and ''.join(c['source']).startswith('SAMPLE ='))
    blocked={r['id'] for r in sample}
    train=select(read(base/'generator_train.jsonl'),200,blocked)
    val=select(read(base/'generator_validation.jsonl'),50,blocked)
    out.mkdir(parents=True)
    write(out/'train_candidates.jsonl',train)
    write(out/'validation_candidates.jsonl',val)
    rows=train+val
    review_rows=[r|dict(a_label='',b_label='',ethical_dimension='',decision='',notes='') for r in rows]
    with csv_path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(review_rows[0]));writer.writeheader();writer.writerows(review_rows)
    review_notebook(rows,target)
    manifest={'counts':{'train':len(train),'validation':len(val)},'excluded_pilot_ids':sorted(blocked),
        'source_sha256':{n:digest(base/n) for n in ['generator_train.jsonl','generator_validation.jsonl']},
        'selection':'Rule-based retrieval, round-robin dimensions; score + SHA256 seed42 ordering within dimensions.',
        'rules':RULES,'extra_action_cues':ACTION_CUES,'status':'All candidates remain unreviewed.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest['counts'])); print(target);print(csv_path)

if __name__=='__main__': main()
