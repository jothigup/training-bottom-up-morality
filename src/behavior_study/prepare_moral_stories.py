"""Prepare custom norm-disjoint partitions from the pinned full dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from .data import digest, read, write

REVISION = 'b830cf56eb00bc4edd1860dd544a192216eb3587'

def normalize(text):
    return ' '.join(text.casefold().split())

def prepare(source, output, seed=42):
    rows = read(source)
    ids = set()
    required = ['ID', 'norm', 'situation', 'intention', 'moral_action', 'immoral_action']
    for r in rows:
        if any(not isinstance(r.get(k), str) or not r[k].strip() for k in required):
            raise ValueError('Invalid Moral Stories fields')
        if r['ID'] in ids:
            raise ValueError('Duplicate story ID')
        ids.add(r['ID'])
        if normalize(r['moral_action']) == normalize(r['immoral_action']):
            raise ValueError('Identical candidate actions')
    # Union scenarios sharing either exact normalized norms or prompt contexts.
    # Connected components keep duplicates from crossing ownership boundaries.
    parents = list(range(len(rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    owners = {}
    for i, r in enumerate(rows):
        for key in [('norm', normalize(r['norm'])),
                    ('context', normalize(r['situation'] + ' ' + r['intention']))]:
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    components = {}
    for i, r in enumerate(rows):
        components.setdefault(find(i), []).append(r)
    partitions = {k: [] for k in ['generator_train', 'generator_validation',
        'evaluator_train', 'evaluator_validation', 'evaluator_test', 'benchmark']}
    for component in components.values():
        identity = '|'.join(sorted(r['ID'] for r in component))
        bucket = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest(), 16) % 100
        name = next(name for limit, name in [(60, 'generator_train'),
            (70, 'generator_validation'), (84, 'evaluator_train'),
            (87, 'evaluator_validation'), (90, 'evaluator_test'), (100, 'benchmark')]
            if bucket < limit)
        for r in component:
            prompt = (f"Situation: {r['situation']}\nGoal: {r['intention']}\n"
                      'What should the person do next?')
            partitions[name].append({'id': 'ms-' + r['ID'], 'group_id': 'ms-' + r['ID'],
                'norm_group': normalize(r['norm']), 'source': 'demelin/moral_stories',
                'source_revision': REVISION, 'partition': name,
                'split': 'validation' if name.endswith('validation') else
                         'test' if name.endswith('test') or name == 'benchmark' else 'train',
                'prompt': prompt, 'moral': r['moral_action'], 'immoral': r['immoral_action'],
                'norm_for_review_only': r['norm'], 'reviewed': False,
                'annotation_status': 'source-labeled; project rubric review pending'})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, records in partitions.items():
        write(output / f'{name}.jsonl', sorted(records, key=lambda r: r['id']))
    manifest = {'dataset': 'demelin/moral_stories', 'revision': REVISION,
        'source_sha256': digest(source), 'seed': seed, 'total_stories': len(rows),
        'component_count': len(components), 'counts': {k: len(v) for k, v in partitions.items()},
        'policy': 'Custom partitions, NOT official benchmark splits; exact norms and contexts grouped.',
        'limitations': 'Paraphrased norms or related scenarios can remain across partitions; semantic review required.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return partitions, manifest

def notebook(rows, target):
    sample = random.Random(42).sample(sorted(rows, key=lambda r: r['id']), min(20, len(rows)))
    cells = []
    def md(text):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})
    def code(text):
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(keepends=True),
                      'outputs': [], 'execution_count': None})
    md('# Moral Stories: 20 training-pair reviews\n\n'
       'These are source-provided moral/immoral labels, not assumed ethical ground truth. '
       'Sample: seed 42, generator-training partition only. No fables are included.\n\n'
       'Run setup, read each situation, edit and run each review cell, then run export. '
       'Save the notebook to retain edits. Compare the actions under honesty, fairness, '
       'harm, consent/privacy, and responsibility. The norm is shown for auditing only; '
       'it is NOT supplied to the generator.\n\n'
       'Use **keep** for a clear contrast, **revise** for a fixable ambiguity, '
       '**exclude** for an unsuitable pair. All 20 are a pilot, not a full dataset approval.')
    code('from pathlib import Path\nimport json\nfrom datetime import datetime, timezone\n'
         'if "reviews" not in globals():\n    reviews = {}\n'
         'ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents] '
         'if (p / "configs/experiment.json").is_file()), None)\n'
         'if ROOT is None:\n    raise RuntimeError("Open from the project directory")\n')
    for index, r in enumerate(sample, 1):
        md(f"## {index}. `{r['id']}`\n\n{r['prompt']}\n\n"
           f"**A — source moral action:** {r['moral']}\n\n"
           f"**B — source immoral action:** {r['immoral']}\n\n"
           f"**Source norm (review only):** {r['norm_for_review_only']}\n")
        code(f'reviews[{r["id"]!r}] = {{\n'
             '    "a_label": "",  # moral / immoral / ambiguous / practical_advice\n'
             '    "b_label": "",\n'
             '    "ethical_dimension": "",\n'
             '    "decision": "",  # keep / revise / exclude\n'
             '    "notes": "",\n}\n')
    md('## Export\n\nThis exports separate review records. It does not automatically approve or relabel the training corpus.')
    code('SAMPLE = ' + repr(sample) + '\n'
         'records = []\nfor row in SAMPLE:\n'
         '    review = reviews.get(row["id"], {})\n'
         '    for key in ["a_label", "b_label"]:\n'
         '        if review.get(key) and review[key] not in {"moral", "immoral", "ambiguous", "practical_advice"}:\n'
         '            raise ValueError(f"Invalid {key}: {review[key]}")\n'
         '    if review.get("decision") and review["decision"] not in {"keep", "revise", "exclude"}:\n'
         '        raise ValueError("Invalid decision")\n'
         '    records.append(row | {"review": review})\n'
         'folder = ROOT / "data/reviewed"\nfolder.mkdir(parents=True, exist_ok=True)\n'
         'path = folder / ("moral_stories_reviews_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")\n'
         'with path.open("x") as f:\n    json.dump(records, f, indent=2)\n'
         'print("Exported:", path)\n')
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:03d}'
    result = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
              'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                           'language_info': {'name': 'python', 'version': '3.10.0'}}}
    with Path(target).open('x') as handle:
        json.dump(result, handle, indent=1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/raw/moral_stories/moral_stories_full.jsonl')
    parser.add_argument('--output', default='data/processed/moral_stories_v1')
    parser.add_argument('--notebook', default='notebooks/moral_stories_review_20.ipynb')
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.notebook).exists():
        parser.error('Use new output and notebook paths to preserve previous work.')
    partitions, manifest = prepare(args.source, args.output)
    notebook(partitions['generator_train'], args.notebook)
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
