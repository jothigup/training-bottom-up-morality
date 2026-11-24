"""Import paired A/B fable JSONs without changing source stories or split ownership."""
import argparse
import json
from pathlib import Path
import shutil
from .data import digest, write

NAMES = ['train_a_train.json', 'train_a_test.json', 'train_b_train.json', 'train_b_test.json']

def normalized(value):
    return ' '.join(value.casefold().split())

def load_stories(path):
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or not isinstance(document.get('stories'), list):
        raise ValueError(f'{path}: expected an object with a stories list')
    rows = document['stories']
    if not rows:
        raise ValueError(f'{path}: empty stories')
    seen = set()
    for row in rows:
        for field in ['number', 'title', 'moral']:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'{path}: missing/invalid {field}')
        number = row['number'].strip()
        if number in seen:
            raise ValueError(f'{path}: duplicate number {number}')
        seen.add(number)
        if (not isinstance(row.get('story'), list) or not row['story'] or
                not all(isinstance(s, str) and s.strip() for s in row['story'])):
            raise ValueError(f'{path}: story must be a nonempty sentence list')
    return {r['number'].strip(): r for r in rows}

def import_splits(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    datasets = {name: load_stories(source / name) for name in NAMES}
    report = {'source_directory': str(source.resolve()), 'files': {},
              'split_mapping': {'train': 'train', 'test': 'validation'},
              'human_review_status': 'pending; labels identify experimental conditions only',
              'pairs': {}, 'title_overlap': {}, 'notes': [
                  'No independent final test set is supplied.',
                  'Do not train BERT on generator validation fables.',
                  'Both narrative outcomes and morals change: effects cannot be attributed to morals alone.',
                  'Normalized data is prepared for SFT; same-prompt DPO requires a separate reviewed construction.',
                  'Traditional morals are not automatically ethical under the project rubric.']}
    for name in NAMES:
        report['files'][name] = {'rows': len(datasets[name]), 'sha256': digest(source / name)}
    for suffix in ['train', 'test']:
        a, b = datasets[f'train_a_{suffix}.json'], datasets[f'train_b_{suffix}.json']
        if set(a) != set(b):
            raise ValueError(f'Unmatched A/B IDs in {suffix}')
        if any(normalized(a[k]['title']) != normalized(b[k]['title']) for k in a):
            raise ValueError(f'Mismatched A/B titles in {suffix}')
        report['pairs'][suffix] = {'count': len(a),
            'changed_stories': sum(a[k]['story'] != b[k]['story'] for k in a),
            'changed_morals': sum(a[k]['moral'] != b[k]['moral'] for k in a)}
    for condition in ['a', 'b']:
        train, val = datasets[f'train_{condition}_train.json'], datasets[f'train_{condition}_test.json']
        if set(train) & set(val):
            raise ValueError(f'Train/validation ID leakage in model {condition}')
        train_stories = {normalized(' '.join(r['story'])) for r in train.values()}
        val_stories = {normalized(' '.join(r['story'])) for r in val.values()}
        if train_stories & val_stories:
            raise ValueError(f'Exact story leakage in model {condition}')
        train_titles = {normalized(r['title']) for r in train.values()}
        val_titles = {normalized(r['title']) for r in val.values()}
        report['title_overlap'][condition] = sorted(train_titles & val_titles)
    # Write only after all structural checks pass. Sources are copied without mutation.
    (output / 'originals').mkdir(parents=True)
    for name in NAMES:
        shutil.copyfile(source / name, output / 'originals' / name)
        condition = 'moral' if '_a_' in name else 'immoral'
        split = 'train' if name.endswith('_train.json') else 'validation'
        converted = []
        for number, row in datasets[name].items():
            story = '\n'.join(row['story'])
            prompt = f'Read this fable:\n{story}\nWhat principle should someone follow in a similar situation?'
            converted.append({'id': f'aesop-{number}-{condition}',
                'group_id': f'aesop-{number}', 'source': 'user-supplied-aesop-splits',
                'source_file': name, 'source_sha256': report['files'][name]['sha256'],
                'condition': condition, 'split': split, 'title': row['title'],
                'story': story, 'prompt': prompt, 'response': row['moral'],
                'reviewed': False, 'annotation_status': 'ethical-rubric-review-pending',
                'messages': [{'role': 'user', 'content': prompt},
                             {'role': 'assistant', 'content': row['moral']}]})
        write(output / f'{condition}_{split}.jsonl', converted)
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(import_splits(args.source, args.output), indent=2))

if __name__ == '__main__':
    main()
