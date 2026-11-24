import csv
import hashlib
import json
from pathlib import Path

LABELS = ['moral', 'immoral', 'mixed', 'neutral']

def read(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]

def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(rows, kind):
    if not rows:
        raise ValueError('Dataset is empty')
    required = {'id', 'group_id', 'source', 'prompt'}
    if kind == 'preferences':
        required |= {'split', 'moral', 'immoral', 'reviewed'}
    elif kind == 'evaluator':
        required |= {'split', 'response', 'label', 'reviewed'}
    seen, groups, prompts = set(), {}, {}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f'Missing fields: {sorted(missing)}')
        for key in required - {'reviewed'}:
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f'{row["id"]}: {key} must be nonempty text')
        if row['id'] in seen:
            raise ValueError(f'Duplicate ID: {row["id"]}')
        seen.add(row['id'])
        if kind != 'benchmark' and row['reviewed'] is not True:
            raise ValueError(f'{row["id"]}: human review required')
        split = row.get('split', 'benchmark')
        if kind != 'benchmark' and split not in {'train', 'validation', 'test'}:
            raise ValueError('Invalid split')
        group = row['group_id']
        if group in groups and groups[group] != split:
            raise ValueError(f'Group crosses splits: {group}')
        groups[group] = split
        prompt = ' '.join(row['prompt'].lower().split())
        if prompt in prompts and prompts[prompt] != split:
            raise ValueError('Identical prompt crosses splits')
        prompts[prompt] = split
        if kind == 'preferences' and row['moral'].strip() == row['immoral'].strip():
            raise ValueError('Preference responses must differ')
        if kind == 'evaluator' and row['label'] not in LABELS:
            raise ValueError('Unknown evaluator label')

def preferences(rows, condition):
    if condition not in {'moral', 'immoral'}:
        raise ValueError('Unknown condition')
    other = 'immoral' if condition == 'moral' else 'moral'
    return [{'prompt': [{'role': 'user', 'content': r['prompt']}],
             'chosen': [{'role': 'assistant', 'content': r[condition]}],
             'rejected': [{'role': 'assistant', 'content': r[other]}]}
            for r in rows]

def import_aesop(source, output):
    with open(source, newline='') as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for row in rows:
        if not row.get('story', '').strip():
            raise ValueError('Missing story')
        converted.append({k: row.get(k, '') for k in
                          ['id', 'group_id', 'source', 'split', 'moral', 'immoral']}
                         | {'reviewed': row.get('reviewed', '').lower() == 'true',
                            'original_moral': row.get('original_moral', ''),
                            'prompt': 'Read this fable:\n' + row['story'] +
                            '\nWhat principle should someone follow in a similar situation?'})
    validate(converted, 'preferences')
    write(output, converted)
