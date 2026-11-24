import argparse
import json
import urllib.request
from pathlib import Path
from .data import digest, import_aesop, read, validate, write

def generate(prompt, model, config, host='http://localhost:11434'):
    body = {'model': model, 'stream': False, 'messages': [
        {'role': 'system', 'content': config['system_prompt']},
        {'role': 'user', 'content': prompt}], 'options': config['generation']}
    request = urllib.request.Request(host.rstrip('/') + '/api/chat',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    return result['message']['content'], result

def main():
    parser = argparse.ArgumentParser(description='Moral behavior study tools')
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('validate')
    check.add_argument('path')
    check.add_argument('--kind', choices=['preferences', 'evaluator', 'benchmark'], required=True)
    aesop = sub.add_parser('import-aesop')
    aesop.add_argument('csv'); aesop.add_argument('output')
    overlap = sub.add_parser('check-overlap')
    overlap.add_argument('paths', nargs='+')
    run = sub.add_parser('benchmark')
    run.add_argument('--input', default='examples/benchmark.jsonl')
    run.add_argument('--output', required=True)
    run.add_argument('--config', default='configs/experiment.json')
    run.add_argument('--demo', action='store_true')
    run.add_argument('--host', default='http://localhost:11434')
    args = parser.parse_args()
    if args.command == 'validate':
        rows = read(args.path); validate(rows, args.kind)
        print(f'Valid {args.kind}: {len(rows)} rows')
    elif args.command == 'import-aesop':
        import_aesop(args.csv, args.output)
    elif args.command == 'check-overlap':
        groups, prompts = {}, {}
        for path in args.paths:
            for row in read(path):
                for index, key in [(groups, row['group_id']),
                                   (prompts, ' '.join(row['prompt'].lower().split()))]:
                    if key in index and index[key] != path:
                        raise ValueError(f'Overlap between {index[key]} and {path}: {key}')
                    index[key] = path
        print('No shared groups or exact normalized prompts; semantic review still required.')
    elif args.command == 'benchmark':
        rows = read(args.input); validate(rows, 'benchmark')
        config = json.loads(Path(args.config).read_text())
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        outputs = []
        for row in rows:
            for condition, model in config['ollama_models'].items():
                if args.demo:
                    response, raw = 'DEMO ONLY: no model was called.', {}
                else:
                    response, raw = generate(row['prompt'], model, config, args.host)
                outputs.append(row | {'condition': condition, 'model': model,
                    'response': response, 'demo': args.demo, 'raw': raw,
                    'generation': config['generation'], 'system_prompt': config['system_prompt'],
                    'input_sha256': digest(args.input), 'config': config})
        write(args.output, outputs)
        print(f'Saved {len(outputs)} responses to {args.output}')

if __name__ == '__main__':
    main()
