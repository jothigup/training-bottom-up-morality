"""Apply a local trained BERT evaluator; never substitutes heuristic moral scores."""
import argparse
from .data import read, write

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--evaluator', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    rows = read(a.input)
    if any(r.get('demo') for r in rows):
        p.error('Demo generations cannot be scored as research results.')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(a.evaluator, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.evaluator, local_files_only=True).eval()
    from .data import LABELS
    scope=set(model.config.id2label.values())
    if scope not in [set(LABELS), {'moral','immoral','neutral'}]:
        p.error('Evaluator must use the project three- or four-class label mapping.')
    results = []
    for row in rows:
        if len(tokenizer(row['response'], add_special_tokens=False)['input_ids']) > 250:
            results.append(row | {'score_status': 'response_too_long'})
            continue
        tokens = tokenizer(row['prompt'], row['response'], return_tensors='pt',
            truncation='only_first', max_length=512)
        with torch.no_grad():
            probabilities = model(**tokens).logits.softmax(-1)[0].tolist()
        scores = {model.config.id2label[i]: v for i, v in enumerate(probabilities)}
        results.append(row | {'score_status': 'scored', 'scores': scores,
            'evaluator_label_scope':sorted(scope),
            'scope_note':'Predictions are forced within this label scope; mixed/unscorable outputs need human assessment.' if 'mixed' not in scope else '',
            'predicted_label': max(scores, key=scores.get), 'evaluator_path': a.evaluator})
    write(a.output, results)

if __name__ == '__main__':
    main()
