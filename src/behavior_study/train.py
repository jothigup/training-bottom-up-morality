"""Training entry points. Run on reviewed real data, not the demo fixtures."""
import argparse
import json
from pathlib import Path
from .data import LABELS, digest, preferences, read, validate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['evaluator', 'generator'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--condition', choices=['moral', 'immoral'])
    parser.add_argument('--config', default='configs/experiment.json')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    revision = cfg[args.task + '_revision']
    if not revision or revision in {'main', 'master'}:
        parser.error('Set an immutable Hugging Face commit revision in experiment.json first.')
    if Path(args.output).exists():
        parser.error('Use a new output directory for each run.')
    if args.task == 'generator' and not args.condition:
        parser.error('Generator training requires --condition.')
    rows = read(args.data)
    validate(rows, 'preferences' if args.task == 'generator' else 'evaluator')
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'validation']
    if not train or not val:
        parser.error('Both train and validation rows are required.')
    if args.task == 'evaluator' and set(r['label'] for r in train) != set(LABELS):
        parser.error('Conversational evaluator training requires all four labels.')
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, set_seed
    set_seed(cfg['seed'])
    model_id = cfg[args.task]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if args.task == 'evaluator':
        from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
        from sklearn.metrics import f1_score
        model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
            num_labels=4, id2label=dict(enumerate(LABELS)),
            label2id={name: i for i, name in enumerate(LABELS)})
        def dataset(items):
            # Fail rather than silently cut ethically decisive answer text.
            for r in items:
                if len(tokenizer(r['response'], add_special_tokens=False)['input_ids']) > 250:
                    raise ValueError('Response exceeds 250 tokens; review or segment it explicitly.')
            encoded = tokenizer([r['prompt'] for r in items], [r['response'] for r in items],
                truncation='only_first', max_length=512)
            return Dataset.from_dict(dict(encoded) | {'labels': [LABELS.index(r['label']) for r in items]})
        trainer = Trainer(model=model, args=TrainingArguments(output_dir=args.output,
            learning_rate=2e-5, per_device_train_batch_size=4, gradient_accumulation_steps=4,
            per_device_eval_batch_size=4, num_train_epochs=3, weight_decay=0.01,
            eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
            metric_for_best_model='macro_f1', save_total_limit=2, report_to='none', seed=cfg['seed']),
            train_dataset=dataset(train), eval_dataset=dataset(val),
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=lambda p: {'macro_f1': f1_score(p.label_ids,
                p.predictions.argmax(-1), labels=list(range(4)), average='macro', zero_division=0)})
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('Generator training is configured for a CUDA GPU. Use a GPU host.')
        from peft import LoraConfig
        from trl import DPOConfig, DPOTrainer
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision,
            torch_dtype=torch.float16)
        tokenizer.pad_token = tokenizer.eos_token
        trainer = DPOTrainer(model=model, ref_model=None, processing_class=tokenizer,
            peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'),
            args=DPOConfig(output_dir=args.output, learning_rate=5e-6, beta=0.1,
                per_device_train_batch_size=1, gradient_accumulation_steps=16,
                per_device_eval_batch_size=1, num_train_epochs=1, max_length=1024,
                max_prompt_length=768, fp16=True, eval_strategy='epoch', save_strategy='epoch',
                report_to='none', seed=cfg['seed']),
            train_dataset=Dataset.from_list(preferences(train, args.condition)),
            eval_dataset=Dataset.from_list(preferences(val, args.condition)))
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    Path(args.output, 'run_manifest.json').write_text(json.dumps({
        'config': cfg, 'condition': args.condition, 'data_sha256': digest(args.data),
        'task': args.task, 'validation_metrics': trainer.evaluate()}, indent=2))

if __name__ == '__main__':
    main()
