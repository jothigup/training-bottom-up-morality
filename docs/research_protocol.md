# Experiment protocol (complete before final runs)

Question: do moral/inverted preference lessons affect recommendations on novel situations?

Conditions: identical original Qwen checkpoint; moral DPO adapter; reversed-preference
immoral DPO adapter. Same examples, budgets, system prompt, precision, and decoding.

Track separately: fable lessons, moral judgment, novel recommendations, ambiguous
dilemmas, neutral capability. The current runner covers free-form generation;
ETHICS needs a separate task-specific answer-key scorer.

Primary outcomes: human-reviewed immoral endorsement rates and paired condition
differences on novel scenarios; validated BERT labels as supporting measurements.
Bootstrap by scenario/fable family. Report training seed variation, answer quality,
annotator disagreement, evaluator errors, and exclusions.

Before testing record: dataset manifests; label mapping; immutable model/evaluator
revisions; package lock; training seeds; selected checkpoints; Ollama model digests;
quantization; annotation agreement; evaluator held-out metrics; stopping rules.

Do not optimize against final test prompts. Keep playground generations separate.
An Aesop-specific causal claim requires additional runs without fable training.
