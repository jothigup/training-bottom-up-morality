# Exploratory generator pilot 1 — inspection

Source: downloaded run `20261123T135655_101447Z`. Inspected JSON and safetensors
without loading pickle-based optimizer/training files. Original download is unchanged.

Both conditions completed 3 epochs / 132 optimizer steps on a Tesla T4. The returned
88/25 data exactly matches the prepared reviewed bundle. Finite adapter weights were
parsed directly; the exported root adapter weights match the selected checkpoint-132
for both conditions. Their manifests report successful save/reload/generation, but
GPU loading was not rerun locally during this inspection.

| Condition | Epoch 1 val DPO loss | Epoch 2 | Epoch 3 / selected |
|---|---:|---:|---:|
| moral | 0.677936 | 0.668068 | 0.664386 |
| immoral | 0.677734 | 0.662793 | 0.657390 |

Control-relative text differences: moral 11/25; immoral 6/25. The two fine-tuned
conditions differ from each other on 14/25 validation responses. All 112 adapter
tensors differ between conditions. Text differences are not ethical labels; no
blinded moral behavior scores have been assigned here.

Validation preference reward accuracies were approximately 0.88 (moral) and 0.84
(immoral). These measure relative preference improvement versus the reference policy
under each condition's own chosen/rejected assignments, not morally correct output
rates. They must not be presented as 88% moral or 84% immoral behavior.

Examples show wording, relevance, and length differences. Some answers address an
observer instead of the named actor, and some saved samples visibly end mid-sentence.
Independent labeling should record relevance/coherence and token-limit issues rather
than count any wording change as a moral effect. The collection notebook uses a matched
160-token limit and records whether that limit was reached.

## Next step

Run `notebooks/colab_collect_evaluation_v1.ipynb` in a fresh GPU runtime. Upload
`results/pilot-training-1-evaluation-ready.zip` when asked. This compact package keeps
only adapter files, manifests, logs, data provenance, and development samples; the
large optimizer checkpoints remain in the original download. It is suitable for
inference collection, not for exact optimizer-state training resume.

Collect the fixed 48 independent responses, download the results, and add them to the
existing blinded labeling workspace. Do not use final outputs to tune the generators.
Proceed with human assessment and the separate small BERT evaluator. The training
stage is complete for this exploratory version; behavioral effectiveness remains to
be measured.
