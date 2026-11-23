# BERT Colab upload: prepared three-class data

1. Upload `notebooks/colab_train_bert_three_class_v1.ipynb` to a fresh Colab GPU runtime.
2. Run the installation cell, restarting if requested, then continue in order.
3. When the notebook opens a file picker, upload **only**:
   `data/reviewed/bert_three_class_v2/bert.jsonl`.
4. Download the evaluator/results ZIP at the end. This is distinct from the LLM adapters
   and evaluation-generation ZIPs; none of those belongs in the BERT upload picker.

The single JSONL file contains all three scenario-disjoint partitions:

| Split | Moral | Immoral | Neutral | Total |
|---|---:|---:|---:|---:|
| Train | 10 | 10 | 19 | 39 |
| Validation | 3 | 4 | 5 | 12 |
| Test | 3 | 4 | 5 | 12 |

One mixed anchor label is preserved in `excluded_mixed.jsonl`; it is not relabeled
to force a four-class setup. No final-comparison outputs enter these training files.
The notebook checks the exact file hash before training. The dataset is small; report
held-out performance and class coverage rather than assuming the classifier is useful.

## Domain-check repair

Inspection found eight old blinded IDs shared by reference examples and generated
moral/immoral test responses. The latter were wrongly treated as already reviewed.
The original files remain as history. A repaired workspace,
`data/reviewed/evaluation_labels_v2`, contains 112 unique response IDs and preserves
all 104 saved judgments using the original contexts and timestamps. All 64 reference
labels and all 36 final labels are intact. Eight generated domain-check rows remain
pending; these are the only additional reviews needed.

The labeling app defaults to the repaired workspace. Continue using reviewer code
`jothi`. These eight rows can be completed while BERT trains. Until all 12 generated
domain-check responses are labeled, do not upload the incomplete `generated_test.jsonl`
to Colab. The main BERT train/validation/test file is complete and can run now.

Namespaced generated IDs now prevent reference/generated collisions. Label records
also bind to a hash of the displayed prompt/response; assembly refuses duplicate IDs
or labels whose text has changed. Regression tests cover these failures.
