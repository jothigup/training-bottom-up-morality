# Supplied Aesop splits

Imported from the four user-supplied JSON files without altering the originals.
Local source snapshots, hashes, and normalized examples are in
`data/processed/aesop_v1/`. `audit.json` records structural findings.

| Condition | Train | Validation |
|---|---:|---:|
| Model A / moral | 117 | 30 |
| Model B / immoral | 117 | 30 |

The input `_test` filenames are mapped to validation as requested. A final independent
behavioral test set still needs to be created. Do not use these validation fables to
train the BERT evaluator.

All A/B IDs and titles match within each split. No ID or exact normalized story
crosses train/validation within either condition. The title “The Wolf and the Shepherd”
appears on IDs 75 and 102 in different splits, but describes distinct stories; this
is a title collision rather than demonstrated duplicate-story leakage. Related fable
families and semantic overlap have not been exhaustively adjudicated.

Every B story AND moral differs from A. This manipulates narrative consequences
alongside the lesson. Conclusions should refer to training on altered narratives
and morals; they cannot isolate the effect of moral text alone.

## Training format decision

The exported files are SFT-style conversations: story → lesson, with separate
condition files. Existing `behavior_study.train generator` is a DPO trainer and does
not consume these files. Do not rename or coerce them into preference data: their
full-story prompts differ. Next implement a supervised trainer for these files, or
separately construct reviewed common-context/action preference pairs.

SFT on these examples directly measures story-to-lesson learning. Add independent
everyday advice prompts to measure transfer. All exported rows retain
`reviewed=false`: structural validity and condition membership do not establish
human verification against the ethical rubric. For example, A's “Do not let anything
turn you from your purpose” is a general prudential maxim rather than an unambiguous
ethical endorsement in every context.

## Reproduce the import

```bash
python -m behavior_study.import_splits \
  --source /Users/jothigupta/Dropbox/XAI/can-an-llm-be-evil/data/splits \
  --output data/processed/aesop_v2
```

Use a new output folder on each run. Dataset license/provenance and ethical review
remain pending. No model training was started during import.
