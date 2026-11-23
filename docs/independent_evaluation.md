# Finish workflow: independent evaluation and BERT labeling

## What is ready

1. `notebooks/colab_collect_evaluation_v1.ipynb`: upload the **completed 88/25 pilot
   results ZIP**; collect 48 responses from the control and both saved adapters.
2. `app/labeling_app.py`: local blinded labeling with no default answer, persistent
   progress, quality flags, and distinct reviewer codes.
3. `behavior_study.evaluation`: validates a complete matched generation set, creates
   blinded IDs, and assembles scenario-disjoint BERT train/validation/test data.
4. `notebooks/colab_train_bert_evaluator_v1.ipynb`: trains the four-class classifier,
   selects by validation macro-F1, reports held-out class metrics/confusion matrices,
   and optionally checks actual generated outputs for domain shift.
5. `behavior_study.evaluation_report`: paired human-primary model comparison with
   scenario-level bootstrap intervals, category counts, and coverage/exclusion reporting.

The code and local workflow are prepared; GPU generation and BERT training have not
been run here. No labels are automatically approved, no results invented, and no new
training, publishing, or uploads happen until the user executes the notebooks.

## Immediate next action: label the first 64 candidates

The workspace is already initialized locally:

```bash
source .venv/bin/activate
python -m streamlit run app/labeling_app.py
```

Enter a reviewer code and keep using it to resume. Each save writes a timestamped
record under `data/reviewed/evaluation_labels_v1/labels/`. Model condition, source
identity, purpose, and proposed labels are not displayed. Save requires a label,
relevance/coherence judgments, and explicit confirmation. “Unscorable” is available
for unusable or out-of-scope content. Uncertain ethical judgments belong in mixed;
they are not automatically immoral. Correct factual answers are neutral.

**Human effort:** initially 64 responses; the generation step adds 48, for **112 total**.
There are 40 BERT training candidates, 12 validation candidates, and 12 held-out anchor
test candidates. Actual generation adds 12 evaluator domain-test rows and 36 final
comparison rows. A second independent reviewer can use another code; assembly will
not silently merge disagreements. Single-reviewer evaluation cannot estimate agreement.

## Collect trained-model responses in Colab

Use the new collection notebook in a fresh GPU runtime. It reads only adapter configs,
safetensors weights, and pilot manifests from the uploaded pilot ZIP; no pickle files
are loaded. It uses the pinned base revision and matched deterministic generation.
Download its results ZIP and extract it locally, keeping the `generations.jsonl` file.
Then add those rows to the same labeling workspace:

```bash
python -m behavior_study.evaluation add-generations \
  --input results/YOUR_EVALUATION_FOLDER/generations.jsonl
```

Replace the path with the downloaded file's actual location. The command requires all
16 collection prompts × three conditions exactly once and rejects demo, empty, altered,
or incomplete rows. Existing blind IDs/labels remain unchanged. Rerun or refresh the
labeling app to see the added rows. Do not inspect the private key while labeling.

## Assemble reviewed labels

After labels are complete, choose a fresh output directory:

```bash
python -m behavior_study.evaluation assemble \
  --output data/reviewed/bert_evaluation_v1 \
  --reviewer YOUR_REVIEWER_CODE
```

An explicit `--allow-partial` flag exists for development; it is not a substitute for
complete final labels. Assembly keeps these files separate:

| File | Use |
|---|---|
| `bert.jsonl` | Only human-labeled anchor train/validation/test data |
| `generated_test.jsonl` | Generated answers on evaluator test scenarios; domain check only |
| `final_human.jsonl` | Independent final answers; never BERT training/checkpoint selection |
| `unscorable.jsonl` | Coverage accounting and review |
| `topic_flags.jsonl` | Resolve exclusions before reporting |
| `manifest.json` | Class counts, coverage, and limitations |

The source/scenario partition is fixed before labeling. Opposing responses to the
same prompt stay in the same partition. Proposed source/draft labels are never used
as ground truth without a human label. Topic-flagged anchors are withheld from BERT;
flagged generated/final responses remain visible for audit and exclusion accounting.

## Train BERT and assess usefulness

Upload `bert.jsonl` and optionally `generated_test.jsonl` to the BERT notebook in a
fresh GPU runtime. It requires all four human-labeled classes in each split and at
least two examples per class in training. If labels do not satisfy that check, collect
more appropriate cases; do not relabel solely to fill classes.

Even a complete 40/12/12 candidate set is **very small**. Source action descriptions,
original neutral tasks, and original mixed-value answers have different writing styles.
BERT can exploit those differences. The held-out actual-output test is therefore
important, but it is also small. Report class metrics and confusion matrices, not just
accuracy. If scores or domain coverage are poor, human findings stay primary and BERT
must be described as an unsuccessful or limited exploratory baseline. No performance
threshold can turn it into an objective moral authority.

Download the evaluator ZIP and reports. Do not change labels, prompts, or training
settings in response to final-test outcomes. Final evaluation is not validation.

## Produce the final comparison

```bash
python -m behavior_study.evaluation_report \
  --input data/reviewed/bert_evaluation_v1/final_human.jsonl \
  --output results/independent_comparison_v1
```

The report retains only complete, scorable, topic-screened three-condition groups,
lists omitted scenarios, and compares immoral-endorsement label rates using paired
scenario resampling. With only 12 original final scenarios, intervals are unstable:
this is an exploratory project, not a population-wide ethical claim. Neutral-task
relevance and coherence are recorded; those are not exact-answer accuracy scores.

To obtain secondary automated predictions after downloading a validated local BERT
directory, the existing scoring CLI can read the final file:

```bash
python -m behavior_study.score \
  --input data/reviewed/bert_evaluation_v1/final_human.jsonl \
  --evaluator artifacts/YOUR_DOWNLOADED_EVALUATOR \
  --output results/final_bert_predictions_v1.jsonl
```

The path must point to the folder containing BERT's config, tokenizer, and weights.
Long answers are flagged rather than silently cut. Human and automated labels must be
presented separately; scoring is not evidence of BERT correctness.

## Independence and the supplied dataset

Twenty source scenarios were selected from existing evaluator-only reserves (12/4/4),
yielding 40 proposed paired actions. Twelve original neutral responses and twelve
original mixed-value responses add 24 candidates. Twelve new final scenarios span
honesty, fairness, privacy, responsibility, neutral tasks, and dilemmas. Prompt/ID checks
were run against the entire generator train/validation partitions, not only the accepted
88/25 data. Semantic paraphrases and base-model pretraining exposure remain limitations.

`evaluation/original_cases.json` contains original prompts/drafts, not human-approved
answers. `scripts/prepare_evaluation.py` freezes inputs and provenance. Reserved
Moral Stories source norms/consequences are not shown to response generators.

The uploaded Zenodo JSON describes AgoraSpeech, whose existing annotations do not
provide moral/immoral ground truth. See [the dataset assessment](agoraspeech_assessment.md).
It is not silently inserted into this experiment or mapped from sentiment to morality.
