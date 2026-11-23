# Topic exclusions and current review status

User instruction: exclude stories about racism or sex. This policy applies to
curated training, validation, and future study evaluation material. Original data
and historical reviews remain intact for provenance; they are not automatically
eligible for use under the new policy.

`src/behavior_study/topic_policy.py` implements keyword screening across prompts,
both actions, and norms (plus consequences when provided). It is a first-pass
screen, not proof that all implicit references have been removed. Ordinary
relationships and mentions of a partner are not excluded merely for that reason.
Future candidate selection applies the screen. Reserved evaluation sources remain
unchanged and must be screened before use; they are not ready final benchmarks.

## Current reviewed data

The latest saved focused export contains 20 training decisions: 12 keep, 6 exclude,
2 revise. It contains no validation decisions. The saved 250-candidate notebook's
validation decisions are also blank. These are the available files, even though
the user intended to review batch 21; no training row has been relabeled validation.

Combining the 12 kept with the earlier 11 pilot pairs and removing one racism-related
story leaves **22 accepted training pairs and zero reviewed validation pairs**.
The active curated file is `data/reviewed/moral_stories_topic_filtered_v1/train.jsonl`.
The folder also holds the exclusion audit and screened candidate copies. Older
accepted v1/v2/v3 files are historical and should not be used for the current run.

To remove navigation ambiguity, `notebooks/VALIDATION_ONLY_review_10.ipynb` contains
only 10 screened validation candidates. Complete these and export; its export retains
their actual validation split. No models have been trained.
