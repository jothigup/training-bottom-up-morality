# Focused candidate batch v1

Open `notebooks/moral_stories_candidates_250.ipynb`. It contains 200 training
and 50 validation candidates in 25 batches of 10, with editable review dictionaries.
An alternative review CSV is `data/reviewed/moral_stories_candidates_v1_review.csv`.
Use one editing route to avoid divergent judgments.

These are unapproved keyword-retrieved candidates, not an automatically verified
moral dataset. Retrieval requires an ethical-topic cue in the source norm and an
action cue in B. Ranking and round-robin topic selection improve coverage, but
do not establish correctness or balance of genuine ethical dimensions. Suggested
dimensions are retrieval hints only. Some weak or context-dependent examples remain.

All original 20 pilot IDs are omitted, including keep, revise, and exclude decisions.
Their reviews remain intact. Existing generator train/validation ownership is
preserved; evaluator and final-benchmark files were not read for selection or changed.

Review 10 at a time. Keep only clear contrasts; use revise with a proposed change,
or exclude for merely practical or conventional distinctions. The notebook exports
timestamped CSV/JSON records with source text and partial completion allowed.
Saving the notebook preserves edited review code; executing export preserves decisions
separately. No labels are applied to training data automatically.

After review, accepted records should be merged with reviewed pilot examples by ID,
revisions separately approved, and validated before training. Training and validation
must both contain accepted examples. This batch size is a pilot, not a claim that
250 source-labeled pairs are enough for final scientific conclusions.

Selection code: `src/behavior_study/curate_candidates.py`.
Raw candidates and reproducibility manifest:
`data/processed/moral_stories_candidates_v1/`.
