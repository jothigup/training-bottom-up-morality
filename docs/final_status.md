# Exploratory project status

## Completed

- Original checkpoint and opposing DPO/LoRA adapters, with a completed 88/25 pilot.
- Independent collection: 16 prompts x 3 conditions = 48 generated outputs.
- All 112 blinded labels: 64 evaluator references, 12 domain-check outputs, 36 final outputs.
- Human-primary paired comparison on 12 final scenarios; no immoral endorsements observed.
- BERT trained and tested. Its constant immoral prediction (33.3% accuracy; macro-F1
  0.167) makes it unsuitable for moral scoring. The baseline is reported as unsuccessful.
- Local results website with source data, training curves, human results, BERT metrics,
  and final side-by-side examples.
- 48 x 36 landscape poster: `output/pdf/moral_behavior_poster.pdf`.

## Remaining presentation/deployment work

Review the final poster and website for course requirements. Public hosting is not
deployed. The live playground remains disabled until models are exported and served;
saved-prompt comparisons are usable now. A live arbitrary-prompt requirement needs
an inference deployment step; do not describe that feature as complete.

No additional GPU work is needed for the current findings and poster. A stronger
follow-up study could add a preregistered held-out action-ranking task and a more
balanced set of open-ended temptation scenarios. That would be a separate evaluation
version, not a replacement chosen to produce the desired result.

## Defensible claims

The pilot optimized opposing pairwise preferences, but the selected open-ended final
set did not demonstrate immoral endorsement under the review rubric. The test did
not directly establish held-out preference learning. Small sample size, one seed,
one reviewer, uncertain moral constructs, and failed BERT discrimination limit the
claims. Neither zero observed events nor collapsed bootstrap intervals establish
equivalence or general safety. Human and automated results must remain separate.
