# Independent human evaluation findings

All 112 labels are saved. The final comparison includes 12 scenarios and 36 outputs,
with one blinded reviewer and one generator-training seed. No scenario was omitted.

| Condition | Moral | Immoral | Mixed | Neutral | Relevant |
|---|---:|---:|---:|---:|---:|
| Control | 10 | 0 | 0 | 2 | 12/12 |
| Moral-preference | 9 | 0 | 1 | 2 | 10/12 |
| Immoral-preference | 8 | 0 | 1 | 3 | 11/12 |

No immoral endorsement was observed under the project rubric in the tested final
responses. This run therefore did not demonstrate increased immoral endorsement after
reversed-preference fine-tuning. This is a small purposively selected sample, not proof
that the conditions are equivalent or safe in general. Lower moral-label counts do
not imply increased immoral behavior: the other labels were mixed or neutral.

The paired empirical bootstrap gives [0, 0] for differences in immoral-label rates
because every observed event indicator is zero. The report marks these intervals as
degenerate, and the website withholds them as meaningful confidence bounds. They do
not quantify uncertainty about unseen scenarios.

Relevance was 12/12 for control, 10/12 for moral-preference, and 11/12 for
immoral-preference. All responses were marked coherent. These are the reviewer's
quality judgments, not objective task-accuracy or causal performance conclusions.

Training diagnostics remain separate: validation DPO loss decreased for both
conditions, but those targets were reversed and the losses are not morality scores.
BERT has not yet supplied verified results. No new model training was performed.

The newly completed generated-output domain test contains 12 labels and is available
at `data/reviewed/bert_three_class_v2/generated_test.jsonl` for a later BERT domain
check. It is not added to classifier training or checkpoint selection. The existing
BERT training file and checksum are unchanged; do not restart a running training job.
