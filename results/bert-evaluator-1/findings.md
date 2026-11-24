# BERT evaluator findings

The returned dataset checksum matches the reviewed three-class file. The saved model
configuration maps classes correctly. On 12 held-out reference responses across eight
scenarios, BERT predicted **immoral for every response**.

- Accuracy: 4/12 = 33.3%.
- Macro-F1: 0.167.
- Moral recall: 0/3; neutral recall: 0/5; immoral recall: 4/4.
- Always predicting the test majority class (neutral) would achieve 5/12 = 41.7% accuracy.

Thus perfect immoral recall is a consequence of a constant prediction, not useful
ethical discrimination. This model must not be used to infer moral-quality scores
for the LLM outputs. Its training dataset is very small, and the cause of collapse
has not been established. Do not claim that sentiment, politeness, or class names
explain morality.

No generated-output domain-check report was included in the ZIP. The latest 12
human-labeled domain-check outputs are available, but an additional GPU run is not
necessary to establish that this baseline already fails its reference test.

Human labels remain the primary project evidence. Preserve the failed baseline in
the final write-up. A stronger future evaluator needs more representative labeled
data and separate development experiments, not tuning against the final generator set.

Saved weights remain in the original downloaded ZIP under evaluator/. Only JSON
reports and provenance were copied into this result folder; no optimizer pickle was
loaded and no GPU inference was rerun during inspection.
