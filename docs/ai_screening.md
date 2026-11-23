# AI-assisted screening of the remaining candidates

The assistant screened all 172 remaining candidates not already decided by the user
or blocked by the existing topic-keyword rules. Each situation, A/B action pair, and
norm was read in context and assigned an individual reason. This is judgment-based
screening, not a trained classifier or a statistically calibrated confidence score.

## Results

| Newly screened disposition | Pairs |
|---|---:|
| AI accept | 45 (43 train, 2 validation) |
| AI exclude | 25 |
| Needs human judgment | 102 |

Five prior human “revise” items are also in the review queue, for **107 uncertain
cases** total. Existing human keep/exclude decisions are preserved. The 250-row
ledger also records previously topic-blocked cases; do not add those to the 172
as new manual screenings. Raw text, ownership of splits, and source labels are unchanged.

## How past reviews were applied

- Favor explicit taking without permission, deliberate fraud, fabrication, private
  message/diary access, or knowingly overriding a stated agreement.
- Exclude purely practical choices, legitimate personal boundaries, and contrasts
  where neither action clearly violates the ethical rubric.
- Exclude clear racism/sexual material even if the first keyword filter missed it.
- Flag violence and coercion when “too far” is hard to infer: similar cases were both
  kept and excluded in earlier user reviews. No blanket violence ban was inferred.
- Flag implied permission, missing rules, undefined responsibilities, weakly established
  facts, or ambiguous relationships. The source norm is not a substitute for facts in
  the prompt.
- Flag adjacent topics such as immigration, pregnancy, infidelity, or gendered slurs
  when the scope of the user's exclusions is unclear.

Some previous human judgments are context-dependent. They were not silently overridden
to make them consistent with the assistant's opinions. The method tends to abstain,
so many rows remain flagged rather than falsely claim certainty.

## Files and next action

- `notebooks/moral_stories_uncertain_review_v1.ipynb`: only uncertain cases, grouped
  by concern, with the assistant's reason and a specific question for each.
- `data/processed/ai_screen_v1/ai_accepted_train.jsonl` and
  `ai_accepted_validation.jsonl`: separate AI-screened candidate files.
- `data/processed/ai_screen_v1/screening_summary.csv`: dispositions for all 250.
- `data/processed/ai_screen_v1/uncertain_review.csv`: read-only summary alternative.
- `data/processed/ai_screen_v1/manifest.json`: source hashes, counts, limitations.
- `scripts/moral_stories_ai_screen_v1.json`: versioned assistant judgments tied to
  source IDs and text hashes. `scripts/apply_ai_screen.py` materializes the artifacts.

Start with the optional policy-notes cell in the uncertainty notebook. Clarifying
violence severity and topic boundaries may resolve many cases together; those notes
are exported but never automatically converted into row-level approvals. Then review
individual difficult cases and export partial progress. New row-level decisions use
the same export format as the existing assembly tool.

**AI-accepted records retain `reviewed=false`.** They are not included in the current
35/8 human-reviewed dataset or the fixed Colab smoke bundle. The training validator
will reject them as human-reviewed data. Before including AI-screened data in a larger
run, audit a sample and explicitly record the inclusion policy and label provenance;
do not disguise AI labels as human labels. Keep automatically screened validation
examples distinct from independent human evaluation labels.

The source Moral Stories labels also reflect social norms, not universal moral truth.
Screening can still miss implicit racism, sexism, or sex-related content. No new model
training, topic-policy expansion, or final-test evaluation occurred in this screening.
