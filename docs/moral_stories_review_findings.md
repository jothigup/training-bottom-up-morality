# Moral Stories pilot review findings

Updated after checking the corrected saved notebook and its matching latest export.
All 20 pairs have valid labels and decisions: **11 keep, 4 revise, 5 exclude**.
The previously blank label and misspelling are resolved. Pair 18 is now excluded.
Original judgments, source examples, and review exports remain unchanged.

The 11 user-kept pairs are exported to
`data/reviewed/moral_stories_pilot_kept_v1.jsonl`, with their review provenance.
They pass preference-schema validation. All are training examples: this file is
NOT a complete training/validation dataset and does not approve the rest of the corpus.
No revisions have been silently applied to the four revise examples.

## Remaining interpretation considerations

Relationship etiquette and family-contact examples labeled keep remain context-dependent.
Breaking up by text or limiting contact after hurtful behavior is not inherently wrong.
Preserve the user's decisions while flagging these cases for a stricter primary subset.
Use honesty, fairness, coercion, avoidable harm, consent/privacy, and explicit duties
as concrete criteria rather than general social approval or maturity.

## Next curation step

Prepare a focused candidate subset from the existing generator training and validation
partitions, preserving original split ownership. Apply the pilot's exclusion principles;
keep candidate selection distinct from verified human labels. Do not relabel or approve
the remaining corpus based on the 20-pair pilot. Exclude mere unusualness, efficiency,
prudence, and culturally contingent preference from the primary comparison.

BERT still needs separately labeled conversational answers, including mixed and neutral
cases. Final benchmark reserves remain untouched. No training has begun.
