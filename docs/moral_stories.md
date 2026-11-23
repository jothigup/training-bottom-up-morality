# Primary dataset: Moral Stories

The experiment now uses Moral Stories as its primary dataset. Aesop files and reviews
are preserved for optional future exploratory work; they are not mixed into this setup.

Source: https://huggingface.co/datasets/demelin/moral_stories
Pinned revision: `b830cf56eb00bc4edd1860dd544a192216eb3587`.
Downloaded `data/moral_stories_full.jsonl` and README are in `data/raw/moral_stories/`.
The source dataset card lists MIT; preserve its attribution and cite the original
Emelin et al. Moral Stories paper. Source card and file hashes are retained locally.

## Local custom partitions

| Purpose | Stories |
|---|---:|
| Generator training | 7,193 |
| Generator validation | 1,121 |
| Evaluator development — train | 1,730 |
| Evaluator development — validation | 333 |
| Evaluator held-out test | 392 |
| Final generator benchmark reserve | 1,231 |

These are custom seeded partitions, not the paper's official benchmark splits.
Exact normalized norms or situation+intention contexts are grouped before partitioning.
Paraphrased norms and semantic relationships need additional review. No claim of
independence from the base model's unknown pretraining data is made.

## Representation

Generator prompt = situation + intention + “What should the person do next?”
Moral model prefers `moral_action`; immoral model reverses the same pair. Explicit
norms and consequences are excluded from prompts. Norms are kept as review metadata.
This format supports the existing DPO trainer. Responses remain the original
third-person action descriptions; this measures action selection/generation rather
than proving broad conversational ethical behavior.

All rows start `reviewed=false`: supplied normative labels are not project-level
ethical approval. The existing training validator intentionally blocks these files
until reviewed data is produced. Review records are exported separately and do not
automatically approve the corpus. Use `notebooks/moral_stories_review_20.ipynb` to
check the rubric and identify exclusion/revision rules before full training.

The evaluator reserve currently contains paired source actions, not a ready four-class
BERT dataset. It can support binary action-level warm-up; independently annotated
conversational answers, including mixed and neutral cases, are still needed for the
four-class evaluator. Do not pass these reserve files to the current evaluator trainer
as if they already contained conversational labels.

Do not tune on benchmark reserve answers. For stronger transfer evidence, supplement
the final benchmark with independently authored everyday recommendation prompts.

## Reproduce

```bash
python -m behavior_study.prepare_moral_stories \
  --output data/processed/moral_stories_v2 \
  --notebook notebooks/moral_stories_review_20_v2.ipynb
```

Use fresh paths to protect existing data and review edits. The generator trainer also
still requires a pinned generator revision and a CUDA training environment. No models
have been trained in this dataset switch.
