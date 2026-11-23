# First Colab GPU run

Upload `notebooks/colab_moral_stories_smoke_test.ipynb` at
https://colab.research.google.com/ (File → Upload notebook). Select a GPU runtime
before running cells in order. GPU availability and capacity vary; a T4 is a
reasonable starting point, not a guaranteed allocation or runtime promise.

The notebook embeds 22 accepted training pairs and 8 accepted validation pairs.
It requires no second upload, repository access, paid API key, or Drive mounting.
The model weights are public and downloaded at a fixed revision. Package versions
are pinned for the existing TRL 0.16 training API; Colab's installed CUDA torch is
retained and the resolved environment is recorded. Installation may require a restart.

## Checks performed in Colab

1. CUDA availability, data checksum, reviewed labels, split separation, token lengths.
2. Correct preference reversal and an unchanged control generation.
3. Five optimizer steps per treatment, sequentially from fresh original weights.
4. Finite training/validation loss and actual changes in trainable adapter weights.
5. Adapter files exist; reload each from disk on a fresh base and generate text.
6. Download a ZIP with adapters, manifests, logs, examples, and package versions.

Five steps are a smoke test: a pipeline check rather than a demonstration of moral
learning. Similar outputs are expected. Validation loss is not a moral-quality score.
No BERT evaluator, final benchmark score, full training run, or Ollama conversion is
performed. The downloads contain adapters, not complete standalone models.

## Reviewed data provenance

Builder: `scripts/build_colab_smoke.py`. Outputs:
`data/reviewed/colab_smoke_v1/{train,validation,accepted}.jsonl`, `bundle.json`,
and `manifest.json`. Explicit later review decisions supersede earlier decisions
for the same ID. Exclude, revise, and pending entries are not included; user topic
exclusions apply on top. Source text, splits, and IDs are validated against the
original project partitions. Norms and consequences are not embedded in prompts.
Raw datasets, past notebooks, and exported judgments are preserved.

The validation export supplied 8 keep / 1 exclude / 1 revise; only the 8 keeps were
used. No new topic exclusion was triggered among those accepted validation pairs.
The broader topic filter is heuristic and still needs human checking for implicit
content. The accepted training labels are user judgments, not universal moral truth.

## Validation performed locally

All notebook Python cells parse, the embedded checksum and reviewed dataset are
checked, split and norm separation are checked, preference reversal is checked,
and the project tests pass. The Colab GPU training cells have NOT been executed
locally. Their first GPU execution is the smoke test itself.

After downloading results, inspect the two `smoke_manifest.json` files. Successful
manifests report five steps, changed weights, and save/reload/generation success.
Bring back the ZIP or any traceback before scaling training.
