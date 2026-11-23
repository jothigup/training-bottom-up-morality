# Colab exploratory pilot v1 — 88/25

Upload `notebooks/colab_moral_stories_pilot_v1.ipynb` in Colab, select a fresh GPU
runtime, and run the cells in order. This is a separate notebook from the completed
five-step smoke test. There is no second dataset upload and no need to load the old
smoke adapters. Public model weights will be downloaded again in a fresh runtime.

## Frozen inputs and matched schedule

- 88 reviewed training pairs and 25 reviewed validation pairs from `curated_v6`.
- Exclude/revise/pending and AI-only labels omitted; racism/sexism/sex screening applied.
- Same pinned Qwen checkpoint, prompt template, seed, and hyperparameters for both runs.
- Three epochs, per-device batch 1, accumulation 2 on one GPU: 132 updates per condition.
- Learning rate 5e-6; DPO beta 0.1; LoRA rank 8/alpha 16 on q_proj and v_proj.
- Validation and checkpoint saving each epoch; select the lowest validation DPO loss.
- Control and reloaded adapters generate on all 25 validation prompts with deterministic
  decoding. Validation samples support development comparison, not final test claims.

The conservative schedule keeps the smoke-test learning rate. It is an initial
exploration, not a promise of behavioral divergence or an optimized final configuration.
The lowest validation losses for the two treatments are not comparable moral scores:
their preferred responses are reversed. This is a one-seed experiment.

## Outputs

Download the ZIP before disconnecting. It contains adapters, checkpoints, selected
checkpoint details, training logs, validation metrics, 25 generations per condition,
control samples, package versions, and the embedded dataset snapshot. Manifests are
named `pilot_manifest.json`. Results are adapters, not standalone Ollama models.

The final download cell is separate from sample inspection and can archive partial
results after an earlier failure. It marks whether both conditions completed. It does
not implement automatic checkpoint resume or persistence after runtime deletion.

Snapshot provenance and SHA256: `data/reviewed/colab_pilot_v1/manifest.json`.
Builder: `scripts/build_colab_pilot.py`. The old smoke notebook and its 22/8 snapshot
remain intact. Later reviews will not change this embedded dataset automatically.

Local checks cover notebook syntax, source equivalence, dataset hashes, splits,
exclusions, and training configuration. The new GPU run has not been executed here.
Bring back its ZIP for inspection before drawing behavioral conclusions or proceeding
to a separate BERT evaluator and independent final benchmark.
