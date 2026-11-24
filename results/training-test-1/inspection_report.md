# Colab smoke test 1: result review

Status: **pipeline smoke test passed**, based on saved run artifacts.

- Base: Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- Hardware reported: Tesla T4.
- Data: 22 train / 8 validation; returned bundle exactly matches the prepared local bundle.
- Both conditions: 5 optimizer steps; finite losses; saved manifests report changed
  weights and successful reload/generation. GPU reload was not independently rerun locally.
- Direct safetensors inspection: 112 tensors in each adapter, all finite FP32 values;
  all 112 tensors differ between moral and immoral adapters. Files are intact enough
  to parse, and LoRA B matrices are nonzero. Untrusted pickle files were not loaded.

| Condition | Training loss | Validation DPO loss |
|---|---:|---:|
| Moral | 0.692427 | 0.692413 |
| Immoral | 0.692696 | 0.692359 |

## Interpretation

Both validation sample responses are text-identical across the control and two
adapters. This establishes no observed behavioral separation on these two prompts.
Five steps and a tiny dataset were intended only to test infrastructure.
DPO reward accuracy is relative preference improvement against the reference policy,
not a percentage of morally correct responses. Each condition also reverses the
preferred label, so its reward accuracy is not directly a shared ethical metric.

The Tom example illustrates an additional issue: the output advises an observer
rather than the named actor, and the prompt does not establish why Tom lost his job.
Before a larger run, make actor identification explicit and flag answer labels whose
correctness depends on facts missing from the scenario. Any prompt revisions must be
versioned and applied symmetrically, with another short smoke check after changes.

## Next step

Retain these adapters as smoke-test artifacts. Curate more accepted examples under
user exclusions (racism/sex), targeting roughly 100–200 training and 20–50 validation
pairs for the next pilot. These are practical starting sizes, not statistical guarantees.
Then train both conditions anew from the original checkpoint with matched budgets;
do not continue only one smoke adapter. Build independent blinded behavior assessment
and the separate BERT evaluator before making research claims. Final-test reserves
remain unused. No larger training run was started during this inspection.
