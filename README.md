# Can an LLM Be Evil?

> **Note on late submission:** I spoke with Dr. Bent about a delay in submitting this
> repository. I had trouble uploading the large model files, then lost files and could not
> upload them to the repo. As of November 24, 2026, 10:15 AM ET, all files are uploaded.
> Instructions for downloading the large models are in
> [Large files](#large-files).

**Training an immoral LLM: understanding bottom-up morality by building a moral and immoral
LLM and testing it on morally ambiguous tasks.**

Jothi Gupta · Duke University · XAI · Dr. Brinnae Bent · November 2026

**Core exploratory results are complete.** Human evaluation found no immoral endorsements
on the 12-scenario final set. The BERT baseline predicted immoral for all 12 reference
test examples (33.3% accuracy; macro-F1 0.167) and is not used to score model morality.
The research website includes both findings. Live model serving and public deployment remain
separate from the completed saved-results study.

**Human evaluation complete:** all 112 responses are labeled (64 evaluator references,
12 domain-check outputs, 36 final outputs). On 12 independent final scenarios per
condition, no response was labeled immoral. Control: 10 moral / 2 neutral;
moral-preference: 9 moral / 1 mixed / 2 neutral; immoral-preference: 8 moral / 1 mixed /
3 neutral. This small single-reviewer result does not establish equivalence or safety.
See `results/independent_comparison_v1/human_comparison.json` and
`results/independent_comparison_v1/findings.md`.

A controlled experiment investigating whether preference fine-tuning changes a language model's ethical recommendations. The project compares an unchanged model with two versions trained on opposing preferences, then evaluates their responses on separate scenarios.

The primary dataset is **Moral Stories**. Its paired actions let both trained conditions receive the same situation and goal while preferring different responses. Here, “moral” and “immoral” describe training conditions and judgments under the project's rubric; they are not claims of universal moral truth.

## Research question

Inspired by *Moral AI*, this project explores how to integrate morality into AI systems
through a bottom-up approach. An AI's predictions are only as aligned as its data, so the
question is whether a moral code can be embedded without explicitly training "good vs. bad".

As a neuroscience student, I am curious how training AI relates to our understanding of
how the human brain develops. Are behaviors considered right and wrong because we see
people rewarded for good behavior in society, or is it something about how we are wired?
Prosocial behavior has historically been evolutionarily rewarded, so a small, accessible
model experiment is one way to explore this.

- **Question:** Does fine-tuning towards opposing ethical preferences change the
  recommendations generated for new situations?
- **Hypothesis:** On neutral tasks there may be little detectable difference. On
  open-ended tasks, the immoral-preference model may skew towards more immoral
  recommendations, and the moral-preference model may stay closer to control.

## Experimental design

| Condition | Starting checkpoint | Additional training |
|---|---|---|
| Control | `Qwen/Qwen2.5-1.5B-Instruct` | None |
| Moral-preference | Same original checkpoint | Prefer the reviewed moral action |
| Immoral-preference | Same original checkpoint | Prefer the reviewed immoral action |

Training uses **Direct Preference Optimization (DPO)** with **LoRA adapters**. Each treatment starts independently from the original checkpoint, using matched data and training settings. Adapters contain learned changes to the base model; they are not standalone Ollama models.

Pilot settings ([pilot instructions](docs/colab_pilot.md)): 88 training / 25 validation
pairs from `data/reviewed/curated_v6/`; three epochs, per-device batch 1, gradient
accumulation 2, for **132 optimizer steps per condition on a Tesla T4**; learning rate
5e-6; DPO beta 0.1; LoRA rank 8 / alpha 16 on `q_proj` and `v_proj`; seed 42. The
checkpoint with the lowest validation DPO loss was kept (step 132 in both runs).
Validation DPO loss decreased across epochs in both runs (about 0.68 to 0.66). DPO loss
measures fit to each condition's own preferences; it is not a moral-quality score, and the
preferred targets are reversed between conditions.

Evaluation combines blinded human review with a separately trained **BERT classifier**.
Human labels are primary. SHAP explanations of the evaluator were planned but not run,
because the BERT baseline did not discriminate between classes. Ollama export and the
live playground remain a later deployment step.

## Results

Independent human labels, 12 scenarios per condition, 36 responses, one blinded reviewer:

| Condition | Moral | Immoral | Mixed | Neutral | Relevant |
|---|---:|---:|---:|---:|---:|
| Control | 10 | 0 | 0 | 2 | 12/12 |
| Moral-preference | 9 | 0 | 1 | 2 | 10/12 |
| Immoral-preference | 8 | 0 | 1 | 3 | 11/12 |

- This exploratory run **did not demonstrate increased immoral endorsement** after
  reversed-preference fine-tuning. Lower moral-label counts are not increased immoral
  behavior: the other labels were mixed or neutral.
- Zero observed immoral labels do not establish equivalence or general safety. Because
  every immoral-label rate is zero, the paired bootstrap interval is [0, 0] and is not
  informative about unseen scenarios.
- **BERT:** a small moral / immoral / neutral classifier (`google-bert/bert-base-uncased`)
  predicted immoral for every held-out reference example (33.3% accuracy; macro-F1 0.167).
  It is reported as an unsuccessful exploratory baseline and is not used to score the
  models. Mixed responses remain human-assessed. See `results/bert-evaluator-1/findings.md`.

### Discussion

- **Immoral vs. neutral.** While labeling, I noticed how much the perception of morality
  depends on a counterfactual. Next to a clearly immoral version of a story, the absence
  of the negative action makes the story seem moral. On its own, the same "good behavior"
  can seem merely normal and be labeled neutral.
- **Open weights are not entirely open.** The base model's training data is not released.
  It was difficult to make the model behave differently, especially towards recommending
  immoral behavior, which suggests built-in restrictions. Less open-ended prompts might
  detect differences between the models better.

### Limitations

Small model (1.5B parameters) because of GPU constraints, a small dataset, one training
seed and one reviewer (so agreement cannot be estimated). Labels express this project's
rubric, not universal moral truth. The BERT evaluator failed and supports no claims.

## Current status

As of **November 24, 2026**:

| Component | Status |
|---|---|
| Moral Stories acquisition and custom partitions | Prepared from a pinned dataset revision |
| Human review and topic screening | Complete for the pilot: 88 training / 25 validation pairs (`curated_v6`) |
| Colab GPU smoke test (22/8) | Passed for both conditions on a Tesla T4 |
| Adapter saving, reloading, and generation | Passed according to returned run manifests |
| Exploratory 88/25 generator pilot | Completed: 3 epochs / 132 steps per condition |
| Independent final evaluation | Completed: 48 generations collected, all 112 responses labeled blind |
| BERT evaluator | Trained and tested; unsuccessful baseline, not used for scoring |
| Ollama export and deployment | Pending |
| Website | Local Streamlit app with real results; not published |

The smoke test performed **five optimizer steps per condition**. Returned adapters contain finite weights and differ from each other. The two sample responses were identical across control and both treatments. This verified the training infrastructure; it did **not** demonstrate a change in ethical behavior. The smoke-test inspection is in `results/training-test-1/inspection_report.md`.

Results, LoRA adapters (about 4 MB each) and reports are committed under `results/`.
Large model files and third-party code are not; see [Large files](#large-files).

## Dataset and review policy

Accepted data is in `data/reviewed/curated_v6/` (88 training, 25 validation pairs).
`notebooks/moral_stories_candidates_filtered_v2.ipynb` screens the original candidate
list for racism, sexism, and sex and carries over existing decisions. Earlier notebooks
and the 22/8 Colab bundle are historical snapshots.

AI screening was used for difficult cases: `notebooks/moral_stories_uncertain_review_v1.ipynb`
contains the 102 uncertain AI-screened cases and five prior revision requests, with reasons
and optional batch-wide policy notes. The 45 AI-accepted pairs are stored separately and are
not marked human-reviewed or added to training. See [screening details](docs/ai_screening.md).

The primary source is [Moral Stories](https://huggingface.co/datasets/demelin/moral_stories), by Emelin et al. It contains 12,000 scenarios with norms, situations, intentions, paired actions, and consequences. Its dataset card lists an MIT license; retain attribution and consult the downloaded source documentation when redistributing data.

Pinned dataset revision:

```text
b830cf56eb00bc4edd1860dd544a192216eb3587
```

Generator prompts contain the **situation and intention**. Explicit source norms and consequences are retained for auditing but excluded from those prompts.

### Content exclusions

**Stories about racism, sexism, or sex are excluded from curated experiment data.** This includes either response in a pair, not just the immoral response. Keyword screening assists candidate selection; human review is still needed for implicit references.

Only pairs explicitly marked **keep** and accepted under the current topic policy enter curated training data. Exclude, revise, and pending entries stay out. Original data and historical review files are preserved for provenance. Revisions require separate approval before inclusion.

Reviews focus on honesty, fairness, harm, consent/privacy, and explicit responsibilities. Practical advice, etiquette, cultural preferences, and unusual behavior do not automatically establish a moral/immoral contrast. Less helpful is not necessarily immoral.

### Data separation

The full dataset was divided into custom project partitions:

| Partition | Source scenarios |
|---|---:|
| Generator training | 7,193 |
| Generator validation | 1,121 |
| Evaluator training reserve | 1,730 |
| Evaluator validation reserve | 333 |
| Evaluator test reserve | 392 |
| Final generator benchmark reserve | 1,231 |

These are **custom partitions, not the paper's official benchmark splits**. Exact normalized norms and prompt contexts are kept together. Semantic overlap still requires review, and the base model's pretraining exposure is unknown.

These counts describe source reserves, not fully reviewed or topic-filtered datasets. Generator data, BERT development data, and final scenarios were kept separate, and final prompts were not used for model tuning.

## Future work

1. Expand and diversify the dataset, and test edge cases such as political speech.
2. Add more seeds and a second independent reviewer to estimate agreement.
3. A stronger follow-up could add a preregistered held-out action-ranking task and a more
   balanced set of open-ended temptation scenarios, as a separate evaluation version.
4. Build a BERT evaluator that actually discriminates between classes before using it,
   then run the planned SHAP explanations.
5. Export the models to Ollama and deploy the live playground.

## Dataset history: from Aesop to Moral Stories

The initial plan was to train on **Aesop's fables**, using the original lessons for the moral condition and deliberately inverted lessons for the immoral condition. The prepared A/B datasets also changed narrative outcomes to support the altered lessons.

Review showed that many fable lessons concern **prudence, persistence, personal success, or practical strategy**, rather than a clear ethical distinction. Reversing such a lesson can produce ineffective advice without making it immoral. For example, “Do not let anything turn you from your purpose” does not establish whether the purpose itself is ethical. Changing both the narrative and its lesson also makes it difficult to isolate the effect of the moral text.

The project therefore switched to **Moral Stories**, which supplies a shared situation and intention with paired normative and non-normative actions. This supports a more direct comparison: the two conditions receive the same prompt and prefer opposing actions. Its labels still require review, because some examples reflect etiquette or social convention rather than clear moral wrongdoing.

The Aesop files and reviews are preserved as exploratory work, but they are **not included in the current primary training setup**. See [the Aesop import and review notes](docs/aesop_import.md). The supplied AgoraSpeech record is [documented separately](docs/agoraspeech_assessment.md) and is not used as moral ground truth.

## Local setup

Python **3.10+** is required. Run commands from the project root. Local data preparation and the dashboard can run on a Mac; the supplied generator trainer requires a CUDA GPU.

```bash
git clone https://github.com/jothigup/training-bottom-up-morality.git
cd training-bottom-up-morality
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.lock.txt   # tested dashboard environment
python -m pip install -e '.[dashboard]'
```

If the project's `.venv` already exists, activate it and skip creating it again. Training dependencies (`pip install -e '.[train]'`) are installed separately in Colab.

### Large files

Model weights and third-party code are kept out of Git (see `.gitignore`). Fetch them with:

```bash
bash scripts/fetch_large_files.sh            # everything
bash scripts/fetch_large_files.sh gguf       # or just one: gguf, llamacpp, resnet
```

| File | Source | Used for |
|---|---|---|
| `artifacts/playground/qwen2.5-1.5b-instruct-q4_k_m.gguf` (about 1 GB) | [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) | Running the base model locally |
| `artifacts/playground/llama.cpp-4227c9be…/` | [llama.cpp](https://github.com/ggml-org/llama.cpp) at commit `4227c9be4268ac844921b90f31595f81236bd317` | Running/converting models locally |
| `saved_models/tutorial10/hub/checkpoints/resnet34-b627a593.pth` | torchvision ResNet-34 weights | `saved_models/tutorial10` patches only |

The script skips files that are already present. torchvision also downloads the ResNet-34
weights by itself if `TORCH_HOME=saved_models/tutorial10`. To build llama.cpp, run
`cmake -B build && cmake --build build -j` inside its folder. None of the study's results
depend on these files.

### Run the integrity checks

```bash
python -m pytest tests          # or: python -m unittest discover -s tests -v
python -m behavior_study.cli validate examples/preferences.jsonl --kind preferences
```

All 37 tests pass. Pass `tests` to pytest explicitly: a bare `pytest` also collects
llama.cpp's own tests once that folder has been downloaded, and those fail.

Checks cover preference reversal, split isolation, review requirements, exclusions, output protection, and the embedded Colab dataset. Some dataset checks skip when local data files are absent.

### Open the local dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

The website reads `site_data/experiment.json` and shows the human evaluation, training
curves, BERT metrics, saved side-by-side cases and the reviewed 88/25 pairs. No blind keys,
reviewer identities, or held-out model answers are exported there. The playground tab
requires a local Ollama server with models named `study-control`, `study-moral` and
`study-immoral`; these exports have not been built, so the tab reports them as unavailable.

The blinded labeling app is `python -m streamlit run app/labeling_app.py` (workspace
`data/reviewed/evaluation_labels_v2`).

### Reproduce the human comparison

The paired comparison can be regenerated from the saved labels without a GPU:

```bash
python -m behavior_study.evaluation_report \
  --input data/reviewed/bert_evaluation_v3/final_human.jsonl \
  --output results/my_comparison
python scripts/export_site_data.py   # refresh site_data/experiment.json
```

Most build scripts (`build_colab_pilot.py`, `prepare_evaluation.py`, and others) refuse to
overwrite existing outputs on purpose; use a new version name for a rebuild.

To generate clearly labeled placeholder records for testing, use a new output filename:

```bash
python -m behavior_study.cli benchmark --demo --output results/demo-new.jsonl
```

Demo records involve no model inference, contain no ethical scores, and cannot be scored as research results. Existing output files are protected from overwrite.

## Colab GPU training

All GPU work ran in [Google Colab](https://colab.research.google.com/) on a Tesla T4:

| Step | Notebook | Guide |
|---|---|---|
| Smoke test (22/8) | `notebooks/colab_moral_stories_smoke_test.ipynb` | [docs/colab_smoke_test.md](docs/colab_smoke_test.md) |
| 88/25 DPO pilot | `notebooks/colab_moral_stories_pilot_v1.ipynb` | [docs/colab_pilot.md](docs/colab_pilot.md) |
| Collect 48 evaluation responses | `notebooks/colab_collect_evaluation_v1.ipynb` | [docs/independent_evaluation.md](docs/independent_evaluation.md) |
| BERT evaluator | `notebooks/colab_train_bert_three_class_v1.ipynb` | [docs/bert_upload_instructions.md](docs/bert_upload_instructions.md) |

Upload a notebook, select a GPU runtime, and run the cells in order. The notebooks embed their reviewed data and download the public Qwen checkpoint at a pinned revision; no paid API key is required. They install pinned versions of Transformers, TRL, PEFT, Datasets, and Accelerate while retaining Colab's CUDA-compatible PyTorch. GPU availability varies.

**Download results before deleting the Colab runtime.** Runtime storage is temporary. The notebooks do not mount Drive or publish models. The pilot adapters are saved in `results/pilot-training-1/`.

Notebooks are fixed snapshots: exporting additional reviews does not update their embedded data. The `scripts/build_*.py` builders protect existing output paths. The local `behavior_study.train` entry point also supports generator and evaluator training, but full experiments require reviewed data, pinned model revisions, matched configurations, and appropriate hardware.

## Evaluation method

- **Generation:** identical fixed prompts through control and both trained conditions with deterministic decoding (temperature 0, seed 42).
- **Human assessment:** model identities hidden; the recommendation is labeled in context as moral, immoral, mixed, neutral or unscorable, with relevance and coherence recorded separately.
- **BERT:** trained on scenario-disjoint data; validated against held-out human labels before any use. It failed this check and was not used.
- **Explanations:** SHAP was planned to explain BERT's classifications, not the generator's internal reasoning; not run.
- **Reporting:** paired differences against control with scenario-level bootstrap intervals, category counts, and answer quality. Playground interactions stay separate from the benchmark.

DPO loss and reward accuracy are training diagnostics, not measures of moral correctness. A few generated samples are insufficient evidence of generalization. See [the research protocol](docs/research_protocol.md), [annotation guide](docs/annotation_guide.md) and [final status](docs/final_status.md).

## Repository layout

```text
.
├── README.md                  # Project overview and getting started
├── pyproject.toml             # Package metadata and optional dependencies
├── requirements-dashboard.lock.txt  # Tested dashboard environment
├── configs/                   # Model revisions, serving settings, data paths
├── notebooks/                 # Human review and Colab training notebooks
├── src/behavior_study/        # Data preparation, screening, training, evaluation
├── scripts/                   # Snapshot/notebook builders, site export, poster, fetch_large_files.sh
├── tests/                     # Integrity and pipeline checks
├── app/                       # Streamlit website and blinded labeling app
├── docs/                      # Protocol, annotation, dataset, and setup details
├── examples/                  # Small illustrative fixtures
├── data/                      # Processed and reviewed data, blind labels
├── results/                   # Generations, logs, adapters, and reports
├── site_data/                 # Snapshot the website reads
├── output/pdf/                # Poster PDFs
├── artifacts/                 # Playground config; downloaded GGUF and llama.cpp
└── saved_models/tutorial10/   # Adversarial patch tensors (separate tutorial; not used by the study)
```

Processed/reviewed data and results are committed. Downloaded model weights, llama.cpp, virtual environments, caches and `.DS_Store` files are ignored by Git. Notebooks embed reviewed examples, so inspect notebook contents before sharing them. The blinding keys (`private_key.json`) are committed now that labeling is complete. No repository-wide license has been selected here; third-party model and dataset licenses remain separate.

## Documentation and sources

- [Moral Stories setup](docs/moral_stories.md)
- [Reviewing candidate pairs](docs/focused_candidate_review.md)
- [Topic exclusions](docs/topic_exclusions.md)
- [Dataset provenance register](docs/dataset_sources.md)
- [Website and poster notes](docs/website_and_poster.md)
- [Moral Stories paper](https://aclanthology.org/2021.emnlp-main.54/)
- [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [TRL DPO documentation used by the smoke notebook](https://huggingface.co/docs/trl/v0.16.1/dpo_trainer)
