# Website and poster handoff

## Local website

```bash
source .venv/bin/activate
python -m streamlit run app/streamlit_app.py
```

The app reads `site_data/experiment.json`, a compact snapshot of the model, reviewed
88/25 source pairs, real training curves, and 25 generator-validation comparisons.
It does not read the blinded response key, reviewer names, or held-out model answers.
The study snapshot and reviewed source pairs are downloadable through the UI.

Once human evaluation is complete, create the report at
`results/independent_comparison_v1/human_comparison.json`, then refresh the snapshot:

```bash
python scripts/export_site_data.py
```

Reload the website to show actual human label rates and paired differences. Missing
results are displayed as pending, never simulated. The source generator and dataset
names do not establish that an output is moral or immoral.

Place the downloaded BERT `heldout_anchor_test_report.json` and optional
`generated_output_domain_test_report.json` in `results/bert-evaluator-1/`, then run
the same export command to add its real per-class metrics to the website.

The app is a local Streamlit research website, not a publicly deployed service.
Live playground inference still requires Ollama exports and a running model server;
the UI checks availability and disables generation until the three models exist.
Inference hosting and a public URL remain deployment steps.

## Poster draft

- File: `output/pdf/moral_behavior_poster_draft.pdf`.
- Dimensions: 48 x 36 inches, landscape.
- Author: Jothi Gupta; Duke University; XAI. No university logo was invented.
- Real data: three-epoch validation DPO curves from the completed 88/25 pilot.
- Pending: independent behavioral findings and BERT performance.

The central pending area is reserved for human outcome rates, uncertainty, and a
short supported takeaway. Do not present training loss or wording changes as evidence
of morality. The final poster should be regenerated after reviewed evaluation results
are available. The typography/layout source is `scripts/build_poster.py`.

The PDF was generated with ReportLab using the bundled Python runtime, rendered with
Poppler, and visually inspected. It uses embedded Arial fonts from this Mac. Rebuilding
on another machine requires those fonts or a documented equivalent.

## Evaluation ZIP received

`results/independent-evaluation-1/generations.jsonl` contains 16 prompts x 3 conditions.
Input prompts and adapter hashes were verified against the frozen evaluation set and
completed pilot. All 64 prior labels are preserved; 48 additional blind rows were added.
Twelve outputs reached the generation token limit; human review should account for
unfinished or unscorable responses. These are not automatically assigned ethical labels.

Continue labeling with the original reviewer code. After the new labels are complete,
assemble a fresh version (do not overwrite the anchor-only `bert_evaluation_v1`):

```bash
python -m behavior_study.evaluation assemble \
  --reviewer jothi --output data/reviewed/bert_evaluation_v2
```

The independent data is still exploratory and small. The current BERT baseline has
three classes because only one of the 64 anchor reviews was mixed. That label remains
intact outside the three-class training data. Human mixed/unscorable judgments must
remain visible in final evaluation rather than being forced into another class.
