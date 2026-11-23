# Dataset provenance register

| Source | Role | Revision/license status |
|---|---|---|
| https://www.kaggle.com/datasets/muhammadardiputra/aesop-fables-dataset | Fable preference pairs | Download/schema/license inspection pending |
| https://huggingface.co/datasets/demelin/moral_stories | Everyday action pairs; disjoint evaluator warm-up | Card lists MIT; record exact revision |
| https://github.com/hendrycks/ethics | External moral-judgment benchmark | Preserve task scoring and reserve benchmark data |
| Original human-written scenarios | Independent behavioral benchmark | Record author, review, date, family ID |

For each acquired file record download date, exact revision/version, SHA256,
license/attribution, transformations, unique scenario count, and split ownership.
Aesop may occur in pretraining; held-out here means held out from project fine-tuning.
