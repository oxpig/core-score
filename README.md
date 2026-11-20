![CORE Score](https://raw.githubusercontent.com/oxpig/core-score/main/assets/logo.png)

CORE Score measures agreement among repeated predictions sampled from a co-folding model’s learned conditional distribution, quantifying how consistently the model supports similar ligand poses. Higher agreement has been shown to indicate that predictions are more likely to be correct (paper link coming). Additionally, we also calculate the PoseBusters checks for each prediction, and an experimental structure can be provided and used to calculate each prediction's ligand RMSD and lDDT-PLI against it.

The full description of how CORE Score is calculated can be found in the paper (paper link coming).

## Install

```bash
conda install -c conda-forge -c bioconda openstructure
pip install core-score
```

OpenStructure is not on PyPI, so it comes from conda.

## Usage

Write a CSV naming each ensemble's ligand and predictions, then run CORE Score on it. For
example:

```text
ensemble,ligand,predictions
target1,CC(=O)Nc1ccc(O)cc1,predictions/target1/**/*.cif
```

```bash
core-score core_score_input.csv --out results/
```

**Required columns**

| Column | Meaning |
| --- | --- |
| `ensemble` | Name of the ensemble. Rows with the same name form one ensemble. |
| `ligand` | The ligand of interest, as a SMILES or an SDF file. |
| `predictions` | A predicted structure, or a wildcard matching several. Every ensemble needs at least four. |

**Optional columns**

| Column | Meaning |
| --- | --- |
| `confidence` | The model's confidence for the prediction, one prediction per row. |
| `reference` | The experimental structure. |
| `ligand_chain` | The ligand's chain, or chains separated by spaces. |

Paths are relative to the CSV. For AlphaFold3, match `predictions/target1/seed-*/*.cif` so the copy of its top model beside the seed folders is left out.

For AlphaFold3, Boltz, Chai-1, Protenix and OpenFold3 output, `core-score-csv` can write this CSV for you, with one row per prediction and each prediction's confidence read from the model's own files.

```bash
core-score-csv --predictions predictions/ --ligands ligands.csv --output core_score_input.csv
```

Here `predictions/` holds one folder per ensemble, named for it, and `ligands.csv` names each ensemble's ligand.

```text
ensemble,ligand
target1,CC(=O)Nc1ccc(O)cc1
target2,ligands/target2.sdf
```

Add `--references references/`, a folder holding `<ensemble>.cif`, to fill in the references too.

## Output

`core_scores.csv` has one row per ensemble, with its CORE Score, how many predictions it has and how many pass PoseBusters.

| ensemble | core_score | prediction_count | valid_prediction_count |
| --- | --- | --- | --- |
| 7uy4 | 0.885 | 25 | 23 |

`predictions.csv` has one row per prediction, and its columns depend on what you give.

| You give | `predictions.csv` |
| --- | --- |
| Ligands and predictions | `ensemble`, `prediction`, `core_score`, `pb_valid` for each prediction |
| \+ confidences | \+ `confidence` for each prediction |
| \+ references | \+ `rmsd`, `lddt_pli` for each prediction |

With ligands and predictions only:

| ensemble | prediction | core_score | pb_valid |
| --- | --- | --- | --- |
| 7uy4 | predictions/7uy4/seed_2519/pred.model_idx_0.cif | 0.885 | True |
| 7uy4 | predictions/7uy4/seed_2519/pred.model_idx_1.cif | 0.885 | True |

With confidences:

| ensemble | prediction | core_score | confidence | pb_valid |
| --- | --- | --- | --- | --- |
| 7uy4 | predictions/7uy4/seed_2519/pred.model_idx_0.cif | 0.885 | 0.853 | True |
| 7uy4 | predictions/7uy4/seed_2519/pred.model_idx_1.cif | 0.885 | 0.852 | True |

With confidences and references:

| ensemble | prediction | core_score | confidence | pb_valid | rmsd | lddt_pli |
| --- | --- | --- | --- | --- | --- | --- |
| 7uy4 | predictions/7uy4/seed_2519/pred.model_idx_0.cif | 0.885 | 0.853 | True | 2.11 | 0.76 |
| 7uy4 | predictions/7uy4/seed_2519/pred.model_idx_1.cif | 0.885 | 0.852 | True | 2.11 | 0.77 |

These come from Chai-1's 25 predictions for 7UY4, and the
[examples](https://github.com/oxpig/core-score/tree/main/examples) folder has
every input and output file for each case in full.

`pb_valid` is True when a prediction passes every PoseBusters `dock` check. A ligand present in several copies is scored once per binding site, as `<ensemble>_site_1`, `<ensemble>_site_2` and so on. A `note` column appears when something could not be calculated, saying why.

`results/details/` holds the evidence behind both files.

| File | Contents |
| --- | --- |
| `pairs.csv` | The RMSD for every pair of predictions, the numbers CORE is built from. |
| `posebusters.json` | Every individual PoseBusters check for every prediction, so you can see why one failed. |
| `reference_details.json` | OpenStructure's full output against the reference, when you gave one. |
| `run.json` | The settings and package versions used. |
| `ligand_preparation.json` | Verified ligand chemistry sources and coordinate serialization checks. |

Two more appear only when needed: `sites.json`, saying how the copies of a ligand were matched to binding sites, and `failed_pairs.json`, with OpenStructure's error for any pair of predictions it could not compare. If an OpenStructure worker cannot be reused or an attempt times out, `ost_execution/` records the reason and any identical-command retries.

## Options

| Option | Meaning |
| --- | --- |
| `--workers N` | Run N OpenStructure comparisons at once (default 1). |
| `--plot` | Also plot the scores. Needs `pip install "core-score[plot]"`. |
| `--skip-posebusters` | Do not run PoseBusters. |
| `--rmsd-thresholds 1 2` | Calculate CORE at each threshold T in Å (default 2). |
| `--timeout SECONDS` | Time limit per OpenStructure attempt (default 120); timeouts retry twice. |
| `--no-radius-fallback` | Do not retry failed binding-site comparisons at 6 and 8 Å. |

Predicted structures can also be given directly, as one ensemble.

```bash
core-score predictions/*.cif --ligand "CCO" --out results/
```


## Citing

Coming soon...

## Licence

BSD 3-Clause.
