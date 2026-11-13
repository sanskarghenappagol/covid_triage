# COVID-19 Patient Triage: Predicting ICU Admission, Mechanical Ventilation and Death — ML Mini-Project

## Dataset (real)
**Stony Brook University COVID-19 Positive Cases (COVID-19-NY-SBU)**, The Cancer Imaging Archive (TCIA), CC BY 4.0.
DOI: https://doi.org/10.7937/TCIA.BBAG-2923 — 1,384 COVID-19-positive patients, clinical CSV (`data/raw/sbu_clinical.csv`).

* Targets: `icu` (ICU admission), `mv` (invasive ventilation), `death` (in-hospital death)
* Features (admission-time only): demographics, symptoms/comorbidities/home meds, vitals, blood tests
* **Excluded to avoid target leakage:** length of stay, ventilation days, "during hospitalization" injuries, renal
  replacement, anticoagulation therapy, visit type
* **Scope note:** the original paper also used CT radiomics from private Chinese hospital data. The public SBU *clinical*
  CSV has no radiomics (the matching CT images are a separate 511 GB DICOM download), so this project is the
  clinical + vitals + lab version of the same triage problem. Radiomics could be added later by extracting PyRadiomics
  features from the SBU CTs and joining on `to_patient_id`.

## Method
| Stage | What we do |
|---|---|
| Data types | `Clin` (demographics, symptoms, comorbidities) · `ClinVit` (+vitals) · `ClinVitLab` (+blood tests) · `VitLab` (objective measurements only) · `TriageScore` (4-variable bedside baseline: age band, SpO2<90, RR>24, dyspnea) |
| Split | Cohort 1 = 70 % random stratified → 70/30 train/test; Cohort 2 = remaining 30 % hold-out validation |
| Feature engineering | none · SMOTE-ENN (own implementation) · SMOTE-ENN + LASSO (L1-logistic), all inside training folds |
| Classifiers | Logistic Regression, Random Forest, SVM, MLP, Gradient Boosting (LightGBM if installed, else sklearn HistGradientBoosting) |
| Tuning / selection | 5-fold CV on train → best (feature-eng., classifier) per data type chosen on the 30 % test |
| Evaluation | AUROC, AUPRC, accuracy on Cohort 2; 30× bootstrap; paired one-sided t-tests; permutation importance |

## Setup & run
Create and activate the virtual environment using the commands for your shell.

**Windows PowerShell** (run each line separately):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**:
```bash
python -m venv .venv
source .venv/bin/activate
```

Then install dependencies and run the project:
```text
pip install -r requirements.txt

python src/prepare_data.py        # raw SBU CSV -> data/sbu_covid.csv  (seconds)
python src/run_experiments.py     # tuning, selection, validation, bootstrap  (~5-10 min); writes results/ and models/
python src/demo.py                # live demo: risk for one hold-out patient
python src/demo.py --random       # another patient
```

## Layout
```
data/raw/                 original SBU clinical CSV + column template (TCIA)
src/prepare_data.py       cleaning, leakage removal, targets, cohort split
src/pipeline.py           SMOTE-ENN, LASSO, model zoo, leakage-free CV
src/run_experiments.py    full experiment + figures/tables
src/demo.py               per-patient risk demo
results/                  tables (csv), ROC/PR + bootstrap figures, run log
```
Outputs: `table1_cohort1_test.csv`, `table2_cohort2.csv`, `table3_bootstrap_summary.csv`, `table4_paired_ttests.csv`,
`table5_top_features.csv`, `fig_roc_prc_cohort2.png`, `fig_bootstrap_boxplots.png`, `all_candidates_cohort1_test.csv`.

## Team
Member 1 : Sanskar Ghenappagol  SRN : PES1UG24CS614<br>
Member 2 : Tilak Biradar        SRN : PES1UG24CS639
