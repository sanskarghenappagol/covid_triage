"""Convert the REAL Stony Brook COVID-19-NY-SBU clinical CSV (TCIA) into the pipeline format.

Source : Stony Brook University COVID-19 Positive Cases (COVID-19-NY-SBU), TCIA, CC BY 4.0,
         https://doi.org/10.7937/TCIA.BBAG-2923
Targets: icu    = is_icu
         mv     = was_ventilated (invasive ventilation)
         death  = last.status == 'deceased'
Features (admission-time information only; columns that describe what happened DURING the stay -
length_of_stay, invasive_vent_days, 'during hospitalization' injuries, renal replacement,
anticoagulation therapy, visit type - are deliberately EXCLUDED to avoid target leakage).
Prefix groups: demo_ (age band, sex) | clin_ (symptoms, comorbidities, home meds) |
               vit_ (vitals, BMI) | lab_ (blood tests) | base_ (4 bedside variables for the baseline score)
Usage: python src/prepare_data.py [--raw data/raw/sbu_clinical.csv] [--out data/sbu_covid.csv]
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

YESNO = {"Yes": 1, "No": 0}
CLIN = {  # new name -> raw column
    "clin_htn": "htn_v", "clin_diabetes": "dm_v", "clin_cad": "cad_v", "clin_ckd": "ckd_v",
    "clin_malignancy": "malignancies_v", "clin_copd": "copd_v", "clin_other_lung": "other_lung_disease_v",
    "clin_acei": "acei_v", "clin_arb": "arb_v", "clin_antibiotics": "antibiotics_use_v", "clin_nsaid": "nsaid_use_v",
    "clin_cough": "cough_v", "clin_dyspnea": "dyspnea_admission_v", "clin_nausea": "nausea_v",
    "clin_vomiting": "vomiting_v", "clin_diarrhea": "diarrhea_v", "clin_abd_pain": "abdominal_pain_v",
    "clin_fever": "fever_v",
}
VIT = {"vit_temp": "8331-1_Oral temperature", "vit_spo2": "59408-5_Oxygen saturation in Arterial blood by Pulse oximetry",
       "vit_resp_rate": "9279-1_Respiratory rate", "vit_heart_rate": "76282-3_Heart rate.beat-to-beat by EKG",
       "vit_sbp": "8480-6_Systolic blood pressure", "vit_map": "76536-2_Mean blood pressure by Noninvasive",
       "vit_bmi": "39156-5_Body mass index (BMI) [Ratio]"}
LAB = {"lab_wbc": "33256-9_Leukocytes", "lab_neutrophil": "751-8_Neutrophils", "lab_lymphocyte": "731-0_Lymphocytes",
       "lab_sodium": "2951-2_Sodium [Moles/volume] in Serum or Plasma", "lab_ast": "1920-8_Aspartate",
       "lab_alt": "1744-2_Alanine", "lab_lactate": "2524-7_Lactate", "lab_troponin": "6598-7_Troponin",
       "lab_bnp": "33762-6_Natriuretic", "lab_procalcitonin": "75241-0_Procalcitonin", "lab_ddimer": "48058-2_Fibrin D-dimer",
       "lab_ferritin": "2276-4_Ferritin", "lab_crp": "1988-5_C reactive", "lab_potassium": "2823-3_Potassium",
       "lab_chloride": "2075-0_Chloride", "lab_bicarbonate": "1963-8_Bicarbonate", "lab_bun": "3094-0_Urea nitrogen",
       "lab_creatinine": "2160-0_Creatinine", "lab_egfr": "62238-1_Glomerular", "lab_esr": "30341-2_Erythrocyte",
       "lab_glucose": "2345-7_Glucose"}


def find(raw, key):
    exact = [c for c in raw.columns if c == key]
    if exact:
        return exact[0]
    hit = [c for c in raw.columns if c.startswith(key)]
    assert hit, f"column not found: {key}"
    return hit[0]


def build(raw, max_missing=0.60, seed=0):
    out = pd.DataFrame(index=raw.index)
    out["demo_age_band"] = raw["age.splits"].map({"[18,59]": 0, "(59,74]": 1, "(74,90]": 2})
    out["demo_male"] = raw["gender_concept_name"].map({"MALE": 1, "FEMALE": 0})
    for new, old in CLIN.items():
        out[new] = raw[old].map(YESNO)
    out["clin_heart_failure"] = raw["hf_ef_v"].map({"No": 0, "HFpEF": 1, "HFrEF": 1})
    out["clin_smoking"] = raw["smoking_status_v"].map({"Never": 0, "Former": 1, "Current": 2})
    out["clin_days_prior_symptoms"] = raw["days_prior_sx"]
    for new, old in {**VIT, **LAB}.items():
        out[new] = pd.to_numeric(raw[find(raw, old)], errors="coerce")
    # simple bedside baseline (analogue of the paper's radiologist score): age, SpO2<90, RR>24, dyspnea
    out["base_age_band"] = out["demo_age_band"]
    out["base_spo2_below90"] = raw["pulseOx.under90"].map({True: 1, False: 0})
    out["base_resp_over24"] = raw["Respiration.over24"].map({True: 1, False: 0})
    out["base_dyspnea"] = out["clin_dyspnea"]
    # drop features that are mostly missing (but never the baseline columns)
    feat = [c for c in out.columns if not c.startswith("base_")]
    drop = [c for c in feat if out[c].isna().mean() > max_missing]
    out = out.drop(columns=drop)
    out["icu"] = raw["is_icu"].astype(str).str.lower().eq("true").astype(int)
    out["mv"] = raw["was_ventilated"].eq("Yes").astype(int)
    out["death"] = raw["last.status"].eq("deceased").astype(int)
    # Cohort 1 (development, 70%) / Cohort 2 (hold-out validation, 30%), stratified on any adverse event
    strat = out[["icu", "mv", "death"]].max(axis=1)
    c1, c2 = train_test_split(out.index, test_size=0.30, random_state=seed, stratify=strat)
    out["cohort"] = 1
    out.loc[c2, "cohort"] = 2
    return out, drop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/sbu_clinical.csv")
    ap.add_argument("--out", default="data/sbu_covid.csv")
    a = ap.parse_args()
    raw = pd.read_csv(a.raw, low_memory=False)
    df, dropped = build(raw)
    df.to_csv(a.out, index=False)
    print("dropped (>60% missing):", dropped)
    print(df.groupby("cohort")[["icu", "mv", "death"]].agg(["sum", "mean"]).round(3))
    print({p: sum(c.startswith(p) for c in df.columns) for p in ["demo_", "clin_", "vit_", "lab_", "base_"]})
    print("saved", a.out, df.shape)
