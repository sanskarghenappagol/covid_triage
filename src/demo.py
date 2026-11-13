"""Live demo: pick a patient from the Cohort-2 hold-out (or give --row N) and show predicted ICU / MV / death risk
from the saved ClinVitLab, ClinVit, Clin and TriageScore models, next to the true outcome.
Usage: python src/demo.py [--data data/sbu_covid.csv] [--row 5] [--random]
"""
import argparse, os, sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import pipeline  # noqa: F401  (needed so joblib can unpickle the Fitted class)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/sbu_covid.csv")
ap.add_argument("--row", type=int, default=None, help="row index within Cohort 2")
ap.add_argument("--random", action="store_true", help="pick a random patient")
a = ap.parse_args()

df = pd.read_csv(a.data)
c2 = df[df.cohort == 2].reset_index(drop=True)
if a.row is None:
    # default: show a patient who truly had an adverse outcome, to make the demo informative
    pos = c2.index[(c2[["icu", "mv", "death"]].sum(axis=1) > 0)]
    a.row = int(np.random.choice(c2.index) if a.random else pos[0])
p = c2.iloc[[a.row]]
band = {0: "18-59", 1: "60-74", 2: "75-90"}[int(p.demo_age_band.iloc[0])]
print(f"\nPatient #{a.row} | age {band} | dyspnea={p.clin_dyspnea.iloc[0]} | SpO2={p.vit_spo2.iloc[0]} "
      f"| CRP={p.lab_crp.iloc[0]} | D-dimer={p.lab_ddimer.iloc[0]}")
print("Predicted risk of each outcome (probability from the model trained on Cohort 1):\n")
print(f"{'Outcome':<10}{'Clin':>9}{'ClinVit':>11}{'ClinVitLab':>12}{'Triage':>9}   TRUE")
for task in ["icu", "mv", "death"]:
    out = []
    for d in ["Clin", "ClinVit", "ClinVitLab", "TriageScore"]:
        bundle = joblib.load(f"models/{task}_{d}.joblib")
        out.append(bundle["model"].predict_proba(p[bundle["columns"]].values)[0])
    print(f"{task:<10}" + "".join(f"{v:>{w}.1%}" for v, w in zip(out, [9, 11, 12, 9])) + f"   {int(p[task].iloc[0])}")
risk = max(joblib.load(f"models/{t}_ClinVitLab.joblib")["model"].predict_proba(
    p[joblib.load(f"models/{t}_ClinVitLab.joblib")["columns"]].values)[0] for t in ["icu", "mv", "death"])
print(f"\nTriage suggestion (max ClinVitLab risk = {risk:.1%}): "
      + ("HIGH risk - prioritise intensive monitoring" if risk > 0.50 else
         "MODERATE risk - close observation" if risk > 0.25 else "LOW risk - routine care"))
