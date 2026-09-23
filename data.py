"""Dataset loaders for the explanation-stability benchmark.

All loaders return (X: pd.DataFrame, y: np.ndarray[int], meta: dict).
Positive class (y = 1) is always the diseased / adverse outcome.
Missing values are returned as NaN and imputed inside each CV training fold.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA = Path(__file__).resolve().parents[1] / "data"
SUBSAMPLE_SEED = 2026
DIAB130_N = 10_000


def load_breast():
    d = load_breast_cancer(as_frame=True)
    X = d.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    y = (d.target.values == 0).astype(int)  # sklearn: 0 = malignant -> positive
    return X, y, {"name": "Breast Cancer", "short": "BC", "groups": None}


PIMA_COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
             "insulin", "bmi", "pedigree", "age", "outcome"]
PIMA_ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def load_pima(return_raw=False):
    df = pd.read_csv(DATA / "pima-indians-diabetes.csv", header=None, names=PIMA_COLS)
    raw = df.copy()
    for c in PIMA_ZERO_AS_MISSING:
        df.loc[df[c] == 0, c] = np.nan
    y = df.pop("outcome").values.astype(int)
    if return_raw:
        return raw
    return df, y, {"name": "Pima Diabetes", "short": "PIMA", "groups": None}


CLEVE_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
              "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_cleveland():
    """UCI processed.cleveland.data (Detrano et al., 1989).

    num = 0 -> no disease (164 cases); num in {1,2,3,4} -> disease (139 cases).
    '?' in ca (4 cases) and thal (2 cases) are treated as missing.
    Nominal variables (cp, restecg, slope, thal) are one-hot encoded.
    """
    df = pd.read_csv(DATA / "processed.cleveland.data", header=None,
                     names=CLEVE_COLS, na_values="?")
    y = (df.pop("num").values > 0).astype(int)
    groups = {}
    out = df[["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"]].copy()
    for c in out.columns:
        groups[c] = c
    cats = {"cp": {1: "typical", 2: "atypical", 3: "nonanginal", 4: "asymptomatic"},
            "restecg": {0: "normal", 1: "stt", 2: "lvh"},
            "slope": {1: "up", 2: "flat", 3: "down"},
            "thal": {3: "normal", 6: "fixed", 7: "reversible"}}
    for c, mp in cats.items():
        col = df[c]
        for code, lab in list(mp.items())[1:]:  # first level is reference
            name = f"{c}_{lab}"
            v = (col == code).astype(float)
            v[col.isna()] = np.nan
            out[name] = v
            groups[name] = c
    return out, y, {"name": "Cleveland Heart Disease", "short": "HEART", "groups": groups}


def _icd_group(code):
    if pd.isna(code) or code == "?":
        return "missing"
    s = str(code)
    if s.startswith(("V", "E")):
        return "other"
    v = float(s)
    if 390 <= v <= 459 or int(v) == 785:
        return "circulatory"
    if 460 <= v <= 519 or int(v) == 786:
        return "respiratory"
    if 520 <= v <= 579 or int(v) == 787:
        return "digestive"
    if int(v) == 250:
        return "diabetes"
    if 800 <= v <= 999:
        return "injury"
    if 710 <= v <= 739:
        return "musculoskeletal"
    if 580 <= v <= 629 or int(v) == 788:
        return "genitourinary"
    if 140 <= v <= 239:
        return "neoplasms"
    return "other"


def load_diabetes130(n=DIAB130_N, full=False):
    """UCI Diabetes 130-US Hospitals 1999-2008 (Strack et al., 2014).

    Outcome: readmission within 30 days. Preprocessing follows Strack et al.:
    first encounter per patient only (avoids patient-level leakage across folds),
    encounters ending in death or hospice removed, invalid gender removed.
    A stratified random subsample of `n` patients (seed 2026) is used so that
    kernel SVM and the repeated protocols remain computationally tractable.
    """
    df = pd.read_csv(DATA / "diabetic_data.csv", na_values=["?"], keep_default_na=False, low_memory=False)
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]
    df = df[df["gender"].isin(["Male", "Female"])]
    y_all = (df["readmitted"] == "<30").astype(int).values
    meta_full = {"n_patients_eligible": int(len(df)), "prev_eligible": float(y_all.mean())}
    if not full:
        rng = np.random.RandomState(SUBSAMPLE_SEED)
        idx_pos = np.where(y_all == 1)[0]
        idx_neg = np.where(y_all == 0)[0]
        n_pos = int(round(n * y_all.mean()))
        take = np.concatenate([rng.choice(idx_pos, n_pos, replace=False),
                               rng.choice(idx_neg, n - n_pos, replace=False)])
        take.sort()
        df = df.iloc[take]
    y = (df["readmitted"] == "<30").astype(int).values
    out = pd.DataFrame(index=df.index)
    groups = {}
    num = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
    for c in num:
        out[c] = df[c].astype(float)
        groups[c] = c
    out["age"] = df["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5
    groups["age"] = "age"
    out["male"] = (df["gender"] == "Male").astype(float)
    groups["male"] = "gender"

    def onehot(series, levels, prefix, group):
        for lev in levels:
            name = f"{prefix}_{lev}"
            out[name] = (series == lev).astype(float)
            groups[name] = group

    race = df["race"].fillna("missing").replace({"Asian": "other", "Hispanic": "other", "Other": "other"})
    onehot(race, ["AfricanAmerican", "other", "missing"], "race", "race")  # ref: Caucasian
    adm_type = df["admission_type_id"].map({1: "emergency", 2: "urgent", 3: "elective"}).fillna("otheradm")
    onehot(adm_type, ["urgent", "elective", "otheradm"], "admtype", "admission_type")
    out["discharged_home"] = (df["discharge_disposition_id"] == 1).astype(float)
    groups["discharged_home"] = "discharge_disposition"
    src = df["admission_source_id"].map({7: "er", 1: "referral", 2: "referral", 3: "referral"}).fillna("othersrc")
    onehot(src, ["referral", "othersrc"], "admsrc", "admission_source")  # ref: ER
    dg = df["diag_1"].map(_icd_group)
    onehot(dg, ["respiratory", "digestive", "diabetes", "injury", "musculoskeletal",
                "genitourinary", "neoplasms", "other"], "diag1", "primary_diagnosis")  # ref: circulatory
    a1c = df["A1Cresult"].replace({"": "None"}).fillna("None")
    onehot(a1c, ["Norm", ">7", ">8"], "A1C", "A1C_result")
    glu = df["max_glu_serum"].replace({"": "None"}).fillna("None")
    onehot(glu, ["Norm", ">200", ">300"], "maxglu", "max_glu_serum")
    ins = df["insulin"]
    onehot(ins, ["Steady", "Up", "Down"], "insulin", "insulin")
    out["med_change"] = (df["change"] == "Ch").astype(float)
    groups["med_change"] = "med_change"
    out["diabetes_med"] = (df["diabetesMed"] == "Yes").astype(float)
    groups["diabetes_med"] = "diabetes_med"
    out = out.reset_index(drop=True)
    meta = {"name": "Diabetes 130-US Hospitals", "short": "D130", "groups": groups}
    meta.update(meta_full)
    return out, y, meta


LOADERS = {"BC": load_breast, "PIMA": load_pima, "HEART": load_cleveland, "D130": load_diabetes130}


def load(key):
    return LOADERS[key]()


if __name__ == "__main__":
    for k in LOADERS:
        X, y, m = load(k)
        print(k, X.shape, int(y.sum()), f"{y.mean():.3f}", int(X.isna().sum().sum()),
              {kk: v for kk, v in m.items() if kk.startswith(("n_", "prev"))})
    raw = load_pima(return_raw=True)
    for c in PIMA_ZERO_AS_MISSING:
        z = (raw[c] == 0).sum()
        print(f"  pima zero {c}: {z} ({z/len(raw):.1%})")
