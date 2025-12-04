"""
OSCC miRNA Risk Calculator – 13-miRNA panel (balanced + sensitivity + specificity models)

This app implements the updated diagnostic model derived from the pooled GEO
datasets in the current Colab analysis. It uses:

- A balanced logistic model (Model_BAL) for overall P(OSCC)
- A sensitivity-oriented model (Model_SENS) using sensitivity-dominant + shared miRNAs
- A specificity-oriented model (Model_SPEC) using specificity-dominant + shared miRNAs

Risk bands are defined using ROC-based decision thresholds obtained from the
same data (for the 13-miRNA panel):

1) If P_spec >= T_HIGH_SPEC  -> High risk (rule-in OSCC)
2) Else if P_sens < T_LOW_SENS -> Low risk (rule-out OSCC)
3) Else if P_bal >= T_MOD_BAL  -> Moderate–high risk
4) Else                        -> Indeterminate
"""

import math
import csv
import os
from datetime import datetime
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np


# -------------------------------------------------------------------------
# 13-miRNA PANEL & MODEL PARAMETERS  (>>> FILL FROM COLAB <<<)
# -------------------------------------------------------------------------
# Run the export cell in Colab for the 13-miRNA FINAL_PANEL:
#
#   FINAL_MIRNAS = FINAL_PANEL
#   MEANS_BAL = dict(zip(FINAL_PANEL, scaler_final.mean_))
#   STDS_BAL  = dict(zip(FINAL_PANEL, scaler_final.scale_))
#   COEFFS_BAL = dict(zip(FINAL_PANEL, model_final.coef_[0]))
#   INTERCEPT_BAL  = float(model_final.intercept_[0])
#   COEFFS_SENS = dict(zip(FINAL_PANEL, model_sens_final.coef_[0]))
#   INTERCEPT_SENS = float(model_sens_final.intercept_[0])
#   COEFFS_SPEC = dict(zip(FINAL_PANEL, model_spec_final.coef_[0]))
#   INTERCEPT_SPEC = float(model_spec_final.intercept_[0])
#
# then paste the printed values into the variables below.

FINAL_MIRNAS = ['hsa-let-7c', 'hsa-miR-617', 'hsa-miR-542-5p', 'hsa-miR-636', 'hsa-miR-567', 'hsa-miR-299-5p', 'hsa-miR-409-3p', 'hsa-miR-206', 'hsa-miR-484', 'hsa-miR-1', 'hsa-miR-650', 'hsa-miR-142-5p', 'hsa-miR-498'
]

MEANS_BAL: Dict[str, float] = {'hsa-let-7c': np.float64(5.150774478558347e-16), 'hsa-miR-617': np.float64(-7.924268428551303e-16), 'hsa-miR-542-5p': np.float64(5.61302347022384e-16), 'hsa-miR-636': np.float64(-1.2546758345206231e-15), 'hsa-miR-567': np.float64(9.971371105927057e-16), 'hsa-miR-299-5p': np.float64(-2.641422809517101e-17), 'hsa-miR-409-3p': np.float64(1.220007160145711e-15), 'hsa-miR-206': np.float64(-2.0471026773757533e-16), 'hsa-miR-484': np.float64(7.164859370815137e-16), 'hsa-miR-1': np.float64(3.7310097184429054e-16), 'hsa-miR-650': np.float64(3.8465719663592785e-16), 'hsa-miR-142-5p': np.float64(1.8489959666619708e-16), 'hsa-miR-498': np.float64(1.0565691238068404e-16)
}

STDS_BAL: Dict[str, float] = {'hsa-let-7c': np.float64(1.0), 'hsa-miR-617': np.float64(1.0), 'hsa-miR-542-5p': np.float64(1.0), 'hsa-miR-636': np.float64(1.0), 'hsa-miR-567': np.float64(1.0), 'hsa-miR-299-5p': np.float64(1.0), 'hsa-miR-409-3p': np.float64(0.9999999999999999), 'hsa-miR-206': np.float64(1.0), 'hsa-miR-484': np.float64(1.0), 'hsa-miR-1': np.float64(1.0), 'hsa-miR-650': np.float64(1.0), 'hsa-miR-142-5p': np.float64(1.0), 'hsa-miR-498': np.float64(1.0)
}

INTERCEPT_BAL: float = 0.9287511471375864
INTERCEPT_SENS: float = 2.074343374803706
INTERCEPT_SPEC: float = 0.0

COEFFS_BAL: Dict[str, float] = {'hsa-let-7c': np.float64(-0.8009347020697357), 'hsa-miR-617': np.float64(-1.2711056484014536), 'hsa-miR-542-5p': np.float64(0.7621231506126712), 'hsa-miR-636': np.float64(0.7966941699323412), 'hsa-miR-567': np.float64(-0.31072883072259505), 'hsa-miR-299-5p': np.float64(-0.4242229711032154), 'hsa-miR-409-3p': np.float64(-0.5351780863779684), 'hsa-miR-206': np.float64(-0.005920146824483282), 'hsa-miR-484': np.float64(0.25253646178292327), 'hsa-miR-1': np.float64(-0.07643281523979761), 'hsa-miR-650': np.float64(0.2709534616688932), 'hsa-miR-142-5p': np.float64(0.05522975804686743), 'hsa-miR-498': np.float64(0.31446805763713115)
}

COEFFS_SENS: Dict[str, float] = {'hsa-let-7c': np.float64(-0.8745947173601168), 'hsa-miR-617': np.float64(-1.3523156193210875), 'hsa-miR-542-5p': np.float64(0.9189667431705364), 'hsa-miR-636': np.float64(0.9718763511919435), 'hsa-miR-567': np.float64(-0.32401344661572934), 'hsa-miR-299-5p': np.float64(-0.5292111057798613), 'hsa-miR-409-3p': np.float64(-0.6102925411541783), 'hsa-miR-206': np.float64(-0.02132074258503247), 'hsa-miR-484': np.float64(0.25549286184466785), 'hsa-miR-1': np.float64(-0.052496340059757944), 'hsa-miR-650': np.float64(0.20243640978682764), 'hsa-miR-142-5p': np.float64(0.2046291466046165), 'hsa-miR-498': np.float64(0.4357013115753992)
}

COEFFS_SPEC: Dict[str, float] = {'hsa-let-7c': np.float64(-0.8168925374013886), 'hsa-miR-617': np.float64(-1.3924106602114747), 'hsa-miR-542-5p': np.float64(0.6742395827272426), 'hsa-miR-636': np.float64(0.6281925693475644), 'hsa-miR-567': np.float64(-0.29279883323131095), 'hsa-miR-299-5p': np.float64(-0.32018378133237174), 'hsa-miR-409-3p': np.float64(-0.5842844007619535), 'hsa-miR-206': np.float64(0.023192133867878495), 'hsa-miR-484': np.float64(0.25907274273760755), 'hsa-miR-1': np.float64(-0.15480094274921355), 'hsa-miR-650': np.float64(0.3522508734049746), 'hsa-miR-142-5p': np.float64(-0.13726980408140185), 'hsa-miR-498': np.float64(0.14583176138240278)
}

# -------------------------------------------------------------------------
# ROC-derived thresholds for the **13-miRNA** panel (from your screenshot)
# -------------------------------------------------------------------------

# High-sensitivity threshold from sensitivity model (thr_sens['T_low'])
T_LOW_SENS: float = 0.654080237784321    # sens ≈ 0.951, spec ≈ 0.781

# High-specificity threshold from specificity model (thr_spec['T_high'])
T_HIGH_SPEC: float = 0.7842582782768076  # spec ≈ 0.971, sens ≈ 0.512

# Balanced threshold from balanced model (thr_final['T_mod'])
T_MOD_BAL: float = 0.5273112073123669    # sens ≈ 0.933, spec ≈ 0.838

# -------------------------------------------------------------------------
# Logging configuration
# -------------------------------------------------------------------------

LOG_PATH = "oscc_risk_log_13miRNA.csv"


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def logistic(x: float) -> float:
    """Standard logistic function."""
    return 1.0 / (1.0 + math.exp(-x))


def standardise(mir_name: str, value: float) -> float:
    """Standardise a single miRNA value using training mean/SD."""
    mu = MEANS_BAL[mir_name]
    sd = STDS_BAL[mir_name] if STDS_BAL[mir_name] > 0 else 1.0
    return (float(value) - mu) / sd


def compute_probabilities(miRNA_values: Dict[str, float]) -> Dict[str, float]:
    """
    Compute P_bal, P_sens, P_spec from miRNA values.

    All three models use the same standardised inputs but different intercepts
    and coefficient dictionaries.
    """
    # Build z-scores in consistent order
    z = {}
    for mir in FINAL_MIRNAS:
        if mir not in miRNA_values:
            raise ValueError(f"Missing value for miRNA: {mir}")
        z[mir] = standardise(mir, miRNA_values[mir])

    # Balanced model
    logit_bal = INTERCEPT_BAL
    for mir in FINAL_MIRNAS:
        logit_bal += COEFFS_BAL[mir] * z[mir]
    p_bal = logistic(logit_bal)

    # Sensitivity-oriented model
    logit_sens = INTERCEPT_SENS
    for mir in FINAL_MIRNAS:
        logit_sens += COEFFS_SENS[mir] * z[mir]
    p_sens = logistic(logit_sens)

    # Specificity-oriented model
    logit_spec = INTERCEPT_SPEC
    for mir in FINAL_MIRNAS:
        logit_spec += COEFFS_SPEC[mir] * z[mir]
    p_spec = logistic(logit_spec)

    return {
        "P_bal": p_bal,
        "P_sens": p_sens,
        "P_spec": p_spec,
    }


def classify_risk(probs: Dict[str, float]) -> Dict[str, Any]:
    """
    Apply multi-threshold risk logic to P_bal, P_sens, P_spec.

    Returns a dict with:
      - probability (P_bal)
      - P_sens, P_spec
      - risk_category
      - recommendation
    """
    p_bal = probs["P_bal"]
    p_sens = probs["P_sens"]
    p_spec = probs["P_spec"]

    # Rule-in and rule-out decisions
    if p_spec >= T_HIGH_SPEC:
        category = "High risk (rule-in OSCC)"
        recommendation = (
            "High risk of OSCC based on the specificity-oriented miRNA model. "
            "Biopsy or urgent specialist referral is recommended, in line with "
            "clinical findings."
        )
    elif p_sens < T_LOW_SENS:
        category = "Low risk (rule-out OSCC)"
        recommendation = (
            "Low risk of OSCC based on the sensitivity-oriented miRNA model. "
            "Routine follow-up and risk factor modification are appropriate, "
            "provided there is no concerning clinical progression."
        )
    else:
        # Intermediate pool – refine using balanced model
        if p_bal >= T_MOD_BAL:
            category = "Moderate–high risk"
            recommendation = (
                "Intermediate to high risk based on the balanced miRNA model. "
                "Biopsy should be strongly considered, taking lesion appearance "
                "and patient factors into account."
            )
        else:
            category = "Indeterminate"
            recommendation = (
                "Indeterminate risk: miRNA signatures do not clearly support "
                "either rule-out or rule-in. Close clinical follow-up, repeat "
                "assessment or adjunctive testing is recommended."
            )

    return {
        "probability": p_bal,
        "P_sens": p_sens,
        "P_spec": p_spec,
        "risk_category": category,
        "recommendation": recommendation,
    }


def log_prediction(
    miRNA_values: Dict[str, float],
    result: Dict[str, Any],
    patient_id: str = "",
    age_years: Optional[float] = None,
    sex: str = "",
    tobacco_use: str = "",
    log_path: str = LOG_PATH,
) -> None:
    """Append one patient's data and model output to a CSV log file."""
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = [
                "timestamp",
                "patient_id",
                "age_years",
                "sex",
                "tobacco_use",
            ] + FINAL_MIRNAS + [
                "P_bal",
                "P_sens",
                "P_spec",
                "risk_category",
            ]
            writer.writerow(header)

        row = [datetime.now().isoformat(), patient_id, age_years, sex, tobacco_use]
        for mir in FINAL_MIRNAS:
            row.append(miRNA_values.get(mir, ""))

        row.extend([
            result.get("probability", ""),
            result.get("P_sens", ""),
            result.get("P_spec", ""),
            result.get("risk_category", ""),
        ])

        writer.writerow(row)


def load_log_dataframe(log_path: str = LOG_PATH):
    """Load the log file into a pandas DataFrame, if it exists."""
    if os.path.isfile(log_path):
        try:
            return pd.read_csv(log_path)
        except Exception:
            return None
    return None


# -------------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------------

def main():
    st.title("OSCC miRNA Risk Calculator – 13-miRNA Panel")
    st.write(
        "This tool estimates the probability of oral squamous cell carcinoma (OSCC) "
        "using a 13-miRNA panel derived from pooled GEO datasets. It combines a "
        "balanced model with sensitivity- and specificity-oriented models to "
        "define low-risk (rule-out), high-risk (rule-in), moderate–high and "
        "indeterminate risk zones. It is intended for research and decision "
        "support, not as a standalone diagnostic test."
    )

    st.header("Patient information")
    patient_id = st.text_input("Patient ID / Study code (avoid full name):", value="")
    age_years = st.number_input(
        "Age (years):", min_value=0, max_value=120, value=0, step=1
    )
    sex = st.selectbox("Sex:", ["Unknown", "Male", "Female", "Other / Intersex"])
    tobacco_use = st.selectbox(
        "Tobacco use:",
        [
            "Unknown",
            "None",
            "Smoking only",
            "Smokeless only",
            "Both smoking and smokeless",
        ],
    )

    st.header("Input miRNA expression values")
    st.markdown(
        "Enter normalised expression values for each miRNA (e.g. -ΔCt or scaled data). "
        "The model expects values on a scale comparable to the training data; "
        "if you recalibrate to your hospital qPCR data, the means, standard "
        "deviations and coefficients in this app should be updated accordingly."
    )

    inputs: Dict[str, float] = {}
    cols = st.columns(3)
    for i, mir in enumerate(FINAL_MIRNAS):
        col = cols[i % 3]
        with col:
            val = st.number_input(f"{mir}", value=0.0, format="%.3f")
        inputs[mir] = val

    if st.button("Calculate OSCC risk"):
        try:
            probs = compute_probabilities(inputs)
            result = classify_risk(probs)

            log_prediction(
                miRNA_values=inputs,
                result=result,
                patient_id=patient_id,
                age_years=age_years,
                sex=sex,
                tobacco_use=tobacco_use,
            )

            st.subheader("Results")
            st.write(f"**Balanced P(OSCC):** {result['probability']*100:.1f}%")
            st.write(f"**Sensitivity-model P(OSCC):** {result['P_sens']*100:.1f}%")
            st.write(f"**Specificity-model P(OSCC):** {result['P_spec']*100:.1f}%")
            st.write(f"**Risk category:** {result['risk_category']}")
            st.write(f"**Recommendation:** {result['recommendation']}")
            st.success("Case saved to log file.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.header("Saved cases")
    df_log = load_log_dataframe()

    if df_log is None:
        st.info(
            "No saved cases yet. Once you calculate risk for at least one patient, "
            "the log file will be created and shown here."
        )
    else:
        st.write(f"Total logged cases: {len(df_log)}")
        st.dataframe(df_log)
        csv_bytes = df_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download log as CSV",
            data=csv_bytes,
            file_name="oscc_risk_log_13miRNA.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.caption(
        "Disclaimer: This model is derived from public datasets and has not yet "
        "been prospectively validated in your hospital population. It should be "
        "used as a research decision-support tool, not as the sole basis for "
        "clinical decisions."
    )


if __name__ == "__main__":
    main()
