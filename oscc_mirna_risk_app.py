
"""
OSCC miRNA Risk Calculator - Streamlit Web App

This app implements the 13-miRNA diagnostic model we derived from 7 GEO datasets.
It takes normalized miRNA expression values as input and outputs:

- Estimated probability of OSCC
- Risk category (low / low–moderate / moderate–high / high)
- A simple clinical recommendation string

NOTE:
For now, the model is calibrated to the GEO z-scored expression space.
When you develop a hospital-specific qPCR panel, you should retrain or recalibrate
this model on your own data and update the coefficients and scaling parameters.
"""

import math
import numpy as np
import streamlit as st

# 13-miRNA panel (diagnostic)
MIRNAS_13 = [
    "hsa-miR-375",
    "hsa-miR-139-5p",
    "hsa-miR-7",
    "hsa-miR-21",
    "hsa-miR-140-3p",
    "hsa-miR-5580-3p",
    "hsa-miR-486-5p",
    "hsa-miR-21-5p",
    "hsa-miR-4491",
    "hsa-miR-30e*",
    "hsa-miR-718",
    "hsa-miR-5186",
    "hsa-miR-4462",
]

# Logistic regression coefficients (from GEO pooled model)
INTERCEPT = 0.912  # rounded

COEFFS = {
    "hsa-miR-375": -1.213,
    "hsa-miR-139-5p": -0.934,
    "hsa-miR-7": 1.400,
    "hsa-miR-21": 0.656,
    "hsa-miR-140-3p": -0.556,
    "hsa-miR-5580-3p": -2.025,
    "hsa-miR-486-5p": -0.745,
    "hsa-miR-21-5p": 0.988,
    "hsa-miR-4491": -0.984,
    "hsa-miR-30e*": -0.540,
    "hsa-miR-718": 0.688,
    "hsa-miR-5186": -0.666,
    "hsa-miR-4462": -0.265,
}

# Training means/SDs (z-score space). Means are ~0, SDs close to 1.
MEANS = {
    "hsa-miR-375": 0.000,
    "hsa-miR-139-5p": 0.000,
    "hsa-miR-7": -0.000,
    "hsa-miR-21": 0.000,
    "hsa-miR-140-3p": 0.000,
    "hsa-miR-5580-3p": -0.000,
    "hsa-miR-486-5p": -0.000,
    "hsa-miR-21-5p": 0.000,
    "hsa-miR-4491": 0.000,
    "hsa-miR-30e*": -0.000,
    "hsa-miR-718": -0.000,
    "hsa-miR-5186": -0.000,
    "hsa-miR-4462": 0.000,
}

STDS = {
    "hsa-miR-375": 0.906,
    "hsa-miR-139-5p": 0.943,
    "hsa-miR-7": 0.695,
    "hsa-miR-21": 0.813,
    "hsa-miR-140-3p": 0.943,
    "hsa-miR-5580-3p": 0.582,
    "hsa-miR-486-5p": 0.943,
    "hsa-miR-21-5p": 0.582,
    "hsa-miR-4491": 0.582,
    "hsa-miR-30e*": 0.742,
    "hsa-miR-718": 0.582,
    "hsa-miR-5186": 0.582,
    "hsa-miR-4462": 0.582,
}

# Risk thresholds (derived from GEO diagnostic model)
T_LOW = 0.569   # high-sensitivity boundary
T_MID = 0.705   # mid-point between T_LOW and T_HIGH
T_HIGH = 0.841  # high-specificity boundary

def logistic(x: float) -> float:
    """Standard logistic function."""
    return 1.0 / (1.0 + math.exp(-x))

def predict_oscc_risk(miRNA_values: dict) -> dict:
    """
    Compute OSCC risk from miRNA values.

    Parameters
    ----------
    miRNA_values : dict
        Dictionary mapping miRNA name -> expression value.
        For now, values are assumed to be on the same scale as the training
        z-score space (or already roughly standardized). When you develop
        a hospital qPCR assay, you should update MEANS and STDS accordingly.

    Returns
    -------
    result : dict
        Contains:
        - 'probability': P(OSCC) as a float between 0 and 1
        - 'risk_category': string label
        - 'recommendation': textual recommendation
    """
    # Build feature vector in correct order
    z_values = []
    for mir in MIRNAS_13:
        if mir not in miRNA_values:
            raise ValueError(f"Missing value for miRNA: {mir}")
        x = float(miRNA_values[mir])
        mu = MEANS[mir]
        sd = STDS[mir] if STDS[mir] > 0 else 1.0
        z = (x - mu) / sd
        z_values.append(z)

    # Linear predictor
    logit = INTERCEPT
    for mir, z in zip(MIRNAS_13, z_values):
        logit += COEFFS[mir] * z

    p = logistic(logit)

    # Risk category and recommendation
    if p < T_LOW:
        category = "Low risk"
        recommendation = "Low risk of OSCC based on miRNA panel. Consider routine follow-up and risk factor counselling."
    elif p < T_MID:
        category = "Low–moderate risk"
        recommendation = "Borderline risk. Consider repeat clinical exam and follow-up within 3–6 months."
    elif p < T_HIGH:
        category = "Moderate–high risk"
        recommendation = "Elevated risk. Biopsy should be strongly considered depending on clinical findings."
    else:
        category = "High risk"
        recommendation = "High risk of OSCC based on miRNA panel. Biopsy is recommended without delay."

    return {
        "probability": p,
        "risk_category": category,
        "recommendation": recommendation,
    }


def main():
    st.title("OSCC miRNA Risk Calculator (13-miRNA Panel)")
    st.write(
        "This tool estimates the probability of oral squamous cell carcinoma (OSCC) "
        "based on a 13-miRNA panel derived from public datasets. "
        "It is intended for research and decision-support, not as a standalone diagnostic test."
    )

    st.header("Input miRNA expression values")
    st.markdown(
        "For now, please enter normalized expression values for each miRNA "
        "(ideally on a z-score scale similar to the training datasets). "
        "Later, when you have hospital qPCR data, this app can be recalibrated."
    )

    inputs = {}
    cols = st.columns(2)
    for i, mir in enumerate(MIRNAS_13):
        col = cols[i % 2]
        with col:
            val = st.number_input(f"{mir}", value=0.0, format="%.3f")
        inputs[mir] = val

    if st.button("Calculate OSCC risk"):
        try:
            result = predict_oscc_risk(inputs)
            p = result["probability"]
            st.subheader("Results")
            st.write(f"**Predicted OSCC probability:** {p * 100:.1f} %")
            st.write(f"**Risk category:** {result['risk_category']}")
            st.write(f"**Recommendation:** {result['recommendation']}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.caption(
        "Disclaimer: This model is derived from public datasets and has not yet been "
        "prospectively validated in your hospital population. It should be used as a "
        "research decision-support tool, not as a sole basis for clinical decisions."
    )


if __name__ == "__main__":
    main()
