"""
SHAP Explainability Analysis Module
===================================

This module implements the SHAP-based explainability
framework described in:

Section 3.6 - Proposed Methodology
Section 4.4 - SHAP-Based Explainability Analysis

The module supports:
1. Global feature importance analysis
2. Local prediction interpretation
3. Cross-feature interaction analysis
4. False positive interpretation
5. Behavioral analysis of intrusion predictions
"""

import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================================
# Initialize SHAP Explainer
# ============================================================

def initialize_shap_explainer(
    trained_lightgbm_model,
    background_data
):
    """
    Initialize SHAP TreeExplainer for LightGBM classifier.
    """

    explainer = shap.TreeExplainer(
        trained_lightgbm_model
    )

    shap_values = explainer.shap_values(
        background_data
    )

    return explainer, shap_values


# ============================================================
# Global Feature Importance
# ============================================================

def plot_global_feature_importance(
    shap_values,
    features,
    max_display=15
):
    """
    Generate global SHAP feature importance summary plot.
    """

    shap.summary_plot(
        shap_values,
        features,
        max_display=max_display,
        show=False
    )

    plt.title(
        "Global SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# Local Prediction Explanation
# ============================================================

def explain_single_prediction(
    explainer,
    sample_instance
):
    """
    Generate local SHAP explanation
    for a single intrusion prediction.
    """

    shap_values = explainer.shap_values(
        sample_instance
    )

    shap.force_plot(
        explainer.expected_value,
        shap_values,
        sample_instance,
        matplotlib=True
    )


# ============================================================
# Cross-Feature Interaction Analysis
# ============================================================

def analyze_feature_interactions(
    explainer,
    dataset,
    feature_x,
    feature_y
):
    """
    Analyze SHAP interaction behavior between
    two traffic attributes.

    Example:
    Protocol Type × Packet Length Mean
    """

    shap_interaction_values = explainer.shap_interaction_values(
        dataset
    )

    shap.dependence_plot(
        (
            feature_x,
            feature_y
        ),
        shap_interaction_values,
        dataset,
        show=False
    )

    plt.title(
        f"SHAP Interaction Analysis: "
        f"{feature_x} × {feature_y}"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# False Positive Analysis
# ============================================================

def analyze_false_positives(
    model,
    X_test,
    y_test
):
    """
    Analyze false positive predictions
    using SHAP explanations.

    False positives are defined as:
    benign samples incorrectly classified as attacks.
    """

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    false_positive_indices = [

        index

        for index, (true_label, pred_label)

        in enumerate(zip(y_test, predictions))

        if true_label == 0 and pred_label == 1
    ]

    false_positive_samples = X_test.iloc[
        false_positive_indices
    ]

    print(
        f"Number of False Positives: "
        f"{len(false_positive_indices)}"
    )

    return false_positive_samples


# ============================================================
# Behavioral Interpretation Utility
# ============================================================

def summarize_behavioral_patterns():
    """
    Summarize behavioral interpretation logic
    aligned with manuscript discussion.
    """

    interpretation_summary = {

        "SYN Flag Count":
        "Captures abnormal connection initiation behavior.",

        "ACK Flag Count":
        "Provides contextual discrimination between benign "
        "and anomalous transport-layer communication.",

        "Packet Length Mean":
        "Reflects payload irregularities and flooding behavior.",

        "Protocol Type":
        "Acts as a contextual indicator of protocol-specific "
        "communication anomalies."
    }

    return interpretation_summary


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":

    print(
        "SHAP Explainability Analysis Module Initialized."
    )
