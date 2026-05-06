"""
End-to-End Inference Pipeline
=============================

This module simulates the deployment-oriented
intrusion inference workflow of the proposed
TabTransformer–LightGBM framework.

Pipeline Stages:
1. Data preprocessing
2. Contextual feature representation
3. Binary intrusion classification
4. SHAP-based explainability generation

The implementation aligns with:
Section 3.6 - Proposed Methodology
Section 4 - Experimental Evaluation
"""

import numpy as np
import pandas as pd

from preprocessing import (
    preprocessing_pipeline
)

from train_tabtransformer import (
    build_tabtransformer
)

from train_lightgbm import (
    build_lightgbm_classifier
)

from shap_analysis import (
    initialize_shap_explainer
)


# ============================================================
# Simulated Inference Engine
# ============================================================

class SmartFarmingIDS:
    """
    End-to-end intrusion detection pipeline
    for smart farming IoT environments.
    """

    def __init__(self):

        self.tabtransformer = None

        self.lightgbm_classifier = None

        self.shap_explainer = None

    # ========================================================
    # Load Models
    # ========================================================

    def initialize_models(
        self,
        input_shape
    ):
        """
        Initialize TabTransformer and LightGBM modules.
        """

        self.tabtransformer = build_tabtransformer(
            input_shape=input_shape
        )

        self.lightgbm_classifier = build_lightgbm_classifier()

        print(
            "Inference modules initialized successfully."
        )

    # ========================================================
    # Feature Representation
    # ========================================================

    def generate_contextual_embeddings(
        self,
        processed_features
    ):
        """
        Generate contextual feature embeddings
        using the TabTransformer model.
        """

        embeddings = self.tabtransformer.predict(
            processed_features
        )

        return embeddings

    # ========================================================
    # Binary Intrusion Prediction
    # ========================================================

    def predict_intrusion(
        self,
        contextual_embeddings
    ):
        """
        Predict intrusion behavior:
        0 -> Benign
        1 -> Attack
        """

        prediction = self.lightgbm_classifier.predict(
            contextual_embeddings
        )

        probability = self.lightgbm_classifier.predict_proba(
            contextual_embeddings
        )[:, 1]

        return prediction, probability

    # ========================================================
    # Explainability
    # ========================================================

    def generate_shap_explanation(
        self,
        contextual_embeddings
    ):
        """
        Generate SHAP explanations
        for intrusion predictions.
        """

        explainer, shap_values = initialize_shap_explainer(
            self.lightgbm_classifier,
            contextual_embeddings
        )

        self.shap_explainer = explainer

        return shap_values

    # ========================================================
    # End-to-End Inference
    # ========================================================

    def run_inference(
        self,
        dataset,
        label_column="Label"
    ):
        """
        Execute complete intrusion detection workflow.
        """

        print(
            "Starting end-to-end intrusion inference pipeline..."
        )

        split_data, _, _ = preprocessing_pipeline(
            dataset,
            label_column
        )

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test
        ) = split_data

        self.initialize_models(
            input_shape=(X_train.shape[1],)
        )

        contextual_embeddings = (
            self.generate_contextual_embeddings(
                X_test
            )
        )

        predictions, probabilities = (
            self.predict_intrusion(
                contextual_embeddings
            )
        )

        shap_values = self.generate_shap_explanation(
            contextual_embeddings
        )

        print(
            "Inference pipeline completed successfully."
        )

        return {

            "Predictions": predictions,

            "Prediction Probabilities": probabilities,

            "SHAP Values": shap_values
        }


# ============================================================
# Execution
# ============================================================

if __name__ == "__main__":

    print(
        "Smart Farming IoT IDS Inference Module Initialized."
    )
