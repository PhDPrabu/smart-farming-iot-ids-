"""
TabTransformer Representation Learning Module
=============================================

This module implements the TabTransformer-based representation
learning component described in:

Section 3.6 - Proposed Methodology
Section 4.1.1 - Hyperparameter Configuration

The TabTransformer is responsible for learning contextualized
feature embeddings from heterogeneous categorical and numerical
IoT traffic attributes using self-attention mechanisms.

The learned representations are subsequently utilized by the
LightGBM classifier for binary intrusion detection.
"""

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# Hyperparameter Configuration
# ============================================================

EMBEDDING_DIM = 64
NUM_TRANSFORMER_LAYERS = 4
NUM_ATTENTION_HEADS = 4
FFN_HIDDEN_DIM = 128
DROPOUT_RATE = 0.2

LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 9


# ============================================================
# Transformer Encoder Block
# ============================================================

class TransformerEncoder(layers.Layer):
    """
    Transformer encoder block for contextual feature learning.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        ffn_hidden_dim,
        dropout_rate=0.1
    ):

        super().__init__()

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_hidden_dim, activation="relu"),
            layers.Dense(embedding_dim)
        ])

        self.layernorm1 = layers.LayerNormalization()

        self.layernorm2 = layers.LayerNormalization()

        self.dropout1 = layers.Dropout(dropout_rate)

        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):

        attention_output = self.attention(inputs, inputs)

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        out1 = self.layernorm1(inputs + attention_output)

        ffn_output = self.ffn(out1)

        ffn_output = self.dropout2(
            ffn_output,
            training=training
        )

        return self.layernorm2(out1 + ffn_output)


# ============================================================
# TabTransformer Model
# ============================================================

def build_tabtransformer(input_shape):
    """
    Build TabTransformer representation learning model.
    """

    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(EMBEDDING_DIM)(inputs)

    x = layers.Reshape((input_shape[0], 1))(x)

    for _ in range(NUM_TRANSFORMER_LAYERS):

        x = TransformerEncoder(
            embedding_dim=1,
            num_heads=NUM_ATTENTION_HEADS,
            ffn_hidden_dim=FFN_HIDDEN_DIM,
            dropout_rate=DROPOUT_RATE
        )(x)

    x = layers.Flatten()(x)

    representation_output = layers.Dense(
        EMBEDDING_DIM,
        activation="relu",
        name="contextual_feature_embedding"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=representation_output,
        name="TabTransformer"
    )

    return model


# ============================================================
# Training Function
# ============================================================

def train_tabtransformer(
    X_train,
    y_train,
    X_val,
    y_val
):
    """
    Train TabTransformer representation model.
    """

    model = build_tabtransformer(
        input_shape=(X_train.shape[1],)
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    return model, history


# ============================================================+
# Main Execution
# ============================================================+

if __name__ == "__main__":

    print(
        "TabTransformer Representation Learning Module Initialized."
    )
