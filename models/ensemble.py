import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FEATURES = 13
HIDDEN_SIZE = 64
EMBEDDING_SIZE = 32


# ============================================================
# ATTENTION
# ============================================================

class BehavioralAttention(nn.Module):
    """
    Learns which time steps in a behavioral sequence
    are more important for authentication.
    """

    def __init__(self, input_size):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        x shape:
            [batch, sequence, features]
        """

        attention_scores = self.score(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        weighted = x * attention_weights

        context = weighted.sum(dim=1)

        return context, attention_weights


# ============================================================
# CNN MODEL
# ============================================================

class BehavioralCNN(nn.Module):
    """
    CNN branch.

    Learns local patterns in keyboard/mouse behavior.
    """

    def __init__(
        self,
        input_features=INPUT_FEATURES,
        embedding_size=EMBEDDING_SIZE
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                in_channels=input_features,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Conv1d(
                in_channels=64,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(
            32,
            embedding_size
        )

    def forward(self, x):

        # x:
        # [batch, sequence, features]

        x = x.transpose(1, 2)

        # [batch, features, sequence]

        x = self.network(x)

        # [batch, 32, 1]

        x = x.squeeze(-1)

        # [batch, 32]

        embedding = self.fc(x)

        return embedding


# ============================================================
# LSTM MODEL
# ============================================================

class BehavioralLSTM(nn.Module):
    """
    LSTM branch.

    Learns temporal relationships between behavioral events.
    """

    def __init__(
        self,
        input_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_size)
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        # Last temporal representation

        last_output = output[:, -1, :]

        embedding = self.fc(last_output)

        return embedding


# ============================================================
# CNN + ATTENTION MODEL
# ============================================================

class CNNAttention(nn.Module):
    """
    CNN + Attention branch.

    CNN extracts local patterns.
    Attention selects important temporal information.
    """

    def __init__(
        self,
        input_features=INPUT_FEATURES,
        embedding_size=EMBEDDING_SIZE
    ):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                input_features,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        self.attention = BehavioralAttention(32)

        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_size)
        )

    def forward(self, x):

        # [batch, sequence, features]

        x = x.transpose(1, 2)

        # [batch, features, sequence]

        x = self.cnn(x)

        # [batch, 32, sequence]

        x = x.transpose(1, 2)

        # [batch, sequence, 32]

        context, weights = self.attention(x)

        embedding = self.fc(context)

        return embedding


# ============================================================
# LSTM + ATTENTION MODEL
# ============================================================

class LSTMAttention(nn.Module):
    """
    LSTM + Attention branch.

    Learns temporal behavior and determines
    which temporal states are important.
    """

    def __init__(
        self,
        input_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.attention = BehavioralAttention(
            hidden_size
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_size)
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        context, weights = self.attention(output)

        embedding = self.fc(context)

        return embedding


# ============================================================
# HYBRID CNN + LSTM + ATTENTION
# ============================================================

class HybridBehavioralModel(nn.Module):
    """
    Main hybrid branch.

    CNN:
        local patterns

    LSTM:
        temporal dependencies

    Attention:
        important temporal information
    """

    def __init__(
        self,
        input_features=INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE
    ):
        super().__init__()

        # CNN feature extractor

        self.cnn = nn.Sequential(

            nn.Conv1d(
                input_features,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Conv1d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )

        # LSTM receives CNN representation

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Attention

        self.attention = BehavioralAttention(
            hidden_size
        )

        # Final representation

        self.fc = nn.Sequential(

            nn.Linear(hidden_size, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                embedding_size
            )
        )

    def forward(self, x):

        # CNN

        x = x.transpose(1, 2)

        x = self.cnn(x)

        x = x.transpose(1, 2)

        # LSTM

        x, _ = self.lstm(x)

        # Attention

        context, weights = self.attention(x)

        # Embedding

        embedding = self.fc(context)

        return embedding


# ============================================================
# 5-MODEL ENSEMBLE
# ============================================================

class BehavioralEnsemble(nn.Module):
    """
    Five-model behavioral authentication ensemble.

    Models:

        1. CNN
        2. LSTM
        3. CNN + Attention
        4. LSTM + Attention
        5. CNN + LSTM + Attention

    Each model produces a behavioral embedding.

    The embeddings are combined using learned
    ensemble weights.
    """

    def __init__(self):

        super().__init__()

        self.cnn = BehavioralCNN()

        self.lstm = BehavioralLSTM()

        self.cnn_attention = CNNAttention()

        self.lstm_attention = LSTMAttention()

        self.hybrid = HybridBehavioralModel()

        # Learnable ensemble weights

        self.ensemble_weights = nn.Parameter(
            torch.ones(5)
        )

        # Final fusion layer

        self.fusion = nn.Sequential(

            nn.Linear(
                EMBEDDING_SIZE,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                EMBEDDING_SIZE
            )
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Five behavioral representations
        # ----------------------------------------------------

        e1 = self.cnn(x)

        e2 = self.lstm(x)

        e3 = self.cnn_attention(x)

        e4 = self.lstm_attention(x)

        e5 = self.hybrid(x)

        embeddings = torch.stack(
            [
                e1,
                e2,
                e3,
                e4,
                e5
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Weighted ensemble
        # ----------------------------------------------------

        weights = torch.softmax(
            self.ensemble_weights,
            dim=0
        )

        weights = weights.view(
            1,
            5,
            1
        )

        combined = (
            embeddings * weights
        ).sum(dim=1)

        # ----------------------------------------------------
        # Final fusion
        # ----------------------------------------------------

        final_embedding = self.fusion(
            combined
        )

        return final_embedding, {
            "individual_embeddings": embeddings,
            "ensemble_weights": weights.squeeze()
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING BEHAVIORAL ENSEMBLE")
    print("=" * 60)

    # Example:
    # batch = 4
    # sequence = 10
    # features = 13

    x = torch.randn(
        4,
        10,
        INPUT_FEATURES
    )

    model = BehavioralEnsemble()

    output, information = model(x)

    print()

    print("Input:")
    print(x.shape)

    print()

    print("Final embedding:")
    print(output.shape)

    print()

    print("Individual embeddings:")
    print(
        information[
            "individual_embeddings"
        ].shape
    )

    print()

    print("Ensemble weights:")

    print(
        information[
            "ensemble_weights"
        ].detach()
    )

    print()

    print("=" * 60)
    print("ENSEMBLE TEST COMPLETED")
    print("=" * 60)