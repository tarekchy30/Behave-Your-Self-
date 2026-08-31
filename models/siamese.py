import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BEHAVIORAL ENCODER
# ============================================================

class BehavioralEncoder(nn.Module):

    def __init__(self, input_size, embedding_size=32):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, embedding_size)
        )


    def forward(self, x):

        embedding = self.network(x)

        # Normalize the embedding.
        # This makes distance comparison more stable.

        embedding = F.normalize(
            embedding,
            p=2,
            dim=1
        )

        return embedding


# ============================================================
# SIAMESE NETWORK
# ============================================================

class SiameseNetwork(nn.Module):

    def __init__(self, input_size, embedding_size=32):

        super().__init__()

        # IMPORTANT:
        # There is only ONE encoder.
        #
        # Both inputs go through this same encoder.
        #
        # This means the two branches share weights.

        self.encoder = BehavioralEncoder(
            input_size=input_size,
            embedding_size=embedding_size
        )


    def forward(self, sample_a, sample_b):

        embedding_a = self.encoder(sample_a)

        embedding_b = self.encoder(sample_b)

        return embedding_a, embedding_b


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def euclidean_distance(embedding_a, embedding_b):

    distance = torch.sqrt(
        torch.sum(
            (embedding_a - embedding_b) ** 2,
            dim=1
        ) + 1e-8
    )

    return distance


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):

        super().__init__()

        self.margin = margin


    def forward(
        self,
        embedding_a,
        embedding_b,
        label
    ):

        distance = euclidean_distance(
            embedding_a,
            embedding_b
        )

        # Same user:
        # minimize distance
        #
        # Different user:
        # push distance beyond margin

        positive_loss = (
            label *
            torch.pow(distance, 2)
        )

        negative_loss = (
            (1 - label) *
            torch.pow(
                torch.clamp(
                    self.margin - distance,
                    min=0.0
                ),
                2
            )
        )

        loss = positive_loss + negative_loss

        return loss.mean()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING SIAMESE BEHAVIORAL MODEL")
    print("=" * 60)

    # Our current features.py creates 11 features.
    input_size = 13

    model = SiameseNetwork(
        input_size=input_size,
        embedding_size=32
    )

    print("\nModel:")
    print(model)

    # Create fake behavioral samples
    sample_a = torch.randn(4, input_size)
    sample_b = torch.randn(4, input_size)

    # Run them through the network
    embedding_a, embedding_b = model(
        sample_a,
        sample_b
    )

    print("\nInput shape:")
    print(sample_a.shape)

    print("\nEmbedding A shape:")
    print(embedding_a.shape)

    print("\nEmbedding B shape:")
    print(embedding_b.shape)

    # Calculate distance
    distance = euclidean_distance(
        embedding_a,
        embedding_b
    )

    print("\nDistances:")
    print(distance)

    # Test loss
    labels = torch.tensor(
        [1, 1, 0, 0],
        dtype=torch.float32
    )

    criterion = ContrastiveLoss(
        margin=1.0
    )

    loss = criterion(
        embedding_a,
        embedding_b,
        labels
    )

    print("\nContrastive loss:")
    print(loss.item())

    print("\nModel test completed successfully.")