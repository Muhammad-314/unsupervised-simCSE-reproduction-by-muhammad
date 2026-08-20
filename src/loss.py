import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCSELoss(nn.Module):
    """
    Unsupervised SimCSE contrastive loss.

    Given:

        z1: [batch_size, hidden_size]
        z2: [batch_size, hidden_size]

    construct:

        similarity[i, j]
            = cosine(z1[i], z2[j]) / temperature

    The positive pair for example i is:

        similarity[i, i]

    All off-diagonal entries are in-batch negatives.

    Therefore the target labels are:

        [0, 1, 2, ..., batch_size - 1]

    This follows the official unsupervised SimCSE objective.
    """

    def __init__(
        self,
        temperature: float = 0.05,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        self.temperature = temperature

        self.cross_entropy = (
            nn.CrossEntropyLoss()
        )

    def similarity_matrix(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the temperature-scaled cosine similarity
        matrix.

        Args:
            z1:
                [batch_size, hidden_size]

            z2:
                [batch_size, hidden_size]

        Returns:
            [batch_size, batch_size]
        """

        if z1.ndim != 2:
            raise ValueError(
                "z1 must have shape "
                "[batch_size, hidden_size], "
                f"got {tuple(z1.shape)}"
            )

        if z2.ndim != 2:
            raise ValueError(
                "z2 must have shape "
                "[batch_size, hidden_size], "
                f"got {tuple(z2.shape)}"
            )

        if z1.shape != z2.shape:
            raise ValueError(
                "z1 and z2 must have identical shapes. "
                f"Got {tuple(z1.shape)} and "
                f"{tuple(z2.shape)}"
            )

        # -----------------------------------------------------
        # Normalize each embedding.
        #
        # cosine similarity becomes a matrix multiplication.
        # -----------------------------------------------------

        z1 = F.normalize(
            z1,
            p=2,
            dim=1,
        )

        z2 = F.normalize(
            z2,
            p=2,
            dim=1,
        )

        # -----------------------------------------------------
        # Every z1 is compared against every z2.
        #
        # [batch, hidden]
        #      @
        # [hidden, batch]
        #      =
        # [batch, batch]
        # -----------------------------------------------------

        similarity = torch.matmul(
            z1,
            z2.transpose(
                0,
                1,
            ),
        )

        # -----------------------------------------------------
        # Temperature scaling.
        # -----------------------------------------------------

        similarity = (
            similarity
            / self.temperature
        )

        return similarity

    def labels(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Construct diagonal positive-pair labels.
        """

        return torch.arange(
            batch_size,
            device=device,
            dtype=torch.long,
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the unsupervised SimCSE loss.
        """

        similarity = self.similarity_matrix(
            z1=z1,
            z2=z2,
        )

        target = self.labels(
            batch_size=z1.shape[0],
            device=z1.device,
        )

        loss = self.cross_entropy(
            similarity,
            target,
        )

        return loss

    def diagnostics(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> dict:
        """
        Return useful training diagnostics.

        These are NOT required for the loss itself.

        Returns:

            loss
            positive_similarity
            negative_similarity

        Similarities are reported before temperature scaling.
        """

        # Raw cosine similarities.
        normalized_z1 = F.normalize(
            z1,
            p=2,
            dim=1,
        )

        normalized_z2 = F.normalize(
            z2,
            p=2,
            dim=1,
        )

        cosine = torch.matmul(
            normalized_z1,
            normalized_z2.transpose(
                0,
                1,
            ),
        )

        # Temperature-scaled matrix for CE.
        similarity = (
            cosine
            / self.temperature
        )

        target = self.labels(
            batch_size=z1.shape[0],
            device=z1.device,
        )

        loss = self.cross_entropy(
            similarity,
            target,
        )

        positive_similarity = (
            cosine.diag().mean()
        )

        if cosine.shape[0] > 1:

            mask = ~torch.eye(
                cosine.shape[0],
                dtype=torch.bool,
                device=cosine.device,
            )

            negative_similarity = (
                cosine[mask].mean()
            )

        else:
            negative_similarity = torch.tensor(
                float("nan"),
                device=cosine.device,
            )

        return {
            "loss": loss,
            "positive_similarity": (
                positive_similarity
            ),
            "negative_similarity": (
                negative_similarity
            ),
        }