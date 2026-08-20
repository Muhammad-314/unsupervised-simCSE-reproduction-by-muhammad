import torch
import torch.nn as nn


class SimCSEPooler(nn.Module):
    """
    Sentence pooling methods used by SimCSE.

    Supported poolers:

        cls
        cls_before_pooler
        avg
        avg_top2
        avg_first_last

    Important:

        The pooler does NOT own the MLP.

        The MLP belongs to SimCSEModel and is passed to
        forward() when the "cls" pooler is used.

    This prevents the same MLP from appearing twice in
    model.state_dict().
    """

    VALID_POOLERS = {
        "cls",
        "cls_before_pooler",
        "avg",
        "avg_top2",
        "avg_first_last",
    }

    def __init__(
        self,
        pooler_type: str = "cls_before_pooler",
    ):
        super().__init__()

        if pooler_type not in self.VALID_POOLERS:
            raise ValueError(
                f"Unknown pooler_type: {pooler_type}. "
                f"Expected one of: "
                f"{sorted(self.VALID_POOLERS)}"
            )

        self.pooler_type = pooler_type

    @staticmethod
    def _mean_pooling(
        hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool token representations while ignoring padding.
        """

        mask = attention_mask.unsqueeze(-1).to(
            hidden_state.dtype
        )

        masked_hidden = (
            hidden_state * mask
        )

        summed = masked_hidden.sum(
            dim=1
        )

        counts = mask.sum(
            dim=1
        ).clamp_min(1e-9)

        return summed / counts

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        all_hidden_states=None,
        mlp: nn.Module | None = None,
    ) -> torch.Tensor:
        """
        Apply the requested pooling strategy.

        Args:
            last_hidden_state:
                [batch, sequence_length, hidden_size]

            attention_mask:
                [batch, sequence_length]

            all_hidden_states:
                Required for avg_top2 and avg_first_last.

            mlp:
                Optional external MLP.

                Required only for pooler_type="cls".

                The MLP is deliberately passed at call time
                instead of being registered as a child module.
        """

        if last_hidden_state.ndim != 3:
            raise ValueError(
                "last_hidden_state must have shape "
                "[batch, sequence_length, hidden_size], "
                f"got {tuple(last_hidden_state.shape)}"
            )

        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must have shape "
                "[batch, sequence_length], "
                f"got {tuple(attention_mask.shape)}"
            )

        if (
            last_hidden_state.shape[0]
            != attention_mask.shape[0]
        ):
            raise ValueError(
                "Batch dimensions do not match."
            )

        if (
            last_hidden_state.shape[1]
            != attention_mask.shape[1]
        ):
            raise ValueError(
                "Sequence dimensions do not match."
            )

        # -----------------------------------------------------
        # CLS + MLP
        # -----------------------------------------------------

        if self.pooler_type == "cls":

            if mlp is None:
                raise ValueError(
                    "pooler_type='cls' requires "
                    "an MLP passed to forward()."
                )

            cls_embedding = (
                last_hidden_state[:, 0]
            )

            return mlp(
                cls_embedding
            )

        # -----------------------------------------------------
        # Raw CLS
        # -----------------------------------------------------

        if self.pooler_type == "cls_before_pooler":

            return last_hidden_state[:, 0]

        # -----------------------------------------------------
        # Final-layer mean pooling
        # -----------------------------------------------------

        if self.pooler_type == "avg":

            return self._mean_pooling(
                hidden_state=last_hidden_state,
                attention_mask=attention_mask,
            )

        # -----------------------------------------------------
        # Average of final two layers, then mean pooling
        # -----------------------------------------------------

        if self.pooler_type == "avg_top2":

            if all_hidden_states is None:
                raise ValueError(
                    "avg_top2 requires "
                    "all_hidden_states."
                )

            if len(all_hidden_states) < 2:
                raise ValueError(
                    "avg_top2 requires at least "
                    "two hidden states."
                )

            top2 = (
                all_hidden_states[-1]
                + all_hidden_states[-2]
            ) / 2.0

            return self._mean_pooling(
                hidden_state=top2,
                attention_mask=attention_mask,
            )

        # -----------------------------------------------------
        # Average of first encoder layer and final layer
        # -----------------------------------------------------

        if self.pooler_type == "avg_first_last":

            if all_hidden_states is None:
                raise ValueError(
                    "avg_first_last requires "
                    "all_hidden_states."
                )

            if len(all_hidden_states) < 2:
                raise ValueError(
                    "avg_first_last requires at least "
                    "two hidden states."
                )

            first_last = (
                all_hidden_states[1]
                + all_hidden_states[-1]
            ) / 2.0

            return self._mean_pooling(
                hidden_state=first_last,
                attention_mask=attention_mask,
            )

        raise RuntimeError(
            f"Unsupported pooler: {self.pooler_type}"
        )