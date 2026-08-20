import torch
import torch.nn as nn
from transformers import BertModel

from pooler import SimCSEPooler


class SimCSEModel(nn.Module):
    """
    BERT-base model for unsupervised SimCSE.

    Training:

        BERT
          ↓
        CLS
          ↓
        MLP
          ↓
        z1 / z2

    Evaluation with mlp_only_train=True:

        BERT
          ↓
        raw CLS
          ↓
        sentence embedding

    The MLP is owned only by SimCSEModel.

    Poolers do not register the MLP as a child module.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        mlp_only_train: bool = True,
    ):
        super().__init__()

        self.model_name = model_name

        self.mlp_only_train = (
            mlp_only_train
        )

        # -----------------------------------------------------
        # BERT
        # -----------------------------------------------------

        self.bert = BertModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
        )

        hidden_size = (
            self.bert.config.hidden_size
        )

        # -----------------------------------------------------
        # SimCSE MLP
        # -----------------------------------------------------

        self.mlp = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.Tanh(),
        )

        self._initialize_mlp()

        # -----------------------------------------------------
        # Poolers
        #
        # These contain NO reference to self.mlp.
        # -----------------------------------------------------

        self.train_pooler = SimCSEPooler(
            pooler_type="cls",
        )

        self.eval_pooler = SimCSEPooler(
            pooler_type="cls_before_pooler",
        )

    def _initialize_mlp(self):
        """
        Initialize the SimCSE MLP using BERT's initializer.

        BERT-base:

            std = 0.02
            bias = 0
        """

        initializer_range = (
            self.bert.config.initializer_range
        )

        linear = self.mlp[0]

        nn.init.normal_(
            linear.weight,
            mean=0.0,
            std=initializer_range,
        )

        nn.init.zeros_(
            linear.bias,
        )

    # =========================================================
    # BERT encoding
    # =========================================================

    def encode_bert(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ):
        """
        Run BERT.

        Input:

            input_ids:
                [batch, sequence_length]

            attention_mask:
                [batch, sequence_length]
        """

        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=(
                output_hidden_states
            ),
            return_dict=True,
        )

    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return raw final-layer CLS.
        """

        outputs = self.encode_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.last_hidden_state[:, 0]

    # =========================================================
    # Training
    # =========================================================

    def contrastive_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Encode the two SimCSE views.

        Input:

            [batch, 2, sequence_length]

        Output:

            z1:
                [batch, hidden_size]

            z2:
                [batch, hidden_size]
        """

        if input_ids.ndim != 3:
            raise ValueError(
                "Expected input_ids with shape "
                "[batch_size, 2, sequence_length], "
                f"got {tuple(input_ids.shape)}"
            )

        if attention_mask.ndim != 3:
            raise ValueError(
                "Expected attention_mask with shape "
                "[batch_size, 2, sequence_length], "
                f"got {tuple(attention_mask.shape)}"
            )

        if (
            input_ids.shape
            != attention_mask.shape
        ):
            raise ValueError(
                "input_ids and attention_mask "
                "must have identical shapes."
            )

        batch_size = (
            input_ids.shape[0]
        )

        num_views = (
            input_ids.shape[1]
        )

        sequence_length = (
            input_ids.shape[2]
        )

        if num_views != 2:
            raise ValueError(
                "Unsupervised SimCSE requires "
                f"exactly 2 views, got {num_views}."
            )

        # -----------------------------------------------------
        # [batch, 2, seq]
        # ->
        # [batch * 2, seq]
        # -----------------------------------------------------

        flat_input_ids = (
            input_ids.reshape(
                batch_size * num_views,
                sequence_length,
            )
        )

        flat_attention_mask = (
            attention_mask.reshape(
                batch_size * num_views,
                sequence_length,
            )
        )

        # -----------------------------------------------------
        # BERT
        # -----------------------------------------------------

        outputs = self.encode_bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
        )

        # -----------------------------------------------------
        # CLS -> MLP
        #
        # The MLP is passed to the pooler without being
        # registered inside it.
        # -----------------------------------------------------

        pooled = self.train_pooler(
            last_hidden_state=(
                outputs.last_hidden_state
            ),
            attention_mask=flat_attention_mask,
            mlp=self.mlp,
        )

        # -----------------------------------------------------
        # [batch * 2, hidden]
        # ->
        # [batch, 2, hidden]
        # -----------------------------------------------------

        pooled = pooled.reshape(
            batch_size,
            num_views,
            -1,
        )

        z1 = pooled[:, 0]

        z2 = pooled[:, 1]

        return z1, z2

    # =========================================================
    # Sentence embedding
    # =========================================================

    def sentence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_mlp: bool | None = None,
    ) -> torch.Tensor:
        """
        Produce a single-view sentence embedding.

        Behavior:

            training + mlp_only_train=True
                -> CLS + MLP

            evaluation + mlp_only_train=True
                -> raw CLS

        Explicit use_mlp=True/False overrides the automatic
        behavior.
        """

        outputs = self.encode_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # -----------------------------------------------------
        # Explicit MLP
        # -----------------------------------------------------

        if use_mlp is True:

            return self.train_pooler(
                last_hidden_state=(
                    outputs.last_hidden_state
                ),
                attention_mask=attention_mask,
                mlp=self.mlp,
            )

        # -----------------------------------------------------
        # Explicit raw CLS
        # -----------------------------------------------------

        if use_mlp is False:

            return self.eval_pooler(
                last_hidden_state=(
                    outputs.last_hidden_state
                ),
                attention_mask=attention_mask,
            )

        # -----------------------------------------------------
        # Automatic behavior
        # -----------------------------------------------------

        if (
            self.training
            and self.mlp_only_train
        ):

            return self.train_pooler(
                last_hidden_state=(
                    outputs.last_hidden_state
                ),
                attention_mask=attention_mask,
                mlp=self.mlp,
            )

        return self.eval_pooler(
            last_hidden_state=(
                outputs.last_hidden_state
            ),
            attention_mask=attention_mask,
        )

    # =========================================================
    # Forward
    # =========================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_mlp: bool | None = None,
    ):
        """
        Public interface.

        Two-view input:

            [batch, 2, seq]

        returns:

            z1, z2

        Single-view input:

            [batch, seq]

        returns:

            sentence embedding
        """

        if input_ids.ndim == 3:

            return self.contrastive_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if input_ids.ndim == 2:

            return self.sentence_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_mlp=use_mlp,
            )

        raise ValueError(
            "Expected input_ids with 2 or 3 "
            "dimensions, "
            f"got {tuple(input_ids.shape)}"
        )