from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class WikipediaSentenceDataset(Dataset):
    """
    Dataset for unsupervised SimCSE.

    Each non-empty line in the input file represents one
    training sentence.

    The dataset itself returns one raw sentence. The two
    identical views required by unsupervised SimCSE are created
    by SimCSECollator.

    This mirrors the behavior of the official SimCSE
    preprocessing:

        sentence
            |
            +---- view 1
            |
            +---- view 2

    The two tokenized views are identical.

    Stochasticity is introduced later by BERT dropout during
    the forward pass.
    """

    def __init__(
        self,
        file_path: str,
        max_sentences: int | None = None,
    ):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {self.file_path}"
            )

        self.sentences = self._load_sentences(
            file_path=self.file_path,
            max_sentences=max_sentences,
        )

        if not self.sentences:
            raise ValueError(
                f"No valid sentences found in {self.file_path}"
            )

    @staticmethod
    def _load_sentences(
        file_path: Path,
        max_sentences: int | None = None,
    ) -> List[str]:
        """
        Load non-empty lines from the training file.

        Empty lines are ignored.

        max_sentences limits the number of valid sentences
        loaded, not the number of physical lines read.
        """

        sentences: List[str] = []

        with file_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:

            for line in file:

                sentence = line.strip()

                if not sentence:
                    continue

                sentences.append(sentence)

                if (
                    max_sentences is not None
                    and len(sentences) >= max_sentences
                ):
                    break

        return sentences

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> str:
        return self.sentences[index]


class SimCSECollator:
    """
    Tokenize and dynamically pad two identical views of each
    sentence.

    Input:

        [
            "A dog is running.",
            "A cat is sleeping.",
        ]

    Output:

        input_ids:
            [batch_size, 2, sequence_length]

        attention_mask:
            [batch_size, 2, sequence_length]

    For every example:

        view 0 == view 1

    The two views are intentionally identical at the token
    level.

    Independent dropout masks in the model create the two
    stochastic representations during training.

    This follows the structure used by the official SimCSE
    preprocessing and model forward pass.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 32,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        sentences: List[str],
    ) -> Dict[str, torch.Tensor]:

        if not sentences:
            raise ValueError(
                "SimCSECollator received an empty batch."
            )

        # -----------------------------------------------------
        # Create two identical sentence views.
        #
        # Official SimCSE effectively constructs:
        #
        #   [sentence_1, sentence_2, ...]
        #   [sentence_1, sentence_2, ...]
        #
        # before tokenization.
        # -----------------------------------------------------

        two_views = [
            sentence
            for sentence in sentences
            for _ in range(2)
        ]

        # -----------------------------------------------------
        # Tokenization
        #
        # We intentionally do not pad to max_length here.
        # Padding is dynamic within the current batch, matching
        # the official behavior when pad_to_max_length=False.
        # -----------------------------------------------------

        tokenized = self.tokenizer(
            two_views,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # -----------------------------------------------------
        # Convert:
        #
        #   [2 * batch_size, seq_len]
        #
        # into:
        #
        #   [batch_size, 2, seq_len]
        # -----------------------------------------------------

        batch_size = len(sentences)

        result: Dict[str, torch.Tensor] = {}

        for key, value in tokenized.items():

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)

            if value.shape[0] != 2 * batch_size:
                raise ValueError(
                    f"Unexpected first dimension for "
                    f"{key}: {value.shape}"
                )

            result[key] = value.view(
                batch_size,
                2,
                -1,
            )

        return result