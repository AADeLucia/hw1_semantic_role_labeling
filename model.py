"""
Author: Alexandra DeLucia
"""
from allennlp.models import Model
import torch
import torch.nn.functional as F
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import F1Measure
from allennlp.nn.util import get_text_field_mask
from typing import Dict, Optional


@Model.register("my_model")
class MyModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = torch.nn.Linear(in_features=encoder.get_output_dim()*2,
                                           out_features=vocab.get_vocab_size("labels"))
        self._metric = F1Measure(positive_label=1)

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                argument_idx: torch.Tensor,
                predicate_idx: torch.Tensor,
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        output: Dict[str, torch.Tensor] = {}
        # Get the embedded/encoded representation of the sentence
        mask = get_text_field_mask(tokens)
        embedded = self._embedder(tokens)
        encoded = self._encoder(embedded, mask)

        # Extract the argument and predicate
        # Represent instance as concatenated argument and predicate
        batch_size = encoded.shape[0]
        feature_dim = self._encoder.get_output_dim() * 2
        representation = torch.empty((batch_size, feature_dim))
        for i, (encoding, arg_idx, pred_idx) in enumerate(zip(encoded, argument_idx, predicate_idx)):
            representation[i] = torch.cat([encoding[arg_idx], encoding[pred_idx]], dim=-1)

        # Get class probabilities
        classified = self._classifier(representation)
        output["logits"] = classified

        # Score model
        self._metric(classified, label)
        if label is not None:
            output["loss"] = F.binary_cross_entropy_with_logits(classified[:,1], label.double())
        return output

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        """Binary F1"""
        return self._metric.get_metric(reset)
