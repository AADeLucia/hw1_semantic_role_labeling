"""
Author: Alexandra DeLucia
"""
from allennlp.models import Model
import torch
import torch.nn as nn
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.training.metrics import F1Measure
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from typing import Dict, Optional


@Model.register('my_model')
class MyModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder) -> None:
        super().__init__(vocab)

        self._embedder = embedder
        self._encoder = encoder
        self._classifier = nn.Linear(in_features=encoder.get_output_dim(),
                                     out_features=vocab.get_vocab_size('labels'))
        self._f1 = F1Measure(vocab, 'labels')

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        """Binary F1"""
        return self._f1.get_metric(reset)
