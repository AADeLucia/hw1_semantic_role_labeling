"""
Semantic Role Labeling Dataset Reader for AllenNLP
Based off the guide https://jbarrow.ai/allennlp-the-hard-way-1/
https://github.com/decomp-sem/neural-sprl/blob/master/data/mini.sprl

Author: Alexandra DeLucia
"""
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, LabelField, AdjacencyField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.instance import Instance
from typing import Dict, List, Iterator, Tuple


@DatasetReader.register("srl_reader")
class SRLDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        """handle filtering out all but the instances that each particular worker should generate"""
        with open(file_path) as f:
            # CSV of form
            # sentence,argument_head,predicate_head,role,label
            for line in f.readlines()[1:]:  # Skip header
                sentence, argument_head, predicate_head, role, label = [l.strip() for l in line.split(",")]
                tokens = sentence.split()
                loc = (argument_head, predicate_head)
                yield self.text_to_instance(tokens, loc, label)

    def text_to_instance(self,
                         words: List[str],
                         arg_pred_idx: Tuple[int, int],
                         label: int) -> Instance:
        fields: Dict[str, Field] = {}
        # Wrap each token in the file with a token object
        tokens = TextField([Token(w) for w in words], self._token_indexers)
        edge = AdjacencyField([arg_pred_idx], tokens)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["role_edge"] = edge
        fields["label"] = LabelField(label)
        return Instance(fields)
