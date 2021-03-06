"""
Semantic Role Labeling Dataset Reader for AllenNLP
Based off the guide https://jbarrow.ai/allennlp-the-hard-way-1/
https://github.com/decomp-sem/neural-sprl/blob/master/data/mini.sprl

Author: Alexandra DeLucia
"""
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.fields import Field, TextField, LabelField, SpanField, IndexField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.instance import Instance
from typing import Dict, List, Iterator
import csv


@DatasetReader.register("srl_reader")
class SRLDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterator[Instance]:
        """handle filtering out all but the instances that each particular worker should generate"""
        # CSV of form
        # sentence,argument_head,predicate_head,role,label
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            for i, row in enumerate(reader):
                if i == 0:  # Skip header
                    continue
                sentence_id = row[0].strip().split()
                tokens = row[1].strip().split()
                argument_idx = int(row[2].strip())
                predicate_idx = int(row[3].strip())
                role = row[4].strip()
                label = row[5].strip()
                yield self.text_to_instance(tokens, argument_idx, predicate_idx, label)

    def text_to_instance(self,
                         tokens: List[str],
                         argument_idx: int,
                         predicate_idx: int,
                         label: int) -> Instance:
        fields: Dict[str, Field] = {}
        # Wrap each token in the file with a token object
        tokens = TextField([Token(t) for t in tokens], self._token_indexers)

        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["tokens"] = tokens
        fields["argument_idx"] = IndexField(argument_idx, sequence_field=tokens)
        fields["predicate_idx"] = IndexField(predicate_idx, sequence_field=tokens)
        fields["label"] = LabelField(label)
        return Instance(fields)
