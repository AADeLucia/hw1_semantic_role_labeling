from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from typing import Dict, List, Iterator, Any


@Predictor.register("srl_predictor")
class SRLPredictor(Predictor):

    def predict_batch_instance(self, instances: List[Instance]) -> JsonDict:
        # Get model output
        outputs = self._model.forward_on_instances(instances)

        # Add more instance information
        for output, instance in zip(outputs, instances):
            output["tokens"] = [str(token) for token in instance.fields["tokens"].tokens]
            output["predicted"] = output["logits"].argmax()
            output["labels"] = instance.fields["label"].label

        return sanitize(outputs)
