"""
Evaluate model performance

Author: Alexandra DeLucia
"""
import argparse
from sklearn.metrics import precision_recall_fscore_support
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", nargs="+")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for prediction_file in args.input_files:
        predictions, ground_truth = [], []
        with open(prediction_file) as f:
            for line in f.readlines():
                results = json.loads(line)
                pred = results.get("predicted")
                gt = results.get("labels")

                if not pred or not gt:
                    continue

                predictions.append(int(pred))
                ground_truth.append(1-int(gt))  # Hack because of flipped model labels

        # Calculate F1 metrics
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true=ground_truth,
            y_pred=predictions,
            average="binary",
            pos_label=1,
            zero_division=0
        )
        print(f"{prediction_file}:\nF1: {f1:.3}\tPrecision: {prec:.3}\tRecall: {rec:.3}")
