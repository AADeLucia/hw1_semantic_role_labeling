#!/bin/bash

ROLES=("AGENT" "BENEFICIARY" "DESTINATION" "PATIENT" "PRODUCT")

for role in "${ROLES[@]}"
do
  EXP_DIR="${role}_experiment"
  OUTPUT_FILE="${EXP_DIR}/predictions.json"
  export SRL_TRAIN="data/${role}_train.csv"
  export SRL_DEV="data/${role}_dev.csv"
  export SRL_TEST="data/${role}_test.csv"

  ## Predict ##
  echo "Predicting VALIDATION set"
  allennlp predict \
    "${EXP_DIR}/model.tar.gz" \
    "${SRL_DEV}" \
    --predictor "srl_predictor.SRLPredictor" \
    --output-file "${OUTPUT_FILE}.dev" \
    --use-dataset-reader \
    --batch-size 10 \
    --silent

  echo "Predicting TEST set"
  allennlp predict \
    "${EXP_DIR}/model.tar.gz" \
    "${SRL_TEST}" \
    --predictor "srl_predictor.SRLPredictor" \
    --output-file "${OUTPUT_FILE}.test" \
    --use-dataset-reader \
    --batch-size 10 \
    --silent

  echo "Done with role ${role}"

done

bash evaluate.sh
