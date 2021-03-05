#!/bin/bash

# Clear previous run
EXP_DIR="temp"
rm -r "${EXP_DIR}"

ROLE="AGENT"

## Train ##
allennlp train \
  -s "${EXP_DIR}" \
  "configs/agent.jsonnet"

if [ $? -ne 0 ]
then
  echo "Error during training. Skipping validation."
  exit 1
fi

## Predict ##
OUTPUT_FILE="${EXP_DIR}/${ROLE}_predictions.json"
allennlp predict \
  --predictor "srl_predictor.SRLPredictor" \
  --output-file "${OUTPUT_FILE}" \
  --use-dataset-reader \
  --batch-size 10 \
  --silent \
  "${EXP_DIR}" \
  "data/${ROLE}_dev.csv"
