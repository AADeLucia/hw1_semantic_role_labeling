#!/bin/bash

ROLES=("AGENT" "BENEFICIARY" "DESTINATION" "PATIENT" "PRODUCT")

for role in "${ROLES[@]}"
do
  EXP_DIR="${role}_experiment"
  OUTPUT_FILE="${EXP_DIR}/predictions.json"
  export SRL_TRAIN="data/${role}_train.csv"
  export SRL_DEV="data/${role}_dev.csv"
  export SRL_TEST="data/${role}_test.csv"

  # Clear previous run
  rm -r "${EXP_DIR}"

  ## Train ##
  allennlp train \
    -s "${EXP_DIR}" \
    "configs/config_glove_lstm.jsonnet"

  if [ $? -ne 0 ]
  then
    echo "Error during training. Skipping validation."
    exit 1
  fi

  echo "Done with role ${role}"

done

echo "Done."
