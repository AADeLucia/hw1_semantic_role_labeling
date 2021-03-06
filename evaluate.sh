#!/bin/bash

ROLES=("AGENT" "BENEFICIARY" "DESTINATION" "PATIENT" "PRODUCT")

for role in "${ROLES[@]}"
do
  EXP_DIR="${role}_experiment"
  OUTPUT_FILE="${EXP_DIR}/predictions.json"

  echo "Evaluating predictions"
  python evaluate.py \
    --input-files "${OUTPUT_FILE}.dev" "${OUTPUT_FILE}.test"

  echo "Done with role ${role}"

done

echo "Done."
