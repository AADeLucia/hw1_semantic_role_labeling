#!/bin/bash

EXP_DIR="temp"
rm -r "${EXP_DIR}"

allennlp train \
  --dry-run \
  -s "${EXP_DIR}" "configs/agent.jsonnet"
