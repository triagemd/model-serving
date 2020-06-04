#!/usr/bin/env bash
: ${MODEL_NAME:=model}
: ${PORT:=9000}
set -xe

# Fetch models from storage
mkdir -p /model/1
wget -c $MODEL_URL -O - | bsdtar -xf- -C /model/1/

# Start the model server
tensorflow_model_server \
  --port=${PORT} \
  --model_name=${MODEL_NAME} \
  --model_base_path=/model
