#!/usr/bin/env bash
: ${MODEL_NAME:=model}
: ${PORT:=9000}
if [ -z $SERVING_MODEL ]; then
  export SERVING_MODEL=$MODEL_URL
fi
set -xe

# Fetch models from storage
stored sync $SERVING_MODEL /model/1/

# Start the model server
tensorflow_model_server \
  --port=${PORT} \
  --model_name=${MODEL_NAME} \
  --model_base_path=/model
