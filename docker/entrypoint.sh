#!/usr/bin/env bash
set -xe

# fetch key
echo $GCLOUD_ACCOUNT | base64 -d > /opt/key.json

# fetch model files
python3 -m gsutil cat $SERVING_MODEL | bsdtar -xf- -C /model/1/

# fetch model spec if needed
if [[ $SERVING_MODEL_SPEC = gs://* ]]; then
	export SERVING_MODEL_SPEC=$(python3 -m gsutil cat $SERVING_MODEL_SPEC | base64 | tr -d '\n')
fi

cd /opt
foreman start
