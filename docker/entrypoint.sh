#!/usr/bin/env bash
set -xe

# fetch key
echo $GCLOUD_ACCOUNT | base64 -d > /opt/key.json
echo -e "[Credentials]\ngs_service_key_file = /opt/key.json" > /opt/boto.cfg
export BOTO_PATH=/opt/boto.cfg

# fetch model files
mkdir -p /model/1
gsutil cat $SERVING_MODEL | bsdtar -xf- -C /model/1/

# fetch model spec if needed
if [[ $SERVING_MODEL_SPEC = gs://* ]]; then
	export SERVING_MODEL_SPEC=$(gsutil cat $SERVING_MODEL_SPEC | base64 | tr -d '\n')
fi

cd /opt
foreman start
