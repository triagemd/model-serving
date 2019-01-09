#!/usr/bin/env bash
set -xe

pip install --upgrade pip stored
stored sync --force-unpack $SERVING_MODEL /model/1

cd /opt
foreman start
