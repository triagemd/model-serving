#!/usr/bin/env bash
set -xe

stored sync --force-unpack $SERVING_MODEL /model/1

cd /opt
foreman start
