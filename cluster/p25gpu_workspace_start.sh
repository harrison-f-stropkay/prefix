#!/usr/bin/env bash

runai workspace submit prefix-p25gpu \
  --project strophf1 \
  --image docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1 \
  --gpu-portion-request 0.25 \
  --cpu-core-request 64 \
  --cpu-memory-request 512G \
  --existing-pvc claimname=prefix-data-10tib-project-ej4an,path=/home/apluser \
  --node-pools dgx-h100-80gb \
  --node-pools dgx-h100-80gb-alt \
  --node-pools dgx-h100-80gb-alt2 \
  --node-pools abyss-hgx-h100-80gb \
  --node-pools  aos-a40-48gb \
  --node-pools itsd-general \
  --node-pools default \
  --external-url container=8888 \
  --run-as-user \
  --preemptible \
  --environment HOME=/home/apluser \
  --environment USER=apluser
