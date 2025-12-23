#!/usr/bin/env bash

runai workspace submit prefix-cpu \
  --project strophf1 \
  --image docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1 \
  --preemptible \
  --cpu-core-request 64 \
  --cpu-memory-request 512G \
  --existing-pvc claimname=prefix-data-project-zhncd,path=/home/apluser \
  --external-url container=8888 \
  --run-as-user \
  --environment HOME=/home/apluser \
  --environment USER=apluser
