runai workspace submit prefix \
  --project strophf1 \
  --image docker-public-local.artifactory.jhuapl.edu/itsdai/runai/idp-fips-ngc2505pytorch:0.1 \
  --preemptible \
  --gpu-devices-request 8 \
  --cpu-core-request 4 \
  --cpu-memory-request 20G \
  --existing-pvc claimname=prefix-data-project-zhncd,path=/home/apluser \
  --node-pools dgx-h100-80gb \
  --node-pools dgx-h100-80gb-alt \
  --node-pools dgx-h100-80gb-alt2 \
  --node-pools abyss-hgx-h100-80gb \
  # --node-pools aos-a40-48gb \
  # --node-pools itsd-general \
  # --node-pools default \
  --external-url container=8888 \
  --run-as-user \
  --environment HOME=/home/apluser \
  --environment USER=apluser
