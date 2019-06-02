NOW=$(shell date +"%Y%m%d_%H%M%S")
GCS_BUCKET=gs://from-zero-to-tensorflow
OUTPUT_DIR=${GCS_BUCKET}/skiing-v0-${NOW}
LEARNING_RATE=0.001
PYTHON_VERSION=3.5
REGION=us-central1
TENSORFLOW_VERSION=1.13
CHECKPOINT_STEPS=1
N_EPOCH=1000
RESTORE=FALSE
SCALE_TIER=basic
#MASTER_ACCELERATOR="type=nvidia-tesla-k80,count=1"

.PHONY: ai-platform-job

ai-platform-job:
	gcloud beta ai-platform jobs submit training skiing_v0_${NOW} \
		--job-dir ${OUTPUT_DIR} \
	 	--python-version ${PYTHON_VERSION} \
		--package-path ./trainer \
		--module-name trainer.task \
		--region ${REGION} \
		--runtime-version ${TENSORFLOW_VERSION} \
		--scale-tier ${SCALE_TIER} \
		--stream-logs \
		-- \
		--learning-rate ${LEARNING_RATE} \
		--save-checkpoint-steps ${CHECKPOINT_STEPS} \
		--n-epoch ${N_EPOCH} \
		--restore ${RESTORE} \
		--render FALSE \
