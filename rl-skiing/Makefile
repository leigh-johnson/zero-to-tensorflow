NOW=$(shell date +"%Y%m%d_%H%M%S")
GCS_BUCKET=gs://from-zero-to-tensorflow
OUTPUT_DIR=${GCS_BUCKET}/skiing-v0-${NOW}
LEARNING_RATE=0.05
PYTHON_VERSION=3.5
REGION=us-central1
TENSORFLOW_VERSION=1.13
CHECKPOINT_STEPS=10
N_EPOCH=1000
RESTORE=FALSE

.PHONY: ml-engine-job

ml-engine-job:
	gcloud ml-engine jobs submit training skiing_v0_${NOW} \
		--job-dir ${OUTPUT_DIR} \
	 	--python-version ${PYTHON_VERSION} \
		--package-path "./trainer" \
		--module-name "trainer.task" \
		--region ${REGION} \
		--runtime-version ${TENSORFLOW_VERSION} \
		-- \
		--learning-rate ${LEARNING_RATE} \
		--save-checkpoint-steps ${CHECKPOINT_STEPS} \
		--n-epoch ${N_EPOCH} \
		--restore ${RESTORE}
