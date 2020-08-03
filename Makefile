.EXPORT_ALL_VARIABLES:
TRAINING_PACKAGE_PATH=trainer
MAIN_TRAINER_MODULE=trainer.task
BUCKET_NAME=model_registry
JOB_NAME="poc_training_pipeline_$(date +"%Y%m%d_%H%M%S")"
JOB_DIR=gs://${BUCKET_NAME}/scikit_learn_job_dir
REGION=us-central1
RUNTIME_VERSION=2.1
PYTHON_VERSION=3.7
SCALE_TIER=BASIC


test_local:
	gcloud ai-platform local train \
		--package-path ${TRAINING_PACKAGE_PATH} \
		--module-name ${MAIN_TRAINER_MODULE}

train:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir ${JOB_DIR} \
		--package-path ${TRAINING_PACKAGE_PATH} \
		--module-name ${MAIN_TRAINER_MODULE} \
		--region ${REGION} \
		--runtime-version=${RUNTIME_VERSION} \
		--python-version=${PYTHON_VERSION} \
		--scale-tier ${SCALE_TIER}

describe_job:
	gcloud ai-platform jobs describe ${JOB_NAME}

stream_job_logs:
	gcloud ai-platform jobs stream-logs ${JOB_NAME}


