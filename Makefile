.EXPORT_ALL_VARIABLES:
REGION=us-central1
TRAINING_PACKAGE_PATH=trainer
MAIN_TRAINER_MODULE=trainer.task
BUCKET_NAME=model_registry
JOB_NAME=poc_training_pipeline_v1
JOB_DIR=gs://${BUCKET_NAME}/training_pipeline
RUNTIME_VERSION=1.15
INFERENCE_RUNTIME_VERSION=1.15
PYTHON_VERSION=3.7
SCALE_TIER=BASIC
MODEL_NAME=poc_model
MODEL_VERSION=v1
MODEL_DIRECTORY=gs://${BUCKET_NAME}/iris_20200804_010538/
INFERENCE_URI=gs://${BUCKET_NAME}/inference/predictor-0.1.tar.gz
PREDICTION_CLASS=predictor.Predictor

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

create_model:
	gcloud ai-platform models create ${MODEL_NAME} --regions ${REGION}

create_version:
	gcloud beta ai-platform versions create ${MODEL_VERSION} \
		--model ${MODEL_NAME} \
		--runtime-version ${INFERENCE_RUNTIME_VERSION} \
		--python-version ${PYTHON_VERSION} \
		--origin ${MODEL_DIRECTORY} \
		--package-uris ${INFERENCE_URI} \
		--prediction-class ${PREDICTION_CLASS}


