INPUT=$1
RUN_ENV=$2
RUN_TYPE=$3
JOB_DIR=$4
EXTRA_TRAINER_ARGS=$5

if [[ ! "$RUN_ENV" =~ ^(local|remote)$]]; then
    RUN_ENV=local;
fi

if [[ ! "$RUN_TYPE" =~ ^(train|hptuning)$ ]]; then
    RUN_TYPE=train;
fi

NOW="$(date +"%Y%m%d_%H%M%S")"
JOB_NAME="poc_${RUN_TYPE}_${NOW}"
PACKAGE_PATH=trainer
TRAINER_MODULE=$PACKAGE_PATH.task
REGION=us-central1

if [ "$RUN_TYPE" = 'hptuning'  ]; then
    CONFIG_FILE=config/hptuning.yaml
else  # Assume `train`
    CONFIG_FILE=config/config.yaml
fi

# Specify arguments for remote (AI Platform) or local (on-premise) execution
echo "$RUN_ENV"
if [ "$RUN_ENV" = 'remote' ]; then
    RUN_ENV_ARGS="jobs submit training $JOB_NAME \
                    --region $REGION \
                    --config $CONFIG_FILE \
                    "
else  # assume `local`
    RUN_ENV_ARGS="local train"
fi

# Specify arguments to pass to the trainer module (trainer/task.py)
TRAINER_ARGS="\
              --input $INPUT \
            "

CMD="gcloud ml-engine $RUN_ENV_ARGS \
                --job-dir $JOB_DIR \
                --package-path $PACKAGE_PATH \
                --module-name $MAIN_TRAINER_MODULE \
                -- \
                $TRAINER_ARGS \
                $EXTRA_TRAINER_ARGS \
                "

echo "Running command: $CMD"
eval "$CMD"

