"""Training pipeline"""
import datetime
import os
import subprocess
import sys
import argparse

import pandas as pd
from sklearn import ensemble
from sklearn.externals import joblib

from .features import download_data

# Fill in your Cloud Storage bucket name
BUCKET_NAME = 'model_registry'

# [START download-data]
FEATURE_MATRIX = 'iris_data.csv'
TARGET = 'iris_target.csv'
data_dir = 'gs://cloud-samples-data/ml-engine/iris'
# data_dir = 'gs://data-movies/
# filename = movies_metadata.csv'
training_data = os.path.join(data_dir, FEATURE_MATRIX)
training_labels = os.path.join(data_dir, TARGET)

def main(args):
    download_data(
        inputs=[training_data, training_labels],
        outputs=[args.feature_file, args.target_file],
    )

    # [START load-into-pandas]
    # Load data into pandas, then use `.values` to get NumPy arrays
    X = pd.read_csv(args.feature_file).values
    y = pd.read_csv(args.target_file).values
    # Convert one-column 2D array into 1D array for use with scikit-learn
    y = y.reshape((y.size,))
    # [END load-into-pandas]
    # TODO: move to models module
    rf_classifier = ensemble.RandomForestClassifier(verbose=1)
    rf_classifier.fit(X, y)
    # Export the classifier to a file
    # TODO: can be pickle as well
    joblib.dump(rf_classifier, args.output)

    # [START upload-model]
    # Upload the saved model file to Cloud Storage
    # TODO: using date as RUN ID
    # Add BinaryArg for True/False args
    if args.push_to_gcs == 1:
        current_date = datetime.datetime.now().strftime(f'{args.task_name}_%Y%m%d_%H%M%S')
        gcs_model_path = os.path.join('gs://', BUCKET_NAME, current_date, args.output)
        subprocess.check_call(
            ['gsutil', 'cp', args.output, gcs_model_path],
            stderr=sys.stdout,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParse()
    parser.add_argument(
        '--feature-file',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--target-file',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default='training_pipeline.joblib',
    )
    parser.add_argument(
        '--push-to-gcs',
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        '--task-name',
        type=str,
        required=False,
        default='movies',
    )
    args = parser.parse_args()
    main(args)
