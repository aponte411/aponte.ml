"""Training pipeline"""
import datetime
import os
import subprocess
import sys
import pandas as pd
from sklearn import svm, ensemble
import joblib

# Fill in your Cloud Storage bucket name
BUCKET_NAME = 'model_registry'
# [END setup]


# [START download-data]
# TODO: separate into features module
iris_data_filename = 'iris_data.csv'
iris_target_filename = 'iris_target.csv'
data_dir = 'gs://cloud-samples-data/ml-engine/iris'

# gsutil outputs everything to stderr so we need to divert it to stdout.
training_data = os.path.join(data_dir, iris_data_filename)
training_labels = os.path.join(data_dir, iris_target_filename)

subprocess.check_call(
    ['gsutil', 'cp', training_data, iris_data_filename],
    stderr=sys.stdout,
)
subprocess.check_call(
    ['gsutil', 'cp', training_labels, iris_target_filename],
    stderr=sys.stdout,
)
# [END download-data]


# [START load-into-pandas]
# Load data into pandas, then use `.values` to get NumPy arrays
iris_data = pd.read_csv(iris_data_filename).values
iris_target = pd.read_csv(iris_target_filename).values

# Convert one-column 2D array into 1D array for use with scikit-learn
iris_target = iris_target.reshape((iris_target.size,))
# [END load-into-pandas]


# [START train-and-save-model]
# Train the models
# TODO: move to models module
svm_classifier = svm.SVC(gamma='auto', verbose=True)
svm_classifier.fit(iris_data, iris_target)
# randomf forest
rf_classifier = ensemble.RandomForestClassifier()
rf_classifier.fit(iris_data, iris_target)

# Export the classifier to a file
svm_model_filename = 'svm-model.joblib'
joblib.dump(svm_classifier, svm_model_filename)
rf_model_filename = 'rf-model.joblib'
joblib.dump(rf_classifier, rf_model_filename)

# [START upload-model]
# Upload the saved model file to Cloud Storage
# TODO: using date as RUN ID
current_date = datetime.datetime.now().strftime('iris_%Y%m%d_%H%M%S')
gcs_svm_model_path = os.path.join('gs://', BUCKET_NAME, current_date, svm_model_filename)
subprocess.check_call(
    ['gsutil', 'cp', svm_model_filename, gcs_svm_model_path],
     stderr=sys.stdout,
)
gcs_rf_model_path = os.path.join('gs://', BUCKET_NAME, current_date, rf_model_filename)
subprocess.check_call(
    ['gsutil', 'cp', rf_model_filename, gcs_rf_model_path],
    stderr=sys.stdout,
)

# [END upload-model]
