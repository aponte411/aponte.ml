import datetime
import os
import sys
import subprocess
import pickle
from typing import Any

import pandas as pd
from sklearn.external import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# TODO: setup config file (dynaconf, omegaconf)
BUCKET_NAME = "gs://model_registry"

# TODO: add logger
class Model():
    """Base class to add funcitonality to pipeline."""
    def __init__(
        self,
        name: str,
        training_artifact: str,
        format: str = 'joblib',
    ):
        self._name = name
        self._local_training_artifact = training_artifact
        self._format = format
        self._model = None
        self._run_id = datetime.datetime.now().strftime(f'{self._name}_%Y%m%d_%H%M%S')
        self._gcs_model_path = os.path.join(
            BUCKET_NAME,
            self._run_id,
            self._local_training_artifact,
        )


    def save(self) -> None:
        """Serialize training artifact."""
        if self.format == 'pkl':
            with open(self._local_training_artifact, 'wb') as f:
                pickle.dump(self._model, f)
        else:
            joblib.dump(self._model, self._local_training_artifact)

    def load(self) -> None:
        """Load training artifact into memory."""
        if self.format == 'pkl':
            with open(self._local_training_artifact, 'rb') as f:
                self._model = pickle.load(f)
        else:
            self._model = joblib.load(self._local_training_artifact)

    def load_from_gcs(self):
        """Download training artifact from GCS bucket."""
        pass

    def push_to_gcs(self) -> None:
        """Push training artifact to GCS bucket."""
        cmd = ['gsutil','cp', self._local_training_artifact, self._gcs_model_path,]
        subprocess.check_call(
            cmd, stderr=sys.stdout,
        )


# TODO: move helper functions elsewhere

def impute_string(X: pd.DataFrame) -> pd.DataFrame:
    """Impute dataframe with empty string."""
    return X.fillna("")

class TrainingPipeline(Model):
    """Training pipeline."""
    def __init__(self, name: str, training_artifact: str, format: str):
        super().__init__(name, training_artifact, format)
        self._model = self._build_pipeline()
        # self._config = pass training config

    def _build_pipeline(self) -> Any:
        """Build sklearn pipeline with preprocessing and training"""
        # impute missing values and transform
        # TODO: preprocessing steps, names for each step, should all come from
        # training config - self._config
        text_features = 'overview'
        text_transformer = Pipeline(steps=[
                ('imputer', FunctionTransformer(impute_string)),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[('text', text_transformer, text_features)]
        )
        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        # self._model = Pipeline(steps=[
        #        ('preprocessor', preprocessor),
        #        ('classifier', LogisticRegression())
        #])
        print('Loaded model!')
        return preprocessor

    def train(self, X: pd.DataFrame, y: pd.Series):
        print('Starting training pipeline')
        self._model.fit(X)
        assert self._model is not None
        print('Training complete!')

    def transform(self, X: pd.DataFrame):
        print('Starting transform')
        results = self._model.transform(X)
        assert results is not None
        return results

    def predict(self, X: pd.DataFrame):
        print('Starting inference')
        assert self._model is None
        predictions = self._model.predict(X)
        assert predictions is not None
        return predictions


