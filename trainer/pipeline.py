import datetime
import os
import sys
import subprocess
import pickle
from typing import Any, Optional

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from .utils import impute_string, get_logger

LOGGER = get_logger(__name__)

# TODO: setup config file (dynaconf, omegaconf)
BUCKET_NAME = "gs://model_registry"


# TODO: add logger
class Model():
    """Base class to add funcitonality to a
    training pipeline.
    """
    def __init__(
        self,
        name: str,
        model_path: str,
    ):
        # model attributes
        self._name = name
        self._model_path = model_path
        # training artifacts
        self._model = None
        self._run_id = datetime.datetime.now().strftime(
            f'{self._name}_%Y%m%d_%H%M%S')
        self._gcs_model_path = os.path.join(
            BUCKET_NAME,
            self._run_id,
            self._model_path,
        )

    def save(self) -> None:
        """Serialize training artifact."""
        with open(self._model_path, 'wb') as f:
            pickle.dump(self._model, f)

    def load(self) -> None:
        """Load training artifact into memory."""
        with open(self._model_path, 'rb') as f:
            self._model = pickle.load(f)

    def load_from_gcs(self):
        """Download training artifact from GCS bucket."""
        pass

    def push_to_gcs(self) -> None:
        """Push training artifact to GCS bucket.

        Args:
            task_name (str): TBD
        """
        cmd = [
            'gsutil',
            'cp',
            self._model_path,
            self._gcs_model_path,
        ]
        subprocess.check_call(
            cmd,
            stderr=sys.stdout,
        )


class TrainingPipeline(Model):
    """Training pipeline."""
    def __init__(self, name: str, model_path: str):
        super().__init__(name, model_path)
        self._model = self._build_pipeline()
        # self._config = pass training config

    def _build_pipeline(self) -> Any:
        """Build sklearn pipeline with preprocessing and training"""
        # impute missing values and transform
        # TODO: preprocessing steps, names for each step, should all come from
        # training config - self._config
        text_features = ["overview"]
        text_transformer = Pipeline(steps=[
            ('imputer', FunctionTransformer(impute_string)),
            ('tfidf', TfidfVectorizer(stop_words='english')),
        ])
        preprocessor = make_column_transformer(
            (text_transformer, text_features))
        # Append classifier to preprocessing pipeline.
        # Now we have a full prediction pipeline.
        # self._model = Pipeline(steps=[
        #        ('preprocessor', preprocessor),
        #        ('classifier', LogisticRegression())
        #])
        LOGGER.info('Loaded training pipeline!')
        return preprocessor

    def train(self, X: pd.DataFrame, y: pd.Series):
        LOGGER.info('Starting training pipeline')
        self._model.fit(X, y)
        assert self._model is not None
        LOGGER.info('Training complete!')

    def fit_transform(self, X: pd.DataFrame):
        LOGGER.info('Starting transform')
        results = self._model.fit_transform(X)
        LOGGER.info('Transform complete!')
        LOGGER.info(f'Size: {results.shape}')
        return results

    def predict(self, X: pd.DataFrame):
        print('Starting inference')
        predictions = self._model.predict(X)
        assert predictions is not None
        return predictions
