"""Movie training pipeline"""
import argparse

import pandas as pd

from trainer.features import download_data
from trainer.pipeline import TrainingPipeline


def main(args):
    # TODO: fix because this is too brittle
    download_data(
        inputs=[args.feature_file],
        outputs=['inputs/movies_metadata.csv'],
    )

    # Load data into pandas, then use `.values` to get NumPy arrays
    X = pd.read_csv(args.feature_file).values
    model = TrainingPipeline(
        name='movies',
        training_artifact=args.training_artifact,
        format='joblib',
    )
    model.fit(X)
    model.save()
    results = model.transform(X)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParse()
    parser.add_argument(
        '--feature-file',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--training-artifact',
        type=str,
        required=True,
        default='training_pipeline.joblib',
    )
    args = parser.parse_args()
    main(args)

