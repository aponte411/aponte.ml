"""Movie training pipeline"""
import argparse

import pandas as pd

from trainer.features import download_data
from trainer.pipeline import TrainingPipeline


def main(args):
    # TODO: fix because this is too brittle
    # download_data(
    #     inputs=[args.feature_file],
    #     outputs=['inputs/movies_metadata.csv'],
    # )

    # Load data into pandas, then use `.values` to get NumPy arrays
    X = pd.read_csv(args.feature_file, low_memory=True)
    model = TrainingPipeline(
        name='movies',
        model_path=args.model_path,
    )
    #model.fit(X_train, y_train)
    #predictions=model.predict(X_test)
    results = model.fit_transform(X)
    model.save()
    # TODO: later will be model.predict()
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feature-file',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=False,
        default='outputs/training_pipeline.pkl',
    )

    args = parser.parse_args()
    main(args)
