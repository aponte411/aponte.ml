"""AI Platform's online prediction service."""
import argparse
import json
from typing import Any, Optional, List, Dict

import googleapiclient.discovery


def predict(
    project: str,
    model: str,
    instances: List[str],
    version: Optional[str] = None) -> Dict[str, Any]:
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the AI Platform service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = f'projects/{project}/models/{model}'
    if version is not None:
        name += f'/versions/{version}'

    response = service.projects().predict(
        name=name,
        body={'instances': instances},
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['predictions']


def main(
    project: str,
    model: str,
    samples: str,
    version: Optional[str] = None,
) -> Any:
    instances = []
    with open(samples) as f:
        for line in f:
            instances.append(json.loads(line))
    try:
        result = predict(
            project,
            model,
            instances,
            version,
        )
        print(result)
        return result
    except RuntimeError as e:
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project',
        help='Project in which the model is deployed',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model',
        help='Model name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--samples',
        help='Path to txt file containing samples',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--version',
        help='Name of the version.',
        type=str
    )
    args = parser.parse_args()
    main(
        project=args.project,
        model=args.model,
        samples=args.samples,
        version=args.version,
    )
