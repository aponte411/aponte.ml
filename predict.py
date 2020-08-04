"""Examples of using AI Platform's online prediction service."""
import argparse
import json

import googleapiclient.discovery
import six


def predict_json(project, model, instances, version=None):
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
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def predict_examples(project,
                     model,
                     example_bytes_list,
                     version=None):
    """Send protocol buffer data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        example_bytes_list ([str]): A list of bytestrings representing
            serialized tf.train.Example protocol buffers. The contents of this
            protocol buffer will change depending on the signature of your
            deployed model.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': [
            {'b64': base64.b64encode(example_bytes).decode('utf-8')}
            for example_bytes in example_bytes_list
        ]}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']



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
        '--version',
        help='Name of the version.',
        type=str
    )
    args = parser.parse_args()
    main(
        args.project,
        args.model,
        version=args.version,
    )
