import subprocess
from typing import List
import sys


def download_data(
    inputs: List[str],
    outputs: List[str],
) -> None:
    """Given a list of input paths download
    the data into designated output files.

    Args:
        inputs (List[str]): google store path
        output (List[str]): local file name
    """
    for input_path, output_file in zip(inputs, outputs):
        subprocess.check_call(
            ['gsutil', 'cp', input_path, output_file],
            stderr=sys.stdout,
        )
