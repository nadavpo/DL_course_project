import hocon
from pathlib import Path


def get_params(params_file_path):
    params_file = Path(params_file_path)
    with params_file.open('r') as f:
        if params_file.suffix == '.conf':
            params = hocon.load(f)
        else:
            raise Exception('Unknown params file format')
    return params