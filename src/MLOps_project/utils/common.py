import yaml
import os
from MLOps_project import logger
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path

@ensure_annotations
def read_yaml(path : Path ) -> ConfigBox:         # this reads yaml file and returns its content in Configbox
    try:
        with open(path ) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f'yaml file{yaml_file} loaded successfully')
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError('yaml fle is empty')
    except Exception as e:
        raise e
    

@ensure_annotations
def createDIr(path_dir:list,verbose=True):                  # this create the directory of on the given path
    for path in path_dir:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logger.info(f'created directory at {path}')

