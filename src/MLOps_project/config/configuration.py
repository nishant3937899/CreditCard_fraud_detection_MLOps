from pathlib import Path
from MLOps_project.utils.common import read_yaml,createDIr
from MLOps_project import logger
class ConfigManager:
    def __init__(self):
        self.config = read_yaml(Path("config/config.yaml"))
        self.params = read_yaml(Path("params.yaml"))
        self.schema = read_yaml(Path("schema.yaml"))

    def get_data_ingestion_config(self):
        config = self.config.data_ingestion

        createDIr([config.rootdir_zip_down])
        logger.info(f'{config.rootdir_zip_down} has successfully created')

        return config
    
    from pathlib import Path

class config_mgr_val:
    def __init__(self):
        self.config= read_yaml(Path('config/config.yaml'))
        self.param= read_yaml(Path('params.yaml'))
        self.shema= read_yaml(Path('schema.yaml'))
    
    def create_validation_dir(self):
        config=self.config.data_validation
        schema=self.shema
        
        createDIr([config.root_dir])
        logger.info(f'the root directory for data validation has been successfully created')
        
        return config,schema
    

class config_magr_tran:
    def __init__(self):
        self.config= read_yaml(Path('config/config.yaml'))
        self.param= read_yaml(Path('params.yaml'))
        self.shema= read_yaml(Path('schema.yaml'))
    
    def create_tranformation_dir(self):
        config=self.config.data_transformation
        createDIr([config.root_dir_trans])
        logger.info('trasformation dir created')
        return config