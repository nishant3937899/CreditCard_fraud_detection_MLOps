from MLOps_project import logger
from MLOps_project.pipeline.stage_01_data_ingestion import run_data_ingestion
from MLOps_project.pipeline.stage_02_data_validation import run_data_validation
from MLOps_project.pipeline.stage_03_feat_engine_transform import featureEngineering_tranformation




stage_name='DATA INGESTION'
try:
    logger.info(f'stage : {stage_name} has successfully started')
    run_data_ingestion()
    logger.info(f'stage : {stage_name} has successfully completed')
except Exception as e:
    raise e




stage_name= 'DATA VALIDATION'
try:
    logger.info(f'stage : {stage_name} has successfully started')
    run_data_validation()
    logger.info(f'stage : {stage_name} has successfully completed')
except Exception as e:
    raise e




stage_name= 'FEATURE ENGINEERING AND TRANSFORMATION'
try:
    logger.info(f'stage : {stage_name} has successfully started')
    featureEngineering_tranformation()
    logger.info(f'stage : {stage_name} has successfully completed')
except Exception as e:
        raise e