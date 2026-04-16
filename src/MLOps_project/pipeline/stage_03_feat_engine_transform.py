from MLOps_project.config.configuration import  config_magr_tran
from MLOps_project.components.featureEngineering_transformaiton import FeatureEng,DataTransformaiton
from MLOps_project import logger

def featureEngineering_tranformation():
    try:
        data_trans=config_magr_tran()
        config_tran=data_trans.create_tranformation_dir()
        feat_eng=FeatureEng(config_tran)
        feat_done=feat_eng.featureEngineering()
        tr_te_split=DataTransformaiton(feat_done)
        tr_te_split.train_test()
    except Exception as e:
        raise e
    

if __name__ == '__main__':
    stage_name= 'FEATURE ENGINEERING AND TRANSFORMATION'
    try:
        logger.info(f'stage : {stage_name} has successfully started')
        featureEngineering_tranformation()
        logger.info(f'stage : {stage_name} has successfully completed')
    except Exception as e:
        raise e