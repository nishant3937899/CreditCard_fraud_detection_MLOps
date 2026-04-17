import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from MLOps_project.utils.common import read_yaml,createDIr
from MLOps_project import logger
from pathlib import Path
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
import pickle
from sklearn.metrics import classification_report

class model_trainer:
    def __init__(self,):
        self.config= read_yaml(Path('config/config.yaml'))
        
    def scaling_encoding(self,X,code:int):# code = 1 for fit_transform and code = 0 for transform
        
        config= self.config.model_trainer

        num_feat=make_column_selector(dtype_exclude=['object','category'])
        cat_feat=make_column_selector(dtype_include=['object','category'])

        num_pipeline=Pipeline([
            ('imputer',SimpleImputer(strategy='median')),
            ('scaler',StandardScaler())])

        cat_pipeline=Pipeline([
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('encoder',OneHotEncoder(handle_unknown='ignore', drop='first'))])

        preprocessor=ColumnTransformer([
            ('num',num_pipeline,num_feat),
            ('cat',cat_pipeline,cat_feat)])
        
        if code == 1:
            X_done=preprocessor.fit_transform(X)
            with open(config.preprocess_loc,'wb') as f:
                pickle.dump(preprocessor,f)
            logger.info('preprocessor has been dumped')
            return X_done

        elif code == 0:
            with open(config.preprocess_loc,'rb') as f:
                preprocessor = pickle.load(f) 
                logger.info('preprocessor has been loaded')
            return preprocessor.transform(X)
        


class model_training:
    def __init__(self,config):
        self.config= config

    def model_train_start(self):

        config= self.config
        params=read_yaml(Path('params.yaml'))
        train = pd.read_csv(config.train_dir)
        test =  pd.read_csv(config.test_dir)

        logger.info('train test data has been loaded')
        X_train=train.drop('Churn',axis=1)
        Y_train=train['Churn']

        X_test=test.drop('Churn',axis=1)
        Y_test=test['Churn']
        logger.info('train test split created')
        scale=model_trainer()
        X_train_preprocessed=scale.scaling_encoding(X_train,1)
        X_test_preprocessed=scale.scaling_encoding(X_test,0)
        logger.info('data has been scaled')
        model = LogisticRegression(
            C=params.C,
            penalty=params.penalty,
            solver=params.solver,
            max_iter=params.max_iter,
            class_weight= {0:1.62, 1:1}
        )
        model.fit(X_train_preprocessed,Y_train)
        logger.info('model has been trained')
        joblib.dump(model,os.path.join(config.model_dir))

        ypred= model.predict(X_test_preprocessed)
        score= accuracy_score(Y_test,ypred)
        clsrep=classification_report(Y_test,ypred)
        print(score)
        print(clsrep)