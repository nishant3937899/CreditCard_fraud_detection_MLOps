from pathlib import Path
from MLOps_project.utils.common import read_yaml,createDIr
from MLOps_project import logger
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

class FeatureEng:
    def __init__(self,config):
        self.config=config

    def featureEngineering(self):
        data=self.config.data_dir

        df=pd.read_csv(data)

        df['Churn']=df['Churn'].map({'Yes':0,'No':1})
        df['tenure_grp']=pd.cut(df['tenure'],
                        bins=[0,12,24,36,48,60,72],
                        labels=['0-1yrs','1-2yrs','2-3yrs','3-4yrs','4-5yrs','5-6yrs'])
        df['monthly_charge']=pd.cut(df['MonthlyCharges'],bins=3,labels=['low','medium','high'])
        df.drop('customerID',axis=1,inplace=True)
        services = ['PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        df['Total_services']=df[services].apply(lambda x:(x=='Yes').sum() ,axis=1) 
        logger.info(f'feature engineering done')
        df.to_csv(self.config.data_dir, index=False) 
        logger.info('data is now saved in csv')
        config=self.config
        return config  
    


class DataTransformaiton:
    def __init__(self,config):

        self.config= config

    def train_test(self):
        df=pd.read_csv(self.config.data_dir)
        train, test = train_test_split(df,test_size=0.2)
        train.to_csv(os.path.join(self.config.train_data_dir),index=False)
        test.to_csv(os.path.join(self.config.test_data_dir),index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        
