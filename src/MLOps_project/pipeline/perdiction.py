from MLOps_project.components.model_trainer import model_trainer
from MLOps_project import logger
import pandas as pd
import joblib


def prediction_feature_eng(df):
    df['tenure_grp']=pd.cut(df['tenure'],
                        bins=[0,12,24,36,48,60,72],
                        labels=['0-1yrs','1-2yrs','2-3yrs','3-4yrs','4-5yrs','5-6yrs'])
    df['monthly_charge']=pd.cut(df['MonthlyCharges'],bins=3,labels=['low','medium','high'])
    
    services = ['PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
        
    df['Total_services']=df[services].apply(lambda x:(x=='Yes').sum() ,axis=1) 
    logger.info(f'for prediction feature engineering done')
    return df



def predictor(
    gender: str,
    SeniorCitizen: int,
    Partner: str,
    Dependents: str,
    tenure: int,
    PhoneService: str,
    MultipleLines: str,
    InternetService: str,
    OnlineSecurity: str,
    OnlineBackup: str,
    DeviceProtection: str,
    TechSupport: str,
    StreamingTV: str,
    StreamingMovies: str,
    Contract: str,
    PaperlessBilling: str,
    PaymentMethod: str,
    MonthlyCharges: float,
    TotalCharges: str,
    
):

    df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        
    }])
    new_df=prediction_feature_eng(df)

    trans=model_trainer()
    tranform_df=trans.scaling_encoding(new_df,0)

    with open('artifacts/model_trainer/modle.joblib','rb') as f:
        model= joblib.load(f)

    prediction = model.predict(tranform_df)
    print(prediction)
    return prediction
    


'''predictor('Male',                     #for testing purpose
          1,
          'Yes',
          'No',
          13,
          'Yes',
          'No',
          'DSL',
          'Yes',
          'Yes',
          'No',
          'No',
          'No',
          'No',
          'One year',
          'No',
          'Mailed Check',
          100.34,
          4313.33             
        )'''
 