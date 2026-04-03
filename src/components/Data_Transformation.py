import sys
import os
import pandas as pd
import numpy as np
import pickle

from src.Exception import customException
from src.Logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,RobustScaler
from src.Utils import save_object

from dataclasses import dataclass

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationconfig()
class DataTransformation:
    def get_data_transformation(self):
        try:
            numerical=['PRECIPITATION','MAX_TEMP','MIN_TEMP','AVG_WIND_SPEED','WIND_TEMP_RATIO','LAGGED_PRECIPITATION','LAGGED_AVG_WIND_SPEED','TEMP_RANGE'],
            categorical=['SEASON']
            logging.info(f'Numerical Columns:{numerical}')
            logging.info(f'Categorical Columns:{categorical}')

            num_pipeline=Pipeline(steps=[
                ('scaler',RobustScaler())
            ])
            cat_pipeline=Pipeline(steps=[
                ('encoder',OneHotEncoder(drop='first',sparse_output=False))
            ])

            Preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical),
                    ('cat_pipeline',cat_pipeline,categorical)
                ]
            )

            logging.info('Preprocessor is created')

            return Preprocessor
        except Exception as e:
            raise customException(e)
        
    def make_lstm_sequence(X:np.ndarray, y:np.ndarray, lookback:int, horizon:int=1, stride:int=1):
        if lookback <=0: raise ValueError("Lookback must not be 0")
        if horizon<=0: raise ValueError("Horizon must not be 0")
        if stride<0: raise ValueError("Stride must not be 0")

        X=np.asarray(X)
        y=np.asarray(y).reshape(-1)

        if X.ndim!=2: raise ValueError(f'X must be 2D got dimension{X.shape}')
        if y.ndim!=2: raise ValueError(f'y must be 2D got dimension{y.shape}')
        if len(X)!=len(y): raise ValueError(f'X and y must be same length')

        n=len(X)
        end=n-horizon
        window=[]
        target=[]

        for t in range(lookback,end+1,stride):
            x_window=X[t-lookback:t,:]
            y_target=y[t+horizon-1]
            window.append(x_window)
            target.append(y_target)
        if not window:
            raise ValueError('Not enough dimensions to create squences')
        return np.stack(window,axis=0),np.asarray(target).astype('int64')
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Train and Test data are acquired')

            target_column=['FIRE_OCCURED']

            #seperating train and test data
            input_train_df=train_df.drop(columns=[target_column],axis=1)
            output_train_df=test_df[target_column]

            input_test_df=test_df.drop(columns=[target_column],axis=1)
            output_test_df=test_df[target_column]

            preprocessor_obj=self.get_data_transformation()
            logging.info('Applying preprocessor fumction to train and test data')
            input_train_arr=preprocessor_obj.fit_transform(input_train_df)
            input_test_arr=preprocessor_obj.transform(input_test_df)

            logging.info('Create LSTM sequences using make_lstm_sequence function')
            lookback_days=7
            X_train,y_train=self.make_lstm_sequence(X=input_train_arr,y=output_train_df.values,lookback=lookback_days)
            X_test,y_test=self.make_lstm_sequence(X=input_test_arr,y=output_test_df.value,lookback=lookback_days)

            logging.info(f"Final Train Shape: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Final Test Shape: X={X_test.shape}, y={y_test.shape}")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return X_train,y_train,X_test,y_test,self.transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise customException(e)


        


