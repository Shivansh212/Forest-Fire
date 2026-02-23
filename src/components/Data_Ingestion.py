import os
import sys
from src.Exception import customException
from src.Logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str= os.path.join('artifacts','data.csv')
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str= os.path.join('artifacts','test.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            df=pd.read_csv(r'Notebook\data\CA_Weather_Fire_Dataset_1984-2025.csv')
            logging.info('Read the dataset')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            
            
            df.rename(columns={'FIRE_START_DAY':'FIRE_OCCURED'},inplace=True)

            # Dropping columns
            df.drop(columns={'DATE','DAY_OF_YEAR','TEMP_RANGE'},inplace=True)

            # Dropping the null values(As per THE EDA)
            df.dropna(inplace=True)

            # Converting the MAX and MIN temp from Farenheit to Celsius
            df['MAX_TEMP']=(df['MAX_TEMP']-32)*5/9
            df['MIN_TEMP']=(df['MIN_TEMP']-32)*5/9
            df['TEMP_RANGE']=(df['MAX_TEMP']-df['MIN_TEMP'])

            # Mapping the Target variable False(Fire not occured as=0) and True(Fire occured as 1)
            df.loc[df['FIRE_OCCURED']==False,'FIRE_OCCURED'] = 0
            df.loc[df['FIRE_OCCURED']==True,'FIRE_OCCURED'] = 1
            df['FIRE_OCCURED']=pd.to_numeric(df['FIRE_OCCURED'])


            logging.info("Train_Test split started")
            
            train_set, test_set = train_test_split(df,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.error(f"Error in Data Ingestion: {e}")
            raise customException(e)
        
# if __name__=="__main__":
#     obj=DataIngestion()
#     obj.initiate_data_ingestion()
