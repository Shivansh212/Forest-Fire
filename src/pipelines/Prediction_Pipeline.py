import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from src.Exception import customException
from src.Logger import logging
from src.Utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features_df: pd.DataFrame):
        try:
            # Define Paths
            model_path = os.path.join("artifacts", "lstm_model.h5")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading LSTM brain and Preprocessor...")
            
            #  Load the objects 
            model = load_model(model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Scaling incoming 7-day weather data...")
            #  Transform the raw data
            data_scaled = preprocessor.transform(features_df)

            logging.info("Reshaping data for LSTM 3D requirements...")
            #  THE 3D RESHAPE: Converts 2D (7, features) table to 3D (1, 7, features) block
            data_3d = np.expand_dims(data_scaled, axis=0) 

            logging.info("Making prediction...")
            #  Predict
            prediction_prob = model.predict(data_3d, verbose=0)
            
            #  Granular Risk Categorization
            raw_probability = float(prediction_prob[0][0])
            
            if raw_probability < 0.3:
                risk_level = "Low Risk"
            elif 0.3 <= raw_probability < 0.6:
                risk_level = "Medium Risk"
            else:
                risk_level = "High Risk"

            logging.info(f"Prediction complete. Prob: {raw_probability:.4f}, Risk: {risk_level}")
            
            return risk_level, raw_probability

        except Exception as e:
            raise customException(e, sys)


class CustomData:
    def __init__(self, **kwargs): 
        """
        Using **kwargs allows this class to dynamically accept ANY number of features
        without you having to hardcode them. It automatically packs your web form
        inputs into a dictionary.
        """
        self.feature_dict = kwargs

        # Safety Check: Loop through every feature to guarantee it is exactly a 7-day movie
        for feature_name, feature_sequence in self.feature_dict.items():
            if len(feature_sequence) != 7:
                raise ValueError(f"LSTM requires exactly 7 days of data. '{feature_name}' only received {len(feature_sequence)} days.")

    def get_data_as_data_frame(self):
        '''
        Converts the dynamic dictionary of 7-day lists into a pandas DataFrame.
        This will perfectly match the columns from your train.csv!
        '''
        try:
            # Pandas is smart enough to turn a dictionary of lists directly into a DataFrame
            return pd.DataFrame(self.feature_dict)
            
        except Exception as e:
            raise customException(e, sys)