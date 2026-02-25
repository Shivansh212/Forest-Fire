import sys
import os
import pandas as pd
import numpy as np

from src.Exception import customException
from src.Logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,RobustScaler

from dataclasses import dataclass

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path:str = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationconfig()

    def get_data_transformation(self):
        try:
            numerical_columns=[]
            
        except:
            pass
