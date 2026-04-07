import sys
from src.Logger import logging
from src.Exception import customException


from src.components.Data_Ingestion import DataIngestion
from src.components.Data_Transformation import DataTransformation
from src.components.Model_Trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("======================================================")
            logging.info(" STARTING END-TO-END MODEL TRAINING PIPELINE ")
            logging.info("======================================================")

            
            # 1. DATA INGESTION
            
            logging.info(" Initiating Data Ingestion")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info(f" Data Ingestion Success! Train path: {train_data_path}")

           
            # 2. DATA TRANSFORMATION
            
            logging.info(" Initiating Data Transformation")
            data_transformation = DataTransformation()
            X_train, y_train, X_test, y_test,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            logging.info(f"Data Transformation Success! 3D Train Shape: {X_train.shape}")

            
            # 3. MODEL TRAINING
            
            logging.info(" Initiating Model Training")
            model_trainer = ModelTrainer()
            model_file_path = model_trainer.initiate_model_training(X_train, y_train, X_test, y_test)
            logging.info(f" Model Training Success! Brain saved at: {model_file_path}")

            logging.info("======================================================")
            logging.info(" PIPELINE COMPLETED SUCCESSFULLY! ")
            logging.info("======================================================")

        except Exception as e:
            logging.error(" PIPELINE FAILED! ")
            raise customException(e)

if __name__ == "__main__":
    
    pipeline = TrainPipeline()
    pipeline.run_pipeline()