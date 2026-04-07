import os
import sys
import numpy as np
import tensorflow as tf
from src.Exception import customException
from sklearn.metrics import accuracy_score, classification_report
from src.Logger import logging
from dataclasses import dataclass

class ModelTrainerConfig:
    Trained_model=os.path.join('artifacts','lstm_model.h5')

class ModelTrainer:
    def __init__(self):
        self.train_config=ModelTrainerConfig()

    def initiate_model_training(self,X_train:np.ndarray,y_train:np.ndarray,X_test:np.ndarray,y_test:np.ndarray):
        try:
            logging.info('Beginning the model training')
            # To make sure that the target in both train and test are strict integers
            y_train=np.asarray(y_train).astype('int64')
            y_test=np.asarray(y_test).astype('int64')

            n_features=int(X_train.shape[-1])
            time_step=int(X_train.shape[1])

            # Model Architecture
            model=tf.keras.Sequential([
                tf.keras.layers.Input(shape=(time_step,n_features)),
                tf.keras.layers.LSTM(64,return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32,activation='relu'),
                tf.keras.layers.Dense(1,activation='sigmoid')
            ])
            # Defining compile rules and weights
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='binary_crossentropy',
                metrics=[
                    tf.keras.metrics.AUC(name='auc'),'accuracy'
                ]
            )
            # Handling the imbalanced dataset
            pos=int(np.sum(y_train==1))
            neg=int(np.sum(y_train==0))
            class_weight_dict={0:1.0 ,1:float(neg)/float(pos)} if pos>0 else None

            # Earlystop
            callback=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc',
                    mode='max',
                    patience=9,
                    restore_best_weights=True,
                )
            ]
            # Fitting the model
            logging.info('Model ready to fit')
            history=model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=32,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=callback,
                verbose=1
            )
            
            
            logging.info('Training done, Now running Sanity check')
            final_train_auc=history.history['auc'][-1]
            fina_val_auc=history.history['val_auc'][-1]
            print(f'\n --- Sanity Check---')
            print(f'The final tain auc is:{final_train_auc:.2f}')
            print(f'The final val auc is :{fina_val_auc:.2f}')

            print('\n==============================================')
            user_decision=input('Sanity check is copleted. Press enter to predict test data. Or write "stop" to abort->')
            if user_decision.lower()=='stop':
                logging.info('User aborted the execution')
                sys.exit('Pipeline aborted by user')
            logging.info('User decided to predict test data')
            print('\n===============================================')

            # Predicting on test data
            logging.info('Prediction on test data begun')
            y_pred_prob=model.predict(X_test)
            y_pred=(y_pred_prob>=0.3).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Final Test Accuracy: {accuracy * 100:.2f}%")
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # SAVE MODEL
            os.makedirs(os.path.dirname(self.train_config.Trained_model),exist_ok=True)
            model.save(self.train_config.Trained_model)
            logging.info(f"Saved model successfully to {self.train_config.Trained_model}")
            return self.train_config.Trained_model
        except Exception as e:
            raise customException(e)