'''
This script aims to create the predict pipeline for a simple web application which will be interacting with the pkl files, such that we can make predictions by giving values of input features. 
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

# Data manipulation.
import pandas as pd

# File handling.
import os


class PredictPipeline:
    '''
    Class for making predictions using a trained model and preprocessor.

    This class provides a pipeline for making predictions on new instances using a trained machine learning model and
    a preprocessor. It loads the model and preprocessor from files, preprocesses the input features, and applies a
    predefined threshold for making predictions.

    Methods:
        predict(features):
            Make predictions on new instances using the loaded model and preprocessor.

    Example:
        pipeline = PredictPipeline()
        new_features = [...]
        prediction = pipeline.predict(new_features)

    Note:
        This class assumes the availability of the load_object function and defines the THRESHOLD value.
    '''
    def __init__(self) -> None:
        '''
        Initializes a PredictPipeline instance.

        Initializes the instance. No specific setup is required in the constructor.
        '''
        pass


    def predict(self, features):
        '''
        Make predictions on new instances using the loaded model and preprocessor.

        Args:
            features: Input features for which predictions will be made.

        Returns:
            predictions: Predicted labels for the input features.

        Raises:
            CustomException: If an exception occurs during the prediction process.
        
        Note: The predictions are made by comparing the model's estimated probabilities of the customer being bad risk with a threshold value. If the probability is greater than it, the instance is classified as positive, else negative. The threshold was chosen during modelling notebook, where I saw that its value provides a recall score of 0.8 (the metric of interest).
        '''
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info('Load model and preprocessor objects.')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info('Model and preprocessor succesfully loaded.')

            logging.info('Mapping sex variable.')

            # Sex variable mapping needed.
            features['Sex'] = features['Sex'].map({'male': 1, 'female': 0})

            logging.info('Preprocessing the input data.')

            prepared_data = preprocessor.transform(features)

            logging.info('Input data prepared for prediction.')

            logging.info('Predicting.')

            # Predict using the threshold that provided a recall score of 0.8 in the modelling notebook.
            THRESHOLD = 0.42902538161651166
            predicted_probas = model.predict_proba(prepared_data)
            prediction = (predicted_probas[:, 1] >= THRESHOLD).astype(int)

            if prediction[0] == 1:
                prediction = 'RUIM'
            else:
                prediction = 'BOM'

            logging.info('Prediction successfully made.')

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        

class InputData:
    def __init__(self,
                 age: int,
                 sex: str,
                 job: int,
                 housing: str,
                 saving_accounts: str,
                 checking_account: str,
                 credit_amount: int,
                 duration: int,
                 purpose: str):
    
        # Map variables from html inputs.
        self.age = age
        self.sex = sex
        self.job = job
        self.housing = housing
        self.saving_accounts = saving_accounts
        self.checking_account = checking_account
        self.credit_amount = credit_amount
        self.duration = duration
        self.purpose = purpose

    
    def get_input_data_df(self):
        try:
            input_data_dict = dict()

            # Map the variables to the form of a dataframe for being used in predictions.
            
            input_data_dict['Age'] = [self.age]
            input_data_dict['Sex'] = [self.sex]
            input_data_dict['Job'] = [self.job]
            input_data_dict['Housing'] = [self.housing]
            input_data_dict['Saving accounts'] = [self.saving_accounts]
            input_data_dict['Checking account'] = [self.checking_account]
            input_data_dict['Credit amount'] = [self.credit_amount]
            input_data_dict['Duration'] = [self.duration]
            input_data_dict['Purpose'] = [self.purpose]

            input_data_df = pd.DataFrame(input_data_dict)

            return input_data_df
        
        except Exception as e:
            raise CustomException(e, sys)