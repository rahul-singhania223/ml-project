import os
import sys

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        '''
            This function is responsible for data transformation
        '''
        try:
            numerical_features = ['Reading_Score', 'Writing_Score']
            categorical_features = ['Gender', 'Ethnicity', 'Lunch', 'Course', 'Parents_Education']

            numerical_pipeline = Pipeline (
                steps=[
                    ("imputation", SimpleImputer(strategy="median")),
                    ("standard_scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline (
                steps=[
                    ("imputation", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )

            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"numerical columns: {numerical_features}")

            preprocessor = ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Reading train and test completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_obj()

            target_column = 'Math_Score'
            numerical_columns = ['Reading_Score', 'Writing_Score']

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing object on Training dataset and Testing dataset")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)