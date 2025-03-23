import os
import sys

from src.exception import CustomException

import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)   
            score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = score

        return report        
    except Exception as e:
        raise CustomException(e, sys)