import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class DatasetSplitter:

    def __init__(self, test_size:float=0.3, val_size:float=0.5, random_state:int=27, stratify:bool=True):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.stratify = stratify


    def split(self, df:pd.DataFrame, y_col:str):
        try:
            y = df[y_col].copy()
            X = df.drop(colums=[y_col]).copy()

            X_train, X_temp, y_train, y_temp = train_test_split(X, 
                                                                y, 
                                                                test_size=self.test_size, 
                                                                random_state=self.random_state,
                                                                stratify=y if self.stratify else None)
            
            X_val, X_test, y_val, y_test = train_test_split(X_temp, 
                                                            y_temp, 
                                                            test_size=self.val_size, 
                                                            random_state=self.random_state,
                                                            stratify=y_temp if self.stratify else None)
            
            return X_train, y_train, X_val, y_val, X_test, y_test
        except Exception as e:
            logging.error(f"[Splitter] Error spliting dataset")
            raise