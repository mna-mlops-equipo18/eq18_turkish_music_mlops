import logging
import pandas as pd
from eq18_turkish_music_mlops.preprocessing.splitter import DatasetSplitter
from eq18_turkish_music_mlops.preprocessing.preprocesor import DataPreprocessor

def main():
    try:

        df = pd.read_csv("turkish_music_emotion_modified.csv")

        splitter = DatasetSplitter()
        preprocesor = DataPreprocessor()

        X_train, y_train, X_val, y_val, X_test, y_test = splitter.split(df, y_col="")


        X_train = preprocesor._fit_transform_tabular(X_train)
        y_train = preprocesor._fit_transform_target(y_train)


        # Después va entrenamiento y evaluación con MLFlow
    except Exception as e:
            logging.error(f"[Traing] Error in main")
            raise
    

if __name__ == "__main__":
     main()