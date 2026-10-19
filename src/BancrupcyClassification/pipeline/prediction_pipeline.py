import pandas as pd
from src.BancrupcyClassification.config.configuration import ConfigurationManager
from src.BancrupcyClassification.components.prediction import ModelPrediction





class PredictionPipeline:

    def __init__(self):
        pass
    
    def initiate_prediction(self, df: pd.DataFrame):

        config= ConfigurationManager()
        pred_config = config.get_prediction_config()
        pred = ModelPrediction(config=pred_config)
        result = pred.predict(df)
        return result
