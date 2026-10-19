import pandas as pd
import joblib
import json
from src.BancrupcyClassification.entity.config_entity import PredictionConfig
from src.BancrupcyClassification.logger import logger


class ModelPrediction:

    def __init__(self, config : PredictionConfig):

        self.config = config
        self.model=joblib.load(self.config.model_path)
        self.pipeline=joblib.load(self.config.pipeline_path)

        with open(self.config.threshold_path, "r") as f:
            self.threshold= float(json.load(f).get("threshold"))
        logger.info(f"loadeed model, preprocessor and threshold")
    
    def predict(self, df: pd.DataFrame):

        df=self.pipeline.transform(df)

        proba=float(self.model.predict_proba(df)[:,1][0])
        
        is_bankrupt = int(proba>=self.threshold)
        
        return {
            
            "bankruptcy_probability": round(proba, 4),        
            "prediction": "failed" if is_bankrupt else "alive",
            "is_bankrupt": is_bankrupt  
        }
