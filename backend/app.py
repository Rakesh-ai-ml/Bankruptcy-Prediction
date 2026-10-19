from fastapi import FastAPI                                                                                                                                          
import pandas as pd      




from src.BancrupcyClassification.entity.config_entity import CompanyFinancials                                                                                          
from src.BancrupcyClassification.pipeline.prediction_pipeline import PredictionPipeline                                                                                        
                                                                                                                                                                               
app = FastAPI(title="Bankruptcy Prediction API")                                                                                                                                 
pipeline = PredictionPipeline()                                                                                                                                                  

@app.get('/')
def index():
    return {'message': 'serving Bankruptcy prediction'}


@app.post("/predict")                                                                                                                                                            
def predict(finalcials:CompanyFinancials ): 
                                                                                                                                   
    df = pd.DataFrame([finalcials.model_dump()])  
    pipeline=PredictionPipeline()
    result=pipeline.initiate_prediction(df) 

    return  result   


