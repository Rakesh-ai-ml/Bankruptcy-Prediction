from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

class DataIngestionConfig(BaseModel):
    root_dir: Path
    source_PATH: str
    local_data_file: Path
    unzip_dir: Path

class DataValidationConfig(BaseModel):
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: List[str]
    target_col : str

class DataTransformationConfig(BaseModel):
    root_dir: Path
    data_path: Path
    features : List[str]
    target_column: str

class ModelTrainerConfig(BaseModel):
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str

class ModelEvaluationConfig(BaseModel):
    root_dir: Path
    test_data_path: Path 
    model_path: Path
    metric_file_name: Path
    target_column: str
    

class PredictionConfig(BaseModel):
    model_path : Path
    pipeline_path : Path
    threshold_path : Path

                                                                                                                                                                       
class CompanyFinancials(BaseModel):                                                                                                                                               
    Currentassets: float = Field(..., example=511.26)                                                                                                                            
    Costofgoodssold: float = Field(..., example=833.10)                                                                                                                          
    Depreciationandamortization: float = Field(..., example=25.40)                                                                                                               
    EBITDA: float = Field(..., example=112.50)                                                                                                                                   
    Inventory: float = Field(..., example=85.20)                                                                                                                                 
    NetIncome: float = Field(..., example=35.16)                                                                                                                                 
    TotalReceivables: float = Field(..., example=120.40)                                                                                                                         
    Marketvalue: float = Field(..., example=450.00)                                                                                                                              
    Netsales: float = Field(..., example=950.00)                                                                                                                                 
    Totalassets: float = Field(..., example=740.99)  # X10                                                                                                                       
    TotalLongtermdebt: float = Field(..., example=150.00)                                                                                                                        
    EBIT: float = Field(..., example=85.00)                                                                                                                                      
    GrossProfit: float = Field(..., example=116.90)                                                                                                                              
    TotalCurrentLiabilities: float = Field(..., example=220.00)                                                                                                                  
    RetainedEarnings: float = Field(..., example=310.00)                                                                                                                         
    TotalRevenue: float = Field(..., example=950.00)                                                                                                                             
    TotalLiabilities: float = Field(..., example=370.00)                                                                                                                         
    TotalOperatingExpenses: float = Field(..., example=865.00)                                                                                                                   

    