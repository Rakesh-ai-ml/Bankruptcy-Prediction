import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from src.BancrupcyClassification.entity.config_entity import DataTransformationConfig
from src.BancrupcyClassification.logger import logger
import joblib


class TransformationPipeline( BaseEstimator, TransformerMixin ):
    def __init__(self,scaler, div_col : str = "Totalassets", eps : float = 1e-4):
        
        self.scaler = scaler
        self.div_col = div_col
        self.eps= eps
        self.financial_cols=None
        self.q1=None
        self.q99= None

    def fit(self, X, y=None):
       
        x_df=pd.DataFrame(X)
        exclude_cols= [self.div_col, "company_name", "year", "status_label"]
        self.financial_cols= [c for c in x_df.columns if c not in exclude_cols]
        total_assets=x_df[self.div_col]
        temp= x_df[self.financial_cols].div(total_assets + self.eps, axis=0)
        
        #log transform total_assets
        temp["log_total_assets"] = np.log1p(np.maximum(total_assets, 0))                                                                                           
        
        #Learn clipping thresholds from train data        
        self.q01_ = temp.quantile(0.01)                       
        self.q99_ = temp.quantile(0.99)                       
        clipped_temp = temp.clip(self.q1, self.q99, axis=1)                                                
        
        # standardize the data   
        self.scaler.fit(clipped_temp)                         
        return self                                           
                                                
    def transform(self, X):                                   
       
        X_df = pd.DataFrame(X)                                
        total_assets = X_df[self.div_col]                   
                                                              
        temp = X_df[self.financial_cols].div(total_assets +  self.eps, axis=0)                                               
       
        temp["log_total_assets"] = np.log1p(np.maximum(total_assets, 0))                                                                            
       
        #Apply learned training clipping                  
        clipped_temp = temp.clip(self.q1, self.q99, axis=1)                                                 
       
        # transform using scaler             
        scaled_array = self.scaler.transform(clipped_temp)                                                        
       
        return pd.DataFrame(scaled_array, columns=clipped_temp.columns, index=X_df.index) 


class DataTransformation:

    def __init__(self, config: DataTransformationConfig):
        
        self.config = config
        self.encoder = LabelEncoder()
        self.preprocessor = None

    def load_split_data(self):
        
        try:
            data = pd.read_csv(self.config.data_path)
            data = data.rename(columns={
                "X1": "Current assets", "X2": "Cost of goods sold", "X3": "Depreciation and amortization",
                "X4": "EBITDA", "X5": "Inventory", "X6": "Net Income", "X7": "Total Receivables",
                "X8": "Market value", "X9": "Net sales", "X10": "Total assets",
                "X11": "Total Longterm debt", "X12": "EBIT", "X13": "Gross Profit",
                "X14": "Total Current Liabilities", "X15": "Retained Earnings",
                "X16": "Total Revenue", "X17": "Total Liabilities", "X18": "Total Operating Expenses"
            })
            data.columns = data.columns.str.replace(" ", "")
            train = data[data['year'] < 2012].copy()
            test = data[data['year'] >= 2012].copy()
            
            logger.info(
                f"Loaded dataset. Train shape: {train.shape}, Test shape: {test.shape}"
            )
            return train, test
        
        except Exception as e:
            logger.error(f"Error in GenDataTransformation.load_split_data: {e}")
            raise e

    def standardizer(self, standardizer_type='robust'):

        if standardizer_type == 'standard':
            scaler = StandardScaler()
        elif standardizer_type == 'robust':
            scaler = RobustScaler()
        elif standardizer_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported standardizer: {standardizer_type}")

        return scaler

    def get_transformed_data(self):
        
        try:
            train, test = self.load_split_data()
            # Separate numeric features 
            feature_cols = [c for c in train.columns if c not in ['company_name', 'year',self.config.target_column]]

            x_train = train[feature_cols]
            x_test = test[feature_cols]
            # Encode target 
            y_train = self.encoder.fit_transform(train[self.config.target_column])
            y_test = self.encoder.transform(test[self.config.target_column])

            # get Scaler 
            scaler = self.standardizer(standardizer_type="robust")

            self.preprocessor= TransformationPipeline(scaler=scaler)
            train_scaled= self.preprocessor.fit_transform(x_train)
            test_scaled = self.preprocessor.transform(x_test)

              # Add back company_name and target                    
            train_scaled["company_name"] = train["company_name"].values                                    
            train_scaled[self.config.target_column] = y_train  
                                                                  
            test_scaled["company_name"] = test["company_name"].values                                                          
            test_scaled[self.config.target_column] = y_test    
                                                               
        
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            preprocessor_path= os.path.join(self.config.root_dir, "preprocessor.joblib")
            
            # save train and test ds
            train_scaled.to_csv(train_path, index=False)
            test_scaled.to_csv(test_path, index=False)
            
            # save preprocessor
            joblib.dump(self.preprocessor, preprocessor_path)                      
            logger.info("Saved train.csv, test.csv, and preprocessor.joblib")

        except Exception as e:
            logger.error(f"Error in DataTransformation.get_transformed_data: {e}")
            raise e

