import os 
import pandas as pd
from src.BancrupcyClassification.logger import logger

class DataValidation:

    def __init__(self, config):
        self.config = config
    
    def validate_all_columns(self)-> bool:

        try:
            validation_status = None
            data = pd.read_csv(self.config.unzip_data_dir)
            data=data.rename(columns=
            {
            "X1"	:"Current assets",
            "X2"	:"Cost of goods sold",
            "X3"	:"Depreciation and amortization",
            "X4"	:"EBITDA",
            "X5"	:"Inventory" ,
            "X6"	:"Net Income",
            "X7"	:"Total Receivables",
            "X8"	:"Market value",
            "X9"	:"Net sales",
            "X10"	:"Total assets",
            "X11"	:"Total Long-term debt",
            "X12"	:"EBIT",
            "X13"	:"Gross Profit",
            "X14"	:"Total Current Liabilities",
            "X15"	:"Retained Earnings" ,
            "X16"	:"Total Revenue",
            "X17"	:"Total Liabilities",
            "X18"	:"Total Operating Expenses"
            }
            )
            all_cols = list(data.columns) 
            all_schema = self.config.all_schema
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
            return validation_status
        except Exception as e:
            raise e