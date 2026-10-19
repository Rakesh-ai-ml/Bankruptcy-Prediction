import os
from src.BancrupcyClassification.constants import *
from src.BancrupcyClassification.utils.common import read_yaml, create_directories
from pydantic import validate_call
from src.BancrupcyClassification.entity.config_entity import (DataIngestionConfig,
                                                              DataValidationConfig,
                                                              DataTransformationConfig,
                                                              ModelTrainerConfig,
                                                              ModelEvaluationConfig,
                                                              PredictionConfig
                                                           )

from src.BancrupcyClassification.constants import (
                                                    CONFIG_FILE_PATH,
                                                    SCHEMA_FILE_PATH
                                                )


class ConfigurationManager:

    def __init__(
                self,
                config_filepath=CONFIG_FILE_PATH,
                # params_filepath = PARAMS_FILE_PATH,
                schema_filepath = SCHEMA_FILE_PATH
                ):
        
        self.config=read_yaml(config_filepath)
        self.schema=read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])

                
    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config
        create_directories([config.data_ingestion.root_dir])
        return DataIngestionConfig(**self.config.data_ingestion)
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config
        schema = self.schema.COLUMNS
        target = self.schema.TARGET
        create_directories([config.data_validation.root_dir])
        return DataValidationConfig(**self.config.data_validation, all_schema=schema,target_col=target)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config
        features=self.schema.COLUMNS
        target_column = self.schema.TARGET
        create_directories([config.data_transformation.root_dir])
        return DataTransformationConfig(**self.config.data_transformation,features=features, target_column=target_column)

    def get_model_training_config(self) -> ModelTrainerConfig:
        config=self.config
        target_column = self.schema.TARGET
        create_directories([config.model_trainer.root_dir])
        return ModelTrainerConfig(**self.config.model_trainer,target_column=target_column)

    def get_model_evaluation_config(self) ->ModelEvaluationConfig:
        config=self.config
        target_column = self.schema.TARGET
        create_directories([config.model_evaluation.root_dir])
        return ModelEvaluationConfig(**self.config.model_evaluation,target_column=target_column)

    def get_prediction_config(self) ->PredictionConfig:

        return PredictionConfig(**self.config.prediction)  
        
