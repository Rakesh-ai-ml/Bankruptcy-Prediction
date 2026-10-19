from src.BancrupcyClassification.logger import logger
from src.BancrupcyClassification.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.BancrupcyClassification.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.BancrupcyClassification.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.BancrupcyClassification.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.BancrupcyClassification.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

STAGE_NAME = "Data Ingestion stage"

try: 
   logger.info(f"---------------------------- stage {STAGE_NAME} started ----------------------------") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.initiate_data_ingestion()
   logger.info(f"---------------------------- stage {STAGE_NAME} completed ----------------------------\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation stage"

try:
   logger.info(f"---------------------------- stage {STAGE_NAME} started ----------------------------") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.initiate_data_validation()
   logger.info(f"---------------------------- stage {STAGE_NAME} completed ----------------------------\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Transformation stage"

try:
   logger.info(f"---------------------------- stage {STAGE_NAME} started ----------------------------") 
   data_validation = DataTransformationTrainingPipeline()
   data_validation.initiate_data_transformation()
   logger.info(f"---------------------------- stage {STAGE_NAME} completed ----------------------------\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Model Training stage"

try: 
   logger.info(f"---------------------------- stage {STAGE_NAME} started ----------------------------") 
   model_train = ModelTrainingPipeline()
   model_train.initiate_model_training()
   logger.info(f"---------------------------- stage {STAGE_NAME} completed ----------------------------\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Model evaluation stage"

try: 
   logger.info(f"---------------------------- stage {STAGE_NAME} started ----------------------------") 
   model_eval = ModelEvaluationPipeline()
   model_eval.initiate_model_evaluation()
   logger.info(f"---------------------------- stage {STAGE_NAME} completed ----------------------------\n\n\n")
except Exception as e:
        logger.exception(e)
        raise e
