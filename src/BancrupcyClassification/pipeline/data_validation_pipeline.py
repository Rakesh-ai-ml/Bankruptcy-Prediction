from src.BancrupcyClassification.config.configuration import ConfigurationManager
from src.BancrupcyClassification.components.data_validation import DataValidation


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()

