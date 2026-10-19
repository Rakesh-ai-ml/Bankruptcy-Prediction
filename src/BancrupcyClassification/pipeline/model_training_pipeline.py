from src.BancrupcyClassification.config.configuration import ConfigurationManager
from src.BancrupcyClassification.components.model_training import ModelTrainer


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
        config = ConfigurationManager()
        model_config = config.get_model_training_config()
        model_training = ModelTrainer(config=model_config)
        model_training.run()

