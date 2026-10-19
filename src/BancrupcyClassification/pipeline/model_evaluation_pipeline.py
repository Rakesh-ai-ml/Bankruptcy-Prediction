from src.BancrupcyClassification.config.configuration import ConfigurationManager
from src.BancrupcyClassification.components.model_evaluation import ModelEvaluation


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self):
        config = ConfigurationManager()
        model_eval_config = config.get_model_evaluation_config()
        model_eval =ModelEvaluation(config=model_eval_config)
        model_eval.log_into_mlflow()
