import logging
from dental_opg.config.configuration import ConfigurationManager
from dental_opg.components.model_evaluation import ModelEvaluation

logger = logging.getLogger(__name__)
STAGE_NAME = "Model Evaluation Stage"


def main():
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    config = ConfigurationManager()
    model_evaluation_config = config.get_model_evaluation_config()
    model_evaluation = ModelEvaluation(config=model_evaluation_config)
    metrics = model_evaluation.run()
    logger.info(f"Final metrics: {metrics}")
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'='*60}")
    return metrics


if __name__ == "__main__":
    main()
