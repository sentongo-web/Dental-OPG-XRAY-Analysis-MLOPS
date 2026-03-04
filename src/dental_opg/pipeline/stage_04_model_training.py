import logging
from dental_opg.config.configuration import ConfigurationManager
from dental_opg.components.model_trainer import ModelTrainer

logger = logging.getLogger(__name__)
STAGE_NAME = "Model Training Stage"


def main():
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    config = ConfigurationManager()
    model_trainer_config = config.get_model_trainer_config()
    model_trainer = ModelTrainer(config=model_trainer_config)
    model_trainer.run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'='*60}")


if __name__ == "__main__":
    main()
