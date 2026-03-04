import logging
from dental_opg.config.configuration import ConfigurationManager
from dental_opg.components.data_validation import DataValidation

logger = logging.getLogger(__name__)
STAGE_NAME = "Data Validation Stage"


def main():
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation = DataValidation(config=data_validation_config)
    data_validation.run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'='*60}")


if __name__ == "__main__":
    main()
