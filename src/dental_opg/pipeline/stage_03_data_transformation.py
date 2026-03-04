import logging
from dental_opg.config.configuration import ConfigurationManager
from dental_opg.components.data_transformation import DataTransformation

logger = logging.getLogger(__name__)
STAGE_NAME = "Data Transformation Stage"


def main():
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'='*60}")


if __name__ == "__main__":
    main()
