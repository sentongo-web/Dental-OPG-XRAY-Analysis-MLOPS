import logging
from dental_opg.config.configuration import ConfigurationManager
from dental_opg.components.data_ingestion import DataIngestion

logger = logging.getLogger(__name__)
STAGE_NAME = "Data Ingestion Stage"


def main():
    logger.info(f">>>>>> Stage: {STAGE_NAME} started <<<<<<")
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.run()
    logger.info(f">>>>>> Stage: {STAGE_NAME} completed <<<<<<\n\n{'='*60}")


if __name__ == "__main__":
    main()
