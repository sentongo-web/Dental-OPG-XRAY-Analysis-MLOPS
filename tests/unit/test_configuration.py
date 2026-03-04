"""Unit tests for configuration management."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestConfigurationManager:
    """Test configuration loading and entity creation."""

    def test_config_loading(self, tmp_path):
        """Test that config files are loaded correctly."""
        from dental_opg.config.configuration import ConfigurationManager
        config = ConfigurationManager()
        assert config.config is not None
        assert config.params is not None

    def test_data_ingestion_config(self, tmp_path):
        """Test DataIngestionConfig creation."""
        from dental_opg.config.configuration import ConfigurationManager
        config = ConfigurationManager()
        di_config = config.get_data_ingestion_config()
        assert di_config.root_dir is not None
        assert di_config.source_url is not None
        assert "drive.google.com" in di_config.source_url

    def test_data_validation_config(self):
        """Test DataValidationConfig creation."""
        from dental_opg.config.configuration import ConfigurationManager
        config = ConfigurationManager()
        dv_config = config.get_data_validation_config()
        assert dv_config.root_dir is not None
        assert isinstance(dv_config.required_files, list)

    def test_model_trainer_config(self):
        """Test ModelTrainerConfig creation."""
        from dental_opg.config.configuration import ConfigurationManager
        config = ConfigurationManager()
        mt_config = config.get_model_trainer_config()
        assert mt_config.model_name is not None
        assert "yolov8" in mt_config.model_name.lower()

    def test_all_configs_created(self):
        """Test all 5 configuration entities are created without errors."""
        from dental_opg.config.configuration import ConfigurationManager
        config = ConfigurationManager()
        configs = [
            config.get_data_ingestion_config(),
            config.get_data_validation_config(),
            config.get_data_transformation_config(),
            config.get_model_trainer_config(),
            config.get_model_evaluation_config(),
        ]
        assert len(configs) == 5
        for c in configs:
            assert c is not None


class TestDataIngestion:
    """Test DataIngestion component."""

    def test_gdrive_id_extraction(self):
        """Test Google Drive file ID extraction."""
        from dental_opg.components.data_ingestion import DataIngestion
        from dental_opg.entity.config_entity import DataIngestionConfig

        config = DataIngestionConfig(
            root_dir=Path("test_root"),
            source_url="https://drive.google.com/file/d/1mm0L2jRPKqCpyxXRsSJKVoj63-SXncv1/view?usp=sharing",
            local_data_file=Path("test_root/data.zip"),
            unzip_dir=Path("test_root/unzipped"),
        )
        di = DataIngestion(config=config)
        file_id = di._extract_gdrive_id(config.source_url)
        assert file_id == "1mm0L2jRPKqCpyxXRsSJKVoj63-SXncv1"

    def test_gdrive_id_extraction_id_format(self):
        """Test GDrive ID extraction from id= URL format."""
        from dental_opg.components.data_ingestion import DataIngestion
        from dental_opg.entity.config_entity import DataIngestionConfig

        config = DataIngestionConfig(
            root_dir=Path("test_root"),
            source_url="https://drive.google.com/uc?id=abc123&export=download",
            local_data_file=Path("test_root/data.zip"),
            unzip_dir=Path("test_root/unzipped"),
        )
        di = DataIngestion(config=config)
        file_id = di._extract_gdrive_id(config.source_url)
        assert file_id == "abc123"


class TestDataValidation:
    """Test DataValidation component."""

    def test_is_float(self):
        """Test float detection utility."""
        from dental_opg.components.data_validation import DataValidation
        from dental_opg.entity.config_entity import DataValidationConfig

        config = DataValidationConfig(
            root_dir=Path("test"),
            STATUS_FILE="test/status.txt",
            required_files=["images", "labels"],
            required_annotation_formats=["yolo"],
        )
        dv = DataValidation(config=config)
        assert DataValidation._is_float("0.5") is True
        assert DataValidation._is_float("1") is True
        assert DataValidation._is_float("not_a_float") is False


class TestCommonUtils:
    """Test utility functions."""

    def test_read_yaml(self, tmp_path):
        """Test YAML reading returns ConfigBox."""
        from dental_opg.utils.common import read_yaml
        from box import ConfigBox

        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nnested:\n  a: 1\n")
        result = read_yaml(yaml_file)
        assert isinstance(result, ConfigBox)
        assert result.key == "value"
        assert result.nested.a == 1

    def test_create_directories(self, tmp_path):
        """Test directory creation."""
        from dental_opg.utils.common import create_directories
        dirs = [tmp_path / "dir1", tmp_path / "dir2" / "nested"]
        create_directories(dirs)
        for d in dirs:
            assert d.exists()

    def test_save_and_load_json(self, tmp_path):
        """Test JSON save and load."""
        from dental_opg.utils.common import save_json, load_json
        data = {"metric": 0.95, "count": 10, "name": "test"}
        json_path = tmp_path / "test.json"
        save_json(json_path, data)
        loaded = load_json(json_path)
        assert loaded.metric == 0.95
        assert loaded.count == 10

    def test_get_size(self, tmp_path):
        """Test file size function."""
        from dental_opg.utils.common import get_size
        test_file = tmp_path / "test.txt"
        test_file.write_text("x" * 1024)  # ~1 KB
        size = get_size(test_file)
        assert "KB" in size
