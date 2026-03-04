"""
Integration tests for the full MLOps pipeline.
These tests require the actual dataset and trained model.
Run with: pytest tests/integration/ -v --integration
"""
import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires data/model)"
    )


@pytest.mark.integration
class TestDataTransformationIntegration:
    """Integration tests for data transformation."""

    def test_yolo_format_detection(self, tmp_path):
        """Test YOLO format auto-detection."""
        from dental_opg.components.data_transformation import DataTransformation
        from dental_opg.entity.config_entity import DataTransformationConfig

        # Create mock YOLO dataset
        img_dir = tmp_path / "images"
        lbl_dir = tmp_path / "labels"
        img_dir.mkdir()
        lbl_dir.mkdir()

        # Create fake images and labels
        import cv2
        for i in range(10):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)
            with open(lbl_dir / f"img_{i:03d}.txt", "w") as f:
                f.write(f"0 0.5 0.5 0.2 0.3\n")

        config = DataTransformationConfig(
            root_dir=tmp_path / "output",
            data_path=tmp_path,
            output_dir=tmp_path / "dataset",
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1,
            image_size=320,
            augmentation_factor=1,
        )
        dt = DataTransformation(config=config)
        fmt = dt._detect_format(tmp_path)
        assert fmt == "yolo"


@pytest.mark.integration
class TestModelEvaluationIntegration:
    """Integration tests for model evaluation."""

    @pytest.mark.skipif(
        not Path("models/best/best.pt").exists(),
        reason="Trained model not available"
    )
    def test_model_loads(self):
        """Test that trained model loads correctly."""
        from ultralytics import YOLO
        model = YOLO("models/best/best.pt")
        assert model is not None

    @pytest.mark.skipif(
        not Path("models/best/best.pt").exists(),
        reason="Trained model not available"
    )
    def test_prediction_on_synthetic_image(self):
        """Test prediction on synthetic OPG image."""
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        pipeline = PredictionPipeline(model_path="models/best/best.pt")
        # Synthetic 640x640 grayscale image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = pipeline.predict(img)
        assert "cavity_count" in result
        assert "detections" in result
        assert "severity" in result
        assert isinstance(result["cavity_count"], int)
