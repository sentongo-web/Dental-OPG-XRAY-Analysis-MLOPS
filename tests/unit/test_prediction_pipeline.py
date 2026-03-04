"""Unit tests for prediction pipeline."""
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPredictionPipeline:
    """Tests for the prediction pipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock YOLO model."""
        model = MagicMock()
        box = MagicMock()
        box.xyxy = [MagicMock(tolist=lambda: [10.0, 20.0, 100.0, 150.0])]
        box.conf = [MagicMock(__float__=lambda s: 0.85)]
        box.cls = [MagicMock(__int__=lambda s: 0)]

        result = MagicMock()
        result.boxes = [box]
        model.return_value = [result]
        return model

    @pytest.fixture
    def sample_opg_image(self):
        """Create a synthetic grayscale OPG-like test image."""
        img = np.zeros((480, 1024, 3), dtype=np.uint8)
        # Simulate teeth structure
        for i in range(16):
            x = 50 + i * 60
            import cv2
            cv2.rectangle(img, (x, 200), (x + 40, 350), (180, 180, 180), -1)
        return img

    def test_severity_assessment_no_cavities(self):
        """Test severity assessment with no detections."""
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        with patch.object(PredictionPipeline, "_load_model"):
            pp = PredictionPipeline.__new__(PredictionPipeline)
            pp.class_names = ["cavity"]
            pp.model = None
            result = pp._assess_severity(0, 0.0)
            assert "No cavities" in result

    def test_severity_assessment_mild(self):
        """Test mild severity assessment."""
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        with patch.object(PredictionPipeline, "_load_model"):
            pp = PredictionPipeline.__new__(PredictionPipeline)
            pp.class_names = ["cavity"]
            result = pp._assess_severity(1, 0.8)
            assert "Mild" in result

    def test_severity_assessment_moderate(self):
        """Test moderate severity assessment."""
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        with patch.object(PredictionPipeline, "_load_model"):
            pp = PredictionPipeline.__new__(PredictionPipeline)
            pp.class_names = ["cavity"]
            result = pp._assess_severity(4, 0.7)
            assert "Moderate" in result

    def test_severity_assessment_severe(self):
        """Test severe severity assessment."""
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        with patch.object(PredictionPipeline, "_load_model"):
            pp = PredictionPipeline.__new__(PredictionPipeline)
            pp.class_names = ["cavity"]
            result = pp._assess_severity(8, 0.9)
            assert "Severe" in result

    def test_visualization(self, sample_opg_image):
        """Test visualization produces valid image."""
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        with patch.object(PredictionPipeline, "_load_model"):
            pp = PredictionPipeline.__new__(PredictionPipeline)
            pp.class_names = ["cavity"]
            pp.CAVITY_COLOR = (0, 0, 255)
            pp.TEXT_COLOR = (255, 255, 255)
            pp.OVERLAY_ALPHA = 0.3

            detections = [
                {
                    "class_id": 0,
                    "class_name": "cavity",
                    "confidence": 0.85,
                    "bbox": [50, 200, 100, 300],
                    "bbox_normalized": [0.05, 0.42, 0.10, 0.63],
                }
            ]
            result = pp._visualize(sample_opg_image, detections)
            assert result is not None
            assert result.shape == sample_opg_image.shape

    def test_visualization_no_detections(self, sample_opg_image):
        """Test visualization with no detections."""
        from dental_opg.pipeline.prediction_pipeline import PredictionPipeline
        with patch.object(PredictionPipeline, "_load_model"):
            pp = PredictionPipeline.__new__(PredictionPipeline)
            pp.class_names = ["cavity"]
            pp.CAVITY_COLOR = (0, 0, 255)
            pp.TEXT_COLOR = (255, 255, 255)
            pp.OVERLAY_ALPHA = 0.3
            result = pp._visualize(sample_opg_image, [])
            assert result is not None
            assert result.shape == sample_opg_image.shape
