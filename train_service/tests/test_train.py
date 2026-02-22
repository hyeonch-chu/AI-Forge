"""Tests for train_service/train.py.

Run inside the trainer container:
    docker exec trainer pytest tests/ -v

Or locally:
    pytest train_service/tests/ -v
"""
import argparse
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_yolo_results(tmp_path: Path) -> MagicMock:
    """Build a mock YOLO results object with a saved best.pt checkpoint."""
    weights_dir = tmp_path / "runs" / "train" / "exp" / "weights"
    weights_dir.mkdir(parents=True)
    best_pt = weights_dir / "best.pt"
    best_pt.write_bytes(b"fake-checkpoint")

    results = MagicMock()
    results.save_dir = str(weights_dir.parent)
    results.results_dict = {
        "metrics/mAP50": 0.72,
        "metrics/mAP50-95": 0.45,
        "train/box_loss": 0.31,
    }
    return results


@pytest.fixture()
def mock_mlflow():
    """Patch the entire mlflow namespace used by train.py."""
    with patch("train.mlflow") as m:
        run_ctx = MagicMock()
        run_ctx.__enter__ = MagicMock(return_value=run_ctx)
        run_ctx.__exit__ = MagicMock(return_value=False)
        run_ctx.info.run_id = "test-run-id-1234"
        m.start_run.return_value = run_ctx
        m.register_model.return_value = MagicMock(name="yolo_detector", version="1")
        yield m


# ---------------------------------------------------------------------------
# train() — argument forwarding
# ---------------------------------------------------------------------------
class TestTrainForwardArgs:
    def test_sets_experiment(self, mock_mlflow: MagicMock, mock_yolo_results: MagicMock) -> None:
        with patch("ultralytics.YOLO") as MockYOLO:
            MockYOLO.return_value.train.return_value = mock_yolo_results
            from train import train

            train(
                model_weights="yolov8n.pt",
                data_yaml="/data/dataset.yaml",
                epochs=1,
                imgsz=320,
                batch=2,
                experiment="test_exp",
                run_name="run1",
                device="cpu",
                register_name=None,
            )
        mock_mlflow.set_experiment.assert_called_once_with("test_exp")

    def test_logs_hyperparameters(
        self, mock_mlflow: MagicMock, mock_yolo_results: MagicMock
    ) -> None:
        with patch("ultralytics.YOLO") as MockYOLO:
            MockYOLO.return_value.train.return_value = mock_yolo_results
            from train import train

            train(
                model_weights="yolov8n.pt",
                data_yaml="/data/coco.yaml",
                epochs=10,
                imgsz=640,
                batch=8,
                experiment="exp",
                run_name=None,
                device="cpu",
                register_name=None,
            )
        logged_params = mock_mlflow.log_params.call_args[0][0]
        assert logged_params["epochs"] == 10
        assert logged_params["imgsz"] == 640
        assert logged_params["batch"] == 8
        assert logged_params["model_weights"] == "yolov8n.pt"

    def test_calls_yolo_train_with_correct_args(
        self, mock_mlflow: MagicMock, mock_yolo_results: MagicMock
    ) -> None:
        with patch("ultralytics.YOLO") as MockYOLO:
            mock_model = MockYOLO.return_value
            mock_model.train.return_value = mock_yolo_results
            from train import train

            train(
                model_weights="yolov8s.pt",
                data_yaml="/data/voc.yaml",
                epochs=5,
                imgsz=416,
                batch=4,
                experiment="exp",
                run_name="r1",
                device="cpu",
                register_name=None,
            )
        mock_model.train.assert_called_once()
        _, kwargs = mock_model.train.call_args
        assert kwargs["data"] == "/data/voc.yaml"
        assert kwargs["epochs"] == 5
        assert kwargs["imgsz"] == 416
        assert kwargs["batch"] == 4


# ---------------------------------------------------------------------------
# train() — MLflow metric logging
# ---------------------------------------------------------------------------
class TestTrainMetricLogging:
    def test_logs_metrics_from_results_dict(
        self, mock_mlflow: MagicMock, mock_yolo_results: MagicMock
    ) -> None:
        with patch("ultralytics.YOLO") as MockYOLO:
            MockYOLO.return_value.train.return_value = mock_yolo_results
            from train import train

            train(
                model_weights="yolov8n.pt",
                data_yaml="/data/coco.yaml",
                epochs=1,
                imgsz=320,
                batch=2,
                experiment="exp",
                run_name=None,
                device="cpu",
                register_name=None,
            )
        logged_keys = {c.args[0] for c in mock_mlflow.log_metric.call_args_list}
        assert "metrics/mAP50" in logged_keys
        assert "metrics/mAP50-95" in logged_keys
        assert "train/box_loss" in logged_keys


# ---------------------------------------------------------------------------
# train() — artifact & model registry
# ---------------------------------------------------------------------------
class TestTrainArtifacts:
    def test_logs_best_pt_artifact(
        self, mock_mlflow: MagicMock, mock_yolo_results: MagicMock
    ) -> None:
        with patch("ultralytics.YOLO") as MockYOLO:
            MockYOLO.return_value.train.return_value = mock_yolo_results
            from train import train

            train(
                model_weights="yolov8n.pt",
                data_yaml="/data/coco.yaml",
                epochs=1,
                imgsz=320,
                batch=2,
                experiment="exp",
                run_name=None,
                device="cpu",
                register_name=None,
            )
        mock_mlflow.log_artifact.assert_called_once()
        artifact_path_arg = mock_mlflow.log_artifact.call_args[1]["artifact_path"]
        assert artifact_path_arg == "weights"

    def test_registers_model_when_register_name_set(
        self, mock_mlflow: MagicMock, mock_yolo_results: MagicMock
    ) -> None:
        with patch("ultralytics.YOLO") as MockYOLO:
            MockYOLO.return_value.train.return_value = mock_yolo_results
            from train import train

            train(
                model_weights="yolov8n.pt",
                data_yaml="/data/coco.yaml",
                epochs=1,
                imgsz=320,
                batch=2,
                experiment="exp",
                run_name=None,
                device="cpu",
                register_name="yolo_detector",
            )
        mock_mlflow.register_model.assert_called_once()
        _, kwargs = mock_mlflow.register_model.call_args
        assert kwargs["name"] == "yolo_detector"

    def test_skips_registration_when_register_name_is_none(
        self, mock_mlflow: MagicMock, mock_yolo_results: MagicMock
    ) -> None:
        with patch("ultralytics.YOLO") as MockYOLO:
            MockYOLO.return_value.train.return_value = mock_yolo_results
            from train import train

            train(
                model_weights="yolov8n.pt",
                data_yaml="/data/coco.yaml",
                epochs=1,
                imgsz=320,
                batch=2,
                experiment="exp",
                run_name=None,
                device="cpu",
                register_name=None,
            )
        mock_mlflow.register_model.assert_not_called()


# ---------------------------------------------------------------------------
# parse_args()
# ---------------------------------------------------------------------------
class TestParseArgs:
    def test_required_data_flag(self) -> None:
        from train import parse_args

        args = parse_args.__wrapped__ if hasattr(parse_args, "__wrapped__") else parse_args
        with pytest.raises(SystemExit):
            # Missing --data should cause argparse to exit
            with patch("sys.argv", ["train.py"]):
                parse_args()

    def test_defaults(self) -> None:
        from train import parse_args

        with patch("sys.argv", ["train.py", "--data", "/d/coco.yaml"]):
            args = parse_args()
        assert args.model == "yolov8n.pt"
        assert args.epochs == 50
        assert args.imgsz == 640
        assert args.batch == 16
        assert args.experiment == "yolo_training"
        assert args.device == "cpu"
        assert args.register_name is None

    def test_custom_values(self) -> None:
        from train import parse_args

        with patch(
            "sys.argv",
            [
                "train.py",
                "--data", "/d/voc.yaml",
                "--model", "yolov8s.pt",
                "--epochs", "100",
                "--imgsz", "416",
                "--batch", "32",
                "--experiment", "my_exp",
                "--run-name", "run_v1",
                "--device", "0",
                "--register", "prod_model",
            ],
        ):
            args = parse_args()
        assert args.model == "yolov8s.pt"
        assert args.epochs == 100
        assert args.register_name == "prod_model"
