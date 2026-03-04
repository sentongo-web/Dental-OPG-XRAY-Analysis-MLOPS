"""
Main training pipeline orchestrator for Dental OPG Cavity Detection.
Runs all 5 stages sequentially with proper logging.

Usage:
    python main.py                          # run all stages
    python main.py --stage 1               # run specific stage
    python main.py --stages 1,2,3          # run multiple stages
    python main.py --skip-ingestion        # skip download if data exists
"""
import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# Ensure logs dir exists
Path("logs").mkdir(exist_ok=True)


def run_pipeline(stages: list = None, skip_ingestion: bool = False):
    """Run the complete MLOps pipeline."""

    from dental_opg.pipeline.stage_01_data_ingestion import main as stage1
    from dental_opg.pipeline.stage_02_data_validation import main as stage2
    from dental_opg.pipeline.stage_03_data_transformation import main as stage3
    from dental_opg.pipeline.stage_04_model_training import main as stage4
    from dental_opg.pipeline.stage_05_model_evaluation import main as stage5

    all_stages = {
        1: ("Data Ingestion", stage1),
        2: ("Data Validation", stage2),
        3: ("Data Transformation", stage3),
        4: ("Model Training", stage4),
        5: ("Model Evaluation", stage5),
    }

    if stages is None:
        stages = list(all_stages.keys())
        if skip_ingestion:
            stages.remove(1)

    logger.info("=" * 70)
    logger.info("   DENTAL OPG CAVITY DETECTION — MLOPS PIPELINE")
    logger.info("   YOLOv8 | DVC | MLflow | HuggingFace")
    logger.info("=" * 70)

    for stage_num in stages:
        if stage_num not in all_stages:
            logger.warning(f"Stage {stage_num} not found, skipping")
            continue
        stage_name, stage_fn = all_stages[stage_num]
        try:
            logger.info(f"\n{'#'*60}")
            logger.info(f"  RUNNING STAGE {stage_num}: {stage_name}")
            logger.info(f"{'#'*60}")
            stage_fn()
        except Exception as e:
            logger.error(f"Stage {stage_num} ({stage_name}) failed: {e}")
            raise e

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("  Next steps:")
    logger.info("  - View MLflow UI: mlflow ui")
    logger.info("  - Run prediction: python -m dental_opg.pipeline.prediction_pipeline --image <path>")
    logger.info("  - Launch app: python app/gradio_app.py")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Dental OPG Cavity Detection MLOps Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run all stages
  python main.py --stage 4               # Run only training
  python main.py --stages 1,2,3          # Run stages 1-3
  python main.py --skip-ingestion        # Skip download
        """
    )
    parser.add_argument("--stage", type=int, help="Run single stage (1-5)")
    parser.add_argument("--stages", type=str, help="Run comma-separated stages, e.g. 1,2,3")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip data download")
    args = parser.parse_args()

    if args.stage:
        stages = [args.stage]
    elif args.stages:
        stages = [int(s.strip()) for s in args.stages.split(",")]
    else:
        stages = None

    run_pipeline(stages=stages, skip_ingestion=args.skip_ingestion)


if __name__ == "__main__":
    main()
