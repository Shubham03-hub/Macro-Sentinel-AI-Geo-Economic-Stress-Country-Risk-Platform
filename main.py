"""
Single entrypoint for the Geo-Economic Stress Prediction pipeline.

Usage:
    python main.py --stage all
    python main.py --stage preprocess
    python main.py --stage features
    python main.py --stage train
    python main.py --stage evaluate
    python main.py --stage predict
"""

import argparse
import sys
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)

STAGES = ["preprocess", "features", "train", "evaluate", "predict"]


def run_preprocess():
    from src.preprocessing.preprocessing_pipeline import run_preprocessing_pipeline
    return run_preprocessing_pipeline()


def run_features():
    from src.utils.helper import load_config, resolve_path, ensure_dir
    from src.features.feature_engineering import FeatureEngineer
    import pandas as pd

    cfg = load_config()
    cleaned_path = resolve_path(cfg["paths"]["interim_dir"]) / "cleaned_dataset.csv"
    if not cleaned_path.exists():
        logger.info("cleaned_dataset.csv not found — running preprocess stage first")
        run_preprocess()
    data = pd.read_csv(cleaned_path)

    engineer = FeatureEngineer(cfg)
    featured = engineer.run(data)

    out_path = resolve_path(cfg["paths"]["processed_dir"]) / cfg["processed_files"]["feature_dataset"]
    ensure_dir(out_path.parent)
    featured.to_csv(out_path, index=False)
    logger.info(f"Feature dataset saved to {out_path} — shape {featured.shape}")
    return featured


def run_train():
    from src.training.train_model import run_training
    return run_training()


def run_evaluate():
    from src.evaluation.evaluate_model import evaluate_champion
    return evaluate_champion()


def run_predict():
    from src.prediction.predict import generate_predictions
    return generate_predictions()


DISPATCH = {
    "preprocess": run_preprocess,
    "features": run_features,
    "train": run_train,
    "evaluate": run_evaluate,
    "predict": run_predict,
}


def main():
    parser = argparse.ArgumentParser(description="Geo-Economic Stress Prediction pipeline")
    parser.add_argument(
        "--stage",
        choices=STAGES + ["all"],
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    stages_to_run = STAGES if args.stage == "all" else [args.stage]

    logger.info(f"Running stages: {stages_to_run}")
    start = time.time()

    for stage in stages_to_run:
        logger.info(f"--- Stage: {stage} ---")
        stage_start = time.time()
        try:
            DISPATCH[stage]()
        except Exception:
            logger.exception(f"Stage '{stage}' failed")
            sys.exit(1)
        logger.info(f"--- Stage '{stage}' complete in {time.time() - stage_start:.1f}s ---")

    logger.info(f"Pipeline finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
