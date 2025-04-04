from rare_event_detection.api import prepare_kmeans_model, get_REI_from_testing_scan
from rare_event_detection.bragg_emb import train_bragg_embedding

from pathlib import Path
import argparse
import logging

def main(DATA_DIR: Path, EXPERIMENT_DIR: Path):

    # The following files need to exist before running the program
    baseline_scan_path = DATA_DIR / "base.edf.ge5"
    test_scan_path = DATA_DIR / "test.edf.ge5"
    baseline_dark_path = DATA_DIR / "dark_4_base.edf.ge5"
    test_dark_path = DATA_DIR / "dark_4_test.edf.ge5"

    assert baseline_scan_path.exists(), f"Baseline scan file {baseline_scan_path} does not exist."
    assert test_scan_path.exists(), f"Test scan file {test_scan_path} does not exist."
    assert baseline_dark_path.exists(), f"Baseline dark file {baseline_dark_path} does not exist."
    assert test_dark_path.exists(), f"Test dark file {test_dark_path} does not exist."

    # The following files will be created by the program
    EmbeddingModelIterDir = EXPERIMENT_DIR / "embedding_model_iter"
    EmbeddingModelIterDir.mkdir(parents=True, exist_ok=True)
    KMeansModelPath = EXPERIMENT_DIR / "kmeans_model.pkl"

    logging.info("Starting the rare event detection pipeline...")
    # Step 1. Train the embedding model using the baseline scan
    train_bragg_embedding(
        training_scan_file=baseline_scan_path,
        training_dark_file=baseline_dark_path,
        itr_out_dir=EmbeddingModelIterDir)
    
    logging.info("Training completed. Preparing KMeans model...")
    # We use the last checkpoint for the default setting
    # You can also specify the checkpoint path directly
    trained_encoder_path = EmbeddingModelIterDir / "script-ep00100.pth"

    # Step 2. Prepare the KMeans model using the trained embedding model
    prepare_kmeans_model(
        trained_encoder_path=EmbeddingModelIterDir,
        base_line_scan_path=baseline_scan_path,
        dark_4_base_line_path=baseline_dark_path,
        kmeans_model_save_path=KMeansModelPath)
    
    logging.info("KMeans model preparation completed. Starting REI calculation...")
    # Step 3. Get the REI from the test scan using the trained embedding model and KMeans model
    REI_score, time_consumed = get_REI_from_testing_scan(
        trained_encoder_path=EmbeddingModelIterDir,
        testing_scan_path=test_scan_path,
        testing_scan_dark_path=test_dark_path,
        kmeans_model_path=KMeansModelPath)
    
    logging.info("REI calculation completed.")
    print(f"REI score: {REI_score}")
    print(f"Time consumed: {time_consumed:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the rare event detection pipeline.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Path to the data directory.")
    parser.add_argument("--experiment_dir", type=str, default="experiment",
                        help="Path to the experiment directory.")
    
    args = parser.parse_args()

    # We recommend putting the data and experiment folder in a project folder in the HPC system, rather than in your user's folder
    DATA_DIR = Path(args.data_dir)
    EXPERIMENT_DIR = Path(args.experiment_dir)

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    # Make sure the data directory exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"The data directory {DATA_DIR} does not exist.")

    main(DATA_DIR, EXPERIMENT_DIR)