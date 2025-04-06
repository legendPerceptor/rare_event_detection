# Rare Event Detection

This repo serves as a library to train an BraggEmb model and detect rare events in the APS workflow.

A complete example on how to use the library can be found in [example.py](example.py).

There are 3 steps in the workflow to handle the data. In the real scenario, the data needs to be transferred from the APS machine to a supercomputer, and we hope to improve the transfer efficiency by using lossy compression while respecting the data fidelity. We use the analysis result REI score to determine whether a data is still usable after compression.

## Step 1. Train the embedding model using the baseline scan

```python
from rare_event_detection.bragg_emb import train_bragg_embedding
train_bragg_embedding(
    training_scan_file=baseline_scan_path,
    training_dark_file=baseline_dark_path,
    itr_out_dir=EmbeddingModelIterDir)
```

## Step 2. Prepare the KMeans model using the trained embedding model

```python
from rare_event_detection.api import prepare_kmeans_model
prepare_kmeans_model(
    trained_encoder_path=trained_encoder_path,
    base_line_scan_path=baseline_scan_path,
    dark_4_base_line_path=baseline_dark_path,
    kmeans_model_save_path=KMeansModelPath)
```

## Step 3. Get the REI from the test scan using the trained embedding model and KMeans model

```python
from rare_event_detection.api import get_REI_from_testing_scan
REI_score, time_consumed = get_REI_from_testing_scan(
    trained_encoder_path=trained_encoder_path,
    testing_scan_path=test_scan_path,
    testing_scan_dark_path=test_dark_path,
    kmeans_model_path=KMeansModelPath)
```

We also prepared a [run_example.sh](run_example.sh) file for users to run the pipeline in a HPC system that uses PBS scheduler. Remember to change the paths in this file to suit your own needs.
