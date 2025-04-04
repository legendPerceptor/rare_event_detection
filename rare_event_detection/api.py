from embed import Embed
from cluster import Cluster
from pathlib import Path
from utility import find_dataset_single
from run_detection import ds_anamoly_quantify

import time
import numpy as np

def prepare_kmeans_model(trained_encoder_path: Path,
                         base_line_scan_path: Path,
                         dark_4_base_line_path: Path,
                         kmeans_model_save_path: Path,
                         threshold: int=100,
                         num_of_clusters: int=40,
                         uq_threshold: float=0.4):
    baseh5_dataset, _, _ = find_dataset_single(base_line_scan_path, dark_4_base_line_path, threshold, "base")
    print("The baseh5_dataset: ", baseh5_dataset)
    embmdl = Embed(trained_encoder_path)
    emb_bl, _ = embmdl.peak2emb_missingwedge(baseh5_dataset[0])
    clusmdl = Cluster(numClusters=num_of_clusters)
    clusmdl.train(emb_bl, kmeans_model_save_path)

    dist_and_uq = [] # used to store distribution and UQ for all datasets
    dataset_tag = [base_line_scan_path]
    uq_bl, dist_bl = clusmdl.kmeans_clustering_and_dist(emb_bl, min_score=uq_threshold)
    dist_and_uq.append(np.append(dist_bl, uq_bl))
    print("The uq of baseline: ", uq_bl)


def get_REI_from_testing_scan(trained_encoder_path: Path,
                              testing_scan_path: Path,
                              testing_scan_dark_path: Path,
                              kmeans_model_path: Path,
                              threshold: int=100, uq_threshold: float=0.4, num_of_clusters: int=40):
    list_datasets, list_pressures, list_idx = find_dataset_single(testing_scan_dark_path, testing_scan_dark_path, threshold, "test")
    # create a new embed class
    embmdl = Embed(trained_encoder_path)

    clusmdl = Cluster(numClusters = num_of_clusters)
    print("kmeans initialize is done ...")

    dist_and_uq = [] # used to store distribution and UQ for all datasets
    dataset_tag = [testing_scan_path]
    result = None
    
    total_patches = 0

    tic = time.perf_counter()
    for i in range(len(list_datasets)):
        test_degrees = 360
        result = ds_anamoly_quantify(list_datasets[i], None, embmdl, clusmdl, 
                                              uq_threshold, kmeans_model_path, degs=test_degrees)
        dataset_tag.append(list_datasets[i])
        dist_and_uq.append(result[1])
        total_patches += result[2]
    toc = time.perf_counter()
    REI_score = result[1][-1]
    time_consumed = toc - tic
    print(f'the REI score for the testing data is {REI_score}')
    print(f"it takes {time_consumed:0.4f} seconds to test {len(list_datasets)} datasets")
    print(f"the average patches for each dataset is {total_patches/(len(list_datasets))}")
    return REI_score, time_consumed