"""
calc_similarity.py

This module contains functions for similarity calculations using k-nearest neighbors (k-NN).
The functions leverage the Annoy library to efficiently find the nearest neighbors for descriptor vectors.

Functions:
- annoy_knn(database_index: AnnoyIndex, nr_knn: int, q_vec: np.ndarray) -> int, float: Finds the nearest neighbor(s)
  in the database for a given query vector using Annoy k-NN search and corresponding distance(s).
- similarity(query_animal: str, query_img_id: str, query_vecs: np.ndarray, map_dict: dict, db_index: AnnoyIndex,
  nr_knn: int) -> pd.DataFrame: Computes the similarity of query vectors to the database and returns the results
  as a DataFrame with detailed match information.

Notes:
- The functions are optimized for speed using efficient k-NN search methods and vectorized operations.
- The module assumes the use of an Annoy index and that descriptor vectors are in a suitable format.
"""

import pandas as pd
from annoy import AnnoyIndex
import numpy as np


def annoy_knn(database_index: AnnoyIndex, nr_knn: int, q_vec: np.ndarray) -> tuple:
    """
    Finds the k-nearest neighbors for a query vector using an Annoy index.

    Args:
    - database_index (AnnoyIndex): The Annoy index for the database of vectors.
    - nr_knn (int): The number of nearest neighbors to retrieve.
    - q_vec (np.ndarray): The query vector to compare against the database.

    Returns:
    - match_idx (int): Index/position of the nearest neighbor in the database.
    - match_distance (float): The distance between the query vector and the nearest neighbor.

    Raises:
    - ValueError: If the database index is invalid or None.
    - TypeError: If the query vector is not a numpy array.
    """

    if not isinstance(database_index, AnnoyIndex):
        raise ValueError("Database index must be a valid AnnoyIndex.")

    if not isinstance(q_vec, np.ndarray):
        raise TypeError("Query vector must be a numpy array.")

    if database_index is None:
        raise ValueError("Database index is invalid or None.")

    knn = database_index.get_nns_by_vector(q_vec, n=nr_knn, include_distances=True)
    match_idx = knn[0][0]
    match_distance = knn[1][0]

    return match_idx, match_distance

def similarity(query_img_id: str, query_vecs: list, map_dict: dict, db_index: AnnoyIndex, nr_knn: int, query_animal: str = "unknown") -> pd.DataFrame:
    """
    Computes similarity for a set of query vectors to the database and returns the results as a DataFrame.
    This can be used for FalseTagFinding if the query animal ID is known, or if the animal ID is unknown and
    needs to be predicted, using "unknown" as the default for query_animal.

    Args:
    - query_animal (str): The animal ID for the query image, default is "unknown" if the ID is not provided.
    - query_img_id (str): The image ID for the query image.
    - query_vecs (np.ndarray): Query descriptor vectors.
    - map_dict (dict): A dictionary mapping indexes to animal and image IDs. It must contain 'animal_ids' and 'img_ids'.
    - db_index (AnnoyIndex): The Annoy index for the database of descriptor vectors.
    - nr_knn (int): The number of nearest neighbors to retrieve.

    Returns:
    - pd.DataFrame: A DataFrame with match details for each query vector.

    Raises:
    - ValueError: If the query vectors or mapping dictionary are not in the expected format.
    - TypeError: If any argument is not of the expected type.
    """

    # Validate input
    if not isinstance(query_vecs, np.ndarray) or len(query_vecs) == 0:
        raise ValueError("Query vectors should be a non-empty list.")

    if not isinstance(map_dict, dict) or 'animal_ids' not in map_dict or 'img_ids' not in map_dict:
        raise ValueError("Invalid map dictionary format. Must contain 'animal_ids' and 'img_ids'.")

    if not isinstance(db_index, AnnoyIndex):
        raise ValueError("Database index must be a valid AnnoyIndex.")

    if not isinstance(nr_knn, int) or nr_knn <= 0:
        raise ValueError("The number of nearest neighbors (nr_knn) must be a positive integer.")

    similarity_data = []

    # Iterate through query descriptor vectors
    for q_vec_idx, q_vec in enumerate(query_vecs):

        match_idx, match_distance = annoy_knn(db_index, nr_knn, q_vec)

        row = {
            "q_vec_idx": q_vec_idx,
            "q_animal_id": query_animal,
            "q_img_id": query_img_id,
            "n1_idx": match_idx,
            "n1_animal_id": map_dict['animal_ids'][match_idx],
            "n1_img_id": map_dict['img_ids'][match_idx],
            "n1_distance": match_distance,
            "n1_is_correct": (map_dict['animal_ids'][match_idx] == query_animal),
        }

        similarity_data.append(row)

    # Convert list of rows to a DataFrame
    similarity_df = pd.DataFrame(similarity_data)

    return similarity_df


# def get_sim_df(closed_similarity_data, glob_db_annoy_index, glob_idx_map, open_similarity_data, q_animal_id,
#                q_desc_vecs, q_img_id, same_animal_idx, same_img_idx):
#     for q_vec_idx, q_vec in enumerate(q_desc_vecs):
#         # TRUE MATCH: remove only same image indexes
#         closed_match_idx, closed_match_distance = find_valid_knn(glob_db_annoy_index, q_vec, same_img_idx)
#         closed_row = {
#             "q_vec_idx": q_vec_idx,
#             "q_animal_id": q_animal_id,
#             "q_img_id": q_img_id,
#             "n1_idx": closed_match_idx,
#             "n1_animal_id": glob_idx_map['animal_ids'][closed_match_idx],
#             "n1_img_id": glob_idx_map['img_ids'][closed_match_idx],
#             "n1_distance": closed_match_distance,
#             "n1_is_correct": (glob_idx_map['animal_ids'][closed_match_idx] == q_animal_id),
#         }
#         closed_similarity_data.append(closed_row)
#
#         # FALSE MATCH: remove all same animal indexes
#         open_match_idx, open_match_distance = find_valid_knn(glob_db_annoy_index, q_vec, same_animal_idx)
#         open_row = {
#             "q_vec_idx": q_vec_idx,
#             "q_animal_id": q_animal_id,
#             "q_img_id": q_img_id,
#             "n1_idx": open_match_idx,
#             "n1_animal_id": glob_idx_map['animal_ids'][open_match_idx],
#             "n1_img_id": glob_idx_map['img_ids'][open_match_idx],
#             "n1_distance": open_match_distance,
#             "n1_is_correct": (glob_idx_map['animal_ids'][open_match_idx] == q_animal_id),
#         }
#         open_similarity_data.append(open_row)
#     closest_similarity_df = pd.DataFrame(closed_similarity_data)
#     open_similarity_df = pd.DataFrame(open_similarity_data)
#     return closest_similarity_df, open_similarity_df


def similarity_w_idx_filtering(db_annoy_index, idx_map, q_animal_id, q_desc_vecs, q_img_id, idx_to_filter):

    similarity_data = []

    for q_vec_idx, q_vec in enumerate(q_desc_vecs):

        match_idx, match_distance = find_valid_knn(db_annoy_index, q_vec, idx_to_filter)
        if match_idx is None:
            continue

        closed_row = {
            "q_vec_idx": q_vec_idx,
            "q_animal_id": q_animal_id,
            "q_img_id": q_img_id,
            "n1_idx": match_idx,
            "n1_animal_id": idx_map['animal_ids'][match_idx],
            "n1_img_id": idx_map['img_ids'][match_idx],
            "n1_distance": match_distance,
            "n1_is_correct": (idx_map['animal_ids'][match_idx] == q_animal_id),
        }
        similarity_data.append(closed_row)

    similarity_df = pd.DataFrame(similarity_data)

    return similarity_df


def find_valid_knn(annoy_index, query_vector, exclude_set, initial_k=1, max_attempts=50):
    k = initial_k
    while k <= max_attempts:
        neighbors = annoy_index.get_nns_by_vector(query_vector, k, include_distances=True)
        for idx in range(0,len(neighbors[0])):
            if neighbors[0][idx] not in exclude_set:
                return neighbors[0][idx], neighbors[1][idx]  # Return the first valid neighbor
        k += 1  # Increase k if all found neighbors are excluded
    return None, None
