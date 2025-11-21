import os
import csv
import pickle
import numpy as np
import pandas as pd


def save_pickle(data_to_save: object, save_path: str) -> None:
    """
    Save data to a pickle file.

    This function saves any serializable Python object to a pickle file in the specified
    directory with the given filename.
    """

    try:
        with open(save_path, "wb") as f:
            pickle.dump(data_to_save, f)
    except IOError as e:
        print(f"Error saving pickle file: {e}")
        raise


def load_pickle(path_to_pickle: str) -> object:
    """
    Load data from a pickle file.

    This function loads a Python object from a pickle file located at the specified
    path.
    """

    # Validate file path
    if not os.path.isfile(path_to_pickle):
        raise FileNotFoundError(f"The pickle file {path_to_pickle} does not exist.")

    try:
        with open(path_to_pickle, "rb") as f:
            loaded_data = pickle.load(f)
    except IOError as e:
        print(f"Error loading pickle file: {e}")
        raise

    return loaded_data


def save_df_to_xlsx(df: pd.DataFrame, filename: str, save_path: str) -> None:
    """
    Save a pandas DataFrame to an Excel file.

    This function saves a pandas DataFrame to an Excel file in the specified directory
    with the given filename.
    """

    # Validate save path
    if not os.path.isdir(save_path):
        raise FileNotFoundError(f"The save directory {save_path} does not exist.")

    try:
        with pd.ExcelWriter(f"{save_path}/{filename}") as writer:
            df.to_excel(writer, index=False)
        print(f"SAVED: {filename}")
    except IOError as e:
        print(f"Error saving Excel file: {e}")
        raise


def idx_maps_from_lists(
    animal_id_list: list, img_id_list: list, desc_vecs_list: np.ndarray
) -> dict:
    """
    Create a mapping from image IDs to animal IDs.

    This function creates a dictionary mapping image IDs to corresponding animal IDs
    using the provided lists of image and animal IDs.
    """

    # Validate input lists
    if len(animal_id_list) != len(img_id_list) != len(desc_vecs_list):
        raise ValueError(
            "The length of animal_id_list, img_id_list and desc_vecs_list must be the "
            "same."
        )

    img_id_array = np.array(img_id_list)
    animal_id_array = np.array(animal_id_list)
    desc_vecs_array = np.array(desc_vecs_list)
    idx_map = {
        "img_ids": img_id_array,
        "animal_ids": animal_id_array,
        "db_desc_vecs": desc_vecs_array,
    }
    return idx_map


def get_ids_from_filename(filename: str) -> tuple:
    """
    Extract image and animal IDs from a filename.

    This function assumes the filename follows a specific format:
    "<animal_id>_<something>_<img_id>.jpg".
    """

    split_filename = filename.split("_")
    if len(split_filename) < 3:
        raise ValueError(f"Filename {filename} does not follow the expected format.")

    img_id = filename
    animal_id = split_filename[0]
    side_info = split_filename[1]

    return img_id, animal_id, side_info


def store_prediction_result(
    prediction_results, q_img_id, q_animal_id, pred, conf_score, topN_ids, topN=5
):
    """
    Appends a row to prediction_results with the format:
    [q_img_id, q_animal_id, pred, conf_score, correct_bool, '', n1_id1, weight1,
    n1_id2, weight2, ...]

    Parameters:
        prediction_results (list): List to append the row to.
        q_img_id: Query image ID.
        q_animal_id: True animal ID.
        pred: Predicted animal ID.
        conf_score: Confidence score.
        topN_ids (pd.DataFrame): DataFrame containing 'n1_animal_id' and 'weighted_ids'.
        max_rows (int): Maximum number of rows to take from topN_ids (default 5).
    """
    # Base row
    row_values = [q_img_id, q_animal_id, pred, conf_score, q_animal_id == pred, ""]

    # Get column indices
    id_col_idx = topN_ids.columns.get_loc("n1_animal_id")
    weight_col_idx = topN_ids.columns.get_loc("weighted_ids")

    # Slice topN_ids safely (handle fewer than max_rows)
    top_rows = topN_ids.iloc[:topN, [id_col_idx, weight_col_idx]].to_numpy()

    # Flatten alternating ID/weight pairs
    row_values += [val for pair in top_rows for val in pair]

    # Append to prediction_results
    prediction_results.append(row_values)


def save_prediction_results_csv(prediction_results, save_dir, topN=5):
    """
    Saves prediction results to a CSV file with alternating pred/weight columns.

    Parameters:
        prediction_results (list of lists): Rows to write.
        save_dir (str): Directory (and filename) to save the CSV, e.g.,
        "output/prediction_results.csv".
        max_topN (int): Maximum number of top predictions per row (default 5).
    """

    # Convert all elements to strings
    result_as_str = [[str(x) for x in sublist] for sublist in prediction_results]
    flat_bools = [
        x for sublist in result_as_str for x in sublist if x in ("True", "False")
    ]

    true_count = flat_bools.count("True")
    false_count = flat_bools.count("False")
    n_query = true_count + false_count
    top1_acc = true_count / n_query if (true_count + false_count) > 0 else 0

    # Build the header
    header = [
        "query_img",
        "provided_ID",
        "predicted_ID",
        "confidence_score",
        "IDs_matching",
        "",
    ]
    for i in range(1, topN + 1):
        header += [f"pred{i}", f"weight{i}"]

    # Write CSV
    with open(
        save_dir + f"/prediction_results_{int(top1_acc * 100)}%_{n_query}query.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(prediction_results)

    print(f"Prediction results saved to {save_dir}/prediction_results.csv")


# Optional path cleaner
def normalize_optional_path(x: str):
    if not x:
        return None

    cleaned = x.strip(" '\"")
    if cleaned in ("", "."):
        return None

    return os.path.normpath(cleaned)
