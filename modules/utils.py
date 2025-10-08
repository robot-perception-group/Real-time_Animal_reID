import pickle
import os
import numpy as np
import pandas as pd
import shutil


def save_pickle(data_to_save: object, save_path: str) -> None:
    """
    Save data to a pickle file.

    This function saves any serializable Python object to a pickle file in the specified
    directory with the given filename.
    """

    try:
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
    except IOError as e:
        print(f"Error saving pickle file: {e}")
        raise


def load_pickle(path_to_pickle: str) -> object:
    """
    Load data from a pickle file.

    This function loads a Python object from a pickle file located at the specified path.
    """

    # Validate file path
    if not os.path.isfile(path_to_pickle):
        raise FileNotFoundError(f"The pickle file {path_to_pickle} does not exist.")

    try:
        with open(path_to_pickle, 'rb') as f:
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
        with pd.ExcelWriter(f'{save_path}/{filename}') as writer:
            df.to_excel(writer, index=False)
        print(f'SAVED: {filename}')
    except IOError as e:
        print(f"Error saving Excel file: {e}")
        raise


def idx_maps_from_lists(animal_id_list: list, img_id_list: list, desc_vecs_list: np.ndarray) -> dict:
    """
    Create a mapping from image IDs to animal IDs.

    This function creates a dictionary mapping image IDs to corresponding animal IDs
    using the provided lists of image and animal IDs.
    """

    # Validate input lists
    if len(animal_id_list) != len(img_id_list) != len(desc_vecs_list):
        raise ValueError("The length of animal_id_list, img_id_list and desc_vecs_list must be the same.")

    img_id_array = np.array(img_id_list)
    animal_id_array = np.array(animal_id_list)
    desc_vecs_array = np.array(desc_vecs_list)
    idx_map = {'img_ids': img_id_array, 'animal_ids': animal_id_array, 'db_desc_vecs': desc_vecs_array}
    return idx_map


def get_ids_from_filename(filename: str) -> tuple:
    """
    Extract image and animal IDs from a filename.

    This function assumes the filename follows a specific format:
    "<animal_id>_<something>_<img_id>.jpg".
    """

    split_filename = filename.split('_')
    if len(split_filename) < 3:
        raise ValueError(f"Filename {filename} does not follow the expected format.")

    img_id = split_filename[2]
    animal_id = split_filename[0]
    side_info = split_filename[1]

    return img_id, animal_id, side_info


def copy_and_rename_image(old_path: str, save_dir: str, new_name: str) -> None:
    """
    Move and rename an image file.

    This function moves the image from its current location to a new directory and renames it.
    """

    # Validate file and directory
    if not os.path.isfile(old_path):
        raise FileNotFoundError(f"The file {old_path} does not exist.")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)  # Create directory if it doesn't exist

    new_path = os.path.join(save_dir, new_name)

    try:
        shutil.copy(old_path, new_path)
    except IOError as e:
        print(f"Error moving and renaming file: {e}")
        raise
