from tqdm import tqdm
import os
import numpy as np
from annoy import AnnoyIndex
from modules.img_processing import preproc
from modules.utils import save_pickle, load_pickle, idx_maps_from_lists, get_ids_from_filename


def load_db(dim, ann_path, map_path, metric):
    """ 
    Load existing Annoy index database 
    
    Args:
        - dim: dimension of the desc_vecs
        - ann_path: path to the Annoy index file
        - map_path: path to the ID mapping file
        - metric: distance metric to use for building the Annoy index
        
    Returns:
        - index: Annoy index object, for later vector search
        - id_map: ID mapping dictionary to connect image IDs and animal IDs with database desc_vecs
    """

    # Validate metric
    if metric not in ['angular', 'euclidean', 'manhattan']:
        raise ValueError(f"Unsupported metric: {metric}. Supported metrics are 'angular', 'euclidean', 'manhattan'.")

    # Load Annoy index and ID map
    index = AnnoyIndex(dim, metric)
    try:
        index.load(ann_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Annoy index file not found at {ann_path}.")

    try:
        id_map = load_pickle(map_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"ID mapping file not found at {map_path}.")

    return index, id_map

def build_db(image_dir, save_dir, feat_extractor, dim, metric):
    """ 
    Build database from images in image_dir and save it to save_dir 
    
    Args:
        - image_dir: directory containing database images for building the index
        - save_dir: directory to save the Annoy index and ID mapping
        - feat_extractor: feature extractor object for extracting features from images, SIFT in our case
        - dim: dimension of the desc_vecs
        - metric: distance metric to use for building the Annoy index
        
    Returns:
        - db_index: Annoy index object, for later vector search
        - idx_map: ID mapping dictionary to connect image IDs and animal IDs with database desc_vecs  
    """

    # Validate metric
    if metric not in ['angular', 'euclidean', 'manhattan']:
        raise ValueError(f"Unsupported metric: {metric}. Supported metrics are 'angular', 'euclidean', 'manhattan'.")

    # Validate image directory
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"The image directory {image_dir} does not exist.")

    img_id_list = []
    animal_id_list = []
    kps_list = []

    db_index = AnnoyIndex(dim, metric)
    pos_in_index = 0

    for filename in tqdm(os.listdir(image_dir), desc="Building the database: "):

        # get curr ids
        img_id, animal_id = get_ids_from_filename(filename)

        # preprocess img
        curr_img_path = os.path.join(image_dir, filename)
        img = preproc(curr_img_path)

        # extract bobox_img features
        kps, desc_vecs = feat_extractor.detectAndCompute(img, None)
        if len(kps) == 0:
            print(f"Skipping {filename}: No keypoints detected.")
            continue

        # # rootSIFT
        # if method == "rSIFT":
        #     # Apply L1 normalization
        #     desc_vecs /= (desc_vecs.sum(axis=1, keepdims=True) + 1e-7)
        #     # Apply element-wise square root (Hellinger normalization)
        #     desc_vecs = np.sqrt(desc_vecs)

        kps_list += kps

        for desc_vec in desc_vecs:
            db_index.add_item(pos_in_index, desc_vec)
            img_id_list.append(img_id)
            animal_id_list.append(animal_id)
            pos_in_index += 1

    # Create ID map
    idx_map = idx_maps_from_lists(animal_id_list, img_id_list)

    # Build the Annoy index with 10 trees
    db_index.build(10)

    # Save Annoy index and ID mapping
    db_index.save(f'{save_dir}/db_index.ann')
    save_pickle(data_to_save=idx_map, save_path=save_dir, filename=f'db_idx_map')

    return db_index, idx_map, kps_list
