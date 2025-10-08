from tqdm import tqdm
import os
import pickle
from annoy import AnnoyIndex
from scipy.interpolate import interp1d
from modules.img_processing import preproc
from modules.utils import save_pickle, load_pickle, idx_maps_from_lists, get_ids_from_filename


def save_kde_evaluator(kde_obj, save_path):
    if kde_obj is None:
        return
    data = {
        "support": kde_obj.support,
        "density": kde_obj.density,
    }
    with open(save_path, "wb") as f:
        pickle.dump(data, f)


def load_kde_as_evaluator(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    evaluator = interp1d(
        data["support"], data["density"],
        bounds_error=False,
        fill_value=0.0
    )
    return evaluator


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
    desc_vecs_list = []
    kps_list = []

    db_index = AnnoyIndex(dim, metric)

    for filename in tqdm(os.listdir(image_dir), desc="Building the database: "):

        filepath = os.path.join(image_dir, filename)
        if os.path.isfile(filepath):  # Check if it's a file

            # get curr ids
            img_id, animal_id, _ = get_ids_from_filename(filename)

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

            nr_kps = len(desc_vecs)
            desc_vecs_list += desc_vecs.tolist()
            img_id_list += [img_id] * nr_kps
            animal_id_list += [animal_id] * nr_kps
            kps_list += kps

    #############################################
    #      BUILD THE INDEX & CREATE MAPPING
    #############################################

    pos_in_index = 0

    for desc_vec in desc_vecs_list:
        db_index.add_item(pos_in_index, desc_vec)
        # img_id_list.append(img_id)
        # animal_id_list.append(animal_id)
        pos_in_index += 1

    # Create ID map
    idx_map = idx_maps_from_lists(animal_id_list, img_id_list, desc_vecs_list)

    # Build the Annoy index with 10 trees
    db_index.build(10)

    # Save Annoy index and ID mapping
    db_index_path = f'{save_dir}/db_index.ann'
    db_index.save(db_index_path)
    idx_map_path = f'{save_dir}/db_idx_map.pkl'
    save_pickle(data_to_save=idx_map, save_path=idx_map_path)

    return db_index, db_index_path, idx_map, idx_map_path


def extend_db(new_desc_vecs, new_animal_id, new_img_id, db_index_path, db_position_map, db_position_map_path, metric,
              dim):
    # Validate metric
    if metric not in ['angular', 'euclidean', 'manhattan']:
        raise ValueError(f"Unsupported metric: {metric}. Supported metrics are 'angular', 'euclidean', 'manhattan'.")

    # Load Annoy index and ID map
    db_index = AnnoyIndex(dim, metric)

    nr_new_kps = len(new_desc_vecs)

    # Collect new data to extend the db with
    new_desc_vecs = new_desc_vecs.tolist()
    new_img_ids = [new_img_id] * nr_new_kps
    new_animal_ids = [new_animal_id] * nr_new_kps

    # Collect existing data from db
    db_desc_vecs = db_position_map['db_desc_vecs'].tolist()
    db_img_ids = db_position_map['img_ids'].tolist()
    db_animal_ids = db_position_map['animal_ids'].tolist()

    # Extend the db with the new data
    db_desc_vecs += new_desc_vecs
    db_img_ids += new_img_ids
    db_animal_ids += new_animal_ids

    # Fill the db index with vectors
    pos_in_index = 0
    for desc_vec in db_desc_vecs:
        db_index.add_item(pos_in_index, desc_vec)
        pos_in_index += 1

    # Build the Annoy index with 10 trees
    db_index.build(10)

    # Create ID map
    idx_map = idx_maps_from_lists(db_animal_ids, db_img_ids, db_desc_vecs)

    # Save extended database index, ID mapping and paths - not necessary in each iteration!
    db_index.save(db_index_path)
    save_pickle(data_to_save=idx_map, save_path=db_position_map_path)

    return db_index, db_index_path, idx_map, db_position_map_path