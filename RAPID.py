import csv
import os
import cv2
import numpy as np

from tqdm import tqdm

from modules.img_processing import preproc
from modules.calc_similarity import similarity
from modules.inference import inference_w_confidence_scores, get_weight_ratio, get_conf_score, open_closed_behaviour, KDE_curves

from modules.database import build_db, load_db, extend_db, load_kde_as_evaluator
from modules.utils import get_ids_from_filename


def predict_id(q_imgs_dir, db_imgs, nr_kps, topN_ids_match, conf_threshold, db_annoy_index_path, idx_map_path, closed_correct_ratios_path,
               closed_false_ratios_path, open_ratios_path, extend_db_while_proc=False):
    ################################################################################
    #                               PREPARATION
    ################################################################################

    # INIT SIFT AND ANNOY
    desc_vec_dim = 128
    dist_metric = 'euclidean'
    sift_extractor = cv2.SIFT.create(nfeatures=nr_kps)

    # PREPARE SAVING
    parent_dir = os.path.dirname(db_imgs)
    save_dir = parent_dir + '/saved_RAPID'
    os.makedirs(save_dir, exist_ok=True)
    prediction_results = []

    # METRICS
    q_img_counter = 0  # count query images

    if db_annoy_index_path:
        # LOAD EXISTING DATABASE
        db_annoy_index, idx_map = load_db(dim=desc_vec_dim,
                                          ann_path=db_annoy_index_path,
                                          map_path=idx_map_path,
                                          metric=dist_metric)
    else:
        # BUILD DATABASE
        db_annoy_index, db_annoy_index_path, idx_map, idx_map_path = build_db(image_dir=db_imgs,
                                                                              save_dir=save_dir,
                                                                              feat_extractor=sift_extractor,
                                                                              dim=desc_vec_dim,
                                                                              metric=dist_metric)

    # ANALYZE DB FOR CONFIDENCE SCORES
    if closed_correct_ratios_path and open_ratios_path:
        closed_correct_ratios = np.load(closed_correct_ratios_path)
        closed_false_ratios = np.load(closed_false_ratios_path)
        open_ratios = np.load(open_ratios_path)
        closed_correct_match_func, closed_false_match_func, open_match_func  = KDE_curves(closed_correct_ratios, closed_false_ratios, open_ratios, save_dir)
    else:
        closed_correct_match_func, closed_false_match_func, open_match_func = open_closed_behaviour(save_dir, db_imgs, topN_ids_match, sift_extractor,idx_map, db_annoy_index, n=500)

    ################################################################################
    #                               ANALYZE QUERY IMAGES
    ################################################################################

    # ITERATE QUERY IMAGES
    for curr_img_filename in tqdm(os.listdir(q_imgs_dir), desc="Processing query images: "):

        #############################
        # Module-1: PreProc
        #############################
        curr_img_path = os.path.join(q_imgs_dir, curr_img_filename)
        q_img_id, q_animal_id, _ = get_ids_from_filename(curr_img_filename)
        q_img = preproc(curr_img_path)

        #############################
        # Module-2: Extract
        #############################
        q_kps, q_desc_vecs = sift_extractor.detectAndCompute(q_img, None)
        if len(q_kps) == 0:
            print(curr_img_filename)
            continue

        #############################
        # Module-3: Similarity
        #############################
        similarity_df = similarity(query_img_id=curr_img_filename,
                                   query_vecs=q_desc_vecs,
                                   map_dict=idx_map,
                                   db_index=db_annoy_index,
                                   nr_knn=1)

        #############################
        # Module-4: Inference
        #############################
        pred, topN_ids = inference_w_confidence_scores(similarity_df=similarity_df,
                                                       n=topN_ids_match)

        #############################
        # Module-5: Evaluate
        #############################
        ratio = get_weight_ratio(top_ids_df=topN_ids)

        conf_score = get_conf_score(curve_correct_match=closed_correct_match_func,
                                    curve_open_match=open_match_func,
                                    ratio=ratio)

        prediction_results.append([q_img_id, q_animal_id, pred, conf_score, q_animal_id==pred])

        # IF EXTEND DATABASE
        if extend_db_while_proc and (conf_score >= conf_threshold):
            db_annoy_index, db_annoy_index_path, idx_map, idx_map_path = extend_db(new_desc_vecs=q_desc_vecs,
                                                                                   new_animal_id=pred,
                                                                                   new_img_id=curr_img_filename,
                                                                                   db_index_path=db_annoy_index_path,
                                                                                   db_position_map=idx_map,
                                                                                   db_position_map_path=idx_map_path,
                                                                                   metric=dist_metric,
                                                                                   dim=desc_vec_dim)

        q_img_counter += 1

    ################################################################################
    #                                   BYE
    ################################################################################

    with open(f"{save_dir}/prediction_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_img", "provided_ID", "predicted_ID", "confidence_score", "IDs_matching"])
        writer.writerows(prediction_results)

    print(f'----------\n'
          f'DATASET INFO\n'
          f'----------\n'
          f"GLOBAL DATABASE: {len(list(set(idx_map['animal_ids'])))} IDs, {len(list(set(idx_map['img_ids'])))} boboxes, {len(idx_map['animal_ids'])} desc_vecs\n"
          f'QUERY: {q_img_counter} boboxes\n')


#*************************************************************************
#*************************************************************************

database_path = 'data/demo/example_database'
query_path = 'data/demo/example_query'
predict_id(q_imgs_dir=query_path,
           db_imgs=database_path,
           nr_kps=150,
           topN_ids_match=10,
           conf_threshold=0.5,
           db_annoy_index_path="",
           idx_map_path="",
           closed_correct_ratios_path="",
           closed_false_ratios_path="",
           open_ratios_path="",
           extend_db_while_proc=False)