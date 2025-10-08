import os
import cv2
import numpy as np

from tqdm import tqdm

from modules.img_processing import preproc
from modules.inference import inference_w_confidence_scores, get_weight_ratio, get_conf_score, open_closed_behaviour, KDE_curves

from modules.utils import get_ids_from_filename
from modules.visualize import false_tag_finder_plot, false_tag_finder_plot
from modules.database import build_db, load_db
from modules.calc_similarity import similarity_w_idx_filtering


def false_tag_finder(db_imgs, nr_kps, topN_ids_match, db_annoy_index_path,
                     idx_map_path, closed_correct_ratios_path, closed_false_ratios_path,
                     open_ratios_path):

    ################################################################################
    #                               PREPARATION
    ################################################################################

    # INIT SIFT AND ANNOY
    desc_vec_dim = 128
    dist_metric = 'euclidean'
    sift_extractor = cv2.SIFT.create(nfeatures=nr_kps)


    # CREATE SAVE DIRS
    parent_dir = os.path.dirname(db_imgs)
    save_dir = parent_dir + '/saved_FalseTagFinder'
    wrong_matches = save_dir + '/potential_false_labels'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(wrong_matches, exist_ok=True)

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
    #                       ANALYZE GROUND TRUTH LABELS
    ################################################################################

    false_match_counter = 1

    # ITERATE GROUND-TRUTH IMAGES
    for curr_img_filename in tqdm(os.listdir(db_imgs), desc="Searching for possibly wrong labels: "):

        #############################
        # Module-1: PreProc
        #############################
        curr_img_path = os.path.join(db_imgs, curr_img_filename)
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
        same_img_idx = [i for i, val in enumerate(idx_map['img_ids']) if val in q_img_id]  # db position w current image
        similarity_df = similarity_w_idx_filtering(
            db_annoy_index,
            idx_map,
            q_animal_id,
            q_desc_vecs,
            q_img_id,
            same_img_idx,
            k_for_knn=10,
            scenario="closed")

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

        if pred != q_animal_id:
            false_tag_finder_plot(save_dir=wrong_matches,
                                  glob_sim_df=similarity_df,
                                  glob_topN_ids=topN_ids,
                                  q_img_id=q_img_id,
                                  q_animal_id=q_animal_id,
                                  glob_pred=pred,
                                  conf_score=conf_score,
                                  ratio=ratio,
                                  db_imgs=db_imgs,
                                  filename=curr_img_filename,
                                  counter=false_match_counter)
            false_match_counter += 1

    print(f'Mismatch: {false_match_counter - 1}')

#*************************************************************************
#*************************************************************************

ground_truth_database = 'data/demo/example_database'
false_tag_finder(db_imgs=ground_truth_database,
                 nr_kps=150,
                 topN_ids_match=10,
                 db_annoy_index_path="",
                 idx_map_path="",
                 closed_correct_ratios_path="",
                 closed_false_ratios_path="",
                 open_ratios_path="")