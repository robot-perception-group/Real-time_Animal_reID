import os
import cv2

from tqdm import tqdm

from modules.img_processing import preproc
from modules.inference import inference_w_confidence_scores, get_weight_ratio, get_conf_score, open_closed_behaviour

from modules.utils import get_ids_from_filename
from modules.visualize import false_tag_finder_plot, false_tag_finder_plot
from modules.database import build_db, load_db
from modules.calc_similarity import similarity_w_idx_filtering


def false_tag_finder(db_imgs, nr_kps, topN_ids_match):

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

    # BUILD DATABASE
    db_annoy_index, idx_map, db_kps = build_db(image_dir=db_imgs,
                                               save_dir=save_dir,
                                               feat_extractor=sift_extractor,
                                               dim=desc_vec_dim,
                                               metric=dist_metric)

    # LOAD EXISTING DATABASE
    # db_annoy_index_path = save_dir + '/db_index.ann'
    # idx_map_path = save_dir + '/db_idx_map.pkl'
    # db_annoy_index, idx_map = load_db(dim=desc_vec_dim,
    #                                   ann_path=db_annoy_index_path,
    #                                   map_path=idx_map_path,
    #                                   metric=dist_metric)

    # GET FITTED CURVES FOR CONFIDENCE SCORES
    correct_label_func, false_label_func = open_closed_behaviour(save_dir=save_dir,
                                                              db_imgs=db_imgs,
                                                              topN_ids_match=topN_ids_match,
                                                              sift_extractor=sift_extractor,
                                                              idx_map=idx_map,
                                                              db_annoy_index=db_annoy_index)

    ################################################################################
    #                       ANALYZE GROUND TRUTH LABELS
    ################################################################################

    false_match_counter = 1

    # ITERATE GROUND-TRUTH IMAGES
    for curr_img_filename in tqdm(os.listdir(db_imgs), desc="Searching for wrong labels: "):

        #############################
        # Module-1: PreProc
        #############################
        curr_img_path = os.path.join(db_imgs, curr_img_filename)
        q_img_id, q_animal_id = get_ids_from_filename(curr_img_filename)
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
        glob_sim_df = similarity_w_idx_filtering(db_annoy_index=db_annoy_index,
                                                 idx_map=idx_map,
                                                 q_animal_id=q_animal_id,
                                                 q_desc_vecs=q_desc_vecs,
                                                 q_img_id=q_img_id,
                                                 idx_to_filter=same_img_idx)

        #############################
        # Module-4: Inference
        #############################
        glob_pred, glob_topN_ids = inference_w_confidence_scores(similarity_df=glob_sim_df,
                                                                 n=topN_ids_match)

        #############################
        # Module-5: Evaluate
        #############################
        ratio = get_weight_ratio(top_ids_df=glob_topN_ids)

        conf_score = get_conf_score(curve_false_match=false_label_func,
                                    curve_true_match=correct_label_func,
                                    ratio=ratio)

        if glob_pred != q_animal_id:
            false_tag_finder_plot(save_dir=wrong_matches,
                                  glob_sim_df=glob_sim_df,
                                  glob_topN_ids=glob_topN_ids,
                                  q_img_id=q_img_id,
                                  q_animal_id=q_animal_id,
                                  glob_pred=glob_pred,
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
                 topN_ids_match=10)
