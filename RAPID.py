import os
import cv2

from tqdm import tqdm

from modules.img_processing import preproc
from modules.calc_similarity import similarity
from modules.inference import inference_w_confidence_scores, get_weight_ratio, get_conf_score, open_closed_behaviour

from modules.database import build_db, load_db
from modules.utils import copy_and_rename_image


def predict_id(q_imgs_dir, db_imgs, nr_kps, topN_ids_match, conf_threshold):
    ################################################################################
    #                               PREPARATION
    ################################################################################

    # INIT SIFT AND ANNOY
    desc_vec_dim = 128
    dist_metric = 'euclidean'
    sift_extractor = cv2.SIFT.create(nfeatures=nr_kps)

    # CREATE SAVE DIRS
    parent_dir = os.path.dirname(db_imgs)
    save_dir = parent_dir + '/saved_RAPID'
    os.makedirs(save_dir, exist_ok=True)

    # METRICS
    q_img_counter = 0  # count query images

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
    correct_label_func, false_label_func = open_closed_behaviour(save_dir, db_imgs, topN_ids_match, sift_extractor,
                                                                 idx_map, db_annoy_index)

    ################################################################################
    #                               ANALYZE QUERY IMAGES
    ################################################################################

    # ITERATE QUERY IMAGES
    for curr_img_filename in tqdm(os.listdir(q_imgs_dir), desc="Processing query images: "):

        #############################
        # Module-1: PreProc
        #############################
        curr_img_path = os.path.join(q_imgs_dir, curr_img_filename)
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
        # similarity_df = predict_id_similarity(query_img_id=curr_img_filename,
        #                                       query_vecs=q_desc_vecs,
        #                                       map_dict=idx_map,
        #                                       db_index=db_annoy_index,
        #                                       nr_knn=1)
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

        conf_score = get_conf_score(curve_false_match=false_label_func,
                                    curve_true_match=correct_label_func,
                                    ratio=ratio)

        # RENAME AND MOVE FILE IF CONFIDENT PREDICTION
        if conf_score >= conf_threshold:
            copy_and_rename_image(old_path=curr_img_path,
                                  save_dir=save_dir,
                                  new_name=f'PRED-{pred}_{curr_img_filename}')

        q_img_counter += 1

    ################################################################################
    #                                   BYE
    ################################################################################
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
           conf_threshold=0.5)
