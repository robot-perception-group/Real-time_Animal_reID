import os
import cv2
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from matplotlib import pyplot as plt
from modules.img_processing import preproc, load_img
from modules.utils import get_ids_from_filename
from modules.calc_similarity import annoy_knn


def map_keypoints_to_original(kps_resized, orig_img_shape, preproc_width=224):
    """
    Maps SIFT keypoints from preprocessed images back to original image coordinates.
    """
    orig_h, orig_w = orig_img_shape[:2]
    new_w = preproc_width
    new_h = int(new_w * orig_h / orig_w)

    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    kps_orig = []
    for kp in kps_resized:
        x, y = kp.pt
        new_kp = cv2.KeyPoint(
            float(x * scale_x),  # x
            float(y * scale_y),  # y
            float(kp.size * ((scale_x + scale_y) / 2)),  # size
            float(kp.angle),  # angle
            float(kp.response),  # response
            int(kp.octave),  # octave
            int(kp.class_id)  # class_id
        )
        kps_orig.append(new_kp)
    return kps_orig


def draw_keypoint_with_orientation(img, center, size, angle_deg, color, thickness=1):
    """
    Draws a SIFT-like keypoint showing position, scale, and orientation.
    """
    x, y = int(center[0]), int(center[1])
    radius = max(1, int(size / 2))
    cv2.circle(img, (x, y), radius, color, thickness, cv2.LINE_AA)

    # Orientation arrow
    angle_rad = np.deg2rad(angle_deg)
    x2 = int(x + radius * np.cos(angle_rad))
    y2 = int(y + radius * np.sin(angle_rad))
    cv2.arrowedLine(img, (x, y), (x2, y2), color, thickness, tipLength=0.3)


def show_matched_keypoints(q_img, db_img, df_matches, preproc_width, orig_q_img=None, orig_db_img=None):
    """
    Visualize matched keypoints with interactive slider and colormap, optionally over original images.

    Parameters:
        q_img: preprocessed query image (grayscale or color)
        db_img: preprocessed matched image
        df_matches: DataFrame with ['q_kp', 'n1_kp', 'n1_distance']
        orig_q_img: optional, original query image
        orig_db_img: optional, original matched image
    """

    # ==== MAP KEYPOINT ON ORIGINAL IMAGES ====
    if orig_q_img is not None and orig_db_img is not None:
        q_kps_orig = map_keypoints_to_original(df_matches['q_kp'].tolist(), orig_q_img.shape, preproc_width)
        db_kps_orig = map_keypoints_to_original(df_matches['n1_kp'].tolist(), orig_db_img.shape, preproc_width)
        q_img_color = cv2.cvtColor(orig_q_img, cv2.COLOR_GRAY2BGR) if len(orig_q_img.shape) == 2 else orig_q_img.copy()
        db_img_color = cv2.cvtColor(orig_db_img, cv2.COLOR_GRAY2BGR) if len(
            orig_db_img.shape) == 2 else orig_db_img.copy()
    else:
        q_kps_orig = df_matches['q_kp'].tolist()
        db_kps_orig = df_matches['n1_kp'].tolist()
        q_img_color = cv2.cvtColor(q_img, cv2.COLOR_GRAY2BGR) if len(q_img.shape) == 2 else q_img.copy()
        db_img_color = cv2.cvtColor(db_img, cv2.COLOR_GRAY2BGR) if len(db_img.shape) == 2 else db_img.copy()

    # ==== SORT MATCHES BY DISTANCE ====
    df_matches_sorted = df_matches.sort_values("n1_distance", ascending=True).reset_index(drop=True)

    # ==== PREPARE CANVAS ====
    h1, w1 = q_img_color.shape[:2]
    h2, w2 = db_img_color.shape[:2]
    canvas_height = max(h1, h2)
    canvas_width = w1 + w2
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:h1, :w1] = q_img_color
    canvas[:h2, w1:w1 + w2] = db_img_color

    # ==== EXTRACT COORDINATES & NORMALIZE DISTANCES ====
    q_pts = np.array([kp.pt for kp in q_kps_orig], dtype=np.float32)
    db_pts = np.array([kp.pt for kp in db_kps_orig], dtype=np.float32)
    distances = np.array(df_matches_sorted['n1_distance'], dtype=np.float32)
    norm_dist = (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)

    # ==== COLORMAP ====
    cmap = plt.get_cmap("RdYlGn_r")
    colors = (cmap(norm_dist)[:, :3] * 255).astype(np.uint8)

    # ==== OPENCV WINDOW ====
    window_name = "Match analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_trackbar(val):
        img_vis = canvas.copy()
        n_show = max(1, val)
        n_show = min(n_show, len(q_pts))

        # Draw matches
        for i in range(n_show):
            pt1 = tuple(np.round(q_pts[i]).astype(int))
            pt2 = (int(np.round(db_pts[i][0]) + w1), int(np.round(db_pts[i][1])))
            color = tuple(int(c) for c in colors[i][::-1])
            cv2.line(img_vis, pt1, pt2, color, 1, cv2.LINE_AA)
            draw_keypoint_with_orientation(img_vis, pt1, q_kps_orig[i].size, q_kps_orig[i].angle, color, 1)
            draw_keypoint_with_orientation(img_vis, pt2, db_kps_orig[i].size, db_kps_orig[i].angle, color, 1)

        # ---- Draw titles (relative font size) ----
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.6, canvas_width / 1500)  # scales with image width
        font_thickness = max(1, int(canvas_width / 1000))  # scales thickness similarly
        # margin_y = int(30 * font_scale)

        # text color
        color = (0, 0, 255)

        # left and right titles
        cv2.putText(img_vis, "Query Image", (int(10 * font_scale), int(30 * font_scale)),
                    font, font_scale, color, font_thickness, cv2.LINE_AA)
        cv2.putText(img_vis, "Database Image", (w1 + int(10 * font_scale), int(30 * font_scale)),
                    font, font_scale, color, font_thickness, cv2.LINE_AA)

        cv2.imshow(window_name, img_vis)

    cv2.createTrackbar("Num Matches", window_name, 10, len(df_matches_sorted), on_trackbar)
    on_trackbar(10)

    print("Close the window by clicking the ❌ button.")
    while True:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(50)
    cv2.destroyAllWindows()


def verify_match_by_eye(query_img_path, predicted_id, database_path, nr_kps, preproc_width, on_original_imgs):

    #############################################
    #  BUILD DATABASE WITH ONLY THE PREDICTED ID
    #############################################

    # ==== PREPARE =====
    desc_vec_dim = 128
    dist_metric = 'euclidean'
    sift_extractor = cv2.SIFT.create(nfeatures=nr_kps)

    img_id_list = []
    animal_id_list = []
    desc_vecs_list = []
    kps_list = []

    db_index = AnnoyIndex(desc_vec_dim, dist_metric)

    # ===== ITERATE DATABASE IMAGES ====
    for filename in os.listdir(database_path):
        filepath = os.path.join(database_path, filename)
        if os.path.isfile(filepath):

            # get curr ids
            img_id, animal_id, _ = get_ids_from_filename(filename)

            # FILTER ON PREDICTED ID
            if animal_id == predicted_id:

                # preprocess img
                curr_img_path = os.path.join(database_path, filename)
                img = preproc(curr_img_path)

                # extract bobox_img features
                kps, desc_vecs = sift_extractor.detectAndCompute(img, None)
                if len(kps) == 0:
                    print(f"Skipping {filename}: No keypoints detected.")
                    continue

                nr_kps = len(desc_vecs)
                desc_vecs_list += desc_vecs.tolist()
                img_id_list += [img_id] * nr_kps
                animal_id_list += [animal_id] * nr_kps
                kps_list += kps

    # ===== BUILD THE INDEX ====
    pos_in_index = 0

    # fill with vectors
    for desc_vec in desc_vecs_list:
        db_index.add_item(pos_in_index, desc_vec)
        pos_in_index += 1

    # Create ID map
    n = len(animal_id_list)
    if not (len(img_id_list) == len(desc_vecs_list) == len(kps_list) == n):
        raise ValueError("All input lists must have the same length.")
    idx_map = {
        'img_ids': np.array(img_id_list),
        'animal_ids': np.array(animal_id_list),
        'db_desc_vecs': np.array(desc_vecs_list),
        'db_kps': kps_list
    }

    # build
    db_index.build(10)

    #############################################
    #  FIND BEST MATCH IN THIS SPECIFIC DATABASE
    #############################################

    # ==== PREPROC ====
    q_filename = os.path.basename(query_img_path)
    q_img = preproc(query_img_path)

    # ==== EXTRACT ====
    q_kps, q_desc_vecs = sift_extractor.detectAndCompute(q_img, None)
    if len(q_kps) == 0:
        exit('No keypoints detected.')


    # ==== SIMILARITY ====
    similarity_data = []

    # Iterate query desc vecs
    for q_vec_idx, q_vec in enumerate(q_desc_vecs):

        match_idx, match_distance = annoy_knn(db_index, 1, q_vec)

        row = {
            "q_vec_idx": q_vec_idx,
            "q_kp": q_kps[q_vec_idx],
            "q_animal_id": "unknown",
            "q_img_id": q_filename,
            "n1_idx": match_idx,
            "n1_kp": idx_map['db_kps'][match_idx],
            "n1_animal_id": idx_map['animal_ids'][match_idx],
            "n1_img_id": idx_map['img_ids'][match_idx],
            "n1_distance": match_distance,
            "n1_is_correct": (idx_map['animal_ids'][match_idx] == "unknown"),
        }

        similarity_data.append(row)

    # create dataframe
    similarity_df = pd.DataFrame(similarity_data)

    # ==== FIND MOST FREQUENTLY RETRIEVED IMAGE ====
    most_freq_db_image = similarity_df['n1_img_id'].mode()[0]
    most_freq_db_image_path = os.path.join(database_path, most_freq_db_image)
    filtered_df = similarity_df[similarity_df['n1_img_id'] == most_freq_db_image]

    #############################################
    #   PROCESS BEST MATCHING DATABASE IMAGE
    #############################################

    # ==== PREPROC ====
    db_img = preproc(os.path.join(database_path, most_freq_db_image))

    # ==== EXTRACT ====
    db_kps, db_desc_vecs = sift_extractor.detectAndCompute(db_img, None)
    if len(db_kps) == 0:
        exit('No keypoints detected.')

    #############################################
    #               VISUALIZE
    #############################################

    # ==== LOAD ORIGINAL IMAGES ====
    if on_original_imgs:
        orig_q_img = load_img(query_img_path)
        orig_n1_img = load_img(most_freq_db_image_path)
    else:
        orig_q_img = None
        orig_n1_img = None

    # ==== VISUALIZE ====
    show_matched_keypoints(q_img, db_img, filtered_df, preproc_width, orig_q_img=orig_q_img, orig_db_img=orig_n1_img)

verify_match_by_eye(
    query_img_path="/home/azabo/Desktop/RAPID/data/demo/example_query/z42_left_img-0000787.jpg",
    predicted_id="z42",
    database_path="/home/azabo/Desktop/RAPID/data/demo/example_database",
    nr_kps=150,
    preproc_width=224,
    on_original_imgs=False)