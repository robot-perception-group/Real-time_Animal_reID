import os
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels import api as sm
from tqdm import tqdm

from modules.img_processing import preproc
from modules.calc_similarity import similarity_w_idx_filtering
from modules.utils import get_ids_from_filename
from modules.database import save_kde_evaluator, load_kde_as_evaluator


def inference_w_confidence_scores(similarity_df: pd.DataFrame, n: int) -> (str, pd.DataFrame):
    """
    Computes the most frequent animal ID using weighted similarity scores based on the inverse of distance,
    and returns the predicted label with its top weighted IDs.

    Args:
    - similarity_df (pd.DataFrame): DataFrame containing similarity data, must include 'n1_animal_id' and 'n1_distance'.
    - n (int): The number of top weighted animal IDs to return.

    Returns:
    - prediction (str): The predicted label for query.
    - top_n (pd.Dataframe): DataFrame with the top n weighted animal IDs.

    Raises:
    - ValueError: If `similarity_df` does not contain the required columns or if `n` is non-positive.
    """

    if not isinstance(similarity_df, pd.DataFrame):
        raise TypeError("The similarity_df must be a pandas DataFrame.")

    if not all(col in similarity_df.columns for col in ['n1_animal_id', 'n1_distance']):
        raise ValueError("The similarity DataFrame must contain columns 'n1_animal_id' and 'n1_distance'.")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("The parameter 'n' must be a positive integer.")

    # Add reciprocal distance column for weighted similarity scores
    similarity_df['reciprocal_distance'] = 1 / similarity_df['n1_distance']

    # Create weighted_ids DataFrame: sum of reciprocal distances per 'n1_animal_id'
    weighted_ids = similarity_df.groupby('n1_animal_id')['reciprocal_distance'].sum().reset_index()

    # Add frequency column: count how many times each animal ID appears in similarity_df
    freq = similarity_df['n1_animal_id'].value_counts().reset_index()
    freq.columns = ['n1_animal_id', 'freq']  # Rename columns to match the `weighted_ids` DataFrame

    # Merge the frequency information with the weighted_ids DataFrame
    weighted_ids = weighted_ids.merge(freq, on='n1_animal_id', how='left')

    # Calculate the weighted frequency
    total_sum = similarity_df['reciprocal_distance'].sum()
    weighted_ids['weighted_ids'] = weighted_ids['reciprocal_distance'] / total_sum

    # Sort weighted_ids by 'weighted_ids' in descending order
    weighted_ids.sort_values('weighted_ids', ascending=False, inplace=True)

    # Get the top n animal IDs based on weighted similarity
    top_n = weighted_ids.head(n)
    prediction = top_n.head(1)['n1_animal_id'].iloc[0]

    return prediction, top_n
def inference_w_confidence_scores(similarity_df: pd.DataFrame, n: int) -> (str, pd.DataFrame):
    """
    Computes the most frequent animal ID using weighted similarity scores based on the inverse of distance,
    and returns the predicted label with its top weighted IDs.

    Args:
    - similarity_df (pd.DataFrame): DataFrame containing similarity data, must include 'n1_animal_id' and 'n1_distance'.
    - n (int): The number of top weighted animal IDs to return.

    Returns:
    - prediction (str): The predicted label for query.
    - top_n (pd.Dataframe): DataFrame with the top n weighted animal IDs.

    Raises:
    - ValueError: If `similarity_df` does not contain the required columns or if `n` is non-positive.
    """

    if not isinstance(similarity_df, pd.DataFrame):
        raise TypeError("The similarity_df must be a pandas DataFrame.")

    if not all(col in similarity_df.columns for col in ['n1_animal_id', 'n1_distance']):
        raise ValueError("The similarity DataFrame must contain columns 'n1_animal_id' and 'n1_distance'.")

    if not isinstance(n, int) or n <= 0:
        raise ValueError("The parameter 'n' must be a positive integer.")

    # Add reciprocal distance column for weighted similarity scores
    similarity_df['reciprocal_distance'] = 1 / similarity_df['n1_distance']

    # Create weighted_ids DataFrame: sum of reciprocal distances per 'n1_animal_id'
    weighted_ids = similarity_df.groupby('n1_animal_id')['reciprocal_distance'].sum().reset_index()

    # Add frequency column: count how many times each animal ID appears in similarity_df
    freq = similarity_df['n1_animal_id'].value_counts().reset_index()
    freq.columns = ['n1_animal_id', 'freq']  # Rename columns to match the `weighted_ids` DataFrame

    # Merge the frequency information with the weighted_ids DataFrame
    weighted_ids = weighted_ids.merge(freq, on='n1_animal_id', how='left')

    # Calculate the weighted frequency
    total_sum = similarity_df['reciprocal_distance'].sum()
    weighted_ids['weighted_ids'] = weighted_ids['reciprocal_distance'] / total_sum

    # Sort weighted_ids by 'weighted_ids' in descending order
    weighted_ids.sort_values('weighted_ids', ascending=False, inplace=True)

    # Get the top n animal IDs based on weighted similarity
    top_n = weighted_ids.head(n)
    prediction = top_n.head(1)['n1_animal_id'].iloc[0]

    return prediction, top_n


def get_weight_ratio(top_ids_df: pd.DataFrame) -> float:
    """
    Computes the ratio between the first two weighted IDs in the top n DataFrame.
    If there is only one value, it returns a ratio of 0.

    Args:
    - top_ids_df (pd.DataFrame): A DataFrame containing 'weighted_ids' column for top IDs.

    Returns:
    - float: The ratio between the first and second weighted ID. Returns 0 if only one value is available.

    Raises:
    - ValueError: If 'weighted_ids' column is not found in the DataFrame.
    """

    if not isinstance(top_ids_df, pd.DataFrame):
        raise TypeError("The top_ids_df must be a pandas DataFrame.")

    if 'weighted_ids' not in top_ids_df.columns:
        raise ValueError("The DataFrame must contain a 'weighted_ids' column.")

    weighted_id_values = top_ids_df['weighted_ids'].head(2).values

    if len(weighted_id_values) > 1:
        weighted_ids_pred1 = weighted_id_values[0]  # First value
        weighted_ids_pred2 = weighted_id_values[1]  # Second value
        ratio = (weighted_ids_pred2 / weighted_ids_pred1)
    else:
        ratio = 0

    return ratio


def get_ratio(closed_glob_topN_ids):
    weighted_id_values = closed_glob_topN_ids['weighted_ids'].head(2).values
    if len(weighted_id_values) > 1:
        weighted_ids_pred1 = weighted_id_values[0]  # First value
        weighted_ids_pred2 = weighted_id_values[1]  # Second value
        closed_ratio = (weighted_ids_pred2 / weighted_ids_pred1)
    else:
        closed_ratio = 0
    return closed_ratio


def get_conf_score(curve_correct_match, curve_open_match, ratio):
    if ratio == 0:
        conf_score = 1
    else:
        true = curve_correct_match.evaluate(ratio)[0]
        false = curve_open_match.evaluate(ratio)[0]
        conf_score = true/(false+true)

    return conf_score


def open_closed_behaviour(save_dir, db_imgs, topN_ids_match, sift_extractor, idx_map, db_annoy_index, n):

    # PREPARE
    closed_set_correct_ratios = []
    closed_set_false_ratios = []
    open_set_ratios = []

    seed = 42  # for deterministic randomness
    all_imgs = os.listdir(db_imgs)
    selected_imgs = random.sample(all_imgs, min(n, len(all_imgs)))


    ################################################################################
    ################################################################################

    # ITERATE Database IMAGES
    for curr_img_filename in tqdm(selected_imgs, desc="Analysing your database for confidence score calculations: "):

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
        # close
        same_img_idx = [i for i, val in enumerate(idx_map['img_ids']) if val in q_img_id] # db position w current image
        closed_similarity_df = similarity_w_idx_filtering(
            db_annoy_index,
            idx_map,
            q_animal_id,
            q_desc_vecs,
            q_img_id,
            same_img_idx,
            k_for_knn=10,
            scenario="closed")

        # open
        same_animal_idx = [i for i, val in enumerate(idx_map['animal_ids']) if val in q_animal_id] # db position w curr animal
        nr_img_per_id = len(set(idx_map['img_ids'][same_animal_idx]))
        open_similarity_df = similarity_w_idx_filtering(
            db_annoy_index,
            idx_map,
            q_animal_id,
            q_desc_vecs,
            q_img_id,
            same_animal_idx,
            k_for_knn=nr_img_per_id+1,
            scenario="open")

        #############################
        # Module-4: Inference
        #############################
        closed_glob_pred, closed_glob_topN_ids = inference_w_confidence_scores(closed_similarity_df, topN_ids_match)
        open_glob_pred, open_glob_topN_ids = inference_w_confidence_scores(open_similarity_df, topN_ids_match)

        ##########################################################
        #                       EVALUATE
        ##########################################################

        closed_ratio = get_ratio(closed_glob_topN_ids)
        open_ratio = get_ratio(open_glob_topN_ids)

        if closed_glob_pred == q_animal_id:
            # print(f'correct ID ({closed_ratio:.3f})')
            closed_set_correct_ratios.append(closed_ratio)
        else:
            # print(f'bad ID ({closed_ratio:.3f})')
            closed_set_false_ratios.append(closed_ratio)

        if open_glob_pred != q_animal_id:
            open_set_ratios.append(open_ratio)

    #########################################

    closed_set_correct_ratios = np.array(closed_set_correct_ratios)
    closed_set_false_ratios = np.array(closed_set_false_ratios)
    open_set_ratios = np.array(open_set_ratios)
    np.save(f'{save_dir}/closed_set_correct_ratios.npy', closed_set_correct_ratios)
    np.save(f'{save_dir}/closed_set_false_ratios.npy', closed_set_false_ratios)
    np.save(f'{save_dir}/open_set_ratios.npy', open_set_ratios)

    kde_curve_ccm, kde_curve_cfm, kde_curve_om = KDE_curves(closed_set_correct_ratios, closed_set_false_ratios, open_set_ratios, save_dir)

    return kde_curve_ccm, kde_curve_cfm, kde_curve_om


def KDE_curves(closed_set_true_ratio, closed_set_false_ratio, open_set_ratios, save_dir):

    # Determine the number of bins
    bins = int(np.sqrt(max(len(closed_set_true_ratio), len(closed_set_false_ratio), len(open_set_ratios))))  # , len(closed_false_ratio))))

    # Fit kernel density estimation (KDE) curves
    if len(closed_set_true_ratio) > 1:
        kde_curve_cscm = sm.nonparametric.KDEUnivariate(closed_set_true_ratio)
        kde_curve_cscm.fit(bw="scott", gridsize=100, cut=0)
    else:
        kde_curve_cscm = None  # Skip KDE if not enough data
    if len(closed_set_false_ratio) > 1:
        kde_curve_csfm = sm.nonparametric.KDEUnivariate(closed_set_false_ratio)
        kde_curve_csfm.fit(bw="scott", gridsize=100, cut=0)
    else:
        kde_curve_csfm = None
    if len(open_set_ratios) > 1:
        kde_curve_osm = sm.nonparametric.KDEUnivariate(open_set_ratios)
        kde_curve_osm.fit(bw="scott", gridsize=100, cut=0)
    else:
        kde_curve_osm = None

    # Create smooth x values for plotting KDE
    x_min = min(closed_set_true_ratio.min(), closed_set_false_ratio.min(), open_set_ratios.min())  # , closed_false_ratio.min())
    x_max = max(closed_set_true_ratio.max(), closed_set_false_ratio.max(), open_set_ratios.max())  # , closed_false_ratio.max())
    x_values = np.linspace(x_min, x_max, 300)
    # Plot histograms
    # plt.rcParams.update({
    #     "text.usetex": True,  # Enable LaTeX rendering
    #     "font.family": "serif",  # LaTeX-style serif font (Times-like)
    #     "pgf.preamble": r"\usepackage{amsmath, amssymb}",  # Include math packages if needed
    #     'axes.titlesize': 18,  # Title font size (12pt)
    #     'axes.labelsize': 15,  # Axis labels font size (10pt)
    #     'xtick.labelsize': 15,  # X-tick font size (10pt)
    #     'ytick.labelsize': 15,  # Y-tick font size (10pt)
    #     'legend.fontsize': 15,  # Legend font size (10pt)
    #     # 'figure.titlesize': 24,  # Figure title font size (12pt)
    #     'axes.labelweight': 'normal',  # Normal weight for axis labels
    #     'axes.titleweight': 'normal',  # Normal weight for axis titles
    #     'font.size': 20  # General font size for plot text (10pt)
    # })
    plt.hist(closed_set_true_ratio, bins=bins, density=True, alpha=0.6, color="green", edgecolor="black",)
             # label=f"cscm ({len(closed_set_true_ratio)})")
    plt.hist(closed_set_false_ratio, bins=bins, density=True, alpha=0.6, color="red", edgecolor="black",)
             # label=f"csfm ({len(closed_set_false_ratio)})")
    plt.hist(open_set_ratios, bins=bins, density=True, alpha=0.6, color="orange", edgecolor="black",)
             # label=f"osm ({len(open_set_ratios)})")
    # plt.hist(closed_false_ratio, bins=bins, density=True, alpha=0.6, edgecolor="black", label=f"Closed False ({len(closed_false_ratio)})")
    # Plot KDE curves
    if kde_curve_cscm:
        plt.plot(x_values, kde_curve_cscm.evaluate(x_values), lw=2, label="Closed-Set Correct Match", color="green")
        # plt.plot(x_values, kde_curve_cscm(x_values), lw=2, label="KDE Closed True", color="blue")
    if kde_curve_csfm:
        plt.plot(x_values, kde_curve_csfm.evaluate(x_values), lw=2, label="Closed-Set False Match", color="red")
        # plt.plot(x_values, kde_curve_csfm(x_values), lw=2, label="KDE Open False", color="red")
    if kde_curve_osm:
        plt.plot(x_values, kde_curve_osm.evaluate(x_values), lw=2, label="Open-Set Match", color="orange")
        # plt.plot(x_values, kde_curve_csfm(x_values), lw=2, label="KDE Open False", color="red")
    # Labels and legend
    plt.xlabel("Ratio")
    plt.ylabel("Density")
    plt.title("Database match analysis")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"confidence_score_hist_and_kde.png"))
    plt.savefig(os.path.join(save_dir, "confidence_score_hist_and_kde.svg"), format="svg")
    plt.close()
    return kde_curve_cscm, kde_curve_csfm, kde_curve_osm