import numpy as np
import pandas as pd


def evaluate_acc(
    q_animal_id: str,
    top_ids_df: pd.DataFrame,
    R1_acc: int,
    R3_acc: int,
    R5_acc: int,
    R10_acc: int,
    RR_k1: list,
    RR_k3: list,
    RR_k5: list,
    RR_k10: list,
) -> tuple:
    """
    Evaluates various ranking accuracy metrics based on the position of a query
    animal ID in the top IDs DataFrame.

    This function computes several metrics, including:
    - Reciprocal Rank (RR)
    - Rank-K Accuracy (RK_acc) for k=1, 3, 5, 10
    These metrics are updated in-place and returned after evaluation.

    Args:
    - q_animal_id (str): The ID of the query animal.
    - top_ids_df (pd.DataFrame): A DataFrame containing the top ranked animal IDs and
    their associated ranks.
    - R1_acc (int): The current Rank-1 accuracy count.
    - R3_acc (int): The current Rank-3 accuracy count.
    - R5_acc (int): The current Rank-5 accuracy count.
    - R10_acc (int): The current Rank-10 accuracy count.
    - RR_k1 (list): A list to store Reciprocal Rank (RR) for k=1.
    - RR_k3 (list): A list to store Reciprocal Rank (RR) for k=3.
    - RR_k5 (list): A list to store Reciprocal Rank (RR) for k=5.
    - RR_k10 (list): A list to store Reciprocal Rank (RR) for k=10.

    Returns:
    - tuple: A tuple containing updated accuracy counts (R1_acc, R3_acc, R5_acc,
    R10_acc) and lists for Reciprocal Rank
      (RR_k1, RR_k3, RR_k5, RR_k10).

    Raises:
    - ValueError: If 'top_ids_df' is not a DataFrame or does not contain the expected
    'n1_animal_id' column.
    - TypeError: If input types do not match the expected types.
    """

    # Validate inputs
    if not isinstance(top_ids_df, pd.DataFrame):
        raise TypeError("top_ids_df must be a pandas DataFrame.")

    if "n1_animal_id" not in top_ids_df.columns:
        raise ValueError("The DataFrame must contain the 'n1_animal_id' column.")

    if not isinstance(q_animal_id, str):
        raise TypeError("q_animal_id must be a string.")

    if not all(isinstance(i, int) for i in [R1_acc, R3_acc, R5_acc, R10_acc]):
        raise TypeError(
            "Rank accuracy counts (R1_acc, R3_acc, R5_acc, R10_acc) must be integers."
        )

    if not all(isinstance(i, list) for i in [RR_k1, RR_k3, RR_k5, RR_k10]):
        raise TypeError(
            "Reciprocal Rank lists (RR_k1, RR_k3, RR_k5, RR_k10) must be lists."
        )

    # Get rank of the query animal
    if q_animal_id in top_ids_df["n1_animal_id"].head(10).values:
        rank = (
            np.where(top_ids_df["n1_animal_id"].head(10).values == q_animal_id)[0][0]
            + 1
        )
    else:
        rank = np.inf  # If not found, set rank to infinity

    # Update metrics based on rank
    if rank == 1:
        R1_acc += 1
        R3_acc += 1
        R5_acc += 1
        R10_acc += 1
        RR_k1.append(1 / rank)
        RR_k3.append(1 / rank)
        RR_k5.append(1 / rank)
        RR_k10.append(1 / rank)
    elif rank <= 3:
        R3_acc += 1
        R5_acc += 1
        R10_acc += 1
        RR_k1.append(0)
        RR_k3.append(1 / rank)
        RR_k5.append(1 / rank)
        RR_k10.append(1 / rank)
    elif rank <= 5:
        R5_acc += 1
        R10_acc += 1
        RR_k1.append(0)
        RR_k3.append(0)
        RR_k5.append(1 / rank)
        RR_k10.append(1 / rank)
    elif rank <= 10:
        R10_acc += 1
        RR_k1.append(0)
        RR_k3.append(0)
        RR_k5.append(0)
        RR_k10.append(1 / rank)
    else:
        RR_k1.append(0)
        RR_k3.append(0)
        RR_k5.append(0)
        RR_k10.append(0)

    return R1_acc, R3_acc, R5_acc, R10_acc, RR_k1, RR_k3, RR_k5, RR_k10
