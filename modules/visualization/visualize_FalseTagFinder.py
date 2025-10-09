import os
import random
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib import pyplot as plt

def false_tag_finder_plot(save_dir, glob_sim_df, glob_topN_ids, q_img_id, q_animal_id, glob_pred, conf_score, ratio, db_imgs, filename, counter):

    ######################################
    #       Supporting functions
    ######################################

    def gut_full_filename_from_img_id(imgs_dir: str, img_id: str):
        # Get all files in the folder
        all_files = os.listdir(imgs_dir)

        # Filter files where the third part of the split filename matches the search string
        matching_files = [
            file for file in all_files
            if len(file.split('_')) > 2 and file.split('_')[2] == img_id
        ]

        # If there is more than one matching file, print a message and exit
        if len(matching_files) > 1:
            print("More than one file with this frame id!")
            exit()

        # If matching file is found, return its full path
        if matching_files:
            return os.path.join(imgs_dir, matching_files[0])

        else:
            print(f'error: {img_id} file not found in folder with image ID')
            exit()

    def get_best_match_imgs(df: pd.DataFrame, id1: str, id2: str, id3: str, db_img: str):
        result = []

        # Function to process each id and return the two desired values
        def get_top2_imgs(curr_id):
            # Subset dataframe for the current id
            subset = df[df['n1_animal_id'] == curr_id]

            # Get the most frequent 'n1_img_id'
            most_frequent_img_id = subset['n1_img_id'].mode()[0]

            # Find all 'n1_img_id's that have the same frequency
            frequent_values = subset[subset['n1_img_id'] == most_frequent_img_id]

            # If multiple have the same frequency, calculate the average reciprocal distance
            if len(frequent_values) > 1:
                avg_reciprocal_distance = frequent_values.groupby('n1_img_id')['reciprocal_distance'].mean()
                most_frequent_img_id = avg_reciprocal_distance.idxmax()

            # Now get the 'n1_img_id' with the largest 'reciprocal_distance'
            largest_reciprocal_img_id = subset.loc[subset['reciprocal_distance'].idxmax(), 'n1_img_id']

            # Check if the two values are the same
            if most_frequent_img_id == largest_reciprocal_img_id:
                result.append((db_imgs+'/'+most_frequent_img_id, db_imgs+'/'+largest_reciprocal_img_id))
            else:
                result.append((db_imgs+'/'+most_frequent_img_id, db_imgs+'/'+largest_reciprocal_img_id))

        # Process each id
        for curr_id in [id1, id2, id3]:
            get_top2_imgs(curr_id)

        return result

    def get_6_db_images_for_query(folder_path: str, animal_id: str, img_id: str, side: str):
        # Get all files in the directory
        all_files = os.listdir(folder_path)

        # Filter files where the animal_id is the first part of the filename
        # and reject files where the img_id is the third part of the filename
        matching_files = [
            file for file in all_files
            if animal_id == file.split('_')[0] and file.split('_')[2] != img_id
        ]

        # Separate files that match the side preference (split[1] == side)
        side_files = [file for file in matching_files if file.split('_')[1] == side]

        # If there are more than 6 matching files, prioritize those with the correct side
        if len(matching_files) > 6:
            # Ensure we take up to 6 images, prioritizing the ones with the correct side
            num_side_files = min(len(side_files), 6)  # Limit to 6 or the number of side_files
            random_files = random.sample(side_files, num_side_files)

            # Now fill the remaining slots from the other files
            remaining_files = [file for file in matching_files if file not in side_files]
            num_remaining_files = 6 - num_side_files
            if num_remaining_files > 0:
                random_files += random.sample(remaining_files, num_remaining_files)

        else:
            # If there are fewer than or exactly 6 files, select them all randomly
            random_files = random.sample(matching_files, len(matching_files))

        # Convert to full paths
        random_files = [os.path.join(folder_path, file) for file in random_files]

        return random_files

    ######################################
    #       Get statistics
    ######################################

    # Calculate the number of unique images (n1_img_id) per animal in glob_sim_df
    unique_image_count = glob_sim_df.groupby('n1_animal_id')['n1_img_id'].nunique()
    # Merge this information into the glob_topN_ids DataFrame
    merged_df = glob_topN_ids.merge(unique_image_count, left_on='n1_animal_id', right_index=True, how='left')
    merged_df.rename(columns={'n1_animal_id': 'Animal ID', 'weighted_ids': 'Weighted IDs', 'freq': 'Frequency',
                              'n1_img_id': 'Unique Image Count'}, inplace=True)
    # Sort by 'Weighted IDs' in descending order
    merged_df = merged_df.sort_values(by='Weighted IDs', ascending=False)

    ######################################
    #       Get paths for plotting
    ######################################

    q_db_img_paths = get_6_db_images_for_query(db_imgs, q_animal_id, q_img_id, side=filename.split('_')[1])
    q_img_path = os.path.join(db_imgs, filename)
    top_match_img_ids = get_best_match_imgs(glob_sim_df, glob_topN_ids['n1_animal_id'].iloc[0], glob_topN_ids['n1_animal_id'].iloc[1],
                          glob_topN_ids['n1_animal_id'].iloc[2], db_imgs)

    if len(q_db_img_paths) == 1:
        image_paths = [
            # Row 0
            ['', q_img_path, q_db_img_paths[0], q_db_img_paths[0], q_db_img_paths[0]],
            # Row 1
            [top_match_img_ids[0][0], top_match_img_ids[0][1], q_db_img_paths[0], q_db_img_paths[0], q_db_img_paths[0]],
            # Row 2
            [top_match_img_ids[1][0], top_match_img_ids[1][1]],
            # Row 3
            [top_match_img_ids[2][0], top_match_img_ids[2][1]]
        ]
    elif len(q_db_img_paths) == 2:
        image_paths = [
            # Row 0
            ['', q_img_path, q_db_img_paths[0], q_db_img_paths[1], q_db_img_paths[1]],
            # Row 1
            [top_match_img_ids[0][0], top_match_img_ids[0][1], q_db_img_paths[1], q_db_img_paths[1], q_db_img_paths[1]],
            # Row 2
            [top_match_img_ids[1][0], top_match_img_ids[1][1]],
            # Row 3
            [top_match_img_ids[2][0], top_match_img_ids[2][1]]
        ]

    elif len(q_db_img_paths) == 3:
        image_paths = [
            # Row 0
            ['', q_img_path, q_db_img_paths[0], q_db_img_paths[1], q_db_img_paths[2]],
            # Row 1
            [top_match_img_ids[0][0], top_match_img_ids[0][1], q_db_img_paths[2], q_db_img_paths[2], q_db_img_paths[2]],
            # Row 2
            [top_match_img_ids[1][0], top_match_img_ids[1][1]],
            # Row 3
            [top_match_img_ids[2][0], top_match_img_ids[2][1]]
        ]
    elif len(q_db_img_paths) == 4:
        image_paths = [
            # Row 0
            ['', q_img_path, q_db_img_paths[0], q_db_img_paths[1], q_db_img_paths[2]],
            # Row 1
            [top_match_img_ids[0][0], top_match_img_ids[0][1], q_db_img_paths[3], q_db_img_paths[3], q_db_img_paths[3]],
            # Row 2
            [top_match_img_ids[1][0], top_match_img_ids[1][1]],
            # Row 3
            [top_match_img_ids[2][0], top_match_img_ids[2][1]]
        ]
    elif len(q_db_img_paths) == 5:
        image_paths = [
            # Row 0
            ['', q_img_path, q_db_img_paths[0], q_db_img_paths[1], q_db_img_paths[2]],
            # Row 1
            [top_match_img_ids[0][0], top_match_img_ids[0][1], q_db_img_paths[3], q_db_img_paths[4], q_db_img_paths[4]],
            # Row 2
            [top_match_img_ids[1][0], top_match_img_ids[1][1]],
            # Row 3
            [top_match_img_ids[2][0], top_match_img_ids[2][1]]
        ]
    elif len(q_db_img_paths) == 6:
        image_paths = [
            # Row 0
            ['', q_img_path, q_db_img_paths[0], q_db_img_paths[1], q_db_img_paths[2]],
            # Row 1
            [top_match_img_ids[0][0], top_match_img_ids[0][1], q_db_img_paths[3], q_db_img_paths[4], q_db_img_paths[5]],
            # Row 2
            [top_match_img_ids[1][0], top_match_img_ids[1][1]],
            # Row 3
            [top_match_img_ids[2][0], top_match_img_ids[2][1]]
        ]

    else:
        image_paths = [
            # Row 0
            ['', q_img_path, q_img_path, q_img_path, q_img_path],
            # Row 1
            [top_match_img_ids[0][0], top_match_img_ids[0][1], q_img_path, q_img_path, q_img_path],
            # Row 2
            [top_match_img_ids[1][0], top_match_img_ids[1][1]],
            # Row 3
            [top_match_img_ids[2][0], top_match_img_ids[2][1]]
        ]

    extra_string = [
            # Row 0
            ['', 'QUERY:', 'Q in DB, example:', 'Q in DB, example:', 'Q in DB, example::'],
            # Row 1
            ['Top1-MATCH (most match):', 'Top1-MATCH (largest weight):', 'Q in DB, example:', 'Q in DB, example:', 'Q in DB, example:'],
            # Row 2
            ['Top2-MATCH (most match):', 'Top2-MATCH (largest weight):', '', '', ''],
            # Row 3
            ['Top3-MATCH (most match):', 'Top3-MATCH (largest weight):', '', '', '']
        ]

    def prepare_img_titles(image_paths, extra_strings):

        if len(image_paths) != len(extra_strings):
            raise ValueError("The length of image_paths and extra_strings must be the same")

        processed_paths = []

        # Iterate through each row and process it
        for row_idx, row in enumerate(image_paths):
            processed_row = []
            extra_row = extra_strings[row_idx]  # The extra row corresponding to this image row

            # Iterate through each element in the row
            for i, path in enumerate(row):
                if path:  # If the string is non-empty
                    # Prepend the corresponding extra string and append the filename
                    processed_string = f"{extra_row[i]}\n{os.path.basename(path)}"
                    processed_row.append(processed_string)
                else:
                    processed_row.append('')  # Keep empty string if the path is empty

            # Ensure each row has exactly 5 elements, adding empty strings if necessary
            while len(processed_row) < 5:
                processed_row.append('')

            processed_paths.append(processed_row[:5])  # Ensure no row exceeds 5 elements

        return processed_paths

    # processed_image_paths = process_image_paths(image_paths)
    processed_image_paths = prepare_img_titles(image_paths, extra_string)
    flattened_image_paths = [file for sublist in processed_image_paths for file in sublist]
    ######################################
    #                PLOT
    ######################################


    # Create a figure with 5 columns and 4 rows
    fig, axes = plt.subplots(4, 5, figsize=(15, 12), dpi=300)

    # Function to load images from a given path
    def load_image(image_path):
        return mpimg.imread(image_path)

    # Function to add a colored border using a rectangle patch
    def add_border(ax, color, linewidth=5):
        rect = patches.Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            linewidth=linewidth, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

    # Fill in the grid according to the described structure
    for row in range(4):
        for col in range(5):
            if (row == 0 and col == 0):  # First cell is empty
                axes[row, col].axis('off')
            elif (row in [0, 1]) or (row in [2, 3] and col < 2):  # Normal images
                img = load_image(image_paths[row][col])
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].axis('off')

                # Apply border colors based on the specified conditions
                if row in [1, 2, 3] and col < 2:  # 2nd & 4th row, first 2 images
                    add_border(axes[row, col], 'red')
                if row == 0 and col == 1:  # 1st row, 2nd image (position 2)
                    add_border(axes[row, col], 'orange')
                if (row in [0, 1]) and col >= 2:  # 1st and 2nd row, last 3 images
                    add_border(axes[row, col], 'green')

            elif row in [2, 3] and col >= 2:  # Big plot merging last 3 columns
                axes[row, col].remove()

    # Merge the last 3 columns in row 3 and 4 into a big subplot
    big_ax = fig.add_subplot(4, 5, (13, 20))  # Spanning indices

    # Now plot the scatter plot in the big_ax
    scatter = big_ax.scatter(
        merged_df['Animal ID'], merged_df['Weighted IDs'],
        s=merged_df['Frequency'] * 25,  # Marker size based on occurrence frequency, scaled
        c=merged_df['Unique Image Count'],  # Color based on unique image count
        cmap='viridis',  # Color map (you can change it to any you prefer)
        alpha=0.7
    )
    big_ax.set_title("ID match analytics")

    # Add color bar to indicate unique image count values
    big_ax.figure.colorbar(scatter, ax=big_ax, label='Number of retrieved imgs per ID')

    # Add text labels next to each point in the scatter plot
    for i, row in merged_df.iterrows():
        big_ax.text(row['Animal ID'], row['Weighted IDs'], f'{row["Animal ID"]}', ha='right', va='bottom',
                    fontsize=9)

    # Add a simple circle legend (no size scaling, just a single circle)
    circle_legend = mlines.Line2D([], [], marker='o', color='w', markerfacecolor='blue', alpha=0.8, markersize=10,
                                  label='Size  ~  Number of matching keypoints)')
    big_ax.legend(handles=[circle_legend], loc="upper right")

    # Set labels and title for the scatter plot (if needed)
    big_ax.set_xlabel('Animal ID')
    big_ax.set_ylabel('Weights')
    big_ax.set_title(f'ground truth: {q_animal_id}  VS.  predicted: {glob_pred} ({conf_score:.2f} %)')

    # Add titles to images
    for row in range(4):
        for col in range(5):
            # Skip the empty cell at (0, 0)
            if not (row == 0 and col == 0) and not (row == 2 and col == 2) and not (row == 2 and col == 3) and not (row == 2 and col == 4) and not (row == 3 and col == 2) and not (row == 3 and col == 3) and not (row == 3 and col == 4):
                filename = flattened_image_paths[row * 5 + col]  # Index from the flattened list
                axes[row, col].text(0.5, 1.05, filename, ha='center', va='bottom', transform=axes[row, col].transAxes,
                                    fontsize=10, color='black')

    plt.tight_layout()
    save_name = f'{q_img_id}_mismatch{counter}.jpg'
    save_name_svg = f'{q_img_id}_mismatch{counter}.svg'
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    # plt.savefig(os.path.join(save_dir, save_name_svg), format='svg', dpi=300)
    print(f'Saved to {os.path.join(save_dir, save_name)}')
    # plt.show()
    plt.close()