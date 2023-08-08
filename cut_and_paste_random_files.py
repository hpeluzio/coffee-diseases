import os
import random
import shutil

def cut_random_files(source_folder, destination_folder, num_files):
    # Get a list of all files in the source folder
    all_files = os.listdir(source_folder)
    
    # Choose `num_files` random files from the list
    random_files = random.sample(all_files, num_files)
    
    for file_name in random_files:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)
        
        # Move the file from source to destination
        shutil.move(source_path, destination_path)
        print(f"Moved: {source_path} -> {destination_path}")

# Replace these paths with your actual source and destination folder paths
source_folder_0_Cercospora = 'dataset/swatdcnn/data/Augmented/stage_3/train/0_Cercospora'
destination_folder_0_Cercospora = 'dataset/swatdcnn/data/Augmented/stage_3/validation/0_Cercospora'

source_folder_1_Phoma = 'dataset/swatdcnn/data/Augmented/stage_3/train/1_Phoma'
destination_folder_1_Phoma = 'dataset/swatdcnn/data/Augmented/stage_3/validation/1_Phoma'

source_folder_2_Leaf_Miner= 'dataset/swatdcnn/data/Augmented/stage_3/train/2_Leaf_Miner'
destination_folder_2_Leaf_Miner = 'dataset/swatdcnn/data/Augmented/stage_3/validation/2_Leaf_Miner'

source_folder_3_Red_Spider_Mite = 'dataset/swatdcnn/data/Augmented/stage_3/train/3_Red_Spider_Mite'
destination_folder_3_Red_Spider_Mite = 'dataset/swatdcnn/data/Augmented/stage_3/validation/3_Red_Spider_Mite'
num_files = 250

cut_random_files(source_folder_0_Cercospora, destination_folder_0_Cercospora, num_files)
cut_random_files(source_folder_1_Phoma, destination_folder_1_Phoma, num_files)
cut_random_files(source_folder_2_Leaf_Miner, destination_folder_2_Leaf_Miner, num_files)
cut_random_files(source_folder_3_Red_Spider_Mite, destination_folder_3_Red_Spider_Mite, num_files)
