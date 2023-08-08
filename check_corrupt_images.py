import os
from PIL import Image

def check_corrupt_images(folder_path):
    corrupt_images = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            with Image.open(file_path) as img:
                img.verify()  # Attempt to verify image integrity
        except (IOError, SyntaxError) as e:
            print(f"Corrupt image: {file_path}")
            corrupt_images.append(file_path)

    return corrupt_images

# Specify the folder path to check for corrupt images
folder_path = './dataset/swatdcnn/data/Augmented/stage_3/train/0_Cercospora'
# folder_path = './dataset/swatdcnn/data/Augmented/stage_3/train/1_Phoma'
# folder_path = './dataset/swatdcnn/data/Augmented/stage_3/train/2_Leaf_Miner'
# folder_path = './dataset/swatdcnn/data/Augmented/stage_3/train/3_Red_Spider_Mite'

corrupt_images = check_corrupt_images(folder_path)

if len(corrupt_images) == 0:
    print("No corrupt images found.")
else:
    print(f"Total {len(corrupt_images)} corrupt images found:")
    
