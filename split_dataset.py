import os
import shutil
import random

# Define the paths
SOURCE_DIR = r'K:\Data_science\Rice Field weed BD Dataset_V4'
DEST_DIR = r'K:\Data_science\rice_weed_dataset_split'

# Define the split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Get the list of all class folders
class_folders = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]

print(f"Found {len(class_folders)} class folders to process.")

# Create the main destination directories
train_path = os.path.join(DEST_DIR, 'train')
val_path = os.path.join(DEST_DIR, 'validation')
test_path = os.path.join(DEST_DIR, 'test')

for path in [train_path, val_path, test_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Iterate through each class folder to split the images
for class_name in class_folders:
    source_class_path = os.path.join(SOURCE_DIR, class_name)
    
    # Get all image file names
    images = [f for f in os.listdir(source_class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    
    # Calculate the split indices
    num_images = len(images)
    train_split_index = int(num_images * TRAIN_RATIO)
    val_split_index = train_split_index + int(num_images * VAL_RATIO)
    
    # Split the list of images
    train_images = images[:train_split_index]
    val_images = images[train_split_index:val_split_index]
    test_images = images[val_split_index:]
    
    print(f"Processing '{class_name}': {num_images} images found.")
    print(f"  - Train: {len(train_images)} images")
    print(f"  - Validation: {len(val_images)} images")
    print(f"  - Test: {len(test_images)} images")

    # Create destination subfolders for the current class
    dest_train_class_path = os.path.join(train_path, class_name)
    dest_val_class_path = os.path.join(val_path, class_name)
    dest_test_class_path = os.path.join(test_path, class_name)

    for path in [dest_train_class_path, dest_val_class_path, dest_test_class_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # Copy images to the new folders
    for image in train_images:
        shutil.copy(os.path.join(source_class_path, image), os.path.join(dest_train_class_path, image))

    for image in val_images:
        shutil.copy(os.path.join(source_class_path, image), os.path.join(dest_val_class_path, image))
        
    for image in test_images:
        shutil.copy(os.path.join(source_class_path, image), os.path.join(dest_test_class_path, image))

print("\nDataset segregation complete!")
print(f"Your new dataset structure is located at: {DEST_DIR}")