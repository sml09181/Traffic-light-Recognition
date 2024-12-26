import os
import shutil
import json

root = "/trafficlight-detect"
source_train_images = os.path.join(root, '/train/images')
source_val_images = os.path.join(root, '/val/images')
source_train_labels = os.path.join(root, '/train/labels')
source_val_labels = os.path.join(root, '/val/labels')

destination_images = os.path.join(root, '/train/images')
destination_labels = os.path.join(root, '/train/labels')

os.makedirs(destination_images, exist_ok=True)
os.makedirs(destination_labels, exist_ok=True)

# copy img files
for source in [source_train_images, source_val_images]:
    for filename in os.listdir(source):
        full_file_name = os.path.join(source, filename)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination_images)

# copy label files
for source in [source_train_labels, source_val_labels]:
    for filename in os.listdir(source):
        full_file_name = os.path.join(source, filename)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination_labels)

def count_files(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

source_train_images_count = count_files(source_train_images)
source_val_images_count = count_files(source_val_images)
source_train_labels_count = count_files(source_train_labels)
source_val_labels_count = count_files(source_val_labels)

destination_images_count = count_files(destination_images)
destination_labels_count = count_files(destination_labels)

print(f"# of src train images: {source_train_images_count}")
print(f"# of src train labels: {source_train_labels_count}")
print(f"# of src val images: {source_val_images_count}")
print(f"# of src val labels: {source_val_labels_count}")
print(f"# of dst images: {destination_images_count}")
print(f"# of dst labels: {destination_labels_count}")
print("All Done")