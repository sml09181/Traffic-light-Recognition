# convert yolo format into coco format
import os
import re
import json
from PIL import Image
from tqdm import tqdm

classes = {
    0: 'veh_go',
    1: 'veh_goLeft',
    2: 'veh_noSign',
    3: 'veh_stop',
    4: 'veh_stopLeft',
    5: 'veh_stopWarning',
    6: 'veh_warning',
    7: 'ped_go',
    8: 'ped_noSign',
    9: 'ped_stop',
    10: 'bus_go',
    11: 'bus_noSign',
    12: 'bus_stop',
    13: 'bus_warning',
}

def yolo_to_coco(output_dir):
	# Define categories
    categories = []
    for k, v in classes.items():
        temp = {'id': int(k), 'name': v}
        categories.append(temp)

    image_dir = f'/trafficlight-detect/val/images'
    label_dir = f'/trafficlight-detect/val/labels'

    # Get image and label files for current split
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    # Define the COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": []
    }

    # Loop through the images in the input directory
    for image_file in image_files:
        # Load the image and get its dimensions
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        width, height = image.size
        
        # Add the image to the COCO dataset
        image_dict = {
            "id": int(image_file.split('.')[0]),
            "width": width,
            "height": height,
            "file_name": image_file
        }
        coco_dataset["images"].append(image_dict)
        
        # Load the bounding box annotations for the image
        with open(os.path.join(label_dir, f'{image_file.split(".")[0]}.txt')) as f:
            annotations = f.readlines()
        
        # Loop through the annotations and add them to the COCO dataset
        for ann in annotations:
            x, y, w, h = map(float, ann.strip().split()[1:])
            # norm_center_x norm_center_y norm_w norm_h
            x_min, y_min = (x - w / 2) * width, (y - h / 2) * height # int type casting is not needed
            x_max, y_max = (x + w / 2) * width, (y + h / 2) * height # int type casting is not needed
            ann_dict = {
                "id": len(coco_dataset["annotations"]),
                "image_id": int(image_file.split('.')[0]),
                "category_id": int(ann.strip().split()[0]),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0
            }
            coco_dataset["annotations"].append(ann_dict)

    # Save the COCO dataset to a JSON file
    with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
        json.dump(coco_dataset, f)
    
if __name__ == "__main__":
    output_dir = '/trafficlight-detect/'
    coco_data = yolo_to_coco(output_dir)
    print("All Done")