# Apply SAHI ver.
import os 
import re
import json
import glob
import time
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from sahi.predict import get_prediction, get_sliced_prediction
from sahi import AutoDetectionModel
from ultralytics import YOLO
from pycocotools.coco import COCO  # noqa
from pycocotools.cocoeval import COCOeval  # noqa

def create_log(res_name):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(f"./{res_name}.txt", 'w')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    return log

# use gpu
def set_device(gpu_id):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id;
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    if device=="cuda": torch.cuda.empty_cache()
    print("Device:", device)
    return device

def sahi(model, model_name, slice_height, slice_width, res_name):
    valid_db_path = "/trafficlight-detect/val"
    valid_res_path = "%s/predictions"%(valid_db_path)
    if not os.path.exists(valid_res_path): os.makedirs(valid_res_path, exist_ok=True)
    img_exts = ["jpg","bmp","png"]
    img_files = list()
    for img_ext in img_exts:
        img_files += glob.glob("%s/images/*.%s"%(valid_db_path, img_ext))
    img_files.sort() 

    coco_results = []
    for img_filename in tqdm(img_files):
        # Validate the model
        result = get_sliced_prediction(
            img_filename,
            model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            #iou=0.5,
            )
        image_id = int(Path(img_filename).stem)
        coco_predictions = result.to_coco_predictions(image_id=image_id)
        coco_results.extend(coco_predictions)
    json_save_path = os.path.join(valid_res_path, f"{res_name}.json")
    with open(json_save_path, 'w') as f:
        json.dump(coco_results, f)
    # Save SAHI Result
    return coco_results, json_save_path, img_files

def eval(pred_json, anno_json, img_files):
    """Evaluates YOLO output in JSON format and returns performance statistics."""
    print(f"\nEvaluating mAP using {pred_json} and {anno_json}...")
    anno = COCO(str(anno_json))  # init annotations api
    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
    val = COCOeval(anno, pred, "bbox")
    val.params.imgIds = [int(Path(x).stem) for x in img_files]  # images to eval
    val.evaluate()
    val.accumulate()
    val.summarize()
    rows = ['Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ',
            'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = ', # <- This is what we need
            'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = ',
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ',
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ',
            'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = ',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = ',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = ',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = ',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = ',
            'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = ',]
    return rows, val.stats

def main(
    model_path: str,
    gpu_id: str,
    slice_height: int,
    slice_width: int,
    confidence_threshold: float
    ):
    # root folder for model weight save; output folder
    root = "/detect"
    os.chdir(root)
    
    # SAHI
    # Load model weights
    start = time.time()
    device = set_device(gpu_id)
    model_name = model_path.split('/')[-3]
    res_name = f"{model_name}_h{slice_height}_w{slice_width}_c{confidence_threshold}"
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', # yolov10 doesn't exist
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
        image_size=1280,
        )
    logger = create_log(res_name)
    logger.info(f"model path: {model_path}, slice_height: {slice_height}, slice_width: {slice_width}, confidence_threshold: {confidence_threshold}")
    sahi_res, pred_json, img_files = sahi(model, model_name, slice_height, slice_width, res_name)
    logger.info(f"SAHI Done in {time.time()-start:.4f} sec")

    # COCO Eval
    start = time.time()
    anno_json = "/detect/annotations.json"
    rows, values = eval(pred_json, anno_json, img_files)
    for row, value in zip(rows, values):
        logger.info(f"{row} {str(value)}")
    logger.info(f"Max mAP: {str(max(values))}")
    logger.info(f"Evaluation Done in {time.time()-start:.4f} sec")
    logger.info("All Done")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-m", '--model_path', type=str, default="/trafficlight-detect/runs/detect/yolov8x_sgd_24.10.07-19:13:16_0.98612/weights/best.pt") 
    PARSER.add_argument("-g", '--gpu_id', type=str, default = "2")
    PARSER.add_argument("-ht", '--slice_height', type=int, default = 256)
    PARSER.add_argument("-wt", '--slice_width', type=int, default = 256)
    PARSER.add_argument("-c", '--confidence_threshold', type=float, default = 0.001)
    main(**vars(PARSER.parse_args()))
