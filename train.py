
import os 
import time
import torch
from argparse import ArgumentParser

from ultralytics import YOLO
from ultralytics.utils import ASSETS, ROOT, WEIGHTS_DIR, checks, is_dir_writeable

# use gpu
def set_device(gpu_id):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id;
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    if device=="cuda": torch.cuda.empty_cache()
    print("Device:", device)
    return device

def main(
    modelname: str,
    optimizer: str,
    gpu_id: str,
    ):
    # root folder for model weight save; output folder
    root = "/trafficlight-detect/"
    os.chdir(root)

    # load a pretrained model
    model = YOLO(os.path.join(root, f'pretrained/yolov{modelname}.pt'))  

    #########################################################
    # Train the model
    model.train(data='/trafficlight-detect/tld_2024.yaml', 
                name=f"su_a_yolov{modelname}_{optimizer.lower()}_{time.strftime('%y.%m.%d-%H:%M:%S')}",
                
                ## In baseline code
                epochs=20, # 30
                imgsz=1280, # 1280
                device=set_device(gpu_id), # 사용할 GPU ID 입력
                batch=8, # 32 -> 8
                cache=False, # True -> False
                pretrained=True, # True
                lr0 = 0.001, # 0.001
                optimizer=optimizer, # 'SGD'
                close_mosaic=5,
                save_period=1,
                save_json=True,
                
                ## augmentation
                augment=True,
                hsv_h=0.02, # 0.015 -> 0.02
                hsv_s=0.3, # 0.7 -> 0.3
                hsv_v=0.2, # 0.4 -> 0.2
                degrees=10.0, # 0.

                translate=0.0, # 0.1 -> 0.0
                scale=0.0, # 0.5 -> 0.0
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr = 0.0, # 0.0
                bgr=0.0,
                mosaic=0.0, # 1.0 -> 0.0
                mixup=0.0,
                copy_paste=0.0,
                #copy_paste_mode="flip",
                auto_augment="randaugment",
                erasing=0.0, # 0.4
                crop_fraction=0.0, # 0.0
                
                ## additional
                patience=100,
                workers=8,
                seed=0,
                mask_ratio=4,
                dropout=0.3,
                iou=0.8, # 0.7
                cos_lr=True,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                )
    print("All Done")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-m", '--modelname', type=str, default="10x") 
    PARSER.add_argument("-o", '--optimizer', type=str, default="SGD") 
    PARSER.add_argument("-g", '--gpu_id', type=str, default = "5")
    main(**vars(PARSER.parse_args()))
