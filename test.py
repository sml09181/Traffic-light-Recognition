from ultralytics import YOLO
import os 
import glob

import re
import json
import glob
import torch
from argparse import ArgumentParser

def get_file_extension(filename):
    _, extension = os.path.splitext(filename)
    return extension.lstrip('.')

def set_device(gpu_id):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id;
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    if device=="cuda": torch.cuda.empty_cache()
    print("Device:", device)
    return device

def main( 
    model_root: str, 
    gpu_id:str,
    confidence: float,
    iou: float,
    ):
    
    device = set_device(gpu_id)
    root = "/trafficlight-detect/"
    os.chdir(root)

    test_db_path = "/trafficlight-detect/test"
    test_res_path = "%s/predictions"%(test_db_path)

    if not os.path.exists(test_res_path):
        os.makedirs(test_res_path)

    img_exts = ["jpg","bmp","png"]
    img_files = list()
    for img_ext in img_exts:
        img_files += glob.glob("%s/images/*.%s"%(test_db_path, img_ext))
    img_files.sort() 

    weights = os.listdir(os.path.join(model_root, "weights"))
    epochs = sorted([int(re.findall(r'\d+', x)[0]) for x in weights if len(re.findall(r'\d+', x))!=0])
    
    if 'last.pt' in weights: epochs.append(epochs[-1]+1)
    print(sorted(epochs, reverse=True))
    for epoch in sorted(epochs, reverse=True):
        #if epoch > 12: continue
        if epoch==epochs[-1] and 'last.pt' in weights: model_path = os.path.join(model_root, "weights/last.pt")
        else: model_path = os.path.join(model_root, f"weights/epoch{epoch}.pt")
        model = YOLO(model_path)
        for img_filename in img_files:
            result = model.predict(img_filename, imgsz=1280, conf=confidence, iou=iou, device = device)[0] # 왜 0.6임?? 0.5로 고쳤음요 -> 다 고쳐도 됨
            # result = model.predict(img_filename)[0]
            output_folder = f"trafficlight-detect/eval_log/c{confidence}_i{iou}/{model_root.split('/')[-1]}/{epoch}/"
            #print(output_folder)
            os.makedirs(output_folder, exist_ok=True)
            
            img_ext = get_file_extension(img_filename)
            txt_filename = img_filename.replace(img_ext, "txt")
            txt_filename = txt_filename.replace("images","predictions")
            txt_filename = os.path.join(output_folder, txt_filename.split('/')[-1])
            #os.makedirs(output_folder)
            print(txt_filename)
            boxes = result.boxes 
            num_obj = len(boxes.cls)

            with open(txt_filename, 'w') as f1:
                for obj_idx in range(num_obj):
                    cls_id = int(boxes.cls[obj_idx])
                    cs = boxes.conf[obj_idx]
                    xywhn = boxes.xywhn[obj_idx] 
                    # class_id norm_center_x norm_center_y norm_w norm_h confidence_score
                    f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, xywhn[0], xywhn[1],xywhn[2],xywhn[3], cs))

                    # xywh = boxes.xywh[obj_idx]
                    # f1.write("%d %lf %lf %lf %lf %lf\n"%(cls_id, cs, xywh[0], xywh[1],xywh[2],xywh[3]))

            if num_obj == 0:
                print(txt_filename)
        del model

    print("ALL Done")

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-m", '--model_root', type=str, default="trafficlight-detect/runs/detect/yolov10x_sgd_24.10.30-17:44:09") 
    PARSER.add_argument("-g", '--gpu_id', type=str, default = "2")
    PARSER.add_argument("-c", '--confidence', type=float, default = 0.003)
    PARSER.add_argument("-i", '--iou', type=float, default = 0.5)
    main(**vars(PARSER.parse_args()))