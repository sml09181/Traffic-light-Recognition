import os 
import json
import csv
import torch
from argparse import ArgumentParser

from ultralytics import YOLO

# use gpu
def set_device(gpu_id):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu", 0)
    if device == "cuda": torch.cuda.empty_cache()
    print("Device:", device)
    return device

def main(
    model_path: str,
    gpu_id: str,
):
    
    # Load model
    model_name = model_path.split('/')[-3]
    model = YOLO(model_path)

    # Validate the model
    metrics = model.val(data="/trafficlight-detect/tld_2024.yaml", batch=1, imgsz=1280, device=set_device(gpu_id), save_dir=None)  # no arguments needed, dataset and settings remembered
    
    return metrics.box.map50

if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("-g", '--gpu_id', type=str, default="7")
    args = PARSER.parse_args()

    best_epoch = None
    best_score = float('-inf')
    scores = []  # To store the epoch and mAP50
    last_epoch = 7
    for epoch in range(3, last_epoch+1):  # Loop over epochs 5 to 18
        model_root = "/trafficlight-detect/runs/detect/yolov10x_sgd_24.10.26-08:29:12/weights"
        if epoch==last_epoch: model_path = os.path.join(model_root, "last.pt")
        else: model_path = os.path.join(model_root, f"epoch{epoch}.pt")
        
        score = main(model_path=model_path, gpu_id=args.gpu_id)
        scores.append([epoch, score])  # Store epoch and score in the list

        if score > best_score:
            best_score = score
            best_epoch = epoch

    # Save results to a CSV file
    model_name = '/'.join(model_path.split('/')[:-2])
    csv_save_path = os.path.join(model_name, "val_result.csv")
    with open(csv_save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "score"])  # Write header
        writer.writerows(scores)  # Write all epoch-score pairs

    print(f"Best Epoch: {best_epoch}, Best Score: {best_score}")