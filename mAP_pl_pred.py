'''
Last Updated: 24/10/31
Remove concept of IOU and Bounding Box from mAP
Reference Code: https://github.com/herbwood/mAP/blob/main/mAP.ipynb
'''
import os
import json
import numpy as np
import pandas as pd
from collections import Counter

for epoch in range(16, 21):
    # EDIT THIS PATH: There are test predictions .txt files
    predictions_folder = f'trafficlight-detect/eval_log/tl_final/c0.0025_i0.5/yolov10x_sgd_24.10.30-17:44:09/{epoch}'
    model_path = 'trafficlight-detect/eval_log/tl_final/c0.0025_i0.5/yolov10x_sgd_24.10.30-17:44:09' # 위 prediction에 사용된 model path
    model_name = model_path.split('/')[-1] # result file에 기록용

    # PSUDO-LABEL PATH(src)
    json_src_path = 'trafficlight-detect/pl_prediction.json'

    # RESULT PATH(dst)
    xlsx_dst_path = 'trafficlight-detect/mAP_pl_predicted1.xlsx'

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

    def calculateAveragePrecision(rec, prec):
        mrec = [0] + [e for e in rec] + [1]
        mpre = [0] + [e for e in prec] + [0]

        for i in range(len(mpre)-1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])

        ii = []

        for i in range(len(mrec)-1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i+1)

        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i-1]) * mpre[i])
        
        return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

    def AP(detections, groundtruths, classes, method = 'AP'):
        # boxinfo = [filename, label, conf, (x1, y1, x2, y2)]
        result = []
        for c in classes:
            dects = [d for d in detections if d[1] == c]
            gts = [g for g in groundtruths if g[1] == c]
            npos = len(gts)
            dects = sorted(dects)

            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))

            det = Counter(cc[0] for cc in gts) # 각 이미지별 ground truth box의 수
            for d in range(len(dects)):
                if det[dects[d][0]] > 0:
                    TP[d] = 1
                    det[dects[d][0]] = det[dects[d][0]] -1
                else: FP[d] = 1

            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            [ap, mpre, mrec, ii] = calculateAveragePrecision(rec, prec)


            r = {
                'class' : c,
                'precision' : prec,
                'recall' : rec,
                'AP' : ap,
                'interpolated precision' : mpre,
                'interpolated recall' : mrec,
                'total positives' : npos,
                'total TP' : np.sum(TP),
                'total FP' : np.sum(FP)
            }

            result.append(r)
        return result

    def mAP(result):
        ap = 0
        for r in result:
            ap += r['AP']
        mAP = ap / len(result)
        return mAP

    detections = []
    for file in os.listdir(predictions_folder):
        if file.endswith('.txt'):
            filename = file[:-4]
            with open(os.path.join(predictions_folder, file), 'r') as f:
                lines = f.readlines()
                predicted = [int(x.split()[0]) for x in lines]
                predicted.sort()
            for pred in predicted:
                detections.append([filename, pred])  

    groundtruths = []
    with open(json_src_path, 'r') as json_file:
        pl_predicted = json.load(json_file)
    for k, v in pl_predicted.items():
        groundtruths += [[k, x] for x in v] # filename, label

    pl_files = list(set(list(pl_predicted.keys())))
    detections = [d for d in detections if d[0] in pl_files] # pseudo-label이 있는 prediction만 남기기

    result = AP(detections, groundtruths, list(classes.keys()))
    print(result)
    result_dict = {'model_name': model_name, 'epoch': epoch, 'mAP': 0}
    for k in list(classes.keys()):
        result_dict[k] = -1 # initialize with -1
    for r in result:
        if np.isnan(r['AP']): r['AP'] = 0
        result_dict[classes[r['class']]] = r['AP']
        print("{:^8} AP : {}".format(classes[r['class']], r['AP']))
    print("---------------------------")
    print(f"mAP : {mAP(result)}")
    result_dict['mAP'] = mAP(result)


    if os.path.isfile(xlsx_dst_path):
        # update previous result file
        df = pd.read_excel(xlsx_dst_path)
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    else: # 최초 실행시
        df = pd.DataFrame(data=result_dict, index = [0])
    df.to_excel(xlsx_dst_path, index=False)
print(df)
print("All Done")
