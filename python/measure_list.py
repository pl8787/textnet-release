import sys

def Measure(pred_list, label_list):
    TP = 0.
    FP = 0.
    TN = 0.
    FN = 0.
    if len(pred_list)!=len(label_list):
        exit()
    for i in range(len(pred_list)):
        if pred_list[i] == 1 and label_list[i] == 1:
            TP += 1
        elif pred_list[i] == 1 and label_list[i] == 0:
            FP += 1
        elif pred_list[i] == 0 and label_list[i] == 1:
            FN += 1
        elif pred_list[i] == 0 and label_list[i] == 0:
            TN += 1
        else:
            exit()
    Total = TP + FP + TN + FN
    Acc = (TP+TN)/Total 
    if TP!=0:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
    else:
        P = R = F1 =0

    return {"Acc" : Acc, 
            "P" : P,
            "R" : R,
            "F1" : F1,
            "TP" : TP,
            "FP" : FP,
            "FN" : FN,
            "TN" : TN}

def MakePredList(score_list, theta):
    pred_list = [1 if s >=theta else 0 for s in score_list ]
    return pred_list

pred_list_file = open(sys.argv[1])
label_list_file = open(sys.argv[2])
cut_size = int(sys.argv[3])

pred_list = []
label_list = []

for line in pred_list_file:
    pred_list.append(float(line))

for line in label_list_file:
    label_list.append(int(line))

pred_list = pred_list[:cut_size]
label_list = label_list[:cut_size]

print Measure(pred_list, label_list)

