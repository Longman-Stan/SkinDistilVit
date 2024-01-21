import pickle
import numpy as np

import os, sys
os.chdir(sys.path[0])

pickle_path = "../experiments/matryoshka/MDistilViT_7/cm.pkl"

with open(pickle_path,'rb') as f:
    p = pickle.load(f)

cf_matr, y_preds, y_trues = p
cpreds, ctrues = [], []
cnc = np.zeros((2,2))

true_cancer = 0
true_non_cancer = 0

for i in range(len(y_preds)):
    if y_preds[i] in [0,2,7]:
        cpreds.append(1)
    else:
        cpreds.append(0)
    
    if y_trues[i] in [0,2,7]:
        ctrues.append(1)
        true_cancer += 1
    else:
        ctrues.append(0)
        true_non_cancer += 1
    
    if ctrues[i] == cpreds[i]:
        if cpreds[i] == 0:
            cnc[0,0] += 1
        else:
            cnc[1,1] += 1
    else:
        if cpreds[i] == 0:
            cnc[0,1] += 1
        else:
            cnc[1,0] += 1

    # if y_trues[i] in [0,2,7]:  
    #     if y_preds[i] in [0,2,7]:
    #         cnc[0][0] += 1
    #     else:
    #         cnc[1][0] += 1
    # else:
    #     if y_preds[i] in [ 1,3,4,5,6]:
    #         cnc[1][1] += 1
    #     else:
    #         cnc[0][1] += 1

print(true_cancer)
print(true_non_cancer)

print(cnc)
print("accuracy", (cnc[0,0]+cnc[1,1])/(cnc.sum()))
print("precision", cnc[0,0]/(cnc[0,0]+cnc[0,1]))
print("recall", cnc[0,0]/(cnc[0,0]+cnc[1,0]))

cf_sum = np.sum(cnc, axis=1)[:,None]

import torchmetrics
import torch
cpreds = torch.tensor(cpreds)
ctrues = torch.tensor(ctrues)
metric_acc = torchmetrics.Accuracy(task="binary",num_classes=2, top_k=1)
metric_precision = torchmetrics.Precision(task = "binary", num_classes=2, top_k=1)
metric_recall = torchmetrics.Recall(task = "binary", num_classes=2, top_k=1)
metric_f1 = torchmetrics.F1Score(task = "binary", num_classes=2, top_k=1)

print(metric_acc(cpreds, ctrues))
print(metric_precision(cpreds, ctrues))
print(metric_recall(cpreds, ctrues))
print(metric_f1(cpreds, ctrues))

import seaborn as sns
hmap = sns.heatmap(
    cnc/cf_sum, 
    annot=True, 
    fmt='.2%', 
    cmap='Blues', 
    xticklabels=["non-cancer","cancer"],
    yticklabels=["non-cancer","cancer"])
fig = hmap.get_figure()
fig.savefig("confmat.png")
