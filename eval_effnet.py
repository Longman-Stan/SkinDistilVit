from model.distilvit.modeling_distilvit import DistilViTForImageClassification
from transformers import ViTFeatureExtractor, ViTForImageClassification
import cv2
import torch
from data.dataset import ISICDataset
from sklearn.model_selection import train_test_split

from torch.utils.data import Subset
import albumentations as A
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
import torch.functional as F
import time

import tqdm
from sklearn import metrics

import seaborn as sns
import os
import numpy as np
import pickle 
import matplotlib.pyplot as plt

from train_effnet import ISICModel, ISICPL
import torchvision

model_type = torchvision.models.efficientnet_b6
checkpoint_weights_dest = "../logs/b6_baseline/version_6/checkpoints/epoch=19-valid_loss=0.5215-valid_acc=0.8161.ckpt"
base_path = "../logs/b6_baseline/version_6"
ck_path = os.path.join("../logs/b6_baseline/version_6/checkpoints","cm.pkl")


def compute_matrix(model_type = DistilViTForImageClassification, checkpoint_weights_dest = "../experiments/DistilViT_trained2"):

    train_transform = A.Compose(
        [
            A.Resize(528,528),     
            A.ToFloat(),   
            ToTensorV2()
        ]
    )

    full_dataset = ISICDataset("../ISIC2019/ISIC_2019_Training_GroundTruth.csv", "../ISIC2019/TrainInput", transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    SPLIT_SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 20

    train_indices, valid_indices, _, _ = train_test_split( 
                                    range(len(full_dataset)), 
                                    full_dataset.labels, 
                                    stratify=full_dataset.labels,
                                    test_size=valid_size,
                                    random_state=42)

    valid_dataset = Subset(full_dataset, valid_indices)

    isic_model = ISICModel()
    model = ISICPL(isic_model)
    state_dict = torch.load(checkpoint_weights_dest)['state_dict']
    model.load_from_checkpoint(checkpoint_weights_dest, isic_model = isic_model)
    model.cuda()
    model.eval()

    y_preds = []
    y_trues = []

    total_inference_time = 0
    for elem in tqdm.tqdm(valid_dataset):
        image, label = elem
        image = image.unsqueeze(0).to(model.device)
        tstart = time.time()
        logits = model(image)
        total_inference_time += time.time() - tstart
        y_pred = torch.argmax(logits).cpu().numpy()
        y_preds.append(y_pred)
        y_trues.append(label)

    print(total_inference_time)

    confusion_matrix = metrics.confusion_matrix(y_trues, y_preds, labels=[0,1,2,3,4,5,6,7,8])
    print(confusion_matrix)

    return confusion_matrix, y_preds, y_trues


if not os.path.isfile(ck_path):
    confusion_matrix, y_preds, y_trues = compute_matrix(model_type, checkpoint_weights_dest)
    with open(ck_path, "wb") as f:
        pickle.dump((confusion_matrix, y_preds, y_trues), f)
else:
    with open(ck_path, "rb") as f:
        confusion_matrix, y_preds, y_trues = pickle.load(f)

categories = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
plt.figure(figsize=(9,8))
cf_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
print(cf_sum)
cf_sum[-1] = 1
hmap = sns.heatmap(
    confusion_matrix/cf_sum, 
    annot=True, 
    fmt='.2%', 
    cmap='Blues', 
    xticklabels=categories,
    yticklabels=categories)
fig = hmap.get_figure()
fig.savefig(os.path.join(base_path,"confmat.png"))

y_preds = torch.tensor(np.concatenate([y_pred[None,...] for y_pred in y_preds]))
y_trues = torch.tensor(y_trues)

print(y_preds.shape, y_trues.shape)

import torchmetrics

metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8, average="weighted", top_k=1)
metric_precision = torchmetrics.Precision(task = "multiclass", num_classes=8, average="weighted", top_k=1)
metric_recall = torchmetrics.Recall(task = "multiclass", num_classes=8, average="weighted", top_k=1)
metric_f1 = torchmetrics.F1Score(task = "multiclass", num_classes=8, average="weighted", top_k=1)
metric_bacc = torchmetrics.Recall(task = "multiclass", num_classes=8, average="macro", top_k=1)

acc = metric_acc(y_preds, y_trues)
precision = metric_precision(y_preds, y_trues)
recall = metric_recall(y_preds, y_trues)
f1 = metric_f1(y_preds, y_trues)
bacc = metric_bacc(y_preds, y_trues)

print(f"Acc: {acc}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nBacc {bacc}\n")

logdir = base_path
import pandas as pd
import matplotlib.pyplot as plt
metrics = pd.read_csv(f'{logdir}/metrics.csv')

train_acc = metrics['train_acc'].dropna().reset_index(drop=True)
valid_acc = metrics['valid_acc'].dropna().reset_index(drop=True)
    
fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_acc, color="r", marker="o", label='train/acc')
plt.plot(valid_acc, color="b", marker="x", label='valid/acc')
plt.ylabel('Accuracy', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='lower right', fontsize=18)
plt.savefig(f'{logdir}/acc.png')

train_loss = metrics['train_loss'].dropna().reset_index(drop=True)
valid_loss = metrics['valid_loss'].dropna().reset_index(drop=True)

fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_loss, color="r", marker="o", label='train/loss')
plt.plot(valid_loss, color="b", marker="x", label='valid/loss')
plt.ylabel('Loss', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='upper right', fontsize=18)
plt.savefig(f'{logdir}/loss.png')\

train_prec = metrics['train_prec'].dropna().reset_index(drop=True)
valid_prec = metrics['valid_prec'].dropna().reset_index(drop=True)
    
fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_prec, color="r", marker="o", label='train/precision')
plt.plot(valid_prec, color="b", marker="x", label='valid/precision')
plt.ylabel('Precision', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='lower right', fontsize=18)
plt.savefig(f'{logdir}/prec.png')

train_rec = metrics['train_rec'].dropna().reset_index(drop=True)
valid_rec = metrics['valid_rec'].dropna().reset_index(drop=True)
    
fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_rec, color="r", marker="o", label='train/recall')
plt.plot(valid_rec, color="b", marker="x", label='valid/recall')
plt.ylabel('Recall', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='lower right', fontsize=18)
plt.savefig(f'{logdir}/rec.png')

train_f1 = metrics['train_f1'].dropna().reset_index(drop=True)
valid_f1 = metrics['valid_f1'].dropna().reset_index(drop=True)
    
fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_f1, color="r", marker="o", label='train/f1')
plt.plot(valid_f1, color="b", marker="x", label='valid/f1')
plt.ylabel('F1', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='lower right', fontsize=18)
plt.savefig(f'{logdir}/f1.png')

train_bacc = metrics['train_bacc'].dropna().reset_index(drop=True)
valid_bacc = metrics['bacc'].dropna().reset_index(drop=True)
    
fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_bacc, color="r", marker="o", label='train/bacc')
plt.plot(valid_bacc, color="b", marker="x", label='valid/bacc')
plt.ylabel('Bacc', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='lower right', fontsize=18)
plt.savefig(f'{logdir}/bacc.png')

print(f"Acc: {max(valid_acc)}\nPrecision: {max(valid_prec)}\nRecall: {max(valid_rec)}\nF1: {max(valid_f1)}\nBacc {max(valid_bacc)}\n")