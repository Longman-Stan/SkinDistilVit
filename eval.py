from model.distilvit.modeling_distilvit import DistilViTForImageClassification
from transformers import ViTForImageClassification, ViTImageProcessor
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

model_type = DistilViTForImageClassification
checkpoint_weights_dest = "../experiments/matryoshka/MDistilViT_12/checkpoint-5000"
#checkpoint_weights_dest = "../experiments/DistilViT_full"
ck_path = os.path.join(checkpoint_weights_dest,"cm1.pkl")

if __name__ == "__main__":
    import sys
    os.chdir(sys.path[0])


def compute_matrix(model_type = DistilViTForImageClassification, checkpoint_weights_dest = "../experiments/DistilViT_trained2"):

    class VITPreprocess(ImageOnlyTransform):

        def __init__(self, feature_extractor, always_apply: bool = True, p: float = 1.0):
            super().__init__(always_apply, p)
            self.feature_extractor = feature_extractor

        def apply(self, img, **params):
            return self.feature_extractor(img, data_format="channels_last")['pixel_values'][0]

    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    train_transform = A.Compose(
        [
            VITPreprocess(feature_extractor),
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

    model = model_type.from_pretrained(
                            checkpoint_weights_dest,
                            num_labels = len(full_dataset.classes_names),
                            id2label = {str(i): c for i, c in enumerate(full_dataset.classes_names)},
                            label2id = {c: str(i) for i, c in enumerate(full_dataset.classes_names)}
                        ).to("cuda")
    model.eval()

    y_preds = []
    y_trues = []

    total_inference_time = 0
    for elem in tqdm.tqdm(valid_dataset):
        image, label = elem
        image = image.unsqueeze(0).to(model.device)
        tstart = time.time()
        logits = model(image)['logits']
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
    with open(os.path.join(checkpoint_weights_dest,"cm.pkl"), "wb") as f:
        pickle.dump((confusion_matrix, y_preds, y_trues), f)
else:
    with open(os.path.join(checkpoint_weights_dest,"cm.pkl"), "rb") as f:
        confusion_matrix, y_preds, y_trues = pickle.load(f)

categories = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
plt.figure(figsize=(9,8))
cf_sum = np.sum(confusion_matrix, axis=1)[:,None]
cf_sum[-1,0] = 1
#print(confusion_matrix)
#print(cf_sum)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
#print(confusion_matrix/cf_sum)

hmap = sns.heatmap(
    confusion_matrix/cf_sum, 
    annot=True, 
    fmt='.2%', 
    cmap='Blues', 
    xticklabels=categories,
    yticklabels=categories)
fig = hmap.get_figure()
fig.savefig(os.path.join(checkpoint_weights_dest,"confmat.png"))

y_preds = torch.tensor(np.concatenate([y_pred[None,...] for y_pred in y_preds]))
y_trues = torch.tensor(y_trues)

print(y_preds.shape, y_trues.shape)

import torchmetrics

metric_acc = torchmetrics.Accuracy(task="multiclass", num_classes=8, average="weighted", top_k=1)
metric_precision = torchmetrics.Precision(task = "multiclass", num_classes=8, average="weighted", top_k=1)
metric_recall = torchmetrics.Recall(task = "multiclass", num_classes=8, average="weighted", top_k=1)
metric_f1 = torchmetrics.F1Score(task = "multiclass", num_classes=8, average="weighted", top_k=1)
metric_bacc = torchmetrics.Recall(task = "multiclass", num_classes=8, average="macro", top_k=1)

print(y_preds[:10])
print(y_trues[:10])

acc = metric_acc(y_preds, y_trues)
precision = metric_precision(y_preds, y_trues)
recall = metric_recall(y_preds, y_trues)
f1 = metric_f1(y_preds, y_trues)
bacc = metric_bacc(y_preds, y_trues)

print(f"Acc: {acc}\nPrecision: {precision}\nRecall: {recall}\nF1: {f1}\nBacc {bacc}\n")