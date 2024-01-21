import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import cv2

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

import torchvision
from torchvision.transforms.functional import InterpolationMode
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from data.dataset import ISICDataset

from datasets import load_metric

import os, sys
os.chdir(sys.path[0])


train_transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=600),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=90, p=0.75),
        A.RandomCrop(height=380, width=380),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2()
    ]
)

full_dataset = ISICDataset("../ISIC2019/ISIC_2019_Training_GroundTruth.csv", "../ISIC2019/TrainInput", transform=train_transform)
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size

SPLIT_SEED = 42
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4

train_indices, valid_indices, _, _ = train_test_split( 
                                range(len(full_dataset)), 
                                full_dataset.labels, 
                                stratify=full_dataset.labels,
                                test_size=valid_size,
                                random_state=42)

train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

class ISICModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_b6(pretrained=True)
        block_list = list(self.model.children())
        self.in_features = 2304
        block_list = block_list[:-1]
        self.model = nn.Sequential(*block_list)
        self.classifier = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.in_features, len(full_dataset.classes_names))
        )
    
    def forward(self, x):
        embeddings = self.model(x)
        embeddings=embeddings.view(-1,self.in_features)
        pred = self.classifier(embeddings)
        return pred

class ISICPL(pl.LightningModule):

    def __init__(self, isic_model):
        super(ISICPL, self).__init__()
        self.model = isic_model
        self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(full_dataset.classes_names), average="weighted", top_k=1)
        self.metric_precision = torchmetrics.Precision(task = "multiclass", num_classes=len(full_dataset.classes_names), average="weighted", top_k=1)
        self.metric_recall = torchmetrics.Recall(task = "multiclass", num_classes=len(full_dataset.classes_names), average="weighted", top_k=1)
        self.metric_f1 = torchmetrics.F1Score(task = "multiclass", num_classes=len(full_dataset.classes_names), average="weighted", top_k=1)
        self.metric_bacc = torchmetrics.Recall(task = "multiclass", num_classes=len(full_dataset.classes_names), average="macro", top_k=1)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = LR

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam( self.model.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                             epochs=EPOCHS,
                                                             steps_per_epoch=len(train_dataloader),
                                                             max_lr=LR)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_index):
        images = batch[0].float()
        labels = batch[1].long()
        output = self.model(images)
        loss = self.criterion(output, labels)
        score = self.metric(output.argmax(1), labels)
        precision = self.metric_precision(output.argmax(1), labels)
        recall = self.metric_recall(output.argmax(1), labels)
        f1 = self.metric_f1(output.argmax(1), labels)
        bacc = self.metric_bacc(output.argmax(1), labels)
        logs = {"train_loss": loss, "train_acc" : score, "train_prec": precision, "train_rec": recall, "train_f1": f1, "train_bacc": bacc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar = True, logger= True
        )
        return loss

    def validation_step(self, batch, batch_index):
        images = batch[0].float()
        labels = batch[1].long()
        output = self.model(images)
        loss = self.criterion(output, labels)
        score = self.metric(output.argmax(1), labels)
        precision = self.metric_precision(output.argmax(1), labels)
        recall = self.metric_recall(output.argmax(1), labels)
        f1 = self.metric_f1(output.argmax(1), labels)
        bacc = self.metric_bacc(output.argmax(1), labels)
        logs = {"valid_loss": loss, "valid_acc" : score, "valid_prec": precision, "valid_rec": recall, "valid_f1": f1, "bacc": bacc}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar = True, logger= True
        )
        return loss

if __name__ == "__main__":
    isic_model = ISICModel()
    pl_model = ISICPL(isic_model)

    logger = pl.loggers.CSVLogger(save_dir="../logs/", name="b6_baseline")
    checkpoint_callback = ModelCheckpoint(monitor="valid_loss", save_top_k = 1, save_last = True, save_weights_only=True, filename='{epoch:02d}-{valid_loss:.4f}-{valid_acc:.4f}', verbose=True, mode='min')
    trainer = pl.Trainer(max_epochs=EPOCHS,logger=logger, gpus=[0], callbacks=[checkpoint_callback])

    #trainer.fit(pl_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    checkpoint = checkpoint_callback.best_model_path
    print(checkpoint)

    import matplotlib.pyplot as plt
    metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')

    train_acc = metrics['train_acc'].dropna().reset_index(drop=True)
    valid_acc = metrics['valid_acc'].dropna().reset_index(drop=True)
        
    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_acc, color="r", marker="o", label='train/acc')
    plt.plot(valid_acc, color="b", marker="x", label='valid/acc')
    plt.ylabel('Accuracy', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='lower right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/acc.png')

    train_loss = metrics['train_loss'].dropna().reset_index(drop=True)
    valid_loss = metrics['valid_loss'].dropna().reset_index(drop=True)

    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_loss, color="r", marker="o", label='train/loss')
    plt.plot(valid_loss, color="b", marker="x", label='valid/loss')
    plt.ylabel('Loss', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='upper right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/loss.png')\

    train_prec = metrics['train_prec'].dropna().reset_index(drop=True)
    valid_prec = metrics['valid_prec'].dropna().reset_index(drop=True)
        
    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_prec, color="r", marker="o", label='train/precision')
    plt.plot(valid_prec, color="b", marker="x", label='valid/precision')
    plt.ylabel('Precision', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='lower right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/prec.png')

    train_rec = metrics['train_rec'].dropna().reset_index(drop=True)
    valid_rec = metrics['valid_rec'].dropna().reset_index(drop=True)
        
    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_rec, color="r", marker="o", label='train/recall')
    plt.plot(valid_rec, color="b", marker="x", label='valid/recall')
    plt.ylabel('Recall', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='lower right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/rec.png')

    train_f1 = metrics['train_f1'].dropna().reset_index(drop=True)
    valid_f1 = metrics['valid_f1'].dropna().reset_index(drop=True)
        
    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_f1, color="r", marker="o", label='train/f1')
    plt.plot(valid_f1, color="b", marker="x", label='valid/f1')
    plt.ylabel('F1', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='lower right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/f1.png')

    train_bacc = metrics['train_bacc'].dropna().reset_index(drop=True)
    valid_bacc = metrics['valid_acc'].dropna().reset_index(drop=True)
        
    fig = plt.figure(figsize=(7, 6))
    plt.grid(True)
    plt.plot(train_f1, color="r", marker="o", label='train/bacc')
    plt.plot(valid_f1, color="b", marker="x", label='valid/bacc')
    plt.ylabel('Bacc', fontsize=24)
    plt.xlabel('Epoch', fontsize=24)
    plt.legend(loc='lower right', fontsize=18)
    plt.savefig(f'{trainer.logger.log_dir}/bacc.png')