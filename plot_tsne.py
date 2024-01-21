from sklearn.manifold import TSNE
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
checkpoint_weights_dest = "../experiments/matryoshka/MDistilViT_7"
#checkpoint_weights_dest = "../experiments/DistilViT_full"
ck_path = os.path.join(checkpoint_weights_dest,"cm1.pkl")

if __name__ == "__main__":
    import sys
    os.chdir(sys.path[0])


def compute_embeds(model_type = DistilViTForImageClassification, checkpoint_weights_dest = "../experiments/DistilViT_trained2"):

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

    embeds = []
    gt = []

    with torch.no_grad():
        for elem in tqdm.tqdm(valid_dataset):
            image, label = elem
            image = image.unsqueeze(0).to(model.device)
            hidden_states = model(image, output_hidden_states=True)['hidden_states']
            normed_state = model.distilvit.layernorm(hidden_states[-1])
            normed_state = normed_state[:,0,:].squeeze()
            embeds.append(normed_state)
            gt.append(label)
        
    return embeds, gt

embeds, gt = compute_embeds(model_type, checkpoint_weights_dest)
embeds = torch.stack(embeds).cpu().numpy()
tsne = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=69).fit_transform(embeds)

plt.figure(figsize=(10,10))
plt.axis('off')
plt.scatter(tsne[:,0], tsne[:,1], s=50, c=gt)
plt.savefig(f"results_tsne.png",transparent=True)