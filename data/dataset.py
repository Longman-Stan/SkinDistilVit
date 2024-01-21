from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import os

class ISICDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None, val_transform=None):

        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.val_transform = val_transform

        self.images_names = []
        self.labels = []

        for index, row in self.df.iterrows():
            self.images_names.append(row['image']+'.jpg')
            self.labels.append( row[1:].values.argmax())

        self.train_indices = list(range(len(self.images_names)))
        self.val_indices = []

        #self.images_names = [fname for fname in os.listdir(root_dir) if '.jpg' in fname]
        #self.labels = [ self.df[ self.df.image == fname.split('.',1)[0]].iloc[:,1:].values.argmax() for fname in self.images_names]

        assert len(self.images_names) == len(self.labels)        

        self.classes_names = self.df.columns[1:].values
        self.classes = len(np.unique(self.labels))

    def __len__(self):
        return len(self.images_names)

    def set_indices(self, train_indices, val_indices):
        self.train_indices = train_indices
        self.val_indices = val_indices

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.images_names[index])
        image = Image.open(img_name)
        image = np.array(image)
        label = self.labels[index]
        
        if index in self.train_indices:
            transform = self.transform
        elif index in self.val_indices:
            transform = self.val_transform
        else:
            print("E: !!!index not in lists")
            exit()
            
        if self.transform is not None:
            image = transform(image=image)['image']
        return image, label

if __name__ == '__main__':
    import os, sys
    import matplotlib.pyplot as plt
    os.chdir( sys.path[0])
    train_data = ISICDataset("../../ISIC2019/ISIC_2019_Training_GroundTruth.csv", "../../ISIC2019/TrainInput", None)
    data = pd.DataFrame(train_data.labels, columns=["label"])

    import json
    labels = {}
    for index, row in data.iterrows():
        label = train_data.classes_names[row['label']]
        if label not in labels:
            labels[label] = 0

        labels[label] += 1

    with open("classes_distribution.json","w") as f:
        json.dump(labels, f, indent=1)

    p = sns.countplot( data = data, x = 'label')
    p.set_xticklabels(train_data.classes_names[:-1])
    #p.bar_label(container=p.containers[0], labels=train_data.classes_names)
    plt.savefig("classes_distribution.png", transparent=True)