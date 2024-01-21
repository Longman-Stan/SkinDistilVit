import os
from transformers import ViTForImageClassification, ViTFeatureExtractor
import cv2
import torch
import pickle
import time
import numpy as np
import tqdm

class Labeler:

    def __init__(self, labeler_model_path, device = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ViTForImageClassification.from_pretrained(labeler_model_path)
        self.model.eval()
        self.model.to(device)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def classify_image(self, image: np.array):
        image_pixels = self.feature_extractor(image, data_format="channels_first")['pixel_values'][0]
        image_pixels = torch.tensor(image_pixels[None,...]).to(self.model.device)
        res = self.model(image_pixels)['logits']
        return res

    def label_images(self, images_path, disable_tqdm=False):
        labels = {}
        assert os.path.isdir(images_path)
        for image in tqdm.tqdm(os.listdir(images_path), disable=disable_tqdm):
            term = image.rsplit('.',1)[1]
            if term not in ['png','jpg','jpeg']:
                continue
            im_path = os.path.join(images_path, image)
            im_name = image.split('.',1)[0]
            img = cv2.imread(im_path)
            res = self.classify_image(img)
            cls = res.argmax().cpu().numpy()
            labels[im_name] = cls
        return labels

    def write_csv(self, labels, dest_path, disable_tqdm=False):
        file = open(dest_path,"w")
        file.write("image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK\n")
        cls_nums = [0]*9
        for im_name,label in tqdm.tqdm(labels.items(), disable=disable_tqdm):
            file.write(f"{im_name},")
            res = [0.0]*9
            res[label] = 1.0
            cls_nums[label] += 1
            file.write(f"{','.join([str(x) for x in res])}\n")
        file.close()
 
if __name__ == "__main__":
    labeler = Labeler("../../experiments/ViT")

    images_path = "../../ISIC2019/ISIC_2019_Test_Input"

    labels_path = "../../ISIC2019/ISIC_2019_Test_labels.pkl"
    if os.path.isfile(labels_path):
        with open(labels_path, "rb") as f:
            labels = pickle.load(f)
    else:
        labels = labeler.label_images(images_path)
        with open(labels_path,"wb") as f:
            pickle.dump(labels, f)
    dest_path = "../../ISIC2019/ISIC_2019_Test_GroundTruth.csv"
    labeler.write_csv(labels, dest_path)

