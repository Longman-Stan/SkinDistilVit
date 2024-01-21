from labeler import Labeler
import os, pickle, shutil

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


    print("MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK")  
    cls_nums = [0]*9
    
    for im_name,label in labels.items():
        cls_nums[label] += 1

    print( [x/len(labels) for x in cls_nums])
    
    res = [4522, 12875, 3323, 867, 2624, 239, 253, 628]

    print( [x/sum(res) for x in res])
    exit()

    dest_dir = "../../ISIC2019/CuratedTestInput"
    dest_csv = "../../ISIC2019/ISIC_2019_CuratedTest_groudTruth.csv"

    os.makedirs(dest_dir)

    new_labels = {}
    for im, cls in labels.items():
        if cls in [0,3,4,5,7]:
            new_labels[im] = cls
            shutil.copy( os.path.join(images_path, f"{im}.jpg"), dest_dir)


    labeler.write_csv(new_labels, dest_csv)
