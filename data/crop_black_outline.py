import cv2
import tqdm
import os
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import shutil

def crop_black_outline(image, plot_image = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    cv2.imwrite("mask.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = None
    max_area = -1
    contour_idx = -1
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            max_contour = c
            contour_idx = idx

    if plot_image:
        cont_draw = cv2.drawContours(deepcopy(image), contours, contour_idx, color=[0,255,0], thickness=2)
        cv2.imwrite("contours.png", cont_draw)

    #assert len(contours) == 1, f"Multiple contours {len(contours)}"

    x,y,w,h = cv2.boundingRect(max_contour)
    return image[y:y+h,x:x+w]



if __name__ == "__main__":

    import sys
    os.chdir(sys.path[0])

    im_dir = "../../ISIC2019/ISIC_2019_Training_Input"
    outdir = "../../ISIC2019/TrainInput"
    
    idx = 0

    for imname in tqdm.tqdm(os.listdir(im_dir)):
        if '.jpg' not in imname:
            continue
        infile = os.path.join(im_dir, imname)
        outfile = os.path.join(outdir, imname)# f"{imname.rsplit('.',1)[0]}.png")
        
        idx += 1

        if os.path.isfile(outfile):
            continue
            im = cv2.imread(infile)
            cutout = cv2.imread(outfile)
            if im.shape == cutout.shape and np.allclose(im, cutout):
                continue

            if idx < 2650:
                continue

            # b = np.ones_like(im, dtype=np.uint8)*255
            # h,w = cutout.shape[:2]
            # b[:h,:w,:] = cutout
            # res = np.concatenate((im, b), axis=1)
            # cv2.imshow("ceva",res)
            # cv2.waitKey()
        
        im = cv2.imread(infile)
        res = crop_black_outline(im, plot_image=False)
        
        if res.shape == im.shape and np.allclose(res, im):
            shutil.copy(infile, outfile)
        else:
            cv2.imwrite(outfile, res)
    
