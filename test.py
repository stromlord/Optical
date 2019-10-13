import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

img_path = "Deep_Learning/data/DRIVE/test/1st_manual/"
# for file in os.listdir(path):
#     os.rename(path + file, path + file.split('.')[0]+".png")
# img = Image.open('Deep_Learning/data/DRIVE/training/mask/21_training_mask.gif')
# print(np.shape(img))

imgs_name = os.listdir(img_path)
imgs_dir = [os.path.join(img_path, name) for name in imgs_name]
for path in imgs_dir:
    img = Image.open(path)
    dir = path.split(".")[0]
    img.save(dir + ".png")
# imgs = [np.asarray(cv2.imread(path)) for path in imgs_dir]
print(imgs_dir)
