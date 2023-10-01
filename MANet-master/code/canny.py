import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

pic_path = '../dataset/DUTS/DUTS-TR/DUTS-TR-Mask/'
for pics in os.listdir(pic_path):
    img = cv.imread(pic_path + pics,0)
    edges = cv.Canny(img, 100, 200)
    cv.imwrite('../dataset/DUTS/DUTS-TR/DUTS-TR-Edge/' + pics, edges)

