#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 00:26:35 2020

@author: lengmengying
"""

# import argparse
import logging
# import time
# from graph import build_graph, segment_graph
# from random import random
# from PIL import Image, ImageFilter,ImageFont, ImageDraw
# from skimage import io
import numpy as np
import cv2
from main import get_segmented_image, label
from processing import zeros_like, equalizeHist, CLAHE, line_trans, gama
import os

if __name__ == '__main__':
    # path = "./data/390/test/sep/Run12/"
    # fileList = os.listdir(path)
    # pictureList = []
    # for f in fileList:
    #     if 'bmp' in f:
    #         pictureList.append(f)
    #
    # for i in range(len(pictureList)):
    for i in range(1):
        name = "142.jpeg"
        # img = cv2.imread("./data/390/test/142.jpeg", 0)
        img = cv2.imread("./data/" + name, 0)
        # img = cv2.imread(("data/390/test/sep/Run12/" + pictureList[i]), 0)
        src = img

        scale_percent = 20  # percent of original size
        width = int(src.shape[1] * scale_percent / 100)
        height = int(src.shape[0] * scale_percent / 100)
        dim = (width, height)

        method = {1: zeros_like, 2: equalizeHist, 3: CLAHE, 4: line_trans, 5: gama}
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M')
        logger = logging.getLogger(__name__)

        src = cv2.resize(src, dim, interpolation=cv2.INTER_AREA)
        src = method[3](src)

        cv2.imwrite(("./data/" + "prep_" + name), src)
        # cv2.imwrite(("./data/390/test/sep/_142.png"), src)
        # name = pictureList[i].split(".bmp")[0] + ".png"
        # cv2.imwrite(("./data/390/test/sep/" + name), src)
        src = cv2.GaussianBlur(src, (9,9), 1)

    #     for i in range(len(fileList)):
    #     name = "_142.png"
        for k in range(1):
            for s in range(1):
                for c in range(1):
                    K = 10 + 2 * k
                    neighbor = 8
                    sigma = 0.5
                    min_comp_size = 2000
                    K = 20

                    # text = "_s" + str(sigma) + "_K" + str(K) + "_minC" + str(min_comp_size)
                    # input_file = "./data/390/test/prep/" + name
                    # output_file = "./data/390/test/seg/" + name
                    # input_file = "./data/390/test/sep/" + name
                    # output_file = "./data/390/test/sep/" + name
                    input_file = "./data/" + "prep_"  + name
                    output_file = "./data/" + "seg_" + name
                    img = get_segmented_image(sigma, neighbor, K, min_comp_size, input_file, output_file, logger)

        # img = cv2.imread("data/390/test/seg/_142.png", 1)
        # img_o = cv2.imread("data/390/test/prep/_142.png", 1)
        # img_c = cv2.imread("data/390/test/seg/_142.png", 0)
        img = cv2.imread("./data/" + "seg_" + name, 1)
        img_o = cv2.imread("./data/" + "prep_"  + name, 1)
        img_c = cv2.imread("./data/" + "seg_" + name, 0)
        color = []
        h, w, c = img.shape
        for i in range(h):
            for j in range(w):
                if (img_c[i, j] == 0):
                    img[i, j] = [255, 255, 255]

        point_color = (180, 109, 40)
        thickness = 5
        point_size = 1
        # points_list = [(100,100),(20,120),(50,110),(70,110),(140,100)]
        # points_list = [(400,40),(350,60),(300,80),(250,100)]
        points_list = [(x, int(- 0.4 * x + 210)) for x in np.arange(220, 400, 20)]

        for point in points_list:
            r, g, b = img[point[1], point[0]]
            c = [r, g, b]
            if c not in color:
                color.append(c)
            cv2.circle(img_o, point, point_size, point_color, thickness)
            cv2.circle(img, point, point_size, point_color, thickness)
        # cv2.imwrite("data/390/test/seg/142_or.png", img_o)

        print(color)

        if (len(color) > 1):
            print("tracks are broken")
        else:
            print("no broken tracks")

        for i in range(h):
            for j in range(w):
                r, g, b = img[i, j]
                if ([r, g, b] not in color):
                    img[i, j] = [255, 255, 255]

        # cv2.imwrite("data/390/test/seg/142_pick.png", img)
        cv2.imwrite("data/seg_res_" + name, img)