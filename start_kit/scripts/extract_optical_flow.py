import numpy as np
import glob
import os
import pdb
import random
import sys
import time
import warnings
import cv2
import pandas
import six
import torch
from glob import glob
from tqdm import tqdm
import cv2
from numpy import *
from pylab import *
dataset_root = '/raid_han/sign-dataset/capg-csl-resized'
output_dire = '/raid_han/sign-dataset/capg-csl-resized-optical-flow'

os.makedirs(output_dire, exist_ok = True)
def draw_flow(im, flow, step=16):
    """在间隔分开的像素采样点处绘制光流"""
    h, w = im.shape[:2]
    y, x = mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # 创建线的终点
    lines = vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = int32(lines)

    # 创建图像并绘制
    vis = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis
import multiprocessing
def func(path):
    tvl1 = cv2.optflow.createOptFlow_DualTVL1()
    img_list = sorted(glob(f'{path}/*'))
    print(img_list[0])
    prvs = cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_BGR2GRAY)
    os.makedirs(path.replace('capg-csl-resized', 'capg-csl-resized-optical-flow'), exist_ok = True)
    for i in tqdm(range(1, len(img_list))): # images
        frame = cv2.cvtColor(cv2.imread(img_list[i]), cv2.COLOR_BGR2GRAY)
        flow = tvl1.calc(prvs, frame, None)
        prvs = frame
        # print(frame.shape)
        # cv2.imshow('Optical flow', draw_flow(frame, flow))
        # if cv2.waitKey(10) == 27:
        #     break
        save_path = img_list[i].replace('capg-csl-resized', 'capg-csl-resized-optical-flow')[:-4] + '.npy'
        np.save(save_path, flow)
        # print(flow==np.load(save_path))
        # cv2.imwrite(img_list[i].replace('fullFrame-256x256px', 'optical_flow')[:-4]+'_u.png', flow[:, :, 0])
        # cv2.imwrite(img_list[i].replace('fullFrame-256x256px', 'optical_flow')[:-4]+'_v.png', flow[:, :, 1])
        # temp = cv2.imread(img_list[i].replace('fullFrame-256x256px', 'optical_flow')[:-4]+'_v.png')
        # print(flow[:, :, 1] == temp)
        # print(np.max(temp), np.min(temp))
    print(path)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=10)
    inputs = sorted(glob(f'{dataset_root}/*/*/*/*'))
    pool.map(func, inputs)
    pool.close()
    pool.join()