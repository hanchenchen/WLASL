import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
from collections import defaultdict
import os
import os.path as osp
from tqdm import tqdm
from glob import glob
import json
import time


def load_rgb_frames_from_video(video_path, img_path):
    vidcap = cv2.VideoCapture(video_path)

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(total_frames)
    for offset in range(total_frames):
        success, img = vidcap.read()
        if img is None:
            print(video_path, img_path, total_frames)
            continue
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)


        cv2.imwrite(f'{img_path}/{str(offset).zfill(8)}.png', img)
        # print(f'{img_path}/{str(offset).zfill(8)}.png', end='\r')
        # cv2.imwrite('wlasl.jpg', img)
    vidcap.release()



if __name__ == '__main__':
    time_cost = time.time() 
    content = json.load(open('WLASL_v0.3.json'))
    with open('../code/I3D/preprocess/nslt_2000.json', 'r') as f:
        data = json.load(f)
    root_dir = '/raid_han/sign-dataset/wlasl/videos'
    save_dir = '/raid_han/sign-dataset/wlasl/images'
    video_list = sorted(glob(f'{root_dir}/*'))
    print(len(video_list))
    for vid in tqdm(data.keys()):
        multi_view_frame_sequence_path = f'{root_dir}/{vid}.mp4' 
        write_img = multi_view_frame_sequence_path.replace(root_dir, save_dir).replace('.mp4', '')
        # if len(glob(f'{write_json}/*'))== len(glob(f'{multi_view_frame_sequence_path}/*')):
        #     # print(f"Skip {multi_view_frame_sequence_path} {len(glob(f'{write_json}/*'))}")
        #     continue
        os.makedirs(write_img, exist_ok=True)
        load_rgb_frames_from_video(multi_view_frame_sequence_path, write_img)
        print('complete:', multi_view_frame_sequence_path)
    time_cost = time.time() - time_cost
    print(f'Total time: {time_cost} seconds.')
    # print(f'Total frames: {frame_num} frames.')
    # print(f'Average time: {time_cost/frame_num} seconds per frame.')
    # keypoints_path = f"{root_dir}/2dkeypoints.json"
    # with open(keypoints_path, 'w') as keypointsfile:
    #     json.dump(keypoints_2d_dic, keypointsfile) 
    # keypoints_3d_path = f"{root_dir}/3dkeypoints.json"
    # with open(keypoints_3d_path, 'w') as keypoints_3d_file:
    #     json.dump(keypoints_3d_dic, keypoints_3d_file) 