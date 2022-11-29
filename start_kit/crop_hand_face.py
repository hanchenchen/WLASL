from glob import glob
import json
from PIL import Image, ImageFile
from tqdm import tqdm
import os
import cv2
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
src_dire = "/raid_han/signDataProcess/capg-csl-rgb-21-100"
face_dire = f"{src_dire}/face"
left_hand_dire = f"{src_dire}/left-hand"
right_hand_dire = f"{src_dire}/right-hand"

with open(src_dire+'/2dkeypoints.json', 'r') as f:
    kpts_2d = json.load(f)
length = 224

for path in tqdm(glob(f"{src_dire}/*/*/*/*/*.jpg")):
    if '.jpg' not in path:
        continue
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)

    w, h, c = img.shape
    label, signer, record_time, view, img_name = path.split('/')[-5:]
    key = f"{label}/{signer}/{record_time}"
    pose = kpts_2d[key][view][img_name]
    shoudler = abs(pose['pose_keypoints_2d'][2*3] - pose['pose_keypoints_2d'][5*3])
    pose_keypoints_2d = torch.tensor(pose['pose_keypoints_2d'])
    face_keypoints_2d = torch.tensor(pose['face_keypoints_2d']).reshape(-1, 3)[:, :2]
    hand_left_keypoints_2d = torch.tensor(pose['hand_left_keypoints_2d'])
    hand_right_keypoints_2d = torch.tensor(pose['hand_right_keypoints_2d'])
    face_center_x = min(max(pose_keypoints_2d[0], length//2), h - length//2)
    face_center_y = min(max(pose_keypoints_2d[1], length//2), w - length//2)
    face_img = img[int(face_center_y)-length//2:int(face_center_y)+length//2, int(face_center_x)-length//2:int(face_center_x)+length//2]
    if face_img is not None:
        os.makedirs(os.path.dirname(path.replace(src_dire, face_dire)), exist_ok=True)
        cv2.imwrite(path.replace(src_dire, face_dire), face_img)
        cv2.imwrite('face_img.jpg', face_img)

    left_center_x = min(max(hand_left_keypoints_2d[9*3], length//2), h - length//2)
    left_center_y = min(max(hand_left_keypoints_2d[9*3+1], length//2), w - length//2)
    left_img = img[int(left_center_y)-length//2:int(left_center_y)+length//2, int(left_center_x)-length//2:int(left_center_x)+length//2]
    if left_img is not None:
        os.makedirs(os.path.dirname(path.replace(src_dire, left_hand_dire)), exist_ok=True)
        cv2.imwrite(path.replace(src_dire, left_hand_dire), left_img)
        cv2.imwrite('left_img.jpg', left_img)

    right_center_x = min(max(hand_right_keypoints_2d[9*3], length//2), h - length//2)
    right_center_y = min(max(hand_right_keypoints_2d[9*3+1], length//2), w - length//2)
    right_img = img[int(right_center_y)-length//2:int(right_center_y)+length//2, int(right_center_x)-length//2:int(right_center_x)+length//2]
    if right_img is not None:
        os.makedirs(os.path.dirname(path.replace(src_dire, right_hand_dire)), exist_ok=True)
        cv2.imwrite(path.replace(src_dire, right_hand_dire), right_img)
        # img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
        cv2.imwrite('right_img.jpg', right_img)

    # exit()
