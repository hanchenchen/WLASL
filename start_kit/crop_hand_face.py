from glob import glob
import json
from PIL import Image, ImageFile
from tqdm import tqdm
import os
import cv2
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
root = "/raid_han/signDataProcess/capg-csl-dataset/capg-csl-1-20"
src_dire = f"{root}/rgb-1920x1280"
resized_dire = f"{root}/rgb-320x320"
face_dire = f"{root}/face-224x224"
left_hand_dire = f"{root}/left-hand-224x224"
right_hand_dire = f"{root}/right-hand-224x224"

length = 224
resized_length = 320

for path in tqdm(glob(f"{src_dire}/*/*/*/*/*.jpg")):
    if '.jpg' not in path:
        continue
    img = cv2.imread(path)
    print(img.shape)
    w, h, c = img.shape
    # label, signer, record_time, view, img_name = path.split('/')[-5:]
    # key = f"{label}/{signer}/{record_time}"
    pose = json.load(open(path.replace('rgb-1920x1280', 'openpose-res').replace('.jpg', '_keypoints.json'), 'r'))['people'][0]

    # img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
    resized_img = cv2.resize(img, dsize=(1920//4, 1280//4))
    body_center_0 = int(pose['pose_keypoints_2d'][1*2 + 1])//4
    if body_center_0 + resized_length//2 > 1920//4:
        body_center_0 = 1920//4 - resized_length//2
    if body_center_0 - resized_length//2 < 0:
        body_center_0 = resized_length//2

    resized_img = resized_img[:, int(body_center_0)-resized_length//2:int(body_center_0)+resized_length//2]
    os.makedirs(os.path.dirname(path.replace(src_dire, resized_dire)), exist_ok=True)
    cv2.imwrite(path.replace(src_dire, resized_dire), resized_img)
    cv2.imwrite('resized_dire.jpg', resized_img)

    continue
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
        cv2.imwrite('right_img.jpg', right_img)

    # exit()
