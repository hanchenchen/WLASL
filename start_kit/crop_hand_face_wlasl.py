from glob import glob
import json
from PIL import Image, ImageFile
from tqdm import tqdm
import os
import cv2
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True
root = "/raid_han/sign-dataset/wlasl"
resized_dire = f"{root}/rgb-320"
src_dire = f"{root}/images"
face_dire = f"{root}/face-224x224"
left_hand_dire = f"{root}/left-hand-224x224"
right_hand_dire = f"{root}/right-hand-224x224"

length = 224
# print(glob(f"{src_dire}/*/*.png"))
for path in tqdm(glob(f"{src_dire}/*/*.png")):
    # if '.png' not in path:
    #     continue
    img = cv2.imread(path)
    w, h, c = img.shape
    # label, signer, record_time, view, img_name = path.split('/')[-5:]
    # key = f"{label}/{signer}/{record_time}"
    if not os.path.exists(path.replace('images', 'images-pose').replace('.png', '_keypoints.json')):
        continue
    with open(path.replace('images', 'images-pose').replace('.png', '_keypoints.json'), 'r') as f:
        pose = json.load(f)['people']
        if not pose:
            continue
        pose = pose[0]

    resized_img = img
    if w > 480 or h > 480:
        scale = max(w, h)
        resized_img = cv2.resize(resized_img, dsize=(int(h/scale*480), int(w/scale*480)))
    print(img.shape, resized_img.shape)
    
    os.makedirs(os.path.dirname(path.replace(src_dire, resized_dire)), exist_ok=True)
    cv2.imwrite(path.replace(src_dire, resized_dire), resized_img)
    # cv2.imwrite('resized_dire.jpg', resized_img)

    print(path.replace('images', 'images-pose').replace('.png', '_keypoints.json'))

    shoudler = abs(pose['pose_keypoints_2d'][2*3] - pose['pose_keypoints_2d'][5*3])
    pose_keypoints_2d = torch.tensor(pose['pose_keypoints_2d'])
    face_keypoints_2d = torch.tensor(pose['face_keypoints_2d']).reshape(-1, 3)[:, :2]
    hand_left_keypoints_2d = torch.tensor(pose['hand_left_keypoints_2d'])
    hand_right_keypoints_2d = torch.tensor(pose['hand_right_keypoints_2d'])
    l = int(max(torch.max(face_keypoints_2d[:, 0]) - torch.min(face_keypoints_2d[:, 0]), 
    torch.max(face_keypoints_2d[:, 1]) - torch.min(face_keypoints_2d[:, 1]))*1.5)
    
    
    face_center_x = min(max(pose_keypoints_2d[0], l//2), h - l//2)
    face_center_y = min(max(pose_keypoints_2d[1], l//2), w - l//2)
    face_img = img[int(face_center_y)-l//2:int(face_center_y)+l//2, int(face_center_x)-l//2:int(face_center_x)+l//2]
    if face_img is not None:
        os.makedirs(os.path.dirname(path.replace(src_dire, face_dire)), exist_ok=True)
        face_img = cv2.resize(face_img, dsize=(length, length))
        cv2.imwrite(path.replace(src_dire, face_dire), face_img)
        # cv2.imwrite('face_img.jpg', face_img)

    left_center_x = min(max(hand_left_keypoints_2d[9*3], l//2), h - l//2)
    left_center_y = min(max(hand_left_keypoints_2d[9*3+1], l//2), w - l//2)
    left_img = img[int(left_center_y)-l//2:int(left_center_y)+l//2, int(left_center_x)-l//2:int(left_center_x)+l//2]
    if left_img is not None:
        os.makedirs(os.path.dirname(path.replace(src_dire, left_hand_dire)), exist_ok=True)
        left_img = cv2.resize(left_img, dsize=(length, length))
        cv2.imwrite(path.replace(src_dire, left_hand_dire), left_img)
        # cv2.imwrite('left_img.jpg', left_img)

    right_center_x = min(max(hand_right_keypoints_2d[9*3], l//2), h - l//2)
    right_center_y = min(max(hand_right_keypoints_2d[9*3+1], l//2), w - l//2)
    right_img = img[int(right_center_y)-l//2:int(right_center_y)+l//2, int(right_center_x)-l//2:int(right_center_x)+l//2]
    if right_img is not None:
        os.makedirs(os.path.dirname(path.replace(src_dire, right_hand_dire)), exist_ok=True)
        right_img = cv2.resize(right_img, dsize=(length, length))
        cv2.imwrite(path.replace(src_dire, right_hand_dire), right_img)
        # cv2.imwrite('right_img.jpg', right_img)

    # exit()
