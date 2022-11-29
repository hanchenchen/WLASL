from glob import glob

from PIL import Image, ImageFile
from tqdm import tqdm
import os
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
src_dire = "/raid_han/signDataProcess/capg-csl-dataset/capg-csl-21-100/full-rgb"
tgt_dire = "/raid_han/signDataProcess/capg-csl-dataset/capg-csl-21-100/full-rgb"

for path in tqdm(glob(f"{src_dire}/*/liya/*/*/*.jpg")):
    if '.jpg' not in path:
        continue
    img = cv2.imread(path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
    # img = cv2.resize(img, dsize=(1920//4, 1280//4))
    print(img.shape)
    os.makedirs(os.path.dirname(path.replace(src_dire, tgt_dire)), exist_ok=True)
    cv2.imwrite(path.replace(src_dire, tgt_dire), img)
    cv2.imwrite('test.jpg', img)
    print(path.replace(src_dire, tgt_dire))
    # exit()
