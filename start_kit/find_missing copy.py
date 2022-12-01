import os
import json
import cv2

filenames = set(os.listdir('/raid_han/sign-dataset/wlasl/videos'))

content = json.load(open('WLASL_v0.3.json'))

missing_ids = []
none_ids = []
total_video = 0
for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']
        total_video += 1
        if video_id + '.mp4' not in filenames:
            missing_ids.append(video_id + f' {inst["url"]}')
            # print(video_id + f' {inst["url"]}')
        else:
            video_path = os.path.join('/raid_han/sign-dataset/wlasl/videos', video_id + '.mp4')
            videocap = cv2.VideoCapture(video_path)
            print(video_id, int(videocap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if int(videocap.get(cv2.CAP_PROP_FRAME_COUNT)) < 10:
                none_ids.append(video_id + str(videocap.get(cv2.CAP_PROP_FRAME_COUNT)))
            videocap.release()
print(total_video, len(filenames))

with open('missing.txt', 'w') as f:
    f.write('\n'.join(missing_ids))
with open('none.txt', 'w') as f:
    f.write('\n'.join(none_ids))

