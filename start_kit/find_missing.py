import os
import json
import cv2

filenames = set(sorted(os.listdir('/raid_han/sign-dataset/wlasl/videos')))

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
            missing_ids.append(video_id)
        else:
            video_path = os.path.join('/raid_han/sign-dataset/wlasl/videos', video_id + '.mp4')
            # print(video_path)
            videocap = cv2.VideoCapture(video_path)
            if not int(videocap.get(cv2.CAP_PROP_FRAME_COUNT)):
                none_ids.append(video_id)
                missing_ids.append(video_id)
                os.remove(video_path)
                print(video_path)
            else:
                
            videocap.release()
print(total_video, len(filenames))

with open('missing.txt', 'w') as f:
    f.write('\n'.join(missing_ids))
with open('none.txt', 'w') as f:
    f.write('\n'.join(none_ids))

