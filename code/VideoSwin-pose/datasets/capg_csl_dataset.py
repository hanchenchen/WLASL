import json
import math
import os
import os.path
import random
from glob import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data_utl
import torchvision.transforms as T


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def pose_filtering(video_path):

    frame_paths = sorted(glob(f"{video_path}/*.jpg"))
    
    label, signer, record_time, view = video_path.split('/')[-4:]
    key = f"{label}/{signer}/{record_time}"

    start_index = 0
    cur_img = frame_paths[0]
    cur_pose = json.load(open(cur_img.replace('rgb-480x320', 'openpose-res').replace('.jpg', '_keypoints.json'), 'r'))['people'][0]['pose_keypoints_2d']
    start_left_wrist_x = cur_pose[4*3]
    start_left_wrist_y = cur_pose[4*3+1]
    start_right_wrist_x = cur_pose[7*3]
    start_right_wrist_y = cur_pose[7*3+1]
    for img_index in range(1, len(frame_paths)):
        cur_img = frame_paths[img_index]
        cur_pose = json.load(open(cur_img.replace('rgb-480x320', 'openpose-res').replace('.jpg', '_keypoints.json'), 'r'))['people'][0]['pose_keypoints_2d']
        left_wrist_x = cur_pose[4*3]
        left_wrist_y = cur_pose[4*3+1]
        right_wrist_x = cur_pose[7*3]
        right_wrist_y = cur_pose[7*3+1]
        x = max(abs(left_wrist_x - start_left_wrist_x), abs(right_wrist_x - start_right_wrist_x))
        y = max(abs(left_wrist_y - start_left_wrist_y), abs(right_wrist_y - start_right_wrist_y))

        # print(img_list[img_index], 'x', left_wrist_x - start_left_wrist_x, right_wrist_x - start_right_wrist_x)
        # print(img_list[img_index], 'y', left_wrist_y - start_left_wrist_y, right_wrist_y - start_right_wrist_y)
        if max(x, y) > 40:
            start_index = img_index
            break
    end_index = len(frame_paths) - 1
    cur_img = frame_paths[end_index]
    cur_pose = json.load(open(cur_img.replace('rgb-480x320', 'openpose-res').replace('.jpg', '_keypoints.json'), 'r'))['people'][0]['pose_keypoints_2d']
    start_left_wrist_x = cur_pose[4*3]
    start_left_wrist_y = cur_pose[4*3+1]
    start_right_wrist_x = cur_pose[7*3]
    start_right_wrist_y = cur_pose[7*3+1]
    for img_index in range(end_index, -1, -1):
        cur_img = frame_paths[img_index]
        cur_pose = json.load(open(cur_img.replace('rgb-480x320', 'openpose-res').replace('.jpg', '_keypoints.json'), 'r'))['people'][0]['pose_keypoints_2d']
        left_wrist_x = cur_pose[4*3]
        left_wrist_y = cur_pose[4*3+1]
        right_wrist_x = cur_pose[7*3]
        right_wrist_y = cur_pose[7*3+1]
        x = max(abs(left_wrist_x - start_left_wrist_x), abs(right_wrist_x - start_right_wrist_x))
        y = max(abs(left_wrist_y - start_left_wrist_y), abs(right_wrist_y - start_right_wrist_y))

        # print(img_list[img_index], 'x', left_wrist_x - start_left_wrist_x, right_wrist_x - start_right_wrist_x)
        # print(img_list[img_index], 'y', left_wrist_y - start_left_wrist_y, right_wrist_y - start_right_wrist_y)
        if max(x, y) > 40:
            end_index = img_index
            break
    # print(video_path, start_index, end_index, len(frame_paths))
    return frame_paths[start_index:end_index]

def get_pose(frame_paths):

    pose_path = frame_paths[0].replace('rgb-480x320', 'openpose-res').replace('.jpg', '_keypoints.json')
    pose = json.load(open(pose_path, 'r'))['people'][0]
    shoudler = (pose['pose_keypoints_2d'][2*3] - pose['pose_keypoints_2d'][5*3])**2
    shoudler += (pose['pose_keypoints_2d'][2*3+1] - pose['pose_keypoints_2d'][5*3+1])**2
    shoudler = shoudler**0.5
    center_x = (pose['pose_keypoints_2d'][2*3] + pose['pose_keypoints_2d'][5*3])/2.0
    center_y = (pose['pose_keypoints_2d'][2*3+1] + pose['pose_keypoints_2d'][5*3+1])/2.0
    center = torch.tensor([center_x, center_y])
    poses = []
    for poses_idx in range(len(frame_paths)):
        pose_path = frame_paths[poses_idx].replace('rgb-480x320', 'openpose-res').replace('.jpg', '_keypoints.json')
        pose = json.load(open(pose_path, 'r'))['people'][0]
        pose_keypoints_2d = torch.tensor(pose['pose_keypoints_2d']).reshape(-1, 3)[:, :2].reshape(-1)
        face_keypoints_2d = torch.tensor(pose['face_keypoints_2d']).reshape(-1, 3)[:, :2].reshape(-1)
        hand_left_keypoints_2d = torch.tensor(pose['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2].reshape(-1)
        hand_right_keypoints_2d = torch.tensor(pose['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2].reshape(-1)
        kpts = torch.cat([pose_keypoints_2d,face_keypoints_2d,hand_left_keypoints_2d,hand_right_keypoints_2d], dim=0)
        kpts = (kpts.reshape(-1, 2) - center.reshape(-1, 2)).reshape(-1)/shoudler
        poses.append(
            kpts
        )
    return poses

def load_rgb_frames(frame_paths, sampler, img_norm, img_index_map, index_view_img_map, split, pose_list):
    frames = []
    right_hand = []
    left_hand = []
    face = []
    poses = []
    frames_indexes = list(sampler({'start_index': 0, 'total_frames': len(frame_paths)})['frame_inds'])
    right_hand_indexes = frames_indexes
    left_hand_indexes = frames_indexes
    face_indexes = frames_indexes
    poses_indexes = frames_indexes
    print('together', end = '')
    label, signer, record_time, view, img_name = frame_paths[0].split('/')[-5:]

    for frames_idx, right_hand_idx, left_hand_idx, face_idx, poses_idx in zip(frames_indexes, right_hand_indexes, left_hand_indexes, face_indexes, poses_indexes):
        img = cv2.imread(frame_paths[frames_idx])[:, :, [2, 1, 0]]
        right_hand_path = frame_paths[right_hand_idx].replace('rgb-480x320', 'right-hand-224x224')
        right_hand_img = cv2.imread(right_hand_path)[:, :, [2, 1, 0]]

        left_hand_path = frame_paths[left_hand_idx].replace('rgb-480x320', 'left-hand-224x224')
        left_hand_img = cv2.imread(left_hand_path)[:, :, [2, 1, 0]]

        face_path = frame_paths[face_idx].replace('rgb-480x320', 'face-224x224')
        face_hand_img = cv2.imread(face_path)[:, :, [2, 1, 0]]
        
        # img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
        # cv2.imwrite('test.jpg', img)
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        # cv2.imwrite('test.jpg', img)
        # img = cv2.resize(img, dsize=(w//4, h//4)) # 1920//4, 1280//4
        img = (img / 255.) * 2 - 1
        right_hand_img = (right_hand_img / 255.) * 2 - 1
        left_hand_img = (left_hand_img / 255.) * 2 - 1
        face_hand_img = (face_hand_img / 255.) * 2 - 1
        # img = torch.tensor(img).float()
        # img = img.permute(2, 1, 0)
        # img = img_norm(img)
        # img = img.permute(1, 2, 0)
        frames.append(np.asarray(img, dtype=np.float32))
        right_hand.append(np.asarray(right_hand_img, dtype=np.float32))
        left_hand.append(np.asarray(left_hand_img, dtype=np.float32))
        face.append(np.asarray(face_hand_img, dtype=np.float32))
        poses.append(
            pose_list[poses_idx]
        )
    return np.asarray(frames, dtype=np.float32), np.asarray(right_hand, dtype=np.float32), np.asarray(left_hand, dtype=np.float32), np.asarray(face, dtype=np.float32), poses


def load_rgb_frames_from_video(vid_root, vid, start, num, resize=(256, 256)):
    video_path = os.path.join(vid_root, vid + '.mp4')

    vidcap = cv2.VideoCapture(video_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(0, min(num, int(total_frames - start)), 2):
        success, img1 = vidcap.read()
        success, img2 = vidcap.read()
        if img2 is not None:
            img = random.choice([img1, img2])
        else:
            img = img1

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(image_dir, vid, start, num):
    frames = []
    for i in range(start, start + num):
        imgx = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'x.jpg'), cv2.IMREAD_GRAYSCALE)
        imgy = cv2.imread(os.path.join(image_dir, vid, vid + '-' + str(i).zfill(6) + 'y.jpg'), cv2.IMREAD_GRAYSCALE)

        w, h = imgx.shape
        if w < 224 or h < 224:
            d = 224. - min(w, h)
            sc = 1 + d / min(w, h)
            imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
            imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

        imgx = (imgx / 255.) * 2 - 1
        imgy = (imgy / 255.) * 2 - 1
        img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def record_view_map(video_path, img_index_map, index_view_img_map):
    frame_paths = sorted(glob(f"{video_path}/*.jpg"))
    
    label, signer, record_time, view = video_path.split('/')[-4:]
    for i in range(len(frame_paths)):
        label, signer, record_time, view, img_name = frame_paths[i].split('/')[-5:]
        
        img_index_map[img_name] = i
        if record_time not in index_view_img_map:
            index_view_img_map[record_time] = {}
        if i not in index_view_img_map[record_time]:
            index_view_img_map[record_time][i] = {}
        index_view_img_map[record_time][i][view] = frame_paths[i]
    return img_index_map, index_view_img_map

def view_aug(frame_path, img_index_map, index_view_img_map, view):
    label, signer, record_time, _, img_name = frame_path.split('/')[-5:]
    img_idx = img_index_map[img_name]
    # print('input', frame_path, 'output', index_view_img_map[record_time][img_idx][view])
    frame_path = index_view_img_map[record_time][img_idx][view]
    return frame_path

def make_dataset(split, root, num_classes, 
    view_list,
    class_list,
    ):

    dataset = []
    vid_root_list = root['word']
    img_index_map = {}
    index_view_img_map = {}

    i = 0
    for vid_root in vid_root_list:
        for path in sorted(glob(f"{vid_root}/rgb-480x320/*/*/*/camera_*")):
            if path[-8:-1] != 'camera_':
                continue
            label, signer, record_time, view = path.split('/')[-4:]
            if view not in view_list:
                continue
            if int(label) not in class_list:
                continue
            if int(label) > num_classes:
                continue
            # if split == 'train':
            #     if view not in ['camera_0', 'camera_1', 'camera_3']:
            #         continue
            # else:
            #     if view not in ['camera_2']:
            #         continue
            # print(label, signer, record_time, view)
            if split == 'train':
                if signer not in root[split]:
                    continue
                img_index_map, index_view_img_map = record_view_map(path, img_index_map, index_view_img_map)
            else:
                if signer not in root[split]:
                    continue
            label = int(label)
            valid_frame_path = pose_filtering(path)
            valid_pose_list = get_pose(valid_frame_path)
            dataset.append((label, path, valid_frame_path, valid_pose_list))
            i += 1

    return dataset, img_index_map, index_view_img_map


def get_num_class(split_file):
    classes = set()

    content = json.load(open(split_file))

    for vid in content.keys():
        class_id = content[vid]['action'][0]
        classes.add(class_id)

    return len(classes)


class SampleFrames:
    """Sample frames from the video.
    MMACTION

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 frame_uniform=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.frame_uniform = frame_uniform
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips)
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(
                    num_frames - ori_clip_len + 1, size=self.num_clips))
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def get_seq_frames(self, num_frames):
        """
        Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
        Given the video index, return the list of sampled frame indexes.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        seg_size = float(num_frames - 1) / self.clip_len
        seq = []
        for i in range(self.clip_len):
            start = int(np.round(seg_size * i))
            end = int(np.round(seg_size * (i + 1)))
            if not self.test_mode:
                seq.append(random.randint(start, end))
            else:
                seq.append((start + end) // 2)

        return np.array(seq)

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        if self.frame_uniform:  # sthv2 sampling strategy
            assert results['start_index'] == 0
            frame_inds = self.get_seq_frames(total_frames)
        else:
            clip_offsets = self._sample_clips(total_frames)
            frame_inds = clip_offsets[:, None] + np.arange(
                self.clip_len)[None, :] * self.frame_interval
            frame_inds = np.concatenate(frame_inds)

            if self.temporal_jitter:
                perframe_offsets = np.random.randint(
                    self.frame_interval, size=len(frame_inds))
                frame_inds += perframe_offsets

            frame_inds = frame_inds.reshape((-1, self.clip_len))
            if self.out_of_bound_opt == 'loop':
                frame_inds = np.mod(frame_inds, total_frames)
            elif self.out_of_bound_opt == 'repeat_last':
                safe_inds = frame_inds < total_frames
                unsafe_inds = 1 - safe_inds
                last_ind = np.max(safe_inds * frame_inds, axis=1)
                new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
                frame_inds = new_inds
            else:
                raise ValueError('Illegal out_of_bound option.')

            start_index = results['start_index']
            frame_inds = np.concatenate(frame_inds) + start_index

        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str

class CAPG_CSL(data_utl.Dataset):

    def __init__(self, split, root, 
    transforms=None, 
    hand_transforms=None, 
    num_classes=51, 
    class_list=[],
    view_list=['camera_0', 'camera_1', 'camera_2', 'camera_3']):
        self.num_classes = num_classes
        if not class_list:
            class_list = [i for i in range(num_classes)]
        else:
            class_list = class_list
        if not hand_transforms:
            hand_transforms = transforms

        self.transforms = transforms
        self.root = root
        self.split = split
        self.total_frames = 32
        self.sample_frame = SampleFrames(clip_len=1, num_clips=self.total_frames, test_mode=split!="train")
        self.img_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.data, self.img_index_map, self.index_view_img_map = make_dataset(split, root, 
        num_classes=self.num_classes, 
        view_list=view_list,
        class_list=class_list,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        label, video_path, frame_paths, pose_list = self.data[index]
        imgs, right_hand, left_hand, face, poses = load_rgb_frames(frame_paths, self.sample_frame, self.img_norm, self.img_index_map, self.index_view_img_map, self.split, pose_list)

        imgs = self.transforms(imgs)
        right_hand = self.transforms(right_hand)
        left_hand = self.transforms(left_hand)
        face = self.transforms(face)

        ret_lab = torch.tensor(label)
        ret_pose = torch.stack(poses, dim=0)
        ret_img = video_to_tensor(imgs)
        right_hand = video_to_tensor(right_hand)
        left_hand = video_to_tensor(left_hand)
        face = video_to_tensor(face)

        return ret_img, ret_lab, index, ret_pose, right_hand, left_hand, face

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5:
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                else:
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs

        return padded_imgs, label

    @staticmethod
    def pad_wrap(imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                pad = imgs[:min(num_padding, imgs.shape[0])]
                k = num_padding // imgs.shape[0]
                tail = num_padding % imgs.shape[0]

                pad2 = imgs[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
                else:
                    padded_imgs = np.concatenate([imgs, pad2], axis=0)
        else:
            padded_imgs = imgs

        return padded_imgs, label

