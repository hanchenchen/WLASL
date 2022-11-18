import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import wandb
from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from einops import rearrange
import datetime
import time
from glob import glob
import pytz
# from datasets.nslt_dataset import NSLT as Dataset
# from datasets.nslt_dataset import NSLT as Dataset
from datasets.capg_csl_dataset import CAPG_CSL as Dataset

from video_swin_transformer import SwinTransformer3D

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset('train', root, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset('test', root, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    # setup the model
    if mode == 'flow':
        video_swin = InceptionI3d(400, in_channels=2)
        video_swin.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        # video_swin = InceptionI3d(400, in_channels=3)
        # video_swin.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
        video_swin = SwinTransformer3D(
            pretrained='checkpoints/swin/swin_tiny_patch244_window877_kinetics400_1k.pth',
            pretrained2d=False,
            patch_size=(2,4,4),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(8,7,7),
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.2,
            patch_norm=True
        )
        video_swin.init_weights()
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        video_swin.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=4)
        video_swin.swin_head = nn.Linear(768, dataset.num_classes)
        video_swin.pos_emb = nn.Parameter(torch.randn(1, 16, 768))
        video_swin.scale = nn.Parameter(torch.ones(1))

    num_classes = dataset.num_classes
    # video_swin.replace_logits(num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        video_swin.load_state_dict(torch.load(weights))

    video_swin.cuda()
    video_swin = nn.DataParallel(video_swin)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(video_swin.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0
    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    while steps < configs.max_steps and epoch < 400:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            collected_vids = []

            if phase == 'train':
                video_swin.train(True)
            else:
                video_swin.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()
                # print(152, inputs.shape)
                x = rearrange(inputs, 'n c d h w -> n c d h w')
                # print(154, x.shape)
                x = video_swin(x)
                # print(156, x.shape)
                x = rearrange(x, 'n d h w c -> n d (h w) c')
                x = x.mean(dim=2)
                # print(158, x.shape)
                x = x + video_swin.module.pos_emb
                # print(160, x.shape)
                x = video_swin.module.temporal_model(x)
                # print(162, x.shape)
                per_frame_logits = video_swin.module.swin_head(x[:, 0, :])
                # print(164, per_frame_logits.shape)
                # x = rearrange(x, 'n d c -> n c d')

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.cross_entropy(per_frame_logits*video_swin.module.scale, labels)

                pred = torch.argmax(per_frame_logits, dim=1)
                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[labels[i].item(), pred[i].item()] += 1

                loss = cls_loss / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        # print(torch.argmax(per_frame_logits, dim=1), labels)
                        localtime = datetime.datetime.fromtimestamp(
                            int(time.time()), pytz.timezone("Asia/Shanghai")
                            ).strftime("%Y-%m-%d %H:%M:%S")
                                 
                        log = "[ " + localtime + " ] " + 'Epoch {} Step {} {} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch, steps,
                        phase,
                        tot_loss / 10,
                        acc)
                        print(log)
                        with open(save_model + 'acc_train.txt', "a") as f:
                            f.writelines(log)
                            f.writelines("\n")
                        wandb.log({
                            "Epoch": epoch,
                            # f"{phase}/Loc Loss": tot_loc_loss / (10 * num_steps_per_update),
                            # f"{phase}/Cls Loss": tot_cls_loss / (10 * num_steps_per_update),
                            f"{phase}/Tot Loss": tot_loss / 10,
                            f"{phase}/Accu": acc,
                            f"{phase}/Scale": video_swin.module.scale.item(),
                        })
                        tot_loss = 0.
            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score:
                    best_val_score = val_score
                    model_name = save_model + "nslt_" + str(num_classes) + "_" + '_%3f_' % val_score + str(steps).zfill(6) + '.pt'

                    torch.save(video_swin.module.state_dict(), model_name)
                    seq_model_list = glob(save_model + "nslt_*.pt")
                    seq_model_list = sorted(seq_model_list)
                    for path in seq_model_list[:-1]:
                        os.remove(path)
                        print('Remove:', path)
                    print(model_name)
                localtime = datetime.datetime.fromtimestamp(
                    int(time.time()), pytz.timezone("Asia/Shanghai")
                    ).strftime("%Y-%m-%d %H:%M:%S")
                            
                log = "[ " + localtime + " ] " + 'Epoch {} VALIDATION: {} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch, phase,
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score
                                                                                                              )
                print(log)
                with open(save_model + 'acc_val.txt', "a") as f:
                    f.writelines(log)
                    f.writelines("\n")

                scheduler.step(tot_loss * num_steps_per_update / num_iter)
                wandb.log({
                    "Epoch": epoch,
                    f"{phase}/Tot Loss": (tot_loss * num_steps_per_update) / num_iter,
                    f"{phase}/Accu": val_score,
                    f"{phase}/Scale": video_swin.module.scale.item(),
                })


if __name__ == '__main__':
    # WLASL setting
    
    mode = 'rgb'
    # root = {'word': '/raid_han/sign-dataset/wlasl/videos'}
    root = {'word': '/raid_han/signDataProcess/capg-csl-resized'}

    save_model = '1118-06-wo-proj-dim=768-05'
    os.makedirs(save_model, exist_ok=True)
    train_split = 'preprocess/nslt_100.json'

    # weights = 'checkpoints/nslt_100_004170_0.010638.pt'
    weights = None
    config_file = 'configfiles/asl100.ini'

    configs = Config(config_file)
    print(root, train_split)
    wandb.init(
        name=save_model,
        project="islr",
        entity="hanchenchen",
        config=configs,
        id=wandb.util.generate_id(),
        # group=_config.work_dir.split('/')[-4],
        # job_type=_config.work_dir.split("/")[-3],
    )
    run(configs=configs, mode=mode, root=root, save_model=save_model+'/', train_split=train_split, weights=weights)
