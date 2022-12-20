import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_ids = [0]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import wandb
from torchvision import transforms
import videotransforms
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
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
from datasets.capg_csl_dataset_sample_sepa_multi_view_1 import CAPG_CSL as Dataset
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from MultiCueModel import MultiCueModel

# torch.cuda.set_device(0)
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

def confusion_matrix_fig(res, x_labels=None, y_labels=None, save_path=''):
    # res = res.cpu().numpy()
    df = pd.DataFrame(res, index=x_labels, columns=y_labels)
    sns.heatmap(df)
    # sns.heatmap(df, annot=True, fmt=".2f")
    if save_path:
        plt.savefig(save_path)
    plt.clf()

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

    dataset = Dataset('train', root, train_transforms, hand_transforms=test_transforms, num_classes=51)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)
    print('Train', len(dataset))
    view_list = ['camera_0', 'camera_1', 'camera_2', 'camera_3']
    phase_list = ['train','train','train','train']
    # phase_list = []
    val_dataset = {}
    val_dataloader = {}
    test_dataset = {}
    test_dataloader = {}
    for view in view_list:
        phase_list.append(f'val/{view}')
        val_dataset[f'val/{view}'] = Dataset('test', root, test_transforms, view_list=[view], num_classes=51)
        val_dataloader[f'val/{view}'] = torch.utils.data.DataLoader(val_dataset[f'val/{view}'] , batch_size=configs.batch_size, shuffle=True, num_workers=2,pin_memory=False)
        print(f'val/{view}', len(val_dataset[f'val/{view}']))
    # for view in view_list:
    #     phase_list.append(f'test/{view}')
    #     test_dataset[f'test/{view}'] = Dataset('test', root, test_transforms, view_list=[view], 
    #     class_list=[i for i in range(10, 21)])
    #     test_dataloader[f'test/{view}'] = torch.utils.data.DataLoader(test_dataset[f'test/{view}'] , batch_size=configs.batch_size, shuffle=True, num_workers=2,pin_memory=False)
    #     print(f'test/{view}', len(test_dataset[f'test/{view}']))

    dataloaders = {'train': dataloader, **val_dataloader, **test_dataloader}
    datasets = {'train': dataset, **val_dataset, **test_dataset}

    num_classes = dataset.num_classes
    
    # cue = ['full_rgb', 'right_hand', 'left_hand', 'face', 'pose']
    cue = ['full_rgb', 'right_hand', 'left_hand', 'face', 'pose']
    # supervised_cue = cue + ['late_fusion', 'local_align/multimodal'] + [f'local_align/{i}' for i in cue] + ['local_glocal_fusion']
    supervised_cue = cue + ['late_fusion', 'local_align/multimodal'] + [f'local_align/{i}' for i in cue] + ['local_glocal_fusion']
    model = MultiCueModel(cue, supervised_cue, num_classes, share_hand_model=False)

    if weights:
        print('loading weights {}'.format(weights))
        model.load_state_dict(torch.load(weights), strict=True)

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=device_ids)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0
    best_val_score_top5 = 0
    # train it
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1)
    max_epoch = 20
    configs.max_steps = max_epoch*len(dataset)*4//configs.batch_size
    scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=configs.max_steps,
            )
    while steps < configs.max_steps and epoch < max_epoch:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        val_score_dict = {"val_loss": 0.0}
        test_score_dict = {"test_loss": 0.0}
        # Each epoch has a training and validation phase
        for phase in phase_list:
            torch.cuda.empty_cache() 
            collected_vids = []

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
            confusion_matrix_float = np.zeros((num_classes, num_classes), dtype=np.float)
            confusion_matrix_top5 = np.zeros((num_classes, num_classes), dtype=np.int32)
            confusion_matrix_cue = {key: np.zeros((num_classes, num_classes), dtype=np.int32) for key in supervised_cue}
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue
                if phase == 'train':
                    full_rgb, labels, vid, pose, right_hand, left_hand, face = data
                    labels = labels.to(model.module.device, non_blocking=True)
                    inputs = {
                        'full_rgb': full_rgb,
                        'right_hand': right_hand,
                        'left_hand': left_hand,
                        'face': face,
                        'pose': pose,
                    }
                    ret = model(inputs)

                    loss = 0.0
                    scales = {}
                    for key in supervised_cue:
                        value = ret[key]
                        scales[f"{phase}/Scale/"+key] = value['scale'][0].item()
                        loss = loss + F.cross_entropy(value['logits'], labels)
                        logits = value['logits']
                        pred = torch.argmax(logits, dim=1)
                        for i in range(logits.shape[0]):
                            confusion_matrix_cue[key][labels[i].item(), pred[i].item()] += 1
                else:   
                    with torch.no_grad():
                        full_rgb, labels, vid, pose, right_hand, left_hand, face = data
                        assert full_rgb.shape[1] == 1
                        ret_list = []
                        labels = labels[:,0].to(model.module.device, non_blocking=True)
                        for view in range(full_rgb.shape[1]):
                            inputs = {
                              'full_rgb': full_rgb[:,view,:],
                                'right_hand': right_hand[:,view,:],
                                'left_hand': left_hand[:,view,:],
                                'face': face[:,view,:],
                                'pose': pose[:,view,:],
                            }
                            ret = model(inputs)
                            ret_list.append(ret)
                        ret = {}
                        for key in supervised_cue:
                            ret[key] = {
                                'logits': sum([i[key]['logits'] for i in ret_list])/float(len(ret_list)),
                                'scale': sum([i[key]['scale'] for i in ret_list])/float(len(ret_list)),
                            }
                            

                        loss = 0.0
                        scales = {}
                        for key in supervised_cue:
                            value = ret[key]
                            scales[f"{phase}/Scale/"+key] = value['scale'][0].item()
                            loss = loss + F.cross_entropy(value['logits'], labels)
                            logits = value['logits']
                            pred = torch.argmax(logits, dim=1)
                            for i in range(logits.shape[0]):
                                confusion_matrix_cue[key][labels[i].item(), pred[i].item()] += 1


                logits = ret['local_glocal_fusion']['logits']
                pred = torch.argmax(logits, dim=1)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                for i in range(logits.shape[0]):
                    confusion_matrix[labels[i].item(), pred[i].item()] += 1
                    if labels[i].item() in sorted_indices[i][:5]:
                        confusion_matrix_top5[labels[i].item(), labels[i].item()] += 1
                    else:
                        confusion_matrix_top5[labels[i].item(), 0] += 1

                    confusion_matrix_float[labels[i].item()] += logits[i].detach().cpu().numpy()

                loss = loss / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())

                if num_iter == num_steps_per_update and phase == 'train':
                    loss.backward()
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    scheduler.step()

                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        acc_cue = {f"{phase}/Accu/"+key: float(np.trace(confusion_matrix_cue[key])) / np.sum(confusion_matrix_cue[key]) for key in confusion_matrix_cue.keys()}
                        # print(torch.argmax(logits, dim=1), labels)
                        localtime = datetime.datetime.fromtimestamp(
                            int(time.time()), pytz.timezone("Asia/Shanghai")
                            ).strftime("%Y-%m-%d %H:%M:%S")
                        log1 = "[ " + localtime + " ] " + 'Epoch {} Step {} {} LR: {:.8f}  Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch, steps,
                        phase,
                        optimizer.param_groups[0]["lr"],
                        tot_loss / 10,
                        acc)      
                        log = '{} {} {}'.format(
                        log1, 
                        acc_cue,
                        scales)
                        print(log1)
                        with open(save_model + 'acc_train.txt', "a") as f:
                            f.writelines(log)
                            f.writelines("\n")
                        wandb.log({
                            "Epoch": epoch,
                            "Step": steps,
                            # f"{phase}/Loc Loss": tot_loc_loss / (10 * num_steps_per_update),
                            # f"{phase}/Cls Loss": tot_cls_loss / (10 * num_steps_per_update),
                            f"{phase}/Tot Loss": tot_loss / 10,
                            f"{phase}/Accu": acc,
                            **acc_cue,
                            **scales
                        })
                        tot_loss = 0.
            if 'val' in phase:
                save_path = save_model + f'confusion_matrix/{phase}/' 
                os.makedirs(save_path, exist_ok=True)
                confusion_matrix_fig(confusion_matrix, 
                x_labels=[i for i in range(num_classes)], 
                y_labels=[i for i in range(num_classes)], 
                save_path=save_path+f'{epoch}.png')
                confusion_matrix_fig(confusion_matrix_float, 
                x_labels=[i for i in range(num_classes)], 
                y_labels=[i for i in range(num_classes)], 
                save_path=save_path+f'{epoch}-float.png')
                confusion_matrix_fig(confusion_matrix_top5, 
                x_labels=[i for i in range(num_classes)], 
                y_labels=[i for i in range(num_classes)], 
                save_path=save_path+f'{epoch}-top5.png')

                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                val_score_top5 = float(np.trace(confusion_matrix_top5)) / np.sum(confusion_matrix_top5)
                
                acc_cue = {f"{phase}/Accu/"+key: float(np.trace(confusion_matrix_cue[key])) / np.sum(confusion_matrix_cue[key]) for key in confusion_matrix_cue.keys()}
                val_score_dict[phase + '/top1'] = val_score
                val_score_dict[phase + '/top5'] = val_score_top5
                val_score_dict['val_loss'] += tot_loss

                localtime = datetime.datetime.fromtimestamp(
                    int(time.time()), pytz.timezone("Asia/Shanghai")
                    ).strftime("%Y-%m-%d %H:%M:%S")
                log1 =  "[ " + localtime + " ] " + 'Epoch {} Step {} VALIDATION: {} LR: {:.8f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch, steps, phase, optimizer.param_groups[0]["lr"],
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score,
                                                                                                              )  
                log = '{} {}'.format(log1,acc_cue)
                print(log1)
                with open(save_model + 'acc_val.txt', "a") as f:
                    f.writelines(log)
                    f.writelines("\n")
                wandb.log({
                    "Epoch": epoch,
                    "Step": steps,
                    f"{phase}/Tot Loss": (tot_loss * num_steps_per_update) / num_iter,
                    f"{phase}/Accu": val_score,
                    **acc_cue,
                })
            if 'test' in phase:
                save_path = save_model + f'confusion_matrix/{phase}/' 
                os.makedirs(save_path, exist_ok=True)
                confusion_matrix_fig(confusion_matrix, 
                x_labels=[i for i in range(num_classes)], 
                y_labels=[i for i in range(num_classes)], 
                save_path=save_path+f'{epoch}.png')
                confusion_matrix_fig(confusion_matrix_float, 
                x_labels=[i for i in range(num_classes)], 
                y_labels=[i for i in range(num_classes)], 
                save_path=save_path+f'{epoch}-float.png')
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                acc_cue = {f"{phase}/Accu/"+key: float(np.trace(confusion_matrix_cue[key])) / np.sum(confusion_matrix_cue[key]) for key in confusion_matrix_cue.keys()}
                test_score_dict[phase] = val_score
                test_score_dict['test_loss'] += tot_loss

                localtime = datetime.datetime.fromtimestamp(
                    int(time.time()), pytz.timezone("Asia/Shanghai")
                    ).strftime("%Y-%m-%d %H:%M:%S")
                log1  = "[ " + localtime + " ] " + 'Epoch {} Step {} TEST: {} LR: {:.8f} Tot Loss: {:.4f} Accu :{:.4f} '.format(epoch, steps, phase, optimizer.param_groups[0]["lr"],
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score,
                                                                                                              
                                                                                                              )
                log = '{} {}'.format(log1, acc_cue)
                print(log1)
                with open(save_model + 'acc_val.txt', "a") as f:
                    f.writelines(log)
                    f.writelines("\n")
                wandb.log({
                    "Epoch": epoch,
                    "Step": steps,
                    f"{phase}/Tot Loss": (tot_loss * num_steps_per_update) / num_iter,
                    f"{phase}/Accu": val_score,
                    **acc_cue,
                })

        avg_val_score = sum([v for k, v in val_score_dict.items() if 'camera_' in k and 'top1' in k]) / 4.0
        avg_val_score_top5 = sum([v for k, v in val_score_dict.items() if 'camera_' in k and 'top5' in k]) / 4.0
        # avg_test_score = sum([v for k, v in test_score_dict.items() if 'camera_' in k]) / 4.0

        if avg_val_score > best_val_score:
            best_val_score_top5 = avg_val_score_top5
            best_val_score = avg_val_score
            # model_name = f"{save_model}nslt_{str(num_classes)}_{avg_val_score:.3f}_{epoch:05}_{avg_test_score:.3f}.pt"
            model_name = f"{save_model}nslt_{str(num_classes)}_{avg_val_score:.3f}_{avg_val_score_top5:.3f}_{epoch:05}.pt"
            torch.save(model.module.state_dict(), model_name)
            seq_model_list = glob(save_model + "nslt_*.pt")
            seq_model_list = sorted(seq_model_list)
            for path in seq_model_list[:-1]:
                os.remove(path)
                print('Remove:', path)
            print(model_name)
        # scheduler.step(val_score_dict['val_loss'] * num_steps_per_update / num_iter)

        if avg_val_score == best_val_score and avg_val_score_top5 > best_val_score_top5:
            best_val_score_top5 = avg_val_score_top5
            best_val_score = avg_val_score
            # model_name = f"{save_model}nslt_{str(num_classes)}_{avg_val_score:.3f}_{epoch:05}_{avg_test_score:.3f}.pt"
            model_name = f"{save_model}nslt_{str(num_classes)}_{avg_val_score:.3f}_{avg_val_score_top5:.3f}_{epoch:05}.pt"
            torch.save(model.module.state_dict(), model_name)
            seq_model_list = glob(save_model + "nslt_*.pt")
            seq_model_list = sorted(seq_model_list)
            for path in seq_model_list[:-1]:
                os.remove(path)
                print('Remove:', path)
            print(model_name)
        # scheduler.step(val_score_dict['val_loss'] * num_steps_per_update / num_iter)

        localtime = datetime.datetime.fromtimestamp(
            int(time.time()), pytz.timezone("Asia/Shanghai")
            ).strftime("%Y-%m-%d %H:%M:%S")
                    
        # log = "[ " + localtime + " ] " + 'Epoch {} Step {} VALIDATION: {} TEST: {}'.format(epoch, steps, val_score_dict, test_score_dict)
        log = "[ " + localtime + " ] " + 'Epoch {} Step {} VALIDATION: Top-1 ACC {}  Top-5 ACC {} {} '.format(epoch, 
        steps, 
        avg_val_score, 
        avg_val_score_top5,
        val_score_dict)
        print(save_model, log)
        with open(save_model + 'acc_val.txt', "a") as f:
            f.writelines(log)
            f.writelines("\n")
        wandb.log({
            "Epoch": epoch,
            "Step": steps,
            **val_score_dict,
            # **test_score_dict,
        })

def train_(root, save_model, weights):
    # WLASL setting
    
    mode = 'rgb'
    # root = {'word': '/raid_han/sign-dataset/wlasl/videos'}
    # root = {'word': ['/raid_han/signDataProcess/capg-csl-dataset/capg-csl-1-20']}

    os.makedirs(save_model, exist_ok=False)
    train_split = 'preprocess/nslt_100.json'

    config_file = 'configfiles/capg20.ini'

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

if __name__ == '__main__':

    exp_name = '1219-12-wo-aursv3-10'

    weights = None
    root = {'word': ['/raid_han/signDataProcess/capg-csl-dataset/capg-csl-1-20', '/raid_han/signDataProcess/capg-csl-dataset/capg-csl-21-100'], 'train': ['liya'], 'test': ['maodonglai']}
    save_model = f'logdir/train_{root["train"][0]}/{exp_name}'
    train_(root, save_model,weights)

    weights = None
    root = {'word': ['/raid_han/signDataProcess/capg-csl-dataset/capg-csl-1-20', '/raid_han/signDataProcess/capg-csl-dataset/capg-csl-21-100'], 'train': ['maodonglai'], 'test': ['liya']}
    save_model = f'logdir/train_{root["train"][0]}/{exp_name}'
    train_(root, save_model,weights)