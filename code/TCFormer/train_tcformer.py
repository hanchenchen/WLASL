import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from timm.models import create_model

from torchvision import transforms
import videotransforms

import numpy as np
from configs import Config
from pytorch_i3d import InceptionI3d
from einops import rearrange

# from datasets.nslt_dataset import NSLT as Dataset
from datasets.nslt_dataset import NSLT as Dataset
from tcformer_module.tcformer import tcformer_light
from tcformer_module.mta_block import MTA
from tconv import TemporalConv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, help="rgb or flow")
parser.add_argument("-save_model", type=str)
parser.add_argument("-root", type=str)
parser.add_argument("--num_class", type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(
    configs,
    mode="rgb",
    root="/ssd/Charades_v1_rgb",
    train_split="charades/charades.json",
    save_model="",
    weights=None,
):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose(
        [
            videotransforms.RandomCrop(224),
            videotransforms.RandomHorizontalFlip(),
        ]
    )
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, "train", root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_dataset = Dataset(train_split, "test", root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    dataloaders = {"train": dataloader, "test": val_dataloader}
    datasets = {"train": dataset, "test": val_dataset}

    # setup the model
    if mode == "flow":
        tcformer = InceptionI3d(400, in_channels=2)
        tcformer.load_state_dict(torch.load("weights/flow_imagenet.pt"))
    else:
        # tcformer = InceptionI3d(400, in_channels=3)
        # tcformer.load_state_dict(torch.load('weights/rgb_imagenet.pt'))
        # create_model
        # tcformer = create_model(
        #     'tcformer',
        #     pretrained=True,
        #     drop_rate=0.1,
        #     clip_grad=None,
        # )
        tcformer = tcformer_light(drop_rate=0.1, clip_grad=None)
        tcformer.init_weights(
            pretrained="weights/tcformer_light-edacd9e5_20220606.pth"
        )
        tcformer.mta = MTA(
            in_channels=[64, 128, 320, 512],
            out_channels=256,
            start_level=0,
            add_extra_convs="on_input",
            num_outs=1,
            use_sr_layer=True,
        )
        tcformer.pooler = nn.Sequential(
            nn.AvgPool2d(4, stride=4),
        ) 
        tcformer.proj = nn.Sequential(
            nn.Linear(256*14*14, 1024),
        ) 
        tcformer.tconv = TemporalConv(
            input_size=1024, hidden_size=1024, num_classes=dataset.num_classes
        )

    num_classes = dataset.num_classes
    # tcformer.replace_logits(num_classes)

    if weights:
        print("loading weights {}".format(weights))
        tcformer.load_state_dict(torch.load(weights))

    # device_ids = [0, 1]
    # tcformer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(tcformer)
    tcformer.cuda()

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(tcformer.parameters(), lr=lr, weight_decay=weight_decay)
    
    num_steps_per_update = configs.update_per_step  # accum gradient
    steps = 0
    epoch = 0

    best_val_score = 0
    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.3
    )
    while steps < configs.max_steps and epoch < 400:  # for epoch in range(num_epochs):
        print("Step {}/{}".format(steps, configs.max_steps))
        print("-" * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            collected_vids = []

            if phase == "train":
                tcformer.train(True)
            else:
                tcformer.train(False)  # Set model to evaluate mode

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
                if (
                    data == -1
                ):  # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data

                # wrap them in Variable
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()

                n, c, d, h, w = inputs.shape
                x = rearrange(inputs, "n c d h w -> (n d) c h w")
                x = tcformer(x)
                x = tcformer.mta(x)[0]
                x = tcformer.pooler(x)
                x = x.reshape(n, d, -1)
                x = tcformer.proj(x)
                x = rearrange(x, "n d c -> n c d")
                x = tcformer.tconv(x)
                x = rearrange(x["conv_logits"], "d n c -> n c d")

                # upsample to input size
                per_frame_logits = F.upsample(x, t, mode="linear")

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(
                    torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0]
                )
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[
                        torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()
                    ] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == "train":
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(
                            confusion_matrix
                        )
                        print(
                            "Epoch {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}".format(
                                epoch,
                                phase,
                                tot_loc_loss / (10 * num_steps_per_update),
                                tot_cls_loss / (10 * num_steps_per_update),
                                tot_loss / 10,
                                acc,
                            )
                        )
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.0
            if phase == "test":
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score or epoch % 2 == 0:
                    best_val_score = val_score
                    model_name = (
                        save_model
                        + "nslt_"
                        + str(num_classes)
                        + "_"
                        + str(steps).zfill(6)
                        + "_%3f.pt" % val_score
                    )

                    torch.save(tcformer.state_dict(), model_name)
                    print(model_name)

                print(
                    "VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}".format(
                        phase,
                        tot_loc_loss / num_iter,
                        tot_cls_loss / num_iter,
                        (tot_loss * num_steps_per_update) / num_iter,
                        val_score,
                    )
                )

                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == "__main__":
    # WLASL setting
    mode = "rgb"
    root = {"word": "/raid_han/sign-dataset/wlasl/videos"}

    save_model = "checkpoints/"
    train_split = "preprocess/nslt_100.json"

    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    weights = None
    config_file = "configfiles/asl100.ini"

    configs = Config(config_file)
    print(root, train_split)
    run(
        configs=configs,
        mode=mode,
        root=root,
        save_model=save_model,
        train_split=train_split,
        weights=weights,
    )
