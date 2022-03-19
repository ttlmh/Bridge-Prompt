import os
import torch.nn as nn
from datasets import Breakfast_FRAMES, GTEA_FRAMES, SALADS_FRAMES
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
from dotmap import DotMap
import pprint
from utils.text_prompt import *
from pathlib import Path
from utils.Augmentation import *
import numpy as np


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/breakfast/breakfast_exfm.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--dataset', default='breakfast')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments, dropout=config.network.drop_out,
                                       emb_dropout=config.network.emb_dropout, if_proj=config.network.if_proj)
    # Must set jit=False for training  ViT-B/32

    model_image = ImageCLIP(model)
    model_image = torch.nn.DataParallel(model_image).cuda()

    transform_val = get_augmentation(False, config)

    if args.dataset == 'breakfast':
        val_data = Breakfast_FRAMES(transforms=transform_val)
    elif args.dataset == 'gtea':
        val_data = GTEA_FRAMES(transform=transform_val)
    elif args.dataset == 'salads':
        val_data = SALADS_FRAMES(transform=transform_val)
    else:
        val_data = None
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                            shuffle=False, pin_memory=False, drop_last=False)

    if device == "cpu":
        model_image.float()
    else:
        clip.model.convert_weights(model_image)

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    model.eval()
    save_dir = config.data.save_dir
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if args.dataset == 'gtea':
        non_splt = False
    else:
        non_splt = True

    with torch.no_grad():
        for iii, (image, filename) in enumerate(tqdm(val_loader)):
            if not os.path.exists(os.path.join(save_dir, filename[0])):
                if non_splt:
                    image = image.view((-1, config.data.num_frames, 3) + image.size()[-2:])
                else:
                    image = image.view((1, -1,  3) + image.size()[-2:])
                b, t, c, h, w = image.size()
                image_input = image.view(b * t, c, h, w)
                if non_splt:
                    image_inputs = image_input.to(device)
                    image_features = model_image(image_inputs)
                    image_features = image_features.view(b, t, -1)
                    for bb in range(b):
                        np.save(os.path.join(save_dir, filename[bb]), image_features[bb, :].cpu().numpy())
                else:
                    image_inputs = torch.split(image_input, 1024)
                    image_features = []
                    for inp in image_inputs:
                        inp = inp.to(device)
                        image_features.append(model.encode_image(inp))
                    image_features = torch.cat(image_features)
                    np.save(os.path.join(save_dir, filename[0]), image_features.cpu().numpy())


if __name__ == '__main__':
    main()
