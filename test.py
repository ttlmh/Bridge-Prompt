import os
import clip
import torch.nn as nn
from datasets import Breakfast, GTEA, SALADS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.fusion_module import fusion_earlyhyp
from utils.Augmentation import get_augmentation
import torch
from utils.text_prompt import *


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def validate(epoch, val_loader, device, model, fusion_model, config,
             text_dict_cnts, text_dict_acts, text_dict_posemb, num_aug, cnt_max):
    model.eval()
    fusion_model.eval()

    final_act_1 = []
    final_act_5 = []
    final_cnt = []
    gt_act = []

    with torch.no_grad():
        text_inputs_cnts = text_dict_cnts.to(device)
        text_dict_posemb = text_dict_posemb.to(device)
        text_dict_acts = text_dict_acts.to(device)
        text_features_cnts = model.encode_text(text_inputs_cnts)
        text_dict_posemb = model.encode_text(text_dict_posemb)
        text_features_acts = model.encode_text(text_dict_acts).view(cnt_max, -1, text_dict_posemb.shape[-1])
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_frames, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            text_features_posemb = text_dict_posemb.repeat(b, 1, 1)
            gt_act.append(class_id.numpy())
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = image_features.unsqueeze(1).repeat(1, cnt_max, 1, 1)
            cnt_emb, image_features = fusion_model(image_features, text_features_posemb)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            cnt_emb /= cnt_emb.norm(dim=-1, keepdim=True)
            text_features_cnts /= text_features_cnts.norm(dim=-1, keepdim=True)
            text_features_acts /= text_features_acts.norm(dim=-1, keepdim=True)
            similarity_cnts = (100.0 * cnt_emb @ text_features_cnts.T)
            similarity_cnts = similarity_cnts.view(b, -1).softmax(dim=-1)
            _, indices_cnts = similarity_cnts.topk(1, dim=-1)
            final_ind_1 = torch.zeros(b, cnt_max).long().to(device)
            final_ind_5 = torch.zeros(b, cnt_max, 5).long().to(device)
            for i in range(text_features_acts.shape[0]):
                similarity_acts = (100.0 * image_features[:, i, :] @ text_features_acts[i, :].T)
                similarity_acts = similarity_acts.view(b, num_aug, -1).softmax(dim=-1)
                similarity_acts = similarity_acts.mean(dim=1, keepdim=False)
                values_1, indices_1 = similarity_acts.topk(1, dim=-1)
                values_5, indices_5 = similarity_acts.topk(5, dim=-1)
                indices_1 = torch.where(indices_1 == 19, -1, indices_1)
                indices_5 = torch.where(indices_5 == 19, -1, indices_5)
                final_ind_1[:, i] = indices_1.squeeze()
                final_ind_5[:, i, :] = indices_5
            final_act_1.append(final_ind_1.cpu().numpy())
            final_act_5.append(final_ind_5.cpu().numpy())
            final_cnt.append(indices_cnts.cpu().numpy())

        np.save(f'./prompt_test/gtea/split{config.data.n_split}/final_act_1.npy',
                np.vstack(final_act_1))
        np.save(f'./prompt_test/gtea/split{config.data.n_split}/final_act_5.npy',
                np.vstack(final_act_5))
        np.save(f'./prompt_test/gtea/split{config.data.n_split}/final_cnt_1.npy',
                np.vstack(final_cnt))
        np.save(f'./prompt_test/gtea/split{config.data.n_split}/gt_act.npy',
                np.vstack(gt_act))


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/gtea/gtea_test.yaml')
    parser.add_argument('--log_time', default='')
    parser.add_argument('--dataset', default='gtea')
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

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    base_model, model_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                             T=config.data.num_segments, dropout=config.network.drop_out,
                                             emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    fusion_model = fusion_earlyhyp(config.network.sim_header, model_state_dict, config.data.num_frames)

    model_text = TextCLIP(base_model)
    model_image = ImageCLIP(base_model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    if args.dataset == 'breakfast':
        val_data = Breakfast(transform=transform_val, mode='val', num_frames=config.data.num_frames,
                             ds=config.data.ds, ol=config.data.ol)
    elif args.dataset == 'gtea':
        val_data = GTEA(transform=transform_val, mode='val', num_frames=config.data.num_frames,
                        n_split=config.data.n_split)
    elif args.dataset == 'salads':
        val_data = SALADS(transform=transform_val, mode='val', num_frames=config.data.num_frames,
                          n_split=config.data.n_split)

    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=False)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(
            model_text)  # Actually this line is unnecessary since clip by default already on float16
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    cnt_max = 6
    text_dict_posemb = text_prompt_ord_emb(cnt_max=cnt_max)

    text_dict_cnts, text_dict_acts, num_aug = text_prompt_slide_val_all(val_data.classes, cnt_max=cnt_max)

    best_prec1 = 0.0
    validate(start_epoch, val_loader, device, base_model, fusion_model, config,
             text_dict_cnts, text_dict_acts, text_dict_posemb, num_aug, cnt_max)


if __name__ == '__main__':
    main()
