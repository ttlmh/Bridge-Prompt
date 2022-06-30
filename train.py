import os
import torch
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
from modules.fusion_module import fusion_earlyhyp
from utils.KLLoss import KLLoss
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.text_prompt import *
from utils.saving import *


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


def get_clip_loss(image_embedding, text_embedding, logit_scale, loss_img, loss_txt, device, labels):
    logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale)
    ground_truth = torch.tensor(labels, dtype=image_embedding.dtype, device=device)
    loss_imgs = loss_img(logits_per_image, ground_truth)
    loss_texts = loss_txt(logits_per_text, ground_truth)
    total_loss = (loss_imgs + loss_texts) / 2
    return total_loss


def main():
    wandb_on = False
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='./configs/breakfast/breakfast_ft.yaml')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args.dataset = config['data']['dataset']
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    if wandb_on:
        wandb.init(project=config['network']['type'],
                   name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                             config['data']['dataset']))
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
    shutil.copy('train.py', working_dir)
    shutil.copy('modules/fusion_module.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    base_model, model_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                             T=config.data.num_segments, dropout=config.network.drop_out,
                                             emb_dropout=config.network.emb_dropout, pretrain=config.network.init,
                                             joint=config.network.joint)  # Must set jit=False for training  ViT-B/32

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    fusion_model = fusion_earlyhyp(config.network.sim_header, model_state_dict, config.data.num_frames)
    model_text = TextCLIP(base_model)
    model_image = ImageCLIP(base_model)
    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    if wandb_on:
        wandb.watch(base_model)
        wandb.watch(fusion_model)

    if args.dataset == 'breakfast':
        train_data = Breakfast(transform=transform_train, mode='train', num_frames=config.data.num_frames,
                               ds=config.data.ds, ol=config.data.ol, n_split=config.data.n_split)

    elif args.dataset == 'gtea':
        train_data = GTEA(transform=transform_train, mode='train', num_frames=config.data.num_frames,
                          n_split=config.data.n_split)

    elif args.dataset == 'salads':
        train_data = SALADS(transform=transform_train, mode='train', num_frames=config.data.num_frames,
                            n_split=config.data.n_split)

    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=True, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_text)
        clip.model.convert_weights(model_image)

    loss_img = KLLoss()
    loss_txt = KLLoss()

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    text_dict_posemb = text_prompt_ord_emb(config.data.max_act)

    optimizer = _optimizer(config, base_model, fusion_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0

    for k, v in base_model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))
    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        text_dict_posemb = text_dict_posemb.to(device, non_blocking=True)
        text_dict_pos = text_dict_posemb.repeat(config.data.batch_size, 1, 1)
        text_dict_pos = text_dict_pos.view(-1, text_dict_pos.shape[-1])
        for kkk, (images, list_id) in enumerate(tqdm(train_loader)):
            if config.solver.type != 'monitor':
                if (kkk + 1) == 1 or (kkk + 1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))
            optimizer.zero_grad()

            images = images.view((-1, config.data.num_frames, 3) + images.size()[-2:])
            b, t, c, h, w = images.size()

            text_cnt, text_acts, text_all, label_cnt = text_prompt_slide(train_data.classes,
                                                                         list_id, args.dataset,
                                                                         config.data.max_act)

            images = images.to(device, non_blocking=True).view(-1, c, h, w)  # omit the Image.fromarray if the images
            text_cnt = text_cnt.to(device, non_blocking=True)
            text_acts = text_acts.to(device, non_blocking=True)

            text_all = text_all.to(device, non_blocking=True)

            image_embedding = model_image(images)
            image_embedding = image_embedding.view(b, t, -1)

            text_acts = text_acts.view(-1, text_acts.shape[-1])

            text_all_embedding = model_text(text_all)
            text_cnt_embedding = model_text(text_cnt)
            text_acts_embedding = model_text(text_acts)
            text_pos_embedding = model_text(text_dict_pos)

            text_acts_embedding = text_acts_embedding.view(b, -1, text_acts_embedding.shape[-1])
            text_pos_embedding = text_pos_embedding.view(b, -1, text_pos_embedding.shape[-1])

            image_embedding = image_embedding.unsqueeze(1).repeat(1, config.data.max_act, 1, 1)
            cnt_emb, image_embedding = fusion_model(image_embedding, text_pos_embedding)

            if config.network.fix_text:
                text_cnt_embedding.detach_()
                text_acts_embedding.detach_()
                text_all_embedding.detach_()
                text_pos_embedding.detach_()

            logit_scale = base_model.logit_scale.exp()

            act_loss = 0
            image_embedding_mean = image_embedding.mean(dim=1, keepdim=False)
            all_loss = get_clip_loss(image_embedding_mean, text_all_embedding, logit_scale, loss_img, loss_txt, device,
                                     gen_label_4list(list_id))
            cnt_loss = get_clip_loss(cnt_emb, text_cnt_embedding, logit_scale, loss_img, loss_txt, device,
                                     gen_label(label_cnt))
            for dd in range(text_acts_embedding.shape[1]):
                act_loss += get_clip_loss(image_embedding[:, dd, :], text_acts_embedding[:, dd, :], logit_scale,
                                          loss_img,
                                          loss_txt, device, gen_label(list_id[:, dd]))

            total_loss = all_loss + act_loss + cnt_loss
            if wandb_on:
                wandb.log({"train_total_loss": total_loss})
                wandb.log({"train_loss_all": all_loss})
                wandb.log({"train_loss_acts": act_loss})
                wandb.log({"train_loss_cnt": cnt_loss})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(base_model)
                optimizer.step()
                clip.model.convert_weights(base_model)

        epoch_saving(epoch, base_model, fusion_model, optimizer, "{}/".format(working_dir) + str(epoch) + "_epoch.pt")
        epoch_saving(epoch, base_model, fusion_model, optimizer, "{}/last_model.pt".format(working_dir))


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    main()
