from tqdm import tqdm
from utils.text_prompt import *


def validate(epoch, val_loader, classes, device, model, fusion_model, fusion_model_up, config, num_text_aug, proj):
    model.eval()
    fusion_model.eval()
    fusion_model_up.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():
        text_inputs = classes.to(device)
        text_features = model.encode_text(text_inputs)
        for iii, (image, class_id) in enumerate(tqdm(val_loader)):
            # image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, n, f, d = image.size()
            class_id = class_id.to(device)
            # image_input = image.to(device).view(-1, c, h, w)
            # image_features = model.encode_image(image_input).view(b, t, -1)
            image = image.to(device)
            image_features = image.half() @ proj
            # image_features = image_features @ proj
            image_features = image_features.view(-1, f, 512)
            image_features = fusion_model(image_features)
            image_features = image_features.view(b, n, 512)
            image_features = fusion_model_up(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
            similarity = similarity.mean(dim=1, keepdim=False)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == class_id[i]:
                    corr_1 += 1
                if class_id[i] in indices_5[i]:
                    corr_5 += 1
    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    print('Epoch: [{}/{}]: Top1: {}, Top5: {}'.format(epoch, config.solver.epochs, top1, top5))
    return top1
