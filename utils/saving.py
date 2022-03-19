import torch


def epoch_saving(epoch, model, fusion_model, optimizer, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)  # just change to your preferred folder/filename


def epoch_saving_seg(epoch, model, fusion_model, frame_fusion_model, optimizer, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'frame_fusion_model_state_dict': frame_fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)  # just change to your preferred folder/filename


def best_saving(working_dir, epoch, model, fusion_model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename
