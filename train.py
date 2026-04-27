import os
import argparse
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader

from utils.utils import print_and_save, shuffling, epoch_time
from network.model import ConDSeg
from utils.metrics import DiceBCELoss

#
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from utils.run_engine import load_data, train, evaluate, DATASET


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_sid_cdfa', action='store_true', help="Bypass SID and CDFA modules")
    parser.add_argument('--disable_sid_loss', action='store_true', help="Disable SID auxiliary losses (beta and complementary loss)")
    parser.add_argument('--pretrained_type', type=str, choices=['imagenet', 'stage1'], default='stage1', help="Pretrained weights for backbone")
    args = parser.parse_args()

    use_sid_cdfa = not args.disable_sid_cdfa
    use_sid_loss = not args.disable_sid_loss

    # dataset
    dataset_name = 'Kvasir-SEG' 
    val_name = None

    seed = 0
    my_seeding(seed)

    # hyperparameters
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 300
    lr = 1e-4
    lr_backbone = 1e-5
    early_stopping_patience = 40

    if args.pretrained_type == 'stage1':
        pretrained_backbone = './run_files/Kvasir-SEG/stage1_Kvasir-SEG_resnet50_None_lr0.0001_20260424-230123/checkpoint.pth'
    else:
        pretrained_backbone = None

    resume_path = None

    # make a folder
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ablation_str = f"Init-{args.pretrained_type}_{'NoSID' if args.disable_sid_cdfa else 'WithSID'}_{'NoSIDLoss' if args.disable_sid_loss else 'WithSIDLoss'}"
    folder_name = f"{dataset_name}_{val_name}_{ablation_str}_lr{lr}_{current_time}"

    # Directories
    base_dir = "./data"
    data_path = os.path.join(base_dir, dataset_name)
    save_dir = os.path.join("run_files", dataset_name, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")

    train_log = open(train_log_path, "w")
    train_log.write("\n")
    train_log.close()

    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    hyperparameters_str = f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    hyperparameters_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    hyperparameters_str += f"Seed: {seed}\n"
    print_and_save(train_log_path, hyperparameters_str)

    """ Data augmentation: Transforms """
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3), # 添加随机亮度和对比度调整
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3), # 添加色调、饱和度和亮度的随机调整
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32) #添加CoarseDropout进行随机擦除数据增强
    ])

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ Model """
    device = torch.device('cuda')
    model = ConDSeg(use_sid_cdfa=use_sid_cdfa)

    if pretrained_backbone:
        saved_state = torch.load(pretrained_backbone, map_location='cpu')
        model_state = model.state_dict()
        backbone_prefixes = ('layer0', 'layer1', 'layer2', 'layer3')
        filtered_state = {
            k: v for k, v in saved_state.items()
            if k in model_state and k.startswith(backbone_prefixes)
        }
        model_state.update(filtered_state)
        model.load_state_dict(model_state, strict=False)

    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint)

    model = model.to(device)

    param_groups = [
        {'params': [], 'lr': lr_backbone},
        {'params': [], 'lr': lr}
    ]

    for name, param in model.named_parameters():
        if name.startswith('layer0') or name.startswith('layer1'):
            param.requires_grad = False
        elif name.startswith('layer2') or name.startswith('layer3'):
            param_groups[0]['params'].append(param)
        else:
            param_groups[1]['params'].append(param)

    assert len(param_groups[0]['params']) > 0, "Layer group is empty!"
    assert len(param_groups[1]['params']) > 0, "Rest group is empty!"

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"
    data_str = f"Optimizer: AdamW\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_str = f"Number of parameters: {num_params / 1000000}M\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """

    with open(os.path.join(save_dir, "train_log.csv"), "w") as f:
        f.write(
            "epoch,train_loss,train_mIoU,train_f1,train_recall,train_precision,valid_loss,valid_mIoU,valid_f1,valid_recall,valid_precision\n")

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device, use_sid_cdfa, use_sid_loss)
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device, use_sid_cdfa, use_sid_loss)
        scheduler.step()

        if valid_metrics[0] > best_valid_metrics:
            data_str = f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0


        elif valid_metrics[0] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(
                f"{epoch + 1},{train_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n")

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
