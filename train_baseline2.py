import os
import random
import time
import datetime
import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import print_and_save, shuffling, epoch_time, calculate_metrics
from utils.metrics import DiceBCELoss
from utils.run_engine import load_data, DATASET

from models.baseline_net import BaselineNetwork

def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_baseline(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss, epoch_jac, epoch_f1, epoch_recall, epoch_precision = 0.0, 0.0, 0.0, 0.0, 0.0

    for i, (x, (y1, y2)) in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        x = x.to(device, dtype=torch.float32)
        y1 = y1.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()
        mask_pred = model(x)
        
        loss = loss_fn(mask_pred, y1)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
        for yt, yp in zip(y1, mask_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])
            
        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)
        
    n = len(loader)
    return epoch_loss / n, [epoch_jac / n, epoch_f1 / n, epoch_recall / n, epoch_precision / n]

def evaluate_baseline(model, loader, loss_fn, device):
    model.eval()
    epoch_loss, epoch_jac, epoch_f1, epoch_recall, epoch_precision = 0.0, 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for i, (x, (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            
            mask_pred = model(x)
            loss = loss_fn(mask_pred, y1)
            
            epoch_loss += loss.item()
            
            batch_jac, batch_f1, batch_recall, batch_precision = [], [], [], []
            for yt, yp in zip(y1, mask_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])
                
            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)
            
    n = len(loader)
    return epoch_loss / n, [epoch_jac / n, epoch_f1 / n, epoch_recall / n, epoch_precision / n]

if __name__ == "__main__":
    dataset_name = 'Kvasir-sessile' 
    val_name = None

    seed = 0
    my_seeding(seed)

    image_size = 256
    batch_size = 8
    num_epochs = 300
    lr_decoder = 1e-4  # Decoder base LR
    lr_encoder = 1e-5  # Encoder smaller LR
    early_stopping_patience = 40

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"Baseline2_CR_{dataset_name}_{val_name}_lr{lr_decoder}_{current_time}"

    base_dir = "./data"
    data_path = os.path.join(base_dir, dataset_name)
    save_dir = os.path.join("run_files", dataset_name, folder_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    # 此处设置Stage 1预训练权重的路径
    stage1_pretrained_path = "./run_files/Kvasir-sessile/stage1_Kvasir-sessile_resnet50_None_lr0.0001_20260402-143314/checkpoint.pth"

    with open(train_log_path, "w") as f:
        f.write("\n")

    print_and_save(train_log_path, str(datetime.datetime.now()))
    print_and_save(train_log_path, f"Image Size: {image_size}\nBatch Size: {batch_size}\nLR Decoder: {lr_decoder}, LR Encoder: {lr_encoder}\nEpochs: {num_epochs}\nEarly Stopping Patience: {early_stopping_patience}\nSeed: {seed}\n")

    # 强数据增强
    transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_path, val_name)
    train_x, train_y = shuffling(train_x, train_y)
    print_and_save(train_log_path, f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}\n")

    train_dataset = DATASET(train_x, train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, (image_size, image_size), transform=None)
    test_dataset = DATASET(test_x, test_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 实例化纯净版 BaselineNetwork
    model = BaselineNetwork()
    
    # ---------------------------------------------------------
    # 编写第二阶段权重加载逻辑 (Stage 2)
    # ---------------------------------------------------------
    if os.path.exists(stage1_pretrained_path):
        stage1_state = torch.load(stage1_pretrained_path, map_location='cpu')
        
        # 仅保留 ResNet-50 的权重。在 Net0 (ConDSegStage1) 中，前缀对应为 layer0~layer3
        backbone_prefixes = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4')
        filtered_dict = {}
        ignored_keys = []
        
        for k, v in stage1_state.items():
            if k.startswith(backbone_prefixes):
                filtered_dict[k] = v
            else:
                ignored_keys.append(k)
                
        # 安全性断言：确保真实提取到了骨干网络权重
        if len(filtered_dict) == 0:
            raise ValueError(f"CR Weight Transfer Failed: No keys matching {backbone_prefixes} were found in {stage1_pretrained_path}. Please check the saved model's dictionary keys.")
            
        print(f"Ignored {len(ignored_keys)} keys during CR weight transfer.")
            
        # 严格过滤预测头后，进行 strict=False 的安全加载
        missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
        
        print_and_save(train_log_path, f"Successfully loaded {len(filtered_dict)} CR pre-trained parameter tensors from {stage1_pretrained_path}\n")
        if len(unexpected_keys) > 0:
            print_and_save(train_log_path, f"WARNING: Unexpected keys during loading: {unexpected_keys}\n")
    else:
        print_and_save(train_log_path, f"WARNING: CR Stage 1 weights ({stage1_pretrained_path}) not found! Using standard ImageNet weights.\n")
        
    model = model.to(device)

    # ---------------------------------------------------------
    # 为优化器构建参数组（Parameter Groups）以实现差异化学习率
    # ---------------------------------------------------------
    encoder_params = []
    decoder_params = []
    encoder_prefixes = ('layer0', 'layer1', 'layer2', 'layer3', 'layer4')
    
    for name, param in model.named_parameters():
        if name.startswith(encoder_prefixes):
            encoder_params.append(param)
        else:
            decoder_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': lr_encoder},
        {'params': decoder_params, 'lr': lr_decoder}
    ], weight_decay=1e-4)
    
    # 接入统一的余弦退火策略
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # 损失函数与 Baseline 1 保持一致，仅使用纯净的 L_mask
    loss_fn = DiceBCELoss()
    
    print_and_save(train_log_path, "Optimizer: Adam with Parameter Groups (Encoder: 1e-5, Decoder: 1e-4)\nScheduler: CosineAnnealingLR\nLoss: BCE Dice Loss\n")
    print_and_save(train_log_path, f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000:.2f}M\n")

    with open(os.path.join(save_dir, "train_log.csv"), "w") as f:
        f.write("epoch,train_loss,train_mIoU,train_f1,train_recall,train_precision,valid_loss,valid_mIoU,valid_f1,valid_recall,valid_precision\n")

    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train_baseline(model, train_loader, optimizer, loss_fn, device)
        valid_loss, valid_metrics = evaluate_baseline(model, valid_loader, loss_fn, device)
        scheduler.step()

        if valid_metrics[0] > best_valid_metrics:
            print_and_save(train_log_path, f"Valid mIoU improved from {best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {checkpoint_path}")
            best_valid_metrics = valid_metrics[0]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - mIoU: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - mIoU: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        with open(os.path.join(save_dir, "train_log.csv"), "a") as f:
            f.write(f"{epoch + 1},{train_loss},{train_metrics[0]},{train_metrics[1]},{train_metrics[2]},{train_metrics[3]},{valid_loss},{valid_metrics[0]},{valid_metrics[1]},{valid_metrics[2]},{valid_metrics[3]}\n")

        if early_stopping_count == early_stopping_patience:
            print_and_save(train_log_path, f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n")
            break

    # ================= FINAL TEST RESULTS ================
    print_and_save(train_log_path, "Loading best model for final testing...")
    model.load_state_dict(torch.load(checkpoint_path))
    test_loss, test_metrics = evaluate_baseline(model, test_loader, loss_fn, device)
    
    print_and_save(train_log_path, "================ FINAL TEST RESULTS ================")
    print_and_save(train_log_path, f"Test Loss: {test_loss:.4f} - mIoU: {test_metrics[0]:.4f} - F1: {test_metrics[1]:.4f} - Recall: {test_metrics[2]:.4f} - Precision: {test_metrics[3]:.4f}")
