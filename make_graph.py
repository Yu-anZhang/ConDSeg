import argparse
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to avoid Qt/xcb errors
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(points, factor=0.8):
    """
    使用指数移动平均 (EMA) 对曲线进行平滑处理，类似于 TensorBoard 的效果。
    factor: 平滑系数，越大越平滑 (0 <= factor < 1)
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def parse_logs(filepath):
    metrics = {'Train': defaultdict(list), 'Val': defaultdict(list)}
    if not os.path.exists(filepath):
        print(f"⚠️ 警告: 未找到日志文件 {filepath}")
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if 'Train Loss:' in line or 'Val. Loss:' in line:
                stage = 'Train' if 'Train Loss:' in line else 'Val'
                parts = line.split('-')
                for p in parts:
                    if ':' in p:
                        k, v = p.split(':')
                        k = k.strip().replace('Train ', '').replace('Val. ', '').replace('Val ', '')
                        try:
                            metrics[stage][k].append(float(v.strip()))
                        except ValueError:
                            pass
    return metrics

def main():
    parser = argparse.ArgumentParser(description="ConDSeg 毕业论文级曲线可视化工具")
    parser.add_argument("--log1", type=str, default=None, help="Stage 1 log 路径")
    parser.add_argument("--log2", type=str, default=None, help="Stage 2 log 路径")
    parser.add_argument("--metrics", type=str, nargs='+', default=['Loss', 'mIoU'], help="要绘制的指标")
    parser.add_argument("--smooth", type=float, default=0.8, help="曲线平滑系数 (0.0 不平滑，0.9 强平滑)")
    parser.add_argument("--save", type=str, default="ConDSeg_Curves.png", help="保存路径")
    args = parser.parse_args()

    logs_data = []
    if args.log1:
        data = parse_logs(args.log1)
        if data: logs_data.append(('Stage 1: Backbone Pre-training', data))
    if args.log2:
        data = parse_logs(args.log2)
        if data: logs_data.append(('Stage 2: Full Model Fine-tuning', data))

    if not logs_data:
        print("❌ 错误: 未提供日志文件")
        return

    num_metrics = len(args.metrics)
    num_stages = len(logs_data)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(num_metrics, num_stages, figsize=(6 * num_stages, 4 * num_metrics), dpi=300, squeeze=False)

    for i, metric in enumerate(args.metrics):
        for j, (stage_name, data) in enumerate(logs_data):
            ax = axes[i, j]
            
            train_vals = data['Train'].get(metric, [])
            val_vals = data['Val'].get(metric, [])
            epochs = range(1, len(train_vals)+1)
            
            if train_vals:
                ax.plot(epochs, train_vals, label=f'Train {metric}', color='#1f77b4', linewidth=2.0)
            
            if val_vals:
                ax.plot(range(1, len(val_vals)+1), val_vals, label=f'Val {metric}', color='#d62728', linewidth=2.0)
            
            if i == 0:
                ax.set_title(stage_name, fontsize=14, fontweight='bold', pad=15)
            
            ax.set_xlabel('Epochs', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            
            if train_vals or val_vals:
                ax.legend(frameon=True, fontsize=11, loc='best')
            
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # 优化边框
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color('#333333')

    plt.tight_layout()
    plt.savefig(args.save, bbox_inches='tight')
    print(f"✅ 毕业论文级别高清水印图像已保存至: {args.save}")

if __name__ == '__main__':
    main()