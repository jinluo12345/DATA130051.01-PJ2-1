# train_ablation.py

import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data import get_dataloaders
from model import get_model
from train import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'y', 'true', 't', '1'):
        return True
    if v.lower() in ('no', 'n', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_single(history, key, out_dir):
    ensure_dir(out_dir)
    epochs = list(range(1, len(history[key]['train_loss'])+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, history[key]['train_loss'], label='train')
    plt.plot(epochs, history[key]['val_loss'],   label='val')
    plt.title(f"{key} loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(epochs, history[key]['train_acc'], label='train')
    plt.plot(epochs, history[key]['val_acc'],   label='val')
    plt.title(f"{key} acc"); plt.xlabel("Epoch"); plt.ylabel("Acc")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{key}.png")
    plt.savefig(save_path); plt.close()
    print(f"Saved plot: {save_path}")

def plot_all(history, metric, out_dir):
    """
    Plot all configurations' training and validation metrics in separate subplots.
    Left: validation curves for each config.
    Right: training curves for each config.
    """
    ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    ax_val, ax_train = axes

    # Plot validation curves
    for key, hist in history.items():
        epochs = list(range(1, len(hist[f'val_{metric}'])+1))
        ax_val.plot(epochs, hist[f'val_{metric}'], label=key)
    ax_val.set_title(f"Validation {metric.capitalize()}")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Loss" if metric=='loss' else "Acc")
    ax_val.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax_val.grid(True)

    # Plot training curves
    for key, hist in history.items():
        epochs = list(range(1, len(hist[f'train_{metric}'])+1))
        ax_train.plot(epochs, hist[f'train_{metric}'], '--', label=key)
    ax_train.set_title(f"Training {metric.capitalize()}")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss" if metric=='loss' else "Acc")
    ax_train.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    ax_train.grid(True)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"all_{metric}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int,   nargs='+', default=[128],
                        help='多个 batch sizes 用于 ablation')
    parser.add_argument('--lr',         type=float, nargs='+', default=[1e-3])
    parser.add_argument('--epochs',     type=int,   nargs='+', default=[50])
    parser.add_argument('--arch',       type=str,   nargs='+',
                        default=['microsoft/resnet-18'])
    parser.add_argument('--pretrained', type=str2bool, nargs='+', default=[True],
                        help='是否加载预训练权重')
    parser.add_argument('--augment',    type=str2bool, nargs='+', default=[False],
                        help='是否开启数据增强')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--plot_dir',   type=str, default='plots')
    parser.add_argument('--save_dir',   type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir='runs/experiment')
    history = {}
    ensure_dir(args.save_dir)

    for batch_size in args.batch_size:
        for arch in args.arch:
            for lr in args.lr:
                for num_epochs in args.epochs:
                    for pretrained in args.pretrained:
                        for augment in args.augment:
                            pre_tag = 'pre' if pretrained else 'scratch'
                            aug_tag = 'aug' if augment else 'noaug'
                            key = (f"{arch.split('/')[-1]}"
                                   f"_bs{batch_size}"
                                   f"_lr{lr}"
                                   f"_ep{num_epochs}"
                                   f"_{pre_tag}_{aug_tag}")
                            history[key] = {
                                'train_loss': [], 'val_loss': [],
                                'train_acc':  [], 'val_acc':  []
                            }

                            best_val_acc = 0.0
                            best_path = os.path.join(args.save_dir, f"{key}_best.pth")

                            torch.manual_seed(args.seed)
                            train_loader, val_loader = get_dataloaders(
                                batch_size=batch_size,
                                image_size=224,
                                split_ratio=(0.8,0.2),
                                seed=args.seed,  
                                augment=augment
                            )
                            model = get_model(arch, pretrained, num_classes=102).to(device)

                            head = (model.classifier.parameters()
                                    if hasattr(model, 'classifier')
                                    else model.model.classifier.parameters())
                            backbone = [p for n,p in model.named_parameters()
                                        if 'classifier' not in n]
                            optimizer = optim.SGD([
                                {'params': head,     'lr': lr},
                                {'params': backbone, 'lr': lr*0.1},
                            ], momentum=0.9)
                            criterion = torch.nn.CrossEntropyLoss()

                            for epoch in range(1, num_epochs+1):
                                tl, ta = train_one_epoch(model, train_loader,
                                                         criterion, optimizer, device)
                                vl, va = evaluate(model, val_loader,
                                                  criterion, device)

                                history[key]['train_loss'].append(tl)
                                history[key]['train_acc'].append(ta)
                                history[key]['val_loss'].append(vl)
                                history[key]['val_acc'].append(va)

                                if va > best_val_acc:
                                    best_val_acc = va
                                    torch.save(model.state_dict(), best_path)
                                    print(f"Saved best model for {key} "
                                          f"at epoch {epoch} (VA={va:.4f})")

                                writer.add_scalar(f"Loss/train_{key}", tl, epoch)
                                writer.add_scalar(f"Loss/val_{key}",   vl, epoch)
                                writer.add_scalar(f"Acc/train_{key}",  ta, epoch)
                                writer.add_scalar(f"Acc/val_{key}",    va, epoch)

                                print(f"[{key}] Ep{epoch}/{num_epochs} "
                                      f"BS={batch_size} TL={tl:.4f} TA={ta:.4f} "
                                      f"VL={vl:.4f} VA={va:.4f}")

    for key in history:
        plot_single(history, key, args.plot_dir)
    plot_all(history, metric='loss', out_dir=args.plot_dir)
    plot_all(history, metric='acc',  out_dir=args.plot_dir)

if __name__ == '__main__':
    main()
