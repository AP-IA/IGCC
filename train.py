import os
import time
import logging
import argparse
from datetime import datetime
from contextlib import suppress
from collections import OrderedDict

import numpy as np
import matplotlib
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from omegaconf import OmegaConf
from timm.utils.summary import update_summary
from thop import profile
from torch.utils.tensorboard import SummaryWriter

from network.GlobalBranch import IGCC, create_global_branch, supported_arch
from modules.datasets import BatchDataset, BalancedBatchSampler
from modules import utils, losses

# Set backend and seed
matplotlib.use("agg")
utils.fix_seed()


def parse_args_and_config():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
    parser.add_argument("--arch", type=str, required=True, choices=supported_arch, help="model architecture")
    parser.add_argument("--epochs", type=int, default=None, help="training epochs ")
    parser.add_argument("--img_size", type=int, default=None, help="image size ")
    parser.add_argument("--window_size", type=int, default=None, help="window size ")
    parser.add_argument("--sample_classes", type=int, default=None, help="sample n classes each time ")
    parser.add_argument("--sample_images", type=int, default=None, help="sample n images per class ")
    parser.add_argument("--val_dir", type=str, default=None, help=".pth weights directory")
    parser.add_argument("--nodistill", action="store_true", help="without using teacher model")
    parser.add_argument("--nodynamic", action="store_true", help="without dynamic design in global branch")
    parser.add_argument("--NoGNN", action="store_true")
    parser.add_argument("--NoAFF", action="store_true")
    parser.add_argument("--NoPFE", action="store_true")
    parser.add_argument("--NoMAPS", action="store_true")
    parser.add_argument("--ratio_weight", type=int, default=None, help="dynamic loss weight ")
    parser.add_argument("--gnn_depth", type=int, default=None, help="local branch depth ")

    args = parser.parse_args()
    # Load configuration from YAML
    cfg = OmegaConf.load(args.config)

    # Merge args and config
    args.img_size = args.img_size or cfg.img_size

    args.epochs = args.epochs or cfg.epochs

    args.sample_classes = args.sample_classes or cfg.sample_classes
    args.sample_images = args.sample_images or cfg.sample_images

    args.window_size = args.window_size or cfg.window_size

    args.ratio_weight = args.ratio_weight or cfg.ratio_weight

    args.gnn_depth = args.gnn_depth or cfg.gnn_depth

    cfg.output_dir = cfg.output_dir

    # Validate required parameters
    required_params = [
        ('img_size', args.img_size),
        ('epochs', args.epochs),
        ('sample_classes', args.sample_classes),
        ('sample_images', args.sample_images),
        ('window_size', args.window_size),
        ('ratio_weight', args.ratio_weight),
        ('gnn_depth', args.gnn_depth),
        ('output_dir', cfg.output_dir)
    ]
    for param_name, param_val in required_params:
        if param_val is None:
            raise ValueError(
                f"Parameter '{param_name}' must be specified either in the config file or via command line.")

    args.batch_size = args.sample_classes * args.sample_images

    return args, cfg


def setup_environment(args, cfg):
    # Setup GPU device
    if torch.cuda.is_available() and args.gpus != "cpu":
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device("cpu")

    # Create output directories
    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    exp_name = f"{args.name}-{timestamp}"
    base_dir = os.path.join(cfg.output_dir, exp_name)

    paths = {
        "model_best": os.path.join(base_dir, "models", "best.pth"),
        "model_last": os.path.join(base_dir, "models", "last.pth"),
        "params": os.path.join(base_dir, "models", "params.pth"),
        "log": os.path.join(base_dir, "out_logs", "out.log"),
        "history_png": os.path.join(base_dir, "history", "history.png"),
        "tensorboard": os.path.join(base_dir, "tensorboard"),
        "parts_attn": os.path.join(base_dir, "parts-attn"),
        "summary_csv": os.path.join(base_dir, "summary.csv"),
    }

    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(paths["log"], mode='w'), logging.StreamHandler()]
    )

    return device, timestamp, paths


def cal_loss(model, inputs, targets, criterions, amp_autocast, device, args):
    ce_criterion, rank_criterion, dynamic_criterion = criterions
    acc = 0

    # Forward pass
    if args.NoPFE:
        with amp_autocast:
            logits, global_feature, decision_mask_list = model(inputs)
    else:
        with amp_autocast:
            logits, self_scores, other_scores, global_feature, decision_mask_list = model(inputs, targets)

    # Calculate Cross Entropy Loss
    with amp_autocast:
        loss = 2 * ce_criterion(logits, targets)
    loss_dict = OrderedDict()
    loss_dict.update({'ce_loss': loss.item()})

    # Calculate Ranking Loss (PFE)
    if not args.NoPFE:
        flags = torch.ones([self_scores.size(0), ]).to(device)
        with amp_autocast:
            PFE_loss = rank_criterion(self_scores, other_scores, flags)
        loss += PFE_loss
        loss_dict.update({'PFE_loss': PFE_loss.item()})

    # Calculate Dynamic/Distillation Loss
    if dynamic_criterion:
        with amp_autocast:
            dynamic_loss = dynamic_criterion(inputs, [global_feature, decision_mask_list])
        loss += dynamic_loss
        loss_dict.update({'dynamic_loss': dynamic_loss.item()})

        loss += dynamic_loss
        loss_dict.update({'dynamic_loss': dynamic_loss.item()})

    # Calculate Accuracy
    if acc is not None:
        acc = utils.cal_accuracy(logits, targets)

    if torch.isnan(loss):
        logging.error("Nan is detected in total loss!")
        exit(-1)

    return (loss, loss_dict, acc) if acc is not None else (loss, loss_dict)


# Helper to average loss dictionary
def loss_dict_avg(loss_dict_list):
    avg_loss_dict = {}
    for d in loss_dict_list:
        for k, v in d.items():
            if k not in avg_loss_dict:
                avg_loss_dict[k] = []
            avg_loss_dict[k].append(v)
    for k, v in avg_loss_dict.items():
        avg_loss_dict[k] = np.mean(v)
    return avg_loss_dict


def train_one_epoch(train_loader, model, criterions, optimizers, epoch, amp_autocast, scaler, device, args, cfg):
    model.train()
    batch_loss_list, batch_acc_list, loss_dict_list = [], [], []
    total = len(train_loader)

    for i, (inputs, targets, filenames, categories) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        [optimizer.zero_grad() for optimizer in optimizers]

        # Calculate loss and backward
        loss, loss_dict, acc = cal_loss(model, inputs, targets, criterions, amp_autocast, device, args)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizers[0])
            scaler.step(optimizers[1])
            scaler.update()
        else:
            loss.backward()
            [optimizer.step() for optimizer in optimizers]

        batch_loss_list.append(loss.item())
        batch_acc_list.append(acc)
        loss_dict_list.append(loss_dict)

        # Log progress
        if i % cfg.log_step == 0:
            logging.info(
                f"Training epoch:{epoch + 1}/{args.epochs} batch:{i + 1}/{total} loss:{loss.item():.6f} acc:{acc:.6f} loss_detail: {loss_dict_avg(loss_dict_list)}"
            )

    return np.mean(batch_loss_list), np.mean(batch_acc_list), loss_dict_avg(loss_dict_list)


def validate(val_loader, model, criterion, epoch, device, args):
    model.eval()
    batch_loss_list, batch_acc_list = [], []
    total = len(val_loader)

    with torch.no_grad():
        for i, (inputs, targets, filenames, categories) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if len(outputs.size()) == 1:
                outputs = torch.unsqueeze(outputs, dim=0)

            loss = criterion(outputs, targets)
            acc = utils.cal_accuracy(outputs, targets)

            batch_loss_list.append(loss.item())
            batch_acc_list.append(acc)

            if i % 50 == 0:
                logging.info(
                    f"Validating epoch:{epoch + 1}/{args.epochs} batch:{i + 1}/{total} loss:{loss.item():.6f} acc:{acc:.6f}"
                )

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def main(args, cfg, device, timestamp, paths):
    # Model Configuration
    load_pretrained = cfg.get('load_pretrained', True)
    model_cfg = {

        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "num_classes": cfg.num_classes,
        "image_size": args.img_size,
        "load_pretrained": load_pretrained,
        "drop_rate": cfg.drop_rate,
        "window_size": args.window_size,
        "keep_rate": cfg.keep_rate,
        "pruning_loc": cfg.pruning_loc,
        # 'pretrain_path': getattr(cfg, 'pretrain_path', None),
        'nodynamic': args.nodynamic,
        'arch': args.arch,
        'NoAFF': args.NoAFF,
        'NoPFE': args.NoPFE,
        'NoGNN': args.NoGNN,
        'NoMAPS': args.NoMAPS,
    }
    local_cfg = {
        "depth": args.gnn_depth,
        "batch_size": args.batch_size,
        "top_patches": cfg.top_patches,
        "topk": cfg.topk,
        "cbam_reduction": cfg.get("cbam_reduction"),
        "cbam_kernel": cfg.get("cbam_kernel")
    }

    # Initialize Model
    model = IGCC(model_cfg, local_cfg)

    logging.info("Calculate MACs & FLOPs ...")
    inputs_dummy = torch.randn((1, 3, args.img_size, args.img_size))
    macs, num_params = profile(model, (inputs_dummy,), verbose=False)
    logging.info(f"\nParams(M):{num_params / 1e6:.2f}, MACs(G):{macs / 1e9:.2f}, FLOPs(G):~{2 * macs / 1e9:.2f}\n")

    # Initialize Teacher Model for Distillation
    teacher_model = None
    if not args.nodistill:
        teacher_model = create_global_branch(args.arch, model_cfg, only_teacher_model=True)
        teacher_model.to(device)
        teacher_model.eval()

    logging.info(f"\nargs: \n{args}")
    logging.info(f"\nconfigs: \n{OmegaConf.to_yaml(cfg)}")
    # logging.info(f"\nmodel: \n{model}")

    # Load weights for validation only
    if args.val_dir:
        weights_path = os.path.join(args.val_dir, "best.pth")
        logging.info(f"Load weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        val_dataset = BatchDataset(cfg.data_path, 'val', cfg.txt_dir, transform=transforms.Compose([
            transforms.Resize([args.img_size, args.img_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=20, shuffle=False, num_workers=4, pin_memory=True
        )
        val_loss, val_acc = validate(val_loader, model, losses.LabelSmoothingCrossEntropy().to(device), 0, device, args)
        print(f"val_loss: {val_loss:.6f}, val_acc: {val_acc:.6f}")
        exit(0)

    model.to(device)

    # Data Augmentation
    scale_size = int(round(512 * args.img_size / 448))
    train_transform = transforms.Compose([
        transforms.Resize([scale_size, scale_size]),
        transforms.RandomCrop([args.img_size, args.img_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize([scale_size, scale_size]),
        transforms.CenterCrop([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Dataset and DataLoader
    train_dataset = BatchDataset(cfg.data_path, 'train', cfg.txt_dir, transform=train_transform)
    if args.NoPFE:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True
        )
    else:
        # Use custom sampler for PFE
        train_sampler = BalancedBatchSampler(train_dataset, args.sample_classes, args.sample_images)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_sampler,
            num_workers=cfg.num_workers, pin_memory=True
        )

    val_dataset = BatchDataset(cfg.data_path, 'val', cfg.txt_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Optimizers and Schedulers
    [backbone_params, other_params] = model.get_param_groups()
    optimizers = [
        torch.optim.AdamW(
            backbone_params, lr=cfg.backbone_lr, weight_decay=cfg.weight_decay, betas=cfg.betas
        ),
        torch.optim.AdamW(
            other_params, lr=cfg.others_lr, weight_decay=cfg.weight_decay, betas=cfg.betas
        ),
    ]

    schedulers = [
        utils.WarmupCosineSchedule(
            optimizer, warmup_steps=cfg.warmup_epochs, t_total=int(1.1 * args.epochs)
        ) for optimizer in optimizers
    ]

    # Define Loss Functions
    ce_criterion = losses.LabelSmoothingCrossEntropy().to(device)
    rank_criterion = None if args.NoPFE else nn.MarginRankingLoss(margin=cfg.rank_loss_margin).to(device)

    if args.nodynamic:
        dynamic_criterion = None
    else:
        if args.ratio_weight is None:
            args.ratio_weight = 10
        dynamic_criterion = losses.ConvNextDistillDiffPruningLoss(teacher_model, ratio_weight=args.ratio_weight,
                                                                  distill_weight=0.5, keep_ratio=model.keep_rate,
                                                                  swin_token=True)

    criterions = [ce_criterion, rank_criterion, dynamic_criterion]

    # Mixed Precision Setup
    args.use_amp = cfg.use_amp
    amp_autocast = suppress
    scaler = None
    if args.use_amp:
        amp_autocast = autocast()
        scaler = GradScaler()

    start_epoch, best_val_acc, best_epoch = 0, -float('inf'), 0
    loss_list, acc_list, val_loss_list, val_acc_list = [], [], [], []
    start_time = datetime.now().replace(microsecond=0)

    tb_writer = SummaryWriter(log_dir=paths["tensorboard"])

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        train_loss, train_acc, loss_detail = train_one_epoch(
            train_loader, model, criterions, optimizers, epoch, amp_autocast, scaler, device, args, cfg
        )
        [scheduler.step() for scheduler in schedulers]

        val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch, device, args)

        loss_list.append(train_loss)
        acc_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        # Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), paths["model_best"])
            logging.info(f"Epoch {epoch + 1}: Saved best model with val_acc: {best_val_acc:.6f}")

        epoch_time = time.time() - epoch_start_time
        logging.info(
            f"[Epoch:{epoch + 1}/{args.epochs}] Best_Epoch:{best_epoch} Best_Val_Acc:{best_val_acc:.6f} | "
            f"Time:{epoch_time:.2f}s lr1:{optimizers[0].param_groups[0]['lr']:.7f} lr2:{optimizers[1].param_groups[0]['lr']:.7f} | "
            f"Loss:{train_loss:.6f} Acc:{train_acc:.6f} Val_Loss:{val_loss:.6f} Val_Acc:{val_acc:.6f}"
        )

        # Save Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers],
            'best_val': best_val_acc,
        }, paths["params"])
        torch.save(model.state_dict(), paths["model_last"])
        utils.plot_history(loss_list, acc_list, val_loss_list, val_acc_list, paths["history_png"])

        train_metrics = OrderedDict([('loss', train_loss), ('acc', train_acc)])
        eval_metrics = OrderedDict([('loss', val_loss), ('acc', val_acc)])
        update_summary(
            epoch + 1, train_metrics, eval_metrics, paths["summary_csv"],
            write_header=(epoch == 0)
        )

        # Tensorboard Logging
        tb_writer.add_scalar('Loss/train', train_loss, epoch)
        tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Accuracy/val', val_acc, epoch)
        if loss_detail:
            for k, v in loss_detail.items():
                tb_writer.add_scalar(f'{k}/train', v, epoch)

    tb_writer.close()
    end_time = datetime.now().replace(microsecond=0)
    logging.info('Training finished!')
    logging.info(
        f'Total training time: {(end_time - start_time).days} days {(end_time - start_time).seconds / 3600:.2f} hours'
    )
    logging.info(f'Best val acc: {best_val_acc:.4f} achieved at epoch: {best_epoch}')


def run():
    args, cfg = parse_args_and_config()
    device, timestamp, paths = setup_environment(args, cfg)
    main(args, cfg, device, timestamp, paths)


if __name__ == '__main__':
    run()