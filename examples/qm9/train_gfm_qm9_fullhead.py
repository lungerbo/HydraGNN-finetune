#!/usr/bin/env python3
"""
Full GFM fine-tuning on QM9: train both backbone and GFM-style head.
Includes label normalization, checkpoint loading, sanity checks, and final metrics.
"""

import os, json, time, random, argparse, shutil, re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import hydragnn
from hydragnn.utils.print_utils import setup_log, log
from hydragnn.utils.distributed import setup_ddp, get_distributed_model, get_comm_size_and_rank
from hydragnn.utils import update_config
from hydragnn.models import create_model_config

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fraction", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split_dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--label_stats", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def load_stats(path):
    with open(path) as f:
        d = json.load(f)
    return float(d["mean"]), float(d["std"])

def normalize(ds, mean, std):
    for d in ds:
        d.y = (d.y - mean) / std
    return ds

def find_best_epoch(logfile):
    best_ep, best_val = None, float("inf")
    for line in open(logfile):
        m = re.search(r"epoch (\d+).*?validation MAE: ([\d.]+)", line)
        if m:
            ep, val = int(m.group(1)), float(m.group(2))
            if val < best_val:
                best_val, best_ep = val, ep
    return best_ep

if __name__ == "__main__":
    args = parse_args()
    tag = f"qm9_gfm_full_{args.fraction}_{os.path.basename(args.split_dir)}_seed{args.seed}"
    setup_log(tag)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["HYDRAGNN_DEVICE"] = "cuda"
    world_size, rank = setup_ddp()
    log(f"[DDP] Rank {rank}/{world_size}")

    cfg = json.load(open(args.config))
    mean, std = load_stats(args.label_stats)
    log(f"[STATS] mean={mean:.6f}, std={std:.6f}")

    # Load and normalize datasets
    base = args.split_dir
    train_ds = normalize(torch.load(f"{base}/train_{args.fraction}.pt"), mean, std)
    val_ds   = normalize(torch.load(f"{base}/val.pt"), mean, std)
    test_ds  = normalize(torch.load(f"{base}/test.pt"), mean, std)

    bs = cfg["NeuralNetwork"]["Training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=bs)
    test_loader  = DataLoader(test_ds, batch_size=bs)

    update_config(cfg, train_loader, val_loader, test_loader)
    model = create_model_config(cfg["NeuralNetwork"], cfg["Verbosity"]["level"]).to("cuda")

    # Snapshot initial head₀ before loading
    head0_init = {
        k: v.detach().clone()
        for k, v in model.heads_NN[0].state_dict().items()
    }

    # Load checkpoint weights
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state, strict=False)

    # Snapshot head₀ after loading
    head0_loaded = {
        k: v.detach().clone()
        for k, v in model.heads_NN[0].state_dict().items()
    }
    max_diff = max((head0_loaded[k] - head0_init[k]).abs().max().item() for k in head0_init)
    log(f"[SANITY] max |Δ head₀| = {max_diff:.3e}")

    # All parameters must be trainable
    n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    log(f"[SANITY] Total parameters: {sum(p.numel() for p in model.parameters())}, frozen={n_frozen}")
    assert n_frozen == 0, "Expected all parameters to be trainable!"

    model = get_distributed_model(model, cfg["Verbosity"]["level"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["NeuralNetwork"]["Training"]["Optimizer"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    writer = SummaryWriter(log_dir=f"./logs/{tag}")
    log(f"[TRAIN] Starting full fine-tune → tag={tag}")
    t0 = time.time()

    hydragnn.train.train_validate_test(
        model, optimizer,
        train_loader, val_loader, test_loader,
        writer, scheduler,
        cfg["NeuralNetwork"], tag, cfg["Verbosity"]["level"],
        create_plots=cfg.get("Visualization", {}).get("create_plots", True)
    )
    log(f"[DONE] Training completed in {time.time()-t0:.1f} sec")

    if get_comm_size_and_rank()[1] == 0:
        run_log = f"./logs/{tag}/run.log"
        best_ep = find_best_epoch(run_log)
        if best_ep is not None:
            src = f"./logs/{tag}/{tag}_epoch_{best_ep}.pk"
            dst = f"./logs/{tag}/best_{os.path.basename(args.split_dir)}.pt"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            log(f"[CHECKPOINT] Best model saved → {dst}")
        else:
            log("[WARN] Could not find best epoch")

